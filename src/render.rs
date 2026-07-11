//! GBM + EGL + GLES2 render pipeline with vblank-driven page-flipping
//! across multiple outputs.
//!
//! A single EGL context + GLES renderer + GBM allocator is shared by
//! every output on a given GPU. Each output has its own
//! `ScanoutSurface` (its own swapchain + page-flip cadence) and
//! is rendered independently when *its* CRTC reports vblank. Outputs
//! sit in a virtual layout — by default left-to-right at `y=0` in
//! connector enumeration order; Lua config will override per-output
//! positions in milestone 3c.
//!
//! Cursor coordinates live in absolute virtual-layout space. On each
//! per-output render we translate to output-local coordinates and
//! draw the cursor only when the hotspot falls within that output's
//! rectangle.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::format::has_alpha;
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBuffer, GbmBufferFlags, GbmDevice};
use smithay::backend::allocator::{Buffer as _, Format, Fourcc};
use smithay::backend::drm::{DrmDeviceFd, DrmNode, VrrSupport};
use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay};
use smithay::backend::renderer::element::Kind;
use smithay::backend::renderer::element::surface::{
    WaylandSurfaceRenderElement, render_elements_from_surface_tree,
};
use smithay::backend::renderer::element::utils::RescaleRenderElement;
use smithay::backend::renderer::gles::{
    GlesFrame, GlesRenderer, GlesTarget, GlesTexProgram, GlesTexture, Uniform, UniformName,
    UniformType,
};
use smithay::backend::renderer::utils::{
    Buffer as ClientBuffer, draw_render_elements, with_renderer_surface_state,
};
use smithay::backend::renderer::{
    Bind as _, Blit as _, Color32F, ExportMem as _, Frame as _, ImportDma as _, ImportMem as _,
    Offscreen as _, Renderer as _, Texture as _, TextureFilter,
};
use smithay::desktop::utils::{
    OutputPresentationFeedback, take_presentation_feedback_surface_tree,
};
use smithay::input::pointer::{CursorIcon, CursorImageStatus, CursorImageSurfaceData};
use smithay::output::Output;
use smithay::reexports::wayland_protocols::wp::presentation_time::server::wp_presentation_feedback::Kind as PresentKind;
use smithay::wayland::presentation::Refresh;
use smithay::reexports::drm::Device as _;
use smithay::reexports::drm::DriverCapability;
use smithay::reexports::drm::control::dumbbuffer::DumbBuffer;
use smithay::reexports::drm::control::{Device as ControlDevice, connector, crtc};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::backend::ObjectId;
use smithay::utils::{
    IsAlive as _, Logical, Monotonic, Physical, Point, Rectangle, Scale, Size, Time, Transform,
};
use smithay::wayland::compositor::{
    BufferAssignment, SurfaceAttributes, TraversalAction, with_states, with_surface_tree_downward,
};
use smithay::wayland::shell::xdg::SurfaceCachedState;
use tracing::{debug, info, warn};

use crate::anim::{Animation, lerp};
use crate::config::{
    AnimationsConfig, BlurConfig, BorderConfig, DecorationConfig, Fill, MonitorsConfig, ScaleMode,
    VrrMode,
};
use crate::drm::DrmOutput;
use crate::layout::{FillMode, Placement};
use crate::scanout::ScanoutSurface;

/// A layer surface to render this frame. Pre-computed by main
/// before calling `render_for_crtc` so the renderer doesn't need
/// to know about `wlr_layer_shell` types or per-output
/// associations — just "draw this surface at this rect, in this
/// layer bucket". `rect` is also the bounding box for pointer
/// hit-testing on the main-side.
#[derive(Debug, Clone)]
pub struct LayerPlacement {
    pub surface: WlSurface,
    /// Surface rect in absolute compositor coords.
    pub rect: Rectangle<i32, Physical>,
    /// Logical "depth" used to interleave with windows in
    /// `render_output`. Renderer treats `Background`/`Bottom` as
    /// below windows and `Top`/`Overlay` as above.
    pub layer: LayerBucket,
    /// wlr-layer-shell namespace the client set at creation (e.g. "rofi",
    /// "quickshell"). Drives per-layer blur rules.
    pub namespace: String,
}

/// Whether a layer surface with `namespace` should get backdrop blur, per the
/// configured `blur.layers` rules (substring match; empty rules ignored).
fn layer_should_blur(blur: &BlurConfig, namespace: &str) -> bool {
    blur.layers
        .iter()
        .any(|rule| !rule.is_empty() && namespace.contains(rule.as_str()))
}

/// An `xdg_popup` (menu / submenu) to render this frame. Built by main
/// from the live popup trees. Drawn with no border, **above every
/// window and layer** (just below the cursor), so menus never hide
/// behind tiled windows. `buffer_origin` is where the popup buffer's
/// `(0, 0)` lands in absolute compositor coords (the popup's own
/// window-geometry offset already subtracted); `rect` is the visible
/// popup rect (for frame callbacks / hit-testing on the main side).
#[derive(Debug, Clone)]
pub struct PopupPlacement {
    pub surface: WlSurface,
    pub buffer_origin: Point<i32, Physical>,
    pub rect: Rectangle<i32, Physical>,
}

/// Where a `zwlr_screencopy` capture writes its pixels.
#[derive(Debug)]
pub enum CaptureTarget {
    /// CPU read-back; the bytes come back in [`CaptureOutcome::Shm`] for
    /// the caller to copy into the client's `wl_shm` buffer.
    Shm,
    /// Zero-copy GPU path: blit the composited framebuffer straight into
    /// this client-provided dmabuf. Nothing comes back — it's filled.
    Dmabuf(Dmabuf),
}

/// One pending `zwlr_screencopy` capture for the output being
/// rendered, in physical/buffer pixels.
#[derive(Debug)]
pub struct CaptureSpec {
    pub region: Rectangle<i32, Physical>,
    pub fourcc: Fourcc,
    pub target: CaptureTarget,
}

/// Result of servicing one [`CaptureSpec`]. Both paths deliver the
/// client an upright (top-down) buffer so we never set the screencopy
/// `y_invert` flag: xdg-desktop-portal-wlr 0.8.2 never implemented
/// `y_invert` handling and self-destructs on the flag (it hits an
/// unimplemented stub that frees the cast instance, then double-frees
/// during teardown → SIGSEGV).
#[derive(Debug)]
pub enum CaptureOutcome {
    /// CPU read-back: a tight buffer (`width * 4` bytes/row), rows
    /// top-down (FBO readbacks are memory-ordered; see `capture_shm`).
    Shm {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// The client's dmabuf was filled directly by a GPU blit between
    /// FBO attachments, which is memory-ordered — memory-row 0 stays
    /// the top of the image, so it's already upright (no `y_invert`).
    Dmabuf,
    /// Capture failed; the caller fails the frame.
    Failed,
}

/// Renderer-side mirror of `smithay::wayland::shell::wlr_layer::Layer`.
/// Defined here so render.rs doesn't depend on smithay's shell
/// module types beyond what's needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerBucket {
    Background,
    Bottom,
    Top,
    Overlay,
}

/// Per-output metadata the Wayland frontend needs to advertise
/// `wl_output` and seed `wp_fractional_scale_manager_v1`. Mirrors
/// the renderer's internal `OutputRender` but exposes only the
/// fields the frontend cares about (no GBM surface handle).
#[derive(Debug, Clone)]
pub struct OutputDescriptor {
    pub name: String,
    pub mode_size: Size<i32, Physical>,
    /// DRM mode refresh rate in milli-Hz. Advertised via
    /// `wl_output.refresh` so clients can choose modes that match
    /// the output they'll fullscreen on.
    pub refresh_mhz: i32,
    pub compositor_position: Point<i32, Physical>,
    /// Logical (= compositor) area covered by this output. Held
    /// so the future `xdg_output` / layer-shell handlers can compute
    /// exclusive zones in the same coordinate space the layout
    /// uses, without recomputing `mode_size / scale`.
    #[allow(
        dead_code,
        reason = "consumer is the upcoming xdg_output / layer-shell hookup; field is held now so the descriptor's surface doesn't need to change later"
    )]
    pub compositor_size: Size<i32, Physical>,
    pub scale: f64,
}

/// Side length of the cursor sprite in physical pixels. The sprite
/// is a right-triangle with apex at the hotspot, so this is also
/// the bounding-box width and height.
const CURSOR_SIZE: i32 = 24;

/// Kawase *dual-filter* blur — downsample half. One bilinear tap at the
/// centre (weighted ×4) plus four diagonal taps, averaged. Run once per
/// shrink step of the backdrop pyramid. `halfpixel` is half a texel of
/// the *destination* level (in [0,1] source UV, which spans the same
/// region 1:1); `offset` scales the tap spread (the configured radius).
/// Custom texture shaders inherit `tex`/`alpha`/`v_coords` from smithay
/// and must keep the `//_DEFINES_` placeholder line.
const BLUR_DOWN: &str = r"#version 100
//_DEFINES_
precision mediump float;
uniform sampler2D tex;
uniform float alpha;
uniform vec2 halfpixel;
uniform float offset;
varying vec2 v_coords;

void main() {
    vec2 uv = v_coords;
    vec4 sum = texture2D(tex, uv) * 4.0;
    sum += texture2D(tex, uv - halfpixel * offset);
    sum += texture2D(tex, uv + halfpixel * offset);
    sum += texture2D(tex, uv + vec2(halfpixel.x, -halfpixel.y) * offset);
    sum += texture2D(tex, uv - vec2(halfpixel.x, -halfpixel.y) * offset);
    gl_FragColor = (sum / 8.0) * alpha;
}
";

/// Kawase *dual-filter* blur — upsample half. Eight taps (the four
/// edge-midpoints weighted ×2) averaged as the pyramid grows back to
/// full resolution. Same uniform contract as [`BLUR_DOWN`].
const BLUR_UP: &str = r"#version 100
//_DEFINES_
precision mediump float;
uniform sampler2D tex;
uniform float alpha;
uniform vec2 halfpixel;
uniform float offset;
varying vec2 v_coords;

void main() {
    vec2 uv = v_coords;
    vec4 sum = texture2D(tex, uv + vec2(-halfpixel.x * 2.0, 0.0) * offset);
    sum += texture2D(tex, uv + vec2(-halfpixel.x, halfpixel.y) * offset) * 2.0;
    sum += texture2D(tex, uv + vec2(0.0, halfpixel.y * 2.0) * offset);
    sum += texture2D(tex, uv + vec2(halfpixel.x, halfpixel.y) * offset) * 2.0;
    sum += texture2D(tex, uv + vec2(halfpixel.x * 2.0, 0.0) * offset);
    sum += texture2D(tex, uv + vec2(halfpixel.x, -halfpixel.y) * offset) * 2.0;
    sum += texture2D(tex, uv + vec2(0.0, -halfpixel.y * 2.0) * offset);
    sum += texture2D(tex, uv + vec2(-halfpixel.x, -halfpixel.y) * offset) * 2.0;
    gl_FragColor = (sum / 12.0) * alpha;
}
";

/// Composite a window's surface (pre-rendered into a cell-sized offscreen
/// texture) through a rounded-rectangle mask: sample the surface in the
/// interior, paint an opaque border-gradient ring just inside the edge, and
/// `discard` outside the rounded boundary so the corners are *genuinely
/// transparent* — the already-drawn backdrop (media wallpaper, a tile under
/// a float, a blurred tier) shows through instead of a faked fill colour.
///
/// Same rounded-rect SDF as the retired pixel-shader frame mask, but as a
/// *texture* shader so it can sample the surface. The surface fills the
/// whole cell (the border overlays its outer edge), so it stays opaque
/// across the border boundary and there's no transparent seam at the
/// border's inner edge. Premultiplied output (blend is
/// `GL_ONE / GL_ONE_MINUS_SRC_ALPHA`); `size` is the cell pixel size, passed
/// in because texture shaders get no built-in `size` uniform.
const ROUND_TEX_SHADER: &str = r"#version 100
//_DEFINES_
#extension GL_OES_standard_derivatives : enable

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

uniform sampler2D tex;
uniform float alpha;
uniform vec2 size;
uniform float radius;
uniform float border_width;
uniform vec3 border_top;
uniform vec3 border_bottom;
uniform float output_height;
uniform float cell_origin_y;

varying vec2 v_coords;

void main() {
    vec2 p = v_coords * size;
    vec2 half_size = size * 0.5;
    vec2 d = abs(p - half_size) - (half_size - vec2(radius));
    float dist = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - radius;
    float aa = max(fwidth(dist) * 0.5, 0.5);

    // Outside the rounded shape: discard so the backdrop shows through.
    if (dist > aa) {
        discard;
    }

    // Interior: the window's surface, already premultiplied in the offscreen.
    vec4 surf = texture2D(tex, v_coords);

    // Border ring colour, a vertical gradient keyed off the fragment's
    // *global* y so the ramp stays continuous between adjacent tiles.
    float global_y = cell_origin_y + p.y;
    float t = clamp(global_y / max(output_height, 1.0), 0.0, 1.0);
    vec3 border_rgb = mix(border_top, border_bottom, t);

    // ring: 0 in the interior, 1 in the border ring (AA across dist=-border).
    float ring = smoothstep(-border_width - aa, -border_width + aa, dist);
    // outer: 1 inside the rounded edge, 0 outside (AA across dist=0).
    float outer = 1.0 - smoothstep(-aa, aa, dist);

    // Both premultiplied, faded by the outer-edge coverage; mix interior
    // surface with the opaque border ring.
    vec4 inner_px = surf * outer;
    vec4 border_px = vec4(border_rgb * outer, outer);
    vec4 color = mix(inner_px, border_px, ring);

    gl_FragColor = color * alpha;
}
";

/// Clips a blurred backdrop tier to the same rounded-rect shape its window
/// uses, so the corners reveal the *sharp* backdrop instead of a square block
/// of blur poking out past the rounded edge. Same SDF as [`ROUND_TEX_SHADER`]
/// but with no border ring — it only masks the blur. `v_coords` here samples
/// the *tier* (the `src` sub-rect normalised over the whole tier texture), so
/// it is NOT 0..1 across the drawn rect; `local_mul`/`local_add` are a
/// CPU-computed affine map from `v_coords` back to rect-local pixels for the
/// SDF (feeding `v_coords * size` in directly only lined up for a rect at the
/// output's top-left corner — corners anywhere else were never clipped).
/// Sampled texture (the tier) is already premultiplied; output stays premult
/// for the `GL_ONE / GL_ONE_MINUS_SRC_ALPHA` blend. With `radius = 0` the SDF
/// is a plain rectangle, so nothing is clipped (the pre-rounding behaviour).
const ROUND_BLUR_SHADER: &str = r"#version 100
//_DEFINES_
#extension GL_OES_standard_derivatives : enable

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

uniform sampler2D tex;
uniform float alpha;
uniform vec2 size;
uniform float radius;
uniform vec2 local_mul;
uniform vec2 local_add;

varying vec2 v_coords;

void main() {
    vec2 p = v_coords * local_mul + local_add;
    vec2 half_size = size * 0.5;
    vec2 d = abs(p - half_size) - (half_size - vec2(radius));
    float dist = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - radius;
    float aa = max(fwidth(dist) * 0.5, 0.5);

    // Outside the rounded shape: discard so the sharp backdrop shows through.
    if (dist > aa) {
        discard;
    }

    vec4 c = texture2D(tex, v_coords);
    float outer = 1.0 - smoothstep(-aa, aa, dist);
    gl_FragColor = c * (outer * alpha);
}
";

/// HDR variant of [`ROUND_TEX_SHADER`]: identical rounded-rect + border
/// composite, but the `win_tex` offscreen and border are sRGB, so the
/// final composited colour is decoded to linear BT.2020 (scaled to
/// `reference_white`) before output, for the fp16 linear scene.
const ROUND_TEX_SHADER_HDR: &str = r"#version 100
//_DEFINES_
#extension GL_OES_standard_derivatives : enable
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D tex;
uniform float alpha;
uniform vec2 size;
uniform float radius;
uniform float border_width;
uniform vec3 border_top;
uniform vec3 border_bottom;
uniform float output_height;
uniform float cell_origin_y;
uniform float reference_white;
uniform float saturation;
varying vec2 v_coords;
vec3 srgb_to_linear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}
void main() {
    vec2 p = v_coords * size;
    vec2 half_size = size * 0.5;
    vec2 d = abs(p - half_size) - (half_size - vec2(radius));
    float dist = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - radius;
    float aa = max(fwidth(dist) * 0.5, 0.5);
    if (dist > aa) { discard; }

    vec4 surf = texture2D(tex, v_coords);
    vec3 surf_straight = surf.a > 0.0 ? (surf.rgb / surf.a) : vec3(0.0);

    float global_y = cell_origin_y + p.y;
    float t = clamp(global_y / max(output_height, 1.0), 0.0, 1.0);
    vec3 border_straight = mix(border_top, border_bottom, t);

    float ring = smoothstep(-border_width - aa, -border_width + aa, dist);
    float outer = 1.0 - smoothstep(-aa, aa, dist);

    // Composite in sRGB straight space (matches the SDR shader's blend),
    // then decode the composited colour once to linear BT.2020.
    vec3 composited = mix(surf_straight, border_straight, ring);
    vec3 lin = srgb_to_linear(composited) * (reference_white / 10000.0);
    mat3 bt709_to_bt2020 = mat3(
        0.627403896, 0.069097289, 0.016391439,
        0.329283038, 0.919540395, 0.088013308,
        0.043313066, 0.011362316, 0.895595253
    );
    vec3 bt2020 = bt709_to_bt2020 * lin;
    float luma = dot(bt2020, vec3(0.2627, 0.6780, 0.0593));
    bt2020 = max(mix(vec3(luma), bt2020, saturation), vec3(0.0));
    gl_FragColor = vec4(bt2020 * outer, outer) * alpha;
}
";

/// Rounded-corner / border composite for an **HDR window** whose surface is
/// already decoded to linear BT.2020 in its fp16 `win_tex` (PQ clients can't
/// round-trip through the 8-bit sRGB offscreen the SDR variant assumes). Same
/// geometry as [`ROUND_TEX_SHADER_HDR`] but it composites the (already-linear)
/// surface with a linear border ring — no sRGB decode, no matrix. The border
/// colours are converted to linear BT.2020 on the CPU and passed in.
const ROUND_TEX_SHADER_LINEAR: &str = r"#version 100
//_DEFINES_
#extension GL_OES_standard_derivatives : enable
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D tex;
uniform float alpha;
uniform vec2 size;
uniform float radius;
uniform float border_width;
uniform vec3 border_top;
uniform vec3 border_bottom;
uniform float output_height;
uniform float cell_origin_y;
varying vec2 v_coords;
void main() {
    vec2 p = v_coords * size;
    vec2 half_size = size * 0.5;
    vec2 d = abs(p - half_size) - (half_size - vec2(radius));
    float dist = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - radius;
    float aa = max(fwidth(dist) * 0.5, 0.5);
    if (dist > aa) { discard; }

    vec4 surf = texture2D(tex, v_coords);
    vec3 surf_straight = surf.a > 0.0 ? (surf.rgb / surf.a) : vec3(0.0);

    float global_y = cell_origin_y + p.y;
    float t = clamp(global_y / max(output_height, 1.0), 0.0, 1.0);
    vec3 border_straight = mix(border_top, border_bottom, t);

    float ring = smoothstep(-border_width - aa, -border_width + aa, dist);
    float outer = 1.0 - smoothstep(-aa, aa, dist);

    // Surface and border are already linear BT.2020 — composite directly.
    vec3 composited = mix(surf_straight, border_straight, ring);
    gl_FragColor = vec4(composited * outer, outer) * alpha;
}
";

/// HDR variant of [`ROUND_BLUR_SHADER`]: clips the (sRGB) blurred tier to
/// the rounded shape, then decodes to linear BT.2020 for the fp16 scene.
const ROUND_BLUR_SHADER_HDR: &str = r"#version 100
//_DEFINES_
#extension GL_OES_standard_derivatives : enable
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D tex;
uniform float alpha;
uniform vec2 size;
uniform float radius;
uniform vec2 local_mul;
uniform vec2 local_add;
uniform float reference_white;
uniform float saturation;
varying vec2 v_coords;
vec3 srgb_to_linear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}
void main() {
    vec2 p = v_coords * local_mul + local_add;
    vec2 half_size = size * 0.5;
    vec2 d = abs(p - half_size) - (half_size - vec2(radius));
    float dist = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - radius;
    float aa = max(fwidth(dist) * 0.5, 0.5);
    if (dist > aa) { discard; }

    vec4 c = texture2D(tex, v_coords);
    vec3 straight = c.a > 0.0 ? (c.rgb / c.a) : vec3(0.0);
    vec3 lin = srgb_to_linear(straight) * (reference_white / 10000.0);
    mat3 bt709_to_bt2020 = mat3(
        0.627403896, 0.069097289, 0.016391439,
        0.329283038, 0.919540395, 0.088013308,
        0.043313066, 0.011362316, 0.895595253
    );
    vec3 bt2020 = bt709_to_bt2020 * lin;
    float luma = dot(bt2020, vec3(0.2627, 0.6780, 0.0593));
    bt2020 = max(mix(vec3(luma), bt2020, saturation), vec3(0.0));
    float outer = 1.0 - smoothstep(-aa, aa, dist);
    gl_FragColor = vec4(bt2020 * c.a, c.a) * (outer * alpha);
}
";

/// Masks a blurred backdrop tier by the *surface's own alpha channel* (its
/// texture, bound on unit 1), so the frost follows exactly the shape the
/// client drew — any corner radius, pills, cut-outs — with no compositor-side
/// radius guess. Used for layer-shell panels (a panel's rounding lives in the
/// client buffer, which the compositor can't predict); windows keep the SDF
/// clip ([`ROUND_BLUR_SHADER`]) since the compositor rounds those itself.
/// `mask_mul`/`mask_add` affinely map `v_coords` (tier UV, see
/// [`ROUND_BLUR_SHADER`]) into the mask's 0..1 UV space, including a y-flip
/// when either texture is y-inverted. The mask alpha is treated as
/// *coverage* (saturated, see the shader body), so a translucent panel body
/// still gets the full frost and only the shape's AA edge blends out. Both
/// textures are premultiplied; output stays premult for the
/// `GL_ONE / GL_ONE_MINUS_SRC_ALPHA` blend.
const MASK_BLUR_SHADER: &str = r"#version 100
//_DEFINES_
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

uniform sampler2D tex;
uniform sampler2D mask;
uniform float alpha;
uniform vec2 mask_mul;
uniform vec2 mask_add;

varying vec2 v_coords;

void main() {
    vec4 c = texture2D(tex, v_coords);
    float m = texture2D(mask, v_coords * mask_mul + mask_add).a;
    // The mask alpha is *coverage*, not frost strength: a translucent panel
    // body (say 0.75) must still get the full blur behind it — otherwise the
    // remainder shows the sharp backdrop and the frost reads weak. Saturate
    // so any pixel the panel meaningfully covers is fully frosted and only
    // the AA ramp at the shape's edge blends out.
    m = min(m * 4.0, 1.0);
    gl_FragColor = c * (m * alpha);
}
";

/// HDR variant of [`MASK_BLUR_SHADER`]: alpha-masks the (sRGB) blurred tier,
/// then decodes to linear BT.2020 for the fp16 scene — same colour math as
/// [`ROUND_BLUR_SHADER_HDR`], with the SDF coverage replaced by the mask's
/// alpha.
const MASK_BLUR_SHADER_HDR: &str = r"#version 100
//_DEFINES_
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D tex;
uniform sampler2D mask;
uniform float alpha;
uniform vec2 mask_mul;
uniform vec2 mask_add;
uniform float reference_white;
uniform float saturation;
varying vec2 v_coords;
vec3 srgb_to_linear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}
void main() {
    vec4 c = texture2D(tex, v_coords);
    float m = texture2D(mask, v_coords * mask_mul + mask_add).a;
    // Coverage saturation — see MASK_BLUR_SHADER.
    m = min(m * 4.0, 1.0);
    vec3 straight = c.a > 0.0 ? (c.rgb / c.a) : vec3(0.0);
    vec3 lin = srgb_to_linear(straight) * (reference_white / 10000.0);
    mat3 bt709_to_bt2020 = mat3(
        0.627403896, 0.069097289, 0.016391439,
        0.329283038, 0.919540395, 0.088013308,
        0.043313066, 0.011362316, 0.895595253
    );
    vec3 bt2020 = bt709_to_bt2020 * lin;
    float luma = dot(bt2020, vec3(0.2627, 0.6780, 0.0593));
    bt2020 = max(mix(vec3(luma), bt2020, saturation), vec3(0.0));
    gl_FragColor = vec4(bt2020 * c.a, c.a) * (m * alpha);
}
";

// ----------------------------------------------------------------------
// HDR colour pipeline (full per-surface linear compositing).
//
// HDR outputs composite the whole scene into an fp16 offscreen in a
// common LINEAR working space: linear light, BT.2020 primaries,
// normalised so 1.0 == 10000 cd/m² (the PQ peak). Every source is
// decoded into that space as it is drawn, then a final pass PQ-encodes
// the linear buffer to the 10-bit scanout. Mechanisms:
//   - `draw_render_elements` / `render_texture_from_to(None)` consult
//     GlesFrame::override_default_tex_program → we set SDR_DECODE as the
//     scene default and swap to HDR_DECODE around PQ-tagged surfaces.
//   - explicit-program composites (rounded windows, blur clip) bypass the
//     override → dedicated *_HDR shader variants bake the decode in.
//   - the per-window `win_tex` offscreen and the Kawase blur pyramid stay
//     sRGB/8-bit (no override on those sub-frames); they are decoded at
//     composite time by ROUND_TEX_SHADER_HDR / ROUND_BLUR_SHADER_HDR.
// All maths needs `highp` (present on the NVIDIA GLES2 path). Textures
// are premultiplied: un-premultiply, transform straight colour,
// re-premultiply. Verified adversarially (workflow hdr-linear-effects).
// ----------------------------------------------------------------------

/// Encode the composited linear-BT.2020 scene (the fp16 offscreen) to PQ
/// for the 10-bit scanout. Input is already in the working space, so this
/// is just the PQ OETF (no decode / matrix / reference-white scaling —
/// those happen per-source at decode time).
const HDR_ENCODE_SHADER: &str = r"#version 100
//_DEFINES_
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D tex;
uniform float alpha;
varying vec2 v_coords;
vec3 pq_oetf(vec3 l) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 lp = pow(max(l, vec3(0.0)), vec3(m1));
    return pow((vec3(c1) + vec3(c2) * lp) / (vec3(1.0) + vec3(c3) * lp), vec3(m2));
}
void main() {
    vec4 premult = texture2D(tex, v_coords);
    vec3 lin = premult.rgb / max(premult.a, 0.001);
    gl_FragColor = vec4(pq_oetf(lin), 1.0) * alpha;
}
";

/// Tonemap the composited linear-BT.2020 scene (the fp16 offscreen) down to
/// an 8-bit **sRGB** image for screenshots, so a capture of an HDR output
/// "looks like SDR". GLES can't read the fp16 scanout back as an 8-bit
/// format, and even if it could the pixels would be linear BT.2020 — so on
/// HDR outputs captures render through this into an `Abgr8888` buffer first.
///
/// Inverse of the SDR decode for SDR content (exact round-trip): BT.2020→
/// BT.709 gamut, scale so `reference_white` maps back to 1.0, clamp (HDR
/// highlights clip to white, as they would on an SDR display), sRGB OETF.
const SCREENSHOT_TONEMAP_SHADER: &str = r"#version 100
//_DEFINES_
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D tex;
uniform float alpha;
uniform float reference_white;
varying vec2 v_coords;
vec3 linear_to_srgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}
void main() {
    vec4 premult = texture2D(tex, v_coords);
    vec3 bt2020 = premult.rgb / max(premult.a, 0.001);
    mat3 bt2020_to_bt709 = mat3(
        1.660491, -0.124550, -0.018151,
        -0.587641, 1.132900, -0.100579,
        -0.072850, -0.008349, 1.118730
    );
    vec3 lin = bt2020_to_bt709 * bt2020;
    lin *= (10000.0 / reference_white);
    lin = clamp(lin, 0.0, 1.0);
    gl_FragColor = vec4(linear_to_srgb(lin), 1.0) * alpha;
}
";

/// Decode an SDR (sRGB / BT.709) source into the linear BT.2020 working
/// space, mapping SDR diffuse white to `reference_white` cd/m². Set as
/// the scene-frame default override for HDR outputs.
const SDR_DECODE_SHADER: &str = r"#version 100
//_DEFINES_
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D tex;
uniform float alpha;
uniform float reference_white;
uniform float saturation;
varying vec2 v_coords;
vec3 srgb_to_linear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}
// Luma-preserving saturation in linear BT.2020 (1.0 = identity). Lets the
// user punch up SDR content that the BT.709->BT.2020 remap leaves tame.
vec3 saturate_bt2020(vec3 c, float s) {
    float luma = dot(c, vec3(0.2627, 0.6780, 0.0593));
    return max(mix(vec3(luma), c, s), vec3(0.0));
}
void main() {
    vec4 premult = texture2D(tex, v_coords);
    vec3 straight = premult.rgb / max(premult.a, 0.001);
    vec3 lin = srgb_to_linear(straight) * (reference_white / 10000.0);
    mat3 bt709_to_bt2020 = mat3(
        0.627403896, 0.069097289, 0.016391439,
        0.329283038, 0.919540395, 0.088013308,
        0.043313066, 0.011362316, 0.895595253
    );
    vec3 bt2020 = saturate_bt2020(bt709_to_bt2020 * lin, saturation);
    gl_FragColor = vec4(bt2020 * premult.a, premult.a) * alpha;
}
";

/// Fused SDR decode + PQ encode for the single-pass HDR fast path: one
/// opaque SDR fullscreen surface covering an HDR output is drawn straight
/// into the 10-bit scanout — sRGB EOTF → BT.709→BT.2020 → reference-white
/// scale → saturation → PQ OETF in a single fragment, skipping the fp16
/// scene buffer and the separate encode pass entirely (two full-output
/// passes saved per frame, which at 4K/high-Hz is the difference between
/// a game's frame budget and the compositor eating into it). Alpha is
/// forced opaque: eligibility proved the surface covers everything.
const SDR_TO_PQ_SHADER: &str = r"#version 100
//_DEFINES_
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D tex;
uniform float alpha;
uniform float reference_white;
uniform float saturation;
varying vec2 v_coords;
vec3 srgb_to_linear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}
vec3 saturate_bt2020(vec3 c, float s) {
    float luma = dot(c, vec3(0.2627, 0.6780, 0.0593));
    return max(mix(vec3(luma), c, s), vec3(0.0));
}
vec3 pq_oetf(vec3 l) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 lp = pow(max(l, vec3(0.0)), vec3(m1));
    return pow((vec3(c1) + vec3(c2) * lp) / (vec3(1.0) + vec3(c3) * lp), vec3(m2));
}
void main() {
    vec4 premult = texture2D(tex, v_coords);
    vec3 straight = premult.rgb / max(premult.a, 0.001);
    vec3 lin = srgb_to_linear(straight) * (reference_white / 10000.0);
    mat3 bt709_to_bt2020 = mat3(
        0.627403896, 0.069097289, 0.016391439,
        0.329283038, 0.919540395, 0.088013308,
        0.043313066, 0.011362316, 0.895595253
    );
    vec3 bt2020 = saturate_bt2020(bt709_to_bt2020 * lin, saturation);
    gl_FragColor = vec4(pq_oetf(bt2020), 1.0) * alpha;
}
";

/// Decode an HDR PQ / BT.2020 source (a colour-managed client's buffer)
/// into the linear BT.2020 working space. Primaries already match and the
/// PQ EOTF lands in the 1.0 == 10000 cd/m² domain, so no rescale.
const HDR_DECODE_SHADER: &str = r"#version 100
//_DEFINES_
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D tex;
uniform float alpha;
varying vec2 v_coords;
vec3 pq_eotf(vec3 e) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 ep = pow(e, vec3(1.0 / m2));
    return pow(max(ep - vec3(c1), vec3(0.0)) / (vec3(c2) - vec3(c3) * ep), vec3(1.0 / m1));
}
void main() {
    vec4 premult = texture2D(tex, v_coords);
    vec3 straight = premult.rgb / max(premult.a, 0.001);
    vec3 lin = pq_eotf(straight);
    gl_FragColor = vec4(lin * premult.a, premult.a) * alpha;
}
";

/// Renderer for every connected output on a single GPU.
pub struct Renderer {
    /// Shared GLES2 renderer; owns the EGL context.
    gles: GlesRenderer,
    /// GBM scanout allocator, retained so a hot-plugged output can have
    /// its swapchain built at runtime (cloned into each new
    /// `ScanoutSurface`). The dmabuf render formats are re-queried
    /// from `gles` on demand.
    allocator: GbmAllocator<DrmDeviceFd>,
    /// One swapchain + framebuffer chain per output.
    outputs: Vec<OutputRender>,
    /// Index into `outputs` of the layout's primary output. Picked
    /// from `monitors.primary` if set, otherwise the first connected
    /// in DRM enumeration order.
    primary_idx: usize,
    /// Bounding box of the virtual layout in **compositor** (= logical)
    /// pixels, anchored at `(0, 0)`. Used to clamp the cursor across
    /// the full multi-output area.
    layout_bounds: Size<i32, Physical>,
    /// Cursor hotspot in **absolute compositor** coordinates (logical
    /// pixels). Each per-output render translates to that output's
    /// local logical, then scales to physical via `OutputRender::scale`.
    cursor_x: f64,
    cursor_y: f64,
    /// Flat wallpaper fill. Painted full-screen when no media wallpaper is
    /// set, and always used by the frame shader for the rounded-corner
    /// cutout (which can't sample a media texture).
    wallpaper: Fill,
    /// Media wallpaper (decoded image/gif/video frame uploaded as a
    /// texture), drawn full-screen per output in place of `wallpaper` when
    /// set. `None` = use the flat fill.
    wallpaper_media: Option<WallpaperMedia>,
    /// Window border width + active / inactive fills.
    border: BorderConfig,
    /// Loaded `XCursor` `default` arrow sprite. `None` (no theme found)
    /// falls back to the built-in triangle. `Arc`-backed texture, so
    /// it's cheap to clone out before borrowing the renderer for the
    /// frame. Also the fallback when a requested *named* cursor isn't
    /// in the theme.
    cursor: Option<CursorSprite>,
    /// Requested cursor size in **logical** pixels (`$XCURSOR_SIZE`,
    /// default 24). The loaded sprite is normalised back to this size
    /// regardless of which physical-pixel image the theme provided.
    cursor_size: i32,
    /// Physical-pixel target the theme images are chosen for
    /// (`cursor_size * max output scale`), so on-demand named cursors
    /// load at the same crispness as the default.
    cursor_target_px: u32,
    /// What the *focused client* last asked the pointer to look like
    /// (`wl_pointer.set_cursor` / `wp_cursor_shape_v1`); default is the
    /// themed arrow. Overridden by [`Self::cursor_override`] while a
    /// compositor grab is active.
    cursor_status: CursorImageStatus,
    /// A compositor-imposed cursor that takes precedence over the
    /// client's while set — e.g. the grabbing hand during a move/resize
    /// drag, the crosshair during a screenshot selection. `None` =
    /// honour the client's [`Self::cursor_status`].
    cursor_override: Option<CursorImageStatus>,
    /// Lazily-loaded + uploaded sprites for *named* theme cursors other
    /// than the default. A cached `None` means "not in the theme" (so
    /// we don't retry the disk every frame) and the renderer falls back
    /// to the default arrow.
    named_cursors: HashMap<CursorIcon, Option<CursorSprite>>,
    /// Hardware cursor plane (`None` when the driver exposes no cursor plane
    /// or allocation failed → software cursor). Lets pointer motion reposition
    /// the cursor without recompositing the output.
    cursor_plane: Option<CursorPlane>,
    /// Raw (un-uploaded) themed cursor images, cached for the hardware cursor
    /// plane keyed by icon (`None` = not in the theme → falls back to default).
    hw_named: HashMap<CursorIcon, Option<HwCursorImage>>,
    /// Origin used for the monotonic ms timestamp fed into
    /// `wl_callback.done` after each output is queued for scanout.
    /// Clients use this value to schedule their next frame's draw —
    /// the spec defines it as an unsigned 32-bit ms count expected
    /// to wrap freely.
    start: Instant,
    /// Frozen snapshot per output (by connector name), drawn full-screen
    /// while a freeze-mode screenshot session is selecting so the live
    /// desktop appears paused. Empty when no session / not frozen.
    /// `GlesTexture` is `Arc`-backed (cheap to clone out before the frame).
    freeze_textures: HashMap<String, GlesTexture>,
    /// Active screenshot selection overlay (dim wash + highlighted rect),
    /// in absolute compositor coords. `None` when no session is running.
    screenshot_overlay: Option<ScreenshotOverlay>,
    /// Drag-and-drop icon surface (role `dnd_icon`) to composite at the
    /// cursor while a client drag is in progress; `None` otherwise. Set by
    /// the `ClientDndGrabHandler`. Its buffer is read fresh each frame.
    dnd_icon: Option<WlSurface>,
    /// Animation timing/curves, read fresh each frame; updated live on
    /// config reload via [`Self::set_animations`].
    animations: AnimationsConfig,
    /// Window opacity + blur, read fresh each frame; updated live on
    /// config reload via [`Self::set_decoration`].
    decoration: DecorationConfig,
    /// Per-window animation state keyed by surface id: the rect we're
    /// drawing at vs. the layout's target, plus any in-flight open/move
    /// animations. Persists across workspace switches (entries are pruned
    /// only when the surface dies) so hidden windows keep their settled
    /// position instead of replaying an open animation when shown again.
    win_anims: HashMap<ObjectId, WindowAnim>,
    /// Surfaces that just mapped (via [`Self::mark_open`]) and should play
    /// an open animation the next time they appear in a frame's
    /// placements. Keyed separately from `win_anims` so a workspace switch
    /// (which surfaces a window without a fresh map) never triggers it.
    pending_open: HashSet<ObjectId>,
    /// Surface currently under an interactive move/resize drag, if any.
    /// Its rect changes every frame to track the cursor, so it must draw
    /// 1:1 (no move animation) — otherwise it visibly trails the pointer.
    /// Cleared on drop, which lets the window animate into its final tile.
    no_anim_move: Option<ObjectId>,
    /// Windows mid close-animation: a snapshot texture taken the moment
    /// the toplevel was destroyed, fading + shrinking out where the
    /// window last sat. Drained as each finishes.
    closing: Vec<ClosingWindow>,
    /// Kawase dual-filter blur shaders (downsample / upsample halves),
    /// run over the backdrop pyramid to produce the blurred backdrop.
    /// `Arc`-backed, cheap to clone out before borrowing the renderer.
    blur_down: GlesTexProgram,
    blur_up: GlesTexProgram,
    /// Texture shader that composites a window's offscreen surface through a
    /// rounded-rectangle mask (transparent corners + opaque border ring).
    /// See [`ROUND_TEX_SHADER`]. `Arc`-backed, cheap to clone out per frame.
    round_tex_shader: GlesTexProgram,
    /// Texture shader that masks a blurred backdrop tier by the surface's
    /// own alpha channel (bound on unit 1), used for layer-shell panels so
    /// the frost follows the client's real shape. See [`MASK_BLUR_SHADER`].
    mask_blur_shader: GlesTexProgram,
    /// Texture shader that clips a blurred backdrop tier to a rounded-rect
    /// shape, so a rounded window / panel's corners reveal the sharp backdrop
    /// rather than a square block of blur. See [`ROUND_BLUR_SHADER`].
    round_blur_shader: GlesTexProgram,
    /// Per-output offscreen scratch for backdrop blur (keyed by output
    /// index): the rendered backdrop snapshot + the downsample/upsample
    /// mip chain. Built lazily, sized to the output, reused every frame
    /// and rebuilt only when the mode size or pass count changes.
    blur_scratch: HashMap<usize, BlurScratch>,
    /// Texture shader that PQ-encodes the composited linear-BT.2020 scene
    /// (the fp16 offscreen) for an HDR output's 10-bit scanout. See
    /// [`HDR_ENCODE_SHADER`]. `Arc`-backed, cheap to clone out per frame.
    hdr_encode_shader: GlesTexProgram,
    /// Tonemaps the linear-BT.2020 scene to 8-bit sRGB for screenshots of
    /// an HDR output (the fp16 scanout can't be read back as 8-bit). See
    /// [`SCREENSHOT_TONEMAP_SHADER`].
    screenshot_tonemap_shader: GlesTexProgram,
    /// Decodes an SDR (sRGB/BT.709) source into the linear BT.2020 working
    /// space; the scene-frame default override for HDR outputs. See
    /// [`SDR_DECODE_SHADER`].
    sdr_decode_shader: GlesTexProgram,
    /// Fused SDR decode → PQ encode for the single-pass HDR fast path
    /// (one opaque SDR fullscreen surface drawn straight into the 10-bit
    /// scanout). See [`SDR_TO_PQ_SHADER`].
    sdr_to_pq_shader: GlesTexProgram,
    /// Decodes an HDR (PQ/BT.2020) source into the linear working space;
    /// swapped in around colour-managed surfaces. See [`HDR_DECODE_SHADER`].
    hdr_decode_shader: GlesTexProgram,
    /// HDR variant of `round_tex_shader` for SDR windows on an HDR output
    /// (decodes their sRGB offscreen → linear BT.2020). See
    /// [`ROUND_TEX_SHADER_HDR`].
    round_tex_shader_hdr: GlesTexProgram,
    /// Composite for HDR *windows* whose fp16 offscreen is already linear
    /// BT.2020 (no decode; border passed pre-linearised). See
    /// [`ROUND_TEX_SHADER_LINEAR`].
    round_tex_shader_linear: GlesTexProgram,
    /// HDR variant of `round_blur_shader`. See [`ROUND_BLUR_SHADER_HDR`].
    round_blur_shader_hdr: GlesTexProgram,
    /// HDR variant of `mask_blur_shader`. See [`MASK_BLUR_SHADER_HDR`].
    mask_blur_shader_hdr: GlesTexProgram,
    /// Per-output offscreen the HDR scene is composited into before the
    /// PQ-encode pass, keyed by output name. fp16 (linear BT.2020), sized
    /// to the output's mode, rebuilt when the size changes; only for HDR.
    hdr_scene: HashMap<String, GlesTexture>,
    /// Per-output 8-bit `Abgr8888` scratch the HDR scene is tonemapped into
    /// for screenshots / screencopy (the fp16 scanout can't be read back as
    /// 8-bit). Cached + reused so continuous capture (OBS) doesn't re-alloc a
    /// full-output buffer every frame; rebuilt only on size change.
    sdr_capture: HashMap<String, GlesTexture>,
}

/// Number of backdrop blur tiers: 0 = base (wallpaper + lower layers,
/// behind tiled windows), 1 = base + tiled windows (behind floating
/// windows), 2 = full backdrop (behind Top/Overlay layers). Each is the
/// scene accumulated up to that z-band, blurred and saved.
const BLUR_TIERS: usize = 3;

/// Offscreen textures backing one output's backdrop blur. `scene` is the
/// progressive backdrop accumulator (each z-band drawn on top of the
/// previous, no clear); `levels` is the dual-filter mip working chain
/// (`levels[k]` at `size >> k`); `tiers[i]` holds the full-resolution
/// blurred backdrop saved for z-band `i`. All `Arc`-backed, so the saved
/// tiers are cheap to clone out for the frame.
struct BlurScratch {
    /// Full-output buffer size the chain was built for (= `mode_size`).
    size: Size<i32, smithay::utils::Buffer>,
    /// The backdrop scene accumulated this frame (unblurred, full res).
    scene: GlesTexture,
    /// Mip chain, `levels[k]` at `size >> k` (full res at `k = 0`).
    levels: Vec<GlesTexture>,
    /// Per-tier full-resolution blurred backdrops (see [`BLUR_TIERS`]).
    tiers: Vec<GlesTexture>,
}

/// The drawable part of a media wallpaper: the current frame uploaded as a
/// GLES texture plus how to fit it. `Arc`-backed texture, so it's cheap to
/// clone into the frame's backdrop closures.
#[derive(Clone)]
struct WpDraw {
    texture: GlesTexture,
    /// Texture dimensions in its own pixels.
    width: i32,
    height: i32,
    /// How to fit it to each output.
    mode: ScaleMode,
}

/// A media wallpaper: the current drawable frame plus its decode source —
/// a background thread feeding new frames (video/gif). A still image's
/// thread self-terminates after one frame, leaving `draw` static.
struct WallpaperMedia {
    draw: WpDraw,
    anim: crate::media::Animation,
    /// Sequence number of the last frame uploaded from `anim`.
    last_seq: u64,
}

/// A window's last frame, captured at destroy time, animating out.
struct ClosingWindow {
    /// Snapshot of the window's content at close (physical pixels).
    texture: GlesTexture,
    /// Where the content sat on screen — absolute compositor pixels.
    rect: Rectangle<i32, Physical>,
    /// The fade/shrink-out timeline.
    anim: Animation,
}

/// Fraction of full size a window starts at when it opens (and shrinks to
/// when it closes): a subtle pop, not a dramatic zoom.
const OPEN_SCALE_FROM: f64 = 0.90;

/// Per-window animation state. Rects are absolute compositor pixels (the
/// same space as [`Placement::cell_rect`]).
struct WindowAnim {
    /// Kept to prune the entry once the window is gone (`!alive()`).
    surface: WlSurface,
    /// The layout's current target rect (last seen `cell_rect`).
    target: Rectangle<i32, Physical>,
    /// The rect actually drawn last frame — the start point a new move
    /// animation interpolates *from*, so retargets mid-flight stay smooth.
    displayed: Rectangle<i32, Physical>,
    /// Rect a running move animation interpolates from.
    move_from: Rectangle<i32, Physical>,
    /// In-flight position/size animation, if any.
    move_anim: Option<Animation>,
    /// In-flight open (fade + scale-in) animation, if any.
    open_anim: Option<Animation>,
}

/// What to draw for one placement this frame, after animation: the
/// on-screen rect (interpolated position/size) and opacity. The element
/// builder derives the surface's content scale from `effective` vs the
/// placement's target `cell_rect`.
#[derive(Debug, Clone, Copy)]
struct WinDraw {
    effective: Rectangle<i32, Physical>,
    alpha: f32,
}

/// What the screenshot selection UI should draw this frame. The
/// rectangle is in absolute compositor coords; `None` means a session is
/// active but nothing is selected yet (just dim every output).
#[derive(Debug, Clone, Copy)]
pub struct ScreenshotOverlay {
    pub selection: Option<Rectangle<i32, Physical>>,
}

/// One output's render state.
///
/// Internally, the layout works in **compositor** pixels (= logical):
/// `compositor_position` + `compositor_size` describe where the output
/// sits in that space. The DRM framebuffer is in **physical** pixels
/// (`mode_size`); `scale` is the multiplier between the two
/// (`mode_size = compositor_size * scale`, give or take rounding).
/// Per-output `render` multiplies everything that hits the
/// `GlesFrame` by `scale` to land at the right physical pixel.
/// Scanout format preference for an output's swapchain. HDR outputs try
/// 10-bit first so the link can carry a Rec.2020 / PQ signal, then fall
/// back to 8-bit so allocation still succeeds on a panel/driver that
/// rejects 10-bit (the HDR apply then logs that the link stayed SDR).
fn scanout_formats(hdr: bool) -> &'static [Fourcc] {
    if hdr {
        &[Fourcc::Abgr2101010, Fourcc::Xbgr2101010, Fourcc::Xrgb8888]
    } else {
        &[Fourcc::Xrgb8888]
    }
}

/// Whether a chosen scanout fourcc carries 10 bits per colour channel.
fn is_10bit(format: Fourcc) -> bool {
    matches!(
        format,
        Fourcc::Abgr2101010 | Fourcc::Xbgr2101010 | Fourcc::Argb2101010 | Fourcc::Xrgb2101010
    )
}

/// SDR saturation multiplier for an HDR output, from config (default 1.0),
/// as the f32 the decode shaders expect.
fn output_sdr_saturation(cfg: Option<&crate::config::OutputConfig>) -> f32 {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "saturation is a small (~1–2) multiplier; f32 is plenty"
    )]
    let sat = cfg.map_or(1.0_f64, |c| c.sdr_saturation) as f32;
    sat
}

/// Stage this output's HDR connector properties on the freshly-built
/// surface so they ride smithay's first modeset (the surface's initial
/// commit) in one coherent commit, rather than a separate side-channel
/// commit that wedges the pipe.
///
/// Only acts when `hdr` is `true`: the SDR path is left completely
/// untouched so a non-HDR output's modeset is byte-for-byte what it was
/// before HDR support existed (no regression risk). A consequence is
/// that toggling HDR *off* at runtime does not actively clear the
/// connector's BT2020/PQ signalling — the panel may stay in HDR mode
/// (showing SDR content) until the compositor restarts. Never fails the
/// build: a connector that can't do HDR just stays SDR, logged.
fn stage_hdr(surface: &ScanoutSurface, connector: connector::Handle, hdr: bool, name: &str) {
    if !hdr {
        return;
    }
    match crate::hdr::hdr_metadata(surface.surface(), connector) {
        Ok(Some(meta)) => {
            if let Err(err) = surface.surface().set_hdr(Some(meta)) {
                warn!(output = %name, error = %err, "DrmSurface::set_hdr failed; output stays SDR");
            } else {
                info!(output = %name, "HDR connector props staged (BT2020/PQ/max-bpc); output is HDR");
            }
        }
        Ok(None) => warn!(
            output = %name,
            "HDR requested but connector exposes no HDR_OUTPUT_METADATA; staying SDR"
        ),
        Err(err) => {
            warn!(output = %name, error = %err, "could not read connector properties for HDR");
        }
    }
}

struct OutputRender {
    name: String,
    crtc: crtc::Handle,
    /// Connector scanning out this output. Kept so idle DPMS power-off can
    /// target it (the DPMS state is a connector property).
    connector: connector::Handle,
    surface: ScanoutSurface,
    /// DRM framebuffer dimensions in physical pixels.
    mode_size: Size<i32, Physical>,
    /// DRM mode refresh rate in milli-Hz (so 144 Hz = `144_000`).
    /// Threaded out to `wl_output.refresh` so clients see the real
    /// rate they're driving against.
    refresh_mhz: i32,
    /// This output's area in absolute compositor coords (logical).
    compositor_position: Point<i32, Physical>,
    compositor_size: Size<i32, Physical>,
    /// Fractional scale; physical = compositor * scale (component-wise).
    scale: f64,
    /// Configured Variable Refresh Rate policy for this output.
    vrr_mode: VrrMode,
    /// Whether this output's connector advertises adaptive-sync, queried
    /// once at init. `NotSupported` outputs ignore `vrr_mode` entirely
    /// (we never touch their `VRR_ENABLED` property).
    vrr_support: VrrSupport,
    /// Whether this output is in HDR mode. When set, the scene is
    /// composited into an offscreen and a post-process pass encodes it to
    /// PQ / BT.2020 (see `render_output`). SDR outputs (`false`) take the
    /// unchanged direct-to-scanout path.
    hdr: bool,
    /// SDR reference white (cd/m²) for this output's HDR encode — how
    /// bright SDR content maps into the PQ signal. Ignored unless `hdr`.
    hdr_reference_white: u32,
    /// Saturation multiplier applied to SDR content in this output's HDR
    /// encode (1.0 = colorimetrically accurate). Ignored unless `hdr`.
    hdr_saturation: f32,
    /// `wp_presentation` feedback for the frame currently in flight on this
    /// output. Collected at queue/flip time, fired with the real vblank
    /// timestamp in `frame_submitted`. `None` between frames. Only one flip
    /// is ever in flight per output (the `WaitingForVblank` guard), so a
    /// single slot suffices.
    pending_feedback: Option<OutputPresentationFeedback>,
}

/// Public snapshot of one output's geometry for callers (the screenshot
/// tool) that need to map between compositor and framebuffer pixels.
#[derive(Debug, Clone)]
pub struct OutputGeom {
    pub name: String,
    /// Area in absolute compositor (logical) coordinates.
    pub compositor: Rectangle<i32, Physical>,
    /// Fractional scale: physical = compositor * scale.
    pub scale: f64,
    /// Framebuffer size in physical pixels.
    pub mode_size: Size<i32, Physical>,
}

impl From<&OutputRender> for OutputGeom {
    fn from(o: &OutputRender) -> Self {
        Self {
            name: o.name.clone(),
            compositor: Rectangle::new(o.compositor_position, o.compositor_size),
            scale: o.scale,
            mode_size: o.mode_size,
        }
    }
}

/// Logical (compositor) size of an output: physical mode pixels divided
/// by the output's scale. Centralised so output construction and
/// `reflow_outputs` round identically.
fn output_compositor_size(mode: Size<i32, Physical>, scale: f64) -> Size<i32, Physical> {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "mode pixels are u16-bounded; divided by scale > 0 fits in i32 trivially"
    )]
    Size::<i32, Physical>::new(
        (f64::from(mode.w) / scale).round() as i32,
        (f64::from(mode.h) / scale).round() as i32,
    )
}

/// Strict rectangle overlap: their X *and* Y ranges must genuinely
/// intersect. Edge-touching (a shared border, where one's right edge
/// equals the other's left) is *not* overlap, so adjacent screens pass.
fn rects_overlap(a: Rectangle<i32, Physical>, b: Rectangle<i32, Physical>) -> bool {
    a.loc.x < b.loc.x + b.size.w
        && b.loc.x < a.loc.x + a.size.w
        && a.loc.y < b.loc.y + b.size.h
        && b.loc.y < a.loc.y + a.size.h
}

/// One pinned output during placement: `(connector name, logical size,
/// requested top-left)`. A named alias keeps [`place_outputs`]'s working
/// vector readable.
type PinnedOutput<'a> = (&'a String, Size<i32, Physical>, (i32, i32));

/// Assign every output an absolute compositor-space position so that no
/// two outputs ever overlap (overlapping outputs scan out the same
/// compositor region onto both screens — a visible "merge").
///
/// `sizes` is each `(connector name, logical size)`. The invariant is
/// upheld in two stages:
///
/// 1. **Configured outputs** (those the user pinned to a `position`) are
///    placed left-to-right by their requested position. Each is honoured
///    at its exact spot *unless* it would overlap an already-placed
///    output, in which case it's pushed right (X only — the configured Y
///    is preserved, so vertical/stacked layouts are untouched) just far
///    enough to clear the collision. This keeps screens adjacent through
///    a live scale change, where a widened output would otherwise grow
///    over its neighbour's pinned position.
/// 2. **Auto-placed outputs** (no `position`) pack left-to-right beyond
///    every pinned one, so a freshly-connected screen never lands on top
///    of a configured one regardless of connector enumeration order.
fn place_outputs(
    monitors: &MonitorsConfig,
    sizes: &[(String, Size<i32, Physical>)],
) -> HashMap<String, Point<i32, Physical>> {
    let mut positions = HashMap::with_capacity(sizes.len());
    let mut placed: Vec<Rectangle<i32, Physical>> = Vec::new();
    let mut auto_x: i32 = 0;

    // Stage 1: configured outputs, leftmost-requested first so the
    // leftmost anchors and only later ones move on collision.
    let mut configured: Vec<PinnedOutput> = sizes
        .iter()
        .filter_map(|(name, size)| {
            monitors
                .outputs
                .get(name)
                .and_then(|c| c.position)
                .map(|pos| (name, *size, pos))
        })
        .collect();
    configured.sort_by(|(na, _, pa), (nb, _, pb)| {
        pa.0.cmp(&pb.0).then(pa.1.cmp(&pb.1)).then_with(|| na.cmp(nb))
    });

    for (name, size, (req_x, req_y)) in configured {
        let mut x = req_x;
        // Push right past any placed rect we'd overlap. Only +x, so the
        // configured Y stays — a vertical stack (same X, different Y)
        // never collides and never moves.
        loop {
            let rect = Rectangle::new(Point::from((x, req_y)), size);
            let Some(blocker) = placed.iter().find(|r| rects_overlap(**r, rect)) else {
                break;
            };
            let cleared = blocker.loc.x.saturating_add(blocker.size.w);
            // Guard against a non-advancing (or backward) step so the
            // loop always terminates.
            if cleared <= x {
                break;
            }
            x = cleared;
        }
        if x != req_x {
            warn!(
                output = %name,
                requested_x = req_x,
                placed_x = x,
                "output position overlapped another output; shifted right to avoid a merge"
            );
        }
        placed.push(Rectangle::new(Point::from((x, req_y)), size));
        positions.insert(name.clone(), Point::from((x, req_y)));
        auto_x = auto_x.max(x.saturating_add(size.w));
    }

    // Stage 2: auto-placed outputs pack left-to-right beyond the pinned
    // set (auto_x is the rightmost configured edge, so they can't overlap
    // any configured output).
    for (name, size) in sizes {
        if monitors.outputs.get(name).and_then(|c| c.position).is_none() {
            positions.insert(name.clone(), Point::from((auto_x, 0)));
            auto_x = auto_x.saturating_add(size.w);
        }
    }
    positions
}

/// A cursor theme image uploaded to a GLES texture, plus the geometry
/// needed to place it. Cheap to clone (texture is `Arc`-backed).
#[derive(Clone)]
struct CursorSprite {
    texture: GlesTexture,
    /// Texture dimensions in its own pixels.
    width: i32,
    height: i32,
    /// Hotspot in texture pixels — the point that sits exactly on the
    /// pointer position.
    xhot: i32,
    yhot: i32,
    /// Nominal size the artwork was authored for. The draw scale is
    /// `cursor_size / nominal * output_scale`, so the sprite always
    /// renders at the requested logical size however many physical
    /// pixels the chosen theme image carried.
    nominal: i32,
}

/// SMPTE 2084 PQ OETF (linear → PQ-encoded, 0..1), the CPU mirror of
/// `HDR_ENCODE_SHADER`'s `pq_oetf`. Used to encode the cursor image for the
/// hardware cursor plane on HDR outputs (the plane is scanned out by the
/// display, bypassing our GLES PQ-encode, so we must bake PQ into its pixels).
fn pq_oetf(l: f32) -> f32 {
    const M1: f32 = 0.159_301_76;
    const M2: f32 = 78.843_75;
    const C1: f32 = 0.835_937_5;
    const C2: f32 = 18.851_562;
    const C3: f32 = 18.687_5;
    let lp = l.max(0.0).powf(M1);
    ((C1 + C2 * lp) / (1.0 + C3 * lp)).powf(M2)
}

/// A themed cursor sprite as raw premultiplied RGBA (top row first), kept so
/// it can be rasterised into the hardware cursor-plane buffer at any output
/// scale / colour space without re-decoding the theme.
#[derive(Clone)]
struct HwCursorImage {
    rgba: Vec<u8>,
    width: i32,
    height: i32,
    xhot: i32,
    yhot: i32,
    /// Nominal authored size (themed cursors) — the basis for the draw scale
    /// `cursor_size / nominal × output_scale`. Unused when `surface_scale` is
    /// `Some` (client surface cursors scale by `output_scale / buffer_scale`).
    nominal: i32,
    /// Client cursor surface buffer scale; `None` for themed cursors.
    surface_scale: Option<i32>,
}

impl From<crate::cursor::CursorImage> for HwCursorImage {
    fn from(c: crate::cursor::CursorImage) -> Self {
        Self {
            rgba: c.rgba,
            width: c.width,
            height: c.height,
            xhot: c.xhot,
            yhot: c.yhot,
            nominal: c.nominal.max(1),
            surface_scale: None,
        }
    }
}

/// Identifies which cursor is loaded into the plane buffer, so a redraw can
/// skip re-rasterising when it hasn't changed (re-reading a client cursor via
/// GPU readback every frame would be wasteful).
#[derive(Clone, PartialEq)]
enum CursorKey {
    Named(CursorIcon),
    Surface(ObjectId),
}

/// The `ObjectId` of a cursor surface's currently-committed buffer, used to
/// detect when a client surface cursor changed (incl. animation frames).
fn current_buffer_id(surface: &WlSurface) -> Option<ObjectId> {
    with_states(surface, |states| {
        let mut attrs = states.cached_state.get::<SurfaceAttributes>();
        match &attrs.current().buffer {
            Some(BufferAssignment::NewBuffer(b)) => Some(b.id()),
            _ => None,
        }
    })
}

/// Key describing what is currently rasterised into the cursor BO, so a
/// reposition or redraw can skip a redundant re-upload + `set_cursor2`.
#[derive(Clone, Copy, PartialEq)]
struct RenderedCursor {
    crtc: crtc::Handle,
    hdr: bool,
    reference_white: u32,
    image_gen: u64,
    /// Scale ×1000 (so the key is `Eq`) the sprite was rasterised at.
    factor_milli: u32,
    /// Hotspot in plane pixels (the point that tracks the pointer).
    hot_x: i32,
    hot_y: i32,
}

/// Hardware cursor plane. The cursor image lives in a small GBM buffer handed
/// to the DRM cursor plane via the legacy `set_cursor2` / `move_cursor`
/// ioctls; on atomic drivers (including NVIDIA) the kernel routes these to the
/// universal cursor plane (same path [`crate::drm`] uses to clear the DM's
/// leftover cursor). Moving the pointer becomes a cheap `move_cursor` instead
/// of recompositing the whole output, and the cursor is scanned out by the
/// display hardware rather than blended into every frame.
struct CursorPlane {
    /// `ControlDevice` handle for the cursor ioctls (clone of the DRM fd).
    fd: DrmDeviceFd,
    /// Plane-sized, `Argb8888`, mappable cursor image buffer.
    bo: GbmBuffer,
    /// CRTC the cursor is currently programmed on (`None` = plane disabled).
    active_crtc: Option<crtc::Handle>,
    /// Sprite the plane should show. `None` while the effective cursor is
    /// hidden or a client surface (handled by the software path).
    image: Option<HwCursorImage>,
    /// `true` while the effective cursor is a client *surface* — the software
    /// path draws it, so pointer motion must still trigger a redraw.
    software: bool,
    /// Bumped whenever `image` changes; part of [`RenderedCursor`].
    image_gen: u64,
    /// Which cursor `image` currently holds — so a redraw can skip rebuilding
    /// it (esp. the GPU readback for client surface cursors) when unchanged.
    loaded_key: Option<CursorKey>,
    /// What's currently in `bo` (skips redundant re-uploads).
    rendered: Option<RenderedCursor>,
}

impl CursorPlane {
    /// Create the cursor plane: query the driver's cursor size cap and
    /// allocate one `CURSOR | WRITE` buffer at that size. Returns `None` (so
    /// the caller keeps the software cursor) if the device reports no cursor
    /// dimensions or the allocation fails.
    fn new(fd: &DrmDeviceFd, gbm: &GbmDevice<DrmDeviceFd>) -> Option<Self> {
        let w = fd.get_driver_capability(DriverCapability::CursorWidth).ok()?;
        let h = fd.get_driver_capability(DriverCapability::CursorHeight).ok()?;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "cursor cap dims are small (<=256 on real hardware)"
        )]
        let (plane_w, plane_h) = (w.max(64) as u32, h.max(64) as u32);
        let bo = gbm
            .create_buffer_object::<()>(
                plane_w,
                plane_h,
                Fourcc::Argb8888,
                GbmBufferFlags::CURSOR | GbmBufferFlags::WRITE,
            )
            .ok()?;
        info!(plane_w, plane_h, "hardware cursor plane ready");
        Some(Self {
            fd: fd.clone(),
            bo: GbmBuffer::from_bo(bo, true),
            active_crtc: None,
            image: None,
            software: false,
            image_gen: 0,
            loaded_key: None,
            rendered: None,
        })
    }

    /// Plane buffer width / height in pixels.
    fn plane_size(&self) -> (i32, i32) {
        use smithay::backend::allocator::Buffer as _;
        let s = self.bo.size();
        (s.w, s.h)
    }

    /// Set the sprite the plane should show (`None` = nothing on the plane).
    /// Bumps the generation so the next program re-rasterises.
    fn set_image(&mut self, image: Option<HwCursorImage>) {
        self.image = image;
        self.image_gen = self.image_gen.wrapping_add(1);
    }

    /// Disable the cursor plane (clear the cursor on its CRTC).
    fn disable(&mut self) {
        if let Some(crtc) = self.active_crtc.take() {
            #[allow(
                deprecated,
                reason = "legacy set_cursor is the portable way to disable the cursor plane on atomic drivers; see crate::drm"
            )]
            let _ = ControlDevice::set_cursor(&self.fd, crtc, None::<&DumbBuffer>);
        }
        self.rendered = None;
    }

    /// (Re)rasterise the current image at `factor` for output colour
    /// (`hdr`/`reference_white`) and bind it to `crtc` via `set_cursor2`,
    /// skipping the work when nothing changed. Returns `false` (caller should
    /// fall back to software) if there's no image or it's too big for the
    /// plane.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::many_single_char_names,
        reason = "cursor dims/scale are small positive values; r/g/b/a are pixel channels"
    )]
    fn program(&mut self, crtc: crtc::Handle, factor: f64, hdr: bool, reference_white: u32) -> bool {
        let Some(image) = self.image.clone() else {
            return false;
        };
        let (plane_w, plane_h) = self.plane_size();
        let dst_w = ((f64::from(image.width) * factor).round() as i32).max(1);
        let dst_h = ((f64::from(image.height) * factor).round() as i32).max(1);
        if dst_w > plane_w || dst_h > plane_h {
            return false; // doesn't fit the plane → software fallback
        }
        let hot_x = (f64::from(image.xhot) * factor).round() as i32;
        let hot_y = (f64::from(image.yhot) * factor).round() as i32;
        let key = RenderedCursor {
            crtc,
            hdr,
            reference_white,
            image_gen: self.image_gen,
            factor_milli: (factor * 1000.0).round() as u32,
            hot_x,
            hot_y,
        };
        if self.rendered == Some(key) && self.active_crtc == Some(crtc) {
            return true; // already programmed identically
        }
        // Crossed to a different output: clear the cursor off the old CRTC so
        // it doesn't leave a frozen ghost on the monitor we just left.
        if let Some(old) = self.active_crtc
            && old != crtc
        {
            #[allow(
                deprecated,
                reason = "legacy set_cursor disables the cursor plane on atomic drivers; see crate::drm"
            )]
            let _ = ControlDevice::set_cursor(&self.fd, old, None::<&DumbBuffer>);
        }
        // Rasterise into a plane-sized ARGB8888 (memory order B,G,R,A),
        // nearest-neighbour scaling the (near-1×) sprite, PQ-encoding when the
        // target output is HDR so the plane's colours match the PQ scanout.
        let (pw, ph) = (plane_w as usize, plane_h as usize);
        let mut buf = vec![0u8; pw * ph * 4];
        let (sw, sh) = (image.width as usize, image.height as usize);
        for dy in 0..dst_h as usize {
            let sy = ((dy as f64) / factor) as usize;
            if sy >= sh {
                break;
            }
            for dx in 0..dst_w as usize {
                let sx = ((dx as f64) / factor) as usize;
                if sx >= sw {
                    break;
                }
                let s = (sy * sw + sx) * 4;
                let (r, g, b, a) = (
                    image.rgba[s],
                    image.rgba[s + 1],
                    image.rgba[s + 2],
                    image.rgba[s + 3],
                );
                let (ob, og, or) = if hdr && a > 0 {
                    // Un-premultiply → sRGB→linear-BT.2020 (ref-white scaled)
                    // → PQ → re-premultiply, matching the SDR-decode + encode
                    // shaders so the cursor reads correctly on a PQ output.
                    let af = f32::from(a) / 255.0;
                    let straight = Color32F::new(
                        f32::from(r) / 255.0 / af,
                        f32::from(g) / 255.0 / af,
                        f32::from(b) / 255.0 / af,
                        1.0,
                    );
                    let lin = srgb_to_linear_bt2020(straight, reference_white, 1.0);
                    let [lr, lg, lb, _] = lin.components();
                    (
                        (pq_oetf(lb) * af * 255.0).round().clamp(0.0, 255.0) as u8,
                        (pq_oetf(lg) * af * 255.0).round().clamp(0.0, 255.0) as u8,
                        (pq_oetf(lr) * af * 255.0).round().clamp(0.0, 255.0) as u8,
                    )
                } else {
                    (b, g, r) // SDR: source is already premultiplied sRGB
                };
                let d = (dy * pw + dx) * 4;
                buf[d] = ob;
                buf[d + 1] = og;
                buf[d + 2] = or;
                buf[d + 3] = a;
            }
        }
        if self.bo.write(&buf).is_err() {
            return false;
        }
        #[allow(
            deprecated,
            reason = "legacy set_cursor2 routes to the cursor plane on atomic drivers (incl. NVIDIA); see crate::drm"
        )]
        let set = ControlDevice::set_cursor2(&self.fd, crtc, Some(&*self.bo), (hot_x, hot_y));
        if let Err(err) = set {
            warn!(error = %err, ?crtc, "set_cursor2 failed; falling back to software cursor");
            self.active_crtc = None;
            self.rendered = None;
            return false;
        }
        self.active_crtc = Some(crtc);
        self.rendered = Some(key);
        true
    }

    /// Move the (already-programmed) cursor so its hotspot sits at output-local
    /// physical pixel `(x, y)`.
    fn position(&self, crtc: crtc::Handle, x: i32, y: i32) {
        let (hot_x, hot_y) = self.rendered.map_or((0, 0), |r| (r.hot_x, r.hot_y));
        #[allow(
            deprecated,
            reason = "legacy move_cursor routes to the cursor plane on atomic drivers; see crate::drm"
        )]
        let _ = ControlDevice::move_cursor(&self.fd, crtc, (x - hot_x, y - hot_y));
    }
}

impl Renderer {
    /// Build the shared EGL/GLES context plus one `ScanoutSurface`
    /// per output. Outputs are placed left-to-right at `y=0` in the
    /// order the DRM layer enumerated them; the cursor is initialised
    /// at the centre of the first output so it's immediately visible.
    #[allow(
        clippy::too_many_lines,
        reason = "linear initialisation sequence (GBM device, EGL display, EGL context, GLES renderer, custom shader, GBM allocator, per-output ScanoutSurfaces). Splitting it forces threading several mid-construction values through extra functions for no real win."
    )]
    pub fn new(
        drm_fd: DrmDeviceFd,
        drm_outputs: Vec<DrmOutput>,
        wallpaper: Fill,
        border: BorderConfig,
        monitors: &MonitorsConfig,
    ) -> Result<Self> {
        info!("phase: opening GBM device");
        // Keep a fd clone for the hardware cursor-plane ioctls (set_cursor2 /
        // move_cursor) before the fd is moved into the GBM device.
        let cursor_fd = drm_fd.clone();
        let gbm_device = GbmDevice::new(drm_fd).context("GbmDevice::new failed")?;
        info!("GBM device created");

        info!("phase: opening EGL display from GBM device");
        #[allow(
            unsafe_code,
            reason = "EGLDisplay::new requires the GbmDevice to outlive the display. \
                      gbm::Device's Clone is Arc-backed; the cloned device lives \
                      inside EGLDisplay for its full lifetime — the underlying \
                      gbm_device stays valid until EGLDisplay drops."
        )]
        // SAFETY: see #[allow] above.
        let egl_display =
            unsafe { EGLDisplay::new(gbm_device.clone()) }.context("EGLDisplay::new failed")?;
        info!("EGL display opened");

        info!("phase: creating EGL context");
        let egl_context = EGLContext::new(&egl_display).context("EGLContext::new failed")?;
        info!("EGL context created");

        info!("phase: creating GLES renderer");
        #[allow(
            unsafe_code,
            reason = "GlesRenderer::new requires single-threaded use of the EGLContext. \
                      The compositor is single-threaded and the Renderer never \
                      crosses threads."
        )]
        // SAFETY: see #[allow].
        let mut gles =
            unsafe { GlesRenderer::new(egl_context) }.context("GlesRenderer::new failed")?;
        info!("GLES renderer created");

        info!("phase: compiling Kawase blur shaders");
        let blur_uniforms = [
            UniformName::new("halfpixel", UniformType::_2f),
            UniformName::new("offset", UniformType::_1f),
        ];
        let blur_down = gles
            .compile_custom_texture_shader(BLUR_DOWN, &blur_uniforms)
            .context("blur downsample shader compile failed")?;
        let blur_up = gles
            .compile_custom_texture_shader(BLUR_UP, &blur_uniforms)
            .context("blur upsample shader compile failed")?;

        info!("phase: compiling rounded-corner composite shader");
        let round_tex_shader = gles
            .compile_custom_texture_shader(
                ROUND_TEX_SHADER,
                &[
                    UniformName::new("size", UniformType::_2f),
                    UniformName::new("radius", UniformType::_1f),
                    UniformName::new("border_width", UniformType::_1f),
                    UniformName::new("border_top", UniformType::_3f),
                    UniformName::new("border_bottom", UniformType::_3f),
                    UniformName::new("output_height", UniformType::_1f),
                    UniformName::new("cell_origin_y", UniformType::_1f),
                ],
            )
            .context("rounded-corner composite shader compile failed")?;
        let round_blur_shader = gles
            .compile_custom_texture_shader(
                ROUND_BLUR_SHADER,
                &[
                    UniformName::new("size", UniformType::_2f),
                    UniformName::new("radius", UniformType::_1f),
                    UniformName::new("local_mul", UniformType::_2f),
                    UniformName::new("local_add", UniformType::_2f),
                ],
            )
            .context("rounded-blur mask shader compile failed")?;
        let mask_blur_shader = gles
            .compile_custom_texture_shader(
                MASK_BLUR_SHADER,
                &[
                    UniformName::new("mask", UniformType::_1i),
                    UniformName::new("mask_mul", UniformType::_2f),
                    UniformName::new("mask_add", UniformType::_2f),
                ],
            )
            .context("alpha-mask blur shader compile failed")?;

        info!("phase: compiling HDR colour-pipeline shaders");
        // PQ-only encode: input is already linear BT.2020 (no extra uniforms).
        let hdr_encode_shader = gles
            .compile_custom_texture_shader(HDR_ENCODE_SHADER, &[])
            .context("HDR output-encode shader compile failed")?;
        let screenshot_tonemap_shader = gles
            .compile_custom_texture_shader(
                SCREENSHOT_TONEMAP_SHADER,
                &[UniformName::new("reference_white", UniformType::_1f)],
            )
            .context("screenshot tonemap shader compile failed")?;
        let sdr_decode_shader = gles
            .compile_custom_texture_shader(
                SDR_DECODE_SHADER,
                &[
                    UniformName::new("reference_white", UniformType::_1f),
                    UniformName::new("saturation", UniformType::_1f),
                ],
            )
            .context("SDR decode shader compile failed")?;
        let sdr_to_pq_shader = gles
            .compile_custom_texture_shader(
                SDR_TO_PQ_SHADER,
                &[
                    UniformName::new("reference_white", UniformType::_1f),
                    UniformName::new("saturation", UniformType::_1f),
                ],
            )
            .context("fused SDR→PQ shader compile failed")?;
        let hdr_decode_shader = gles
            .compile_custom_texture_shader(HDR_DECODE_SHADER, &[])
            .context("HDR decode shader compile failed")?;
        // Rounded-corner / blur HDR variants: same uniforms as their SDR
        // counterparts plus `reference_white`.
        let round_tex_shader_hdr = gles
            .compile_custom_texture_shader(
                ROUND_TEX_SHADER_HDR,
                &[
                    UniformName::new("size", UniformType::_2f),
                    UniformName::new("radius", UniformType::_1f),
                    UniformName::new("border_width", UniformType::_1f),
                    UniformName::new("border_top", UniformType::_3f),
                    UniformName::new("border_bottom", UniformType::_3f),
                    UniformName::new("output_height", UniformType::_1f),
                    UniformName::new("cell_origin_y", UniformType::_1f),
                    UniformName::new("reference_white", UniformType::_1f),
                    UniformName::new("saturation", UniformType::_1f),
                ],
            )
            .context("HDR rounded-corner composite shader compile failed")?;
        // Linear variant for HDR *windows* (surface already decoded in its
        // fp16 win_tex): same geometry uniforms as the SDR shader, no decode.
        let round_tex_shader_linear = gles
            .compile_custom_texture_shader(
                ROUND_TEX_SHADER_LINEAR,
                &[
                    UniformName::new("size", UniformType::_2f),
                    UniformName::new("radius", UniformType::_1f),
                    UniformName::new("border_width", UniformType::_1f),
                    UniformName::new("border_top", UniformType::_3f),
                    UniformName::new("border_bottom", UniformType::_3f),
                    UniformName::new("output_height", UniformType::_1f),
                    UniformName::new("cell_origin_y", UniformType::_1f),
                ],
            )
            .context("HDR-window linear composite shader compile failed")?;
        let round_blur_shader_hdr = gles
            .compile_custom_texture_shader(
                ROUND_BLUR_SHADER_HDR,
                &[
                    UniformName::new("size", UniformType::_2f),
                    UniformName::new("radius", UniformType::_1f),
                    UniformName::new("local_mul", UniformType::_2f),
                    UniformName::new("local_add", UniformType::_2f),
                    UniformName::new("reference_white", UniformType::_1f),
                    UniformName::new("saturation", UniformType::_1f),
                ],
            )
            .context("HDR rounded-blur mask shader compile failed")?;
        let mask_blur_shader_hdr = gles
            .compile_custom_texture_shader(
                MASK_BLUR_SHADER_HDR,
                &[
                    UniformName::new("mask", UniformType::_1i),
                    UniformName::new("mask_mul", UniformType::_2f),
                    UniformName::new("mask_add", UniformType::_2f),
                    UniformName::new("reference_white", UniformType::_1f),
                    UniformName::new("saturation", UniformType::_1f),
                ],
            )
            .context("HDR alpha-mask blur shader compile failed")?;

        info!("phase: creating GBM allocator");
        // Clone the GBM device for cursor-BO allocation before it's moved
        // into the swapchain allocator, then build the hardware cursor plane
        // (None → keep the software cursor).
        let cursor_plane = CursorPlane::new(&cursor_fd, &gbm_device);
        if cursor_plane.is_none() {
            warn!("hardware cursor plane unavailable; using software cursor");
        }
        let allocator = GbmAllocator::new(
            gbm_device,
            GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT,
        );

        info!("phase: building per-output GBM buffered surfaces");
        let renderer_formats = gles.egl_context().dmabuf_render_formats().clone();
        let mut outputs = Vec::with_capacity(drm_outputs.len());
        // Resolve every output's compositor position up front, before
        // the per-output surface loop, so user-pinned monitors are laid
        // out before the auto-placed ones — otherwise an unconfigured
        // second screen would stack on top of a configured one at x=0
        // instead of landing to its right (see `place_outputs`).
        let output_sizes: Vec<(String, Size<i32, Physical>)> = drm_outputs
            .iter()
            .map(|o| {
                let (w, h) = o.mode.size();
                let mode = Size::<i32, Physical>::new(i32::from(w), i32::from(h));
                let scale = monitors.outputs.get(&o.name).map_or(1.0, |c| c.scale);
                (o.name.clone(), output_compositor_size(mode, scale))
            })
            .collect();
        let output_positions = place_outputs(monitors, &output_sizes);

        for drm_output in drm_outputs {
            let (mode_w, mode_h) = drm_output.mode.size();
            let mode_size = Size::<i32, Physical>::new(i32::from(mode_w), i32::from(mode_h));
            // DRM reports vrefresh in Hz (u32). Convert to milli-Hz
            // for wl_output, clamping at i32::MAX in the absurd
            // case of a connector reporting a refresh past ~2 MHz.
            let refresh_mhz =
                i32::try_from(drm_output.mode.vrefresh().saturating_mul(1000)).unwrap_or(i32::MAX);
            let output_cfg = monitors.outputs.get(&drm_output.name);
            let scale = output_cfg.map_or(1.0, |c| c.scale);
            let compositor_size = output_compositor_size(mode_size, scale);
            // Placed in the pre-pass above; every output name is present,
            // so the fallback is unreachable (kept for panic-freedom).
            let compositor_position = output_positions
                .get(&drm_output.name)
                .copied()
                .unwrap_or_default();

            // Grab the connector before the surface is moved into the
            // GBM swapchain — adaptive-sync support is a connector property.
            let connector = drm_output.connector;
            let hdr = output_cfg.is_some_and(|c| c.hdr);
            let surface = ScanoutSurface::new(
                drm_output.surface,
                &allocator,
                scanout_formats(hdr),
                renderer_formats.clone(),
            )
            .with_context(|| {
                format!(
                    "ScanoutSurface::new failed for {} (no compatible scanout format?)",
                    drm_output.name
                )
            })?;
            if hdr && !is_10bit(surface.format()) {
                warn!(
                    output = %drm_output.name,
                    format = ?surface.format(),
                    "HDR requested but driver/plane selected a non-10-bit scanout format; HDR will likely not engage"
                );
            }
            // Stage HDR (or SDR reset) so the surface's first modeset
            // carries the connector properties in one coherent commit.
            stage_hdr(&surface, connector, hdr, &drm_output.name);

            let vrr_mode = output_cfg.map_or_else(VrrMode::default, |c| c.vrr);
            // Query once: the connector's advertised adaptive-sync support.
            // Errors (inactive device, missing property) degrade to
            // NotSupported so the output simply never uses VRR.
            let vrr_support = surface
                .vrr_supported(connector)
                .unwrap_or(VrrSupport::NotSupported);

            info!(
                output = %drm_output.name,
                pos_x = compositor_position.x,
                pos_y = compositor_position.y,
                comp_w = compositor_size.w,
                comp_h = compositor_size.h,
                phys_w = mode_size.w,
                phys_h = mode_size.h,
                refresh_mhz,
                scale,
                ?vrr_mode,
                ?vrr_support,
                hdr,
                "output swapchain ready"
            );

            outputs.push(OutputRender {
                name: drm_output.name,
                crtc: drm_output.crtc,
                connector,
                surface,
                mode_size,
                refresh_mhz,
                compositor_position,
                compositor_size,
                scale,
                vrr_mode,
                vrr_support,
                hdr,
                hdr_reference_white: output_cfg
                    .and_then(|c| c.sdr_reference_white)
                    .unwrap_or(crate::color_management::DEFAULT_SDR_REFERENCE_WHITE),
                hdr_saturation: output_sdr_saturation(output_cfg),
                pending_feedback: None,
            });
        }

        // Compositor-space union of every output's rect. Used by
        // `on_pointer_motion` to clamp the cursor — it can roam
        // anywhere a real pixel exists.
        let mut layout_w: i32 = 0;
        let mut layout_h: i32 = 0;
        for o in &outputs {
            layout_w = layout_w.max(o.compositor_position.x + o.compositor_size.w);
            layout_h = layout_h.max(o.compositor_position.y + o.compositor_size.h);
        }
        let layout_bounds = Size::<i32, Physical>::new(layout_w, layout_h);

        let primary_idx = monitors
            .primary
            .as_deref()
            .and_then(|name| outputs.iter().position(|o| o.name == name))
            .unwrap_or(0);

        info!(
            outputs = outputs.len(),
            primary = %outputs[primary_idx].name,
            layout_w = layout_bounds.w,
            layout_h = layout_bounds.h,
            "render layout finalised"
        );

        // Cursor starts at the centre of the primary output.
        let primary = &outputs[primary_idx];
        let cursor_x =
            f64::from(primary.compositor_position.x) + f64::from(primary.compositor_size.w) / 2.0;
        let cursor_y =
            f64::from(primary.compositor_position.y) + f64::from(primary.compositor_size.h) / 2.0;

        // Load the pointer cursor from the configured XCursor theme.
        // Pick the image sized for the sharpest output (highest scale)
        // so it stays crisp there; lower-scale outputs downscale it.
        let cursor_size = crate::cursor::configured_size();
        let max_scale = outputs.iter().map(|o| o.scale).fold(1.0_f64, f64::max);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "cursor_size and scale are small positive values; the product is a sane pixel count well within u32"
        )]
        let target_px = (f64::from(cursor_size) * max_scale).round() as u32;
        let cursor = Self::upload_cursor(&mut gles, target_px);

        Ok(Self {
            gles,
            allocator,
            outputs,
            primary_idx,
            layout_bounds,
            cursor_x,
            cursor_y,
            wallpaper,
            wallpaper_media: None,
            border,
            cursor,
            #[allow(
                clippy::cast_possible_wrap,
                reason = "cursor_size is a small positive pixel count"
            )]
            cursor_size: cursor_size as i32,
            cursor_target_px: target_px,
            cursor_status: CursorImageStatus::default_named(),
            cursor_override: None,
            named_cursors: HashMap::new(),
            start: Instant::now(),
            freeze_textures: HashMap::new(),
            screenshot_overlay: None,
            dnd_icon: None,
            animations: AnimationsConfig::default(),
            decoration: DecorationConfig::default(),
            win_anims: HashMap::new(),
            pending_open: HashSet::new(),
            no_anim_move: None,
            closing: Vec::new(),
            blur_down,
            blur_up,
            mask_blur_shader,
            mask_blur_shader_hdr,
            round_tex_shader,
            round_blur_shader,
            blur_scratch: HashMap::new(),
            hdr_encode_shader,
            screenshot_tonemap_shader,
            sdr_decode_shader,
            sdr_to_pq_shader,
            hdr_decode_shader,
            round_tex_shader_hdr,
            round_tex_shader_linear,
            round_blur_shader_hdr,
            hdr_scene: HashMap::new(),
            sdr_capture: HashMap::new(),
            cursor_plane,
            hw_named: HashMap::new(),
        })
    }

    /// Load the configured `XCursor` theme's pointer and upload it as a
    /// GLES texture. Returns `None` (caller falls back to the built-in
    /// triangle) if no theme/image is found, or if the upload fails —
    /// a missing cursor must never be fatal.
    fn upload_cursor(gles: &mut GlesRenderer, target_px: u32) -> Option<CursorSprite> {
        let image = crate::cursor::load_default_cursor(target_px)?;
        Self::upload_cursor_image(gles, &image)
    }

    /// Upload a decoded [`crate::cursor::CursorImage`] as a GLES texture.
    /// Returns `None` on upload failure (the caller falls back).
    fn upload_cursor_image(
        gles: &mut GlesRenderer,
        image: &crate::cursor::CursorImage,
    ) -> Option<CursorSprite> {
        let size = Size::<i32, smithay::utils::Buffer>::from((image.width, image.height));
        // `pixels_rgba` is byte order R,G,B,A, which DRM names
        // `Abgr8888` (little-endian, alpha in the MSB). `flipped =
        // false`: XCursor rows run top-to-bottom, same as our render.
        match gles.import_memory(&image.rgba, Fourcc::Abgr8888, size, false) {
            Ok(texture) => Some(CursorSprite {
                texture,
                width: image.width,
                height: image.height,
                xhot: image.xhot,
                yhot: image.yhot,
                nominal: image.nominal,
            }),
            Err(err) => {
                tracing::warn!(error = %err, "failed to upload cursor texture; using built-in sprite");
                None
            }
        }
    }

    /// Record the cursor the focused client requested (via
    /// `wl_pointer.set_cursor` or `wp_cursor_shape_v1`). Takes effect
    /// next frame unless a compositor override is active.
    pub fn set_cursor_status(&mut self, status: CursorImageStatus) {
        self.cursor_status = status;
    }

    /// Impose (or clear, with `None`) a compositor cursor that overrides
    /// the client's — used for the grabbing hand during a move/resize
    /// and the crosshair during a screenshot selection.
    pub fn set_cursor_override(&mut self, status: Option<CursorImageStatus>) {
        self.cursor_override = status;
    }

    /// Resolve a named cursor to an uploaded sprite, loading + caching it
    /// from the theme on first use. Falls back to the default arrow when
    /// the theme doesn't ship the requested cursor.
    fn named_cursor_sprite(&mut self, icon: CursorIcon) -> Option<CursorSprite> {
        if icon == CursorIcon::Default {
            return self.cursor.clone();
        }
        if !self.named_cursors.contains_key(&icon) {
            let sprite = crate::cursor::load_named_cursor(icon, self.cursor_target_px)
                .and_then(|image| Self::upload_cursor_image(&mut self.gles, &image));
            self.named_cursors.insert(icon, sprite);
        }
        self.named_cursors
            .get(&icon)
            .and_then(Clone::clone)
            .or_else(|| self.cursor.clone())
    }

    /// Whether the hardware cursor plane is currently showing the cursor
    /// (themed or client surface) on `crtc` — so the render path can skip
    /// compositing it.
    fn hw_cursor_active(&self, crtc: crtc::Handle) -> bool {
        self.cursor_plane
            .as_ref()
            .is_some_and(|cp| cp.image.is_some() && cp.active_crtc == Some(crtc))
    }

    /// Whether the pointer must be drawn into output `idx`'s *composite*
    /// this frame — the exact inverse of "the cursor plane shows it, or
    /// nothing is shown at all". Mirrors the composite path's cursor arm
    /// (see `render_output`): nothing is drawn when the pointer is
    /// hidden (`hide_cursor` — a lock or a cursorless capture), when its
    /// hotspot is on another output, when the status is `Hidden`, or
    /// when a client cursor surface has no committed buffer (that's how
    /// some clients hide the pointer); a plane-resident cursor scans out
    /// beside any frame. Only the remainder — a software cursor, or a
    /// capture forcing a bake-in — really needs compositing.
    ///
    /// Direct scanout keys off this rather than `hw_cursor_active`
    /// alone: requiring an *active plane* meant a fullscreen game hiding
    /// or locking the pointer (i.e. every game, all session long) could
    /// never scan out, and neither could a game on one output while the
    /// pointer sat on another.
    fn cursor_needs_composite(&self, idx: usize, hide_cursor: bool, compose_cursor: bool) -> bool {
        if hide_cursor || self.cursor_output_idx() != Some(idx) {
            return false;
        }
        let status = self.cursor_override.as_ref().unwrap_or(&self.cursor_status);
        let plane = self.hw_cursor_active(self.outputs[idx].crtc);
        match status {
            CursorImageStatus::Hidden => false,
            CursorImageStatus::Surface(surface) => {
                let mapped = with_renderer_surface_state(surface, |s| s.buffer().is_some())
                    .unwrap_or(false);
                mapped && (compose_cursor || !plane)
            }
            CursorImageStatus::Named(_) => compose_cursor || !plane,
        }
    }

    /// True while a drag-and-drop icon is following the pointer (it's drawn in
    /// the composite, so motion must redraw even with a hardware cursor).
    pub fn has_dnd_icon(&self) -> bool {
        self.dnd_icon.is_some()
    }

    /// Index of the output whose compositor rect contains the cursor hotspot.
    fn cursor_output_idx(&self) -> Option<usize> {
        let (cx, cy) = (self.cursor_x, self.cursor_y);
        self.outputs.iter().position(|o| {
            let r = Rectangle::new(o.compositor_position, o.compositor_size);
            cx >= f64::from(r.loc.x)
                && cy >= f64::from(r.loc.y)
                && cx < f64::from(r.loc.x + r.size.w)
                && cy < f64::from(r.loc.y + r.size.h)
        })
    }

    /// Resolve a named cursor to a raw image for the hardware plane, caching
    /// by icon (falls back to the default arrow when the theme lacks it).
    fn hw_cursor_image_for(&mut self, icon: CursorIcon) -> Option<HwCursorImage> {
        if let Some(cached) = self.hw_named.get(&icon) {
            return cached.clone();
        }
        let img = crate::cursor::load_named_cursor(icon, self.cursor_target_px)
            .or_else(|| crate::cursor::load_default_cursor(self.cursor_target_px))
            .map(HwCursorImage::from);
        self.hw_named.insert(icon, img.clone());
        img
    }

    /// Sync the hardware cursor plane to the effective cursor status (client
    /// request or compositor override) and program it on the output under the
    /// pointer. Idempotent + cheap — safe to call each redraw; it rebuilds the
    /// plane image only when the cursor actually changed (keyed by icon /
    /// surface buffer). No-op without a cursor plane.
    pub fn refresh_hw_cursor(&mut self, pointer_locked: bool) {
        if self.cursor_plane.is_none() {
            return;
        }
        let status = self
            .cursor_override
            .clone()
            .unwrap_or_else(|| self.cursor_status.clone());
        if pointer_locked {
            self.clear_hw_cursor_image();
            return;
        }
        match status {
            CursorImageStatus::Hidden => self.clear_hw_cursor_image(),
            CursorImageStatus::Named(icon) => {
                let key = CursorKey::Named(icon);
                let unchanged = self
                    .cursor_plane
                    .as_ref()
                    .is_some_and(|cp| cp.loaded_key.as_ref() == Some(&key) && cp.image.is_some());
                if !unchanged {
                    let img = self.hw_cursor_image_for(icon);
                    if let Some(cp) = self.cursor_plane.as_mut() {
                        cp.software = false;
                        cp.loaded_key = Some(key);
                        cp.set_image(img);
                    }
                }
                self.program_hw_cursor_current();
            }
            CursorImageStatus::Surface(surface) => {
                let key = current_buffer_id(&surface).map(CursorKey::Surface);
                let unchanged = key.is_some()
                    && self
                        .cursor_plane
                        .as_ref()
                        .is_some_and(|cp| cp.loaded_key == key && cp.image.is_some());
                if !unchanged {
                    let img = self.hw_cursor_from_surface(&surface);
                    let ok = img.is_some();
                    if let Some(cp) = self.cursor_plane.as_mut() {
                        if ok {
                            cp.software = false;
                            cp.loaded_key = key;
                            cp.set_image(img);
                        } else {
                            // No buffer yet, or readback failed → software path.
                            cp.software = true;
                            cp.loaded_key = None;
                            cp.set_image(None);
                            cp.disable();
                        }
                    }
                }
                self.program_hw_cursor_current();
            }
        }
    }

    /// Clear the plane image (hidden / locked): nothing on the plane, not a
    /// software cursor either.
    fn clear_hw_cursor_image(&mut self) {
        if let Some(cp) = self.cursor_plane.as_mut() {
            cp.software = false;
            cp.loaded_key = None;
            cp.set_image(None);
            cp.disable();
        }
    }

    /// Rasterise a client cursor *surface* into a hardware-cursor image by
    /// rendering its buffer (shm or dmabuf — the GLES importer handles both)
    /// into a native-size offscreen and reading it back. `None` (→ software
    /// fallback) when there's no committed buffer or the readback fails.
    fn hw_cursor_from_surface(&mut self, surface: &WlSurface) -> Option<HwCursorImage> {
        use smithay::backend::renderer::buffer_dimensions;
        let (buffer, hot, bscale) = with_states(surface, |states| {
            let hot = states
                .data_map
                .get::<CursorImageSurfaceData>()
                .map(|a| a.lock().unwrap().hotspot)
                .unwrap_or_default();
            let mut attrs = states.cached_state.get::<SurfaceAttributes>();
            let cur = attrs.current();
            let buffer = match &cur.buffer {
                Some(BufferAssignment::NewBuffer(b)) => Some(b.clone()),
                _ => None,
            };
            (buffer, hot, cur.buffer_scale.max(1))
        });
        let buffer = buffer?;
        let dims = buffer_dimensions(&buffer)?;
        if dims.w <= 0 || dims.h <= 0 {
            return None;
        }
        // Render the surface at scale 1.0 into a native-buffer-sized offscreen,
        // then read it back to premultiplied RGBA (same path as screenshots).
        let elements: Vec<WaylandSurfaceRenderElement<GlesRenderer>> =
            render_elements_from_surface_tree(
                &mut self.gles,
                surface,
                Point::from((0, 0)),
                1.0,
                1.0_f32,
                Kind::Cursor,
            );
        if elements.is_empty() {
            return None;
        }
        let tex_size = Size::<i32, smithay::utils::Buffer>::from((dims.w, dims.h));
        let phys = Size::<i32, Physical>::from((dims.w, dims.h));
        let full = [Rectangle::<i32, Physical>::from_size(phys)];
        let mut texture: GlesTexture = self.gles.create_buffer(Fourcc::Abgr8888, tex_size).ok()?;
        let mut target = self.gles.bind(&mut texture).ok()?;
        {
            let mut frame = self.gles.render(&mut target, phys, Transform::Normal).ok()?;
            frame.clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &full).ok()?;
            draw_render_elements::<GlesRenderer, _, _>(&mut frame, 1.0, &elements, &full).ok()?;
            let _ = frame.finish().ok()?;
        }
        let region = Rectangle::<i32, smithay::utils::Buffer>::from_size(tex_size);
        let mapping = self.gles.copy_framebuffer(&target, region, Fourcc::Abgr8888).ok()?;
        let rgba = self.gles.map_texture(&mapping).ok()?.to_vec();
        drop(target);
        Some(HwCursorImage {
            rgba,
            width: dims.w,
            height: dims.h,
            xhot: hot.x * bscale,
            yhot: hot.y * bscale,
            nominal: 1,
            surface_scale: Some(bscale),
        })
    }

    /// Reposition (re-programming if needed) the hardware cursor for the
    /// current pointer location. Returns `true` if the plane handled it (the
    /// caller can skip a full redraw), `false` if the cursor is software
    /// (client surface / no plane / oversize) and a redraw is still needed.
    pub fn move_hw_cursor(&mut self) -> bool {
        let Some(cp) = self.cursor_plane.as_ref() else {
            return false;
        };
        if cp.image.is_none() {
            // Hidden → handled (nothing to draw); surface → software redraw.
            return !cp.software;
        }
        self.program_hw_cursor_current()
    }

    /// Program + position the cursor plane on the output under the pointer.
    /// Returns whether the plane is showing the cursor.
    fn program_hw_cursor_current(&mut self) -> bool {
        if self.cursor_plane.as_ref().is_none_or(|cp| cp.image.is_none()) {
            return false;
        }
        let Some(idx) = self.cursor_output_idx() else {
            if let Some(cp) = self.cursor_plane.as_mut() {
                cp.disable();
            }
            return false;
        };
        let o = &self.outputs[idx];
        let (crtc, scale, hdr, refw, opos) = (
            o.crtc,
            o.scale,
            o.hdr,
            o.hdr_reference_white,
            o.compositor_position,
        );
        let cursor_size = self.cursor_size;
        let (cx, cy) = (self.cursor_x, self.cursor_y);
        let Some(cp) = self.cursor_plane.as_mut() else {
            return false;
        };
        // Themed cursors normalise to the configured logical size; client
        // surface cursors scale by output_scale / buffer_scale.
        let factor = if let Some(bs) = cp.image.as_ref().and_then(|i| i.surface_scale) {
            scale / f64::from(bs.max(1))
        } else {
            let nominal = cp.image.as_ref().map_or(1, |i| i.nominal.max(1));
            f64::from(cursor_size) / f64::from(nominal) * scale
        };
        if !cp.program(crtc, factor, hdr, refw) {
            cp.disable();
            return false;
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "output-local physical cursor coords fit i32"
        )]
        let lx = ((cx - f64::from(opos.x)) * scale) as i32;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "output-local physical cursor coords fit i32"
        )]
        let ly = ((cy - f64::from(opos.y)) * scale) as i32;
        cp.position(crtc, lx, ly);
        true
    }

    /// Render every output's initial frame to prime its swapchain.
    /// Called once at startup before the event loop runs; thereafter
    /// each output's frames are driven by its own vblank events. No
    /// Wayland clients have connected yet at this point, so we pass
    /// an empty placement slice — only the wallpaper + cursor land.
    pub fn render_initial(&mut self) -> Result<()> {
        for idx in 0..self.outputs.len() {
            // Followup ignored: each output is primed once, then parks until
            // a redraw is queued (a flip is now in flight, acked on vblank).
            let _ = self
                .render_output(idx, &[], &[], &[], false, &[], &HashSet::new(), false, None)
                .with_context(|| format!("initial render of output #{idx} failed"))?;
        }
        Ok(())
    }

    /// Full physical (framebuffer) size of the output named `name`,
    /// used by screencopy to tell a client what buffer to allocate.
    pub fn output_mode_size(&self, name: &str) -> Option<Size<i32, Physical>> {
        self.outputs
            .iter()
            .find(|o| o.name == name)
            .map(|o| o.mode_size)
    }

    /// Connector name of the output driven by `crtc`, so the vblank
    /// path can match pending screencopy captures to the output it is
    /// about to render.
    pub fn output_name_for_crtc(&self, crtc: crtc::Handle) -> Option<String> {
        self.outputs
            .iter()
            .find(|o| o.crtc == crtc)
            .map(|o| o.name.clone())
    }

    /// Clamp the cursor hotspot into `rect`. Used while a
    /// confined-pointer constraint is active so the cursor can't leave
    /// the constraining surface. A degenerate rect is ignored.
    ///
    /// The upper bound is `loc + size - 1`, not `loc + size`: hit-tests
    /// use a half-open interval (`pos < loc + size`), so a cursor
    /// clamped exactly to `loc + size` would fall *outside* the surface
    /// on the next frame, fire a `wl_pointer.leave`, and make smithay
    /// auto-deactivate the constraint — letting the cursor escape, the
    /// opposite of confinement. `saturating_add` guards against an
    /// `i32` overflow for a pathological monitor position.
    pub fn confine_cursor(&mut self, rect: Rectangle<i32, Physical>) {
        if rect.size.w <= 0 || rect.size.h <= 0 {
            return;
        }
        self.cursor_x = self.cursor_x.clamp(
            f64::from(rect.loc.x),
            f64::from(rect.loc.x.saturating_add(rect.size.w)) - 1.0,
        );
        self.cursor_y = self.cursor_y.clamp(
            f64::from(rect.loc.y),
            f64::from(rect.loc.y.saturating_add(rect.size.h)) - 1.0,
        );
    }

    /// Render the output driven by `crtc`, in response to its vblank.
    /// `placements` is the caller-snapshot of every visible window as
    /// `(wl_surface, top-left in absolute virtual-layout coords)`;
    /// the layout module owns positioning, the renderer just paints.
    #[allow(
        clippy::too_many_arguments,
        reason = "thin pass-through to render_output; the per-frame inputs are all distinct"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "per-frame render inputs; threading them through a struct would not simplify"
    )]
    pub fn render_for_crtc(
        &mut self,
        crtc: crtc::Handle,
        placements: &[Placement],
        layers: &[LayerPlacement],
        popups: &[PopupPlacement],
        hide_cursor: bool,
        captures: &[CaptureSpec],
        hdr_surface_ids: &HashSet<ObjectId>,
        compose_cursor: bool,
        output: Option<&Output>,
    ) -> Result<(Vec<CaptureOutcome>, bool)> {
        let idx = self
            .outputs
            .iter()
            .position(|o| o.crtc == crtc)
            .with_context(|| format!("vblank for unknown CRTC {crtc:?}"))?;
        self.render_output(
            idx,
            placements,
            layers,
            popups,
            hide_cursor,
            captures,
            hdr_surface_ids,
            compose_cursor,
            output,
        )
    }

    /// Ack a completed page-flip for `crtc` so its swapchain frees the
    /// scanned-out buffer, and send `wp_presentation` feedback for the frame
    /// that just hit the screen using the real vblank timestamp/sequence.
    /// Called from the vblank handler; lets the on-demand driver acknowledge a
    /// flip without being forced to render the next frame (the free-run loop
    /// used to do both at once).
    ///
    /// `present_time` is a `CLOCK_MONOTONIC` instant, `seq` the page-flip
    /// sequence, and `base_flags` the presentation kind (vsync, plus hw-clock
    /// when the timestamp came from the DRM page-flip event). Per-surface
    /// zero-copy flags were already merged in at collection time.
    pub fn frame_submitted(
        &mut self,
        crtc: crtc::Handle,
        present_time: Duration,
        seq: u32,
        base_flags: PresentKind,
    ) {
        let Some(o) = self.outputs.iter_mut().find(|o| o.crtc == crtc) else {
            return;
        };
        if let Err(err) = o.surface.frame_submitted() {
            warn!(error = %err, crtc = ?crtc, "frame_submitted failed");
        }
        if let Some(mut feedback) = o.pending_feedback.take() {
            // refresh_mhz is milli-Hz (144 Hz = 144_000); the frame period is
            // 1/Hz = 1000/mHz seconds.
            let period = Duration::from_secs_f64(1000.0 / f64::from(o.refresh_mhz.max(1)));
            feedback.presented(
                Time::<Monotonic>::from(present_time),
                Refresh::fixed(period),
                u64::from(seq),
                base_flags,
            );
        }
    }

    /// Every output's CRTC, for the driver to iterate when scheduling
    /// redraws across all outputs.
    pub fn crtcs(&self) -> Vec<crtc::Handle> {
        self.outputs.iter().map(|o| o.crtc).collect()
    }

    /// Connector names of every output currently driven, for the
    /// hotplug path to diff against a fresh connector scan.
    pub fn output_names(&self) -> Vec<String> {
        self.outputs.iter().map(|o| o.name.clone()).collect()
    }

    /// Bind a freshly hot-plugged DRM output into the render pipeline:
    /// build its GBM swapchain over the retained allocator, query
    /// adaptive-sync support, and append an [`OutputRender`]. The
    /// compositor position is provisional (`0,0`) — call
    /// [`Self::reflow_outputs`] afterwards to pack every output and
    /// recompute the layout bounds. Per-output scratch caches keyed by
    /// index are cleared (cheap; rebuilt next frame) since the indices
    /// shift. No-op if an output with this connector name already exists.
    pub fn add_output(
        &mut self,
        drm_output: crate::drm::DrmOutput,
        monitors: &MonitorsConfig,
    ) -> Result<()> {
        if self.outputs.iter().any(|o| o.name == drm_output.name) {
            return Ok(());
        }
        let (mode_w, mode_h) = drm_output.mode.size();
        let mode_size = Size::<i32, Physical>::new(i32::from(mode_w), i32::from(mode_h));
        let refresh_mhz =
            i32::try_from(drm_output.mode.vrefresh().saturating_mul(1000)).unwrap_or(i32::MAX);
        let output_cfg = monitors.outputs.get(&drm_output.name);
        let scale = output_cfg.map_or(1.0, |c| c.scale);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "mode pixels are u16-bounded; divided by scale > 0 fits in i32 trivially"
        )]
        let compositor_size = Size::<i32, Physical>::new(
            (f64::from(mode_size.w) / scale).round() as i32,
            (f64::from(mode_size.h) / scale).round() as i32,
        );

        let connector = drm_output.connector;
        let hdr = output_cfg.is_some_and(|c| c.hdr);
        let renderer_formats = self.gles.egl_context().dmabuf_render_formats().clone();
        let surface = ScanoutSurface::new(
            drm_output.surface,
            &self.allocator,
            scanout_formats(hdr),
            renderer_formats,
        )
        .with_context(|| {
            format!(
                "ScanoutSurface::new failed for hot-plugged {} (no compatible scanout format?)",
                drm_output.name
            )
        })?;
        if hdr && !is_10bit(surface.format()) {
            warn!(
                output = %drm_output.name,
                format = ?surface.format(),
                "HDR requested but driver/plane selected a non-10-bit scanout format; HDR will likely not engage"
            );
        }
        stage_hdr(&surface, connector, hdr, &drm_output.name);

        let vrr_mode = output_cfg.map_or_else(VrrMode::default, |c| c.vrr);
        let vrr_support = surface
            .vrr_supported(connector)
            .unwrap_or(VrrSupport::NotSupported);

        info!(output = %drm_output.name, "hot-plugged output swapchain ready");
        self.outputs.push(OutputRender {
            name: drm_output.name,
            crtc: drm_output.crtc,
            connector,
            surface,
            mode_size,
            refresh_mhz,
            // Provisional; `reflow_outputs` rewrites this.
            compositor_position: Point::<i32, Physical>::from((0, 0)),
            compositor_size,
            scale,
            vrr_mode,
            vrr_support,
            hdr,
            hdr_reference_white: output_cfg
                .and_then(|c| c.sdr_reference_white)
                .unwrap_or(crate::color_management::DEFAULT_SDR_REFERENCE_WHITE),
            hdr_saturation: output_sdr_saturation(output_cfg),
            pending_feedback: None,
        });
        self.blur_scratch.clear();
        Ok(())
    }

    /// Tear a hot-unplugged output out of the pipeline. Returns its
    /// connector name (for the caller to clean up its protocol globals).
    /// Drops the output's frozen-snapshot texture and clears the
    /// index-keyed scratch caches. Caller should follow with
    /// [`Self::reflow_outputs`].
    pub fn remove_output(&mut self, crtc: crtc::Handle) -> Option<String> {
        let idx = self.outputs.iter().position(|o| o.crtc == crtc)?;
        let removed = self.outputs.remove(idx);
        self.freeze_textures.remove(&removed.name);
        self.blur_scratch.clear();
        Some(removed.name)
    }

    /// The connector + CRTC currently driving the named output, if any.
    /// Used by a live mode change to rebuild the DRM surface on the same
    /// pipe (drop the old surface, modeset a new one on this CRTC).
    pub fn output_connector_crtc(
        &self,
        name: &str,
    ) -> Option<(connector::Handle, crtc::Handle)> {
        self.outputs
            .iter()
            .find(|o| o.name == name)
            .map(|o| (o.connector, o.crtc))
    }

    /// Recompute every output's compositor position after the output set
    /// changed: outputs the user pinned in config keep their position,
    /// the rest pack left-to-right (new screens land to the right).
    /// Refreshes each output's scale + compositor size from config,
    /// recomputes the layout bounding box, re-resolves the primary
    /// output, clamps the cursor back inside the new bounds, and returns
    /// fresh [`OutputDescriptor`]s for the Wayland layer to re-advertise.
    pub fn reflow_outputs(&mut self, monitors: &MonitorsConfig) -> Vec<OutputDescriptor> {
        // Refresh scale + compositor size from config first, then assign
        // non-overlapping positions in a second pass (configured monitors
        // pinned, the rest packed past them — see `place_outputs`).
        for o in &mut self.outputs {
            let cfg = monitors.outputs.get(&o.name);
            let scale = cfg.map_or(1.0, |c| c.scale);
            o.scale = scale;
            o.compositor_size = output_compositor_size(o.mode_size, scale);
            // VRR policy is read fresh each flip in `apply_vrr`, so
            // refreshing it here makes a config-reload change take effect
            // on the next frame.
            o.vrr_mode = cfg.map_or_else(VrrMode::default, |c| c.vrr);
            // Refresh the HDR tone params so config-reload tuning of
            // `sdr_reference_white` / `sdr_saturation` applies live (the
            // `hdr` toggle itself still needs a swapchain rebuild).
            o.hdr_reference_white = cfg
                .and_then(|c| c.sdr_reference_white)
                .unwrap_or(crate::color_management::DEFAULT_SDR_REFERENCE_WHITE);
            o.hdr_saturation = output_sdr_saturation(cfg);
        }
        let sizes: Vec<(String, Size<i32, Physical>)> = self
            .outputs
            .iter()
            .map(|o| (o.name.clone(), o.compositor_size))
            .collect();
        let positions = place_outputs(monitors, &sizes);
        for o in &mut self.outputs {
            if let Some(&pos) = positions.get(&o.name) {
                o.compositor_position = pos;
            }
        }

        let mut layout_w: i32 = 0;
        let mut layout_h: i32 = 0;
        for o in &self.outputs {
            layout_w = layout_w.max(o.compositor_position.x + o.compositor_size.w);
            layout_h = layout_h.max(o.compositor_position.y + o.compositor_size.h);
        }
        self.layout_bounds = Size::<i32, Physical>::new(layout_w, layout_h);

        self.primary_idx = monitors
            .primary
            .as_deref()
            .and_then(|name| self.outputs.iter().position(|o| o.name == name))
            .unwrap_or(0)
            .min(self.outputs.len().saturating_sub(1));

        // The cursor may now sit beyond the shrunken union (an output to
        // its right vanished); pull it back onto a real pixel.
        self.cursor_x = self.cursor_x.clamp(0.0, f64::from(layout_w));
        self.cursor_y = self.cursor_y.clamp(0.0, f64::from(layout_h));

        self.output_descriptors()
    }

    /// Connectors of every output, for idle DPMS power control.
    pub fn output_connectors(&self) -> Vec<connector::Handle> {
        self.outputs.iter().map(|o| o.connector).collect()
    }

    /// CRTC of the output named `name` (connector name), if present.
    pub fn crtc_for_output_name(&self, name: &str) -> Option<crtc::Handle> {
        self.outputs
            .iter()
            .find(|o| o.name == name)
            .map(|o| o.crtc)
    }

    /// Advance the cursor hotspot by libinput-reported deltas, clamped
    /// to the virtual layout's bounding box.
    pub fn on_pointer_motion(&mut self, dx: f64, dy: f64) {
        let max_x = f64::from(self.layout_bounds.w);
        let max_y = f64::from(self.layout_bounds.h);
        self.cursor_x = (self.cursor_x + dx).clamp(0.0, max_x);
        self.cursor_y = (self.cursor_y + dy).clamp(0.0, max_y);
    }

    /// Current cursor hotspot in absolute virtual-layout coordinates.
    /// Exposed for input routing — the seat needs to compute the
    /// surface-local pointer position for `wl_pointer.motion` events.
    pub fn cursor_pos(&self) -> (f64, f64) {
        (self.cursor_x, self.cursor_y)
    }

    /// Rectangle of the configured primary output in absolute
    /// **compositor** (= logical) coordinates. Used by the tiling
    /// layer to bound its initial workspace before per-output
    /// workspaces exist. `primary_idx` is set in `new()` from
    /// `monitors.primary` (falling back to the first connected),
    /// so the indexing is always safe.
    pub fn primary_output_rect(&self) -> Rectangle<i32, Physical> {
        let o = &self.outputs[self.primary_idx];
        Rectangle::new(o.compositor_position, o.compositor_size)
    }

    /// Every output's `(connector name, compositor rect)` in absolute
    /// compositor pixels. The layout consumes this to build one
    /// tiling tree per output, so windows can tile on any monitor —
    /// not just the primary.
    pub fn output_rects(&self) -> Vec<(String, Rectangle<i32, Physical>)> {
        self.outputs
            .iter()
            .map(|o| {
                (
                    o.name.clone(),
                    Rectangle::new(o.compositor_position, o.compositor_size),
                )
            })
            .collect()
    }

    /// The compositor rect (absolute pixels) of a named output, or
    /// `None` if no connector by that name is present. Used to place a
    /// `wlr_layer_shell` surface on the output it asked for.
    pub fn output_rect(&self, name: &str) -> Option<Rectangle<i32, Physical>> {
        self.outputs
            .iter()
            .find(|o| o.name == name)
            .map(|o| Rectangle::new(o.compositor_position, o.compositor_size))
    }

    /// Geometry of the output containing `point` (absolute compositor
    /// px), if any — its name, compositor rect, fractional scale, and
    /// physical mode size. Used by the screenshot tool to map a
    /// selection in compositor space to one output's framebuffer pixels.
    pub fn output_at(&self, point: Point<i32, Physical>) -> Option<OutputGeom> {
        self.outputs
            .iter()
            .find(|o| {
                let r = Rectangle::new(o.compositor_position, o.compositor_size);
                let local = point - r.loc;
                local.x >= 0 && local.y >= 0 && local.x < r.size.w && local.y < r.size.h
            })
            .map(OutputGeom::from)
    }

    /// Geometry of every output — used by the screenshot tool to snapshot
    /// all outputs for a freeze.
    pub fn output_geometries(&self) -> Vec<OutputGeom> {
        self.outputs.iter().map(OutputGeom::from).collect()
    }

    /// Connector name of the primary output, or `None` when no output is
    /// connected (every monitor unplugged — the compositor runs headless
    /// until one returns). Used by the layer-shell reflow to attribute
    /// exclusive zones to the primary by name.
    pub fn primary_output_name(&self) -> Option<&str> {
        self.outputs.get(self.primary_idx).map(|o| o.name.as_str())
    }

    /// Swap the wallpaper + border styling used from the next frame
    /// on (for live config reload). The frame shader and wallpaper
    /// fill are read fresh each render, so the change shows up on the
    /// next vblank with no further action. Border *width* also feeds
    /// client window sizing, which the layout updates separately.
    pub fn set_appearance(&mut self, wallpaper: Fill, border: BorderConfig) {
        self.wallpaper = wallpaper;
        self.border = border;
    }

    /// Set (or clear) the media wallpaper. `Some((rgba, w, h, mode, anim))`
    /// uploads the first packed-RGBA frame as a texture drawn full-screen
    /// per output in `mode`, and keeps `anim` (the decode thread) feeding
    /// later frames via [`Self::refresh_wallpaper`]; `None` reverts to the
    /// flat [`Self::set_appearance`] fill. Returns whether the upload
    /// succeeded — a failure clears the media so the flat fill shows.
    pub fn set_wallpaper_media(
        &mut self,
        init: Option<(&[u8], i32, i32, ScaleMode, crate::media::Animation)>,
    ) -> bool {
        let Some((rgba, width, height, mode, anim)) = init else {
            self.wallpaper_media = None;
            return true;
        };
        let size = Size::<i32, smithay::utils::Buffer>::from((width, height));
        match self.gles.import_memory(rgba, Fourcc::Abgr8888, size, false) {
            Ok(texture) => {
                self.wallpaper_media = Some(WallpaperMedia {
                    draw: WpDraw {
                        texture,
                        width,
                        height,
                        mode,
                    },
                    anim,
                    last_seq: 0,
                });
                true
            }
            Err(err) => {
                warn!(error = %err, "wallpaper: media texture upload failed");
                self.wallpaper_media = None;
                false
            }
        }
    }

    /// Poll the media wallpaper's decode thread and, if it has produced a
    /// newer frame, upload it as the current wallpaper texture. Called once
    /// per output render; the sequence check makes the extra calls when
    /// several outputs render per vblank cheap no-ops, and re-uploads
    /// happen only at the media's frame rate.
    fn refresh_wallpaper(&mut self) {
        let Some(media) = self.wallpaper_media.as_ref() else {
            return;
        };
        let Some((frame, seq)) = media.anim.take_new(media.last_seq) else {
            return;
        };
        #[allow(
            clippy::cast_possible_wrap,
            reason = "decoded dims are capped to output size, well within i32"
        )]
        let (width, height) = (frame.width as i32, frame.height as i32);
        let size = Size::<i32, smithay::utils::Buffer>::from((width, height));
        match self.gles.import_memory(&frame.rgba, Fourcc::Abgr8888, size, false) {
            Ok(texture) => {
                if let Some(media) = self.wallpaper_media.as_mut() {
                    media.draw.texture = texture;
                    media.draw.width = width;
                    media.draw.height = height;
                    media.last_seq = seq;
                }
            }
            Err(err) => warn!(error = %err, "wallpaper: animated frame upload failed"),
        }
    }

    /// Set (or clear) the screenshot selection overlay drawn over every
    /// output from the next frame on. The rectangle is in absolute
    /// compositor coords; each output renders the part that falls on it.
    pub fn set_screenshot_overlay(&mut self, overlay: Option<ScreenshotOverlay>) {
        self.screenshot_overlay = overlay;
    }

    /// Upload a captured frame as the frozen backdrop for `output` (used
    /// by freeze-mode screenshots). `rgba` is **top-down, fully-opaque
    /// RGBA** (see `screenshot::to_rgba_topdown`) — the same byte order
    /// and orientation as the cursor sprite, so it imports via the
    /// renderer's known-good `Abgr8888` / `flipped = false` path and
    /// displays upright + opaque. Returns whether the upload succeeded.
    pub fn set_freeze_texture(&mut self, output: &str, rgba: &[u8], width: i32, height: i32) -> bool {
        let size = Size::<i32, smithay::utils::Buffer>::from((width, height));
        match self.gles.import_memory(rgba, Fourcc::Abgr8888, size, false) {
            Ok(texture) => {
                self.freeze_textures.insert(output.to_owned(), texture);
                true
            }
            Err(err) => {
                warn!(error = %err, output, "screenshot: freeze texture upload failed");
                false
            }
        }
    }

    /// Tear down all screenshot state (overlay + frozen textures) when a
    /// session ends or is cancelled, so the next frame renders live again.
    pub fn clear_screenshot(&mut self) {
        self.screenshot_overlay = None;
        self.freeze_textures.clear();
    }

    /// Set (or clear) the drag-and-drop icon surface composited at the
    /// cursor while a client drag is active.
    pub fn set_dnd_icon(&mut self, icon: Option<WlSurface>) {
        self.dnd_icon = icon;
    }

    /// Replace the animation timing/curves (live config reload). Takes
    /// effect next frame; animations already in flight keep their timing.
    pub fn set_animations(&mut self, cfg: AnimationsConfig) {
        self.animations = cfg;
    }

    /// Replace the decoration config (window opacity + blur). Live config
    /// reload; read fresh next frame.
    pub fn set_decoration(&mut self, cfg: DecorationConfig) {
        self.decoration = cfg;
    }

    /// Ensure output `idx` has a backdrop-blur scratch chain sized for its
    /// `mode_size` with at least `passes + 1` mip levels, building (or
    /// rebuilding) it on the first frame or after a size / pass-count
    /// change. Returns `false` if any GPU texture allocation fails, in
    /// which case the caller skips blur for this frame.
    fn ensure_blur_scratch(
        &mut self,
        idx: usize,
        mode_size: Size<i32, Physical>,
        passes: u32,
    ) -> bool {
        let size = Size::<i32, smithay::utils::Buffer>::from((mode_size.w, mode_size.h));
        let need = passes as usize + 1;
        if let Some(s) = self.blur_scratch.get(&idx)
            && s.size == size
            && s.levels.len() >= need
        {
            return true;
        }
        let mut make = |w: i32, h: i32| {
            self.gles.create_buffer(
                Fourcc::Abgr8888,
                Size::<i32, smithay::utils::Buffer>::from((w.max(1), h.max(1))),
            )
        };
        let scene = match make(size.w, size.h) {
            Ok(t) => t,
            Err(err) => {
                warn!(error = %err, "blur: scene buffer alloc failed");
                return false;
            }
        };
        let mut levels = Vec::with_capacity(need);
        for k in 0..need {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "k <= passes <= 10, so the shift never overflows i32"
            )]
            let (w, h) = (size.w >> k as u32, size.h >> k as u32);
            match make(w, h) {
                Ok(t) => levels.push(t),
                Err(err) => {
                    warn!(error = %err, "blur: mip level alloc failed");
                    return false;
                }
            }
        }
        let mut tiers = Vec::with_capacity(BLUR_TIERS);
        for _ in 0..BLUR_TIERS {
            match make(size.w, size.h) {
                Ok(t) => tiers.push(t),
                Err(err) => {
                    warn!(error = %err, "blur: tier buffer alloc failed");
                    return false;
                }
            }
        }
        self.blur_scratch.insert(
            idx,
            BlurScratch {
                size,
                scene,
                levels,
                tiers,
            },
        );
        true
    }

    /// Mark a freshly-mapped toplevel so it plays an open animation the
    /// next time it appears in a frame's placements (not on a later
    /// workspace switch that merely surfaces it again).
    pub fn mark_open(&mut self, surface: &WlSurface) {
        self.pending_open.insert(surface.id());
    }

    /// Set (`Some`) or clear (`None`) the window being interactively
    /// moved/resized, which draws 1:1 instead of animating its rect.
    pub fn set_no_anim_move(&mut self, surface: Option<&WlSurface>) {
        self.no_anim_move = surface.map(Resource::id);
    }

    /// Begin a close animation for a toplevel that's being destroyed.
    /// Snapshots the window's current content into a texture (while the
    /// surface still has its last buffer) and registers a fading,
    /// shrinking ghost where it last sat. A no-op (instant close) if the
    /// close animation is disabled, the window isn't tracked, or its
    /// buffer is already gone. Must run *before* the window leaves the
    /// layout so its last drawn rect is still known.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "physical pixel sizes derived from output-bounded rects are small non-negative values"
    )]
    pub fn start_close(&mut self, surface: &WlSurface) {
        let cfg = self.animations.clone();
        if !(cfg.enabled && cfg.window_close.enabled) {
            return;
        }
        let id = surface.id();
        let Some(entry) = self.win_anims.remove(&id) else {
            return;
        };
        let cell = entry.displayed;
        // Content rect = cell minus the border ring (Normal windows). We
        // don't track fill per window; a borderless maximized/fullscreen
        // window closing would inset by a few px, which is invisible.
        let bw = self.border.width.max(0);
        let inner = Rectangle::<i32, Physical>::new(
            Point::from((cell.loc.x + bw, cell.loc.y + bw)),
            Size::from(((cell.size.w - 2 * bw).max(1), (cell.size.h - 2 * bw).max(1))),
        );
        let center = Point::<i32, Physical>::from((
            inner.loc.x + inner.size.w / 2,
            inner.loc.y + inner.size.h / 2,
        ));
        let scale = self.output_at(center).map_or(1.0, |o| o.scale);

        // Build the surface's elements with its content origin at the
        // texture's (0, 0) (shift past the CSD shadow margin).
        let (geo_x, geo_y) = window_geometry_offset(surface);
        let origin = Point::<i32, Physical>::from((
            -scale_f(f64::from(geo_x), scale),
            -scale_f(f64::from(geo_y), scale),
        ));
        let elements: Vec<WaylandSurfaceRenderElement<GlesRenderer>> =
            render_elements_from_surface_tree(
                &mut self.gles,
                surface,
                origin,
                scale,
                1.0_f32,
                Kind::Unspecified,
            );
        if elements.is_empty() {
            return; // no buffer left to snapshot — close instantly
        }

        let tex_size = Size::<i32, smithay::utils::Buffer>::from((
            scale_f(f64::from(inner.size.w), scale).max(1),
            scale_f(f64::from(inner.size.h), scale).max(1),
        ));
        let mut texture = match self.gles.create_buffer(Fourcc::Abgr8888, tex_size) {
            Ok(t) => t,
            Err(err) => {
                warn!(error = %err, "close snapshot: create_buffer failed");
                return;
            }
        };
        let phys = Size::<i32, Physical>::from((tex_size.w, tex_size.h));
        let full = [Rectangle::<i32, Physical>::from_size(phys)];
        let mut target = match self.gles.bind(&mut texture) {
            Ok(t) => t,
            Err(err) => {
                warn!(error = %err, "close snapshot: bind failed");
                return;
            }
        };
        match self.gles.render(&mut target, phys, Transform::Normal) {
            Ok(mut frame) => {
                let _ = frame.clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &full);
                if let Err(err) =
                    draw_render_elements::<GlesRenderer, _, _>(&mut frame, scale, &elements, &full)
                {
                    warn!(error = %err, "close snapshot: draw failed");
                    return;
                }
                if let Err(err) = frame.finish() {
                    warn!(error = %err, "close snapshot: finish failed");
                    return;
                }
            }
            Err(err) => {
                warn!(error = %err, "close snapshot: render failed");
                return;
            }
        }
        drop(target);

        let now = self.start.elapsed().as_secs_f64();
        self.closing.push(ClosingWindow {
            texture,
            rect: inner,
            anim: Animation::start(
                now,
                cfg.window_close.duration_secs(),
                cfg.window_close.curve,
            ),
        });
    }

    /// Tonemap an HDR output's linear-BT.2020 scene to an 8-bit sRGB scratch
    /// buffer and service `captures` from it. GLES can't read the fp16 scanout
    /// back as an 8-bit format (and it'd be linear BT.2020 anyway), so HDR
    /// captures go through [`SCREENSHOT_TONEMAP_SHADER`] first; the result is
    /// SDR-correct ("looks like SDR"). The scratch buffer is cached per output
    /// (reused across frames). Any GL failure fails just the captures, not the
    /// frame.
    fn capture_tonemapped(
        &mut self,
        output_name: &str,
        mode_size: Size<i32, Physical>,
        reference_white: f32,
        captures: &[CaptureSpec],
    ) -> Vec<CaptureOutcome> {
        let failed = || captures.iter().map(|_| CaptureOutcome::Failed).collect();
        // Ensure a cached 8-bit scratch sized to the output (reused across
        // frames so continuous screencopy doesn't re-alloc every frame).
        let mode_w = u32::try_from(mode_size.w).unwrap_or(0);
        let mode_h = u32::try_from(mode_size.h).unwrap_or(0);
        let needs_alloc = match self.sdr_capture.get(output_name) {
            Some(tex) => tex.width() != mode_w || tex.height() != mode_h,
            None => true,
        };
        if needs_alloc {
            let buf_size = Size::<i32, smithay::utils::Buffer>::from((mode_size.w, mode_size.h));
            match self.gles.create_buffer(Fourcc::Abgr8888, buf_size) {
                Ok(b) => {
                    self.sdr_capture.insert(output_name.to_string(), b);
                }
                Err(err) => {
                    warn!(error = %err, output = %output_name, "screenshot: tonemap buffer alloc failed");
                    self.sdr_capture.remove(output_name);
                    return failed();
                }
            }
        }
        let tonemap = self.screenshot_tonemap_shader.clone();
        // `GlesTexture` is `Arc`-backed, so clone the scene handle out to drop
        // the immutable `hdr_scene` borrow before re-borrowing `self.gles`.
        let scene_tex = self
            .hdr_scene
            .get(output_name)
            .expect("HDR scene buffer present")
            .clone();
        // Disjoint field borrows: `&mut self.sdr_capture[..]` and `&mut
        // self.gles` are separate fields, so binding the scratch is fine.
        let mut sdr = self.sdr_capture.remove(output_name).expect("just ensured");
        let mut target = match self.gles.bind(&mut sdr) {
            Ok(t) => t,
            Err(err) => {
                warn!(error = %err, output = %output_name, "screenshot: bind tonemap buffer failed");
                // `sdr` drops here (cache entry stays removed → re-alloc next frame).
                return failed();
            }
        };
        let render = (|| -> Result<()> {
            let mut frame = self
                .gles
                .render(&mut target, mode_size, Transform::Normal)
                .context("screenshot tonemap render")?;
            let dst = Rectangle::from_size(mode_size);
            let src = Rectangle::<f64, smithay::utils::Buffer>::from_size(Size::from((
                f64::from(mode_size.w),
                f64::from(mode_size.h),
            )));
            frame
                .render_texture_from_to(
                    &scene_tex,
                    src,
                    dst,
                    &[dst],
                    &[dst],
                    Transform::Normal,
                    1.0,
                    Some(&tonemap),
                    &[Uniform::new("reference_white", reference_white)],
                )
                .context("screenshot tonemap pass")?;
            // Same-context sequential GL: the copy_framebuffer read-back below
            // is ordered after this draw, so the sync fence needn't be awaited.
            let _ = frame.finish().context("screenshot tonemap finish")?;
            Ok(())
        })();
        if let Err(err) = render {
            warn!(error = %err, output = %output_name, "screenshot: tonemap failed");
            return failed();
        }
        // The scratch buffer now holds an upright 8-bit sRGB copy — service
        // every capture from it exactly like the SDR scanout path (both the
        // CPU read-back and the zero-copy dmabuf blit, so OBS et al. record
        // SDR-correct frames instead of the dark linear scene).
        let results: Vec<CaptureOutcome> = captures
            .iter()
            .map(|spec| match &spec.target {
                CaptureTarget::Shm => capture_shm(&mut self.gles, &target, spec, output_name),
                CaptureTarget::Dmabuf(client) => {
                    capture_dmabuf(&mut self.gles, &target, client, spec, output_name)
                }
            })
            .collect();
        // `target` borrows `sdr`; drop it before caching the buffer back.
        drop(target);
        self.sdr_capture.insert(output_name.to_string(), sdr);
        results
    }

    /// Render `surface`'s current surface tree into an offscreen and read it
    /// back as premultiplied-RGBA8 bytes — an on-demand per-window thumbnail
    /// for the IPC. Independent of any output: a window on another workspace
    /// or screen still captures its last-committed content, in isolation (no
    /// other windows, no cursor). The longest side is capped at `max`
    /// (downscaled, never upscaled). Returns `(width, height, rgba)` —
    /// premultiplied RGBA8, bottom-up (the encoder flips it upright).
    pub fn capture_window(
        &mut self,
        surface: &WlSurface,
        max: i32,
    ) -> Result<(i32, i32, Vec<u8>)> {
        // Visible window rect (excludes the CSD shadow); fall back to the full
        // surface-tree bbox when the client set no window geometry (e.g. some
        // XWayland surfaces).
        let (gx, gy, gw, gh) = window_geometry_size(surface).map_or_else(
            || {
                let bb = smithay::desktop::utils::bbox_from_surface_tree(surface, (0, 0));
                (bb.loc.x, bb.loc.y, bb.size.w, bb.size.h)
            },
            |(w, h)| {
                let (ox, oy) = window_geometry_offset(surface);
                (ox, oy, w, h)
            },
        );
        let (gw, gh) = (gw.max(1), gh.max(1));
        let cap = if max > 0 { max } else { 512 };
        // Downscale only — never enlarge a small window.
        let scale = (f64::from(cap) / f64::from(gw.max(gh))).min(1.0);

        // Anchor the window geometry's top-left at the texture origin.
        let origin = Point::<i32, Physical>::from((
            -scale_f(f64::from(gx), scale),
            -scale_f(f64::from(gy), scale),
        ));
        let elements: Vec<WaylandSurfaceRenderElement<GlesRenderer>> =
            render_elements_from_surface_tree(
                &mut self.gles,
                surface,
                origin,
                scale,
                1.0_f32,
                Kind::Unspecified,
            );
        if elements.is_empty() {
            anyhow::bail!("window has no current buffer to capture");
        }

        let tw = scale_f(f64::from(gw), scale).max(1);
        let th = scale_f(f64::from(gh), scale).max(1);
        let tex_size = Size::<i32, smithay::utils::Buffer>::from((tw, th));
        let phys = Size::<i32, Physical>::from((tw, th));
        let full = [Rectangle::<i32, Physical>::from_size(phys)];

        let mut texture: GlesTexture = self
            .gles
            .create_buffer(Fourcc::Abgr8888, tex_size)
            .context("capture_window: create_buffer")?;
        let mut target = self.gles.bind(&mut texture).context("capture_window: bind")?;
        {
            let mut frame = self
                .gles
                .render(&mut target, phys, Transform::Normal)
                .context("capture_window: render")?;
            frame
                .clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &full)
                .context("capture_window: clear")?;
            draw_render_elements::<GlesRenderer, _, _>(&mut frame, scale, &elements, &full)
                .context("capture_window: draw")?;
            // Same-context sequential GL: the copy_framebuffer below is ordered
            // after these writes, so the fence can be dropped.
            let _ = frame.finish().context("capture_window: finish")?;
        }

        let region = Rectangle::<i32, smithay::utils::Buffer>::from_size(tex_size);
        let mapping = self
            .gles
            .copy_framebuffer(&target, region, Fourcc::Abgr8888)
            .context("capture_window: copy_framebuffer")?;
        let bytes = self
            .gles
            .map_texture(&mapping)
            .context("capture_window: map_texture")?
            .to_vec();
        drop(target);
        Ok((tw, th, bytes))
    }

    /// Advance per-window animations against `now` (seconds on the shared
    /// clock) and return the on-screen rect + opacity to draw each
    /// placement at, in placement order. Position/size interpolate toward
    /// the layout's target; a just-mapped window fades + scales in.
    fn animate_placements(&mut self, now: f64, placements: &[Placement]) -> Vec<WinDraw> {
        let cfg = self.animations.clone();
        let move_enabled = cfg.enabled && cfg.window_move.enabled;
        let open_enabled = cfg.enabled && cfg.window_open.enabled;
        let no_anim_move = self.no_anim_move.clone();

        let mut draws = Vec::with_capacity(placements.len());
        for p in placements {
            let id = p.surface.id();
            let target = p.cell_rect;
            // The interactively dragged window tracks the cursor 1:1.
            let snap = no_anim_move.as_ref() == Some(&id);
            let entry = self.win_anims.entry(id.clone()).or_insert_with(|| WindowAnim {
                surface: p.surface.clone(),
                target,
                displayed: target,
                move_from: target,
                move_anim: None,
                open_anim: None,
            });

            // Target moved (reflow / interactive move / fullscreen
            // toggle): start or retarget a move animation from where the
            // window is being drawn right now, so retargets stay smooth.
            if target != entry.target {
                entry.move_anim = (move_enabled && !snap).then(|| {
                    entry.move_from = entry.displayed;
                    Animation::start(now, cfg.window_move.duration_secs(), cfg.window_move.curve)
                });
                entry.target = target;
            }

            // A just-mapped window starts opening the first frame it's
            // here. Consume the mark regardless so a disabled open
            // animation doesn't leave it pending forever.
            if self.pending_open.remove(&id) && open_enabled {
                entry.open_anim = Some(Animation::start(
                    now,
                    cfg.window_open.duration_secs(),
                    cfg.window_open.curve,
                ));
            }

            // Position/size: interpolate displayed → target.
            if let Some(a) = entry.move_anim {
                entry.displayed = lerp_rect(entry.move_from, entry.target, a.value(now));
                if a.done(now) {
                    entry.move_anim = None;
                    entry.displayed = entry.target;
                }
            } else {
                entry.displayed = entry.target;
            }

            // Open: fade + scale-about-centre layered on the displayed
            // rect.
            let (mut effective, mut alpha) = (entry.displayed, 1.0_f32);
            if let Some(a) = entry.open_anim {
                let v = a.value(now);
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "eased progress is in [0,1]; the f32 cast is exact enough for an opacity"
                )]
                let a32 = v as f32;
                alpha = a32;
                effective = scale_rect_about_center(entry.displayed, lerp(OPEN_SCALE_FROM, 1.0, v));
                if a.done(now) {
                    entry.open_anim = None;
                }
            }
            // Workspace slide: a uniform vertical offset applied *after*
            // the per-window animation (so it doesn't perturb move/open),
            // translating the whole workspace during a switch.
            effective.loc.y += p.slide_dy;
            draws.push(WinDraw { effective, alpha });
        }

        // Drop tracking for windows whose surface has died.
        self.win_anims.retain(|_, w| w.surface.alive());
        // Drop finished close-out ghosts (frees their snapshot textures).
        self.closing.retain(|c| !c.anim.done(now));
        draws
    }

    /// GPU buffer (dmabuf) formats this renderer can import as
    /// textures — advertised via `zwp_linux_dmabuf_v1` so clients
    /// (and Xwayland) can hand us
    /// GPU-rendered content. Without this, GPU-composited apps (e.g.
    /// the Steam client) commit dmabuf buffers we can't sample and
    /// render blank.
    pub fn dmabuf_formats(&self) -> Vec<Format> {
        self.gles.dmabuf_formats().into_iter().collect()
    }

    /// Whether the renderer can bind a dmabuf of `format` as a *render
    /// target* (the subset of formats we can draw/blit *into*, which is
    /// smaller than the texture-import set). Screencopy's GPU path
    /// blits into the client's dmabuf, so it must be render-capable;
    /// otherwise we fall back to the shm path. NVIDIA in particular has
    /// a narrower render set than texture set.
    pub fn can_render_to(&self, format: Format) -> bool {
        self.gles
            .egl_context()
            .dmabuf_render_formats()
            .contains(&format)
    }

    /// The render `DrmNode` backing our EGL context, used as the
    /// dmabuf-feedback *main device* so clients (and Xwayland) know
    /// which GPU to allocate dmabufs on. `None` if the EGL device
    /// can't be resolved (then we advertise a v3 dmabuf global, which
    /// modern Xwayland's glamor won't use — GPU X apps stay blank).
    pub fn render_drm_node(&self) -> Option<DrmNode> {
        EGLDevice::device_for_display(self.gles.egl_context().display())
            .ok()?
            .try_get_render_node()
            .ok()
            .flatten()
    }

    /// Try to import a client's dmabuf into the GLES renderer,
    /// returning whether it succeeded. Used by the dmabuf protocol
    /// handler to accept or reject a buffer up front; the texture is
    /// cached internally so the per-frame render reuses it.
    pub fn import_dmabuf(&mut self, dmabuf: &Dmabuf) -> bool {
        self.gles.import_dmabuf(dmabuf, None).is_ok()
    }

    /// Per-output `(name, mode_size_physical, compositor_size,
    /// position_compositor, scale)`. Used by the Wayland frontend
    /// to advertise `wl_output` globals to clients (one per DRM
    /// output) and to seed the fractional-scale state.
    pub fn output_descriptors(&self) -> Vec<OutputDescriptor> {
        self.outputs
            .iter()
            .map(|o| OutputDescriptor {
                name: o.name.clone(),
                mode_size: o.mode_size,
                refresh_mhz: o.refresh_mhz,
                compositor_position: o.compositor_position,
                compositor_size: o.compositor_size,
                scale: o.scale,
            })
            .collect()
    }

    /// Scale of the configured primary output. The Wayland frontend
    /// sends this as the preferred fractional scale to every surface
    /// (since the layout is single-output for now — multi-output
    /// per-surface scale tracking is a later milestone).
    pub fn primary_scale(&self) -> f64 {
        self.outputs.get(self.primary_idx).map_or(1.0, |o| o.scale)
    }

    /// Settle one output's Variable Refresh Rate state for the frame about
    /// to be queued, per its configured [`VrrMode`].
    ///
    /// Idempotent and cheap to call every vblank: smithay's `use_vrr`
    /// early-returns when the pending adaptive-sync state already matches,
    /// so we only do work on an actual transition. Outputs whose connector
    /// doesn't advertise adaptive-sync are left untouched.
    fn apply_vrr(&self, idx: usize, placements: &[Placement]) {
        let output = &self.outputs[idx];
        if output.vrr_support == VrrSupport::NotSupported {
            return;
        }
        let desired = match output.vrr_mode {
            VrrMode::Off => false,
            VrrMode::Always => true,
            // Auto: enabled only while a fullscreen/maximized window fills
            // this output.
            VrrMode::Auto => self.output_has_fill_window(idx, placements),
        };
        if output.surface.vrr_enabled() == desired {
            return;
        }
        let support = output.vrr_support;
        match output.surface.use_vrr(desired) {
            Ok(()) => info!(
                output = %output.name,
                enabled = desired,
                ?support,
                "adaptive-sync (VRR) state changed"
            ),
            Err(err) => warn!(
                output = %output.name,
                enabled = desired,
                error = %err,
                "could not set adaptive-sync (VRR)"
            ),
        }
    }

    /// Render one output's frame: wallpaper, then per window in
    /// bottom-up draw order render its border + surface, then the
    /// cursor sprite on top if its hotspot falls in this output.
    /// Sends `wl_callback.done` on each surface after the buffer is
    /// queued so clients know they can draw the next frame.
    #[allow(
        clippy::too_many_lines,
        reason = "this is the per-output render loop — wallpaper, per-window border+surface+rounded-mask, cursor, queue, frame callbacks. Splitting any one piece out would require threading the dmabuf/frame borrow through another method, which adds more friction than length removes."
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "per-frame render inputs (placements, layers, popups, captures, HDR set) are all distinct; bundling them into a struct just moves the noise to the call site"
    )]
    fn render_output(
        &mut self,
        idx: usize,
        placements: &[Placement],
        layers: &[LayerPlacement],
        popups: &[PopupPlacement],
        hide_cursor: bool,
        captures: &[CaptureSpec],
        hdr_surface_ids: &HashSet<ObjectId>,
        compose_cursor: bool,
        present_output: Option<&Output>,
    ) -> Result<(Vec<CaptureOutcome>, bool)> {
        // Is the cursor already on the hardware plane for this output? If so,
        // skip compositing it (the plane scans it out) — unless this frame
        // must bake the cursor into the framebuffer for a capture.
        let hw_cursor_active = self.hw_cursor_active(self.outputs[idx].crtc);
        // Upload the latest media-wallpaper frame (if the decode thread has
        // produced one) before snapshotting the drawable below, so animated
        // wallpapers advance each vblank.
        self.refresh_wallpaper();

        // Pull everything we need before the mutable borrows on
        // `self.outputs[idx].surface` / `self.gles` kick in. All
        // *_phys helpers below take pre-scaled physical pixel
        // values; this function is the one place compositor →
        // physical conversion happens.
        let cursor_abs_x = self.cursor_x;
        let cursor_abs_y = self.cursor_y;
        let wallpaper = self.wallpaper.clone();
        // Cheap Arc-backed clone of just the drawable wallpaper frame (if
        // any), so the backdrop closures can paint it without borrowing
        // `self` (the decode-thread handle stays on the renderer).
        let wallpaper_media = self.wallpaper_media.as_ref().map(|m| m.draw.clone());
        let border = self.border.clone();
        let round_tex_shader = self.round_tex_shader.clone();
        let round_blur_shader = self.round_blur_shader.clone();
        let mask_blur_shader = self.mask_blur_shader.clone();
        let hdr_encode_shader = self.hdr_encode_shader.clone();
        let sdr_decode_shader = self.sdr_decode_shader.clone();
        let sdr_to_pq_shader = self.sdr_to_pq_shader.clone();
        let hdr_decode_shader = self.hdr_decode_shader.clone();
        let round_tex_shader_hdr = self.round_tex_shader_hdr.clone();
        let round_tex_shader_linear = self.round_tex_shader_linear.clone();
        let round_blur_shader_hdr = self.round_blur_shader_hdr.clone();
        let mask_blur_shader_hdr = self.mask_blur_shader_hdr.clone();
        let cursor_size = self.cursor_size;
        let window_opacity = self.decoration.window_opacity;
        // The effective cursor this frame: a compositor override (grab /
        // screenshot) wins over the client's request. For a Named cursor
        // we resolve (and lazily upload) its themed sprite now, while we
        // still hold `&mut self`. A client Surface cursor is drawn as a
        // surface tree further down (it needs `cursor_in_bounds` first).
        let cursor_status = self
            .cursor_override
            .clone()
            .unwrap_or_else(|| self.cursor_status.clone());
        let cursor_sprite = match &cursor_status {
            CursorImageStatus::Named(icon) => self.named_cursor_sprite(*icon),
            // Hidden → no sprite; Surface → drawn separately below.
            _ => None,
        };
        let screenshot_overlay = self.screenshot_overlay;
        let dnd_icon = self.dnd_icon.clone();
        // Advance window animations and resolve each placement's on-screen
        // rect + opacity for this frame (before the immutable `outputs`
        // borrow below). `now` is seconds on the shared render clock.
        let now = self.start.elapsed().as_secs_f64();
        let win_draws = self.animate_placements(now, placements);
        let output = &self.outputs[idx];
        let mode_size = output.mode_size;
        let compositor_position = output.compositor_position;
        let compositor_size = output.compositor_size;
        let scale = output.scale;
        let output_name = output.name.clone();
        // HDR outputs composite into an offscreen, then a PQ-encode pass
        // writes the 10-bit scanout (see below). SDR is unaffected.
        // `hdr` may be downgraded to false below if the fp16 scene buffer
        // can't be allocated (then this output renders as SDR for the frame).
        let mut hdr = output.hdr;
        let hdr_reference_white = output.hdr_reference_white;
        let hdr_saturation = output.hdr_saturation;
        #[allow(
            clippy::cast_precision_loss,
            reason = "reference white is a small cd/m² value, exact in f32"
        )]
        let ref_white_f32 = hdr_reference_white as f32;

        // Solo-fullscreen scene test, shared by the two fast paths below
        // (direct scanout, single-pass HDR) and the wallpaper skip.
        let out_rect = Rectangle::new(compositor_position, compositor_size);
        let solo = self.solo_fullscreen_scene(
            idx,
            &win_draws,
            placements,
            layers,
            popups,
            hide_cursor,
            captures,
            compose_cursor,
        );

        // ── Direct-scanout fast path ──────────────────────────────────
        // A single settled fullscreen opaque client whose colour mode
        // matches the output: scan its buffer straight to the primary
        // plane, skipping ALL compositing (≈ zero GPU for this output).
        // Anything that needs compositing — overlays, popups, animations,
        // captures, a non-1:1 buffer, or a buffer the plane rejects —
        // falls through to the composite path below.
        if let Some(direct) = self.direct_scanout_inputs(idx, solo, placements, hdr_surface_ids) {
            // VRR must settle before the flip (it may promote the flip to a
            // modeset); harmlessly re-applied by the composite path on a miss.
            self.apply_vrr(idx, placements);
            match self.outputs[idx].surface.try_queue_external(
                direct.buffer,
                &direct.dmabuf,
                direct.use_opaque,
            ) {
                Ok(true) => {
                    debug!(output = %output_name, "frame direct-scanned to primary plane (no compositing)");
                    self.send_output_frame_callbacks(placements, layers, popups, out_rect);
                    // Zero-copy presentation: the client's own buffer is on the
                    // plane, so flag ZeroCopy. Fired on this flip's vblank.
                    if let Some(out) = present_output {
                        self.outputs[idx].pending_feedback = Some(collect_presentation_feedback(
                            out, placements, layers, popups, out_rect, true,
                        ));
                    }
                    // No transient state is active (eligibility required it),
                    // so the output parks until the client's next commit.
                    return Ok((Vec::new(), false));
                }
                Ok(false) => {} // not scannable this frame → composite below
                Err(err) => {
                    warn!(output = %output_name, error = %err, "direct scanout failed; compositing");
                }
            }
        }

        // The solo window, when it's additionally a *provably opaque* single
        // surface: with every output pixel guaranteed overwritten opaquely,
        // the wallpaper/base pass underneath is pure waste — skip it (any
        // colour mode; this also trims the composite for SDR outputs when a
        // game's buffer isn't plane-scannable that frame).
        let solo_opaque = solo.filter(|&i| {
            let surface = &placements[i].surface;
            surface_is_single_node(surface) && surface_provably_opaque(surface)
        });

        // ── Single-pass HDR fast path ─────────────────────────────────
        // An SDR game filling an HDR output can't direct-scan (its pixels
        // need the PQ encode), but it doesn't need the generic HDR pipeline
        // either — compositing into the fp16 linear scene and PQ-encoding it
        // in a second pass costs two extra full-output passes per game
        // frame. Instead render this frame like an SDR output (straight into
        // the scanout dmabuf, no fp16 scene, no encode pass) with the fused
        // SDR→PQ program as the frame default, which colour-matches the
        // generic pipeline exactly (same decode, saturation, and OETF, one
        // fragment instead of two passes).
        let single_pass_hdr = hdr
            && solo_opaque.is_some_and(|i| !hdr_surface_ids.contains(&placements[i].surface.id()));
        if single_pass_hdr {
            debug!(output = %output_name, "single-pass HDR: fused SDR→PQ straight to scanout");
            hdr = false;
        }

        // Ensure the fp16 linear scene buffer for HDR outputs *before* the
        // draw closures capture `hdr` (they read it for shader selection).
        // 8-bit can't hold HDR headroom; if the driver rejects fp16 as a
        // render target, downgrade this frame to SDR (render straight to the
        // dmabuf) rather than black-screening — the connector still carries
        // the HDR signal, so content just looks washed until alloc succeeds.
        if hdr {
            let mode_w = u32::try_from(mode_size.w).unwrap_or(0);
            let mode_h = u32::try_from(mode_size.h).unwrap_or(0);
            let needs_alloc = match self.hdr_scene.get(&output_name) {
                Some(tex) => tex.width() != mode_w || tex.height() != mode_h,
                None => true,
            };
            if needs_alloc {
                match self.gles.create_buffer(
                    Fourcc::Abgr16161616f,
                    Size::<i32, smithay::utils::Buffer>::from((mode_size.w, mode_size.h)),
                ) {
                    Ok(scene) => {
                        self.hdr_scene.insert(output_name.clone(), scene);
                    }
                    Err(err) => {
                        warn!(output = %output_name, error = %err,
                            "fp16 HDR scene buffer alloc failed; rendering this output as SDR");
                        self.hdr_scene.remove(&output_name);
                        hdr = false;
                    }
                }
            }
        }

        // Frozen backdrop for this output (freeze-mode screenshot). Cheap
        // Arc-backed clone out before the `self.gles` frame borrow.
        let freeze_texture = self.freeze_textures.get(&output_name).cloned();

        // The previous frame's flip is acked separately, on its vblank
        // (see `Renderer::frame_submitted`), so the on-demand driver can
        // ack a completed flip without being forced to render another.
        let (mut dmabuf, _age) = self.outputs[idx]
            .surface
            .next_buffer()
            .with_context(|| format!("next_buffer failed for {output_name}"))?;

        // Cursor in this output's local compositor space; convert to
        // physical for drawing. Bounds check uses the compositor
        // size so cursors that fall outside the visible area of
        // this output are skipped (cursor may be on a different
        // output in a multi-display setup).
        let cursor_local_x = cursor_abs_x - f64::from(compositor_position.x);
        let cursor_local_y = cursor_abs_y - f64::from(compositor_position.y);
        let cursor_in_bounds = cursor_local_x >= 0.0
            && cursor_local_y >= 0.0
            && cursor_local_x < f64::from(compositor_size.w)
            && cursor_local_y < f64::from(compositor_size.h);

        // Build client-surface render elements *before* binding the
        // dmabuf. `render_elements_from_surface_tree` uses the
        // renderer to import each surface's buffer as a GLES texture
        // (via the `ImportAll` trait GlesRenderer impls); that has
        // to happen while no `Frame` is alive. The resulting Vec
        // owns its `TextureId`s, so it's free to outlive the
        // renderer borrow and be drawn during the frame below.
        //
        // Per placement: the surface itself draws inside the cell,
        // shrunk by `border` (in compositor px) on every side; the
        // resulting position is multiplied by `scale` so the
        // texture lands at the right physical pixel on the
        // framebuffer. We also pass `scale` to smithay so it
        // composes the client buffer at the right size for
        // fractional displays.
        let bw_comp = border.width.max(0);
        let radius_comp = border.rounded_corners.max(0);
        // A Normal window with a border and/or rounded corners is composited
        // through an offscreen texture + the rounded mask shader (so its
        // corners are genuinely transparent). Without either it's a plain
        // rectangle drawn straight to the frame, like fullscreen/maximized.
        let decorated = radius_comp > 0 || bw_comp > 0;
        #[allow(
            clippy::type_complexity,
            reason = "one frame's worth of per-window, rescale-wrapped surface elements"
        )]
        let grouped: Vec<Vec<RescaleRenderElement<WaylandSurfaceRenderElement<GlesRenderer>>>> =
            placements
                .iter()
                .zip(win_draws.iter())
                .map(|(p, wd)| {
                    // CSD clients pad their buffer with an invisible
                    // drop-shadow margin and report the real window rect
                    // via xdg_surface.set_window_geometry. Shift the
                    // buffer up-left by that margin so the *visible*
                    // content (not the buffer's padded corner) lands at
                    // the cell origin; the shadow then falls outside the
                    // cell. Maximized/fullscreen windows have no border
                    // and fill their output flush at the cell origin — no
                    // inset and no CSD shadow offset.
                    let (geo_x, geo_y) = if p.fill == FillMode::Normal {
                        window_geometry_offset(&p.surface)
                    } else {
                        (0, 0)
                    };
                    let bw_p = if p.fill == FillMode::Normal { bw_comp } else { 0 };
                    // A decorated Normal window is rendered into a *cell-sized
                    // offscreen* (origin (0,0)) and masked in the composite, so
                    // here its surface fills the WHOLE cell — the opaque border
                    // ring overlays the outer edge, which keeps the surface
                    // opaque across the border boundary (no transparent seam).
                    // Everything else (fullscreen/maximized/undecorated) draws
                    // straight to the frame at its output-local cell position,
                    // inset by the border. HDR surfaces use this offscreen path
                    // too — the offscreen is fp16 and the surface is decoded
                    // into it (see Phase A), so decoration works in HDR.
                    let offscreen = p.fill == FillMode::Normal && decorated;
                    // Draw the window into its *animated* rect
                    // (`wd.effective`), scaling the surface's actual
                    // content to fill it. `render_elements_from_surface_tree`'s
                    // scale only positions subsurfaces — the drawn size
                    // comes from the *draw* scale — so the surface is built
                    // at the output scale (content origin on `origin`) and
                    // then wrapped in a RescaleRenderElement that scales the
                    // whole tree about that origin. The denominator is the
                    // client's current geometry size (its real size right
                    // now), so a resize looks correct even while the client
                    // is a frame behind its configure; when settled,
                    // `effective == cell_rect` and the scale is 1 (crisp).
                    let eff = wd.effective;
                    let (content_w, content_h) =
                        window_geometry_size(&p.surface).unwrap_or((
                            p.cell_rect.size.w - 2 * bw_p,
                            p.cell_rect.size.h - 2 * bw_p,
                        ));
                    // Offscreen: fill the cell, anchored at the cell origin
                    // (0,0). Direct: inset by the border, anchored at the
                    // output-local cell position.
                    let (target_w, target_h, anchor_x, anchor_y) = if offscreen {
                        (
                            f64::from(eff.size.w.max(1)),
                            f64::from(eff.size.h.max(1)),
                            0.0,
                            0.0,
                        )
                    } else {
                        (
                            f64::from((eff.size.w - 2 * bw_p).max(1)),
                            f64::from((eff.size.h - 2 * bw_p).max(1)),
                            f64::from(eff.loc.x + bw_p - compositor_position.x),
                            f64::from(eff.loc.y + bw_p - compositor_position.y),
                        )
                    };
                    let csx = target_w / f64::from(content_w.max(1));
                    let csy = target_h / f64::from(content_h.max(1));
                    let origin = Point::<i32, Physical>::from((
                        scale_f(anchor_x, scale),
                        scale_f(anchor_y, scale),
                    ));
                    // Build at output scale so the content's geometry origin
                    // lands on `origin`; the rescale below shrinks/grows it.
                    let location = Point::<i32, Physical>::from((
                        origin.x - scale_f(f64::from(geo_x), scale),
                        origin.y - scale_f(f64::from(geo_y), scale),
                    ));
                    // The window's configurable opacity (Normal only) plus its
                    // animation alpha. For the offscreen path this is applied
                    // in the *composite* (the shader's `alpha`), so the surface
                    // itself is rendered fully opaque (1.0) — keeping the
                    // client's own per-pixel translucency intact — and we don't
                    // double-apply.
                    let alpha = if offscreen {
                        1.0
                    } else if p.fill == FillMode::Normal {
                        wd.alpha * window_opacity
                    } else {
                        wd.alpha
                    };
                    let elements: Vec<WaylandSurfaceRenderElement<GlesRenderer>> =
                        render_elements_from_surface_tree(
                            &mut self.gles,
                            &p.surface,
                            location,
                            scale,
                            alpha,
                            Kind::Unspecified,
                        );
                    let content_scale = Scale::from((csx, csy));
                    elements
                        .into_iter()
                        .map(|e| RescaleRenderElement::from_element(e, origin, content_scale))
                        .collect()
                })
                .collect();

        // Layer surfaces: pre-import textures while we still
        // hold `&mut self.gles` outside the frame scope, like we
        // do for window placements. Each entry pairs the layer
        // bucket with the imported elements so we can paint them
        // in the correct z-order during the frame block below.
        let layer_groups: Vec<(LayerBucket, Vec<WaylandSurfaceRenderElement<GlesRenderer>>)> =
            layers
                .iter()
                .map(|l| {
                    let local_phys = Point::<i32, Physical>::from((
                        scale_i(l.rect.loc.x - compositor_position.x, scale),
                        scale_i(l.rect.loc.y - compositor_position.y, scale),
                    ));
                    let elements = render_elements_from_surface_tree(
                        &mut self.gles,
                        &l.surface,
                        local_phys,
                        scale,
                        1.0_f32,
                        Kind::Unspecified,
                    );
                    (l.layer, elements)
                })
                .collect();

        // Each layer surface's imported texture (populated by the
        // `render_elements_from_surface_tree` import above), used to
        // alpha-mask that layer's backdrop blur to the shape the client
        // actually drew. `None` for a layer with no committed buffer.
        let ctx_id = self.gles.context_id();
        let layer_masks: Vec<Option<GlesTexture>> = layers
            .iter()
            .map(|l| {
                with_renderer_surface_state(&l.surface, |state| {
                    state.texture::<GlesTexture>(ctx_id.clone()).cloned()
                })
                .flatten()
            })
            .collect();

        // Popups (menus/submenus): pre-import like layers. Each
        // `buffer_origin` is already absolute compositor px with the
        // popup's own geometry offset folded in, so this is just the
        // local-space + scale conversion (no border, no extra
        // geometry subtraction). Snapshot order is parent-before-child
        // so nested submenus draw on top in iteration order.
        let popup_groups: Vec<Vec<WaylandSurfaceRenderElement<GlesRenderer>>> = popups
            .iter()
            .map(|pp| {
                let local_phys = Point::<i32, Physical>::from((
                    scale_i(pp.buffer_origin.x - compositor_position.x, scale),
                    scale_i(pp.buffer_origin.y - compositor_position.y, scale),
                ));
                render_elements_from_surface_tree(
                    &mut self.gles,
                    &pp.surface,
                    local_phys,
                    scale,
                    1.0_f32,
                    Kind::Unspecified,
                )
            })
            .collect();

        // Drag-and-drop icon: composite the drag surface at the cursor
        // (only on the output the cursor is on). Pre-imported here while we
        // still hold `&mut self.gles`, like the surface groups above.
        let dnd_icon_elements: Vec<WaylandSurfaceRenderElement<GlesRenderer>> =
            match (dnd_icon.as_ref(), cursor_in_bounds) {
                (Some(icon), true) => {
                    let local_phys = Point::<i32, Physical>::from((
                        scale_f(cursor_local_x, scale),
                        scale_f(cursor_local_y, scale),
                    ));
                    render_elements_from_surface_tree(
                        &mut self.gles,
                        icon,
                        local_phys,
                        scale,
                        1.0_f32,
                        Kind::Unspecified,
                    )
                }
                _ => Vec::new(),
            };

        // Client surface cursor (`wl_pointer.set_cursor` with a surface;
        // this is how native and Xwayland games supply their own
        // pointer). Positioned so the surface's hotspot — stored in the
        // cursor-image role data — sits on the pointer. Pre-imported here
        // while we still hold `&mut self.gles`, like the DnD icon above.
        let cursor_surface_elements: Vec<WaylandSurfaceRenderElement<GlesRenderer>> =
            match &cursor_status {
                CursorImageStatus::Surface(surface)
                    if cursor_in_bounds && !hide_cursor && surface.alive() =>
                {
                    let hotspot = with_states(surface, |states| {
                        states
                            .data_map
                            .get::<CursorImageSurfaceData>()
                            .map(|attrs| attrs.lock().unwrap().hotspot)
                            .unwrap_or_default()
                    });
                    let origin = Point::<i32, Physical>::from((
                        scale_f(cursor_local_x, scale) - scale_f(f64::from(hotspot.x), scale),
                        scale_f(cursor_local_y, scale) - scale_f(f64::from(hotspot.y), scale),
                    ));
                    render_elements_from_surface_tree(
                        &mut self.gles,
                        surface,
                        origin,
                        scale,
                        1.0_f32,
                        Kind::Cursor,
                    )
                }
                _ => Vec::new(),
            };

        // Close-out ghosts whose snapshot sits on this output: a fading,
        // shrinking copy of where the window last was. Cloned out
        // (textures are Arc-backed) so they outlive the renderer borrow
        // during the frame block below.
        #[allow(
            clippy::cast_possible_truncation,
            reason = "eased progress in [0,1] casts to an f32 opacity exactly enough"
        )]
        let closing_draws: Vec<(GlesTexture, Rectangle<i32, Physical>, f32)> = self
            .closing
            .iter()
            .filter_map(|c| {
                let cx = c.rect.loc.x + c.rect.size.w / 2;
                let cy = c.rect.loc.y + c.rect.size.h / 2;
                let on_output = cx >= compositor_position.x
                    && cy >= compositor_position.y
                    && cx < compositor_position.x + compositor_size.w
                    && cy < compositor_position.y + compositor_size.h;
                if !on_output {
                    return None;
                }
                let v = c.anim.value(now);
                let alpha = (1.0 - v) as f32;
                let eff = scale_rect_about_center(c.rect, lerp(1.0, OPEN_SCALE_FROM, v));
                let dest = Rectangle::<i32, Physical>::new(
                    Point::from((
                        scale_i(eff.loc.x - compositor_position.x, scale),
                        scale_i(eff.loc.y - compositor_position.y, scale),
                    )),
                    Size::from((scale_i(eff.size.w, scale), scale_i(eff.size.h, scale))),
                );
                Some((c.texture.clone(), dest, alpha))
            })
            .collect();

        let full_damage = [Rectangle::<i32, Physical>::from_size(mode_size)];
        // --- Backdrop bands, factored so the same draw logic feeds both
        // the on-screen frame and the offscreen blur snapshots. ---
        //
        // `draw_base`   : wallpaper + Background/Bottom layers.
        // `cell_local`  : a window's animated rect → output-local physical.
        // `draw_window` : one window's surface, composited through the
        //                 rounded mask (decorated Normal) or drawn straight.
        // `linear` true when drawing into the fp16 HDR scene (vs the sRGB
        // blur pyramid): solid fills are then converted to linear BT.2020.
        // The wallpaper *texture* goes through render_texture_from_to(None),
        // so it picks up the frame's decode override regardless.
        let draw_base = |frame: &mut GlesFrame<'_, '_>, linear: bool| -> Result<()> {
            if let Some(wp) = &wallpaper_media {
                draw_wallpaper_texture(frame, wp, mode_size, linear, hdr_reference_white, hdr_saturation)?;
            } else {
                draw_fill(frame, &wallpaper, mode_size, mode_size, linear, hdr_reference_white, hdr_saturation)?;
            }
            for (bucket, elements) in &layer_groups {
                if matches!(bucket, LayerBucket::Background | LayerBucket::Bottom) {
                    draw_render_elements::<GlesRenderer, _, _>(frame, scale, elements, &full_damage)
                        .context("draw_render_elements (layer bg/bottom) failed")?;
                }
            }
            Ok(())
        };
        let cell_local = |eff: Rectangle<i32, Physical>| {
            Rectangle::<i32, Physical>::new(
                Point::new(
                    scale_i(eff.loc.x - compositor_position.x, scale),
                    scale_i(eff.loc.y - compositor_position.y, scale),
                ),
                Size::new(scale_i(eff.size.w, scale), scale_i(eff.size.h, scale)),
            )
        };

        // --- Phase A: render each decorated Normal window's surface into its
        // own cell-sized offscreen texture (cleared transparent). `draw_window`
        // then composites that texture through the rounded-mask shader, so the
        // corners are genuinely transparent and the backdrop shows through.
        // Undecorated / fullscreen / maximized windows get `None` and draw
        // straight to the frame. No cross-frame pooling: with on-demand
        // rendering an idle output allocates nothing, and these free at frame
        // end. Mirrors the close-snapshot offscreen above.
        let mut win_tex: Vec<Option<GlesTexture>> = Vec::with_capacity(placements.len());
        for ((p, elements), wd) in placements
            .iter()
            .zip(grouped.iter())
            .zip(win_draws.iter())
        {
            if p.fill != FillMode::Normal || !decorated {
                win_tex.push(None);
                continue;
            }
            // An HDR window's offscreen is fp16 and holds *linear BT.2020*
            // (the surface is PQ-decoded into it here), so its decoration can
            // be composited in linear by `ROUND_TEX_SHADER_LINEAR`. SDR windows
            // keep the 8-bit sRGB offscreen the SDR/HDR-decode composite expects.
            let win_is_hdr = hdr && hdr_surface_ids.contains(&p.surface.id());
            let fmt = if win_is_hdr {
                Fourcc::Abgr16161616f
            } else {
                Fourcc::Abgr8888
            };
            let cell = cell_local(wd.effective);
            let size = Size::<i32, smithay::utils::Buffer>::from((
                cell.size.w.max(1),
                cell.size.h.max(1),
            ));
            let phys = Size::<i32, Physical>::from((size.w, size.h));
            let full = [Rectangle::<i32, Physical>::from_size(phys)];
            let tex = (|| -> Option<GlesTexture> {
                let mut tex = self
                    .gles
                    .create_buffer(fmt, size)
                    .inspect_err(|err| warn!(error = %err, "rounded window: offscreen alloc failed"))
                    .ok()?;
                let mut target = self
                    .gles
                    .bind(&mut tex)
                    .inspect_err(|err| warn!(error = %err, "rounded window: bind failed"))
                    .ok()?;
                let mut frame = self
                    .gles
                    .render(&mut target, phys, Transform::Normal)
                    .inspect_err(|err| warn!(error = %err, "rounded window: render failed"))
                    .ok()?;
                let _ = frame.clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &full);
                // HDR window: decode its PQ surface to linear BT.2020 as it's
                // drawn into the fp16 offscreen (the composite then stays linear).
                if win_is_hdr {
                    frame.override_default_tex_program(hdr_decode_shader.clone(), Vec::new());
                }
                draw_render_elements::<GlesRenderer, _, _>(&mut frame, scale, elements, &full)
                    .inspect_err(|err| warn!(error = %err, "rounded window: draw failed"))
                    .ok()?;
                // Same-context sequential GL: the composite that samples this
                // texture is ordered after these writes, so the fence is dropped.
                let _ = frame
                    .finish()
                    .inspect_err(|err| warn!(error = %err, "rounded window: finish failed"))
                    .ok()?;
                drop(target);
                Some(tex)
            })();
            win_tex.push(tex);
        }

        // `linear` is true when drawing into the fp16 linear-BT.2020 HDR
        // scene (vs the sRGB blur pyramid): it selects the HDR shader
        // variants and the per-surface PQ decode. Blur-replay callers pass
        // false so the pyramid stays sRGB.
        let draw_window = |frame: &mut GlesFrame<'_, '_>,
                           p: &Placement,
                           elements: &[RescaleRenderElement<
            WaylandSurfaceRenderElement<GlesRenderer>,
        >],
                           wd: &WinDraw,
                           tex: Option<&GlesTexture>,
                           linear: bool|
         -> Result<()> {
            // A colour-managed (PQ) surface in the linear scene is drawn
            // straight (skipping decoration — option A) with the PQ decode,
            // so its content isn't mis-decoded as SDR.
            let surface_is_hdr = linear && hdr_surface_ids.contains(&p.surface.id());
            // Whether THIS window's offscreen is the fp16 *linear* one built in
            // Phase A — format-based, so independent of `linear` (which is
            // false during the sRGB blur replay).
            let win_is_hdr = hdr && hdr_surface_ids.contains(&p.surface.id());
            if p.fill == FillMode::Normal && decorated {
                // An HDR window's fp16-linear offscreen can't composite into the
                // sRGB blur pyramid, so skip it during the blur replay; it still
                // gets its own background blur in the main pass.
                if win_is_hdr && !linear {
                    return Ok(());
                }
                // Composite the window's pre-rendered surface through the
                // rounded mask: surface inside, opaque border ring, and the
                // corners discarded → genuinely transparent so the backdrop
                // shows through. `None` means the offscreen alloc failed this
                // frame (logged in Phase A) — skip rather than draw garbage.
                let Some(tex) = tex else {
                    return Ok(());
                };
                let dst = cell_local(wd.effective);
                let fill = if p.focused {
                    &border.active
                } else {
                    &border.inactive
                };
                let (mut border_top, mut border_bottom) = match fill {
                    Fill::Solid(rgb) => (*rgb, *rgb),
                    Fill::VerticalGradient { top, bottom } => (*top, *bottom),
                };
                // The linear composite (HDR window) needs the border in linear
                // BT.2020 too — the surface in its fp16 offscreen is already
                // decoded to linear, so the shader doesn't decode.
                if win_is_hdr {
                    let to_lin = |c: [f32; 3]| {
                        let lc = srgb_to_linear_bt2020(
                            Color32F::new(c[0], c[1], c[2], 1.0),
                            hdr_reference_white,
                            hdr_saturation,
                        );
                        let [r, g, b, _] = lc.components();
                        [r, g, b]
                    };
                    border_top = to_lin(border_top);
                    border_bottom = to_lin(border_bottom);
                }
                // Clamp like the old frame mask: radius/border never exceed
                // half the cell, and leave >=1px of surface for the border.
                let max_half = (dst.size.w / 2).min(dst.size.h / 2);
                let radius = scale_i(radius_comp, scale).min(max_half).max(0);
                let bw = scale_i(bw_comp, scale).min((max_half - 1).max(0)).max(0);
                let src = Rectangle::<f64, smithay::utils::Buffer>::from_size(tex.size().to_f64());
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "cell pixel sizes / radius / border are bounded by the output, exact in f32"
                )]
                let mut uniforms = vec![
                    Uniform::new("size", (dst.size.w as f32, dst.size.h as f32)),
                    Uniform::new("radius", radius as f32),
                    Uniform::new("border_width", bw as f32),
                    Uniform::new("border_top", border_top),
                    Uniform::new("border_bottom", border_bottom),
                    Uniform::new("output_height", mode_size.h as f32),
                    Uniform::new("cell_origin_y", dst.loc.y as f32),
                ];
                // HDR window: surface + border already linear → composite
                // directly. SDR window into the linear scene: decode its sRGB
                // offscreen (HDR variant, needs reference_white). SDR scene /
                // blur replay: the plain sRGB composite.
                let program = if win_is_hdr {
                    &round_tex_shader_linear
                } else if linear {
                    uniforms.push(Uniform::new("reference_white", ref_white_f32));
                    uniforms.push(Uniform::new("saturation", hdr_saturation));
                    &round_tex_shader_hdr
                } else {
                    &round_tex_shader
                };
                frame
                    .render_texture_from_to(
                        tex,
                        src,
                        dst,
                        &[Rectangle::from_size(dst.size)],
                        &[],
                        Transform::Normal,
                        // Window opacity + animation alpha, applied here (the
                        // offscreen surface itself was rendered fully opaque).
                        wd.alpha * window_opacity,
                        Some(program),
                        &uniforms,
                    )
                    .context("rounded window composite failed")?;
            } else {
                // Plain rectangle (fullscreen / maximized / undecorated, or an
                // HDR-tagged surface): draw the surface straight to the frame.
                // For HDR content swap the frame's decode override to PQ for
                // this draw, then restore the scene's SDR default.
                if surface_is_hdr {
                    frame.override_default_tex_program(hdr_decode_shader.clone(), Vec::new());
                }
                let res =
                    draw_render_elements::<GlesRenderer, _, _>(frame, scale, elements, &full_damage);
                if surface_is_hdr {
                    frame.override_default_tex_program(
                        sdr_decode_shader.clone(),
                        vec![
                        Uniform::new("reference_white", ref_white_f32),
                        Uniform::new("saturation", hdr_saturation),
                    ],
                    );
                }
                res.context("draw_render_elements failed")?;
            }
            Ok(())
        };
        // Scene stages replayed into the blur accumulator (each on top of
        // the previous): the tiled band, then the floating + maximized band.
        let draw_tiled = |frame: &mut GlesFrame<'_, '_>| -> Result<()> {
            for (((p, elements), wd), tex) in placements
                .iter()
                .zip(grouped.iter())
                .zip(win_draws.iter())
                .zip(win_tex.iter())
                .filter(|(((p, _), _), _)| p.fill == FillMode::Normal && !p.floating)
            {
                draw_window(frame, p, elements, wd, tex.as_ref(), false)?;
            }
            Ok(())
        };
        let draw_floating_max = |frame: &mut GlesFrame<'_, '_>| -> Result<()> {
            for (((p, elements), wd), tex) in placements
                .iter()
                .zip(grouped.iter())
                .zip(win_draws.iter())
                .zip(win_tex.iter())
                .filter(|(((p, _), _), _)| p.fill == FillMode::Normal && p.floating)
            {
                draw_window(frame, p, elements, wd, tex.as_ref(), false)?;
            }
            for (((p, elements), wd), tex) in placements
                .iter()
                .zip(grouped.iter())
                .zip(win_draws.iter())
                .zip(win_tex.iter())
                .filter(|(((p, _), _), _)| p.fill == FillMode::Maximized)
            {
                draw_window(frame, p, elements, wd, tex.as_ref(), false)?;
            }
            Ok(())
        };

        // Backdrop blur (Kawase dual filter). Three z-tiers, each computed
        // only when something above it needs it:
        //   tier 0 = base                        → behind tiled windows
        //   tier 1 = base + tiled windows        → behind floating windows
        //   tier 2 = base + tiled + floating/max → behind Top/Overlay layers
        // Window blur (decoration.blur.windows) drives tiers 0/1; layer
        // blur drives tier 2. We don't probe per-surface alpha, so a mapped
        // opaque panel/window still pays while it's up; the cost is bounded.
        let blur = self.decoration.blur.clone();
        let passes_ok = blur.enabled && blur.passes > 0;
        let any_normal = placements.iter().any(|p| p.fill == FillMode::Normal);
        let need_window = passes_ok && blur.windows && any_normal;
        // Layer blur is opt-in per namespace (config `blur.layers`), so a
        // fullscreen always-mapped overlay (e.g. a launcher) doesn't frost the
        // whole screen — only the layers the user named are blurred.
        let need_layer = passes_ok
            && layers.iter().any(|l| {
                matches!(l.layer, LayerBucket::Top | LayerBucket::Overlay)
                    && layer_should_blur(&blur, &l.namespace)
            });
        // Saved per-tier blurred backdrops. Pull the scratch out of the map
        // so the blur helpers borrow only `self.gles`; on any GPU failure
        // we clear the tiers and fall back to sharp rendering. Programs are
        // `Arc`-cloned so the staging closure captures only `self.gles`.
        let mut tier_tiled: Option<GlesTexture> = None;
        let mut tier_float: Option<GlesTexture> = None;
        let mut tier_layer: Option<GlesTexture> = None;
        if (need_window || need_layer) && self.ensure_blur_scratch(idx, mode_size, blur.passes) {
            let down = self.blur_down.clone();
            let up = self.blur_up.clone();
            let passes = blur.passes as usize;
            let radius = blur.radius;
            let mut scratch = self
                .blur_scratch
                .remove(&idx)
                .expect("ensure_blur_scratch inserted it");
            let res: Result<()> = (|| {
                render_scene_stage(&mut self.gles, &mut scratch, mode_size, &|f| {
                    draw_base(f, false)
                })?;
                if need_window {
                    run_pyramid(&mut self.gles, &mut scratch, passes, radius, &down, &up, 0)?;
                    tier_tiled = Some(scratch.tiers[0].clone());
                }
                // The tiled band feeds both the floating tier and the layer tier.
                if need_window || need_layer {
                    render_scene_stage(&mut self.gles, &mut scratch, mode_size, &draw_tiled)?;
                }
                if need_window {
                    run_pyramid(&mut self.gles, &mut scratch, passes, radius, &down, &up, 1)?;
                    tier_float = Some(scratch.tiers[1].clone());
                }
                if need_layer {
                    render_scene_stage(
                        &mut self.gles,
                        &mut scratch,
                        mode_size,
                        &draw_floating_max,
                    )?;
                    run_pyramid(&mut self.gles, &mut scratch, passes, radius, &down, &up, 2)?;
                    tier_layer = Some(scratch.tiers[2].clone());
                }
                Ok(())
            })();
            if let Err(err) = res {
                warn!(error = %err, output = %output_name, "backdrop blur failed; rendering sharp");
                tier_tiled = None;
                tier_float = None;
                tier_layer = None;
            }
            self.blur_scratch.insert(idx, scratch);
        }
        // Paint a full-res tier's sub-rect behind a translucent surface. The
        // tier texture is 1:1 with the framebuffer, so the source sub-rect
        // matches the on-screen destination rect. With a `mask` texture
        // (layer-shell panels) the blur is alpha-masked by the surface's own
        // buffer, so the frost follows whatever shape the client drew — the
        // compositor can't know a panel's corner radius. Without one
        // (windows) an SDF clips the tier to the same rounded rect
        // `draw_window` composites.
        let blur_rect = |frame: &mut GlesFrame<'_, '_>,
                         tier: &GlesTexture,
                         dst: Rectangle<i32, Physical>,
                         mask: Option<&GlesTexture>|
         -> Result<()> {
            let src = Rectangle::<f64, smithay::utils::Buffer>::new(
                Point::from((f64::from(dst.loc.x), f64::from(dst.loc.y))),
                Size::from((f64::from(dst.size.w), f64::from(dst.size.h))),
            );
            // `render_texture_from_to` treats the damage / opaque rects as
            // *relative to `dst`'s origin* (it constrains them to `dst.size`),
            // so they must be `(0,0)`-anchored. Passing the absolute `dst`
            // collapses to a zero-size instance for any offset surface — the
            // whole reason blur only ever showed full-screen.
            let local = [Rectangle::from_size(dst.size)];
            // `v_coords` in the blur shaders samples the tier: it spans the
            // `src` sub-rect normalised over the *whole* tier, not 0..1
            // across `dst`. Hand the shaders a CPU-computed affine map from
            // `v_coords` back to rect-local pixels / mask UV instead, flipped
            // when a texture is y-inverted (create_buffer offscreens are not,
            // but derive it from the texture rather than assume).
            let tier_sz = tier.size();
            #[allow(
                clippy::cast_precision_loss,
                reason = "cell pixel sizes / radius are bounded by the output, exact in f32"
            )]
            let (dst_w, dst_h, loc_x, loc_y, tier_w, tier_h) = (
                dst.size.w as f32,
                dst.size.h as f32,
                dst.loc.x as f32,
                dst.loc.y as f32,
                tier_sz.w as f32,
                tier_sz.h as f32,
            );
            // v_coords → absolute output px: abs.x = v.x * tier_w,
            // abs.y = v.y * ay + by.
            let (ay, by) = if tier.is_y_inverted() {
                (-tier_h, tier_h)
            } else {
                (tier_h, 0.0)
            };

            if let Some(mask) = mask {
                // abs px → mask UV: shift by the rect origin, normalise by
                // the rect size, and flip y again if the client's buffer is
                // y-inverted.
                let mut mask_mul = (tier_w / dst_w, ay / dst_h);
                let mut mask_add = (-loc_x / dst_w, (by - loc_y) / dst_h);
                if mask.is_y_inverted() {
                    mask_mul.1 = -mask_mul.1;
                    mask_add.1 = 1.0 - mask_add.1;
                }
                let mut uniforms = vec![
                    Uniform::new("mask", 1i32),
                    Uniform::new("mask_mul", mask_mul),
                    Uniform::new("mask_add", mask_add),
                ];
                // The blur pyramid is sRGB; into the linear HDR scene the HDR
                // variant decodes it to linear BT.2020 (needs reference_white).
                let program = if hdr {
                    uniforms.push(Uniform::new("reference_white", ref_white_f32));
                    uniforms.push(Uniform::new("saturation", hdr_saturation));
                    &mask_blur_shader_hdr
                } else {
                    &mask_blur_shader
                };
                // The shader's `mask` sampler reads texture unit 1; smithay's
                // draw only drives unit 0, so the secondary binding (made and
                // restored by the vendored helper) survives the call.
                frame
                    .with_secondary_texture(mask, |frame| {
                        frame.render_texture_from_to(
                            tier,
                            src,
                            dst,
                            &local,
                            &[],
                            Transform::Normal,
                            1.0,
                            Some(program),
                            &uniforms,
                        )
                    })
                    .context("blur: alpha-masked backdrop sub-rect")?;
                return Ok(());
            }

            // No mask (windows): SDF-clip the tier to the rounded rect the
            // window composite draws. Radius matches `draw_window`'s clamp.
            // The AA edge needs blending, so the opaque-region hint is empty;
            // with `radius_comp == 0` the SDF is a plain rect (old behaviour).
            let max_half = (dst.size.w / 2).min(dst.size.h / 2);
            let radius = scale_i(radius_comp, scale).min(max_half).max(0);
            #[allow(
                clippy::cast_precision_loss,
                reason = "cell pixel sizes / radius are bounded by the output, exact in f32"
            )]
            let mut uniforms = vec![
                Uniform::new("size", (dst_w, dst_h)),
                Uniform::new("radius", radius as f32),
                Uniform::new("local_mul", (tier_w, ay)),
                Uniform::new("local_add", (-loc_x, by - loc_y)),
            ];
            // The blur pyramid is sRGB; into the linear HDR scene the HDR
            // variant decodes it to linear BT.2020 (needs reference_white).
            let program = if hdr {
                uniforms.push(Uniform::new("reference_white", ref_white_f32));
                uniforms.push(Uniform::new("saturation", hdr_saturation));
                &round_blur_shader_hdr
            } else {
                &round_blur_shader
            };
            frame
                .render_texture_from_to(
                    tier,
                    src,
                    dst,
                    &local,
                    &[],
                    Transform::Normal,
                    1.0,
                    Some(program),
                    &uniforms,
                )
                .context("blur: backdrop sub-rect")?;
            Ok(())
        };

        let mut target = if hdr {
            let scene = self
                .hdr_scene
                .get_mut(&output_name)
                .expect("HDR scene buffer just ensured");
            self.gles.bind(scene).with_context(|| {
                format!("GlesRenderer::bind (HDR scene) failed for {output_name}")
            })?
        } else {
            self.gles
                .bind(&mut dmabuf)
                .with_context(|| format!("GlesRenderer::bind failed for {output_name}"))?
        };
        let scene_sync = {
            let mut frame = self
                .gles
                .render(&mut target, mode_size, Transform::Normal)
                .with_context(|| format!("GlesRenderer::render failed for {output_name}"))?;

            // HDR output: composite in the linear BT.2020 working space.
            // Default every override-respecting source draw (wallpaper,
            // windows, layers, popups, cursor) to the SDR decode; HDR-tagged
            // surfaces swap to the PQ decode inside draw_window. The
            // single-pass fast path (`hdr` demoted to false above) instead
            // defaults to the fused SDR→PQ program — the only draw in that
            // frame is the solo SDR window, going straight to the scanout.
            if hdr {
                frame.override_default_tex_program(
                    sdr_decode_shader.clone(),
                    vec![
                        Uniform::new("reference_white", ref_white_f32),
                        Uniform::new("saturation", hdr_saturation),
                    ],
                );
            } else if single_pass_hdr {
                frame.override_default_tex_program(
                    sdr_to_pq_shader.clone(),
                    vec![
                        Uniform::new("reference_white", ref_white_f32),
                        Uniform::new("saturation", hdr_saturation),
                    ],
                );
            }

            // Backdrop bands drawn fresh, interleaving the blurred tiers so
            // each translucent surface reveals a blurred copy of whatever
            // sits beneath it. Layer-shell order (wlr-layer-shell spec):
            //   wallpaper → Background → Bottom → windows → Top → Overlay → cursor.
            // Each window keeps its own `draw_render_elements` call
            // (single-element slice) so smithay's opaque-region culling
            // can't skip floats behind earlier tiles.
            // A provably-opaque solo fullscreen window overwrites every
            // output pixel, so the base band beneath it is skipped outright.
            if solo_opaque.is_none() {
                draw_base(&mut frame, hdr)?;
            }
            // Tiled windows blur against the base (tier 0).
            for (((p, elements), wd), tex) in placements
                .iter()
                .zip(grouped.iter())
                .zip(win_draws.iter())
                .zip(win_tex.iter())
                .filter(|(((p, _), _), _)| p.fill == FillMode::Normal && !p.floating)
            {
                if let Some(t) = &tier_tiled {
                    blur_rect(&mut frame, t, cell_local(wd.effective), None)?;
                }
                draw_window(&mut frame, p, elements, wd, tex.as_ref(), hdr)?;
            }
            // Floating windows draw above tiled and blur against base +
            // tiled (tier 1), so a float reveals the windows beneath it.
            for (((p, elements), wd), tex) in placements
                .iter()
                .zip(grouped.iter())
                .zip(win_draws.iter())
                .zip(win_tex.iter())
                .filter(|(((p, _), _), _)| p.fill == FillMode::Normal && p.floating)
            {
                if let Some(t) = &tier_float {
                    blur_rect(&mut frame, t, cell_local(wd.effective), None)?;
                }
                draw_window(&mut frame, p, elements, wd, tex.as_ref(), hdr)?;
            }
            // Maximized windows: borderless, above normal windows but below
            // Top/Overlay panels. Opaque — no backdrop blur.
            for (((p, elements), wd), tex) in placements
                .iter()
                .zip(grouped.iter())
                .zip(win_draws.iter())
                .zip(win_tex.iter())
                .filter(|(((p, _), _), _)| p.fill == FillMode::Maximized)
            {
                draw_window(&mut frame, p, elements, wd, tex.as_ref(), hdr)?;
            }

            // Top layer surfaces go above windows but below a fullscreen
            // window (status bar above kitty, but a fullscreen game covers the
            // bar). Blur against the full backdrop (tier 2) so a translucent
            // panel reveals a frosted desktop.
            for ((l, (bucket, elements)), mask) in layers
                .iter()
                .zip(layer_groups.iter())
                .zip(layer_masks.iter())
            {
                if !matches!(bucket, LayerBucket::Top) {
                    continue;
                }
                if let Some(t) = &tier_layer
                    && layer_should_blur(&blur, &l.namespace)
                {
                    let dst = Rectangle::<i32, Physical>::new(
                        Point::new(
                            scale_i(l.rect.loc.x - compositor_position.x, scale),
                            scale_i(l.rect.loc.y - compositor_position.y, scale),
                        ),
                        Size::new(scale_i(l.rect.size.w, scale), scale_i(l.rect.size.h, scale)),
                    );
                    blur_rect(&mut frame, t, dst, mask.as_ref())?;
                }
                draw_render_elements::<GlesRenderer, _, _>(&mut frame, scale, elements, &full_damage)
                    .context("draw_render_elements (layer top) failed")?;
            }

            // Fullscreen windows: borderless, above tiled/maximized windows and
            // Top panels (a fullscreen game/video covers the bar), but BELOW
            // Overlay layers (launcher / toasts / OSDs stay visible) and below
            // popups and the cursor.
            for (p, elements) in placements
                .iter()
                .zip(grouped.iter())
                .filter(|(p, _)| p.fill == FillMode::Fullscreen)
            {
                // Colour-managed (PQ) fullscreen surface (e.g. an HDR game):
                // swap the frame's decode override to PQ for this draw, then
                // restore the scene's SDR default.
                let surface_is_hdr = hdr && hdr_surface_ids.contains(&p.surface.id());
                if surface_is_hdr {
                    frame.override_default_tex_program(hdr_decode_shader.clone(), Vec::new());
                }
                let res = draw_render_elements::<GlesRenderer, _, _>(
                    &mut frame,
                    scale,
                    elements,
                    &full_damage,
                );
                if surface_is_hdr {
                    frame.override_default_tex_program(
                        sdr_decode_shader.clone(),
                        vec![
                        Uniform::new("reference_white", ref_white_f32),
                        Uniform::new("saturation", hdr_saturation),
                    ],
                    );
                }
                res.context("draw_render_elements (fullscreen) failed")?;
            }

            // Overlay layer surfaces go above everything else below the cursor —
            // above windows AND fullscreen, so a launcher / toast / OSD stays on
            // top of a fullscreen game. Same tier-2 blur as the Top layer.
            for ((l, (bucket, elements)), mask) in layers
                .iter()
                .zip(layer_groups.iter())
                .zip(layer_masks.iter())
            {
                if !matches!(bucket, LayerBucket::Overlay) {
                    continue;
                }
                if let Some(t) = &tier_layer
                    && layer_should_blur(&blur, &l.namespace)
                {
                    let dst = Rectangle::<i32, Physical>::new(
                        Point::new(
                            scale_i(l.rect.loc.x - compositor_position.x, scale),
                            scale_i(l.rect.loc.y - compositor_position.y, scale),
                        ),
                        Size::new(scale_i(l.rect.size.w, scale), scale_i(l.rect.size.h, scale)),
                    );
                    blur_rect(&mut frame, t, dst, mask.as_ref())?;
                }
                draw_render_elements::<GlesRenderer, _, _>(&mut frame, scale, elements, &full_damage)
                    .context("draw_render_elements (layer overlay) failed")?;
            }

            // Closing windows: the fade/shrink-out snapshot, above the
            // windows reflowing to fill the freed space, below popups.
            for (texture, dest, alpha) in &closing_draws {
                frame
                    .render_texture_from_to(
                        texture,
                        Rectangle::from_size(texture.size()).to_f64(),
                        *dest,
                        &[Rectangle::from_size(dest.size)],
                        &[],
                        Transform::Normal,
                        *alpha,
                        None,
                        &[],
                    )
                    .context("render_texture_from_to (closing window) failed")?;
            }

            // Popups draw above everything except the cursor — above
            // tiled/floating windows AND above Top/Overlay layers, so a
            // menu opened from a panel is never occluded. Parent-first
            // snapshot order means nested submenus land on top.
            for elements in &popup_groups {
                draw_render_elements::<GlesRenderer, _, _>(
                    &mut frame,
                    scale,
                    elements,
                    &full_damage,
                )
                .context("draw_render_elements (popup) failed")?;
            }

            // Screenshot session: cover the (possibly still-updating)
            // scene with the frozen snapshot so selection happens against
            // a paused image, then dim + outline the selection. Drawn
            // after the scene and before the cursor so the pointer stays
            // visible while you select.
            if let Some(tex) = &freeze_texture {
                let dst = Rectangle::from_size(mode_size);
                let src = Rectangle::from_size(tex.size()).to_f64();
                let damage = [dst];
                frame
                    .render_texture_from_to(
                        tex,
                        src,
                        dst,
                        &damage,
                        // The captured frame is opaque (the X byte is not
                        // alpha); mark it fully opaque so the garbage pad
                        // never blends.
                        &damage,
                        Transform::Normal,
                        1.0,
                        None,
                        &[],
                    )
                    .context("render_texture_from_to (freeze) failed")?;
            }
            if let Some(overlay) = screenshot_overlay {
                draw_screenshot_overlay(
                    &mut frame,
                    &overlay,
                    compositor_position,
                    mode_size,
                    scale,
                    hdr,
                    hdr_reference_white,
                    hdr_saturation,
                )?;
            }

            // Drag-and-drop icon, just under the cursor sprite.
            if !dnd_icon_elements.is_empty() {
                draw_render_elements::<GlesRenderer, _, _>(
                    &mut frame,
                    scale,
                    &dnd_icon_elements,
                    &full_damage,
                )
                .context("draw_render_elements (dnd icon) failed")?;
            }

            // Skip the cursor entirely while the pointer is locked (a
            // game with an active pointer lock draws its own crosshair;
            // ours would sit frozen at the lock point). Otherwise draw
            // whatever the effective cursor status calls for: a client
            // surface (its own pointer image), a themed named sprite, or
            // — when Hidden — nothing.
            if cursor_in_bounds && !hide_cursor {
                match &cursor_status {
                    CursorImageStatus::Hidden => {}
                    CursorImageStatus::Surface(_) => {
                        // The plane scans the client cursor out directly, so
                        // skip compositing it — unless a capture needs it baked
                        // in, or the plane isn't handling it (readback failed /
                        // no buffer yet). An empty element list (surface with no
                        // committed buffer) is the client hiding the cursor.
                        if (compose_cursor || !hw_cursor_active)
                            && !cursor_surface_elements.is_empty()
                        {
                            draw_render_elements::<GlesRenderer, _, _>(
                                &mut frame,
                                scale,
                                &cursor_surface_elements,
                                &full_damage,
                            )
                            .context("draw_render_elements (cursor surface) failed")?;
                        }
                    }
                    CursorImageStatus::Named(_) => {
                        // The hardware cursor plane scans the themed cursor out
                        // directly, so skip compositing it — unless this frame
                        // must bake it into the framebuffer for a capture, or
                        // the plane isn't handling it (no plane / oversize).
                        if compose_cursor || !hw_cursor_active {
                            // Pointer hotspot in this output's physical pixels.
                            let hotspot = Point::<i32, Physical>::from((
                                scale_f(cursor_local_x, scale),
                                scale_f(cursor_local_y, scale),
                            ));
                            draw_cursor(
                                &mut frame,
                                cursor_sprite.as_ref(),
                                cursor_size,
                                hotspot,
                                scale,
                                hdr,
                                hdr_reference_white,
                                hdr_saturation,
                            )?;
                        }
                    }
                }
            }

            frame.finish().context("Frame::finish failed")?
        };

        // Service pending screencopy captures off the freshly
        // composited framebuffer. `frame` is finished (so it no longer
        // borrows the renderer) but `target` is still bound, which is
        // exactly what `copy_framebuffer` needs. Pixels go back to the
        // caller, which writes them into client buffers + signals the
        // frames. Done before `queue_buffer` so we read the buffer
        // while it's unambiguously ours.
        // SDR composited straight to the 8-bit scanout, so captures read
        // `target` directly. HDR's `target` is the fp16 linear-BT.2020
        // offscreen, which GLES can't read back as an 8-bit format (and
        // wouldn't be SDR colour anyway) — those are serviced below via a
        // tonemap-to-sRGB pass once `target` is released.
        let mut capture_results: Vec<CaptureOutcome> = if hdr {
            Vec::new()
        } else {
            captures
                .iter()
                .map(|spec| match &spec.target {
                    CaptureTarget::Shm => {
                        capture_shm(&mut self.gles, &target, spec, &output_name)
                    }
                    CaptureTarget::Dmabuf(client) => {
                        capture_dmabuf(&mut self.gles, &target, client, spec, &output_name)
                    }
                })
                .collect()
        };
        drop(target);

        // HDR screenshots: tonemap the linear scene to 8-bit sRGB and read
        // that, so a capture of an HDR output "looks like SDR".
        if hdr && !captures.is_empty() {
            capture_results =
                self.capture_tonemapped(&output_name, mode_size, ref_white_f32, captures);
        }

        // HDR: encode the composited linear-BT.2020 scene (the fp16 offscreen)
        // to PQ / BT.2020 into the 10-bit scanout dmabuf. SDR keeps the
        // scene's own sync — it composited straight to the dmabuf.
        let sync = if hdr {
            let scene_tex = self
                .hdr_scene
                .get(&output_name)
                .expect("HDR scene buffer present");
            let mut hdr_target = self.gles.bind(&mut dmabuf).with_context(|| {
                format!("GlesRenderer::bind (HDR scanout) failed for {output_name}")
            })?;
            let encoded = {
                let mut frame = self
                    .gles
                    .render(&mut hdr_target, mode_size, Transform::Normal)
                    .with_context(|| format!("HDR encode render failed for {output_name}"))?;
                let dst = Rectangle::from_size(mode_size);
                let src = Rectangle::<f64, smithay::utils::Buffer>::from_size(Size::from((
                    f64::from(mode_size.w),
                    f64::from(mode_size.h),
                )));
                frame
                    .render_texture_from_to(
                        scene_tex,
                        src,
                        dst,
                        &[dst],
                        &[dst],
                        Transform::Normal,
                        1.0,
                        Some(&hdr_encode_shader),
                        // PQ-only encode: scene is already linear BT.2020.
                        &[],
                    )
                    .context("HDR encode pass")?;
                frame.finish().context("HDR encode finish")?
            };
            drop(hdr_target);
            encoded
        } else {
            scene_sync
        };

        // Settle this output's adaptive-sync state for the frame we're
        // about to queue. Must run before `queue_buffer` so the commit it
        // triggers carries the right VRR_ENABLED (smithay promotes the
        // commit to a modeset itself when the toggle demands one).
        self.apply_vrr(idx, placements);

        self.outputs[idx]
            .surface
            .queue_buffer(Some(sync), None)
            .with_context(|| format!("queue_buffer failed for {output_name}"))?;
        debug!(output = %output_name, "frame queued for scanout");

        // Fire wl_callback.done on every surface we rendered, per-output
        // filtered (see `send_output_frame_callbacks` — shared with the
        // direct-scanout path).
        self.send_output_frame_callbacks(placements, layers, popups, out_rect);

        // Collect wp_presentation feedback for the surfaces in this composited
        // frame; fired with the real vblank timestamp in `frame_submitted`.
        // Not zero-copy (the scene went through the GLES compositor).
        if let Some(out) = present_output {
            self.outputs[idx].pending_feedback = Some(collect_presentation_feedback(
                out, placements, layers, popups, out_rect, false,
            ));
        }

        // Tell the on-demand driver whether this output still produces
        // frames on its own — an in-flight window/close/open animation, or
        // a visible media wallpaper (which advances every frame). When
        // none hold, the output may park until the next external trigger.
        // A media wallpaper hidden behind a fullscreen/maximized window is
        // occluded, so it doesn't count (letting a fullscreen game's output
        // park between the game's own commits — the whole point of VRR).
        let wallpaper_live = self
            .wallpaper_media
            .as_ref()
            .is_some_and(|m| m.anim.is_live());
        // `win_anims` holds one entry per *tracked* window (for smooth move
        // retargeting), so it's non-empty whenever any window exists — check
        // for an actually-running move/open animation instead, or every
        // output would free-run forever the moment a window maps.
        let anim_running = self
            .win_anims
            .values()
            .any(|w| w.move_anim.is_some() || w.open_anim.is_some());
        let followup = anim_running
            || !self.closing.is_empty()
            || !self.pending_open.is_empty()
            || (wallpaper_live && !self.output_has_fill_window(idx, placements));
        Ok((capture_results, followup))
    }

    /// Whether a fullscreen or maximized window currently covers this
    /// output — the windows for which `Auto` VRR engages, and behind which
    /// the media wallpaper is fully occluded.
    fn output_has_fill_window(&self, idx: usize, placements: &[Placement]) -> bool {
        let rect = Rectangle::new(
            self.outputs[idx].compositor_position,
            self.outputs[idx].compositor_size,
        );
        placements
            .iter()
            .any(|p| p.fill != FillMode::Normal && p.cell_rect.overlaps(rect))
    }

    /// Per-output `wl_callback.done` dispatch. Fires on every surface visible
    /// on this output (windows/layers filtered by overlap; popups always),
    /// draining each surface's callback queue so a second output's render is a
    /// no-op. Per-output filtering keeps a fast output (a fullscreen game)
    /// from driving clients on other outputs and pegging them to its refresh
    /// rate — preserving VRR isolation. Shared by the composite and
    /// direct-scanout paths.
    fn send_output_frame_callbacks(
        &self,
        placements: &[Placement],
        layers: &[LayerPlacement],
        popups: &[PopupPlacement],
        out_rect: Rectangle<i32, Physical>,
    ) {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "wl_callback.done takes u32 ms which the spec expects to wrap freely (~50d period)"
        )]
        let elapsed_ms = self.start.elapsed().as_millis() as u32;
        for p in placements {
            if p.cell_rect.overlaps(out_rect) {
                send_frame_callbacks(&p.surface, elapsed_ms);
            }
        }
        for l in layers {
            if l.rect.overlaps(out_rect) {
                send_frame_callbacks(&l.surface, elapsed_ms);
            }
        }
        // Popups are tiny, transient, and tied to a parent already covered
        // above; fire unconditionally rather than track their output.
        for p in popups {
            send_frame_callbacks(&p.surface, elapsed_ms);
        }
    }

    /// Decide whether this output's frame can be served by latching a single
    /// client's buffer straight onto the primary plane (direct scanout),
    /// returning the buffer keep-alive + dmabuf when so. `None` means the
    /// frame must be composited.
    ///
    /// Whether this output's frame is exactly one settled fullscreen
    /// window and nothing else — the scene precondition shared by direct
    /// scanout and the single-pass HDR fast path. Returns the covering
    /// placement's index when: no captures, the cursor needs no
    /// compositing (hidden / off-output / on the HW plane), no popup or
    /// layer-shell/session-lock surface overlaps this output, no
    /// transient overlay or window animation is running, and exactly one
    /// placement covers the output — `Fullscreen` fill, settled at 1:1
    /// (`effective == out_rect`), fully visible (`alpha == 1`).
    ///
    /// Note the popup check is overlap-based, not `popups.is_empty()`:
    /// an X11 override-redirect window some client keeps mapped on
    /// another output (Steam does) must not veto this output's fast
    /// paths.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors render_output's per-frame inputs; a struct would not simplify"
    )]
    fn solo_fullscreen_scene(
        &self,
        idx: usize,
        win_draws: &[WinDraw],
        placements: &[Placement],
        layers: &[LayerPlacement],
        popups: &[PopupPlacement],
        hide_cursor: bool,
        captures: &[CaptureSpec],
        compose_cursor: bool,
    ) -> Option<usize> {
        let output = &self.outputs[idx];
        let out_rect = Rectangle::new(output.compositor_position, output.compositor_size);

        // A capture must read a composited framebuffer; the cursor must not
        // need drawing into this output's frame (hidden and off-output
        // pointers need nothing; a plane-resident one scans out alongside).
        if !captures.is_empty() || self.cursor_needs_composite(idx, hide_cursor, compose_cursor) {
            return None;
        }
        // Anything that draws above a fullscreen window forces compositing.
        // `layers` carries real layer-shell surfaces *and* the session-lock
        // surface (injected as an Overlay layer by render_crtc).
        if popups.iter().any(|pp| pp.rect.overlaps(out_rect))
            || layers.iter().any(|l| l.rect.overlaps(out_rect))
        {
            return None;
        }
        // Transient overlays / running window animations (any of which draws
        // over, or distorts, the fullscreen window) → composite.
        if !self.closing.is_empty()
            || !self.pending_open.is_empty()
            || self.screenshot_overlay.is_some()
            || self.dnd_icon.is_some()
            || self.freeze_textures.contains_key(&output.name)
            || self
                .win_anims
                .values()
                .any(|w| w.move_anim.is_some() || w.open_anim.is_some())
        {
            return None;
        }

        // Exactly one placement may cover the output.
        let mut covering = placements
            .iter()
            .enumerate()
            .filter(|(_, p)| p.cell_rect.overlaps(out_rect));
        let (i, p) = covering.next()?;
        if covering.next().is_some() {
            return None;
        }

        // It must be a settled (1:1, fully visible) fullscreen window.
        if p.fill != FillMode::Fullscreen || p.slide_dy != 0 {
            return None;
        }
        let draw = win_draws.get(i)?;
        if draw.effective != out_rect || draw.alpha < 1.0 {
            return None;
        }
        Some(i)
    }

    /// Eligible only when the scene is a solo fullscreen window (see
    /// [`Self::solo_fullscreen_scene`], decided by the caller) whose colour
    /// mode matches the output (HDR output ⇔ PQ surface), backed by a
    /// single dmabuf buffer with no transform that is pixel-exact with the
    /// mode.
    fn direct_scanout_inputs(
        &self,
        idx: usize,
        solo: Option<usize>,
        placements: &[Placement],
        hdr_surface_ids: &HashSet<ObjectId>,
    ) -> Option<DirectInputs> {
        let output = &self.outputs[idx];
        let p = placements.get(solo?)?;

        // The window's colour mode must match the output: an SDR surface on
        // an HDR output needs the compositor's PQ encode (see the single-pass
        // fast path), a PQ surface on an SDR output needs a tonemap.
        if output.hdr != hdr_surface_ids.contains(&p.surface.id()) {
            return None;
        }
        // A toplevel with subsurfaces isn't a single scannable buffer.
        if !surface_is_single_node(&p.surface) {
            return None;
        }

        // Extract a scanout-ready dmabuf + keep-alive from the committed
        // buffer. Rejects shm buffers, transformed buffers, viewport-cropped
        // buffers, buffers whose pixels don't match the mode 1:1, and buffers
        // we can't prove are opaque.
        let mode_size = output.mode_size;
        let out_size = output.compositor_size;
        with_renderer_surface_state(&p.surface, |state| {
            // The buffer must land on the plane 1:1: no rotation, no crop
            // (src covers the whole surface), and a logical destination that
            // covers exactly this output. dst is compared against the OUTPUT
            // size, not the buffer: a fractional-aware client (oversized
            // buffer + viewport) and an Xwayland client under the client
            // scale (physical-sized buffer that smithay shrinks logically)
            // both have dst < buffer *by design* while their pixels still
            // match the mode exactly — `dmabuf.size == mode_size` below is
            // the pixel-exactness gate. Requiring `dst == buffer` here kept
            // every such fullscreen game compositing (0 scanned-out frames in
            // a whole session). `buffer_scale` needs no check of its own —
            // it's already folded into both src (surface-logical units) and
            // dst.
            if state.buffer_transform() != Transform::Normal {
                return None;
            }
            let buf = state.buffer_size()?;
            let view = state.view()?;
            let src_loc = view.src.loc.to_i32_round::<i32>();
            if src_loc.x != 0
                || src_loc.y != 0
                || view.src.size.to_i32_round::<i32>() != buf
                || view.dst.w != out_size.w
                || view.dst.h != out_size.h
            {
                return None;
            }

            let buffer = state.buffer()?.clone();
            let dmabuf = smithay::wayland::dmabuf::get_dmabuf(&buffer).ok()?.clone();
            let size = dmabuf.size();
            if size.w != mode_size.w || size.h != mode_size.h {
                return None;
            }

            // Provable opacity. The composite path blends an alpha buffer over
            // the wallpaper; direct scanout ignores the alpha (nothing is below
            // the primary plane), so the two only agree when the surface is
            // actually opaque. A no-alpha format is inherently opaque; an alpha
            // format must declare a full opaque region. An opaque alpha buffer
            // is scanned out via the opaque sibling fourcc.
            let code = dmabuf.format().code;
            if has_alpha(code) && !opaque_region_covers(state.opaque_regions(), buf) {
                return None;
            }
            let use_opaque = has_alpha(code);
            Some(DirectInputs {
                buffer,
                dmabuf,
                use_opaque,
            })
        })
        .flatten()
    }
}

/// Buffer to latch directly onto the primary plane (direct scanout). Produced
/// by [`Renderer::direct_scanout_inputs`] and consumed by
/// [`ScanoutSurface::try_queue_external`].
struct DirectInputs {
    /// Keep-alive for the client buffer; holding it defers `wl_buffer.release`
    /// until a later flip replaces this buffer on the plane.
    buffer: ClientBuffer,
    dmabuf: Dmabuf,
    /// Program the plane with the opaque sibling fourcc (ignore the alpha
    /// channel) — set when the client buffer's format carries unused alpha.
    use_opaque: bool,
}

/// Whether the surface's opaque regions cover the whole surface — i.e. it is
/// provably fully opaque. (smithay auto-fills a full opaque region for no-alpha
/// buffers; alpha buffers carry whatever region the client declared.) A single
/// region covering the surface is the common case; partial-tiling regions
/// conservatively read as "not provably opaque".
fn opaque_region_covers(
    regions: Option<&[Rectangle<i32, Logical>]>,
    size: Size<i32, Logical>,
) -> bool {
    regions.is_some_and(|rs| {
        rs.iter().any(|r| {
            r.loc.x <= 0
                && r.loc.y <= 0
                && r.loc.x + r.size.w >= size.w
                && r.loc.y + r.size.h >= size.h
        })
    })
}

/// Collect `wp_presentation` feedback for every surface visible on this output
/// this frame into an [`OutputPresentationFeedback`], to be fired on the next
/// vblank. Mirrors [`Renderer::send_output_frame_callbacks`]'s per-output
/// filtering (windows/layers by overlap, popups unconditionally). `zero_copy`
/// tags surfaces scanned out directly (no compositing copy).
fn collect_presentation_feedback(
    output: &Output,
    placements: &[Placement],
    layers: &[LayerPlacement],
    popups: &[PopupPlacement],
    out_rect: Rectangle<i32, Physical>,
    zero_copy: bool,
) -> OutputPresentationFeedback {
    let mut feedback = OutputPresentationFeedback::new(output);
    let flags = if zero_copy {
        PresentKind::ZeroCopy
    } else {
        PresentKind::empty()
    };
    {
        let mut collect = |surface: &WlSurface| {
            take_presentation_feedback_surface_tree(
                surface,
                &mut feedback,
                |_, _| Some(output.clone()),
                |_, _| flags,
            );
        };
        for p in placements {
            if p.cell_rect.overlaps(out_rect) {
                collect(&p.surface);
            }
        }
        for l in layers {
            if l.rect.overlaps(out_rect) {
                collect(&l.surface);
            }
        }
        for p in popups {
            collect(&p.surface);
        }
    }
    feedback
}

/// Whether `surface`'s committed buffer provably covers its whole extent
/// opaquely, per smithay's computed opaque regions (a no-alpha buffer gets
/// a full-extent region automatically; an alpha buffer carries the client's
/// declared region). Checked against both the buffer's logical size and the
/// surface view's destination — client-declared regions are surface-local
/// while smithay's auto-region is view-sized, and a full cover in either
/// unit system is a genuine full-coverage declaration. Used by the
/// composite fast paths that skip drawing anything underneath the surface.
fn surface_provably_opaque(surface: &WlSurface) -> bool {
    with_renderer_surface_state(surface, |state| {
        let (Some(buf), Some(view)) = (state.buffer_size(), state.view()) else {
            return false;
        };
        opaque_region_covers(state.opaque_regions(), buf)
            || opaque_region_covers(state.opaque_regions(), view.dst)
    })
    .unwrap_or(false)
}

/// Whether `surface`'s tree is a single node (no subsurfaces). A prerequisite
/// for direct scanout: subsurfaces would be lost if we latched only the root
/// buffer onto the plane.
fn surface_is_single_node(surface: &WlSurface) -> bool {
    let mut count = 0usize;
    with_surface_tree_downward(
        surface,
        (),
        |_, _, &()| TraversalAction::DoChildren(()),
        |_, _, &()| {
            count += 1;
        },
        |_, _, &()| true,
    );
    count == 1
}

/// CPU read-back: copy `spec.region` of `target` into a tight buffer.
///
/// Coordinates and rows are memory-ordered, not GL-bottom-left: every
/// capture target here is an FBO attachment (the scanout dmabuf or an
/// offscreen texture), and `glReadPixels` on an FBO preserves
/// texel-row = memory-row order. The rendered framebuffer is top-down
/// in memory (scanout displays memory-row 0 as the top scanline), so
/// `spec.region`'s top-left coordinates index it directly and the
/// read-back rows are already upright. Do NOT consult
/// `mapping.flipped()`: smithay hard-codes it `true`, which describes
/// default-framebuffer (`ReadBuffer(BACK)`) readbacks — it does not
/// apply to FBO reads, and honouring it here delivers vertically
/// mirrored frames.
fn capture_shm(
    gles: &mut GlesRenderer,
    target: &GlesTarget<'_>,
    spec: &CaptureSpec,
    output_name: &str,
) -> CaptureOutcome {
    let region = Rectangle::<i32, smithay::utils::Buffer>::new(
        (spec.region.loc.x, spec.region.loc.y).into(),
        (spec.region.size.w, spec.region.size.h).into(),
    );
    let mapping = match gles.copy_framebuffer(target, region, spec.fourcc) {
        Ok(mapping) => mapping,
        Err(err) => {
            warn!(error = %err, output = %output_name, "screencopy: copy_framebuffer failed");
            return CaptureOutcome::Failed;
        }
    };
    let (width, height) = (mapping.width(), mapping.height());
    match gles.map_texture(&mapping) {
        Ok(bytes) => CaptureOutcome::Shm {
            bytes: bytes.to_vec(),
            width,
            height,
        },
        Err(err) => {
            warn!(error = %err, output = %output_name, "screencopy: map_texture failed");
            CaptureOutcome::Failed
        }
    }
}

/// Zero-copy GPU path: bind the client's dmabuf as a framebuffer and
/// blit `spec.region` of the composited output into it. Both src and
/// dst are FBO attachments, so the blit is memory-ordered (see
/// `capture_shm`): `spec.region`'s top-left coordinates index the
/// source directly and the result lands upright in the client's dmabuf
/// (memory-row 0 = top) — no `y_invert` flag needed.
fn capture_dmabuf(
    gles: &mut GlesRenderer,
    target: &GlesTarget<'_>,
    client: &Dmabuf,
    spec: &CaptureSpec,
    output_name: &str,
) -> CaptureOutcome {
    let mut client = client.clone();
    let mut dst = match gles.bind(&mut client) {
        Ok(dst) => dst,
        Err(err) => {
            warn!(error = %err, output = %output_name, "screencopy: bind client dmabuf failed");
            return CaptureOutcome::Failed;
        }
    };
    let src = spec.region;
    let dst_rect = Rectangle::<i32, Physical>::from_size(spec.region.size);
    match gles.blit(target, &mut dst, src, dst_rect, TextureFilter::Linear) {
        Ok(()) => CaptureOutcome::Dmabuf,
        Err(err) => {
            warn!(error = %err, output = %output_name, "screencopy: blit to client dmabuf failed");
            CaptureOutcome::Failed
        }
    }
}

/// Draw one z-band of the backdrop into `scratch.scene`, *accumulating*
/// on top of whatever earlier bands already painted (GLES `render` never
/// clears). The first band's wallpaper fill covers the whole output, so
/// no explicit clear is needed before it.
fn render_scene_stage(
    gles: &mut GlesRenderer,
    scratch: &mut BlurScratch,
    mode_size: Size<i32, Physical>,
    draw: &dyn Fn(&mut GlesFrame<'_, '_>) -> Result<()>,
) -> Result<()> {
    let mut target = gles
        .bind(&mut scratch.scene)
        .context("blur: bind scene buffer")?;
    let mut frame = gles
        .render(&mut target, mode_size, Transform::Normal)
        .context("blur: render scene stage")?;
    draw(&mut frame)?;
    // Same-context sequential GL: the next pass that samples this texture
    // is ordered after these writes, so the fence is dropped.
    let _ = frame.finish().context("blur: finish scene stage")?;
    Ok(())
}

/// Run the Kawase dual-filter pyramid over the current `scratch.scene`
/// (`passes` downsamples then `passes` upsamples) and save the blurred
/// full-resolution result into `scratch.tiers[tier]`.
///
/// All work is render-to-texture (never a raw blit), so orientation stays
/// consistent with the closing-window snapshot path: every pass samples
/// then re-encodes, and the final composite samples once more to land
/// upright on the framebuffer.
fn run_pyramid(
    gles: &mut GlesRenderer,
    scratch: &mut BlurScratch,
    passes: usize,
    radius: f32,
    down: &GlesTexProgram,
    up: &GlesTexProgram,
    tier: usize,
) -> Result<()> {
    // Downsample: scene → level1 → level2 → … → level(passes).
    for k in 1..=passes {
        let src = if k == 1 {
            scratch.scene.clone()
        } else {
            scratch.levels[k - 1].clone()
        };
        blur_pass(gles, &src, &mut scratch.levels[k], down, radius)?;
    }
    // Upsample back up the chain; the last step lands in tiers[tier]
    // (full resolution) instead of overwriting level0, so the saved tier
    // survives the next pyramid run.
    for k in (1..passes).rev() {
        let src = scratch.levels[k + 1].clone();
        blur_pass(gles, &src, &mut scratch.levels[k], up, radius)?;
    }
    let src = scratch.levels[1].clone();
    blur_pass(gles, &src, &mut scratch.tiers[tier], up, radius)?;
    Ok(())
}

/// One Kawase pass: sample `src` (its full extent) into `dst` at `dst`'s
/// own resolution using the blur `program`. `halfpixel` is half a texel
/// of the destination level; `offset` is the configured radius.
fn blur_pass(
    gles: &mut GlesRenderer,
    src: &GlesTexture,
    dst: &mut GlesTexture,
    program: &GlesTexProgram,
    radius: f32,
) -> Result<()> {
    let (dw, dh) = (dst.size().w.max(1), dst.size().h.max(1));
    let phys = Size::<i32, Physical>::from((dw, dh));
    let dst_rect = Rectangle::<i32, Physical>::from_size(phys);
    let st = src.size();
    let src_rect = Rectangle::<f64, smithay::utils::Buffer>::from_size(
        Size::<f64, smithay::utils::Buffer>::from((f64::from(st.w), f64::from(st.h))),
    );
    #[allow(
        clippy::cast_precision_loss,
        reason = "mip dimensions are small positive pixel counts; exact in f32"
    )]
    let halfpixel = (0.5_f32 / dw as f32, 0.5_f32 / dh as f32);
    let uniforms = [
        Uniform::new("halfpixel", halfpixel),
        Uniform::new("offset", radius),
    ];
    let mut target = gles.bind(dst).context("blur: bind mip level")?;
    let mut frame = gles
        .render(&mut target, phys, Transform::Normal)
        .context("blur: render mip level")?;
    frame
        .render_texture_from_to(
            src,
            src_rect,
            dst_rect,
            &[dst_rect],
            // A blur pass fully repaints its destination, so mark the whole
            // rect opaque: smithay then disables blending and *overwrites*
            // the (never-cleared, frame-reused) mip instead of blending the
            // premultiplied result over stale content where the source has
            // any sub-1 alpha.
            &[dst_rect],
            Transform::Normal,
            1.0,
            Some(program),
            &uniforms,
        )
        .context("blur: render_texture_from_to (pass)")?;
    let _ = frame.finish().context("blur: finish pass")?;
    Ok(())
}

/// Convert an sRGB / BT.709 straight colour into the linear BT.2020 HDR
/// working space (1.0 == 10000 cd/m², SDR white at `reference_white`).
///
/// Solid draws (`GlesFrame::draw_solid`) bypass the texture-decode
/// override, so their colours must be converted here when compositing
/// into the fp16 HDR scene. Matches the GLSL decode (column-major
/// BT.709→BT.2020). Alpha is preserved; the solids we draw are opaque or
/// black-translucent, so premultiplication is a no-op.
fn srgb_to_linear_bt2020(color: Color32F, reference_white: u32, saturation: f32) -> Color32F {
    fn eotf(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "reference white is a small cd/m² value, exact in f32"
    )]
    let scale = reference_white as f32 / 10000.0;
    let [r, g, b, a] = color.components();
    let (lr, lg, lb) = (eotf(r) * scale, eotf(g) * scale, eotf(b) * scale);
    // BT.709 → BT.2020 (same coefficients as the GLSL `mat3 * vec`).
    let mut br = 0.627_403_9 * lr + 0.329_283_04 * lg + 0.043_313_06 * lb;
    let mut bg = 0.069_097_29 * lr + 0.919_540_4 * lg + 0.011_362_316 * lb;
    let mut bb = 0.016_391_44 * lr + 0.088_013_31 * lg + 0.895_595_3 * lb;
    // Luma-preserving saturation (matches the decode shaders; 1.0 = identity).
    let luma = 0.2627 * br + 0.6780 * bg + 0.0593 * bb;
    br = (luma + (br - luma) * saturation).max(0.0);
    bg = (luma + (bg - luma) * saturation).max(0.0);
    bb = (luma + (bb - luma) * saturation).max(0.0);
    Color32F::new(br, bg, bb, a)
}

/// Paint `fill` inside the output-local rect `rect`. `Solid` is
/// one `draw_solid` call. `VerticalGradient` walks 256 horizontal
/// stripes spanning the full output height (so the gradient stays
/// continuous with the wallpaper even when only the border edges
/// are being painted); each stripe is clipped to `rect` and
/// skipped if it lies entirely outside, so border edges that
/// only intersect a few stripes don't pay for the rest.
/// Draw the media wallpaper `wp` across one output, fitted per its mode.
/// `Fit`/`Center` don't cover the whole output, so the background is
/// filled black first.
#[allow(
    clippy::cast_possible_truncation,
    reason = "destination pixel sizes are bounded by the output dimensions (i32)"
)]
fn draw_wallpaper_texture(
    frame: &mut GlesFrame<'_, '_>,
    wp: &WpDraw,
    output: Size<i32, Physical>,
    hdr: bool,
    reference_white: u32,
    saturation: f32,
) -> Result<()> {
    let (ow, oh) = (f64::from(output.w), f64::from(output.h));
    let (tw, th) = (f64::from(wp.width.max(1)), f64::from(wp.height.max(1)));
    let full_dst = Rectangle::<i32, Physical>::from_size(output);
    let buf = |x: f64, y: f64, w: f64, h: f64| {
        Rectangle::<f64, smithay::utils::Buffer>::new(Point::from((x, y)), Size::from((w, h)))
    };
    let draw =
        |frame: &mut GlesFrame<'_, '_>,
         src: Rectangle<f64, smithay::utils::Buffer>,
         dst: Rectangle<i32, Physical>|
         -> Result<()> {
            frame
                .render_texture_from_to(
                    &wp.texture,
                    src,
                    dst,
                    &[dst],
                    &[dst],
                    Transform::Normal,
                    1.0,
                    None,
                    &[],
                )
                .context("render_texture_from_to (wallpaper) failed")
        };
    let black = Fill::Solid([0.0, 0.0, 0.0]);
    match wp.mode {
        ScaleMode::Stretch => draw(frame, buf(0.0, 0.0, tw, th), full_dst)?,
        ScaleMode::Fill => {
            // Cover: sample the centred sub-rect of the texture that
            // matches the output aspect, stretched across the full output.
            let scale = (ow / tw).max(oh / th);
            let (vis_w, vis_h) = (ow / scale, oh / scale);
            draw(
                frame,
                buf((tw - vis_w) / 2.0, (th - vis_h) / 2.0, vis_w, vis_h),
                full_dst,
            )?;
        }
        ScaleMode::Fit => {
            draw_fill(frame, &black, output, output, hdr, reference_white, saturation)?;
            let scale = (ow / tw).min(oh / th);
            let (dw, dh) = ((tw * scale) as i32, (th * scale) as i32);
            let dst = Rectangle::new(
                Point::from(((output.w - dw) / 2, (output.h - dh) / 2)),
                Size::from((dw, dh)),
            );
            draw(frame, buf(0.0, 0.0, tw, th), dst)?;
        }
        ScaleMode::Center => {
            draw_fill(frame, &black, output, output, hdr, reference_white, saturation)?;
            // Native size, centred, cropped to the output.
            let (off_x, off_y) = ((output.w - wp.width) / 2, (output.h - wp.height) / 2);
            let (x0, x1) = (off_x.max(0), (off_x + wp.width).min(output.w));
            let (y0, y1) = (off_y.max(0), (off_y + wp.height).min(output.h));
            if x1 > x0 && y1 > y0 {
                let dst =
                    Rectangle::new(Point::from((x0, y0)), Size::from((x1 - x0, y1 - y0)));
                let src = buf(
                    f64::from(x0 - off_x),
                    f64::from(y0 - off_y),
                    f64::from(x1 - x0),
                    f64::from(y1 - y0),
                );
                draw(frame, src, dst)?;
            }
        }
    }
    Ok(())
}

fn draw_fill(
    frame: &mut GlesFrame<'_, '_>,
    fill: &Fill,
    rect: Size<i32, Physical>,
    output_size: Size<i32, Physical>,
    hdr: bool,
    reference_white: u32,
    saturation: f32,
) -> Result<()> {
    draw_fill_rect(
        frame,
        fill,
        Rectangle::<i32, Physical>::from_size(rect),
        output_size,
        hdr,
        reference_white,
        saturation,
    )
}

fn draw_fill_rect(
    frame: &mut GlesFrame<'_, '_>,
    fill: &Fill,
    rect: Rectangle<i32, Physical>,
    output_size: Size<i32, Physical>,
    hdr: bool,
    reference_white: u32,
    saturation: f32,
) -> Result<()> {
    if rect.size.w <= 0 || rect.size.h <= 0 {
        return Ok(());
    }
    // Solid fills bypass the texture-decode override, so convert to the
    // linear BT.2020 working space ourselves when drawing into the HDR scene.
    let conv = |c: Color32F| {
        if hdr {
            srgb_to_linear_bt2020(c, reference_white, saturation)
        } else {
            c
        }
    };
    match fill {
        Fill::Solid(rgb) => {
            let damage = [Rectangle::from_size(rect.size)];
            frame
                .draw_solid(rect, &damage, conv(Color32F::new(rgb[0], rgb[1], rgb[2], 1.0)))
                .context("Frame::draw_solid (fill solid) failed")?;
        }
        Fill::VerticalGradient { top, bottom } => {
            const STRIPE_COUNT: i32 = 256;
            let height = output_size.h.max(1);
            let rect_y_end = rect.loc.y + rect.size.h;
            for stripe in 0u8..=u8::MAX {
                let t = f32::from(stripe) / 255.0;
                let color = conv(Color32F::new(
                    top[0].mul_add(1.0 - t, bottom[0] * t),
                    top[1].mul_add(1.0 - t, bottom[1] * t),
                    top[2].mul_add(1.0 - t, bottom[2] * t),
                    1.0,
                ));

                let idx = i32::from(stripe);
                let stripe_y_start = (idx * height) / STRIPE_COUNT;
                let stripe_y_end = ((idx + 1) * height) / STRIPE_COUNT;
                if stripe_y_end <= rect.loc.y || stripe_y_start >= rect_y_end {
                    continue;
                }
                let clipped_y = stripe_y_start.max(rect.loc.y);
                let clipped_h = stripe_y_end.min(rect_y_end) - clipped_y;
                if clipped_h <= 0 {
                    continue;
                }

                let stripe_dst = Rectangle::<i32, Physical>::new(
                    Point::from((rect.loc.x, clipped_y)),
                    Size::new(rect.size.w, clipped_h),
                );
                let damage = [Rectangle::from_size(stripe_dst.size)];
                frame
                    .draw_solid(stripe_dst, &damage, color)
                    .context("Frame::draw_solid (fill stripe) failed")?;
            }
        }
    }
    Ok(())
}


/// Multiply an i32 by a positive f64 scale and round to the nearest
/// integer. The cast can't truncate in any practical case: input is
/// bounded by i32 cell coords and scale is configured-positive.
#[allow(
    clippy::cast_possible_truncation,
    reason = "compositor coordinates are bounded by total display dimensions; scale * coord stays within i32 with room to spare"
)]
fn scale_i(v: i32, scale: f64) -> i32 {
    (f64::from(v) * scale).round() as i32
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "cursor coords are clamped to layout_bounds (i32) in on_pointer_motion; scale * coord stays within i32"
)]
fn scale_f(v: f64, scale: f64) -> i32 {
    (v * scale).round() as i32
}

/// Paint the screenshot selection UI onto one output: a translucent dim
/// wash over everything outside the selection (or the whole output when
/// the selection is elsewhere / not started yet), plus a bright outline
/// around the selection. `selection` is in absolute compositor coords;
/// it's converted to this output's physical pixels and clipped to it.
#[allow(
    clippy::too_many_arguments,
    reason = "geometry + HDR colour params (hdr/reference_white/saturation); a struct would just move the noise"
)]
fn draw_screenshot_overlay(
    frame: &mut GlesFrame<'_, '_>,
    overlay: &ScreenshotOverlay,
    compositor_position: Point<i32, Physical>,
    mode_size: Size<i32, Physical>,
    scale: f64,
    hdr: bool,
    reference_white: u32,
    saturation: f32,
) -> Result<()> {
    const DIM: Color32F = Color32F::new(0.0, 0.0, 0.0, 0.45);
    const OUTLINE: Color32F = Color32F::new(0.25, 0.62, 1.0, 1.0);
    let (mode_w, mode_h) = (mode_size.w, mode_size.h);

    let solid = |frame: &mut GlesFrame<'_, '_>, x: i32, y: i32, w: i32, h: i32, color: Color32F| {
        if w <= 0 || h <= 0 {
            return Ok(());
        }
        // draw_solid bypasses the decode override → convert for the HDR scene.
        let color = if hdr {
            srgb_to_linear_bt2020(color, reference_white, saturation)
        } else {
            color
        };
        let rect = Rectangle::<i32, Physical>::new(Point::from((x, y)), Size::from((w, h)));
        frame
            .draw_solid(rect, &[Rectangle::from_size(rect.size)], color)
            .context("Frame::draw_solid (screenshot overlay) failed")
    };

    // The selection rect in this output's physical pixels, clipped to the
    // output. `None`/no-intersection => dim the entire output.
    let clip = overlay.selection.and_then(|sel| {
        let sx = scale_i(sel.loc.x - compositor_position.x, scale);
        let sy = scale_i(sel.loc.y - compositor_position.y, scale);
        let x0 = sx.clamp(0, mode_w);
        let y0 = sy.clamp(0, mode_h);
        let x1 = (sx + scale_i(sel.size.w, scale)).clamp(0, mode_w);
        let y1 = (sy + scale_i(sel.size.h, scale)).clamp(0, mode_h);
        (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
    });

    let Some((x0, y0, x1, y1)) = clip else {
        // No selection on this output: dim it whole.
        return solid(frame, 0, 0, mode_w, mode_h, DIM);
    };

    // Dim everything except the selection (four bands).
    solid(frame, 0, 0, mode_w, y0, DIM)?; // top
    solid(frame, 0, y1, mode_w, mode_h - y1, DIM)?; // bottom
    solid(frame, 0, y0, x0, y1 - y0, DIM)?; // left
    solid(frame, x1, y0, mode_w - x1, y1 - y0, DIM)?; // right

    // Bright outline framing the selection.
    let t = scale_i(2, scale).max(2);
    let (w, h) = (x1 - x0, y1 - y0);
    solid(frame, x0, y0, w, t, OUTLINE)?; // top edge
    solid(frame, x0, y1 - t, w, t, OUTLINE)?; // bottom edge
    solid(frame, x0, y0, t, h, OUTLINE)?; // left edge
    solid(frame, x1 - t, y0, t, h, OUTLINE)?; // right edge
    Ok(())
}

/// Walk a surface tree and drain every queued `wl_callback`, firing
/// `done(time_ms)` on each so the client knows to schedule its next
/// frame. Smithay's `desktop::send_frames_surface_tree` does this
/// plus primary-scanout-output filtering and throttling, all of
/// which presuppose a `Space<Window>` we don't have yet (4d); this
/// minimal version is enough for 4b — every visible surface gets a
/// callback per vblank cycle.
fn send_frame_callbacks(surface: &WlSurface, time_ms: u32) {
    with_surface_tree_downward(
        surface,
        (),
        |_, _, &()| TraversalAction::DoChildren(()),
        |_surf, states, &()| {
            let mut attrs = states.cached_state.get::<SurfaceAttributes>();
            for callback in attrs.current().frame_callbacks.drain(..) {
                callback.done(time_ms);
            }
        },
        |_, _, &()| true,
    );
}

/// Read a toplevel's `xdg_surface.set_window_geometry` origin, in
/// compositor (logical) pixels. CSD clients set this to the top-left
/// of their visible window inside a larger, shadow-padded buffer;
/// returns `(0, 0)` when the client never set a geometry (e.g. SSD
/// apps with no shadow). Returned as a raw `(i32, i32)` so the caller
/// can fold it straight into the compositor-pixel position math
/// without juggling the `Logical`/`Physical` unit tags.
pub(crate) fn window_geometry_offset(surface: &WlSurface) -> (i32, i32) {
    with_states(surface, |states| {
        states
            .cached_state
            .get::<SurfaceCachedState>()
            .current()
            .geometry
            .map_or((0, 0), |g| (g.loc.x, g.loc.y))
    })
}

/// The surface's current visible content size (`set_window_geometry`),
/// in compositor pixels, if the client set one and it's non-degenerate.
/// Used as the denominator when scaling a window's *actual* buffer to its
/// animated rect — so a resize stays correct even while the client is a
/// frame or two behind reconfiguring.
fn window_geometry_size(surface: &WlSurface) -> Option<(i32, i32)> {
    with_states(surface, |states| {
        states
            .cached_state
            .get::<SurfaceCachedState>()
            .current()
            .geometry
            .map(|g| (g.size.w, g.size.h))
            .filter(|&(w, h)| w > 0 && h > 0)
    })
}

/// Interpolate every component of two rects by eased `t`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "interpolated pixel coordinates are bounded by output size, well within i32"
)]
fn lerp_rect(
    a: Rectangle<i32, Physical>,
    b: Rectangle<i32, Physical>,
    t: f64,
) -> Rectangle<i32, Physical> {
    Rectangle::new(
        Point::from((
            lerp(f64::from(a.loc.x), f64::from(b.loc.x), t).round() as i32,
            lerp(f64::from(a.loc.y), f64::from(b.loc.y), t).round() as i32,
        )),
        Size::from((
            lerp(f64::from(a.size.w), f64::from(b.size.w), t).round() as i32,
            lerp(f64::from(a.size.h), f64::from(b.size.h), t).round() as i32,
        )),
    )
}

/// Shrink/grow a rect about its centre by factor `s` (keeps the centre
/// fixed) — the geometry of an open/close scale-in/out.
#[allow(
    clippy::cast_possible_truncation,
    reason = "scaled pixel coordinates are bounded by output size, well within i32"
)]
fn scale_rect_about_center(r: Rectangle<i32, Physical>, s: f64) -> Rectangle<i32, Physical> {
    let cx = f64::from(r.loc.x) + f64::from(r.size.w) / 2.0;
    let cy = f64::from(r.loc.y) + f64::from(r.size.h) / 2.0;
    let w = f64::from(r.size.w) * s;
    let h = f64::from(r.size.h) * s;
    Rectangle::new(
        Point::from(((cx - w / 2.0).round() as i32, (cy - h / 2.0).round() as i32)),
        Size::from((w.round() as i32, h.round() as i32)),
    )
}

/// Draw the pointer with its hotspot at `hotspot` (this output's
/// physical pixels). When an `XCursor` theme loaded, render its sprite;
/// otherwise fall back to the built-in white right-triangle so the
/// pointer is always visible.
///
/// `cursor_size` is the requested logical size; `scale` is this
/// output's fractional scale. The themed sprite is scaled by
/// `cursor_size / nominal * scale` so it lands at the requested
/// logical size in physical pixels no matter which image the theme
/// supplied, with the hotspot offset scaled to match.
#[allow(
    clippy::too_many_arguments,
    reason = "cursor geometry + HDR colour params (hdr/reference_white/saturation)"
)]
fn draw_cursor(
    frame: &mut GlesFrame<'_, '_>,
    sprite: Option<&CursorSprite>,
    cursor_size: i32,
    hotspot: Point<i32, Physical>,
    scale: f64,
    hdr: bool,
    reference_white: u32,
    saturation: f32,
) -> Result<()> {
    if let Some(sprite) = sprite {
        // Image px → physical px: normalise to the requested logical
        // size, then apply the output scale.
        let factor = f64::from(cursor_size) / f64::from(sprite.nominal) * scale;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "cursor dimensions and scale factor are small positive values; products stay well within i32"
        )]
        let (dst_w, dst_h, off_x, off_y) = (
            (f64::from(sprite.width) * factor).round() as i32,
            (f64::from(sprite.height) * factor).round() as i32,
            (f64::from(sprite.xhot) * factor).round() as i32,
            (f64::from(sprite.yhot) * factor).round() as i32,
        );
        // Position the sprite so its hotspot sits on the pointer.
        let origin = Point::<i32, Physical>::from((hotspot.x - off_x, hotspot.y - off_y));
        let dst = Rectangle::new(origin, Size::new(dst_w.max(1), dst_h.max(1)));
        let src = Rectangle::from_size(sprite.texture.size()).to_f64();
        let damage = [Rectangle::from_size(dst.size)];
        frame
            .render_texture_from_to(
                &sprite.texture,
                src,
                dst,
                &damage,
                // Cursors have transparent regions: no opaque hint, and
                // the renderer's premultiplied-alpha blend handles the
                // edges.
                &[],
                Transform::Normal,
                1.0,
                // No custom shader override; default texture program
                // with no extra uniforms.
                None,
                &[],
            )
            .context("render_texture_from_to (cursor) failed")?;
        return Ok(());
    }

    // Fallback: built-in white right-triangle, apex at the hotspot.
    // Damage stripes are anchored at `(0, row)` relative to `dst.loc`
    // — see the long note in milestone 2c about `Frame::draw_solid`'s
    // damage-coordinate semantics.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "CURSOR_SIZE is 24 and scale is bounded; product stays in i32"
    )]
    let size = ((f64::from(CURSOR_SIZE) * scale).round() as i32).max(1);
    let cursor_bbox = Rectangle::new(hotspot, Size::new(size, size));
    let cursor_damage: Vec<Rectangle<i32, Physical>> = (0..size)
        .map(|row| Rectangle::new(Point::from((0, row)), Size::new(row + 1, 1)))
        .collect();
    let white = Color32F::new(1.0, 1.0, 1.0, 1.0);
    let color = if hdr {
        srgb_to_linear_bt2020(white, reference_white, saturation)
    } else {
        white
    };
    frame
        .draw_solid(cursor_bbox, &cursor_damage, color)
        .context("Frame::draw_solid (cursor) failed")?;
    Ok(())
}
