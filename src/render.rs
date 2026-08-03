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
use std::collections::VecDeque;
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
    CommitCounter, RendererSurfaceStateUserData, draw_render_elements,
    with_renderer_surface_state,
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
use smithay::wayland::shell::xdg::{SurfaceCachedState, XdgToplevelSurfaceData};
use tracing::{debug, info, warn};

use crate::anim::{Animation, lerp};
use libreland_text::Fonts;

use crate::config::{
    AnimSpec, AnimationsConfig, BlurConfig, BorderConfig, DecorationConfig, Fill, MonitorsConfig,
    ScaleMode, TearingMode, TitlebarConfig, VrrMode,
};
use crate::color_management::Encoding;
use crate::drm::DrmOutput;
use crate::layout::{FillMode, Placement, ZBand};
use crate::titlebar::{BarState, BarStyle, rasterize as rasterize_bar};
use crate::scanout::{DirectPlacement, ScanoutLayer, ScanoutSurface};

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

/// Whether the client asked for backdrop blur behind this surface via
/// `ext-background-effect-v1` (a committed blur region on the root
/// surface). The region's rects are ignored — Libreland's blur is
/// already masked by the surface's own alpha (layers) or clipped to the
/// window shape, which is strictly finer than rect masking.
fn surface_requests_blur(surface: &WlSurface) -> bool {
    smithay::wayland::compositor::with_states(surface, |states| {
        states
            .cached_state
            .get::<smithay::wayland::background_effect::BackgroundEffectSurfaceCachedState>()
            .current()
            .blur_region
            .as_ref()
            // An explicitly EMPTY region means "blur nowhere" per the
            // protocol — only a region with actual rects opts in.
            .is_some_and(|r| !r.rects.is_empty())
    })
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

/// What a composited frame is *for*.
///
/// A frame is normally rendered to be shown: it ends in a page flip, and
/// the frame callbacks, presentation feedback and adaptive-sync state that
/// go with one. A capture-only frame is composited and read back and then
/// thrown away — which is what lets a workspace nobody is looking at be
/// photographed exactly as it would appear if you switched to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FramePurpose {
    /// Compose and present.
    Present,
    /// Compose, service the captures, and stop. Nothing is flipped, no
    /// client is told it was displayed, and the output's damage history is
    /// invalidated afterwards — the swapchain slot now holds a frame that
    /// was never shown, so the next real frame has to repaint in full.
    CaptureOnly,
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
/// How far, in physical pixels, the temporal blur veto is dilated before it
/// can suppress a pixel (see `prev_coverage` in [`MASK_BLUR_SHADER`]).
///
/// It has to exceed how far a client's content can travel between two frames,
/// or the leading edge of a sliding popup is vetoed and left un-frosted. The
/// worst case is the first frame of a decelerating slide: an ease-out cubic
/// leaves at 3x its average speed, so a card crossing a 1440 px display in
/// 300 ms peaks near 14 px/ms — about 87 px on a 60 Hz frame. 128 px covers
/// that with margin.
///
/// Erring large is close to free: the current mask still confines the frost
/// exactly, so a wider radius cannot smear it, and the only cost is that a
/// transient full-surface fill keeps a 128 px halo around last frame's card
/// instead of being suppressed outright — invisible next to the full-screen
/// flash this guard exists to stop.
const MASK_DILATE_PX: f32 = 128.0;

const MASK_BLUR_SHADER: &str = r"#version 100
//_DEFINES_
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

uniform sampler2D tex;
uniform sampler2D mask;
uniform sampler2D mask_prev;
uniform float alpha;
uniform vec2 mask_mul;
uniform vec2 mask_add;
uniform vec2 mask_dilate;

varying vec2 v_coords;

// Was this pixel covered by the surface *near here* last frame? A straight
// point sample assumes the client's content is stationary; a popup that
// slides into place lands on new pixels every frame, so the leading edge of
// a moving card would be vetoed and left un-frosted (it reads as the frost
// tearing along the card as it travels). Dilating the veto by `mask_dilate`
// lets content that merely *moved* still count as persistent.
//
// This cannot smear the frost: `min` with the current mask still confines it
// to exactly what the client draws this frame. Dilation only relaxes the
// veto, so the sole cost is that a transient full-surface fill keeps its
// frost within `mask_dilate` of last frame's card — a small halo instead of
// a full-screen flash. Taps at half and full radius so the reach is granular
// enough for a card thinner than the radius (a notification toast).
float prev_coverage(vec2 uv) {
    vec2 d = mask_dilate;
    vec2 h = d * 0.5;
    float p = texture2D(mask_prev, uv).a;
    p = max(p, texture2D(mask_prev, clamp(uv + vec2(h.x, 0.0), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv - vec2(h.x, 0.0), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv + vec2(0.0, h.y), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv - vec2(0.0, h.y), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv + vec2(d.x, 0.0), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv - vec2(d.x, 0.0), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv + vec2(0.0, d.y), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv - vec2(0.0, d.y), 0.0, 1.0)).a);
    return p;
}

void main() {
    vec4 c = texture2D(tex, v_coords);
    vec2 muv = v_coords * mask_mul + mask_add;
    // Temporal coverage: a real panel (the bar, a popup's card) covers the
    // same pixels frame after frame; a client's transient full-surface frame
    // (Qt paints one panel-coloured frame across the whole popup surface when
    // it maps, before its content settles to just the card) covers a pixel on
    // *one* frame only. Take the min of this frame's and last frame's surface
    // alpha, so newly-covered pixels get no frost until they persist — that
    // one bad frame can no longer flash the whole screen, and it needs no
    // guess about the alpha (the flash is the same 0.79 as the real card).
    float m = min(texture2D(mask, muv).a, prev_coverage(muv));
    // Coverage -> frost strength: a translucent panel body (~0.76-0.79) must
    // still get the full blur behind it, else its transparency shows the
    // sharp backdrop. Saturate so any meaningfully-covered pixel is fully
    // frosted and only the shape's AA edge blends out.
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
uniform sampler2D mask_prev;
uniform float alpha;
uniform vec2 mask_mul;
uniform vec2 mask_add;
uniform vec2 mask_dilate;
uniform float reference_white;
uniform float saturation;
varying vec2 v_coords;
// Dilated temporal veto — see MASK_BLUR_SHADER's prev_coverage.
float prev_coverage(vec2 uv) {
    vec2 d = mask_dilate;
    vec2 h = d * 0.5;
    float p = texture2D(mask_prev, uv).a;
    p = max(p, texture2D(mask_prev, clamp(uv + vec2(h.x, 0.0), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv - vec2(h.x, 0.0), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv + vec2(0.0, h.y), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv - vec2(0.0, h.y), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv + vec2(d.x, 0.0), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv - vec2(d.x, 0.0), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv + vec2(0.0, d.y), 0.0, 1.0)).a);
    p = max(p, texture2D(mask_prev, clamp(uv - vec2(0.0, d.y), 0.0, 1.0)).a);
    return p;
}
vec3 srgb_to_linear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}
void main() {
    vec4 c = texture2D(tex, v_coords);
    vec2 muv = v_coords * mask_mul + mask_add;
    // Temporal coverage min + saturation — see MASK_BLUR_SHADER.
    float m = min(texture2D(mask, muv).a, prev_coverage(muv));
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

/// Draw anti-aliased line segments over a quad — the compositor's whole
/// vocabulary for strokes, on the GPU.
///
/// Every glyph the screenshot toolbar draws (a tick, a pencil, a T, an X)
/// is a short list of segments, and so is every freehand annotation
/// stroke, so both go through this one program instead of a CPU
/// rasteriser and an upload. Coordinates are in destination pixels with
/// the origin at the quad's top-left, which keeps the caller's arithmetic
/// in the units it already thinks in.
///
/// The segment count is fixed at compile time because GLES 2.0 requires
/// constant loop bounds; `count` masks the unused tail. Longer strokes
/// are drawn as several quads.
const SEGMENTS_MAX: usize = 12;

const SEGMENT_SHADER: &str = r"#version 100
//_DEFINES_
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D tex;
uniform float alpha;
uniform vec4 segments[12];
uniform int count;
uniform vec3 colour;
uniform float thickness;
uniform vec2 quad;
varying vec2 v_coords;

// Distance from `p` to the segment `a`-`b`, the standard projection with
// the parameter clamped to the segment so the ends are round rather than
// running off to infinity.
float seg_dist(vec2 p, vec2 a, vec2 b) {
    vec2 ab = b - a;
    float len2 = max(dot(ab, ab), 1e-6);
    float t = clamp(dot(p - a, ab) / len2, 0.0, 1.0);
    return distance(p, a + t * ab);
}

void main() {
    vec2 p = v_coords * quad;
    float d = 1.0e9;
    for (int i = 0; i < 12; i++) {
        if (i >= count) { break; }
        d = min(d, seg_dist(p, segments[i].xy, segments[i].zw));
    }
    // One pixel of feather either side of the stroke's half-width: enough
    // to kill the staircase, narrow enough that a 2 px stroke still reads
    // as 2 px rather than as a smudge.
    float half_w = thickness * 0.5;
    float cov = 1.0 - smoothstep(half_w - 0.5, half_w + 0.5, d);
    // The sampler is bound to a 1x1 opaque white texture and contributes
    // nothing. It is here because smithay's texture programs require the
    // uniform to exist, and a driver that optimises away an unreferenced
    // sampler makes the program fail to build.
    cov *= texture2D(tex, v_coords).a;
    gl_FragColor = vec4(colour * cov, cov) * alpha;
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
uniform float knee;
varying vec2 v_coords;
vec3 linear_to_srgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}
// Highlight shoulder: identity below `knee`, then a rational roll-off joining
// it at equal value and slope and approaching 1.0 asymptotically. Nothing is
// ever clipped, so highlights keep their ordering and their detail instead of
// collapsing into one flat white.
float shoulder(float v, float k) {
    float d = 1.0 - k;
    float rolled = 1.0 - (d * d) / max(v - 2.0 * k + 1.0, 1e-5);
    return mix(v, rolled, step(k, v));
}
void main() {
    vec4 premult = texture2D(tex, v_coords);
    vec3 bt2020 = premult.rgb / max(premult.a, 0.001);
    // Renormalise so SDR diffuse white lands on 1.0. The scene stores
    // nits/10000, so a 1000-nit HDR highlight arrives here at ~4.9 -- which is
    // the whole problem: it has to survive the trip into a range that stops
    // at 1.0 without simply becoming white.
    bt2020 *= (10000.0 / reference_white);

    mat3 bt2020_to_bt709 = mat3(
        1.660491, -0.124550, -0.018151,
        -0.587641, 1.132900, -0.100579,
        -0.072850, -0.008349, 1.118730
    );
    vec3 lin = bt2020_to_bt709 * bt2020;

    // BT.2020 can express colours BT.709 cannot, and those land with a
    // negative channel. Desaturate toward the pixel's own luminance by exactly
    // as much as it takes to bring them back, which holds the hue; clamping
    // the channel to zero would swing it instead. Only the gamut is fixed
    // here -- brightness is the shoulder's job, below.
    float l709 = max(dot(lin, vec3(0.2126, 0.7152, 0.0722)), 0.0);
    float lo = min(min(lin.r, lin.g), lin.b);
    float s = mix(1.0, clamp(l709 / max(l709 - lo, 1e-5), 0.0, 1.0), step(lo, -1e-6));
    lin = max(mix(vec3(l709), lin, s), vec3(0.0));

    // Roll highlights off by compressing the *peak* channel and scaling the
    // whole colour by that one ratio, so chromaticity survives exactly.
    //
    // Running the curve per channel instead -- the obvious way, and what this
    // did first -- compresses the brightest channel hardest, which squeezes
    // the ratios between channels and bleaches bright colour toward white. A
    // sunlit orange came out (252, 242, 128), pale yellow, where this keeps it
    // (250, 140, 52). Measured over a colour held at constant hue while its
    // brightness rises, per-channel let saturation fall 0.69 -> 0.58 -> 0.42;
    // scaling by the peak holds it at 0.79 throughout.
    //
    // No highlight-bleach term on purpose. Real film and the eye do desaturate
    // very bright colour, and the usual formulation blends toward the
    // compressed peak -- but it is strong enough to undo most of what this
    // buys (0.05 already drags that ladder back to 0.67/0.55/0.42), and a
    // faithful capture is the goal here, not a photographic look.
    float peak = max(max(lin.r, lin.g), lin.b);
    lin *= mix(1.0, shoulder(peak, knee) / max(peak, 1e-5), step(knee, peak));

    // Re-premultiply and keep the source alpha. An output capture is opaque
    // everywhere (the wallpaper fill covers it), so this is the identity
    // there; a *window* capture has transparent pixels outside the client's
    // own shape, and forcing them to 1.0 would fill the thumbnail's corners.
    vec3 srgb = linear_to_srgb(clamp(lin, 0.0, 1.0));
    gl_FragColor = vec4(srgb * premult.a, premult.a) * alpha;
}
";

/// Where the screenshot tone curve leaves linear response and starts rolling
/// highlights off, in units of SDR diffuse white (see `shoulder` in
/// [`SCREENSHOT_TONEMAP_SHADER`]).
///
/// The trade is forced by arithmetic, not taste: an SDR screenshot tops out at
/// 1.0, so representing *any* headroom above diffuse white means diffuse white
/// itself has to sit below 1.0.
///
/// Picked by measuring rather than by feel. Against an ACES filmic fit this
/// curve holds mid-tones *exactly* (0.464, identical to no tone mapping at
/// all), where ACES lifts them to 0.557 and would visibly distort an ordinary
/// SDR desktop capture, which goes through this same path.
///
/// 0.50 rather than the 0.60 first shipped: at 0.60 the highlights read a
/// little hot against the display. It moves the neutral ladder from
/// 231/245/251 down to 225/240/249 for 203/400/1000 cd/m², which is the
/// "slightly too bright" complaint's worth of change, and leaves mid-tones
/// untouched either way since they never reach the knee.
///
/// Raise it toward 1.0 for brighter SDR whites and harsher highlight
/// compression, lower it for more highlight separation and a dimmer desktop.
const SCREENSHOT_TONEMAP_KNEE: f32 = 0.50;

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

/// Decode a Windows-scRGB source (Wine/Proton tags a game's scRGB swapchain
/// with the protocol's pre-defined `windows_scrgb` description and passes the
/// pixels through untouched) into the linear BT.2020 working space.
///
/// Per the protocol: scRGB is **already linear light** on BT.709 primaries,
/// R=G=B=1.0 is 80 cd/m² and R=G=B=125.0 is the 10000 cd/m² PQ peak. The
/// working space is linear BT.2020 normalised to 1.0 == 10000 cd/m², so the
/// whole luminance mapping is exactly `/125`. Critically there is **no EOTF**
/// (the data is not gamma-encoded) and **no reference-white scaling** (scRGB
/// carries its own 80 cd/m² anchor) — running scRGB through the SDR decode
/// instead applies an sRGB curve to linear data, anchors it at the wrong
/// luminance, and NaNs every negative channel (`pow()` of a negative is
/// undefined, and `mix(lo, hi, 0.0)` still yields NaN because `0 * NaN =
/// NaN`) — which is what made id Tech (DOOM) titles render wrong.
const SCRGB_DECODE_SHADER: &str = r"#version 100
//_DEFINES_
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D tex;
uniform float alpha;
varying vec2 v_coords;
void main() {
    vec4 premult = texture2D(tex, v_coords);
    vec3 straight = premult.rgb / max(premult.a, 0.001);
    vec3 lin = straight / 125.0;
    mat3 bt709_to_bt2020 = mat3(
        0.627403896, 0.069097289, 0.016391439,
        0.329283038, 0.919540395, 0.088013308,
        0.043313066, 0.011362316, 0.895595253
    );
    // Negative channels are scRGB escaping the sRGB gamut. The matrix maps
    // them into BT.2020, where most land back in range; clamp only AFTER it,
    // so wide-gamut colour survives instead of being crushed at the source.
    vec3 bt2020 = max(bt709_to_bt2020 * lin, vec3(0.0));
    gl_FragColor = vec4(bt2020 * premult.a, premult.a) * alpha;
}
";

/// The scRGB counterpart of [`SDR_TO_PQ_SHADER`]: decode → BT.2020 → PQ OETF
/// in one fragment for the single-pass fast path, so a solo fullscreen scRGB
/// game (DOOM et al) costs exactly what it did before — one full-output pass,
/// no fp16 scene buffer — just with the correct maths. scRGB can never take
/// direct scanout (its pixels are linear, the display wants PQ), so this
/// fused program is that path's floor.
const SCRGB_TO_PQ_SHADER: &str = r"#version 100
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
    vec3 straight = premult.rgb / max(premult.a, 0.001);
    vec3 lin = straight / 125.0;
    mat3 bt709_to_bt2020 = mat3(
        0.627403896, 0.069097289, 0.016391439,
        0.329283038, 0.919540395, 0.088013308,
        0.043313066, 0.011362316, 0.895595253
    );
    vec3 bt2020 = max(bt709_to_bt2020 * lin, vec3(0.0));
    gl_FragColor = vec4(pq_oetf(bt2020), 1.0) * alpha;
}
";

/// Decode an HDR PQ / BT.2020 source (a colour-managed client's buffer)
/// into the linear BT.2020 working space. Primaries already match and the
/// PQ EOTF lands in the 1.0 == 10000 cd/m² domain, so no rescale.
/// R↔B-swizzling variant of [`HDR_DECODE_SHADER`], for PQ client buffers in
/// the XRGB/ARGB channel order (XR30/AR30 and fp16 ARGB). NVIDIA's Wayland
/// HDR10 swapchains allocate XR30, and its EGL dmabuf import hands GLES the
/// components in buffer order rather than swizzling to RGBA — sampled
/// naively, red and blue trade places and the Slayer's green armour turns
/// turquoise. The plane consumes these buffers natively (fourcc-aware), so
/// only the GL-sampled paths need this.
const HDR_DECODE_SWIZZLE_SHADER: &str = r"#version 100
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
    vec3 straight = premult.bgr / max(premult.a, 0.001);
    vec3 lin = pq_eotf(straight);
    gl_FragColor = vec4(lin * premult.a, premult.a) * alpha;
}
";

/// R↔B-swizzling identity copy for the single-pass PQ passthrough (see
/// [`HDR_DECODE_SWIZZLE_SHADER`] for why): the solo fullscreen HDR game's
/// pixels are already exactly what the PQ scanout wants, except sampled in
/// buffer order — reorder the channels, touch nothing else.
const PQ_PASSTHROUGH_SWIZZLE_SHADER: &str = r"#version 100
//_DEFINES_
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D tex;
uniform float alpha;
varying vec2 v_coords;
void main() {
    vec4 c = texture2D(tex, v_coords);
    gl_FragColor = vec4(c.b, c.g, c.r, c.a) * alpha;
}
";

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

/// Which decode each colour-managed surface needs this frame. A surface in
/// neither set is plain SDR and takes the renderer's default path.
///
/// Split rather than a single "is HDR" flag because the two HDR encodings are
/// not interchangeable: PQ pixels are already what a PQ output wants (so they
/// stay eligible for direct scanout and the single-pass passthrough), while
/// scRGB is linear light that *must* be converted by a shader first.
#[derive(Debug, Default)]
pub struct SurfaceEncodings {
    /// PQ / BT.2100-tagged surfaces. Passthrough-compatible on a PQ output.
    pub pq: HashSet<ObjectId>,
    /// Windows-scRGB-tagged surfaces. Never passthrough, never scanout.
    pub scrgb: HashSet<ObjectId>,
}

impl SurfaceEncodings {
    /// Any colour-managed tag — i.e. needs the linear/fp16 treatment rather
    /// than the sRGB default.
    pub fn is_managed(&self, id: &ObjectId) -> bool {
        self.pq.contains(id) || self.scrgb.contains(id)
    }
}

/// Renderer for every connected output on a single GPU.
/// Which windows skip the move animation while an interactive drag is
/// in flight. A dragged window's rect changes every frame to track the
/// cursor, so it must draw 1:1 or it visibly trails the pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
enum NoAnim {
    /// No drag: everything animates normally.
    None,
    /// A move drag — only the dragged window snaps; anything the layout
    /// reflows around it should still animate.
    One(ObjectId),
    /// A resize drag — the dragged window *and* every neighbour it
    /// reflows must snap, or the edge they share trails the divider.
    All,
}

impl NoAnim {
    /// Whether the window with this id must draw at its target rect now.
    fn covers(&self, id: &ObjectId) -> bool {
        match self {
            Self::None => false,
            Self::One(dragged) => dragged == id,
            Self::All => true,
        }
    }
}

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
    /// Where a window under an interactive move would land if dropped
    /// now (quick-tile), in absolute compositor coords. `None` when no
    /// snap is armed — which is most of any drag.
    snap_preview: Option<Rectangle<i32, Physical>>,
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
    ///
    /// The flag is "this is a *restore*, so rise into place" (see
    /// [`Self::mark_restore`]); a plain map is `false` and scales in where
    /// it stands.
    pending_open: HashMap<ObjectId, bool>,
    /// Which windows skip the move animation while an interactive drag
    /// is in flight (see [`NoAnim`]). Cleared on drop, which lets them
    /// animate into their final tiles again.
    no_anim: NoAnim,
    /// Tearing (async page-flip) policy, from `misc.tearing`. Live-reloaded
    /// via [`Self::set_tearing_mode`].
    tearing: TearingMode,
    /// Surfaces that asked for immediate presentation through
    /// `wp_tearing_control_v1`. Consulted under [`TearingMode::Auto`].
    tearing_hints: HashSet<ObjectId>,
    /// Windows mid close-animation: a snapshot texture taken the moment
    /// the toplevel was destroyed, fading + shrinking out where the
    /// window last sat. Drained as each finishes.
    closing: Vec<ClosingWindow>,
    /// Layer surfaces that just mapped and should play an open animation
    /// the next frame they appear. Mirrors [`Self::pending_open`].
    pending_layer_open: HashSet<ObjectId>,
    /// In-flight layer open animations, by surface id. Entries are dropped
    /// when they finish or the surface goes away.
    layer_anims: HashMap<ObjectId, Animation>,
    /// Layer surfaces mid close-animation, drained as each finishes.
    closing_layers: Vec<ClosingLayer>,
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
    /// Anti-aliased stroke program: toolbar glyphs and annotation.
    segment_shader: GlesTexProgram,
    /// 1x1 opaque white, bound as the sampler for programs that draw
    /// procedurally and never read a texture (see [`SEGMENT_SHADER`]).
    blank_tex: GlesTexture,
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
    /// [`HDR_DECODE_SWIZZLE_SHADER`] — PQ decode for XRGB-order buffers.
    hdr_decode_swizzle_shader: GlesTexProgram,
    /// [`PQ_PASSTHROUGH_SWIZZLE_SHADER`] — identity + R↔B for the
    /// single-pass passthrough of XRGB-order PQ buffers.
    pq_passthrough_swizzle_shader: GlesTexProgram,
    /// Decodes a Windows-scRGB source into the linear working space; swapped
    /// in around scRGB-tagged surfaces. See [`SCRGB_DECODE_SHADER`].
    scrgb_decode_shader: GlesTexProgram,
    /// Fused scRGB decode → PQ encode for the single-pass fast path (a solo
    /// fullscreen scRGB game). See [`SCRGB_TO_PQ_SHADER`].
    scrgb_to_pq_shader: GlesTexProgram,
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
    /// Cached decoration offscreens (`win_tex`) per window, re-rendered
    /// only when the surface tree committed new content or the cell
    /// resized — previously every decorated window paid a fresh texture
    /// allocation AND a full redraw every frame, idle or not. Evicted on
    /// surface destroy ([`Renderer::forget_surface`]) and on size/format
    /// change. `LIBRELAND_NO_WINTEX_CACHE=1` disables for A/B runs.
    wintex_cache: HashMap<ObjectId, WinTexCache>,
    /// Titlebar configuration (height, font size, buttons). Applied live
    /// on reload like the rest of the appearance settings.
    titlebar: TitlebarConfig,
    /// UI faces for titlebar text. Loading is deferred to the first bar
    /// that needs one rather than done at startup — walking every font
    /// directory is not something a tiling session with no titlebars
    /// should pay for — and [`FontState`] makes the scan run exactly
    /// once even when it comes back empty.
    fonts: FontState,
    /// Rasterized titlebars by [`bar_key`]. Small (one bar-sized RGBA
    /// texture per distinct title/size/focus), and bounded — a window
    /// whose title animates would otherwise grow this without limit.
    bar_cache: HashMap<u64, GlesTexture>,
    /// The titlebar button the pointer is currently over, if any. Set by
    /// the input path, which already resolves the region on every motion
    /// for the resize cursor.
    hovered_button: Option<(ObjectId, crate::config::TitlebarButton)>,
    /// Application icons by `(app_id, side)`, already decoded and scaled
    /// to the slot. `None` records a lookup that found nothing, so a
    /// window whose app ships no PNG icon doesn't rescan every icon
    /// theme on every retitle. Cleared when the titlebar config changes,
    /// since the slot size moves with the bar height.
    icon_cache: HashMap<(String, u32), Option<std::sync::Arc<crate::icon::Icon>>>,
    /// `LIBRELAND_NO_OCCLUSION=1`: disable the occluded/off-output window
    /// prune, for A/B benchmarking against the render-profile log.
    no_occlusion: bool,
    /// `LIBRELAND_NO_WINTEX_CACHE=1`: disable the decoration offscreen
    /// cache, for A/B benchmarking.
    no_wintex_cache: bool,
    /// `LIBRELAND_NO_DAMAGE=1`: paint every frame in full, for A/B
    /// benchmarking against the render-profile log.
    no_damage: bool,
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
const BLUR_TIERS: usize = 4;

/// How many *extra* tiers may be handed out to individual floating
/// windows, so a translucent window frosts the translucent window beneath
/// it rather than the desktop they both sit on.
///
/// Capped because each tier is a full-resolution texture — ~33 MB at 4K —
/// and because each one costs its own pyramid. Past the cap the remaining
/// windows share the deepest tier built, which is what every window did
/// before this existed. Allocated on demand: a desktop that never
/// overlaps two translucent windows never pays for any of them.
const MAX_WINDOW_TIERS: usize = 6;

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
    /// Fraction of full size the ghost shrinks to.
    scale_to: f64,
    /// How far it sinks on the way out, in compositor pixels. `0` for a
    /// close, which goes nowhere; a minimize drops as it shrinks.
    sink: i32,
}

/// Fraction of full size a window starts at when it opens (and shrinks to
/// when it closes): a subtle pop, not a dramatic zoom.
const OPEN_SCALE_FROM: f64 = 0.90;

/// Fraction of full size a *minimizing* window shrinks to. Harder than a
/// close's [`OPEN_SCALE_FROM`] on purpose: the two use the same ghost, and
/// a window being put away should not look like a window being destroyed.
const MINIMIZE_SCALE_TO: f64 = 0.62;

/// How far a minimizing window sinks as it goes, as a fraction of its own
/// height — and how far a restoring one rises back from.
///
/// Downward because that is where a taskbar nearly always is. It is a
/// *direction*, not a destination: the compositor has no idea where in a
/// shell's bar a given window's entry sits, and a wrong destination reads
/// far worse than an honest "away, that way".
const MINIMIZE_SINK_FRAC: f64 = 0.30;

/// How far a layer surface slides in from its anchored edge, as a fraction
/// of its own size along that axis. A short travel: a bar should look like
/// it settled into place, not like it flew in.
const LAYER_SLIDE_FROM: f64 = 0.35;

/// Which screen edge a layer surface animates from.
///
/// Derived from where the surface actually sits rather than from its
/// anchors: a panel flush against the top of the output slides down from
/// the top, and anything not touching an edge — a centred launcher — just
/// fades, which is what you want for it anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerEdge {
    Top,
    Bottom,
    Left,
    Right,
    /// Not against any edge: fade only, no travel.
    Center,
}

impl LayerEdge {
    /// The offset to draw at, given how far through the animation we are
    /// (`1.0` = fully settled) and the surface's size.
    fn offset(self, progress: f64, size: Size<i32, Physical>) -> Point<i32, Physical> {
        let back = 1.0 - progress.clamp(0.0, 1.0);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a fraction of a surface dimension; bounded by the output"
        )]
        let along = |extent: i32| (f64::from(extent) * LAYER_SLIDE_FROM * back).round() as i32;
        match self {
            LayerEdge::Top => Point::from((0, -along(size.h))),
            LayerEdge::Bottom => Point::from((0, along(size.h))),
            LayerEdge::Left => Point::from((-along(size.w), 0)),
            LayerEdge::Right => Point::from((along(size.w), 0)),
            LayerEdge::Center => Point::from((0, 0)),
        }
    }

    /// Pick the edge `rect` is against within `output`, if any. A surface
    /// touching two edges (a full-height side bar) takes the one it spans
    /// *less* of, which is the direction it visibly came from.
    fn of(rect: Rectangle<i32, Physical>, output: Rectangle<i32, Physical>) -> Self {
        // A few pixels of slack: bars are commonly placed with a small gap.
        const SLACK: i32 = 8;
        let touches_top = rect.loc.y - output.loc.y <= SLACK;
        let touches_bottom =
            (output.loc.y + output.size.h) - (rect.loc.y + rect.size.h) <= SLACK;
        let touches_left = rect.loc.x - output.loc.x <= SLACK;
        let touches_right = (output.loc.x + output.size.w) - (rect.loc.x + rect.size.w) <= SLACK;

        // Spanning the full width means the short axis is vertical, so it
        // came from top or bottom — and vice versa.
        let spans_width = touches_left && touches_right;
        let spans_height = touches_top && touches_bottom;
        match (spans_width, spans_height) {
            (true, true) => LayerEdge::Center, // covers the output: fade only
            (true, false) if touches_top => LayerEdge::Top,
            (true, false) if touches_bottom => LayerEdge::Bottom,
            (false, true) if touches_left => LayerEdge::Left,
            (false, true) if touches_right => LayerEdge::Right,
            _ if touches_top => LayerEdge::Top,
            _ if touches_bottom => LayerEdge::Bottom,
            _ if touches_left => LayerEdge::Left,
            _ if touches_right => LayerEdge::Right,
            _ => LayerEdge::Center,
        }
    }
}

/// A layer surface's last frame, captured at destroy time, animating out.
struct ClosingLayer {
    texture: GlesTexture,
    /// Where the content sat on screen — absolute compositor pixels.
    rect: Rectangle<i32, Physical>,
    edge: LayerEdge,
    anim: Animation,
}

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
    /// Rect a running move/resize animation interpolates from. One rect
    /// serves both: the position animation reads its `loc`, the resize its
    /// `size`, and both start from wherever the window was actually drawn.
    move_from: Rectangle<i32, Physical>,
    /// In-flight *position* animation, if any.
    move_anim: Option<Animation>,
    /// In-flight *size* animation, if any. Separate from `move_anim` so a
    /// reflow can glide into place at one pace while growing at another;
    /// with the default config the two are identical and run in lockstep.
    resize_anim: Option<Animation>,
    /// In-flight open (fade + scale-in) animation, if any.
    open_anim: Option<Animation>,
    /// How far below its rect the open animation starts, in compositor
    /// pixels. Non-zero only for a restore from minimize, which rises the
    /// distance the minimize sank; a plain map scales in where it stands.
    open_rise: i32,
    /// Focus state as of the last frame, to notice the flip.
    focused: bool,
    /// In-flight border crossfade, if any.
    focus_anim: Option<Animation>,
    /// Focus level the crossfade started from, so a window refocused
    /// mid-fade continues from the colour on screen instead of snapping.
    focus_from: f32,
    /// Focus level actually drawn last frame — `focus_anim`'s output, kept
    /// so a retarget has something to start from.
    focus_now: f32,
}

impl WindowAnim {
    /// Whether anything about this window is still moving, so the frame
    /// can't be considered settled.
    ///
    /// One place rather than a disjunction repeated at each call site: the
    /// three of them decide damage, whether to schedule another frame, and
    /// whether direct scanout may take the plane, and a new animation that
    /// only reaches two of them produces a window that animates without
    /// being repainted.
    /// Point position and size at `target`, starting (or retargeting) the
    /// move and resize animations that are enabled.
    ///
    /// `move_cfg` / `resize_cfg` are `Some` only when that animation should
    /// actually run — disabled in the config, or suppressed because the
    /// window is being dragged and must track the cursor 1:1.
    ///
    /// The two start independently: a pure move must not run the resize
    /// animation, or a window sliding across the screen would also be told
    /// to "resize" to the size it already has, on the resize timing.
    fn retarget(
        &mut self,
        now: f64,
        target: Rectangle<i32, Physical>,
        move_cfg: Option<AnimSpec>,
        resize_cfg: Option<AnimSpec>,
    ) {
        if target == self.target {
            return;
        }
        let moved = target.loc != self.target.loc;
        let resized = target.size != self.target.size;
        // Both animations read `move_from`, so latch where the window is
        // actually drawn once, before either overwrites it.
        if moved && move_cfg.is_some() || resized && resize_cfg.is_some() {
            self.move_from = self.displayed;
        }
        if moved {
            self.move_anim =
                move_cfg.map(|c| Animation::start(now, c.duration_secs(), c.curve));
        }
        if resized {
            self.resize_anim =
                resize_cfg.map(|c| Animation::start(now, c.duration_secs(), c.curve));
        }
        self.target = target;
    }

    /// Interpolate position and size toward the target on their own clocks,
    /// leaving the result in [`Self::displayed`]. Whichever isn't animating
    /// snaps straight to the target.
    fn advance_geometry(&mut self, now: f64) {
        let loc = match self.move_anim {
            Some(a) if !a.done(now) => lerp_point(self.move_from.loc, self.target.loc, a.value(now)),
            Some(_) => {
                self.move_anim = None;
                self.target.loc
            }
            None => self.target.loc,
        };
        let size = match self.resize_anim {
            Some(a) if !a.done(now) => {
                lerp_size(self.move_from.size, self.target.size, a.value(now))
            }
            Some(_) => {
                self.resize_anim = None;
                self.target.size
            }
            None => self.target.size,
        };
        self.displayed = Rectangle::new(loc, size);
    }

    /// Advance the border crossfade toward `focused`, starting one if focus
    /// just changed. Leaves [`Self::focus_now`] at the level to draw.
    fn advance_focus(&mut self, now: f64, focused: bool, enabled: bool, cfg: AnimSpec) {
        if focused != self.focused {
            self.focused = focused;
            // Start from the colour actually on screen, so refocusing
            // mid-fade continues rather than snapping.
            self.focus_from = self.focus_now;
            self.focus_anim =
                enabled.then(|| Animation::start(now, cfg.duration_secs(), cfg.curve));
        }
        let target = f32::from(u8::from(self.focused));
        self.focus_now = match self.focus_anim {
            Some(a) => {
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "eased progress is in [0,1]; f32 is plenty for a colour mix"
                )]
                let v = a.value(now) as f32;
                if a.done(now) {
                    self.focus_anim = None;
                    target
                } else {
                    self.focus_from + (target - self.focus_from) * v
                }
            }
            None => target,
        };
    }

    fn is_animating(&self) -> bool {
        self.move_anim.is_some()
            || self.resize_anim.is_some()
            || self.open_anim.is_some()
            || self.focus_anim.is_some()
    }
}

/// What to draw for one placement this frame, after animation: the
/// on-screen rect (interpolated position/size) and opacity. The element
/// builder derives the surface's content scale from `effective` vs the
/// placement's target `cell_rect`.
#[derive(Debug, Clone, Copy)]
struct WinDraw {
    effective: Rectangle<i32, Physical>,
    alpha: f32,
    /// How focused the border should look: `1.0` fully the active fill,
    /// `0.0` fully the inactive one, in between while the focus crossfade
    /// runs. Not a bool, so focus moving between two windows reads as one
    /// colour handing over rather than two hard flips.
    focus: f32,
}

/// What the screenshot selection UI should draw this frame. The
/// rectangle is in absolute compositor coords; `None` means a session is
/// active but nothing is selected yet (just dim every output).
#[derive(Debug, Clone)]
pub struct ScreenshotOverlay {
    pub selection: Option<Rectangle<i32, Physical>>,
    /// Draw grab handles on the selection's corners and edge midpoints —
    /// set once it is *committed* (the drag has been released) and can be
    /// adjusted. A rect still being dragged out shows none: there is
    /// nothing to grab while the pointer is already holding a corner.
    pub handles: bool,
    /// The options bar, once a selection is committed.
    pub toolbar: Option<Toolbar>,
    /// Annotation strokes, in absolute compositor coords, oldest first.
    /// The one being drawn right now is last.
    pub strokes: Vec<StrokeDraw>,
}

/// One annotation polyline ready to draw.
#[derive(Debug, Clone)]
pub struct StrokeDraw {
    pub colour: [f32; 3],
    pub width: f32,
    pub points: Vec<(i32, i32)>,
}

/// The screenshot options bar: where it is and what is on it. Laid out by
/// `screenshot::toolbar_layout` so the hit-testing and the drawing can't
/// disagree about where a button is.
#[derive(Debug, Clone)]
pub struct Toolbar {
    pub bar: Rectangle<i32, Physical>,
    pub buttons: Vec<ToolButton>,
}

#[derive(Debug, Clone, Copy)]
pub struct ToolButton {
    pub rect: Rectangle<i32, Physical>,
    pub icon: ToolIcon,
    /// Currently in effect — annotation is on, or this is the pen colour.
    pub active: bool,
    pub hovered: bool,
}

/// What a toolbar button draws. Every one is a short segment list run
/// through [`SEGMENT_SHADER`], except a colour swatch, which is a filled
/// square showing the colour itself.
#[derive(Debug, Clone, Copy)]
pub enum ToolIcon {
    Take,
    Draw,
    Swatch([f32; 3]),
    /// The pen-width slider: a track, the filled part up to `frac`, and a
    /// knob whose size *is* the width being chosen — the control shows
    /// you the stroke rather than a number you have to imagine.
    Slider { frac: f32, width: f32, colour: [f32; 3] },
    Text,
    Cancel,
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
    /// Root surfaces of everything drawn into the frame currently in
    /// flight, for `wl_callback.done` at vblank. Firing frame callbacks
    /// when the flip *completes* (not when it's queued) paces clients to
    /// the display: a queue-time callback lets a fast client commit a
    /// second frame inside the same refresh period, which supersedes the
    /// first and forces its `wp_presentation` feedback to be discarded —
    /// present-timing consumers (Wine/NVIDIA HDR swapchains) treat those
    /// discards as "frame never shown" and rebuild their swapchain.
    pending_frame_roots: Vec<WlSurface>,
    /// Whether the frame currently in flight is a DIRECT-SCANOUT frame (the
    /// client's own buffer on the plane). The vblank release-point signaling
    /// must skip those: the client buffer is literally what's on screen, so
    /// releasing it would let the client overwrite the displayed image.
    pending_direct: bool,
    /// The surface + commit of the last frame we direct-scanned onto this
    /// output's plane, so the next one can ask for damage *since* it and hand
    /// KMS a `FB_DAMAGE_CLIPS` blob. `None` means the plane holds something
    /// whose delta we can't describe (a composite frame, a different surface,
    /// or a failed flip) and the next direct frame must claim full damage.
    direct_damage_ref: Option<(ObjectId, CommitCounter)>,
    /// Rolling per-phase frame-cost accumulator, logged + reset every ~5 s
    /// (see [`RenderProfile`]). Always on: the bookkeeping is a handful of
    /// `Instant::now()` calls per frame.
    profile: RenderProfile,
    /// Per-output damage diffing state (see [`DamageTracker`]).
    damage_tracker: DamageTracker,
    /// Last frame's imported surface-alpha texture per blur-eligible layer
    /// surface (by id), for the backdrop blur's temporal coverage min (see
    /// `MASK_BLUR_SHADER`). A surface absent here has no history yet — its
    /// blur is skipped for one frame rather than trusting a single frame's
    /// alpha, which is what lets a client's transient full-surface frame not
    /// flash. `GlesTexture` is `Arc`-backed, so holding last frame's handle
    /// is cheap and keeps that buffer's texture alive until the next frame.
    prev_layer_masks: HashMap<ObjectId, GlesTexture>,
}

/// What one drawn thing (window / layer / popup) looked like last frame,
/// for damage diffing: its content identity, where it was drawn, and the
/// non-content inputs that change its pixels (focus flips the border
/// colour; alpha animates during open/fade).
struct DrawnState {
    fingerprint: Vec<(ObjectId, CommitCounter)>,
    rect: Rectangle<i32, Physical>,
    focused: bool,
    alpha_bits: u32,
    /// Whether backdrop blur was drawn behind it — an
    /// ext-background-effect opt-in flip changes the pixels without any
    /// buffer commit, so it must damage like a focus flip does.
    blur: bool,
    /// Identity of the titlebar drawn on it ([`bar_key`]), or `0` for a
    /// window with no bar.
    ///
    /// A retitled window commits no buffer and moves no commit counter,
    /// so without this the bar repaints only when something else happens
    /// to damage the window — a browser whose tab changed would keep the
    /// old title until you moved it. (Focus is already covered by
    /// `focused`, which is the other half of the bar's identity.)
    bar: u64,
}

/// Per-output damage tracker: diffs each frame's drawn set against the
/// previous frame to produce the region that actually changed, and keeps
/// a short history so a swapchain buffer that last presented `age` frames
/// ago can be repaired by re-drawing the union of the last `age` frames'
/// damage. `None` anywhere means "full frame" — the tracker falls back to
/// full liberally (transient overlays, media wallpaper, workspace slides,
/// composited cursors) because a missed-damage artifact is strictly worse
/// than a full repaint. `LIBRELAND_NO_DAMAGE=1` forces full frames for
/// A/B runs against the render-profile log.
struct DamageTracker {
    prev: HashMap<ObjectId, DrawnState>,
    /// Close-animation rects drawn last frame (they fade → damage every
    /// frame any exist, plus the frame after the last one vanishes).
    prev_closing: Vec<Rectangle<i32, Physical>>,
    /// Newest-last per-frame damage, capped at [`DAMAGE_HISTORY`].
    history: VecDeque<Vec<Rectangle<i32, Physical>>>,
    /// Frames since the persistent fp16 HDR scene texture was last drawn
    /// into (1 = last frame). Single-pass frames skip the scene, so its
    /// staleness is tracked separately from the swapchain's buffer age.
    scene_age: usize,
    /// Force a full frame once (config/output change invalidation).
    force_full: bool,
}

/// Swapchain depth is 3-4; anything older is repainted in full.
const DAMAGE_HISTORY: usize = 8;

/// Damage rect-count cap: beyond this the set coalesces to its bounding
/// box (`FB_DAMAGE_CLIPS` arrays shouldn't grow unbounded, and neither
/// should per-element intersection work).
const DAMAGE_MAX_RECTS: usize = 32;

impl DamageTracker {
    fn new() -> Self {
        DamageTracker {
            prev: HashMap::new(),
            prev_closing: Vec::new(),
            history: VecDeque::new(),
            scene_age: usize::MAX,
            force_full: true,
        }
    }

    /// The damage needed to repair a target whose content is `age` frames
    /// old, given this frame's `current` damage. `None` = repaint in full
    /// (unknown age, or history too short).
    fn accumulated(
        &self,
        age: usize,
        current: &[Rectangle<i32, Physical>],
    ) -> Option<Vec<Rectangle<i32, Physical>>> {
        if age == 0 || age > self.history.len() + 1 {
            return None;
        }
        let mut out = current.to_vec();
        for frame in self.history.iter().rev().take(age - 1) {
            out.extend_from_slice(frame);
        }
        Some(coalesce_damage(out))
    }

    /// Record this frame's damage (`None` = the frame was painted in
    /// full, which repairs every buffer — recorded as a full-frame rect).
    fn push(
        &mut self,
        current: Option<Vec<Rectangle<i32, Physical>>>,
        full: Rectangle<i32, Physical>,
    ) {
        self.history.push_back(current.unwrap_or_else(|| vec![full]));
        if self.history.len() > DAMAGE_HISTORY {
            self.history.pop_front();
        }
    }
}

/// Clamp a damage set's cardinality: past [`DAMAGE_MAX_RECTS`], coalesce
/// to the single bounding box. (Overlapping rects are fine — GL and KMS
/// both tolerate them; only unbounded growth isn't.)
fn coalesce_damage(mut damage: Vec<Rectangle<i32, Physical>>) -> Vec<Rectangle<i32, Physical>> {
    damage.retain(|r| r.size.w > 0 && r.size.h > 0);
    // Make the set disjoint. Damage is assembled from independent sources
    // that routinely cover the same pixels — a surface's own rect, its
    // previous rect, and the conservative re-damage of every frosted
    // backdrop — so overlap is the norm, not an edge case.
    //
    // Overlap is harmless for an opaque draw and for KMS FB_DAMAGE_CLIPS,
    // but the renderer draws once *per rect*: a translucent draw lands on
    // the same pixel several times and blends several times. A frosted
    // panel covered by three rects composites its backdrop three times and
    // goes visibly opaque — and because the rect count changes frame to
    // frame, it flickers between correct and opaque as the count moves.
    //
    // Subtracting what is already accepted keeps the union identical while
    // guaranteeing every pixel is drawn exactly once.
    let mut disjoint: Vec<Rectangle<i32, Physical>> = Vec::with_capacity(damage.len());
    for r in damage {
        disjoint.extend(
            r.subtract_rects(disjoint.iter().copied())
                .into_iter()
                .filter(|p| p.size.w > 0 && p.size.h > 0),
        );
    }
    // Splitting can multiply the count, so cap afterwards. The bbox is a
    // single rect, so it is trivially disjoint.
    if disjoint.len() > DAMAGE_MAX_RECTS {
        let bbox = disjoint
            .iter()
            .skip(1)
            .fold(disjoint[0], |acc, r| acc.merge(*r));
        return vec![bbox];
    }
    disjoint
}

/// Union of the drawn geometries of an element group — the on-screen
/// bbox the group occupies this frame, in output-local physical pixels.
fn elements_bbox<E: smithay::backend::renderer::element::Element>(
    elements: &[E],
    scale: Scale<f64>,
) -> Option<Rectangle<i32, Physical>> {
    let mut it = elements.iter().map(|e| e.geometry(scale));
    let first = it.next()?;
    Some(it.fold(first, smithay::utils::Rectangle::merge))
}

/// Translate frame-absolute damage into `dst`-relative rects (the
/// convention `render_texture_from_to` expects), dropping the parts
/// outside `dst`. An empty result means the draw can be skipped.
fn damage_rel(
    damage: &[Rectangle<i32, Physical>],
    dst: Rectangle<i32, Physical>,
) -> Vec<Rectangle<i32, Physical>> {
    damage
        .iter()
        .filter_map(|d| {
            d.intersection(dst)
                .map(|mut r| {
                    r.loc -= dst.loc;
                    r
                })
        })
        .collect()
}

/// One cached decoration offscreen (`win_tex`): the rendered texture plus
/// the content identity it was rendered from. Valid while the window's
/// surface tree hasn't committed (fingerprint match) and the cell size /
/// colour format are unchanged.
struct WinTexCache {
    tex: GlesTexture,
    size: Size<i32, smithay::utils::Buffer>,
    fmt: Fourcc,
    /// Per-node commit counters of the window's surface tree at render
    /// time; a commit anywhere in the tree changes the fingerprint.
    fingerprint: Vec<(ObjectId, CommitCounter)>,
    /// Identity of the titlebar drawn into this offscreen ([`bar_key`]),
    /// or `0` when the window has none.
    ///
    /// The offscreen is cached against the *surface tree*, which knows
    /// nothing about a title change or a focus change — so without this
    /// the bar would draw once and then never update, a stale-cache bug
    /// of exactly the shape 94145d3 fixed for blank offscreens.
    bar: u64,
}

/// Lazily-loaded UI faces for titlebar text.
///
/// The distinction that matters is "not looked yet" versus "looked and
/// found nothing": without it, a system with no usable font rescans
/// every font directory for every bar it draws.
enum FontState {
    Unscanned,
    Scanned(Option<Fonts>),
}

/// [`bar_key`] for a placement as it will actually be drawn this frame.
///
/// One definition shared by Phase A (which decides whether the cached
/// offscreen is current) and the damage tracker (which decides whether
/// the window is repainted at all). Two definitions that drifted would
/// give a bar that redraws but is never damaged, or damages every frame.
fn placement_bar_key(
    p: &Placement,
    wd: &WinDraw,
    scale: f64,
    buttons: usize,
    state: BarState,
) -> u64 {
    let bar_h = scale_i(p.deco.titlebar, scale);
    if bar_h <= 0 {
        return 0;
    }
    let cell_w = scale_i(wd.effective.size.w, scale).max(1);
    let title = window_title(&p.surface).unwrap_or_default();
    let app_id = window_app_id(&p.surface);
    bar_key(
        cell_w,
        bar_h,
        &title,
        app_id.as_deref(),
        wd.focus >= 0.5,
        buttons,
        state,
    )
}

/// Icon slot side for a bar `height` px tall. Mirrors `titlebar`'s own
/// slot geometry, so the decoded icon is the size it will be drawn at
/// and never rescaled twice.
fn icon_side(height: i32) -> u32 {
    u32::try_from(crate::titlebar::icon_side_for(height)).unwrap_or(1).max(1)
}

/// Cap on rasterized titlebars held at once. A bar is one small RGBA
/// texture, so this is generous; it exists so a client whose title
/// animates (a download percentage, a clock) can't grow the cache
/// without limit.
const BAR_CACHE_MAX: usize = 64;

/// Identity of a window's titlebar: everything that changes its pixels.
///
/// Focus is deliberately a *bool* rather than the crossfade's `f32`. The
/// border colour animates continuously (in the composite shader, where
/// it costs a uniform), but a bar that re-rasterized on every frame of
/// a 220 ms crossfade would re-render the window's whole offscreen ~35
/// times per focus change. Titlebars switch, borders fade.
fn bar_key(
    width: i32,
    height: i32,
    title: &str,
    app_id: Option<&str>,
    focused: bool,
    buttons: usize,
    state: BarState,
) -> u64 {
    use std::hash::{Hash as _, Hasher as _};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    width.hash(&mut h);
    height.hash(&mut h);
    title.hash(&mut h);
    // The icon is drawn from this, and two windows can share a title.
    app_id.hash(&mut h);
    focused.hash(&mut h);
    buttons.hash(&mut h);
    state.maximized.hash(&mut h);
    // Hover is in the key, so moving onto a button re-rasterizes the bar
    // *and* damages the window through the same value — the two can't
    // disagree about whether the highlight is on screen.
    match state.hovered {
        None => 0u8,
        Some(crate::config::TitlebarButton::Minimize) => 1,
        Some(crate::config::TitlebarButton::Maximize) => 2,
        Some(crate::config::TitlebarButton::Close) => 3,
    }
    .hash(&mut h);
    // 0 is reserved for "no titlebar", so a real bar can never collide
    // with one.
    h.finish() | 1
}

/// The window tree's content identity — decides whether a cached
/// decoration offscreen is still current without touching the GPU.
fn surface_tree_fingerprint(root: &WlSurface) -> Vec<(ObjectId, CommitCounter)> {
    let mut out = Vec::new();
    with_surface_tree_downward(
        root,
        (),
        |_, _, &()| TraversalAction::DoChildren(()),
        |surface, states, &()| {
            if let Some(data) = states.data_map.get::<RendererSurfaceStateUserData>() {
                out.push((surface.id(), data.lock().unwrap().current_commit()));
            }
        },
        |_, _, &()| true,
    );
    out
}

/// Rolling accumulator of where a composited frame's CPU/GL-submission
/// time goes, per output. Phases (all wall-clock around the GL calls, so
/// stalls — allocations, readbacks — show up even though true GPU
/// execution is async): element import, decoration offscreens, blur
/// pyramid, scene draw, HDR encode. Logged at info every ~5 s so a
/// session log doubles as a benchmark record, then reset.
#[derive(Debug)]
struct RenderProfile {
    since: Instant,
    frames: u32,
    total: Duration,
    max: Duration,
    import: Duration,
    wintex: Duration,
    blur: Duration,
    scene: Duration,
    encode: Duration,
}

impl RenderProfile {
    fn new() -> Self {
        RenderProfile {
            since: Instant::now(),
            frames: 0,
            total: Duration::ZERO,
            max: Duration::ZERO,
            import: Duration::ZERO,
            wintex: Duration::ZERO,
            blur: Duration::ZERO,
            scene: Duration::ZERO,
            encode: Duration::ZERO,
        }
    }

    /// Fold one frame's phase timings in; every ~5 s emit the averages and
    /// reset. `ms` values are averages per *composited* frame (direct-scanout
    /// frames never get here — that path is effectively free by design).
    #[allow(
        clippy::cast_precision_loss,
        reason = "microsecond sums over a 5 s window are far below f64's exact-integer range"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "one Duration per profiled phase; a struct would restate the same six names"
    )]
    fn record(
        &mut self,
        name: &str,
        total: Duration,
        import: Duration,
        wintex: Duration,
        blur: Duration,
        scene: Duration,
        encode: Duration,
    ) {
        self.frames += 1;
        self.total += total;
        self.max = self.max.max(total);
        self.import += import;
        self.wintex += wintex;
        self.blur += blur;
        self.scene += scene;
        self.encode += encode;
        if self.since.elapsed() < Duration::from_secs(5) {
            return;
        }
        let per = |d: Duration| d.as_micros() as f64 / f64::from(self.frames.max(1)) / 1000.0;
        info!(
            output = %name,
            frames = self.frames,
            avg_ms = format!("{:.2}", per(self.total)),
            max_ms = format!("{:.2}", self.max.as_secs_f64() * 1000.0),
            import_ms = format!("{:.2}", per(self.import)),
            wintex_ms = format!("{:.2}", per(self.wintex)),
            blur_ms = format!("{:.2}", per(self.blur)),
            scene_ms = format!("{:.2}", per(self.scene)),
            encode_ms = format!("{:.2}", per(self.encode)),
            "render profile (composited frames, 5 s window)"
        );
        *self = RenderProfile::new();
    }
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

/// The scale that maps an output's `compositor` rect onto its `mode`
/// pixels without losing one to rounding — see
/// [`Renderer::xwayland_client_scale`], which is the only caller and
/// carries the full story. The larger axis ratio wins so neither axis
/// can round short; `fallback` covers a degenerate (zero-sized, or
/// mid-teardown) output, since every consumer divides by this number.
fn client_scale_for(
    mode: Size<i32, Physical>,
    compositor: Size<i32, Physical>,
    fallback: f64,
) -> f64 {
    if compositor.w <= 0 || compositor.h <= 0 {
        return fallback;
    }
    let ratio = (f64::from(mode.w) / f64::from(compositor.w))
        .max(f64::from(mode.h) / f64::from(compositor.h));
    if ratio.is_finite() && ratio > 0.0 {
        ratio
    } else {
        fallback
    }
}

#[cfg(test)]
mod damage_tests {
    use super::{DAMAGE_MAX_RECTS, coalesce_damage};
    use smithay::utils::{Physical, Point, Rectangle, Size};

    fn r(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, Physical> {
        Rectangle::new(Point::from((x, y)), Size::from((w, h)))
    }

    /// How many of the returned rects contain a given point. Anything above
    /// one means the renderer draws that pixel more than once, which blends
    /// a translucent draw repeatedly.
    fn hits(set: &[Rectangle<i32, Physical>], x: i32, y: i32) -> usize {
        set.iter().filter(|d| d.contains(Point::from((x, y)))).count()
    }

    /// The frosted-panel flicker: damage is assembled from sources that
    /// overlap by design (a surface's rect, its previous rect, and the
    /// conservative re-damage of every blurred backdrop), and the renderer
    /// draws once per rect. Three rects over the bar composited its
    /// backdrop three times and it went opaque.
    #[test]
    fn overlapping_damage_is_made_disjoint() {
        // A bar strip, plus two window rects that both cross it.
        let out = coalesce_damage(vec![
            r(0, 0, 2560, 40),
            r(100, 0, 400, 300),
            r(300, 0, 400, 300),
        ]);
        for (x, y) in [(120, 10), (350, 10), (450, 20), (600, 200), (150, 100)] {
            assert_eq!(hits(&out, x, y), 1, "pixel ({x},{y}) drawn {} times", hits(&out, x, y));
        }
    }

    /// Splitting must not change which pixels are damaged, only how they
    /// are partitioned — otherwise stale content survives where cover was lost.
    #[test]
    fn disjoint_split_preserves_the_union() {
        let input = vec![r(0, 0, 100, 100), r(50, 50, 100, 100), r(90, 10, 30, 200)];
        let out = coalesce_damage(input.clone());
        for x in 0..200 {
            for y in 0..220 {
                let p = Point::from((x, y));
                let before = input.iter().any(|d| d.contains(p));
                let after = out.iter().any(|d| d.contains(p));
                assert_eq!(before, after, "coverage changed at ({x},{y})");
            }
        }
    }

    /// Fully-contained duplicates are the common case (the re-damage pass
    /// re-adds a rect already covered) and must collapse to nothing extra.
    #[test]
    fn duplicate_rects_collapse() {
        let out = coalesce_damage(vec![r(10, 10, 100, 100); 4]);
        assert_eq!(out.len(), 1);
        assert_eq!(hits(&out, 50, 50), 1);
    }

    /// Splitting multiplies the rect count, so the cap has to be applied
    /// after it — and the bbox fallback is itself a single, disjoint rect.
    #[test]
    fn split_still_respects_the_cap() {
        let many: Vec<_> = (0..DAMAGE_MAX_RECTS as i32 + 8)
            .map(|i| r(i * 4, 0, 200, 200))
            .collect();
        let out = coalesce_damage(many);
        assert!(out.len() <= DAMAGE_MAX_RECTS, "got {} rects", out.len());
        assert_eq!(hits(&out, 100, 100), 1);
    }
}

#[cfg(test)]
mod opaque_region_tests {
    use super::opaque_region_covers;
    use smithay::utils::{Logical, Rectangle, Size};

    fn size(w: i32, h: i32) -> Size<i32, Logical> {
        Size::from((w, h))
    }
    fn rect(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, Logical> {
        Rectangle::new((x, y).into(), (w, h).into())
    }
    /// `opaque_region_covers` with the borrow spelled out, so each case below
    /// reads as just its rects and the surface they should (not) cover.
    fn covers(rects: &[Rectangle<i32, Logical>], s: Size<i32, Logical>) -> bool {
        opaque_region_covers(Some(rects), s)
    }

    #[test]
    fn no_declared_region_is_not_provably_opaque() {
        assert!(!opaque_region_covers(None, size(100, 100)));
        assert!(!opaque_region_covers(Some(&[]), size(100, 100)));
    }

    #[test]
    fn one_covering_rect_is_the_fast_path() {
        assert!(covers(&[rect(0, 0, 100, 100)], size(100, 100)));
        // Oversized (a client may declare past its own edges).
        assert!(covers(&[rect(-5, -5, 200, 200)], size(100, 100)));
    }

    #[test]
    fn one_partial_rect_does_not_cover() {
        assert!(!covers(&[rect(0, 0, 100, 99)], size(100, 100)));
        assert!(!covers(&[rect(1, 0, 100, 100)], size(100, 100)));
    }

    /// The case the single-rect check used to miss: a toolkit declaring its
    /// opacity as tiles. Together they cover the surface, so the surface is
    /// opaque and may go on the primary plane.
    #[test]
    fn tiled_rects_that_together_cover_do_count() {
        // Two halves, split horizontally.
        assert!(covers(&[rect(0, 0, 100, 50), rect(0, 50, 100, 50)], size(100, 100)
        ));
        // Two halves, split vertically.
        assert!(covers(&[rect(0, 0, 50, 100), rect(50, 0, 50, 100)], size(100, 100)
        ));
        // Four quadrants, out of order, with overlap.
        assert!(covers(&[
                rect(50, 50, 50, 50),
                rect(0, 0, 60, 60),
                rect(50, 0, 50, 50),
                rect(0, 50, 50, 50),
            ], size(100, 100)
        ));
    }

    #[test]
    fn tiled_rects_with_a_gap_do_not_cover() {
        // A one-pixel column missed between the two halves.
        assert!(!covers(&[rect(0, 0, 49, 100), rect(50, 0, 50, 100)], size(100, 100)
        ));
        // Full-width bands that don't reach the bottom.
        assert!(!covers(&[rect(0, 0, 100, 40), rect(0, 40, 100, 40)], size(100, 100)
        ));
        // A hole in the middle: bands cover top and bottom, sides cover the
        // middle band only partially.
        assert!(!covers(&[
                rect(0, 0, 100, 30),
                rect(0, 70, 100, 30),
                rect(0, 30, 40, 40),
                rect(60, 30, 40, 40),
            ], size(100, 100)
        ));
    }

    /// The sweep is bounded: a client declaring its opacity in dozens of
    /// pieces is not the fullscreen game this fast path exists for, and the
    /// scan would cost more than the frame it might save.
    #[test]
    fn absurdly_many_rects_are_declined_rather_than_swept() {
        let strips: Vec<_> = (0..100).map(|i| rect(i, 0, 1, 100)).collect();
        assert!(!covers(&strips, size(100, 100)));
    }

    /// A zero-sized surface is vacuously "covered" by any rect, which would
    /// hand the primary plane a buffer with no pixels. `Size` refuses to hold
    /// a negative dimension at all, so zero is the whole degenerate case.
    #[test]
    fn a_degenerate_surface_is_never_opaque() {
        assert!(!covers(&[rect(0, 0, 100, 100)], size(0, 100)));
        assert!(!covers(&[rect(0, 0, 100, 100)], size(100, 0)));
    }
}

#[cfg(test)]
mod client_scale_tests {
    use super::{client_scale_for, output_compositor_size};
    use smithay::utils::{Physical, Size};

    /// smithay's own logical→client conversion (`to_client_precise_round`):
    /// multiply by the scale, round each axis independently.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "test sizes are display pixels; the product is a few thousand"
    )]
    fn to_client(logical: i32, scale: f64) -> i32 {
        (f64::from(logical) * scale).round() as i32
    }

    #[test]
    fn derived_scale_round_trips_a_full_output_rect() {
        // The box this was reported on: 4K panel, configured scale 1.35.
        let mode = Size::<i32, Physical>::from((3840, 2160));
        let comp = output_compositor_size(mode, 1.35);
        assert_eq!((comp.w, comp.h), (2844, 1600));
        // The configured scale loses a pixel on the way back — this is the
        // bug: a "fullscreen" X11 window that doesn't reach the X screen's
        // right edge, so Wine strips _NET_WM_STATE_FULLSCREEN and the
        // fullscreen⇄tiled fight starts.
        assert_eq!(to_client(comp.w, 1.35), 3839);
        // The derived scale covers both axes, exactly on the wide one.
        let derived = client_scale_for(mode, comp, 1.35);
        assert_eq!(to_client(comp.w, derived), mode.w);
        assert!(to_client(comp.h, derived) >= mode.h);
    }

    #[test]
    fn derived_scale_is_the_configured_one_when_the_mode_divides_evenly() {
        let mode = Size::<i32, Physical>::from((3840, 2160));
        let comp = output_compositor_size(mode, 2.0);
        assert_eq!((comp.w, comp.h), (1920, 1080));
        assert!((client_scale_for(mode, comp, 2.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn degenerate_output_falls_back_instead_of_handing_out_zero() {
        let mode = Size::<i32, Physical>::from((3840, 2160));
        let zero = Size::<i32, Physical>::from((0, 0));
        assert!((client_scale_for(mode, zero, 1.35) - 1.35).abs() < 1e-12);
        assert!((client_scale_for(zero, mode, 1.35) - 1.35).abs() < 1e-12);
    }
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
                    UniformName::new("mask_prev", UniformType::_1i),
                    UniformName::new("mask_mul", UniformType::_2f),
                    UniformName::new("mask_add", UniformType::_2f),
                    UniformName::new("mask_dilate", UniformType::_2f),
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
                &[
                    UniformName::new("reference_white", UniformType::_1f),
                    UniformName::new("knee", UniformType::_1f),
                ],
            )
            .context("screenshot tonemap shader compile failed")?;
        // Anti-aliased strokes: toolbar glyphs and freehand annotation,
        // both as segment lists. `_4f` per segment because GLES 2.0 takes
        // a uniform array as one named element per index.
        let segment_shader = gles
            .compile_custom_texture_shader(
                SEGMENT_SHADER,
                &[
                    UniformName::new("count", UniformType::_1i),
                    UniformName::new("colour", UniformType::_3f),
                    UniformName::new("thickness", UniformType::_1f),
                    UniformName::new("quad", UniformType::_2f),
                ],
            )
            .context("segment shader compile failed")?;
        // 1x1 opaque white for the procedural programs' unused sampler.
        let mut blank_tex = gles
            .create_buffer(
                Fourcc::Abgr8888,
                Size::<i32, smithay::utils::Buffer>::from((1, 1)),
            )
            .context("blank texture alloc failed")?;
        {
            let one = Size::<i32, Physical>::from((1, 1));
            let mut target = gles.bind(&mut blank_tex).context("blank texture bind")?;
            let mut frame = gles
                .render(&mut target, one, Transform::Normal)
                .context("blank texture render")?;
            frame
                .clear(Color32F::new(1.0, 1.0, 1.0, 1.0), &[Rectangle::from_size(one)])
                .context("blank texture clear")?;
            let _ = frame.finish().context("blank texture finish")?;
        }
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
        let hdr_decode_swizzle_shader = gles
            .compile_custom_texture_shader(HDR_DECODE_SWIZZLE_SHADER, &[])
            .context("HDR decode swizzle shader compile failed")?;
        let pq_passthrough_swizzle_shader = gles
            .compile_custom_texture_shader(PQ_PASSTHROUGH_SWIZZLE_SHADER, &[])
            .context("PQ passthrough swizzle shader compile failed")?;
        // scRGB carries its own 80 cd/m² anchor and is HDR content, so neither
        // variant takes `reference_white` or `saturation` (both are SDR-only
        // knobs).
        let scrgb_decode_shader = gles
            .compile_custom_texture_shader(SCRGB_DECODE_SHADER, &[])
            .context("scRGB decode shader compile failed")?;
        let scrgb_to_pq_shader = gles
            .compile_custom_texture_shader(SCRGB_TO_PQ_SHADER, &[])
            .context("fused scRGB→PQ shader compile failed")?;
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
                    UniformName::new("mask_prev", UniformType::_1i),
                    UniformName::new("mask_mul", UniformType::_2f),
                    UniformName::new("mask_add", UniformType::_2f),
                    UniformName::new("mask_dilate", UniformType::_2f),
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
                pending_frame_roots: Vec::new(),
                pending_direct: false,
                direct_damage_ref: None,
                profile: RenderProfile::new(),
                damage_tracker: DamageTracker::new(),
                prev_layer_masks: HashMap::new(),
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
            snap_preview: None,
            dnd_icon: None,
            animations: AnimationsConfig::default(),
            decoration: DecorationConfig::default(),
            win_anims: HashMap::new(),
            pending_open: HashMap::new(),
            tearing: TearingMode::default(),
            tearing_hints: HashSet::new(),
            no_anim: NoAnim::None,
            closing: Vec::new(),
            pending_layer_open: HashSet::new(),
            layer_anims: HashMap::new(),
            closing_layers: Vec::new(),
            blur_down,
            blur_up,
            mask_blur_shader,
            mask_blur_shader_hdr,
            round_tex_shader,
            round_blur_shader,
            blur_scratch: HashMap::new(),
            hdr_encode_shader,
            screenshot_tonemap_shader,
            segment_shader,
            blank_tex,
            sdr_decode_shader,
            sdr_to_pq_shader,
            hdr_decode_shader,
            hdr_decode_swizzle_shader,
            pq_passthrough_swizzle_shader,
            scrgb_decode_shader,
            scrgb_to_pq_shader,
            round_tex_shader_hdr,
            round_tex_shader_linear,
            round_blur_shader_hdr,
            hdr_scene: HashMap::new(),
            wintex_cache: HashMap::new(),
            titlebar: TitlebarConfig::default(),
            fonts: FontState::Unscanned,
            bar_cache: HashMap::new(),
            hovered_button: None,
            icon_cache: HashMap::new(),
            no_occlusion: std::env::var_os("LIBRELAND_NO_OCCLUSION").is_some(),
            no_wintex_cache: std::env::var_os("LIBRELAND_NO_WINTEX_CACHE").is_some(),
            no_damage: std::env::var_os("LIBRELAND_NO_DAMAGE").is_some(),
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
                .render_output(
                    idx,
                    &[],
                    &[],
                    &[],
                    false,
                    &[],
                    &SurfaceEncodings::default(),
                    false,
                    None,
                    FramePurpose::Present,
                )
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
        enc: &SurfaceEncodings,
        compose_cursor: bool,
        output: Option<&Output>,
        purpose: FramePurpose,
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
            enc,
            compose_cursor,
            output,
            purpose,
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
    ///
    /// Returns the root surfaces whose frame was just presented — so the
    /// caller (which holds `State`) can signal their `wp_fifo` barriers and
    /// explicit-sync release points — plus whether that frame was
    /// DIRECT-SCANOUT (client buffer on the plane: release points must NOT
    /// be signalled for it, the buffer is what's on screen).
    pub fn frame_submitted(
        &mut self,
        crtc: crtc::Handle,
        present_time: Duration,
        seq: u32,
        base_flags: PresentKind,
    ) -> (Vec<WlSurface>, bool) {
        let Some(o) = self.outputs.iter_mut().find(|o| o.crtc == crtc) else {
            return (Vec::new(), false);
        };
        if let Err(err) = o.surface.frame_submitted() {
            warn!(error = %err, crtc = ?crtc, "frame_submitted failed");
        }
        // Fire the frame callbacks queued when this flip was submitted —
        // at the actual vblank, so clients are paced to real presents
        // (see `pending_frame_roots` for why queue-time firing is wrong).
        let roots = std::mem::take(&mut o.pending_frame_roots);
        if let Some(mut feedback) = o.pending_feedback.take() {
            // refresh_mhz is milli-Hz (144 Hz = 144_000); the frame period is
            // 1/Hz = 1000/mHz seconds.
            let period = Duration::from_secs_f64(1000.0 / f64::from(o.refresh_mhz.max(1)));
            // Deliberately Fixed even while adaptive sync is engaged.
            // `Refresh::Variable` would be the honest answer under VRR,
            // but Wine's Vulkan present-timing (winevulkan asserts on any
            // driver error) falls over the moment presented events stop
            // carrying a fixed cadence — an HDR game crashed within
            // seconds of gameplay on Variable, and ran for hours on
            // Fixed. The mode period is what every other consumer
            // (frame schedulers, video players) expects as an upper
            // bound, so Fixed is also the safer lie.
            feedback.presented(
                Time::<Monotonic>::from(present_time),
                Refresh::fixed(period),
                u64::from(seq),
                base_flags,
            );
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "wl_callback.done takes u32 ms which the spec expects to wrap freely (~50d period)"
        )]
        let elapsed_ms = self.start.elapsed().as_millis() as u32;
        for surface in &roots {
            send_frame_callbacks(surface, elapsed_ms);
        }
        let direct = self
            .outputs
            .iter()
            .find(|o| o.crtc == crtc)
            .is_some_and(|o| o.pending_direct);
        (roots, direct)
    }

    /// Every output's CRTC, for the driver to iterate when scheduling
    /// redraws across all outputs.
    /// Monotonic milliseconds since renderer start — the same clock the
    /// vblank path stamps `wl_callback.done` with, so offscreen-heartbeat
    /// callbacks (see `wayland::commit`) tick the same timeline clients
    /// already observe.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "wl_callback.done takes u32 ms which the spec expects to wrap freely (~50d period)"
    )]
    pub fn frame_time_ms(&self) -> u32 {
        self.start.elapsed().as_millis() as u32
    }

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
            pending_frame_roots: Vec::new(),
            pending_direct: false,
            direct_damage_ref: None,
            profile: RenderProfile::new(),
            damage_tracker: DamageTracker::new(),
            prev_layer_masks: HashMap::new(),
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
        // The name-keyed offscreens too — an HDR scene buffer is a
        // full-resolution fp16 texture; leaving it behind on unplug
        // parks tens of MB of GPU memory for the session.
        self.hdr_scene.remove(&removed.name);
        self.sdr_capture.remove(&removed.name);
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
        // Output-local coordinates shift wholesale on a reflow.
        self.invalidate_damage();
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
        self.invalidate_damage();
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

    /// Set (or clear) the quick-tile preview — the rect a dragged window
    /// would snap to on release, in absolute compositor coords.
    ///
    /// Returns whether it *changed*, so the caller can redraw only then:
    /// this is set from every pointer motion during a drag, and the
    /// answer is the same for most of them.
    pub fn set_snap_preview(&mut self, rect: Option<Rectangle<i32, Physical>>) -> bool {
        if self.snap_preview == rect {
            return false;
        }
        self.snap_preview = rect;
        true
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
    /// The rasterized titlebar for `key`, drawing it if it isn't cached.
    ///
    /// `None` on an upload failure, which the caller treats as "no bar
    /// this frame" — a window without its titlebar is still usable, and
    /// the alternative is refusing to draw the window at all.
    fn bar_texture(
        &mut self,
        key: u64,
        width: i32,
        height: i32,
        title: &str,
        focused: bool,
        font_px: f32,
        state: BarState,
        app_id: Option<&str>,
    ) -> Option<GlesTexture> {
        if let Some(tex) = self.bar_cache.get(&key) {
            return Some(tex.clone());
        }
        // Bounded so a client with an animating title (a progress
        // percentage, a clock) can't grow this without limit. Cleared
        // wholesale rather than evicted one at a time: the bars are
        // small, and picking a victim needs recency bookkeeping that
        // costs more than re-rasterizing the handful still on screen.
        if self.bar_cache.len() >= BAR_CACHE_MAX {
            debug!(
                entries = self.bar_cache.len(),
                "titlebar: cache full, clearing"
            );
            self.bar_cache.clear();
        }
        if matches!(self.fonts, FontState::Unscanned) {
            let loaded = Fonts::load();
            if loaded.is_none() {
                warn!("titlebar: no usable UI font found; bars will draw without titles");
            }
            self.fonts = FontState::Scanned(loaded);
        }
        // The bar's colours come from the border fill for the same focus
        // state, so the frame and the bar are one palette with no second
        // set of config keys to keep in sync. A gradient contributes its
        // top stop — the bar is short enough that sampling the ramp
        // across it would read as flat anyway.
        let border_rgb = match if focused {
            &self.border.active
        } else {
            &self.border.inactive
        } {
            Fill::Solid(rgb) => *rgb,
            Fill::VerticalGradient { top, .. } => *top,
        };
        // Resolved before the borrow below: this needs `&mut self` for
        // the cache, and `fonts` holds a shared borrow across the call.
        let icon = app_id.and_then(|id| self.app_icon(id, icon_side(height)));
        let fonts = match &self.fonts {
            FontState::Scanned(f) => f.as_ref(),
            FontState::Unscanned => None,
        };
        let rgba = rasterize_bar(
            fonts,
            width,
            height,
            title,
            BarStyle::from_border(border_rgb, focused),
            &self.titlebar.buttons,
            font_px,
            state,
            icon.as_deref(),
        );
        let size = Size::<i32, smithay::utils::Buffer>::from((width.max(1), height.max(1)));
        match self
            .gles
            .import_memory(&rgba, Fourcc::Abgr8888, size, false)
        {
            Ok(tex) => {
                self.bar_cache.insert(key, tex.clone());
                Some(tex)
            }
            Err(err) => {
                warn!(error = %err, "titlebar: texture upload failed");
                None
            }
        }
    }

    /// The decoded, slot-sized icon for `app_id`, looking it up once.
    ///
    /// Themes that ship only SVG (Breeze is one — 19827 SVGs, no PNGs)
    /// resolve to `None`, and the bar simply draws no icon. A
    /// placeholder box would be worse: an empty square in a titlebar
    /// reads as a broken icon rather than as a missing one.
    fn app_icon(&mut self, app_id: &str, side: u32) -> Option<std::sync::Arc<crate::icon::Icon>> {
        let key = (app_id.to_owned(), side);
        if let Some(hit) = self.icon_cache.get(&key) {
            return hit.clone();
        }
        let loaded = crate::icon::lookup(app_id, side)
            .and_then(|path| {
                let icon = crate::icon::load(&path);
                if icon.is_none() {
                    debug!(app_id, path = %path.display(), "titlebar: icon decode failed");
                }
                icon
            })
            .map(|icon| std::sync::Arc::new(crate::icon::resize(&icon, side)));
        if loaded.is_none() {
            debug!(app_id, side, "titlebar: no raster icon found");
        }
        self.icon_cache.insert(key, loaded.clone());
        loaded
    }

    /// Record which titlebar button the pointer is over, so its bar can
    /// draw the highlight. Returns whether it changed — the caller
    /// redraws on `true`.
    pub fn set_hovered_button(
        &mut self,
        hovered: Option<(ObjectId, crate::config::TitlebarButton)>,
    ) -> bool {
        if self.hovered_button == hovered {
            return false;
        }
        self.hovered_button = hovered;
        true
    }

    /// Swap the titlebar settings (live reload). Drops the rasterized
    /// bars, since height, font size and the button set all change their
    /// pixels; they re-rasterize on the next frame that needs them.
    pub fn set_titlebar(&mut self, cfg: TitlebarConfig) {
        if self.titlebar == cfg {
            return;
        }
        self.titlebar = cfg;
        self.bar_cache.clear();
        self.icon_cache.clear();
        // Every cached decoration offscreen has a bar drawn into it.
        self.wintex_cache.clear();
        self.invalidate_damage();
    }

    pub fn set_decoration(&mut self, cfg: DecorationConfig) {
        self.decoration = cfg;
        self.invalidate_damage();
    }

    /// Replace the tearing policy (`misc.tearing`). Live config reload; the
    /// next frame's [`Self::apply_tearing`] settles each output.
    pub fn set_tearing_mode(&mut self, mode: TearingMode) {
        self.tearing = mode;
    }

    /// Record a surface's `wp_tearing_control_v1` presentation hint.
    /// `immediate` is the client asking to be shown as soon as possible,
    /// tearing if need be; `false` restores vsync for it.
    pub fn set_tearing_hint(&mut self, surface: &WlSurface, immediate: bool) {
        if immediate {
            self.tearing_hints.insert(surface.id());
        } else {
            self.tearing_hints.remove(&surface.id());
        }
    }

    /// Forget a dead surface's tearing hint.
    pub fn forget_surface_scanout_state(&mut self, id: &ObjectId) {
        self.tearing_hints.remove(id);
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
    /// A layer surface just mapped: play its open animation next frame.
    pub fn mark_layer_open(&mut self, surface: &WlSurface) {
        self.pending_layer_open.insert(surface.id());
    }

    /// A layer surface is going away: snapshot its last frame and fade it
    /// back out toward the edge it came from.
    ///
    /// Called while the surface is still alive (from `layer_destroyed`), so
    /// there is still a buffer to capture. A failure anywhere just means the
    /// surface vanishes instantly, which is the old behaviour.
    pub fn mark_layer_closing(&mut self, surface: &WlSurface, rect: Rectangle<i32, Physical>) {
        let cfg = self.animations.clone();
        if !cfg.enabled || !cfg.layer_close.enabled || rect.size.w <= 0 || rect.size.h <= 0 {
            return;
        }
        self.pending_layer_open.remove(&surface.id());
        self.layer_anims.remove(&surface.id());

        let center = Point::<i32, Physical>::from((
            rect.loc.x + rect.size.w / 2,
            rect.loc.y + rect.size.h / 2,
        ));
        let Some(output) = self.output_at(center) else {
            return;
        };
        let (scale, out_rect) = (output.scale, output.compositor);
        let Some(texture) = self.snapshot_surface(surface, rect.size, scale) else {
            return;
        };
        let now = self.start.elapsed().as_secs_f64();
        debug!(
            surface = ?surface.id(),
            edge = ?LayerEdge::of(rect, out_rect),
            ?rect,
            "layer close animation started"
        );
        self.closing_layers.push(ClosingLayer {
            texture,
            rect,
            edge: LayerEdge::of(rect, out_rect),
            anim: Animation::start(now, cfg.layer_close.duration_secs(), cfg.layer_close.curve),
        });
    }

    pub fn mark_open(&mut self, surface: &WlSurface) {
        self.pending_open.insert(surface.id(), false);
    }

    /// Like [`Self::mark_open`], but for a window coming back from being
    /// minimized: it rises into place instead of just scaling up, mirroring
    /// the sink it left on.
    ///
    /// The arrival runs on `window_open`'s timing rather than
    /// `window_minimize`'s. They are the same gesture reversed, but not the
    /// same *feel*: leaving wants to be brief and arriving wants to be seen,
    /// which is exactly the split those two specs already encode.
    pub fn mark_restore(&mut self, surface: &WlSurface) {
        self.pending_open.insert(surface.id(), true);
    }

    /// Set (`Some`) or clear (`None`) the window being interactively
    /// moved/resized, which draws 1:1 instead of animating its rect.
    pub fn set_no_anim_move(&mut self, surface: Option<&WlSurface>) {
        self.no_anim = surface.map_or(NoAnim::None, |s| NoAnim::One(s.id()));
    }

    /// Suppress the move animation for *every* window, not just one.
    /// An interactive tiled resize reflows the dragged window's
    /// neighbours as well: animating them would let the edge they share
    /// with it trail the divider the user is dragging, which reads as
    /// the cells coming apart. Cleared when the drag ends.
    pub fn set_no_anim_all(&mut self, on: bool) {
        self.no_anim = if on { NoAnim::All } else { NoAnim::None };
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
        self.start_ghost(surface, false);
    }

    /// Begin a minimize animation: the same snapshot ghost as a close, but
    /// shrinking harder and sinking as it fades, so being put away doesn't
    /// look like being destroyed. Must run *before* the layout hides the
    /// window, while its buffer and its last drawn rect are still around.
    pub fn start_minimize(&mut self, surface: &WlSurface) {
        self.start_ghost(surface, true);
    }

    /// Snapshot `surface` where it sits and register the fading ghost that
    /// plays out after it is gone. A no-op (instant close/minimize) if the
    /// animation is disabled, the window isn't tracked, or its buffer has
    /// already gone.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "physical pixel sizes derived from output-bounded rects are small non-negative values"
    )]
    fn start_ghost(&mut self, surface: &WlSurface, minimizing: bool) {
        let cfg = self.animations.clone();
        let spec = if minimizing {
            cfg.window_minimize
        } else {
            cfg.window_close
        };
        if !(cfg.enabled && spec.enabled) {
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

        let Some(texture) = self.snapshot_surface(surface, inner.size, scale) else {
            return; // no buffer left to snapshot — close instantly
        };

        let now = self.start.elapsed().as_secs_f64();
        self.closing.push(ClosingWindow {
            texture,
            rect: inner,
            anim: Animation::start(now, spec.duration_secs(), spec.curve),
            scale_to: if minimizing {
                MINIMIZE_SCALE_TO
            } else {
                OPEN_SCALE_FROM
            },
            sink: if minimizing {
                (f64::from(inner.size.h) * MINIMIZE_SINK_FRAC) as i32
            } else {
                0
            },
        });
    }

    /// Render `surface`'s current tree into an offscreen texture of `size`
    /// (compositor pixels, scaled by `scale`), for a close animation to keep
    /// drawing after the surface itself is gone.
    ///
    /// `None` on any failure — an unmapped surface with no buffer left, or a
    /// GL error — which callers treat as "no animation", not as an error.
    fn snapshot_surface(
        &mut self,
        surface: &WlSurface,
        size: Size<i32, Physical>,
        scale: f64,
    ) -> Option<GlesTexture> {
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
            return None;
        }

        let tex_size = Size::<i32, smithay::utils::Buffer>::from((
            scale_f(f64::from(size.w), scale).max(1),
            scale_f(f64::from(size.h), scale).max(1),
        ));
        let mut texture = self
            .gles
            .create_buffer(Fourcc::Abgr8888, tex_size)
            .inspect_err(|err| warn!(%err, "close snapshot: create_buffer failed"))
            .ok()?;
        let phys = Size::<i32, Physical>::from((tex_size.w, tex_size.h));
        let full = [Rectangle::<i32, Physical>::from_size(phys)];
        {
            let mut target = self
                .gles
                .bind(&mut texture)
                .inspect_err(|err| warn!(%err, "close snapshot: bind failed"))
                .ok()?;
            let mut frame = self
                .gles
                .render(&mut target, phys, Transform::Normal)
                .inspect_err(|err| warn!(%err, "close snapshot: render failed"))
                .ok()?;
            let _ = frame.clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &full);
            draw_render_elements::<GlesRenderer, _, _>(&mut frame, scale, &elements, &full)
                .inspect_err(|err| warn!(%err, "close snapshot: draw failed"))
                .ok()?;
            // Same-context sequential GL: the texture is sampled by later
            // draws on this context, so the sync point needn't be awaited.
            drop(
                frame
                    .finish()
                    .inspect_err(|err| warn!(%err, "close snapshot: finish failed"))
                    .ok()?,
            );
        }
        Some(texture)
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
                    &[
                        Uniform::new("reference_white", reference_white),
                        Uniform::new("knee", SCREENSHOT_TONEMAP_KNEE),
                    ],
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

    /// Composite one workspace exactly as it would appear on `output` —
    /// wallpaper, layer surfaces, windows, decorations, blur and all — and
    /// read it back as premultiplied RGBA8. Nothing is presented.
    ///
    /// This is the real render path with `FramePurpose::CaptureOnly`, not a
    /// second implementation of it: `placements` is already how a frame
    /// says what to draw (the workspace slide hands it two workspaces at
    /// once), so a workspace that isn't on screen composites the same way
    /// the visible one does, and cannot drift from it. HDR outputs tonemap
    /// through the same shader screenshots use.
    ///
    /// The cursor is left out — it belongs to wherever the pointer actually
    /// is, not to the workspace being photographed.
    ///
    /// Returns `(width, height, rgba)` with rows top-down.
    pub fn capture_workspace(
        &mut self,
        output: &str,
        placements: &[Placement],
        layers: &[LayerPlacement],
        enc: &SurfaceEncodings,
    ) -> Result<(i32, i32, Vec<u8>)> {
        let idx = self
            .outputs
            .iter()
            .position(|o| o.name == output)
            .with_context(|| format!("no output named {output}"))?;
        let mode_size = self.outputs[idx].mode_size;
        let specs = [CaptureSpec {
            region: Rectangle::from_size(mode_size),
            fourcc: Fourcc::Abgr8888,
            target: CaptureTarget::Shm,
        }];
        let (results, _) = self.render_output(
            idx,
            placements,
            layers,
            &[],
            true,
            &specs,
            enc,
            false,
            None,
            FramePurpose::CaptureOnly,
        )?;
        match results.into_iter().next() {
            Some(CaptureOutcome::Shm {
                bytes,
                width,
                height,
            }) => Ok((
                i32::try_from(width).unwrap_or(mode_size.w),
                i32::try_from(height).unwrap_or(mode_size.h),
                bytes,
            )),
            _ => anyhow::bail!("workspace capture read-back failed"),
        }
    }

    /// Draw a colour-managed window's elements into a fresh fp16 offscreen,
    /// decoded to linear BT.2020 — the same treatment its surface gets in an
    /// HDR output's scene, in isolation. The caller tonemaps it down.
    ///
    /// fp16 rather than 8-bit because the decode's output doesn't fit in
    /// 8 bits: an HDR highlight lands several times above diffuse white and
    /// has to survive until the shoulder can roll it back in.
    fn decode_to_fp16(
        &mut self,
        surface: &WlSurface,
        elements: &[WaylandSurfaceRenderElement<GlesRenderer>],
        scale: f64,
        encoding: Encoding,
        tex_size: Size<i32, smithay::utils::Buffer>,
    ) -> Result<GlesTexture> {
        let decode = match encoding {
            Encoding::Scrgb => self.scrgb_decode_shader.clone(),
            _ if window_buffer_rb_swapped(surface) => self.hdr_decode_swizzle_shader.clone(),
            _ => self.hdr_decode_shader.clone(),
        };
        let phys = Size::<i32, Physical>::from((tex_size.w, tex_size.h));
        let full = [Rectangle::<i32, Physical>::from_size(phys)];
        let mut scene = self
            .gles
            .create_buffer(Fourcc::Abgr16161616f, tex_size)
            .context("capture_window: create fp16 scene")?;
        {
            let mut target = self
                .gles
                .bind(&mut scene)
                .context("capture_window: bind fp16 scene")?;
            let mut frame = self
                .gles
                .render(&mut target, phys, Transform::Normal)
                .context("capture_window: fp16 render")?;
            frame
                .clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &full)
                .context("capture_window: fp16 clear")?;
            frame.override_default_tex_program(decode, Vec::new());
            draw_render_elements::<GlesRenderer, _, _>(&mut frame, scale, elements, &full)
                .context("capture_window: fp16 draw")?;
            let _ = frame.finish().context("capture_window: fp16 finish")?;
        }
        Ok(scene)
    }

    /// The SDR reference white configured for `output`, in cd/m². Falls back
    /// to the primary output's for an unknown or absent name — a window
    /// capture is output-independent, so it may have no screen to ask.
    fn reference_white_for(&self, output: Option<&str>) -> u32 {
        output
            .and_then(|name| self.outputs.iter().find(|o| o.name == name))
            .or_else(|| self.outputs.get(self.primary_idx))
            .map_or(203, |o| o.hdr_reference_white)
    }

    /// Render `surface`'s current surface tree into an offscreen and read it
    /// back as premultiplied-RGBA8 bytes — an on-demand per-window thumbnail
    /// for the IPC. Independent of any output: a window on another workspace
    /// or screen still captures its last-committed content, in isolation (no
    /// other windows, no cursor). The longest side is capped at `max`
    /// (downscaled, never upscaled). Returns `(width, height, rgba)` —
    /// premultiplied RGBA8, bottom-up (the encoder flips it upright).
    ///
    /// `encoding` is the decode the window's buffer needs (see
    /// `State::window_encoding`). A colour-managed one — an HDR game or
    /// video — holds PQ/BT.2100 or scRGB pixels, and sampling those with the
    /// sRGB default reads encoded values as if they were display colours:
    /// wrong transfer curve, wrong primaries, not subtly. Those go through
    /// the same decode → linear BT.2020 → tonemap the on-screen path and the
    /// screenshots use, so an HDR window's thumbnail matches its screenshot.
    ///
    /// SDR windows are copied through untouched, and deliberately *don't*
    /// get the screenshot's highlight roll-off: a thumbnail is displayed
    /// back on the desktop, where it is re-encoded from sRGB like any other
    /// SDR content, so tone-mapping it here would render every preview
    /// dimmer than the window it depicts.
    ///
    /// `output` names the screen whose reference white the tonemap uses;
    /// unknown or `None` falls back to the primary output's.
    pub fn capture_window(
        &mut self,
        surface: &WlSurface,
        max: i32,
        encoding: Encoding,
        output: Option<&str>,
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
        // Cloned out (both are Arc-backed / Copy) before the `self.gles`
        // borrows below, which would otherwise conflict with reading `self`.
        let tonemap = self.screenshot_tonemap_shader.clone();
        #[allow(
            clippy::cast_precision_loss,
            reason = "reference white is a small cd/m² value, exact in f32"
        )]
        let reference_white = self.reference_white_for(output) as f32;
        let tex_size = Size::<i32, smithay::utils::Buffer>::from((tw, th));
        let phys = Size::<i32, Physical>::from((tw, th));
        let full = [Rectangle::<i32, Physical>::from_size(phys)];

        let mut texture: GlesTexture = self
            .gles
            .create_buffer(Fourcc::Abgr8888, tex_size)
            .context("capture_window: create_buffer")?;

        // A colour-managed window is composited into an fp16 linear-BT.2020
        // scene of its own first, exactly as it would be on an HDR output,
        // then tonemapped down into the 8-bit buffer. Two passes, because
        // the decode's output doesn't fit in 8 bits: an HDR highlight lands
        // several times above diffuse white and has to survive until the
        // shoulder can roll it back in.
        let hdr_scene = (encoding != Encoding::Sdr)
            .then(|| self.decode_to_fp16(surface, &elements, scale, encoding, tex_size))
            .transpose()?;

        let mut target = self.gles.bind(&mut texture).context("capture_window: bind")?;
        {
            let mut frame = self
                .gles
                .render(&mut target, phys, Transform::Normal)
                .context("capture_window: render")?;
            frame
                .clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &full)
                .context("capture_window: clear")?;
            if let Some(scene) = &hdr_scene {
                let src = Rectangle::<f64, smithay::utils::Buffer>::from_size(Size::from((
                    f64::from(tw),
                    f64::from(th),
                )));
                frame
                    .render_texture_from_to(
                        scene,
                        src,
                        Rectangle::from_size(phys),
                        &full,
                        &full,
                        Transform::Normal,
                        1.0,
                        Some(&tonemap),
                        &[
                            Uniform::new("reference_white", reference_white),
                            Uniform::new("knee", SCREENSHOT_TONEMAP_KNEE),
                        ],
                    )
                    .context("capture_window: tonemap")?;
            } else {
                draw_render_elements::<GlesRenderer, _, _>(&mut frame, scale, &elements, &full)
                    .context("capture_window: draw")?;
            }
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
        let resize_enabled = cfg.enabled && cfg.window_resize.enabled;
        let open_enabled = cfg.enabled && cfg.window_open.enabled;
        let focus_enabled = cfg.enabled && cfg.focus.enabled;
        let no_anim = self.no_anim.clone();

        let mut draws = Vec::with_capacity(placements.len());
        for p in placements {
            let id = p.surface.id();
            let target = p.cell_rect;
            // The interactively dragged window tracks the cursor 1:1.
            let snap = no_anim.covers(&id);
            let entry = self.win_anims.entry(id.clone()).or_insert_with(|| WindowAnim {
                surface: p.surface.clone(),
                target,
                displayed: target,
                move_from: target,
                move_anim: None,
                resize_anim: None,
                open_anim: None,
                open_rise: 0,
                // Seeded to the window's current focus so the very first
                // frame doesn't fade in from "unfocused".
                focused: p.focused,
                focus_anim: None,
                focus_from: f32::from(u8::from(p.focused)),
                focus_now: f32::from(u8::from(p.focused)),
            });

            entry.retarget(
                now,
                target,
                (move_enabled && !snap).then_some(cfg.window_move),
                (resize_enabled && !snap).then_some(cfg.window_resize),
            );

            // A just-mapped window starts opening the first frame it's
            // here. Consume the mark regardless so a disabled open
            // animation doesn't leave it pending forever.
            if let Some(restoring) = self.pending_open.remove(&id)
                && open_enabled
            {
                entry.open_anim = Some(Animation::start(
                    now,
                    cfg.window_open.duration_secs(),
                    cfg.window_open.curve,
                ));
                // Coming back from a minimize: rise the same distance it
                // sank, so the two halves of the gesture mirror.
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "a fraction of a window height — small, non-negative"
                )]
                let rise = (f64::from(target.size.h) * MINIMIZE_SINK_FRAC) as i32;
                entry.open_rise = if restoring { rise } else { 0 };
            }

            entry.advance_geometry(now);

            entry.advance_focus(now, p.focused, focus_enabled, cfg.focus);

            // Open: fade + scale-about-centre layered on the displayed
            // rect.
            let (mut effective, mut alpha) = (entry.displayed, 1.0_f32);
            if let Some(a) = entry.open_anim {
                let v = a.value(now);
                alpha = a.alpha(now);
                // The scale *does* take the raw value: an overshoot curve
                // popping a window a touch past full size and settling back
                // is the whole point of using one.
                effective = scale_rect_about_center(entry.displayed, lerp(OPEN_SCALE_FROM, 1.0, v));
                // A restore starts low and rises to nothing. `open_rise` is
                // 0 for an ordinary map, so this is inert there — including
                // under an overshooting curve, where `v > 1` would otherwise
                // lift the window past its own rect.
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "a fraction of a window height — small, non-negative"
                )]
                let rise = ((1.0 - v).max(0.0) * f64::from(entry.open_rise)) as i32;
                effective.loc.y += rise;
                if a.done(now) {
                    entry.open_anim = None;
                    entry.open_rise = 0;
                }
            }
            // Workspace slide: a uniform vertical offset applied *after*
            // the per-window animation (so it doesn't perturb move/open),
            // translating the whole workspace during a switch.
            effective.loc += p.slide;
            draws.push(WinDraw {
                effective,
                alpha,
                focus: entry.focus_now,
            });
        }

        // Drop tracking for windows whose surface has died.
        self.win_anims.retain(|_, w| w.surface.alive());
        // Drop finished close-out ghosts (frees their snapshot textures).
        self.closing.retain(|c| !c.anim.done(now));
        self.closing_layers.retain(|c| !c.anim.done(now));
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

    /// Drop per-surface render caches (the decoration offscreen). Called
    /// when a surface is destroyed so the cache never accumulates dead
    /// entries (a `wl_surface` id is not reused after destruction).
    pub fn forget_surface(&mut self, surface: &WlSurface) {
        self.wintex_cache.remove(&surface.id());
        // A surface that was marked for an open animation but destroyed
        // before ever being placed would otherwise pin its id here forever.
        self.pending_open.remove(&surface.id());
    }

    /// Repaint every output's next frame in full and restart damage
    /// diffing from scratch. Called whenever pixels can change without any
    /// tracked input changing: appearance/decoration reload (border
    /// colours, opacity), output layout changes, and VT re-activation
    /// (the kernel may have scribbled over our buffers while away).
    pub fn invalidate_damage(&mut self) {
        for o in &mut self.outputs {
            o.damage_tracker = DamageTracker::new();
        }
    }

    /// Scanout-capable `(fourcc, modifier)` pairs for each output, by
    /// connector name: that output's plane formats, intersected with what the
    /// renderer can import as a texture (the composite fallback must be able
    /// to draw the very same buffers on frames the plane can't take), with
    /// implicit modifiers dropped — the whole point is steering clients
    /// toward *explicit* plane-compatible modifiers; an Invalid entry would
    /// invite the implicit allocations that can never be latched. Feeds the
    /// per-surface dmabuf-feedback scanout tranche (see wayland.rs).
    ///
    /// Keyed per output because planes differ between connectors — and on a
    /// multi-GPU machine between devices. A single set describing the primary
    /// output is the wrong answer for a game fullscreened anywhere else: it
    /// would allocate modifiers that output's plane never advertised, fail
    /// the import every frame, and composite forever with no indication why.
    pub fn scanout_formats_by_output(&self) -> Vec<(String, Vec<Format>)> {
        self.outputs
            .iter()
            .map(|o| (o.name.clone(), self.scanout_formats_of(o)))
            .collect()
    }

    /// One output's scanout formats by connector name, for the hot-plug path
    /// that has to build its dmabuf-feedback variant after startup. Empty if
    /// no such output is bound.
    pub fn scanout_formats_for(&self, name: &str) -> Vec<Format> {
        self.outputs
            .iter()
            .find(|o| o.name == name)
            .map(|o| self.scanout_formats_of(o))
            .unwrap_or_default()
    }

    /// One output's primary-plane formats, narrowed to those our renderer can
    /// also import. Implicit (`Invalid`) modifiers are dropped: KMS can't
    /// scan out a buffer whose layout it can't name, so advertising them in a
    /// *scanout* tranche would be a lie.
    fn scanout_formats_of(&self, output: &OutputRender) -> Vec<Format> {
        let importable = self.gles.dmabuf_formats();
        output
            .surface
            .plane_formats()
            .into_iter()
            .filter(|f| {
                f.modifier != smithay::backend::allocator::Modifier::Invalid
                    && importable.contains(f)
            })
            .collect()
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

    /// The scale Xwayland's pixel space runs at: the exact ratio between
    /// the primary output's DRM mode and the compositor rect the layout
    /// hands out for it. NOT the same number as [`Self::primary_scale`],
    /// which stays whatever the user configured.
    ///
    /// X11 clients render at the output's *physical* resolution, and
    /// Xwayland takes its screen size straight from the `wl_output` mode
    /// (3840×2160 here — verified with `xrandr`). Every window rect we
    /// send it, though, is a compositor rect that smithay multiplies by
    /// this scale and rounds. Handing over the configured scale is a
    /// pixel short whenever the mode doesn't divide evenly by it:
    /// `round(3840 / 1.35) = 2844` logical, and `round(2844 × 1.35)`
    /// comes back as **3839** — so a window configured to *fill* the
    /// output lands one pixel narrower than the X screen it is meant to
    /// cover.
    ///
    /// That single pixel is a live-lock with Wine. `is_window_rect_full_screen`
    /// wants the window to cover the monitor on all four edges; 3839 < 3840
    /// fails it, so Wine strips `_NET_WM_STATE_FULLSCREEN` — we honour the
    /// unfullscreen and tile the window — the game immediately re-asserts
    /// exclusive fullscreen — we fullscreen it to 3839 again. Measured on
    /// Honkai: Star Rail under Proton-Wine: up to 19 fullscreen↔tiled
    /// round trips per second, i.e. a violently flickering game, with the
    /// output's `Auto` VRR flipping on every one of them.
    ///
    /// Deriving the scale from the two sizes closes the round trip
    /// instead: 3840 / 2844 = 1.350211… maps 2844 back to exactly 3840.
    /// The larger of the two axis ratios wins so neither axis can come up
    /// short (the other overshoots by a fraction of a pixel), and X11
    /// buffers for a fullscreen window now match the mode exactly, which
    /// is also what the 1:1 scanout fast paths require.
    pub fn xwayland_client_scale(&self) -> f64 {
        self.outputs
            .get(self.primary_idx)
            .map_or(1.0, |o| client_scale_for(o.mode_size, o.compositor_size, o.scale))
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

    /// Service screencopy captures from the client buffer we just put on the
    /// primary plane, instead of compositing the scene to read it back.
    ///
    /// Direct-scanout eligibility already proved this buffer *is* the whole
    /// output, so the capture is one textured blit rather than a full
    /// wallpaper-plus-windows-plus-effects pass. It runs after the flip, so a
    /// slow consumer costs the recorder frames, never the game.
    ///
    /// Any GL failure fails just the captures, not the frame — the buffer is
    /// already scanning out by the time we get here.
    fn capture_direct(
        &mut self,
        idx: usize,
        dmabuf: &Dmabuf,
        place: DirectPlacement,
        captures: &[CaptureSpec],
    ) -> Vec<CaptureOutcome> {
        let output_name = self.outputs[idx].name.clone();
        let mode_size = self.outputs[idx].mode_size;
        let failed = || captures.iter().map(|_| CaptureOutcome::Failed).collect();

        let tex = match self.gles.import_dmabuf(dmabuf, None) {
            Ok(tex) => tex,
            Err(err) => {
                warn!(error = %err, output = %output_name, "screencopy: importing the scanned-out buffer failed");
                return failed();
            }
        };

        // Output-physical → buffer pixels. The eligibility gate pinned the
        // source to exactly the mode's worth of pixels, so this is a
        // translation in practice; deriving it anyway keeps the two in step
        // if the gate ever loosens.
        let (sx, sy) = (
            place.src.size.w / f64::from(mode_size.w.max(1)),
            place.src.size.h / f64::from(mode_size.h.max(1)),
        );
        let src_of = |region: Rectangle<i32, Physical>| {
            Rectangle::<f64, smithay::utils::Buffer>::new(
                (
                    place.src.loc.x + f64::from(region.loc.x) * sx,
                    place.src.loc.y + f64::from(region.loc.y) * sy,
                )
                    .into(),
                (f64::from(region.size.w) * sx, f64::from(region.size.h) * sy).into(),
            )
        };

        // Shm captures have to read back from an FBO, so they go through the
        // per-output scratch buffer (allocated once, reused across frames);
        // dmabuf captures render straight into the consumer's own buffer.
        let needs_scratch = captures
            .iter()
            .any(|s| matches!(s.target, CaptureTarget::Shm));
        if needs_scratch && !self.ensure_capture_scratch(&output_name, mode_size) {
            return failed();
        }

        let mut results = Vec::with_capacity(captures.len());
        for spec in captures {
            let src = src_of(spec.region);
            let dst = Rectangle::<i32, Physical>::from_size(spec.region.size);
            match &spec.target {
                CaptureTarget::Dmabuf(client) => {
                    let mut client = client.clone();
                    let Ok(mut target) = self.gles.bind(&mut client).inspect_err(|err| {
                        warn!(error = %err, output = %output_name, "screencopy: bind client dmabuf failed");
                    }) else {
                        results.push(CaptureOutcome::Failed);
                        continue;
                    };
                    results.push(
                        match blit_texture(&mut self.gles, &mut target, &tex, src, dst) {
                            Ok(()) => CaptureOutcome::Dmabuf,
                            Err(err) => {
                                warn!(error = %err, output = %output_name, "screencopy: blit to client dmabuf failed");
                                CaptureOutcome::Failed
                            }
                        },
                    );
                }
                CaptureTarget::Shm => {
                    // Render the requested region to the scratch's origin,
                    // then read that back — the same shape as the composited
                    // path, minus the composite.
                    let mut scratch = self
                        .sdr_capture
                        .remove(&output_name)
                        .expect("scratch ensured above");
                    let outcome = match self.gles.bind(&mut scratch) {
                        Ok(mut target) => {
                            match blit_texture(&mut self.gles, &mut target, &tex, src, dst) {
                                Ok(()) => {
                                    let at_origin = CaptureSpec {
                                        region: dst,
                                        fourcc: spec.fourcc,
                                        target: CaptureTarget::Shm,
                                    };
                                    capture_shm(&mut self.gles, &target, &at_origin, &output_name)
                                }
                                Err(err) => {
                                    warn!(error = %err, output = %output_name, "screencopy: blit to scratch failed");
                                    CaptureOutcome::Failed
                                }
                            }
                        }
                        Err(err) => {
                            warn!(error = %err, output = %output_name, "screencopy: bind scratch failed");
                            CaptureOutcome::Failed
                        }
                    };
                    self.sdr_capture.insert(output_name.clone(), scratch);
                    results.push(outcome);
                }
            }
        }
        results
    }

    /// Ensure the per-output 8-bit capture scratch exists at `size`,
    /// (re)allocating when absent or stale. `false` means allocation failed
    /// and the caller must fail its captures.
    fn ensure_capture_scratch(&mut self, output_name: &str, size: Size<i32, Physical>) -> bool {
        let (w, h) = (
            u32::try_from(size.w).unwrap_or(0),
            u32::try_from(size.h).unwrap_or(0),
        );
        if self
            .sdr_capture
            .get(output_name)
            .is_some_and(|t| t.width() == w && t.height() == h)
        {
            return true;
        }
        let buf_size = Size::<i32, smithay::utils::Buffer>::from((size.w, size.h));
        match self.gles.create_buffer(Fourcc::Abgr8888, buf_size) {
            Ok(b) => {
                self.sdr_capture.insert(output_name.to_string(), b);
                true
            }
            Err(err) => {
                warn!(error = %err, output = %output_name, "screencopy: scratch buffer alloc failed");
                self.sdr_capture.remove(output_name);
                false
            }
        }
    }

    /// Settle this output's tearing (async page-flip) state for the frame
    /// about to be queued.
    ///
    /// Tearing is off unless the config asks for it *and* the frame is the
    /// one it exists for: a single fullscreen window, which is also the only
    /// shape a driver will accept an async flip for (typically "the primary
    /// plane's framebuffer and nothing else changed"). `TearingMode::Always`
    /// still respects that — it means "don't wait for the client to ask", not
    /// "tear the desktop".
    fn apply_tearing(&mut self, idx: usize, solo: Option<usize>, placements: &[Placement]) {
        let want = match self.tearing {
            TearingMode::Never => false,
            // The client asked for immediate presentation via
            // wp_tearing_control_v1.
            TearingMode::Auto => solo
                .and_then(|i| placements.get(i))
                .is_some_and(|p| self.tearing_hints.contains(&p.surface.id())),
            TearingMode::Always => solo.is_some(),
        };
        self.outputs[idx].surface.set_tearing(want);
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
        enc: &SurfaceEncodings,
        compose_cursor: bool,
        present_output: Option<&Output>,
        purpose: FramePurpose,
    ) -> Result<(Vec<CaptureOutcome>, bool)> {
        let capture_only = purpose == FramePurpose::CaptureOnly;
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
        let hdr_decode_swizzle_shader = self.hdr_decode_swizzle_shader.clone();
        let pq_passthrough_swizzle_shader = self.pq_passthrough_swizzle_shader.clone();
        let scrgb_decode_shader = self.scrgb_decode_shader.clone();
        let scrgb_to_pq_shader = self.scrgb_to_pq_shader.clone();
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
        let screenshot_overlay = self.screenshot_overlay.clone();
        let segment_shader = self.segment_shader.clone();
        let blank_tex = self.blank_tex.clone();
        let snap_preview = self.snap_preview;
        // The quick-tile preview borrows the focused window's accent, so
        // the drop target reads as part of the same theme as the border
        // and titlebar. A gradient contributes its top stop, matching
        // what the titlebar does with it.
        let accent = match &self.border.active {
            Fill::Solid(rgb) => *rgb,
            Fill::VerticalGradient { top, .. } => *top,
        };
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
        let scene = self.solo_fullscreen_scene(
            idx,
            &win_draws,
            placements,
            layers,
            popups,
            hide_cursor,
            compose_cursor,
        );
        // The *strict* reading: a fullscreen window with nothing whatsoever
        // above it. Direct scanout can handle a notification or a menu by
        // handing it to an overlay plane, but the composite-side fast paths
        // below can't — the single-pass HDR program and the wallpaper skip
        // both assume the window is the only thing in the frame.
        let solo = scene
            .as_ref()
            .filter(|s| s.above.is_empty())
            .map(|s| s.solo);

        // ── Direct-scanout fast path ──────────────────────────────────
        // A settled fullscreen opaque client whose colour mode matches the
        // output: scan its buffer straight to the primary plane, skipping ALL
        // compositing (≈ zero GPU for this output). A little client content
        // drawn above it — a notification, a menu — rides overlay planes
        // instead of dragging the whole frame back through the GPU. Anything
        // else that needs compositing (animations, a non-1:1 buffer, more
        // content than there are planes, or a buffer the driver rejects)
        // falls through to the composite path below.
        //
        // A pending capture does NOT disqualify the frame: eligibility
        // already proved the client's buffer *is* the whole output, so a
        // screencopy can be served from it with one textured blit instead of
        // a full scene composite (`capture_direct`). That is the difference
        // between "screen-sharing a game costs nothing" and "screen-sharing a
        // game turns the fast path off for the whole session".
        // A capture-only frame never takes this path. It ends in a flip by
        // construction — the client's own buffer goes on the primary plane —
        // and a capture of a workspace nobody is looking at must not put
        // that workspace on the screen.
        if let Some(direct) = (!capture_only)
            .then(|| self.direct_scanout_inputs(idx, scene.as_ref(), placements, enc))
            .flatten()
        {
            // Two exceptions, both about the capture:
            // - HDR, because the client's buffer is PQ/BT.2020 and our capture
            //   path can only tonemap the *linear* scene;
            // - overlay planes, because what a capture must record is the
            //   composed result, and the composition happens in the display
            //   engine where we can't read it back.
            let capture_ok =
                captures.is_empty() || (!self.outputs[idx].hdr && direct.overlays.is_empty());
            if capture_ok {
                let n_overlays = direct.overlays.len();
                // Kept for the capture below, which needs the buffer after
                // ownership of the layer has moved into the flip.
                let capture_src = (!captures.is_empty())
                    .then(|| (direct.primary.dmabuf.clone(), direct.primary.place));

                // VRR must settle before the flip (it may promote the flip to
                // a modeset); harmlessly re-applied by the composite path on
                // a miss. Same for the tearing hint.
                self.apply_vrr(idx, placements);
                self.apply_tearing(idx, solo, placements);

                let commit = direct.commit.clone();
                match self.outputs[idx]
                    .surface
                    .try_queue_direct(direct.primary, direct.overlays)
                {
                    Ok(true) => {
                        debug!(
                            output = %output_name,
                            overlays = n_overlays,
                            tearing = self.outputs[idx].surface.tearing(),
                            "frame direct-scanned to hardware planes (no compositing)"
                        );
                        // The plane now holds this exact commit — the next
                        // direct frame can describe its damage relative to it.
                        self.outputs[idx].direct_damage_ref = Some(commit);
                        self.queue_output_frame_callbacks(idx, placements, layers, popups, out_rect);
                        self.outputs[idx].pending_direct = true;
                        // Zero-copy presentation: the client's own buffer is on the
                        // plane, so flag ZeroCopy. Fired on this flip's vblank. A
                        // feedback still parked from a frame that never reached
                        // its flip (mailbox replacement) must be DISCARDED, not
                        // dropped — per wp_presentation every feedback resolves
                        // exactly once, and a present-timing consumer (Vulkan
                        // present timing rides these events) errors out waiting
                        // on one that never fires.
                        if let Some(out) = present_output {
                            let replaced = self.outputs[idx].pending_feedback.replace(
                                collect_presentation_feedback(
                                    out, placements, layers, popups, out_rect, true,
                                ),
                            );
                            if let Some(mut old) = replaced {
                                debug!(output = %output_name, "wp_presentation: feedback discarded (flip replaced, direct scanout)");
                                old.discarded();
                            }
                        }
                        // Serve any screencopy from the buffer we just put on
                        // the plane. This runs *after* the flip so a slow
                        // capture consumer can never delay the game's frame.
                        let results = match capture_src {
                            Some((dmabuf, place)) => {
                                self.capture_direct(idx, &dmabuf, place, captures)
                            }
                            None => Vec::new(),
                        };
                        // No transient state is active (eligibility required it),
                        // so the output parks until the client's next commit.
                        return Ok((results, false));
                    }
                    Ok(false) => {} // not scannable this frame → composite below
                    Err(err) => {
                        warn!(output = %output_name, error = %err, "direct scanout failed; compositing");
                    }
                }
            } else {
                debug!(output = %output_name, "direct scanout rejected; compositing (capture needs a composed framebuffer)");
            }
        }
        // Compositing from here on: the plane's contents are about to stop
        // being a client buffer, so damage can no longer be described
        // relative to one.
        self.outputs[idx].direct_damage_ref = None;

        // The solo window, when its visually-topmost mapped node covers the
        // output *provably opaquely*: with every output pixel guaranteed
        // overwritten, the wallpaper/base pass underneath is pure waste —
        // skip it (any colour mode; this also trims the composite for SDR
        // outputs when a game's buffer isn't plane-scannable that frame).
        let solo_opaque = solo.filter(|&i| {
            covering_top_node(&placements[i].surface).is_some_and(|(node, rect)| {
                rect.loc == Point::from((0, 0))
                    && rect.size.w == compositor_size.w
                    && rect.size.h == compositor_size.h
                    && surface_provably_opaque(&node)
            })
        });

        // ── Single-pass HDR fast path ─────────────────────────────────
        // A solo opaque window filling an HDR output doesn't need the
        // generic HDR pipeline (composite into the fp16 linear scene +
        // a second PQ-encode pass = two extra full-output passes per game
        // frame). Render the frame like an SDR output instead — straight
        // into the scanout dmabuf, no fp16 scene, no encode pass:
        //
        // - an SDR window draws with the fused SDR→PQ program as the frame
        //   default, which colour-matches the generic pipeline exactly
        //   (same decode, saturation and OETF maths, one fragment);
        // - a PQ-tagged window (an HDR game that direct scanout couldn't
        //   take this frame — e.g. its buffer was rejected by the plane)
        //   needs NO program at all: on a PQ output, decode→encode is the
        //   identity, so the default sampler copies its pixels straight
        //   to the 10-bit scanout.
        let solo_hdr_surface =
            solo_opaque.is_some_and(|i| enc.pq.contains(&placements[i].surface.id()));
        // A solo scRGB game (id Tech / DOOM): same single-pass shape as an SDR
        // solo window — one full-output draw straight to the 10-bit scanout —
        // but through the fused scRGB→PQ program instead of the SDR one. It
        // can never be `solo_hdr_surface` (a surface carries one encoding), so
        // the passthrough branch below stays PQ-only.
        let solo_scrgb_surface =
            solo_opaque.is_some_and(|i| enc.scrgb.contains(&placements[i].surface.id()));
        // ...but not on a frame that has to be captured. The fast path works by
        // clearing `hdr`, which makes everything downstream treat this as an
        // SDR output -- including the capture dispatch, which then reads the
        // scanout directly. On this path the scanout is PQ-encoded BT.2020, so
        // a capture came out as raw PQ code values written into an sRGB image:
        // lifted blacks, flat mid-tones, highlights that never reach white.
        // (Inverting a real capture through the PQ curve reproduced the game's
        // luminance histogram exactly -- 106 cd/m² median, 1930 cd/m² peak.)
        //
        // The generic path composites into the fp16 linear scene, which is what
        // `capture_tonemapped` needs, so give up the fast path for those frames.
        // A screenshot costs one slower frame; continuous screencopy of a solo
        // HDR game gives it up for the duration, which is the same pipeline
        // every non-solo frame already uses.
        //
        // Direct scanout already refuses captures on an HDR output for exactly
        // this reason (see `capture_ok` above) -- this path simply never
        // inherited the rule.
        let single_pass_hdr = hdr && solo_opaque.is_some() && captures.is_empty();
        if single_pass_hdr {
            debug!(
                output = %output_name,
                passthrough = solo_hdr_surface,
                scrgb = solo_scrgb_surface,
                "single-pass HDR: solo window straight to scanout"
            );
            hdr = false;
        }

        // Everything below is the composited path — profile it (see
        // [`RenderProfile`]; direct-scanout frames returned above).
        let t_frame = Instant::now();
        let mut t_import = Duration::ZERO;
        let mut t_wintex = Duration::ZERO;
        let mut t_blur = Duration::ZERO;

        // Per-placement visibility on THIS output. Two prunes, applied by
        // every stage below (element import, decoration offscreens, draw
        // loops): windows fully occluded by a provably-opaque solo
        // fullscreen window — the Steam client behind the game paid an
        // offscreen + full draw per game frame — and windows that don't
        // even touch this output (each output renders its own frame;
        // drawing another output's windows at out-of-view coordinates was
        // pure waste). `LIBRELAND_NO_OCCLUSION=1` disables the prune for
        // A/B measurements against the render-profile log.
        let visible: Vec<bool> = placements
            .iter()
            .zip(win_draws.iter())
            .enumerate()
            .map(|(i, (p, wd))| {
                if self.no_occlusion {
                    return true;
                }
                if solo_opaque.is_some_and(|j| i != j) {
                    return false;
                }
                p.cell_rect.overlaps(out_rect) || wd.effective.overlaps(out_rect)
            })
            .collect();

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
        let (mut dmabuf, buffer_age) = self.outputs[idx]
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
        let radius_comp = border.rounded_corners.max(0);
        // Rounded corners belong to Normal windows: a window filling the
        // work area or the output wants square corners, which is what
        // every desktop draws there.
        let radius_for = |p: &Placement| {
            if p.fill == FillMode::Normal {
                radius_comp
            } else {
                0
            }
        };
        // Whether this window is composited through an offscreen texture
        // + the rounded mask shader (so its corners are genuinely
        // transparent and its bar is clipped to them), rather than drawn
        // as a plain rectangle straight to the frame.
        //
        // Per placement, not per config: `p.deco` is already resolved
        // against the window's fill, so a maximized window keeps its
        // titlebar while floating (you need it to un-maximize) and a
        // fullscreen one carries nothing. Deciding this from
        // `p.fill == Normal` instead — as it did before titlebars — left
        // a maximized window configured a bar smaller than it was drawn,
        // with no bar drawn at all.
        let decorated_win =
            |p: &Placement| p.deco != crate::layout::Deco::none() || radius_for(p) > 0;
        #[allow(
            clippy::type_complexity,
            reason = "one frame's worth of per-window, rescale-wrapped surface elements"
        )]
        let t = Instant::now();
        let grouped: Vec<Vec<RescaleRenderElement<WaylandSurfaceRenderElement<GlesRenderer>>>> =
            placements
                .iter()
                .zip(win_draws.iter())
                .enumerate()
                .map(|(i, (p, wd))| {
                    // Occluded / off-output windows don't get their trees
                    // walked or their textures touched at all.
                    if !visible[i] {
                        return Vec::new();
                    }
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
                    let bw_p = p.deco.border;
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
                    let offscreen = decorated_win(p);
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
                    let fallback = p.deco.content_size(p.cell_rect.size);
                    let (content_w, content_h) = window_geometry_size(&p.surface)
                        .unwrap_or((fallback.w, fallback.h));
                    // Where the buffer actually lands inside the cell.
                    // `paint_origin` is the single definition of that, and
                    // the pointer hit-test reads the same one — see its
                    // docs for why x anchors at the cell edge while y
                    // clears the titlebar.
                    let paint = p.deco.paint_origin();
                    let (target_w, target_h, anchor_x, anchor_y) = if offscreen {
                        // 1:1 with the size the client was configured to,
                        // so the buffer is not scaled at all when settled
                        // — see `Deco::paint_origin` for why that matters
                        // to the pointer.
                        let content = p.deco.content_size(eff.size);
                        (
                            f64::from(content.w.max(1)),
                            f64::from(content.h.max(1)),
                            f64::from(paint.x),
                            f64::from(paint.y),
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
        // Open animations: a layer surface fades in while sliding a short
        // way from whichever screen edge it sits against. Advanced here,
        // where the elements are built, so the alpha and the offset are
        // applied by the same import that would have happened anyway.
        let layer_open_cfg = self.animations.layer_open;
        let layer_open_on = self.animations.enabled && layer_open_cfg.enabled;
        let layer_groups: Vec<(LayerBucket, Vec<WaylandSurfaceRenderElement<GlesRenderer>>)> =
            layers
                .iter()
                .map(|l| {
                    let id = l.surface.id();
                    if self.pending_layer_open.remove(&id) && layer_open_on {
                        debug!(
                            surface = ?id,
                            namespace = %l.namespace,
                            edge = ?LayerEdge::of(l.rect, out_rect),
                            ?l.rect,
                            "layer open animation started"
                        );
                        self.layer_anims.insert(
                            id.clone(),
                            Animation::start(
                                now,
                                layer_open_cfg.duration_secs(),
                                layer_open_cfg.curve,
                            ),
                        );
                    }
                    let (mut alpha, mut offset) = (1.0_f32, Point::<i32, Physical>::from((0, 0)));
                    if let Some(a) = self.layer_anims.get(&id).copied() {
                        let v = a.value(now);
                        alpha = a.alpha(now);
                        offset = LayerEdge::of(l.rect, out_rect).offset(v, l.rect.size);
                        if a.done(now) {
                            self.layer_anims.remove(&id);
                            alpha = 1.0;
                            offset = Point::from((0, 0));
                        }
                    }
                    let local_phys = Point::<i32, Physical>::from((
                        scale_i(l.rect.loc.x + offset.x - compositor_position.x, scale),
                        scale_i(l.rect.loc.y + offset.y - compositor_position.y, scale),
                    ));
                    let elements = render_elements_from_surface_tree(
                        &mut self.gles,
                        &l.surface,
                        local_phys,
                        scale,
                        alpha,
                        Kind::Unspecified,
                    );
                    (l.layer, elements)
                })
                .collect();
        // Drop entries whose surfaces have gone (destroyed without a close
        // animation, e.g. when the animation is disabled).
        let live: HashSet<ObjectId> = layers.iter().map(|l| l.surface.id()).collect();
        self.layer_anims.retain(|id, _| live.contains(id));
        self.pending_layer_open.retain(|id| live.contains(id));

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

        t_import += t.elapsed();

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
                let alpha = 1.0 - c.anim.alpha(now);
                let eff = scale_rect_about_center(c.rect, lerp(1.0, c.scale_to, v));
                // A minimize sinks as it shrinks; a close has `sink == 0`
                // and goes nowhere.
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "sink is a fraction of a window height — small, non-negative"
                )]
                let sunk = (v * f64::from(c.sink)) as i32;
                let dest = Rectangle::<i32, Physical>::new(
                    Point::from((
                        scale_i(eff.loc.x - compositor_position.x, scale),
                        scale_i(eff.loc.y + sunk - compositor_position.y, scale),
                    )),
                    Size::from((scale_i(eff.size.w, scale), scale_i(eff.size.h, scale))),
                );
                Some((c.texture.clone(), dest, alpha))
            })
            .collect();

        // Closing layer surfaces: the same idea, but sliding back toward the
        // edge they came from rather than shrinking about their centre — a
        // bar should look like it withdrew, not like it imploded.
        let closing_layer_draws: Vec<(GlesTexture, Rectangle<i32, Physical>, f32)> = self
            .closing_layers
            .iter()
            .filter_map(|c| {
                if !c.rect.overlaps(out_rect) {
                    return None;
                }
                let v = c.anim.value(now);
                // `offset` takes "how settled", so a close is the open run
                // backwards.
                let off = c.edge.offset(1.0 - v, c.rect.size);
                let dest = Rectangle::<i32, Physical>::new(
                    Point::from((
                        scale_i(c.rect.loc.x + off.x - compositor_position.x, scale),
                        scale_i(c.rect.loc.y + off.y - compositor_position.y, scale),
                    )),
                    Size::from((scale_i(c.rect.size.w, scale), scale_i(c.rect.size.h, scale))),
                );
                Some((c.texture.clone(), dest, 1.0 - c.anim.alpha(now)))
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
        let draw_base = |frame: &mut GlesFrame<'_, '_>,
                         linear: bool,
                         damage: &[Rectangle<i32, Physical>]|
         -> Result<()> {
            if let Some(wp) = &wallpaper_media {
                // Media wallpapers force full-frame damage upstream, so no
                // damage threading is needed here.
                draw_wallpaper_texture(frame, wp, mode_size, linear, hdr_reference_white, hdr_saturation)?;
            } else {
                draw_fill(frame, &wallpaper, mode_size, mode_size, damage, linear, hdr_reference_white, hdr_saturation)?;
            }
            for (bucket, elements) in &layer_groups {
                if matches!(bucket, LayerBucket::Background | LayerBucket::Bottom) {
                    draw_render_elements::<GlesRenderer, _, _>(frame, scale, elements, damage)
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
        let t = Instant::now();
        let mut win_tex: Vec<Option<GlesTexture>> = Vec::with_capacity(placements.len());
        for (i, ((p, elements), wd)) in placements
            .iter()
            .zip(grouped.iter())
            .zip(win_draws.iter())
            .enumerate()
        {
            if !decorated_win(p) || !visible[i] {
                win_tex.push(None);
                continue;
            }
            // A colour-managed window's offscreen is fp16 and holds *linear
            // BT.2020* (the surface is PQ- or scRGB-decoded into it here), so
            // its decoration can be composited in linear by
            // `ROUND_TEX_SHADER_LINEAR`. SDR windows keep the 8-bit sRGB
            // offscreen the SDR/HDR-decode composite expects.
            let win_is_hdr = hdr && enc.is_managed(&p.surface.id());
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
            // The titlebar's identity joins the surface fingerprint in
            // deciding whether the offscreen is current: the bar is drawn
            // *into* it, and a title or focus change moves no commit
            // counter at all.
            let bar_h = scale_i(p.deco.titlebar, scale);
            let bar_state = BarState {
                hovered: self
                    .hovered_button
                    .as_ref()
                    .filter(|(id, _)| *id == p.surface.id())
                    .map(|(_, kind)| *kind),
                maximized: p.fill == FillMode::Maximized,
            };
            let bar = placement_bar_key(p, wd, scale, self.titlebar.buttons.len(), bar_state);
            let (bar_title, bar_app_id) = if bar_h > 0 {
                (
                    window_title(&p.surface).unwrap_or_default(),
                    window_app_id(&p.surface),
                )
            } else {
                (String::new(), None)
            };
            // Cached offscreen still current (same content, cell and
            // format)? Reuse it — an idle window costs a fingerprint walk
            // instead of an allocation + full redraw. On mismatch, reuse
            // at least the allocation when the geometry still fits.
            let fingerprint = surface_tree_fingerprint(&p.surface);
            let cached = (!self.no_wintex_cache)
                .then(|| self.wintex_cache.remove(&p.surface.id()))
                .flatten();
            let mut reusable = None;
            if let Some(cached) = cached {
                if cached.fmt == fmt
                    && cached.size == size
                    && cached.fingerprint == fingerprint
                    && cached.bar == bar
                {
                    win_tex.push(Some(cached.tex.clone()));
                    self.wintex_cache.insert(p.surface.id(), cached);
                    continue;
                }
                // The tree committed, but it yields nothing to draw this
                // frame. Re-rendering now would clear the offscreen to
                // transparent and cache that blank, so the window vanishes
                // from the composite *and* from the blur scene until it
                // commits again -- which drops tier 2 to wallpaper alone and
                // flips every frosted panel flat for a frame.
                //
                // Keep the old texture, and deliberately keep its old
                // fingerprint too: the entry must still look stale so the
                // next frame that does have elements re-renders it.
                if elements.is_empty() && cached.fmt == fmt && cached.size == size && cached.bar == bar {
                    debug!(
                        surface = ?p.surface.id(),
                        "wintex: empty elements on a committed tree; keeping last offscreen"
                    );
                    win_tex.push(Some(cached.tex.clone()));
                    self.wintex_cache.insert(p.surface.id(), cached);
                    continue;
                }
                // Stale content: still reuse the allocation when it fits.
                reusable = (cached.fmt == fmt && cached.size == size).then_some(cached.tex);
            }
            let phys = Size::<i32, Physical>::from((size.w, size.h));
            let full = [Rectangle::<i32, Physical>::from_size(phys)];
            // Rasterized here, before the closure below borrows
            // `self.gles` for the whole offscreen render.
            let bar_tex = (bar_h > 0)
                .then(|| {
                    self.bar_texture(
                        bar,
                        size.w,
                        bar_h,
                        &bar_title,
                        wd.focus >= 0.5,
                        // The bar is rasterized in PHYSICAL pixels (the
                        // offscreen is), so the point size scales with
                        // the output or the text is tiny on HiDPI.
                        bar_font_px(self.titlebar.font_size, scale),
                        bar_state,
                        bar_app_id.as_deref(),
                    )
                })
                .flatten();
            let tex = (|| -> Option<GlesTexture> {
                let mut tex = match reusable {
                    Some(tex) => tex,
                    None => self
                        .gles
                        .create_buffer(fmt, size)
                        .inspect_err(
                            |err| warn!(error = %err, "rounded window: offscreen alloc failed"),
                        )
                        .ok()?,
                };
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
                // Colour-managed window: decode its surface to linear BT.2020
                // as it's drawn into the fp16 offscreen (the composite then
                // stays linear), picking the decode its encoding calls for.
                if win_is_hdr {
                    let decode = if enc.scrgb.contains(&p.surface.id()) {
                        scrgb_decode_shader.clone()
                    } else if window_buffer_rb_swapped(&p.surface) {
                        hdr_decode_swizzle_shader.clone()
                    } else {
                        hdr_decode_shader.clone()
                    };
                    frame.override_default_tex_program(decode, Vec::new());
                }
                draw_render_elements::<GlesRenderer, _, _>(&mut frame, scale, elements, &full)
                    .inspect_err(|err| warn!(error = %err, "rounded window: draw failed"))
                    .ok()?;
                // The titlebar goes on last, over the top strip the
                // surface was inset out of. Into the offscreen rather
                // than into the frame, so the composite's rounded-rect
                // mask clips the bar's top corners for free — drawn
                // after the mask it would square them off again.
                if let Some(tex) = &bar_tex {
                    let dst = Rectangle::<i32, Physical>::new(
                        Point::from((0, 0)),
                        Size::from((size.w, bar_h)),
                    );
                    let src = Rectangle::<f64, smithay::utils::Buffer>::from_size(
                        Size::from((f64::from(size.w), f64::from(bar_h))),
                    );
                    if let Err(err) = frame.render_texture_from_to(
                        tex,
                        src,
                        dst,
                        &full,
                        &[dst],
                        Transform::Normal,
                        1.0,
                        None,
                        &[],
                    ) {
                        warn!(error = %err, "titlebar: blit into the window offscreen failed");
                    }
                }
                // Same-context sequential GL: the composite that samples this
                // texture is ordered after these writes, so the fence is dropped.
                let _ = frame
                    .finish()
                    .inspect_err(|err| warn!(error = %err, "rounded window: finish failed"))
                    .ok()?;
                drop(target);
                Some(tex)
            })();
            if let Some(tex) = &tex
                && !self.no_wintex_cache
            {
                self.wintex_cache.insert(
                    p.surface.id(),
                    WinTexCache {
                        tex: tex.clone(),
                        size,
                        fmt,
                        fingerprint,
                        bar,
                    },
                );
            }
            win_tex.push(tex);
        }
        t_wintex += t.elapsed();

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
                           linear: bool,
                           damage: &[Rectangle<i32, Physical>]|
         -> Result<()> {
            // A colour-managed (PQ or scRGB) surface in the linear scene is
            // drawn straight (skipping decoration — option A) with its own
            // decode, so its content isn't mis-decoded as SDR.
            let surface_is_hdr = linear && enc.is_managed(&p.surface.id());
            // Whether THIS window's offscreen is the fp16 *linear* one built in
            // Phase A — format-based, so independent of `linear` (which is
            // false during the sRGB blur replay).
            let win_is_hdr = hdr && enc.is_managed(&p.surface.id());
            if decorated_win(p) {
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
                // Mix the two fills by how focused the window currently
                // looks. `wd.focus` is 0 or 1 outside a crossfade, so this
                // reduces to picking one of them.
                let (mut border_top, mut border_bottom) =
                    mix_fills(&border.inactive, &border.active, wd.focus);
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
                let radius = scale_i(radius_for(p), scale).min(max_half).max(0);
                let bw = scale_i(p.deco.border, scale).min((max_half - 1).max(0)).max(0);
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
                let rel = damage_rel(damage, dst);
                if rel.is_empty() {
                    return Ok(());
                }
                frame
                    .render_texture_from_to(
                        tex,
                        src,
                        dst,
                        &rel,
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
                // Plain rectangle (fullscreen / maximized / undecorated, or a
                // colour-managed surface): draw the surface straight to the
                // frame. For colour-managed content swap the frame's decode
                // override to that encoding's decode for this draw, then
                // restore the scene's SDR default.
                if surface_is_hdr {
                    let decode = if enc.scrgb.contains(&p.surface.id()) {
                        scrgb_decode_shader.clone()
                    } else if window_buffer_rb_swapped(&p.surface) {
                        hdr_decode_swizzle_shader.clone()
                    } else {
                        hdr_decode_shader.clone()
                    };
                    frame.override_default_tex_program(decode, Vec::new());
                }
                let res =
                    draw_render_elements::<GlesRenderer, _, _>(frame, scale, elements, damage);
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
        // Buried windows ride along in the first stage — they are below
        // everything, and giving an unused window its own tier would cost
        // a full-resolution texture for a backdrop nobody looks at.
        let draw_band = |frame: &mut GlesFrame<'_, '_>, want: &[ZBand]| -> Result<()> {
            for band in want {
                for (((p, elements), wd), tex) in placements
                    .iter()
                    .zip(grouped.iter())
                    .zip(win_draws.iter())
                    .zip(win_tex.iter())
                    .filter(|(((p, _), _), _)| p.band == *band)
                {
                    draw_window(frame, p, elements, wd, tex.as_ref(), false, &full_damage)?;
                }
            }
            Ok(())
        };
        let draw_tiled =
            |frame: &mut GlesFrame<'_, '_>| draw_band(frame, &[ZBand::Buried, ZBand::Tiled]);
        // One floating window, by placement index — the unit the per-window
        // backdrop tiers are built around (see the staging loop below).
        let draw_float_one = |frame: &mut GlesFrame<'_, '_>, i: usize| -> Result<()> {
            draw_window(
                frame,
                &placements[i],
                &grouped[i],
                &win_draws[i],
                win_tex[i].as_ref(),
                false,
                &full_damage,
            )
        };
        // Split out from the floating band so a *filled* window can blur
        // against everything below it without blurring against itself —
        // it is drawn into the stage that produces the tier the layers
        // above it sample, one band later.
        let draw_filled = |frame: &mut GlesFrame<'_, '_>| {
            draw_band(frame, &[ZBand::Maximized, ZBand::Fullscreen])
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
        // Per-surface opt-in via ext-background-effect-v1: a window or
        // layer that committed a blur region gets backdrop blur even when
        // the config didn't opt it in. Collected once per frame; the
        // config's blur.enabled/passes still gate the pyramid itself.
        let protocol_blur: HashSet<ObjectId> = placements
            .iter()
            .map(|p| &p.surface)
            .chain(layers.iter().map(|l| &l.surface))
            .filter(|s| surface_requests_blur(s))
            .map(Resource::id)
            .collect();
        let layer_blurs = |l: &LayerPlacement| {
            layer_should_blur(&blur, &l.namespace) || protocol_blur.contains(&l.surface.id())
        };
        // Temporal blur masking (see MASK_BLUR_SHADER + `prev_layer_masks`):
        // for each blur-eligible layer, fetch last frame's surface-alpha
        // texture so the blur can mask by the *min* of this and last frame's
        // coverage — a client's transient full-surface frame then frosts
        // nothing new. A surface with no stored previous mask (its first
        // frame) has its blur skipped this frame. Computed here, a clean
        // borrow point before the frame binds `self.gles`; the store is then
        // replaced with this frame's masks for next time.
        let prev_masks_now: Vec<Option<GlesTexture>> = layers
            .iter()
            .zip(layer_masks.iter())
            .map(|(l, cur)| {
                if cur.is_some() && layer_blurs(l) {
                    self.outputs[idx].prev_layer_masks.get(&l.surface.id()).cloned()
                } else {
                    None
                }
            })
            .collect();
        self.outputs[idx].prev_layer_masks = layers
            .iter()
            .zip(layer_masks.iter())
            .filter(|(l, _)| layer_blurs(l))
            .filter_map(|(l, cur)| cur.clone().map(|t| (l.surface.id(), t)))
            .collect();
        // A provably-opaque solo fullscreen window covers everything a blur
        // tier could ever show through — a blur-opted bar occluded behind a
        // game kept a full 6-pass pyramid running on every single-pass game
        // frame. No visible translucency ⇒ no pyramid.
        let passes_ok = blur.enabled && blur.passes > 0 && solo_opaque.is_none();
        // Whether a window in a given band wants a blurred backdrop.
        // Deliberately not restricted to `fill == Normal` any more: a
        // *translucent* maximized or fullscreen window needs its
        // backdrop blurred exactly as much as a floating one does, and
        // assuming filled windows are opaque is what made a maximized
        // kitty show the sharp wallpaper straight through itself.
        let wants_blur = |i: usize, p: &Placement| {
            visible[i] && (blur.windows || protocol_blur.contains(&p.surface.id()))
        };
        let need_window = passes_ok
            && placements
                .iter()
                .enumerate()
                .any(|(i, p)| p.fill == FillMode::Normal && wants_blur(i, p));
        // A filled window blurs against base + tiled + floating, which is
        // one band deeper than the layer tier.
        let need_filled = passes_ok
            && placements
                .iter()
                .enumerate()
                .any(|(i, p)| p.fill != FillMode::Normal && wants_blur(i, p));
        // Layer blur is opt-in per namespace (config `blur.layers`), so a
        // fullscreen always-mapped overlay (e.g. a launcher) doesn't frost the
        // whole screen — only the layers the user named are blurred.
        let need_layer = passes_ok
            && layers.iter().any(|l| {
                matches!(l.layer, LayerBucket::Top | LayerBucket::Overlay)
                    && layer_blurs(l)
            });
        let t = Instant::now();
        // Saved per-tier blurred backdrops. Pull the scratch out of the map
        // so the blur helpers borrow only `self.gles`; on any GPU failure
        // we clear the tiers and fall back to sharp rendering. Programs are
        // `Arc`-cloned so the staging closure captures only `self.gles`.
        let mut tier_tiled: Option<GlesTexture> = None;
        let mut tier_float: Option<GlesTexture> = None;
        let mut tier_filled: Option<GlesTexture> = None;
        let mut tier_layer: Option<GlesTexture> = None;
        // Backdrops handed to individual floating windows, keyed by surface.
        // A window not in here falls back to `tier_float` — the whole band's
        // shared backdrop, which is right for the bottom-most one and for
        // any window past `MAX_WINDOW_TIERS`.
        let mut win_tiers: HashMap<ObjectId, GlesTexture> = HashMap::new();
        // The deepest backdrop built so far while walking the floating band.
        let mut newest_tier: Option<GlesTexture> = None;
        // Floating windows bottom-up: placement order *is* stack order, and
        // the staging loop has to walk it the same way the draw loop does or
        // a window would frost something drawn above it.
        let float_order: Vec<usize> = placements
            .iter()
            .enumerate()
            .filter(|(i, p)| visible[*i] && p.band == ZBand::Floating)
            .map(|(i, _)| i)
            .collect();
        if (need_window || need_filled || need_layer)
            && self.ensure_blur_scratch(idx, mode_size, blur.passes)
        {
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
                    draw_base(f, false, &full_damage)
                })?;
                if need_window {
                    run_pyramid(&mut self.gles, &mut scratch, passes, radius, &down, &up, 0)?;
                    tier_tiled = Some(scratch.tiers[0].clone());
                }
                // Each band feeds every tier above it, so a stage runs
                // whenever anything deeper still needs building.
                let below_float = need_window || need_filled || need_layer;
                if below_float {
                    render_scene_stage(&mut self.gles, &mut scratch, mode_size, &draw_tiled)?;
                }
                if need_window {
                    run_pyramid(&mut self.gles, &mut scratch, passes, radius, &down, &up, 1)?;
                    tier_float = Some(scratch.tiers[1].clone());
                }
                // The floating band, one window at a time and bottom-up, so
                // a translucent window frosts the translucent window beneath
                // it instead of the desktop they are both sitting on. A
                // single pyramid over the whole band — what this used to do
                // — can't: it has one answer for every window in it.
                //
                // A pyramid runs only when a blur-eligible window has
                // something new underneath it since the last one, so the
                // common cases (one translucent window, or several that
                // don't overlap anything translucent) cost exactly what they
                // did before.
                if need_filled || need_layer || need_window {
                    for (n, &i) in float_order.iter().enumerate() {
                        // The bottom-most window (`n == 0`) already has its
                        // backdrop: `tier_float` is the scene under the whole
                        // band. Every later one needs a fresh pyramid, since
                        // the window below it has just been drawn into the
                        // scene and has to show through.
                        if need_window && n > 0 && wants_blur(i, &placements[i]) {
                            let slot = BLUR_TIERS + win_tiers.len();
                            if win_tiers.len() >= MAX_WINDOW_TIERS {
                                debug!(
                                    cap = MAX_WINDOW_TIERS,
                                    "blur: per-window tier cap reached; deeper windows share the last backdrop"
                                );
                            } else if ensure_tier_slot(&mut self.gles, &mut scratch, slot) {
                                run_pyramid(
                                    &mut self.gles, &mut scratch, passes, radius, &down, &up, slot,
                                )?;
                                newest_tier = Some(scratch.tiers[slot].clone());
                            }
                            // `None` only before the first tier is built, i.e.
                            // for a window that falls back to `tier_float`.
                            if let Some(t) = &newest_tier {
                                win_tiers.insert(placements[i].surface.id(), t.clone());
                            }
                        }
                        render_scene_stage(&mut self.gles, &mut scratch, mode_size, &|f| {
                            draw_float_one(f, i)
                        })?;
                    }
                }
                if need_filled {
                    run_pyramid(&mut self.gles, &mut scratch, passes, radius, &down, &up, 2)?;
                    tier_filled = Some(scratch.tiers[2].clone());
                }
                if need_layer {
                    render_scene_stage(&mut self.gles, &mut scratch, mode_size, &draw_filled)?;
                    run_pyramid(&mut self.gles, &mut scratch, passes, radius, &down, &up, 3)?;
                    tier_layer = Some(scratch.tiers[3].clone());
                }
                Ok(())
            })();
            if let Err(err) = res {
                warn!(error = %err, output = %output_name, "backdrop blur failed; rendering sharp");
                tier_tiled = None;
                tier_float = None;
                tier_filled = None;
                tier_layer = None;
            }
            self.blur_scratch.insert(idx, scratch);
        }
        t_blur += t.elapsed();


        // ── Damage ────────────────────────────────────────────────────
        // Diff this frame's drawn set against the previous frame's (see
        // [`DamageTracker`]); `None` = paint in full. Falls back to full
        // whenever pixels can change outside the tracked inputs.
        let current_damage: Option<Vec<Rectangle<i32, Physical>>> = 'damage: {
            if self.no_damage
                // A capture wants every pixel, not the ones that changed
                // since a frame of some *other* workspace.
                || capture_only
                || self.outputs[idx].damage_tracker.force_full
                || self.cursor_needs_composite(idx, hide_cursor, compose_cursor)
                || !dnd_icon_elements.is_empty()
                || self.screenshot_overlay.is_some()
                || self.snap_preview.is_some()
                || freeze_texture.is_some()
                // A media wallpaper's texture is refreshed out-of-band
                // (refresh_wallpaper) — untracked, so full unless occluded.
                || (wallpaper_media.is_some() && solo_opaque.is_none())
                || placements.iter().any(|p| p.slide != Point::from((0, 0)))
                // Closing ghosts have no surface to diff against.
                || !self.closing_layers.is_empty()
            {
                break 'damage None;
            }
            let elem_scale = Scale::from(scale);
            let mut prev = std::mem::take(&mut self.outputs[idx].damage_tracker.prev);
            let mut new_map: HashMap<ObjectId, DrawnState> = HashMap::with_capacity(prev.len() + 4);
            let mut damage: Vec<Rectangle<i32, Physical>> = Vec::new();
            {
                let mut note = |id: ObjectId,
                                fingerprint: Vec<(ObjectId, CommitCounter)>,
                                rect: Rectangle<i32, Physical>,
                                focused: bool,
                                alpha: f32,
                                animating: bool,
                                blur: bool,
                                bar: u64| {
                    let alpha_bits = alpha.to_bits();
                    match prev.remove(&id) {
                        Some(old)
                            if !animating
                                && old.rect == rect
                                && old.focused == focused
                                && old.alpha_bits == alpha_bits
                                && old.blur == blur
                                && old.bar == bar
                                && old.fingerprint == fingerprint => {}
                        Some(old) => {
                            damage.push(old.rect);
                            damage.push(rect);
                        }
                        None => damage.push(rect),
                    }
                    new_map.insert(
                        id,
                        DrawnState {
                            fingerprint,
                            rect,
                            focused,
                            alpha_bits,
                            blur,
                            bar,
                        },
                    );
                };
                // Windows, at their drawn rects: the composite dst for
                // decorated Normal windows, the element bbox (which
                // includes CSD shadows) otherwise. Alpha + focus feed the
                // pixels (opacity, border colour); a running open/move
                // animation forces per-frame damage (alpha/rect sweep).
                for (i, (p, wd)) in placements.iter().zip(win_draws.iter()).enumerate() {
                    if !visible[i] {
                        continue;
                    }
                    let rect = if decorated_win(p) {
                        cell_local(wd.effective)
                    } else {
                        match elements_bbox(&grouped[i], elem_scale) {
                            Some(r) => r,
                            None => continue,
                        }
                    };
                    let animating = self
                        .win_anims
                        .get(&p.surface.id())
                        .is_some_and(WindowAnim::is_animating);
                    note(
                        p.surface.id(),
                        surface_tree_fingerprint(&p.surface),
                        rect,
                        p.focused,
                        wd.alpha,
                        animating,
                        protocol_blur.contains(&p.surface.id()),
                        placement_bar_key(
                            p,
                            wd,
                            scale,
                            self.titlebar.buttons.len(),
                            BarState {
                                hovered: self
                                    .hovered_button
                                    .as_ref()
                                    .filter(|(id, _)| *id == p.surface.id())
                                    .map(|(_, kind)| *kind),
                                maximized: p.fill == FillMode::Maximized,
                            },
                        ),
                    );
                }
                // Layer surfaces, only the ones this frame actually draws:
                // Overlay always; the rest only when the base bands run.
                for (l, (bucket, elements)) in layers.iter().zip(layer_groups.iter()) {
                    let drawn =
                        matches!(bucket, LayerBucket::Overlay) || solo_opaque.is_none();
                    if !drawn {
                        continue;
                    }
                    let animating = self.layer_anims.contains_key(&l.surface.id());
                    let Some(rect) = elements_bbox(elements, elem_scale) else {
                        continue;
                    };
                    note(
                        l.surface.id(),
                        surface_tree_fingerprint(&l.surface),
                        rect,
                        false,
                        1.0,
                        animating,
                        layer_blurs(l),
                        0,
                    );
                }
                // Popups (always drawn).
                for (pp, elements) in popups.iter().zip(popup_groups.iter()) {
                    let Some(rect) = elements_bbox(elements, elem_scale) else {
                        continue;
                    };
                    note(
                        pp.surface.id(),
                        surface_tree_fingerprint(&pp.surface),
                        rect,
                        false,
                        1.0,
                        false,
                        false,
                        0,
                    );
                }
            }
            // Whatever was drawn last frame and isn't this frame exposes
            // what's beneath it.
            damage.extend(prev.into_values().map(|s| s.rect));
            // Close-animation snapshots fade per frame: damage their rects
            // every frame any exist, plus the frame after the last one.
            let closing_rects: Vec<Rectangle<i32, Physical>> =
                closing_draws.iter().map(|(_, dest, _)| *dest).collect();
            damage.extend_from_slice(&self.outputs[idx].damage_tracker.prev_closing);
            damage.extend_from_slice(&closing_rects);
            self.outputs[idx].damage_tracker.prev_closing = closing_rects;
            // Frosted backdrops sample a neighbourhood of the scene: any
            // change may alter every frosted rect (the blur spread at 4K
            // rivals the output), so conservatively re-damage them all.
            if !damage.is_empty()
                && (tier_tiled.is_some() || tier_float.is_some() || tier_layer.is_some())
            {
                if tier_tiled.is_some() || tier_float.is_some() {
                    for (i, (p, wd)) in placements.iter().zip(win_draws.iter()).enumerate() {
                        if visible[i] && p.fill == FillMode::Normal {
                            damage.push(cell_local(wd.effective));
                        }
                    }
                }
                if tier_layer.is_some() {
                    for (l, (bucket, elements)) in layers.iter().zip(layer_groups.iter()) {
                        if matches!(bucket, LayerBucket::Top | LayerBucket::Overlay)
                            && layer_blurs(l)
                            && let Some(rect) = elements_bbox(elements, elem_scale)
                        {
                            damage.push(rect);
                        }
                    }
                }
            }
            self.outputs[idx].damage_tracker.prev = new_map;
            Some(coalesce_damage(damage))
        };
        // Repair damage for each render target from its own staleness:
        // the swapchain buffer by its age, the persistent fp16 scene by
        // frames-since-last-scene-draw. History is per-frame damage.
        let (scene_damage_vec, swap_damage_vec);
        {
            let hdr_scene_pass = hdr;
            let tracker = &mut self.outputs[idx].damage_tracker;
            tracker.force_full = false;
            swap_damage_vec = current_damage
                .as_ref()
                .and_then(|c| tracker.accumulated(usize::from(buffer_age), c));
            scene_damage_vec = if hdr_scene_pass {
                current_damage
                    .as_ref()
                    .and_then(|c| tracker.accumulated(tracker.scene_age, c))
            } else {
                None
            };
            tracker.push(current_damage, full_damage[0]);
            tracker.scene_age = if hdr_scene_pass {
                1
            } else {
                tracker.scene_age.saturating_add(1)
            };
        }
        // The damage set for the target the scene block draws into.
        let draw_damage: &[Rectangle<i32, Physical>] = if hdr {
            scene_damage_vec.as_deref().unwrap_or(&full_damage)
        } else {
            swap_damage_vec.as_deref().unwrap_or(&full_damage)
        };
        // Paint a full-res tier's sub-rect behind a translucent surface. The
        // tier texture is 1:1 with the framebuffer, so the source sub-rect
        // matches the on-screen destination rect. With a `mask` texture
        // (layer-shell panels) the blur is alpha-masked by the surface's own
        // buffer, so the frost follows whatever shape the client drew — the
        // compositor can't know a panel's corner radius. Without one
        // (windows) an SDF clips the tier to the same rounded rect
        // `draw_window` composites.
        // `mask`, when present, is `(current, previous)` surface-alpha
        // textures — the blur is masked by their temporal minimum (see
        // MASK_BLUR_SHADER). `None` is the window path (SDF-clipped).
        let blur_rect = |frame: &mut GlesFrame<'_, '_>,
                         tier: &GlesTexture,
                         dst: Rectangle<i32, Physical>,
                         mask: Option<(&GlesTexture, &GlesTexture)>,
                         damage: &[Rectangle<i32, Physical>]|
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
            let local = damage_rel(damage, dst);
            if local.is_empty() {
                return Ok(());
            }
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

            if let Some((mask, mask_prev)) = mask {
                // abs px → mask UV: shift by the rect origin, normalise by
                // the rect size, and flip y again if the client's buffer is
                // y-inverted. Current and previous masks are the same
                // surface's buffers (same size, position, inversion), so one
                // mapping serves both.
                let mut mask_mul = (tier_w / dst_w, ay / dst_h);
                let mut mask_add = (-loc_x / dst_w, (by - loc_y) / dst_h);
                if mask.is_y_inverted() {
                    mask_mul.1 = -mask_mul.1;
                    mask_add.1 = 1.0 - mask_add.1;
                }
                // `mask_add`/`mask_mul` normalise the rect to 0..1 of mask UV,
                // so one physical pixel is 1/dst_w by 1/dst_h regardless of
                // tier size or y-inversion (the taps are symmetric, so the
                // sign does not matter).
                //
                // Cap the reach at a quarter of each axis. A bar is only tens
                // of pixels tall, so a flat 128 px would put every vertical
                // tap past the far edge — they all clamp to the same edge row
                // and the veto degenerates into "was this column ever
                // covered", which is no veto at all. Content cannot travel
                // far across an axis it barely spans, so capping costs
                // nothing on exactly the surfaces where it matters.
                let dilate_px = |extent: f32| MASK_DILATE_PX.min(extent * 0.25);
                let mask_dilate = (dilate_px(dst_w) / dst_w, dilate_px(dst_h) / dst_h);
                let mut uniforms = vec![
                    Uniform::new("mask", 1i32),
                    Uniform::new("mask_prev", 2i32),
                    Uniform::new("mask_mul", mask_mul),
                    Uniform::new("mask_add", mask_add),
                    Uniform::new("mask_dilate", mask_dilate),
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
                // The shader's `mask`/`mask_prev` samplers read texture units
                // 1 and 2; smithay's draw only drives unit 0, so the bindings
                // (made and restored by the vendored helper) survive the call.
                frame
                    .with_secondary_textures(mask, mask_prev, |frame| {
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

        let t = Instant::now();
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
            } else if single_pass_hdr && solo_scrgb_surface {
                // Solo scRGB game: fused scRGB→PQ. No reference_white /
                // saturation — scRGB anchors itself at 80 cd/m² and both knobs
                // are SDR-only.
                frame.override_default_tex_program(scrgb_to_pq_shader, Vec::new());
            } else if single_pass_hdr && !solo_hdr_surface {
                frame.override_default_tex_program(
                    sdr_to_pq_shader,
                    vec![
                        Uniform::new("reference_white", ref_white_f32),
                        Uniform::new("saturation", hdr_saturation),
                    ],
                );
            } else if single_pass_hdr
                && solo_hdr_surface
                && solo_opaque
                    .and_then(|i| placements.get(i))
                    .is_some_and(|p| window_buffer_rb_swapped(&p.surface))
            {
                // Solo PQ game whose buffer is XRGB-order (NVIDIA HDR10
                // allocates XR30): the pixels are right, the sampled channel
                // order isn't — identity copy with R↔B restored.
                frame.override_default_tex_program(pq_passthrough_swizzle_shader, Vec::new());
            }
            // (single_pass_hdr && solo_hdr_surface with an RGBA-order buffer:
            // no override — the solo window's pixels are already PQ/BT.2020,
            // exactly what the 10-bit scanout wants; the default sampler is
            // the identity.)

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
                draw_base(&mut frame, hdr, draw_damage)?;
            }
            // Buried windows, then tiled ones; both blur against the base
            // (tier 0). Occluded / off-output windows (`!visible[i]`) are
            // skipped in every band — their element vectors are empty
            // anyway, but the skip also avoids painting backdrop frost
            // behind an invisible window.
            for band in [ZBand::Buried, ZBand::Tiled] {
                for (_, (((p, elements), wd), tex)) in placements
                    .iter()
                    .zip(grouped.iter())
                    .zip(win_draws.iter())
                    .zip(win_tex.iter())
                    .enumerate()
                    .filter(|(i, (((p, _), _), _))| visible[*i] && p.band == band)
                {
                    if let Some(t) = &tier_tiled
                        && (blur.windows || protocol_blur.contains(&p.surface.id()))
                    {
                        blur_rect(&mut frame, t, cell_local(wd.effective), None, draw_damage)?;
                    }
                    draw_window(&mut frame, p, elements, wd, tex.as_ref(), hdr, draw_damage)?;
                }
            }
            // Floating windows draw above tiled and blur against base +
            // tiled (tier 1), so a float reveals the windows beneath it.
            for (_, (((p, elements), wd), tex)) in placements
                .iter()
                .zip(grouped.iter())
                .zip(win_draws.iter())
                .zip(win_tex.iter())
                .enumerate()
                .filter(|(i, (((p, _), _), _))| visible[*i] && p.band == ZBand::Floating)
            {
                // This window's own backdrop if it got one — which includes
                // the translucent windows below it — else the band's shared
                // one, which is the wallpaper and the tiles.
                if let Some(t) = win_tiers.get(&p.surface.id()).or(tier_float.as_ref())
                    && (blur.windows || protocol_blur.contains(&p.surface.id()))
                {
                    blur_rect(&mut frame, t, cell_local(wd.effective), None, draw_damage)?;
                }
                draw_window(&mut frame, p, elements, wd, tex.as_ref(), hdr, draw_damage)?;
            }
            // Maximized windows: above normal windows, below Top/Overlay
            // panels, and — while floating — still decorated. They blur
            // against base + tiled + floating (tier 2), because a
            // translucent one is just as see-through maximized as it is
            // at any other size. Assuming otherwise is what put the
            // sharp wallpaper straight through a maximized kitty.
            for (_, (((p, elements), wd), tex)) in placements
                .iter()
                .zip(grouped.iter())
                .zip(win_draws.iter())
                .zip(win_tex.iter())
                .enumerate()
                .filter(|(i, (((p, _), _), _))| visible[*i] && p.band == ZBand::Maximized)
            {
                if let Some(t) = &tier_filled
                    && (blur.windows || protocol_blur.contains(&p.surface.id()))
                {
                    blur_rect(&mut frame, t, cell_local(wd.effective), None, draw_damage)?;
                }
                draw_window(&mut frame, p, elements, wd, tex.as_ref(), hdr, draw_damage)?;
            }

            // Top layer surfaces go above windows but below a fullscreen
            // window (status bar above kitty, but a fullscreen game covers the
            // bar). Blur against the full backdrop (tier 2) so a translucent
            // panel reveals a frosted desktop. Skipped wholesale under a
            // provably-opaque solo fullscreen window — the game covers them.
            for (((l, (bucket, elements)), mask), mask_prev) in layers
                .iter()
                .zip(layer_groups.iter())
                .zip(layer_masks.iter())
                .zip(prev_masks_now.iter())
            {
                if solo_opaque.is_some() || !matches!(bucket, LayerBucket::Top) {
                    continue;
                }
                if let Some(t) = &tier_layer
                    && let Some(mask) = mask
                    && let Some(mask_prev) = mask_prev
                    && layer_blurs(l)
                {
                    // A layer's blur is masked by the temporal min of its
                    // current + previous surface alpha. Both `let Some`s gate
                    // it: no current buffer would take blur_rect's window
                    // fallback (whole-rect frost); no previous mask means the
                    // surface just appeared, so skip one frame rather than
                    // trust a lone frame's alpha (that is the flash guard).
                    let dst = Rectangle::<i32, Physical>::new(
                        Point::new(
                            scale_i(l.rect.loc.x - compositor_position.x, scale),
                            scale_i(l.rect.loc.y - compositor_position.y, scale),
                        ),
                        Size::new(scale_i(l.rect.size.w, scale), scale_i(l.rect.size.h, scale)),
                    );
                    blur_rect(&mut frame, t, dst, Some((mask, mask_prev)), draw_damage)?;
                }
                draw_render_elements::<GlesRenderer, _, _>(&mut frame, scale, elements, draw_damage)
                    .context("draw_render_elements (layer top) failed")?;
            }

            // Fullscreen windows: borderless, above tiled/maximized windows and
            // Top panels (a fullscreen game/video covers the bar), but BELOW
            // Overlay layers (launcher / toasts / OSDs stay visible) and below
            // popups and the cursor.
            for (fs_i, (p, elements)) in placements
                .iter()
                .zip(grouped.iter())
                .enumerate()
                .filter(|(i, (p, _))| visible[*i] && p.band == ZBand::Fullscreen)
            {
                // Same tier as a maximized window: everything below it.
                // A fullscreen *opaque* window never gets here with a
                // tier built — `solo_opaque` vetoes the pyramid outright
                // — so this only costs anything for a genuinely
                // translucent one, which is the case that needs it.
                if let Some(t) = &tier_filled
                    && (blur.windows || protocol_blur.contains(&p.surface.id()))
                {
                    let wd = &win_draws[fs_i];
                    blur_rect(&mut frame, t, cell_local(wd.effective), None, draw_damage)?;
                }
                // Colour-managed fullscreen surface (an HDR game): swap the
                // frame's decode override to its encoding's decode for this
                // draw, then restore the scene's SDR default.
                let surface_is_hdr = hdr && enc.is_managed(&p.surface.id());
                if surface_is_hdr {
                    let decode = if enc.scrgb.contains(&p.surface.id()) {
                        scrgb_decode_shader.clone()
                    } else if window_buffer_rb_swapped(&p.surface) {
                        hdr_decode_swizzle_shader.clone()
                    } else {
                        hdr_decode_shader.clone()
                    };
                    frame.override_default_tex_program(decode, Vec::new());
                }
                let res = draw_render_elements::<GlesRenderer, _, _>(
                    &mut frame,
                    scale,
                    elements,
                    draw_damage,
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
            for (((l, (bucket, elements)), mask), mask_prev) in layers
                .iter()
                .zip(layer_groups.iter())
                .zip(layer_masks.iter())
                .zip(prev_masks_now.iter())
            {
                if !matches!(bucket, LayerBucket::Overlay) {
                    continue;
                }
                if let Some(t) = &tier_layer
                    && let Some(mask) = mask
                    && let Some(mask_prev) = mask_prev
                    && layer_blurs(l)
                {
                    // Temporal-min masked blur; both `let Some`s gate it (no
                    // buffer → window-fallback whole-rect frost; no previous
                    // mask → skip one frame). See the Top-layer loop above.
                    let dst = Rectangle::<i32, Physical>::new(
                        Point::new(
                            scale_i(l.rect.loc.x - compositor_position.x, scale),
                            scale_i(l.rect.loc.y - compositor_position.y, scale),
                        ),
                        Size::new(scale_i(l.rect.size.w, scale), scale_i(l.rect.size.h, scale)),
                    );
                    blur_rect(&mut frame, t, dst, Some((mask, mask_prev)), draw_damage)?;
                }
                draw_render_elements::<GlesRenderer, _, _>(&mut frame, scale, elements, draw_damage)
                    .context("draw_render_elements (layer overlay) failed")?;
            }

            // Closing layer surfaces, in the Overlay band they were drawn
            // in while alive — above windows, below popups and the cursor.
            for (texture, dest, alpha) in &closing_layer_draws {
                let rel = damage_rel(draw_damage, *dest);
                if rel.is_empty() {
                    continue;
                }
                frame
                    .render_texture_from_to(
                        texture,
                        Rectangle::from_size(texture.size()).to_f64(),
                        *dest,
                        &rel,
                        &[],
                        Transform::Normal,
                        *alpha,
                        None,
                        &[],
                    )
                    .context("render_texture_from_to (closing layer) failed")?;
            }

            // Closing windows: the fade/shrink-out snapshot, above the
            // windows reflowing to fill the freed space, below popups.
            for (texture, dest, alpha) in &closing_draws {
                let rel = damage_rel(draw_damage, *dest);
                if rel.is_empty() {
                    continue;
                }
                frame
                    .render_texture_from_to(
                        texture,
                        Rectangle::from_size(texture.size()).to_f64(),
                        *dest,
                        &rel,
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
                    draw_damage,
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
            // Quick-tile preview: where the window being dragged would
            // land. Drawn over the scene (including the dragged window
            // itself) rather than under it — a wash this light reads
            // fine through, and drawing it below would hide it behind
            // every window it overlaps, which is most of them.
            if let Some(rect) = snap_preview {
                draw_snap_preview(
                    &mut frame,
                    rect,
                    accent,
                    compositor_position,
                    mode_size,
                    scale,
                    hdr,
                    hdr_reference_white,
                    hdr_saturation,
                )?;
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
                // The options bar draws over the dim wash, as its own
                // pass — it is chrome, not part of the selection.
                if !overlay.strokes.is_empty() {
                    draw_strokes(
                        &mut frame,
                        &overlay.strokes,
                        &OverlayPaint {
                            segment: &segment_shader,
                            blank: &blank_tex,
                            origin: compositor_position,
                            scale,
                            hdr,
                            reference_white: hdr_reference_white,
                            saturation: hdr_saturation,
                        },
                    )?;
                }
                if let Some(bar) = &overlay.toolbar {
                    draw_toolbar(
                        &mut frame,
                        bar,
                        &OverlayPaint {
                            segment: &segment_shader,
                            blank: &blank_tex,
                            origin: compositor_position,
                            scale,
                            hdr,
                            reference_white: hdr_reference_white,
                            saturation: hdr_saturation,
                        },
                    )?;
                }
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
        let t_scene = t.elapsed();

        // HDR screenshots: tonemap the linear scene to 8-bit sRGB and read
        // that, so a capture of an HDR output "looks like SDR".
        if hdr && !captures.is_empty() {
            capture_results =
                self.capture_tonemapped(&output_name, mode_size, ref_white_f32, captures);
        }

        // HDR: encode the composited linear-BT.2020 scene (the fp16 offscreen)
        // to PQ / BT.2020 into the 10-bit scanout dmabuf. SDR keeps the
        // scene's own sync — it composited straight to the dmabuf.
        let t = Instant::now();
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
                        // The swapchain target is repaired by its own age
                        // (dst sits at the origin, so relative == absolute).
                        swap_damage_vec.as_deref().unwrap_or(&[dst]),
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

        let t_encode = t.elapsed();
        self.outputs[idx].profile.record(
            &output_name,
            t_frame.elapsed(),
            t_import,
            t_wintex,
            t_blur,
            t_scene,
            t_encode,
        );

        // A capture-only frame stops here: the pixels have been read back
        // and nothing else about it is real. No flip, so no adaptive-sync
        // settle, no `wl_callback.done` (the clients in it were never
        // displayed, and telling them otherwise would have them render
        // frames for a screen that isn't showing them), and no presentation
        // feedback.
        //
        // The swapchain slot it drew into is *not* queued, so `next_buffer`
        // hands the same slot back next frame — with a `buffer_age` that
        // now lies, since the slot holds a frame that was never shown.
        // Forcing the next frame full is what squares that: age only ever
        // feeds the damage diff, and skipping the diff makes the age moot.
        if capture_only {
            self.outputs[idx].damage_tracker.force_full = true;
            return Ok((capture_results, false));
        }

        // Settle this output's adaptive-sync state for the frame we're
        // about to queue. Must run before `queue_buffer` so the commit it
        // triggers carries the right VRR_ENABLED (smithay promotes the
        // commit to a modeset itself when the toggle demands one).
        self.apply_vrr(idx, placements);

        // Hand KMS the real damage as FB_DAMAGE_CLIPS (None = full frame;
        // also None when nothing changed — an empty clip array is invalid).
        let clips = swap_damage_vec.filter(|v| !v.is_empty());
        self.outputs[idx]
            .surface
            .queue_buffer(Some(sync), clips)
            .with_context(|| format!("queue_buffer failed for {output_name}"))?;
        debug!(output = %output_name, "frame queued for scanout");

        // Queue wl_callback.done for every surface in this frame; fired at
        // vblank by `frame_submitted` (shared with the direct-scanout path).
        self.queue_output_frame_callbacks(idx, placements, layers, popups, out_rect);
        self.outputs[idx].pending_direct = false;

        // Collect wp_presentation feedback for the surfaces in this composited
        // frame; fired with the real vblank timestamp in `frame_submitted`.
        // Not zero-copy (the scene went through the GLES compositor). A
        // feedback still parked from a frame that never reached its flip is
        // discarded, never dropped — see the direct-scanout twin above.
        if let Some(out) = present_output {
            let replaced = self.outputs[idx]
                .pending_feedback
                .replace(collect_presentation_feedback(
                    out, placements, layers, popups, out_rect, false,
                ));
            if let Some(mut old) = replaced {
                debug!(output = %output_name, "wp_presentation: feedback discarded (flip replaced, composite)");
                old.discarded();
            }
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
        let anim_running = self.win_anims.values().any(WindowAnim::is_animating);
        let followup = anim_running
            || !self.closing.is_empty()
            || !self.pending_open.is_empty()
            || !self.layer_anims.is_empty()
            || !self.closing_layers.is_empty()
            || !self.pending_layer_open.is_empty()
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
    fn queue_output_frame_callbacks(
        &mut self,
        idx: usize,
        placements: &[Placement],
        layers: &[LayerPlacement],
        popups: &[PopupPlacement],
        out_rect: Rectangle<i32, Physical>,
    ) {
        let roots = &mut self.outputs[idx].pending_frame_roots;
        roots.clear();
        for p in placements {
            if p.cell_rect.overlaps(out_rect) {
                roots.push(p.surface.clone());
            }
        }
        for l in layers {
            if l.rect.overlaps(out_rect) {
                roots.push(l.surface.clone());
            }
        }
        // Popups are tiny, transient, and tied to a parent already covered
        // above; fire unconditionally rather than track their output.
        for p in popups {
            roots.push(p.surface.clone());
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
    /// placement's index when: the cursor needs no compositing (hidden /
    /// off-output / on the HW plane), no popup or
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
        compose_cursor: bool,
    ) -> Option<SoloScene> {
        let output = &self.outputs[idx];
        let out_rect = Rectangle::new(output.compositor_position, output.compositor_size);

        // Every rejection below names itself in the debug log, but only
        // while a fullscreen window is actually covering this output —
        // that's the situation where a vetoed fast path costs real frames
        // and the reason is worth reading back from a session log; on a
        // plain desktop the veto is the permanent normal case.
        let fullscreen_covers = placements
            .iter()
            .any(|p| p.band == ZBand::Fullscreen && p.cell_rect.overlaps(out_rect));
        macro_rules! veto {
            ($reason:literal) => {{
                if fullscreen_covers {
                    debug!(output = %output.name, reason = $reason, "solo-fullscreen fast paths vetoed");
                }
                return None;
            }};
        }

        // The cursor must not need drawing into this output's frame (hidden
        // and off-output pointers need nothing; a plane-resident one scans
        // out alongside). Note a *capture* is not disqualifying: a
        // cursorless one reads back from whatever we present, and a
        // cursor-inclusive one already forces `compose_cursor` here.
        if self.cursor_needs_composite(idx, hide_cursor, compose_cursor) {
            veto!("software cursor");
        }
        // Client content that draws ABOVE a fullscreen window: *Overlay*
        // layer surfaces (which include the session-lock surface, injected
        // as Overlay by render_crtc), then popups on top of those — the same
        // order the composite path draws them in, which is the order the
        // overlay planes have to be handed out in.
        //
        // Top/Bottom/Background layers draw BELOW fullscreen windows — a
        // fullscreen game covers the bar — so a mapped panel is not "above"
        // and never appears here. And an Overlay only counts when it actually
        // has a committed buffer: shells pre-create buffer-less popup slots
        // that sit mapped over the whole session (quickshell parks ten such
        // `qs-popup` Overlay surfaces at startup), and a buffer-less surface
        // draws nothing.
        //
        // None of this vetoes on its own any more. A little content above the
        // window can ride overlay planes and leave the game on the primary
        // one; the callers that need a *strictly* solo scene (the HDR
        // single-pass path, the wallpaper skip) filter on `above.is_empty()`.
        let mut above: Vec<OverlayCandidate> = Vec::new();
        for l in layers {
            if matches!(l.layer, LayerBucket::Overlay)
                && l.rect.overlaps(out_rect)
                && with_renderer_surface_state(&l.surface, |state| state.buffer().is_some())
                    .unwrap_or(false)
            {
                // A layer surface's `rect` *is* its buffer rect — layer shell
                // has no window geometry to subtract.
                above.push(OverlayCandidate {
                    surface: l.surface.clone(),
                    buffer_origin: l.rect.loc,
                });
            }
        }
        for pp in popups {
            if pp.rect.overlaps(out_rect) {
                above.push(OverlayCandidate {
                    surface: pp.surface.clone(),
                    buffer_origin: pp.buffer_origin,
                });
            }
        }
        // Transient overlays that draw ABOVE a fullscreen window →
        // composite. Scoped to this output where possible: a closing
        // window's fade-out snapshot draws above fullscreen, so it only
        // matters when its rect overlaps this output; the screenshot
        // overlay / DnD icon / frozen backdrop always composite. (Pending
        // open marks and running move/open animations are checked against
        // the covering window itself below — on any *other* window they
        // animate beneath the fullscreen one, invisibly.)
        if self.closing.iter().any(|c| c.rect.overlaps(out_rect))
            || self
                .closing_layers
                .iter()
                .any(|c| c.rect.overlaps(out_rect))
        {
            veto!("closing animation overlaps");
        }
        // A layer sliding in is drawn at an offset and a partial alpha, so
        // it can't ride a plane the way a settled one can.
        if layers
            .iter()
            .any(|l| l.rect.overlaps(out_rect) && self.layer_anims.contains_key(&l.surface.id()))
        {
            veto!("layer surface animating");
        }
        if self.screenshot_overlay.is_some()
            || self.snap_preview.is_some()
            || self.dnd_icon.is_some()
            || self.freeze_textures.contains_key(&output.name)
        {
            veto!("screenshot/DnD/snap overlay active");
        }

        // Exactly one placement in the *fullscreen band* may cover the
        // output. Every other band draws BENEATH it (buried → tiled →
        // floating → maximized → Top layers → fullscreen), so tiled and
        // floating windows sharing the workspace — the Steam client behind
        // the game it launched — are invisible behind it and don't matter.
        //
        // Filtering on the band and not on `fill` is load-bearing: a
        // fullscreen window that isn't active is *buried*, with ordinary
        // windows drawing over it, and handing it the whole scanout plane
        // would erase them.
        let mut covering = placements
            .iter()
            .enumerate()
            .filter(|(_, p)| p.band == ZBand::Fullscreen && p.cell_rect.overlaps(out_rect));
        let (i, p) = covering.next()?;
        if covering.next().is_some() {
            veto!("two fullscreen windows cover the output");
        }

        // It must be settled: parked at 1:1 over the whole output, fully
        // visible, mid-workspace-slide excluded, and neither waiting on its
        // open animation nor animating its rect.
        if p.slide != Point::from((0, 0)) {
            veto!("workspace slide in progress");
        }
        let draw = win_draws.get(i)?;
        if draw.effective != out_rect || draw.alpha < 1.0 {
            veto!("window not settled at 1:1 opaque");
        }
        if self.pending_open.contains_key(&p.surface.id())
            || self
                .win_anims
                .get(&p.surface.id())
                .is_some_and(WindowAnim::is_animating)
        {
            veto!("a window animation is still running");
        }
        Some(SoloScene { solo: i, above })
    }

    /// Eligible only when the scene is a fullscreen window (see
    /// [`Self::solo_fullscreen_scene`], decided by the caller) whose colour
    /// mode matches the output (HDR output ⇔ PQ surface), backed by a single
    /// dmabuf buffer that is pixel-exact with the mode — plus, optionally, a
    /// little client content above it that fits on the overlay planes.
    fn direct_scanout_inputs(
        &self,
        idx: usize,
        scene: Option<&SoloScene>,
        placements: &[Placement],
        enc: &SurfaceEncodings,
    ) -> Option<DirectInputs> {
        let scene = scene?;
        let above = &scene.above;
        let output = &self.outputs[idx];
        let p = placements.get(scene.solo)?;

        // More things above the window than there are planes to put them on:
        // the composite path is the only way to draw them all.
        if above.len() > output.surface.overlay_capacity() {
            debug!(
                output = %output.name,
                above = above.len(),
                planes = output.surface.overlay_capacity(),
                "not enough overlay planes for the content above the window; compositing"
            );
            return None;
        }

        // Every rejection names itself at debug so a session log can say
        // exactly why a fullscreen game isn't on the plane (the caller
        // already established the solo-fullscreen scene, so this never
        // fires for plain desktop use).
        macro_rules! reject {
            ($reason:literal) => {{
                debug!(output = %output.name, reason = $reason, "direct scanout rejected; compositing");
                return None;
            }};
        }

        // The window's colour mode must match the output: an SDR surface on
        // an HDR output needs the compositor's PQ encode (see the single-pass
        // fast path), a PQ surface on an SDR output needs a tonemap.
        // scRGB is linear light: the display expects PQ and no plane can
        // convert, so a scRGB surface is never scanout-eligible — it has to go
        // through the fused scRGB→PQ program. (On an HDR output the PQ check
        // below already excludes it; this also covers an SDR output, where
        // both sides would otherwise be `false` and match.)
        if enc.scrgb.contains(&p.surface.id()) {
            return None;
        }
        if output.hdr != enc.pq.contains(&p.surface.id()) {
            reject!("surface/output colour mode mismatch (SDR on HDR takes the single-pass path)");
        }
        // The window's visually-topmost mapped node must itself cover the
        // whole output; that node's buffer is the scanout candidate. (A bare
        // toplevel is its own top node; Wine Wayland's game swapchain is a
        // subsurface stacked above the host toplevel's buffer — requiring a
        // single-node tree here kept every Wine Wayland game compositing.)
        // Node coordinates are window-buffer-local, and a fullscreen
        // window's buffer origin sits at the cell origin (= the output
        // origin), so covering the output means exactly (0,0)..out_size.
        let out_size = output.compositor_size;
        let Some((node, node_rect)) = covering_top_node(&p.surface) else {
            reject!("window has no mapped surface node");
        };
        if node_rect.loc != Point::from((0, 0))
            || node_rect.size.w != out_size.w
            || node_rect.size.h != out_size.h
        {
            reject!("topmost surface node doesn't cover the output");
        }

        // Extract a scanout-ready dmabuf + keep-alive from the node's
        // committed buffer. Rejects shm buffers, buffers whose pixels don't
        // match the mode 1:1, and buffers we can't prove are opaque.
        //
        // Damage is only meaningful relative to the buffer currently ON the
        // plane, so it needs the previous *direct* frame's commit — cleared
        // whenever a composite frame intervenes.
        let damage_ref = output.direct_damage_ref.clone();
        let (primary, commit) =
            self.scanout_layer_for(idx, &node, LayerTarget::Primary, damage_ref.as_ref())?;

        // Content above the window, each onto its own overlay plane. All or
        // nothing: one candidate the hardware can't take means the frame is
        // composited whole, because a game with its notification silently
        // missing is worse than a composited game.
        let mut overlays = Vec::with_capacity(above.len());
        for cand in above {
            // One buffer per plane: a candidate with subsurfaces is a *tree*,
            // and flattening it is exactly the compositing we're avoiding.
            let Some(node) = single_node_surface(&cand.surface) else {
                debug!(output = %output.name, "overlay candidate has subsurfaces; compositing");
                return None;
            };
            let origin = self.output_local_point(idx, cand.buffer_origin);
            let (layer, _) =
                self.scanout_layer_for(idx, &node, LayerTarget::Overlay(origin), None)?;
            overlays.push(layer);
        }

        Some(DirectInputs {
            primary,
            overlays,
            commit,
        })
    }

    /// Build the scanout layer for one surface node, or `None` when it can't
    /// go on a plane (which sends the whole frame to the composite path).
    ///
    /// `damage_ref` is the commit this plane currently shows, if it is this
    /// very surface; damage is only expressible relative to that.
    fn scanout_layer_for(
        &self,
        idx: usize,
        node: &WlSurface,
        target: LayerTarget,
        damage_ref: Option<&(ObjectId, CommitCounter)>,
    ) -> Option<(ScanoutLayer, (ObjectId, CommitCounter))> {
        let output = &self.outputs[idx];
        let name = &output.name;
        let mode_rect = Rectangle::from_size(output.mode_size);
        let out_scale = output.scale;
        let role = target.role();
        let node_id = node.id();
        let damage_ref = damage_ref.filter(|(id, _)| *id == node_id).cloned();
        with_renderer_surface_state(node, |state| {
            macro_rules! reject {
                ($reason:literal) => {{
                    debug!(output = %name, ?role, reason = $reason, "direct scanout rejected; compositing");
                    return None;
                }};
            }
            // The buffer's own transform rides the plane's `rotation`
            // property rather than disqualifying the frame; `test_state`
            // rejects it on hardware that can't rotate, which costs the one
            // probe and falls back to compositing.
            let transform = state.buffer_transform();
            // For the primary layer the destination was already checked
            // against the OUTPUT size, not the buffer: a fractional-aware
            // client (oversized buffer + viewport) and an Xwayland client
            // under the client scale (physical-sized buffer that smithay
            // shrinks logically) both have dst < buffer *by design* while
            // their pixels still match the mode exactly. `buffer_scale` needs
            // no check of its own — it's already folded into both src and dst.
            let buf = state.buffer_size()?;
            let view = state.view()?;

            let Some(dst) = layer_dst(target, view.dst, out_scale, mode_rect) else {
                reject!("layer has no on-screen size, or hangs off the output edge");
            };

            let buffer = state.buffer()?.clone();
            let Ok(dmabuf) = smithay::wayland::dmabuf::get_dmabuf(&buffer) else {
                reject!("buffer is not a dmabuf (shm)");
            };
            let dmabuf = dmabuf.clone();
            let size = dmabuf.size();

            // A crop is expressible as the plane's source rectangle, so a
            // viewport no longer forces compositing. `view.src` is in
            // surface-logical units; scale it into buffer pixels with the
            // logical→buffer ratio the surface itself defines.
            let (sx, sy) = (
                f64::from(size.w) / f64::from(buf.w.max(1)),
                f64::from(size.h) / f64::from(buf.h.max(1)),
            );
            let src = Rectangle::<f64, smithay::utils::Buffer>::new(
                (view.src.loc.x * sx, view.src.loc.y * sy).into(),
                (view.src.size.w * sx, view.src.size.h * sy).into(),
            );
            // It must lie inside the buffer.
            if src.loc.x < 0.0
                || src.loc.y < 0.0
                || src.loc.x + src.size.w > f64::from(size.w) + 0.5
                || src.loc.y + src.size.h > f64::from(size.h) + 0.5
            {
                reject!("source rectangle falls outside the buffer");
            }
            // Pixel-exactness gate for the primary layer: whatever
            // sub-rectangle we show must be the mode's worth of pixels.
            // Anything else would need the plane to scale, which primary
            // planes generally can't. An overlay may scale — plenty of
            // overlay planes can, and `test_state` refuses the ones that
            // can't, at the cost of a single probe.
            let src_px = src.size.to_i32_round::<i32>();
            if role == LayerRole::Primary && (src_px.w != dst.size.w || src_px.h != dst.size.h) {
                reject!("buffer pixels don't match the mode 1:1");
            }

            // Provable opacity, for the primary layer only. The composite
            // path blends an alpha buffer over what's beneath (wallpaper, or
            // lower tree nodes); on the primary plane there is nothing
            // beneath, and we may drop the alpha channel entirely (the opaque
            // sibling fourcc), so the two only agree when the surface really
            // is opaque. A no-alpha format is inherently opaque; an 8-bit
            // alpha format must declare a covering opaque region (a
            // translucent fullscreen terminal must keep compositing); a
            // [`vestigial_alpha`] HDR swapchain format is accepted as-is.
            //
            // An overlay is the opposite case: it is *supposed* to blend over
            // the window below, and the plane does that in hardware.
            let code = dmabuf.format().code;
            if role == LayerRole::Primary
                && has_alpha(code)
                && !vestigial_alpha(code)
                && !opaque_region_covers(state.opaque_regions(), buf)
            {
                reject!("alpha buffer without a covering opaque region");
            }

            // Damage, in buffer pixels, since the buffer currently on the
            // plane. Only usable when the last flip put this same surface
            // there — otherwise the screen holds something whose delta we
            // can't describe, and the full plane is the only honest answer.
            let commit = (node_id.clone(), state.current_commit());
            let damage = damage_ref
                .map(|(_, since)| {
                    state
                        .damage_since(Some(since))
                        .iter()
                        .map(|r| {
                            Rectangle::<i32, Physical>::new(
                                (r.loc.x, r.loc.y).into(),
                                (r.size.w, r.size.h).into(),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                // An empty clip array is invalid; "nothing changed" has to be
                // spelled as a full-plane flip.
                .filter(|d: &Vec<_>| !d.is_empty());

            Some((
                ScanoutLayer {
                    buffer,
                    dmabuf,
                    place: DirectPlacement {
                        src,
                        dst,
                        transform,
                    },
                    damage,
                },
                commit,
            ))
        })
        .flatten()
    }

    /// Convert an absolute compositor point into this output's physical
    /// (mode) pixels — what a plane's destination rectangle is measured in.
    fn output_local_point(&self, idx: usize, p: Point<i32, Physical>) -> Point<i32, Physical> {
        let output = &self.outputs[idx];
        let local = p - output.compositor_position;
        let s = output.scale;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "output-sized pixel coordinates, well inside i32 after scaling"
        )]
        Point::from((
            (f64::from(local.x) * s).round() as i32,
            (f64::from(local.y) * s).round() as i32,
        ))
    }
}

/// Where a [`Renderer::scanout_layer_for`] layer should land.
#[derive(Debug, Clone, Copy)]
enum LayerTarget {
    /// The primary plane: the whole mode, by construction.
    Primary,
    /// An overlay plane, with the buffer's `(0, 0)` at this output-local
    /// physical point and the size taken from the surface itself.
    Overlay(Point<i32, Physical>),
}

impl LayerTarget {
    fn role(self) -> LayerRole {
        match self {
            LayerTarget::Primary => LayerRole::Primary,
            LayerTarget::Overlay(_) => LayerRole::Overlay,
        }
    }
}

/// Where a layer lands on the CRTC, in physical pixels, or `None` when it
/// can't go on a plane at all.
///
/// The primary layer is the whole mode by construction. An overlay sits at
/// its buffer origin, sized by the surface's own destination size scaled into
/// physical pixels, and must fit entirely on the output: a plane's
/// destination is *clipped* to the CRTC, so content hanging off the edge
/// would be cut rather than positioned, which only the composite path gets
/// right.
fn layer_dst(
    target: LayerTarget,
    view_dst: Size<i32, Logical>,
    out_scale: f64,
    mode_rect: Rectangle<i32, Physical>,
) -> Option<Rectangle<i32, Physical>> {
    let LayerTarget::Overlay(origin) = target else {
        return Some(mode_rect);
    };
    #[allow(
        clippy::cast_possible_truncation,
        reason = "surface sizes are display pixels; the product stays small"
    )]
    let size = Size::<i32, Physical>::from((
        (f64::from(view_dst.w) * out_scale).round() as i32,
        (f64::from(view_dst.h) * out_scale).round() as i32,
    ));
    let dst = Rectangle::new(origin, size);
    (size.w > 0 && size.h > 0 && mode_rect.contains_rect(dst)).then_some(dst)
}

/// Which plane a layer is destined for. The two differ in what they must
/// prove: the primary layer has nothing beneath it and must be opaque and
/// unscaled, an overlay blends and may scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerRole {
    Primary,
    Overlay,
}

/// The surface itself, if its tree is exactly that one node — no subsurfaces.
/// A multi-node tree needs compositing to flatten, which is the thing an
/// overlay plane exists to avoid.
fn single_node_surface(surface: &WlSurface) -> Option<WlSurface> {
    let mut count = 0usize;
    with_surface_tree_downward(
        surface,
        (),
        |_, _, ()| TraversalAction::DoChildren(()),
        |_, _, ()| count += 1,
        |_, _, ()| true,
    );
    (count == 1).then(|| surface.clone())
}

/// The shape of a frame that *might* skip compositing: one fullscreen window
/// covering the output, plus whatever little draws above it.
struct SoloScene {
    /// Index into `placements` of the covering fullscreen window.
    solo: usize,
    /// Client surfaces drawn above it, bottom-up. Empty is the pure case —
    /// the game and nothing else. Non-empty means the frame can still avoid a
    /// composite *if* every one of these fits on an overlay plane.
    above: Vec<OverlayCandidate>,
}

/// A client surface drawn above the fullscreen window, and where.
struct OverlayCandidate {
    surface: WlSurface,
    /// Where the surface *buffer*'s `(0, 0)` lands, in absolute compositor
    /// coordinates.
    ///
    /// Deliberately not the visible rect. A popup's `rect` is its window
    /// geometry, which for a client-decorated menu is smaller than the buffer
    /// by however much shadow it draws around itself; feeding that to a plane
    /// as the destination would squash the whole buffer into the menu's
    /// frame. The composite path positions popups by `buffer_origin` for the
    /// same reason.
    buffer_origin: Point<i32, Physical>,
}

/// Buffers to latch directly onto the hardware planes (direct scanout).
/// Produced by [`Renderer::direct_scanout_inputs`] and consumed by
/// [`ScanoutSurface::try_queue_direct`].
struct DirectInputs {
    /// The fullscreen window, for the primary plane.
    primary: ScanoutLayer,
    /// Content for overlay planes above it, bottom-up.
    overlays: Vec<ScanoutLayer>,
    /// Identity of the primary layer's commit, so the *next* frame can ask
    /// for damage relative to it (see [`OutputRender::direct_damage_ref`]).
    commit: (ObjectId, CommitCounter),
}

/// Whether the surface's opaque regions cover the whole surface — i.e. it is
/// provably fully opaque. (smithay auto-fills a full opaque region for
/// no-alpha buffers; alpha buffers carry whatever region the client
/// declared.)
///
/// A single region covering the surface is the common case and is answered by
/// inspection. Toolkits that declare a *tiled* opaque region — several rects
/// that only together cover the surface — used to read as "not provably
/// opaque" and could never scan out, so a union test backs the fast path up:
/// sweep the rows the rect edges cut the surface into and require every band
/// to be spanned edge to edge.
fn opaque_region_covers(
    regions: Option<&[Rectangle<i32, Logical>]>,
    size: Size<i32, Logical>,
) -> bool {
    let Some(rs) = regions else { return false };
    if size.w <= 0 || size.h <= 0 {
        return false;
    }
    let covers_all = |r: &Rectangle<i32, Logical>| {
        r.loc.x <= 0 && r.loc.y <= 0 && r.loc.x + r.size.w >= size.w && r.loc.y + r.size.h >= size.h
    };
    if rs.iter().any(covers_all) {
        return true;
    }
    // More than a handful of rects and the sweep costs more than the frame it
    // might save; a client declaring its opacity in that many pieces is not
    // the fullscreen game this fast path exists for.
    if rs.len() < 2 || rs.len() > 16 {
        return false;
    }

    // Horizontal bands, cut at every rect's top and bottom edge (clamped to
    // the surface). Within a band no rect starts or stops, so a band is
    // covered iff the rects spanning it tile [0, size.w] with no gap.
    let mut edges: Vec<i32> = Vec::with_capacity(rs.len() * 2 + 2);
    edges.push(0);
    edges.push(size.h);
    for r in rs {
        for y in [r.loc.y, r.loc.y + r.size.h] {
            if y > 0 && y < size.h {
                edges.push(y);
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();

    edges.windows(2).all(|band| {
        let (top, bottom) = (band[0], band[1]);
        // Rects covering this whole band, by left edge.
        let mut spans: Vec<(i32, i32)> = rs
            .iter()
            .filter(|r| r.loc.y <= top && r.loc.y + r.size.h >= bottom)
            .map(|r| (r.loc.x, r.loc.x + r.size.w))
            .collect();
        spans.sort_unstable();
        let mut reached = 0;
        for (start, end) in spans {
            if start > reached {
                return false; // gap
            }
            reached = reached.max(end);
            if reached >= size.w {
                return true;
            }
        }
        reached >= size.w
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

/// Formats whose alpha channel is vestigial: 2-bit-alpha 10-bit and fp16 —
/// the HDR swapchain formats. No UI translucency fits in 2 alpha bits,
/// Vulkan swapchains present opaque frames, and no client declares opaque
/// regions on them (Wine doesn't) — so the fast paths treat them as opaque
/// rather than reject every HDR game.
fn vestigial_alpha(code: Fourcc) -> bool {
    matches!(
        code,
        Fourcc::Argb2101010
            | Fourcc::Abgr2101010
            | Fourcc::Rgba1010102
            | Fourcc::Bgra1010102
            | Fourcc::Abgr16161616f
            | Fourcc::Argb16161616f
    )
}

/// Whether `surface`'s committed buffer provably covers its whole extent
/// opaquely: per smithay's computed opaque regions (a no-alpha buffer gets
/// a full-extent region automatically; an alpha buffer carries the client's
/// declared region), or by carrying a [`vestigial_alpha`] format. Regions
/// are checked against both the buffer's logical size and the surface
/// view's destination — client-declared regions are surface-local while
/// smithay's auto-region is view-sized, and a full cover in either unit
/// system is a genuine full-coverage declaration. Used by the composite
/// fast paths that skip drawing anything underneath the surface.
fn surface_provably_opaque(surface: &WlSurface) -> bool {
    with_renderer_surface_state(surface, |state| {
        let (Some(buf), Some(view)) = (state.buffer_size(), state.view()) else {
            return false;
        };
        if let Some(buffer) = state.buffer()
            && let Ok(dmabuf) = smithay::wayland::dmabuf::get_dmabuf(buffer)
            && vestigial_alpha(dmabuf.format().code)
        {
            return true;
        }
        opaque_region_covers(state.opaque_regions(), buf)
            || opaque_region_covers(state.opaque_regions(), view.dst)
    })
    .unwrap_or(false)
}

/// Whether `root`'s visually-topmost buffer (the game's swapchain content —
/// see [`covering_top_node`]) is a dmabuf in the XRGB/ARGB channel order
/// whose GL sampling arrives R↔B swapped (see [`HDR_DECODE_SWIZZLE_SHADER`]).
fn window_buffer_rb_swapped(root: &WlSurface) -> bool {
    let Some((node, _)) = covering_top_node(root) else {
        return false;
    };
    with_renderer_surface_state(&node, |state| {
        state.buffer().and_then(|buffer| {
            smithay::wayland::dmabuf::get_dmabuf(buffer).ok().map(|d| {
                matches!(
                    d.format().code,
                    Fourcc::Argb2101010
                        | Fourcc::Xrgb2101010
                        | Fourcc::Argb16161616f
                        | Fourcc::Xrgb16161616f
                )
            })
        })
    })
    .flatten()
    .unwrap_or(false)
}

/// The visually-topmost mapped node of `root`'s surface tree — the first
/// buffer-carrying surface in top-to-bottom traversal — with its view
/// rect in window-buffer-local coordinates (subsurface offsets folded
/// in). The scanout / base-skip fast paths need "one buffer visually IS
/// this window": a plain toplevel is its own top node, while e.g. Wine
/// Wayland mounts a game's Vulkan swapchain on a subsurface stacked
/// above the host toplevel's buffer — that subsurface is what's actually
/// on screen, and when its rect covers the whole output, everything
/// beneath it in the tree is invisible.
fn covering_top_node(root: &WlSurface) -> Option<(WlSurface, Rectangle<i32, Logical>)> {
    use std::cell::RefCell;
    let found: RefCell<Option<(WlSurface, Rectangle<i32, Logical>)>> = RefCell::new(None);
    with_surface_tree_downward(
        root,
        Point::<i32, Logical>::from((0, 0)),
        |_, states, loc| {
            if found.borrow().is_some() {
                return TraversalAction::SkipChildren;
            }
            let mut loc = *loc;
            let data = states.data_map.get::<RendererSurfaceStateUserData>();
            if let Some(view) = data.and_then(|d| d.lock().unwrap().view()) {
                loc += view.offset;
                TraversalAction::DoChildren(loc)
            } else {
                // An unmapped parent hides its children too.
                TraversalAction::SkipChildren
            }
        },
        |surface, states, loc| {
            if found.borrow().is_some() {
                return;
            }
            let data = states.data_map.get::<RendererSurfaceStateUserData>();
            let Some(data) = data else { return };
            let state = data.lock().unwrap();
            if state.buffer().is_none() {
                return;
            }
            let Some(view) = state.view() else { return };
            found.replace(Some((
                surface.clone(),
                Rectangle::new(*loc + view.offset, view.dst),
            )));
        },
        |_, _, _| true,
    );
    found.into_inner()
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

/// Draw `src` of `tex` into `dst` of the bound `target`, opaquely and with no
/// shader of its own — the one-pass copy the direct-scanout capture path uses
/// in place of compositing the scene.
fn blit_texture(
    gles: &mut GlesRenderer,
    target: &mut GlesTarget<'_>,
    tex: &GlesTexture,
    src: Rectangle<f64, smithay::utils::Buffer>,
    dst: Rectangle<i32, Physical>,
) -> Result<()> {
    let mut frame = gles
        .render(target, dst.size, Transform::Normal)
        .context("capture blit: begin frame")?;
    frame
        .render_texture_from_to(
            tex,
            src,
            dst,
            &[dst],
            &[dst],
            Transform::Normal,
            1.0,
            None,
            &[],
        )
        .context("capture blit: draw")?;
    // Same-context sequential GL: the read-back (or the client's own use of
    // the dmabuf, ordered by its implicit fence) follows this draw, so the
    // returned sync point needn't be awaited — matching `capture_dmabuf`.
    let _ = frame.finish().context("capture blit: finish")?;
    Ok(())
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
        // The returned SyncPoint is dropped: same-context GL ordering plus
        // the dmabuf's implicit fence cover the client's read, matching the
        // pre-git-smithay behaviour (blit used to return no sync at all).
        Ok(_) => CaptureOutcome::Dmabuf,
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
/// Ensure `scratch` has a tier buffer at `slot`, allocating on demand.
///
/// The four band tiers exist from the start; the per-window ones
/// ([`MAX_WINDOW_TIERS`]) are only built when a frame actually stacks that
/// many translucent windows, so an ordinary desktop pays for none of them.
fn ensure_tier_slot(gles: &mut GlesRenderer, scratch: &mut BlurScratch, slot: usize) -> bool {
    while scratch.tiers.len() <= slot {
        match gles.create_buffer(Fourcc::Abgr8888, scratch.size) {
            Ok(t) => scratch.tiers.push(t),
            Err(err) => {
                warn!(error = %err, slot, "blur: per-window tier alloc failed");
                return false;
            }
        }
    }
    true
}

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
            draw_fill(frame, &black, output, output, &[full_dst], hdr, reference_white, saturation)?;
            let scale = (ow / tw).min(oh / th);
            let (dw, dh) = ((tw * scale) as i32, (th * scale) as i32);
            let dst = Rectangle::new(
                Point::from(((output.w - dw) / 2, (output.h - dh) / 2)),
                Size::from((dw, dh)),
            );
            draw(frame, buf(0.0, 0.0, tw, th), dst)?;
        }
        ScaleMode::Center => {
            draw_fill(frame, &black, output, output, &[full_dst], hdr, reference_white, saturation)?;
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

#[allow(
    clippy::too_many_arguments,
    reason = "mirrors draw_fill_rect; the args are the fill's full draw context"
)]
fn draw_fill(
    frame: &mut GlesFrame<'_, '_>,
    fill: &Fill,
    rect: Size<i32, Physical>,
    output_size: Size<i32, Physical>,
    damage: &[Rectangle<i32, Physical>],
    hdr: bool,
    reference_white: u32,
    saturation: f32,
) -> Result<()> {
    draw_fill_rect(
        frame,
        fill,
        Rectangle::<i32, Physical>::from_size(rect),
        output_size,
        damage,
        hdr,
        reference_white,
        saturation,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "one value per independent draw input (colour pipeline + damage); a struct would restate the same names"
)]
fn draw_fill_rect(
    frame: &mut GlesFrame<'_, '_>,
    fill: &Fill,
    rect: Rectangle<i32, Physical>,
    output_size: Size<i32, Physical>,
    damage: &[Rectangle<i32, Physical>],
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
            let rel = damage_rel(damage, rect);
            frame
                .draw_solid(rect, &rel, conv(Color32F::new(rgb[0], rgb[1], rgb[2], 1.0)))
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
                let rel = damage_rel(damage, stripe_dst);
                if rel.is_empty() {
                    continue;
                }
                frame
                    .draw_solid(stripe_dst, &rel, color)
                    .context("Frame::draw_solid (fill stripe) failed")?;
            }
        }
    }
    Ok(())
}


/// Titlebar point size in physical pixels for an output at `scale`.
/// The bar is rasterized into a physical-pixel offscreen, so an
/// unscaled point size would draw tiny text on a `HiDPI` display.
#[allow(
    clippy::cast_possible_truncation,
    reason = "a config-bounded point size (1.0..=200.0) times a display scale; f32 is exact well past that"
)]
fn bar_font_px(font_size: f32, scale: f64) -> f32 {
    (f64::from(font_size) * scale) as f32
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

/// Draw the quick-tile preview: a tinted wash where the dragged window
/// would land, framed so its edges read against a busy desktop.
///
/// `rect` is in absolute compositor coords and is clipped to this
/// output, so a preview on another monitor simply draws nothing here.
#[allow(
    clippy::too_many_arguments,
    reason = "one call site; the alternative is a struct that exists only to be destructured back into these"
)]
fn draw_snap_preview(
    frame: &mut GlesFrame<'_, '_>,
    rect: Rectangle<i32, Physical>,
    accent: [f32; 3],
    compositor_position: Point<i32, Physical>,
    mode_size: Size<i32, Physical>,
    scale: f64,
    hdr: bool,
    reference_white: u32,
    saturation: f32,
) -> Result<()> {
    // Light enough to read the window and the desktop through, strong
    // enough to be unmistakable at a glance.
    let wash = Color32F::new(accent[0], accent[1], accent[2], 0.22);
    let edge = Color32F::new(accent[0], accent[1], accent[2], 0.85);

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
        let r = Rectangle::<i32, Physical>::new(Point::from((x, y)), Size::from((w, h)));
        frame
            .draw_solid(r, &[Rectangle::from_size(r.size)], color)
            .context("Frame::draw_solid (snap preview) failed")
    };

    let x0 = scale_i(rect.loc.x - compositor_position.x, scale).clamp(0, mode_size.w);
    let y0 = scale_i(rect.loc.y - compositor_position.y, scale).clamp(0, mode_size.h);
    let x1 = (scale_i(rect.loc.x - compositor_position.x, scale) + scale_i(rect.size.w, scale))
        .clamp(0, mode_size.w);
    let y1 = (scale_i(rect.loc.y - compositor_position.y, scale) + scale_i(rect.size.h, scale))
        .clamp(0, mode_size.h);
    let (w, h) = (x1 - x0, y1 - y0);
    if w <= 0 || h <= 0 {
        return Ok(());
    }
    solid(frame, x0, y0, w, h, wash)?;
    let t = scale_i(2, scale).max(2);
    solid(frame, x0, y0, w, t, edge)?;
    solid(frame, x0, y1 - t, w, t, edge)?;
    solid(frame, x0, y0, t, h, edge)?;
    solid(frame, x1 - t, y0, t, h, edge)?;
    Ok(())
}

/// Everything the toolbar needs from the frame's colour state, bundled so
/// the draw call doesn't take a dozen arguments.
struct OverlayPaint<'a> {
    segment: &'a GlesTexProgram,
    blank: &'a GlesTexture,
    origin: Point<i32, Physical>,
    scale: f64,
    hdr: bool,
    reference_white: u32,
    saturation: f32,
}

impl OverlayPaint<'_> {
    /// A solid rect in output-physical pixels, converted for the HDR scene
    /// when there is one (`draw_solid` bypasses the decode override).
    fn fill(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        r: Rectangle<i32, Physical>,
        colour: Color32F,
    ) -> Result<()> {
        if r.size.w <= 0 || r.size.h <= 0 {
            return Ok(());
        }
        let colour = if self.hdr {
            srgb_to_linear_bt2020(colour, self.reference_white, self.saturation)
        } else {
            colour
        };
        frame
            .draw_solid(r, &[Rectangle::from_size(r.size)], colour)
            .context("Frame::draw_solid (screenshot toolbar) failed")
    }

    /// A compositor rect in this output's physical pixels.
    fn phys(&self, r: Rectangle<i32, Physical>) -> Rectangle<i32, Physical> {
        Rectangle::new(
            Point::from((
                scale_i(r.loc.x - self.origin.x, self.scale),
                scale_i(r.loc.y - self.origin.y, self.scale),
            )),
            Size::from((scale_i(r.size.w, self.scale), scale_i(r.size.h, self.scale))),
        )
    }

    /// Stroke a unit-box segment list across `dst`, on the GPU.
    fn strokes(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        dst: Rectangle<i32, Physical>,
        segs: &[[f32; 4]],
        colour: [f32; 3],
        thickness: f32,
    ) -> Result<()> {
        if segs.is_empty() || dst.size.w <= 0 || dst.size.h <= 0 {
            return Ok(());
        }
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "button sizes are a few dozen pixels and the segment count is capped at SEGMENTS_MAX"
        )]
        let (w, h) = (dst.size.w as f32, dst.size.h as f32);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "capped at SEGMENTS_MAX (12)"
        )]
        let count = segs.len().min(SEGMENTS_MAX) as i32;
        let mut uniforms = vec![
            Uniform::new("count", count),
            Uniform::new("colour", colour),
            Uniform::new("thickness", thickness),
            Uniform::new("quad", (w, h)),
        ];
        for (i, s) in segs.iter().take(SEGMENTS_MAX).enumerate() {
            uniforms.push(Uniform::new(
                format!("segments[{i}]"),
                [s[0] * w, s[1] * h, s[2] * w, s[3] * h],
            ));
        }
        let src = Rectangle::<f64, smithay::utils::Buffer>::from_size(Size::from((1.0, 1.0)));
        frame
            .render_texture_from_to(
                self.blank,
                src,
                dst,
                &[Rectangle::from_size(dst.size)],
                &[],
                Transform::Normal,
                1.0,
                Some(self.segment),
                &uniforms,
            )
            .context("segment stroke draw failed")
    }
}

/// Draw annotation strokes straight onto the framebuffer.
///
/// Each polyline is cut into runs of at most [`SEGMENTS_MAX`] segments,
/// and each run is stroked over its own bounding quad — the shader takes
/// a fixed-size uniform array (GLES 2.0 wants constant loop bounds), and
/// a tight quad per run also keeps the fragment work proportional to the
/// ink rather than to the screen.
fn draw_strokes(
    frame: &mut GlesFrame<'_, '_>,
    strokes: &[StrokeDraw],
    paint: &OverlayPaint<'_>,
) -> Result<()> {
    for stroke in strokes {
        if stroke.points.is_empty() {
            continue;
        }
        // A single point is a dot: a zero-length segment, which the
        // shader's clamped projection renders as a round cap.
        let pts: Vec<(i32, i32)> = if stroke.points.len() == 1 {
            vec![stroke.points[0], stroke.points[0]]
        } else {
            stroke.points.clone()
        };
        for run in pts.windows(2).collect::<Vec<_>>().chunks(SEGMENTS_MAX) {
            // Bounding box of this run, grown by the stroke's half-width
            // plus a pixel of feather so nothing is clipped at the edge.
            #[allow(
                clippy::cast_possible_truncation,
                reason = "pen widths are bounded by PEN_MAX"
            )]
            let pad = (stroke.width * 0.5).ceil() as i32 + 2;
            let (mut x0, mut y0) = (i32::MAX, i32::MAX);
            let (mut x1, mut y1) = (i32::MIN, i32::MIN);
            for seg in run {
                for (px, py) in *seg {
                    x0 = x0.min(*px);
                    y0 = y0.min(*py);
                    x1 = x1.max(*px);
                    y1 = y1.max(*py);
                }
            }
            let quad = Rectangle::<i32, Physical>::new(
                Point::from((x0 - pad, y0 - pad)),
                Size::from(((x1 - x0 + 2 * pad).max(1), (y1 - y0 + 2 * pad).max(1))),
            );
            let dst = paint.phys(quad);
            if dst.size.w <= 0 || dst.size.h <= 0 {
                continue;
            }
            // Segment coordinates are given to the shader as a fraction of
            // the quad, which is what `strokes` scales by the quad size.
            #[allow(
                clippy::cast_precision_loss,
                reason = "screen-sized pixel counts are exact in f32 well past 4K"
            )]
            let segs: Vec<[f32; 4]> = run
                .iter()
                .map(|seg| {
                    let f = |v: i32, lo: i32, span: i32| (v - lo) as f32 / span.max(1) as f32;
                    [
                        f(seg[0].0, quad.loc.x, quad.size.w),
                        f(seg[0].1, quad.loc.y, quad.size.h),
                        f(seg[1].0, quad.loc.x, quad.size.w),
                        f(seg[1].1, quad.loc.y, quad.size.h),
                    ]
                })
                .collect();
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_precision_loss,
                reason = "output scale is a small positive factor"
            )]
            let thickness = stroke.width * paint.scale as f32;
            paint.strokes(frame, dst, &segs, stroke.colour, thickness)?;
        }
    }
    Ok(())
}

/// Draw the options bar: a dark slab, a hover/active wash per button, and
/// each glyph stroked straight onto the framebuffer.
fn draw_toolbar(
    frame: &mut GlesFrame<'_, '_>,
    bar: &Toolbar,
    paint: &OverlayPaint<'_>,
) -> Result<()> {
    const SLAB: Color32F = Color32F::new(0.09, 0.09, 0.11, 0.94);
    const HOVER: Color32F = Color32F::new(1.0, 1.0, 1.0, 0.14);
    const ACTIVE: Color32F = Color32F::new(0.25, 0.62, 1.0, 0.55);
    const GLYPH: [f32; 3] = [0.93, 0.93, 0.95];

    paint.fill(frame, paint.phys(bar.bar), SLAB)?;
    for b in &bar.buttons {
        let dst = paint.phys(b.rect);
        if b.active {
            paint.fill(frame, dst, ACTIVE)?;
        } else if b.hovered {
            paint.fill(frame, dst, HOVER)?;
        }
        match b.icon {
            ToolIcon::Slider { frac, width, colour } => {
                // Track down the middle, filled to the current value.
                let th = (dst.size.h / 8).max(2);
                let ty = dst.loc.y + (dst.size.h - th) / 2;
                paint.fill(
                    frame,
                    Rectangle::new(
                        Point::from((dst.loc.x, ty)),
                        Size::from((dst.size.w, th)),
                    ),
                    Color32F::new(1.0, 1.0, 1.0, 0.22),
                )?;
                #[allow(
                    clippy::cast_precision_loss,
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "frac is 0..1 and the track is a few dozen pixels"
                )]
                let filled = (dst.size.w as f32 * frac.clamp(0.0, 1.0)) as i32;
                paint.fill(
                    frame,
                    Rectangle::new(
                        Point::from((dst.loc.x, ty)),
                        Size::from((filled, th)),
                    ),
                    Color32F::new(colour[0], colour[1], colour[2], 0.95),
                )?;
                // The knob is drawn at the pen's actual width, so the
                // slider is a preview of the stroke as well as a control.
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "pen widths are bounded by PEN_MAX"
                )]
                let knob = (width.max(4.0) as i32).min(dst.size.h);
                paint.fill(
                    frame,
                    Rectangle::new(
                        Point::from((
                            (dst.loc.x + filled - knob / 2)
                                .clamp(dst.loc.x, dst.loc.x + dst.size.w - knob),
                            dst.loc.y + (dst.size.h - knob) / 2,
                        )),
                        Size::from((knob, knob)),
                    ),
                    Color32F::new(colour[0], colour[1], colour[2], 1.0),
                )?;
            }
            ToolIcon::Swatch(rgb) => {
                // The swatch *is* the icon: inset so the active wash
                // behind it still reads as a ring around the colour.
                let inset = (dst.size.w / 5).max(2);
                let sw = Rectangle::new(
                    Point::from((dst.loc.x + inset, dst.loc.y + inset)),
                    Size::from((dst.size.w - 2 * inset, dst.size.h - 2 * inset)),
                );
                paint.fill(frame, sw, Color32F::new(rgb[0], rgb[1], rgb[2], 1.0))?;
            }
            icon => {
                // Stroke width tracks the button, so the glyphs stay in
                // proportion at any output scale.
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "button sizes are a few dozen pixels"
                )]
                let thickness = (dst.size.w as f32 / 14.0).max(1.5);
                paint.strokes(frame, dst, tool_segments(icon), GLYPH, thickness)?;
            }
        }
    }
    Ok(())
}

/// The unit-square segment list for a tool glyph, in a 0..1 box. Scaled
/// to the button at draw time so the icons stay sharp at any output
/// scale — there is no bitmap anywhere in this path.
fn tool_segments(icon: ToolIcon) -> &'static [[f32; 4]] {
    match icon {
        // A tick: down-right, then up-right and longer.
        ToolIcon::Take => &[[0.20, 0.52, 0.42, 0.74], [0.42, 0.74, 0.80, 0.26]],
        // A pencil: the shaft, then the two short edges of its nib.
        ToolIcon::Draw => &[
            [0.24, 0.76, 0.68, 0.24],
            [0.68, 0.24, 0.80, 0.34],
            [0.80, 0.34, 0.36, 0.84],
            [0.36, 0.84, 0.24, 0.76],
        ],
        // A capital T.
        ToolIcon::Text => &[[0.22, 0.24, 0.78, 0.24], [0.50, 0.24, 0.50, 0.78]],
        ToolIcon::Cancel => &[[0.26, 0.26, 0.74, 0.74], [0.74, 0.26, 0.26, 0.74]],
        // Drawn as filled quads instead; no strokes.
        ToolIcon::Swatch(_) | ToolIcon::Slider { .. } => &[],
    }
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

    if overlay.handles {
        // Eight grips: four corners, four edge midpoints. They say the
        // rect can still be changed, which is the whole point of not
        // taking the picture on release.
        let g = scale_i(8, scale).max(6);
        let half = g / 2;
        let (mx, my) = (x0 + w / 2 - half, y0 + h / 2 - half);
        for (gx, gy) in [
            (x0, y0),
            (x1 - g, y0),
            (x0, y1 - g),
            (x1 - g, y1 - g),
            (mx, y0),
            (mx, y1 - g),
            (x0, my),
            (x1 - g, my),
        ] {
            solid(frame, gx, gy, g, g, OUTLINE)?;
        }
    }
    Ok(())
}

/// Walk a surface tree and drain every queued `wl_callback`, firing
/// `done(time_ms)` on each so the client knows to schedule its next
/// frame. Smithay's `desktop::send_frames_surface_tree` does this
/// plus primary-scanout-output filtering and throttling, all of
/// which presuppose a `Space<Window>` we don't have yet (4d); this
/// minimal version is enough for 4b — every visible surface gets a
/// callback per vblank cycle.
pub(crate) fn send_frame_callbacks(surface: &WlSurface, time_ms: u32) {
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

/// A toplevel's title, for its titlebar.
///
/// Xdg only: this reads the xdg-shell role data, and Xwayland surfaces
/// carry their `WM_NAME` on the X11 side instead. X11 windows therefore
/// get a bar with buttons and no text — worth fixing, but a titleless
/// bar still drags, maximizes and closes.
fn window_title(surface: &WlSurface) -> Option<String> {
    with_states(surface, |states| {
        states
            .data_map
            .get::<XdgToplevelSurfaceData>()
            .and_then(|d| d.lock().ok().and_then(|d| d.title.clone()))
    })
}

/// A toplevel's `app_id`, which names its icon. Xdg only, like
/// [`window_title`].
fn window_app_id(surface: &WlSurface) -> Option<String> {
    with_states(surface, |states| {
        states
            .data_map
            .get::<XdgToplevelSurfaceData>()
            .and_then(|d| d.lock().ok().and_then(|d| d.app_id.clone()))
    })
}

/// Resolve a [`Fill`] to its top and bottom stop colours, and mix two of
/// them by `t` (`0.0` = all of `a`, `1.0` = all of `b`).
///
/// A solid fill is just a gradient whose stops match, so mixing a solid with
/// a gradient works without a special case — which is what lets the focused
/// and unfocused border fills be different kinds.
fn mix_fills(a: &Fill, b: &Fill, t: f32) -> ([f32; 3], [f32; 3]) {
    let stops = |f: &Fill| match f {
        Fill::Solid(rgb) => (*rgb, *rgb),
        Fill::VerticalGradient { top, bottom } => (*top, *bottom),
    };
    let (a_top, a_bottom) = stops(a);
    let (b_top, b_bottom) = stops(b);
    let t = t.clamp(0.0, 1.0);
    let mix = |x: [f32; 3], y: [f32; 3]| {
        [
            x[0] + (y[0] - x[0]) * t,
            x[1] + (y[1] - x[1]) * t,
            x[2] + (y[2] - x[2]) * t,
        ]
    };
    (mix(a_top, b_top), mix(a_bottom, b_bottom))
}

/// Interpolate two positions by eased `t`. Position and size are separate
/// helpers because a window's move and resize animations run on independent
/// clocks (`animations.window_move` / `window_resize`).
#[allow(
    clippy::cast_possible_truncation,
    reason = "interpolated pixel coordinates are bounded by output size, well within i32"
)]
fn lerp_point(
    a: Point<i32, Physical>,
    b: Point<i32, Physical>,
    t: f64,
) -> Point<i32, Physical> {
    Point::from((
        lerp(f64::from(a.x), f64::from(b.x), t).round() as i32,
        lerp(f64::from(a.y), f64::from(b.y), t).round() as i32,
    ))
}

/// Interpolate two sizes by eased `t`. See [`lerp_point`].
#[allow(
    clippy::cast_possible_truncation,
    reason = "interpolated pixel dimensions are bounded by output size, well within i32"
)]
fn lerp_size(a: Size<i32, Physical>, b: Size<i32, Physical>, t: f64) -> Size<i32, Physical> {
    Size::from((
        lerp(f64::from(a.w), f64::from(b.w), t).round() as i32,
        lerp(f64::from(a.h), f64::from(b.h), t).round() as i32,
    ))
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

/// Headless GPU pass benchmark: measures the raw cost of every composite
/// pass this renderer uses, on a surfaceless EGL context over the render
/// node — no DRM master, no seat, safe to run inside a live session:
///
/// ```text
/// cargo test gpu_bench -- --ignored --nocapture
/// ```
///
/// What it can measure: per-pass GPU milliseconds at 4K (plain copy, SDR
/// decode, fused SDR→PQ, HDR encode, the blur pyramid, decoration
/// offscreens with/without a cached allocation). What it can't: KMS
/// flips, VRR pacing, real client buffers — those need the live session
/// (see `RenderProfile`, logged every 5 s). Timing brackets each
/// iteration with a fence wait, so numbers are slightly pessimistic
/// (no cross-frame pipelining) but directly comparable to each other.
#[cfg(test)]
mod gpu_bench {
    use super::*;
    use smithay::backend::allocator::gbm::GbmDevice;
    use smithay::reexports::drm::node::DrmNode as BenchDrmNode;
    use smithay::utils::DeviceFd;
    use std::fs::OpenOptions;
    use std::os::fd::OwnedFd;

    const W: i32 = 3840;
    const H: i32 = 2160;
    const ITERS: u32 = 60;

    fn bench<F: FnMut() -> Result<()>>(name: &str, mut f: F) {
        // Warmup (shader compile, first-use allocations).
        for _ in 0..3 {
            f().expect("bench warmup failed");
        }
        let t0 = Instant::now();
        for _ in 0..ITERS {
            f().expect("bench iteration failed");
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(ITERS);
        println!("{name:<44} {ms:>8.3} ms");
    }

    #[test]
    #[ignore = "GPU benchmark; run manually with --ignored --nocapture"]
    fn gpu_bench() {
        let _ = BenchDrmNode::from_path("/dev/dri/renderD128");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/dri/renderD128")
            .expect("open render node");
        let fd = DrmDeviceFd::new(DeviceFd::from(OwnedFd::from(file)));
        let gbm = GbmDevice::new(fd).expect("GbmDevice");
        #[allow(
            unsafe_code,
            reason = "same contract as Renderer::new: the Arc-backed GbmDevice clone lives inside EGLDisplay for its full lifetime"
        )]
        // SAFETY: see #[allow] above.
        let display = unsafe { EGLDisplay::new(gbm.clone()) }.expect("EGLDisplay");
        let context = EGLContext::new(&display).expect("EGLContext");
        #[allow(
            unsafe_code,
            reason = "same contract as Renderer::new: the context is used from this single test thread only"
        )]
        // SAFETY: see #[allow] above.
        let mut gles = unsafe { GlesRenderer::new(context) }.expect("GlesRenderer");

        let uniforms = [
            UniformName::new("reference_white", UniformType::_1f),
            UniformName::new("saturation", UniformType::_1f),
        ];
        let sdr_decode = gles
            .compile_custom_texture_shader(SDR_DECODE_SHADER, &uniforms)
            .expect("sdr decode shader");
        let sdr_to_pq = gles
            .compile_custom_texture_shader(SDR_TO_PQ_SHADER, &uniforms)
            .expect("fused shader");
        let hdr_encode = gles
            .compile_custom_texture_shader(HDR_ENCODE_SHADER, &[])
            .expect("encode shader");
        let blur_uniforms = [
            UniformName::new("half_pixel", UniformType::_2f),
            UniformName::new("radius", UniformType::_1f),
        ];
        let blur_down = gles
            .compile_custom_texture_shader(BLUR_DOWN, &blur_uniforms)
            .expect("blur down");
        let blur_up = gles
            .compile_custom_texture_shader(BLUR_UP, &blur_uniforms)
            .expect("blur up");

        let size = Size::<i32, smithay::utils::Buffer>::from((W, H));
        let phys = Size::<i32, Physical>::from((W, H));
        let full = [Rectangle::<i32, Physical>::from_size(phys)];
        let src_rect = Rectangle::<f64, smithay::utils::Buffer>::from_size(Size::from((
            f64::from(W),
            f64::from(H),
        )));
        let dst_rect = Rectangle::<i32, Physical>::from_size(phys);

        let source: GlesTexture = gles
            .create_buffer(Fourcc::Abgr8888, size)
            .expect("source tex");
        let source_fp16: GlesTexture = gles
            .create_buffer(Fourcc::Abgr16161616f, size)
            .expect("fp16 source");
        let mut target8: GlesTexture = gles
            .create_buffer(Fourcc::Abgr8888, size)
            .expect("8-bit target");
        let mut target_fp16: GlesTexture = gles
            .create_buffer(Fourcc::Abgr16161616f, size)
            .expect("fp16 target");
        let mut target10: GlesTexture = gles
            .create_buffer(Fourcc::Abgr2101010, size)
            .expect("10-bit target");

        // One full-screen textured pass into `target` with `program`.
        macro_rules! pass {
            ($target:expr, $tex:expr, $program:expr, $unis:expr) => {{
                let mut bound = gles.bind($target)?;
                let mut frame = gles.render(&mut bound, phys, Transform::Normal)?;
                frame.render_texture_from_to(
                    $tex,
                    src_rect,
                    dst_rect,
                    &full,
                    &full,
                    Transform::Normal,
                    1.0,
                    $program,
                    $unis,
                )?;
                let sync = frame.finish()?;
                drop(bound);
                let _ = sync.wait();
                Result::<()>::Ok(())
            }};
        }
        let unis = [
            Uniform::new("reference_white", 400.0_f32),
            Uniform::new("saturation", 1.0_f32),
        ];

        println!("\n== libreland GPU pass benchmark: {W}x{H}, {ITERS} iters, fence-bounded ==");
        bench("copy 8-bit -> 8-bit (default sampler)", || {
            pass!(&mut target8, &source, None, &[])
        });
        bench("SDR decode 8-bit -> fp16 (HDR scene draw)", || {
            pass!(&mut target_fp16, &source, Some(&sdr_decode), &unis)
        });
        bench("PQ encode fp16 -> 10-bit (HDR final pass)", || {
            pass!(&mut target10, &source_fp16, Some(&hdr_encode), &[])
        });
        bench("fused SDR->PQ 8-bit -> 10-bit (single-pass)", || {
            pass!(&mut target10, &source, Some(&sdr_to_pq), &unis)
        });

        // Damage tracking: the same full-screen composite clipped to a
        // damage region ~5% of the output — what a partial repaint frame
        // costs vs the full-frame passes above.
        let small = [Rectangle::<i32, Physical>::new(
            Point::from((100, 100)),
            Size::from((860, 540)),
        )];
        bench("copy pass clipped to 5% damage", || {
            let mut bound = gles.bind(&mut target8)?;
            let mut frame = gles.render(&mut bound, phys, Transform::Normal)?;
            frame.render_texture_from_to(
                &source,
                src_rect,
                dst_rect,
                &small,
                &full,
                Transform::Normal,
                1.0,
                None,
                &[],
            )?;
            let sync = frame.finish()?;
            drop(bound);
            let _ = sync.wait();
            Ok(())
        });

        // Decoration offscreen: fresh allocation + draw each frame (the
        // old behaviour) vs drawing into a kept allocation (the cache's
        // stale-content path; a cache HIT costs no GPU work at all).
        let cell = Size::<i32, smithay::utils::Buffer>::from((1280, 1440));
        let cell_phys = Size::<i32, Physical>::from((1280, 1440));
        let cell_full = [Rectangle::<i32, Physical>::from_size(cell_phys)];
        let cell_src = Rectangle::<f64, smithay::utils::Buffer>::from_size(Size::from((
            1280.0_f64, 1440.0_f64,
        )));
        let cell_dst = Rectangle::<i32, Physical>::from_size(cell_phys);
        bench("win_tex: alloc + draw (old, per frame)", || {
            let mut tex: GlesTexture = gles.create_buffer(Fourcc::Abgr8888, cell)?;
            let mut bound = gles.bind(&mut tex)?;
            let mut frame = gles.render(&mut bound, cell_phys, Transform::Normal)?;
            frame.clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &cell_full)?;
            frame.render_texture_from_to(
                &source,
                cell_src,
                cell_dst,
                &cell_full,
                &cell_full,
                Transform::Normal,
                1.0,
                None,
                &[],
            )?;
            let sync = frame.finish()?;
            drop(bound);
            let _ = sync.wait();
            Ok(())
        });
        let mut kept: GlesTexture = gles.create_buffer(Fourcc::Abgr8888, cell).expect("kept tex");
        bench("win_tex: draw into kept alloc (stale cache)", || {
            let mut bound = gles.bind(&mut kept)?;
            let mut frame = gles.render(&mut bound, cell_phys, Transform::Normal)?;
            frame.clear(Color32F::new(0.0, 0.0, 0.0, 0.0), &cell_full)?;
            frame.render_texture_from_to(
                &source,
                cell_src,
                cell_dst,
                &cell_full,
                &cell_full,
                Transform::Normal,
                1.0,
                None,
                &[],
            )?;
            let sync = frame.finish()?;
            drop(bound);
            let _ = sync.wait();
            Ok(())
        });

        // Blur pyramid: N down + N up dual-filter passes over halving mips,
        // the shape of `run_pyramid` (per tier; three tiers can run per
        // frame). Levels allocated once like BlurScratch.
        let passes = 6usize;
        let mut levels: Vec<GlesTexture> = (0..=passes)
            .map(|k| {
                let w = (W >> k).max(1);
                let h = (H >> k).max(1);
                gles.create_buffer(
                    Fourcc::Abgr8888,
                    Size::<i32, smithay::utils::Buffer>::from((w, h)),
                )
                .expect("blur level")
            })
            .collect();
        bench("blur pyramid, 6 passes (one tier)", || {
            for k in 0..passes {
                let (src_slice, dst_slice) = levels.split_at_mut(k + 1);
                let s = &src_slice[k];
                let d = &mut dst_slice[0];
                let dw = (W >> (k + 1)).max(1);
                let dh = (H >> (k + 1)).max(1);
                let dphys = Size::<i32, Physical>::from((dw, dh));
                let dfull = [Rectangle::<i32, Physical>::from_size(dphys)];
                let hp = [
                    Uniform::new("half_pixel", [0.5 / f64::from(dw) as f32, 0.5 / f64::from(dh) as f32]),
                    Uniform::new("radius", 8.0_f32),
                ];
                let mut bound = gles.bind(d)?;
                let mut frame = gles.render(&mut bound, dphys, Transform::Normal)?;
                frame.render_texture_from_to(
                    s,
                    Rectangle::<f64, smithay::utils::Buffer>::from_size(Size::from((
                        f64::from((W >> k).max(1)),
                        f64::from((H >> k).max(1)),
                    ))),
                    Rectangle::<i32, Physical>::from_size(dphys),
                    &dfull,
                    &dfull,
                    Transform::Normal,
                    1.0,
                    Some(&blur_down),
                    &hp,
                )?;
                let _ = frame.finish()?;
                drop(bound);
            }
            for k in (0..passes).rev() {
                let (dst_slice, src_slice) = levels.split_at_mut(k + 1);
                let s = &src_slice[0];
                let d = &mut dst_slice[k];
                let dw = (W >> k).max(1);
                let dh = (H >> k).max(1);
                let dphys = Size::<i32, Physical>::from((dw, dh));
                let dfull = [Rectangle::<i32, Physical>::from_size(dphys)];
                let hp = [
                    Uniform::new("half_pixel", [0.5 / f64::from(dw) as f32, 0.5 / f64::from(dh) as f32]),
                    Uniform::new("radius", 8.0_f32),
                ];
                let mut bound = gles.bind(d)?;
                let mut frame = gles.render(&mut bound, dphys, Transform::Normal)?;
                let sync = frame.render_texture_from_to(
                    s,
                    Rectangle::<f64, smithay::utils::Buffer>::from_size(Size::from((
                        f64::from((W >> (k + 1)).max(1)),
                        f64::from((H >> (k + 1)).max(1)),
                    ))),
                    Rectangle::<i32, Physical>::from_size(dphys),
                    &dfull,
                    &dfull,
                    Transform::Normal,
                    1.0,
                    Some(&blur_up),
                    &hp,
                )
                .map(|()| frame.finish())?;
                drop(bound);
                let sync = sync?;
                if k == 0 {
                    let _ = sync.wait();
                }
            }
            Ok(())
        });
        println!();
    }

    /// Verifies the temporal-min masking of the layer backdrop blur (the fix
    /// for the popup-open fullscreen frost flash): with the *same* current
    /// coverage, whether a pixel gets frosted depends on whether it was also
    /// covered last frame. A full-surface frame that was not covered before
    /// (the client's transient map frame) frosts nothing; a stably-covered
    /// pixel (a real panel body) frosts fully. This is what no alpha threshold
    /// could achieve, since the flash is the same 0.79 alpha as the real card.
    #[test]
    #[ignore = "GPU test; run manually with --ignored --nocapture"]
    fn temporal_mask_blur_confines_frost() {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/dri/renderD128")
            .expect("open render node");
        let fd = DrmDeviceFd::new(DeviceFd::from(OwnedFd::from(file)));
        let gbm = GbmDevice::new(fd).expect("GbmDevice");
        #[allow(unsafe_code, reason = "test-only surfaceless EGL, single thread")]
        // SAFETY: GbmDevice clone lives in the EGLDisplay for its lifetime.
        let display = unsafe { EGLDisplay::new(gbm.clone()) }.expect("EGLDisplay");
        let context = EGLContext::new(&display).expect("EGLContext");
        #[allow(unsafe_code, reason = "test-only single-thread renderer")]
        // SAFETY: used only from this test thread.
        let mut gles = unsafe { GlesRenderer::new(context) }.expect("GlesRenderer");

        let shader = gles
            .compile_custom_texture_shader(
                MASK_BLUR_SHADER,
                &[
                    UniformName::new("mask", UniformType::_1i),
                    UniformName::new("mask_prev", UniformType::_1i),
                    UniformName::new("mask_mul", UniformType::_2f),
                    UniformName::new("mask_add", UniformType::_2f),
                    UniformName::new("mask_dilate", UniformType::_2f),
                ],
            )
            .expect("compile mask blur shader");

        let size = Size::<i32, smithay::utils::Buffer>::from((64, 64));
        let phys = Size::<i32, Physical>::from((64, 64));
        let full = [Rectangle::<i32, Physical>::from_size(phys)];
        let src = Rectangle::<f64, smithay::utils::Buffer>::from_size(Size::from((64.0, 64.0)));
        let dst = Rectangle::<i32, Physical>::from_size(phys);

        // Uniform-colour textures via import_memory — imported textures have
        // sampler filters set (create_buffer ones don't, so they'd sample as
        // incomplete). Uniform content makes the mask UV mapping irrelevant.
        // Tier (blurred backdrop) is white/opaque so a frosted pixel reads white.
        let white: Vec<u8> = [255u8; 4].iter().copied().cycle().take(64 * 64 * 4).collect();
        let tier = gles
            .import_memory(&white, Fourcc::Abgr8888, size, false)
            .expect("import tier");
        let mut mk = |alpha: u8| -> GlesTexture {
            let data: Vec<u8> = [0u8, 0u8, 0u8, alpha]
                .iter()
                .copied()
                .cycle()
                .take(64 * 64 * 4)
                .collect();
            gles
                .import_memory(&data, Fourcc::Abgr8888, size, false)
                .expect("import mask")
        };
        let cur = mk(201); // panel material everywhere (0.79) — the flash
        let prev_none = mk(0); // not covered last frame
        let prev_full = mk(201); // stably covered

        // Render one masked blur over a red backdrop; return the centre texel.
        let unis = [
            Uniform::new("mask", 1i32),
            Uniform::new("mask_prev", 2i32),
            Uniform::new("mask_mul", (1.0f32, 1.0f32)),
            Uniform::new("mask_add", (0.0f32, 0.0f32)),
            // Zero radius: this test is about the veto itself, so pin the
            // dilation off and let `moving_mask_blur_survives_dilation` cover
            // the moving-content relaxation separately.
            Uniform::new("mask_dilate", (0.0f32, 0.0f32)),
        ];
        let mut run = |prev: &GlesTexture| -> [u8; 4] {
            let mut target: GlesTexture =
                gles.create_buffer(Fourcc::Abgr8888, size).expect("target");
            {
                let mut b = gles.bind(&mut target).expect("bind target");
                let mut f = gles.render(&mut b, phys, Transform::Normal).expect("render target");
                f.clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &full).expect("clear red");
                f.with_secondary_textures(&cur, prev, |f| {
                    f.render_texture_from_to(
                        &tier, src, dst, &full, &[], Transform::Normal, 1.0, Some(&shader), &unis,
                    )
                })
                .expect("masked blur draw");
                let s = f.finish().expect("finish");
                drop(b);
                let _ = s.wait();
            }
            let region = Rectangle::<i32, smithay::utils::Buffer>::from_size(Size::from((64, 64)));
            let bound = gles.bind(&mut target).expect("rebind");
            let mapping = gles
                .copy_framebuffer(&bound, region, Fourcc::Abgr8888)
                .expect("copy_framebuffer");
            let bytes = gles.map_texture(&mapping).expect("map").to_vec();
            let c = (32 * 64 + 32) * 4; // centre pixel
            [bytes[c], bytes[c + 1], bytes[c + 2], bytes[c + 3]]
        };

        let no_history = run(&prev_none);
        let stable = run(&prev_full);
        let sum = |p: [u8; 4]| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]);
        println!("no-history centre = {no_history:?} (sum {})", sum(no_history));
        println!("stable     centre = {stable:?} (sum {})", sum(stable));
        // Not covered last frame -> min coverage 0 -> no frost -> stays red backdrop.
        assert!(
            sum(no_history) < 400,
            "a full-surface frame with no prior coverage must NOT frost (got {no_history:?})"
        );
        // Covered both frames -> full frost -> becomes the white tier.
        assert!(
            sum(stable) > 600,
            "a stably-covered pixel must frost fully (got {stable:?})"
        );
    }

    /// The temporal veto must tolerate content that *moved*. Its original form
    /// point-sampled last frame's alpha, which silently assumed the client's
    /// content is stationary — true of a bar, false of a popup that slides
    /// into place. A sliding card lands on new pixels every frame, so its
    /// leading edge was vetoed and left un-frosted, and the frost tore along
    /// the card for the whole entrance (fine once it stopped).
    ///
    /// Dilating the veto (`MASK_DILATE_PX`) fixes it. Probed with vertical
    /// bands so the result cannot depend on texture y-inversion.
    #[test]
    #[ignore = "GPU test; run manually with --ignored --nocapture"]
    fn moving_mask_blur_survives_dilation() {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/dri/renderD128")
            .expect("open render node");
        let fd = DrmDeviceFd::new(DeviceFd::from(OwnedFd::from(file)));
        let gbm = GbmDevice::new(fd).expect("GbmDevice");
        #[allow(unsafe_code, reason = "test-only surfaceless EGL, single thread")]
        // SAFETY: GbmDevice clone lives in the EGLDisplay for its lifetime.
        let display = unsafe { EGLDisplay::new(gbm.clone()) }.expect("EGLDisplay");
        let context = EGLContext::new(&display).expect("EGLContext");
        #[allow(unsafe_code, reason = "test-only single-thread renderer")]
        // SAFETY: used only from this test thread.
        let mut gles = unsafe { GlesRenderer::new(context) }.expect("GlesRenderer");

        let shader = gles
            .compile_custom_texture_shader(
                MASK_BLUR_SHADER,
                &[
                    UniformName::new("mask", UniformType::_1i),
                    UniformName::new("mask_prev", UniformType::_1i),
                    UniformName::new("mask_mul", UniformType::_2f),
                    UniformName::new("mask_add", UniformType::_2f),
                    UniformName::new("mask_dilate", UniformType::_2f),
                ],
            )
            .expect("compile mask blur shader");

        let size = Size::<i32, smithay::utils::Buffer>::from((64, 64));
        let phys = Size::<i32, Physical>::from((64, 64));
        let full = [Rectangle::<i32, Physical>::from_size(phys)];
        let src = Rectangle::<f64, smithay::utils::Buffer>::from_size(Size::from((64.0, 64.0)));
        let dst = Rectangle::<i32, Physical>::from_size(phys);

        let white: Vec<u8> = [255u8; 4].iter().copied().cycle().take(64 * 64 * 4).collect();
        let tier = gles
            .import_memory(&white, Fourcc::Abgr8888, size, false)
            .expect("import tier");

        // The card, as a vertical band of columns [lo, hi) at panel alpha.
        let mut band = |lo: usize, hi: usize| -> GlesTexture {
            let mut data = vec![0u8; 64 * 64 * 4];
            for y in 0..64 {
                for x in lo..hi {
                    data[(y * 64 + x) * 4 + 3] = 201;
                }
            }
            gles
                .import_memory(&data, Fourcc::Abgr8888, size, false)
                .expect("import mask")
        };
        // Last frame the card sat over columns 16..32; this frame it covers
        // column 40 too — the pixel a moving card has just arrived on.
        let prev = band(16, 32);
        let cur = band(16, 48);

        let mut run = |dilate: f32| -> [u8; 4] {
            let unis = [
                Uniform::new("mask", 1i32),
                Uniform::new("mask_prev", 2i32),
                Uniform::new("mask_mul", (1.0f32, 1.0f32)),
                Uniform::new("mask_add", (0.0f32, 0.0f32)),
                Uniform::new("mask_dilate", (dilate / 64.0, dilate / 64.0)),
            ];
            let mut target: GlesTexture =
                gles.create_buffer(Fourcc::Abgr8888, size).expect("target");
            {
                let mut b = gles.bind(&mut target).expect("bind target");
                let mut f = gles.render(&mut b, phys, Transform::Normal).expect("render target");
                f.clear(Color32F::new(1.0, 0.0, 0.0, 1.0), &full).expect("clear red");
                f.with_secondary_textures(&cur, &prev, |f| {
                    f.render_texture_from_to(
                        &tier, src, dst, &full, &[], Transform::Normal, 1.0, Some(&shader), &unis,
                    )
                })
                .expect("masked blur draw");
                let s = f.finish().expect("finish");
                drop(b);
                let _ = s.wait();
            }
            let region = Rectangle::<i32, smithay::utils::Buffer>::from_size(Size::from((64, 64)));
            let bound = gles.bind(&mut target).expect("rebind");
            let mapping = gles
                .copy_framebuffer(&bound, region, Fourcc::Abgr8888)
                .expect("copy_framebuffer");
            let bytes = gles.map_texture(&mapping).expect("map").to_vec();
            let c = (32 * 64 + 40) * 4; // column 40: 8 px beyond last frame's card
            [bytes[c], bytes[c + 1], bytes[c + 2], bytes[c + 3]]
        };

        let undilated = run(0.0);
        // 32 px reach => half-radius taps land 16 px away, inside the old band.
        let dilated = run(32.0);
        let sum = |p: [u8; 4]| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]);
        println!("moved, undilated = {undilated:?} (sum {})", sum(undilated));
        println!("moved, dilated   = {dilated:?} (sum {})", sum(dilated));
        // The bug: a point-sampled veto leaves the newly-covered pixel unfrosted.
        assert!(
            sum(undilated) < 400,
            "point-sampled veto should suppress the moved-onto pixel (got {undilated:?})"
        );
        // The fix: content that merely moved still counts as persistent.
        assert!(
            sum(dilated) > 600,
            "a dilated veto must let moved-onto content frost (got {dilated:?})"
        );
    }

    /// Windows-scRGB must be decoded on its own terms — extended *linear*
    /// light where 1.0 is 80 cd/m² — and not as sRGB-gamma SDR anchored at
    /// `reference_white`.
    ///
    /// Wine/Proton tags a game's scRGB swapchain with the protocol's
    /// pre-defined `windows_scrgb` description and sets the Vulkan colour
    /// space to PASS_THROUGH, i.e. it hands the pixels over untouched and the
    /// compositor owns the entire conversion. Sending them through the SDR
    /// decode instead applies a gamma curve to linear data and anchors it at
    /// 203 rather than 80 cd/m² — which is what mis-rendered id Tech (DOOM)
    /// titles, the classic scRGB users, while HDR10/PQ games looked fine.
    ///
    /// An HDR screenshot has to keep the highlights *apart*.
    ///
    /// The tonemap used to end in `clamp(lin, 0.0, 1.0)`, which gave every
    /// value from SDR diffuse white up to the 10000-nit peak the same output.
    /// A game's sky, sun and sunlit ground all sit above diffuse white, so the
    /// bright majority of the frame collapsed into one flat white sheet —
    /// which is what "washed out, not what's on screen" looks like. Per-channel
    /// clamping also pins the top two channels together, so a sunlit orange
    /// (3.0, 1.4, 0.3) came out (1.0, 1.0, 0.3): yellow, not orange.
    ///
    /// Asserts the two properties that fixes it: distinct brightnesses stay
    /// distinct, and mid-tones below the knee are left exactly alone so an
    /// ordinary SDR desktop capture is unaffected.
    #[test]
    #[ignore = "GPU test; run manually with --ignored --nocapture"]
    fn hdr_screenshot_keeps_highlight_separation() {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/dri/renderD128")
            .expect("open render node");
        let fd = DrmDeviceFd::new(DeviceFd::from(OwnedFd::from(file)));
        let gbm = GbmDevice::new(fd).expect("GbmDevice");
        #[allow(unsafe_code, reason = "test-only surfaceless EGL, single thread")]
        // SAFETY: GbmDevice clone lives in the EGLDisplay for its lifetime.
        let display = unsafe { EGLDisplay::new(gbm.clone()) }.expect("EGLDisplay");
        let context = EGLContext::new(&display).expect("EGLContext");
        #[allow(unsafe_code, reason = "test-only single-thread renderer")]
        // SAFETY: used only from this test thread.
        let mut gles = unsafe { GlesRenderer::new(context) }.expect("GlesRenderer");

        let tonemap = gles
            .compile_custom_texture_shader(
                SCREENSHOT_TONEMAP_SHADER,
                &[
                    UniformName::new("reference_white", UniformType::_1f),
                    UniformName::new("knee", UniformType::_1f),
                ],
            )
            .expect("compile screenshot tonemap");

        const RW: f32 = 203.0;
        let size = Size::<i32, smithay::utils::Buffer>::from((16, 16));
        let phys = Size::<i32, Physical>::from((16, 16));
        let full = [Rectangle::<i32, Physical>::from_size(phys)];
        let src = Rectangle::<f64, smithay::utils::Buffer>::from_size(Size::from((16.0, 16.0)));
        let dst = Rectangle::<i32, Physical>::from_size(phys);

        // Tonemap one scene colour, given in cd/m² of linear BT.2020. The
        // scene stores nits/10000, which is what the shader renormalises.
        let mut shot = |nits: [f32; 3]| -> [u8; 4] {
            let mut scene: GlesTexture = gles
                .create_buffer(Fourcc::Abgr16161616f, size)
                .expect("fp16 scene");
            {
                let mut b = gles.bind(&mut scene).expect("bind scene");
                let mut f = gles.render(&mut b, phys, Transform::Normal).expect("render scene");
                f.clear(
                    Color32F::new(
                        nits[0] / 10000.0,
                        nits[1] / 10000.0,
                        nits[2] / 10000.0,
                        1.0,
                    ),
                    &full,
                )
                .expect("fill scene");
                let _ = f.finish().expect("finish scene");
            }
            let mut target: GlesTexture =
                gles.create_buffer(Fourcc::Abgr8888, size).expect("8-bit target");
            {
                let mut b = gles.bind(&mut target).expect("bind target");
                let mut f = gles.render(&mut b, phys, Transform::Normal).expect("render target");
                f.render_texture_from_to(
                    &scene,
                    src,
                    dst,
                    &full,
                    &[],
                    Transform::Normal,
                    1.0,
                    Some(&tonemap),
                    &[
                        Uniform::new("reference_white", RW),
                        Uniform::new("knee", SCREENSHOT_TONEMAP_KNEE),
                    ],
                )
                .expect("tonemap draw");
                let _ = f.finish().expect("finish target");
            }
            let region = Rectangle::<i32, smithay::utils::Buffer>::from_size(Size::from((16, 16)));
            let bound = gles.bind(&mut target).expect("rebind");
            let mapping = gles
                .copy_framebuffer(&bound, region, Fourcc::Abgr8888)
                .expect("copy_framebuffer");
            let bytes = gles.map_texture(&mapping).expect("map").to_vec();
            let c = (8 * 16 + 8) * 4;
            [bytes[c], bytes[c + 1], bytes[c + 2], bytes[c + 3]]
        };

        // Neutral steps from diffuse white up to well past it. Each has to be
        // strictly brighter than the last; under the old clamp all four read 255.
        let steps: Vec<(f32, [u8; 4])> = [203.0_f32, 400.0, 1000.0, 4000.0]
            .into_iter()
            .map(|n| (n, shot([n, n, n])))
            .collect();
        for (n, px) in &steps {
            println!("{n:>6} cd/m² -> {px:?}");
        }
        for w in steps.windows(2) {
            let (lo_n, lo) = w[0];
            let (hi_n, hi) = w[1];
            assert!(
                hi[0] > lo[0],
                "{hi_n} cd/m² must read brighter than {lo_n} (got {hi:?} vs {lo:?}); \
                 equal values mean the highlights are being clipped flat"
            );
        }
        // And the span has to be usable, not a rounding artefact.
        let span = i32::from(steps[3].1[0]) - i32::from(steps[0].1[0]);
        println!("diffuse-white -> 4000 cd/m² span: {span} codes");
        assert!(span >= 8, "highlight range collapsed to {span} codes");

        // A sunlit orange must stay orange: red clearly above green, not pinned
        // to it the way independent clamping did.
        let orange = shot([900.0, 400.0, 90.0]);
        println!("sunlit orange 900/400/90 -> {orange:?}");
        assert!(
            orange[0] > orange[1] && orange[1] > orange[2],
            "channel ordering must survive tone mapping, got {orange:?}"
        );

        // Saturation must not drain away as a colour gets brighter. Ordering
        // alone does not catch this — running the curve per channel kept
        // R>G>B while still washing a sunlit orange to pale yellow, because
        // the brightest channel compresses hardest and closes the gap to the
        // others. Compressing the peak and scaling the whole colour by that
        // one ratio holds chromaticity, so the same hue at rising brightness
        // holds its saturation instead of fading toward white.
        let sat = |p: [u8; 4]| -> f32 {
            let (mx, mn) = (
                f32::from(p[0].max(p[1]).max(p[2])),
                f32::from(p[0].min(p[1]).min(p[2])),
            );
            if mx <= 0.0 { 0.0 } else { (mx - mn) / mx }
        };
        let ramp: Vec<(u32, f32)> = [1_u32, 2, 4]
            .into_iter()
            .map(|m| {
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "small integer multipliers, exact in f32"
                )]
                let f = m as f32;
                (m, sat(shot([300.0 * f, 133.0 * f, 30.0 * f])))
            })
            .collect();
        for (m, s) in &ramp {
            println!("orange x{m}: saturation {s:.2}");
        }
        let (base, top) = (ramp[0].1, ramp[2].1);
        assert!(
            base > 0.6,
            "a saturated colour must stay saturated, got {base:.2}"
        );
        assert!(
            (base - top).abs() < 0.12,
            "saturation must survive a 4x brightness rise (got {base:.2} -> {top:.2}); \
             a large drop means the curve is bleaching colour toward white"
        );

        // Mid-tones sit below the knee and must pass through untouched, so an
        // ordinary SDR capture on an HDR output is not altered. 0.18x diffuse
        // white is scene mid grey; sRGB(0.18) ≈ 0.4613 -> 118.
        let mid = shot([203.0 * 0.18, 203.0 * 0.18, 203.0 * 0.18]);
        println!("mid grey (0.18x diffuse white) -> {mid:?} (want ~118)");
        assert!(
            (i32::from(mid[0]) - 118).abs() <= 2,
            "mid-tones must be left alone by the shoulder, got {mid:?}"
        );
    }

    /// Checks the fused single-pass program (the path a solo fullscreen game
    /// takes) against the PQ code the protocol's definition demands, and
    /// pins that the SDR program really does disagree.
    #[test]
    #[ignore = "GPU test; run manually with --ignored --nocapture"]
    fn scrgb_decodes_as_linear_80_nits() {
        /// Reference PQ OETF (ST.2084), 1.0 == 10000 cd/m².
        fn pq_oetf(l: f64) -> f64 {
            const M1: f64 = 0.1593017578125;
            const M2: f64 = 78.84375;
            const C1: f64 = 0.8359375;
            const C2: f64 = 18.8515625;
            const C3: f64 = 18.6875;
            let lp = l.max(0.0).powf(M1);
            ((C1 + C2 * lp) / (1.0 + C3 * lp)).powf(M2)
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/dri/renderD128")
            .expect("open render node");
        let fd = DrmDeviceFd::new(DeviceFd::from(OwnedFd::from(file)));
        let gbm = GbmDevice::new(fd).expect("GbmDevice");
        #[allow(unsafe_code, reason = "test-only surfaceless EGL, single thread")]
        // SAFETY: GbmDevice clone lives in the EGLDisplay for its lifetime.
        let display = unsafe { EGLDisplay::new(gbm.clone()) }.expect("EGLDisplay");
        let context = EGLContext::new(&display).expect("EGLContext");
        #[allow(unsafe_code, reason = "test-only single-thread renderer")]
        // SAFETY: used only from this test thread.
        let mut gles = unsafe { GlesRenderer::new(context) }.expect("GlesRenderer");

        let scrgb_shader = gles
            .compile_custom_texture_shader(SCRGB_TO_PQ_SHADER, &[])
            .expect("compile fused scRGB→PQ");
        let sdr_shader = gles
            .compile_custom_texture_shader(
                SDR_TO_PQ_SHADER,
                &[
                    UniformName::new("reference_white", UniformType::_1f),
                    UniformName::new("saturation", UniformType::_1f),
                ],
            )
            .expect("compile fused SDR→PQ");

        let size = Size::<i32, smithay::utils::Buffer>::from((64, 64));
        let phys = Size::<i32, Physical>::from((64, 64));
        let full = [Rectangle::<i32, Physical>::from_size(phys)];
        let src = Rectangle::<f64, smithay::utils::Buffer>::from_size(Size::from((64.0, 64.0)));
        let dst = Rectangle::<i32, Physical>::from_size(phys);

        // Opaque grey source textures. Grey keeps the BT.709→BT.2020 matrix a
        // no-op (its rows sum to 1), so the readback isolates the transfer
        // maths. import_memory (not create_buffer) so sampler filters are set.
        // Scoped so the closure's mutable borrow of `gles` ends here.
        let (white, quarter) = {
            let mut mk = |v: u8| -> GlesTexture {
                let data: Vec<u8> = [v, v, v, 255u8]
                    .iter()
                    .copied()
                    .cycle()
                    .take(64 * 64 * 4)
                    .collect();
                gles
                    .import_memory(&data, Fourcc::Abgr8888, size, false)
                    .expect("import source")
            };
            (mk(255), mk(64))
        };

        let mut run = |tex: &GlesTexture,
                       shader: &GlesTexProgram,
                       unis: &[Uniform<'_>]|
         -> [u8; 4] {
            let mut target: GlesTexture =
                gles.create_buffer(Fourcc::Abgr8888, size).expect("target");
            {
                let mut b = gles.bind(&mut target).expect("bind target");
                let mut f = gles.render(&mut b, phys, Transform::Normal).expect("render target");
                f.clear(Color32F::new(0.0, 0.0, 0.0, 1.0), &full).expect("clear");
                f.render_texture_from_to(
                    tex,
                    src,
                    dst,
                    &full,
                    &[],
                    Transform::Normal,
                    1.0,
                    Some(shader),
                    unis,
                )
                .expect("draw");
                let s = f.finish().expect("finish");
                drop(b);
                let _ = s.wait();
            }
            let region = Rectangle::<i32, smithay::utils::Buffer>::from_size(Size::from((64, 64)));
            let bound = gles.bind(&mut target).expect("rebind");
            let mapping = gles
                .copy_framebuffer(&bound, region, Fourcc::Abgr8888)
                .expect("copy_framebuffer");
            let bytes = gles.map_texture(&mapping).expect("map").to_vec();
            let c = (32 * 64 + 32) * 4;
            [bytes[c], bytes[c + 1], bytes[c + 2], bytes[c + 3]]
        };

        // scRGB 1.0 == 80 cd/m² and 0.25 == 20 cd/m², linearly — so the fused
        // program must emit exactly PQ(value / 125).
        for (tex, value, label) in [(&white, 1.0_f64, "1.0"), (&quarter, 64.0 / 255.0, "0.25")] {
            let got = run(tex, &scrgb_shader, &[]);
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "PQ code in [0,1] scaled to a byte"
            )]
            let want = (pq_oetf(value / 125.0) * 255.0).round() as i32;
            println!(
                "scRGB {label} -> {got:?} (want ~{want}, i.e. {:.1} cd/m²)",
                value * 80.0
            );
            assert!(
                (i32::from(got[0]) - want).abs() <= 2,
                "scRGB {label} must decode to PQ({value}/125) ≈ {want}, got {got:?}"
            );
        }

        // And pin the bug this fixes: the SDR program (gamma + 203 cd/m²
        // anchor) genuinely disagrees, so routing scRGB through it was not a
        // harmless approximation.
        let sdr_white = run(
            &white,
            &sdr_shader,
            &[
                Uniform::new("reference_white", 203.0_f32),
                Uniform::new("saturation", 1.0_f32),
            ],
        );
        let scrgb_white = run(&white, &scrgb_shader, &[]);
        println!("white: scRGB decode {scrgb_white:?} vs SDR decode {sdr_white:?}");
        assert!(
            i32::from(sdr_white[0]) - i32::from(scrgb_white[0]) > 10,
            "the SDR decode must visibly differ from the scRGB decode \
             (scRGB {scrgb_white:?}, SDR {sdr_white:?})"
        );
    }
}
