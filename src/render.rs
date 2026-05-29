//! GBM + EGL + GLES2 render pipeline with vblank-driven page-flipping
//! across multiple outputs.
//!
//! A single EGL context + GLES renderer + GBM allocator is shared by
//! every output on a given GPU. Each output has its own
//! `GbmBufferedSurface` (its own swapchain + page-flip cadence) and
//! is rendered independently when *its* CRTC reports vblank. Outputs
//! sit in a virtual layout — by default left-to-right at `y=0` in
//! connector enumeration order; Lua config will override per-output
//! positions in milestone 3c.
//!
//! Cursor coordinates live in absolute virtual-layout space. On each
//! per-output render we translate to output-local coordinates and
//! draw the cursor only when the hotspot falls within that output's
//! rectangle.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context as _, Result};
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
use smithay::backend::allocator::{Format, Fourcc};
use smithay::backend::drm::{DrmDeviceFd, DrmNode, GbmBufferedSurface};
use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay};
use smithay::backend::renderer::element::Kind;
use smithay::backend::renderer::element::surface::{
    WaylandSurfaceRenderElement, render_elements_from_surface_tree,
};
use smithay::backend::renderer::gles::{
    GlesFrame, GlesPixelProgram, GlesRenderer, GlesTarget, GlesTexture, Uniform, UniformName,
    UniformType,
};
use smithay::backend::renderer::utils::draw_render_elements;
use smithay::backend::renderer::{
    Bind as _, Blit as _, Color32F, ExportMem as _, Frame as _, ImportDma as _, ImportMem as _,
    Renderer as _, Texture as _, TextureFilter, TextureMapping as _,
};
use smithay::input::pointer::{CursorIcon, CursorImageStatus, CursorImageSurfaceData};
use smithay::reexports::drm::control::crtc;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{IsAlive as _, Physical, Point, Rectangle, Size, Transform};
use smithay::wayland::compositor::{
    SurfaceAttributes, TraversalAction, with_states, with_surface_tree_downward,
};
use smithay::wayland::shell::xdg::SurfaceCachedState;
use tracing::{debug, info, warn};

use crate::config::{BorderConfig, Fill, MonitorsConfig};
use crate::drm::DrmOutput;
use crate::layout::{FillMode, Placement};

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
/// during teardown → SIGSEGV). `flipped` on `Shm` only tells the writer
/// whether the read-back rows need reversing into the client buffer.
#[derive(Debug)]
pub enum CaptureOutcome {
    /// CPU read-back: a tight buffer (`width * 4` bytes/row). `flipped`
    /// means the rows are bottom-up (GL origin) and must be reversed
    /// when written to the client buffer.
    Shm {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        flipped: bool,
    },
    /// The client's dmabuf was filled directly by a GPU blit, already
    /// upright (the blit copies GL→GL coordinates, so memory-row 0 is
    /// the top of the image — no `y_invert` needed).
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

/// GLES2 fragment shader that paints the *frame* of a window —
/// the rounded-corner cutout (wallpaper) and the border ring
/// (border colour) — in one pass over the window's cell rect.
/// Runs after the surface is drawn, so the surface fills the
/// interior and this shader overpaints everything from the
/// border ring outward.
///
/// Region selection uses the signed-distance field of a rounded
/// rectangle. With `dist` = SDF distance from the rounded
/// boundary (positive outside, negative inside):
///
/// - `dist >  0`: outside the cell shape → paint wallpaper.
/// - `dist in (-border_width, 0]`: in the border ring → paint
///   border colour, which itself can be a vertical gradient.
/// - `dist <= -border_width`: interior → discard, keeping the
///   surface pixel that was drawn before.
///
/// Both transitions get a `smoothstep` ramp so the curve and
/// the border's inner edge are anti-aliased. Colours are sampled
/// from vertical gradients keyed off the fragment's *global* y
/// (output-space), so the active-border gradient stays
/// continuous between adjacent tiles instead of resetting per
/// cell, and similarly for the wallpaper cutout.
///
/// Uniforms: `size` and `alpha` come from smithay; the rest are
/// registered at compile time and re-set each frame.
const FRAME_SHADER: &str = r"
#extension GL_OES_standard_derivatives : enable

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

uniform vec2 size;
uniform float alpha;
uniform float radius;
uniform float border_width;
uniform vec3 grad_top;
uniform vec3 grad_bottom;
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

    // Screen-space AA half-width via the derivative of the SDF.
    // Falls back to 0.5 px when `fwidth` isn't supported or
    // returns 0 (e.g. on perfectly axis-aligned edges of an
    // un-rotated frame). This makes the AA ramp exactly one
    // *output* pixel wide regardless of fractional scale, so a
    // 4K display at scale 1.5 gets the same crisp 1-pixel ramp
    // as a 1080p panel at scale 1.
    float aa = max(fwidth(dist) * 0.5, 0.5);

    // Inner cutoff (cell interior, surface kept). The AA ramp
    // for the inner edge fits below this threshold.
    if (dist <= -border_width - aa) {
        discard;
    }

    float global_y = cell_origin_y + p.y;
    float t = clamp(global_y / max(output_height, 1.0), 0.0, 1.0);
    vec3 wallpaper_rgb = mix(grad_top, grad_bottom, t);
    vec3 border_rgb = mix(border_top, border_bottom, t);

    // Pick wallpaper outside the rounded boundary, border colour
    // inside, with a derivative-sized smoothstep across `dist = 0`.
    float outer_blend = smoothstep(-aa, aa, dist);
    vec3 color = mix(border_rgb, wallpaper_rgb, outer_blend);

    // Alpha: 1 from the outside through the border ring, fading
    // to 0 across the inner edge (`dist = -border_width`) so the
    // surface peeks through smoothly. With border_width = 0 the
    // two transitions coincide and the shader collapses to the
    // wallpaper-only mask case.
    float a = smoothstep(-border_width - aa, -border_width + aa, dist) * alpha;

    // The frame's blend mode is GL_ONE / GL_ONE_MINUS_SRC_ALPHA
    // (premultiplied source-over), so RGB has to be multiplied
    // by alpha here. Without this, partially-transparent edge
    // fragments come out over-bright and the smoothstep AA on
    // the curve and the border's inner edge looks like a hard
    // halo instead of a smooth ramp.
    gl_FragColor = vec4(color * a, a);
}
";

/// Renderer for every connected output on a single GPU.
pub struct Renderer {
    /// Shared GLES2 renderer; owns the EGL context.
    gles: GlesRenderer,
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
    /// Wallpaper drawn under the cursor on every output.
    wallpaper: Fill,
    /// Window border width + active / inactive fills.
    border: BorderConfig,
    /// Custom GLES pixel shader used to mask rounded corners.
    /// `Arc`-backed so it's cheap to clone out before borrowing
    /// the renderer for the frame.
    frame_shader: GlesPixelProgram,
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
struct OutputRender {
    name: String,
    crtc: crtc::Handle,
    surface: GbmBufferedSurface<GbmAllocator<DrmDeviceFd>, ()>,
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

impl Renderer {
    /// Build the shared EGL/GLES context plus one `GbmBufferedSurface`
    /// per output. Outputs are placed left-to-right at `y=0` in the
    /// order the DRM layer enumerated them; the cursor is initialised
    /// at the centre of the first output so it's immediately visible.
    #[allow(
        clippy::too_many_lines,
        reason = "linear initialisation sequence (GBM device, EGL display, EGL context, GLES renderer, custom shader, GBM allocator, per-output GbmBufferedSurfaces). Splitting it forces threading several mid-construction values through extra functions for no real win."
    )]
    pub fn new(
        drm_fd: DrmDeviceFd,
        drm_outputs: Vec<DrmOutput>,
        wallpaper: Fill,
        border: BorderConfig,
        monitors: &MonitorsConfig,
    ) -> Result<Self> {
        info!("phase: opening GBM device");
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

        info!("phase: compiling window-frame pixel shader");
        let frame_shader = gles
            .compile_custom_pixel_shader(
                FRAME_SHADER,
                &[
                    UniformName::new("radius", UniformType::_1f),
                    UniformName::new("border_width", UniformType::_1f),
                    UniformName::new("grad_top", UniformType::_3f),
                    UniformName::new("grad_bottom", UniformType::_3f),
                    UniformName::new("border_top", UniformType::_3f),
                    UniformName::new("border_bottom", UniformType::_3f),
                    UniformName::new("output_height", UniformType::_1f),
                    UniformName::new("cell_origin_y", UniformType::_1f),
                ],
            )
            .context("window-frame shader compile failed")?;

        info!("phase: creating GBM allocator");
        let allocator = GbmAllocator::new(
            gbm_device,
            GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT,
        );

        info!("phase: building per-output GBM buffered surfaces");
        let renderer_formats = gles.egl_context().dmabuf_render_formats().clone();
        let mut outputs = Vec::with_capacity(drm_outputs.len());
        // Running compositor-x cursor for outputs the user didn't
        // pin to a specific position. Configured positions can
        // overlap with the auto cursor; that's the user's call.
        let mut auto_x: i32 = 0;

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
            #[allow(
                clippy::cast_possible_truncation,
                reason = "mode pixels are u16-bounded; divided by scale > 0 fits in i32 trivially"
            )]
            let compositor_size = Size::<i32, Physical>::new(
                (f64::from(mode_size.w) / scale).round() as i32,
                (f64::from(mode_size.h) / scale).round() as i32,
            );
            let compositor_position = match output_cfg.and_then(|c| c.position) {
                Some((x, y)) => Point::<i32, Physical>::from((x, y)),
                None => Point::<i32, Physical>::from((auto_x, 0)),
            };
            if output_cfg.and_then(|c| c.position).is_none() {
                auto_x = auto_x.saturating_add(compositor_size.w);
            }

            let surface = GbmBufferedSurface::new(
                drm_output.surface,
                allocator.clone(),
                &[Fourcc::Xrgb8888],
                renderer_formats.clone(),
            )
            .with_context(|| {
                format!(
                    "GbmBufferedSurface::new failed for {} (no compatible scanout format?)",
                    drm_output.name
                )
            })?;

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
                "output swapchain ready"
            );

            outputs.push(OutputRender {
                name: drm_output.name,
                crtc: drm_output.crtc,
                surface,
                mode_size,
                refresh_mhz,
                compositor_position,
                compositor_size,
                scale,
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
            outputs,
            primary_idx,
            layout_bounds,
            cursor_x,
            cursor_y,
            wallpaper,
            border,
            frame_shader,
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

    /// Render every output's initial frame to prime its swapchain.
    /// Called once at startup before the event loop runs; thereafter
    /// each output's frames are driven by its own vblank events. No
    /// Wayland clients have connected yet at this point, so we pass
    /// an empty placement slice — only the wallpaper + cursor land.
    pub fn render_initial(&mut self) -> Result<()> {
        for idx in 0..self.outputs.len() {
            self.render_output(idx, &[], &[], &[], false, &[])
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
    pub fn render_for_crtc(
        &mut self,
        crtc: crtc::Handle,
        placements: &[Placement],
        layers: &[LayerPlacement],
        popups: &[PopupPlacement],
        hide_cursor: bool,
        captures: &[CaptureSpec],
    ) -> Result<Vec<CaptureOutcome>> {
        let idx = self
            .outputs
            .iter()
            .position(|o| o.crtc == crtc)
            .with_context(|| format!("vblank for unknown CRTC {crtc:?}"))?;
        self.render_output(idx, placements, layers, popups, hide_cursor, captures)
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

    /// Connector name of the primary output. Used by the layer-shell
    /// reflow to attribute exclusive zones to the primary by name.
    pub fn primary_output_name(&self) -> &str {
        &self.outputs[self.primary_idx].name
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

    /// GPU buffer (dmabuf) formats this renderer can import as
    /// textures — advertised via `zwp_linux_dmabuf_v1` so clients
    /// (and Xwayland via xwayland-satellite) can hand us
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
        self.outputs[self.primary_idx].scale
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
    fn render_output(
        &mut self,
        idx: usize,
        placements: &[Placement],
        layers: &[LayerPlacement],
        popups: &[PopupPlacement],
        hide_cursor: bool,
        captures: &[CaptureSpec],
    ) -> Result<Vec<CaptureOutcome>> {
        // Pull everything we need before the mutable borrows on
        // `self.outputs[idx].surface` / `self.gles` kick in. All
        // *_phys helpers below take pre-scaled physical pixel
        // values; this function is the one place compositor →
        // physical conversion happens.
        let cursor_abs_x = self.cursor_x;
        let cursor_abs_y = self.cursor_y;
        let wallpaper = self.wallpaper.clone();
        let border = self.border.clone();
        let frame_shader = self.frame_shader.clone();
        let cursor_size = self.cursor_size;
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
        let output = &self.outputs[idx];
        let mode_size = output.mode_size;
        let compositor_position = output.compositor_position;
        let compositor_size = output.compositor_size;
        let scale = output.scale;
        let output_name = output.name.clone();
        // Frozen backdrop for this output (freeze-mode screenshot). Cheap
        // Arc-backed clone out before the `self.gles` frame borrow.
        let freeze_texture = self.freeze_textures.get(&output_name).cloned();

        // No-op on the first call (no pending fb), the ack of the
        // previous frame's flip thereafter.
        let _ = self.outputs[idx]
            .surface
            .frame_submitted()
            .with_context(|| format!("frame_submitted failed for {output_name}"))?;

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
        let grouped: Vec<Vec<WaylandSurfaceRenderElement<GlesRenderer>>> = placements
            .iter()
            .map(|p| {
                // CSD clients pad their buffer with an invisible
                // drop-shadow margin and report the real window rect
                // via xdg_surface.set_window_geometry. Shift the
                // buffer up-left by that margin so the *visible*
                // content (not the buffer's padded corner) lands at
                // the cell origin; the shadow then falls outside the
                // cell instead of pushing content down-right.
                // Maximized/fullscreen windows have no border and fill
                // their output flush at the cell origin — no border
                // inset and no CSD shadow offset. A spec-compliant
                // client zeroes its window geometry on the transition;
                // gating here also defends against one that doesn't.
                let (geo_x, geo_y) = if p.fill == FillMode::Normal {
                    window_geometry_offset(&p.surface)
                } else {
                    (0, 0)
                };
                let bw_p = if p.fill == FillMode::Normal {
                    bw_comp
                } else {
                    0
                };
                let surface_local_phys = Point::<i32, Physical>::from((
                    scale_i(
                        p.cell_rect.loc.x + bw_p - compositor_position.x - geo_x,
                        scale,
                    ),
                    scale_i(
                        p.cell_rect.loc.y + bw_p - compositor_position.y - geo_y,
                        scale,
                    ),
                ));
                render_elements_from_surface_tree(
                    &mut self.gles,
                    &p.surface,
                    surface_local_phys,
                    scale,
                    1.0_f32,
                    Kind::Unspecified,
                )
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

        let mut target = self
            .gles
            .bind(&mut dmabuf)
            .with_context(|| format!("GlesRenderer::bind failed for {output_name}"))?;
        let sync = {
            let mut frame = self
                .gles
                .render(&mut target, mode_size, Transform::Normal)
                .with_context(|| format!("GlesRenderer::render failed for {output_name}"))?;

            // Background fills first, then for each placement
            // (bottom-up) the border frame followed by the surface,
            // so that floating-window borders and surfaces end up
            // visually above the tiled cells they overlap. Each
            // window's `draw_render_elements` call carries just
            // that window's elements — smithay's opaque-region
            // culling can't accidentally skip floats behind
            // earlier tiles when there's only ever one element in
            // the slice.
            draw_fill(&mut frame, &wallpaper, mode_size, mode_size)?;

            let full_damage = [Rectangle::<i32, Physical>::from_size(mode_size)];

            // Layer-shell render order, per wlr-layer-shell spec:
            //   wallpaper → Background → Bottom → windows → Top → Overlay → cursor.
            // Background + Bottom go between wallpaper and tiles
            // so panels and wallpaper-like surfaces sit behind
            // application windows; Top + Overlay go on top of
            // windows so notifications, launchers, OSDs are
            // visible. We draw each layer surface with its own
            // `draw_render_elements` call (single-element slice)
            // for the same opaque-region reason the window loop
            // uses below.
            for (bucket, elements) in &layer_groups {
                if matches!(bucket, LayerBucket::Background | LayerBucket::Bottom) {
                    draw_render_elements::<GlesRenderer, _, _>(
                        &mut frame,
                        scale,
                        elements,
                        &full_damage,
                    )
                    .context("draw_render_elements (layer bg/bottom) failed")?;
                }
            }
            let radius_comp = border.rounded_corners.max(0);
            // Normal (tiled/floating) windows: surface, then the border
            // ring + rounded-corner cutout painted over it.
            for (p, elements) in placements
                .iter()
                .zip(grouped.iter())
                .filter(|(p, _)| p.fill == FillMode::Normal)
            {
                let cell_local_phys = Rectangle::<i32, Physical>::new(
                    Point::new(
                        scale_i(p.cell_rect.loc.x - compositor_position.x, scale),
                        scale_i(p.cell_rect.loc.y - compositor_position.y, scale),
                    ),
                    Size::new(
                        scale_i(p.cell_rect.size.w, scale),
                        scale_i(p.cell_rect.size.h, scale),
                    ),
                );
                // Surface first; the frame shader will overpaint
                // the border ring and corner cutout on top.
                draw_render_elements::<GlesRenderer, _, _>(
                    &mut frame,
                    scale,
                    elements,
                    &full_damage,
                )
                .context("draw_render_elements failed")?;

                // Frame shader runs whenever there's *anything*
                // to paint over the surface — a border ring, a
                // rounded-corner cutout, or both. With both at 0
                // it would be a no-op so we skip the GL call.
                if bw_comp > 0 || radius_comp > 0 {
                    let fill = if p.focused {
                        &border.active
                    } else {
                        &border.inactive
                    };
                    draw_window_frame(
                        &mut frame,
                        &frame_shader,
                        cell_local_phys,
                        scale_i(bw_comp, scale),
                        scale_i(radius_comp, scale),
                        fill,
                        &wallpaper,
                        mode_size,
                    )?;
                }
            }

            // Maximized windows: borderless, no corners, drawn above
            // normal windows but below Top/Overlay panels (which stay
            // visible). Fullscreen windows are drawn later, above the
            // panels too.
            for (_p, elements) in placements
                .iter()
                .zip(grouped.iter())
                .filter(|(p, _)| p.fill == FillMode::Maximized)
            {
                draw_render_elements::<GlesRenderer, _, _>(
                    &mut frame,
                    scale,
                    elements,
                    &full_damage,
                )
                .context("draw_render_elements (maximized) failed")?;
            }

            // Top + Overlay layer surfaces go above windows but
            // below the cursor, matching common compositor
            // behaviour (rofi above kitty, status bar above
            // everything but the cursor).
            for (bucket, elements) in &layer_groups {
                if matches!(bucket, LayerBucket::Top | LayerBucket::Overlay) {
                    draw_render_elements::<GlesRenderer, _, _>(
                        &mut frame,
                        scale,
                        elements,
                        &full_damage,
                    )
                    .context("draw_render_elements (layer top/overlay) failed")?;
                }
            }

            // Fullscreen windows: borderless and above everything,
            // including Top/Overlay panels (a fullscreen game/video
            // covers the bar), but still below popups and the cursor.
            for (_p, elements) in placements
                .iter()
                .zip(grouped.iter())
                .filter(|(p, _)| p.fill == FillMode::Fullscreen)
            {
                draw_render_elements::<GlesRenderer, _, _>(
                    &mut frame,
                    scale,
                    elements,
                    &full_damage,
                )
                .context("draw_render_elements (fullscreen) failed")?;
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
                        // An empty element list (surface with no committed
                        // buffer) is the client's way of hiding the
                        // cursor — draw nothing in that case.
                        if !cursor_surface_elements.is_empty() {
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
                        )?;
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
        let capture_results: Vec<CaptureOutcome> = captures
            .iter()
            .map(|spec| match &spec.target {
                CaptureTarget::Shm => {
                    capture_shm(&mut self.gles, &target, spec, mode_size.h, &output_name)
                }
                CaptureTarget::Dmabuf(client) => capture_dmabuf(
                    &mut self.gles,
                    &target,
                    client,
                    spec,
                    mode_size.h,
                    &output_name,
                ),
            })
            .collect();
        drop(target);

        self.outputs[idx]
            .surface
            .queue_buffer(Some(sync), None, ())
            .with_context(|| format!("queue_buffer failed for {output_name}"))?;
        debug!(output = %output_name, "frame queued for scanout");

        // Fire wl_callback.done on every surface we rendered. The
        // callback queue is drained per surface, so calling this
        // again from a second output's render is a harmless no-op
        // (which is what we want — one done() per frame, not one
        // per output).
        #[allow(
            clippy::cast_possible_truncation,
            reason = "wl_callback.done takes u32 ms which the spec expects to wrap freely (~50d period)"
        )]
        let elapsed_ms = self.start.elapsed().as_millis() as u32;
        for p in placements {
            send_frame_callbacks(&p.surface, elapsed_ms);
        }
        for l in layers {
            send_frame_callbacks(&l.surface, elapsed_ms);
        }
        for p in popups {
            send_frame_callbacks(&p.surface, elapsed_ms);
        }
        Ok(capture_results)
    }
}

/// Convert a screencopy region (top-left origin) to the bottom-left
/// origin both `glReadPixels` and `glBlitFramebuffer` read from, so a
/// partial region (`grim -g`) reads the band the client asked for. A
/// full-output capture is unchanged (loc.y 0, full height -> 0).
fn region_gl(spec: &CaptureSpec, fb_height: i32) -> Rectangle<i32, Physical> {
    let gl_y = fb_height - spec.region.loc.y - spec.region.size.h;
    Rectangle::new(
        (spec.region.loc.x, gl_y).into(),
        (spec.region.size.w, spec.region.size.h).into(),
    )
}

/// CPU read-back: copy `spec.region` of `target` into a tight buffer.
fn capture_shm(
    gles: &mut GlesRenderer,
    target: &GlesTarget<'_>,
    spec: &CaptureSpec,
    fb_height: i32,
    output_name: &str,
) -> CaptureOutcome {
    let gl = region_gl(spec, fb_height);
    let region = Rectangle::<i32, smithay::utils::Buffer>::new(
        (gl.loc.x, gl.loc.y).into(),
        (gl.size.w, gl.size.h).into(),
    );
    let mapping = match gles.copy_framebuffer(target, region, spec.fourcc) {
        Ok(mapping) => mapping,
        Err(err) => {
            warn!(error = %err, output = %output_name, "screencopy: copy_framebuffer failed");
            return CaptureOutcome::Failed;
        }
    };
    // Rows come back bottom-up (GL origin); `write_to_shm` reverses them
    // into the client buffer so the delivered frame is upright.
    let flipped = mapping.flipped();
    let (width, height) = (mapping.width(), mapping.height());
    match gles.map_texture(&mapping) {
        Ok(bytes) => CaptureOutcome::Shm {
            bytes: bytes.to_vec(),
            width,
            height,
            flipped,
        },
        Err(err) => {
            warn!(error = %err, output = %output_name, "screencopy: map_texture failed");
            CaptureOutcome::Failed
        }
    }
}

/// Zero-copy GPU path: bind the client's dmabuf as a framebuffer and
/// blit `spec.region` of the composited output into it. The blit copies
/// GL→GL coordinates; our output framebuffer and the client's dmabuf
/// both map GL `(0,0)` to the bottom of the image, so the result lands
/// upright in memory (row 0 = top) — no `y_invert` flag needed.
fn capture_dmabuf(
    gles: &mut GlesRenderer,
    target: &GlesTarget<'_>,
    client: &Dmabuf,
    spec: &CaptureSpec,
    fb_height: i32,
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
    let src = region_gl(spec, fb_height);
    let dst_rect = Rectangle::<i32, Physical>::from_size(spec.region.size);
    match gles.blit(target, &mut dst, src, dst_rect, TextureFilter::Linear) {
        Ok(()) => CaptureOutcome::Dmabuf,
        Err(err) => {
            warn!(error = %err, output = %output_name, "screencopy: blit to client dmabuf failed");
            CaptureOutcome::Failed
        }
    }
}

/// Paint `fill` inside the output-local rect `rect`. `Solid` is
/// one `draw_solid` call. `VerticalGradient` walks 256 horizontal
/// stripes spanning the full output height (so the gradient stays
/// continuous with the wallpaper even when only the border edges
/// are being painted); each stripe is clipped to `rect` and
/// skipped if it lies entirely outside, so border edges that
/// only intersect a few stripes don't pay for the rest.
fn draw_fill(
    frame: &mut GlesFrame<'_, '_>,
    fill: &Fill,
    rect: Size<i32, Physical>,
    output_size: Size<i32, Physical>,
) -> Result<()> {
    draw_fill_rect(
        frame,
        fill,
        Rectangle::<i32, Physical>::from_size(rect),
        output_size,
    )
}

fn draw_fill_rect(
    frame: &mut GlesFrame<'_, '_>,
    fill: &Fill,
    rect: Rectangle<i32, Physical>,
    output_size: Size<i32, Physical>,
) -> Result<()> {
    if rect.size.w <= 0 || rect.size.h <= 0 {
        return Ok(());
    }
    match fill {
        Fill::Solid(rgb) => {
            let damage = [Rectangle::from_size(rect.size)];
            frame
                .draw_solid(rect, &damage, Color32F::new(rgb[0], rgb[1], rgb[2], 1.0))
                .context("Frame::draw_solid (fill solid) failed")?;
        }
        Fill::VerticalGradient { top, bottom } => {
            const STRIPE_COUNT: i32 = 256;
            let height = output_size.h.max(1);
            let rect_y_end = rect.loc.y + rect.size.h;
            for stripe in 0u8..=u8::MAX {
                let t = f32::from(stripe) / 255.0;
                let color = Color32F::new(
                    top[0].mul_add(1.0 - t, bottom[0] * t),
                    top[1].mul_add(1.0 - t, bottom[1] * t),
                    top[2].mul_add(1.0 - t, bottom[2] * t),
                    1.0,
                );

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

/// Mask a window's corners into a rounded shape by running the
/// custom GLES pixel shader over the cell rect. The shader writes
/// alpha = 1 with the wallpaper colour outside the rounded
/// boundary and discards / writes alpha = 0 inside, so the
/// already-drawn border + surface remain visible inside and the
/// wallpaper colour appears in the corner cutouts. The
/// `smoothstep` in the shader gives ~1 px of anti-aliasing along
/// the curve.
///
/// Per-cell effective radius is clamped to half the cell's
/// smaller dimension so two corners can never overlap on a tiny
/// tile.
///
/// Trade-off: a floating window over a tile shows wallpaper (not
/// the tile) at the rounded corners — the shader paints the mask
/// colour rather than sampling whatever was underneath the
/// surface. True transparency at corners needs per-window
/// offscreen rendering, which is later polish.
#[allow(
    clippy::too_many_arguments,
    reason = "all 8 args are first-class inputs to one pixel-shader call; bundling them into a struct would just spread the call site across more lines for no real readability win"
)]
fn draw_window_frame(
    frame: &mut GlesFrame<'_, '_>,
    shader: &GlesPixelProgram,
    cell_rect_phys: Rectangle<i32, Physical>,
    border_width_phys: i32,
    radius_phys: i32,
    border_fill: &Fill,
    wallpaper: &Fill,
    output_size: Size<i32, Physical>,
) -> Result<()> {
    let max_half = (cell_rect_phys.size.w / 2).min(cell_rect_phys.size.h / 2);
    let radius = radius_phys.min(max_half).max(0);
    // Don't let the border eat the entire cell — leave at least
    // 1 px of surface visible. For tiny tiles this clamps the
    // configured border down so the shader still has an interior.
    let border = border_width_phys.min(max_half - 1).max(0);
    if radius <= 0 && border <= 0 {
        return Ok(());
    }
    let (grad_top, grad_bottom) = match wallpaper {
        Fill::Solid(rgb) => (*rgb, *rgb),
        Fill::VerticalGradient { top, bottom } => (*top, *bottom),
    };
    let (border_top, border_bottom) = match border_fill {
        Fill::Solid(rgb) => (*rgb, *rgb),
        Fill::VerticalGradient { top, bottom } => (*top, *bottom),
    };
    #[allow(
        clippy::cast_precision_loss,
        reason = "radius, border, and cell origin are bounded by i32 cell / output sizes; fit f32 exactly for any realistic value"
    )]
    let uniforms = [
        Uniform::new("radius", radius as f32),
        Uniform::new("border_width", border as f32),
        Uniform::new("grad_top", grad_top),
        Uniform::new("grad_bottom", grad_bottom),
        Uniform::new("border_top", border_top),
        Uniform::new("border_bottom", border_bottom),
        Uniform::new("output_height", output_size.h as f32),
        Uniform::new("cell_origin_y", cell_rect_phys.loc.y as f32),
    ];
    // The shader doesn't sample any texture but the API still
    // wants a `src` + `size`; passing the cell rect makes the
    // built-in `size` uniform come out as the cell's pixel size.
    let src = Rectangle::<f64, smithay::utils::Buffer>::from_size(smithay::utils::Size::<
        f64,
        smithay::utils::Buffer,
    >::from((
        f64::from(cell_rect_phys.size.w),
        f64::from(cell_rect_phys.size.h),
    )));
    let size = smithay::utils::Size::<i32, smithay::utils::Buffer>::from((
        cell_rect_phys.size.w,
        cell_rect_phys.size.h,
    ));
    frame
        .render_pixel_shader_to(shader, src, cell_rect_phys, size, None, 1.0, &uniforms)
        .context("render_pixel_shader_to (window frame) failed")?;
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
fn draw_screenshot_overlay(
    frame: &mut GlesFrame<'_, '_>,
    overlay: &ScreenshotOverlay,
    compositor_position: Point<i32, Physical>,
    mode_size: Size<i32, Physical>,
    scale: f64,
) -> Result<()> {
    const DIM: Color32F = Color32F::new(0.0, 0.0, 0.0, 0.45);
    const OUTLINE: Color32F = Color32F::new(0.25, 0.62, 1.0, 1.0);
    let (mode_w, mode_h) = (mode_size.w, mode_size.h);

    let solid = |frame: &mut GlesFrame<'_, '_>, x: i32, y: i32, w: i32, h: i32, color: Color32F| {
        if w <= 0 || h <= 0 {
            return Ok(());
        }
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
fn window_geometry_offset(surface: &WlSurface) -> (i32, i32) {
    with_states(surface, |states| {
        states
            .cached_state
            .get::<SurfaceCachedState>()
            .current()
            .geometry
            .map_or((0, 0), |g| (g.loc.x, g.loc.y))
    })
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
fn draw_cursor(
    frame: &mut GlesFrame<'_, '_>,
    sprite: Option<&CursorSprite>,
    cursor_size: i32,
    hotspot: Point<i32, Physical>,
    scale: f64,
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
    frame
        .draw_solid(
            cursor_bbox,
            &cursor_damage,
            Color32F::new(1.0, 1.0, 1.0, 1.0),
        )
        .context("Frame::draw_solid (cursor) failed")?;
    Ok(())
}
