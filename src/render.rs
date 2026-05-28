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

use std::time::Instant;

use anyhow::{Context as _, Result};
use smithay::backend::allocator::Fourcc;
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
use smithay::backend::drm::{DrmDeviceFd, GbmBufferedSurface};
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::element::Kind;
use smithay::backend::renderer::element::surface::{
    WaylandSurfaceRenderElement, render_elements_from_surface_tree,
};
use smithay::backend::renderer::gles::{
    GlesFrame, GlesPixelProgram, GlesRenderer, Uniform, UniformName, UniformType,
};
use smithay::backend::renderer::utils::draw_render_elements;
use smithay::backend::renderer::{Bind as _, Color32F, Frame as _, Renderer as _};
use smithay::reexports::drm::control::crtc;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Physical, Point, Rectangle, Size, Transform};
use smithay::wayland::compositor::{
    SurfaceAttributes, TraversalAction, with_surface_tree_downward,
};
use tracing::{debug, info};

use crate::config::{BorderConfig, Fill, MonitorsConfig};
use crate::drm::DrmOutput;
use crate::layout::Placement;

/// Per-output metadata the Wayland frontend needs to advertise
/// `wl_output` and seed `wp_fractional_scale_manager_v1`. Mirrors
/// the renderer's internal `OutputRender` but exposes only the
/// fields the frontend cares about (no GBM surface handle).
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
    /// Origin used for the monotonic ms timestamp fed into
    /// `wl_callback.done` after each output is queued for scanout.
    /// Clients use this value to schedule their next frame's draw —
    /// the spec defines it as an unsigned 32-bit ms count expected
    /// to wrap freely.
    start: Instant,
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
            start: Instant::now(),
        })
    }

    /// Render every output's initial frame to prime its swapchain.
    /// Called once at startup before the event loop runs; thereafter
    /// each output's frames are driven by its own vblank events. No
    /// Wayland clients have connected yet at this point, so we pass
    /// an empty placement slice — only the wallpaper + cursor land.
    pub fn render_initial(&mut self) -> Result<()> {
        for idx in 0..self.outputs.len() {
            self.render_output(idx, &[], &[])
                .with_context(|| format!("initial render of output #{idx} failed"))?;
        }
        Ok(())
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
    ) -> Result<()> {
        let idx = self
            .outputs
            .iter()
            .position(|o| o.crtc == crtc)
            .with_context(|| format!("vblank for unknown CRTC {crtc:?}"))?;
        self.render_output(idx, placements, layers)
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
    ) -> Result<()> {
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
        let output = &self.outputs[idx];
        let mode_size = output.mode_size;
        let compositor_position = output.compositor_position;
        let compositor_size = output.compositor_size;
        let scale = output.scale;
        let output_name = output.name.clone();

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
                let surface_local_phys = Point::<i32, Physical>::from((
                    scale_i(p.cell_rect.loc.x + bw_comp - compositor_position.x, scale),
                    scale_i(p.cell_rect.loc.y + bw_comp - compositor_position.y, scale),
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

        let sync = {
            let mut target = self
                .gles
                .bind(&mut dmabuf)
                .with_context(|| format!("GlesRenderer::bind failed for {output_name}"))?;
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
            for (p, elements) in placements.iter().zip(grouped.iter()) {
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

            if cursor_in_bounds {
                let cursor_origin = Point::<i32, Physical>::from((
                    scale_f(cursor_local_x, scale),
                    scale_f(cursor_local_y, scale),
                ));
                draw_cursor(&mut frame, cursor_origin, scale)?;
            }

            frame.finish().context("Frame::finish failed")?
        };

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
        Ok(())
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

/// Draw the 24×24 white right-triangle cursor with its apex at
/// `local_origin` (top-left of the bbox = hotspot). Damage stripes
/// are anchored at `(0, row)` relative to `dst.loc` — see the long
/// note in milestone 2c about `Frame::draw_solid`'s damage-coordinate
/// semantics.
fn draw_cursor(
    frame: &mut GlesFrame<'_, '_>,
    local_origin: Point<i32, Physical>,
    scale: f64,
) -> Result<()> {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "CURSOR_SIZE is 24 and scale is bounded; product stays in i32"
    )]
    let size = ((f64::from(CURSOR_SIZE) * scale).round() as i32).max(1);
    let cursor_bbox = Rectangle::new(local_origin, Size::new(size, size));
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
