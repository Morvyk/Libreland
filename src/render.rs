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
use smithay::backend::renderer::gles::{GlesFrame, GlesRenderer};
use smithay::backend::renderer::utils::draw_render_elements;
use smithay::backend::renderer::{Bind as _, Color32F, Frame as _, Renderer as _};
use smithay::reexports::drm::control::crtc;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Physical, Point, Rectangle, Size, Transform};
use smithay::wayland::compositor::{
    SurfaceAttributes, TraversalAction, with_surface_tree_downward,
};
use tracing::{debug, info};

use crate::config::Wallpaper;
use crate::drm::DrmOutput;

/// Side length of the cursor sprite in physical pixels. The sprite
/// is a right-triangle with apex at the hotspot, so this is also
/// the bounding-box width and height.
const CURSOR_SIZE: i32 = 24;

/// Renderer for every connected output on a single GPU.
pub struct Renderer {
    /// Shared GLES2 renderer; owns the EGL context.
    gles: GlesRenderer,
    /// One swapchain + framebuffer chain per output.
    outputs: Vec<OutputRender>,
    /// Bounding box of the virtual layout, anchored at `(0, 0)`.
    /// Used to clamp the cursor.
    layout_bounds: Size<i32, Physical>,
    /// Cursor hotspot in **absolute** virtual-layout coordinates.
    /// Each per-output render translates to local coords by
    /// subtracting that output's `position`.
    cursor_x: f64,
    cursor_y: f64,
    /// Wallpaper drawn under the cursor on every output.
    wallpaper: Wallpaper,
    /// Origin used for the monotonic ms timestamp fed into
    /// `wl_callback.done` after each output is queued for scanout.
    /// Clients use this value to schedule their next frame's draw —
    /// the spec defines it as an unsigned 32-bit ms count expected
    /// to wrap freely.
    start: Instant,
}

/// One output's render state: swapchain, dimensions, and position in
/// the virtual layout.
struct OutputRender {
    name: String,
    crtc: crtc::Handle,
    surface: GbmBufferedSurface<GbmAllocator<DrmDeviceFd>, ()>,
    /// Output dimensions in physical pixels (from its DRM mode).
    size: Size<i32, Physical>,
    /// Top-left of this output in absolute virtual-layout coords.
    position: Point<i32, Physical>,
}

impl Renderer {
    /// Build the shared EGL/GLES context plus one `GbmBufferedSurface`
    /// per output. Outputs are placed left-to-right at `y=0` in the
    /// order the DRM layer enumerated them; the cursor is initialised
    /// at the centre of the first output so it's immediately visible.
    pub fn new(
        drm_fd: DrmDeviceFd,
        drm_outputs: Vec<DrmOutput>,
        wallpaper: Wallpaper,
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
        let gles = unsafe { GlesRenderer::new(egl_context) }.context("GlesRenderer::new failed")?;
        info!("GLES renderer created");

        info!("phase: creating GBM allocator");
        let allocator = GbmAllocator::new(
            gbm_device,
            GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT,
        );

        info!("phase: building per-output GBM buffered surfaces");
        let renderer_formats = gles.egl_context().dmabuf_render_formats().clone();
        let mut outputs = Vec::with_capacity(drm_outputs.len());
        // Running cursor for the left-to-right layout. `max_height`
        // captures the tallest output so the layout bounding box is
        // (sum_widths, max_height).
        let mut layout_x: i32 = 0;
        let mut layout_max_h: i32 = 0;

        for drm_output in drm_outputs {
            let (mode_w, mode_h) = drm_output.mode.size();
            let size = Size::<i32, Physical>::new(i32::from(mode_w), i32::from(mode_h));
            let position = Point::<i32, Physical>::from((layout_x, 0));

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
                pos_x = layout_x,
                pos_y = 0,
                width = size.w,
                height = size.h,
                "output swapchain ready"
            );

            outputs.push(OutputRender {
                name: drm_output.name,
                crtc: drm_output.crtc,
                surface,
                size,
                position,
            });

            layout_x = layout_x.saturating_add(size.w);
            layout_max_h = layout_max_h.max(size.h);
        }

        let layout_bounds = Size::<i32, Physical>::new(layout_x, layout_max_h);
        info!(
            outputs = outputs.len(),
            layout_w = layout_bounds.w,
            layout_h = layout_bounds.h,
            "render layout finalised"
        );

        // Cursor starts at the centre of the first output. `unwrap`
        // is safe because `drm::open_display` errors out if there are
        // zero outputs.
        let first = outputs
            .first()
            .expect("Renderer::new given empty outputs vec");
        let cursor_x = f64::from(first.position.x) + f64::from(first.size.w) / 2.0;
        let cursor_y = f64::from(first.position.y) + f64::from(first.size.h) / 2.0;

        Ok(Self {
            gles,
            outputs,
            layout_bounds,
            cursor_x,
            cursor_y,
            wallpaper,
            start: Instant::now(),
        })
    }

    /// Render every output's initial frame to prime its swapchain.
    /// Called once at startup before the event loop runs; thereafter
    /// each output's frames are driven by its own vblank events. No
    /// Wayland clients have connected yet at this point, so we pass
    /// an empty surface slice — only the wallpaper + cursor land.
    pub fn render_initial(&mut self) -> Result<()> {
        for idx in 0..self.outputs.len() {
            self.render_output(idx, &[])
                .with_context(|| format!("initial render of output #{idx} failed"))?;
        }
        Ok(())
    }

    /// Render the output driven by `crtc`, in response to its vblank.
    /// `surfaces` is the snapshot of every live `xdg_toplevel`'s
    /// `wl_surface` that the caller wants composited this frame; for
    /// 4b every entry is pinned to the absolute origin (0, 0) of the
    /// virtual layout, so on a multi-output setup the same window
    /// only appears on whichever output covers that point. Real
    /// per-window placement is window-management (4d).
    pub fn render_for_crtc(&mut self, crtc: crtc::Handle, surfaces: &[WlSurface]) -> Result<()> {
        let idx = self
            .outputs
            .iter()
            .position(|o| o.crtc == crtc)
            .with_context(|| format!("vblank for unknown CRTC {crtc:?}"))?;
        self.render_output(idx, surfaces)
    }

    /// Advance the cursor hotspot by libinput-reported deltas, clamped
    /// to the virtual layout's bounding box.
    pub fn on_pointer_motion(&mut self, dx: f64, dy: f64) {
        let max_x = f64::from(self.layout_bounds.w);
        let max_y = f64::from(self.layout_bounds.h);
        self.cursor_x = (self.cursor_x + dx).clamp(0.0, max_x);
        self.cursor_y = (self.cursor_y + dy).clamp(0.0, max_y);
    }

    /// Render one output's frame: wallpaper, then every client
    /// surface positioned in this output's local space, then the
    /// cursor sprite on top if its hotspot falls in this output.
    /// Sends `wl_callback.done` on each surface after the buffer is
    /// queued so clients know they can draw the next frame.
    fn render_output(&mut self, idx: usize, surfaces: &[WlSurface]) -> Result<()> {
        // Pull everything we need before the mutable borrows on
        // `self.outputs[idx].surface` / `self.gles` kick in.
        let cursor_x = self.cursor_x;
        let cursor_y = self.cursor_y;
        let wallpaper = self.wallpaper.clone();
        let output_size = self.outputs[idx].size;
        let output_pos = self.outputs[idx].position;
        let output_name = self.outputs[idx].name.clone();

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

        // Cursor in this output's local coord space (subtract the
        // output's origin). Bounds check on the hotspot — if the
        // hotspot is off this output, don't draw the cursor here at
        // all. Sprite may still partially overflow the output's
        // bottom-right edge; that's clipped by GLES viewport.
        let cursor_local_x = cursor_x - f64::from(output_pos.x);
        let cursor_local_y = cursor_y - f64::from(output_pos.y);
        let cursor_in_bounds = cursor_local_x >= 0.0
            && cursor_local_y >= 0.0
            && cursor_local_x < f64::from(output_size.w)
            && cursor_local_y < f64::from(output_size.h);

        // Build client-surface render elements *before* binding the
        // dmabuf. `render_elements_from_surface_tree` uses the
        // renderer to import each surface's buffer as a GLES texture
        // (via the `ImportAll` trait GlesRenderer impls); that has
        // to happen while no `Frame` is alive. The resulting Vec
        // owns its `TextureId`s, so it's free to outlive the
        // renderer borrow and be drawn during the frame below.
        //
        // 4b pins every surface at (0, 0) absolute; on this output
        // that means (-output_pos.x, -output_pos.y) locally. Outputs
        // not covering (0, 0) just get a fully-clipped surface —
        // the GLES viewport handles it. Real placement is 4d.
        let local_origin = Point::<i32, Physical>::from((-output_pos.x, -output_pos.y));
        let surface_elements: Vec<WaylandSurfaceRenderElement<GlesRenderer>> = surfaces
            .iter()
            .flat_map(|surface| {
                render_elements_from_surface_tree(
                    &mut self.gles,
                    surface,
                    local_origin,
                    1.0_f64,
                    1.0_f32,
                    Kind::Unspecified,
                )
            })
            .collect();

        let sync = {
            let mut target = self
                .gles
                .bind(&mut dmabuf)
                .with_context(|| format!("GlesRenderer::bind failed for {output_name}"))?;
            let mut frame = self
                .gles
                .render(&mut target, output_size, Transform::Normal)
                .with_context(|| format!("GlesRenderer::render failed for {output_name}"))?;

            draw_wallpaper(&mut frame, &wallpaper, output_size)?;

            // Full-output damage for 4b — we redraw everything every
            // vblank. Per-element damage tracking is a later
            // optimisation (matters more for partial-occlusion
            // perf and battery, neither of which we care about yet).
            let full_damage = [Rectangle::<i32, Physical>::from_size(output_size)];
            draw_render_elements::<GlesRenderer, _, _>(
                &mut frame,
                1.0_f64,
                &surface_elements,
                &full_damage,
            )
            .context("draw_render_elements failed")?;

            if cursor_in_bounds {
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "cursor coords are clamped to layout_bounds (i32) in on_pointer_motion, so this truncation is bounded"
                )]
                let cursor_origin =
                    Point::<i32, Physical>::from((cursor_local_x as i32, cursor_local_y as i32));
                draw_cursor(&mut frame, cursor_origin)?;
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
        for surface in surfaces {
            send_frame_callbacks(surface, elapsed_ms);
        }
        Ok(())
    }
}

/// Paint the wallpaper across the full output. `Solid` is one
/// `draw_solid` call; `VerticalGradient` does 256 horizontal stripes
/// with colours linearly interpolated between top and bottom — that
/// many stripes keeps banding imperceptible on a 2160-px-tall display.
/// On shorter outputs some stripes collapse to zero height and are
/// skipped harmlessly.
fn draw_wallpaper(
    frame: &mut GlesFrame<'_, '_>,
    wallpaper: &Wallpaper,
    output_size: Size<i32, Physical>,
) -> Result<()> {
    match wallpaper {
        Wallpaper::Solid(rgb) => {
            let dst = Rectangle::<i32, Physical>::from_size(output_size);
            let damage = [Rectangle::from_size(output_size)];
            frame
                .draw_solid(dst, &damage, Color32F::new(rgb[0], rgb[1], rgb[2], 1.0))
                .context("Frame::draw_solid (wallpaper solid) failed")?;
        }
        Wallpaper::VerticalGradient { top, bottom } => {
            const STRIPE_COUNT: i32 = 256;
            let height = output_size.h;
            for stripe in 0u8..=u8::MAX {
                let t = f32::from(stripe) / 255.0;
                let color = Color32F::new(
                    top[0].mul_add(1.0 - t, bottom[0] * t),
                    top[1].mul_add(1.0 - t, bottom[1] * t),
                    top[2].mul_add(1.0 - t, bottom[2] * t),
                    1.0,
                );

                let idx = i32::from(stripe);
                let y_start = (idx * height) / STRIPE_COUNT;
                let y_end = ((idx + 1) * height) / STRIPE_COUNT;
                let stripe_h = y_end - y_start;
                if stripe_h <= 0 {
                    continue;
                }

                let stripe_dst = Rectangle::<i32, Physical>::new(
                    Point::from((0, y_start)),
                    Size::new(output_size.w, stripe_h),
                );
                let damage = [Rectangle::from_size(stripe_dst.size)];
                frame
                    .draw_solid(stripe_dst, &damage, color)
                    .context("Frame::draw_solid (wallpaper stripe) failed")?;
            }
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
fn draw_cursor(frame: &mut GlesFrame<'_, '_>, local_origin: Point<i32, Physical>) -> Result<()> {
    let cursor_bbox = Rectangle::new(local_origin, Size::new(CURSOR_SIZE, CURSOR_SIZE));
    let cursor_damage: Vec<Rectangle<i32, Physical>> = (0..CURSOR_SIZE)
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
