//! GBM + EGL + GLES2 render pipeline with vblank-driven page-flipping.
//!
//! Built on top of a `DrmSurface` acquired by [`crate::drm`]. Maintains
//! a double-buffered scanout via smithay's `GbmBufferedSurface` and
//! renders through `GlesRenderer`. Each frame is currently just a
//! clear-to-a-time-varying hue — proof that the render loop is alive
//! and pacing with the display's vblank. Real compositing (cursor,
//! client surfaces) layers on top of this same pipeline.
//!
//! Render loop shape:
//!
//! ```text
//! startup ────▶ render_and_queue ──┐
//!                                  │
//!  vblank ────▶ render_and_queue ──┤
//!                                  ▼
//!                              queue_buffer ──▶ kernel ──▶ scanout ──▶ vblank ──▶ …
//! ```
//!
//! `render_and_queue` is called once at startup and again on every
//! `DrmEvent::VBlank`. The first call kicks the cycle off (no pending
//! frame to ack), each subsequent call acks the previous frame via
//! `frame_submitted` before rendering and queuing the next.

use std::time::Instant;

use anyhow::{Context as _, Result};
use smithay::backend::allocator::Fourcc;
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
use smithay::backend::drm::{DrmDeviceFd, DrmSurface, GbmBufferedSurface};
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::gles::GlesRenderer;
use smithay::backend::renderer::{Bind as _, Color32F, Frame as _, Renderer as _};
use smithay::reexports::drm::control::Mode;
use smithay::utils::{Physical, Point, Rectangle, Size, Transform};
use tracing::info;

/// Render pipeline for one display: GBM allocator, EGL context, GLES
/// renderer, and the buffered surface that page-flips on vblank.
pub struct Renderer {
    /// Holds the `DrmSurface` and a swapchain of GBM buffers that get
    /// scanned out in turn.
    surface: GbmBufferedSurface<GbmAllocator<DrmDeviceFd>, ()>,
    /// GLES2 renderer; owns the `EGLContext` we render through.
    gles: GlesRenderer,
    /// Output dimensions in physical pixels — passed to `render` so
    /// the renderer knows the viewport size.
    output_size: Size<i32, Physical>,
    /// Wall-clock origin for the hue animation. Fixed at startup so
    /// the cycle is stable across runs (modulo wall-clock skew).
    start: Instant,
    /// Cursor hotspot position in physical pixels, advanced by
    /// [`Self::on_pointer_motion`] and read each frame to position
    /// the cursor sprite. Stored as `f64` so libinput's sub-pixel
    /// deltas accumulate without integer rounding losses; truncated
    /// to `i32` only at the draw call.
    cursor_x: f64,
    cursor_y: f64,
}

/// Side length of the cursor sprite in physical pixels. The cursor
/// is a right-triangle with apex at the hotspot, so this is also
/// the bounding-box width and height.
const CURSOR_SIZE: i32 = 24;

impl Renderer {
    /// Wire GBM + EGL + GLES on top of an already-modeset `DrmSurface`.
    /// Consumes the surface — from here on it lives inside the
    /// `GbmBufferedSurface`. `drm_fd` is the same fd the DRM device was
    /// opened on; it gets cloned (cheap, Arc-backed) into a `GbmDevice`
    /// for the EGL display and into the `GbmAllocator` for scanout
    /// buffers.
    pub fn new(drm_fd: DrmDeviceFd, drm_surface: DrmSurface, mode: Mode) -> Result<Self> {
        info!("phase: opening GBM device");
        let gbm_device = GbmDevice::new(drm_fd).context("GbmDevice::new failed")?;
        info!("GBM device created");

        info!("phase: opening EGL display from GBM device");
        #[allow(
            unsafe_code,
            reason = "EGLDisplay::new is unsafe because the GbmDevice it stores must outlive the display. gbm::Device's clone is Arc-backed (DrmDeviceFd is Clone), and the cloned handle lives inside EGLDisplay for its entire lifetime — we never free or invalidate the underlying gbm_device while EGLDisplay references it."
        )]
        // SAFETY: see #[allow] above. The cloned GbmDevice is owned by
        // EGLDisplay and its underlying gbm_device is Arc-managed so it
        // stays valid until EGLDisplay drops.
        let egl_display =
            unsafe { EGLDisplay::new(gbm_device.clone()) }.context("EGLDisplay::new failed")?;
        info!("EGL display opened");

        info!("phase: creating EGL context");
        let egl_context = EGLContext::new(&egl_display).context("EGLContext::new failed")?;
        info!("EGL context created");

        info!("phase: creating GLES renderer");
        #[allow(
            unsafe_code,
            reason = "GlesRenderer::new requires the EGLContext to be used from a single thread (it calls make_current internally and assumes that's safe). The compositor is single-threaded and the Renderer never crosses threads — we own the EGLContext exclusively from here on."
        )]
        // SAFETY: see #[allow]. EGLContext is moved into GlesRenderer
        // and stays on this thread for the lifetime of the renderer.
        let gles = unsafe { GlesRenderer::new(egl_context) }.context("GlesRenderer::new failed")?;
        info!("GLES renderer created");

        info!("phase: creating GBM allocator");
        let allocator = GbmAllocator::new(
            gbm_device,
            GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT,
        );

        info!("phase: creating GBM buffered surface (swapchain + modeset)");
        let renderer_formats = gles.egl_context().dmabuf_render_formats().clone();
        let surface = GbmBufferedSurface::new(
            drm_surface,
            allocator,
            &[Fourcc::Xrgb8888],
            renderer_formats,
        )
        .context("GbmBufferedSurface::new failed (no compatible scanout format?)")?;
        info!("GBM buffered surface ready");

        let (w, h) = mode.size();
        Ok(Self {
            surface,
            gles,
            output_size: Size::new(i32::from(w), i32::from(h)),
            start: Instant::now(),
            // Start the cursor at the centre of the output rather
            // than (0, 0) so it's immediately visible after the
            // first paint without needing pointer movement.
            cursor_x: f64::from(w) / 2.0,
            cursor_y: f64::from(h) / 2.0,
        })
    }

    /// Advance the cursor hotspot by libinput-reported relative
    /// deltas (already acceleration-adjusted by libinput), clamping
    /// to the output rectangle so it can't run off-screen. Called
    /// once per `InputEvent::PointerMotion`.
    pub fn on_pointer_motion(&mut self, dx: f64, dy: f64) {
        let max_x = f64::from(self.output_size.w);
        let max_y = f64::from(self.output_size.h);
        self.cursor_x = (self.cursor_x + dx).clamp(0.0, max_x);
        self.cursor_y = (self.cursor_y + dy).clamp(0.0, max_y);
    }

    /// Render one frame and queue it for scanout. Called once at
    /// startup to prime the swapchain, and again on every
    /// `DrmEvent::VBlank` to keep the cycle going. The `frame_submitted`
    /// up-front is a no-op on the first call (no pending frame yet) and
    /// the ack for the previous frame on every subsequent call.
    pub fn render_and_queue(&mut self) -> Result<()> {
        let _ = self
            .surface
            .frame_submitted()
            .context("GbmBufferedSurface::frame_submitted failed")?;

        let (mut dmabuf, _age) = self
            .surface
            .next_buffer()
            .context("GbmBufferedSurface::next_buffer failed")?;

        // Hue cycles 360° every 8 seconds (45°/s). Time origin is
        // fixed at construction so the cycle is stable across vblanks
        // rather than reset on each frame.
        let elapsed = self.start.elapsed().as_secs_f32();
        let hue = (elapsed * 45.0) % 360.0;
        let (r, g, b) = hue_to_rgb(hue);

        // GLES's `Frame::clear` is damage-aware: an empty `at` slice
        // is treated as "nothing changed, nothing to paint" and the
        // function early-returns without touching pixels. Pass the
        // whole-output rectangle so the entire framebuffer actually
        // gets coloured every frame.
        let full_damage = [Rectangle::<i32, Physical>::from_size(self.output_size)];

        // Build the cursor draw: a right-triangle with apex at the
        // hotspot. We pass the bounding box as `dst` and a per-row
        // stripe list as the damage to `draw_solid`. Each damage rect's
        // `loc` is **relative to `dst.loc`** — `GlesFrame::draw_solid`
        // computes the final vertex as `dst.loc + damage.loc` and
        // clamps damage to the local `0..dst.size` range first. So
        // stripe[row] lives at (0, row), not at the absolute cursor
        // position. (Got this wrong on the first pass and the cursor
        // was invisible — every stripe got clamped to a zero-size rect.)
        //
        // Truncation `as i32` is bounded: `cursor_x`/`cursor_y` are
        // clamped to `output_size` (i32) on every motion event.
        #[allow(
            clippy::cast_possible_truncation,
            reason = "cursor coords are clamped to output_size (i32) in on_pointer_motion, so truncation is bounded and intentional"
        )]
        let cursor_origin =
            Point::<i32, Physical>::from((self.cursor_x as i32, self.cursor_y as i32));
        let cursor_bbox = Rectangle::new(cursor_origin, Size::new(CURSOR_SIZE, CURSOR_SIZE));
        // Row `n` is `n+1` pixels wide, anchored at the left edge of
        // the bbox: row 0 is a single pixel at the apex (top-left,
        // which is also the hotspot), row CURSOR_SIZE-1 is the full
        // base. The result is a top-left-pointing arrow silhouette.
        let cursor_damage: Vec<Rectangle<i32, Physical>> = (0..CURSOR_SIZE)
            .map(|row| Rectangle::new(Point::from((0, row)), Size::new(row + 1, 1)))
            .collect();

        // The sync point from `Frame::finish` is handed to
        // `queue_buffer` so the kernel waits for GPU completion
        // before scanning out — otherwise we'd race the page flip
        // against the GL submission and see tearing or stale frames.
        let sync = {
            let mut target = self
                .gles
                .bind(&mut dmabuf)
                .context("GlesRenderer::bind dmabuf failed")?;
            let mut frame = self
                .gles
                .render(&mut target, self.output_size, Transform::Normal)
                .context("GlesRenderer::render begin failed")?;
            frame
                .clear(Color32F::new(r, g, b, 1.0), &full_damage)
                .context("Frame::clear failed")?;
            frame
                .draw_solid(
                    cursor_bbox,
                    &cursor_damage,
                    Color32F::new(1.0, 1.0, 1.0, 1.0),
                )
                .context("Frame::draw_solid (cursor) failed")?;
            frame.finish().context("Frame::finish failed")?
        };

        self.surface
            .queue_buffer(Some(sync), None, ())
            .context("GbmBufferedSurface::queue_buffer failed")?;
        Ok(())
    }
}

/// HSV(`hue`, 1.0, 1.0) → linear RGB in [0, 1]. Standard 6-region
/// formula. Hue is in degrees and wraps around 360. Uses `f32`
/// comparisons rather than `h as i32` to keep `clippy::pedantic` happy
/// without an allow — and the if-ladder is no harder to read than the
/// cast-and-match form.
fn hue_to_rgb(hue: f32) -> (f32, f32, f32) {
    let h = (hue % 360.0) / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    if h < 1.0 {
        (1.0, x, 0.0)
    } else if h < 2.0 {
        (x, 1.0, 0.0)
    } else if h < 3.0 {
        (0.0, 1.0, x)
    } else if h < 4.0 {
        (0.0, x, 1.0)
    } else if h < 5.0 {
        (x, 0.0, 1.0)
    } else {
        (1.0, 0.0, x)
    }
}
