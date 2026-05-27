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

use anyhow::{Context as _, Result};
use smithay::backend::allocator::Fourcc;
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBufferFlags, GbmDevice};
use smithay::backend::drm::{DrmDeviceFd, GbmBufferedSurface};
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::gles::{GlesFrame, GlesRenderer};
use smithay::backend::renderer::{Bind as _, Color32F, Frame as _, Renderer as _};
use smithay::reexports::drm::control::crtc;
use smithay::utils::{Physical, Point, Rectangle, Size, Transform};
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
        })
    }

    /// Render every output's initial frame to prime its swapchain.
    /// Called once at startup before the event loop runs; thereafter
    /// each output's frames are driven by its own vblank events.
    pub fn render_initial(&mut self) -> Result<()> {
        for idx in 0..self.outputs.len() {
            self.render_output(idx)
                .with_context(|| format!("initial render of output #{idx} failed"))?;
        }
        Ok(())
    }

    /// Render the output driven by `crtc`, in response to its vblank.
    pub fn render_for_crtc(&mut self, crtc: crtc::Handle) -> Result<()> {
        let idx = self
            .outputs
            .iter()
            .position(|o| o.crtc == crtc)
            .with_context(|| format!("vblank for unknown CRTC {crtc:?}"))?;
        self.render_output(idx)
    }

    /// Advance the cursor hotspot by libinput-reported deltas, clamped
    /// to the virtual layout's bounding box.
    pub fn on_pointer_motion(&mut self, dx: f64, dy: f64) {
        let max_x = f64::from(self.layout_bounds.w);
        let max_y = f64::from(self.layout_bounds.h);
        self.cursor_x = (self.cursor_x + dx).clamp(0.0, max_x);
        self.cursor_y = (self.cursor_y + dy).clamp(0.0, max_y);
    }

    /// Render one output's frame: wallpaper, then cursor sprite if
    /// the global hotspot falls within this output's rectangle.
    fn render_output(&mut self, idx: usize) -> Result<()> {
        // Pull everything we need before the GLES bind borrows
        // `self.gles`. After that point we can still read other
        // fields of `self` (split borrows) but it's clearer to
        // localise.
        let cursor_x = self.cursor_x;
        let cursor_y = self.cursor_y;
        let wallpaper = self.wallpaper.clone();

        let output = &mut self.outputs[idx];
        let output_size = output.size;
        let output_pos = output.position;
        let output_name = output.name.clone();

        // No-op on the first call (no pending fb), the ack of the
        // previous frame's flip thereafter.
        let _ = output
            .surface
            .frame_submitted()
            .with_context(|| format!("frame_submitted failed for {output_name}"))?;

        let (mut dmabuf, _age) = output
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

            if cursor_in_bounds {
                // Truncation `as i32` is bounded: cursor coords are
                // clamped to layout_bounds (i32), and local coords
                // here are within that minus a positive offset.
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "cursor coords are clamped to layout_bounds (i32) in on_pointer_motion, so this truncation is bounded"
                )]
                let local_origin =
                    Point::<i32, Physical>::from((cursor_local_x as i32, cursor_local_y as i32));
                draw_cursor(&mut frame, local_origin)?;
            }

            frame.finish().context("Frame::finish failed")?
        };

        output
            .surface
            .queue_buffer(Some(sync), None, ())
            .with_context(|| format!("queue_buffer failed for {output_name}"))?;
        debug!(output = %output_name, "frame queued for scanout");
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
