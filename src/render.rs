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

use crate::config::{BorderConfig, Fill};
use crate::drm::DrmOutput;
use crate::layout::Placement;

/// Side length of the cursor sprite in physical pixels. The sprite
/// is a right-triangle with apex at the hotspot, so this is also
/// the bounding-box width and height.
const CURSOR_SIZE: i32 = 24;

/// GLES2 fragment shader that masks rounded-rectangle corners on
/// top of an already-drawn window. The shader is called once per
/// window over the window's cell rect; for fragments inside the
/// rounded shape it writes alpha = 0 (existing surface pixel
/// kept), for fragments outside the rounded shape it writes the
/// wallpaper colour with alpha = 1, and the ~1 px transition
/// between the two ends gets a smoothstep ramp so the rounded
/// edge looks anti-aliased rather than staircased. Mask colour
/// is sampled from a vertical gradient (`top`/`bottom`) using the
/// fragment's *global* y so it stays continuous with the
/// wallpaper drawn underneath.
///
/// Uniforms: `size` and `alpha` come from smithay (viewport size
/// and per-call alpha multiplier); the rest are registered when
/// we compile the shader and updated each frame.
const ROUNDED_MASK_SHADER: &str = r"
precision mediump float;

uniform vec2 size;
uniform float alpha;
uniform float radius;
uniform vec3 grad_top;
uniform vec3 grad_bottom;
uniform float output_height;
uniform float cell_origin_y;

varying vec2 v_coords;

void main() {
    vec2 p = v_coords * size;
    vec2 half_size = size * 0.5;
    vec2 d = abs(p - half_size) - (half_size - vec2(radius));
    float dist = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - radius;
    float mask_alpha = smoothstep(-0.5, 0.5, dist);
    if (mask_alpha <= 0.0) {
        discard;
    }
    float global_y = cell_origin_y + p.y;
    float t = clamp(global_y / max(output_height, 1.0), 0.0, 1.0);
    vec3 mask_rgb = mix(grad_top, grad_bottom, t);
    gl_FragColor = vec4(mask_rgb, mask_alpha * alpha);
}
";

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
    wallpaper: Fill,
    /// Window border width + active / inactive fills.
    border: BorderConfig,
    /// Custom GLES pixel shader used to mask rounded corners.
    /// `Arc`-backed so it's cheap to clone out before borrowing
    /// the renderer for the frame.
    rounded_shader: GlesPixelProgram,
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
    #[allow(
        clippy::too_many_lines,
        reason = "linear initialisation sequence (GBM device, EGL display, EGL context, GLES renderer, custom shader, GBM allocator, per-output GbmBufferedSurfaces). Splitting it forces threading several mid-construction values through extra functions for no real win."
    )]
    pub fn new(
        drm_fd: DrmDeviceFd,
        drm_outputs: Vec<DrmOutput>,
        wallpaper: Fill,
        border: BorderConfig,
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

        info!("phase: compiling rounded-corner pixel shader");
        let rounded_shader = gles
            .compile_custom_pixel_shader(
                ROUNDED_MASK_SHADER,
                &[
                    UniformName::new("radius", UniformType::_1f),
                    UniformName::new("grad_top", UniformType::_3f),
                    UniformName::new("grad_bottom", UniformType::_3f),
                    UniformName::new("output_height", UniformType::_1f),
                    UniformName::new("cell_origin_y", UniformType::_1f),
                ],
            )
            .context("rounded-corner shader compile failed")?;

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
            border,
            rounded_shader,
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
            self.render_output(idx, &[])
                .with_context(|| format!("initial render of output #{idx} failed"))?;
        }
        Ok(())
    }

    /// Render the output driven by `crtc`, in response to its vblank.
    /// `placements` is the caller-snapshot of every visible window as
    /// `(wl_surface, top-left in absolute virtual-layout coords)`;
    /// the layout module owns positioning, the renderer just paints.
    pub fn render_for_crtc(&mut self, crtc: crtc::Handle, placements: &[Placement]) -> Result<()> {
        let idx = self
            .outputs
            .iter()
            .position(|o| o.crtc == crtc)
            .with_context(|| format!("vblank for unknown CRTC {crtc:?}"))?;
        self.render_output(idx, placements)
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

    /// Rectangle of the first connected output in absolute virtual-
    /// layout coordinates. Used by the tiling layer to bound its
    /// initial workspace before per-output workspaces exist.
    /// Renderer guarantees a non-empty `outputs` (panic at
    /// construction otherwise), so the `expect` is unreachable.
    pub fn primary_output_rect(&self) -> Rectangle<i32, Physical> {
        let o = self
            .outputs
            .first()
            .expect("Renderer constructed with zero outputs");
        Rectangle::new(o.position, o.size)
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
    fn render_output(&mut self, idx: usize, placements: &[Placement]) -> Result<()> {
        // Pull everything we need before the mutable borrows on
        // `self.outputs[idx].surface` / `self.gles` kick in.
        let cursor_x = self.cursor_x;
        let cursor_y = self.cursor_y;
        let wallpaper = self.wallpaper.clone();
        let border = self.border.clone();
        let rounded_shader = self.rounded_shader.clone();
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
        // Per placement: the surface itself draws inside the cell,
        // shrunk by `border` on every side. Cell -> output-local
        // coords by subtracting `output_pos`. Surfaces whose
        // bounding rect doesn't overlap the output get fully
        // clipped by the GLES viewport — no need to early-skip.
        let bw = border.width.max(0);
        let grouped: Vec<Vec<WaylandSurfaceRenderElement<GlesRenderer>>> = placements
            .iter()
            .map(|p| {
                let surface_local = Point::<i32, Physical>::from((
                    p.cell_rect.loc.x + bw - output_pos.x,
                    p.cell_rect.loc.y + bw - output_pos.y,
                ));
                render_elements_from_surface_tree(
                    &mut self.gles,
                    &p.surface,
                    surface_local,
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

            // Background fills first, then for each placement
            // (bottom-up) the border frame followed by the surface,
            // so that floating-window borders and surfaces end up
            // visually above the tiled cells they overlap. Each
            // window's `draw_render_elements` call carries just
            // that window's elements — smithay's opaque-region
            // culling can't accidentally skip floats behind
            // earlier tiles when there's only ever one element in
            // the slice.
            draw_fill(&mut frame, &wallpaper, output_size, output_size)?;

            let full_damage = [Rectangle::<i32, Physical>::from_size(output_size)];
            let radius = border.rounded_corners.max(0);
            for (p, elements) in placements.iter().zip(grouped.iter()) {
                if bw > 0 {
                    let fill = if p.focused {
                        &border.active
                    } else {
                        &border.inactive
                    };
                    draw_border(&mut frame, p.cell_rect, output_pos, bw, fill, output_size)?;
                }
                draw_render_elements::<GlesRenderer, _, _>(
                    &mut frame,
                    1.0_f64,
                    elements,
                    &full_damage,
                )
                .context("draw_render_elements failed")?;
                if radius > 0 {
                    draw_rounded_corner_mask(
                        &mut frame,
                        &rounded_shader,
                        p.cell_rect,
                        output_pos,
                        radius,
                        &wallpaper,
                        output_size,
                    )?;
                }
            }

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
        for p in placements {
            send_frame_callbacks(&p.surface, elapsed_ms);
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
fn draw_rounded_corner_mask(
    frame: &mut GlesFrame<'_, '_>,
    shader: &GlesPixelProgram,
    cell_rect: Rectangle<i32, Physical>,
    output_pos: Point<i32, Physical>,
    radius: i32,
    wallpaper: &Fill,
    output_size: Size<i32, Physical>,
) -> Result<()> {
    let r = radius.min(cell_rect.size.w / 2).min(cell_rect.size.h / 2);
    if r <= 0 {
        return Ok(());
    }
    let local_x = cell_rect.loc.x - output_pos.x;
    let local_y = cell_rect.loc.y - output_pos.y;
    let dest = Rectangle::<i32, Physical>::new(Point::new(local_x, local_y), cell_rect.size);
    let (grad_top, grad_bottom) = match wallpaper {
        Fill::Solid(rgb) => (*rgb, *rgb),
        Fill::VerticalGradient { top, bottom } => (*top, *bottom),
    };
    #[allow(
        clippy::cast_precision_loss,
        reason = "radius and output_height are bounded by i32 cell / output sizes; fit f32 exactly for any realistic value"
    )]
    let uniforms = [
        Uniform::new("radius", r as f32),
        Uniform::new("grad_top", grad_top),
        Uniform::new("grad_bottom", grad_bottom),
        Uniform::new("output_height", output_size.h as f32),
        Uniform::new("cell_origin_y", local_y as f32),
    ];
    // The source rect / sample size are unused by our shader (we
    // don't sample any texture) but the API requires them; pass
    // the cell size so the `size` uniform the shader does read
    // ends up as the cell's pixel size.
    let src = Rectangle::<f64, smithay::utils::Buffer>::from_size(smithay::utils::Size::<
        f64,
        smithay::utils::Buffer,
    >::from((
        f64::from(cell_rect.size.w),
        f64::from(cell_rect.size.h),
    )));
    let size = smithay::utils::Size::<i32, smithay::utils::Buffer>::from((
        cell_rect.size.w,
        cell_rect.size.h,
    ));
    frame
        .render_pixel_shader_to(shader, src, dest, size, None, 1.0, &uniforms)
        .context("render_pixel_shader_to (rounded mask) failed")?;
    Ok(())
}

/// Draw a window's border as four rectangles around the cell.
/// `cell_rect` is in absolute virtual-layout coords; the four
/// edges are converted to output-local by subtracting
/// `output_pos`. Edge sizes that go negative (cell smaller than
/// `2 * width`) are clamped and the affected edges no-op.
fn draw_border(
    frame: &mut GlesFrame<'_, '_>,
    cell_rect: Rectangle<i32, Physical>,
    output_pos: Point<i32, Physical>,
    width: i32,
    fill: &Fill,
    output_size: Size<i32, Physical>,
) -> Result<()> {
    let local = Rectangle::<i32, Physical>::new(
        Point::new(
            cell_rect.loc.x - output_pos.x,
            cell_rect.loc.y - output_pos.y,
        ),
        cell_rect.size,
    );
    let mid_h = (local.size.h - 2 * width).max(0);
    let top = Rectangle::<i32, Physical>::new(local.loc, Size::new(local.size.w, width));
    let bottom = Rectangle::<i32, Physical>::new(
        Point::new(local.loc.x, local.loc.y + local.size.h - width),
        Size::new(local.size.w, width),
    );
    let left = Rectangle::<i32, Physical>::new(
        Point::new(local.loc.x, local.loc.y + width),
        Size::new(width, mid_h),
    );
    let right = Rectangle::<i32, Physical>::new(
        Point::new(local.loc.x + local.size.w - width, local.loc.y + width),
        Size::new(width, mid_h),
    );
    for edge in [top, bottom, left, right] {
        draw_fill_rect(frame, fill, edge, output_size)?;
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
