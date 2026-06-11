//! `zwlr_screencopy_v1` — frame capture for screenshots and screen
//! sharing.
//!
//! This is the one compositor-side piece the wlroots portal route
//! needs: `grim`/`satty` use it directly for screenshots, and
//! `xdg-desktop-portal-wlr` uses it (feeding `PipeWire`) for
//! screencast, so OBS / Discord / browser screen-share work.
//!
//! A client asks to capture an output; we reply with the buffer
//! parameters it should allocate (`buffer` + `buffer_done`), it
//! creates a matching `wl_shm` buffer and calls `copy`. We queue the
//! request and, on the next render of that output, read the freshly
//! composited framebuffer back into the client's buffer and signal
//! `ready`. The actual GPU readback lives in [`crate::render`]; this
//! module owns the protocol + buffer bookkeeping.

use std::sync::Mutex;

use smithay::output::Output;
use smithay::reexports::wayland_protocols_wlr::screencopy::v1::server::{
    zwlr_screencopy_frame_v1::{self, ZwlrScreencopyFrameV1},
    zwlr_screencopy_manager_v1::{self, ZwlrScreencopyManagerV1},
};
use smithay::reexports::wayland_server::protocol::{
    wl_buffer::WlBuffer, wl_output::WlOutput, wl_shm,
};
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};
use smithay::utils::{Physical, Rectangle};
use tracing::{debug, warn};

use crate::State;

const MANAGER_VERSION: u32 = 3;

/// The shm pixel format we capture into. xRGB8888 (little-endian,
/// 4 bytes/px) is what `grim` and `xdg-desktop-portal-wlr` expect and
/// what our renderer reads back; we advertise only this one.
const CAPTURE_SHM_FORMAT: wl_shm::Format = wl_shm::Format::Xrgb8888;
/// The `Fourcc` the renderer reads back, matching `CAPTURE_SHM_FORMAT`.
/// Exposed so the render path captures in the same format we advertise.
pub(crate) const CAPTURE_FOURCC: smithay::backend::allocator::Fourcc =
    smithay::backend::allocator::Fourcc::Xrgb8888;

/// Per-frame state stored as the `zwlr_screencopy_frame_v1` user data:
/// what to capture, resolved at `capture_output` time, plus a guard so
/// `copy` can only be honoured once (a second `copy` is a protocol
/// error per the spec).
#[derive(Debug)]
pub struct FrameData {
    inner: Mutex<FrameInner>,
}

#[derive(Debug)]
struct FrameInner {
    /// Connector name of the output to capture.
    output: String,
    /// Region to capture, in the output's buffer (physical) pixels.
    region: Rectangle<i32, Physical>,
    overlay_cursor: bool,
    /// Set once `copy`/`copy_with_damage` has been accepted.
    used: bool,
}

/// A `copy` request awaiting the next render of its output. The vblank
/// path drains these, has the renderer read the framebuffer back, and
/// writes the pixels into `buffer` before signalling the frame.
#[derive(Debug)]
pub struct PendingCapture {
    pub frame: ZwlrScreencopyFrameV1,
    pub buffer: WlBuffer,
    pub output: String,
    pub region: Rectangle<i32, Physical>,
    pub overlay_cursor: bool,
}

/// Holds the `zwlr_screencopy_manager_v1` global alive.
#[derive(Debug)]
pub struct ScreencopyManagerState {
    #[allow(dead_code, reason = "held to keep the global alive")]
    global: smithay::reexports::wayland_server::backend::GlobalId,
}

impl ScreencopyManagerState {
    pub fn new(dh: &DisplayHandle) -> Self
    where
        State: GlobalDispatch<ZwlrScreencopyManagerV1, ()>,
    {
        let global = dh.create_global::<State, ZwlrScreencopyManagerV1, ()>(MANAGER_VERSION, ());
        Self { global }
    }
}

impl GlobalDispatch<ZwlrScreencopyManagerV1, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwlrScreencopyManagerV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<ZwlrScreencopyManagerV1, ()> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _manager: &ZwlrScreencopyManagerV1,
        request: zwlr_screencopy_manager_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwlr_screencopy_manager_v1::Request::CaptureOutput {
                frame,
                overlay_cursor,
                output,
            } => {
                let region = output_full_region(state, &output);
                begin_frame(
                    state,
                    frame,
                    overlay_cursor != 0,
                    &output,
                    region,
                    data_init,
                );
            }
            zwlr_screencopy_manager_v1::Request::CaptureOutputRegion {
                frame,
                overlay_cursor,
                output,
                x,
                y,
                width,
                height,
            } => {
                // Clamp the requested region to the output below.
                let requested = Rectangle::new((x, y).into(), (width.max(1), height.max(1)).into());
                let full = output_full_region(state, &output);
                let region = full.intersection(requested).unwrap_or(full);
                begin_frame(
                    state,
                    frame,
                    overlay_cursor != 0,
                    &output,
                    region,
                    data_init,
                );
            }
            // Destroy + any future requests: nothing to do (smithay
            // tears the resource down).
            _ => {}
        }
    }
}

/// Resolve the full output rect (in physical/buffer pixels) for the
/// `wl_output`, or a zero rect if it isn't one of ours.
fn output_full_region(state: &State, output: &WlOutput) -> Rectangle<i32, Physical> {
    let Some(smithay_output) = Output::from_resource(output) else {
        return Rectangle::default();
    };
    let name = smithay_output.name();
    state
        .renderer
        .output_mode_size(&name)
        .map(|size| Rectangle::new((0, 0).into(), size))
        .unwrap_or_default()
}

/// Create the frame object, stash its capture parameters, and tell the
/// client which buffer to allocate (`buffer` + `buffer_done`). A
/// zero-size region means the output wasn't found — fail the frame.
fn begin_frame(
    state: &State,
    frame: New<ZwlrScreencopyFrameV1>,
    overlay_cursor: bool,
    output: &WlOutput,
    region: Rectangle<i32, Physical>,
    data_init: &mut DataInit<'_, State>,
) {
    let output_name = Output::from_resource(output)
        .map(|o| o.name())
        .unwrap_or_default();
    let data = FrameData {
        inner: Mutex::new(FrameInner {
            output: output_name,
            region,
            overlay_cursor,
            used: false,
        }),
    };
    let frame = data_init.init(frame, data);

    if region.size.w <= 0 || region.size.h <= 0 {
        warn!("screencopy: capture of unknown/empty output; failing frame");
        frame.failed();
        return;
    }

    #[allow(
        clippy::cast_sign_loss,
        reason = "region size is clamped > 0 above; stride/dims are positive"
    )]
    let (w, h) = (region.size.w as u32, region.size.h as u32);
    let stride = w * 4;
    // `linux_dmabuf` and `buffer_done` are both `since = 3`; sending them
    // to a client that bound an older manager version is a protocol error
    // (the client aborts with "interface has no event N"). Only advertise
    // the dmabuf path to v3+ clients (xdg-desktop-portal-wlr) so they get
    // the zero-copy GPU blit; older/simple clients (grim) bind v1/v2 and
    // get just the shm `buffer` event. `linux_dmabuf` carries the Fourcc
    // as a raw u32.
    if frame.version() >= 3 {
        frame.linux_dmabuf(CAPTURE_FOURCC as u32, w, h);
    }
    frame.buffer(CAPTURE_SHM_FORMAT, w, h, stride);
    if frame.version() >= 3 {
        frame.buffer_done();
    }
    let _ = state;
}

impl Dispatch<ZwlrScreencopyFrameV1, FrameData> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        frame: &ZwlrScreencopyFrameV1,
        request: zwlr_screencopy_frame_v1::Request,
        data: &FrameData,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwlr_screencopy_frame_v1::Request::Copy { buffer }
            | zwlr_screencopy_frame_v1::Request::CopyWithDamage { buffer } => {
                let (output, region, overlay_cursor) = {
                    let mut inner = data.inner.lock().unwrap();
                    if inner.used {
                        frame.post_error(
                            zwlr_screencopy_frame_v1::Error::AlreadyUsed,
                            "copy already requested on this frame",
                        );
                        return;
                    }
                    inner.used = true;
                    (inner.output.clone(), inner.region, inner.overlay_cursor)
                };

                // Classify the buffer. A dmabuf takes the GPU blit path,
                // but only if we can actually render *into* it — we
                // advertise the texture-import format set, which is a
                // superset of what we can blit to (notably on NVIDIA).
                // A dmabuf we can't render to is failed *gracefully*
                // (frame.failed, not a protocol error) so the client
                // retries with the shm buffer we also advertised. An shm
                // buffer is validated against the advertised format/size;
                // a genuinely malformed one is a protocol error.
                if let Ok(dmabuf) = smithay::wayland::dmabuf::get_dmabuf(&buffer) {
                    use smithay::backend::allocator::Buffer as _;
                    let dims_ok = i32::try_from(dmabuf.width()).is_ok_and(|w| w == region.size.w)
                        && i32::try_from(dmabuf.height()).is_ok_and(|h| h == region.size.h)
                        && dmabuf.format().code == CAPTURE_FOURCC;
                    let renderable = state.renderer.can_render_to(dmabuf.format());
                    if !dims_ok || !renderable {
                        debug!(
                            dims_ok,
                            renderable,
                            "screencopy: dmabuf unusable; failing frame so the client falls back to shm"
                        );
                        frame.failed();
                        return;
                    }
                } else if !shm_buffer_valid(&buffer, region) {
                    frame.post_error(
                        zwlr_screencopy_frame_v1::Error::InvalidBuffer,
                        "buffer does not match the advertised format/size",
                    );
                    return;
                }

                // Wake the captured output so the pending grab is serviced;
                // an idle on-demand output would never flip on its own.
                let crtc = state.renderer.crtc_for_output_name(&output);
                state.screencopy_pending.push(PendingCapture {
                    frame: frame.clone(),
                    buffer,
                    output,
                    region,
                    overlay_cursor,
                });
                if let Some(crtc) = crtc {
                    state.queue_redraw(crtc);
                }
            }
            zwlr_screencopy_frame_v1::Request::Destroy => {
                state.screencopy_pending.retain(|p| &p.frame != frame);
            }
            _ => {}
        }
    }
}

/// Check the client's `wl_shm` buffer matches what we advertised
/// (xRGB8888, `region` size). Accepts any `stride >= w*4` — `PipeWire`
/// / toolkits often align the stride, which is fine since the write
/// honours the buffer's own stride.
fn shm_buffer_valid(buffer: &WlBuffer, region: Rectangle<i32, Physical>) -> bool {
    let (Ok(w), Ok(h)) = (
        usize::try_from(region.size.w),
        usize::try_from(region.size.h),
    ) else {
        return false;
    };
    smithay::wayland::shm::with_buffer_contents(buffer, |_ptr, len, spec| {
        let stride = usize::try_from(spec.stride).unwrap_or(0);
        spec.format == CAPTURE_SHM_FORMAT
            && spec.width == region.size.w
            && spec.height == region.size.h
            && stride >= w * 4
            && len >= stride * h
    })
    .unwrap_or(false)
}

/// Finish a serviced capture: for the shm path write the read-back
/// pixels (already upright — FBO readbacks are memory-ordered) into
/// the client buffer; for the dmabuf path the GPU blit already filled
/// it upright. Then signal `ready`, or `failed` if the capture didn't
/// happen.
///
/// We always deliver an upright buffer and never set the `y_invert`
/// flag: xdg-desktop-portal-wlr 0.8.2 has no `y_invert` handling — its
/// enqueue path hits an unimplemented `//TODO` stub that destroys the
/// cast instance, which then double-frees during stream teardown and
/// crashes the portal (SIGSEGV). Handling the flip ourselves keeps the
/// stock portal working.
pub fn complete(pending: &PendingCapture, outcome: crate::render::CaptureOutcome) {
    use crate::render::CaptureOutcome;
    match outcome {
        CaptureOutcome::Failed => {
            pending.frame.failed();
            return;
        }
        CaptureOutcome::Shm {
            bytes,
            width,
            height,
        } => {
            if !write_to_shm(&pending.buffer, &bytes, width, height) {
                pending.frame.failed();
                return;
            }
        }
        CaptureOutcome::Dmabuf => {}
    }

    let (secs, nsecs) = monotonic_now();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "tv_sec is split into the hi/lo u32 halves the protocol's ready event takes"
    )]
    let (hi, lo) = ((secs >> 32) as u32, (secs & 0xFFFF_FFFF) as u32);
    pending.frame.ready(hi, lo, nsecs);
}

/// Copy the tight captured pixels (row = `width * 4`, rows top-down)
/// into the client's shm buffer, honouring its stride. The source is
/// already upright (we never use the `y_invert` flag — see
/// [`complete`]). Returns whether it fit.
fn write_to_shm(buffer: &WlBuffer, bytes: &[u8], width: u32, height: u32) -> bool {
    let row = width as usize * 4;
    let rows = height as usize;
    smithay::wayland::shm::with_buffer_contents_mut(buffer, |ptr, len, spec| {
        let Ok(dst_stride) = usize::try_from(spec.stride) else {
            return false;
        };
        if dst_stride < row || len < dst_stride * rows || bytes.len() < row * rows {
            return false;
        }
        // SAFETY: smithay's shm contract guarantees `ptr` is valid for
        // `len` writable bytes for the duration of this callback. We
        // write `rows` rows of `row` bytes at `dst_stride` spacing,
        // bounded above by the `len`/`dst_stride` checks, so every
        // write stays within `[ptr, ptr + len)`. The source row index is
        // also in `0..rows`, so reads stay within `bytes`. Source and
        // dest don't overlap (different allocations).
        #[allow(
            unsafe_code,
            reason = "shm buffer access is a raw pointer per smithay's API; bounds are checked above"
        )]
        unsafe {
            for y in 0..rows {
                let src = bytes.as_ptr().add(y * row);
                let dst = ptr.add(y * dst_stride);
                std::ptr::copy_nonoverlapping(src, dst, row);
            }
        }
        true
    })
    .unwrap_or(false)
}

/// Monotonic clock as `(seconds, nanoseconds)` for the `ready` event's
/// presentation timestamp.
fn monotonic_now() -> (u64, u32) {
    let ts = smithay::reexports::rustix::time::clock_gettime(
        smithay::reexports::rustix::time::ClockId::Monotonic,
    );
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "monotonic clock values are non-negative and tv_nsec is < 1e9"
    )]
    (ts.tv_sec as u64, ts.tv_nsec as u32)
}
