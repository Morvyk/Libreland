//! `ext-image-copy-capture-v1` + `ext-image-capture-source-v1` — the
//! modern screen-capture stack (the successor to wlr-screencopy, spoken
//! by new xdg-desktop-portal backends). Output capture sources only for
//! now: toplevel sources need `ext-foreign-toplevel-list`, which we
//! don't implement yet.
//!
//! The heavy lifting rides the SAME render-path capture machinery as
//! wlr-screencopy: a requested frame is parked in
//! [`crate::State::copy_frames`], the captured output's next render
//! drains it into a [`crate::render::CaptureSpec`] (GPU blit into a
//! client dmabuf, or CPU read-back for shm), and the outcome completes
//! the frame here. Formats match screencopy exactly
//! ([`crate::screencopy::CAPTURE_FOURCC`] / Xrgb8888), so both stacks
//! capture identical pixels.
//!
//! Cursor sessions are not offered (`cursor_capture_constraints` stays
//! `None`); clients that want the cursor composited use the session's
//! `paint_cursors` option, which maps onto the same overlay-cursor flag
//! screencopy uses.

use smithay::output::Output;
use smithay::utils::{Clock, Monotonic, Rectangle, Transform};
use smithay::wayland::image_capture_source::{
    ImageCaptureSource, ImageCaptureSourceHandler, OutputCaptureSourceHandler,
    OutputCaptureSourceState,
};
use smithay::wayland::image_copy_capture::{
    BufferConstraints, CaptureFailureReason as FailureReason, DmabufConstraints, Frame,
    ImageCopyCaptureHandler, ImageCopyCaptureState, Session, SessionRef,
};
use tracing::{debug, warn};

use crate::State;
use crate::render::{CaptureOutcome, CaptureSpec, CaptureTarget};
use crate::screencopy::{CAPTURE_FOURCC, CAPTURE_SHM_FORMAT, write_to_shm};

/// The captured output's connector name, stashed on an
/// [`ImageCaptureSource`]'s user data when the source is created.
pub(crate) struct SourceOutput(pub String);

/// The captured output's connector name, stashed on a SESSION's user
/// data at creation — resolved from the source then, so the legal
/// create-source → create-session → destroy-source pattern keeps
/// working after the source object dies.
struct SessionOutput(String);

/// Monotonic key stashed on a session's user data so
/// `session_destroyed` (which only hands back a [`SessionRef`]) can find
/// the owned [`Session`] in [`State::capture_sessions`]. Dropping the
/// owned session is what stops it, so removal must be exact.
struct SessionKey(u64);

/// A requested capture frame parked until its output's next render.
pub(crate) struct PendingCopyFrame {
    /// Connector name of the output to capture.
    pub(crate) output: String,
    /// The protocol frame; completed via [`Frame::success`]/[`Frame::fail`]
    /// once the render's capture outcome is known.
    pub(crate) frame: Frame,
    /// Whether the session asked for the cursor composited into the
    /// frame (`paint_cursors`) — same semantics as screencopy's
    /// `overlay_cursor`.
    pub(crate) overlay_cursor: bool,
}

impl State {
    /// Current buffer constraints for capturing the named output, or
    /// `None` if it's gone. Shm is always offered (the render path's CPU
    /// read-back); dmabuf when a render node exists, restricted to the
    /// same fourcc screencopy advertises so both paths capture
    /// identically.
    fn copy_capture_constraints_for(&self, name: &str) -> Option<BufferConstraints> {
        let size = self.renderer.output_mode_size(name)?;
        let dma = self.renderer.render_drm_node().map(|node| DmabufConstraints {
            node,
            formats: vec![(
                CAPTURE_FOURCC,
                self.renderer
                    .dmabuf_formats()
                    .into_iter()
                    .filter(|f| f.code == CAPTURE_FOURCC)
                    .map(|f| f.modifier)
                    .collect(),
            )],
        });
        Some(BufferConstraints {
            size: smithay::utils::Size::from((size.w, size.h)),
            // Exactly screencopy's format: the read-back is Xrgb8888 and
            // write_to_shm does no conversion, so advertising Argb8888
            // would hand clients uninitialised alpha.
            shm: vec![CAPTURE_SHM_FORMAT],
            dma,
        })
    }

    /// Re-announce constraints to every live capture session — called
    /// after anything that changes an output's pixel size (mode set,
    /// hotplug reflow). A session whose output vanished is stopped
    /// (dropping the owned session sends `stopped` to the client).
    pub(crate) fn refresh_copy_capture_constraints(&mut self) {
        let mut keep = Vec::new();
        for session in std::mem::take(&mut self.capture_sessions) {
            let name = session
                .as_ref()
                .user_data()
                .get::<SessionOutput>()
                .map(|s| s.0.clone());
            if let Some(constraints) = name.and_then(|n| self.copy_capture_constraints_for(&n)) {
                // Re-announce only on an actual change: an unconditional
                // buffer_size/done volley on every hotplug event is the
                // same renegotiation-storm species as the dmabuf-feedback
                // mid-life switch that once black-screened games.
                if session.as_ref().current_constraints().map(|c| c.size) != Some(constraints.size)
                {
                    session.as_ref().update_constraints(constraints);
                }
                keep.push(session);
            } else {
                debug!("copy-capture: output gone; stopping session");
                drop(session);
            }
        }
        self.capture_sessions = keep;
    }

    /// Turn the copy frames due on `output` into render capture specs.
    /// Returns the drained frames (order-matched to the pushed specs) so
    /// the render result can complete them.
    pub(crate) fn drain_copy_frames(
        &mut self,
        output: &str,
        specs: &mut Vec<CaptureSpec>,
    ) -> Vec<PendingCopyFrame> {
        let Some(size) = self.renderer.output_mode_size(output) else {
            // Output vanished between request and render: fail the frames.
            let (gone, keep) = std::mem::take(&mut self.copy_frames)
                .into_iter()
                .partition(|p| p.output == output);
            self.copy_frames = keep;
            for p in gone {
                p.frame.fail(FailureReason::Stopped);
            }
            return Vec::new();
        };
        let region = Rectangle::from_size(size);
        let (mine, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.copy_frames)
            .into_iter()
            .partition(|p| p.output == output);
        self.copy_frames = keep;
        // smithay validates a frame's buffer against the constraints
        // SNAPSHOTTED at frame creation; the protocol demands a buffer
        // matching the LATEST constraints. A mode change between the two
        // leaves an old-size buffer here — blitting the new-size region
        // into it would silently crop (GLES clips out-of-bounds blits)
        // and report a wrong frame as ready. Fail those instead so the
        // client re-allocates.
        let mut due = Vec::new();
        for p in mine {
            let dims = smithay::backend::renderer::buffer_dimensions(&p.frame.buffer());
            if dims != Some(smithay::utils::Size::from((size.w, size.h))) {
                debug!("copy-capture: buffer predates a mode change; failing for re-allocation");
                p.frame.fail(FailureReason::BufferConstraints);
                continue;
            }
            let buffer = p.frame.buffer();
            specs.push(CaptureSpec {
                region,
                fourcc: CAPTURE_FOURCC,
                target: match smithay::wayland::dmabuf::get_dmabuf(&buffer) {
                    Ok(dmabuf) => CaptureTarget::Dmabuf(dmabuf.clone()),
                    Err(_) => CaptureTarget::Shm,
                },
            });
            due.push(p);
        }
        due
    }

    /// Deliver a render capture outcome to a copy frame. Mirrors
    /// `screencopy::complete`: dmabuf blits are done on the GPU already,
    /// shm outcomes carry the read-back bytes to copy into the client
    /// buffer. Damage is reported as full (we captured the whole output).
    pub(crate) fn complete_copy_frame(pending: PendingCopyFrame, outcome: CaptureOutcome) {
        let now = Clock::<Monotonic>::new().now();
        // Report FULL damage explicitly: `None` would send zero damage
        // events, which damage-tracking consumers read as "nothing
        // changed" (and the spec requires the first frame to carry full
        // damage).
        let full = smithay::backend::renderer::buffer_dimensions(&pending.frame.buffer())
            .map(|s| vec![Rectangle::from_size(smithay::utils::Size::from((s.w, s.h)))]);
        match outcome {
            CaptureOutcome::Dmabuf => {
                pending.frame.success(Transform::Normal, full, now);
            }
            CaptureOutcome::Shm {
                bytes,
                width,
                height,
            } => {
                if write_to_shm(&pending.frame.buffer(), &bytes, width, height) {
                    pending.frame.success(Transform::Normal, full, now);
                } else {
                    warn!("copy-capture: client shm buffer too small for the capture");
                    pending.frame.fail(FailureReason::BufferConstraints);
                }
            }
            CaptureOutcome::Failed => {
                pending.frame.fail(FailureReason::Unknown);
            }
        }
    }
}

impl ImageCaptureSourceHandler for State {
    // Sessions referencing a destroyed source stop through their own
    // lifecycle (`frame` checks source liveness; `session_destroyed`
    // prunes) — nothing extra to clean here.
}

impl OutputCaptureSourceHandler for State {
    fn output_capture_source_state(&mut self) -> &mut OutputCaptureSourceState {
        &mut self.output_capture_source_state
    }

    fn output_source_created(&mut self, source: ImageCaptureSource, output: &Output) {
        let name = output.name();
        debug!(output = %name, "copy-capture: output source created");
        source
            .user_data()
            .insert_if_missing(|| SourceOutput(name));
    }
}

impl ImageCopyCaptureHandler for State {
    fn image_copy_capture_state(&mut self) -> &mut ImageCopyCaptureState {
        &mut self.image_copy_capture_state
    }

    fn capture_constraints(&mut self, source: &ImageCaptureSource) -> Option<BufferConstraints> {
        if !source.alive() {
            return None;
        }
        let name = source.user_data().get::<SourceOutput>()?;
        self.copy_capture_constraints_for(&name.0.clone())
    }

    fn new_session(&mut self, session: Session) {
        let key = self.capture_session_key;
        self.capture_session_key += 1;
        session
            .as_ref()
            .user_data()
            .insert_if_missing(|| SessionKey(key));
        // Pin the output name to the SESSION: the client may legally
        // destroy the source object right after creating the session.
        if let Some(name) = session
            .as_ref()
            .source()
            .user_data()
            .get::<SourceOutput>()
            .map(|s| s.0.clone())
        {
            session
                .as_ref()
                .user_data()
                .insert_if_missing(|| SessionOutput(name));
        }
        debug!(key, "copy-capture: session started");
        self.capture_sessions.push(session);
    }

    fn frame(&mut self, session: &SessionRef, frame: Frame) {
        let Some(name) = session
            .user_data()
            .get::<SessionOutput>()
            .map(|s| s.0.clone())
        else {
            frame.fail(FailureReason::Stopped);
            return;
        };
        // A dmabuf we can't render INTO is failed gracefully so the
        // client retries with shm — smithay validated size/format
        // against the session constraints, but renderability (blit
        // target support, notably on NVIDIA) is ours to check. Same
        // policy as screencopy.
        if let Ok(dmabuf) = smithay::wayland::dmabuf::get_dmabuf(&frame.buffer()) {
            use smithay::backend::allocator::Buffer as _;
            if !self.renderer.can_render_to(dmabuf.format()) {
                debug!("copy-capture: dmabuf not renderable; failing so the client falls back");
                frame.fail(FailureReason::BufferConstraints);
                return;
            }
        }
        let overlay_cursor = session.draw_cursor();
        // Wake the captured output; an idle on-demand output would never
        // flip (and thus never capture) on its own. No CRTC means the
        // output is gone — fail NOW rather than parking a frame no render
        // will ever drain.
        let Some(crtc) = self.renderer.crtc_for_output_name(&name) else {
            frame.fail(FailureReason::Stopped);
            return;
        };
        self.copy_frames.push(PendingCopyFrame {
            output: name,
            frame,
            overlay_cursor,
        });
        self.queue_redraw(crtc);
    }

    fn frame_aborted(&mut self, frame: smithay::wayland::image_copy_capture::FrameRef) {
        // The client destroyed the buffer (or died) before the capture
        // rendered — drop the parked frame so the render path never
        // touches the dead buffer.
        self.copy_frames.retain(|p| p.frame != frame);
    }

    fn session_destroyed(&mut self, session: SessionRef) {
        let key = session.user_data().get::<SessionKey>().map(|k| k.0);
        self.capture_sessions.retain(|s| {
            s.as_ref().user_data().get::<SessionKey>().map(|k| k.0) != key
        });
        self.image_copy_capture_state.cleanup();
    }
}
