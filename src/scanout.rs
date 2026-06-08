//! Self-managed scanout surface: a swapchain on a [`DrmSurface`] that we
//! drive ourselves with atomic page-flips, replacing smithay's
//! `GbmBufferedSurface`.
//!
//! Targeted direct-scanout (Option C). The surface drives the primary
//! plane with a four-stage pipeline (next → queued → pending → current)
//! built on smithay's public primitives ([`Swapchain`], [`DrmSurface`]'s
//! `test_state`/`commit`/`page_flip`, [`framebuffer_from_bo`]).
//!
//! Each frame in the pipeline is a [`Frame`]: normally a compositor-rendered
//! [`Swapchain`] buffer (`Frame::Composite`), but for a single fullscreen
//! opaque client whose colour mode matches the output we latch the client's
//! own buffer straight onto the primary plane (`Frame::Direct`) — zero
//! compositing, the whole point of Stage 2. A direct frame holds the
//! client's [`ClientBuffer`] keep-alive so `wl_buffer.release` only fires
//! once a later flip replaces it on its vblank, and the imported KMS
//! framebuffer is cached per [`WlBuffer`] so re-scanning a cycled buffer
//! doesn't re-import.
//!
//! ## The pipeline
//!
//! At most one frame sits in each of four roles:
//! - `next`    — a composite buffer acquired by [`ScanoutSurface::next_buffer`] for the renderer.
//! - `queued`  — a frame queued for scanout while a flip is already in flight.
//! - `pending` — its page-flip/commit is submitted to KMS; awaiting vblank.
//! - `current` — currently scanned out (the front frame).
//!
//! A flip is issued only when nothing is pending; otherwise the frame parks
//! in `queued` and [`ScanoutSurface::frame_submitted`] (on the vblank)
//! drains it. Dropping the old `current` frame on vblank releases it — a
//! `Composite` slot back to the swapchain, a `Direct` frame's client buffer
//! back to the client (via `wl_buffer.release`).

use std::os::unix::io::AsFd;
use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow, ensure};
use smithay::backend::allocator::dmabuf::{AsDmabuf as _, Dmabuf};
use smithay::backend::allocator::format::get_opaque;
use smithay::backend::allocator::gbm::{GbmAllocator, GbmBuffer, GbmDevice};
use smithay::backend::allocator::{Buffer as _, Format, Fourcc, Modifier, Slot, Swapchain};
use smithay::backend::drm::gbm::{
    GbmFramebuffer, framebuffer_from_bo, framebuffer_from_dmabuf,
};
use smithay::backend::drm::{
    DrmDeviceFd, DrmSurface, PlaneConfig, PlaneDamageClips, PlaneState, VrrSupport,
};
use smithay::backend::renderer::sync::SyncPoint;
use smithay::backend::renderer::utils::Buffer as ClientBuffer;
use smithay::reexports::drm::Device as _;
use smithay::reexports::drm::DriverCapability;
use smithay::reexports::drm::control::{Device as ControlDevice, connector, framebuffer, plane};
use smithay::reexports::wayland_server::Weak;
use smithay::reexports::wayland_server::Resource as _;
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::utils::{Physical, Rectangle, Transform};
use tracing::{debug, warn};

/// One frame in the present pipeline.
enum Frame {
    /// A compositor-rendered swapchain buffer. Dropping the slot releases it
    /// back to the swapchain.
    Composite(Slot<GbmBuffer>),
    /// A client buffer scanned out directly on the primary plane. Holds the
    /// `wl_buffer` keep-alive (dropping it sends `wl_buffer.release`) and the
    /// imported KMS framebuffer (shared with the FB cache; `rmfb` on last drop).
    Direct {
        #[allow(dead_code, reason = "kept alive to gate wl_buffer.release until replaced")]
        buffer: ClientBuffer,
        fb: Arc<GbmFramebuffer>,
    },
}

impl Frame {
    /// The KMS framebuffer handle to put on the plane for this frame.
    fn fb_handle(&self) -> framebuffer::Handle {
        match self {
            Frame::Composite(slot) => handle_of(
                slot.userdata()
                    .get::<GbmFramebuffer>()
                    .expect("composite slot carries its cached framebuffer"),
            ),
            Frame::Direct { fb, .. } => handle_of(fb),
        }
    }
}

/// Extract the raw KMS framebuffer handle from a [`GbmFramebuffer`]. Takes
/// `&GbmFramebuffer` (an `&Arc<GbmFramebuffer>` coerces in) so it works for
/// both pipeline frame kinds.
fn handle_of(fb: &GbmFramebuffer) -> framebuffer::Handle {
    *fb.as_ref()
}

/// A frame queued for scanout while a previous flip is in flight.
struct QueuedFrame {
    frame: Frame,
    /// GPU completion fence for a composited frame; `None` for a direct frame
    /// (client buffers use implicit sync via the dmabuf's own fence).
    sync: Option<SyncPoint>,
    /// Optional `FB_DAMAGE_CLIPS` damage (compositor currently passes `None`).
    damage: Option<Vec<Rectangle<i32, Physical>>>,
}

/// A cached KMS framebuffer for a client buffer, keyed by a weak ref so it is
/// evicted (and `rmfb`'d) once the client destroys the buffer.
struct ClientFb {
    buffer: Weak<WlBuffer>,
    use_opaque: bool,
    fb: Arc<GbmFramebuffer>,
}

/// A swapchain bound to a [`DrmSurface`]'s primary plane that we flip
/// ourselves, with a direct-scanout fast path for fullscreen clients.
pub struct ScanoutSurface {
    /// Front frame (currently scanned out). Always present.
    current: Frame,
    /// Frame whose flip is submitted to KMS, awaiting its vblank.
    pending: Option<Frame>,
    /// Rendered/latched frame waiting because a flip is already in flight.
    queued: Option<QueuedFrame>,
    /// Composite buffer handed to the renderer to draw into (back buffer).
    next_fb: Option<Slot<GbmBuffer>>,
    swapchain: Swapchain<GbmAllocator<DrmDeviceFd>>,
    /// Retained allocator clone, used to reach the GBM device for importing
    /// client dmabufs as scanout framebuffers (direct-scanout fast path).
    allocator: GbmAllocator<DrmDeviceFd>,
    drm: Arc<DrmSurface>,
    /// Whether our composite framebuffers use the opaque sibling fourcc.
    is_opaque: bool,
    /// Whether explicit `IN_FENCE_FD` sync may be used on this plane.
    supports_fencing: bool,
    /// Imported framebuffers for client buffers scanned out directly.
    fb_cache: Vec<ClientFb>,
}

impl ScanoutSurface {
    /// Build a scanout surface, trying `color_formats` in order until one is
    /// accepted by both the plane and the renderer (and passes a real KMS
    /// test commit). `renderer_formats` are the dmabuf formats the GLES
    /// renderer can render into; the swapchain is negotiated to their
    /// intersection with the primary plane's formats.
    pub fn new(
        drm: DrmSurface,
        allocator: &GbmAllocator<DrmDeviceFd>,
        color_formats: &[Fourcc],
        renderer_formats: impl IntoIterator<Item = Format>,
    ) -> Result<Self> {
        let drm = Arc::new(drm);
        let renderer_formats = renderer_formats.into_iter().collect::<Vec<_>>();

        let mut last_err = None;
        for &code in color_formats {
            debug!(?code, "testing scanout color format");
            // GbmAllocator is Arc-backed and cheaply cloneable, so each
            // attempt gets its own clone (smithay instead moves the
            // allocator through and recovers it on failure — same effect).
            match Self::new_internal(&drm, allocator.clone(), &renderer_formats, code) {
                Ok((current_fb, swapchain, is_opaque)) => {
                    // Explicit IN_FENCE sync requires an atomic surface whose
                    // driver supports SyncObj and whose primary plane exposes
                    // the IN_FENCE_FD property. Otherwise we fall back to a
                    // CPU wait on the render fence before flipping.
                    let supports_fencing = !drm.is_legacy()
                        && drm
                            .device_fd()
                            .get_driver_capability(DriverCapability::SyncObj)
                            .map(|val| val != 0)
                            .context("query SyncObj driver capability")?
                        && plane_has_property(drm.device_fd(), drm.plane(), "IN_FENCE_FD")?;

                    return Ok(Self {
                        current: Frame::Composite(current_fb),
                        pending: None,
                        queued: None,
                        next_fb: None,
                        swapchain,
                        allocator: allocator.clone(),
                        drm,
                        is_opaque,
                        supports_fencing,
                        fb_cache: Vec::new(),
                    });
                }
                Err(err) => {
                    warn!(?code, error = %err, "scanout format not usable; trying next");
                    last_err = Some(err);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("no scanout color formats provided")))
    }

    /// Negotiate format/modifiers for one candidate `code`, build the
    /// swapchain, allocate a test buffer and validate it with a KMS test
    /// commit (modeset allowed). Mirrors `GbmBufferedSurface::new_internal`.
    fn new_internal(
        drm: &Arc<DrmSurface>,
        allocator: GbmAllocator<DrmDeviceFd>,
        renderer_formats: &[Format],
        code: Fourcc,
    ) -> Result<(Slot<GbmBuffer>, Swapchain<GbmAllocator<DrmDeviceFd>>, bool)> {
        // Restrict both sides to the requested fourcc (or its opaque
        // sibling, which a plane may advertise instead of an alpha format).
        let opaque_code = get_opaque(code).unwrap_or(code);
        let plane_formats = drm
            .plane_info()
            .formats
            .iter()
            .copied()
            .filter(|fmt| fmt.code == code || fmt.code == opaque_code)
            .collect::<Vec<Format>>();
        let renderer_formats = renderer_formats
            .iter()
            .copied()
            .filter(|fmt| fmt.code == code)
            .collect::<Vec<Format>>();

        ensure!(
            !plane_formats.is_empty(),
            "plane advertises no format compatible with {code:?}"
        );
        ensure!(
            !renderer_formats.is_empty(),
            "renderer cannot render into {code:?}"
        );

        let plane_modifiers = dedup_modifiers(plane_formats.iter().map(|f| f.modifier));
        let renderer_modifiers = dedup_modifiers(renderer_formats.iter().map(|f| f.modifier));

        // Special case (from smithay): if one side advertises only implicit
        // (Invalid) modifiers while the other supports explicit LINEAR, force
        // an implicit modifier so allocation still works (likely linear).
        let force_implicit = (plane_formats.len() == 1
            && plane_formats[0].modifier == Modifier::Invalid
            && renderer_formats.iter().all(|x| x.modifier != Modifier::Invalid)
            && renderer_formats.iter().any(|x| x.modifier == Modifier::Linear))
            || (renderer_formats.len() == 1
                && renderer_formats[0].modifier == Modifier::Invalid
                && plane_formats.iter().all(|x| x.modifier != Modifier::Invalid)
                && plane_formats.iter().any(|x| x.modifier == Modifier::Linear));

        let modifiers: Vec<Modifier> = if force_implicit {
            vec![Modifier::Invalid]
        } else {
            // Intersection, preserving the plane's preference order.
            plane_modifiers
                .iter()
                .copied()
                .filter(|m| renderer_modifiers.contains(m))
                .collect()
        };
        debug!(?code, ?modifiers, "negotiated scanout modifiers");

        let (w, h) = drm.pending_mode().size();
        let mut swapchain =
            Swapchain::new(allocator, u32::from(w), u32::from(h), code, modifiers);

        // Allocate one buffer and prove the whole pipeline: dmabuf export,
        // framebuffer creation, and a KMS test commit with it on the plane.
        let buffer = swapchain
            .acquire()
            .context("allocate test scanout buffer")?
            .context("swapchain returned no buffer for test allocation")?;

        // A plane that only advertised the opaque sibling needs the
        // framebuffer built from the opaque format (alpha ignored on scanout).
        let use_opaque = !plane_formats.iter().any(|f| f.code == code);
        let fb = framebuffer_from_bo(drm.device_fd(), &buffer, use_opaque)
            .map_err(|err| anyhow!("create framebuffer for test buffer: {err}"))?;
        // Validate the buffer can be exported as a dmabuf (the renderer path).
        buffer.export().context("export test buffer as dmabuf")?;
        buffer.userdata().insert_if_missing(|| fb);
        let handle = buffer
            .userdata()
            .get::<GbmFramebuffer>()
            .expect("framebuffer just inserted into slot userdata");

        let plane_state = PlaneState {
            handle: drm.plane(),
            config: Some(PlaneConfig {
                src: Rectangle::from_size((i32::from(w), i32::from(h)).into()).to_f64(),
                dst: Rectangle::from_size((i32::from(w), i32::from(h)).into()),
                alpha: 1.0,
                transform: Transform::Normal,
                damage_clips: None,
                fb: *handle.as_ref(),
                fence: None,
            }),
        };

        drm.test_state([plane_state], true)
            .map_err(|err| anyhow!("KMS test commit rejected {code:?}: {err}"))?;
        Ok((buffer, swapchain, use_opaque))
    }

    /// Acquire the next composite buffer for the renderer to draw into, plus
    /// its buffer age. Idempotent: returns the same buffer until it is queued.
    pub fn next_buffer(&mut self) -> Result<(Dmabuf, u8)> {
        ensure!(self.drm.is_active(), "DRM device is inactive");

        if self.next_fb.is_none() {
            let slot = self
                .swapchain
                .acquire()
                .context("acquire swapchain buffer")?
                .context("swapchain exhausted (no free buffers)")?;

            // Cache the scanout framebuffer in the slot's userdata so reusing
            // this buffer next frame doesn't re-create (and re-rmfb) it.
            if slot.userdata().get::<GbmFramebuffer>().is_none() {
                let fb = framebuffer_from_bo(self.drm.device_fd(), &slot, self.is_opaque)
                    .map_err(|err| anyhow!("create scanout framebuffer: {err}"))?;
                slot.userdata().insert_if_missing(|| fb);
            }

            self.next_fb = Some(slot);
        }

        let slot = self.next_fb.as_ref().expect("next_fb just set");
        Ok((slot.export().context("export buffer as dmabuf")?, slot.age()))
    }

    /// Queue the composite buffer last returned by [`Self::next_buffer`] for
    /// scanout, with an optional GPU completion fence and damage. If no flip
    /// is in flight it is submitted immediately; otherwise it waits for the
    /// next [`Self::frame_submitted`].
    pub fn queue_buffer(
        &mut self,
        sync: Option<SyncPoint>,
        damage: Option<Vec<Rectangle<i32, Physical>>>,
    ) -> Result<()> {
        ensure!(self.drm.is_active(), "DRM device is inactive");

        let next_fb = self
            .next_fb
            .take()
            .context("queue_buffer called before next_buffer")?;

        // Update buffer ages now, at queue time (matches smithay), so the
        // next acquire sees correct damage history.
        self.swapchain.submitted(&next_fb);

        self.queued = Some(QueuedFrame {
            frame: Frame::Composite(next_fb),
            sync,
            damage,
        });
        if self.pending.is_none() {
            self.submit()?;
        }
        Ok(())
    }

    /// Try to scan a client buffer straight onto the primary plane (direct
    /// scanout), skipping compositing entirely. The caller has already
    /// verified the frame is geometrically eligible (one settled fullscreen
    /// opaque client, colour mode matched, 1:1 with the mode).
    ///
    /// Returns `Ok(true)` when the buffer was latched and flipped; `Ok(false)`
    /// when it isn't scannable (implicit modifier, un-importable, or rejected
    /// by the plane) and the caller must fall back to compositing this frame.
    /// `Err` is a real failure (inactive device, flip error).
    ///
    /// `use_opaque` requests the opaque sibling fourcc for the plane (when the
    /// client buffer carries an unused alpha channel). On success ownership of
    /// `buffer` (the `wl_buffer` keep-alive) is taken so `wl_buffer.release`
    /// is deferred until a later flip replaces it.
    pub fn try_queue_external(
        &mut self,
        buffer: ClientBuffer,
        dmabuf: &Dmabuf,
        use_opaque: bool,
    ) -> Result<bool> {
        ensure!(self.drm.is_active(), "DRM device is inactive");

        // A legacy (non-atomic) surface can't reliably test a foreign FB.
        if self.drm.is_legacy() {
            return Ok(false);
        }
        // KMS can't safely scan out a buffer allocated with an implicit
        // (Invalid) modifier — its tiling/layout is unknown (the Weston rule).
        if dmabuf.format().modifier == Modifier::Invalid {
            return Ok(false);
        }

        let fb = match self.import_client_fb(&buffer, dmabuf, use_opaque) {
            Ok(fb) => fb,
            Err(err) => {
                debug!(error = %err, "client buffer not importable for scanout; compositing");
                return Ok(false);
            }
        };

        // The caller guaranteed the buffer is 1:1 with the mode, so src == dst.
        let (w, h) = self.drm.pending_mode().size();
        let src = Rectangle::from_size((i32::from(w), i32::from(h)).into()).to_f64();
        let dst = Rectangle::from_size((i32::from(w), i32::from(h)).into());
        let plane_state = PlaneState {
            handle: self.drm.plane(),
            config: Some(PlaneConfig {
                src,
                dst,
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: handle_of(&fb),
                // Implicit sync: the kernel waits on the dmabuf's own fence.
                fence: None,
            }),
        };

        // Authoritative gate: ask the driver whether it can actually scan this
        // out. Match the flip's modeset-ness (VRR/mode change → commit).
        let allow_modeset = self.drm.commit_pending();
        if let Err(err) = self.drm.test_state([plane_state], allow_modeset) {
            debug!(error = %err, "primary plane rejected client buffer; compositing");
            return Ok(false);
        }

        // Committed to direct scanout: queue the client frame and flip it.
        self.queued = Some(QueuedFrame {
            frame: Frame::Direct { buffer, fb },
            sync: None,
            damage: None,
        });
        if self.pending.is_none() {
            self.submit()?;
        }
        Ok(true)
    }

    /// Import a client `dmabuf` as a scanout framebuffer, caching it per
    /// `wl_buffer` so a cycled buffer isn't re-imported each frame.
    fn import_client_fb(
        &mut self,
        buffer: &ClientBuffer,
        dmabuf: &Dmabuf,
        use_opaque: bool,
    ) -> Result<Arc<GbmFramebuffer>, smithay::backend::drm::gbm::Error> {
        let wl: &WlBuffer = buffer;
        let weak = wl.downgrade();
        if let Some(entry) = self
            .fb_cache
            .iter()
            .find(|e| e.use_opaque == use_opaque && e.buffer == weak)
        {
            return Ok(entry.fb.clone());
        }

        // `framebuffer_from_dmabuf` imports the dmabuf into our GBM device and
        // adds a scanout framebuffer (addfb2) in one call.
        let gbm: &GbmDevice<DrmDeviceFd> = self.allocator.as_ref();
        let fb = Arc::new(framebuffer_from_dmabuf(
            self.drm.device_fd(),
            gbm,
            dmabuf,
            use_opaque,
            false,
        )?);
        self.fb_cache.push(ClientFb {
            buffer: weak,
            use_opaque,
            fb: fb.clone(),
        });
        Ok(fb)
    }

    /// Acknowledge the vblank for the in-flight flip: promote the pending
    /// frame to current (releasing the old front frame — a swapchain slot, or
    /// a client buffer via `wl_buffer.release`) and submit any queued frame.
    /// Must be called once per vblank after a flip was queued.
    pub fn frame_submitted(&mut self) -> Result<()> {
        if let Some(mut pending) = self.pending.take() {
            std::mem::swap(&mut pending, &mut self.current);
            // A frame may have queued while this flip was in flight.
            if self.queued.is_some() {
                self.submit()?;
            }
            // `pending` now holds the old front frame; dropping it here (after
            // the submit, matching smithay) releases it.
        }
        // Drop cached framebuffers for client buffers the client has destroyed.
        self.fb_cache.retain(|e| e.buffer.is_alive());
        Ok(())
    }

    /// Build the plane state for the queued frame and issue the atomic flip —
    /// a full `commit` (modeset) when state is pending (first frame,
    /// mode/VRR/HDR change), otherwise a plain `page_flip`.
    fn submit(&mut self) -> Result<()> {
        let QueuedFrame { frame, sync, damage } =
            self.queued.take().expect("submit called with a queued frame");
        let fb = frame.fb_handle();

        let (w, h) = self.drm.pending_mode().size();
        let src = Rectangle::from_size((i32::from(w), i32::from(h)).into()).to_f64();
        let dst = Rectangle::from_size((i32::from(w), i32::from(h)).into());

        let damage_clips = damage.and_then(|damage| {
            PlaneDamageClips::from_damage(self.drm.device_fd(), src, dst, damage)
                .ok()
                .flatten()
        });

        // Explicit sync: hand the render fence to KMS as IN_FENCE_FD when the
        // plane supports it; otherwise (or if the fence isn't exportable)
        // block on the GPU here and rely on implicit sync. Direct frames carry
        // no SyncPoint (implicit sync via the dmabuf's own fence).
        let fence = match sync {
            Some(sync) if self.supports_fencing => {
                let fence = sync.export();
                if fence.is_none() {
                    let _ = sync.wait();
                }
                fence
            }
            Some(sync) => {
                let _ = sync.wait();
                None
            }
            None => None,
        };

        let plane_state = PlaneState {
            handle: self.drm.plane(),
            config: Some(PlaneConfig {
                src,
                dst,
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: damage_clips.as_ref().map(PlaneDamageClips::blob),
                fb,
                fence: fence.as_ref().map(AsFd::as_fd),
            }),
        };

        let flip = if self.drm.commit_pending() {
            self.drm.commit([plane_state], true)
        } else {
            self.drm.page_flip([plane_state], true)
        };
        if flip.is_ok() {
            self.pending = Some(frame);
        }
        flip.context("atomic page-flip/commit failed")
    }

    /// The swapchain's scanout fourcc.
    pub fn format(&self) -> Fourcc {
        self.swapchain.format()
    }

    /// Whether the connector advertises adaptive-sync (VRR) support.
    pub fn vrr_supported(&self, conn: connector::Handle) -> Result<VrrSupport> {
        self.drm.vrr_supported(conn).context("query VRR support")
    }

    /// Whether the next frame's state would have VRR enabled.
    pub fn vrr_enabled(&self) -> bool {
        self.drm.vrr_enabled()
    }

    /// Request VRR (may force the next frame to be a modeset commit).
    pub fn use_vrr(&self, vrr: bool) -> Result<()> {
        self.drm.use_vrr(vrr).context("set VRR state")
    }

    /// The underlying [`DrmSurface`] (for HDR connector metadata staging).
    pub fn surface(&self) -> &DrmSurface {
        &self.drm
    }
}

/// Collect a modifier sequence preserving first-seen order and dropping
/// duplicates (an order-preserving set, avoiding an `IndexSet` dependency).
fn dedup_modifiers(modifiers: impl IntoIterator<Item = Modifier>) -> Vec<Modifier> {
    let mut out: Vec<Modifier> = Vec::new();
    for m in modifiers {
        if !out.contains(&m) {
            out.push(m);
        }
    }
    out
}

/// Whether `plane` exposes a property named `name`. Replicates smithay's
/// private `plane_has_property` (used here to detect `IN_FENCE_FD` support).
fn plane_has_property(dev: &DrmDeviceFd, plane: plane::Handle, name: &str) -> Result<bool> {
    let props = dev
        .get_properties(plane)
        .context("get properties of primary plane")?;
    let (ids, _values) = props.as_props_and_values();
    for &id in ids {
        let info = dev.get_property(id).context("get plane property info")?;
        if info.name().to_str().is_ok_and(|n| n == name) {
            return Ok(true);
        }
    }
    Ok(false)
}
