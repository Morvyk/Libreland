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

use std::os::fd::BorrowedFd;
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
    DrmDeviceFd, DrmSurface, PlaneClaim, PlaneConfig, PlaneDamageClips, PlaneInfo, PlaneState,
    VrrSupport,
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

/// One frame in the present pipeline: everything on every plane for one flip.
///
/// The primary layer is always present. Overlay layers appear when the scene
/// is a fullscreen client with a *little* on top of it — a notification, a
/// popup, an oversized cursor — which the hardware can composite for free
/// instead of us doing a full GPU pass to draw the same thing.
struct Frame {
    primary: Layer,
    /// Content on overlay planes, bottom-up. Empty for a composited frame,
    /// which by definition already has everything in its one buffer.
    overlays: Vec<(plane::Handle, Layer)>,
}

/// What one plane shows for one flip, and what keeps it alive that long.
struct Layer {
    /// The buffer's owner. A composited frame owns a swapchain slot (dropping
    /// it returns the buffer); a direct one owns the client's `wl_buffer`
    /// keep-alive (dropping it sends `wl_buffer.release`).
    hold: Hold,
    placement: Placement,
    /// `FB_DAMAGE_CLIPS` for this plane, in buffer pixels. `None` = all of it.
    damage: Option<Vec<Rectangle<i32, Physical>>>,
    /// Per-plane blend factor. Always `1.0` today — the field exists because
    /// `PlaneConfig` demands one, and an overlay plane is where a future
    /// hardware-accelerated window opacity would apply it.
    alpha: f32,
}

/// What a [`Layer`] holds to keep its framebuffer valid until a later flip
/// replaces it.
enum Hold {
    /// A compositor-rendered swapchain buffer; its framebuffer lives in the
    /// slot's userdata.
    Composite(Slot<GbmBuffer>),
    /// A client buffer, with the framebuffer we imported for it (shared with
    /// [`ScanoutSurface::fb_cache`]; `rmfb` on last drop).
    Client {
        #[allow(dead_code, reason = "kept alive to gate wl_buffer.release until replaced")]
        buffer: ClientBuffer,
        fb: Arc<GbmFramebuffer>,
    },
}

impl Layer {
    fn fb(&self) -> framebuffer::Handle {
        match &self.hold {
            Hold::Composite(slot) => *slot
                .userdata()
                .get::<GbmFramebuffer>()
                .expect("composite slot carries its cached framebuffer")
                .as_ref(),
            Hold::Client { fb, .. } => *fb.as_ref().as_ref(),
        }
    }

    fn plane_state<'a>(
        &self,
        handle: plane::Handle,
        clips: Option<&'a PlaneDamageClips>,
        fence: Option<BorrowedFd<'a>>,
    ) -> PlaneState<'a> {
        PlaneState {
            handle,
            config: Some(PlaneConfig {
                src: self.placement.src,
                dst: self.placement.dst,
                transform: self.placement.transform,
                alpha: self.alpha,
                damage_clips: clips.map(PlaneDamageClips::blob),
                fb: self.fb(),
                fence,
            }),
        }
    }
}

impl Frame {
    /// Whether this frame puts a client's own buffer on the primary plane.
    fn is_direct(&self) -> bool {
        matches!(self.primary.hold, Hold::Client { .. })
    }

    /// Every layer, primary first — the order plane states and damage clips
    /// are built in.
    fn layers(&self) -> impl Iterator<Item = &Layer> {
        std::iter::once(&self.primary).chain(self.overlays.iter().map(|(_, l)| l))
    }

    /// The overlay planes this frame drives, so a later frame knows which
    /// ones it has to switch back off.
    fn overlay_planes(&self) -> Vec<plane::Handle> {
        self.overlays.iter().map(|(h, _)| *h).collect()
    }
}

/// A frame queued for scanout while a previous flip is in flight.
struct QueuedFrame {
    frame: Frame,
    /// GPU completion fence for a composited frame, or a direct frame's
    /// exported explicit-sync acquire fence. `None` means the buffer carries
    /// an implicit fence the kernel will wait on by itself.
    sync: Option<FrameFence>,
}

/// The GPU work a frame must wait on before it may be scanned out.
///
/// Only ever our own render fence. A *client's* explicit-sync acquire point
/// never reaches here: the commit that carried it is gated on that fence
/// before the buffer is visible to us at all (see the pre-commit hook in
/// [`crate::wayland`]), so by the time a buffer can be latched its GPU work
/// is already done.
enum FrameFence {
    Render(SyncPoint),
}

/// Where a frame's pixels land on the primary plane.
#[derive(Clone, Copy, PartialEq)]
struct Placement {
    /// Source rectangle within the buffer, in buffer pixels.
    src: Rectangle<f64, smithay::utils::Buffer>,
    /// Destination rectangle on the CRTC, in physical pixels.
    dst: Rectangle<i32, Physical>,
    /// Plane rotation/reflection applied between the two.
    transform: Transform,
}

/// The client-side half of a direct-scanout placement: how the client's
/// buffer should be mapped onto the plane. Produced by the renderer's
/// eligibility check, which has the surface's view and transform to hand.
#[derive(Clone, Copy, Debug)]
pub struct DirectPlacement {
    /// Source rectangle within the client buffer, in buffer pixels. Lets a
    /// viewport-cropped client (fractional scaling, Xwayland under a client
    /// scale) scan out instead of falling back to compositing.
    pub src: Rectangle<f64, smithay::utils::Buffer>,
    /// Where it goes on the CRTC, in physical pixels. The whole mode for the
    /// primary layer; an overlay's own rect otherwise.
    pub dst: Rectangle<i32, Physical>,
    /// Buffer transform to undo on the plane. `Normal` for almost everything;
    /// a rotated output's client may hand us a pre-rotated buffer, and most
    /// planes expose a `rotation` property that can absorb it.
    pub transform: Transform,
}

/// One client buffer the renderer wants on a plane this frame.
///
/// The primary layer is the fullscreen window; overlay layers are whatever
/// little is drawn above it. Handing them over together lets the whole frame
/// be validated and flipped as one atomic commit — either the hardware takes
/// the lot, or we composite the lot.
pub struct ScanoutLayer {
    /// Keep-alive for the client buffer; holding it defers
    /// `wl_buffer.release` until a later flip replaces this buffer.
    pub buffer: ClientBuffer,
    pub dmabuf: Dmabuf,
    pub place: DirectPlacement,
    /// The client's own damage since what this plane currently shows, in
    /// buffer pixels, or `None` for "assume all of it changed".
    pub damage: Option<Vec<Rectangle<i32, Physical>>>,
}

/// A cached KMS framebuffer for a client buffer, keyed by a weak ref so it is
/// evicted (and `rmfb`'d) once the client destroys the buffer.
struct ClientFb {
    buffer: Weak<WlBuffer>,
    use_opaque: bool,
    fb: Arc<GbmFramebuffer>,
}

/// The plane configuration a direct-scanout `test_state` verdict applies to.
///
/// The driver's answer is a function of the buffer's format and the geometry
/// we ask it to put on the plane — never of the framebuffer *handle*. A game
/// cycling a swapchain re-asks the identical question every frame, so once a
/// probe is accepted we can skip the `TEST_ONLY` atomic ioctl until something
/// in the key actually changes. `allow_modeset` is part of the key because a
/// modeset commit is a materially different request, and it flips back to
/// `false` on the frame after a mode/VRR change — which re-probes exactly
/// when the new mode needs validating.
#[derive(Clone, PartialEq)]
struct DirectProbe {
    /// One entry per plane, primary first. A frame that gains or loses an
    /// overlay is a different request and re-probes.
    layers: Vec<ProbeLayer>,
    allow_modeset: bool,
}

#[derive(Clone, Copy, PartialEq)]
struct ProbeLayer {
    plane: plane::Handle,
    code: Fourcc,
    modifier: Modifier,
    placement: Placement,
    use_opaque: bool,
}

/// An overlay plane we may put content on, claimed for this CRTC's exclusive
/// use for as long as the surface lives.
struct OverlayPlane {
    info: PlaneInfo,
    #[allow(dead_code, reason = "the claim's lifetime is the point; it is never read")]
    claim: PlaneClaim,
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
    /// Overlay planes usable for content *above* the primary one, ordered
    /// bottom-up by the z-position the driver gives them. Claimed at
    /// construction so no other CRTC can take them mid-session.
    overlays: Vec<OverlayPlane>,
    /// Overlay planes the last submitted commit turned on. A later frame that
    /// doesn't use one must explicitly switch it off, or the hardware keeps
    /// scanning stale content over the new frame.
    lit_overlays: Vec<plane::Handle>,
    /// Last direct-scanout plane configuration the driver accepted, so an
    /// unchanged one skips its per-frame `test_state`. Cleared whenever a
    /// flip fails, so a bad guess costs one frame and re-probes.
    last_probe: Option<DirectProbe>,
    /// Whether the *next* flip should be an async (tearing) one. Set per
    /// frame by the renderer from the config plus the focused client's
    /// `wp_tearing_control_v1` hint; see [`Self::set_tearing`].
    tearing: bool,
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

                    let overlays = claim_overlay_planes(&drm);
                    let (w, h) = drm.pending_mode().size();
                    let placement = full_placement(w, h);
                    return Ok(Self {
                        current: Frame {
                            primary: Layer {
                                hold: Hold::Composite(current_fb),
                                placement,
                                damage: None,
                                alpha: 1.0,
                            },
                            overlays: Vec::new(),
                        },
                        pending: None,
                        queued: None,
                        next_fb: None,
                        swapchain,
                        allocator: allocator.clone(),
                        drm,
                        is_opaque,
                        supports_fencing,
                        fb_cache: Vec::new(),
                        overlays,
                        lit_overlays: Vec::new(),
                        last_probe: None,
                        tearing: false,
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

    /// The primary plane's supported `(fourcc, modifier)` pairs — the set a
    /// client buffer must land in to be directly scannable on this output.
    /// Feeds the per-surface dmabuf-feedback scanout tranche (see
    /// wayland.rs) so fullscreen clients allocate plane-compatible buffers.
    pub fn plane_formats(&self) -> Vec<Format> {
        self.drm.plane_info().formats.iter().copied().collect()
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

        let (w, h) = self.drm.pending_mode().size();
        self.queued = Some(QueuedFrame {
            frame: Frame {
                primary: Layer {
                    hold: Hold::Composite(next_fb),
                    placement: full_placement(w, h),
                    damage,
                    alpha: 1.0,
                },
                // A composited frame already contains everything, so every
                // overlay plane goes dark — `submit` emits the disables.
                overlays: Vec::new(),
            },
            sync: sync.map(FrameFence::Render),
        });
        if self.pending.is_none() {
            self.submit()?;
        }
        Ok(())
    }

    /// How many overlay planes are free for content above a direct-scanned
    /// window. `0` means the renderer must composite anything drawn on top.
    pub fn overlay_capacity(&self) -> usize {
        self.overlays.len()
    }

    /// Try to scan client buffers straight onto the hardware planes, skipping
    /// compositing entirely: `primary` onto the primary plane, and each of
    /// `overlays` (bottom-up) onto an overlay plane above it.
    ///
    /// The caller has already verified the frame is geometrically eligible —
    /// one settled fullscreen opaque client covering the output, colour mode
    /// matched, and at most [`Self::overlay_capacity`] things drawn over it.
    ///
    /// Returns `Ok(true)` when the whole frame was latched and flipped;
    /// `Ok(false)` when any layer isn't scannable (implicit modifier,
    /// un-importable, or rejected by the driver) and the caller must fall back
    /// to compositing *the entire frame*. It is all-or-nothing on purpose: a
    /// half-assigned frame would show the game with its notification missing.
    /// `Err` is a real failure (inactive device, flip error).
    ///
    /// On success ownership of each layer's `buffer` keep-alive is taken, so
    /// `wl_buffer.release` is deferred until a later flip replaces it.
    pub fn try_queue_direct(
        &mut self,
        primary: ScanoutLayer,
        overlays: Vec<ScanoutLayer>,
    ) -> Result<bool> {
        ensure!(self.drm.is_active(), "DRM device is inactive");

        // A legacy (non-atomic) surface can't reliably test a foreign FB, and
        // has no overlay planes to speak of.
        if self.drm.is_legacy() {
            debug!("legacy DRM surface can't test client buffers; compositing");
            return Ok(false);
        }
        if overlays.len() > self.overlays.len() {
            debug!(
                wanted = overlays.len(),
                have = self.overlays.len(),
                "more content above the window than there are overlay planes; compositing"
            );
            return Ok(false);
        }

        // An acquire fence is only usable where the plane exposes IN_FENCE_FD;
        // without that the commit blocker upstream has already waited for us.
        // Build every layer before touching any state, so a rejection part
        // way through leaves nothing half-applied.
        let planes: Vec<plane::Handle> = std::iter::once(self.drm.plane())
            .chain(self.overlays.iter().take(overlays.len()).map(|o| o.info.handle))
            .collect();
        let mut built = Vec::with_capacity(planes.len());
        for (plane, layer) in planes.iter().zip(std::iter::once(primary).chain(overlays)) {
            match self.build_layer(*plane, layer) {
                Some(built_layer) => built.push(built_layer),
                None => return Ok(false),
            }
        }

        // Authoritative gate: ask the driver whether it can actually scan this
        // out. Match the flip's modeset-ness (VRR/mode change → commit). The
        // question is identical frame after frame while a game holds the
        // plane, so an accepted probe short-circuits the ioctl.
        let allow_modeset = self.drm.commit_pending();
        let probe = DirectProbe {
            layers: built.iter().map(|(_, _, p)| *p).collect(),
            allow_modeset,
        };
        if self.last_probe.as_ref() != Some(&probe) {
            let mut states: Vec<PlaneState<'_>> = built
                .iter()
                .map(|(plane, layer, _)| layer.plane_state(*plane, None, None))
                .collect();
            states.extend(self.dark_overlay_states(&planes[1..]));
            if let Err(err) = self.drm.test_state(states, allow_modeset) {
                debug!(
                    error = %err,
                    overlays = built.len() - 1,
                    "driver rejected the direct-scanout plane set; compositing"
                );
                // Don't cache rejections: the reason may be transient (a
                // mode change mid-settle), and re-probing a rejected config
                // costs nothing — we're compositing that frame regardless.
                self.last_probe = None;
                return Ok(false);
            }
            self.last_probe = Some(probe);
        }

        // Committed to direct scanout: queue the client frame and flip it.
        let mut built = built.into_iter();
        let (_, primary_layer, _) = built.next().expect("primary layer always built");
        self.queued = Some(QueuedFrame {
            frame: Frame {
                primary: primary_layer,
                overlays: built.map(|(plane, layer, _)| (plane, layer)).collect(),
            },
            // Direct frames need no fence of their own: the client's
            // buffer is GPU-complete by the time its commit was applied.
            sync: None,
        });
        if self.pending.is_none() {
            self.submit()?;
        }
        Ok(true)
    }

    /// Turn one requested layer into a plane-ready [`Layer`] plus the probe
    /// entry describing it. `None` means it can't go on a plane at all and
    /// the caller must composite the frame.
    fn build_layer(
        &mut self,
        plane: plane::Handle,
        layer: ScanoutLayer,
    ) -> Option<(plane::Handle, Layer, ProbeLayer)> {
        // KMS can't safely scan out a buffer allocated with an implicit
        // (Invalid) modifier — its tiling/layout is unknown (the Weston rule).
        let fmt = layer.dmabuf.format();
        if fmt.modifier == Modifier::Invalid {
            debug!(code = ?fmt.code, "client buffer has an implicit modifier; compositing");
            return None;
        }

        // Prefer the buffer's own fourcc when the plane advertises it with
        // this modifier: on the primary plane the caller proved the content
        // opaque, and an unused alpha channel scans out over the CRTC's black
        // background — indistinguishable. Swap to the opaque sibling only when
        // the plane lacks the native format. Forcing the sibling
        // unconditionally broke the import on planes that advertise the alpha
        // fourcc but not its sibling (NVIDIA lists AB30 but no XB30: addfb2
        // failed for every HDR game frame → zero direct scans).
        //
        // An *overlay* must keep its alpha either way — it is composited over
        // the window below it, so dropping the channel would paint a hard
        // rectangle where a rounded, shadowed notification belongs.
        let plane_formats = self.plane_formats_of(plane);
        let has_native = plane_formats
            .iter()
            .any(|f| f.code == fmt.code && f.modifier == fmt.modifier);
        let use_opaque = !has_native;
        if use_opaque && plane != self.drm.plane() {
            debug!(
                code = ?fmt.code,
                modifier = ?fmt.modifier,
                "overlay plane lacks the buffer's format and its alpha can't be dropped; compositing"
            );
            return None;
        }

        let fb = match self.import_client_fb(&layer.buffer, &layer.dmabuf, use_opaque) {
            Ok(fb) => fb,
            Err(err) => {
                debug!(
                    error = %err,
                    code = ?fmt.code,
                    modifier = ?fmt.modifier,
                    use_opaque,
                    "client buffer not importable for scanout; compositing"
                );
                return None;
            }
        };

        let placement = Placement {
            src: layer.place.src,
            dst: layer.place.dst,
            transform: layer.place.transform,
        };
        Some((
            plane,
            Layer {
                hold: Hold::Client {
                    buffer: layer.buffer,
                    fb,
                },
                placement,
                damage: layer.damage,
                alpha: 1.0,
            },
            ProbeLayer {
                plane,
                code: fmt.code,
                modifier: fmt.modifier,
                placement,
                use_opaque,
            },
        ))
    }

    /// `config: None` states for every overlay plane the last commit lit that
    /// `keeping` doesn't. Without these the hardware happily keeps scanning
    /// last frame's notification over this frame.
    fn dark_overlay_states(&self, keeping: &[plane::Handle]) -> Vec<PlaneState<'static>> {
        self.lit_overlays
            .iter()
            .filter(|h| !keeping.contains(h))
            .map(|h| PlaneState {
                handle: *h,
                config: None,
            })
            .collect()
    }

    /// The `(fourcc, modifier)` pairs one of our planes advertises.
    fn plane_formats_of(&self, plane: plane::Handle) -> Vec<Format> {
        if plane == self.drm.plane() {
            return self.drm.plane_info().formats.iter().copied().collect();
        }
        self.overlays
            .iter()
            .find(|o| o.info.handle == plane)
            .map(|o| o.info.formats.iter().copied().collect())
            .unwrap_or_default()
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
        let QueuedFrame { frame, sync } =
            self.queued.take().expect("submit called with a queued frame");

        // Damage blobs have to outlive the plane states that borrow them, so
        // they are built up front, one per layer in `frame.layers()` order.
        let clips: Vec<Option<PlaneDamageClips>> = frame
            .layers()
            .map(|layer| {
                let damage = layer.damage.clone()?;
                PlaneDamageClips::from_damage(
                    self.drm.device_fd(),
                    layer.placement.src,
                    layer.placement.dst,
                    layer.placement.transform,
                    // Damage arrives in buffer pixels already, so it needs no
                    // transform of its own — only the src→dst mapping applies.
                    Transform::Normal,
                    damage,
                )
                .ok()
                .flatten()
            })
            .collect();

        // Explicit sync: hand our render fence to KMS as IN_FENCE_FD when the
        // plane supports it; otherwise block on the GPU here and rely on
        // implicit sync.
        let fence = match sync {
            Some(FrameFence::Render(sync)) if self.supports_fencing => {
                let fence = sync.export();
                if fence.is_none() {
                    let _ = sync.wait();
                }
                fence
            }
            Some(FrameFence::Render(sync)) => {
                let _ = sync.wait();
                None
            }
            None => None,
        };

        let lit = frame.overlay_planes();
        let mut states = vec![frame.primary.plane_state(
            self.drm.plane(),
            clips[0].as_ref(),
            fence.as_ref().map(AsFd::as_fd),
        )];
        for (i, (plane, layer)) in frame.overlays.iter().enumerate() {
            states.push(layer.plane_state(*plane, clips[i + 1].as_ref(), None));
        }
        // Switch off any overlay plane the previous commit lit that this
        // frame doesn't use — including all of them for a composited frame,
        // whose single buffer already contains everything.
        states.extend(self.dark_overlay_states(&lit));

        // Tearing is only offered for a plain primary-plane flip. Drivers
        // reject an async commit that carries anything else, and an overlay
        // update is exactly that; the extra plane also means the frame isn't
        // the pure game-swapchain swap tearing exists to accelerate.
        let tearing = self.tearing && frame.is_direct() && frame.overlays.is_empty();

        let flip = if self.drm.commit_pending() {
            self.drm.commit(states, true)
        } else if tearing {
            // Immediate presentation: the flip takes effect as soon as the
            // hardware can latch it rather than at the next vblank. Falls
            // back to a normal flip when the driver refuses (see `tearing`).
            self.drm
                .page_flip_async(states.clone(), true)
                .or_else(|err| {
                    debug!(error = %err, "async page-flip rejected; falling back to vsync for this frame");
                    self.drm.page_flip(states, true)
                })
        } else {
            self.drm.page_flip(states, true)
        };
        if flip.is_ok() {
            self.lit_overlays = lit;
            self.pending = Some(frame);
        } else {
            // The driver disagreed with a configuration we may have cached as
            // good; make the next direct frame re-probe rather than repeat it.
            self.last_probe = None;
        }
        flip.context("atomic page-flip/commit failed")
    }

    /// The swapchain's scanout fourcc.
    pub fn format(&self) -> Fourcc {
        self.swapchain.format()
    }

    /// Ask for the next flips to be async (tearing) ones.
    ///
    /// Tearing is only ever *offered* — `submit` retries synchronously the
    /// moment a driver refuses, and a modeset commit always ignores it. A
    /// legacy surface can't do async flips at all, so the request is dropped
    /// there rather than failing every frame.
    pub fn set_tearing(&mut self, tearing: bool) {
        let tearing = tearing && !self.drm.is_legacy();
        if self.tearing != tearing {
            debug!(tearing, "tearing (async page-flip) state changed");
            self.tearing = tearing;
            // A fenced or damage-clipped request is exactly the kind of extra
            // state a driver rejects on an async flip, so the accepted-probe
            // cache no longer describes the request we're about to make.
            self.last_probe = None;
        }
    }

    /// Whether the next flip would be an async (tearing) one.
    pub fn tearing(&self) -> bool {
        self.tearing
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

/// The placement a compositor-rendered frame always uses: the whole buffer
/// onto the whole mode, untransformed.
fn full_placement(w: u16, h: u16) -> Placement {
    let size = (i32::from(w), i32::from(h));
    Placement {
        src: Rectangle::from_size(size.into()).to_f64(),
        dst: Rectangle::from_size(size.into()),
        transform: Transform::Normal,
    }
}

/// Claim every overlay plane this CRTC can drive that sits *above* its
/// primary plane, ordered bottom-up.
///
/// Only planes with a higher z-position qualify. Smithay's atomic commit
/// doesn't program `zpos`, so a plane's position is whatever the driver
/// defaults it to — which makes the ones below the primary underlays, useful
/// only for content behind an opaque window (i.e. never, for us). Claiming
/// them up front stops another CRTC taking one mid-session, which would turn
/// a working fast path into a per-frame rejection.
///
/// A legacy surface gets none: it has no atomic commit to put them in.
fn claim_overlay_planes(drm: &Arc<DrmSurface>) -> Vec<OverlayPlane> {
    if drm.is_legacy() {
        return Vec::new();
    }
    let primary_zpos = drm.plane_info().zpos.unwrap_or_default();
    let mut usable: Vec<&PlaneInfo> = drm
        .planes()
        .overlay
        .iter()
        .filter(|p| p.zpos.unwrap_or_default() > primary_zpos)
        .collect();
    usable.sort_by_key(|p| p.zpos.unwrap_or_default());

    let claimed: Vec<OverlayPlane> = usable
        .into_iter()
        .filter_map(|info| {
            drm.claim_plane(info.handle).map(|claim| OverlayPlane {
                info: info.clone(),
                claim,
            })
        })
        .collect();
    debug!(
        count = claimed.len(),
        primary_zpos, "claimed overlay planes for direct scanout"
    );
    claimed
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
