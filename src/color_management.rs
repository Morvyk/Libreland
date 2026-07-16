//! `wp_color_management_v1` (colour-management-v1, staging) — lets
//! clients learn an output's colour properties and tag their surfaces
//! with an image description (colour space + transfer function +
//! luminance). This is what makes HDR-capable apps (mpv, games via
//! Proton's Wayland path) *detect* HDR support and submit BT.2020 / PQ
//! content.
//!
//! smithay 0.7 has no handler for this protocol, so the global,
//! `GlobalDispatch`/`Dispatch` impls and all the object plumbing live
//! here, generated from the staging XML in `wayland-protocols`.
//!
//! Scope/notes:
//! - Bound at **version 1** (the widely-supported baseline; avoids the
//!   v2 `ready2`/`preferred_changed2` 64-bit identity split and the
//!   `windows_scrgb` / parametric-feedback additions).
//! - Only **parametric** image descriptions are supported (no ICC).
//! - `set_image_description` is applied immediately rather than on the
//!   next `wl_surface.commit`; clients set it once up front so this is
//!   fine in practice. Stored in [`State::color_surfaces`] for the
//!   renderer to consume.
//! - Surface *feedback* reports the **primary** output's image
//!   description as preferred (per-surface output tracking is a later
//!   refinement); `get_output` reports the exact output's description.

use std::sync::{Arc, Mutex};

use smithay::output::Output;
use smithay::reexports::wayland_protocols::wp::color_management::v1::server::{
    wp_color_management_output_v1::{self, WpColorManagementOutputV1},
    wp_color_management_surface_feedback_v1::{self, WpColorManagementSurfaceFeedbackV1},
    wp_color_management_surface_v1::{self, WpColorManagementSurfaceV1},
    wp_color_manager_v1::{
        self, Feature, Primaries, RenderIntent, TransferFunction, WpColorManagerV1,
    },
    wp_image_description_creator_params_v1::{self, WpImageDescriptionCreatorParamsV1},
    wp_image_description_info_v1::{self, WpImageDescriptionInfoV1},
    wp_image_description_v1::{self, WpImageDescriptionV1},
};
use smithay::reexports::wayland_server::backend::GlobalId;
use smithay::reexports::wayland_server::protocol::wl_output::WlOutput;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource, WEnum,
};
use tracing::debug;

use crate::State;

/// Protocol version we advertise. v3 brings the `windows_scrgb` /
/// `windows_bt2100` pre-defined descriptions, which are what Wine/Proton's
/// winewayland keys its whole HDR support on: with them (plus
/// `extended_target_volume`) advertised, Wine reports the display's real
/// HDR capabilities to the game, maps HDR10/scRGB swapchains to
/// `PASS_THROUGH` on the host driver, and attaches the description itself —
/// once. Without them the NVIDIA WSI runs its own color-management path,
/// which rebuilds the swapchain every present on this compositor.
const MANAGER_VERSION: u32 = 3;

/// Default SDR reference white, BT.2408 (cd/m²). Overridable per-output
/// via config; used when building an output's image description.
pub const DEFAULT_SDR_REFERENCE_WHITE: u32 = 203;

// ---------------------------------------------------------------------------
// Image description model
// ---------------------------------------------------------------------------

/// A parametric image description: the subset of colour parameters we
/// model. Stored on surfaces (what the client tagged) and built for
/// outputs (what the display expects). Luminances are in the protocol's
/// own units (min in 0.0001 cd/m², max/reference in 1 cd/m²).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageDescription {
    pub primaries: Primaries,
    pub tf: TransferFunction,
    pub min_lum: u32,
    pub max_lum: u32,
    pub reference_lum: u32,
    pub max_cll: Option<u32>,
    pub max_fall: Option<u32>,
}

impl ImageDescription {
    /// sRGB / BT.709 SDR description at the given reference white.
    pub fn srgb(reference_lum: u32) -> Self {
        Self {
            primaries: Primaries::Srgb,
            tf: TransferFunction::Srgb,
            min_lum: 2, // 0.0002 cd/m²
            max_lum: 80,
            reference_lum,
            max_cll: None,
            max_fall: None,
        }
    }

    /// BT.2020 / PQ HDR description with the given reference white and
    /// peak luminance (cd/m²).
    pub fn pq_bt2020(reference_lum: u32, max_lum: u32) -> Self {
        Self {
            primaries: Primaries::Bt2020,
            tf: TransferFunction::St2084Pq,
            min_lum: 5, // 0.0005 cd/m²
            max_lum,
            reference_lum,
            max_cll: None,
            max_fall: None,
        }
    }

    /// The protocol's pre-defined Windows-BT.2100 description: BT.2020
    /// primaries, PQ transfer, reference white assumed 203 cd/m² (ITU-R
    /// BT.2408-7), PQ system peak 10000 cd/m². Field-identical to the
    /// parametric PQ description HDR10 clients used to build by hand, so
    /// the renderer's existing PQ path applies unchanged.
    pub fn windows_bt2100() -> Self {
        Self {
            primaries: Primaries::Bt2020,
            tf: TransferFunction::St2084Pq,
            min_lum: 0,
            max_lum: 10_000,
            reference_lum: 203,
            max_cll: None,
            max_fall: None,
        }
    }

    /// The protocol's pre-defined Windows-scRGB description: sRGB
    /// primaries, extended-linear transfer where R=G=B=1.0 is 80 cd/m²
    /// (so 125.0 is the 10k cd/m² PQ peak), reference white assumed
    /// 203 cd/m² (R=G=B=2.5375).
    pub fn windows_scrgb() -> Self {
        Self {
            primaries: Primaries::Srgb,
            tf: TransferFunction::ExtLinear,
            min_lum: 0,
            max_lum: 10_000,
            reference_lum: 203,
            max_cll: None,
            max_fall: None,
        }
    }

    /// Which decode a surface tagged with this description needs.
    pub fn encoding(&self) -> Encoding {
        match self.tf {
            TransferFunction::St2084Pq | TransferFunction::Hlg => Encoding::Pq,
            // Extended-linear is Windows-scRGB in practice: the pre-defined
            // description is the only way clients reach it (Wine's
            // `create_windows_scrgb`), and it fixes BT.709 primaries with the
            // 1.0 == 80 cd/m² anchor the scRGB decode assumes.
            TransferFunction::ExtLinear => Encoding::Scrgb,
            _ => Encoding::Sdr,
        }
    }

    /// Whether this description denotes HDR content. scRGB counts: its
    /// extended-linear range reaches 10000 cd/m² at 125.0 — it is just
    /// carried on a linear rather than a PQ curve.
    pub fn is_hdr(&self) -> bool {
        !matches!(self.encoding(), Encoding::Sdr)
    }
}

/// How a colour-managed surface's pixels are encoded — this is what picks
/// the renderer's decode. Deliberately distinct from "is it HDR": scRGB and
/// PQ are both HDR but need completely different maths, and only PQ is
/// passthrough-compatible with a PQ-signalled output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// sRGB / BT.709, gamma-encoded SDR. The renderer's default.
    Sdr,
    /// PQ (or HLG) on BT.2100. Already what a PQ output wants, so these
    /// surfaces stay eligible for direct scanout / single-pass passthrough.
    Pq,
    /// Windows-scRGB: extended *linear* light on BT.709 primaries, where
    /// 1.0 == 80 cd/m² and 125.0 == 10000 cd/m², and channels may be
    /// negative to escape the sRGB gamut. Must be converted to PQ by a
    /// shader — never passed through.
    Scrgb,
}

/// Send `ready` in the version-appropriate shape: `ready2` (64-bit
/// identity split) replaces `ready` from v2 on. Our identities are
/// 32-bit, so the high word is always zero.
fn send_ready(obj: &WpImageDescriptionV1, identity: u32) {
    if obj.version() >= 2 {
        obj.ready2(0, identity);
    } else {
        obj.ready(identity);
    }
}

/// Hands out stable identities for image descriptions: identical
/// descriptions get the same identity (per the protocol's identity
/// semantics), distinct ones get distinct identities.
#[derive(Debug, Default)]
struct IdentityRegistry {
    next: u32,
    items: Vec<(u32, ImageDescription)>,
}

impl IdentityRegistry {
    fn identity_for(&mut self, desc: &ImageDescription) -> u32 {
        if let Some((id, _)) = self.items.iter().find(|(_, d)| d == desc) {
            return *id;
        }
        self.next = self.next.wrapping_add(1);
        let id = self.next;
        self.items.push((id, *desc));
        id
    }
}

/// Per-image-description object state (user data).
#[derive(Debug, Clone, Copy)]
pub struct ImageDescriptionData {
    desc: ImageDescription,
    /// Whether `get_information` is allowed on this object. The
    /// pre-defined Windows descriptions forbid it per spec ("does not
    /// allow `get_information` request").
    allow_info: bool,
}

/// A `get_information` request whose info events are sent *after* the
/// current dispatch returns. `wp_image_description_info_v1.done` is a
/// destructor event: sending it inside the creating request's handler
/// destroys the object before the wayland backend assigns its data,
/// which panics (rs backend) — so we queue and flush post-dispatch.
pub type PendingImageInfo = (WpImageDescriptionInfoV1, ImageDescription);

/// Colour state a client attached to one of its surfaces. `image_description`
/// is read by the HDR colour pipeline (render.rs) to decode tagged surfaces.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceColor {
    pub image_description: ImageDescription,
    /// Render intent the client requested. Not yet differentiated in
    /// compositing (we always composite perceptually).
    #[allow(dead_code, reason = "render intent not yet acted on in compositing")]
    pub render_intent: RenderIntent,
}

// ---------------------------------------------------------------------------
// Manager state / global
// ---------------------------------------------------------------------------

/// Holds the `wp_color_manager_v1` global alive and the identity registry.
#[derive(Debug)]
pub struct ColorManagementState {
    #[allow(dead_code, reason = "held to keep the global alive")]
    global: GlobalId,
    registry: Arc<Mutex<IdentityRegistry>>,
}

impl ColorManagementState {
    pub fn new(dh: &DisplayHandle) -> Self
    where
        State: GlobalDispatch<WpColorManagerV1, ()>,
    {
        let global = dh.create_global::<State, WpColorManagerV1, ()>(MANAGER_VERSION, ());
        Self {
            global,
            registry: Arc::new(Mutex::new(IdentityRegistry::default())),
        }
    }

    fn identity_for(&self, desc: &ImageDescription) -> u32 {
        self.registry.lock().unwrap().identity_for(desc)
    }
}

/// Mutable accumulation for a parametric image description creator. Each
/// `set_*` request fills one field; `create` validates completeness.
#[derive(Debug, Default)]
pub struct ParamsBuilder {
    inner: Mutex<ParamsInner>,
}

#[derive(Debug, Default)]
struct ParamsInner {
    primaries: Option<Primaries>,
    tf: Option<TransferFunction>,
    /// `set_*` already used → a second call is `already_set`.
    primaries_set: bool,
    tf_set: bool,
    min_lum: Option<u32>,
    max_lum: Option<u32>,
    reference_lum: Option<u32>,
    max_cll: Option<u32>,
    max_fall: Option<u32>,
}

// ---------------------------------------------------------------------------
// Output / surface helpers
// ---------------------------------------------------------------------------

impl State {
    /// The image description describing `wl_output`'s current colour
    /// signalling (HDR when the output's config has `hdr = true`).
    fn output_image_description(&self, wl_output: &WlOutput) -> ImageDescription {
        let name = Output::from_resource(wl_output).map(|o| o.name());
        self.output_image_description_by_name(name.as_deref())
    }

    fn output_image_description_by_name(&self, name: Option<&str>) -> ImageDescription {
        let cfg = name.and_then(|n| self.config.monitors.outputs.get(n));
        let reference = cfg
            .and_then(|c| c.sdr_reference_white)
            .unwrap_or(DEFAULT_SDR_REFERENCE_WHITE);
        if cfg.is_some_and(|c| c.hdr) {
            ImageDescription::pq_bt2020(reference, 1000)
        } else {
            ImageDescription::srgb(reference)
        }
    }

    /// The preferred image description for a surface — currently the
    /// primary output's description (per-surface output tracking TBD).
    fn preferred_image_description(&self) -> ImageDescription {
        let primary = self.config.monitors.primary.clone();
        self.output_image_description_by_name(primary.as_deref())
    }
}

// ---------------------------------------------------------------------------
// wp_color_manager_v1
// ---------------------------------------------------------------------------

impl GlobalDispatch<WpColorManagerV1, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WpColorManagerV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let manager = data_init.init(resource, ());
        let version = manager.version();
        debug!(version, "wp_color_manager_v1: client bound the manager");

        // Advertise capabilities (each as its own event), then `done`.
        for intent in [RenderIntent::Perceptual, RenderIntent::Relative] {
            manager.supported_intent(intent);
        }
        // The Windows pre-defined descriptions plus extended target
        // volume are the trio Wine/Proton gates its HDR reporting on.
        // `windows_bt2100` (the `create_windows_bt2100` request) only
        // exists from v3.
        let mut features = vec![
            Feature::Parametric,
            Feature::SetPrimaries,
            Feature::SetTfPower,
            Feature::SetLuminances,
            Feature::SetMasteringDisplayPrimaries,
            Feature::ExtendedTargetVolume,
            Feature::WindowsScrgb,
        ];
        if version >= 3 {
            features.push(Feature::WindowsBt2100);
        }
        for feature in features {
            manager.supported_feature(feature);
        }
        // `srgb` (the pure-power approximation) is deprecated-since v2 —
        // a compositor must not advertise deprecated transfer functions
        // to clients binding at that version or newer.
        let mut tfs = vec![
            TransferFunction::Bt1886,
            TransferFunction::Gamma22,
            TransferFunction::ExtLinear,
            TransferFunction::St2084Pq,
            TransferFunction::Hlg,
        ];
        if version < 2 {
            tfs.push(TransferFunction::Srgb);
        }
        for tf in tfs {
            manager.supported_tf_named(tf);
        }
        for primaries in [Primaries::Srgb, Primaries::Bt2020, Primaries::DisplayP3] {
            manager.supported_primaries_named(primaries);
        }
        manager.done();
    }
}

impl Dispatch<WpColorManagerV1, ()> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        manager: &WpColorManagerV1,
        request: wp_color_manager_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_color_manager_v1::Request::GetOutput { id, output } => {
                debug!(
                    output = ?Output::from_resource(&output).map(|o| o.name()),
                    "wp_color_manager_v1: get_output"
                );
                data_init.init(id, output);
            }
            wp_color_manager_v1::Request::GetSurface { id, surface } => {
                // The spec makes a second get_surface for the same wl_surface
                // a `surface_exists` protocol error, but we stay lenient: we
                // must always consume `id` (init it) or wayland-server panics,
                // and killing an otherwise-fine client over a duplicate isn't
                // worth it — the latest image description simply wins.
                debug!(surface = ?surface.id(), "wp_color_manager_v1: get_surface");
                state.color_surface_objects.insert(surface.id());
                data_init.init(id, surface);
            }
            wp_color_manager_v1::Request::GetSurfaceFeedback { id, surface } => {
                let sid = surface.id();
                let feedback = data_init.init(id, surface);
                // Inform the client of the current preferred description.
                let desc = state.preferred_image_description();
                let identity = state.color_management.identity_for(&desc);
                debug!(
                    surface = ?sid,
                    preferred_hdr = desc.is_hdr(),
                    preferred_tf = ?desc.tf,
                    "wp_color_manager_v1: get_surface_feedback (reported preferred description)"
                );
                if feedback.version() >= 2 {
                    feedback.preferred_changed2(0, identity);
                } else {
                    feedback.preferred_changed(identity);
                }
            }
            wp_color_manager_v1::Request::CreateParametricCreator { obj } => {
                data_init.init(obj, ParamsBuilder::default());
            }
            wp_color_manager_v1::Request::CreateWindowsScrgb { image_description } => {
                let desc = ImageDescription::windows_scrgb();
                let identity = state.color_management.identity_for(&desc);
                debug!(identity, "wp_color_manager_v1: create_windows_scrgb");
                let img = data_init.init(
                    image_description,
                    ImageDescriptionData {
                        desc,
                        allow_info: false,
                    },
                );
                send_ready(&img, identity);
            }
            wp_color_manager_v1::Request::CreateWindowsBt2100 { image_description } => {
                let desc = ImageDescription::windows_bt2100();
                let identity = state.color_management.identity_for(&desc);
                debug!(identity, "wp_color_manager_v1: create_windows_bt2100");
                let img = data_init.init(
                    image_description,
                    ImageDescriptionData {
                        desc,
                        allow_info: false,
                    },
                );
                send_ready(&img, identity);
            }
            wp_color_manager_v1::Request::GetImageDescription {
                image_description, ..
            } => {
                // References are minted by *other* protocols; we implement
                // none that do, so no valid reference object can reach us.
                // Consume the new_id (mandatory) and fail it gracefully.
                let img = data_init.init(
                    image_description,
                    ImageDescriptionData {
                        desc: state.preferred_image_description(),
                        allow_info: false,
                    },
                );
                img.failed(
                    wp_image_description_v1::Cause::Unsupported,
                    "image description references are not supported".to_owned(),
                );
            }
            wp_color_manager_v1::Request::CreateIccCreator { obj } => {
                // We don't advertise the icc_v2_v4 feature.
                data_init.init(obj, ());
                manager.post_error(
                    wp_color_manager_v1::Error::UnsupportedFeature,
                    "ICC image descriptions are not supported",
                );
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// wp_color_management_output_v1  (user data: the WlOutput)
// ---------------------------------------------------------------------------

impl Dispatch<WpColorManagementOutputV1, WlOutput> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _obj: &WpColorManagementOutputV1,
        request: wp_color_management_output_v1::Request,
        wl_output: &WlOutput,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        if let wp_color_management_output_v1::Request::GetImageDescription { image_description } =
            request
        {
            let desc = state.output_image_description(wl_output);
            debug!(
                output = ?Output::from_resource(wl_output).map(|o| o.name()),
                hdr = desc.is_hdr(),
                tf = ?desc.tf,
                primaries = ?desc.primaries,
                "wp_color_management_output_v1: get_image_description (sent output's colour description)"
            );
            let identity = state.color_management.identity_for(&desc);
            let obj = data_init.init(
                image_description,
                ImageDescriptionData {
                    desc,
                    allow_info: true,
                },
            );
            send_ready(&obj, identity);
        }
    }
}

// ---------------------------------------------------------------------------
// wp_color_management_surface_v1  (user data: the WlSurface)
// ---------------------------------------------------------------------------

impl Dispatch<WpColorManagementSurfaceV1, WlSurface> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        obj: &WpColorManagementSurfaceV1,
        request: wp_color_management_surface_v1::Request,
        surface: &WlSurface,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            wp_color_management_surface_v1::Request::SetImageDescription {
                image_description,
                render_intent,
            } => {
                let Some(data) = image_description.data::<ImageDescriptionData>().copied() else {
                    obj.post_error(
                        wp_color_management_surface_v1::Error::ImageDescription,
                        "image description object is not ready / not from this compositor",
                    );
                    return;
                };
                let intent = match render_intent {
                    WEnum::Value(v) => v,
                    WEnum::Unknown(_) => {
                        obj.post_error(
                            wp_color_management_surface_v1::Error::RenderIntent,
                            "unknown render intent",
                        );
                        return;
                    }
                };
                state.color_surfaces.insert(
                    surface.id(),
                    SurfaceColor {
                        image_description: data.desc,
                        render_intent: intent,
                    },
                );
                debug!(
                    surface = ?surface.id(),
                    ?intent,
                    hdr = data.desc.is_hdr(),
                    tf = ?data.desc.tf,
                    primaries = ?data.desc.primaries,
                    max_lum = data.desc.max_lum,
                    "surface image description set"
                );
                if data.desc.tf == TransferFunction::ExtLinear {
                    tracing::warn!(
                        surface = ?surface.id(),
                        "scRGB (extended-linear) surface attached; composited as SDR — scRGB decode is not implemented yet"
                    );
                }
            }
            wp_color_management_surface_v1::Request::UnsetImageDescription => {
                state.color_surfaces.remove(&surface.id());
            }
            wp_color_management_surface_v1::Request::Destroy => {
                state.color_surfaces.remove(&surface.id());
                state.color_surface_objects.remove(&surface.id());
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// wp_color_management_surface_feedback_v1  (user data: the WlSurface)
// ---------------------------------------------------------------------------

impl Dispatch<WpColorManagementSurfaceFeedbackV1, WlSurface> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _obj: &WpColorManagementSurfaceFeedbackV1,
        request: wp_color_management_surface_feedback_v1::Request,
        _surface: &WlSurface,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        if let wp_color_management_surface_feedback_v1::Request::GetPreferred { image_description } =
            request
        {
            let desc = state.preferred_image_description();
            let identity = state.color_management.identity_for(&desc);
            let obj = data_init.init(
                image_description,
                ImageDescriptionData {
                    desc,
                    allow_info: true,
                },
            );
            send_ready(&obj, identity);
        }
    }
}

// ---------------------------------------------------------------------------
// wp_image_description_creator_params_v1  (user data: ParamsBuilder)
// ---------------------------------------------------------------------------

impl Dispatch<WpImageDescriptionCreatorParamsV1, ParamsBuilder> for State {
    #[allow(
        clippy::too_many_lines,
        reason = "one match over the parametric creator's set_* requests plus create-time validation; splitting would scatter the builder logic"
    )]
    fn request(
        state: &mut Self,
        _client: &Client,
        obj: &WpImageDescriptionCreatorParamsV1,
        request: wp_image_description_creator_params_v1::Request,
        builder: &ParamsBuilder,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        use wp_image_description_creator_params_v1::{Error, Request};
        let mut params = builder.inner.lock().unwrap();
        match request {
            Request::SetTfNamed { tf } => {
                if params.tf_set {
                    obj.post_error(Error::AlreadySet, "transfer function already set");
                    return;
                }
                match tf {
                    WEnum::Value(v) => {
                        params.tf = Some(v);
                        params.tf_set = true;
                    }
                    WEnum::Unknown(_) => obj.post_error(Error::InvalidTf, "unknown transfer function"),
                }
            }
            Request::SetTfPower { .. } => {
                // Advertised, but we only act on named TFs; accept and ignore
                // the power exponent (treated as its named approximation).
                if params.tf_set {
                    obj.post_error(Error::AlreadySet, "transfer function already set");
                    return;
                }
                params.tf_set = true;
                params.tf = Some(TransferFunction::Gamma22);
            }
            Request::SetPrimariesNamed { primaries } => {
                if params.primaries_set {
                    obj.post_error(Error::AlreadySet, "primaries already set");
                    return;
                }
                match primaries {
                    WEnum::Value(v) => {
                        params.primaries = Some(v);
                        params.primaries_set = true;
                    }
                    WEnum::Unknown(_) => {
                        obj.post_error(Error::InvalidPrimariesNamed, "unknown named primaries");
                    }
                }
            }
            Request::SetPrimaries { .. } => {
                // Custom primaries accepted but approximated as BT.2020 for
                // now (we model named primaries only).
                if params.primaries_set {
                    obj.post_error(Error::AlreadySet, "primaries already set");
                    return;
                }
                params.primaries_set = true;
                params.primaries = Some(Primaries::Bt2020);
            }
            Request::SetLuminances {
                min_lum,
                max_lum,
                reference_lum,
            } => {
                params.min_lum = Some(min_lum);
                params.max_lum = Some(max_lum);
                params.reference_lum = Some(reference_lum);
            }
            Request::SetMaxCll { max_cll } => params.max_cll = Some(max_cll),
            Request::SetMaxFall { max_fall } => params.max_fall = Some(max_fall),
            Request::Create { image_description } => {
                // `create` is a destructor: we MUST consume `image_description`
                // (init it) on every path or wayland-server panics and takes
                // the compositor down. Stay lenient — default any unset
                // primaries / transfer function to sRGB rather than raising
                // `incomplete_set` and killing the client.
                let primaries = params.primaries.unwrap_or(Primaries::Srgb);
                let tf = params.tf.unwrap_or(TransferFunction::Srgb);
                // Default luminances per TF when unset.
                let (def_min, def_max, def_ref) = match tf {
                    TransferFunction::St2084Pq => (5u32, 10000u32, DEFAULT_SDR_REFERENCE_WHITE),
                    TransferFunction::Hlg => (5, 1000, DEFAULT_SDR_REFERENCE_WHITE),
                    _ => (2, 80, 80),
                };
                let desc = ImageDescription {
                    primaries,
                    tf,
                    min_lum: params.min_lum.unwrap_or(def_min),
                    max_lum: params.max_lum.unwrap_or(def_max),
                    reference_lum: params.reference_lum.unwrap_or(def_ref),
                    max_cll: params.max_cll,
                    max_fall: params.max_fall,
                };
                debug!(
                    hdr = desc.is_hdr(),
                    tf = ?desc.tf,
                    primaries = ?desc.primaries,
                    max_lum = desc.max_lum,
                    "wp_image_description_creator_params_v1: create (client built its own description)"
                );
                let identity = state.color_management.identity_for(&desc);
                let img = data_init.init(
                    image_description,
                    ImageDescriptionData {
                        desc,
                        allow_info: true,
                    },
                );
                send_ready(&img, identity);
            }
            // set_mastering_display_primaries / set_mastering_luminance are
            // accepted and ignored for now (we don't yet model the target
            // volume), plus any future requests.
            _ => {}
        }
    }
}

// ICC creator: we never hand these out (the feature isn't advertised and
// create_icc_creator errors), but the type still needs a Dispatch impl.
impl Dispatch<smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_image_description_creator_icc_v1::WpImageDescriptionCreatorIccV1, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _obj: &smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_image_description_creator_icc_v1::WpImageDescriptionCreatorIccV1,
        _request: smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_image_description_creator_icc_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
    }
}

// ---------------------------------------------------------------------------
// wp_image_description_v1  (user data: ImageDescriptionData)
// ---------------------------------------------------------------------------

impl Dispatch<WpImageDescriptionV1, ImageDescriptionData> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        obj: &WpImageDescriptionV1,
        request: wp_image_description_v1::Request,
        data: &ImageDescriptionData,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        if let wp_image_description_v1::Request::GetInformation { information } = request {
            // The pre-defined Windows descriptions forbid get_information
            // (spec: "does not allow get_information request").
            if !data.allow_info {
                obj.post_error(
                    wp_image_description_v1::Error::NoInformation,
                    "get_information is not allowed on this image description",
                );
                // Still must consume the new_id.
                data_init.init(information, ());
                return;
            }
            // Init the info object, but DEFER its events: send_information
            // ends with `done`, a destructor that would destroy this
            // just-created object before the wayland backend assigns its
            // data (panic). Flushed right after dispatch instead.
            debug!(
                hdr = data.desc.is_hdr(),
                tf = ?data.desc.tf,
                primaries = ?data.desc.primaries,
                "wp_image_description_v1: get_information (client reading a description's details)"
            );
            let info = data_init.init(information, ());
            state.pending_image_info.push((info, data.desc));
        }
    }
}

/// A description that immediately fails (e.g. inert output object).
impl Dispatch<WpImageDescriptionV1, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _obj: &WpImageDescriptionV1,
        _request: wp_image_description_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
    }
}

// ---------------------------------------------------------------------------
// wp_image_description_info_v1
// ---------------------------------------------------------------------------

impl Dispatch<WpImageDescriptionInfoV1, ()> for State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _obj: &WpImageDescriptionInfoV1,
        _request: wp_image_description_info_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
    }
}

/// Emit the `wp_image_description_info_v1` events describing `desc`, then
/// `done` (a destructor event).
fn send_information(info: &WpImageDescriptionInfoV1, desc: &ImageDescription) {
    info.primaries_named(desc.primaries);
    info.tf_named(desc.tf);
    info.luminances(desc.min_lum, desc.max_lum, desc.reference_lum);
    // Target color volume luminance. For an output description this is
    // the display's real range — Wine gates its whole HDR reporting on
    // `max_target_lum > ref_lum`, and DXGI's MaxLuminance (what games
    // tonemap to) comes from here.
    info.target_luminance(desc.min_lum, desc.max_lum);
    if let Some(max_cll) = desc.max_cll {
        info.target_max_cll(max_cll);
    }
    if let Some(max_fall) = desc.max_fall {
        info.target_max_fall(max_fall);
    }
    info.done();
}

/// Send the deferred `get_information` responses queued during dispatch.
/// Call once per event-loop iteration, after `dispatch_clients` (so the
/// info objects' data is already assigned) and before `flush_clients`.
pub fn flush_pending_image_info(state: &mut State) {
    for (info, desc) in std::mem::take(&mut state.pending_image_info) {
        send_information(&info, &desc);
    }
}
