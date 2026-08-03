//! `wp_tearing_control_v1` — per-surface immediate-presentation hints.
//!
//! A client that would rather see its frame *now*, tearing, than wait for the
//! next vblank says so through this protocol. Proton/DXVK request it for
//! `VK_PRESENT_MODE_IMMEDIATE_KHR` swapchains; SDL and Xwayland forward it for
//! games that asked X11 for tearing presents.
//!
//! The hint is only ever a request. Libreland honours it when
//! [`crate::config::TearingMode`] permits (`misc.tearing = "auto"`), the
//! window is the single fullscreen client on the
//! [direct-scanout](crate::scanout) fast path, and the driver accepts the
//! async page-flip — see [`crate::render::Renderer::apply_tearing`]. On the
//! composited desktop, or under the default `misc.tearing = "off"`, it is
//! recorded and ignored, which the protocol explicitly allows.
//!
//! We advertise the global unconditionally rather than gating it on the
//! config: a client that binds it at startup must not have to re-check after
//! a live reload flips the policy on, and toolkits treat the global's absence
//! as "this compositor can never tear" and stop asking.

use smithay::reexports::wayland_protocols::wp::tearing_control::v1::server::{
    wp_tearing_control_manager_v1::{self, WpTearingControlManagerV1},
    wp_tearing_control_v1::{self, WpTearingControlV1},
};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};
use tracing::debug;

use crate::State;

const MANAGER_VERSION: u32 = 1;

/// The `wp_tearing_control_v1` global.
#[derive(Debug)]
pub struct TearingControlState {
    #[allow(dead_code, reason = "held to keep the global alive for the session")]
    global: smithay::reexports::wayland_server::backend::GlobalId,
}

impl TearingControlState {
    pub fn new(dh: &DisplayHandle) -> Self {
        let global = dh.create_global::<State, WpTearingControlManagerV1, ()>(MANAGER_VERSION, ());
        Self { global }
    }
}

impl GlobalDispatch<WpTearingControlManagerV1, ()> for State {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WpTearingControlManagerV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<WpTearingControlManagerV1, ()> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        manager: &WpTearingControlManagerV1,
        request: wp_tearing_control_manager_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        // Destroy (a destructor) and any future request need nothing from
        // us: the manager is a pure factory.
        if let wp_tearing_control_manager_v1::Request::GetTearingControl { id, surface } = request {
            // One controller per surface, per the spec's
            // `tearing_control_exists` error.
            if state.tearing_controls.contains(&surface) {
                manager.post_error(
                    wp_tearing_control_manager_v1::Error::TearingControlExists,
                    "the surface already has a tearing controller",
                );
                return;
            }
            state.tearing_controls.push(surface.clone());
            data_init.init(id, surface);
        }
    }
}

impl Dispatch<WpTearingControlV1, WlSurface> for State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _control: &WpTearingControlV1,
        request: wp_tearing_control_v1::Request,
        surface: &WlSurface,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        // Destroy is handled in `destroyed` below, which also runs when the
        // client disconnects without sending it.
        if let wp_tearing_control_v1::Request::SetPresentationHint { hint } = request {
            // The hint is double-buffered in the spec (it applies on the
            // surface's next commit). We apply it immediately: it only ever
            // selects between two flip flags for a frame that has not been
            // submitted yet, so the one-commit skew is unobservable, and
            // tracking it through the commit cache would buy nothing.
            let immediate = matches!(
                hint,
                smithay::reexports::wayland_server::WEnum::Value(
                    wp_tearing_control_v1::PresentationHint::Async
                )
            );
            debug!(surface = ?surface.id(), immediate, "wp_tearing_control: presentation hint");
            state.renderer.set_tearing_hint(surface, immediate);
        }
    }

    fn destroyed(
        state: &mut Self,
        _client: smithay::reexports::wayland_server::backend::ClientId,
        _resource: &WpTearingControlV1,
        surface: &WlSurface,
    ) {
        // Destroying the controller reverts the surface to vsync (the spec's
        // "the content is not suitable for tearing" default).
        state.renderer.set_tearing_hint(surface, false);
        state.tearing_controls.retain(|s| s != surface);
    }
}
