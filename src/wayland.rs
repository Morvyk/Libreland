//! Wayland frontend — milestone 4a.
//!
//! Brings up enough of the protocol surface that an unmodified
//! Wayland client (kitty, foot, weston-terminal) can connect, query
//! the seat and outputs, allocate surfaces, get an `xdg_toplevel`
//! configured, and have its lifecycle reach our logs. Rendering
//! client buffers is the 4b milestone; input forwarding is 4c.
//!
//! Wayland state lives on [`crate::State`] as flat fields so the
//! `delegate_*` macros work without intermediate wrappers. The owned
//! `Display<State>` itself can't live inside `State` (the type would
//! be circular), so it lives in a sibling struct
//! ([`crate::LoopData`]) alongside `State`; per-tick dispatch and
//! flush happens there.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use smithay::backend::renderer::utils::on_commit_buffer_handler;
use smithay::delegate_compositor;
use smithay::delegate_output;
use smithay::delegate_seat;
use smithay::delegate_shm;
use smithay::delegate_xdg_shell;
use smithay::input::keyboard::XkbConfig;
use smithay::input::pointer::CursorImageStatus;
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::reexports::wayland_server::backend::{ClientData, ClientId, DisconnectReason};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_seat::WlSeat;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{Client, Display, DisplayHandle, Resource as _};
use smithay::utils::Serial;
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{CompositorClientState, CompositorHandler, CompositorState};
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
};
use smithay::wayland::shm::{ShmHandler, ShmState};
use tracing::{debug, info, warn};

use crate::State;
use crate::config::Config;

/// Per-client state attached to every Wayland client at
/// `insert_client` time. Smithay's `CompositorClientState` has to
/// live per-client (it tracks the client's pending-state queue) so
/// we wrap it in our own struct that also serves as the
/// `ClientData` impl for connect/disconnect lifecycle hooks.
#[derive(Default)]
pub struct ClientState {
    pub compositor_state: CompositorClientState,
}

impl ClientData for ClientState {
    fn initialized(&self, client_id: ClientId) {
        info!(?client_id, "wayland: client connected");
    }
    fn disconnected(&self, client_id: ClientId, reason: DisconnectReason) {
        info!(?client_id, ?reason, "wayland: client disconnected");
    }
}

/// Bundle of Wayland substate produced by [`init`] and consumed by
/// `State`'s constructor. Kept separate from [`crate::State`] so
/// `wayland.rs` is the single owner of the build-up.
pub struct WaylandInit {
    pub display_handle: DisplayHandle,
    pub compositor_state: CompositorState,
    pub shm_state: ShmState,
    pub seat_state: SeatState<State>,
    pub seat: Seat<State>,
    pub xdg_shell_state: XdgShellState,
    pub output_manager_state: OutputManagerState,
}

/// Build every Wayland substate, register the corresponding globals
/// on the display, and bind keyboard + pointer capabilities to the
/// seat (so clients see them advertised). Forwarding events to those
/// capabilities is milestone 4c.
pub fn init(display: &Display<State>, config: &Config) -> Result<WaylandInit> {
    let dh = display.handle();

    let compositor_state = CompositorState::new::<State>(&dh);
    let shm_state = ShmState::new::<State>(&dh, vec![]);
    let xdg_shell_state = XdgShellState::new::<State>(&dh);
    let output_manager_state = OutputManagerState::new();

    let mut seat_state = SeatState::<State>::new();
    let mut seat = seat_state.new_wl_seat(&dh, "seat0");

    // Capabilities. Default XkbConfig means xkbcommon system
    // defaults (matches `crate::keyboard::Keyboard::new("")`).
    // repeat_delay/rate from Config — clamped to i32 because
    // smithay's add_keyboard takes signed ints (negative values are
    // invalid; saturating downward to i32::MAX is harmless for
    // values that are already absurdly large).
    let repeat_delay = i32::try_from(config.input.repeat_delay).unwrap_or(i32::MAX);
    let repeat_rate = i32::try_from(config.input.repeat_rate).unwrap_or(i32::MAX);
    seat.add_keyboard(XkbConfig::default(), repeat_delay, repeat_rate)
        .context("seat.add_keyboard failed")?;
    seat.add_pointer();

    info!(
        seat = "seat0",
        "wayland: seat with keyboard + pointer ready"
    );

    Ok(WaylandInit {
        display_handle: dh,
        compositor_state,
        shm_state,
        seat_state,
        seat,
        xdg_shell_state,
        output_manager_state,
    })
}

/// Spawn each configured startup command as a child process. Each
/// entry is whitespace-split into program + args; children inherit
/// the compositor's environment (notably `$WAYLAND_DISPLAY`) so
/// they connect to *our* socket. A failed spawn is logged and
/// skipped — one broken command shouldn't crash the compositor.
pub fn spawn_startup(commands: &[String]) {
    for raw in commands {
        let parts: Vec<&str> = raw.split_whitespace().collect();
        let Some((program, args)) = parts.split_first() else {
            warn!(command = raw, "startup command is empty — skipping");
            continue;
        };
        match std::process::Command::new(program).args(args).spawn() {
            Ok(child) => info!(pid = child.id(), command = raw, "spawned startup command"),
            Err(err) => warn!(
                command = raw,
                error = %err,
                "failed to spawn startup command"
            ),
        }
    }
}

/// Wrap a freshly-accepted client stream into a registered
/// Wayland client, attaching the per-client `ClientState`. The
/// resulting `Arc<dyn ClientData>` is what smithay needs so it can
/// route the per-client `CompositorClientState` lookup during
/// dispatch.
pub fn new_client_data() -> Arc<ClientState> {
    Arc::new(ClientState::default())
}

// ---- Handler implementations on `crate::State` ------------------

impl CompositorHandler for State {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client
            .get_data::<ClientState>()
            .expect("client inserted without ClientState — see wayland::new_client_data")
            .compositor_state
    }

    fn new_surface(&mut self, surface: &WlSurface) {
        info!(surface = ?surface.id(), "wayland: new surface");
    }

    fn commit(&mut self, surface: &WlSurface) {
        debug!(surface = ?surface.id(), "wayland: surface commit");
        // Hands the freshly-attached buffer over to smithay's
        // RendererSurfaceState — uploads it on the next render and
        // makes `render_elements_from_surface_tree` produce a
        // non-empty element. Skipping this leaves the surface
        // invisible (no texture) even though the client did
        // everything right.
        on_commit_buffer_handler::<State>(surface);
    }

    fn destroyed(&mut self, surface: &WlSurface) {
        info!(surface = ?surface.id(), "wayland: surface destroyed");
    }
}

impl ShmHandler for State {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}

impl BufferHandler for State {
    fn buffer_destroyed(&mut self, _buffer: &WlBuffer) {
        // Buffer lifecycle becomes meaningful in 4b when we
        // actually composite client buffers; until then there's
        // nothing to release here.
    }
}

impl SeatHandler for State {
    type KeyboardFocus = WlSurface;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    fn focus_changed(&mut self, _seat: &Seat<Self>, _focused: Option<&WlSurface>) {
        // No-op in 4a; 4c wires this up to actually shift focus.
    }

    fn cursor_image(&mut self, _seat: &Seat<Self>, _image: CursorImageStatus) {
        // No-op in 4a; clients can request cursor images but we
        // keep drawing our procedural cursor until cursor-shape /
        // surface-cursor support lands.
    }
}

impl XdgShellHandler for State {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell_state
    }

    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        info!(surface = ?surface.wl_surface().id(), "wayland: new xdg_toplevel");
        // Mandatory: clients hang waiting for an initial configure
        // before they start drawing. The default configure carries
        // no size, which means "you decide" — fine for 4a since
        // we're not rendering the result anyway.
        surface.send_configure();
    }

    fn new_popup(&mut self, surface: PopupSurface, _positioner: PositionerState) {
        info!(surface = ?surface.wl_surface().id(), "wayland: new xdg_popup");
        if let Err(err) = surface.send_configure() {
            warn!(?err, "xdg_popup send_configure failed");
        }
    }

    fn grab(&mut self, _surface: PopupSurface, _seat: WlSeat, _serial: Serial) {
        // Popup grabs require pointer-focus plumbing; 4c.
    }

    fn reposition_request(
        &mut self,
        _surface: PopupSurface,
        _positioner: PositionerState,
        _token: u32,
    ) {
        // Repositioning needs the same window-management machinery
        // we don't have until 4d.
    }

    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        info!(surface = ?surface.wl_surface().id(), "wayland: xdg_toplevel destroyed");
    }
}

impl OutputHandler for State {
    // All trait methods have sensible defaults for 4a.
}

impl AsMut<CompositorState> for State {
    fn as_mut(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }
}

delegate_compositor!(State);
delegate_shm!(State);
delegate_seat!(State);
delegate_xdg_shell!(State);
delegate_output!(State);
