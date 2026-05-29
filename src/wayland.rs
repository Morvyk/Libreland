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
use smithay::delegate_fractional_scale;
use smithay::delegate_layer_shell;
use smithay::delegate_viewporter;
use smithay::delegate_output;
use smithay::delegate_seat;
use smithay::delegate_shm;
use smithay::delegate_xdg_decoration;
use smithay::delegate_xdg_shell;
use smithay::input::keyboard::XkbConfig;
use smithay::input::pointer::CursorImageStatus;
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::output::{Mode as OutputMode, Output, PhysicalProperties, Scale, Subpixel};
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode as DecorationMode;
use smithay::reexports::wayland_server::backend::{ClientData, ClientId, DisconnectReason};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_output::WlOutput;
use smithay::reexports::wayland_server::protocol::wl_seat::WlSeat;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{Client, Display, DisplayHandle, Resource as _};
use smithay::utils::{SERIAL_COUNTER, Serial, Transform};
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    CompositorClientState, CompositorHandler, CompositorState, with_states,
};
use smithay::wayland::fractional_scale::{
    self, FractionalScaleHandler, FractionalScaleManagerState,
};
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::shell::wlr_layer::{
    Layer, LayerSurface, WlrLayerShellHandler, WlrLayerShellState,
};
use smithay::wayland::shell::xdg::decoration::{XdgDecorationHandler, XdgDecorationState};
use smithay::wayland::viewporter::ViewporterState;
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
};
use smithay::wayland::shm::{ShmHandler, ShmState};
use tracing::{debug, info, warn};

use crate::State;
use crate::config::Config;
use crate::render::OutputDescriptor;

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
    pub xdg_decoration_state: XdgDecorationState,
    pub output_manager_state: OutputManagerState,
    pub fractional_scale_state: FractionalScaleManagerState,
    /// `wp_viewporter` global. Fractional-scale-aware clients render
    /// an oversized buffer and use `wp_viewport` to map it down to
    /// the logical surface rect; without this global they can't, and
    /// their content composites at the wrong size. Held so the global
    /// stays alive (dropping it removes the global).
    pub viewporter_state: ViewporterState,
    pub layer_shell_state: WlrLayerShellState,
    /// One smithay `Output` per DRM connector. Each carries its
    /// physical mode + configured scale and is advertised to
    /// clients as a `wl_output` global so they can pick a target
    /// output for fullscreen / fractional scale.
    pub outputs: Vec<Output>,
    /// Preferred fractional scale shipped to every new
    /// `wp_fractional_scale` object. For now this is the primary
    /// output's scale; multi-output per-surface scale tracking
    /// lands with workspaces.
    pub preferred_scale: f64,
}

/// Build every Wayland substate, register the corresponding globals
/// on the display, and bind keyboard + pointer capabilities to the
/// seat (so clients see them advertised). Forwarding events to those
/// capabilities is milestone 4c.
pub fn init(
    display: &Display<State>,
    config: &Config,
    output_descs: &[OutputDescriptor],
    preferred_scale: f64,
) -> Result<WaylandInit> {
    let dh = display.handle();

    let compositor_state = CompositorState::new::<State>(&dh);
    let shm_state = ShmState::new::<State>(&dh, vec![]);
    let xdg_shell_state = XdgShellState::new::<State>(&dh);
    // Advertise xdg_decoration with our reply hard-coded to
    // ServerSide (= "compositor draws decorations"). Libreland
    // is a tiler and deliberately draws none, so the visible
    // effect is "client doesn't draw a title bar / border". A
    // client requesting ClientSide is overridden — we don't
    // accept CSD.
    let xdg_decoration_state = XdgDecorationState::new::<State>(&dh);
    let output_manager_state = OutputManagerState::new_with_xdg_output::<State>(&dh);
    let fractional_scale_state = FractionalScaleManagerState::new::<State>(&dh);
    // wp_viewporter: lets clients crop/scale a buffer to a different
    // destination size. Required for fractional scaling — a client
    // told scale 1.5 renders a 1.5x buffer and viewports it down to
    // the logical size; smithay's surface state then composites it at
    // the right size automatically.
    let viewporter_state = ViewporterState::new::<State>(&dh);
    // wlr_layer_shell: panels, bars, lockscreens, launchers, OSDs.
    // We render layer surfaces above or below the tile area
    // depending on their `Layer`, and honour their exclusive
    // zones by shrinking the layout's bounds.
    let layer_shell_state = WlrLayerShellState::new::<State>(&dh);

    // One smithay `Output` per DRM connector. Each becomes a
    // `wl_output` global the client can bind to learn the mode
    // and scale. `wl_output.scale` is integer-only per protocol;
    // we ceil the fractional scale so legacy clients get sharp
    // text. Fractional-aware clients see the exact scale via
    // `wp_fractional_scale_manager_v1`.
    let mut outputs = Vec::with_capacity(output_descs.len());
    for desc in output_descs {
        let output = Output::new(
            desc.name.clone(),
            PhysicalProperties {
                size: smithay::utils::Size::from((0, 0)),
                subpixel: Subpixel::Unknown,
                make: "libreland".into(),
                model: desc.name.clone(),
            },
        );
        let mode = OutputMode {
            size: desc.mode_size,
            // Refresh in milli-Hz, threaded through from the
            // active DRM mode (so a 4K@144 monitor advertises
            // 144 000 here, not a placeholder).
            refresh: desc.refresh_mhz,
        };
        output.change_current_state(
            Some(mode),
            Some(Transform::Normal),
            Some(Scale::Fractional(desc.scale)),
            Some(smithay::utils::Point::<i32, smithay::utils::Logical>::from(
                (desc.compositor_position.x, desc.compositor_position.y),
            )),
        );
        output.set_preferred(mode);
        output.create_global::<State>(&dh);
        outputs.push(output);
    }

    let mut seat_state = SeatState::<State>::new();
    let mut seat = seat_state.new_wl_seat(&dh, "seat0");

    // Capabilities. Use the configured keyboard layout so the keymap
    // smithay ships to clients matches the one our own xkbcommon
    // wrapper (`crate::keyboard::Keyboard`) uses for hotkey matching —
    // otherwise compositor-level binds and the focused client would
    // see different keysyms on non-`us` layouts.
    //
    // repeat_delay/rate from Config — clamped to i32 because
    // smithay's add_keyboard takes signed ints (negative values are
    // invalid; saturating downward to i32::MAX is harmless for
    // values that are already absurdly large).
    let xkb = XkbConfig {
        layout: &config.input.keyboard_layout,
        ..XkbConfig::default()
    };
    let repeat_delay = i32::try_from(config.input.repeat_delay).unwrap_or(i32::MAX);
    let repeat_rate = i32::try_from(config.input.repeat_rate).unwrap_or(i32::MAX);
    seat.add_keyboard(xkb, repeat_delay, repeat_rate)
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
        xdg_decoration_state,
        output_manager_state,
        fractional_scale_state,
        viewporter_state,
        layer_shell_state,
        outputs,
        preferred_scale,
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
        maybe_handle_layer_commit(self, surface);
    }

    fn destroyed(&mut self, surface: &WlSurface) {
        info!(surface = ?surface.id(), "wayland: surface destroyed");
    }
}

/// Layer-surface focus + layout reflow happens on commit
/// (not on `new_layer_surface`) because the client's
/// `keyboard_interactivity` / `anchor` / `exclusive_zone`
/// are only readable from `LayerSurfaceCachedState` after
/// they've been committed. Doing it here means the very
/// first time rofi (or anything else requesting an
/// exclusive overlay) commits its state, the seat hands
/// it focus, and the layout reflows around its zone.
fn maybe_handle_layer_commit(state: &mut State, surface: &WlSurface) {
    use smithay::wayland::shell::wlr_layer::{KeyboardInteractivity, Layer};
    let is_layer = state
        .layer_shell_state
        .layer_surfaces()
        .any(|l| l.wl_surface() == surface);
    if !is_layer {
        return;
    }
    let cached = layer_cached_state(surface);
    state.recompute_layer_layout();
    let wants_exclusive_kbd = matches!(
        cached.keyboard_interactivity,
        KeyboardInteractivity::Exclusive
    ) && matches!(cached.layer, Layer::Top | Layer::Overlay);
    if !wants_exclusive_kbd {
        return;
    }
    let Some(kbd) = state.seat.get_keyboard() else {
        return;
    };
    if kbd.current_focus().as_ref() == Some(surface) {
        return;
    }
    // Save the focus we're displacing, but only the first time —
    // repeated commits from the same exclusive layer shouldn't
    // overwrite the saved "before" with itself.
    if state.kbd_focus_before_layer.is_none() {
        state.kbd_focus_before_layer = kbd.current_focus();
    }
    info!(
        surface = ?surface.id(),
        "wayland: exclusive layer surface grabbing keyboard focus"
    );
    kbd.set_focus(state, Some(surface.clone()), SERIAL_COUNTER.next_serial());
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
        // Hand the toplevel to the tiler at the cursor's current
        // position. The layout splits whichever existing leaf the
        // cursor is over, so spawning a terminal with the mouse
        // over the right half makes room there instead of always
        // dropping the new tile in the deepest dwindle cell.
        let (cx, cy) = self.renderer.cursor_pos();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "cursor coords are clamped to layout_bounds (i32) in Renderer::on_pointer_motion"
        )]
        let cursor =
            smithay::utils::Point::<i32, smithay::utils::Physical>::from((cx as i32, cy as i32));
        self.layout.insert(surface.clone(), Some(cursor));
        // Promote the new toplevel to keyboard focus. For both
        // hover and click models a fresh window should start with
        // focus so the user can type into it immediately, even if
        // the pointer hasn't moved onto it yet.
        let wl_surface = surface.wl_surface().clone();
        if let Some(kbd) = self.seat.get_keyboard() {
            kbd.set_focus(self, Some(wl_surface), SERIAL_COUNTER.next_serial());
        }
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
        // Pull from the tiler — its sibling takes the freed cell
        // and every remaining window receives a fresh configure.
        self.layout.remove(surface.wl_surface());
        // Clear keyboard focus only if the destroyed surface was
        // actually focused — otherwise leave whatever is focused
        // alone (it might be a different live toplevel). 4d.2 will
        // pick a sensible replacement instead of dropping to None.
        if let Some(kbd) = self.seat.get_keyboard() {
            let was_focused = kbd
                .current_focus()
                .as_ref()
                .is_some_and(|f| f == surface.wl_surface());
            if was_focused {
                kbd.set_focus(self, None, SERIAL_COUNTER.next_serial());
            }
        }
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

impl XdgDecorationHandler for State {
    fn new_decoration(&mut self, toplevel: ToplevelSurface) {
        // Client created a decoration object — pin us to
        // ServerSide before the first configure so the client
        // never starts with CSD then has to redraw without it.
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(DecorationMode::ServerSide);
        });
        toplevel.send_configure();
    }

    fn request_mode(&mut self, toplevel: ToplevelSurface, _mode: DecorationMode) {
        // Client preference is ignored; tiling WM doesn't have
        // optional decorations to negotiate over.
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(DecorationMode::ServerSide);
        });
        toplevel.send_configure();
    }

    fn unset_mode(&mut self, toplevel: ToplevelSurface) {
        // Client released the decoration object. Keep ServerSide
        // pinned so it doesn't accidentally fall back to CSD.
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(DecorationMode::ServerSide);
        });
        toplevel.send_configure();
    }
}

impl WlrLayerShellHandler for State {
    fn shell_state(&mut self) -> &mut WlrLayerShellState {
        &mut self.layer_shell_state
    }

    fn new_layer_surface(
        &mut self,
        surface: LayerSurface,
        _output: Option<WlOutput>,
        layer: Layer,
        namespace: String,
    ) {
        info!(
            namespace,
            ?layer,
            surface = ?surface.wl_surface().id(),
            "wayland: new layer surface"
        );
        // Send the initial configure carrying the layout area the
        // surface can occupy. Anchor + size + exclusive-zone math
        // is handled by the renderer / layout once the client
        // commits a buffer; here we just bootstrap the configure
        // cycle. Clients that requested an explicit size (rofi:
        // anchored centre, 800x600) keep that; clients that asked
        // for "0" in some axis get the matching output dimension.
        // Initial configure with the primary output's compositor
        // rect as a size hint. The client-side keyboard / anchor /
        // exclusive-zone state isn't readable yet — that gets
        // looked at on the first commit, in `CompositorHandler::commit`.
        let primary = self.renderer.primary_output_rect();
        let bounds_size = primary.size;
        surface.with_pending_state(|state| {
            state.size = Some(smithay::utils::Size::<i32, smithay::utils::Logical>::from(
                (bounds_size.w, bounds_size.h),
            ));
        });
        surface.send_configure();
    }

    fn layer_destroyed(&mut self, surface: LayerSurface) {
        info!(surface = ?surface.wl_surface().id(), "wayland: layer surface destroyed");
        let cur_focus = self.seat.get_keyboard().and_then(|k| k.current_focus());
        if cur_focus.as_ref() == Some(surface.wl_surface())
            && let Some(kbd) = self.seat.get_keyboard()
        {
            let restore = self.kbd_focus_before_layer.take();
            kbd.set_focus(self, restore, SERIAL_COUNTER.next_serial());
        }
        self.recompute_layer_layout();
    }
}

/// Read the current cached `LayerSurfaceCachedState` of a layer
/// surface — pulled out into a free function because the closure
/// in `with_states` can't directly return a non-`'static` borrow,
/// so we clone the whole cached struct.
pub fn layer_cached_state(
    surface: &WlSurface,
) -> smithay::wayland::shell::wlr_layer::LayerSurfaceCachedState {
    use smithay::wayland::shell::wlr_layer::LayerSurfaceCachedState;
    let mut out: LayerSurfaceCachedState = LayerSurfaceCachedState::default();
    with_states(surface, |states| {
        let mut cached = states.cached_state.get::<LayerSurfaceCachedState>();
        out = *cached.current();
    });
    out
}

impl FractionalScaleHandler for State {
    fn new_fractional_scale(&mut self, surface: WlSurface) {
        // The preferred fractional scale is whatever the layout's
        // primary output is configured to. Once we have per-output
        // workspaces we'll re-send this per-surface based on which
        // output that surface ends up on.
        let preferred = self.preferred_scale;
        with_states(&surface, |states| {
            fractional_scale::with_fractional_scale(states, |fs| {
                fs.set_preferred_scale(preferred);
            });
        });
    }
}

delegate_compositor!(State);
delegate_shm!(State);
delegate_seat!(State);
delegate_xdg_shell!(State);
delegate_xdg_decoration!(State);
delegate_output!(State);
delegate_fractional_scale!(State);
delegate_layer_shell!(State);
delegate_viewporter!(State);
