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
use smithay::delegate_cursor_shape;
use smithay::delegate_data_control;
use smithay::delegate_data_device;
use smithay::delegate_ext_data_control;
use smithay::delegate_idle_inhibit;
use smithay::delegate_dmabuf;
use smithay::delegate_fractional_scale;
use smithay::delegate_kde_decoration;
use smithay::delegate_layer_shell;
use smithay::delegate_viewporter;
use smithay::delegate_output;
use smithay::delegate_pointer_constraints;
use smithay::delegate_primary_selection;
use smithay::delegate_relative_pointer;
use smithay::delegate_seat;
use smithay::delegate_shm;
use smithay::delegate_xdg_decoration;
use smithay::delegate_xdg_shell;
use smithay::input::keyboard::XkbConfig;
use smithay::input::pointer::{CursorImageStatus, PointerHandle};
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::output::{Mode as OutputMode, Output, PhysicalProperties, Scale, Subpixel};
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode as DecorationMode;
use smithay::reexports::wayland_server::backend::{
    ClientData, ClientId, DisconnectReason, GlobalId,
};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_output::WlOutput;
use smithay::reexports::wayland_server::protocol::wl_seat::WlSeat;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{Client, Display, DisplayHandle, Resource as _};
use smithay::utils::{SERIAL_COUNTER, Serial, Transform};
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::{Buffer as _, Format};
use smithay::backend::drm::DrmNode;
use smithay::desktop::{PopupKind, PopupManager};
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    CompositorClientState, CompositorHandler, CompositorState, with_states,
};
use smithay::wayland::dmabuf::{
    DmabufFeedbackBuilder, DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier,
};
use smithay::wayland::fractional_scale::{
    self, FractionalScaleHandler, FractionalScaleManagerState,
};
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::pointer_constraints::{
    PointerConstraintsHandler, PointerConstraintsState, with_pointer_constraint,
};
use smithay::wayland::relative_pointer::RelativePointerManagerState;
use smithay::wayland::selection::data_device::{
    ClientDndGrabHandler, DataDeviceHandler, DataDeviceState, ServerDndGrabHandler,
    set_data_device_focus,
};
use smithay::wayland::selection::primary_selection::{
    PrimarySelectionHandler, PrimarySelectionState, set_primary_focus,
};
use smithay::wayland::idle_inhibit::{IdleInhibitHandler, IdleInhibitManagerState};
use smithay::wayland::selection::wlr_data_control::{
    DataControlHandler as WlrDataControlHandler, DataControlState as WlrDataControlState,
};
use smithay::wayland::selection::ext_data_control::{
    DataControlHandler as ExtDataControlHandler, DataControlState as ExtDataControlState,
};
use smithay::wayland::selection::{SelectionHandler, SelectionSource, SelectionTarget};
use smithay::wayland::cursor_shape::CursorShapeManagerState;
use smithay::wayland::session_lock::{
    LockSurface, SessionLockHandler, SessionLockManagerState, SessionLocker,
};
use smithay::wayland::tablet_manager::TabletSeatHandler;
use smithay::wayland::shell::wlr_layer::{
    Layer, LayerSurface, WlrLayerShellHandler, WlrLayerShellState,
};
use smithay::reexports::wayland_protocols_misc::server_decoration::server::org_kde_kwin_server_decoration::{
    Mode as KdeMode, OrgKdeKwinServerDecoration,
};
use smithay::reexports::wayland_protocols_misc::server_decoration::server::org_kde_kwin_server_decoration_manager::Mode as KdeDefaultMode;
use smithay::reexports::wayland_server::WEnum;
use smithay::wayland::shell::kde::decoration::{KdeDecorationHandler, KdeDecorationState};
use smithay::wayland::shell::xdg::decoration::{XdgDecorationHandler, XdgDecorationState};
use smithay::wayland::viewporter::ViewporterState;
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
};
use smithay::wayland::shm::{ShmHandler, ShmState};
use tracing::{debug, info, warn};

use crate::State;
use crate::config::Config;
use crate::layout::FillMode;
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
    /// `wp_cursor_shape_v1` global. Lets clients request a named cursor
    /// shape (arrow, text, grab, …) instead of supplying a surface;
    /// smithay funnels the request through `SeatHandler::cursor_image`
    /// as a `CursorImageStatus::Named`. Held to keep the global alive.
    pub cursor_shape_state: CursorShapeManagerState,
    pub xdg_shell_state: XdgShellState,
    pub xdg_decoration_state: XdgDecorationState,
    /// KDE `org_kde_kwin_server_decoration` global, advertising a
    /// default mode of *Server*. Toolkits (notably GTK/Firefox) read
    /// this manager's `default_mode` on bind to decide whether to
    /// draw their own client-side decorations; advertising Server
    /// suppresses their titlebar even when they never touch
    /// `zxdg_decoration` (which Firefox doesn't). This is the same
    /// trick wlroots compositors use for "prefer no CSD".
    pub kde_decoration_state: KdeDecorationState,
    pub output_manager_state: OutputManagerState,
    pub fractional_scale_state: FractionalScaleManagerState,
    /// `wl_data_device_manager` global — clipboard and drag-and-drop.
    /// GTK/Qt toolkits set up their seat's clipboard through this on
    /// startup; without it, GDK's seat is left half-initialised (the
    /// `gdk_seat_get_keyboard` criticals) and apps like Firefox crash
    /// when input focus arrives.
    pub data_device_state: DataDeviceState,
    /// `zwp_linux_dmabuf_v1` global — lets clients hand us
    /// GPU-rendered content as dmabuf buffers. Required for
    /// GPU-composited apps (e.g. the Steam client via Xwayland) to
    /// show anything; without it their surfaces render blank.
    pub dmabuf_state: DmabufState,
    pub dmabuf_global: DmabufGlobal,
    /// `wp_viewporter` global. Fractional-scale-aware clients render
    /// an oversized buffer and use `wp_viewport` to map it down to
    /// the logical surface rect; without this global they can't, and
    /// their content composites at the wrong size. Held so the global
    /// stays alive (dropping it removes the global).
    pub viewporter_state: ViewporterState,
    pub layer_shell_state: WlrLayerShellState,
    /// `zwp_relative_pointer_manager_v1` — lets clients receive raw
    /// relative motion deltas (mouse-look in games). Held so the
    /// global stays alive.
    pub relative_pointer_state: RelativePointerManagerState,
    /// `zwp_pointer_constraints_v1` — lets clients lock or confine the
    /// pointer (FPS games lock it in place and read relative motion).
    /// Held so the global stays alive.
    pub pointer_constraints_state: PointerConstraintsState,
    /// `zwp_primary_selection_v1` — the primary (middle-click) selection.
    /// Held so the global stays alive.
    pub primary_selection_state: PrimarySelectionState,
    /// `zwlr_data_control_manager_v1` — privileged selection access for
    /// clipboard managers (wlroots flavour). Read by the
    /// `DataControlHandler` impl; held so the global stays alive.
    pub wlr_data_control_state: WlrDataControlState,
    /// `ext_data_control_manager_v1` — the standardized successor to
    /// `wlr_data_control`, same role. Held so the global stays alive.
    pub ext_data_control_state: ExtDataControlState,
    /// `zwp_idle_inhibit_manager_v1` — clients inhibit idle (lock/DPMS)
    /// while a surface is up. Held so the global stays alive.
    pub idle_inhibit_state: IdleInhibitManagerState,
    /// `zwlr_screencopy_manager_v1` — output capture for screenshots
    /// and screen sharing. Held so the global stays alive.
    pub screencopy_manager: crate::screencopy::ScreencopyManagerState,
    /// Tracks `xdg_popup` parent→child trees (menus / submenus).
    pub popup_manager: PopupManager,
    /// One smithay `Output` per DRM connector. Each carries its
    /// physical mode + configured scale and is advertised to
    /// clients as a `wl_output` global so they can pick a target
    /// output for fullscreen / fractional scale.
    pub outputs: Vec<Output>,
    /// The `wl_output` global id for each output, keyed by connector
    /// name. Kept so a hot-unplugged output's global can be removed
    /// (and a hot-plugged one's added) at runtime.
    pub output_globals: std::collections::HashMap<String, GlobalId>,
    /// Preferred fractional scale shipped to every new
    /// `wp_fractional_scale` object. For now this is the primary
    /// output's scale; multi-output per-surface scale tracking
    /// lands with workspaces.
    pub preferred_scale: f64,
}

/// Build a smithay [`Output`] from a descriptor and apply its current
/// mode, transform, fractional scale, and position. Does **not** create
/// the `wl_output` global — the caller does that and keeps the returned
/// [`GlobalId`]. Shared by [`init`] (startup) and the hotplug path (a
/// monitor connecting at runtime) so both configure outputs identically.
pub(crate) fn make_output(desc: &OutputDescriptor) -> Output {
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
        // Refresh in milli-Hz, threaded through from the active DRM mode
        // (so a 4K@144 monitor advertises 144 000 here, not a placeholder).
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
    output
}

/// Build every Wayland substate, register the corresponding globals
/// on the display, and bind keyboard + pointer capabilities to the
/// seat (so clients see them advertised). Forwarding events to those
/// capabilities is milestone 4c.
#[allow(
    clippy::too_many_lines,
    reason = "flat one-time build-up: construct + register each Wayland global in sequence. Splitting it would only scatter the single owner of the setup across helpers for no clarity gain."
)]
pub fn init(
    display: &Display<State>,
    config: &Config,
    output_descs: &[OutputDescriptor],
    preferred_scale: f64,
    dmabuf_formats: Vec<Format>,
    render_node: Option<DrmNode>,
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
    // KDE server-decoration: advertise a *Server* default so toolkits
    // that ignore zxdg_decoration (Firefox/GTK) still drop their CSD
    // titlebar. We draw nothing, so the result is a bare window.
    let kde_decoration_state = KdeDecorationState::new::<State>(&dh, KdeDefaultMode::Server);
    let output_manager_state = OutputManagerState::new_with_xdg_output::<State>(&dh);
    let fractional_scale_state = FractionalScaleManagerState::new::<State>(&dh);
    // wl_data_device_manager: clipboard + drag-and-drop. Toolkits
    // initialise their seat's clipboard through this; its absence
    // leaves GTK's seat half-set-up and crashes Firefox on focus.
    let data_device_state = DataDeviceState::new::<State>(&dh);
    // wp_cursor_shape_v1: clients request named cursor shapes we theme.
    let cursor_shape_state = CursorShapeManagerState::new::<State>(&dh);
    // zwp_linux_dmabuf_v1: advertise the GPU buffer formats our GLES
    // renderer can import, so GPU-composited clients (and Xwayland's
    // glamor-rendered windows via xwayland-satellite) can present
    // dmabuf content instead of rendering blank. We advertise a *v4*
    // global with default feedback (main render device + format
    // table) — modern Xwayland/glamor needs the feedback to pick a
    // render format and won't use a plain v3 global, leaving GPU X
    // apps (the Steam client) blank. Falls back to v3 only if the
    // render node or feedback can't be built.
    let mut dmabuf_state = DmabufState::new();
    let dmabuf_global = if let Some(node) = render_node {
        match DmabufFeedbackBuilder::new(node.dev_id(), dmabuf_formats.clone()).build() {
            Ok(feedback) => {
                info!(node = ?node, "advertising zwp_linux_dmabuf_v1 with default feedback (v4)");
                dmabuf_state.create_global_with_default_feedback::<State>(&dh, &feedback)
            }
            Err(err) => {
                warn!(error = %err, "dmabuf feedback build failed; advertising v3 dmabuf global");
                dmabuf_state.create_global::<State>(&dh, dmabuf_formats)
            }
        }
    } else {
        warn!(
            "no render DRM node resolved; advertising v3 dmabuf global (GPU X apps may stay blank)"
        );
        dmabuf_state.create_global::<State>(&dh, dmabuf_formats)
    };
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
    // zwp_relative_pointer_manager_v1 + zwp_pointer_constraints_v1:
    // together these let games do proper mouse-look — the client locks
    // the pointer in place (so the cursor can't leave the window or hit
    // another monitor) and reads raw relative-motion deltas (so it
    // doesn't derive bogus deltas from a clamped absolute position,
    // which is what makes the view spin).
    let relative_pointer_state = RelativePointerManagerState::new::<State>(&dh);
    let pointer_constraints_state = PointerConstraintsState::new::<State>(&dh);
    // zwp_primary_selection_v1: the middle-click "primary" selection.
    // Both it and the regular clipboard are persisted compositor-side
    // (see crate::clipboard) so a copied buffer survives the source app
    // exiting.
    let primary_selection_state = PrimarySelectionState::new::<State>(&dh);
    // zwlr_data_control_manager_v1 + ext_data_control_manager_v1:
    // privileged selection access for clipboard managers (cliphist,
    // clipman, `wl-paste --watch`) — they read every new selection and
    // can set their own. We advertise both the wlroots protocol (what
    // current tooling targets) and its standardized successor `ext`.
    // Both reuse our `SelectionHandler`, so the compositor's clipboard
    // cache backs them with no extra plumbing. Passing
    // `primary_selection_state` also exposes the middle-click primary
    // selection; the `|_| true` filter lets any client bind (these
    // protocols grant unrestricted clipboard read, by design).
    let wlr_data_control_state =
        WlrDataControlState::new::<State, _>(&dh, Some(&primary_selection_state), |_| true);
    let ext_data_control_state =
        ExtDataControlState::new::<State, _>(&dh, Some(&primary_selection_state), |_| true);
    // zwp_idle_inhibit_manager_v1: lets a client (e.g. a video player)
    // ask us not to idle while its surface is up. We honour it by
    // suppressing the built-in lock/DPMS (see `State::idle_tick`).
    let idle_inhibit_state = IdleInhibitManagerState::new::<State>(&dh);
    // zwlr_screencopy_manager_v1: lets grim / xdg-desktop-portal-wlr
    // capture outputs for screenshots and screen sharing.
    let screencopy_manager = crate::screencopy::ScreencopyManagerState::new(&dh);
    // xdg_popup tracking (menus / submenus). No global of its own —
    // popups arrive through xdg_wm_base; this just bookkeeps the
    // parent→child trees so we can position + render them.
    let popup_manager = PopupManager::default();

    // One smithay `Output` per DRM connector. Each becomes a
    // `wl_output` global the client can bind to learn the mode
    // and scale. `wl_output.scale` is integer-only per protocol;
    // we ceil the fractional scale so legacy clients get sharp
    // text. Fractional-aware clients see the exact scale via
    // `wp_fractional_scale_manager_v1`.
    let mut outputs = Vec::with_capacity(output_descs.len());
    let mut output_globals = std::collections::HashMap::with_capacity(output_descs.len());
    for desc in output_descs {
        let output = make_output(desc);
        let global = output.create_global::<State>(&dh);
        output_globals.insert(desc.name.clone(), global);
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
        cursor_shape_state,
        xdg_shell_state,
        xdg_decoration_state,
        kde_decoration_state,
        output_manager_state,
        fractional_scale_state,
        data_device_state,
        dmabuf_state,
        dmabuf_global,
        viewporter_state,
        layer_shell_state,
        relative_pointer_state,
        pointer_constraints_state,
        primary_selection_state,
        wlr_data_control_state,
        ext_data_control_state,
        idle_inhibit_state,
        screencopy_manager,
        popup_manager,
        outputs,
        output_globals,
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
        // Some clients ignore the size in the *initial* configure (sent before
        // they map) and render at their own default size, only resizing when a
        // later configure arrives — MPV's idle "Drop files" window is the
        // notorious one: it maps small inside its cell and snaps to size only
        // on the next configure (e.g. when the user moves it). The first time a
        // tracked toplevel commits a buffer, re-send its layout configure so it
        // resizes itself with no user interaction.
        if smithay::wayland::compositor::get_role(surface)
            == Some(smithay::wayland::shell::xdg::XDG_TOPLEVEL_ROLE)
            && !self.mapped_toplevels.contains(surface)
            && smithay::backend::renderer::utils::with_renderer_surface_state(surface, |s| {
                s.buffer().is_some()
            })
            .unwrap_or(false)
        {
            self.mapped_toplevels.insert(surface.clone());
            // Child/dialog toplevels (a properties window, a login/preferences
            // dialog) auto-float centred instead of wedging into the tiling —
            // their parent + size hints are only set now, at first map. If it's
            // not a dialog, fall back to the size nudge for clients that
            // ignored the initial configure.
            if !self.layout.float_if_dialog(surface) {
                self.layout.reconfigure(surface);
            }
        }
        maybe_handle_layer_commit(self, surface);
        // Promote a freshly-mapped popup into its parent's tree (and
        // let smithay send its initial configure if needed), then reap
        // popups whose surfaces died so they stop being rendered.
        self.popup_manager.commit(surface);
        self.popup_manager.cleanup();
        // On-demand render: a committed buffer is new pixels — redraw the
        // output this surface is on. This is the hot path that lets a
        // fullscreen client's output flip at the client's frame rate (the
        // basis of working VRR), while a background or off-output commit
        // leaves an unrelated output asleep.
        self.queue_redraw_for_surface(surface);
    }

    fn destroyed(&mut self, surface: &WlSurface) {
        info!(surface = ?surface.id(), "wayland: surface destroyed");
        // Drop the first-map nudge record so the set doesn't accumulate dead
        // surfaces (a toplevel's wl_surface is never reused once destroyed).
        self.mapped_toplevels.remove(surface);
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
    // Configure the client with its real size — honouring anchors/stretch/
    // margins — so an anchored bar (which requests a 0-size "stretch" axis)
    // renders at the right size instead of the full output. Re-configure
    // only on an actual size change, or every commit would send a configure
    // and loop with the client's re-render.
    if let Some(layer) = state
        .layer_shell_state
        .layer_surfaces()
        .find(|l| l.wl_surface() == surface)
    {
        let (w, h) = layer_size(state.layer_output_rect(surface), &cached);
        let new_size = Some(smithay::utils::Size::<i32, smithay::utils::Logical>::from((w, h)));
        let changed = layer.with_pending_state(|st| {
            if st.size == new_size {
                false
            } else {
                st.size = new_size;
                true
            }
        });
        if changed {
            layer.send_configure();
        }
    }
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
    // Don't fight a *sibling* exclusive layer for the keyboard. slurp
    // maps one exclusive Overlay surface per output and grabs the
    // keyboard on each; without this guard every per-frame commit would
    // steal focus back from the other surface, flooding both with a
    // wl_keyboard.leave/enter storm each redraw. The first exclusive
    // layer to grab the keyboard keeps it until it (and its siblings)
    // close. `focus_locked_by_layer()` is true exactly when the current
    // keyboard focus is already an exclusive Top/Overlay layer surface.
    if state.focus_locked_by_layer() {
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

    fn focus_changed(&mut self, seat: &Seat<Self>, focused: Option<&WlSurface>) {
        // Hand clipboard + primary-selection *focus* to the keyboard-
        // focused client. Smithay only delivers selection offers (and
        // thus enables paste) to the client holding data-device focus,
        // and it does NOT derive that from keyboard focus on its own —
        // without this, no client ever receives the selection and paste
        // silently does nothing.
        let client = focused.and_then(smithay::reexports::wayland_server::Resource::client);
        let dh = self.display_handle.clone();
        set_data_device_focus(&dh, seat, client.clone());
        set_primary_focus(&dh, seat, client);
    }

    fn cursor_image(&mut self, _seat: &Seat<Self>, image: CursorImageStatus) {
        // The focused client set its pointer image — a surface
        // (`wl_pointer.set_cursor`, used by toolkits and games incl.
        // Xwayland via the satellite), a named shape
        // (`wp_cursor_shape_v1`, which smithay funnels here as
        // `Named`), or `Hidden`. The renderer draws it next frame,
        // unless a compositor grab override is active.
        self.renderer.set_cursor_status(image);
        // The cursor sprite changed in place (no motion). Redraw so it
        // updates; skip fullscreen outputs (the cursor is hidden there and
        // we don't want to disturb a game's VRR).
        self.queue_redraw_nonfullscreen();
    }
}

// `wp_cursor_shape_v1` covers tablet tools too, so the delegate
// requires this. We don't advertise tablets, so the default no-op
// (ignore tablet-tool cursor requests) is all we need.
impl TabletSeatHandler for State {}

impl PointerConstraintsHandler for State {
    fn new_constraint(&mut self, surface: &WlSurface, pointer: &PointerHandle<Self>) {
        // A constraint is created inactive; the compositor decides when
        // to honour it. Activate immediately if the pointer is already
        // over this surface (the common case — a game requests the lock
        // while focused); otherwise it activates on the next motion into
        // the surface (see the motion path in main.rs). Smithay
        // deactivates automatically when the surface loses pointer focus.
        if pointer.current_focus().as_ref() == Some(surface) {
            with_pointer_constraint(surface, pointer, |constraint| {
                if let Some(constraint) = constraint {
                    constraint.activate();
                }
            });
        }
    }

    fn cursor_position_hint(
        &mut self,
        _surface: &WlSurface,
        _pointer: &PointerHandle<Self>,
        _location: smithay::utils::Point<f64, smithay::utils::Logical>,
    ) {
        // The locked client reports where it draws its own cursor so we
        // could warp there on unlock. We keep the visible cursor parked
        // where the lock engaged (it doesn't move during the lock and
        // reappears in place on unlock), so there's nothing to do.
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
        // Play the open (fade + scale-in) animation the first frame this
        // toplevel is drawn.
        self.renderer.mark_open(surface.wl_surface());
        // Promote the new toplevel to keyboard focus. For both
        // hover and click models a fresh window should start with
        // focus so the user can type into it immediately, even if
        // the pointer hasn't moved onto it yet.
        let wl_surface = surface.wl_surface().clone();
        if let Some(kbd) = self.seat.get_keyboard() {
            kbd.set_focus(self, Some(wl_surface), SERIAL_COUNTER.next_serial());
        }
        // New window reflows the layout + starts its open animation.
        self.queue_redraw_all();
    }

    fn maximize_request(&mut self, surface: ToplevelSurface) {
        info!(surface = ?surface.wl_surface().id(), "wayland: maximize request");
        if !self
            .layout
            .set_fill(surface.wl_surface(), FillMode::Maximized)
        {
            // Untracked surface (shouldn't happen — insert runs at
            // toplevel creation); still answer with a configure as the
            // protocol requires.
            surface.send_configure();
        }
        self.queue_redraw_all();
    }

    fn unmaximize_request(&mut self, surface: ToplevelSurface) {
        info!(surface = ?surface.wl_surface().id(), "wayland: unmaximize request");
        if !self.layout.set_fill(surface.wl_surface(), FillMode::Normal) {
            surface.send_configure();
        }
        self.queue_redraw_all();
    }

    fn fullscreen_request(&mut self, surface: ToplevelSurface, _output: Option<WlOutput>) {
        info!(surface = ?surface.wl_surface().id(), "wayland: fullscreen request");
        // We fullscreen on the output the window already lives on, so
        // the requested `output` hint is ignored.
        if !self
            .layout
            .set_fill(surface.wl_surface(), FillMode::Fullscreen)
        {
            surface.send_configure();
        }
        // Redraw so the window goes fullscreen and the output re-evaluates
        // VRR (Auto engages adaptive-sync now that a window fills it).
        self.queue_redraw_all();
    }

    fn unfullscreen_request(&mut self, surface: ToplevelSurface) {
        info!(surface = ?surface.wl_surface().id(), "wayland: unfullscreen request");
        if !self.layout.set_fill(surface.wl_surface(), FillMode::Normal) {
            surface.send_configure();
        }
        self.queue_redraw_all();
    }

    fn new_popup(&mut self, surface: PopupSurface, positioner: PositionerState) {
        info!(surface = ?surface.wl_surface().id(), "wayland: new xdg_popup");
        // Stamp the positioner-computed geometry into pending state
        // before the first configure so the popup reports the right
        // size/anchor; then track it so it joins its parent's tree
        // (and so the renderer can find + place it).
        surface.with_pending_state(|state| {
            state.geometry = positioner.get_geometry();
            state.positioner = positioner;
        });
        if let Err(err) = self
            .popup_manager
            .track_popup(PopupKind::Xdg(surface.clone()))
        {
            warn!(?err, "track_popup failed (dead popup surface)");
            return;
        }
        if let Err(err) = surface.send_configure() {
            warn!(?err, "xdg_popup send_configure failed");
        }
    }

    fn grab(&mut self, _surface: PopupSurface, _seat: WlSeat, _serial: Serial) {
        // A real seat grab needs `SeatHandler::KeyboardFocus: From<PopupKind>`
        // (ours is `WlSurface`), so smithay's PopupGrab can't be used
        // without a focus-type refactor. Dismiss-on-click-outside is
        // handled pragmatically in `forward_pointer_button` instead;
        // keyboard menu navigation is deferred.
    }

    fn reposition_request(
        &mut self,
        surface: PopupSurface,
        positioner: PositionerState,
        token: u32,
    ) {
        // Recompute geometry from the new positioner and tell the
        // client (send_repositioned must precede the configure so it
        // can correlate the new geometry with its token).
        surface.with_pending_state(|state| {
            state.geometry = positioner.get_geometry();
            state.positioner = positioner;
        });
        surface.send_repositioned(token);
        if let Err(err) = surface.send_configure() {
            warn!(?err, "xdg_popup reposition send_configure failed");
        }
    }

    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        info!(surface = ?surface.wl_surface().id(), "wayland: xdg_toplevel destroyed");
        // Snapshot the window's last frame for the close (fade + shrink)
        // animation *before* it leaves the layout — once removed, its
        // last drawn rect is gone. No-op if close animation is disabled
        // or the buffer's already gone.
        self.renderer.start_close(surface.wl_surface());
        // Pull from the tiler — its sibling takes the freed cell
        // and every remaining window receives a fresh configure.
        self.layout.remove(surface.wl_surface());
        // Release the window's stable IPC id so it isn't reused.
        self.ipc.forget(surface.wl_surface());
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
        // Reflow + close animation need to be drawn.
        self.queue_redraw_all();
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

// Clipboard + primary selection. Beyond advertising the globals (so
// toolkits set up their seat clipboard and smithay routes offers
// between clients), we persist every selection compositor-side: on
// each copy we cache the data and take ownership as a server-side
// source, so a buffer survives the source app closing (see
// `crate::clipboard`). `SelectionUserData` stays `()` — there's one
// cache per target, looked up by `SelectionTarget` in `send_selection`.
// DnD adds no compositor behaviour, so the grab handlers are empty.
impl SelectionHandler for State {
    type SelectionUserData = ();

    fn new_selection(
        &mut self,
        ty: SelectionTarget,
        source: Option<SelectionSource>,
        _seat: Seat<State>,
    ) {
        crate::clipboard::on_new_selection(self, ty, source.map(|s| s.mime_types()));
    }

    fn send_selection(
        &mut self,
        ty: SelectionTarget,
        mime_type: String,
        fd: std::os::fd::OwnedFd,
        _seat: Seat<State>,
        _user_data: &(),
    ) {
        crate::clipboard::on_send_selection(self, ty, &mime_type, fd);
    }
}

impl ClientDndGrabHandler for State {
    /// A client started a drag. Smithay has already installed the
    /// drag-and-drop pointer grab (which routes the offer to whatever
    /// surface our pointer focus lands on); we just composite the drag
    /// icon at the cursor for the duration.
    fn started(
        &mut self,
        _source: Option<
            smithay::reexports::wayland_server::protocol::wl_data_source::WlDataSource,
        >,
        icon: Option<WlSurface>,
        _seat: Seat<Self>,
    ) {
        self.renderer.set_dnd_icon(icon);
        // Show/hide the drag icon (it then follows the cursor via motion).
        self.queue_redraw_nonfullscreen();
    }

    /// The drag ended (dropped or cancelled) — remove the icon.
    fn dropped(&mut self, _target: Option<WlSurface>, _validated: bool, _seat: Seat<Self>) {
        self.renderer.set_dnd_icon(None);
        self.queue_redraw_nonfullscreen();
    }
}
impl ServerDndGrabHandler for State {}

impl DataDeviceHandler for State {
    fn data_device_state(&self) -> &DataDeviceState {
        &self.data_device_state
    }
}

impl PrimarySelectionHandler for State {
    fn primary_selection_state(&self) -> &PrimarySelectionState {
        &self.primary_selection_state
    }
}

// data-control (wlr + ext): both protocols drive selections through the
// shared `SelectionHandler` above, so a clipboard manager setting the
// selection routes through `crate::clipboard` exactly like a normal
// client would — no extra handling needed beyond exposing the state.
impl WlrDataControlHandler for State {
    fn data_control_state(&self) -> &WlrDataControlState {
        &self.wlr_data_control_state
    }
}

impl ExtDataControlHandler for State {
    fn data_control_state(&self) -> &ExtDataControlState {
        &self.ext_data_control_state
    }
}

// idle-inhibit: a client (typically a video player) holds an inhibitor
// while its surface is up so the session doesn't idle. We just track the
// inhibiting surfaces; `State::idle_tick` suppresses the built-in
// lock/DPMS while any is alive. smithay calls `uninhibit` on a clean
// destroy; a crashed client's stale surface is pruned in `idle_tick`.
impl IdleInhibitHandler for State {
    fn inhibit(&mut self, surface: WlSurface) {
        self.idle_inhibitors.insert(surface);
    }

    fn uninhibit(&mut self, surface: WlSurface) {
        self.idle_inhibitors.remove(&surface);
    }
}

// GPU buffer sharing. When a client (or Xwayland via the satellite)
// offers a dmabuf, try to import it into the GLES renderer and accept
// or reject accordingly — a rejected buffer makes the client fall
// back to another format rather than render blank.
impl DmabufHandler for State {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.dmabuf_state
    }

    fn dmabuf_imported(
        &mut self,
        _global: &DmabufGlobal,
        dmabuf: Dmabuf,
        notifier: ImportNotifier,
    ) {
        let ok = self.renderer.import_dmabuf(&dmabuf);
        info!(
            format = ?dmabuf.format(),
            planes = dmabuf.num_planes(),
            ok,
            "dmabuf import request"
        );
        if ok {
            if let Err(err) = notifier.successful::<State>() {
                warn!(error = %err, "dmabuf import succeeded but notifying the client failed");
            }
        } else {
            warn!(format = ?dmabuf.format(), "rejected client dmabuf: renderer import failed");
            notifier.failed();
        }
    }
}

delegate_dmabuf!(State);

// KDE server-side decoration. We force Server for every decoration
// object regardless of what the client asks for — Libreland is a
// tiler and draws no decorations, so "server-side" here means "no
// titlebar at all". Combined with the manager's Server default mode,
// this stops GTK/Firefox from drawing a client-side titlebar.
impl KdeDecorationHandler for State {
    fn kde_decoration_state(&self) -> &KdeDecorationState {
        &self.kde_decoration_state
    }

    fn new_decoration(&mut self, _surface: &WlSurface, decoration: &OrgKdeKwinServerDecoration) {
        decoration.mode(KdeMode::Server);
    }

    fn request_mode(
        &mut self,
        _surface: &WlSurface,
        decoration: &OrgKdeKwinServerDecoration,
        _mode: WEnum<KdeMode>,
    ) {
        // Ignore the client's preference; always server-side.
        decoration.mode(KdeMode::Server);
    }
}

impl WlrLayerShellHandler for State {
    fn shell_state(&mut self) -> &mut WlrLayerShellState {
        &mut self.layer_shell_state
    }

    fn new_layer_surface(
        &mut self,
        surface: LayerSurface,
        output: Option<WlOutput>,
        layer: Layer,
        namespace: String,
    ) {
        // Remember which output the client bound this surface to so it
        // lands on that monitor, not always the primary. A null output
        // means "compositor's choice" — left unrecorded so placement
        // falls back to primary.
        let output_name =
            output.and_then(|wl| smithay::output::Output::from_resource(&wl).map(|o| o.name()));
        if let Some(name) = output_name.clone() {
            self.layer_outputs.insert(surface.wl_surface().clone(), name);
        }
        self.layer_namespaces
            .insert(surface.wl_surface().clone(), namespace.clone());
        info!(
            namespace,
            ?layer,
            output = ?output_name,
            surface = ?surface.wl_surface().id(),
            "wayland: new layer surface"
        );
        // No configure here: per the wlr-layer-shell handshake the client
        // performs an initial (bufferless) commit AFTER setting its anchor /
        // size / exclusive-zone, and only then can we read that state and
        // reply with a correctly-sized configure — which we do in
        // `maybe_handle_layer_commit`. Sending one now (before the state is
        // readable) is what forced bars to the full output size.
    }

    fn layer_destroyed(&mut self, surface: LayerSurface) {
        info!(surface = ?surface.wl_surface().id(), "wayland: layer surface destroyed");
        self.layer_outputs.remove(surface.wl_surface());
        self.layer_namespaces.remove(surface.wl_surface());
        let cur_focus = self.seat.get_keyboard().and_then(|k| k.current_focus());
        if cur_focus.as_ref() == Some(surface.wl_surface())
            && let Some(kbd) = self.seat.get_keyboard()
        {
            use smithay::reexports::wayland_server::Resource;
            // Prefer handing the keyboard to a still-mapped sibling
            // exclusive layer (slurp's other overlay); only when none
            // remain fall back to the pre-layer focus — and only if that
            // surface is still alive, so we never focus a dead surface.
            let restore = self
                .first_exclusive_layer_surface(surface.wl_surface())
                .or_else(|| self.kbd_focus_before_layer.take().filter(Resource::is_alive));
            kbd.set_focus(self, restore, SERIAL_COUNTER.next_serial());
        }
        self.recompute_layer_layout();
        // A panel vanished: its exclusive zone is reclaimed and windows
        // reflow into it.
        self.queue_redraw_all();
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

/// The size (in compositor/logical px) a layer surface should be configured
/// and rendered at, within its output `area`: a stretched span across two
/// anchored opposite edges (minus margins), else the client's explicit size,
/// else the full output dimension when it left an axis at 0 for the
/// compositor to choose. Clamped to the output. Shared by the renderer's
/// placement and the `configure` we send the client, so the size we tell the
/// client and the size we draw it at can never disagree — the bug that made
/// an anchored bar (which asks for a 0-width "stretch" surface) render at the
/// full output until we started sending this on commit.
pub(crate) fn layer_size(
    area: smithay::utils::Rectangle<i32, smithay::utils::Physical>,
    cached: &smithay::wayland::shell::wlr_layer::LayerSurfaceCachedState,
) -> (i32, i32) {
    use smithay::wayland::shell::wlr_layer::Anchor;
    let stretch_x = cached.anchor.contains(Anchor::LEFT) && cached.anchor.contains(Anchor::RIGHT);
    let stretch_y = cached.anchor.contains(Anchor::TOP) && cached.anchor.contains(Anchor::BOTTOM);
    let width = if stretch_x {
        area.size.w - cached.margin.left - cached.margin.right
    } else if cached.size.w > 0 {
        cached.size.w
    } else {
        area.size.w
    };
    let height = if stretch_y {
        area.size.h - cached.margin.top - cached.margin.bottom
    } else if cached.size.h > 0 {
        cached.size.h
    } else {
        area.size.h
    };
    (width.clamp(1, area.size.w), height.clamp(1, area.size.h))
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
delegate_data_device!(State);
delegate_kde_decoration!(State);
delegate_relative_pointer!(State);
delegate_pointer_constraints!(State);
delegate_cursor_shape!(State);
delegate_primary_selection!(State);
delegate_data_control!(State);
delegate_ext_data_control!(State);
delegate_idle_inhibit!(State);
smithay::delegate_session_lock!(State);

impl SessionLockHandler for State {
    fn lock_state(&mut self) -> &mut SessionLockManagerState {
        &mut self.session_lock_state
    }

    fn lock(&mut self, confirmation: SessionLocker) {
        // We can always lock: confirm to the client, then switch rendering and
        // input over to the lock surfaces (dropping `confirmation` instead
        // would tell the client the lock failed).
        confirmation.lock();
        self.on_session_locked();
    }

    fn unlock(&mut self) {
        self.on_session_unlocked();
    }

    fn new_surface(&mut self, surface: LockSurface, output: WlOutput) {
        self.add_lock_surface(surface, &output);
    }
}
