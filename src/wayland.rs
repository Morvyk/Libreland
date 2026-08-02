//! Wayland frontend — milestone 4a.
//!
//! Brings up enough of the protocol surface that an unmodified
//! Wayland client (kitty, foot, weston-terminal) can connect, query
//! the seat and outputs, allocate surfaces, get an `xdg_toplevel`
//! configured, and have its lifecycle reach our logs. Rendering
//! client buffers is the 4b milestone; input forwarding is 4c.
//!
//! Wayland state lives on [`crate::State`] as flat fields so the
//! `delegate_dispatch2!` macro works without intermediate wrappers. The calloop
//! loop data type *is* `State`; the owned `Display<State>` (which can't
//! live inside `State` without making the type circular) is moved into
//! the dispatch source's closure, and outbound flushing goes through the
//! `DisplayHandle` on `State`.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use smithay::backend::renderer::sync::Fence as _;
use smithay::backend::renderer::utils::on_commit_buffer_handler;
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
use smithay::desktop::{PopupKind, PopupManager, find_popup_root_surface};
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
use smithay::input::dnd::{DnDGrab, DndGrabHandler, DndTarget, GrabType, Source as DndSource};
use smithay::input::pointer::Focus as GrabFocus;
use smithay::wayland::selection::data_device::{
    DataDeviceHandler, DataDeviceState, WaylandDndGrabHandler, set_data_device_focus,
};
use smithay::wayland::selection::primary_selection::{
    PrimarySelectionHandler, PrimarySelectionState, set_primary_focus,
};
use smithay::wayland::idle_inhibit::{IdleInhibitHandler, IdleInhibitManagerState};
use smithay::wayland::idle_notify::{IdleNotifierHandler, IdleNotifierState};
use smithay::wayland::xdg_activation::{
    XdgActivationHandler, XdgActivationState, XdgActivationToken, XdgActivationTokenData,
};
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
use smithay::input::tablet::TabletSeatHandler;
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
    /// The v4 default dmabuf feedback (render device + import formats) —
    /// what every surface receives until it fullscreens. `None` when
    /// only a v3 global could be advertised. NEVER re-sent to a window
    /// leaving fullscreen: feedback is sticky for the surface's life
    /// (every switch is a swapchain rebuild; see
    /// `State::sync_scanout_feedback`).
    pub dmabuf_default_feedback: Option<smithay::wayland::dmabuf::DmabufFeedback>,
    /// The default feedback plus a Scanout-flagged tranche of one output's
    /// primary-plane modifiers, keyed by connector name. Sent per-surface to
    /// fullscreen windows only (see `State::sync_scanout_feedback`) so their
    /// swapchains re-allocate into plane-scannable buffers; never the
    /// default — see the comment at its construction site.
    ///
    /// One entry per output because planes differ between connectors: the
    /// primary output's modifier list is simply wrong for a game on the
    /// second monitor.
    pub dmabuf_scanout_feedback: HashMap<String, smithay::wayland::dmabuf::DmabufFeedback>,
    /// What a *hot-plugged* output needs to build its own scanout feedback
    /// variant later: the render node's device id and the full import format
    /// list. `None` when only a v3 dmabuf global could be advertised.
    pub dmabuf_feedback_seed: Option<(u64, Vec<Format>)>,
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
    /// `xdg_activation_v1` — clients request focus/raise for a surface.
    /// Read by the `XdgActivationHandler` impl.
    pub xdg_activation_state: XdgActivationState,
    /// `zwp_pointer_gestures_v1` — touchpad gestures to clients. Held so
    /// the global stays alive; dispatch routes through it.
    pub pointer_gestures_state: smithay::wayland::pointer_gestures::PointerGesturesState,
    /// `zwlr_screencopy_manager_v1` — output capture for screenshots
    /// and screen sharing. Held so the global stays alive.
    pub screencopy_manager: crate::screencopy::ScreencopyManagerState,
    /// `wp_tearing_control_manager_v1` — clients request immediate
    /// presentation. Held so the global stays alive.
    pub tearing_control: crate::tearing::TearingControlState,
    /// `wp_color_management_v1` — clients detect output HDR and tag their
    /// surfaces' colour space. Held so the global stays alive.
    pub color_management: crate::color_management::ColorManagementState,
    /// `wp_content_type_v1` — clients hint a surface's content (game / video /
    /// photo). Held so the global stays alive; the hint is read from the
    /// surface's cached state when we want to drive per-content behaviour.
    pub content_type_state: smithay::wayland::content_type::ContentTypeState,
    pub presentation_state: smithay::wayland::presentation::PresentationState,
    /// `wp_fifo_manager_v1` — FIFO (vsync) present-mode barriers. Managed:
    /// smithay blocks a `wait_barrier` commit until we clear the barrier
    /// (at vblank, see [`signal_fifo_barriers`]).
    ///
    /// This is advertised but `wp_commit_timing_manager_v1` deliberately is
    /// NOT — matching `KWin`, the config where NVIDIA's Vulkan Wayland WSI
    /// runs `VK_EXT_present_timing` cleanly. Advertising commit-timing makes
    /// the NVIDIA WSI take its absolute-time present path, whose per-present
    /// stage-timing buffer it never allocates for our surface — it then
    /// writes timing results through a NULL pointer (SIGSEGV, caught by
    /// Wine as the `vkGetPastPresentationTimingEXT` assert). See the git
    /// history around this line for the full Ghidra/Mesa/KWin analysis.
    ///
    /// History: the same SIGSEGV once fired even WITHOUT commit-timing, which
    /// is why a `no-present-timing` Vulkan layer used to ship in contrib. It
    /// stopped reproducing after the present-pipeline overhaul (dmabuf
    /// feedback stability, colour-management v2/KWin parity, pacing/release
    /// fixes — the "fix Proton/Vulkan games" series), verified across native
    /// VK and D3D-translated titles with the layer disabled, so the layer
    /// was removed. If a present-timing SIGSEGV ever returns, that layer
    /// (git history: contrib/no-present-timing-layer) is the known shield.
    #[allow(dead_code, reason = "held so the wp_fifo global stays alive; dispatch routes through it")]
    pub fifo_manager_state: Option<smithay::wayland::fifo::FifoManagerState>,
    /// `xwayland_shell_v1` — the protocol Xwayland uses to associate its
    /// `wl_surface`s with X11 windows (see `src/xwayland.rs`). Held so
    /// the global stays alive; only the Xwayland client may bind it.
    pub xwayland_shell_state: smithay::wayland::xwayland_shell::XWaylandShellState,
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
    /// `xdg_wm_dialog_v1` — clients declare "this toplevel is a
    /// dialog/modal"; feeds the auto-float decision so it no longer
    /// rests on heuristics alone (see `layout::dialog_size`).
    #[allow(dead_code, reason = "held so the xdg-wm-dialog global stays alive")]
    pub xdg_dialog_state: smithay::wayland::shell::xdg::dialog::XdgDialogState,
    /// `wp_pointer_warp_v1` — clients ask the compositor to move the
    /// pointer to a surface-local position (games/emulators recentring
    /// the cursor without a lock). Honoured only for the surface that
    /// currently holds pointer focus; see `PointerWarpHandler` on
    /// `State`.
    #[allow(dead_code, reason = "held so the wp_pointer_warp global stays alive")]
    pub pointer_warp_manager: smithay::wayland::pointer_warp::PointerWarpManager,
    /// `wl_fixes` — lets clients destroy `wl_registry` objects they no
    /// longer need (long-lived clients otherwise leak registries
    /// server-side). Entirely handled inside smithay.
    #[allow(dead_code, reason = "held so the wl_fixes global stays alive")]
    pub fixes_state: smithay::wayland::fixes::FixesState,
    /// `ext-background-effect-v1` — clients request backdrop blur behind
    /// their surfaces; the renderer honours the committed region as an
    /// opt-in equivalent to `blur.windows`/`blur.layers` (config
    /// `blur.enabled`/`passes` still gate the pyramid).
    #[allow(dead_code, reason = "held so the ext-background-effect global stays alive")]
    pub background_effect_state: smithay::wayland::background_effect::BackgroundEffectState,
    /// `ext-output-image-capture-source-manager-v1` — moved into
    /// [`crate::State`] (its handler needs `&mut` access).
    pub output_capture_source_state:
        smithay::wayland::image_capture_source::OutputCaptureSourceState,
    /// `ext-image-copy-capture-manager-v1` — moved into [`crate::State`].
    pub image_copy_capture_state: smithay::wayland::image_copy_capture::ImageCopyCaptureState,
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
            serial_number: "unknown".into(),
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
    scanout_formats: Vec<(String, Vec<Format>)>,
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
    // glamor-rendered windows via our Xwayland) can present
    // dmabuf content instead of rendering blank. We advertise a *v4*
    // global with default feedback (main render device + format
    // table) — modern Xwayland/glamor needs the feedback to pick a
    // render format and won't use a plain v3 global, leaving GPU X
    // apps (the Steam client) blank. Falls back to v3 only if the
    // render node or feedback can't be built.
    let mut dmabuf_state = DmabufState::new();
    let mut dmabuf_default_feedback = None;
    let mut dmabuf_scanout_feedback = HashMap::new();
    let mut dmabuf_feedback_seed = None;
    let dmabuf_global = if let Some(node) = render_node {
        match DmabufFeedbackBuilder::new(node.dev_id(), dmabuf_formats.clone()).build() {
            Ok(feedback) => {
                info!(node = ?node, "advertising zwp_linux_dmabuf_v1 with default feedback (v4)");
                let global =
                    dmabuf_state.create_global_with_default_feedback::<State>(&dh, &feedback);
                // A second feedback variant per output, carrying a
                // Scanout-flagged preference tranche of that output's
                // primary-plane modifiers. Sent ONLY per-surface, to
                // fullscreen windows (`State::sync_scanout_feedback`) —
                // putting the tranche in the *default* feedback broke every
                // GPU client's EGL init wholesale (see the revert of
                // fe142c7). The tranche's target is the render node — the
                // device clients already open; the primary card node may not
                // be openable from inside a session while the compositor
                // holds the seat.
                for (output, formats) in scanout_formats {
                    if formats.is_empty() {
                        continue;
                    }
                    match build_scanout_feedback(node.dev_id(), &dmabuf_formats, formats) {
                        Ok(scanout) => {
                            info!(%output, "built per-output scanout dmabuf feedback (fullscreen windows)");
                            dmabuf_scanout_feedback.insert(output, scanout);
                        }
                        Err(err) => {
                            warn!(%output, error = %err, "scanout dmabuf feedback build failed; fullscreen windows there keep the default");
                        }
                    }
                }
                dmabuf_feedback_seed = Some((node.dev_id(), dmabuf_formats));
                dmabuf_default_feedback = Some(feedback);
                global
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
    // zwp_pointer_gestures_v1: forward touchpad pinch/swipe/hold gestures
    // to clients (handled in the libinput event loop).
    let pointer_gestures_state =
        smithay::wayland::pointer_gestures::PointerGesturesState::new::<State>(&dh);
    // xdg_wm_dialog_v1: clients declare dialog/modal toplevels; the
    // auto-float path reads the stored hint (layout::dialog_size).
    let xdg_dialog_state = smithay::wayland::shell::xdg::dialog::XdgDialogState::new::<State>(&dh);
    // wp_pointer_warp_v1: client-requested pointer warps (see the
    // PointerWarpHandler impl for the honour/reject policy).
    let pointer_warp_manager =
        smithay::wayland::pointer_warp::PointerWarpManager::new::<State>(&dh);
    // wl_fixes: registry destruction for long-lived clients.
    let fixes_state = smithay::wayland::fixes::FixesState::new::<State>(&dh);
    // ext-background-effect-v1: clients opt into backdrop blur behind
    // their own surfaces (the render path honours a committed blur
    // region alongside the config's blur.windows/blur.layers opt-ins).
    let background_effect_state =
        smithay::wayland::background_effect::BackgroundEffectState::new::<State>(&dh);
    // ext-image-capture-source (outputs) + ext-image-copy-capture: the
    // modern capture stack, riding the same render-path machinery as
    // wlr-screencopy (see src/copycapture.rs).
    let output_capture_source_state =
        smithay::wayland::image_capture_source::OutputCaptureSourceState::new::<State>(&dh);
    let image_copy_capture_state =
        smithay::wayland::image_copy_capture::ImageCopyCaptureState::new::<State>(&dh);
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
    // xdg_activation_v1: lets a client request focus for a surface (open
    // a link → the browser raises, notification click → app raises). We
    // reveal + focus the target if the token is fresh.
    let xdg_activation_state = XdgActivationState::new::<State>(&dh);
    // zwlr_screencopy_manager_v1: lets grim / the desktop portal
    // capture outputs for screenshots and screen sharing.
    let screencopy_manager = crate::screencopy::ScreencopyManagerState::new(&dh);
    // wp_tearing_control_v1: clients (Proton/DXVK immediate swapchains) ask to
    // be presented without waiting for vblank. Always advertised; whether we
    // act on it is `misc.tearing` — see crate::tearing.
    let tearing_control = crate::tearing::TearingControlState::new(&dh);
    // wp_color_management_v1: clients detect output HDR + tag surface
    // colour spaces (Proton/mpv use this to enable HDR).
    // Re-enabled at KWin parity (v2, matching feature set) after the bisect
    // proved our v3 + scRGB/BT2100 advertisement drove Wine's untested code
    // paths into a swapchain-rebuild storm — see MANAGER_VERSION in
    // color_management.rs for the measured evidence.
    let color_management = crate::color_management::ColorManagementState::new(&dh);
    // wp_content_type_v1: clients tag a surface's content type (game / video /
    // photo). Advertised now so clients can hint; read from cached state when
    // we drive per-content behaviour (e.g. future tearing / scanout choices).
    let content_type_state =
        smithay::wayland::content_type::ContentTypeState::new::<State>(&dh);
    // wp_presentation: feeds clients accurate per-frame presentation timing
    // (the real vblank timestamp + sequence). Advertise CLOCK_MONOTONIC (1) —
    // the clock our DRM page-flip timestamps and feedback use.
    let presentation_state =
        smithay::wayland::presentation::PresentationState::new::<State>(&dh, 1);
    // wp_fifo_manager_v1: advertised (with the wp_presentation clock id)
    // so NVIDIA's Vulkan Wayland WSI runs VK_EXT_present_timing — the
    // KWin-parity config a Proton/DXVK HDR game needs. We deliberately do
    // NOT advertise wp_commit_timing_manager_v1: doing so flips the NVIDIA
    // WSI onto its absolute-time present path, which writes stage-timing
    // through an unallocated (NULL) buffer and SIGSEGVs — surfacing as the
    // vkGetPastPresentationTimingEXT assert. KWin ships fifo without
    // commit-timing and works; we match it.
    // `LIBRELAND_NO_FIFO_BLOCK=1` drops fifo to unmanaged (advertised but
    // never blocking) as a recovery valve.
    // (Bisect result: advertising fifo or not made NO difference to the
    // idTech freeze — fifo is exonerated and stays enabled.)
    let fifo_manager_state = Some(if std::env::var_os("LIBRELAND_NO_FIFO_BLOCK").is_some() {
        smithay::wayland::fifo::FifoManagerState::unmanaged::<State>(&dh)
    } else {
        smithay::wayland::fifo::FifoManagerState::new::<State>(&dh)
    });
    // xwayland_shell_v1: how Xwayland associates the wl_surface it
    // creates for each X11 window with that window (a shared serial;
    // see src/xwayland.rs). smithay only lets Xwayland clients bind it.
    let xwayland_shell_state =
        smithay::wayland::xwayland_shell::XWaylandShellState::new::<State>(&dh);
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
        dmabuf_default_feedback,
        dmabuf_scanout_feedback,
        dmabuf_feedback_seed,
        viewporter_state,
        layer_shell_state,
        relative_pointer_state,
        pointer_constraints_state,
        primary_selection_state,
        wlr_data_control_state,
        ext_data_control_state,
        idle_inhibit_state,
        xdg_activation_state,
        pointer_gestures_state,
        screencopy_manager,
        tearing_control,
        color_management,
        content_type_state,
        presentation_state,
        fifo_manager_state,
        xwayland_shell_state,
        popup_manager,
        outputs,
        output_globals,
        preferred_scale,
        background_effect_state,
        output_capture_source_state,
        image_copy_capture_state,
        xdg_dialog_state,
        pointer_warp_manager,
        fixes_state,
    })
}

/// Spawn each configured startup command as a child process. Each
/// entry is whitespace-split into program + args; children inherit
/// the compositor's environment (notably `$WAYLAND_DISPLAY`) so
/// they connect to *our* socket. A failed spawn is logged and
/// skipped — one broken command shouldn't crash the compositor.
/// The `Child` handles are returned for `State::children`, whose reap
/// timer `try_wait`s them — a dropped handle is never waited on, so
/// every exited child would linger as a zombie.
/// Autostart the bundled polkit authentication agent so GUI programs (and
/// `pkexec`) get a password prompt. It's a normal per-session process — a
/// sibling binary installed alongside the compositor — that registers with
/// polkitd and drives the themed quickshell dialog. Failure is non-fatal (the
/// desktop just won't have polkit prompts); the returned child is reaped like
/// any other startup process.
pub fn spawn_polkit_agent() -> Option<std::process::Child> {
    match std::process::Command::new("libreland-polkit-agent").spawn() {
        Ok(child) => {
            info!(pid = child.id(), "spawned polkit authentication agent");
            Some(child)
        }
        Err(err) => {
            warn!(error = %err, "failed to spawn libreland-polkit-agent (GUI polkit prompts won't work)");
            None
        }
    }
}

pub fn spawn_startup(commands: &[String]) -> Vec<std::process::Child> {
    let mut children = Vec::new();
    for raw in commands {
        let parts: Vec<&str> = raw.split_whitespace().collect();
        let Some((program, args)) = parts.split_first() else {
            warn!(command = raw, "startup command is empty — skipping");
            continue;
        };
        match std::process::Command::new(program).args(args).spawn() {
            Ok(child) => {
                info!(pid = child.id(), command = raw, "spawned startup command");
                children.push(child);
            }
            Err(err) => warn!(
                command = raw,
                error = %err,
                "failed to spawn startup command"
            ),
        }
    }
    children
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

/// Build the Scanout-tranche feedback variant for one output's plane
/// formats. Shared by [`init`] and the hot-plug path, which has to build the
/// same thing for an output that didn't exist at startup.
pub(crate) fn build_scanout_feedback(
    dev_id: u64,
    dmabuf_formats: &[Format],
    scanout_formats: Vec<Format>,
) -> Result<smithay::wayland::dmabuf::DmabufFeedback, std::io::Error> {
    DmabufFeedbackBuilder::new(dev_id, dmabuf_formats.to_vec())
        .add_preference_tranche(
            dev_id,
            smithay::reexports::wayland_protocols::wp::linux_dmabuf::zv1::server
                ::zwp_linux_dmabuf_feedback_v1::TrancheFlags::Scanout,
            scanout_formats,
            // Same version span the default feedback serves.
            3u32..=6,
        )
        .build()
}

/// Absolute `CLOCK_MONOTONIC` deadline ~1s out (nanoseconds) for the
/// explicit-sync CPU-wait fallback. `DrmSyncPoint::wait` takes an absolute
/// monotonic deadline; the 1s cap stops a never-signalling fence from wedging
/// the whole compositor on the (rare) eventfd-setup failure path — at worst
/// that one frame tears, versus the normal eventfd path which only stalls the
/// single surface.
fn acquire_wait_deadline() -> i64 {
    let now = std::time::Duration::from(
        smithay::utils::Clock::<smithay::utils::Monotonic>::new().now(),
    );
    i64::try_from(now.as_nanos().saturating_add(1_000_000_000)).unwrap_or(i64::MAX)
}

/// The per-client compositor state for both kinds of client we host:
/// regular Wayland clients carry our [`ClientState`] (attached in
/// `new_client_data` when the listening socket accepts them), while the
/// Xwayland client is inserted by smithay's `XWayland::spawn` with its
/// own `XWaylandClientData` — which also carries the client-scale
/// override that drives X11 `HiDPI`, so returning *its* state here is
/// what makes the scale mapping reach Xwayland's surfaces. `None` means
/// the client is gone (or was inserted by neither path, which doesn't
/// happen).
fn compositor_client_state(client: &Client) -> Option<&CompositorClientState> {
    if let Some(data) = client.get_data::<smithay::xwayland::XWaylandClientData>() {
        return Some(&data.compositor_state);
    }
    client
        .get_data::<ClientState>()
        .map(|data| &data.compositor_state)
}

impl CompositorHandler for State {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        compositor_client_state(client)
            .expect("client inserted without ClientState — see wayland::new_client_data")
    }

    fn new_surface(&mut self, surface: &WlSurface) {
        info!(surface = ?surface.id(), "wayland: new surface");
        // Explicit sync (linux-drm-syncobj-v1): gate this surface's commits on
        // its acquire fence. When a client (e.g. Proton/DXVK) commits a buffer
        // with an acquire timeline point, the GPU hasn't finished rendering it
        // yet. Hold the commit — so the compositor never composites OR
        // scans-out a half-rendered buffer (no tearing/glitch) — until the
        // acquire fence's eventfd signals, then re-apply the transaction. This
        // is the mechanism smithay's own DrmCompositor assumes; with it in
        // place a directly-scanned buffer is GPU-ready by render time.
        smithay::wayland::compositor::add_pre_commit_hook::<State, _>(
            surface,
            |state, _dh, surface| {
                if state.drm_syncobj_state.is_none() {
                    return;
                }
                // Build an eventfd-backed blocker for this surface's acquire
                // fence (if any). If setting up the eventfd blocker fails, fall
                // back to a bounded CPU wait so the commit is NEVER left
                // un-gated — correctness (no half-rendered buffer reaches the
                // screen) over the rare stall.
                let blocker_source =
                    smithay::wayland::compositor::with_states(surface, |states| {
                        let mut cached = states
                            .cached_state
                            .get::<smithay::wayland::drm_syncobj::DrmSyncobjCachedState>();
                        let acquire = cached.pending().acquire_point.as_ref()?;
                        // Already signalled (fast GPU / reused fence): no wait.
                        if acquire.is_signaled() {
                            return None;
                        }
                        match acquire.generate_blocker() {
                            Ok(blocker_source) => Some(blocker_source),
                            Err(err) => {
                                warn!(error = %err, "explicit sync: acquire blocker unavailable; CPU-waiting on fence");
                                let _ = acquire.wait(acquire_wait_deadline());
                                None
                            }
                        }
                    });
                let Some((blocker, source)) = blocker_source else {
                    return;
                };
                let Some(client) = surface.client() else {
                    return;
                };
                let inserted = state.loop_handle.insert_source(
                    source,
                    move |(), _metadata, state: &mut State| {
                        // The client may have disconnected while its fence was
                        // in flight; skip rather than panic in
                        // `client_compositor_state`'s expect. Checked through
                        // the same both-kinds helper — Xwayland (DXVK games
                        // running through it use explicit sync too) has
                        // XWaylandClientData, not our ClientState, and being
                        // skipped here would leave its commit blocked forever.
                        if compositor_client_state(&client).is_some() {
                            let dh = state.display_handle.clone();
                            state
                                .client_compositor_state(&client)
                                .blocker_cleared(state, &dh);
                        }
                        Ok(())
                    },
                );
                if inserted.is_ok() {
                    smithay::wayland::compositor::add_blocker(surface, blocker);
                } else {
                    // Couldn't register the eventfd source: CPU-wait now so the
                    // commit isn't left un-gated (would tear).
                    warn!("explicit sync: failed to register acquire source; CPU-waiting on fence");
                    smithay::wayland::compositor::with_states(surface, |states| {
                        let mut cached = states
                            .cached_state
                            .get::<smithay::wayland::drm_syncobj::DrmSyncobjCachedState>();
                        if let Some(acquire) = cached.pending().acquire_point.as_ref() {
                            let _ = acquire.wait(acquire_wait_deadline());
                        }
                    });
                }
            },
        );
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
        // A commit nobody will ever present — the window sits on a
        // hidden workspace, or the session is locked — gets its
        // wp_presentation feedback DISCARDED right here. Per the
        // protocol every feedback resolves exactly once; parking them
        // until the workspace returns starves present-timing consumers
        // (wine's vkGetPastPresentationTimingEXT asserts once its
        // timing queue errors — a game crashed on every workspace
        // switch this way).
        discard_hidden_presentation_feedback(self, surface);
        // Offscreen heartbeat (see [`root_offscreen`]): a toplevel-shaped
        // surface committing from outside the scene must still have its
        // pacing completed — fifo barrier + feedback are handled by the
        // discard call above; fire its frame callbacks here, throttled to
        // roughly a fast display's refresh so a misbehaving client can't
        // spin the compositor with commit→callback→commit loops. Without
        // this, an X11 game whose first commits land in the map→manage gap
        // starves, Xwayland drops to ~1 Hz present completions, and the
        // game freezes for good (black window, GPU busy).
        {
            let root = crate::root_surface(surface);
            if root_offscreen(self, &root) {
                let now = std::time::Instant::now();
                let due = self
                    .offscreen_frame_ts
                    .get(&root.id())
                    .is_none_or(|last| now.duration_since(*last) >= std::time::Duration::from_millis(3));
                if due {
                    if self.offscreen_frame_ts.insert(root.id(), now).is_none() {
                        debug!(root = ?root.id(), "offscreen heartbeat engaged (out-of-scene surface committing)");
                    }
                    crate::render::send_frame_callbacks(&root, self.renderer.frame_time_ms());
                }
            }
        }
        // Some clients ignore the size in the *initial* configure (sent before
        // they map) and render at their own default size, only resizing when a
        // later configure arrives — MPV's idle "Drop files" window is the
        // notorious one: it maps small inside its cell and snaps to size only
        // on the next configure (e.g. when the user moves it). The first time a
        // tracked toplevel commits a buffer, re-send its layout configure so it
        // resizes itself with no user interaction.
        if matches!(
            smithay::wayland::compositor::get_role(surface),
            Some(
                smithay::wayland::shell::xdg::XDG_TOPLEVEL_ROLE
                    | smithay::wayland::xwayland_shell::XWAYLAND_SHELL_ROLE
            )
        ) && !self.mapped_toplevels.contains(surface)
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
        // Floating xdg windows own their size: a client answering an
        // un-honourable configure (below its declared minimum) commits
        // its minimum instead, and dialogs resize themselves to fit
        // content. Adopt the committed window geometry into the stored
        // rect, or the decoration offscreen keeps compositing at the
        // wrong size (cropped content, mis-scaled corners, stale hit
        // rect). Skipped while an interactive drag owns this surface —
        // the drag's rect is authoritative and the client is merely
        // catching up to it.
        if self.mapped_toplevels.contains(surface)
            && self.drag.as_ref().is_none_or(|d| d.surface != *surface)
            // Only adopt a size the client CHOSE. A commit still carrying
            // geometry from before our latest configure is stale — the
            // client simply hasn't caught up — and adopting it would undo
            // our resize (and, if a reflow re-configured from the reverted
            // rect, ping-pong). `with_committed_state` is the state the
            // client acked for THIS commit: when its size matches what we
            // last asked for, the client is answering us, so any geometry
            // difference is the client's own decision (e.g. it refused a
            // size below its minimum).
            && self
                .layout
                .window_surface(surface)
                .and_then(|w| match w {
                    crate::layout::WindowSurface::Xdg(t) => Some(t),
                    crate::layout::WindowSurface::X11 { .. } => None,
                })
                .is_some_and(|t| {
                    let requested = t.with_pending_state(|s| s.size);
                    let confirmed = t.with_committed_state(|s| s.and_then(|st| st.size));
                    // No configure in flight (the client confirmed what we
                    // last set), or we never sized it at all.
                    requested.is_none() || requested == confirmed
                })
        {
            let committed = smithay::wayland::compositor::with_states(surface, |states| {
                states
                    .cached_state
                    .get::<smithay::wayland::shell::xdg::SurfaceCachedState>()
                    .current()
                    .geometry
                    .map(|g| g.size)
            })
            .or_else(|| {
                // No explicit window geometry: the surface extent is the
                // window per xdg-shell.
                smithay::backend::renderer::utils::with_renderer_surface_state(surface, |s| {
                    s.surface_size()
                })
                .flatten()
            });
            if let Some(size) = committed
                && self.layout.reconcile_floating_size(
                    surface,
                    smithay::utils::Size::from((size.w, size.h)),
                )
            {
                debug!(
                    surface = ?surface.id(),
                    w = size.w,
                    h = size.h,
                    "float: adopted client-committed size"
                );
                self.queue_redraw_for_surface(surface);
            }
        }
        // Track fifo-barrier surfaces for the fallback clearer (see
        // [`fifo_fallback_tick`]). Only NVIDIA's Wayland WSI (and other
        // fifo-v1 clients) ever set barriers, so the map stays tiny.
        {
            let has_barrier = smithay::wayland::compositor::with_states(surface, |states| {
                states
                    .cached_state
                    .get::<smithay::wayland::fifo::FifoBarrierCachedState>()
                    .current()
                    .barrier
                    .is_some()
            });
            if has_barrier {
                self.fifo_barrier_watch
                    .entry(surface.id())
                    .or_insert_with(|| (surface.clone(), std::time::Instant::now()));
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
        // And the offscreen-heartbeat throttle entry + fifo watch entry.
        self.offscreen_frame_ts.remove(&surface.id());
        self.fifo_barrier_watch.remove(&surface.id());
        // Same for the sticky scanout-feedback record, the tearing hint and
        // the direct-scanout marker.
        self.scanout_feedback_given.borrow_mut().remove(&surface.id());
        self.renderer.forget_surface_scanout_state(&surface.id());
        self.tearing_controls.retain(|s| s != surface);
        // Color-management records are otherwise only removed by explicit
        // client requests — a client that crashes without sending destroy
        // (a Proton game, say) would leak its entries.
        self.color_surfaces.remove(&surface.id());
        self.color_surface_objects.remove(&surface.id());
        // And the renderer's per-surface caches (decoration offscreen).
        self.renderer.forget_surface(surface);
    }
}

/// Discard the `wp_presentation` feedback of a commit that nothing will
/// ever present: the committing window sits on a non-active workspace,
/// or the session is locked (where only the lock surfaces reach the
/// screen). Resolved immediately at commit time — the only point where
/// we reliably see every hidden producer — so a game left running on a
/// hidden workspace keeps getting `discarded` events instead of an
/// ever-growing queue of feedback that never fires. Visible surfaces
/// are untouched; their feedback resolves through the per-frame
/// collect → vblank `presented` path.
fn discard_hidden_presentation_feedback(state: &mut State, surface: &WlSurface) {
    use smithay::desktop::utils::SurfacePresentationFeedback;
    use smithay::reexports::wayland_protocols::wp::presentation_time::server::wp_presentation_feedback;

    let root = crate::root_surface(surface);
    let hidden = if state.session_locked {
        !state
            .lock_surfaces
            .values()
            .any(|lock| lock.wl_surface() == &root)
    } else {
        state.layout.on_inactive_workspace(&root) || root_offscreen(state, &root)
    };
    if !hidden {
        return;
    }
    // Same rationale, fifo edge: an inactive-workspace surface is never
    // presented, so release its fifo barrier here or its next frame hangs.
    signal_hidden_fifo_barrier(&root);
    smithay::wayland::compositor::with_states(surface, |states| {
        if let Some(mut feedback) =
            SurfacePresentationFeedback::from_states(states, wp_presentation_feedback::Kind::empty())
        {
            feedback.discarded();
        }
    });
}

/// Whether a toplevel-shaped ROOT surface is committing from *outside the
/// rendered scene*: an xdg toplevel or `XWayland` surface that is in no
/// workspace (`window_surface` finds nothing anywhere) and isn't a tracked
/// override-redirect window. Two ways to get here: the map→manage gap (an
/// X11 window's first commits land BEFORE `try_manage_x11` places it — seen
/// in every game trace), and windows the WM never places at all.
///
/// Such a surface gets none of the per-frame pacing the render loop
/// provides (frame callbacks / fifo barriers / presentation feedback are
/// vblank-driven for scene surfaces only) — and a starved surface is not a
/// cosmetic problem: Xwayland falls back to ~1 Hz present completions when
/// frame callbacks stop, and games freeze on that PERMANENTLY, staying
/// frozen even after callbacks resume (`KWin` documents this exact failure
/// and ships a heartbeat timer for it; see `X11Window::wantsFrameCallbackHeartbeat`).
/// The commit handler therefore completes pacing for these surfaces
/// immediately (the "offscreen heartbeat").
///
/// Only toplevel-ish roles qualify: layers, popups, cursors, `DnD` icons and
/// lock surfaces all have their own render-path pacing.
fn root_offscreen(state: &State, root: &WlSurface) -> bool {
    if !matches!(
        smithay::wayland::compositor::get_role(root),
        Some(
            smithay::wayland::shell::xdg::XDG_TOPLEVEL_ROLE
                | smithay::wayland::xwayland_shell::XWAYLAND_SHELL_ROLE
        )
    ) {
        return false;
    }
    state.layout.window_surface(root).is_none()
        && !state.x11_or_windows.iter().any(|(_, s)| s == root)
}

/// Clear the `wp_fifo` barriers set by these root surfaces' latest
/// applied commit, then re-drive each affected client so a commit it
/// held waiting on the barrier can proceed.
///
/// The fifo protocol says a `set_barrier` condition clears "immediately
/// after the following latching deadline" — i.e. once the frame carrying
/// it has been latched for the coming refresh. That deadline is exactly
/// the vblank we call this from, so a client in FIFO present mode (a
/// vsync'd game) is paced to one frame in flight: its next `wait_barrier`
/// commit stays blocked until here. Signalling the [`Barrier`] alone
/// isn't enough — smithay only re-evaluates a held commit's blockers on
/// [`CompositorClientState::blocker_cleared`], so we prod each client
/// whose surface actually had a barrier.
///
/// [`Barrier`]: smithay::wayland::compositor::Barrier
pub(crate) fn signal_fifo_barriers(state: &mut State, roots: &[WlSurface]) {
    use smithay::wayland::fifo::FifoBarrierCachedState;
    if roots.is_empty() {
        return;
    }
    let mut clients = Vec::new();
    let mut cleared_ids = Vec::new();
    for root in roots {
        let mut had_barrier = false;
        smithay::wayland::compositor::with_surface_tree_downward(
            root,
            (),
            |_, _, ()| smithay::wayland::compositor::TraversalAction::DoChildren(()),
            |surface, states, ()| {
                if let Some(barrier) = states
                    .cached_state
                    .get::<FifoBarrierCachedState>()
                    .current()
                    .barrier
                    .take()
                {
                    barrier.signal();
                    had_barrier = true;
                    cleared_ids.push(surface.id());
                }
            },
            |_, _, ()| true,
        );
        if had_barrier && let Some(client) = root.client() {
            clients.push(client);
        }
    }
    // A vblank-cleared surface leaves the fallback watch; its next
    // barrier-setting commit re-enters it with a fresh timestamp, so the
    // fallback only ever sees barriers that vblank clearing MISSED.
    for id in cleared_ids {
        state.fifo_barrier_watch.remove(&id);
    }
    let dh = state.display_handle.clone();
    for client in clients {
        // Xwayland clients don't carry our ClientState; skip (their
        // blocker bookkeeping lives elsewhere), matching the explicit-sync
        // blocker path.
        if compositor_client_state(&client).is_some() {
            state.client_compositor_state(&client).blocker_cleared(state, &dh);
        }
    }
}

/// Signal the explicit-sync RELEASE points of these roots' current
/// (displayed) buffers — called at vblank for COMPOSITED frames only.
///
/// smithay's stock model signals a buffer's release point when the NEXT
/// buffer commit merges over it (hold-until-replaced). Vulkan clients
/// tearing down a swapchain wait for every image's release point BEFORE
/// they will ever commit again — idTech games rebuild their swapchain
/// constantly, so hold-until-replaced deadlocks them at a
/// timing-dependent rebuild (games froze after 1/12/38/99 frames; the
/// `WAYLAND_DEBUG` trace showed the wedge with no successor commit ever
/// coming). After a composited flip the displayed image is the
/// compositor's own framebuffer, not the client buffer, so releasing is
/// safe; the one caveat is a later damage-repaint re-sampling a released
/// buffer the client has begun reusing — a transient artifact at worst,
/// against a guaranteed permanent freeze. The point is `take()`n so
/// `merge_into` can't double-signal.
pub(crate) fn signal_release_points(roots: &[WlSurface]) {
    use smithay::wayland::drm_syncobj::DrmSyncobjCachedState;
    for root in roots {
        smithay::wayland::compositor::with_surface_tree_downward(
            root,
            (),
            |_, _, ()| smithay::wayland::compositor::TraversalAction::DoChildren(()),
            |_s, states, ()| {
                let mut cached = states.cached_state.get::<DrmSyncobjCachedState>();
                if let Some(release_point) = cached.current().release_point.take()
                    && let Err(err) = release_point.signal()
                {
                    warn!(error = %err, "explicit sync: release-point signal failed");
                }
            },
            |_, _, ()| true,
        );
    }
}

/// KWin-style fifo fallback (its `fifoFallbackTimer`): force-clear any
/// `wp_fifo` barrier that has stood longer than a few frames, then re-drive
/// the owning client. The vblank path above only clears barriers for
/// surfaces of RENDERED frames — a fifo client whose next commit is the
/// only thing that would trigger a render deadlocks with it (blocked
/// commit → no damage → no render → no vblank → barrier never cleared),
/// and NVIDIA's Wayland WSI games froze after exactly a swapchain of
/// frames this way. The fifo spec explicitly permits clearing early "to
/// ensure client forward progress". Runs from a 25 ms timer; 60 ms of age
/// means several missed vblanks even at 60 Hz, so healthy vblank-paced
/// clients never hit it.
pub(crate) fn fifo_fallback_tick(state: &mut State) {
    use smithay::wayland::fifo::FifoBarrierCachedState;
    if state.fifo_barrier_watch.is_empty() {
        return;
    }
    let stale_after = std::time::Duration::from_millis(60);
    let now = std::time::Instant::now();
    let mut clients = Vec::new();
    let mut drop_ids = Vec::new();
    for (id, (surface, since)) in &state.fifo_barrier_watch {
        if !surface.is_alive() {
            drop_ids.push(id.clone());
            continue;
        }
        let cleared_elsewhere = smithay::wayland::compositor::with_states(surface, |states| {
            states
                .cached_state
                .get::<FifoBarrierCachedState>()
                .current()
                .barrier
                .is_none()
        });
        if cleared_elsewhere {
            drop_ids.push(id.clone());
            continue;
        }
        if now.duration_since(*since) < stale_after {
            continue;
        }
        smithay::wayland::compositor::with_states(surface, |states| {
            if let Some(barrier) = states
                .cached_state
                .get::<FifoBarrierCachedState>()
                .current()
                .barrier
                .take()
            {
                barrier.signal();
            }
        });
        debug!(surface = ?id, "fifo fallback: cleared stale barrier (no vblank was coming)");
        if let Some(client) = surface.client() {
            clients.push(client);
        }
        drop_ids.push(id.clone());
    }
    for id in drop_ids {
        state.fifo_barrier_watch.remove(&id);
    }
    let dh = state.display_handle.clone();
    for client in clients {
        if compositor_client_state(&client).is_some() {
            state.client_compositor_state(&client).blocker_cleared(state, &dh);
        }
    }
}

/// Signal a hidden surface's `wp_fifo` barrier at commit time so a FIFO
/// client on an inactive workspace keeps making forward progress. Such a
/// surface is never presented, so [`signal_fifo_barriers`]' vblank path
/// never fires for it, and its next `wait_barrier` commit would block
/// forever. The spec explicitly lets the compositor clear the condition
/// for a surface it isn't updating. No `blocker_cleared` here: this runs
/// inside commit handling, and pre-signalling means the *next* commit
/// simply won't be blocked (smithay skips an already-signalled barrier).
fn signal_hidden_fifo_barrier(surface: &WlSurface) {
    use smithay::wayland::fifo::FifoBarrierCachedState;
    smithay::wayland::compositor::with_surface_tree_downward(
        surface,
        (),
        |_, _, ()| smithay::wayland::compositor::TraversalAction::DoChildren(()),
        |_surface, states, ()| {
            if let Some(barrier) = states
                .cached_state
                .get::<FifoBarrierCachedState>()
                .current()
                .barrier
                .take()
            {
                barrier.signal();
            }
        },
        |_, _, ()| true,
    );
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
    // First commit that actually carries a buffer is the map: play the open
    // animation from here. Tracked in the same set the toplevel path uses so
    // a re-commit doesn't restart it.
    if !state.mapped_layers.contains(surface)
        && smithay::backend::renderer::utils::with_renderer_surface_state(surface, |s| {
            s.buffer().is_some()
        })
        .unwrap_or(false)
    {
        state.mapped_layers.insert(surface.clone());
        state.renderer.mark_layer_open(surface);
    }
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
        // X11 windows additionally need X-side input focus
        // (SetInputFocus / WM_TAKE_FOCUS) — wl_keyboard events alone
        // aren't accepted by X clients. No-op for Wayland↔Wayland moves.
        self.sync_x11_focus(focused);
    }

    fn led_state_changed(
        &mut self,
        _seat: &Seat<Self>,
        led_state: smithay::input::keyboard::LedState,
    ) {
        // xkb tracks the lock-key state, but the physical Caps/Num Lock
        // lights only change if someone hands that state to libinput —
        // without this impl (the trait default is a no-op) they never lit.
        self.keyboard_leds = led_state;
        self.apply_keyboard_leds();
    }

    fn cursor_image(&mut self, _seat: &Seat<Self>, image: CursorImageStatus) {
        // The focused client set its pointer image — a surface
        // (`wl_pointer.set_cursor`, used by toolkits and games incl.
        // Xwayland forwarding an X11 cursor), a named shape
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
impl TabletSeatHandler for State {
    // `wp_cursor_shape_v1` covers tablet tools, so its dispatch requires
    // this handler. No tablets are advertised on the seat; the focus type
    // just has to name a valid target.
    type ToolFocus = WlSurface;
}

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

    fn remove_constraint(&mut self, _surface: &WlSurface, _pointer: &PointerHandle<Self>) {
        // A removed lock/confine can unhide the cursor (visibility keys
        // off the active constraint each frame) — repaint promptly.
        self.queue_redraw_all();
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

impl smithay::wayland::shell::xdg::dialog::XdgDialogHandler for State {
    /// A client declared (or cleared) its toplevel's dialog/modal hint.
    /// smithay has already stored the hint in the toplevel's role data —
    /// `layout::dialog_size` reads it at map time. The hint usually
    /// arrives before the first commit; when it lands on an
    /// already-mapped tiled window, re-run the auto-float so the
    /// declaration still takes effect.
    fn dialog_hint_changed(
        &mut self,
        toplevel: smithay::wayland::shell::xdg::ToplevelSurface,
        hint: smithay::wayland::shell::xdg::dialog::ToplevelDialogHint,
    ) {
        use smithay::wayland::shell::xdg::dialog::ToplevelDialogHint;
        if matches!(hint, ToplevelDialogHint::Dialog | ToplevelDialogHint::Modal)
            && self.mapped_toplevels.contains(toplevel.wl_surface())
            && self.layout.float_if_dialog(toplevel.wl_surface())
        {
            self.queue_redraw_all();
        }
    }
}

impl smithay::wayland::background_effect::ExtBackgroundEffectHandler for State {
    // Region storage needs no overrides: smithay stores the pending blur
    // region in the surface's cached state at commit, and the renderer
    // reads it per frame (`render::surface_requests_blur`). The commit
    // itself already queues the redraw that makes the frost
    // appear/disappear.

    /// Advertise blur only when the config's blur pipeline can actually
    /// deliver it — a capability-keyed client otherwise renders itself
    /// translucent expecting frost that never comes. Sent at bind time,
    /// so a live `blur.enabled` flip only reaches clients that bind
    /// afterwards (protocol limitation; existing binds keep their
    /// snapshot).
    fn capabilities(&self) -> smithay::wayland::background_effect::Capability {
        let blur = &self.config.decoration.blur;
        if blur.enabled && blur.passes > 0 {
            smithay::wayland::background_effect::Capability::Blur
        } else {
            smithay::wayland::background_effect::Capability::empty()
        }
    }
}

impl smithay::wayland::pointer_warp::PointerWarpHandler for State {
    /// A client asked to move the pointer to a surface-local position
    /// (`wp_pointer_warp_v1` — games and emulators recentring the cursor
    /// without taking a full pointer lock).
    ///
    /// Honoured only when `surface` is the surface currently holding
    /// pointer focus (the protocol's own baseline) and nothing else owns
    /// the pointer: not during a compositor drag, screenshot selection,
    /// client grab, or an active pointer lock (constraints forbid
    /// synthetic motion while locked — the client should use the lock's
    /// own position hint instead). The target is clamped into the
    /// focused surface's global rect, so a stale/oversized position can
    /// never throw the cursor onto another window. The enter serial is
    /// not cross-checked: the focus requirement already pins the request
    /// to the surface the pointer is actually on.
    fn warp_pointer(
        &mut self,
        surface: WlSurface,
        _pointer: smithay::reexports::wayland_server::protocol::wl_pointer::WlPointer,
        pos: smithay::utils::Point<f64, smithay::utils::Logical>,
        _serial: Serial,
    ) {
        let Some(pointer) = self.seat.get_pointer() else {
            return;
        };
        // ANY active constraint refuses the warp — not just a lock. A
        // confined client warping outside its confine region would have
        // the forced focus change deactivate its own confinement; a
        // client wanting to steer a locked pointer has the lock's
        // position hint.
        let constrained = pointer.current_focus().as_ref().is_some_and(|focus| {
            with_pointer_constraint(focus, &pointer, |constraint| {
                constraint.is_some_and(|c| c.is_active())
            })
        });
        if self.drag.is_some() || self.screenshot.is_some() || pointer.is_grabbed() || constrained
        {
            return;
        }
        if pointer.current_focus().as_ref() != Some(&surface) {
            return;
        }
        // The focused surface's global origin comes from the same
        // hit-test that granted it focus, so surface-local + origin is
        // exactly the space the client computed `pos` in.
        let (cx, cy) = self.renderer.cursor_pos();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "cursor coords are clamped to layout_bounds (i32) in Renderer::on_pointer_motion"
        )]
        let cursor_i =
            smithay::utils::Point::<i32, smithay::utils::Physical>::from((cx as i32, cy as i32));
        let (hit, _, _) = self.pointer_hit_test(cursor_i);
        let Some((hit_surface, origin)) = hit else {
            return;
        };
        if hit_surface != surface {
            return;
        }
        let bbox = smithay::desktop::utils::bbox_from_surface_tree(&surface, (0, 0));
        let tx = origin.x + pos.x.clamp(0.0, f64::from(bbox.size.w.max(0)));
        let ty = origin.y + pos.y.clamp(0.0, f64::from(bbox.size.h.max(0)));
        // The bbox spans the full buffer including any CSD shadow margin,
        // which hangs OUTSIDE the window's hit rect — a warp into it (or
        // any stale position) must not land the pointer on a neighbouring
        // window, a gap, or an overlaying popup. Only honour targets that
        // hit-test back to the requesting surface; refuse the rest
        // outright rather than second-guessing where the client wanted.
        #[allow(
            clippy::cast_possible_truncation,
            reason = "target coords derive from output-bounded rects and a clamped pos"
        )]
        let target_i =
            smithay::utils::Point::<i32, smithay::utils::Physical>::from((tx as i32, ty as i32));
        let (target_hit, _, _) = self.pointer_hit_test(target_i);
        if target_hit.map(|(s, _)| s) != Some(surface.clone()) {
            debug!(surface = ?surface.id(), x = tx, y = ty, "wp_pointer_warp: target outside the surface's hit rect; refused");
            return;
        }
        self.renderer.on_pointer_motion(tx - cx, ty - cy);
        self.resend_pointer_focus(true);
        if self.renderer.has_dnd_icon() || !self.renderer.move_hw_cursor() {
            self.queue_redraw_all();
        }
        debug!(surface = ?surface.id(), x = tx, y = ty, "wp_pointer_warp: pointer warped");
    }
}

impl XdgShellHandler for State {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell_state
    }

    /// `xdg_toplevel.move`: a client-side-decorated window is being
    /// dragged by the titlebar it drew itself.
    ///
    /// A CSD window has no server chrome to grab, so this is its only
    /// way to be moved. Both shells route through the same entry point;
    /// see `State::begin_client_drag`.
    fn move_request(
        &mut self,
        surface: ToplevelSurface,
        _seat: smithay::reexports::wayland_server::protocol::wl_seat::WlSeat,
        _serial: smithay::utils::Serial,
    ) {
        self.begin_client_drag(surface.wl_surface(), crate::DragMode::Move, None);
    }

    /// `xdg_toplevel.resize`: the client is dragging one of its own
    /// resize handles.
    fn resize_request(
        &mut self,
        surface: ToplevelSurface,
        _seat: smithay::reexports::wayland_server::protocol::wl_seat::WlSeat,
        _serial: smithay::utils::Serial,
        edges: smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel::ResizeEdge,
    ) {
        use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel::ResizeEdge as E;
        // A side pins the other axis; a corner takes both. `None`
        // (the client left it unspecified) resizes from the bottom
        // right, which is what a resize with no stated edge means
        // everywhere else.
        let (x, y) = match edges {
            E::Top => (None, Some(crate::layout::EdgeY::Top)),
            E::Bottom => (None, Some(crate::layout::EdgeY::Bottom)),
            E::Left => (Some(crate::layout::EdgeX::Left), None),
            E::Right => (Some(crate::layout::EdgeX::Right), None),
            E::TopLeft => (
                Some(crate::layout::EdgeX::Left),
                Some(crate::layout::EdgeY::Top),
            ),
            E::TopRight => (
                Some(crate::layout::EdgeX::Right),
                Some(crate::layout::EdgeY::Top),
            ),
            E::BottomLeft => (
                Some(crate::layout::EdgeX::Left),
                Some(crate::layout::EdgeY::Bottom),
            ),
            E::BottomRight | E::None | _ => (
                Some(crate::layout::EdgeX::Right),
                Some(crate::layout::EdgeY::Bottom),
            ),
        };
        self.begin_client_drag(
            surface.wl_surface(),
            crate::DragMode::Resize,
            Some(crate::layout::ResizeEdges { x, y }),
        );
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
        self.layout
            .insert(crate::layout::WindowSurface::Xdg(surface.clone()), Some(cursor));
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
        // Stamp the constrained geometry into pending state before the
        // first configure so the popup reports the right size/anchor and,
        // crucially, is kept on-screen (flip/slide/resize per the client's
        // constraint_adjustment) instead of opening into the void near a
        // screen edge; then track it so it joins its parent's tree (and so
        // the renderer can find + place it).
        let geometry = self.unconstrain_popup_geometry(&surface, &positioner);
        surface.with_pending_state(|state| {
            state.geometry = geometry;
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

    fn grab(&mut self, surface: PopupSurface, _seat: WlSeat, serial: Serial) {
        // A full smithay `PopupGrab` needs `SeatHandler::KeyboardFocus:
        // From<PopupKind>` (ours is `WlSurface`), so we can't use it
        // without a focus-type refactor. Instead, honour the grab by
        // pinning keyboard focus to the popup's root toplevel/layer
        // surface and recording the grab: while it's set, the hover focus
        // model is frozen (see `forward_pointer_motion`), so moving the
        // pointer off the parent no longer makes the client dismiss its
        // own menu. Dismiss-on-click-outside stays in `forward_pointer_button`,
        // and the grab expires via `refresh_popup_grab` once the chain closes.
        let Ok(root) = find_popup_root_surface(&PopupKind::Xdg(surface)) else {
            return;
        };
        if let Some(kbd) = self.seat.get_keyboard()
            && kbd.current_focus().as_ref() != Some(&root)
        {
            kbd.set_focus(self, Some(root.clone()), serial);
        }
        self.popup_grab = Some(root);
    }

    fn reposition_request(
        &mut self,
        surface: PopupSurface,
        positioner: PositionerState,
        token: u32,
    ) {
        // Recompute constrained geometry from the new positioner and tell
        // the client (send_repositioned must precede the configure so it
        // can correlate the new geometry with its token).
        let geometry = self.unconstrain_popup_geometry(&surface, &positioner);
        surface.with_pending_state(|state| {
            state.geometry = geometry;
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
        // If the pointer was focused on (and possibly locked to) the
        // dying window — a game exiting with the cursor hidden — hand
        // pointer focus to whatever takes its cell, so the constraint
        // deactivates and the cursor image resets instead of staying
        // hidden. Safe on a dying surface: the first physical motion
        // after any close already walks this exact leave/replace path.
        self.refresh_pointer_focus();
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
        // Default to server-side before the first configure, so a client
        // that expresses no preference never starts with CSD and then has
        // to redraw without it.
        self.apply_decoration_mode(&toplevel, DecorationMode::ServerSide);
    }

    fn request_mode(&mut self, toplevel: ToplevelSurface, mode: DecorationMode) {
        // The client's preference is honoured. It used to be ignored —
        // reasonable while the compositor drew nothing but a hairline
        // border, since a client drawing its own on top of that costs
        // almost nothing. With a titlebar it is the double-titlebar
        // every desktop has a bug report about, so a client that says it
        // draws its own gets to.
        self.apply_decoration_mode(&toplevel, mode);
    }

    fn unset_mode(&mut self, toplevel: ToplevelSurface) {
        // Client dropped its preference: back to the default.
        self.apply_decoration_mode(&toplevel, DecorationMode::ServerSide);
    }
}

impl State {
    /// Answer a decoration negotiation and tell the layout, so the
    /// window's cell stops reserving space for a bar we won't draw.
    fn apply_decoration_mode(&mut self, toplevel: &ToplevelSurface, mode: DecorationMode) {
        let csd = matches!(mode, DecorationMode::ClientSide);
        toplevel.with_pending_state(|state| {
            state.decoration_mode = Some(mode);
        });
        self.layout.set_csd(toplevel.wl_surface(), csd);
        // Always, even though `set_csd` may also have reconfigured. This
        // usually runs during the *initial* configure sequence, before
        // the surface has a buffer — so before it is in the layout at
        // all, and nothing else would answer. A client left waiting on
        // that first configure never maps. A duplicate configure on the
        // post-map path is harmless by comparison.
        toplevel.send_configure();
        debug!(
            surface = ?toplevel.wl_surface().id(),
            ?mode,
            csd,
            "xdg-decoration: mode negotiated",
        );
        self.queue_redraw_all();
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
        // A Wayland client took (or cleared) the selection, so any X11
        // ownership is over — Wayland pastes are served from our cache
        // again, not routed through the XWM.
        self.x11_owns_selection.set(ty, false);
        let mimes = source.map(|s| s.mime_types());
        // Mirror the selection into the X world so X clients can paste
        // what Wayland clients copy (`None` clears the X-side offer).
        if let Some(xwm) = &mut self.xwm
            && let Err(err) = xwm.new_selection(ty, mimes.clone())
        {
            warn!(?ty, error = %err, "failed to forward the selection to Xwayland");
        }
        crate::clipboard::on_new_selection(self, ty, mimes);
    }

    fn send_selection(
        &mut self,
        ty: SelectionTarget,
        mime_type: String,
        fd: std::os::fd::OwnedFd,
        _seat: Seat<State>,
        _user_data: &(),
    ) {
        // When an X11 client owns the selection the bytes live on the X
        // side — have Xwayland fetch and stream them into the paster's
        // fd. Otherwise serve from the compositor's clipboard cache.
        if self.x11_owns_selection.owns(ty)
            && let Some(xwm) = &mut self.xwm
        {
            if let Err(err) = xwm.send_selection(ty, mime_type, fd) {
                warn!(?ty, error = %err, "failed to request the selection from Xwayland");
            }
            return;
        }
        crate::clipboard::on_send_selection(self, ty, &mime_type, fd);
    }
}

impl WaylandDndGrabHandler for State {
    /// A client requested a drag as response to a pointer action. Unlike
    /// the pre-0.7-git API (which installed the grab itself), the handler
    /// now owns installing the [`DnDGrab`] — the grab then routes the
    /// offer to whatever surface pointer focus lands on. We composite
    /// the drag icon at the cursor for the duration. Touch drags are
    /// declined (no touch devices are advertised on this seat).
    fn dnd_requested<S: DndSource>(
        &mut self,
        source: S,
        icon: Option<WlSurface>,
        seat: Seat<Self>,
        serial: Serial,
        type_: GrabType,
    ) {
        let GrabType::Pointer = type_ else {
            source.cancel();
            return;
        };
        let Some(pointer) = seat.get_pointer() else {
            source.cancel();
            return;
        };
        let Some(start_data) = pointer.grab_start_data() else {
            source.cancel();
            return;
        };
        // Install the grab BEFORE setting the icon: set_grab unsets any
        // still-active DnD grab, whose `cancelled` clears the icon — the
        // other order wipes the icon set for THIS drag. `Focus::Clear`
        // matches 0.7.0's StartDrag path (the origin surface gets its
        // wl_pointer.leave at drag start, not at the first motion).
        let grab = DnDGrab::new_pointer(&self.display_handle, start_data, source, seat.clone());
        pointer.set_grab(self, grab, serial, GrabFocus::Clear);
        self.renderer.set_dnd_icon(icon);
        // Show/hide the drag icon (it then follows the cursor via motion).
        self.queue_redraw_nonfullscreen();
    }
}

impl DndGrabHandler for State {
    /// The drag ended in a drop — remove the icon.
    fn dropped(
        &mut self,
        _target: Option<DndTarget<'_, Self>>,
        _validated: bool,
        _seat: Seat<Self>,
        _location: smithay::utils::Point<f64, smithay::utils::Logical>,
    ) {
        self.renderer.set_dnd_icon(None);
        self.queue_redraw_nonfullscreen();
    }

    /// The drag was cancelled — same cleanup.
    fn cancelled(
        &mut self,
        _seat: Seat<Self>,
        _location: smithay::utils::Point<f64, smithay::utils::Logical>,
    ) {
        self.renderer.set_dnd_icon(None);
        self.queue_redraw_nonfullscreen();
    }
}

impl DataDeviceHandler for State {
    fn data_device_state(&mut self) -> &mut DataDeviceState {
        &mut self.data_device_state
    }
}

impl PrimarySelectionHandler for State {
    fn primary_selection_state(&mut self) -> &mut PrimarySelectionState {
        &mut self.primary_selection_state
    }
}

// data-control (wlr + ext): both protocols drive selections through the
// shared `SelectionHandler` above, so a clipboard manager setting the
// selection routes through `crate::clipboard` exactly like a normal
// client would — no extra handling needed beyond exposing the state.
impl WlrDataControlHandler for State {
    fn data_control_state(&mut self) -> &mut WlrDataControlState {
        &mut self.wlr_data_control_state
    }
}

impl ExtDataControlHandler for State {
    fn data_control_state(&mut self) -> &mut ExtDataControlState {
        &mut self.ext_data_control_state
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
        self.sync_idle_inhibition();
    }

    fn uninhibit(&mut self, surface: WlSurface) {
        self.idle_inhibitors.remove(&surface);
        self.sync_idle_inhibition();
    }
}

// ext-idle-notify: lets idle daemons (swayidle, etc.) learn when the
// user goes idle. We feed it `notify_activity` on every input event
// (see `State::note_input_activity`) and pause it while an idle
// inhibitor is active (`State::sync_idle_inhibition`).
impl IdleNotifierHandler for State {
    fn idle_notifier_state(&mut self) -> &mut IdleNotifierState<Self> {
        &mut self.idle_notifier
    }
}

// xdg-activation: a client asks us to focus a surface (it has a token it
// got from another client — e.g. a terminal handing a token to the
// program it launches). We honour it as a reveal + keyboard focus,
// ignoring stale tokens as basic focus-stealing prevention.
impl XdgActivationHandler for State {
    fn activation_state(&mut self) -> &mut XdgActivationState {
        &mut self.xdg_activation_state
    }

    fn request_activation(
        &mut self,
        _token: XdgActivationToken,
        token_data: XdgActivationTokenData,
        surface: WlSurface,
    ) {
        // Drop activations older than 10 s: a token that's been sitting
        // around shouldn't be able to yank focus out from under you.
        if token_data.timestamp.elapsed().as_secs() < 10 {
            self.focus_surface(&surface);
        }
    }
}

// GPU buffer sharing. When a client (or Xwayland)
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


// KDE server-side decoration. We force Server for every decoration
// object regardless of what the client asks for — Libreland is a
// tiler and draws no decorations, so "server-side" here means "no
// titlebar at all". Combined with the manager's Server default mode,
// this stops GTK/Firefox from drawing a client-side titlebar.
impl KdeDecorationHandler for State {
    fn kde_decoration_state(&self) -> &KdeDecorationState {
        &self.kde_decoration_state
    }

    fn new_decoration(&mut self, surface: &WlSurface, decoration: &OrgKdeKwinServerDecoration) {
        decoration.mode(KdeMode::Server);
        self.layout.set_csd(surface, false);
    }

    fn request_mode(
        &mut self,
        surface: &WlSurface,
        decoration: &OrgKdeKwinServerDecoration,
        mode: WEnum<KdeMode>,
    ) {
        // The older KDE decoration protocol, which some toolkits use
        // instead of xdg-decoration. Honoured for the same reason: a
        // server-side bar over a client-side one is a double titlebar,
        // and the client is the one that knows.
        let csd = matches!(mode, WEnum::Value(KdeMode::Client | KdeMode::None));
        decoration.mode(if csd {
            KdeMode::Client
        } else {
            KdeMode::Server
        });
        self.layout.set_csd(surface, csd);
        self.queue_redraw_all();
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
        // Snapshot it before anything else: the wl_surface is still alive
        // here (smithay resets its attributes *after* this callback), so
        // there is a last frame to capture and fade back out.
        //
        // The rect is computed from the surface directly rather than looked
        // up in `snapshot_layer_placements`: smithay removes the entry from
        // its known-layers list before calling us, so the lookup found
        // nothing and every close animation was silently skipped.
        let rect = self.layer_placement_of(surface.wl_surface()).rect;
        self.renderer.mark_layer_closing(surface.wl_surface(), rect);
        self.layer_outputs.remove(surface.wl_surface());
        self.layer_namespaces.remove(surface.wl_surface());
        self.mapped_layers.remove(surface.wl_surface());
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

// One macro replaces the per-protocol delegate_* family upstream removed:
// it blanket-implements wayland-server's Dispatch/GlobalDispatch for every
// user-data type that implements smithay's Dispatch2/GlobalDispatch2 —
// i.e. all of smithay's protocols. Libreland's OWN protocols
// (color_management, screencopy) keep their direct old-style
// `impl Dispatch<…> for State` blocks: those coexist with this blanket
// because their user-data types never implement Dispatch2, so there is
// no overlap — do NOT "convert" them on a future rebase.
smithay::delegate_dispatch2!(State);

impl smithay::wayland::drm_syncobj::DrmSyncobjHandler for State {
    fn drm_syncobj_state(
        &mut self,
    ) -> Option<&mut smithay::wayland::drm_syncobj::DrmSyncobjState> {
        self.drm_syncobj_state.as_mut()
    }
}

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
