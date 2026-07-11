//! Libreland: a Wayland compositor in pure Rust, configured in Lua.
//!
//! Binary entry point. Current scope: open a libseat session, enumerate
//! input devices through udev + libinput (with libinput accel / profile
//! from [`config::Config`]), set up a GBM + EGL + GLES render loop on
//! every connected output, paint each vblank with the configured
//! wallpaper plus a mouse-following cursor sprite, route key events
//! through xkbcommon, host a minimal Wayland frontend that composites
//! every live `xdg_toplevel` between wallpaper and cursor, and forward
//! pointer + keyboard events to the focused client. Window placement /
//! focus model are still the 4d milestone — surfaces stack at the
//! virtual origin and the most-recently-mapped toplevel takes focus.
//!
//! Run on a free virtual terminal (e.g. Ctrl+Alt+F2), `cargo run`, then
//! type and move the pointer. Press `Super+Shift+E` to exit. Once DRM
//! takes the mode the kernel TTY console can't repaint that VT — that's
//! expected, not a freeze — and your shell prompt reappears when we
//! exit and the seat is handed back to logind.
//! Configure log output with `RUST_LOG`; the default is
//! `info,libreland=debug` so our own messages show up while third-party
//! crates stay quiet. The same records are also written to
//! `$XDG_STATE_HOME/libreland/<TIMESTAMP>.log` (default
//! `~/.local/state/libreland/`); read that file when stderr isn't visible
//! (e.g. after a freeze that needs recovering from another TTY).

use anyhow::{Context as _, Result};
use smithay::backend::drm::{DrmDevice, DrmEventMetadata, DrmEventTime};
use smithay::backend::input::{
    Axis, AxisSource, Event as _, InputBackend, InputEvent, KeyState, KeyboardKeyEvent as _,
    PointerAxisEvent as _, PointerButtonEvent as _, PointerMotionEvent as _,
};
use smithay::backend::libinput::{LibinputInputBackend, LibinputSessionInterface};
use smithay::backend::session::Session as _;
use smithay::backend::session::libseat::{LibSeatSession, LibSeatSessionNotifier};
use smithay::backend::udev::{UdevBackend, UdevEvent};
use smithay::input::keyboard::FilterResult;
use smithay::input::pointer::{
    AxisFrame, ButtonEvent, CursorIcon, CursorImageStatus, MotionEvent, RelativeMotionEvent,
};
use smithay::input::{Seat, SeatState};
use smithay::reexports::calloop::generic::Generic;
use smithay::reexports::calloop::{EventLoop, Interest, LoopSignal, Mode, PostAction};
use smithay::reexports::drm::control::crtc;
use smithay::reexports::input as libinput;
use smithay::reexports::input::Libinput;
use smithay::reexports::input::event::keyboard::KeyboardKeyEvent as LibinputKeyEvent;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{Display, DisplayHandle, Resource as _};
use smithay::reexports::wayland_protocols::wp::presentation_time::server::wp_presentation_feedback::Kind as PresentKind;
use smithay::utils::{Clock, IsAlive, Logical, Monotonic, Physical, Point, SERIAL_COUNTER};
use smithay::wayland::compositor::CompositorState;
use smithay::wayland::output::OutputManagerState;
use smithay::wayland::pointer_constraints::{PointerConstraint, with_pointer_constraint};
use smithay::wayland::shell::xdg::XdgShellState;
use smithay::wayland::shm::ShmState;
use smithay::wayland::socket::ListeningSocketSource;
use std::fs::File;
use std::io;
use tracing::{debug, info, warn};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

mod anim;
mod clipboard;
mod color_management;
mod config;
mod cursor;
mod drm;
mod hdr;
mod ipc;
mod keyboard;
mod media;
mod layout;
mod render;
mod scanout;
mod screencopy;
mod screenshot;
mod wayland;
mod xwayland;

/// Walk subsurface parents up to the root surface, so a subsurface commit
/// resolves to the toplevel (or layer surface) it belongs to.
pub(crate) fn root_surface(surface: &WlSurface) -> WlSurface {
    let mut root = surface.clone();
    while let Some(parent) = smithay::wayland::compositor::get_parent(&root) {
        root = parent;
    }
    root
}

/// Per-output render-scheduling state for on-demand rendering. The
/// compositor renders an output only when [`State::queue_redraw`] marks it,
/// rather than flipping every vblank — so an idle output draws nothing, and
/// a fullscreen client's output flips at the client's pace (which is what
/// makes Variable Refresh Rate actually take effect).
#[derive(Debug, Clone, Copy)]
enum RedrawState {
    /// No flip in flight and nothing scheduled — waiting for a trigger.
    Idle,
    /// A render is queued to run on the next event-loop turn (an idle
    /// callback is live); further triggers coalesce into it.
    Scheduled,
    /// A flip is in flight; `dirty` records whether a trigger arrived while
    /// we waited, so the vblank re-renders instead of parking. We must
    /// never queue a second flip before the first completes.
    WaitingForVblank { dirty: bool },
}

/// Mutable state threaded through every event-loop callback.
///
/// Holds the existing libseat / DRM / renderer / xkb / config state
/// plus the Wayland frontend substate added in milestone 4a
/// (compositor, shm, seat, `xdg_shell`, `output_manager`). `State` is
/// the calloop loop-data type itself; the owned `Display<State>` can't
/// live here (the type would be circular), so it's moved into the
/// Wayland dispatch source's closure instead.
///
/// All fields are `pub(crate)` so handler impls in sibling modules
/// (especially [`crate::wayland`]) can reach them directly without
/// every field needing its own accessor method.
#[allow(
    clippy::struct_field_names,
    reason = "the *_state suffix is smithay's convention and matches each field's type name; renaming would just diverge from upstream docs"
)]
pub(crate) struct State {
    /// The libseat session is retained so future code can query its
    /// active flag and switch VTs. libinput already holds an internal
    /// clone for opening `/dev/input/*` device nodes.
    #[allow(
        dead_code,
        reason = "session is held for upcoming VT-switch and activation tracking; not read yet"
    )]
    pub(crate) session: LibSeatSession,
    /// Used by the exit hotkey to break calloop's `run` cleanly.
    pub(crate) loop_signal: LoopSignal,
    /// DRM master claim. Held by the State so the master claim
    /// outlives the renderer's swapchain — dropping it releases the
    /// display back to logind on clean shutdown.
    #[allow(
        dead_code,
        reason = "kept alive for the DRM master claim; will be queried by VT-switch / session-activation code"
    )]
    pub(crate) drm_device: DrmDevice,
    /// GBM + EGL + GLES render pipeline. The vblank callback drives
    /// it once per refresh.
    pub(crate) renderer: render::Renderer,
    /// xkbcommon keymap + state. Every libinput key event flows
    /// through this to get a layout-aware keysym + modifier mask,
    /// which the hotkey logic matches on.
    pub(crate) keyboard: keyboard::Keyboard,
    /// All runtime settings (monitors, input, binds, wallpaper, …)
    /// in one place. Defaults today; the Lua loader in milestone
    /// 3c will replace this from `$XDG_CONFIG_HOME/libreland/config.lua`.
    pub(crate) config: config::Config,
    /// Pointer/keyboard libinput devices seen so far (added, not yet
    /// removed). libinput exposes no device enumeration, so we keep the
    /// handles ourselves — that lets a config reload re-apply mouse
    /// acceleration to the *live* devices. Updated on
    /// `DeviceAdded`/`DeviceRemoved`.
    pub(crate) input_devices: Vec<smithay::reexports::input::Device>,
    /// Children spawned at runtime (startup commands, bind/IPC `spawn`,
    /// the idle lock command). Held so [`State::reap_children`] can
    /// `try_wait` them — a dropped `Child` handle is never waited on, so
    /// every exited child would otherwise linger as a zombie.
    pub(crate) children: Vec<std::process::Child>,
    /// X display our Xwayland serves (e.g. `":1"`). Exported as
    /// `$DISPLAY` to children we spawn (see [`State::build_command`]) so
    /// X clients connect; `None` when X support is off.
    pub(crate) xwayland_display: Option<String>,
    /// The X11 window manager connection to our Xwayland, once it
    /// reported ready. All XWM callbacks (`XwmHandler`) route through
    /// this; `None` while Xwayland is off/starting/dead.
    pub(crate) xwm: Option<smithay::xwayland::X11Wm>,
    /// Registration of the Xwayland readiness source. Removing it drops
    /// the [`smithay::xwayland::XWayland`] instance, which disconnects
    /// and terminates the Xwayland server — that's how a live
    /// `xwayland = false` reload stops X support.
    pub(crate) xwayland_source: Option<smithay::reexports::calloop::RegistrationToken>,
    /// Xwayland's Wayland client handle: carries the per-client scale
    /// override that maps X pixel space to our logical space (see
    /// `xwayland.rs` module docs). Held to re-apply on scale changes.
    pub(crate) xwayland_client: Option<smithay::reexports::wayland_server::Client>,
    /// Managed (layout-resident) X11 windows, paired with the
    /// `wl_surface` Xwayland associated. The pair is the lookup table in
    /// both directions: focus sync (wl → X11) and unmap cleanup
    /// (X11 → wl, whose association Xwayland may already have dropped).
    pub(crate) x11_windows: Vec<(smithay::xwayland::X11Surface, WlSurface)>,
    /// Mapped override-redirect X11 windows (menus, tooltips). Never in
    /// the layout — rendered topmost through the popup path at whatever
    /// global position the client set.
    pub(crate) x11_or_windows: Vec<(smithay::xwayland::X11Surface, WlSurface)>,
    /// The X11 window currently holding X input focus, if keyboard
    /// focus is on an X11 window. Tracked so focus moves can unfocus
    /// the previous X window (see `sync_x11_focus`).
    pub(crate) x11_kbd_focus: Option<smithay::xwayland::X11Surface>,
    /// Which selections (clipboard / primary) are currently owned by an
    /// X11 client. Routes Wayland-side paste requests back through the
    /// XWM instead of the compositor's clipboard cache.
    pub(crate) x11_owns_selection: crate::xwayland::X11SelectionOwnership,
    /// `xwayland_shell_v1` global state: how Xwayland associates its
    /// `wl_surface`s with X11 windows. Held to keep the global alive.
    pub(crate) xwayland_shell_state: smithay::wayland::xwayland_shell::XWaylandShellState,
    /// Cheap-to-clone handle to the Wayland display. Used by handler
    /// impls that need to create new globals or look up clients.
    #[allow(
        dead_code,
        reason = "held for future use by handlers (creating outputs, surfaces); not read directly yet"
    )]
    pub(crate) display_handle: DisplayHandle,
    /// `wl_compositor` + `wl_subcompositor` substate.
    pub(crate) compositor_state: CompositorState,
    /// `wl_shm` substate.
    pub(crate) shm_state: ShmState,
    /// `wl_seat` substate; tracks all seats on the compositor.
    pub(crate) seat_state: SeatState<State>,
    /// The single seat we currently advertise. The input-forwarding
    /// paths in [`State::handle_key`], [`State::forward_pointer_motion`]
    /// and [`State::forward_pointer_button`] reach `KeyboardHandle` /
    /// `PointerHandle` through this field on every event.
    pub(crate) seat: Seat<State>,
    /// `xdg_wm_base` + `xdg_surface` + `xdg_toplevel` substate.
    pub(crate) xdg_shell_state: XdgShellState,
    /// `zxdg_decoration_manager_v1` substate. Held so the
    /// `delegate_xdg_decoration!` macro routes per-toplevel
    /// decoration objects through us; our handler pins every
    /// client to `ServerSide` mode and we then draw no
    /// decorations at all (it's a tiler).
    #[allow(
        dead_code,
        reason = "held so delegate_xdg_decoration! can route global dispatch through it; the global is the only externally-visible effect"
    )]
    pub(crate) xdg_decoration_state: smithay::wayland::shell::xdg::decoration::XdgDecorationState,
    /// KDE `org_kde_kwin_server_decoration` substate. Held so the
    /// global stays registered and `KdeDecorationHandler` can borrow
    /// it; advertising it with a Server default mode is what stops
    /// GTK/Firefox from drawing a client-side titlebar.
    pub(crate) kde_decoration_state: smithay::wayland::shell::kde::decoration::KdeDecorationState,
    /// `wl_output` substate. The `OutputManagerState` carries the
    /// `xdg_output_manager_v1` global; per-output `wl_output`
    /// globals live on the individual `Output`s in `outputs`.
    #[allow(
        dead_code,
        reason = "held so delegate_output! can route global dispatch through it; the outputs vec is the per-display source of truth"
    )]
    pub(crate) output_manager_state: OutputManagerState,
    /// One `smithay::output::Output` per DRM connector. Held so
    /// each global's lifetime is the compositor's; the renderer
    /// owns the framebuffer side, this owns the protocol side.
    #[allow(
        dead_code,
        reason = "held so each wl_output global stays alive; reads happen via the Output objects themselves on focus/resize"
    )]
    pub(crate) outputs: Vec<smithay::output::Output>,
    /// `wl_output` global id per output (by connector name), so the
    /// hotplug path can remove a vanished output's global and register a
    /// newly-connected one. Kept in lock-step with `outputs`.
    pub(crate) output_globals:
        std::collections::HashMap<String, smithay::reexports::wayland_server::backend::GlobalId>,
    /// `wp_fractional_scale_manager_v1` substate.
    #[allow(
        dead_code,
        reason = "held so delegate_fractional_scale! routes through it; new_fractional_scale callbacks read preferred_scale"
    )]
    pub(crate) fractional_scale_state:
        smithay::wayland::fractional_scale::FractionalScaleManagerState,
    /// `wp_viewporter` global. Held so the global stays registered
    /// (dropping it removes it) — `delegate_viewporter!` routes the
    /// `wp_viewport` requests, and smithay's surface state applies
    /// the viewport when the renderer composites each surface.
    /// Required for fractional scaling to size client buffers right.
    #[allow(
        dead_code,
        reason = "held so the wp_viewporter global stays alive and delegate_viewporter! can route through State; smithay reads the per-surface viewport during compositing"
    )]
    pub(crate) viewporter_state: smithay::wayland::viewporter::ViewporterState,
    /// `wl_data_device_manager` global — clipboard + drag-and-drop.
    /// Held so the global stays registered and `delegate_data_device!`
    /// can route through it; `DataDeviceHandler::data_device_state`
    /// borrows it.
    pub(crate) data_device_state: smithay::wayland::selection::data_device::DataDeviceState,
    /// `wp_cursor_shape_v1` global. Held so the global stays registered
    /// for the process lifetime and `delegate_cursor_shape!` can route
    /// through it; the cursor-shape requests themselves arrive via
    /// `SeatHandler::cursor_image`, so the instance is never read.
    #[allow(
        dead_code,
        reason = "owns the wp_cursor_shape_v1 global registration; dispatch routes through State, not this handle"
    )]
    pub(crate) cursor_shape_state: smithay::wayland::cursor_shape::CursorShapeManagerState,
    /// `zwp_linux_dmabuf_v1` substate + global. Held so the global
    /// stays registered and `delegate_dmabuf!` / `DmabufHandler` can
    /// route through it; the handler imports offered GPU buffers into
    /// the renderer so GPU-composited (incl. Xwayland) apps display.
    pub(crate) dmabuf_state: smithay::wayland::dmabuf::DmabufState,
    #[allow(
        dead_code,
        reason = "held to keep the zwp_linux_dmabuf_v1 global alive for the compositor's lifetime"
    )]
    pub(crate) dmabuf_global: smithay::wayland::dmabuf::DmabufGlobal,
    /// The v4 default dmabuf feedback, re-sent per-surface when a window
    /// leaves fullscreen; `None` on a v3-only global.
    pub(crate) dmabuf_default_feedback: Option<smithay::wayland::dmabuf::DmabufFeedback>,
    /// Default feedback + a Scanout-flagged tranche of the primary
    /// plane's explicit modifiers, sent per-surface to fullscreen windows
    /// so their swapchains re-allocate into plane-scannable buffers (the
    /// missing piece between the single-pass composite and true
    /// zero-copy direct scanout). See [`State::sync_scanout_feedback`].
    pub(crate) dmabuf_scanout_feedback: Option<smithay::wayland::dmabuf::DmabufFeedback>,
    /// Last known lock-key LED state (num/caps/scroll) from xkb, pushed
    /// to every keyboard by [`State::apply_keyboard_leds`]. Kept so a
    /// hot-plugged keyboard (which arrives LEDs-off) can be synced to the
    /// session's actual lock state.
    pub(crate) keyboard_leds: smithay::input::keyboard::LedState,
    /// Bumped on every redraw trigger; versions the pointer hit-test
    /// snapshot below (a stale epoch means the scene may have changed).
    pub(crate) scene_epoch: std::cell::Cell<u64>,
    /// Cached popup-placement snapshot for pointer hit-testing, keyed by
    /// the epoch it was built at. Every mouse motion used to rebuild the
    /// full placement + popup snapshot (layout walk + allocations); at
    /// 1000 Hz polling that's a thousand rebuilds a second for a scene
    /// that changes at most once per frame.
    pub(crate) popup_snapshot: std::cell::RefCell<(u64, Vec<render::PopupPlacement>)>,
    /// Surfaces that have ever been handed the scanout feedback variant —
    /// they keep it for life (see `sync_scanout_feedback`; every feedback
    /// change costs the client a swapchain rebuild). `RefCell` because the
    /// per-frame sync runs under `&self` alongside the renderer borrows.
    pub(crate) scanout_feedback_given:
        std::cell::RefCell<std::collections::HashSet<smithay::reexports::wayland_server::backend::ObjectId>>,
    /// Fractional scale to send to every new
    /// `wp_fractional_scale` object. Currently the primary
    /// output's configured scale; will become per-surface once
    /// per-output workspaces ship.
    pub(crate) preferred_scale: f64,
    /// `wlr_layer_shell` substate. The handler in
    /// [`crate::wayland`] reads / writes it for new + destroyed
    /// layer surfaces; renderer reads it each frame.
    pub(crate) layer_shell_state: smithay::wayland::shell::wlr_layer::WlrLayerShellState,
    /// The output each `wlr_layer_shell` surface asked to live on, by
    /// connector name. A layer surface (panel, bar, OSD, slurp's
    /// per-output overlay) is created bound to one `wl_output`; without
    /// tracking it we'd place every layer on the primary, so slurp's
    /// second-monitor overlay would stack on the first. Keyed by the
    /// layer surface's `wl_surface`; absent ⇒ the client let the
    /// compositor choose ⇒ fall back to primary.
    pub(crate) layer_outputs: std::collections::HashMap<WlSurface, String>,
    /// wlr-layer-shell namespace each layer surface set at creation, kept for
    /// per-layer blur rules and the `layers` IPC query.
    pub(crate) layer_namespaces: std::collections::HashMap<WlSurface, String>,
    /// Toplevels that have already been re-configured once after mapping their
    /// first buffer. Some clients (MPV's idle window) ignore the size in the
    /// initial configure and only resize on a later one, so we nudge each
    /// exactly once on first map — this set stops it firing every frame.
    pub(crate) mapped_toplevels: std::collections::HashSet<WlSurface>,
    /// Built-in idle handling (`config.idle`). `idle_last_input` is bumped on
    /// every input event; the idle timer compares its age to the configured
    /// thresholds. `idle_screen_off` tracks whether the panels are DPMS-off so
    /// input can wake them; `idle_lock_spawned` stops the lock command being
    /// re-spawned every tick while idle (reset when the session unlocks).
    pub(crate) idle_last_input: std::time::Instant,
    pub(crate) idle_screen_off: bool,
    pub(crate) idle_lock_spawned: bool,
    /// `ext_idle_notifier_v1` — lets idle daemons (swayidle, etc.) learn
    /// when the user goes idle. Fed `notify_activity` on every input and
    /// told `set_is_inhibited` while an idle inhibitor is up. It schedules
    /// its own calloop timers, which is why the loop data type is `State`.
    pub(crate) idle_notifier: smithay::wayland::idle_notify::IdleNotifierState<State>,
    /// `zwp_idle_inhibit_manager_v1` global. Held so it stays registered;
    /// dispatch routes through `State`.
    #[allow(dead_code, reason = "held to keep the idle-inhibit global alive")]
    pub(crate) idle_inhibit_state: smithay::wayland::idle_inhibit::IdleInhibitManagerState,
    /// `xdg_activation_v1` global — read by the `XdgActivationHandler`
    /// impl to focus a surface a client asks to raise.
    pub(crate) xdg_activation_state: smithay::wayland::xdg_activation::XdgActivationState,
    /// `zwp_pointer_gestures_v1` global. Held so it stays registered;
    /// dispatch routes through `State`.
    #[allow(dead_code, reason = "held to keep the pointer-gestures global alive")]
    pub(crate) pointer_gestures_state: smithay::wayland::pointer_gestures::PointerGesturesState,
    /// `wp_color_management_v1` global + image-description identity
    /// registry. Held so the global stays registered; dispatch routes
    /// through `State`.
    pub(crate) color_management: crate::color_management::ColorManagementState,
    /// `wp_content_type_v1` global. Held so the global stays registered;
    /// clients tag a surface's content type (game / video / photo).
    #[allow(dead_code, reason = "held to keep the wp_content_type_v1 global alive")]
    pub(crate) content_type_state: smithay::wayland::content_type::ContentTypeState,
    /// `wp_presentation` global. Held so it stays registered; per-frame
    /// feedback flows through per-surface cached state, not this handle.
    #[allow(dead_code, reason = "held to keep the wp_presentation global alive")]
    pub(crate) presentation_state: smithay::wayland::presentation::PresentationState,
    /// `linux-drm-syncobj-v1` (explicit sync) global. `None` when the DRM
    /// device lacks `syncobj_eventfd` support (then the protocol isn't
    /// advertised and clients fall back to implicit sync). Returned to smithay
    /// via the `DrmSyncobjHandler` impl.
    pub(crate) drm_syncobj_state: Option<smithay::wayland::drm_syncobj::DrmSyncobjState>,
    /// Per-surface colour state set via the colour-management protocol,
    /// keyed by `wl_surface` id. Read by the renderer to colour-manage an
    /// HDR output; pruned on surface/object destroy.
    pub(crate) color_surfaces: std::collections::HashMap<
        smithay::reexports::wayland_server::backend::ObjectId,
        crate::color_management::SurfaceColor,
    >,
    /// `wl_surface`s that already own a `wp_color_management_surface_v1`.
    pub(crate) color_surface_objects:
        std::collections::HashSet<smithay::reexports::wayland_server::backend::ObjectId>,
    /// Deferred `wp_image_description_info_v1` responses: drained after
    /// each dispatch (see [`color_management::flush_pending_image_info`])
    /// because sending the `done` destructor inside the creating request
    /// panics the wayland backend.
    pub(crate) pending_image_info: Vec<color_management::PendingImageInfo>,
    /// Surfaces holding an active idle inhibitor (e.g. a playing video).
    /// While any is alive, `idle_tick` suppresses the built-in lock/DPMS.
    /// Populated by the `IdleInhibitHandler`; pruned of dead surfaces each
    /// tick (a crashed client never sends the clean destroy).
    pub(crate) idle_inhibitors: std::collections::HashSet<WlSurface>,
    /// `ext-session-lock-v1` manager global. Held so the global stays
    /// registered; dispatch routes through `State`.
    pub(crate) session_lock_state: smithay::wayland::session_lock::SessionLockManagerState,
    /// While the session is locked, the lock surface for each output (by
    /// connector name). The renderer draws only these (full-screen, above
    /// everything) and the pointer/keyboard are routed solely to them.
    pub(crate) lock_surfaces:
        std::collections::HashMap<String, smithay::wayland::session_lock::LockSurface>,
    /// Whether a client has locked the session (the `SessionLocker` confirmed).
    pub(crate) session_locked: bool,
    /// `zwp_relative_pointer_manager_v1` global — held alive so clients
    /// keep receiving relative motion (mouse-look). Dispatched via the
    /// delegate; not otherwise read.
    #[allow(dead_code, reason = "held to keep the global alive")]
    pub(crate) relative_pointer_state:
        smithay::wayland::relative_pointer::RelativePointerManagerState,
    /// `zwp_pointer_constraints_v1` global — held alive so clients can
    /// lock/confine the pointer. Dispatched via the delegate; the
    /// active constraint is read per-motion via `with_pointer_constraint`.
    #[allow(dead_code, reason = "held to keep the global alive")]
    pub(crate) pointer_constraints_state:
        smithay::wayland::pointer_constraints::PointerConstraintsState,
    /// `zwp_primary_selection_v1` global — read by the
    /// `PrimarySelectionHandler` impl; held so the global stays alive.
    pub(crate) primary_selection_state:
        smithay::wayland::selection::primary_selection::PrimarySelectionState,
    /// `zwlr_data_control_manager_v1` global — privileged selection
    /// access for clipboard managers (cliphist, clipman, `wl-paste
    /// --watch`). Read by the wlr `DataControlHandler` impl; held so
    /// the global stays alive.
    pub(crate) wlr_data_control_state:
        smithay::wayland::selection::wlr_data_control::DataControlState,
    /// `ext_data_control_manager_v1` global — the standardized
    /// successor to `wlr_data_control`, same role. Read by the ext
    /// `DataControlHandler` impl; held so the global stays alive.
    pub(crate) ext_data_control_state:
        smithay::wayland::selection::ext_data_control::DataControlState,
    /// Compositor-side clipboard + primary-selection caches, so a
    /// copied buffer survives the source client closing. See
    /// [`crate::clipboard`].
    pub(crate) clipboard: clipboard::Selections,
    /// `zwlr_screencopy_manager_v1` global — held alive so screenshot
    /// tools and `xdg-desktop-portal-wlr` can capture outputs.
    #[allow(dead_code, reason = "held to keep the global alive")]
    pub(crate) screencopy_manager: screencopy::ScreencopyManagerState,
    /// `zwlr_screencopy` `copy` requests awaiting the next render of
    /// their output (see [`crate::screencopy`]).
    pub(crate) screencopy_pending: Vec<screencopy::PendingCapture>,
    /// Calloop handle, used to register the async pipe reads/writes
    /// that drain and serve cached selections without blocking.
    pub(crate) loop_handle: smithay::reexports::calloop::LoopHandle<'static, State>,
    /// Per-output on-demand render state, keyed by CRTC. See
    /// [`RedrawState`] and [`State::queue_redraw`].
    pub(crate) redraw: std::collections::HashMap<crtc::Handle, RedrawState>,
    /// Keyboard focus saved when a layer-shell surface grabs
    /// exclusive focus (e.g. rofi), restored when that surface
    /// is destroyed.
    pub(crate) kbd_focus_before_layer:
        Option<smithay::reexports::wayland_server::protocol::wl_surface::WlSurface>,
    /// Dwindle tiling layout — owns every visible window's
    /// `(wl_surface, rect)` pair. The vblank handler snapshots
    /// this each frame; xdg handlers in [`crate::wayland`] insert
    /// + remove entries on toplevel map / destroy.
    pub(crate) layout: layout::Layout,
    /// Tracks `xdg_popup` parent→child trees (smithay owns the
    /// per-surface `PopupTree`; this manager holds the bookkeeping +
    /// the `cleanup` entry point). The vblank handler reads it each
    /// frame via `PopupManager::popups_for_surface` to render menus /
    /// submenus on top of their parent window.
    pub(crate) popup_manager: smithay::desktop::PopupManager,
    /// Root toplevel/layer surface that owns the currently-grabbing
    /// popup chain (set from `XdgShellHandler::grab`), or `None` when no
    /// menu has an active grab. While set, the hover focus model is
    /// frozen on this root so moving the pointer off the parent can't
    /// steal keyboard focus and make the client dismiss its own menu.
    /// Expired by `refresh_popup_grab` once the chain closes.
    pub(crate) popup_grab: Option<WlSurface>,
    /// Active interactive drag (Super + LMB to move, Super + RMB
    /// to resize). `Some` only between the initiating press and
    /// the matching release; during that window, pointer motion
    /// events update the dragged surface's rect instead of
    /// reaching its client, and intervening button events are
    /// swallowed.
    pub(crate) drag: Option<DragState>,
    /// Accumulated high-resolution scroll (v120 units) for the
    /// `Super`+scroll workspace gesture. One physical wheel notch is
    /// 120 units; we fire one workspace step per ±120 accumulated so
    /// a hi-res / free-spinning wheel that emits sub-notch events
    /// doesn't switch several workspaces at once.
    pub(crate) ws_scroll_accum: f64,
    /// Active built-in screenshot session (region drag / window pick).
    /// `Some` only between the trigger keypress and the capture/cancel;
    /// while set, pointer + keyboard input drives the selection instead
    /// of reaching clients. `None` disables the screenshot UI.
    pub(crate) screenshot: Option<ScreenshotState>,
    /// Compositor-originated capture requests (full-output freeze
    /// snapshots and final region grabs) awaiting the next render of
    /// their output — the internal sibling of [`Self::screencopy_pending`].
    pub(crate) screenshot_pending: Vec<InternalCapture>,
    /// UTC offset captured once at startup (on the main thread, before
    /// the process goes multithreaded) so screenshot filenames can use
    /// local time reliably.
    pub(crate) local_offset: time::UtcOffset,
    /// PNG encode + disk write for a finished screenshot run on a worker
    /// thread (they'd otherwise freeze the compositor for the duration of a
    /// large capture's zlib compression). The worker sends the encoded PNG
    /// back over this channel when the bind also wants the clipboard, since
    /// setting the selection must happen on the main thread.
    pub(crate) screenshot_clipboard_tx:
        smithay::reexports::calloop::channel::Sender<Vec<u8>>,
    /// Control-IPC state: the stable window-id registry (and, later,
    /// event subscribers). The socket lives on the event loop; this is
    /// the bookkeeping its dispatch reads + writes.
    pub(crate) ipc: ipc::IpcState,
}

/// An in-progress screenshot. `bind` carries the configured behaviour
/// (mode, freeze, save dir, clipboard); the rest is selection progress.
pub(crate) struct ScreenshotState {
    pub(crate) bind: std::sync::Arc<config::ScreenshotBind>,
    /// Region drag start corner (absolute compositor px), set on press.
    pub(crate) anchor: Option<(f64, f64)>,
    /// Frozen full-output snapshots (freeze mode), by output name, used
    /// both as the displayed backdrop and as the source the final image
    /// is cropped from.
    pub(crate) frozen: std::collections::HashMap<String, FrozenFrame>,
}

/// A frozen full-output capture kept for the duration of a freeze-mode
/// session — the backdrop shown during selection and the source the final
/// crop is taken from. Bytes are the raw `CaptureOutcome::Shm` read-back
/// (memory order B,G,R,X, natural top-down row order).
pub(crate) struct FrozenFrame {
    pub(crate) bytes: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// A compositor-originated capture serviced in the vblank handler.
pub(crate) struct InternalCapture {
    pub(crate) output: String,
    pub(crate) region: smithay::utils::Rectangle<i32, Physical>,
    /// Bake the cursor into this capture's frame (per the bind's
    /// `show_cursor`). Drives `hide_cursor` for the rendering vblank.
    pub(crate) show_cursor: bool,
    pub(crate) purpose: CapturePurpose,
}

/// What to do with an [`InternalCapture`]'s pixels once they come back.
pub(crate) enum CapturePurpose {
    /// Store + display as the frozen backdrop for its output.
    Freeze,
    /// Encode + save/clipboard immediately (full-output or live region).
    Finalize {
        bind: std::sync::Arc<config::ScreenshotBind>,
    },
}

/// In-progress interactive drag. The dragged surface is always
/// floating (we promote it on drag start if it was tiled).
#[derive(Debug, Clone)]
pub(crate) struct DragState {
    pub(crate) surface: WlSurface,
    pub(crate) mode: DragMode,
    /// Cursor position (compositor coords, `f64`) at the moment
    /// the drag began.
    pub(crate) cursor_start: (f64, f64),
    /// Window rect at the moment the drag began. Motion deltas
    /// transform this into the current rect.
    pub(crate) rect_start: smithay::utils::Rectangle<i32, Physical>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DragMode {
    /// Translate the rect by the cursor delta; size unchanged.
    Move,
    /// Stretch the rect's size by the cursor delta from the
    /// initial bottom-right corner; clamped to a sane minimum so
    /// the user can never resize a window into invisibility.
    Resize,
}

/// evdev button codes for the two buttons we react to. See
/// `linux/input-event-codes.h`. Anything else falls through to
/// the focused client as a normal pointer button event.
const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;

/// Session-identity environment defaults set at startup (overridable
/// by the user's `env` config). Also exported to the D-Bus / systemd
/// activation environment so D-Bus-activated services — notably
/// `xdg-desktop-portal` — see them.
const DEFAULT_SESSION_ENV: &[(&str, &str)] = &[
    ("XDG_CURRENT_DESKTOP", "libreland"),
    ("XDG_SESSION_TYPE", "wayland"),
    ("XDG_SESSION_DESKTOP", "libreland"),
];

/// Lower bound a drag-resize will clamp the floating rect to so
/// the window can't be resized into a slice too small to grab
/// again. Conservative — most useful clients render correctly
/// well above this.
const MIN_DRAG_RESIZE_W: i32 = 100;
const MIN_DRAG_RESIZE_H: i32 = 60;

impl State {
    /// Feed a key event through xkbcommon to get its layout-aware
    /// keysym + effective modifier mask, log the result at debug,
    /// and decide whether it fires a compositor-level binding or
    /// gets forwarded to the focused Wayland client.
    ///
    /// Smithay's `KeyboardHandle::input` is always called so its
    /// internal modifier tracking stays in sync (even on
    /// intercepted presses) and so `wl_keyboard.modifiers` events
    /// reach the client correctly. The filter returns
    /// `Intercept(action)` for hotkey hits (which we then
    /// dispatch and *don't* forward), `Forward` otherwise.
    /// Releases never match bindings but still need to flow through
    /// for modifier release + `wl_keyboard.key` forwarding.
    fn handle_key(&mut self, event: &LibinputKeyEvent) {
        let pressed = matches!(event.state(), KeyState::Pressed);
        let result = self.keyboard.process(event.key_code(), pressed);

        debug!(
            keysym = ?result.keysym,
            mods = format!("{:#06b}", result.mods),
            pressed,
            "key processed through xkb"
        );

        // While a screenshot session owns the screen, the keyboard drives
        // it: Esc cancels, Enter confirms a region drag. Every key is
        // swallowed (intercepted, never forwarded to a client). We still
        // pump kbd.input so smithay's modifier bookkeeping stays in sync.
        if self.screenshot.is_some() {
            if pressed {
                use xkbcommon::xkb::keysyms;
                match result.keysym.raw() {
                    keysyms::KEY_Escape => self.cancel_screenshot(),
                    keysyms::KEY_Return | keysyms::KEY_KP_Enter => {
                        self.confirm_screenshot_region();
                    }
                    _ => {}
                }
            }
            let serial = SERIAL_COUNTER.next_serial();
            if let Some(kbd) = self.seat.get_keyboard() {
                kbd.input::<(), _>(
                    self,
                    event.key_code(),
                    event.state(),
                    serial,
                    event.time_msec(),
                    |_, _, _| FilterResult::Intercept(()),
                );
            }
            return;
        }

        // While the session is locked (ext-session-lock-v1), NO user bind
        // may fire — otherwise `exit`, a `spawn`, or a screenshot bind
        // would be a trivial lock bypass. Every key forwards to the lock
        // surface (which holds keyboard focus) so the password can be
        // typed; the compositor enforces this regardless of the user's
        // configured binds.
        let matched_action = if pressed && !self.session_locked {
            let normal = self
                .config
                .binds
                .bindings
                .iter()
                .find(|b| {
                    keyboard::fold_keysym(result.keysym) == keyboard::fold_keysym(b.keysym)
                        && result.has_all_mods(b.mods)
                })
                .map(|b| b.action.clone());
            // Screenshot binds (if configured) are matched the same way;
            // normal binds win a tie.
            normal.or_else(|| {
                self.config.screenshot.as_ref().and_then(|binds| {
                    binds
                        .iter()
                        .find(|b| {
                            keyboard::fold_keysym(result.keysym) == keyboard::fold_keysym(b.keysym)
                                && result.has_all_mods(b.mods)
                        })
                        .map(|b| config::Action::Screenshot(std::sync::Arc::new(b.clone())))
                })
            })
        } else {
            None
        };

        let key_code = event.key_code();
        let key_state = event.state();
        let time = event.time_msec();
        let serial = SERIAL_COUNTER.next_serial();
        let Some(kbd) = self.seat.get_keyboard() else {
            return;
        };
        let action = kbd.input::<config::Action, _>(
            self,
            key_code,
            key_state,
            serial,
            time,
            |_data, _mods, _keysym| {
                matched_action.map_or(FilterResult::Forward, FilterResult::Intercept)
            },
        );
        if let Some(action) = action {
            self.dispatch_action(action);
        }
    }

    /// Whether the focused surface currently holds an *active*
    /// pointer-lock constraint. The renderer hides our cursor while
    /// this is true (the locked client — a game — draws its own).
    fn pointer_locked(&self) -> bool {
        let Some(pointer) = self.seat.get_pointer() else {
            return false;
        };
        let Some(surface) = pointer.current_focus() else {
            return false;
        };
        with_pointer_constraint(&surface, &pointer, |constraint| {
            constraint.is_some_and(|c| c.is_active() && matches!(*c, PointerConstraint::Locked(_)))
        })
    }

    /// Request that `crtc`'s output render a fresh frame. This is the
    /// on-demand core: instead of flipping every vblank, each output sits
    /// idle until a trigger (client commit, input, layout change,
    /// animation, …) calls this. Coalescing is automatic — many triggers
    /// between two frames schedule at most one render.
    fn queue_redraw(&mut self, crtc: crtc::Handle) {
        // Any redraw trigger may mean the scene changed — invalidate the
        // pointer hit-test snapshot (see popup_at).
        self.scene_epoch.set(self.scene_epoch.get().wrapping_add(1));
        let schedule = match self.redraw.entry(crtc).or_insert(RedrawState::Idle) {
            slot @ RedrawState::Idle => {
                *slot = RedrawState::Scheduled;
                true
            }
            // Already going to render this turn.
            RedrawState::Scheduled => false,
            // A flip is in flight; remember to render again on its vblank.
            RedrawState::WaitingForVblank { dirty } => {
                *dirty = true;
                false
            }
        };
        if schedule {
            self.loop_handle
                .insert_idle(move |state| state.render_crtc(crtc));
        }
    }

    /// Queue a redraw on every output. Used for changes we can't (or needn't)
    /// pin to one output — layout reflows, focus changes, config reloads.
    fn queue_redraw_all(&mut self) {
        for crtc in self.renderer.crtcs() {
            self.queue_redraw(crtc);
        }
    }

    /// Power every output's panel on (`true`) or off (`false`) via the
    /// connector's DPMS property. Off blanks the panel while keeping the
    /// session and clients intact; on re-enables it and forces a redraw so
    /// fresh pixels scan out at once. Best-effort: a driver that rejects the
    /// legacy DPMS property just logs and stays as-is.
    pub(crate) fn set_screen_power(&mut self, on: bool) {
        use smithay::reexports::drm::control::Device as _;
        // DPMS enum: 0 = On, 1 = Standby, 2 = Suspend, 3 = Off.
        let value: u64 = if on { 0 } else { 3 };
        for conn in self.renderer.output_connectors() {
            let Ok(props) = self.drm_device.get_properties(conn) else {
                continue;
            };
            let (handles, _values) = props.as_props_and_values();
            for &handle in handles {
                let Ok(info) = self.drm_device.get_property(handle) else {
                    continue;
                };
                if info.name().to_bytes() == b"DPMS" {
                    if let Err(err) = self.drm_device.set_property(conn, handle, value) {
                        warn!(error = %err, ?conn, on, "failed to set DPMS property");
                    }
                    break;
                }
            }
        }
        self.idle_screen_off = !on;
        if on {
            self.queue_redraw_all();
        }
    }

    /// Record input activity for the idle timer: reset the clock and, if the
    /// panels were powered off, wake them. Called on every input event.
    pub(crate) fn note_input_activity(&mut self) {
        self.idle_last_input = std::time::Instant::now();
        // Reset the ext-idle-notify timers too, so idle daemons see the
        // user as active.
        self.idle_notifier.notify_activity(&self.seat);
        if self.idle_screen_off {
            self.set_screen_power(true);
        }
    }

    /// Reconcile idle inhibition: drop inhibitors whose surface died
    /// (a crashed client never sends the clean destroy) and tell the
    /// ext-idle-notify clients whether idle is currently inhibited.
    pub(crate) fn sync_idle_inhibition(&mut self) {
        self.idle_inhibitors.retain(IsAlive::alive);
        self.idle_notifier
            .set_is_inhibited(!self.idle_inhibitors.is_empty());
    }

    /// Reveal and keyboard-focus a managed surface: switch to its
    /// workspace first (so focusing a window on a hidden workspace shows
    /// it), then move keyboard focus and repaint. No-op if the surface
    /// isn't a managed window. Shared by `xdg_activation` and the IPC
    /// `focus-window`.
    pub(crate) fn focus_surface(&mut self, surface: &WlSurface) {
        let Some(entry) = self
            .layout
            .window_entries()
            .into_iter()
            .find(|e| &e.surface == surface)
        else {
            return;
        };
        self.layout
            .switch_workspace_to(&entry.output, entry.workspace);
        if let Some(kbd) = self.seat.get_keyboard() {
            kbd.set_focus(self, Some(surface.clone()), SERIAL_COUNTER.next_serial());
        }
        // The switch changed what's under the cursor; see the IPC
        // dispatch — same reasoning (stale pointer lock / hidden cursor).
        self.refresh_pointer_focus();
        self.queue_redraw_all();
    }

    /// Idle-timer tick: spawn the lock command and/or DPMS the screens off once
    /// their configured idle thresholds elapse. No-op unless `config.idle` is
    /// set. The lock command spawns once per idle period (re-armed when the
    /// session unlocks); the screen-off fires once and is undone by input.
    pub(crate) fn idle_tick(&mut self) {
        // Prune dead inhibitors and refresh idle-notify's inhibition flag
        // (runs every tick even when our own idle config is unset).
        self.sync_idle_inhibition();
        let Some(idle) = self.config.idle.clone() else {
            return;
        };
        if !self.idle_inhibitors.is_empty() {
            // An idle inhibitor (e.g. a playing video) is active: count it
            // as activity so lock/DPMS never fire and the idle countdown
            // restarts cleanly once the inhibitor goes away.
            self.note_input_activity();
            return;
        }
        let idle_for = self.idle_last_input.elapsed();
        if let Some(after) = idle.lock_after
            && idle_for >= after
            && !self.idle_lock_spawned
            && !self.session_lock_active()
            && let Some(cmd) = &idle.lock_command
        {
            info!(command = %cmd, "idle: spawning lock command");
            if let Some(mut command) = self.build_command(cmd) {
                match command.spawn() {
                    Ok(child) => {
                        info!(pid = child.id(), command = %cmd, "idle: lock command spawned");
                        self.children.push(child);
                    }
                    Err(err) => warn!(error = %err, command = %cmd, "idle: lock command failed"),
                }
            }
            self.idle_lock_spawned = true;
        }
        if let Some(after) = idle.screen_off_after
            && idle_for >= after
            && !self.idle_screen_off
        {
            info!("idle: powering screens off (DPMS)");
            self.set_screen_power(false);
        }
    }

    /// Whether the session is currently locked (`ext-session-lock-v1`).
    pub(crate) fn session_lock_active(&self) -> bool {
        self.session_locked
    }

    /// Engage the lock: from now the renderer draws only lock surfaces and
    /// input goes only to them. Keyboard focus is dropped from the desktop; it
    /// moves to a lock surface as soon as one is created. Called when a client
    /// confirms the lock.
    pub(crate) fn on_session_locked(&mut self) {
        self.session_locked = true;
        if let Some(kbd) = self.seat.get_keyboard() {
            kbd.set_focus(self, None, SERIAL_COUNTER.next_serial());
        }
        self.queue_redraw_all();
    }

    /// Release the lock: drop the lock surfaces, re-arm idle locking, and
    /// redraw the desktop.
    pub(crate) fn on_session_unlocked(&mut self) {
        self.session_locked = false;
        self.lock_surfaces.clear();
        self.idle_lock_spawned = false;
        self.idle_last_input = std::time::Instant::now();
        self.queue_redraw_all();
    }

    /// Register a lock surface for an output: size it to that output, give it
    /// keyboard focus, and redraw so it appears.
    pub(crate) fn add_lock_surface(
        &mut self,
        surface: smithay::wayland::session_lock::LockSurface,
        output: &smithay::reexports::wayland_server::protocol::wl_output::WlOutput,
    ) {
        let Some(name) = smithay::output::Output::from_resource(output).map(|o| o.name()) else {
            return;
        };
        if let Some(rect) = self.renderer.output_rect(&name) {
            surface.with_pending_state(|state| {
                state.size = Some(smithay::utils::Size::from((
                    u32::try_from(rect.size.w).unwrap_or(0),
                    u32::try_from(rect.size.h).unwrap_or(0),
                )));
            });
            surface.send_configure();
        }
        if let Some(kbd) = self.seat.get_keyboard() {
            kbd.set_focus(
                self,
                Some(surface.wl_surface().clone()),
                SERIAL_COUNTER.next_serial(),
            );
        }
        self.lock_surfaces.insert(name, surface);
        self.queue_redraw_all();
    }

    /// While locked, the lock surface (and its output's origin) under `cursor`,
    /// so the pointer/keyboard route only to it. `None` if that output has no
    /// lock surface yet.
    fn locked_surface_at(
        &self,
        cursor: Point<i32, Physical>,
    ) -> Option<(WlSurface, Point<f64, Logical>)> {
        let geom = self.renderer.output_at(cursor)?;
        let surface = self.lock_surfaces.get(&geom.name)?;
        Some((
            surface.wl_surface().clone(),
            Point::<f64, Logical>::from((
                f64::from(geom.compositor.loc.x),
                f64::from(geom.compositor.loc.y),
            )),
        ))
    }

    /// Queue a redraw on every output that *isn't* currently showing a
    /// fullscreen window. Used for redraws we can't pin to one output
    /// (popups, cursor/drag surfaces) and for the safety heartbeat, so we
    /// never stutter a fullscreen client's VRR by flipping its output for
    /// an unrelated reason.
    fn queue_redraw_nonfullscreen(&mut self) {
        for crtc in self.renderer.crtcs() {
            let fullscreen = self
                .renderer
                .output_name_for_crtc(crtc)
                .is_some_and(|name| self.layout.output_has_fullscreen(&name));
            if !fullscreen {
                self.queue_redraw(crtc);
            }
        }
    }

    /// Queue a redraw for the output a committing `surface` is visible on.
    /// Resolving the surface to its output keeps a window on one output (or
    /// in a background workspace) from waking an unrelated output — the
    /// isolation VRR needs. A surface occluded behind a fullscreen window
    /// is skipped; an unplaceable surface (popup, cursor, drag icon) falls
    /// back to every non-fullscreen output.
    fn queue_redraw_for_surface(&mut self, surface: &WlSurface) {
        let root = root_surface(surface);
        // A tracked toplevel (or a subsurface of one) → its output, unless
        // it's hidden behind a fullscreen window there.
        if let Some(name) = self.layout.output_of(&root) {
            if self.layout.output_fullscreen_other_than(&name, &root) {
                return;
            }
            if let Some(crtc) = self.renderer.crtc_for_output_name(&name) {
                self.queue_redraw(crtc);
            }
            return;
        }
        // A layer-shell surface → its bound output, unless a fullscreen
        // window covers it.
        if let Some(name) = self.layer_outputs.get(&root).cloned() {
            if !self.layout.output_has_fullscreen(&name)
                && let Some(crtc) = self.renderer.crtc_for_output_name(&name)
            {
                self.queue_redraw(crtc);
            }
            return;
        }
        // Popup / cursor / drag icon / not-yet-mapped → every output not
        // running a fullscreen window.
        self.queue_redraw_nonfullscreen();
    }

    /// Render and queue one output's frame, then park it (a flip is now in
    /// flight, acked on the next vblank). Re-arms itself via the dirty flag
    /// while a window animation or workspace slide is still running. This is
    /// the old free-run vblank body, now driven on demand by
    /// [`Self::queue_redraw`] and re-driven by the vblank handler.
    /// Toplevel placement surfaces whose window should be HDR-decoded. A
    /// client may tag a *subsurface* (e.g. a Proton/Wine game's swapchain)
    /// rather than its toplevel, and the renderer keys on the toplevel
    /// placement surface — so a window counts as HDR if its toplevel OR any
    /// surface in its tree carries an HDR (PQ/HLG) image description.
    fn hdr_surface_ids(
        &self,
        placements: &[layout::Placement],
    ) -> std::collections::HashSet<smithay::reexports::wayland_server::backend::ObjectId> {
        let tagged: std::collections::HashSet<_> = self
            .color_surfaces
            .iter()
            .filter(|(_, c)| c.image_description.is_hdr())
            .map(|(id, _)| id.clone())
            .collect();
        let mut out = std::collections::HashSet::new();
        if tagged.is_empty() {
            return out;
        }
        // A window counts as HDR if any surface in its tree (toplevel or a
        // subsurface — e.g. a game's swapchain child) carries an HDR-tagged
        // image description.
        for p in placements {
            let mut hit = false;
            smithay::wayland::compositor::with_surface_tree_downward(
                &p.surface,
                (),
                |_, _, ()| smithay::wayland::compositor::TraversalAction::DoChildren(()),
                |s, _, ()| {
                    if tagged.contains(&s.id()) {
                        hit = true;
                    }
                },
                |_, _, ()| true,
            );
            if hit {
                out.insert(p.surface.id());
            }
        }
        out
    }

    #[allow(
        clippy::too_many_lines,
        reason = "linear per-output render driver: snapshot placements/layers/popups, drain captures, sync the hardware cursor, render, then park/retry on the result. Splitting it just threads frame state through extra functions."
    )]
    fn render_crtc(&mut self, crtc: crtc::Handle) {
        let focused = self.seat.get_keyboard().and_then(|k| k.current_focus());
        // Workspace slide spec (None when disabled). Clear finished slides,
        // then emit (both workspaces during one, the active one otherwise).
        let ws_anim = self.config.animations.workspace;
        let slide = (self.config.animations.enabled && ws_anim.enabled).then_some(ws_anim);
        self.layout.tick_transitions(slide);
        let placements = self.layout.placements(focused.as_ref(), slide);
        let layer_placements = self.snapshot_layer_placements();
        let popup_placements = self.snapshot_popup_placements(&placements, &layer_placements);

        // Drain pending captures for the output this CRTC drives so the
        // renderer can read them off this frame's framebuffer.
        let out_name = self.renderer.output_name_for_crtc(crtc);

        // Point each visible window's per-surface dmabuf feedback at the
        // right variant (scanout tranche ⇔ fullscreen). Deduped inside
        // smithay, so calling per frame only costs a short tree walk.
        self.sync_scanout_feedback(&placements, out_name.as_deref());

        // Session locked: blank everything and draw only this output's lock
        // surface, full-size, as an Overlay (above all windows/layers). Reusing
        // the Overlay layer path means no special render branch. Nothing else
        // (windows, real layers, popups) reaches the screen.
        let (placements, layer_placements, popup_placements) = if self.session_locked {
            let lock = out_name.as_deref().and_then(|name| {
                let surface = self.lock_surfaces.get(name)?;
                let rect = self.renderer.output_rect(name)?;
                Some(render::LayerPlacement {
                    surface: surface.wl_surface().clone(),
                    rect,
                    layer: render::LayerBucket::Overlay,
                    namespace: String::new(),
                })
            });
            (Vec::new(), lock.into_iter().collect::<Vec<_>>(), Vec::new())
        } else {
            (placements, layer_placements, popup_placements)
        };
        let mut captures: Vec<screencopy::PendingCapture> = Vec::new();
        let mut internal: Vec<InternalCapture> = Vec::new();
        if let Some(name) = out_name.as_deref() {
            let pending = std::mem::take(&mut self.screencopy_pending);
            let (mine, rest): (Vec<_>, Vec<_>) =
                pending.into_iter().partition(|p| p.output == name);
            self.screencopy_pending = rest;
            captures = mine;

            let ipending = std::mem::take(&mut self.screenshot_pending);
            let (imine, irest): (Vec<_>, Vec<_>) =
                ipending.into_iter().partition(|c| c.output == name);
            self.screenshot_pending = irest;
            internal = imine;
        }
        let mut specs: Vec<render::CaptureSpec> = captures
            .iter()
            .map(|c| render::CaptureSpec {
                region: c.region,
                fourcc: screencopy::CAPTURE_FOURCC,
                target: match smithay::wayland::dmabuf::get_dmabuf(&c.buffer) {
                    Ok(dmabuf) => render::CaptureTarget::Dmabuf(dmabuf.clone()),
                    Err(_) => render::CaptureTarget::Shm,
                },
            })
            .collect();
        for c in &internal {
            specs.push(render::CaptureSpec {
                region: c.region,
                fourcc: screencopy::CAPTURE_FOURCC,
                target: render::CaptureTarget::Shm,
            });
        }
        let capture_hides_cursor =
            !captures.is_empty() && captures.iter().all(|c| !c.overlay_cursor);
        let internal_hides_cursor =
            !internal.is_empty() && internal.iter().all(|c| !c.show_cursor);
        let locked = self.pointer_locked();
        let hide_cursor = locked || capture_hides_cursor || internal_hides_cursor;
        // A capture this frame wants the cursor baked into the framebuffer —
        // composite the themed cursor even though the hardware plane shows it.
        let compose_cursor = captures.iter().any(|c| c.overlay_cursor)
            || internal.iter().any(|c| c.show_cursor);
        // Sync the hardware cursor plane to the current status before rendering
        // (cheap + idempotent; programs it on the output under the pointer).
        self.renderer.refresh_hw_cursor(locked);
        let client_n = captures.len();
        let hdr_surface_ids = self.hdr_surface_ids(&placements);
        // The smithay `Output` this CRTC drives, for `wp_presentation`
        // feedback collection (Arc-backed, cheap to clone).
        let present_output = out_name
            .as_deref()
            .and_then(|name| self.outputs.iter().find(|o| o.name() == name))
            .cloned();
        let followup = match self.renderer.render_for_crtc(
            crtc,
            &placements,
            &layer_placements,
            &popup_placements,
            hide_cursor,
            &specs,
            &hdr_surface_ids,
            compose_cursor,
            present_output.as_ref(),
        ) {
            Ok((mut results, followup)) => {
                // Trailing results belong to the internal specs.
                let internal_results = results.split_off(client_n.min(results.len()));
                for (pending, captured) in captures.iter().zip(results) {
                    screencopy::complete(pending, captured);
                }
                for (cap, outcome) in internal.into_iter().zip(internal_results) {
                    self.complete_internal_capture(cap, outcome);
                }
                followup
            }
            Err(err) => {
                // Don't kill the loop on a render hiccup; a later trigger
                // retries this output. Fail any captures riding this frame.
                warn!(error = %err, ?crtc, "render_for_crtc failed");
                for pending in &captures {
                    screencopy::complete(pending, render::CaptureOutcome::Failed);
                }
                if !internal.is_empty() {
                    warn!("screenshot: capture dropped (render failed)");
                }
                // No flip was queued, so park (don't wait for a vblank that
                // won't come); the next trigger retries.
                self.redraw.insert(crtc, RedrawState::Idle);
                return;
            }
        };
        // A flip is in flight; park until its vblank. Keep rendering while a
        // window animation (followup) or a workspace slide is still running.
        let dirty = followup || self.layout.is_animating();
        self.redraw
            .insert(crtc, RedrawState::WaitingForVblank { dirty });
    }

    /// Hit-test the desktop at `cursor_i`: returns the surface that should
    /// take the *pointer* (with the buffer origin that makes surface-local
    /// coordinates correct) and the surface that should take the
    /// *keyboard* in the Hover model. A popup under the cursor wins the
    /// pointer (menus draw on top of everything), then layer surfaces
    /// (rofi, panels, OSDs), then the tile / floating layout. The keyboard
    /// target skips popups — we don't run a popup grab yet, and a menu
    /// shouldn't pull keyboard focus off its parent toplevel. While the
    /// session is locked, only the lock surface is reachable — no popups,
    /// layers, or windows take either focus.
    #[allow(
        clippy::type_complexity,
        reason = "the tuple mirrors smithay's pointer focus type (surface + f64 origin); naming it would add a struct used by exactly two callers"
    )]
    fn pointer_hit_test(
        &self,
        cursor_i: Point<i32, Physical>,
    ) -> (
        Option<(WlSurface, Point<f64, Logical>)>,
        Option<WlSurface>,
    ) {
        let popup_hit = if self.session_locked {
            None
        } else {
            self.popup_at(cursor_i)
        };
        let surface_hit = if self.session_locked {
            self.locked_surface_at(cursor_i)
        } else {
            self.layer_at(cursor_i)
                .map(|(surface, rect)| {
                    (
                        surface,
                        Point::<f64, Logical>::from((f64::from(rect.loc.x), f64::from(rect.loc.y))),
                    )
                })
                .or_else(|| {
                    self.layout.window_at(cursor_i).map(|(w, rect)| {
                        // A CSD client pads its buffer with an invisible shadow
                        // margin and reports the real content rect via
                        // set_window_geometry; the render path shifts the buffer
                        // up-left by that offset so the visible content lands at
                        // the cell origin. The pointer focus origin must use that
                        // SAME shifted buffer origin — otherwise surface-local
                        // coordinates are off by the shadow margin and clicks land
                        // below where the content visually is (the Lutris "+"
                        // button). The shadow only exists on Normal windows;
                        // maximized/fullscreen drop it, matching `grouped` in
                        // render_output.
                        let (gx, gy) = if w.fill == crate::layout::FillMode::Normal {
                            crate::render::window_geometry_offset(w.toplevel.wl_surface())
                        } else {
                            (0, 0)
                        };
                        (
                            w.toplevel.wl_surface().clone(),
                            Point::<f64, Logical>::from((
                                f64::from(rect.loc.x - gx),
                                f64::from(rect.loc.y - gy),
                            )),
                        )
                    })
                })
        };
        let kbd_target = surface_hit.as_ref().map(|(surface, _)| surface.clone());
        (popup_hit.or(surface_hit), kbd_target)
    }

    /// Re-aim pointer focus after a compositor-driven scene change
    /// (workspace switch, window moved to another workspace, close, …):
    /// re-hit-test the stationary cursor and, if a different surface is
    /// now under it, hand pointer focus over with a synthetic motion.
    ///
    /// Pointer focus otherwise only updates on physical motion — and the
    /// motion path can't recover on its own, because its pointer-lock
    /// check reads the *current* focus: a game that locked the pointer
    /// (cursor hidden) keeps that focus through a workspace switch, so
    /// every subsequent motion short-circuits down the locked path and
    /// the user is stranded on the new workspace with a frozen, invisible
    /// cursor. Handing focus to the real hit makes smithay's leave /
    /// replace path deactivate the stale constraint and reset the cursor
    /// image to the default arrow (`vendor/smithay` `PointerTarget::leave`
    /// / `replace`).
    ///
    /// No-op while a pointer grab, compositor drag, or screenshot session
    /// owns the pointer (those route by position and must not have focus
    /// yanked mid-gesture), and when the hit is unchanged — a still
    /// visible locked surface must not receive a synthetic absolute
    /// motion (pointer-constraints forbids motion events while locked).
    pub(crate) fn refresh_pointer_focus(&mut self) {
        let Some(pointer) = self.seat.get_pointer() else {
            return;
        };
        if self.drag.is_some() || self.screenshot.is_some() || pointer.is_grabbed() {
            return;
        }
        let (cx, cy) = self.renderer.cursor_pos();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "cursor coords are clamped to layout_bounds (i32) in Renderer::on_pointer_motion"
        )]
        let cursor_i = Point::<i32, Physical>::from((cx as i32, cy as i32));
        let (hit, _) = self.pointer_hit_test(cursor_i);
        if pointer.current_focus() == hit.as_ref().map(|(surface, _)| surface.clone()) {
            return;
        }
        pointer.motion(
            self,
            hit,
            &MotionEvent {
                location: Point::<f64, Logical>::from((cx, cy)),
                serial: SERIAL_COUNTER.next_serial(),
                time: Clock::<Monotonic>::new().now().as_millis(),
            },
        );
        pointer.frame(self);
    }

    /// Forward the current cursor location to the focused client as
    /// a `wl_pointer.motion` event (smithay generates enter/leave
    /// when the focus surface changes). Hit-tests the layout to
    /// pick the surface under the cursor; in `FocusModel::Hover` the
    /// same surface also takes keyboard focus, but only on actual
    /// change so we don't flood `wl_keyboard.enter` /
    /// `wl_keyboard.leave` on every motion event.
    ///
    /// When [`State::drag`] is active, the motion is consumed by
    /// the drag instead: the dragged window's rect is updated
    /// (translated for Move, stretched for Resize) and no
    /// `wl_pointer.motion` is sent — the focused client should
    /// see a still pointer until the drag ends.
    #[allow(
        clippy::too_many_lines,
        reason = "single decision tree for one event source (pointer motion): drag, constraint check, locked short-circuit, confine clamp, hit-test, relative + absolute motion, and constraint activation. Splitting any piece out would thread the cursor/focus/constraint state through another method for no clarity gain."
    )]
    fn forward_pointer_motion<B: InputBackend>(&mut self, evt: &B::PointerMotionEvent) {
        let delta = evt.delta();
        let delta_unaccel = evt.delta_unaccel();
        let time = evt.time_msec();
        let utime = evt.time();

        // Move the pointer. With a hardware cursor plane the themed cursor is
        // repositioned by a cheap `move_cursor` ioctl with NO full re-render —
        // the whole point of the plane. Fall back to a redraw only when the
        // cursor is software (client surface / no plane / oversize) or
        // something else tracks the pointer (a DnD icon or screenshot
        // selection follows it in the composite). Skip entirely while the
        // pointer is locked (cursor hidden; the client drives its own frames).
        if !self.pointer_locked() {
            let needs_redraw = self.renderer.has_dnd_icon()
                || self.screenshot.is_some()
                || !self.renderer.move_hw_cursor();
            if needs_redraw {
                self.queue_redraw_all();
            }
        }

        // A screenshot session owns the pointer: move the cursor and
        // update the live selection (region drag corner / hovered
        // window), swallowing the event so no client sees it.
        if self.screenshot.is_some() {
            self.renderer.on_pointer_motion(delta.x, delta.y);
            self.screenshot_pointer_motion();
            return;
        }

        // Drag in flight: the cursor follows the drag (a compositor
        // gesture, unaffected by client pointer constraints). Move it,
        // update the dragged window's rect, and swallow the event so
        // the client sees a still pointer under the grab. Move drags
        // update `in_transit` (a float that follows the cursor and
        // rejoins the tree on drop); Resize drags resize in place.
        if let Some(drag) = self.drag.clone() {
            self.renderer.on_pointer_motion(delta.x, delta.y);
            let (cx, cy) = self.renderer.cursor_pos();
            #[allow(
                clippy::cast_possible_truncation,
                reason = "cursor deltas are bounded by layout_bounds (i32) from on_pointer_motion"
            )]
            let delta_x = (cx - drag.cursor_start.0) as i32;
            #[allow(
                clippy::cast_possible_truncation,
                reason = "cursor deltas are bounded by layout_bounds (i32) from on_pointer_motion"
            )]
            let delta_y = (cy - drag.cursor_start.1) as i32;
            match drag.mode {
                DragMode::Move => {
                    let new_rect = smithay::utils::Rectangle::new(
                        Point::<i32, Physical>::new(
                            drag.rect_start.loc.x + delta_x,
                            drag.rect_start.loc.y + delta_y,
                        ),
                        drag.rect_start.size,
                    );
                    self.layout.update_in_transit_rect(new_rect);
                }
                DragMode::Resize => {
                    let new_rect = smithay::utils::Rectangle::new(
                        drag.rect_start.loc,
                        smithay::utils::Size::<i32, Physical>::new(
                            (drag.rect_start.size.w + delta_x).max(MIN_DRAG_RESIZE_W),
                            (drag.rect_start.size.h + delta_y).max(MIN_DRAG_RESIZE_H),
                        ),
                    );
                    self.layout.set_floating_rect(&drag.surface, new_rect);
                }
            }
            return;
        }

        let Some(pointer) = self.seat.get_pointer() else {
            return;
        };

        // Inspect any active pointer constraint on the surface the
        // cursor is currently over. A locked pointer must not move the
        // cursor (the client reads relative motion instead — this is
        // mouse-look in games); a confined one keeps the cursor inside
        // the surface. Smithay deactivates the constraint automatically
        // when the surface loses pointer focus.
        let current = pointer.current_focus();
        let mut locked = false;
        let mut confined = false;
        if let Some(surface) = current.as_ref() {
            with_pointer_constraint(surface, &pointer, |constraint| {
                if let Some(constraint) = constraint
                    && constraint.is_active()
                {
                    match &*constraint {
                        PointerConstraint::Locked(_) => locked = true,
                        PointerConstraint::Confined(_) => confined = true,
                    }
                }
            });
        }

        let relative = RelativeMotionEvent {
            delta,
            delta_unaccel,
            utime,
        };

        if locked {
            // Cursor frozen; deliver only relative motion to the locked
            // client (the focus origin is unused by relative_motion).
            let focus = current.map(|s| (s, Point::<f64, Logical>::from((0.0, 0.0))));
            pointer.relative_motion(self, focus, &relative);
            pointer.frame(self);
            return;
        }

        // Confined: the cursor is currently over the confining surface,
        // so grab that surface's rect to clamp the move back inside.
        let confine_rect = if confined {
            let (cx, cy) = self.renderer.cursor_pos();
            #[allow(
                clippy::cast_possible_truncation,
                reason = "cursor coords are clamped to layout_bounds (i32) in Renderer::on_pointer_motion"
            )]
            let ci = Point::<i32, Physical>::from((cx as i32, cy as i32));
            self.layer_at(ci)
                .map(|(_, rect)| rect)
                .or_else(|| self.layout.window_at(ci).map(|(_, rect)| rect))
        } else {
            None
        };

        // Move the absolute cursor, then confine it if needed.
        self.renderer.on_pointer_motion(delta.x, delta.y);
        if let Some(rect) = confine_rect {
            self.renderer.confine_cursor(rect);
        }

        let (cx, cy) = self.renderer.cursor_pos();
        let location = Point::<f64, Logical>::from((cx, cy));
        #[allow(
            clippy::cast_possible_truncation,
            reason = "cursor coords are clamped to layout_bounds (i32) in Renderer::on_pointer_motion"
        )]
        let cursor_i = Point::<i32, Physical>::from((cx as i32, cy as i32));

        // Snapshot the hit surface + its top-left into owned values
        // so the layout borrow ends before the mut-borrow on self
        // (via kbd.set_focus / pointer.motion).
        let (hit, kbd_target) = self.pointer_hit_test(cursor_i);

        // Relative motion goes to the *pre-move* pointer focus
        // (`pointer.motion` below updates it), matching how
        // compositors attribute the delta to where the pointer was.
        pointer.relative_motion(self, hit.clone(), &relative);

        // Pull keyboard focus to the surface under the cursor in the Hover
        // focus model — but NOT while a pointer grab is active. During a
        // drag (DnD, or an interactive move/resize) the offer/grab is
        // routed by pointer position alone, so changing keyboard focus is
        // both unnecessary and disruptive: `focus_changed` re-points the
        // data-device selection (`set_data_device_focus`), which would push
        // the hovered client a clipboard offer mid-drag, and for a
        // move/resize the cursor sweeps over windows that shouldn't grab
        // focus. Never steal focus from an exclusive-keyboard layer surface
        // (e.g. rofi) either.
        // An open popup grab (a menu) pins keyboard focus to its owner
        // so hover can't move focus away — otherwise the client sees its
        // toplevel lose focus and dismisses the menu. Expire the grab
        // first so focus resumes the instant the chain closes.
        self.refresh_popup_grab();
        if matches!(self.config.input.focus_model, config::FocusModel::Hover)
            && !pointer.is_grabbed()
            && !self.focus_locked_by_layer()
            && self.popup_grab.is_none()
            && let Some(kbd) = self.seat.get_keyboard()
            && kbd.current_focus() != kbd_target
        {
            kbd.set_focus(self, kbd_target, SERIAL_COUNTER.next_serial());
        }

        let serial = SERIAL_COUNTER.next_serial();
        pointer.motion(
            self,
            hit.clone(),
            &MotionEvent {
                location,
                serial,
                time,
            },
        );
        pointer.frame(self);

        // Activate a not-yet-active constraint once the pointer enters
        // its surface (and region, if any) — covers a lock requested
        // while the surface was unfocused.
        if let Some((surface, origin)) = hit.as_ref() {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "surface-local coords are bounded by the output rect (i32)"
            )]
            let local = Point::<i32, Logical>::from((
                (location.x - origin.x) as i32,
                (location.y - origin.y) as i32,
            ));
            with_pointer_constraint(surface, &pointer, |constraint| {
                if let Some(constraint) = constraint
                    && !constraint.is_active()
                    && constraint.region().is_none_or(|r| r.contains(local))
                {
                    constraint.activate();
                }
            });
        }
    }

    /// Forward a pointer button press/release to the focused client.
    /// `button` is the raw evdev button code (`BTN_LEFT = 0x110`, …)
    /// which is exactly what `wl_pointer.button` carries. In
    /// `FocusModel::Click` a *press* also promotes the surface under
    /// the cursor to keyboard focus before the button event is sent,
    /// so the focused client sees its first key as expected.
    ///
    /// Super + LMB press starts an interactive Move drag on the
    /// window under the cursor: the window is pulled out of its
    /// current home (tree or floating list) into `in_transit` and
    /// follows the cursor; on release it rejoins the tree at the
    /// drop position (if it came from the tree) or the floating
    /// stack (if it was already floating). Super + RMB starts a
    /// Resize drag, which only works on floating windows — resize
    /// on a tile is a logged no-op (use Super+F first). While a
    /// drag is active, any release ends it, and no press / release
    /// leaks to the focused client.
    #[allow(
        clippy::too_many_lines,
        reason = "this function is a single decision tree for one event source (pointer button) — drag end, drag start, click-to-focus, and normal forwarding all live here. Splitting any of those out duplicates the active-drag short-circuit checks at every site, which is worse than the length."
    )]
    fn forward_pointer_button(
        &mut self,
        button: u32,
        state: smithay::backend::input::ButtonState,
        time: u32,
    ) {
        use smithay::backend::input::ButtonState;
        // A button can change focus (border colour), start/end a drag, or
        // drive the screenshot UI — all visual. Queue a redraw up front so
        // every path is covered; it coalesces into an actively-rendering
        // output, so it doesn't perturb a focused game's VRR.
        self.queue_redraw_all();
        // A screenshot session owns the pointer. Left button: PRESS starts
        // the region drag (anchor) / RELEASE finalizes it; for window mode
        // a press captures the hovered window. Right button cancels. All
        // buttons are swallowed (never reach a client).
        if self.screenshot.is_some() {
            if button == BTN_LEFT {
                match state {
                    ButtonState::Pressed => self.screenshot_pointer_press(),
                    ButtonState::Released => self.screenshot_pointer_release(),
                }
            } else if button == BTN_RIGHT && matches!(state, ButtonState::Pressed) {
                self.cancel_screenshot();
            }
            return;
        }

        // Active drag: any release ends it. Other buttons are
        // swallowed so we don't accidentally cancel mid-drag.
        if self.drag.is_some() {
            if matches!(state, smithay::backend::input::ButtonState::Released) {
                let drag = self.drag.take().expect("checked is_some above");
                if matches!(drag.mode, DragMode::Move) {
                    let (cx, cy) = self.renderer.cursor_pos();
                    #[allow(
                        clippy::cast_possible_truncation,
                        reason = "cursor coords are clamped to layout_bounds (i32) in Renderer::on_pointer_motion"
                    )]
                    let cursor_i = Point::<i32, Physical>::from((cx as i32, cy as i32));
                    self.layout.finish_move_drag(cursor_i);
                }
                // Drop the gesture-cursor override; the client under the
                // pointer drives the cursor again from here.
                self.renderer.set_cursor_override(None);
                // Re-enable move animation for the window so it eases into
                // its final tile (move) or stays put (resize).
                self.renderer.set_no_anim_move(None);
            }
            return;
        }

        // Dismiss-on-click-outside: a press that lands outside every
        // open popup closes the whole menu chain (pragmatic stand-in
        // for a real popup grab). The press still forwards below to
        // whatever's under the cursor, so the click also lands.
        if matches!(state, smithay::backend::input::ButtonState::Pressed) {
            let (cx, cy) = self.renderer.cursor_pos();
            #[allow(
                clippy::cast_possible_truncation,
                reason = "cursor coords are clamped to layout_bounds (i32) in Renderer::on_pointer_motion"
            )]
            let cursor_i = Point::<i32, Physical>::from((cx as i32, cy as i32));
            if self.popup_at(cursor_i).is_none() {
                self.dismiss_all_popups();
            }
        }

        // Drag start: Super + (LMB or RMB) press on a window.
        if matches!(state, smithay::backend::input::ButtonState::Pressed) {
            let super_held = self
                .seat
                .get_keyboard()
                .is_some_and(|k| k.modifier_state().logo);
            let mode = if super_held {
                match button {
                    BTN_LEFT => Some(DragMode::Move),
                    BTN_RIGHT => Some(DragMode::Resize),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(mode) = mode {
                let (cx, cy) = self.renderer.cursor_pos();
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "cursor coords are clamped to layout_bounds (i32) in Renderer::on_pointer_motion"
                )]
                let cursor_i = Point::<i32, Physical>::from((cx as i32, cy as i32));
                let hit_surface = self
                    .layout
                    .window_at(cursor_i)
                    .map(|(w, _)| w.toplevel.wl_surface().clone());
                if let Some(surface) = hit_surface {
                    // Focus the dragged window before the grab so
                    // releases-after-drag still find the right
                    // client for keyboard input. In hover mode
                    // it's usually already focused; in click mode
                    // this is the refocus-on-press path.
                    if let Some(kbd) = self.seat.get_keyboard()
                        && kbd.current_focus().as_ref() != Some(&surface)
                    {
                        kbd.set_focus(self, Some(surface.clone()), SERIAL_COUNTER.next_serial());
                    }
                    let rect_start = match mode {
                        DragMode::Move => self.layout.start_move_drag(&surface),
                        DragMode::Resize => self.layout.start_resize_drag(&surface),
                    };
                    if let Some(rect_start) = rect_start {
                        info!(
                            ?mode,
                            surface = ?surface.id(),
                            "drag start"
                        );
                        // Draw the dragged window 1:1 with the cursor
                        // (no move-animation lag); it animates into place
                        // on drop when the override is cleared.
                        self.renderer.set_no_anim_move(Some(&surface));
                        self.drag = Some(DragState {
                            surface,
                            mode,
                            cursor_start: (cx, cy),
                            rect_start,
                        });
                        // Show the gesture cursor for the drag: the
                        // grabbing hand while moving, a resize cursor
                        // while resizing. Overrides the client's cursor
                        // until the drag ends.
                        let icon = match mode {
                            DragMode::Move => CursorIcon::Grabbing,
                            DragMode::Resize => CursorIcon::SeResize,
                        };
                        self.renderer
                            .set_cursor_override(Some(CursorImageStatus::Named(icon)));
                    } else if matches!(mode, DragMode::Resize) {
                        warn!(
                            surface = ?surface.id(),
                            "Super+RMB resize is only supported on floating windows; toggle floating (Super+F) first"
                        );
                    }
                    // Either way we don't forward this press to
                    // the client — Super+click is the compositor's
                    // gesture, not the client's.
                    return;
                }
                // Super + click on empty wallpaper: falls through
                // to the normal forward path below (which sends
                // the press to whatever pointer-focus surface, if
                // any — likely none over the wallpaper).
            }
        }

        if matches!(state, smithay::backend::input::ButtonState::Pressed)
            && matches!(self.config.input.focus_model, config::FocusModel::Click)
            && !self.focus_locked_by_layer()
        {
            let (cx, cy) = self.renderer.cursor_pos();
            #[allow(
                clippy::cast_possible_truncation,
                reason = "cursor coords are clamped to layout_bounds (i32) in Renderer::on_pointer_motion"
            )]
            let cursor_i = Point::<i32, Physical>::from((cx as i32, cy as i32));
            // Clicking inside a popup must not re-home keyboard focus
            // to the window/layer beneath it — leave focus on the
            // popup's parent. While locked, keyboard focus stays on the lock
            // surface — never re-homed to a (hidden) window/layer on click.
            if !self.session_locked && self.popup_at(cursor_i).is_none() {
                // Prefer a layer surface under the cursor (rofi / OSDs)
                // over a tile so click-to-focus on a panel works without
                // a separate path.
                let target = self.layer_at(cursor_i).map(|(s, _)| s).or_else(|| {
                    self.layout
                        .window_at(cursor_i)
                        .map(|(w, _)| w.toplevel.wl_surface().clone())
                });
                if let Some(kbd) = self.seat.get_keyboard()
                    && kbd.current_focus() != target
                {
                    kbd.set_focus(self, target, SERIAL_COUNTER.next_serial());
                }
            }
        }

        let Some(pointer) = self.seat.get_pointer() else {
            return;
        };
        let serial = SERIAL_COUNTER.next_serial();
        pointer.button(
            self,
            &ButtonEvent {
                button,
                state,
                serial,
                time,
            },
        );
        pointer.frame(self);
    }

    /// Forward a scroll (axis) event to the pointer-focused client.
    /// Without this, scrollable surfaces never receive
    /// `wl_pointer.axis` and can't scroll at all.
    ///
    /// Mirrors smithay's anvil reference: send the continuous `value`
    /// (touchpads and high-resolution wheels) alongside discrete
    /// `v120` steps (notched mouse wheels), tag the axis source and
    /// per-axis relative direction, and emit a `stop` when a finger
    /// scroll ends so kinetic scrolling halts cleanly.
    fn forward_pointer_axis<B: InputBackend>(&mut self, evt: &B::PointerAxisEvent) {
        // Super[+Shift] + vertical wheel is the workspace gesture, not
        // a client scroll: intercept it (don't forward). We key off
        // the discrete v120 wheel signal and accumulate to ±120 so one
        // physical notch = exactly one workspace step. A touchpad
        // (no v120) under Super falls through to normal forwarding.
        let mods = self.seat.get_keyboard().map(|k| k.modifier_state());
        if mods.as_ref().is_some_and(|m| m.logo)
            && let Some(v120) = evt.amount_v120(Axis::Vertical)
            && v120 != 0.0
        {
            let shift = mods.as_ref().is_some_and(|m| m.shift);
            self.ws_scroll_accum += v120;
            // Positive v120 = scroll-down = +1 workspace.
            while self.ws_scroll_accum >= 120.0 {
                self.ws_scroll_accum -= 120.0;
                self.workspace_gesture(shift, 1);
            }
            while self.ws_scroll_accum <= -120.0 {
                self.ws_scroll_accum += 120.0;
                self.workspace_gesture(shift, -1);
            }
            return;
        }

        // Continuous amount; for wheels that only report discrete
        // detents, synthesise a small continuous value from v120 so
        // clients that ignore v120 still scroll.
        let amount = |axis: Axis| {
            evt.amount(axis)
                .unwrap_or_else(|| evt.amount_v120(axis).unwrap_or(0.0) * 3.0 / 120.0)
        };
        let horizontal = amount(Axis::Horizontal);
        let vertical = amount(Axis::Vertical);
        let source = evt.source();

        let mut frame = AxisFrame::new(evt.time_msec()).source(source);
        if horizontal == 0.0 {
            if source == AxisSource::Finger {
                frame = frame.stop(Axis::Horizontal);
            }
        } else {
            frame = frame
                .relative_direction(Axis::Horizontal, evt.relative_direction(Axis::Horizontal))
                .value(Axis::Horizontal, horizontal);
            if let Some(v120) = evt.amount_v120(Axis::Horizontal) {
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "v120 is an integer detent count (multiple of 120) carried as f64"
                )]
                let steps = v120 as i32;
                frame = frame.v120(Axis::Horizontal, steps);
            }
        }
        if vertical == 0.0 {
            if source == AxisSource::Finger {
                frame = frame.stop(Axis::Vertical);
            }
        } else {
            frame = frame
                .relative_direction(Axis::Vertical, evt.relative_direction(Axis::Vertical))
                .value(Axis::Vertical, vertical);
            if let Some(v120) = evt.amount_v120(Axis::Vertical) {
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "v120 is an integer detent count (multiple of 120) carried as f64"
                )]
                let steps = v120 as i32;
                frame = frame.v120(Axis::Vertical, steps);
            }
        }

        let Some(pointer) = self.seat.get_pointer() else {
            return;
        };
        pointer.axis(self, frame);
        pointer.frame(self);
    }

    /// One workspace step for the `Super`[+`Shift`]+scroll gesture.
    /// `delta` is `+1` (scroll-down / next) or `-1` (scroll-up /
    /// prev). With `shift`, move the keyboard-focused window to the
    /// adjacent workspace on its own output and follow it; otherwise
    /// switch the workspace on the output under the cursor and
    /// re-derive focus (the previously-focused window is now hidden).
    fn workspace_gesture(&mut self, shift: bool, delta: i32) {
        // Switching or moving across workspaces reflows + starts a slide;
        // redraw so it shows (the slide then self-sustains via followup).
        self.queue_redraw_all();
        if shift {
            if let Some(surface) = self.seat.get_keyboard().and_then(|k| k.current_focus()) {
                let moved = self.layout.move_focused_window(&surface, delta);
                debug!(delta, moved, "move focused window to adjacent workspace");
            }
            return;
        }
        let (cx, cy) = self.renderer.cursor_pos();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "cursor coords are clamped to layout_bounds (i32) in Renderer::on_pointer_motion"
        )]
        let cursor = Point::<i32, Physical>::from((cx as i32, cy as i32));
        if !self.layout.switch_at(cursor, delta) {
            return;
        }
        // The switch hid the old focus; focus the window now under the
        // cursor on the new workspace (or clear focus if it's empty
        // there) so keyboard input and the active border follow.
        let new_focus = self
            .layout
            .window_at(cursor)
            .map(|(w, _)| w.toplevel.wl_surface().clone());
        if let Some(kbd) = self.seat.get_keyboard() {
            kbd.set_focus(self, new_focus, SERIAL_COUNTER.next_serial());
        }
    }

    /// The compositor rect of the output a layer surface is bound to —
    /// the one it named at creation (recorded in [`Self::layer_outputs`]),
    /// or the primary output when it named none or the output is gone.
    pub(crate) fn layer_output_rect(
        &self,
        surface: &WlSurface,
    ) -> smithay::utils::Rectangle<i32, smithay::utils::Physical> {
        self.layer_outputs
            .get(surface)
            .and_then(|name| self.renderer.output_rect(name))
            .unwrap_or_else(|| self.renderer.primary_output_rect())
    }

    /// Build one `LayerPlacement` per live layer surface for this
    /// frame. The renderer needs each surface's top-left in
    /// compositor coords + a layer bucket so it knows whether to
    /// paint background/bottom surfaces between the wallpaper and
    /// the tiles, or top/overlay surfaces between the tiles and
    /// the cursor.
    ///
    /// Position is derived from the cached layer state's anchor +
    /// margin + size, evaluated against **the output the surface asked
    /// for** (tracked in [`Self::layer_outputs`]; falls back to primary
    /// when the client let the compositor choose). Surfaces with a
    /// non-positive size (set by clients that want the compositor to
    /// choose) get the full output dimension in the unsized axis;
    /// non-anchored surfaces are centred on their output.
    pub(crate) fn snapshot_layer_placements(&self) -> Vec<render::LayerPlacement> {
        use smithay::wayland::shell::wlr_layer::{Anchor, Layer};
        let mut out = Vec::new();
        for layer_surface in self.layer_shell_state.layer_surfaces() {
            let surface = layer_surface.wl_surface();
            let area = self.layer_output_rect(surface);
            let cached = crate::wayland::layer_cached_state(surface);
            let anchor = cached.anchor;
            // Size honours anchors/stretch/margins, shared with the
            // `configure` we send the client so the two can't disagree.
            let (width, height) = crate::wayland::layer_size(area, &cached);
            // Position: pinned to an anchored edge (+ its margin), else
            // centred. When stretched, LEFT/TOP is set so the surface
            // starts at the margin and spans to the far margin.
            let x = if anchor.contains(Anchor::LEFT) {
                area.loc.x + cached.margin.left
            } else if anchor.contains(Anchor::RIGHT) {
                area.loc.x + area.size.w - width - cached.margin.right
            } else {
                area.loc.x + (area.size.w - width) / 2
            };
            let y = if anchor.contains(Anchor::TOP) {
                area.loc.y + cached.margin.top
            } else if anchor.contains(Anchor::BOTTOM) {
                area.loc.y + area.size.h - height - cached.margin.bottom
            } else {
                area.loc.y + (area.size.h - height) / 2
            };
            let bucket = match cached.layer {
                Layer::Background => render::LayerBucket::Background,
                Layer::Bottom => render::LayerBucket::Bottom,
                Layer::Top => render::LayerBucket::Top,
                Layer::Overlay => render::LayerBucket::Overlay,
            };
            out.push(render::LayerPlacement {
                surface: surface.clone(),
                rect: smithay::utils::Rectangle::new(
                    smithay::utils::Point::new(x, y),
                    smithay::utils::Size::new(width, height),
                ),
                layer: bucket,
                namespace: self.layer_namespaces.get(surface).cloned().unwrap_or_default(),
            });
        }
        out
    }

    /// Snapshot every live `xdg_popup` (menus, submenus, combo
    /// dropdowns) to draw this frame. xdg-shell positions a popup
    /// relative to its parent's *window geometry*: for a
    /// toplevel/floating window we paint the surface buffer (0,0) at
    /// `cell_rect.loc + border_width` after folding out the client's
    /// own geometry offset, so the parent's window-geometry origin is
    /// exactly `cell_rect.loc + border_width`; for a layer surface it
    /// is the surface rect origin.
    ///
    /// `PopupManager::popups_for_surface(root)` yields each popup in
    /// the tree with its cumulative location relative to that origin,
    /// so the popup's window-geometry top-left is `origin + location`
    /// and its buffer (0,0) is that minus the popup's own geometry
    /// offset. Each popup is then clamped fully inside its parent's
    /// output (slide, never resize) so a menu opened near an edge — or
    /// a client requesting an off-screen / negative position — stays
    /// visible and as close to its intended spot as the output allows.
    pub(crate) fn snapshot_popup_placements(
        &self,
        placements: &[layout::Placement],
        layers: &[render::LayerPlacement],
    ) -> Vec<render::PopupPlacement> {
        use smithay::desktop::{PopupKind, PopupManager};
        use smithay::utils::{Physical, Point, Rectangle};

        // A popup parent: its root surface, window-geometry origin
        // (raw i32 so the Logical-tagged popup offsets from smithay
        // compose with our Physical-tagged compositor coords without a
        // unit cast), and the output to clamp its popups into.
        struct Parent {
            root: WlSurface,
            gx: i32,
            gy: i32,
            clamp: Option<Rectangle<i32, Physical>>,
        }

        let bw = self.layout.border_width();
        let outputs = self.renderer.output_rects();
        // Output rect containing `(x, y)`; falls back to the first
        // output so a popup whose anchor sits in an inter-output gap is
        // never dropped entirely.
        let output_for = |x: i32, y: i32| -> Option<Rectangle<i32, Physical>> {
            outputs
                .iter()
                .find(|(_, r)| {
                    x >= r.loc.x && x < r.loc.x + r.size.w && y >= r.loc.y && y < r.loc.y + r.size.h
                })
                .or_else(|| outputs.first())
                .map(|(_, r)| *r)
        };

        let mut parents: Vec<Parent> = Vec::new();
        for p in placements {
            let (gx, gy) = (p.cell_rect.loc.x + bw, p.cell_rect.loc.y + bw);
            parents.push(Parent {
                root: p.surface.clone(),
                gx,
                gy,
                clamp: output_for(gx, gy),
            });
        }
        for l in layers {
            let (gx, gy) = (l.rect.loc.x, l.rect.loc.y);
            parents.push(Parent {
                root: l.surface.clone(),
                gx,
                gy,
                clamp: output_for(gx, gy),
            });
        }

        let mut out = Vec::new();
        for Parent {
            root,
            gx,
            gy,
            clamp,
        } in &parents
        {
            for (popup, location) in PopupManager::popups_for_surface(root) {
                let PopupKind::Xdg(_) = popup else { continue };
                let surface = popup.wl_surface().clone();
                let pgeo = popup.geometry();
                // The popup's window-geometry rect within its buffer
                // (loc = buffer→visible offset, size = visible extent).
                // Some clients never call set_window_geometry on their
                // popups, so popup.geometry() is a zero
                // rect there; fall back to the actual committed surface
                // extent (loc AND size, per xdg-shell: an unset window
                // geometry is the full surface bounds). Without this the
                // hit rect is empty and clicks fall *through* the menu
                // (it just dismisses instead of activating the item).
                let (geo_loc, geo_size) = if pgeo.size.w > 0 && pgeo.size.h > 0 {
                    (pgeo.loc, pgeo.size)
                } else {
                    let bbox = smithay::desktop::utils::bbox_from_surface_tree(&surface, (0, 0));
                    (bbox.loc, bbox.size)
                };
                // Window-geometry top-left of the popup (the rect we
                // keep on-screen).
                let mut left = gx + location.x;
                let mut top = gy + location.y;
                let (w, h) = (geo_size.w, geo_size.h);
                if let Some(o) = clamp {
                    // Slide back on-screen; if the popup is larger than
                    // the output on an axis, the final `.max` pins the
                    // near (top/left) edge so the anchor stays visible.
                    if left + w > o.loc.x + o.size.w {
                        left = o.loc.x + o.size.w - w;
                    }
                    if top + h > o.loc.y + o.size.h {
                        top = o.loc.y + o.size.h - h;
                    }
                    left = left.max(o.loc.x);
                    top = top.max(o.loc.y);
                }
                out.push(render::PopupPlacement {
                    surface,
                    // Buffer (0,0) = clamped window-geometry top-left
                    // minus the geometry offset within the buffer.
                    buffer_origin: Point::new(left - geo_loc.x, top - geo_loc.y),
                    rect: Rectangle::new(Point::new(left, top), smithay::utils::Size::new(w, h)),
                });
            }
        }
        // Mapped X11 override-redirect windows (menus, tooltips,
        // dropdowns) ride the same pipeline: topmost, at the global
        // position the client chose. X apps position these against
        // their toplevel's geometry — which our configures keep in
        // sync — and constrain them on-screen themselves, so unlike
        // xdg popups there's nothing to clamp or offset (an X window
        // has no shadow-padded window geometry; buffer (0,0) is the
        // rect's top-left).
        for (window, surface) in &self.x11_or_windows {
            if !window.alive() {
                continue;
            }
            let geo = window.geometry();
            out.push(render::PopupPlacement {
                surface: surface.clone(),
                buffer_origin: Point::new(geo.loc.x, geo.loc.y),
                rect: Rectangle::new(
                    Point::new(geo.loc.x, geo.loc.y),
                    smithay::utils::Size::new(geo.size.w, geo.size.h),
                ),
            });
        }
        out
    }

    /// Topmost popup whose visible rect contains `pos`, paired with
    /// the logical point where that popup's surface buffer (0,0)
    /// sits, so the pointer path can forward motion / clicks into the
    /// popup's subsurface tree. Popups are hit-tested before layers
    /// and windows because they draw on top of both; iterating the
    /// snapshot in reverse yields nested submenus (pushed last) before
    /// their parents.
    pub(crate) fn popup_at(
        &self,
        pos: smithay::utils::Point<i32, smithay::utils::Physical>,
    ) -> Option<(
        WlSurface,
        smithay::utils::Point<f64, smithay::utils::Logical>,
    )> {
        // Rebuild the snapshot only when the scene may have changed since
        // it was built (see `scene_epoch`); consecutive motion events —
        // the hot caller — reuse it.
        let epoch = self.scene_epoch.get();
        if self.popup_snapshot.borrow().0 != epoch {
            let focused = self.seat.get_keyboard().and_then(|k| k.current_focus());
            let placements = self.layout.placements(focused.as_ref(), None);
            let layers = self.snapshot_layer_placements();
            let popups = self.snapshot_popup_placements(&placements, &layers);
            *self.popup_snapshot.borrow_mut() = (epoch, popups);
        }
        let snapshot = self.popup_snapshot.borrow();
        snapshot.1.iter().rev().find_map(|pp| {
            let r = pp.rect;
            let inside = r.size.w > 0
                && r.size.h > 0
                && pos.x >= r.loc.x
                && pos.x < r.loc.x + r.size.w
                && pos.y >= r.loc.y
                && pos.y < r.loc.y + r.size.h;
            inside.then(|| {
                (
                    pp.surface.clone(),
                    smithay::utils::Point::<f64, smithay::utils::Logical>::from((
                        f64::from(pp.buffer_origin.x),
                        f64::from(pp.buffer_origin.y),
                    )),
                )
            })
        })
    }

    /// Send `xdg_popup.popup_done` to every open popup (deepest child
    /// first, via `dismiss_popup`'s recursion) so a click outside any
    /// menu closes the whole chain. This is the pragmatic
    /// dismiss-on-click-outside path — we don't run a real popup grab
    /// yet. Errors (already-dead popups) are ignored.
    fn dismiss_all_popups(&mut self) {
        use smithay::desktop::PopupManager;
        let mut roots: Vec<WlSurface> = self
            .layout
            .placements(None, None)
            .into_iter()
            .map(|p| p.surface)
            .collect();
        roots.extend(
            self.layer_shell_state
                .layer_surfaces()
                .map(|l| l.wl_surface().clone()),
        );
        // Collect (root, popup) pairs before dismissing so we aren't
        // walking each tree while `dismiss_popup` mutates it.
        let mut pairs = Vec::new();
        for root in &roots {
            for (popup, _) in PopupManager::popups_for_surface(root) {
                pairs.push((root.clone(), popup));
            }
        }
        for (root, popup) in pairs {
            let _ = PopupManager::dismiss_popup(&root, &popup);
        }
        // The whole chain is gone — release the grab so hover focus resumes.
        self.popup_grab = None;
    }

    /// Drop the popup grab once its menu chain has closed (the client
    /// tore the popups down itself, e.g. after an item was clicked) so
    /// the hover focus model can move focus again. Cheap; called from the
    /// pointer-motion focus path.
    fn refresh_popup_grab(&mut self) {
        use smithay::desktop::PopupManager;
        if let Some(root) = &self.popup_grab {
            let still_open =
                root.alive() && PopupManager::popups_for_surface(root).next().is_some();
            if !still_open {
                self.popup_grab = None;
            }
        }
    }

    /// Constrain a popup's window geometry to fit on the output it opens
    /// on, honouring the client's `constraint_adjustment` (flip → slide →
    /// resize) per xdg-shell, and return geometry relative to the popup's
    /// parent surface's window geometry (ready to stamp into pending
    /// state and send to the client). Without this a menu near a screen
    /// edge is placed off-screen ("into the void"); telling the client
    /// the constrained geometry lets it flip submenus and render where it
    /// will actually be shown. Falls back to the raw positioner geometry
    /// when the parent window / output can't be located.
    ///
    /// Coordinates: the compositor works in logical units (the output
    /// rects are `mode / scale`, just `Physical`-tagged), which is the
    /// same space the client's positioner uses, so the arithmetic is a
    /// plain translate; we only rebuild the `target` as `Logical` to
    /// match the API's unit tag.
    pub(crate) fn unconstrain_popup_geometry(
        &self,
        surface: &smithay::wayland::shell::xdg::PopupSurface,
        positioner: &smithay::wayland::shell::xdg::PositionerState,
    ) -> smithay::utils::Rectangle<i32, Logical> {
        use smithay::desktop::{PopupKind, PopupManager, find_popup_root_surface};
        use smithay::utils::{Rectangle, Size};

        let Ok(root) = find_popup_root_surface(&PopupKind::Xdg(surface.clone())) else {
            return positioner.get_geometry();
        };
        // Root window-geometry origin in compositor space: a tiled/floating
        // toplevel's cell origin + border, or a layer surface's rect origin.
        let bw = self.layout.border_width();
        let root_origin = self
            .layout
            .placements(None, None)
            .into_iter()
            .find(|p| p.surface == root)
            .map(|p| (p.cell_rect.loc.x + bw, p.cell_rect.loc.y + bw))
            .or_else(|| {
                self.snapshot_layer_placements()
                    .into_iter()
                    .find(|l| l.surface == root)
                    .map(|l| (l.rect.loc.x, l.rect.loc.y))
            });
        let Some((rgx, rgy)) = root_origin else {
            return positioner.get_geometry();
        };
        // Output to fit into: the one containing the root origin, falling
        // back to the first so a root in an inter-output gap still gets
        // constrained somewhere on-screen.
        let outputs = self.renderer.output_rects();
        let Some((_, out)) = outputs
            .iter()
            .find(|(_, r)| {
                rgx >= r.loc.x && rgx < r.loc.x + r.size.w && rgy >= r.loc.y && rgy < r.loc.y + r.size.h
            })
            .or_else(|| outputs.first())
        else {
            return positioner.get_geometry();
        };
        // The positioner's anchor rect is relative to the *immediate*
        // parent's window geometry: the root itself for a top-level popup,
        // or the parent popup's on-screen geometry top-left for a submenu
        // (its location from the manager is relative to the root geometry).
        let (pgx, pgy) = match surface.get_parent_surface() {
            Some(parent) if parent != root => PopupManager::popups_for_surface(&root)
                .find(|(p, _)| p.wl_surface() == &parent)
                .map_or((rgx, rgy), |(_, loc)| (rgx + loc.x, rgy + loc.y)),
            _ => (rgx, rgy),
        };
        let target = Rectangle::<i32, Logical>::new(
            Point::from((out.loc.x - pgx, out.loc.y - pgy)),
            Size::from((out.size.w, out.size.h)),
        );
        (*positioner).get_unconstrained_geometry(target)
    }

    /// Walk the layer-surface list in top-down z-order (`Overlay`
    /// first, then `Top`, then `Bottom`, then `Background`) and
    /// return the first one whose rect contains `pos`, plus its
    /// rect. Used by the pointer paths so mouse motion over rofi
    /// / waybar / etc. reaches them.
    pub(crate) fn layer_at(
        &self,
        pos: smithay::utils::Point<i32, smithay::utils::Physical>,
    ) -> Option<(
        WlSurface,
        smithay::utils::Rectangle<i32, smithay::utils::Physical>,
    )> {
        fn priority(bucket: render::LayerBucket) -> u8 {
            match bucket {
                render::LayerBucket::Overlay => 3,
                render::LayerBucket::Top => 2,
                render::LayerBucket::Bottom => 1,
                render::LayerBucket::Background => 0,
            }
        }
        let mut best: Option<(render::LayerPlacement, u8)> = None;
        for placement in self.snapshot_layer_placements() {
            let r = placement.rect;
            let inside = r.size.w > 0
                && r.size.h > 0
                && pos.x >= r.loc.x
                && pos.x < r.loc.x + r.size.w
                && pos.y >= r.loc.y
                && pos.y < r.loc.y + r.size.h;
            if !inside {
                continue;
            }
            // A fullscreen window sits above the Top/Bottom/Background layers
            // on its output (the render path draws `Fullscreen` over them — a
            // fullscreen game/video hides the bar), so the pointer must fall
            // through those now-invisible panels to the window. Otherwise a
            // click over the hidden bar would land on the bar, not the game.
            // `Overlay` stays above fullscreen (launcher / toasts / OSDs remain
            // interactive), and maximized windows draw below Top, so neither is
            // occluded here.
            if !matches!(placement.layer, render::LayerBucket::Overlay)
                && let Some(name) = self.layer_outputs.get(&placement.surface)
                && self.layout.output_has_fullscreen(name)
            {
                continue;
            }
            // Respect the surface's input region: a layer that committed an
            // empty/partial input region — a click-through fullscreen overlay
            // (an idle launcher), or a toast strip masked to just its cards —
            // must NOT capture the pointer for the parts it doesn't claim,
            // even though the cursor is within its rect. Compositor coords are
            // logical-scale, so the Physical-tagged values map straight to
            // Logical here.
            let point = smithay::utils::Point::<f64, smithay::utils::Logical>::from((
                f64::from(pos.x),
                f64::from(pos.y),
            ));
            let location =
                smithay::utils::Point::<i32, smithay::utils::Logical>::from((r.loc.x, r.loc.y));
            if smithay::desktop::utils::under_from_surface_tree(
                &placement.surface,
                point,
                location,
                smithay::desktop::WindowSurfaceType::ALL,
            )
            .is_none()
            {
                continue;
            }
            let p = priority(placement.layer);
            if best.as_ref().is_none_or(|(_, bp)| *bp < p) {
                best = Some((placement, p));
            }
        }
        best.map(|(p, _)| (p.surface, p.rect))
    }

    /// `true` when the keyboard's current focus is a layer surface
    /// that requested `KeyboardInteractivity::Exclusive` on the
    /// `Top` or `Overlay` layer. While this holds, pointer-driven
    /// focus changes (hover or click) must be suppressed so the
    /// modal layer keeps the keyboard — otherwise the hover focus
    /// model would yank focus back the moment the user moved the
    /// mouse off the layer surface.
    /// Any *other* still-mapped exclusive Top/Overlay layer surface
    /// (skipping `exclude`) — i.e. one that should hold the keyboard.
    /// Used when the focused exclusive layer is destroyed to hand the
    /// keyboard to a live sibling (e.g. slurp's other-monitor overlay)
    /// rather than dropping focus back to a window mid-selection.
    pub(crate) fn first_exclusive_layer_surface(&self, exclude: &WlSurface) -> Option<WlSurface> {
        use smithay::reexports::wayland_server::Resource;
        use smithay::wayland::shell::wlr_layer::{KeyboardInteractivity, Layer};
        self.layer_shell_state.layer_surfaces().find_map(|ls| {
            let surface = ls.wl_surface();
            if surface == exclude || !surface.is_alive() {
                return None;
            }
            let cached = crate::wayland::layer_cached_state(surface);
            (matches!(
                cached.keyboard_interactivity,
                KeyboardInteractivity::Exclusive
            ) && matches!(cached.layer, Layer::Top | Layer::Overlay))
            .then(|| surface.clone())
        })
    }

    pub(crate) fn focus_locked_by_layer(&self) -> bool {
        use smithay::wayland::shell::wlr_layer::{KeyboardInteractivity, Layer};
        let Some(kbd) = self.seat.get_keyboard() else {
            return false;
        };
        let Some(focus) = kbd.current_focus() else {
            return false;
        };
        for layer_surface in self.layer_shell_state.layer_surfaces() {
            if layer_surface.wl_surface() == &focus {
                let cached = crate::wayland::layer_cached_state(&focus);
                return matches!(
                    cached.keyboard_interactivity,
                    KeyboardInteractivity::Exclusive
                ) && matches!(cached.layer, Layer::Top | Layer::Overlay);
            }
        }
        false
    }

    /// Walk every known layer surface, sum its exclusive zones
    /// per anchored edge (`top`/`bottom`/`left`/`right`), and
    /// shrink the layout's bounds by the totals. Called whenever
    /// a layer surface is added, destroyed, or commits a new
    /// state — anything that might change a `Layer::Top` /
    /// `Bottom` reservation. Background and overlay layers are
    /// rendered, but exclusive zones from those layers are not
    /// honoured per protocol (overlay floats on top; background
    /// always renders below).
    pub(crate) fn recompute_layer_layout(&mut self) {
        use smithay::wayland::shell::wlr_layer::Anchor;
        // Exclusive zones (top, bottom, left, right) accumulated per
        // output the reserving surface is bound to — so a bar on the
        // secondary monitor shrinks *that* monitor's tile area, not the
        // primary's.
        // No outputs (every monitor unplugged) — nothing to lay out.
        let Some(primary_name) = self.renderer.primary_output_name().map(str::to_owned) else {
            return;
        };
        let mut zones: std::collections::HashMap<String, (i32, i32, i32, i32)> =
            std::collections::HashMap::new();
        for layer in self.layer_shell_state.layer_surfaces() {
            let cached = crate::wayland::layer_cached_state(layer.wl_surface());
            let exclusive: i32 = cached.exclusive_zone.into();
            if exclusive <= 0 {
                continue;
            }
            // Per spec: exclusive is meaningful only when anchored
            // to one edge or to one edge + the two perpendicular
            // ones. We approximate by attributing the zone to
            // whichever single edge is anchored without its
            // opposite.
            let anchor = cached.anchor;
            let at_top = anchor.contains(Anchor::TOP);
            let at_bottom = anchor.contains(Anchor::BOTTOM);
            let at_left = anchor.contains(Anchor::LEFT);
            let at_right = anchor.contains(Anchor::RIGHT);
            let out_name = self
                .layer_outputs
                .get(layer.wl_surface())
                .cloned()
                .unwrap_or_else(|| primary_name.clone());
            let zone = zones.entry(out_name).or_insert((0, 0, 0, 0));
            if at_top && !at_bottom {
                zone.0 = zone.0.max(exclusive);
            } else if at_bottom && !at_top {
                zone.1 = zone.1.max(exclusive);
            } else if at_left && !at_right {
                zone.2 = zone.2.max(exclusive);
            } else if at_right && !at_left {
                zone.3 = zone.3.max(exclusive);
            }
        }
        for (name, rect) in self.renderer.output_rects() {
            let (top, bottom, left, right) = zones.get(&name).copied().unwrap_or((0, 0, 0, 0));
            let new_bounds = smithay::utils::Rectangle::<i32, smithay::utils::Physical>::new(
                smithay::utils::Point::new(rect.loc.x + left, rect.loc.y + top),
                smithay::utils::Size::new(
                    (rect.size.w - left - right).max(1),
                    (rect.size.h - top - bottom).max(1),
                ),
            );
            // `rect` is the full output; `new_bounds` is the work area
            // (full minus exclusive zones). Fullscreen fills the
            // former, tiling/maximized the latter.
            self.layout.set_output_bounds(&name, rect, new_bounds);
        }
    }

    /// Reconcile the output set after a udev "device changed" event —
    /// the hotplug entry point. Re-scans the DRM device's connectors,
    /// then brings up any newly-connected monitor (a fresh CRTC +
    /// swapchain + `wl_output` global, packed to the right of the
    /// existing ones) and tears down any that vanished (its windows are
    /// moved onto a surviving output so their clients live on). A change
    /// that doesn't alter our output set is a cheap no-op.
    pub(crate) fn handle_drm_changed(&mut self) {
        let existing: std::collections::HashSet<String> =
            self.renderer.output_names().into_iter().collect();
        let used_crtcs: std::collections::HashSet<crtc::Handle> =
            self.renderer.crtcs().into_iter().collect();
        let rescan = match crate::drm::rescan_connectors(
            &mut self.drm_device,
            &self.config.monitors,
            &existing,
            &used_crtcs,
        ) {
            Ok(r) => r,
            Err(err) => {
                warn!(error = %err, "connector rescan failed — leaving outputs unchanged");
                return;
            }
        };
        let connected: std::collections::HashSet<String> =
            rescan.connected.into_iter().collect();
        let removed: Vec<String> = existing
            .iter()
            .filter(|n| !connected.contains(*n))
            .cloned()
            .collect();
        if removed.is_empty() && rescan.added.is_empty() {
            // EDID refresh or a change on a card we don't drive: nothing to do.
            return;
        }

        // --- Disconnects: evacuate windows, then drop the output. ---
        for name in &removed {
            // Move this output's windows onto any surviving output (one
            // that isn't itself being removed). `None` ⇒ it was the last
            // output; the layout keeps the windows parked for its return.
            let fallback = self
                .renderer
                .output_names()
                .into_iter()
                .find(|n| n != name && !removed.contains(n));
            self.layout.remove_output(name, fallback.as_deref());
            if let Some(crtc) = self.renderer.crtc_for_output_name(name) {
                self.renderer.remove_output(crtc);
                self.redraw.remove(&crtc);
            }
            if let Some(global) = self.output_globals.remove(name) {
                self.display_handle.remove_global::<State>(global);
            }
            self.outputs.retain(|o| o.name() != *name);
            self.lock_surfaces.remove(name);
            self.layer_outputs.retain(|_, v| v != name);
            info!(output = %name, "output disconnected");
        }

        // --- Connects: build the swapchain, register the layout pane. ---
        for drm_output in rescan.added {
            let name = drm_output.name.clone();
            let crtc = drm_output.crtc;
            match self.renderer.add_output(drm_output, &self.config.monitors) {
                Ok(()) => {
                    // Provisional rect; `reflow_outputs` + `recompute_layer_layout`
                    // below set the real position and work area.
                    let rect = self.renderer.output_rect(&name).unwrap_or_default();
                    self.layout.add_output(name.clone(), rect);
                    self.redraw.insert(crtc, RedrawState::Idle);
                    info!(output = %name, "output connected");
                }
                Err(err) => {
                    warn!(output = %name, error = %err, "failed to bring up hot-plugged output");
                }
            }
        }

        // --- Re-pack positions, re-advertise to clients, re-tile, repaint. ---
        let descs = self.renderer.reflow_outputs(&self.config.monitors);
        self.sync_output_globals(&descs);
        self.recompute_layer_layout();
        self.preferred_scale = self.renderer.primary_scale();
        // A hotplug can change which output is primary (and so its
        // scale) — keep Xwayland's client scale + XSETTINGS DPI in step.
        self.update_xwayland_scale();
        self.queue_redraw_all();
    }

    /// Reconcile the `wl_output` globals with the renderer's reflowed
    /// output set: update each surviving output's advertised
    /// position/mode/scale (a neighbour vanishing shifts everyone left),
    /// and create + advertise a global for any output that's new since
    /// the last sync. Removal of vanished globals happens in
    /// [`Self::handle_drm_changed`] before this runs.
    fn sync_output_globals(&mut self, descs: &[render::OutputDescriptor]) {
        use smithay::output::{Mode as OutputMode, Scale};
        use smithay::utils::{Logical, Point, Transform};
        for desc in descs {
            if let Some(output) = self.outputs.iter().find(|o| o.name() == desc.name) {
                let mode = OutputMode {
                    size: desc.mode_size,
                    refresh: desc.refresh_mhz,
                };
                output.change_current_state(
                    Some(mode),
                    Some(Transform::Normal),
                    Some(Scale::Fractional(desc.scale)),
                    Some(Point::<i32, Logical>::from((
                        desc.compositor_position.x,
                        desc.compositor_position.y,
                    ))),
                );
            } else {
                let output = crate::wayland::make_output(desc);
                let global = output.create_global::<State>(&self.display_handle);
                self.output_globals.insert(desc.name.clone(), global);
                self.outputs.push(output);
            }
        }
    }

    /// Run a bound action. Grows as we add more actions (`reload`,
    /// `spawn`, `change_vt`, …).
    fn dispatch_action(&mut self, action: config::Action) {
        // Most actions change what's on screen (fullscreen/maximize/float
        // toggles, workspace switches, screenshots). A few (spawn, exit)
        // don't, but a redundant redraw is harmless and coalesces away.
        self.queue_redraw_all();
        match action {
            config::Action::Exit => {
                info!("exit action fired — stopping event loop");
                self.loop_signal.stop();
            }
            config::Action::ToggleFloating => {
                // Toggle the currently keyboard-focused surface.
                // No focus = nothing to toggle, silent no-op.
                let focus = self.seat.get_keyboard().and_then(|k| k.current_focus());
                if let Some(surface) = focus {
                    info!(surface = ?surface.id(), "togglefloating action fired");
                    self.layout.toggle_floating(&surface);
                }
            }
            config::Action::ToggleFullscreen => {
                // Flip the keyboard-focused window in/out of fullscreen.
                // No focus = nothing to toggle, silent no-op.
                let focus = self.seat.get_keyboard().and_then(|k| k.current_focus());
                if let Some(surface) = focus {
                    info!(surface = ?surface.id(), "togglefullscreen action fired");
                    self.layout.toggle_fullscreen(&surface);
                }
            }
            config::Action::Close => {
                // Politely ask the focused window to close — via
                // xdg_toplevel.close or WM_DELETE_WINDOW, whichever
                // protocol it speaks; the client drives its own
                // teardown (which unmaps/destroys the surface, and the
                // shell handlers remove it from the layout). No focus /
                // no matching window (e.g. a layer surface like rofi is
                // focused) = no-op.
                let focus = self.seat.get_keyboard().and_then(|k| k.current_focus());
                if let Some(surface) = focus
                    && let Some(handle) = self.layout.window_surface(&surface)
                {
                    info!(surface = ?surface.id(), "close action fired");
                    handle.send_close();
                }
            }
            config::Action::Spawn(cmd) => {
                // Runs at bind-press time. `build_command` whitespace-splits
                // into program + args and layers the live `env` + X
                // `$DISPLAY` over the inherited process env (which carries
                // `$WAYLAND_DISPLAY`). Empty commands and failures are logged
                // and the loop keeps running.
                let Some(mut command) = self.build_command(&cmd) else {
                    warn!(command = %cmd, "spawn action: empty command");
                    return;
                };
                match command.spawn() {
                    Ok(child) => {
                        info!(
                            pid = child.id(),
                            command = %cmd,
                            "spawn action fired"
                        );
                        self.children.push(child);
                    }
                    Err(err) => warn!(
                        error = %err,
                        command = %cmd,
                        "spawn action failed"
                    ),
                }
            }
            config::Action::Screenshot(bind) => self.start_screenshot(&bind),
        }
    }

    /// Build a child [`std::process::Command`] from a whitespace-split
    /// command line, applying the configured `env` overrides and the
    /// live X `$DISPLAY` so spawned clients inherit the *current*
    /// environment. Returns `None` for an empty command line.
    ///
    /// We apply the environment per-child here rather than mutating the
    /// process environment: `std::env::set_var` is `unsafe` and unsound
    /// once worker threads are running (the media decoder), so a live
    /// `env`/`xwayland` reload must not touch the process env — it only
    /// changes what we pass to children spawned from now on. Already-
    /// running clients and compositor-consumed vars (e.g. `XCURSOR_*`)
    /// still need a restart.
    fn build_command(&self, raw: impl AsRef<str>) -> Option<std::process::Command> {
        let raw = raw.as_ref();
        let parts: Vec<&str> = raw.split_whitespace().collect();
        let (program, args) = parts.split_first()?;
        let mut cmd = std::process::Command::new(program);
        cmd.args(args);
        self.apply_child_env(&mut cmd);
        Some(cmd)
    }

    /// Apply the configured `env` overrides and the live X `$DISPLAY` to a
    /// child `cmd`. Centralised so every runtime spawn path (bind spawn,
    /// idle lock, IPC spawn) picks up an `env`/`xwayland` reload — we
    /// can't mutate the process env at runtime (`set_var` is unsound with
    /// the media worker threads running), so the current environment is
    /// applied per-child instead.
    pub(crate) fn apply_child_env(&self, cmd: &mut std::process::Command) {
        cmd.envs(self.config.env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
        // Export (or hide) `$DISPLAY` to match the live xwayland state.
        match &self.xwayland_display {
            Some(disp) => {
                cmd.env("DISPLAY", disp);
            }
            None => {
                cmd.env_remove("DISPLAY");
            }
        }
    }

    /// Apply changed input settings on reload: key repeat rate/delay,
    /// keyboard layout (both the seat keymap clients receive and our own
    /// hotkey-matching xkb state), and mouse acceleration (re-applied to
    /// Push the current lock-key LED state to every keyboard device.
    /// Called when xkb lock state changes (`led_state_changed`) and for
    /// hot-plugged keyboards. Without this the physical Caps/Num Lock
    /// lights never change — xkb tracks the state, but something must
    /// hand it to libinput.
    pub(crate) fn apply_keyboard_leds(&mut self) {
        for device in &mut self.input_devices {
            if device.has_capability(smithay::reexports::input::DeviceCapability::Keyboard) {
                device.led_update(self.keyboard_leds.into());
            }
        }
    }

    /// Set (or clear) the xkb *locked* Num Lock modifier — the
    /// `input.numlock` option. Locked-modifier state means clients see
    /// the modifier and the LED path above lights the keyboards.
    fn set_numlock(&mut self, engage: bool) {
        let Some(kbd) = self.seat.get_keyboard() else {
            return;
        };
        let applied = kbd.with_xkb_state(self, |mut ctx| {
            ctx.set_lock_modifier(xkbcommon::xkb::MOD_NAME_NUM, engage)
        });
        if applied {
            info!(engage, "input: numlock lock state applied");
        } else {
            warn!("input: keymap has no Num Lock modifier; numlock option ignored");
        }
    }

    /// every live libinput device). The `*_changed` flags are computed by
    /// the caller against the old config before it's swapped.
    fn apply_input_reload(
        &mut self,
        new_input: &config::InputConfig,
        repeat_changed: bool,
        layout_changed: bool,
        accel_changed: bool,
    ) {
        if repeat_changed && let Some(kbd) = self.seat.get_keyboard() {
            let rate = i32::try_from(new_input.repeat_rate).unwrap_or(i32::MAX);
            let delay = i32::try_from(new_input.repeat_delay).unwrap_or(i32::MAX);
            kbd.change_repeat_info(rate, delay);
            info!(rate, delay, "input: applied new key repeat");
        }
        if layout_changed {
            // Seat keymap (what clients receive) …
            if let Some(kbd) = self.seat.get_keyboard() {
                let xkb = smithay::input::keyboard::XkbConfig {
                    layout: &new_input.keyboard_layout,
                    ..smithay::input::keyboard::XkbConfig::default()
                };
                if let Err(err) = kbd.set_xkb_config(self, xkb) {
                    warn!(?err, "input: applying new keyboard layout to the seat failed");
                }
            }
            // … and our own xkb wrapper used for hotkey matching.
            match keyboard::Keyboard::new(&new_input.keyboard_layout) {
                Ok(k) => {
                    self.keyboard = k;
                    info!(layout = %new_input.keyboard_layout, "input: applied new keyboard layout");
                }
                Err(err) => warn!(
                    error = %err,
                    "input: new keyboard layout failed to compile; keeping the old one for hotkeys"
                ),
            }
        }
        if accel_changed {
            for device in &mut self.input_devices {
                apply_input_config(device, new_input);
            }
            info!("input: re-applied mouse acceleration to live devices");
        }
    }

    /// Point per-surface dmabuf feedback at the right variant for every
    /// window visible on `out_name`: fullscreen windows get the scanout
    /// feedback (a Scanout-flagged tranche of the primary plane's explicit
    /// modifiers — what makes a Vulkan/EGL swapchain re-allocate into
    /// plane-scannable buffers instead of implicit render-optimal ones),
    /// everything else the default. This is the per-surface half of the
    /// dmabuf-feedback protocol: it only reaches clients that called
    /// `get_surface_feedback` (game WSIs do; plain EGL toolkits use the
    /// default feedback and never see a tranche — putting it in the
    /// default broke them, see the revert of fe142c7).
    ///
    /// Keyed off the *fill mode*, not per-frame scanout eligibility, so a
    /// menu opening over the game doesn't flap the feedback and force
    /// swapchain rebuilds. smithay dedupes internally (`set_feedback`
    /// no-ops when unchanged), so per-frame calls cost a short tree walk.
    fn sync_scanout_feedback(&self, placements: &[layout::Placement], out_name: Option<&str>) {
        let (Some(default_fb), Some(scanout_fb)) = (
            self.dmabuf_default_feedback.as_ref(),
            self.dmabuf_scanout_feedback.as_ref(),
        ) else {
            return;
        };
        let Some(name) = out_name else { return };
        let Some(output) = self.outputs.iter().find(|o| o.name() == name) else {
            return;
        };
        let Some(rect) = self.renderer.output_rect(name) else {
            return;
        };
        for p in placements.iter().filter(|p| p.cell_rect.overlaps(rect)) {
            // Sticky: once a window has been given the scanout variant it
            // keeps it for its lifetime. Every feedback change makes the
            // client's WSI rebuild its swapchain, and swapchain recreation
            // is exactly the moment fragile frame-pacing code runs (a Wine
            // build with a broken present-timing thunk asserts there) — so
            // never flip back on unfullscreen. Keeping the tranche is
            // harmless: scanout-capable modifiers render fine windowed.
            let fullscreen = p.fill == layout::FillMode::Fullscreen;
            if fullscreen {
                self.scanout_feedback_given
                    .borrow_mut()
                    .insert(p.surface.id());
            }
            let feedback = if fullscreen
                || self.scanout_feedback_given.borrow().contains(&p.surface.id())
            {
                scanout_fb
            } else {
                default_fb
            };
            smithay::desktop::utils::send_dmabuf_feedback_surface_tree(
                &p.surface,
                output,
                |_, _| Some(output.clone()),
                |_, _| feedback,
            );
        }
    }

    /// Reap exited children so they don't accumulate as zombies. Every
    /// runtime spawn path parks its `Child` handle in `self.children`;
    /// this sweeps them with `try_wait` (never blocks) on a timer.
    /// (Xwayland isn't in this set — smithay owns its child handle, and
    /// a dying Xwayland surfaces as `XwmHandler::disconnected`.)
    pub(crate) fn reap_children(&mut self) {
        self.children.retain_mut(|child| match child.try_wait() {
            Ok(Some(status)) => {
                debug!(pid = child.id(), %status, "reaped exited child");
                false
            }
            Ok(None) => true,
            Err(err) => {
                warn!(pid = child.id(), error = %err, "reaping child failed; dropping the handle");
                false
            }
        });
    }

    /// Start or stop Xwayland to match a live `xwayland` toggle.
    /// Enabling spawns a fresh server and records the display (which
    /// [`State::build_command`] exports as `$DISPLAY` to children we
    /// spawn). Disabling tears the server down — any X11 clients
    /// connected to it lose their server.
    fn apply_xwayland_toggle(&mut self, enable: bool) {
        if enable {
            let dh = self.display_handle.clone();
            let scale = self.renderer.primary_scale();
            if let Some(spawned) = crate::xwayland::spawn_xwayland(&self.loop_handle, &dh, scale) {
                info!(
                    x_display = %spawned.display,
                    "xwayland enabled; children spawned from now get $DISPLAY"
                );
                self.xwayland_source = Some(spawned.source);
                self.xwayland_client = Some(spawned.client);
                self.xwayland_display = Some(spawned.display);
            } else {
                warn!("xwayland enabled but Xwayland failed to start");
            }
        } else {
            self.teardown_xwayland();
            warn!("xwayland disabled (running X11 clients lost their server)");
        }
    }

    /// Apply a live mode change to one output: drop its DRM surface (which
    /// frees its CRTC), then modeset a fresh surface on the *same*
    /// connector/CRTC with the new mode and re-add it. The layout pane —
    /// and therefore the windows on it — is left untouched, so a mode
    /// change doesn't disturb window placement. On failure the output is
    /// left down (logged); a replug or restart recovers it.
    fn change_output_mode(&mut self, name: &str, monitors: &config::MonitorsConfig) {
        let Some((connector, crtc)) = self.renderer.output_connector_crtc(name) else {
            return;
        };
        self.renderer.remove_output(crtc);
        self.redraw.remove(&crtc);
        match crate::drm::rebuild_output_mode(&mut self.drm_device, connector, crtc, monitors) {
            Ok(drm_output) => match self.renderer.add_output(drm_output, monitors) {
                Ok(()) => {
                    // Prime the CRTC so the next `queue_redraw_all` flips it.
                    self.redraw.insert(crtc, RedrawState::Idle);
                    info!(output = %name, "applied live mode change (modeset)");
                }
                Err(err) => warn!(
                    output = %name, error = %err,
                    "mode change: re-adding output failed; output down until replug/restart"
                ),
            },
            Err(err) => warn!(
                output = %name, error = %err,
                "mode change: rebuilding DRM surface failed; output down until replug/restart"
            ),
        }
    }

    /// Re-read the config file and apply every setting that can change at
    /// runtime. A parse/validation error keeps the running config
    /// untouched (logged, never fatal) so the user can fix and save to
    /// recover. The few settings that genuinely can't hot-apply (startup
    /// commands are one-shot; `env`/`XCURSOR_*` consumed by the
    /// compositor itself) are logged rather than applied.
    pub(crate) fn reload_config(&mut self, path: &std::path::Path) {
        let new = match config::Config::load_from_file(path) {
            Ok(new) => new,
            Err(err) => {
                warn!(error = %err, "config reload failed; keeping the running config");
                return;
            }
        };

        // ---- Detect what changed (before any mutation). ----
        let monitors_changed = new.monitors != self.config.monitors;
        // Outputs whose forced-mode override OR HDR toggle changed need a
        // real DRM surface rebuild (modeset / new buffer format); reflow
        // alone only re-positions/-scales/-VRRs. The HDR scanout format is
        // fixed at swapchain creation, so toggling it means recreating the
        // surface — the same path a mode change takes.
        let rebuild_outputs: Vec<String> = if monitors_changed {
            self.renderer
                .output_names()
                .into_iter()
                .filter(|name| {
                    let old = self.config.monitors.outputs.get(name);
                    let new_o = new.monitors.outputs.get(name);
                    let old_mode = old.and_then(|c| c.mode);
                    let new_mode = new_o.and_then(|c| c.mode);
                    let old_hdr = old.is_some_and(|c| c.hdr);
                    let new_hdr = new_o.is_some_and(|c| c.hdr);
                    old_mode != new_mode || old_hdr != new_hdr
                })
                .collect()
        } else {
            Vec::new()
        };
        let repeat_changed = new.input.repeat_rate != self.config.input.repeat_rate
            || new.input.repeat_delay != self.config.input.repeat_delay;
        let layout_changed = new.input.keyboard_layout != self.config.input.keyboard_layout;
        #[allow(
            clippy::float_cmp,
            reason = "exact change detection — did the configured accel speed differ at all, not 'approximately equal'"
        )]
        let accel_changed = new.input.mouse_accel_profile != self.config.input.mouse_accel_profile
            || new.input.mouse_accel_speed != self.config.input.mouse_accel_speed;
        let numlock_changed = new.input.numlock != self.config.input.numlock;
        let xwayland_changed = new.xwayland != self.config.xwayland;
        if new.env != self.config.env {
            info!("env changed; applies to children spawned from now on (restart for XCURSOR_* etc.)");
        }
        if new.startup != self.config.startup {
            info!("startup commands changed; they only run at launch");
        }

        // ---- Input: key repeat, keymap, mouse acceleration. ----
        self.apply_input_reload(&new.input, repeat_changed, layout_changed, accel_changed);
        // A *changed* numlock option applies in either direction (an
        // unchanged one never fights the user's own toggling).
        if numlock_changed {
            self.set_numlock(new.input.numlock);
        }

        // ---- Monitors: modeset changed modes, then reflow the rest. ----
        if monitors_changed {
            for name in &rebuild_outputs {
                self.change_output_mode(name, &new.monitors);
            }
            let descs = self.renderer.reflow_outputs(&new.monitors);
            self.sync_output_globals(&descs);
            self.recompute_layer_layout();
            self.preferred_scale = self.renderer.primary_scale();
            // Xwayland's client scale + XSETTINGS DPI follow the primary
            // output's scale; re-publish and re-push X window configures
            // so X apps re-map into the new pixel space.
            self.update_xwayland_scale();
            info!("monitor config reloaded (position/scale/primary/vrr/mode/hdr)");
        }

        // ---- XWayland: start or stop the server to match the toggle. ----
        if xwayland_changed {
            self.apply_xwayland_toggle(new.xwayland);
        }

        // ---- Appearance. Layout FIRST (it reflows and sends new
        // configures to clients), the renderer LAST: the renderer's
        // border width drives where it draws the surface and the border
        // ring, so changing it only after clients have been asked to
        // resize avoids a one-frame window where a new border is drawn
        // around an old-sized buffer. Binds and focus model are read live
        // from `self.config`, so swapping it suffices. ----
        self.layout.set_appearance(
            layout::Gaps {
                outer: new.layout.gaps_outer,
                inner: new.layout.gaps_inner,
            },
            new.border.width,
        );
        apply_wallpaper(&mut self.renderer, &new.misc.wallpaper, &new.border);
        self.renderer.set_animations(new.animations.clone());
        self.renderer.set_decoration(new.decoration.clone());
        self.config = new;
        info!("config reloaded");
        self.queue_redraw_all();
    }
}

/// Apply `wallpaper` to the renderer: either a flat fill, or a
/// libav-decoded media frame (image/gif/video, fitted per `mode`). On a
/// decode/upload failure the media is dropped and a neutral fallback fill
/// shows instead of a black screen. Also (re)applies `border`. Shared by
/// startup and live reload.
fn apply_wallpaper(
    renderer: &mut render::Renderer,
    wallpaper: &config::Wallpaper,
    border: &config::BorderConfig,
) {
    match wallpaper {
        config::Wallpaper::Fill(fill) => {
            renderer.set_appearance(fill.clone(), border.clone());
            renderer.set_wallpaper_media(None);
        }
        config::Wallpaper::Media { path, mode } => {
            // Cap the decode to the largest output dimension so a huge
            // source doesn't allocate an oversized texture; the GPU scales
            // per output from there.
            #[allow(clippy::cast_sign_loss, reason = "output dimensions are positive")]
            let cap = renderer
                .output_descriptors()
                .iter()
                .map(|d| d.mode_size.w.max(d.mode_size.h))
                .max()
                .unwrap_or(3840)
                .max(1) as u32;
            match media::decode_first_frame(path, cap) {
                Ok(frame) => {
                    #[allow(
                        clippy::cast_possible_wrap,
                        reason = "decoded dims are capped to output size, well within i32"
                    )]
                    let (w, h) = (frame.width as i32, frame.height as i32);
                    // The decode thread loops video/gif and self-terminates
                    // for a still image; it shares the slot the renderer
                    // polls for new frames.
                    let anim = media::Animation::start(path, cap);
                    if renderer.set_wallpaper_media(Some((&frame.rgba, w, h, *mode, anim))) {
                        // The rounded-corner shader can't sample media, so
                        // the corner cutout falls back to black.
                        renderer
                            .set_appearance(config::Fill::Solid([0.0, 0.0, 0.0]), border.clone());
                        info!(
                            path = %path.display(),
                            width = frame.width,
                            height = frame.height,
                            "media wallpaper loaded"
                        );
                    } else {
                        renderer.set_appearance(wallpaper_fallback(), border.clone());
                    }
                }
                Err(err) => {
                    warn!(error = %err, path = %path.display(), "media wallpaper failed; flat background");
                    renderer.set_wallpaper_media(None);
                    renderer.set_appearance(wallpaper_fallback(), border.clone());
                }
            }
        }
    }
}

/// A neutral flat fill shown when a media wallpaper can't be loaded, so
/// the screen isn't left black.
fn wallpaper_fallback() -> config::Fill {
    config::Fill::Solid([0.08, 0.08, 0.10])
}

/// Axis-aligned rect (absolute compositor px) spanning two cursor
/// corners, regardless of drag direction.
fn rect_from_corners(ax: f64, ay: f64, bx: f64, by: f64) -> smithay::utils::Rectangle<i32, Physical> {
    let x0 = ax.min(bx);
    let y0 = ay.min(by);
    let w = ax.max(bx) - x0;
    let h = ay.max(by) - y0;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "cursor coordinates are clamped to the i32 layout bounds in on_pointer_motion"
    )]
    smithay::utils::Rectangle::new(
        Point::from((x0.round() as i32, y0.round() as i32)),
        smithay::utils::Size::from((w.round() as i32, h.round() as i32)),
    )
}

/// Map an absolute-compositor rect into one output's framebuffer
/// (physical) pixels: subtract the output origin, scale by its factor.
fn compositor_rect_to_physical(
    rect: smithay::utils::Rectangle<i32, Physical>,
    origin: Point<i32, Physical>,
    scale: f64,
) -> smithay::utils::Rectangle<i32, Physical> {
    let lx = f64::from(rect.loc.x - origin.x) * scale;
    let ly = f64::from(rect.loc.y - origin.y) * scale;
    let w = f64::from(rect.size.w) * scale;
    let h = f64::from(rect.size.h) * scale;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "physical dimensions are bounded by the output's framebuffer size, well within i32"
    )]
    smithay::utils::Rectangle::new(
        Point::from((lx.round() as i32, ly.round() as i32)),
        smithay::utils::Size::from((w.round() as i32, h.round() as i32)),
    )
}

impl State {
    /// Cursor hotspot as integer physical/compositor pixels.
    fn cursor_point(&self) -> Point<i32, Physical> {
        let (cx, cy) = self.renderer.cursor_pos();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "cursor is clamped to the i32 layout bounds in Renderer::on_pointer_motion"
        )]
        Point::from((cx.round() as i32, cy.round() as i32))
    }

    /// Enter a screenshot session (Region/Window) or, for Output mode,
    /// immediately enqueue a full-output grab. Re-triggers while a session
    /// runs are ignored.
    fn start_screenshot(&mut self, bind: &std::sync::Arc<config::ScreenshotBind>) {
        if self.screenshot.is_some() {
            return;
        }
        match bind.mode {
            config::ScreenshotMode::Output => {
                let pt = self.cursor_point();
                if let Some(geom) = self.renderer.output_at(pt) {
                    info!(output = %geom.name, "screenshot: capturing full output");
                    self.screenshot_pending.push(InternalCapture {
                        output: geom.name,
                        region: smithay::utils::Rectangle::from_size(geom.mode_size),
                        show_cursor: bind.show_cursor,
                        purpose: CapturePurpose::Finalize { bind: bind.clone() },
                    });
                } else {
                    warn!("screenshot: no output under the cursor");
                }
            }
            config::ScreenshotMode::Region | config::ScreenshotMode::Window => {
                self.screenshot = Some(ScreenshotState {
                    bind: bind.clone(),
                    anchor: None,
                    frozen: std::collections::HashMap::new(),
                });
                if bind.freeze {
                    // Snapshot every output. The overlay (dim) is held back
                    // until those captures land (in `complete_internal_capture`)
                    // so the snapshot itself is clean — no dim baked in.
                    for geom in self.renderer.output_geometries() {
                        self.screenshot_pending.push(InternalCapture {
                            output: geom.name,
                            region: smithay::utils::Rectangle::from_size(geom.mode_size),
                            show_cursor: bind.show_cursor,
                            purpose: CapturePurpose::Freeze,
                        });
                    }
                } else {
                    // Live selection: dim immediately.
                    self.update_screenshot_overlay();
                }
                // Crosshair while selecting a region/window.
                self.renderer
                    .set_cursor_override(Some(CursorImageStatus::Named(CursorIcon::Crosshair)));
                info!("screenshot: session started");
            }
        }
    }

    /// Cancel an in-progress session: drop the UI + every pending capture
    /// (nothing is saved or copied).
    fn cancel_screenshot(&mut self) {
        self.screenshot = None;
        self.screenshot_pending.clear();
        self.renderer.clear_screenshot();
        self.renderer.set_cursor_override(None);
        info!("screenshot: cancelled");
    }

    /// End a session after a successful selection: drop the UI + leftover
    /// freeze snapshots, but keep any enqueued `Finalize` capture so its
    /// delivery on the next vblank still happens.
    fn end_screenshot(&mut self) {
        self.screenshot = None;
        self.screenshot_pending
            .retain(|c| matches!(c.purpose, CapturePurpose::Finalize { .. }));
        self.renderer.clear_screenshot();
        self.renderer.set_cursor_override(None);
    }

    /// Recompute the selection from the cursor + session state and hand it
    /// to the renderer for this and following frames.
    fn update_screenshot_overlay(&mut self) {
        let selection = self.current_screenshot_selection();
        self.renderer
            .set_screenshot_overlay(Some(render::ScreenshotOverlay { selection }));
    }

    /// The current selection in absolute compositor coords: the
    /// anchor→cursor rect (Region, once dragging) or the hovered window
    /// (Window). `None` dims the whole screen.
    fn current_screenshot_selection(&self) -> Option<smithay::utils::Rectangle<i32, Physical>> {
        let session = self.screenshot.as_ref()?;
        match session.bind.mode {
            config::ScreenshotMode::Region => {
                let (ax, ay) = session.anchor?;
                let (cx, cy) = self.renderer.cursor_pos();
                Some(rect_from_corners(ax, ay, cx, cy))
            }
            config::ScreenshotMode::Window => {
                self.layout.window_at(self.cursor_point()).map(|(_, r)| r)
            }
            config::ScreenshotMode::Output => None,
        }
    }

    /// Pointer moved during a session — refresh the selection UI.
    fn screenshot_pointer_motion(&mut self) {
        self.update_screenshot_overlay();
    }

    /// Left-button press during a session.
    fn screenshot_pointer_press(&mut self) {
        let Some(mode) = self.screenshot.as_ref().map(|s| s.bind.mode) else {
            return;
        };
        match mode {
            config::ScreenshotMode::Region => {
                let corner = self.renderer.cursor_pos();
                if let Some(s) = &mut self.screenshot {
                    s.anchor = Some(corner);
                }
                self.update_screenshot_overlay();
            }
            config::ScreenshotMode::Window => self.finalize_window(),
            config::ScreenshotMode::Output => {}
        }
    }

    /// Left-button release during a session — finalize a region drag.
    fn screenshot_pointer_release(&mut self) {
        let region_drag = self.screenshot.as_ref().is_some_and(|s| {
            matches!(s.bind.mode, config::ScreenshotMode::Region) && s.anchor.is_some()
        });
        if region_drag {
            self.finalize_region();
        }
    }

    /// Enter key during a session — confirm the current selection.
    fn confirm_screenshot_region(&mut self) {
        let Some(session) = self.screenshot.as_ref() else {
            return;
        };
        match (session.bind.mode, session.anchor.is_some()) {
            (config::ScreenshotMode::Region, true) => self.finalize_region(),
            (config::ScreenshotMode::Window, _) => self.finalize_window(),
            _ => {}
        }
    }

    /// Capture the dragged region (clamped to the output the drag started
    /// on) and deliver it.
    fn finalize_region(&mut self) {
        let (cx, cy) = self.renderer.cursor_pos();
        let Some(session) = self.screenshot.as_ref() else {
            return;
        };
        let Some((ax, ay)) = session.anchor else {
            return;
        };
        let sel = rect_from_corners(ax, ay, cx, cy);
        if sel.size.w < 1 || sel.size.h < 1 {
            self.cancel_screenshot();
            return;
        }
        self.finalize_compositor_rect(sel);
    }

    /// Capture the window under the cursor and deliver it.
    fn finalize_window(&mut self) {
        let Some(rect) = self.layout.window_at(self.cursor_point()).map(|(_, r)| r) else {
            self.cancel_screenshot();
            return;
        };
        self.finalize_compositor_rect(rect);
    }

    /// Shared region/window finalize: resolve the output, map the rect to
    /// its framebuffer pixels, then crop the frozen snapshot (freeze) or
    /// enqueue a live capture, and deliver.
    fn finalize_compositor_rect(&mut self, rect: smithay::utils::Rectangle<i32, Physical>) {
        let center = rect.loc + Point::from((rect.size.w / 2, rect.size.h / 2));
        let Some(geom) = self
            .renderer
            .output_at(center)
            .or_else(|| self.renderer.output_at(rect.loc))
        else {
            warn!("screenshot: selection isn't on any output");
            self.cancel_screenshot();
            return;
        };
        let Some(clamped) = rect.intersection(geom.compositor) else {
            self.cancel_screenshot();
            return;
        };
        let phys = compositor_rect_to_physical(clamped, geom.compositor.loc, geom.scale);
        if phys.size.w < 1 || phys.size.h < 1 {
            self.cancel_screenshot();
            return;
        }
        let Some(bind) = self.screenshot.as_ref().map(|s| s.bind.clone()) else {
            return;
        };
        if bind.freeze {
            // Clone the frozen output's raw bytes (a fast memcpy) so the encode
            // can run on a worker thread instead of freezing the compositor.
            let frozen = self
                .screenshot
                .as_ref()
                .and_then(|s| s.frozen.get(&geom.name))
                .map(|f| (f.bytes.clone(), f.width, f.height));
            if let Some((bytes, w, h)) = frozen {
                self.spawn_screenshot_encode(bytes, w, h, phys, &bind);
            } else {
                warn!(output = %geom.name, "screenshot: freeze snapshot unavailable");
            }
        } else {
            self.screenshot_pending.push(InternalCapture {
                output: geom.name,
                region: phys,
                show_cursor: bind.show_cursor,
                purpose: CapturePurpose::Finalize { bind },
            });
        }
        self.end_screenshot();
    }

    /// Encode `region` of `bytes` (BGRX, `width`×`height`) to PNG on a worker
    /// thread, then save it to the bind's directory and/or hand it back for
    /// the clipboard. Keeps the heavy zlib compression off the render thread
    /// so a large capture doesn't freeze the compositor.
    fn spawn_screenshot_encode(
        &self,
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        region: smithay::utils::Rectangle<i32, Physical>,
        bind: &config::ScreenshotBind,
    ) {
        let dir = bind.directory.as_ref().map(|d| screenshot::expand_dir(d));
        let filename = screenshot::timestamp_filename(self.local_offset);
        let clipboard = bind.clipboard;
        let tx = self.screenshot_clipboard_tx.clone();
        let spawned = std::thread::Builder::new()
            .name("screenshot-encode".to_owned())
            .spawn(move || {
                let png = match screenshot::encode_region(&bytes, width, height, region) {
                    Ok(png) => png,
                    Err(err) => {
                        warn!(error = %err, "screenshot: PNG encode failed");
                        return;
                    }
                };
                if let Some(dir) = dir {
                    match screenshot::save(&dir, &filename, &png) {
                        Ok(path) => info!(path = %path.display(), "screenshot saved"),
                        Err(err) => {
                            warn!(error = %err, dir = %dir.display(), "screenshot save failed");
                        }
                    }
                }
                if clipboard {
                    // Receiver gone (compositor exiting) → just drop it.
                    let _ = tx.send(png);
                }
            });
        if let Err(err) = spawned {
            warn!(error = %err, "screenshot: failed to spawn encode thread");
        }
    }

    /// Put a finished screenshot PNG on the clipboard. Called on the main
    /// thread from the worker-thread channel (selections must be set here).
    fn set_screenshot_clipboard(&mut self, png: Vec<u8>) {
        self.clipboard.set_image(
            smithay::wayland::selection::SelectionTarget::Clipboard,
            "image/png".to_owned(),
            png,
        );
        let dh = self.display_handle.clone();
        smithay::wayland::selection::data_device::set_data_device_selection::<State>(
            &dh,
            &self.seat,
            vec!["image/png".to_owned()],
            (),
        );
        info!("screenshot copied to clipboard (image/png)");
    }

    /// Route a compositor-originated capture's pixels (serviced in the
    /// vblank handler): store + display a freeze snapshot, or encode +
    /// deliver a finished grab.
    fn complete_internal_capture(&mut self, cap: InternalCapture, outcome: render::CaptureOutcome) {
        let render::CaptureOutcome::Shm {
            bytes,
            width,
            height,
        } = outcome
        else {
            warn!(output = %cap.output, "screenshot: internal capture failed");
            return;
        };
        let (Ok(w_i), Ok(h_i)) = (i32::try_from(width), i32::try_from(height)) else {
            return;
        };
        match cap.purpose {
            CapturePurpose::Freeze => {
                // Upload an upright, opaque RGBA copy as the backdrop; keep
                // the raw bytes for cropping the final image.
                let rgba = screenshot::to_rgba_topdown(&bytes, width, height);
                self.renderer
                    .set_freeze_texture(&cap.output, &rgba, w_i, h_i);
                if let Some(session) = &mut self.screenshot {
                    session.frozen.insert(cap.output, FrozenFrame { bytes, width, height });
                }
                // Reveal the dim overlay only once EVERY output's snapshot
                // has landed — otherwise a not-yet-frozen output would bake
                // the dim into its own snapshot.
                let more_freezes = self
                    .screenshot_pending
                    .iter()
                    .any(|c| matches!(c.purpose, CapturePurpose::Freeze));
                if !more_freezes {
                    self.update_screenshot_overlay();
                }
            }
            CapturePurpose::Finalize { bind } => {
                // The capture already read exactly the wanted region, so encode
                // the whole buffer — off the render thread (see helper).
                let full = smithay::utils::Rectangle::from_size(smithay::utils::Size::from((
                    w_i, h_i,
                )));
                self.spawn_screenshot_encode(bytes, width, height, full, &bind);
            }
        }
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "init flow is naturally linear (session → udev → DRM → renderer → keyboard → libinput → Wayland → state → run); extracting sub-helpers would obscure ownership/order more than the function being long does"
)]
fn main() -> Result<()> {
    // `libreland msg …` is the IPC control client, not the compositor.
    // Detect it before any compositor init (logging, config, DRM) so it
    // stays a fast, side-effect-free CLI that just talks to the socket.
    if std::env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new("msg")) {
        return ipc::run_client();
    }

    // The WorkerGuard MUST stay alive for the whole of main; dropping it
    // releases the tracing-appender worker thread and flushes the file
    // log. Bind it with a leading underscore so clippy doesn't nag, but
    // do NOT use `_` (anonymous) — that would drop it immediately.
    let _log_guard = init_tracing()?;
    info!("libreland starting");

    // Capture the local UTC offset NOW, while we're still single-threaded:
    // `time`'s local-offset lookup is unsound (and returns Err) once the
    // process is multithreaded, which a compositor quickly becomes. Used
    // for local-time screenshot filenames; falls back to UTC if unknown.
    let local_offset = time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC);

    // Compositor configuration: load Lua from $XDG_CONFIG_HOME/libreland/
    // config.lua if present, else fall back to compiled-in defaults.
    // A missing file is fine (logged at info); a present-but-broken
    // file logs the error and falls back to defaults rather than
    // aborting — the same path live-reload uses, so the user can fix
    // and save to recover. The file is watched for live reload below.
    let config_path = config::Config::path();
    let config = config::Config::load_or_default();
    info!("compositor config ready");

    // Apply the user's configured `env` table immediately — before the
    // renderer (which reads `$XCURSOR_THEME` / `$XCURSOR_SIZE` for its
    // pointer cursor) and before any child is spawned, so both the
    // compositor itself and every client inherit these. `WAYLAND_DISPLAY`
    // is set later, once the socket name is known.
    //
    // SAFETY: `std::env::set_var` is `unsafe` because changing the
    // process environment races with concurrent readers. We're at the
    // very top of `main`, still single-threaded apart from the
    // tracing-appender worker (which never reads env), so there's no
    // race window.
    #[allow(
        unsafe_code,
        reason = "set_var is unsafe due to multi-threaded env races; we call it at the top of main before spawning any thread that reads env (tracing-appender, the only background thread, never does), so the race window doesn't exist"
    )]
    // SAFETY: see #[allow] above.
    unsafe {
        // Session-identity defaults so apps + the desktop portal know
        // what they're running in without the user repeating them in
        // `env`. These describe the session truthfully — a Wayland
        // session named "libreland" — and are exactly what e.g. the
        // portal's config matching keys off (XDG_CURRENT_DESKTOP). The
        // user's `env` table is applied right after and overrides any
        // of these. (XDG base dirs / XDG_RUNTIME_DIR are deliberately
        // not touched — those are managed by the system, not the WM.)
        for (name, value) in DEFAULT_SESSION_ENV {
            std::env::set_var(name, value);
        }
        for (name, value) in &config.env {
            info!(name, value, "applying configured env var");
            std::env::set_var(name, value);
        }
        // $XCURSOR_SIZE is deliberately NOT exported (unless the user
        // set it themselves). The env var means a *logical* size to
        // native Wayland clients and to our own loader
        // (`cursor::configured_size`), but a *physical* size to X11
        // apps (their pixel space is physical-sized under the client
        // scale — see src/xwayland.rs), so no single value fits both.
        // Leaving it unset gives every consumer its correct source:
        // native clients fall back to the standard 24-logical default
        // (identical to what pinning it provided), and libXcursor in X
        // apps falls through to the `Xcursor.size` root resource — the
        // XWM publishes the physical size there (and via XSETTINGS for
        // toolkits), re-published on scale changes. An env value would
        // shadow that resource in libXcursor's lookup order and break
        // X cursor sizing on fractional outputs. A user-set value is
        // respected but skews X apps' cursors on fractional outputs by
        // the same logic.
    }

    // Wayland frontend bootstrap. The calloop loop data type is `State`
    // itself; the `Display<State>` is owned by the dispatch source's
    // closure (below) and outbound flushing goes through the
    // `DisplayHandle` on `State`, so the two never need to be nested.
    info!("phase: creating Wayland Display + substate");
    let mut display: Display<State> = Display::new().context("wayland Display::new failed")?;
    // Wayland init runs *after* the renderer is up — it needs the
    // renderer's per-output descriptors (mode size + compositor
    // position + scale) to create the `wl_output` globals and
    // seed the fractional-scale state.

    let mut event_loop: EventLoop<State> =
        EventLoop::try_new().context("failed to create calloop event loop")?;
    let handle = event_loop.handle();
    let loop_signal = event_loop.get_signal();

    // Phase markers around each blocking external call: combined with
    // the durable file writer this gives us "we got past step N but not
    // step N+1" granularity if anything hangs again.
    info!("phase: opening libseat session (logind via D-Bus)");
    // libseat probes for systemd-logind first and falls back to seatd.
    // It refuses if neither grants seat access (no logind session and the
    // user isn't a member of the seat group / not running under seatd-launch).
    let (mut session, notifier) = LibSeatSession::new()
        .context("failed to open libseat session (need a logind session or seat-group access)")?;
    let seat_name = session.seat();
    info!(seat = %seat_name, "libseat session acquired");

    info!("phase: opening udev backend");
    // udev gives us both initial device enumeration and live hotplug events
    // for the same seat. We snapshot paths into owned PathBufs so we can
    // keep using them after moving `udev` into the event loop below.
    let udev = UdevBackend::new(&seat_name).context("failed to start udev backend")?;
    let initial_devices: Vec<(_, std::path::PathBuf)> = udev
        .device_list()
        .map(|(id, path)| (id, path.to_path_buf()))
        .collect();
    info!(count = initial_devices.len(), "udev backend ready");
    for (device_id, path) in &initial_devices {
        debug!(device_id, path = %path.display(), "udev pre-existing device");
    }

    // A multi-GPU box exposes several `cardN` nodes (e.g. an iGPU with no
    // monitors plus a discrete card the displays hang off). udev lists them
    // in an unstable order, so try each in turn and keep the first that
    // actually brings up a connected output. A card that has no displays —
    // or that we momentarily can't drive — yields an error and we fall
    // through to the next one instead of failing the whole launch.
    let drm_paths = drm_card_paths(&initial_devices)?;
    let mut drm_init = None;
    let mut last_err: Option<anyhow::Error> = None;
    for drm_path in &drm_paths {
        info!(drm_path = %drm_path.display(), "trying DRM device");
        match drm::open_display(&mut session, drm_path, &config.monitors) {
            Ok(init) => {
                info!(drm_path = %drm_path.display(), "selected DRM device");
                drm_init = Some(init);
                break;
            }
            Err(err) => {
                warn!(drm_path = %drm_path.display(), error = %err, "DRM device unusable — trying next");
                last_err = Some(err);
            }
        }
    }
    let drm_init = drm_init.ok_or_else(|| {
        last_err
            .unwrap_or_else(|| anyhow::anyhow!("no usable DRM device"))
            .context("DRM device init failed (no enumerated card could drive a display)")
    })?;
    let drm::DrmInit {
        device: drm_device,
        fd: drm_fd,
        notifier: drm_notifier,
        outputs: drm_outputs,
    } = drm_init;

    // Explicit sync (linux-drm-syncobj-v1): advertise it only when this DRM
    // node supports syncobj eventfds (needed to build the acquire blocker);
    // otherwise clients use implicit dma-buf sync. The import device is the
    // single display/render GPU node — clone its fd before it moves into the
    // renderer. (NVIDIA's syncobj-eventfd support is probed at runtime, not
    // assumed.)
    let syncobj_import_fd = drm_fd.clone();
    let drm_syncobj_state = if smithay::wayland::drm_syncobj::supports_syncobj_eventfd(
        &syncobj_import_fd,
    ) {
        info!("explicit sync: advertising linux-drm-syncobj-v1 (device supports syncobj eventfd)");
        Some(smithay::wayland::drm_syncobj::DrmSyncobjState::new::<State>(
            &display.handle(),
            syncobj_import_fd,
        ))
    } else {
        warn!("explicit sync: device lacks syncobj eventfd; clients use implicit sync only");
        None
    };

    let mut renderer = render::Renderer::new(
        drm_fd,
        drm_outputs,
        // Bootstrap fill; `apply_wallpaper` below sets the real wallpaper
        // (flat or decoded media) once the renderer exists.
        config::Fill::Solid([0.0, 0.0, 0.0]),
        config.border.clone(),
        &config.monitors,
    )
    .context("render pipeline init failed")?;
    renderer.set_animations(config.animations.clone());
    renderer.set_decoration(config.decoration.clone());
    apply_wallpaper(&mut renderer, &config.misc.wallpaper, &config.border);

    info!("phase: priming swapchains (one initial frame per output)");
    renderer.render_initial().context("initial render failed")?;
    info!("all outputs primed for scanout");

    info!("phase: building Wayland frontend substate (post-renderer)");
    let output_descs = renderer.output_descriptors();
    let preferred_scale = renderer.primary_scale();
    let dmabuf_formats = renderer.dmabuf_formats();
    let scanout_formats = renderer.primary_scanout_formats();
    let render_node = renderer.render_drm_node();
    info!(
        count = dmabuf_formats.len(),
        scanout = scanout_formats.len(),
        render_node = ?render_node,
        "dmabuf import formats advertised to clients"
    );
    if dmabuf_formats.is_empty() {
        warn!(
            "renderer reports zero importable dmabuf formats; GPU clients (Steam, etc.) will \
             render blank — likely an EGL/driver dmabuf-import limitation"
        );
    }
    let wayland_init = wayland::init(
        &display,
        &config,
        &output_descs,
        preferred_scale,
        dmabuf_formats,
        scanout_formats,
        render_node,
    )
    .context("wayland substate init failed")?;
    info!("Wayland substate ready");

    info!("phase: initialising xkbcommon keyboard");
    let keyboard =
        keyboard::Keyboard::new(&config.input.keyboard_layout).context("keyboard init failed")?;
    info!("xkb keymap compiled");

    info!("phase: creating libinput context");
    // libinput opens `/dev/input/*` via the session interface so libseat
    // can grant the file descriptors under its permission model.
    // The session is cloned cheaply (Arc-based internally).
    let libinput_iface = LibinputSessionInterface::from(session.clone());
    let mut libinput_ctx = Libinput::new_with_udev(libinput_iface);
    info!("phase: assigning libinput to seat (enumerates and opens input devices)");
    libinput_ctx
        .udev_assign_seat(&seat_name)
        .map_err(|()| anyhow::anyhow!("libinput refused to assign seat {seat_name}"))?;
    info!("libinput seat assigned");
    let libinput_backend = LibinputInputBackend::new(libinput_ctx);

    info!("phase: opening Wayland listening socket");
    let listening_socket = ListeningSocketSource::new_auto()
        .context("Wayland ListeningSocketSource::new_auto failed")?;
    let socket_name = listening_socket.socket_name().to_os_string();
    info!(socket = ?socket_name, "Wayland socket listening");

    // SAFETY: `std::env::set_var` is `unsafe` in modern Rust because
    // setting the process environment is racy if other threads are
    // reading it concurrently. We're still in single-threaded init
    // (only the tracing-appender worker exists, and it doesn't read
    // env vars), so the call is safe here. We set `WAYLAND_DISPLAY`
    // so child processes spawned via `wayland::spawn_startup`
    // (below), `spawn` binds, and ad-hoc shell launches from the same
    // login session connect to *our* socket. The user's `env` table
    // was applied far earlier (right after config load) so the
    // renderer's cursor-theme lookup sees it.
    #[allow(
        unsafe_code,
        reason = "set_var is unsafe due to multi-threaded env races; we call it before spawning any non-trivial thread (tracing-appender is the only background thread and never reads env), so the race window doesn't exist"
    )]
    // SAFETY: see #[allow] above.
    unsafe {
        std::env::set_var("WAYLAND_DISPLAY", &socket_name);
    }

    // Control IPC socket, derived from the same display name. Export
    // `$LIBRELAND_SOCKET` so children + the `libreland msg` client connect
    // without re-deriving the path. The listener itself is registered on
    // the loop further down (it needs the loop handle).
    let ipc_socket = ipc::socket_path(&socket_name);
    if let Some(path) = &ipc_socket {
        // SAFETY: same single-threaded-init reasoning as the
        // WAYLAND_DISPLAY set_var above — still pre-event-loop.
        #[allow(
            unsafe_code,
            reason = "set_var is unsafe due to multi-threaded env races; called in single-threaded init before the event loop, same as WAYLAND_DISPLAY"
        )]
        // SAFETY: see #[allow] above.
        unsafe {
            std::env::set_var("LIBRELAND_SOCKET", path);
        }
    } else {
        warn!("XDG_RUNTIME_DIR unset; control IPC socket disabled");
    }

    // Native Xwayland (see src/xwayland.rs): spawn `Xwayland -rootless`
    // on a display smithay picks, with this compositor acting as its
    // window manager in-process — X11 windows tile like xdg toplevels,
    // X cursors ride the normal wl_pointer path, and X apps are told
    // the real output scale (client-scale mapping + XSETTINGS Xft/DPI).
    // Xwayland talks to us over a socketpair (no dependency on
    // WAYLAND_DISPLAY), but it's spawned here, pre-event-loop, so the
    // `$DISPLAY` export is still single-threaded-safe and the D-Bus
    // activation env below picks it up. The WM attaches once the loop
    // runs and Xwayland reports ready; X clients started before that
    // just block on the X socket until then.
    let mut xwayland_source = None;
    let mut xwayland_client = None;
    let mut xwayland_display = None;
    if config.xwayland
        && let Some(spawned) = xwayland::spawn_xwayland(
            &handle,
            &wayland_init.display_handle,
            renderer.primary_scale(),
        )
    {
        // SAFETY: same single-threaded-init reasoning as the
        // WAYLAND_DISPLAY set_var above — still pre-event-loop.
        #[allow(
            unsafe_code,
            reason = "set_var is unsafe due to multi-threaded env races; called in single-threaded init before the event loop, same as WAYLAND_DISPLAY"
        )]
        // SAFETY: see #[allow] above.
        unsafe {
            std::env::set_var("DISPLAY", &spawned.display);
        }
        info!(x_display = %spawned.display, "$DISPLAY exported for X11 clients");
        xwayland_source = Some(spawned.source);
        xwayland_client = Some(spawned.client);
        xwayland_display = Some(spawned.display);
    }

    // D-Bus-activated services (notably xdg-desktop-portal) are spawned
    // by the session bus, not by us, so they don't inherit our process
    // environment. Push the session-identity vars + WAYLAND_DISPLAY (and
    // DISPLAY) into the D-Bus/systemd activation environment so the
    // portal connects to us and matches the right config.
    export_activation_environment();

    // Snapshot the Display's poll fd so calloop can register a
    // `Generic` source over it. `try_clone_to_owned` gives us a
    // separate kernel fd referring to the same underlying file
    // description, sidestepping the lifetime issue of registering a
    // `BorrowedFd` whose lifetime is tied to `display`.
    let poll_fd = display
        .backend()
        .poll_fd()
        .try_clone_to_owned()
        .context("clone Display::poll_fd")?;

    wire_event_sources(&handle, notifier, udev, drm_notifier, libinput_backend)?;

    // Wayland socket source: each accepted UnixStream is registered
    // as a client on the display, attaching our per-client state.
    handle
        .insert_source(listening_socket, |stream, (), state: &mut State| {
            info!("Wayland: accepting new client");
            if let Err(err) = state
                .display_handle
                .insert_client(stream, wayland::new_client_data())
            {
                warn!(error = %err, "Wayland: insert_client failed");
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to insert wayland listener source: {e}"))?;

    // Wayland dispatch source: epoll-wakes us when any client has
    // sent a request; we drain it via `display.dispatch_clients`.
    // `flush_clients` runs in the event-loop post-batch callback
    // below so outbound messages don't accumulate.
    handle
        .insert_source(
            Generic::new(poll_fd, Interest::READ, Mode::Level),
            // The `Display` is owned here by the dispatch closure (calloop's
            // loop data is `State`); requests are drained into `State` and
            // outbound flushing happens via `state.display_handle` post-batch.
            move |_, _, state: &mut State| {
                if let Err(err) = display.dispatch_clients(state) {
                    warn!(error = %err, "wayland dispatch_clients failed");
                }
                Ok(PostAction::Continue)
            },
        )
        .map_err(|e| anyhow::anyhow!("failed to insert wayland dispatch source: {e}"))?;

    // Control IPC: bind the socket derived above and register it on the
    // loop. A bind failure is non-fatal — the compositor runs fine
    // without IPC, you just can't `libreland msg` it.
    if let Some(path) = &ipc_socket
        && let Err(err) = ipc::setup(&handle, path)
    {
        warn!(error = %err, "control IPC unavailable");
    }

    // Now that the socket is listening and `$WAYLAND_DISPLAY` is
    // set, spawn any configured startup commands. Their stdout /
    // stderr inherit ours (so they share the file log via
    // descriptors, if relevant).
    let startup_children = wayland::spawn_startup(&config.startup);

    // Live config reload: poll the config file once a second and
    // re-apply on change. Polling (vs inotify) is dependency-free and
    // robust to editors that save by atomic rename. A parse error is
    // non-fatal — `reload_config` keeps the running config. Skip
    // entirely if XDG couldn't resolve a config path.
    if let Some(watch_path) = config_path {
        use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
        // Change signal is `(mtime, size)`: mtime catches edits (ns
        // resolution on ext4/btrfs); size is a cheap belt-and-braces
        // for atomic-rename saves on filesystems with coarse mtime.
        // `metadata()` returns both in one stat, so size is free.
        let stamp = |path: &std::path::Path| {
            std::fs::metadata(path)
                .map(|m| (m.modified().ok(), m.len()))
                .ok()
        };
        let poll = std::time::Duration::from_secs(1);
        let mut last = stamp(&watch_path);
        handle
            .insert_source(
                Timer::from_duration(poll),
                move |_, (), state: &mut State| {
                    let current = stamp(&watch_path);
                    if current != last {
                        // Reload on edit / (re-)creation; ignore deletion
                        // (keep the running config until a file returns).
                        if current.is_some() {
                            info!(path = %watch_path.display(), "config changed; reloading");
                            state.reload_config(&watch_path);
                        }
                        last = current;
                    }
                    TimeoutAction::ToDuration(poll)
                },
            )
            .map_err(|e| anyhow::anyhow!("failed to insert config-watch timer: {e}"))?;
        info!("watching config file for live reload");
    }

    let layout = layout::Layout::new(
        renderer.output_rects(),
        layout::Gaps {
            outer: config.layout.gaps_outer,
            inner: config.layout.gaps_inner,
        },
        config.border.width,
    );
    // On-demand render bookkeeping: each output already has its priming
    // flip in flight (from `render_initial` above). Start every CRTC in
    // WaitingForVblank with `dirty` set, so the first vblank runs a real
    // `render_crtc` (not just a park) — that establishes the followup loop
    // for an animated wallpaper and draws any client that connected before
    // the loop started; outputs with nothing to animate park right after.
    let redraw = renderer
        .crtcs()
        .into_iter()
        .map(|crtc| (crtc, RedrawState::WaitingForVblank { dirty: true }))
        .collect();

    // ext-session-lock-v1 manager. Built here (before `display_handle` moves
    // into State) so it can borrow the handle; allow any client to lock.
    let session_lock_state =
        smithay::wayland::session_lock::SessionLockManagerState::new::<State, _>(
            &wayland_init.display_handle,
            |_client| true,
        );
    // ext-idle-notify: needs the loop handle to run its own timers (hence
    // the State-as-loop-data event loop). Built here so it can borrow the
    // display handle before it moves into State.
    let idle_notifier = smithay::wayland::idle_notify::IdleNotifierState::<State>::new(
        &wayland_init.display_handle,
        handle.clone(),
    );

    // Worker-thread screenshots: the encode/save run off the render thread;
    // when the bind also wants the clipboard the worker sends the encoded PNG
    // back here, and the loop sets the selection on the main thread.
    let (screenshot_clipboard_tx, screenshot_clipboard_rx) =
        smithay::reexports::calloop::channel::channel::<Vec<u8>>();
    handle
        .insert_source(screenshot_clipboard_rx, |event, (), state: &mut State| {
            if let smithay::reexports::calloop::channel::Event::Msg(png) = event {
                state.set_screenshot_clipboard(png);
            }
        })
        .expect("insert screenshot clipboard channel");

    let mut state = State {
        session,
        loop_signal,
        drm_device,
        renderer,
        keyboard,
        config,
        input_devices: Vec::new(),
        children: startup_children,
        xwayland_display,
        xwm: None,
        xwayland_source,
        xwayland_client,
        x11_windows: Vec::new(),
        x11_or_windows: Vec::new(),
        x11_kbd_focus: None,
        x11_owns_selection: crate::xwayland::X11SelectionOwnership::default(),
        xwayland_shell_state: wayland_init.xwayland_shell_state,
        display_handle: wayland_init.display_handle,
        compositor_state: wayland_init.compositor_state,
        shm_state: wayland_init.shm_state,
        seat_state: wayland_init.seat_state,
        seat: wayland_init.seat,
        cursor_shape_state: wayland_init.cursor_shape_state,
        xdg_shell_state: wayland_init.xdg_shell_state,
        xdg_decoration_state: wayland_init.xdg_decoration_state,
        kde_decoration_state: wayland_init.kde_decoration_state,
        output_manager_state: wayland_init.output_manager_state,
        outputs: wayland_init.outputs,
        output_globals: wayland_init.output_globals,
        fractional_scale_state: wayland_init.fractional_scale_state,
        viewporter_state: wayland_init.viewporter_state,
        data_device_state: wayland_init.data_device_state,
        dmabuf_state: wayland_init.dmabuf_state,
        dmabuf_global: wayland_init.dmabuf_global,
        dmabuf_default_feedback: wayland_init.dmabuf_default_feedback,
        dmabuf_scanout_feedback: wayland_init.dmabuf_scanout_feedback,
        keyboard_leds: smithay::input::keyboard::LedState::default(),
        scene_epoch: std::cell::Cell::new(1),
        popup_snapshot: std::cell::RefCell::new((0, Vec::new())),
        scanout_feedback_given: std::cell::RefCell::new(std::collections::HashSet::new()),
        preferred_scale: wayland_init.preferred_scale,
        layer_shell_state: wayland_init.layer_shell_state,
        layer_outputs: std::collections::HashMap::new(),
        layer_namespaces: std::collections::HashMap::new(),
        mapped_toplevels: std::collections::HashSet::new(),
        idle_last_input: std::time::Instant::now(),
        idle_screen_off: false,
        idle_lock_spawned: false,
        idle_notifier,
        idle_inhibit_state: wayland_init.idle_inhibit_state,
        idle_inhibitors: std::collections::HashSet::new(),
        xdg_activation_state: wayland_init.xdg_activation_state,
        pointer_gestures_state: wayland_init.pointer_gestures_state,
        color_management: wayland_init.color_management,
        content_type_state: wayland_init.content_type_state,
        presentation_state: wayland_init.presentation_state,
        drm_syncobj_state,
        color_surfaces: std::collections::HashMap::new(),
        color_surface_objects: std::collections::HashSet::new(),
        pending_image_info: Vec::new(),
        session_lock_state,
        lock_surfaces: std::collections::HashMap::new(),
        session_locked: false,
        relative_pointer_state: wayland_init.relative_pointer_state,
        pointer_constraints_state: wayland_init.pointer_constraints_state,
        primary_selection_state: wayland_init.primary_selection_state,
        wlr_data_control_state: wayland_init.wlr_data_control_state,
        ext_data_control_state: wayland_init.ext_data_control_state,
        clipboard: clipboard::Selections::default(),
        screencopy_manager: wayland_init.screencopy_manager,
        screencopy_pending: Vec::new(),
        loop_handle: handle.clone(),
        redraw,
        popup_manager: wayland_init.popup_manager,
        popup_grab: None,
        kbd_focus_before_layer: None,
        layout,
        drag: None,
        ws_scroll_accum: 0.0,
        screenshot: None,
        screenshot_pending: Vec::new(),
        local_offset,
        screenshot_clipboard_tx,
        ipc: ipc::IpcState::default(),
    };
    // Engage Num Lock at startup when configured. Goes through the xkb
    // locked-modifier state, so clients see the modifier and the LED sync
    // lights the keyboards (once libinput reports them).
    if state.config.input.numlock {
        state.set_numlock(true);
    }

    info!("entering event loop — type to generate events, super+shift+e to exit");
    event_loop
        .run(None, &mut state, |state| {
            // Post-batch: broadcast any IPC state changes (focus, windows,
            // workspaces) to subscribers, then flush Wayland clients so
            // their pending outbound messages don't accumulate. A flush
            // failure typically means a client died mid-flight; log and
            // move on rather than crash the compositor.
            ipc::poll_events(state);
            // Send any colour-management get_information responses queued
            // during this iteration's dispatch (deferred so the `done`
            // destructor doesn't fire inside the creating request).
            color_management::flush_pending_image_info(state);
            if let Err(err) = state.display_handle.flush_clients() {
                warn!(error = %err, "wayland flush_clients failed");
            }
        })
        .map_err(|e| anyhow::anyhow!("event loop error: {e}"))?;

    info!("libreland exiting");
    Ok(())
}

/// `io::Write` adaptor that calls `sync_data()` after each write,
/// giving us durable per-record file logging when paired with
/// tracing-appender's non-blocking worker. The main-loop side of the
/// channel never waits for disk (the worker thread absorbs the sync
/// cost); the only events that can be lost on a hard reboot are those
/// still queued in the worker's channel, which is sub-millisecond in
/// practice.
///
/// `sync_data` (fdatasync) is preferred over `sync_all` (fsync): we
/// don't care about non-essential metadata like access time, and
/// fdatasync is noticeably cheaper on ext4 / xfs.
struct DurableWriter(File);

impl io::Write for DurableWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.0.write(buf)?;
        self.0.sync_data()?;
        Ok(n)
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.0.write_all(buf)?;
        self.0.sync_data()
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()?;
        self.0.sync_data()
    }
}

/// Initialise tracing-subscriber with two sinks:
///
/// 1. A per-startup file under `$XDG_STATE_HOME/libreland/` — the
///    canonical, ANSI-free record we can read after the fact. Wrapped
///    in [`DurableWriter`] so every record reaches stable storage
///    before the worker dequeues the next one; this is what makes the
///    file usable after a hard reboot (without fsync, the previous
///    freeze gave us a zero-byte file because the page cache was lost).
/// 2. `stderr` — ANSI-coloured, for live development on the host.
///
/// Honours `RUST_LOG` if set; otherwise defaults to a developer-friendly
/// mix that hides smithay/calloop noise while keeping our own messages
/// visible.
///
/// The returned [`WorkerGuard`] MUST be held for the lifetime of the
/// program. `tracing-appender`'s non-blocking writer drains records on a
/// background thread; dropping the guard flushes pending records and
/// shuts that thread down cleanly. If the guard is dropped early, in-
/// flight log records are silently lost.
fn init_tracing() -> Result<WorkerGuard> {
    // Resolve $XDG_STATE_HOME/libreland/ (default ~/.local/state/libreland)
    // and create it if absent.
    let dirs = xdg::BaseDirectories::with_prefix("libreland");
    let log_dir = dirs
        .create_state_directory("")
        .context("failed to create XDG state directory for libreland")?;

    // Per-startup filename: ISO-8601 UTC with `:` swapped for `-` so the
    // name stays friendly on FAT/exFAT (Linux allows colons in filenames,
    // but cross-filesystem portability is cheap to keep). UTC dodges the
    // soundness landmines around `time::OffsetDateTime::now_local()` in
    // multi-threaded contexts — the appender's worker thread we spawn
    // below is one such context.
    let now = time::OffsetDateTime::now_utc();
    let stamp_fmt =
        time::macros::format_description!("[year]-[month]-[day]T[hour]-[minute]-[second]Z");
    let stamp = now
        .format(&stamp_fmt)
        .context("failed to format current timestamp for log filename")?;
    let log_path = log_dir.join(format!("{stamp}.log"));

    let log_file = File::create(&log_path)
        .with_context(|| format!("failed to create log file at {}", log_path.display()))?;
    let log_file = DurableWriter(log_file);

    let (file_writer, guard) = tracing_appender::non_blocking(log_file);

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,libreland=debug"));

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(file_writer)
                .with_ansi(false)
                .with_target(true),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_ansi(true)
                .with_target(true),
        )
        .init();

    info!(log_file = %log_path.display(), "tracing initialised (file + stderr)");
    install_panic_hook(&log_dir);
    Ok(guard)
}

/// Chain a tracing layer onto the default panic hook so panic messages
/// also land in the file log. Stderr-only panic output is invisible
/// during the TTY freeze scenario that motivated file logging in the
/// first place — without this, a panic would crash the compositor with
/// no on-disk record. We delegate to the previous hook so the default
/// stderr + backtrace behaviour is preserved unchanged.
fn install_panic_hook(log_dir: &std::path::Path) {
    use std::io::Write as _;
    // The normal log uses a non-blocking writer whose in-flight buffer is
    // lost when a panic aborts the process — so panics were vanishing. Also
    // append synchronously to a dedicated panic.log (with a backtrace) that
    // survives the abort, on top of the tracing + default stderr behaviour.
    let panic_log = log_dir.join("panic.log");
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&panic_log)
        {
            let _ = writeln!(file, "==== compositor panic ====\n{panic_info}\n{backtrace}");
            let _ = file.flush();
        }
        tracing::error!(panic = %panic_info, "compositor panicked");
        default_hook(panic_info);
    }));
}

/// Apply the user's libinput config (acceleration profile + speed) to
/// a newly-added pointer-capable device. No-op for non-pointers
/// (keyboards / touch / tablet). Logs and continues on failure — a
/// device that refuses one of the config calls still works at its
/// previous setting.
fn apply_input_config(device: &mut libinput::Device, input_config: &config::InputConfig) {
    if !device.has_capability(libinput::DeviceCapability::Pointer) {
        return;
    }
    let profile = match input_config.mouse_accel_profile {
        config::AccelProfile::Flat => libinput::AccelProfile::Flat,
        config::AccelProfile::Adaptive => libinput::AccelProfile::Adaptive,
    };
    if let Err(err) = device.config_accel_set_profile(profile) {
        warn!(?err, ?device, "failed to set libinput accel profile");
    }
    if let Err(err) = device.config_accel_set_speed(input_config.mouse_accel_speed) {
        warn!(?err, ?device, "failed to set libinput accel speed");
    }
}

/// Export the session-identity vars (+ `WAYLAND_DISPLAY` / `DISPLAY`)
/// into the D-Bus and systemd-user activation environment, so
/// D-Bus-activated services started by the session bus — above all
/// `xdg-desktop-portal` and its backends — see them. Without this the
/// portal can't tell which desktop it's in or which compositor socket
/// to use, and screencast/screenshot break. Best-effort: a missing
/// `dbus-update-activation-environment` (or no session bus) is logged,
/// never fatal.
fn export_activation_environment() {
    let mut names: Vec<&str> = DEFAULT_SESSION_ENV.iter().map(|(name, _)| *name).collect();
    names.push("WAYLAND_DISPLAY");
    if std::env::var_os("DISPLAY").is_some() {
        names.push("DISPLAY");
    }
    match std::process::Command::new("dbus-update-activation-environment")
        .arg("--systemd")
        .args(&names)
        .status()
    {
        Ok(status) if status.success() => {
            info!(vars = ?names, "exported session env to the D-Bus activation environment");
        }
        Ok(status) => warn!(
            ?status,
            "dbus-update-activation-environment exited non-zero; portals may not see the session env"
        ),
        Err(err) => warn!(
            error = %err,
            "could not run dbus-update-activation-environment (is dbus installed?); portals may not see the session env"
        ),
    }
}

/// Pick a `/dev/dri/cardN` node from a udev enumeration — render nodes
/// (`renderD128`) come through the same DRM subsystem and we
/// explicitly don't want them for modesetting. First card wins for
/// now; multi-GPU is a later milestone.
/// Collect every `/dev/dri/cardN` path udev enumerated, sorted by name
/// for deterministic ordering (udev's own iteration order is not stable
/// across runs). On a multi-GPU machine this returns e.g. `card0`,
/// `card1`; the caller tries each in turn until one yields a connected
/// output, so we don't hard-fail when udev happens to list a display-less
/// render GPU (iGPU, headless card) first.
fn drm_card_paths<T>(devices: &[(T, std::path::PathBuf)]) -> Result<Vec<std::path::PathBuf>> {
    let mut paths: Vec<std::path::PathBuf> = devices
        .iter()
        .filter(|(_, p)| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("card"))
        })
        .map(|(_, p)| p.clone())
        .collect();
    paths.sort();
    if paths.is_empty() {
        anyhow::bail!("no /dev/dri/cardN device enumerated by udev — no display to drive");
    }
    Ok(paths)
}

/// Resolve a DRM page-flip event's metadata into a `wp_presentation`
/// presentation time (`CLOCK_MONOTONIC`), page-flip sequence, and base
/// feedback flags. A kernel-provided monotonic flip timestamp is a true
/// hardware-clock completion time (`Vsync | HwClock | HwCompletion`); a
/// realtime or absent timestamp falls back to sampling `CLOCK_MONOTONIC` now,
/// flagged only `Vsync`.
fn present_info(meta: Option<&DrmEventMetadata>) -> (std::time::Duration, u32, PresentKind) {
    match meta {
        Some(m) => match m.time {
            DrmEventTime::Monotonic(d) => (
                d,
                m.sequence,
                PresentKind::Vsync | PresentKind::HwClock | PresentKind::HwCompletion,
            ),
            DrmEventTime::Realtime(_) => (
                Clock::<Monotonic>::new().now().into(),
                m.sequence,
                PresentKind::Vsync,
            ),
        },
        None => (Clock::<Monotonic>::new().now().into(), 0, PresentKind::Vsync),
    }
}

/// Insert the libseat/udev/DRM/libinput event sources into the calloop
/// handle. Pulled out of `main` so the init flow stays under clippy's
/// `too_many_lines` threshold without losing per-source visibility.
/// The Wayland-related sources (listener + dispatch fd) are inserted
/// directly in `main` because they share lifetimes with the `Display`
/// + `ListeningSocketSource` constructed there.
#[allow(
    clippy::too_many_lines,
    reason = "one function that wires every backend event source (session, udev, DRM vblank incl. screencopy servicing, libinput); splitting a source out means threading the loop handle + closures through another fn for no clarity gain"
)]
fn wire_event_sources(
    handle: &smithay::reexports::calloop::LoopHandle<'_, State>,
    session_notifier: LibSeatSessionNotifier,
    udev: UdevBackend,
    drm_notifier: smithay::backend::drm::DrmDeviceNotifier,
    libinput_backend: LibinputInputBackend,
) -> Result<()> {
    handle
        .insert_source(session_notifier, |event, (), state: &mut State| match event {
            smithay::backend::session::Event::PauseSession => warn!("session paused"),
            smithay::backend::session::Event::ActivateSession => {
                info!("session activated");
                // The kernel dropped any flip that was in flight when we
                // switched VTs away, so its vblank will never arrive. Discard
                // the stale WaitingForVblank bookkeeping (reset to Idle) and
                // force a fresh render of every output to restore scanout.
                // Buffers may have been scribbled over while away — restart
                // damage diffing with a full repaint.
                state.renderer.invalidate_damage();
                for crtc in state.renderer.crtcs() {
                    state.redraw.insert(crtc, RedrawState::Idle);
                }
                state.queue_redraw_all();
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to insert session source: {e}"))?;

    handle
        .insert_source(udev, |event, (), state: &mut State| match event {
            UdevEvent::Added { device_id, path } => {
                info!(device_id, path = %path.display(), "udev: device added");
            }
            UdevEvent::Removed { device_id } => {
                info!(device_id, "udev: device removed");
            }
            UdevEvent::Changed { device_id } => {
                // A connector changed state on a DRM device — most often a
                // monitor was plugged in or unplugged. Re-scan and reconcile
                // our output set. We drive a single GPU, so a change on any
                // card triggers a rescan of ours; the diff makes it a no-op
                // when nothing we own actually changed.
                debug!(device_id, "udev: device changed");
                state.handle_drm_changed();
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to insert udev source: {e}"))?;

    handle
        .insert_source(
            drm_notifier,
            |event, meta, state: &mut State| match event {
                smithay::backend::drm::DrmEvent::VBlank(crtc) => {
                    // The flip for this output just completed — ack it so the
                    // swapchain frees the scanned-out buffer, and feed
                    // `wp_presentation` the real flip timestamp + sequence.
                    // A monotonic page-flip timestamp from the kernel is a
                    // hardware-clock presentation time; otherwise fall back to
                    // sampling CLOCK_MONOTONIC now (no hw-clock flags then).
                    let (present_time, seq, base_flags) = present_info(meta.as_ref());
                    state.renderer.frame_submitted(crtc, present_time, seq, base_flags);
                    // Re-render only if a trigger arrived while the flip was in
                    // flight, or an animation/slide is still running. Otherwise
                    // the output parks until the next trigger queues a redraw.
                    let again = matches!(
                        state.redraw.get(&crtc),
                        Some(RedrawState::WaitingForVblank { dirty: true })
                    );
                    if again {
                        state.render_crtc(crtc);
                    } else {
                        state.redraw.insert(crtc, RedrawState::Idle);
                    }
                }
                smithay::backend::drm::DrmEvent::Error(err) => {
                    warn!(error = %err, "drm: event-source error");
                }
            },
        )
        .map_err(|e| anyhow::anyhow!("failed to insert drm source: {e}"))?;

    // Safety heartbeat for on-demand rendering. Correctness depends on every
    // visual change queueing a redraw; should a trigger ever be missed, an
    // output would otherwise freeze. This ticks each *non-fullscreen* output
    // about once a second, so a missed trigger degrades to <=1s of staleness
    // instead of a stuck frame. It skips fullscreen outputs, so it never
    // disturbs a game's VRR (those redraw on the client's own commits). Once
    // the triggers are proven on hardware this can be lengthened or removed.
    {
        use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
        let beat = std::time::Duration::from_secs(1);
        handle
            .insert_source(Timer::from_duration(beat), move |_, (), state: &mut State| {
                state.queue_redraw_nonfullscreen();
                TimeoutAction::ToDuration(beat)
            })
            .map_err(|e| anyhow::anyhow!("failed to insert redraw heartbeat: {e}"))?;
    }

    // Built-in idle handling: a low-frequency tick that locks / powers off the
    // screens once the configured idle thresholds elapse (input wakes them).
    // `idle_tick` early-returns unless `config.idle` is set, so an unconfigured
    // session just no-ops here. 5 s granularity is plenty for minute timeouts.
    {
        use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
        let tick = std::time::Duration::from_secs(5);
        handle
            .insert_source(Timer::from_duration(tick), move |_, (), state: &mut State| {
                state.idle_tick();
                TimeoutAction::ToDuration(tick)
            })
            .map_err(|e| anyhow::anyhow!("failed to insert idle timer: {e}"))?;
    }

    // Reap exited children so zombies don't accumulate: every runtime
    // spawn path (startup, binds, idle lock, IPC) parks its `Child` in
    // `state.children`; this sweeps them with `try_wait` (never blocks).
    {
        use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
        let sweep = std::time::Duration::from_secs(5);
        handle
            .insert_source(Timer::from_duration(sweep), move |_, (), state: &mut State| {
                state.reap_children();
                TimeoutAction::ToDuration(sweep)
            })
            .map_err(|e| anyhow::anyhow!("failed to insert child-reap timer: {e}"))?;
    }

    handle
        .insert_source(libinput_backend, |event, (), state: &mut State| {
            log_input_event(&event);
            // Any real user input resets the idle clock and wakes powered-off
            // screens (so a mouse move turns the panels back on). Device
            // add/remove isn't user activity, so it's excluded.
            if matches!(
                event,
                InputEvent::Keyboard { .. }
                    | InputEvent::PointerMotion { .. }
                    | InputEvent::PointerButton { .. }
                    | InputEvent::PointerAxis { .. }
            ) {
                state.note_input_activity();
            }
            match event {
                InputEvent::Keyboard { event: ke } => state.handle_key(&ke),
                InputEvent::PointerMotion { event: pm } => {
                    state
                        .forward_pointer_motion::<LibinputInputBackend>(&pm);
                }
                InputEvent::PointerButton { event: pb } => {
                    state
                        .forward_pointer_button(pb.button_code(), pb.state(), pb.time_msec());
                }
                InputEvent::PointerAxis { event: pa } => {
                    state.forward_pointer_axis::<LibinputInputBackend>(&pa);
                }
                // Touchpad gestures (pinch / swipe / hold) — forwarded to
                // the client under the pointer (browser pinch-zoom, GTK
                // swipe). Purely client-facing, so no compositor handling.
                InputEvent::GestureSwipeBegin { event } => {
                    state.gesture_swipe_begin::<LibinputInputBackend>(&event);
                }
                InputEvent::GestureSwipeUpdate { event } => {
                    state.gesture_swipe_update::<LibinputInputBackend>(&event);
                }
                InputEvent::GestureSwipeEnd { event } => {
                    state.gesture_swipe_end::<LibinputInputBackend>(&event);
                }
                InputEvent::GesturePinchBegin { event } => {
                    state.gesture_pinch_begin::<LibinputInputBackend>(&event);
                }
                InputEvent::GesturePinchUpdate { event } => {
                    state.gesture_pinch_update::<LibinputInputBackend>(&event);
                }
                InputEvent::GesturePinchEnd { event } => {
                    state.gesture_pinch_end::<LibinputInputBackend>(&event);
                }
                InputEvent::GestureHoldBegin { event } => {
                    state.gesture_hold_begin::<LibinputInputBackend>(&event);
                }
                InputEvent::GestureHoldEnd { event } => {
                    state.gesture_hold_end::<LibinputInputBackend>(&event);
                }
                InputEvent::DeviceAdded { mut device } => {
                    apply_input_config(&mut device, &state.config.input);
                    // A fresh keyboard arrives with its LEDs dark no matter
                    // the session's lock state — sync it immediately.
                    if device.has_capability(smithay::reexports::input::DeviceCapability::Keyboard)
                    {
                        device.led_update(state.keyboard_leds.into());
                    }
                    // Keep the handle so a config reload can re-apply
                    // mouse-accel settings to this live device.
                    state.input_devices.push(device);
                }
                InputEvent::DeviceRemoved { device } => {
                    state.input_devices.retain(|d| *d != device);
                }
                _ => {}
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to insert libinput source: {e}"))?;

    Ok(())
}

/// Touchpad gesture forwarding (`zwp_pointer_gestures_v1`). Each method
/// translates a libinput gesture event into the matching smithay pointer
/// event and sends it to the client under the pointer, then `frame()`s.
/// Gestures don't change compositor state, so there's no redraw or focus
/// handling — they ride the pointer's existing focus.
impl State {
    fn gesture_swipe_begin<I: InputBackend>(&mut self, e: &I::GestureSwipeBeginEvent) {
        use smithay::backend::input::{Event as _, GestureBeginEvent as _};
        let Some(ptr) = self.seat.get_pointer() else {
            return;
        };
        let evt = smithay::input::pointer::GestureSwipeBeginEvent {
            serial: SERIAL_COUNTER.next_serial(),
            time: e.time_msec(),
            fingers: e.fingers(),
        };
        ptr.gesture_swipe_begin(self, &evt);
        ptr.frame(self);
    }

    fn gesture_swipe_update<I: InputBackend>(&mut self, e: &I::GestureSwipeUpdateEvent) {
        use smithay::backend::input::{Event as _, GestureSwipeUpdateEvent as _};
        let Some(ptr) = self.seat.get_pointer() else {
            return;
        };
        let evt = smithay::input::pointer::GestureSwipeUpdateEvent {
            time: e.time_msec(),
            delta: e.delta(),
        };
        ptr.gesture_swipe_update(self, &evt);
        ptr.frame(self);
    }

    fn gesture_swipe_end<I: InputBackend>(&mut self, e: &I::GestureSwipeEndEvent) {
        use smithay::backend::input::{Event as _, GestureEndEvent as _};
        let Some(ptr) = self.seat.get_pointer() else {
            return;
        };
        let evt = smithay::input::pointer::GestureSwipeEndEvent {
            serial: SERIAL_COUNTER.next_serial(),
            time: e.time_msec(),
            cancelled: e.cancelled(),
        };
        ptr.gesture_swipe_end(self, &evt);
        ptr.frame(self);
    }

    fn gesture_pinch_begin<I: InputBackend>(&mut self, e: &I::GesturePinchBeginEvent) {
        use smithay::backend::input::{Event as _, GestureBeginEvent as _};
        let Some(ptr) = self.seat.get_pointer() else {
            return;
        };
        let evt = smithay::input::pointer::GesturePinchBeginEvent {
            serial: SERIAL_COUNTER.next_serial(),
            time: e.time_msec(),
            fingers: e.fingers(),
        };
        ptr.gesture_pinch_begin(self, &evt);
        ptr.frame(self);
    }

    fn gesture_pinch_update<I: InputBackend>(&mut self, e: &I::GesturePinchUpdateEvent) {
        use smithay::backend::input::{Event as _, GesturePinchUpdateEvent as _};
        let Some(ptr) = self.seat.get_pointer() else {
            return;
        };
        let evt = smithay::input::pointer::GesturePinchUpdateEvent {
            time: e.time_msec(),
            delta: e.delta(),
            scale: e.scale(),
            rotation: e.rotation(),
        };
        ptr.gesture_pinch_update(self, &evt);
        ptr.frame(self);
    }

    fn gesture_pinch_end<I: InputBackend>(&mut self, e: &I::GesturePinchEndEvent) {
        use smithay::backend::input::{Event as _, GestureEndEvent as _};
        let Some(ptr) = self.seat.get_pointer() else {
            return;
        };
        let evt = smithay::input::pointer::GesturePinchEndEvent {
            serial: SERIAL_COUNTER.next_serial(),
            time: e.time_msec(),
            cancelled: e.cancelled(),
        };
        ptr.gesture_pinch_end(self, &evt);
        ptr.frame(self);
    }

    fn gesture_hold_begin<I: InputBackend>(&mut self, e: &I::GestureHoldBeginEvent) {
        use smithay::backend::input::{Event as _, GestureBeginEvent as _};
        let Some(ptr) = self.seat.get_pointer() else {
            return;
        };
        let evt = smithay::input::pointer::GestureHoldBeginEvent {
            serial: SERIAL_COUNTER.next_serial(),
            time: e.time_msec(),
            fingers: e.fingers(),
        };
        ptr.gesture_hold_begin(self, &evt);
        ptr.frame(self);
    }

    fn gesture_hold_end<I: InputBackend>(&mut self, e: &I::GestureHoldEndEvent) {
        use smithay::backend::input::{Event as _, GestureEndEvent as _};
        let Some(ptr) = self.seat.get_pointer() else {
            return;
        };
        let evt = smithay::input::pointer::GestureHoldEndEvent {
            serial: SERIAL_COUNTER.next_serial(),
            time: e.time_msec(),
            cancelled: e.cancelled(),
        };
        ptr.gesture_hold_end(self, &evt);
        ptr.frame(self);
    }
}

/// Log a single libinput event. Keyboard / pointer events are what we
/// care about for the TTY sanity check; touch and tablet variants are
/// intentionally elided here.
fn log_input_event(event: &InputEvent<LibinputInputBackend>) {
    match event {
        InputEvent::DeviceAdded { device } => {
            info!(?device, "input: device added");
        }
        InputEvent::DeviceRemoved { device } => {
            info!(?device, "input: device removed");
        }
        InputEvent::Keyboard { event } => {
            // Keycode (from xkbcommon) doesn't implement tracing's Value
            // trait, so we Debug-format it. Same goes for KeyState.
            debug!(
                key_code = ?event.key_code(),
                state = ?event.state(),
                "input: key"
            );
        }
        InputEvent::PointerMotion { event } => {
            debug!(dx = event.dx(), dy = event.dy(), "input: pointer motion");
        }
        InputEvent::PointerButton { event } => {
            debug!(
                button = event.button_code(),
                state = ?event.state(),
                "input: pointer button"
            );
        }
        _ => {}
    }
}
