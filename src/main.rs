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
use smithay::backend::drm::DrmDevice;
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
use smithay::reexports::input as libinput;
use smithay::reexports::input::Libinput;
use smithay::reexports::input::event::keyboard::KeyboardKeyEvent as LibinputKeyEvent;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{Display, DisplayHandle, Resource as _};
use smithay::utils::{Logical, Physical, Point, SERIAL_COUNTER};
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
mod config;
mod cursor;
mod drm;
mod ipc;
mod keyboard;
mod media;
mod layout;
mod render;
mod screencopy;
mod screenshot;
mod wayland;

/// Mutable state threaded through every event-loop callback.
///
/// Holds the existing libseat / DRM / renderer / xkb / config state
/// plus the Wayland frontend substate added in milestone 4a
/// (compositor, shm, seat, `xdg_shell`, `output_manager`). The owned
/// `Display<State>` itself can't live here (the type would be
/// circular), so it sits beside `State` in [`LoopData`].
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
    pub(crate) loop_handle: smithay::reexports::calloop::LoopHandle<'static, LoopData>,
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

/// Calloop user-data wrapper: owns the Wayland `Display<State>` and
/// the compositor `State` side by side, since they can't be nested
/// (the type would be circular). Calloop source callbacks receive
/// `&mut LoopData` and split-borrow the two fields independently.
pub(crate) struct LoopData {
    pub(crate) state: State,
    pub(crate) display: Display<State>,
}

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

        let matched_action = if pressed {
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
        // (via kbd.set_focus / pointer.motion). A popup under the
        // cursor wins the *pointer* (menus draw on top of everything),
        // then layer surfaces (rofi, panels, OSDs), then the tile /
        // floating layout.
        let popup_hit = self.popup_at(cursor_i);
        let surface_hit = self
            .layer_at(cursor_i)
            .map(|(surface, rect)| {
                (
                    surface,
                    Point::<f64, Logical>::from((f64::from(rect.loc.x), f64::from(rect.loc.y))),
                )
            })
            .or_else(|| {
                self.layout.window_at(cursor_i).map(|(w, rect)| {
                    (
                        w.toplevel.wl_surface().clone(),
                        // Effective rect origin: the output origin for a
                        // fullscreen/maximized window, else its cell.
                        Point::<f64, Logical>::from((f64::from(rect.loc.x), f64::from(rect.loc.y))),
                    )
                })
            });
        // Keyboard focus follows windows / layers only — never popups.
        // We don't run a popup grab yet, and a menu shouldn't pull
        // keyboard focus off its parent toplevel.
        let kbd_target = surface_hit.as_ref().map(|(surface, _)| surface.clone());
        let hit = popup_hit.or(surface_hit);

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
        if matches!(self.config.input.focus_model, config::FocusModel::Hover)
            && !pointer.is_grabbed()
            && !self.focus_locked_by_layer()
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
            // popup's parent.
            if self.popup_at(cursor_i).is_none() {
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
            // Anchoring to BOTH opposite edges stretches the surface
            // across that axis (output minus the two margins) — e.g.
            // a waybar anchored top+left+right spans the full width.
            let stretch_x = anchor.contains(Anchor::LEFT) && anchor.contains(Anchor::RIGHT);
            let stretch_y = anchor.contains(Anchor::TOP) && anchor.contains(Anchor::BOTTOM);
            // Size: stretched span, else the client's requested size,
            // else the full output dimension when it asked the
            // compositor to choose (size 0). Clamp to the output so a
            // misbehaving client can't drive a negative offset below.
            let mut width = if stretch_x {
                area.size.w - cached.margin.left - cached.margin.right
            } else if cached.size.w > 0 {
                cached.size.w
            } else {
                area.size.w
            };
            let mut height = if stretch_y {
                area.size.h - cached.margin.top - cached.margin.bottom
            } else if cached.size.h > 0 {
                cached.size.h
            } else {
                area.size.h
            };
            width = width.clamp(1, area.size.w);
            height = height.clamp(1, area.size.h);
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
                // XWayland (xwayland-satellite) menus never call
                // set_window_geometry, so popup.geometry() is a zero
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
        let focused = self.seat.get_keyboard().and_then(|k| k.current_focus());
        let placements = self.layout.placements(focused.as_ref(), None);
        let layers = self.snapshot_layer_placements();
        let popups = self.snapshot_popup_placements(&placements, &layers);
        popups.iter().rev().find_map(|pp| {
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
        let primary_name = self.renderer.primary_output_name().to_owned();
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

    /// Run a bound action. Grows as we add more actions (`reload`,
    /// `spawn`, `change_vt`, …).
    fn dispatch_action(&mut self, action: config::Action) {
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
                // Politely ask the focused toplevel to close. Match
                // the keyboard-focused surface against the live
                // toplevels and send xdg_toplevel.close; the client
                // drives its own teardown (which destroys the
                // surface, and our XdgShellHandler removes it from
                // the layout). No focus / no matching toplevel (e.g.
                // a layer surface like rofi is focused) = no-op.
                let focus = self.seat.get_keyboard().and_then(|k| k.current_focus());
                if let Some(surface) = focus
                    && let Some(toplevel) = self
                        .xdg_shell_state
                        .toplevel_surfaces()
                        .iter()
                        .find(|t| t.wl_surface() == &surface)
                {
                    info!(surface = ?surface.id(), "close action fired");
                    toplevel.send_close();
                }
            }
            config::Action::Spawn(cmd) => {
                // Identical semantics to `wayland::spawn_startup`
                // but runs at bind-press time: whitespace-split
                // into program + args, inherit our env so
                // `$WAYLAND_DISPLAY` reaches the child. Empty
                // commands and failures are logged and the loop
                // keeps running.
                let parts: Vec<&str> = cmd.split_whitespace().collect();
                let Some((program, args)) = parts.split_first() else {
                    warn!(command = %cmd, "spawn action: empty command");
                    return;
                };
                match std::process::Command::new(program).args(args).spawn() {
                    Ok(child) => info!(
                        pid = child.id(),
                        command = %cmd,
                        "spawn action fired"
                    ),
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

    /// Re-read the config file and apply the settings that can change
    /// at runtime. A parse/validation error keeps the currently
    /// running config untouched (logged, never fatal) so the user can
    /// fix and save to recover. Settings that can't be hot-applied
    /// (monitor modes, env, input device/keymap setup) are flagged
    /// with a "restart to apply" log when they change.
    pub(crate) fn reload_config(&mut self, path: &std::path::Path) {
        let new = match config::Config::load_from_file(path) {
            Ok(new) => new,
            Err(err) => {
                warn!(error = %err, "config reload failed; keeping the running config");
                return;
            }
        };

        // Settings whose runtime consumers run once at startup.
        if new.monitors != self.config.monitors {
            warn!("monitor config changed; restart Libreland to apply mode/position/scale/primary");
        }
        if new.env != self.config.env {
            warn!("env changed; restart to re-export environment variables");
        }
        if new.startup != self.config.startup {
            info!("startup commands changed; they only run at launch");
        }
        if new.xwayland != self.config.xwayland {
            warn!("xwayland setting changed; restart to start/stop xwayland-satellite");
        }
        let (old_in, new_in) = (&self.config.input, &new.input);
        #[allow(
            clippy::float_cmp,
            reason = "exact change detection — did the configured accel speed differ at all, not 'is it approximately equal'"
        )]
        let input_changed = old_in.repeat_rate != new_in.repeat_rate
            || old_in.repeat_delay != new_in.repeat_delay
            || old_in.keyboard_layout != new_in.keyboard_layout
            || old_in.mouse_accel_profile != new_in.mouse_accel_profile
            || old_in.mouse_accel_speed != new_in.mouse_accel_speed;
        if input_changed {
            warn!(
                "keyboard/pointer input settings changed; restart to apply (focus model applies live)"
            );
        }

        // Hot-apply. Update the layout FIRST (it reflows and sends new
        // configures to clients), the renderer LAST: the renderer's
        // border width drives where it draws the surface and the
        // border ring, so changing it only after the clients have been
        // asked to resize avoids a one-frame window where a new border
        // is drawn around an old-sized buffer. Binds and focus model
        // are read live from `self.config`, so swapping it suffices.
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
            let png = self
                .screenshot
                .as_ref()
                .and_then(|s| s.frozen.get(&geom.name))
                .and_then(|f| {
                    screenshot::encode_region(&f.bytes, f.width, f.height, phys).ok()
                });
            if let Some(bytes) = png {
                self.deliver_screenshot(&bytes, &bind);
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

    /// Save a finished PNG to disk (if a directory is configured) and copy
    /// it to the clipboard (if requested).
    fn deliver_screenshot(&mut self, png: &[u8], bind: &config::ScreenshotBind) {
        if let Some(dir) = &bind.directory {
            let dir = screenshot::expand_dir(dir);
            let filename = screenshot::timestamp_filename(self.local_offset);
            match screenshot::save(&dir, &filename, png) {
                Ok(path) => info!(path = %path.display(), "screenshot saved"),
                Err(err) => warn!(error = %err, dir = %dir.display(), "screenshot save failed"),
            }
        }
        if bind.clipboard {
            self.clipboard.set_image(
                smithay::wayland::selection::SelectionTarget::Clipboard,
                "image/png".to_owned(),
                png.to_vec(),
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
    }

    /// Route a compositor-originated capture's pixels (serviced in the
    /// vblank handler): store + display a freeze snapshot, or encode +
    /// deliver a finished grab.
    fn complete_internal_capture(&mut self, cap: InternalCapture, outcome: render::CaptureOutcome) {
        let render::CaptureOutcome::Shm {
            bytes,
            width,
            height,
            flipped: _,
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
                // The capture already read exactly the wanted region, so
                // encode the whole buffer.
                let full = smithay::utils::Rectangle::from_size(smithay::utils::Size::from((
                    w_i, h_i,
                )));
                match screenshot::encode_region(&bytes, width, height, full) {
                    Ok(png) => self.deliver_screenshot(&png, &bind),
                    Err(err) => warn!(error = %err, "screenshot: PNG encode failed"),
                }
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
    }

    // Wayland frontend bootstrap. Display must exist before the
    // EventLoop because the EventLoop's user-data type
    // (`LoopData`) contains the `Display<State>`.
    info!("phase: creating Wayland Display + substate");
    let mut display: Display<State> = Display::new().context("wayland Display::new failed")?;
    // Wayland init runs *after* the renderer is up — it needs the
    // renderer's per-output descriptors (mode size + compositor
    // position + scale) to create the `wl_output` globals and
    // seed the fractional-scale state. The Display itself, on
    // the other hand, has to exist before the EventLoop because
    // `LoopData` carries it.

    let mut event_loop: EventLoop<LoopData> =
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

    let drm_path = pick_drm_card_path(&initial_devices)?;
    info!(drm_path = %drm_path.display(), "selected DRM device");

    let drm_init = drm::open_display(&mut session, &drm_path, &config.monitors)
        .context("DRM device init failed")?;
    let drm::DrmInit {
        device: drm_device,
        fd: drm_fd,
        notifier: drm_notifier,
        outputs: drm_outputs,
    } = drm_init;

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
    let render_node = renderer.render_drm_node();
    info!(
        count = dmabuf_formats.len(),
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

    // XWayland via xwayland-satellite: a rootless Xwayland that
    // connects to *our* socket as a normal Wayland client (so X
    // windows arrive as ordinary xdg_toplevels) and serves X11 on a
    // display we pick. It must start after WAYLAND_DISPLAY is set (it
    // inherits it) and before X clients. It scales X apps itself via
    // wp_fractional_scale + wp_viewporter; cursors stay consistent
    // because we draw our own over everything and export XCURSOR_*.
    if config.xwayland
        && let Some(disp) = start_xwayland_satellite()
    {
        // SAFETY: same single-threaded-init reasoning as the
        // WAYLAND_DISPLAY set_var above — still pre-event-loop.
        #[allow(
            unsafe_code,
            reason = "set_var is unsafe due to multi-threaded env races; called in single-threaded init before the event loop, same as WAYLAND_DISPLAY"
        )]
        // SAFETY: see #[allow] above.
        unsafe {
            std::env::set_var("DISPLAY", &disp);
        }
        info!(x_display = %disp, "XWayland ready; $DISPLAY exported for X11 clients");
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
        .insert_source(listening_socket, |stream, (), data: &mut LoopData| {
            info!("Wayland: accepting new client");
            if let Err(err) = data
                .display
                .handle()
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
            |_, _, data: &mut LoopData| {
                if let Err(err) = data.display.dispatch_clients(&mut data.state) {
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
    wayland::spawn_startup(&config.startup);

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
                move |_, (), data: &mut LoopData| {
                    let current = stamp(&watch_path);
                    if current != last {
                        // Reload on edit / (re-)creation; ignore deletion
                        // (keep the running config until a file returns).
                        if current.is_some() {
                            info!(path = %watch_path.display(), "config changed; reloading");
                            data.state.reload_config(&watch_path);
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
    let state = State {
        session,
        loop_signal,
        drm_device,
        renderer,
        keyboard,
        config,
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
        fractional_scale_state: wayland_init.fractional_scale_state,
        viewporter_state: wayland_init.viewporter_state,
        data_device_state: wayland_init.data_device_state,
        dmabuf_state: wayland_init.dmabuf_state,
        dmabuf_global: wayland_init.dmabuf_global,
        preferred_scale: wayland_init.preferred_scale,
        layer_shell_state: wayland_init.layer_shell_state,
        layer_outputs: std::collections::HashMap::new(),
        relative_pointer_state: wayland_init.relative_pointer_state,
        pointer_constraints_state: wayland_init.pointer_constraints_state,
        primary_selection_state: wayland_init.primary_selection_state,
        clipboard: clipboard::Selections::default(),
        screencopy_manager: wayland_init.screencopy_manager,
        screencopy_pending: Vec::new(),
        loop_handle: handle.clone(),
        popup_manager: wayland_init.popup_manager,
        kbd_focus_before_layer: None,
        layout,
        drag: None,
        ws_scroll_accum: 0.0,
        screenshot: None,
        screenshot_pending: Vec::new(),
        local_offset,
        ipc: ipc::IpcState::default(),
    };
    let mut loop_data = LoopData { state, display };

    info!("entering event loop — type to generate events, super+shift+e to exit");
    event_loop
        .run(None, &mut loop_data, |data| {
            // Post-batch: broadcast any IPC state changes (focus, windows,
            // workspaces) to subscribers, then flush Wayland clients so
            // their pending outbound messages don't accumulate. A flush
            // failure typically means a client died mid-flight; log and
            // move on rather than crash the compositor.
            ipc::poll_events(&mut data.state);
            if let Err(err) = data.display.flush_clients() {
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
    install_panic_hook();
    Ok(guard)
}

/// Chain a tracing layer onto the default panic hook so panic messages
/// also land in the file log. Stderr-only panic output is invisible
/// during the TTY freeze scenario that motivated file logging in the
/// first place — without this, a panic would crash the compositor with
/// no on-disk record. We delegate to the previous hook so the default
/// stderr + backtrace behaviour is preserved unchanged.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
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

/// Launch `xwayland-satellite` on the first free X display and return
/// that display (e.g. `":1"`) so the caller can export `$DISPLAY`.
/// Returns `None` if no display is free or the binary isn't installed
/// — in both cases X11 support is simply absent (logged), never fatal.
/// The satellite inherits our environment, so it connects to
/// `$WAYLAND_DISPLAY` and inherits `$XCURSOR_*` for its cursor theme.
fn start_xwayland_satellite() -> Option<String> {
    let Some(n) = first_free_x_display() else {
        warn!("no free X display in :0..:32; not starting xwayland-satellite");
        return None;
    };
    let disp = format!(":{n}");
    match std::process::Command::new("xwayland-satellite")
        .arg(&disp)
        .spawn()
    {
        Ok(child) => {
            info!(pid = child.id(), x_display = %disp, "spawned xwayland-satellite");
            Some(disp)
        }
        Err(err) => {
            warn!(
                error = %err,
                "could not start xwayland-satellite (is it installed?); X11 apps unavailable"
            );
            None
        }
    }
}

/// Lowest X display number `N` in `0..=32` whose socket
/// (`/tmp/.X11-unix/XN`) and lock (`/tmp/.XN-lock`) are both absent,
/// i.e. free for a new X server. There's a benign TOCTOU window
/// between this check and the satellite claiming it; on a contended
/// system the satellite simply fails to bind and logs.
fn first_free_x_display() -> Option<u32> {
    (0u32..=32).find(|n| {
        !std::path::Path::new(&format!("/tmp/.X11-unix/X{n}")).exists()
            && !std::path::Path::new(&format!("/tmp/.X{n}-lock")).exists()
    })
}

/// Pick a `/dev/dri/cardN` node from a udev enumeration — render nodes
/// (`renderD128`) come through the same DRM subsystem and we
/// explicitly don't want them for modesetting. First card wins for
/// now; multi-GPU is a later milestone.
fn pick_drm_card_path<T>(devices: &[(T, std::path::PathBuf)]) -> Result<std::path::PathBuf> {
    devices
        .iter()
        .find(|(_, p)| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("card"))
        })
        .map(|(_, p)| p.clone())
        .context("no /dev/dri/cardN device enumerated by udev — no display to drive")
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
    handle: &smithay::reexports::calloop::LoopHandle<'_, LoopData>,
    session_notifier: LibSeatSessionNotifier,
    udev: UdevBackend,
    drm_notifier: smithay::backend::drm::DrmDeviceNotifier,
    libinput_backend: LibinputInputBackend,
) -> Result<()> {
    handle
        .insert_source(session_notifier, |event, (), _data| match event {
            smithay::backend::session::Event::PauseSession => warn!("session paused"),
            smithay::backend::session::Event::ActivateSession => info!("session activated"),
        })
        .map_err(|e| anyhow::anyhow!("failed to insert session source: {e}"))?;

    handle
        .insert_source(udev, |event, (), _data| match event {
            UdevEvent::Added { device_id, path } => {
                info!(device_id, path = %path.display(), "udev: device added");
            }
            UdevEvent::Removed { device_id } => {
                info!(device_id, "udev: device removed");
            }
            UdevEvent::Changed { device_id } => {
                debug!(device_id, "udev: device changed");
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to insert udev source: {e}"))?;

    handle
        .insert_source(
            drm_notifier,
            |event, _meta, data: &mut LoopData| match event {
                smithay::backend::drm::DrmEvent::VBlank(crtc) => {
                    // Snapshot every laid-out window's placement so
                    // the borrow on `layout` ends before the mut
                    // borrow on `renderer` starts. WlSurface clones
                    // are Arc-backed — cheap. Layout returns
                    // placements in draw order (tiled tree first,
                    // floats next, in-transit on top) along with
                    // each window's focused flag so the renderer
                    // picks active vs inactive border colour.
                    let focused = data
                        .state
                        .seat
                        .get_keyboard()
                        .and_then(|k| k.current_focus());
                    // Workspace slide spec (None when disabled). Clear
                    // finished slides, then emit (both workspaces during
                    // one, the active one otherwise).
                    let ws_anim = data.state.config.animations.workspace;
                    let slide = (data.state.config.animations.enabled && ws_anim.enabled)
                        .then_some(ws_anim);
                    data.state.layout.tick_transitions(slide);
                    let placements = data.state.layout.placements(focused.as_ref(), slide);
                    let layer_placements = data.state.snapshot_layer_placements();
                    let popup_placements = data
                        .state
                        .snapshot_popup_placements(&placements, &layer_placements);
                    // Drain any pending screencopy captures for the
                    // output this CRTC drives, so the renderer can read
                    // them off this frame's framebuffer.
                    let out_name = data.state.renderer.output_name_for_crtc(crtc);
                    // Client zwlr_screencopy captures + compositor-internal
                    // screenshot captures both ride this output's frame.
                    let mut captures: Vec<screencopy::PendingCapture> = Vec::new();
                    let mut internal: Vec<InternalCapture> = Vec::new();
                    if let Some(name) = out_name.as_deref() {
                        let pending = std::mem::take(&mut data.state.screencopy_pending);
                        let (mine, rest): (Vec<_>, Vec<_>) =
                            pending.into_iter().partition(|p| p.output == name);
                        data.state.screencopy_pending = rest;
                        captures = mine;

                        let ipending = std::mem::take(&mut data.state.screenshot_pending);
                        let (imine, irest): (Vec<_>, Vec<_>) =
                            ipending.into_iter().partition(|c| c.output == name);
                        data.state.screenshot_pending = irest;
                        internal = imine;
                    }
                    // Client specs first (shm or zero-copy dmabuf blit),
                    // then internal screenshot specs (always CPU shm).
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
                    // Hide our cursor while a game holds a pointer lock (it
                    // draws its own), for a client capture that asked for no
                    // cursor, and for an internal screenshot capture unless
                    // its bind set `show_cursor`.
                    let capture_hides_cursor =
                        !captures.is_empty() && captures.iter().all(|c| !c.overlay_cursor);
                    let internal_hides_cursor =
                        !internal.is_empty() && internal.iter().all(|c| !c.show_cursor);
                    let hide_cursor = data.state.pointer_locked()
                        || capture_hides_cursor
                        || internal_hides_cursor;
                    let client_n = captures.len();
                    match data.state.renderer.render_for_crtc(
                        crtc,
                        &placements,
                        &layer_placements,
                        &popup_placements,
                        hide_cursor,
                        &specs,
                    ) {
                        Ok(mut results) => {
                            // Trailing results belong to the internal specs.
                            let internal_results = results.split_off(client_n.min(results.len()));
                            for (pending, captured) in captures.iter().zip(results) {
                                screencopy::complete(pending, captured);
                            }
                            for (cap, outcome) in internal.into_iter().zip(internal_results) {
                                data.state.complete_internal_capture(cap, outcome);
                            }
                        }
                        Err(err) => {
                            // Don't kill the event loop on a render
                            // hiccup — log and let the next vblank try
                            // again. A persistent failure on one CRTC
                            // freezes that output but leaves the others
                            // (and the exit hotkey) responsive. Fail any
                            // captures that were riding this frame.
                            warn!(error = %err, ?crtc, "render_for_crtc failed on vblank");
                            for pending in &captures {
                                screencopy::complete(pending, render::CaptureOutcome::Failed);
                            }
                            if !internal.is_empty() {
                                warn!("screenshot: capture dropped (render failed)");
                            }
                        }
                    }
                }
                smithay::backend::drm::DrmEvent::Error(err) => {
                    warn!(error = %err, "drm: event-source error");
                }
            },
        )
        .map_err(|e| anyhow::anyhow!("failed to insert drm source: {e}"))?;

    handle
        .insert_source(libinput_backend, |event, (), data: &mut LoopData| {
            log_input_event(&event);
            match event {
                InputEvent::Keyboard { event: ke } => data.state.handle_key(&ke),
                InputEvent::PointerMotion { event: pm } => {
                    data.state
                        .forward_pointer_motion::<LibinputInputBackend>(&pm);
                }
                InputEvent::PointerButton { event: pb } => {
                    data.state
                        .forward_pointer_button(pb.button_code(), pb.state(), pb.time_msec());
                }
                InputEvent::PointerAxis { event: pa } => {
                    data.state.forward_pointer_axis::<LibinputInputBackend>(&pa);
                }
                InputEvent::DeviceAdded { mut device } => {
                    apply_input_config(&mut device, &data.state.config.input);
                }
                _ => {}
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to insert libinput source: {e}"))?;

    Ok(())
}

/// Log a single libinput event. Keyboard / pointer events are what we
/// care about for the TTY sanity check; touch, gestures and tablet
/// variants are intentionally elided here and will be wired up later
/// when there's something to do with them.
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
