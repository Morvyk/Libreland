//! Native Xwayland integration — rootless Xwayland + an in-process XWM.
//!
//! We spawn `Xwayland -rootless` ourselves (via smithay) and act as its
//! X11 window manager over the `-wm` socket, instead of delegating to
//! `xwayland-satellite`. Owning the WM fixes the three chronic
//! satellite problems:
//!
//! - **Cursors**: Xwayland is a direct Wayland client of ours, so X
//!   apps' cursors arrive as ordinary `wl_pointer.set_cursor` calls and
//!   flow through the exact same path as native clients (no satellite
//!   re-encoding). The X *root* cursor (shown when an X app doesn't set
//!   one) is uploaded from our own cursor theme at WM start.
//!
//! - **Scale/DPI**: the Xwayland client gets a *client scale* equal to
//!   the primary output's fractional scale (smithay translates every
//!   coordinate — configures, pointer, outputs — between our logical
//!   space and Xwayland's pixel space). X windows therefore render
//!   their buffers at **physical** resolution (a 1.5× output means a
//!   tile of 800 logical px is configured as 1200 X px), which the
//!   renderer draws 1:1 — pixel-sharp, no upscale. X apps are told to
//!   match via XSETTINGS `Xft/DPI` (96 × scale, e.g. 144 at 1.5×), so
//!   toolkits draw at the right visual size in that space.
//!
//! - **Control**: window management is first-class — X11 windows enter
//!   the same tiling layout as xdg toplevels ([`WindowSurface::X11`]),
//!   get real tile configures (position + size), fullscreen/maximize
//!   requests route through the same layout paths, and clipboard /
//!   primary selection are bridged in-process.
//!
//! # Lifecycle
//!
//! [`spawn_xwayland`] starts the server and registers a calloop source;
//! when Xwayland signals readiness, [`State::on_xwayland_ready`]
//! attaches the WM ([`X11Wm`]), publishes XSETTINGS, and sets the root
//! cursor. `$DISPLAY` is known at spawn time (before the event loop, so
//! the env export is still safe) — X clients can connect immediately;
//! Xwayland queues them until ready.
//!
//! # Window mapping
//!
//! An X11 window becomes a layout window in two steps that can race:
//! the WM allows the map (`map_window_request` → `set_mapped`), and
//! Xwayland associates a `wl_surface` with the window (the
//! `xwayland_shell` protocol). Only when **both** have happened is
//! there something to manage, so `try_manage_x11` runs from both hooks
//! and is idempotent. Override-redirect windows (menus, tooltips,
//! dropdowns) bypass the WM by design; they never enter the layout —
//! they're tracked in [`State::x11_or_windows`] and rendered through
//! the same topmost path as xdg popups, at whatever global position
//! the client chose.

use std::os::fd::OwnedFd;
use std::process::Stdio;

use smithay::reexports::calloop::LoopHandle;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{Client, DisplayHandle};
use smithay::utils::{Logical, Physical, Point, Rectangle, SERIAL_COUNTER, Size};
use smithay::wayland::selection::SelectionTarget;
use smithay::wayland::selection::data_device::{
    clear_data_device_selection, set_data_device_selection,
};
use smithay::wayland::selection::primary_selection::{
    clear_primary_selection, set_primary_selection,
};
use smithay::wayland::xwayland_shell::{XWaylandShellHandler, XWaylandShellState};
use smithay::xwayland::xwm::{Reorder, ResizeEdge, XwmId, settings::Value};
use smithay::xwayland::{
    X11Surface, X11Wm, XWayland, XWaylandClientData, XWaylandEvent, XwmHandler,
};
use tracing::{debug, info, warn};

use crate::State;
use crate::layout::{FillMode, WindowSurface};

/// Which selections (clipboard / primary) the X side currently owns.
/// While a target is owned by X, Wayland-side paste requests for it are
/// routed through the XWM (Xwayland streams the bytes from the X owner)
/// instead of the compositor's clipboard cache — see
/// `SelectionHandler::send_selection` in wayland.rs.
#[derive(Debug, Default)]
pub(crate) struct X11SelectionOwnership {
    clipboard: bool,
    primary: bool,
}

impl X11SelectionOwnership {
    pub(crate) fn owns(&self, ty: SelectionTarget) -> bool {
        match ty {
            SelectionTarget::Clipboard => self.clipboard,
            SelectionTarget::Primary => self.primary,
        }
    }

    pub(crate) fn set(&mut self, ty: SelectionTarget, owned: bool) {
        match ty {
            SelectionTarget::Clipboard => self.clipboard = owned,
            SelectionTarget::Primary => self.primary = owned,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.clipboard = false;
        self.primary = false;
    }
}

/// Everything a caller must hold onto after [`spawn_xwayland`]: the
/// source registration (removing it tears the server down), the
/// Xwayland client (carries the scale override), and the display
/// string for `$DISPLAY`. Startup threads these into the `State`
/// literal; the live toggle stores them on the existing `State`.
pub(crate) struct SpawnedXwayland {
    pub(crate) source: smithay::reexports::calloop::RegistrationToken,
    pub(crate) client: Client,
    pub(crate) display: String,
}

/// Spawn a rootless Xwayland bound to our display and register its
/// readiness source on the event loop. The WM attaches later, when the
/// server reports ready (see [`State::on_xwayland_ready`]) — callable
/// before `State` exists (startup) because calloop callbacks only need
/// `State` at *dispatch* time.
///
/// The Xwayland *client scale* is set to `scale` immediately — before
/// any X11 surface exists — so every coordinate smithay relays is
/// consistently translated from the start (see the module docs for the
/// scale model). Returns `None` when Xwayland can't be spawned (not
/// installed / no free display); X11 support is simply absent, never
/// fatal — matching how the satellite used to degrade.
pub(crate) fn spawn_xwayland(
    loop_handle: &LoopHandle<'static, State>,
    dh: &DisplayHandle,
    scale: f64,
) -> Option<SpawnedXwayland> {
    let (xwayland, client) = match XWayland::spawn(
        dh,
        None,
        std::iter::empty::<(String, String)>(),
        true,
        Stdio::null(),
        Stdio::null(),
        |_| (),
    ) {
        Ok(pair) => pair,
        Err(err) => {
            warn!(error = %err, "could not spawn Xwayland (is it installed?); X11 apps unavailable");
            return None;
        }
    };
    let x_display = format!(":{}", xwayland.display_number());

    // The client-scale mapping is what makes X windows render at
    // physical resolution: Xwayland sees a world already multiplied by
    // the output scale and commits buffers at that size, which the
    // renderer then draws 1:1.
    if let Some(data) = client.get_data::<XWaylandClientData>() {
        data.compositor_state.set_client_scale(scale);
    }

    let ready_client = client.clone();
    let source = loop_handle.insert_source(xwayland, move |event, (), state| match event {
        XWaylandEvent::Ready {
            x11_socket,
            display_number,
        } => {
            info!(x_display = display_number, "Xwayland ready; attaching WM");
            state.on_xwayland_ready(x11_socket, ready_client.clone());
        }
        XWaylandEvent::Error => {
            warn!("Xwayland failed to start; X11 support is gone until an `xwayland` toggle or restart");
            state.teardown_xwayland();
        }
    });
    match source {
        Ok(token) => {
            info!(x_display = %x_display, "spawned Xwayland (native XWM)");
            Some(SpawnedXwayland {
                source: token,
                client,
                display: x_display,
            })
        }
        Err(err) => {
            warn!(error = %err, "failed to register the Xwayland event source");
            None
        }
    }
}

impl State {
    /// Xwayland reported ready: attach the X11 window manager, publish
    /// XSETTINGS (DPI + cursor theme), and install the root cursor.
    pub(crate) fn on_xwayland_ready(&mut self, x11_socket: std::os::unix::net::UnixStream, client: Client) {
        let mut xwm = match X11Wm::start_wm(self.loop_handle.clone(), x11_socket, client) {
            Ok(wm) => wm,
            Err(err) => {
                warn!(error = %err, "failed to attach the X11 window manager; X11 apps unavailable");
                return;
            }
        };
        let scale = self.renderer.primary_scale();
        apply_xsettings(&mut xwm, scale);
        // Root cursor: what X windows show when the app doesn't set its
        // own. Loaded from our theme at the same physical size the
        // compositor draws, so hovering an idle X window looks identical
        // to hovering a native one. The X pixel space is physical-sized
        // (client scale), so load at logical-nominal × scale.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "cursor size is a small positive number (theme sizes are tens of pixels)"
        )]
        let cursor_px = (f64::from(crate::cursor::configured_size()) * scale).round() as u32;
        if let Some(image) = crate::cursor::load_default_cursor(cursor_px) {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "cursor images are far smaller than u16::MAX on every axis, with non-negative sizes/hotspots"
            )]
            if let Err(err) = xwm.set_cursor(
                &image.rgba,
                Size::<u16, Logical>::from((image.width as u16, image.height as u16)),
                Point::<u16, Logical>::from((image.xhot as u16, image.yhot as u16)),
            ) {
                warn!(error = %err, "failed to set the X root cursor");
            }
        }
        self.xwm = Some(xwm);
    }

    /// Drop every trace of a dead (or disabled) Xwayland *except* the WM
    /// handle: its windows (pulled from the layout like closed
    /// toplevels), the `$DISPLAY` we hand to children, and the server
    /// itself (removing the readiness source drops the `XWayland`
    /// instance, which terminates the process). Safe to call twice.
    ///
    /// `self.xwm` is deliberately NOT cleared here: the WM's calloop
    /// source may still hold queued X events (the dying server's unmap
    /// storm), and every one of them resolves `xwm_state` — which must
    /// keep finding the WM until the source delivers `Closed` and
    /// removes itself. `XwmHandler::disconnected` (that `Closed` path)
    /// is the one place the handle is dropped.
    pub(crate) fn teardown_xwayland(&mut self) {
        self.xwayland_client = None;
        self.xwayland_display = None;
        if let Some(token) = self.xwayland_source.take() {
            self.loop_handle.remove(token);
        }
        self.x11_kbd_focus = None;
        self.x11_owns_selection.clear();
        self.x11_or_windows.clear();
        for (_, wl_surface) in std::mem::take(&mut self.x11_windows) {
            self.layout.remove(&wl_surface);
            self.ipc.forget(&wl_surface);
        }
        self.refresh_pointer_focus();
        self.queue_redraw_all();
    }

    /// Live `xwayland` toggle / scale change: re-publish XSETTINGS and
    /// the client scale, then re-push every X11 window's configure so
    /// Xwayland re-maps them into the new pixel space. Called from the
    /// config-reload path when the primary scale changes.
    pub(crate) fn update_xwayland_scale(&mut self) {
        let scale = self.renderer.primary_scale();
        if let Some(client) = &self.xwayland_client
            && let Some(data) = client.get_data::<XWaylandClientData>()
        {
            data.compositor_state.set_client_scale(scale);
        }
        if let Some(xwm) = &mut self.xwm {
            apply_xsettings(xwm, scale);
        }
        let surfaces: Vec<WlSurface> = self
            .x11_windows
            .iter()
            .map(|(_, wl)| wl.clone())
            .collect();
        for wl_surface in surfaces {
            self.layout.reconfigure(&wl_surface);
        }
    }

    /// Both map preconditions met? — manage the window. Called from
    /// every hook that can complete the pair (map notify, surface
    /// association), so it must be — and is — idempotent: a window
    /// already managed (or with no `wl_surface` yet) is left alone.
    fn try_manage_x11(&mut self, window: &X11Surface) {
        if window.is_override_redirect() {
            self.track_x11_or(window);
            return;
        }
        let Some(wl_surface) = window.wl_surface() else {
            return;
        };
        if self
            .x11_windows
            .iter()
            .any(|(w, _)| w.window_id() == window.window_id())
        {
            return;
        }
        info!(
            window = window.window_id(),
            class = %window.class(),
            "xwayland: managing X11 window"
        );
        self.x11_windows.push((window.clone(), wl_surface.clone()));
        // Same insert path as a new xdg toplevel: tile at the cursor,
        // animate open, take keyboard focus.
        let (cx, cy) = self.renderer.cursor_pos();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "cursor coords are clamped to layout_bounds (i32) in Renderer::on_pointer_motion"
        )]
        let cursor = Point::<i32, Physical>::from((cx as i32, cy as i32));
        self.layout.insert(
            WindowSurface::X11 {
                surface: Box::new(window.clone()),
                wl_surface: wl_surface.clone(),
            },
            Some(cursor),
        );
        // A window that asked for fullscreen before mapping (games
        // setting a mode) gets it immediately, through the same layout
        // path an xdg fullscreen request takes.
        if window.is_fullscreen() {
            self.layout.toggle_fullscreen(&wl_surface);
        }
        self.renderer.mark_open(&wl_surface);
        if let Some(kbd) = self.seat.get_keyboard() {
            kbd.set_focus(self, Some(wl_surface), SERIAL_COUNTER.next_serial());
        }
        self.queue_redraw_all();
    }

    /// Track a mapped override-redirect window (menu/tooltip). Needs
    /// the `wl_surface` (for render + hit-test), so like managed
    /// windows it's attempted from both the map and association hooks.
    fn track_x11_or(&mut self, window: &X11Surface) {
        let Some(wl_surface) = window.wl_surface() else {
            return;
        };
        if self
            .x11_or_windows
            .iter()
            .any(|(w, _)| w.window_id() == window.window_id())
        {
            return;
        }
        debug!(window = window.window_id(), "xwayland: override-redirect mapped");
        self.x11_or_windows.push((window.clone(), wl_surface));
        self.queue_redraw_all();
    }

    /// A window is gone (unmapped or destroyed) — pull it from
    /// whichever list holds it and clean up exactly like a destroyed
    /// xdg toplevel (close animation, IPC id, focus, pointer refresh).
    fn unmanage_x11(&mut self, window: &X11Surface) {
        let id = window.window_id();
        if let Some(pos) = self
            .x11_or_windows
            .iter()
            .position(|(w, _)| w.window_id() == id)
        {
            self.x11_or_windows.remove(pos);
            self.refresh_pointer_focus();
            self.queue_redraw_all();
            return;
        }
        let Some(pos) = self
            .x11_windows
            .iter()
            .position(|(w, _)| w.window_id() == id)
        else {
            return;
        };
        let (_, wl_surface) = self.x11_windows.remove(pos);
        info!(window = id, "xwayland: X11 window unmanaged");
        self.renderer.start_close(&wl_surface);
        self.layout.remove(&wl_surface);
        self.ipc.forget(&wl_surface);
        if let Some(kbd) = self.seat.get_keyboard() {
            let was_focused = kbd.current_focus().as_ref() == Some(&wl_surface);
            if was_focused {
                kbd.set_focus(self, None, SERIAL_COUNTER.next_serial());
            }
        }
        self.refresh_pointer_focus();
        self.queue_redraw_all();
    }

    /// Mirror a keyboard-focus change onto the X11 side. Our focus type
    /// is `WlSurface`, which delivers `wl_keyboard` events — but X
    /// clients only *accept* keys once the WM sets X input focus
    /// (`SetInputFocus` / `WM_TAKE_FOCUS`, per ICCCM input mode). Called
    /// from `SeatHandler::focus_changed` with the new focus; no-op when
    /// the X11 focus target didn't change, so ordinary Wayland↔Wayland
    /// focus moves don't touch the X connection at all.
    pub(crate) fn sync_x11_focus(&mut self, focused: Option<&WlSurface>) {
        let new = focused.and_then(|surface| {
            self.x11_windows
                .iter()
                .find(|(_, wl)| wl == surface)
                .map(|(window, _)| window.clone())
        });
        let same = match (&self.x11_kbd_focus, &new) {
            (Some(a), Some(b)) => a.window_id() == b.window_id(),
            (None, None) => true,
            _ => false,
        };
        if same {
            return;
        }
        if let Some(old) = self.x11_kbd_focus.take() {
            let _ = old.x11_unfocus();
            let _ = old.set_activated(false);
        }
        if let Some(window) = &new {
            let _ = window.x11_take_focus();
            let _ = window.set_activated(true);
            // Keep the X stacking order in step with focus so apps that
            // position override-redirect popups relative to "their"
            // window agree with what's actually on top.
            if let Some(xwm) = &mut self.xwm {
                let _ = xwm.raise_window(window);
            }
        }
        self.x11_kbd_focus = new;
    }

    /// The X11 window behind a layout surface, if any. Used by the IPC
    /// window info to read class/title where xdg role data doesn't exist.
    pub(crate) fn x11_window_for(&self, surface: &WlSurface) -> Option<&X11Surface> {
        self.x11_windows
            .iter()
            .find(|(_, wl)| wl == surface)
            .map(|(window, _)| window)
    }
}

/// Publish the XSETTINGS X apps read for scale + cursors:
/// `Xft/DPI` in 1/1024ths (96 × the output scale — the "144 DPI at
/// 1.5×" the compositor advertises to fractional-aware Wayland
/// clients), and the cursor theme/size matching the compositor's own
/// (`$XCURSOR_THEME` / physical cursor pixels), so X apps request
/// cursors that render identical to native ones.
fn apply_xsettings(xwm: &mut X11Wm, scale: f64) {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "96 * scale * 1024 for sane scales (0.5..4) is far below i32::MAX"
    )]
    let dpi = (96.0 * scale * 1024.0).round() as i32;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "cursor size is a small positive number (theme sizes are tens of pixels)"
    )]
    let cursor_px = (f64::from(crate::cursor::configured_size()) * scale).round() as i32;
    let mut settings: Vec<(String, Value)> = vec![
        ("Xft/DPI".to_owned(), Value::Integer(dpi)),
        ("Gtk/CursorThemeSize".to_owned(), Value::Integer(cursor_px)),
    ];
    if let Ok(theme) = std::env::var("XCURSOR_THEME") {
        settings.push(("Gtk/CursorThemeName".to_owned(), Value::String(theme)));
    }
    if let Err(err) = xwm.set_xsettings(settings.into_iter()) {
        warn!(error = %err, "failed to publish XSETTINGS (X apps fall back to 96 DPI)");
    }
}

impl XWaylandShellHandler for State {
    fn xwayland_shell_state(&mut self) -> &mut XWaylandShellState {
        &mut self.xwayland_shell_state
    }

    fn surface_associated(&mut self, _xwm_id: XwmId, _surface: WlSurface, window: X11Surface) {
        // Second half of the map handshake (see module docs) — the
        // window may or may not be mapped yet; try_manage sorts it out.
        self.try_manage_x11(&window);
    }
}

impl XwmHandler for State {
    fn xwm_state(&mut self, _xwm: XwmId) -> &mut X11Wm {
        self.xwm
            .as_mut()
            .expect("XWM callbacks only fire while the WM connection is alive")
    }

    fn new_window(&mut self, _xwm: XwmId, _window: X11Surface) {
        // Created ≠ mapped; nothing to do until the map request.
    }

    fn new_override_redirect_window(&mut self, _xwm: XwmId, _window: X11Surface) {
        // Tracked when actually mapped (mapped_override_redirect_window).
    }

    fn map_window_request(&mut self, _xwm: XwmId, window: X11Surface) {
        // Allow the map. This kicks Xwayland into creating + associating
        // the wl_surface; management happens when that lands.
        if let Err(err) = window.set_mapped(true) {
            warn!(window = window.window_id(), error = %err, "xwayland: map failed");
        }
        self.try_manage_x11(&window);
    }

    fn map_window_notify(&mut self, _xwm: XwmId, window: X11Surface) {
        self.try_manage_x11(&window);
    }

    fn mapped_override_redirect_window(&mut self, _xwm: XwmId, window: X11Surface) {
        self.track_x11_or(&window);
    }

    fn unmapped_window(&mut self, _xwm: XwmId, window: X11Surface) {
        self.unmanage_x11(&window);
    }

    fn destroyed_window(&mut self, _xwm: XwmId, window: X11Surface) {
        self.unmanage_x11(&window);
    }

    #[allow(
        clippy::cast_possible_wrap,
        reason = "X11 sizes are 16-bit on the wire; u32 → i32 cannot wrap"
    )]
    fn configure_request(
        &mut self,
        _xwm: XwmId,
        window: X11Surface,
        x: Option<i32>,
        y: Option<i32>,
        w: Option<u32>,
        h: Option<u32>,
        _reorder: Option<Reorder>,
    ) {
        let id = window.window_id();
        let managed = self.x11_windows.iter().find(|(win, _)| win.window_id() == id);
        match managed {
            // Not ours (yet): pre-map windows sizing themselves (games
            // picking a resolution) get exactly what they asked for, so
            // the size they see before mapping matches reality.
            None => {
                let mut rect = window.geometry();
                if let Some(x) = x {
                    rect.loc.x = x;
                }
                if let Some(y) = y {
                    rect.loc.y = y;
                }
                if let Some(w) = w {
                    rect.size.w = w as i32;
                }
                if let Some(h) = h {
                    rect.size.h = h as i32;
                }
                if let Err(err) = window.configure(rect) {
                    debug!(window = id, %err, "xwayland: pre-map configure failed");
                }
            }
            // Managed: the layout owns geometry. A floating window may
            // resize itself (dialogs growing to fit content); a tiled
            // one gets its cell re-asserted so the client can't drift.
            Some((_, wl_surface)) => {
                let wl_surface = wl_surface.clone();
                let entry = self
                    .layout
                    .window_entries()
                    .into_iter()
                    .find(|e| e.surface == wl_surface);
                if let Some(entry) = entry
                    && entry.floating
                    && entry.fill == FillMode::Normal
                    && (w.is_some() || h.is_some())
                {
                    let border = self.layout.border_width();
                    let mut rect = entry.rect;
                    if let Some(w) = w {
                        rect.size.w = w as i32 + 2 * border;
                    }
                    if let Some(h) = h {
                        rect.size.h = h as i32 + 2 * border;
                    }
                    self.layout.set_floating_rect(&wl_surface, rect);
                }
                self.layout.reconfigure(&wl_surface);
                self.queue_redraw_all();
            }
        }
    }

    fn configure_notify(
        &mut self,
        _xwm: XwmId,
        window: X11Surface,
        _geometry: Rectangle<i32, Logical>,
        _above: Option<smithay::xwayland::xwm::X11Window>,
    ) {
        // Override-redirect windows position themselves (a menu opening,
        // a tooltip following the pointer) — the render snapshot reads
        // geometry live, it just needs a frame.
        let id = window.window_id();
        if self.x11_or_windows.iter().any(|(w, _)| w.window_id() == id) {
            self.queue_redraw_all();
        }
    }

    fn maximize_request(&mut self, _xwm: XwmId, window: X11Surface) {
        self.x11_fill_request(&window, FillMode::Maximized, true);
    }

    fn unmaximize_request(&mut self, _xwm: XwmId, window: X11Surface) {
        self.x11_fill_request(&window, FillMode::Maximized, false);
    }

    fn fullscreen_request(&mut self, _xwm: XwmId, window: X11Surface) {
        self.x11_fill_request(&window, FillMode::Fullscreen, true);
    }

    fn unfullscreen_request(&mut self, _xwm: XwmId, window: X11Surface) {
        self.x11_fill_request(&window, FillMode::Fullscreen, false);
    }

    fn resize_request(&mut self, _xwm: XwmId, _window: X11Surface, _button: u32, _edge: ResizeEdge) {
        // Interactive resize is compositor-driven here (Super+RMB), same
        // as for xdg toplevels — client-initiated resize drags are ignored.
    }

    fn move_request(&mut self, _xwm: XwmId, _window: X11Surface, _button: u32) {
        // Same as resize_request: moves are compositor-driven (Super+LMB).
    }

    fn allow_selection_access(&mut self, _xwm: XwmId, _selection: SelectionTarget) -> bool {
        // The only client on the WM connection is our own Xwayland.
        true
    }

    fn send_selection(
        &mut self,
        _xwm: XwmId,
        selection: SelectionTarget,
        mime_type: String,
        fd: OwnedFd,
    ) {
        // An X client pastes while a Wayland client owns the selection:
        // serve it from the compositor's clipboard cache — the same
        // bytes any Wayland paster would get.
        crate::clipboard::on_send_selection(self, selection, &mime_type, fd);
    }

    fn new_selection(&mut self, _xwm: XwmId, selection: SelectionTarget, mime_types: Vec<String>) {
        // An X client copied. Drop our cache of the previous Wayland
        // selection (it's stale now) and make the compositor the
        // Wayland-side owner offering the X mime types; Wayland pastes
        // are routed back through the XWM by the flag (see
        // `SelectionHandler::send_selection` in wayland.rs).
        debug!(?selection, ?mime_types, "xwayland: X client took the selection");
        crate::clipboard::on_new_selection(self, selection, None);
        self.x11_owns_selection.set(selection, true);
        let dh = self.display_handle.clone();
        let seat = self.seat.clone();
        match selection {
            SelectionTarget::Clipboard => {
                set_data_device_selection::<State>(&dh, &seat, mime_types, ());
            }
            SelectionTarget::Primary => {
                set_primary_selection::<State>(&dh, &seat, mime_types, ());
            }
        }
    }

    fn cleared_selection(&mut self, _xwm: XwmId, selection: SelectionTarget) {
        // The X owner went away. Only clear the Wayland side if the X
        // side still owned it — a Wayland client re-claiming the
        // selection also triggers this (we replaced the X owner), and
        // clearing then would kill the *new* selection.
        if !self.x11_owns_selection.owns(selection) {
            return;
        }
        self.x11_owns_selection.set(selection, false);
        let dh = self.display_handle.clone();
        let seat = self.seat.clone();
        match selection {
            SelectionTarget::Clipboard => clear_data_device_selection::<State>(&dh, &seat),
            SelectionTarget::Primary => clear_primary_selection::<State>(&dh, &seat),
        }
    }

    fn disconnected(&mut self, xwm: XwmId) {
        // A stale WM instance (already replaced by a rapid disable →
        // enable) closing must not tear down its successor's state.
        if self.xwm.as_ref().is_some_and(|wm| wm.id() != xwm) {
            debug!("stale Xwayland WM connection closed; current instance unaffected");
            return;
        }
        warn!("Xwayland disconnected; X11 support is gone until an `xwayland` toggle or restart");
        // The source has delivered `Closed` and removes itself — no
        // further event can resolve `xwm_state`, so the handle can go.
        self.xwm = None;
        self.teardown_xwayland();
    }
}

impl State {
    /// Shared handler for the four `NET_WM_STATE` fill requests: flip the
    /// layout fill through the same toggles the xdg + IPC paths use
    /// (which re-push configures, so `NET_WM_STATE` is echoed back to
    /// the client by `push_x11_configure`).
    fn x11_fill_request(&mut self, window: &X11Surface, mode: FillMode, want: bool) {
        let id = window.window_id();
        let Some((_, wl_surface)) = self
            .x11_windows
            .iter()
            .find(|(win, _)| win.window_id() == id)
        else {
            return;
        };
        let wl_surface = wl_surface.clone();
        let Some(entry) = self
            .layout
            .window_entries()
            .into_iter()
            .find(|e| e.surface == wl_surface)
        else {
            return;
        };
        let has = entry.fill == mode;
        if has == want {
            return;
        }
        let changed = match mode {
            FillMode::Fullscreen => self.layout.toggle_fullscreen(&wl_surface),
            FillMode::Maximized => self.layout.toggle_maximized(&wl_surface),
            FillMode::Normal => false,
        };
        if changed {
            self.queue_redraw_all();
        }
    }
}

smithay::delegate_xwayland_shell!(State);
