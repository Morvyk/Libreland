//! Control IPC — a JSON-over-Unix-socket protocol plus the
//! `libreland msg` client.
//!
//! The compositor listens on `$XDG_RUNTIME_DIR/libreland-<display>.sock`
//! (exported as `$LIBRELAND_SOCKET` so children and the CLI find it).
//! Each connection is line-delimited JSON: one [`Request`] object per
//! line in, one [`Reply`] (a serialized `Result<Response, String>`) per
//! line out. Bars, scripts, and the bundled `libreland msg` subcommand
//! all speak this protocol.
//!
//! This module is split into three parts: the wire **protocol** (shared
//! by client and server), the **server** (socket wiring + request
//! dispatch, compositor-side), and the **client** (the clap CLI that
//! `libreland msg …` runs).

use serde::{Deserialize, Serialize};

// ======================================================================
// Protocol — shared by the client and the compositor.
// ======================================================================

/// A request from a client to the compositor. Serialized internally
/// tagged on a `cmd` field, e.g. `{"cmd":"windows"}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "kebab-case")]
pub enum Request {
    // --- queries ---
    /// Compositor name + version.
    Version,
    /// Every connected output.
    Outputs,
    /// Every workspace across every output.
    Workspaces,
    /// Every live layer-shell surface (bars, launchers, rofi, …) with its
    /// namespace — for discovering names to use in `blur.layers` rules.
    Layers,
    /// Every managed window.
    Windows,
    /// The keyboard-focused window, if any.
    FocusedWindow,
    /// The configured keybindings.
    Binds,
    /// Render a thumbnail of a window (by id) to a PNG and return its path.
    /// Works for any window regardless of workspace or output. `max` caps
    /// the longest side in pixels (downscaled only; default 512).
    CaptureWindow {
        id: u64,
        #[serde(default)]
        max: Option<i32>,
    },

    // --- actions ---
    /// Focus a window by id (revealing its workspace first).
    FocusWindow { id: u64 },
    /// Close a window (the focused one if `id` is omitted).
    Close {
        #[serde(default)]
        id: Option<u64>,
    },
    /// Toggle a window between tiled and floating.
    ToggleFloating {
        #[serde(default)]
        id: Option<u64>,
    },
    /// Toggle a window's fullscreen state.
    ToggleFullscreen {
        #[serde(default)]
        id: Option<u64>,
    },
    /// Toggle a window's maximized state.
    ToggleMaximized {
        #[serde(default)]
        id: Option<u64>,
    },
    /// Switch the active workspace of an output (the primary if
    /// `output` is omitted).
    FocusWorkspace {
        #[serde(default)]
        output: Option<String>,
        target: WorkspaceTarget,
    },
    /// Move a window (the focused one if `id` is omitted) to a workspace
    /// on its own output and follow it.
    MoveToWorkspace {
        #[serde(default)]
        id: Option<u64>,
        target: WorkspaceTarget,
    },
    /// Spawn a child process. `command` is an argv (program + args).
    Spawn { command: Vec<String> },
    /// Re-read the config file now.
    Reload,
    /// Exit the compositor.
    Exit,

    // --- events ---
    /// Subscribe to the live event stream. The connection stays open and
    /// the compositor pushes one [`Event`] per line as state changes.
    /// `events` filters to specific kinds; empty = all. On subscribe the
    /// current focus + workspaces are sent immediately as a snapshot.
    Subscribe {
        #[serde(default)]
        events: Vec<EventKind>,
    },
}

/// One kind of [`Event`], used to filter a subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
#[clap(rename_all = "kebab-case")]
pub enum EventKind {
    WindowOpened,
    WindowClosed,
    WindowFocused,
    WorkspacesChanged,
}

/// A pushed event on a subscribed connection. Serialized internally
/// tagged on an `event` field, e.g. `{"event":"window-focused",…}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "kebab-case")]
pub enum Event {
    /// A window was mapped.
    WindowOpened { window: WindowInfo },
    /// A window was unmapped.
    WindowClosed { id: u64 },
    /// Keyboard focus moved (to a window, or to nothing). Re-emitted when
    /// the focused window's title changes, so title modules stay live.
    WindowFocused { window: Option<WindowInfo> },
    /// The set of workspaces changed (switch, add/remove, window counts).
    WorkspacesChanged { workspaces: Vec<WorkspaceInfo> },
}

impl Event {
    /// The [`EventKind`] discriminant, for subscription filtering.
    fn kind(&self) -> EventKind {
        match self {
            Event::WindowOpened { .. } => EventKind::WindowOpened,
            Event::WindowClosed { .. } => EventKind::WindowClosed,
            Event::WindowFocused { .. } => EventKind::WindowFocused,
            Event::WorkspacesChanged { .. } => EventKind::WorkspacesChanged,
        }
    }
}

/// A workspace to switch/move to: an absolute index, or one step in
/// either direction. Serializes as `{"index":N}`, `"next"`, or `"prev"`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkspaceTarget {
    Index(usize),
    Next,
    Prev,
}

/// The compositor's reply to a [`Request`]: the payload on success, or a
/// human-readable error string. Serializes as `{"Ok":…}` / `{"Err":…}`.
pub type Reply = Result<Response, String>;

/// A successful reply payload, one variant per [`Request`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Response {
    Version(VersionInfo),
    Outputs(Vec<OutputInfo>),
    Workspaces(Vec<WorkspaceInfo>),
    Layers(Vec<LayerInfo>),
    Windows(Vec<WindowInfo>),
    FocusedWindow(Option<WindowInfo>),
    Binds(Vec<BindInfo>),
    /// A window thumbnail was written to `path` (a PNG of `width`x`height`).
    WindowCapture {
        path: String,
        width: i32,
        height: i32,
    },
    /// An action completed successfully (no payload).
    Handled,
}

/// Compositor identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub name: String,
    pub version: String,
}

/// One connected output (monitor).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub make: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Current mode, physical pixels.
    pub width: i32,
    pub height: i32,
    /// Refresh rate in milli-Hz (so 240 Hz = `240000`).
    pub refresh_mhz: i32,
    pub scale: f64,
    /// Top-left in the logical layout.
    pub x: i32,
    pub y: i32,
    /// Logical size (physical / scale).
    pub logical_width: i32,
    pub logical_height: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_workspace: Option<usize>,
}

/// One live layer-shell surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    /// Namespace the client set at creation (e.g. "rofi", "quickshell").
    pub namespace: String,
    /// `background` | `bottom` | `top` | `overlay`.
    pub layer: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    pub width: i32,
    pub height: i32,
    /// Whether the surface takes keyboard focus (exclusive/on-demand).
    pub keyboard: bool,
    pub exclusive_zone: i32,
}

/// One workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceInfo {
    pub output: String,
    pub index: usize,
    pub active: bool,
    pub window_count: usize,
}

/// One managed window.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "these are independent window-state flags in a wire DTO, not a state machine"
)]
pub struct WindowInfo {
    /// Stable id assigned by the compositor (survives reflows / moves).
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace: Option<usize>,
    /// Cell rect in logical pixels.
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub floating: bool,
    pub fullscreen: bool,
    pub maximized: bool,
    pub focused: bool,
    /// PID of the window's Wayland client, from its socket credentials.
    /// `None` if the client has no resolvable peer pid. Lets a panel kill
    /// or match audio streams to the owning process.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<i32>,
}

/// One configured keybinding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindInfo {
    pub mods: Vec<String>,
    pub key: String,
    pub action: String,
}

// ======================================================================
// Server — compositor-side socket wiring + request dispatch.
// ======================================================================

mod server {
    use std::collections::{HashMap, HashSet};
    use std::io::{Read as _, Write as _};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::{Path, PathBuf};

    use anyhow::{Context as _, Result};
    use smithay::reexports::calloop::generic::Generic;
    use smithay::reexports::calloop::{Interest, LoopHandle, Mode, PostAction};
    use smithay::reexports::wayland_server::Resource as _;
    use smithay::reexports::wayland_server::DisplayHandle;
    use smithay::reexports::wayland_server::backend::ObjectId;
    use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
    use smithay::utils::{IsAlive, SERIAL_COUNTER};
    use smithay::wayland::compositor::with_states;
    use smithay::wayland::shell::xdg::XdgToplevelSurfaceData;
    use tracing::{info, warn};

    use super::{
        BindInfo, Event, EventKind, LayerInfo, OutputInfo, Reply, Request, Response, VersionInfo,
        WindowInfo, WorkspaceInfo, WorkspaceTarget,
    };
    use crate::layout::{FillMode, Layout, WindowEntry};
    use crate::{LoopData, State};

    /// One subscribed connection: a write handle to its socket plus the
    /// event kinds it wants (`None` = all).
    struct Subscriber {
        stream: UnixStream,
        kinds: Option<Vec<EventKind>>,
    }

    /// Stable window-id registry + event-stream bookkeeping. Lives in
    /// [`State`]; ids are assigned on map and reused for the window's
    /// lifetime so a client can name a specific window across reflows.
    #[derive(Default)]
    pub struct IpcState {
        next_id: u64,
        by_surface: HashMap<ObjectId, u64>,
        by_id: HashMap<u64, WlSurface>,
        /// Live event subscribers, keyed by a per-connection id.
        subscribers: HashMap<u64, Subscriber>,
        next_conn: u64,
        /// Diff baselines so the per-iteration poll only emits changes.
        last_window_ids: HashSet<u64>,
        last_focus: Option<(u64, Option<String>)>,
        last_workspaces: Vec<WorkspaceInfo>,
    }

    impl IpcState {
        /// Id for `surface`, allocating one on first sight.
        pub fn assign(&mut self, surface: &WlSurface) -> u64 {
            if let Some(&id) = self.by_surface.get(&surface.id()) {
                return id;
            }
            self.next_id += 1;
            let id = self.next_id;
            self.by_surface.insert(surface.id(), id);
            self.by_id.insert(id, surface.clone());
            id
        }

        /// The surface for a previously-assigned id, if it's still known.
        pub fn surface_of(&self, id: u64) -> Option<WlSurface> {
            self.by_id.get(&id).cloned()
        }

        /// Drop a window's id once its surface is gone.
        pub fn forget(&mut self, surface: &WlSurface) {
            if let Some(id) = self.by_surface.remove(&surface.id()) {
                self.by_id.remove(&id);
            }
        }

        /// Register a subscriber. Returns its connection id and whether it
        /// was the first (so the caller can seed the diff baselines).
        fn add_subscriber(&mut self, stream: UnixStream, events: Vec<EventKind>) -> (u64, bool) {
            let first = self.subscribers.is_empty();
            self.next_conn += 1;
            let id = self.next_conn;
            let kinds = (!events.is_empty()).then_some(events);
            self.subscribers.insert(id, Subscriber { stream, kinds });
            (id, first)
        }

        /// Drop a subscriber (its connection closed).
        fn remove_subscriber(&mut self, id: u64) {
            self.subscribers.remove(&id);
        }

        /// Whether anyone is listening (the poll early-outs if not).
        fn has_subscribers(&self) -> bool {
            !self.subscribers.is_empty()
        }

        /// Write `event` to every subscriber that wants its kind, dropping
        /// any whose socket has gone away.
        fn broadcast(&mut self, event: &Event) {
            let kind = event.kind();
            let Ok(mut payload) = serde_json::to_vec(event) else {
                return;
            };
            payload.push(b'\n');
            let mut dead = Vec::new();
            for (id, sub) in &self.subscribers {
                if sub.kinds.as_ref().is_some_and(|k| !k.contains(&kind)) {
                    continue;
                }
                if (&sub.stream).write_all(&payload).is_err() {
                    dead.push(*id);
                }
            }
            for id in dead {
                self.subscribers.remove(&id);
            }
        }

        /// Send one event to a single subscriber (the subscribe snapshot).
        fn send_to(&mut self, id: u64, event: &Event) {
            let Some(sub) = self.subscribers.get(&id) else {
                return;
            };
            let Ok(mut payload) = serde_json::to_vec(event) else {
                return;
            };
            payload.push(b'\n');
            if (&sub.stream).write_all(&payload).is_err() {
                self.subscribers.remove(&id);
            }
        }

        /// Seed the diff baselines from the current state so the next poll
        /// doesn't re-announce everything that already exists.
        fn set_baseline(
            &mut self,
            windows: &[WindowInfo],
            workspaces: &[WorkspaceInfo],
            focus: Option<&WindowInfo>,
        ) {
            self.last_window_ids = windows.iter().map(|w| w.id).collect();
            self.last_focus = focus.map(|w| (w.id, w.title.clone()));
            self.last_workspaces = workspaces.to_vec();
        }

        /// Diff the gathered state against the baselines, broadcasting one
        /// event per change, then update the baselines.
        fn emit_changes(
            &mut self,
            windows: Vec<WindowInfo>,
            workspaces: Vec<WorkspaceInfo>,
            focus: Option<WindowInfo>,
        ) {
            // Compute the open/close deltas first (borrowing the old
            // baseline), then update it, then broadcast — broadcast needs
            // `&mut self`, so it can't run while the baseline is borrowed.
            let cur_ids: HashSet<u64> = windows.iter().map(|w| w.id).collect();
            let opened: Vec<WindowInfo> = windows
                .into_iter()
                .filter(|w| !self.last_window_ids.contains(&w.id))
                .collect();
            let closed: Vec<u64> = self
                .last_window_ids
                .iter()
                .filter(|id| !cur_ids.contains(id))
                .copied()
                .collect();
            self.last_window_ids = cur_ids;
            for window in opened {
                self.broadcast(&Event::WindowOpened { window });
            }
            for id in closed {
                self.broadcast(&Event::WindowClosed { id });
            }

            let focus_key = focus.as_ref().map(|w| (w.id, w.title.clone()));
            if focus_key != self.last_focus {
                self.last_focus = focus_key;
                self.broadcast(&Event::WindowFocused { window: focus });
            }

            if workspaces != self.last_workspaces {
                self.broadcast(&Event::WorkspacesChanged {
                    workspaces: workspaces.clone(),
                });
                self.last_workspaces = workspaces;
            }
        }
    }

    /// Path of the control socket for the given Wayland display name.
    /// `None` if `$XDG_RUNTIME_DIR` is unset.
    pub fn socket_path(display: &std::ffi::OsStr) -> Option<PathBuf> {
        let runtime = std::env::var_os("XDG_RUNTIME_DIR")?;
        let mut path = PathBuf::from(runtime);
        path.push(format!("libreland-{}.sock", display.to_string_lossy()));
        Some(path)
    }

    /// Bind the control socket and register it on the event loop. Any
    /// stale socket file at `path` is removed first. Accepted connections
    /// are handled by [`accept_connection`].
    pub fn setup(handle: &LoopHandle<'static, LoopData>, path: &Path) -> Result<()> {
        // A leftover socket from a crashed run would make bind fail with
        // EADDRINUSE; clearing it is safe because two compositors can't
        // share one Wayland display name anyway.
        let _ = std::fs::remove_file(path);
        let listener =
            UnixListener::bind(path).with_context(|| format!("bind IPC socket {}", path.display()))?;
        listener
            .set_nonblocking(true)
            .context("set IPC socket non-blocking")?;
        handle
            .insert_source(
                Generic::new(listener, Interest::READ, Mode::Level),
                |_, listener, data: &mut LoopData| {
                    loop {
                        match listener.accept() {
                            Ok((stream, _)) => accept_connection(&data.state.loop_handle, stream),
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                            Err(e) => {
                                warn!(error = %e, "IPC accept failed");
                                break;
                            }
                        }
                    }
                    Ok(PostAction::Continue)
                },
            )
            .map_err(|e| anyhow::anyhow!("insert IPC listener source: {e}"))?;
        info!(path = %path.display(), "IPC control socket listening");
        Ok(())
    }

    /// Register one accepted connection as its own event-loop source. The
    /// per-connection read buffer is owned by the closure. Newline-framed
    /// requests are dispatched as they complete; the source removes
    /// itself on EOF or a hard read error.
    fn accept_connection(handle: &LoopHandle<'static, LoopData>, stream: UnixStream) {
        if let Err(e) = stream.set_nonblocking(true) {
            warn!(error = %e, "IPC connection non-blocking failed");
            return;
        }
        let mut pending: Vec<u8> = Vec::new();
        // Set once this connection issues `subscribe`, so its subscriber
        // entry is dropped when the socket closes.
        let mut subscriber_id: Option<u64> = None;
        let res = handle.insert_source(
            Generic::new(stream, Interest::READ, Mode::Level),
            move |_, stream, data: &mut LoopData| {
                // calloop hands us a `&mut NoIoDrop<UnixStream>`; `UnixStream`
                // implements `Read`/`Write` for `&UnixStream`, so a mutable
                // *binding* to the shared reference is all we need.
                let mut conn: &UnixStream = stream;
                let mut buf = [0u8; 4096];
                let mut closed = false;
                loop {
                    match conn.read(&mut buf) {
                        Ok(0) => {
                            closed = true;
                            break;
                        }
                        Ok(n) => pending.extend_from_slice(&buf[..n]),
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                        Err(e) => {
                            warn!(error = %e, "IPC connection read failed");
                            closed = true;
                            break;
                        }
                    }
                }
                while let Some(nl) = pending.iter().position(|&b| b == b'\n') {
                    let line: Vec<u8> = pending.drain(..=nl).collect();
                    match serde_json::from_slice::<Request>(&line[..nl]) {
                        // `subscribe` turns this connection into an event
                        // stream: clone the write half for the broadcaster,
                        // then push the current snapshot. No request/reply.
                        Ok(Request::Subscribe { events }) => match conn.try_clone() {
                            Ok(write_half) => {
                                let (id, first) =
                                    data.state.ipc.add_subscriber(write_half, events);
                                subscriber_id = Some(id);
                                on_subscribe(&mut data.state, id, first);
                            }
                            Err(e) => write_reply(conn, &Err(format!("subscribe failed: {e}"))),
                        },
                        Ok(req) => {
                            let reply = dispatch(&mut data.state, req);
                            write_reply(conn, &reply);
                        }
                        Err(e) => write_reply(conn, &Err(format!("invalid request: {e}"))),
                    }
                }
                if closed {
                    if let Some(id) = subscriber_id.take() {
                        data.state.ipc.remove_subscriber(id);
                    }
                    return Ok(PostAction::Remove);
                }
                Ok(PostAction::Continue)
            },
        );
        if let Err(e) = res {
            warn!(error = %e, "insert IPC connection source failed");
        }
    }

    /// Serialize and write one reply line. Best effort: a client that hung
    /// up just loses it; the source is reaped on the next read's EOF.
    fn write_reply(mut conn: &UnixStream, reply: &Reply) {
        match serde_json::to_vec(reply) {
            Ok(mut bytes) => {
                bytes.push(b'\n');
                let _ = conn.write_all(&bytes);
            }
            Err(e) => warn!(error = %e, "IPC serialize reply failed"),
        }
    }

    /// Send a fresh subscriber its initial snapshot (focus + workspaces),
    /// and seed the diff baselines when it's the first subscriber so the
    /// next poll doesn't re-announce everything already on screen.
    fn on_subscribe(state: &mut State, id: u64, first: bool) {
        let focus = focused_window(state);
        let workspaces = workspaces(state);
        state.ipc.send_to(
            id,
            &Event::WindowFocused {
                window: focus.clone(),
            },
        );
        state.ipc.send_to(
            id,
            &Event::WorkspacesChanged {
                workspaces: workspaces.clone(),
            },
        );
        if first {
            let windows = windows(state);
            state.ipc.set_baseline(&windows, &workspaces, focus.as_ref());
        }
    }

    /// Run once per event-loop iteration (after each batch): diff the
    /// current state against the baselines and broadcast any changes.
    /// Cheap no-op when nobody is subscribed.
    pub fn poll_events(state: &mut State) {
        if !state.ipc.has_subscribers() {
            return;
        }
        let windows = windows(state);
        let workspaces = workspaces(state);
        let focus = focused_window(state);
        state.ipc.emit_changes(windows, workspaces, focus);
    }

    /// Route a request to its query or action and wrap the result as a
    /// [`Reply`]. Queries can't fail; actions return `Err(message)` when
    /// the target can't be resolved.
    fn dispatch(state: &mut State, req: Request) -> Reply {
        // Requests that change what's on screen need an on-demand redraw.
        // (Reload redraws itself via reload_config; Spawn's new window
        // redraws when it maps; queries change nothing.)
        let mutating = matches!(
            req,
            Request::FocusWindow { .. }
                | Request::Close { .. }
                | Request::ToggleFloating { .. }
                | Request::ToggleFullscreen { .. }
                | Request::ToggleMaximized { .. }
                | Request::FocusWorkspace { .. }
                | Request::MoveToWorkspace { .. }
        );
        let reply = match req {
            Request::Version => Ok(Response::Version(VersionInfo {
                name: "libreland".to_owned(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
            })),
            Request::Outputs => Ok(Response::Outputs(outputs(state))),
            Request::Workspaces => Ok(Response::Workspaces(workspaces(state))),
            Request::Layers => Ok(Response::Layers(layers(state))),
            Request::Windows => Ok(Response::Windows(windows(state))),
            Request::FocusedWindow => Ok(Response::FocusedWindow(focused_window(state))),
            Request::Binds => Ok(Response::Binds(binds(state))),
            Request::CaptureWindow { id, max } => capture_window(state, id, max),
            Request::FocusWindow { id } => focus_window(state, id),
            Request::Close { id } => close_window(state, id),
            Request::ToggleFloating { id } => toggle(state, id, Layout::toggle_floating),
            Request::ToggleFullscreen { id } => toggle(state, id, Layout::toggle_fullscreen),
            Request::ToggleMaximized { id } => toggle(state, id, Layout::toggle_maximized),
            Request::FocusWorkspace { output, target } => focus_workspace(state, output, target),
            Request::MoveToWorkspace { id, target } => move_to_workspace(state, id, target),
            Request::Spawn { command } => spawn(state, &command),
            Request::Reload => reload(state),
            Request::Exit => {
                info!("exit requested via IPC");
                state.loop_signal.stop();
                Ok(Response::Handled)
            }
            // `subscribe` is intercepted before dispatch (it streams
            // rather than replying once); reaching here means a client
            // sent it on a plain request/response connection.
            Request::Subscribe { .. } => {
                Err("subscribe must be used as a streaming connection".to_owned())
            }
        };
        if mutating {
            state.queue_redraw_all();
        }
        reply
    }

    /// Render a window thumbnail to a PNG and return its path. Works for any
    /// window, on any workspace or output (its surface tree is rendered in
    /// isolation — no other windows, no cursor).
    fn capture_window(state: &mut State, id: u64, max: Option<i32>) -> Reply {
        let surface = state
            .ipc
            .surface_of(id)
            .filter(IsAlive::alive)
            .ok_or_else(|| format!("no window with id {id}"))?;
        let (w, h, rgba) = state
            .renderer
            .capture_window(&surface, max.unwrap_or(512))
            .map_err(|e| format!("capture failed: {e}"))?;
        let png = crate::screenshot::encode_rgba(&rgba, w, h)
            .map_err(|e| format!("png encode failed: {e}"))?;
        let path = capture_path(id);
        write_atomic(&path, &png)
            .map_err(|e| format!("writing {}: {e}", path.display()))?;
        Ok(Response::WindowCapture {
            path: path.to_string_lossy().into_owned(),
            width: w,
            height: h,
        })
    }

    /// `$XDG_RUNTIME_DIR/libreland-window-<id>.png` (falls back to `/tmp`).
    fn capture_path(id: u64) -> std::path::PathBuf {
        let mut dir = std::env::var_os("XDG_RUNTIME_DIR").map_or_else(
            || std::path::PathBuf::from("/tmp"),
            std::path::PathBuf::from,
        );
        dir.push(format!("libreland-window-{id}.png"));
        dir
    }

    /// Write atomically (temp + rename) so a reader never sees a half-file.
    fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)
    }

    /// Resolve a window target: the surface for `id`, or the focused
    /// window when `id` is `None`. Errors if the id is unknown / dead or
    /// nothing is focused.
    fn resolve(state: &State, id: Option<u64>) -> Result<WlSurface, String> {
        match id {
            Some(id) => state
                .ipc
                .surface_of(id)
                .filter(IsAlive::alive)
                .ok_or_else(|| format!("no window with id {id}")),
            None => state
                .seat
                .get_keyboard()
                .and_then(|k| k.current_focus())
                .ok_or_else(|| "no focused window".to_owned()),
        }
    }

    fn focus_window(state: &mut State, id: u64) -> Reply {
        let surface = resolve(state, Some(id))?;
        // Reveal the window's workspace first so a focus request for a
        // window on a hidden workspace actually shows it.
        if let Some(entry) = state
            .layout
            .window_entries()
            .into_iter()
            .find(|e| e.surface == surface)
        {
            state
                .layout
                .switch_workspace_to(&entry.output, entry.workspace);
        }
        if let Some(kbd) = state.seat.get_keyboard() {
            kbd.set_focus(state, Some(surface), SERIAL_COUNTER.next_serial());
        }
        Ok(Response::Handled)
    }

    fn close_window(state: &mut State, id: Option<u64>) -> Reply {
        let surface = resolve(state, id)?;
        let toplevel = state
            .xdg_shell_state
            .toplevel_surfaces()
            .iter()
            .find(|t| t.wl_surface() == &surface)
            .cloned()
            .ok_or_else(|| "target is not a toplevel window".to_owned())?;
        toplevel.send_close();
        Ok(Response::Handled)
    }

    /// Shared body for the three toggle actions. `apply` runs the layout
    /// mutation and reports whether the window was found.
    fn toggle(
        state: &mut State,
        id: Option<u64>,
        apply: impl FnOnce(&mut crate::layout::Layout, &WlSurface) -> bool,
    ) -> Reply {
        let surface = resolve(state, id)?;
        if apply(&mut state.layout, &surface) {
            Ok(Response::Handled)
        } else {
            Err("window is not managed by the tiler".to_owned())
        }
    }

    fn focus_workspace(
        state: &mut State,
        output: Option<String>,
        target: WorkspaceTarget,
    ) -> Reply {
        let output = output
            .or_else(|| state.renderer.primary_output_name().map(str::to_owned))
            .ok_or_else(|| "no output connected".to_string())?;
        let active = state
            .layout
            .active_workspace(&output)
            .ok_or_else(|| format!("no output named {output}"))?;
        let index = resolve_target(target, active);
        if state.layout.switch_workspace_to(&output, index) {
            refocus_active(state, &output);
        }
        Ok(Response::Handled)
    }

    fn move_to_workspace(state: &mut State, id: Option<u64>, target: WorkspaceTarget) -> Reply {
        let surface = resolve(state, id)?;
        let entry = state
            .layout
            .window_entries()
            .into_iter()
            .find(|e| e.surface == surface)
            .ok_or_else(|| "window is not on a workspace".to_owned())?;
        let index = resolve_target(target, entry.workspace);
        if state.layout.move_window_to_workspace(&surface, index) {
            Ok(Response::Handled)
        } else {
            Err("could not move the window (is it on the active workspace?)".to_owned())
        }
    }

    /// Resolve a [`WorkspaceTarget`] to an absolute index relative to the
    /// `current` active workspace.
    fn resolve_target(target: WorkspaceTarget, current: usize) -> usize {
        match target {
            WorkspaceTarget::Index(i) => i,
            WorkspaceTarget::Next => current + 1,
            WorkspaceTarget::Prev => current.saturating_sub(1),
        }
    }

    /// After a workspace switch, move keyboard focus to a window on the
    /// now-active workspace (or clear it when that workspace is empty), so
    /// input and the active border follow the switch.
    fn refocus_active(state: &mut State, output: &str) {
        let Some(active) = state.layout.active_workspace(output) else {
            return;
        };
        let next = state
            .layout
            .window_entries()
            .into_iter()
            .find(|e| e.output == output && e.workspace == active)
            .map(|e| e.surface);
        if let Some(kbd) = state.seat.get_keyboard() {
            kbd.set_focus(state, next, SERIAL_COUNTER.next_serial());
        }
    }

    fn spawn(state: &State, command: &[String]) -> Reply {
        let Some((program, args)) = command.split_first() else {
            return Err("empty command".to_owned());
        };
        let mut cmd = std::process::Command::new(program);
        cmd.args(args);
        // Inherit the live env + X `$DISPLAY` like every other spawn path.
        state.apply_child_env(&mut cmd);
        match cmd.spawn() {
            Ok(child) => {
                info!(pid = child.id(), ?command, "spawned via IPC");
                Ok(Response::Handled)
            }
            Err(e) => Err(format!("spawn failed: {e}")),
        }
    }

    fn reload(state: &mut State) -> Reply {
        let path = crate::config::Config::path()
            .ok_or_else(|| "no config path (XDG dirs unset)".to_owned())?;
        state.reload_config(&path);
        Ok(Response::Handled)
    }

    fn outputs(state: &mut State) -> Vec<OutputInfo> {
        state
            .renderer
            .output_descriptors()
            .into_iter()
            .map(|d| {
                let (make, model) = make_model(state, &d.name);
                OutputInfo {
                    active_workspace: state.layout.active_workspace(&d.name),
                    name: d.name,
                    make,
                    model,
                    width: d.mode_size.w,
                    height: d.mode_size.h,
                    refresh_mhz: d.refresh_mhz,
                    scale: d.scale,
                    x: d.compositor_position.x,
                    y: d.compositor_position.y,
                    logical_width: d.compositor_size.w,
                    logical_height: d.compositor_size.h,
                }
            })
            .collect()
    }

    /// EDID make/model for a connector, from the matching `wl_output`.
    /// Empty strings (no EDID) come back as `None`.
    fn make_model(state: &State, name: &str) -> (Option<String>, Option<String>) {
        state
            .outputs
            .iter()
            .find(|o| o.name() == name)
            .map_or((None, None), |o| {
                let p = o.physical_properties();
                let nz = |s: String| (!s.is_empty()).then_some(s);
                (nz(p.make), nz(p.model))
            })
    }

    fn workspaces(state: &State) -> Vec<WorkspaceInfo> {
        state
            .layout
            .workspace_entries()
            .into_iter()
            .map(|e| WorkspaceInfo {
                output: e.output,
                index: e.index,
                active: e.active,
                window_count: e.windows,
            })
            .collect()
    }

    fn layers(state: &State) -> Vec<LayerInfo> {
        use smithay::wayland::shell::wlr_layer::{KeyboardInteractivity, Layer};
        state
            .layer_shell_state
            .layer_surfaces()
            .map(|ls| {
                let surface = ls.wl_surface();
                let cached = crate::wayland::layer_cached_state(surface);
                let (width, height) =
                    crate::wayland::layer_size(state.layer_output_rect(surface), &cached);
                LayerInfo {
                    namespace: state.layer_namespaces.get(surface).cloned().unwrap_or_default(),
                    layer: match cached.layer {
                        Layer::Background => "background",
                        Layer::Bottom => "bottom",
                        Layer::Top => "top",
                        Layer::Overlay => "overlay",
                    }
                    .to_owned(),
                    output: state.layer_outputs.get(surface).cloned(),
                    width,
                    height,
                    keyboard: !matches!(
                        cached.keyboard_interactivity,
                        KeyboardInteractivity::None
                    ),
                    exclusive_zone: cached.exclusive_zone.into(),
                }
            })
            .collect()
    }

    fn windows(state: &mut State) -> Vec<WindowInfo> {
        let focus = state.seat.get_keyboard().and_then(|k| k.current_focus());
        let dh = state.display_handle.clone();
        let entries = state.layout.window_entries();
        entries
            .into_iter()
            .map(|e| window_info(&mut state.ipc, &dh, e, focus.as_ref()))
            .collect()
    }

    fn focused_window(state: &mut State) -> Option<WindowInfo> {
        let focus = state.seat.get_keyboard().and_then(|k| k.current_focus())?;
        let dh = state.display_handle.clone();
        let entry = state
            .layout
            .window_entries()
            .into_iter()
            .find(|e| e.surface == focus)?;
        Some(window_info(&mut state.ipc, &dh, entry, Some(&focus)))
    }

    /// Build a [`WindowInfo`] from a layout entry, assigning its stable
    /// id and reading title/app-id off the surface.
    fn window_info(
        ipc: &mut IpcState,
        dh: &DisplayHandle,
        e: WindowEntry,
        focus: Option<&WlSurface>,
    ) -> WindowInfo {
        let id = ipc.assign(&e.surface);
        let (app_id, title) = toplevel_strings(&e.surface);
        let focused = focus == Some(&e.surface);
        // Peer pid from the client's socket credentials.
        let pid = e
            .surface
            .client()
            .and_then(|c| c.get_credentials(dh).ok())
            .map(|cred| cred.pid);
        WindowInfo {
            id,
            app_id,
            title,
            output: Some(e.output),
            workspace: Some(e.workspace),
            x: e.rect.loc.x,
            y: e.rect.loc.y,
            width: e.rect.size.w,
            height: e.rect.size.h,
            floating: e.floating,
            fullscreen: e.fill == FillMode::Fullscreen,
            maximized: e.fill == FillMode::Maximized,
            focused,
            pid,
        }
    }

    /// `(app_id, title)` from a toplevel's xdg role data.
    fn toplevel_strings(surface: &WlSurface) -> (Option<String>, Option<String>) {
        with_states(surface, |states| {
            states
                .data_map
                .get::<XdgToplevelSurfaceData>()
                .map_or((None, None), |d| {
                    let a = d.lock().unwrap();
                    (a.app_id.clone(), a.title.clone())
                })
        })
    }

    fn binds(state: &State) -> Vec<BindInfo> {
        state
            .config
            .binds
            .bindings
            .iter()
            .map(|b| BindInfo {
                mods: mod_names(b.mods),
                key: xkbcommon::xkb::keysym_get_name(b.keysym),
                action: action_label(&b.action),
            })
            .collect()
    }

    /// Render a modifier mask as canonical names, in a stable order.
    fn mod_names(mask: u32) -> Vec<String> {
        use crate::keyboard::{MOD_ALT, MOD_CTRL, MOD_SHIFT, MOD_SUPER};
        let mut out = Vec::new();
        for (bit, name) in [
            (MOD_SUPER, "Super"),
            (MOD_CTRL, "Ctrl"),
            (MOD_ALT, "Alt"),
            (MOD_SHIFT, "Shift"),
        ] {
            if mask & bit != 0 {
                out.push(name.to_owned());
            }
        }
        out
    }

    /// A short label for an action (matches the config action names).
    fn action_label(action: &crate::config::Action) -> String {
        use crate::config::Action;
        match action {
            Action::Exit => "exit".to_owned(),
            Action::ToggleFloating => "toggle-floating".to_owned(),
            Action::ToggleFullscreen => "toggle-fullscreen".to_owned(),
            Action::Close => "close".to_owned(),
            Action::Spawn(cmd) => format!("spawn {cmd}"),
            Action::Screenshot(_) => "screenshot".to_owned(),
        }
    }
}

pub use server::{IpcState, poll_events, setup, socket_path};

// ======================================================================
// Client — the `libreland msg …` subcommand.
// ======================================================================

mod client {
    use std::io::{BufRead as _, Write as _};
    use std::os::unix::net::UnixStream;
    use std::path::PathBuf;

    use anyhow::{Context as _, Result, bail};
    use clap::{Parser, Subcommand};

    use super::{Event, EventKind, LayerInfo, Reply, Request, Response, WorkspaceTarget};

    /// A workspace argument: a number, `next`, or `prev`.
    #[derive(Clone)]
    enum WsTargetArg {
        Index(usize),
        Next,
        Prev,
    }

    impl std::str::FromStr for WsTargetArg {
        type Err = String;
        fn from_str(s: &str) -> Result<Self, String> {
            match s.to_ascii_lowercase().as_str() {
                "next" => Ok(Self::Next),
                "prev" | "previous" => Ok(Self::Prev),
                _ => s
                    .parse::<usize>()
                    .map(Self::Index)
                    .map_err(|_| format!(r#"expected a workspace index, "next", or "prev", got {s:?}"#)),
            }
        }
    }

    impl WsTargetArg {
        fn to_target(&self) -> WorkspaceTarget {
            match self {
                Self::Index(i) => WorkspaceTarget::Index(*i),
                Self::Next => WorkspaceTarget::Next,
                Self::Prev => WorkspaceTarget::Prev,
            }
        }
    }

    #[derive(Parser)]
    #[command(
        name = "libreland msg",
        about = "Query and control a running Libreland compositor",
        disable_help_subcommand = true
    )]
    struct Cli {
        /// Print the raw JSON reply instead of formatted text.
        #[arg(long, global = true)]
        json: bool,
        #[command(subcommand)]
        command: Command,
    }

    #[derive(Subcommand)]
    enum Command {
        /// Print the compositor name and version.
        Version,
        /// List connected outputs (monitors).
        Outputs,
        /// List workspaces across all outputs.
        Workspaces,
        /// List live layer-shell surfaces (with their namespaces).
        Layers,
        /// List all managed windows.
        Windows,
        /// Show the keyboard-focused window.
        #[command(alias = "focused")]
        FocusedWindow,
        /// List the configured keybindings.
        Binds,
        /// Render a window (by id) to a PNG thumbnail and print its path.
        /// Works regardless of the window's workspace or output.
        CaptureWindow {
            id: u64,
            /// Cap the longest side in pixels (downscaled only; default 512).
            #[arg(long)]
            max: Option<i32>,
        },
        /// Focus a window by id (revealing its workspace).
        FocusWindow { id: u64 },
        /// Close a window (the focused one if no id is given).
        Close { id: Option<u64> },
        /// Toggle a window between tiled and floating.
        ToggleFloating { id: Option<u64> },
        /// Toggle a window's fullscreen state.
        ToggleFullscreen { id: Option<u64> },
        /// Toggle a window's maximized state.
        ToggleMaximized { id: Option<u64> },
        /// Switch a workspace. TARGET is a number, "next", or "prev".
        FocusWorkspace {
            target: WsTargetArg,
            /// Output to switch (defaults to the primary).
            #[arg(long)]
            output: Option<String>,
        },
        /// Move a window to a workspace and follow it. TARGET is a
        /// number, "next", or "prev".
        MoveToWorkspace {
            target: WsTargetArg,
            /// Window id (defaults to the focused window).
            id: Option<u64>,
        },
        /// Spawn a command (everything after the verb is the argv).
        Spawn {
            #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true)]
            command: Vec<String>,
        },
        /// Re-read the config file now.
        Reload,
        /// Exit the compositor.
        Exit,
        /// Stream live events until interrupted. With no kinds given,
        /// every event is streamed.
        Subscribe {
            #[arg(value_enum)]
            events: Vec<EventKind>,
        },
    }

    impl Command {
        fn to_request(&self) -> Request {
            match self {
                Command::Version => Request::Version,
                Command::Outputs => Request::Outputs,
                Command::Workspaces => Request::Workspaces,
                Command::Layers => Request::Layers,
                Command::Windows => Request::Windows,
                Command::FocusedWindow => Request::FocusedWindow,
                Command::Binds => Request::Binds,
                Command::CaptureWindow { id, max } => Request::CaptureWindow { id: *id, max: *max },
                Command::FocusWindow { id } => Request::FocusWindow { id: *id },
                Command::Close { id } => Request::Close { id: *id },
                Command::ToggleFloating { id } => Request::ToggleFloating { id: *id },
                Command::ToggleFullscreen { id } => Request::ToggleFullscreen { id: *id },
                Command::ToggleMaximized { id } => Request::ToggleMaximized { id: *id },
                Command::FocusWorkspace { target, output } => Request::FocusWorkspace {
                    output: output.clone(),
                    target: target.to_target(),
                },
                Command::MoveToWorkspace { target, id } => Request::MoveToWorkspace {
                    id: *id,
                    target: target.to_target(),
                },
                Command::Spawn { command } => Request::Spawn {
                    command: command.clone(),
                },
                Command::Reload => Request::Reload,
                Command::Exit => Request::Exit,
                Command::Subscribe { events } => Request::Subscribe {
                    events: events.clone(),
                },
            }
        }
    }

    /// Entry point for `libreland msg …`. Parses the subcommand, sends it
    /// to the compositor, and prints the reply (formatted, or raw JSON
    /// with `--json`). Errors propagate to a non-zero exit.
    pub fn run() -> Result<()> {
        // argv is `libreland msg <args…>`; hand clap a program name plus
        // everything after `msg` so its help/usage reads naturally.
        let argv = std::iter::once(std::ffi::OsString::from("libreland msg"))
            .chain(std::env::args_os().skip(2));
        let cli = Cli::parse_from(argv);
        let request = cli.command.to_request();
        // `subscribe` is a long-lived stream, not a single round-trip.
        if let Request::Subscribe { .. } = request {
            return stream_events(&request, cli.json);
        }
        let reply = send(&request).context("talking to the compositor")?;
        match reply {
            Ok(response) => print_response(&response, cli.json),
            Err(message) => bail!("{message}"),
        }
    }

    /// Open a subscription and print each event as it arrives, until the
    /// compositor closes the connection or the user interrupts.
    fn stream_events(request: &Request, json: bool) -> Result<()> {
        let path = socket_path().context(
            "could not locate the control socket; is the compositor running? \
             (set $LIBRELAND_SOCKET or $WAYLAND_DISPLAY)",
        )?;
        let stream = UnixStream::connect(&path)
            .with_context(|| format!("connecting to {}", path.display()))?;
        let mut writer = stream.try_clone().context("clone IPC stream")?;
        let mut line = serde_json::to_vec(request).context("serialize request")?;
        line.push(b'\n');
        writer.write_all(&line).context("send subscribe")?;
        writer.flush().ok();

        let reader = std::io::BufReader::new(stream);
        for line in reader.lines() {
            let line = line.context("read event")?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<Event>(trimmed) {
                Ok(event) => print_event(&event, json),
                // A non-event line can only be an early error reply
                // (e.g. the socket refused the subscription).
                Err(_) => {
                    if let Ok(Err(message)) = serde_json::from_str::<Reply>(trimmed) {
                        bail!("{message}");
                    }
                }
            }
        }
        Ok(())
    }

    fn print_event(event: &Event, json: bool) {
        use std::io::Write as _;
        if json {
            if let Ok(s) = serde_json::to_string(event) {
                println!("{s}");
            }
            return;
        }
        match event {
            Event::WindowOpened { window } => {
                println!("opened   #{} {}", window.id, window_label(window));
            }
            Event::WindowClosed { id } => println!("closed   #{id}"),
            Event::WindowFocused { window } => match window {
                Some(w) => println!("focused  #{} {}", w.id, window_label(w)),
                None => println!("focused  (none)"),
            },
            Event::WorkspacesChanged { workspaces } => {
                let active: Vec<String> = workspaces
                    .iter()
                    .filter(|w| w.active)
                    .map(|w| format!("{}:{}", w.output, w.index))
                    .collect();
                println!("workspaces  active=[{}]", active.join(" "));
            }
        }
        // Events stream live; keep them flushing for `| while read` loops.
        let _ = std::io::stdout().flush();
    }

    /// A short `app_id — title` label for a window event line.
    fn window_label(w: &super::WindowInfo) -> String {
        match (w.app_id.as_deref(), w.title.as_deref()) {
            (Some(app), Some(title)) => format!("{app} — {title}"),
            (Some(s), None) | (None, Some(s)) => s.to_owned(),
            (None, None) => "(untitled)".to_owned(),
        }
    }

    /// Connect, send one request line, read one reply line.
    fn send(request: &Request) -> Result<Reply> {
        let path = socket_path().context(
            "could not locate the control socket; is the compositor running? \
             (set $LIBRELAND_SOCKET or $WAYLAND_DISPLAY)",
        )?;
        let stream = UnixStream::connect(&path)
            .with_context(|| format!("connecting to {}", path.display()))?;
        let mut writer = stream.try_clone().context("clone IPC stream")?;
        let mut line = serde_json::to_vec(request).context("serialize request")?;
        line.push(b'\n');
        writer.write_all(&line).context("send request")?;
        writer.flush().ok();

        let mut reader = std::io::BufReader::new(stream);
        let mut response = String::new();
        reader
            .read_line(&mut response)
            .context("read reply")?;
        if response.trim().is_empty() {
            bail!("the compositor closed the connection without replying");
        }
        serde_json::from_str(response.trim()).context("parse reply")
    }

    /// The control socket path: `$LIBRELAND_SOCKET`, else derived from
    /// `$WAYLAND_DISPLAY` + `$XDG_RUNTIME_DIR`.
    fn socket_path() -> Option<PathBuf> {
        if let Some(explicit) = std::env::var_os("LIBRELAND_SOCKET") {
            return Some(PathBuf::from(explicit));
        }
        let display = std::env::var_os("WAYLAND_DISPLAY")?;
        super::socket_path(&display)
    }

    fn print_response(response: &Response, json: bool) -> Result<()> {
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(response).context("format JSON")?
            );
            return Ok(());
        }
        match response {
            Response::Version(v) => println!("{} {}", v.name, v.version),
            Response::Outputs(outputs) => print_outputs(outputs),
            Response::Workspaces(workspaces) => print_workspaces(workspaces),
            Response::Layers(layers) => print_layers(layers),
            Response::Windows(windows) => print_windows(windows),
            Response::FocusedWindow(window) => match window {
                Some(w) => print_windows(std::slice::from_ref(w)),
                None => println!("no focused window"),
            },
            Response::Binds(binds) => print_binds(binds),
            Response::WindowCapture { path, width, height } => {
                println!("{path} ({width}x{height})");
            }
            // Actions succeed silently (Unix convention); `--json` above
            // still prints the `"Handled"` marker for scripts that check.
            Response::Handled => {}
        }
        Ok(())
    }

    fn print_outputs(outputs: &[super::OutputInfo]) {
        if outputs.is_empty() {
            println!("no outputs");
            return;
        }
        for o in outputs {
            let model = match (&o.make, &o.model) {
                (Some(mk), Some(md)) => format!("  ({mk} {md})"),
                (Some(s), None) | (None, Some(s)) => format!("  ({s})"),
                (None, None) => String::new(),
            };
            println!("{}{model}", o.name);
            println!(
                "  mode:  {}x{} @ {:.3} Hz   scale {}",
                o.width,
                o.height,
                f64::from(o.refresh_mhz) / 1000.0,
                o.scale
            );
            println!(
                "  pos:   {},{}   logical {}x{}",
                o.x, o.y, o.logical_width, o.logical_height
            );
            if let Some(ws) = o.active_workspace {
                println!("  active workspace: {ws}");
            }
        }
    }

    fn print_workspaces(workspaces: &[super::WorkspaceInfo]) {
        if workspaces.is_empty() {
            println!("no workspaces");
            return;
        }
        println!("{:<10} {:>5}  {:<6} {:>7}", "OUTPUT", "INDEX", "ACTIVE", "WINDOWS");
        for w in workspaces {
            println!(
                "{:<10} {:>5}  {:<6} {:>7}",
                w.output,
                w.index,
                if w.active { "*" } else { "" },
                w.window_count
            );
        }
    }

    fn print_layers(layers: &[LayerInfo]) {
        if layers.is_empty() {
            println!("no layer surfaces");
            return;
        }
        println!(
            "{:<22} {:<9} {:<8} {:>9}  {:<3} {:>8}",
            "NAMESPACE", "LAYER", "OUTPUT", "SIZE", "KBD", "EXCL"
        );
        for l in layers {
            println!(
                "{:<22} {:<9} {:<8} {:>4}x{:<4} {:<3} {:>8}",
                l.namespace,
                l.layer,
                l.output.as_deref().unwrap_or("-"),
                l.width,
                l.height,
                if l.keyboard { "yes" } else { "" },
                l.exclusive_zone
            );
        }
    }

    fn print_windows(windows: &[super::WindowInfo]) {
        if windows.is_empty() {
            println!("no windows");
            return;
        }
        println!(
            "{:>4}  {:<16} {:<28} {:<9} {:<8} STATE",
            "ID", "APP_ID", "TITLE", "WORKSPACE", "OUTPUT"
        );
        for w in windows {
            let mut state = Vec::new();
            state.push(if w.floating { "floating" } else { "tiled" });
            if w.fullscreen {
                state.push("fullscreen");
            }
            if w.maximized {
                state.push("maximized");
            }
            if w.focused {
                state.push("focused");
            }
            let ws = w.workspace.map_or_else(|| "-".to_owned(), |n| n.to_string());
            println!(
                "{:>4}  {:<16} {:<28} {:<9} {:<8} {}",
                w.id,
                truncate(w.app_id.as_deref().unwrap_or("-"), 16),
                truncate(w.title.as_deref().unwrap_or("-"), 28),
                ws,
                w.output.as_deref().unwrap_or("-"),
                state.join(" ")
            );
        }
    }

    fn print_binds(binds: &[super::BindInfo]) {
        if binds.is_empty() {
            println!("no binds");
            return;
        }
        for b in binds {
            let mut combo = b.mods.clone();
            combo.push(b.key.clone());
            println!("{:<24} {}", combo.join("+"), b.action);
        }
    }

    /// Trim `s` to `max` display columns, marking truncation with `…`.
    fn truncate(s: &str, max: usize) -> String {
        if s.chars().count() <= max {
            s.to_owned()
        } else {
            let kept: String = s.chars().take(max.saturating_sub(1)).collect();
            format!("{kept}…")
        }
    }
}

pub use client::run as run_client;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_wire_format_is_tagged() {
        assert_eq!(
            serde_json::to_string(&Request::FocusedWindow).unwrap(),
            r#"{"cmd":"focused-window"}"#
        );
        assert_eq!(
            serde_json::to_string(&Request::Windows).unwrap(),
            r#"{"cmd":"windows"}"#
        );
    }

    #[test]
    fn requests_round_trip() {
        for req in [
            Request::Version,
            Request::Outputs,
            Request::Workspaces,
            Request::Windows,
            Request::FocusedWindow,
            Request::Binds,
        ] {
            let json = serde_json::to_string(&req).unwrap();
            let back: Request = serde_json::from_str(&json).unwrap();
            // Re-serializing the decoded value must reproduce the wire form.
            assert_eq!(serde_json::to_string(&back).unwrap(), json);
        }
    }

    #[test]
    fn ok_reply_round_trips() {
        let reply: Reply = Ok(Response::Version(VersionInfo {
            name: "libreland".to_owned(),
            version: "0.1.0".to_owned(),
        }));
        let json = serde_json::to_string(&reply).unwrap();
        assert!(json.starts_with(r#"{"Ok":"#), "got {json}");
        let back: Reply = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, Ok(Response::Version(_))));
    }

    #[test]
    fn error_reply_is_compact() {
        let reply: Reply = Err("no such window".to_owned());
        assert_eq!(
            serde_json::to_string(&reply).unwrap(),
            r#"{"Err":"no such window"}"#
        );
    }

    #[test]
    fn window_info_omits_absent_optionals() {
        let w = WindowInfo {
            id: 7,
            app_id: None,
            title: None,
            output: None,
            workspace: None,
            x: 0,
            y: 0,
            width: 100,
            height: 100,
            floating: false,
            fullscreen: false,
            maximized: false,
            focused: true,
            pid: None,
        };
        let json = serde_json::to_string(&w).unwrap();
        // Skipped Nones don't appear; present fields do.
        assert!(!json.contains("app_id"), "got {json}");
        assert!(!json.contains("pid"), "got {json}");
        assert!(json.contains(r#""id":7"#));
        assert!(json.contains(r#""focused":true"#));
    }
}
