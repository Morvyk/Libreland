//! Libreland: a Wayland compositor in pure Rust, configured in Lua.
//!
//! Binary entry point. Current scope: open a libseat session, enumerate
//! input devices through udev + libinput (with libinput accel /
//! profile applied from [`config::Config`]), set up a GBM + EGL + GLES
//! render loop on the first connected display, and paint each vblank
//! with the configured wallpaper plus a mouse-following cursor sprite.
//! Key events are routed through xkbcommon and matched against the
//! keybind list in [`config::Config::binds`].
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
    InputEvent, KeyState, KeyboardKeyEvent as _, PointerButtonEvent as _,
};
use smithay::backend::libinput::{LibinputInputBackend, LibinputSessionInterface};
use smithay::backend::session::Session as _;
use smithay::backend::session::libseat::{LibSeatSession, LibSeatSessionNotifier};
use smithay::backend::udev::{UdevBackend, UdevEvent};
use smithay::reexports::calloop::{EventLoop, LoopSignal};
use smithay::reexports::input as libinput;
use smithay::reexports::input::Libinput;
use smithay::reexports::input::event::keyboard::KeyboardKeyEvent as LibinputKeyEvent;
use std::fs::File;
use std::io;
use tracing::{debug, info, warn};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

mod config;
mod drm;
mod keyboard;
mod render;

/// Mutable state threaded through every event-loop callback.
///
/// Kept deliberately small for now — once we add a renderer, protocol
/// handlers and the Lua config bridge, those will move in here too so
/// the calloop closures can mutate them without juggling `Rc<RefCell<_>>`.
struct State {
    /// The libseat session is retained so future code can query its
    /// active flag and switch VTs. libinput already holds an internal
    /// clone for opening `/dev/input/*` device nodes.
    #[allow(
        dead_code,
        reason = "session is held for upcoming VT-switch and activation tracking; not read yet"
    )]
    session: LibSeatSession,
    /// Used by the exit hotkey to break calloop's `run` cleanly.
    loop_signal: LoopSignal,
    /// DRM master claim. Held by the State so the master claim
    /// outlives the renderer's swapchain — dropping it releases the
    /// display back to logind on clean shutdown.
    #[allow(
        dead_code,
        reason = "kept alive for the DRM master claim; will be queried by VT-switch / session-activation code"
    )]
    drm_device: DrmDevice,
    /// GBM + EGL + GLES render pipeline. The vblank callback drives
    /// it once per refresh.
    renderer: render::Renderer,
    /// xkbcommon keymap + state. Every libinput key event flows
    /// through this to get a layout-aware keysym + modifier mask,
    /// which the hotkey logic matches on.
    keyboard: keyboard::Keyboard,
    /// All runtime settings (monitors, input, binds, wallpaper, …)
    /// in one place. Defaults today; the Lua loader in milestone
    /// 3c will replace this from `$XDG_CONFIG_HOME/libreland/config.lua`.
    config: config::Config,
}

impl State {
    /// Feed a key event through xkbcommon to get its layout-aware
    /// keysym + effective modifier mask, log the result at debug,
    /// and fire whichever `config.binds.bindings` entry matches the
    /// press (keysym AND every required modifier present, extras
    /// tolerated). First match wins. Release events don't fire
    /// actions but still update xkb's internal modifier state.
    fn handle_key(&mut self, event: &LibinputKeyEvent) {
        let pressed = matches!(event.state(), KeyState::Pressed);
        let result = self.keyboard.process(event.key_code(), pressed);

        debug!(
            keysym = ?result.keysym,
            mods = format!("{:#06b}", result.mods),
            pressed,
            "key processed through xkb"
        );

        if !pressed {
            return;
        }

        // Find the matching binding's action first so the borrow on
        // `self.config.binds` ends before we dispatch — `Action` is
        // `Copy`, so this is a cheap byte-copy out.
        let matched = self
            .config
            .binds
            .bindings
            .iter()
            .find(|b| result.keysym == b.keysym && result.has_all_mods(b.mods))
            .map(|b| b.action);

        if let Some(action) = matched {
            self.dispatch_action(action);
        }
    }

    /// Run a bound action. Single-armed today; grows as we add more
    /// actions (`reload`, `spawn`, `change_vt`, …).
    fn dispatch_action(&mut self, action: config::Action) {
        match action {
            config::Action::Exit => {
                info!("exit action fired — stopping event loop");
                self.loop_signal.stop();
            }
        }
    }
}

fn main() -> Result<()> {
    // The WorkerGuard MUST stay alive for the whole of main; dropping it
    // releases the tracing-appender worker thread and flushes the file
    // log. Bind it with a leading underscore so clippy doesn't nag, but
    // do NOT use `_` (anonymous) — that would drop it immediately.
    let _log_guard = init_tracing()?;
    info!("libreland starting");

    // Compositor configuration. Defaults today; replaced wholesale
    // by the Lua loader in milestone 3c.
    let config = config::Config::default();
    info!("compositor config initialised from defaults");

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

    let drm_path = pick_drm_card_path(&initial_devices)?;
    info!(drm_path = %drm_path.display(), "selected DRM device");

    let drm_init = drm::open_display(&mut session, &drm_path).context("DRM device init failed")?;
    let drm::DrmInit {
        device: drm_device,
        fd: drm_fd,
        notifier: drm_notifier,
        outputs: drm_outputs,
    } = drm_init;

    let mut renderer = render::Renderer::new(drm_fd, drm_outputs, config.misc.wallpaper.clone())
        .context("render pipeline init failed")?;

    info!("phase: priming swapchains (one initial frame per output)");
    renderer.render_initial().context("initial render failed")?;
    info!("all outputs primed for scanout");

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

    wire_event_sources(&handle, notifier, udev, drm_notifier, libinput_backend)?;

    let mut state = State {
        session,
        loop_signal,
        drm_device,
        renderer,
        keyboard,
        config,
    };

    info!("entering event loop — type to generate events, super+shift+e to exit");
    event_loop
        .run(None, &mut state, |_state| {
            // Called after each batch of dispatched events. We have no
            // per-tick work yet; that changes once we add a renderer.
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

/// Insert all four event sources (libseat session, udev, DRM vblank,
/// libinput) into the calloop handle. Pulled out of `main` so the
/// init flow stays under clippy's `too_many_lines` threshold without
/// losing per-source visibility.
fn wire_event_sources(
    handle: &smithay::reexports::calloop::LoopHandle<'_, State>,
    session_notifier: LibSeatSessionNotifier,
    udev: UdevBackend,
    drm_notifier: smithay::backend::drm::DrmDeviceNotifier,
    libinput_backend: LibinputInputBackend,
) -> Result<()> {
    handle
        .insert_source(session_notifier, |event, (), _state| match event {
            smithay::backend::session::Event::PauseSession => warn!("session paused"),
            smithay::backend::session::Event::ActivateSession => info!("session activated"),
        })
        .map_err(|e| anyhow::anyhow!("failed to insert session source: {e}"))?;

    handle
        .insert_source(udev, |event, (), _state| match event {
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
        .insert_source(drm_notifier, |event, _meta, state| match event {
            smithay::backend::drm::DrmEvent::VBlank(crtc) => {
                if let Err(err) = state.renderer.render_for_crtc(crtc) {
                    // Don't kill the event loop on a render hiccup —
                    // log and let the next vblank try again. A
                    // persistent failure on one CRTC freezes that
                    // output but leaves the others (and the exit
                    // hotkey) responsive.
                    warn!(error = %err, ?crtc, "render_for_crtc failed on vblank");
                }
            }
            smithay::backend::drm::DrmEvent::Error(err) => {
                warn!(error = %err, "drm: event-source error");
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to insert drm source: {e}"))?;

    handle
        .insert_source(libinput_backend, |event, (), state| {
            log_input_event(&event);
            match event {
                InputEvent::Keyboard { event: ke } => state.handle_key(&ke),
                InputEvent::PointerMotion { event: pm } => {
                    state.renderer.on_pointer_motion(pm.dx(), pm.dy());
                }
                InputEvent::DeviceAdded { mut device } => {
                    apply_input_config(&mut device, &state.config.input);
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
