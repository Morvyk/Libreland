//! Client for Libreland's control socket.
//!
//! Two portals need the compositor to *do* something rather than just tell
//! them about the display: `Wallpaper` (show this image) and `GlobalShortcuts`
//! (route this key combination to me). Both ride the same JSON-over-unix
//! socket the `libreland msg` CLI speaks, found through `$LIBRELAND_SOCKET`.
//!
//! The protocol is line-delimited JSON: one request object per line in, one
//! `{"Ok":…}` / `{"Err":…}` per line out. Shortcut activations arrive on a
//! subscription that stays open for the life of the session, which is why the
//! subscriber below runs on its own thread with its own connection.

use std::io::{BufRead as _, BufReader, Write as _};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

use serde_json::json;

/// The compositor's control socket, as exported into our environment.
fn socket_path() -> anyhow::Result<PathBuf> {
    if let Ok(path) = std::env::var("LIBRELAND_SOCKET")
        && !path.is_empty()
    {
        return Ok(PathBuf::from(path));
    }
    // The compositor derives the name from $WAYLAND_DISPLAY; reconstruct it
    // for the case where only that made it into our environment.
    let runtime = std::env::var("XDG_RUNTIME_DIR")
        .map_err(|_| anyhow::anyhow!("neither LIBRELAND_SOCKET nor XDG_RUNTIME_DIR is set"))?;
    let display = std::env::var("WAYLAND_DISPLAY")
        .map_err(|_| anyhow::anyhow!("LIBRELAND_SOCKET is not set and WAYLAND_DISPLAY is unset"))?;
    Ok(PathBuf::from(runtime).join(format!("libreland-{display}.sock")))
}

/// Send one request and read one reply.
fn request(payload: &serde_json::Value) -> anyhow::Result<serde_json::Value> {
    let stream = UnixStream::connect(socket_path()?)?;
    // A hung compositor must not hang a portal method: both directions get a
    // deadline, and a timeout surfaces as a plain error.
    let timeout = std::time::Duration::from_secs(3);
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let mut writer = &stream;
    writeln!(writer, "{payload}")?;
    writer.flush()?;
    let mut line = String::new();
    BufReader::new(&stream).read_line(&mut line)?;
    let reply: serde_json::Value = serde_json::from_str(line.trim())?;
    if let Some(error) = reply.get("Err") {
        anyhow::bail!("compositor refused the request: {error}");
    }
    Ok(reply.get("Ok").cloned().unwrap_or(serde_json::Value::Null))
}

/// Ask the compositor to display `path` as the wallpaper right now.
pub fn set_wallpaper(path: &Path) -> anyhow::Result<()> {
    request(&json!({ "cmd": "set-wallpaper", "path": path.to_string_lossy() }))?;
    Ok(())
}

/// One shortcut, as the compositor's dynamic-bind table wants it.
pub struct Bind {
    /// Opaque id echoed back on activation. We namespace it by session so two
    /// apps can register the same shortcut id without colliding.
    pub id: String,
    /// A trigger in the portal's own notation, e.g. `LOGO+SHIFT+e`.
    pub trigger: String,
    pub description: String,
}

/// Register a shortcut. Returns the trigger the compositor actually bound,
/// which may differ from what was asked for (it normalizes the spelling).
pub fn register_bind(bind: &Bind) -> anyhow::Result<String> {
    let reply = request(&json!({
        "cmd": "register-bind",
        "id": bind.id,
        "trigger": bind.trigger,
        "description": bind.description,
    }))?;
    Ok(reply
        .get("Bind")
        .and_then(|b| b.get("trigger"))
        .and_then(|t| t.as_str())
        .unwrap_or(&bind.trigger)
        .to_string())
}

/// Drop a shortcut. Unregistering something that isn't registered is fine.
pub fn unregister_bind(id: &str) -> anyhow::Result<()> {
    request(&json!({ "cmd": "unregister-bind", "id": id }))?;
    Ok(())
}

/// A shortcut activation pushed by the compositor.
pub struct Activation {
    pub id: String,
    /// True on press, false on release.
    pub pressed: bool,
}

/// Subscribe to shortcut activations, calling `on_event` for each.
///
/// Runs until the socket closes (i.e. the compositor exits), on its own
/// thread: this is a long-lived stream, and the alternative — folding a raw
/// socket into the tokio reactor — buys nothing for one line-oriented reader.
pub fn subscribe_binds(on_event: impl Fn(Activation) + Send + 'static) -> anyhow::Result<()> {
    let stream = UnixStream::connect(socket_path()?)?;
    let mut writer = &stream;
    writeln!(
        writer,
        "{}",
        json!({ "cmd": "subscribe", "events": ["bind-activated"] })
    )?;
    writer.flush()?;
    std::thread::spawn(move || {
        let reader = BufReader::new(&stream);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            // The first line is the reply to `subscribe`; events follow.
            if value.get("event").and_then(|e| e.as_str()) != Some("bind-activated") {
                continue;
            }
            let Some(id) = value.get("id").and_then(|i| i.as_str()) else {
                continue;
            };
            on_event(Activation {
                id: id.to_string(),
                pressed: value
                    .get("pressed")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
            });
        }
        tracing::info!("compositor shortcut stream closed");
    });
    Ok(())
}
