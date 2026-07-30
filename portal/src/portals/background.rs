//! `org.freedesktop.impl.portal.Background` — background running and
//! autostart.
//!
//! Two halves with very different weights:
//!
//! * `GetAppState` / `NotifyBackground` exist so a desktop shell can show
//!   "these apps are running in the background" and offer to stop them.
//!   That needs a shell component tracking app lifetimes, which Libreland
//!   doesn't have, so we report nothing running and allow anything that asks.
//!   Silently allowing is the same answer the user would give to a dialog they
//!   didn't ask for, and it avoids inventing a permission UI for a permission
//!   nobody here can enforce.
//! * `EnableAutostart` is real and is implemented: it writes a desktop entry
//!   into `~/.config/autostart`, which is what every session honours.

use std::collections::HashMap;
use std::path::PathBuf;

use zbus::object_server::SignalEmitter;
use zbus::zvariant::{OwnedObjectPath, OwnedValue};
use zbus::interface;

use super::{PortalResult, SUCCESS, ov};

/// `NotifyBackground` result codes.
const BACKGROUND_ALLOW: u32 = 1;

pub struct Background;

impl Background {
    pub const fn new() -> Self {
        Self
    }
}

fn autostart_dir() -> PathBuf {
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()));
    std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map_or_else(|| home.join(".config"), PathBuf::from)
        .join("autostart")
}

/// Quote an argv into an `Exec=` line the way the Desktop Entry spec wants.
fn exec_line(commandline: &[String]) -> String {
    commandline
        .iter()
        .map(|arg| {
            if arg.contains([' ', '\t', '"', '\'', '\\']) {
                format!("\"{}\"", arg.replace('\\', r"\\").replace('"', "\\\""))
            } else {
                arg.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[interface(name = "org.freedesktop.impl.portal.Background")]
impl Background {
    /// No shell-side app tracking, so nothing to report.
    fn get_app_state(&self) -> HashMap<String, OwnedValue> {
        HashMap::new()
    }

    fn notify_background(
        &self,
        _handle: OwnedObjectPath,
        app_id: &str,
        name: &str,
    ) -> PortalResult {
        tracing::info!(app = %app_id, %name, "app reported running in the background");
        let mut results = HashMap::new();
        results.insert("result".to_string(), ov(BACKGROUND_ALLOW));
        (SUCCESS, results)
    }

    /// Write (or remove) `~/.config/autostart/<app_id>.desktop`.
    fn enable_autostart(
        &self,
        app_id: String,
        enable: bool,
        commandline: Vec<String>,
        flags: u32,
    ) -> bool {
        let dir = autostart_dir();
        let file = dir.join(format!("{app_id}.desktop"));
        if !enable {
            let removed = std::fs::remove_file(&file).is_ok();
            tracing::info!(app = %app_id, removed, "autostart disabled");
            return true;
        }
        if commandline.is_empty() {
            tracing::warn!(app = %app_id, "autostart requested with an empty command line");
            return false;
        }
        if std::fs::create_dir_all(&dir).is_err() {
            return false;
        }
        // Flag bit 1 means "activatable": the app wants to be D-Bus activated
        // rather than spawned. We have no session-wide activation manager, so
        // the entry is written either way and the command line is what runs.
        let dbus_activatable = flags & 1 != 0;
        let entry = format!(
            "[Desktop Entry]\n\
             Type=Application\n\
             Name={app_id}\n\
             Exec={}\n\
             X-Flatpak={app_id}\n\
             DBusActivatable={dbus_activatable}\n\
             X-GNOME-Autostart-enabled=true\n",
            exec_line(&commandline)
        );
        match std::fs::write(&file, entry) {
            Ok(()) => {
                tracing::info!(app = %app_id, path = %file.display(), "autostart enabled");
                true
            }
            Err(err) => {
                tracing::error!(%err, app = %app_id, "could not write the autostart entry");
                false
            }
        }
    }

    #[zbus(signal)]
    async fn running_applications_changed(emitter: &SignalEmitter<'_>) -> zbus::Result<()>;

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        2
    }
}
