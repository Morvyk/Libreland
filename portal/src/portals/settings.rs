//! `org.freedesktop.impl.portal.Settings` — desktop appearance, served to
//! apps and used by our own dialogs.
//!
//! This is the interface behind "my GTK/Qt/Electron app follows the system
//! dark mode". Apps read `org.freedesktop.appearance color-scheme` (and,
//! historically, the `org.gnome.desktop.interface` keys) and subscribe to
//! `SettingChanged`.
//!
//! GNOME's backend reads `GSettings` for this; we deliberately do not, because
//! that would make Libreland's appearance depend on a dconf database it
//! doesn't otherwise use. Instead the source of truth is a small keyfile,
//! `$XDG_CONFIG_HOME/libreland/portal.conf`, watched with inotify so an edit
//! (or `libreland`'s own future config write) repaints running apps live:
//!
//! ```ini
//! [org.freedesktop.appearance]
//! color-scheme = dark        ; dark | light | default
//! accent-color = #4a9eff
//! contrast = normal          ; normal | high
//!
//! [org.gnome.desktop.interface]
//! gtk-theme = Adwaita-dark   ; anything here is passed through verbatim
//! ```
//!
//! Unknown sections and keys pass straight through as strings, so a toolkit
//! that wants some namespace we've never heard of can be fed it without a
//! code change here.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

use zbus::object_server::SignalEmitter;
use zbus::zvariant::{OwnedValue, Structure, Value};
use zbus::{Connection, fdo, interface};

use super::{PORTAL_PATH, ov};

/// The `org.freedesktop.appearance` namespace, which is the standardized one.
const APPEARANCE: &str = "org.freedesktop.appearance";
/// GNOME's namespace. Still what a lot of GTK3-era software reads, so we
/// mirror the appearance values into it rather than making users care.
const GNOME_INTERFACE: &str = "org.gnome.desktop.interface";

/// `color-scheme` values, as the spec numbers them.
const SCHEME_DEFAULT: u32 = 0;
const SCHEME_DARK: u32 = 1;
const SCHEME_LIGHT: u32 = 2;

type Namespaces = HashMap<String, HashMap<String, OwnedValue>>;

/// The live settings, shared by the D-Bus interface, the file watcher and the
/// portal's own dialogs (which theme themselves from `color-scheme`).
pub struct SettingsState {
    values: RwLock<Namespaces>,
}

impl SettingsState {
    /// Read a plain string out of any namespace in the keyfile.
    pub fn string_in(&self, namespace: &str, key: &str) -> Option<String> {
        let values = self.values.read().ok()?;
        match &**values.get(namespace)?.get(key)? {
            Value::Str(s) => Some(s.to_string()),
            _ => None,
        }
    }

    /// Read a boolean out of any namespace in the keyfile. Values land as
    /// strings unless a key is specifically typed, so "true"/"yes"/"1" all
    /// count — this is a hand-edited file.
    pub fn bool_in(&self, namespace: &str, key: &str) -> bool {
        let Ok(values) = self.values.read() else {
            return false;
        };
        values
            .get(namespace)
            .and_then(|ns| ns.get(key))
            .is_some_and(|value| match &**value {
                Value::Bool(b) => *b,
                Value::Str(s) => matches!(
                    s.as_str().to_ascii_lowercase().as_str(),
                    "true" | "yes" | "1" | "on"
                ),
                _ => false,
            })
    }
}

static STATE: OnceLock<Arc<SettingsState>> = OnceLock::new();

/// The process-wide settings, loaded from disk on first use.
pub fn state() -> Arc<SettingsState> {
    Arc::clone(STATE.get_or_init(|| {
        Arc::new(SettingsState {
            values: RwLock::new(load()),
        })
    }))
}

/// Whether the desktop is currently in dark mode. Used to pick the palette for
/// the portal's own windows, so they agree with what we tell apps.
pub fn prefers_dark() -> bool {
    let state = state();
    let Ok(values) = state.values.read() else {
        return true;
    };
    values
        .get(APPEARANCE)
        .and_then(|ns| ns.get("color-scheme"))
        .and_then(|v| match &**v {
            Value::U32(n) => Some(*n),
            _ => None,
        })
        // "No preference" gets the dark palette: Libreland's own defaults are
        // dark, and a light dialog over a dark desktop is the jarring option.
        .is_none_or(|scheme| scheme != SCHEME_LIGHT)
}

fn config_path() -> PathBuf {
    let base = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map_or_else(
            || {
                PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
                    .join(".config")
            },
            PathBuf::from,
        );
    base.join("libreland/portal.conf")
}

/// Parse `#rrggbb` / `rgb(r,g,b)` / `r,g,b` into the `(ddd)` triple the
/// appearance spec uses (each channel 0.0–1.0).
fn parse_accent(raw: &str) -> Option<(f64, f64, f64)> {
    let raw = raw.trim();
    if let Some(hex) = raw.strip_prefix('#')
        && hex.len() == 6
    {
        let channel = |i: usize| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .ok()
                .map(|v| f64::from(v) / 255.0)
        };
        return Some((channel(0)?, channel(2)?, channel(4)?));
    }
    let inner = raw
        .strip_prefix("rgb(")
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(raw);
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return None;
    }
    let value = |s: &str| -> Option<f64> {
        let n: f64 = s.parse().ok()?;
        // Accept both 0–255 and 0.0–1.0 spellings.
        Some(if n > 1.0 { n / 255.0 } else { n })
    };
    Some((value(parts[0])?, value(parts[1])?, value(parts[2])?))
}

/// Coerce a keyfile string into the variant type the key is specified to
/// carry. Anything we don't know stays a string, which is the right answer
/// for pass-through keys like `gtk-theme`.
fn typed(namespace: &str, key: &str, raw: &str) -> OwnedValue {
    let raw = raw.trim();
    match (namespace, key) {
        (APPEARANCE, "color-scheme") => ov(match raw.to_ascii_lowercase().as_str() {
            "dark" | "prefer-dark" | "1" => SCHEME_DARK,
            "light" | "prefer-light" | "2" => SCHEME_LIGHT,
            _ => SCHEME_DEFAULT,
        }),
        (APPEARANCE, "contrast") => ov(u32::from(matches!(
            raw.to_ascii_lowercase().as_str(),
            "high" | "1"
        ))),
        (APPEARANCE, "accent-color") => {
            parse_accent(raw).map_or_else(|| ov(raw), |rgb| ov(Value::from(Structure::from(rgb))))
        }
        (GNOME_INTERFACE, "cursor-size" | "text-scaling-factor") => {
            raw.parse::<i32>().map_or_else(|_| ov(raw), ov)
        }
        _ => ov(raw),
    }
}

/// Read the keyfile (absent is fine) into namespaces, then fill in the
/// defaults and the mirrored GNOME keys.
fn load() -> Namespaces {
    let mut out: Namespaces = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(config_path()) {
        let mut section = String::new();
        for line in text.lines() {
            let line = line.trim();
            // `;` and `#` both start comments — but `#` also starts a colour
            // literal, so only treat it as a comment at the start of a line.
            if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                section = name.trim().to_string();
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            if section.is_empty() {
                continue;
            }
            let (key, value) = (key.trim(), value.trim());
            out.entry(section.clone())
                .or_default()
                .insert(key.to_string(), typed(&section, key, value));
        }
    }

    let appearance = out.entry(APPEARANCE.to_string()).or_default();
    appearance
        .entry("color-scheme".to_string())
        .or_insert_with(|| ov(SCHEME_DARK));
    appearance
        .entry("accent-color".to_string())
        .or_insert_with(|| ov(Value::from(Structure::from((0.290_196, 0.619_608, 1.0)))));
    appearance
        .entry("contrast".to_string())
        .or_insert_with(|| ov(0u32));

    // Mirror into GNOME's namespace for toolkits that only look there, and
    // seed the cursor keys from the same environment the compositor itself
    // reads (so a dialog's cursor matches every other window's).
    let dark = matches!(
        appearance.get("color-scheme").map(std::ops::Deref::deref),
        Some(Value::U32(n)) if *n == SCHEME_DARK
    );
    let gnome = out.entry(GNOME_INTERFACE.to_string()).or_default();
    gnome
        .entry("color-scheme".to_string())
        .or_insert_with(|| ov(if dark { "prefer-dark" } else { "default" }));
    if let Ok(theme) = std::env::var("XCURSOR_THEME") {
        gnome
            .entry("cursor-theme".to_string())
            .or_insert_with(|| ov(theme.as_str()));
    }
    if let Ok(size) = std::env::var("XCURSOR_SIZE")
        && let Ok(size) = size.parse::<i32>()
    {
        gnome
            .entry("cursor-size".to_string())
            .or_insert_with(|| ov(size));
    }
    out
}

/// Does `pattern` (a namespace, optionally ending in `*`) select `namespace`?
fn matches(pattern: &str, namespace: &str) -> bool {
    pattern.strip_suffix('*').map_or_else(
        || pattern == namespace,
        |prefix| namespace.starts_with(prefix),
    )
}

pub struct Settings {
    state: Arc<SettingsState>,
}

impl Settings {
    pub fn new() -> Self {
        Self { state: state() }
    }
}

#[interface(name = "org.freedesktop.impl.portal.Settings")]
impl Settings {
    /// Every namespace the caller asked for (all of them, if it asked for
    /// nothing).
    fn read_all(&self, namespaces: Vec<String>) -> fdo::Result<Namespaces> {
        let values = self
            .state
            .values
            .read()
            .map_err(|_| fdo::Error::Failed("settings lock poisoned".into()))?;
        if namespaces.is_empty() {
            return Ok(values.clone());
        }
        Ok(values
            .iter()
            .filter(|(name, _)| namespaces.iter().any(|p| matches(p, name)))
            .map(|(name, keys)| (name.clone(), keys.clone()))
            .collect())
    }

    /// Portal spec v1. The value comes back double-wrapped in a variant — that
    /// is not a mistake here, it is what the frontend unwraps, and what every
    /// other backend sends.
    fn read(&self, namespace: &str, key: &str) -> fdo::Result<OwnedValue> {
        let value = self.lookup(namespace, key)?;
        Ok(ov(Value::from(value)))
    }

    /// Portal spec v2 — same lookup, without the historical extra wrapper.
    fn read_one(&self, namespace: &str, key: &str) -> fdo::Result<OwnedValue> {
        self.lookup(namespace, key)
    }

    #[zbus(signal)]
    pub async fn setting_changed(
        emitter: &SignalEmitter<'_>,
        namespace: &str,
        key: &str,
        value: Value<'_>,
    ) -> zbus::Result<()>;

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        2
    }
}

impl Settings {
    fn lookup(&self, namespace: &str, key: &str) -> fdo::Result<OwnedValue> {
        let values = self
            .state
            .values
            .read()
            .map_err(|_| fdo::Error::Failed("settings lock poisoned".into()))?;
        values
            .get(namespace)
            .and_then(|ns| ns.get(key))
            .cloned()
            // The spec's own error for "no such setting"; callers treat it as
            // "use your own default" rather than as a failure.
            .ok_or_else(|| {
                fdo::Error::Failed(format!(
                    "org.freedesktop.portal.Error.NotFound: no setting {namespace}.{key}"
                ))
            })
    }
}

/// Watch the keyfile and emit `SettingChanged` for anything that moved.
///
/// inotify rather than a poll timer, and on the *directory* rather than the
/// file, because every editor worth using writes config by atomic rename —
/// watching the inode would see one modification and then never fire again.
pub fn spawn_watcher(conn: Connection) {
    let path = config_path();
    let Some(dir) = path.parent().map(std::path::Path::to_path_buf) else {
        return;
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<()>();

    std::thread::spawn(move || {
        use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};
        let Ok(inotify) = Inotify::init(InitFlags::empty()) else {
            return;
        };
        if inotify
            .add_watch(
                &dir,
                AddWatchFlags::IN_CLOSE_WRITE
                    | AddWatchFlags::IN_MOVED_TO
                    | AddWatchFlags::IN_CREATE
                    | AddWatchFlags::IN_DELETE,
            )
            .is_err()
        {
            return;
        }
        let want = path.file_name().map(std::ffi::OsString::from);
        loop {
            let Ok(events) = inotify.read_events() else {
                return;
            };
            let touched = events
                .iter()
                .any(|e| e.name.as_ref().map(std::ffi::OsString::from) == want);
            if touched && tx.send(()).is_err() {
                return;
            }
        }
    });

    tokio::spawn(async move {
        while rx.recv().await.is_some() {
            // Coalesce the burst a save produces (create, write, rename) into
            // one reload.
            tokio::time::sleep(std::time::Duration::from_millis(80)).await;
            while rx.try_recv().is_ok() {}

            let state = state();
            let fresh = load();
            let changed: Vec<(String, String, OwnedValue)> = {
                let Ok(current) = state.values.read() else {
                    continue;
                };
                let mut diff = Vec::new();
                for (namespace, keys) in &fresh {
                    for (key, value) in keys {
                        if current.get(namespace).and_then(|ns| ns.get(key)) != Some(value) {
                            diff.push((namespace.clone(), key.clone(), value.clone()));
                        }
                    }
                }
                diff
            };
            if changed.is_empty() {
                continue;
            }
            if let Ok(mut current) = state.values.write() {
                *current = fresh;
            }
            let Ok(emitter) = SignalEmitter::new(&conn, PORTAL_PATH) else {
                continue;
            };
            for (namespace, key, value) in changed {
                tracing::info!(%namespace, %key, "settings changed");
                let _ =
                    Settings::setting_changed(&emitter, &namespace, &key, Value::from(value)).await;
            }
        }
    });
}
