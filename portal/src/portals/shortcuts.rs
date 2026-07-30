//! `org.freedesktop.impl.portal.GlobalShortcuts` — keyboard shortcuts an app
//! can claim while it isn't focused.
//!
//! Neither backend this portal replaces implemented this: `-wlr` has no
//! keyboard machinery and `-gtk` has no compositor to ask. It works here
//! because the compositor grew a matching primitive — binds registered over
//! the control IPC, which fire an event instead of a compositor action (see
//! `Request::RegisterBind` in the compositor's `ipc` module).
//!
//! Flow: the app creates a session, asks to bind some shortcuts (each with a
//! `preferred_trigger` like `LOGO+SHIFT+e`), the user approves them in one
//! dialog, and from then on every press arrives as an `Activated` signal
//! until the session closes.
//!
//! Shortcut ids are namespaced by session before they reach the compositor,
//! so two apps that both call their shortcut `toggle` don't collide.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, OnceLock};

use zbus::object_server::SignalEmitter;
use zbus::zvariant::{ObjectPath, OwnedObjectPath, OwnedValue};
use zbus::{Connection, interface};

use crate::ipc;
use crate::ui;
use crate::ui::prompt::{Prompt, Spec};

use super::{
    CANCELLED, FAILED, PORTAL_PATH, PortalResult, SUCCESS, SessionSink, empty, export_session,
    opt_str, ov, with_request,
};

/// One bound shortcut.
#[derive(Clone, Debug)]
struct Shortcut {
    /// The app's own id for it.
    id: String,
    description: String,
    /// The trigger the compositor actually bound, for `ListShortcuts`.
    trigger: String,
}

type Sessions = Arc<Mutex<HashMap<String, Vec<Shortcut>>>>;

/// Shared with the activation listener, which starts before any session
/// exists and outlives all of them.
static SESSIONS: OnceLock<Sessions> = OnceLock::new();

fn sessions() -> Sessions {
    Arc::clone(SESSIONS.get_or_init(|| Arc::new(Mutex::new(HashMap::new()))))
}

/// Compositor-facing id: `<session path>\u{1}<app's shortcut id>`.
///
/// The separator is a control character precisely because it can't appear in
/// either half, so splitting it back apart is unambiguous.
fn compositor_id(session: &str, shortcut: &str) -> String {
    format!("{session}\u{1}{shortcut}")
}

fn split_id(id: &str) -> Option<(&str, &str)> {
    id.split_once('\u{1}')
}

pub struct GlobalShortcuts {
    sessions: Sessions,
}

impl GlobalShortcuts {
    pub fn new() -> Self {
        Self {
            sessions: sessions(),
        }
    }
}

/// Unbinding a session's shortcuts when the peer closes it.
struct Sink(Sessions);

impl SessionSink for Sink {
    fn session_closed(&self, handle: &str) {
        let Ok(mut sessions) = self.0.lock() else {
            return;
        };
        if let Some(shortcuts) = sessions.remove(handle) {
            for shortcut in shortcuts {
                let _ = ipc::unregister_bind(&compositor_id(handle, &shortcut.id));
            }
            tracing::info!(session = handle, "global shortcuts released");
        }
    }
}

/// Read the `shortcuts` argument: `a(sa{sv})`, each entry an id plus
/// `description` and `preferred_trigger`.
fn parse_shortcuts(raw: &[(String, HashMap<String, OwnedValue>)]) -> Vec<(String, String, String)> {
    raw.iter()
        .map(|(id, options)| {
            (
                id.clone(),
                opt_str(options, "description").unwrap_or_else(|| id.clone()),
                opt_str(options, "preferred_trigger").unwrap_or_default(),
            )
        })
        .collect()
}

/// `a(sa{sv})` for `ListShortcuts` / the `shortcuts` result.
fn shortcuts_value(shortcuts: &[Shortcut]) -> OwnedValue {
    let listed: Vec<(String, HashMap<String, OwnedValue>)> = shortcuts
        .iter()
        .map(|shortcut| {
            let mut options = HashMap::new();
            options.insert("description".to_string(), ov(shortcut.description.as_str()));
            // The spec's name for "what the user will actually press".
            options.insert(
                "trigger_description".to_string(),
                ov(shortcut.trigger.as_str()),
            );
            (shortcut.id.clone(), options)
        })
        .collect();
    ov(listed)
}

#[interface(name = "org.freedesktop.impl.portal.GlobalShortcuts")]
impl GlobalShortcuts {
    async fn create_session(
        &self,
        #[zbus(connection)] conn: &Connection,
        _handle: OwnedObjectPath,
        session_handle: OwnedObjectPath,
        app_id: String,
        _options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        tracing::info!(app = %app_id, session = %session_handle.as_str(), "shortcuts session");
        if let Ok(mut sessions) = self.sessions.lock() {
            sessions.insert(session_handle.as_str().to_string(), Vec::new());
        }
        export_session(
            conn,
            &session_handle,
            Arc::new(Sink(Arc::clone(&self.sessions))),
        )
        .await;
        (SUCCESS, HashMap::new())
    }

    async fn bind_shortcuts(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        session_handle: OwnedObjectPath,
        shortcuts: Vec<(String, HashMap<String, OwnedValue>)>,
        _parent_window: String,
        _options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        let wanted = parse_shortcuts(&shortcuts);
        let session = session_handle.as_str().to_string();
        tracing::info!(session = %session, count = wanted.len(), "bind shortcuts");
        if wanted.is_empty() {
            return (SUCCESS, HashMap::new());
        }

        // One dialog for the whole set: approving shortcuts one at a time
        // would be a prompt storm for an app that wants five of them.
        let mut body = String::from(
            "An application wants these keyboard shortcuts to work even while it isn't focused:\n",
        );
        for (_, description, trigger) in &wanted {
            if trigger.is_empty() {
                let _ = write!(body, "\n• {description}  (no key requested)");
            } else {
                let _ = write!(body, "\n• {description}  —  {trigger}");
            }
        }
        let spec = Spec {
            title: "Allow global shortcuts?".to_string(),
            subtitle: String::new(),
            body,
            accept_label: Some("Allow".to_string()),
            deny_label: Some("Deny".to_string()),
            destructive: false,
            toggles: Vec::new(),
        };
        let sessions = Arc::clone(&self.sessions);

        with_request(conn, &handle, |cancel| async move {
            match ui::dialog(Prompt::new(spec), cancel).await {
                Ok(prompt) if prompt.accepted => {}
                Ok(_) => return empty(CANCELLED),
                Err(err) => {
                    tracing::error!(%err, "shortcut dialog failed");
                    return empty(FAILED);
                }
            }

            let mut bound: Vec<Shortcut> = Vec::new();
            for (id, description, trigger) in wanted {
                // A shortcut with no preferred trigger has nothing for us to
                // bind. The spec allows the backend to offer its own
                // configuration UI here; we don't have one, so it's recorded
                // (and listed) but not live, rather than silently invented.
                if trigger.is_empty() {
                    bound.push(Shortcut {
                        id,
                        description,
                        trigger: String::new(),
                    });
                    continue;
                }
                let bind = ipc::Bind {
                    id: compositor_id(&session, &id),
                    trigger: trigger.clone(),
                    description: description.clone(),
                };
                match ipc::register_bind(&bind) {
                    Ok(normalized) => bound.push(Shortcut {
                        id,
                        description,
                        trigger: normalized,
                    }),
                    Err(err) => {
                        // One unusable trigger (a key this keymap doesn't
                        // have) shouldn't sink the whole request.
                        tracing::warn!(%err, %trigger, "could not bind a shortcut");
                        bound.push(Shortcut {
                            id,
                            description,
                            trigger: String::new(),
                        });
                    }
                }
            }

            let value = shortcuts_value(&bound);
            if let Ok(mut sessions) = sessions.lock() {
                sessions.insert(session.clone(), bound);
            }
            let mut results = HashMap::new();
            results.insert("shortcuts".to_string(), value);
            (SUCCESS, results)
        })
        .await
    }

    async fn list_shortcuts(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        session_handle: OwnedObjectPath,
    ) -> PortalResult {
        let listed = self
            .sessions
            .lock()
            .ok()
            .and_then(|sessions| sessions.get(session_handle.as_str()).cloned())
            .unwrap_or_default();
        with_request(conn, &handle, |_cancel| async move {
            let mut results = HashMap::new();
            results.insert("shortcuts".to_string(), shortcuts_value(&listed));
            (SUCCESS, results)
        })
        .await
    }

    /// The spec's "let the user re-bind these" entry point. We have no
    /// shortcut editor — the compositor's binds live in its Lua config — so
    /// this is a no-op rather than a dialog that can't change anything.
    fn configure_shortcuts(
        &self,
        session_handle: OwnedObjectPath,
        _parent_window: String,
        _options: HashMap<String, OwnedValue>,
    ) {
        tracing::info!(
            session = %session_handle.as_str(),
            "ConfigureShortcuts is not supported (no shortcut editor)"
        );
    }

    #[zbus(signal)]
    async fn activated(
        emitter: &SignalEmitter<'_>,
        session_handle: ObjectPath<'_>,
        shortcut_id: &str,
        timestamp: u64,
        options: HashMap<String, OwnedValue>,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn deactivated(
        emitter: &SignalEmitter<'_>,
        session_handle: ObjectPath<'_>,
        shortcut_id: &str,
        timestamp: u64,
        options: HashMap<String, OwnedValue>,
    ) -> zbus::Result<()>;

    #[zbus(signal)]
    async fn shortcuts_changed(
        emitter: &SignalEmitter<'_>,
        session_handle: ObjectPath<'_>,
        shortcuts: Vec<(String, HashMap<String, OwnedValue>)>,
    ) -> zbus::Result<()>;

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        1
    }
}

/// Subscribe to the compositor's bind activations and turn them into portal
/// signals.
///
/// Best-effort: without a compositor connection the rest of the portal is
/// unaffected, and shortcuts simply never fire (which `BindShortcuts` already
/// reports honestly, since registering them would have failed too).
pub fn spawn_listener(conn: Connection) {
    let sessions = sessions();
    let handle = tokio::runtime::Handle::current();
    let result = ipc::subscribe_binds(move |activation| {
        let Some((session, shortcut)) = split_id(&activation.id) else {
            return;
        };
        // Ignore activations for sessions that have gone away; the
        // compositor may still have the bind for a moment.
        let known = sessions
            .lock()
            .ok()
            .is_some_and(|sessions| sessions.contains_key(session));
        if !known {
            return;
        }
        let (session, shortcut) = (session.to_string(), shortcut.to_string());
        let conn = conn.clone();
        let pressed = activation.pressed;
        // The subscriber runs on its own thread; hop back onto the runtime
        // to emit, since signal emission is async.
        handle.spawn(async move {
            let Ok(path) = ObjectPath::try_from(session.clone()) else {
                return;
            };
            let Ok(emitter) = SignalEmitter::new(&conn, PORTAL_PATH) else {
                return;
            };
            // Portal timestamps are milliseconds since an arbitrary epoch;
            // consumers only compare them, so the wall clock is fine.
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
            let options = HashMap::new();
            let _ = if pressed {
                GlobalShortcuts::activated(&emitter, path, &shortcut, timestamp, options).await
            } else {
                GlobalShortcuts::deactivated(&emitter, path, &shortcut, timestamp, options).await
            };
        });
    });
    match result {
        Ok(()) => tracing::info!("listening for compositor shortcut activations"),
        Err(err) => tracing::warn!(%err, "global shortcuts will not fire (no compositor IPC)"),
    }
}
