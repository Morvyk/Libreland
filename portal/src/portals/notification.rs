//! `org.freedesktop.impl.portal.Notification` — sandboxed apps posting
//! notifications.
//!
//! There is already a notification server on the session bus
//! (`org.freedesktop.Notifications`) — the user's bar, `mako`, `dunst`,
//! whatever they run. Drawing our own popups here would put two notification
//! systems on one desktop, so this portal translates and forwards, then
//! translates the daemon's `ActionInvoked` back into the portal signal the
//! frontend expects.
//!
//! The translation that matters is identity: portal notifications are keyed by
//! `(app_id, id)` strings the app chooses, while the daemon hands out u32 ids.
//! We keep the mapping so `RemoveNotification` and action callbacks can find
//! their way home.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zbus::object_server::SignalEmitter;
use zbus::zvariant::{OwnedValue, Value};
use zbus::{Connection, Proxy, interface};

use super::{PORTAL_PATH, opt_str, ov};

/// `(app_id, portal id)` for one live notification, keyed by the daemon's id.
type Live = Arc<Mutex<HashMap<u32, (String, String)>>>;

pub struct Notification {
    live: Live,
    /// Set once the daemon's `ActionInvoked` signal is being watched.
    watching: Arc<std::sync::atomic::AtomicBool>,
}

impl Notification {
    pub fn new() -> Self {
        Self {
            live: Arc::new(Mutex::new(HashMap::new())),
            watching: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    async fn daemon(conn: &Connection) -> zbus::Result<Proxy<'static>> {
        Proxy::new_owned(
            conn.clone(),
            "org.freedesktop.Notifications".to_string(),
            "/org/freedesktop/Notifications".to_string(),
            "org.freedesktop.Notifications".to_string(),
        )
        .await
    }

    /// Start forwarding the daemon's `ActionInvoked` to our own signal, once.
    fn watch_actions(&self, conn: &Connection) {
        use std::sync::atomic::Ordering;
        if self.watching.swap(true, Ordering::SeqCst) {
            return;
        }
        let conn = conn.clone();
        let live = Arc::clone(&self.live);
        tokio::spawn(async move {
            use futures_util::StreamExt as _;
            let Ok(daemon) = Notification::daemon(&conn).await else {
                return;
            };
            let Ok(mut actions) = daemon.receive_signal("ActionInvoked").await else {
                return;
            };
            while let Some(message) = actions.next().await {
                let Ok((id, action)) = message.body().deserialize::<(u32, String)>() else {
                    continue;
                };
                let Some((app_id, portal_id)) =
                    live.lock().ok().and_then(|map| map.get(&id).cloned())
                else {
                    continue;
                };
                // The spec's "default" action means "the notification body was
                // clicked"; it travels as an empty action name.
                let action = if action == "default" {
                    String::new()
                } else {
                    action
                };
                if let Ok(emitter) = SignalEmitter::new(&conn, PORTAL_PATH) {
                    let _ = Notification::action_invoked(
                        &emitter,
                        &app_id,
                        &portal_id,
                        &action,
                        Vec::new(),
                    )
                    .await;
                }
            }
        });
    }
}

/// Portal priority → freedesktop urgency (0 low, 1 normal, 2 critical).
fn urgency(priority: Option<&str>) -> u8 {
    match priority {
        Some("low") => 0,
        Some("urgent") => 2,
        _ => 1,
    }
}

#[interface(name = "org.freedesktop.impl.portal.Notification")]
impl Notification {
    async fn add_notification(
        &self,
        #[zbus(connection)] conn: &Connection,
        app_id: String,
        id: String,
        notification: HashMap<String, OwnedValue>,
    ) {
        self.watch_actions(conn);
        let title = opt_str(&notification, "title").unwrap_or_default();
        let body = opt_str(&notification, "body").unwrap_or_default();
        let icon = opt_str(&notification, "icon").unwrap_or_default();
        let priority = opt_str(&notification, "priority");
        tracing::info!(app = %app_id, %id, "notification");

        // Buttons: `a{sv}` entries with `label` and `action`. The default
        // action (the whole notification being clicked) is a separate key.
        let mut actions: Vec<String> = Vec::new();
        if notification.contains_key("default-action") {
            actions.push("default".to_string());
            actions.push("Open".to_string());
        }
        if let Some(Value::Array(buttons)) = notification.get("buttons").map(std::ops::Deref::deref)
        {
            for button in buttons.iter() {
                let Value::Dict(entries) = button else {
                    continue;
                };
                // Dict lookup by &str needs a Value key, so iterate instead:
                // these dictionaries have two entries.
                let get = |want: &str| -> Option<String> {
                    entries.iter().find_map(|(key, value)| match (key, value) {
                        (Value::Str(key), Value::Str(text)) if key.as_str() == want => {
                            Some(text.to_string())
                        }
                        (Value::Str(key), Value::Value(inner)) if key.as_str() == want => {
                            match &**inner {
                                Value::Str(text) => Some(text.to_string()),
                                _ => None,
                            }
                        }
                        _ => None,
                    })
                };
                if let (Some(action), Some(label)) = (get("action"), get("label")) {
                    actions.push(action);
                    actions.push(label);
                }
            }
        }

        let mut hints: HashMap<&str, Value<'_>> = HashMap::new();
        hints.insert("urgency", Value::U8(urgency(priority.as_deref())));
        // Tag the sender so a daemon with per-app rules can act on it.
        if !app_id.is_empty() {
            hints.insert("desktop-entry", Value::from(app_id.clone()));
        }

        let Ok(daemon) = Self::daemon(conn).await else {
            tracing::warn!("no notification daemon on the bus; notification dropped");
            return;
        };
        let sent: zbus::Result<u32> = daemon
            .call(
                "Notify",
                &(
                    app_id.as_str(),
                    // replaces_id 0: replacement is handled through our own
                    // (app_id, id) map below, since the app chose the id.
                    0u32,
                    icon.as_str(),
                    title.as_str(),
                    body.as_str(),
                    actions.as_slice(),
                    hints,
                    // Let the daemon's own policy decide the timeout.
                    -1i32,
                ),
            )
            .await;
        match sent {
            Ok(daemon_id) => {
                if let Ok(mut live) = self.live.lock() {
                    // Replacing an id the app already used: forget the old one.
                    live.retain(|_, (owner, key)| !(owner == &app_id && key == &id));
                    live.insert(daemon_id, (app_id, id));
                }
            }
            Err(err) => tracing::warn!(%err, "the notification daemon rejected the notification"),
        }
    }

    async fn remove_notification(
        &self,
        #[zbus(connection)] conn: &Connection,
        app_id: String,
        id: String,
    ) {
        let daemon_id = self.live.lock().ok().and_then(|mut live| {
            let found = live
                .iter()
                .find(|(_, (owner, key))| owner == &app_id && key == &id)
                .map(|(daemon_id, _)| *daemon_id);
            if let Some(daemon_id) = found {
                live.remove(&daemon_id);
            }
            found
        });
        let Some(daemon_id) = daemon_id else {
            return;
        };
        if let Ok(daemon) = Self::daemon(conn).await {
            let _: zbus::Result<()> = daemon.call("CloseNotification", &(daemon_id,)).await;
        }
    }

    #[zbus(signal)]
    async fn action_invoked(
        emitter: &SignalEmitter<'_>,
        app_id: &str,
        id: &str,
        action: &str,
        parameter: Vec<OwnedValue>,
    ) -> zbus::Result<()>;

    /// What of the notification schema we actually honour. The frontend
    /// validates against this and drops anything else before it reaches us.
    #[zbus(property, name = "SupportedOptions")]
    fn supported_options(&self) -> HashMap<String, OwnedValue> {
        let mut options = HashMap::new();
        options.insert(
            "category".to_string(),
            ov(Vec::<String>::new()),
        );
        options.insert(
            "priority".to_string(),
            ov(vec![
                "low".to_string(),
                "normal".to_string(),
                "high".to_string(),
                "urgent".to_string(),
            ]),
        );
        options
    }

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        2
    }
}
