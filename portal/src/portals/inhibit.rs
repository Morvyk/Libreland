//! `org.freedesktop.impl.portal.Inhibit` — "don't idle/suspend/log out while
//! I'm doing this", plus the session-state monitor.
//!
//! The inhibition itself is logind's job: we take a delay/block inhibitor fd
//! from `org.freedesktop.login1.Manager.Inhibit` and hold it until the
//! frontend closes the request. Holding an fd is the whole mechanism — drop
//! it and the inhibition lifts, including if this process dies, which is
//! exactly the behaviour you want from something a video player asked for.
//!
//! The monitor half reports session state (running/query-end/ending) so an app
//! can save work before logout. Libreland has no logout choreography of its
//! own, so we report a running session and never query-end; the session object
//! still exists and closes cleanly, which is what callers actually depend on.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zbus::object_server::SignalEmitter;
use zbus::zvariant::{OwnedFd, OwnedObjectPath, OwnedValue};
use zbus::{Connection, Proxy, interface};

use super::{Cancel, PORTAL_PATH, SUCCESS, SessionSink, export_session, opt_str, ov};

/// Inhibit flags, as the portal spec numbers them.
const LOGOUT: u32 = 1;
const USER_SWITCH: u32 = 2;
const SUSPEND: u32 = 4;
const IDLE: u32 = 8;

/// Session states reported by the monitor.
const SESSION_RUNNING: u32 = 1;

pub struct Inhibit {
    /// Live monitor sessions, so `Close` can stop tracking them.
    monitors: Arc<Mutex<Vec<String>>>,
}

impl Inhibit {
    pub fn new() -> Self {
        Self {
            monitors: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl SessionSink for Arc<Mutex<Vec<String>>> {
    fn session_closed(&self, handle: &str) {
        if let Ok(mut sessions) = self.lock() {
            sessions.retain(|s| s != handle);
        }
    }
}

/// Translate portal flags into logind's `what` string.
fn logind_what(flags: u32) -> String {
    let mut what: Vec<&str> = Vec::new();
    if flags & IDLE != 0 {
        what.push("idle");
    }
    if flags & SUSPEND != 0 {
        what.push("sleep");
    }
    if flags & LOGOUT != 0 {
        // logind has no "logout" inhibitor; shutdown is the closest thing it
        // can actually enforce, and it's what GNOME maps this to.
        what.push("shutdown");
    }
    if flags & USER_SWITCH != 0 {
        what.push("handle-switch");
    }
    if what.is_empty() {
        what.push("idle");
    }
    what.join(":")
}

/// Take a logind inhibitor and return its fd. Dropping the fd releases it.
async fn take_inhibitor(what: &str, who: &str, why: &str) -> anyhow::Result<OwnedFd> {
    let system = Connection::system().await?;
    let manager = Proxy::new(
        &system,
        "org.freedesktop.login1",
        "/org/freedesktop/login1",
        "org.freedesktop.login1.Manager",
    )
    .await?;
    // "block" rather than "delay": the app is asking us to prevent the action,
    // not to be notified before it happens.
    let fd: OwnedFd = manager.call("Inhibit", &(what, who, why, "block")).await?;
    Ok(fd)
}

#[interface(name = "org.freedesktop.impl.portal.Inhibit")]
impl Inhibit {
    /// Hold an inhibition for as long as the request lives.
    ///
    /// This method has no return value by design: the frontend keeps the
    /// `Request` object alive and calls `Close()` when the app is done, so the
    /// work here is "acquire, then wait for the close".
    async fn inhibit(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _window: String,
        flags: u32,
        options: HashMap<String, OwnedValue>,
    ) {
        let reason = opt_str(&options, "reason").unwrap_or_else(|| "Application request".into());
        let what = logind_what(flags);
        tracing::info!(app = %app_id, %what, %reason, "inhibit requested");

        let cancel = Arc::new(Cancel::default());
        // The Request object is exported by hand rather than through
        // `with_request`: that helper unexports as soon as its future
        // resolves, and here the future *is* the lifetime of the inhibition.
        let exported = conn
            .object_server()
            .at(
                &handle,
                RequestGuard {
                    cancel: Arc::clone(&cancel),
                },
            )
            .await
            .unwrap_or(false);

        let who = if app_id.is_empty() {
            "libreland-portal".to_string()
        } else {
            app_id.clone()
        };
        let fd = match take_inhibitor(&what, &who, &reason).await {
            Ok(fd) => Some(fd),
            Err(err) => {
                // No logind (or a policy refusal) means we can't inhibit.
                // Reporting success anyway would be a lie an app can't detect;
                // reporting an error would break callers that treat any error
                // as fatal. Log it and keep the request alive but toothless.
                tracing::warn!(%err, "could not take a logind inhibitor");
                None
            }
        };

        let conn = conn.clone();
        tokio::spawn(async move {
            cancel.cancelled().await;
            // Dropping the fd is the release.
            drop(fd);
            tracing::info!(app = %app_id, "inhibition released");
            if exported {
                let _ = conn
                    .object_server()
                    .remove::<RequestGuard, _>(&handle)
                    .await;
            }
        });
    }

    /// Create a session that reports the session state.
    async fn create_monitor(
        &self,
        #[zbus(connection)] conn: &Connection,
        _handle: OwnedObjectPath,
        session_handle: OwnedObjectPath,
        app_id: String,
        _window: String,
    ) -> u32 {
        tracing::info!(app = %app_id, "inhibit monitor created");
        if let Ok(mut monitors) = self.monitors.lock() {
            monitors.push(session_handle.as_str().to_string());
        }
        export_session(conn, &session_handle, Arc::new(Arc::clone(&self.monitors))).await;

        // Report the initial state once the reply is on the wire — the peer
        // isn't listening for signals on a session it hasn't been told about.
        let conn = conn.clone();
        tokio::spawn(async move {
            let mut state: HashMap<String, OwnedValue> = HashMap::new();
            state.insert("screensaver-active".to_string(), ov(false));
            state.insert("session-state".to_string(), ov(SESSION_RUNNING));
            if let Ok(emitter) = SignalEmitter::new(&conn, PORTAL_PATH) {
                let _ = Inhibit::state_changed(&emitter, session_handle.as_ref(), state).await;
            }
        });
        SUCCESS
    }

    /// Response to a query-end we never send; nothing to do.
    fn query_end_response(&self, session_handle: OwnedObjectPath) {
        tracing::debug!(session = %session_handle.as_str(), "query end response");
    }

    #[zbus(signal)]
    async fn state_changed(
        emitter: &SignalEmitter<'_>,
        session_handle: zbus::zvariant::ObjectPath<'_>,
        state: HashMap<String, OwnedValue>,
    ) -> zbus::Result<()>;

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        3
    }
}

/// The `Request` object for a live inhibition: closing it lifts the hold.
struct RequestGuard {
    cancel: Arc<Cancel>,
}

#[interface(name = "org.freedesktop.impl.portal.Request")]
impl RequestGuard {
    fn close(&self) {
        self.cancel.cancel();
    }
}
