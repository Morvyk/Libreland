//! `org.freedesktop.impl.portal.ScreenCast` — screen sharing.
//!
//! Three calls in sequence, all keyed on a session object:
//!
//! 1. `CreateSession` — the app declares intent; we allocate state.
//! 2. `SelectSources` — the user chooses *what* to share. This is where the
//!    monitor picker comes up. `xdg-desktop-portal-wlr` shells out to an
//!    external chooser here (`slurp`, a dmenu list) and needs a config file to
//!    say which; ours is built in, which removes both the configuration step
//!    and the failure mode where the chooser is missing or crashes.
//! 3. `Start` — the cast begins and we answer with a `PipeWire` node id the app
//!    connects to.
//!
//! Sessions persist across `Start` only as long as the app keeps them: closing
//! the session object stops the cast thread, and so does the app exiting,
//! because the frontend closes the session for it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zbus::zvariant::{OwnedObjectPath, OwnedValue, Value};
use zbus::{Connection, interface};

use crate::pw;
use crate::ui;
use crate::ui::picker::OutputPicker;

use super::{
    CANCELLED, Cancel, FAILED, PortalResult, SUCCESS, SessionSink, empty, export_session, opt_u32,
    ov, with_request,
};

/// Source types, as the spec's bitmask.
const SOURCE_MONITOR: u32 = 1;

/// Cursor modes.
const CURSOR_HIDDEN: u32 = 1;
const CURSOR_EMBEDDED: u32 = 2;

/// Persist modes: 0 none, 1 while the app runs, 2 until revoked.
const PERSIST_NONE: u32 = 0;

/// What the user's config says about the pointer in a screencast.
///
/// The app's `cursor_mode` is only ever a guess about what the *user* wants —
/// Chromium picks HIDDEN whenever it thinks the backend can't do better, and
/// once an app persists its source choice the picker (and its toggle) never
/// appears again. So the config gets the final word:
///
/// ```ini
/// [screencast]
/// cursor = always   ; always | never | app (default)
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
enum CursorPolicy {
    /// Follow the app's request, and the picker toggle when it's shown.
    App,
    Always,
    Never,
}

impl CursorPolicy {
    fn read() -> Self {
        match crate::portals::settings::state()
            .string_in("screencast", "cursor")
            .as_deref()
            .map(str::trim)
        {
            Some("always" | "shown" | "true" | "yes") => Self::Always,
            Some("never" | "hidden" | "false" | "no") => Self::Never,
            _ => Self::App,
        }
    }

    /// Apply the policy to whatever the app and the picker settled on.
    fn apply(self, chosen: bool) -> bool {
        match self {
            Self::App => chosen,
            Self::Always => true,
            Self::Never => false,
        }
    }
}

/// One screencast session.
#[derive(Default)]
struct SessionState {
    /// Connector name chosen in `SelectSources`.
    output: Option<String>,
    cursor: bool,
    /// How long the app may reuse this choice without asking again.
    persist_mode: u32,
    /// Set when the cast thread is running; tripping it stops the thread.
    stop: Option<Arc<Cancel>>,
}

type Sessions = Arc<Mutex<HashMap<String, SessionState>>>;

pub struct ScreenCast {
    sessions: Sessions,
}

impl ScreenCast {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// Stopping a session's cast thread when the peer closes it.
struct Sink(Sessions);

impl SessionSink for Sink {
    fn session_closed(&self, handle: &str) {
        let Ok(mut sessions) = self.0.lock() else {
            return;
        };
        if let Some(state) = sessions.remove(handle)
            && let Some(stop) = state.stop
        {
            tracing::info!(session = handle, "screencast session closed");
            stop.cancel();
        }
    }
}

/// `restore_data` is `(sv)`: a backend id, a version, and whatever that
/// backend stashed. Ours stashes the connector name and the cursor choice.
fn parse_restore(options: &HashMap<String, OwnedValue>) -> Option<(String, bool)> {
    let Some(Value::Structure(entry)) = options.get("restore_data").map(std::ops::Deref::deref)
    else {
        return None;
    };
    let fields = entry.fields();
    // Only restore data we wrote: another backend's blob means nothing here.
    if !matches!(fields.first(), Some(Value::Str(vendor)) if vendor.as_str() == super::BUS_NAME) {
        return None;
    }
    let Some(Value::Value(payload)) = fields.get(2) else {
        return None;
    };
    let Value::Str(encoded) = &**payload else {
        return None;
    };
    // "<connector>:<cursor>" — small enough that a struct would be ceremony.
    let (output, cursor) = encoded.as_str().split_once(':')?;
    Some((output.to_string(), cursor == "1"))
}

fn build_restore(output: &str, cursor: bool) -> OwnedValue {
    ov(Value::from(zbus::zvariant::Structure::from((
        super::BUS_NAME.to_string(),
        1u32,
        Value::from(format!("{output}:{}", u8::from(cursor))),
    ))))
}

#[interface(name = "org.freedesktop.impl.portal.ScreenCast")]
impl ScreenCast {
    async fn create_session(
        &self,
        #[zbus(connection)] conn: &Connection,
        _handle: OwnedObjectPath,
        session_handle: OwnedObjectPath,
        app_id: String,
        _options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        tracing::info!(app = %app_id, session = %session_handle.as_str(), "screencast session");
        if let Ok(mut sessions) = self.sessions.lock() {
            sessions.insert(session_handle.as_str().to_string(), SessionState::default());
        }
        export_session(
            conn,
            &session_handle,
            Arc::new(Sink(Arc::clone(&self.sessions))),
        )
        .await;
        (SUCCESS, HashMap::new())
    }

    async fn select_sources(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        session_handle: OwnedObjectPath,
        app_id: String,
        options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        let cursor_mode = opt_u32(&options, "cursor_mode").unwrap_or(CURSOR_HIDDEN);
        let persist_mode = opt_u32(&options, "persist_mode").unwrap_or(PERSIST_NONE);
        let restored = parse_restore(&options);
        let policy = CursorPolicy::read();
        // Whether a restore token came back is the whole question when an
        // app re-prompts: `offered` false means the app (or the frontend)
        // never sent one and the second prompt is theirs, not ours;
        // `offered` true with `usable` false means we wrote something we
        // could not read back, which would be ours.
        tracing::info!(
            app = %app_id,
            cursor_mode,
            persist_mode,
            offered = options.contains_key("restore_data"),
            usable = restored.is_some(),
            "select sources"
        );

        let sessions = Arc::clone(&self.sessions);
        let key = session_handle.as_str().to_string();
        let want_cursor = cursor_mode & CURSOR_EMBEDDED != 0;

        with_request(conn, &handle, |cancel| async move {
            // A restored choice skips the picker entirely — that's the point
            // of persistence: the second call doesn't interrupt the user.
            // It's only honoured if the monitor is still there.
            if let Some((output, _)) = &restored
                && !output_exists(output).await
            {
                tracing::info!(%output, "a restored source names a monitor that is gone; asking again");
            }
            if let Some((output, cursor)) = restored
                && output_exists(&output).await
            {
                tracing::info!(%output, "restored a previous screencast source");
                if let Ok(mut sessions) = sessions.lock()
                    && let Some(state) = sessions.get_mut(&key)
                {
                    state.output = Some(output);
                    // The policy still applies to a restored choice: the
                    // picker isn't shown here, so it's the only say the user
                    // gets on a session that repeats forever.
                    state.cursor = policy.apply(cursor);
                    state.persist_mode = persist_mode;
                }
                return (SUCCESS, HashMap::new());
            }

            // The cursor toggle is always offered, even when the app asked
            // for a hidden cursor. An app's `cursor_mode` is a preference
            // formed without asking anyone — Chromium, for instance, picks
            // HIDDEN whenever it thinks the backend can't do better — while
            // the person at the keyboard is the one who actually knows
            // whether their pointer should be in the stream. Their pointer,
            // on a screen they have already agreed to share, is not a
            // disclosure the app needs protecting from.
            let picker = OutputPicker::new(true, policy.apply(want_cursor));
            let picker = match ui::overlay(picker, cancel).await {
                Ok(picker) => picker,
                Err(err) => {
                    tracing::error!(%err, "output picker failed");
                    return empty(FAILED);
                }
            };
            let Some(output) = picker.chosen else {
                return empty(CANCELLED);
            };
            if picker.with_cursor && !want_cursor {
                tracing::info!(
                    "including the cursor at the user's request (the app asked for it hidden)"
                );
            }
            if let Ok(mut sessions) = sessions.lock()
                && let Some(state) = sessions.get_mut(&key)
            {
                state.output = Some(output);
                state.cursor = policy.apply(picker.with_cursor);
                state.persist_mode = persist_mode;
            }
            (SUCCESS, HashMap::new())
        })
        .await
    }

    async fn start(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        session_handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        _options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        let key = session_handle.as_str().to_string();
        let Some((output, cursor, persist_mode)) = self.sessions.lock().ok().and_then(|sessions| {
            sessions
                .get(&key)
                .and_then(|s| s.output.clone().map(|o| (o, s.cursor, s.persist_mode)))
        }) else {
            tracing::warn!(app = %app_id, "Start with no source selected");
            return empty(CANCELLED);
        };
        tracing::info!(app = %app_id, %output, cursor, "starting screencast");

        let stop = Arc::new(Cancel::default());
        let sessions = Arc::clone(&self.sessions);
        if let Ok(mut locked) = sessions.lock()
            && let Some(state) = locked.get_mut(&key)
        {
            state.stop = Some(Arc::clone(&stop));
        }

        let request = pw::Request {
            output: output.clone(),
            cursor,
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let thread_stop = Arc::clone(&stop);
        std::thread::Builder::new()
            .name("screencast".into())
            .spawn(move || pw::run(request, tx, thread_stop))
            .ok();

        // Wait for the node id off the D-Bus runtime: the channel recv is
        // blocking, and the cast thread answers within a couple of round
        // trips or not at all.
        let started = tokio::task::spawn_blocking(move || {
            rx.recv_timeout(std::time::Duration::from_secs(10))
        })
        .await;

        let started = match started {
            Ok(Ok(Ok(started))) => started,
            Ok(Ok(Err(err))) => {
                tracing::error!(%err, "screencast failed to start");
                stop.cancel();
                return empty(FAILED);
            }
            Ok(Err(_)) | Err(_) => {
                tracing::error!("screencast thread did not report a node id");
                stop.cancel();
                return empty(FAILED);
            }
        };

        with_request(conn, &handle, |_cancel| async move {
            // `streams` is `a(ua{sv})`: node id plus properties. The frontend
            // passes it through to the app untouched.
            let mut properties: HashMap<String, OwnedValue> = HashMap::new();
            properties.insert("size".to_string(), ov((started.width, started.height)));
            properties.insert("source_type".to_string(), ov(SOURCE_MONITOR));
            properties.insert("position".to_string(), ov((0i32, 0i32)));
            properties.insert("id".to_string(), ov(output.as_str()));
            let streams = vec![(started.node_id, properties)];

            let mut results = HashMap::new();
            results.insert("streams".to_string(), ov(streams));
            results.insert("persist_mode".to_string(), ov(persist_mode));
            if persist_mode != PERSIST_NONE {
                results.insert("restore_data".to_string(), build_restore(&output, cursor));
            }
            (SUCCESS, results)
        })
        .await
    }

    /// Only whole monitors. Window capture needs a per-toplevel capture source
    /// (`ext-image-copy-capture` over `ext-foreign-toplevel-list`), which the
    /// compositor doesn't advertise yet; claiming it here would put a "share a
    /// window" option in every browser that then failed.
    #[zbus(property, name = "AvailableSourceTypes")]
    fn available_source_types(&self) -> u32 {
        SOURCE_MONITOR
    }

    /// Hidden or composited into the frame. `METADATA` (cursor delivered
    /// out-of-band so the consumer can draw it) has no screencopy equivalent.
    #[zbus(property, name = "AvailableCursorModes")]
    fn available_cursor_modes(&self) -> u32 {
        CURSOR_HIDDEN | CURSOR_EMBEDDED
    }

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        4
    }
}

/// Is this connector still connected?
async fn output_exists(name: &str) -> bool {
    let name = name.to_string();
    tokio::task::spawn_blocking(move || {
        crate::capture::Capturer::new().is_ok_and(|capturer| capturer.index_of(&name).is_some())
    })
    .await
    .unwrap_or(false)
}
