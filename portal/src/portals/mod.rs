//! The `org.freedesktop.impl.portal.*` interfaces, and the plumbing they
//! share.
//!
//! Every portal interaction has the same shape, defined by the portal spec:
//!
//! * The frontend (`xdg-desktop-portal`) calls us with a **handle** — an
//!   object path we are expected to export an
//!   `org.freedesktop.impl.portal.Request` object at for the duration of the
//!   call. If the user closes the app mid-dialog, the frontend calls `Close()`
//!   on that object and we must abandon the interaction. [`Cancel`] is the
//!   token that propagates that into a UI loop or a capture thread.
//! * Long-lived interactions (screencast, global shortcuts, inhibit monitors)
//!   additionally get a **session handle**, an
//!   `org.freedesktop.impl.portal.Session` object with the same lifecycle
//!   rules; see [`SessionSink`].
//! * Every user-facing method answers `(response: u32, results: a{sv})`, where
//!   the response code is [`SUCCESS`], [`CANCELLED`] or [`FAILED`]. Returning
//!   a D-Bus *error* instead is a protocol violation the frontend logs and
//!   turns into a generic failure, so the impls below only ever error out for
//!   genuinely broken calls.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use zbus::object_server::SignalEmitter;
use zbus::zvariant::{OwnedObjectPath, OwnedValue, Value};
use zbus::{Connection, interface};

pub mod access;
pub mod account;
pub mod appchooser;
pub mod background;
pub mod dynamic_launcher;
pub mod email;
pub mod filechooser;
pub mod inhibit;
pub mod lockdown;
pub mod notification;
pub mod print;
pub mod screencast;
pub mod screenshot;
pub mod settings;
pub mod shortcuts;
pub mod wallpaper;

/// The object path every backend exports its interfaces at. Fixed by the
/// spec — the frontend looks here and nowhere else.
pub const PORTAL_PATH: &str = "/org/freedesktop/portal/desktop";

/// Well-known name we claim on the session bus. Must match the `DBusName` in
/// the installed `libreland.portal` file.
pub const BUS_NAME: &str = "org.freedesktop.impl.portal.desktop.libreland";

// ── Response codes (portal spec) ───────────────────────────────────────────

/// The interaction completed and `results` is meaningful.
pub const SUCCESS: u32 = 0;
/// The user dismissed the dialog (or the frontend closed the request).
pub const CANCELLED: u32 = 1;
/// Something went wrong that isn't the user's doing.
pub const FAILED: u32 = 2;

/// What every user-facing portal method returns: a response code plus the
/// per-interface result dictionary.
pub type PortalResult = (u32, HashMap<String, OwnedValue>);

/// A [`PortalResult`] carrying no results — the shape of both "the user said
/// no" and "we couldn't do it".
pub fn empty(response: u32) -> PortalResult {
    (response, HashMap::new())
}

// ── Variant helpers ────────────────────────────────────────────────────────
//
// Portal options arrive as `a{sv}` with everything optional and nothing
// guaranteed to be the type the spec says. These accessors are all
// "give me this key if it's there and it's the right type", never a panic.

/// Wrap a value for a results dictionary.
///
/// `OwnedValue::try_from` only fails for values containing file descriptors,
/// which nothing we put in a results map ever does; the fallback keeps a
/// hypothetical failure from taking the service down mid-reply.
pub fn ov<'a>(value: impl Into<Value<'a>>) -> OwnedValue {
    OwnedValue::try_from(value.into()).unwrap_or_else(|_| OwnedValue::from(0u32))
}

/// Read a string-valued option.
pub fn opt_str(options: &HashMap<String, OwnedValue>, key: &str) -> Option<String> {
    match options.get(key).map(std::ops::Deref::deref) {
        Some(Value::Str(s)) => Some(s.to_string()),
        _ => None,
    }
}

/// Read a boolean option.
pub fn opt_bool(options: &HashMap<String, OwnedValue>, key: &str) -> Option<bool> {
    match options.get(key).map(std::ops::Deref::deref) {
        Some(Value::Bool(b)) => Some(*b),
        _ => None,
    }
}

/// Read an unsigned option, accepting any of the integer types a caller might
/// plausibly have packed it as.
pub fn opt_u32(options: &HashMap<String, OwnedValue>, key: &str) -> Option<u32> {
    match options.get(key).map(std::ops::Deref::deref) {
        Some(Value::U32(v)) => Some(*v),
        Some(Value::I32(v)) => u32::try_from(*v).ok(),
        Some(Value::U64(v)) => u32::try_from(*v).ok(),
        Some(Value::I64(v)) => u32::try_from(*v).ok(),
        Some(Value::U16(v)) => Some(u32::from(*v)),
        Some(Value::U8(v)) => Some(u32::from(*v)),
        _ => None,
    }
}

/// Read an option holding an array of strings (`as`).
pub fn opt_str_array(options: &HashMap<String, OwnedValue>, key: &str) -> Vec<String> {
    let Some(Value::Array(array)) = options.get(key).map(std::ops::Deref::deref) else {
        return Vec::new();
    };
    array
        .iter()
        .filter_map(|v| match v {
            Value::Str(s) => Some(s.to_string()),
            _ => None,
        })
        .collect()
}

/// Read an option holding a byte array (`ay`) — how the spec passes paths,
/// which are bytes rather than UTF-8 strings (and are NUL-terminated).
pub fn opt_bytes(options: &HashMap<String, OwnedValue>, key: &str) -> Option<Vec<u8>> {
    let Some(Value::Array(array)) = options.get(key).map(std::ops::Deref::deref) else {
        return None;
    };
    let mut out: Vec<u8> = array
        .iter()
        .filter_map(|v| match v {
            Value::U8(b) => Some(*b),
            _ => None,
        })
        .collect();
    // Paths on the wire are NUL-terminated; strip it so callers get a plain path.
    if out.last() == Some(&0) {
        out.pop();
    }
    Some(out)
}

// ── URIs ───────────────────────────────────────────────────────────────────

/// Percent-encode a path into a `file://` URI, which is how every portal
/// hands file locations back.
///
/// The unreserved set is RFC 3986's, plus `/` (a path separator, not data).
/// Non-UTF-8 path bytes encode fine because we work on the raw bytes.
pub fn path_to_uri(path: &std::path::Path) -> String {
    use std::os::unix::ffi::OsStrExt as _;
    let mut uri = String::from("file://");
    for &byte in path.as_os_str().as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                uri.push(byte as char);
            }
            _ => {
                // Two hex digits, written without a temporary allocation.
                const HEX: &[u8; 16] = b"0123456789ABCDEF";
                uri.push('%');
                uri.push(HEX[(byte >> 4) as usize] as char);
                uri.push(HEX[(byte & 0xF) as usize] as char);
            }
        }
    }
    uri
}

/// Inverse of [`path_to_uri`], tolerant of a bare path (some callers pass one
/// where the spec says URI).
pub fn uri_to_path(uri: &str) -> std::path::PathBuf {
    use std::os::unix::ffi::OsStringExt as _;
    let encoded = uri.strip_prefix("file://").unwrap_or(uri);
    let bytes = encoded.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let Some(hex) = encoded.get(i + 1..i + 3)
            && let Ok(byte) = u8::from_str_radix(hex, 16)
        {
            out.push(byte);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    std::path::PathBuf::from(std::ffi::OsString::from_vec(out))
}

// ── Cancellation ───────────────────────────────────────────────────────────

/// The cancellation token behind one in-flight request.
///
/// Set by `Request.Close()` from the frontend. Two ways to observe it, because
/// the two sides of this service wait differently: blocking UI loops poll
/// [`Cancel::is_cancelled`] between Wayland dispatches, async work awaits
/// [`Cancel::cancelled`].
#[derive(Debug, Default)]
pub struct Cancel {
    flag: AtomicBool,
    notify: tokio::sync::Notify,
}

impl Cancel {
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// Resolves once cancelled (immediately if it already happened).
    pub async fn cancelled(&self) {
        // Register interest *before* re-checking the flag, so a cancel racing
        // in between can't be missed.
        let waiter = self.notify.notified();
        if self.is_cancelled() {
            return;
        }
        waiter.await;
    }
}

/// The `org.freedesktop.impl.portal.Request` object we export at a call's
/// handle path. Its only job is to flip the token.
struct RequestObj {
    cancel: Arc<Cancel>,
}

#[interface(name = "org.freedesktop.impl.portal.Request")]
impl RequestObj {
    fn close(&self) {
        self.cancel.cancel();
    }
}

/// Run one portal interaction with a live `Request` object exported at
/// `handle`.
///
/// The object is unexported before we return, so a `Close()` arriving after
/// the fact lands on nothing rather than on the next request that happens to
/// reuse the path. A closed request still answers [`CANCELLED`] rather than a
/// D-Bus error: the frontend expects a reply either way.
pub async fn with_request<F, Fut>(
    conn: &Connection,
    handle: &OwnedObjectPath,
    work: F,
) -> PortalResult
where
    F: FnOnce(Arc<Cancel>) -> Fut,
    Fut: Future<Output = PortalResult>,
{
    let cancel = Arc::new(Cancel::default());
    let exported = conn
        .object_server()
        .at(
            handle,
            RequestObj {
                cancel: Arc::clone(&cancel),
            },
        )
        .await
        .unwrap_or(false);

    let result = work(Arc::clone(&cancel)).await;

    if exported {
        let _ = conn.object_server().remove::<RequestObj, _>(handle).await;
    }
    // A cancelled interaction that still produced a result (a race between the
    // user clicking OK and the app going away) is reported as cancelled: the
    // frontend has already told the app the request is gone.
    if cancel.is_cancelled() && result.0 == SUCCESS {
        return empty(CANCELLED);
    }
    result
}

// ── Sessions ───────────────────────────────────────────────────────────────

/// Implemented by portals that own long-lived sessions, so the exported
/// `Session` object can tell them when the peer closed one.
pub trait SessionSink: Send + Sync + 'static {
    fn session_closed(&self, handle: &str);
}

/// The `org.freedesktop.impl.portal.Session` object exported at a session
/// handle. Owns nothing itself — it forwards teardown to the portal that
/// created it.
pub struct SessionObj {
    handle: OwnedObjectPath,
    sink: Arc<dyn SessionSink>,
}

#[interface(name = "org.freedesktop.impl.portal.Session")]
impl SessionObj {
    /// The peer is done with this session.
    async fn close(
        &self,
        #[zbus(object_server)] server: &zbus::ObjectServer,
        #[zbus(signal_emitter)] emitter: SignalEmitter<'_>,
    ) {
        self.sink.session_closed(self.handle.as_str());
        let _ = Self::closed(&emitter).await;
        // Unexporting from inside a method of the object being removed is
        // supported (the server drops it once this call returns), but the
        // borrow of `self` has to end first — hence no use of `self` after.
        let _ = server.remove::<SessionObj, _>(&self.handle).await;
    }

    #[zbus(signal)]
    async fn closed(emitter: &SignalEmitter<'_>) -> zbus::Result<()>;

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        2
    }
}

/// Export a `Session` object at `handle`. Returns false if the path was
/// already taken (a frontend bug, or a replayed handle).
pub async fn export_session(
    conn: &Connection,
    handle: &OwnedObjectPath,
    sink: Arc<dyn SessionSink>,
) -> bool {
    conn.object_server()
        .at(
            handle,
            SessionObj {
                handle: handle.clone(),
                sink,
            },
        )
        .await
        .unwrap_or(false)
}
