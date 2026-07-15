//! Libreland's polkit (`PolicyKit`) authentication agent.
//!
//! polkit only shows a password prompt when a *registered* authentication
//! agent exists for the session; nothing on a bare Arch install provides one,
//! so GUI programs that call into polkit (and `pkexec` launched from a GUI)
//! silently do nothing. This binary is that agent.
//!
//! It is a sibling of the compositor rather than code inside it: a polkit
//! agent is a normal per-session process, and keeping the async D-Bus stack
//! out of the compositor's single-threaded calloop is the right layering. The
//! compositor just autostarts it (see `misc.polkit_agent`).
//!
//! Flow, end to end:
//!
//! 1. Register on the **system bus** with
//!    `org.freedesktop.PolicyKit1.Authority.RegisterAuthenticationAgent`,
//!    passing a `unix-session` subject for our logind session and the object
//!    path we export.
//! 2. polkitd calls back `BeginAuthentication(action, message, icon, details,
//!    cookie, identities)` whenever the session needs to authenticate.
//! 3. We pick an identity (prefer the logged-in user — self-auth), resolve its
//!    uid to a username, and spawn the setuid/socket-activated helper
//!    `/usr/lib/polkit-1/polkit-agent-helper-1 <username>`. The helper — not
//!    us — runs PAM as root and reports the result to polkitd using the
//!    cookie, which is why the password never needs root here.
//! 4. We speak the helper's line protocol: write `<cookie>\n`, then for each
//!    `PAM_PROMPT_ECHO_OFF `/`PAM_PROMPT_ECHO_ON ` line we ask the user (via the
//!    quickshell dialog) and write `<response>\n`; `PAM_ERROR_MSG `/
//!    `PAM_TEXT_INFO ` become dialog messages; `SUCCESS`/`FAILURE` end it.
//! 5. The prompt UI is a themed quickshell card (`PolkitAgent.qml`). We reach
//!    it over a dedicated unix socket (line-delimited JSON) so the password
//!    travels a private local stream — never argv, never the compositor's IPC.
//!
//! No `unsafe`: the helper is a subprocess (not FFI), and uid→name resolution
//! goes through `nix`'s safe `getpwuid_r` wrapper.

use std::collections::HashMap;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader, Lines};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Notify, OwnedMutexGuard, mpsc};
use tracing_subscriber::EnvFilter;
use zbus::zvariant::{OwnedObjectPath, OwnedValue, Value};
use zbus::{Connection, Proxy, interface};

/// The PAM helper shipped by polkit. Fixed path (`PACKAGE_PREFIX/lib/...`
/// with the Arch prefix `/usr`).
const HELPER_PATH: &str = "/usr/lib/polkit-1/polkit-agent-helper-1";
/// polkit ≥126 ships the helper non-setuid and runs the privileged half as a
/// systemd socket-activated service. When the helper isn't setuid, the agent
/// must talk to this socket instead of spawning the binary (which would just
/// error "needs to be setuid root"). The root service reads the connecting
/// peer's uid via `SO_PEERCRED`.
const HELPER_SOCKET: &str = "/run/polkit/agent-helper.socket";
/// Object path we export the agent interface at (registered with polkitd).
const OBJECT_PATH: &str = "/org/libreland/PolkitAgent";

// ---------------------------------------------------------------------------
// Dialog protocol (agent <-> quickshell PolkitAgent.qml), line-delimited JSON.
// ---------------------------------------------------------------------------

/// Agent → dialog. One JSON object per line.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum AgentMsg<'a> {
    /// Open the dialog with the action context (no prompt yet).
    Show {
        id: u64,
        action_id: &'a str,
        message: &'a str,
        icon: &'a str,
        user: &'a str,
    },
    /// Ask a question. `echo` false = password field (hidden).
    Prompt {
        id: u64,
        prompt: &'a str,
        echo: bool,
    },
    /// `PAM_TEXT_INFO` — an informational line.
    Info { id: u64, text: &'a str },
    /// `PAM_ERROR_MSG` or a failed attempt — shown in red, dialog stays open.
    Error { id: u64, text: &'a str },
    /// Auth finished; the dialog should dismiss.
    Complete { id: u64, success: bool },
}

/// Dialog → agent.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ClientMsg {
    /// The user submitted an answer to the current prompt.
    Response { id: u64, value: String },
    /// The user dismissed the dialog (Escape / Cancel).
    Cancel { id: u64 },
}

/// Shared connection to the (single) quickshell dialog client.
///
/// The accept loop parks the latest client's write half here and forwards its
/// incoming lines to whichever session currently owns `inbox`. Only one auth
/// runs at a time (serialized by [`Shared::dialog_lock`]), so a single inbox
/// slot is enough.
struct DialogInner {
    writer: Option<OwnedWriteHalf>,
    inbox: Option<mpsc::UnboundedSender<ClientMsg>>,
}

type Dialog = Arc<tokio::sync::Mutex<DialogInner>>;

// ---------------------------------------------------------------------------
// One authentication session.
// ---------------------------------------------------------------------------

enum Prompt {
    Answer(String),
    Dismissed,
    Disconnected,
}

/// The outcome of one helper (PAM) run.
enum Outcome {
    Success,
    /// PAM said no *after prompting* (e.g. wrong password) — caller re-prompts.
    Failed,
    /// The helper failed before ever prompting (misconfig, can't escalate).
    /// Not retried — retrying would spin with no user interaction.
    HardFailure,
    /// User dismissed the dialog.
    Dismissed,
    /// No dialog / socket dropped mid-auth.
    Disconnected,
}

/// A live dialog session, held for the duration of one `BeginAuthentication`.
/// Holds the exclusivity guard so no other auth can drive the dialog at once.
struct Session {
    id: u64,
    dialog: Dialog,
    rx: mpsc::UnboundedReceiver<ClientMsg>,
    _guard: OwnedMutexGuard<()>,
}

impl Session {
    /// Write one message to the dialog client. Errors if no client is
    /// connected (quickshell not up) or the stream dropped.
    async fn send(&self, msg: &AgentMsg<'_>) -> anyhow::Result<()> {
        let mut line = serde_json::to_string(msg)?;
        line.push('\n');
        let mut d = self.dialog.lock().await;
        let w = d.writer.as_mut().context("no dialog client connected")?;
        w.write_all(line.as_bytes()).await?;
        w.flush().await?;
        Ok(())
    }

    /// Send a prompt and await the user's reply for *this* session id.
    async fn prompt(&mut self, text: &str, echo: bool) -> Prompt {
        if self
            .send(&AgentMsg::Prompt {
                id: self.id,
                prompt: text,
                echo,
            })
            .await
            .is_err()
        {
            return Prompt::Disconnected;
        }
        loop {
            match self.rx.recv().await {
                Some(ClientMsg::Response { id, value }) if id == self.id => {
                    return Prompt::Answer(value);
                }
                Some(ClientMsg::Cancel { id }) if id == self.id => return Prompt::Dismissed,
                // stale message from a previous dialog id — ignore.
                Some(_) => {}
                None => return Prompt::Disconnected,
            }
        }
    }
}

/// A duplex line channel to the polkit PAM helper. Two transports, one
/// protocol: after the initial handshake both speak `PAM_*` lines to us and
/// take `<response>\n` back.
enum Helper {
    /// Setuid helper: spawned as `polkit-agent-helper-1 <user>`, cookie on
    /// stdin. Boxed — a `Child` is much larger than the socket variant.
    Spawn(Box<SpawnHelper>),
    /// Socket-activated (polkit ≥126, non-setuid): we connect to the root
    /// service's socket and send `<user>\n<cookie>\n` before the conversation.
    Socket(SocketHelper),
}

struct SpawnHelper {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
}

struct SocketHelper {
    write: OwnedWriteHalf,
    lines: Lines<BufReader<OwnedReadHalf>>,
}

/// Pick the transport: the setuid spawn path if the helper is actually setuid,
/// otherwise the socket-activated service (the modern default).
fn choose_helper() -> &'static str {
    if let Ok(meta) = std::fs::metadata(HELPER_PATH)
        && meta.permissions().mode() & 0o4000 != 0
    {
        return "spawn";
    }
    if Path::new(HELPER_SOCKET).exists() {
        return "socket";
    }
    "spawn" // best effort; surfaces a clear PAM_ERROR_MSG if unusable
}

/// Open a helper channel and perform the mode-specific handshake.
async fn open_helper(username: &str, cookie: &str) -> anyhow::Result<Helper> {
    if choose_helper() == "socket" {
        let stream = UnixStream::connect(HELPER_SOCKET)
            .await
            .context("connect polkit agent-helper socket")?;
        let (read, mut write) = stream.into_split();
        // Socket mode: username line, then cookie line (read_cookie ×2).
        write.write_all(username.as_bytes()).await?;
        write.write_all(b"\n").await?;
        write.write_all(cookie.as_bytes()).await?;
        write.write_all(b"\n").await?;
        write.flush().await?;
        Ok(Helper::Socket(SocketHelper {
            write,
            lines: BufReader::new(read).lines(),
        }))
    } else {
        let mut child = Command::new(HELPER_PATH)
            .arg(username)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            // If we bail, the helper dies with us — no lingering PAM session.
            .kill_on_drop(true)
            .spawn()
            .context("spawn polkit-agent-helper-1")?;
        let mut stdin = child.stdin.take().context("helper stdin")?;
        let stdout = child.stdout.take().context("helper stdout")?;
        // Setuid mode: only the cookie on stdin (user came in on argv).
        stdin.write_all(cookie.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        stdin.flush().await?;
        Ok(Helper::Spawn(Box::new(SpawnHelper {
            child,
            stdin,
            lines: BufReader::new(stdout).lines(),
        })))
    }
}

impl Helper {
    async fn next_line(&mut self) -> std::io::Result<Option<String>> {
        match self {
            Helper::Spawn(s) => s.lines.next_line().await,
            Helper::Socket(s) => s.lines.next_line().await,
        }
    }

    async fn write_line(&mut self, value: &str) -> std::io::Result<()> {
        let w: &mut (dyn tokio::io::AsyncWrite + Unpin + Send) = match self {
            Helper::Spawn(s) => &mut s.stdin,
            Helper::Socket(s) => &mut s.write,
        };
        w.write_all(value.as_bytes()).await?;
        w.write_all(b"\n").await?;
        w.flush().await
    }

    async fn finish(&mut self) {
        if let Helper::Spawn(s) = self {
            let _ = s.child.wait().await;
        }
    }
}

/// Run one helper (PAM) conversation to a verdict.
async fn run_helper_session(
    session: &mut Session,
    username: &str,
    cookie: &str,
) -> anyhow::Result<Outcome> {
    let mut helper = open_helper(username, cookie).await?;
    // A FAILURE that never prompted is a hard failure (e.g. the helper can't
    // escalate) — retrying it would spin with no user interaction.
    let mut prompted = false;

    while let Some(line) = helper.next_line().await? {
        if let Some(prompt) = line.strip_prefix("PAM_PROMPT_ECHO_OFF ") {
            prompted = true;
            match session.prompt(&unescape(prompt), false).await {
                Prompt::Answer(v) => helper.write_line(&v).await?,
                Prompt::Dismissed => return Ok(Outcome::Dismissed),
                Prompt::Disconnected => return Ok(Outcome::Disconnected),
            }
        } else if let Some(prompt) = line.strip_prefix("PAM_PROMPT_ECHO_ON ") {
            prompted = true;
            match session.prompt(&unescape(prompt), true).await {
                Prompt::Answer(v) => helper.write_line(&v).await?,
                Prompt::Dismissed => return Ok(Outcome::Dismissed),
                Prompt::Disconnected => return Ok(Outcome::Disconnected),
            }
        } else if let Some(text) = line.strip_prefix("PAM_ERROR_MSG ") {
            let _ = session
                .send(&AgentMsg::Error {
                    id: session.id,
                    text: &unescape(text),
                })
                .await;
        } else if let Some(text) = line.strip_prefix("PAM_TEXT_INFO ") {
            let _ = session
                .send(&AgentMsg::Info {
                    id: session.id,
                    text: &unescape(text),
                })
                .await;
        } else if line.starts_with("SUCCESS") {
            helper.finish().await;
            return Ok(Outcome::Success);
        } else if line.starts_with("FAILURE") {
            helper.finish().await;
            return Ok(if prompted {
                Outcome::Failed
            } else {
                Outcome::HardFailure
            });
        }
        // Unknown lines: ignore (forward-compatible).
    }

    // Helper closed without a verdict.
    Ok(if prompted {
        Outcome::Failed
    } else {
        Outcome::HardFailure
    })
}

/// polkit's helper escapes the PAM message text so it stays single-line. The
/// scheme is backslash escaping of control bytes; real prompts ("Password: ")
/// contain none, but we decode defensively so a multi-line PAM message renders.
fn unescape(s: &str) -> String {
    if !s.contains('\\') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                // any other escaped char (incl. a literal backslash) passes through
                Some(other) => out.push(other),
                // trailing backslash with nothing after it: keep it literal
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Loop helper runs until success or the user gives up. PAM's own retry budget
/// governs a single run; on FAILURE we surface it and offer another attempt,
/// matching how polkit-gnome/mate behave.
async fn converse(
    session: &mut Session,
    username: &str,
    cookie: &str,
) -> anyhow::Result<Outcome> {
    loop {
        match run_helper_session(session, username, cookie).await? {
            Outcome::Success => return Ok(Outcome::Success),
            Outcome::Dismissed => return Ok(Outcome::Dismissed),
            Outcome::Disconnected => return Ok(Outcome::Disconnected),
            // Never prompted → don't retry (would spin). The helper's
            // PAM_ERROR_MSG is already on screen; end here.
            Outcome::HardFailure => return Ok(Outcome::HardFailure),
            Outcome::Failed => {
                // A prompted failure (wrong password). Re-running is paced by
                // the user re-typing, so this can't spin; offer another go.
                let _ = session
                    .send(&AgentMsg::Error {
                        id: session.id,
                        text: "Authentication failed. Please try again.",
                    })
                    .await;
            }
        }
    }
}

/// The context of one `BeginAuthentication` call, resolved and ready to drive.
struct AuthRequest<'a> {
    id: u64,
    action_id: &'a str,
    message: &'a str,
    icon: &'a str,
    username: &'a str,
    cookie: &'a str,
}

/// Everything one `BeginAuthentication` needs to drive a dialog to a verdict.
async fn run_dialog(
    shared: &Shared,
    req: &AuthRequest<'_>,
    cancel: &Notify,
) -> Result<(), PolkitError> {
    // Serialize: one dialog at a time.
    let guard = shared.dialog_lock.clone().lock_owned().await;
    let (tx, rx) = mpsc::unbounded_channel();
    shared.dialog.lock().await.inbox = Some(tx);
    let mut session = Session {
        id: req.id,
        dialog: shared.dialog.clone(),
        rx,
        _guard: guard,
    };

    // Open the dialog. If quickshell isn't listening, there's no way to ask.
    if session
        .send(&AgentMsg::Show {
            id: req.id,
            action_id: req.action_id,
            message: req.message,
            icon: req.icon,
            user: req.username,
        })
        .await
        .is_err()
    {
        shared.dialog.lock().await.inbox = None;
        return Err(PolkitError::Failed(
            "no authentication dialog is available".into(),
        ));
    }

    // Race the conversation against a polkitd Cancel for this cookie.
    let outcome = tokio::select! {
        biased;
        () = cancel.notified() => Outcome::Dismissed,
        res = converse(&mut session, req.username, req.cookie) => res.unwrap_or(Outcome::Disconnected),
    };

    let success = matches!(outcome, Outcome::Success);
    let _ = session
        .send(&AgentMsg::Complete {
            id: req.id,
            success,
        })
        .await;
    shared.dialog.lock().await.inbox = None;

    match outcome {
        Outcome::Success => Ok(()),
        Outcome::Dismissed => Err(PolkitError::Cancelled("authentication dismissed".into())),
        Outcome::Failed | Outcome::HardFailure | Outcome::Disconnected => {
            Err(PolkitError::Failed("authentication failed".into()))
        }
    }
}

// ---------------------------------------------------------------------------
// D-Bus: the AuthenticationAgent interface polkitd calls back into.
// ---------------------------------------------------------------------------

/// Errors mapped to polkit's D-Bus error namespace. `Cancelled` is the
/// specific name polkitd expects when the user dismisses the prompt.
#[derive(Debug, zbus::DBusError)]
#[zbus(prefix = "org.freedesktop.PolicyKit1.Error")]
enum PolkitError {
    #[zbus(error)]
    ZBus(zbus::Error),
    Failed(String),
    Cancelled(String),
}

struct Shared {
    dialog: Dialog,
    /// Held for the length of one dialog so auths don't interleave.
    dialog_lock: Arc<tokio::sync::Mutex<()>>,
    /// cookie → cancel handle, so `CancelAuthentication` can abort a live auth.
    active: tokio::sync::Mutex<HashMap<String, Arc<Notify>>>,
    counter: AtomicU64,
    /// Our own uid — preferred identity when polkit offers "authenticate as
    /// yourself or an admin".
    our_uid: u32,
}

struct AuthAgent {
    shared: Arc<Shared>,
}

#[interface(name = "org.freedesktop.PolicyKit1.AuthenticationAgent")]
impl AuthAgent {
    #[zbus(name = "BeginAuthentication")]
    async fn begin_authentication(
        &self,
        action_id: String,
        message: String,
        icon_name: String,
        details: HashMap<String, String>,
        cookie: String,
        identities: Vec<(String, HashMap<String, OwnedValue>)>,
    ) -> Result<(), PolkitError> {
        // `details` (a{ss}) is a required positional arg but we don't surface it.
        let _ = details;
        let shared = &self.shared;

        let Some(username) = pick_username(&identities, shared.our_uid) else {
            tracing::warn!(action = %action_id, "no unix-user identity offered; cannot authenticate");
            return Err(PolkitError::Failed(
                "no unix-user identity offered".into(),
            ));
        };

        let id = shared.counter.fetch_add(1, Ordering::Relaxed);
        tracing::info!(action = %action_id, user = %username, "authentication requested");

        let cancel = Arc::new(Notify::new());
        shared
            .active
            .lock()
            .await
            .insert(cookie.clone(), cancel.clone());

        let req = AuthRequest {
            id,
            action_id: &action_id,
            message: &message,
            icon: &icon_name,
            username: &username,
            cookie: &cookie,
        };
        let result = run_dialog(shared, &req, &cancel).await;

        shared.active.lock().await.remove(&cookie);

        match &result {
            Ok(()) => tracing::info!(user = %username, "authentication succeeded"),
            Err(e) => tracing::info!(?e, "authentication ended"),
        }
        result
    }

    #[zbus(name = "CancelAuthentication")]
    async fn cancel_authentication(&self, cookie: String) {
        if let Some(notify) = self.shared.active.lock().await.get(&cookie) {
            tracing::info!("authentication cancelled by polkitd");
            notify.notify_one();
        }
    }
}

/// Choose which identity to authenticate as. Prefer the logged-in user
/// (self-auth) when offered; otherwise the first `unix-user` identity.
fn pick_username(
    identities: &[(String, HashMap<String, OwnedValue>)],
    our_uid: u32,
) -> Option<String> {
    let uids: Vec<u32> = identities
        .iter()
        .filter(|(kind, _)| kind == "unix-user")
        .filter_map(|(_, details)| details.get("uid").and_then(value_to_u32))
        .collect();
    let chosen = if uids.contains(&our_uid) {
        our_uid
    } else {
        *uids.first()?
    };
    uid_to_username(chosen)
}

fn value_to_u32(v: &OwnedValue) -> Option<u32> {
    match &**v {
        Value::U32(u) => Some(*u),
        Value::U64(u) => u32::try_from(*u).ok(),
        Value::I32(i) => u32::try_from(*i).ok(),
        Value::I64(i) => u32::try_from(*i).ok(),
        Value::U16(u) => Some(u32::from(*u)),
        Value::U8(u) => Some(u32::from(*u)),
        _ => None,
    }
}

fn uid_to_username(uid: u32) -> Option<String> {
    nix::unistd::User::from_uid(nix::unistd::Uid::from_raw(uid))
        .ok()
        .flatten()
        .map(|u| u.name)
}

/// The `unix-session` subject we register for: `("unix-session",
/// {"session-id": <id>})`, matching `(sa{sv})`.
fn build_subject(session_id: &str) -> (String, HashMap<String, Value<'static>>) {
    let mut details = HashMap::new();
    details.insert("session-id".to_string(), Value::from(session_id.to_string()));
    ("unix-session".to_string(), details)
}

/// Determine our logind session id: `$XDG_SESSION_ID` if set, else ask logind
/// for the session owning this process.
async fn get_session_id(conn: &Connection) -> Option<String> {
    if let Ok(id) = std::env::var("XDG_SESSION_ID")
        && !id.is_empty()
    {
        return Some(id);
    }
    let manager = Proxy::new(
        conn,
        "org.freedesktop.login1",
        "/org/freedesktop/login1",
        "org.freedesktop.login1.Manager",
    )
    .await
    .ok()?;
    let pid: u32 = std::process::id();
    let path: OwnedObjectPath = manager.call("GetSessionByPID", &(pid,)).await.ok()?;
    let session = Proxy::new(
        conn,
        "org.freedesktop.login1",
        path.into_inner(),
        "org.freedesktop.login1.Session",
    )
    .await
    .ok()?;
    session.get_property::<String>("Id").await.ok()
}

fn socket_path() -> PathBuf {
    let dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(dir).join("libreland-polkit.sock")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let conn = Connection::system()
        .await
        .context("connect to the system D-Bus")?;

    let session_id = get_session_id(&conn).await.context(
        "could not determine the logind session id (XDG_SESSION_ID unset and login1 query failed)",
    )?;
    let our_uid = nix::unistd::getuid().as_raw();

    // Private socket to the quickshell dialog.
    let sock_path = socket_path();
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path)
        .with_context(|| format!("bind dialog socket {}", sock_path.display()))?;
    tracing::info!(socket = %sock_path.display(), "polkit dialog socket listening");

    let dialog: Dialog = Arc::new(tokio::sync::Mutex::new(DialogInner {
        writer: None,
        inbox: None,
    }));

    // Accept quickshell connections. The latest connection is the live dialog;
    // its incoming lines are forwarded to the active session's inbox.
    {
        let dialog = dialog.clone();
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        let (read, write) = stream.into_split();
                        dialog.lock().await.writer = Some(write);
                        tracing::info!("dialog client connected");
                        let dialog = dialog.clone();
                        tokio::spawn(async move {
                            let mut lines = BufReader::new(read).lines();
                            while let Ok(Some(line)) = lines.next_line().await {
                                if let Ok(msg) = serde_json::from_str::<ClientMsg>(&line)
                                    && let Some(tx) = &dialog.lock().await.inbox
                                {
                                    let _ = tx.send(msg);
                                }
                            }
                        });
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "dialog socket accept failed");
                        break;
                    }
                }
            }
        });
    }

    let shared = Arc::new(Shared {
        dialog,
        dialog_lock: Arc::new(tokio::sync::Mutex::new(())),
        active: tokio::sync::Mutex::new(HashMap::new()),
        counter: AtomicU64::new(1),
        our_uid,
    });

    conn.object_server()
        .at(OBJECT_PATH, AuthAgent { shared })
        .await
        .context("export the agent object")?;

    let authority = Proxy::new(
        &conn,
        "org.freedesktop.PolicyKit1",
        "/org/freedesktop/PolicyKit1/Authority",
        "org.freedesktop.PolicyKit1.Authority",
    )
    .await?;
    let locale = std::env::var("LANG").unwrap_or_else(|_| "en_US.UTF-8".to_string());
    authority
        .call_method(
            "RegisterAuthenticationAgent",
            &(build_subject(&session_id), locale.as_str(), OBJECT_PATH),
        )
        .await
        .context("RegisterAuthenticationAgent")?;
    tracing::info!(session = %session_id, uid = our_uid, "registered as polkit authentication agent");

    // Run until the compositor stops us; then unregister and clean up.
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    tokio::select! {
        _ = sigterm.recv() => {}
        _ = sigint.recv() => {}
    }
    tracing::info!("shutting down; unregistering agent");
    let _ = authority
        .call_method(
            "UnregisterAuthenticationAgent",
            &(build_subject(&session_id), OBJECT_PATH),
        )
        .await;
    let _ = std::fs::remove_file(&sock_path);
    Ok(())
}
