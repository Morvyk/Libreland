//! `org.freedesktop.impl.portal.Email` — compose an email on the user's
//! behalf.
//!
//! GTK's backend renders its own compose window; we hand the message to
//! whatever already handles `mailto:` instead, because the user's mail client
//! is where their signature, identities and drafts live. What stays here is
//! the part that has to be a portal: showing the user what an app is about to
//! send, and letting them refuse.
//!
//! Attachments arrive as file descriptors (a sandboxed app can't hand over
//! paths). We materialize them into a private spool directory and attach those
//! paths, since no `mailto:` client can read an fd.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::os::fd::AsRawFd as _;
use std::path::PathBuf;

use zbus::zvariant::{Fd, OwnedObjectPath, OwnedValue, Value};
use zbus::{Connection, interface};

use crate::apps;
use crate::ui;
use crate::ui::prompt::{Prompt, Spec};

use super::{
    CANCELLED, FAILED, PortalResult, SUCCESS, empty, opt_str, opt_str_array, path_to_uri,
    with_request,
};

pub struct Email;

impl Email {
    pub const fn new() -> Self {
        Self
    }
}

/// Percent-encode one `mailto:` header value (RFC 6068: everything outside
/// the unreserved set gets escaped, and `&`/`=`/`?` especially).
fn encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                const HEX: &[u8; 16] = b"0123456789ABCDEF";
                out.push('%');
                out.push(HEX[(byte >> 4) as usize] as char);
                out.push(HEX[(byte & 0xF) as usize] as char);
            }
        }
    }
    out
}

/// Copy the attachment fds into a spool directory, returning their paths.
///
/// The fds are already open and already checked by the frontend; reading them
/// here is the only way to give a `mailto:` handler something it can attach.
fn spool_attachments(fds: &[Fd<'_>]) -> Vec<PathBuf> {
    if fds.is_empty() {
        return Vec::new();
    }
    let base = std::env::var("XDG_RUNTIME_DIR")
        .map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from)
        .join("libreland-portal/email");
    if std::fs::create_dir_all(&base).is_err() {
        return Vec::new();
    }
    let mut paths = Vec::new();
    for (index, fd) in fds.iter().enumerate() {
        // Duplicate rather than take ownership: the fd belongs to the D-Bus
        // message and is closed when it drops.
        let Ok(duplicated) = nix::unistd::dup(fd.as_raw_fd()) else {
            continue;
        };
        // SAFETY: `dup` just returned this descriptor to us and nothing else
        // holds it, so wrapping it in a File that owns it is sound.
        #[allow(
            unsafe_code,
            reason = "from_raw_fd is the only way to adopt a descriptor we just dup'd; ownership is unambiguous here"
        )]
        // SAFETY: see the #[allow] above.
        let mut file = unsafe {
            <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(duplicated)
        };
        let path = base.join(format!("attachment-{index}"));
        let Ok(mut out) = std::fs::File::create(&path) else {
            continue;
        };
        if std::io::copy(&mut file, &mut out).is_ok() && out.flush().is_ok() {
            paths.push(path);
        }
    }
    paths
}

#[interface(name = "org.freedesktop.impl.portal.Email")]
impl Email {
    async fn compose_email(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        let mut to = opt_str_array(&options, "addresses");
        if let Some(single) = opt_str(&options, "address") {
            to.insert(0, single);
        }
        let cc = opt_str_array(&options, "cc");
        let bcc = opt_str_array(&options, "bcc");
        let subject = opt_str(&options, "subject").unwrap_or_default();
        let body = opt_str(&options, "body").unwrap_or_default();
        let fds: Vec<Fd<'_>> = match options.get("attachment_fds").map(std::ops::Deref::deref) {
            Some(Value::Array(array)) => array
                .iter()
                .filter_map(|v| match v {
                    Value::Fd(fd) => Some(fd.try_clone().ok()?),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        tracing::info!(app = %app_id, recipients = to.len(), attachments = fds.len(), "compose email");

        let attachments = spool_attachments(&fds);

        // Build the mailto: URI now so the confirmation shows exactly what
        // will be handed over.
        let mut uri = format!("mailto:{}", to.join(","));
        let mut params: Vec<String> = Vec::new();
        if !cc.is_empty() {
            params.push(format!("cc={}", encode(&cc.join(","))));
        }
        if !bcc.is_empty() {
            params.push(format!("bcc={}", encode(&bcc.join(","))));
        }
        if !subject.is_empty() {
            params.push(format!("subject={}", encode(&subject)));
        }
        if !body.is_empty() {
            params.push(format!("body={}", encode(&body)));
        }
        for path in &attachments {
            params.push(format!("attach={}", encode(&path_to_uri(path))));
        }
        if !params.is_empty() {
            uri.push('?');
            uri.push_str(&params.join("&"));
        }

        let app = if app_id.is_empty() {
            "An application".to_string()
        } else {
            app_id.clone()
        };
        let mut summary = format!("{app} wants to open your mail client.\n");
        if !to.is_empty() {
            let _ = write!(summary, "\nTo: {}", to.join(", "));
        }
        if !subject.is_empty() {
            let _ = write!(summary, "\nSubject: {subject}");
        }
        if !attachments.is_empty() {
            let _ = write!(summary, "\nAttachments: {}", attachments.len());
        }
        let spec = Spec {
            title: "Compose email?".to_string(),
            subtitle: String::new(),
            body: summary,
            accept_label: Some("Compose".to_string()),
            deny_label: Some("Cancel".to_string()),
            destructive: false,
            toggles: Vec::new(),
        };

        with_request(conn, &handle, |cancel| async move {
            match ui::dialog(Prompt::new(spec), cancel).await {
                Ok(prompt) if prompt.accepted => {
                    let Some(handler) = apps::default_handler("x-scheme-handler/mailto") else {
                        tracing::warn!("no mailto: handler is installed");
                        return empty(FAILED);
                    };
                    match apps::launch(&handler, &[uri]) {
                        Ok(()) => (SUCCESS, HashMap::new()),
                        Err(err) => {
                            tracing::error!(%err, app = %handler.id, "failed to launch the mail client");
                            empty(FAILED)
                        }
                    }
                }
                Ok(_) => empty(CANCELLED),
                Err(err) => {
                    tracing::error!(%err, "email dialog failed");
                    empty(FAILED)
                }
            }
        })
        .await
    }

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        3
    }
}
