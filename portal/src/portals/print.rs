//! `org.freedesktop.impl.portal.Print` — print a document.
//!
//! The interaction is two-step by design: `PreparePrint` asks the user where
//! the job should go and hands back a token plus the settings the app should
//! render with, then `Print` arrives with a file descriptor holding the
//! rendered PDF and that token.
//!
//! What this does *not* do is reimplement GTK's print dialog. There is no page
//! setup, no per-printer option tree, no preview — those are a CUPS UI, not a
//! portal. What it does is the part that has to work: let the user choose a
//! destination (any queue CUPS knows about, or a file), and then actually send
//! the job with `lp`. Apps that need fine-grained print control ship their own
//! dialog anyway; apps that just want "print this" get exactly that.

use std::collections::HashMap;
use std::io::Write as _;
use std::os::fd::AsRawFd as _;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU32, Ordering},
};

use zbus::zvariant::{Fd, OwnedObjectPath, OwnedValue};
use zbus::{Connection, interface};

use crate::ui;
use crate::ui::filechooser::{FileChooser, Kind, Spec as ChooserSpec};
use crate::ui::prompt::{Prompt, Spec as PromptSpec, Toggle};

use super::{CANCELLED, FAILED, PortalResult, SUCCESS, empty, opt_str, ov, with_request};

/// A destination the user picked in `PreparePrint`, waiting for its `Print`.
#[derive(Clone, Debug)]
enum Destination {
    /// A CUPS queue name.
    Printer(String),
    /// Write the PDF out instead of printing it.
    File,
}

pub struct Print {
    /// token → destination, filled by `PreparePrint` and consumed by `Print`.
    pending: Arc<Mutex<HashMap<u32, Destination>>>,
    next_token: AtomicU32,
}

impl Print {
    pub fn new() -> Self {
        Self {
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_token: AtomicU32::new(1),
        }
    }
}

/// Queues CUPS knows about, via `lpstat -e` (which lists every destination
/// without contacting each one — `lpstat -a` blocks on unreachable printers).
fn printers() -> Vec<String> {
    let Ok(output) = Command::new("lpstat").arg("-e").output() else {
        return Vec::new();
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect()
}

/// The default queue, from `lpstat -d` ("system default destination: name").
fn default_printer() -> Option<String> {
    let output = Command::new("lpstat").arg("-d").output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    text.split(':').nth(1).map(|name| name.trim().to_string())
}

/// Copy the fd's contents into a spool file we can hand to `lp`.
fn spool(fd: &Fd<'_>, title: &str) -> anyhow::Result<PathBuf> {
    let dir = std::env::var("XDG_RUNTIME_DIR")
        .map_or_else(|_| PathBuf::from("/tmp"), PathBuf::from)
        .join("libreland-portal/print");
    std::fs::create_dir_all(&dir)?;
    let safe: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .take(40)
        .collect();
    let path = dir.join(format!(
        "{}-{}.pdf",
        if safe.is_empty() { "job" } else { &safe },
        std::process::id()
    ));

    let duplicated = nix::unistd::dup(fd.as_raw_fd())?;
    // SAFETY: `dup` just handed us this descriptor and nothing else owns it,
    // so adopting it into a File is sound.
    #[allow(
        unsafe_code,
        reason = "adopting a descriptor we just dup'd is the only way to read a D-Bus fd argument"
    )]
    // SAFETY: see the #[allow] above.
    let mut source = unsafe { <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(duplicated) };
    let mut out = std::fs::File::create(&path)?;
    std::io::copy(&mut source, &mut out)?;
    out.flush()?;
    Ok(path)
}

#[interface(name = "org.freedesktop.impl.portal.Print")]
impl Print {
    /// Ask where the job should go, and hand back a token for it.
    #[allow(
        clippy::too_many_arguments,
        reason = "the argument list is fixed by the portal interface definition"
    )]
    async fn prepare_print(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        title: String,
        settings: HashMap<String, OwnedValue>,
        page_setup: HashMap<String, OwnedValue>,
        _options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        tracing::info!(app = %app_id, %title, "prepare print");
        let queues = printers();
        let default = default_printer().filter(|d| queues.contains(d));

        // One combo listing every queue plus "Save as PDF"; whatever the app
        // suggested in `settings` pre-selects.
        let mut options: Vec<(String, String)> = queues
            .iter()
            .map(|name| (name.clone(), name.clone()))
            .collect();
        options.push(("__file__".to_string(), "Save as PDF…".to_string()));
        let selected = opt_str(&settings, "printer")
            .filter(|p| queues.contains(p))
            .or(default)
            .unwrap_or_else(|| {
                options
                    .first()
                    .map_or_else(|| "__file__".to_string(), |(id, _)| id.clone())
            });

        let body = if queues.is_empty() {
            format!("No printers are configured.\n\n“{title}” can still be saved as a PDF.")
        } else {
            format!("Send “{title}” to:")
        };
        let spec = PromptSpec {
            title: "Print".to_string(),
            subtitle: String::new(),
            body,
            accept_label: Some("Print".to_string()),
            deny_label: Some("Cancel".to_string()),
            destructive: false,
            toggles: vec![Toggle {
                id: "printer".to_string(),
                label: "Destination".to_string(),
                options,
                selected,
            }],
        };
        // Page setup is echoed back untouched: we have no UI for it, and the
        // app's own defaults are better than ones we'd invent.
        let echoed_setup = page_setup;
        let echoed_settings = settings;
        let pending = Arc::clone(&self.pending);
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);

        with_request(conn, &handle, |cancel| async move {
            match ui::dialog(Prompt::new(spec), cancel).await {
                Ok(prompt) if prompt.accepted => {
                    let choice = prompt
                        .choices()
                        .into_iter()
                        .find(|(id, _)| id == "printer")
                        .map_or_else(|| "__file__".to_string(), |(_, value)| value);
                    let destination = if choice == "__file__" {
                        Destination::File
                    } else {
                        Destination::Printer(choice.clone())
                    };
                    if let Ok(mut pending) = pending.lock() {
                        pending.insert(token, destination);
                    }
                    let mut settings = echoed_settings;
                    settings.insert("printer".to_string(), ov(choice.as_str()));
                    // Tell the app to render for a real printer, not a
                    // preview: `output-basename`/`output-uri` would make it
                    // write the file itself, which is our job here.
                    settings.remove("output-uri");
                    settings.remove("output-basename");
                    let mut results = HashMap::new();
                    results.insert("settings".to_string(), ov(settings));
                    results.insert("page-setup".to_string(), ov(echoed_setup));
                    results.insert("token".to_string(), ov(token));
                    (SUCCESS, results)
                }
                Ok(_) => empty(CANCELLED),
                Err(err) => {
                    tracing::error!(%err, "print dialog failed");
                    empty(FAILED)
                }
            }
        })
        .await
    }

    /// Send the rendered document to the destination chosen earlier.
    #[allow(
        clippy::too_many_arguments,
        reason = "the argument list is fixed by the portal interface definition"
    )]
    async fn print(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        title: String,
        fd: Fd<'_>,
        options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        let token = super::opt_u32(&options, "token");
        tracing::info!(app = %app_id, %title, ?token, "print");

        // Without a token (or with a stale one) the app never went through
        // PreparePrint; ask now rather than failing, since that's a legal flow.
        let destination = token
            .and_then(|token| {
                self.pending
                    .lock()
                    .ok()
                    .and_then(|mut pending| pending.remove(&token))
            })
            .or_else(|| default_printer().map(Destination::Printer))
            .unwrap_or(Destination::File);

        let spooled = match spool(&fd, &title) {
            Ok(path) => path,
            Err(err) => {
                tracing::error!(%err, "could not spool the document");
                return empty(FAILED);
            }
        };

        match destination {
            Destination::Printer(queue) => {
                let status = Command::new("lp")
                    .arg("-d")
                    .arg(&queue)
                    .arg("-t")
                    .arg(&title)
                    .arg(&spooled)
                    .status();
                let _ = std::fs::remove_file(&spooled);
                match status {
                    Ok(status) if status.success() => {
                        tracing::info!(%queue, "job submitted");
                        (SUCCESS, HashMap::new())
                    }
                    Ok(status) => {
                        tracing::error!(?status, %queue, "lp refused the job");
                        empty(FAILED)
                    }
                    Err(err) => {
                        tracing::error!(%err, "could not run lp (is cups installed?)");
                        empty(FAILED)
                    }
                }
            }
            Destination::File => {
                let suggested =
                    format!("{}.pdf", if title.is_empty() { "document" } else { &title });
                let spec = ChooserSpec {
                    title: "Save as PDF".to_string(),
                    kind: Kind::Save,
                    accept_label: Some("Save".to_string()),
                    filters: vec![crate::ui::filechooser::Filter {
                        name: "PDF documents".to_string(),
                        rules: vec![(0, "*.pdf".to_string())],
                    }],
                    current_filter: Some(0),
                    choices: Vec::new(),
                    start_dir: None,
                    suggested_name: Some(suggested),
                };
                with_request(conn, &handle, |cancel| async move {
                    let outcome = match ui::dialog(FileChooser::new(spec), cancel).await {
                        Ok(dialog) => dialog.outcome,
                        Err(err) => {
                            tracing::error!(%err, "save-as-PDF dialog failed");
                            let _ = std::fs::remove_file(&spooled);
                            return empty(FAILED);
                        }
                    };
                    let Some(target) = outcome.and_then(|o| o.paths.into_iter().next()) else {
                        let _ = std::fs::remove_file(&spooled);
                        return empty(CANCELLED);
                    };
                    // Copy rather than rename: the spool is on the runtime
                    // filesystem, which is rarely the same one as $HOME.
                    let result = std::fs::copy(&spooled, &target);
                    let _ = std::fs::remove_file(&spooled);
                    match result {
                        Ok(_) => {
                            tracing::info!(path = %target.display(), "document saved");
                            (SUCCESS, HashMap::new())
                        }
                        Err(err) => {
                            tracing::error!(%err, "could not write the document");
                            empty(FAILED)
                        }
                    }
                })
                .await
            }
        }
    }

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        2
    }
}
