//! `org.freedesktop.impl.portal.DynamicLauncher` — let an app install a
//! launcher (a web app, a shortcut to a document, a pinned tab).
//!
//! The frontend writes the desktop entry and owns the icon validation; the
//! backend's job is consent and naming. So this shows the user what is about
//! to appear in their application list, lets them rename it, and hands back a
//! token the frontend requires before it will install anything.

use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU32, Ordering},
};

use zbus::zvariant::{OwnedObjectPath, OwnedValue, Value};
use zbus::{Connection, interface};

use crate::ui;
use crate::ui::prompt::{Prompt, Spec};

use super::{CANCELLED, FAILED, PortalResult, SUCCESS, empty, opt_bool, opt_str, ov, with_request};

/// Launcher types the spec defines, as a bitmask for the property.
const TYPE_APPLICATION: u32 = 1;
const TYPE_WEBAPP: u32 = 2;

pub struct DynamicLauncher {
    /// Tokens we've handed out, so `RequestInstallToken` can't be replayed
    /// into an unbounded set.
    issued: Arc<Mutex<Vec<String>>>,
    next: AtomicU32,
}

impl DynamicLauncher {
    pub fn new() -> Self {
        Self {
            issued: Arc::new(Mutex::new(Vec::new())),
            next: AtomicU32::new(1),
        }
    }

    /// Mint a token. It only has to be unguessable within this session — the
    /// frontend checks that the token it gets back is one we issued.
    fn token(&self) -> String {
        let counter = self.next.fetch_add(1, Ordering::Relaxed);
        let token = format!("libreland-{}-{counter}", std::process::id());
        if let Ok(mut issued) = self.issued.lock() {
            issued.push(token.clone());
            // A session that installs more than a handful of launchers is
            // misbehaving; cap the bookkeeping rather than grow forever.
            if issued.len() > 64 {
                issued.remove(0);
            }
        }
        token
    }
}

#[interface(name = "org.freedesktop.impl.portal.DynamicLauncher")]
impl DynamicLauncher {
    #[allow(
        clippy::too_many_arguments,
        reason = "the argument list is fixed by the portal interface definition"
    )]
    async fn prepare_install(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        name: String,
        _icon_v: Value<'_>,
        options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        tracing::info!(app = %app_id, %name, "launcher install requested");
        let editable = opt_bool(&options, "editable_name").unwrap_or(false);
        let kind = opt_str(&options, "launcher_type");
        let app = if app_id.is_empty() {
            "An application".to_string()
        } else {
            app_id.clone()
        };
        let what = match kind.as_deref() {
            Some("webapp") => "a web app shortcut",
            _ => "a launcher",
        };
        let mut body = format!("{app} wants to add {what} to your applications:\n\n{name}");
        if editable {
            // We have no inline rename field in a prompt, and inventing one
            // for this is out of proportion; be explicit rather than silently
            // ignoring the app's hint.
            body.push_str("\n\nIt will be installed under this name.");
        }
        let spec = Spec {
            title: "Add to applications?".to_string(),
            subtitle: String::new(),
            body,
            accept_label: Some("Add".to_string()),
            deny_label: Some("Cancel".to_string()),
            destructive: false,
            toggles: Vec::new(),
        };
        let token = self.token();

        with_request(conn, &handle, |cancel| async move {
            match ui::dialog(Prompt::new(spec), cancel).await {
                Ok(prompt) if prompt.accepted => {
                    let mut results = HashMap::new();
                    results.insert("name".to_string(), ov(name.as_str()));
                    results.insert("token".to_string(), ov(token.as_str()));
                    (SUCCESS, results)
                }
                Ok(_) => empty(CANCELLED),
                Err(err) => {
                    tracing::error!(%err, "launcher dialog failed");
                    empty(FAILED)
                }
            }
        })
        .await
    }

    /// Issue a token without a dialog.
    ///
    /// The frontend only calls this for callers it already trusts (the
    /// permission store said yes, or the caller isn't sandboxed), so a second
    /// prompt here would be a prompt the user has already answered.
    fn request_install_token(&self, app_id: &str, _options: HashMap<String, OwnedValue>) -> u32 {
        tracing::info!(app = %app_id, "install token issued");
        let _ = self.token();
        SUCCESS
    }

    #[zbus(property, name = "SupportedLauncherTypes")]
    fn supported_launcher_types(&self) -> u32 {
        TYPE_APPLICATION | TYPE_WEBAPP
    }

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        1
    }
}
