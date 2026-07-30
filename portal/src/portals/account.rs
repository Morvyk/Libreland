//! `org.freedesktop.impl.portal.Account` — share the user's identity with an
//! app, if they say yes.
//!
//! The information is whatever the system already knows: the login name, the
//! real name from the passwd GECOS field, and an avatar if one of the usual
//! files exists. Nothing is invented, and nothing leaves without consent —
//! which is the entire reason this is a portal instead of a `getpwuid` call
//! inside the app.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use zbus::zvariant::{OwnedObjectPath, OwnedValue};
use zbus::{Connection, interface};

use crate::ui;
use crate::ui::prompt::{Prompt, Spec};

use super::{
    CANCELLED, FAILED, PortalResult, SUCCESS, empty, opt_str, ov, path_to_uri, with_request,
};

pub struct Account;

impl Account {
    pub const fn new() -> Self {
        Self
    }
}

/// Login name, real name, and avatar path for the current user.
fn user_info() -> (String, String, Option<PathBuf>) {
    let user = nix::unistd::User::from_uid(nix::unistd::Uid::current())
        .ok()
        .flatten();
    let login = user
        .as_ref()
        .map(|u| u.name.clone())
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_default();
    // GECOS is comma-separated; the first field is the real name.
    let real = user
        .as_ref()
        .map(|u| u.gecos.to_string_lossy().to_string())
        .and_then(|gecos| {
            gecos
                .split(',')
                .next()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| login.clone());
    let home = user.as_ref().map_or_else(
        || PathBuf::from(std::env::var("HOME").unwrap_or_default()),
        |u| u.dir.clone(),
    );
    // The two places an avatar conventionally lives: the AccountsService
    // cache (what GNOME/GDM write) and the classic ~/.face.
    let avatar = [
        PathBuf::from("/var/lib/AccountsService/icons").join(&login),
        home.join(".face"),
        home.join(".face.icon"),
    ]
    .into_iter()
    .find(|p| p.is_file());
    (login, real, avatar)
}

#[interface(name = "org.freedesktop.impl.portal.Account")]
impl Account {
    async fn get_user_information(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _window: String,
        options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        let (login, real, avatar) = user_info();
        let reason = opt_str(&options, "reason").unwrap_or_default();
        let app = if app_id.is_empty() {
            "An application".to_string()
        } else {
            app_id.clone()
        };
        tracing::info!(app = %app_id, "user information requested");

        let mut body = format!("{app} wants to know who you are.\n\nName: {real}\nUser: {login}");
        if avatar.is_some() {
            body.push_str("\nAvatar: yes");
        }
        if !reason.is_empty() {
            let _ = write!(body, "\n\n{reason}");
        }
        let spec = Spec {
            title: "Share your information?".to_string(),
            subtitle: String::new(),
            body,
            accept_label: Some("Share".to_string()),
            deny_label: Some("Don't share".to_string()),
            destructive: false,
            toggles: Vec::new(),
        };

        with_request(conn, &handle, |cancel| async move {
            match ui::dialog(Prompt::new(spec), cancel).await {
                Ok(prompt) if prompt.accepted => {
                    let mut results = HashMap::new();
                    results.insert("id".to_string(), ov(login.as_str()));
                    results.insert("name".to_string(), ov(real.as_str()));
                    // `image` is a URI, and an empty string is the spec's way
                    // of saying "no avatar".
                    results.insert(
                        "image".to_string(),
                        ov(avatar
                            .map(|p| path_to_uri(&p))
                            .unwrap_or_default()
                            .as_str()),
                    );
                    (SUCCESS, results)
                }
                Ok(_) => empty(CANCELLED),
                Err(err) => {
                    tracing::error!(%err, "account dialog failed");
                    empty(FAILED)
                }
            }
        })
        .await
    }

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        1
    }
}
