//! `org.freedesktop.impl.portal.Wallpaper` — let an app set the desktop
//! background.
//!
//! Libreland's wallpaper is part of its Lua config (`misc.wallpaper`), and the
//! compositor can switch to a media file at runtime over its control IPC. So
//! the flow is: confirm with the user, ask the compositor to show the image
//! now, and record it in `$XDG_STATE_HOME/libreland/wallpaper` so a shell
//! script or the user's own config can make it stick across restarts.
//!
//! We deliberately do *not* rewrite the user's `libreland.lua`: a portal that
//! edits hand-written config on an app's say-so is a portal that eventually
//! eats someone's comments and formatting.

use std::collections::HashMap;
use std::path::PathBuf;

use zbus::zvariant::{OwnedObjectPath, OwnedValue};
use zbus::{Connection, interface};

use crate::ipc;
use crate::ui;
use crate::ui::prompt::{Prompt, Spec};

use super::{CANCELLED, FAILED, SUCCESS, opt_str, uri_to_path, with_request};

pub struct Wallpaper;

impl Wallpaper {
    pub const fn new() -> Self {
        Self
    }
}

/// Where we record the current wallpaper for anything that wants to restore it.
fn state_file() -> PathBuf {
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()));
    std::env::var("XDG_STATE_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map_or_else(|| home.join(".local/state"), PathBuf::from)
        .join("libreland/wallpaper")
}

#[interface(name = "org.freedesktop.impl.portal.Wallpaper")]
impl Wallpaper {
    async fn set_wallpaper_uri(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        uri: String,
        options: HashMap<String, OwnedValue>,
    ) -> u32 {
        let path = uri_to_path(&uri);
        tracing::info!(app = %app_id, path = %path.display(), "wallpaper requested");
        if !path.is_file() {
            tracing::warn!(path = %path.display(), "wallpaper file does not exist");
            return FAILED;
        }
        // `set-on` is background/lockscreen/both. Libreland has no separate
        // lock-screen wallpaper (the lock client draws its own), so a
        // lockscreen-only request has nothing to act on.
        let target = opt_str(&options, "set-on").unwrap_or_else(|| "background".into());
        if target == "lockscreen" {
            tracing::info!("ignoring a lockscreen-only wallpaper request");
            return FAILED;
        }
        let app = if app_id.is_empty() {
            "An application".to_string()
        } else {
            app_id.clone()
        };
        let spec = Spec {
            title: "Set desktop background?".to_string(),
            subtitle: String::new(),
            body: format!(
                "{app} wants to change your wallpaper to:\n\n{}",
                path.display()
            ),
            accept_label: Some("Set background".to_string()),
            deny_label: Some("Keep current".to_string()),
            destructive: false,
            toggles: Vec::new(),
        };

        let (response, _) = with_request(conn, &handle, |cancel| async move {
            match ui::dialog(Prompt::new(spec), cancel).await {
                Ok(prompt) if prompt.accepted => {
                    if let Err(err) = ipc::set_wallpaper(&path) {
                        tracing::error!(%err, "the compositor refused the wallpaper");
                        return super::empty(FAILED);
                    }
                    // Best-effort record for restore-at-login; failing to
                    // write it doesn't undo a wallpaper that is already up.
                    if let Some(dir) = state_file().parent() {
                        let _ = std::fs::create_dir_all(dir);
                    }
                    let _ = std::fs::write(state_file(), path.to_string_lossy().as_bytes());
                    super::empty(SUCCESS)
                }
                Ok(_) => super::empty(CANCELLED),
                Err(err) => {
                    tracing::error!(%err, "wallpaper dialog failed");
                    super::empty(FAILED)
                }
            }
        })
        .await;
        response
    }

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        1
    }
}
