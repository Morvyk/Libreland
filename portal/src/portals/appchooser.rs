//! `org.freedesktop.impl.portal.AppChooser` — "Open With…".
//!
//! The frontend hands us the applications it thinks can handle the content and
//! expects a desktop id back (it does the launching and the
//! remember-my-choice bookkeeping). It may also call `UpdateChoices` while the
//! dialog is already on screen, after a slower content-type sniff finishes —
//! so the live dialog needs a way to hear about it, which is what the inbox
//! below is.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zbus::zvariant::{OwnedObjectPath, OwnedValue};
use zbus::{Connection, interface};

use crate::apps;
use crate::ui;
use crate::ui::appgrid::AppChooser as Dialog;

use super::{CANCELLED, FAILED, PortalResult, SUCCESS, empty, opt_str, ov, with_request};

/// Pending `UpdateChoices` payloads, keyed by request handle.
type Inboxes = Arc<Mutex<HashMap<String, Arc<Mutex<Option<Vec<String>>>>>>>;

pub struct AppChooser {
    inboxes: Inboxes,
}

impl AppChooser {
    pub fn new() -> Self {
        Self {
            inboxes: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// Resolve desktop ids to entries, dropping any that no longer parse.
fn resolve(ids: &[String]) -> Vec<apps::DesktopApp> {
    ids.iter().filter_map(|id| apps::find(id)).collect()
}

/// A one-line description of what is being opened, for the dialog's subtitle.
fn heading(options: &HashMap<String, OwnedValue>) -> String {
    if let Some(uri) = opt_str(options, "uri") {
        return uri;
    }
    if let Some(name) = opt_str(options, "filename") {
        return name;
    }
    opt_str(options, "content_type").unwrap_or_else(|| "Choose an application".to_string())
}

#[interface(name = "org.freedesktop.impl.portal.AppChooser")]
impl AppChooser {
    async fn choose_application(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        choices: Vec<String>,
        options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        tracing::info!(app = %app_id, count = choices.len(), "app chooser");
        let mut suggested = resolve(&choices);
        // `last_choice` is what the user picked last time for this content
        // type; float it to the top rather than pre-selecting it silently.
        if let Some(last) = opt_str(&options, "last_choice")
            && let Some(index) = suggested
                .iter()
                .position(|a| a.id.trim_end_matches(".desktop") == last)
        {
            let app = suggested.remove(index);
            suggested.insert(0, app);
        }

        let inbox: Arc<Mutex<Option<Vec<String>>>> = Arc::new(Mutex::new(None));
        if let Ok(mut inboxes) = self.inboxes.lock() {
            inboxes.insert(handle.as_str().to_string(), Arc::clone(&inbox));
        }
        let inboxes = Arc::clone(&self.inboxes);
        let key = handle.as_str().to_string();
        let heading = heading(&options);

        let result = with_request(conn, &handle, |cancel| async move {
            let dialog = Dialog::new(heading, suggested, inbox);
            match ui::dialog(dialog, cancel).await {
                Ok(dialog) => dialog.chosen.map_or_else(
                    || empty(CANCELLED),
                    |choice| {
                        let mut results = HashMap::new();
                        results.insert("choice".to_string(), ov(choice.as_str()));
                        (SUCCESS, results)
                    },
                ),
                Err(err) => {
                    tracing::error!(%err, "app chooser failed");
                    empty(FAILED)
                }
            }
        })
        .await;

        if let Ok(mut inboxes) = inboxes.lock() {
            inboxes.remove(&key);
        }
        result
    }

    /// The frontend refined its list while the dialog is up.
    fn update_choices(&self, handle: OwnedObjectPath, choices: Vec<String>) {
        let Ok(inboxes) = self.inboxes.lock() else {
            return;
        };
        if let Some(inbox) = inboxes.get(handle.as_str())
            && let Ok(mut slot) = inbox.lock()
        {
            *slot = Some(choices);
        }
    }

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        2
    }
}
