//! `org.freedesktop.impl.portal.Access` — the generic "may this app do that?"
//! dialog.
//!
//! The frontend routes every permission question it can't answer from the
//! permission store through here: camera, microphone, location, background
//! running. All the wording comes from the caller, so this is a thin shell
//! over [`Prompt`].

use std::collections::HashMap;

use zbus::zvariant::{OwnedObjectPath, OwnedValue, Value};
use zbus::{Connection, interface};

use crate::ui;
use crate::ui::prompt::{Prompt, Spec, Toggle};

use super::{CANCELLED, FAILED, PortalResult, SUCCESS, empty, opt_str, ov, with_request};

pub struct Access;

impl Access {
    pub const fn new() -> Self {
        Self
    }
}

/// Parse `choices` (`a(ssa(ss)s)`) — the same encoding the file chooser uses,
/// shared with the other prompt-shaped portals.
pub fn parse_toggles(options: &HashMap<String, OwnedValue>) -> Vec<Toggle> {
    let Some(Value::Array(array)) = options.get("choices").map(std::ops::Deref::deref) else {
        return Vec::new();
    };
    array
        .iter()
        .filter_map(|value| {
            let Value::Structure(entry) = value else {
                return None;
            };
            let fields = entry.fields();
            let (Some(Value::Str(id)), Some(Value::Str(label))) = (fields.first(), fields.get(1))
            else {
                return None;
            };
            let mut options = Vec::new();
            if let Some(Value::Array(pairs)) = fields.get(2) {
                for option in pairs.iter() {
                    let Value::Structure(pair) = option else {
                        continue;
                    };
                    let pair = pair.fields();
                    if let (Some(Value::Str(id)), Some(Value::Str(label))) =
                        (pair.first(), pair.get(1))
                    {
                        options.push((id.to_string(), label.to_string()));
                    }
                }
            }
            let initial = match fields.get(3) {
                Some(Value::Str(s)) if !s.is_empty() => s.to_string(),
                _ if options.is_empty() => "false".to_string(),
                _ => options.first().map(|(id, _)| id.clone()).unwrap_or_default(),
            };
            Some(Toggle {
                id: id.to_string(),
                label: label.to_string(),
                options,
                selected: initial,
            })
        })
        .collect()
}

#[interface(name = "org.freedesktop.impl.portal.Access")]
impl Access {
    #[allow(
        clippy::too_many_arguments,
        reason = "the argument list is fixed by the portal interface definition"
    )]
    async fn access_dialog(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        title: String,
        subtitle: String,
        body: String,
        options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        tracing::info!(app = %app_id, %title, "access request");
        let spec = Spec {
            title,
            subtitle,
            body,
            // Mnemonic underscores are a GTK convention we don't draw.
            accept_label: opt_str(&options, "grant_label").map(|l| l.replace('_', "")),
            deny_label: opt_str(&options, "deny_label").map(|l| l.replace('_', "")),
            destructive: false,
            toggles: parse_toggles(&options),
        };
        with_request(conn, &handle, |cancel| async move {
            match ui::dialog(Prompt::new(spec), cancel).await {
                Ok(prompt) if prompt.accepted => {
                    let mut results = HashMap::new();
                    let choices = prompt.choices();
                    if !choices.is_empty() {
                        results.insert("choices".to_string(), ov(choices));
                    }
                    (SUCCESS, results)
                }
                Ok(_) => empty(CANCELLED),
                Err(err) => {
                    tracing::error!(%err, "access dialog failed");
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
