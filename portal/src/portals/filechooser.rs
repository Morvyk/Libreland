//! `org.freedesktop.impl.portal.FileChooser` — Open, Save, and Save-multiple.
//!
//! The interface is thin: unpack the options into a [`Spec`], show the dialog,
//! pack what came back into URIs. The sandboxing half lives in the frontend,
//! which turns our URIs into document-store handles for confined callers — so
//! this side never reasons about sandboxes, it only has to be honest about
//! what the user actually picked.

use std::collections::HashMap;
use std::path::PathBuf;

use zbus::zvariant::{OwnedObjectPath, OwnedValue, Value};
use zbus::{Connection, interface};

use crate::ui;
use crate::ui::filechooser::{Choice, FileChooser as Dialog, Filter, Kind, Outcome, Spec};

use super::{
    CANCELLED, FAILED, PortalResult, SUCCESS, empty, opt_bool, opt_bytes, opt_str, ov,
    path_to_uri, with_request,
};

pub struct FileChooser;

/// Read `filters` (`a(sa(us))`) out of the options.
fn parse_filters(options: &HashMap<String, OwnedValue>) -> Vec<Filter> {
    let Some(Value::Array(array)) = options.get("filters").map(std::ops::Deref::deref) else {
        return Vec::new();
    };
    array.iter().filter_map(parse_filter).collect()
}

/// Read one filter (`(sa(us))`).
fn parse_filter(value: &Value<'_>) -> Option<Filter> {
    let Value::Structure(entry) = value else {
        return None;
    };
    let fields = entry.fields();
    let Some(Value::Str(name)) = fields.first() else {
        return None;
    };
    let mut rules = Vec::new();
    if let Some(Value::Array(patterns)) = fields.get(1) {
        for pattern in patterns.iter() {
            let Value::Structure(rule) = pattern else {
                continue;
            };
            let rule = rule.fields();
            if let (Some(Value::U32(kind)), Some(Value::Str(text))) = (rule.first(), rule.get(1)) {
                rules.push((*kind, text.to_string()));
            }
        }
    }
    Some(Filter {
        name: name.to_string(),
        rules,
    })
}

/// Read `choices` (`a(ssa(ss)s)`).
fn parse_choices(options: &HashMap<String, OwnedValue>) -> Vec<Choice> {
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
                Some(Value::Str(s)) => s.to_string(),
                _ => String::new(),
            };
            // An empty option list means a checkbox carrying "true"/"false";
            // either way the dialog needs a definite starting value.
            let selected = if !initial.is_empty() {
                initial
            } else if options.is_empty() {
                "false".to_string()
            } else {
                options.first().map(|(id, _)| id.clone()).unwrap_or_default()
            };
            Some(Choice {
                id: id.to_string(),
                label: label.to_string(),
                options,
                selected,
            })
        })
        .collect()
}

/// `current_folder` is a NUL-terminated byte path.
fn folder_option(options: &HashMap<String, OwnedValue>, key: &str) -> Option<PathBuf> {
    use std::os::unix::ffi::OsStringExt as _;
    opt_bytes(options, key).map(|bytes| PathBuf::from(std::ffi::OsString::from_vec(bytes)))
}

/// Build the shared parts of a spec from the options dictionary.
fn base_spec(title: &str, options: &HashMap<String, OwnedValue>, kind: Kind) -> Spec {
    let filters = parse_filters(options);
    // `current_filter` repeats one of the filters; match it by name so the
    // dialog opens on the one the app asked for.
    let current_filter = options
        .get("current_filter")
        .map(std::ops::Deref::deref)
        .and_then(parse_filter)
        .and_then(|wanted| filters.iter().position(|f| f.name == wanted.name));
    Spec {
        title: title.to_string(),
        kind,
        // Accept labels arrive with GTK mnemonic underscores ("_Open").
        accept_label: opt_str(options, "accept_label").map(|label| label.replace('_', "")),
        filters,
        current_filter,
        choices: parse_choices(options),
        start_dir: folder_option(options, "current_folder").filter(|p| p.is_dir()),
        suggested_name: None,
    }
}

/// Turn the dialog's outcome into the results dictionary the spec defines.
fn results(outcome: &Outcome, spec: &Spec, writable: bool) -> HashMap<String, OwnedValue> {
    let mut results = HashMap::new();
    let uris: Vec<String> = outcome.paths.iter().map(|p| path_to_uri(p)).collect();
    results.insert("uris".to_string(), ov(uris));
    results.insert("writable".to_string(), ov(writable));
    if !outcome.choices.is_empty() {
        results.insert("choices".to_string(), ov(outcome.choices.clone()));
    }
    if let Some(filter) = outcome.filter.and_then(|i| spec.filters.get(i)) {
        results.insert(
            "current_filter".to_string(),
            ov((filter.name.clone(), filter.rules.clone())),
        );
    }
    results
}

/// Run one chooser to a portal result.
async fn choose(
    conn: &Connection,
    handle: &OwnedObjectPath,
    spec: Spec,
    writable: bool,
) -> PortalResult {
    with_request(conn, handle, |cancel| async move {
        match ui::dialog(Dialog::new(spec.clone()), cancel).await {
            Ok(dialog) => dialog.outcome.map_or_else(
                || empty(CANCELLED),
                |outcome| (SUCCESS, results(&outcome, &spec, writable)),
            ),
            Err(err) => {
                tracing::error!(%err, "file chooser failed");
                empty(FAILED)
            }
        }
    })
    .await
}

impl FileChooser {
    pub const fn new() -> Self {
        Self
    }
}

#[interface(name = "org.freedesktop.impl.portal.FileChooser")]
impl FileChooser {
    async fn open_file(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        title: String,
        options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        tracing::info!(app = %app_id, "OpenFile");
        let kind = Kind::Open {
            multiple: opt_bool(&options, "multiple").unwrap_or(false),
            directory: opt_bool(&options, "directory").unwrap_or(false),
        };
        choose(conn, &handle, base_spec(&title, &options, kind), false).await
    }

    async fn save_file(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        title: String,
        options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        tracing::info!(app = %app_id, "SaveFile");
        let mut spec = base_spec(&title, &options, Kind::Save);
        spec.suggested_name = opt_str(&options, "current_name");
        // `current_file` is the full path of a file being re-saved: it sets
        // both the folder and the name, and outranks `current_folder`.
        if let Some(path) = folder_option(&options, "current_file") {
            if let Some(parent) = path.parent().filter(|p| p.is_dir()) {
                spec.start_dir = Some(parent.to_path_buf());
            }
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                spec.suggested_name = Some(name.to_string());
            }
        }
        choose(conn, &handle, spec, true).await
    }

    async fn save_files(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        title: String,
        options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        tracing::info!(app = %app_id, "SaveFiles");
        // `files` is an array of NUL-terminated byte names, one per file the
        // app intends to write into the folder the user picks.
        let files: Vec<String> = match options.get("files").map(std::ops::Deref::deref) {
            Some(Value::Array(array)) => array
                .iter()
                .filter_map(|value| {
                    let Value::Array(bytes) = value else {
                        return None;
                    };
                    let mut raw: Vec<u8> = bytes
                        .iter()
                        .filter_map(|b| match b {
                            Value::U8(byte) => Some(*byte),
                            _ => None,
                        })
                        .collect();
                    if raw.last() == Some(&0) {
                        raw.pop();
                    }
                    String::from_utf8(raw).ok()
                })
                .collect(),
            _ => Vec::new(),
        };
        let spec = base_spec(&title, &options, Kind::SaveFolder { files });
        choose(conn, &handle, spec, true).await
    }

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        3
    }
}
