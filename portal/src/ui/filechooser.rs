//! The file chooser dialog — open, open-multiple, select-folder and save.
//!
//! This is the portal interaction users see most often, and the one that made
//! a GTK backend feel mandatory. It isn't: a file chooser is a directory
//! listing, a sort, a filter and a text field.
//!
//! Behaviour is modelled on what people already have in their fingers —
//! double-click or Enter to descend, Backspace to go up, type to filter,
//! Ctrl+H for hidden files, Ctrl+L to type a path — rather than on any one
//! toolkit's chrome.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel geometry: every value here is a surface- or image-sized non-negative integer, and the conversions between i32/u32/usize/f32 are all inside that range. Checked conversions at each site would be noise around arithmetic that cannot overflow."
)]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::draw::{Canvas, Rect};
use super::widgets::{self, Emphasis, Scroll, TextField};
use super::{Ctx, Flow, Input, Key, Mods, Screen};

// ── Public shape ───────────────────────────────────────────────────────────

/// One entry of the portal's `filters` option: a display name and the rules
/// that decide what it matches.
#[derive(Clone, Debug)]
pub struct Filter {
    pub name: String,
    /// `(kind, pattern)` where kind 0 is a shell glob and 1 is a MIME type.
    pub rules: Vec<(u32, String)>,
}

impl Filter {
    fn accepts(&self, path: &Path) -> bool {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        self.rules.iter().any(|(kind, pattern)| match kind {
            1 => mime_matches(pattern, &name),
            _ => glob_matches(&pattern.to_ascii_lowercase(), &name),
        })
    }
}

/// One entry of the `choices` option: a labelled combo the app wants answered
/// alongside the file (an export format, an encoding, …).
#[derive(Clone, Debug)]
pub struct Choice {
    pub id: String,
    pub label: String,
    /// `(id, label)` pairs. An empty list means a boolean toggle whose value
    /// is the string "true"/"false", which is how the spec encodes checkboxes.
    pub options: Vec<(String, String)>,
    pub selected: String,
}

/// Which flavour of chooser to present.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Open { multiple: bool, directory: bool },
    Save,
    /// `SaveFiles`: the app supplies the names, the user only picks a folder.
    SaveFolder { files: Vec<String> },
}

/// Everything the portal method knows, handed to the dialog.
#[derive(Clone, Debug)]
pub struct Spec {
    pub title: String,
    pub kind: Kind,
    pub accept_label: Option<String>,
    pub filters: Vec<Filter>,
    pub current_filter: Option<usize>,
    pub choices: Vec<Choice>,
    pub start_dir: Option<PathBuf>,
    pub suggested_name: Option<String>,
}

/// What the user settled on.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub paths: Vec<PathBuf>,
    pub filter: Option<usize>,
    pub choices: Vec<(String, String)>,
}

// ── Matching helpers ───────────────────────────────────────────────────────

/// Glob match supporting `*` and `?`, which is everything file-dialog filters
/// use in practice (`*.png`, `*.tar.*`).
fn glob_matches(pattern: &str, name: &str) -> bool {
    fn walk(p: &[u8], n: &[u8]) -> bool {
        match (p.first(), n.first()) {
            (None, None) => true,
            // Either the star eats nothing, or it eats one more byte.
            (Some(b'*'), _) => walk(&p[1..], n) || (!n.is_empty() && walk(p, &n[1..])),
            (Some(b'?'), Some(_)) => walk(&p[1..], &n[1..]),
            (Some(a), Some(b)) if a == b => walk(&p[1..], &n[1..]),
            _ => false,
        }
    }
    walk(pattern.as_bytes(), name.as_bytes())
}

/// Map a filename to a MIME type by extension, then match it against a
/// possibly-wildcarded pattern (`image/*`).
///
/// A real MIME database (shared-mime-info plus content sniffing) is a
/// dependency and a runtime cost out of proportion to what a filter needs; the
/// table below covers the types apps actually filter on, and an unknown
/// extension simply doesn't match — the user can still switch filters.
fn mime_matches(pattern: &str, name: &str) -> bool {
    const TYPES: &[(&str, &str)] = &[
        ("png", "image/png"),
        ("jpg", "image/jpeg"),
        ("jpeg", "image/jpeg"),
        ("gif", "image/gif"),
        ("webp", "image/webp"),
        ("bmp", "image/bmp"),
        ("tif", "image/tiff"),
        ("tiff", "image/tiff"),
        ("svg", "image/svg+xml"),
        ("ico", "image/x-icon"),
        ("avif", "image/avif"),
        ("heic", "image/heif"),
        ("pdf", "application/pdf"),
        ("txt", "text/plain"),
        ("md", "text/markdown"),
        ("csv", "text/csv"),
        ("html", "text/html"),
        ("htm", "text/html"),
        ("css", "text/css"),
        ("js", "text/javascript"),
        ("json", "application/json"),
        ("xml", "application/xml"),
        ("rs", "text/rust"),
        ("py", "text/x-python"),
        ("sh", "application/x-shellscript"),
        ("c", "text/x-csrc"),
        ("h", "text/x-chdr"),
        ("cpp", "text/x-c++src"),
        ("mp3", "audio/mpeg"),
        ("flac", "audio/flac"),
        ("wav", "audio/wav"),
        ("ogg", "audio/ogg"),
        ("opus", "audio/opus"),
        ("m4a", "audio/mp4"),
        ("mp4", "video/mp4"),
        ("mkv", "video/x-matroska"),
        ("webm", "video/webm"),
        ("avi", "video/x-msvideo"),
        ("mov", "video/quicktime"),
        ("zip", "application/zip"),
        ("gz", "application/gzip"),
        ("xz", "application/x-xz"),
        ("zst", "application/zstd"),
        ("tar", "application/x-tar"),
        ("7z", "application/x-7z-compressed"),
        ("rar", "application/vnd.rar"),
        ("iso", "application/x-cd-image"),
        ("deb", "application/vnd.debian.binary-package"),
        ("rpm", "application/x-rpm"),
        ("odt", "application/vnd.oasis.opendocument.text"),
        ("ods", "application/vnd.oasis.opendocument.spreadsheet"),
        ("odp", "application/vnd.oasis.opendocument.presentation"),
        ("doc", "application/msword"),
        (
            "docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        ("xls", "application/vnd.ms-excel"),
        (
            "xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        ("epub", "application/epub+zip"),
        ("ttf", "font/ttf"),
        ("otf", "font/otf"),
        ("desktop", "application/x-desktop"),
    ];
    let Some(ext) = name.rsplit_once('.').map(|(_, e)| e) else {
        return false;
    };
    let Some((_, mime)) = TYPES.iter().find(|(e, _)| *e == ext) else {
        return false;
    };
    match pattern.split_once('/') {
        Some((group, "*")) => mime.starts_with(group) && mime[group.len()..].starts_with('/'),
        _ => pattern.eq_ignore_ascii_case(mime),
    }
}

// ── Listing ────────────────────────────────────────────────────────────────

struct Entry {
    name: String,
    path: PathBuf,
    is_dir: bool,
    size: u64,
    mtime: Option<SystemTime>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SortBy {
    Name,
    Size,
    Modified,
}

/// Human-readable size in binary units (what file managers show).
fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else if value < 10.0 {
        format!("{value:.1} {}", UNITS[unit])
    } else {
        format!("{value:.0} {}", UNITS[unit])
    }
}

/// `YYYY-MM-DD HH:MM` in UTC.
///
/// Local time would mean carrying a tz database into a dialog process; an ISO
/// stamp is unambiguous, which is the property that matters when you're
/// choosing between two similarly-named files.
fn format_time(time: Option<SystemTime>) -> String {
    let Some(secs) = time
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
    else {
        return String::new();
    };
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Howard Hinnant's civil-from-days: exact across the proleptic Gregorian
    // range, no tables and no dependency.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}",
        rem / 3600,
        (rem % 3600) / 60
    )
}

/// The XDG user directories, read from `user-dirs.dirs` (the file
/// `xdg-user-dirs` writes) so the sidebar matches the rest of the desktop
/// instead of hardcoding English folder names.
fn user_dirs(home: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    let config = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map_or_else(|| home.join(".config"), PathBuf::from);
    let Ok(text) = std::fs::read_to_string(config.join("user-dirs.dirs")) else {
        return out;
    };
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("XDG_") else {
            continue;
        };
        let Some((key, value)) = rest.split_once('=') else {
            continue;
        };
        let Some(key) = key.strip_suffix("_DIR") else {
            continue;
        };
        let value = value.trim().trim_matches('"');
        let path = value
            .strip_prefix("$HOME/")
            .map_or_else(|| PathBuf::from(value), |rest| home.join(rest));
        if path == home || !path.is_dir() {
            continue;
        }
        // DOWNLOAD -> Download, PUBLICSHARE -> Publicshare.
        let mut label = key.to_ascii_lowercase();
        if let Some(first) = label.get_mut(0..1) {
            first.make_ascii_uppercase();
        }
        out.push((label, path));
    }
    out
}

/// Sidebar shortcuts: home, the XDG dirs, the filesystem root, and whatever is
/// mounted under the usual removable-media paths.
fn places() -> Vec<(String, PathBuf)> {
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/".into()));
    let mut out = vec![("Home".to_string(), home.clone())];
    out.extend(user_dirs(&home));
    out.push(("Filesystem".to_string(), PathBuf::from("/")));
    let user = std::env::var("USER").unwrap_or_default();
    for base in [
        PathBuf::from("/run/media").join(&user),
        PathBuf::from("/media").join(&user),
        PathBuf::from("/mnt"),
    ] {
        let Ok(entries) = std::fs::read_dir(&base) else {
            continue;
        };
        for entry in entries.flatten().take(8) {
            if entry.path().is_dir()
                && let Some(name) = entry.file_name().to_str()
            {
                out.push((name.to_string(), entry.path()));
            }
        }
    }
    out
}

/// Expand a leading `~` or `$HOME`, so a typed path behaves like a shell one.
fn expand(input: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    if input == "~" {
        return PathBuf::from(&home);
    }
    if let Some(rest) = input.strip_prefix("~/") {
        return PathBuf::from(home).join(rest);
    }
    if let Some(rest) = input.strip_prefix("$HOME/") {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(input)
}

// ── The dialog ─────────────────────────────────────────────────────────────

/// Rectangles recorded during render and tested on the next pointer event.
#[derive(Default)]
struct Hit {
    close: Rect,
    up: Rect,
    crumbs: Vec<(Rect, PathBuf)>,
    places: Vec<(Rect, PathBuf)>,
    rows: Rect,
    headers: [Rect; 3],
    name_field: Rect,
    filter_combo: Rect,
    choice_combos: Vec<Rect>,
    popup: Vec<(Rect, usize)>,
    hidden_toggle: Rect,
    cancel: Rect,
    accept: Rect,
}

/// Which combo an open popup belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Popup {
    Filter,
    Choice(usize),
}

pub struct FileChooser {
    spec: Spec,
    dir: PathBuf,
    entries: Vec<Entry>,
    /// Indices into `entries` that survive the search text and active filter.
    visible: Vec<usize>,
    /// Selection, held as indices into `entries`.
    selected: HashSet<usize>,
    /// Keyboard cursor, an index into `visible`.
    cursor: usize,
    scroll: Scroll,
    search: TextField,
    name: TextField,
    /// True when the name field (save mode) owns the keyboard.
    naming: bool,
    show_hidden: bool,
    sort: SortBy,
    sort_desc: bool,
    filter: Option<usize>,
    choices: Vec<Choice>,
    popup: Option<Popup>,
    places: Vec<(String, PathBuf)>,
    hover: Option<(f64, f64)>,
    last_click: Option<(usize, Instant)>,
    hit: Hit,
    error: Option<String>,
    /// Set when the user accepts; stays `None` on cancel.
    pub outcome: Option<Outcome>,
}

const ROW_H: i32 = 30;
const HEADER_H: i32 = 52;
const TOOLBAR_H: i32 = 44;
const SIDEBAR_W: i32 = 190;
const COLS_H: i32 = 26;

impl FileChooser {
    pub fn new(spec: Spec) -> Self {
        let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/".into()));
        let dir = spec
            .start_dir
            .clone()
            .filter(|d| d.is_dir())
            .unwrap_or(home);
        let filter = spec
            .current_filter
            .or_else(|| (!spec.filters.is_empty()).then_some(0));
        let naming = matches!(spec.kind, Kind::Save);
        let mut chooser = Self {
            dir,
            entries: Vec::new(),
            visible: Vec::new(),
            selected: HashSet::new(),
            cursor: 0,
            scroll: Scroll::default(),
            search: TextField::default(),
            name: TextField::new(spec.suggested_name.clone().unwrap_or_default()),
            naming,
            show_hidden: false,
            sort: SortBy::Name,
            sort_desc: false,
            filter,
            choices: spec.choices.clone(),
            popup: None,
            places: places(),
            hover: None,
            last_click: None,
            hit: Hit::default(),
            error: None,
            outcome: None,
            spec,
        };
        chooser.search.focused = !naming;
        chooser.name.focused = naming;
        chooser.reload();
        chooser
    }

    fn accept_label(&self) -> String {
        self.spec.accept_label.clone().unwrap_or_else(|| {
            match &self.spec.kind {
                Kind::Save | Kind::SaveFolder { .. } => "Save",
                Kind::Open {
                    directory: true, ..
                } => "Select",
                Kind::Open { .. } => "Open",
            }
            .to_string()
        })
    }

    /// Re-read the current directory from disk, then re-apply search + filter.
    fn reload(&mut self) {
        self.entries.clear();
        self.selected.clear();
        self.cursor = 0;
        self.scroll.offset = 0.0;
        self.error = None;
        match std::fs::read_dir(&self.dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                        continue;
                    };
                    // `metadata` follows symlinks, which is what a chooser
                    // wants: a symlink to a directory should descend.
                    let meta = std::fs::metadata(&path).ok();
                    let is_dir = meta.as_ref().is_some_and(std::fs::Metadata::is_dir);
                    self.entries.push(Entry {
                        name: name.to_string(),
                        path,
                        is_dir,
                        size: meta.as_ref().map_or(0, std::fs::Metadata::len),
                        mtime: meta.and_then(|m| m.modified().ok()),
                    });
                }
            }
            Err(err) => self.error = Some(format!("Can't open {}: {err}", self.dir.display())),
        }
        self.resort();
        self.refilter();
    }

    fn resort(&mut self) {
        let by = self.sort;
        self.entries.sort_by(|a, b| {
            // Directories always lead, in both sort directions: a chooser
            // where folders scatter through the file list is unusable.
            a.is_dir
                .cmp(&b.is_dir)
                .reverse()
                .then_with(|| match by {
                    SortBy::Name => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                    SortBy::Size => a.size.cmp(&b.size),
                    SortBy::Modified => a.mtime.cmp(&b.mtime),
                })
                .then_with(|| a.name.cmp(&b.name))
        });
        if self.sort_desc {
            // Reverse within each group, so folders stay on top.
            let split = self.entries.iter().filter(|e| e.is_dir).count();
            self.entries[..split].reverse();
            self.entries[split..].reverse();
        }
    }

    fn refilter(&mut self) {
        let needle = self.search.text.to_lowercase();
        let filter = self.filter.and_then(|i| self.spec.filters.get(i));
        let dirs_only = matches!(
            self.spec.kind,
            Kind::Open {
                directory: true,
                ..
            } | Kind::SaveFolder { .. }
        );
        self.visible = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                if !self.show_hidden && entry.name.starts_with('.') {
                    return false;
                }
                if !needle.is_empty() && !entry.name.to_lowercase().contains(&needle) {
                    return false;
                }
                if entry.is_dir {
                    return true;
                }
                if dirs_only {
                    return false;
                }
                filter.is_none_or(|f| f.accepts(&entry.path))
            })
            .map(|(i, _)| i)
            .collect();
        self.cursor = self.cursor.min(self.visible.len().saturating_sub(1));
    }

    fn navigate(&mut self, to: PathBuf) {
        if !to.is_dir() {
            return;
        }
        self.dir = to;
        self.search.set(String::new());
        self.reload();
    }

    /// Descend into a directory, or accept a file.
    fn activate(&mut self, index: usize) -> Flow {
        let Some(&entry_index) = self.visible.get(index) else {
            return Flow::Idle;
        };
        let entry = &self.entries[entry_index];
        if entry.is_dir {
            // Even in directory-select mode a double click descends; the
            // Select button is what confirms "this one".
            let path = entry.path.clone();
            self.navigate(path);
            return Flow::Redraw;
        }
        if matches!(self.spec.kind, Kind::Save) {
            let name = entry.name.clone();
            self.name.set(name);
            return Flow::Redraw;
        }
        self.selected.clear();
        self.selected.insert(entry_index);
        self.finish()
    }

    /// Turn the current state into an [`Outcome`] and end the dialog.
    fn finish(&mut self) -> Flow {
        let choices = self
            .choices
            .iter()
            .map(|c| (c.id.clone(), c.selected.clone()))
            .collect();
        let paths: Vec<PathBuf> = match &self.spec.kind {
            Kind::Save => {
                let typed = self.name.text.trim();
                if typed.is_empty() {
                    self.error = Some("Enter a file name".into());
                    return Flow::Redraw;
                }
                // An absolute or ~-rooted name in the field wins over the
                // browsed directory — people paste whole paths in there.
                let path = expand(typed);
                vec![if path.is_absolute() {
                    path
                } else {
                    self.dir.join(path)
                }]
            }
            Kind::SaveFolder { files } => files.iter().map(|f| self.dir.join(f)).collect(),
            Kind::Open {
                directory: true, ..
            } => {
                let chosen: Vec<PathBuf> = self
                    .selected
                    .iter()
                    .filter_map(|i| self.entries.get(*i))
                    .filter(|e| e.is_dir)
                    .map(|e| e.path.clone())
                    .collect();
                // Nothing highlighted means "this folder" — the one the path
                // bar is showing.
                if chosen.is_empty() {
                    vec![self.dir.clone()]
                } else {
                    chosen
                }
            }
            Kind::Open { .. } => {
                let chosen: Vec<PathBuf> = self
                    .selected
                    .iter()
                    .filter_map(|i| self.entries.get(*i))
                    .map(|e| e.path.clone())
                    .collect();
                if chosen.is_empty() {
                    self.error = Some("Select a file".into());
                    return Flow::Redraw;
                }
                chosen
            }
        };
        self.outcome = Some(Outcome {
            paths,
            filter: self.filter,
            choices,
        });
        Flow::Done
    }

    fn select(&mut self, index: usize, mods: Mods) {
        let Some(&entry_index) = self.visible.get(index) else {
            return;
        };
        let multiple = matches!(self.spec.kind, Kind::Open { multiple: true, .. });
        if multiple && mods.ctrl {
            if !self.selected.remove(&entry_index) {
                self.selected.insert(entry_index);
            }
        } else if multiple && mods.shift {
            let (lo, hi) = if index < self.cursor {
                (index, self.cursor)
            } else {
                (self.cursor, index)
            };
            for i in lo..=hi {
                if let Some(&e) = self.visible.get(i) {
                    self.selected.insert(e);
                }
            }
        } else {
            self.selected.clear();
            self.selected.insert(entry_index);
        }
        self.cursor = index;
        // Clicking a file in save mode fills the name field, so "overwrite
        // this one" is one click plus Enter.
        if matches!(self.spec.kind, Kind::Save)
            && let Some(entry) = self.entries.get(entry_index)
            && !entry.is_dir
        {
            let name = entry.name.clone();
            self.name.set(name);
        }
    }

    /// Height reserved at the bottom for the name field, choices and buttons.
    fn footer_height(&self) -> i32 {
        56 + i32::from(matches!(self.spec.kind, Kind::Save)) * 42
            + i32::from(matches!(self.spec.kind, Kind::SaveFolder { .. })) * 28
            + i32::from(!self.choices.is_empty()) * 40
    }
}

impl Screen for FileChooser {
    fn title(&self) -> String {
        self.spec.title.clone()
    }

    fn size(&self) -> (i32, i32) {
        (980, 640)
    }

    fn tick(&mut self) -> Flow {
        let blinked = if self.naming {
            self.name.tick()
        } else {
            self.search.tick()
        };
        if blinked { Flow::Redraw } else { Flow::Idle }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one layout pass reads better whole than split across helpers that each need the same dozen locals"
    )]
    fn render(&mut self, _surface: usize, canvas: &mut Canvas<'_>, ctx: &Ctx) {
        let theme = &ctx.theme;
        let (w, h) = canvas.size();
        let hover = self.hover;
        let hovered = |rect: Rect| hover.is_some_and(|(x, y)| rect.contains(x, y));
        let mut hit = Hit::default();
        canvas.clear(theme.bg);

        // ── Header ────────────────────────────────────────────────────────
        canvas.fill(Rect::new(0, 0, w, HEADER_H), theme.header);
        canvas.fill(Rect::new(0, HEADER_H - 1, w, 1), theme.border);
        canvas.text_in(
            &ctx.fonts,
            Rect::new(20, 0, w - 80, HEADER_H),
            15.0,
            true,
            theme.text,
            &self.spec.title,
        );
        hit.close = Rect::new(w - 42, 12, 28, 28);
        if hovered(hit.close) {
            canvas.fill_rounded(hit.close, 6, theme.raised);
        }
        for i in 0..10 {
            canvas.fill(
                Rect::new(hit.close.x + 9 + i, hit.close.y + 9 + i, 2, 2),
                theme.text,
            );
            canvas.fill(
                Rect::new(hit.close.x + 18 - i, hit.close.y + 9 + i, 2, 2),
                theme.text,
            );
        }

        // ── Toolbar: up, breadcrumbs, search ──────────────────────────────
        let toolbar_y = HEADER_H;
        hit.up = Rect::new(14, toolbar_y + 7, 30, 30);
        widgets::button(canvas, ctx, hit.up, "", Emphasis::Flat, hovered(hit.up), true);
        widgets::chevron(
            canvas,
            (hit.up.x + 12, hit.up.y + 9),
            6,
            false,
            if self.dir.parent().is_some() {
                theme.text
            } else {
                theme.dim
            },
        );

        let search_w = 220;
        let crumb_limit = w - search_w - 30;
        let mut chain: Vec<PathBuf> = Vec::new();
        let mut cursor = Some(self.dir.as_path());
        while let Some(path) = cursor {
            chain.push(path.to_path_buf());
            cursor = path.parent();
        }
        chain.reverse();
        let mut labels: Vec<(String, PathBuf)> = chain
            .into_iter()
            .map(|path| {
                let label = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map_or_else(|| "/".to_string(), ToString::to_string);
                (label, path)
            })
            .collect();
        // Drop leading crumbs until the trail fits: the tail (where you are)
        // is the part worth keeping.
        let mut crumb_x = hit.up.right() + 10;
        loop {
            let total: i32 = labels
                .iter()
                .map(|(l, _)| canvas.measure(&ctx.fonts, l, 13.0, false) as i32 + 22)
                .sum();
            if labels.len() <= 1 || crumb_x + total <= crumb_limit {
                break;
            }
            labels.remove(0);
        }
        for (label, path) in labels {
            let width = canvas.measure(&ctx.fonts, &label, 13.0, false) as i32 + 22;
            let rect = Rect::new(crumb_x, toolbar_y + 7, width, 30);
            if rect.right() > crumb_limit {
                break;
            }
            let current = path == self.dir;
            widgets::button(
                canvas,
                ctx,
                rect,
                &label,
                if current {
                    Emphasis::Normal
                } else {
                    Emphasis::Flat
                },
                hovered(rect),
                true,
            );
            hit.crumbs.push((rect, path));
            crumb_x = rect.right() + 2;
        }

        self.search.render(
            canvas,
            ctx,
            Rect::new(w - search_w - 14, toolbar_y + 7, search_w, 30),
            "Search",
        );

        // ── Sidebar ───────────────────────────────────────────────────────
        let body_y = toolbar_y + TOOLBAR_H;
        let footer_h = self.footer_height();
        let body_h = h - body_y - footer_h;
        canvas.fill(Rect::new(0, body_y, SIDEBAR_W, body_h), theme.header);
        canvas.fill(Rect::new(SIDEBAR_W - 1, body_y, 1, body_h), theme.border);
        let mut place_y = body_y + 8;
        for (label, path) in &self.places {
            if place_y + 28 > body_y + body_h {
                break;
            }
            let rect = Rect::new(8, place_y, SIDEBAR_W - 16, 28);
            let current = *path == self.dir;
            if current || hovered(rect) {
                canvas.fill_rounded(
                    rect,
                    6,
                    if current {
                        theme.accent.with_alpha(0.22)
                    } else {
                        theme.raised
                    },
                );
            }
            widgets::folder_icon(
                canvas,
                Rect::new(rect.x + 8, rect.y + 7, 14, 14),
                if current { theme.accent } else { theme.dim },
            );
            canvas.text_in(
                &ctx.fonts,
                Rect::new(rect.x + 30, rect.y, rect.w - 38, rect.h),
                13.0,
                false,
                theme.text,
                label,
            );
            hit.places.push((rect, path.clone()));
            place_y += 30;
        }

        // ── Column headers ────────────────────────────────────────────────
        let list_x = SIDEBAR_W;
        let list_w = w - SIDEBAR_W;
        canvas.fill(Rect::new(list_x, body_y, list_w, COLS_H), theme.header);
        canvas.fill(
            Rect::new(list_x, body_y + COLS_H - 1, list_w, 1),
            theme.border,
        );
        let size_x = list_x + list_w - 250;
        let time_x = list_x + list_w - 160;
        hit.headers = [
            Rect::new(list_x, body_y, size_x - list_x, COLS_H),
            Rect::new(size_x, body_y, time_x - size_x, COLS_H),
            Rect::new(time_x, body_y, list_w - (time_x - list_x) - 12, COLS_H),
        ];
        for (index, (label, kind)) in [
            ("Name", SortBy::Name),
            ("Size", SortBy::Size),
            ("Modified", SortBy::Modified),
        ]
        .into_iter()
        .enumerate()
        {
            let rect = hit.headers[index];
            let active = self.sort == kind;
            let label = if active {
                format!("{label} {}", if self.sort_desc { "▾" } else { "▴" })
            } else {
                label.to_string()
            };
            canvas.text_in(
                &ctx.fonts,
                Rect::new(rect.x + 12, rect.y, rect.w - 16, rect.h),
                12.0,
                active,
                if active { theme.text } else { theme.dim },
                &label,
            );
        }

        // ── Rows ──────────────────────────────────────────────────────────
        let rows = Rect::new(list_x, body_y + COLS_H, list_w, body_h - COLS_H);
        hit.rows = rows;
        let content_h = self.visible.len() as f32 * ROW_H as f32;
        self.scroll.offset = self
            .scroll
            .offset
            .clamp(0.0, (content_h - rows.h as f32).max(0.0));
        canvas.clipped(rows, |c| {
            if let Some(error) = &self.error {
                c.text_in(
                    &ctx.fonts,
                    Rect::new(rows.x + 16, rows.y + 12, rows.w - 32, 24),
                    13.0,
                    false,
                    theme.danger,
                    error,
                );
                return;
            }
            if self.visible.is_empty() {
                c.text_in(
                    &ctx.fonts,
                    Rect::new(rows.x + 16, rows.y + 12, rows.w - 32, 24),
                    13.0,
                    false,
                    theme.dim,
                    if self.search.text.is_empty() {
                        "Empty folder"
                    } else {
                        "Nothing matches"
                    },
                );
                return;
            }
            // Only the rows on screen are drawn: a directory with 50 000
            // entries costs the same as one with 20.
            let first = (self.scroll.offset / ROW_H as f32).floor().max(0.0) as usize;
            let last = (first + (rows.h / ROW_H) as usize + 2).min(self.visible.len());
            for index in first..last {
                let entry_index = self.visible[index];
                let entry = &self.entries[entry_index];
                let y = rows.y + (index as f32 * ROW_H as f32 - self.scroll.offset) as i32;
                let row = Rect::new(rows.x, y, rows.w, ROW_H);
                let selected = self.selected.contains(&entry_index);
                if selected {
                    c.fill(row, theme.accent.with_alpha(0.28));
                } else if hover.is_some_and(|(hx, hy)| row.contains(hx, hy)) {
                    c.fill(row, theme.raised);
                }
                if index == self.cursor && !selected {
                    c.stroke_rounded(row.inset(1), 4, 1, theme.border);
                }
                let icon = Rect::new(row.x + 14, row.y + 8, 14, 14);
                if entry.is_dir {
                    widgets::folder_icon(c, icon, theme.accent);
                } else {
                    widgets::file_icon(c, icon, theme.dim);
                }
                c.text_in(
                    &ctx.fonts,
                    Rect::new(row.x + 38, row.y, size_x - row.x - 50, row.h),
                    13.0,
                    false,
                    theme.text,
                    &entry.name,
                );
                if !entry.is_dir {
                    c.text_in(
                        &ctx.fonts,
                        Rect::new(size_x, row.y, 80, row.h),
                        12.0,
                        false,
                        theme.dim,
                        &format_size(entry.size),
                    );
                }
                c.text_in(
                    &ctx.fonts,
                    Rect::new(time_x, row.y, 150, row.h),
                    12.0,
                    false,
                    theme.dim,
                    &format_time(entry.mtime),
                );
            }
        });
        self.scroll.render(canvas, theme, rows, content_h);

        // ── Footer ────────────────────────────────────────────────────────
        let footer_y = h - footer_h;
        canvas.fill(Rect::new(0, footer_y, w, footer_h), theme.header);
        canvas.fill(Rect::new(0, footer_y, w, 1), theme.border);
        let mut y = footer_y + 12;

        if matches!(self.spec.kind, Kind::Save) {
            canvas.text_in(
                &ctx.fonts,
                Rect::new(20, y, 60, 30),
                13.0,
                false,
                theme.dim,
                "Name",
            );
            hit.name_field = Rect::new(70, y, w - 90, 30);
            self.name.render(canvas, ctx, hit.name_field, "File name");
            y += 42;
        }
        if let Kind::SaveFolder { files } = &self.spec.kind {
            canvas.text_in(
                &ctx.fonts,
                Rect::new(20, y, w - 40, 24),
                12.0,
                false,
                theme.dim,
                &format!(
                    "Saving {} file{} into this folder",
                    files.len(),
                    if files.len() == 1 { "" } else { "s" }
                ),
            );
            y += 28;
        }
        if !self.choices.is_empty() {
            let mut x = 20;
            for choice in &self.choices {
                let label = format!(
                    "{}: {}",
                    choice.label,
                    choice
                        .options
                        .iter()
                        .find(|(id, _)| *id == choice.selected)
                        .map_or(choice.selected.as_str(), |(_, l)| l.as_str())
                );
                let width = (canvas.measure(&ctx.fonts, &label, 13.0, false) as i32 + 38).min(w / 3);
                let rect = Rect::new(x, y, width, 30);
                widgets::button(canvas, ctx, rect, &label, Emphasis::Normal, hovered(rect), true);
                widgets::chevron(canvas, (rect.right() - 18, rect.y + 12), 4, false, theme.dim);
                hit.choice_combos.push(rect);
                x = rect.right() + 10;
            }
        }

        // Bottom row: hidden toggle, filter combo, cancel/accept.
        let bottom_y = h - 44;
        hit.hidden_toggle = widgets::checkbox(
            canvas,
            ctx,
            (20, bottom_y + 6),
            "Show hidden",
            self.show_hidden,
            hovered(self.hit.hidden_toggle),
        );
        if !self.spec.filters.is_empty() {
            let label = self
                .filter
                .and_then(|i| self.spec.filters.get(i))
                .map_or("All files", |f| f.name.as_str());
            let width = canvas.measure(&ctx.fonts, label, 13.0, false) as i32 + 40;
            let rect = Rect::new(hit.hidden_toggle.right() + 24, bottom_y, width, 30);
            widgets::button(canvas, ctx, rect, label, Emphasis::Normal, hovered(rect), true);
            widgets::chevron(canvas, (rect.right() - 18, rect.y + 12), 4, false, theme.dim);
            hit.filter_combo = rect;
        }
        let accept_label = self.accept_label();
        let accept_w = (canvas.measure(&ctx.fonts, &accept_label, 13.0, false) as i32 + 44).max(96);
        hit.accept = Rect::new(w - accept_w - 20, bottom_y, accept_w, 30);
        hit.cancel = Rect::new(hit.accept.x - 110, bottom_y, 100, 30);
        widgets::button(
            canvas,
            ctx,
            hit.cancel,
            "Cancel",
            Emphasis::Normal,
            hovered(hit.cancel),
            true,
        );
        widgets::button(
            canvas,
            ctx,
            hit.accept,
            &accept_label,
            Emphasis::Primary,
            hovered(hit.accept),
            true,
        );

        // ── Combo popup, drawn last so it lands over everything ───────────
        if let Some(popup) = self.popup {
            let (anchor, items): (Rect, Vec<String>) = match popup {
                Popup::Filter => (
                    hit.filter_combo,
                    self.spec.filters.iter().map(|f| f.name.clone()).collect(),
                ),
                Popup::Choice(index) => (
                    hit.choice_combos.get(index).copied().unwrap_or_default(),
                    self.choices.get(index).map_or_else(Vec::new, |c| {
                        c.options.iter().map(|(_, l)| l.clone()).collect()
                    }),
                ),
            };
            if !items.is_empty() {
                let item_h = 28;
                let popup_h = items.len() as i32 * item_h + 8;
                let rect = Rect::new(
                    anchor.x,
                    (anchor.y - popup_h - 4).max(HEADER_H + 4),
                    anchor.w.max(180),
                    popup_h,
                );
                canvas.fill_rounded(rect, 8, theme.surface);
                canvas.stroke_rounded(rect, 8, 1, theme.border);
                for (index, label) in items.iter().enumerate() {
                    let item = Rect::new(
                        rect.x + 4,
                        rect.y + 4 + index as i32 * item_h,
                        rect.w - 8,
                        item_h,
                    );
                    if hovered(item) {
                        canvas.fill_rounded(item, 6, theme.raised);
                    }
                    canvas.text_in(
                        &ctx.fonts,
                        Rect::new(item.x + 10, item.y, item.w - 20, item.h),
                        13.0,
                        false,
                        theme.text,
                        label,
                    );
                    hit.popup.push((item, index));
                }
            }
        }

        self.hit = hit;
    }

    #[allow(
        clippy::too_many_lines,
        reason = "an event router for one dialog; splitting it would hide the ordering that matters (popup first, then chrome, then the list)"
    )]
    fn input(&mut self, _surface: usize, event: &Input, _ctx: &Ctx) -> Flow {
        match event {
            Input::Motion { x, y } => {
                self.hover = Some((*x, *y));
                Flow::Redraw
            }
            Input::Leave => {
                self.hover = None;
                Flow::Redraw
            }
            Input::Scroll { dy, .. } => {
                self.scroll.by(
                    *dy,
                    self.visible.len() as f32 * ROW_H as f32,
                    self.hit.rows.h as f32,
                );
                Flow::Redraw
            }
            Input::Press { x, y, mods } => self.press(*x, *y, *mods),
            Input::Release => Flow::Idle,
            Input::Key { key, mods } => self.key(key, *mods),
        }
    }
}

impl FileChooser {
    #[allow(
        clippy::too_many_lines,
        reason = "flat click routing; the sequence is the logic"
    )]
    fn press(&mut self, x: f64, y: f64, mods: Mods) -> Flow {
        // An open popup swallows every click, wherever it lands.
        if let Some(popup) = self.popup.take() {
            let chosen = self
                .hit
                .popup
                .iter()
                .find(|(rect, _)| rect.contains(x, y))
                .map(|(_, index)| *index);
            if let Some(index) = chosen {
                match popup {
                    Popup::Filter => {
                        self.filter = Some(index);
                        self.refilter();
                    }
                    Popup::Choice(combo) => {
                        if let Some(choice) = self.choices.get_mut(combo)
                            && let Some((id, _)) = choice.options.get(index)
                        {
                            choice.selected = id.clone();
                        }
                    }
                }
            }
            return Flow::Redraw;
        }
        if self.hit.close.contains(x, y) || self.hit.cancel.contains(x, y) {
            return Flow::Done;
        }
        if self.hit.accept.contains(x, y) {
            return self.finish();
        }
        if self.hit.up.contains(x, y) {
            if let Some(parent) = self.dir.parent().map(Path::to_path_buf) {
                self.navigate(parent);
            }
            return Flow::Redraw;
        }
        if self.hit.hidden_toggle.contains(x, y) {
            self.show_hidden = !self.show_hidden;
            self.refilter();
            return Flow::Redraw;
        }
        if !self.hit.filter_combo.is_empty() && self.hit.filter_combo.contains(x, y) {
            self.popup = Some(Popup::Filter);
            return Flow::Redraw;
        }
        if let Some(index) = self
            .hit
            .choice_combos
            .iter()
            .position(|rect| rect.contains(x, y))
        {
            self.popup = Some(Popup::Choice(index));
            return Flow::Redraw;
        }
        if let Some(path) = self
            .hit
            .crumbs
            .iter()
            .find(|(rect, _)| rect.contains(x, y))
            .map(|(_, p)| p.clone())
        {
            self.navigate(path);
            return Flow::Redraw;
        }
        if let Some(path) = self
            .hit
            .places
            .iter()
            .find(|(rect, _)| rect.contains(x, y))
            .map(|(_, p)| p.clone())
        {
            self.navigate(path);
            return Flow::Redraw;
        }
        for (index, kind) in [SortBy::Name, SortBy::Size, SortBy::Modified]
            .into_iter()
            .enumerate()
        {
            if self.hit.headers[index].contains(x, y) {
                if self.sort == kind {
                    self.sort_desc = !self.sort_desc;
                } else {
                    self.sort = kind;
                    self.sort_desc = false;
                }
                self.resort();
                self.refilter();
                return Flow::Redraw;
            }
        }
        if !self.hit.name_field.is_empty() && self.hit.name_field.contains(x, y) {
            self.naming = true;
            self.name.focus();
            self.search.focused = false;
            return Flow::Redraw;
        }
        if self.hit.rows.contains(x, y) {
            self.naming = false;
            self.name.focused = false;
            self.search.focused = true;
            let index = ((y - f64::from(self.hit.rows.y) + f64::from(self.scroll.offset))
                / f64::from(ROW_H)) as usize;
            if index >= self.visible.len() {
                return Flow::Redraw;
            }
            // 400 ms double-click window — close enough to every toolkit's
            // default that muscle memory carries over.
            let now = Instant::now();
            let double = self
                .last_click
                .is_some_and(|(last, at)| last == index && now.duration_since(at).as_millis() < 400);
            self.last_click = Some((index, now));
            if double {
                return self.activate(index);
            }
            self.select(index, mods);
            return Flow::Redraw;
        }
        Flow::Redraw
    }

    fn key(&mut self, key: &Key, mods: Mods) -> Flow {
        if self.popup.is_some() {
            if key == &Key::Escape {
                self.popup = None;
                return Flow::Redraw;
            }
            return Flow::Idle;
        }
        match key {
            Key::Escape => return Flow::Done,
            Key::Enter => {
                if self.naming || self.visible.is_empty() {
                    return self.finish();
                }
                return self.activate(self.cursor);
            }
            Key::Tab if matches!(self.spec.kind, Kind::Save) => {
                self.naming = !self.naming;
                self.name.focused = self.naming;
                self.search.focused = !self.naming;
                return Flow::Redraw;
            }
            Key::Char('h') if mods.ctrl => {
                self.show_hidden = !self.show_hidden;
                self.refilter();
                return Flow::Redraw;
            }
            Key::Char('l') if mods.ctrl => {
                // Ctrl+L: type a path. The search field doubles as the
                // location bar — a `/` in it makes it one.
                self.naming = false;
                self.name.focused = false;
                self.search.focus();
                self.search.set(format!("{}/", self.dir.display()));
                return Flow::Redraw;
            }
            Key::Char('a') if mods.ctrl => {
                if matches!(self.spec.kind, Kind::Open { multiple: true, .. }) {
                    self.selected = self.visible.iter().copied().collect();
                    return Flow::Redraw;
                }
            }
            Key::Up | Key::Down | Key::PageUp | Key::PageDown | Key::Home | Key::End
                if !self.naming =>
            {
                if self.visible.is_empty() {
                    return Flow::Idle;
                }
                let page = (self.hit.rows.h / ROW_H).max(1).unsigned_abs() as usize;
                let last = self.visible.len() - 1;
                self.cursor = match key {
                    Key::Up => self.cursor.saturating_sub(1),
                    Key::Down => (self.cursor + 1).min(last),
                    Key::PageUp => self.cursor.saturating_sub(page),
                    Key::PageDown => (self.cursor + page).min(last),
                    Key::Home => 0,
                    _ => last,
                };
                self.select(self.cursor, mods);
                self.scroll
                    .reveal(self.cursor, ROW_H as f32, self.hit.rows.h as f32);
                return Flow::Redraw;
            }
            Key::Backspace if !self.naming && self.search.text.is_empty() => {
                if let Some(parent) = self.dir.parent().map(Path::to_path_buf) {
                    self.navigate(parent);
                }
                return Flow::Redraw;
            }
            _ => {}
        }
        // Anything else goes to whichever field has focus.
        if self.naming {
            if self.name.key(key, mods) {
                return Flow::Redraw;
            }
            return Flow::Idle;
        }
        if self.search.key(key, mods) {
            // A path typed into the search box navigates as soon as it
            // resolves to a directory — which is what Ctrl+L sets up.
            let typed = expand(self.search.text.trim());
            if self.search.text.contains('/') && typed.is_dir() {
                self.navigate(typed);
            } else {
                self.refilter();
            }
            return Flow::Redraw;
        }
        Flow::Idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs_match_the_way_filters_expect() {
        assert!(glob_matches("*.png", "shot.png"));
        assert!(!glob_matches("*.png", "shot.png.bak"));
        assert!(glob_matches("*.tar.*", "src.tar.zst"));
        assert!(glob_matches("*", "anything"));
        assert!(glob_matches("photo?.jpg", "photo1.jpg"));
    }

    #[test]
    fn mime_filters_resolve_by_extension_and_wildcard() {
        assert!(mime_matches("image/png", "a.png"));
        assert!(mime_matches("image/*", "a.jpeg"));
        assert!(!mime_matches("image/*", "a.mp3"));
        // An unknown extension matches nothing rather than everything.
        assert!(!mime_matches("image/*", "a.qqq"));
    }

    #[test]
    fn sizes_are_binary_and_short() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(2048), "2.0 KiB");
        assert_eq!(format_size(20 * 1024 * 1024), "20 MiB");
    }

    #[test]
    fn epoch_formats_as_utc_iso() {
        let stamp = format_time(Some(UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000)));
        assert_eq!(stamp, "2023-11-14 22:13");
    }
}
