//! Desktop entries: finding installed applications, and launching them.
//!
//! Needed by three portals — `AppChooser` (show the user what can open this
//! file), `Email` (hand a `mailto:` URI to whatever handles mail) and
//! `DynamicLauncher` (write a new entry into the user's applications dir) —
//! so the scanning and the `Exec` field-code expansion live here once.
//!
//! This is a deliberately small reading of the Desktop Entry spec: the keys
//! that decide whether to show an entry and how to run it. Nothing here caches
//! across processes; a scan of the XDG application dirs is a few milliseconds
//! and the portal is not a launcher running in a loop.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// One installed application.
#[derive(Clone, Debug)]
pub struct DesktopApp {
    /// The entry's id, i.e. its filename (`org.mozilla.firefox.desktop`).
    pub id: String,
    pub name: String,
    pub comment: String,
    /// Raw `Exec=` line, field codes and all.
    pub exec: String,
    pub terminal: bool,
    pub mime_types: Vec<String>,
}

/// The directories desktop entries live in, most specific first — a user entry
/// shadows a system one with the same id.
fn application_dirs() -> Vec<PathBuf> {
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/".into()));
    let mut dirs = Vec::new();
    dirs.push(
        std::env::var("XDG_DATA_HOME")
            .ok()
            .filter(|s| !s.is_empty())
            .map_or_else(|| home.join(".local/share"), PathBuf::from)
            .join("applications"),
    );
    let data_dirs = std::env::var("XDG_DATA_DIRS")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "/usr/local/share:/usr/share".to_string());
    for dir in data_dirs.split(':').filter(|s| !s.is_empty()) {
        dirs.push(PathBuf::from(dir).join("applications"));
    }
    dirs
}

/// The language tags to try for localized keys, best first (`de_DE`, then
/// `de`). Derived from the same environment every toolkit reads.
fn locales() -> Vec<String> {
    let raw = ["LC_ALL", "LC_MESSAGES", "LANG"]
        .iter()
        .find_map(|var| std::env::var(var).ok())
        .unwrap_or_default();
    // "de_DE.UTF-8@euro" -> ["de_DE", "de"]
    let base = raw.split('.').next().unwrap_or("").split('@').next().unwrap_or("");
    let mut out = Vec::new();
    if !base.is_empty() && base != "C" && base != "POSIX" {
        out.push(base.to_string());
        if let Some((lang, _)) = base.split_once('_') {
            out.push(lang.to_string());
        }
    }
    out
}

/// Parse one `.desktop` file's `[Desktop Entry]` group.
///
/// Returns `None` for entries that shouldn't be offered: hidden ones, ones
/// with no `Exec`, ones whose `TryExec` binary is missing, and ones excluded
/// from this desktop by `NotShowIn` / `OnlyShowIn`.
pub fn parse_entry(path: &Path) -> Option<DesktopApp> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut keys: HashMap<String, String> = HashMap::new();
    let mut in_entry = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_entry = line == "[Desktop Entry]";
            continue;
        }
        if !in_entry || line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            keys.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    if keys.get("Type").map(String::as_str) != Some("Application") {
        return None;
    }
    if keys.get("Hidden").map(String::as_str) == Some("true") {
        return None;
    }
    let exec = keys.get("Exec")?.clone();
    if let Some(try_exec) = keys.get("TryExec")
        && which(try_exec).is_none()
    {
        return None;
    }
    let desktop = std::env::var("XDG_CURRENT_DESKTOP").unwrap_or_default();
    let matches_desktop = |list: &str| {
        list.split(';')
            .filter(|s| !s.is_empty())
            .any(|name| desktop.split(':').any(|d| d.eq_ignore_ascii_case(name)))
    };
    if keys.get("NotShowIn").is_some_and(|l| matches_desktop(l)) {
        return None;
    }
    if let Some(only) = keys.get("OnlyShowIn")
        && !matches_desktop(only)
    {
        return None;
    }

    // Localized Name/Comment win over the plain key.
    let localized = |base: &str| -> Option<String> {
        locales()
            .iter()
            .find_map(|tag| keys.get(&format!("{base}[{tag}]")).cloned())
            .or_else(|| keys.get(base).cloned())
    };

    let fallback_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Application")
        .to_string();

    Some(DesktopApp {
        id: path.file_name()?.to_str()?.to_string(),
        name: localized("Name").unwrap_or(fallback_name),
        comment: localized("Comment").unwrap_or_default(),
        exec,
        terminal: keys.get("Terminal").map(String::as_str) == Some("true"),
        mime_types: keys
            .get("MimeType")
            .map(|m| {
                m.split(';')
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string)
                    .collect()
            })
            .unwrap_or_default(),
    })
}

/// Is `program` runnable — an existing path, or something on `$PATH`?
fn which(program: &str) -> Option<PathBuf> {
    let candidate = Path::new(program);
    if candidate.is_absolute() {
        return candidate.is_file().then(|| candidate.to_path_buf());
    }
    let path = std::env::var("PATH").ok()?;
    path.split(':')
        .map(|dir| Path::new(dir).join(program))
        .find(|p| p.is_file())
}

/// Every installed application, deduplicated by id (first wins, and the dirs
/// are ordered most-specific first) and sorted by display name.
pub fn scan() -> Vec<DesktopApp> {
    let mut seen: HashMap<String, DesktopApp> = HashMap::new();
    for dir in application_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("desktop") {
                continue;
            }
            if let Some(app) = parse_entry(&path)
                && !seen.contains_key(&app.id)
            {
                seen.insert(app.id.clone(), app);
            }
        }
    }
    let mut apps: Vec<DesktopApp> = seen.into_values().collect();
    apps.sort_by_key(|a| a.name.to_lowercase());
    apps
}

/// Look one application up by desktop id (with or without the `.desktop`).
pub fn find(id: &str) -> Option<DesktopApp> {
    let file = if id.ends_with(".desktop") {
        id.to_string()
    } else {
        format!("{id}.desktop")
    };
    application_dirs()
        .iter()
        .map(|dir| dir.join(&file))
        .find(|p| p.is_file())
        .and_then(|p| parse_entry(&p))
}

/// Split an `Exec` line into argv, honouring the spec's quoting, and
/// substituting the field codes with `args`.
///
/// `%f`/`%u` take one argument, `%F`/`%U` take all of them, and the rest
/// (`%i`, `%c`, `%k`, deprecated `%d`…) are dropped — an unexpanded field code
/// reaching the child is the classic "app opens with a literal %U in argv" bug.
pub fn expand_exec(exec: &str, args: &[String]) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut chars = exec.chars().peekable();
    let mut quoted = false;
    let mut has_token = false;

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                quoted = !quoted;
                has_token = true;
            }
            '\\' if quoted => {
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            ' ' | '\t' if !quoted => {
                if has_token || !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                    has_token = false;
                }
            }
            '%' => match chars.next() {
                Some('f' | 'u') => {
                    if let Some(first) = args.first() {
                        current.push_str(first);
                        has_token = true;
                    }
                }
                Some('F' | 'U') => {
                    // Multi-argument codes stand alone in practice; flush what
                    // we have and append each argument as its own word.
                    if has_token || !current.is_empty() {
                        words.push(std::mem::take(&mut current));
                        has_token = false;
                    }
                    words.extend(args.iter().cloned());
                }
                Some('%') => current.push('%'),
                // Any other code (icon, translated name, entry path) expands
                // to nothing.
                Some(_) | None => {}
            },
            _ => {
                current.push(ch);
                has_token = true;
            }
        }
    }
    if has_token || !current.is_empty() {
        words.push(current);
    }
    words.retain(|a| !a.is_empty());
    words
}

/// Launch an application with optional arguments (paths or URIs).
///
/// Detached: the child gets its own session so it outlives this short-lived
/// portal process, and its output goes nowhere rather than into the portal's
/// journal.
pub fn launch(app: &DesktopApp, args: &[String]) -> anyhow::Result<()> {
    let mut command = expand_exec(&app.exec, args);
    if command.is_empty() {
        anyhow::bail!("{} has an empty Exec line", app.id);
    }
    if app.terminal {
        // Terminal=true entries need a terminal emulator. $TERMINAL is what
        // people set for exactly this; fall back to the common ones.
        let terminal = std::env::var("TERMINAL")
            .ok()
            .filter(|t| which(t).is_some())
            .or_else(|| {
                ["foot", "alacritty", "kitty", "wezterm", "xterm"]
                    .into_iter()
                    .find(|t| which(t).is_some())
                    .map(ToString::to_string)
            })
            .unwrap_or_else(|| "xterm".to_string());
        let mut wrapped = vec![terminal, "-e".to_string()];
        wrapped.append(&mut command);
        command = wrapped;
    }
    let (program, rest) = command.split_first().expect("checked non-empty above");
    Command::new(program)
        .args(rest)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    Ok(())
}

/// The desktop id registered as the default handler for `mime`, from
/// `mimeapps.list`. Used to hand off `mailto:` URIs without shelling out to
/// `xdg-open`.
pub fn default_handler(mime: &str) -> Option<DesktopApp> {
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/".into()));
    let config = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map_or_else(|| home.join(".config"), PathBuf::from);
    let mut candidates = vec![config.join("mimeapps.list")];
    for dir in application_dirs() {
        candidates.push(dir.join("mimeapps.list"));
        candidates.push(dir.join("defaults.list"));
    }
    for file in candidates {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        let mut in_defaults = false;
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with('[') {
                in_defaults = line == "[Default Applications]";
                continue;
            }
            if !in_defaults {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            if key.trim() != mime {
                continue;
            }
            // The value is a `;`-separated preference list.
            for id in value.split(';').filter(|s| !s.is_empty()) {
                if let Some(app) = find(id.trim()) {
                    return Some(app);
                }
            }
        }
    }
    // No explicit default: fall back to anything that claims the type.
    scan().into_iter().find(|a| a.mime_types.iter().any(|m| m == mime))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_field_codes_expand() {
        let args = vec!["/tmp/a b.txt".to_string(), "/tmp/c.txt".to_string()];
        assert_eq!(
            expand_exec("gedit %U", &args),
            vec!["gedit", "/tmp/a b.txt", "/tmp/c.txt"]
        );
        assert_eq!(
            expand_exec("gedit %f", &args),
            vec!["gedit", "/tmp/a b.txt"]
        );
        // Unused codes vanish rather than reaching the child.
        assert_eq!(expand_exec("app --icon %i %c", &[]), vec!["app", "--icon"]);
    }

    #[test]
    fn exec_quoting_survives_spaces() {
        assert_eq!(
            expand_exec(r#""/opt/My App/run" --flag"#, &[]),
            vec!["/opt/My App/run", "--flag"]
        );
    }

    #[test]
    fn literal_percent_is_preserved() {
        assert_eq!(expand_exec("app 100%%", &[]), vec!["app", "100%"]);
    }
}
