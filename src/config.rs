//! Compositor configuration.
//!
//! One place that holds every runtime setting the user can influence.
//! Populated with [`Default`] at startup and then optionally
//! overridden by a Lua file at `$XDG_CONFIG_HOME/libreland/config.lua`
//! (loaded by [`Config::load_or_default`]).
//!
//! Not every field is *applied* at runtime yet — `repeat_rate` /
//! `repeat_delay` wait for the Wayland frontend (key-repeat is a
//! client-facing thing), and the per-output `mode` override is held
//! for a follow-up that lets us request specific modes from DRM
//! instead of taking the EDID-preferred one. Lua can set all these
//! today; the values just live in `Config` until their runtime
//! consumer ships.

#![allow(
    dead_code,
    reason = "config schema is intentionally complete ahead of consumption; \
              fields/variants get wired into the runtime as later milestones \
              land (3b: per-output position/scale/mode/primary; Wayland frontend: \
              repeat_rate / repeat_delay. Keeping the schema stable saves Lua \
              users from breaking re-runs when we wire new bits up.)"
)]

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context as _, Result};
use mlua::{ErrorContext as _, Lua, Table};
use tracing::info;
use xkbcommon::xkb;
use xkbcommon::xkb::Keysym;

use crate::keyboard;

/// Build a `mlua::Error::RuntimeError` for a schema mismatch we want
/// to surface to the user with a clear message. mlua's error type
/// isn't `Send + Sync` (it stores an `Arc<dyn Error>` without those
/// bounds), so anyhow can't accept it directly — we stay in
/// `mlua::Result` inside the parsers and only convert at the
/// `load_from_file` boundary.
fn lua_runtime_err(msg: impl Into<String>) -> mlua::Error {
    mlua::Error::RuntimeError(msg.into())
}

macro_rules! lua_bail {
    ($($arg:tt)*) => {
        return Err(crate::config::lua_runtime_err(format!($($arg)*)))
    };
}

/// All runtime configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub monitors: MonitorsConfig,
    pub input: InputConfig,
    pub binds: BindsConfig,
    pub misc: MiscConfig,
    /// Commands to spawn as children once the Wayland socket is
    /// listening. Each entry is whitespace-split into program +
    /// args; needs a shell wrapper (`"sh -c '…'"`) for shell
    /// features. Children inherit the compositor's environment
    /// (notably `$WAYLAND_DISPLAY`).
    pub startup: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct MonitorsConfig {
    /// Per-output settings keyed by connector name (`"DP-1"`,
    /// `"HDMI-A-1"`, etc.). Outputs without an entry get
    /// [`OutputConfig::default`].
    pub outputs: HashMap<String, OutputConfig>,
    /// Connector name of the primary output. `None` means automatic
    /// (first connected enumerated by udev). Read by future code
    /// that needs to pick a "default" output for things like the
    /// initial cursor position or freshly-launched client windows.
    pub primary: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OutputConfig {
    /// Mode override: `Some((width, height, refresh_mHz))` to force
    /// a specific mode; `None` uses the connector's `PREFERRED` flag
    /// from the EDID (or the first advertised mode).
    pub mode: Option<(u32, u32, u32)>,
    /// Top-left position in the virtual layout, in logical pixels.
    pub position: (i32, i32),
    /// Fractional scale factor. `1.0` is unscaled; `1.5`, `2.0`
    /// typical for `HiDPI`. The internal value is the source of truth;
    /// `wp_fractional_scale_manager_v1` will expose it to clients
    /// once the Wayland frontend lands.
    pub scale: f64,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            mode: None,
            position: (0, 0),
            scale: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InputConfig {
    /// Key repeats per second once the delay has elapsed. 25 matches
    /// X11's classic default and feels natural to most users.
    pub repeat_rate: u32,
    /// Milliseconds the user has to hold a key before repeat starts.
    /// 600 is X11's default.
    pub repeat_delay: u32,
    /// xkb `RMLVO` layout field. Empty string defers to
    /// `XKB_DEFAULT_LAYOUT` env var, then libxkbcommon's compile-time
    /// default (`"us"`).
    pub keyboard_layout: String,
    /// libinput pointer acceleration profile.
    pub mouse_accel_profile: AccelProfile,
    /// libinput pointer acceleration speed in `[-1.0, 1.0]`. `0.0`
    /// is the device's neutral position; with [`AccelProfile::Flat`]
    /// this also means "no extra sensitivity multiplier".
    pub mouse_accel_speed: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelProfile {
    /// 1:1 device-to-cursor motion, no acceleration ramp.
    Flat,
    /// libinput's adaptive curve (the system default).
    Adaptive,
}

#[derive(Debug, Clone)]
pub struct BindsConfig {
    /// Keybindings in order. The first one whose keysym + mods
    /// match the press wins.
    pub bindings: Vec<KeyBinding>,
}

#[derive(Debug, Clone)]
pub struct KeyBinding {
    /// Required modifier mask, built from the `MOD_*` constants in
    /// [`crate::keyboard`]. Extras (e.g. `NumLock`) are tolerated.
    pub mods: u32,
    /// Keysym to match against the xkb-resolved keysym for the
    /// press. Shift-induced case shifts are already applied by xkb,
    /// so the Super+Shift+E binding uses `Keysym::E` (uppercase).
    pub keysym: Keysym,
    /// Action to fire on the matching press.
    pub action: Action,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Break the calloop event loop and exit the compositor cleanly.
    Exit,
}

#[derive(Debug, Clone)]
pub struct MiscConfig {
    /// Background painted before any cursor / surface composition.
    pub wallpaper: Wallpaper,
}

#[derive(Debug, Clone)]
pub enum Wallpaper {
    /// Single solid colour. RGB components in `[0.0, 1.0]`.
    Solid([f32; 3]),
    /// Vertical linear gradient from `top` at `y=0` to `bottom` at
    /// `y=output_height`. RGB components in `[0.0, 1.0]`.
    VerticalGradient { top: [f32; 3], bottom: [f32; 3] },
}

impl Default for Config {
    fn default() -> Self {
        Self {
            monitors: MonitorsConfig::default(),
            input: InputConfig {
                repeat_rate: 25,
                repeat_delay: 600,
                keyboard_layout: String::new(),
                mouse_accel_profile: AccelProfile::Flat,
                mouse_accel_speed: 0.0,
            },
            binds: BindsConfig {
                // Single default binding: Super+Shift+E exits.
                // Anything else the user wants comes from Lua.
                bindings: vec![KeyBinding {
                    mods: keyboard::MOD_SHIFT | keyboard::MOD_SUPER,
                    keysym: Keysym::E,
                    action: Action::Exit,
                }],
            },
            misc: MiscConfig {
                wallpaper: Wallpaper::VerticalGradient {
                    top: [0.40, 0.60, 0.90],    // light sky blue
                    bottom: [0.10, 0.20, 0.50], // deep navy
                },
            },
            startup: Vec::new(),
        }
    }
}

impl Config {
    /// Locate `$XDG_CONFIG_HOME/libreland/config.lua` (with the
    /// standard `$XDG_CONFIG_DIRS` fallback), parse it, and return
    /// the resulting `Config`. If no file is found, return the
    /// compiled-in defaults — silent fallback would mask typos in
    /// the filename, so we log explicitly. If a file exists but
    /// fails to parse or validate, return the error so `main` can
    /// abort with a clear startup-time message rather than running
    /// with a half-applied config.
    pub fn load_or_default() -> Result<Self> {
        let dirs = xdg::BaseDirectories::with_prefix("libreland");
        let Some(path) = dirs.find_config_file("config.lua") else {
            info!("no config.lua found in XDG search path; using defaults");
            return Ok(Self::default());
        };
        info!(path = %path.display(), "loading Lua config");
        Self::load_from_file(&path)
            .with_context(|| format!("failed to load Lua config from {}", path.display()))
    }

    /// Read `path`, execute it as a Lua chunk (which sets the
    /// top-level globals our schema reads), and walk the globals
    /// to build a `Config`. Anything the file doesn't set keeps
    /// its `Default` value.
    fn load_from_file(path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path).context("failed to read config file")?;

        let lua = Lua::new();
        lua.load(&source)
            .exec()
            .map_err(|e| anyhow::anyhow!("Lua chunk execution failed: {e}"))?;

        Self::populate_from_globals(&lua.globals())
            .map_err(|e| anyhow::anyhow!("config schema: {e}"))
    }

    fn populate_from_globals(globals: &Table) -> mlua::Result<Self> {
        let mut config = Self::default();

        if let Some(t) = globals.get::<Option<Table>>("monitors")? {
            config.monitors = parse_monitors(&t).context("monitors")?;
        }
        if let Some(t) = globals.get::<Option<Table>>("input")? {
            config.input = parse_input(&t, config.input).context("input")?;
        }
        if let Some(t) = globals.get::<Option<Table>>("binds")? {
            config.binds = parse_binds(&t).context("binds")?;
        }
        if let Some(t) = globals.get::<Option<Table>>("misc")? {
            config.misc = parse_misc(&t, config.misc).context("misc")?;
        }
        if let Some(t) = globals.get::<Option<Table>>("startup")? {
            config.startup = parse_startup(&t).context("startup")?;
        }

        Ok(config)
    }
}

// ---- Lua → schema parsers ---------------------------------------
//
// Each parser takes the corresponding sub-table and returns a fully
// populated struct, using either values supplied by Lua or defaults
// passed in by the caller. Anyhow's `.context(...)` is used liberally
// so a failure deep in the schema produces a breadcrumb-style error
// (`misc → wallpaper → top: expected 3 components, got 2`).

fn parse_monitors(t: &Table) -> mlua::Result<MonitorsConfig> {
    let mut cfg = MonitorsConfig::default();
    if let Some(primary) = t.get::<Option<String>>("primary")? {
        cfg.primary = Some(primary);
    }
    if let Some(outputs) = t.get::<Option<Table>>("outputs")? {
        for pair in outputs.pairs::<String, Table>() {
            let (name, output_table) = pair?;
            let output =
                parse_output(&output_table).with_context(|_| format!("outputs[\"{name}\"]"))?;
            cfg.outputs.insert(name, output);
        }
    }
    Ok(cfg)
}

fn parse_output(t: &Table) -> mlua::Result<OutputConfig> {
    let mut cfg = OutputConfig::default();
    if let Some(mode) = t.get::<Option<Table>>("mode")? {
        let width: u32 = mode.get("width").context("mode.width (expected u32)")?;
        let height: u32 = mode.get("height").context("mode.height (expected u32)")?;
        let refresh: u32 = mode
            .get("refresh_mhz")
            .context("mode.refresh_mhz (expected u32, refresh in milli-Hz)")?;
        cfg.mode = Some((width, height, refresh));
    }
    if let Some(pos) = t.get::<Option<Table>>("position")? {
        let x: i32 = pos.get("x").context("position.x (expected i32)")?;
        let y: i32 = pos.get("y").context("position.y (expected i32)")?;
        cfg.position = (x, y);
    }
    if let Some(scale) = t.get::<Option<f64>>("scale")? {
        cfg.scale = scale;
    }
    Ok(cfg)
}

fn parse_input(t: &Table, defaults: InputConfig) -> mlua::Result<InputConfig> {
    let mut cfg = defaults;
    if let Some(r) = t.get::<Option<u32>>("repeat_rate")? {
        cfg.repeat_rate = r;
    }
    if let Some(d) = t.get::<Option<u32>>("repeat_delay")? {
        cfg.repeat_delay = d;
    }
    if let Some(layout) = t.get::<Option<String>>("keyboard_layout")? {
        cfg.keyboard_layout = layout;
    }
    if let Some(profile) = t.get::<Option<String>>("mouse_accel_profile")? {
        cfg.mouse_accel_profile = match profile.to_lowercase().as_str() {
            "flat" => AccelProfile::Flat,
            "adaptive" => AccelProfile::Adaptive,
            other => lua_bail!(
                "unknown mouse_accel_profile {other:?}; expected \"flat\" or \"adaptive\""
            ),
        };
    }
    if let Some(speed) = t.get::<Option<f64>>("mouse_accel_speed")? {
        if !(-1.0..=1.0).contains(&speed) {
            lua_bail!("mouse_accel_speed {speed} out of range; expected [-1.0, 1.0]");
        }
        cfg.mouse_accel_speed = speed;
    }
    Ok(cfg)
}

fn parse_binds(t: &Table) -> mlua::Result<BindsConfig> {
    let mut bindings = Vec::new();
    for (i, entry) in t.sequence_values::<Table>().enumerate() {
        let bind_table = entry.with_context(|_| format!("binds[{i}] not a table"))?;
        let bind = parse_bind(&bind_table).with_context(|_| format!("binds[{i}]"))?;
        bindings.push(bind);
    }
    Ok(BindsConfig { bindings })
}

fn parse_bind(t: &Table) -> mlua::Result<KeyBinding> {
    let mods_table: Table = t
        .get("mods")
        .context("missing or invalid `mods` (expected array of modifier names)")?;
    let mut mods: u32 = 0;
    for (i, entry) in mods_table.sequence_values::<String>().enumerate() {
        let name = entry.with_context(|_| format!("mods[{i}] not a string"))?;
        mods |= parse_modifier(&name).with_context(|_| format!("mods[{i}]"))?;
    }

    let key_name: String = t
        .get("key")
        .context("missing or invalid `key` (expected xkb keysym name as a string)")?;
    let keysym = xkb::keysym_from_name(&key_name, xkb::KEYSYM_NO_FLAGS);
    if keysym.raw() == 0 {
        lua_bail!(
            "unknown key name {key_name:?}; must be a name xkbcommon's \
             xkb_keysym_from_name accepts (e.g. \"E\", \"Return\", \"F1\", \"space\")"
        );
    }

    let action_name: String = t
        .get("action")
        .context("missing or invalid `action` (expected string)")?;
    let action = parse_action(&action_name)?;

    Ok(KeyBinding {
        mods,
        keysym,
        action,
    })
}

fn parse_modifier(name: &str) -> mlua::Result<u32> {
    match name.to_lowercase().as_str() {
        "shift" => Ok(keyboard::MOD_SHIFT),
        "ctrl" | "control" => Ok(keyboard::MOD_CTRL),
        "alt" | "mod1" => Ok(keyboard::MOD_ALT),
        "super" | "logo" | "mod4" => Ok(keyboard::MOD_SUPER),
        other => lua_bail!(
            "unknown modifier {other:?}; expected one of \
             Shift / Ctrl / Alt / Super (case-insensitive; \
             aliases: Control, Mod1, Logo, Mod4)"
        ),
    }
}

fn parse_action(name: &str) -> mlua::Result<Action> {
    match name.to_lowercase().as_str() {
        "exit" => Ok(Action::Exit),
        other => lua_bail!("unknown action {other:?}; supported actions: \"exit\""),
    }
}

fn parse_misc(t: &Table, defaults: MiscConfig) -> mlua::Result<MiscConfig> {
    let mut cfg = defaults;
    if let Some(w) = t.get::<Option<Table>>("wallpaper")? {
        cfg.wallpaper = parse_wallpaper(&w).context("wallpaper")?;
    }
    Ok(cfg)
}

fn parse_wallpaper(t: &Table) -> mlua::Result<Wallpaper> {
    let kind: String = t
        .get("type")
        .context("missing or invalid `type` (expected \"solid\" or \"vertical_gradient\")")?;
    match kind.to_lowercase().as_str() {
        "solid" => {
            let color: Table = t
                .get("color")
                .context("`color` (expected {r, g, b} array of 3 numbers)")?;
            Ok(Wallpaper::Solid(parse_rgb_triple(&color).context("color")?))
        }
        "vertical_gradient" => {
            let top: Table = t
                .get("top")
                .context("`top` (expected {r, g, b} array of 3 numbers)")?;
            let bottom: Table = t
                .get("bottom")
                .context("`bottom` (expected {r, g, b} array of 3 numbers)")?;
            Ok(Wallpaper::VerticalGradient {
                top: parse_rgb_triple(&top).context("top")?,
                bottom: parse_rgb_triple(&bottom).context("bottom")?,
            })
        }
        other => {
            lua_bail!(
                "unknown wallpaper type {other:?}; expected \"solid\" or \"vertical_gradient\""
            )
        }
    }
}

fn parse_startup(t: &Table) -> mlua::Result<Vec<String>> {
    let mut commands = Vec::new();
    for (i, entry) in t.sequence_values::<String>().enumerate() {
        let cmd = entry.with_context(|_| format!("startup[{i}] not a string"))?;
        commands.push(cmd);
    }
    Ok(commands)
}

fn parse_rgb_triple(t: &Table) -> mlua::Result<[f32; 3]> {
    let values: Vec<f32> = t
        .sequence_values::<f32>()
        .collect::<mlua::Result<_>>()
        .context("RGB components must be numbers")?;
    if values.len() != 3 {
        lua_bail!(
            "RGB triple must have exactly 3 components (got {}); expected {{r, g, b}}",
            values.len()
        );
    }
    Ok([values[0], values[1], values[2]])
}
