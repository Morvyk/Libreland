//! Compositor configuration.
//!
//! One place that holds every runtime setting the user can influence.
//! Populated with [`Default`] at startup and then optionally
//! overridden by a Lua file at `$XDG_CONFIG_HOME/libreland/config.lua`
//! (loaded by [`Config::load_or_default`]). Every field here is
//! applied at runtime: monitors (`mode`/`position`/`scale`/`primary`)
//! by [`crate::drm`] + [`crate::render`], `repeat_rate`/`repeat_delay`
//! and `keyboard_layout` by the seat in [`crate::wayland`], the input
//! accel/focus settings per pointer device, and the rest by the
//! layout / renderer.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

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
    pub layout: LayoutConfig,
    pub border: BorderConfig,
    /// Environment variables to export into the compositor's own
    /// process before spawning any children, so every client we
    /// launch (startup commands, `spawn` binds, ad-hoc shells in
    /// the same session) inherits them. Typical use is theming
    /// hints like `XCURSOR_THEME` or `QT_QPA_PLATFORMTHEME`. Sorted
    /// by name for deterministic application/logging. Applied once
    /// at startup; changing them needs a restart.
    pub env: Vec<(String, String)>,
    /// Commands to spawn as children once the Wayland socket is
    /// listening. Each entry is whitespace-split into program +
    /// args; needs a shell wrapper (`"sh -c '…'"`) for shell
    /// features. Children inherit the compositor's environment
    /// (notably `$WAYLAND_DISPLAY`).
    pub startup: Vec<String>,
    /// Run `xwayland-satellite` at startup so X11 apps work (rootless
    /// Xwayland as a normal Wayland client). When `true` (default) and
    /// the binary is installed, the compositor picks a free X display,
    /// launches the satellite on it, and exports `$DISPLAY` so X
    /// clients connect. The satellite scales X apps itself via
    /// `wp_fractional_scale` + `wp_viewporter`. Toggling needs a
    /// restart (the satellite is spawned once at launch).
    pub xwayland: bool,
}

#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// Pixels of empty space between the tile area and each edge
    /// of the layout's bounds. Wallpaper shows through the gap.
    /// Default `8`.
    pub gaps_outer: i32,
    /// Pixels of empty space between adjacent tile cells.
    /// Centred on each split divider — each cell gives up
    /// `inner / 2` on the dividing side. Default `3`.
    pub gaps_inner: i32,
}

#[derive(Debug, Clone, Default, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
pub struct OutputConfig {
    /// Mode override: `Some((width, height, refresh_mHz))` to force
    /// a specific mode; `None` uses the connector's `PREFERRED` flag
    /// from the EDID (or the first advertised mode).
    pub mode: Option<(u32, u32, u32)>,
    /// Top-left position in the virtual layout, in *logical* pixels
    /// (after applying this output's `scale`). `None` means "auto",
    /// which is the left-to-right enumeration order at `y = 0`.
    pub position: Option<(i32, i32)>,
    /// Fractional scale factor. `1.0` is unscaled; `1.5`, `2.0`
    /// typical for `HiDPI`. Applied to the renderer's output rect
    /// (physical = logical * scale) and advertised to clients via
    /// both `wl_output.scale` (rounded up to the nearest integer
    /// for legacy clients) and `wp_fractional_scale_manager_v1`
    /// (full fractional value, for clients that support it).
    pub scale: f64,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            mode: None,
            position: None,
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
    /// Which surface receives keyboard focus when the pointer moves
    /// or a button is pressed. [`FocusModel::Hover`] is the default
    /// (focus follows the surface under the cursor on every motion
    /// event); [`FocusModel::Click`] only refocuses on press.
    pub focus_model: FocusModel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelProfile {
    /// 1:1 device-to-cursor motion, no acceleration ramp.
    Flat,
    /// libinput's adaptive curve (the system default).
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusModel {
    /// Keyboard focus follows the surface under the cursor on every
    /// pointer motion event. The default.
    Hover,
    /// Keyboard focus only changes when the user presses a pointer
    /// button on a surface.
    Click,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Break the calloop event loop and exit the compositor cleanly.
    Exit,
    /// Flip the focused window between tiled and floating. A newly
    /// floating window is centred at ~70 % of its previous tiled
    /// cell; a newly tiled window rejoins the dwindle flow.
    ToggleFloating,
    /// Flip the focused window between fullscreen and normal. A
    /// fullscreen window fills its whole output with no border /
    /// rounded corners and draws on top of everything (incl. panels).
    ToggleFullscreen,
    /// Ask the keyboard-focused toplevel to close, via
    /// `xdg_toplevel.close`. This is a polite request — the client
    /// runs its own close path (e.g. "save before quit?"), so the
    /// window may not disappear immediately, and a client is free to
    /// ignore it. No focused toplevel = silent no-op.
    Close,
    /// Spawn a child process from the configured command string.
    /// The string is whitespace-split into program + args; wrap in
    /// `"sh -c '…'"` for shell features. Inherits the compositor's
    /// environment (notably `$WAYLAND_DISPLAY`). `Arc<str>` so
    /// `Action` is cheap to clone but doesn't need to be `Copy`.
    Spawn(Arc<str>),
}

#[derive(Debug, Clone)]
pub struct MiscConfig {
    /// Background painted before any cursor / surface composition.
    pub wallpaper: Fill,
}

#[derive(Debug, Clone)]
pub struct BorderConfig {
    /// Border width in pixels around every window. `0` disables
    /// borders entirely (no draw, no client-side shrinking).
    pub width: i32,
    /// Fill drawn around the keyboard-focused window.
    pub active: Fill,
    /// Fill drawn around every other window.
    pub inactive: Fill,
    /// Corner radius in pixels. `0` disables; non-zero values
    /// mask each window's four corners with the wallpaper after
    /// the border + surface are drawn, so the visible cell looks
    /// rounded. Per-cell radius is clamped so it never exceeds
    /// half the cell's smaller dimension.
    pub rounded_corners: i32,
}

/// A paint pattern. Used for the wallpaper background and for
/// active / inactive window borders.
#[derive(Debug, Clone)]
pub enum Fill {
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
                focus_model: FocusModel::Hover,
            },
            binds: BindsConfig {
                // Default bindings. The user's `binds` table is
                // merged *on top* of these (see `parse_binds`):
                // unmatched defaults stay, matching triggers get
                // overridden. Order matters because the first match
                // wins, but neither default overlaps the other so
                // the order here is just readability.
                bindings: vec![
                    KeyBinding {
                        mods: keyboard::MOD_SHIFT | keyboard::MOD_SUPER,
                        keysym: Keysym::E,
                        action: Action::Exit,
                    },
                    KeyBinding {
                        mods: keyboard::MOD_SUPER,
                        keysym: Keysym::F,
                        action: Action::ToggleFloating,
                    },
                    KeyBinding {
                        mods: keyboard::MOD_SUPER,
                        keysym: Keysym::F11,
                        action: Action::ToggleFullscreen,
                    },
                    KeyBinding {
                        mods: keyboard::MOD_SUPER,
                        keysym: Keysym::C,
                        action: Action::Close,
                    },
                ],
            },
            misc: MiscConfig {
                wallpaper: Fill::VerticalGradient {
                    top: [0.40, 0.60, 0.90],    // light sky blue
                    bottom: [0.10, 0.20, 0.50], // deep navy
                },
            },
            layout: LayoutConfig {
                gaps_outer: 8,
                gaps_inner: 3,
            },
            border: BorderConfig {
                width: 1,
                // Active border keeps the wallpaper gradient
                // family but brightens it noticeably so the
                // frame stands out from the background.
                active: Fill::VerticalGradient {
                    top: [0.55, 0.80, 1.00],
                    bottom: [0.30, 0.55, 0.95],
                },
                // Inactive: neutral medium grey so unfocused
                // windows visibly de-emphasise without competing
                // with the wallpaper.
                inactive: Fill::Solid([0.30, 0.30, 0.30]),
                // Default radius is in *compositor* pixels and
                // gets multiplied by output scale at render time.
                // 4 is a subtle rounding; bump higher for more
                // pronounced corners. `0` disables. With the
                // fwidth-based shader AA the curve is crisp at
                // any radius >= 1, so the only reason to go
                // bigger is taste.
                rounded_corners: 4,
            },
            env: Vec::new(),
            startup: Vec::new(),
            xwayland: true,
        }
    }
}

impl Config {
    /// Locate `$XDG_CONFIG_HOME/libreland/config.lua` (with the
    /// standard `$XDG_CONFIG_DIRS` fallback), parse it, and return
    /// the resulting `Config`. If no file is found, fall back to the
    /// compiled-in defaults (logged). If a file exists but fails to
    /// parse or validate, log the error prominently and *still* fall
    /// back to defaults rather than aborting — a typo in the config
    /// must never stop the compositor from coming up. The same error
    /// is what live-reload surfaces, so the user can fix it and save
    /// to recover without a restart.
    #[must_use]
    pub fn load_or_default() -> Self {
        let dirs = xdg::BaseDirectories::with_prefix("libreland");
        let Some(path) = dirs.find_config_file("config.lua") else {
            info!(
                "no config.lua found in XDG search path; using defaults (create one to live-load it)"
            );
            return Self::default();
        };
        info!(path = %path.display(), "loading Lua config");
        match Self::load_from_file(&path) {
            Ok(config) => config,
            Err(err) => {
                tracing::error!(
                    path = %path.display(),
                    error = %err,
                    "config failed to load; using defaults (fix the file and save to live-reload)"
                );
                Self::default()
            }
        }
    }

    /// The path live-reload watches and the initial load reads: the
    /// existing `config.lua` in the XDG search path, or — when none
    /// exists yet — its canonical location under `$XDG_CONFIG_HOME`,
    /// so creating it later is picked up. `None` only if XDG can't
    /// resolve a config home at all.
    #[must_use]
    pub fn path() -> Option<std::path::PathBuf> {
        let dirs = xdg::BaseDirectories::with_prefix("libreland");
        dirs.find_config_file("config.lua")
            .or_else(|| dirs.get_config_file("config.lua"))
    }

    /// Read `path`, execute it as a Lua chunk (which sets the
    /// top-level globals our schema reads), and walk the globals
    /// to build a `Config`. Anything the file doesn't set keeps
    /// its `Default` value.
    pub(crate) fn load_from_file(path: &Path) -> Result<Self> {
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
            config.binds = parse_binds(&t, config.binds).context("binds")?;
        }
        if let Some(t) = globals.get::<Option<Table>>("misc")? {
            config.misc = parse_misc(&t, config.misc).context("misc")?;
        }
        if let Some(t) = globals.get::<Option<Table>>("layout")? {
            config.layout = parse_layout(&t, config.layout).context("layout")?;
        }
        if let Some(t) = globals.get::<Option<Table>>("border")? {
            config.border = parse_border(&t, config.border).context("border")?;
        }
        if let Some(t) = globals.get::<Option<Table>>("env")? {
            config.env = parse_env(&t).context("env")?;
        }
        if let Some(t) = globals.get::<Option<Table>>("startup")? {
            config.startup = parse_startup(&t).context("startup")?;
        }
        if let Some(x) = globals.get::<Option<bool>>("xwayland")? {
            config.xwayland = x;
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
        cfg.position = Some((x, y));
    }
    if let Some(scale) = t.get::<Option<f64>>("scale")? {
        if !scale.is_finite() || scale <= 0.0 {
            lua_bail!("scale {scale} invalid; expected a positive finite number");
        }
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
    if let Some(model) = t.get::<Option<String>>("focus_model")? {
        cfg.focus_model = match model.to_lowercase().as_str() {
            "hover" => FocusModel::Hover,
            "click" => FocusModel::Click,
            other => lua_bail!("unknown focus_model {other:?}; expected \"hover\" or \"click\""),
        };
    }
    Ok(cfg)
}

/// Merge the user's `binds` table onto the built-in defaults. A
/// user bind whose trigger (mods + keysym) matches a default
/// *overrides* that default's action; any default the user never
/// touches stays active. So setting `binds` to add `Super+Space`
/// doesn't silently disable `Super+Shift+E` (exit) etc.
fn parse_binds(t: &Table, defaults: BindsConfig) -> mlua::Result<BindsConfig> {
    let mut bindings = defaults.bindings;
    for (i, entry) in t.sequence_values::<Table>().enumerate() {
        let bind_table = entry.with_context(|_| format!("binds[{i}] not a table"))?;
        let bind = parse_bind(&bind_table).with_context(|_| format!("binds[{i}]"))?;
        if let Some(existing) = bindings
            .iter_mut()
            .find(|b| b.mods == bind.mods && b.keysym == bind.keysym)
        {
            existing.action = bind.action;
        } else {
            bindings.push(bind);
        }
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

    let action = parse_action(t)?;

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

fn parse_action(t: &Table) -> mlua::Result<Action> {
    let name: String = t
        .get("action")
        .context("missing or invalid `action` (expected string)")?;
    match name.to_lowercase().as_str() {
        "exit" => Ok(Action::Exit),
        "togglefloating" | "toggle_floating" => Ok(Action::ToggleFloating),
        "togglefullscreen" | "toggle_fullscreen" | "fullscreen" => Ok(Action::ToggleFullscreen),
        "close" | "closewindow" | "close_window" | "kill" => Ok(Action::Close),
        "spawn" => {
            let command: String = t
                .get("command")
                .context("spawn action requires `command` (expected string)")?;
            if command.trim().is_empty() {
                lua_bail!("spawn action `command` is empty");
            }
            Ok(Action::Spawn(Arc::from(command)))
        }
        other => lua_bail!(
            "unknown action {other:?}; supported actions: \"exit\", \"togglefloating\", \"togglefullscreen\", \"close\", \"spawn\""
        ),
    }
}

fn parse_misc(t: &Table, defaults: MiscConfig) -> mlua::Result<MiscConfig> {
    let mut cfg = defaults;
    if let Some(w) = t.get::<Option<Table>>("wallpaper")? {
        cfg.wallpaper = parse_fill(&w).context("wallpaper")?;
    }
    Ok(cfg)
}

fn parse_layout(t: &Table, defaults: LayoutConfig) -> mlua::Result<LayoutConfig> {
    let mut cfg = defaults;
    if let Some(g) = t.get::<Option<i32>>("gaps_outer")? {
        if g < 0 {
            lua_bail!("gaps_outer {g} out of range; expected >= 0");
        }
        cfg.gaps_outer = g;
    }
    if let Some(g) = t.get::<Option<i32>>("gaps_inner")? {
        if g < 0 {
            lua_bail!("gaps_inner {g} out of range; expected >= 0");
        }
        cfg.gaps_inner = g;
    }
    Ok(cfg)
}

fn parse_border(t: &Table, defaults: BorderConfig) -> mlua::Result<BorderConfig> {
    let mut cfg = defaults;
    if let Some(w) = t.get::<Option<i32>>("width")? {
        if w < 0 {
            lua_bail!("border.width {w} out of range; expected >= 0");
        }
        cfg.width = w;
    }
    if let Some(f) = t.get::<Option<Table>>("active")? {
        cfg.active = parse_fill(&f).context("active")?;
    }
    if let Some(f) = t.get::<Option<Table>>("inactive")? {
        cfg.inactive = parse_fill(&f).context("inactive")?;
    }
    if let Some(r) = t.get::<Option<i32>>("rounded_corners")? {
        if r < 0 {
            lua_bail!("border.rounded_corners {r} out of range; expected >= 0");
        }
        cfg.rounded_corners = r;
    }
    Ok(cfg)
}

fn parse_fill(t: &Table) -> mlua::Result<Fill> {
    let kind: String = t
        .get("type")
        .context("missing or invalid `type` (expected \"solid\" or \"vertical_gradient\")")?;
    match kind.to_lowercase().as_str() {
        "solid" => {
            let color: Table = t
                .get("color")
                .context("`color` (expected {r, g, b} array of 3 numbers)")?;
            Ok(Fill::Solid(parse_rgb_triple(&color).context("color")?))
        }
        "vertical_gradient" => {
            let top: Table = t
                .get("top")
                .context("`top` (expected {r, g, b} array of 3 numbers)")?;
            let bottom: Table = t
                .get("bottom")
                .context("`bottom` (expected {r, g, b} array of 3 numbers)")?;
            Ok(Fill::VerticalGradient {
                top: parse_rgb_triple(&top).context("top")?,
                bottom: parse_rgb_triple(&bottom).context("bottom")?,
            })
        }
        other => {
            lua_bail!("unknown fill type {other:?}; expected \"solid\" or \"vertical_gradient\"")
        }
    }
}

fn parse_env(t: &Table) -> mlua::Result<Vec<(String, String)>> {
    let mut vars = Vec::new();
    for pair in t.pairs::<String, String>() {
        let (name, value) =
            pair.context("env entries must be `NAME = \"value\"` (string keys and values)")?;
        // POSIX names can't be empty or contain `=`; both would
        // make `setenv` behave surprisingly. NUL is rejected by
        // `env::set_var` with a panic, so catch it here with a
        // readable message instead.
        if name.is_empty() {
            lua_bail!("env variable name is empty");
        }
        if name.contains('=') || name.contains('\0') {
            lua_bail!("env variable name {name:?} is invalid (must not contain '=' or NUL)");
        }
        if value.contains('\0') {
            lua_bail!("env value for {name:?} contains a NUL byte");
        }
        vars.push((name, value));
    }
    // Deterministic order so logging and application don't depend on
    // Lua's hash-table iteration order.
    vars.sort();
    Ok(vars)
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
    for v in &values {
        if !(0.0..=1.0).contains(v) {
            lua_bail!("RGB component {v} out of range; expected [0.0, 1.0]");
        }
    }
    Ok([values[0], values[1], values[2]])
}
