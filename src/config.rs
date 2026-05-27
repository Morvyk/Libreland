//! Compositor configuration.
//!
//! One place that holds every runtime setting the user can influence.
//! Populated with sensible defaults at startup; milestone 3c will
//! replace them with values loaded from
//! `$XDG_CONFIG_HOME/libreland/config.lua`.
//!
//! Not every field is wired into the runtime yet — per-output
//! position/scale, keyboard layout, key-repeat rate/delay all live
//! here so the schema is complete, but they only start *doing*
//! something in subsequent milestones (3b for multi-monitor, the
//! Wayland frontend for per-client scale + repeat). Defining them
//! now keeps the Lua loader from needing schema churn later.

#![allow(
    dead_code,
    reason = "config schema is intentionally complete ahead of consumption; \
              fields/variants get wired into the runtime as later milestones \
              land (3b: per-output position/scale/mode/primary; Wayland frontend: \
              repeat_rate / repeat_delay; Lua loader: triggers AccelProfile::Adaptive, \
              Wallpaper::Solid, and the rest. Keeping the schema stable saves Lua \
              users from breaking re-runs when we wire new bits up.)"
)]

use std::collections::HashMap;

use xkbcommon::xkb::Keysym;

use crate::keyboard;

/// All runtime configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub monitors: MonitorsConfig,
    pub input: InputConfig,
    pub binds: BindsConfig,
    pub misc: MiscConfig,
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
        }
    }
}
