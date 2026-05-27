//! xkbcommon-based keyboard handling.
//!
//! Replaces the milestone-1 approach of matching raw evdev scancode
//! constants (`KEY_E`, `KEY_LEFTSHIFT`, …) with a real xkb keymap +
//! state machine. Each libinput key event is fed to xkbcommon, which
//! tracks modifier latching/locking and returns the layout-aware
//! keysym at the moment of press / release. Hotkeys are matched on
//! `Keysym` + a bitmask of effective modifiers — layout-correct on
//! any keyboard the user has configured.
//!
//! Keymap source: xkbcommon's standard `RMLVO` (rules / model /
//! layout / variant / options). Passing empty strings means "use
//! `XKB_DEFAULT_*` env vars, or the libxkbcommon system defaults
//! (`evdev` / `pc105` / `us` / `` / ``) if those aren't set" — which
//! gives us the user's existing keyboard config out of the box.
//! When the Lua config layer lands, the user can override these
//! explicitly.

use anyhow::{Context as _, Result};
use smithay::backend::input::Keycode;
use xkbcommon::xkb;
pub use xkbcommon::xkb::Keysym;

/// `Shift` (left or right) is currently held.
pub const MOD_SHIFT: u32 = 1 << 0;
/// `Control` (left or right) is currently held.
pub const MOD_CTRL: u32 = 1 << 1;
/// `Alt`/`Mod1` is currently held.
pub const MOD_ALT: u32 = 1 << 2;
/// `Super`/`Logo`/`Mod4` (the Windows / command key) is currently held.
pub const MOD_SUPER: u32 = 1 << 3;

/// Outcome of feeding a single libinput key event through xkbcommon:
/// the layout-aware keysym at this moment (with modifier effects
/// applied — `Shift+e` becomes `Keysym::E`), and a bitmask of the
/// effective modifiers.
pub struct KeyResult {
    pub keysym: Keysym,
    pub mods: u32,
}

impl KeyResult {
    /// True if every modifier in `required` is currently held. Extra
    /// modifiers (e.g. `NumLock`) don't disqualify the match — this
    /// is the conventional "hotkey wants these mods, but tolerates
    /// extras" semantics.
    pub fn has_all_mods(&self, required: u32) -> bool {
        self.mods & required == required
    }
}

/// xkb keymap + state. The keymap is held via the C library's
/// internal refcount through `State` (`xkb_state_new` bumps the
/// keymap's ref), so we don't need to store it explicitly.
pub struct Keyboard {
    state: xkb::State,
}

impl Keyboard {
    pub fn new() -> Result<Self> {
        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        // Empty RMLVO strings → xkbcommon consults `XKB_DEFAULT_*`
        // env vars and falls back to its compile-time defaults
        // (`evdev` / `pc105` / `us` / `` / ``) when those are unset.
        let keymap = xkb::Keymap::new_from_names(
            &context,
            "",
            "",
            "",
            "",
            None,
            xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
        .context("xkb_keymap_new_from_names failed (couldn't compile a keymap from $XKB_DEFAULT_* / system defaults)")?;
        Ok(Self {
            state: xkb::State::new(&keymap),
        })
    }

    /// Feed a single key event through the xkb state machine and
    /// read out the keysym + effective modifier mask.
    pub fn process(&mut self, keycode: Keycode, pressed: bool) -> KeyResult {
        let direction = if pressed {
            xkb::KeyDirection::Down
        } else {
            xkb::KeyDirection::Up
        };
        self.state.update_key(keycode, direction);

        let keysym = self.state.key_get_one_sym(keycode);
        let mods = self.effective_mods();
        KeyResult { keysym, mods }
    }

    /// Bundle the four modifiers we care about into a single
    /// bitmask. `STATE_MODS_EFFECTIVE` rolls depressed + latched +
    /// locked into one query, which is what hotkey matching wants.
    fn effective_mods(&self) -> u32 {
        let mut mods = 0;
        if self
            .state
            .mod_name_is_active(xkb::MOD_NAME_SHIFT, xkb::STATE_MODS_EFFECTIVE)
        {
            mods |= MOD_SHIFT;
        }
        if self
            .state
            .mod_name_is_active(xkb::MOD_NAME_CTRL, xkb::STATE_MODS_EFFECTIVE)
        {
            mods |= MOD_CTRL;
        }
        if self
            .state
            .mod_name_is_active(xkb::MOD_NAME_ALT, xkb::STATE_MODS_EFFECTIVE)
        {
            mods |= MOD_ALT;
        }
        if self
            .state
            .mod_name_is_active(xkb::MOD_NAME_LOGO, xkb::STATE_MODS_EFFECTIVE)
        {
            mods |= MOD_SUPER;
        }
        mods
    }
}
