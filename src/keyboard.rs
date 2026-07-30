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

/// Fold an ASCII-letter keysym to its lowercase form for hotkey
/// matching. xkbcommon hands us the *shifted* keysym, so an unshifted
/// `c` arrives as `Keysym::c` (`0x63`) while a bind written as `"C"`
/// (or the built-in `Keysym::C`) is `0x43` — they'd never compare
/// equal without folding, so letter binds silently never fired unless
/// Shift happened to be held. Modifiers are matched separately, so
/// folding case here doesn't conflate `Super+C` with `Super+Shift+C`:
/// the required-mods check still distinguishes binds that ask for
/// Shift. Non-letter keysyms (`space`, `F1`, `Return`) are returned
/// unchanged.
#[must_use]
pub fn fold_keysym(k: Keysym) -> Keysym {
    let raw = k.raw();
    if (0x41..=0x5A).contains(&raw) {
        Keysym::new(raw + 0x20)
    } else {
        k
    }
}

/// Whether a key event can satisfy a bind written for `want`.
///
/// Either the symbol actually produced or the key's unmodified symbol will
/// do, so `Super+Shift+2` matches a bind on `2` regardless of what the
/// layout puts on `Shift`+`2`.
#[must_use]
pub fn matches_key(result: &KeyResult, want: Keysym) -> bool {
    let want = fold_keysym(want);
    fold_keysym(result.keysym) == want || fold_keysym(result.base_keysym) == want
}

/// Whether `k` is a *held* modifier key — the ones that mean nothing on
/// their own and exist to qualify another key.
///
/// A bind whose key is one of these can only sensibly mean "tap it"
/// (see `State::update_tap_state`), because its press is also the press
/// that begins every combo built on it.
///
/// The latching locks (`Caps_Lock`, `Num_Lock`, `Shift_Lock`,
/// `Scroll_Lock`) are deliberately NOT here: they toggle a state on
/// press, so a plain press-bind on one is already meaningful and must
/// keep working.
#[must_use]
pub fn is_modifier_keysym(k: Keysym) -> bool {
    use xkbcommon::xkb::keysyms;
    matches!(
        k.raw(),
        keysyms::KEY_Shift_L
            | keysyms::KEY_Shift_R
            | keysyms::KEY_Control_L
            | keysyms::KEY_Control_R
            | keysyms::KEY_Meta_L
            | keysyms::KEY_Meta_R
            | keysyms::KEY_Alt_L
            | keysyms::KEY_Alt_R
            | keysyms::KEY_Super_L
            | keysyms::KEY_Super_R
            | keysyms::KEY_Hyper_L
            | keysyms::KEY_Hyper_R
            | keysyms::KEY_ISO_Level3_Shift
            | keysyms::KEY_ISO_Level5_Shift
    )
}

/// Outcome of feeding a single libinput key event through xkbcommon:
/// the layout-aware keysym at this moment (with modifier effects
/// applied — `Shift+e` becomes `Keysym::E`), and a bitmask of the
/// effective modifiers.
pub struct KeyResult {
    pub keysym: Keysym,
    /// The same key's symbol at shift level 1 — what the key produces with
    /// no modifiers applied.
    ///
    /// Binds are written the way the key is *labelled* (`key = "2"`), but
    /// with Shift held xkb resolves that keycode to whatever the layout puts
    /// on its shifted level — `@` on US, `"` on Swedish, and so on. No
    /// layout leaves the digits alone, so matching on [`Self::keysym`] alone
    /// breaks `Shift`+digit binds *everywhere*, not just on exotic layouts.
    /// Silently, too: the bind still lists correctly in
    /// `libreland msg binds`, it simply never fires.
    ///
    /// [`fold_keysym`] can't fix this: it folds A-Z case, and `Shift`+`2`
    /// is not a case change but a different symbol entirely.
    pub base_keysym: Keysym,
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
    /// Compile a keymap using the given `RMLVO` layout (the rest of
    /// the fields use xkbcommon's env-or-default fallback). Passing
    /// `""` defers entirely to `XKB_DEFAULT_LAYOUT` / system default,
    /// which is what `Config::default` does so the user's existing
    /// keyboard config applies untouched.
    pub fn new(layout: &str) -> Result<Self> {
        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let keymap = xkb::Keymap::new_from_names(
            &context,
            "",
            "",
            layout,
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
        // Level 0 of the key's *current* layout group, so a layout switch is
        // still honoured; only the shift/level part is discarded.
        let layout = self.state.key_get_layout(keycode);
        let base_keysym = self
            .state
            .get_keymap()
            .key_get_syms_by_level(keycode, layout, 0)
            .first()
            .copied()
            .unwrap_or(keysym);
        let mods = self.effective_mods();
        KeyResult {
            keysym,
            base_keysym,
            mods,
        }
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

/// Parse a `SUPER+SHIFT+e`-style trigger into `(mods, keysym)`.
///
/// This is the spelling the desktop portal's `GlobalShortcuts` clients use
/// (`LOGO`, `CTRL`, `ALT`, `SHIFT` plus an xkb key name), and it's what
/// [`crate::ipc::Request::RegisterBind`] takes. Key names are resolved by
/// xkbcommon, case-insensitively as a fallback so `E` and `e` both work —
/// the bind matcher folds case anyway.
///
/// `None` when no part of it names a key we can bind.
#[must_use]
pub fn parse_trigger(trigger: &str) -> Option<(u32, Keysym)> {
    let mut mods = 0;
    let mut key = None;
    for part in trigger.split('+').map(str::trim).filter(|p| !p.is_empty()) {
        match part.to_ascii_uppercase().as_str() {
            "SUPER" | "LOGO" | "META" | "MOD4" | "CMD" => mods |= MOD_SUPER,
            "CTRL" | "CONTROL" => mods |= MOD_CTRL,
            "ALT" | "MOD1" => mods |= MOD_ALT,
            "SHIFT" => mods |= MOD_SHIFT,
            _ => {
                let sym = xkb::keysym_from_name(part, xkb::KEYSYM_NO_FLAGS);
                let sym = if sym == Keysym::NoSymbol {
                    xkb::keysym_from_name(part, xkb::KEYSYM_CASE_INSENSITIVE)
                } else {
                    sym
                };
                if sym == Keysym::NoSymbol {
                    return None;
                }
                key = Some(sym);
            }
        }
    }
    key.map(|k| (mods, k))
}

/// Render `(mods, keysym)` back into the trigger notation, so a caller can
/// see what its request actually bound.
#[must_use]
pub fn format_trigger(mods: u32, keysym: Keysym) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for (bit, name) in [
        (MOD_SUPER, "SUPER"),
        (MOD_CTRL, "CTRL"),
        (MOD_ALT, "ALT"),
        (MOD_SHIFT, "SHIFT"),
    ] {
        if mods & bit != 0 {
            parts.push(name);
        }
    }
    let key = xkb::keysym_get_name(keysym);
    if parts.is_empty() {
        key
    } else {
        format!("{}+{key}", parts.join("+"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        KeyResult, Keysym, MOD_SHIFT, MOD_SUPER, fold_keysym, format_trigger,
        is_modifier_keysym, matches_key, parse_trigger,
    };

    fn ev(produced: Keysym, base: Keysym) -> KeyResult {
        KeyResult {
            keysym: produced,
            base_keysym: base,
            mods: 0,
        }
    }

    /// The bug this exists for, and it is not layout-specific: no layout
    /// leaves the digit row alone under Shift. US gives `@` for Shift+2,
    /// Swedish gives `"`. The shifted level is a different symbol, not a
    /// case change, so matching only the produced keysym meant every
    /// `Shift`+digit bind silently never fired — on every layout — while
    /// still listing correctly in `libreland msg binds`.
    #[test]
    fn a_shifted_digit_matches_the_bind_written_for_the_digit() {
        for (produced, digit) in [
            (Keysym::quotedbl, Keysym::_2),   // se
            (Keysym::at, Keysym::_2),         // us
            (Keysym::exclam, Keysym::_1),     // both
            (Keysym::numbersign, Keysym::_3), // us
        ] {
            assert!(
                matches_key(&ev(produced, digit), digit),
                "{produced:?} should satisfy a bind on {digit:?}"
            );
        }
    }

    /// The fix reads the keycode's own level 0 rather than consulting a
    /// table of substitutions, so a layout nobody anticipated works too.
    #[test]
    fn the_fix_is_layout_agnostic() {
        assert!(matches_key(&ev(Keysym::eacute, Keysym::_2), Keysym::_2));
        assert!(matches_key(&ev(Keysym::periodcentered, Keysym::_3), Keysym::_3));
    }

    /// A bind written for the *shifted* symbol still matches it directly, so
    /// `key = "exclam"` keeps working for anyone who wrote it that way.
    #[test]
    fn a_bind_on_the_shifted_symbol_still_matches() {
        assert!(matches_key(&ev(Keysym::exclam, Keysym::_1), Keysym::exclam));
    }

    /// And it must not match everything: a different key is still a
    /// different key.
    #[test]
    fn unrelated_keys_do_not_match() {
        assert!(!matches_key(&ev(Keysym::quotedbl, Keysym::_2), Keysym::_3));
        assert!(!matches_key(&ev(Keysym::a, Keysym::a), Keysym::b));
    }

    /// Case folding still applies on top, so `Super+Shift+E` reaches a bind
    /// written `key = "E"` or `key = "e"`.
    #[test]
    fn case_folding_still_applies() {
        assert!(matches_key(&ev(Keysym::E, Keysym::e), Keysym::e));
        assert!(matches_key(&ev(Keysym::E, Keysym::e), Keysym::E));
    }

    #[test]
    fn triggers_round_trip() {
        let (mods, key) = parse_trigger("SUPER+SHIFT+e").expect("parses");
        assert_eq!(mods, MOD_SUPER | MOD_SHIFT);
        assert_eq!(key, Keysym::e);
        assert_eq!(format_trigger(mods, key), "SUPER+SHIFT+e");
    }

    #[test]
    fn trigger_modifier_aliases_are_accepted() {
        // The portal spells Super as LOGO; xkb tools spell it SUPER.
        assert_eq!(parse_trigger("LOGO+p"), parse_trigger("SUPER+p"));
    }

    #[test]
    fn a_trigger_without_a_key_is_rejected() {
        assert!(parse_trigger("SUPER+SHIFT").is_none());
        assert!(parse_trigger("").is_none());
        assert!(parse_trigger("SUPER+notakey").is_none());
    }

    #[test]
    fn function_and_named_keys_parse() {
        assert_eq!(parse_trigger("F5").map(|(_, k)| k), Some(Keysym::F5));
        assert_eq!(
            parse_trigger("CTRL+Print").map(|(_, k)| k),
            Some(Keysym::Print)
        );
    }

    #[test]
    fn held_modifiers_are_recognised() {
        for k in [
            Keysym::Super_L,
            Keysym::Super_R,
            Keysym::Alt_L,
            Keysym::Alt_R,
            Keysym::Control_L,
            Keysym::Control_R,
            Keysym::Shift_L,
            Keysym::Shift_R,
            Keysym::Meta_L,
            Keysym::Hyper_R,
            Keysym::ISO_Level3_Shift,
        ] {
            assert!(is_modifier_keysym(k), "{k:?} should be a held modifier");
        }
    }

    #[test]
    fn ordinary_and_latching_keys_are_not() {
        // Latching locks keep their press semantics on purpose.
        for k in [
            Keysym::Caps_Lock,
            Keysym::Num_Lock,
            Keysym::Scroll_Lock,
            Keysym::a,
            Keysym::E,
            Keysym::Return,
            Keysym::F1,
            Keysym::space,
            Keysym::Print,
        ] {
            assert!(!is_modifier_keysym(k), "{k:?} should not be a held modifier");
        }
    }

    #[test]
    fn folding_leaves_modifier_keysyms_alone() {
        // The tap path compares folded keysyms; folding must be a no-op
        // here or a bind would never match its own release.
        for k in [Keysym::Super_L, Keysym::Alt_R, Keysym::Shift_L] {
            assert_eq!(fold_keysym(k), k);
        }
    }
}
