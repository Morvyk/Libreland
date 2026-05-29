//! `XCursor` theme loading for the compositor's pointer.
//!
//! We load a real `XCursor` theme the same way every other Wayland/X11
//! app does: read `$XCURSOR_THEME` / `$XCURSOR_SIZE` from the
//! environment and pull cursors out of the matching theme on disk. The
//! `default` arrow is loaded up front; *named* cursors (grabbing,
//! crosshair, resize, …) are loaded on demand when a client requests a
//! [`smithay::input::pointer::CursorImageStatus::Named`] cursor (via
//! `wp_cursor_shape_v1`) or the compositor sets one for its own grabs.
//! This module is renderer-agnostic — it produces raw RGBA pixels;
//! `render` uploads them to a texture.

use smithay::input::pointer::CursorIcon;
use tracing::{info, warn};
use xcursor::CursorTheme;
use xcursor::parser::parse_xcursor;

/// Nominal cursor side length in logical pixels when `$XCURSOR_SIZE`
/// is unset. 24 is the de-facto desktop default (GTK, libwayland).
pub const DEFAULT_SIZE: u32 = 24;

/// One decoded cursor image, ready for texture upload. Pixels are
/// `RGBA8888`, row-major, top row first.
pub struct CursorImage {
    /// Image width in its own pixels.
    pub width: i32,
    /// Image height in its own pixels.
    pub height: i32,
    /// Hotspot X in image pixels (the "tip" that tracks the pointer).
    pub xhot: i32,
    /// Hotspot Y in image pixels.
    pub yhot: i32,
    /// Nominal size the artwork was authored for (the `XCursor` `size`
    /// field). Used to normalise the sprite back to the requested
    /// logical size at draw time, independent of how many physical
    /// pixels the chosen image happens to carry.
    pub nominal: i32,
    /// `RGBA8888` pixels, top-to-bottom.
    pub rgba: Vec<u8>,
}

/// Resolve the configured cursor size from `$XCURSOR_SIZE`, falling
/// back to [`DEFAULT_SIZE`] when unset or unparseable.
pub fn configured_size() -> u32 {
    std::env::var("XCURSOR_SIZE")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_SIZE)
}

/// Load the `default` pointer (the standard arrow), choosing the image
/// whose nominal size is closest to `target_px` physical pixels.
/// Returns `None` — leaving the caller to fall back to its built-in
/// sprite — if the theme or icon can't be found or parsed.
pub fn load_default_cursor(target_px: u32) -> Option<CursorImage> {
    // "default" is the modern CSS/X11 name for the standard arrow;
    // older themes only ship the legacy "left_ptr" alias, so try both.
    load_named(&["default", "left_ptr"], target_px, true)
}

/// Load a *named* cursor ([`CursorIcon`]) by its CSS name plus the
/// theme-specific aliases the `cursor-icon` crate knows about (e.g.
/// `grabbing` → `closedhand`, `crosshair` → `cross`). Returns `None`
/// when the theme doesn't ship that cursor, so the caller can fall back
/// to the default arrow rather than draw nothing.
pub fn load_named_cursor(icon: CursorIcon, target_px: u32) -> Option<CursorImage> {
    let mut names: Vec<&str> = Vec::with_capacity(1 + icon.alt_names().len());
    names.push(icon.name());
    names.extend_from_slice(icon.alt_names());
    load_named(&names, target_px, false)
}

/// Load the first of `names` present in the `$XCURSOR_THEME` theme
/// (falling back to the theme literally named "default"). `warn_missing`
/// logs when none of the names resolve — used only for the default
/// cursor, since a missing *named* cursor is expected and handled by
/// falling back to the arrow.
fn load_named(names: &[&str], target_px: u32, warn_missing: bool) -> Option<CursorImage> {
    let theme_name = std::env::var("XCURSOR_THEME").unwrap_or_else(|_| "default".to_owned());
    let theme = CursorTheme::load(&theme_name);

    let Some(path) = names.iter().find_map(|n| theme.load_icon(n)) else {
        if warn_missing {
            warn!(theme = %theme_name, ?names, "cursor not found in XCursor theme; using built-in sprite");
        }
        return None;
    };

    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(err) => {
            warn!(path = %path.display(), error = %err, "failed to read cursor file");
            return None;
        }
    };

    let Some(images) = parse_xcursor(&bytes) else {
        warn!(path = %path.display(), "failed to parse XCursor file");
        return None;
    };

    // An XCursor file holds every size (and animation frame). Pick the
    // image whose nominal size is nearest the target; on a tie, the
    // earliest in file order wins — for animated cursors that's frame
    // 0, which is what we want for a static cursor. (Tie-breaking on
    // `delay` would pick an arbitrary frame, not the first.)
    let image = images
        .iter()
        .enumerate()
        .min_by_key(|(idx, img)| {
            let diff = i64::from(img.size).abs_diff(i64::from(target_px));
            (diff, *idx)
        })
        .map(|(_, img)| img)?;

    info!(
        theme = %theme_name,
        nominal = image.size,
        width = image.width,
        height = image.height,
        "loaded XCursor theme cursor"
    );

    Some(CursorImage {
        // XCursor dimensions are small (<= a few hundred px); the
        // casts can't realistically overflow i32.
        width: i32::try_from(image.width).ok()?,
        height: i32::try_from(image.height).ok()?,
        xhot: i32::try_from(image.xhot).ok()?,
        yhot: i32::try_from(image.yhot).ok()?,
        nominal: i32::try_from(image.size).ok()?.max(1),
        rgba: image.pixels_rgba.clone(),
    })
}
