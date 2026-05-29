//! `XCursor` theme loading for the compositor's own pointer.
//!
//! The compositor draws its own pointer cursor (clients only get a
//! say via `wl_pointer.set_cursor`, which we don't honour yet). Rather
//! than a hardcoded sprite we load a real `XCursor` theme the same way
//! every other Wayland/X11 app does: read `$XCURSOR_THEME` /
//! `$XCURSOR_SIZE` from the environment and pull the `default` cursor
//! out of the matching theme on disk. This module is renderer-agnostic
//! — it produces raw RGBA pixels; `render` uploads them to a texture.

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

/// Load the `default` pointer of the theme named by `$XCURSOR_THEME`
/// (falling back to the theme literally named "default"), choosing
/// the image whose nominal size is closest to `target_px` physical
/// pixels. Returns `None` — leaving the caller to fall back to its
/// built-in sprite — if the theme or icon can't be found or parsed.
pub fn load_default_cursor(target_px: u32) -> Option<CursorImage> {
    let theme_name = std::env::var("XCURSOR_THEME").unwrap_or_else(|_| "default".to_owned());
    let theme = CursorTheme::load(&theme_name);

    // "default" is the modern CSS/X11 name for the standard arrow;
    // older themes only ship the legacy "left_ptr" alias, so try both.
    let Some(path) = theme
        .load_icon("default")
        .or_else(|| theme.load_icon("left_ptr"))
    else {
        warn!(theme = %theme_name, "no `default`/`left_ptr` cursor in XCursor theme; using built-in sprite");
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
