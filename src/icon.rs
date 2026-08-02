//! Application icons for titlebars.
//!
//! Resolves a window's `app_id` to a raster icon the titlebar can blit,
//! using the XDG icon-theme directory layout directly rather than
//! linking an icon-theme library — the same trade the font engine makes
//! next door, and for the same reason: what a titlebar needs is "find me
//! a picture for this app", not a theme engine.
//!
//! **PNG only.** The decoder is the `png` crate the screenshot tool
//! already pulls in, so this costs nothing new. It does mean icon themes
//! that ship only SVG resolve to nothing — Breeze is one of them, with
//! 19827 SVGs and no PNGs at all — and those windows get a bar with no
//! icon rather than a placeholder box. A placeholder is worse than a
//! gap: an empty square in a titlebar reads as a *broken* icon, which is
//! exactly the complaint that started this.
//!
//! Adding SVG means adding a rasterizer (resvg and its dependency tree);
//! the lookup below already returns whichever file it finds, so that is
//! a change to [`load`] alone.

use std::path::{Path, PathBuf};

/// Decoded icon: premultiplied RGBA8, row-major.
pub struct Icon {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Directories holding icon themes, most specific first.
fn icon_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(PathBuf::from(&home).join(".local/share/icons"));
        dirs.push(PathBuf::from(&home).join(".icons"));
    }
    if let Ok(data) = std::env::var("XDG_DATA_HOME") {
        dirs.push(PathBuf::from(data).join("icons"));
    }
    for base in std::env::var("XDG_DATA_DIRS")
        .unwrap_or_else(|_| "/usr/local/share:/usr/share".to_owned())
        .split(':')
        .filter(|s| !s.is_empty())
    {
        dirs.push(PathBuf::from(base).join("icons"));
    }
    dirs.push(PathBuf::from("/usr/share/icons"));
    dirs
}

/// Icon *names* to try for an `app_id`, best first.
///
/// A reverse-DNS app id (`org.kde.dolphin`) usually has its icon filed
/// under the whole id, but sometimes only under the last segment, so
/// both are tried. Lowercasing catches clients that report a
/// capitalised id while shipping a lowercase icon file.
fn name_candidates(app_id: &str) -> Vec<String> {
    let mut out = vec![app_id.to_owned()];
    let lower = app_id.to_ascii_lowercase();
    if lower != app_id {
        out.push(lower.clone());
    }
    if let Some(last) = lower.rsplit('.').next()
        && last != lower
    {
        out.push(last.to_owned());
    }
    out
}

/// Find a PNG icon for `app_id` at roughly `want` pixels.
///
/// Prefers the size closest to (and ideally at least) `want`: scaling a
/// 256 px icon down to 20 looks fine, scaling a 16 px one up does not.
#[must_use]
pub fn lookup(app_id: &str, want: u32) -> Option<PathBuf> {
    let names = name_candidates(app_id);
    let mut best: Option<(u32, PathBuf)> = None;
    let mut consider = |size: u32, path: PathBuf| {
        // Rank: an icon at or above the target beats one below it;
        // within each group, closest wins.
        let rank = |s: u32| {
            if s >= want {
                (0u8, s - want)
            } else {
                (1, want - s)
            }
        };
        let better = best
            .as_ref()
            .is_none_or(|(bs, _)| rank(size) < rank(*bs));
        if better {
            best = Some((size, path));
        }
    };
    for dir in icon_dirs() {
        let Ok(themes) = std::fs::read_dir(&dir) else {
            continue;
        };
        for theme in themes.flatten() {
            let theme_dir = theme.path();
            if !theme_dir.is_dir() {
                continue;
            }
            // `<theme>/<size>x<size>/apps/<name>.png`, plus the
            // `<theme>/apps/<size>/<name>.png` layout some themes use.
            let Ok(entries) = std::fs::read_dir(&theme_dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let sub = entry.path();
                let Some(stem) = sub.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                let size = stem
                    .split_once('x')
                    .and_then(|(w, _)| w.parse::<u32>().ok())
                    .or_else(|| stem.parse::<u32>().ok());
                let Some(size) = size else { continue };
                for name in &names {
                    let candidate = sub.join("apps").join(format!("{name}.png"));
                    if candidate.is_file() {
                        consider(size, candidate);
                    }
                }
            }
        }
    }
    // Legacy flat directory, size unknown — treated as a last resort by
    // ranking it as if it were tiny.
    if best.is_none() {
        for name in &names {
            let flat = PathBuf::from("/usr/share/pixmaps").join(format!("{name}.png"));
            if flat.is_file() {
                return Some(flat);
            }
        }
    }
    best.map(|(_, p)| p)
}

/// Decode a PNG to premultiplied RGBA8.
///
/// `None` on anything unreadable or unsupported — a titlebar without an
/// icon is not worth failing a frame over.
#[must_use]
pub fn load(path: &Path) -> Option<Icon> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width, info.height);
    if w == 0 || h == 0 {
        return None;
    }
    let px = usize::try_from(w).ok()? * usize::try_from(h).ok()?;
    let mut rgba = vec![0u8; px * 4];
    match info.color_type {
        png::ColorType::Rgba => rgba.copy_from_slice(&buf[..px * 4]),
        png::ColorType::Rgb => {
            for i in 0..px {
                rgba[i * 4] = buf[i * 3];
                rgba[i * 4 + 1] = buf[i * 3 + 1];
                rgba[i * 4 + 2] = buf[i * 3 + 2];
                rgba[i * 4 + 3] = 255;
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for i in 0..px {
                let (v, a) = (buf[i * 2], buf[i * 2 + 1]);
                rgba[i * 4] = v;
                rgba[i * 4 + 1] = v;
                rgba[i * 4 + 2] = v;
                rgba[i * 4 + 3] = a;
            }
        }
        png::ColorType::Grayscale => {
            for i in 0..px {
                let v = buf[i];
                rgba[i * 4] = v;
                rgba[i * 4 + 1] = v;
                rgba[i * 4 + 2] = v;
                rgba[i * 4 + 3] = 255;
            }
        }
        // Indexed is expanded by the transformations the decoder applies
        // by default; anything else we simply don't draw.
        png::ColorType::Indexed => return None,
    }
    Some(Icon {
        rgba,
        width: w,
        height: h,
    })
}

/// Box-filter an icon down (or nearest-neighbour up) to `size` square.
///
/// A box filter rather than nearest: icons are usually fetched at 2–10×
/// the size a titlebar wants, and point-sampling that ratio produces the
/// crawling, aliased edges that make an icon look cheap.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "icon dimensions are small positive pixel counts and every channel is clamped to [0, 255] before narrowing; the f32 accumulator is exact well past 4096x4096"
)]
pub fn resize(icon: &Icon, size: u32) -> Icon {
    let size = size.max(1);
    if icon.width == size && icon.height == size {
        return Icon {
            rgba: icon.rgba.clone(),
            width: size,
            height: size,
        };
    }
    let mut out = vec![0u8; (size as usize) * (size as usize) * 4];
    for y in 0..size {
        // Source rows this destination row averages over.
        let y0 = y * icon.height / size;
        let y1 = (((y + 1) * icon.height).div_ceil(size)).max(y0 + 1).min(icon.height);
        for x in 0..size {
            let x0 = x * icon.width / size;
            let x1 = (((x + 1) * icon.width).div_ceil(size)).max(x0 + 1).min(icon.width);
            let mut acc = [0f32; 4];
            let mut n = 0f32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let i = ((sy * icon.width + sx) as usize) * 4;
                    // Weight colour by alpha so transparent pixels can't
                    // drag the edges toward black.
                    let a = f32::from(icon.rgba[i + 3]) / 255.0;
                    acc[0] += f32::from(icon.rgba[i]) * a;
                    acc[1] += f32::from(icon.rgba[i + 1]) * a;
                    acc[2] += f32::from(icon.rgba[i + 2]) * a;
                    acc[3] += a;
                    n += 1.0;
                }
            }
            let o = ((y * size + x) as usize) * 4;
            if n > 0.0 && acc[3] > 0.0 {
                out[o] = (acc[0] / acc[3]).clamp(0.0, 255.0) as u8;
                out[o + 1] = (acc[1] / acc[3]).clamp(0.0, 255.0) as u8;
                out[o + 2] = (acc[2] / acc[3]).clamp(0.0, 255.0) as u8;
                out[o + 3] = (acc[3] / n * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Icon {
        rgba: out,
        width: size,
        height: size,
    }
}

#[cfg(test)]
mod tests {
    use super::{Icon, name_candidates, resize};

    #[test]
    fn a_reverse_dns_id_also_tries_its_last_segment() {
        let names = name_candidates("org.kde.dolphin");
        assert_eq!(names[0], "org.kde.dolphin");
        assert!(names.contains(&"dolphin".to_owned()));
    }

    #[test]
    fn a_capitalised_id_also_tries_lowercase() {
        assert!(name_candidates("Alacritty").contains(&"alacritty".to_owned()));
    }

    /// A plain id yields exactly one candidate — no duplicates to stat.
    #[test]
    fn a_simple_id_yields_one_candidate() {
        assert_eq!(name_candidates("kitty"), vec!["kitty".to_owned()]);
    }

    fn solid(w: u32, h: u32, px: [u8; 4]) -> Icon {
        Icon {
            rgba: px.repeat((w * h) as usize),
            width: w,
            height: h,
        }
    }

    /// Downscaling a solid icon must preserve its colour exactly — an
    /// unweighted average over transparent pixels is what turns icon
    /// edges muddy.
    #[test]
    fn downscaling_preserves_a_solid_colour() {
        let big = solid(64, 64, [200, 100, 50, 255]);
        let small = resize(&big, 16);
        assert_eq!(small.width, 16);
        assert_eq!(small.height, 16);
        for px in small.rgba.chunks_exact(4) {
            assert_eq!(px, [200, 100, 50, 255]);
        }
    }

    /// A fully transparent icon stays transparent rather than becoming a
    /// black square.
    #[test]
    fn transparency_survives_resizing() {
        let clear = solid(32, 32, [0, 0, 0, 0]);
        assert!(resize(&clear, 8).rgba.chunks_exact(4).all(|p| p[3] == 0));
    }

    /// Upscaling and same-size are both legal and keep the requested
    /// dimensions — a 16 px icon in a 24 px slot must not panic.
    #[test]
    fn resizing_up_and_to_the_same_size_is_safe() {
        let small = solid(8, 8, [10, 20, 30, 255]);
        assert_eq!(resize(&small, 24).width, 24);
        assert_eq!(resize(&small, 8).rgba.len(), 8 * 8 * 4);
        assert_eq!(resize(&small, 1).rgba.len(), 4);
    }
}
