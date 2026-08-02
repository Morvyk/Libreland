//! Text: font discovery, glyph rasterization, and a coverage cache.
//!
//! Shared by the two things in this workspace that draw their own text: the
//! portal's dialogs and the compositor's window titlebars. Both want the same
//! narrow thing — "give me a UI font and draw this string" — and neither wants
//! a font *system*, so we deliberately do not link fontconfig, pango or cairo.
//! That request is a directory walk plus a rasterizer: [`fontdue`] is the
//! rasterizer, and the walk below is the discovery.
//!
//! Font selection is by filename against a preference list, which is crude
//! next to fontconfig's matching but has the property that matters here: it
//! resolves to the same well-known families every desktop already ships, with
//! no config, no daemon, and no failure mode where a broken fontconfig cache
//! takes the file dialog — or every titlebar on the desktop — down with it.
//! Missing glyphs fall through a chain of whatever wide-coverage faces were
//! found (Noto, `DejaVu`), so CJK and Cyrillic render even though the primary
//! UI face has no glyphs for them.
//!
//! [`Fonts::load`] scans the font directories, so callers build **one**
//! instance and keep it: the compositor holds it for the process lifetime and
//! rasterizes titlebar text through it on demand.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel geometry: every value here is a surface- or image-sized non-negative integer, and the conversions between i32/u32/usize/f32 are all inside that range. Checked conversions at each site would be noise around arithmetic that cannot overflow."
)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use fontdue::{Font, FontSettings};

/// Directories scanned for fonts, in ascending priority (user fonts win, so
/// they come last and overwrite earlier hits of the same filename).
fn font_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![
        PathBuf::from("/usr/share/fonts"),
        PathBuf::from("/usr/local/share/fonts"),
    ];
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(PathBuf::from(&home).join(".fonts"));
        dirs.push(PathBuf::from(&home).join(".local/share/fonts"));
    }
    if let Ok(data) = std::env::var("XDG_DATA_HOME") {
        dirs.push(PathBuf::from(data).join("fonts"));
    }
    dirs
}

/// Filename stems we accept as "the UI font", best first. These are the
/// regular weights; [`BOLD_CANDIDATES`] mirrors them.
const UI_CANDIDATES: &[&str] = &[
    "inter-regular",
    "inter_24pt-regular",
    "cantarell-regular",
    "notosans-regular",
    "dejavusans",
    "liberationsans-regular",
    "roboto-regular",
    "ubuntu-r",
    "opensans-regular",
    "firasans-regular",
    "sourcesans3-regular",
    "arimo-regular",
];

const BOLD_CANDIDATES: &[&str] = &[
    "inter-bold",
    "inter_24pt-bold",
    "cantarell-bold",
    "notosans-bold",
    "dejavusans-bold",
    "liberationsans-bold",
    "roboto-bold",
    "ubuntu-b",
    "opensans-bold",
    "firasans-bold",
    "sourcesans3-bold",
    "arimo-bold",
];

/// Stem fragments that mark a font as likely to carry the odd glyph a
/// UI face lacks. Used to *order* the exhaustive search below, so the
/// symbol and icon fonts are tried before the hundreds of ordinary text
/// faces a desktop accumulates.
///
/// Best first; [`Fonts::load`] reverses this into the search stack.
const WIDE_COVERAGE_HINTS: &[&str] = &[
    "symbols", "symbol2", "nerd", "unifont", "cjk", "emoji", "noto", "dejavu",
];

#[cfg(test)]
mod tests {
    use super::{WIDE_COVERAGE_HINTS, coverage_rank};

    /// The search stack is popped from the end, so the fonts most likely
    /// to carry an unusual glyph have to sort *last*. Getting this
    /// backwards is silent — it still finds a face eventually, just after
    /// loading every ordinary text font on the system first.
    #[test]
    fn wide_coverage_fonts_sort_to_the_end_of_the_search_stack() {
        let mut stack = vec![
            "adwaitasans-regular".to_owned(),
            "symbolsnerdfont-regular".to_owned(),
            "liberationserif-italic".to_owned(),
            "notosanssymbols2-regular".to_owned(),
        ];
        stack.sort_by_key(|s| (std::cmp::Reverse(coverage_rank(s)), s.clone()));
        let last_two = &stack[stack.len() - 2..];
        assert!(
            last_two.iter().all(|s| s.contains("symbols") || s.contains("nerd")),
            "symbol/nerd fonts must be popped first, got {stack:?}"
        );
    }

    /// A font matching no hint still sorts, and behind the hinted ones.
    #[test]
    fn unhinted_fonts_rank_last() {
        assert_eq!(coverage_rank("randomfont"), WIDE_COVERAGE_HINTS.len());
        assert!(coverage_rank("notosans-regular") < WIDE_COVERAGE_HINTS.len());
    }
}

/// How likely a font stem is to carry unusual glyphs: lower is better.
fn coverage_rank(stem: &str) -> usize {
    WIDE_COVERAGE_HINTS
        .iter()
        .position(|h| stem.contains(h))
        .unwrap_or(WIDE_COVERAGE_HINTS.len())
}

/// Wide-coverage faces preloaded as the first fallbacks.
const FALLBACK_CANDIDATES: &[&str] = &[
    "notosanscjk-regular",
    "notosanscjkjp-regular",
    "notosanscjksc-regular",
    "notosansjp-regular",
    "notosanssc-regular",
    "notosanskr-regular",
    "notosansarabic-regular",
    "notosanshebrew-regular",
    "notosansthai-regular",
    "notosansdevanagari-regular",
    "dejavusans",
    "notosans-regular",
    "unifont",
];

/// One scanned font file: the lowercased stem we match on, and its path.
struct Candidate {
    stem: String,
    path: PathBuf,
}

/// Walk the font directories (bounded depth — font trees are shallow, and an
/// unbounded walk over a symlinked share dir is a hang waiting to happen).
fn scan_fonts() -> Vec<Candidate> {
    fn walk(dir: &Path, depth: usize, out: &mut Vec<Candidate>) {
        if depth > 4 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            // `file_type` on the DirEntry avoids a stat per file; symlinked
            // directories are followed only via the metadata fallback below.
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() {
                walk(&path, depth + 1, out);
                continue;
            }
            let is_font = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "ttf" | "otf"));
            if !is_font {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                out.push(Candidate {
                    stem: stem.to_ascii_lowercase(),
                    path,
                });
            }
        }
    }

    let mut out = Vec::new();
    for dir in font_dirs() {
        walk(&dir, 0, &mut out);
    }
    out
}

/// Load the first candidate whose stem matches one of `wanted`, in `wanted`'s
/// order (so the preference list, not the directory order, decides).
fn load_preferred(all: &[Candidate], wanted: &[&str]) -> Option<Font> {
    for want in wanted {
        for candidate in all {
            if candidate.stem == *want
                && let Some(font) = load_font(&candidate.path)
            {
                return Some(font);
            }
        }
    }
    None
}

fn load_font(path: &Path) -> Option<Font> {
    let bytes = std::fs::read(path).ok()?;
    // Collections (.ttc) and variable fonts both resolve to index 0 here,
    // which is the regular/default instance — good enough for a UI face.
    Font::from_bytes(
        bytes,
        FontSettings {
            collection_index: 0,
            scale: 40.0,
            ..FontSettings::default()
        },
    )
    .ok()
}

/// A rasterized glyph: 8-bit coverage plus its placement relative to the pen.
struct Glyph {
    width: usize,
    height: usize,
    /// Offset from the pen position to the bitmap's top-left corner.
    left: i32,
    top: i32,
    advance: f32,
    coverage: Vec<u8>,
}

/// Cache key: a character at a size, in a weight. Sizes are quantized to
/// 1/4 px so the cache can't be blown up by a scroll animation.
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct GlyphKey {
    ch: char,
    quarter_px: u32,
    bold: bool,
}

/// The loaded faces plus the rasterization cache.
///
/// One instance is shared by every dialog (built once, on first use); the
/// cache is behind the same lock as the faces because rasterizing is the only
/// thing we do with them.
pub struct Fonts {
    regular: Arc<Font>,
    bold: Option<Arc<Font>>,
    fallbacks: Vec<Arc<Font>>,
    /// Every other font file found, in search order, not yet loaded.
    ///
    /// Loading all of them up front would be hundreds of multi-megabyte
    /// reads for glyphs almost nothing needs; leaving them out entirely
    /// is what put tofu in titlebars, because the preloaded fallbacks
    /// are chosen by name and a desktop's *actual* missing glyphs are
    /// whatever its terminal prompt happens to use.
    unscanned: std::sync::Mutex<Vec<Candidate>>,
    /// Faces loaded on demand by the exhaustive search, kept so a second
    /// character from the same font costs nothing.
    extra: std::sync::Mutex<Vec<Arc<Font>>>,
    /// Which face covers a character, resolved once. `None` records
    /// "nothing on this system has it", so an un-drawable glyph doesn't
    /// re-scan every font at every size.
    resolved: std::sync::Mutex<HashMap<char, Option<Arc<Font>>>>,
    cache: std::sync::Mutex<HashMap<GlyphKey, Arc<Glyph>>>,
}

impl Fonts {
    /// Discover and load the UI faces. `None` when the system has no usable
    /// font at all, which the callers treat as "run without text" rather than
    /// as a fatal error — a file dialog with no labels still beats no dialog,
    /// and a titlebar with no title still closes and drags.
    ///
    /// Scans every font directory, so call it once and keep the result.
    #[must_use]
    pub fn load() -> Option<Self> {
        let all = scan_fonts();
        let regular = load_preferred(&all, UI_CANDIDATES)
            // Nothing from the preference list: take any face at all, sorted
            // for determinism so we don't pick a different font per boot.
            .or_else(|| {
                let mut sorted: Vec<&Candidate> = all.iter().collect();
                sorted.sort_by(|a, b| a.stem.cmp(&b.stem));
                sorted.iter().find_map(|c| load_font(&c.path))
            })?;
        let bold = load_preferred(&all, BOLD_CANDIDATES).map(Arc::new);
        // Preload a couple of named wide-coverage faces so the common
        // cases never touch the search below. The cap is small on
        // purpose — the search is what provides real coverage, and it
        // only loads a font when something actually needs it.
        let mut fallbacks = Vec::new();
        for want in FALLBACK_CANDIDATES {
            if let Some(c) = all.iter().find(|c| c.stem == *want)
                && let Some(font) = load_font(&c.path)
            {
                fallbacks.push(Arc::new(font));
                if fallbacks.len() >= 2 {
                    break;
                }
            }
        }
        // Everything else, ordered so the fonts most likely to carry an
        // unusual glyph are taken *last* — the search pops from the end.
        let mut unscanned = all;
        unscanned.sort_by_key(|c| {
            // Descending rank, so `pop` yields the best candidate first.
            (std::cmp::Reverse(coverage_rank(&c.stem)), c.stem.clone())
        });
        Some(Self {
            regular: Arc::new(regular),
            bold,
            fallbacks,
            unscanned: std::sync::Mutex::new(unscanned),
            extra: std::sync::Mutex::new(Vec::new()),
            resolved: std::sync::Mutex::new(HashMap::new()),
            cache: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// The face that actually has `ch`: the requested weight if it covers the
    /// character, else the first fallback that does, else the requested weight
    /// again (so we draw its .notdef box rather than nothing).
    fn face_for(&self, ch: char, bold: bool) -> Arc<Font> {
        let primary = if bold {
            self.bold.as_ref().unwrap_or(&self.regular)
        } else {
            &self.regular
        };
        if primary.lookup_glyph_index(ch) != 0 || ch == ' ' {
            return Arc::clone(primary);
        }
        for face in &self.fallbacks {
            if face.lookup_glyph_index(ch) != 0 {
                return Arc::clone(face);
            }
        }
        if let Some(found) = self.search_for(ch) {
            return found;
        }
        // Nothing on the system has it: draw the primary's .notdef box,
        // which at least shows *something* is there.
        Arc::clone(primary)
    }

    /// Find any installed font covering `ch`, loading candidates in
    /// coverage-likelihood order until one does.
    ///
    /// This is what "the full font" means here: rather than trusting a
    /// hardcoded list of family names — which on a real desktop resolved
    /// to Arabic, Hebrew and Thai while the Nerd Font and symbol faces
    /// carrying the actually-missing glyphs sat unused — ask every font
    /// present, and remember the answer.
    fn search_for(&self, ch: char) -> Option<Arc<Font>> {
        if let Ok(resolved) = self.resolved.lock()
            && let Some(hit) = resolved.get(&ch)
        {
            return hit.clone();
        }
        let mut extra = self.extra.lock().ok()?;
        let found = extra
            .iter()
            .find(|f| f.lookup_glyph_index(ch) != 0)
            .map(Arc::clone)
            .or_else(|| {
                let mut unscanned = self.unscanned.lock().ok()?;
                while let Some(candidate) = unscanned.pop() {
                    let Some(font) = load_font(&candidate.path) else {
                        continue;
                    };
                    let font = Arc::new(font);
                    let covers = font.lookup_glyph_index(ch) != 0;
                    extra.push(Arc::clone(&font));
                    if covers {
                        return Some(font);
                    }
                }
                None
            });
        if let Ok(mut resolved) = self.resolved.lock() {
            resolved.insert(ch, found.clone());
        }
        found
    }

    fn glyph(&self, ch: char, px: f32, bold: bool) -> Arc<Glyph> {
        let key = GlyphKey {
            ch,
            quarter_px: (px * 4.0).max(0.0) as u32,
            bold,
        };
        if let Ok(cache) = self.cache.lock()
            && let Some(hit) = cache.get(&key)
        {
            return Arc::clone(hit);
        }
        let face = self.face_for(ch, bold);
        let (metrics, coverage) = face.rasterize(ch, px);
        let glyph = Arc::new(Glyph {
            width: metrics.width,
            height: metrics.height,
            left: metrics.xmin,
            // fontdue reports ymin as the distance from the baseline to the
            // bitmap's BOTTOM (y up). Our canvas is y-down and we draw from a
            // baseline, so the top edge sits height + ymin above it.
            top: -(metrics.height as i32 + metrics.ymin),
            advance: metrics.advance_width,
            coverage,
        });
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key, Arc::clone(&glyph));
        }
        glyph
    }

    /// Width of `text` at `px`, in pixels.
    pub fn measure(&self, text: &str, px: f32, bold: bool) -> f32 {
        text.chars()
            .map(|ch| self.glyph(ch, px, bold).advance)
            .sum()
    }

    /// Truncate `text` to fit `max_width`, appending an ellipsis when it had
    /// to cut. Used everywhere a filename can be arbitrarily long.
    pub fn ellipsize(&self, text: &str, px: f32, bold: bool, max_width: f32) -> String {
        if self.measure(text, px, bold) <= max_width {
            return text.to_string();
        }
        let ellipsis = '…';
        let budget = max_width - self.glyph(ellipsis, px, bold).advance;
        if budget <= 0.0 {
            return String::new();
        }
        let mut used = 0.0;
        let mut out = String::new();
        for ch in text.chars() {
            let advance = self.glyph(ch, px, bold).advance;
            if used + advance > budget {
                break;
            }
            used += advance;
            out.push(ch);
        }
        out.push(ellipsis);
        out
    }

    /// Iterate the glyph bitmaps of `text`, calling `emit(x, y, w, h,
    /// coverage)` per glyph with `(x, y)` the bitmap's top-left in canvas
    /// space. `origin` is the pen start and `baseline` the baseline y.
    ///
    /// The canvas does the blending; this only decides where the pixels go.
    pub fn layout<F: FnMut(i32, i32, usize, usize, &[u8])>(
        &self,
        text: &str,
        px: f32,
        bold: bool,
        origin_x: f32,
        baseline_y: f32,
        mut emit: F,
    ) {
        let mut pen = origin_x;
        for ch in text.chars() {
            let glyph = self.glyph(ch, px, bold);
            if glyph.width > 0 && glyph.height > 0 {
                let x = pen as i32 + glyph.left;
                let y = baseline_y as i32 + glyph.top;
                emit(x, y, glyph.width, glyph.height, &glyph.coverage);
            }
            pen += glyph.advance;
        }
    }
}
