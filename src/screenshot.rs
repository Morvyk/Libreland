//! Built-in screenshot tool — the pure edges.
//!
//! Pixel cropping + PNG encoding, the timestamped filename, save-path
//! expansion, and writing the file. The interactive session (selection
//! UI, freeze, capture wiring, clipboard) lives on [`crate::State`] in
//! `main.rs`; this module is the stateless, testable parts that don't
//! touch compositor state.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use smithay::utils::{Physical, Rectangle};
use time::OffsetDateTime;
use time::UtcOffset;
use time::macros::format_description;

/// Extract `region` (top-left origin, in the upright image) from a
/// captured framebuffer read-back and encode it as a PNG (RGB, opaque).
///
/// `src` is `src_w * src_h * 4` bytes in memory order **B, G, R, X** (the
/// `Xrgb8888` read-back; X is undefined padding, never alpha), in **natural
/// top-down row order** (row 0 = top of the image — confirmed visually for
/// our scanout framebuffers; no row reversal needed). The region is
/// clamped to `src`.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "region/src dimensions are non-negative physical pixel counts bounded by output size, well within usize/u32"
)]
pub(crate) fn encode_region(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    region: Rectangle<i32, Physical>,
) -> Result<Vec<u8>, png::EncodingError> {
    let (sw, sh) = (src_w as usize, src_h as usize);
    let src_stride = sw * 4;
    let rx = (region.loc.x.max(0) as usize).min(sw);
    let ry = (region.loc.y.max(0) as usize).min(sh);
    let rw = (region.size.w.max(0) as usize).min(sw - rx);
    let rh = (region.size.h.max(0) as usize).min(sh - ry);

    let mut rgb = vec![0u8; rw * rh * 3];
    for out_y in 0..rh {
        let s = &src[(ry + out_y) * src_stride..];
        let d = &mut rgb[out_y * rw * 3..out_y * rw * 3 + rw * 3];
        for out_x in 0..rw {
            let p = &s[(rx + out_x) * 4..(rx + out_x) * 4 + 4]; // B, G, R, X
            let q = &mut d[out_x * 3..out_x * 3 + 3];
            q[0] = p[2]; // R
            q[1] = p[1]; // G
            q[2] = p[0]; // B
        }
    }

    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(Cursor::new(&mut out), rw as u32, rh as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        // Favour encode speed over file size — a screenshot is written once and
        // the worker thread should finish quickly even for a 4K capture.
        enc.set_compression(png::Compression::Fast);
        let mut writer = enc.write_header()?;
        writer.write_image_data(&rgb)?;
    } // writer dropped here — flushes IDAT/IEND into `out`
    Ok(out)
}

/// Encode a per-window capture read-back as an RGBA PNG.
///
/// `src` is `width * height * 4` bytes in **R, G, B, A** order with
/// **premultiplied** alpha (the renderer composites into a transparent
/// `Abgr8888` offscreen). `copy_framebuffer` hands rows back **top-down**
/// (row 0 = top — same as the screenshot read-back, which also doesn't
/// reverse), so we copy straight; colours are un-premultiplied so translucent
/// windows / rounded corners don't darken. Used by the IPC `capture-window`
/// command.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "width/height are non-negative pixel counts bounded by output size, well within usize/u32"
)]
pub(crate) fn encode_rgba(
    src: &[u8],
    width: i32,
    height: i32,
) -> Result<Vec<u8>, png::EncodingError> {
    let cols = width.max(0) as usize;
    let rows = height.max(0) as usize;
    let stride = cols * 4;
    let mut data = vec![0u8; cols * rows * 4];
    for out_y in 0..rows {
        let s_off = out_y * stride;
        if s_off + stride > src.len() {
            continue;
        }
        let src_row = &src[s_off..s_off + stride];
        let dst_row = &mut data[out_y * stride..out_y * stride + stride];
        for col in 0..cols {
            let px = &src_row[col * 4..col * 4 + 4];
            let alpha = px[3];
            let out_px = &mut dst_row[col * 4..col * 4 + 4];
            if alpha == 0 || alpha == 255 {
                out_px.copy_from_slice(px);
            } else {
                // un-premultiply: straight = premul * 255 / alpha (rounded)
                let unp = |c: u8| {
                    ((u32::from(c) * 255 + u32::from(alpha) / 2) / u32::from(alpha)).min(255) as u8
                };
                out_px[0] = unp(px[0]);
                out_px[1] = unp(px[1]);
                out_px[2] = unp(px[2]);
                out_px[3] = alpha;
            }
        }
    }

    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(Cursor::new(&mut out), cols as u32, rows as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header()?;
        writer.write_image_data(&data)?;
    }
    Ok(out)
}

/// Convert a captured BGRX read-back into a fully-opaque RGBA buffer for
/// uploading as the freeze backdrop texture. The read-back is already
/// top-down (natural row order); alpha is forced to 255 (the captured X
/// byte is undefined, not real alpha) so the backdrop is opaque — same
/// shape as the cursor sprite upload, the renderer's known-good
/// `Abgr8888` / `flipped = false` path. Empty if `src` is too small.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "width/height are non-negative physical pixel counts bounded by output size, well within usize"
)]
pub(crate) fn to_rgba_topdown(src: &[u8], width: u32, height: u32) -> Vec<u8> {
    let cols = width as usize;
    let rows = height as usize;
    let stride = cols * 4;
    if stride == 0 || src.len() < stride * rows {
        return Vec::new();
    }
    let mut out = vec![0u8; stride * rows];
    for row in 0..rows {
        let src_line = &src[row * stride..row * stride + stride];
        let dst_line = &mut out[row * stride..row * stride + stride];
        for col in 0..cols {
            let i = col * 4; // pixel offset within the row
            dst_line[i] = src_line[i + 2]; // R <- src R
            dst_line[i + 1] = src_line[i + 1]; // G
            dst_line[i + 2] = src_line[i]; // B <- src B
            dst_line[i + 3] = 255; // opaque
        }
    }
    out
}

/// `Screenshot_YYYYMMDD_HHMMSS.png` at the current time in `offset`
/// (captured once at startup; see `State::local_offset`).
pub(crate) fn timestamp_filename(offset: UtcOffset) -> String {
    let now = OffsetDateTime::now_utc().to_offset(offset);
    let fmt = format_description!("[year][month][day]_[hour][minute][second]");
    let stamp = now
        .format(&fmt)
        .unwrap_or_else(|_| "00000000_000000".to_owned());
    format!("Screenshot_{stamp}.png")
}

/// Expand a configured save directory: a leading `~` becomes `$HOME`, and
/// `$VAR` / `${VAR}` are substituted from the environment (empty if unset).
/// Lets the config use `~/Pictures/Screenshots` or
/// `$XDG_PICTURES_DIR/Screenshots` directly.
pub(crate) fn expand_dir(path: &Path) -> PathBuf {
    PathBuf::from(expand(&path.to_string_lossy()))
}

fn expand(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let rest = match input.strip_prefix('~') {
        Some(after) if after.is_empty() || after.starts_with('/') => {
            if let Ok(home) = std::env::var("HOME") {
                out.push_str(&home);
            }
            after
        }
        _ => input,
    };
    let mut chars = rest.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '$' {
            out.push(c);
            continue;
        }
        let braced = chars.peek() == Some(&'{');
        if braced {
            chars.next();
        }
        let mut name = String::new();
        while let Some(&n) = chars.peek() {
            let part_of_name = if braced {
                n != '}'
            } else {
                n.is_ascii_alphanumeric() || n == '_'
            };
            if part_of_name {
                name.push(n);
                chars.next();
            } else {
                break;
            }
        }
        if braced && chars.peek() == Some(&'}') {
            chars.next();
        }
        if let Ok(val) = std::env::var(&name) {
            out.push_str(&val);
        }
    }
    out
}

/// Create `dir` (and parents) and write `bytes` to `dir/filename`,
/// never clobbering an existing file: the timestamped name only has
/// second resolution, so two captures in the same second would
/// otherwise silently overwrite each other. On a collision a `_N`
/// counter is inserted before the extension (`create_new` makes the
/// existence check race-free).
pub(crate) fn save(dir: &Path, filename: &str, bytes: &[u8]) -> std::io::Result<PathBuf> {
    use std::io::Write as _;

    std::fs::create_dir_all(dir)?;
    let (stem, ext) = filename
        .rsplit_once('.')
        .map_or((filename, None), |(s, e)| (s, Some(e)));
    for n in 0..100u32 {
        let candidate = match (n, ext) {
            (0, _) => dir.join(filename),
            (n, Some(ext)) => dir.join(format!("{stem}_{n}.{ext}")),
            (n, None) => dir.join(format!("{stem}_{n}")),
        };
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut file) => {
                file.write_all(bytes)?;
                return Ok(candidate);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::other(format!(
        "gave up finding a free name for {filename} after 100 collisions"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two captures with the same timestamped filename (same second) must
    /// both survive — the second gets a `_1` suffix instead of silently
    /// overwriting the first.
    #[test]
    fn save_never_clobbers() {
        let dir = std::env::temp_dir().join(format!("libreland-save-test-{}", std::process::id()));
        let a = save(&dir, "Screenshot_20260708_120000.png", b"first").unwrap();
        let b = save(&dir, "Screenshot_20260708_120000.png", b"second").unwrap();
        let c = save(&dir, "Screenshot_20260708_120000.png", b"third").unwrap();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_eq!(std::fs::read(&a).unwrap(), b"first");
        assert_eq!(std::fs::read(&b).unwrap(), b"second");
        assert_eq!(std::fs::read(&c).unwrap(), b"third");
        assert_eq!(b.file_name().unwrap(), "Screenshot_20260708_120000_1.png");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
