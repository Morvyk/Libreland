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

/// Recognise the text in `region` of a captured frame and return it.
///
/// Runs `tesseract` over a PNG of the crop. Out of process on purpose:
/// it keeps the compositor's build pure Rust and, more to the point,
/// keeps a crash in a large C++ library from taking the desktop with it —
/// an OCR failure should cost you a clipboard copy, not your session.
///
/// English is fixed. Recognition accuracy depends on being told the
/// language, "whatever the locale says" is usually not what is on the
/// screen, and one right answer beats a configurable wrong one.
///
/// Errors when tesseract isn't installed, which is the expected case on
/// a machine that has never wanted this — the caller logs and moves on.
pub(crate) fn read_text(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    region: Rectangle<i32, Physical>,
) -> std::io::Result<String> {
    use std::io::{Error, ErrorKind, Write};
    use std::process::{Command, Stdio};

    let png = encode_region(src, src_w, src_h, region)
        .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
    // `-` for both paths: PNG in on stdin, text out on stdout, so nothing
    // is written to disk and there is no temp file to leak or collide.
    let mut child = Command::new("tesseract")
        .args(["stdin", "stdout", "-l", "eng"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| {
            Error::new(
                e.kind(),
                format!("running tesseract (is it installed?): {e}"),
            )
        })?;
    child
        .stdin
        .take()
        .ok_or_else(|| Error::new(ErrorKind::BrokenPipe, "tesseract stdin"))?
        .write_all(&png)?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(Error::other(format!("tesseract exited {}", out.status)));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// A button on the options toolbar that appears once a selection is
/// committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tool {
    /// Take the shot: save and/or copy per the bind.
    Take,
    /// Toggle freehand annotation. While on, dragging inside the
    /// selection draws instead of moving it.
    Draw,
    /// Pick the annotation colour.
    Colour(PenColour),
    /// Run OCR over the selection and put the text on the clipboard.
    CopyText,
    /// Abandon the session.
    Cancel,
}

/// The annotation palette. Deliberately short: a screenshot annotation is
/// an arrow and a circle, not a painting, and a five-swatch row can be
/// hit without aiming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PenColour {
    /// The default, and the one nearly every annotation wants.
    Red,
    Yellow,
    Green,
    Blue,
    White,
}

impl PenColour {
    /// Every colour, in toolbar order.
    pub(crate) const ALL: [Self; 5] = [
        Self::Red,
        Self::Yellow,
        Self::Green,
        Self::Blue,
        Self::White,
    ];

    /// Linear-ish sRGB components, chosen to stay legible on both a dark
    /// screenshot and a light one.
    pub(crate) fn rgb(self) -> [f32; 3] {
        match self {
            Self::Red => [0.91, 0.20, 0.20],
            Self::Yellow => [0.98, 0.79, 0.19],
            Self::Green => [0.30, 0.80, 0.35],
            Self::Blue => [0.25, 0.62, 1.00],
            Self::White => [0.97, 0.97, 0.97],
        }
    }
}

/// One button: what it does and where it is.
pub(crate) type ToolSlot = (Tool, Rectangle<i32, Physical>);

/// Toolbar metrics, in compositor pixels.
pub(crate) const TOOL_SIZE: i32 = 32;
pub(crate) const TOOL_PAD: i32 = 6;
/// Gap between the selection and the toolbar.
pub(crate) const TOOL_GAP: i32 = 10;

/// The buttons, left to right.
pub(crate) fn tools() -> Vec<Tool> {
    let mut v = vec![Tool::Take, Tool::Draw];
    v.extend(PenColour::ALL.map(Tool::Colour));
    v.push(Tool::CopyText);
    v.push(Tool::Cancel);
    v
}

/// Where the toolbar sits for a selection of `sel` on an output of
/// `bounds`, and where each button lands inside it.
///
/// Below the selection by preference, above it when there is no room
/// below, and inside it when there is room for neither — a selection
/// taller than the screen is rare but a toolbar you cannot reach is
/// useless. Always clamped horizontally so the last button stays on
/// screen.
pub(crate) fn toolbar_layout(
    sel: Rectangle<i32, Physical>,
    bounds: Rectangle<i32, Physical>,
) -> (Rectangle<i32, Physical>, Vec<ToolSlot>) {
    use smithay::utils::{Point, Size};
    let tools = tools();
    let count = i32::try_from(tools.len()).unwrap_or(1);
    let bar_w = count * TOOL_SIZE + (count + 1) * TOOL_PAD;
    let bar_h = TOOL_SIZE + 2 * TOOL_PAD;

    let below = sel.loc.y + sel.size.h + TOOL_GAP;
    let above = sel.loc.y - TOOL_GAP - bar_h;
    let bar_y = if below + bar_h <= bounds.loc.y + bounds.size.h {
        below
    } else if above >= bounds.loc.y {
        above
    } else {
        // Neither side fits: sit just inside the selection's bottom edge.
        (sel.loc.y + sel.size.h - bar_h - TOOL_GAP).max(bounds.loc.y)
    };
    // Centred on the selection, then pushed back on screen.
    let bar_x = (sel.loc.x + (sel.size.w - bar_w) / 2).clamp(
        bounds.loc.x,
        (bounds.loc.x + bounds.size.w - bar_w).max(bounds.loc.x),
    );

    let bar = Rectangle::new(
        Point::from((bar_x, bar_y)),
        Size::from((bar_w, bar_h)),
    );
    let mut slots = Vec::with_capacity(tools.len());
    for (i, tool) in tools.into_iter().enumerate() {
        let i = i32::try_from(i).unwrap_or(0);
        slots.push((
            tool,
            Rectangle::new(
                Point::from((
                    bar_x + TOOL_PAD + i * (TOOL_SIZE + TOOL_PAD),
                    bar_y + TOOL_PAD,
                )),
                Size::from((TOOL_SIZE, TOOL_SIZE)),
            ),
        ));
    }
    (bar, slots)
}

/// The tool under `pos`, if any.
pub(crate) fn tool_at(
    sel: Rectangle<i32, Physical>,
    bounds: Rectangle<i32, Physical>,
    pos: (f64, f64),
) -> Option<Tool> {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "cursor coords are clamped to the i32 layout bounds"
    )]
    let (px, py) = (pos.0.round() as i32, pos.1.round() as i32);
    let (_, slots) = toolbar_layout(sel, bounds);
    slots
        .into_iter()
        .find(|(_, r)| {
            px >= r.loc.x && px < r.loc.x + r.size.w && py >= r.loc.y && py < r.loc.y + r.size.h
        })
        .map(|(t, _)| t)
}

/// Whether `pos` is anywhere on the toolbar — used to keep a press there
/// from being read as "start a new selection".
pub(crate) fn on_toolbar(
    sel: Rectangle<i32, Physical>,
    bounds: Rectangle<i32, Physical>,
    pos: (f64, f64),
) -> bool {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "cursor coords are clamped to the i32 layout bounds"
    )]
    let (px, py) = (pos.0.round() as i32, pos.1.round() as i32);
    let (bar, _) = toolbar_layout(sel, bounds);
    px >= bar.loc.x
        && px < bar.loc.x + bar.size.w
        && py >= bar.loc.y
        && py < bar.loc.y + bar.size.h
}

/// What a press on a committed selection is about to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectionGrab {
    /// Slide the whole rect, size unchanged.
    Move,
    /// Drag the named edges. Both axes for a corner, one for an edge.
    Resize(crate::layout::ResizeEdges),
}

/// How close to an edge a press has to land to grab it, in compositor
/// pixels. Measured on *both* sides of the boundary: a selection is a
/// hairline, so an inside-only band would be as hard to hit as the line.
pub(crate) const HANDLE: i32 = 10;

/// Smallest selection an interactive resize will leave you with. Small
/// enough to crop a word, large enough that the handles don't overlap
/// into ambiguity.
pub(crate) const MIN_SELECTION: i32 = 16;

/// Which part of `rect` a press at `pos` takes hold of.
///
/// Corners win over edges (they are the intersection, and a corner is
/// what you meant if you aimed at one), edges win over the interior, and
/// a press outside the handle band and outside the rect grabs nothing —
/// which is what lets a press on the dimmed backdrop start a fresh
/// selection instead of nudging the old one.
pub(crate) fn grab_at(
    rect: Rectangle<i32, Physical>,
    pos: (f64, f64),
    handle: i32,
) -> Option<SelectionGrab> {
    use crate::layout::{EdgeX, EdgeY, ResizeEdges};
    #[allow(
        clippy::cast_possible_truncation,
        reason = "cursor coords are clamped to the i32 layout bounds"
    )]
    let (px, py) = (pos.0.round() as i32, pos.1.round() as i32);
    let (left, top) = (rect.loc.x, rect.loc.y);
    let (right, bottom) = (left + rect.size.w, top + rect.size.h);
    let near = |v: i32, edge: i32| (v - edge).abs() <= handle;
    // Within the rect's span on the *other* axis, grown by the handle so
    // a corner still reads as a corner from just outside it.
    let spans_x = px >= left - handle && px <= right + handle;
    let spans_y = py >= top - handle && py <= bottom + handle;

    let x = if near(px, left) && spans_y {
        Some(EdgeX::Left)
    } else if near(px, right) && spans_y {
        Some(EdgeX::Right)
    } else {
        None
    };
    let y = if near(py, top) && spans_x {
        Some(EdgeY::Top)
    } else if near(py, bottom) && spans_x {
        Some(EdgeY::Bottom)
    } else {
        None
    };
    if x.is_some() || y.is_some() {
        return Some(SelectionGrab::Resize(ResizeEdges { x, y }));
    }
    let inside = px >= left && px < right && py >= top && py < bottom;
    inside.then_some(SelectionGrab::Move)
}

/// Apply a grab that started with `start` and has travelled `(dx, dy)`.
///
/// A resize that would invert the rect is clamped at [`MIN_SELECTION`]
/// rather than flipped: dragging the left edge past the right one reads
/// as "I have gone too far", not as "I meant the other side".
pub(crate) fn apply_grab(
    start: Rectangle<i32, Physical>,
    grab: SelectionGrab,
    dx: i32,
    dy: i32,
) -> Rectangle<i32, Physical> {
    use crate::layout::{EdgeX, EdgeY};
    use smithay::utils::{Point, Size};
    let SelectionGrab::Resize(edges) = grab else {
        return Rectangle::new(
            Point::from((start.loc.x + dx, start.loc.y + dy)),
            start.size,
        );
    };
    let (mut x, mut w) = (start.loc.x, start.size.w);
    match edges.x {
        Some(EdgeX::Left) => {
            let moved = dx.min(w - MIN_SELECTION);
            x += moved;
            w -= moved;
        }
        Some(EdgeX::Right) => w = (w + dx).max(MIN_SELECTION),
        None => {}
    }
    let (mut y, mut h) = (start.loc.y, start.size.h);
    match edges.y {
        Some(EdgeY::Top) => {
            let moved = dy.min(h - MIN_SELECTION);
            y += moved;
            h -= moved;
        }
        Some(EdgeY::Bottom) => h = (h + dy).max(MIN_SELECTION),
        None => {}
    }
    Rectangle::new(Point::from((x, y)), Size::from((w, h)))
}

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

/// Box-downscale a **premultiplied** RGBA image so its longest side is at
/// most `max`. Never enlarges; returns the input untouched when it already
/// fits (or when `max` is nonsense).
///
/// Premultiplied is what makes a plain average correct here. Straight-alpha
/// pixels would have to be weighted by their alpha first, or a transparent
/// pixel's leftover colour bleeds into its neighbours — which is exactly
/// how a downscaled window with rounded corners grows a dark fringe.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "pixel counts are non-negative and output-bounded; the accumulator is u32 over a bounded box"
)]
pub(crate) fn downscale_rgba(
    src: &[u8],
    width: i32,
    height: i32,
    max: i32,
) -> (i32, i32, Vec<u8>) {
    let longest = width.max(height);
    if max <= 0 || longest <= max || width <= 0 || height <= 0 {
        return (width, height, src.to_vec());
    }
    let scale = f64::from(max) / f64::from(longest);
    let (dw, dh) = (
        ((f64::from(width) * scale).round() as i32).max(1),
        ((f64::from(height) * scale).round() as i32).max(1),
    );
    let (sw, sh) = (width as usize, height as usize);
    let (dwu, dhu) = (dw as usize, dh as usize);
    let mut out = vec![0u8; dwu * dhu * 4];
    for y in 0..dhu {
        // Source rows this destination row averages over, at least one.
        let y0 = y * sh / dhu;
        let y1 = (((y + 1) * sh).div_ceil(dhu)).min(sh).max(y0 + 1);
        for x in 0..dwu {
            let x0 = x * sw / dwu;
            let x1 = (((x + 1) * sw).div_ceil(dwu)).min(sw).max(x0 + 1);
            let mut acc = [0u32; 4];
            let mut n = 0u32;
            for sy in y0..y1 {
                let row = sy * sw * 4;
                for sx in x0..x1 {
                    let px = row + sx * 4;
                    if px + 4 > src.len() {
                        continue;
                    }
                    for c in 0..4 {
                        acc[c] += u32::from(src[px + c]);
                    }
                    n += 1;
                }
            }
            let dst = (y * dwu + x) * 4;
            for c in 0..4 {
                out[dst + c] = acc[c].checked_div(n).unwrap_or(0) as u8;
            }
        }
    }
    (dw, dh, out)
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
    use crate::layout::{EdgeX, EdgeY, ResizeEdges};
    use smithay::utils::{Point, Size};

    fn sel() -> Rectangle<i32, Physical> {
        Rectangle::new(Point::from((100, 100)), Size::from((200, 150)))
    }

    /// The whole selection is grabbable, and each part gives what it looks
    /// like it gives: a corner both axes, an edge one, the middle a move.
    #[test]
    fn every_part_of_a_selection_grabs_what_it_looks_like() {
        let r = sel();
        assert_eq!(
            grab_at(r, (100.0, 100.0), HANDLE),
            Some(SelectionGrab::Resize(ResizeEdges {
                x: Some(EdgeX::Left),
                y: Some(EdgeY::Top)
            }))
        );
        assert_eq!(
            grab_at(r, (300.0, 250.0), HANDLE),
            Some(SelectionGrab::Resize(ResizeEdges {
                x: Some(EdgeX::Right),
                y: Some(EdgeY::Bottom)
            }))
        );
        // Mid-edge: one axis only.
        assert_eq!(
            grab_at(r, (100.0, 175.0), HANDLE),
            Some(SelectionGrab::Resize(ResizeEdges {
                x: Some(EdgeX::Left),
                y: None
            }))
        );
        assert_eq!(grab_at(r, (200.0, 175.0), HANDLE), Some(SelectionGrab::Move));
    }

    /// An edge is grabbable from *outside* it too. A selection is a
    /// hairline; a band that only reached inwards would be as hard to hit
    /// as the line itself.
    #[test]
    fn edges_grab_from_both_sides_and_the_backdrop_grabs_nothing() {
        let r = sel();
        assert!(matches!(
            grab_at(r, (95.0, 175.0), HANDLE),
            Some(SelectionGrab::Resize(_))
        ));
        // Well clear of the rect: nothing, which is what lets a press
        // out there start a fresh selection.
        assert_eq!(grab_at(r, (400.0, 400.0), HANDLE), None);
        assert_eq!(grab_at(r, (100.0, 400.0), HANDLE), None);
    }

    /// Moving keeps the size; resizing keeps the opposite edge pinned.
    #[test]
    fn a_move_translates_and_a_resize_anchors() {
        let r = sel();
        let moved = apply_grab(r, SelectionGrab::Move, 10, -20);
        assert_eq!(moved.loc, Point::from((110, 80)));
        assert_eq!(moved.size, r.size);

        // Dragging the left edge right shrinks the width and holds the
        // right edge exactly where it was.
        let left = apply_grab(
            r,
            SelectionGrab::Resize(ResizeEdges { x: Some(EdgeX::Left), y: None }),
            30,
            0,
        );
        assert_eq!(left.loc.x, 130);
        assert_eq!(left.loc.x + left.size.w, r.loc.x + r.size.w);
        assert_eq!(left.size.h, r.size.h, "an untouched axis doesn't move");
    }

    /// Dragging an edge past its opposite clamps instead of inverting the
    /// rect. "I have gone too far" is what that gesture means; it is not
    /// a request to select the other side of the screen.
    #[test]
    fn a_resize_cannot_turn_the_selection_inside_out() {
        let r = sel();
        for (edges, dx, dy) in [
            (ResizeEdges { x: Some(EdgeX::Left), y: None }, 10_000, 0),
            (ResizeEdges { x: Some(EdgeX::Right), y: None }, -10_000, 0),
            (ResizeEdges { x: None, y: Some(EdgeY::Top) }, 0, 10_000),
            (ResizeEdges { x: None, y: Some(EdgeY::Bottom) }, 0, -10_000),
        ] {
            let out = apply_grab(r, SelectionGrab::Resize(edges), dx, dy);
            assert!(out.size.w >= MIN_SELECTION, "width collapsed: {out:?}");
            assert!(out.size.h >= MIN_SELECTION, "height collapsed: {out:?}");
        }
    }

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
