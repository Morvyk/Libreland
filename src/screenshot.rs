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

    // No annotations: OCR reads the screenshot, not the drawing on it.
    let png = encode_region(src, src_w, src_h, region, &[])
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
    /// Drag to set the pen width.
    Width,
    /// Run OCR over the selection and put the text on the clipboard.
    CopyText,
    /// Abandon the session.
    Cancel,
}

/// The pen colour, held as hue/saturation/value rather than as RGB.
///
/// HSV because that is the space the picker is *shaped* like — a
/// saturation/value plane under a hue strip — so the widget's geometry
/// and the stored value are the same two numbers, with no round-trip to
/// lose. Every component is `0.0..=1.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Hsv {
    pub(crate) h: f32,
    pub(crate) s: f32,
    pub(crate) v: f32,
}

impl Default for Hsv {
    /// Red, which is what an annotation wants nine times in ten. Value
    /// just under 1.0: pure #ff0000 reads as harsh against a screenshot.
    fn default() -> Self {
        Self { h: 0.0, s: 0.85, v: 0.95 }
    }
}

impl Hsv {
    /// Straight sRGB components, matching the picker's own shader so the
    /// swatch, the stroke on screen and the ink in the saved file are all
    /// the same colour.
    pub(crate) fn rgb(self) -> [f32; 3] {
        // Hue in sixths: which of the six colour sectors, and how far
        // through it. Saturation and value are plain fractions.
        let sixths = self.h.rem_euclid(1.0) * 6.0;
        let sat = self.s.clamp(0.0, 1.0);
        let val = self.v.clamp(0.0, 1.0);
        let sector = sixths.floor();
        let frac = sixths - sector;
        let down = val * (1.0 - sat);
        let falling = val * frac.mul_add(-sat, 1.0);
        let rising = val * (1.0 - frac).mul_add(-sat, 1.0);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "sector is floor(h*6) with h in 0..1, so 0..=5"
        )]
        match sector as u32 % 6 {
            0 => [val, rising, down],
            1 => [falling, val, down],
            2 => [down, val, rising],
            3 => [down, falling, val],
            4 => [rising, down, val],
            _ => [val, down, falling],
        }
    }
}

/// Which part of the colour picker a press landed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PickerPart {
    /// The saturation/value plane: x is saturation, y is value inverted
    /// (bright at the top, which is the way every picker does it).
    Plane,
    /// The hue strip down the side.
    Hue,
}

/// Picker metrics, in compositor pixels.
pub(crate) const PICKER_PLANE: i32 = 148;
pub(crate) const PICKER_HUE_W: i32 = 20;
pub(crate) const PICKER_PAD: i32 = 8;

/// Where the colour picker sits for a toolbar at `bar`, and the rects of
/// its two parts: `(panel, plane, hue)`.
///
/// Directly above the toolbar, or below it when the toolbar is high
/// enough on screen that above would not fit. Left-aligned with the bar,
/// then clamped, so it never hangs off the edge.
pub(crate) fn picker_layout(
    bar: Rectangle<i32, Physical>,
    bounds: Rectangle<i32, Physical>,
) -> (
    Rectangle<i32, Physical>,
    Rectangle<i32, Physical>,
    Rectangle<i32, Physical>,
) {
    use smithay::utils::{Point, Size};
    let w = PICKER_PLANE + PICKER_PAD + PICKER_HUE_W + 2 * PICKER_PAD;
    let h = PICKER_PLANE + 2 * PICKER_PAD;
    let above = bar.loc.y - PICKER_PAD - h;
    let y = if above >= bounds.loc.y {
        above
    } else {
        (bar.loc.y + bar.size.h + PICKER_PAD)
            .min((bounds.loc.y + bounds.size.h - h).max(bounds.loc.y))
    };
    let x = bar
        .loc
        .x
        .clamp(bounds.loc.x, (bounds.loc.x + bounds.size.w - w).max(bounds.loc.x));
    let panel = Rectangle::new(Point::from((x, y)), Size::from((w, h)));
    let plane = Rectangle::new(
        Point::from((x + PICKER_PAD, y + PICKER_PAD)),
        Size::from((PICKER_PLANE, PICKER_PLANE)),
    );
    let hue = Rectangle::new(
        Point::from((plane.loc.x + PICKER_PLANE + PICKER_PAD, y + PICKER_PAD)),
        Size::from((PICKER_HUE_W, PICKER_PLANE)),
    );
    (panel, plane, hue)
}

/// Which part of the picker `pos` is over, if any.
pub(crate) fn picker_hit(
    bar: Rectangle<i32, Physical>,
    bounds: Rectangle<i32, Physical>,
    pos: (f64, f64),
) -> Option<PickerPart> {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "cursor coords are clamped to the i32 layout bounds"
    )]
    let (px, py) = (pos.0.round() as i32, pos.1.round() as i32);
    let inside = |r: Rectangle<i32, Physical>| {
        px >= r.loc.x && px < r.loc.x + r.size.w && py >= r.loc.y && py < r.loc.y + r.size.h
    };
    let (panel, plane, hue) = picker_layout(bar, bounds);
    if !inside(panel) {
        return None;
    }
    if inside(plane) {
        return Some(PickerPart::Plane);
    }
    inside(hue).then_some(PickerPart::Hue)
}

/// The colour a press at `pos` on `part` asks for, starting from `from`
/// so the untouched components are carried through.
///
/// Clamped rather than wrapped: dragging off the top of the hue strip
/// pins at red instead of leaping to magenta, which is what "I have gone
/// as far as I meant to" should do.
pub(crate) fn picker_colour(
    bar: Rectangle<i32, Physical>,
    bounds: Rectangle<i32, Physical>,
    part: PickerPart,
    pos: (f64, f64),
    from: Hsv,
) -> Hsv {
    let (_, plane, hue) = picker_layout(bar, bounds);
    let frac = |v: f64, lo: i32, span: i32| {
        #[allow(clippy::cast_possible_truncation, reason = "a 0..1 fraction")]
        let f = ((v - f64::from(lo)) / f64::from(span.max(1))).clamp(0.0, 1.0) as f32;
        f
    };
    match part {
        PickerPart::Plane => Hsv {
            s: frac(pos.0, plane.loc.x, plane.size.w),
            // Bright at the top, which is how every picker is drawn.
            v: 1.0 - frac(pos.1, plane.loc.y, plane.size.h),
            ..from
        },
        PickerPart::Hue => Hsv {
            h: frac(pos.1, hue.loc.y, hue.size.h),
            ..from
        },
    }
}

/// One button: what it does and where it is.
pub(crate) type ToolSlot = (Tool, Rectangle<i32, Physical>);

/// One freehand annotation: a polyline in **compositor** coordinates, so
/// it survives the selection being moved or resized underneath it. Points
/// are stored as drawn and thinned only by the input path.
#[derive(Debug, Clone)]
pub(crate) struct Stroke {
    pub(crate) colour: [f32; 3],
    pub(crate) width: i32,
    pub(crate) points: Vec<(i32, i32)>,
}

impl Stroke {
    /// Convert from absolute compositor coordinates into one output's
    /// framebuffer pixels — the space the saved crop lives in.
    ///
    /// Strokes are stored in compositor coordinates so they stay put when
    /// the selection moves under them, but the image they are painted
    /// into is physical. Skipping this is why annotations came out
    /// crowded into the top-left of the saved file and running off it:
    /// at scale 1.5 every offset was two-thirds of what it should have
    /// been, and the output origin was never subtracted at all.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "screen-sized pixel counts, scaled by a small factor"
    )]
    pub(crate) fn to_physical(
        &self,
        origin: smithay::utils::Point<i32, Physical>,
        scale: f64,
    ) -> Self {
        let map = |v: i32, o: i32| (f64::from(v - o) * scale).round() as i32;
        Self {
            colour: self.colour,
            // The pen is a width on screen, so it scales with everything
            // else — a 4 px stroke on a 1.5x display is 6 px of image.
            width: ((f64::from(self.width) * scale).round() as i32).max(1),
            points: self
                .points
                .iter()
                .map(|(x, y)| (map(*x, origin.x), map(*y, origin.y)))
                .collect(),
        }
    }
}

/// Pen width bounds, in compositor pixels. The floor is 1 because a
/// hairline is a legitimate thing to want; the ceiling is where a stroke
/// stops annotating a screenshot and starts hiding it.
pub(crate) const PEN_MIN: i32 = 1;
pub(crate) const PEN_MAX: i32 = 24;
pub(crate) const PEN_DEFAULT: i32 = 4;

/// Width of the pen-width slider's slot on the toolbar.
pub(crate) const SLIDER_W: i32 = 96;

/// Where a pen width sits on the slider, as a 0..1 fraction.
pub(crate) fn pen_fraction(width: i32) -> f32 {
    let span = (PEN_MAX - PEN_MIN).max(1);
    f32::from(i16::try_from(width.clamp(PEN_MIN, PEN_MAX) - PEN_MIN).unwrap_or(0))
        / f32::from(i16::try_from(span).unwrap_or(1))
}

/// The pen width a press at `x` on a slider occupying `slot` asks for.
/// Clamped, so dragging past either end pins rather than wrapping.
pub(crate) fn pen_width_at(slot: Rectangle<i32, Physical>, x: i32) -> i32 {
    let span = (slot.size.w - 1).max(1);
    let t = f64::from((x - slot.loc.x).clamp(0, span)) / f64::from(span);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the product is bounded by PEN_MAX"
    )]
    let w = PEN_MIN + (t * f64::from(PEN_MAX - PEN_MIN)).round() as i32;
    w.clamp(PEN_MIN, PEN_MAX)
}

/// Toolbar metrics, in compositor pixels.
pub(crate) const TOOL_SIZE: i32 = 32;
pub(crate) const TOOL_PAD: i32 = 6;
/// Gap between the selection and the toolbar.
pub(crate) const TOOL_GAP: i32 = 10;

/// The buttons, left to right.
///
/// The bar's contents never change.
///
/// They did briefly — the pen's settings were added and removed with the
/// pen — and that broke the pen. The bar is centred on the selection, so
/// growing it by a slider's width slid every button half that distance
/// sideways, and the pen you had just pressed moved out from under the
/// pointer before you could press it again. A toggle you cannot press
/// twice is not a toggle.
///
/// What appears with the pen is the colour picker, in its own panel
/// anchored above the bar, where it can come and go without moving
/// anything you might be about to click.
pub(crate) fn tools() -> Vec<Tool> {
    vec![
        Tool::Take,
        Tool::Draw,
        Tool::Width,
        Tool::CopyText,
        Tool::Cancel,
    ]
}

/// How wide a tool's slot is. Everything is a square button except the
/// width slider, which needs a track long enough to aim along.
pub(crate) fn tool_width(tool: Tool) -> i32 {
    match tool {
        Tool::Width => SLIDER_W,
        _ => TOOL_SIZE,
    }
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
    let bar_w: i32 = tools.iter().map(|t| tool_width(*t)).sum::<i32>() + (count + 1) * TOOL_PAD;
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
    let mut x = bar_x + TOOL_PAD;
    for tool in tools {
        let w = tool_width(tool);
        slots.push((
            tool,
            Rectangle::new(
                Point::from((x, bar_y + TOOL_PAD)),
                Size::from((w, TOOL_SIZE)),
            ),
        ));
        x += w + TOOL_PAD;
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
    strokes: &[Stroke],
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

    for stroke in strokes {
        paint_stroke(&mut rgb, rw, rh, stroke, region.loc);
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

/// Blend one annotation stroke into a cropped RGB buffer.
///
/// The same distance-to-segment coverage the GPU preview uses, so what is
/// saved matches what was on screen. Done on the CPU because it happens
/// once, on a worker thread, after the session is over — there is no
/// frame to keep up with.
///
/// `origin` is the crop's top-left in compositor coordinates, which is
/// what turns the stroke's absolute points into image-local ones.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "crop dimensions are output-bounded pixel counts"
)]
fn paint_stroke(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    stroke: &Stroke,
    origin: smithay::utils::Point<i32, Physical>,
) {
    if stroke.points.is_empty() || width == 0 || height == 0 {
        return;
    }
    let pts: Vec<(f32, f32)> = stroke
        .points
        .iter()
        .map(|(x, y)| ((x - origin.x) as f32, (y - origin.y) as f32))
        .collect();
    let half = stroke.width.max(1) as f32 * 0.5;
    // Only the rows and columns the stroke can reach.
    let pad = half.ceil() as i32 + 2;
    let (mut x0, mut y0) = (i32::MAX, i32::MAX);
    let (mut x1, mut y1) = (i32::MIN, i32::MIN);
    for (px, py) in &pts {
        x0 = x0.min(*px as i32);
        y0 = y0.min(*py as i32);
        x1 = x1.max(*px as i32);
        y1 = y1.max(*py as i32);
    }
    let xs = (x0 - pad).max(0) as usize..((x1 + pad).max(0) as usize).min(width);
    let ys = (y0 - pad).max(0) as usize..((y1 + pad).max(0) as usize).min(height);

    let dist = |p: (f32, f32)| -> f32 {
        let mut best = f32::MAX;
        // A lone point is a dot: treated as a zero-length segment, whose
        // clamped projection gives the same round cap the shader draws.
        if pts.len() == 1 {
            let d = (p.0 - pts[0].0).hypot(p.1 - pts[0].1);
            return d;
        }
        for seg in pts.windows(2) {
            let (from, to) = (seg[0], seg[1]);
            let along = (to.0 - from.0, to.1 - from.1);
            let len2 = along.0.mul_add(along.0, along.1 * along.1).max(1e-6);
            let t = (((p.0 - from.0) * along.0 + (p.1 - from.1) * along.1) / len2)
                .clamp(0.0, 1.0);
            let near = (t.mul_add(along.0, from.0), t.mul_add(along.1, from.1));
            best = best.min((p.0 - near.0).hypot(p.1 - near.1));
        }
        best
    };

    for y in ys {
        for x in xs.clone() {
            let d = dist((x as f32 + 0.5, y as f32 + 0.5));
            // One pixel of feather each side, matching SEGMENT_SHADER.
            let cov = (1.0 - ((d - (half - 0.5)) / 1.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            if cov <= 0.0 {
                continue;
            }
            let px = (y * width + x) * 3;
            for c in 0..3 {
                let src = f32::from(rgb[px + c]) / 255.0;
                let ink = stroke.colour[c];
                rgb[px + c] = (cov.mul_add(ink - src, src) * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
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

    /// The HSV→RGB here has to match the picker shader's, or the colour
    /// you point at, the stroke on screen and the ink in the saved file
    /// are three different colours.
    #[test]
    fn hsv_hits_the_primaries_and_the_greys() {
        let at = |h: f32, sat: f32, val: f32| Hsv { h, s: sat, v: val }.rgb();
        let near = |got: [f32; 3], want: [f32; 3], what: &str| {
            for i in 0..3 {
                assert!(
                    (got[i] - want[i]).abs() < 1e-3,
                    "{what}: got {got:?}, want {want:?}"
                );
            }
        };
        near(at(0.0, 1.0, 1.0), [1.0, 0.0, 0.0], "red");
        near(at(1.0 / 3.0, 1.0, 1.0), [0.0, 1.0, 0.0], "green");
        near(at(2.0 / 3.0, 1.0, 1.0), [0.0, 0.0, 1.0], "blue");
        // Zero saturation is grey at whatever value, whatever the hue.
        for h in [0.0_f32, 0.25, 0.8] {
            near(at(h, 0.0, 0.5), [0.5, 0.5, 0.5], "grey");
        }
        near(at(0.5, 1.0, 0.0), [0.0, 0.0, 0.0], "value 0 is black");
        // Hue wraps rather than clamping — 1.0 is 0.0 again.
        near(at(1.0, 1.0, 1.0), at(0.0, 1.0, 1.0), "hue wraps");
    }

    /// Every corner of the picker is reachable, and the plane is drawn
    /// bright-at-the-top so the value axis has to be inverted.
    #[test]
    fn the_picker_reaches_every_corner() {
        let bar = Rectangle::new(Point::from((200, 900)), Size::from((300, 44)));
        let bounds = Rectangle::new(Point::from((0, 0)), Size::from((2560, 1440)));
        let (_, plane, hue) = picker_layout(bar, bounds);
        let from = Hsv::default();
        let pick = |part, x: i32, y: i32| {
            picker_colour(bar, bounds, part, (f64::from(x), f64::from(y)), from)
        };

        // Top-left of the plane: no saturation, full value — white.
        let tl = pick(PickerPart::Plane, plane.loc.x, plane.loc.y);
        assert!((tl.s - 0.0).abs() < 1e-3 && (tl.v - 1.0).abs() < 1e-3, "{tl:?}");
        // Bottom-right: fully saturated, no value — black.
        let br = pick(
            PickerPart::Plane,
            plane.loc.x + plane.size.w,
            plane.loc.y + plane.size.h,
        );
        assert!((br.s - 1.0).abs() < 1e-3 && (br.v - 0.0).abs() < 1e-3, "{br:?}");
        // The plane never touches the hue, and the strip never touches
        // saturation or value — each axis is independent.
        assert!((tl.h - from.h).abs() < 1e-6);
        let far = pick(PickerPart::Hue, hue.loc.x, hue.loc.y + hue.size.h);
        assert!((far.h - 1.0).abs() < 1e-3, "the far end of the strip: {far:?}");
        assert!((far.s - from.s).abs() < 1e-6 && (far.v - from.v).abs() < 1e-6);

        // Dragging past an edge pins rather than wrapping round.
        let past = pick(PickerPart::Hue, hue.loc.x, hue.loc.y - 10_000_i32);
        assert!((past.h - 0.0).abs() < 1e-6, "{past:?}");
    }

    /// The bar must not move when the pen is toggled. It is centred on
    /// the selection, so anything that changes its width slides every
    /// button sideways — and the pen sliding out from under the pointer
    /// is exactly what made it impossible to press twice.
    #[test]
    fn the_toolbar_does_not_move_when_the_pen_toggles() {
        let sel = Rectangle::new(Point::from((300, 200)), Size::from((400, 300)));
        let bounds = Rectangle::new(Point::from((0, 0)), Size::from((1920, 1080)));
        let (bar, slots) = toolbar_layout(sel, bounds);
        // The layout takes no notion of pen state at all, which is the
        // strongest form of "it cannot move": there is nothing to vary.
        let pen = slots
            .iter()
            .find(|(t, _)| *t == Tool::Draw)
            .map(|(_, r)| *r)
            .expect("the pen is always on the bar");
        let (bar2, slots2) = toolbar_layout(sel, bounds);
        assert_eq!(bar, bar2);
        assert_eq!(
            slots2.iter().find(|(t, _)| *t == Tool::Draw).map(|(_, r)| *r),
            Some(pen)
        );
        for always in tools() {
            assert!(
                slots.iter().any(|(t, _)| *t == always),
                "{always:?} is missing from the bar"
            );
        }
    }

    /// Strokes live in compositor coordinates and are painted into a
    /// *physical* crop, so they have to be converted — both the offset
    /// from the output origin and the scale. Missing the scale put every
    /// annotation two-thirds of the way toward the top-left of the saved
    /// file on a 1.5x display, with the far end off the edge.
    #[test]
    fn strokes_convert_from_compositor_space_to_framebuffer_pixels() {
        let stroke = Stroke {
            colour: [1.0, 0.0, 0.0],
            width: 4,
            points: vec![(100, 200), (300, 400)],
        };
        // Output at the origin, 1.5x: pure scale.
        let s = stroke.to_physical(Point::from((0, 0)), 1.5);
        assert_eq!(s.points, vec![(150, 300), (450, 600)]);
        assert_eq!(s.width, 6, "the pen is a width on screen, so it scales too");

        // Output offset in the compositor layout: the origin comes off
        // *before* the scale, exactly as `compositor_rect_to_physical`
        // does it for the crop the stroke is painted into.
        let s = stroke.to_physical(Point::from((100, 200)), 2.0);
        assert_eq!(s.points, vec![(0, 0), (400, 400)]);

        // 1.0 is the identity, so an unscaled display is untouched.
        let s = stroke.to_physical(Point::from((0, 0)), 1.0);
        assert_eq!(s.points, stroke.points);
        assert_eq!(s.width, stroke.width);

        // A hairline never rounds away to nothing.
        let hair = Stroke { width: 1, ..stroke };
        assert!(hair.to_physical(Point::from((0, 0)), 0.5).width >= 1);
    }

    /// The slider maps its full track onto the full pen range, both ends
    /// included — a slider you cannot drag to its own maximum is a bug
    /// people notice immediately.
    #[test]
    fn the_width_slider_reaches_both_ends() {
        let slot = Rectangle::new(Point::from((100, 50)), Size::from((SLIDER_W, TOOL_SIZE)));
        assert_eq!(pen_width_at(slot, 100), PEN_MIN);
        assert_eq!(pen_width_at(slot, 100 + SLIDER_W - 1), PEN_MAX);
        // Past either end pins rather than wrapping.
        assert_eq!(pen_width_at(slot, -5000), PEN_MIN);
        assert_eq!(pen_width_at(slot, 5000), PEN_MAX);
        // And it is monotonic across the track.
        let mut last = 0;
        for x in 0..SLIDER_W {
            let w = pen_width_at(slot, 100 + x);
            assert!(w >= last, "width went backwards at x={x}");
            last = w;
        }
    }

    /// The knob's position and the width it means have to agree, or the
    /// control lies about what it is set to.
    #[test]
    fn the_slider_fraction_round_trips() {
        let slot = Rectangle::new(Point::from((0, 0)), Size::from((SLIDER_W, TOOL_SIZE)));
        for w in [PEN_MIN, PEN_DEFAULT, 12, PEN_MAX] {
            let frac = pen_fraction(w);
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let x = (frac * f32::from(i16::try_from(SLIDER_W - 1).unwrap_or(1))).round() as i32;
            assert_eq!(pen_width_at(slot, x), w, "width {w} moved when drawn");
        }
    }

    /// The toolbar's slots must not overlap, and must all sit inside the
    /// bar — the slider is a different width from the buttons, which is
    /// exactly the kind of thing a layout loop gets wrong.
    #[test]
    fn toolbar_slots_tile_the_bar_without_overlapping() {
        let sel = Rectangle::new(Point::from((300, 200)), Size::from((400, 300)));
        let bounds = Rectangle::new(Point::from((0, 0)), Size::from((1920, 1080)));
        let (bar, slots) = toolbar_layout(sel, bounds);
        assert!(!slots.is_empty());
        let mut prev_right = bar.loc.x;
        for (tool, r) in &slots {
            assert!(r.loc.x >= prev_right, "slot for {tool:?} overlaps its left neighbour");
            assert!(r.loc.y >= bar.loc.y && r.loc.y + r.size.h <= bar.loc.y + bar.size.h);
            assert_eq!(r.size.w, tool_width(*tool));
            prev_right = r.loc.x + r.size.w;
        }
        assert!(prev_right <= bar.loc.x + bar.size.w, "last slot runs off the bar");
    }

    /// A toolbar that would fall off the bottom of the screen goes above
    /// the selection instead, and one that fits neither is still on
    /// screen. A control you cannot reach is worse than none.
    #[test]
    fn the_toolbar_stays_on_screen() {
        let bounds = Rectangle::new(Point::from((0, 0)), Size::from((1920, 1080)));
        for sel in [
            // Hard against the bottom: must flip above.
            Rectangle::new(Point::from((100, 900)), Size::from((400, 179))),
            // Hard against the top: must stay below.
            Rectangle::new(Point::from((100, 0)), Size::from((400, 200))),
            // Full height: fits neither side.
            Rectangle::new(Point::from((100, 0)), Size::from((400, 1080))),
            // Hard against the right edge: must clamp left.
            Rectangle::new(Point::from((1800, 400)), Size::from((119, 200))),
        ] {
            let (bar, _) = toolbar_layout(sel, bounds);
            assert!(bar.loc.x >= 0, "{sel:?} pushed the bar off the left");
            assert!(
                bar.loc.x + bar.size.w <= bounds.size.w,
                "{sel:?} pushed the bar off the right"
            );
            assert!(bar.loc.y >= 0, "{sel:?} pushed the bar off the top");
            assert!(
                bar.loc.y + bar.size.h <= bounds.size.h,
                "{sel:?} pushed the bar off the bottom"
            );
        }
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
