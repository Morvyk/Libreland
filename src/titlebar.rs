//! Server-side titlebars: geometry, hit regions, and rasterization.
//!
//! A titlebar is one RGBA image per (title, size, focus, button set),
//! rasterized on the CPU here and uploaded as a texture the renderer
//! blits over the top strip of a window's decoration offscreen. One
//! texture rather than a bar quad plus a text quad plus three button
//! quads, because the whole thing changes together and rarely: a title
//! edit or a focus change, not a frame.
//!
//! Geometry and drawing live in the same module on purpose. The button
//! a click lands on and the button that was drawn have to be the same
//! rectangle, and the cheapest way to guarantee that is for both callers
//! — the renderer and the input path — to ask [`bar_layout`].
//!
//! Button glyphs are drawn from distance fields rather than from a font.
//! An icon font is one more thing that can be missing or have the wrong
//! coverage, and "✕" in a UI face is not reliably the glyph anyone wants;
//! shapes made of line segments always look the same and are
//! anti-aliased by construction.
//!
//! The glyph set follows Breeze: chevrons for minimize/maximize/restore
//! and a cross for close. Chevrons rather than the more literal square
//! outline, which at button size is indistinguishable from the box a
//! font draws for a glyph it doesn't have — it reads as a broken icon
//! rather than as "maximize".

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel geometry and 8-bit colour: every value here is a bar-sized non-negative pixel count or a channel already clamped to [0, 255], and the i32/u32/usize/f32/u8 conversions are all well inside that range. Checked conversions at each site would be noise around arithmetic that cannot overflow."
)]

use libreland_text::Fonts;
use smithay::utils::{Physical, Point, Rectangle, Size};

use crate::config::TitlebarButton;
use crate::layout::{Deco, EdgeX, EdgeY, ResizeEdges};

/// Button width as a multiple of the bar height. Wider than tall, the
/// way every desktop draws them — a square button reads as cramped and
/// gives the pointer less to aim at along the axis it travels.
const BUTTON_ASPECT: f32 = 1.35;

/// Padding from the bar's left edge to the title text, and from the
/// rightmost button to the right edge, as a multiple of bar height.
const EDGE_PAD: f32 = 0.45;

/// Glyph extent inside a button, as a multiple of bar height.
const GLYPH: f32 = 0.30;

/// Application-icon side, as a multiple of bar height.
const ICON_SIDE: f32 = 0.62;

/// Stroke half-width of a button glyph, in pixels at scale 1.
const STROKE: f32 = 0.70;

/// How much lighter the top of the bar is than the bottom. Subtle on
/// purpose: enough to give the bar a direction, not enough to read as a
/// gradient in its own right.
const SHEEN: f32 = 0.055;

/// Tint strength of the hovered button's backing.
const HOVER_TINT: f32 = 0.14;

/// What the close button's hover tints toward. The one button worth
/// signalling before it is released.
const CLOSE_HOVER: [f32; 3] = [0.75, 0.18, 0.18];

/// Where the pieces of a titlebar sit, in bar-local pixels.
///
/// Produced by [`bar_layout`] and used by both the rasterizer and the
/// input hit-test, so a click can never land somewhere different from
/// where the glyph was drawn.
#[derive(Debug, Clone)]
pub struct BarLayout {
    /// Buttons left-to-right in draw order, each with its full clickable
    /// rect (the whole cell, not just the glyph).
    pub buttons: Vec<(TitlebarButton, Rectangle<i32, Physical>)>,
    /// Square slot at the left for the application icon.
    pub icon: Rectangle<i32, Physical>,
    /// Space left for the title, between the icon and the buttons.
    pub title: Rectangle<i32, Physical>,
}

impl BarLayout {
    /// The button containing `pos`, given in bar-local pixels.
    pub fn button_at(&self, pos: Point<i32, Physical>) -> Option<TitlebarButton> {
        self.buttons.iter().find_map(|(kind, rect)| {
            let inside = pos.x >= rect.loc.x
                && pos.y >= rect.loc.y
                && pos.x < rect.loc.x + rect.size.w
                && pos.y < rect.loc.y + rect.size.h;
            inside.then_some(*kind)
        })
    }
}

/// Side of the square application-icon slot on a bar `height` px tall.
#[must_use]
pub fn icon_side_for(height: i32) -> i32 {
    ((height.max(0) as f32) * ICON_SIDE) as i32
}

/// Lay out a bar `width` x `height` pixels carrying `buttons`.
///
/// Buttons are right-aligned in the order given, so the last entry in
/// the config is the rightmost — which is what "close is on the right"
/// means when you write `{ "minimize", "maximize", "close" }`.
///
/// A bar too narrow for its buttons drops them from the left (the title
/// is never the thing that survives at the cost of the close button).
#[must_use]
pub fn bar_layout(width: i32, height: i32, buttons: &[TitlebarButton]) -> BarLayout {
    let h = height.max(0);
    let pad = (h as f32 * EDGE_PAD) as i32;
    let bw = (h as f32 * BUTTON_ASPECT) as i32;
    let mut placed: Vec<(TitlebarButton, Rectangle<i32, Physical>)> = Vec::new();
    // Walk right-to-left so the rightmost button is the last configured
    // one, then flip back to draw order.
    let mut right = width - pad;
    for kind in buttons.iter().rev() {
        let left = right - bw;
        if left < pad {
            break;
        }
        placed.push((
            *kind,
            Rectangle::new(Point::from((left, 0)), Size::from((bw, h))),
        ));
        right = left;
    }
    placed.reverse();
    // Icon slot at the left, vertically centred and square.
    let icon_side = icon_side_for(h);
    let icon = Rectangle::new(
        Point::from((pad, (h - icon_side) / 2)),
        Size::from((icon_side, icon_side)),
    );
    let title_left = icon.loc.x + icon.size.w + pad / 2;
    let title_right = placed.first().map_or(width - pad, |(_, r)| r.loc.x);
    let title_w = (title_right - title_left).max(0);
    BarLayout {
        buttons: placed,
        icon,
        title: Rectangle::new(Point::from((title_left, 0)), Size::from((title_w, h))),
    }
}

/// What part of a decorated window a pointer position lands on.
///
/// Resolved once per press (and per motion, for the cursor shape), so
/// the whole "is this a drag, a resize, or the client's problem"
/// decision lives in one place instead of being spread across the
/// button handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// A titlebar button. Acts on *release* over the same button, so a
    /// press can be aborted by moving off it — which is what every
    /// desktop does, and the only reason a mis-aimed click on Close is
    /// recoverable.
    Button(TitlebarButton),
    /// The draggable part of the titlebar: moves the window.
    Titlebar,
    /// Within the grab margin of an edge or corner: resizes the window.
    Resize(ResizeEdges),
    /// The client's own surface. The compositor forwards the press.
    Content,
}

/// Resolve which part of `cell` the position `pos` lands on.
///
/// Order matters and is not arbitrary: the resize margin wins over the
/// titlebar, so the bar's top edge and its top corners still resize.
/// Losing that would make the top edge of every window the one edge you
/// cannot grab.
#[must_use]
pub fn region_at(
    cell: Rectangle<i32, Physical>,
    deco: Deco,
    buttons: &[TitlebarButton],
    resize_zone: i32,
    pos: Point<i32, Physical>,
) -> Region {
    let zone = resize_zone.max(0);
    if zone > 0 {
        // Clamp the margin to under half the window: on a very small
        // window an over-wide margin would otherwise make every pixel a
        // resize and leave nothing to drag or click.
        let zx = zone.min((cell.size.w - 1).max(0) / 2);
        let zy = zone.min((cell.size.h - 1).max(0) / 2);
        let left = pos.x < cell.loc.x + zx;
        let right = pos.x >= cell.loc.x + cell.size.w - zx;
        let top = pos.y < cell.loc.y + zy;
        let bottom = pos.y >= cell.loc.y + cell.size.h - zy;
        let x = if left {
            Some(EdgeX::Left)
        } else if right {
            Some(EdgeX::Right)
        } else {
            None
        };
        let y = if top {
            Some(EdgeY::Top)
        } else if bottom {
            Some(EdgeY::Bottom)
        } else {
            None
        };
        if x.is_some() || y.is_some() {
            return Region::Resize(ResizeEdges { x, y });
        }
    }
    if deco.titlebar > 0 {
        // The bar is blitted over the cell's top strip at (0, 0), full
        // cell width — the border ring is painted *over* it by the
        // composite rather than the bar being inset inside it. Insetting
        // this hit-test by the border instead put the whole button row a
        // border away from where its glyphs actually are.
        let bar_bottom = cell.loc.y + deco.titlebar;
        if pos.y < bar_bottom {
            let local = Point::from((pos.x - cell.loc.x, pos.y - cell.loc.y));
            let bar = bar_layout(cell.size.w, deco.titlebar, buttons);
            return match bar.button_at(local) {
                Some(kind) => Region::Button(kind),
                None => Region::Titlebar,
            };
        }
    }
    Region::Content
}

/// Colours for one bar, already resolved for its focus state. Linear-ish
/// sRGB components in `[0, 1]`, matching the rest of the config.
#[derive(Debug, Clone, Copy)]
pub struct BarStyle {
    pub background: [f32; 3],
    pub text: [f32; 3],
}

/// Neutral dark the bar is mixed *towards*, so an already-dark border
/// colour can't drag the bar to near-black. Breeze's dark titlebar sits
/// around here.
const SURFACE: f32 = 0.105;

impl BarStyle {
    /// Derive a bar's colours from the window border's fill for the same
    /// focus state, so the titlebar and the frame around it are one
    /// palette without a second set of config keys to keep in sync.
    ///
    /// The border colour is *mixed toward* [`SURFACE`] rather than
    /// scaled: a border is a hairline and can be fully saturated, a bar
    /// is a solid area and at that saturation would dominate the window
    /// it belongs to. Scaling was the first attempt and it fails at the
    /// other end — an already-dark inactive border (0.16 grey) scaled
    /// down lands at 0.026, which reads as a hole in the window rather
    /// than as chrome. Mixing keeps a floor.
    #[must_use]
    pub fn from_border(border: [f32; 3], focused: bool) -> Self {
        let t = if focused { 0.35 } else { 0.12 };
        let mix = |c: f32| SURFACE.mul_add(1.0 - t, c * t);
        Self {
            background: [mix(border[0]), mix(border[1]), mix(border[2])],
            text: if focused {
                [0.94, 0.94, 0.96]
            } else {
                [0.58, 0.58, 0.62]
            },
        }
    }
}

/// What a bar is showing beyond its title: which button the pointer is
/// on, and whether the window is already maximized (so the maximize
/// button shows *restore* instead).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BarState {
    pub hovered: Option<TitlebarButton>,
    pub maximized: bool,
}

/// Rasterize a bar to premultiplied RGBA8, row-major, `width * height`.
///
/// `fonts` may be `None` (no usable font on the system): the bar still
/// draws its background and buttons, because a window you can close and
/// drag without a readable title beats no decoration at all.
#[must_use]
/// Everything one titlebar needs to be drawn. A struct rather than nine
/// arguments because that is what it is: a single description of one
/// bar, and every caller has all of it or none of it.
pub struct BarSpec<'a> {
    pub fonts: Option<&'a Fonts>,
    pub width: i32,
    pub height: i32,
    pub title: &'a str,
    pub style: BarStyle,
    pub buttons: &'a [TitlebarButton],
    pub font_px: f32,
    pub state: BarState,
    pub icon: Option<&'a crate::icon::Icon>,
}

pub fn rasterize(spec: &BarSpec<'_>) -> Vec<u8> {
    let &BarSpec {
        fonts,
        width,
        height,
        title,
        style,
        buttons,
        font_px,
        state,
        icon,
    } = spec;
    let (cols, rows) = (width.max(1) as usize, height.max(1) as usize);
    let mut buf = fill_background(cols, rows, style);
    let layout = bar_layout(width, height, buttons);
    let fg = style.text;

    // Hovered button backing, under its glyph. Close gets a red so the
    // destructive one announces itself before you commit to it.
    if let Some(hovered) = state.hovered
        && let Some((kind, rect)) = layout.buttons.iter().find(|(k, _)| *k == hovered)
    {
        let (tint, strength) = if *kind == TitlebarButton::Close {
            (CLOSE_HOVER, 0.85)
        } else {
            ([1.0, 1.0, 1.0], HOVER_TINT)
        };
        for y in rect.loc.y.max(0)..(rect.loc.y + rect.size.h).min(rows as i32) {
            for x in rect.loc.x.max(0)..(rect.loc.x + rect.size.w).min(cols as i32) {
                blend(&mut buf, cols, x as usize, y as usize, tint, strength);
            }
        }
    }

    // --- title -------------------------------------------------------
    if let Some(fonts) = fonts
        && layout.title.size.w > 0
        && !title.is_empty()
    {
        let max_w = layout.title.size.w as f32;
        let shown = fonts.ellipsize(title, font_px, false, max_w);
        // Vertically centre on the font's own size rather than on the
        // bar: baseline at centre + roughly half the cap height reads
        // level, where centring the em box sits visibly low.
        let baseline = (rows as f32 / 2.0 + font_px * 0.34).round();
        // Centred in the *bar*, Breeze-style, then pushed right if that
        // would start it before the title area (a long title in a narrow
        // window) or pulled left if it would run under the buttons.
        // Centring in the title area instead would make the text drift
        // left as buttons are added, which looks like a bug.
        let text_w = fonts.measure(&shown, font_px, false);
        let centred = (cols as f32 - text_w) / 2.0;
        let origin = centred
            .max(layout.title.loc.x as f32)
            .min((layout.title.loc.x + layout.title.size.w) as f32 - text_w)
            .max(layout.title.loc.x as f32);
        fonts.layout(
            &shown,
            font_px,
            false,
            origin,
            baseline,
            |gx, gy, gw, gh, coverage| {
                for row in 0..gh {
                    let y = gy + row as i32;
                    if y < 0 || y as usize >= rows {
                        continue;
                    }
                    for col in 0..gw {
                        let x = gx + col as i32;
                        if x < 0 || x as usize >= cols {
                            continue;
                        }
                        let a = f32::from(coverage[row * gw + col]) / 255.0;
                        if a > 0.0 {
                            blend(&mut buf, cols, x as usize, y as usize, fg, a);
                        }
                    }
                }
            },
        );
    }

    if let Some(icon) = icon {
        draw_icon(&mut buf, cols, rows, layout.icon, icon);
    }

    // --- buttons -----------------------------------------------------
    for (kind, rect) in &layout.buttons {
        draw_glyph(
            &mut Canvas {
                buf: &mut buf,
                stride: cols,
                rows,
            },
            *kind,
            *rect,
            height,
            fg,
            state,
        );
    }
    buf
}

/// The bar's backing: a faint top-to-bottom sheen, then a darker rule
/// along the bottom edge so the bar reads as a surface distinct from the
/// client's content rather than as part of it.
///
/// Opaque throughout — the bar covers the client's buffer underneath,
/// and any translucency here would show *that*, not the backdrop.
fn fill_background(cols: usize, rows: usize, style: BarStyle) -> Vec<u8> {
    let mut buf = vec![0u8; cols * rows * 4];
    // The last row (two at HiDPI) is the rule.
    let rule = rows.saturating_sub(1 + usize::from(rows > 40));
    for y in 0..rows {
        let t = if rows > 1 {
            y as f32 / (rows - 1) as f32
        } else {
            0.0
        };
        let mut shade = SHEEN.mul_add(-t, SHEEN * (1.0 - t));
        if y >= rule {
            shade -= 0.055;
        }
        for x in 0..cols {
            let i = (y * cols + x) * 4;
            for c in 0..3 {
                buf[i + c] = ((style.background[c] + shade).clamp(0.0, 1.0) * 255.0) as u8;
            }
            buf[i + 3] = 255;
        }
    }
    buf
}

/// Alpha-over the application icon into its slot. Straight compositing:
/// the icon is the one part of the bar that isn't ours to recolour.
fn draw_icon(
    buf: &mut [u8],
    cols: usize,
    rows: usize,
    slot: Rectangle<i32, Physical>,
    icon: &crate::icon::Icon,
) {
    for row in 0..icon.height.min(slot.size.h.max(0) as u32) {
        let y = slot.loc.y + row as i32;
        if y < 0 || y as usize >= rows {
            continue;
        }
        for col in 0..icon.width.min(slot.size.w.max(0) as u32) {
            let x = slot.loc.x + col as i32;
            if x < 0 || x as usize >= cols {
                continue;
            }
            let i = ((row * icon.width + col) as usize) * 4;
            let a = f32::from(icon.rgba[i + 3]) / 255.0;
            if a > 0.0 {
                let rgb = [
                    f32::from(icon.rgba[i]) / 255.0,
                    f32::from(icon.rgba[i + 1]) / 255.0,
                    f32::from(icon.rgba[i + 2]) / 255.0,
                ];
                blend(buf, cols, x as usize, y as usize, rgb, a);
            }
        }
    }
}

/// Alpha-blend `colour` at `a` coverage over the pixel at `(x, y)`.
/// The destination is opaque, so this is a plain lerp and the result
/// stays opaque — no premultiplication bookkeeping.
fn blend(buf: &mut [u8], stride: usize, x: usize, y: usize, colour: [f32; 3], a: f32) {
    let i = (y * stride + x) * 4;
    let a = a.clamp(0.0, 1.0);
    for c in 0..3 {
        let dst = f32::from(buf[i + c]) / 255.0;
        let src = colour[c];
        buf[i + c] = (src.mul_add(a, dst * (1.0 - a)) * 255.0).clamp(0.0, 255.0) as u8;
    }
}

/// Distance from `p` to the segment `a`–`b`.
fn dist_to_segment(point: (f32, f32), from: (f32, f32), to: (f32, f32)) -> f32 {
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let (px, py) = (point.0 - from.0, point.1 - from.1);
    let len2 = dx.mul_add(dx, dy * dy);
    let along = if len2 <= f32::EPSILON {
        0.0
    } else {
        (px.mul_add(dx, py * dy) / len2).clamp(0.0, 1.0)
    };
    let nearest = (dx.mul_add(along, from.0), dy.mul_add(along, from.1));
    (point.0 - nearest.0).hypot(point.1 - nearest.1)
}

/// Draw one button glyph, anti-aliased from its distance field.
///
/// Every glyph is a set of line segments, so one rasterizer covers all
/// three: minimize is one segment, close is two crossed ones, and
/// maximize is a four-segment outline.
/// A mutable RGBA surface: the pixels and the two numbers needed to
/// index them. Passed as one thing because they are only ever meaningful
/// together — a stride without its buffer indexes nothing.
struct Canvas<'a> {
    buf: &'a mut [u8],
    stride: usize,
    rows: usize,
}

fn draw_glyph(
    canvas: &mut Canvas<'_>,
    kind: TitlebarButton,
    rect: Rectangle<i32, Physical>,
    bar_h: i32,
    colour: [f32; 3],
    state: BarState,
) {
    let Canvas { buf, stride, rows } = canvas;
    let (stride, rows) = (*stride, *rows);
    let extent = (bar_h as f32 * GLYPH).max(4.0);
    let mid_x = rect.loc.x as f32 + rect.size.w as f32 / 2.0;
    let mid_y = rect.loc.y as f32 + rect.size.h as f32 / 2.0;
    let (left, right) = (mid_x - extent / 2.0, mid_x + extent / 2.0);
    let (top, bottom) = (mid_y - extent / 2.0, mid_y + extent / 2.0);
    // Chevrons are flatter than they are wide, so they don't read as
    // arrowheads.
    let (chev_top, chev_bottom) = (mid_y - extent * 0.28, mid_y + extent * 0.28);
    let segments: Vec<((f32, f32), (f32, f32))> = match kind {
        // Breeze: a chevron pointing down — the window goes away
        // downwards, to the taskbar.
        TitlebarButton::Minimize => vec![
            ((left, chev_top), (mid_x, chev_bottom)),
            ((mid_x, chev_bottom), (right, chev_top)),
        ],
        // Up to maximize, down to restore. Never a square outline: at
        // button size that is exactly the box a font draws for a missing
        // glyph, and it reads as a broken icon.
        TitlebarButton::Maximize if state.maximized => vec![
            ((left, mid_y), (mid_x, chev_bottom)),
            ((mid_x, chev_bottom), (right, mid_y)),
            (
                (left, chev_top - extent * 0.10),
                (mid_x, mid_y - extent * 0.10),
            ),
            (
                (mid_x, mid_y - extent * 0.10),
                (right, chev_top - extent * 0.10),
            ),
        ],
        TitlebarButton::Maximize => vec![
            ((left, chev_bottom), (mid_x, chev_top)),
            ((mid_x, chev_top), (right, chev_bottom)),
        ],
        // Two diagonals.
        TitlebarButton::Close => vec![
            ((left, top), (right, bottom)),
            ((right, top), (left, bottom)),
        ],
    };
    // The hovered close button paints its glyph white on red rather than
    // in the bar's text colour, which would disappear into it.
    let colour = if state.hovered == Some(kind) && kind == TitlebarButton::Close {
        [1.0, 1.0, 1.0]
    } else {
        colour
    };
    // Only the glyph's own neighbourhood can be covered, so bound the
    // scan to it rather than sweeping the whole button.
    let pad = STROKE.ceil() as i32 + 2;
    let x0 = (left as i32 - pad).max(0);
    let x1 = (right as i32 + pad).min(stride as i32 - 1);
    let y0 = (top as i32 - pad).max(0);
    let y1 = (bottom as i32 + pad).min(rows as i32 - 1);
    for y in y0..=y1 {
        for x in x0..=x1 {
            // Sample at the pixel centre.
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            let d = segments
                .iter()
                .map(|(from, to)| dist_to_segment(p, *from, *to))
                .fold(f32::INFINITY, f32::min);
            // One pixel of feather across the stroke edge: coverage 1
            // inside, 0 outside, linear between.
            let a = (STROKE + 0.5 - d).clamp(0.0, 1.0);
            if a > 0.0 {
                blend(buf, stride, x as usize, y as usize, colour, a);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BarSpec, BarState, BarStyle, Deco, EdgeX, EdgeY, Point, Rectangle, Region, Size,
        TitlebarButton, bar_layout, rasterize, region_at,
    };

    /// A 2 px border with a 28 px bar — the shipped default.
    const DECO: Deco = Deco {
        border: 2,
        titlebar: 28,
    };

    /// A window at (100, 100), 300x400 — offset so a bug that assumes a
    /// zero origin shows up.
    /// A bar spec with the test defaults filled in — every test varies
    /// one or two of these and would otherwise repeat the other seven.
    fn spec<'a>(
        width: i32,
        height: i32,
        title: &'a str,
        buttons: &'a [TitlebarButton],
    ) -> BarSpec<'a> {
        BarSpec {
            fonts: None,
            width,
            height,
            title,
            style: style(),
            buttons,
            font_px: 13.0,
            state: BarState::default(),
            icon: None,
        }
    }

    fn cell() -> Rectangle<i32, super::Physical> {
        Rectangle::new(Point::from((100, 100)), Size::from((300, 400)))
    }

    const ALL: &[TitlebarButton] = &[
        TitlebarButton::Minimize,
        TitlebarButton::Maximize,
        TitlebarButton::Close,
    ];

    fn style() -> BarStyle {
        BarStyle::from_border([0.4, 0.6, 0.9], true)
    }

    /// Buttons are right-aligned in configured order, so the last entry
    /// written in the config is the one nearest the right edge.
    #[test]
    fn the_last_configured_button_is_rightmost() {
        let bar = bar_layout(400, 28, ALL);
        assert_eq!(bar.buttons.len(), 3);
        assert_eq!(bar.buttons[0].0, TitlebarButton::Minimize);
        assert_eq!(bar.buttons[2].0, TitlebarButton::Close);
        let xs: Vec<i32> = bar.buttons.iter().map(|(_, r)| r.loc.x).collect();
        assert!(xs[0] < xs[1] && xs[1] < xs[2], "not left-to-right: {xs:?}");
        // The rightmost button ends before the bar does.
        let last = bar.buttons[2].1;
        assert!(last.loc.x + last.size.w <= 400);
    }

    /// Buttons never overlap, or a click would be ambiguous between two
    /// of them and `button_at` would silently prefer the earlier one.
    #[test]
    fn buttons_do_not_overlap() {
        let bar = bar_layout(400, 28, ALL);
        for pair in bar.buttons.windows(2) {
            let (a, b) = (pair[0].1, pair[1].1);
            assert!(
                a.loc.x + a.size.w <= b.loc.x,
                "buttons overlap: {a:?} then {b:?}"
            );
        }
    }

    /// The title area stops where the buttons start; overlapping them
    /// would draw the title under the close button.
    #[test]
    fn the_title_never_reaches_the_buttons() {
        let bar = bar_layout(400, 28, ALL);
        let first_button = bar.buttons[0].1.loc.x;
        assert!(bar.title.loc.x + bar.title.size.w <= first_button);
    }

    /// A bar too narrow for every button keeps the rightmost ones —
    /// losing close before minimize would be the wrong way round.
    #[test]
    fn a_narrow_bar_drops_buttons_from_the_left() {
        // Wide enough for one button (28-tall bar: 12 px padding either
        // side, 37 px per button), not for three.
        let bar = bar_layout(70, 28, ALL);
        assert!(
            !bar.buttons.is_empty() && bar.buttons.len() < 3,
            "expected a partial set: {bar:?}"
        );
        assert_eq!(
            bar.buttons.last().map(|(k, _)| *k),
            Some(TitlebarButton::Close),
            "close is the last button to go"
        );
        assert!(bar.title.size.w >= 0);
    }

    /// Narrower still and nothing fits. The bar is then a title strip:
    /// no buttons, no negative title width, no panic.
    #[test]
    fn a_bar_too_narrow_for_any_button_still_lays_out() {
        let bar = bar_layout(40, 28, ALL);
        assert!(bar.buttons.is_empty());
        assert!(bar.title.size.w >= 0);
    }

    /// Degenerate sizes must not panic or produce a negative extent —
    /// a window can be configured to 1x1 while a client catches up.
    #[test]
    fn degenerate_bars_are_safe() {
        for (w, h) in [(0, 0), (1, 1), (10, 0), (0, 28)] {
            let bar = bar_layout(w, h, ALL);
            assert!(bar.title.size.w >= 0 && bar.title.size.h >= 0);
            let px = rasterize(&spec(w, h, "x", ALL));
            assert_eq!(px.len(), (w.max(1) as usize) * (h.max(1) as usize) * 4);
        }
    }

    /// `button_at` has to agree with the drawn rects — that agreement is
    /// the reason geometry and drawing live in one module.
    #[test]
    fn hit_testing_matches_the_drawn_rects() {
        let bar = bar_layout(400, 28, ALL);
        for (kind, rect) in &bar.buttons {
            let centre = smithay::utils::Point::from((
                rect.loc.x + rect.size.w / 2,
                rect.loc.y + rect.size.h / 2,
            ));
            assert_eq!(bar.button_at(centre), Some(*kind));
        }
        // The title area belongs to no button, so a press there drags.
        let in_title = smithay::utils::Point::from((bar.title.loc.x + 1, 14));
        assert_eq!(bar.button_at(in_title), None);
    }

    /// The bar is opaque everywhere: it covers the client's surface, and
    /// any hole would show the stretched top edge of that surface.
    #[test]
    fn the_rasterized_bar_is_fully_opaque() {
        let px = rasterize(&spec(200, 28, "title", ALL));
        assert!(px.chunks_exact(4).all(|p| p[3] == 255));
    }

    /// The glyphs actually mark the buffer — a silent no-op rasterizer
    /// would otherwise look like a correctly drawn flat bar.
    #[test]
    fn button_glyphs_are_drawn() {
        let bare = rasterize(&spec(200, 28, "", &[]));
        let with = rasterize(&spec(200, 28, "", ALL));
        assert_ne!(bare, with, "buttons left no marks");
    }

    /// A press in the middle of the bar drags; on a button it doesn't.
    #[test]
    fn the_bar_drags_except_on_its_buttons() {
        let cell = cell();
        assert_eq!(
            region_at(cell, DECO, ALL, 8, Point::from((150, 112))),
            Region::Titlebar
        );
        // Same row, over the close button.
        let bar = bar_layout(cell.size.w - 2 * DECO.border, DECO.titlebar, ALL);
        let close = bar.buttons.last().expect("close is placed").1;
        let x = cell.loc.x + DECO.border + close.loc.x + close.size.w / 2;
        assert_eq!(
            region_at(cell, DECO, ALL, 8, Point::from((x, 112))),
            Region::Button(TitlebarButton::Close)
        );
    }

    /// The resize margin beats the titlebar, or the top edge of a window
    /// is the one edge that cannot be grabbed.
    #[test]
    fn the_top_edge_resizes_rather_than_dragging() {
        let cell = cell();
        // 2 px below the cell's top: inside the bar's rows, inside the
        // margin.
        let r = region_at(cell, DECO, ALL, 8, Point::from((200, 102)));
        match r {
            Region::Resize(e) => {
                assert_eq!(e.y, Some(EdgeY::Top), "should take the top edge");
                assert_eq!(e.x, None, "middle of the edge is not a corner");
            }
            other => panic!("expected a resize, got {other:?}"),
        }
    }

    /// A corner takes both axes; an edge takes one. The distinction is
    /// what stops a left-edge drag from also resizing the height.
    #[test]
    fn corners_take_both_axes_and_edges_take_one() {
        let cell = cell();
        let corner = region_at(cell, DECO, ALL, 8, Point::from((101, 101)));
        match corner {
            Region::Resize(e) => {
                assert_eq!((e.x, e.y), (Some(EdgeX::Left), Some(EdgeY::Top)));
            }
            other => panic!("expected a resize, got {other:?}"),
        }
        // Half way down the left edge: horizontal only.
        let edge = region_at(cell, DECO, ALL, 8, Point::from((101, 250)));
        match edge {
            Region::Resize(e) => {
                assert_eq!((e.x, e.y), (Some(EdgeX::Left), None));
            }
            other => panic!("expected a resize, got {other:?}"),
        }
    }

    /// Everything below the bar and inside the margins belongs to the
    /// client — the compositor must not eat presses there.
    #[test]
    fn the_middle_of_a_window_is_the_clients() {
        assert_eq!(
            region_at(cell(), DECO, ALL, 8, Point::from((200, 250))),
            Region::Content
        );
    }

    /// With no titlebar (the tiling default) a window has only edges and
    /// content, and the rows a bar would have occupied stay the client's.
    #[test]
    fn a_bare_window_has_no_titlebar_region() {
        let bare = Deco::new(1, 0);
        let cell = cell();
        assert_eq!(
            region_at(cell, bare, ALL, 8, Point::from((200, 112))),
            Region::Content
        );
    }

    /// `resize_zone = 0` disables edge grabs entirely rather than
    /// producing a zero-width margin that swallows the border row.
    #[test]
    fn a_zero_resize_zone_disables_edge_grabs() {
        let cell = cell();
        assert_eq!(
            region_at(cell, DECO, ALL, 0, Point::from((100, 400))),
            Region::Content
        );
    }

    /// On a window smaller than twice the margin, the margin must not
    /// swallow the whole window — there would be nothing left to drag,
    /// click or hand to the client.
    #[test]
    fn the_margin_never_swallows_a_small_window() {
        let tiny = Rectangle::new(Point::from((0, 0)), Size::from((20, 20)));
        let mut saw_other = false;
        for y in 0..20 {
            for x in 0..20 {
                if !matches!(
                    region_at(tiny, DECO, ALL, 40, Point::from((x, y))),
                    Region::Resize(_)
                ) {
                    saw_other = true;
                }
            }
        }
        assert!(saw_other, "an oversized margin swallowed the whole window");
    }

    /// The maximize glyph must not be a square outline: at button size
    /// that is exactly the box a font draws for a missing glyph, which
    /// is what it was mistaken for. A chevron touches the middle row at
    /// only one x; a square's outline spans the full width on its top
    /// and bottom rows.
    #[test]
    fn the_maximize_glyph_is_a_chevron_not_a_box() {
        let bg = rasterize(&spec(200, 28, "", &[]));
        let with = rasterize(&spec(200, 28, "", &[TitlebarButton::Maximize]));
        // Rows where the glyph differs from the plain bar, and how wide
        // that difference is on each.
        let widths: Vec<usize> = (0..28)
            .map(|y| {
                (0..200)
                    .filter(|x| {
                        let i = (y * 200 + x) * 4;
                        bg[i..i + 3] != with[i..i + 3]
                    })
                    .count()
            })
            .collect();
        let marked: Vec<usize> = widths.iter().copied().filter(|&w| w > 0).collect();
        assert!(!marked.is_empty(), "maximize drew nothing");
        // A box outline has two rows as wide as the glyph itself; a
        // chevron's widest row is its base and it narrows to a point.
        let widest = *marked.iter().max().expect("non-empty");
        let narrowest = *marked.iter().min().expect("non-empty");
        assert!(
            narrowest * 2 < widest,
            "glyph never narrows ({narrowest}..{widest}) — that's a box, not a chevron"
        );
    }

    /// Maximize and restore have to be distinguishable, or the button
    /// gives no clue what it will do.
    #[test]
    fn restore_draws_differently_from_maximize() {
        let one = |maximized| {
            rasterize(&BarSpec {
                state: BarState {
                    hovered: None,
                    maximized,
                },
                ..spec(200, 28, "", &[TitlebarButton::Maximize])
            })
        };
        assert_ne!(one(false), one(true));
    }

    /// Hover has to be visible, and close specifically has to look
    /// different from the others — it is the destructive one.
    #[test]
    fn hover_marks_the_button_and_close_is_special() {
        let plain = rasterize(&spec(200, 28, "", ALL));
        let on = |kind| {
            rasterize(&BarSpec {
                state: BarState {
                    hovered: Some(kind),
                    maximized: false,
                },
                ..spec(200, 28, "", ALL)
            })
        };
        assert_ne!(plain, on(TitlebarButton::Close), "close hover invisible");
        assert_ne!(plain, on(TitlebarButton::Minimize), "hover invisible");
        assert_ne!(
            on(TitlebarButton::Close),
            on(TitlebarButton::Minimize),
            "close should not look like the others on hover"
        );
    }

    /// An already-dark inactive border must not drag the bar to
    /// near-black — the first attempt scaled the colour and produced a
    /// hole in the window rather than chrome.
    #[test]
    fn a_dark_border_still_gives_a_visible_bar() {
        let dark = BarStyle::from_border([0.16, 0.16, 0.20], false);
        for c in dark.background {
            assert!(
                c > 0.06,
                "unfocused bar is effectively black: {:?}",
                dark.background
            );
        }
        // ...and the text still has somewhere to sit above it.
        assert!(dark.text[0] - dark.background[0] > 0.3);
    }

    /// Focus has to be visible without reading the text, so the two
    /// backgrounds must actually differ.
    #[test]
    fn focus_changes_the_bar() {
        let on = BarStyle::from_border([0.4, 0.6, 0.9], true);
        let off = BarStyle::from_border([0.4, 0.6, 0.9], false);
        assert!(
            on.background
                .iter()
                .zip(off.background)
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "focused and unfocused bars look identical"
        );
        assert!(
            on.text
                .iter()
                .zip(off.text)
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
    }
}
