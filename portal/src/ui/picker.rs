//! Full-screen pickers: choose a monitor, drag a region, sample a colour.
//!
//! All three are `wlr-layer-shell` overlays on every output at once, which is
//! the only way to make "point at the thing you mean" work across a multi-head
//! desktop.
//!
//! The monitor picker replaces what wlroots portals shell out to (`slurp`, or
//! a dmenu list) — folded in here so screen sharing has no external chooser to
//! configure, and so it can't crash the way slurp does on this compositor (an
//! unguarded null deref on a `wl_pointer.motion` with no preceding `enter`).
//! The region and colour pickers draw a frozen capture of the desktop, so what
//! you select is exactly what was on screen when the request arrived.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel geometry: every value here is a surface- or image-sized non-negative integer, and the conversions between i32/u32/usize/f32 are all inside that range. Checked conversions at each site would be noise around arithmetic that cannot overflow."
)]

use crate::capture::Frame;

use super::draw::{Canvas, Color, Image, Rect};
use super::{Ctx, Flow, Input, Key, OutputDesc, Screen};

/// Highlight colours, shared by all three pickers.
const DIM: Color = Color::rgba(0x8C_00_00_00);
const WASH: Color = Color::rgba(0x4A_4A_9E_FF);

// ── Monitor picker ─────────────────────────────────────────────────────────

/// Pick one output. Used by `ScreenCast` source selection.
#[derive(Default)]
pub struct OutputPicker {
    outputs: Vec<OutputDesc>,
    hovered: Option<usize>,
    /// The connector name the user clicked.
    pub chosen: Option<String>,
    /// Set when the user asked to include the cursor in the stream.
    pub with_cursor: bool,
    /// Whether to draw (and honour) the cursor toggle at all.
    pub offer_cursor_toggle: bool,
    toggle: Rect,
}

impl OutputPicker {
    pub fn new(offer_cursor_toggle: bool, with_cursor: bool) -> Self {
        Self {
            with_cursor,
            offer_cursor_toggle,
            ..Self::default()
        }
    }
}

impl Screen for OutputPicker {
    fn outputs(&mut self, outputs: &[OutputDesc]) {
        self.outputs = outputs.to_vec();
    }

    fn render(&mut self, surface: usize, canvas: &mut Canvas<'_>, ctx: &Ctx) {
        let theme = &ctx.theme;
        let (w, h) = canvas.size();
        let hovered = self.hovered == Some(surface);
        canvas.clear(if hovered { WASH } else { DIM });
        if hovered {
            let thickness = (w / 240).clamp(3, 10);
            canvas.stroke_rounded(Rect::new(0, 0, w, h), 0, thickness, theme.accent);
        }

        let Some(desc) = self.outputs.get(surface) else {
            return;
        };
        let card = Rect::new(w / 2 - 190, h / 2 - 70, 380, 140);
        canvas.fill_rounded(card, 14, theme.bg.with_alpha(0.92));
        canvas.stroke_rounded(card, 14, 1, theme.border);
        canvas.text_centered(
            &ctx.fonts,
            Rect::new(card.x, card.y + 22, card.w, 34),
            26.0,
            true,
            theme.text,
            &desc.name,
        );
        canvas.text_centered(
            &ctx.fonts,
            Rect::new(card.x, card.y + 58, card.w, 22),
            14.0,
            false,
            theme.dim,
            &format!("{}×{}", desc.mode_width, desc.mode_height),
        );
        canvas.text_centered(
            &ctx.fonts,
            Rect::new(card.x, card.y + 88, card.w, 22),
            13.0,
            false,
            if hovered { theme.accent } else { theme.dim },
            // Enter takes the only monitor without aiming at it (see
            // `input`), which is no use to anyone who can't see that it
            // is there. With a choice to make, Enter needs a hover to
            // act on, so it is only worth advertising once there is
            // exactly one thing it could mean.
            if self.outputs.len() == 1 {
                "Press Enter or click to share · Esc to cancel"
            } else {
                "Click to share this monitor · Esc to cancel"
            },
        );

        if self.offer_cursor_toggle {
            let label = if self.with_cursor {
                "Cursor: shown  (press C)"
            } else {
                "Cursor: hidden  (press C)"
            };
            let width = canvas.measure(&ctx.fonts, label, 13.0, false) as i32 + 28;
            self.toggle = Rect::new(w / 2 - width / 2, card.bottom() + 16, width, 30);
            canvas.fill_rounded(self.toggle, 8, theme.bg.with_alpha(0.92));
            canvas.stroke_rounded(
                self.toggle,
                8,
                1,
                if self.with_cursor {
                    theme.accent
                } else {
                    theme.border
                },
            );
            canvas.text_centered(&ctx.fonts, self.toggle, 13.0, false, theme.text, label);
        }
    }

    fn input(&mut self, surface: usize, event: &Input, _ctx: &Ctx) -> Flow {
        match event {
            Input::Motion { .. } => {
                if self.hovered == Some(surface) {
                    return Flow::Idle;
                }
                self.hovered = Some(surface);
                Flow::RedrawAll
            }
            Input::Leave => {
                if self.hovered == Some(surface) {
                    self.hovered = None;
                    return Flow::RedrawAll;
                }
                Flow::Idle
            }
            Input::Press { x, y, .. } => {
                if self.offer_cursor_toggle && self.toggle.contains(*x, *y) {
                    self.with_cursor = !self.with_cursor;
                    return Flow::Redraw;
                }
                self.chosen = self.outputs.get(surface).map(|o| o.name.clone());
                Flow::Done
            }
            Input::Key { key, .. } => match key {
                Key::Escape => Flow::Done,
                Key::Char('c' | 'C') if self.offer_cursor_toggle => {
                    self.with_cursor = !self.with_cursor;
                    Flow::RedrawAll
                }
                Key::Enter => {
                    // Enter picks whatever is hovered, or the only monitor.
                    let index = self.hovered.or(if self.outputs.len() == 1 {
                        Some(0)
                    } else {
                        None
                    });
                    self.chosen = index
                        .and_then(|i| self.outputs.get(i))
                        .map(|o| o.name.clone());
                    if self.chosen.is_some() {
                        Flow::Done
                    } else {
                        Flow::Idle
                    }
                }
                _ => Flow::Idle,
            },
            Input::Release | Input::Scroll { .. } => Flow::Idle,
        }
    }
}

// ── Region selector ────────────────────────────────────────────────────────

/// Look one output's capture up by connector name.
fn frame_named<'a>(frames: &'a [(String, Frame)], name: &str) -> Option<&'a Frame> {
    frames.iter().find(|(n, _)| n == name).map(|(_, f)| f)
}

/// What a region drag produced: an output and a rectangle inside it, in that
/// output's *physical* pixels (the coordinate space captures come back in).
///
/// The output is named, not indexed. Two orderings meet here and they do not
/// agree: the capture list is sorted by connector name, the overlay's
/// surfaces by layout position. Passing an index across that boundary silently
/// picked a different monitor whenever alphabetical order differed from
/// left-to-right order.
#[derive(Clone, Debug)]
pub struct Region {
    pub output: String,
    pub rect: Rect,
}

/// Drag a rectangle over a frozen capture of the desktop.
pub struct RegionPicker {
    outputs: Vec<OutputDesc>,
    frames: Vec<(String, Frame)>,
    /// Drag start, in logical coordinates of the output it began on.
    anchor: Option<(usize, f64, f64)>,
    current: Option<(f64, f64)>,
    hovered: Option<usize>,
    pub region: Option<Region>,
    /// Set when the user clicked without dragging: take the whole output.
    pub whole_output: Option<String>,
}

impl RegionPicker {
    pub fn new(frames: Vec<(String, Frame)>) -> Self {
        Self {
            outputs: Vec::new(),
            frames,
            anchor: None,
            current: None,
            hovered: None,
            region: None,
            whole_output: None,
        }
    }

    /// The capture belonging to overlay surface `surface`, matched by
    /// connector name rather than by position in either list.
    fn frame_for(&self, surface: usize) -> Option<&Frame> {
        let name = &self.outputs.get(surface)?.name;
        frame_named(&self.frames, name)
    }

    /// The in-progress selection in logical coordinates.
    fn selection(&self) -> Option<(usize, Rect)> {
        let (surface, ax, ay) = self.anchor?;
        let (cx, cy) = self.current?;
        let rect = Rect::new(
            ax.min(cx) as i32,
            ay.min(cy) as i32,
            (ax - cx).abs() as i32,
            (ay - cy).abs() as i32,
        );
        Some((surface, rect))
    }
}

impl Screen for RegionPicker {
    fn outputs(&mut self, outputs: &[OutputDesc]) {
        self.outputs = outputs.to_vec();
    }

    fn render(&mut self, surface: usize, canvas: &mut Canvas<'_>, ctx: &Ctx) {
        let theme = &ctx.theme;
        let (w, h) = canvas.size();
        // The frozen desktop underneath, then a dim wash over it.
        if let Some(frame) = self.frame_for(surface) {
            canvas.blit(
                Rect::new(0, 0, w, h),
                &Image {
                    data: &frame.data,
                    width: frame.width,
                    height: frame.height,
                    stride: frame.stride,
                },
                Rect::new(0, 0, frame.width, frame.height),
            );
        } else {
            canvas.clear(Color::rgb(0x0010_1216));
        }
        canvas.fill(Rect::new(0, 0, w, h), DIM);

        if let Some((sel_surface, rect)) = self.selection()
            && sel_surface == surface
            && !rect.is_empty()
        {
            // Punch the selection back through to the unwashed capture.
            if let Some(frame) = self.frame_for(surface) {
                let scale = f64::from(frame.width) / f64::from(w.max(1));
                let src = Rect::new(
                    (f64::from(rect.x) * scale) as i32,
                    (f64::from(rect.y) * scale) as i32,
                    (f64::from(rect.w) * scale) as i32,
                    (f64::from(rect.h) * scale) as i32,
                );
                canvas.blit(
                    rect,
                    &Image {
                        data: &frame.data,
                        width: frame.width,
                        height: frame.height,
                        stride: frame.stride,
                    },
                    src,
                );
            }
            canvas.stroke_rounded(rect, 0, 1, theme.accent);
            let label = format!("{} × {}", rect.w, rect.h);
            let width = canvas.measure(&ctx.fonts, &label, 13.0, false) as i32 + 20;
            let badge = Rect::new(
                rect.x.min(w - width - 8).max(8),
                (rect.y - 30).max(8),
                width,
                24,
            );
            canvas.fill_rounded(badge, 6, theme.bg.with_alpha(0.9));
            canvas.text_centered(&ctx.fonts, badge, 13.0, false, theme.text, &label);
        } else if self.hovered == Some(surface) {
            let hint = "Drag to select a region · click for the whole monitor · Esc to cancel";
            let width = canvas.measure(&ctx.fonts, hint, 14.0, false) as i32 + 36;
            let card = Rect::new(w / 2 - width / 2, h / 2 - 22, width, 44);
            canvas.fill_rounded(card, 10, theme.bg.with_alpha(0.9));
            canvas.stroke_rounded(card, 10, 1, theme.border);
            canvas.text_centered(&ctx.fonts, card, 14.0, false, theme.text, hint);
        }
    }

    fn input(&mut self, surface: usize, event: &Input, _ctx: &Ctx) -> Flow {
        match event {
            Input::Press { x, y, .. } => {
                self.anchor = Some((surface, *x, *y));
                self.current = Some((*x, *y));
                Flow::Redraw
            }
            Input::Motion { x, y } => {
                let moved = self.hovered != Some(surface);
                self.hovered = Some(surface);
                if self.anchor.is_some() {
                    self.current = Some((*x, *y));
                    return Flow::Redraw;
                }
                if moved { Flow::RedrawAll } else { Flow::Idle }
            }
            Input::Release => {
                let Some((sel_surface, rect)) = self.selection() else {
                    return Flow::Idle;
                };
                let Some(name) = self.outputs.get(sel_surface).map(|o| o.name.clone()) else {
                    return Flow::Idle;
                };
                // A click (or a slip of a few pixels) means the whole monitor;
                // anything bigger is a deliberate region.
                if rect.w < 8 || rect.h < 8 {
                    self.whole_output = Some(name);
                    return Flow::Done;
                }
                // Logical → physical, using the capture's own dimensions so
                // the region lands on the right pixels under fractional scale.
                let scale = self
                    .frame_for(sel_surface)
                    .zip(self.outputs.get(sel_surface))
                    .map_or(1.0, |(frame, desc)| {
                        f64::from(frame.width) / f64::from(desc.width.max(1))
                    });
                let physical = Rect::new(
                    (f64::from(rect.x) * scale) as i32,
                    (f64::from(rect.y) * scale) as i32,
                    (f64::from(rect.w) * scale).round() as i32,
                    (f64::from(rect.h) * scale).round() as i32,
                );
                self.region = Some(Region {
                    output: name,
                    rect: physical,
                });
                Flow::Done
            }
            Input::Leave => {
                if self.hovered == Some(surface) {
                    self.hovered = None;
                    return Flow::RedrawAll;
                }
                Flow::Idle
            }
            Input::Key { key, .. } => {
                if key == &Key::Escape {
                    return Flow::Done;
                }
                Flow::Idle
            }
            Input::Scroll { .. } => Flow::Idle,
        }
    }
}

// ── Colour picker ──────────────────────────────────────────────────────────

/// Sample one pixel from a frozen capture, with a magnifier to aim by.
pub struct ColorPicker {
    outputs: Vec<OutputDesc>,
    frames: Vec<(String, Frame)>,
    at: Option<(usize, f64, f64)>,
    /// The picked colour as straight RGB in 0.0–1.0, which is what the portal
    /// returns.
    pub picked: Option<(f64, f64, f64)>,
}

impl ColorPicker {
    pub fn new(frames: Vec<(String, Frame)>) -> Self {
        Self {
            outputs: Vec::new(),
            frames,
            at: None,
            picked: None,
        }
    }

    /// The captured pixel under a logical position on `surface`. Matched by
    /// connector name — see [`Region`] for why an index would not do.
    fn sample(&self, surface: usize, x: f64, y: f64) -> Option<(u8, u8, u8)> {
        let desc = self.outputs.get(surface)?;
        let frame = frame_named(&self.frames, &desc.name)?;
        let scale_x = f64::from(frame.width) / f64::from(desc.width.max(1));
        let scale_y = f64::from(frame.height) / f64::from(desc.height.max(1));
        let (px, py) = (
            ((x * scale_x) as i32).clamp(0, frame.width - 1) as usize,
            ((y * scale_y) as i32).clamp(0, frame.height - 1) as usize,
        );
        let offset = py * frame.stride + px * 4;
        let bgra = frame.data.get(offset..offset + 3)?;
        Some((bgra[2], bgra[1], bgra[0]))
    }
}

impl Screen for ColorPicker {
    fn outputs(&mut self, outputs: &[OutputDesc]) {
        self.outputs = outputs.to_vec();
    }

    fn render(&mut self, surface: usize, canvas: &mut Canvas<'_>, ctx: &Ctx) {
        /// Side of the magnified neighbourhood, in source pixels.
        const CELLS: i32 = 17;
        let theme = &ctx.theme;
        let (w, h) = canvas.size();
        let Some(frame) = self
            .outputs
            .get(surface)
            .and_then(|d| frame_named(&self.frames, &d.name))
        else {
            canvas.clear(Color::rgb(0x0010_1216));
            return;
        };
        let image = Image {
            data: &frame.data,
            width: frame.width,
            height: frame.height,
            stride: frame.stride,
        };
        canvas.blit(
            Rect::new(0, 0, w, h),
            &image,
            Rect::new(0, 0, frame.width, frame.height),
        );

        let Some((at_surface, x, y)) = self.at else {
            return;
        };
        if at_surface != surface {
            return;
        }
        // Magnifier: a 17×17 pixel neighbourhood blown up, with the centre
        // pixel ringed so there's no ambiguity about what will be sampled.
        let scale_x = f64::from(frame.width) / f64::from(w.max(1));
        let scale_y = f64::from(frame.height) / f64::from(h.max(1));
        let src_center = (
            ((x * scale_x) as i32).clamp(0, frame.width - 1),
            ((y * scale_y) as i32).clamp(0, frame.height - 1),
        );
        let lens = 170;
        let lens_rect = Rect::new(
            (x as i32 + 24).min(w - lens - 12),
            (y as i32 + 24).min(h - lens - 56),
            lens,
            lens,
        );
        canvas.blit(
            lens_rect,
            &image,
            Rect::new(
                src_center.0 - CELLS / 2,
                src_center.1 - CELLS / 2,
                CELLS,
                CELLS,
            ),
        );
        canvas.stroke_rounded(lens_rect, 8, 2, theme.border);
        let cell = lens / CELLS;
        canvas.stroke_rounded(
            Rect::new(
                lens_rect.x + cell * (CELLS / 2),
                lens_rect.y + cell * (CELLS / 2),
                cell,
                cell,
            ),
            0,
            1,
            Color::rgb(0xFF_FF_FF),
        );

        if let Some((r, g, b)) = self.sample(surface, x, y) {
            let swatch = Rect::new(lens_rect.x, lens_rect.bottom() + 8, lens, 34);
            canvas.fill_rounded(swatch, 8, theme.bg.with_alpha(0.94));
            canvas.fill_rounded(
                Rect::new(swatch.x + 6, swatch.y + 6, 22, 22),
                4,
                Color { r, g, b, a: 255 },
            );
            canvas.text_in(
                &ctx.fonts,
                Rect::new(swatch.x + 36, swatch.y, swatch.w - 42, swatch.h),
                13.0,
                false,
                theme.text,
                &format!("#{r:02X}{g:02X}{b:02X}"),
            );
        }
    }

    fn input(&mut self, surface: usize, event: &Input, _ctx: &Ctx) -> Flow {
        match event {
            Input::Motion { x, y } => {
                self.at = Some((surface, *x, *y));
                Flow::RedrawAll
            }
            Input::Leave => {
                if self.at.is_some_and(|(s, _, _)| s == surface) {
                    self.at = None;
                    return Flow::Redraw;
                }
                Flow::Idle
            }
            Input::Press { x, y, .. } => {
                if let Some((r, g, b)) = self.sample(surface, *x, *y) {
                    self.picked = Some((
                        f64::from(r) / 255.0,
                        f64::from(g) / 255.0,
                        f64::from(b) / 255.0,
                    ));
                }
                Flow::Done
            }
            Input::Key { key, .. } => {
                if key == &Key::Escape {
                    return Flow::Done;
                }
                Flow::Idle
            }
            Input::Release | Input::Scroll { .. } => Flow::Idle,
        }
    }
}
