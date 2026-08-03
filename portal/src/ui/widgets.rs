//! The handful of controls the dialogs are built from: buttons, a single-line
//! text field, a checkbox and a scrollbar.
//!
//! Immediate-mode with retained hit rects — a view computes its layout while
//! rendering, stashes the rectangles it drew, and tests pointer events against
//! them. That keeps layout in one place (the render function) instead of split
//! across a build step and an event step, which for dialogs this size is the
//! difference between one screenful of code and three.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel geometry: every value here is a surface- or image-sized non-negative integer, and the conversions between i32/u32/usize/f32 are all inside that range. Checked conversions at each site would be noise around arithmetic that cannot overflow."
)]

use super::draw::{Canvas, Color, Rect, Theme};
use super::{Ctx, Key, Mods};

/// Visual weight of a button.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Emphasis {
    /// The default action: filled with the accent colour.
    Primary,
    /// A normal action: filled with the raised surface colour.
    Normal,
    /// No fill until hovered — for toolbar/breadcrumb affordances.
    Flat,
    /// A destructive action.
    Danger,
}

/// Draw a button and return its rect (so the caller can record a hit box).
pub fn button(
    canvas: &mut Canvas<'_>,
    ctx: &Ctx,
    rect: Rect,
    label: &str,
    emphasis: Emphasis,
    hovered: bool,
    enabled: bool,
) {
    let theme = &ctx.theme;
    let (fill, fg) = match emphasis {
        Emphasis::Primary => (theme.accent, theme.on_accent),
        Emphasis::Danger => (theme.danger, theme.on_accent),
        Emphasis::Normal => (theme.raised, theme.text),
        Emphasis::Flat => (theme.raised.with_alpha(0.0), theme.text),
    };
    let fill = if !enabled {
        fill.with_alpha(0.35)
    } else if hovered {
        // Lift towards white in dark mode and towards black in light mode:
        // mixing with the text colour does both without a mode check.
        fill.mix(theme.text, 0.12)
    } else {
        fill
    };
    let fg = if enabled { fg } else { fg.with_alpha(0.4) };
    if emphasis != Emphasis::Flat || hovered {
        canvas.fill_rounded(rect, 6, fill);
    }
    if emphasis == Emphasis::Normal {
        canvas.stroke_rounded(rect, 6, 1, theme.border);
    }
    canvas.text_centered(&ctx.fonts, rect, 13.0, false, fg, label);
}

/// A checkbox with its label to the right. The returned rect covers both, so
/// clicking the label toggles.
pub fn checkbox(
    canvas: &mut Canvas<'_>,
    ctx: &Ctx,
    origin: (i32, i32),
    label: &str,
    checked: bool,
    hovered: bool,
) -> Rect {
    let theme = &ctx.theme;
    let box_rect = Rect::new(origin.0, origin.1 + 2, 16, 16);
    canvas.fill_rounded(
        box_rect,
        4,
        if checked { theme.accent } else { theme.raised },
    );
    if !checked {
        canvas.stroke_rounded(box_rect, 4, 1, theme.border);
    }
    if checked {
        // A tick drawn as stacked blocks: cheaper than a glyph lookup and it
        // stays crisp at every scale.
        let c = theme.on_accent;
        canvas.fill(Rect::new(box_rect.x + 4, box_rect.y + 8, 2, 4), c);
        canvas.fill(Rect::new(box_rect.x + 5, box_rect.y + 10, 2, 2), c);
        canvas.fill(Rect::new(box_rect.x + 7, box_rect.y + 8, 2, 2), c);
        canvas.fill(Rect::new(box_rect.x + 9, box_rect.y + 5, 2, 4), c);
    }
    let text_x = box_rect.right() + 8;
    let width = canvas.measure(&ctx.fonts, label, 13.0, false);
    let text_rect = Rect::new(text_x, origin.1, width as i32 + 2, 20);
    canvas.text_in(
        &ctx.fonts,
        text_rect,
        13.0,
        false,
        if hovered { theme.text } else { theme.dim },
        label,
    );
    Rect::new(
        box_rect.x,
        origin.1,
        text_rect.right() - box_rect.x,
        20.max(box_rect.h),
    )
}

/// A single-line editable text field.
///
/// Caret movement, selection-free editing, and horizontal scrolling when the
/// text outgrows the box. Deliberately no selection: a save-as filename field
/// with Ctrl+A/Home/End/word-delete covers what people actually do, and
/// selection rendering would double the size of this for the last 5%.
#[derive(Default)]
pub struct TextField {
    pub text: String,
    /// Caret position as a **byte** offset into `text`, always on a char
    /// boundary (every mutation goes through the helpers below).
    caret: usize,
    /// Horizontal scroll, in logical pixels.
    offset: f32,
    /// Blink phase, advanced by the owner's `tick`.
    blink: u8,
    pub focused: bool,
}

impl TextField {
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            caret: text.len(),
            text,
            offset: 0.0,
            blink: 0,
            focused: true,
        }
    }

    /// Put the caret at the end and make the field the input target.
    pub fn focus(&mut self) {
        self.focused = true;
        self.caret = self.text.len();
        self.blink = 0;
    }

    /// Select-nothing equivalent of "select all then type": used when a save
    /// dialog opens with a suggested name the user immediately replaces.
    pub fn set(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.caret = self.text.len();
        self.offset = 0.0;
    }

    /// Advance the blink phase; returns true when the caret needs repainting.
    pub fn tick(&mut self) -> bool {
        if !self.focused {
            return false;
        }
        self.blink = self.blink.wrapping_add(1);
        // 33 ms ticks, so 16 ticks ≈ half a second per phase.
        self.blink.is_multiple_of(16)
    }

    fn prev_boundary(&self, from: usize) -> usize {
        self.text[..from]
            .char_indices()
            .next_back()
            .map_or(0, |(i, _)| i)
    }

    fn next_boundary(&self, from: usize) -> usize {
        self.text[from..]
            .chars()
            .next()
            .map_or(from, |c| from + c.len_utf8())
    }

    /// Handle a key. Returns true if the field consumed it.
    pub fn key(&mut self, key: &Key, mods: Mods) -> bool {
        if !self.focused {
            return false;
        }
        self.blink = 0;
        match key {
            Key::Char(ch) if !mods.ctrl && !mods.alt => {
                self.text.insert(self.caret, *ch);
                self.caret += ch.len_utf8();
                true
            }
            Key::Backspace => {
                if mods.ctrl {
                    // Word-wise: eat trailing separators, then the word.
                    let head = self.text[..self.caret].trim_end();
                    let cut = head
                        .rfind(['/', ' ', '.', '-', '_'])
                        .map_or(0, |i| i + 1)
                        .min(self.caret);
                    self.text.replace_range(cut..self.caret, "");
                    self.caret = cut;
                } else if self.caret > 0 {
                    let prev = self.prev_boundary(self.caret);
                    self.text.replace_range(prev..self.caret, "");
                    self.caret = prev;
                }
                true
            }
            Key::Delete => {
                if self.caret < self.text.len() {
                    let next = self.next_boundary(self.caret);
                    self.text.replace_range(self.caret..next, "");
                }
                true
            }
            Key::Left => {
                self.caret = self.prev_boundary(self.caret);
                true
            }
            Key::Right => {
                self.caret = self.next_boundary(self.caret);
                true
            }
            Key::Home => {
                self.caret = 0;
                true
            }
            Key::End => {
                self.caret = self.text.len();
                true
            }
            _ => false,
        }
    }

    /// Draw the field. `placeholder` shows when it's empty.
    pub fn render(&mut self, canvas: &mut Canvas<'_>, ctx: &Ctx, rect: Rect, placeholder: &str) {
        let theme = &ctx.theme;
        canvas.fill_rounded(rect, 6, theme.surface);
        canvas.stroke_rounded(
            rect,
            6,
            1,
            if self.focused {
                theme.accent
            } else {
                theme.border
            },
        );
        let inner = Rect::new(rect.x + 10, rect.y, rect.w - 20, rect.h);
        let baseline = inner.y as f32 + f32::midpoint(inner.h as f32, 13.0 * 0.72);

        if self.text.is_empty() {
            canvas.text_in(&ctx.fonts, inner, 13.0, false, theme.dim, placeholder);
        }
        // Keep the caret in view: scroll only as far as needed, so the text
        // doesn't jump around while typing in the middle of a long name.
        let caret_x = canvas.measure(&ctx.fonts, &self.text[..self.caret], 13.0, false);
        if caret_x - self.offset > inner.w as f32 - 2.0 {
            self.offset = caret_x - inner.w as f32 + 2.0;
        }
        if caret_x < self.offset {
            self.offset = caret_x;
        }
        canvas.clipped(inner, |c| {
            c.text(
                &ctx.fonts,
                inner.x as f32 - self.offset,
                baseline,
                13.0,
                false,
                theme.text,
                &self.text,
            );
            if self.focused && self.blink % 32 < 16 {
                c.fill(
                    Rect::new(
                        inner.x + (caret_x - self.offset) as i32,
                        inner.y + 8,
                        1,
                        inner.h - 16,
                    ),
                    theme.text,
                );
            }
        });
    }
}

/// Vertical scroll state for a list of fixed-height rows.
#[derive(Default, Clone, Copy)]
pub struct Scroll {
    /// Pixels scrolled from the top.
    pub offset: f32,
}

impl Scroll {
    /// Apply a wheel/touchpad delta, clamped to the content.
    pub fn by(&mut self, dy: f64, content_height: f32, viewport_height: f32) {
        let delta = dy as f32;
        self.offset = (self.offset + delta).clamp(0.0, (content_height - viewport_height).max(0.0));
    }

    /// Scroll the row at `index` fully into view.
    pub fn reveal(&mut self, index: usize, row_height: f32, viewport_height: f32) {
        let top = index as f32 * row_height;
        if top < self.offset {
            self.offset = top;
        } else if top + row_height > self.offset + viewport_height {
            self.offset = top + row_height - viewport_height;
        }
        self.offset = self.offset.max(0.0);
    }

    /// Draw the scrollbar for a viewport, if the content overflows.
    pub fn render(
        self,
        canvas: &mut Canvas<'_>,
        theme: &Theme,
        viewport: Rect,
        content_height: f32,
    ) {
        let view_h = viewport.h as f32;
        if content_height <= view_h {
            return;
        }
        let track = Rect::new(viewport.right() - 8, viewport.y, 4, viewport.h);
        canvas.fill_rounded(track, 2, theme.border.with_alpha(0.5));
        let thumb_h = (view_h / content_height * view_h).max(24.0);
        let travel = view_h - thumb_h;
        let progress = self.offset / (content_height - view_h);
        let thumb = Rect::new(
            track.x,
            viewport.y + (travel * progress) as i32,
            track.w,
            thumb_h as i32,
        );
        canvas.fill_rounded(thumb, 2, theme.dim);
    }
}

/// A left-pointing chevron (used for "up one level" and breadcrumbs), drawn as
/// stacked pixels rather than a glyph so it never depends on the font's
/// symbol coverage.
pub fn chevron(canvas: &mut Canvas<'_>, at: (i32, i32), size: i32, right: bool, color: Color) {
    for i in 0..size {
        let dx = if right { i } else { size - 1 - i };
        canvas.fill(Rect::new(at.0 + dx, at.1 + i, 2, 2), color);
        canvas.fill(Rect::new(at.0 + dx, at.1 + 2 * size - 2 - i, 2, 2), color);
    }
}

/// A folder pictogram, again drawn rather than glyph-mapped.
pub fn folder_icon(canvas: &mut Canvas<'_>, rect: Rect, color: Color) {
    let tab = Rect::new(rect.x, rect.y + 2, rect.w * 2 / 5, 3);
    canvas.fill_rounded(tab, 1, color);
    canvas.fill_rounded(Rect::new(rect.x, rect.y + 4, rect.w, rect.h - 5), 2, color);
}

/// A document pictogram with a folded corner.
pub fn file_icon(canvas: &mut Canvas<'_>, rect: Rect, color: Color) {
    let body = Rect::new(rect.x + 2, rect.y + 1, rect.w - 4, rect.h - 2);
    canvas.fill_rounded(body, 2, color);
    // Notch the top-right corner by painting the background back over it.
    // The caller draws on an opaque row, so a transparent punch isn't an
    // option; use the same trick every icon theme does and just overlay a
    // lighter triangle.
    for i in 0..4 {
        canvas.fill(
            Rect::new(body.right() - 4 + i, body.y, 4 - i, 1 + i),
            color.mix(Color::rgba(0x00_00_00_00), 0.75),
        );
    }
}
