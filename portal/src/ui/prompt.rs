//! A message dialog with two buttons and optional toggles.
//!
//! Backs every portal interaction that is really just a question: `Access`
//! ("this app wants to use your camera"), `Account` ("share your user
//! information?"), `Background`, `Wallpaper`, `DynamicLauncher`. They differ
//! only in wording and in which toggles ride along, so they share one screen
//! rather than getting one dialog each.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel geometry: every value here is a surface- or image-sized non-negative integer, and the conversions between i32/u32/usize/f32 are all inside that range. Checked conversions at each site would be noise around arithmetic that cannot overflow."
)]

use super::draw::{Canvas, Rect};
use super::widgets::{self, Emphasis};
use super::{Ctx, Flow, Input, Key, Screen};

/// A toggle shown under the body text. Boolean choices (no options) render as
/// a checkbox; multi-valued ones cycle through their options on click, which
/// is enough for the two or three states these ever carry.
#[derive(Clone, Debug)]
pub struct Toggle {
    pub id: String,
    pub label: String,
    /// `(id, label)` pairs; empty means a boolean.
    pub options: Vec<(String, String)>,
    pub selected: String,
}

impl Toggle {
    fn is_bool(&self) -> bool {
        self.options.is_empty()
            || (self.options.len() == 2
                && self
                    .options
                    .iter()
                    .all(|(id, _)| id == "true" || id == "false"))
    }

    fn checked(&self) -> bool {
        self.selected == "true"
    }

    fn advance(&mut self) {
        if self.is_bool() {
            self.selected = if self.checked() { "false" } else { "true" }.to_string();
            return;
        }
        let next = self
            .options
            .iter()
            .position(|(id, _)| *id == self.selected)
            .map_or(0, |i| (i + 1) % self.options.len());
        if let Some((id, _)) = self.options.get(next) {
            self.selected = id.clone();
        }
    }

    fn display(&self) -> String {
        if self.is_bool() {
            return self.label.clone();
        }
        let value = self
            .options
            .iter()
            .find(|(id, _)| *id == self.selected)
            .map_or(self.selected.as_str(), |(_, label)| label.as_str());
        format!("{}: {value}", self.label)
    }
}

/// What the dialog should present.
#[derive(Clone, Debug, Default)]
pub struct Spec {
    pub title: String,
    pub subtitle: String,
    pub body: String,
    pub accept_label: Option<String>,
    pub deny_label: Option<String>,
    /// Draw the accept button as destructive.
    pub destructive: bool,
    pub toggles: Vec<Toggle>,
}

pub struct Prompt {
    spec: Spec,
    hover: Option<(f64, f64)>,
    hits: Vec<(Rect, usize)>,
    accept: Rect,
    deny: Rect,
    /// True once the user accepted; a cancel leaves it false.
    pub accepted: bool,
}

const WIDTH: i32 = 460;

impl Prompt {
    pub fn new(spec: Spec) -> Self {
        Self {
            spec,
            hover: None,
            hits: Vec::new(),
            accept: Rect::default(),
            deny: Rect::default(),
            accepted: false,
        }
    }

    /// The toggles as `(id, value)` pairs, for the results dictionary.
    pub fn choices(&self) -> Vec<(String, String)> {
        self.spec
            .toggles
            .iter()
            .map(|t| (t.id.clone(), t.selected.clone()))
            .collect()
    }

    /// Wrap `text` to `width` logical pixels, breaking on spaces.
    fn wrap(canvas: &Canvas<'_>, ctx: &Ctx, text: &str, width: f32) -> Vec<String> {
        let mut lines = Vec::new();
        for paragraph in text.split('\n') {
            let mut line = String::new();
            for word in paragraph.split_whitespace() {
                let candidate = if line.is_empty() {
                    word.to_string()
                } else {
                    format!("{line} {word}")
                };
                if canvas.measure(&ctx.fonts, &candidate, 13.0, false) > width && !line.is_empty() {
                    lines.push(std::mem::take(&mut line));
                    line = word.to_string();
                } else {
                    line = candidate;
                }
            }
            lines.push(line);
        }
        lines
    }
}

impl Screen for Prompt {
    fn title(&self) -> String {
        self.spec.title.clone()
    }

    fn size(&self) -> (i32, i32) {
        // The height follows the content. `size` is called before there's a
        // canvas to measure with, so line count is estimated from character
        // count at the fixed wrap width — an over-estimate just leaves a
        // little extra space under the text.
        let lines: i32 = self
            .spec
            .body
            .split('\n')
            .map(|p| i32::try_from(p.chars().count() / 62 + 1).unwrap_or(1))
            .sum();
        let toggles = i32::try_from(self.spec.toggles.len()).unwrap_or(0) * 30;
        (WIDTH, (150 + lines.max(1) * 20 + toggles).clamp(180, 520))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one layout pass: heading, wrapped body, toggles, buttons"
    )]
    fn render(&mut self, _surface: usize, canvas: &mut Canvas<'_>, ctx: &Ctx) {
        let theme = &ctx.theme;
        let (w, h) = canvas.size();
        let hover = self.hover;
        let hovered = |rect: Rect| hover.is_some_and(|(x, y)| rect.contains(x, y));
        canvas.clear(theme.bg);
        self.hits.clear();

        let mut y = 26;
        canvas.text_in(
            &ctx.fonts,
            Rect::new(24, y, w - 48, 24),
            16.0,
            true,
            theme.text,
            &self.spec.title,
        );
        y += 28;
        if !self.spec.subtitle.is_empty() {
            canvas.text_in(
                &ctx.fonts,
                Rect::new(24, y, w - 48, 22),
                13.0,
                false,
                theme.dim,
                &self.spec.subtitle,
            );
            y += 24;
        }
        if !self.spec.body.is_empty() {
            let lines = Self::wrap(canvas, ctx, &self.spec.body, (w - 48) as f32);
            for line in lines {
                canvas.text_in(
                    &ctx.fonts,
                    Rect::new(24, y, w - 48, 20),
                    13.0,
                    false,
                    theme.text,
                    &line,
                );
                y += 20;
            }
            y += 8;
        }

        for (index, toggle) in self.spec.toggles.iter().enumerate() {
            let rect = if toggle.is_bool() {
                widgets::checkbox(
                    canvas,
                    ctx,
                    (24, y),
                    &toggle.label,
                    toggle.checked(),
                    hovered(Rect::new(24, y, w - 48, 22)),
                )
            } else {
                let label = toggle.display();
                let width = canvas.measure(&ctx.fonts, &label, 13.0, false) as i32 + 32;
                let rect = Rect::new(24, y, width.min(w - 48), 28);
                widgets::button(
                    canvas,
                    ctx,
                    rect,
                    &label,
                    Emphasis::Normal,
                    hovered(rect),
                    true,
                );
                rect
            };
            self.hits.push((rect, index));
            y += 30;
        }

        // Buttons pin to the bottom, however the body wrapped.
        let deny_label = self
            .spec
            .deny_label
            .clone()
            .unwrap_or_else(|| "Deny".into());
        let accept_label = self
            .spec
            .accept_label
            .clone()
            .unwrap_or_else(|| "Allow".into());
        let accept_w = (canvas.measure(&ctx.fonts, &accept_label, 13.0, false) as i32 + 44).max(96);
        let deny_w = (canvas.measure(&ctx.fonts, &deny_label, 13.0, false) as i32 + 44).max(96);
        self.accept = Rect::new(w - accept_w - 24, h - 50, accept_w, 32);
        self.deny = Rect::new(self.accept.x - deny_w - 10, h - 50, deny_w, 32);
        widgets::button(
            canvas,
            ctx,
            self.deny,
            &deny_label,
            Emphasis::Normal,
            hovered(self.deny),
            true,
        );
        widgets::button(
            canvas,
            ctx,
            self.accept,
            &accept_label,
            if self.spec.destructive {
                Emphasis::Danger
            } else {
                Emphasis::Primary
            },
            hovered(self.accept),
            true,
        );
    }

    fn input(&mut self, _surface: usize, event: &Input, _ctx: &Ctx) -> Flow {
        match event {
            Input::Motion { x, y } => {
                self.hover = Some((*x, *y));
                Flow::Redraw
            }
            Input::Leave => {
                self.hover = None;
                Flow::Redraw
            }
            Input::Press { x, y, .. } => {
                if self.accept.contains(*x, *y) {
                    self.accepted = true;
                    return Flow::Done;
                }
                if self.deny.contains(*x, *y) {
                    return Flow::Done;
                }
                if let Some(index) = self
                    .hits
                    .iter()
                    .find(|(rect, _)| rect.contains(*x, *y))
                    .map(|(_, i)| *i)
                    && let Some(toggle) = self.spec.toggles.get_mut(index)
                {
                    toggle.advance();
                    return Flow::Redraw;
                }
                Flow::Idle
            }
            Input::Key { key, .. } => match key {
                Key::Escape => Flow::Done,
                Key::Enter => {
                    self.accepted = true;
                    Flow::Done
                }
                _ => Flow::Idle,
            },
            Input::Release | Input::Scroll { .. } => Flow::Idle,
        }
    }
}
