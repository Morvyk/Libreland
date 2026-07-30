//! "Open with…" — the application chooser.
//!
//! Shown when an app asks the portal to pick a handler for a URI it can't open
//! itself. The candidate list comes from the frontend (which has already
//! matched the content type); we add a search box over every installed
//! application, because the useful case is precisely the one where the
//! suggested handlers are wrong.
//!
//! Icons are drawn as a coloured tile with the application's initial rather
//! than loaded from an icon theme: theme lookup means parsing index.theme
//! files, resolving inheritance, and rasterizing SVGs — a lot of machinery for
//! a list where the name is what people actually read.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel geometry: every value here is a surface- or image-sized non-negative integer, and the conversions between i32/u32/usize/f32 are all inside that range. Checked conversions at each site would be noise around arithmetic that cannot overflow."
)]

use std::sync::{Arc, Mutex};

use crate::apps::{self, DesktopApp};

use super::draw::{Canvas, Color, Rect};
use super::widgets::{self, Emphasis, Scroll, TextField};
use super::{Ctx, Flow, Input, Key, Screen};

pub struct AppChooser {
    /// The apps the frontend suggested, in its order (best match first).
    suggested: Vec<DesktopApp>,
    /// Everything installed, for the search fallback.
    all: Vec<DesktopApp>,
    /// Indices into whichever list is in play.
    visible: Vec<usize>,
    searching: bool,
    search: TextField,
    cursor: usize,
    scroll: Scroll,
    hover: Option<(f64, f64)>,
    rows_rect: Rect,
    open_rect: Rect,
    cancel_rect: Rect,
    search_rect: Rect,
    /// What the user chose, as a desktop id.
    pub chosen: Option<String>,
    heading: String,
    /// Filled by the portal when the frontend sends `UpdateChoices` while the
    /// dialog is already up (a slower content-type sniff finished). Drained on
    /// the next tick.
    inbox: Arc<Mutex<Option<Vec<String>>>>,
}

const ROW_H: i32 = 44;

impl AppChooser {
    pub fn new(
        heading: String,
        suggested: Vec<DesktopApp>,
        inbox: Arc<Mutex<Option<Vec<String>>>>,
    ) -> Self {
        let mut chooser = Self {
            suggested,
            all: crate::apps::scan(),
            visible: Vec::new(),
            searching: false,
            search: TextField::default(),
            cursor: 0,
            scroll: Scroll::default(),
            hover: None,
            rows_rect: Rect::default(),
            open_rect: Rect::default(),
            cancel_rect: Rect::default(),
            search_rect: Rect::default(),
            chosen: None,
            heading,
            inbox,
        };
        chooser.search.focused = true;
        chooser.refilter();
        chooser
    }

    /// Take any `UpdateChoices` payload the portal parked for us.
    fn drain_inbox(&mut self) -> bool {
        let Ok(mut slot) = self.inbox.lock() else {
            return false;
        };
        let Some(ids) = slot.take() else {
            return false;
        };
        drop(slot);
        self.suggested = ids.iter().filter_map(|id| apps::find(id)).collect();
        self.refilter();
        true
    }

    fn source(&self) -> &[DesktopApp] {
        if self.searching {
            &self.all
        } else {
            &self.suggested
        }
    }

    fn refilter(&mut self) {
        let needle = self.search.text.trim().to_lowercase();
        // An empty search shows the frontend's suggestions; typing widens the
        // net to everything installed.
        self.searching = !needle.is_empty();
        self.visible = self
            .source()
            .iter()
            .enumerate()
            .filter(|(_, app)| {
                needle.is_empty()
                    || app.name.to_lowercase().contains(&needle)
                    || app.id.to_lowercase().contains(&needle)
            })
            .map(|(i, _)| i)
            .collect();
        self.cursor = self.cursor.min(self.visible.len().saturating_sub(1));
    }

    fn accept(&mut self) -> Flow {
        let Some(&index) = self.visible.get(self.cursor) else {
            return Flow::Idle;
        };
        let Some(app) = self.source().get(index) else {
            return Flow::Idle;
        };
        // The portal spec wants the id without the `.desktop` suffix.
        self.chosen = Some(
            app.id
                .strip_suffix(".desktop")
                .unwrap_or(&app.id)
                .to_string(),
        );
        Flow::Done
    }

    /// A stable tile colour per application, so the same app looks the same
    /// every time the chooser opens.
    fn tile_color(app: &DesktopApp) -> Color {
        let hash = app
            .id
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(u32::from(b)));
        // Fixed saturation/value, hue from the hash: distinct but never garish.
        let hue = f32::from(u16::try_from(hash % 360).unwrap_or(0));
        let (r, g, b) = hsv_to_rgb(hue, 0.45, 0.75);
        Color { r, g, b, a: 255 }
    }
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> (u8, u8, u8) {
    let chroma = value * saturation;
    let second = chroma * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
    let base = value - chroma;
    let (red, green, blue) = match hue as u32 / 60 {
        0 => (chroma, second, 0.0),
        1 => (second, chroma, 0.0),
        2 => (0.0, chroma, second),
        3 => (0.0, second, chroma),
        4 => (second, 0.0, chroma),
        _ => (chroma, 0.0, second),
    };
    (
        ((red + base) * 255.0) as u8,
        ((green + base) * 255.0) as u8,
        ((blue + base) * 255.0) as u8,
    )
}

impl Screen for AppChooser {
    fn title(&self) -> String {
        "Open With".to_string()
    }

    fn size(&self) -> (i32, i32) {
        (520, 560)
    }

    fn tick(&mut self) -> Flow {
        let updated = self.drain_inbox();
        if updated | self.search.tick() {
            Flow::Redraw
        } else {
            Flow::Idle
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one layout pass for the whole dialog"
    )]
    fn render(&mut self, _surface: usize, canvas: &mut Canvas<'_>, ctx: &Ctx) {
        let theme = &ctx.theme;
        let (w, h) = canvas.size();
        let hover = self.hover;
        let hovered = |rect: Rect| hover.is_some_and(|(x, y)| rect.contains(x, y));
        canvas.clear(theme.bg);

        canvas.text_in(
            &ctx.fonts,
            Rect::new(24, 20, w - 48, 24),
            16.0,
            true,
            theme.text,
            "Open With",
        );
        canvas.text_in(
            &ctx.fonts,
            Rect::new(24, 46, w - 48, 20),
            13.0,
            false,
            theme.dim,
            &self.heading,
        );

        self.search_rect = Rect::new(24, 76, w - 48, 32);
        self.search
            .render(canvas, ctx, self.search_rect, "Search applications");

        let rows = Rect::new(24, 120, w - 48, h - 120 - 64);
        self.rows_rect = rows;
        canvas.fill_rounded(rows, 8, theme.surface);
        let content_h = self.visible.len() as f32 * ROW_H as f32;
        self.scroll.offset = self
            .scroll
            .offset
            .clamp(0.0, (content_h - rows.h as f32).max(0.0));

        canvas.clipped(rows, |c| {
            if self.visible.is_empty() {
                c.text_in(
                    &ctx.fonts,
                    Rect::new(rows.x + 16, rows.y + 14, rows.w - 32, 22),
                    13.0,
                    false,
                    theme.dim,
                    "No application found",
                );
                return;
            }
            let first = (self.scroll.offset / ROW_H as f32).floor().max(0.0) as usize;
            let last = (first + (rows.h / ROW_H) as usize + 2).min(self.visible.len());
            let source = if self.searching {
                &self.all
            } else {
                &self.suggested
            };
            for slot in first..last {
                let Some(app) = source.get(self.visible[slot]) else {
                    continue;
                };
                let y = rows.y + (slot as f32 * ROW_H as f32 - self.scroll.offset) as i32;
                let row = Rect::new(rows.x, y, rows.w, ROW_H);
                if slot == self.cursor {
                    c.fill(row, theme.accent.with_alpha(0.28));
                } else if hover.is_some_and(|(hx, hy)| row.contains(hx, hy)) {
                    c.fill(row, theme.raised);
                }
                let tile = Rect::new(row.x + 10, row.y + 7, 30, 30);
                c.fill_rounded(tile, 7, Self::tile_color(app));
                let initial = app
                    .name
                    .chars()
                    .next()
                    .map(|ch| ch.to_uppercase().to_string())
                    .unwrap_or_default();
                c.text_centered(&ctx.fonts, tile, 15.0, true, theme.on_accent, &initial);
                c.text_in(
                    &ctx.fonts,
                    Rect::new(row.x + 50, row.y + 5, row.w - 66, 18),
                    13.0,
                    false,
                    theme.text,
                    &app.name,
                );
                if !app.comment.is_empty() {
                    c.text_in(
                        &ctx.fonts,
                        Rect::new(row.x + 50, row.y + 22, row.w - 66, 16),
                        11.0,
                        false,
                        theme.dim,
                        &app.comment,
                    );
                }
            }
        });
        self.scroll.render(canvas, theme, rows, content_h);

        self.open_rect = Rect::new(w - 120, h - 48, 96, 32);
        self.cancel_rect = Rect::new(w - 230, h - 48, 100, 32);
        widgets::button(
            canvas,
            ctx,
            self.cancel_rect,
            "Cancel",
            Emphasis::Normal,
            hovered(self.cancel_rect),
            true,
        );
        widgets::button(
            canvas,
            ctx,
            self.open_rect,
            "Open",
            Emphasis::Primary,
            hovered(self.open_rect),
            !self.visible.is_empty(),
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
            Input::Scroll { dy, .. } => {
                self.scroll.by(
                    *dy,
                    self.visible.len() as f32 * ROW_H as f32,
                    f32::from(i16::try_from(self.rows_rect.h).unwrap_or(0)),
                );
                Flow::Redraw
            }
            Input::Press { x, y, .. } => {
                if self.cancel_rect.contains(*x, *y) {
                    return Flow::Done;
                }
                if self.open_rect.contains(*x, *y) {
                    return self.accept();
                }
                if self.rows_rect.contains(*x, *y) {
                    let slot = ((y - f64::from(self.rows_rect.y) + f64::from(self.scroll.offset))
                        / f64::from(ROW_H)) as usize;
                    if slot < self.visible.len() {
                        // Single click selects; the second click on an
                        // already-selected row opens, which is what a short
                        // list wants (no double-click timing to get right).
                        if self.cursor == slot {
                            return self.accept();
                        }
                        self.cursor = slot;
                    }
                    return Flow::Redraw;
                }
                Flow::Idle
            }
            Input::Release => Flow::Idle,
            Input::Key { key, mods } => {
                match key {
                    Key::Escape => return Flow::Done,
                    Key::Enter => return self.accept(),
                    Key::Up | Key::Down | Key::Home | Key::End => {
                        if self.visible.is_empty() {
                            return Flow::Idle;
                        }
                        let last = self.visible.len() - 1;
                        self.cursor = match key {
                            Key::Up => self.cursor.saturating_sub(1),
                            Key::Down => (self.cursor + 1).min(last),
                            Key::Home => 0,
                            _ => last,
                        };
                        self.scroll.reveal(
                            self.cursor,
                            ROW_H as f32,
                            self.rows_rect.h as f32,
                        );
                        return Flow::Redraw;
                    }
                    _ => {}
                }
                if self.search.key(key, *mods) {
                    self.refilter();
                    return Flow::Redraw;
                }
                Flow::Idle
            }
        }
    }
}
