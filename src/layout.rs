//! Window layout — dwindle tiling with a per-window floating flag.
//!
//! Tiled windows are assigned rects by the dwindle rule: a "current
//! cell" starts at the layout's bounds; each non-final tiled window
//! takes half of the current cell along the cell's longer axis; the
//! other half becomes the next current cell; the final tile takes
//! whatever remains. Floating windows are skipped by that pass and
//! keep their explicitly-stored rect.
//!
//! Storage invariant: every floating window sits *after* every
//! tiled window in the backing `Vec`. The renderer iterates in
//! order, so floats draw on top of tiles; the hit-tester iterates
//! in reverse, so floats win over tiles for pointer focus.
//!
//! Toggling between tiled and floating moves the window to the
//! end of its destination section (top of float stack / end of
//! tile order). A newly floating window is centred at ~70 % of
//! its previous tiled cell so the user sees a smooth size change
//! rather than a jarring jump.
//!
//! Coordinates are stored as `Physical` because the renderer
//! consumes physical pixels; for `scale = 1.0` outputs (the only
//! case 4d covers) physical and logical coincide numerically, so
//! the `Logical`-typed size we ship to `xdg_toplevel.configure` can
//! be cast component-wise. Per-output fractional scale lands with
//! 4d's later polish.

use smithay::reexports::wayland_server::Resource as _;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Physical, Point, Rectangle, Size};
use smithay::wayland::shell::xdg::ToplevelSurface;
use tracing::debug;

/// One window managed by the layout, plus its current placement.
#[derive(Debug, Clone)]
pub struct Window {
    pub toplevel: ToplevelSurface,
    pub rect: Rectangle<i32, Physical>,
    pub floating: bool,
}

/// Dwindle tiler bounded to a single rectangle in virtual-layout
/// space. 4d uses one Layout for the primary output; per-output
/// workspaces are a future milestone.
pub struct Layout {
    /// Tiled windows first, floating windows last (see the module
    /// doc for why). Order within each section is insertion-newest
    /// at the back.
    windows: Vec<Window>,
    /// Bounding rect every tiled window is laid out inside.
    bounds: Rectangle<i32, Physical>,
}

impl Layout {
    /// Build an empty layout that will place windows inside `bounds`.
    pub fn new(bounds: Rectangle<i32, Physical>) -> Self {
        Self {
            windows: Vec::new(),
            bounds,
        }
    }

    /// Append a new toplevel as a tiled window. Recomputes every
    /// tiled rect, then ships every window (tiled or floating) its
    /// updated configure. Where in the Vec it lands: end of the
    /// tiled section (= just before the first floating window).
    pub fn insert(&mut self, toplevel: ToplevelSurface) {
        let entry = Window {
            toplevel,
            rect: self.bounds,
            floating: false,
        };
        let pos = self
            .windows
            .iter()
            .position(|w| w.floating)
            .unwrap_or(self.windows.len());
        self.windows.insert(pos, entry);
        self.recompute();
        self.push_configures();
    }

    /// Remove a toplevel matching `surface` and reflow tiled windows.
    pub fn remove(&mut self, surface: &WlSurface) {
        let before = self.windows.len();
        self.windows.retain(|w| w.toplevel.wl_surface() != surface);
        if self.windows.len() == before {
            // Not in our list — never mapped, popup, etc. No
            // reflow needed.
            return;
        }
        self.recompute();
        self.push_configures();
    }

    /// Flip the floating flag on the window matching `surface`.
    /// Tiled → floating: window gets a centred rect at 70 % of its
    /// previous tiled cell and is moved to the top of the float
    /// stack. Floating → tiled: rejoins the dwindle flow at the
    /// end of the tiled section (recompute overwrites its rect).
    /// A missing surface (already destroyed, never tracked) is a
    /// silent no-op.
    pub fn toggle_floating(&mut self, surface: &WlSurface) {
        let Some(idx) = self
            .windows
            .iter()
            .position(|w| w.toplevel.wl_surface() == surface)
        else {
            return;
        };

        let now_floating = !self.windows[idx].floating;
        if now_floating {
            // Shrink to 70 % of current rect, keep the same centre,
            // so the user sees a continuous size change.
            let prev = self.windows[idx].rect;
            let new_size =
                Size::<i32, Physical>::new((prev.size.w * 7) / 10, (prev.size.h * 7) / 10);
            let new_loc = Point::<i32, Physical>::new(
                prev.loc.x + (prev.size.w - new_size.w) / 2,
                prev.loc.y + (prev.size.h - new_size.h) / 2,
            );
            self.windows[idx].rect = Rectangle::new(new_loc, new_size);
            self.windows[idx].floating = true;
            // Move to end of Vec → top of float stack.
            let entry = self.windows.remove(idx);
            self.windows.push(entry);
        } else {
            // Becoming tiled — clear the flag, move just before
            // the first floating window so the invariant holds.
            self.windows[idx].floating = false;
            let entry = self.windows.remove(idx);
            let pos = self
                .windows
                .iter()
                .position(|w| w.floating)
                .unwrap_or(self.windows.len());
            self.windows.insert(pos, entry);
        }
        self.recompute();
        self.push_configures();
    }

    /// Current window placements in render order: tiled first, then
    /// floating (which means floats draw on top).
    pub fn windows(&self) -> &[Window] {
        &self.windows
    }

    /// Find the topmost window whose rect contains `pos`. Reverse-
    /// iterates so floating windows (stored at the tail) are
    /// checked first — they always win over tiled windows they
    /// overlap.
    pub fn window_at(&self, pos: Point<i32, Physical>) -> Option<&Window> {
        self.windows.iter().rev().find(|w| {
            pos.x >= w.rect.loc.x
                && pos.x < w.rect.loc.x + w.rect.size.w
                && pos.y >= w.rect.loc.y
                && pos.y < w.rect.loc.y + w.rect.size.h
        })
    }

    /// Prepare a window to be the target of an interactive drag
    /// (move or resize). Promotes it to floating *in place* (no
    /// 70 % shrink — the user wants to drag from where they
    /// clicked) and raises it to the top of the float stack so
    /// subsequent pointer events draw / hit it first. Returns the
    /// rect the caller should record as the drag's "start rect";
    /// the caller will translate this rect with cursor deltas via
    /// [`Layout::set_floating_rect`]. Returns `None` if `surface`
    /// isn't tracked.
    pub fn start_drag_for(&mut self, surface: &WlSurface) -> Option<Rectangle<i32, Physical>> {
        let idx = self
            .windows
            .iter()
            .position(|w| w.toplevel.wl_surface() == surface)?;
        let was_tiled = !self.windows[idx].floating;
        if was_tiled {
            self.windows[idx].floating = true;
        }
        // Move to end of Vec → top of float stack. Skip the
        // remove/push if already at the end (e.g. a re-drag of
        // the topmost float).
        if idx != self.windows.len() - 1 {
            let entry = self.windows.remove(idx);
            self.windows.push(entry);
        }
        let rect = self.windows.last().expect("just pushed").rect;
        if was_tiled {
            // Tiles need to reflow now that one of them just left
            // the dwindle flow. Floating windows are unaffected.
            self.recompute();
            self.push_configures();
        }
        Some(rect)
    }

    /// Update the rect of a floating window during a drag. Pushes
    /// a fresh `xdg_toplevel.configure` so the client redraws at
    /// the new size; clients that don't honour resize fall back to
    /// stretching their old buffer. Silent no-op if `surface`
    /// isn't tracked or is currently tiled (tiles get their rect
    /// from the dwindle pass, not from this method).
    pub fn set_floating_rect(&mut self, surface: &WlSurface, rect: Rectangle<i32, Physical>) {
        let Some(window) = self
            .windows
            .iter_mut()
            .find(|w| w.toplevel.wl_surface() == surface)
        else {
            return;
        };
        if !window.floating {
            return;
        }
        window.rect = rect;
        let size = Size::<i32, Logical>::from((rect.size.w, rect.size.h));
        window.toplevel.with_pending_state(|state| {
            state.size = Some(size);
        });
        window.toplevel.send_configure();
    }

    /// Walk every tiled window in storage order and assign rects
    /// via dwindle. Floating windows are skipped — their rects
    /// stay wherever the user (or a previous toggle) put them.
    fn recompute(&mut self) {
        let tiled: Vec<usize> = self
            .windows
            .iter()
            .enumerate()
            .filter_map(|(i, w)| (!w.floating).then_some(i))
            .collect();
        let n = tiled.len();
        if n == 0 {
            return;
        }
        let mut cur = self.bounds;
        for (i, &idx) in tiled.iter().enumerate() {
            if i == n - 1 {
                self.windows[idx].rect = cur;
            } else {
                let dir = if cur.size.w >= cur.size.h {
                    SplitDir::Horizontal
                } else {
                    SplitDir::Vertical
                };
                let (a, b) = split_half(cur, dir);
                self.windows[idx].rect = a;
                cur = b;
            }
        }
    }

    /// Ship every window its current rect via `xdg_toplevel.configure`,
    /// regardless of tiled vs floating — both flavours need to keep
    /// their client buffer in sync with the rect we draw.
    fn push_configures(&self) {
        for window in &self.windows {
            let size = Size::<i32, Logical>::from((window.rect.size.w, window.rect.size.h));
            window.toplevel.with_pending_state(|state| {
                state.size = Some(size);
            });
            window.toplevel.send_configure();
            debug!(
                surface = ?window.toplevel.wl_surface().id(),
                x = window.rect.loc.x,
                y = window.rect.loc.y,
                w = window.rect.size.w,
                h = window.rect.size.h,
                floating = window.floating,
                "layout: configure sent",
            );
        }
    }
}

#[derive(Copy, Clone)]
enum SplitDir {
    Horizontal,
    Vertical,
}

fn split_half(
    r: Rectangle<i32, Physical>,
    dir: SplitDir,
) -> (Rectangle<i32, Physical>, Rectangle<i32, Physical>) {
    match dir {
        SplitDir::Horizontal => {
            let half = r.size.w / 2;
            let a = Rectangle::new(r.loc, Size::new(half, r.size.h));
            let b = Rectangle::new(
                Point::new(r.loc.x + half, r.loc.y),
                Size::new(r.size.w - half, r.size.h),
            );
            (a, b)
        }
        SplitDir::Vertical => {
            let half = r.size.h / 2;
            let a = Rectangle::new(r.loc, Size::new(r.size.w, half));
            let b = Rectangle::new(
                Point::new(r.loc.x, r.loc.y + half),
                Size::new(r.size.w, r.size.h - half),
            );
            (a, b)
        }
    }
}
