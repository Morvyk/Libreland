//! Window layout — dwindle tiling.
//!
//! Each new toplevel takes half of the previously-last cell, split
//! along that cell's longer axis. So:
//!
//! ```text
//! 1 window:   2 windows:   3 windows:    4 windows:
//! ┌──────┐   ┌───┬──┐    ┌───┬──┐      ┌───┬──┐
//! │  A   │   │ A │B │    │ A │B │      │ A │B │
//! │      │   │   │  │    │   ├──┤      │   ├──┤
//! └──────┘   └───┴──┘    └───┴──┘      └───┴┬─┘
//!                          (C below B)      │D│ (D right of C)
//! ```
//!
//! Removing a window collapses its cell into its sibling; the rest
//! of the tree reflows. The layout drives `xdg_toplevel.configure`
//! sizes — clients then commit buffers matching the assigned rect.
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

/// One tiled toplevel and the rect the layout currently assigns it.
#[derive(Debug, Clone)]
pub struct TiledWindow {
    pub toplevel: ToplevelSurface,
    pub rect: Rectangle<i32, Physical>,
}

/// Dwindle tiler bounded to a single rectangle in virtual-layout
/// space. 4d.1 uses one Layout for the primary output; per-output
/// workspaces are a future milestone.
pub struct Layout {
    tiled: Vec<TiledWindow>,
    bounds: Rectangle<i32, Physical>,
}

impl Layout {
    /// Build an empty layout that will place windows inside `bounds`.
    pub fn new(bounds: Rectangle<i32, Physical>) -> Self {
        Self {
            tiled: Vec::new(),
            bounds,
        }
    }

    /// Append a new toplevel to the tiling order, recompute every
    /// rect, then push the resulting sizes to clients via
    /// `xdg_toplevel.configure`. Order matters: a window inserted
    /// last lives in the deepest dwindle cell.
    pub fn insert(&mut self, toplevel: ToplevelSurface) {
        self.tiled.push(TiledWindow {
            toplevel,
            rect: self.bounds,
        });
        self.recompute();
        self.push_configures();
    }

    /// Remove a toplevel matching `surface` and reflow.
    pub fn remove(&mut self, surface: &WlSurface) {
        let before = self.tiled.len();
        self.tiled.retain(|w| w.toplevel.wl_surface() != surface);
        if self.tiled.len() == before {
            // Not in the tiled set — destroyed surface was already
            // removed, or this was never tiled (popup, never mapped,
            // …). Nothing to reflow.
            return;
        }
        self.recompute();
        self.push_configures();
    }

    /// Current window placements. Returned in insertion order, which
    /// is also draw order (last = top-most when stacking applies).
    pub fn windows(&self) -> &[TiledWindow] {
        &self.tiled
    }

    /// Find the topmost window whose rect contains `pos`. Tiled
    /// cells don't overlap, so iteration order doesn't change the
    /// result today; reverse-iterating is forward-compatible with
    /// the floating-mode stack order that lands in 4d.3.
    pub fn window_at(&self, pos: Point<i32, Physical>) -> Option<&TiledWindow> {
        self.tiled.iter().rev().find(|w| {
            pos.x >= w.rect.loc.x
                && pos.x < w.rect.loc.x + w.rect.size.w
                && pos.y >= w.rect.loc.y
                && pos.y < w.rect.loc.y + w.rect.size.h
        })
    }

    /// Walk every tiled window in order and assign rects using the
    /// dwindle rule: a "current cell" starts at `bounds`; each
    /// non-final window takes half of the current cell along the
    /// cell's longer axis, and the other half becomes the new
    /// current cell. The final window takes whatever remains.
    fn recompute(&mut self) {
        let n = self.tiled.len();
        if n == 0 {
            return;
        }
        let mut cur = self.bounds;
        for window in self.tiled.iter_mut().take(n - 1) {
            let dir = if cur.size.w >= cur.size.h {
                SplitDir::Horizontal
            } else {
                SplitDir::Vertical
            };
            let (a, b) = split_half(cur, dir);
            window.rect = a;
            cur = b;
        }
        // n >= 1 so this index is in bounds.
        self.tiled[n - 1].rect = cur;
    }

    /// Send each tiled window its new `xdg_toplevel.configure` so
    /// the client commits a buffer at the assigned size on its
    /// next paint. `xdg_toplevel.size` is in `Logical` pixels;
    /// at `scale = 1.0` (the only 4d case) the numeric values are
    /// identical to our `Physical` rect, so the cast is exact.
    fn push_configures(&self) {
        for window in &self.tiled {
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
            // Bias `a` by 1 px when the width is odd so `a + b ==
            // r.size.w` exactly — keeps the layout pixel-tight.
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
