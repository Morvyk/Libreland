//! Window layout — binary tree of splits with cursor-driven snap.
//!
//! Tiled windows are leaves in a binary tree. Each interior
//! `Split` node carries an axis (`LeftRight` for side-by-side
//! cells, `TopBottom` for stacked cells) and a ratio. Adding a
//! window descends the tree to the leaf whose rect contains the
//! cursor and replaces that leaf with a fresh `Split` whose two
//! children are the existing leaf and the new one. The split
//! axis is picked perpendicular to the target leaf's longer
//! side so the resulting cells stay roughly square; which child
//! is "first" (= left/top) depends on which side of the leaf's
//! centre the cursor sits.
//!
//! Removing a window collapses the matched leaf — its parent
//! `Split` is replaced in place by the sibling subtree, which
//! cascades upward as expected.
//!
//! Each output has a dynamic list of **workspaces** (niri-style):
//! one dwindle tree + its own floating stack per workspace, and an
//! active index. Only the active workspace of an output is rendered.
//! `Super`+scroll switches the workspace on the output under the
//! cursor (a fresh trailing-empty workspace is materialized on
//! demand; empty non-active workspaces are compacted away);
//! `Super`+`Shift`+scroll moves the focused window to the adjacent
//! workspace on its own output and follows it.
//!
//! Floating windows live in a per-workspace `Vec` and always draw on
//! top of that workspace's tree. Toggling a window between tiled and
//! floating (`Super+F`) removes it from one set and inserts into the
//! other on the same workspace; the tree-side promote applies a 70 %
//! centre shrink so the transition reads as a smooth resize.
//!
//! Interactive **move** drags (`Super+LMB`) pull the window out
//! of its current set into `in_transit`, where it follows the
//! cursor as a free-floating rect. On release:
//! - if the source was **tiled**, the window is re-inserted into
//!   the tree at the cursor's drop position — the same insertion
//!   rule that drives spawn-at-cursor, so two windows can swap
//!   places by dragging one onto the other.
//! - if the source was **floating**, the window stays floating
//!   at its drop rect and rejoins the top of the float stack.
//!
//! Interactive **resize** drags (`Super+RMB`) only act on
//! floating windows — tiled cells can't be resized today without
//! a separate "drag the split divider" gesture, which is later
//! polish. Resize on a tile is a logged no-op.
//!
//! Coordinates are stored in **compositor** (= logical) pixels but
//! carry the `Physical` type tag — a historical quirk. Every output
//! occupies a rect in this one shared compositor space; the layout
//! keeps a separate dwindle tree per output and a window tiles only
//! within its output's rect. The renderer multiplies these
//! coordinates by the target output's fractional scale when it
//! composites (so `HiDPI` works) and ships the same values as the
//! `Logical`-typed `xdg_toplevel.configure` size.

use std::time::Instant;

use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
use smithay::reexports::wayland_server::Resource as _;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Physical, Point, Rectangle, Size};
use smithay::wayland::shell::xdg::ToplevelSurface;
use tracing::debug;

use crate::config::AnimSpec;

/// How a window fills its output. `Maximized` and `Fullscreen` both
/// cover the window's whole output with no border or rounded corners
/// and draw on top of normal windows; the state lives on the `Window`
/// so it travels when the window is moved between workspaces (a
/// maximized/fullscreen window stays that way). The two differ only
/// in the `xdg_toplevel` state flag we send (clients render
/// differently) and in z-order: a fullscreen window draws above
/// layer-shell panels too, a maximized one stays below them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FillMode {
    #[default]
    Normal,
    Maximized,
    Fullscreen,
}

/// One window managed by the layout, plus its current placement.
/// The `rect` is the cell the layout has assigned (refreshed by
/// every reflow) — clients see the same size via
/// `xdg_toplevel.configure`.
#[derive(Debug, Clone)]
pub struct Window {
    pub toplevel: ToplevelSurface,
    pub rect: Rectangle<i32, Physical>,
    /// Maximized/fullscreen override: when set, the window fills its
    /// output (ignoring `rect`), drops its border/corners, and draws
    /// on top. Travels with the window across workspaces.
    pub fill: FillMode,
}

/// Origin of an in-flight interactive move. Decides what happens
/// to the dragged window when the user releases the button —
/// tiled drops re-enter the tree at the cursor; floating drops
/// stay floating at their final rect.
#[derive(Debug, Clone, Copy)]
pub enum DragSource {
    Tiled,
    Floating,
}

/// Window currently being moved by an interactive drag. Drawn at
/// `window.rect` (which the caller updates with cursor deltas);
/// the entry sits outside the tree and the floating list until
/// the drag finishes.
pub struct InTransit {
    pub window: Window,
    pub source: DragSource,
    /// Output index the drag started on. The source workspace is
    /// emptied at drag start but not normalized (the drag may abort
    /// back), so `finish_move_drag` normalizes this output to reap a
    /// workspace the drag emptied.
    source_output: usize,
}

/// One workspace: a dwindle tree of tiled windows plus that
/// workspace's own floating stack. A workspace is a self-contained
/// scene — only the active workspace of each output is rendered, so
/// floating windows are scoped here (not global) and don't bleed
/// across workspaces.
#[derive(Default)]
struct Workspace {
    tree: Option<Node>,
    floating: Vec<Window>,
}

impl Workspace {
    /// No tiled windows and no floats — a candidate for compaction
    /// (and what the trailing scroll-into slot looks like).
    fn is_empty(&self) -> bool {
        self.tree.is_none() && self.floating.is_empty()
    }
}

/// One output's region: a stable connector name, that output's full
/// rect in absolute compositor space, and its dynamic list of
/// workspaces. Invariants (maintained by `normalize_output`):
/// `workspaces.len() >= 1`, `active < workspaces.len()`, and the
/// last workspace is always empty (the trailing slot you scroll
/// into). A window's owning output is implicit in which `Outpane`
/// holds it; the renderer paints each window on whichever CRTC its
/// absolute rect falls on.
struct Outpane {
    name: String,
    /// Entire physical output rect. Fullscreen windows fill this so
    /// they cover layer-shell panels.
    full: Rectangle<i32, Physical>,
    /// Usable work area — `full` minus any layer-shell exclusive zones
    /// (panels). Tiling lays out inside this and maximized windows
    /// fill it, so a panel stays visible.
    bounds: Rectangle<i32, Physical>,
    workspaces: Vec<Workspace>,
    active: usize,
    /// In-flight workspace switch animation, if any. Holds a *snapshot*
    /// of the outgoing workspace's placements (immune to the workspace
    /// reindexing `normalize_output` does on switch) plus the direction
    /// and start time, so the slide can render both workspaces.
    transition: Option<WsTransition>,
}

/// A workspace-switch slide in progress on one output.
struct WsTransition {
    /// Outgoing workspace's placements, captured at switch time.
    from: Vec<Placement>,
    /// Slide direction: `+1` slides everything down (incoming from the
    /// top), `-1` slides up (incoming from the bottom). Switching to the
    /// next workspace slides up; to the previous, down.
    dir: i32,
    /// When the slide began.
    start: Instant,
}

/// The two rects a fill mode can target: `full` (entire output, for
/// fullscreen) and `work` (output minus exclusive zones, for maximized
/// + tiling). Bundled so the configure helpers can resolve per-window.
#[derive(Clone, Copy)]
struct OutputArea {
    full: Rectangle<i32, Physical>,
    work: Rectangle<i32, Physical>,
}

impl OutputArea {
    /// The rect a window with the given fill mode should occupy.
    /// `Normal` callers never use the result (they keep their cell).
    fn fill(self, mode: FillMode) -> Rectangle<i32, Physical> {
        match mode {
            FillMode::Fullscreen => self.full,
            FillMode::Maximized | FillMode::Normal => self.work,
        }
    }
}

impl Outpane {
    fn new(name: String, bounds: Rectangle<i32, Physical>) -> Self {
        Self {
            name,
            // A fresh output has no panels, so the work area is the
            // full output until an exclusive zone shrinks `bounds`.
            full: bounds,
            bounds,
            // A fresh output is one empty workspace; index 0 doubles
            // as the active and the trailing-empty slot until a
            // window lands on it.
            workspaces: vec![Workspace::default()],
            active: 0,
            transition: None,
        }
    }

    fn area(&self) -> OutputArea {
        OutputArea {
            full: self.full,
            work: self.bounds,
        }
    }
}

/// Per-output dynamic workspaces. Each output owns a `Vec<Workspace>`
/// (each workspace owns its own tree + floating stack) and an active
/// index. Only the active workspace of an output is emitted by
/// [`Layout::placements`] / rendered. The in-transit drag is global
/// and transient — it follows the cursor across outputs/workspaces
/// and only commits to a concrete home on release. All coordinates
/// are absolute compositor pixels.
pub struct Layout {
    outputs: Vec<Outpane>,
    in_transit: Option<InTransit>,
    gaps: Gaps,
    border_width: i32,
}

/// One window + its current placement, as the renderer consumes
/// it. `cell_rect` is the full cell the layout allocates; the
/// renderer paints the border in `cell_rect` and the surface
/// inside it (`cell_rect` shrunk by `border_width`).
#[derive(Debug, Clone)]
pub struct Placement {
    pub surface: WlSurface,
    pub cell_rect: Rectangle<i32, Physical>,
    pub focused: bool,
    /// Fill mode — the renderer suppresses the border/rounded corners
    /// for non-`Normal` placements and draws them in a higher z-bucket
    /// (maximized above windows, fullscreen above panels too).
    pub fill: FillMode,
    /// `true` for floating (and in-transit) windows, which draw above the
    /// tiled tree. The renderer uses this to pick the blur backdrop tier:
    /// tiled windows blur against the base (wallpaper + lower layers),
    /// floating windows against the base *plus* the tiled windows beneath.
    pub floating: bool,
    /// Extra vertical offset (compositor px) the renderer adds *after*
    /// per-window animation, used for the workspace slide so both the
    /// outgoing and incoming workspaces translate together without
    /// disturbing each window's own move animation (`cell_rect` stays
    /// the settled target). `0` outside a workspace transition.
    pub slide_dy: i32,
}

/// Gap configuration. `outer` is empty space between the tile
/// area and each edge of an output's bounds; `inner` is empty
/// space between adjacent tile cells, centred on each split.
/// Floating windows are unaffected by both — they're positioned
/// freely by the user.
#[derive(Debug, Clone, Copy)]
pub struct Gaps {
    pub outer: i32,
    pub inner: i32,
}

/// One window's structural info for the IPC `windows` query. The caller
/// (the IPC dispatcher, which holds `State`) reads title/app-id off the
/// surface and pairs it with a stable id.
pub struct WindowEntry {
    pub surface: WlSurface,
    /// Cell rect in absolute compositor (logical) pixels.
    pub rect: Rectangle<i32, Physical>,
    pub fill: FillMode,
    pub floating: bool,
    pub output: String,
    pub workspace: usize,
}

/// One workspace's info for the IPC `workspaces` query.
pub struct WorkspaceEntry {
    pub output: String,
    pub index: usize,
    pub active: bool,
    pub windows: usize,
}

enum Node {
    Leaf(Window),
    Split {
        axis: SplitAxis,
        ratio: f32,
        first: Box<Node>,
        second: Box<Node>,
    },
}

#[derive(Debug, Clone, Copy)]
enum SplitAxis {
    /// Cells positioned left-right; the split divider is vertical.
    /// `first` is the left cell, `second` is the right cell.
    LeftRight,
    /// Cells positioned top-bottom; the split divider is horizontal.
    /// `first` is the top cell, `second` is the bottom cell.
    TopBottom,
}

impl Layout {
    /// Build a layout spanning every output. `outputs` pairs each
    /// output's stable connector name with its full rect in absolute
    /// compositor pixels. Windows tile within the output the cursor
    /// is over at spawn / drop time.
    pub fn new(
        outputs: impl IntoIterator<Item = (String, Rectangle<i32, Physical>)>,
        gaps: Gaps,
        border_width: i32,
    ) -> Self {
        Self {
            outputs: outputs
                .into_iter()
                .map(|(name, bounds)| Outpane::new(name, bounds))
                .collect(),
            in_transit: None,
            gaps,
            border_width: border_width.max(0),
        }
    }

    /// Tile area of the output at `idx`: its bounds shrunk by the
    /// outer gap. Every workspace of an output shares its tile area.
    fn tile_bounds(&self, idx: usize) -> Rectangle<i32, Physical> {
        shrink_for_outer(self.outputs[idx].bounds, self.gaps.outer)
    }

    /// The active (visible) workspace of output `oi`.
    fn active_ws(&self, oi: usize) -> &Workspace {
        let o = &self.outputs[oi];
        &o.workspaces[o.active]
    }

    /// Mutable handle to the active workspace of output `oi`. Bind
    /// this once when mutating, so the borrow checker doesn't choke
    /// on `outputs[oi].workspaces[active]` being indexed twice.
    fn active_ws_mut(&mut self, oi: usize) -> &mut Workspace {
        let o = &mut self.outputs[oi];
        &mut o.workspaces[o.active]
    }

    /// Index of the first output whose **full** bounds contain `p`.
    /// Full bounds (not the gap-shrunk tile area) so a point in an
    /// output's outer-gap margin still resolves to that output
    /// instead of falling through to the fallback.
    fn outpane_at(&self, p: Point<i32, Physical>) -> Option<usize> {
        self.outputs.iter().position(|o| rect_contains(o.bounds, p))
    }

    /// Pick the output a new / dropped window belongs to: the one
    /// the cursor is over, else the first output as a sensible
    /// default, else `None` when there are no outputs at all.
    fn outpane_for_point(&self, p: Option<Point<i32, Physical>>) -> Option<usize> {
        p.and_then(|c| self.outpane_at(c))
            .or_else(|| (!self.outputs.is_empty()).then_some(0))
    }

    /// Insert a freshly-mapped toplevel. When `cursor` is `Some`,
    /// the new window splits whichever existing leaf contains
    /// that point — so a window opened over a particular cell
    /// makes room there. When `cursor` is `None` (no pointer
    /// position known) or doesn't land in any leaf, the new
    /// window splits the deepest leaf as a fallback. The first
    /// window in an empty layout becomes the root, full bounds.
    pub fn insert(&mut self, toplevel: ToplevelSurface, cursor: Option<Point<i32, Physical>>) {
        // Tile the new window on the output under the cursor (else
        // the first output). With no outputs at all there's nowhere
        // to put it — silent no-op.
        let Some(idx) = self.outpane_for_point(cursor) else {
            return;
        };
        let tile_bounds = self.tile_bounds(idx);
        let inner = self.gaps.inner;
        let window = Window {
            toplevel,
            rect: tile_bounds,
            fill: FillMode::Normal,
        };
        let leaf = Node::Leaf(window);
        // Lands on the visible (active) workspace of that output.
        let ws = self.active_ws_mut(idx);
        ws.tree = Some(match ws.tree.take() {
            None => leaf,
            Some(root) => insert_at_cursor(root, leaf, tile_bounds, cursor, inner),
        });
        self.recompute_and_push();
    }

    /// Remove a toplevel matching `surface` from wherever it lives
    /// (tree, floating list, or `in_transit`). Reflows the tree if
    /// something changed there. Silent no-op for surfaces we
    /// don't track.
    pub fn remove(&mut self, surface: &WlSurface) {
        // A client may close while on a non-active workspace, so scan
        // every workspace of every output (index loops because
        // `normalize_output` needs `&mut self` afterwards).
        for oi in 0..self.outputs.len() {
            for wi in 0..self.outputs[oi].workspaces.len() {
                let ws = &mut self.outputs[oi].workspaces[wi];
                if let Some(root) = ws.tree.take() {
                    let (root_after, removed) = remove_from_tree(root, surface);
                    ws.tree = root_after;
                    if removed.is_some() {
                        self.normalize_output(oi);
                        self.recompute_and_push();
                        return;
                    }
                }
                let ws = &mut self.outputs[oi].workspaces[wi];
                let len = ws.floating.len();
                ws.floating.retain(|w| w.toplevel.wl_surface() != surface);
                if ws.floating.len() != len {
                    self.normalize_output(oi);
                    self.recompute_and_push();
                    return;
                }
            }
        }
        if self
            .in_transit
            .as_ref()
            .is_some_and(|t| t.window.toplevel.wl_surface() == surface)
        {
            self.in_transit = None;
        }
    }

    /// Flip the window between tiled and floating.
    ///
    /// Tiled → floating: leaf is removed from the tree, its rect
    /// shrinks to 70 % centred on its previous centre (so the
    /// transition reads as a smooth resize rather than a jump),
    /// and the entry is pushed to the top of the float stack.
    /// Remaining tiles reflow.
    ///
    /// Floating → tiled: entry is removed from the float list and
    /// re-inserted into the tree at the window's current centre,
    /// so it lands where it visually was.
    ///
    /// Silent no-op for surfaces we don't track.
    pub fn toggle_floating(&mut self, surface: &WlSurface) {
        // Toggling never crosses workspaces: the window stays on the
        // active workspace it's currently visible on.
        //
        // Tile -> float: scan each output's ACTIVE workspace tree.
        for oi in 0..self.outputs.len() {
            let area = self.outputs[oi].area();
            let ws = self.active_ws_mut(oi);
            if let Some(root) = ws.tree.take() {
                let (root_after, removed) = remove_from_tree(root, surface);
                ws.tree = root_after;
                if let Some(mut window) = removed {
                    let prev = window.rect;
                    let new_size =
                        Size::<i32, Physical>::new((prev.size.w * 7) / 10, (prev.size.h * 7) / 10);
                    let new_loc = Point::<i32, Physical>::new(
                        prev.loc.x + (prev.size.w - new_size.w) / 2,
                        prev.loc.y + (prev.size.h - new_size.h) / 2,
                    );
                    window.rect = Rectangle::new(new_loc, new_size);
                    push_configure_for_floating(&window, self.border_width, area);
                    self.active_ws_mut(oi).floating.push(window);
                    self.recompute_and_push();
                    return;
                }
            }
        }
        // Float -> tile: find the float on whichever output's active
        // workspace holds it, and re-tile it into that same active
        // workspace's tree.
        for oi in 0..self.outputs.len() {
            let ws = self.active_ws_mut(oi);
            let Some(fidx) = ws
                .floating
                .iter()
                .position(|w| w.toplevel.wl_surface() == surface)
            else {
                continue;
            };
            let window = ws.floating.remove(fidx);
            let center = Point::<i32, Physical>::new(
                window.rect.loc.x + window.rect.size.w / 2,
                window.rect.loc.y + window.rect.size.h / 2,
            );
            let tile_bounds = self.tile_bounds(oi);
            let inner = self.gaps.inner;
            let leaf = Node::Leaf(window);
            let ws = self.active_ws_mut(oi);
            ws.tree = Some(match ws.tree.take() {
                None => leaf,
                Some(root) => insert_at_cursor(root, leaf, tile_bounds, Some(center), inner),
            });
            self.recompute_and_push();
            return;
        }
    }

    /// Start an interactive *move* drag. Pulls the matched window
    /// out of the tree (with a reflow) or the floating list and
    /// stores it as `in_transit`; returns its rect at the moment
    /// the drag started so the caller can record it for drag-math.
    /// Returns `None` if `surface` isn't tracked or another drag
    /// is already in flight.
    pub fn start_move_drag(&mut self, surface: &WlSurface) -> Option<Rectangle<i32, Physical>> {
        if self.in_transit.is_some() {
            return None;
        }
        // A maximized/fullscreen window owns its whole output; moving
        // it is meaningless (and would desync its filled configure), so
        // refuse the drag — the user unmaximizes first.
        if self.is_filled(surface) {
            return None;
        }
        // Only a visible window can be dragged, so scan active
        // workspaces only. We don't normalize the emptied source
        // workspace here (the drag may abort back); finish_move_drag
        // normalizes `source_output`.
        for oi in 0..self.outputs.len() {
            let ws = self.active_ws_mut(oi);
            if let Some(root) = ws.tree.take() {
                let (root_after, removed) = remove_from_tree(root, surface);
                ws.tree = root_after;
                if let Some(window) = removed {
                    let rect = window.rect;
                    self.in_transit = Some(InTransit {
                        window,
                        source: DragSource::Tiled,
                        source_output: oi,
                    });
                    self.recompute_and_push();
                    return Some(rect);
                }
            }
            let ws = self.active_ws_mut(oi);
            if let Some(fidx) = ws
                .floating
                .iter()
                .position(|w| w.toplevel.wl_surface() == surface)
            {
                let window = ws.floating.remove(fidx);
                let rect = window.rect;
                self.in_transit = Some(InTransit {
                    window,
                    source: DragSource::Floating,
                    source_output: oi,
                });
                return Some(rect);
            }
        }
        None
    }

    /// Start an interactive *resize* drag. Only floating windows
    /// can be drag-resized today; resize on a tile is rejected so
    /// the caller can log + swallow the press. Returns the rect
    /// to use as the drag's start rect, or `None`.
    pub fn start_resize_drag(&self, surface: &WlSurface) -> Option<Rectangle<i32, Physical>> {
        // Can't drag-resize a window that's pinned to fill its output.
        if self.is_filled(surface) {
            return None;
        }
        self.outputs.iter().find_map(|op| {
            op.workspaces[op.active]
                .floating
                .iter()
                .find(|w| w.toplevel.wl_surface() == surface)
                .map(|w| w.rect)
        })
    }

    /// Update the `in_transit` window's rect during a move drag
    /// and ship the corresponding configure. Silent no-op when
    /// nothing is in transit.
    pub fn update_in_transit_rect(&mut self, rect: Rectangle<i32, Physical>) {
        // Drag-start refuses filled windows, so the in-transit window
        // is always Normal here and the area goes unused by the
        // floating configure; resolve it from the source output anyway.
        let area = self
            .in_transit
            .as_ref()
            .and_then(|t| self.outputs.get(t.source_output))
            .map_or_else(
                || OutputArea {
                    full: Rectangle::default(),
                    work: Rectangle::default(),
                },
                Outpane::area,
            );
        if let Some(t) = &mut self.in_transit {
            t.window.rect = rect;
            // An in-transit window is conceptually floating until
            // it either drops onto a tile cell or rejoins the
            // float stack, so configure it as such (no Tiled*
            // states, free-form resize).
            push_configure_for_floating(&t.window, self.border_width, area);
        }
    }

    /// Update a floating window's rect during a resize drag and
    /// ship the corresponding configure. Silent no-op for surfaces
    /// that aren't currently floating.
    pub fn set_floating_rect(&mut self, surface: &WlSurface, rect: Rectangle<i32, Physical>) {
        let border = self.border_width;
        for op in &mut self.outputs {
            let active = op.active;
            let area = op.area();
            if let Some(window) = op.workspaces[active]
                .floating
                .iter_mut()
                .find(|w| w.toplevel.wl_surface() == surface)
            {
                window.rect = rect;
                push_configure_for_floating(window, border, area);
                return;
            }
        }
    }

    /// Finish an interactive move drag at `cursor`.
    ///
    /// - `Tiled` source: re-insert the dragged window into the
    ///   tree at the cursor's drop position (same rule as spawn).
    /// - `Floating` source: window goes back into the floating
    ///   list at the top of the stack, with whatever rect it
    ///   has now.
    ///
    /// Silent no-op when there's no drag in flight.
    pub fn finish_move_drag(&mut self, cursor: Point<i32, Physical>) {
        let Some(t) = self.in_transit.take() else {
            return;
        };
        let source_output = t.source_output;
        let center = Point::<i32, Physical>::new(
            t.window.rect.loc.x + t.window.rect.size.w / 2,
            t.window.rect.loc.y + t.window.rect.size.h / 2,
        );
        // Resolve the destination output: under the drop cursor, else
        // under the window's centre (cursor in a monitor gap), else
        // the first output. `None` only when there are no outputs.
        let Some(idx) = self
            .outpane_at(cursor)
            .or_else(|| self.outpane_at(center))
            .or_else(|| (!self.outputs.is_empty()).then_some(0))
        else {
            // No outputs at all: nowhere visible to home it. The
            // in-transit window is always Normal (drag-start refuses
            // filled windows), so the area here goes unused.
            push_configure_for_floating(
                &t.window,
                self.border_width,
                OutputArea {
                    full: Rectangle::default(),
                    work: Rectangle::default(),
                },
            );
            return;
        };
        match t.source {
            DragSource::Tiled => {
                // Re-tile into the destination output's ACTIVE
                // workspace. Dropping on another monitor (or, mid-drag
                // workspace switch, a different workspace) re-tiles
                // into whatever is now visible there.
                let tile_bounds = self.tile_bounds(idx);
                let inner = self.gaps.inner;
                let leaf = Node::Leaf(t.window);
                let ws = self.active_ws_mut(idx);
                ws.tree = Some(match ws.tree.take() {
                    None => leaf,
                    Some(root) => insert_at_cursor(root, leaf, tile_bounds, Some(cursor), inner),
                });
            }
            DragSource::Floating => {
                push_configure_for_floating(&t.window, self.border_width, self.outputs[idx].area());
                self.active_ws_mut(idx).floating.push(t.window);
            }
        }
        // Normalize both the destination (gained a window) and the
        // source (its workspace was emptied at drag start, never
        // reaped) so any phantom empty workspace is compacted.
        self.normalize_output(idx);
        if source_output != idx {
            self.normalize_output(source_output);
        }
        self.recompute_and_push();
    }

    /// Renderer snapshot: every visible window with its cell rect
    /// and a focused flag, in **bottom-up** draw order. The
    /// renderer paints each placement individually (border then
    /// surface) in this order, so floating windows draw on top
    /// of tiles and the in-transit drag follower draws on top of
    /// everything.
    ///
    /// Order: tiled leaves (which don't overlap each other) then
    /// floating bottom-of-stack upward then in-transit (top).
    ///
    /// `focused` lets the caller mark which surface gets the
    /// `active` border colour; the focus surface is owned by the
    /// seat, not the layout, so it comes in as a parameter.
    pub fn placements(&self, focused: Option<&WlSurface>, slide: Option<AnimSpec>) -> Vec<Placement> {
        let is_focused = |surface: &WlSurface| focused.is_some_and(|f| f == surface);
        let mut out = Vec::new();
        // Only the active workspace of each output is visible — except
        // mid workspace-switch, where the outgoing (captured) and
        // incoming workspaces are both emitted, translated vertically.
        for op in &self.outputs {
            let area = op.area();
            #[allow(
                clippy::cast_possible_truncation,
                reason = "slide offset = small fraction × output height (i32), well within range"
            )]
            if let (Some(t), Some(spec)) = (op.transition.as_ref(), slide)
                && let Some(p) = transition_eased(t, spec)
            {
                let h = f64::from(op.full.size.h);
                let off_from = (f64::from(t.dir) * p * h).round() as i32;
                let off_to = (f64::from(-t.dir) * (1.0 - p) * h).round() as i32;
                for fp in &t.from {
                    out.push(Placement {
                        slide_dy: off_from,
                        focused: false, // the outgoing workspace isn't focused
                        ..fp.clone()
                    });
                }
                let base = out.len();
                collect_workspace(&op.workspaces[op.active], &is_focused, area, &mut out);
                for q in &mut out[base..] {
                    q.slide_dy = off_to;
                }
            } else {
                collect_workspace(&op.workspaces[op.active], &is_focused, area, &mut out);
            }
        }
        if let Some(t) = &self.in_transit {
            let surface = t.window.toplevel.wl_surface();
            out.push(Placement {
                surface: surface.clone(),
                cell_rect: t.window.rect,
                focused: is_focused(surface),
                fill: t.window.fill,
                // A window being dragged floats freely over everything.
                floating: true,
                slide_dy: 0,
            });
        }
        out
    }

    /// Clear workspace-switch transitions that have finished (or that
    /// can't run because the slide is disabled), freeing their captured
    /// snapshots. Call once per frame before [`Self::placements`].
    pub fn tick_transitions(&mut self, slide: Option<AnimSpec>) {
        for op in &mut self.outputs {
            let done = match (op.transition.as_ref(), slide) {
                (Some(t), Some(spec)) => transition_eased(t, spec).is_none(),
                (Some(_), None) => true,
                (None, _) => false,
            };
            if done {
                op.transition = None;
            }
        }
    }

    /// Hit-test the topmost window at `pos`, returning it together
    /// with its *effective* on-screen rect (the full output for a
    /// maximized/fullscreen window, otherwise its cell). The rect is
    /// what the caller uses as the surface origin for pointer events.
    ///
    /// A maximized/fullscreen window covers its whole output and draws
    /// on top, so it captures the pointer anywhere on that output
    /// (fullscreen beats maximized; later-drawn beats earlier). Below
    /// that, floating windows win over tiled, and within floating the
    /// top-of-stack (last-clicked / last-floated) wins. The in-transit
    /// window is intentionally skipped — it tracks the cursor by
    /// construction, so reporting it as "under the cursor" would just
    /// defeat focus changes for the duration of the drag.
    pub fn window_at(
        &self,
        pos: Point<i32, Physical>,
    ) -> Option<(&Window, Rectangle<i32, Physical>)> {
        // Hit-test only the active workspace of the output `pos` falls
        // in — windows on other workspaces aren't visible/clickable.
        let i = self.outpane_at(pos)?;
        let area = self.outputs[i].area();
        let ws = self.active_ws(i);

        // Filled windows first. Collect in draw order (tree leaves,
        // then floating) and pick the topmost fullscreen, else the
        // topmost maximized — `rfind` within a tier is the one drawn
        // on top. The effective rect must match what the renderer
        // draws (`area.fill`): the full output for fullscreen, the work
        // area for maximized — otherwise the surface origin handed to
        // the pointer is offset by any panel's exclusive zone.
        let mut filled: Vec<&Window> = Vec::new();
        if let Some(tree) = &ws.tree {
            collect_filled(tree, &mut filled);
        }
        filled.extend(ws.floating.iter().filter(|w| w.fill != FillMode::Normal));
        if let Some(w) = filled
            .iter()
            .rfind(|w| w.fill == FillMode::Fullscreen)
            .or_else(|| filled.iter().rfind(|w| w.fill == FillMode::Maximized))
        {
            return Some((w, area.fill(w.fill)));
        }

        for w in ws.floating.iter().rev() {
            if rect_contains(w.rect, pos) {
                return Some((w, w.rect));
            }
        }
        ws.tree
            .as_ref()
            .and_then(|t| leaf_at(t, pos))
            .map(|w| (w, w.rect))
    }

    fn recompute_and_push(&mut self) {
        let inner = self.gaps.inner;
        let outer = self.gaps.outer;
        let border = self.border_width;
        // Reflow every workspace (not just the active one) so a parked
        // workspace keeps correct saved sizes — switching to it is then
        // paint-only with no reflow flash.
        for op in &mut self.outputs {
            let tile_bounds = shrink_for_outer(op.bounds, outer);
            for ws in &mut op.workspaces {
                if let Some(tree) = &mut ws.tree {
                    assign_rects(tree, tile_bounds, inner);
                }
            }
        }
        for op in &self.outputs {
            let area = op.area();
            for ws in &op.workspaces {
                if let Some(tree) = &ws.tree {
                    push_configures_tree(tree, border, area);
                }
                for w in &ws.floating {
                    push_configure_for_floating(w, border, area);
                }
            }
        }
    }

    /// Update an output's full rect and usable work area, then reflow.
    /// Called when the geometry changes — e.g. a `wlr_layer_shell`
    /// panel reserves an exclusive zone, shrinking `work_area` below
    /// `full` (tiling + maximized avoid the panel; fullscreen still
    /// covers `full`). The output is keyed by connector name; an
    /// unknown name is a silent no-op so the renderer's and layout's
    /// output sets can drift without panicking.
    pub fn set_output_bounds(
        &mut self,
        name: &str,
        full: Rectangle<i32, Physical>,
        work_area: Rectangle<i32, Physical>,
    ) {
        let Some(op) = self.outputs.iter_mut().find(|o| o.name == name) else {
            return;
        };
        if op.full == full && op.bounds == work_area {
            return;
        }
        op.full = full;
        op.bounds = work_area;
        self.recompute_and_push();
    }

    /// Swap the gap + border-width settings and reflow every
    /// workspace (for live config reload). Tiles get re-laid-out with
    /// the new gaps and re-configured to the new inside-border size;
    /// no-op-cheap when the values are unchanged.
    pub fn set_appearance(&mut self, gaps: Gaps, border_width: i32) {
        self.gaps = gaps;
        self.border_width = border_width.max(0);
        self.recompute_and_push();
    }

    /// Current border width. A placement's surface buffer (0,0) is
    /// painted at `cell_rect.loc + border_width`, so a popup parent's
    /// window-geometry origin (which xdg popups are positioned
    /// relative to) is `cell_rect.loc + border_width`.
    pub fn border_width(&self) -> i32 {
        self.border_width
    }

    /// Snapshot every workspace across every output for the IPC
    /// `workspaces` query. One entry per workspace (including the
    /// trailing empty slot), in output-then-index order.
    pub fn workspace_entries(&self) -> Vec<WorkspaceEntry> {
        let mut out = Vec::new();
        for op in &self.outputs {
            for (index, ws) in op.workspaces.iter().enumerate() {
                out.push(WorkspaceEntry {
                    output: op.name.clone(),
                    index,
                    active: index == op.active,
                    windows: workspace_window_count(ws),
                });
            }
        }
        out
    }

    /// Snapshot every managed window across every output and workspace
    /// for the IPC `windows` query: its surface (so the caller can read
    /// title/app-id + assign a stable id), cell rect, fill mode, and
    /// whether it floats, plus which output/workspace holds it. The
    /// transient in-transit drag window is omitted (it has no settled
    /// home until release).
    pub fn window_entries(&self) -> Vec<WindowEntry> {
        let mut out = Vec::new();
        for op in &self.outputs {
            for (index, ws) in op.workspaces.iter().enumerate() {
                if let Some(tree) = &ws.tree {
                    collect_window_entries(tree, &op.name, index, &mut out);
                }
                for w in &ws.floating {
                    out.push(WindowEntry {
                        surface: w.toplevel.wl_surface().clone(),
                        rect: w.rect,
                        fill: w.fill,
                        floating: true,
                        output: op.name.clone(),
                        workspace: index,
                    });
                }
            }
        }
        out
    }

    /// Active workspace index of the named output, or `None` if no such
    /// output. Used to annotate the IPC `outputs` query.
    pub fn active_workspace(&self, output: &str) -> Option<usize> {
        self.outputs
            .iter()
            .find(|op| op.name == output)
            .map(|op| op.active)
    }

    /// Set a window's fill mode (normal / maximized / fullscreen) and
    /// reflow so it picks up its new size, border state, and z-order.
    /// The state lives on the `Window`, so it survives moves between
    /// workspaces. Returns whether `surface` was found. Always reflows
    /// (even for a redundant request) so the client gets the
    /// configure xdg-shell expects in response.
    pub fn set_fill(&mut self, surface: &WlSurface, fill: FillMode) -> bool {
        let Some(w) = self.window_mut(surface) else {
            return false;
        };
        w.fill = fill;
        self.recompute_and_push();
        true
    }

    /// Flip `surface` between fullscreen and normal (the Super+F11
    /// gesture). A maximized window becomes fullscreen; anything else
    /// toggles against fullscreen. Returns whether `surface` was
    /// found.
    pub fn toggle_fullscreen(&mut self, surface: &WlSurface) -> bool {
        let Some(w) = self.window_mut(surface) else {
            return false;
        };
        w.fill = if w.fill == FillMode::Fullscreen {
            FillMode::Normal
        } else {
            FillMode::Fullscreen
        };
        self.recompute_and_push();
        true
    }

    /// Whether `surface` is a tracked window that's maximized or
    /// fullscreen (used to refuse interactive move/resize on it).
    fn is_filled(&self, surface: &WlSurface) -> bool {
        self.window_ref(surface)
            .is_some_and(|w| w.fill != FillMode::Normal)
    }

    /// Find a window by surface anywhere it can live (any output's any
    /// workspace tree or floating stack, or the in-transit drag).
    fn window_ref(&self, surface: &WlSurface) -> Option<&Window> {
        if let Some(t) = &self.in_transit
            && t.window.toplevel.wl_surface() == surface
        {
            return Some(&t.window);
        }
        for op in &self.outputs {
            for ws in &op.workspaces {
                if let Some(w) = ws
                    .floating
                    .iter()
                    .find(|w| w.toplevel.wl_surface() == surface)
                {
                    return Some(w);
                }
                if let Some(t) = &ws.tree
                    && let Some(w) = leaf_ref(t, surface)
                {
                    return Some(w);
                }
            }
        }
        None
    }

    /// Mutable [`Self::window_ref`]. In-transit is checked first so the
    /// loops are the function tail — the borrow checker rejects
    /// reborrowing `self` after a loop that conditionally returns a
    /// `&mut` from inside it.
    fn window_mut(&mut self, surface: &WlSurface) -> Option<&mut Window> {
        if self
            .in_transit
            .as_ref()
            .is_some_and(|t| t.window.toplevel.wl_surface() == surface)
        {
            return Some(&mut self.in_transit.as_mut().expect("checked Some above").window);
        }
        for op in &mut self.outputs {
            for ws in &mut op.workspaces {
                if let Some(w) = ws
                    .floating
                    .iter_mut()
                    .find(|w| w.toplevel.wl_surface() == surface)
                {
                    return Some(w);
                }
                if let Some(t) = ws.tree.as_mut()
                    && let Some(w) = leaf_mut(t, surface)
                {
                    return Some(w);
                }
            }
        }
        None
    }

    /// Switch the active workspace on the output under `cursor` by
    /// `delta` (`+1` = next / scroll-down, `-1` = previous /
    /// scroll-up). No-op if the cursor is over no output. No wrap:
    /// scrolling up past the first workspace stays put. Returns
    /// whether the active workspace actually changed (so the caller
    /// can re-derive keyboard focus only when it did).
    pub fn switch_at(&mut self, cursor: Point<i32, Physical>, delta: i32) -> bool {
        self.outpane_at(cursor)
            .is_some_and(|oi| self.switch(oi, delta))
    }

    /// Switch output `oi`'s active workspace by `delta`. Materializes
    /// a fresh trailing-empty workspace to scroll into when moving
    /// past the end; `normalize_output` then compacts the workspace
    /// we left if it became empty and trims back to one trailing
    /// empty, so the list can't grow without bound. Returns whether
    /// the active workspace changed.
    fn switch(&mut self, oi: usize, delta: i32) -> bool {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "workspace index is a small Vec index, never near i32 bounds"
        )]
        let target = self.outputs[oi].active as i32 + delta;
        if target < 0 {
            return false; // scroll-up at workspace 0: no wrap, no-op.
        }
        #[allow(clippy::cast_sign_loss, reason = "target >= 0 checked just above")]
        let target = target as usize;
        while target >= self.outputs[oi].workspaces.len() {
            self.outputs[oi].workspaces.push(Workspace::default());
        }
        if target == self.outputs[oi].active {
            return false;
        }
        // Snapshot the outgoing workspace for the slide animation before
        // `active` moves (and before `normalize_output` may reindex the
        // workspace list, which would invalidate a stored index).
        let area = self.outputs[oi].area();
        let mut from = Vec::new();
        collect_workspace(
            &self.outputs[oi].workspaces[self.outputs[oi].active],
            &|_| false,
            area,
            &mut from,
        );
        self.outputs[oi].transition = Some(WsTransition {
            from,
            // Next workspace slides up (incoming from the bottom),
            // previous slides down — the natural scroll mapping.
            dir: -delta.signum(),
            start: Instant::now(),
        });
        self.outputs[oi].active = target;
        self.normalize_output(oi);
        self.recompute_and_push();
        true
    }

    /// Move the keyboard-focused window to the adjacent workspace on
    /// **its own** output and follow it there (the destination
    /// becomes active). Handles both tiled and floating focused
    /// windows. No wrap: `Shift`+scroll-up while on workspace 0 is a
    /// no-op. Returns `true` if a window actually moved; `false` if
    /// `surface` isn't on any visible workspace or the move was a
    /// no-op (at the top edge).
    pub fn move_focused_window(&mut self, surface: &WlSurface, delta: i32) -> bool {
        // Locate the window on a visible (active) workspace.
        let mut found: Option<(usize, bool)> = None;
        for (oi, op) in self.outputs.iter().enumerate() {
            let ws = &op.workspaces[op.active];
            if ws
                .floating
                .iter()
                .any(|w| w.toplevel.wl_surface() == surface)
            {
                found = Some((oi, true));
                break;
            }
            if ws.tree.as_ref().is_some_and(|t| tree_contains(t, surface)) {
                found = Some((oi, false));
                break;
            }
        }
        let Some((oi, is_floating)) = found else {
            return false;
        };

        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "workspace index is a small Vec index, never near i32 bounds"
        )]
        let dst = self.outputs[oi].active as i32 + delta;
        if dst < 0 {
            return false; // Shift+scroll-up at workspace 0: no-op.
        }
        #[allow(clippy::cast_sign_loss, reason = "dst >= 0 checked just above")]
        let dst = dst as usize;
        while dst >= self.outputs[oi].workspaces.len() {
            self.outputs[oi].workspaces.push(Workspace::default());
        }
        if dst == self.outputs[oi].active {
            return false;
        }

        // Extract from the source (active) workspace. The unwraps
        // can't fire: the search loop above just confirmed `surface`
        // lives on output `oi`'s active workspace (as float or tile),
        // and nothing mutates the layout between then and here.
        let window = if is_floating {
            let ws = self.active_ws_mut(oi);
            let pos = ws
                .floating
                .iter()
                .position(|w| w.toplevel.wl_surface() == surface)
                .expect("surface was just found in this floating list");
            ws.floating.remove(pos)
        } else {
            let ws = self.active_ws_mut(oi);
            let tree = ws.tree.take().expect("surface was just found in this tree");
            let (root_after, removed) = remove_from_tree(tree, surface);
            ws.tree = root_after;
            removed.expect("surface was just found in this tree")
        };

        // Insert into the destination workspace, preserving kind.
        if is_floating {
            // Keeps its absolute rect — both workspaces share the
            // output's bounds, so it stays visually put on the new
            // scene.
            self.outputs[oi].workspaces[dst].floating.push(window);
        } else {
            let tile_bounds = self.tile_bounds(oi);
            let inner = self.gaps.inner;
            let leaf = Node::Leaf(window);
            let dws = &mut self.outputs[oi].workspaces[dst];
            dws.tree = Some(match dws.tree.take() {
                None => leaf,
                Some(root) => insert_at_cursor(root, leaf, tile_bounds, None, inner),
            });
        }

        // Follow the window: make the destination active, then
        // normalize (compacts the now-possibly-empty source).
        self.outputs[oi].active = dst;
        self.normalize_output(oi);
        self.recompute_and_push();
        true
    }

    /// Re-establish output `oi`'s workspace invariants: drop empty
    /// workspaces that are neither the active one nor the trailing
    /// slot, keep `active` pointing at the same logical workspace
    /// across the renumbering, and guarantee exactly one trailing
    /// empty workspace (`len >= 1`). Idempotent.
    fn normalize_output(&mut self, oi: usize) {
        let o = &mut self.outputs[oi];
        let old_active = o.active;

        // Pass 1: keep the active workspace (always) and every
        // non-empty workspace; drop empty non-active ones. Record
        // where the active workspace lands in the compacted list.
        let mut kept: Vec<Workspace> = Vec::with_capacity(o.workspaces.len());
        let mut new_active = 0;
        for (i, ws) in std::mem::take(&mut o.workspaces).into_iter().enumerate() {
            if i == old_active {
                new_active = kept.len();
                kept.push(ws);
            } else if !ws.is_empty() {
                kept.push(ws);
            }
        }
        o.workspaces = kept;
        o.active = new_active;

        // Pass 2: trim extra trailing empties beyond the active one,
        // then ensure exactly one trailing empty exists (it may
        // coincide with the active workspace when active is empty).
        while o.workspaces.len() > o.active + 1
            && o.workspaces.last().is_some_and(Workspace::is_empty)
        {
            o.workspaces.pop();
        }
        if !o.workspaces.last().is_some_and(Workspace::is_empty) {
            o.workspaces.push(Workspace::default());
        }
        debug_assert!(o.active < o.workspaces.len());
        debug_assert!(o.workspaces.last().is_some_and(Workspace::is_empty));
    }
}

// ---- Tree internals ---------------------------------------------

/// Replace the leaf containing `cursor` (or, if `cursor` is outside
/// the layout or `None`, the deepest leaf reachable by always
/// picking `second`) with a fresh split: the existing leaf as one
/// side and `new_leaf` as the other. The split axis is picked
/// perpendicular to the target leaf's longer side; the cursor's
/// half of the leaf decides which side gets the new window.
/// `inner` is the inter-cell gap passed down so each split's
/// child rect calculation matches what `assign_rects` will
/// reproduce — keeps cursor-vs-cell hit-testing accurate.
fn insert_at_cursor(
    node: Node,
    new_leaf: Node,
    bounds: Rectangle<i32, Physical>,
    cursor: Option<Point<i32, Physical>>,
    inner: i32,
) -> Node {
    match node {
        Node::Leaf(existing) => {
            let leaf_rect = bounds;
            let (axis, new_first) = pick_split(leaf_rect, cursor);
            let existing_leaf = Node::Leaf(existing);
            let (first, second) = if new_first {
                (Box::new(new_leaf), Box::new(existing_leaf))
            } else {
                (Box::new(existing_leaf), Box::new(new_leaf))
            };
            Node::Split {
                axis,
                ratio: 0.5,
                first,
                second,
            }
        }
        Node::Split {
            axis,
            ratio,
            first,
            second,
        } => {
            let (b1, b2) = split_bounds(bounds, axis, ratio, inner);
            // Pick the child whose rect contains the cursor. If
            // the cursor is missing or outside this split, fall
            // through to `second` (the "deepest" branch in our
            // recursion convention) so the new window still
            // lands somewhere sensible.
            let go_first = match cursor {
                Some(c) if rect_contains(b1, c) => true,
                Some(c) if rect_contains(b2, c) => false,
                _ => false,
            };
            if go_first {
                Node::Split {
                    axis,
                    ratio,
                    first: Box::new(insert_at_cursor(*first, new_leaf, b1, cursor, inner)),
                    second,
                }
            } else {
                Node::Split {
                    axis,
                    ratio,
                    first,
                    second: Box::new(insert_at_cursor(*second, new_leaf, b2, cursor, inner)),
                }
            }
        }
    }
}

/// Choose how to split `leaf_rect` for a new window when the user
/// drops/spawns at `cursor`. Split direction is perpendicular to
/// the leaf's longer side (so cells stay roughly square); which
/// side of the leaf's centre the cursor sits on decides whether
/// the new window goes first (= left/top) or second (= right/
/// bottom). A missing or out-of-leaf cursor falls back to "new on
/// the right/bottom".
fn pick_split(
    leaf_rect: Rectangle<i32, Physical>,
    cursor: Option<Point<i32, Physical>>,
) -> (SplitAxis, bool) {
    let axis = if leaf_rect.size.w >= leaf_rect.size.h {
        SplitAxis::LeftRight
    } else {
        SplitAxis::TopBottom
    };
    let new_first = match (axis, cursor) {
        (SplitAxis::LeftRight, Some(c)) => {
            let center_x = leaf_rect.loc.x + leaf_rect.size.w / 2;
            c.x < center_x
        }
        (SplitAxis::TopBottom, Some(c)) => {
            let center_y = leaf_rect.loc.y + leaf_rect.size.h / 2;
            c.y < center_y
        }
        (_, None) => false,
    };
    (axis, new_first)
}

/// Walk the tree to find and remove the leaf whose toplevel
/// matches `surface`. Returns the (possibly collapsed) tree and
/// the removed window if found.
fn remove_from_tree(node: Node, surface: &WlSurface) -> (Option<Node>, Option<Window>) {
    match node {
        Node::Leaf(w) => {
            if w.toplevel.wl_surface() == surface {
                (None, Some(w))
            } else {
                (Some(Node::Leaf(w)), None)
            }
        }
        Node::Split {
            axis,
            ratio,
            first,
            second,
        } => {
            let (first_after, removed) = remove_from_tree(*first, surface);
            if let Some(window) = removed {
                let node_after = match first_after {
                    Some(n) => Some(Node::Split {
                        axis,
                        ratio,
                        first: Box::new(n),
                        second,
                    }),
                    None => Some(*second),
                };
                return (node_after, Some(window));
            }
            let first_kept = first_after.expect("unchanged subtree must come back as Some");
            let (second_after, removed) = remove_from_tree(*second, surface);
            if let Some(window) = removed {
                let node_after = match second_after {
                    Some(n) => Some(Node::Split {
                        axis,
                        ratio,
                        first: Box::new(first_kept),
                        second: Box::new(n),
                    }),
                    None => Some(first_kept),
                };
                (node_after, Some(window))
            } else {
                let second_kept = second_after.expect("unchanged subtree must come back as Some");
                (
                    Some(Node::Split {
                        axis,
                        ratio,
                        first: Box::new(first_kept),
                        second: Box::new(second_kept),
                    }),
                    None,
                )
            }
        }
    }
}

/// Reassign every leaf's rect by walking the tree top-down. Each
/// `Split` shrinks its children by `inner` along the split axis
/// (centred on the divider) so adjacent cells get visible space
/// between them.
fn assign_rects(node: &mut Node, bounds: Rectangle<i32, Physical>, inner: i32) {
    match node {
        Node::Leaf(w) => w.rect = bounds,
        Node::Split {
            axis,
            ratio,
            first,
            second,
        } => {
            let (b1, b2) = split_bounds(bounds, *axis, *ratio, inner);
            assign_rects(first, b1, inner);
            assign_rects(second, b2, inner);
        }
    }
}

/// Walk the tree and emit a `Placement` for every leaf. A maximized
/// or fullscreen leaf reports `output_bounds` as its cell so it covers
/// the whole output instead of its tiled slot.
fn collect_placements(
    node: &Node,
    is_focused: &impl Fn(&WlSurface) -> bool,
    area: OutputArea,
    out: &mut Vec<Placement>,
) {
    match node {
        Node::Leaf(w) => {
            let surface = w.toplevel.wl_surface();
            out.push(Placement {
                surface: surface.clone(),
                cell_rect: if w.fill == FillMode::Normal {
                    w.rect
                } else {
                    area.fill(w.fill)
                },
                focused: is_focused(surface),
                fill: w.fill,
                floating: false,
                slide_dy: 0,
            });
        }
        Node::Split { first, second, .. } => {
            collect_placements(first, is_focused, area, out);
            collect_placements(second, is_focused, area, out);
        }
    }
}

/// Build one workspace's placements: the tiled tree, then floating
/// windows bottom-up (drawn above the tiles they overlap).
fn collect_workspace(
    ws: &Workspace,
    is_focused: &impl Fn(&WlSurface) -> bool,
    area: OutputArea,
    out: &mut Vec<Placement>,
) {
    if let Some(tree) = &ws.tree {
        collect_placements(tree, is_focused, area, out);
    }
    for w in &ws.floating {
        let surface = w.toplevel.wl_surface();
        out.push(Placement {
            surface: surface.clone(),
            // A maximized float covers the work area, a fullscreen one
            // the whole output; both ignore the floating rect.
            cell_rect: if w.fill == FillMode::Normal {
                w.rect
            } else {
                area.fill(w.fill)
            },
            focused: is_focused(surface),
            fill: w.fill,
            floating: true,
            slide_dy: 0,
        });
    }
}

/// Total windows in a workspace: tiled leaves plus floats.
fn workspace_window_count(ws: &Workspace) -> usize {
    fn leaves(node: &Node) -> usize {
        match node {
            Node::Leaf(_) => 1,
            Node::Split { first, second, .. } => leaves(first) + leaves(second),
        }
    }
    ws.tree.as_ref().map_or(0, leaves) + ws.floating.len()
}

/// Push a [`WindowEntry`] for every tiled leaf in `node` (recursively).
fn collect_window_entries(
    node: &Node,
    output: &str,
    workspace: usize,
    out: &mut Vec<WindowEntry>,
) {
    match node {
        Node::Leaf(w) => out.push(WindowEntry {
            surface: w.toplevel.wl_surface().clone(),
            rect: w.rect,
            fill: w.fill,
            floating: false,
            output: output.to_owned(),
            workspace,
        }),
        Node::Split { first, second, .. } => {
            collect_window_entries(first, output, workspace, out);
            collect_window_entries(second, output, workspace, out);
        }
    }
}

/// Eased progress `[0, 1)` of a workspace slide, or `None` once it has
/// run its course (so the caller emits only the active workspace).
fn transition_eased(t: &WsTransition, spec: AnimSpec) -> Option<f64> {
    let dur = spec.duration_secs();
    let elapsed = t.start.elapsed().as_secs_f64();
    if dur <= 0.0 || elapsed >= dur {
        return None;
    }
    Some(spec.curve.eval(elapsed / dur))
}

/// Walk the tree, find the leaf containing `pos`, return it.
fn leaf_at(node: &Node, pos: Point<i32, Physical>) -> Option<&Window> {
    match node {
        Node::Leaf(w) => {
            if rect_contains(w.rect, pos) {
                Some(w)
            } else {
                None
            }
        }
        Node::Split { first, second, .. } => leaf_at(first, pos).or_else(|| leaf_at(second, pos)),
    }
}

/// Push every maximized/fullscreen leaf onto `out` in tree (draw)
/// order — used by `window_at` to find the topmost filled window.
fn collect_filled<'a>(node: &'a Node, out: &mut Vec<&'a Window>) {
    match node {
        Node::Leaf(w) => {
            if w.fill != FillMode::Normal {
                out.push(w);
            }
        }
        Node::Split { first, second, .. } => {
            collect_filled(first, out);
            collect_filled(second, out);
        }
    }
}

/// Find the leaf whose window is `surface` (shared borrow).
fn leaf_ref<'a>(node: &'a Node, surface: &WlSurface) -> Option<&'a Window> {
    match node {
        Node::Leaf(w) => (w.toplevel.wl_surface() == surface).then_some(w),
        Node::Split { first, second, .. } => {
            leaf_ref(first, surface).or_else(|| leaf_ref(second, surface))
        }
    }
}

/// Find the leaf whose window is `surface` (mutable borrow). `first`
/// and `second` are disjoint fields, so the early-return-then-reborrow
/// is accepted by the borrow checker.
fn leaf_mut<'a>(node: &'a mut Node, surface: &WlSurface) -> Option<&'a mut Window> {
    match node {
        Node::Leaf(w) => (w.toplevel.wl_surface() == surface).then_some(w),
        Node::Split { first, second, .. } => {
            if let Some(w) = leaf_mut(first, surface) {
                return Some(w);
            }
            leaf_mut(second, surface)
        }
    }
}

/// True if any leaf in the tree is `surface`. Used to find which
/// workspace's tree holds the focused window for the move gesture.
fn tree_contains(node: &Node, surface: &WlSurface) -> bool {
    match node {
        Node::Leaf(w) => w.toplevel.wl_surface() == surface,
        Node::Split { first, second, .. } => {
            tree_contains(first, surface) || tree_contains(second, surface)
        }
    }
}

/// Ship `xdg_toplevel.configure` for every leaf in the tree.
/// Tiles are configured with `Activated + Tiled{Left,Right,Top,
/// Bottom}` so that clients (notably kitty) treat the cell as a
/// hard size to fill, without leaving margins for their own
/// resize handles or rounding to a font grid.
fn push_configures_tree(node: &Node, border: i32, area: OutputArea) {
    match node {
        Node::Leaf(w) => push_configure_for_tile(w, border, area),
        Node::Split { first, second, .. } => {
            push_configures_tree(first, border, area);
            push_configures_tree(second, border, area);
        }
    }
}

/// Configure a maximized/fullscreen window to fill `rect` (the work
/// area for maximized, the full output for fullscreen — already
/// resolved by the caller) with no border inset and no `Tiled*` flags
/// (the client owns every edge), and set the matching `xdg_toplevel`
/// state so the client drops its own decorations/shadow and sizes to
/// the target. Shared by the tiled and floating paths — fill mode
/// dominates either home.
fn push_configure_filled(w: &Window, rect: Rectangle<i32, Physical>) {
    let size = Size::<i32, Logical>::from((rect.size.w.max(1), rect.size.h.max(1)));
    w.toplevel.with_pending_state(|state| {
        state.size = Some(size);
        state.states.set(xdg_toplevel::State::Activated);
        state.states.unset(xdg_toplevel::State::TiledLeft);
        state.states.unset(xdg_toplevel::State::TiledRight);
        state.states.unset(xdg_toplevel::State::TiledTop);
        state.states.unset(xdg_toplevel::State::TiledBottom);
        match w.fill {
            FillMode::Maximized => {
                state.states.set(xdg_toplevel::State::Maximized);
                state.states.unset(xdg_toplevel::State::Fullscreen);
            }
            FillMode::Fullscreen => {
                state.states.set(xdg_toplevel::State::Fullscreen);
                state.states.unset(xdg_toplevel::State::Maximized);
            }
            // Caller only reaches here for non-Normal fills.
            FillMode::Normal => {}
        }
    });
    w.toplevel.send_configure();
    debug!(
        surface = ?w.toplevel.wl_surface().id(),
        w = rect.size.w,
        h = rect.size.h,
        fill = ?w.fill,
        "layout: fullscreen/maximized configure sent",
    );
}

/// Shrink `cell_size` by `2 * border` on each axis (clamped to a
/// minimum of `1` so we never ship a zero-size configure, which
/// the client can't render) and return the result as a
/// `Logical`-typed `Size` ready for `state.size`.
fn surface_size(cell_size: Size<i32, Physical>, border: i32) -> Size<i32, Logical> {
    let border = border.max(0);
    Size::<i32, Logical>::from((
        (cell_size.w - 2 * border).max(1),
        (cell_size.h - 2 * border).max(1),
    ))
}

/// Configure a tiled window: send the inside-border size, and
/// set the activated + tiled-on-all-sides state set so the
/// client fills the cell exactly. Each `TiledX` flag tells the
/// client "the X edge is shared with the compositor / another
/// window, so don't draw a resize handle or border on that side".
/// A tiling WM cell is tiled on every side.
fn push_configure_for_tile(w: &Window, border: i32, area: OutputArea) {
    if w.fill != FillMode::Normal {
        push_configure_filled(w, area.fill(w.fill));
        return;
    }
    let size = surface_size(w.rect.size, border);
    w.toplevel.with_pending_state(|state| {
        state.size = Some(size);
        state.states.set(xdg_toplevel::State::Activated);
        state.states.set(xdg_toplevel::State::TiledLeft);
        state.states.set(xdg_toplevel::State::TiledRight);
        state.states.set(xdg_toplevel::State::TiledTop);
        state.states.set(xdg_toplevel::State::TiledBottom);
        // Clear any prior fill so unmaximize/unfullscreen → tile works.
        state.states.unset(xdg_toplevel::State::Maximized);
        state.states.unset(xdg_toplevel::State::Fullscreen);
    });
    w.toplevel.send_configure();
    debug!(
        surface = ?w.toplevel.wl_surface().id(),
        x = w.rect.loc.x,
        y = w.rect.loc.y,
        w = w.rect.size.w,
        h = w.rect.size.h,
        border,
        "layout: tile configure sent",
    );
}

/// Configure a floating (or in-transit) window: send the inside-
/// border size, clear the `Tiled*` flags so the client knows it
/// can resize freely, but still set `Activated` so the focused
/// float doesn't dim or hide its content.
fn push_configure_for_floating(w: &Window, border: i32, area: OutputArea) {
    if w.fill != FillMode::Normal {
        push_configure_filled(w, area.fill(w.fill));
        return;
    }
    let size = surface_size(w.rect.size, border);
    w.toplevel.with_pending_state(|state| {
        state.size = Some(size);
        state.states.set(xdg_toplevel::State::Activated);
        state.states.unset(xdg_toplevel::State::TiledLeft);
        state.states.unset(xdg_toplevel::State::TiledRight);
        state.states.unset(xdg_toplevel::State::TiledTop);
        state.states.unset(xdg_toplevel::State::TiledBottom);
        // Clear any prior fill so unmaximize/unfullscreen → float works.
        state.states.unset(xdg_toplevel::State::Maximized);
        state.states.unset(xdg_toplevel::State::Fullscreen);
    });
    w.toplevel.send_configure();
    debug!(
        surface = ?w.toplevel.wl_surface().id(),
        x = w.rect.loc.x,
        y = w.rect.loc.y,
        w = w.rect.size.w,
        h = w.rect.size.h,
        border,
        "layout: floating configure sent",
    );
}

/// Split `bounds` into `(first, second)` along `axis` at `ratio`,
/// leaving `inner` pixels of empty space straddling the divider
/// (`inner / 2` taken from each child on the dividing side; for
/// odd values the extra pixel goes to the second child's side
/// so the sum still equals `bounds`). Clamps each child's
/// dividing dimension to at least 1 px so neither collapses to
/// zero — clients can't render a zero-sized surface and would
/// just hang.
fn split_bounds(
    bounds: Rectangle<i32, Physical>,
    axis: SplitAxis,
    ratio: f32,
    inner: i32,
) -> (Rectangle<i32, Physical>, Rectangle<i32, Physical>) {
    let inner = inner.max(0);
    let half_a = inner / 2;
    let half_b = inner - half_a;
    match axis {
        SplitAxis::LeftRight => {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_precision_loss,
                clippy::cast_sign_loss,
                reason = "bounds.size.w is bounded by layout_bounds (i32); ratio is 0..1; product fits in i32 with room to spare"
            )]
            let split = ((bounds.size.w as f32) * ratio.clamp(0.0, 1.0)) as i32;
            // `.max(1)` on the upper bound keeps it >= the lower bound
            // (1), so `clamp` never sees min > max — which would panic
            // for a 0/1-px-wide cell.
            let split = split.clamp(1, (bounds.size.w - 1).max(1));
            let a_w = (split - half_a).max(1);
            let b_w = (bounds.size.w - split - half_b).max(1);
            let a = Rectangle::new(bounds.loc, Size::new(a_w, bounds.size.h));
            let b = Rectangle::new(
                Point::new(bounds.loc.x + split + half_b, bounds.loc.y),
                Size::new(b_w, bounds.size.h),
            );
            (a, b)
        }
        SplitAxis::TopBottom => {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_precision_loss,
                clippy::cast_sign_loss,
                reason = "bounds.size.h is bounded by layout_bounds (i32); ratio is 0..1; product fits in i32 with room to spare"
            )]
            let split = ((bounds.size.h as f32) * ratio.clamp(0.0, 1.0)) as i32;
            // See the LeftRight arm: `.max(1)` keeps min <= max so
            // `clamp` can't panic on a 0/1-px-tall cell.
            let split = split.clamp(1, (bounds.size.h - 1).max(1));
            let a_h = (split - half_a).max(1);
            let b_h = (bounds.size.h - split - half_b).max(1);
            let a = Rectangle::new(bounds.loc, Size::new(bounds.size.w, a_h));
            let b = Rectangle::new(
                Point::new(bounds.loc.x, bounds.loc.y + split + half_b),
                Size::new(bounds.size.w, b_h),
            );
            (a, b)
        }
    }
}

/// Shrink `bounds` by `outer` pixels on every side. Used to
/// reserve the outer-gap area around the tile region.
fn shrink_for_outer(bounds: Rectangle<i32, Physical>, outer: i32) -> Rectangle<i32, Physical> {
    let outer = outer.max(0);
    let new_w = (bounds.size.w - 2 * outer).max(1);
    let new_h = (bounds.size.h - 2 * outer).max(1);
    Rectangle::new(
        Point::new(bounds.loc.x + outer, bounds.loc.y + outer),
        Size::new(new_w, new_h),
    )
}

fn rect_contains(r: Rectangle<i32, Physical>, p: Point<i32, Physical>) -> bool {
    r.size.w > 0
        && r.size.h > 0
        && p.x >= r.loc.x
        && p.x < r.loc.x + r.size.w
        && p.y >= r.loc.y
        && p.y < r.loc.y + r.size.h
}
