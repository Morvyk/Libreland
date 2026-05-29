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
//! Floating windows live in a separate `Vec` and always draw on
//! top of the tree. Toggling a window between tiled and floating
//! (`Super+F`) removes it from one set and inserts into the
//! other; the tree-side promote applies a 70 % centre shrink so
//! the transition reads as a smooth resize.
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

use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
use smithay::reexports::wayland_server::Resource as _;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Physical, Point, Rectangle, Size};
use smithay::wayland::shell::xdg::ToplevelSurface;
use tracing::debug;

/// One window managed by the layout, plus its current placement.
/// The `rect` is the cell the layout has assigned (refreshed by
/// every reflow) — clients see the same size via
/// `xdg_toplevel.configure`.
#[derive(Debug, Clone)]
pub struct Window {
    pub toplevel: ToplevelSurface,
    pub rect: Rectangle<i32, Physical>,
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
}

/// One output's tiling region: a stable connector name, that
/// output's full rect in absolute compositor space, and the dwindle
/// tree of windows tiled on it. A window's owning output is implicit
/// in which `Outpane`'s tree holds it — no per-window output id is
/// stored. The renderer paints each window on whichever CRTC its
/// absolute rect falls on, so tiling a window inside output B's
/// `bounds` is what makes it appear on monitor B.
struct Outpane {
    name: String,
    bounds: Rectangle<i32, Physical>,
    tree: Option<Node>,
}

/// One dwindle tree **per output** plus a global floating-window
/// list. Tiled windows live in the `Outpane` whose tree holds them;
/// floating windows and the in-transit drag are global and
/// absolute-positioned, so they're painted on whichever output
/// contains their rect. All coordinates are absolute compositor
/// pixels. Per-output workspaces are a future milestone — `Outpane`
/// is shaped so a later `Vec<Workspace>` wrap is mechanical.
pub struct Layout {
    outputs: Vec<Outpane>,
    floating: Vec<Window>,
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
                .map(|(name, bounds)| Outpane {
                    name,
                    bounds,
                    tree: None,
                })
                .collect(),
            floating: Vec::new(),
            in_transit: None,
            gaps,
            border_width: border_width.max(0),
        }
    }

    /// Tile area of the output at `idx`: its bounds shrunk by the
    /// outer gap.
    fn tile_bounds(&self, idx: usize) -> Rectangle<i32, Physical> {
        shrink_for_outer(self.outputs[idx].bounds, self.gaps.outer)
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
        };
        let leaf = Node::Leaf(window);
        self.outputs[idx].tree = Some(match self.outputs[idx].tree.take() {
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
        for op in &mut self.outputs {
            if let Some(root) = op.tree.take() {
                let (root_after, removed) = remove_from_tree(root, surface);
                op.tree = root_after;
                if removed.is_some() {
                    self.recompute_and_push();
                    return;
                }
            }
        }
        let len = self.floating.len();
        self.floating.retain(|w| w.toplevel.wl_surface() != surface);
        if self.floating.len() != len {
            return;
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
        // Tile -> float: find the window in whichever output's tree
        // holds it. Non-matching outputs get their tree restored
        // unchanged (remove_from_tree hands the node back on a miss).
        for op in &mut self.outputs {
            if let Some(root) = op.tree.take() {
                let (root_after, removed) = remove_from_tree(root, surface);
                op.tree = root_after;
                if let Some(mut window) = removed {
                    let prev = window.rect;
                    let new_size =
                        Size::<i32, Physical>::new((prev.size.w * 7) / 10, (prev.size.h * 7) / 10);
                    let new_loc = Point::<i32, Physical>::new(
                        prev.loc.x + (prev.size.w - new_size.w) / 2,
                        prev.loc.y + (prev.size.h - new_size.h) / 2,
                    );
                    window.rect = Rectangle::new(new_loc, new_size);
                    push_configure_for_floating(&window, self.border_width);
                    self.floating.push(window);
                    self.recompute_and_push();
                    return;
                }
            }
        }
        // Float -> tile: re-tile on the output under the float's
        // centre (a float straddling two outputs resolves by centre),
        // else the first output.
        let Some(fidx) = self
            .floating
            .iter()
            .position(|w| w.toplevel.wl_surface() == surface)
        else {
            return;
        };
        let window = self.floating.remove(fidx);
        let center = Point::<i32, Physical>::new(
            window.rect.loc.x + window.rect.size.w / 2,
            window.rect.loc.y + window.rect.size.h / 2,
        );
        let Some(idx) = self.outpane_for_point(Some(center)) else {
            // No outputs: keep it floating so we don't drop it.
            self.floating.push(window);
            return;
        };
        let tile_bounds = self.tile_bounds(idx);
        let inner = self.gaps.inner;
        let leaf = Node::Leaf(window);
        self.outputs[idx].tree = Some(match self.outputs[idx].tree.take() {
            None => leaf,
            Some(root) => insert_at_cursor(root, leaf, tile_bounds, Some(center), inner),
        });
        self.recompute_and_push();
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
        for op in &mut self.outputs {
            if let Some(root) = op.tree.take() {
                let (root_after, removed) = remove_from_tree(root, surface);
                op.tree = root_after;
                if let Some(window) = removed {
                    let rect = window.rect;
                    self.in_transit = Some(InTransit {
                        window,
                        source: DragSource::Tiled,
                    });
                    self.recompute_and_push();
                    return Some(rect);
                }
            }
        }
        if let Some(idx) = self
            .floating
            .iter()
            .position(|w| w.toplevel.wl_surface() == surface)
        {
            let window = self.floating.remove(idx);
            let rect = window.rect;
            self.in_transit = Some(InTransit {
                window,
                source: DragSource::Floating,
            });
            return Some(rect);
        }
        None
    }

    /// Start an interactive *resize* drag. Only floating windows
    /// can be drag-resized today; resize on a tile is rejected so
    /// the caller can log + swallow the press. Returns the rect
    /// to use as the drag's start rect, or `None`.
    pub fn start_resize_drag(&self, surface: &WlSurface) -> Option<Rectangle<i32, Physical>> {
        self.floating
            .iter()
            .find(|w| w.toplevel.wl_surface() == surface)
            .map(|w| w.rect)
    }

    /// Update the `in_transit` window's rect during a move drag
    /// and ship the corresponding configure. Silent no-op when
    /// nothing is in transit.
    pub fn update_in_transit_rect(&mut self, rect: Rectangle<i32, Physical>) {
        if let Some(t) = &mut self.in_transit {
            t.window.rect = rect;
            // An in-transit window is conceptually floating until
            // it either drops onto a tile cell or rejoins the
            // float stack, so configure it as such (no Tiled*
            // states, free-form resize).
            push_configure_for_floating(&t.window, self.border_width);
        }
    }

    /// Update a floating window's rect during a resize drag and
    /// ship the corresponding configure. Silent no-op for surfaces
    /// that aren't currently floating.
    pub fn set_floating_rect(&mut self, surface: &WlSurface, rect: Rectangle<i32, Physical>) {
        let border = self.border_width;
        let Some(window) = self
            .floating
            .iter_mut()
            .find(|w| w.toplevel.wl_surface() == surface)
        else {
            return;
        };
        window.rect = rect;
        push_configure_for_floating(window, border);
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
        match t.source {
            DragSource::Tiled => {
                // Re-tile on the output under the drop cursor. If the
                // cursor landed in a gap between monitors, fall back
                // to the output under the window's centre, then to
                // the first output. This is the cross-output move:
                // dropping on monitor B re-tiles into B's tree.
                let center = Point::<i32, Physical>::new(
                    t.window.rect.loc.x + t.window.rect.size.w / 2,
                    t.window.rect.loc.y + t.window.rect.size.h / 2,
                );
                let idx = self
                    .outpane_at(cursor)
                    .or_else(|| self.outpane_at(center))
                    .or_else(|| (!self.outputs.is_empty()).then_some(0));
                let Some(idx) = idx else {
                    // No outputs: keep the window alive as floating.
                    push_configure_for_floating(&t.window, self.border_width);
                    self.floating.push(t.window);
                    return;
                };
                let tile_bounds = self.tile_bounds(idx);
                let inner = self.gaps.inner;
                let leaf = Node::Leaf(t.window);
                self.outputs[idx].tree = Some(match self.outputs[idx].tree.take() {
                    None => leaf,
                    Some(root) => insert_at_cursor(root, leaf, tile_bounds, Some(cursor), inner),
                });
                self.recompute_and_push();
            }
            DragSource::Floating => {
                push_configure_for_floating(&t.window, self.border_width);
                self.floating.push(t.window);
            }
        }
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
    pub fn placements(&self, focused: Option<&WlSurface>) -> Vec<Placement> {
        let is_focused = |surface: &WlSurface| focused.is_some_and(|f| f == surface);
        let mut out = Vec::new();
        for op in &self.outputs {
            if let Some(tree) = &op.tree {
                collect_placements(tree, &is_focused, &mut out);
            }
        }
        for w in &self.floating {
            out.push(Placement {
                surface: w.toplevel.wl_surface().clone(),
                cell_rect: w.rect,
                focused: is_focused(w.toplevel.wl_surface()),
            });
        }
        if let Some(t) = &self.in_transit {
            out.push(Placement {
                surface: t.window.toplevel.wl_surface().clone(),
                cell_rect: t.window.rect,
                focused: is_focused(t.window.toplevel.wl_surface()),
            });
        }
        out
    }

    /// Hit-test the topmost window at `pos`. Floating windows win
    /// over tiled (they're on top), and within floating the
    /// top-of-stack (last-clicked / last-floated) wins. The
    /// in-transit window is intentionally skipped — it tracks
    /// the cursor by construction, so reporting it as "under the
    /// cursor" would just defeat focus changes for the duration
    /// of the drag.
    pub fn window_at(&self, pos: Point<i32, Physical>) -> Option<&Window> {
        for w in self.floating.iter().rev() {
            if rect_contains(w.rect, pos) {
                return Some(w);
            }
        }
        // Tiled hit-test only the tree of the output `pos` falls in.
        self.outpane_at(pos)
            .and_then(|i| self.outputs[i].tree.as_ref())
            .and_then(|t| leaf_at(t, pos))
    }

    fn recompute_and_push(&mut self) {
        let inner = self.gaps.inner;
        let outer = self.gaps.outer;
        let border = self.border_width;
        for op in &mut self.outputs {
            let tile_bounds = shrink_for_outer(op.bounds, outer);
            if let Some(tree) = &mut op.tree {
                assign_rects(tree, tile_bounds, inner);
            }
        }
        for op in &self.outputs {
            if let Some(tree) = &op.tree {
                push_configures_tree(tree, border);
            }
        }
        for w in &self.floating {
            push_configure_for_floating(w, border);
        }
    }

    /// Update the rectangle a specific output's tiles lay out inside,
    /// then reflow. Called when an output's usable area changes — e.g.
    /// a `wlr_layer_shell` panel reserves an exclusive zone, shrinking
    /// the tile area to avoid it. The output is keyed by connector
    /// name; an unknown name is a silent no-op so the renderer's and
    /// layout's output sets can drift without panicking.
    pub fn set_output_bounds(&mut self, name: &str, new_bounds: Rectangle<i32, Physical>) {
        let Some(op) = self.outputs.iter_mut().find(|o| o.name == name) else {
            return;
        };
        if op.bounds == new_bounds {
            return;
        }
        op.bounds = new_bounds;
        self.recompute_and_push();
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

/// Walk the tree and emit a `Placement` for every leaf.
fn collect_placements(
    node: &Node,
    is_focused: &impl Fn(&WlSurface) -> bool,
    out: &mut Vec<Placement>,
) {
    match node {
        Node::Leaf(w) => out.push(Placement {
            surface: w.toplevel.wl_surface().clone(),
            cell_rect: w.rect,
            focused: is_focused(w.toplevel.wl_surface()),
        }),
        Node::Split { first, second, .. } => {
            collect_placements(first, is_focused, out);
            collect_placements(second, is_focused, out);
        }
    }
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

/// Ship `xdg_toplevel.configure` for every leaf in the tree.
/// Tiles are configured with `Activated + Tiled{Left,Right,Top,
/// Bottom}` so that clients (notably kitty) treat the cell as a
/// hard size to fill, without leaving margins for their own
/// resize handles or rounding to a font grid.
fn push_configures_tree(node: &Node, border: i32) {
    match node {
        Node::Leaf(w) => push_configure_for_tile(w, border),
        Node::Split { first, second, .. } => {
            push_configures_tree(first, border);
            push_configures_tree(second, border);
        }
    }
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
fn push_configure_for_tile(w: &Window, border: i32) {
    let size = surface_size(w.rect.size, border);
    w.toplevel.with_pending_state(|state| {
        state.size = Some(size);
        state.states.set(xdg_toplevel::State::Activated);
        state.states.set(xdg_toplevel::State::TiledLeft);
        state.states.set(xdg_toplevel::State::TiledRight);
        state.states.set(xdg_toplevel::State::TiledTop);
        state.states.set(xdg_toplevel::State::TiledBottom);
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
fn push_configure_for_floating(w: &Window, border: i32) {
    let size = surface_size(w.rect.size, border);
    w.toplevel.with_pending_state(|state| {
        state.size = Some(size);
        state.states.set(xdg_toplevel::State::Activated);
        state.states.unset(xdg_toplevel::State::TiledLeft);
        state.states.unset(xdg_toplevel::State::TiledRight);
        state.states.unset(xdg_toplevel::State::TiledTop);
        state.states.unset(xdg_toplevel::State::TiledBottom);
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
            let split = split.clamp(1, bounds.size.w.max(1) - 1);
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
            let split = split.clamp(1, bounds.size.h.max(1) - 1);
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
