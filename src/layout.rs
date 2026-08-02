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
use smithay::wayland::compositor::with_states;
use smithay::wayland::shell::xdg::dialog::ToplevelDialogHint;
use smithay::wayland::shell::xdg::{SurfaceCachedState, ToplevelSurface, XdgToplevelSurfaceData};
use smithay::xwayland::X11Surface;
use smithay::xwayland::xwm::WmWindowType;
use tracing::debug;

use crate::config::{AnimSpec, LayoutMode, SlideAxis, SlideSpec};

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

/// The protocol handle behind a managed window: a native Wayland
/// `xdg_toplevel`, or an Xwayland (X11) window managed via the XWM.
/// The layout treats both uniformly — everything is keyed by the
/// window's `wl_surface`, and only the configure push (and the dialog
/// heuristic) dispatches on the protocol.
#[derive(Debug, Clone, PartialEq)]
pub enum WindowSurface {
    Xdg(ToplevelSurface),
    /// An X11 window plus the `wl_surface` Xwayland associated with it.
    /// The surface is cached here at insert time — a window only enters
    /// the layout once the xwayland-shell association exists — so
    /// [`WindowSurface::wl_surface`] stays total (no `Option`), keeping
    /// every surface-keyed lookup identical for both kinds. Boxed
    /// because [`X11Surface`] carries the whole XWM atom table inline,
    /// which would otherwise balloon every layout tree node for the
    /// common (pure-Wayland) case.
    X11 {
        surface: Box<X11Surface>,
        wl_surface: WlSurface,
    },
}

impl WindowSurface {
    /// The `wl_surface` that renders this window — the universal key
    /// for layout lookups, focus, and the render placements.
    pub fn wl_surface(&self) -> &WlSurface {
        match self {
            Self::Xdg(toplevel) => toplevel.wl_surface(),
            Self::X11 { wl_surface, .. } => wl_surface,
        }
    }

    /// Politely ask the window to close — `xdg_toplevel.close` or the
    /// X11 `WM_DELETE_WINDOW` protocol. The client drives its own
    /// teardown; the destroy/unmap handlers pull it from the layout.
    pub fn send_close(&self) {
        match self {
            Self::Xdg(toplevel) => toplevel.send_close(),
            Self::X11 { surface, .. } => {
                if let Err(err) = surface.close() {
                    debug!(window = surface.window_id(), %err, "X11 close failed");
                }
            }
        }
    }
}

/// One window managed by the layout, plus its current placement.
/// The `rect` is the cell the layout has assigned (refreshed by
/// every reflow) — clients see the same size via
/// `xdg_toplevel.configure` (or an X11 `ConfigureWindow`).
#[derive(Debug, Clone)]
pub struct Window {
    pub toplevel: WindowSurface,
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
    /// Net workspace steps from the snapshot's workspace to the current
    /// active one, signed the same way `delta` is (`+1` = towards later
    /// workspaces).
    ///
    /// A *count*, not an index, precisely because `normalize_output`
    /// compacts empty workspaces out of the list mid-slide and would
    /// invalidate any index we stored. Retargeting reads it to tell
    /// "still heading the same way" from "scrolled back over the start".
    steps: i32,
}

impl WsTransition {
    /// Linear progress through the slide, `0.0` at the start and `>= 1.0`
    /// once it is over. Unlike [`transition_eased`] this is raw time, which
    /// is what the retarget decision wants — whether the animation is young
    /// enough to redirect, not where the eased curve currently sits.
    fn elapsed_frac(&self, spec: AnimSpec) -> f64 {
        let dur = spec.duration_secs();
        if dur <= 0.0 {
            return 1.0;
        }
        self.start.elapsed().as_secs_f64() / dur
    }
}

/// What a workspace switch should do with the slide already in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlideAction {
    /// Keep the running slide's origin snapshot and clock, re-aimed to end
    /// this many steps from where it started. One continuous motion.
    Aim(i32),
    /// Leave the output with no slide at all: the new destination is the
    /// workspace the running one started from, and a slide whose snapshot
    /// *is* its destination would draw that workspace twice, at two
    /// different offsets.
    Drop,
    /// Snapshot the current workspace and start a new slide.
    Fresh,
}

/// Decide what a switch of `delta` workspaces does to the slide in flight.
///
/// `running` describes that slide: its net steps from its own origin, and
/// whether it is still young enough to redirect (see [`WS_RETARGET_UNTIL`]).
/// `None` means no slide is running.
fn slide_action(running: Option<(i32, bool)>, delta: i32) -> SlideAction {
    // No slide, or one too far along to redirect without the incoming
    // workspace appearing already almost in place.
    let Some((steps, true)) = running else {
        return SlideAction::Fresh;
    };
    let aimed = steps + delta;
    if aimed == 0 {
        SlideAction::Drop
    } else if aimed.signum() == steps.signum() {
        SlideAction::Aim(aimed)
    } else {
        // Reversed past the origin: the snapshot would have to travel back
        // the way it came, flipping direction mid-flight.
        SlideAction::Fresh
    }
}

/// How far into a slide a new switch may still redirect it instead of
/// starting over.
///
/// Retargeting keeps the origin snapshot and the clock, so the incoming
/// workspace inherits however much of the leg is left. Early on that is
/// nearly all of it and the result is one smooth slide; late on it would be a
/// sliver, and the new workspace would appear already almost in place. Past
/// this point a fresh slide looks better, and a switch that late is a
/// deliberate second scroll rather than the flick this exists for.
const WS_RETARGET_UNTIL: f64 = 0.5;

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
    /// Decoration inset for a window that carries decoration. Windows
    /// that don't (fullscreen, and maximized while tiling) resolve to
    /// [`Deco::none`] through [`Layout::deco_for`].
    deco: Deco,
    /// How windows are arranged. Tiling descends the dwindle tree;
    /// floating never touches it.
    mode: LayoutMode,
}

/// One window + its current placement, as the renderer consumes
/// it. `cell_rect` is the full cell the layout allocates; the
/// renderer paints the border in `cell_rect` and the surface
/// inside it (`cell_rect` inset by [`Placement::deco`]).
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
    /// This window's decoration inset, already resolved against its fill
    /// (see [`Layout::deco_for`]) — so a fullscreen window carries
    /// [`Deco::none`] here and a maximized one carries the real inset
    /// only while floating.
    ///
    /// Shipped per placement rather than read from the config at the
    /// consumer, because both consumers need it *per window*: the
    /// renderer insets the surface inside the cell, and popup
    /// positioning needs the parent's window-geometry origin, which is
    /// `cell_rect.loc + deco.content_offset()`.
    pub deco: Deco,
    /// Extra vertical offset (compositor px) the renderer adds *after*
    /// per-window animation, used for the workspace slide so both the
    /// outgoing and incoming workspaces translate together without
    /// disturbing each window's own move animation (`cell_rect` stays
    /// the settled target). `0` outside a workspace transition.
    pub slide: Point<i32, Physical>,
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

/// The decoration inset between a window's **cell** (what the layout
/// allocates, what the renderer paints into) and its **content** (what
/// the client is configured to, what its buffer covers).
///
/// The border ring is symmetric; the titlebar sits above the content and
/// inside the ring, so only the top edge differs. Every conversion
/// between the two spaces goes through here — get one of them wrong and
/// the stored rect stops describing the window, after which the
/// decoration offscreen composites at the wrong scale and hit-testing
/// uses a rect the window doesn't occupy.
///
/// A window that carries no decoration at the moment (fullscreen, or
/// maximized while tiling) uses [`Deco::none`] rather than a special
/// case at each call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Deco {
    /// Border ring width, on all four sides.
    pub border: i32,
    /// Titlebar height, above the content and inside the ring. `0` when
    /// titlebars are off.
    pub titlebar: i32,
}

impl Deco {
    /// Clamps both components non-negative, so a hostile config can't
    /// produce a cell smaller than its own content.
    #[must_use]
    pub fn new(border: i32, titlebar: i32) -> Self {
        Self {
            border: border.max(0),
            titlebar: titlebar.max(0),
        }
    }

    /// No decoration at all — a fullscreen window's content *is* its cell.
    #[must_use]
    pub fn none() -> Self {
        Self {
            border: 0,
            titlebar: 0,
        }
    }

    /// Inset on the top edge: the border ring plus the titlebar under it.
    #[must_use]
    pub fn top(self) -> i32 {
        self.border + self.titlebar
    }

    /// Where a window's content starts, relative to its cell origin.
    /// This is also a popup's frame of reference: xdg positioners are
    /// relative to the parent's *window geometry*, which is the content,
    /// not the cell.
    #[must_use]
    pub fn content_offset(self) -> Point<i32, Physical> {
        Point::from((self.border, self.top()))
    }

    /// Content size for a cell of `cell_size`, clamped to a minimum of
    /// `1` on each axis — a zero-size configure is one the client cannot
    /// render, so it must never be shipped even for a degenerate cell.
    #[must_use]
    pub fn content_size(self, cell_size: Size<i32, Physical>) -> Size<i32, Logical> {
        Size::<i32, Logical>::from((
            (cell_size.w - 2 * self.border).max(1),
            (cell_size.h - self.top() - self.border).max(1),
        ))
    }

    /// The content rect inside `cell`: the origin shifted in by the
    /// inset, the size shrunk to match. X11 configures need the whole
    /// positioned rect rather than just a size.
    #[must_use]
    pub fn content_rect(self, cell: Rectangle<i32, Physical>) -> Rectangle<i32, Physical> {
        let size = self.content_size(cell.size);
        Rectangle::new(
            cell.loc + self.content_offset(),
            Size::from((size.w, size.h)),
        )
    }

    /// Inverse of [`Deco::content_size`]: the cell that would give a
    /// client exactly `content`. Saturating, because `content` is often
    /// client-supplied and the protocols validate only positivity.
    #[must_use]
    pub fn cell_size_for(self, content: Size<i32, Physical>) -> Size<i32, Physical> {
        Size::from((
            content.w.saturating_add(2 * self.border),
            content.h.saturating_add(self.top() + self.border),
        ))
    }
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

/// Which edges an interactive resize moves, picked from the half of the
/// window the press landed in (the *nearer* edge on each axis).
///
/// A tiled window can only move an edge that has a split divider on it,
/// and the right-most / bottom-most cells have none on those sides —
/// their outer edges are the workspace boundary. Choosing by press
/// position means every window is resizable from somewhere: grab a
/// right-most cell's left half and you drag the divider it shares with
/// its neighbour. Floating windows get the same rule, so one gesture
/// behaves identically everywhere and any corner works.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeEdges {
    /// Move the right edge (else the left one).
    pub right: bool,
    /// Move the bottom edge (else the top one).
    pub bottom: bool,
}

impl ResizeEdges {
    /// Pick the edges from where `cursor` sits inside `rect`.
    pub fn from_press(rect: Rectangle<i32, Physical>, cursor: Point<i32, Physical>) -> Self {
        Self {
            right: cursor.x >= rect.loc.x + rect.size.w / 2,
            bottom: cursor.y >= rect.loc.y + rect.size.h / 2,
        }
    }
}

/// Smallest a cell may be squeezed to by an interactive tiled resize,
/// so a drag can't collapse a neighbour to nothing (or hand its client
/// a degenerate size).
const MIN_TILE_W: i32 = 120;
/// Vertical counterpart of [`MIN_TILE_W`].
const MIN_TILE_H: i32 = 80;

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
        deco: Deco,
        mode: LayoutMode,
    ) -> Self {
        Self {
            outputs: outputs
                .into_iter()
                .map(|(name, bounds)| Outpane::new(name, bounds))
                .collect(),
            in_transit: None,
            gaps,
            deco,
            mode,
        }
    }

    /// The decoration a window carries *right now*.
    ///
    /// Fullscreen never has any — it owns the whole output and a bar
    /// across the top of a game is not what anyone asked for. Maximized
    /// keeps its decoration while floating (you need the titlebar to
    /// un-maximize, which is how every stacking WM behaves) and loses it
    /// while tiling, which is the behaviour tiling has always had.
    fn deco_for(&self, w: &Window) -> Deco {
        deco_for_fill(self.deco, self.mode, w.fill)
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
    /// The output's *full* rect, not the work area (`bounds`, which is
    /// `full` minus layer-shell exclusive zones). A point in an output's
    /// outer-gap margin OR a panel's exclusive-zone strip (e.g. under the
    /// bar) must still resolve to that output — otherwise a fullscreen
    /// window, which fills `full`, isn't hit-testable under the bar and the
    /// click falls through to nothing.
    fn outpane_at(&self, p: Point<i32, Physical>) -> Option<usize> {
        self.outputs.iter().position(|o| rect_contains(o.full, p))
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
    pub fn insert(&mut self, toplevel: WindowSurface, cursor: Option<Point<i32, Physical>>) {
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
    /// Returns whether `surface` was a tracked window that got toggled
    /// (a silent no-op, `false`, for surfaces we don't track).
    /// The client's declared size limits — xdg `min_size`/`max_size`,
    /// X11 `WM_NORMAL_HINTS` — converted to CELL space (content limits
    /// grown by the border ring) so callers can clamp cell rects
    /// directly. Unset axes come back as 0 (min) / `i32::MAX` (max).
    ///
    /// Configuring a floating window below its declared minimum is a
    /// protocol violation the client answers by committing its minimum
    /// anyway — leaving the stored rect lying about the window's real
    /// size (mis-scaled decoration compositing, wrong hit-testing).
    /// Every path that PICKS a float size clamps through this.
    pub fn client_size_limits(
        &self,
        surface: &WlSurface,
    ) -> (Size<i32, Physical>, Size<i32, Physical>) {
        let unbounded = (
            Size::<i32, Physical>::from((0, 0)),
            Size::<i32, Physical>::from((i32::MAX, i32::MAX)),
        );
        let Some(w) = self.window_ref(surface) else {
            return unbounded;
        };
        self.window_limits(w)
    }

    /// [`Self::client_size_limits`] for a window already in hand — needed
    /// mid-move, when the window is owned by neither the tree nor the
    /// float list and a surface lookup can't find it.
    fn window_limits(&self, w: &Window) -> (Size<i32, Physical>, Size<i32, Physical>) {
        let (min, max) = match &w.toplevel {
            WindowSurface::Xdg(t) => with_states(t.wl_surface(), |states| {
                let mut cached = states.cached_state.get::<SurfaceCachedState>();
                let cur = cached.current();
                (cur.min_size, cur.max_size)
            }),
            WindowSurface::X11 { surface, .. } => (
                surface.min_size().unwrap_or_default(),
                surface.max_size().unwrap_or_default(),
            ),
        };
        // The limits are CONTENT space (what the client is configured
        // to); the rects they constrain are CELL space, so each axis
        // grows by that axis's decoration before it can be compared.
        let deco = self.deco_for(w);
        // 0 means "unconstrained" on that axis, per both protocols.
        let grow_w = |v: i32| deco.cell_size_for(Size::from((v, 0))).w;
        let grow_h = |v: i32| deco.cell_size_for(Size::from((0, v))).h;
        let min = Size::<i32, Physical>::from((
            if min.w > 0 { grow_w(min.w) } else { 0 },
            if min.h > 0 { grow_h(min.h) } else { 0 },
        ));
        let max = Size::<i32, Physical>::from((
            if max.w > 0 { grow_w(max.w) } else { i32::MAX },
            if max.h > 0 { grow_h(max.h) } else { i32::MAX },
        ));
        (min, max)
    }

    /// Adopt a client-chosen size for a visible floating window: xdg
    /// clients answer an un-honourable configure (below their minimum)
    /// by committing their minimum, and self-resizing dialogs commit new
    /// sizes unprompted — in both cases the client's committed window
    /// geometry is authoritative and the stored rect must follow, or the
    /// decoration offscreen composites a lie (cropped/mis-scaled
    /// content, wrong hit rect). `content` is the committed window
    /// geometry size; the rect resizes around its top-left. Returns
    /// whether anything changed. No configure is pushed back — the size
    /// IS the client's, and echoing it would only add an ack cycle.
    ///
    /// Xdg only: X11 float resizes are granted (and the rect updated) in
    /// the `ConfigureRequest` path.
    pub fn reconcile_floating_size(
        &mut self,
        surface: &WlSurface,
        content: Size<i32, Physical>,
    ) -> bool {
        if content.w <= 0 || content.h <= 0 {
            return false;
        }
        let (deco, mode) = (self.deco, self.mode);
        for oi in 0..self.outputs.len() {
            // A window geometry is entirely client-chosen and validated by
            // the protocol only for positivity, so it must never reach the
            // rect raw: the rect drives a cell-sized GPU offscreen (a
            // 30000×30000 "window" is multi-GiB, re-allocated every frame
            // while the size animates) and the pointer hit-test (an
            // oversized invisible float would swallow every click on the
            // desktop). Cap at the output's full size — no real window is
            // usefully larger, and the client keeps rendering whatever it
            // likes inside. Saturating maths so a near-i32::MAX geometry
            // can't overflow the border add.
            let full = self.outputs[oi].full.size;
            let active = self.outputs[oi].active;
            if let Some(w) = self.outputs[oi].workspaces[active]
                .floating
                .iter_mut()
                .find(|w| w.toplevel.wl_surface() == surface)
            {
                if w.fill != FillMode::Normal || !matches!(w.toplevel, WindowSurface::Xdg(_)) {
                    return false;
                }
                let grown = deco_for_fill(deco, mode, w.fill).cell_size_for(content);
                let cell = Size::<i32, Physical>::from((
                    grown.w.min(full.w.max(1)),
                    grown.h.min(full.h.max(1)),
                ));
                if w.rect.size == cell {
                    return false;
                }
                w.rect.size = cell;
                return true;
            }
        }
        false
    }

    pub fn toggle_floating(&mut self, surface: &WlSurface) -> bool {
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
                    // 70% of the tile cell, but never outside the
                    // client's declared limits — a float configured
                    // below its min is answered with the min anyway,
                    // and the stored rect would lie about it.
                    let shrunk =
                        Size::<i32, Physical>::new((prev.size.w * 7) / 10, (prev.size.h * 7) / 10);
                    let (lmin, lmax) = self.window_limits(&window);
                    // A client minimum larger than the tile cell makes the
                    // float GROW, so the recentring offset goes negative —
                    // keep the top-left inside the work area (the user
                    // needs the titlebar/edges reachable).
                    let new_size = Size::<i32, Physical>::new(
                        shrunk.w.min(lmax.w).max(lmin.w).min(area.work.size.w),
                        shrunk.h.min(lmax.h).max(lmin.h).min(area.work.size.h),
                    );
                    let new_loc = Point::<i32, Physical>::new(
                        (prev.loc.x + (prev.size.w - new_size.w) / 2).max(area.work.loc.x),
                        (prev.loc.y + (prev.size.h - new_size.h) / 2).max(area.work.loc.y),
                    );
                    window.rect = Rectangle::new(new_loc, new_size);
                    push_configure_for_floating(&window, self.deco_for(&window), area);
                    self.active_ws_mut(oi).floating.push(window);
                    self.recompute_and_push();
                    return true;
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
            return true;
        }
        false
    }

    /// If `surface` is a child/dialog toplevel, pull it out of the tile
    /// tree and place it floating, centred on its output's work area —
    /// Hyprland-style auto-float so transient windows (a file-properties
    /// dialog, an app's preferences or login window) don't wedge into the
    /// tiling. Called once when the toplevel first maps, where its parent
    /// and size hints are finally set. Returns whether it was floated; the
    /// caller falls back to a normal tile reconfigure on `false`.
    ///
    /// "Dialog" is the same heuristic mature compositors use without a
    /// window rule: it has an xdg `parent`, or it pins itself to a fixed
    /// size (`min_size == max_size`), which only non-tileable windows do.
    pub fn float_if_dialog(&mut self, surface: &WlSurface) -> bool {
        // A "dialog" the size of a display is a game, not a dialog: games pin
        // min == max to their resolution, which is exactly the fixed-size
        // heuristic below. Floating one — worse, RESETTING ITS FILL — yanks a
        // freshly-fullscreened game back to a float, and the resulting
        // fullscreen↔floating configure ping-pong makes the client rebuild
        // its swapchain per flip until it gives up (DOOM under Wine-Wayland
        // rebuilt 40 times, then quit to black).
        let full_sizes: Vec<Size<i32, Physical>> =
            self.outputs.iter().map(|op| op.full.size).collect();
        for oi in 0..self.outputs.len() {
            let area = self.outputs[oi].area();
            let work = area.work;
            let ws = self.active_ws_mut(oi);
            let Some(root) = ws.tree.take() else {
                continue;
            };
            // Decide before disturbing the tree: only float genuine dialogs,
            // and put the untouched tree back if this isn't one (or the
            // surface lives on another output's active workspace). A window
            // already filled (fullscreen/maximized) is never a dialog — the
            // user/client explicitly asked for that state.
            let Some(pref) = leaf_ref(&root, surface).and_then(|w| {
                if w.fill != FillMode::Normal {
                    return None;
                }
                dialog_size(&w.toplevel)
                    .filter(|s| !(s.w > 0 && full_sizes.contains(s)))
            }) else {
                ws.tree = Some(root);
                continue;
            };
            let (root_after, removed) = remove_from_tree(root, surface);
            ws.tree = root_after;
            let Some(mut window) = removed else { continue };
            // Honour the dialog's requested size, clamped to the work area;
            // fall back to a third of it on any axis it left unconstrained.
            // The client's declared minimum outranks the work-area clamp
            // (a dialog larger than the screen is the client's own call —
            // shrinking it below its min just makes the client override
            // us and the rect lie), and its maximum caps the fallback.
            // `pref` is CONTENT space (the client's window geometry);
            // `window.rect` and the limits are CELL space, so grow by the
            // border ring before comparing or storing — otherwise the
            // dialog is configured 2*border smaller than it asked for.
            let (lmin, lmax) = self.window_limits(&window);
            let deco = self.deco_for(&window);
            let want = deco.cell_size_for(pref);
            let w = if pref.w > 0 {
                want.w.min(work.size.w)
            } else {
                work.size.w / 3
            }
            .min(lmax.w)
            .max(lmin.w);
            let h = if pref.h > 0 {
                want.h.min(work.size.h)
            } else {
                work.size.h / 3
            }
            .min(lmax.h)
            .max(lmin.h);
            let loc = Point::<i32, Physical>::new(
                work.loc.x + (work.size.w - w) / 2,
                work.loc.y + (work.size.h - h) / 2,
            );
            window.rect = Rectangle::new(loc, Size::<i32, Physical>::from((w, h)));
            window.fill = FillMode::Normal;
            push_configure_for_floating(&window, deco, area);
            self.active_ws_mut(oi).floating.push(window);
            self.recompute_and_push();
            return true;
        }
        false
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

    /// Start an interactive *resize* drag. Floating and tiled windows
    /// both qualify — a tile resizes by moving the split dividers on its
    /// edges (see [`Self::apply_resize`]). Returns the rect to use as
    /// the drag's start rect, or `None` if the window can't be resized.
    ///
    /// A maximized or fullscreen window is refused: it owns its whole
    /// output, so there is no divider to move and a free resize would
    /// desync its filled configure. Un-fill it first.
    pub fn start_resize_drag(&self, surface: &WlSurface) -> Option<Rectangle<i32, Physical>> {
        if self.is_filled(surface) {
            return None;
        }
        // Only a visible window can be under the press, so the active
        // workspaces are the whole search space.
        self.outputs.iter().find_map(|op| {
            let ws = &op.workspaces[op.active];
            ws.floating
                .iter()
                .find(|w| w.toplevel.wl_surface() == surface)
                .or_else(|| ws.tree.as_ref().and_then(|t| tree_leaf(t, surface)))
                .map(|w| w.rect)
        })
    }

    /// Apply an interactive resize: move the window's `edges` so they
    /// land on `target`'s corresponding sides.
    ///
    /// A floating window simply takes the rect. A **tiled** one has no
    /// rect of its own — its cell is derived from the split ratios — so
    /// the moved edges are translated into new ratios on the ancestor
    /// splits that own them and the workspace reflows around it. The
    /// neighbouring cells give up exactly the space the window gains,
    /// which is what makes the drag read as moving a shared divider.
    ///
    /// Silent no-op for an untracked surface, and per-edge for one whose
    /// moved edge is the workspace boundary (no divider there to move).
    pub fn apply_resize(
        &mut self,
        surface: &WlSurface,
        target: Rectangle<i32, Physical>,
        edges: ResizeEdges,
    ) {
        let floating = self.outputs.iter().any(|op| {
            op.workspaces[op.active]
                .floating
                .iter()
                .any(|w| w.toplevel.wl_surface() == surface)
        });
        if floating {
            // A float's rect *is* its state.
            self.set_floating_rect(surface, target);
            return;
        }
        let (inner, outer) = (self.gaps.inner, self.gaps.outer);
        let (deco, mode) = (self.deco, self.mode);
        for op in &mut self.outputs {
            let tile_bounds = shrink_for_outer(op.bounds, outer);
            let area = op.area();
            let active = op.active;
            let Some(tree) = op.workspaces[active].tree.as_mut() else {
                continue;
            };
            if !tree_contains(tree, surface) {
                continue;
            }
            let (mut done_h, mut done_v) = (false, false);
            resize_leaf(
                tree,
                tile_bounds,
                inner,
                surface,
                target,
                edges,
                &mut done_h,
                &mut done_v,
            );
            // Reflow and configure just this workspace: a drag ships one
            // of these per motion event, so walking every other
            // workspace's clients (what `recompute_and_push` does) would
            // be pure waste.
            assign_rects(tree, tile_bounds, inner);
            push_configures_tree(tree, deco, mode, area);
            return;
        }
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
            let deco = deco_for_fill(self.deco, self.mode, t.window.fill);
            push_configure_for_floating(&t.window, deco, area);
        }
    }

    /// Update a floating window's rect during a resize drag and
    /// ship the corresponding configure. Silent no-op for surfaces
    /// that aren't currently floating.
    pub fn set_floating_rect(&mut self, surface: &WlSurface, rect: Rectangle<i32, Physical>) {
        let (deco, mode) = (self.deco, self.mode);
        for op in &mut self.outputs {
            let active = op.active;
            let area = op.area();
            if let Some(window) = op.workspaces[active]
                .floating
                .iter_mut()
                .find(|w| w.toplevel.wl_surface() == surface)
            {
                window.rect = rect;
                push_configure_for_floating(window, deco_for_fill(deco, mode, window.fill), area);
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
                self.deco_for(&t.window),
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
                push_configure_for_floating(
                    &t.window,
                    self.deco_for(&t.window),
                    self.outputs[idx].area(),
                );
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
    pub fn placements(&self, focused: Option<&WlSurface>, slide: Option<SlideSpec>) -> Vec<Placement> {
        let is_focused = |surface: &WlSurface| focused.is_some_and(|f| f == surface);
        let (deco, mode) = (self.deco, self.mode);
        let mut out = Vec::new();
        // Only the active workspace of each output is visible — except
        // mid workspace-switch, where the outgoing (captured) and
        // incoming workspaces are both emitted, translated vertically.
        for op in &self.outputs {
            let area = op.area();
            #[allow(
                clippy::cast_possible_truncation,
                reason = "slide offset = small fraction × the output's extent (i32), well within range"
            )]
            if let (Some(t), Some(spec)) = (op.transition.as_ref(), slide)
                && let Some(p) = transition_eased(t, spec.for_steps(t.steps))
            {
                // The travel distance is the output's extent along the slide
                // axis; `dir` gives the sign, and the two workspaces are
                // always exactly one screen apart.
                let span = match spec.axis {
                    SlideAxis::Vertical => f64::from(op.full.size.h),
                    SlideAxis::Horizontal => f64::from(op.full.size.w),
                };
                let along = |d: f64| -> Point<i32, Physical> {
                    let n = (d * span).round() as i32;
                    match spec.axis {
                        SlideAxis::Vertical => Point::new(0, n),
                        SlideAxis::Horizontal => Point::new(n, 0),
                    }
                };
                let off_from = along(f64::from(t.dir) * p);
                let off_to = along(f64::from(-t.dir) * (1.0 - p));
                for fp in &t.from {
                    out.push(Placement {
                        slide: off_from,
                        focused: false, // the outgoing workspace isn't focused
                        ..fp.clone()
                    });
                }
                let base = out.len();
                collect_workspace(
                    &op.workspaces[op.active],
                    &is_focused,
                    area,
                    deco,
                    mode,
                    &mut out,
                );
                for q in &mut out[base..] {
                    q.slide = off_to;
                }
            } else {
                collect_workspace(
                    &op.workspaces[op.active],
                    &is_focused,
                    area,
                    deco,
                    mode,
                    &mut out,
                );
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
                deco: self.deco_for(&t.window),
                slide: Point::from((0, 0)),
            });
        }
        out
    }

    /// Clear workspace-switch transitions that have finished (or that
    /// can't run because the slide is disabled), freeing their captured
    /// snapshots. Call once per frame before [`Self::placements`].
    pub fn tick_transitions(&mut self, slide: Option<SlideSpec>) {
        for op in &mut self.outputs {
            let done = match (op.transition.as_ref(), slide) {
                (Some(t), Some(s)) => transition_eased(t, s.for_steps(t.steps)).is_none(),
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
        let (deco, mode) = (self.deco, self.mode);
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
                    push_configures_tree(tree, deco, mode, area);
                }
                for w in &ws.floating {
                    push_configure_for_floating(w, deco_for_fill(deco, mode, w.fill), area);
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

    /// Register a freshly hot-plugged output, appending it with one
    /// empty workspace at `bounds`. The caller positions/sizes it for
    /// real via [`Self::set_output_bounds`] right after. Idempotent: an
    /// output whose connector name we already track is a no-op, so a
    /// monitor reconnecting on the same port reuses its retained pane
    /// (and the windows still on it) rather than spawning a duplicate.
    pub fn add_output(&mut self, name: String, bounds: Rectangle<i32, Physical>) {
        if self.outputs.iter().any(|o| o.name == name) {
            return;
        }
        self.outputs.push(Outpane::new(name, bounds));
    }

    /// Drop a hot-unplugged output. Any windows it held (every
    /// workspace, tiled and floating) are re-homed onto `fallback`'s
    /// active workspace as tiles so their clients survive the unplug.
    ///
    /// When `fallback` is `None` — the last output just went away — the
    /// pane is *kept* intact instead of dropped: its windows stay parked
    /// in the layout (clients live on, headless) and reappear when an
    /// output returns under the same connector name. A `name` we don't
    /// track is a silent no-op.
    pub fn remove_output(&mut self, name: &str, fallback: Option<&str>) {
        let Some(idx) = self.outputs.iter().position(|o| o.name == name) else {
            return;
        };
        let Some(dst_name) = fallback else {
            // Nowhere to move the windows; keep the pane so they aren't lost.
            return;
        };
        let Some(mut di) = self.outputs.iter().position(|o| o.name == dst_name) else {
            return;
        };
        let pane = self.outputs.remove(idx);
        // `remove` shifted every later index down by one.
        if di > idx {
            di -= 1;
        }
        let mut rescued: Vec<Window> = Vec::new();
        for ws in pane.workspaces {
            if let Some(tree) = ws.tree {
                collect_windows_owned(tree, &mut rescued);
            }
            rescued.extend(ws.floating);
        }
        let tile_bounds = self.tile_bounds(di);
        let inner = self.gaps.inner;
        for mut window in rescued {
            // The fill mode targeted the vanished output; re-tile cleanly.
            window.fill = FillMode::Normal;
            let leaf = Node::Leaf(window);
            let ws = self.active_ws_mut(di);
            ws.tree = Some(match ws.tree.take() {
                None => leaf,
                Some(root) => insert_at_cursor(root, leaf, tile_bounds, None, inner),
            });
        }
        self.normalize_output(di);
        self.recompute_and_push();
    }

    /// Swap the gap + decoration settings and reflow every workspace
    /// (for live config reload). Tiles get re-laid-out with the new gaps
    /// and re-configured to the new content size; no-op-cheap when the
    /// values are unchanged.
    ///
    /// The layout **mode** is deliberately not settable here: switching
    /// it has to migrate windows between the tree and the float stack,
    /// which is [`Layout::set_mode`].
    pub fn set_appearance(&mut self, gaps: Gaps, deco: Deco) {
        self.gaps = gaps;
        self.deco = deco;
        self.recompute_and_push();
    }

    /// The decoration inset a window would carry if it is decorated —
    /// i.e. ignoring its fill. Callers that have the window in hand want
    /// [`Layout::deco_for`]; this is for the render/config side, which
    /// asks about the configuration rather than about one window.
    pub fn deco(&self) -> Deco {
        self.deco
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
    /// Whether `w × h` (compositor px) exactly matches some output's full
    /// rect. Wine/Proton games present through a borderless window sized
    /// to the display; the X11 manage path fullscreens such windows
    /// instead of tiling them (tiling triggers Wine's destroy/recreate
    /// churn — xwayland-satellite ships this same size heuristic).
    pub fn any_output_full_size(&self, w: i32, h: i32) -> bool {
        self.outputs
            .iter()
            .any(|op| op.full.size.w == w && op.full.size.h == h)
    }

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

    /// Flip `surface` between maximized and normal (IPC
    /// `toggle-maximized`). A fullscreen window drops to maximized;
    /// anything else toggles against maximized. Returns whether `surface`
    /// was found.
    pub fn toggle_maximized(&mut self, surface: &WlSurface) -> bool {
        let Some(w) = self.window_mut(surface) else {
            return false;
        };
        w.fill = if w.fill == FillMode::Maximized {
            FillMode::Normal
        } else {
            FillMode::Maximized
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

    /// Re-send `surface`'s configure (size + tiled/maximized/fullscreen
    /// states) for its current rect, on whatever workspace it lives. A client
    /// that ignored the size in its *initial* configure — MPV's idle window
    /// maps at a default size and only repaints when a *later* configure
    /// arrives — snaps to its cell when nudged this way (the same thing a
    /// window move does). Returns whether `surface` is a window we track.
    /// Whether `surface` is a managed window parked on a **non-active**
    /// workspace — mapped, committing, but invisible, with nothing ever
    /// presenting its frames. Drives the immediate `discarded` for its
    /// `wp_presentation` feedback (see the commit handler): parking the
    /// feedback until the workspace returns starves present-timing
    /// consumers.
    pub fn on_inactive_workspace(&self, surface: &WlSurface) -> bool {
        for op in &self.outputs {
            for (index, ws) in op.workspaces.iter().enumerate() {
                if index == op.active {
                    continue;
                }
                if ws
                    .tree
                    .as_ref()
                    .is_some_and(|t| leaf_ref(t, surface).is_some())
                    || ws.floating.iter().any(|w| w.toplevel.wl_surface() == surface)
                {
                    return true;
                }
            }
        }
        false
    }

    /// The protocol handle of the window matching `surface`, wherever
    /// it lives (tree or floating, any workspace). Lets callers act on
    /// the right protocol — e.g. a close request — without caring
    /// whether the window is xdg or X11.
    pub fn window_surface(&self, surface: &WlSurface) -> Option<WindowSurface> {
        for op in &self.outputs {
            for ws in &op.workspaces {
                if let Some(t) = &ws.tree
                    && let Some(w) = leaf_ref(t, surface)
                {
                    return Some(w.toplevel.clone());
                }
                if let Some(w) = ws
                    .floating
                    .iter()
                    .find(|w| w.toplevel.wl_surface() == surface)
                {
                    return Some(w.toplevel.clone());
                }
            }
        }
        None
    }

    pub fn reconfigure(&self, surface: &WlSurface) -> bool {
        for op in &self.outputs {
            let area = op.area();
            for ws in &op.workspaces {
                if let Some(t) = &ws.tree
                    && let Some(w) = leaf_ref(t, surface)
                {
                    push_configure_for_tile(w, self.deco_for(w), area);
                    return true;
                }
                if let Some(w) = ws
                    .floating
                    .iter()
                    .find(|w| w.toplevel.wl_surface() == surface)
                {
                    push_configure_for_floating(w, self.deco_for(w), area);
                    return true;
                }
            }
        }
        false
    }

    /// Mutable [`Self::window_ref`]. In-transit is checked first so the
    /// loops are the function tail — the borrow checker rejects
    /// reborrowing `self` after a loop that conditionally returns a
    /// `&mut` from inside it.
    /// Connector name of the output whose *active* workspace currently
    /// holds `surface` (tiled or floating). The on-demand renderer redraws
    /// just this output when `surface` commits, so a window elsewhere — on
    /// another output or in a hidden workspace — doesn't wake (and stutter
    /// the VRR of) an unrelated output. `None` for a hidden / in-transit
    /// window or a non-toplevel surface.
    pub fn output_of(&self, surface: &WlSurface) -> Option<String> {
        for op in &self.outputs {
            let Some(ws) = op.workspaces.get(op.active) else {
                continue;
            };
            let here = ws
                .floating
                .iter()
                .any(|w| w.toplevel.wl_surface() == surface)
                || ws.tree.as_ref().is_some_and(|t| tree_contains(t, surface));
            if here {
                return Some(op.name.clone());
            }
        }
        None
    }

    /// Whether output `name`'s active workspace shows a fullscreen window
    /// whose surface isn't `surface` — i.e. `surface` is occluded behind a
    /// fullscreen window there. The on-demand renderer skips redrawing such
    /// an output for `surface`'s commits so it doesn't stutter the
    /// fullscreen window's VRR.
    pub fn output_fullscreen_other_than(&self, name: &str, surface: &WlSurface) -> bool {
        self.fullscreen_surface(name).is_some_and(|s| s != surface)
    }

    /// Whether output `name`'s active workspace shows a fullscreen window.
    pub fn output_has_fullscreen(&self, name: &str) -> bool {
        self.fullscreen_surface(name).is_some()
    }

    /// The surface of the fullscreen window in output `name`'s active
    /// workspace, if any (tiled or floating).
    fn fullscreen_surface(&self, name: &str) -> Option<&WlSurface> {
        let op = self.outputs.iter().find(|o| o.name == name)?;
        let ws = op.workspaces.get(op.active)?;
        if let Some(w) = ws
            .floating
            .iter()
            .find(|w| w.fill == FillMode::Fullscreen)
        {
            return Some(w.toplevel.wl_surface());
        }
        ws.tree
            .as_ref()
            .and_then(tree_fullscreen)
            .map(|w| w.toplevel.wl_surface())
    }

    /// Whether any output has a workspace-switch slide in progress, so the
    /// on-demand renderer keeps redrawing until it finishes.
    pub fn is_animating(&self) -> bool {
        self.outputs.iter().any(|op| op.transition.is_some())
    }

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

    /// Name of the output under `cursor`, if any. Lets a workspace keybind
    /// act on the same monitor `Super`+scroll would have.
    #[must_use]
    pub fn output_name_at(&self, cursor: Point<i32, Physical>) -> Option<&str> {
        self.outpane_at(cursor).map(|oi| self.outputs[oi].name.as_str())
    }

    /// Switch the active workspace on the output under `cursor` by
    /// `delta` (`+1` = next / scroll-down, `-1` = previous /
    /// scroll-up). No-op if the cursor is over no output. No wrap:
    /// scrolling up past the first workspace stays put. Returns
    /// whether the active workspace actually changed (so the caller
    /// can re-derive keyboard focus only when it did).
    ///
    /// `slide` is the workspace-slide animation spec (`None` when disabled),
    /// needed to tell a flick of the scroll wheel — which redirects the slide
    /// already running — from a deliberate later switch, which starts a fresh
    /// one. See [`WS_RETARGET_UNTIL`].
    pub fn switch_at(
        &mut self,
        cursor: Point<i32, Physical>,
        delta: i32,
        slide: Option<SlideSpec>,
    ) -> bool {
        self.outpane_at(cursor)
            .is_some_and(|oi| self.switch(oi, delta, slide))
    }

    /// Switch output `oi`'s active workspace by `delta`. Materializes
    /// a fresh trailing-empty workspace to scroll into when moving
    /// past the end; `normalize_output` then compacts the workspace
    /// we left if it became empty and trims back to one trailing
    /// empty, so the list can't grow without bound. Returns whether
    /// the active workspace changed.
    fn switch(&mut self, oi: usize, delta: i32, slide: Option<SlideSpec>) -> bool {
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
        self.switch_to_index(oi, target as usize, slide)
    }

    /// Switch output `oi` to the absolute workspace `target`, growing the
    /// list with empties if `target` is past the end. Shared by the
    /// scroll gesture ([`Self::switch`]) and the IPC
    /// [`Self::switch_workspace_to`]. Returns whether `active` changed.
    fn switch_to_index(&mut self, oi: usize, target: usize, slide: Option<SlideSpec>) -> bool {
        // `target` arrives unchecked from IPC (`focus-workspace {index}`);
        // growing is only ever meant to open ONE fresh workspace past the
        // end, so clamp before the loop — an arbitrary index must not
        // drive an allocation storm.
        let target = target.min(self.outputs[oi].workspaces.len());
        while target >= self.outputs[oi].workspaces.len() {
            self.outputs[oi].workspaces.push(Workspace::default());
        }
        let active = self.outputs[oi].active;
        if target == active {
            return false;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "workspace indices are small Vec indices, never near i32 bounds"
        )]
        let delta = target as i32 - active as i32;

        // Redirect a slide that is already running rather than replacing it.
        //
        // Without this, every switch snapshots the *current* workspace as the
        // new origin and resets the clock — so scrolling three notches
        // quickly starts three slides, each superseded a few milliseconds in.
        // Nothing visibly moves and you simply arrive three workspaces later.
        // Keeping the origin and the clock turns that into what the user
        // actually did: one continuous slide from where they were to where
        // they stopped.
        //
        // Only while the slide is young, though: past [`WS_RETARGET_UNTIL`]
        // the incoming workspace would inherit a sliver of the leg's
        // remaining time and appear already almost in place. A switch that
        // late is a deliberate second scroll, and deserves its own slide.
        let running = self.outputs[oi].transition.as_ref().map(|t| {
            (
                t.steps,
                slide.is_some_and(|s| t.elapsed_frac(s.for_steps(t.steps)) < WS_RETARGET_UNTIL),
            )
        });
        let action = slide_action(running, delta);
        match action {
            SlideAction::Aim(steps) => {
                self.outputs[oi]
                    .transition
                    .as_mut()
                    .expect("Aim is only returned for a running slide")
                    .steps = steps;
            }
            SlideAction::Drop => self.outputs[oi].transition = None,
            SlideAction::Fresh => {}
        }

        if action == SlideAction::Fresh {
            // Snapshot the outgoing workspace for the slide animation before
            // `active` moves (and before `normalize_output` may reindex the
            // workspace list, which would invalidate a stored index).
            let area = self.outputs[oi].area();
            let mut from = Vec::new();
            collect_workspace(
                &self.outputs[oi].workspaces[active],
                &|_| false,
                area,
                self.deco,
                self.mode,
                &mut from,
            );
            self.outputs[oi].transition = Some(WsTransition {
                from,
                // Moving to a later workspace slides up (incoming from the
                // bottom), to an earlier one slides down — the natural
                // scroll mapping, also correct for absolute IPC jumps.
                dir: if delta > 0 { -1 } else { 1 },
                start: Instant::now(),
                steps: delta,
            });
        }
        self.outputs[oi].active = target;
        self.normalize_output(oi);
        self.recompute_and_push();
        true
    }

    /// IPC: switch the named output to workspace `index` (absolute).
    /// No-op (returns `false`) if there's no output by that name.
    pub fn switch_workspace_to(
        &mut self,
        output: &str,
        index: usize,
        slide: Option<SlideSpec>,
    ) -> bool {
        let Some(oi) = self.outputs.iter().position(|o| o.name == output) else {
            return false;
        };
        self.switch_to_index(oi, index, slide)
    }

    /// Move the keyboard-focused window to the adjacent workspace on
    /// **its own** output and follow it there (the destination
    /// becomes active). Handles both tiled and floating focused
    /// windows. No wrap: `Shift`+scroll-up while on workspace 0 is a
    /// no-op. Returns `true` if a window actually moved; `false` if
    /// `surface` isn't on any visible workspace or the move was a
    /// no-op (at the top edge).
    pub fn move_focused_window(&mut self, surface: &WlSurface, delta: i32) -> bool {
        let Some((oi, is_floating)) = self.find_visible(surface) else {
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
        self.relocate(surface, oi, is_floating, dst as usize)
    }

    /// IPC: move `surface` to workspace `index` (absolute) on its own
    /// output and follow it. Only finds windows on a *visible* (active)
    /// workspace; returns `false` if `surface` isn't currently visible or
    /// the move is a no-op.
    pub fn move_window_to_workspace(&mut self, surface: &WlSurface, index: usize) -> bool {
        let Some((oi, is_floating)) = self.find_visible(surface) else {
            return false;
        };
        self.relocate(surface, oi, is_floating, index)
    }

    /// Locate `surface` on some output's active workspace, returning that
    /// output index and whether it's floating. `None` if it isn't on any
    /// visible workspace.
    fn find_visible(&self, surface: &WlSurface) -> Option<(usize, bool)> {
        for (oi, op) in self.outputs.iter().enumerate() {
            let ws = &op.workspaces[op.active];
            if ws
                .floating
                .iter()
                .any(|w| w.toplevel.wl_surface() == surface)
            {
                return Some((oi, true));
            }
            if ws.tree.as_ref().is_some_and(|t| tree_contains(t, surface)) {
                return Some((oi, false));
            }
        }
        None
    }

    /// Move `surface` (known to be `is_floating` on output `oi`'s active
    /// workspace) to absolute workspace `dst`, growing the list if needed,
    /// then follow it (the destination becomes active). Returns whether a
    /// move happened (`false` if `dst` is already the active workspace).
    fn relocate(&mut self, surface: &WlSurface, oi: usize, is_floating: bool, dst: usize) -> bool {
        // Same clamp as `switch_to_index`: `dst` is IPC-controlled
        // (`move-to-workspace {index}`), grow by at most one.
        let dst = dst.min(self.outputs[oi].workspaces.len());
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
    deco: Deco,
    mode: LayoutMode,
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
                deco: deco_for_fill(deco, mode, w.fill),
                slide: Point::from((0, 0)),
            });
        }
        Node::Split { first, second, .. } => {
            collect_placements(first, is_focused, area, deco, mode, out);
            collect_placements(second, is_focused, area, deco, mode, out);
        }
    }
}

/// Build one workspace's placements: the tiled tree, then floating
/// windows bottom-up (drawn above the tiles they overlap).
fn collect_workspace(
    ws: &Workspace,
    is_focused: &impl Fn(&WlSurface) -> bool,
    area: OutputArea,
    deco: Deco,
    mode: LayoutMode,
    out: &mut Vec<Placement>,
) {
    if let Some(tree) = &ws.tree {
        collect_placements(tree, is_focused, area, deco, mode, out);
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
            deco: deco_for_fill(deco, mode, w.fill),
            slide: Point::from((0, 0)),
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
/// Consume a dwindle tree, draining every window it holds into `out`
/// (depth-first). Used when an output is unplugged and its windows must
/// be moved wholesale onto another output.
fn collect_windows_owned(node: Node, out: &mut Vec<Window>) {
    match node {
        Node::Leaf(w) => out.push(w),
        Node::Split { first, second, .. } => {
            collect_windows_owned(*first, out);
            collect_windows_owned(*second, out);
        }
    }
}

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

/// Classify a toplevel as a dialog that should auto-float, returning its
/// preferred size (logical px, re-tagged to the layout's `Physical`-as-logical
/// space) — or `None` to leave it tiled. A zero axis means "no preference"; the
/// caller substitutes a fraction of the work area.
///
/// A toplevel is a dialog if it declares an xdg `parent` (a transient child —
/// a properties or preferences window over its app) or pins a fixed size
/// (`min_size == max_size`, which only non-tileable windows do). The preferred
/// size is the client's window geometry, else its fixed size.
fn dialog_size(toplevel: &WindowSurface) -> Option<Size<i32, Physical>> {
    match toplevel {
        WindowSurface::Xdg(toplevel) => {
            let has_parent = toplevel.parent().is_some();
            let (min, max, geo, hint) = with_states(toplevel.wl_surface(), |states| {
                let hint = states
                    .data_map
                    .get::<XdgToplevelSurfaceData>()
                    .map(|d| d.lock().unwrap().dialog_hint)
                    .unwrap_or_default();
                let mut cached = states.cached_state.get::<SurfaceCachedState>();
                let cur = cached.current();
                (cur.min_size, cur.max_size, cur.geometry.map(|g| g.size), hint)
            });
            // xdg-wm-dialog: the client's own declaration beats any
            // heuristic — a window that says "I'm a dialog" floats, full
            // stop (the output-sized guard in the caller still applies).
            let declared = matches!(
                hint,
                ToplevelDialogHint::Dialog | ToplevelDialogHint::Modal
            );
            let fixed = min.w > 0 && min.h > 0 && min == max;
            if !declared && !has_parent && !fixed {
                return None;
            }
            // Window geometry is the visible size sans shadows; prefer it,
            // else the pinned size, else leave both axes 0 for the caller
            // to fill in.
            let size = geo
                .filter(|s| s.w > 0 && s.h > 0)
                .or(fixed.then_some(min))
                .unwrap_or_default();
            Some(Size::<i32, Physical>::from((size.w, size.h)))
        }
        WindowSurface::X11 { surface, .. } => {
            // The X11 equivalents of the xdg heuristic: a transient-for
            // hint is the xdg `parent`, a dialog/utility/splash/toolbar
            // `NET_WM_WINDOW_TYPE` is an explicit "I'm not a main
            // window", and pinned WM_NORMAL_HINTS (min == max) marks the
            // same non-tileable windows it does on Wayland.
            let has_parent = surface.is_transient_for().is_some();
            let typed_dialog = matches!(
                surface.window_type(),
                Some(
                    WmWindowType::Dialog
                        | WmWindowType::Utility
                        | WmWindowType::Splash
                        | WmWindowType::Toolbar
                )
            );
            let (min, max) = (surface.min_size(), surface.max_size());
            let fixed = min.is_some() && min == max;
            // `is_modal()`: an explicit _NET_WM_STATE_MODAL declaration —
            // the X11 twin of the xdg dialog hint honoured above (tracked
            // by git-smithay; 0.7.0 ignored MODAL entirely).
            if !has_parent && !typed_dialog && !fixed && !surface.is_modal() {
                return None;
            }
            // `last_configure()` is the size the client asked to map at
            // (the tracked X-side rect — 0.7.0's `geometry()`), which is
            // exactly the dialog's preferred size; fall back to the
            // pinned minimum. The NEW `geometry()` subtracts
            // _GTK_FRAME_EXTENTS, and configuring an X window (full-size
            // space) to the shadow-subtracted size shrinks the visible
            // dialog by its shadows on every map.
            let size = Some(surface.last_configure().size)
                .filter(|s| s.w > 0 && s.h > 0)
                .or(if fixed { min } else { None })
                .unwrap_or_default();
            Some(Size::<i32, Physical>::from((size.w, size.h)))
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

/// First fullscreen window in a tiling tree, if any.
fn tree_fullscreen(node: &Node) -> Option<&Window> {
    match node {
        Node::Leaf(w) => (w.fill == FillMode::Fullscreen).then_some(w),
        Node::Split { first, second, .. } => {
            tree_fullscreen(first).or_else(|| tree_fullscreen(second))
        }
    }
}

/// Ship `xdg_toplevel.configure` for every leaf in the tree.
/// Tiles are configured with `Activated + Tiled{Left,Right,Top,
/// Bottom}` so that clients (notably kitty) treat the cell as a
/// hard size to fill, without leaving margins for their own
/// resize handles or rounding to a font grid.
/// The leaf holding `surface`, anywhere in this tree.
fn tree_leaf<'a>(node: &'a Node, surface: &WlSurface) -> Option<&'a Window> {
    match node {
        Node::Leaf(w) => (w.toplevel.wl_surface() == surface).then_some(w),
        Node::Split { first, second, .. } => {
            tree_leaf(first, surface).or_else(|| tree_leaf(second, surface))
        }
    }
}

/// Retarget the split ratios that own `surface`'s moving edges so those
/// edges land on `target`. Returns whether this subtree contains it.
///
/// The divider that owns an edge is the *nearest* ancestor split on that
/// axis with the leaf on the matching side: for a `LeftRight` split the
/// divider is the leaf's right edge when the leaf sits in `first`, and
/// its left edge when it sits in `second` (`TopBottom` likewise for
/// bottom/top). Walking back up post-order, the first such ancestor wins
/// and the `done` flags keep any higher one from moving the same edge
/// again. An edge with no owning ancestor is the workspace boundary and
/// simply doesn't move.
///
/// The two axes never interfere: a `LeftRight` ratio only shifts its
/// children's x/width and a `TopBottom` ratio only their y/height, so
/// retargeting one axis can't invalidate the bounds the other was
/// computed from — which is why one pass gets both edges exactly right.
#[allow(
    clippy::too_many_arguments,
    reason = "one recursive walk carrying the bounds, the target, and the per-axis done flags; bundling them into a struct would only rename the same state"
)]
fn resize_leaf(
    node: &mut Node,
    bounds: Rectangle<i32, Physical>,
    inner: i32,
    surface: &WlSurface,
    target: Rectangle<i32, Physical>,
    edges: ResizeEdges,
    done_h: &mut bool,
    done_v: &mut bool,
) -> bool {
    match node {
        Node::Leaf(w) => w.toplevel.wl_surface() == surface,
        Node::Split {
            axis,
            ratio,
            first,
            second,
        } => {
            let (b1, b2) = split_bounds(bounds, *axis, *ratio, inner);
            let in_first = resize_leaf(first, b1, inner, surface, target, edges, done_h, done_v);
            let in_second = !in_first
                && resize_leaf(second, b2, inner, surface, target, edges, done_h, done_v);
            if !in_first && !in_second {
                return false;
            }
            let gap = inner.max(0);
            let (half_a, half_b) = (gap / 2, gap - gap / 2);
            // Where this split's divider would have to sit for the edge
            // it owns to land on the target. Per `split_bounds`, `first`
            // ends at `split - half_a` and `second` starts at
            // `split + half_b`.
            let (want, span, origin, min) = match axis {
                SplitAxis::LeftRight => (
                    if *done_h {
                        None
                    } else if in_first && edges.right {
                        Some(target.loc.x + target.size.w + half_a)
                    } else if in_second && !edges.right {
                        Some(target.loc.x - half_b)
                    } else {
                        None
                    },
                    bounds.size.w,
                    bounds.loc.x,
                    MIN_TILE_W,
                ),
                SplitAxis::TopBottom => (
                    if *done_v {
                        None
                    } else if in_first && edges.bottom {
                        Some(target.loc.y + target.size.h + half_a)
                    } else if in_second && !edges.bottom {
                        Some(target.loc.y - half_b)
                    } else {
                        None
                    },
                    bounds.size.h,
                    bounds.loc.y,
                    MIN_TILE_H,
                ),
            };
            if let Some(split) = want {
                *ratio = ratio_for_divider(span, origin, split, min);
                match axis {
                    SplitAxis::LeftRight => *done_h = true,
                    SplitAxis::TopBottom => *done_v = true,
                }
            }
            true
        }
    }
}

/// The ratio that puts a split's divider at the absolute coordinate
/// `split_px`, within a cell starting at `origin` and `span` px long.
/// Clamped so neither side falls below `min` px.
///
/// Pure geometry, split out from [`resize_leaf`] because the clamp is
/// the fiddly part: `lo`/`hi` stay ordered even when the span is
/// narrower than two minimums (a very small cell), so `clamp` can never
/// see `min > max` — which would panic.
fn ratio_for_divider(span: i32, origin: i32, split_px: i32, min: i32) -> f32 {
    let lo = min.min(span / 2);
    let hi = (span - min).max(lo);
    let split = (split_px - origin).clamp(lo, hi);
    #[allow(
        clippy::cast_precision_loss,
        reason = "both are output-bounded pixel counts; f32 is exact well past any real display size"
    )]
    let ratio = split as f32 / span.max(1) as f32;
    ratio
}

fn push_configures_tree(node: &Node, deco: Deco, mode: LayoutMode, area: OutputArea) {
    match node {
        Node::Leaf(w) => push_configure_for_tile(w, deco_for_fill(deco, mode, w.fill), area),
        Node::Split { first, second, .. } => {
            push_configures_tree(first, deco, mode, area);
            push_configures_tree(second, deco, mode, area);
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
    match &w.toplevel {
        WindowSurface::Xdg(toplevel) => {
            let size = Size::<i32, Logical>::from((rect.size.w.max(1), rect.size.h.max(1)));
            toplevel.with_pending_state(|state| {
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
            toplevel.send_configure();
        }
        WindowSurface::X11 { surface, .. } => push_x11_configure(surface, rect, w.fill),
    }
    debug!(
        surface = ?w.toplevel.wl_surface().id(),
        w = rect.size.w,
        h = rect.size.h,
        fill = ?w.fill,
        "layout: fullscreen/maximized configure sent",
    );
}

/// Configure an X11 window to `rect` (global compositor coordinates —
/// unlike xdg, X11 configures carry position too, which is what keeps
/// Xwayland's idea of the window's place in sync for override-redirect
/// popup positioning) and mirror the fill mode into `NET_WM_STATE` so
/// the client sees the same maximized/fullscreen state an xdg client
/// would. smithay translates the rect into the client's coordinate
/// space per the Xwayland client scale. Errors mean the window (or
/// Xwayland itself) is going away — nothing sensible to do, so they're
/// logged and swallowed like a configure to a dying xdg surface.
fn push_x11_configure(surface: &X11Surface, rect: Rectangle<i32, Physical>, fill: FillMode) {
    let rect = Rectangle::<i32, Logical>::new(
        Point::from((rect.loc.x, rect.loc.y)),
        Size::from((rect.size.w.max(1), rect.size.h.max(1))),
    );
    // Deliberately a PLAIN configure, never `configure_with_sync`.
    //
    // `_NET_WM_SYNC_REQUEST` blocks the client's buffer commits
    // (`_XWAYLAND_ALLOW_COMMITS=0`) until it acks the sync counter or
    // smithay's 1s timeout expires. That is fine for an interactive
    // resize — the point is to not show a half-repainted window — but
    // this function serves EVERY configure: window map, sibling
    // open/close reflow, fullscreen/maximize toggles, workspace moves,
    // exclusive-zone changes. Routing all of them through the handshake
    // made fullscreen X11 games freeze and flicker: fullscreen configure
    // → commits blocked → nothing on screen → 1s timeout → one frame →
    // repeat. Games advertise the protocol (SDL/Wine do) but present
    // through their own swapchain, so they are exactly the clients that
    // stall the handshake.
    //
    // If the resize polish is wanted back, it has to be scoped to the
    // interactive-drag path alone (thread a flag from
    // `Layout::apply_resize`), never to reflow-driven configures.
    //
    // NET_WM_STATE goes out BEFORE the geometry, and the order is
    // load-bearing. Wine (and Proton's fork) guards its fullscreen
    // transition with a `pending_fullscreen` flag: from the moment it
    // asks for `_NET_WM_STATE_FULLSCREEN` it DISCARDS every
    // ConfigureNotify until a `_NET_WM_STATE` PropertyNotify confirms the
    // state landed. Resizing first therefore throws the fullscreen
    // geometry into that hole — Wine clears the flag on the property that
    // arrives after it, still believing the window has its old, small
    // rect. Its own fullscreen test is "does the window cover a monitor",
    // that stale rect fails it, and Wine answers the state we just set by
    // asking us to remove it again. The compositor grants the
    // unfullscreen, the game re-asserts, and the two trade
    // fullscreen⇄tiled configures ~30 times a second.
    //
    // State first, geometry second: the property clears Wine's guard, and
    // the ConfigureNotify that follows is the one it accepts.
    //
    // Only touch NET_WM_STATE on an actual change — this runs on every
    // reflow, and rewriting the property each time would spam X clients
    // with PropertyNotify events they may react to.
    if surface.is_maximized() != (fill == FillMode::Maximized) {
        let _ = surface.set_maximized(fill == FillMode::Maximized);
    }
    if surface.is_fullscreen() != (fill == FillMode::Fullscreen) {
        let _ = surface.set_fullscreen(fill == FillMode::Fullscreen);
    }
    if let Err(err) = surface.configure(rect) {
        debug!(window = surface.window_id(), %err, "layout: X11 configure failed");
    }
}

/// Configure a tiled window: send the content size, and set the
/// activated + tiled-on-all-sides state set so the
/// client fills the cell exactly. Each `TiledX` flag tells the
/// client "the X edge is shared with the compositor / another
/// window, so don't draw a resize handle or border on that side".
/// A tiling WM cell is tiled on every side.
fn push_configure_for_tile(w: &Window, deco: Deco, area: OutputArea) {
    if w.fill != FillMode::Normal {
        push_configure_filled(w, area.fill(w.fill));
        return;
    }
    match &w.toplevel {
        WindowSurface::Xdg(toplevel) => {
            let size = deco.content_size(w.rect.size);
            toplevel.with_pending_state(|state| {
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
            toplevel.send_configure();
        }
        // X11: one call carries position + size (inside the border) and
        // clears any stale maximized/fullscreen state.
        WindowSurface::X11 { surface, .. } => {
            push_x11_configure(surface, deco.content_rect(w.rect), FillMode::Normal);
        }
    }
    // Logged for BOTH shells. The X11 arm used to return early, so an X11
    // window dropping out of fullscreen back into its tile was the one
    // layout transition that left no trace at all — which is exactly how a
    // ~19 Hz fullscreen↔tiled fight with Wine stayed invisible in session
    // logs (see `Renderer::xwayland_client_scale`).
    debug!(
        surface = ?w.toplevel.wl_surface().id(),
        x = w.rect.loc.x,
        y = w.rect.loc.y,
        w = w.rect.size.w,
        h = w.rect.size.h,
        border = deco.border,
        titlebar = deco.titlebar,
        "layout: tile configure sent",
    );
}

/// Configure a floating (or in-transit) window: send the content
/// size, clear the `Tiled*` flags so the client knows it
/// can resize freely, but still set `Activated` so the focused
/// float doesn't dim or hide its content.
///
/// A **maximized** window comes through here decorated while floating
/// (`Layout::deco_for`), so its cell is the work area and its content is
/// that minus the titlebar — which is why the fill short-circuit below
/// only fires for a `deco`-less fill.
fn push_configure_for_floating(w: &Window, deco: Deco, area: OutputArea) {
    if w.fill != FillMode::Normal && deco == Deco::none() {
        push_configure_filled(w, area.fill(w.fill));
        return;
    }
    let cell = if w.fill == FillMode::Normal {
        w.rect
    } else {
        area.fill(w.fill)
    };
    let size = deco.content_size(cell.size);
    let toplevel = match &w.toplevel {
        WindowSurface::Xdg(toplevel) => toplevel,
        WindowSurface::X11 { surface, .. } => {
            push_x11_configure(surface, deco.content_rect(cell), w.fill);
            return;
        }
    };
    toplevel.with_pending_state(|state| {
        state.size = Some(size);
        state.states.set(xdg_toplevel::State::Activated);
        state.states.unset(xdg_toplevel::State::TiledLeft);
        state.states.unset(xdg_toplevel::State::TiledRight);
        state.states.unset(xdg_toplevel::State::TiledTop);
        state.states.unset(xdg_toplevel::State::TiledBottom);
        // A decorated maximized window still has to be *told* it is
        // maximized — clients draw differently (and some refuse to be
        // dragged) based on this flag, and it is the only thing
        // distinguishing it from a float that happens to fill the work
        // area. Anything else clears both, so unmaximize → float works.
        if w.fill == FillMode::Maximized {
            state.states.set(xdg_toplevel::State::Maximized);
        } else {
            state.states.unset(xdg_toplevel::State::Maximized);
        }
        state.states.unset(xdg_toplevel::State::Fullscreen);
    });
    toplevel.send_configure();
    debug!(
        surface = ?w.toplevel.wl_surface().id(),
        x = w.rect.loc.x,
        y = w.rect.loc.y,
        w = w.rect.size.w,
        h = w.rect.size.h,
        border = deco.border,
        titlebar = deco.titlebar,
        "layout: floating configure sent",
    );
}

/// [`Layout::deco_for`] without the `&self` borrow, for the reflow loops
/// that already hold `&mut self.outputs` and so cannot call a method.
fn deco_for_fill(deco: Deco, mode: LayoutMode, fill: FillMode) -> Deco {
    match fill {
        FillMode::Normal => deco,
        FillMode::Maximized if mode == LayoutMode::Floating => deco,
        _ => Deco::none(),
    }
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

#[cfg(test)]
mod workspace_switch_tests {
    use super::{SlideAction, slide_action};

    /// Shorthand for the common case: a slide `steps` from its origin, still
    /// young enough to redirect.
    fn act(steps: i32, delta: i32) -> SlideAction {
        slide_action(Some((steps, true)), delta)
    }

    /// The bug this exists for. Three quick scroll notches used to start
    /// three slides, each replaced a few milliseconds in, so nothing visibly
    /// moved and you simply arrived three workspaces later. Each notch must
    /// instead re-aim the slide already running, which keeps its origin and
    /// its clock — one continuous motion ending three workspaces out.
    #[test]
    fn a_flick_in_one_direction_keeps_re_aiming_the_same_slide() {
        assert_eq!(act(1, 1), SlideAction::Aim(2));
        assert_eq!(act(2, 1), SlideAction::Aim(3));
        // And the same going the other way.
        assert_eq!(act(-1, -1), SlideAction::Aim(-2));
        assert_eq!(act(-2, -1), SlideAction::Aim(-3));
    }

    /// A bigger jump than one notch — an IPC `focus-workspace` landing
    /// mid-slide — re-aims just the same, as long as it keeps going.
    #[test]
    fn a_larger_jump_the_same_way_also_re_aims() {
        assert_eq!(act(1, 4), SlideAction::Aim(5));
        assert_eq!(act(-1, -4), SlideAction::Aim(-5));
    }

    /// Scrolling back onto the workspace the slide started from leaves *no*
    /// slide. Starting a fresh one instead would snap the half-arrived
    /// workspace to centre before sliding it back out — the very snap this
    /// change exists to remove.
    #[test]
    fn returning_to_the_origin_drops_the_slide_rather_than_restarting() {
        assert_eq!(act(1, -1), SlideAction::Drop);
        assert_eq!(act(-1, 1), SlideAction::Drop);
        assert_eq!(act(3, -3), SlideAction::Drop);
    }

    /// Reversing *past* the origin can't reuse the snapshot: it would have to
    /// travel back the way it came, flipping direction mid-flight.
    #[test]
    fn reversing_past_the_origin_starts_over() {
        assert_eq!(act(1, -2), SlideAction::Fresh);
        assert_eq!(act(2, -5), SlideAction::Fresh);
        assert_eq!(act(-1, 2), SlideAction::Fresh);
    }

    /// With no slide running there is nothing to redirect.
    #[test]
    fn a_switch_from_rest_starts_a_slide() {
        assert_eq!(slide_action(None, 1), SlideAction::Fresh);
        assert_eq!(slide_action(None, -3), SlideAction::Fresh);
    }

    /// A slide too far along must be *replaced*, not reused. Reusing it would
    /// hand the incoming workspace the sliver of time left on the old leg, so
    /// it would appear already almost in place.
    #[test]
    fn a_switch_arriving_late_replaces_the_slide() {
        let old = Some((1, false));
        assert_eq!(slide_action(old, 1), SlideAction::Fresh);
        // Even the cases that would otherwise drop or reverse.
        assert_eq!(slide_action(old, -1), SlideAction::Fresh);
        assert_eq!(slide_action(old, -5), SlideAction::Fresh);
    }

    /// Whatever happens, the decision never leaves a slide aimed at its own
    /// origin — that is the one state the renderer cannot draw.
    #[test]
    fn no_decision_ever_yields_a_zero_aim() {
        for steps in -5..=5 {
            for delta in -5..=5 {
                if steps == 0 || delta == 0 {
                    continue; // neither is reachable: no slide, or no switch
                }
                for fresh_enough in [true, false] {
                    if let SlideAction::Aim(aimed) =
                        slide_action(Some((steps, fresh_enough)), delta)
                    {
                        assert_ne!(aimed, 0, "steps={steps} delta={delta}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod deco_tests {
    use super::{Deco, FillMode, LayoutMode, Point, Rectangle, Size, deco_for_fill};

    /// The whole point of the type: the top edge differs from the other
    /// three, and every conversion has to agree about that.
    #[test]
    fn the_titlebar_only_insets_the_top_edge() {
        let deco = Deco::new(2, 28);
        assert_eq!(deco.top(), 30);
        assert_eq!(deco.content_offset(), Point::from((2, 30)));

        let cell = Rectangle::new(Point::from((100, 200)), Size::from((500, 400)));
        let content = deco.content_rect(cell);
        assert_eq!(content.loc, Point::from((102, 230)));
        // width loses 2 borders, height loses 2 borders AND the titlebar
        assert_eq!(content.size, Size::from((496, 368)));
    }

    /// `cell_size_for` is the inverse of `content_size`, which is what
    /// lets a client's declared size be turned into a cell and back
    /// without drifting.
    #[test]
    fn cell_and_content_sizes_round_trip() {
        for deco in [Deco::none(), Deco::new(1, 0), Deco::new(2, 28), Deco::new(6, 13)] {
            for content in [Size::from((1, 1)), Size::from((800, 600)), Size::from((3840, 2160))] {
                let cell = deco.cell_size_for(content);
                let back = deco.content_size(cell);
                assert_eq!(
                    (back.w, back.h),
                    (content.w, content.h),
                    "{deco:?} lost size round-tripping {content:?} through {cell:?}"
                );
            }
        }
    }

    /// A zero-size configure is one the client cannot render, so a cell
    /// too small for its own decoration must still yield 1x1 rather than
    /// zero or negative.
    #[test]
    fn a_degenerate_cell_never_configures_to_zero() {
        let deco = Deco::new(4, 28);
        let content = deco.content_size(Size::from((2, 2)));
        assert_eq!((content.w, content.h), (1, 1));
    }

    /// A client can commit a window geometry near `i32::MAX`; growing it
    /// into a cell must saturate rather than wrap into a negative rect.
    #[test]
    fn growing_a_huge_content_size_saturates() {
        let deco = Deco::new(2, 28);
        let cell = deco.cell_size_for(Size::from((i32::MAX, i32::MAX)));
        assert_eq!(cell.w, i32::MAX);
        assert_eq!(cell.h, i32::MAX);
    }

    /// Fullscreen owns the output, so it never carries decoration.
    /// Maximized is the interesting one: a stacking WM keeps its
    /// titlebar (you need it to un-maximize), a tiling one never had it.
    #[test]
    fn fill_decides_the_inset_per_mode() {
        let deco = Deco::new(2, 28);
        for mode in [LayoutMode::Tiling, LayoutMode::Floating] {
            assert_eq!(deco_for_fill(deco, mode, FillMode::Normal), deco);
            assert_eq!(
                deco_for_fill(deco, mode, FillMode::Fullscreen),
                Deco::none()
            );
        }
        assert_eq!(
            deco_for_fill(deco, LayoutMode::Floating, FillMode::Maximized),
            deco
        );
        assert_eq!(
            deco_for_fill(deco, LayoutMode::Tiling, FillMode::Maximized),
            Deco::none()
        );
    }

    /// Border-only is exactly what it was before titlebars existed.
    #[test]
    fn a_zero_titlebar_is_the_old_symmetric_border() {
        let deco = Deco::new(3, 0);
        let cell = Rectangle::new(Point::from((0, 0)), Size::from((100, 100)));
        assert_eq!(deco.content_rect(cell).loc, Point::from((3, 3)));
        assert_eq!(deco.content_rect(cell).size, Size::from((94, 94)));
    }
}

#[cfg(test)]
mod resize_tests {
    use super::{MIN_TILE_H, MIN_TILE_W, Point, Rectangle, ResizeEdges, Size, ratio_for_divider};

    /// A divider dropped at the middle of a cell is a 0.5 ratio, and the
    /// cell's absolute position is subtracted out (a second monitor's
    /// tiles start at a large x).
    #[test]
    fn divider_maps_to_ratio_relative_to_the_cell() {
        assert!((ratio_for_divider(1000, 0, 500, MIN_TILE_W) - 0.5).abs() < f32::EPSILON);
        assert!((ratio_for_divider(1000, 1920, 2420, MIN_TILE_W) - 0.5).abs() < f32::EPSILON);
    }

    /// Dragging a divider past either end leaves both cells at least the
    /// minimum wide, so a neighbour can't be squeezed out of existence.
    #[test]
    fn divider_is_clamped_to_keep_both_cells_usable() {
        let squashed_left = ratio_for_divider(1000, 0, -400, MIN_TILE_W);
        let squashed_right = ratio_for_divider(1000, 0, 9999, MIN_TILE_W);
        #[allow(
            clippy::cast_precision_loss,
            reason = "test constants, far inside f32's exact-integer range"
        )]
        let min_ratio = MIN_TILE_W as f32 / 1000.0;
        assert!((squashed_left - min_ratio).abs() < f32::EPSILON);
        assert!((squashed_right - (1.0 - min_ratio)).abs() < f32::EPSILON);
    }

    /// A cell too narrow to hold two minimums must still produce a sane
    /// ratio rather than panicking on an inverted clamp range.
    #[test]
    fn degenerate_cell_does_not_panic() {
        let r = ratio_for_divider(100, 0, 9999, MIN_TILE_W);
        assert!((0.0..=1.0).contains(&r), "ratio out of range: {r}");
        let r = ratio_for_divider(1, 0, 0, MIN_TILE_H);
        assert!((0.0..=1.0).contains(&r), "ratio out of range: {r}");
    }

    /// The press half picks the edge: the drag moves whichever edge the
    /// cursor started nearest, so every tile is resizable from somewhere.
    #[test]
    fn press_half_picks_the_near_edge() {
        let rect = Rectangle::new(Point::<i32, _>::new(100, 100), Size::new(400, 200));
        let edges = |x, y| ResizeEdges::from_press(rect, Point::new(x, y));
        assert_eq!(
            edges(450, 250),
            ResizeEdges {
                right: true,
                bottom: true
            }
        );
        assert_eq!(
            edges(150, 150),
            ResizeEdges {
                right: false,
                bottom: false
            }
        );
        assert_eq!(
            edges(450, 150),
            ResizeEdges {
                right: true,
                bottom: false
            }
        );
        assert_eq!(
            edges(150, 250),
            ResizeEdges {
                right: false,
                bottom: true
            }
        );
    }
}
