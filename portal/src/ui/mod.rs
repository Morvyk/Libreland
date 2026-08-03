//! The portal's own windowing layer.
//!
//! Every dialog this backend shows is drawn here — there is no toolkit under
//! it. That is the whole point of the exercise: a portal that pulls in GTK
//! drags a second widget stack, a second theme engine and a second set of
//! settings daemons into a session that already has a compositor which can
//! draw. What the dialogs actually need is a surface, a font and hit-testing,
//! which is what [`draw`], [`text`] and the runtime below provide.
//!
//! Two surface shapes cover everything:
//!
//! * [`Mode::Dialog`] — one `xdg_toplevel`, sized exactly by the screen and
//!   pinned with equal min/max size. Libreland floats fixed-size toplevels
//!   (its `float_if_dialog` heuristic), so a file chooser lands centred over
//!   the tiling instead of being wedged into it, and other compositors treat
//!   the same hint the same way.
//! * [`Mode::Overlay`] — one `wlr-layer-shell` surface per output, for the
//!   screencast picker, the region selector and the colour picker.
//!
//! The dispatch loop is deliberately poll-based rather than calloop-driven:
//! it has to wake on a [`Cancel`] flipped from the D-Bus thread (the frontend
//! closing the request mid-dialog), and it has to synthesize key repeat, both
//! of which want a timeout on every iteration.

pub mod appgrid;
pub mod draw;
pub mod filechooser;
pub mod picker;
pub mod prompt;
pub mod widgets;

/// Font discovery + rasterization, shared with the compositor's titlebars
/// (`libreland-text`). Re-exported under the name it had while it lived here,
/// so `text::Fonts` keeps resolving for everything under `ui::`.
pub use libreland_text as text;

use std::os::fd::AsFd as _;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_keyboard, delegate_layer, delegate_output, delegate_pointer,
    delegate_registry, delegate_seat, delegate_shm, delegate_xdg_shell, delegate_xdg_window,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        Capability, SeatHandler, SeatState,
        keyboard::{KeyEvent, KeyboardHandler, Keysym, Modifiers, RawModifiers},
        pointer::{PointerEvent, PointerEventKind, PointerHandler},
    },
    shell::{
        WaylandSurface,
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
        xdg::{
            XdgShell,
            window::{Window, WindowConfigure, WindowDecorations, WindowHandler},
        },
    },
    shm::{Shm, ShmHandler, slot::SlotPool},
};
use wayland_client::{
    Connection, QueueHandle,
    globals::registry_queue_init,
    protocol::{wl_keyboard, wl_output, wl_pointer, wl_seat, wl_shm, wl_surface},
};

use crate::portals::Cancel;
use draw::{Canvas, Theme};
use text::Fonts;

// ── The screen interface ───────────────────────────────────────────────────

/// A key press, already resolved from keysym + UTF-8 into something a view can
/// match on.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Key {
    Escape,
    Enter,
    Tab,
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    /// A printable character (already shifted/composed by xkb).
    Char(char),
    Other,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Mods {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

/// One input event, in **logical** surface coordinates.
#[derive(Clone, Debug)]
pub enum Input {
    Motion {
        x: f64,
        y: f64,
    },
    /// A press of any button. Dialogs treat every button alike, so which one
    /// it was isn't carried.
    Press {
        x: f64,
        y: f64,
        mods: Mods,
    },
    /// Button release. Position isn't carried: nothing in these dialogs
    /// acts on where a release landed, only on where the press did.
    Release,
    /// Wheel/touchpad scroll. Positive `dy` scrolls content down.
    Scroll {
        dy: f64,
    },
    /// The pointer left the surface — drop any hover highlight.
    Leave,
    Key {
        key: Key,
        mods: Mods,
    },
}

/// What the runtime should do after handing a screen an event.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Flow {
    /// Nothing changed.
    Idle,
    /// Repaint the surface the event came from.
    Redraw,
    /// Repaint every surface (overlay hover moving between monitors).
    RedrawAll,
    /// The interaction is over; tear the surfaces down and return.
    Done,
}

/// One connected output, as an overlay screen sees it.
#[derive(Clone, Debug)]
pub struct OutputDesc {
    /// Connector name (`DP-1`), which is also what the compositor's IPC and
    /// the screencopy capture path key off.
    pub name: String,
    /// Logical size and position in the compositor's global space.
    pub width: i32,
    pub height: i32,
    pub x: i32,
    pub y: i32,
    pub scale: i32,
    /// Physical mode size, for the "3840x2160" subtitle in the picker.
    pub mode_width: i32,
    pub mode_height: i32,
}

/// Everything a screen needs to draw itself that isn't the canvas.
pub struct Ctx {
    pub fonts: Arc<Fonts>,
    pub theme: Theme,
}

/// A drawable, interactive surface set: the portal's equivalent of a window
/// class. Implementors own their result and expose it after [`run`] returns.
pub trait Screen {
    /// Window title (dialog mode only).
    fn title(&self) -> String {
        String::new()
    }

    /// Preferred logical size (dialog mode only).
    fn size(&self) -> (i32, i32) {
        (720, 480)
    }

    /// Called once before the first frame in overlay mode, with every output
    /// the compositor advertises. Surface indices passed to [`Screen::render`]
    /// and [`Screen::input`] index this slice.
    fn outputs(&mut self, _outputs: &[OutputDesc]) {}

    fn render(&mut self, surface: usize, canvas: &mut Canvas<'_>, ctx: &Ctx);

    fn input(&mut self, surface: usize, event: &Input, ctx: &Ctx) -> Flow;

    /// Called on every idle tick (~33 ms). Drives the text caret blink.
    fn tick(&mut self) -> Flow {
        Flow::Idle
    }
}

/// How to put a screen on the display.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// A floating dialog window.
    Dialog,
    /// A fullscreen overlay on every output, above everything else.
    Overlay,
}

// ── Shared font/theme state ────────────────────────────────────────────────

static FONTS: OnceLock<Option<Arc<Fonts>>> = OnceLock::new();

/// Load (once) the UI faces shared by every dialog. `None` when the system has
/// no usable font — callers still draw, just without labels.
pub fn fonts() -> Option<Arc<Fonts>> {
    FONTS.get_or_init(|| Fonts::load().map(Arc::new)).clone()
}

/// Build the drawing context, picking the palette from the appearance the
/// Settings portal is currently reporting to apps.
pub fn ctx() -> Option<Ctx> {
    Some(Ctx {
        fonts: fonts()?,
        theme: if crate::portals::settings::prefers_dark() {
            Theme::DARK
        } else {
            Theme::LIGHT
        },
    })
}

// ── Runtime ────────────────────────────────────────────────────────────────

/// One surface the runtime manages.
struct Surf {
    wl: wl_surface::WlSurface,
    role: Role,
    /// Logical size; `0` until the first configure.
    width: i32,
    height: i32,
    scale: i32,
    configured: bool,
    dirty: bool,
}

enum Role {
    Dialog(Window),
    Overlay(LayerSurface),
}

impl Role {
    fn commit(&self) {
        match self {
            Role::Dialog(w) => w.commit(),
            Role::Overlay(l) => l.commit(),
        }
    }
}

struct App<S: Screen + 'static> {
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    shm: Shm,
    pool: SlotPool,
    compositor: CompositorState,
    xdg: Option<XdgShell>,
    layer_shell: Option<LayerShell>,

    mode: Mode,
    screen: S,
    ctx: Ctx,
    outputs: Vec<OutputDesc>,
    /// Parallel to `outputs` in overlay mode; a single entry in dialog mode.
    surfaces: Vec<Surf>,
    wl_outputs: Vec<wl_output::WlOutput>,

    keyboard: Option<wl_keyboard::WlKeyboard>,
    pointer: Option<wl_pointer::WlPointer>,
    mods: Mods,
    /// Surface the pointer is over, for routing motion/press.
    pointer_on: Option<usize>,
    /// Surface with keyboard focus; keys route here (or to surface 0).
    key_on: Option<usize>,
    /// Held key, when it repeats, and what it is — synthesized locally
    /// because we dispatch by hand instead of through calloop.
    repeat: Option<(Key, Instant)>,

    done: bool,
}

/// Keys worth repeating: navigation and deletion. Repeating a character key
/// would need xkb's own rate/delay and matters far less in a dialog.
fn repeats(key: &Key) -> bool {
    matches!(
        key,
        Key::Backspace
            | Key::Delete
            | Key::Left
            | Key::Right
            | Key::Up
            | Key::Down
            | Key::PageUp
            | Key::PageDown
    )
}

const REPEAT_DELAY: Duration = Duration::from_millis(400);
const REPEAT_INTERVAL: Duration = Duration::from_millis(33);
const TICK: Duration = Duration::from_millis(33);

fn key_from(event: &KeyEvent) -> Key {
    match event.keysym {
        Keysym::Escape => Key::Escape,
        Keysym::Return | Keysym::KP_Enter | Keysym::ISO_Enter => Key::Enter,
        Keysym::Tab | Keysym::ISO_Left_Tab => Key::Tab,
        Keysym::BackSpace => Key::Backspace,
        Keysym::Delete | Keysym::KP_Delete => Key::Delete,
        Keysym::Left | Keysym::KP_Left => Key::Left,
        Keysym::Right | Keysym::KP_Right => Key::Right,
        Keysym::Up | Keysym::KP_Up => Key::Up,
        Keysym::Down | Keysym::KP_Down => Key::Down,
        Keysym::Home | Keysym::KP_Home => Key::Home,
        Keysym::End | Keysym::KP_End => Key::End,
        Keysym::Page_Up | Keysym::KP_Page_Up => Key::PageUp,
        Keysym::Page_Down | Keysym::KP_Page_Down => Key::PageDown,
        _ => event
            .utf8
            .as_deref()
            .and_then(|s| s.chars().next())
            // Control characters aren't text; a bare Ctrl+A arrives as \x01.
            .filter(|c| !c.is_control())
            .map_or(Key::Other, Key::Char),
    }
}

impl<S: Screen + 'static> App<S> {
    /// Apply a [`Flow`] to the surface it came from.
    fn apply(&mut self, surface: usize, flow: Flow) {
        match flow {
            Flow::Idle => {}
            Flow::Redraw => {
                if let Some(s) = self.surfaces.get_mut(surface) {
                    s.dirty = true;
                }
            }
            Flow::RedrawAll => {
                for s in &mut self.surfaces {
                    s.dirty = true;
                }
            }
            Flow::Done => self.done = true,
        }
    }

    fn index_of(&self, surface: &wl_surface::WlSurface) -> Option<usize> {
        self.surfaces.iter().position(|s| &s.wl == surface)
    }

    /// Repaint every dirty, configured surface.
    fn flush_frames(&mut self) {
        for idx in 0..self.surfaces.len() {
            let (width, height, scale) = {
                let s = &self.surfaces[idx];
                if !s.dirty || !s.configured || s.width <= 0 || s.height <= 0 {
                    continue;
                }
                (s.width, s.height, s.scale)
            };
            let (dev_w, dev_h) = (width * scale, height * scale);
            let stride = dev_w * 4;
            let Ok((buffer, canvas_buf)) =
                self.pool
                    .create_buffer(dev_w, dev_h, stride, wl_shm::Format::Argb8888)
            else {
                continue;
            };
            let mut canvas = Canvas::new(canvas_buf, dev_w, dev_h, stride, scale);
            self.screen.render(idx, &mut canvas, &self.ctx);

            let surf = &mut self.surfaces[idx];
            surf.wl.set_buffer_scale(scale);
            surf.wl.damage_buffer(0, 0, dev_w, dev_h);
            if buffer.attach_to(&surf.wl).is_ok() {
                surf.role.commit();
                surf.dirty = false;
            }
        }
    }

    /// Snapshot the compositor's outputs into [`OutputDesc`]s, sorted by
    /// position so indices are stable and left-to-right.
    fn collect_outputs(&mut self) {
        let mut found: Vec<(wl_output::WlOutput, OutputDesc)> = Vec::new();
        for output in self.output_state.outputs() {
            let Some(info) = self.output_state.info(&output) else {
                continue;
            };
            let (lw, lh) = info.logical_size.unwrap_or_else(|| {
                info.modes
                    .iter()
                    .find(|m| m.current)
                    .map_or((0, 0), |m| m.dimensions)
            });
            let (px, py) = info.logical_position.unwrap_or((0, 0));
            let mode = info
                .modes
                .iter()
                .find(|m| m.current)
                .map_or((lw, lh), |m| m.dimensions);
            found.push((
                output.clone(),
                OutputDesc {
                    name: info
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("output-{}", info.id)),
                    width: lw.max(1),
                    height: lh.max(1),
                    x: px,
                    y: py,
                    scale: info.scale_factor.max(1),
                    mode_width: mode.0,
                    mode_height: mode.1,
                },
            ));
        }
        found.sort_by_key(|(_, d)| (d.x, d.y));
        self.wl_outputs = found.iter().map(|(o, _)| o.clone()).collect();
        self.outputs = found.into_iter().map(|(_, d)| d).collect();
    }
}

// ── sctk handlers ──────────────────────────────────────────────────────────

impl<S: Screen + 'static> CompositorHandler for App<S> {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        scale: i32,
    ) {
        if let Some(idx) = self.index_of(surface)
            && let Some(s) = self.surfaces.get_mut(idx)
        {
            s.scale = scale.max(1);
            s.dirty = true;
        }
    }
    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }
    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {}
    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
}

impl<S: Screen + 'static> OutputHandler for App<S> {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {
        // An overlay whose monitor vanished mid-pick can't finish meaningfully.
        if self.mode == Mode::Overlay {
            self.done = true;
        }
    }
}

impl<S: Screen + 'static> WindowHandler for App<S> {
    fn request_close(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &Window) {
        tracing::debug!("the compositor asked the dialog to close");
        self.done = true;
    }

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        window: &Window,
        configure: WindowConfigure,
        _: u32,
    ) {
        let preferred = self.screen.size();
        let Some(idx) = self
            .surfaces
            .iter()
            .position(|s| matches!(&s.role, Role::Dialog(w) if w == window))
        else {
            return;
        };
        let surf = &mut self.surfaces[idx];
        // A zero dimension means "you decide" — which, for a dialog we pinned
        // to a fixed size, is always our preferred size.
        surf.width = configure.new_size.0.map_or(preferred.0, |w| {
            i32::try_from(w.get()).unwrap_or(preferred.0)
        });
        surf.height = configure.new_size.1.map_or(preferred.1, |h| {
            i32::try_from(h.get()).unwrap_or(preferred.1)
        });
        surf.configured = true;
        surf.dirty = true;
    }
}

impl<S: Screen + 'static> LayerShellHandler for App<S> {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {
        self.done = true;
    }

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _: u32,
    ) {
        let Some(idx) = self
            .surfaces
            .iter()
            .position(|s| matches!(&s.role, Role::Overlay(l) if l == layer))
        else {
            return;
        };
        let fallback = self
            .outputs
            .get(idx)
            .map_or((1, 1), |o| (o.width, o.height));
        let (w, h) = configure.new_size;
        let surf = &mut self.surfaces[idx];
        surf.width = if w == 0 {
            fallback.0
        } else {
            i32::try_from(w).unwrap_or(fallback.0)
        };
        surf.height = if h == 0 {
            fallback.1
        } else {
            i32::try_from(h).unwrap_or(fallback.1)
        };
        surf.configured = true;
        surf.dirty = true;
    }
}

impl<S: Screen + 'static> SeatHandler for App<S> {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }
    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
    fn new_capability(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        cap: Capability,
    ) {
        if cap == Capability::Keyboard && self.keyboard.is_none() {
            self.keyboard = self.seat_state.get_keyboard(qh, &seat, None).ok();
        }
        if cap == Capability::Pointer && self.pointer.is_none() {
            self.pointer = self.seat_state.get_pointer(qh, &seat).ok();
        }
    }
    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        cap: Capability,
    ) {
        if cap == Capability::Keyboard
            && let Some(k) = self.keyboard.take()
        {
            k.release();
        }
        if cap == Capability::Pointer
            && let Some(p) = self.pointer.take()
        {
            p.release();
        }
    }
    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl<S: Screen + 'static> KeyboardHandler for App<S> {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        _: u32,
        _: &[u32],
        _: &[Keysym],
    ) {
        self.key_on = self.index_of(surface);
    }

    fn leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: &wl_surface::WlSurface,
        _: u32,
    ) {
        self.key_on = None;
        self.repeat = None;
    }

    fn press_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        event: KeyEvent,
    ) {
        let key = key_from(&event);
        if repeats(&key) {
            self.repeat = Some((key.clone(), Instant::now() + REPEAT_DELAY));
        }
        let target = self.key_on.unwrap_or(0);
        let flow = self.screen.input(
            target,
            &Input::Key {
                key,
                mods: self.mods,
            },
            &self.ctx,
        );
        self.apply(target, flow);
    }

    fn release_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        event: KeyEvent,
    ) {
        if self
            .repeat
            .as_ref()
            .is_some_and(|(held, _)| *held == key_from(&event))
        {
            self.repeat = None;
        }
    }

    fn repeat_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: KeyEvent,
    ) {
        // Repeat is synthesized in the dispatch loop (we don't run the
        // calloop-based keyboard), so the compositor's own repeats are unused.
    }

    fn update_modifiers(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        modifiers: Modifiers,
        _: RawModifiers,
        _: u32,
    ) {
        self.mods = Mods {
            ctrl: modifiers.ctrl,
            shift: modifiers.shift,
            alt: modifiers.alt,
        };
    }
}

impl<S: Screen + 'static> PointerHandler for App<S> {
    fn pointer_frame(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            // Resolve the surface first; an event for something we don't own
            // (or a motion with no preceding enter) is dropped, never a panic.
            let Some(idx) = self.index_of(&event.surface) else {
                continue;
            };
            let (x, y) = event.position;
            let input = match event.kind {
                PointerEventKind::Enter { .. } | PointerEventKind::Motion { .. } => {
                    self.pointer_on = Some(idx);
                    Input::Motion { x, y }
                }
                PointerEventKind::Leave { .. } => {
                    if self.pointer_on == Some(idx) {
                        self.pointer_on = None;
                    }
                    Input::Leave
                }
                // Pointer events don't carry modifier state on the wire;
                // pair the press with what the keyboard last reported, so
                // Ctrl/Shift-click selection works.
                PointerEventKind::Press { .. } => Input::Press {
                    x,
                    y,
                    mods: self.mods,
                },
                PointerEventKind::Release { .. } => Input::Release,
                PointerEventKind::Axis { vertical, .. } => {
                    // Prefer the smooth (touchpad) value; a wheel only fills
                    // in discrete steps, scaled here to a comfortable glide.
                    let dy = if vertical.absolute.abs() > f64::EPSILON {
                        vertical.absolute
                    } else {
                        f64::from(vertical.discrete) * 40.0
                    };
                    Input::Scroll { dy }
                }
            };
            let flow = self.screen.input(idx, &input, &self.ctx);
            self.apply(idx, flow);
        }
    }
}

impl<S: Screen + 'static> ShmHandler for App<S> {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

delegate_compositor!(@<S: Screen + 'static> App<S>);
delegate_output!(@<S: Screen + 'static> App<S>);
delegate_shm!(@<S: Screen + 'static> App<S>);
delegate_seat!(@<S: Screen + 'static> App<S>);
delegate_keyboard!(@<S: Screen + 'static> App<S>);
delegate_pointer!(@<S: Screen + 'static> App<S>);
delegate_layer!(@<S: Screen + 'static> App<S>);
delegate_xdg_shell!(@<S: Screen + 'static> App<S>);
delegate_xdg_window!(@<S: Screen + 'static> App<S>);
delegate_registry!(@<S: Screen + 'static> App<S>);

impl<S: Screen + 'static> ProvidesRegistryState for App<S> {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState];
}

// ── Entry point ────────────────────────────────────────────────────────────

/// Show a dialog from async code, returning the screen once the user is done
/// with it.
///
/// The Wayland loop is blocking, so it goes on a blocking task; the D-Bus
/// runtime keeps serving other requests (and this request's `Close`) while the
/// dialog is up.
pub async fn dialog<S: Screen + Send + 'static>(
    screen: S,
    cancel: std::sync::Arc<Cancel>,
) -> anyhow::Result<S> {
    tokio::task::spawn_blocking(move || run(Mode::Dialog, screen, &cancel)).await?
}

/// As [`dialog`], for a fullscreen picker.
pub async fn overlay<S: Screen + Send + 'static>(
    screen: S,
    cancel: std::sync::Arc<Cancel>,
) -> anyhow::Result<S> {
    tokio::task::spawn_blocking(move || run(Mode::Overlay, screen, &cancel)).await?
}

/// Show `screen` and pump it until it finishes, the compositor closes it, or
/// `cancel` is tripped by the frontend.
///
/// Blocking: callers run this on a blocking task, never on the D-Bus runtime.
#[allow(
    clippy::too_many_lines,
    reason = "surface construction differs per mode and the dispatch loop reads as one sequence; splitting it would thread a dozen locals through helpers"
)]
pub fn run<S: Screen + 'static>(mode: Mode, screen: S, cancel: &Cancel) -> anyhow::Result<S> {
    let ctx = ctx().ok_or_else(|| {
        anyhow::anyhow!("no usable font found — install a TTF/OTF font (e.g. ttf-dejavu)")
    })?;
    let conn =
        Connection::connect_to_env().map_err(|e| anyhow::anyhow!("no Wayland display: {e}"))?;
    let (globals, mut queue) = registry_queue_init(&conn)?;
    let qh = queue.handle();

    let compositor =
        CompositorState::bind(&globals, &qh).map_err(|e| anyhow::anyhow!("wl_compositor: {e}"))?;
    let shm = Shm::bind(&globals, &qh).map_err(|e| anyhow::anyhow!("wl_shm: {e}"))?;
    let pool = SlotPool::new(1024 * 1024, &shm)?;
    let xdg = XdgShell::bind(&globals, &qh).ok();
    let layer_shell = LayerShell::bind(&globals, &qh).ok();

    let mut app = App {
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &qh),
        output_state: OutputState::new(&globals, &qh),
        shm,
        pool,
        compositor,
        xdg,
        layer_shell,
        mode,
        screen,
        ctx,
        outputs: Vec::new(),
        surfaces: Vec::new(),
        wl_outputs: Vec::new(),
        keyboard: None,
        pointer: None,
        mods: Mods::default(),
        pointer_on: None,
        key_on: None,
        repeat: None,
        done: false,
    };

    // One roundtrip so outputs carry their names, sizes and scales.
    queue.roundtrip(&mut app)?;
    app.collect_outputs();

    match mode {
        Mode::Dialog => {
            let xdg = app
                .xdg
                .take()
                .ok_or_else(|| anyhow::anyhow!("compositor has no xdg_wm_base"))?;
            let (w, h) = app.screen.size();
            let surface = app.compositor.create_surface(&qh);
            let window = xdg.create_window(surface.clone(), WindowDecorations::RequestClient, &qh);
            window.set_title(app.screen.title());
            // Matching the frontend's own app id keeps the dialog attributable
            // to the portal in window lists and rules.
            window.set_app_id(crate::portals::BUS_NAME.to_string());
            // Equal min and max is the "this is a dialog, float me" signal
            // (Libreland's `float_if_dialog`), and it keeps our fixed layout
            // from being stretched into a tiled cell.
            window.set_min_size(Some((w.unsigned_abs(), h.unsigned_abs())));
            window.set_max_size(Some((w.unsigned_abs(), h.unsigned_abs())));
            window.commit();
            app.surfaces.push(Surf {
                wl: surface,
                role: Role::Dialog(window),
                width: w,
                height: h,
                scale: 1,
                configured: false,
                dirty: true,
            });
        }
        Mode::Overlay => {
            let shell = app
                .layer_shell
                .take()
                .ok_or_else(|| anyhow::anyhow!("compositor has no wlr-layer-shell"))?;
            let outputs = app.outputs.clone();
            app.screen.outputs(&outputs);
            for (idx, desc) in outputs.iter().enumerate() {
                let Some(output) = app.wl_outputs.get(idx) else {
                    continue;
                };
                let surface = app.compositor.create_surface(&qh);
                let layer = shell.create_layer_surface(
                    &qh,
                    surface.clone(),
                    Layer::Overlay,
                    Some("libreland-portal"),
                    Some(output),
                );
                layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
                layer.set_exclusive_zone(-1);
                // Exclusive keyboard focus so Escape cancels regardless of
                // which monitor's sibling overlay the seat happens to focus.
                layer.set_keyboard_interactivity(KeyboardInteractivity::Exclusive);
                layer.commit();
                app.surfaces.push(Surf {
                    wl: surface,
                    role: Role::Overlay(layer),
                    width: desc.width,
                    height: desc.height,
                    scale: desc.scale,
                    configured: false,
                    dirty: true,
                });
            }
            if app.surfaces.is_empty() {
                anyhow::bail!("no outputs to show the overlay on");
            }
        }
    }

    let mut next_tick = Instant::now() + TICK;
    while !app.done && !cancel.is_cancelled() {
        app.flush_frames();
        queue.flush()?;

        // Block until the compositor has something for us, the next tick is
        // due, or the repeat timer fires — whichever is soonest. The timeout
        // is what lets a `Request.Close()` from the D-Bus thread take effect
        // on a dialog nobody is touching.
        let now = Instant::now();
        let mut wake = next_tick.max(now);
        if let Some((_, due)) = &app.repeat {
            wake = wake.min((*due).max(now));
        }
        let timeout = wake.saturating_duration_since(now);

        if let Some(guard) = queue.prepare_read() {
            let fd = conn.as_fd();
            let mut poll_fds = [nix::poll::PollFd::new(fd, nix::poll::PollFlags::POLLIN)];
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the timeout is at most REPEAT/TICK-sized, far under u16::MAX ms"
            )]
            let ms = timeout.as_millis().min(1000) as u16;
            match nix::poll::poll(&mut poll_fds, nix::poll::PollTimeout::from(ms)) {
                Ok(_) if poll_fds[0].revents().is_none_or(|r| r.is_empty()) => drop(guard),
                Ok(_) => {
                    // `read()` reporting EAGAIN means another wakeup drained
                    // the socket, or the bytes that arrived don't complete a
                    // message yet. Both are ordinary; only a real protocol or
                    // I/O failure ends the dialog.
                    match guard.read() {
                        Ok(_) => {}
                        Err(wayland_client::backend::WaylandError::Io(err))
                            if err.kind() == std::io::ErrorKind::WouldBlock => {}
                        Err(err) => {
                            tracing::error!(%err, "wayland read failed; closing the dialog");
                            break;
                        }
                    }
                }
                // A signal during poll is not an error worth ending a dialog on.
                Err(nix::errno::Errno::EINTR) => drop(guard),
                Err(err) => return Err(err.into()),
            }
        }
        if let Err(err) = queue.dispatch_pending(&mut app) {
            tracing::error!(%err, "wayland dispatch failed; closing the dialog");
            break;
        }

        let now = Instant::now();
        if now >= next_tick {
            next_tick = now + TICK;
            let flow = app.screen.tick();
            let target = app.key_on.unwrap_or(0);
            app.apply(target, flow);
        }
        if let Some((key, due)) = app.repeat.clone()
            && now >= due
        {
            app.repeat = Some((key.clone(), now + REPEAT_INTERVAL));
            let target = app.key_on.unwrap_or(0);
            let flow = app.screen.input(
                target,
                &Input::Key {
                    key,
                    mods: app.mods,
                },
                &app.ctx,
            );
            app.apply(target, flow);
        }
    }

    // Drop the surfaces before returning so the dialog disappears the moment
    // the portal answers, not whenever the connection happens to be collected.
    app.surfaces.clear();
    let _ = queue.flush();
    Ok(app.screen)
}
