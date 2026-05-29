//! Screencast output chooser for Libreland.
//!
//! Invoked by xdg-desktop-portal-wlr as its `chooser_cmd`: it shows a
//! translucent `wlr-layer-shell` overlay on every output, highlights the
//! monitor under the pointer (with its connector name), and on click prints
//! that output's name to stdout — which xdpw matches against its output list
//! (`chooser_type=simple`; the bare name matches via xdpw's pre-0.8.0
//! fallback). Escape, or no selection, prints nothing and exits non-zero,
//! which xdpw treats as "cancelled".
//!
//! Replaces slurp, which crashes on this (and any) compositor that delivers a
//! `wl_pointer.motion` without a usable preceding `enter` — slurp 1.5.0
//! dereferences an unset output with no null-guard. Every event here resolves
//! its surface through [`Picker::overlay_index_for_surface`] and bails on a
//! miss, so an unknown/early event can never panic.

use std::process::ExitCode;

use font8x8::{BASIC_FONTS, UnicodeFonts};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_keyboard, delegate_layer, delegate_output, delegate_pointer,
    delegate_registry, delegate_seat, delegate_shm,
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
    },
    shm::{
        Shm, ShmHandler,
        slot::SlotPool,
    },
};
use wayland_client::{
    Connection, QueueHandle,
    globals::registry_queue_init,
    protocol::{wl_keyboard, wl_output, wl_pointer, wl_seat, wl_shm, wl_surface},
};

// ── Colours ────────────────────────────────────────────────────────────────
//
// `wl_shm` Argb8888 is a packed little-endian `0xAARRGGBB`, so the four bytes
// in memory are [B, G, R, A], and the compositor expects *premultiplied*
// alpha. Constants below are straight (non-premultiplied) ARGB; they're
// premultiplied at write time.

/// A straight (non-premultiplied) `0xAARRGGBB` colour.
#[derive(Clone, Copy)]
struct Color {
    a: u8,
    r: u8,
    g: u8,
    b: u8,
}

impl Color {
    const fn from_u32(argb: u32) -> Self {
        // 0xAARRGGBB big-endian bytes are [A, R, G, B] — no lossy casts.
        let [a, r, g, b] = argb.to_be_bytes();
        Self { a, r, g, b }
    }

    /// The four bytes to store in an Argb8888 buffer: premultiplied B, G, R, A.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "(channel * alpha) / 255 is always <= 255, so the u16 -> u8 cast never truncates"
    )]
    fn premul_bgra(self) -> [u8; 4] {
        let a = u16::from(self.a);
        let m = |c: u8| ((u16::from(c) * a) / 255) as u8;
        [m(self.b), m(self.g), m(self.r), self.a]
    }
}

/// Dim wash over a non-selected output (~50% black).
const DIM: Color = Color::from_u32(0x8000_0000);
/// Brighter wash over the hovered output (~50% azure).
const HOVER: Color = Color::from_u32(0x804A_9EFF);
/// Opaque azure frame around the hovered output.
const ACCENT: Color = Color::from_u32(0xFF4A_9EFF);
/// Label text and its drop shadow.
const TEXT: Color = Color::from_u32(0xFFFF_FFFF);
const SHADOW: Color = Color::from_u32(0xFF00_0000);

const GLYPH: usize = 8;

// ── Drawing (operates on the raw shm canvas; no `self` borrow) ───────────────

/// Write one premultiplied pixel at `(x, y)` if it lies inside the canvas.
fn put(buf: &mut [u8], stride: usize, x: usize, y: usize, px: [u8; 4]) {
    let off = y * stride + x * 4;
    if let Some(slot) = buf.get_mut(off..off + 4) {
        slot.copy_from_slice(&px);
    }
}

/// Fill the whole canvas with one colour (fast path — every pixel identical).
fn fill_all(buf: &mut [u8], color: Color) {
    let px = color.premul_bgra();
    for chunk in buf.chunks_exact_mut(4) {
        chunk.copy_from_slice(&px);
    }
}

/// Draw a `thickness`-px frame just inside the canvas edges.
fn draw_border(buf: &mut [u8], stride: usize, width: usize, height: usize, thickness: usize) {
    let px = ACCENT.premul_bgra();
    for t in 0..thickness {
        for x in 0..width {
            put(buf, stride, x, t, px);
            put(buf, stride, x, height - 1 - t, px);
        }
        for y in 0..height {
            put(buf, stride, t, y, px);
            put(buf, stride, width - 1 - t, y, px);
        }
    }
}

/// Blit one `font8x8` glyph scaled `k`× with its top-left at `(gx, gy)`.
fn draw_glyph(buf: &mut [u8], stride: usize, ch: char, gx: usize, gy: usize, k: usize, px: [u8; 4]) {
    let rows = BASIC_FONTS.get(ch).unwrap_or([0; GLYPH]);
    for (row_idx, row) in rows.iter().enumerate() {
        for col in 0..GLYPH {
            // font8x8 rows are LSB-first: bit `col` is the column from the left.
            if (row >> col) & 1 == 1 {
                let base_x = gx + col * k;
                let base_y = gy + row_idx * k;
                for dy in 0..k {
                    for dx in 0..k {
                        put(buf, stride, base_x + dx, base_y + dy, px);
                    }
                }
            }
        }
    }
}

/// Rendered width of `text` at scale `k` (1px inter-glyph gap, no trailing gap).
fn text_width(text: &str, k: usize) -> usize {
    let n = text.chars().count();
    if n == 0 { 0 } else { n * (GLYPH + 1) * k - k }
}

/// Largest scale in `3..=12` that keeps `text` within ~70% of `width` and
/// ~1/5 of `height`, so a label always fits its monitor.
fn fit_scale(text: &str, width: usize, height: usize) -> usize {
    (3..=12)
        .rev()
        .find(|&k| text_width(text, k) <= width * 7 / 10 && GLYPH * k <= height / 5)
        .unwrap_or(3)
}

/// Draw `text` centred horizontally with its top at `top`, white on a 1px-`k`
/// black drop shadow so it stays legible over any desktop content.
fn draw_label(buf: &mut [u8], stride: usize, width: usize, top: usize, text: &str, k: usize) {
    let start_x = (width.saturating_sub(text_width(text, k))) / 2;
    let advance = (GLYPH + 1) * k;
    let fg = TEXT.premul_bgra();
    let sh = SHADOW.premul_bgra();
    for (i, ch) in text.chars().enumerate() {
        let gx = start_x + i * advance;
        draw_glyph(buf, stride, ch, gx + k, top + k, k, sh);
        draw_glyph(buf, stride, ch, gx, top, k, fg);
    }
}

/// Paint a whole overlay: dim/hover wash, hover frame, and the name +
/// resolution labels stacked in the centre.
fn render(buf: &mut [u8], width: usize, height: usize, hovered: bool, name: &str, res: &str) {
    let stride = width * 4;
    fill_all(buf, if hovered { HOVER } else { DIM });
    if hovered {
        draw_border(buf, stride, width, height, (width / 200).clamp(4, 16));
    }
    let k_name = fit_scale(name, width, height);
    let k_res = (k_name / 2).max(2);
    let gap = 2 * k_res;
    let block_h = GLYPH * k_name + gap + GLYPH * k_res;
    let top = height.saturating_sub(block_h) / 2;
    draw_label(buf, stride, width, top, name, k_name);
    draw_label(buf, stride, width, top + GLYPH * k_name + gap, res, k_res);
}

// ── Client state ─────────────────────────────────────────────────────────────

struct OutputOverlay {
    output: wl_output::WlOutput,
    name: String,
    layer: LayerSurface,
    width: u32,
    height: u32,
    configured: bool,
}

struct Picker {
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    shm: Shm,
    pool: SlotPool,
    compositor: CompositorState,
    layer_shell: LayerShell,
    overlays: Vec<OutputOverlay>,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    pointer: Option<wl_pointer::WlPointer>,
    /// Index of the overlay under the pointer, if any (always `None`-checked).
    hovered: Option<usize>,
    /// Connector name chosen by a click; printed on exit.
    chosen: Option<String>,
    /// Set by a click or Escape to leave the dispatch loop.
    exit: bool,
}

impl Picker {
    /// Map an incoming `wl_surface` to one of our overlays — `None` (never a
    /// panic) for any surface we don't own. This is the null-safety slurp
    /// lacks: events for an unknown/late surface are simply ignored.
    fn overlay_index_for_surface(&self, surface: &wl_surface::WlSurface) -> Option<usize> {
        self.overlays
            .iter()
            .position(|o| o.layer.wl_surface() == surface)
    }

    /// Create one fullscreen Overlay layer surface per known output, bound to
    /// that output. Idempotent (skips outputs we already cover), so it's safe
    /// to call after the initial roundtrip and again on hotplug.
    fn create_overlays(&mut self, qh: &QueueHandle<Self>) {
        let outputs: Vec<_> = self.output_state.outputs().collect();
        for output in outputs {
            if self.overlays.iter().any(|o| o.output == output) {
                continue;
            }
            let Some(info) = self.output_state.info(&output) else {
                continue;
            };
            let name = info.name.clone().unwrap_or_else(|| format!("output-{}", info.id));
            let surface = self.compositor.create_surface(qh);
            let layer = self.layer_shell.create_layer_surface(
                qh,
                surface,
                Layer::Overlay,
                Some("screencast-picker"),
                Some(&output),
            );
            layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
            layer.set_exclusive_zone(-1);
            // Exclusive so Escape reaches us regardless of which monitor's
            // sibling overlay the seat focuses; the compositor latches focus to
            // one and forwards keys there.
            layer.set_keyboard_interactivity(KeyboardInteractivity::Exclusive);
            layer.commit(); // no buffer yet — this elicits the first configure
            let (w, h) = info.logical_size.unwrap_or((0, 0));
            self.overlays.push(OutputOverlay {
                output,
                name,
                layer,
                width: w.max(1).unsigned_abs(),
                height: h.max(1).unsigned_abs(),
                configured: false,
            });
        }
    }

    /// Redraw one overlay from its current size + hover state.
    fn draw(&mut self, idx: usize) {
        let (width, height, hovered, name, res) = {
            let Some(o) = self.overlays.get(idx) else {
                return;
            };
            if !o.configured {
                return;
            }
            (
                o.width as usize,
                o.height as usize,
                self.hovered == Some(idx),
                o.name.clone(),
                format!("{}x{}", o.width, o.height),
            )
        };
        let stride = width * 4;
        let (Ok(w_i), Ok(h_i), Ok(stride_i)) = (
            i32::try_from(width),
            i32::try_from(height),
            i32::try_from(stride),
        ) else {
            return;
        };
        let Ok((buffer, canvas)) =
            self.pool
                .create_buffer(w_i, h_i, stride_i, wl_shm::Format::Argb8888)
        else {
            return;
        };
        render(canvas, width, height, hovered, &name, &res);
        let surface = self.overlays[idx].layer.wl_surface();
        surface.damage_buffer(0, 0, w_i, h_i);
        if buffer.attach_to(surface).is_ok() {
            self.overlays[idx].layer.commit();
        }
    }

    /// Move the highlight to `idx`, redrawing the old and new overlays.
    fn set_hovered(&mut self, idx: Option<usize>) {
        if self.hovered == idx {
            return;
        }
        let previous = self.hovered;
        self.hovered = idx;
        if let Some(p) = previous {
            self.draw(p);
        }
        if let Some(n) = idx {
            self.draw(n);
        }
    }
}

// ── Handlers ─────────────────────────────────────────────────────────────────

impl CompositorHandler for Picker {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: i32,
    ) {
    }
    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }
    // Static overlay — we never request frame callbacks, so this never fires.
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

impl OutputHandler for Picker {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(&mut self, _: &Connection, qh: &QueueHandle<Self>, _: wl_output::WlOutput) {
        self.create_overlays(qh);
    }
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        self.overlays.retain(|o| o.output != output);
        self.hovered = None;
    }
}

impl LayerShellHandler for Picker {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {
        self.exit = true;
    }
    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _: u32,
    ) {
        let Some(idx) = self.overlays.iter().position(|o| &o.layer == layer) else {
            return;
        };
        let (mut w, mut h) = configure.new_size;
        // A zero dimension means "pick your own" — fall back to the logical size.
        if (w == 0 || h == 0)
            && let Some((lw, lh)) = self
                .output_state
                .info(&self.overlays[idx].output)
                .and_then(|i| i.logical_size)
        {
            if w == 0 {
                w = lw.max(1).unsigned_abs();
            }
            if h == 0 {
                h = lh.max(1).unsigned_abs();
            }
        }
        self.overlays[idx].width = w.max(1);
        self.overlays[idx].height = h.max(1);
        self.overlays[idx].configured = true;
        self.draw(idx);
    }
}

impl SeatHandler for Picker {
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

impl KeyboardHandler for Picker {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: &wl_surface::WlSurface,
        _: u32,
        _: &[u32],
        _: &[Keysym],
    ) {
    }
    fn leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: &wl_surface::WlSurface,
        _: u32,
    ) {
    }
    fn press_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        event: KeyEvent,
    ) {
        if event.keysym == Keysym::Escape {
            self.chosen = None;
            self.exit = true;
        }
    }
    // Required by the trait (no default impl); a picker has nothing to repeat.
    fn repeat_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: KeyEvent,
    ) {
    }
    fn release_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: KeyEvent,
    ) {
    }
    fn update_modifiers(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: Modifiers,
        _: RawModifiers,
        _: u32,
    ) {
    }
}

impl PointerHandler for Picker {
    fn pointer_frame(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            // Resolve the surface first and skip anything we don't own — a
            // motion without a preceding enter (slurp's crash) just no-ops.
            let Some(idx) = self.overlay_index_for_surface(&event.surface) else {
                continue;
            };
            match event.kind {
                PointerEventKind::Enter { .. } | PointerEventKind::Motion { .. } => {
                    self.set_hovered(Some(idx));
                }
                PointerEventKind::Leave { .. } => {
                    if self.hovered == Some(idx) {
                        self.set_hovered(None);
                    }
                }
                PointerEventKind::Press { .. } => {
                    self.chosen = Some(self.overlays[idx].name.clone());
                    self.exit = true;
                }
                PointerEventKind::Release { .. } | PointerEventKind::Axis { .. } => {}
            }
        }
    }
}

impl ShmHandler for Picker {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

delegate_compositor!(Picker);
delegate_output!(Picker);
delegate_shm!(Picker);
delegate_seat!(Picker);
delegate_keyboard!(Picker);
delegate_pointer!(Picker);
delegate_layer!(Picker);
delegate_registry!(Picker);

impl ProvidesRegistryState for Picker {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState];
}

fn main() -> ExitCode {
    let Ok(conn) = Connection::connect_to_env() else {
        eprintln!("output-picker: no Wayland display (WAYLAND_DISPLAY unset?)");
        return ExitCode::FAILURE;
    };
    let Ok((globals, mut event_queue)) = registry_queue_init(&conn) else {
        eprintln!("output-picker: failed to initialise the Wayland registry");
        return ExitCode::FAILURE;
    };
    let qh = event_queue.handle();

    let (Ok(compositor), Ok(layer_shell), Ok(shm)) = (
        CompositorState::bind(&globals, &qh),
        LayerShell::bind(&globals, &qh),
        Shm::bind(&globals, &qh),
    ) else {
        eprintln!("output-picker: compositor is missing wl_compositor / layer-shell / wl_shm");
        return ExitCode::FAILURE;
    };
    let Ok(pool) = SlotPool::new(256 * 256 * 4, &shm) else {
        eprintln!("output-picker: failed to create the shm pool");
        return ExitCode::FAILURE;
    };

    let mut picker = Picker {
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &qh),
        output_state: OutputState::new(&globals, &qh),
        shm,
        pool,
        compositor,
        layer_shell,
        overlays: Vec::new(),
        keyboard: None,
        pointer: None,
        hovered: None,
        chosen: None,
        exit: false,
    };

    // One roundtrip so OutputState learns each output's name + logical size.
    if event_queue.roundtrip(&mut picker).is_err() {
        return ExitCode::FAILURE;
    }
    picker.create_overlays(&qh);

    while !picker.exit {
        if event_queue.blocking_dispatch(&mut picker).is_err() {
            break;
        }
    }

    // Returning from main flushes stdout; a click prints the connector name
    // (xdpw selects it), Escape/cancel prints nothing and exits non-zero.
    match picker.chosen {
        Some(name) => {
            println!("{name}");
            ExitCode::SUCCESS
        }
        None => ExitCode::FAILURE,
    }
}
