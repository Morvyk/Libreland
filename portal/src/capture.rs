//! Screen capture over `zwlr_screencopy_v1`.
//!
//! One client, two consumers: [`Screenshot`](crate::portals::screenshot) takes
//! single frames and encodes PNGs, [`ScreenCast`](crate::portals::screencast)
//! pumps frames into `PipeWire`. Both want the same three things — enumerate
//! outputs, negotiate a buffer, ask the compositor to fill it — so they share
//! this module and the connection type below.
//!
//! Two buffer paths:
//!
//! * **dmabuf** ([`Target::Dmabuf`]) — the compositor blits straight into a
//!   GPU buffer we allocated with gbm and imported through
//!   `zwp_linux_dmabuf_v1`. Nothing crosses the CPU, which is what makes
//!   sharing a 4K display at a high refresh rate viable at all.
//! * **shm** ([`Target::Shm`]) — a memfd the compositor memcpys into. Always
//!   available, always correct, and the only option when the consumer won't
//!   take dmabufs (or when there's no render node to allocate from).
//!
//! Screenshots always use shm: they need the pixels on the CPU anyway to
//! encode them.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel geometry: every value here is a surface- or image-sized non-negative integer, and the conversions between i32/u32/usize/f32 are all inside that range. Checked conversions at each site would be noise around arithmetic that cannot overflow."
)]

use std::os::fd::{AsFd, AsRawFd as _, BorrowedFd, OwnedFd};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use wayland_client::backend::ObjectId;
use wayland_client::protocol::{wl_buffer, wl_output, wl_registry, wl_shm, wl_shm_pool};
use wayland_client::{Connection, Dispatch, EventQueue, Proxy as _, QueueHandle, WEnum};
use wayland_protocols::wp::linux_dmabuf::zv1::client::{
    zwp_linux_buffer_params_v1, zwp_linux_dmabuf_v1,
};
use wayland_protocols::xdg::xdg_output::zv1::client::{zxdg_output_manager_v1, zxdg_output_v1};
use wayland_protocols_wlr::screencopy::v1::client::{
    zwlr_screencopy_frame_v1, zwlr_screencopy_manager_v1,
};

/// A captured frame on the CPU: tightly-described BGRA/BGRX bytes, top row
/// first (any compositor-reported flip is undone on the way out).
#[derive(Clone)]
pub struct Frame {
    pub width: i32,
    pub height: i32,
    pub stride: usize,
    pub data: Vec<u8>,
}

impl Frame {
    /// Crop to `rect` (in this frame's pixels), clamped to its bounds.
    pub fn crop(&self, rect: crate::ui::draw::Rect) -> Self {
        let x0 = rect.x.clamp(0, self.width);
        let y0 = rect.y.clamp(0, self.height);
        let x1 = (rect.x + rect.w).clamp(x0, self.width);
        let y1 = (rect.y + rect.h).clamp(y0, self.height);
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let mut data = Vec::with_capacity(w * h * 4);
        for row in 0..h {
            let start = (y0 as usize + row) * self.stride + x0 as usize * 4;
            if let Some(slice) = self.data.get(start..start + w * 4) {
                data.extend_from_slice(slice);
            }
        }
        Self {
            width: x1 - x0,
            height: y1 - y0,
            stride: w * 4,
            data,
        }
    }

    /// Encode as a PNG at `path`, converting BGRA to the RGBA the encoder
    /// wants. Alpha is forced opaque: a screenshot of a translucent window
    /// should look like what was on screen, not like a hole.
    pub fn write_png(&self, path: &PathBuf) -> anyhow::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut encoder = png::Encoder::new(
            std::io::BufWriter::new(file),
            self.width.unsigned_abs(),
            self.height.unsigned_abs(),
        );
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        let mut rgba = Vec::with_capacity(self.width as usize * self.height as usize * 4);
        for row in 0..self.height as usize {
            let start = row * self.stride;
            let Some(line) = self.data.get(start..start + self.width as usize * 4) else {
                break;
            };
            for px in line.chunks_exact(4) {
                rgba.extend_from_slice(&[px[2], px[1], px[0], 255]);
            }
        }
        writer.write_image_data(&rgba)?;
        Ok(())
    }
}

/// One output, as the capture side sees it.
#[derive(Clone, Debug)]
pub struct Output {
    pub name: String,
    /// Current mode, in physical pixels — the size a capture comes back at.
    pub width: i32,
    pub height: i32,
    /// Position in the compositor's global layout, for stitching a
    /// multi-monitor screenshot back together. **Logical**, so it is not
    /// in the same space as `width`/`height` above — see `logical_width`.
    pub x: i32,
    pub y: i32,
    pub refresh_mhz: i32,
    /// `wl_output.scale`, which is an integer and therefore rounded up on a
    /// fractionally-scaled output (1.5 arrives as 2). Good enough to pick a
    /// cursor size, useless for geometry — use the logical size below.
    pub scale: i32,
    /// Size in the compositor's global layout, from `xdg_output`. This is
    /// the space `x`/`y` are in, and it is *not* `width`/`height` unless the
    /// output is at scale 1: a 4K at scale 1.5 is 3840x2160 physical and
    /// 2560x1440 logical. Zero if the compositor has no `xdg_output`
    /// support, in which case callers fall back to the physical size.
    pub logical_width: i32,
    pub logical_height: i32,
}

impl Output {
    /// Logical size, falling back to the physical one when `xdg_output` is
    /// unavailable (which is also the correct answer at scale 1).
    #[must_use]
    pub const fn logical_size(&self) -> (i32, i32) {
        if self.logical_width > 0 && self.logical_height > 0 {
            (self.logical_width, self.logical_height)
        } else {
            (self.width, self.height)
        }
    }
}

/// What the compositor is willing to give us for one output.
#[derive(Clone, Copy, Debug)]
pub struct BufferSpec {
    /// `wl_shm` format for the CPU path.
    pub shm_format: wl_shm::Format,
    pub shm_stride: u32,
    /// DRM fourcc for the dmabuf path, when the compositor offers one.
    pub drm_format: Option<u32>,
    pub width: u32,
    pub height: u32,
}

/// One dma-buf plane, as [`Capturer::import_dmabuf`] needs it described.
pub struct DmabufPlane<'a> {
    pub fd: BorrowedFd<'a>,
    pub width: i32,
    pub height: i32,
    /// DRM fourcc.
    pub fourcc: u32,
    pub stride: u32,
    pub offset: u32,
    pub modifier: u64,
}

/// Where a capture should land.
pub enum Target<'a> {
    Shm(&'a wl_buffer::WlBuffer),
    Dmabuf(&'a wl_buffer::WlBuffer),
}

impl Target<'_> {
    const fn buffer(&self) -> &wl_buffer::WlBuffer {
        match self {
            Target::Shm(b) | Target::Dmabuf(b) => b,
        }
    }
}

/// Result of one completed copy.
#[derive(Clone, Copy, Debug, Default)]
pub struct CaptureMeta {
    /// The compositor stored the image bottom-up.
    pub y_invert: bool,
    /// Presentation timestamp, nanoseconds, as `PipeWire` wants it.
    pub timestamp_ns: u64,
}

#[derive(Default)]
struct FrameState {
    spec: Option<BufferSpec>,
    /// Set once `buffer_done` arrives (protocol v3+), or after `buffer` on v2.
    negotiated: bool,
    ready: Option<CaptureMeta>,
    failed: bool,
}

/// How far along a dmabuf import is.
#[derive(Default)]
enum ImportState {
    #[default]
    Pending,
    Imported(wl_buffer::WlBuffer),
    Rejected,
}

/// Registry + output bookkeeping for the capture connection.
struct State {
    outputs: Vec<(wl_output::WlOutput, Output)>,
    screencopy: Option<zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1>,
    /// Bound so each `wl_output` can be asked for its *logical* geometry.
    /// `wl_output` alone reports a physical mode and an integer scale, which
    /// cannot describe a fractionally-scaled output's place in the layout.
    xdg_output_manager: Option<zxdg_output_manager_v1::ZxdgOutputManagerV1>,
    shm: Option<wl_shm::WlShm>,
    dmabuf: Option<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1>,
    frame: FrameState,
    /// Outcome of the in-flight `zwp_linux_buffer_params_v1.create`.
    params_buffer: ImportState,
}

/// A Wayland connection dedicated to capture.
///
/// Kept separate from the dialogs' connections so a screencast can't be
/// disturbed by a file chooser, and so the screencast thread owns everything
/// it touches.
pub struct Capturer {
    queue: EventQueue<State>,
    state: State,
}

impl Capturer {
    pub fn new() -> anyhow::Result<Self> {
        let conn = Connection::connect_to_env()
            .map_err(|e| anyhow::anyhow!("no Wayland display for capture: {e}"))?;
        let display = conn.display();
        let mut queue: EventQueue<State> = conn.new_event_queue();
        let qh = queue.handle();
        display.get_registry(&qh, ());
        let mut state = State {
            outputs: Vec::new(),
            screencopy: None,
            xdg_output_manager: None,
            shm: None,
            dmabuf: None,
            frame: FrameState::default(),
            params_buffer: ImportState::Pending,
        };
        // Two roundtrips: one for the globals, one for the output properties
        // (name/mode/scale) that arrive after binding.
        queue.roundtrip(&mut state)?;
        queue.roundtrip(&mut state)?;
        if state.screencopy.is_none() {
            anyhow::bail!(
                "the compositor does not implement zwlr_screencopy_v1 — screen capture is unavailable"
            );
        }
        // Now that every `wl_output` exists, ask each for its logical
        // geometry. Requires a third roundtrip: the request can only be made
        // once the output object is bound, and its reply is another event.
        if let Some(mgr) = state.xdg_output_manager.clone() {
            for (wl, _) in &state.outputs {
                mgr.get_xdg_output(wl, &qh, wl.id());
            }
            queue.roundtrip(&mut state)?;
        } else {
            tracing::warn!(
                "compositor has no xdg_output_manager_v1; assuming scale 1 for screenshot layout"
            );
        }
        state.outputs.sort_by(|a, b| a.1.name.cmp(&b.1.name));
        Ok(Self { queue, state })
    }

    pub fn outputs(&self) -> Vec<Output> {
        self.state.outputs.iter().map(|(_, o)| o.clone()).collect()
    }

    /// Each output's rect in the compositor's **logical** layout —
    /// `(x, y, width, height)` — parallel to [`Capturer::outputs`].
    ///
    /// Position and size come from the same space, which is the whole point:
    /// a frame's size is *physical*, so pairing it with a logical position
    /// silently mixes units on any fractionally-scaled output.
    pub fn output_layout_rects(&self) -> Vec<(i32, i32, i32, i32)> {
        self.state
            .outputs
            .iter()
            .map(|(_, o)| {
                let (w, h) = o.logical_size();
                (o.x, o.y, w, h)
            })
            .collect()
    }

    /// Index of the output with this connector name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.state.outputs.iter().position(|(_, o)| o.name == name)
    }

    fn output(&self, index: usize) -> anyhow::Result<&wl_output::WlOutput> {
        self.state
            .outputs
            .get(index)
            .map(|(wl, _)| wl)
            .ok_or_else(|| anyhow::anyhow!("no such output"))
    }

    /// Pump the queue until `done` says the frame reached a terminal state, or
    /// the deadline passes.
    fn pump(&mut self, done: impl Fn(&FrameState) -> bool) -> anyhow::Result<()> {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !done(&self.state.frame) {
            if self.state.frame.failed {
                anyhow::bail!("the compositor failed the capture");
            }
            if Instant::now() > deadline {
                anyhow::bail!("timed out waiting for the compositor to capture a frame");
            }
            self.queue.blocking_dispatch(&mut self.state)?;
        }
        if self.state.frame.failed {
            anyhow::bail!("the compositor failed the capture");
        }
        Ok(())
    }

    /// Ask what a capture of `output` would look like, without copying.
    ///
    /// The frame object is destroyed immediately: a screencopy frame that is
    /// never copied into still holds a compositor-side capture request, and
    /// leaking one per negotiation would pin resources for the session.
    pub fn probe(&mut self, output: usize, cursor: bool) -> anyhow::Result<BufferSpec> {
        let qh = self.queue.handle();
        let manager = self
            .state
            .screencopy
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no screencopy manager"))?;
        let wl_output = self.output(output)?.clone();
        self.state.frame = FrameState::default();
        let frame = manager.capture_output(i32::from(cursor), &wl_output, &qh, ());
        self.pump(|f| f.negotiated || f.spec.is_some())?;
        frame.destroy();
        self.state
            .frame
            .spec
            .ok_or_else(|| anyhow::anyhow!("the compositor advertised no buffer format"))
    }

    /// Capture one frame of `output` into `target` and wait for it to land.
    pub fn capture_into(
        &mut self,
        output: usize,
        cursor: bool,
        target: &Target<'_>,
    ) -> anyhow::Result<CaptureMeta> {
        let qh = self.queue.handle();
        let manager = self
            .state
            .screencopy
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no screencopy manager"))?;
        let wl_output = self.output(output)?.clone();
        self.state.frame = FrameState::default();
        let frame = manager.capture_output(i32::from(cursor), &wl_output, &qh, ());
        // Wait for the format advertisement before committing to a copy: on
        // v3+ the compositor sends `buffer`, optionally `linux_dmabuf`, then
        // `buffer_done`, and copying early is a protocol error.
        self.pump(|f| f.negotiated || f.spec.is_some())?;
        frame.copy(target.buffer());
        self.pump(|f| f.ready.is_some())?;
        frame.destroy();
        let meta = self.state.frame.ready.unwrap_or_default();
        Ok(meta)
    }

    /// Capture one frame of `output` to CPU memory.
    pub fn capture_frame(&mut self, output: usize, cursor: bool) -> anyhow::Result<Frame> {
        let spec = self.probe(output, cursor)?;
        let stride = spec.shm_stride as usize;
        let len = stride * spec.height as usize;
        let mut pool = ShmPool::new(self, len)?;
        let buffer = pool.buffer(
            self,
            0,
            spec.width.cast_signed(),
            spec.height.cast_signed(),
            spec.shm_stride.cast_signed(),
            spec.shm_format,
        );
        let meta = self.capture_into(output, cursor, &Target::Shm(&buffer))?;
        let mut data = pool.read(len);
        buffer.destroy();

        // Undo a bottom-up capture. Libreland's own screencopy never sets the
        // flag (it flips in the compositor, because xdg-desktop-portal-wlr
        // never implemented y_invert and crashed on it), but other
        // compositors do, and a mirrored screenshot is a memorable bug.
        if meta.y_invert {
            let height = spec.height as usize;
            for row in 0..height / 2 {
                let (top, bottom) = data.split_at_mut((row + 1) * stride);
                let top_row = &mut top[row * stride..];
                let bottom_row = &mut bottom[(height - 2 * row - 2) * stride..][..stride];
                top_row[..stride].swap_with_slice(bottom_row);
            }
        }
        // Normalize the byte order to BGRA regardless of what the compositor
        // handed back, so everything downstream can assume one layout.
        normalize(&mut data, spec.shm_format);
        Ok(Frame {
            width: spec.width as i32,
            height: spec.height as i32,
            stride,
            data,
        })
    }

    /// Capture every output, in [`Capturer::outputs`] order.
    pub fn capture_all(&mut self, cursor: bool) -> anyhow::Result<Vec<Frame>> {
        let count = self.state.outputs.len();
        let mut frames = Vec::with_capacity(count);
        for index in 0..count {
            frames.push(self.capture_frame(index, cursor)?);
        }
        Ok(frames)
    }

    /// Import a dmabuf as a `wl_buffer`, so the compositor can render into it.
    pub fn import_dmabuf(
        &mut self,
        plane: &DmabufPlane<'_>,
    ) -> anyhow::Result<wl_buffer::WlBuffer> {
        let &DmabufPlane {
            fd,
            width,
            height,
            fourcc,
            stride,
            offset,
            modifier,
        } = plane;
        let qh = self.queue.handle();
        let dmabuf = self
            .state
            .dmabuf
            .clone()
            .ok_or_else(|| anyhow::anyhow!("compositor has no zwp_linux_dmabuf_v1"))?;
        let params = dmabuf.create_params(&qh, ());
        params.add(
            fd,
            0,
            offset,
            stride,
            (modifier >> 32) as u32,
            (modifier & 0xFFFF_FFFF) as u32,
        );
        self.state.params_buffer = ImportState::Pending;
        params.create(
            width,
            height,
            fourcc,
            zwp_linux_buffer_params_v1::Flags::empty(),
        );
        let deadline = Instant::now() + Duration::from_secs(2);
        while matches!(self.state.params_buffer, ImportState::Pending) {
            if Instant::now() > deadline {
                anyhow::bail!("timed out importing a dmabuf");
            }
            self.queue.blocking_dispatch(&mut self.state)?;
        }
        params.destroy();
        match std::mem::replace(&mut self.state.params_buffer, ImportState::Pending) {
            ImportState::Imported(buffer) => Ok(buffer),
            _ => Err(anyhow::anyhow!("the compositor rejected the dmabuf import")),
        }
    }

    /// A `wl_shm` pool over a fresh memfd, for the CPU path.
    pub fn shm_pool(&mut self, len: usize) -> anyhow::Result<ShmPool> {
        ShmPool::new(self, len)
    }
}

/// Convert a captured buffer to BGRA in place.
///
/// wlr-screencopy hands back whatever the compositor finds cheapest;
/// everything downstream (PNG encoding, the pickers' blits) assumes BGRA, so
/// the swizzle happens once, here.
fn normalize(data: &mut [u8], format: wl_shm::Format) {
    match format {
        wl_shm::Format::Xbgr8888 | wl_shm::Format::Abgr8888 => {
            // RGBA byte order -> BGRA: swap R and B.
            for px in data.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        // Argb8888/Xrgb8888 are already B,G,R,A in memory on little-endian.
        _ => {}
    }
}

/// A memfd-backed `wl_shm` pool plus its mapping.
pub struct ShmPool {
    pool: wl_shm_pool::WlShmPool,
    map: memmap2::MmapMut,
    fd: OwnedFd,
}

impl ShmPool {
    fn new(capturer: &mut Capturer, len: usize) -> anyhow::Result<Self> {
        use nix::sys::memfd::{MemFdCreateFlag, memfd_create};
        let fd = memfd_create(c"libreland-portal-capture", MemFdCreateFlag::MFD_CLOEXEC)?;
        nix::unistd::ftruncate(&fd, i64::try_from(len)?)?;
        // SAFETY: the memfd was just created, is sized to `len`, and nothing
        // else holds a mapping of it; the mapping lives as long as this pool.
        #[allow(
            unsafe_code,
            reason = "memmap2's map_mut is unsafe by definition (another process could resize the file); this fd is private to us and never resized"
        )]
        // SAFETY: see the #[allow] above.
        let map = unsafe { memmap2::MmapOptions::new().len(len).map_mut(&fd)? };
        let shm = capturer
            .state
            .shm
            .clone()
            .ok_or_else(|| anyhow::anyhow!("compositor has no wl_shm"))?;
        let qh = capturer.queue.handle();
        let pool = shm.create_pool(fd.as_fd(), i32::try_from(len)?, &qh, ());
        Ok(Self { pool, map, fd })
    }

    /// Carve a `wl_buffer` out of the pool.
    #[allow(
        clippy::too_many_arguments,
        reason = "these are exactly wl_shm_pool.create_buffer's arguments"
    )]
    pub fn buffer(
        &mut self,
        capturer: &Capturer,
        offset: i32,
        width: i32,
        height: i32,
        stride: i32,
        format: wl_shm::Format,
    ) -> wl_buffer::WlBuffer {
        let qh = capturer.queue.handle();
        self.pool
            .create_buffer(offset, width, height, stride, format, &qh, ())
    }

    /// Copy `len` bytes out of the mapping.
    fn read(&self, len: usize) -> Vec<u8> {
        self.map.get(..len).unwrap_or(&self.map).to_vec()
    }

    pub fn raw_fd(&self) -> i32 {
        self.fd.as_raw_fd()
    }
}

impl Drop for ShmPool {
    fn drop(&mut self) {
        self.pool.destroy();
    }
}

// ── Wayland dispatch ───────────────────────────────────────────────────────

impl Dispatch<wl_registry::WlRegistry, ()> for State {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        (): &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        else {
            return;
        };
        match interface.as_str() {
            "zwlr_screencopy_manager_v1" => {
                state.screencopy = Some(registry.bind(name, version.min(3), qh, ()));
            }
            "wl_shm" => state.shm = Some(registry.bind(name, 1, qh, ())),
            "zwp_linux_dmabuf_v1" => {
                // v3 is enough to create buffers from a single plane with an
                // explicit modifier; v4/v5 only add feedback we don't use.
                state.dmabuf = Some(registry.bind(name, version.min(3), qh, ()));
            }
            "zxdg_output_manager_v1" => {
                // v2 is where `name` moved to wl_output; we only need
                // logical_position/logical_size, which are v1.
                state.xdg_output_manager = Some(registry.bind(name, version.min(3), qh, ()));
            }
            "wl_output" => {
                let output: wl_output::WlOutput = registry.bind(name, version.min(4), qh, ());
                state.outputs.push((
                    output,
                    Output {
                        name: format!("output-{name}"),
                        width: 0,
                        height: 0,
                        x: 0,
                        y: 0,
                        refresh_mhz: 0,
                        scale: 1,
                        logical_width: 0,
                        logical_height: 0,
                    },
                ));
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_output::WlOutput, ()> for State {
    fn event(
        state: &mut Self,
        output: &wl_output::WlOutput,
        event: wl_output::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let Some((_, info)) = state
            .outputs
            .iter_mut()
            .find(|(wl, _)| wl.id() == output.id())
        else {
            return;
        };
        match event {
            wl_output::Event::Name { name } => info.name = name,
            wl_output::Event::Mode {
                flags,
                width,
                height,
                refresh,
            } => {
                if matches!(flags, WEnum::Value(f) if f.contains(wl_output::Mode::Current)) {
                    info.width = width;
                    info.height = height;
                    info.refresh_mhz = refresh;
                }
            }
            wl_output::Event::Geometry { x, y, .. } => {
                info.x = x;
                info.y = y;
            }
            wl_output::Event::Scale { factor } => info.scale = factor.max(1),
            _ => {}
        }
    }
}

impl Dispatch<zxdg_output_manager_v1::ZxdgOutputManagerV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &zxdg_output_manager_v1::ZxdgOutputManagerV1,
        _: zxdg_output_manager_v1::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // The manager itself emits nothing.
    }
}

/// Keyed by the `wl_output`'s id so the reply can be matched back to the
/// output it describes — `xdg_output` has no back-pointer of its own.
impl Dispatch<zxdg_output_v1::ZxdgOutputV1, ObjectId> for State {
    fn event(
        state: &mut Self,
        _: &zxdg_output_v1::ZxdgOutputV1,
        event: zxdg_output_v1::Event,
        output_id: &ObjectId,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let Some((_, info)) = state
            .outputs
            .iter_mut()
            .find(|(wl, _)| wl.id() == *output_id)
        else {
            return;
        };
        match event {
            // Logical geometry supersedes `wl_output.geometry`/`mode` for
            // layout purposes: it is exact under fractional scaling, where
            // the integer `wl_output.scale` is not.
            zxdg_output_v1::Event::LogicalPosition { x, y } => {
                info.x = x;
                info.y = y;
            }
            zxdg_output_v1::Event::LogicalSize { width, height } => {
                info.logical_width = width;
                info.logical_height = height;
            }
            _ => {}
        }
    }
}

impl Dispatch<zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &zwlr_screencopy_manager_v1::ZwlrScreencopyManagerV1,
        _: zwlr_screencopy_manager_v1::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zwlr_screencopy_frame_v1::ZwlrScreencopyFrameV1, ()> for State {
    fn event(
        state: &mut Self,
        _: &zwlr_screencopy_frame_v1::ZwlrScreencopyFrameV1,
        event: zwlr_screencopy_frame_v1::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_screencopy_frame_v1::Event::Buffer {
                format,
                width,
                height,
                stride,
            } => {
                let shm_format = match format {
                    WEnum::Value(f) => f,
                    WEnum::Unknown(_) => wl_shm::Format::Xrgb8888,
                };
                let spec = state.frame.spec.get_or_insert(BufferSpec {
                    shm_format,
                    shm_stride: stride,
                    drm_format: None,
                    width,
                    height,
                });
                spec.shm_format = shm_format;
                spec.shm_stride = stride;
                spec.width = width;
                spec.height = height;
            }
            zwlr_screencopy_frame_v1::Event::LinuxDmabuf {
                format,
                width,
                height,
            } => {
                let spec = state.frame.spec.get_or_insert(BufferSpec {
                    shm_format: wl_shm::Format::Xrgb8888,
                    shm_stride: width * 4,
                    drm_format: None,
                    width,
                    height,
                });
                spec.drm_format = Some(format);
            }
            zwlr_screencopy_frame_v1::Event::BufferDone => state.frame.negotiated = true,
            zwlr_screencopy_frame_v1::Event::Flags { flags } => {
                if matches!(flags, WEnum::Value(f) if f.contains(zwlr_screencopy_frame_v1::Flags::YInvert))
                {
                    state
                        .frame
                        .ready
                        .get_or_insert_with(CaptureMeta::default)
                        .y_invert = true;
                }
            }
            zwlr_screencopy_frame_v1::Event::Ready {
                tv_sec_hi,
                tv_sec_lo,
                tv_nsec,
            } => {
                let secs = (u64::from(tv_sec_hi) << 32) | u64::from(tv_sec_lo);
                let meta = state.frame.ready.get_or_insert_with(CaptureMeta::default);
                meta.timestamp_ns = secs * 1_000_000_000 + u64::from(tv_nsec);
                // `Ready` implies negotiation finished, for v2 compositors
                // that never send `buffer_done`.
                state.frame.negotiated = true;
            }
            zwlr_screencopy_frame_v1::Event::Failed => state.frame.failed = true,
            _ => {}
        }
    }
}

impl Dispatch<zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &zwp_linux_dmabuf_v1::ZwpLinuxDmabufV1,
        _: zwp_linux_dmabuf_v1::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1, ()> for State {
    // `created` hands back a brand-new `wl_buffer`, and wayland-client needs
    // to be told what interface and user data that child gets *before* it
    // deserializes the event. Without this the library panics on the first
    // successful dmabuf import — and since the panic unwinds through
    // PipeWire's C callback, it aborts the whole process rather than failing
    // the one capture.
    wayland_client::event_created_child!(State, zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1, [
        zwp_linux_buffer_params_v1::EVT_CREATED_OPCODE => (wl_buffer::WlBuffer, ()),
    ]);

    fn event(
        state: &mut Self,
        _: &zwp_linux_buffer_params_v1::ZwpLinuxBufferParamsV1,
        event: zwp_linux_buffer_params_v1::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwp_linux_buffer_params_v1::Event::Created { buffer } => {
                state.params_buffer = ImportState::Imported(buffer);
            }
            zwp_linux_buffer_params_v1::Event::Failed => {
                state.params_buffer = ImportState::Rejected;
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_shm::WlShm, ()> for State {
    fn event(
        _: &mut Self,
        _: &wl_shm::WlShm,
        _: wl_shm::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for State {
    fn event(
        _: &mut Self,
        _: &wl_shm_pool::WlShmPool,
        _: wl_shm_pool::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_buffer::WlBuffer, ()> for State {
    fn event(
        _: &mut Self,
        _: &wl_buffer::WlBuffer,
        _: wl_buffer::Event,
        (): &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
