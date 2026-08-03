//! The `PipeWire` half of screen casting.
//!
//! A screencast session is a `PipeWire` node that other apps (browsers, OBS,
//! conferencing clients) connect to. This module owns one such node per
//! session, on its own thread, and pumps frames into it from
//! [`crate::capture`].
//!
//! # Buffers
//!
//! We connect with `ALLOC_BUFFERS`, meaning *we* provide the memory behind
//! each `PipeWire` buffer. That isn't an optimization detail, it's a
//! requirement: the compositor has to render into memory we can hand it as a
//! `wl_buffer`, so the same allocation must be visible to both Wayland and
//! `PipeWire`. Two ways to satisfy that:
//!
//! * **dmabuf** — a gbm buffer object, exported as a dma-buf fd, imported into
//!   Wayland through `zwp_linux_dmabuf_v1` and into `PipeWire` as
//!   `SPA_DATA_DmaBuf`. The pixels never touch the CPU. The layout is
//!   whichever one the driver will render into, probed once and then used
//!   for both the offer and every allocation — see `probe_modifier`.
//! * **memfd** — the same buffer as plain shared memory, which every consumer
//!   accepts and no driver can refuse.
//!
//! The consumer picks by which format it accepts; we offer dmabuf first when a
//! render node is usable and fall back to memfd otherwise.
//!
//! # Threading
//!
//! One thread per session runs the `PipeWire` loop and owns the capture
//! connection. Frames are captured inside the stream's `process` callback: the
//! screencopy round-trip is synchronous, and serializing it against the
//! consumer's demand is exactly the back-pressure a screencast wants — a
//! consumer that stops asking stops us capturing.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel geometry: every value here is a surface- or image-sized non-negative integer, and the conversions between i32/u32/usize/f32 are all inside that range. Checked conversions at each site would be noise around arithmetic that cannot overflow."
)]

use std::collections::HashMap;
use std::io::Cursor;
use std::os::fd::{AsRawFd as _, OwnedFd};
use std::sync::Arc;
use std::time::{Duration, Instant};

use libspa::param::ParamType;
use libspa::param::format::{FormatProperties, MediaSubtype, MediaType};
use libspa::param::video::VideoFormat;
use libspa::pod::{
    ChoiceValue, Object, Pod, Property, PropertyFlags, Value, serialize::PodSerializer,
};
use libspa::utils::{Choice, ChoiceEnum, ChoiceFlags, Fraction, Id, Rectangle, SpaTypes};
use pipewire as pw;
use pipewire::loop_::Timeout;
use wayland_client::protocol::wl_buffer;

use crate::capture::{BufferSpec, Capturer, DmabufPlane, ShmPool, Target};
use crate::portals::Cancel;

/// What to cast.
pub struct Request {
    /// Connector name, as the compositor knows it.
    pub output: String,
    /// Composite the pointer into the stream.
    pub cursor: bool,
}

/// What the session needs to answer `Start` with.
pub struct Started {
    pub node_id: u32,
    /// Physical size of the stream, for the `size` result.
    pub width: i32,
    pub height: i32,
}

/// Map a `wl_shm/DRM` 32-bit packed format to the SPA video format with the same
/// byte order in memory.
///
/// The two vocabularies disagree about direction: DRM names the *word* as it
/// reads on a little-endian machine (`XR24` = 0xXXRRGGBB), SPA names the bytes
/// in memory order (`BGRx`). They describe the same buffer.
const fn spa_format(fourcc: u32) -> VideoFormat {
    match fourcc {
        // AR24 / ARGB8888
        0x3432_5241 => VideoFormat::BGRA,
        // XB24 / XBGR8888
        0x3432_4258 => VideoFormat::RGBx,
        // AB24 / ABGR8888
        0x3432_4241 => VideoFormat::RGBA,
        // XR24 / XRGB8888, and anything else we'd only guess at
        _ => VideoFormat::BGRx,
    }
}

/// The DRM fourcc matching a `wl_shm` format, for the dmabuf path.
const fn fourcc_of(format: wayland_client::protocol::wl_shm::Format) -> u32 {
    use wayland_client::protocol::wl_shm::Format;
    match format {
        Format::Argb8888 => 0x3432_5241,
        Format::Xbgr8888 => 0x3432_4258,
        Format::Abgr8888 => 0x3432_4241,
        _ => 0x3432_5258,
    }
}

/// Serialize one pod object into bytes we can hand to `PipeWire`.
fn to_pod(object: Object) -> Vec<u8> {
    PodSerializer::serialize(Cursor::new(Vec::new()), &Value::Object(object))
        .map(|(cursor, _)| cursor.into_inner())
        .unwrap_or_default()
}

/// The modifier this driver will hand out for a render target of
/// `spec`'s size and format — allocated, read, and thrown away.
///
/// The offer and the allocation have to name the *same* layout. They
/// didn't for one release: the offer said LINEAR while the allocator
/// asked for whatever the driver preferred, so a consumer negotiated
/// linear, received tiled, imported it as linear and got garbage. Both
/// OBS and Discord dropped to shared memory about three milliseconds
/// in. Probing once and using the answer in both places makes them
/// agree by construction rather than by matching literals.
fn probe_modifier(device: &gbm::Device<std::fs::File>, spec: &BufferSpec) -> Option<u64> {
    let fourcc = drm_fourcc::DrmFourcc::try_from(fourcc_of(spec.shm_format)).ok()?;
    let bo = device
        .create_buffer_object::<()>(
            spec.width,
            spec.height,
            fourcc,
            gbm::BufferObjectFlags::RENDERING,
        )
        .ok()?;
    Some(u64::from(bo.modifier()))
}

/// One `EnumFormat` offer. With `modifier`, it advertises the dmabuf variant.
fn format_pod(spec: &BufferSpec, fps: u32, modifier: Option<u64>) -> Vec<u8> {
    let mut properties = vec![
        Property {
            key: FormatProperties::MediaType.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Id(Id(MediaType::Video.as_raw())),
        },
        Property {
            key: FormatProperties::MediaSubtype.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Id(Id(MediaSubtype::Raw.as_raw())),
        },
        Property {
            key: FormatProperties::VideoFormat.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Id(Id(spa_format(fourcc_of(spec.shm_format)).as_raw())),
        },
        Property {
            key: FormatProperties::VideoSize.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Rectangle(Rectangle {
                width: spec.width,
                height: spec.height,
            }),
        },
        // A variable framerate: the compositor decides when there's a new
        // frame, so pinning a rate here would just make us lie about it.
        Property {
            key: FormatProperties::VideoFramerate.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Fraction(Fraction { num: 0, denom: 1 }),
        },
        Property {
            key: FormatProperties::VideoMaxFramerate.as_raw(),
            flags: PropertyFlags::empty(),
            value: Value::Choice(ChoiceValue::Fraction(Choice(
                ChoiceFlags::empty(),
                ChoiceEnum::Range {
                    default: Fraction { num: fps, denom: 1 },
                    min: Fraction { num: 1, denom: 1 },
                    max: Fraction { num: fps, denom: 1 },
                },
            ))),
        },
    ];
    if let Some(modifier) = modifier {
        // MANDATORY marks this as a dmabuf offer: a consumer that can't take
        // dmabufs skips the whole format rather than silently accepting it
        // and then failing to map anything.
        properties.push(Property {
            key: FormatProperties::VideoModifier.as_raw(),
            flags: PropertyFlags::MANDATORY,
            value: Value::Long(modifier as i64),
        });
    }
    to_pod(Object {
        type_: SpaTypes::ObjectParamFormat.as_raw(),
        id: ParamType::EnumFormat.as_raw(),
        properties,
    })
}

/// The `Buffers` param: how many buffers, how big, and what memory types we
/// will be handing over.
fn buffers_pod(spec: &BufferSpec, dmabuf: bool) -> Vec<u8> {
    let size = spec.shm_stride * spec.height;
    // SPA_DATA_DmaBuf / SPA_DATA_MemFd as a bitmask of data types.
    let data_type = if dmabuf {
        1 << libspa::sys::SPA_DATA_DmaBuf
    } else {
        1 << libspa::sys::SPA_DATA_MemFd
    };
    to_pod(Object {
        type_: SpaTypes::ObjectParamBuffers.as_raw(),
        id: ParamType::Buffers.as_raw(),
        properties: vec![
            Property {
                key: libspa::sys::SPA_PARAM_BUFFERS_buffers,
                flags: PropertyFlags::empty(),
                // Three is the usual producer depth: one being filled, one in
                // flight, one held by the consumer.
                value: Value::Choice(ChoiceValue::Int(Choice(
                    ChoiceFlags::empty(),
                    ChoiceEnum::Range {
                        default: 3,
                        min: 2,
                        max: 8,
                    },
                ))),
            },
            Property {
                key: libspa::sys::SPA_PARAM_BUFFERS_blocks,
                flags: PropertyFlags::empty(),
                value: Value::Int(1),
            },
            Property {
                key: libspa::sys::SPA_PARAM_BUFFERS_size,
                flags: PropertyFlags::empty(),
                value: Value::Int(size as i32),
            },
            Property {
                key: libspa::sys::SPA_PARAM_BUFFERS_stride,
                flags: PropertyFlags::empty(),
                value: Value::Int(spec.shm_stride as i32),
            },
            Property {
                key: libspa::sys::SPA_PARAM_BUFFERS_align,
                flags: PropertyFlags::empty(),
                value: Value::Int(16),
            },
            Property {
                key: libspa::sys::SPA_PARAM_BUFFERS_dataType,
                flags: PropertyFlags::empty(),
                value: Value::Choice(ChoiceValue::Int(Choice(
                    ChoiceFlags::empty(),
                    ChoiceEnum::Flags {
                        default: data_type,
                        flags: vec![data_type],
                    },
                ))),
            },
        ],
    })
}

/// The `Meta` param asking for a header on each buffer (sequence numbers and
/// timestamps, which consumers use to detect drops).
fn header_pod() -> Vec<u8> {
    to_pod(Object {
        type_: SpaTypes::ObjectParamMeta.as_raw(),
        id: ParamType::Meta.as_raw(),
        properties: vec![
            Property {
                key: libspa::sys::SPA_PARAM_META_type,
                flags: PropertyFlags::empty(),
                value: Value::Id(Id(libspa::sys::SPA_META_Header)),
            },
            Property {
                key: libspa::sys::SPA_PARAM_META_size,
                flags: PropertyFlags::empty(),
                value: Value::Int(std::mem::size_of::<libspa::sys::spa_meta_header>() as i32),
            },
        ],
    })
}

/// Run a stream callback so a panic inside it can't abort the process.
///
/// These closures are called from `PipeWire`'s C code, and a panic unwinding
/// across that boundary aborts immediately — taking every other portal in this
/// process with it (file dialogs, settings, notifications). That is exactly
/// what one missing Wayland declaration did on the first dmabuf import. Only
/// the capture should die when the capture breaks.
fn guard<T>(what: &'static str, body: impl FnOnce() -> T) -> Option<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(body))
        .inspect_err(|_| {
            tracing::error!(
                callback = what,
                "screencast callback panicked; ending the cast"
            );
        })
        .ok()
}

/// One `PipeWire` buffer and the Wayland object the compositor renders into.
struct Slot {
    wl: wl_buffer::WlBuffer,
    /// Kept alive for the buffer's lifetime; dropping either would unmap the
    /// memory `PipeWire` still thinks it owns.
    _shm: Option<ShmPool>,
    _dmabuf: Option<OwnedFd>,
    _bo: Option<gbm::BufferObject<()>>,
}

/// Everything the stream callbacks need. `PipeWire` hands this back to each
/// callback as `&mut`, which is what lets the capture connection live here
/// instead of behind a lock.
struct Cast {
    capturer: Capturer,
    output: usize,
    cursor: bool,
    spec: BufferSpec,
    gbm: Option<gbm::Device<std::fs::File>>,
    /// The layout advertised to the consumer, and therefore the only one
    /// we may allocate — see [`probe_modifier`].
    dmabuf_modifier: Option<u64>,
    /// Set once the format is negotiated; until then we don't know which
    /// allocation path the consumer accepted.
    dmabuf: bool,
    /// Buffers keyed by their file descriptor, which is unique per buffer and
    /// is the one identifier available in both `add_buffer` and `process`.
    slots: HashMap<i32, Slot>,
    sequence: u64,
    /// Set when the compositor stops answering, so the thread can exit.
    broken: bool,
}

impl Cast {
    /// Allocate one buffer's memory and register it with Wayland.
    fn allocate(&mut self) -> anyhow::Result<(Slot, i32, u32, u32)> {
        let (width, height) = (self.spec.width, self.spec.height);
        let stride = self.spec.shm_stride;
        if self.dmabuf {
            let device = self
                .gbm
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no gbm device for the dmabuf path"))?;
            let fourcc = drm_fourcc::DrmFourcc::try_from(fourcc_of(self.spec.shm_format))
                .unwrap_or(drm_fourcc::DrmFourcc::Xrgb8888);
            // Let the driver pick the layout for a render target, and
            // only ask for LINEAR if it can't.
            //
            // Forcing LINEAR is what this did first, on the reasoning
            // that linear is universally *importable*. It is — but this
            // buffer is not imported, it is rendered into, and NVIDIA's
            // EGL exposes no linear render target at all. Every dmabuf
            // capture on such a GPU was refused by the compositor and
            // silently fell back to a CPU copy of a 4K frame, per frame.
            //
            // `RENDERING` is the flag that says so; gbm then answers with
            // a modifier the driver will actually render into, which is
            // the one thing the compositor needs to be true.
            // Exactly the layout the consumer was offered. Asking the
            // driver again and taking whatever it says would be the bug
            // this replaced: the two answers only *usually* match, and a
            // consumer that imports a tiled buffer as linear shows
            // garbage and drops to shared memory.
            let modifier = self
                .dmabuf_modifier
                .ok_or_else(|| anyhow::anyhow!("dmabuf path without a negotiated modifier"))?;
            let bo = device.create_buffer_object_with_modifiers::<()>(
                width,
                height,
                fourcc,
                [drm_fourcc::DrmModifier::from(modifier)].into_iter(),
            )?;
            let fd = bo.fd()?;
            let bo_stride = bo.stride();
            let offset = bo.offset(0);
            let raw = fd.as_raw_fd();
            let wl = self.capturer.import_dmabuf(&DmabufPlane {
                fd: std::os::fd::AsFd::as_fd(&fd),
                width: width.try_into()?,
                height: height.try_into()?,
                fourcc: fourcc as u32,
                stride: bo_stride,
                offset,
                modifier,
            })?;
            Ok((
                Slot {
                    wl,
                    _shm: None,
                    _dmabuf: Some(fd),
                    _bo: Some(bo),
                },
                raw,
                bo_stride,
                bo_stride * height,
            ))
        } else {
            let len = (stride * height) as usize;
            let mut pool = self.capturer.shm_pool(len)?;
            let wl = pool.buffer(
                &self.capturer,
                0,
                width.try_into()?,
                height.try_into()?,
                stride.try_into()?,
                self.spec.shm_format,
            );
            let raw = pool.raw_fd();
            Ok((
                Slot {
                    wl,
                    _shm: Some(pool),
                    _dmabuf: None,
                    _bo: None,
                },
                raw,
                stride,
                stride * height,
            ))
        }
    }
}

/// Open a render node for the dmabuf path. `None` disables it (we then offer
/// only the memfd format, which always works).
fn open_render_node() -> Option<gbm::Device<std::fs::File>> {
    let entries = std::fs::read_dir("/dev/dri").ok()?;
    let mut nodes: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("renderD"))
        })
        .collect();
    nodes.sort();
    for node in nodes {
        if let Ok(file) = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&node)
            && let Ok(device) = gbm::Device::new(file)
        {
            tracing::info!(node = %node.display(), "dmabuf capture path enabled");
            return Some(device);
        }
    }
    tracing::info!("no usable render node; screencast will use shared memory");
    None
}

/// Run one cast to completion on the calling thread.
///
/// Sends the negotiated node id (or the failure) through `ready` as soon as
/// the stream is connected, then loops until `stop` is tripped or the
/// compositor goes away.
#[allow(
    clippy::needless_pass_by_value,
    reason = "the request is consumed by the thread that owns the cast"
)]
pub fn run(
    request: Request,
    ready: std::sync::mpsc::Sender<anyhow::Result<Started>>,
    stop: Arc<Cancel>,
) {
    if let Err(err) = run_inner(&request, &ready, &stop) {
        // If we never got as far as reporting readiness, the waiting Start
        // call is still blocked on the channel; unblock it with the error.
        let _ = ready.send(Err(err));
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one stream's lifecycle: negotiate, register callbacks, connect, pump. The callbacks close over the same state and reading them in order is the point"
)]
fn run_inner(
    request: &Request,
    ready: &std::sync::mpsc::Sender<anyhow::Result<Started>>,
    stop: &Arc<Cancel>,
) -> anyhow::Result<()> {
    let mut capturer = Capturer::new()?;
    let output = capturer
        .index_of(&request.output)
        .ok_or_else(|| anyhow::anyhow!("output {} is gone", request.output))?;
    let spec = capturer.probe(output, request.cursor)?;
    let refresh = capturer
        .outputs()
        .get(output)
        .map_or(60_000, |o| o.refresh_mhz.max(1000));
    let fps = (refresh as u32 / 1000).clamp(1, 240);

    // The loop below drives the graph, and it can only do that while the
    // stream is actually streaming — which only the callback learns. Same for
    // the rate: the consumer negotiates a maximum framerate, and driving
    // faster than it asked for is pure waste (a 240 Hz monitor shared into a
    // 30 fps call would capture eight frames per frame anyone sees).
    let streaming = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let negotiated_fps = Arc::new(std::sync::atomic::AtomicU32::new(0));

    pw::init();
    let main_loop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&main_loop, None)?;
    let core = context.connect_rc(None)?;

    let properties = pw::properties::properties! {
        *pw::keys::MEDIA_CLASS => "Video/Source",
        *pw::keys::MEDIA_ROLE => "Screen",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::NODE_NAME => "libreland-screencast",
        *pw::keys::NODE_DESCRIPTION => "Libreland screen capture",
    };
    let stream = pw::stream::StreamRc::new(core, "libreland-screencast", properties)?;

    let gbm = open_render_node();
    // Probe before offering: an offer we cannot then allocate to match
    // is worse than no offer, because the consumer commits to it.
    let dmabuf_modifier = gbm.as_ref().and_then(|d| probe_modifier(d, &spec));
    if gbm.is_some() && dmabuf_modifier.is_none() {
        tracing::warn!("no renderable dmabuf layout for this format; offering shared memory only");
    }
    let cast = Cast {
        capturer,
        output,
        cursor: request.cursor,
        spec,
        dmabuf_modifier,
        dmabuf: false,
        gbm,
        slots: HashMap::new(),
        sequence: 0,
        broken: false,
    };

    let _listener = stream
        .add_local_listener_with_user_data(cast)
        .state_changed({
            let streaming = Arc::clone(&streaming);
            move |_, _, old, new| {
                tracing::info!(?old, ?new, "screencast stream state");
                streaming.store(
                    matches!(new, pw::stream::StreamState::Streaming),
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        })
        .param_changed({
            let negotiated_fps = Arc::clone(&negotiated_fps);
            move |stream, cast, id, pod| {
            if id != ParamType::Format.as_raw() {
                return;
            }
            let Some(pod) = pod else { return };
            // Which format the consumer picked tells us which allocation path
            // to use: a modifier property is only present in the dmabuf offer.
            let Ok((media_type, media_subtype)) = libspa::param::format_utils::parse_format(pod)
            else {
                return;
            };
            if media_type != MediaType::Video || media_subtype != MediaSubtype::Raw {
                return;
            }
            let mut info = libspa::param::video::VideoInfoRaw::default();
            if info.parse(pod).is_err() {
                return;
            }
            cast.dmabuf = info.flags().contains(libspa::param::video::VideoFlags::MODIFIER);
            // A zero denominator (or a zero rate) means "unspecified"; leave
            // the fallback in place rather than dividing by it.
            let max = info.max_framerate();
            if max.denom > 0 && max.num > 0 {
                negotiated_fps.store(
                    max.num / max.denom,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            tracing::info!(
                dmabuf = cast.dmabuf,
                width = info.size().width,
                height = info.size().height,
                max_fps = max.num / max.denom.max(1),
                "screencast format negotiated"
            );
            let buffers = buffers_pod(&cast.spec, cast.dmabuf);
            let meta = header_pod();
            let (Some(buffers), Some(meta)) = (Pod::from_bytes(&buffers), Pod::from_bytes(&meta))
            else {
                return;
            };
            if let Err(err) = stream.update_params(&mut [buffers, meta]) {
                tracing::error!(%err, "could not publish buffer parameters");
            }
            }
        })
        .add_buffer(|_, cast, buffer| {
            let Some(allocated) = guard("add_buffer", || cast.allocate()) else {
                cast.broken = true;
                return;
            };
            let (slot, fd, stride, size) = match allocated {
                Ok(allocated) => allocated,
                Err(err) => {
                    tracing::error!(%err, "could not allocate a screencast buffer");
                    cast.broken = true;
                    return;
                }
            };
            // SAFETY: PipeWire hands us this buffer to fill in exactly here,
            // it has at least one data block (we asked for blocks=1), and we
            // only write the descriptor fields — the memory behind `fd` is
            // owned by the slot we keep alive in `cast.slots`.
            #[allow(
                unsafe_code,
                reason = "filling in producer-allocated buffer memory is a raw spa_buffer operation; the Rust bindings expose the pw_buffer pointer for precisely this"
            )]
            // SAFETY: see the #[allow] above.
            unsafe {
                let spa_buffer = (*buffer).buffer;
                if spa_buffer.is_null() || (*spa_buffer).n_datas < 1 {
                    return;
                }
                let data = &mut *(*spa_buffer).datas;
                data.type_ = if cast.dmabuf {
                    libspa::sys::SPA_DATA_DmaBuf
                } else {
                    libspa::sys::SPA_DATA_MemFd
                };
                data.flags = libspa::sys::SPA_DATA_FLAG_READABLE;
                data.fd = i64::from(fd);
                data.mapoffset = 0;
                data.maxsize = size;
                // Null `data`: the consumer maps it itself (or imports the
                // dmabuf); we never touch the pixels from the CPU.
                data.data = std::ptr::null_mut();
                if !data.chunk.is_null() {
                    {
                        (*data.chunk).offset = 0;
                        (*data.chunk).stride = stride as i32;
                        (*data.chunk).size = size;
                    }
                }
            }
            cast.slots.insert(fd, slot);
        })
        .remove_buffer(|_, cast, buffer| {
            // SAFETY: same contract as `add_buffer` — the pointer is live for
            // the duration of the callback and we only read the descriptor.
            #[allow(
                unsafe_code,
                reason = "reading back the fd we stored is the only way to find the slot this buffer belongs to"
            )]
            // SAFETY: see the #[allow] above.
            let fd = unsafe {
                let spa_buffer = (*buffer).buffer;
                if spa_buffer.is_null() || (*spa_buffer).n_datas < 1 {
                    return;
                }
                {
                    (*(*spa_buffer).datas).fd as i32
                }
            };
            if let Some(slot) = cast.slots.remove(&fd) {
                slot.wl.destroy();
            }
        })
        .process(|stream, cast| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            let Some(data) = datas.first_mut() else {
                return;
            };
            let key = data.fd();
            let Some(slot) = cast.slots.get(&key) else {
                return;
            };
            let wl = slot.wl.clone();
            let target = if cast.dmabuf {
                Target::Dmabuf(&wl)
            } else {
                Target::Shm(&wl)
            };
            let Some(outcome) = guard("process", || {
                cast.capturer.capture_into(cast.output, cast.cursor, &target)
            }) else {
                cast.broken = true;
                return;
            };
            match outcome {
                Ok(_) => {
                    cast.sequence += 1;
                    // One line at the first frame, then one per ~10 seconds of
                    // capture: enough to tell "streaming but blank" (the
                    // consumer's problem) from "never captured a frame" (ours).
                    if cast.sequence == 1 || cast.sequence % 600 == 0 {
                        tracing::info!(frames = cast.sequence, "screencast capturing");
                    }
                    let stride = cast.spec.shm_stride;
                    let size = stride * cast.spec.height;
                    let chunk = data.chunk_mut();
                    *chunk.offset_mut() = 0;
                    {
                        *chunk.stride_mut() = stride as i32;
                    }
                    *chunk.size_mut() = size;
                }
                Err(err) => {
                    tracing::error!(%err, "screencast capture failed");
                    cast.broken = true;
                }
            }
            // Dropping the buffer queues it back to the consumer.
        })
        .register()?;

    // Offer dmabuf first (when we can allocate one) so a capable consumer
    // takes the zero-copy path, then plain memory as the always-works option.
    let dmabuf_pod = format_pod(&spec, fps, dmabuf_modifier);
    let shm_pod = format_pod(&spec, fps, None);
    let mut params: Vec<&Pod> = Vec::new();
    if dmabuf_modifier.is_some()
        && let Some(pod) = Pod::from_bytes(&dmabuf_pod)
    {
        params.push(pod);
    }
    if let Some(pod) = Pod::from_bytes(&shm_pod) {
        params.push(pod);
    }

    stream.connect(
        libspa::utils::Direction::Output,
        None,
        pw::stream::StreamFlags::DRIVER | pw::stream::StreamFlags::ALLOC_BUFFERS,
        &mut params,
    )?;

    // The node id isn't assigned until the core has processed the connect, so
    // spin the loop until it appears.
    let mut node_id = pw::constants::ID_ANY;
    for _ in 0..200 {
        main_loop
            .loop_()
            .iterate(Timeout::Finite(Duration::from_millis(10)));
        node_id = stream.node_id();
        if node_id != pw::constants::ID_ANY {
            break;
        }
    }
    if node_id == pw::constants::ID_ANY {
        anyhow::bail!("PipeWire never assigned a node id to the stream");
    }
    tracing::info!(node_id, output = %request.output, "screencast started");
    let _ = ready.send(Ok(Started {
        node_id,
        width: spec.width as i32,
        height: spec.height as i32,
    }));

    // We connected as the graph DRIVER, which means PipeWire calls `process`
    // only when we ask it to: a driver decides when a frame exists. Screen
    // capture has no natural clock to hang that off — the compositor doesn't
    // push us frames, we pull them — so the pull is paced here, at the
    // output's refresh rate. Without this the node exists, the consumer links
    // to it, `process` is never called and the viewer sees nothing at all.
    let mut interval = Duration::from_micros(1_000_000 / u64::from(fps.max(1)));
    let mut next_frame = Instant::now();
    while !stop.is_cancelled() {
        // Wake at least once per frame, so the trigger below isn't late.
        let timeout = next_frame
            .saturating_duration_since(Instant::now())
            .min(interval);
        main_loop.loop_().iterate(Timeout::Finite(timeout));

        // Adopt the negotiated rate once it lands (it arrives with the
        // format, after the loop is already running).
        let wanted = negotiated_fps.load(std::sync::atomic::Ordering::Relaxed);
        if wanted > 0 {
            interval = Duration::from_micros(1_000_000 / u64::from(wanted));
        }
        if streaming.load(std::sync::atomic::Ordering::Relaxed) && Instant::now() >= next_frame {
            // Skip missed deadlines rather than trying to catch up: a
            // consumer that stalled shouldn't earn a burst of captures.
            next_frame = Instant::now() + interval;
            if stream.is_driving()
                && let Err(err) = stream.trigger_process()
            {
                tracing::warn!(%err, "could not trigger a screencast frame");
            }
        }
        // A capture that started failing (the output was unplugged, the
        // compositor exited) ends the session rather than spinning: the
        // consumer sees the node disappear, which is the signal it expects.
        if matches!(stream.state(), pw::stream::StreamState::Error(_)) {
            tracing::warn!("screencast stream entered an error state");
            break;
        }
    }
    tracing::info!(node_id, "screencast stopped");
    let _ = stream.disconnect();
    Ok(())
}
