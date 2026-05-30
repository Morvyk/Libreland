//! Media wallpaper decoding via libav (`FFmpeg`).
//!
//! [`decode_first_frame`] grabs a single frame (for the immediate, static
//! display) and [`Animation`] runs a background thread that decodes a
//! video/gif on a loop, handing the renderer the latest frame to upload.
//! Both work on anything `FFmpeg` can open; a still image just yields one
//! frame and the animation thread self-terminates.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use ffmpeg_the_third as ffmpeg;
use tracing::warn;

/// A decoded frame: tightly packed RGBA (byte order R, G, B, A), ready to
/// import as a GLES texture with `Fourcc::Abgr8888` (the same convention
/// the screenshot-freeze path uses).
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Decode the first video frame of `path`, downscaled to fit within a
/// `max_dim`×`max_dim` box (preserving aspect; never upscaled), as RGBA.
/// Works for stills and for the first frame of animated/video media.
pub fn decode_first_frame(path: &Path, max_dim: u32) -> Result<Frame> {
    ffmpeg::init().context("FFmpeg init")?;
    let mut dec = Decoder::open(path, max_dim)?;
    for item in dec.input.packets() {
        let (stream, packet) = item.context("read packet")?;
        if stream.index() != dec.stream_index {
            continue;
        }
        dec.video.send_packet(&packet).context("send packet")?;
        if let Some(frame) = next_rgba(&mut dec.video, &mut dec.scaler) {
            return Ok(frame);
        }
    }
    // Flush a frame the decoder may still hold (single-image inputs often
    // emit theirs only on EOF).
    dec.video.send_eof().context("flush decoder")?;
    if let Some(frame) = next_rgba(&mut dec.video, &mut dec.scaler) {
        return Ok(frame);
    }
    bail!("no decodable frame in {}", path.display())
}

/// A running media-wallpaper decode: a background thread loops the file
/// and publishes the latest decoded frame into a shared slot, paced at the
/// stream's frame rate. The renderer polls [`Animation::take_new`] each
/// frame and uploads anything new. Dropping the `Animation` signals the
/// thread to stop.
pub struct Animation {
    slot: Arc<Mutex<Slot>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

/// The shared latest-frame mailbox: at most one pending frame (newer
/// frames overwrite unseen ones — fine for a wallpaper) plus a sequence
/// number so the renderer can tell when something changed.
#[derive(Default)]
struct Slot {
    frame: Option<Frame>,
    seq: u64,
}

impl Animation {
    /// Spawn the decode thread for `path`. Returns immediately; the first
    /// frame appears in the slot once decoded.
    pub fn start(path: &Path, max_dim: u32) -> Self {
        let slot = Arc::new(Mutex::new(Slot::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let handle = {
            let (slot, stop, path) = (slot.clone(), stop.clone(), path.to_owned());
            std::thread::Builder::new()
                .name("wallpaper-decode".to_owned())
                .spawn(move || {
                    if let Err(err) = decode_loop(&path, max_dim, &slot, &stop) {
                        warn!(error = %err, path = %path.display(), "wallpaper animation stopped");
                    }
                })
                .ok()
        };
        Self { slot, stop, handle }
    }

    /// Take the latest frame if the slot has advanced past `last_seq`.
    /// Returns the frame and its new sequence number.
    pub fn take_new(&self, last_seq: u64) -> Option<(Frame, u64)> {
        let mut slot = self.slot.lock().unwrap();
        if slot.seq == last_seq {
            return None;
        }
        let seq = slot.seq;
        slot.frame.take().map(|frame| (frame, seq))
    }
}

impl Drop for Animation {
    fn drop(&mut self) {
        // Tell the thread to exit, then join briefly. It checks `stop`
        // between frames (and during its inter-frame sleep it wakes within
        // one frame interval), so this won't block long.
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Decode `path` on a loop, publishing each frame into `slot` paced at the
/// stream's frame rate, until `stop` is set. A single-frame input (still
/// image) publishes once and returns — no busy loop. The file is reopened
/// each loop rather than seeked, which is simpler and codec-agnostic.
fn decode_loop(path: &Path, max_dim: u32, slot: &Mutex<Slot>, stop: &AtomicBool) -> Result<()> {
    ffmpeg::init().context("FFmpeg init")?;
    let mut seq = 0u64;
    loop {
        let mut dec = Decoder::open(path, max_dim)?;
        let interval = Duration::from_secs_f64(1.0 / dec.fps);
        let mut next = Instant::now();
        let mut frames = 0u64;

        let mut publish = |frame: Frame| {
            seq += 1;
            frames += 1;
            {
                let mut s = slot.lock().unwrap();
                s.frame = Some(frame);
                s.seq = seq;
            }
            // Pace to the frame rate, catching up if we've fallen behind.
            next += interval;
            let now = Instant::now();
            if next > now {
                std::thread::sleep(next - now);
            } else {
                next = now;
            }
        };

        for item in dec.input.packets() {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let (stream, packet) = item.context("read packet")?;
            if stream.index() != dec.stream_index {
                continue;
            }
            dec.video.send_packet(&packet).context("send packet")?;
            while let Some(frame) = next_rgba(&mut dec.video, &mut dec.scaler) {
                publish(frame);
                if stop.load(Ordering::Relaxed) {
                    return Ok(());
                }
            }
        }
        dec.video.send_eof().ok();
        while let Some(frame) = next_rgba(&mut dec.video, &mut dec.scaler) {
            publish(frame);
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
        }

        // A still image (one frame total) needs no looping.
        if frames <= 1 {
            return Ok(());
        }
    }
}

/// An opened input + its video decoder + an RGBA scaler + the stream's
/// frame rate. Shared by the still and animated paths.
struct Decoder {
    input: ffmpeg::format::context::Input,
    stream_index: usize,
    video: ffmpeg::decoder::Video,
    scaler: ffmpeg::software::scaling::Context,
    fps: f64,
}

impl Decoder {
    fn open(path: &Path, max_dim: u32) -> Result<Self> {
        let input = ffmpeg::format::input(path)
            .with_context(|| format!("FFmpeg could not open {}", path.display()))?;

        // Scope the immutable stream borrow so `input.packets()` (a mutable
        // borrow) is free for the caller. `from_parameters` copies the
        // codec params, so the decoder doesn't keep borrowing the stream.
        let (stream_index, fps, decoder) = {
            let stream = input
                .streams()
                .best(ffmpeg::media::Type::Video)
                .context("file has no image/video stream")?;
            let rate = stream.avg_frame_rate();
            let fps = if rate.denominator() != 0 && rate.numerator() != 0 {
                f64::from(rate.numerator()) / f64::from(rate.denominator())
            } else {
                30.0
            }
            .clamp(1.0, 240.0);
            let decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
                .context("build decoder context")?
                .decoder()
                .video()
                .context("open video decoder")?;
            (stream.index(), fps, decoder)
        };

        let (dst_w, dst_h) = fit_within(decoder.width(), decoder.height(), max_dim);
        let scaler = ffmpeg::software::scaling::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg::format::Pixel::RGBA,
            dst_w,
            dst_h,
            ffmpeg::software::scaling::Flags::BILINEAR,
        )
        .context("create RGBA scaler")?;

        Ok(Self {
            input,
            stream_index,
            video: decoder,
            scaler,
            fps,
        })
    }
}

/// Pull one decoded frame (if ready), scale it to RGBA, pack it tightly.
fn next_rgba(
    decoder: &mut ffmpeg::decoder::Video,
    scaler: &mut ffmpeg::software::scaling::Context,
) -> Option<Frame> {
    let mut src = ffmpeg::frame::Video::empty();
    if decoder.receive_frame(&mut src).is_err() {
        return None;
    }
    let mut rgba = ffmpeg::frame::Video::empty();
    scaler.run(&src, &mut rgba).ok()?;
    Some(pack(&rgba))
}

/// Copy a (possibly stride-padded) RGBA frame into a tight `w*h*4` buffer.
fn pack(frame: &ffmpeg::frame::Video) -> Frame {
    let (width, height) = (frame.width(), frame.height());
    let stride = frame.stride(0);
    let data = frame.data(0);
    let row = width as usize * 4;
    let mut rgba = Vec::with_capacity(row * height as usize);
    for y in 0..height as usize {
        let start = y * stride;
        rgba.extend_from_slice(&data[start..start + row]);
    }
    Frame {
        width,
        height,
        rgba,
    }
}

/// Largest `w×h`-aspect size fitting in a `max`×`max` box; never upscales.
/// `max == 0` disables the cap.
fn fit_within(w: u32, h: u32, max: u32) -> (u32, u32) {
    let (w, h) = (w.max(1), h.max(1));
    if max == 0 || (w <= max && h <= max) {
        return (w, h);
    }
    let scale = f64::from(max) / f64::from(w.max(h));
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scaled dims are positive and smaller than the source, well within u32"
    )]
    let dims = (
        ((f64::from(w) * scale) as u32).max(1),
        ((f64::from(h) * scale) as u32).max(1),
    );
    dims
}
