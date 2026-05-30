//! Media wallpaper decoding via libav (`FFmpeg`).
//!
//! One entry point for now: [`decode_first_frame`] turns any file `FFmpeg`
//! can open (image, gif, video, …) into a single packed-RGBA frame the
//! renderer uploads as the wallpaper texture. Animated playback (a decode
//! thread feeding frames over time) will build on the same primitives.

use std::path::Path;

use anyhow::{Context as _, Result, bail};
use ffmpeg_the_third as ffmpeg;

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
    // Idempotent; cheap to call before every decode.
    ffmpeg::init().context("FFmpeg init")?;

    let mut input = ffmpeg::format::input(path)
        .with_context(|| format!("FFmpeg could not open {}", path.display()))?;

    // Scope the immutable stream borrow so `input.packets()` (a mutable
    // borrow) is free below. `from_parameters` copies the codec params, so
    // the decoder doesn't keep borrowing the stream.
    let (stream_index, mut decoder) = {
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .context("file has no image/video stream")?;
        let decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .context("build decoder context")?
            .decoder()
            .video()
            .context("open video decoder")?;
        (stream.index(), decoder)
    };

    let (dst_w, dst_h) = fit_within(decoder.width(), decoder.height(), max_dim);
    let mut scaler = ffmpeg::software::scaling::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg::format::Pixel::RGBA,
        dst_w,
        dst_h,
        ffmpeg::software::scaling::Flags::BILINEAR,
    )
    .context("create RGBA scaler")?;

    for item in input.packets() {
        let (stream, packet) = item.context("read packet")?;
        if stream.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet).context("send packet")?;
        if let Some(frame) = next_rgba(&mut decoder, &mut scaler) {
            return Ok(frame);
        }
    }
    // Flush a frame the decoder may still be holding (single-image inputs
    // often emit theirs only on EOF).
    decoder.send_eof().context("flush decoder")?;
    if let Some(frame) = next_rgba(&mut decoder, &mut scaler) {
        return Ok(frame);
    }
    bail!("no decodable frame in {}", path.display())
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
