//! `org.freedesktop.impl.portal.Screenshot` — screenshots and the colour
//! picker.
//!
//! Both ride the compositor's own `zwlr_screencopy_v1` (see
//! [`crate::capture`]), so what an app gets through the portal is the same
//! pixels `grim` or the compositor's built-in screenshot key would produce.
//!
//! Interactive mode freezes the desktop first — every output is captured, the
//! overlay draws those frames, and the region is cut out of them. Selecting
//! against a live desktop means the thing you were pointing at can move while
//! you drag; selecting against a freeze means what you framed is what you get.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel geometry: every value here is a surface- or image-sized non-negative integer, and the conversions between i32/u32/usize/f32 are all inside that range. Checked conversions at each site would be noise around arithmetic that cannot overflow."
)]

use std::collections::HashMap;
use std::path::PathBuf;

use zbus::zvariant::{OwnedObjectPath, OwnedValue};
use zbus::{Connection, interface};

use crate::capture::{Capturer, Frame};
use crate::ui;
use crate::ui::picker::{ColorPicker, RegionPicker};

use super::{
    CANCELLED, FAILED, PortalResult, SUCCESS, empty, opt_bool, ov, path_to_uri, with_request,
};

pub struct Screenshot;

impl Screenshot {
    pub const fn new() -> Self {
        Self
    }
}

/// Where a screenshot lands: the user's Pictures/Screenshots if that exists,
/// else the cache directory.
///
/// The frontend copies the file into the app's sandbox and the app is expected
/// to treat it as its own; leaving a copy somewhere the user can find it is
/// deliberate, because "I took a screenshot and the app lost it" is the
/// failure people actually hit.
fn output_path() -> PathBuf {
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()));
    let pictures = std::env::var("XDG_PICTURES_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .map_or_else(|| home.join("Pictures"), PathBuf::from);
    let dir = if pictures.is_dir() {
        let shots = pictures.join("Screenshots");
        if std::fs::create_dir_all(&shots).is_ok() {
            shots
        } else {
            pictures
        }
    } else {
        let cache = std::env::var("XDG_CACHE_HOME")
            .ok()
            .filter(|s| !s.is_empty())
            .map_or_else(|| home.join(".cache"), PathBuf::from)
            .join("libreland-portal");
        let _ = std::fs::create_dir_all(&cache);
        cache
    };
    // Seconds since the epoch keeps names unique and sortable without pulling
    // in a date formatter for a filename.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    dir.join(format!("Screenshot_{stamp}.png"))
}

/// Where one output's capture goes in the stitched image, in the canvas
/// pixel space chosen by [`stitch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tile {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

/// Lay the outputs out in one coherent pixel space.
///
/// The two numbers a capture comes with are in *different* spaces: a frame's
/// size is physical (`wl_output.mode`) while its position is logical
/// (`xdg_output.logical_position`, the compositor's layout). Adding them —
/// which is what this used to do — only works when every output is at a
/// scale of one. Otherwise an output occupies fewer layout units than its
/// frame has pixels, and the next output is pasted over the top of it.
///
/// So the canvas is the logical layout scaled by `k`, the largest scale
/// factor in play. That choice means the sharpest output maps 1:1 and is
/// never resampled; the rest are only ever scaled *up*, so no monitor's
/// detail is thrown away. With every output at the same scale (including the
/// common all-1.0 case) every tile is 1:1 and this is a plain copy.
fn tiles_for(outputs: &[(i32, i32, i32, i32)], frames: &[Frame]) -> Vec<Tile> {
    // (logical x, y, w, h) per output; `k` = max physical/logical ratio.
    let k = outputs
        .iter()
        .zip(frames)
        .filter(|((_, _, lw, _), _)| *lw > 0)
        .map(|((_, _, lw, _), f)| f64::from(f.width) / f64::from(*lw))
        .fold(1.0_f64, f64::max);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "logical coordinates times a small scale factor: display pixels, far inside i32"
    )]
    let px = |v: i32| (f64::from(v) * k).round() as i32;
    outputs
        .iter()
        .zip(frames)
        .map(|(&(x, y, lw, lh), f)| {
            // Fall back to the frame's own size when the compositor gave us
            // no logical size — that is the scale-1 reading, and the only
            // sane guess.
            let (lw, lh) = if lw > 0 && lh > 0 {
                (lw, lh)
            } else {
                (f.width, f.height)
            };
            Tile {
                x: px(x),
                y: px(y),
                w: px(lw).max(1),
                h: px(lh).max(1),
            }
        })
        .collect()
}

/// Compose every output's capture into one image, laid out the way the
/// compositor lays out the monitors.
///
/// A single-output desktop skips the copy entirely; a multi-head one gets the
/// whole desktop, which is what an app asking for "a screenshot" with no
/// further qualification means. `outputs` is `(logical x, y, w, h)` per
/// frame — see [`tiles_for`] for why the logical size is needed and not just
/// the position.
fn stitch(frames: &[Frame], outputs: &[(i32, i32, i32, i32)]) -> Option<Frame> {
    if frames.len() == 1 {
        return frames.first().cloned();
    }
    let tiles = tiles_for(outputs, frames);
    let bounds = tiles.iter().fold(None::<(i32, i32, i32, i32)>, |acc, t| {
        let rect = (t.x, t.y, t.x + t.w, t.y + t.h);
        Some(acc.map_or(rect, |(x0, y0, x1, y1)| {
            (
                x0.min(rect.0),
                y0.min(rect.1),
                x1.max(rect.2),
                y1.max(rect.3),
            )
        }))
    })?;
    let (x0, y0, x1, y1) = bounds;
    let (width, height) = ((x1 - x0).max(1), (y1 - y0).max(1));
    let stride = width as usize * 4;
    // Opaque black behind the monitors, so gaps in an irregular layout read as
    // background rather than as garbage.
    let mut data = vec![0u8; stride * height as usize];
    for chunk in data.chunks_exact_mut(4) {
        chunk[3] = 255;
    }
    let mut canvas = StitchCanvas {
        data: &mut data,
        stride,
        width,
        height,
        origin: (x0, y0),
    };
    for (frame, tile) in frames.iter().zip(&tiles) {
        canvas.blit_scaled(frame, *tile);
    }
    Some(Frame {
        width,
        height,
        stride,
        data,
    })
}

/// The image [`stitch`] is composing, plus where its top-left sits in tile
/// coordinates (the union's corner, which is negative when a monitor is
/// placed left of or above the primary).
struct StitchCanvas<'a> {
    data: &'a mut [u8],
    stride: usize,
    width: i32,
    height: i32,
    origin: (i32, i32),
}

impl StitchCanvas<'_> {
    /// Copy `frame` in at `tile`, resampling if the tile is not the frame's
    /// own size. Nearest-neighbour: [`tiles_for`] guarantees the tile is never
    /// *smaller* than the frame, so this only ever duplicates pixels. At 1:1
    /// nearest is an exact copy, and the case where it would alias
    /// (downscaling) cannot arise.
    fn blit_scaled(&mut self, frame: &Frame, tile: Tile) {
        if frame.width <= 0 || frame.height <= 0 {
            return;
        }
        let (x0, y0) = self.origin;
        let step = (
            f64::from(frame.width) / f64::from(tile.w.max(1)),
            f64::from(frame.height) / f64::from(tile.h.max(1)),
        );
        for row in 0..tile.h {
            let dy = tile.y - y0 + row;
            if dy < 0 || dy >= self.height {
                continue;
            }
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "row/col are bounded by the tile, and the step maps them back inside the frame"
            )]
            let src_row =
                ((f64::from(row) * step.1) as i32).min(frame.height - 1) as usize * frame.stride;
            for col in 0..tile.w {
                let dx = tile.x - x0 + col;
                if dx < 0 || dx >= self.width {
                    continue;
                }
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "see above"
                )]
                let src_col = ((f64::from(col) * step.0) as i32).min(frame.width - 1) as usize * 4;
                let s = src_row + src_col;
                let d = dy as usize * self.stride + dx as usize * 4;
                let (Some(src), Some(slot)) =
                    (frame.data.get(s..s + 4), self.data.get_mut(d..d + 4))
                else {
                    continue;
                };
                slot.copy_from_slice(src);
            }
        }
    }
}

/// One frozen capture of the whole desktop.
struct Desktop {
    /// `(connector name, capture)`, in the capturer's own order. The name
    /// travels with the frame because the *overlay* orders its surfaces by
    /// layout position while the capturer orders by connector name; anything
    /// crossing between them has to be keyed, not indexed.
    frames: Vec<(String, Frame)>,
    /// Logical layout rect `(x, y, w, h)` per frame, parallel to `frames`.
    rects: Vec<(i32, i32, i32, i32)>,
}

/// Capture every output on a blocking thread (the Wayland round-trips are
/// synchronous).
async fn capture_desktop(cursor: bool) -> anyhow::Result<Desktop> {
    tokio::task::spawn_blocking(move || {
        let mut capturer = Capturer::new()?;
        let names: Vec<String> = capturer.outputs().into_iter().map(|o| o.name).collect();
        let rects = capturer.output_layout_rects();
        let frames = capturer.capture_all(cursor)?;
        Ok(Desktop {
            frames: names.into_iter().zip(frames).collect(),
            rects,
        })
    })
    .await?
}

#[interface(name = "org.freedesktop.impl.portal.Screenshot")]
impl Screenshot {
    async fn screenshot(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        let interactive = opt_bool(&options, "interactive").unwrap_or(false);
        // Not in the spec, but harmless to honour if an app asks for it.
        let cursor = opt_bool(&options, "include-cursor").unwrap_or(false);
        tracing::info!(app = %app_id, interactive, "screenshot");

        with_request(conn, &handle, |cancel| async move {
            let desktop = match capture_desktop(cursor).await {
                Ok(captured) => captured,
                Err(err) => {
                    tracing::error!(%err, "capture failed");
                    return empty(FAILED);
                }
            };
            if desktop.frames.is_empty() {
                return empty(FAILED);
            }
            let named = |name: &str| {
                desktop
                    .frames
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, f)| f)
            };

            let image = if interactive {
                let picker = RegionPicker::new(desktop.frames.clone());
                let picker = match ui::overlay(picker, cancel).await {
                    Ok(picker) => picker,
                    Err(err) => {
                        tracing::error!(%err, "region selector failed");
                        return empty(FAILED);
                    }
                };
                if let Some(region) = &picker.region {
                    let Some(frame) = named(&region.output) else {
                        return empty(FAILED);
                    };
                    frame.crop(region.rect)
                } else if let Some(output) = &picker.whole_output {
                    let Some(frame) = named(output) else {
                        return empty(FAILED);
                    };
                    frame.clone()
                } else {
                    return empty(CANCELLED);
                }
            } else {
                let frames: Vec<Frame> = desktop.frames.iter().map(|(_, f)| f.clone()).collect();
                let Some(image) = stitch(&frames, &desktop.rects) else {
                    return empty(FAILED);
                };
                image
            };

            let path = output_path();
            if let Err(err) = image.write_png(&path) {
                tracing::error!(%err, path = %path.display(), "could not write the screenshot");
                return empty(FAILED);
            }
            tracing::info!(path = %path.display(), "screenshot saved");
            let mut results = HashMap::new();
            results.insert("uri".to_string(), ov(path_to_uri(&path).as_str()));
            (SUCCESS, results)
        })
        .await
    }

    async fn pick_color(
        &self,
        #[zbus(connection)] conn: &Connection,
        handle: OwnedObjectPath,
        app_id: String,
        _parent_window: String,
        _options: HashMap<String, OwnedValue>,
    ) -> PortalResult {
        tracing::info!(app = %app_id, "pick colour");
        with_request(conn, &handle, |cancel| async move {
            let desktop = match capture_desktop(false).await {
                Ok(captured) => captured,
                Err(err) => {
                    tracing::error!(%err, "capture failed");
                    return empty(FAILED);
                }
            };
            match ui::overlay(ColorPicker::new(desktop.frames), cancel).await {
                Ok(picker) => picker.picked.map_or_else(
                    || empty(CANCELLED),
                    |rgb| {
                        let mut results = HashMap::new();
                        // `(ddd)`, each channel 0.0–1.0.
                        results.insert("color".to_string(), ov(rgb));
                        (SUCCESS, results)
                    },
                ),
                Err(err) => {
                    tracing::error!(%err, "colour picker failed");
                    empty(FAILED)
                }
            }
        })
        .await
    }

    /// Bit 1 = we can screenshot the whole screen. Window and region targets
    /// are the *frontend's* enumeration for its own UI; interactive region
    /// selection is offered through the `interactive` option instead.
    #[zbus(property, name = "AvailableTargets")]
    fn available_targets(&self) -> u32 {
        1
    }

    #[zbus(property, name = "version")]
    fn version(&self) -> u32 {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::{Frame, Tile, tiles_for};

    fn frame(w: i32, h: i32) -> Frame {
        Frame {
            width: w,
            height: h,
            stride: w as usize * 4,
            data: vec![0; (w * h) as usize * 4],
        }
    }

    /// Every monitor at scale 1: the layout is already in the frames' own
    /// space, so each tile is exactly the frame and nothing is resampled.
    #[test]
    fn unscaled_outputs_tile_one_to_one() {
        let outs = [(0, 0, 1920, 1080), (1920, 0, 1920, 1080)];
        let frames = [frame(1920, 1080), frame(1920, 1080)];
        assert_eq!(
            tiles_for(&outs, &frames),
            [
                Tile {
                    x: 0,
                    y: 0,
                    w: 1920,
                    h: 1080
                },
                Tile {
                    x: 1920,
                    y: 0,
                    w: 1920,
                    h: 1080
                },
            ]
        );
    }

    /// The bug this exists for. A 4K at scale 1.5 is 3840 px wide but only
    /// 2560 layout units, so adding its *physical* width to the next
    /// output's *logical* position pasted the second monitor over the first.
    /// In the scaled canvas the two must not overlap.
    #[test]
    fn a_fractionally_scaled_output_does_not_get_overwritten() {
        let outs = [(0, 0, 2560, 1440), (2560, 0, 1920, 1080)];
        let frames = [frame(3840, 2160), frame(1920, 1080)];
        let tiles = tiles_for(&outs, &frames);
        // k = 1.5 (the 4K's ratio), so the 4K is 1:1 and keeps every pixel.
        assert_eq!(
            tiles[0],
            Tile {
                x: 0,
                y: 0,
                w: 3840,
                h: 2160
            }
        );
        // The 1080p starts exactly where the 4K ends, not 1280 px inside it.
        assert_eq!(tiles[1].x, 3840);
        assert_eq!(tiles[0].x + tiles[0].w, tiles[1].x);
        // Only ever scaled up, never down: no monitor loses detail.
        assert!(tiles[1].w >= frames[1].width);
    }

    /// No `xdg_output` (logical size 0) is read as scale 1, which is both the
    /// safe guess and the correct one on a compositor that only speaks
    /// `wl_output`.
    #[test]
    fn a_missing_logical_size_falls_back_to_the_frame() {
        let outs = [(0, 0, 0, 0), (1920, 0, 0, 0)];
        let frames = [frame(1920, 1080), frame(1920, 1080)];
        assert_eq!(
            tiles_for(&outs, &frames),
            [
                Tile {
                    x: 0,
                    y: 0,
                    w: 1920,
                    h: 1080
                },
                Tile {
                    x: 1920,
                    y: 0,
                    w: 1920,
                    h: 1080
                },
            ]
        );
    }

    /// A monitor above or left of the primary has a negative layout origin;
    /// the tiles keep it, and `stitch` shifts by the union's corner.
    #[test]
    fn negative_positions_survive_the_scaling() {
        let outs = [(-1920, 0, 1920, 1080), (0, 0, 2560, 1440)];
        let frames = [frame(1920, 1080), frame(3840, 2160)];
        let tiles = tiles_for(&outs, &frames);
        assert_eq!(tiles[0].x, -2880); // -1920 * 1.5
        assert_eq!(tiles[1].x, 0);
        assert_eq!(tiles[0].x + tiles[0].w, tiles[1].x);
    }
}
