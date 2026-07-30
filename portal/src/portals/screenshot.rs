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

/// Compose every output's capture into one image, laid out the way the
/// compositor lays out the monitors.
///
/// A single-output desktop skips the copy entirely; a multi-head one gets the
/// whole desktop, which is what an app asking for "a screenshot" with no
/// further qualification means.
fn stitch(frames: &[Frame], positions: &[(i32, i32)]) -> Option<Frame> {
    if frames.len() == 1 {
        return frames.first().cloned();
    }
    let bounds = frames.iter().zip(positions).fold(
        None::<(i32, i32, i32, i32)>,
        |acc, (frame, (x, y))| {
            let rect = (*x, *y, x + frame.width, y + frame.height);
            Some(acc.map_or(rect, |(x0, y0, x1, y1)| {
                (x0.min(rect.0), y0.min(rect.1), x1.max(rect.2), y1.max(rect.3))
            }))
        },
    )?;
    let (x0, y0, x1, y1) = bounds;
    let (width, height) = ((x1 - x0).max(1), (y1 - y0).max(1));
    let stride = width as usize * 4;
    // Opaque black behind the monitors, so gaps in an irregular layout read as
    // background rather than as garbage.
    let mut data = vec![0u8; stride * height as usize];
    for chunk in data.chunks_exact_mut(4) {
        chunk[3] = 255;
    }
    for (frame, (x, y)) in frames.iter().zip(positions) {
        for row in 0..frame.height as usize {
            let dst_y = (y - y0) as usize + row;
            let dst_x = (x - x0) as usize;
            let dst_start = dst_y * stride + dst_x * 4;
            let src_start = row * frame.stride;
            let len = (frame.width as usize * 4).min(stride.saturating_sub(dst_x * 4));
            let (Some(src), Some(dst)) = (
                frame.data.get(src_start..src_start + len),
                data.get_mut(dst_start..dst_start + len),
            ) else {
                continue;
            };
            dst.copy_from_slice(src);
        }
    }
    Some(Frame {
        width,
        height,
        stride,
        data,
    })
}

/// Capture every output on a blocking thread (the Wayland round-trips are
/// synchronous) and hand back the frames plus their layout positions.
async fn capture_desktop(cursor: bool) -> anyhow::Result<(Vec<Frame>, Vec<(i32, i32)>)> {
    tokio::task::spawn_blocking(move || {
        let mut capturer = Capturer::new()?;
        let frames = capturer.capture_all(cursor)?;
        let positions = capturer.output_positions();
        Ok((frames, positions))
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
            let (frames, positions) = match capture_desktop(cursor).await {
                Ok(captured) => captured,
                Err(err) => {
                    tracing::error!(%err, "capture failed");
                    return empty(FAILED);
                }
            };
            if frames.is_empty() {
                return empty(FAILED);
            }

            let image = if interactive {
                let picker = RegionPicker::new(frames.clone());
                let picker = match ui::overlay(picker, cancel).await {
                    Ok(picker) => picker,
                    Err(err) => {
                        tracing::error!(%err, "region selector failed");
                        return empty(FAILED);
                    }
                };
                if let Some(region) = picker.region {
                    let Some(frame) = frames.get(region.output) else {
                        return empty(FAILED);
                    };
                    frame.crop(region.rect)
                } else if let Some(output) = picker.whole_output {
                    let Some(frame) = frames.get(output) else {
                        return empty(FAILED);
                    };
                    frame.clone()
                } else {
                    return empty(CANCELLED);
                }
            } else {
                let Some(image) = stitch(&frames, &positions) else {
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
            let (frames, _) = match capture_desktop(false).await {
                Ok(captured) => captured,
                Err(err) => {
                    tracing::error!(%err, "capture failed");
                    return empty(FAILED);
                }
            };
            match ui::overlay(ColorPicker::new(frames), cancel).await {
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
