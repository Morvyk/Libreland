//! DRM/KMS device + per-output surface initialisation.
//!
//! Opens the DRM device via libseat, enumerates *every* connected
//! output (not just the first one), picks a compatible CRTC for each
//! while tracking which ones have already been assigned, and creates
//! a `DrmSurface` per output. The renderer in [`crate::render`]
//! consumes the surface vec to build a single GBM-backed GLES
//! pipeline shared across outputs.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context as _, Result};
use smithay::backend::drm::{DrmDevice, DrmDeviceFd, DrmDeviceNotifier, DrmSurface};
use smithay::backend::session::Session as _;
use smithay::backend::session::libseat::LibSeatSession;
use smithay::reexports::drm;
use smithay::reexports::drm::control::dumbbuffer::DumbBuffer;
use smithay::reexports::drm::control::{
    Device as ControlDevice, Mode, ModeTypeFlags, connector, crtc,
};
use smithay::reexports::rustix::fs::OFlags;
use smithay::utils::DeviceFd;
use tracing::{info, warn};

use crate::config::MonitorsConfig;

/// Top-level result of [`open_display`]. Holds the master-claim
/// device, the fd (Arc-backed; clone for additional owners), the
/// calloop notifier, and one [`DrmOutput`] per connected output.
pub struct DrmInit {
    pub device: DrmDevice,
    pub fd: DrmDeviceFd,
    pub notifier: DrmDeviceNotifier,
    pub outputs: Vec<DrmOutput>,
}

/// One physical output's worth of DRM resources.
pub struct DrmOutput {
    /// Kernel connector name (`"HDMI-A-1"`, `"DP-1"`, `"eDP-1"`, …),
    /// formatted via drm-rs's `connector::Info` Display impl. This
    /// is the stable identifier the user puts in Lua config to
    /// target a specific output.
    pub name: String,
    /// CRTC driving this surface. Vblank events arrive tagged with
    /// the CRTC, so the renderer uses this to look up which output
    /// to re-render.
    pub crtc: crtc::Handle,
    /// Connector this output scans out on. Kept so the renderer can
    /// query `vrr_capable` (adaptive-sync support is a connector
    /// property, not a CRTC one) before deciding whether VRR is usable.
    pub connector: connector::Handle,
    /// CRTC + connector + mode tuple, consumed by the renderer to
    /// build a GBM swapchain over this output.
    pub surface: DrmSurface,
    /// Mode in use. Renderer reads its size for framebuffer
    /// dimensions; the user-facing name is just the connector.
    pub mode: Mode,
}

/// Open the DRM device at `path` through `session`, enumerate every
/// connected output, and bind a `DrmSurface` to each. Returns an
/// error only if literally no outputs could be brought up; a
/// connector that fails individually (no compatible free CRTC,
/// no modes, query error) is logged and skipped.
///
/// `monitors` carries optional per-connector mode overrides. For
/// each connector that finds a matching entry in
/// `monitors.outputs[name].mode`, the requested
/// `(width, height, refresh_mHz)` is matched against the EDID
/// mode list; the first match wins. Unmatched overrides fall back
/// to the EDID-preferred mode with a warning so the user knows
/// their request didn't take.
pub fn open_display(
    session: &mut LibSeatSession,
    path: &Path,
    monitors: &MonitorsConfig,
) -> Result<DrmInit> {
    info!(path = %path.display(), "phase: opening DRM device via libseat");
    let owned_fd = session
        .open(path, OFlags::RDWR | OFlags::NONBLOCK)
        .context("libseat refused to open the DRM device")?;
    let fd = DrmDeviceFd::new(DeviceFd::from(owned_fd));
    info!("DRM fd acquired");

    info!("phase: initialising DrmDevice");
    let (mut device, notifier) =
        DrmDevice::new(fd.clone(), false).context("DrmDevice::new failed")?;
    info!(atomic = device.is_atomic(), "DrmDevice initialised");

    let resources = device
        .resource_handles()
        .context("failed to read DRM resource handles")?;

    info!("phase: enumerating connectors");
    let mut outputs = Vec::new();
    let mut used_crtcs: HashSet<crtc::Handle> = HashSet::new();

    for &conn_handle in resources.connectors() {
        let conn_info = match device.get_connector(conn_handle, false) {
            Ok(info) => info,
            Err(err) => {
                warn!(?err, ?conn_handle, "failed to query connector — skipping");
                continue;
            }
        };
        if conn_info.state() != connector::State::Connected {
            continue;
        }
        // A connector that finds no mode or no free CRTC is skipped
        // (Ok(None)); a surface-creation failure aborts the launch (the
        // `?`), matching the original strictness for startup.
        if let Some(output) =
            build_output(&mut device, &conn_info, conn_handle, &resources, monitors, &used_crtcs)?
        {
            used_crtcs.insert(output.crtc);
            outputs.push(output);
        }
    }

    if outputs.is_empty() {
        anyhow::bail!("no connected outputs with available CRTCs — nothing to drive");
    }
    info!(count = outputs.len(), "all connected outputs bound");

    // Wipe each CRTC's cursor plane inherited from the display manager so
    // its pointer doesn't linger as a ghost over our scene (see fn docs).
    clear_leftover_cursors(&device, &outputs);

    Ok(DrmInit {
        device,
        fd,
        notifier,
        outputs,
    })
}

/// Outcome of a live connector re-scan ([`rescan_connectors`]).
pub struct RescanResult {
    /// Connector names currently in the `Connected` state — the caller
    /// diffs this against the outputs it already drives to find which
    /// ones were unplugged.
    pub connected: Vec<String>,
    /// Newly-connected outputs (not among `existing`) brought up with a
    /// fresh CRTC + surface, ready to hand to the renderer.
    pub added: Vec<DrmOutput>,
}

/// Re-enumerate `device`'s connectors after a udev "changed" event
/// (monitor plugged or unplugged) and bind a surface for every
/// connected output whose connector name is not already in `existing`.
/// `used_crtcs` are the CRTCs the compositor's current outputs hold, so
/// a freshly-plugged monitor is assigned a genuinely free one. Already-
/// known connectors are reported in `connected` (so the caller can spot
/// removals) without being rebuilt. Unlike [`open_display`] this never
/// aborts: a connector that fails to bind is logged and skipped, because
/// a hotplug must not bring the whole session down.
pub fn rescan_connectors(
    device: &mut DrmDevice,
    monitors: &MonitorsConfig,
    existing: &HashSet<String>,
    used_crtcs: &HashSet<crtc::Handle>,
) -> Result<RescanResult> {
    let resources = device
        .resource_handles()
        .context("failed to read DRM resource handles on rescan")?;
    let mut connected = Vec::new();
    let mut added = Vec::new();
    let mut used = used_crtcs.clone();

    for &conn_handle in resources.connectors() {
        let conn_info = match device.get_connector(conn_handle, false) {
            Ok(info) => info,
            Err(err) => {
                warn!(?err, ?conn_handle, "failed to query connector on rescan — skipping");
                continue;
            }
        };
        if conn_info.state() != connector::State::Connected {
            continue;
        }
        let name = conn_info.to_string();
        connected.push(name.clone());
        if existing.contains(&name) {
            continue;
        }
        match build_output(device, &conn_info, conn_handle, &resources, monitors, &used) {
            Ok(Some(output)) => {
                used.insert(output.crtc);
                added.push(output);
            }
            Ok(None) => {}
            Err(err) => {
                warn!(connector = %name, error = %err, "failed to bind hot-plugged output — skipping");
            }
        }
    }

    Ok(RescanResult { connected, added })
}

/// Build one [`DrmOutput`] for an already-known-connected connector:
/// pick its mode (honouring any config override), allocate a CRTC not in
/// `used_crtcs`, and create the surface. `Ok(None)` means the connector
/// has no usable mode or no free CRTC (skip it); `Err` means surface
/// creation failed.
fn build_output(
    device: &mut DrmDevice,
    conn_info: &connector::Info,
    conn_handle: connector::Handle,
    resources: &drm::control::ResourceHandles,
    monitors: &MonitorsConfig,
    used_crtcs: &HashSet<crtc::Handle>,
) -> Result<Option<DrmOutput>> {
    let name = conn_info.to_string();

    let requested_mode = monitors.outputs.get(&name).and_then(|cfg| cfg.mode);
    let Some(mode) = pick_mode(conn_info, requested_mode, &name) else {
        warn!(connector = %name, "connector reports no modes — skipping");
        return Ok(None);
    };
    let (mode_w, mode_h) = mode.size();

    let Some(crtc) = pick_unused_crtc(device, conn_info, resources, used_crtcs) else {
        warn!(
            connector = %name,
            "no unused CRTC compatible with this connector — skipping"
        );
        return Ok(None);
    };

    let surface = device
        .create_surface(crtc, mode, &[conn_handle])
        .with_context(|| format!("DrmDevice::create_surface failed for {name}"))?;
    info!(
        connector = %name,
        crtc = ?crtc,
        width = mode_w,
        height = mode_h,
        refresh = mode.vrefresh(),
        legacy = surface.is_legacy(),
        "output bound"
    );

    Ok(Some(DrmOutput {
        name,
        crtc,
        connector: conn_handle,
        surface,
        mode,
    }))
}

/// Rebuild a [`DrmOutput`] for an *already-connected* connector on a
/// specific, now-free CRTC, honouring the current mode override in
/// `monitors`. Used for live mode changes on config reload: the caller
/// drops the output's old surface (which frees its CRTC) and hands that
/// same CRTC back here, so the modeset stays on the connector's existing
/// pipe and no other output is disturbed. Unlike [`build_output`] this
/// does not search for a free CRTC — it reuses the one given.
pub fn rebuild_output_mode(
    device: &mut DrmDevice,
    connector: connector::Handle,
    crtc: crtc::Handle,
    monitors: &MonitorsConfig,
) -> Result<DrmOutput> {
    let conn_info = device
        .get_connector(connector, false)
        .context("failed to query connector for live mode rebuild")?;
    let name = conn_info.to_string();
    let requested = monitors.outputs.get(&name).and_then(|cfg| cfg.mode);
    let mode = pick_mode(&conn_info, requested, &name)
        .with_context(|| format!("{name} reports no usable mode for rebuild"))?;
    let (mode_w, mode_h) = mode.size();
    let surface = device
        .create_surface(crtc, mode, &[connector])
        .with_context(|| format!("DrmDevice::create_surface failed rebuilding {name}"))?;
    info!(
        connector = %name,
        crtc = ?crtc,
        width = mode_w,
        height = mode_h,
        refresh = mode.vrefresh(),
        "output mode rebuilt"
    );
    Ok(DrmOutput {
        name,
        crtc,
        connector,
        surface,
        mode,
    })
}

/// Clear any hardware cursor the display manager left on each CRTC's
/// cursor plane before we took DRM master. Libreland composites its own
/// (GPU-rendered) cursor into the primary framebuffer and never programs
/// the KMS cursor plane, so without this the DM's last cursor image keeps
/// scanning out as a frozen "ghost" over our scene.
///
/// The legacy `set_cursor` ioctl is the simplest portable way to *disable*
/// it: on atomic drivers (including NVIDIA) the kernel routes the legacy
/// call through its universal-cursor path, which disables the cursor
/// plane — so this works without us implementing atomic plane programming
/// we'd otherwise never need. Per-CRTC failures are non-fatal (at worst
/// the ghost remains), so they're logged and ignored.
fn clear_leftover_cursors(device: &DrmDevice, outputs: &[DrmOutput]) {
    for output in outputs {
        #[allow(
            deprecated,
            reason = "drm-rs deprecates set_cursor in favour of programming a cursor plane; we deliberately don't use the plane, and this is the portable way to *disable* the one the DM left behind"
        )]
        let cleared = ControlDevice::set_cursor(device, output.crtc, None::<&DumbBuffer>);
        if let Err(err) = cleared {
            warn!(error = %err, crtc = ?output.crtc, "could not clear the DM's leftover cursor");
        }
    }
}

/// Pick a mode for `conn`. If `requested` is `Some`, look for an
/// EDID entry whose size and refresh rate match (refresh compared
/// in milli-Hz to allow rates like 59.94 vs 60.00 to disambiguate).
/// On a miss, fall back to the EDID-preferred mode and log a
/// warning so the user knows the override didn't take.
fn pick_mode(
    conn: &connector::Info,
    requested: Option<(u32, u32, u32)>,
    name: &str,
) -> Option<Mode> {
    if let Some((req_w, req_h, req_mhz)) = requested {
        let matched = conn.modes().iter().find(|m| {
            let (mw, mh) = m.size();
            u32::from(mw) == req_w && u32::from(mh) == req_h && m.vrefresh() * 1000 == req_mhz
        });
        if let Some(m) = matched {
            info!(
                connector = %name,
                width = req_w,
                height = req_h,
                refresh_mhz = req_mhz,
                "mode override matched"
            );
            return Some(*m);
        }
        warn!(
            connector = %name,
            width = req_w,
            height = req_h,
            refresh_mhz = req_mhz,
            "mode override didn't match any advertised mode — falling back to EDID preferred"
        );
    }
    conn.modes()
        .iter()
        .find(|m| m.mode_type().contains(ModeTypeFlags::PREFERRED))
        .copied()
        .or_else(|| conn.modes().first().copied())
}

/// Find a CRTC that (a) can drive `conn` and (b) isn't already in
/// `used`. Prefers the kernel's existing encoder/CRTC pair when it's
/// both compatible and still free — that avoids an unnecessary
/// remap. Falls back to iterating the connector's possible encoders
/// and filtering their possible CRTCs against `used`.
fn pick_unused_crtc(
    device: &DrmDevice,
    conn: &connector::Info,
    resources: &drm::control::ResourceHandles,
    used: &HashSet<crtc::Handle>,
) -> Option<crtc::Handle> {
    if let Some(encoder_handle) = conn.current_encoder()
        && let Ok(encoder) = device.get_encoder(encoder_handle)
        && let Some(crtc) = encoder.crtc()
        && !used.contains(&crtc)
    {
        return Some(crtc);
    }

    conn.encoders().iter().find_map(|&encoder_handle| {
        let encoder = device.get_encoder(encoder_handle).ok()?;
        resources
            .filter_crtcs(encoder.possible_crtcs())
            .iter()
            .find(|c| !used.contains(c))
            .copied()
    })
}
