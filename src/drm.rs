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
use smithay::reexports::drm::control::{
    Device as ControlDevice, Mode, ModeTypeFlags, connector, crtc,
};
use smithay::reexports::rustix::fs::OFlags;
use smithay::utils::DeviceFd;
use tracing::{info, warn};

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
pub fn open_display(session: &mut LibSeatSession, path: &Path) -> Result<DrmInit> {
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
        let name = conn_info.to_string();

        let Some(mode) = pick_preferred_mode(&conn_info) else {
            warn!(connector = %name, "connector reports no modes — skipping");
            continue;
        };
        let (mode_w, mode_h) = mode.size();

        let Some(crtc) = pick_unused_crtc(&device, &conn_info, &resources, &used_crtcs) else {
            warn!(
                connector = %name,
                "no unused CRTC compatible with this connector — skipping"
            );
            continue;
        };
        used_crtcs.insert(crtc);

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

        outputs.push(DrmOutput {
            name,
            crtc,
            surface,
            mode,
        });
    }

    if outputs.is_empty() {
        anyhow::bail!("no connected outputs with available CRTCs — nothing to drive");
    }
    info!(count = outputs.len(), "all connected outputs bound");

    Ok(DrmInit {
        device,
        fd,
        notifier,
        outputs,
    })
}

/// Pick the connector's preferred mode if the EDID advertises one,
/// falling back to the first advertised mode.
fn pick_preferred_mode(conn: &connector::Info) -> Option<Mode> {
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
