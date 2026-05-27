//! DRM/KMS device + surface initialisation.
//!
//! Opens the DRM device via libseat, finds the first connected output
//! and its preferred mode, picks a compatible CRTC, and creates a
//! `DrmSurface` bound to that combination. The actual *rendering*
//! happens in [`crate::render`], which consumes the surface to build a
//! GBM-backed GLES render pipeline on top.

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
use tracing::info;

/// Everything the caller needs to wire DRM into calloop and hand off
/// to the renderer.
pub struct DrmInit {
    /// The DRM master claim. Held by the caller so the master claim
    /// outlives the surface and the renderer's swapchain.
    pub device: DrmDevice,
    /// The bound CRTC + connector + mode tuple. Consumed by the
    /// renderer to build a GBM swapchain over it.
    pub surface: DrmSurface,
    /// Refcounted file-descriptor handle (Arc-backed). Cloned where
    /// additional owners are needed (GBM allocator, EGL display).
    pub fd: DrmDeviceFd,
    /// Calloop event source for vblank and device errors. Caller
    /// inserts into the event loop.
    pub notifier: DrmDeviceNotifier,
    /// The mode we picked. Renderer needs its size for the framebuffer
    /// dimensions; the caller logs it.
    pub mode: Mode,
}

/// Open the DRM device at `path` through `session`, find the first
/// connected output, its preferred mode and a compatible CRTC, and
/// create a `DrmSurface` bound to that combination. The caller hands
/// [`DrmInit::surface`] to [`crate::render::Renderer::new`] to build
/// the rendering pipeline.
pub fn open_display(session: &mut LibSeatSession, path: &Path) -> Result<DrmInit> {
    info!(path = %path.display(), "phase: opening DRM device via libseat");
    // libseat's `open` ignores the flags argument internally, but the
    // trait signature requires one. RDWR | NONBLOCK matches what other
    // smithay-based compositors pass.
    let owned_fd = session
        .open(path, OFlags::RDWR | OFlags::NONBLOCK)
        .context("libseat refused to open the DRM device")?;
    let fd = DrmDeviceFd::new(DeviceFd::from(owned_fd));
    info!("DRM fd acquired");

    info!("phase: initialising DrmDevice");
    // `disable_connectors = false`: don't reset anything until the
    // renderer is ready to paint, so the screen doesn't flash to
    // black between fd-acquire and the first frame.
    let (mut device, notifier) =
        DrmDevice::new(fd.clone(), false).context("DrmDevice::new failed")?;
    info!(atomic = device.is_atomic(), "DrmDevice initialised");

    let resources = device
        .resource_handles()
        .context("failed to read DRM resource handles")?;

    info!("phase: enumerating connectors");
    let (conn_handle, conn_info, mode) = find_connected_output_and_mode(&device, &resources)?;
    let (mode_w, mode_h) = mode.size();
    info!(
        connector = ?conn_handle,
        interface = ?conn_info.interface(),
        width = mode_w,
        height = mode_h,
        refresh = mode.vrefresh(),
        "found connected output and selected its mode"
    );

    let crtc_handle = pick_compatible_crtc(&device, &conn_info, &resources)
        .context("no CRTC compatible with the chosen connector")?;
    info!(crtc = ?crtc_handle, "selected CRTC");

    let surface = device
        .create_surface(crtc_handle, mode, &[conn_handle])
        .context("DrmDevice::create_surface failed")?;
    info!(legacy = surface.is_legacy(), "DRM surface bound");

    Ok(DrmInit {
        device,
        surface,
        fd,
        notifier,
        mode,
    })
}

/// Find a CRTC that can drive `connector`. Prefer the encoder/CRTC
/// pair the firmware/previous owner already left configured — that
/// avoids an unnecessary remap — otherwise fall back to the first
/// CRTC the kernel says is compatible.
fn pick_compatible_crtc(
    device: &DrmDevice,
    conn: &connector::Info,
    resources: &drm::control::ResourceHandles,
) -> Option<crtc::Handle> {
    if let Some(encoder_handle) = conn.current_encoder()
        && let Ok(encoder) = device.get_encoder(encoder_handle)
        && let Some(crtc) = encoder.crtc()
    {
        return Some(crtc);
    }

    conn.encoders().iter().find_map(|&encoder_handle| {
        let encoder = device.get_encoder(encoder_handle).ok()?;
        resources
            .filter_crtcs(encoder.possible_crtcs())
            .first()
            .copied()
    })
}

/// Find the first connected output and pick its preferred mode
/// (falling back to the first advertised mode if no `PREFERRED` flag
/// is set). Returns the connector handle, its [`connector::Info`]
/// (the caller needs it for [`pick_compatible_crtc`] which walks the
/// encoder list off it), and the chosen mode.
fn find_connected_output_and_mode(
    device: &DrmDevice,
    resources: &drm::control::ResourceHandles,
) -> Result<(connector::Handle, connector::Info, drm::control::Mode)> {
    let (handle, info) = resources
        .connectors()
        .iter()
        .find_map(|&h| {
            let info = device.get_connector(h, false).ok()?;
            (info.state() == connector::State::Connected).then_some((h, info))
        })
        .context("no connected outputs on this DRM device")?;

    let mode = info
        .modes()
        .iter()
        .find(|m| m.mode_type().contains(ModeTypeFlags::PREFERRED))
        .copied()
        .or_else(|| info.modes().first().copied())
        .context("connector reports no modes")?;

    Ok((handle, info, mode))
}
