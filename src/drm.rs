//! DRM/KMS setup for bare-TTY display.
//!
//! Milestone 2a — proof of life: open the DRM device acquired through
//! libseat, find the first connected output, allocate a dumb framebuffer
//! matching the output's preferred mode, fill it with a solid colour,
//! and mode-set so it actually shows on screen.
//!
//! Deliberately tiny in scope: no GBM, no EGL, no renderer, no
//! page-flipping, no hotplug. Each of those becomes its own follow-up
//! milestone once the basic chain (libseat → `DrmDevice` → `DrmSurface`
//! → dumb FB → commit) is proven on real hardware.

use std::path::Path;

use anyhow::{Context as _, Result};
use smithay::backend::drm::{
    DrmDevice, DrmDeviceFd, DrmDeviceNotifier, DrmSurface, PlaneConfig, PlaneState,
};
use smithay::backend::session::Session as _;
use smithay::backend::session::libseat::LibSeatSession;
use smithay::reexports::drm;
use smithay::reexports::drm::buffer::Buffer as _;
use smithay::reexports::drm::buffer::DrmFourcc;
use smithay::reexports::drm::control::{
    Device as ControlDevice, ModeTypeFlags, connector, crtc, dumbbuffer::DumbBuffer, framebuffer,
};
use smithay::reexports::rustix::fs::OFlags;
use smithay::utils::{Buffer as BufferCoords, DeviceFd, Physical, Rectangle, Size, Transform};
use tracing::info;

/// Resources that must outlive any frame we leave on screen. Dropping
/// any of them tears the modeset down — the dumb buffer's pages get
/// reclaimed, the surface's plane is released, and the device fd
/// closes (which hands the seat back to logind).
#[allow(
    dead_code,
    reason = "fields are held for their Drop side effects (keeping the modeset live); each is read by subsequent milestones — surface for page-flip, dumb_buffer for repaint, device for further KMS operations"
)]
pub struct DrmKeepalive {
    /// Kept for symmetry and future API access (mode change, plane
    /// queries). Holds the DRM master claim alive.
    pub device: DrmDevice,
    /// The bound CRTC + connector + mode tuple we committed against.
    pub surface: DrmSurface,
    /// Backing pages for the on-screen framebuffer. dropping this
    /// invalidates the kernel framebuffer object below.
    pub dumb_buffer: DumbBuffer,
    /// Kernel-side framebuffer handle pointing at `dumb_buffer`.
    /// Currently only used for diagnostic logging; held so a future
    /// `rm_framebuffer` cleanup path has something to reference.
    pub framebuffer: framebuffer::Handle,
}

/// Open the DRM device at `path` through `session`, find the first
/// connected output, allocate a dumb framebuffer matching its preferred
/// mode, fill it with `color_xrgb` (0xRRGGBB in the low 24 bits), and
/// commit a modeset so it appears on screen.
///
/// Returns the keepalive bundle (caller stores in compositor state) and
/// the device's calloop notifier (caller inserts into the event loop).
pub fn open_and_paint(
    session: &mut LibSeatSession,
    path: &Path,
    color_xrgb: u32,
) -> Result<(DrmKeepalive, DrmDeviceNotifier)> {
    info!(path = %path.display(), "phase: opening DRM device via libseat");
    // libseat's `open` ignores the flags arg internally, but the trait
    // signature requires one. RDWR | NONBLOCK matches what other
    // smithay-based compositors pass.
    let owned_fd = session
        .open(path, OFlags::RDWR | OFlags::NONBLOCK)
        .context("libseat refused to open the DRM device")?;
    let drm_fd = DrmDeviceFd::new(DeviceFd::from(owned_fd));
    info!("DRM fd acquired");

    info!("phase: initialising DrmDevice");
    // `disable_connectors = false`: don't reset anything until we're
    // ready to paint, so the screen doesn't flash to black between
    // fd-acquire and our own modeset.
    let (mut device, notifier) =
        DrmDevice::new(drm_fd.clone(), false).context("DrmDevice::new failed")?;
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

    info!("phase: allocating dumb framebuffer");
    // Bypassing smithay's DumbAllocator wrapper because it doesn't
    // expose `&mut` to the underlying drm-rs `DumbBuffer`, which
    // `map_dumb_buffer` requires. drm-rs's `create_dumb_buffer` gives
    // us the raw handle directly and ownership is simpler this way.
    let mut dumb_buffer = drm_fd
        .create_dumb_buffer(
            (u32::from(mode_w), u32::from(mode_h)),
            DrmFourcc::Xrgb8888,
            32,
        )
        .context("create_dumb_buffer failed")?;
    info!(
        width = mode_w,
        height = mode_h,
        pitch = dumb_buffer.pitch(),
        "dumb buffer allocated"
    );

    paint_dumb_buffer(&drm_fd, &mut dumb_buffer, color_xrgb)?;
    info!(color = format!("{color_xrgb:#08x}"), "dumb buffer painted");

    let fb_handle = drm_fd
        .add_framebuffer(&dumb_buffer, 24, 32)
        .context("add_framebuffer failed")?;
    info!(framebuffer = ?fb_handle, "framebuffer registered");

    info!("phase: committing modeset");
    surface
        .commit(
            [PlaneState {
                handle: surface.plane(),
                config: Some(PlaneConfig {
                    src: Rectangle::<f64, BufferCoords>::from_size(Size::new(
                        f64::from(mode_w),
                        f64::from(mode_h),
                    )),
                    dst: Rectangle::<i32, Physical>::from_size(Size::new(
                        i32::from(mode_w),
                        i32::from(mode_h),
                    )),
                    transform: Transform::Normal,
                    alpha: 1.0,
                    damage_clips: None,
                    fb: fb_handle,
                    fence: None,
                }),
            }],
            false,
        )
        .context("DrmSurface::commit failed (kernel rejected the modeset)")?;
    info!("DRM commit ok — framebuffer should now be live on the display");

    Ok((
        DrmKeepalive {
            device,
            surface,
            dumb_buffer,
            framebuffer: fb_handle,
        },
        notifier,
    ))
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

/// Map the dumb buffer and splat `color_xrgb` over every pixel.
/// Lifted into its own function so the orchestrating
/// [`open_and_paint`] stays under clippy's `too_many_lines`
/// threshold without losing per-phase visibility.
fn paint_dumb_buffer(
    drm_fd: &DrmDeviceFd,
    dumb_buffer: &mut DumbBuffer,
    color_xrgb: u32,
) -> Result<()> {
    let mut mapping = drm_fd
        .map_dumb_buffer(dumb_buffer)
        .context("map_dumb_buffer failed")?;
    // XRGB8888 in little-endian memory is stored as B, G, R, X.
    // Masking the 24-bit colour into its three bytes once and
    // splatting into every pixel avoids a per-iteration shift.
    let b = (color_xrgb & 0xFF) as u8;
    let g = ((color_xrgb >> 8) & 0xFF) as u8;
    let r = ((color_xrgb >> 16) & 0xFF) as u8;
    for chunk in mapping.chunks_exact_mut(4) {
        chunk[0] = b;
        chunk[1] = g;
        chunk[2] = r;
        chunk[3] = 0;
    }
    Ok(())
}
