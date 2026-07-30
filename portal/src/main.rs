//! Libreland's `xdg-desktop-portal` backend.
//!
//! One binary in place of the two backends a wlroots-style desktop normally
//! bolts together: `xdg-desktop-portal-wlr` (screencast + screenshot, via
//! screencopy and `PipeWire`) and `xdg-desktop-portal-gtk` (file chooser,
//! settings, notifications, and the long tail). Merging them removes the
//! per-interface routing users had to configure, the GTK stack the desktop
//! otherwise doesn't need, and the split-brain where the two halves disagreed
//! about the theme, the fonts, and which monitor you meant.
//!
//! # What this is, and isn't
//!
//! This is a **backend**. Apps never talk to it: they call
//! `org.freedesktop.portal.Desktop` on the `xdg-desktop-portal` frontend
//! daemon, which does the sandbox checks, owns the document store and the
//! permission store, and forwards the user-facing half to whichever backend
//! the desktop selected — us. That frontend is part of the portal
//! architecture, not a desktop dependency we can shed, so it stays; the two
//! *desktop* backends are what this replaces.
//!
//! # Shape of the process
//!
//! * A tokio runtime owns the session-bus connection and every interface
//!   object. All the D-Bus work is async and none of it blocks.
//! * Dialogs run on blocking tasks, one Wayland connection each (see
//!   [`ui`]). A dialog is therefore never able to stall the bus, and two apps
//!   asking for a file chooser at once get two dialogs rather than a queue.
//! * Screencast sessions each own a thread running a `PipeWire` loop with the
//!   capture connection's Wayland fd folded into it (see [`capture`]).
//!
//! The service is D-Bus activated: the frontend starts it on first use and it
//! exits when the session bus goes away.

// Every method below implements a fixed D-Bus signature. Two consequences
// clippy reads as smells but which are the interface contract here:
//
//   * many handlers don't touch our own state (`unused_self`), and
//   * many take arguments we deliberately ignore, spelled `_parent_window`
//     and friends, which the zbus macro then reads back out of the message
//     (`used_underscore_binding`).
//
// Neither can be fixed without lying about the interface.
// A third: arguments are deserialized from the message and so must be owned,
// even when a handler only reads them (`needless_pass_by_value`).
#![allow(
    clippy::unused_self,
    clippy::used_underscore_binding,
    clippy::needless_pass_by_value,
    reason = "the D-Bus interface definitions fix these signatures"
)]

mod apps;
mod capture;
mod ipc;
mod portals;
mod pw;
mod ui;

use anyhow::Context as _;
use tracing_subscriber::EnvFilter;

use portals::{BUS_NAME, PORTAL_PATH};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    // Everything here is a Wayland client of the compositor that started the
    // session. Without a display we can still serve Settings and Lockdown,
    // but the dialogs and capture can't work, so say so once and loudly
    // rather than failing per-request.
    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        tracing::warn!(
            "WAYLAND_DISPLAY is unset — dialogs and screen capture will fail. \
             The compositor exports it to the D-Bus activation environment; a \
             portal started outside the session won't see it."
        );
    }

    let conn = zbus::conn::Builder::session()
        .context("connect to the session bus")?
        // Interfaces first, name last: claiming the name before the objects
        // exist is a window in which the frontend can call us and get an
        // UnknownObject back.
        .serve_at(PORTAL_PATH, portals::access::Access::new())?
        .serve_at(PORTAL_PATH, portals::account::Account::new())?
        .serve_at(PORTAL_PATH, portals::appchooser::AppChooser::new())?
        .serve_at(PORTAL_PATH, portals::background::Background::new())?
        .serve_at(
            PORTAL_PATH,
            portals::dynamic_launcher::DynamicLauncher::new(),
        )?
        .serve_at(PORTAL_PATH, portals::email::Email::new())?
        .serve_at(PORTAL_PATH, portals::filechooser::FileChooser::new())?
        .serve_at(PORTAL_PATH, portals::inhibit::Inhibit::new())?
        .serve_at(PORTAL_PATH, portals::lockdown::Lockdown::new())?
        .serve_at(PORTAL_PATH, portals::notification::Notification::new())?
        .serve_at(PORTAL_PATH, portals::print::Print::new())?
        .serve_at(PORTAL_PATH, portals::screencast::ScreenCast::new())?
        .serve_at(PORTAL_PATH, portals::screenshot::Screenshot::new())?
        .serve_at(PORTAL_PATH, portals::settings::Settings::new())?
        .serve_at(PORTAL_PATH, portals::shortcuts::GlobalShortcuts::new())?
        .serve_at(PORTAL_PATH, portals::wallpaper::Wallpaper::new())?
        .name(BUS_NAME)
        .context("claim the portal bus name (another instance running?)")?
        .build()
        .await
        .context("export the portal interfaces")?;

    // Live appearance updates, and the compositor link that global shortcuts
    // ride on. Both are best-effort: the portal serves everything else fine
    // without them.
    portals::settings::spawn_watcher(conn.clone());
    portals::shortcuts::spawn_listener(conn.clone());

    tracing::info!(name = BUS_NAME, "libreland portal ready");

    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    tokio::select! {
        _ = sigterm.recv() => {}
        _ = sigint.recv() => {}
    }
    tracing::info!("shutting down");
    Ok(())
}
