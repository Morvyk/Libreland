# Libreland

A Wayland compositor written in **Rust**, configured in **Lua**.

Completely made by AI — vibe coded with **Claude**. If you don't like that,
don't use the compositor. That's fine.

If you find a bug or some weird behaviour, **feel free to open an issue** — I'm
happy to fix it.

## What it is

Libreland is a standalone Wayland compositor. You drop a Lua file at
`$XDG_CONFIG_HOME/libreland/config.lua` and it configures keybinds, monitors,
animations, blur, decorations, HDR, and the rest. See
[Documentation.md](Documentation.md) for the full config reference.

## Protocol support

Most of the important Wayland protocols are supported:

- **xdg-shell** (`xdg_wm_base`) — application windows, popups
- **wlr-layer-shell** — bars, panels, launchers, lock screens, OSDs
- **linux-dmabuf** (v5 feedback) — zero-copy GPU buffer sharing, with a
  per-output scanout tranche that steers fullscreen clients into
  plane-compatible buffers
- **wp_fractional_scale** + **wp_viewporter** — exact fractional scaling
- **wp_color_management** — HDR (BT.2020 / PQ), detected by Proton/mpv
- **wp_presentation** — accurate per-frame presentation timing from the real
  DRM page-flip clock
- **linux-drm-syncobj** — explicit GPU sync (Proton/DXVK/Vulkan); no tearing
- **pointer-constraints** + **relative-pointer** — pointer lock & raw motion
  for games
- **tearing-control** — immediate presentation for `IMMEDIATE` swapchains
  (opt-in; `misc.tearing`)
- **xdg-decoration** — server-side decorations (toolkits drop their CSD)
- **xdg-activation** — focus/raise requests
- **wp_cursor_shape** — themed named cursors
- **wlr-screencopy** — screenshots & screen sharing (drives the bundled
  desktop portal)
- **wlr-data-control** + **ext-data-control** — clipboard managers
- **primary-selection** — middle-click paste
- **ext-session-lock** — screen lockers
- **idle-inhibit** + **ext-idle-notify** — idle handling
- **XWayland** — X11 apps

…plus the core globals. The full list with notes lives in
[Documentation.md](Documentation.md#wayland-protocols).

## Direct scanout

A fullscreen game's buffer goes **straight to the display hardware** — no
compositing pass, no copy, no extra frame of latency. A notification or menu
above it rides a spare hardware plane rather than dragging the whole frame
back through the GPU, and screen-sharing the game doesn't turn any of it off:
the capture is served from the same buffer that's on the plane.

It needs no configuration and engages by itself; every frame that doesn't
qualify just composites as usual. See
[Direct scanout](Documentation.md#direct-scanout) for what qualifies and why.

## Desktop integration

Libreland ships **its own `xdg-desktop-portal` backend** — one binary
covering what `xdg-desktop-portal-wlr` and `xdg-desktop-portal-gtk` covered
between them, so screen sharing, file dialogs, dark mode, notifications and
global shortcuts all work with nothing to configure and no GTK stack
installed. Its dialogs are drawn by the portal itself: no toolkit, no
fontconfig, no second theme engine.

It also brings the piece neither of those backends had: **global
shortcuts**, backed by keybinds an app can register over the compositor's
control socket. See
[Screen capture & the desktop portal](Documentation.md#screen-capture--the-desktop-portal).

## Building

See the [PKGBUILD](contrib/PKGBUILD) (Arch) or build directly:

```sh
cargo build --release
```

## License

MIT — see [LICENSE](LICENSE).
