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
- **linux-dmabuf** (v5 feedback) — zero-copy GPU buffer sharing
- **wp_fractional_scale** + **wp_viewporter** — exact fractional scaling
- **wp_color_management** — HDR (BT.2020 / PQ), detected by Proton/mpv
- **wp_presentation** — accurate per-frame presentation timing from the real
  DRM page-flip clock
- **linux-drm-syncobj** — explicit GPU sync (Proton/DXVK/Vulkan); no tearing
- **pointer-constraints** + **relative-pointer** — pointer lock & raw motion
  for games
- **xdg-decoration** — server-side decorations (toolkits drop their CSD)
- **xdg-activation** — focus/raise requests
- **wp_cursor_shape** — themed named cursors
- **wlr-screencopy** — screenshots & screen sharing (via xdg-desktop-portal)
- **wlr-data-control** + **ext-data-control** — clipboard managers
- **primary-selection** — middle-click paste
- **ext-session-lock** — screen lockers
- **idle-inhibit** + **ext-idle-notify** — idle handling
- **XWayland** — X11 apps

…plus the core globals. The full list with notes lives in
[Documentation.md](Documentation.md#wayland-protocols).

## Building

See the [PKGBUILD](contrib/PKGBUILD) (Arch) or build directly:

```sh
cargo build --release
```

## License

MIT — see [LICENSE](LICENSE).
