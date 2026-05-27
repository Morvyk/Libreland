# Libreland

A Wayland compositor written in pure Rust, configured in Lua.

## Status

Pre-alpha. Each `cargo run` currently:

1. Opens a libseat session and enumerates input devices via udev +
   libinput. Every pointer-capable device gets the configured accel
   profile + speed applied on `DeviceAdded` (defaults: `Flat` / `0.0`
   — 1:1 motion, no acceleration ramp).
2. Opens the first DRM card, picks the first connected output and its
   preferred mode, then sets up a **GBM + EGL + GLES2 render pipeline**
   over it (via smithay's `GbmBufferedSurface`).
3. Each vblank renders the configured wallpaper (default: vertical
   sky-blue → navy gradient via 256 horizontal stripes) with a 24×24
   white right-triangle cursor that follows mouse motion. Frame GPU
   work is fenced before scanout (tearing-free).
4. Routes every key event through **xkbcommon** for layout-aware
   keysym + modifier handling, then matches against the keybind list
   in [`config.binds`](#binds). The default binding fires the `Exit`
   action on `Super+Shift+E`.
5. Sits in the calloop event loop until an `Exit` action runs.

All user-tunable behaviour lives in a single `Config` struct (see
[Configuration](#configuration)). Defaults today; the Lua loader is
milestone 3c.

Still to come: multi-output (3b), Lua config loading (3c), Wayland
protocol handling (`wl_compositor` / `xdg_shell` / clients).

## Configuration

Every runtime setting lives in `src/config.rs` as `Config`. Today the
struct is populated with sensible defaults at startup. Milestone 3c
adds a Lua loader that reads
`$XDG_CONFIG_HOME/libreland/config.lua` and replaces those defaults.
The fields listed below are the full schema — some are wired into the
runtime today (✅), others sit in the struct waiting for the milestone
that uses them (⏳).

### monitors

| Field                    | Default  | State  | Notes                                                                           |
| ------------------------ | -------- | ------ | ------------------------------------------------------------------------------- |
| `outputs[name].mode`     | `None`   | ⏳ 3b  | `Some((width, height, refresh_mHz))` forces a mode; `None` uses EDID-preferred. |
| `outputs[name].position` | `(0, 0)` | ⏳ 3b  | Top-left of the output in the virtual layout, logical pixels.                   |
| `outputs[name].scale`    | `1.0`    | ⏳ 3b  | Fractional scale. Exposed to clients via `wp_fractional_scale_manager_v1` once the Wayland frontend lands. |
| `primary`                | `None`   | ⏳ 3b  | Connector name of the primary output; `None` = first connected.                 |

### input

| Field                  | Default | State                            | Notes                                                                                  |
| ---------------------- | ------- | -------------------------------- | -------------------------------------------------------------------------------------- |
| `repeat_rate`          | `25`    | ⏳ Wayland frontend              | Repeats per second after the delay elapses. 25 matches X11's classic default.          |
| `repeat_delay`         | `600`   | ⏳ Wayland frontend              | Milliseconds before repeat fires.                                                      |
| `keyboard_layout`      | `""`    | ✅                               | xkb RMLVO layout field. Empty defers to `$XKB_DEFAULT_LAYOUT` / system default.        |
| `mouse_accel_profile`  | `Flat`  | ✅ (applied per pointer device)  | `Flat` (1:1, no ramp) or `Adaptive` (libinput's curve, system default).                |
| `mouse_accel_speed`    | `0.0`   | ✅ (applied per pointer device)  | libinput speed in `[-1.0, 1.0]`. `0.0` is neutral; with `Flat` this is "no extra sensitivity". |

### binds

A list of keybindings. A press matches when its xkb keysym equals the
binding's `keysym` **and** every modifier in the binding's `mods` mask is
held. Extras like `NumLock` are tolerated. First match wins.

Built-in default: `Super+Shift+E → Action::Exit`.

Available actions today: `Exit`. The list grows as we add `Reload`,
`Spawn`, `ChangeVt`, …

### misc

| Field        | Default            | State | Notes                                                                                                          |
| ------------ | ------------------ | ----- | -------------------------------------------------------------------------------------------------------------- |
| `wallpaper`  | vertical gradient  | ✅    | `Solid([r, g, b])` or `VerticalGradient { top, bottom }`. RGB components in `[0, 1]`. Drawn every frame.       |

## Keybindings

Bindings are hard-coded in `src/main.rs` for now. They will move to the Lua
config layer once that exists.

| Combo           | Action                       |
| --------------- | ---------------------------- |
| `Super+Shift+E` | Exit the compositor cleanly. |

The hotkey is matched against raw libinput key codes, so it will keep working
once a future DRM grab disables the kernel's Ctrl+C path. Until that grab
exists, Ctrl+C on the host TTY also exits — but treat `Super+Shift+E` as the
canonical exit.

## Running

Switch to a free TTY (e.g. `Ctrl+Alt+F2`), log in, then:

    cd /path/to/Libreland
    cargo run

### Logging

Tracing output goes to two sinks:

- `stderr` — ANSI-coloured, for live development on the host.
- `$XDG_STATE_HOME/libreland/<TIMESTAMP>.log` — per-startup, ANSI-free
  file you can read after a crash or freeze. The directory is created
  automatically; default is `~/.local/state/libreland/`. Each `cargo run`
  produces a fresh file named with the UTC timestamp at startup.

Configure level with `RUST_LOG`; default is `info,libreland=debug`, which
keeps our own messages visible while smithay/calloop internals stay quiet.
Raise to `debug` or `trace` for more detail.

### Testing safely on a TTY

Running libreland from a free TTY **while a graphical session is still
running on another VT is dangerous**. When libreland acquires `seat0`
from logind, your existing compositor loses input — and if it handles
that poorly, logind cannot hand the seat back cleanly afterwards. The
whole system can appear frozen, requiring a hard reboot.

Before testing:

1. Cleanly log out of any X/Wayland session you have running
   (e.g. `loginctl terminate-session <ID>` from a TTY, or close it
   normally from inside the session).
2. Log in fresh on a text TTY (`Ctrl+Alt+F2` etc.) and run `cargo run`
   from there.
3. *Optional recovery preparation:* `echo 1 | sudo tee /proc/sys/kernel/sysrq`
   re-enables the full Magic SysRq set so `Alt+SysRq+R E I S U B` can
   save the system if something does freeze. (Default Arch
   `kernel.sysrq` is `16` — sync only.)

After `Super+Shift+E` exits the binary, check
`~/.local/state/libreland/` for the latest timestamped log — that's the
authoritative record of what happened, even when stderr wasn't visible.

## Building

    cargo build              # dev profile (tuned for compile speed)
    cargo build --release    # release profile (fat LTO, opt-level 3)

The `mold` linker is enabled globally via `.cargo/config.toml`, so even the
dev profile gets fast linking.

## Code quality

- Edition 2024, Rust `1.95.0` (pinned in `rust-toolchain.toml`).
- `#[deny(unsafe_code)]` at the crate root; any unsafe must override with
  `#[allow(unsafe_code, reason = "…")]` and ship with a `// SAFETY:` comment.
- `clippy::all` denied, `clippy::pedantic` warned. Warnings are fixed at the
  source — `#[allow]` is only acceptable with a `reason = "…"`.
- Always `cargo clippy` and `cargo fmt --check` clean before committing.
