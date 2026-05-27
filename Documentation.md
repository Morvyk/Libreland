# Libreland

A Wayland compositor written in pure Rust, configured in Lua.

## Status

Pre-alpha. Each `cargo run` currently:

1. Opens a libseat session, enumerates input devices via udev + libinput,
   and logs every event that flows through (keys, pointer motion, buttons).
2. Opens the first DRM card, picks the first connected output and its
   preferred mode, then sets up a **GBM + EGL + GLES2 render pipeline**
   over it (via smithay's `GbmBufferedSurface`).
3. Renders a vblank-paced hue cycle to the display: the screen sweeps
   red → yellow → green → cyan → blue → magenta → red, full cycle every
   8 seconds. Each frame is fsync'd through the GPU before scanout, so
   timing should be tearing-free.
4. Sits in the calloop event loop until you press `Super+Shift+E`.

Still to come: cursor sprite tracking pointer motion, xkbcommon
keyboard handling, multi-output, Wayland protocol handling
(`wl_compositor` / `xdg_shell` / clients), and the Lua config layer.

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
