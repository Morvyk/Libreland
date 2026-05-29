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
5. Brings up a minimal **Wayland frontend** — `wl_compositor`,
   `wl_subcompositor`, `wl_shm`, `wl_seat` (with keyboard + pointer
   capabilities advertised), `wl_output`, and `xdg_wm_base`/
   `xdg_toplevel`/`xdg_surface`. Sets `$WAYLAND_DISPLAY` and spawns
   every `config.startup` command as a child.
6. Composites every live `xdg_toplevel`'s surface onto the
   framebuffer between wallpaper and cursor, by uploading each
   client buffer as a GLES texture and drawing it through smithay's
   surface render-element pipeline. After each output is queued for
   scanout, drains the surface tree's `wl_callback` queue so clients
   know to draw the next frame.
7. Forwards pointer motion + button events to the focused client
   through `wl_pointer.motion` / `wl_pointer.button` (plus
   smithay-driven `enter`/`leave`), and forwards keyboard keys —
   including modifier tracking — through `wl_keyboard.key` /
   `wl_keyboard.modifiers`. Compositor-level hotkeys (see
   `config.binds`) are filtered out of forwarding so e.g. typing in
   a focused client can't accidentally trigger them. Pointer focus
   is set by hit-testing the layout each motion; keyboard focus
   follows the [`input.focus_model`](#input) (`"hover"` by default,
   `"click"` available). Newly mapped windows take focus on map in
   either model.
8. Sits in the calloop event loop until an `Exit` action runs.

All user-tunable behaviour lives in a single `Config` struct (see
[Configuration](#configuration)).

Still to come: window management (4d).

## Configuration

Drop a Lua file at `$XDG_CONFIG_HOME/libreland/config.lua` (typically
`~/.config/libreland/config.lua`). Anything you set there overrides
the corresponding default; anything you don't set keeps its default.

- **No file present**: libreland logs `no config.lua found, using
  defaults` and starts with the defaults below.
- **File present but Lua syntax or schema error**: libreland fails at
  startup with the file path, the Lua error message (line + column
  for syntax errors), and a breadcrumb chain through the schema
  (`binds[2] → mods[0] → unknown modifier "Sper"`).
- **File present and valid**: values flow into `Config` and propagate
  through the runtime.

### Complete example

```lua
-- ~/.config/libreland/config.lua

monitors = {
    primary = "DP-1",          -- optional; defaults to first connected
    outputs = {
        ["DP-1"] = {
            mode = { width = 3840, height = 2160, refresh_mhz = 144000 },  -- optional; defaults to EDID-preferred
            position = { x = 0, y = 0 },
            scale = 1.0,
        },
        ["HDMI-A-1"] = {
            position = { x = 3840, y = 0 },
            scale = 1.5,
        },
    },
}

input = {
    repeat_rate = 25,
    repeat_delay = 600,
    keyboard_layout = "",                 -- "" defers to $XKB_DEFAULT_LAYOUT
    mouse_accel_profile = "flat",         -- "flat" or "adaptive"
    mouse_accel_speed = 0.0,              -- [-1.0, 1.0]
    focus_model = "hover",                -- "hover" or "click"
}

binds = {
    { mods = { "Super", "Shift" }, key = "E",     action = "exit" },
    { mods = { "Super" },          key = "F",     action = "togglefloating" },
    { mods = { "Super" },          key = "space", action = "spawn", command = "rofi -show drun" },
}

misc = {
    wallpaper = {
        type = "vertical_gradient",
        top    = { 0.40, 0.60, 0.90 },
        bottom = { 0.10, 0.20, 0.50 },
    },
    -- Or a solid colour:
    -- wallpaper = { type = "solid", color = { 0.20, 0.40, 0.80 } },
}

layout = {
    gaps_outer = 8,                       -- px between tile area and screen edges
    gaps_inner = 3,                       -- px between adjacent tile cells
}

border = {
    width = 1,                            -- 0 disables
    rounded_corners = 4,                  -- 0 disables; radius is in compositor px
    active = {
        type = "vertical_gradient",
        top    = { 0.55, 0.80, 1.00 },
        bottom = { 0.30, 0.55, 0.95 },
    },
    inactive = { type = "solid", color = { 0.30, 0.30, 0.30 } },
}

-- Environment variables exported into the compositor's own process
-- before any client is launched, so every child (startup commands,
-- `spawn` binds, shells) inherits them. Handy for theming hints.
env = {
    XCURSOR_THEME = "Breeze_Light",
    QT_QPA_PLATFORMTHEME = "kde",
}

-- Commands to spawn once the Wayland socket is listening. Each
-- string is whitespace-split into program + args; children inherit
-- $WAYLAND_DISPLAY so they connect to *our* compositor. For shell
-- features (pipes, env, &), wrap with `"sh -c '…'"`.
startup = {
    "kitty",
    -- "sh -c 'swaybg -i ~/wallpapers/blue.png &'",
}
```

### env

| Field | Default      | State | Notes                                                                          |
| ----- | ------------ | ----- | ------------------------------------------------------------------------------ |
| `env` | `{}` (empty) | ✅    | Map of `NAME = "value"` pairs exported via `setenv` at startup, before any child is spawned, so all clients inherit them. Applied before `WAYLAND_DISPLAY` (which can't be overridden). Names can't be empty or contain `=`/NUL. Changing them needs a restart. |

`XCURSOR_THEME` and `XCURSOR_SIZE` set here do double duty: clients
inherit them *and* the compositor reads them for its own pointer
cursor, so `env = { XCURSOR_THEME = "Breeze_Light" }` themes both. The
compositor reads the env once at startup, so a change needs a restart.

### startup

| Field     | Default      | State | Notes                                                                          |
| --------- | ------------ | ----- | ------------------------------------------------------------------------------ |
| `startup` | `{}` (empty) | ✅    | Vec of command strings spawned once the Wayland socket is up. Whitespace-split into program + args. |

### Modifier names (case-insensitive)

`"Shift"`, `"Ctrl"` (alias `"Control"`), `"Alt"` (alias `"Mod1"`),
`"Super"` (aliases `"Logo"`, `"Mod4"`).

### Key names

Anything xkbcommon's `xkb_keysym_from_name` accepts — `"E"`,
`"Return"`, `"F1"`, `"space"`, `"comma"`, …

### Actions

| Action              | Effect                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `"exit"`            | Cleanly exit the compositor.                                                                                 |
| `"togglefloating"`  | Flip the focused window between tiled and floating. A newly floating window centres at ~70% of its previous cell. |
| `"spawn"`           | Run an arbitrary command. Requires an additional `command = "…"` field on the bind table; the string is whitespace-split into program + args, children inherit our env (so `$WAYLAND_DISPLAY` reaches them). Wrap with `"sh -c '…'"` for shell features (pipes, env, `&`). |

(More actions land as features grow: `"reload"`, `"change_vt"`, …)

### Schema reference

Every field, its default, and whether it's plumbed all the way into
the runtime today (✅) or just held in `Config` for a later consumer
(⏳). Lua can set every field regardless.

### monitors

| Field                    | Default  | State | Notes                                                                                                                                                                                                                                                                                                          |
| ------------------------ | -------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `outputs[name].mode`     | `nil`    | ✅    | `{ width = …, height = …, refresh_mhz = … }` to force a mode. The override is matched against the EDID mode list by size and refresh (mHz); on a miss it logs and falls back to the EDID-preferred mode. `nil` uses EDID-preferred directly.                                                                   |
| `outputs[name].position` | `nil`    | ✅    | Top-left of this output in the virtual layout, in *logical* pixels (`{ x = …, y = … }`). `nil` falls back to the auto left-to-right layout. Mixing configured and auto-positioned outputs is fine; positions can overlap if you let them.                                                                       |
| `outputs[name].scale`    | `1.0`    | ✅    | Fractional scale. The renderer scales every layout coordinate from compositor (= logical) to physical by this factor. Clients see the exact fractional value via `wp_fractional_scale_manager_v1` and a rounded integer fallback via `wl_output.scale`. Must be positive. Per-surface scale tracking is single-output for now — every surface gets the primary's scale until per-output workspaces ship. |
| `primary`                | `nil`    | ✅    | Connector name of the primary output. The tile area's bounds + the initial cursor position come from this output. `nil` falls back to the first connected output in DRM enumeration order.                                                                                                                      |

### input

| Field                  | Default  | State                            | Notes                                                                                      |
| ---------------------- | -------- | -------------------------------- | ------------------------------------------------------------------------------------------ |
| `repeat_rate`          | `25`     | ✅                               | Repeats per second after the delay elapses. 25 matches X11's classic default. Passed to `wl_keyboard` via the seat at startup. |
| `repeat_delay`         | `600`    | ✅                               | Milliseconds before repeat fires. Passed to `wl_keyboard` at startup.                      |
| `keyboard_layout`      | `""`     | ✅                               | xkb RMLVO layout. Empty defers to `$XKB_DEFAULT_LAYOUT` / system default.                  |
| `mouse_accel_profile`  | `"flat"` | ✅ (applied per pointer device)  | `"flat"` (1:1, no ramp) or `"adaptive"` (libinput's curve, system default).                |
| `mouse_accel_speed`    | `0.0`    | ✅ (applied per pointer device)  | libinput speed in `[-1.0, 1.0]`. `0.0` is neutral; with `"flat"` this is "no extra sensitivity". |
| `focus_model`          | `"hover"`| ✅                               | `"hover"`: keyboard focus follows the surface under the cursor on every motion event. `"click"`: focus only changes on a pointer-button press. New windows take focus on map either way. |

### binds

A list of keybindings. A press matches when its xkb keysym equals the
binding's `keysym` **and** every modifier in the binding's `mods` mask is
held. Extras like `NumLock` are tolerated. First match wins.

Built-in defaults:

- `Super+Shift+E → exit`
- `Super+F → togglefloating`

Your `binds` table is **merged on top of** these defaults, not
swapped in for them: a bind whose trigger (`mods` + `key`) matches a
default overrides that default's action, and any default you don't
touch stays active. So adding a single `Super+Space` bind keeps
`Super+Shift+E` and `Super+F` working.

Available actions today: `exit`, `togglefloating`, `spawn`. The list
grows as we add `reload`, `change_vt`, …

### misc

| Field        | Default            | State | Notes                                                                                                          |
| ------------ | ------------------ | ----- | -------------------------------------------------------------------------------------------------------------- |
| `wallpaper`  | vertical gradient  | ✅    | `Solid([r, g, b])` or `VerticalGradient { top, bottom }`. RGB components in `[0, 1]`. Drawn every frame.       |

### layout

| Field         | Default | State | Notes                                                                                                |
| ------------- | ------- | ----- | ---------------------------------------------------------------------------------------------------- |
| `gaps_outer`  | `8`     | ✅    | Pixels of empty space between the tile area and the screen edge. Wallpaper shows through. `>= 0`.    |
| `gaps_inner`  | `3`     | ✅    | Pixels of empty space between adjacent tile cells. Centred on each split divider. `>= 0`.            |

### border

| Field             | Default                              | State | Notes                                                                                                                                              |
| ----------------- | ------------------------------------ | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `width`           | `1`                                  | ✅    | Border width in pixels around every window. `0` disables borders entirely. `>= 0`.                                                                 |
| `active`          | bright blue gradient                 | ✅    | Fill drawn around the keyboard-focused window. Same `Solid` / `VerticalGradient` types as `misc.wallpaper`.                                        |
| `inactive`        | `Solid([0.30, 0.30, 0.30])` (grey)   | ✅    | Fill drawn around every unfocused window.                                                                                                          |
| `rounded_corners` | `4`                                  | ✅    | Corner radius in **compositor** pixels (multiplied by the output's scale at render time). `0` disables. Per-window effective radius is clamped to half the cell's smaller dimension so corners never overlap on tiny tiles. The frame shader paints both the corner cutout and the border ring along the curve, so the border follows the rounded shape instead of stopping at square corners. |

The client's surface is shrunk by `2 * width` per axis before
configure so the buffer doesn't overlap the border. Rounded
corners are masked with the wallpaper after the border + surface
draw, so floats over tiles show wallpaper (not the tile) at the
rounded corners — proper shader-based rounding is later polish.

## Keybindings

Bindings are hard-coded in `src/main.rs` for now. They will move to the Lua
config layer once that exists.

| Combo           | Action                                              |
| --------------- | --------------------------------------------------- |
| `Super+Shift+E` | Exit the compositor cleanly.                        |
| `Super+F`       | Toggle floating mode on the focused window.         |
| `Super+LMB`-drag | Interactively move the window under the cursor (auto-floats it if tiled). |
| `Super+RMB`-drag | Interactively resize the window under the cursor from its bottom-right corner (auto-floats it if tiled). |

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
