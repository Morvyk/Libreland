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
   sky-blue → navy gradient via 256 horizontal stripes) with a pointer
   cursor that follows mouse motion. The cursor is loaded from the
   `XCursor` theme named by `$XCURSOR_THEME` (size from `$XCURSOR_SIZE`,
   default 24), falling back to a built-in white right-triangle when no
   theme is found. Frame GPU work is fenced before scanout
   (tearing-free).
4. Routes every key event through **xkbcommon** for layout-aware
   keysym + modifier handling, then matches against the keybind list
   in [`config.binds`](#binds). The default binding fires the `Exit`
   action on `Super+Shift+E`.
5. Brings up a minimal **Wayland frontend** — `wl_compositor`,
   `wl_subcompositor`, `wl_shm`, `wl_seat` (with keyboard + pointer
   capabilities advertised), `wl_output`, `xdg_wm_base`/
   `xdg_toplevel`/`xdg_surface`, `wl_data_device_manager` (clipboard +
   drag-and-drop), `wp_viewporter` + `wp_fractional_scale_manager_v1`
   (fractional scaling), and `wlr_layer_shell`. Sets `$WAYLAND_DISPLAY`
   and spawns every `config.startup` command as a child.
   Decorations are forced **server-side** (and Libreland draws none, so
   windows are bare): both `zxdg_decoration_manager_v1` and the legacy
   KDE `org_kde_kwin_server_decoration` are advertised with a Server
   default, since some toolkits (GTK/Firefox) only honour the KDE one.
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
- **File present but Lua syntax or schema error**: libreland logs the
  error prominently — the file path, the Lua error message (line +
  column for syntax errors), and a breadcrumb chain through the schema
  (`binds[2] → mods[0] → unknown modifier "Sper"`) — and **falls back
  to defaults rather than crashing**. Fix the file and save to
  recover via live reload; no restart needed.
- **File present and valid**: values flow into `Config` and propagate
  through the runtime.

### Live reload

The config file is watched (its mtime is polled once a second) and
re-applied on save — no restart needed for most settings. A save that
fails to parse is logged and **ignored**, leaving the running config
untouched, so a typo never breaks your session.

Applied live: `binds`, `screenshot`, `input.focus_model`,
`misc.wallpaper`, the whole `border` section, `layout` gaps, and the
whole `animations` section. Changing these takes effect on the next
frame / window reconfigure.

Needs a restart (a "restart to apply" line is logged when they change):
`monitors` (mode/position/scale/primary), the keyboard/pointer
`input` settings other than `focus_model` (`repeat_rate`,
`repeat_delay`, `keyboard_layout`, `mouse_accel_*`), `env`, and
`startup` (`env`/`startup` only act at launch).

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

-- Window + workspace motion. All on by default; this block just
-- restates the defaults. `enabled = false` disables everything;
-- each animation can be tuned or disabled individually.
animations = {
    window_open  = { duration = 250, curve = "ease-out"    }, -- fade + scale-in
    window_close = { duration = 200, curve = "ease-in"     }, -- fade + scale-out
    window_move  = { duration = 250, curve = "ease-out"    }, -- reflow / move / resize
    workspace    = { duration = 300, curve = "ease-in-out" }, -- vertical slide
}

-- Window opacity + Kawase backdrop blur. Default: opaque windows, blur
-- behind layer surfaces (rofi/panels) only. Set opacity < 1 and/or
-- blur.windows = true to frost behind windows as well.
decoration = {
    opacity = 1.0,
    blur = { enabled = true, layers = true, windows = false, passes = 3, radius = 5.0 },
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

-- Run xwayland-satellite at startup for X11 app support (default true).
xwayland = true
```

### env

| Field | Default      | State | Notes                                                                          |
| ----- | ------------ | ----- | ------------------------------------------------------------------------------ |
| `env` | `{}` (empty) | ✅    | Map of `NAME = "value"` pairs exported via `setenv` at startup, before any child is spawned, so all clients inherit them. Applied before `WAYLAND_DISPLAY` (which can't be overridden). Names can't be empty or contain `=`/NUL. Changing them needs a restart. |

`XCURSOR_THEME` and `XCURSOR_SIZE` set here do double duty: clients
inherit them *and* the compositor reads them for its own pointer
cursor, so `env = { XCURSOR_THEME = "Breeze_Light" }` themes both. The
compositor reads the env once at startup, so a change needs a restart.

**Session defaults.** Libreland sets these session-identity vars at
startup so apps and the desktop portal know what they're in — you don't
need to add them, and your `env` table overrides any of them:

| Variable              | Default     |
| --------------------- | ----------- |
| `XDG_CURRENT_DESKTOP`  | `libreland` |
| `XDG_SESSION_TYPE`     | `wayland`   |
| `XDG_SESSION_DESKTOP`  | `libreland` |

They're also pushed (with `WAYLAND_DISPLAY` / `DISPLAY`) into the D-Bus
+ systemd-user activation environment via
`dbus-update-activation-environment`, so D-Bus-activated services like
`xdg-desktop-portal` see them. (The XDG *base directories* and
`XDG_RUNTIME_DIR` are left untouched — those are the system's to set.)

### startup

| Field     | Default      | State | Notes                                                                          |
| --------- | ------------ | ----- | ------------------------------------------------------------------------------ |
| `startup` | `{}` (empty) | ✅    | Vec of command strings spawned once the Wayland socket is up. Whitespace-split into program + args. |

### xwayland

| Field      | Default | State | Notes |
| ---------- | ------- | ----- | ----- |
| `xwayland` | `true`  | ✅    | Run [`xwayland-satellite`](https://github.com/Supreeeme/xwayland-satellite) at startup so X11 apps work. The compositor picks a free X display (`:0`..`:32`), launches the satellite on it, and exports `$DISPLAY`. If the binary isn't installed it's logged and skipped (never fatal). Toggling needs a restart. |

XWayland runs **rootless** via `xwayland-satellite`: it connects to
Libreland as an ordinary Wayland client, so X windows arrive as normal
`xdg_toplevel`s and tile/float like any other window. The satellite
scales X apps itself through `wp_fractional_scale` + `wp_viewporter`
(on a mixed-scale multi-monitor setup it uses the *smallest* output's
scale). Cursors stay consistent because Libreland draws its own pointer
over every surface and exports `XCURSOR_THEME`/`XCURSOR_SIZE` to the
satellite. Requires `xwayland-satellite` (and `Xwayland`) installed.

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
| `"close"`           | Politely ask the focused toplevel to close (`xdg_toplevel.close`). The client runs its own close path, so it may prompt or ignore the request. Aliases: `"closewindow"`, `"kill"`. |
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
| `outputs[name].scale`    | `1.0`    | ✅    | Fractional scale. The renderer scales every layout coordinate from compositor (= logical) to physical by this factor. Clients see the exact fractional value via `wp_fractional_scale_manager_v1` and a rounded integer fallback via `wl_output.scale`; `wp_viewporter` is advertised so fractional-aware clients can map their oversized buffer down to the logical rect (without it their content composites at the wrong size). Must be positive. Per-surface scale tracking is single-output for now — every surface gets the primary's scale until per-output workspaces ship. |
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
- `Super+C → close`

Your `binds` table is **merged on top of** these defaults, not
swapped in for them: a bind whose trigger (`mods` + `key`) matches a
default overrides that default's action, and any default you don't
touch stays active. So adding a single `Super+Space` bind keeps
`Super+Shift+E` and `Super+F` working.

Available actions today: `exit`, `togglefloating`, `close`, `spawn`.
The list grows as we add `reload`, `change_vt`, …

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

### animations

Window and workspace motion. All on by default with sane, brief
timings; tune or disable per animation, or kill the lot with
`enabled = false`. Each animation is `{ enabled, duration, curve }`;
omitted fields fall back to the section-level `duration` / `curve`,
which in turn default per the table.

| Field          | Default                  | State | Notes                                                                                                       |
| -------------- | ------------------------ | ----- | ----------------------------------------------------------------------------------------------------------- |
| `enabled`      | `true`                   | ✅    | Master switch. `false` disables every animation regardless of the per-animation flags.                     |
| `duration`     | per-animation (see below)| ✅    | Section-level default duration in **milliseconds**, inherited by any animation that doesn't set its own.    |
| `curve`        | per-animation (see below)| ✅    | Section-level default easing, inherited likewise.                                                           |
| `window_open`  | `250ms`, `ease-out`      | ✅    | A window mapping: fades + scales in.                                                                        |
| `window_close` | `200ms`, `ease-in`       | ✅    | A window closing: a snapshot of its last frame fades + scales out. Falls back to an instant close if the client tears its buffer down before the toplevel is destroyed. |
| `window_move`  | `250ms`, `ease-out`      | ✅    | A window's tile changing position/size — reflow on open/close, fullscreen toggle, or the drop after an interactive move/resize. Slides + scales to the new rect. The window under an active drag tracks the cursor 1:1 (no animation) and eases into place on release. |
| `workspace`    | `300ms`, `ease-in-out`   | ✅    | Switching workspaces: the outgoing and incoming workspaces slide vertically. Next slides up (incoming from the bottom), previous slides down. |

A `curve` is either a **named** string — `"linear"`, `"ease-in"`,
`"ease-out"`, `"ease-in-out"` (`_` and `-` are interchangeable, case
insensitive) — or a **`{x1, y1, x2, y2}`** cubic-Bézier with CSS
semantics (the `x` control points must be in `[0, 1]`).

```lua
animations = {
    -- enabled = false,        -- uncomment to turn everything off
    duration = 250,            -- default ms for any animation below
    curve = "ease-out",        -- default easing

    window_open  = { duration = 250, curve = "ease-out"    },
    window_close = { duration = 200, curve = "ease-in"     },
    window_move  = { duration = 250, curve = "ease-out"    },
    workspace    = { duration = 300, curve = "ease-in-out" },

    -- e.g. a snappier, custom-bezier move; disable the workspace slide:
    -- window_move = { duration = 200, curve = { 0.05, 0.9, 0.1, 1.0 } },
    -- workspace   = { enabled = false },
}
```

### decoration

Window opacity and Kawase backdrop blur. Applied live on reload.

| Field          | Default | State | Notes                                                                                                                                  |
| -------------- | ------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `opacity`      | `1.0`   | ✅    | Compositor-applied alpha for **windows** (multiplies the client's own alpha). `1.0` = fully opaque. Tiled/floating windows only; maximized/fullscreen stay opaque. `[0.0, 1.0]`. |
| `blur`         | see below | ✅  | Background blur sub-table (below). Only **visible** where a surface is translucent (client alpha or `opacity` < 1).                     |

The `blur` sub-table:

| Field      | Default | State | Notes                                                                                                                                          |
| ---------- | ------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`  | `true`  | ✅    | Master switch for all blur.                                                                                                                    |
| `layers`   | `true`  | ✅    | Blur behind **Top/Overlay** layer-shell surfaces (panels, launchers like rofi, notifications). Sampled against the whole desktop beneath them. |
| `windows`  | `false` | ✅    | Blur behind **windows**. Tiled windows blur against the base (wallpaper + lower layers); floating windows blur against the base **plus the tiled windows underneath**, so a float reveals a blurred copy of the windows it covers. |
| `passes`   | `3`     | ✅    | Dual-filter passes — each is a downsample + later upsample. More passes = a wider, softer (and costlier) blur. `0` disables. `0..=10`.          |
| `radius`   | `5.0`   | ✅    | Per-tap sample offset in pixels; scales the blur's spread. `>= 0`.                                                                             |

Blur only runs when something translucent is on screen that needs it: the
backdrop is snapshotted per z-band (base → +tiled → +floating/maximized)
and each band is blurred at most once, so nothing is ever double-blurred.
Surface alpha isn't probed, so a mapped opaque panel/window still pays for
its tier while it's up — the cost is bounded.

```lua
decoration = {
    opacity = 0.9,                    -- windows slightly see-through (1.0 = opaque)
    blur = {
        -- enabled = false,           -- uncomment to turn all blur off
        layers  = true,               -- frost behind rofi/panels/notifications
        windows = true,               -- frost behind translucent windows too
        passes  = 3,
        radius  = 5.0,
    },
}
```

## Keybindings

Key bindings are configurable via [`binds`](#binds) (the table below lists the
built-in defaults). Pointer gestures (drags and `Super`+scroll) are hard-coded
in `src/main.rs`.

| Combo           | Action                                              |
| --------------- | --------------------------------------------------- |
| `Super+Shift+E` | Exit the compositor cleanly.                        |
| `Super+F`       | Toggle floating mode on the focused window.         |
| `Super+F11`     | Toggle fullscreen on the focused window.             |
| `Super+C`       | Close the focused window (`xdg_toplevel.close`).    |
| `Super+LMB`-drag | Interactively move the window under the cursor (auto-floats it if tiled; drop on another monitor to move it there). |
| `Super+RMB`-drag | Interactively resize the window under the cursor from its bottom-right corner (auto-floats it if tiled). |
| `Super`+scroll down / up | Switch to the next / previous workspace on the output **under the cursor**. |
| `Super+Shift`+scroll down / up | Move the focused window to the next / previous workspace on **its** output and follow it there. |

Workspaces are per-output and dynamic (niri-style): each output starts with
one, scrolling down materializes a fresh empty workspace to move into, and
empty workspaces are compacted away as you leave them (no wrap at the top).
Only the active workspace of each output is rendered; each workspace keeps its
own tiled tree and floating stack.

Letter key bindings match case-insensitively, and the hotkey path uses
xkb-resolved keysyms, so it keeps working once a future DRM grab disables the
kernel's Ctrl+C path. Until that grab exists, Ctrl+C on the host TTY also
exits — but treat `Super+Shift+E` as the canonical exit.

## Screenshots (built in)

Libreland has its own screenshot tool — no `grim`/`slurp` needed. It's
**off by default**; add a `screenshot` list to the config to enable it.
Each entry binds a key (`key`, optional `mods`) to a capture:

| Field       | Default | Notes                                                                                 |
| ----------- | ------- | ------------------------------------------------------------------------------------- |
| `key`       | —       | xkb keysym name, e.g. `"Print"`. Required.                                            |
| `mods`      | none    | Optional modifier array (`{ "Super" }`).                                              |
| `mode`      | —       | `"region"` (drag a rectangle), `"window"` (click a window), `"output"` (whole monitor under the cursor, instant). Required. |
| `freeze`    | `false` | Pause the screen while selecting (so video/animation doesn't move under you). Ignored for `"output"`. |
| `clipboard` | `false` | Copy the PNG to the clipboard as `image/png` (paste with `Ctrl+V`).                    |
| `show_cursor` | `false` | Bake the pointer cursor into the capture.                                          |
| `directory` | none    | Where to save a PNG; omit to not save. `~` and `$VAR`/`${VAR}` are expanded. Files are named `Screenshot_YYYYMMDD_HHMMSS.png` (local time). |

During a region/window capture the screen dims, the selection is
outlined, **Esc** (or right-click) cancels, and **Enter** confirms. The
selection UI is never in the saved image, and the cursor is excluded
unless `show_cursor = true`. Example —
`Print` freezes, drags a rectangle, copies it, and saves under the XDG
pictures dir:

```lua
screenshot = {
    {
        key = "Print",
        mode = "region",
        freeze = true,
        clipboard = true,
        directory = (os.getenv("XDG_PICTURES_DIR") or (os.getenv("HOME") .. "/Pictures")) .. "/Screenshots",
    },
}
```

(Region capture currently resolves to the single output the drag starts
on; cross-monitor region grabs are a follow-up.)

## Screen capture & desktop portals

Libreland implements `zwlr_screencopy_v1`, so external capture also works
— e.g. `grim` (whole screen) or, for sharing, the portal below.

For app-facing functionality — screen **sharing** (OBS, Discord,
browsers), **file dialogs**, dark-mode/appearance **settings** — install
`xdg-desktop-portal` plus the wlroots and a generic backend, and route
them with a portals config:

1. Install `xdg-desktop-portal`, `xdg-desktop-portal-wlr`, and
   `xdg-desktop-portal-gtk` (or `-kde` to match `QT_QPA_PLATFORMTHEME`).
2. Route the portal backends: copy
   [`contrib/libreland-portals.conf`](contrib/libreland-portals.conf)
   to `~/.config/xdg-desktop-portal/libreland-portals.conf`. That routes
   `ScreenCast` + `Screenshot` to `xdg-desktop-portal-wlr` (which uses
   Libreland's screencopy) and everything else to the generic backend.

   Libreland already sets `XDG_CURRENT_DESKTOP=libreland` (see
   [env defaults](#env)) and exports it to the D-Bus activation
   environment, so this config is selected automatically — no manual
   env needed.
3. Pick which monitor to share via the portal's **output chooser**.
   Libreland ships its own: `libreland-output-picker` (the `output-picker`
   workspace member) — a `wlr-layer-shell` overlay that dims every monitor,
   highlights + labels the one under the cursor, and prints its connector
   name on click (Esc cancels). Install it and point xdpw at it:

       cargo install --path output-picker     # -> ~/.cargo/bin/libreland-output-picker

   then copy [`contrib/xdg-desktop-portal-wlr.config`](contrib/xdg-desktop-portal-wlr.config)
   to `~/.config/xdg-desktop-portal-wlr/config` (its `chooser_cmd` is the
   picker; use an absolute path if `~/.cargo/bin` isn't on the portal's
   `PATH`). It replaces `slurp`, which crashes here: slurp 1.5.0 has an
   unguarded NULL deref in its `wl_pointer.motion` handler (no released
   fix). A text menu also works (`chooser_type=dmenu` + `chooser_cmd=fuzzel
   --dmenu`) if you prefer picking a name from a list.

`xdg-desktop-portal-wlr` drives screen sharing off the same screencopy
implementation, so OBS / Discord / browser screen-share work once the
above is in place. Global shortcuts (the `GlobalShortcuts` portal) are
not yet provided — no off-the-shelf backend covers them for us, so they
need a dedicated Libreland backend (planned).

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
