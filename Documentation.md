# Libreland

A Wayland compositor written in pure Rust, configured in Lua.

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
re-applied on save. A save that fails to parse is logged and
**ignored**, leaving the running config untouched, so a typo never
breaks your session. You can also force a reload at any time with
`libreland msg reload`.

**Applied live — effectively the whole config:**

- `binds`, `screenshot`, `input.focus_model` — read live each event.
- `misc.wallpaper`, the whole `border` section, `layout` gaps, the
  whole `animations` and `decoration` sections — take effect on the
  next frame / window reconfigure.
- `monitors` — `position`, `scale`, `primary` and `vrr` reflow the
  outputs immediately; a changed `mode` or `hdr` toggle rebuilds the
  affected output's swapchain via a live DRM modeset (the output blinks
  once, windows are preserved).
- `input` — `repeat_rate`/`repeat_delay`, `keyboard_layout` (both the
  keymap clients receive and the one hotkeys match against), and the
  `mouse_accel_*` settings (re-applied to the connected pointers).
- `idle` — the new timeouts/command are picked up on the next idle tick.
- `xwayland` — toggling it starts or stops the Xwayland server
  (stopping it disconnects any running X11 clients).
- `env` — applies to **children spawned from now on** (`spawn` binds,
  the idle lock command, `libreland msg spawn`). Already-running
  clients are unaffected, and vars the compositor itself consumes
  (`XCURSOR_*`) still need a restart. The process environment is not
  mutated at runtime — that's unsafe once worker threads are running —
  so the values are layered onto each child instead.

**Only a restart re-runs:** `startup` (those commands are one-shot at
launch; editing the list just logs that it changed).

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
            vrr = "auto",  -- "auto" (default) | "always" | "off"
            hdr = false,   -- default false; true = 10-bit Rec.2020/PQ signal (see note below)
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
    -- Or any image/gif/video (needs the ffmpeg package; videos loop).
    -- Paths are literal — use os.getenv for $HOME, like the screenshot dir:
    -- wallpaper = { type = "media", path = os.getenv("HOME") .. "/Pictures/bg.mp4", mode = "fill" },
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

-- Window opacity + Kawase backdrop blur. Default: opaque windows and no layer
-- blur. `blur.layers` is a list of layer-shell namespaces to frost behind
-- (substring match) — run `libreland msg layers` to see the names in use.
-- Set opacity < 1 and/or blur.windows = true to frost behind windows too.
decoration = {
    opacity = 1.0,
    blur = { enabled = true, layers = { "rofi" }, windows = false, passes = 3, radius = 5.0 },
}

-- Environment variables layered onto every child we spawn (startup
-- commands, `spawn` binds, the idle lock command, `libreland msg
-- spawn`). Handy for theming hints. Edits apply to children spawned
-- after the reload; `XCURSOR_*` is also read by the compositor itself
-- (for its own pointer), which only re-reads it on restart.
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

-- Run Xwayland at startup for X11 app support (default true).
-- Native integration: Libreland is the X11 window manager itself.
-- Toggling on a live reload starts/stops the Xwayland server.
xwayland = true

-- Built-in idle handling (off by default). Lock the session and/or
-- power the screens off after a period of no input; any input wakes
-- the screens. A `0` or omitted timeout disables that action.
idle = {
    lock_after_secs       = 300,           -- spawn lock_command after 5 min idle
    screen_off_after_secs = 600,           -- DPMS the outputs off after 10 min
    lock_command          = "swaylock -f",  -- a lock client (ext-session-lock-v1)
}
```

### env

| Field | Default      | State | Notes                                                                          |
| ----- | ------------ | ----- | ------------------------------------------------------------------------------ |
| `env` | `{}` (empty) | ✅    | Map of `NAME = "value"` pairs. At startup they're exported into the compositor's process (so all clients inherit them); on a live reload they're layered onto each child we spawn from then on (already-running clients are untouched). Names can't be empty or contain `=`/NUL. The process environment is never mutated at runtime — `std::env::set_var` is unsafe once worker threads run — so a reload only changes what new children receive. |

`XCURSOR_THEME` and `XCURSOR_SIZE` set here do double duty: clients
inherit them *and* the compositor reads them for its own pointer
cursor, so `env = { XCURSOR_THEME = "Breeze_Light" }` themes both. The
compositor reads those two once at startup, so changing the cursor
theme/size needs a restart even though other `env` edits apply live to
new children.

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
| `xwayland` | `true`  | ✅    | Run Xwayland at startup so X11 apps work. Libreland spawns a rootless `Xwayland` on a free X display, acts as its **window manager in-process** (native integration, no `xwayland-satellite`), and exports `$DISPLAY` to children. If `Xwayland` isn't installed it's logged and skipped (never fatal). Toggling on a live reload starts or stops the server — turning it **off** disconnects any running X11 clients from their server. |

XWayland runs **rootless** with Libreland as its native X11 window
manager: X windows enter the same tiling layout as Wayland windows
(tile, float, fullscreen, workspaces, IPC — all identical), and
override-redirect windows (menus, tooltips) draw topmost like popups.
Scaling is native: Xwayland is given a *client scale* equal to the
primary output's scale, so X windows render their buffers at physical
resolution (pixel-sharp, no upscale) and X apps are told the matching
DPI via XSETTINGS `Xft/DPI` (96 × scale — e.g. `144` at 1.5×) plus
cursor theme/size (`Gtk/CursorThemeName`/`Gtk/CursorThemeSize`).
Cursors stay consistent because X apps' cursors arrive through the
normal `wl_pointer` path, the X root cursor is uploaded from
Libreland's own cursor theme, and `XCURSOR_SIZE` is pinned to the
physical cursor size. Clipboard and primary selection are bridged both
directions in-process. Requires `Xwayland` installed (the standalone
X server, usually packaged as `xorg-xwayland`).

### idle

Built-in idle handling — lock the session and/or power the screens off
after a stretch of no input. **Off by default** (omit the `idle` table
to disable it entirely). Any input (key, pointer motion, button) resets
the idle timer and wakes powered-off screens.

| Field                   | Default | State | Notes                                                                                                              |
| ----------------------- | ------- | ----- | ---------------------------------------------------------------------------------------------------------------- |
| `lock_after_secs`       | `nil`   | ✅    | Seconds of inactivity before `lock_command` is spawned. `0` or omitted = never lock. Negative is an error.        |
| `screen_off_after_secs` | `nil`   | ✅    | Seconds of inactivity before the outputs are powered off via DPMS. `0` or omitted = never power off. Any input wakes them. |
| `lock_command`          | `nil`   | ✅    | Command spawned at the lock threshold — whitespace-split into program + args (no shell), same rules as `startup`. Typically a lock client speaking `ext-session-lock-v1` (e.g. `swaylock`, `quickshell`). Without it, `lock_after_secs` does nothing. |

Read live each idle tick, so edits apply without a restart. The lock is
spawned at most once per idle period; it re-arms after the session
unlocks.

**Idle inhibitors are honoured.** While any client holds a
`zwp_idle_inhibitor_v1` (video players, presentation tools, etc. create
one during playback), both the lock and the screen-off are suppressed,
and the idle countdown restarts cleanly once the inhibitor goes away —
so watching a video doesn't blank or lock the screen.

```lua
idle = {
    lock_after_secs       = 300,            -- lock after 5 min idle
    screen_off_after_secs = 600,            -- screens off after 10 min
    lock_command          = "swaylock -f",
}
```

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
| `"togglefloating"`  | Flip the focused window between tiled and floating. A newly floating window centres at ~70% of its previous cell. Alias: `"toggle_floating"`. |
| `"togglefullscreen"`| Flip the focused window in/out of fullscreen. Aliases: `"toggle_fullscreen"`, `"fullscreen"`.                |
| `"close"`           | Politely ask the focused toplevel to close (`xdg_toplevel.close`). The client runs its own close path, so it may prompt or ignore the request. Aliases: `"closewindow"`, `"close_window"`, `"kill"`. |
| `"spawn"`           | Run an arbitrary command. Requires an additional `command = "…"` field on the bind table; the string is whitespace-split into program + args, children inherit our env (so `$WAYLAND_DISPLAY`, the configured `env`, and X `$DISPLAY` reach them). Wrap with `"sh -c '…'"` for shell features (pipes, env, `&`). |

(More actions land as features grow: `"reload"`, `"change_vt"`, …)

### Schema reference

Every field, its default, and whether it's plumbed all the way into
the runtime today (✅) or just held in `Config` for a later consumer
(⏳). Lua can set every field regardless.

### monitors

| Field                    | Default  | State | Notes                                                                                                                                                                                                                                                                                                          |
| ------------------------ | -------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `outputs[name].mode`     | `nil`    | ✅    | `{ width = …, height = …, refresh_mhz = … }` to force a mode. The override is matched against the EDID mode list by size and refresh (mHz); on a miss it logs and falls back to the EDID-preferred mode. `nil` uses EDID-preferred directly.                                                                   |
| `outputs[name].position` | `nil`    | ✅    | Top-left of this output in the virtual layout, in *logical* pixels (`{ x = …, y = … }`). `nil` falls back to the auto left-to-right layout. Mixing configured and auto-positioned outputs is fine. Outputs are never allowed to overlap: a configured position is honoured exactly unless it would collide with an already-placed output (e.g. a live `scale` change widens a monitor past its neighbour's `x`), in which case it's nudged right just enough to clear the collision — only on the X axis, so vertical/stacked layouts (same `x`, different `y`) keep their exact position. The shift is logged. |
| `outputs[name].scale`    | `1.0`    | ✅    | Fractional scale. The renderer scales every layout coordinate from compositor (= logical) to physical by this factor. Clients see the exact fractional value via `wp_fractional_scale_manager_v1` and a rounded integer fallback via `wl_output.scale`; `wp_viewporter` is advertised so fractional-aware clients can map their oversized buffer down to the logical rect (without it their content composites at the wrong size). Must be positive. Per-surface scale tracking is single-output for now — every surface gets the primary's scale until per-output workspaces ship. |
| `outputs[name].vrr`      | `"auto"` | ✅    | Variable Refresh Rate (adaptive-sync / FreeSync / G-Sync) policy. `"auto"` enables VRR only while a window fills this output (fullscreen or maximized) — where it actually helps (games, fullscreen video) — and disables it on the desktop, avoiding the flicker some panels show under idle VRR. `"always"` keeps it on; `"off"` never uses it. A no-op on outputs whose connector doesn't advertise adaptive-sync (logged at startup as `vrr_support=NotSupported`). On DisplayPort toggling is seamless; on HDMI the kernel currently needs a modeset (brief blink) to switch, which Libreland performs automatically. |
| `outputs[name].hdr`      | `false`  | ✅    | Enable HDR on this output. `true` requests a 10-bit scanout buffer and folds the connector's HDR properties (`Colorspace=BT2020_RGB`, `max bpc=10`, `HDR_OUTPUT_METADATA` — PQ / Rec.2020) into the same atomic modeset that brings the pipe up. A no-op (output stays SDR, logged) on connectors/drivers that don't expose the HDR properties or reject 10-bit. Toggling at runtime rebuilds the output's swapchain (brief modeset/blink), same as a `mode` change. **Turning HDR off** currently does not actively clear the signalling, so the panel may stay in HDR mode until the compositor restarts. Requires the vendored smithay patch under `vendor/smithay` (upstream smithay 0.7 can't attach these properties to its modeset, and a separate side-channel commit wedges the display). Clients detect this output's HDR via `wp_color_management_v1` (so Proton with `PROTON_ENABLE_WAYLAND=1 PROTON_ENABLE_HDR=1` enables HDR). Compositing is fully colour-managed **per surface**: the scene is composited in a linear BT.2020 fp16 buffer (1.0 = 10000 cd/m²) — SDR sources decoded from sRGB and mapped to `sdr_reference_white`, HDR (PQ) client surfaces decoded from PQ — then PQ-encoded to the 10-bit scanout. Rounded corners, borders and blur are linear-aware; screenshots of an HDR output are tonemapped (BT.2020 linear → clamped → sRGB) to an 8-bit buffer so captures look like SDR. If the GPU can't allocate the fp16 buffer the output falls back to SDR for that frame (logged). HDR (colour-managed) windows are decorated too — their per-window offscreen is fp16 and the surface is decoded into it, so rounded corners/border/blur composite in linear. Limitations: an HDR window doesn't appear in *another* blurred window's backdrop (it still gets its own background blur); HDR↔SDR blending at overlaps isn't linear; turning HDR off doesn't clear signalling until restart. |
| `outputs[name].sdr_reference_white` | `203` | ✅ | How bright (cd/m²) SDR content is mapped to inside the HDR signal on this output — i.e. how bright the desktop/SDR apps look in HDR mode. `nil` uses the BT.2408 standard 203 cd/m². Raise it (e.g. `250`) if SDR looks too dim. Only meaningful while `hdr = true`; applies live on config reload. |
| `outputs[name].sdr_saturation` | `1.0` | ✅ | Saturation multiplier for SDR content in HDR mode. `1.0` is colorimetrically accurate, but mapping BT.709 into the wider BT.2020 container makes SDR look slightly tame on a wide-gamut panel, so raise it (e.g. `1.1`–`1.3`) to punch SDR up. Luma-preserving; applied to SDR sources only (HDR content untouched). Must be positive. Only meaningful while `hdr = true`; applies live on config reload. |
| `primary`                | `nil`    | ✅    | Connector name of the primary output. The tile area's bounds + the initial cursor position come from this output. `nil` falls back to the first connected output in DRM enumeration order.                                                                                                                      |

### input

| Field                  | Default  | State                            | Notes                                                                                      |
| ---------------------- | -------- | -------------------------------- | ------------------------------------------------------------------------------------------ |
| `repeat_rate`          | `25`     | ✅                               | Repeats per second after the delay elapses. 25 matches X11's classic default. Sent to the seat keyboard; re-applied live on reload. |
| `repeat_delay`         | `600`    | ✅                               | Milliseconds before repeat fires. Re-applied live on reload.                               |
| `keyboard_layout`      | `""`     | ✅                               | xkb RMLVO layout. Empty defers to `$XKB_DEFAULT_LAYOUT` / system default. On a live reload the seat keymap (what clients receive) and the compositor's own hotkey-matching keymap are both rebuilt. |
| `mouse_accel_profile`  | `"flat"` | ✅ (applied per pointer device)  | `"flat"` (1:1, no ramp) or `"adaptive"` (libinput's curve, system default). Re-applied to every connected pointer on reload. |
| `mouse_accel_speed`    | `0.0`    | ✅ (applied per pointer device)  | libinput speed in `[-1.0, 1.0]`. `0.0` is neutral; with `"flat"` this is "no extra sensitivity". Re-applied live on reload. |
| `focus_model`          | `"hover"`| ✅                               | `"hover"`: keyboard focus follows the surface under the cursor on every motion event. `"click"`: focus only changes on a pointer-button press. New windows take focus on map either way. |

### binds

A list of keybindings. A press matches when its xkb keysym equals the
binding's `keysym` **and** every modifier in the binding's `mods` mask is
held. Extras like `NumLock` are tolerated. First match wins.

Built-in defaults:

- `Super+Shift+E → exit`
- `Super+F → togglefloating`
- `Super+F11 → togglefullscreen`
- `Super+C → close`

Your `binds` table is **merged on top of** these defaults, not
swapped in for them: a bind whose trigger (`mods` + `key`) matches a
default overrides that default's action, and any default you don't
touch stays active. So adding a single `Super+Space` bind keeps
`Super+Shift+E` and `Super+F` working.

Available actions today: `exit`, `togglefloating`, `togglefullscreen`,
`close`, `spawn`. The list grows as we add `reload`, `change_vt`, …

### misc

| Field        | Default            | State | Notes                                                       |
| ------------ | ------------------ | ----- | ----------------------------------------------------------- |
| `wallpaper`  | vertical gradient  | ✅    | A flat fill, or a media file (image/gif/video). See below.  |

`misc.wallpaper` is a table whose `type` selects the kind:

- **`"solid"`** — `{ type = "solid", color = { r, g, b } }`. RGB in `[0, 1]`.
- **`"vertical_gradient"`** — `{ type = "vertical_gradient", top = { r, g, b }, bottom = { r, g, b } }`.
- **`"media"`** — `{ type = "media", path = "…", mode = "fill" }`. Decodes any
  file FFmpeg can read (png/jpg/webp/… , gif, mp4/webm/…) via libav and draws
  it per output. Videos and gifs **animate and loop**; a still image is decoded
  once. `mode` (default `"fill"`) controls fitting:
  - `"fill"` / `"cover"` — scale to cover the screen, cropping the overflow (no bars).
  - `"fit"` / `"contain"` — scale to fit entirely, letterboxing the remainder.
  - `"stretch"` — fill exactly, ignoring aspect ratio.
  - `"center"` — native size, centred (cropped if larger than the output).

  Requires the **`ffmpeg`** package at runtime. Decoding is software (CPU) for
  now. The source is downscaled to your largest output, then scaled per-monitor
  on the GPU. The rounded-corner cutout falls back to black behind a media
  wallpaper; a decode failure falls back to a flat fill (logged) rather than a
  black screen. Live-reloads on change.

```lua
-- a flat fill
misc = { wallpaper = { type = "solid", color = { 0.10, 0.10, 0.12 } } }

-- an image, gif, or video
misc = { wallpaper = {
    type = "media",
    path = "/home/me/Pictures/wallpaper.jpg",  -- or a .mp4 / .gif / .webm
    mode = "fill",
} }
```

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
| `layers`   | `{}`    | ✅    | List of layer-shell **namespaces** to blur behind (a surface matches if its namespace *contains* any entry, so `"quickshell"` matches `"quickshell-bar"`). Empty = no layer blur. Run **`libreland msg layers`** to discover the namespaces in use (rofi → `"rofi"`, etc.). Sampled against the whole desktop beneath them. |
| `windows`  | `false` | ✅    | Blur behind **windows**. Tiled windows blur against the base (wallpaper + lower layers); floating windows blur against the base **plus the tiled windows underneath**, so a float reveals a blurred copy of the windows it covers. |
| `passes`   | `3`     | ✅    | Dual-filter passes — each is a downsample + later upsample. More passes = a wider, softer (and costlier) blur. `0` disables. `0..=10`.          |
| `radius`   | `5.0`   | ✅    | Per-tap sample offset in pixels; scales the blur's spread. `>= 0`.                                                                             |

Blur only runs when something translucent is on screen that needs it: the
backdrop is snapshotted per z-band (base → +tiled → +floating/maximized)
and each band is blurred at most once, so nothing is ever double-blurred.
Surface alpha isn't probed, so a mapped opaque panel/window still pays for
its tier while it's up — the cost is bounded.

**Layer blur follows the panel's real shape.** The blurred backdrop behind
a layer surface is alpha-masked by the panel's own buffer, so wherever the
client leaves pixels transparent — rounded corners of any radius, pill
shapes, cut-outs — the sharp desktop shows instead of a square block of
frost. The mask is treated as *coverage*: a translucent panel body still
gets the full frost behind it (only near-invisible pixels, alpha < 0.25,
fade the frost proportionally), and the shape's antialiased edge blends
out cleanly. Nothing to configure or keep in sync. Blur behind *windows*
is clipped to the same rounded rect the compositor draws
(`border.rounded_corners`).

```lua
decoration = {
    opacity = 0.9,                    -- windows slightly see-through (1.0 = opaque)
    blur = {
        -- enabled = false,           -- uncomment to turn all blur off
        layers  = { "rofi" },         -- namespaces to frost behind (see `libreland msg layers`)
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

## Clipboard & selections

Both the regular clipboard (`Ctrl+C`/`Ctrl+V`) and the primary
selection (highlight-to-copy, middle-click-to-paste) work out of the
box — `wl_data_device_manager` and `zwp_primary_selection_v1` are
advertised, and drag-and-drop rides the same machinery (the drag icon
is composited at the cursor).

**Copies survive the source app closing.** In stock Wayland a selection
is owned by the client that set it, so it dies when that client exits.
Libreland avoids that "copy something, close the app, paste is empty"
trap by eagerly **caching every selection** (clipboard *and* primary)
and taking server-side ownership of it — the built-in equivalent of
[`wl-clip-persist`](https://github.com/Linus789/wl-clip-persist), no
external daemon needed. A single copy larger than **128 MiB** isn't
cached (the source keeps ownership — normal Wayland behaviour, just no
cross-close persistence) so a huge copy can't balloon compositor memory.

**Clipboard managers work** via the data-control protocols. Both
[`zwlr_data_control_v1`](https://wayland.app/protocols/wlr-data-control-unstable-v1)
(v2, what most current tools target) and the standardized successor
[`ext_data_control_v1`](https://wayland.app/protocols/ext-data-control-v1)
(v1) are advertised, with the primary selection exposed through them as
well. So `cliphist`, `clipman`, `copyq`, and `wl-paste --watch` can
observe every new selection and set their own. Any client may bind them
(these protocols grant unrestricted clipboard read by design).

The [built-in screenshot tool](#screenshots-built-in) can put a PNG
straight on the clipboard (`clipboard = true`), served as `image/png`.

## Control IPC

Libreland exposes a control socket for querying and driving the
compositor — the same idea as `swaymsg` / `hyprctl` / `niri msg`. The
bundled `libreland msg` subcommand is the client; bars and scripts can
also speak the wire protocol directly.

### Socket & protocol

On startup the compositor binds a Unix socket at
`$XDG_RUNTIME_DIR/libreland-<wayland-display>.sock` and exports its path
as `$LIBRELAND_SOCKET`, which every child (terminals, `spawn` binds,
startup commands) inherits — so `libreland msg` run from inside the
session finds it with no configuration.

The protocol is newline-delimited JSON: one request object per line in,
one reply per line out. Each reply is a serialized `Result`: `{"Ok":…}`
on success, `{"Err":"message"}` on failure. A request is tagged on a
`cmd` field, e.g. `{"cmd":"windows"}`. You can drive it by hand:

    echo '{"cmd":"focused-window"}' | socat - UNIX-CONNECT:$LIBRELAND_SOCKET

### `libreland msg`

    libreland msg [--json] <command> [args…]

`--json` prints the raw JSON reply instead of formatted text (for
scripting). Actions succeed silently and fail with a message on stderr +
a non-zero exit. Run `libreland msg --help` (or `… <command> --help`) for
the full usage.

**Queries**

| Command            | Result                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------ |
| `version`          | Compositor name + version.                                                                  |
| `outputs`          | Connected outputs: make/model, mode, refresh, scale, logical position/size, active workspace. |
| `workspaces`       | Every workspace across all outputs: output, index, active, window count.                    |
| `layers`           | Every live layer-shell surface: namespace, layer, output, size, keyboard, exclusive zone. Use it to find names for `blur.layers`. |
| `windows`          | Every managed window: stable id, app-id, title, output, workspace, geometry, state flags, **pid**. |
| `focused-window`   | The keyboard-focused window (alias `focused`).                                               |
| `capture-window <id> [--max N]` | Render a window (any workspace/output) to a PNG thumbnail and print its path. `--max` caps the longest side (default 512). |
| `binds`            | The configured keybindings.                                                                  |
| `cursor`           | The pointer position: global logical coordinates, the output under it, and output-local coordinates. |

**Actions** — windows are addressed by the stable **id** from `windows`
(or the focused window when the id is omitted):

| Command                                      | Effect                                                              |
| -------------------------------------------- | ------------------------------------------------------------------- |
| `focus-window <id>`                          | Focus a window, revealing its workspace first.                      |
| `close [id]`                                 | Ask a window to close.                                              |
| `toggle-floating [id]`                       | Flip tiled ↔ floating.                                              |
| `toggle-fullscreen [id]`                     | Flip fullscreen.                                                    |
| `toggle-maximized [id]`                      | Flip maximized.                                                     |
| `focus-workspace <N\|next\|prev> [--output NAME]` | Switch a workspace (the primary output unless `--output` is given). |
| `move-to-workspace <N\|next\|prev> [id]`     | Move a window to a workspace and follow it.                          |
| `spawn <cmd…>`                               | Run a program (everything after `spawn` is the argv).               |
| `reload`                                     | Re-read the config file now.                                        |
| `exit`                                       | Quit the compositor.                                                |

`move-to-workspace` only acts on a window that's currently on a visible
(active) workspace.

### Event stream

    libreland msg subscribe [KINDS…]

Holds the connection open and prints one event per line as state changes
(add `--json` for raw JSON lines — what a bar consumes). On connect you
immediately get a snapshot (`window-focused` + `workspaces-changed`) so a
bar renders correctly from the start. With no kinds listed every event is
streamed; otherwise only the named ones.

| Event                | Fires when                                                                       |
| -------------------- | -------------------------------------------------------------------------------- |
| `window-opened`      | A window maps. Payload: the window.                                               |
| `window-closed`      | A window unmaps. Payload: its id.                                                 |
| `window-focused`     | Keyboard focus moves (and when the focused window's *title* changes). Payload: the window or null. |
| `workspaces-changed` | A workspace is switched, added, removed, or its window count changes. Payload: the full workspace list. |

Raw event lines are internally tagged on an `event` field, e.g.
`{"event":"window-focused","window":{…}}`. For a full window list, query
`windows` once and then track `window-opened` / `window-closed`.

### Examples

    libreland msg windows                 # list windows + their ids
    libreland msg focus-window 3          # focus window id 3
    libreland msg focus-workspace next    # next workspace on the primary output
    libreland msg move-to-workspace 2     # move the focused window to workspace 2
    libreland msg spawn kitty --hold      # launch a program
    libreland msg --json subscribe        # live event feed for a bar

A minimal "focused window title" bar module:

    libreland msg --json subscribe window-focused | while read -l line
        echo $line | jq -r '.window.title // "—"'
    end

## Wayland protocols

The globals Libreland advertises to clients. (Core globals like
`wl_compositor`, `wl_subcompositor`, `wl_shm` and `wl_seat` are always
present and elided here.)

| Global                                                         | Purpose                                                                 |
| -------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `wl_output` + `xdg_output_manager_v1`                          | Output geometry, mode, scale, logical position.                         |
| `xdg_wm_base` (xdg-shell)                                      | Application windows (`xdg_toplevel` / `xdg_popup`).                      |
| `xdg_activation_v1`                                            | A client requests focus/raise for a surface (link opens → browser raises). Honoured as reveal + focus; stale tokens (>10 s) ignored. |
| `zwlr_layer_shell_v1`                                          | Bars, panels, launchers, lock screens, OSDs.                            |
| `zxdg_decoration_manager_v1` + `org_kde_kwin_server_decoration` | Decoration negotiation — both advertise a **Server** default, so toolkits drop their CSD titlebars (Libreland draws none). |
| `wp_fractional_scale_manager_v1`                               | Exact fractional output scale to clients.                               |
| `wp_viewporter`                                                | Buffer crop/scale — required for fractional scaling.                    |
| `zwp_linux_dmabuf_v1` (v5 default feedback, v3 fallback)       | GPU buffer sharing (GPU-composited + XWayland/glamor clients).          |
| `wp_cursor_shape_v1`                                           | Named cursor shapes the compositor themes.                              |
| `zwp_relative_pointer_manager_v1`                              | Raw relative motion deltas (mouse-look in games).                       |
| `zwp_pointer_gestures_v1`                                      | Touchpad pinch / swipe / hold gestures to clients (browser pinch-zoom, GTK swipe). |
| `zwp_pointer_constraints_v1`                                   | Pointer lock / confinement (FPS games).                                 |
| `wp_color_manager_v1` (colour-management-v1, staging, v1)      | Clients detect an output's HDR colour properties and tag surfaces with an image description (BT.2020/PQ). Enables HDR detection for Proton/mpv. Parametric descriptions only (no ICC); per-surface state feeds the HDR colour pipeline. |
| `wp_content_type_v1`                                           | Clients hint a surface's content type (game / video / photo). Advertised so toolkits/Proton can tag content; read from surface cached state to drive future per-content behaviour (e.g. tearing / scanout choices). |
| `wp_presentation`                                              | Per-frame presentation feedback (`CLOCK_MONOTONIC`). Each presented surface gets the real DRM page-flip timestamp + sequence + refresh on the matching vblank, with `HwClock`/`HwCompletion` flags when the kernel supplies a monotonic flip time, and `ZeroCopy` when the surface was direct-scanned. Lets clients (games/video) pace frames accurately. |
| `linux-drm-syncobj-v1` (`wp_linux_drm_syncobj_manager_v1`)     | Explicit GPU synchronisation (timeline syncobjs) — used heavily by Proton/DXVK/Vulkan. A committed buffer's **acquire** fence gates the commit (held until the GPU finishes rendering) so the compositor never composites or scans out a half-rendered buffer (no tearing); the **release** fence is signalled when the buffer leaves the screen. Advertised only when the DRM device supports `syncobj_eventfd` (probed at startup); otherwise clients fall back to implicit dma-buf sync. |
| `zwp_idle_inhibit_manager_v1`                                  | Clients (e.g. video players) inhibit idle while a surface is up — see [idle](#idle). |
| `ext_idle_notifier_v1`                                         | Idle daemons (e.g. swayidle) learn when the user goes idle; paused while an inhibitor is active. |
| `wl_data_device_manager`                                       | Clipboard + drag-and-drop.                                              |
| `zwp_primary_selection_v1`                                     | Primary (middle-click) selection.                                       |
| `zwlr_data_control_v1` (v2) + `ext_data_control_v1` (v1)       | Clipboard managers — see [Clipboard & selections](#clipboard--selections). |
| `ext_session_lock_v1`                                          | Screen lockers (used by the [idle](#idle) locker). **While a lock is active, all user keybinds are suppressed** — `exit`, `spawn`, and screenshot binds can't fire, so there's no way to bypass the lock from the keyboard; every key forwards to the locker for password entry. |
| `zwlr_screencopy_v1`                                           | Output capture — screenshots + screen sharing via the portal.           |