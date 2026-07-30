-- Libreland — example configuration.
--
-- Copy this to ~/.config/libreland/config.lua and edit. You do NOT need to
-- keep all of it: every setting below is already the default, so anything you
-- delete simply keeps working. The file exists to show you what exists.
--
-- Reload happens automatically on save (or `libreland msg reload`). A file
-- that fails to parse is logged and ignored, leaving your session untouched.
--
-- The full reference, including everything not shown here, is Documentation.md.

-- ---------------------------------------------------------------------------
-- Monitors
-- ---------------------------------------------------------------------------
-- Outputs are keyed by connector name — run `libreland msg outputs` to see
-- yours. Everything is optional; unlisted outputs use their preferred mode at
-- scale 1.0, laid out left to right.
monitors = {
    -- primary = "DP-1",
    outputs = {
        -- ["DP-1"] = {
        --     mode = { width = 2560, height = 1440, refresh_mhz = 144000 },
        --     position = { x = 0, y = 0 },
        --     scale = 1.0,
        --     vrr = "auto",          -- "auto" (fullscreen only) | "always" | "off"
        --     hdr = false,
        --     sdr_reference_white = 203,   -- cd/m², how bright SDR looks in HDR
        --     sdr_saturation = 1.0,
        -- },
    },
}

-- ---------------------------------------------------------------------------
-- Keybindings
-- ---------------------------------------------------------------------------
-- Your binds are merged over the built-in defaults, and a bind on the same
-- mods+key replaces the default. Modifiers: Super, Alt, Ctrl, Shift.
-- Key names are xkb keysyms ("E", "Return", "F1", "space", "Print").
binds = {
    { mods = { "Super", "Shift" }, key = "E",      action = "exit" },
    { mods = { "Super" },          key = "Q",      action = "close" },
    { mods = { "Super" },          key = "F",      action = "togglefloating" },
    { mods = { "Super" },          key = "F11",    action = "togglefullscreen" },
    { mods = { "Super" },          key = "Return", action = "spawn", command = "kitty" },

    -- A bind on a bare modifier fires as a *tap* — press and release it
    -- alone, with nothing else pressed.
    -- { mods = {}, key = "Super_L", action = "spawn", command = "rofi -show drun" },
}

-- Super+1..9 switch workspace, Super+Shift+1..9 move the focused window there.
-- Written as a loop because nine near-identical pairs is nine chances to typo.
-- `workspace` counts from 1, matching the key you press.
for i = 1, 9 do
    table.insert(binds, {
        mods = { "Super" }, key = tostring(i),
        action = "workspace", workspace = i,
    })
    table.insert(binds, {
        mods = { "Super", "Shift" }, key = tostring(i),
        action = "movetoworkspace", workspace = i,
    })
end

-- ---------------------------------------------------------------------------
-- Input
-- ---------------------------------------------------------------------------
input = {
    keyboard_layout = "us",      -- xkb layout, e.g. "us", "se", "de"
    repeat_rate = 25,            -- repeats per second
    repeat_delay = 600,          -- ms before repeat starts
    numlock = false,
    focus_model = "hover",       -- "hover" (follows the pointer) | "click"
    mouse_accel_profile = "adaptive",  -- "adaptive" | "flat"
    mouse_accel_speed = 0.0,     -- -1.0 .. 1.0
    -- Super+scroll switches workspace, Super+Shift+scroll moves the focused
    -- window between them. Set false if you drive workspaces from keybinds
    -- and would rather the wheel never move you by accident.
    scroll_workspaces = true,
}

-- ---------------------------------------------------------------------------
-- Layout & borders
-- ---------------------------------------------------------------------------
layout = {
    gaps_outer = 8,   -- px between the tile area and the screen edge
    gaps_inner = 3,   -- px between neighbouring tiles
}

border = {
    width = 1,
    rounded_corners = 0,   -- px corner radius; 0 disables
    -- A fill is either a solid colour or a vertical gradient. Components are
    -- 0.0 .. 1.0, not 0 .. 255.
    active   = { type = "solid", color = { 0.40, 0.60, 0.90 } },
    inactive = { type = "solid", color = { 0.25, 0.25, 0.28 } },
}

-- ---------------------------------------------------------------------------
-- Animations
-- ---------------------------------------------------------------------------
-- `enabled = false` at the top level switches everything off. Each animation
-- takes `duration` (ms), `curve`, and its own `enabled`.
--
-- Curves: "linear", "ease-in", "ease-out", "ease-in-out", or four numbers for
-- a custom cubic bezier, e.g. curve = { 0.05, 0.9, 0.1, 1.0 }.
animations = {
    enabled = true,

    window_open   = { duration = 250, curve = "ease-out"    },
    window_close  = { duration = 200, curve = "ease-in"     },
    -- Position and size animate separately. `window_resize` inherits from
    -- `window_move` when you don't set it, so setting only `window_move`
    -- still moves both.
    window_move   = { duration = 250, curve = "ease-out"    },
    window_resize = { duration = 250, curve = "ease-out"    },

    -- Layer surfaces: bars, launchers, notifications. They fade while sliding
    -- a short way from whichever screen edge they sit against; one that isn't
    -- against an edge just fades.
    layer_open    = { duration = 180, curve = "ease-out"    },
    layer_close   = { duration = 150, curve = "ease-in"     },

    -- The border colour crossfading as focus moves, rather than switching in
    -- one frame.
    focus         = { duration = 150, curve = "ease-out"    },

    workspace = {
        duration = 300,
        curve = "ease-in-out",
        direction = "vertical",   -- or "horizontal"
        -- Going back can have its own feel; inherits from the above if unset.
        -- back = { duration = 200, curve = "ease-out" },
    },
}

-- ---------------------------------------------------------------------------
-- Decoration
-- ---------------------------------------------------------------------------
decoration = {
    opacity = 1.0,     -- compositor alpha for windows; 1.0 = opaque
    blur = {
        enabled = false,
        passes = 2,
        radius = 2.0,
        -- Layer-shell namespaces to blur behind, e.g. { "rofi", "quickshell" }.
        layers = {},
        windows = false,
    },
}

-- ---------------------------------------------------------------------------
-- Misc
-- ---------------------------------------------------------------------------
misc = {
    wallpaper = {
        type = "vertical_gradient",
        top    = { 0.40, 0.60, 0.90 },
        bottom = { 0.10, 0.20, 0.50 },
        -- Or a solid colour:
        -- type = "solid", color = { 0.20, 0.40, 0.80 },
        -- Or any image/gif/video (needs ffmpeg; videos loop). Paths are
        -- literal, so use os.getenv for $HOME:
        -- type = "media", path = os.getenv("HOME") .. "/Pictures/bg.jpg",
        -- mode = "fill",   -- "fill" | "fit" | "stretch" | "center"
    },
    polkit_agent = true,
    -- Present a fullscreen window immediately instead of waiting for vblank:
    -- lower latency, at the cost of a tear line. "off" | "auto" | "always".
    -- "auto" only tears when the game asks (Proton/DXVK IMMEDIATE swapchains).
    tearing = "off",
}

-- ---------------------------------------------------------------------------
-- Screenshots
-- ---------------------------------------------------------------------------
-- Omit this table entirely to disable the built-in screenshot tool.
screenshot = {
    {
        key = "Print",
        mode = "region",        -- "region" | "window" | "output"
        freeze = true,          -- pause the screen while selecting
        clipboard = true,
        directory = os.getenv("HOME") .. "/Pictures/Screenshots",
    },
}

-- ---------------------------------------------------------------------------
-- Session
-- ---------------------------------------------------------------------------
xwayland = true

-- Lock and blank on inactivity. Omit to disable.
-- idle = {
--     lock_after_secs       = 300,
--     screen_off_after_secs = 600,
--     lock_command          = "swaylock -f",
-- }

-- Environment for every child the compositor spawns from now on.
env = {
    -- XCURSOR_THEME = "Adwaita",
    -- QT_QPA_PLATFORMTHEME = "qt6ct",
}

-- Run once, after the Wayland socket is listening. Restart to re-run.
startup = {
    -- "waybar",
}
