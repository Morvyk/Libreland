# Libreland's vendored smithay — every deviation from upstream

Upstream: <https://github.com/Smithay/smithay>, pinned at the commit in
[`UPSTREAM-COMMIT`](UPSTREAM-COMMIT) (master, not a release — 0.7.0 is a year
stale and upstream development is active). To rebase: check out the new
upstream commit, re-apply every change below, update `UPSTREAM-COMMIT`, and
re-verify each patch's *reason* still holds — several guard against upstream
bugs that may get fixed.

Every code patch is marked in-source with a `Libreland` comment. This file is
the index; the in-source comments carry the full rationale. Keep both in sync.

## Code patches

### 1. HDR connector properties in the atomic modeset path
`src/backend/drm/mod.rs` (reexport),
`src/backend/drm/surface/mod.rs` (`HdrMetadata`, `DrmSurface::set_hdr`),
`src/backend/drm/surface/atomic.rs` (property plumbing, ~140 lines)

Upstream smithay has **no HDR output support**. This adds an `HdrMetadata`
struct (EOTF + `HDR_OUTPUT_METADATA` blob + `Colorspace`/`max_bpc` connector
properties) and folds it into smithay's own atomic commit — the properties
must ride the SAME atomic request as the modeset or the change flickers/fails
on NVIDIA. Candidate for upstreaming (they'd need to design the API; ours is
deliberately minimal).

### 2. GLES secondary texture units
`src/backend/renderer/gles/mod.rs` — `GlesFrame::with_secondary_texture` /
`with_secondary_textures` (~70 lines)

Binds one or two extra textures on units 1/2 (with a mipmap-free filter so
single-level imports sample correctly) so a custom texture shader can read
them. Libreland's temporally-stable backdrop-blur mask (current + previous
surface alpha) needs two samplers; upstream's custom-shader API only feeds
unit 0.

### 3. Locked-modifier control on the keyboard handle
`src/input/keyboard/mod.rs` — `set_lock_modifier` (~38 lines)

Set/clear a locked xkb modifier by name (Num Lock at startup —
`input.numlock`). Upstream gained `advertise_modifier_state` (re-send only)
but still has no way to *change* the locked state.

### 4. Signal the committed syncobj release point on object destroy
`src/wayland/drm_syncobj/mod.rs` (Destroy arm, ~24 lines)

**Upstream bug, still present at the pinned commit.** A Vulkan client that
destroys its `wp_linux_drm_syncobj_surface_v1` mid-swapchain-rebuild (idTech
does constantly) then waits on every image's release point before it will
ever commit again; upstream only signals the *committed* point when a future
commit merges over it — which never comes for a client blocked in teardown:
deadlock (games froze). Mirrors upstream's own full-surface-destruction hook.
**Prime upstreaming candidate.**

### 5. Presentation-feedback discard logging
`src/wayland/presentation/mod.rs` (~12 lines, two `tracing::debug!` lines)

Names every `wp_presentation` feedback discard (surface + why) in the session
log. Present-timing consumers (Wine/NVIDIA WSI) react badly to discards; a
session log must show who received one. Diagnostics only — safe to drop on a
rebase if it ever conflicts.

### 6. Tolerate a NULL-buffer commit on a session-lock surface
`src/wayland/session_lock/surface.rs` (was `lock.rs` pre-rebase, ~15 lines)

**Upstream still posts the fatal `NullBuffer` protocol error.** Qt/quickshell
legitimately `attach(null)+commit` to unmap during `unlock_and_destroy`;
killing the locker mid-unlock left the session with no bar and can strand a
lock. We ignore the unmap instead (the lock surface simply stops updating —
it can never reveal the desktop). A second guard skips the whole commit hook
once the role object is destroyed: lockers that destroy
`ext_session_lock_surface_v1` BEFORE the final unmap commit would otherwise
die on the (post-0.7) `CommitBeforeFirstAck` check, since upstream's new
`destroyed()` handler resets `last_acked`. Upstreaming candidate, though the
spec nominally sides with upstream.

### 7. Skip layer-shell validation for a destroyed role object
`src/wayland/shell/wlr_layer/mod.rs` `pre_commit_hook`
(was `handlers.rs` pre-rebase, ~10 lines)

**Upstream still unguarded.** The pre-commit hook outlives the role object;
after `zwlr_layer_surface_v1.destroy()` a client's legitimate null-buffer
unmap commit hits the size validation against reset role state and kills the
client with a spurious protocol error. Upstreaming candidate.

### 8. X11-side-only focus helpers
`src/xwayland/xwm/surface.rs` — `x11_take_focus` / `x11_unfocus` (~50 lines)

Perform only the X11 half of focus (SetInputFocus / `WM_TAKE_FOCUS` per the
ICCCM input model, mirroring the deferred-focus-release cancel from upstream's
`KeyboardTarget::enter`). Compositors whose keyboard-focus type is
`WlSurface` deliver `wl_keyboard` events through the surface's own
`KeyboardTarget`; using `X11Surface`'s `KeyboardTarget` impl would
double-send enter/leave. Rebase note: upstream renamed
`InputMode`→`WmInputModel` and `input_mode()`→`input_model()`; the patch
follows upstream naming.

### 9. `RESOURCE_MANAGER` root property setter
`src/xwayland/xwm/mod.rs` — `X11Wm::set_resource_manager` (~17 lines)

Publish the xrdb resource database (`Xcursor.size`, `Xft.dpi`) on the root
window for X clients that don't speak XSETTINGS. Upstream has no API for it.

## Packaging deviations (not code)

- `Cargo.toml`: the upstream repo's `[workspace]` members (anvil, smallvil,
  wlcs_anvil, test_clients, smithay-drm-extras) are not vendored — replaced
  with an empty `[workspace]` table so the crate is a standalone root inside
  Libreland's tree; the `[[example]]`/`[[bench]]` target sections are removed
  (those directories aren't vendored). Dependencies and features untouched.
- Only the files the crate needs to build are vendored (`src/`, `build.rs`,
  `Cargo.toml`, `clippy.toml`, licenses/docs); no `.git`.
- `UPSTREAM-COMMIT` records the exact upstream SHA + date this tree matches.
