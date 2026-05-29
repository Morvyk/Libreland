//! Compositor-side clipboard persistence.
//!
//! In Wayland the selection (clipboard + primary) is owned by the
//! source *client*: when that client exits, the selection normally
//! dies with it. To keep "copy in one app, paste in another — even
//! after the first app closed" working without an external clipboard
//! manager, the compositor eagerly *caches* every selection and then
//! takes ownership of it as a server-side source. This is the
//! `wl-clip-persist` algorithm, built in.
//!
//! Flow on each new client selection (`new_selection`):
//! 1. Bump a per-target epoch so reads from a superseded selection are
//!    discarded.
//! 2. Defer to a calloop idle — `new_selection` fires *before* smithay
//!    stores the seat's selection, so we must wait until the new source
//!    is the seat's current one before requesting its data.
//! 3. For each offered mime type, pipe the source's data (via
//!    `request_*_client_selection`) and read it asynchronously into
//!    memory (never blocking the event loop — the source client only
//!    writes when *we* let the loop run).
//! 4. Once every mime is read, store the cache and call
//!    `set_*_selection` so the compositor owns the selection. The
//!    original client is then free to exit; we hand out the cached
//!    bytes on paste (`send_selection`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::rc::Rc;

use smithay::input::Seat;
use smithay::reexports::calloop::generic::{Generic, NoIoDrop};
use smithay::reexports::calloop::{Interest, Mode, PostAction};
use smithay::reexports::rustix;
use smithay::wayland::selection::SelectionTarget;
use smithay::wayland::selection::data_device::{
    request_data_device_client_selection, set_data_device_selection,
};
use smithay::wayland::selection::primary_selection::{
    request_primary_client_selection, set_primary_selection,
};
use tracing::{debug, warn};

use crate::{LoopData, State};

/// Read granularity for draining a source pipe. Kept at 16 KiB so the
/// per-read stack buffer stays small; the level-triggered source just
/// re-fires until the pipe drains.
const READ_CHUNK: usize = 16 * 1024;
/// Upper bound on a single cached selection (summed over mime types).
/// A copy larger than this isn't cached — the source client keeps
/// ownership (normal Wayland behaviour, just no cross-close
/// persistence) rather than letting a client balloon our memory.
const MAX_CACHE_BYTES: usize = 128 * 1024 * 1024;

/// Cached bytes for one selection target, keyed by mime type.
#[derive(Default)]
struct CachedSelection {
    data: HashMap<String, Vec<u8>>,
}

/// Compositor clipboard + primary caches with per-target epoch
/// counters. Lives on [`State`]; the renderer never touches it.
#[derive(Default)]
pub(crate) struct Selections {
    clipboard: Option<CachedSelection>,
    primary: Option<CachedSelection>,
    clipboard_epoch: u64,
    primary_epoch: u64,
}

impl Selections {
    /// Bump and return the epoch for `ty` (called when a new client
    /// selection arrives, invalidating any in-flight read).
    fn bump(&mut self, ty: SelectionTarget) -> u64 {
        let epoch = match ty {
            SelectionTarget::Clipboard => &mut self.clipboard_epoch,
            SelectionTarget::Primary => &mut self.primary_epoch,
        };
        *epoch += 1;
        *epoch
    }

    fn epoch(&self, ty: SelectionTarget) -> u64 {
        match ty {
            SelectionTarget::Clipboard => self.clipboard_epoch,
            SelectionTarget::Primary => self.primary_epoch,
        }
    }

    fn clear(&mut self, ty: SelectionTarget) {
        match ty {
            SelectionTarget::Clipboard => self.clipboard = None,
            SelectionTarget::Primary => self.primary = None,
        }
    }

    fn store(&mut self, ty: SelectionTarget, cache: CachedSelection) {
        match ty {
            SelectionTarget::Clipboard => self.clipboard = Some(cache),
            SelectionTarget::Primary => self.primary = Some(cache),
        }
    }

    fn cached_bytes(&self, ty: SelectionTarget, mime: &str) -> Option<Vec<u8>> {
        let cache = match ty {
            SelectionTarget::Clipboard => self.clipboard.as_ref(),
            SelectionTarget::Primary => self.primary.as_ref(),
        };
        cache.and_then(|c| c.data.get(mime)).cloned()
    }
}

/// Accumulator shared across the per-mime read sources of one
/// in-flight selection. `remaining` counts mime reads not yet at EOF;
/// when it hits zero the cache is finalized.
struct InProgress {
    ty: SelectionTarget,
    epoch: u64,
    mimes: Vec<String>,
    data: HashMap<String, Vec<u8>>,
    remaining: usize,
    total: usize,
    aborted: bool,
}

/// A client set the selection (`SelectionHandler::new_selection`).
/// `mimes == None` means the selection was cleared.
pub(crate) fn on_new_selection(state: &mut State, ty: SelectionTarget, mimes: Option<Vec<String>>) {
    let epoch = state.clipboard.bump(ty);
    let Some(mimes) = mimes.filter(|m| !m.is_empty()) else {
        // Cleared (or an empty offer): drop our cache so we don't serve
        // stale data. The selection is already empty on smithay's side.
        state.clipboard.clear(ty);
        return;
    };
    // `new_selection` runs before smithay stores the seat's selection,
    // so requesting the source now would read the *previous* one. Defer
    // to an idle, which runs once the new source is current.
    state
        .loop_handle
        .insert_idle(move |data| start_reads(&mut data.state, ty, epoch, mimes));
}

/// Begin draining the (now-current) selection source into a cache.
fn start_reads(state: &mut State, ty: SelectionTarget, epoch: u64, mimes: Vec<String>) {
    // A newer selection superseded this one between the commit and this
    // idle: nothing to do.
    if state.clipboard.epoch(ty) != epoch {
        return;
    }

    let progress = Rc::new(RefCell::new(InProgress {
        ty,
        epoch,
        mimes: mimes.clone(),
        data: HashMap::new(),
        remaining: 0,
        total: 0,
        aborted: false,
    }));

    let mut started = 0usize;
    for mime in mimes {
        // Read end non-blocking (we drain it from the loop); the write
        // end stays blocking so the source client writes normally.
        let (read_fd, write_fd) = match rustix::pipe::pipe_with(rustix::pipe::PipeFlags::CLOEXEC) {
            Ok(pair) => pair,
            Err(err) => {
                warn!(error = %err, "clipboard: pipe() failed");
                continue;
            }
        };
        if let Err(err) = rustix::fs::fcntl_setfl(&read_fd, rustix::fs::OFlags::NONBLOCK) {
            warn!(error = %err, "clipboard: set read end non-blocking failed");
            continue;
        }

        // The two request fns return distinct error enums; collapse to
        // ok/err. On error the request consumed and dropped write_fd.
        let requested_ok = match ty {
            SelectionTarget::Clipboard => {
                request_data_device_client_selection::<State>(&state.seat, mime.clone(), write_fd)
                    .is_ok()
            }
            SelectionTarget::Primary => {
                request_primary_client_selection::<State>(&state.seat, mime.clone(), write_fd)
                    .is_ok()
            }
        };
        if !requested_ok {
            // Source vanished or doesn't actually offer this mime; skip.
            debug!(mime, "clipboard: source request failed for mime");
            continue;
        }

        let progress = progress.clone();
        let seat = state.seat.clone();
        let mut buf: Vec<u8> = Vec::new();
        let insert = state.loop_handle.insert_source(
            Generic::new(read_fd, Interest::READ, Mode::Level),
            move |_, fd: &mut NoIoDrop<OwnedFd>, data: &mut LoopData| {
                Ok::<_, std::io::Error>(read_ready(
                    fd.as_fd(),
                    &mut buf,
                    &mime,
                    &progress,
                    &seat,
                    &mut data.state,
                ))
            },
        );
        if let Err(err) = insert {
            warn!(error = %err, "clipboard: registering read source failed");
            continue;
        }
        started += 1;
    }

    if started == 0 {
        // Couldn't read anything — leave the source as the live owner
        // (no persistence for this one) and drop any prior cache.
        state.clipboard.clear(ty);
        return;
    }
    progress.borrow_mut().remaining = started;
}

/// Drain whatever is readable on one mime pipe. Returns
/// [`PostAction::Remove`] at EOF (or error) after folding the bytes
/// into the shared accumulator and finalizing if it was the last mime.
fn read_ready(
    fd: BorrowedFd<'_>,
    buf: &mut Vec<u8>,
    mime: &str,
    progress: &Rc<RefCell<InProgress>>,
    seat: &Seat<State>,
    state: &mut State,
) -> PostAction {
    let mut chunk = [0u8; READ_CHUNK];
    loop {
        match rustix::io::read(fd, &mut chunk[..]) {
            Ok(0) => break,
            Ok(n) => {
                let oversized = {
                    let mut p = progress.borrow_mut();
                    p.total += n;
                    p.total > MAX_CACHE_BYTES
                };
                if oversized {
                    progress.borrow_mut().aborted = true;
                    finish_mime(buf, mime, progress, seat, state);
                    return PostAction::Remove;
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            // `Generic` is level-triggered, so EAGAIN just means we've
            // drained what's buffered for now; INTR retries.
            Err(rustix::io::Errno::AGAIN) => return PostAction::Continue,
            Err(rustix::io::Errno::INTR) => {}
            Err(err) => {
                debug!(error = %err, mime, "clipboard: read error; finishing mime");
                break;
            }
        }
    }
    finish_mime(buf, mime, progress, seat, state);
    PostAction::Remove
}

/// Move a finished mime's bytes into the accumulator and, when the last
/// mime completes, store the cache and take ownership of the selection.
fn finish_mime(
    buf: &mut Vec<u8>,
    mime: &str,
    progress: &Rc<RefCell<InProgress>>,
    seat: &Seat<State>,
    state: &mut State,
) {
    let done = {
        let mut p = progress.borrow_mut();
        if !p.aborted {
            p.data.insert(mime.to_owned(), std::mem::take(buf));
        }
        p.remaining = p.remaining.saturating_sub(1);
        p.remaining == 0
    };
    if done {
        finalize(progress, seat, state);
    }
}

/// All mime reads finished: if still current and not aborted, cache the
/// data and make the compositor the selection owner so it survives the
/// source client closing.
fn finalize(progress: &Rc<RefCell<InProgress>>, seat: &Seat<State>, state: &mut State) {
    let p = progress.borrow();
    if p.aborted {
        warn!(ty = ?p.ty, "clipboard: selection exceeded cache cap; not persisting");
        // Leave the live source as owner; just drop any stale cache.
        let ty = p.ty;
        drop(p);
        state.clipboard.clear(ty);
        return;
    }
    if state.clipboard.epoch(p.ty) != p.epoch {
        // Superseded mid-read by a newer copy; the newer read owns it.
        return;
    }
    if p.total == 0 {
        // Every mime read hit immediate EOF with no bytes — the source
        // closed before it answered our request (the request-to-drain
        // window). Don't take ownership of an empty selection (that
        // would make pastes silently yield nothing); leave it be, and
        // smithay drops the now-dead source on its next access.
        warn!(ty = ?p.ty, "clipboard: source produced no data (closed before writing?); not persisting");
        let ty = p.ty;
        drop(p);
        state.clipboard.clear(ty);
        return;
    }
    let ty = p.ty;
    let mimes = p.mimes.clone();
    let bytes = p.total;
    let cache = CachedSelection {
        data: p.data.clone(),
    };
    drop(p);

    state.clipboard.store(ty, cache);
    let dh = state.display_handle.clone();
    match ty {
        SelectionTarget::Clipboard => set_data_device_selection::<State>(&dh, seat, mimes, ()),
        SelectionTarget::Primary => set_primary_selection::<State>(&dh, seat, mimes, ()),
    }
    debug!(
        ?ty,
        bytes, "clipboard: cached selection; compositor now owns it"
    );
}

/// A client is reading the compositor-owned selection
/// (`SelectionHandler::send_selection`). Write the cached bytes into
/// `fd` asynchronously so a slow reader can't stall the event loop.
pub(crate) fn on_send_selection(state: &mut State, ty: SelectionTarget, mime: &str, fd: OwnedFd) {
    let Some(bytes) = state.clipboard.cached_bytes(ty, mime) else {
        // Not cached: drop `fd`, the client reads EOF (empty paste).
        debug!(?ty, mime, "clipboard: paste for uncached mime");
        return;
    };
    if let Err(err) = rustix::fs::fcntl_setfl(&fd, rustix::fs::OFlags::NONBLOCK) {
        warn!(error = %err, "clipboard: set paste fd non-blocking failed");
        return;
    }

    let mut offset = 0usize;
    let insert = state.loop_handle.insert_source(
        Generic::new(fd, Interest::WRITE, Mode::Level),
        move |_, fd: &mut NoIoDrop<OwnedFd>, _data: &mut LoopData| {
            loop {
                if offset >= bytes.len() {
                    return Ok::<_, std::io::Error>(PostAction::Remove);
                }
                match rustix::io::write(fd.as_fd(), &bytes[offset..]) {
                    Ok(0) => return Ok(PostAction::Remove),
                    Ok(n) => offset += n,
                    Err(rustix::io::Errno::AGAIN) => return Ok(PostAction::Continue),
                    Err(rustix::io::Errno::INTR) => {}
                    // Reader closed early (EPIPE) or other error: done.
                    Err(_) => return Ok(PostAction::Remove),
                }
            }
        },
    );
    if let Err(err) = insert {
        warn!(error = %err, "clipboard: registering paste-write source failed");
    }
}
