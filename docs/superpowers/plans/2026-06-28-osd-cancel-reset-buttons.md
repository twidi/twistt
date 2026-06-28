# OSD Cancel/Reset Buttons — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two clickable buttons (Reset ↻ / Cancel ✕) to the bottom-right of the otherwise click-through OSD overlay, letting the user wipe-and-stop or wipe-and-restart a transcription session from the mouse.

**Architecture:** The overlay's input region is scoped to the two button rectangles (rest stays click-through). A click is sent back to the main process over the existing Unix socket (new OSD→main direction) via a reader thread, injected into `HotKeyTask`'s event loop to keep toggle/PTT state coherent, and handled by a new cooperative `Comm.abort_session(restart)` that tears the pipeline down and (for Reset) re-triggers the same mode.

**Tech Stack:** Python (asyncio), GTK4 + gtk4-layer-shell + Cairo (OSD daemon under system Python), Unix domain socket IPC (length-prefixed JSON), evdev.

---

## Testing note (project-specific, overrides skill's TDD default)

This project has **no automated test suite** (`CLAUDE.md`: "Testing is manual") and the touched surface (GTK4/Wayland, evdev, real-time audio/WebSocket) is not unit-testable. Each task therefore ends with an explicit **manual verification** instead of a failing-test cycle, plus a commit. Keep changes small; run the app between tasks.

## Environment notes

- Work on `main` (personal project; user commits directly to main — no worktree).
- Run the app with `./twistt.py` (uv handles deps). The OSD daemon runs under `/usr/bin/python3` and is spawned automatically.
- Commit messages in English (repo convention), each ending with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.
- Provider for quick manual tests: whichever is already configured. Use batch mode to see streamed text get erased (most visible).

## File structure / responsibilities

- `twistt_osd.py`
  - `compute_button_rects(width, height)` (new module-level helper): single source of truth for the two button rectangles. Used by renderer (draw) + window (hit-test + input region).
  - `OSDRenderer`: draw the two buttons when `session_active`.
  - `OSDWindow`: dynamic input region (button rects vs empty), `Gtk.GestureClick`, hit-test, `on_action` callback.
  - `TranscriptionOSD`: wire `OSDWindow.on_action` to send `{"type":"action","action":...}` over `_client_conn`; toggle button visibility/input-region on session start/end.
- `twistt.py`
  - `OsdRunner`: socket reader thread (OSD→main), reconnect-aware; `on_action` callback.
  - `OsdTask`: pass `comm`-bound action handler to `OsdRunner`.
  - `Comm`: `abort_session(restart)`, `_abort`/`is_aborting`, `dispatch_overlay_action()` (thread-safe), overlay-action injection into `HotKeyTask`.
  - `HotKeyTask.run`: handle injected `ABORT`/`RESET` items (toggle/PTT state, PTT hold neutralization, re-trigger).
  - `BaseTranscriptionTask`: honor `_abort` in `_run_session` / delta handlers; skip `_queue_full_mode_result`.
  - `BufferTask` + `Manager`: `Reset` command + `reset_all()`; skip queued inserts while aborting.
  - `PostTreatmentTask`: honor abort (extend speculative cancel to non-speculative).

---

## Task 1: Draw the two buttons (visual only, not yet clickable)

**Files:**
- Modify: `twistt_osd.py` — add `compute_button_rects()` (module level, near other helpers), call new `OSDRenderer._draw_buttons()` at end of `OSDRenderer.draw()` (~line 225).

- [ ] **Step 1: Add the geometry helper**

Add near the top-level helpers (after imports / constants):

```python
# Button geometry (bottom-right corner). Single source of truth for
# drawing, hit-testing and the input region.
BTN_SIZE = 27
BTN_GAP = 8
BTN_MARGIN = 10

def compute_button_rects(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    """Return {'reset': (x,y,w,h), 'cancel': (x,y,w,h)} in widget coords."""
    y = height - BTN_SIZE - BTN_MARGIN
    cancel_x = width - BTN_MARGIN - BTN_SIZE
    reset_x = cancel_x - BTN_GAP - BTN_SIZE
    return {
        "reset": (reset_x, y, BTN_SIZE, BTN_SIZE),
        "cancel": (cancel_x, y, BTN_SIZE, BTN_SIZE),
    }
```

- [ ] **Step 2: Draw buttons in the renderer**

`draw()` wraps its content in a `cr.push_group()` … `cr.pop_group_to_source()` + `paint_with_alpha(self._opacity)` block (~line 227). Insert the button draw **just after the `_draw_state_indicator(...)` call (~line 225) and before `pop_group_to_source()`** — i.e. still inside the group — so the buttons inherit the overlay opacity. Drawing after the `pop_group` would paint onto the already-composited target with no opacity.

```python
if text_state.get("session_active", False):
    self._draw_buttons(cr, width, height)
```

Add the method (rounded translucent pill + vector icon; reuse `_rounded_rect`):

```python
def _draw_buttons(self, cr, width, height):
    rects = compute_button_rects(width, height)
    for name, (x, y, w, h) in rects.items():
        # pill background
        self._rounded_rect(cr, x, y, w, h, 8)
        cr.set_source_rgba(0.55, 0.63, 0.78, 0.12)
        cr.fill()
        self._rounded_rect(cr, x, y, w, h, 8)
        cr.set_source_rgba(0.60, 0.69, 0.84, 0.20)
        cr.set_line_width(1.0)
        cr.stroke()
        cx, cy = x + w / 2, y + h / 2
        cr.set_line_width(2.0)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        if name == "cancel":
            cr.set_source_rgba(0.80, 0.83, 0.90, 0.85)
            r = 4.5
            cr.move_to(cx - r, cy - r); cr.line_to(cx + r, cy + r)
            cr.move_to(cx + r, cy - r); cr.line_to(cx - r, cy + r)
            cr.stroke()
        else:  # reset: circular arrow
            cr.set_source_rgba(0.80, 0.83, 0.90, 0.85)
            r = 5.0
            cr.arc(cx, cy, r, math.radians(60), math.radians(360))
            cr.stroke()
            # arrowhead at the open end (~60°)
            ax = cx + r * math.cos(math.radians(60))
            ay = cy + r * math.sin(math.radians(60))
            cr.move_to(ax, ay); cr.line_to(ax - 3, ay - 1)
            cr.move_to(ax, ay); cr.line_to(ax + 1, ay - 3)
            cr.stroke()
```

- [ ] **Step 3: Manual verification**

Run a session (hold/toggle hotkey). Expected: two small pills appear bottom-right (↻ then ✕), only while the overlay shows an active session, and disappear when it hides. They are **not yet clickable** (clicks still pass through — input region is still empty).

- [ ] **Step 4: Commit**

```bash
git add twistt_osd.py
git commit -m "Draw cancel/reset buttons on the OSD (visual only)"
```

---

## Task 2: Scope input region to the buttons + capture clicks (hit-test → log)

**Files:**
- Modify: `twistt_osd.py` — `OSDWindow` (`_on_map_click_through` → generalize to `_apply_input_region`, add `set_buttons_active`, add `GestureClick`), `TranscriptionOSD._handle_message` (toggle button activity on session start/end).

- [ ] **Step 1: Replace the empty-region handler with a scoped one**

In `OSDWindow.__init__`, add `self._buttons_active = False` and `self.on_action = None` (callback set by `TranscriptionOSD`).

Replace `_on_map_click_through` with:

```python
def _on_map_click_through(self, _widget):
    self._apply_input_region()

def set_buttons_active(self, active: bool):
    self._buttons_active = active
    self._apply_input_region()

def _apply_input_region(self):
    surface = self.get_surface()
    if surface is None:
        return
    region = cairo.Region()
    if self._buttons_active:
        for x, y, w, h in compute_button_rects(self._width, self._height).values():
            region.union(cairo.RectangleInt(int(x), int(y), int(w), int(h)))
    surface.set_input_region(region)
```

(An empty region keeps full click-through; the union makes only the buttons catch input.)

- [ ] **Step 2: Add the click gesture + hit-test**

In `_setup_drawing_area`, after creating `self.drawing_area`:

```python
click = Gtk.GestureClick()
click.set_button(1)  # left button
click.connect("pressed", self._on_button_pressed)
self.drawing_area.add_controller(click)
```

Add the handler:

```python
def _on_button_pressed(self, _gesture, _n_press, x, y):
    if not self._buttons_active:
        return
    for name, (bx, by, bw, bh) in compute_button_rects(self._width, self._height).items():
        if bx <= x <= bx + bw and by <= y <= by + bh:
            if self.on_action:
                self.on_action(name)  # "reset" | "cancel"
            return
```

- [ ] **Step 3: Toggle button activity on session start/end**

In `TranscriptionOSD.__init__` / after window creation, set the callback (temporary log for this task):

```python
self.window.on_action = lambda action: print(f"[osd] button: {action}", flush=True)
```

In `_handle_message`, on `session_start` call `self.window.set_buttons_active(True)`; in `_session_end_hide` call `self.window.set_buttons_active(False)` (before hide). Also call `set_buttons_active(True)` in `_show` if `session_active`, and `False` in `_hide`, so the input region tracks visibility.

- [ ] **Step 4: Manual verification**

Run a session. Click each button → the daemon's stdout/log shows `[osd] button: reset|cancel`. Click anywhere else on the overlay → the click reaches the window underneath (still click-through). Between sessions (buttons hidden) → whole overlay click-through.

Note: OSD stdout is `DEVNULL` when spawned by the app; for this check run the daemon standalone, e.g. `/usr/bin/python3 twistt_osd.py` (non-daemon) and click, or temporarily route the print to the log file.

- [ ] **Step 5: Commit**

```bash
git add twistt_osd.py
git commit -m "Make OSD buttons hit-testable with a scoped input region"
```

---

## Task 3: Return channel OSD → main process (reader thread)

**Files:**
- Modify: `twistt_osd.py` — `TranscriptionOSD`: send action frame over `_client_conn`.
- Modify: `twistt.py` — `OsdRunner`: reader thread + `on_action`; `OsdTask`: bind handler to `comm`.

- [ ] **Step 1: Send the action from the OSD over the socket**

Replace the temporary lambda in `TranscriptionOSD` with a real sender:

```python
self.window.on_action = self._send_action

def _send_action(self, action: str):
    if self._client_conn is None:
        return
    try:
        data = OSDProtocol.encode_message({"type": "action", "action": action})
        self._client_conn.setblocking(True)
        self._client_conn.sendall(data)
        self._client_conn.setblocking(False)
    except (BrokenPipeError, ConnectionResetError, OSError):
        self._disconnect_client()
```

- [ ] **Step 2: Add a reader thread in `OsdRunner`**

In `OsdRunner.__init__` add: `self._reader_thread = None`, `self._reader_stop = False`, `self._on_action = None`, `self._loop = None`.

Add a method to start the reader (call it once a socket is connected, e.g. at the end of `_connect_socket` on success, guarded so it starts only once):

```python
def start_reader(self, loop, on_action):
    self._loop = loop
    self._on_action = on_action
    if self._reader_thread is None:
        self._reader_stop = False
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

def _reader_loop(self):
    buf = b""
    while not self._reader_stop:
        sock = self._socket
        if sock is None:
            time.sleep(0.2); continue
        try:
            data = sock.recv(4096)
        except OSError:
            time.sleep(0.2); continue
        if not data:
            time.sleep(0.2); continue
        buf += data
        while len(buf) >= 4:
            (length,) = struct.unpack("!I", buf[:4])
            if len(buf) < 4 + length:
                break
            payload = buf[4:4 + length]; buf = buf[4 + length:]
            try:
                msg = json.loads(payload.decode("utf-8"))
            except Exception:
                continue
            if msg.get("type") == "action" and self._on_action and self._loop:
                action = msg.get("action")
                self._loop.call_soon_threadsafe(self._on_action, action)
```

In `stop()`, set `self._reader_stop = True` before closing the socket.

Reconnect note: `_reader_loop` always reads `self._socket` fresh each iteration, so when `send_message` replaces a dead socket the reader picks up the new one (risk #5 in the spec).

- [ ] **Step 3: Bind the handler in `OsdTask`**

After `self.comm.enable_osd_queue()` in `OsdTask.run`, once the socket exists, start the reader:

```python
loop = asyncio.get_running_loop()
self._runner._connect_socket()
self._runner.start_reader(loop, self.comm.dispatch_overlay_action)
```

Add `Comm.dispatch_overlay_action(action)` (thread-safe entry point; for this task just log, real routing in Task 4):

```python
def dispatch_overlay_action(self, action: str):
    debug(f"[overlay] action received: {action}")
```

- [ ] **Step 4: Manual verification**

Run the app normally. Start a session, click Reset/Cancel → the **main** process logs `[overlay] action received: reset|cancel` (enable debug logging). Confirm clicking elsewhere still passes through and the app is otherwise unaffected.

- [ ] **Step 5: Commit**

```bash
git add twistt_osd.py twistt.py
git commit -m "Add OSD->main return channel for overlay button actions"
```

---

## Task 4: Route actions through HotKeyTask + minimal Cancel (stop the mode)

**Files:**
- Modify: `twistt.py` — `Comm`: overlay-action injection into `HotKeyTask`, `abort_session` skeleton, `_abort` event; `HotKeyTask.run`: handle injected items.

- [ ] **Step 1: Inject overlay actions into HotKeyTask's event queue**

`HotKeyTask` owns `self._event_queue` (asyncio.Queue). Register it on `Comm` so overlay actions can be pushed as synthetic items distinguishable from `(device, event)` tuples.

- In `HotKeyTask.run` start: `self.comm.register_hotkey_queue(asyncio.get_running_loop(), self._event_queue)`.
- `Comm.register_hotkey_queue(loop, queue)` stores both.
- Change `Comm.dispatch_overlay_action` to inject:

```python
def dispatch_overlay_action(self, action: str):
    # action: "cancel" | "reset"; pushed as a synthetic hotkey-queue item
    if self._hotkey_queue is not None and self._hotkey_loop is not None:
        item = ("__overlay__", action)
        self._hotkey_loop.call_soon_threadsafe(self._hotkey_queue.put_nowait, item)
```

- [ ] **Step 2: Handle the synthetic item in `HotKeyTask.run`**

Right after `device, event = item` is unpacked, intercept overlay items before evdev handling:

```python
if device == "__overlay__":
    action = event  # "cancel" | "reset"
    if not self.comm.is_session_active:
        continue
    restart = (action == "reset")
    name = next((k for k, v in F_KEY_CODES.items() if v == active_hotkey), None) if active_hotkey else None
    await self.comm.abort_session(restart=False)
    if is_toggle_mode and not restart:
        is_toggle_mode = False; active_hotkey = None; hotkey_pressed = False
        toggle_stop_time = current_time
    elif not is_toggle_mode and not restart:
        # PTT cancel: neutralize the still-held key until its KEY_UP
        ptt_aborted = True
    # (reset handling added in Task 7)
    continue
```

Note: `item = await self._event_queue.get()` returns `(device, event)`; for overlay items `device == "__overlay__"`. Declare `ptt_aborted = False` among the loop locals (line ~2521) and, in the PTT `KEY_UP` / `active_hotkey` guard branches, when `ptt_aborted` is set, swallow events for `active_hotkey` until its `KEY_UP`, then reset `ptt_aborted`, `hotkey_pressed`, `active_hotkey`.

- [ ] **Step 3: Add `Comm.abort_session` skeleton**

```python
async def abort_session(self, restart: bool):
    if not self.is_session_active:
        return
    self._abort.set()
    self.toggle_recording_internal_stop()   # clear _recording, restore ducking
    self.empty_audio_chunks()
    # (full pipeline teardown added in Task 5; text wipe in Task 6)
    self._is_session_finishing = False
    self._abort.clear()
```

Add `self._abort = asyncio.Event()` in `Comm.__init__` and `is_aborting` property (`self._abort.is_set()`). `toggle_recording_internal_stop` = the stop half of `toggle_recording(False, ...)` factored out (clear `_recording`, ducking restore) without the normal finalization that pastes results.

- [ ] **Step 4: Manual verification**

Toggle mode: start a session, click **Cancel** → recording stops, the mode turns off (state indicator clears, no further text). PTT: hold the mouse hotkey, click Cancel → recording stops and does **not** restart while held; releasing then pressing again starts a fresh session. (Text already pasted may still be present — wiped in Task 6.)

- [ ] **Step 5: Commit**

```bash
git add twistt.py
git commit -m "Route overlay actions through HotKeyTask; minimal Cancel stops the mode"
```

---

## Task 5: Complete pipeline abort (no result paste, close WS, cancel post-treatment)

**Files:**
- Modify: `twistt.py` — `BaseTranscriptionTask._run_session` + delta handlers (`_handle_new_delta`, `_handle_done_segment`), `_queue_full_mode_result`; Mistral override `_run_session` (~3552); `PostTreatmentTask` / `Comm.request_speculative_cancel`.

- [ ] **Step 1: Make `_run_session` abort-aware**

Race the receiver against the abort event; on abort, cancel both sub-tasks and skip the result:

```python
abort_waiter = create_task(self.comm.wait_for_abort())
try:
    done, _ = await asyncio.wait({receiver_task, abort_waiter}, return_when=asyncio.FIRST_COMPLETED)
finally:
    for t in (sender_task, receiver_task, abort_waiter):
        t.cancel()
    for t in (sender_task, receiver_task, abort_waiter):
        with suppress(CancelledError):
            await t
if not self.comm.is_aborting:
    await self._queue_full_mode_result(previous_transcriptions)
```

Add `Comm.wait_for_abort()` = `await self._abort.wait()`. Apply the same guard to the Mistral `_run_session` override (~3552).

- [ ] **Step 2: Suppress new output while aborting**

At the top of `_handle_new_delta` and `_handle_done_segment` (and any place that queues `InsertSegment`/`ProcessSegment`), early-return when `self.comm.is_aborting`.

- [ ] **Step 3: Cancel in-flight post-treatment**

Extend the existing speculative-cancel path to the non-speculative one: in `abort_session`, call `self.request_speculative_cancel()` and ensure `_post_process`'s `cancel_check` also returns True when `is_aborting`. In `PostTreatmentTask`, when dequeuing a command and `is_aborting`, drop it (don't run the LLM call).

- [ ] **Step 4: Wire teardown into `abort_session`** (extend Task 4 skeleton)

Between `empty_audio_chunks()` and clearing flags: drain pending `_post_commands` / `_buffer_commands` of session work (or rely on the `is_aborting` skips), request post cancel, and clear `is_speech_active` / `is_post_treatment_active`.

- [ ] **Step 5: Manual verification**

Cancel mid-recording: the WebSocket closes promptly, no final text is pasted, the session ends (state clears, `is_session_active` returns False — a subsequent hotkey works immediately). Cancel during post-treatment (release key first in full mode, then click Cancel): the LLM stream stops and the session ends cleanly. (Already-pasted text still present — Task 6.)

- [ ] **Step 6: Commit**

```bash
git add twistt.py
git commit -m "Cooperative pipeline abort: stop transcription and post-treatment without pasting"
```

---

## Task 6: Wipe already-pasted text (BufferTask.Reset)

**Files:**
- Modify: `twistt.py` — `BufferTask.Commands` (new `Reset`), `BufferTask` dispatch, `Manager.reset_all`, `BufferTask` skip while aborting; `abort_session` queues the Reset.

- [ ] **Step 1: Add the `Reset` command + manager method**

In `BufferTask.Commands` add (high seq so it sorts after the indicator at 2_000_000_000, before Shutdown at 3_000_000_000):

```python
class Reset(NamedTuple):
    seq_num: int = 2_500_000_000
```

In `Manager`:

```python
async def reset_all(self):
    async with self.lock:
        if self.text:
            await self._move_cursor_to(len(self.text))
            await self._enqueue(OutputTask.Commands.DeleteCharsBackward(len(self.text)))
        self.text = ""
        self.cursor = 0
        self.segments = {}
        self.segment_order = []
```

- [ ] **Step 2: Dispatch + skip-while-aborting in `BufferTask`**

In the `BufferTask` command dispatch (`_handle_cmd` / match): handle `Reset` → `await self.manager.reset_all()`. At the top of the dispatch, while `self.comm.is_aborting`, skip any command that is not `Reset`/`Shutdown` (drop queued inserts/corrections so they don't re-paste).

- [ ] **Step 3: Queue the Reset from `abort_session`**

In `abort_session`, after suppressing new output and before clearing flags:

```python
await self.queue_buffer_command(BufferTask.Commands.Reset())
await self.wait_keyboard_idle()  # let OutputTask drain so the wipe completes
```

Add `wait_keyboard_idle()` if not present (poll `is_keyboard_busy` / queue empty with a short timeout).

- [ ] **Step 4: Manual verification**

Batch mode (text streams in as you speak): Cancel mid-session → all pasted text **and** the `(Twistting...)` indicator are removed, cursor back to start state. Repeat in post-correct mode (raw text pasted then corrected) → fully wiped. Output mode `none` → nothing pasted, Cancel is a clean no-op. Verify no residual characters and no stuck indicator.

- [ ] **Step 5: Commit**

```bash
git add twistt.py
git commit -m "Wipe pasted text and indicator on session abort"
```

---

## Task 7: Reset = abort + re-trigger; OSD notifications; PTT hold neutralization polish

**Files:**
- Modify: `twistt.py` — `HotKeyTask.run` (reset branch + PTT swallow), `OsdTask`/`Comm` notifications.

- [ ] **Step 1: Implement Reset re-trigger in `HotKeyTask`**

Extend the overlay-item branch (Task 4 Step 2) for `restart`:

- Toggle + reset: `await abort_session(restart=False)`, then re-arm toggle immediately, bypassing `toggle_cooldown`: keep `is_toggle_mode = True`, `active_hotkey` unchanged, and call `self.comm.toggle_recording(True, name, True)`.
- PTT + reset: `await abort_session(restart=False)`, then `self.comm.toggle_recording(True, name, False)`; keep `hotkey_pressed = True` and `active_hotkey` so the held key keeps the session alive; do **not** set `ptt_aborted`.

Ensure `abort_session` has fully reset `is_session_active` to False before the re-trigger (Tasks 4–6 guarantee this) so the `is_session_active` KEY_DOWN guard / cooldown don't block the internal re-trigger.

- [ ] **Step 2: OSD notifications on abort/restart**

On Cancel, the existing session-finalization already emits `session_end` to the OSD as state clears — verify the OSD hides. On Reset, ensure a fresh `session_start` is emitted by the re-trigger path so the overlay shows a clean new session. If the display state machine doesn't emit these from an aborted session, add explicit `OsdTask`/display notifications in `abort_session` (session_end) and rely on `toggle_recording(True, ...)` for the new `session_start`.

- [ ] **Step 3: Manual verification (full matrix)**

For each mode (toggle / PTT) × output mode (batch, full, none, post-correct):
1. Cancel mid-recording → wiped + mode off (PTT: no auto-restart while held; new press works).
2. Cancel during post-treatment → stream stops, wiped, session ends.
3. Reset mid-session → wiped + listening resumes in the same mode (toggle re-armed / PTT continues while held).
4. Rapid repeated clicks → no ghost session, no residual text, `is_session_active` returns to False.
5. Click-through unaffected outside the two buttons.

- [ ] **Step 4: Commit**

```bash
git add twistt.py
git commit -m "Reset re-triggers the same mode after wiping; coherent OSD notifications"
```

---

## Task 8: Documentation

**Files:**
- Modify: `CHANGELOG.md` (new dated entry), `README.md` (if user-facing behavior warrants a note — no new args/env, but a new interaction).

- [ ] **Step 1: CHANGELOG entry**

Add under a `## 2026-06-28` section (the OSD click-through entry from earlier today already exists there — append to the same `### Added`):

```markdown
### Added

- Cancel (✕) and Reset (↻) buttons in the OSD overlay (bottom-right). Cancel wipes
  the in-progress transcription (including text already pasted) and stops the active
  mode; Reset does the same and immediately resumes in the same mode (toggle or
  push-to-talk). The rest of the overlay stays click-through.
```

- [ ] **Step 2: README note (if relevant)**

If the README documents the OSD, add a short line describing the two buttons. No new env vars / CLI args are introduced.

- [ ] **Step 3: Manual verification**

Re-read CHANGELOG/README for accuracy.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md README.md
git commit -m "Document OSD cancel/reset buttons"
```

---

## Risks recap (from spec §6 — verify during Tasks 5–6)

1. **Reset ordering vs in-flight output.** `_buffer_commands` is a PriorityQueue; the `Reset` seq (2_500_000_000) sorts after segments and the indicator. The `is_aborting` skip in `BufferTask` drops still-queued inserts so they don't re-paste after the wipe. `Manager.text` is the source of truth for what to backspace. Watch for a brief paste-then-erase flash (acceptable for v1).
2. **Deltas in flight.** The `is_aborting` early-returns in delta handlers must be in place before flags are cleared, so a late WS event can't re-insert text.
3. **Toggle cooldown / session-active guard bypass for Reset re-trigger.** `abort_session` must bring `is_session_active` to False before the internal re-trigger.
4. **Socket send robustness** (OSD side) and **reader reconnect** (main side): reader reads `self._socket` fresh each loop.
