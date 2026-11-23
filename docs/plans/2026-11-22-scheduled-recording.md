# Scheduled recording — plan (2026-11-22)

Locked design (John, Nov 22 2026 — not relitigated here): one-off
schedules; auto-start at the set time; only while the app process is
running (no background service in v1 — closing the app entirely
means schedules do not fire; minimized-to-tray counts as running);
stop only on manual Stop (the auto-detect stop gates must NEVER stop
a scheduled recording, exactly as manual recordings are never
stopped by the watcher); skip when already recording; created via
new MCP write tools AND by hand in the GUI; MCP write tools are an
explicit, scheduling-scoped exception to the read-only MCP rule;
consent note everywhere (scheduled auto-start does not trigger
Teams' recording banner).

## Decisions

- **D1 — Shared store file.** The MCP server is a separate stdio
  process, so schedules live in `~/.offair/schedules.json`, written
  atomically (temp file + `os.replace` in the same directory).
  The app reloads on mtime change; MCP cancel is picked up live.
  Writes merge against a fresh re-read (by id) so an app-side
  mark-fired cannot clobber an MCP-side create that landed
  between the app's load and its write. Last-writer-wins per
  record only. No cross-process lock (no new dependency for it).
- **D2 — Store module first, pure, no Qt.** `app/storage/schedules.py`
  owns the record shape, validation, atomic I/O, corrupt-file
  behaviour (log + treat as empty, never raise into the app or
  the MCP server, never silently discard valid schedules —
  malformed *entries* are skipped, valid ones kept), expiry and
  due detection. Both the GUI and MCP go through it; neither
  hand-rolls JSON.
- **D3 — Fire through the one Record path.** `MainWindow.on_record`
  gains a `schedule=` keyword. A scheduled start reuses the exact
  manual start (same `create_session`, same `DualCapture`, same
  live transcription, same Stop/finalize), with: the schedule's
  target app (not the picker's), the no-audio-session dialog
  skipped (a meeting app that launches a minute late must still
  be captured — `capture_system_mode="auto"` falls back to
  loopback with the warning recorded, as manual "record anyway"
  does), dictation never, `_auto_recording` stays False (so the
  watcher can never stop it), and the session metadata tagged
  under a `scheduled` key (schedule id, subject, attendees,
  organiser). The `calendar_event` key is NOT touched — Outlook
  linking owns it. Pre-roll: whatever the manual path already
  does (feeder drain if enabled); no schedule-specific arming.
- **D4 — Scheduler timing.** One 2 s `QTimer` in the main window
  (`_schedule_timer`): reload-if-mtime-changed, refresh the GUI
  list when anything changed, expire-then-fire. A due schedule
  fires when `fire_at <= now <= fire_at + 300 s` (covers
  sleep/wake and timer stalls without recording hours late);
  older due schedules expire instead. At startup, ALL pending
  schedules with `fire_at <= now` expire before the timer
  starts — a schedule whose time passed while the app was closed
  never fires late (locked requirement).
- **D5 — Skip, don't double-start.** If a capture is RECORDING or
  PAUSED, or a stop/finalize is in flight, a due schedule is
  marked `skipped` (with a `note`) and logged. A failed start
  (capture not RECORDING after `on_record`) is also marked
  `skipped` with note "recording failed to start" — a pending
  schedule must never re-fire into a dialog loop.
- **D6 — GUI placement.** A compact "Scheduled recordings" card
  UNDER the Record card in the Record view's left column (the
  Record card itself is untouched — the 1.0.4 simplification
  stands). The card: pending + recent-terminal list (terminal
  rows show their status, no cancel), Cancel-selected button
  (pending only), add row (date/time edit + app combo + subject
  field + Add), consent hint. The geometry test at 1280x800 in
  both themes is the arbiter; the Record card has ~400 px of
  slack in the column, so the card fits.
- **D7 — App combo values.** The add form's app combo mirrors the
  picker's rows (live audio sessions + configured default +
  "Other app…" manual entry via the same dialog). Target stored
  as an exe name; at fire time the display name is derived with
  `display_name_for_exe`.
- **D8 — MCP surface.** Three tools on `OffAirMCPTools`
  (`schedules_path` ctor param, default `~/.offair/schedules.json`):
  `schedule_recording` (validates: fire time parses, is in the
  future, target non-empty; naive datetimes read as local wall
  time; known display names map to exe names via the known-app
  map; returns the schedule id), `list_schedules` (pending
  earliest-first, then terminal newest-first), `cancel_schedule`
  (pending only, else ValueError). The tools touch ONLY the store
  file — asserted in tests. `list_schedules` is `read_only_hint:
  true`; the two mutating tools are `read_only_hint: false`, and
  the exception is flagged loudly in `docs/MCP.md`, the MCP
  setup page, and the server instructions (which stop claiming
  the server is purely read-only).
- **D9 — History display.** The store keeps terminal records
  indefinitely (tiny file). The GUI shows pending + terminal
  records from the last 7 days. No pruning UI in v1.
- **D10 — Time storage.** `fire_at` is ISO-8601 with offset,
  stored exactly as the user/agent gave it (naive input is
  localised first). All comparisons in aware datetimes.
  Display is local time ("Today 14:30", "Tomorrow 09:00", else
  "Mon 12 Jan 09:00").

## Files

- NEW `app/storage/schedules.py` — store module (D2).
- NEW `tests/test_schedules.py` — store unit tests.
- NEW `tests/test_ui_schedules.py` — Qt GUI + firing tests.
- NEW `docs/SCHEDULED_RECORDING.md` — user doc + agent workflow.
- NEW `docs/plans/2026-11-22-scheduled-recording.md` — this file.
- EDIT `app/ui/main_window.py` — timer, `on_record(schedule=)`,
  startup expiry, schedule widgets/state, status text.
- EDIT `app/ui/views/record_view.py` — schedules card layout.
- EDIT `app/mcp/tools.py` — three tools + ctor param + docstring.
- EDIT `app/mcp/server.py` — register tools, instructions text.
- EDIT `app/ui/mcp_setup.py` — intro/privacy text + 3 tool lines.
- EDIT `app/ui/help_view.py` — register the new doc section.
- EDIT `docs/MCP.md` — read-only section split + scheduling section.
- EDIT `README.md` — "nine tools" → twelve everywhere; feature
  coverage for scheduled recording; `--mcp` run text.
- EDIT `tests/test_mcp.py` — hermetic tool tests (temp store).
- EDIT `tests/test_ui_mcp_setup.py` if it pins the tool list.
- EDIT `tests/test_ui_help.py` if it pins the section list.

## Verification

- Phase2 venv full suite + Qt venv full suite (2 deselects: the
  known flaky native-crash tests in `tests/test_ui_phase5.py`
  only). Clear `/tmp/pytest-of-*` first; reinstall libEGL if Qt
  imports fail. No modal dialog in the firing path (a modal
  dialog in a record path hangs offscreen runs).
- Qt fire test uses the real 2 s timer path with a near-future
  schedule and the `_FakeDualCapture` stand-in pattern from
  `tests/test_auto_detect.py` — not a mocked fire call only.
- Screenshots re-rendered through the real pipeline (seed →
  prep → rename → shots.py); at minimum the Record-view shots
  (`main-window.png`, `main-window-dark.png`,
  `recording-in-progress.png`); check the rest for shifts.
  Local files only — no upload (parent handles that).

## Could not verify (needs John's PC — never claim)

Real Teams call fired by a schedule; real Outlook-driven agent
scheduling end to end; tray-minimized firing on real Windows;
the frozen MCP smoke covering the three new tools (runs at
release-build time, not here).

## Build notes — what actually landed (Nov 22 2026)

Decisions D1–D10 held. Deviations from the step list above,
all minor, all for stated reasons:

* The Record view turned out to be a single `build_record_view`
  function (no separate `build_record_card` builder) — the
  schedules card is a new `build_schedules_card(win)` builder in
  the same file, placed under the Record card inside a new left
  column layout. The Record card itself is untouched.
* The schedule list is capped at 84 px (not 132): the left
  column must still fit the 1280x800 geometry audit, and 84 px
  shows ~3 rows. The consent hint carries the full wording.
* The schedule form's app combo is a second `_AppPickerCombo`
  (same rows as the main picker, same "Other app…" escape
  hatch), with its own `_manual_schedule_app` / last-exe
  tracking — the main picker's state is never shared.
* `on_record` gained `schedule=None` (not a separate start
  function): target from the schedule, dialog skipped, dictation
  excluded, `_auto_recording` left False, metadata tagged under
  a `scheduled` key, `_scheduled_recording_id` set/cleared in
  the start / failure / stop paths. A scheduled status line was
  added beside the auto/manual ones; the AUTO badge stays hidden
  for scheduled recordings (it implies watcher control).
* `create_main_window` gained the `schedules_path` test seam,
  matching `MainWindow.__init__`.
* `tests/test_ui_help.py` needed no edit: it enforces
  uniqueness + file existence, and the new doc file exists.
* `app/mcp/__init__.py`, `app/mcp/__main__.py`, and
  `main.py --mcp` help text were also reworded ("nine tools" →
  twelve; "read-only MCP server" → MCP server with the
  scheduling exception) — the same sweep as the README.
* The `test_ui_schedules.py` stop assertion follows the
  existing pattern: a successful stop keeps the STOPPED capture
  object (only a failed stop discards it) — the test waits for
  `_stopping` False + the "Saved:" label, not `capture is None`.
* One real bug caught during the build: an f-string format
  spec split across two implicit-concatenation literals in
  `_refresh_schedule_list` (a `SyntaxError` at import) — fixed
  by keeping the whole format spec in one literal.
* Geometry (Nov 22 2026): the first schedules card (list 84 px,
  one row per field) pushed the left column ~98 px past the
  1280x800 view and Qt compressed the Record card's own
  contents (clipped mode label, overlapping transport buttons)
  — the same "Qt violates minimums when the total exceeds the
  space" lesson as the Settings page. The card was reworked to
  two shared form rows, Cancel in the title row, list capped at
  48 px, 8 px vertical margins: the column fits with headroom
  in both themes. The recording-state geometry test needed a
  second event-loop pump before its stopped-state audit: the
  two-card column settles on the second turn, and one pump
  audits a transitional layout (same rule the main geometry
  test already documents).
* Code review (ce-code-review, focused local pass) caught two
  more before sign-off, both fixed with regression tests in
  `tests/test_ui_schedules.py`:
  - A scheduled start with the configured mode `per-app` would
    have recorded silence (or nothing) when the target had no
    live session yet — the brief requires "auto" semantics.
    `on_record` now forces the local `capture_mode` to `"auto"`
    for schedule starts (persisted setting untouched).
  - A failed scheduled start showed the modal "Recording
    failed" dialog — an automatic, possibly unattended firing
    must never block on a dialog. The failure path now logs +
    status-lines for schedule starts only (manual/auto starts
    keep the dialog), and `_fire_schedule` marks the schedule
    skipped.

