# Plan — OffAir 1.0.4 follow-up: Stop freeze + UI simplification

Date: 2026-11-20. Parent task: John's 1.0.3 report (Nov 20, work PC).
Repo: `~/workspace/offair`, main @ `cfaddc36`. Do NOT push — Larry pushes.

## John's report (verbatim substance)

1. The app **froze when he stopped the recording**.
2. The Record page's settings panel is too complicated — **microphone, speakers,
   record mode, and source should be hidden settings**; only the app picker and
   action buttons visible on the main Record view.
3. The button to the right of the app picker is unexplained (the `↻` refresh
   button).
4. The **Settings page UI is badly formatted**.
5. **Library and Calendar are one page — split them.**
6. Calendar needs **work-week and day views** in addition to month.
7. **"More" vs Settings is unclear** — what is the difference?

## Decisions (settled in planning, from evidence in the tree)

- **D1 — Nav becomes a flat list, no section headers.** `VIEW_NAMES` becomes
  `("Record", "Library", "Calendar", "Session", "People", "Settings",
  "MCP setup", "Help")` — 8 selectable entries, daily work first, setup/help
  last. The "More" header disappears entirely: there is no longer anything
  whose purpose needs explaining. `_on_nav_row_changed` / `_on_view_changed`
  lose the "More" offset handling (MCP setup is no longer shifted a row).
- **D2 — The `↻` button is deleted, not relabelled.** `_AppPickerCombo`
  already refreshes its list on popup (`showPopup` → `refresh_callback` =
  `_refresh_app_picker`, `main_window.py:203-221`). The button is redundant.
  Delete `win.refresh_apps_btn`, its wiring, and its tests. Add/keep a test
  that opening the picker refreshes the list.
- **D3 — Device pickers move as live widgets, not as saved settings.**
  `mic_combo` / `loopback_combo` are MainWindow-owned widgets populated by
  `refresh_devices()` and read at record time (`on_record`, ~line 2134-2147);
  device choice is NOT persisted today (config has
  `microphone_device_index` / `loopback_device_index` fields but nothing in
  the window reads them). So: move the two combos + the Refresh-devices
  button into a "Recording devices" section of the Settings view, OUTSIDE the
  dialog-form scroll area (so `_refresh_settings_view` rebuilds never touch
  them). `refresh_devices()` and `on_record` are unchanged. Do NOT add
  device persistence — that is scope creep; behaviour stays exactly as today.
- **D4 — Source becomes a persisted default.** The Record-view `source_combo`
  ("Auto (detect from app)" + explicit pick) is read once at record time
  (`override=self.source_combo.currentData()`, ~line 2123). Add a
  `default_source: str | None = None` field to `AppConfig` (persisted in
  settings.json — extend the field allowlist/serialization the same way the
  other fields are handled), add the field to the Settings dialog, and have
  `on_record` read `config.default_source` instead of the combo. The Session
  view's per-session `session_source_combo` (~line 1158, correct a session's
  source after the fact) is untouched. Auto-detection at record time is
  unchanged: `None`/Auto still resolves from the picked app.
- **D5 — Capture mode has ONE editor after this round.** The Record-view
  `capture_mode_combo` (2 options) is deleted. The Settings dialog's
  `capture_mode_combo` (3 options incl. `per-app`) is the only editor.
  `on_record` already reads `config.capture_system_mode` (~line 2100) — no
  change there. `_on_capture_mode_changed` (writes the Record combo's choice
  into config) and the combo-sync half of `_update_capture_mode_ui`
  (~lines 1717-1738) go away; the `mode_label` explanation half stays (it is
  feedback, not a setting: it names the picked app and what will be
  recorded).
- **D6 — Calendar modes share one widget, one data path.** `CalendarWidget`
  (`app/ui/calendar_view.py`) gains a mode switcher: Month (current grid),
  Work week (Mon–Fri), Day. All three reuse `collect_calendar_entries` /
  `group_sessions_by_day` / `badge_text` / `_BadgeButton` and the
  `session_requested` signal. State: one anchor date; Month derives
  (year, month) from it, Week shows the Mon–Fri week containing it, Day
  shows it. Prev/Next step one month / one week / one day; Today jumps to
  the current date in the current mode. Keep the existing public API stable
  (`session_requested`, `refresh()`, `set_output_root()`, `set_month`,
  `visible_month`) and add the mode API beside it.

- **D7 — One Whisper model everywhere: the configured one.**
  `NEARLIVE_DEFAULT_MODEL` stays as a code fallback only (the
  `ChunkedTranscriber` default parameter, tests) and is never the
  user-facing default. Every user-facing transcription path uses
  `_configured_model()` (`model_combo.currentData() or
  config.whisper_model` — base on a fresh install; a setting saved
  under 1.0.2 carries over, so an upgrader may still be on small,
  and that is their choice, honoured everywhere). No user-visible
  string may name a model other than the one actually in use.

## Item 7 — Unify the Whisper model (manual near-live was still tiny)

Verified in the code Nov 20 (not from the report alone):

- The live path already uses the configured model:
  `_start_live_transcription` builds its `ChunkedTranscriber` with
  `model=self._configured_model()` (`main_window.py:4039`), and
  `_configured_model` (line 4018) already documents itself as "The
  Whisper model the live path and the manual path share" — but the
  manual path does not share it today.
- The manual path is hardcoded to tiny: `on_nearlive` (~line 4445)
  constructs `_NearLiveWorker(self.session_dir,
  NEARLIVE_DEFAULT_MODEL)` (line 4459) and its status + finished
  strings interpolate `NEARLIVE_DEFAULT_MODEL` (lines 4455, 4487).
  `NEARLIVE_DEFAULT_MODEL = "tiny"` is defined twice
  (`app/config.py:191`, `app/nearlive/chunker.py:51`); the window
  imports it from `app.config` (line 50).
- User-visible hardcoded "tiny": the live hint (`_NEARLIVE_HINT`,
  ~line 753: "re-runs chunked transcription on a finished session
  with Whisper tiny"), the live list tooltip (~line 1106: "Near-live
  transcript: 25 s chunks, Whisper tiny, provisional speaker
  labels"), and the `on_nearlive` status/finished strings. (The
  `_NearLiveWorker` docstring ~line 566 is stale the same way, but
  is code-only.)
- Stale rationale: `app/config.py:186-188` ("tiny is the near-live
  default: it must keep up with real time … small is ~5x slower"),
  `app/nearlive/chunker.py:27`, `app/nearlive/__init__.py:14`
  ("Whisper ``tiny`` (or ``base`` at most)").

Fix (files: `app/ui/main_window.py`, `app/ui/views/record_view.py`,
`app/config.py`, `app/nearlive/chunker.py`, `app/nearlive/__init__.py`,
tests, docs):

- `on_nearlive` resolves the model ONCE via `self._configured_model()`
  and passes that to the worker and to both strings, so the label can
  never disagree with the worker.
- The live hint and the live-list tooltip name the model actually in
  use (resolve at show time, not a hardcoded name), or drop the model
  name where the active-model label (below) already states it.
- Show the active model in the Record view's live-transcript area —
  e.g. a "Transcribing with Whisper base" line under the "Live
  transcript" title — so the user sees at a glance which model is
  transcribing. It must update when the configured model changes
  (Settings save / model combo change), not only at startup.
- Update the stale rationale comment in `app/config.py` (base keeps up
  too: RTF ~0.135, a 25 s chunk transcribes in ~3.4 s, comfortably
  real-time, with small-level accuracy) and the matching docstrings in
  `app/nearlive/chunker.py` and `app/nearlive/__init__.py`. Keep
  `NEARLIVE_DEFAULT_MODEL` as the code fallback only (D7).
- Tests: manual near-live uses the configured model (set config to
  base, run `on_nearlive` with a fake engine, assert the worker got
  base; repeat with small to prove it follows the setting, not a
  constant); labels/tooltip/active-model line name the model in use.
- Docs: README/docs anywhere near-live is described as tiny
  (`docs/TRANSCRIPTION.md`, `docs/CPU_TIMES.md`, `docs/OVERVIEW.md`,
  `docs/archive/.../PHASE4.md` if it still makes the claim) updated to
  the configured model.

Execution note: implement with Item 2 — both touch the Record view's
live-transcript column and `main_window.py`'s near-live strings.

## Item 1 — Stop freeze (TOP PRIORITY)

Status: INVESTIGATED Nov 20 (report below). Implementation follows this
section, not the earlier placeholder.

### What the investigator reproduced (harness `/tmp/stop_freeze_harness.py`,
gui-shots venv, offscreen Qt; real `MainWindow.on_stop` + real `DualCapture`
with fast silence chunk-sources + fake engines; 10-min simulated meeting
at 48 kHz; responsiveness via a 50 ms QTimer probe + wall-timing of
`on_stop()`)

**Cause #1 — synchronous `mix_files` inside `on_stop` (proven hard
UI-thread block, best-supported root cause).** `on_stop`
(`main_window.py:2274`) runs synchronously on the UI thread and calls
`self.capture.stop()` (line 2296) → `DualCapture.stop()`
(`app/audio/capture.py:837`) → `mix_files(mic, system, combined)`
(`capture.py:861`) → `app/audio/mixer.py:125` / `:55`, which loads both
full tracks and mixes in float64 (`np.asarray(..., dtype=np.float64)`
per track plus padded copies). Measured: 1 min 0.21 s; 5 min 0.80–1.77 s;
10 min **1.29–4.09 s**, peak RSS ~1.1 GB. 30-min extrapolation: **~4–12 s
wall, ~3.3 GB transient RSS**. On a work laptop already running
Teams/Outlook, paging can stretch that well beyond the measured range.
The capture-thread joins ride in the same synchronous block:
`WASAPICapture.stop()` (`capture.py:437`) does `self._thread.join(
timeout=10)` (`:450`) — **up to 10 s per track, 20 s total** — plus
`ProcessLoopbackCapture.stop()` (`perapp_capture.py:964`, COM
`IAudioClient.Stop()` + `close()`/`CoUninitialize()`) on the UI thread.

**Cause #2 — the 60 s abandon wait is a UX dead end (reproduced
exactly).** During `_stop_finalize_pending` the UI thread is alive, but
`on_cancel_transcribe` (`:2411`) only handles `_finalizing` /
`_transcribing` and `_update_buttons` (`:1996`) enables Cancel only when
those are true — so Cancel is DISABLED, the label is a static "Saved.
Finishing the live transcript…", and Record is RE-ENABLED (capture state
is STOPPED), so a second Record can start while the old finalize is still
pending. To the user this is indistinguishable from a freeze. Most
plausible stuck-live-worker trigger (hypothesis, not reproduced): a
first-run base-model download inside the live chunk, or a slow chunk on a
loaded CPU. Measured with a stuck live worker: `on_stop()` wall 1.73 s,
UI probe kept firing, abandon fired at 5.0 s (harness value; shipped
default 60 s), Stop→transcript 9.87 s total.

**Cause #3 — the live-poll snapshot is synchronous on the UI thread
every 5 s.** `_on_live_poll` (`:4063`) does `shutil.copyfile` of both
growing tracks (`:4104`) + `mix_files` (`:4106`) synchronously before
handing the snapshot to the worker. Measured one tick at 10 min:
**0.73–1.87 s** on the UI thread, growing with meeting length. A tick in
flight when Stop is pressed also delays the Stop handler itself.

**Ruled out / not claimed.** Long tail (live died early, tail ≈ 575 s)
runs in a worker with per-chunk progress and a working Cancel — it looks
busy, not frozen. The `gc.collect()` hypothesis for finalize-phase gaps
is REFUTED (focused probe: `gc.collect()` 0.01 s, max UI gap 0.06 s).
No `.wait()`/`.join()` on any QThread exists in `main_window.py` — the
earlier "synchronous tail" fix held. NOT checked: real Windows audio
teardown (WASAPI/COM, `join(timeout=10)` ×2 worst case — unmeasured on
real hardware, could add ~20 s+), real faster-whisper base-model
download, Outlook `link_session` COM latency on a work PC, John's actual
meeting length and log.

### Fix design (implement all five)

1. **Move capture stop + mix off the UI thread.** On Stop: immediately
   show "Stopping…" and disable Stop/Record; stop capture threads and
   run `mix_files` in a worker; only then start `_begin_stop_finalize`.
   In parallel, rewrite `mix_files`/`mix_arrays` to stream in blocks
   (int16/float32, no whole-file float64) — cuts wall time ~3–5× and RSS
   ~8× even before threading.
2. **Make the abandon wait cancellable and visible.** Track a
   `_waiting_for_live` flag; enable Cancel during `_stop_finalize_pending`
   and have `on_cancel_transcribe` abandon immediately (set
   `_live_abandoned`, stop the timeout timer, proceed to
   `_finalize_transcription` from the saved live transcript). Show a live
   countdown/progress ("Waiting for live chunk… 12s / 60s — Cancel to
   finalize now"). Keep the 60 s cap, but never leave the user with no
   way out.
3. **Disable Record until Stop fully settles** (`_stop_finalize_pending`
   or `_finalizing` → record disabled in `_update_buttons`).
4. **Move the live-poll snapshot off the UI thread** (or make it
   incremental: copy only bytes since the last tick, mix only the new
   window). It freezes the UI 1–2 s per tick late in a meeting and can
   delay Stop handling.
5. **Cheap always-on signal:** log wall time of `on_stop()`'s synchronous
   section and of `mix_files` at Stop, so the next freeze report carries a
   number instead of a feeling.

### Regression tests (extend `tests/test_live_transcription.py`, Qt-gated, fake engines)

- Stop with stuck live worker: fake engine blocking on first chunk,
  short `_live_finalize_wait_seconds`; assert `on_stop()` returns
  promptly (< ~3 s wall with 10-min audio), the 50 ms probe fires at
  least every 200 ms during the wait, Cancel is enabled during the
  wait, and pressing Cancel abandons immediately with finalize
  proceeding. (Existing
  `test_stop_finalize_timeout_abandons_stuck_live_worker` covers the
  abandon, not the responsiveness/Cancel state.)
- Stop wall-time bound: 10-min (ideally 30-min) meeting via real
  `DualCapture` + chunk sources; assert `on_stop()` wall < threshold
  (fails today once mix is counted at 30 min).
- Live-poll snapshot cost: assert one `_on_live_poll()` call on long
  tracks completes below threshold (fails today at ~1–2 s for 10 min).
- Record disabled during finalize: after `on_stop` with pending
  finalize, `record_btn` disabled until finalize finishes/cancels.

## Item 2 — Simplify the Record view

Files: `app/ui/views/record_view.py`, `app/ui/main_window.py`
(creation ~lines 820-900, `_update_capture_mode_ui`, `_on_capture_mode_changed`,
`on_record` source read), `app/config.py` (D4 field),
`app/ui/settings_dialog.py` (D4 field), `app/ui/views/settings_view.py`
(D3 section), tests.

Record view keeps ONLY: the app picker row (`process_combo`, no refresh
button), the `mode_label` callout, mic/system level meters, Record / Pause /
Resume / Stop, the status/session labels, and the live transcript column
(list + manual near-live button). Everything else leaves the card:

- `mic_combo`, `loopback_combo`, `refresh_btn` (refresh devices) → the
  Settings view's "Recording devices" section (D3).
- `capture_mode_combo` → deleted (D5); the dialog's combo is the editor.
- `source_combo` → deleted; replaced by the persisted default source (D4).
- `browse_btn` (choose output folder) → deleted from Record; the Settings
  form already has "Choose output folder…".
- `refresh_apps_btn` → deleted (D2).

Invariants: defaults still apply (no devices picked = system defaults);
auto-detect still works; source auto-detection still runs; the app picker
still refreshes on popup; `on_record` behaviour for a user who never opens
Settings is identical to today. Update every test that references the moved
or deleted widgets; add a test that Record shows no device/mode/source
controls.

## Item 3 — Settings page redesign

Files: `app/ui/settings_dialog.py` (the form — the view embeds it),
`app/ui/views/settings_view.py`, `app/ui/main_window.py`
(`_refresh_settings_view`, `_on_settings_save`, `_apply_saved_config`),
tests.

Restructure the flat form (today: `_field_label` + widget + hint repeated,
one `QGroupBox` for Sherpa) into labelled `QGroupBox` sections with uniform
margins/spacing, one consistent field pattern (label above the control,
one-line hint below only where the label does not explain itself), and no
clipped/overlapping controls at 1280x800 in either theme. Proposed sections
(adjust only if the fields do not fit — NO field may be dropped, renamed in
meaning, or change behaviour):

- **Recording** — output folder, system-audio capture mode, default
  recording source (D4), auto-process.
- **Transcription** — model, near-live enabled + chunk length.
- **Speakers & diarization** — diarization mode, Sherpa group (model folder,
  expected speakers, threshold), embedding backend, HF-token checkbox.
- **Automation** — auto-detect (with consent flow unchanged), Outlook link.
- **Storage & search** — audio retention, search mode.
- **Local AI** — Ollama enabled / model / URL, AI polish.
- **Names** — names dictionary.
- **Appearance** — theme.

The "Recording devices" section (D3, live combos + refresh) sits in the
Settings view OUTSIDE the dialog-form scroll area, above or below it, with
its own header — it is not part of `SettingsDialog` and must survive form
rebuilds. The dialog stays usable standalone (its tests use it directly);
`config_from_dialog` / `apply_and_save` gain only the D4 field. Keep Save /
Discard; keep the dirty-form guard (`_settings_form_dirty`) — navigating
away and back must not discard unsaved edits (regression from the last
round). Verify geometry at 1280x800 in both themes; extend the geometry
tests if the layout changes shape.

## Item 4 — Split Library and Calendar

Files: `app/ui/views/library_view.py`, new `app/ui/views/calendar_page.py`
(`build_calendar_view(win)` — do NOT name it `calendar_view.py`, that name
is taken by the widget module), `app/ui/views/__init__.py`,
`app/ui/main_window.py` (nav + `_on_view_changed`), tests, docs.

- Library view: search row, session list, Open button, status label. The
  `CalendarWidget` leaves this page.
- Calendar page: the `CalendarWidget` full-page (`win.calendar_widget`
  attribute stays — tests reference it), same badge content and
  click-to-open behaviour (`session_requested` → `_on_calendar_open_session`
  → `load_session`, Session view still loads).
- `_on_view_changed`: showing Calendar refreshes the grid (the refresh that
  Library used to do); showing Library refreshes only the list.
- Nav order per D1: Record, Library, Calendar, Session, People, Settings,
  MCP setup, Help.

## Item 5 — Calendar week + day views

Files: `app/ui/calendar_view.py` (widget + Qt-free helpers), tests
(`tests/test_ui_calendar.py`), `docs/CALENDAR_VIEW.md`.

- Mode switcher on the Calendar page (segmented buttons or combo — pick the
  one that fits the header row without crowding the prev/next/Today
  controls): **Month | Work week | Day**.
- Work week: Mon–Fri columns for the week containing the anchor date.
  NOT a 7-day week. Day headers show the date; each column lists that
  day's badges.
- Day: one column for the anchor date, badges in time order, with the date
  as the header.
- Badge content identical in all modes (`badge_text`: time · app · source
  [— subject]); click-to-open identical.
- Qt-free helpers stay Qt-free: add pure functions for week computation
  (Monday of the week containing a date; the Mon–Fri days of a week) and
  unit-test them without PyQt6. Widget tests cover mode switching, prev/next
  stepping per mode, Today, and badge presence per mode.
- Docs: `docs/CALENDAR_VIEW.md` describes all three modes.

## Item 6 — Nav clarity (More vs Settings)

Covered by D1: the "More" bucket is deleted, not explained. Every nav entry
is a view with a self-explanatory name. Consequences to implement:

- `VIEW_NAMES` per D1; the nav-list construction loop drops the "More"
  header insertion; `nav_list` rows map 1:1 to views.
- `more_view.py` (`build_mcp_view` / `build_help_view`) stays as the builders
  for the MCP setup and Help pages — only the grouping goes away. (Rename
  the module only if the name becomes actively misleading; do not churn
  imports for cosmetics.)
- Update the tests that pin the old nav: `tests/test_ui_phase6.py`
  (~lines 309-317 tab order, ~525-543 "More" header + row shift),
  `tests/test_ui_calendar.py:289`, `tests/test_ui_mcp_setup.py:290`,
  `tests/test_ui_help.py:229`, `tests/test_ui_visual.py:402` (iterates all
  views — must now include Calendar). Assert the new order and that every
  nav row is selectable and maps to exactly one view.

## Item 8 — Model default must not carry over between versions

The problem (John, Nov 20): `settings.json` persists `whisper_model`, so
a setting saved under 1.0.2 (`"small"`, the old default) is kept on
upgrade to 1.0.3 and the user never gets the new base default. The
current version's default must win on upgrade — unless the user
explicitly picked a model. Scope is the model setting only; devices,
folders, and other settings keep carrying over as today.

### Current mechanism (verified in the code Nov 20)

- `AppConfig.whisper_model: str = WHISPER_DEFAULT_MODEL` (`app/config.py`,
  `"base"` now). Persisted via `_SETTINGS_KEYS` + `settings_to_dict` +
  `save_settings`; loaded via `config_from_settings` (per-key validation,
  invalid value falls back to the default for that key). There is no
  settings-version field and no explicit-choice tracking.
- TWO model combos exist. The MainWindow session combo
  (`main_window.py:960`, Session view, Process section) is a per-session
  transcribe picker: `on_transcribe` (`:2587`) and `_configured_model`
  (`:4250`) read `model_combo.currentData() or
  config.whisper_model`; `_apply_saved_config` (`:3325`) syncs it from
  `new_config.whisper_model` after a Settings save. The Settings dialog
  combo (`settings_dialog.py:243`) is the persistent editor;
  `config_from_dialog` (`:591`) reads it. The session combo never
  persists on its own — it re-initialises from config at startup and
  after every Settings save.

### Design (D8 — one rule, one helper, one flag)

- **New field:** `whisper_model_explicit: bool = False` on `AppConfig`,
  persisted in `settings.json` as a new allowlisted key (bool, default
  `False`). It is set `True` ONLY by a user-driven change of the Settings
  dialog's model combo — never by a plain Settings save, by loading, by
  syncing combos from config, or by programmatic `setCurrentText` during
  construction/refresh.
- **The discriminator is the `activated` signal, not
  `currentIndexChanged`.** `QComboBox.activated` fires only when the
  user picks an item (mouse or keyboard); programmatic
  `setCurrentIndex`/`setCurrentText` do not emit it. The dialog
  connects `activated` to a handler that sets a per-dialog
  `touched` flag. (The MainWindow session combo's `currentIndexChanged`
  is already wired to `_refresh_live_model_label`; the session combo
  does NOT set the persistent flag — it stays an ephemeral per-session
  override, as today. Only the Settings pick persists.)
- **Single source of truth:** one helper, `AppConfig.effective_whisper_model`
  (property) or module function `effective_whisper_model(config)` —
  implementer picks the shape, but every reader uses it: the dialog's
  combo initialisation, `config_from_dialog` (when untouched),
  `_configured_model`, `on_transcribe`'s fallback, `_apply_saved_config`'s
  combo sync, the window constructor's combo init, and the live-model
  label refresh. Rule: explicit flag `True` → stored `whisper_model`;
  otherwise → `WHISPER_DEFAULT_MODEL` (the current build's default).
- **Dialog save semantics:** the dialog initialises its combo from the
  EFFECTIVE model and its `touched` flag from the incoming explicit
  flag. On save, `config_from_dialog` returns
  `whisper_model` = combo value and `whisper_model_explicit` = incoming
  explicit OR touched-this-dialog. So an untouched save writes the
  effective model with explicit `False` (completing a migration), and a
  touched save writes the picked model with explicit `True`. The
  `_settings_form_dirty` snapshot comparison must include the new field
  (verify: if it compares config objects, dataclass equality covers it
  automatically — if it compares field-by-field, add the field).
- **Retroactive migration (pre-flag files):** in `config_from_settings`,
  when the `whisper_model_explicit` key is ABSENT, infer explicitness
  from the stored model with the rule:
  `explicit = stored is not None and stored != WHISPER_DEFAULT_MODEL
  and stored not in PAST_WHISPER_DEFAULTS`, where
  `PAST_WHISPER_DEFAULTS: tuple[str, ...] = ("small",)` is a named
  constant with a comment (the 1.0.2 default; the only past default
  that differs from the current one). Consequences, all intended:
  stored `"small"` → non-explicit → base; stored `"tiny"`/`"medium"` →
  explicit (no past default was ever tiny/medium — they could only
  have been picked); stored `"base"` or missing/invalid → non-explicit,
  effective model base either way. Going forward the flag is always
  written, so the migration triggers only for pre-flag files.
- **Honest caveat (state it in the code comment and the docs):** an
  explicit `"small"` pick made under 1.0.2/1.0.3 is indistinguishable
  from the old default and resets to base once on upgrade. That is a
  one-time migration cost; re-picking small in Settings sets the flag
  and persists forever after.
- **Settings UI:** the dialog's Transcription section marks the current
  default in the combo (e.g. the base item reads `base — … (default)`)
  and shows a hint line under the combo stating the state, updated live
  on `activated` (and initialised from the incoming config): default
  state — "Using the default model (base). Pick a model to keep it
  across updates."; explicit state — "You picked small — this choice
  is kept across updates." (Wording may be adjusted, but the two states
  must be distinguishable at a glance.) The session combo area keeps
  the Item 7 live-model label as-is; the default-vs-explicit
  distinction lives in Settings, per the requirement.

### Tests (Qt-free where possible, Qt-gated for the dialog)

- Fresh install (no settings file) → effective model base.
- Upgrade: settings file with `whisper_model: "small"` and NO explicit
  key → effective model base after load.
- Upgrade: settings file with `whisper_model: "tiny"` and NO explicit
  key → effective model tiny (explicit pick, kept).
- Explicit choice survives a simulated version change: save explicit
  `"small"`, monkeypatch `WHISPER_DEFAULT_MODEL` to another model,
  reload → still `"small"`; save non-explicit, change the default,
  reload → follows the new default.
- Dialog: untouched save does NOT set the flag; changing the combo
  (via `activated`, not programmatic set) sets it; the hint line shows
  the default vs explicit state correctly in both cases.
- Session combo change does not set the persistent flag.

### Docs

Update wherever settings persistence is described (grep for
`settings.json` / "persist" / "carried over" in README.md and docs/):
the model default follows the current build unless explicitly picked,
with the one-time small→base migration noted honestly.

### Execution note

Implement AFTER the review-fix pass lands (same files:
`app/config.py`, `app/ui/settings_dialog.py`, `app/ui/main_window.py`) —
a separate agent, so it cannot collide with the fix-up agent currently
working on S1–S6/N1–N5/N7.

## Verification (before reporting done)

- Full phase2 suite green, full Qt suite green (with the 2 known flaky
  deselects in `tests/test_ui_phase5.py`), `compileall` clean.
- README + docs accurate for every behaviour change in the SAME change:
  README §§ Record/Library/Calendar/Settings (nav, Record card contents,
  source default, calendar modes), `docs/CALENDAR_VIEW.md`,
  `docs/SCREENSHOTS.md`, `docs/OVERVIEW.md` where it describes the views.
- Re-render screenshots for every changed view (`screenshots/` +
  `docs/SCREENSHOTS.md`) via `~/workspace/meettrace-tryouts/gui-shots/`
  (`prep_demo_data.py` + `shots.py`; re-seed demo data first — /tmp gets
  cleaned; new Calendar view needs a shot; images for John live under
  `~/workspace`, never /tmp). Delete screenshots of removed UI (none
  expected to be removed outright — the Library shot changes shape).
- ce-code-review on the finished diff before reporting; ce-compound for
  any lesson a fix taught (esp. the Stop freeze: whatever the cause was,
  capture it in the plan file / docs so the next change cannot reintroduce
  it).

## Execution

Single implementation agent, items in priority order (1, then 2+3+7, then
4+5+6), because all of them touch `app/ui/main_window.py` and the view
modules — parallel builders would collide on the same files. Item 8 is
implemented after the review-fix pass lands, by a separate agent (same
files: `app/config.py`, `app/ui/settings_dialog.py`,
`app/ui/main_window.py`). The agent
verifies the suites after each item. No pushes (Larry pushes). Report back
per item: files changed, test results, how John verifies each item on his
PC (esp. the Stop freeze: exactly what to try), deviations and why, what
could NOT be verified (real Teams audio on Windows — never claim it).
Suggested next step is a 1.0.4 build — do not build it.
