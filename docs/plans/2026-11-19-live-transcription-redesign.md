# Plan: live transcription + base default + UI redesign (Nov 19, 2026)

Source of the work: John's Nov 19 report — 1.0.2 crashed a few minutes
into transcribing a 30-minute meeting, transcription takes forever /
never finishes, and the UI is "all ugly, not modular/focused, not
separation of tasks". His direction: chunk the recording and transcribe
WHILE recording, switch the default model to base, full UI redesign.

Status: plan written by the coordinator (ce-plan). Implementation by
subagents afterwards; the coordinator runs ce-code-review and ce-compound
before reporting done. No push by the coordinator — Larry pushes.

## Settled decisions (do not relitigate)

- Default Whisper model becomes **base** (was small). Small stays
  selectable. Evidence: measured Nov 8 on the 61.3 s fixture — base
  RTF 0.135 / WER 7.5%, small RTF 0.333 / WER 7.5%. Base matched
  small's accuracy at 2.5x less time.
- Live transcription uses the **configured Whisper model** (base after
  the switch), not tiny. Rationale: base RTF 0.135 means a 25 s chunk
  transcribes in ~3.4 s — comfortably real-time — and using the same
  model for live chunks and the final tail means the live and final
  transcripts agree. Tiny stays available as a manual choice.
- Chunk geometry stays **25 s windows, 0.5 s overlap**, reusing
  `ChunkedTranscriber` (`app/nearlive/chunker.py`) — the dedup and
  watermark logic already exist and are tested. Change only with a
  measured reason.
- Everything stays fully local, CPU-only, no network for core paths.
- No unrequested additions. README + docs update in the same change
  for every behaviour/default change (standing rule).
- Saved user settings keep working: a user who explicitly picked
  small keeps small; only the *default* changes.

## Item 1 — Default model small → base

Files: `app/config.py` (WHISPER_DEFAULT_MODEL, WHISPER_MODELS notes),
`app/ui/settings_dialog.py` (default display), `app/nearlive/chunker.py`
(NEARLIVE_DEFAULT_MODEL is "tiny" — leave it; live path uses the
configured model, see item 2), README.md, docs/CPU_TIMES.md,
docs/TRANSCRIPTION.md, docs/OVERVIEW.md (check each for "small" as
default), tests.

Work:
1. `WHISPER_DEFAULT_MODEL = "base"`. Update the comment block in
   `app/config.py` (the RTF table stays — it is the evidence) and the
   per-model notes: base becomes "default", small becomes "opt-in;
   same accuracy as base on clean audio, 2.5x slower".
2. Settings dialog: the model combo must show base selected for a
   fresh config; a saved "small" still loads as small.
3. The auto-process chain and the manual Transcribe path both read
   `config.whisper_model` already — verify no hard-coded "small"
   remains as a fallback in `main_window.py` (`currentData() or
   config.whisper_model` is fine) or in workers.
4. README: model table (base = default), the auto-process line
   ("transcribe (small)" → "(base)"), the first-run download note
   (~466 MB → ~142 MB), the CPU-times paragraph.
5. docs/CPU_TIMES.md: record the new default and the Nov 19 reason
   (John's crash/slowness report); keep the measured table.
6. Tests: assert `AppConfig().whisper_model == "base"`,
   `WHISPER_DEFAULT_MODEL == "base"`; assert a settings file that
   says `"whisper_model": "small"` still loads small (explicit choice
   preserved); update any test asserting small as the default
   (grep `tests/` for "small" — known files: test_transcription.py,
   test_settings.py, test_auto_process.py, test_ui_sherpa_default.py,
   test_ui_cleanup.py).

## Item 2 — Chunked recording + transcribe while recording

Goal: the transcript exists *during* the meeting, a crash loses at
most the in-flight chunk, and Stop is fast (only the tail is left).

### Design

- **Polling loop.** A QTimer running only while recording (start in
  `on_record`, stop in `on_stop`/`on_pause` handling — see pause
  below), firing every **5 s**. Each tick:
  1. **Snapshot the growing tracks — NOT `combined.wav`.**
     Correction (research dossier, Nov 19): `combined.wav` does
     NOT grow during recording; it is created only at stop by
     mixing (`capture.stop()` → `mix_files`). The tracks that grow
     incrementally are `mic.wav` / `system.wav` (incremental WAV
     writers, valid headers from the start). So each tick: copy
     `mic.wav` + `system.wav` to temp snapshots (capture is still
     writing the originals — never read the live files directly),
     mix the snapshots to a temp combined snapshot with the
     existing `mix_files` helper (`app/audio/mixer.py`), then call
     `ChunkedTranscriber.process_new_audio(combined_snapshot,
     mic_path=mic_snapshot, system_path=system_snapshot)` — the
     mic/system snapshots also give the provisional You/Remote
     labels via dual-channel energy, as in the post-stop path.
  2. Append new segments to the Record view's live transcript list
     and **save the progressive transcript to disk immediately**
     (see file format below).
- **Progressive save.** After every chunk, write
  `transcript-live.json` in the session folder: the same payload
  shape as `transcript.json` (model, segments with absolute
  timestamps, `chunks_done`, `watermark_seconds`) plus
  `"live": true`. It is the crash-recovery record. It is NOT a
  substitute for `transcript.json`: the retention rule (prune audio
  only when `transcript.json` exists) must keep requiring the final
  file, and the browser/semantic index must keep reading the final
  file.
- **Stop / finalize.** `on_stop` keeps its current behaviour
  (stop capture, finalize session, Outlook link) and then, instead
  of the old full-file transcribe:
  1. Run the chunker once more over the tail (audio after the
     watermark — a partial final window, transcribed by the same
     chunk path; `process_file` over the whole file is the fallback
     if no live state exists, e.g. live was disabled or the app
     restarted mid-recording). At stop the real `combined.wav`
     exists (created by `capture.stop()`), so the tail pass reads
     the final files, not snapshots.
  2. Merge live segments + tail segments (dedup by the chunker's
     existing `(start, text)` rule) into the canonical
     `transcript.json` / `.txt` / `.srt` via the existing writers.
  3. The existing auto-process chain then runs diarization +
     suggest/identify on the final transcript, unchanged.
  4. Delete `transcript-live.json` after a successful finalize
     (it has served its purpose; the final files are the record).
- **Manual Transcribe stays.** The Process/Session view's Transcribe
  button keeps the full-file `transcribe_session` path for old
  sessions with no live data. Add a **Cancel** button: the engine
  checks a cancel flag per segment (see below), so cancel is safe
  and fast on both paths.
- **Model download (first use).** The live worker must never
  silently hang: before constructing the Whisper model, check
  whether the model is cached; if not, show "Downloading Whisper
  base (~142 MB) — first run only" with elapsed time in the Record
  view status, and do the download on the worker thread (model
  construction does it today). The UI stays responsive; the first
  chunk simply waits. Do NOT attempt a byte-level progress bar
  (faster-whisper gives no download-progress hook) — say so in the
  code comment and the docs instead of faking one.
- **Worker errors.** A chunk failure surfaces loudly in the Record
  view status + log, stops live transcription, and NEVER loses the
  recording or the already-saved live transcript. The Stop/finalize
  path still runs (it falls back to the full-file transcribe).
- **Pause/resume.** Capture pause discards audio while paused
  (tracks stay aligned). The live poller must pause with it: stop
  the QTimer on pause, restart on resume. The chunker's watermark
  is file-duration based, so a paused stretch (no new audio) is
  naturally a no-op — but do not poll a file that is not growing.
- **Stop mid-chunk.** The in-flight chunk (audio after the last
  completed window) is transcribed by the finalize tail pass. The
  live transcript never shows a half-chunk.
- **Auto-detect recordings.** Auto-started recordings get live
  transcription too (same `on_record` path).

### New/changed code

- `app/transcription/engine.py`: `transcribe_file` accepts an
  optional `is_cancelled: Callable[[], bool]` checked per segment;
  on cancel it returns the segments so far (do not raise). Keep the
  existing signature compatible (keyword-only, default None).
- `app/nearlive/chunker.py`: accept a `model` (already does) and an
  optional `is_cancelled` passed through to the engine; add
  `save_live_transcript(session_dir)` / `load_live_transcript`
  helpers, or put them in `app/transcription/exports.py` — pick the
  home that keeps the chunker Qt-free and testable without Qt.
- `app/ui/main_window.py`: the live poller (QTimer + snapshot +
  chunker), the Record view live-transcript widget, finalize
  (live + tail → canonical files), Cancel wiring. Keep the worker
  threading pattern already used by `_TranscribeWorker`
  (QObject + QThread + progress/finished/error signals).
- `app/storage/sessions.py`: if a `transcript-live.json` filename
  constant is needed, define it next to the other file constants;
  do NOT add the live file to the retention "has transcript" check.

### Tests (new)

- Chunk + poll: fake engine, growing WAV — assert chunks land in
  order, the live file is written after each chunk, and the file
  contains all segments up to the last completed window.
- Crash mid-transcription: raise in the fake engine mid-chunk (and
  separately: kill the worker after N chunks) — assert
  `transcript-live.json` on disk holds exactly the completed chunks
  and the session can be finalized from it (finalize path produces
  a complete canonical transcript).
- Finalize: live segments + tail merge with no duplicates and no
  gaps (overlap dedup).
- Cancel: `is_cancelled` set mid-file — assert transcribe returns
  partial segments, no exception, no partial `transcript.json`
  written by the manual path.
- Pause: poller does not advance the watermark while paused.
- Model download: engine construction with an uncached model shows
  the download status (fake the cache check; do not download a real
  model in tests).

## Item 3 — Mid-transcribe crash

John has no log to share, so no root cause is claimed. What the
investigation must do:

1. Read the single-pass path (`TranscriptionEngine.transcribe_file`
   → `transcribe_session`, `_TranscribeWorker.run` in main_window)
   and list the concrete crash vectors on a busy work laptop:
   - Whole-file single pass with the small model: model (~466 MB
     download + ~1 GB RAM while running) + full audio in memory;
     on an 8 GB machine running Teams + browser, memory pressure
     is the prime suspect (note `docs/CPU_TIMES.md`: small needed
     swap on the 4 vCPU test VM).
   - First-use model download inside `WhisperModel(...)` with no
     progress feedback — indistinguishable from a hang, and a
     stalled download looks like "never finishes".
   - QThread worker: an unhandled exception in the worker thread
     (e.g. OOM, model-load failure) — check what `_on_transcribe_error`
     does today and whether the auto-process chain state
     (`_transcribing`, `_auto_processing`) can be left stuck.
2. The fix is the item-2 design: bounded memory per chunk
   (one 25 s window + one model), progressive saves, no single giant
   pass. State in the report which vectors the redesign removes and
   which remain hypotheses (e.g. memory pressure is unproven without
   the log or a repro).
3. Do NOT invent a repro on Windows (no real Teams audio here).
   The crash-simulation tests in item 2 are the verification: a
   crash mid-transcription loses at most the in-flight chunk.
4. If the investigation finds a *separate*, concrete, fixable defect
   in the transcribe path (e.g. a stuck flag, an unguarded slot),
   fix it in the same round and say so; do not go hunting for
   speculative fixes beyond that.

### Findings (implemented Nov 19, 2026 — no root cause claimed)

Read: `TranscriptionEngine.transcribe_file` → `transcribe_session`
(`app/transcription/engine.py`), `_TranscribeWorker.run`,
`_on_transcribe_error` / `_on_transcribe_finished`
(`app/ui/main_window.py`), and the installed faster-whisper
(`transcribe.py:387`, `audio.py: decode_audio`). John has no log,
so nothing below is a proven root cause — vectors are what the
code can actually do, and the redesign removes or mitigates them.

Concrete crash vectors on a busy work laptop:

1. **Single pass, all-or-nothing output.** `transcribe_session`
   writes `transcript.json` / `.txt` / `.srt` only after the full
   pass returns. A crash, OOM-kill, or force-quit at any point
   before that leaves zero transcript and the whole meeting must
   be re-transcribed from scratch.
2. **Whole-file decode + model resident at once.** faster-whisper's
   `transcribe()` decodes the entire audio file into one float32
   array first (`decode_audio`, verified in the installed
   faster-whisper), then runs the model over it. With the old
   `small` default that is a 466 MB download and ~943 MB peak RSS
   (measured; it needed swap on the 4 vCPU build VM — see
   `docs/CPU_TIMES.md`), on top of Teams + a browser on an 8 GB
   machine. Memory pressure is the prime suspect — and it is still
   only a hypothesis: no log, no repro on Windows.
3. **First-use download looks exactly like a hang.**
   `_load_model` constructs `WhisperModel(...)` directly; the
   download happens inside it with no progress feedback, and
   faster-whisper offers no download-progress hook. A stalled
   download is indistinguishable from "never finishes".
4. **No timeout, no cancel on the single pass (before this
   change).** The worker's `try/except` covers exceptions, not
   hangs: a stalled download or a pathological decode leaves
   `_transcribing` True, Transcribe disabled, and only an app
   restart clears it.
5. **The error path itself is sound — not a vector.** An exception
   in the worker is caught, surfaced via the `error` signal, and
   `_on_transcribe_error` clears `_transcribing` and
   `_auto_processing` and shows a critical dialog; the finished
   path clears `_transcribing` too. No stuck-flag defect was
   found on the exception path. A native crash (segfault,
   OOM-kill) kills the whole process, so no in-app handler could
   have recovered — only progressive saves bound the loss.
6. **Model left resident after a failed run (fixed this round).**
   `_TranscribeWorker.run` called `engine.unload()` only after a
   successful `transcribe_session`; on the exception path it
   returned early, and MainWindow keeps the worker until the next
   run — so a failed run's model stayed in RAM and a retry piled
   a second load onto the first. Fixed: `unload()` now runs in a
   `finally` (test: `test_failed_transcription_unloads_the_model`).

What the item-2 redesign removes, mitigates, or leaves:

- **Removed:** the all-or-nothing output (vector 1) — the live
  transcript is saved after every chunk, so a crash loses at most
  the in-flight chunk; and the single giant Whisper pass — the
  live path transcribes one 25 s window at a time (base RTF 0.135
  → ~3.4 s per chunk) and Stop only transcribes the tail.
- **Removed for manual transcription:** the no-cancel hang
  (vector 4) — the Cancel button checks `is_cancelled` per
  segment and writes no partial `transcript.json`. The
  auto-process chain still has no cancel: if it hangs, only
  stopping the app clears it.
- **Mitigated, not removed:** the first-use download (vector 3)
  — the UI now shows "Downloading Whisper base (~142 MB) — first
  run only" with elapsed time, but the download still happens
  inside `WhisperModel(...)` with no byte-level progress (there
  is no hook to show one). And memory pressure (vector 2) —
  Item 1's base default (~392 MB peak vs ~943 MB for small) plus
  chunking lowers it, but the fallback full-file path still
  decodes the whole file at once, and `process_new_audio` still
  loads the whole combined snapshot into RAM via `load_wav_mono`
  on every tick (float32 mono; a 60-minute 48 kHz snapshot is
  ~690 MB) and re-copies + re-mixes the growing tracks each
  tick — a known remaining limitation, not fixed in this item.
- **Remains hypothesis:** memory pressure as the actual cause of
  John's 1.0.2 crash. No repro on Windows was invented; the
  crash-simulation tests in item 2 are the verification (a crash
  mid-transcription loses at most the in-flight chunk, and the
  session finalizes from `transcript-live.json`).

## Item 4 — Full UI redesign (modular, focused, separation of tasks)

### Target shape

Left navigation (one focused view per task), Record view as the
default landing view. Proposed views (adapt if the research dossier
shows a better split — the principle is one job per view, nothing
unrelated on screen):

1. **Record** — only recording: target-app picker, capture-mode,
   source dropdown, mic/system device pickers, level meters, timer,
   Record/Pause/Stop, live transcript as it lands, status line.
   Nothing about diarization, speakers, or settings here.
2. **Library** — the session list (today's Recording Browser),
   search (text/semantic/hybrid), source filter, calendar view,
   "Open in Session view". Double-click / Open loads the session.
3. **Session** — the one open session: transcript (Original/Cleaned
   toggle, Re-clean, Polish), Process actions (Transcribe, Diarize,
   Suggest) with progress, speaker rename + suggestions for this
   session, recording source editor, Outlook link display + "Open in
   Outlook". This is where the old Transcript/Process/Speakers tabs
   land, scoped to one session.
4. **People** — known speakers: list, play sample, rename, learned
   identities. Nothing session-specific here.
5. **Settings** — the existing Settings dialog content as a view
   (or keep the dialog — decide in implementation; a view is
   preferred for consistency, the dialog is acceptable if moving it
   costs more than it saves). MCP setup and Help are reachable from
   here / the nav footer, out of the daily path: Help (docs) and
   MCP setup as their own nav entries under a "More" group or as
   footer items — they must stay reachable, not hidden.

### Rules

- Every existing capability stays reachable: calendar view, MCP
  setup tab content (incl. the Codex snippet), Help docs rendering,
  recording source editing, Outlook link display, auto-detect
  settings + consent, retention settings, names dictionary,
  semantic search, export/summarize actions.
- Preserve widget attribute names the tests rely on where possible
  (see research dossier for the inventory); rename only where the
  redesign forces it, and update the tests in the same change.
- Keep the theme system (`app/ui/theme.py`) — the redesign is
  layout/navigation, not a new visual language. Keep light default.
- Keep the 1280x800 fit: every view must be geometry-clean at
  1280x800 in both themes (the existing geometry tests are the bar).
- The Record view must work with no session loaded (fresh start);
  the Session view shows a real empty state when nothing is loaded.

### Screenshots

The layout changes materially, so re-render `screenshots/` and
`docs/SCREENSHOTS.md` from the new UI using the existing pipeline
(`~/workspace/meettrace-tryouts/gui-shots/`: re-seed demo data first
— `seed_demo_session.py` + `prep_demo_data.py` — /tmp gets cleaned;
restore demo speaker names via `rename_identity`
SPEAKER_00→John, 01→Priya, 02→Marcus, regenerate suggestions.json).
New shots: Record view (idle + recording), Library, Session,
People, Settings. Keep the dark/light main shots. Images for John
live under `~/workspace`, never /tmp.

## Sequencing

1. Item 1 (base default) — small, independent, first.
2. Items 2 + 3 (live transcription + crash resilience) — the core
   fix; item 3's investigation rides with it.
3. Item 4 (redesign) — on top of 1 + 2, because it moves the Record
   view and the live-transcript widget item 2 adds. Do not build 4
   in parallel with 2 (both touch `main_window.py` heavily).
4. Suites green after each item, not just at the end.

## Lessons captured (Nov 19 review round — ce-compound)

Three beliefs this round corrected; each is guarded by a test, not
just by this note.

1. **Never run a transcription pass on the UI thread.** The item-2
   design put the Stop/finalize tail (`chunker.finish()`) in a UI
   slot, assuming the tail is always short. Wrong: when live dies
   early the tail is the bulk of the meeting, so Stop froze for
   minutes — the exact symptom this round was built to fix. Rule:
   every transcription pass (live chunk, tail finalize, full-file)
   runs in a worker thread with progress + cancel. Guarded by
   `test_stop_finalize_merges_live_and_tail` (waits for the async
   worker) and the timeout-abandon / finalize-cancel tests in
   `tests/test_live_transcription.py`.
2. **A deferred finalize needs a timeout.** Waiting for an in-flight
   live chunk at Stop had no bound — a worker stuck in a model
   download would leave Stop hanging forever ("Finishing the live
   transcript…"). Rule: 60 s (`LIVE_FINALIZE_WAIT_SECONDS`), then
   abandon the in-flight chunk and finalize from the saved live
   file. Guarded by the timeout-abandon test.
3. **The blocking model constructor cannot be interrupted.** Cancel
   during a first-use download cannot stop the download itself.
   Rule: check the cancel flag before *and* after `_load_model()`,
   and tell the user honestly that cancel takes effect once the
   model has loaded. Guarded by the engine cancel tests.
4. **Design correction from research (not a bug): `combined.wav`
   does not grow during recording** — it is created only at stop by
   mixing. Any live feature must poll the growing `mic.wav` /
   `system.wav` (snapshotted + mixed), never `combined.wav`.
   Guarded by the live-poller tests, which snapshot the tracks.

## Verification gates (before the coordinator reports done)

- Full phase2 suite green, full Qt suite green (2 deselects),
  `python -m compileall` clean.
- README + docs describe the new default, live transcription,
  progressive saves, cancel, and the new navigation — verified by
  reading, not memory.
- Crash-simulation tests pass (item 2).
- Screenshots re-rendered from the new UI.
- ce-code-review run on the diff; ce-compound captures any lesson
  a fix taught.
- Report to Larry: per-item file list, test results, how John can
  verify each item on his PC, deviations, what could NOT be verified
  (real Teams audio on Windows — never claim it), suggested next
  step (1.0.3 build) without doing it.
