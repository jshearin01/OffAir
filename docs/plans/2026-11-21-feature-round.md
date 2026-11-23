---
title: OffAir feature round (fix-first + people pages + bookmarks + dictation + enterprise cleanup + UI) - Plan
type: feat
date: 2026-11-21
artifact_contract: ce-unified-plan/v1
product_contract_source: user-directed (Nov 21 session)
execution: code
---

# OffAir feature round — Plan

## Goal Capsule

- Objective: OffAir is more correct on real meetings and more useful day to day — diarization labels can no longer silently poison speaker profiles, the Ollama summary path is proven to work, and the user gets people pages, bookmarks, dictation, enterprise-LLM cleanup, and a round of UI improvements — without re-cluttering the Record view, without regressing the 1.0.4 Stop-freeze fixes, and without any new write surface on MCP.
- Means: implement the 18 approved items in three parallel core copies, integrate, then build the UI layer and the MCP read-only tools on the integrated tree (KTD1).
- Authority hierarchy: John's Nov 21 approvals > this plan > prior plans. The PRD hard constraints (CPU-only, fully local core, Windows-first, PyQt6) still bind. The 1.0.4 invariants (flat 8-entry nav, minimal Record view, model-default rules, worker finalize) are not to be regressed.
- Stop conditions: any item that cannot be built honestly (no fake verification) ships as the honest subset with the gap reported, never faked. No push to GitHub — the coordinator (parent agent) pushes after the report.

## Product Contract

Summary: implement exactly the 18 items John approved Nov 21: 5 fix-first items from real testing, 4 features (people pages, bookmarks, dictation, enterprise-LLM cleanup), and 9 UI improvements — plus read-only MCP coverage for the new data (added Nov 21). Nothing else. No commitments tracker (not picked). No write MCP tools. No unrequested additions.

Problem Frame: the YouTube test proved two learning-integrity defects (persisted labels can disagree with returned labels; mixed clusters can be learned from); the Ollama summary path was never run for real; long recordings have no disk warning; auto-detect misses opening syllables. The features John picked are the ones that fit the product's edge (private + knows your people) and the UI items are the ones he named after living with 1.0.3/1.0.4.

### Requirements

Fix-first:

- R1. Learning consumes exactly the labels diarization returned: the file write path writes the returned segments verbatim (no re-derivation), and a reconciliation check fails loudly if persisted labels ever disagree with the returned payload. Covers item 1.
- R2. A mixed-speaker cluster is never enrolled or learned from: the existing auto-enrol guardrails (clean embedding, ≥5 s clean audio, half-split consistency) extend to every learn path; a cluster failing them stays Unknown. Covers item 1.
- R3. Regression tests reproduce the persisted-label disagreement synthetically and prove the guard blocks a mixed cluster. The original YouTube sessions are gone; no real-audio claim. Covers item 1.
- R4. Sherpa clustering threshold (0.5) and ECAPA match threshold (0.55) are swept on the 3-speaker fixture (9 turns) and the numbers (turn accuracy, speaker accuracy, cluster count) are reported. A default changes only on clear evidence; otherwise defaults stay and the open gate (real-meeting validation) is documented. Covers item 2.
- R5. The Ollama summary wiring is proven end-to-end with whatever can actually run on this VM (a small real model if installable, else the probe path) and the exact real-machine procedure is documented. Never claim a qwen3.5:9b summary ran if it didn't. Covers item 3.
- R6. While recording, free disk on the recordings volume is checked (sane threshold, documented) and a loud warning (status + dialog, once per session) fires before the disk fills. A recording is never stopped silently for disk reasons. Covers item 4.
- R7. Pre-roll ring buffer is opt-in, default OFF everywhere, with a plain-language consent note (it captures before Record is pressed; memory-only, never written to disk unless a recording starts). When enabled, auto-detect start prepends the preceding N seconds (default 30 s, configurable). Covers item 5.

Features:

- R8. People pages: clicking a person in People opens their page — every meeting with them, everything they said (segments, click-to-play where audio exists), their voice sample, rename. No commitments section, no placeholder for one. Covers item 6.
- R9. Bookmarks: a Bookmark button during recording and on a loaded session marks the current moment; stored per session in metadata; listed on the Session view; clicking one seeks and highlights. A bookmark pointing at deleted audio says so plainly. Covers item 7.
- R10. Bookmark hotkey: an in-app keyboard shortcut bookmarks while recording; a global hotkey only if feasible on Windows without new dependencies, otherwise documented as needing window focus. Covers item 7.
- R11. Dictation mode: a mic-only flow producing a cleaned transcript note, reusing the solo source and session machinery, entered via exactly one clearly-labeled control that does not re-clutter the Record view. UI/docs state the transcript captures the spoken side only. Covers item 8.
- R12. Enterprise-LLM cleanup via `claude -p` headless subprocess (NOT ACP, NOT A2A): opt-in, off by default, a clearly-marked derived artifact; verbatim transcript never modified; transcript TEXT ONLY leaves the machine, audio never does — stated plainly in UI and docs; guided error when the CLI is absent/fails; timeout + cancel; never blocks the UI thread; existing deterministic + Ollama layers untouched. Covers item 9.

UI improvements:

- R13. Waveform timeline on the Session view: speaker-colored segments, bookmark markers, click a segment to hear that exact moment. Covers item 10.
- R14. Always-visible recording indicator: tray icon or mini window with elapsed time + Stop; never the only way to stop. Covers item 11.
- R15. Keyboard shortcuts: bookmark, pause/resume, jump-to-search; documented in Help. Covers item 12.
- R16. Search by person: filter sessions/segments by speaker in Library search and calendar. Covers item 13.
- R17. Live transcript provisional styling: unconfirmed chunks greyed, speakers colored, model/diarization-mode shown honestly; confidence shown only when the engine reports it — never an invented number. Covers item 14.
- R18. First-run onboarding: a short first-launch pass (what happens, what stays on the machine, the ~175 MB first-run downloads happen once); never nags on later launches. Covers item 15.
- R19. Session list: pin/star sessions + group by week. Covers item 16.
- R20. Follow-along highlighting during playback: the segment being played lights up in the transcript. Covers item 17.
- R21. One-click theme toggle + a disk-usage line (free space / what retention is about to delete), composed with R6's machinery, not duplicated. Covers item 18.

MCP (read-only, added Nov 21):

- R22. Read-only MCP coverage for the new data: per-speaker detail tool; bookmarks in get_session and/or a list_bookmarks tool; dictation notes reachable via existing tools (no new tool if covered — report says so); get_transcript variant for the enterprise-cleaned artifact, clearly labeled; search_meetings speaker filter; pinned state in list_sessions/get_session (+ filter if cheap). No write tools, no new network surface, stdio, same files the app reads. MCP setup page tool list + docs/MCP.md updated; smoke test covers the new tools at build time.
- R23. README.md describes every behaviour/default/feature change in the same change; docs/ updated where applicable; screenshots re-rendered for every changed view.

Scope Boundaries: no commitments tracker; no CRM/sales features; no cloud sync; no joining calls as a bot; no video; no phone companion; no translation; no new dependencies for a global hotkey; no changes to the 1.0.4 Stop-freeze machinery except where a unit explicitly needs a hook (reported as a deviation if touched).

## Planning Contract

- KTD1. Three parallel core copies (`~/workspace/offair-u1-labels`, `~/workspace/offair-u2-infra`, `~/workspace/offair-u3-data`), each a copy of main @ 9d6efcd7 excluding `installer/` and `.venv-phase2`; coordinator integrates sequentially with diff review; UI (U4) then MCP (U5) run on the integrated tree. Prevents same-tree edit conflicts; file ownership below is binding — a unit that must touch another unit's file reports it instead of doing it.
- KTD2. Persisted-label fix: single source of truth is the segment list `diarize_session` returns; the write path serializes exactly that list (same objects, no re-rounding or re-derivation); learning reads `transcript.json` only after a write-time reconciliation (compare persisted speaker labels to the returned list; mismatch raises, never silently persists). Mixed-cluster guard reuses the auto-enrol guardrails (`app/speakers/session.py` clean-window rules) as one shared predicate used by manual and automatic learn paths.
- KTD3. Threshold sweep: sherpa threshold in {0.4, 0.5, 0.6} × ECAPA match threshold in {0.45, 0.55, 0.65} on `tests/fixtures/meeting-3speakers*` (9 turns, 3 TTS voices); report turn accuracy, speaker accuracy, cluster count vs 3, over-segmentation notes. Change a default only if one setting wins clearly on all three metrics; otherwise leave defaults and document the open gate.
- KTD4. Ollama proof: check for an Ollama binary on this VM; if present, run the smallest available model end-to-end through `OllamaClient.summarize` on the fixture transcript and record wall time + output shape; if absent, attempt an install only if it is quick and CPU-feasible, else prove via `is_available` probe + fake-client tests and document the exact real-machine procedure (install Ollama, `ollama pull qwen3.5:9b`, OFF by default, localhost only). Never claim 9b quality without a 9b run.
- KTD5. Disk warning: `shutil.disk_usage` on the recordings volume, threshold 1 GB free, checked every 30 s while recording; first crossing fires a status message + one dialog per session; the check is a pure helper in `app/storage/retention.py` so the disk-usage line (R21) reuses it; retention gains a dry-run scan returning bytes that would be deleted.
- KTD6. Pre-roll: `app/audio/preroll.py` ring buffer (collections.deque of timestamped chunks, memory-only) fed by the capture layer only while pre-roll is enabled; `MeetingWatcher.on_meeting_started` prepends the buffered seconds to the new session; config `preroll_enabled` (default False) + `preroll_seconds` (default 30, range 10–120); Settings shows the consent note verbatim. Manual Record prepends only if the same opt-in is on.
- KTD7. People pages: new `app/speakers/people.py` query module (speaker → sessions, utterances, sample availability) so U1's `speakers/session.py` is untouched by U3; new `app/ui/views/person_view.py` page; People view click navigates to it; back button returns. Rename reuses the existing rename panel path.
- KTD8. Bookmarks: new `app/storage/bookmarks.py` (add/list for a session, stored in `metadata.json` under `bookmarks: [{t, label, created}]` via `update_metadata`); Session view lists them; click seeks via `_play_audio` and highlights the segment; missing audio → disabled entry with the retention reason in plain text. Hotkey is an in-app `QShortcut` (Ctrl+B); global hotkey only if Windows-feasible with stdlib/ctypes and no new dependency, else documented as focus-required.
- KTD9. Dictation entry: one `Dictation` toggle button on the Record view beside Record; when on, the app picker and system meter hide, capture is mic-only, `source=solo`, metadata `dictation: true`; Stop produces the normal session plus the deterministic cleaned note as the primary readable artifact. One control is the most the Record view gains.
- KTD10. Enterprise cleanup: new `app/transcription/enterprise_cleanup.py` — `subprocess.run(["claude", "-p", prompt], input=text, timeout=120)` in a worker thread with cancel; artifact `transcript-enterprise-clean.txt` (+ `.json` if segment shape is kept — plan leaves the shape to the unit, documented); Session view button labeled with the text-leaves-machine warning; missing CLI → guided error naming the install; the fixed prompt states the same cleanup rules as the deterministic layer. Verbatim, deterministic-clean, and enterprise-clean are three distinct artifacts; the Original/Cleaned toggle gains the third variant explicitly labeled.
- KTD11. Tray: `QSystemTrayIcon` created only when `QSystemTrayIcon.isSystemTrayAvailable()`; menu Show / Stop / Quit; tooltip shows elapsed time; tray Stop calls the same `on_stop` path; no tray on offscreen/Linux without a system tray → silent degrade, main window remains the stop path.
- KTD12. Waveform: new `app/ui/waveform.py` custom-painted widget; RMS buckets (0.25 s) from `combined.wav` (fallback `mic.wav`) computed in a worker, cached in memory per session; segments painted in speaker colors from the loaded transcript; bookmark markers from R9; click maps x → time → `_play_audio(start, end)` + segment highlight. No UI-thread audio decoding.
- KTD13. Theme toggle: one button in the nav header toggling `config.theme` light/dark through the existing `apply_theme` machinery, persisted in settings.
- KTD14. Onboarding: modal dialog on first launch only, gated by `onboarding_shown` in settings; content: record → transcribe → diarize → learn, everything stays on the machine, first-run downloads (~142 MB Whisper base + ~33 MB sherpa, once); a "Show this again from Help" line; Help gains the same content.
- KTD15. Pins: metadata `pinned: bool` via `update_metadata`; Library sorts pinned first then newest; week-group headers (ISO week, "This week" for the current week); star toggle per row; pinned state is data, so R22 exposes it.
- KTD16. Follow-along: `QMediaPlayer.positionChanged` (plus a QTimer poll fallback) maps playback position to the current transcript segment; the transcript list highlights it and scrolls it into view only while playback is active (never yanks scroll on manual browsing).
- KTD17. Provisional styling: live-transcript list items carry a provisional flag until finalize confirms them (grey + italic); speaker color chips from a fixed palette keyed by label; the live status line shows model + diarization mode; no confidence number is shown unless the engine reported one (neither Whisper segments nor sherpa turns carry per-segment confidence — the UI says "provisional" instead).
- KTD18. Search by person: `search_sessions(..., speaker=...)` in `app/storage/browser.py` (U3 core) matching transcript speaker labels + speakers.db names; Library gains a speaker combo; Calendar gains the same filter dimming non-matching days (or filtering badges — unit decides, documented).
- KTD19. Disk line: Library footer + Settings line from KTD5's helpers: free space on the recordings volume + bytes retention would delete on next prune.
- KTD20. MCP tools: `get_speaker(name)`, `list_bookmarks(session)` (and bookmarks embedded in `get_session`), `get_transcript(..., variant="verbatim"|"clean"|"enterprise")`, `search_meetings(..., speaker=...)`, `list_sessions`/`get_session` pinned fields + `pinned_only` filter. All read-only; same file-reading code paths as the app; `tests/test_mcp.py` covers each; `docs/MCP.md` + the MCP setup page list them in plain language.

## Implementation Units

Unit index: U1 labels+thresholds · diarization/speakers · — · U2 ollama+disk+preroll · retention/audio/exports · — · U3 data cores · speakers/storage/transcription · — · U4 UI layer · ui/views/theme · waits on U1–U3 integration · U5 MCP tools · mcp/ · waits on U4 · U6 screenshots+verification · coordinator · waits on U5.

### U1. Persisted-label fix + mixed-cluster guard + threshold sweep (items 1–2)

- Goal: learning can no longer be poisoned by a persisted label that disagrees with the returned payload; thresholds are measured honestly.
- Requirements: R1, R2, R3, R4, R23 (docs only for the touched areas).
- Files: `app/diarization/session.py`, `app/diarization/sherpa.py` (threshold plumbing only), `app/speakers/session.py`, `app/speakers/matching.py` (threshold const only if changed), `tests/test_diarization*.py`, `tests/test_speakers.py`, `tests/test_auto_identify.py`, `docs/DIARIZATION.md`, `docs/SPEAKERS.md`, new `docs/notes/threshold-sweep-2026-11-21.md`. Copy: `~/workspace/offair-u1-labels`. Do NOT touch `app/ui/*`, `app/mcp/*`, `app/speakers/people.py` (does not exist yet).
- Approach: KTD2, KTD3. Find the exact write-path divergence first (compare returned vs persisted labels on a synthetic boundary case), fix at the source, add the reconciliation check, unify the guard predicate, run the sweep, write the note.
- Test scenarios: synthetic boundary segment persists exactly the returned label; a mixed cluster is refused by every learn path; sweep table reproduced from a script in the report.
- Verification: scoped tests + full phase2 suite in the copy; report the sweep table and whether any default changed.

### U2. Ollama proof + low-disk warning core + pre-roll core (items 3–5)

- Goal: the Ollama path is proven as far as this VM allows; disk warnings and pre-roll exist as tested core machinery with documented hooks for the UI layer.
- Requirements: R5, R6, R7, R23 (docs for touched areas). UI wiring is U4's job — U2 exposes hooks and reports their exact signatures.
- Files: `app/exports/local_llm.py` (evidence only unless a bug is found), `app/storage/retention.py` (disk helpers + dry-run scan), new `app/audio/preroll.py`, `app/audio/auto_detect.py` (start hook), `app/audio/capture.py` (minimal pre-roll handoff), `app/config.py` (preroll_* + disk_* settings blocks only), `tests/test_retention.py`, `tests/test_auto_detect.py`, new `tests/test_preroll.py`, `docs/RETENTION.md`, `docs/PER_APP_CAPTURE.md`, new `docs/notes/ollama-proof-2026-11-21.md`. Copy: `~/workspace/offair-u2-infra`. Do NOT touch `app/ui/*`, `app/mcp/*`.
- Approach: KTD4, KTD5, KTD6. Report: Ollama evidence (what ran, wall time, output shape, or why not + procedure doc), disk helper signatures, pre-roll API + config keys + consent-note text.
- Test scenarios: disk helper thresholds; dry-run scan byte counts; ring buffer prepend correctness on synthetic audio; pre-roll OFF leaves capture byte-identical.
- Verification: scoped tests + full phase2 suite in the copy.

### U3. Data cores: people queries, bookmarks, dictation pipeline, enterprise cleanup, search-by-person core, pins core (items 6–9 cores, 13-core, 16-core)

- Goal: every new data shape exists, tested, with exact on-disk formats reported for the UI and MCP layers.
- Requirements: R8 (data), R9 (storage), R11 (pipeline), R12 (runner + artifact), R16 (search core), R19 (pin storage), R23 (docs for touched areas).
- Files: new `app/speakers/people.py`, new `app/storage/bookmarks.py`, `app/storage/sessions.py` (minimal helpers only), `app/storage/browser.py` (speaker param + pinned sort), new `app/transcription/enterprise_cleanup.py`, `app/transcription/cleanup.py` (third-variant registration only), `app/config.py` (enterprise-cleanup settings block only), `tests/test_browser.py` (additions), new `tests/test_people.py`, `tests/test_bookmarks.py`, `tests/test_dictation.py`, `tests/test_enterprise_cleanup.py`, `docs/SPEAKERS.md`, `docs/TRANSCRIPT_CLEANUP.md`, `docs/RECORDING_SOURCE.md`. Copy: `~/workspace/offair-u3-data`. Do NOT touch `app/ui/*`, `app/mcp/*`, `app/speakers/session.py`.
- Approach: KTD7, KTD8, KTD9 (pipeline side), KTD10, KTD15 (storage), KTD18 (core). The `claude` runner must be subprocess-only, timeout + cancel, worker-thread-safe; the dictation pipeline reuses the existing session/transcribe/cleanup path with `source=solo` + `dictation: true`.
- Test scenarios: people queries across sessions incl. renamed speakers; bookmark round-trip + missing-audio honesty; dictation session shape; enterprise runner with a fake `claude` on PATH (success, missing CLI, timeout, cancel); speaker filter + pinned sort.
- Verification: scoped tests + full phase2 suite in the copy; report lists every new on-disk format verbatim (file names, JSON keys) for U4/U5.

### U4. UI layer: people pages, bookmarks UI, dictation entry, enterprise-cleanup UI, all 9 UI improvements (items 6–9 UI, 10–18)

- Goal: everything from U1–U3 is reachable in the app, and the 9 UI improvements land, without re-cluttering Record or regressing 1.0.4.
- Requirements: R8–R21 (UI halves), R23.
- Files: `app/ui/main_window.py`, `app/ui/views/*`, new `app/ui/views/person_view.py`, new `app/ui/waveform.py`, `app/ui/theme.py`, `app/ui/browser.py` (library UI only), `docs/SCREENSHOTS.md`, `docs/OVERVIEW.md` (+ new `docs/KEYBOARD_SHORTCUTS.md`), README (UI descriptions). Runs in `~/workspace/offair` AFTER coordinator integration. Do NOT touch `app/mcp/*`.
- Approach: KTD7–KTD19 as applicable. Preserve: Record view gains exactly one control (Dictation toggle, KTD9); nav stays 8 flat entries (person pages are reached from People, not a new nav entry); Stop-freeze guards untouched; `_play_audio` is the single playback path; waveform decoding never on the UI thread.
- Test scenarios: Qt tests for each new widget/behavior (person page navigation + back; bookmark add/list/seek; dictation toggle hides picker; tray degrade offscreen; shortcuts fire; speaker filter; provisional styling flags; onboarding once-only; pin sort + week groups; follow-along highlight; theme toggle persists; disk line renders).
- Verification: full phase2 + full Qt suite (2 deselects) in the tree; screenshots re-rendered for every changed view (coordinator pass U6).

### U5. MCP read-only tools for the new data

- Goal: a connected client can read everything new the app can show, read-only.
- Requirements: R22, R23 (docs/MCP.md + setup page).
- Files: `app/mcp/tools.py`, `app/mcp/server.py` (registration), `tests/test_mcp.py`, `docs/MCP.md`, `app/ui/mcp_setup.py` (tool list text), `~/workspace/mcp_smoke_test.py` (extend with the new tool calls). Runs in `~/workspace/offair` AFTER U4.
- Approach: KTD20. No write tools; reuse the app's file-reading code; `get_transcript` variant param defaults to verbatim (existing callers unchanged).
- Test scenarios: each new tool against synthetic session data; variant labeling; speaker filter; pinned fields; dictation note reachable (or a report line saying the existing tools already reach it, with the evidence).
- Verification: scoped tests + full phase2 suite; report lists the final tool inventory.

### U6. Screenshots + final verification (coordinator)

- Goal: the tree is green, reviewed, and visually current.
- Requirements: R23.
- Approach: re-render screenshots via `~/workspace/meettrace-tryouts/gui-shots/` for every changed view; run both suites; compileall; blob-SHA diff vs main 9d6efcd7; direct review pass (ce-code-review single-pass form — the full multi-agent spine is unavailable at this depth, as in the 1.0.4 round) with a receipt at `~/workspace/offair-review-2026-11-21.md`; ce-compound capture of any corrected belief.
- Verification: phase2 553+/150 baseline must not regress (counts may grow); Qt 696+/5/2 baseline must not regress; diff lists exactly the changed set.

## Verification Contract

- Phase2 venv: `cd <tree> && ~/workspace/offair/.venv-phase2/bin/python -m pytest tests/ -q -p no:cacheprovider`. Baseline at 1.0.4: 553 passed / 150 skipped.
- Qt venv: `QT_QPA_PLATFORM=offscreen ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python -m pytest tests/ -q -p no:cacheprovider --deselect tests/test_ui_phase5.py::test_show_crash_dialog_without_qt_raises_clear_error --deselect tests/test_ui_phase5.py::test_main_window_imports_without_qt_and_settings_dialog_guards`. Baseline: 696 passed / 5 skipped / 2 deselected. Clear stale `/tmp/pytest-of-*` before Qt runs (`/tmp` is a 512 MB tmpfs; the 10-minute-track tests fill it). If Qt imports fail with a missing libEGL, `sudo apt-get install -y libegl1 libopengl0` and retry.
- `python3 -m compileall -q app tests` clean. Blob-SHA diff: `python3 ~/workspace/diff_vs_main.py 9d6efcd7` (pass the HEAD explicitly; the script's default SHA is stale).
- Screenshots: gui-shots pipeline (`shots.py` + `prep_demo_data.py`, phase2 venv; re-seed demo data if cleaned, restore speaker names via `rename_identity`, regenerate suggestions.json); images under `~/workspace`, never `/tmp`.

## Definition of Done

- All 18 items + R22 built or honestly subsetted with the gap named; no item faked; no unrequested addition in the diff.
- Both suites green at coordinator verification; review receipt exists; plan doc is in the tree.
- README + docs describe the new behaviour; screenshots current for changed views.
- Report to the parent per item: what was built, how John tests it on his PC, deviations and why, what could NOT be verified (real Teams audio, real Outlook, real Ollama 9b, RDP-only visual checks — never claimed).
- Nothing pushed to GitHub by this round; the parent pushes after the report.
