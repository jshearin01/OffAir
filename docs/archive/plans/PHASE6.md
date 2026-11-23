# PHASE6 — Sherpa diarization + modern UI + Linux/Windows

**Date:** 2026-11-08. **Status:** built locally, verified, NOT pushed (per
instructions — push is a separate batched step). Baseline at start: main @
`0323cc1` (all 5 phases + ECAPA swap + semantic search), 171 tests passing.
**End state: 236 passed, 7 skipped** (repo venv) **+ 18 Qt tests passed**
(gui-shots venv, offscreen). Three tracks ran as subagents under one
coordinator with strict file ownership so no two tracks edited the same
file at once; Track 2 (UI) ran after Tracks 1 and 3 and wired both in.

## 1. What was built

### Track 1 — Sherpa diarization
- `app/diarization/sherpa.py` (new): `SherpaDiarizer`,
  `ensure_sherpa_models`, `check_availability`,
  `validate_sherpa_settings`, `SherpaDiarizationResult`. sherpa-onnx
  `OfflineSpeakerDiarization`: pyannote segmentation ONNX + 3D-Speaker
  ERes2Net embeddings. Lazy import; models (~33 MB) download once on
  first use from public GitHub releases into
  `~/.offair/models/sherpa-diarization/` (segmentation tag
  `speaker-segmentation-models`, embedding tag
  `speaker-recongition-models` — sherpa-onnx's own typo'd tag), with
  progress + a guided offline error. **No HF token anywhere.**
- Threshold-based clustering for unknown speaker counts (default
  threshold 0.5, documented as midway between measured same-speaker
  cosine 0.82–0.89 and cross-speaker ~0.0–0.02; 0.5/0.6/0.7 all gave
  identical results). Optional expected-speaker-count hint from
  settings. Raw cluster numbering gaps (00/01/03) are renumbered to
  contiguous `SPEAKER_00/01/02` in order of first appearance, matching
  the pyannote convention.
- `app/diarization/__init__.py`: `DIARIZATION_MODES` registry
  (`simple` / `full` / `sherpa`, label + description) for the UI picker.
- `app/config.py` (additive only): `sherpa_model_dir`,
  `sherpa_expected_speakers`, `sherpa_clustering_threshold`;
  `"sherpa"` added to valid `diarization_mode`;
  `sherpa-eres2net` added to `EMBEDDING_BACKENDS` and
  `sherpa-eres2net-v1: 0.55` to `SPEAKER_MATCH_THRESHOLDS`.
- `app/speakers/embeddings.py`: `SherpaEres2NetEmbeddingExtractor`
  (`sherpa-eres2net-v1`, 192-dim, same `extract`/`extract_samples`
  interface, pinnable via `get_extractor("sherpa-eres2net")`; the
  `auto` chain is unchanged). It had the best separation of everything
  tested (margin 0.810, cross ≈ 0) and fit the interface cleanly.
- `app/diarization/session.py`: new `"sherpa"` branch in
  `diarize_session()`, mirroring the `"full"` branch
  (`SherpaDiarizer(model_dir / expected_speakers /
  clustering_threshold)` from config + `_label_segments_from_turns`,
  same `to_dict()` shape and output files). Added by the UI track.
- `requirements.txt`: `sherpa-onnx>=2.0` as an optional line only.
- Tests: `tests/test_diarization_sherpa.py` (20 tests, incl. a gated
  real-model test that downloads models to a tmp dir and skips
  gracefully offline) + `tests/manual_sherpa.py`.

### Track 2 — Modern consumer-facing UI (PyQt6)
- `app/ui/main_window.py` rewritten layout (workers/handlers
  preserved): header with app name, status pill, large recording
  timer; a 340px Record card on the left (devices, live meters,
  prominent red Record button, Pause/Resume/Stop, Refresh, output
  folder); a tabbed work area on the right — **Transcript**
  (click-to-play segments), **Process** (Transcription / Diarization /
  Export & local LLM groups, scrollable), **Speakers** (known speakers,
  suggestions, Confirm/Correct, rename panel), **Near-live**. Fits
  1280×800 (was a 1100×2500 single-column stack).
- `app/ui/theme.py`: modern QSS (cards, tab styling, primary/record/
  danger buttons, splitters, scrollbars) for dark (default) and light.
  `app/ui/level_meter.py` got a bordered track so it reads as a meter
  in the light theme. `app/ui/browser.py` is now side-by-side
  (sessions left, transcript right, word-wrapped rows) with search
  behaviour unchanged (Text/Semantic/Hybrid + scores intact).
- **Theme bug fixed:** `main.py` now calls
  `apply_theme_to_app(app, config.theme)` at the QApplication level,
  so Browser and Settings follow the theme (verified by test +
  screenshots). The main window still applies it to itself for
  standalone use.
- Wiring: diarization picker built from `DIARIZATION_MODES`
  (simple/full/sherpa); selecting sherpa shows
  `check_availability()` state with guided text (install guide /
  first-use download / ready) and disables the HF-token field.
  Device pickers/status use `describe_capture_state()` +
  `mic_empty_message()` / `loopback_empty_message()`; process names
  from `default_process_names()` (Linux shows `teams` first). The old
  "Windows + pyaudiowpatch required" text and the stale `main.py`
  smoke-check note are gone (source-level tests guard this).
- Settings gains a Sherpa group (model folder + chooser, expected
  speakers 0–10 with "0 — auto", clustering threshold 0.05–0.95) plus
  the `sherpa-eres2net` embedding backend; dialog is now scrollable.
- **Nothing removed:** all old-UI controls are reachable by the same
  attribute names — asserted in tests (38 attributes incl. the new
  `sherpa_status_label`). Moved: everything from the single column
  into header/card/tabs; rename panel moved into the Speakers tab
  (where Correct also uses its name field). Removed: nothing.
- Tests: `tests/test_ui_phase6.py` (18 Qt tests in the gui-shots venv
  offscreen + Qt-free layout tests in the main suite).

### Track 3 — Linux + Windows
- `app/platform_info.py` (new): `get_capture_backend()` →
  `"wasapi"` | `"linux-sounddevice"` | `"unavailable"`;
  `describe_capture_state()` → `CaptureStateInfo` (platform, backend,
  sounddevice availability/error, devices, mic-only flag, ready,
  status one-liner, detail, install hint);
  `mic_empty_message()` / `loopback_empty_message()` (correct
  per-platform dropdown placeholders);
  `default_process_names()` (`teams`/`ms-teams` on Linux);
  `is_windows()` / `is_linux()` / `is_macos()`, `platform_name()`.
  Usage example in the module docstring — this is the API the UI
  track consumed.
- `app/audio/linux_capture.py` (new): `LinuxSounddeviceCapture`,
  `LinuxMicCapture`, `LinuxSystemCapture`, `LinuxDualCapture` —
  mic + system audio via sounddevice/PortAudio, PulseAudio/PipeWire
  monitor source auto-selected, mic-only mode (empty `system.wav`,
  `mode="linux-mic-only"`) when no monitor source exists. Same
  session format (`mic.wav`/`system.wav`/`combined.wav`/
  `metadata.json`), same pause-discard and mix behaviour, same
  downstream pipeline.
- `app/audio/capture.py`: `DualCapture` takes pluggable
  `mic_cls`/`system_cls`; `create_dual_capture()` factory (WASAPI on
  Windows, Linux backend on Linux, guided `RuntimeError` when no
  backend exists). WASAPI behaviour untouched otherwise.
- `app/audio/devices.py`: lazy `load_sounddevice()` /
  `sounddevice_status()` and Linux enumeration; `list_microphones()` /
  `list_loopback_devices()` dispatch Windows→pyaudiowpatch,
  Linux→sounddevice, else `[]`. Monitor sources detected by
  "Monitor" in the PortAudio device name.
- `requirements.txt`: `sounddevice>=0.5; sys_platform == "linux"`
  only — runtime-optional via lazy import (the app runs without it).
- `README.md`: new "Install (Linux)" section (install steps incl.
  `libportaudio2`, what works, limits vs Windows, known gaps) +
  a Sherpa-mode section under Phase 3. Windows section verified
  still accurate.
- Tests: `tests/test_linux_capture.py` (15) +
  `tests/test_platform.py` (18), all mocked — no hardware needed.

## 2. How to test it

```bash
cd ~/workspace/offair
.venv-phase2/bin/python -m py_compile $(find main.py app tests -name '*.py' | sort)
.venv-phase2/bin/python -m pytest tests/ -q          # 236 passed, 7 skipped
.venv-phase2/bin/python main.py --no-gui             # exit 0; prints per-platform capture state
~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python -m pytest tests/test_ui_phase6.py -q  # 18 passed (offscreen Qt)
```

- Sherpa, real model (gated test):
  `OFFAIR_TEST_REAL_SHERPA=1 .venv-phase2/bin/python -m pytest tests/test_diarization_sherpa.py -q`
  (needs `pip install sherpa-onnx` in the repo venv; models already
  under `~/.offair/models/sherpa-diarization/`).
- Sherpa, manual: `tests/manual_sherpa.py` on
  `tests/fixtures/meeting-3speakers-16k.wav`.
- Linux backend: mocked tests only here (no PortAudio on this VM) —
  real-desktop manual test still needed (README says so).
- Screenshots (offscreen, seeded demo sessions in
  `/tmp/offair-demo/recordings`): `screenshots/main-window.png`
  (1280×800 dark), `screenshots/main-window-light.png` (light,
  Process tab + sherpa selected), `screenshots/browser-all.png`,
  `screenshots/browser-search-december.png`,
  `screenshots/browser-search-twelve-thousand.png`,
  `screenshots/settings.png`. Each was visually checked: legible, no
  blank windows, no clipped controls except the two cosmetic notes
  in §4.

## 3. Deviations and why

- **Threshold clustering, not `num_clusters=3`.** The tryout
  hardcoded 3 clusters; the shipped code uses threshold clustering
  (default 0.5) with an optional expected-speaker hint, so unknown
  speaker counts work. Measured result is the same 9/9 on the
  fixture.
- **Turn-boundary difference vs the tryout.** Threshold clustering
  merges the tryout's 0.32 s turn overlap differently: turn 3
  (Marcus) runs 14.54–23.03 instead of 14.54–22.41, turn 4 starts at
  23.03, so there is no overlap. Identity is still 9/9.
- **First-guess ERes2Net URL 404'd.** Track 1's first-guess model URL
  (`speaker-embedding-models` tag) was wrong; the shipped code uses
  the verified `speaker-recongition-models` tag (sherpa-onnx's own
  typo). Download path was verified end-to-end into a fresh dir.
- **`main.py` smoke output still lists `ms-teams.exe` first** in the
  "Target app" line on Linux (pre-existing `AppConfig` default) —
  the UI itself uses `default_process_names()` and shows `teams`
  first. Cosmetic only; noted as an open item, not fixed here.
- **Two cosmetic clipping notes** (both reachable, both minor): the
  mic dropdown placeholder truncates ("…PortAudio not ir…") at
  1280×800, and the Settings dialog's bottom field ("Recording
  Browser search mode") sits at the scroll edge; the hint line under
  Summarise sits at the very bottom edge of the Process tab
  (reachable by its scrollbar).
- **Environment fixes, not repo changes:** the gui-shots venv's
  PyQt6 needed `libEGL.so.1` — installed via `apt install libegl1`
  (system package, won't survive a VM rebuild); `pytest` installed
  into that venv to run the Qt tests.

## 4. Measured CPU / RAM (61.3 s fixture, this VM)

| Step | Wall | RTF | Peak RSS |
|---|---|---|---|
| Diarize sherpa, threshold 0.5 | 23.6 s (24.2 s incl. pipeline build) | 0.385 | 274 MB |
| Diarize sherpa, threshold 0.6 | 22.6 s | 0.369 | — |
| Diarize sherpa, expected-speakers=3 | 21.3 s | 0.347 | — |
| Diarize sherpa, clean single run (0.5) | 20.2 s | 0.330 | 274 MB |
| Tryout reference (num_clusters=3) | 21.9 s | 0.358 | 262 MB |

All configs: **9 turns, 3 speakers, 9/9 identities correct**
(best-match cosine 0.85–0.93, cross-scores near zero). Models:
segmentation `model.onnx` 6.0 MB + ERes2Net 26.5 MB (~33 MB
total). Full table in `docs/CPU_TIMES.md`. Extrapolated (labelled
as such): a 30-minute meeting ≈ 10 min of sherpa diarization at
RTF ~0.35; a 60-minute meeting ≈ 20 min.

## 5. Open gates (priority order)

1. **Windows manual test** — real Teams call, whole pipeline, both
   tracks by ear. Still the one gate covering everything.
2. **Linux manual test** — real desktop recording (mic + monitor
   source) on a machine with PortAudio + sounddevice; the hardware
   path here is mocked-tests only.
3. **First real sherpa run through `diarize_session()`** — the
   `SherpaDiarizer` itself ran real (9/9); the session branch was
   verified with injected fakes (sherpa-onnx isn't installed in the
   repo venv). Also re-measure the 0.5 clustering threshold on real
   meeting audio — the fixture is clean TTS with no overlap/noise.
4. First real Ollama summary; Whisper small-vs-base re-check on real
   audio; live-during-recording near-live polling still unwired.
5. Cosmetic: mic placeholder truncation at 1280×800; Settings bottom
   field / Process-tab hint line at scroll edges; `main.py` smoke
   "Target app" order on Linux.

## 6. Visual redesign (Nov 8 2026, after Phase 6 — local, unpushed)

Phase 6 reorganised the layout but kept the same dark-gray panels,
flat buttons and dense type, and the UI still read as "exactly the
same". This pass is the visual layer only — layout structure,
widget attributes and behaviour are unchanged. **Light is now the
default theme** (warm off-white `#F7F6F3` background, white cards,
soft borders, near-black text); dark is a finished equal (`#17171B`
layered grays), switchable in Settings → Theme and applied
app-wide (main window, Browser, Settings). One indigo accent
(`#4F46E5` light / `#6366F1` dark) is used sparingly for primary
actions, active tabs and focus. Type scale: large app title,
semibold small-caps section labels, monospace timer/timestamps.
Spacing follows 8/12/16/24; cards are rounded with 1px borders.
Record is a big red button with a red recording pill + live timer
state; secondary buttons are quiet; level meters are custom-painted
(slim rounded track, green/amber/red fill); empty states carry
helpful copy. Fixed along the way: the capture-status label
clipping at 1280×800 (card spacing 10→8px + Minimum vertical size
policy on the wrapped labels), the Settings bottom field and
Process-tab hint line at the scroll edge, and the theme change
applying at QApplication level. Verified: full suite green
(240 passed, 13 skipped; 28 Qt tests), offscreen screenshots
re-rendered in `screenshots/` (light default + dark). Screenshots
are local-only and not part of any push unless asked. The cosmetic
items in §5 are closed except the `main.py` smoke "Target app"
order on Linux, which is pre-existing and untouched.
