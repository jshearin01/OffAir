# OffAir — Phase 1 Build Report

Per the PRD's instructions for the coding agent: what was built, how to
test it, deviations, observed performance, and recommended next-phase
adjustments. **Phase 1 only** — no transcription, diarization, speaker
learning, live transcript, or AI summaries were implemented.

## 1. What was built

Audio Capture & Recording Foundation, in `~/workspace/offair/` (local
files only; nothing pushed to GitHub in this step):

- **`main.py`** — entry point. `--output-dir` overrides the recording
  root, `--no-gui` runs a smoke check (config + device list, no window),
  `--list-devices` lists devices and exits. Exits with a clear error if
  PyQt6 is missing and a GUI was requested.
- **`app/config.py`** — defaults: 48,000 Hz, 16-bit PCM mono, chunk
  1024 (2048 alternate constant), output root `~/OffAir/recordings`
  (overridable), Teams process names
  `ms-teams.exe`, `Teams.exe`, `ms-teamsupdate.exe`.
- **`app/audio/devices.py`** — `list_microphones()`,
  `list_loopback_devices()`, `find_teams_process()` (psutil, else None),
  `default_microphone()`, `default_loopback()`. On non-Windows or when
  `pyaudiowpatch` is missing, every function returns `[]` / `None` and
  never raises — the UI shows "no devices found" instead of crashing.
- **`app/audio/capture.py`** — `WASAPICapture` thread class with
  start/pause/resume/stop, per-chunk RMS level callback, incremental WAV
  writing (soundfile, stdlib `wave` fallback) so a crash loses at most
  the in-flight chunk. `MicCapture` and `SystemCapture` share the base;
  `PerAppCapture` targets a process name (see §3); `DualCapture` runs
  mic + system in lock-step and mixes to `combined.wav` on stop.
- **`app/audio/mixer.py`** — pure, hardware-free functions:
  `mix_arrays()` (0.5 gain per channel, hard clip, zero-pad unequal
  lengths), `mix_files()` (normalises sample rate/channels to 48 kHz
  mono int16), `rms_level()` / `peak_level()` returning 0..1 floats.
- **`app/storage/sessions.py`** — `create_session()` makes
  `YYYY-MM-DD_HH-MM-SS_<slug>/` with `mic.wav`, `system.wav`,
  `combined.wav`, `metadata.json`; metadata is written at creation and
  finalised on stop (start/end time, duration, devices, mode, warning).
  `list_sessions()` / `load_metadata()` included.
- **`app/ui/level_meter.py`** — Qt level meter (green/amber/red),
  import-guarded so the module loads without PyQt6.
- **`app/ui/main_window.py`** — PyQt6 window: mic dropdown, loopback
  dropdown, process dropdown (default Teams), mode label, Record /
  Pause / Resume / Stop with correct enable/disable states, two live
  level meters (via a thread-safe signal bridge — capture threads never
  touch Qt directly), elapsed timer, status bar, session folder + files
  shown after Stop, error dialog on capture failure.
- **`app/logging_setup.py`** — console + `offair.log` in the output root.
- **`requirements.txt` / `requirements-dev.txt` / `.gitignore`** —
  PyQt6, pyaudiowpatch (Windows-only marker), soundfile, numpy, psutil;
  pytest for dev. `.gitignore` covers `recordings/`, `*.wav`, `.venv/`,
  `__pycache__/` — PRD.md and README.md are not ignored.
- **`tests/`** — 4 pytest files, fake audio sources only, no hardware,
  no network (see §2).

## 2. How to test it

### Automated (run on this Linux VM — passed, see §4)

```bash
cd ~/workspace/offair
python3 -m py_compile main.py app/config.py app/logging_setup.py \
  app/audio/devices.py app/audio/capture.py app/audio/mixer.py \
  app/storage/sessions.py app/ui/level_meter.py app/ui/main_window.py \
  tests/test_mixer.py tests/test_storage.py tests/test_devices.py \
  tests/test_capture.py
python3 -m pytest tests/ -q
python3 main.py --no-gui
```

Test coverage:

- `tests/test_mixer.py` — RMS/peak (silence = 0, full-scale square = 1,
  sine = A/√2), mixing clipping (two full-scale channels stay at 32767,
  no wrap-around), unequal lengths (zero-pad, correct tail values),
  `mix_files` round-trip and sample-rate/channel normalisation, missing
  input raises `FileNotFoundError`.
- `tests/test_storage.py` — session folder naming, metadata round-trip
  and update, finalise (duration rounding, `files_present`), collision
  suffix, `list_sessions` newest-first and non-session folders excluded,
  missing metadata raises.
- `tests/test_devices.py` — enumeration degrades to `[]` / `None` on
  non-Windows, `find_teams_process` never raises (including with psutil
  blocked), display names.
- `tests/test_capture.py` — full state machine (start → pause → resume →
  stop, illegal transitions raise), pause discards audio (writer frame
  count does not move while paused), stop-from-idle yields a valid empty
  WAV, no-source capture fails cleanly on non-Windows, `PerAppCapture`
  reports fallback mode, `DualCapture` produces all three files with
  matching lengths.

### Manual (Windows 10/11 — still required, see §4)

1. On a Windows machine with Teams: create a venv, install
   `requirements.txt`, run `python main.py`.
2. Select Teams + mic, record 2–3 minutes of a Teams call (PRD: 5 min
   for the full test), confirm both level meters move.
3. Stop and inspect the session folder: `mic.wav`, `system.wav`,
   `combined.wav`, `metadata.json` all present; mic track = your voice
   only; system track = remote participants (plus any other app audio —
   see §3); `metadata.json` mode is `system-loopback-fallback`.

## 3. Deviations from the PRD and why

1. **Per-app capture is a documented fallback, not true isolation.**
   The PRD asks for Windows 11 per-app capture targeting Teams, with
   system-wide loopback as the Windows 10 fallback. `pyaudiowpatch`
   exposes only system-wide WASAPI loopback — there is no API in it to
   capture a single process. Rather than fake isolation, `PerAppCapture`
   finds the Teams process, records system loopback, and records
   `mode="system-loopback-fallback"` + a warning in `metadata.json` and
   the UI. The honest fix (Windows 11 audio-session enumeration via
   `IAudioSessionManager2` / per-session capture, or a different capture
   library) is real work, not a config tweak — flagged as the first
   follow-up below. On Windows 10 the behaviour is identical, so the
   "fallback" is currently the only mode on both OS versions.
2. **Pause discards audio (documented choice).** The PRD does not specify
   pause semantics. This build drops chunks received while paused, so
   mic and system tracks stay aligned with each other and with
   wall-clock time (a gap, not a time-shift). The policy is recorded in
   metadata as `pause_policy: "discard"`. Buffering instead would
   desynchronise the two tracks.
3. **Live WASAPI stream open is implemented but untested on Windows.**
   `_run_hardware()` opens the pyaudiowpatch stream (mic: input stream;
   system: loopback stream via the same `input_device_index` mechanism
   pyaudiowpatch documents). It could not be exercised here — no Windows
   machine and no audio hardware on this VM — so the Windows manual test
   in §2 is a genuine gate, not a formality. If the loopback stream
   needs `get_loopback_device_info_generator()` device handling beyond a
   plain `input_device_index`, that will surface in that test.
4. **`soundfile` is used for WAV I/O, with a stdlib `wave` fallback.**
   The PRD lists "soundfile / pydub" — soundfile alone covers Phase 1
   (PCM WAV read/write); pydub (and its ffmpeg dependency) was not
   needed and was left out to keep the install light.
5. **No `app/transcription/`, `app/diarization/`, `app/speakers/` or
   `app/exports/` packages yet.** The PRD's suggested tree includes them,
   but they belong to Phases 2–5; creating empty placeholder packages
   would imply work that does not exist. They should be added when their
   phase starts.

## 4. Observed CPU performance and memory usage

**Honest statement: no Windows CPU/memory measurements exist for this
build.** It was built and tested on a Linux VM (Python 3.12, no audio
hardware, no GPU) — `pytest` and the `--no-gui` smoke check ran there,
but no real recording happened, so there are no timings to report and
none are invented here.

What can be said from the design:

- Phase 1 does almost no computation: it copies int16 PCM chunks to disk
  and, on stop, reads two WAVs and mixes them. Expected footprint on the
  target CPU-only HP laptop (8–16 GB RAM) is a few tens of MB and
  negligible CPU — orders of magnitude below the Whisper/pyannote work
  in later phases. A 60-minute stereo-equivalent recording at 48 kHz
  mono int16 is ~345 MB per track (~690 MB for mic + system, ~1 GB with
  the combined mix), so disk space, not CPU or RAM, is the Phase 1
  resource to watch.
- **Expected guidance only (not measured):** recording itself should be
  real-time with no backlog on any modern laptop CPU; the stop-time mix
  of a 60-minute call should take seconds. These numbers must be
  confirmed in the Windows manual test — record a 5-minute call and
  note wall-clock time for stop/mix and peak RAM (Task Manager) in the
  test notes before calling Phase 1 done.

Automated test results on this VM (exact commands in §2):

```text
python3 -m py_compile <all .py files>   → passed (no output, exit 0)
python3 -m pytest tests/ -q             → 30 passed in 0.69s
python3 main.py --no-gui                → exit 0; config printed,
  0 microphones, 0 loopback devices, Teams not running (expected on Linux)
```

Note: `pytest`, `soundfile`, and `psutil` were not pre-installed on this
VM's system Python; they were installed with
`pip install --user --break-system-packages pytest soundfile psutil`
for the test run. On Windows they come from `requirements.txt` inside
the venv, so no action is needed there.

## 5. Recommended next-phase adjustments

- **Before Phase 2, close the per-app gap (or re-scope it).** Either
  implement true Windows 11 per-session capture, or formally accept
  system-wide loopback as the Phase 1 behaviour and document "mute other
  apps" as the user workflow. Do not let Phase 2's diarization inherit
  an unacknowledged mixed system track — "You vs Remote" separation in
  Phase 3 depends on knowing what is actually in `system.wav`.
- **Run the Windows manual test and record the numbers.** 5-minute
  Teams call, both tracks verified by ear, stop/mix time and peak RAM
  noted. Until that passes, Phase 1 success criteria are only
  partially met (code + unit tests done, hardware verification open).
- **Phase 2 model default deserves a re-check on the real laptop.**
  The PRD defaults to Whisper `small` on CPU; on an 8 GB laptop that
  may be slow enough to frustrate. Measure `tiny` vs `base` vs `small`
  on a 10-minute recording on the actual HP hardware in Phase 2 and be
  willing to default to `base` with `small` as an opt-in.
- **Disk-space guard for long calls.** At ~5.7 MB/minute per track,
  multi-hour sessions add up. Phase 2 (or a small Phase 1 follow-up)
  should show the session folder size live and warn before disk fills.
- **No changes needed to the session format for Phase 2.** `metadata.json`
  already carries mode, devices, and duration; Phase 2 can add
  `transcript.json` / `transcript.txt` alongside the WAVs without
  restructuring.

## Short TODO list (not in core paths)

- [ ] Windows manual test (§2) — the one open gate for Phase 1.
- [ ] True per-app capture investigation (see §3.1).
- [ ] Live disk-usage display during recording.
