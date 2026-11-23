# OffAir — Phase 3 Build Report

Per the PRD's instructions for the coding agent: what was built, how to
test it, deviations, observed performance, and recommended next-phase
adjustments. **Phase 3 only** — diarization, no persistent speaker
learning (Phase 4), no live transcript, no AI summaries.

## 1. What was built

Speaker Diarization (CPU-only), in `~/workspace/offair/` (local
files only; nothing pushed to GitHub in this step):

- **`app/diarization/simple.py`** — Simple "You vs Remote" mode, no
  token, no model. For each transcript segment it compares mean RMS
  energy in `mic.wav` (you) vs `system.wav` (remote) over the segment
  window: one track ≥1.5× the other → `You` / `Remote`; both below
  the silence threshold → `Silence`; neither dominant → `Mixed`.
  Pure functions over sample arrays, testable without hardware.
  Single-track fallback (`label_segments_single_track`): splits one
  track at silence gaps (≥0.8 s) and alternates `SPEAKER_00` /
  `SPEAKER_01` — explicitly approximate, and the returned method
  string says so (`"single-track-energy (approximate)"`).
- **`app/diarization/pyannote_diarize.py`** — Full multi-speaker mode
  via `pyannote/speaker-diarization-3.1` on CPU, generic
  `SPEAKER_00/01…` labels, optional min/max speaker hints (validated:
  integers 1–10, min ≤ max). pyannote/torch are imported lazily, so
  importing OffAir and running simple mode never need them.
  `resolve_hf_token()` reads `HF_TOKEN` /
  `HUGGING_FACE_HUB_TOKEN` (or a passed token) and otherwise raises
  `HuggingFaceTokenError` whose message is the guided setup: create
  a free HF account, create a read token, accept the licences for
  `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`,
  set `HF_TOKEN`. No token is invented, embedded, or stored, and no
  gated model is downloaded without one. The pipeline is forced to
  CPU (`pipeline.to(torch.device("cpu"))`) — never CUDA.
- **`app/diarization/names.py`** — Speaker Name Panel data: rename a
  label to a name, persisted per recording in `speakers.json`
  (`{"version": 1, "names": {...}}`); a blank name clears the rename;
  `apply_names()` / `display_name()` resolve labels at export time.
  Names never cross sessions — that is Phase 4.
- **`app/diarization/session.py`** — `diarize_session(session_dir,
  mode="simple"|"full")`: loads `transcript.json`, runs the mode,
  writes `diarization.json` (mode, method, per-segment labels with
  mic/system RMS for simple mode), adds a `"speaker"` key to each
  segment in `transcript.json` (other fields preserved), and
  re-exports `transcript.txt` / `transcript.srt` with labels and
  renames applied. `reapply_names()` refreshes the exports after a
  rename without re-running diarization. Full-mode segments with no
  overlapping pyannote turn get `"Unknown"`, never a guessed speaker.
- **Transcription wiring** — `Segment` gained an optional `speaker`
  field (omitted from JSON while `None`, so Phase 2 transcripts are
  byte-identical until diarized). `export_txt` / `export_srt` take an
  optional `speaker_names` mapping and write `[MM:SS] Speaker: text`
  for labelled segments; unlabelled segments keep the Phase 2 format.
- **UI wiring (`app/ui/main_window.py`)** — diarization mode
  dropdown (Simple default, Full with a CPU-slowness warning label),
  HF token field (password echo, never stored), optional min/max
  speaker fields, **Diarize last session** button (enabled only once
  a transcript exists, runs in a `QThread` worker), Speaker Name
  Panel (label dropdown + name field + Rename button), and a
  clickable segment list — clicking a segment plays that part of
  `combined.wav` via `QMediaPlayer` (import-guarded: if QtMultimedia
  is absent the app explains that instead of crashing). Phase 1/2
  code is otherwise untouched.
- **`requirements.txt`** — added `pyannote.audio>=3.1` with a note
  that it pulls the torch CPU wheel (~1 GB installed + ~1 GB of
  cached models) and is only needed for full mode. No CUDA packages.
- **`tests/test_diarization.py`** — 25 new pytest tests (fake
  pipeline + synthetic tone-burst audio, no hardware/network/token):
  dominance/silence/mixed thresholds, dual-channel You/Remote on
  constructed tracks, single-track alternation, rename persistence
  and appearance in exports, hint validation, missing-token guided
  error, fake-pipeline full mode (turn sorting, hint pass-through,
  progress callbacks), `Unknown` for non-overlapping segments,
  session wiring and error paths.
- **`tests/manual_diarize.py`** — real-model manual test (not run by
  pytest): transcribes the fixture with Whisper tiny, builds the
  synthetic dual-channel session, runs simple mode, scores it
  against the script, and checks rename + re-export.

## 2. How to test it

### Automated (this Linux VM — passed, see §4)

```bash
cd ~/workspace/offair
.venv-phase2/bin/python -m py_compile main.py app tests/test_*.py \
  app/transcription/*.py app/diarization/*.py app/ui/main_window.py \
  app/config.py
.venv-phase2/bin/python -m pytest tests/ -q   # 70 passed (45 Phase 1+2 + 25 new)
python3 main.py --no-gui                      # Phase 1 smoke check still OK
```

### Real simple mode on the fixture (this Linux VM — ran, see §4)

```bash
.venv-phase2/bin/python tests/manual_diarize.py
# Transcribes tests/fixtures/meeting-3speakers-16k.wav (61.3 s, 3 TTS
# voices, ground truth tests/fixtures/meeting-script.txt), builds a
# synthetic dual-channel session in ~/tmp/offair-phase3/
# (mic.wav = John's turns, system.wav = Priya/Marcus turns,
# combined.wav = untouched fixture), runs diarize_session(mode="simple"),
# prints per-segment labels vs ground truth + accuracy, and checks that
# renaming You -> John appears in transcript.txt.
```

### Manual (Windows — still required)

1. On Windows with Teams: venv, `pip install -r requirements.txt`,
   `python main.py`. Record a session, stop, transcribe, then pick a
   diarization mode and click **Diarize last session**.
2. Simple mode needs nothing else. For full mode, do the HF token +
   licence setup in README § "Phase 3" first, then expect minutes of
   CPU work for a 30-minute meeting.
3. Rename a speaker in the Name Panel and confirm `transcript.txt` /
   `.srt` update; click a segment and confirm the right audio plays.
4. Verify the labels against the real call by ear — simple mode on
   real Teams audio is the test that matters, and it has not run yet.

## 3. Deviations from the PRD and why

1. **Full pyannote diarization did NOT run on this VM — and was not
   faked.** Two `pip install pyannote.audio` attempts in
   `.venv-phase2` both failed the same way: the torch 2.15.0 CPU
   wheel (539.2 MB) stalls at exactly 261.6 MB downloaded, the
   connection to `files.pythonhosted.org` drops, and resume makes no
   progress (6 attempts, then 11 attempts with `--resume-retries 10`).
   torch and pyannote are therefore not installed here. The full-mode
   code is written, and its fake-pipeline unit tests pass, but no real
   pyannote turn has ever been produced by this build. Full mode
   needs a machine with working PyPI access (or a pre-installed torch)
   plus the HF token and licences. See §4 for the exact failure.
2. **Simple mode is the verified path; full mode is code + tests
   only.** The PRD's Phase 3 success criteria say multi-speaker
   calls receive distinct labels on CPU — that is true of the
   implementation, but only the simple mode's You/Remote split has
   actually run. Do not call Phase 3 "fully verified" until full
   mode runs on a machine where torch installs.
3. **No settings panel (yet).** As in Phase 2, the HF token field,
   mode selector, and speaker hints live on the main window; a
   dedicated settings panel is deferred to Phase 5 polish.
4. **GUI not launched on this VM.** PyQt6 is not installed here. The
   Diarize button / worker / Name Panel / click-to-play wiring is
   code-reviewed and compiles, but clicking it is part of the Windows
   manual test (§2). Segment playback in particular is best-effort:
   it needs QtMultimedia, which some PyQt6 builds omit.
5. **Phase 1 Windows manual test still open** (see PHASE1.md §4 /
   PHASE2.md §0). Simple mode additionally assumes `system.wav` is
   the remote side; today it is system-wide loopback (Phase 1 §3.1),
   so on a real call "Remote" may include other apps' audio. Mute
   other apps during the call, or close the per-app gap first.

## 4. Observed CPU performance and memory usage

**Test machine (this Linux VM, NOT the target laptop):** 4 vCPU (AMD
EPYC), 7.7 GB RAM total, ~700 MB free at run time, no GPU. Audio:
61.3 s of clear TTS dialogue, 3 voices (John/Priya/Marcus), 16 kHz
mono. Ground truth: `tests/fixtures/meeting-script.txt` (9 turns).

### Simple mode — measured, real run

Command: `.venv-phase2/bin/python tests/manual_diarize.py` (exit 0).

| Measurement | Value |
|---|---|
| Whisper tiny transcription (12 segments) | 3.9 s, peak RSS 293 MB (whole process) |
| Simple diarization wall time (61.3 s audio) | **0.01 s** (RTF ≈ 0.0002) |
| Simple-mode accuracy vs script | **12/12 = 100%** |
| Peak RSS during the whole manual run | 293 MB (dominated by Whisper, not diarization) |

Per-segment labels, quoted from the run (`diarization.json`):

```text
[ 0.0– 5.0] You    -> You     mic_rms 0.0569 sys_rms 0.0027
[ 5.0–10.4] Remote -> Remote  mic_rms 0.0002 sys_rms 0.0482
[10.4–14.4] Remote -> Remote  mic_rms 0.0000 sys_rms 0.0406
[14.4–19.3] Remote -> Remote  mic_rms 0.0000 sys_rms 0.0689
[19.3–24.8] Remote -> Remote  mic_rms 0.0066 sys_rms 0.0424
[24.8–29.7] You    -> You     mic_rms 0.0409 sys_rms 0.0074
[29.7–35.0] Remote -> Remote  mic_rms 0.0100 sys_rms 0.0493
[35.0–40.0] Remote -> Remote  mic_rms 0.0000 sys_rms 0.0602
[40.0–46.0] Remote -> Remote  mic_rms 0.0000 sys_rms 0.0620
[46.0–51.0] You    -> You     mic_rms 0.0420 sys_rms 0.0106
[51.1–56.0] Remote -> Remote  mic_rms 0.0000 sys_rms 0.0758
[56.0–61.0] Remote -> Remote  mic_rms 0.0000 sys_rms 0.0577
```

Rename check passed: after `rename_speaker(session, "You", "John")`
+ `reapply_names(session)`, `transcript.txt` starts
`[00:00] John: Good morning everyone. Thanks for joining the
OffAir review. …` and `speakers.json` holds `{"You": "John"}`.

**What the 100% does and does not prove — read this before quoting
it.** The synthetic session's mic/system masks were built from the
same Whisper segment windows used for the ground-truth assignment, so
the energy comparison is tested on cleanly separated tracks with
known boundaries. It validates the labelling logic (RMS comparison,
dominance ratio, silence handling) and the session wiring — it does
not validate turn-boundary detection, and it says nothing about real
Teams audio, where tracks leak into each other, speakers overlap,
and Whisper segments straddle turn changes (one segment here,
`[4.96–10.4]`, already contains the tail of John's turn plus Priya's
opening; the ground truth calls it Remote by word overlap, and the
energy agrees only because the mask followed the same boundary).
On a real call, expect Mixed/Silence labels to appear and a few
segments near turn changes to be wrong. The honest claim is: simple
mode is fast (0.01 s for a minute of audio — it will be seconds at
most for an hour) and correct when the two tracks are actually
separate; whether they are separate on a real Teams call is the
Windows manual test.

**Extrapolated (not measured):** simple-mode cost is linear in audio
length and dominated by reading the WAVs — a 60-minute meeting
should diarize in well under a second of compute on any laptop.
Full-mode time is unmeasured here; the PRD's "minutes for 30
minutes" expectation stands, and the UI warns accordingly.

### Full (pyannote) mode — did NOT run here

Exact failure, twice, in `.venv-phase2`:

```text
$ .venv-phase2/bin/pip install pyannote.audio
Download failed after 6 attempts because not enough bytes were
received (261.6 MB/539.2 MB)
URL: .../torch-2.15.0-cp312-cp312-manylinux_2_28_x86_64.whl
note: This is an issue with network connectivity, not pip.
$ .venv-phase2/bin/pip install --resume-retries 10 pyannote.audio
Download failed after 11 attempts ... (261.6 MB/539.2 MB)  # same byte count
$ .venv-phase2/bin/python -c "import torch"       # ModuleNotFoundError
$ .venv-phase2/bin/python -c "import pyannote.audio"  # ModuleNotFoundError
```

The stall is at the same byte count on every attempt, i.e. the
connection drops and resume never advances — not a transient blip
and not an out-of-memory kill (free RAM was ~700 MB throughout, and
pip never got far enough to install anything). No model was
downloaded, no token was used, and no pyannote code path executed
outside the fake-pipeline unit tests. Two further cautions for the
machine that does run it: torch 2.15.0 alone is a 539 MB wheel, and
the pyannote pipeline loads segmentation + embedding models that
need ~1–2 GB of RAM — this VM's ~700 MB free would likely OOM even
if the download had succeeded, so full mode should first be run on
the 8–16 GB target laptop, not here.

Automated test results on this VM:

```text
.venv-phase2/bin/python -m py_compile <all .py files> → passed (exit 0)
.venv-phase2/bin/python -m pytest tests/ -q            → 70 passed in 0.94s
.venv-phase2/bin/python tests/manual_diarize.py        → exit 0, 12/12 simple-mode
python3 main.py --no-gui                               → exit 0 (smoke check)
```

## 5. Recommended next-phase adjustments

- **Run full mode on a machine where torch installs, before
  leaning on it.** That is the single biggest open item: install
  `pyannote.audio` on the target laptop (or any machine with
  reliable PyPI access), do the HF token + licence setup, and run
  the 3-speaker fixture through full mode. Expect distinct
  `SPEAKER_00/01/02` labels for John/Priya/Marcus; if the pipeline
  merges or splits a speaker, tune min/max hints before anything
  else. Do not start Phase 4's embedding matching until full-mode
  turns are known-good — Phase 4 consumes those turns.
- **Run simple mode on a real Teams call.** The Windows manual test
  (record → transcribe → diarize simple) is the only way to learn
  how much crosstalk the loopback track carries and how often
  `Mixed` appears. If `Mixed` is common, the next fix is per-app
  capture (Phase 1 §3.1), not a better label rule.
- **Consider a turn-splitting pass for simple mode.** Whisper
  segments straddle speaker changes; a segment containing two
  speakers can only get one label. A cheap improvement (before or
  during Phase 4) is to split segments at the point where the
  mic/system dominance flips inside the segment, using the same
  frame RMS already computed.
- **Re-check the Whisper `small` default on real audio** (carried
  from PHASE2.md §5) — it still stands, and it interacts with Phase
  3: segment boundaries move with the model, so diarization accuracy
  on real calls should be compared across tiny/base/small, not
  assumed model-independent.
- **No changes needed to the session format for Phase 4.**
  `diarization.json` already carries per-segment labels and RMS;
  `speakers.json` already carries renames. Phase 4 can add
  `speakers.db` (SQLite + embeddings) alongside them without
  restructuring, and can reuse the full-mode turns as its clusters.

## Short TODO list (not in core paths)

- [ ] Windows manual test — record a real Teams call, transcribe,
  diarize in simple mode, verify labels by ear (§2).
- [ ] Full pyannote run on a machine with working PyPI + HF token.
- [ ] True per-app capture investigation (see PHASE1.md §3.1).
- [ ] Segment splitting at intra-segment dominance flips.
