# OffAir — Phase 2 Build Report

Per the PRD's instructions for the coding agent: what was built, how to
test it, deviations, observed performance, and recommended next-phase
adjustments. **Phase 2 only** — transcription, no diarization, no speaker
learning, no live transcript, no AI summaries.

## 0. Windows VM verdict (checked before this phase, not retried)

No usable Windows VM exists on this host: 4 vCPU, no `vmx`/`svm` CPU
flags, no `/dev/kvm`, no qemu installed, 7.7 GB RAM total with only
~650 MB free, no Windows ISO/license. Transcription was therefore
tested on this Linux VM with generated TTS voice samples. That is valid
for Phase 2 — the transcription code is cross-platform Python; only the
Phase 1 WASAPI capture is Windows-only. **The Phase 1 Windows manual
test (record a real Teams call, verify both tracks by ear) is still
open**, and transcription has not yet been run on real Teams audio.

## 1. What was built

Transcription Engine (CPU-only), in `~/workspace/offair/` (local
files only; nothing pushed to GitHub in this step):

- **`app/transcription/engine.py`** — `TranscriptionEngine`:
  faster-whisper, forced `device="cpu"`, `compute_type="int8"` (any
  other device raises `ValueError`). Lazy model loading, `unload()`
  to free memory. `transcribe_file()` returns timestamped `Segment`s
  (`start`, `end`, `text`) and calls a progress callback with
  `(fraction, elapsed_seconds, estimated_remaining_seconds)`.
  `transcribe_session()` reads `combined.wav` (fallback `mic.wav`)
  from a Phase 1 session folder and writes `transcript.json`
  (segments + model/device/wall time/RTF), `transcript.txt` and
  `transcript.srt` next to the WAVs. Supported models: tiny, base,
  small, medium; default `small` per PRD.
- **`app/transcription/exports.py`** — pure functions: `export_txt`
  (`[MM:SS] text` per segment), `export_srt` (standard SubRip,
  `HH:MM:SS,mmm`), JSON segment round-trip, `load_transcript_json`.
- **`app/transcription/__init__.py`** — public exports.
- **`app/config.py`** — `WHISPER_MODELS` table (params, size, relative
  speed/accuracy, expected 30-min time), `WHISPER_DEFAULT_MODEL =
  "small"`, `AppConfig.whisper_model` (validated). Expected times are
  extrapolations from the measured RTFs (§4), labelled as such;
  medium is still an unmeasured estimate.
- **UI wiring (`app/ui/main_window.py`)** — model dropdown, "Transcribe
  last session" button (enabled only after a session is stopped),
  progress bar + status text with elapsed / estimated-remaining, work
  done in a `QThread` worker so the UI never freezes. Phase 1 capture
  code untouched except this addition. PyQt6 is not installed on this
  VM, so the GUI itself was not launched here — the wiring is verified
  by `py_compile` and the import-guard behaviour, not by clicking it.
- **`requirements.txt`** — added `faster-whisper>=1.1` plus a PyAV pin
  `av>=11,<15` (see §3.1 — without the pin, transcription is broken).
  No pyannote (Phase 3), no torch/CUDA.
- **`tests/test_transcription.py`** — 15 new pytest tests with a fake
  Whisper model: segment JSON round-trip, TXT/SRT formatting incl.
  timestamps, progress callback invoked and ending at 1.0, missing file
  raises, model-size and GPU-device validation, session write +
  mic fallback, config model table completeness.
- **`tests/manual_transcribe.py`** — real-model manual test (not run by
  pytest): runs tiny/base/small on an audio file, prints wall time,
  RTF, peak RSS, segments, and a computed Levenshtein WER against a
  ground-truth script; writes outputs to `/tmp`, never into the repo.

## 2. How to test it

### Automated (this Linux VM — passed, see §4)

```bash
cd ~/workspace/offair
.venv-phase2/bin/python -m py_compile main.py app tests/test_*.py \
  app/transcription/*.py app/ui/main_window.py app/config.py
.venv-phase2/bin/python -m pytest tests/ -q   # 45 passed (30 Phase 1 + 15 new)
python3 main.py --no-gui                      # Phase 1 smoke check still OK
```

### Real model (this Linux VM — ran, see §4)

```bash
# Fixture: tests/fixtures/meeting-3speakers.mp3 (61.3 s, 3 TTS voices)
# converted first: ffmpeg -y -i tests/fixtures/meeting-3speakers.mp3 \
#   -ar 16000 -ac 1 tests/fixtures/meeting-3speakers-16k.wav
.venv-phase2/bin/python tests/manual_transcribe.py --all
# Single model: ... --model base
```

### End-to-end session flow (real tiny model, this VM — ran)

A temp session folder with `combined.wav` = the fixture WAV was
transcribed via `TranscriptionEngine.transcribe_session()`: 12 segments
in 3.9 s (RTF 0.064), `transcript.json` / `.txt` / `.srt` all written
and well-formed.

### Manual (Windows — still required)

1. On Windows with Teams: venv, `pip install -r requirements.txt`,
   `python main.py`. First transcription downloads the model from
   Hugging Face (~75–466 MB); after that it is fully offline.
2. Record a session, stop, pick a model, click Transcribe, confirm the
   progress bar moves and the transcript files appear in the session
   folder with timestamps roughly matching speech.

## 3. Deviations from the PRD and why

1. **PyAV pin (`av>=11,<15`) — required, not optional.** faster-whisper
   1.2.1 declares only `av>=11`, so pip installed PyAV 19.0.2 — and
   every `transcribe()` call then failed with
   `TypeError: open() got an unexpected keyword argument
   'metadata_errors'`, because faster-whisper calls
   `av.open(..., metadata_errors="ignore")` and PyAV removed that kwarg
   in v15. Pinning to av 13.1.0 (last line with a prebuilt wheel here;
   14.4.0 tried to build from source and failed on this VM) fixes it.
   This is an upstream version-skew bug, and the pin is documented in
   `requirements.txt`. Revisit it when faster-whisper ships a fix.
2. **Model download needs network once.** The PRD's "zero network calls"
   applies to core processing; the Whisper weights must still be
   downloaded on first use (Hugging Face, unauthenticated, ~75–466 MB).
   After that, transcription is fully offline. Documented in README.
3. **Medium was not measured.** It is 1.5 GB and too large for this
   VM's ~650 MB free RAM. Its table entry stays an estimate. It is an
   opt-in only in the UI; nobody should need it on CPU.
4. **GUI not launched on this VM.** PyQt6 is not installed here and the
   transcription run did not need it. The Transcribe button / progress
   bar / QThread wiring is code-reviewed and compiles, but clicking it
   is part of the Windows manual test (§2).
5. **No settings panel (yet).** The PRD asks for a settings panel
   showing relative speed expectations per model. The model dropdown
   shows speed + size per model, and the full table with measured times
   is in README §"Phase 2" and `app/config.py`. A dedicated settings
   panel is deferred to Phase 5 polish rather than built twice.
6. **Phase 1 Windows manual test still open** (see §0). Phase 2 does not
   depend on it — transcription reads whatever WAVs exist — but the
   overall "record → transcribe" path on real Teams audio is unverified
   until that test passes.

## 4. Observed CPU performance and memory usage

**Test machine (this Linux VM, NOT the target laptop):** 4 vCPU (AMD
EPYC), 7.7 GB RAM total, ~650 MB free at run time, no GPU. Audio:
61.3 s of clear TTS dialogue, 3 voices (John/Priya/Marcus), 16 kHz
mono WAV converted from the fixture MP3. Ground truth:
`tests/fixtures/meeting-script.txt` (199 reference words). WER is a
computed Levenshtein word-error rate (substitutions + deletions +
insertions / reference words), via `tests/manual_transcribe.py`.
faster-whisper 1.2.1, ctranslate2, int8, beam_size 5, `language="en"`.
Command: `.venv-phase2/bin/python tests/manual_transcribe.py --all`
(exit 0).

| Model | Wall (61.3 s) | Real-time factor | Peak RSS* | Segments | WER |
|-------|---------------|------------------|-----------|----------|-----|
| tiny  | 4.3 s  | 0.070 | 294 MB | 12 | 8.0% (11 sub, 4 del, 1 ins) |
| base  | 8.3 s  | 0.135 | 392 MB | 18 | 7.5% (12 sub, 2 del, 1 ins) |
| small | 20.4 s | 0.333 | 943 MB | 17 | 7.5% (11 sub, 2 del, 2 ins) |

\* Peak RSS is `ru_maxrss` for the whole `--all` process (models loaded
in sequence; freed memory is not always returned to the OS), so small's
943 MB overstates small alone. Even so, small needed swap on this
memory-starved VM and did not OOM. A second standalone tiny run on a
session folder took 3.9 s (RTF 0.064), consistent with the `--all` run.

**Extrapolated (extrapolation, not measurement):** 30-min meeting ≈
30 min × RTF → tiny ~2 min, base ~4 min, small ~10 min. 60-min meeting
→ tiny ~4 min, base ~8 min, small ~20 min. The target laptop (8–16 GB
RAM, modern laptop CPU) should beat these; this VM is a lower bound.

**Accuracy, qualitative (do not over-read the WER):** the WERs are
close because the fixture is clean, single-channel TTS with no overlap,
no crosstalk, and no room noise. Most counted "errors" are formatting,
not lost content: number words ("seventh" → "7th"),
"markdown" → "marked down" (tiny, small), "Teams" → "team's". No
sentence was dropped by any model; every agenda item, date, amount
($12,000, December 15th), and action item survived in all three
transcripts. Timestamps tracked speech closely (e.g. base segment
`[44.2–46.0] "Does that still work for everyone?"` matches the script
order). One minor observation: small's final segment ends at 62.9 s
against 61.3 s of audio — a ~1.6 s end overshoot, harmless for
transcripts, worth watching if segment playback is added in Phase 3.

**The notable finding:** on this clean fixture, **base matched small's
accuracy (both 7.5% WER) at 2.5× less wall time and ~2.4× less RAM**.
Small is the PRD default and stays the default in code, but the
default deserves a re-check on real, noisy Teams audio, where the gap
may widen — or may not. See §5.

Automated test results on this VM:

```text
.venv-phase2/bin/python -m py_compile <all .py files> → passed (exit 0)
.venv-phase2/bin/python -m pytest tests/ -q            → 45 passed in 0.97s
.venv-phase2/bin/python tests/manual_transcribe.py --all → exit 0
```

## 5. Recommended next-phase adjustments

- **Re-check the `small` default on real meeting audio before locking
  it in.** On clean TTS, base ≈ small. Run the same 10-minute real
  Teams recording through tiny/base/small on the actual HP laptop in
  Phase 2 follow-up / Phase 3 and be willing to default to `base`
  with `small` as an opt-in — it would roughly halve transcription
  time and memory for most meetings.
- **Close the Phase 1 per-app gap before Phase 3 leans on it.** Phase 3's
  "You vs Remote" mode assumes `system.wav` is the remote side; today
  it is system-wide loopback (see PHASE1.md §3.1). Either implement
  true Win11 per-session capture or formally accept loopback + "mute
  other apps" as the workflow.
- **Phase 3 will need a Hugging Face token** for the full pyannote
  diarization models — plan the guided setup the PRD asks for, and keep
  the no-token "You vs Remote" mode as the default path.
- **Keep the av pin in mind.** If faster-whisper or PyAV releases a
  fix, the `av>=11,<15` pin can be relaxed — but re-run
  `manual_transcribe.py` after any bump; this exact failure mode
  (TypeError on `transcribe()`) is silent until the first real run.
- **Medium stays opt-in only.** Do not surface it as a recommended
  choice on 8 GB machines.
- **No changes needed to the session format for Phase 3.**
  `transcript.json` sits alongside the WAVs; diarization can add
  speaker labels to the same segment list without restructuring.
