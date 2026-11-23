# OffAir — Phase 5 Build Report + Whole-Project Summary

Per the PRD's instructions for the coding agent: what was built, how
to test it, deviations, observed performance, and recommended
follow-ups. **Phase 5 only** in §1–§5; §6 is the whole-project
summary across Phases 1–5.

## 0. What was already true when Phase 5 started

- Phases 1–4 built and pushed (99 pytest tests passing).
- No Hugging Face account/token exists; pyannote/torch did NOT
  install/run on this VM (torch 2.15 CPU wheel stalls at 261.6 MB
  from PyPI; ~700 MB free RAM would likely OOM a 1–2 GB pipeline
  anyway). Per the task instructions, torch install was **not**
  retried and no pyannote run was faked. Phase 5 therefore works
  with what is verified here: Phase 1 capture (Windows-only,
  manual test still open), Phase 2 faster-whisper CPU, Phase 3
  simple diarization, Phase 4 fallback speaker learning.
- No Windows VM exists on this host (no vmx/svm, no /dev/kvm, no
  qemu, no ISO/license) — not retried. Ollama is not running here
  and llama-cpp-python is not installed here; both are probed, not
  faked (§4).
- Test audio: `tests/fixtures/meeting-3speakers.mp3` (61.3 s,
  John/Priya/Marcus) + isolated `speaker-*.mp3` samples, local
  only, never committed.

## 1. What was built

Polish, Export & LLM Handoff (PRD Phase 5), in
`~/workspace/offair/` (local files only; nothing pushed to
GitHub in this step):

- **`app/exports/llm_export.py`** — `export_llm_markdown()` /
  `export_llm_markdown_from_session()` write `transcript-llm.md`:
  a "Context for summariser" block (date, duration, participants,
  transcription model, diarization method, capture mode), a
  participants list, and the timestamped speaker-labelled
  transcript (`[00:00] **John:** …`). **No invented action
  items**: the "Action items" section appears only when the
  caller passes an explicit list; with none passed it is omitted
  entirely. The TXT export from Phase 2 is reused unchanged.
- **`app/exports/local_llm.py`** — optional local LLM summary,
  OFF by default, never a cloud API. `OllamaClient` (stdlib
  `urllib` only, no `requests` dependency) talks to
  `http://localhost:11434`; `is_available()` probes it, and
  summarising when it is not running raises
  `OllamaNotAvailableError` with install/start instructions.
  Any non-localhost base URL is rejected in the constructor.
  `LlamaCppClient` is lazy/optional (imports `llama_cpp` only on
  first use, needs a local GGUF file path — nothing is ever
  downloaded by this code). `summarize_locally()` is the single
  entry point and refuses to run when disabled.
- **`app/storage/browser.py`** — Recording Browser data layer:
  `summarize_session()` / `list_session_summaries()` (newest
  first, with date, duration, speakers, transcript/diarized/
  LLM-export/suggestions flags) and `search_sessions()`
  (case-insensitive substring over transcript text, speaker
  names, session id, app name; blank query returns all). Reads
  the Phase 1 session format as-is — no restructuring.
- **`app/config.py`** — settings persistence: new `AppConfig`
  fields (`diarization_mode`, `hf_token_configured` (bool only),
  `near_live_enabled`, `nearlive_chunk_seconds`,
  `theme` (default `"dark"`), `ollama_enabled`,
  `ollama_model`, `ollama_base_url`), plus `save_settings()` /
  `load_settings()` / `config_from_settings()` for
  `~/.offair/settings.json`. Only an allowlisted key set is
  ever written — **a token is never stored** (see §3.5).
  Unknown keys are ignored; any invalid value falls back to its
  default, so a corrupt/partial settings file can never break
  startup.
- **`app/ui/theme.py`** — dark (default, per PRD) and light QSS
  stylesheets, `apply_theme()` / `stylesheet_for()`; pure text,
  testable without Qt.
- **`app/ui/settings_dialog.py`** — Settings panel dialog
  (output folder, Whisper model, diarization mode, HF-token-
  configured checkbox — display only, never a token field that
  stores — near-live on/off + chunk length, theme, local-LLM
  options). `edit_settings()` returns the saved config or None
  on cancel; constructing it without PyQt6 raises a clear error.
- **`app/ui/browser.py`** — Recording Browser window: session
  list, live search box, click a session to read its transcript,
  export the selected session as LLM Markdown. Import-guarded.
- **`app/ui/crash_dialog.py`** + **`app/logging_setup.py`** —
  log rotation (`RotatingFileHandler`, ~5 MB per file, one
  backup) and `install_exception_hook()`: uncaught exceptions
  are logged at CRITICAL with a full traceback and, when a Qt
  application is running, shown in a friendly dialog with the
  log file path and a "Copy error" button. No telemetry, no
  network. `main.py` installs the hook on startup and loads
  persisted settings (`--output-dir` still wins for one run).
- **UI wiring (`app/ui/main_window.py`)** — theme applied on
  startup; Settings / Recording Browser / "Export last session
  as LLM Markdown" / "Summarise last session (local LLM)"
  buttons; the summarise worker runs off the UI thread and
  explains when local LLM is off or Ollama is not running
  instead of failing obscurely. Two Phase 4 follow-ups closed
  here: suggestions now show the embedding **backend**
  (fallback vs pyannote), and the status line says out loud
  that **Remote is not suggested** in simple mode instead of
  silently returning only `You`.
- **`docs/CPU_TIMES.md`** — the measured + extrapolated CPU
  times table for all phases (see §4).
- **Tests** — `tests/test_exports.py` (Markdown export incl.
  no-invented-action-items, Ollama fake-HTTP client,
  localhost-only validation, llama-cpp unavailable path),
  `tests/test_browser.py` (synthetic sessions, search),
  `tests/test_settings.py` (round-trip, corrupt/partial files,
  token-never-stored), `tests/test_ui_phase5.py` (theme QSS,
  crash report/dialog guard, log rotation, exception hook,
  UI import guards), `tests/test_e2e.py` (full pipeline with
  fake models), `tests/manual_e2e.py` (the real-model run, §2).
- **README.md** — full end-to-end section, Phase 5 docs (LLM
  export, Ollama/llama-cpp setup, browser, settings, dark
  theme, crash dialog), the honest known-limitations list, and
  the updated project layout.

## 2. How to test it

### Automated (this Linux VM — passed, see §4)

```bash
cd ~/workspace/offair
.venv-phase2/bin/python -m py_compile main.py app tests/test_*.py \
  app/transcription/*.py app/diarization/*.py app/speakers/*.py \
  app/nearlive/*.py app/exports/*.py app/storage/*.py app/ui/*.py \
  app/config.py app/logging_setup.py
.venv-phase2/bin/python -m pytest tests/ -q   # 140 passed (99 Phase 1-4 + 41 new)
python3 main.py --no-gui                      # smoke check still OK
```

### Real end-to-end (this Linux VM — ran, see §4)

```bash
.venv-phase2/bin/python tests/manual_e2e.py
# transcribe (tiny) -> diarize simple -> rename You -> John ->
# seed DB from isolated samples -> suggest -> learn ->
# LLM Markdown export -> local-LLM probe (Ollama, expected NOT
# running here). Outputs in ~/tmp/offair-phase5 — never in the repo.
```

### Manual (Windows — still required)

1. On Windows with Teams: venv, `pip install -r requirements.txt`,
   `python main.py`. Record a call, stop, transcribe, diarize,
   rename, Suggest/Confirm, then click **Export last session as
   LLM Markdown** and paste `transcript-llm.md` into an LLM.
2. Click **Recording Browser…**, search for a word from the call,
   open a session, export it from the browser.
3. Open **Settings…**, change the theme to light and back, change
   the Whisper model, save, restart the app and confirm the
   settings stuck (and that no token was ever written anywhere).
4. Optional: start Ollama (`ollama serve`, `ollama pull llama3.1`),
   enable local-LLM summaries in Settings, click **Summarise last
   session** and confirm a summary appears; stop Ollama and
   confirm the button explains it is not running.

## 3. Deviations from the PRD and why

1. **No auto-generated action items in the export — by design.**
   The PRD asks for exports "suitable for LLM summarization /
   todo generation", not for OffAir to generate the todos
   itself. The Markdown export carries everything an LLM needs
   to derive them, and inventing action items from a transcript
   (which may contain Whisper errors — "marked down" for
   "markdown" in this fixture) would put words in people's mouths.
   The section exists only for an explicit caller-supplied list
   (e.g. items a local LLM produced and the user approved).
2. **Local LLM summary is real code, untested end-to-end here.**
   Ollama is not running on this VM and llama-cpp-python is not
   installed, so the real Ollama generation path has never
   executed in this build — the probe, the guided error, the
   localhost-only validation and the fake-client unit tests are
   what was verified. The first real Ollama summary must run on
   a machine with Ollama installed. This is stated in README
   and §5, not hidden.
3. **Live polling during recording is still not wired in the UI.**
   This was a Phase 4 follow-up ("small, separately tested
   piece") and Phase 5's scope did not include it: the UI still
   runs the chunked near-live pass on a finished session.
   `ChunkedTranscriber.process_new_audio` remains the tested
   engine-side hook. Wiring it needs a mic-track snapshot +
   timer loop and reading a partially-written WAV while capture
   holds it open — fragile enough to deserve its own tested
   pass (see §5).
4. **Near-live segment merging was not done.** The chunked
   output still has more, shorter segments than the post-stop
   transcript (17 vs 12 on the fixture). The pane text already
   says boundaries can split segments; a merging pass is a
   follow-up, not a Phase 5 blocker.
5. **The HF token is never stored — the flag is the only trace.**
   The PRD's Phase 5 "settings persistence" plus Phase 3's
   "never stored" token rule combine here: `settings.json`
   holds `hf_token_configured: true/false` and nothing else
   token-shaped. `settings_to_dict()` allowlists the keys, and
   `config_from_settings()` ignores an `hf_token` key even if
   one is planted in the file (tested).
6. **GUI not launched on this VM (as in Phases 2–4).** PyQt6 is
   not installed here. The Settings dialog, Recording Browser,
   theme application, crash dialog and new buttons are
   code-reviewed and compile, and their pure parts are unit
   tested — clicking them is part of the Windows manual test.
7. **Phase 1 Windows manual test still open** (see PHASE1.md).
   Phase 5 changes nothing about that gate: no real Teams call
   has been recorded, transcribed, diarized, learned or
   exported yet.

## 4. Observed CPU performance and memory usage

**Test machine (this Linux VM, NOT the target laptop):** 4 vCPU
(AMD EPYC), 7.7 GB RAM total, ~600 MB free at run time, no GPU.
Audio: 61.3 s meeting fixture (3 TTS voices) + three isolated
~9 s speaker samples. Command:
`.venv-phase2/bin/python tests/manual_e2e.py` (exit 0).

### End-to-end — measured, real run

```text
== 1. Transcribe (Whisper tiny, CPU int8) ==
  12 segments in 4.2s (RTF 0.068), peak RSS 292 MB
== 3. Diarize (simple, You vs Remote) ==
  method: dual-channel-energy, accuracy 12/12 = 100.0%
  wall: 0.01s, peak RSS 292 MB
== 4. Seed DB + suggest (fallback extractor) ==
  label 'You': suggested 'John' confidence 0.968
    (best John 0.968, second Marcus 0.910, margin 0.059,
     ambiguous=False, backend=fallback-spectral-v1)
  suggest wall: 0.42s (Remote skipped by design)
== 5. Rename You -> John, learn from session ==
  learned: {'You': {'name': 'John', 'speaker_id': 1, 'seconds': 14.88}}
  John: 2 sample(s), Marcus: 1, Priya: 1 (all fallback-spectral-v1)
== 6. LLM Markdown export ==
  wrote transcript-llm.md in 0.00s
  check speaker John present: PASS
  check speaker Remote present: PASS
  check timestamps present: PASS
  check budget content present: PASS
  check December content present: PASS
  check no invented action items: PASS
  check context block present: PASS
== 7. Local LLM probe (expected: not running here) ==
  Ollama is_available(): False
  llama-cpp is_available(): False
  summarize_locally raised OllamaNotAvailableError (expected)

E2E OK in 10.4s total (transcribe 7.1s, diarize 0.01s,
suggest 0.42s, learn 0.51s, export 0.00s), peak RSS 292 MB
```

Notes on the numbers: the 7.1 s transcribe step includes ~2.9 s
of Whisper model loading (the engine's own 4.2 s wall excludes
it); diarize/suggest/learn/export together take ~1 s; the export
itself is effectively free. Peak RSS (292 MB) is dominated by the
Whisper tiny model. The export preview (quoted in the run)
shows the known tiny-model quirks — "marked down" for markdown —
which is exactly why OffAir
does not auto-generate action items from it.

Full measured table (all phases, measured vs extrapolated):
`docs/CPU_TIMES.md`. Headline: transcription is the only
expensive step (tiny ~2 min / base ~4 min / small ~10 min for a
30-minute meeting, extrapolated); everything else is seconds.

Automated test results on this VM:

```text
.venv-phase2/bin/python -m py_compile <all .py files> → passed (exit 0)
.venv-phase2/bin/python -m pytest tests/ -q            → 140 passed in 1.54s
  (99 Phase 1–4 + 41 new: 15 exports, 7 browser, 8 settings,
   10 UI/theme/crash/log, 1 e2e-fake)
.venv-phase2/bin/python tests/manual_e2e.py            → exit 0, 7/7 checks
python3 main.py --no-gui                               → exit 0 (smoke check)
```

## 5. Recommended follow-ups (the open gates)

In priority order — these are the things that must happen on
the target laptop / with the user before OffAir can be called
done, not more polish:

1. **Windows manual test — the big one.** Record a real Teams
   call on the HP laptop, run the whole pipeline
   (record → transcribe → diarize → rename/suggest → export →
   paste into an LLM), and verify both audio tracks by ear,
   the labels, and the export. Until this passes, every phase's
   success criteria are only partially met (code + unit tests +
   TTS-fixture runs done, real-audio verification open).
2. **First real pyannote run (full diarization + embeddings).**
   Install torch/pyannote on the 8–16 GB laptop, do the HF
   token + licence setup, run the 3-speaker fixture through
   full mode, and re-measure John–Marcus separation with the
   pyannote embedding backend. Do not trust a suggestion until
   that separation is clearly better than the fallback's 0.949.
3. **SpeechBrain ECAPA swap (John's decision, Nov 8 — DONE Nov 8,
   after Phase 5 landed).** Swap the Phase 4 production speaker
   embedding from `pyannote/embedding` to SpeechBrain
   ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, 192-dim,
   Apache 2.0, public weights, **no HF token/login**), keeping
   pyannote as opt-in. This removes the HF-account requirement
   from OffAir's default path. It was deliberately queued
   until after Phase 5 landed so two agents were not editing
   the same files at once. Note: SpeechBrain still needs
   PyTorch, so it shares pyannote's torch-install blocker on
   this VM — its first real run is also on the target laptop.
   As implemented (see PHASE4.md, "ECAPA swap" section), the
   swap also added a second production backend, ONNX ECAPA
   (`speakeronnx` wespeaker-ecapa512, no torch at all), which
   is the default when torch/speechbrain is absent and is the
   only production backend verified working on this VM
   (John–Marcus 0.064). After it lands, decide the fallback's
   fate explicitly (keep as a clearly-labelled dev/test backend,
   or remove it so nobody can accidentally rely on it).
4. **First real Ollama summary.** On a machine with Ollama
   installed (`ollama serve`, `ollama pull llama3.1`), enable
   local-LLM summaries and summarise a real exported
   transcript; note the model, wall time and RAM. Until then
   the Ollama path is code + fake-client tests only.
5. **Re-check the Whisper `small` default on real audio**
   (carried from PHASE2.md): on the clean fixture `base`
   matched `small`'s accuracy at 2.5× less time and RAM.
   Measure tiny/base/small on a 10-minute real Teams recording
   on the laptop and be willing to default to `base`.
6. **Wire live polling during recording** (Phase 4 follow-up):
   `process_new_audio` + a mic-track snapshot + timer loop in
   the UI, as its own tested piece — reading a
   partially-written WAV while capture holds it open is the
   fragile part. Do not bolt it on untested.
7. **Near-live segment merging** (Phase 4 follow-up): join
   adjacent segments from neighbouring chunks when their text
   is a continuation, so the chunked transcript matches the
   post-stop segmentation (17 vs 12 on the fixture).
8. **Close the per-app capture gap or formally accept it**
   (carried from PHASE1.md): true Win11 per-session capture
   vs system-wide loopback + "mute other apps". This is what
   "Remote" actually contains on a real call, and Phase 3/4
   inherit that uncertainty.
9. **Disk-space guard for long calls** (carried from PHASE1.md):
   at ~5.7 MB/min per track, multi-hour sessions add up — show
   the session folder size live and warn before disk fills.

## 6. Whole-project summary (Phases 1–5)

### What exists now

| Phase | What | Status here | Open gate |
|---|---|---|---|
| 1 | Dual-channel WASAPI capture, Record/Pause/Resume/Stop, level meters, mic/system/combined WAVs + metadata.json, PyQt6 UI | Code + 30 unit tests; **not run on Windows** | Windows manual test (real Teams call, both tracks by ear) |
| 2 | faster-whisper CPU int8 (tiny/base/small/medium, default small), transcript.json/.txt/.srt, progress, Transcribe button | Code + 15 tests; real runs on 61 s TTS fixture (tiny 4.3 s/8.0% WER, base 8.3 s/7.5%, small 20.4 s/7.5%) | Re-check small-vs-base on real audio; medium unmeasured |
| 3 | Simple You/Remote diarization (0.01 s, 12/12 synthetic) + pyannote full mode + rename panel + click-to-play | Code + 25 tests; simple mode real, **pyannote never ran** (torch won't install here, no HF token) | First pyannote run on target laptop + HF token |
| 4 | SQLite speakers.db, cosine matching (threshold 0.90, margin/ambiguous), confirm/correct learning, near-live 25 s chunks (RTF 0.086) | Code + 29 tests; real runs with **fallback** extractor; suggest You→John 0.968 (margin 0.059) | First pyannote/ECAPA embedding run; live polling unwired |
| 5 | LLM Markdown export (no invented action items), local LLM (Ollama/llama-cpp, off by default), recording browser + search, settings persistence, dark theme, log rotation + crash dialog | Code + 41 tests; real e2e run 10.4 s, 7/7 checks PASS | First real Ollama summary; GUI clicking on Windows |

Total: **140 pytest tests passing** (fake models / synthetic
audio only — the real-model runs are the `tests/manual_*.py`
scripts, quoted in PHASE2–5.md).

### PRD success criteria — honest scorecard

| PRD criterion | Verdict |
|---|---|
| Records Teams calls without joining as a bot | **Partial** — capture code written, not run on Windows; per-app isolation is a documented fallback (system loopback) |
| Timestamped, speaker-labelled transcripts entirely on-device | **Yes, on the fixture** — 61 s TTS meeting, end-to-end, all local |
| Stores and matches speaker embeddings across sessions | **Yes** — ONNX ECAPA production backend verified on the fixture (John–Marcus 0.064, see PHASE4.md "ECAPA swap"); SpeechBrain ECAPA still unrun, pyannote opt-in only |
| All processing stays local | **Yes** — zero network calls in core paths (model downloads once, then offline); local LLM is localhost-only |
| Exports clean transcripts suitable for LLM summarization/todos | **Yes** — `transcript-llm.md` verified end-to-end, no invented content |
| Runs on CPU-only hardware, no NVIDIA GPU | **Yes, by construction** — CPU/int8 forced, GPU requests rejected; tested on a 4-vCPU VM with ~600 MB free |

### File map (final)

```
main.py                  # entry point (--output-dir, --no-gui, --list-devices)
app/
  config.py              # defaults + settings persistence (~/.offair/settings.json)
  logging_setup.py       # console + rotating log (~5 MB) + crash hook
  audio/                 # Phase 1: devices, WASAPI capture, mixer
  transcription/         # Phase 2: faster-whisper engine + txt/srt exports
  diarization/           # Phase 3: simple You/Remote + pyannote + names
  speakers/              # Phase 4: SQLite DB + embeddings + matching
  nearlive/              # Phase 4: 25 s chunked transcription
  exports/               # Phase 5: LLM Markdown + local LLM (Ollama/llama-cpp)
  storage/               # sessions.py + browser.py (Phase 5 data layer)
  ui/                    # level meter, main window, theme, settings, browser, crash dialog
tests/                   # pytest (140 tests) + manual_*.py real-model scripts
docs/CPU_TIMES.md        # measured + extrapolated CPU times
PRD.md, PHASE1–5.md      # requirements + per-phase reports
```

### Short TODO list (not in core paths)

- [ ] Windows manual test — the one gate that covers all five phases (§5.1).
- [ ] First pyannote run on the target laptop + HF token (§5.2).
- [x] SpeechBrain ECAPA swap — DONE Nov 8 2026, after Phase 5
  landed (§5.3; see PHASE4.md "ECAPA swap"). SpeechBrain's first
  real run is still on the target laptop; ONNX ECAPA is verified
  working on this VM.
- [ ] First real Ollama summary (§5.4).
- [ ] Live-during-recording polling + near-live segment merging (§5.6–5.7).
- [ ] Per-app capture investigation (§5.8) and disk-space guard (§5.9).
