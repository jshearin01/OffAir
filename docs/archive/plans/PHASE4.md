# OffAir — Phase 4 Build Report

Per the PRD's instructions for the coding agent: what was built, how to
test it, deviations, observed performance, and recommended next-phase
adjustments. **Phase 4 only** — persistent speaker learning +
near-live mode. No AI summaries, no LLM handoff (Phase 5).

> **Update (Nov 8 2026):** the production embedding backend described
> below as pyannote was swapped after Phase 5 landed — see the
> "ECAPA swap" section at the end of this file. The sections above it
> are the original Phase 4 report, kept as history.

## 1. What was built

Persistent Speaker Learning + Near-Live Mode (CPU-only), in
`~/workspace/offair/` (local files only; nothing pushed to GitHub
in this step):

- **`app/speakers/database.py`** — `SpeakerDatabase`: SQLite
  `speakers.db` in the recordings root. Tables `speakers` (id, name
  UNIQUE, backend, created/updated, sample_count) and `embeddings`
  (speaker_id, float32 embedding blob, dim, backend, source
  session/label, created). A speaker's stored embedding is the mean
  of its samples, renormalised; every confirmed sample adds a row,
  so a correction is auditable back to its session. A speaker's
  backend is fixed at creation — adding an embedding from another
  backend raises, because the vectors live in different spaces.
- **`app/speakers/embeddings.py`** — two backends, one interface
  (both return L2-normalised float32 vectors):
  - `PyannoteEmbeddingExtractor` — production backend,
    `pyannote/embedding` on CPU via `pyannote.audio.Inference`,
    audio chunked into 5 s pieces and averaged. Lazy import; clear
    error if pyannote/torch is missing or the HF token/licence
    refuses access. **It has never run here** (see §3).
  - `FallbackEmbeddingExtractor` — `fallback-spectral-v1`, dim 38:
    mean/std of MFCCs 1–12 (log-mel + DCT) plus a log-pitch
    histogram (12 bins, 60–400 Hz) and pitch median/std. Numpy +
    soundfile only, ~0.3 s per 9 s file. It is labelled everywhere
    as a spectral proxy, **not a voice fingerprint** — see §4 for
    the measured limits. `get_default_extractor()` returns pyannote
    when importable, else the fallback, and always says which.
- **`app/speakers/matching.py`** — cosine matching with a
  threshold (default 0.90, `SPEAKER_MATCH_THRESHOLD` in config),
  best/second-best scores and margin on every result, and an
  `is_ambiguous` flag when the margin is under 0.05 (two speakers
  nearly tied). `confirm_suggestion` adds the sample to the
  suggested speaker; `correct_suggestion` adds it to the right
  speaker only — a correction never reinforces the mistake.
  Matching only ever compares speakers of the same backend.
- **`app/speakers/session.py`** — `learn_from_session` (one
  embedding per labelled speaker, name taken from the session's
  `speakers.json` rename when present) and `suggest_for_session`
  (per-cluster suggestion with confidence; nothing written to the
  DB), plus `write_suggestions` / `load_suggestions`
  (`speaker_suggestions.json`), and `confirm_suggestion` /
  `correct_suggestion` which learn *and* rename the label in the
  session and re-export transcript.txt/.srt. Audio source per
  label: `You` from `mic.wav`, `Remote` from `system.wav`,
  everything else from `combined.wav`.
- **Learnable-cluster rule (a deliberate design decision, §3).**
  Which clusters are a single person depends on the diarization
  method (read from `diarization.json`): simple dual-channel mode
  learns only `You` — `Remote` mixes every remote speaker into
  one cluster, so learning it would store a blended voice;
  single-track fallback learns nothing (labels alternate by
  pause, not identity); full pyannote mode learns every
  `SPEAKER_xx` cluster. The same filter applies to suggestions.
- **`app/nearlive/chunker.py`** — `ChunkedTranscriber`: 25 s
  windows (configurable 5–60 s), 0.5 s overlap, Whisper `tiny`
  default, one window at a time. Segments are offset to absolute
  session time; the overlap's second copy is de-duplicated by
  (start, text). Provisional labels come from dual-channel energy
  when mic + system chunk files are given, else `"Provisional"`.
  `process_file` chunks a finished WAV; `process_new_audio`
  transcribes newly completed windows of a growing WAV, anchored
  at the watermark minus the overlap (a precomputed-from-zero
  window list would skip audio — this was caught and fixed during
  development, see the test). The engine is unloaded on demand.
- **UI wiring (`app/ui/main_window.py`)** — Known Speakers panel
  (from `speakers.db`), **Suggest** button (background worker),
  suggestions list with confidence / best / margin / ambiguous
  flag, **Confirm** / **Correct** buttons (Correct uses the name
  field), and a near-live pane + **Run near-live (chunked)**
  button (background worker, progress bar, pane fills with
  provisional segments when the run finishes). PyQt6 is not
  installed on this VM, so the GUI was not launched here — wiring
  is verified by `py_compile` + the import-guard behaviour, as in
  Phases 2–3.
- **`app/config.py`** — `SPEAKER_MATCH_THRESHOLD = 0.90`,
  `SPEAKER_MATCH_AMBIGUOUS_MARGIN = 0.05`, near-live defaults
  (25 s chunk, 0.5 s overlap, model `tiny`).
- **`requirements.txt`** — no new dependency. SpeechBrain is
  deliberately NOT added (it is an alternative backend, not the
  one implemented, and would pull a second torch stack); the
  production embedding backend is `pyannote/embedding`, which
  comes with the `pyannote.audio` already listed for Phase 3.
- **`tests/test_speakers.py`** — 19 new tests (fake embeddings,
  no models): cosine basics, match / threshold boundary /
  ambiguity / cross-backend refusal, confirm moving the mean,
  correct not reinforcing the wrong speaker, DB round-trip /
  rename / delete / backend mismatch, learn + suggest + confirm
  + correct on synthetic sessions, simple-mode Remote skip,
  single-track learns nothing, missing-transcript error.
- **`tests/test_nearlive.py`** — 10 new tests (fake engine):
  window splitting + coverage, validation, offsets + ordering,
  overlap dedup, dual-channel provisional labels (You/Remote),
  watermark advance on a growing file, missing-audio error.
- **`tests/manual_speakers.py`** — real-model manual test (not
  run by pytest): real fallback extractor on the isolated
  speakers (similarity matrix), suggest + learn on the synthetic
  session, and a real near-live chunked run with Whisper tiny.

## 2. How to test it

### Automated (this Linux VM — passed, see §4)

```bash
cd ~/workspace/offair
.venv-phase2/bin/python -m py_compile main.py app tests/test_*.py \
  app/transcription/*.py app/diarization/*.py app/speakers/*.py \
  app/nearlive/*.py app/ui/main_window.py app/config.py
.venv-phase2/bin/python -m pytest tests/ -q   # 99 passed (70 Phase 1-3 + 29 new)
python3 main.py --no-gui                      # Phase 1 smoke check still OK
```

### Real extractor + learn/suggest + near-live (this Linux VM — ran, see §4)

```bash
.venv-phase2/bin/python tests/manual_speakers.py
# 1. Fallback embeddings for the 3 isolated speakers -> matrix.
# 2. Transcribe (tiny) + diarize simple on the 61.3 s meeting,
#    seed speakers.db from the isolated samples, suggest, learn.
# 3. Near-live chunked run (tiny, 25 s) over the meeting WAV.
# Outputs go to ~/tmp/offair-phase4 — never into the repo.
```

### Manual (Windows — still required)

1. On Windows with Teams: venv, `pip install -r requirements.txt`,
   `python main.py`. Record a call, stop, transcribe, diarize.
2. Rename a speaker (e.g. `You` -> John), click **Suggest** —
   with an empty DB the first call suggests nothing (Unknown),
   which is correct. Confirm/correct nothing yet.
3. Do a second call with the same people, diarize, click
   **Suggest** — with the pyannote backend installed this is the
   test that matters: the same names should be suggested. With
   the fallback backend, treat suggestions as "probably, check
   the margin" (§4).
4. Click **Run near-live (chunked)** and watch the pane fill;
   compare its text with the post-stop transcript (same words,
   more segments — §3.4).
5. Confirm the DB file `speakers.db` appears in the recordings
   root and grows a sample per confirmed call.

## 3. Deviations from the PRD and why

1. **Pyannote embeddings did NOT run here — and were not faked.**
   torch/pyannote are still not installed on this VM (Phase 3:
   the torch 2.15 CPU wheel stalls at 261.6 MB from PyPI, and
   ~700 MB free RAM would likely OOM on a 1–2 GB pipeline
   anyway). Per the task instructions the same install was not
   retried in a loop — conditions have not changed. The
   `PyannoteEmbeddingExtractor` code is written, its interface
   matches the fallback's, and the database/matching/session
   layers are backend-agnostic (they only ever compare same-
   backend vectors). The first real pyannote embedding run must
   happen on the 8–16 GB target laptop, with the HF token from
   Phase 3. No HF account/token exists yet and none was invented.
2. **The fallback extractor is a weak discriminator for similar
   male voices — measured, not assumed (§4).** It separates
   Priya cleanly but cannot reliably separate John from Marcus:
   cross-speaker cosine 0.949, *above* John's same-speaker
   half-vs-half score of 0.876. The threshold (0.90) therefore
   does not by itself prevent a John/Marcus confusion; the
   margin is the guard, and even the margin was thin in the
   manual run (0.059). This is why the fallback is labelled a
   spectral proxy everywhere, why suggestions show best/second/
   margin, and why nothing is learned without an explicit
   Confirm/Correct. On the target laptop, switch to the pyannote
   backend — do not ship the fallback as the voice fingerprint.
3. **Simple-mode `Remote` is skipped for learning/suggestions —
   by design.** The PRD's Phase 4 assumes diarization clusters
   are people; simple mode's `Remote` cluster is not one person,
   it is every remote speaker mixed. Learning it would poison
   the database with a blended embedding and then "suggest" that
   blend for every remote speaker. In a 1-on-1 call Remote
   happens to be one person, but the label cannot tell that, so
   the safe rule is to skip it. Full-mode clusters do not have
   this problem. If simple mode is the only diarization a user
   runs, speaker learning effectively learns `You` only — that
   limitation is stated in README § Phase 4 and should be
   surfaced in the UI text in Phase 5.
4. **Near-live chunk boundaries split Whisper segments.** The
   chunked run on the fixture produced 17 segments vs 12 for
   the single-pass tiny run on the same audio — same words,
   split differently at the 25 s / 49.5 s boundaries (plus the
   0.5 s overlap). The dedup only removes exact (start, text)
   duplicates, not re-split segments. For a progressive
   transcript this is cosmetic; for a final transcript the
   post-stop run is still the canonical one. A segment-merging
   pass (join adjacent segments from neighbouring chunks when
   their text is a continuation) is a Phase 5 follow-up.
5. **Live polling during recording is not wired in the UI.**
   `ChunkedTranscriber.process_new_audio` exists, is unit-tested
   (watermark advance on a growing file), and is the engine-side
   hook — but the UI runs the chunked pass on a finished session
   (post-stop simulation). Polling a growing recording needs a
   snapshot copy of the mic track plus a timer/worker loop, and
   reading a partially-written WAV while capture holds it open
   is fragile enough to deserve its own tested pass rather than
   being bolted onto this phase. The primary workflow
   (process after stop) is unaffected.
6. **SpeechBrain not added** (see §1): it is an alternative
   backend, not the implemented one. The requirements note says
   so, so a future switch is a deliberate choice, not an
   accident.
7. **No settings panel (yet)** — as in Phases 2–3, the threshold
   and near-live settings live in `app/config.py` and README;
   a dedicated settings panel is deferred to Phase 5 polish.
8. **Phase 1 Windows manual test still open** (see PHASE1.md §4 /
   PHASE2.md §0). Speaker learning and near-live were tested on
   clean TTS, not real Teams audio — and with only ~9 s isolated
   samples per speaker, not the 20–30 s of varied speech a real
   embedding wants.

## 4. Observed CPU performance and memory usage

**Test machine (this Linux VM, NOT the target laptop):** 4 vCPU
(AMD EPYC), 7.7 GB RAM total, ~700 MB free at run time, no GPU.
Audio: 61.3 s meeting fixture (3 TTS voices) + three isolated
~9 s speaker samples. Command:
`.venv-phase2/bin/python tests/manual_speakers.py` (exit 0).

### Fallback extractor — measured, real run

| Measurement | Value |
|---|---|
| Extraction time, one ~9 s isolated file | **0.28–0.32 s** |
| Peak RSS during extraction | **~40 MB** (extractor alone) |
| Embedding dim | 38 (24 MFCC stats + 14 pitch features) |

Cosine similarity matrix (isolated speakers, whole file each):

```text
John   vs Priya   cos=0.702
John   vs Marcus  cos=0.949   <- cannot separate the two male voices
Priya  vs Marcus  cos=0.597
```

Same-speaker half-vs-half (4–5 s halves, different words):

```text
John:   0.876
Priya:  0.978
Marcus: 0.977
```

Read it plainly: Priya is far from both men (0.60–0.70 cross
vs 0.98 self) and would be suggested correctly with a wide
margin. John vs Marcus is the failure mode — cross-speaker
0.949 is *higher* than John's own half-vs-half score of 0.876,
so no threshold separates them: 0.90 lets the confusion
through, 0.96 would miss John himself. The pitch histogram
helps Priya (median F0 ~193 Hz vs ~106 Hz John / ~94 Hz
Marcus) but the two male voices overlap in pitch and in MFCC
timbre on 9 s of TTS. Longer, more varied samples might help
slightly; a real embedding model is the actual fix. Prototype
variants were tried before locking this in (MFCC-only:
John–Marcus 0.980; pitch-only: 0.918 but John's self-score
fell to 0.721; combined w=1.0 was the best compromise).

### Suggest + learn — measured, real run

DB seeded with one isolated sample per speaker (simulating
"learned in earlier calls"); suggestions on the synthetic
meeting session:

```text
label 'You': suggested 'John' confidence 0.968
  (best John 0.968, second Marcus 0.910, margin 0.059, ambiguous=False)
```

The suggestion was right, but the margin (0.059) is only just
above the 0.05 ambiguous flag — a second Marcus sample, or a
noisier John turn, could easily flip it to ambiguous or wrong.
`Remote` was skipped by design (see §3.3). `suggest_for_session`
took 0.40 s for the session. Learning then stored John's
meeting sample (14.88 s of `You` audio from `mic.wav`):
John 2 samples, Priya 1, Marcus 1 in `speakers.db`.

### Near-live chunked transcription — measured, real run

Whisper tiny, 25 s chunks, 0.5 s overlap, on the 61.3 s meeting:

```text
chunk 0 [0.0-25.0s]:   5 segments in 2.3s (RTF 0.094)
chunk 1 [24.5-49.5s]:  7 segments in 1.6s (RTF 0.064)
chunk 2 [49.0-61.3s]:  5 segments in 1.2s (RTF 0.097)
total: 17 segments, 5.3s wall for 61.3s audio (RTF 0.086),
peak RSS 307 MB (whole process, tiny model loaded)
```

A 25 s window takes ~2 s to transcribe, so near-live keeps up
with real time with a large margin on CPU (RTF ~0.09 — roughly
10x faster than real time). Extrapolated, not measured: a
60-minute meeting chunked this way takes ~5–6 minutes of CPU
with tiny, in 25 s steps, and the transcript is visible from
the first chunk instead of only at the end. Peak RSS (307 MB)
is dominated by the Whisper model, not the chunker.

Automated test results on this VM:

```text
.venv-phase2/bin/python -m py_compile <all .py files> → passed (exit 0)
.venv-phase2/bin/python -m pytest tests/ -q            → 99 passed in 1.59s
  (70 Phase 1–3 + 19 speakers + 10 near-live)
.venv-phase2/bin/python tests/manual_speakers.py       → exit 0
python3 main.py --no-gui                               → exit 0 (smoke check)
```

## 5. Recommended next-phase adjustments

- **First real pyannote embedding run is the top priority for
  Phase 5's end-to-end test, not more fallback tuning.** On the
  target laptop (torch installs, HF token set), extract
  embeddings for John/Priya/Marcus with the pyannote backend and
  re-run the manual script's matrix: if John–Marcus separation
  is not clearly better than the fallback's 0.949, the matching
  threshold and the ambiguous-margin flag need re-tuning before
  anyone trusts a suggestion. Do not tune the fallback further
  to chase the two-male-voices case — it is the wrong tool.
- **Decide the fallback's fate explicitly.** Either keep it as a
  clearly-labelled dev/test backend (current state), or remove
  it once pyannote runs, so a user can never accidentally rely
  on it. If it stays, the UI should say which backend produced
  each suggestion (the data is already in the DB and in
  `speaker_suggestions.json` — it just is not shown in the
  suggestions list yet).
- **Wire live polling during recording** (`process_new_audio` +
  a mic-track snapshot + timer) as a small, separately tested
  piece — it is the one Phase 4 scope item that is engine-ready
  but UI-unwired.
- **Add a segment-merging pass for near-live transcripts** so
  the chunked output matches the post-stop segmentation (17 vs
  12 segments on the fixture), or document the split as
  expected behaviour in the pane itself.
- **Surface the learnable-cluster rule in the UI.** When a user
  clicks Suggest on a simple-mode session, the panel should say
  "Remote is not suggested — it mixes all remote speakers"
  instead of silently returning only `You`.
- **Re-check the Whisper `small` default on real audio**
  (carried from PHASE2.md §5) — it still stands, and it
  interacts with near-live: `small` must never become the
  near-live model (RTF 0.333 would fall behind real time).
- **No changes needed to the session format for Phase 5.**
  `speakers.db` lives in the recordings root, suggestions in
  `speaker_suggestions.json` per session; Phase 5's recording
  browser can read both without restructuring.

## Short TODO list (not in core paths)

- [ ] Real SpeechBrain ECAPA run on the target laptop (top
  priority — the backend John chose has still never run
  anywhere; torch + speechbrain install, 3-speaker fixture,
  re-measure John–Marcus separation and the 0.55 threshold).
- [ ] Real pyannote embedding run on the target laptop + HF token
  (opt-in backend now, not the default).
- [ ] Live-during-recording polling (process_new_audio + snapshot + timer).
- [ ] Near-live segment merging across chunk boundaries.
- [ ] Show the embedding backend + "Remote not suggested" note in the UI.
- [ ] Windows manual test — record a real Teams call, transcribe, diarize, suggest, confirm, second call with the same people.
- [ ] Decide whether the fallback extractor stays or is removed once an ECAPA backend runs on the target laptop.

## ECAPA swap (Nov 8 2026 — after Phase 5 landed)

John's decision (Nov 8): swap the production speaker embedding
from `pyannote/embedding` to SpeechBrain ECAPA-TDNN
(`speechbrain/spkrec-ecapa-voxceleb`, 192-dim, Apache 2.0,
public weights, no HF token/login), keep pyannote as opt-in.
Queued until after Phase 5 landed so two agents were not
editing the same files at once.

### What changed

- **Two production ECAPA backends, one interface** (both
  192-dim, L2-normalised, 16 kHz in):
  - `speechbrain-ecapa-v1` — `SpeechBrainEcapaEmbeddingExtractor`
    (`speechbrain/spkrec-ecapa-voxceleb`). The backend John
    chose. Needs torch + speechbrain (separate install:
    `requirements-speechbrain.txt`, deliberately NOT in
    `requirements.txt` so a plain install never pulls torch).
    Lazy import; 10 s chunks averaged. **Has not run yet** —
    torch does not install on this VM (PyPI stalls at 261.6 MB
    into the 539 MB wheel; ~600 MB free RAM would likely OOM).
    First real run is on the target laptop.
  - `onnx-ecapa-v1` — `OnnxEcapaEmbeddingExtractor`
    (`speakeronnx` `wespeaker-ecapa512`, source
    `Wespeaker/wespeaker-ecapa-tdnn512-LM`, CC-BY-4.0, public,
    no token, ~20 MB download to the HF cache on first use).
    Needs only onnxruntime + numpy + soundfile — **no torch**.
    Added because it is the only production backend that
    actually runs on this VM; it is the default whenever
    torch/speechbrain is absent.
- **Pyannote is opt-in only now** (`pyannote-embedding-v1`).
  It is no longer in the default chain — select it explicitly
  with `get_extractor("pyannote")` or the `embedding_backend`
  setting. It is still installed with pyannote.audio (needed
  for Phase 3 full diarization).
- **Fallback stays, test-only** (`fallback-spectral-v1`). The
  "decide the fallback's fate" item above still stands.
- **Backend selection** (`get_extractor` / `get_default_extractor`,
  `embedding_backend` setting, default `"auto"`): SpeechBrain
  ECAPA when torch + speechbrain import, otherwise ONNX ECAPA
  when speakeronnx imports, otherwise the fallback. The UI
  threads the configured backend through suggest / confirm /
  correct, so a confirmed sample always lands under the same
  backend that produced the suggestion.
- **Per-backend match thresholds** (`app/config.py`,
  `SPEAKER_MATCH_THRESHOLDS`): ECAPA backends use 0.55,
  fallback and pyannote keep 0.90. This was not optional — the
  old single threshold (0.90) would have made ECAPA backends
  never suggest anyone, because ECAPA same-speaker pairs score
  ~0.8, not ~0.95. `match_embedding` and `suggest_for_session`
  now default to the per-backend threshold; an explicit
  threshold still overrides it.
- **Docs**: README "Speaker embedding backends" section,
  PHASE5.md §5.3 marked done, this section.

### Measured verification (real run, this VM, Nov 8 2026)

`OnnxEcapaEmbeddingExtractor` against the 3 isolated fixture
speakers (`tests/manual` one-off script, tryouts venv with
speakeronnx + onnxruntime 1.31.1; model already cached from
the tryout run):

| pair | cosine |
|---|---|
| John–Priya | 0.120 |
| John–Marcus | **0.064** |
| Priya–Marcus | 0.046 |

Half-vs-half self-similarity: John 0.880, Priya 0.808,
Marcus 0.882. First extract 3.6 s (model load, cached),
~0.4 s per ~9 s file after that, peak RSS 169 MB.

The John–Marcus test is the deciding one: the fallback scored
0.949 cross-speaker against John's own 0.876 — no threshold
could separate them. ONNX ECAPA scores 0.064 cross vs 0.88
self. The 0.55 threshold sits between the measured cross
(≤0.12) and self (≥0.81) with margin on both sides; it is
still a fixture measurement on ~9 s of clean TTS per speaker
and must be re-measured on real meeting audio.

### What did NOT run

- SpeechBrain ECAPA itself (no torch here) — code + fake-
  classifier tests only. Its 0.55 threshold is provisional:
  same ECAPA-TDNN family and VoxCeleb training as the ONNX
  model, but unmeasured. Re-measure it on its first real run.
- Pyannote embeddings (opt-in, unchanged — still never ran).
- No Windows / GUI run: the UI wiring (backend threaded
  through `_SuggestWorker` and confirm/correct) is
  py_compile-checked and covered by the existing UI tests'
  import path, but not clicked through here.
