# OffAir — Expected CPU Processing Times

All processing is CPU-only, no GPU/CUDA. Times below are **measured**
on the Linux VM used to build OffAir unless marked otherwise, or
**extrapolated** (RTF × meeting length) and labelled as such. The VM
(4 vCPU AMD EPYC, 7.7 GB RAM, only ~650 MB free at run time, no GPU)
is much weaker than the 8–16 GB target HP laptop, so treat the
measured numbers as a lower bound on speed — the laptop should be
noticeably faster, especially for Whisper `small` (which needed swap
here). No timings were ever measured on Windows hardware or on real
Teams audio in this build; the Windows manual test is still open
(see README).

Test audio throughout: 61.3 s of clear TTS dialogue, 3 voices
(John/Priya/Marcus), 16 kHz mono, ground truth in
`tests/fixtures/meeting-script.txt`. Clean TTS with no overlap,
crosstalk or room noise — real meetings will be slower to process
in the sense that accuracy drops, not that wall time changes much
(Whisper/pyannote wall time depends on audio length, not content).

## Measured (this VM)

| Step | Model / mode | Wall (61.3 s audio) | Real-time factor | Peak RSS | Notes |
|---|---|---|---|---|---|
| Record (Phase 1) | — | real-time | 1.0 (by definition) | tens of MB | PCM to disk + stop-time mix; ~5.7 MB/min per track |
| Transcribe (Phase 2) | Whisper tiny, int8 | 4.3 s | 0.070 | 294 MB | WER 8.0% vs script |
| Transcribe (Phase 2) | Whisper base, int8 | 8.3 s | 0.135 | 392 MB | WER 7.5% |
| Transcribe (Phase 2) | Whisper small, int8 | 20.4 s | 0.333 | 943 MB* | WER 7.5% — same as base, 2.5× slower |
| Transcribe (Phase 2) | Whisper medium | **not measured** | — | — | 1.5 GB model, too large for this VM |
| Diarize simple (Phase 3) | You vs Remote, energy | **0.01 s** | 0.0002 | — | 12/12 on synthetic dual-channel fixture† |
| Diarize full (Phase 3) | pyannote 3.1 | **not measured** | — | — | torch/pyannote never installed/ran here |
| Diarize sherpa (Nov 8 2026) | sherpa-onnx, threshold 0.5 | **20.2 s** (diarize only; 24.2 s incl. pipeline build) | 0.330 | 274 MB | 9/9 turns correct on the fixture‡ |
| Diarize sherpa (Nov 8 2026) | sherpa-onnx, expected-speakers=3 | **21.3 s** | 0.347 | — | 9/9 turns correct on the fixture‡ |
| Extract embedding (Phase 4) | fallback-spectral-v1 | 0.28–0.32 s per ~9 s file | — | ~40 MB | Spectral proxy, not a voice fingerprint |
| Suggest (Phase 4) | fallback, 3 speakers in DB | 0.40–0.42 s | — | — | You → John 0.968, margin 0.059 |
| Near-live (Phase 4) | Whisper tiny, 25 s chunks | 5.3 s | 0.086 | 307 MB | 17 segments vs 12 post-stop (chunk splits). Measured with tiny; since Nov 20 2026 the near-live run uses the configured model (base by default), like live transcription — base's numbers are the "Transcribe (Phase 2) | Whisper base" row above (RTF 0.135, still comfortably real-time). |
| Live transcription (Nov 19 2026) | Whisper base, 25 s chunks, while recording | ~3.4 s per 25 s chunk (RTF 0.135) | 0.135 | ~392 MB (base model) | Progressive saves to `transcript-live.json` after every chunk; Stop transcribes only the tail. A 5 s poll copies + mixes the growing tracks, and the chunker loads the whole combined snapshot each tick — the remaining memory cost on very long meetings (see the plan's Item 3 findings). |
| End-to-end (Phase 5) | tiny + simple + suggest + learn + export | **10.4 s total** | 0.17 | 292 MB | Incl. model load (~2.9 s); export 0.00 s |

\* Small's 943 MB is the process high-water mark for a run that
loaded tiny/base/small in sequence in one process, so it overstates
small alone. Even so, small needed swap on this VM.

† The 12/12 simple-mode figure is on a synthetic fixture whose
mic/system masks were built from the same Whisper windows used for
ground truth — it validates the labelling logic, not turn-boundary
detection or real Teams audio (see `archive/plans/PHASE3.md`).

‡ Sherpa diarization, measured Nov 8 2026 on the same 61.3 s /
3-voice / 9-turn fixture, with the new `app/diarization/sherpa.py`
(`tests/manual_sherpa.py`, sherpa-onnx 2.0.1): threshold clustering
(default 0.5) and the expected-speakers=3 hint both produced 9
turns, 3 speakers, and 9/9 identity when every turn was
embedding-checked against the isolated references with the
ERes2Net extractor (best-match cosine 0.85–0.93, cross-scores
near zero). Thresholds 0.5, 0.6 and 0.7 all gave identical results,
so the outcome is not knife-edge sensitive to the threshold. One
real deviation from the Nov 8 tryout (`num_clusters=3`): the
threshold runs label Marcus's first turn `SPEAKER_02` but extend
it to 23.03 s (absorbing the 22.09–22.41 overlap the tryout kept
as two turns), so turn 3 is 14.54–23.03 and turn 4 starts at
23.03 — no overlap, same 9/9 identity. Raw cluster ids are
renumbered to contiguous `SPEAKER_00/01/02` in order of first
appearance (threshold clustering can leave gaps, e.g. 00/01/03).
Models: segmentation `model.onnx` 6.0 MB + ERes2Net 26.5 MB
(~33 MB total, downloaded once from public GitHub releases, no
token); pipeline build ~0.5 s. Extrapolated: ~10 min for a
30-minute meeting, ~20 min for 60 minutes (RTF ~0.33–0.39).
Clean TTS with no overlap or room noise — real meetings will be
less accurate, not much slower.

## Extrapolated (not measured) — 30- and 60-minute meetings

Extrapolation = measured RTF × meeting length. Linear in audio
length for every step except model loading (a fixed ~3 s for tiny).

| Step | 30 min meeting | 60 min meeting |
|---|---|---|
| Transcribe, tiny | ~2 min | ~4 min |
| Transcribe, base | ~4 min | ~8 min |
| Transcribe, small | ~10 min | ~20 min |
| Transcribe, medium | ~45 min (estimate, unmeasured) | ~90 min (estimate) |
| Diarize, simple | well under 1 s | well under 1 s |
| Diarize, full (pyannote) | minutes (PRD expectation, unmeasured) | minutes (unmeasured) |
| Diarize, sherpa (RTF ~0.35, measured) | ~10 min | ~20 min |
| Near-live, base (since Nov 20 2026) | ~5 min of CPU, in 25 s steps, visible from the first chunk | ~10 min of CPU, same pattern |
| Live transcription, base (Nov 19 2026) | ~4 min of CPU, spread across the meeting in 25 s chunks as it records — Stop then transcribes only the tail (seconds) | ~8 min of CPU, same pattern |
| LLM Markdown export | < 1 s | < 1 s |
| Local LLM summary (Ollama) | depends on the model — not measured here (Ollama was not running on this VM); expect minutes on CPU for a 7B model | same |

## What this means for the target laptop (8–16 GB, modern CPU)

- **Transcription is the bottleneck.** Budget ~4 min for a
  30-minute meeting with `base` — the default since Nov 19 2026.
  On the clean fixture base matched small's accuracy (both 7.5%
  WER) at 2.5x less time, and a Nov 19 crash report — 1.0.2
  crashing / never finishing transcribing a 30-minute meeting
  with `small` — settled the re-check the PRD had left open:
  `base` is now the daily default and `small` is opt-in. (Budget
  ~10 min for a 30-minute meeting if you do pick `small`.)
  Re-check the choice on real meeting audio when the Windows
  manual test runs.
- **Live transcription (Nov 19 2026) moves the work into the
  meeting.** With `base` as the live model a 25 s chunk costs
  ~3.4 s of CPU, so transcription keeps up with the recording and
  Stop only has the tail left — the ~4 min post-stop wait for a
  30-minute meeting is gone, and a crash loses at most the chunk
  in flight (`transcript-live.json` is saved after every chunk).
  The remaining cost is the per-tick snapshot: every 5 s the
  growing tracks are copied, mixed, and the whole combined
  snapshot loaded, so very long meetings still carry that
  whole-file load on each tick.
- **Everything else is fast.** Simple diarization, speaker
  matching, learning and export together take seconds, not
  minutes, even on this weak VM.
- **Full pyannote diarization is the one unmeasured cost.** It
  needs ~1–2 GB of RAM and "minutes for 30 minutes" per the PRD;
  measure it on the target laptop with the HF token before
  relying on it. Speaker embeddings are no longer blocked on
  it: the ECAPA swap (Nov 8 2026, see `archive/plans/PHASE4.md`) made ONNX
  ECAPA the torch-free production default — 169 MB peak RSS,
  ~0.4 s per ~9 s of audio on this VM — while SpeechBrain
  ECAPA still shares the torch-install requirement and is
  unmeasured here.
- **Disk, not CPU, is the Phase 1 limit:** ~5.7 MB/min per track
  (~345 MB per track for an hour, ~1 GB with the combined mix).
