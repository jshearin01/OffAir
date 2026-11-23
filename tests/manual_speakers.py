#!/usr/bin/env python3
"""Manual Phase 4 test — real fallback extractor + learn/suggest + near-live.

This is NOT a pytest test: it runs the real fallback spectral
extractor and the real Whisper tiny model. Run it from the repo root
with the Phase 2 venv:

    .venv-phase2/bin/python tests/manual_speakers.py

What it does (outputs go to --output-dir, default
~/tmp/offair-phase4 — never into the repo):

1. Converts the three isolated speaker MP3s to 16 kHz WAV and
   extracts fallback embeddings. Prints the 3x3 cosine similarity
   matrix and each speaker's half-vs-half self score — the honest
   measure of what the fallback can and cannot separate.
2. Builds the synthetic dual-channel session from
   tests/manual_diarize.py (transcribe the 61.3 s meeting with
   Whisper tiny, mask mic/system by script turn, diarize simple).
3. Seeds the speaker DB with one isolated sample per speaker
   (simulating "learned in earlier calls"), then runs
   suggest_for_session on the meeting session: the "You" cluster
   should suggest John. "Remote" is skipped by design — it mixes
   Priya and Marcus into one cluster (see PHASE4.md).
4. Runs learn_from_session (renaming You -> John first, as a user
   would) and reports the learned speakers + sample counts.
5. Runs the near-live ChunkedTranscriber (real tiny model, 25 s
   chunks) over the meeting WAV and prints per-chunk wall time /
   RTF — the CPU-stability check for near-live mode.

No pyannote/torch is used here (it does not install on this VM —
see PHASE3.md). The fallback is labelled as such throughout.
"""
from __future__ import annotations

import argparse
import itertools
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.diarization.names import rename_speaker  # noqa: E402
from app.diarization.session import diarize_session  # noqa: E402
from app.nearlive.chunker import ChunkedTranscriber  # noqa: E402
from app.speakers.database import SpeakerDatabase  # noqa: E402
from app.speakers.embeddings import FallbackEmbeddingExtractor  # noqa: E402
from app.speakers.session import (  # noqa: E402
    learn_from_session,
    suggest_for_session,
)
from app.transcription.engine import TranscriptionEngine  # noqa: E402
from tests.manual_diarize import (  # noqa: E402
    assign_turns,
    mask_audio,
    parse_script,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MEETING_WAV = FIXTURES / "meeting-3speakers-16k.wav"
MEETING_MP3 = FIXTURES / "meeting-3speakers.mp3"
TRUTH = FIXTURES / "meeting-script.txt"
ISOLATED = {"John": FIXTURES / "speaker-john.mp3",
            "Priya": FIXTURES / "speaker-priya.mp3",
            "Marcus": FIXTURES / "speaker-marcus.mp3"}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def mp3_to_wav(mp3_path: Path, out_dir: Path) -> Path:
    out = out_dir / f"{mp3_path.stem}-16k.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp3_path),
                    "-ar", "16000", "-ac", "1", str(out)], check=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path.home()
                        / "tmp" / "offair-phase4")
    args = parser.parse_args(argv)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = FallbackEmbeddingExtractor()
    print(f"Extractor backend: {extractor.backend} (dim {extractor.dim})",
          flush=True)

    # 1. Isolated-speaker similarity matrix (real fallback extractor).
    print("\n== 1. Fallback extractor on isolated speakers ==", flush=True)
    embeddings: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for name, mp3 in ISOLATED.items():
        wav = mp3_to_wav(mp3, out_dir)
        t0 = time.monotonic()
        embeddings[name] = extractor.extract(wav)
        timings[name] = time.monotonic() - t0
        print(f"  {name}: extracted in {timings[name]:.2f}s "
              f"(peak RSS {peak_rss_mb():.0f} MB)", flush=True)
    print("  Cosine similarity matrix:", flush=True)
    for a, b in itertools.combinations(embeddings, 2):
        cos = float(embeddings[a] @ embeddings[b])
        print(f"    {a:6s} vs {b:6s}  cos={cos:.3f}", flush=True)
    print("  Same-speaker half-vs-half (4-5 s halves, different words):",
          flush=True)
    for name, mp3 in ISOLATED.items():
        wav = out_dir / f"{mp3.stem}-16k.wav"
        data, rate = sf.read(str(wav), dtype="float32")
        half = len(data) // 2
        e1 = extractor.extract_samples(data[:half], rate)
        e2 = extractor.extract_samples(data[half:], rate)
        print(f"    {name}: half-vs-half cos={float(e1 @ e2):.3f}",
              flush=True)

    # 2. Synthetic dual-channel session (same construction as Phase 3).
    print("\n== 2. Synthetic session: transcribe + diarize simple ==",
          flush=True)
    engine = TranscriptionEngine(model_size="tiny")
    result = engine.transcribe_file(MEETING_WAV)
    engine.unload()
    print(f"  Transcription (tiny): {len(result.segments)} segments in "
          f"{result.wall_seconds:.1f}s", flush=True)
    turns = parse_script(TRUTH)
    truth_speakers = assign_turns(result.segments, turns)
    truth_labels = [("You" if s == "John" else
                     "Remote" if s else None) for s in truth_speakers]
    session = out_dir / "synthetic-session"
    session.mkdir(parents=True, exist_ok=True)
    from app.diarization.simple import load_wav_mono
    samples, rate = load_wav_mono(MEETING_WAV)
    you_windows = [(s.start, s.end) for s, lab in
                   zip(result.segments, truth_labels) if lab == "You"]
    remote_windows = [(s.start, s.end) for s, lab in
                      zip(result.segments, truth_labels) if lab == "Remote"]
    sf.write(str(session / "mic.wav"),
             mask_audio(samples, rate, you_windows), rate)
    sf.write(str(session / "system.wav"),
             mask_audio(samples, rate, remote_windows), rate)
    sf.write(str(session / "combined.wav"), samples, rate)
    (session / "transcript.json").write_text(json.dumps(
        {"model": "tiny", "device": "cpu", "audio_file": "combined.wav",
         "segments": [s.to_dict() for s in result.segments]}, indent=2),
        encoding="utf-8")
    diar_payload = diarize_session(session, mode="simple")
    print(f"  Diarization (simple): {diar_payload['method']}, "
          f"{len(diar_payload['segments'])} segments", flush=True)

    # 3. Seed the DB from the isolated samples, then suggest.
    print("\n== 3. Suggest (DB seeded from isolated samples) ==", flush=True)
    db = SpeakerDatabase(out_dir / "speakers.db")
    for name, emb in embeddings.items():
        db.add_embedding(name, emb, backend=extractor.backend,
                         source_session="isolated-fixture",
                         source_label=name)
    t0 = time.monotonic()
    suggestions = suggest_for_session(session, db, extractor=extractor)
    print(f"  suggest_for_session: {len(suggestions)} cluster(s) in "
          f"{time.monotonic() - t0:.2f}s", flush=True)
    for sug in suggestions:
        print(f"    label {sug.label!r}: suggested "
              f"{sug.suggested_name!r} confidence "
              f"{sug.match.confidence:.3f} (best {sug.match.best_name} "
              f"{sug.match.best_score:.3f}, second "
              f"{sug.match.second_best_name} "
              f"{sug.match.second_best_score:.3f}, margin "
              f"{sug.match.margin:.3f}, ambiguous="
              f"{sug.match.is_ambiguous})", flush=True)
    print("  (Remote is skipped by design: it mixes Priya + Marcus.)",
          flush=True)

    # 4. Learn from this session (You -> John, as a user would rename).
    print("\n== 4. Learn from this session ==", flush=True)
    rename_speaker(session, "You", "John")
    learned = learn_from_session(session, db, extractor=extractor)
    print(f"  learned: {learned}", flush=True)
    for speaker in db.list_speakers():
        print(f"    {speaker['name']}: {speaker['sample_count']} sample(s), "
              f"backend {speaker['backend']}", flush=True)
    db.close()

    # 5. Near-live chunked transcription (real tiny model).
    print("\n== 5. Near-live chunked transcription (tiny, 25 s) ==",
          flush=True)
    chunker = ChunkedTranscriber(model="tiny")
    t0 = time.monotonic()
    chunks = chunker.process_file(MEETING_WAV, work_dir=out_dir / "chunks")
    total_wall = time.monotonic() - t0
    for chunk in chunks:
        print(f"  chunk {chunk.index} [{chunk.start:.1f}-{chunk.end:.1f}s]: "
              f"{len(chunk.segments)} segments in {chunk.wall_seconds:.1f}s "
              f"(RTF {chunk.real_time_factor:.3f})", flush=True)
    print(f"  total: {len(chunker.segments)} segments, "
          f"{total_wall:.1f}s wall for 61.3 s audio "
          f"(RTF {total_wall / 61.3:.3f}), "
          f"peak RSS {peak_rss_mb():.0f} MB", flush=True)
    print(f"  first segments: {[s.text[:40] for s in chunker.segments[:2]]}",
          flush=True)
    chunker.unload()
    print(f"\nSession files: {sorted(p.name for p in session.iterdir())}",
          flush=True)
    print(f"Outputs in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
