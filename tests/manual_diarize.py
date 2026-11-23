#!/usr/bin/env python3
"""Manual Phase 3 test — simple-mode diarization on the real fixture.

This is NOT a pytest test: it runs the real faster-whisper tiny model
and the real simple-mode diarizer. Run it from the repo root with the
Phase 2 venv:

    .venv-phase2/bin/python tests/manual_diarize.py [--model tiny]
    .venv-phase2/bin/python tests/manual_diarize.py --session-dir DIR

How the synthetic dual-channel session is built (documented, because
the fixture is a single mixed track):

1. The 61.3 s meeting fixture (3 TTS voices, John/Priya/Marcus) is
   transcribed with Whisper tiny -> timestamped segments.
2. Each segment is assigned to a script turn (the 9 turns in
   tests/fixtures/meeting-script.txt, in order) by best word overlap.
   That gives ground truth per segment: John = "You", Priya/Marcus =
   "Remote" in simple mode (simple mode cannot tell Priya from
   Marcus — that is the full mode's job).
3. ``mic.wav`` is the fixture audio masked to John segments only
   (silence elsewhere); ``system.wav`` is masked to Priya/Marcus
   segments only; ``combined.wav`` is the untouched fixture. This
   mimics a Phase 1 session: mic holds only your voice, system holds
   only the remote side.
4. :func:`diarize_session` (simple mode) runs on the session, and the
   per-segment labels are compared with the ground truth.

Outputs go to --output-dir (default ~/tmp/offair-phase3) — never
into the repo, and the fixture audio itself is never modified.
Ground truth is the script, not the model: a segment whose words
match no turn is reported, not silently assigned.
"""
from __future__ import annotations

import argparse
import json
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.diarization.names import rename_speaker  # noqa: E402
from app.diarization.session import diarize_session, reapply_names  # noqa: E402
from app.diarization.simple import load_wav_mono  # noqa: E402
from app.transcription.engine import TranscriptionEngine  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DEFAULT_AUDIO = FIXTURES / "meeting-3speakers-16k.wav"
DEFAULT_TRUTH = FIXTURES / "meeting-script.txt"
YOU_SPEAKER = "John"  # the "You" side in this fixture


def normalize_words(text: str) -> list[str]:
    text = re.sub(r"^(John|Priya|Marcus):\s*", "", text, flags=re.M)
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return text.split()


def parse_script(path: Path) -> list[tuple[str, list[str]]]:
    """[(speaker, words)] in order, one entry per script line."""
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        speaker, text = line.split(":", 1)
        turns.append((speaker.strip(), normalize_words(text)))
    return turns


def assign_turns(segments, turns: list[tuple[str, list[str]]]
                 ) -> list[str | None]:
    """Best-overlap turn speaker per segment (bag-of-words overlap)."""
    assigned: list[str | None] = []
    for seg in segments:
        words = set(normalize_words(seg.text))
        best, best_score = None, 0
        for speaker, turn_words in turns:
            score = len(words & set(turn_words))
            if score > best_score:
                best, best_score = speaker, score
        assigned.append(best if best_score >= 2 else None)
    return assigned


def mask_audio(samples: np.ndarray, rate: int,
               windows: list[tuple[float, float]], pad: float = 0.15
               ) -> np.ndarray:
    """Keep audio inside ``windows`` (plus a small pad), silence elsewhere."""
    out = np.zeros_like(samples)
    for start, end in windows:
        i0 = max(0, int((start - pad) * rate))
        i1 = min(samples.size, int((end + pad) * rate))
        out[i0:i1] = samples[i0:i1]
    return out


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--model", default="tiny",
                        choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output-dir", type=Path, default=Path.home()
                        / "tmp" / "offair-phase3")
    args = parser.parse_args(argv)

    if not args.audio.exists():
        print(f"Audio not found: {args.audio}", file=sys.stderr)
        return 2
    turns = parse_script(args.ground_truth)
    print(f"Script turns: {len(turns)} "
          f"({', '.join(s for s, _ in turns)})", flush=True)

    # 1. Transcribe (tiny by default — the point here is diarization,
    #    not Whisper accuracy; Phase 2 already measured the models).
    engine = TranscriptionEngine(model_size=args.model)
    t0 = time.monotonic()
    result = engine.transcribe_file(args.audio)
    engine.unload()
    print(f"Transcription ({args.model}): {len(result.segments)} segments "
          f"in {result.wall_seconds:.1f}s, peak RSS {peak_rss_mb():.0f} MB",
          flush=True)

    # 2. Ground truth per segment.
    truth_speakers = assign_turns(result.segments, turns)
    truth_labels = [("You" if s == YOU_SPEAKER else
                     "Remote" if s else None) for s in truth_speakers]
    unassigned = sum(1 for t in truth_labels if t is None)
    print(f"Segments with no matching script turn: {unassigned} "
          f"(these are excluded from accuracy)", flush=True)

    # 3. Build the synthetic dual-channel session.
    session = args.output_dir / "synthetic-session"
    session.mkdir(parents=True, exist_ok=True)
    samples, rate = load_wav_mono(args.audio)
    you_windows = [(s.start, s.end) for s, lab in
                   zip(result.segments, truth_labels) if lab == "You"]
    remote_windows = [(s.start, s.end) for s, lab in
                      zip(result.segments, truth_labels) if lab == "Remote"]
    import soundfile as sf
    sf.write(str(session / "mic.wav"),
             mask_audio(samples, rate, you_windows), rate)
    sf.write(str(session / "system.wav"),
             mask_audio(samples, rate, remote_windows), rate)
    sf.write(str(session / "combined.wav"), samples, rate)
    payload = {"model": args.model, "device": "cpu",
               "audio_file": "combined.wav",
               "segments": [s.to_dict() for s in result.segments]}
    (session / "transcript.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Synthetic session: {session}", flush=True)
    print(f"  mic.wav: {len(you_windows)} You windows, "
          f"system.wav: {len(remote_windows)} Remote windows", flush=True)

    # 4. Run simple-mode diarization and score it.
    t0 = time.monotonic()
    diar_payload = diarize_session(session, mode="simple")
    wall = time.monotonic() - t0
    got = [s["speaker"] for s in diar_payload["segments"]]
    correct = total = 0
    print("\nPer-segment labels (ground truth -> simple mode):", flush=True)
    for seg, truth, label in zip(result.segments, truth_labels, got):
        mark = ""
        if truth is not None:
            total += 1
            correct += (truth == label)
            mark = "OK " if truth == label else "MISS"
        print(f"  [{seg.start:5.1f}-{seg.end:5.1f}] "
              f"{str(truth):6s} -> {label:7s} {mark} {seg.text[:60]}",
              flush=True)
    acc = correct / total if total else 0.0
    print(f"\nSimple-mode accuracy: {correct}/{total} = {acc:.1%} "
          f"(excl. {unassigned} unassigned segments)", flush=True)
    print(f"Diarization wall time: {wall:.2f}s for "
          f"{result.audio_duration_seconds:.1f}s of audio "
          f"(RTF {wall / result.audio_duration_seconds:.3f}), "
          f"peak RSS {peak_rss_mb():.0f} MB", flush=True)
    print(f"Method: {diar_payload['method']}", flush=True)

    # 5. Rename check: You -> John, then re-export.
    rename_speaker(session, "You", "John")
    reapply_names(session)
    txt = (session / "transcript.txt").read_text(encoding="utf-8")
    print("Rename check — transcript.txt first line: "
          f"{txt.splitlines()[0] if txt else '(empty)'}", flush=True)
    print(f"Session files: {sorted(p.name for p in session.iterdir())}",
          flush=True)
    return 0 if acc >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
