#!/usr/bin/env python3
"""Manual Phase 2 test — run the REAL faster-whisper model on audio.

This is NOT a pytest test: it downloads a Whisper model on first use and
needs real CPU time. Run it from the repo root with the Phase 2 venv:

    .venv-phase2/bin/python tests/manual_transcribe.py --model tiny
    .venv-phase2/bin/python tests/manual_transcribe.py --all

It prints wall time, real-time factor, peak RSS, the full transcript,
and a word-error-rate (WER) against the ground-truth script when one is
given. Outputs (transcript.json/.txt/.srt) go to --output-dir, defaulting
to a temp folder — never into the repo, and never alongside the fixtures.
"""
from __future__ import annotations

import argparse
import re
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.transcription.engine import TranscriptionEngine  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DEFAULT_AUDIO = FIXTURES / "meeting-3speakers-16k.wav"
DEFAULT_TRUTH = FIXTURES / "meeting-script.txt"


def normalize_words(text: str) -> list[str]:
    """Lowercase, strip punctuation and speaker labels, split to words."""
    text = re.sub(r"^(John|Priya|Marcus):\s*", "", text, flags=re.M)
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return text.split()


def word_error_rate(reference: list[str], hypothesis: list[str]) -> dict:
    """Levenshtein WER on word lists. Returns rate + S/D/I counts."""
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return {"wer": 1.0 if m else 0.0, "sub": 0, "del": 0, "ins": m}
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    # Backtrack for S/D/I counts.
    i, j, sub, dele, ins = n, m, 0, 0, 0
    # Recompute the full matrix for the backtrack.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for a in range(n + 1):
        dp[a][0] = a
    for b in range(m + 1):
        dp[0][b] = b
    for a in range(1, n + 1):
        for b in range(1, m + 1):
            cost = 0 if reference[a - 1] == hypothesis[b - 1] else 1
            dp[a][b] = min(dp[a - 1][b] + 1, dp[a][b - 1] + 1,
                           dp[a - 1][b - 1] + cost)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            sub, i, j = sub + 1, i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dele, i = dele + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return {"wer": (sub + dele + ins) / n, "sub": sub, "del": dele,
            "ins": ins, "ref_words": n, "hyp_words": m}


def peak_rss_mb() -> float:
    # ru_maxrss is KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_one(model: str, audio: Path, output_dir: Path,
            truth: Path | None) -> dict:
    engine = TranscriptionEngine(model_size=model)
    progress_marks: list[float] = []

    def on_progress(frac: float, elapsed: float, remaining: float | None):
        progress_marks.append(frac)

    print(f"\n=== model={model} audio={audio.name} ===", flush=True)
    result = engine.transcribe_file(
        audio, progress_callback=on_progress)
    engine.unload()
    rtf = result.real_time_factor
    print(f"wall={result.wall_seconds:.1f}s  audio="
          f"{result.audio_duration_seconds:.1f}s  "
          f"RTF={rtf:.3f}  peak RSS={peak_rss_mb():.0f} MB  "
          f"segments={len(result.segments)}  "
          f"progress callbacks={len(progress_marks)}", flush=True)
    for s in result.segments:
        print(f"  [{s.start:6.1f}-{s.end:6.1f}] {s.text}", flush=True)

    wer = None
    if truth is not None and truth.exists():
        ref = normalize_words(truth.read_text(encoding="utf-8"))
        hyp = normalize_words(result.text)
        wer = word_error_rate(ref, hyp)
        print(f"WER vs ground truth: {wer['wer']:.1%} "
              f"(sub={wer['sub']} del={wer['del']} ins={wer['ins']} "
              f"of {wer['ref_words']} ref words)", flush=True)

    out = output_dir / model
    out.mkdir(parents=True, exist_ok=True)
    from app.transcription.exports import export_srt, export_txt
    import json
    (out / "transcript.json").write_text(json.dumps(
        {"model": model, "wall_seconds": round(result.wall_seconds, 2),
         "real_time_factor": round(rtf, 3) if rtf else None,
         "peak_rss_mb": round(peak_rss_mb()),
         "wer": wer,
         "segments": [s.to_dict() for s in result.segments]},
        indent=2), encoding="utf-8")
    export_txt(result.segments, out / "transcript.txt")
    export_srt(result.segments, out / "transcript.srt")
    print(f"outputs: {out}", flush=True)
    return {"model": model, "wall_seconds": result.wall_seconds,
            "rtf": rtf, "peak_rss_mb": peak_rss_mb(),
            "segments": len(result.segments), "wer": wer}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--model", default="tiny",
                        choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--all", action="store_true",
                        help="Run tiny, base, small in sequence.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--no-ground-truth", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.audio.exists():
        print(f"Audio not found: {args.audio}\nConvert the fixture first:\n"
              f"  ffmpeg -y -i tests/fixtures/meeting-3speakers.mp3 "
              f"-ar 16000 -ac 1 tests/fixtures/meeting-3speakers-16k.wav",
              file=sys.stderr)
        return 2
    truth = None if args.no_ground_truth else args.ground_truth
    out_root = args.output_dir or Path(
        f"/tmp/offair-manual-{time.strftime('%Y%m%d-%H%M%S')}")
    models = ["tiny", "base", "small"] if args.all else [args.model]
    results = []
    for model in models:
        results.append(run_one(model, args.audio, out_root, truth))
    print("\n=== summary ===")
    for r in results:
        wer = f" WER={r['wer']['wer']:.1%}" if r["wer"] else ""
        print(f"  {r['model']:6s} wall={r['wall_seconds']:6.1f}s "
              f"RTF={r['rtf']:.3f} peak RSS={r['peak_rss_mb']:.0f} MB"
              f"{wer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
