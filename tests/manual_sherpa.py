"""Manual sherpa verification — real model, NOT run by pytest.

Usage (sherpa-onnx must be importable — use the tryouts venv):
    ~/workspace/meettrace-tryouts/sherpa-venv/bin/python \\
        tests/manual_sherpa.py [model_dir]

``model_dir`` defaults to the tryout models
(~/workspace/meettrace-tryouts/sherpa-models), which already holds
both models — no download. Pass another dir to exercise the
download path (needs network; models are public GitHub releases,
no token).

Runs the new app.diarization.sherpa.SherpaDiarizer on the
3-speaker fixture (61.3 s, 9 turns: John, Priya, Marcus x3) with
threshold clustering (default 0.5, plus 0.6) and with the
expected-speakers=3 hint, identity-checks every turn with the new
SherpaEres2NetEmbeddingExtractor against the isolated references,
and prints wall time, RTF, peak RSS and model sizes.
"""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.diarization.sherpa import (  # noqa: E402
    SherpaDiarizer,
    check_availability,
    ensure_sherpa_models,
)
from app.speakers.embeddings import (  # noqa: E402
    SherpaEres2NetEmbeddingExtractor,
)

FIXTURE = REPO / "tests" / "fixtures" / "meeting-3speakers-16k.wav"
TRYOUT_MODELS = Path.home() / "workspace" / "meettrace-tryouts" / "sherpa-models"
TRUTH = ["john", "priya", "marcus"] * 3


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else TRYOUT_MODELS
    print(f"model dir: {model_dir}")
    print(f"availability: {check_availability(model_dir=model_dir)}")
    seg, emb = ensure_sherpa_models(model_dir=model_dir)
    print(f"segmentation: {seg} ({seg.stat().st_size / 1e6:.1f} MB)")
    print(f"embedding:    {emb} ({emb.stat().st_size / 1e6:.1f} MB)")
    print(f"fixture: {FIXTURE} (61.3 s, 9 turns)")

    extractor = SherpaEres2NetEmbeddingExtractor(model_path=emb)
    refs = {}
    for name in ("john", "priya", "marcus"):
        refs[name] = extractor.extract(
            FIXTURE.parent / f"speaker-{name}-16k.wav")
    print("reference embeddings loaded (ERes2Net, 192-dim)")

    def identity_check(result) -> int:
        correct = 0
        for (start, end, _label), expected in zip(result.turns, TRUTH):
            vec = extractor.extract(FIXTURE, windows=[(start, end)])
            scores = {n: float(vec @ r) for n, r in refs.items()}
            best = max(scores, key=scores.get)
            ok = best == expected
            correct += int(ok)
            print(f"  {start:6.2f}-{end:6.2f} {_label} -> {best:6s} "
                  f"(J={scores['john']:.2f} P={scores['priya']:.2f} "
                  f"M={scores['marcus']:.2f}) truth={expected} "
                  f"{'OK' if ok else 'WRONG'}")
        return correct

    for label, kwargs in (
            ("threshold 0.5 (default)", {}),
            ("threshold 0.6", {"clustering_threshold": 0.6}),
            ("expected_speakers=3 hint", {"expected_speakers": 3})):
        t0 = time.monotonic()
        diarizer = SherpaDiarizer(model_dir=model_dir, **kwargs)
        result = diarizer.diarize(FIXTURE)
        wall = time.monotonic() - t0
        rtf = result.wall_seconds / (result.audio_duration_seconds or 1)
        print(f"\n== {label} ==  pipeline+run {wall:.1f}s, "
              f"diarize wall {result.wall_seconds:.1f}s, RTF {rtf:.3f}, "
              f"{len(result.turns)} turns, {len(result.speakers)} speakers "
              f"({result.speakers}), clustering={result.clustering}")
        correct = identity_check(result)
        print(f"  identity: {correct}/{len(result.turns)}")
        assert len(result.turns) == 9 and len(result.speakers) == 3, \
            f"unexpected shape: {result.turns}"

    print(f"\nPeak RSS (whole process, incl. identity checks): "
          f"{_rss_mb():.0f} MB")
    print("MANUAL SHERPA VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
