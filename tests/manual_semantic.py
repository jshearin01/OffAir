"""Manual real-model semantic search check (not part of pytest).

Builds one session from the real fixture script
(tests/fixtures/meeting-script.txt), indexes it with the real
ONNX MiniLM model (downloaded once into ~/.offair/models/),
and prints top hits + scores + timings + memory for the
queries the PRD example cares about: "money", "budget",
"deadline", "timeline".

Run:  .venv-phase2/bin/python tests/manual_semantic.py
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.storage.semantic_search import (
    MiniLMEmbedder,
    SemanticSearchIndex,
)
from app.storage.sessions import create_session, update_metadata
from app.transcription.engine import Segment

FIXTURE = Path(__file__).parent / "fixtures" / "meeting-script.txt"


def main() -> int:
    lines = [ln.strip() for ln in FIXTURE.read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    segments = []
    cursor = 0.0
    for line in lines:
        speaker, _, text = line.partition(": ")
        segments.append(Segment(cursor, cursor + 6.0,
                                text.strip(), speaker or None))
        cursor += 6.0

    root = Path("/tmp/offair-manual-semantic")
    session = create_session(root, app_name="Teams")
    update_metadata(session, duration_seconds=cursor)
    (session / "transcript.json").write_text(json.dumps(
        {"model": "tiny",
         "segments": [s.to_dict() for s in segments]},
        indent=2), encoding="utf-8")
    print(f"Session: {session.name} ({len(segments)} segments, "
          f"{cursor:.0f}s)")

    embedder = MiniLMEmbedder()
    t0 = time.monotonic()
    index = SemanticSearchIndex(root, embedder=embedder)
    counts = index.ensure_indexed()
    t_index = time.monotonic() - t0
    model_dir = embedder.model_dir
    model_files = sorted(model_dir.glob("*.onnx"))
    model_mb = sum(p.stat().st_size for p in model_files) / 1e6
    vocab_kb = (model_dir / "vocab.txt").stat().st_size / 1e3
    print(f"Index built: {counts} in {t_index:.1f}s "
          f"(incl. first-use model download + load)")
    print(f"Model dir: {model_dir} — "
          f"{[p.name for p in model_files]} {model_mb:.1f} MB, "
          f"vocab {vocab_kb:.0f} KB")
    print(f"Chunks indexed: {index.chunk_count()}")

    for query in ("money", "budget", "deadline", "timeline"):
        t0 = time.monotonic()
        hits = index.semantic_search(query, top_k=3)
        dt = time.monotonic() - t0
        print(f"\nQuery {query!r} ({dt * 1000:.0f} ms):")
        for hit in hits:
            print(f"  [{hit.score:.3f}] {hit.session_id} "
                  f"[{hit.start_s:.0f}s–{hit.end_s:.0f}s] "
                  f"{hit.snippet[:160]}")
        if not hits:
            print("  (no hits above threshold)")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\nPeak RSS: {peak_rss:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
