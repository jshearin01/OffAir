#!/usr/bin/env python3
"""Manual Phase 5 end-to-end test — FULL pipeline, real models.

This is NOT a pytest test: it runs the real faster-whisper tiny
model and the real fallback spectral extractor. Run it from the
repo root with the Phase 2 venv:

    .venv-phase2/bin/python tests/manual_e2e.py

Pipeline (on the 61.3 s, 3-voice TTS fixture):

    transcribe (tiny) -> diarize simple -> rename You -> John
    -> seed DB from isolated samples -> suggest -> learn
    -> LLM Markdown export -> local-LLM probe (Ollama, expected
    NOT running here)

Outputs go to --output-dir (default ~/tmp/offair-phase5) —
never into the repo. The script asserts the export contains the
speakers, timestamps and key content ("twelve thousand" / $12,000,
"December") and that no action items were invented. It exits
non-zero if any assertion fails.

Ollama is expected NOT to be running on this VM: the script probes
it, prints the guided "not running" error, and treats that as the
expected outcome — the real Ollama path is untested here and is
documented as such in PHASE5.md. llama-cpp-python is likewise not
installed here; its is_available() is probed, not faked.
"""
from __future__ import annotations

import argparse
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
from app.diarization.simple import load_wav_mono  # noqa: E402
from app.exports.llm_export import (  # noqa: E402
    export_llm_markdown_from_session,
)
from app.exports.local_llm import (  # noqa: E402
    LlamaCppClient,
    LocalLLMConfig,
    OllamaClient,
    OllamaNotAvailableError,
    summarize_locally,
)
from app.speakers.database import SpeakerDatabase  # noqa: E402
from app.speakers.embeddings import FallbackEmbeddingExtractor  # noqa: E402
from app.speakers.session import (  # noqa: E402
    learn_from_session,
    suggest_for_session,
    write_suggestions,
)
from app.transcription.engine import TranscriptionEngine  # noqa: E402
from tests.manual_diarize import (  # noqa: E402
    assign_turns,
    mask_audio,
    parse_script,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MEETING_WAV = FIXTURES / "meeting-3speakers-16k.wav"
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
                        / "tmp" / "offair-phase5")
    args = parser.parse_args(argv)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    total_t0 = time.monotonic()
    timings: dict[str, float] = {}

    # 1. Transcribe (tiny).
    print("== 1. Transcribe (Whisper tiny, CPU int8) ==", flush=True)
    engine = TranscriptionEngine(model_size="tiny")
    t0 = time.monotonic()
    result = engine.transcribe_file(MEETING_WAV)
    engine.unload()
    timings["transcribe"] = time.monotonic() - t0
    print(f"  {len(result.segments)} segments in "
          f"{result.wall_seconds:.1f}s (RTF "
          f"{result.real_time_factor:.3f}), peak RSS "
          f"{peak_rss_mb():.0f} MB", flush=True)
    for seg in result.segments[:3]:
        print(f"    [{seg.start:5.1f}-{seg.end:5.1f}] {seg.text[:70]}",
              flush=True)

    # 2. Synthetic dual-channel session (same construction as Phase 3).
    print("\n== 2. Synthetic dual-channel session ==", flush=True)
    turns = parse_script(TRUTH)
    truth_speakers = assign_turns(result.segments, turns)
    truth_labels = [("You" if s == "John" else
                     "Remote" if s else None) for s in truth_speakers]
    session = out_dir / "synthetic-session"
    session.mkdir(parents=True, exist_ok=True)
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
    (session / "metadata.json").write_text(json.dumps(
        {"session_id": session.name, "app_name": "Teams",
         "mode": "system-loopback-fallback", "status": "completed",
         "start_time": "2026-11-07T10:00:00+00:00",
         "duration_seconds": 61.3}, indent=2), encoding="utf-8")
    print(f"  session: {session} "
          f"({len(you_windows)} You windows, "
          f"{len(remote_windows)} Remote windows)", flush=True)

    # 3. Diarize simple.
    print("\n== 3. Diarize (simple, You vs Remote) ==", flush=True)
    t0 = time.monotonic()
    diar_payload = diarize_session(session, mode="simple")
    timings["diarize"] = time.monotonic() - t0
    got = [s["speaker"] for s in diar_payload["segments"]]
    correct = total = 0
    for truth, label in zip(truth_labels, got):
        if truth is not None:
            total += 1
            correct += (truth == label)
    print(f"  method: {diar_payload['method']}, "
          f"accuracy {correct}/{total} = "
          f"{correct / total:.1%}" if total else "  (no labelled segments)",
          flush=True)
    print(f"  wall: {timings['diarize']:.2f}s, peak RSS "
          f"{peak_rss_mb():.0f} MB", flush=True)

    # 4. Seed DB from isolated samples, then suggest.
    print("\n== 4. Seed DB + suggest (fallback extractor) ==", flush=True)
    extractor = FallbackEmbeddingExtractor()
    print(f"  extractor backend: {extractor.backend} (dim "
          f"{extractor.dim}) — spectral proxy, not a voice fingerprint",
          flush=True)
    db = SpeakerDatabase(out_dir / "speakers.db")
    for name, mp3 in ISOLATED.items():
        wav = mp3_to_wav(mp3, out_dir)
        emb = extractor.extract(wav)
        db.add_embedding(name, emb, backend=extractor.backend,
                         source_session="isolated-fixture",
                         source_label=name)
    t0 = time.monotonic()
    suggestions = suggest_for_session(session, db, extractor=extractor)
    timings["suggest"] = time.monotonic() - t0
    write_suggestions(session, suggestions)
    for sug in suggestions:
        print(f"    label {sug.label!r}: suggested "
              f"{sug.suggested_name!r} confidence "
              f"{sug.match.confidence:.3f} (best {sug.match.best_name} "
              f"{sug.match.best_score:.3f}, second "
              f"{sug.match.second_best_name} "
              f"{sug.match.second_best_score:.3f}, margin "
              f"{sug.match.margin:.3f}, ambiguous="
              f"{sug.match.is_ambiguous}, backend={sug.backend})",
              flush=True)
    print(f"  suggest wall: {timings['suggest']:.2f}s "
          f"(Remote skipped by design)", flush=True)

    # 5. Rename You -> John, then learn.
    print("\n== 5. Rename You -> John, learn from session ==", flush=True)
    rename_speaker(session, "You", "John")
    t0 = time.monotonic()
    learned = learn_from_session(session, db, extractor=extractor)
    timings["learn"] = time.monotonic() - t0
    print(f"  learned: {learned}", flush=True)
    for speaker in db.list_speakers():
        print(f"    {speaker['name']}: {speaker['sample_count']} sample(s), "
              f"backend {speaker['backend']}", flush=True)
    db.close()

    # 6. LLM Markdown export + assertions.
    print("\n== 6. LLM Markdown export ==", flush=True)
    t0 = time.monotonic()
    export_path = export_llm_markdown_from_session(session)
    timings["export"] = time.monotonic() - t0
    text = export_path.read_text(encoding="utf-8")
    print(f"  wrote {export_path} in {timings['export']:.2f}s",
          flush=True)
    print("  --- export (first 30 lines) ---", flush=True)
    for line in text.splitlines()[:30]:
        print(f"  {line}", flush=True)
    print("  --- end preview ---", flush=True)
    lowered = text.lower()
    checks = {
        "speaker John present": "john" in lowered,
        "speaker Remote present": "remote" in lowered,
        "timestamps present": "[00:00]" in text,
        "budget content present": ("twelve thousand" in lowered
                                   or "12,000" in text),
        "December content present": "december" in lowered,
        "no invented action items": "Action items" not in text,
        "context block present": "Context for summariser" in text,
    }
    failed = [k for k, ok in checks.items() if not ok]
    for name, ok in checks.items():
        print(f"  check {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    if failed:
        print(f"\nE2E FAILED: {failed}", flush=True)
        return 1

    # 7. Local LLM probe — expected NOT available on this VM.
    print("\n== 7. Local LLM probe (expected: not running here) ==",
          flush=True)
    ollama = OllamaClient()
    print(f"  Ollama is_available(): {ollama.is_available()}",
          flush=True)
    print(f"  llama-cpp is_available(): "
          f"{LlamaCppClient('/tmp/does-not-exist.gguf').is_available()}",
          flush=True)
    try:
        summarize_locally(
            "test transcript",
            config=LocalLLMConfig(enabled=True, provider="ollama"))
        print("  UNEXPECTED: summarize_locally succeeded", flush=True)
        return 1
    except OllamaNotAvailableError as exc:
        print(f"  summarize_locally raised OllamaNotAvailableError "
              f"(expected): {exc}", flush=True)
    except Exception as exc:  # noqa: BLE001 — any other error is a failure
        print(f"  UNEXPECTED error type: {type(exc).__name__}: {exc}",
              flush=True)
        return 1

    total = time.monotonic() - total_t0
    print(f"\nE2E OK in {total:.1f}s total "
          f"(transcribe {timings['transcribe']:.1f}s, diarize "
          f"{timings['diarize']:.2f}s, suggest {timings['suggest']:.2f}s, "
          f"learn {timings['learn']:.2f}s, export "
          f"{timings['export']:.2f}s), peak RSS {peak_rss_mb():.0f} MB",
          flush=True)
    print(f"Session files: {sorted(p.name for p in session.iterdir())}",
          flush=True)
    print(f"Outputs in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
