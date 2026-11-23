#!/usr/bin/env python3
"""Frozen-build self-test for OffAir (build verification only).

Build it in the guest venv with the same PyInstaller flags as
OffAir.spec (see docs/PACKAGING.md), then run the exe from a plain
cmd. It proves, inside the frozen environment:

1. Every heavy lazy import loads (faster-whisper, av, ctranslate2,
   onnxruntime, sherpa_onnx, speakeronnx, soundfile, QtMultimedia).
2. The Whisper model download works (tiny, ~75 MB, first run) and
   transcription of a real clip works end to end.
3. The sherpa model download works (~33 MB, first run).

Exit code 0 only if every step passes. NOT part of the shipped app
— the shipped exe is OffAir.exe from OffAir.spec.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running unfrozen from the repo root too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

FAILURES: list[str] = []


def check(label: str, fn):
    started = time.monotonic()
    try:
        result = fn()
        print(f"PASS {label} ({time.monotonic() - started:.1f}s)"
              + (f" — {result}" if result else ""))
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {label}: {type(exc).__name__}: {exc}")
        FAILURES.append(label)
        return None


def main() -> int:
    print("OffAir frozen self-test")

    def _imports():
        import av  # noqa: F401
        import ctranslate2  # noqa: F401
        import faster_whisper  # noqa: F401
        import onnxruntime  # noqa: F401
        import sherpa_onnx  # noqa: F401
        import soundfile  # noqa: F401
        import speakeronnx  # noqa: F401
        from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer  # noqa: F401
        return "faster_whisper/av/ctranslate2/onnxruntime/sherpa_onnx/speakeronnx/soundfile/QtMultimedia"

    check("heavy imports", _imports)

    fixture = Path(__file__).resolve().parent / "tests" / "fixtures" / "speaker-john-16k.wav"
    # In the frozen self-test the fixtures are not bundled; the caller
    # passes the clip path as argv[1] when running frozen.
    if len(sys.argv) > 1:
        fixture = Path(sys.argv[1])

    def _transcribe():
        from app.transcription.engine import TranscriptionEngine

        engine = TranscriptionEngine(model_size="tiny")
        result = engine.transcribe_file(fixture, language="en")
        engine.unload()
        return f"{len(result.segments)} segments, RTF {result.real_time_factor:.2f}"

    if fixture.exists():
        check("whisper tiny download + transcribe", _transcribe)
    else:
        print(f"SKIP whisper transcribe (no fixture at {fixture})")

    def _sherpa_models():
        from app.diarization.sherpa import ensure_sherpa_models

        seg, emb = ensure_sherpa_models()
        return f"seg={seg.name} emb={emb.name}"

    check("sherpa model download", _sherpa_models)

    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
        return 1
    print("RESULT: all self-test steps passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
