"""Transcription (PRD Phase 2) — faster-whisper, CPU-only, int8.

No CUDA / GPU anywhere. The engine is importable without faster-whisper
installed: :class:`TranscriptionEngine` raises a clear error only when a
real model is loaded. Unit tests inject a fake model.
"""
from __future__ import annotations

from app.transcription.engine import Segment, TranscriptionEngine
from app.transcription.exports import (
    export_srt,
    export_txt,
    load_transcript_json,
    segments_from_dicts,
    segments_to_dicts,
)

__all__ = [
    "Segment",
    "TranscriptionEngine",
    "export_srt",
    "export_txt",
    "load_transcript_json",
    "segments_from_dicts",
    "segments_to_dicts",
]
