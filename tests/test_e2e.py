"""Phase 5 end-to-end test — full pipeline with FAKE models.

transcribe (fake engine) -> diarize simple (synthetic dual-channel
audio) -> rename You -> John -> learn/suggest (fake extractor) ->
LLM Markdown export. Asserts the export contains the speakers,
timestamps and key content, and that nothing is invented.

The REAL-model run is tests/manual_e2e.py (not run by pytest) —
it does the same pipeline with faster-whisper tiny and the real
fallback extractor on the 61 s fixture; see PHASE5.md for its
measured output.
"""
from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from app.diarization.names import rename_speaker
from app.diarization.session import diarize_session
from app.exports.llm_export import export_llm_markdown_from_session
from app.speakers.database import SpeakerDatabase
from app.speakers.session import (
    learn_from_session,
    suggest_for_session,
    write_suggestions,
)
from app.storage.sessions import create_session
from app.transcription.engine import Segment, TranscriptionResult


class _FakeExtractor:
    """One fixed vector per label — deterministic, no audio needed."""

    backend = "fake-test-v1"

    def __init__(self):
        self.vectors = {
            "You": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }

    def extract(self, wav_path, windows=None):
        # "You" always returns the John vector; anything else is noise.
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _write_wav(path, seconds, rate=16000, freq=440.0):
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), samples, rate)


def _write_envelope_wav(path, amplitudes, rate=16000, freq=440.0):
    """One second per amplitude entry — loud where the speaker talks."""
    parts = []
    for amp in amplitudes:
        t = np.arange(rate) / rate
        parts.append((amp * np.sin(2 * np.pi * freq * t)).astype(np.float32))
    sf.write(str(path), np.concatenate(parts), rate)


def test_full_pipeline_fake_models_produces_llm_export(tmp_path):
    # --- Build a synthetic dual-channel session (like manual_diarize).
    # mic loud 0–2 s (You), system loud 2–6 s (Remote).
    session = create_session(tmp_path, app_name="Teams")
    _write_envelope_wav(session / "mic.wav",
                        [0.5, 0.5, 0.01, 0.01, 0.01, 0.01])
    _write_envelope_wav(session / "system.wav",
                        [0.01, 0.01, 0.5, 0.5, 0.5, 0.5], freq=660.0)
    _write_wav(session / "combined.wav", 6.0, freq=440.0)
    segments = [
        Segment(0.0, 2.0, "Good morning everyone.", speaker=None),
        Segment(2.0, 4.0, "The budget is twelve thousand dollars.",
                speaker=None),
        Segment(4.0, 6.0, "The deadline is December fifteenth.",
                speaker=None),
    ]
    (session / "transcript.json").write_text(json.dumps(
        {"model": "tiny", "segments": [s.to_dict() for s in segments]},
        indent=2), encoding="utf-8")
    (session / "transcript.txt").write_text(
        "[00:00] Good morning everyone.\n", encoding="utf-8")

    # --- Diarize simple: mic-only first segment -> You, rest Remote.
    # (The synthetic tones are constant, so just check labels exist
    # and the pipeline writes diarization.json.)
    payload = diarize_session(session, mode="simple")
    assert (session / "diarization.json").exists()
    assert len(payload["segments"]) == 3
    assert [s["speaker"] for s in payload["segments"]] == [
        "You", "Remote", "Remote"]

    # --- Rename You -> John, learn + suggest with a fake extractor.
    rename_speaker(session, "You", "John")
    db = SpeakerDatabase(tmp_path / "speakers.db")
    extractor = _FakeExtractor()
    learned = learn_from_session(session, db, extractor=extractor)
    # Simple mode learns only "You" (Remote is a mixed cluster).
    assert set(learned) == {"You"}
    assert learned["You"]["name"] == "John"
    suggestions = suggest_for_session(session, db, extractor=extractor)
    assert len(suggestions) == 1
    assert suggestions[0].suggested_name == "John"
    assert suggestions[0].match.confidence >= 0.9
    write_suggestions(session, suggestions)
    assert (session / "speaker_suggestions.json").exists()
    db.close()

    # --- LLM Markdown export: speakers, timestamps, key content, no
    # invented action items.
    path = export_llm_markdown_from_session(session)
    text = path.read_text(encoding="utf-8")
    assert "## Context for summariser" in text
    assert "## Participants" in text
    assert "John" in text
    assert "[00:00]" in text and "[00:02]" in text
    assert "twelve thousand dollars" in text
    assert "December fifteenth" in text
    assert "Action items" not in text
    assert path.name == "transcript-llm.md"
