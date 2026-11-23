"""Phase 2 transcription tests — fake Whisper model only.

No model download, no hardware, no network. The real model is exercised
only by tests/manual_transcribe.py, never by this pytest file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import WHISPER_DEFAULT_MODEL, WHISPER_MODELS
from app.transcription.engine import (
    Segment,
    TranscriptionEngine,
    TranscriptionResult,
    audio_duration_seconds,
)
from app.transcription.exports import (
    export_srt,
    export_txt,
    format_srt_timestamp,
    format_txt_timestamp,
    load_transcript_json,
    segments_from_dicts,
    segments_to_dicts,
)


class _FakeRaw:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _FakeModel:
    """Mimics faster-whisper's WhisperModel.transcribe() interface."""

    def __init__(self, raws):
        self.raws = raws
        self.calls = []

    def transcribe(self, path, language="en", beam_size=5):
        self.calls.append((path, language))
        return iter(self.raws), {"language": language}


def _engine():
    raws = [_FakeRaw(0.0, 2.5, "Hello everyone."),
            _FakeRaw(2.5, 5.0, "Thanks for joining the review.")]
    return TranscriptionEngine(model_size="tiny", model=_FakeModel(raws))


# ------------------------------------------------------------------ engine --
def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="unknown Whisper model"):
        TranscriptionEngine(model_size="huge")


def test_gpu_device_rejected():
    with pytest.raises(ValueError, match="CPU-only"):
        TranscriptionEngine(model_size="tiny", device="cuda")


def test_missing_audio_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _engine().transcribe_file(tmp_path / "nope.wav")


def test_transcribe_file_returns_segments_and_progress(tmp_path):
    import wave
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 6)  # 6 s of silence
    seen = []
    result = _engine().transcribe_file(
        wav, progress_callback=lambda f, e, r: seen.append((f, e, r)))
    assert isinstance(result, TranscriptionResult)
    assert [s.text for s in result.segments] == [
        "Hello everyone.", "Thanks for joining the review."]
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 2.5
    assert result.model == "tiny"
    assert result.audio_duration_seconds == pytest.approx(6.0)
    assert result.real_time_factor is not None
    assert seen and seen[-1][0] == 1.0  # progress ends at 100%
    assert all(0.0 <= f <= 1.0 for f, _, _ in seen)


def test_transcribe_session_writes_json_txt_srt(tmp_path):
    import wave
    session = tmp_path / "2026-11-07_10-00-00_teams"
    session.mkdir()
    wav = session / "combined.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 6)
    result = _engine().transcribe_session(session)
    assert len(result.segments) == 2
    payload = json.loads((session / "transcript.json").read_text())
    assert payload["model"] == "tiny"
    assert payload["device"] == "cpu"
    assert payload["compute_type"] == "int8"
    assert payload["audio_file"] == "combined.wav"
    assert len(payload["segments"]) == 2
    assert payload["segments"][0]["text"] == "Hello everyone."
    assert (session / "transcript.txt").exists()
    assert (session / "transcript.srt").exists()
    # Round-trip through the loader.
    assert [s.text for s in load_transcript_json(
        session / "transcript.json")] == [s.text for s in result.segments]


def test_transcribe_session_falls_back_to_mic(tmp_path):
    import wave
    session = tmp_path / "s"
    session.mkdir()
    wav = session / "mic.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 2)
    result = _engine().transcribe_session(session)
    assert result.audio_path == wav


def test_transcribe_session_no_audio_raises(tmp_path):
    session = tmp_path / "empty"
    session.mkdir()
    with pytest.raises(FileNotFoundError, match="combined.wav"):
        _engine().transcribe_session(session)


def test_audio_duration_unknown_for_non_wav(tmp_path):
    assert audio_duration_seconds(tmp_path / "x.mp3") is None


# ----------------------------------------------------------------- exports --
def test_srt_timestamp_format():
    assert format_srt_timestamp(0) == "00:00:00,000"
    assert format_srt_timestamp(61.5) == "00:01:01,500"
    assert format_srt_timestamp(3661.234) == "01:01:01,234"


def test_txt_timestamp_format():
    assert format_txt_timestamp(3) == "[00:03]"
    assert format_txt_timestamp(61) == "[01:01]"
    assert format_txt_timestamp(3661) == "[01:01:01]"


def test_export_txt(tmp_path):
    segs = [Segment(0.0, 2.5, "Hello."), Segment(65.0, 67.0, "Second.")]
    path = export_txt(segs, tmp_path / "t.txt")
    assert path.read_text() == "[00:00] Hello.\n[01:05] Second.\n"


def test_export_srt(tmp_path):
    segs = [Segment(0.0, 2.5, "Hello."), Segment(2.5, 5.0, "World.")]
    path = export_srt(segs, tmp_path / "t.srt")
    assert path.read_text() == (
        "1\n00:00:00,000 --> 00:00:02,500\nHello.\n\n"
        "2\n00:00:02,500 --> 00:00:05,000\nWorld.\n")


def test_export_empty_segments(tmp_path):
    assert export_txt([], tmp_path / "t.txt").read_text() == ""
    assert export_srt([], tmp_path / "t.srt").read_text() == ""


def test_segment_json_round_trip():
    segs = [Segment(0.5, 1.25, "Hi there.")]
    assert segments_from_dicts(segments_to_dicts(segs)) == segs


def test_all_config_models_have_required_fields():
    for name, info in WHISPER_MODELS.items():
        assert info["params"] and info["size_mb"] > 0
        assert info["speed"] and info["accuracy"]
        assert info["expected_30min_seconds"] > 0
    assert "small" in WHISPER_MODELS  # opt-in; "base" is the default
    assert WHISPER_DEFAULT_MODEL == "base"
