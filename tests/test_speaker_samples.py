"""Voice-sample lookup for the Speakers tab (rename + play sample).

A sample is a real stretch of recorded audio — the speaker's longest
turn — played through the same QMediaPlayer as transcript
click-to-play. Nothing here stores audio: the database keeps only
embeddings plus the source session + label, so a sample is found by
going back to that session's audio file.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from app.speakers.database import SpeakerDatabase
from app.speakers.session import (
    find_sample_for_speaker,
    sample_window_for_label,
)

REPO = Path(__file__).resolve().parent.parent
FIXTURE_WAV = REPO / "tests" / "fixtures" / "meeting-3speakers-16k.wav"

SEGMENTS = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "hello"},
    {"start": 10.0, "end": 13.0, "speaker": "SPEAKER_00", "text": "again"},
    {"start": 30.0, "end": 38.0, "speaker": "SPEAKER_01", "text": "other"},
]


def _make_session(root: Path, name: str, *,
                  segments: list[dict] | None = SEGMENTS,
                  with_audio: bool = True) -> Path:
    session = root / name
    session.mkdir(parents=True)
    if segments is not None:
        (session / "transcript.json").write_text(
            json.dumps({"segments": segments}), encoding="utf-8")
    if with_audio:
        shutil.copyfile(FIXTURE_WAV, session / "combined.wav")
    return session


def _add_sample(db: SpeakerDatabase, name: str, session: str, label: str):
    db.add_embedding(name, np.array([1.0, 0.0, 0.0], dtype=np.float32),
                     backend="test-backend", source_session=session,
                     source_label=label)


# ------------------------------------------------- sample_window_for_label ---
def test_sample_window_is_the_longest_turn(tmp_path):
    session = _make_session(tmp_path, "s1")
    audio, start, end = sample_window_for_label(session, "SPEAKER_00")
    assert audio == session / "combined.wav"
    assert (start, end) == (0.0, 5.0)


def test_sample_window_none_for_unknown_label(tmp_path):
    session = _make_session(tmp_path, "s1")
    assert sample_window_for_label(session, "SPEAKER_99") is None


def test_sample_window_none_without_transcript(tmp_path):
    session = _make_session(tmp_path, "s1", segments=None)
    assert sample_window_for_label(session, "SPEAKER_00") is None


def test_sample_window_none_without_audio(tmp_path):
    session = _make_session(tmp_path, "s1", with_audio=False)
    assert sample_window_for_label(session, "SPEAKER_00") is None


def test_sample_window_prefers_mic_for_you(tmp_path):
    session = _make_session(tmp_path, "s1", segments=[
        {"start": 0.0, "end": 4.0, "speaker": "You", "text": "hi"}])
    shutil.copyfile(FIXTURE_WAV, session / "mic.wav")
    audio, start, end = sample_window_for_label(session, "You")
    assert audio == session / "mic.wav"
    assert (start, end) == (0.0, 4.0)


# ------------------------------------------------------------ speaker_samples ---
def test_speaker_samples_newest_first(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        _add_sample(db, "John", "session-old", "SPEAKER_00")
        _add_sample(db, "John", "session-new", "SPEAKER_01")
        samples = db.speaker_samples("John")
        assert [s["source_session"] for s in samples] == [
            "session-new", "session-old"]
        assert samples[0]["source_label"] == "SPEAKER_01"
    finally:
        db.close()


def test_speaker_samples_empty_for_unknown(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        assert db.speaker_samples("Nobody") == []
    finally:
        db.close()


# -------------------------------------------------------- find_sample_for_speaker ---
def test_find_sample_prefers_newest_session_with_audio(tmp_path):
    old = _make_session(tmp_path, "session-old")
    new = _make_session(tmp_path, "session-new")
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        _add_sample(db, "John", "session-old", "SPEAKER_00")
        _add_sample(db, "John", "session-new", "SPEAKER_01")
        audio, start, end, session = find_sample_for_speaker(
            tmp_path, db, "John")
        assert session == "session-new"
        assert audio == new / "combined.wav"
        assert (start, end) == (30.0, 38.0)
    finally:
        db.close()


def test_find_sample_falls_back_when_newest_audio_is_gone(tmp_path):
    _make_session(tmp_path, "session-old")
    _make_session(tmp_path, "session-new", with_audio=False)
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        _add_sample(db, "John", "session-old", "SPEAKER_00")
        _add_sample(db, "John", "session-new", "SPEAKER_01")
        found = find_sample_for_speaker(tmp_path, db, "John")
        assert found is not None
        assert found[3] == "session-old"
        assert (found[1], found[2]) == (0.0, 5.0)
    finally:
        db.close()


def test_find_sample_none_when_everything_is_gone(tmp_path):
    _make_session(tmp_path, "session-old", with_audio=False)
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        _add_sample(db, "John", "session-old", "SPEAKER_00")
        _add_sample(db, "John", "session-missing", "SPEAKER_00")
        assert find_sample_for_speaker(tmp_path, db, "John") is None
    finally:
        db.close()


def test_find_sample_matches_renamed_display_name(tmp_path):
    # The sample was learned as SPEAKER_00, but the session's
    # segments now carry the display name "John" (rename applied).
    session = _make_session(tmp_path, "s1", segments=[
        {"start": 2.0, "end": 9.0, "speaker": "John", "text": "hello"}])
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        _add_sample(db, "John", "s1", "SPEAKER_00")
        found = find_sample_for_speaker(tmp_path, db, "John")
        assert found is not None
        assert found[0] == session / "combined.wav"
        assert (found[1], found[2]) == (2.0, 9.0)
    finally:
        db.close()
