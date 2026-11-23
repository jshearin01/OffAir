"""People queries: sessions, utterances, samples.

Synthetic session folders only (no audio decoding, no Qt). A
speaker is matched by display name after speakers.json renames
are applied to the stored transcript labels, so a rename in one
session and a display name already stored in another both count.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.diarization.names import save_speaker_names
from app.speakers.database import SpeakerDatabase
from app.speakers.people import (
    get_speaker_detail,
    has_speaker_sample,
    speaker_sample,
    speaker_sessions,
    speaker_utterances,
)
from app.storage.sessions import create_session, update_metadata


def _make_session(root, segments, *, names=None, with_audio=True):
    session = create_session(root, app_name="Teams")
    update_metadata(session, duration_seconds=60.0)
    (session / "transcript.json").write_text(json.dumps(
        {"model": "tiny", "segments": segments}, indent=2),
        encoding="utf-8")
    if names:
        save_speaker_names(session, names)
    if with_audio:
        (session / "combined.wav").write_bytes(b"")
    return session


SEGMENTS_A = [
    {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00",
     "text": "Hello from Priya."},
    {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01",
     "text": "Marcus replies."},
    {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_00",
     "text": "Priya again, longer."},
]


def test_speaker_sessions_newest_first(tmp_path):
    older = _make_session(tmp_path, SEGMENTS_A,
                          names={"SPEAKER_00": "Priya"})
    newer = _make_session(tmp_path, SEGMENTS_A,
                          names={"SPEAKER_00": "Priya"})
    sessions = speaker_sessions(tmp_path, "Priya")
    assert [s.session_id for s in sessions] == [newer.name, older.name]
    assert sessions[0].utterance_count == 2
    assert sessions[0].path == newer


def test_speaker_sessions_honors_rename_in_one_session_only(tmp_path):
    # Session A stores the display name directly; session B stores
    # the raw label with a speakers.json rename. Both are Priya's.
    a = _make_session(tmp_path, [
        {"start": 0.0, "end": 3.0, "speaker": "Priya",
         "text": "Direct label."}])
    b = _make_session(tmp_path, SEGMENTS_A,
                      names={"SPEAKER_00": "Priya"})
    found = {s.session_id for s in speaker_sessions(tmp_path, "Priya")}
    assert found == {a.name, b.name}


def test_speaker_sessions_empty_for_unknown_speaker(tmp_path):
    _make_session(tmp_path, SEGMENTS_A, names={"SPEAKER_00": "Priya"})
    assert speaker_sessions(tmp_path, "Nobody") == []
    assert speaker_utterances(tmp_path, "Nobody") == []


def test_speaker_sessions_blank_name_raises(tmp_path):
    with pytest.raises(ValueError, match="blank"):
        speaker_sessions(tmp_path, "  ")


def test_utterances_are_display_named_and_ordered(tmp_path):
    _make_session(tmp_path, SEGMENTS_A, names={"SPEAKER_00": "Priya"})
    utterances = speaker_utterances(tmp_path, "priya")
    assert [(u.start, u.end) for u in utterances] == [(0.0, 4.0),
                                                      (10.0, 20.0)]
    assert all(u.speaker == "Priya" for u in utterances)
    assert utterances[0].text == "Hello from Priya."
    # The other speaker's turns are not included.
    assert "Marcus" not in " ".join(u.text for u in utterances)


def test_utterances_skip_sessions_without_transcript(tmp_path):
    session = create_session(tmp_path)
    update_metadata(session, duration_seconds=10.0)
    assert speaker_utterances(tmp_path, "Priya") == []
    assert speaker_sessions(tmp_path, "Priya") == []


def _add_sample(db, name, session, label):
    db.add_embedding(name, np.array([1.0, 0.0, 0.0], dtype=np.float32),
                     backend="test-backend", source_session=session,
                     source_label=label)


def test_sample_from_database_newest_session(tmp_path):
    old = _make_session(tmp_path, SEGMENTS_A,
                        names={"SPEAKER_00": "Priya"})
    new = _make_session(tmp_path, SEGMENTS_A,
                        names={"SPEAKER_00": "Priya"})
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        _add_sample(db, "Priya", old.name, "SPEAKER_00")
        _add_sample(db, "Priya", new.name, "SPEAKER_00")
        sample = speaker_sample(tmp_path, db, "Priya")
        assert sample is not None
        assert sample.session_id == new.name
        # Longest turn rule: the 10 s turn, from combined.wav.
        assert (sample.start, sample.end) == (10.0, 20.0)
        assert sample.audio_path == new / "combined.wav"
        assert has_speaker_sample(tmp_path, db, "Priya") is True
    finally:
        db.close()


def test_sample_falls_back_to_session_scan_without_database_entry(
        tmp_path):
    session = _make_session(tmp_path, SEGMENTS_A,
                            names={"SPEAKER_00": "Priya"})
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        # No embedding stored for Priya at all: the sample still
        # comes from her longest turn in the session itself.
        sample = speaker_sample(tmp_path, db, "Priya")
        assert sample is not None
        assert sample.session_id == session.name
        assert (sample.start, sample.end) == (10.0, 20.0)
    finally:
        db.close()


def test_sample_none_when_audio_is_gone(tmp_path):
    _make_session(tmp_path, SEGMENTS_A, names={"SPEAKER_00": "Priya"},
                  with_audio=False)
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        assert speaker_sample(tmp_path, db, "Priya") is None
        assert has_speaker_sample(tmp_path, db, "Priya") is False
    finally:
        db.close()


def test_sample_prefers_mic_for_you_and_system_for_remote(tmp_path):
    session = _make_session(tmp_path, [
        {"start": 0.0, "end": 4.0, "speaker": "You", "text": "hi"}],
        with_audio=False)
    (session / "mic.wav").write_bytes(b"")
    (session / "combined.wav").write_bytes(b"")
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        sample = speaker_sample(tmp_path, db, "You")
        assert sample is not None
        assert sample.audio_path == session / "mic.wav"
    finally:
        db.close()


def test_get_speaker_detail_shape(tmp_path):
    session = _make_session(tmp_path, SEGMENTS_A,
                            names={"SPEAKER_00": "Priya"})
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        _add_sample(db, "Priya", session.name, "SPEAKER_00")
        detail = get_speaker_detail(tmp_path, db, "Priya")
        assert detail["name"] == "Priya"
        assert detail["known_in_database"] is True
        assert detail["session_count"] == 1
        assert detail["utterance_count"] == 2
        assert detail["sample_available"] is True
        assert detail["sample"]["session_id"] == session.name
        assert detail["sample"]["audio_path"] == str(
            session / "combined.wav")
        assert detail["sessions"][0]["session_id"] == session.name
        assert detail["utterances"][0]["text"] == "Hello from Priya."
    finally:
        db.close()


def test_get_speaker_detail_unknown_speaker(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        detail = get_speaker_detail(tmp_path, db, "Nobody")
        assert detail["known_in_database"] is False
        assert detail["sessions"] == []
        assert detail["utterances"] == []
        assert detail["sample"] is None
        assert detail["sample_available"] is False
    finally:
        db.close()
