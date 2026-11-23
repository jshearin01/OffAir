"""Bookmarks: stored in metadata.json, audio honesty.

Synthetic session folders only. A bookmark never stores audio
state: whether its audio still exists is computed when listed,
so retention deleting the WAVs is reflected immediately.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from app.storage.bookmarks import (
    add_bookmark,
    has_playable_audio,
    list_bookmarks,
)
from app.storage.sessions import create_session, load_metadata, update_metadata


def _session(tmp_path, *, with_audio=True):
    session = create_session(tmp_path, app_name="Teams")
    if with_audio:
        (session / "combined.wav").write_bytes(b"")
    return session


def test_add_and_list_round_trip_stores_metadata_shape(tmp_path):
    session = _session(tmp_path)
    entry = add_bookmark(session, 42.5, "Budget decision")
    assert entry == {"t": 42.5, "label": "Budget decision",
                     "created": entry["created"]}
    # The on-disk shape is exactly the documented one.
    raw = json.loads((session / "metadata.json").read_text(
        encoding="utf-8"))
    assert raw["bookmarks"] == [entry]
    assert set(entry) == {"t", "label", "created"}
    # "created" is an ISO timestamp.
    datetime.fromisoformat(entry["created"])
    listed = list_bookmarks(session)
    assert len(listed) == 1
    assert listed[0].t == 42.5
    assert listed[0].label == "Budget decision"
    assert listed[0].audio_available is True


def test_list_is_sorted_by_time(tmp_path):
    session = _session(tmp_path)
    add_bookmark(session, 90.0, "late")
    add_bookmark(session, 10.0, "early")
    add_bookmark(session, 50.0, "middle")
    assert [b.label for b in list_bookmarks(session)] == [
        "early", "middle", "late"]


def test_label_defaults_to_empty_and_t_must_be_valid(tmp_path):
    session = _session(tmp_path)
    entry = add_bookmark(session, 0)
    assert entry["label"] == ""
    assert entry["t"] == 0.0
    with pytest.raises(ValueError, match="bookmark time"):
        add_bookmark(session, -1.0)
    with pytest.raises(ValueError, match="bookmark time"):
        add_bookmark(session, float("nan"))
    with pytest.raises(ValueError, match="bookmark time"):
        add_bookmark(session, "ten")


def test_list_empty_when_no_bookmarks(tmp_path):
    session = _session(tmp_path)
    assert list_bookmarks(session) == []


def test_audio_available_goes_false_when_wavs_are_deleted(tmp_path):
    session = _session(tmp_path)
    add_bookmark(session, 5.0, "mark")
    assert list_bookmarks(session)[0].audio_available is True
    # Simulate retention pruning the audio: the stored bookmark
    # stays, but listing now says the audio is gone.
    (session / "combined.wav").unlink()
    assert has_playable_audio(session) is False
    listed = list_bookmarks(session)
    assert listed[0].audio_available is False
    assert listed[0].label == "mark"


def test_bookmarks_survive_other_metadata_updates(tmp_path):
    session = _session(tmp_path)
    add_bookmark(session, 7.0, "keep me")
    update_metadata(session, pinned=True)
    metadata = load_metadata(session)
    assert len(metadata["bookmarks"]) == 1
    assert metadata["pinned"] is True
    assert list_bookmarks(session)[0].label == "keep me"


def test_add_bookmark_without_metadata_raises(tmp_path):
    session = tmp_path / "no-metadata"
    session.mkdir()
    with pytest.raises(FileNotFoundError):
        add_bookmark(session, 1.0, "x")
