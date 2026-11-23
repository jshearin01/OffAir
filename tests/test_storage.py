"""Tests for app.storage.sessions — session folders + metadata."""
import json
import re

import pytest

from app.storage.sessions import (
    create_session,
    finalize_session,
    list_sessions,
    load_metadata,
    update_metadata,
)


def test_create_session_folder_name_and_metadata(tmp_path):
    session = create_session(tmp_path, app_name="Teams")
    assert session.is_dir()
    assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_teams",
                    session.name)
    metadata = load_metadata(session)
    assert metadata["app_name"] == "Teams"
    assert metadata["status"] == "recording"
    assert metadata["start_time"]
    assert metadata["sample_rate"] == 48000
    assert metadata["files"]["mic"] == "mic.wav"
    assert metadata["pause_policy"] == "discard"


def test_create_session_collision_gets_suffix(tmp_path):
    first = create_session(tmp_path, app_name="Teams")
    # Same second, same slug: create a second one and check it differs.
    second = create_session(tmp_path, app_name="Teams")
    assert second != first
    assert second.is_dir()
    assert load_metadata(second)["session_id"] == second.name


def test_metadata_round_trip_and_update(tmp_path):
    session = create_session(tmp_path, app_name="Teams", mode="test-mode",
                             warning="test warning")
    updated = update_metadata(session, mode="other-mode", extra="x")
    assert updated["mode"] == "other-mode"
    assert updated["extra"] == "x"
    assert load_metadata(session)["mode"] == "other-mode"
    raw = json.loads((session / "metadata.json").read_text())
    assert raw["warning"] == "test warning"


def test_finalize_session(tmp_path):
    session = create_session(tmp_path)
    (session / "mic.wav").write_bytes(b"fake")
    (session / "system.wav").write_bytes(b"fake")
    result = finalize_session(session, duration_seconds=123.456)
    assert result["status"] == "completed"
    assert result["duration_seconds"] == pytest.approx(123.46)
    assert result["end_time"]
    assert result["files_present"]["mic"] is True
    assert result["files_present"]["system"] is True
    assert result["files_present"]["combined"] is False


def test_list_sessions_newest_first(tmp_path):
    a = create_session(tmp_path, app_name="A")
    b = create_session(tmp_path, app_name="B")
    sessions = list_sessions(tmp_path)
    assert sessions == sorted([a, b], reverse=True)
    assert list_sessions(tmp_path / "does-not-exist") == []
    # A folder without metadata.json is not a session.
    (tmp_path / "random-folder").mkdir()
    assert len(list_sessions(tmp_path)) == 2


def test_load_metadata_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path)
