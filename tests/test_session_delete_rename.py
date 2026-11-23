"""Session delete + rename + display-name rule (storage layer).

Delete removes the session folder and every shared-store trace
(search-index rows, contributed voice samples) while leaving
other sessions and speaker identities untouched. Rename sets a
custom name only for sessions not tied to an Outlook item; the
display name resolves through the one rule in
app.storage.sessions.session_display_name.
"""
from __future__ import annotations

import json
import re
import sqlite3
import zlib
from pathlib import Path

import numpy as np
import pytest

from app.speakers.database import SpeakerDatabase
from app.storage.browser import (
    library_row_lines,
    search_sessions,
    summarize_session,
)
from app.storage.semantic_search import SemanticSearchIndex
from app.storage.sessions import (
    create_session,
    delete_session,
    is_calendar_linked,
    load_metadata,
    rename_session,
    session_display_name,
    update_metadata,
)
from app.transcription.engine import Segment

EVENT = {"subject": "Budget review", "start": "2026-11-07T10:00:00+00:00",
         "end": "2026-11-07T11:00:00+00:00", "attendees": ["Priya"],
         "organiser": "John", "entry_id": "abc123"}


def _make_session(root: Path, name: str | None = None, **metadata) -> Path:
    session = create_session(root, app_name="Teams")
    if name is not None:
        target = root / name
        session.rename(target)
        session = target
        update_metadata(session, session_id=name)
    if metadata:
        update_metadata(session, **metadata)
    return session


def _write_transcript(session: Path) -> None:
    (session / "transcript.json").write_text(json.dumps(
        {"model": "tiny", "segments": [
            Segment(0.0, 5.0, "Hello everyone.", "SPEAKER_00").to_dict(),
        ]}, indent=2), encoding="utf-8")


class _FakeEmbedder:
    dim = 16

    def _word_vec(self, word: str) -> np.ndarray:
        rng = np.random.default_rng(zlib.crc32(word.encode("utf-8")))
        vec = rng.normal(size=self.dim).astype(np.float32)
        return vec / max(np.linalg.norm(vec), 1e-9)

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec += self._word_vec(word)
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm > 1e-9 else vec)
        return np.asarray(rows, dtype=np.float32)


# ------------------------------------------------------- display-name rule ---

def test_display_name_default_is_derived_from_app():
    assert session_display_name({"app_name": "Teams"}) == "Teams recording"
    assert session_display_name({}) == "Recording"


def test_display_name_custom_name_wins_over_default():
    metadata = {"app_name": "Teams", "custom_name": "Chat with Priya"}
    assert session_display_name(metadata) == "Chat with Priya"


def test_display_name_outlook_subject_wins_over_custom_name():
    metadata = {"app_name": "Teams", "custom_name": "My name",
                "calendar_event": dict(EVENT)}
    assert session_display_name(metadata) == "Budget review"
    assert is_calendar_linked(metadata) is True


def test_display_name_stored_null_link_is_not_linked():
    metadata = {"app_name": "Teams", "calendar_event": None}
    assert is_calendar_linked(metadata) is False
    assert session_display_name(metadata) == "Teams recording"


def test_display_name_blank_subject_falls_back_to_untitled():
    metadata = {"app_name": "Teams",
                "calendar_event": {"subject": "  ", "entry_id": "x"}}
    assert session_display_name(metadata) == "(untitled event)"


def test_summary_carries_display_name_and_link_flag(tmp_path):
    linked = _make_session(tmp_path, calendar_event=dict(EVENT))
    plain = _make_session(tmp_path)
    linked_summary = summarize_session(linked)
    assert linked_summary.display_name == "Budget review"
    assert linked_summary.calendar_linked is True
    plain_summary = summarize_session(plain)
    assert plain_summary.display_name == "Teams recording"
    assert plain_summary.calendar_linked is False


def test_row_lines_title_date_and_meta(tmp_path):
    session = _make_session(
        tmp_path, start_time="2026-11-07T10:00:00",
        duration_seconds=61.3, status="completed", source="teams")
    _write_transcript(session)
    (session / "speakers.json").write_text(
        json.dumps({"names": {"SPEAKER_00": "John"}}), encoding="utf-8")
    summary = summarize_session(session)
    title, meta = library_row_lines(summary)
    assert title == "Teams recording — Nov 07, 2026, 10:00"
    assert meta == "1m 01s • John • Teams"
    pinned = _make_session(tmp_path, pinned=True)
    title, _meta = library_row_lines(summarize_session(pinned))
    assert title.startswith("★ Teams recording — ")
    untranscribed = _make_session(tmp_path, status="completed")
    _title, meta = library_row_lines(summarize_session(untranscribed))
    assert meta.endswith("• No transcript yet")


def test_text_search_finds_custom_name_and_outlook_subject(tmp_path):
    renamed = _make_session(tmp_path)
    rename_session(renamed, "Q3 planning sync")
    linked = _make_session(tmp_path, calendar_event=dict(EVENT))
    assert [s.session_id for s in
            search_sessions(tmp_path, "planning sync",
                            mode="text")] == [renamed.name]
    assert [s.session_id for s in
            search_sessions(tmp_path, "budget review",
                            mode="text")] == [linked.name]


# ------------------------------------------------------------------ rename ---

def test_rename_persists_and_survives_reload(tmp_path):
    session = _make_session(tmp_path)
    rename_session(session, "  Chat with Priya  ")
    assert load_metadata(session)["custom_name"] == "Chat with Priya"
    assert load_metadata(session, backfill=False)[
        "custom_name"] == "Chat with Priya"
    assert summarize_session(session).display_name == "Chat with Priya"
    assert session.name == load_metadata(session)["session_id"]


def test_rename_blank_is_refused(tmp_path):
    session = _make_session(tmp_path)
    with pytest.raises(ValueError, match="blank"):
        rename_session(session, "   ")
    assert "custom_name" not in load_metadata(session)


def test_rename_outlook_linked_is_refused_at_storage_layer(tmp_path):
    session = _make_session(tmp_path, calendar_event=dict(EVENT))
    with pytest.raises(ValueError, match="Outlook"):
        rename_session(session, "My name")
    metadata = load_metadata(session)
    assert "custom_name" not in metadata
    assert metadata["calendar_event"]["subject"] == "Budget review"


# ------------------------------------------------------------------ delete ---

def test_delete_removes_folder_and_all_artifacts(tmp_path):
    session = _make_session(tmp_path, duration_seconds=61.3)
    _write_transcript(session)
    (session / "transcript.txt").write_text("Hello.\n", encoding="utf-8")
    (session / "transcript-clean.txt").write_text("Hello.\n",
                                                  encoding="utf-8")
    (session / "diarization.json").write_text("{}", encoding="utf-8")
    (session / "speakers.json").write_text("{}", encoding="utf-8")
    (session / "speaker_suggestions.json").write_text("{}",
                                                      encoding="utf-8")
    (session / "transcript-llm.md").write_text("# x\n", encoding="utf-8")
    delete_session(session)
    assert not session.exists()
    assert list(tmp_path.iterdir()) == []


def test_delete_removes_search_index_rows_but_not_neighbours(tmp_path):
    doomed = _make_session(tmp_path)
    _write_transcript(doomed)
    kept = _make_session(tmp_path)
    _write_transcript(kept)
    index = SemanticSearchIndex(tmp_path, embedder=_FakeEmbedder())
    assert index.index_session(doomed) > 0
    assert index.index_session(kept) > 0
    delete_session(doomed)
    conn = sqlite3.connect(str(tmp_path / "search_index.db"))
    rows = conn.execute("SELECT session_id FROM chunks").fetchall()
    assert [r[0] for r in rows] == [kept.name]
    state = conn.execute(
        "SELECT session_id FROM index_state").fetchall()
    assert [r[0] for r in state] == [kept.name]
    conn.close()
    assert kept.exists()


def test_delete_removes_voice_samples_but_keeps_speakers(tmp_path):
    session = _make_session(tmp_path)
    other = _make_session(tmp_path)
    db = SpeakerDatabase(tmp_path / "speakers.db")
    vec = np.ones(4, dtype=np.float32)
    db.add_embedding("John", vec, backend="fallback",
                     source_session=session.name)
    db.add_embedding("John", vec, backend="fallback",
                     source_session=other.name)
    db.add_embedding("Priya", vec, backend="fallback",
                     source_session=session.name)
    db.close()
    delete_session(session)
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        assert [s["name"] for s in db.list_speakers()] == ["John", "Priya"]
        john = db.get_speaker("John")
        assert john["sample_count"] == 1
        assert db.speaker_embedding("John") is not None
        assert db.get_speaker("Priya")["sample_count"] == 0
        assert db.speaker_embedding("Priya") is None
        remaining = [s["source_session"]
                     for s in db.speaker_samples("John")]
        assert remaining == [other.name]
    finally:
        db.close()


def test_delete_live_session_is_refused(tmp_path):
    session = _make_session(tmp_path)
    with pytest.raises(RuntimeError, match="being recorded"):
        delete_session(session, live_dir=session)
    assert session.exists()
    assert (session / "metadata.json").exists()


def test_delete_non_session_folder_is_refused(tmp_path):
    plain = tmp_path / "not-a-session"
    plain.mkdir()
    with pytest.raises(FileNotFoundError):
        delete_session(plain)
    assert plain.exists()


def test_delete_without_shared_stores_creates_none(tmp_path):
    session = _make_session(tmp_path)
    delete_session(session)
    assert not (tmp_path / "search_index.db").exists()
    assert not (tmp_path / "speakers.db").exists()


def test_delete_neighbour_session_untouched(tmp_path):
    doomed = _make_session(tmp_path)
    kept = _make_session(tmp_path, custom_name="Keep me")
    delete_session(doomed)
    assert kept.exists()
    assert load_metadata(kept)["custom_name"] == "Keep me"
