"""Phase 5 tests — Recording Browser data layer.

Synthetic session folders only (no audio, no hardware, no Qt).
The UI itself is import-guarded and tested only for its pure
format helper.
"""
from __future__ import annotations

import json
import os

import pytest

from app.diarization.names import save_speaker_names
from app.storage.browser import (
    format_local_time,
    list_session_summaries,
    search_sessions,
    session_transcript_text,
    summarize_session,
)
from app.storage.sessions import create_session, update_metadata
from app.ui.browser import format_session_line


def _make_session(root, app_name="Teams", transcript_text=None,
                  speakers=None, duration=61.3):
    session = create_session(root, app_name=app_name)
    update_metadata(session, duration_seconds=duration)
    if transcript_text is not None:
        (session / "transcript.txt").write_text(transcript_text,
                                                encoding="utf-8")
        (session / "transcript.json").write_text(json.dumps(
            {"model": "tiny", "segments": []}, indent=2), encoding="utf-8")
    if speakers:
        save_speaker_names(session, speakers)
    return session


def test_summarize_session_reads_metadata_and_speakers(tmp_path):
    session = _make_session(
        tmp_path, transcript_text="[00:00] John: Hello everyone.\n",
        speakers={"You": "John"}, duration=61.3)
    summary = summarize_session(session)
    assert summary.session_id == session.name
    assert summary.duration_seconds == 61.3
    assert summary.app_name == "Teams"
    assert summary.has_transcript is True
    assert summary.has_diarization is False
    assert summary.has_llm_export is False
    assert "John" in summary.speakers
    assert "61" in summary.display or "1m" in summary.display


def test_list_session_summaries_newest_first_and_empty_root(tmp_path):
    assert list_session_summaries(tmp_path / "missing") == []
    a = _make_session(tmp_path, transcript_text="first session")
    b = _make_session(tmp_path, transcript_text="second session")
    summaries = list_session_summaries(tmp_path)
    assert [s.path for s in summaries] == sorted([a, b], reverse=True)


def test_search_matches_transcript_text(tmp_path):
    _make_session(tmp_path, transcript_text="The budget is twelve "
                                            "thousand dollars.")
    _make_session(tmp_path, transcript_text="Nothing about money here.")
    hits = search_sessions(tmp_path, "budget")
    assert len(hits) == 1
    assert "budget" in session_transcript_text(hits[0].path).lower()


def test_search_matches_speaker_names_case_insensitive(tmp_path):
    _make_session(tmp_path, transcript_text="Hello.",
                  speakers={"You": "Priya"})
    _make_session(tmp_path, transcript_text="Hello.",
                  speakers={"You": "Marcus"})
    hits = search_sessions(tmp_path, "pRiYa")
    assert len(hits) == 1
    assert hits[0].speakers == ["Priya"]


def test_search_matches_session_id_and_empty_query_returns_all(tmp_path):
    a = _make_session(tmp_path, transcript_text="alpha")
    b = _make_session(tmp_path, transcript_text="beta")
    assert len(search_sessions(tmp_path, "")) == 2
    assert len(search_sessions(tmp_path, "   ")) == 2
    # Search by a fragment of b's session id that a does not share:
    # the two sessions usually share a timestamp prefix (created in
    # the same second), so use the part of b's name past the common
    # prefix. Never assume the "-02" collision suffix exists — that
    # depends on both landing in the same second, and under a loaded
    # full-suite run they often do not (flaked Nov 20 2026).
    common = os.path.commonprefix([a.name, b.name])
    fragment = b.name[len(common):]
    assert fragment and fragment not in a.name
    hits = search_sessions(tmp_path, fragment)
    assert [h.path for h in hits] == [b]
    assert search_sessions(tmp_path, "no-such-word-anywhere") == []


def test_session_transcript_text_falls_back_to_json(tmp_path):
    session = create_session(tmp_path)
    (session / "transcript.json").write_text(json.dumps(
        {"segments": [{"start": 0, "end": 1, "text": "From JSON."}]},
        indent=2), encoding="utf-8")
    assert session_transcript_text(session) == "From JSON."
    assert session_transcript_text(tmp_path) == ""


def test_format_session_line_is_pure_text(tmp_path):
    session = _make_session(tmp_path, transcript_text="Hello.",
                            speakers={"You": "John"})
    line = format_session_line(summarize_session(session))
    assert session.name in line
    assert "John" in line
    assert "transcript" in line


def test_status_text_human_status_per_state(tmp_path):
    # Recorded only: says plainly it is not transcribed yet.
    recorded = _make_session(tmp_path)
    assert summarize_session(recorded).status_text == (
        "Recorded — not transcribed yet")
    # Transcribed: the flags accumulate in words.
    transcribed = _make_session(tmp_path, transcript_text="Hello.")
    assert summarize_session(transcribed).status_text == (
        "Recorded · Transcribed")
    # Diarized + suggestions + LLM export all show.
    full = _make_session(tmp_path, transcript_text="Hello.")
    (full / "diarization.json").write_text("{}", encoding="utf-8")
    (full / "speaker_suggestions.json").write_text(
        '{"version": 1, "suggestions": []}', encoding="utf-8")
    (full / "transcript-llm.md").write_text("# x\n", encoding="utf-8")
    assert summarize_session(full).status_text == (
        "Recorded · Transcribed · Diarized · Speakers identified · "
        "LLM export ready")
    # The browser row carries the status line as its second line.
    line = format_session_line(summarize_session(full))
    assert line.split("\n")[1].startswith("Recorded · Transcribed")


def test_hybrid_default_falls_back_to_text_when_model_unavailable(
        tmp_path, monkeypatch):
    """Regression (Nov 8 2026): hybrid is the default mode, so a
    machine without the MiniLM model (no onnxruntime, offline)
    must still get text results instead of an exception — the
    Recording Browser runs this on every keystroke."""
    import app.storage.browser as browser_mod
    from app.storage.semantic_search import ModelNotAvailableError

    _make_session(tmp_path, transcript_text="The budget is twelve "
                                            "thousand dollars.")
    _make_session(tmp_path, transcript_text="Nothing about money here.")

    def _boom(*args, **kwargs):
        raise ModelNotAvailableError("onnxruntime is not installed")

    monkeypatch.setattr(browser_mod, "hybrid_search", _boom)
    hits = search_sessions(tmp_path, "budget")  # default mode: hybrid
    assert len(hits) == 1
    assert "budget" in session_transcript_text(hits[0].path).lower()


def test_semantic_mode_still_raises_when_model_unavailable(
        tmp_path, monkeypatch):
    """Explicit semantic mode keeps the loud failure — only the
    hybrid default degrades silently to text."""
    import app.storage.browser as browser_mod
    from app.storage.semantic_search import ModelNotAvailableError

    _make_session(tmp_path, transcript_text="The budget is twelve "
                                            "thousand dollars.")

    def _boom(*args, **kwargs):
        raise ModelNotAvailableError("onnxruntime is not installed")

    monkeypatch.setattr(browser_mod, "hybrid_search", _boom)
    monkeypatch.setattr(browser_mod, "_semantic_hits_to_sessions", _boom)
    with pytest.raises(ModelNotAvailableError):
        search_sessions(tmp_path, "budget", mode="semantic")

# ------------------------------------------------------ local-time display --
def test_format_local_time_converts_utc_to_local(monkeypatch):
    """metadata.json stores UTC; the UI must show local time.

    Nov 10 2026: "Loaded session" showed the raw UTC value, hours
    ahead of the (local-time) folder name on non-UTC machines.
    America/New_York is UTC-5 in November, so 15:12 UTC displays
    as 10:12. Skipped where the platform cannot set the local zone
    (Windows has no time.tzset).
    """
    import os
    import time

    if not hasattr(time, "tzset"):
        pytest.skip("no time.tzset on this platform")
    old_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        assert (format_local_time("2026-11-10T15:12:32+00:00")
                == "Nov 10, 2026, 10:12")
    finally:
        # Restore the pre-test zone AND call tzset(): deleting TZ
        # alone would leave the process on /etc/localtime's zone
        # (UTC on this VM), leaking it into every later test.
        if old_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", old_tz)
        time.tzset()


def test_format_local_time_leaves_naive_values_and_bad_input_alone():
    """Naive timestamps (old sessions, no offset) are already local;
    missing/unparsable values return "" so callers fall back to the
    folder-name stamp."""
    assert format_local_time("2026-11-10T10:12:32") == "Nov 10, 2026, 10:12"
    assert format_local_time(None) == ""
    assert format_local_time("") == ""
    assert format_local_time("not a timestamp") == ""


# ------------------------------------------- speaker filter + pins --
def test_summarize_session_reads_pinned_flag(tmp_path):
    session = _make_session(tmp_path, transcript_text="Hello.")
    assert summarize_session(session).pinned is False
    update_metadata(session, pinned=True)
    assert summarize_session(session).pinned is True
    update_metadata(session, pinned=False)
    assert summarize_session(session).pinned is False


def test_list_session_summaries_pinned_first_then_newest(tmp_path):
    older = _make_session(tmp_path, transcript_text="older")
    newer = _make_session(tmp_path, transcript_text="newer")
    assert [s.path for s in list_session_summaries(tmp_path)] == [
        newer, older]
    update_metadata(older, pinned=True)
    assert [s.path for s in list_session_summaries(tmp_path)] == [
        older, newer]
    # Two pinned sessions stay newest-first among themselves.
    update_metadata(newer, pinned=True)
    assert [s.path for s in list_session_summaries(tmp_path)] == [
        newer, older]


def _make_segment_session(root, segments, names=None):
    session = create_session(root, app_name="Teams")
    (session / "transcript.json").write_text(json.dumps(
        {"model": "tiny", "segments": segments}, indent=2),
        encoding="utf-8")
    # transcript.txt is what the text search reads first, so it
    # must carry the same words as the segments.
    (session / "transcript.txt").write_text(
        "".join(f"[00:00] {s['text']}\n" for s in segments),
        encoding="utf-8")
    if names:
        save_speaker_names(session, names)
    return session


def test_search_speaker_filter_matches_display_name_after_rename(
        tmp_path):
    priya = _make_segment_session(tmp_path, [
        {"start": 0, "end": 3, "speaker": "SPEAKER_00",
         "text": "Hello."}], names={"SPEAKER_00": "Priya"})
    _make_segment_session(tmp_path, [
        {"start": 0, "end": 3, "speaker": "SPEAKER_00",
         "text": "Hello."}], names={"SPEAKER_00": "Marcus"})
    hits = search_sessions(tmp_path, "", mode="text", speaker="Priya")
    assert [h.path for h in hits] == [priya]
    # Case-insensitive, and the other name finds only its session.
    assert len(search_sessions(tmp_path, "", mode="text",
                               speaker="pRiYa")) == 1
    assert search_sessions(tmp_path, "", mode="text",
                           speaker="Nobody") == []


def test_search_speaker_filter_applies_with_a_query(tmp_path):
    priya = _make_segment_session(tmp_path, [
        {"start": 0, "end": 3, "speaker": "SPEAKER_00",
         "text": "The budget is approved."}],
        names={"SPEAKER_00": "Priya"})
    _make_segment_session(tmp_path, [
        {"start": 0, "end": 3, "speaker": "SPEAKER_00",
         "text": "The budget is approved."}],
        names={"SPEAKER_00": "Marcus"})
    hits = search_sessions(tmp_path, "budget", mode="text",
                           speaker="Priya")
    assert [h.path for h in hits] == [priya]
    assert search_sessions(tmp_path, "budget", mode="text",
                           speaker="Marcus")[0].speakers == ["Marcus"]


def test_search_speaker_and_source_filters_combine(tmp_path):
    session = _make_segment_session(tmp_path, [
        {"start": 0, "end": 3, "speaker": "SPEAKER_00",
         "text": "Hello."}], names={"SPEAKER_00": "Priya"})
    update_metadata(session, source="solo")
    assert search_sessions(tmp_path, "", mode="text", speaker="Priya",
                           source="solo") != []
    assert search_sessions(tmp_path, "", mode="text", speaker="Priya",
                           source="teams") == []


def test_text_search_keeps_pinned_first(tmp_path):
    older = _make_session(tmp_path, transcript_text="shared word")
    newer = _make_session(tmp_path, transcript_text="shared word")
    update_metadata(older, pinned=True)
    hits = search_sessions(tmp_path, "shared", mode="text")
    assert [h.path for h in hits] == [older, newer]
