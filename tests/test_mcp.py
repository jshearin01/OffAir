"""MCP server tests — every tool against seeded demo sessions.

Tool logic is tested directly on :class:`app.mcp.tools.OffAirMCPTools`
(no SDK needed — the shared venv has no ``mcp`` package), with a
deterministic fake embedder standing in for the real MiniLM model
(same word-overlap approach as tests/test_semantic_search.py).
The stdio end-to-end test at the bottom is skipped when the SDK is
not installed; run it with the scratch venv (see docs/MCP.md).

No network anywhere in this file: the no-network test blocks
``socket.socket`` and ``urllib.request.urlopen`` and then calls
every tool.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import socket
import sys
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pytest

from app.diarization.names import save_speaker_names
from app.mcp.tools import (
    SOURCE_VALUES,
    OffAirMCPTools,
)
from app.speakers.database import SpeakerDatabase
from app.storage.bookmarks import add_bookmark
from app.storage.browser import search_sessions
from app.storage.semantic_search import (
    ModelNotAvailableError,
    SemanticSearchIndex,
)
from app.storage.sessions import (
    create_dictation_session,
    create_session,
    update_metadata,
)
from app.transcription.cleanup import clean_session
from app.transcription.engine import Segment

HAS_MCP = importlib.util.find_spec("mcp") is not None
REPO_ROOT = Path(__file__).resolve().parent.parent

FAKE_DIM = 1024


class FakeEmbedder:
    """Deterministic word-overlap embedder (test-only).

    Each word -> a fixed pseudo-random unit vector; a text's
    embedding is the L2-normalised sum of its words' vectors.
    """

    def __init__(self, dim: int = FAKE_DIM) -> None:
        self.dim = dim

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
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class BrokenEmbedder:
    """Embedder that always fails — stands in for 'model unavailable'."""

    def embed(self, texts: list[str]):
        raise ModelNotAvailableError("fake model unavailable")

    def embed_one(self, text: str):
        raise ModelNotAvailableError("fake model unavailable")


class FakeOllama:
    """Ollama stand-in: available or not, never touches localhost."""

    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.model = "fake-model"
        self.last_prompt_template: str | None = None

    def is_available(self) -> bool:
        return self._available

    def summarize(self, transcript_text: str, *,
                  prompt_template: str | None = None) -> str:
        self.last_prompt_template = prompt_template
        return f"FAKE ANSWER from excerpts: {transcript_text[:60]}"


def _write_transcript(session, segments):
    (session / "transcript.json").write_text(json.dumps(
        {"model": "tiny",
         "segments": [s.to_dict() for s in segments]},
        indent=2), encoding="utf-8")


def _budget_segments():
    return [
        Segment(0.0, 5.0, "Good morning everyone, thanks for joining "
                          "the OffAir review.", "SPEAKER_00"),
        Segment(5.0, 15.0, "One more thing. The budget for the pilot is "
                           "twelve thousand dollars, and the deadline is "
                           "the fifteenth of December.", "SPEAKER_01"),
        Segment(15.0, 20.0, "Yes, December fifteenth works for me.",
                "SPEAKER_00"),
    ]


def _timeline_segments():
    return [
        Segment(0.0, 8.0, "First item is the recording test. I recorded "
                          "a five minute Teams call yesterday and both "
                          "audio tracks came through clearly.",
                "SPEAKER_02"),
        Segment(8.0, 18.0, "Second item is transcription. The small "
                           "Whisper model should handle a thirty minute "
                           "meeting in about fifteen minutes on a laptop "
                           "with no graphics card.", "SPEAKER_01"),
        Segment(18.0, 26.0, "Priya, can you send the updated timeline "
                            "by Monday morning?", "SPEAKER_00"),
    ]


def _make_session(root, segments, app_name="Teams", source=None,
                  source_detail=None, names=None, duration=61.3):
    session = create_session(root, app_name=app_name)
    fields = {"duration_seconds": duration, "status": "completed"}
    if source is not None:
        fields["source"] = source
    if source_detail is not None:
        fields["source_detail"] = source_detail
    update_metadata(session, **fields)
    _write_transcript(session, segments)
    if names:
        save_speaker_names(session, names)
    return session


@pytest.fixture
def demo_root(tmp_path):
    """Two sessions: a Teams budget meeting + a Claude Code timeline one.

    The Teams session uses SPEAKER_00/01 labels renamed to
    John/Marcus (tests rename application); the Claude Code session
    uses plain labels. A third legacy session (no source key) is
    added by the source tests that need it.
    """
    budget = _make_session(
        tmp_path, _budget_segments(), source="teams",
        source_detail="Microsoft Teams",
        names={"SPEAKER_00": "John", "SPEAKER_01": "Marcus"})
    timeline = _make_session(
        tmp_path, _timeline_segments(), app_name="Claude Code",
        source="claude-code", source_detail="agent work session",
        names={"SPEAKER_00": "John", "SPEAKER_01": "Marcus",
               "SPEAKER_02": "Priya"})
    return tmp_path, budget, timeline


@pytest.fixture
def tools(demo_root):
    root, _, _ = demo_root
    index = SemanticSearchIndex(root, embedder=FakeEmbedder())
    return OffAirMCPTools(root, index=index,
                          ollama_client=FakeOllama(available=False))


# ------------------------------------------------------------ search -------
def test_search_meetings_text_finds_budget(tools, demo_root):
    _, budget, _ = demo_root
    out = tools.search_meetings("budget", mode="text")
    assert out["mode"] == "text"
    assert [r["session_id"] for r in out["results"]] == [budget.name]
    assert out["results"][0]["match_type"] == "text"
    assert "budget" in (out["results"][0]["snippet"] or "").lower()


def test_search_meetings_empty_query_lists_all(tools, demo_root):
    out = tools.search_meetings("", mode="text")
    assert len(out["results"]) == 2


def test_search_meetings_unknown_mode_raises(tools):
    with pytest.raises(ValueError, match="unknown search mode"):
        tools.search_meetings("budget", mode="weird")


def test_search_meetings_hybrid_with_fake_index(tools, demo_root):
    _, budget, _ = demo_root
    out = tools.search_meetings("budget pilot twelve thousand",
                                mode="hybrid")
    assert out["mode"] == "hybrid"
    assert out["note"] is None
    assert out["results"][0]["session_id"] == budget.name
    assert out["results"][0]["score"] is not None
    assert out["results"][0]["match_type"] in ("both", "semantic", "text")


def test_search_meetings_semantic_falls_back_when_model_missing(demo_root):
    root, budget, _ = demo_root
    broken = OffAirMCPTools(
        root, index=SemanticSearchIndex(root, embedder=BrokenEmbedder()))
    out = broken.search_meetings("budget", mode="semantic")
    assert out["mode"] == "text"  # fell back, and says so
    assert "fell back to text search" in (out["note"] or "")
    assert [r["session_id"] for r in out["results"]] == [budget.name]


def test_search_meetings_hybrid_falls_back_when_model_missing(demo_root):
    root, budget, _ = demo_root
    broken = OffAirMCPTools(
        root, index=SemanticSearchIndex(root, embedder=BrokenEmbedder()))
    out = broken.search_meetings("budget", mode="hybrid")
    assert out["mode"] == "text"
    assert [r["session_id"] for r in out["results"]] == [budget.name]


def test_search_meetings_limit_validation(tools):
    with pytest.raises(ValueError, match="between 1 and 100"):
        tools.search_meetings("x", limit=0)
    with pytest.raises(ValueError, match="between 1 and 100"):
        tools.search_meetings("x", limit=101)


# ------------------------------------------------------------ list ---------
def test_list_sessions_newest_first_and_limit(tmp_path):
    # Folder names sort chronologically, so the two sessions need
    # distinct second-stamps for a meaningful newest-first order.
    import time

    first = _make_session(tmp_path, _budget_segments())
    time.sleep(1.1)
    second = _make_session(tmp_path, _timeline_segments())
    tools = OffAirMCPTools(tmp_path)
    out = tools.list_sessions(limit=1)
    assert [r["session_id"] for r in out["results"]] == [second.name]
    out = tools.list_sessions(limit=10)
    assert [r["session_id"] for r in out["results"]] == [second.name,
                                                         first.name]


def test_list_sessions_since_filter(tools, demo_root):
    out = tools.list_sessions(since="2999-01-01")
    assert out["results"] == []
    out = tools.list_sessions(since="2000-01-01")
    assert len(out["results"]) == 2
    with pytest.raises(ValueError, match="Could not parse 'since'"):
        tools.list_sessions(since="not-a-date")


# ------------------------------------------------------------ source -------
def test_get_session_returns_source_fields(tools, demo_root):
    _, budget, _ = demo_root
    out = tools.get_session(budget.name)
    assert out["source"] == "teams"
    assert out["source_detail"] == "Microsoft Teams"
    assert out["calendar_event"] is None
    assert out["status"] == "completed"
    assert out["speakers"] == ["John", "Marcus"]
    assert out["duration_seconds"] == 61.3


def test_get_session_legacy_session_without_source_is_other(tmp_path):
    legacy = _make_session(tmp_path, _budget_segments())  # no source key
    tools = OffAirMCPTools(tmp_path)
    out = tools.get_session(legacy.name)
    assert out["source"] == "other"
    assert out["source_detail"] is None


def test_get_session_unknown_session_raises(tools):
    with pytest.raises(ValueError, match="No session"):
        tools.get_session("no-such-session")


def test_list_sessions_source_filter(tmp_path):
    teams = _make_session(tmp_path, _budget_segments(), source="teams",
                          source_detail="Microsoft Teams")
    code = _make_session(tmp_path, _timeline_segments(),
                         app_name="Claude Code", source="claude-code")
    legacy = _make_session(tmp_path, _budget_segments())  # no source key
    tools = OffAirMCPTools(tmp_path)
    assert [r["session_id"] for r in
            tools.list_sessions(source="teams")["results"]] == [teams.name]
    assert [r["session_id"] for r in
            tools.list_sessions(source="claude-code")["results"]] == [code.name]
    # Legacy sessions (no source key) read as "other".
    assert [r["session_id"] for r in
            tools.list_sessions(source="other")["results"]] == [legacy.name]
    # No filter = everything, and every entry carries the fields.
    all_out = tools.list_sessions()["results"]
    assert len(all_out) == 3
    assert {r["source"] for r in all_out} == {"teams", "claude-code",
                                              "other"}
    with pytest.raises(ValueError, match="unknown source"):
        tools.list_sessions(source="skype")


def test_search_meetings_source_filter(tmp_path):
    teams = _make_session(tmp_path, _budget_segments(), source="teams")
    _make_session(tmp_path, _budget_segments(), source="zoom")
    tools = OffAirMCPTools(tmp_path)
    # "budget" matches both sessions' transcripts; the filter keeps one.
    out = tools.search_meetings("budget", mode="text", source="teams")
    assert [r["session_id"] for r in out["results"]] == [teams.name]
    out = tools.search_meetings("budget", mode="text", source="zoom")
    assert len(out["results"]) == 1
    assert out["results"][0]["source"] == "zoom"
    out = tools.search_meetings("budget", mode="text", source="other")
    assert out["results"] == []
    with pytest.raises(ValueError, match="unknown source"):
        tools.search_meetings("budget", source="skype")


def test_search_meetings_source_filter_hybrid(tmp_path):
    teams = _make_session(tmp_path, _budget_segments(), source="teams")
    _make_session(tmp_path, _budget_segments(), source="zoom")
    tools = OffAirMCPTools(
        tmp_path,
        index=SemanticSearchIndex(tmp_path, embedder=FakeEmbedder()))
    out = tools.search_meetings("budget pilot twelve thousand",
                                mode="hybrid", source="teams")
    assert [r["session_id"] for r in out["results"]] == [teams.name]


def test_source_enum_is_the_contract():
    assert set(SOURCE_VALUES) == {"teams", "zoom", "webex", "meet",
                                  "browser", "solo", "claude-code",
                                  "other"}


# ------------------------------------------------------------ transcript ---
def test_get_transcript_text_applies_renames(tools, demo_root):
    _, budget, _ = demo_root
    out = tools.get_transcript(budget.name, format="text")
    assert out["format"] == "text"
    assert "[00:00] John: Good morning everyone" in out["text"]
    assert "[00:05] Marcus: One more thing." in out["text"]
    assert "SPEAKER_00" not in out["text"]


def test_get_transcript_segments(tools, demo_root):
    _, budget, _ = demo_root
    out = tools.get_transcript(budget.name, format="segments")
    assert out["format"] == "segments"
    assert out["text"] is None
    first = out["segments"][0]
    assert first == {"start": 0.0, "end": 5.0, "speaker": "John",
                     "text": "Good morning everyone, thanks for joining "
                             "the OffAir review."}


def test_get_transcript_txt_only_session(tmp_path):
    session = create_session(tmp_path, app_name="Teams")
    update_metadata(session, duration_seconds=30.0)
    (session / "transcript.txt").write_text(
        "[00:00] Hello from a text-only transcript.\n", encoding="utf-8")
    out = OffAirMCPTools(tmp_path).get_transcript(session.name)
    assert "Hello from a text-only transcript." in out["text"]


def test_get_transcript_unknown_format_and_missing(tools, demo_root):
    _, budget, _ = demo_root
    with pytest.raises(ValueError, match="unknown transcript format"):
        tools.get_transcript(budget.name, format="pdf")
    with pytest.raises(ValueError, match="No session"):
        tools.get_transcript("no-such-session")
    empty = create_session(demo_root[0], app_name="Teams")
    with pytest.raises(ValueError, match="has no transcript"):
        tools.get_transcript(empty.name)


# ------------------------------------------------------------ speakers -----
def test_list_speakers_from_speakers_db(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    try:
        for name, session_name in (("John", "2026-11-07_17-32-10_teams"),
                                   ("Priya", "2026-11-07_17-32-10_teams")):
            db.add_embedding(
                name, np.ones(8, dtype=np.float32),
                backend="fallback-spectral-v1",
                source_session=session_name, source_label="SPEAKER_00")
    finally:
        db.close()
    out = OffAirMCPTools(tmp_path).list_speakers()
    assert [r["name"] for r in out["results"]] == ["John", "Priya"]
    john = out["results"][0]
    assert john["backend"] == "fallback-spectral-v1"
    assert john["sample_count"] == 1
    assert john["sessions"] == ["2026-11-07_17-32-10_teams"]


def test_list_speakers_empty_without_db_and_no_db_created(tmp_path):
    out = OffAirMCPTools(tmp_path).list_speakers()
    assert out == {"results": []}
    assert not (tmp_path / "speakers.db").exists()


# ------------------------------------------------------------ speaker say --
def test_what_did_speaker_say(tools, demo_root):
    out = tools.what_did_speaker_say("Marcus")
    assert len(out["results"]) == 2  # one segment in each session
    assert all(r["speaker"] == "Marcus" for r in out["results"])
    assert any("budget for the pilot" in r["text"]
               for r in out["results"])
    # Case-insensitive match, and the query narrows to one segment.
    out = tools.what_did_speaker_say("marcus", query="budget")
    assert len(out["results"]) == 1
    assert "budget" in out["results"][0]["text"].lower()
    # Unknown speaker: empty results, not an error.
    assert tools.what_did_speaker_say("Nobody") == {"speaker": "Nobody",
                                                    "results": []}
    with pytest.raises(ValueError, match="must not be blank"):
        tools.what_did_speaker_say("  ")


# ------------------------------------------------------------ ask ----------
def test_ask_meetings_falls_back_without_ollama(tools):
    out = tools.ask_meetings("What is the budget?")
    assert out["answer"] is None
    assert out["source"] == "search-fallback"
    assert "Ollama is not running" in (out["note"] or "")
    assert out["results"], "fallback must return the excerpts"
    assert any("budget" in r["text"].lower() for r in out["results"])


def test_ask_meetings_with_ollama_uses_excerpts_only(demo_root):
    root, _, _ = demo_root
    fake = FakeOllama(available=True)
    tools = OffAirMCPTools(
        root, index=SemanticSearchIndex(root, embedder=FakeEmbedder()),
        ollama_client=fake)
    out = tools.ask_meetings("What is the budget?")
    assert out["source"] == "ollama"
    assert out["model"] == "fake-model"
    assert out["answer"].startswith("FAKE ANSWER")
    assert out["sessions"], "answer must name its source sessions"
    # The prompt tells the model to answer ONLY from the excerpts.
    assert "ONLY from the transcript excerpts" in (
        fake.last_prompt_template or "")


def test_ask_meetings_no_matches_says_so(tools):
    out = tools.ask_meetings("zxqvkj wobbleflop")
    assert out["answer"] is None
    assert out["source"] == "search-fallback"
    assert "No meetings matched" in (out["note"] or "")
    assert out["results"] == []
    with pytest.raises(ValueError, match="must not be blank"):
        tools.ask_meetings("  ")


# ------------------------------------------------------------ speaker -----
def test_get_speaker_detail_via_mcp(tmp_path):
    session = _make_session(
        tmp_path, _budget_segments(), source="teams",
        names={"SPEAKER_00": "John", "SPEAKER_01": "Marcus"})
    (session / "combined.wav").write_bytes(b"")
    tools = OffAirMCPTools(tmp_path)
    # Case-insensitive display-name match; the detail is the
    # people-query shape, reused verbatim.
    out = tools.get_speaker("john")
    assert out["session_count"] == 1
    assert out["utterance_count"] == 2
    assert out["sessions"][0]["session_id"] == session.name
    assert all(u["speaker"] == "John" for u in out["utterances"])
    assert out["known_in_database"] is False
    assert out["sample_available"] is True
    assert out["sample"]["session_id"] == session.name
    # Unknown speaker: empty detail, not an error.
    nobody = tools.get_speaker("Nobody")
    assert nobody["known_in_database"] is False
    assert nobody["sessions"] == []
    assert nobody["utterances"] == []
    assert nobody["sample"] is None
    assert nobody["sample_available"] is False
    with pytest.raises(ValueError, match="must not be blank"):
        tools.get_speaker("  ")
    # Reading a speaker must not create speakers.db.
    assert not (tmp_path / "speakers.db").exists()


# ------------------------------------------------------------ bookmarks --
def test_list_bookmarks_and_get_session_bookmarks(tmp_path):
    session = _make_session(tmp_path, _budget_segments())
    add_bookmark(session, 30.0, "Later point")
    add_bookmark(session, 10.0, "Budget decision")
    tools = OffAirMCPTools(tmp_path)
    # Earliest first; no audio file yet, so audio is unavailable.
    out = tools.list_bookmarks(session.name)
    assert out["session_id"] == session.name
    assert [(b["t"], b["label"]) for b in out["results"]] == [
        (10.0, "Budget decision"), (30.0, "Later point")]
    assert all(b["audio_available"] is False for b in out["results"])
    assert all(b["created"] for b in out["results"])
    # Audio appears: the same bookmarks now report playable audio.
    (session / "combined.wav").write_bytes(b"")
    assert tools.list_bookmarks(session.name)["results"][0][
        "audio_available"] is True
    # get_session embeds the same bookmarks.
    got = tools.get_session(session.name)
    assert got["bookmarks"] == tools.list_bookmarks(session.name)[
        "results"]
    with pytest.raises(ValueError, match="No session"):
        tools.list_bookmarks("no-such-session")


def test_list_bookmarks_empty_session(tmp_path):
    session = _make_session(tmp_path, _budget_segments())
    assert OffAirMCPTools(tmp_path).list_bookmarks(session.name) == {
        "session_id": session.name, "results": []}


# ------------------------------------------------------------ pinned -----
def test_list_sessions_pinned_field_and_pinned_only(tmp_path):
    import time

    first = _make_session(tmp_path, _budget_segments())
    time.sleep(1.1)
    second = _make_session(tmp_path, _timeline_segments())
    update_metadata(second, pinned=True)
    tools = OffAirMCPTools(tmp_path)
    out = tools.list_sessions()
    # Pinned first, and every entry carries the pinned flag.
    assert [r["session_id"] for r in out["results"]] == [second.name,
                                                         first.name]
    assert [r["pinned"] for r in out["results"]] == [True, False]
    assert all("dictation" in r for r in out["results"])
    pinned = tools.list_sessions(pinned_only=True)
    assert [r["session_id"] for r in pinned["results"]] == [second.name]
    assert tools.get_session(second.name)["pinned"] is True
    assert tools.get_session(first.name)["pinned"] is False


# ------------------------------------------------------------ speaker filter
def test_search_meetings_speaker_filter(tools, demo_root):
    _, budget, timeline = demo_root
    # Empty query + speaker: only sessions that speaker spoke in.
    out = tools.search_meetings("", mode="text", speaker="Priya")
    assert [r["session_id"] for r in out["results"]] == [timeline.name]
    # Session-level filter: Priya spoke in no budget session.
    out = tools.search_meetings("budget", mode="text", speaker="Priya")
    assert out["results"] == []
    out = tools.search_meetings("budget", mode="text", speaker="marcus")
    assert [r["session_id"] for r in out["results"]] == [budget.name]
    # Blank speaker = no filtering.
    out = tools.search_meetings("budget", mode="text", speaker="  ")
    assert len(out["results"]) == 1


def test_search_meetings_speaker_filter_hybrid(tools, demo_root):
    _, budget, _ = demo_root
    out = tools.search_meetings("budget pilot twelve thousand",
                                mode="hybrid", speaker="Marcus")
    assert [r["session_id"] for r in out["results"]] == [budget.name]
    out = tools.search_meetings("budget pilot twelve thousand",
                                mode="hybrid", speaker="Nobody")
    assert out["results"] == []


# ------------------------------------------------------------ variants ---
def test_get_transcript_variant_clean_and_enterprise(tools, demo_root):
    _, budget, timeline = demo_root
    (budget / "transcript-clean.txt").write_text(
        "[00:00] John: Good morning everyone.\n", encoding="utf-8")
    (budget / "transcript-enterprise-clean.txt").write_text(
        "# Transcript (enterprise-cleaned)\n\n"
        "[00:00] John: Good morning.\n", encoding="utf-8")
    # Default is the verbatim transcript, labelled as such.
    default = tools.get_transcript(budget.name)
    assert default["variant"] == "verbatim"
    assert default["variant_label"] == "Original (verbatim)"
    assert default["note"] is None
    assert "twelve thousand" in default["text"]
    clean = tools.get_transcript(budget.name, variant="clean")
    assert clean["variant"] == "clean"
    assert clean["variant_label"] == "Cleaned"
    assert clean["format"] == "text"
    assert clean["segments"] is None
    assert clean["text"] == ("[00:00] John: Good morning everyone.\n")
    assert clean["note"] is None
    enterprise = tools.get_transcript(budget.name, variant="enterprise")
    assert enterprise["variant"] == "enterprise"
    assert enterprise["variant_label"] == "Enterprise-cleaned"
    assert "Good morning." in enterprise["text"]
    assert "external LLM" in (enterprise["note"] or "")
    assert "transcript text only" in (enterprise["note"] or "")
    # The canonical registry keys are accepted as aliases.
    assert tools.get_transcript(budget.name,
                                variant="original")["variant"] == "verbatim"
    assert tools.get_transcript(budget.name,
                                variant="cleaned")["variant"] == "clean"
    # Missing variant file: a clear error, not invented content.
    with pytest.raises(ValueError, match="has no Cleaned transcript"):
        tools.get_transcript(timeline.name, variant="clean")
    with pytest.raises(ValueError, match="unknown transcript variant"):
        tools.get_transcript(budget.name, variant="polished")
    # Clean / enterprise variants are text only.
    with pytest.raises(ValueError, match="text only"):
        tools.get_transcript(budget.name, format="segments",
                             variant="clean")


def test_get_transcript_segments_still_default_verbatim(tools, demo_root):
    _, budget, _ = demo_root
    out = tools.get_transcript(budget.name, format="segments")
    assert out["variant"] == "verbatim"
    assert out["segments"][0]["speaker"] == "John"


# ------------------------------------------------------------ dictation ---
def _make_dictation_session(root):
    session = create_dictation_session(root)
    update_metadata(session, duration_seconds=30.0,
                    status="completed")
    (session / "transcript.json").write_text(json.dumps(
        {"model": "base", "segments": [
            {"start": 0.0, "end": 3.0,
             "text": "Um, remember to call Priya about the budget."},
            {"start": 4.0, "end": 7.0,
             "text": "The the deadline is Friday."}]},
        indent=2), encoding="utf-8")
    clean_session(session, tinyfix_enabled=False)
    return session


def test_dictation_reachable_via_existing_tools(tmp_path):
    """No new dictation tool: source=solo finds it, the clean
    variant returns its cleaned note."""
    import time

    dictation = _make_dictation_session(tmp_path)
    time.sleep(1.1)
    solo = _make_session(tmp_path, _budget_segments(), source="solo")
    tools = OffAirMCPTools(tmp_path)
    listed = tools.list_sessions(source="solo")["results"]
    assert {r["session_id"] for r in listed} == {dictation.name,
                                                 solo.name}
    by_id = {r["session_id"]: r for r in listed}
    assert by_id[dictation.name]["dictation"] is True
    # Solo source alone is not dictation: the flag is what marks it.
    assert by_id[solo.name]["dictation"] is False
    found = tools.search_meetings("", mode="text", source="solo")
    assert {r["session_id"] for r in found["results"]} == {
        dictation.name, solo.name}
    assert tools.get_session(dictation.name)["dictation"] is True
    # The cleaned note is the primary readable artifact.
    note = tools.get_transcript(dictation.name, variant="clean")
    assert "Remember to call Priya about the budget." in note["text"]
    assert "Um," not in note["text"]
    # The repeat is kept verbatim: repeat dedup was removed Nov 22
    # 2026 (see docs/notes/cleanup-repeat-rule-2026-11-22.md).
    assert "The the deadline is Friday." in note["text"]
    # The verbatim variant is untouched by the clean pass.
    verbatim = tools.get_transcript(dictation.name)
    assert "Um, remember to call Priya" in verbatim["text"]


# ------------------------------------------------------------ no network ---
def test_semantic_tools_survive_different_calling_threads(demo_root):
    """Regression: the SDK runs each tool on a worker thread.

    The thread varies per call, so a cached sqlite connection (bound
    to the creating thread) raised ``sqlite3.ProgrammingError`` on
    the second semantic search that landed on another thread. Calls
    from a thread pool must all succeed.
    """
    import concurrent.futures

    root, _, _ = demo_root
    tools = OffAirMCPTools(root)  # no injected index: real MiniLM path
    # When the model is missing the tools fall back to text search,
    # so this stays green offline — but then it cannot exercise the
    # sqlite path, and the regression only bites when the model IS
    # available (the cached model in ~/.offair/models).
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for i in range(8):
            if i % 2:
                futures.append(pool.submit(
                    tools.search_meetings, "budget pilot", "hybrid"))
            else:
                futures.append(pool.submit(
                    tools.ask_meetings, "What is the budget?"))
        for future in futures:
            out = future.result()
            assert "results" in out


def test_no_tool_path_touches_the_network(tools, monkeypatch):
    """Every tool must work with the network unplugged.

    socket.socket and urllib.request.urlopen both raise; the only
    HTTP the tools can ever attempt is the Ollama localhost probe,
    which must fail soft (is_available -> False -> search fallback).
    """
    def _blocked(*args, **kwargs):
        raise OSError("network is unplugged for this test")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    assert tools.search_meetings("budget", mode="text")["results"]
    assert tools.search_meetings("budget", mode="hybrid")["results"]
    assert len(tools.list_sessions()["results"]) == 2
    session_id = tools.list_sessions()["results"][0]["session_id"]
    assert tools.get_session(session_id)["session_id"] == session_id
    assert tools.get_transcript(session_id)["text"]
    assert tools.list_speakers() == {"results": []}
    assert tools.what_did_speaker_say("Marcus")["results"]
    assert tools.get_speaker("Marcus")["utterances"]
    assert tools.list_bookmarks(session_id)["results"] == []
    out = tools.ask_meetings("What is the budget?")
    assert out["answer"] is None and out["source"] == "search-fallback"


# ------------------------------------------------------------ stdio e2e ----
@pytest.mark.skipif(not HAS_MCP, reason="mcp SDK not installed")
def test_stdio_server_lists_and_calls_all_tools(tmp_path):
    """Real MCP client over stdio against the real server entry point.

    Spawns `python -m app.mcp --output-dir <tmp>` as a subprocess
    (the same code `OffAir.exe --mcp` / `offair-mcp` runs) and
    calls all twelve tools through the SDK client. The scheduling
    tools write only to a `--schedules-file` under tmp — never
    the real ~/.offair/schedules.json.
    """
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    budget = _make_session(tmp_path, _budget_segments(), source="teams",
                           source_detail="Microsoft Teams",
                           names={"SPEAKER_00": "John",
                                  "SPEAKER_01": "Marcus"})
    update_metadata(budget, pinned=True,
                    bookmarks=[{"t": 5.0, "label": "Budget decision",
                                "created": "2026-11-21T20:00:00+00:00"}])
    (budget / "combined.wav").write_bytes(b"")
    (budget / "transcript-clean.txt").write_text(
        "[00:00] John: Good morning everyone.\n", encoding="utf-8")
    (budget / "transcript-enterprise-clean.txt").write_text(
        "# Transcript (enterprise-cleaned)\n\n"
        "[00:00] John: Good morning.\n", encoding="utf-8")
    # The server subprocess needs `mcp` importable: when the test
    # runner's own PYTHONPATH already carries the SDK (e.g. the
    # scratch venv's site-packages, see docs/MCP.md), pass it
    # through; otherwise the repo root alone suffices (a venv with
    # `pip install -r requirements.txt` has `mcp` already).
    env = dict(os.environ)
    path_entries = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        path_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(path_entries)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp", "--output-dir", str(tmp_path),
              "--schedules-file", str(tmp_path / "schedules.json")],
        env=env)

    async def _run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                names = {t.name for t in listed.tools}
                assert names == {"search_meetings", "list_sessions",
                                 "get_transcript", "get_session",
                                 "list_bookmarks", "list_speakers",
                                 "get_speaker", "what_did_speaker_say",
                                 "ask_meetings", "schedule_recording",
                                 "list_schedules", "cancel_schedule"}
                write_tools = {"schedule_recording", "cancel_schedule"}
                for tool in listed.tools:
                    assert tool.annotations is not None
                    assert tool.annotations.read_only_hint is \
                        (tool.name not in write_tools)

                async def call(name, args):
                    result = await session.call_tool(name, args)
                    assert not result.is_error, (name, result)
                    text = "".join(
                        c.text for c in result.content
                        if c.type == "text")
                    return json.loads(text)

                out = await call("search_meetings",
                                 {"query": "budget", "mode": "text"})
                assert out["results"][0]["session_id"] == budget.name
                out = await call("search_meetings",
                                 {"query": "budget", "source": "teams"})
                assert len(out["results"]) == 1
                out = await call("list_sessions", {"limit": 5})
                assert len(out["results"]) == 1
                out = await call("list_sessions",
                                 {"limit": 5, "source": "zoom"})
                assert out["results"] == []
                out = await call("get_session",
                                 {"session_id": budget.name})
                assert out["source"] == "teams"
                assert out["source_detail"] == "Microsoft Teams"
                assert out["calendar_event"] is None
                assert out["pinned"] is True
                assert out["bookmarks"] == [
                    {"t": 5.0, "label": "Budget decision",
                     "created": "2026-11-21T20:00:00+00:00",
                     "audio_available": True}]
                out = await call("list_bookmarks",
                                 {"session_id": budget.name})
                assert len(out["results"]) == 1
                assert out["results"][0]["label"] == "Budget decision"
                out = await call("list_sessions",
                                 {"limit": 5, "pinned_only": True})
                assert len(out["results"]) == 1
                out = await call("search_meetings",
                                 {"query": "budget", "mode": "text",
                                  "speaker": "Priya"})
                assert out["results"] == []
                out = await call("get_speaker", {"name": "Marcus"})
                assert out["utterance_count"] == 1
                assert "budget for the pilot" in out["utterances"][0][
                    "text"]
                out = await call("get_transcript",
                                 {"session_id": budget.name,
                                  "variant": "enterprise"})
                assert out["variant"] == "enterprise"
                assert "external LLM" in (out["note"] or "")
                assert "Good morning." in out["text"]
                out = await call("get_transcript",
                                 {"session_id": budget.name,
                                  "format": "segments"})
                assert out["segments"][0]["speaker"] == "John"
                out = await call("list_speakers", {})
                assert out == {"results": []}
                out = await call("what_did_speaker_say",
                                 {"speaker": "Marcus"})
                assert len(out["results"]) == 1
                out = await call("ask_meetings",
                                 {"question": "What is the budget?"})
                # No Ollama in the test env: fallback, never invented.
                assert out["answer"] is None
                assert out["source"] == "search-fallback"
                # Scheduling tools: create, list, cancel — writing
                # only to the tmp schedules file.
                from datetime import datetime, timedelta, timezone
                fire_at = (datetime.now(timezone.utc)
                           + timedelta(hours=2)).isoformat()
                out = await call("schedule_recording",
                                 {"fire_at": fire_at,
                                  "target_app": "Microsoft Teams",
                                  "subject": "Weekly sync"})
                schedule_id = out["schedule"]["id"]
                assert out["schedule"]["target_app"] == "ms-teams.exe"
                assert out["schedule"]["status"] == "pending"
                out = await call("list_schedules", {})
                assert [s["id"] for s in out["results"]] == [schedule_id]
                out = await call("cancel_schedule",
                                 {"schedule_id": schedule_id})
                assert out["schedule"]["status"] == "cancelled"

    asyncio.run(_run())


# ------------------------------------------------------- scheduling tools ----
def _sched_tools(tmp_path):
    return OffAirMCPTools(tmp_path,
                          schedules_path=tmp_path / "schedules.json")


def _future_iso(hours=2):
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc)
            + timedelta(hours=hours)).isoformat(timespec="seconds")


def test_schedule_recording_creates_pending_schedule(tmp_path):
    tools = _sched_tools(tmp_path)
    out = tools.schedule_recording(
        _future_iso(), "Microsoft Teams", subject="Weekly sync",
        attendees=["Priya"], organiser="John")
    schedule = out["schedule"]
    assert schedule["status"] == "pending"
    assert schedule["target_app"] == "ms-teams.exe"
    assert schedule["subject"] == "Weekly sync"
    assert schedule["attendees"] == ["Priya"]
    assert schedule["organiser"] == "John"
    assert schedule["created_by"] == "mcp"
    assert schedule["id"]
    listed = tools.list_schedules()["results"]
    assert [s["id"] for s in listed] == [schedule["id"]]


def test_schedule_recording_display_name_and_exe_kept(tmp_path):
    tools = _sched_tools(tmp_path)
    out = tools.schedule_recording(_future_iso(), "zoom.exe")
    assert out["schedule"]["target_app"] == "zoom.exe"
    out = tools.schedule_recording(_future_iso(3), "Google Chrome")
    assert out["schedule"]["target_app"] == "chrome.exe"
    # Unknown names are kept as given (per-app capture fails open).
    out = tools.schedule_recording(_future_iso(4), "weird-app.exe")
    assert out["schedule"]["target_app"] == "weird-app.exe"


def test_schedule_recording_validation_errors(tmp_path):
    tools = _sched_tools(tmp_path)
    with pytest.raises(ValueError, match="not in the future"):
        tools.schedule_recording("2020-01-01T09:00:00+00:00",
                                 "ms-teams.exe")
    with pytest.raises(ValueError, match="Could not parse"):
        tools.schedule_recording("tomorrow-ish", "ms-teams.exe")
    with pytest.raises(ValueError, match="must not be blank"):
        tools.schedule_recording(_future_iso(), "   ")


def test_schedule_recording_naive_fire_time_is_local(tmp_path):
    from datetime import datetime, timedelta
    tools = _sched_tools(tmp_path)
    naive = (datetime.now().astimezone().replace(tzinfo=None)
             + timedelta(hours=2))
    out = tools.schedule_recording(naive.isoformat(), "ms-teams.exe")
    # Naive input must come back offset-aware. Parse instead of hunting
    # for "+" — a western-hemisphere offset is "-05:00", which the old
    # "+"-or-"Z" check rejected even though the value was correct
    # (Nov 22, 2026: this VM runs EST).
    assert datetime.fromisoformat(out["schedule"]["fire_at"]).tzinfo \
        is not None


def test_list_schedules_pending_first_then_history(tmp_path):
    tools = _sched_tools(tmp_path)
    first = tools.schedule_recording(
        _future_iso(1), "ms-teams.exe")["schedule"]
    second = tools.schedule_recording(
        _future_iso(2), "zoom.exe")["schedule"]
    tools.cancel_schedule(first["id"])
    results = tools.list_schedules()["results"]
    assert [s["id"] for s in results] == [second["id"], first["id"]]
    assert results[1]["status"] == "cancelled"


def test_cancel_schedule_errors(tmp_path):
    tools = _sched_tools(tmp_path)
    with pytest.raises(ValueError, match="must not be blank"):
        tools.cancel_schedule("  ")
    with pytest.raises(ValueError, match="No schedule"):
        tools.cancel_schedule("missing")
    created = tools.schedule_recording(
        _future_iso(), "ms-teams.exe")["schedule"]
    tools.cancel_schedule(created["id"])
    with pytest.raises(ValueError, match="not pending"):
        tools.cancel_schedule(created["id"])


def test_scheduling_tools_touch_only_the_store_file(tmp_path):
    # The scoped write exception: scheduling writes the store file
    # and nothing else under the output root — no session, no
    # transcript, no database, no setting.
    before = {p.name for p in tmp_path.iterdir()}
    tools = _sched_tools(tmp_path)
    created = tools.schedule_recording(
        _future_iso(), "ms-teams.exe")["schedule"]
    tools.list_schedules()
    tools.cancel_schedule(created["id"])
    after = {p.name for p in tmp_path.iterdir()}
    assert after - before == {"schedules.json"}
