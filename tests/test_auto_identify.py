"""Unit tests — automatic speaker identification (Nov 8 2026).

Default behaviour after a recording stops: unnamed clusters are
identified without a Confirm click. A cluster that matches a known
speaker confidently is applied; a cluster that matches nobody and
passes the clean-cluster guards is enrolled as a persistent unnamed
identity ("Speaker A", "Speaker B", …) so the same voice keeps the
same label in later recordings. Ambiguous or dirty clusters are
left Unknown. All embeddings here are fake (SequenceExtractor);
no model is loaded.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from app.speakers.database import SpeakerDatabase
from app.speakers.session import (
    auto_identify_session,
    load_suggestions,
    rename_identity,
)
from app.transcription.engine import TRANSCRIPT_JSON

SHERPA_METHOD = ("sherpa-onnx (pyannote segmentation ONNX + "
                 "ERes2Net embeddings, threshold clustering)")


def unit(*values):
    vec = np.array(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


class SequenceExtractor:
    """Fake extractor: returns the next vector per call, in order."""

    backend = "fallback-spectral-v1"
    dim = 2

    def __init__(self, vectors):
        self._vectors = list(vectors)

    def extract(self, wav_path, windows=None):
        if not self._vectors:
            raise AssertionError("SequenceExtractor ran out of vectors")
        return self._vectors.pop(0)


def _make_auto_session(root: Path, name: str,
                       segments: list[tuple[float, float, str]],
                       turns: list[tuple[float, float, str]]) -> Path:
    """Session dir with transcript.json + diarization.json (sherpa turns)."""
    session = root / name
    session.mkdir(parents=True)
    payload_segments = [
        {"start": start, "end": end, "text": f"segment {i}",
         "speaker": label}
        for i, (start, end, label) in enumerate(segments)]
    (session / TRANSCRIPT_JSON).write_text(json.dumps(
        {"segments": payload_segments}), encoding="utf-8")
    (session / "combined.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    (session / "diarization.json").write_text(json.dumps(
        {"mode": "sherpa", "method": SHERPA_METHOD,
         "turns": [{"start": s, "end": e, "speaker": sp}
                   for s, e, sp in turns],
         "segments": payload_segments}), encoding="utf-8")
    return session


def _speaker_names(session: Path) -> dict:
    path = session / "speakers.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["names"]


# ------------------------------------------------------- enrol + recognise --
def test_auto_enrols_unknown_voice_and_recognises_it_next_session(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    voice = unit(1, 0)
    # Session 1: a new voice, three clean 5 s segments, one turn.
    s1 = _make_auto_session(
        tmp_path, "s1",
        [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_00"),
         (10.0, 15.0, "SPEAKER_00")],
        [(0.0, 15.0, "SPEAKER_00")])
    # Calls: suggest (dirty) + halves (2) + clean (1).
    actions = auto_identify_session(
        s1, db, extractor=SequenceExtractor([voice] * 4))
    assert actions["enrolled"] == [
        {"label": "SPEAKER_00", "name": "Speaker A", "speaker_id": 1}]
    assert actions["applied"] == [] and actions["left_unknown"] == []
    assert db.get_speaker("Speaker A")["sample_count"] == 1
    assert _speaker_names(s1) == {"SPEAKER_00": "Speaker A"}
    assert "Speaker A" in (s1 / "transcript.txt").read_text(encoding="utf-8")
    # Session 2: the same voice, slightly different sample.
    s2 = _make_auto_session(
        tmp_path, "s2",
        [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_00"),
         (10.0, 15.0, "SPEAKER_00")],
        [(0.0, 15.0, "SPEAKER_00")])
    voice2 = unit(0.99, 0.14)
    actions = auto_identify_session(
        s2, db, extractor=SequenceExtractor([voice2] * 4))
    assert actions["applied"] == [
        {"label": "SPEAKER_00", "name": "Speaker A"}]
    assert actions["enrolled"] == []
    assert db.get_speaker("Speaker A")["sample_count"] == 2
    assert _speaker_names(s2) == {"SPEAKER_00": "Speaker A"}
    # One identity, one label, across both sessions.
    assert [s["name"] for s in db.list_speakers()] == ["Speaker A"]
    db.close()


def test_over_segmented_fragments_share_one_identity(tmp_path):
    """Sherpa splits one voice into several clusters; they must merge.

    The first fragment enrols "Speaker A"; the second fragment's
    clean embedding is matched against the live database (which
    now holds Speaker A) and applied to the same identity — no
    second "Speaker B" for the same voice.
    """
    db = SpeakerDatabase(tmp_path / "speakers.db")
    session = _make_auto_session(
        tmp_path, "s1",
        [(0.0, 5.0, "SPEAKER_01"), (5.0, 10.0, "SPEAKER_01"),
         (20.0, 25.0, "SPEAKER_02"), (25.0, 30.0, "SPEAKER_02")],
        [(0.0, 10.0, "SPEAKER_01"), (20.0, 30.0, "SPEAKER_02")])
    frag1, frag2 = unit(1, 0), unit(0.99, 0.14)
    # Calls in order: suggest SPEAKER_01, suggest SPEAKER_02, then
    # per cluster halves (2) + clean (1).
    vectors = [frag1, frag2,
               frag1, frag1, frag1,
               frag2, frag2, frag2]
    actions = auto_identify_session(
        session, db, extractor=SequenceExtractor(vectors))
    assert actions["enrolled"] == [
        {"label": "SPEAKER_01", "name": "Speaker A", "speaker_id": 1}]
    assert actions["applied"] == [
        {"label": "SPEAKER_02", "name": "Speaker A"}]
    assert [s["name"] for s in db.list_speakers()] == ["Speaker A"]
    assert db.get_speaker("Speaker A")["sample_count"] == 2
    assert _speaker_names(session) == {"SPEAKER_01": "Speaker A",
                                       "SPEAKER_02": "Speaker A"}
    db.close()


# ------------------------------------------------------------ guardrails ----
def test_mixed_cluster_is_not_enrolled(tmp_path):
    """Two halves of one cluster sounding like different voices."""
    db = SpeakerDatabase(tmp_path / "speakers.db")
    session = _make_auto_session(
        tmp_path, "s1",
        [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_00"),
         (10.0, 15.0, "SPEAKER_00"), (15.0, 20.0, "SPEAKER_00")],
        [(0.0, 20.0, "SPEAKER_00")])
    # Dirty (1,0); first half (1,0); second half (0,1) — different voice.
    actions = auto_identify_session(
        session, db,
        extractor=SequenceExtractor([unit(1, 0), unit(1, 0), unit(0, 1)]))
    assert actions["enrolled"] == [] and actions["applied"] == []
    assert actions["left_unknown"] == [
        {"label": "SPEAKER_00", "reason": "mixed_cluster"}]
    assert db.list_speakers() == []
    assert _speaker_names(session) == {}
    saved = load_suggestions(session)
    assert saved[0]["auto_action"] == "left_unknown"
    assert "mixed cluster" in saved[0]["auto_detail"]
    db.close()


def test_boundary_spill_cluster_is_not_enrolled(tmp_path):
    """YouTube ep1 shape: a 4.6 s segment mislabelled across a turn
    boundary, plus 0.8 s of real audio — only 0.8 s is clean."""
    db = SpeakerDatabase(tmp_path / "speakers.db")
    session = _make_auto_session(
        tmp_path, "ep1",
        [(82.8, 87.4, "SPEAKER_01"), (87.4, 88.2, "SPEAKER_01")],
        [(0.0, 85.0, "SPEAKER_00"), (85.0, 88.2, "SPEAKER_01")])
    # Only the suggest (dirty) extraction runs — the clean-audio
    # gate fails before any enrolment extraction.
    actions = auto_identify_session(
        session, db, extractor=SequenceExtractor([unit(1, 0)]))
    assert actions["enrolled"] == [] and actions["applied"] == []
    assert actions["left_unknown"] == [
        {"label": "SPEAKER_01", "reason": "too_little_clean_audio"}]
    assert db.list_speakers() == []
    db.close()


def test_ambiguous_match_is_left_unknown_not_enrolled(tmp_path):
    """A near-tie between two known speakers must not fork a new identity."""
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("John", unit(1, 0), backend="fallback-spectral-v1")
    db.add_embedding("Marcus", unit(0.99, 0.14),
                     backend="fallback-spectral-v1")
    session = _make_auto_session(
        tmp_path, "s1",
        [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_00"),
         (10.0, 15.0, "SPEAKER_00")],
        [(0.0, 15.0, "SPEAKER_00")])
    voice = unit(1, 0.02)  # ~0.9998 vs John, ~0.993 vs Marcus: margin < 0.10
    actions = auto_identify_session(
        session, db, extractor=SequenceExtractor([voice] * 4))
    assert actions["enrolled"] == [] and actions["applied"] == []
    assert actions["left_unknown"] == [
        {"label": "SPEAKER_00", "reason": "ambiguous_match"}]
    assert [s["name"] for s in db.list_speakers()] == ["John", "Marcus"]
    db.close()


# ------------------------------------------------------- rename propagation -
def test_rename_identity_propagates_to_past_sessions(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Speaker A", unit(1, 0), backend="fallback-spectral-v1")
    sessions = []
    for name in ("2026-11-08_10-00-00_teams",
                 "2026-11-08_11-00-00_teams"):
        session = _make_auto_session(
            tmp_path, name,
            [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_00")],
            [(0.0, 10.0, "SPEAKER_00")])
        (session / "speakers.json").write_text(json.dumps(
            {"version": 1, "names": {"SPEAKER_00": "Speaker A"}}),
            encoding="utf-8")
        sessions.append(session)
    result = rename_identity(db, tmp_path, "Speaker A", "Priya")
    assert result["name"] == "Priya"
    assert sorted(result["sessions_updated"]) == sorted(
        s.name for s in sessions)
    assert result["sessions_skipped"] == []
    assert db.get_speaker("Priya") is not None
    assert db.get_speaker("Speaker A") is None
    for session in sessions:
        assert _speaker_names(session) == {"SPEAKER_00": "Priya"}
        assert "Priya" in (session / "transcript.txt").read_text(
            encoding="utf-8")
    db.close()


def test_rename_identity_rejects_unknown_and_taken_names(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Speaker A", unit(1, 0), backend="fallback-spectral-v1")
    db.add_embedding("Priya", unit(0, 1), backend="fallback-spectral-v1")
    with pytest.raises(ValueError, match="No known speaker"):
        rename_identity(db, tmp_path, "Nobody", "Priya")
    with pytest.raises(ValueError, match="already exists"):
        rename_identity(db, tmp_path, "Speaker A", "Priya")
    assert db.get_speaker("Speaker A") is not None  # unchanged
    db.close()


def test_next_unnamed_name_skips_taken_letters(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Speaker A", unit(1, 0), backend="fallback-spectral-v1")
    db.add_embedding("Speaker B", unit(0, 1), backend="fallback-spectral-v1")
    session = _make_auto_session(
        tmp_path, "s1",
        [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_00"),
         (10.0, 15.0, "SPEAKER_00")],
        [(0.0, 15.0, "SPEAKER_00")])
    actions = auto_identify_session(
        session, db, extractor=SequenceExtractor([unit(0.7, 0.7)] * 4))
    assert actions["enrolled"][0]["name"] == "Speaker C"
    db.close()
