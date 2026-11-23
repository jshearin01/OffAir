"""Unit tests — Phase 4 speaker learning (fake embeddings, no models)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from app.speakers.database import SpeakerDatabase
from app.speakers.matching import (
    confirm_suggestion,
    correct_suggestion,
    cosine_similarity,
    match_embedding,
)
from app.speakers.session import (
    MixedClusterError,
    confirm_suggestion as session_confirm,
    correct_suggestion as session_correct,
    evaluate_cluster_cleanliness,
    learn_from_session,
    load_suggestions,
    suggest_for_session,
    write_suggestions,
)
from app.transcription.engine import TRANSCRIPT_JSON


def unit(*values):
    vec = np.array(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


# ------------------------------------------------------------ cosine/match ---
def test_cosine_similarity_basics():
    assert cosine_similarity(unit(1, 0), unit(1, 0)) == pytest.approx(1.0)
    assert cosine_similarity(unit(1, 0), unit(0, 1)) == pytest.approx(0.0)
    assert cosine_similarity(unit(1, 0), unit(-1, 0)) == pytest.approx(-1.0)
    assert cosine_similarity(np.zeros(3), unit(1, 0, 0)) == 0.0


def test_match_empty_db_is_unknown(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    result = match_embedding(unit(1, 0), db, backend="fallback-spectral-v1")
    assert result.is_unknown and result.suggested_name is None
    db.close()


def test_match_same_speaker_high_different_low(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Priya", unit(1, 0), backend="fallback-spectral-v1")
    db.add_embedding("Marcus", unit(0, 1), backend="fallback-spectral-v1")
    hit = match_embedding(unit(0.99, 0.14), db,
                          backend="fallback-spectral-v1")
    assert hit.suggested_name == "Priya"
    assert hit.confidence == pytest.approx(0.99, abs=0.01)
    assert hit.best_name == "Priya" and hit.second_best_name == "Marcus"
    assert hit.margin > 0.5 and not hit.is_ambiguous
    miss = match_embedding(unit(0.707, 0.707), db,
                           backend="fallback-spectral-v1")
    assert miss.is_unknown  # 0.707 < 0.90 threshold
    db.close()


def test_match_threshold_boundary(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Priya", unit(1, 0), backend="fallback-spectral-v1")
    # cos(25 deg) = 0.906 clears 0.90; cos(30 deg) = 0.866 does not.
    assert match_embedding(unit(math.cos(0.44), math.sin(0.44)), db,
                           backend="fallback-spectral-v1").suggested_name == "Priya"
    assert match_embedding(unit(math.cos(0.52), math.sin(0.52)), db,
                           backend="fallback-spectral-v1").is_unknown
    db.close()


def test_match_ambiguous_when_two_speakers_tied(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("John", unit(1, 0), backend="fallback-spectral-v1")
    db.add_embedding("Marcus", unit(0.99, 0.14), backend="fallback-spectral-v1")
    result = match_embedding(unit(1, 0.02), db,
                             backend="fallback-spectral-v1")
    assert result.suggested_name == "John"
    assert result.is_ambiguous  # margin < 0.05: nearly tied
    db.close()


def test_match_ignores_other_backends(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Priya", unit(1, 0), backend="pyannote-embedding-v1")
    result = match_embedding(unit(1, 0), db, backend="fallback-spectral-v1")
    assert result.is_unknown  # pyannote vectors are a different space
    db.close()


def test_confirm_moves_mean_towards_new_sample(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Priya", unit(1, 0), backend="fallback-spectral-v1")
    before = db.speaker_embedding("Priya")
    confirm_suggestion(db, "Priya", unit(0, 1),
                       backend="fallback-spectral-v1")
    after = db.speaker_embedding("Priya")
    assert after is not None
    # Mean of (1,0) and (0,1) is (0.707, 0.707) — it moved.
    assert after[0] == pytest.approx(0.707, abs=0.01)
    assert after[1] == pytest.approx(0.707, abs=0.01)
    assert db.get_speaker("Priya")["sample_count"] == 2
    assert before is not None and not np.allclose(before, after)
    db.close()


def test_correct_does_not_reinforce_wrong_speaker(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("John", unit(1, 0), backend="fallback-spectral-v1")
    correct_suggestion(db, "Marcus", unit(0, 1),
                       backend="fallback-spectral-v1")
    assert db.get_speaker("John")["sample_count"] == 1
    assert db.get_speaker("Marcus")["sample_count"] == 1
    db.close()


# ------------------------------------------------------------------ database ---
def test_database_round_trip_and_rename_delete(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Priya", unit(1, 0), backend="fallback-spectral-v1",
                     source_session="s1", source_label="SPEAKER_00")
    assert db.get_speaker("Priya")["sample_count"] == 1
    assert [s["name"] for s in db.list_speakers()] == ["Priya"]
    db.rename_speaker("Priya", "Priya K")
    assert db.get_speaker("Priya") is None
    assert db.get_speaker("Priya K") is not None
    assert db.speaker_embedding("Priya K") is not None
    assert db.delete_speaker("Priya K") is True
    assert db.delete_speaker("Priya K") is False
    assert db.list_speakers() == []
    db.close()


def test_database_backend_mismatch_raises(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Priya", unit(1, 0), backend="fallback-spectral-v1")
    with pytest.raises(ValueError, match="different backend"):
        db.add_embedding("Priya", unit(1, 0), backend="pyannote-embedding-v1")
    with pytest.raises(ValueError, match="zero embedding"):
        db.add_embedding("Priya", np.zeros(4), backend="fallback-spectral-v1")
    db.close()


def test_database_blank_name_and_rename_errors(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    with pytest.raises(ValueError, match="blank"):
        db.create_speaker("   ", "fallback-spectral-v1")
    db.create_speaker("Priya", "fallback-spectral-v1")
    with pytest.raises(ValueError, match="already exists"):
        db.rename_speaker("Priya", "Priya")
    with pytest.raises(ValueError, match="No known speaker"):
        db.rename_speaker("Nobody", "Someone")
    db.close()


# ------------------------------------------------------------------- session ---
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


def _make_session(tmp_path, labels):
    """Session dir with transcript.json + a dummy combined.wav."""
    session = tmp_path / "session"
    session.mkdir()
    segments = [{"start": float(i * 5), "end": float(i * 5 + 4),
                 "text": f"segment {i}", "speaker": label}
                for i, label in enumerate(labels)]
    (session / TRANSCRIPT_JSON).write_text(json.dumps(
        {"segments": segments}), encoding="utf-8")
    (session / "combined.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    return session


def test_learn_from_session_full_mode_learns_all_clusters(tmp_path):
    session = _make_session(tmp_path, ["SPEAKER_00", "SPEAKER_01"])
    db = SpeakerDatabase(tmp_path / "speakers.db")
    extractor = SequenceExtractor([unit(1, 0), unit(0, 1)])
    learned = learn_from_session(session, db, extractor=extractor)
    assert set(learned) == {"SPEAKER_00", "SPEAKER_01"}
    assert learned["SPEAKER_00"]["name"] == "SPEAKER_00"
    assert db.get_speaker("SPEAKER_01")["sample_count"] == 1
    db.close()


def test_learn_from_session_uses_rename_as_name(tmp_path):
    session = _make_session(tmp_path, ["SPEAKER_00"])
    (session / "speakers.json").write_text(json.dumps(
        {"version": 1, "names": {"SPEAKER_00": "Priya"}}), encoding="utf-8")
    db = SpeakerDatabase(tmp_path / "speakers.db")
    learned = learn_from_session(
        session, db, extractor=SequenceExtractor([unit(1, 0)]))
    assert learned["SPEAKER_00"]["name"] == "Priya"
    assert db.get_speaker("Priya") is not None
    db.close()


def test_learn_simple_mode_skips_remote_mixture(tmp_path):
    session = _make_session(tmp_path, ["You", "Remote"])
    (session / "diarization.json").write_text(json.dumps(
        {"mode": "simple", "method": "dual-channel-energy",
         "segments": []}), encoding="utf-8")
    db = SpeakerDatabase(tmp_path / "speakers.db")
    learned = learn_from_session(
        session, db, extractor=SequenceExtractor([unit(1, 0)]))
    assert set(learned) == {"You"}  # Remote is a mixture, not a person
    assert db.get_speaker("Remote") is None
    db.close()


def test_learn_single_track_mode_learns_nothing(tmp_path):
    session = _make_session(tmp_path, ["SPEAKER_00", "SPEAKER_01"])
    (session / "diarization.json").write_text(json.dumps(
        {"mode": "simple", "method": "single-track-energy (approximate)",
         "segments": []}), encoding="utf-8")
    db = SpeakerDatabase(tmp_path / "speakers.db")
    learned = learn_from_session(session, db, extractor=SequenceExtractor([]))
    assert learned == {}
    db.close()


def test_suggest_and_confirm_renames_session(tmp_path):
    session = _make_session(tmp_path, ["SPEAKER_00"])
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Priya", unit(1, 0), backend="fallback-spectral-v1")
    # Same fake extractor for suggest + confirm: suggest consumes one
    # vector, confirm re-extracts the clean embedding (one more —
    # the 4 s cluster's halves are under the 2.5 s half-split
    # minimum, so no half-split extractions run).
    extractor = SequenceExtractor([unit(0.99, 0.14), unit(0.99, 0.14)])
    suggestions = suggest_for_session(
        session, db, extractor=extractor)
    assert len(suggestions) == 1
    assert suggestions[0].suggested_name == "Priya"
    assert suggestions[0].match.confidence == pytest.approx(0.99, abs=0.01)
    write_suggestions(session, suggestions)
    assert load_suggestions(session)[0]["suggested_name"] == "Priya"
    out = session_confirm(session, db, suggestions[0], extractor=extractor)
    assert out["suggested_name"] == "Priya"
    assert db.get_speaker("Priya")["sample_count"] == 2  # confirmed sample added
    names = json.loads((session / "speakers.json").read_text(encoding="utf-8"))
    assert names["names"]["SPEAKER_00"] == "Priya"
    txt = (session / "transcript.txt").read_text(encoding="utf-8")
    assert "Priya" in txt
    db.close()


def test_suggest_unknown_when_nobody_known(tmp_path):
    session = _make_session(tmp_path, ["SPEAKER_00"])
    db = SpeakerDatabase(tmp_path / "speakers.db")
    suggestions = suggest_for_session(
        session, db, extractor=SequenceExtractor([unit(1, 0)]))
    assert suggestions[0].suggested_name is None
    assert suggestions[0].match.is_unknown
    db.close()


def test_correct_suggestion_learns_right_name_not_wrong_one(tmp_path):
    session = _make_session(tmp_path, ["SPEAKER_00"])
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("John", unit(1, 0), backend="fallback-spectral-v1")
    # Suggest consumes one vector; correcting re-extracts the clean
    # embedding (one more vector) instead of trusting the
    # full-cluster embedding the suggestion carries.
    extractor = SequenceExtractor([unit(0.99, 0.14), unit(0.99, 0.14)])
    suggestions = suggest_for_session(
        session, db, extractor=extractor)
    assert suggestions[0].suggested_name == "John"
    out = session_correct(session, db, suggestions[0], "Marcus",
                          extractor=extractor)
    assert out["corrected_to"] == "Marcus"
    assert db.get_speaker("John")["sample_count"] == 1
    assert db.get_speaker("Marcus")["sample_count"] == 1
    db.close()


def test_learn_requires_transcript(tmp_path):
    session = tmp_path / "empty"
    session.mkdir()
    db = SpeakerDatabase(tmp_path / "speakers.db")
    with pytest.raises(FileNotFoundError, match="transcript"):
        learn_from_session(session, db, extractor=SequenceExtractor([]))
    db.close()


def test_confirm_without_suggestion_raises(tmp_path):
    session = _make_session(tmp_path, ["SPEAKER_00"])
    db = SpeakerDatabase(tmp_path / "speakers.db")
    suggestions = suggest_for_session(
        session, db, extractor=SequenceExtractor([unit(1, 0)]))
    with pytest.raises(ValueError, match="no suggested name"):
        session_confirm(session, db, suggestions[0])
    db.close()


# ------------------------------------------- shared mixed-cluster guard --
def _write_turns(session, turns):
    (session / "diarization.json").write_text(json.dumps(
        {"mode": "sherpa",
         "method": ("sherpa-onnx (pyannote segmentation ONNX + "
                    "ERes2Net embeddings, threshold clustering)"),
         "turns": [{"start": s, "end": e, "speaker": sp}
                   for s, e, sp in turns]}), encoding="utf-8")


def test_learn_from_session_refuses_mixed_cluster(tmp_path):
    """Two halves sounding like different voices: learn_from_session
    must skip the cluster and store nothing for it."""
    session = _make_session(tmp_path, ["SPEAKER_00"] * 4)
    # Four 4 s segments, one 16 s turn: halves are 8 s each.
    _write_turns(session, [(0.0, 20.0, "SPEAKER_00")])
    db = SpeakerDatabase(tmp_path / "speakers.db")
    extractor = SequenceExtractor([unit(1, 0), unit(0, 1)])
    assert learn_from_session(session, db, extractor=extractor) == {}
    assert db.list_speakers() == []
    db.close()


def test_learn_from_session_refuses_boundary_spill_cluster(tmp_path):
    """YouTube ep1 shape: only 0.8 s of a 5.4 s cluster is clean.
    The guard must refuse it without extracting any embedding."""
    session = _make_session(tmp_path, ["SPEAKER_01", "SPEAKER_01"])
    # Rewrite the segments to the ep1 times (82.8–88.2 s).
    segments = [{"start": 82.8, "end": 87.4, "text": "segment 0",
                 "speaker": "SPEAKER_01"},
                {"start": 87.4, "end": 88.2, "text": "segment 1",
                 "speaker": "SPEAKER_01"}]
    (session / TRANSCRIPT_JSON).write_text(json.dumps(
        {"segments": segments}), encoding="utf-8")
    _write_turns(session, [(0.0, 85.0, "SPEAKER_00"),
                           (85.0, 88.2, "SPEAKER_01")])
    db = SpeakerDatabase(tmp_path / "speakers.db")
    assert learn_from_session(session, db,
                              extractor=SequenceExtractor([])) == {}
    assert db.list_speakers() == []
    db.close()


def test_learn_from_session_uses_clean_windows_only(tmp_path):
    """The learned embedding must come from the clean windows only:
    a recording extractor must never be handed the spill segment."""
    seen_windows: list = []

    class RecordingExtractor:
        backend = "fallback-spectral-v1"

        def extract(self, wav_path, windows=None):
            seen_windows.extend(windows or [])
            return unit(1, 0)

    session = _make_session(tmp_path, ["SPEAKER_00", "SPEAKER_00"])
    # Segment 0 is fully inside its own turn; segment 1 (5–9 s)
    # sits entirely inside SPEAKER_01's turn — pure spill.
    _write_turns(session, [(0.0, 5.0, "SPEAKER_00"),
                           (5.0, 10.0, "SPEAKER_01")])
    db = SpeakerDatabase(tmp_path / "speakers.db")
    learned = learn_from_session(session, db,
                                 extractor=RecordingExtractor())
    assert set(learned) == {"SPEAKER_00"}
    assert learned["SPEAKER_00"]["seconds"] == 4.0
    assert seen_windows, "no extraction ran"
    assert all(start >= 0.0 and end <= 5.0
               for start, end in seen_windows), seen_windows
    db.close()


def test_confirm_refuses_mixed_cluster(tmp_path):
    """An explicit Confirm must not learn a mixed cluster: the
    profile stays as it was and no rename is written."""
    session = _make_session(tmp_path, ["SPEAKER_00"] * 4)
    _write_turns(session, [(0.0, 20.0, "SPEAKER_00")])
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Priya", unit(1, 0), backend="fallback-spectral-v1")
    # Suggest (1,0) -> Priya; confirm halves disagree -> mixed.
    extractor = SequenceExtractor(
        [unit(1, 0), unit(1, 0), unit(0, 1)])
    suggestions = suggest_for_session(session, db, extractor=extractor)
    assert suggestions[0].suggested_name == "Priya"
    with pytest.raises(MixedClusterError, match="mixed"):
        session_confirm(session, db, suggestions[0], extractor=extractor)
    assert db.get_speaker("Priya")["sample_count"] == 1
    assert not (session / "speakers.json").exists()
    db.close()


def test_correct_refuses_mixed_cluster(tmp_path):
    """An explicit Correction must not learn a mixed cluster under
    the corrected name either."""
    session = _make_session(tmp_path, ["SPEAKER_00"] * 4)
    _write_turns(session, [(0.0, 20.0, "SPEAKER_00")])
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("John", unit(1, 0), backend="fallback-spectral-v1")
    extractor = SequenceExtractor(
        [unit(1, 0), unit(1, 0), unit(0, 1)])
    suggestions = suggest_for_session(session, db, extractor=extractor)
    assert suggestions[0].suggested_name == "John"
    with pytest.raises(MixedClusterError, match="mixed"):
        session_correct(session, db, suggestions[0], "Marcus",
                        extractor=extractor)
    assert db.get_speaker("John")["sample_count"] == 1
    assert db.get_speaker("Marcus") is None
    db.close()


def test_shared_guard_judges_halves_and_spill_the_same_way(tmp_path):
    """The predicate itself: disagreeing halves are mixed, a
    boundary-spill cluster is mixed, and a clean cluster passes
    with a clean embedding."""
    session = _make_session(tmp_path, ["SPEAKER_00"] * 4)
    _write_turns(session, [(0.0, 20.0, "SPEAKER_00")])
    mixed = evaluate_cluster_cleanliness(
        session, "SPEAKER_00",
        [(0.0, 4.0), (5.0, 9.0), (10.0, 14.0), (15.0, 19.0)],
        SequenceExtractor([unit(1, 0), unit(0, 1)]),
        enforce_minimum_clean_audio=False)
    assert mixed.mixed and mixed.reason == "mixed_cluster"
    assert mixed.clean_embedding is None
    clean = evaluate_cluster_cleanliness(
        session, "SPEAKER_00",
        [(0.0, 4.0), (5.0, 9.0), (10.0, 14.0), (15.0, 19.0)],
        SequenceExtractor([unit(1, 0)] * 3),
        enforce_minimum_clean_audio=False)
    assert not clean.mixed and clean.clean_embedding is not None
    assert clean.clean_seconds == 16.0
