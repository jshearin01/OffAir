"""Semantic transcript search tests.

A deterministic fake embedder stands in for the real MiniLM
model: each word maps to a fixed pseudo-random unit vector
(seeded by the word itself), so texts that share words score
higher — enough to test ranking, thresholds, hybrid merging
and index persistence without any download. The fake cannot
do real semantics ("money" finding "budget"); that is covered
by the gated real-model test at the bottom and by the manual
verification run, not by the fake.

The real-model test only runs with OFFAIR_TEST_REAL_MINILM=1
and skips gracefully (never fails) when the model cannot be
downloaded — the suite must stay green offline.
"""
from __future__ import annotations

import json
import os
import re
import zlib

import numpy as np
import pytest

from app.config import AppConfig
from app.storage.browser import hybrid_search, search_sessions
from app.storage.semantic_search import (
    MiniLMEmbedder,
    ModelNotAvailableError,
    SemanticSearchIndex,
    WordPieceTokenizer,
    _mean_pool,
    build_chunks,
)
from app.storage.sessions import create_session, update_metadata
from app.transcription.engine import Segment

FAKE_DIM = 1024  # high dim keeps word-overlap signal above
# the cross-term noise (at dim 64, unrelated texts scored
# up to ~0.2 by chance — too noisy for ranking tests).


class FakeEmbedder:
    """Deterministic word-overlap embedder (test-only).

    Each word -> a fixed pseudo-random unit vector; a text's
    embedding is the L2-normalised sum of its words' vectors.
    Cosine similarity then tracks word overlap, which is all
    the mechanics tests need.
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


def _write_transcript(session, segments):
    (session / "transcript.json").write_text(json.dumps(
        {"model": "tiny",
         "segments": [s.to_dict() for s in segments]},
        indent=2), encoding="utf-8")


def _make_session(root, segments, app_name="Teams"):
    session = create_session(root, app_name=app_name)
    update_metadata(session, duration_seconds=61.3)
    _write_transcript(session, segments)
    return session


def _budget_segments():
    return [
        Segment(0.0, 5.0, "Good morning everyone, thanks for joining "
                          "the OffAir review.", "John"),
        Segment(5.0, 15.0, "One more thing. The budget for the pilot is "
                           "twelve thousand dollars, and the deadline is "
                           "the fifteenth of December.", "Marcus"),
        Segment(15.0, 20.0, "Yes, December fifteenth works for me.",
                "John"),
    ]


def _timeline_segments():
    return [
        Segment(0.0, 8.0, "First item is the recording test. I recorded "
                          "a five minute Teams call yesterday and both "
                          "audio tracks came through clearly.", "Priya"),
        Segment(8.0, 18.0, "Second item is transcription. The small "
                           "Whisper model should handle a thirty minute "
                           "meeting in about fifteen minutes on a laptop "
                           "with no graphics card.", "Marcus"),
        Segment(18.0, 26.0, "Priya, can you send the updated timeline "
                            "by Monday morning?", "John"),
        Segment(26.0, 34.0, "I will send it by nine o clock on Monday. "
                            "Marcus, please confirm the hardware list by "
                            "the same day.", "Priya"),
    ]


def _index_with_fixture(tmp_path):
    """Two sessions, one chunk each: budget vs timeline."""
    budget = _make_session(tmp_path, _budget_segments())
    timeline = _make_session(tmp_path, _timeline_segments())
    index = SemanticSearchIndex(tmp_path, embedder=FakeEmbedder())
    return budget, timeline, index


# --- chunking ---------------------------------------------------------------
def test_chunking_merges_segments_up_to_word_cap_with_overlap():
    segments = [Segment(i * 5.0, i * 5.0 + 5.0, f"word{i} " * 10,
                        "John")
                for i in range(6)]
    chunks = build_chunks("s1", segments, max_words=25)
    # 10 words per segment: chunk 1 = segs 0-1 (20 words), chunk 2
    # starts 1 segment back (overlap) = segs 1-2, etc.
    assert len(chunks) == 5
    assert chunks[0].start_s == 0.0
    assert chunks[0].end_s == 10.0
    assert "John: word0" in chunks[0].text
    assert "word1" in chunks[0].text and "word2" not in chunks[0].text
    assert chunks[1].text.startswith("John: word1")
    assert chunks[1].end_s == 15.0
    for chunk in chunks:
        assert len(chunk.text.split()) <= 25


def test_chunking_single_long_segment_becomes_its_own_chunk():
    segments = [Segment(0.0, 60.0, "filler " * 200, "Priya")]
    chunks = build_chunks("s1", segments, max_words=150)
    assert len(chunks) == 1
    assert chunks[0].start_s == 0.0
    assert chunks[0].end_s == 60.0
    assert len(chunks[0].text.split()) == 201  # 200 words + speaker


def test_chunking_skips_empty_segments_and_keeps_speaker_labels():
    segments = [
        Segment(0.0, 2.0, "   ", "John"),
        Segment(2.0, 5.0, "hello there", "Priya"),
        Segment(5.0, 9.0, "", None),
        Segment(9.0, 12.0, "goodbye now", None),
    ]
    chunks = build_chunks("s1", segments)
    assert len(chunks) == 1
    assert chunks[0].text == "Priya: hello there goodbye now"
    assert chunks[0].start_s == 2.0
    assert chunks[0].end_s == 12.0


# --- tokenizer / pooling ----------------------------------------------------
def test_wordpiece_tokenizer_uses_vocab_and_truncates(tmp_path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthe\nbudget\n##s\n"
        "deadline\n", encoding="utf-8")
    tokenizer = WordPieceTokenizer.from_vocab_file(vocab_path)
    ids, mask = tokenizer.encode("The budgets deadline")
    assert ids[0] == tokenizer.cls_id
    assert ids[-1] == tokenizer.sep_id
    assert ids[1] == tokenizer.vocab["the"]
    # "budgets" splits into budget + ##s
    assert tokenizer.vocab["budget"] in ids
    assert tokenizer.vocab["##s"] in ids
    assert len(ids) == len(mask) == 6
    assert all(m == 1 for m in mask)
    ids, _ = tokenizer.encode("the " * 500, max_len=10)
    assert len(ids) == 10
    # Unknown words become [UNK], never crash.
    ids, _ = tokenizer.encode("quizzical")
    assert tokenizer.unk_id in ids


def test_mean_pool_is_l2_normalised_and_ignores_padding():
    hidden = np.array([[[1.0, 0.0], [0.0, 2.0], [9.0, 9.0]]],
                      dtype=np.float32)
    mask = np.array([[1, 1, 0]])
    pooled = _mean_pool(hidden, mask)
    assert pooled.shape == (1, 2)
    assert np.linalg.norm(pooled[0]) == pytest.approx(1.0)
    # Mean of (1,0) and (0,2) is (0.5, 1), normalised.
    expected = np.array([0.5, 1.0])
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(pooled[0], expected)


# --- ranking / threshold ----------------------------------------------------
def test_semantic_search_ranks_word_overlap_first(tmp_path):
    budget, timeline, index = _index_with_fixture(tmp_path)
    hits = index.semantic_search("budget deadline December",
                                 top_k=10, threshold=0.0)
    assert hits
    assert hits[0].session_id == budget.name
    assert "budget" in hits[0].text.lower()
    assert "deadline" in hits[0].text.lower()
    hits2 = index.semantic_search("timeline hardware Monday",
                                  top_k=10, threshold=0.0)
    assert hits2
    assert hits2[0].session_id == timeline.name
    assert "timeline" in hits2[0].text.lower()
    # Scores are sorted best-first.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_threshold_filters_and_empty_query_returns_nothing(tmp_path):
    _budget, _timeline, index = _index_with_fixture(tmp_path)
    all_hits = index.semantic_search("budget deadline", top_k=10,
                                     threshold=0.0)
    assert len(all_hits) >= 2
    top_score = all_hits[0].score
    second_score = all_hits[1].score
    mid = (top_score + second_score) / 2
    filtered = index.semantic_search("budget deadline", top_k=10,
                                     threshold=mid)
    assert len(filtered) == 1
    assert filtered[0].chunk_id == all_hits[0].chunk_id
    assert index.semantic_search("budget", threshold=top_score + 0.01) == []
    assert index.semantic_search("") == []
    assert index.semantic_search("   ") == []


# --- hybrid -----------------------------------------------------------------
def test_hybrid_boosts_exact_matches_and_dedups(tmp_path):
    session, _timeline, index = _index_with_fixture(tmp_path)
    # A second session with transcript.txt only (no transcript.json):
    # not indexed, but still findable by substring.
    txt_only = create_session(tmp_path, app_name="Zoom")
    update_metadata(txt_only, duration_seconds=30.0)
    (txt_only / "transcript.txt").write_text(
        "[00:00] The budget review is next week.\n", encoding="utf-8")
    hits = hybrid_search(tmp_path, "budget", index=index, top_k=10,
                         threshold=0.0)
    assert hits
    # The indexed budget chunk matches both ways: once, boosted.
    budget_hits = [h for h in hits if h.session_id == session.name]
    assert len(budget_hits) == 1
    assert budget_hits[0].match_type == "both"
    assert budget_hits[0].score > 1.0  # cosine + text boost
    assert hits[0].session_id == session.name  # exact match first
    # The txt-only session appears once, as a text hit.
    txt_hits = [h for h in hits if h.session_id == txt_only.name]
    assert len(txt_hits) == 1
    assert txt_hits[0].match_type == "text"
    assert txt_hits[0].chunk_id == -1
    # No (session, chunk) pair appears twice.
    keys = [(h.session_id, h.chunk_id) for h in hits]
    assert len(keys) == len(set(keys))


# --- index persistence / incremental ----------------------------------------
def test_remove_session_drops_only_its_rows(tmp_path):
    budget, timeline, index = _index_with_fixture(tmp_path)
    assert index.index_session(budget) > 0
    assert index.index_session(timeline) > 0
    total = index.chunk_count()
    removed = index.remove_session(budget.name)
    assert removed > 0
    assert index.chunk_count() == total - removed + 1  # +1: index_state row
    assert budget.name not in index.indexed_sessions()
    assert timeline.name in index.indexed_sessions()
    assert index.remove_session(budget.name) == 0
    index.close()


def test_remove_session_without_index_file_is_a_no_op(tmp_path):
    index = SemanticSearchIndex(tmp_path, embedder=FakeEmbedder())
    assert index.remove_session("no-such-session") == 0
    assert not (tmp_path / "search_index.db").exists()


def test_index_is_incremental_and_persists_across_instances(tmp_path):
    session, _timeline, index = _index_with_fixture(tmp_path)
    first = index.index_session(session)
    assert first > 0
    assert index.chunk_count() == first
    # Second run: transcript unchanged, skipped.
    assert index.index_session(session) == 0
    assert index.chunk_count() == first
    # A new index object on the same db sees the same chunks.
    index2 = SemanticSearchIndex(tmp_path, embedder=FakeEmbedder())
    assert index2.chunk_count() == first
    assert session.name in index2.indexed_sessions()
    # Changing the transcript re-indexes.
    segments = _budget_segments()
    segments.append(Segment(46.0, 50.0, "Extra closing remarks "
                                        "about the budget.", "John"))
    _write_transcript(session, segments)
    assert index.index_session(session) > 0
    index.close()
    index2.close()


def test_rebuild_index_reproduces_the_same_chunks(tmp_path):
    _budget, _timeline, index = _index_with_fixture(tmp_path)
    index.ensure_indexed()
    before = index.chunk_count()
    assert before > 0
    counts = index.rebuild_index()
    assert index.chunk_count() == before
    assert sum(counts.values()) == before


def test_session_without_transcript_is_skipped(tmp_path):
    session = create_session(tmp_path, app_name="Teams")
    index = SemanticSearchIndex(tmp_path, embedder=FakeEmbedder())
    assert index.index_session(session) == 0
    assert index.chunk_count() == 0


# --- search_sessions backward compat ----------------------------------------
def test_search_sessions_modes_and_unknown_mode(tmp_path):
    session, _timeline, index = _index_with_fixture(tmp_path)
    # Default (hybrid) behaviour: exact matches first, then meaning.
    hits = search_sessions(tmp_path, "budget", index=index, threshold=0.0)
    assert hits and hits[0].session_id == session.name
    # Text mode behaviour is unchanged.
    hits = search_sessions(tmp_path, "budget", mode="text")
    assert [s.session_id for s in hits] == [session.name]
    assert search_sessions(tmp_path, "no-such-word-anywhere",
                           mode="text") == []
    # Semantic / hybrid modes return sessions ordered by hit.
    sem = search_sessions(tmp_path, "budget deadline", mode="semantic",
                          index=index, threshold=0.0)
    assert sem and sem[0].session_id == session.name
    hyb = search_sessions(tmp_path, "budget", mode="hybrid",
                          index=index, threshold=0.0)
    assert hyb and hyb[0].session_id == session.name
    # Empty query returns everything in every mode (newest
    # first — both sessions were created in the same second,
    # so order between them is by folder-name suffix).
    assert {s.session_id for s in search_sessions(tmp_path, "")} == {
        session.name, _timeline.name}
    assert len(search_sessions(tmp_path, "", mode="semantic",
                               index=index)) == 2
    with pytest.raises(ValueError, match="unknown search mode"):
        search_sessions(tmp_path, "budget", mode="weird")


def test_search_mode_setting_round_trips(tmp_path):
    config = AppConfig(search_mode="hybrid")
    assert config.search_mode == "hybrid"
    from app.config import config_from_settings, settings_to_dict

    data = settings_to_dict(config)
    assert data["search_mode"] == "hybrid"
    restored = config_from_settings(data)
    assert restored.search_mode == "hybrid"
    with pytest.raises(ValueError, match="unknown search mode"):
        AppConfig(search_mode="weird")
    # A bad value in a settings file falls back to the default.
    restored = config_from_settings({"search_mode": "weird"})
    assert restored.search_mode == "hybrid"


# --- real model (gated, skips offline) --------------------------------------
@pytest.mark.skipif(
    os.environ.get("OFFAIR_TEST_REAL_MINILM") != "1",
    reason="Real MiniLM download test: set "
           "OFFAIR_TEST_REAL_MINILM=1 to run it.")
def test_real_minilm_model_ranks_meaning(tmp_path):
    """The real ONNX model: 'money' must find the budget chunk.

    Downloads ~22 MB on first run; skips (never fails) when
    the model cannot be fetched.
    """
    session, _timeline, _ = _index_with_fixture(tmp_path)
    model_dir = tmp_path / "models"
    embedder = MiniLMEmbedder(model_dir=model_dir)
    try:
        index = SemanticSearchIndex(tmp_path, embedder=embedder)
        index.ensure_indexed()
    except ModelNotAvailableError:
        pytest.skip("MiniLM model could not be downloaded here")
    assert index.chunk_count() > 0
    hits = index.semantic_search("money", top_k=3)
    assert hits, "expected 'money' to match something"
    assert "budget" in hits[0].text.lower() or \
        "twelve thousand" in hits[0].text.lower()
    deadline = index.semantic_search("deadline", top_k=3)
    assert deadline and "december" in deadline[0].text.lower()
    from app.storage.semantic_search import SEMANTIC_DEFAULT_THRESHOLD
    assert all(h.score >= SEMANTIC_DEFAULT_THRESHOLD for h in hits)
