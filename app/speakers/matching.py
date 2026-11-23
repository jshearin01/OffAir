"""Match a new speaker embedding against known speakers (PRD Phase 4).

Cosine similarity (a dot product, since every embedding is unit
length). A suggestion is made only when the best match clears
``threshold``; the result always carries the best and second-best
scores and their margin, so the UI can show *how* confident the
suggestion is instead of a bare name. A small margin (two speakers
nearly tied) is a weak suggestion even at a high absolute score —
the fallback extractor's John/Marcus confusion is exactly this
shape (see PHASE4.md), so callers should surface the margin.

``confirm_suggestion`` / ``correct_suggestion`` are the learning
loop: a confirmed suggestion adds the new sample to that speaker
(the stored mean moves towards the new voice); a correction adds
it to the right speaker instead. Both are explicit user actions —
nothing is auto-learned without a confirm/correct, so a wrong
auto-label can never silently poison a speaker's profile.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.config import (
    SPEAKER_MATCH_THRESHOLD,
    SPEAKER_MATCH_THRESHOLDS,
)
from app.speakers.database import SpeakerDatabase

logger = logging.getLogger("offair.speakers.matching")

# Fallback when a caller passes an explicit threshold, or for a
# backend with no measured threshold of its own (see
# SPEAKER_MATCH_THRESHOLDS in app/config.py).
DEFAULT_MATCH_THRESHOLD = SPEAKER_MATCH_THRESHOLD
# Below this margin the suggestion is flagged as ambiguous even if
# the top score clears the threshold (two speakers nearly tied).
AMBIGUOUS_MARGIN = 0.05


@dataclass
class MatchResult:
    """One matching decision for a new speaker cluster."""

    suggested_name: str | None
    confidence: float  # cosine similarity of the best match (0..1), or 0.0
    best_name: str | None
    best_score: float
    second_best_name: str | None
    second_best_score: float
    threshold: float
    backend: str

    @property
    def margin(self) -> float:
        return self.best_score - self.second_best_score

    @property
    def is_ambiguous(self) -> bool:
        """Best match clears the threshold but a rival is nearly tied."""
        return (self.suggested_name is not None
                and self.margin < AMBIGUOUS_MARGIN)

    @property
    def is_unknown(self) -> bool:
        return self.suggested_name is None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (0..1 for unit embeddings)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.clip(float(a @ b) / denom, -1.0, 1.0))


def threshold_for_backend(backend: str) -> float:
    """Match threshold for ``backend`` (per-backend, see config).

    ECAPA backends score same-speaker pairs around 0.8 and
    different-speaker pairs around 0.1 on the fixture, so they
    need a much lower threshold than the fallback (0.90) — with
    0.90 an ECAPA backend would never suggest anyone. Unknown
    backends get the conservative default (0.90).
    """
    return SPEAKER_MATCH_THRESHOLDS.get(backend, DEFAULT_MATCH_THRESHOLD)


def match_embedding(embedding: np.ndarray, db: SpeakerDatabase, *,
                    backend: str,
                    threshold: float | None = None) -> MatchResult:
    """Match ``embedding`` against every known speaker of ``backend``.

    Only speakers of the same backend are considered — embeddings
    from different backends live in different vector spaces.
    ``threshold`` defaults to the per-backend threshold
    (:func:`threshold_for_backend`); pass an explicit value to
    override it. Returns a MatchResult; ``suggested_name`` is None
    ("Unknown") when nothing clears the threshold or the database
    is empty.
    """
    if threshold is None:
        threshold = threshold_for_backend(backend)
    vec = np.asarray(embedding, dtype=np.float32).ravel()
    known = db.all_speaker_embeddings(backend=backend)
    if not known or vec.size == 0:
        return MatchResult(None, 0.0, None, 0.0, None, 0.0,
                           threshold, backend)
    scored = sorted(
        ((name, cosine_similarity(vec, emb)) for name, emb in known.items()),
        key=lambda item: item[1], reverse=True)
    best_name, best_score = scored[0]
    second_name, second_score = (
        scored[1] if len(scored) > 1 else (None, 0.0))
    suggested = best_name if best_score >= threshold else None
    return MatchResult(
        suggested_name=suggested,
        confidence=float(best_score) if suggested else 0.0,
        best_name=best_name, best_score=float(best_score),
        second_best_name=second_name,
        second_best_score=float(second_score),
        threshold=threshold, backend=backend)


def confirm_suggestion(db: SpeakerDatabase, name: str,
                       embedding: np.ndarray, *, backend: str,
                       source_session: str | None = None,
                       source_label: str | None = None) -> int:
    """User confirmed the suggestion: add the sample to ``name``.

    Returns the speaker id. This is the learning step — the stored
    mean embedding moves towards the confirmed voice.
    """
    speaker_id = db.add_embedding(
        name, embedding, backend=backend,
        source_session=source_session, source_label=source_label)
    logger.info("Confirmed speaker %r (sample from %s)", name, source_session)
    return speaker_id


def correct_suggestion(db: SpeakerDatabase, correct_name: str,
                       embedding: np.ndarray, *, backend: str,
                       source_session: str | None = None,
                       source_label: str | None = None) -> int:
    """User corrected the suggestion: add the sample to ``correct_name``.

    The wrongly-suggested speaker is NOT given the sample, so a
    correction never reinforces the mistake. Returns the speaker id.
    """
    speaker_id = db.add_embedding(
        correct_name, embedding, backend=backend,
        source_session=source_session, source_label=source_label)
    logger.info("Corrected speaker to %r (sample from %s)",
                correct_name, source_session)
    return speaker_id
