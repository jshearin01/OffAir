"""Learn from / suggest for one recorded session (PRD Phase 4).

Wiring between the diarization output (Phase 3) and the speaker
database (:mod:`app.speakers.database`):

* :func:`learn_from_session` — after diarization, extract one
  embedding per labelled speaker from that session's audio and
  store/update it in the database. The name learned is the session's
  display name for the label (``speakers.json`` rename if one
  exists, else the label itself). ``Silence`` / ``Mixed`` /
  ``Unknown`` clusters are skipped — they are not a person.
* :func:`suggest_for_session` — for a new (diarized) session,
  extract one embedding per label cluster and match it against the
  database. Returns a suggestion per cluster with a confidence
  score; nothing is written to the database here.
* :func:`confirm_suggestion` / :func:`correct_suggestion` — the
  user-facing learning loop: confirming adds the cluster's
  embedding to the suggested speaker *and* renames the label in
  this session; correcting adds it to the right speaker instead.
  Both are explicit user actions (see matching.py).

* :func:`auto_identify_session` — the default (Nov 8 2026):
  after a recording is diarized, unnamed clusters are identified
  automatically instead of waiting for a Confirm click. A cluster
  that matches a known speaker confidently is applied (label
  renamed, sample learned); a cluster that matches nobody and
  passes the clean-cluster guards is enrolled as a persistent
  unnamed identity ("Speaker A", "Speaker B", …) so the same
  voice keeps the same label in later recordings. Ambiguous or
  dirty clusters are left Unknown for the user.

* :func:`rename_identity` — renaming an identity (a name that
  lives in the database, e.g. "Speaker A" → "Priya") renames the
  database speaker *and* every past session that used the old
  name, so the name propagates instead of forking.

Auto-enrolment guards (Nov 8 2026 — from the YouTube test, where
one mislabelled boundary segment silently poisoned a new
speaker's first sample, ~85% wrong-speaker audio, and a
guest+host mixed cluster was nearly taught as one person):

* **Boundary rule.** A segment is only "clean" if at least 80%
  of its duration lies inside diarization turns carrying its own
  label (sherpa/full modes, which publish turns). A segment
  straddling a turn boundary — the exact shape of the YouTube
  poisoning — is excluded from the enrolment audio. Modes
  without turns (simple) have no boundaries to check; their
  learnable clusters are single-speaker by construction ("You"
  is the mic track).
* **Minimum clean audio.** At least 5 s of clean audio, and the
  clean part must be at least half of the cluster's total audio.
  A cluster that is mostly boundary spill fails both.
* **Half-split consistency.** When the clean audio can be split
  into two halves of >= 2.5 s each, both halves are embedded
  separately and must agree (cosine >= 0.5 — same-speaker pairs
  score ~0.8, cross-speaker ~0.1 for ECAPA-family backends).
  A mixed cluster's halves are two different voices and fail.
* **Confident application only.** A match is auto-applied only
  when the best score clears the threshold *and* the margin over
  the runner-up is >= 0.10 — stricter than the 0.05 "ambiguous"
  flag. A near-tie is left for the user.
* **Same-session merge.** Sherpa over-segments short turns (one
  host became four clusters in the YouTube test), and the
  fragments of one voice must merge into one identity, not four.
  This happens through the live database, not a separate check:
  the first fragment enrols the identity, and the clean
  embedding of every later fragment is matched against the
  database *after* that enrolment, so it is applied to the same
  identity.

Audio source per label: ``"You"`` comes from ``mic.wav`` and
``"Remote"`` from ``system.wav`` when those files exist (the two
tracks are the cleanest single-speaker sources); every other
label uses ``combined.wav``. If the preferred track is missing,
the code falls back to ``combined.wav``.

One shared clean-cluster guard (see :func:`evaluate_cluster_cleanliness`)
stands in front of *every* learn path — :func:`learn_from_session`,
:func:`confirm_suggestion`, :func:`correct_suggestion` and
:func:`auto_identify_session`. A cluster whose clean fraction is
below half, or whose two halves sound like different voices, is
mixed and is never enrolled or learned from by any of them; the
embedding learned is always the clean embedding (clean windows
only), never the full-cluster embedding. Automatic enrolment
additionally requires at least 5 s of clean audio — an explicit
confirm/correct of a short but clean cluster is the user's own
decision, while a mixed cluster is refused even then, because
shortness alone does not poison a profile but mixed audio does.

Which clusters are *learnable* depends on how the session was
diarized (read from ``diarization.json``):

* Simple dual-channel mode: only ``"You"`` is one person (your
  mic). ``"Remote"`` mixes every remote speaker into one cluster,
  so learning it would store a blended embedding and poison the
  database — it is skipped with a warning. (In a 1-on-1 call
  Remote happens to be one person, but the label cannot tell
  that, so the safe rule is to skip it.)
* Single-track fallback: labels alternate by pause, not by
  identity, so nothing is learnable.
* Full (pyannote) mode: every ``SPEAKER_xx`` cluster is one
  person, so all of them are learnable.

The same filter applies to suggestions: suggesting a name for a
mixed cluster would be a guess, not a match.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.diarization.names import (
    load_speaker_names,
    rename_speaker as rename_in_session,
    save_speaker_names,
)
from app.diarization.session import reapply_names
from app.speakers.database import SpeakerDatabase
from app.speakers.embeddings import get_default_extractor
from app.speakers.matching import (
    MatchResult,
    confirm_suggestion as db_confirm,
    correct_suggestion as db_correct,
    cosine_similarity,
    match_embedding,
    threshold_for_backend,
)
from app.storage.sessions import COMBINED_FILE, MIC_FILE, SYSTEM_FILE
from app.transcription.engine import TRANSCRIPT_JSON
from app.transcription.exports import load_transcript_json
from app.transcription.names_dictionary import add_name

logger = logging.getLogger("offair.speakers.session")

SUGGESTIONS_FILE = "speaker_suggestions.json"
SKIP_LABELS = {"Silence", "Mixed", "Unknown", "silence", "mixed", "unknown"}


@dataclass
class ClusterSuggestion:
    """One diarized label cluster and what the database thinks it is."""

    label: str
    match: MatchResult
    seconds: float
    embedding: np.ndarray = field(repr=False)
    backend: str = ""

    @property
    def suggested_name(self) -> str | None:
        return self.match.suggested_name

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "suggested_name": self.match.suggested_name,
            "confidence": round(self.match.confidence, 3),
            "best_name": self.match.best_name,
            "best_score": round(self.match.best_score, 3),
            "second_best_name": self.match.second_best_name,
            "second_best_score": round(self.match.second_best_score, 3),
            "margin": round(self.match.margin, 3),
            "ambiguous": self.match.is_ambiguous,
            "unknown": self.match.is_unknown,
            "threshold": self.match.threshold,
            "backend": self.backend,
            "seconds": round(self.seconds, 2),
        }


def _audio_path_for_label(session_dir: Path, label: str) -> Path:
    """Best audio file for one label cluster (see module docstring)."""
    combined = session_dir / COMBINED_FILE
    if label == "You" and (session_dir / MIC_FILE).exists():
        return session_dir / MIC_FILE
    if label == "Remote" and (session_dir / SYSTEM_FILE).exists():
        return session_dir / SYSTEM_FILE
    if combined.exists():
        return combined
    mic = session_dir / MIC_FILE
    if mic.exists():
        return mic
    raise FileNotFoundError(
        f"No audio file in {session_dir} to extract a speaker embedding")


def _clusters(session_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """{speaker label: [(start, end), ...]} from transcript.json."""
    transcript_path = session_dir / TRANSCRIPT_JSON
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"No {TRANSCRIPT_JSON} in {session_dir} — transcribe and "
            "diarize the session first.")
    segments = load_transcript_json(transcript_path)
    clusters: dict[str, list[tuple[float, float]]] = {}
    for seg in segments:
        label = getattr(seg, "speaker", None)
        if not label or label in SKIP_LABELS:
            continue
        clusters.setdefault(str(label), []).append(
            (float(seg.start), float(seg.end)))
    return clusters


def _diarization_method(session_dir: Path) -> str | None:
    """The ``method`` string from diarization.json, or None."""
    path = session_dir / "diarization.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("method")
    except (json.JSONDecodeError, OSError):
        return None


def _learnable_clusters(session_dir: Path,
                        clusters: dict[str, list[tuple[float, float]]]
                        ) -> tuple[dict[str, list[tuple[float, float]]], list[str]]:
    """Filter clusters to the ones that are a single person (see docstring).

    Returns (learnable, skipped-labels). Simple dual-channel mode
    keeps only ``"You"``; single-track mode keeps nothing; full
    mode (or no diarization.json, i.e. clusters were supplied
    another way) keeps every non-skipped cluster.
    """
    method = _diarization_method(session_dir)
    if method is None:
        return dict(clusters), []
    if "single-track" in method:
        logger.warning(
            "Not learning from %s: single-track diarization labels "
            "alternate by pause, not by identity", session_dir.name)
        return {}, list(clusters)
    if "dual-channel" in method:
        learnable = {k: v for k, v in clusters.items() if k == "You"}
        skipped = [k for k in clusters if k != "You"]
        if skipped:
            logger.warning(
                "Skipping %s in %s: simple-mode %s mixes all remote "
                "speakers into one cluster, so it is not one person",
                ", ".join(skipped), session_dir.name, "Remote")
        return learnable, skipped
    return dict(clusters), []


def _cluster_embedding(session_dir: Path, label: str,
                       windows: list[tuple[float, float]],
                       extractor) -> tuple[np.ndarray, float]:
    """Embedding for one cluster + how many seconds of audio it used."""
    audio_path = _audio_path_for_label(session_dir, label)
    embedding = extractor.extract(audio_path, windows=windows)
    seconds = sum(end - start for start, end in windows)
    return embedding, seconds


def learn_from_session(session_dir: str | Path, db: SpeakerDatabase, *,
                       extractor=None) -> dict[str, dict]:
    """Learn every labelled speaker in a diarized session.

    Returns ``{label: {"name": ..., "speaker_id": ..., "seconds": ...}}``.
    A label whose name already exists in the database gets a new
    sample (the stored mean moves towards the new voice); a new name
    creates a speaker. Raises FileNotFoundError if the session has
    no transcript / audio. A cluster that fails the shared
    clean-cluster guard (see :func:`evaluate_cluster_cleanliness`)
    is skipped and never learned from — a mixed cluster stays out
    of the database; ``seconds`` is the clean audio actually
    learned from, not the cluster's full duration.
    """
    session_dir = Path(session_dir)
    extractor = extractor or get_default_extractor()
    names = load_speaker_names(session_dir)
    clusters, _skipped = _learnable_clusters(session_dir,
                                             _clusters(session_dir))
    learned: dict[str, dict] = {}
    for label, windows in clusters.items():
        assessment = evaluate_cluster_cleanliness(
            session_dir, label, windows, extractor,
            enforce_minimum_clean_audio=False)
        if assessment.mixed:
            logger.warning(
                "Skipping %s in %s: mixed cluster (%s) — not learned",
                label, session_dir.name, assessment.reason)
            continue
        if assessment.clean_embedding is None:
            logger.warning("Skipping %s in %s: %s", label,
                           session_dir.name, assessment.detail)
            continue
        name = names.get(label, label)
        speaker_id = db.add_embedding(
            name, assessment.clean_embedding, backend=extractor.backend,
            source_session=session_dir.name, source_label=label)
        learned[label] = {"name": name, "speaker_id": speaker_id,
                          "seconds": round(assessment.clean_seconds, 2)}
        logger.info("Learned %s as %r from %s (%.1fs of clean audio)",
                    label, name, session_dir.name,
                    assessment.clean_seconds)
    return learned


def suggest_for_session(session_dir: str | Path, db: SpeakerDatabase, *,
                        extractor=None,
                        threshold: float | None = None
                        ) -> list[ClusterSuggestion]:
    """Suggest a known name for each label cluster in a session.

    ``threshold`` defaults to the per-backend threshold for the
    extractor's backend (see matching.threshold_for_backend);
    pass an explicit value to override it. Nothing is written to
    the database. Callers should persist the JSON form with
    :func:`write_suggestions` and let the user confirm / correct
    each suggestion.
    """
    session_dir = Path(session_dir)
    extractor = extractor or get_default_extractor()
    clusters, _skipped = _learnable_clusters(session_dir,
                                             _clusters(session_dir))
    suggestions: list[ClusterSuggestion] = []
    for label, windows in clusters.items():
        try:
            embedding, seconds = _cluster_embedding(
                session_dir, label, windows, extractor)
        except ValueError as exc:
            logger.warning("Skipping %s in %s: %s", label,
                           session_dir.name, exc)
            continue
        match = match_embedding(embedding, db, backend=extractor.backend,
                                threshold=threshold)
        suggestions.append(ClusterSuggestion(
            label=label, match=match, seconds=seconds,
            embedding=embedding, backend=extractor.backend))
    return suggestions


def sample_window_for_label(session_dir: str | Path, label: str
                            ) -> tuple[Path, float, float] | None:
    """A playable voice sample for one label cluster in a session.

    Returns (audio_path, start, end) for the label's *longest*
    transcript segment — a single turn, so it is one voice and the
    most identifiable. The audio file follows the same rule as
    embedding extraction (``mic.wav`` for "You", ``system.wav``
    for "Remote", else ``combined.wav``). Returns None when the
    session has no transcript, the label has no segments, or no
    audio file exists.
    """
    session_dir = Path(session_dir)
    try:
        windows = _clusters(session_dir).get(str(label), [])
    except FileNotFoundError:
        return None
    if not windows:
        return None
    start, end = max(windows, key=lambda w: w[1] - w[0])
    try:
        audio = _audio_path_for_label(session_dir, label)
    except FileNotFoundError:
        return None
    return audio, start, end


def find_sample_for_speaker(output_root: str | Path, db: SpeakerDatabase,
                            name: str
                            ) -> tuple[Path, float, float, str] | None:
    """A playable voice sample for a known speaker, newest first.

    Walks the speaker's stored samples (newest first) and returns
    the first whose source session still exists with audio and a
    transcript: (audio_path, start, end, session_name). Segments
    carry the display name after a rename, so both the stored
    label and the current name are tried. Returns None when no
    sample survives — audio older than the retention window is
    deleted, so an older speaker may genuinely have no sample
    left; callers must say so, not fail silently.
    """
    root = Path(output_root)
    seen: set[str] = set()
    for sample in db.speaker_samples(name):
        session_name = sample.get("source_session")
        if not session_name or session_name in seen:
            continue
        seen.add(session_name)
        session_dir = root / str(session_name)
        if not session_dir.is_dir():
            continue
        labels = {str(sample.get("source_label") or ""), str(name)}
        labels.discard("")
        best: tuple[float, float] | None = None
        best_audio: Path | None = None
        for label in labels:
            found = sample_window_for_label(session_dir, label)
            if found is None:
                continue
            audio, start, end = found
            if best is None or (end - start) > (best[1] - best[0]):
                best, best_audio = (start, end), audio
        if best is not None and best_audio is not None:
            return best_audio, best[0], best[1], session_name
    return None


def write_suggestions(session_dir: str | Path,
                      suggestions: list[ClusterSuggestion]) -> Path:
    """Write ``speaker_suggestions.json`` (no embeddings — names only)."""
    path = Path(session_dir) / SUGGESTIONS_FILE
    payload = {"version": 1,
               "suggestions": [s.to_dict() for s in suggestions]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_suggestions(session_dir: str | Path) -> list[dict]:
    """Load ``speaker_suggestions.json``; [] if none was written."""
    path = Path(session_dir) / SUGGESTIONS_FILE
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("suggestions", []) if isinstance(payload, dict) else []


class MixedClusterError(ValueError):
    """A cluster failed the shared clean-cluster guard.

    Raised by the manual learn paths (:func:`confirm_suggestion`,
    :func:`correct_suggestion`) when the cluster to learn is mixed
    — mostly boundary spill, or two halves that sound like
    different voices. Learning it would poison the speaker's
    profile, so the user is told plainly instead of the sample
    being stored. Automatic identification never raises this; it
    leaves the cluster Unknown.
    """


def _clean_embedding_for_learning(session_dir: Path, label: str,
                                  extractor) -> np.ndarray:
    """Clean embedding for ``label``, or raise if it must not be learned.

    Runs the shared guard (:func:`evaluate_cluster_cleanliness`):
    a mixed cluster raises :class:`MixedClusterError`, a failed
    extraction raises ValueError. The returned embedding covers
    the cluster's clean windows only.
    """
    windows = _clusters(session_dir).get(str(label), [])
    if not windows:
        raise ValueError(
            f"Cluster {label!r} has no segments in "
            f"{session_dir.name} — nothing to learn")
    assessment = evaluate_cluster_cleanliness(
        session_dir, label, windows, extractor,
        enforce_minimum_clean_audio=False)
    if assessment.mixed:
        raise MixedClusterError(
            f"Cluster {label!r} in {session_dir.name} is mixed "
            f"({assessment.reason}) — it was not learned, so the "
            "speaker profile is unchanged")
    if assessment.clean_embedding is None:
        raise ValueError(
            f"Could not extract a clean embedding for {label!r} "
            f"in {session_dir.name}: {assessment.detail}")
    return assessment.clean_embedding


def confirm_suggestion(session_dir: str | Path, db: SpeakerDatabase,
                       suggestion: ClusterSuggestion, *,
                       extractor=None) -> dict:
    """User confirmed a suggestion: learn it and rename the label here.

    Adds the cluster's *clean* embedding to the suggested speaker
    in the database, renames the label to that name in this
    session, and re-exports transcript.txt/.srt. A mixed cluster
    raises :class:`MixedClusterError` and is not learned, even on
    an explicit confirm. Returns the suggestion as a dict.
    """
    session_dir = Path(session_dir)
    name = suggestion.suggested_name
    if name is None:
        raise ValueError("Cannot confirm a suggestion with no suggested name")
    extractor = extractor or get_default_extractor()
    embedding = _clean_embedding_for_learning(
        session_dir, suggestion.label, extractor)
    _apply_identity(session_dir, db, suggestion.label, name,
                    embedding, backend=extractor.backend)
    return suggestion.to_dict()


def correct_suggestion(session_dir: str | Path, db: SpeakerDatabase,
                       suggestion: ClusterSuggestion, correct_name: str, *,
                       extractor=None) -> dict:
    """User corrected a suggestion: learn the right name, rename here.

    The *clean* sample goes to ``correct_name`` only — the
    wrongly-suggested speaker is not reinforced. A mixed cluster
    raises :class:`MixedClusterError` and is not learned, even on
    an explicit correction. Returns the updated suggestion dict.
    """
    session_dir = Path(session_dir)
    correct_name = str(correct_name).strip()
    if not correct_name:
        raise ValueError("Correct name must not be blank")
    extractor = extractor or get_default_extractor()
    embedding = _clean_embedding_for_learning(
        session_dir, suggestion.label, extractor)
    db_correct(db, correct_name, embedding,
               backend=extractor.backend,
               source_session=session_dir.name,
               source_label=suggestion.label)
    rename_in_session(session_dir, suggestion.label, correct_name)
    reapply_names(session_dir)
    out = suggestion.to_dict()
    out["corrected_to"] = correct_name
    return out


# ------------------------------------------------------- auto-identify ---
# Guards for automatic enrolment (Nov 8 2026 — see the module
# docstring and /tmp/meettrace-youtube-test/REPORT.md). A cluster is
# only auto-enrolled or auto-applied from *clean* audio: segments
# that sit inside their own diarization turns, enough of it, and
# two halves of it that sound like the same voice.
AUTO_MIN_CLEAN_SECONDS: float = 5.0
AUTO_MIN_CLEAN_FRACTION: float = 0.5
AUTO_BOUNDARY_COVERAGE: float = 0.8
AUTO_HALF_MIN_SECONDS: float = 2.5
AUTO_CONSISTENCY_MIN_COSINE: float = 0.5
AUTO_APPLY_MIN_MARGIN: float = 0.10
_UNNAMED_PREFIX = "Speaker "


def _diarization_turns(session_dir: Path
                       ) -> list[tuple[float, float, str]] | None:
    """Turns from diarization.json, or None when the mode has none.

    Sherpa/full modes publish turns; simple mode does not. A
    missing or malformed file also yields None (all windows count
    as clean — the boundary rule cannot apply without turns).
    """
    path = session_dir / "diarization.json"
    if not path.exists():
        return None
    try:
        turns = json.loads(path.read_text(encoding="utf-8")).get("turns")
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(turns, list):
        return None
    out: list[tuple[float, float, str]] = []
    for turn in turns:
        try:
            out.append((float(turn["start"]), float(turn["end"]),
                        str(turn["speaker"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out or None


def _clean_windows(session_dir: Path, label: str,
                   windows: list[tuple[float, float]]
                   ) -> tuple[list[tuple[float, float]], float, float]:
    """Windows of ``label`` that are clean (see module docstring).

    Returns (clean_windows, clean_seconds, total_seconds). A window
    is clean when at least AUTO_BOUNDARY_COVERAGE of its duration
    lies inside diarization turns carrying its own label. When the
    session has no turns, every window is clean.
    """
    total = sum(end - start for start, end in windows)
    turns = _diarization_turns(session_dir)
    if turns is None:
        return list(windows), total, total
    clean: list[tuple[float, float]] = []
    clean_seconds = 0.0
    for start, end in windows:
        duration = end - start
        if duration <= 0:
            continue
        own = sum(max(0.0, min(end, t_end) - max(start, t_start))
                  for t_start, t_end, t_label in turns
                  if t_label == label)
        if own / duration >= AUTO_BOUNDARY_COVERAGE:
            clean.append((start, end))
            clean_seconds += duration
    return clean, clean_seconds, total


def _split_halves(windows: list[tuple[float, float]]
                  ) -> tuple[list[tuple[float, float]],
                             list[tuple[float, float]]]:
    """Split windows into two halves by cumulative duration.

    A window straddling the midpoint is cut in two, so each half
    holds exactly half the total duration.
    """
    total = sum(end - start for start, end in windows)
    half = total / 2.0
    first: list[tuple[float, float]] = []
    second: list[tuple[float, float]] = []
    seen = 0.0
    for start, end in windows:
        duration = end - start
        if seen + duration <= half:
            first.append((start, end))
        elif seen >= half:
            second.append((start, end))
        else:
            cut = start + (half - seen)
            first.append((start, cut))
            second.append((cut, end))
        seen += duration
    return first, second


@dataclass
class ClusterCleanliness:
    """Result of the shared clean-cluster guard for one cluster.

    ``mixed`` is True when the cluster must never be learned from
    by any path: too little of it is clean audio, or its two
    halves sound like different voices. ``clean_embedding`` is
    the embedding of the clean windows only (None when the guard
    stopped before extracting it, or extraction failed).
    ``reason`` is one of the auto-identify reason codes
    (``"too_little_clean_audio"``, ``"mixed_cluster"``,
    ``"extraction_failed"``) or None when the cluster passed every
    check that was enforced.
    """

    label: str
    clean_windows: list[tuple[float, float]]
    clean_seconds: float
    total_seconds: float
    half_cosine: float | None = None
    clean_embedding: np.ndarray | None = field(default=None, repr=False)
    mixed: bool = False
    reason: str | None = None
    detail: str = ""
    passes_minimum_clean_audio: bool = True
    passes_clean_fraction: bool = True
    passes_consistency: bool = True


def evaluate_cluster_cleanliness(
        session_dir: Path, label: str,
        windows: list[tuple[float, float]], extractor, *,
        enforce_minimum_clean_audio: bool = True) -> ClusterCleanliness:
    """Run the shared clean-cluster guard on one cluster.

    Every learn path calls this one function, so a mixed cluster
    is judged the same way whether learning is automatic or the
    user clicked Confirm. The checks, in order (no embedding is
    extracted once a check has failed):

    * clean fraction — at least AUTO_MIN_CLEAN_FRACTION of the
      cluster's audio must be clean (inside its own diarization
      turns). A cluster that is mostly boundary spill is mixed;
    * minimum clean audio — at least AUTO_MIN_CLEAN_SECONDS of
      clean audio, enforced only when
      ``enforce_minimum_clean_audio`` is True (automatic
      enrolment). Manual paths pass False: a short but clean
      cluster is the user's explicit decision, and shortness
      alone does not poison a profile;
    * half-split consistency — when both halves hold at least
      AUTO_HALF_MIN_SECONDS, their embeddings must agree
      (cosine >= AUTO_CONSISTENCY_MIN_COSINE); disagreeing halves
      are two voices, so the cluster is mixed.

    On passing every enforced check, ``clean_embedding`` holds
    the embedding extracted from the clean windows only.
    """
    clean, clean_seconds, total_seconds = _clean_windows(
        session_dir, label, windows)
    passes_fraction = (total_seconds <= 0 or clean_seconds / total_seconds
                       >= AUTO_MIN_CLEAN_FRACTION)
    passes_minimum = clean_seconds >= AUTO_MIN_CLEAN_SECONDS
    if not passes_fraction:
        return ClusterCleanliness(
            label=label, clean_windows=clean,
            clean_seconds=clean_seconds, total_seconds=total_seconds,
            mixed=True, reason="too_little_clean_audio",
            detail=(f"Only {clean_seconds:.1f}s of {total_seconds:.1f}s "
                    "is clean audio"),
            passes_minimum_clean_audio=passes_minimum,
            passes_clean_fraction=False)
    if enforce_minimum_clean_audio and not passes_minimum:
        return ClusterCleanliness(
            label=label, clean_windows=clean,
            clean_seconds=clean_seconds, total_seconds=total_seconds,
            reason="too_little_clean_audio",
            detail=(f"Too little clean audio ({clean_seconds:.1f}s of "
                    f"{total_seconds:.1f}s)"),
            passes_minimum_clean_audio=False,
            passes_clean_fraction=True)
    audio_path = _audio_path_for_label(session_dir, label)
    half_cosine: float | None = None
    passes_consistency = True
    first, second = _split_halves(clean)
    first_seconds = sum(end - start for start, end in first)
    second_seconds = sum(end - start for start, end in second)
    if (first_seconds >= AUTO_HALF_MIN_SECONDS
            and second_seconds >= AUTO_HALF_MIN_SECONDS):
        try:
            emb_first = extractor.extract(audio_path, windows=first)
            emb_second = extractor.extract(audio_path, windows=second)
        except ValueError as exc:
            return ClusterCleanliness(
                label=label, clean_windows=clean,
                clean_seconds=clean_seconds, total_seconds=total_seconds,
                reason="extraction_failed", detail=str(exc),
                passes_minimum_clean_audio=passes_minimum,
                passes_clean_fraction=True)
        half_cosine = cosine_similarity(emb_first, emb_second)
        if half_cosine < AUTO_CONSISTENCY_MIN_COSINE:
            return ClusterCleanliness(
                label=label, clean_windows=clean,
                clean_seconds=clean_seconds, total_seconds=total_seconds,
                half_cosine=half_cosine, mixed=True,
                reason="mixed_cluster",
                detail=(f"Cluster halves sound like different voices "
                        f"(cosine {half_cosine:.2f})"),
                passes_minimum_clean_audio=passes_minimum,
                passes_clean_fraction=True, passes_consistency=False)
    try:
        clean_embedding = extractor.extract(audio_path, windows=clean)
    except ValueError as exc:
        return ClusterCleanliness(
            label=label, clean_windows=clean,
            clean_seconds=clean_seconds, total_seconds=total_seconds,
            half_cosine=half_cosine, reason="extraction_failed",
            detail=str(exc),
            passes_minimum_clean_audio=passes_minimum,
            passes_clean_fraction=True,
            passes_consistency=passes_consistency)
    return ClusterCleanliness(
        label=label, clean_windows=clean,
        clean_seconds=clean_seconds, total_seconds=total_seconds,
        half_cosine=half_cosine, clean_embedding=clean_embedding,
        passes_minimum_clean_audio=passes_minimum,
        passes_clean_fraction=True,
        passes_consistency=passes_consistency)


def _next_letters(letters: str) -> str:
    """Base-26 increment over A–Z: A→B … Z→AA, AA→AB …"""
    chars = list(letters)
    i = len(chars) - 1
    while i >= 0:
        if chars[i] != "Z":
            chars[i] = chr(ord(chars[i]) + 1)
            return "".join(chars)
        chars[i] = "A"
        i -= 1
    return "A" * (len(chars) + 1)


def _next_unnamed_name(db: SpeakerDatabase) -> str:
    """Next free "Speaker A/B/…/Z/AA/…" identity name."""
    import re

    used: set[str] = set()
    for speaker in db.list_speakers():
        match = re.fullmatch(r"Speaker ([A-Z]+)", speaker["name"])
        if match:
            used.add(match.group(1))
    letters = "A"
    while letters in used:
        letters = _next_letters(letters)
    return f"{_UNNAMED_PREFIX}{letters}"


def _apply_identity(session_dir: Path, db: SpeakerDatabase, label: str,
                    name: str, embedding: np.ndarray, *,
                    backend: str) -> None:
    """Learn ``embedding`` under ``name`` and rename ``label`` to it here.

    The sample joins the speaker's stored mean, the label is
    renamed in this session's speakers.json, and transcript.txt /
    .srt are re-exported. Shared by confirm and auto-identify.
    """
    db_confirm(db, name, embedding, backend=backend,
               source_session=session_dir.name, source_label=label)
    rename_in_session(session_dir, label, name)
    reapply_names(session_dir)


def auto_identify_session(session_dir: str | Path, db: SpeakerDatabase, *,
                          extractor=None,
                          threshold: float | None = None,
                          suggestions: list[ClusterSuggestion] | None = None
                          ) -> dict:
    """Automatically identify every unnamed cluster in a session.

    The default after a recording stops (Nov 8 2026): no Confirm
    click needed. For each learnable cluster, the decision is made
    from a *clean* embedding — extracted only from the cluster's
    clean windows (see :func:`_clean_windows`) — never from the
    full-cluster embedding, which can carry boundary spill from
    another speaker:

    * clean audio below the minimum (5 s, and at least half the
      cluster) → left Unknown;
    * the two halves of the clean audio disagree (cosine < 0.5)
      → the cluster is mixed → left Unknown;
    * the clean embedding matches a known speaker confidently
      (clears the threshold with margin >= 0.10) → applied:
      sample learned, label renamed to that name. Because the
      database is updated live, a fragment enrolled earlier in
      the *same* session is already a known speaker for the next
      fragment — sherpa's over-segmentation of one voice into
      several clusters collapses back into one identity;
    * otherwise → enrolled as a new unnamed identity
      ("Speaker A", "Speaker B", …): sample stored, label renamed.

    Returns {"applied": [...], "enrolled": [...],
    "left_unknown": [...]} (each entry {"label", "name"/"reason"})
    and rewrites ``speaker_suggestions.json`` with an
    "auto_action" / "auto_detail" per cluster. ``suggestions`` may
    be passed in (e.g. from the worker that just computed them) to
    avoid extracting the full-cluster embeddings twice. Nothing
    here raises for a single bad cluster — a failure is recorded
    as left_unknown and the rest of the session is still processed.
    """
    session_dir = Path(session_dir)
    extractor = extractor or get_default_extractor()
    backend = extractor.backend
    match_threshold = (threshold if threshold is not None
                       else threshold_for_backend(backend))
    if suggestions is None:
        suggestions = suggest_for_session(
            session_dir, db, extractor=extractor, threshold=threshold)
    clusters = _clusters(session_dir)
    actions: dict[str, list[dict]] = {
        "applied": [], "enrolled": [], "left_unknown": []}
    entries: list[dict] = []
    for suggestion in suggestions:
        label = suggestion.label
        entry = suggestion.to_dict()
        windows = clusters.get(label, [])
        # The shared guard: clean windows, minimum clean audio and
        # half-split consistency, judged exactly as every other
        # learn path judges them. A failed guard leaves the cluster
        # Unknown; no embedding is extracted past the failure.
        assessment = evaluate_cluster_cleanliness(
            session_dir, label, windows, extractor,
            enforce_minimum_clean_audio=True)
        clean_seconds = assessment.clean_seconds
        total_seconds = assessment.total_seconds
        if assessment.reason == "too_little_clean_audio":
            entry["auto_action"] = "left_unknown"
            entry["auto_detail"] = (
                f"Too little clean audio ({clean_seconds:.1f}s of "
                f"{total_seconds:.1f}s) — left for the user to confirm")
            actions["left_unknown"].append(
                {"label": label, "reason": "too_little_clean_audio"})
            entries.append(entry)
            continue
        if assessment.reason == "extraction_failed":
            entry["auto_action"] = "left_unknown"
            entry["auto_detail"] = f"Embedding failed: {assessment.detail}"
            actions["left_unknown"].append(
                {"label": label, "reason": "extraction_failed"})
            entries.append(entry)
            continue
        if assessment.mixed:
            entry["auto_action"] = "left_unknown"
            entry["auto_detail"] = (
                "Cluster halves sound like different voices "
                f"(cosine {assessment.half_cosine:.2f}) — mixed cluster, "
                "left for the user")
            actions["left_unknown"].append(
                {"label": label, "reason": "mixed_cluster"})
            entries.append(entry)
            continue
        clean_embedding = assessment.clean_embedding
        if clean_embedding is None:  # defensive: guard passed but no embedding
            entry["auto_action"] = "left_unknown"
            entry["auto_detail"] = "Embedding failed: no clean embedding"
            actions["left_unknown"].append(
                {"label": label, "reason": "extraction_failed"})
            entries.append(entry)
            continue
        clean_match = match_embedding(clean_embedding, db, backend=backend,
                                      threshold=match_threshold)
        if clean_match.suggested_name is not None:
            if clean_match.margin >= AUTO_APPLY_MIN_MARGIN:
                _apply_identity(session_dir, db, label,
                                clean_match.suggested_name, clean_embedding,
                                backend=backend)
                entry["auto_action"] = "applied"
                entry["auto_detail"] = (
                    f"Matched {clean_match.suggested_name} on clean audio "
                    f"(confidence {clean_match.confidence:.2f}, margin "
                    f"{clean_match.margin:.2f})")
                actions["applied"].append(
                    {"label": label, "name": clean_match.suggested_name})
                entries.append(entry)
                continue
            # Cleared the threshold but nearly tied with a rival —
            # a guess here would fork a known voice, so leave it.
            entry["auto_action"] = "left_unknown"
            entry["auto_detail"] = (
                f"Ambiguous match: {clean_match.best_name} "
                f"({clean_match.best_score:.2f}) nearly tied with "
                f"{clean_match.second_best_name} "
                f"({clean_match.second_best_score:.2f}, margin "
                f"{clean_match.margin:.2f}) — left for the user")
            actions["left_unknown"].append(
                {"label": label, "reason": "ambiguous_match"})
            entries.append(entry)
            continue
        name = _next_unnamed_name(db)
        speaker_id = db.add_embedding(
            name, clean_embedding, backend=backend,
            source_session=session_dir.name, source_label=label)
        rename_in_session(session_dir, label, name)
        reapply_names(session_dir)
        entry["auto_action"] = "enrolled"
        entry["auto_detail"] = (
            f"New unnamed identity {name} "
            f"({clean_seconds:.1f}s of clean audio)")
        actions["enrolled"].append(
            {"label": label, "name": name, "speaker_id": speaker_id})
        entries.append(entry)
    path = Path(session_dir) / SUGGESTIONS_FILE
    path.write_text(json.dumps({"version": 1, "suggestions": entries},
                               indent=2), encoding="utf-8")
    logger.info("Auto-identified %s: applied %d, enrolled %d, "
                "left unknown %d", session_dir.name,
                len(actions["applied"]), len(actions["enrolled"]),
                len(actions["left_unknown"]))
    return actions


def rename_identity(db: SpeakerDatabase, output_root: str | Path,
                    old_name: str, new_name: str,
                    dictionary_path: str | Path | None = None) -> dict:
    """Rename a speaker identity everywhere it is used.

    Renames the database speaker and every past session that used
    the old name — either through a speakers.json mapping or
    because the old name is itself a stored segment label — and
    re-exports those sessions' transcript.txt/.srt via
    :func:`reapply_names`. The new name is also added to the names
    dictionary (``dictionary_path``, default ``~/.offair/
    names_dictionary.json``) so future transcriptions are biased
    toward hearing it — a dictionary failure is logged, never
    raised, so it cannot break a rename. Returns {"name": new_name,
    "sessions_updated": [...], "sessions_skipped": [...]} (skipped
    = speakers.json was updated but the exports could not be
    refreshed, e.g. no speaker labels yet). Raises ValueError when
    ``old_name`` is unknown or ``new_name`` is blank/taken (from
    :meth:`SpeakerDatabase.rename_speaker`).
    """
    old_name, new_name = str(old_name).strip(), str(new_name).strip()
    db.rename_speaker(old_name, new_name)
    try:
        add_name(new_name, dictionary_path)
    except Exception:  # noqa: BLE001 — a rename must not fail on this
        logger.warning("Could not add %r to the names dictionary",
                       new_name, exc_info=True)
    root = Path(output_root)
    updated: list[str] = []
    skipped: list[str] = []
    if root.is_dir():
        for session_dir in sorted(root.iterdir()):
            if not session_dir.is_dir():
                continue
            try:
                names = load_speaker_names(session_dir)
            except ValueError:
                continue
            try:
                labels = {str(getattr(s, "speaker", ""))
                          for s in load_transcript_json(
                              session_dir / TRANSCRIPT_JSON)}
            except (FileNotFoundError, ValueError, KeyError,
                    json.JSONDecodeError, OSError):
                labels = set()
            touched = False
            if old_name in names.values():
                names = {label: (new_name if name == old_name else name)
                         for label, name in names.items()}
                touched = True
            elif old_name in labels:
                names[old_name] = new_name
                touched = True
            if not touched:
                continue
            save_speaker_names(session_dir, names)
            try:
                reapply_names(session_dir)
            except (FileNotFoundError, ValueError):
                skipped.append(session_dir.name)
                continue
            updated.append(session_dir.name)
    return {"name": new_name, "sessions_updated": updated,
            "sessions_skipped": skipped}
