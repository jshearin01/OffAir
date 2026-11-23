"""People queries: one speaker's sessions, utterances, and sample.

Read-only queries over the recordings root, for the person pages.
A speaker is matched by display name — the name after
``speakers.json`` renames are applied to the stored transcript
labels — so a rename in one session and a display name already
stored in another session both count, and a rename propagates
here without any extra bookkeeping. Speakers need not be in
``speakers.db`` to have sessions and utterances here; the
database is only consulted for the voice sample (and even then
a session scan is the fallback, so a speaker whose embeddings
were never stored still gets a sample while their audio
survives).

Sample availability reuses the existing sample-finding rules
from :mod:`app.speakers.session`: the speaker's longest turn,
``mic.wav`` for "You", ``system.wav`` for "Remote", and
``combined.wav`` otherwise. When the database has samples, the
newest surviving sample session wins; otherwise sessions are
scanned newest-first. No audio is stored here — a sample is a
pointer back into a session folder, so retention deleting the
WAVs makes the sample unavailable, honestly, rather than stale.

No commitments data of any kind lives here: person pages show
meetings, utterances, the voice sample, and rename only.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.diarization.names import load_speaker_names
from app.speakers.session import (
    find_sample_for_speaker,
    sample_window_for_label,
)
from app.storage.sessions import list_sessions, load_metadata
from app.transcription.engine import TRANSCRIPT_JSON
from app.transcription.exports import load_transcript_json

logger = logging.getLogger("offair.speakers.people")


@dataclass
class SpeakerSession:
    """One session the speaker spoke in."""

    session_id: str
    path: Path
    start_time: str | None = None
    utterance_count: int = 0


@dataclass
class Utterance:
    """One segment spoken by the speaker (display name applied)."""

    session_id: str
    path: Path
    start: float
    end: float
    text: str
    speaker: str


@dataclass
class SpeakerSample:
    """A playable voice sample: the speaker's longest turn."""

    audio_path: Path
    start: float
    end: float
    session_id: str


def _check_name(name: str) -> str:
    name = str(name).strip()
    if not name:
        raise ValueError("Speaker name must not be blank")
    return name


def _matches(display: str, name: str) -> bool:
    return str(display).strip().lower() == name.lower()


def _session_segments(session_dir: Path):
    """(segments, names) for a session, or None when unreadable.

    A corrupt speakers.json is treated as no renames, and a
    missing/unreadable transcript means the session contributes
    nothing — one bad session never breaks the whole query.
    """
    try:
        names = load_speaker_names(session_dir)
    except ValueError:
        names = {}
    try:
        segments = load_transcript_json(session_dir / TRANSCRIPT_JSON)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError,
            OSError):
        return None
    return segments, names


def _display_name(label: str | None, names: dict[str, str]) -> str | None:
    if not label:
        return None
    return names.get(str(label), str(label))


def speaker_sessions(output_root: str | Path, name: str
                     ) -> list[SpeakerSession]:
    """Sessions where ``name`` spoke, newest first.

    A session counts when at least one of its transcript segments
    carries ``name`` as its display name (after renames).
    """
    name = _check_name(name)
    sessions: list[SpeakerSession] = []
    for session_dir in list_sessions(output_root):
        found = _session_segments(session_dir)
        if found is None:
            continue
        segments, names = found
        count = sum(
            1 for seg in segments
            if _matches(_display_name(getattr(seg, "speaker", None),
                                      names) or "", name))
        if not count:
            continue
        try:
            metadata = load_metadata(session_dir, backfill=False)
            start_time = metadata.get("start_time")
        except (FileNotFoundError, ValueError, json.JSONDecodeError,
                OSError):
            start_time = None
        sessions.append(SpeakerSession(
            session_id=session_dir.name, path=session_dir,
            start_time=start_time, utterance_count=count))
    return sessions


def speaker_utterances(output_root: str | Path, name: str
                       ) -> list[Utterance]:
    """Everything ``name`` said, across sessions.

    Sessions newest first; within a session, chronological.
    Each utterance carries its session, its timestamps, and its
    display name.
    """
    name = _check_name(name)
    utterances: list[Utterance] = []
    for session_dir in list_sessions(output_root):
        found = _session_segments(session_dir)
        if found is None:
            continue
        segments, names = found
        for seg in segments:
            display = _display_name(getattr(seg, "speaker", None), names)
            if display is None or not _matches(display, name):
                continue
            utterances.append(Utterance(
                session_id=session_dir.name, path=session_dir,
                start=float(seg.start), end=float(seg.end),
                text=seg.text, speaker=display))
    return utterances


def _session_sample(session_dir: Path, name: str) -> SpeakerSample | None:
    """Longest matching turn in one session, or None."""
    found = _session_segments(session_dir)
    if found is None:
        return None
    segments, names = found
    labels = {str(getattr(seg, "speaker", ""))
              for seg in segments
              if getattr(seg, "speaker", None)
              and _matches(_display_name(getattr(seg, "speaker", None),
                                         names) or "", name)}
    best: SpeakerSample | None = None
    for label in labels:
        found_window = sample_window_for_label(session_dir, label)
        if found_window is None:
            continue
        audio, start, end = found_window
        if best is None or (end - start) > (best.end - best.start):
            best = SpeakerSample(audio_path=audio, start=start, end=end,
                                 session_id=session_dir.name)
    return best


def speaker_sample(output_root: str | Path, db, name: str
                   ) -> SpeakerSample | None:
    """A playable voice sample for ``name``, or None.

    ``db`` is a :class:`app.speakers.database.SpeakerDatabase` (or
    None to skip the database lookup). The database path is tried
    first — newest sample session that still has audio — then a
    newest-first scan of the sessions themselves, so a speaker
    with no stored embedding still gets a sample while audio
    survives. None means no playable sample exists (audio deleted
    by retention, or the speaker never spoke in a surviving
    session) — callers must say so, not fail silently.
    """
    name = _check_name(name)
    root = Path(output_root)
    if db is not None:
        try:
            found = find_sample_for_speaker(root, db, name)
        except Exception:  # noqa: BLE001 — fall back to the scan
            logger.warning("Database sample lookup failed for %r",
                           name, exc_info=True)
            found = None
        if found is not None:
            audio, start, end, session_id = found
            return SpeakerSample(audio_path=audio, start=start, end=end,
                                 session_id=session_id)
    for session_dir in list_sessions(root):
        sample = _session_sample(session_dir, name)
        if sample is not None:
            return sample
    return None


def has_speaker_sample(output_root: str | Path, db, name: str) -> bool:
    """Whether a playable voice sample exists for ``name``."""
    return speaker_sample(output_root, db, name) is not None


def get_speaker_detail(output_root: str | Path, db, name: str) -> dict:
    """Everything a person page shows, as plain JSON-friendly data.

    ``{"name", "known_in_database", "sessions": [...],
    "utterances": [...], "sample": {...} | None,
    "sample_available": bool, "session_count": int,
    "utterance_count": int}``. Paths are strings. No commitments
    data: meetings, utterances, sample, and rename only.
    """
    name = _check_name(name)
    sessions = speaker_sessions(output_root, name)
    utterances = speaker_utterances(output_root, name)
    sample = speaker_sample(output_root, db, name)
    known = False
    if db is not None:
        try:
            known = db.get_speaker(name) is not None
        except Exception:  # noqa: BLE001 — a db read failure is not fatal
            logger.warning("Could not check database for %r",
                           name, exc_info=True)
    return {
        "name": name,
        "known_in_database": known,
        "sessions": [{
            "session_id": s.session_id,
            "path": str(s.path),
            "start_time": s.start_time,
            "utterance_count": s.utterance_count,
        } for s in sessions],
        "utterances": [{
            "session_id": u.session_id,
            "start": u.start,
            "end": u.end,
            "text": u.text,
            "speaker": u.speaker,
        } for u in utterances],
        "sample": (None if sample is None else {
            "audio_path": str(sample.audio_path),
            "start": sample.start,
            "end": sample.end,
            "session_id": sample.session_id,
        }),
        "sample_available": sample is not None,
        "session_count": len(sessions),
        "utterance_count": len(utterances),
    }


__all__ = [
    "SpeakerSample",
    "SpeakerSession",
    "Utterance",
    "get_speaker_detail",
    "has_speaker_sample",
    "speaker_sample",
    "speaker_sessions",
    "speaker_utterances",
]
