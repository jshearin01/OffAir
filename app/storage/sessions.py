"""Recording sessions: one folder per recording, with metadata.json.

Layout (PRD Phase 1)::

    <output_root>/YYYY-MM-DD_HH-MM-SS_<app-slug>/
        mic.wav
        system.wav
        combined.wav
        metadata.json

``metadata.json`` is written at session creation and updated on stop, so
a crash still leaves a folder that says what it is and when it started.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from app.storage.source import (
    SOURCE_OTHER,
    SOURCE_SOLO,
    backfill_source,
    normalize_source,
)

logger = logging.getLogger("offair.storage")

MIC_FILE = "mic.wav"
SYSTEM_FILE = "system.wav"
COMBINED_FILE = "combined.wav"
METADATA_FILE = "metadata.json"
#: Progressive live transcript (Nov 19 2026): written after every
#: live chunk during recording, deleted after a successful Stop
#: finalize. It is the crash-recovery record only — the retention
#: "has transcript" check must keep requiring transcript.json.
TRANSCRIPT_LIVE_FILE = "transcript-live.json"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "session"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_metadata(session_dir: Path, metadata: dict) -> Path:
    path = session_dir / METADATA_FILE
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def create_session(output_root: str | Path, app_name: str = "Teams", *,
                   mode: str = "system-loopback-fallback",
                   sample_rate: int = 48000, channels: int = 1,
                   devices: dict | None = None,
                   warning: str | None = None,
                   source: str | None = None,
                   source_detail: str | None = None) -> Path:
    """Create a new session folder and its initial metadata.json.

    Folder name is ``YYYY-MM-DD_HH-MM-SS_<slug>`` (local time); a numeric
    suffix is added if that name already exists. Returns the folder path.

    ``source`` / ``source_detail`` are the recording source (see
    :mod:`app.storage.source`): the caller passes the resolved value
    from :func:`app.storage.source.resolve_recording_source` —
    detection from the capture target, or the user's manual
    override. When omitted the source is ``"other"``: never a
    guess. Raises ValueError for a source outside the enum.
    """
    if source is not None and normalize_source(source) is None:
        raise ValueError(
            f"unknown recording source {source!r}")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{stamp}_{slugify(app_name)}"
    session_dir = root / base
    counter = 1
    while session_dir.exists():
        counter += 1
        session_dir = root / f"{base}-{counter:02d}"
    session_dir.mkdir(parents=True)
    metadata = {
        "session_id": session_dir.name,
        "app_name": app_name,
        "mode": mode,
        "status": "recording",
        "start_time": _now_iso(),
        "end_time": None,
        "duration_seconds": None,
        "sample_rate": sample_rate,
        "channels": channels,
        "devices": devices or {},
        "warning": warning,
        # Recording source (app.storage.source): what this session
        # is a recording of. "other" when unknown — never guessed.
        "source": source if source is not None else SOURCE_OTHER,
        "source_detail": source_detail or "",
        "pause_policy": "discard",  # audio received while paused is dropped
        "files": {
            "mic": MIC_FILE,
            "system": SYSTEM_FILE,
            "combined": COMBINED_FILE,
        },
    }
    _write_metadata(session_dir, metadata)
    logger.info("Session created: %s", session_dir)
    return session_dir


def create_dictation_session(output_root: str | Path, *,
                           app_name: str = "Dictation",
                           mode: str = "system-loopback-fallback",
                           sample_rate: int = 48000, channels: int = 1,
                           devices: dict | None = None,
                           warning: str | None = None,
                           source_detail: str | None = None) -> Path:
    """Create a dictation session: mic-only, ``source="solo"``.

    Dictation reuses the ordinary session / transcribe / cleanup
    path. The only things that mark it are its metadata:
    ``"source": "solo"`` and ``"dictation": true``. The primary
    readable artifact is the deterministic cleaned note
    (``transcript-clean.txt``, from :func:`app.transcription.
    cleanup.clean_session`). See docs/RECORDING_SOURCE.md for the
    full session shape and the spoken-side-only wording.
    """
    session_dir = create_session(
        output_root, app_name=app_name, mode=mode,
        sample_rate=sample_rate, channels=channels, devices=devices,
        warning=warning, source=SOURCE_SOLO,
        source_detail=source_detail)
    update_metadata(session_dir, dictation=True)
    return session_dir


def is_dictation_session(session_dir: str | Path) -> bool:
    """Whether a session is a dictation session (``dictation: true``).

    A ``"solo"`` source alone does not make a session dictation —
    only the flag does. Returns False when the session has no
    metadata.json.
    """
    try:
        metadata = load_metadata(session_dir, backfill=False)
    except FileNotFoundError:
        return False
    return metadata.get("dictation") is True


def load_metadata(session_dir: str | Path, *,
                  backfill: bool = True) -> dict:
    """Load metadata.json for a session. Raises FileNotFoundError if absent.

    With ``backfill=True`` (the default) a session recorded before
    the ``source`` field existed gets one derived from its stored
    ``app_name`` where that maps cleanly, else ``"other"`` (see
    :func:`app.storage.source.backfill_source`), and the derived
    value is written back into metadata.json. The write happens
    only when the source is missing/invalid, so it is idempotent
    — a second load finds the source present and writes nothing.
    A session that already has a valid source is left alone.
    Pass ``backfill=False`` for a pure read with no side effects.
    """
    path = Path(session_dir) / METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"No metadata.json in {session_dir}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if backfill:
        fields = backfill_source(metadata)
        if fields is not None:
            metadata.update(fields)
            _write_metadata(Path(session_dir), metadata)
    return metadata


def set_session_source(session_dir: str | Path, source: str,
                       source_detail: str | None = None) -> dict:
    """Change a session's recording source (the edit path).

    Writes ``source`` (+ ``source_detail`` when given) into the
    session's metadata.json and returns the updated metadata.
    Raises ValueError for a source outside the enum — the stored
    value is never an invented one.
    """
    if normalize_source(source) is None:
        raise ValueError(
            f"unknown recording source {source!r}")
    fields: dict = {"source": source}
    if source_detail is not None:
        fields["source_detail"] = source_detail
    return update_metadata(session_dir, **fields)


def update_metadata(session_dir: str | Path, **fields) -> dict:
    """Merge ``fields`` into the session's metadata.json and return it."""
    session_dir = Path(session_dir)
    metadata = load_metadata(session_dir)
    metadata.update(fields)
    _write_metadata(session_dir, metadata)
    return metadata


def finalize_session(session_dir: str | Path, *,
                     duration_seconds: float | None = None,
                     status: str = "completed") -> dict:
    """Mark a session finished: end time, duration, and which files exist."""
    session_dir = Path(session_dir)
    files_present = {
        name: (session_dir / filename).exists()
        for name, filename in
        (("mic", MIC_FILE), ("system", SYSTEM_FILE), ("combined", COMBINED_FILE))
    }
    return update_metadata(
        session_dir,
        status=status,
        end_time=_now_iso(),
        duration_seconds=(round(float(duration_seconds), 2)
                          if duration_seconds is not None else None),
        files_present=files_present,
    )


def list_sessions(output_root: str | Path) -> list[Path]:
    """Session folders under ``output_root`` that contain metadata.json.

    Newest first (folder names sort chronologically, so reverse sort works).
    """
    root = Path(output_root)
    if not root.exists():
        return []
    sessions = [p for p in root.iterdir()
                if p.is_dir() and (p / METADATA_FILE).exists()]
    return sorted(sessions, reverse=True)


# ------------------------------------------------------- display naming ------

def is_calendar_linked(metadata: dict) -> bool:
    """Whether ``metadata`` ties the session to an Outlook item.

    Linked means a ``calendar_event`` dict was stored by Outlook
    linking (see :mod:`app.calendar.outlook`). A stored ``null``
    (match attempted, nothing found) or a missing key means the
    session is NOT tied to an Outlook item and may be renamed.
    """
    event = metadata.get("calendar_event")
    return isinstance(event, dict) and bool(event)


def session_display_name(metadata: dict) -> str:
    """The one display-name rule for a session.

    An Outlook-linked session IS its calendar event's subject —
    always, with no rename. Otherwise the user-set custom name
    (``custom_name`` in metadata.json), or the derived default:
    ``"<app> recording"`` from the recorded app name. Every view
    (Library list, Session view, search) resolves the name through
    this function so the views can never disagree.
    """
    if is_calendar_linked(metadata):
        event = metadata["calendar_event"]
        subject = str(event.get("subject") or "").strip()
        return subject if subject else "(untitled event)"
    custom = str(metadata.get("custom_name") or "").strip()
    if custom:
        return custom
    app_name = str(metadata.get("app_name") or "").strip()
    return f"{app_name} recording" if app_name else "Recording"


def rename_session(session_dir: str | Path, name: str) -> dict:
    """Set a session's custom display name.

    Writes ``custom_name`` into metadata.json and returns the
    updated metadata. Raises ValueError for a blank name and for
    an Outlook-linked session — a linked session's name is its
    calendar event's subject and cannot be renamed. Renaming
    never touches the folder name, the calendar link, or any
    other session data.
    """
    session_dir = Path(session_dir)
    metadata = load_metadata(session_dir, backfill=False)
    if is_calendar_linked(metadata):
        raise ValueError(
            "This session is named by its Outlook calendar event "
            "and cannot be renamed.")
    cleaned = str(name or "").strip()
    if not cleaned:
        raise ValueError("A session name cannot be blank.")
    return update_metadata(session_dir, custom_name=cleaned)


def delete_session(session_dir: str | Path, *,
                   live_dir: str | Path | None = None) -> Path:
    """Permanently delete one session and all of its artifacts.

    Removes the session folder (audio, transcripts, diarization,
    speaker renames/suggestions, bookmarks, metadata) plus the
    session's rows in the shared search index (``search_index.db``)
    and the voice-embedding samples it contributed to
    ``speakers.db``. Speaker identities themselves are never
    deleted — they are cross-session — and other sessions are
    untouched. Raises FileNotFoundError when ``session_dir`` is
    not a session folder, and RuntimeError when it is the live
    recording directory (``live_dir``) — a live session is never
    deleted. The shared-store rows go first so a failed folder
    removal can simply be retried.
    """
    from app.speakers.database import SPEAKERS_DB_FILE, SpeakerDatabase
    from app.storage.semantic_search import (
        SEMANTIC_INDEX_FILE,
        SemanticSearchIndex,
    )

    session_dir = Path(session_dir)
    if not (session_dir / METADATA_FILE).exists():
        raise FileNotFoundError(
            f"No session folder (no metadata.json): {session_dir}")
    if live_dir is not None:
        try:
            if Path(live_dir).resolve() == session_dir.resolve():
                raise RuntimeError(
                    "Cannot delete the session being recorded — "
                    "stop the recording first.")
        except OSError:
            pass
    session_id = session_dir.name
    output_root = session_dir.parent
    index_path = output_root / SEMANTIC_INDEX_FILE
    if index_path.exists():
        removed = SemanticSearchIndex(output_root).remove_session(
            session_id)
        logger.info("Deleted %d search-index row(s) for %s",
                    removed, session_id)
    speakers_path = output_root / SPEAKERS_DB_FILE
    if speakers_path.exists():
        with SpeakerDatabase(speakers_path) as db:
            removed = db.delete_embeddings_for_session(session_id)
        logger.info("Deleted %d speaker sample(s) for %s",
                    removed, session_id)
    import shutil
    shutil.rmtree(session_dir)
    logger.info("Session deleted: %s", session_dir)
    return session_dir
