"""Bookmarks: marked moments in a session, stored in metadata.json.

On-disk format (verbatim — the Session view and MCP read the same
key)::

    metadata.json:
        "bookmarks": [
            {"t": 42.5, "label": "Budget decision",
             "created": "2026-11-21T20:00:00+00:00"},
            ...
        ]

``t`` is seconds (float) from the start of the session,
``label`` is free text (empty when the user did not name it),
``created`` is an ISO-8601 UTC timestamp. Entries are written via
:func:`app.storage.sessions.update_metadata`, so every other
metadata key is untouched.

Whether a bookmark's audio still exists is NOT stored: it is
computed when listing (see :func:`has_playable_audio`), because
retention may delete the WAVs after the bookmark was made. A
bookmark whose audio is gone is still listed — with
``audio_available`` False — so the UI can say plainly that the
audio was deleted instead of pretending the moment is playable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.storage.retention import AUDIO_FILES
from app.storage.sessions import load_metadata, update_metadata

BOOKMARKS_KEY = "bookmarks"


@dataclass
class Bookmark:
    """One bookmark, with its audio availability at list time."""

    t: float
    label: str
    created: str
    audio_available: bool


def has_playable_audio(session_dir: str | Path) -> bool:
    """Whether any recording audio still exists for the session."""
    session_dir = Path(session_dir)
    return any((session_dir / name).exists() for name in AUDIO_FILES)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stored_entries(session_dir: str | Path) -> list[dict]:
    """Raw bookmark entries from metadata.json, in stored order.

    Malformed entries (not a dict, no numeric ``t``) are skipped
    rather than breaking the list — a hand-edited metadata file
    must not hide the good bookmarks.
    """
    metadata = load_metadata(session_dir)
    raw = metadata.get(BOOKMARKS_KEY, [])
    if not isinstance(raw, list):
        return []
    entries: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            t = float(item["t"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(t) or t < 0:
            continue
        entries.append({"t": t,
                        "label": str(item.get("label") or ""),
                        "created": str(item.get("created") or "")})
    return entries


def add_bookmark(session_dir: str | Path, t: float,
                 label: str = "") -> dict:
    """Add a bookmark at ``t`` seconds and return its stored entry.

    Raises ValueError for a negative, non-finite, or non-numeric
    time. Raises FileNotFoundError when the session has no
    metadata.json.
    """
    try:
        t_value = float(t)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bookmark time must be a number, got {t!r}"
                         ) from exc
    if not math.isfinite(t_value) or t_value < 0:
        raise ValueError(
            f"bookmark time must be >= 0 and finite, got {t_value!r}")
    entry = {"t": t_value, "label": str(label or "").strip(),
             "created": _now_iso()}
    entries = _stored_entries(session_dir)
    entries.append(entry)
    update_metadata(session_dir, **{BOOKMARKS_KEY: entries})
    return dict(entry)


def list_bookmarks(session_dir: str | Path) -> list[Bookmark]:
    """Bookmarks for a session, earliest first.

    Each entry reports ``audio_available`` as of now — False when
    retention has deleted the session's WAVs. Raises
    FileNotFoundError when the session has no metadata.json.
    """
    audio = has_playable_audio(session_dir)
    entries = sorted(_stored_entries(session_dir),
                     key=lambda e: e["t"])
    return [Bookmark(t=e["t"], label=e["label"], created=e["created"],
                     audio_available=audio) for e in entries]


__all__ = [
    "BOOKMARKS_KEY",
    "Bookmark",
    "add_bookmark",
    "has_playable_audio",
    "list_bookmarks",
]
