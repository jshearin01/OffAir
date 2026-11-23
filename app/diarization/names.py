"""Speaker Name Panel — rename generic labels, per recording.

PRD Phase 3: the user can rename a label (``SPEAKER_00``, ``You``,
``Remote``) to a real name; names are saved per recording in
``speakers.json`` next to ``transcript.json`` and applied to exports.
Nothing here learns across sessions — that is Phase 4. A rename only
ever affects the one session folder it was saved in.

``speakers.json`` format::

    {"version": 1, "names": {"SPEAKER_00": "Priya", "You": "John"}}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("offair.diarization.names")

SPEAKERS_FILE = "speakers.json"


def _speakers_path(session_dir: str | Path) -> Path:
    return Path(session_dir) / SPEAKERS_FILE


def load_speaker_names(session_dir: str | Path) -> dict[str, str]:
    """Load {label: name} for a session; {} if no speakers.json exists."""
    path = _speakers_path(session_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    names = payload.get("names", {}) if isinstance(payload, dict) else {}
    if not isinstance(names, dict):
        raise ValueError(f"Invalid speakers file {path}: 'names' is not an object")
    return {str(k): str(v) for k, v in names.items() if str(v).strip()}


def save_speaker_names(session_dir: str | Path,
                       names: dict[str, str]) -> Path:
    """Persist {label: name} for a session. Blank names are dropped."""
    clean = {str(k).strip(): str(v).strip()
             for k, v in names.items()
             if str(k).strip() and str(v).strip()}
    path = _speakers_path(session_dir)
    path.write_text(json.dumps({"version": 1, "names": clean}, indent=2),
                    encoding="utf-8")
    logger.info("Saved %d speaker name(s) to %s", len(clean), path)
    return path


def rename_speaker(session_dir: str | Path, label: str, name: str) -> dict[str, str]:
    """Rename one label in a session. An empty name removes the rename.

    Returns the full updated {label: name} mapping. Raises ValueError
    for a blank label.
    """
    label = str(label).strip()
    if not label:
        raise ValueError("Speaker label must not be blank")
    names = load_speaker_names(session_dir)
    name = str(name).strip()
    if name:
        names[label] = name
    else:
        names.pop(label, None)
    save_speaker_names(session_dir, names)
    return names


def display_name(label: str, names: dict[str, str] | None) -> str:
    """The name to show for a label: the rename if one exists, else the label."""
    return (names or {}).get(label, label)


def apply_names(segments: list, names: dict[str, str] | None) -> list:
    """Return a copy of ``segments`` with each segment's ``speaker``
    replaced by its display name (in place on copies, inputs untouched).

    Segments are ``Segment`` objects (``.speaker``) or dicts with a
    ``"speaker"`` key; both are supported so the function works on
    transcript payloads as well as live objects.
    """
    names = names or {}
    out = []
    for seg in segments:
        if isinstance(seg, dict):
            copy = dict(seg)
            if copy.get("speaker"):
                copy["speaker"] = display_name(str(copy["speaker"]), names)
            out.append(copy)
        else:
            import copy as _copy
            copy = _copy.copy(seg)
            speaker = getattr(copy, "speaker", None)
            if speaker:
                copy.speaker = display_name(str(speaker), names)
            out.append(copy)
    return out
