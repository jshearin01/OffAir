"""Recording source: what a session is a recording *of*.

Every session carries ``source`` (a small enum) + ``source_detail``
(a free string) in ``metadata.json`` — the contract the Recording
Browser filter, the calendar view, and the MCP server
(``list_sessions`` / ``search_meetings`` filtering) all read:

* ``source`` — one of :data:`SOURCES`; never a guess. When the
  source cannot be determined it is ``"other"``.
* ``source_detail`` — the app display name (``"Google Chrome"``),
  a window title, or a user note. It is *evidence*, not a second
  guess: for an unknown exe it holds the prettified exe name so
  the user can correct the source later.

Detection (at record start, from the capture target process) and
backfill (for sessions recorded before this feature, from their
stored ``app_name``) both live here, so the mapping is defined
once. The enum values are stable API — do not rename them.
"""
from __future__ import annotations

from app.audio.app_sessions import display_name_for_exe

#: The ``source`` enum, in the order the UI lists it.
SOURCE_TEAMS = "teams"
SOURCE_ZOOM = "zoom"
SOURCE_WEBEX = "webex"
SOURCE_MEET = "meet"
SOURCE_BROWSER = "browser"
SOURCE_SOLO = "solo"
SOURCE_CLAUDE_CODE = "claude-code"
SOURCE_OTHER = "other"
SOURCES: tuple[str, ...] = (
    SOURCE_TEAMS,
    SOURCE_ZOOM,
    SOURCE_WEBEX,
    SOURCE_MEET,
    SOURCE_BROWSER,
    SOURCE_SOLO,
    SOURCE_CLAUDE_CODE,
    SOURCE_OTHER,
)

#: Human labels for the enum (browser rows, filter dropdown, editor).
SOURCE_LABELS: dict[str, str] = {
    SOURCE_TEAMS: "Teams",
    SOURCE_ZOOM: "Zoom",
    SOURCE_WEBEX: "Webex",
    SOURCE_MEET: "Google Meet",
    SOURCE_BROWSER: "Browser",
    SOURCE_SOLO: "Solo (mic only)",
    SOURCE_CLAUDE_CODE: "Claude Code",
    SOURCE_OTHER: "Other",
}

#: Process exe name (lower-case) -> source. Google Meet never
#: appears here: it runs inside a browser, so a Meet call is
#: ``browser`` by process and ``meet`` only via the manual
#: override (or a window-title check, which OffAir does not do —
#: see docs/RECORDING_SOURCE.md). Webex names are best-effort:
#: ``webex.exe`` is the Webex app; ``atmgr.exe`` is the Webex
#: Meetings desktop-app manager (its meeting process). Zoom's
#: Windows process is ``zoom.exe`` (``zoom.us`` on macOS, where
#: OffAir has no capture backend yet — mapped anyway, harmless).
_PROCESS_SOURCE: dict[str, str] = {
    # Teams (Windows exes from app.config.TEAMS_PROCESS_NAMES,
    # plus the Linux client names from app.platform_info).
    "ms-teams.exe": SOURCE_TEAMS,
    "teams.exe": SOURCE_TEAMS,
    "ms-teamsupdate.exe": SOURCE_TEAMS,
    "teams": SOURCE_TEAMS,
    "ms-teams": SOURCE_TEAMS,
    # Zoom.
    "zoom.exe": SOURCE_ZOOM,
    "zoom": SOURCE_ZOOM,
    "zoom.us": SOURCE_ZOOM,
    # Webex (best-effort — see the note above).
    "webex.exe": SOURCE_WEBEX,
    "atmgr.exe": SOURCE_WEBEX,
    # Browsers.
    "chrome.exe": SOURCE_BROWSER,
    "msedge.exe": SOURCE_BROWSER,
    "firefox.exe": SOURCE_BROWSER,
    "brave.exe": SOURCE_BROWSER,
    "opera.exe": SOURCE_BROWSER,
    "chromium.exe": SOURCE_BROWSER,
    "chrome": SOURCE_BROWSER,
    "chromium": SOURCE_BROWSER,
    "firefox": SOURCE_BROWSER,
    "msedge": SOURCE_BROWSER,
    "brave": SOURCE_BROWSER,
}

#: Stored ``app_name`` (lower-case) -> source, for backfilling
#: sessions recorded before ``source`` existed. Only clean maps
#: are listed — anything else is ``other``, never guessed.
_APP_NAME_SOURCE: dict[str, str] = {
    "microsoft teams": SOURCE_TEAMS,
    "teams": SOURCE_TEAMS,
    "zoom": SOURCE_ZOOM,
    "webex": SOURCE_WEBEX,
    "cisco webex": SOURCE_WEBEX,
    "webex meetings": SOURCE_WEBEX,
    "google meet": SOURCE_MEET,
    "google chrome": SOURCE_BROWSER,
    "chrome": SOURCE_BROWSER,
    "microsoft edge": SOURCE_BROWSER,
    "edge": SOURCE_BROWSER,
    "mozilla firefox": SOURCE_BROWSER,
    "firefox": SOURCE_BROWSER,
    "brave": SOURCE_BROWSER,
    "opera": SOURCE_BROWSER,
    "chromium": SOURCE_BROWSER,
}


def normalize_source(value: str | None) -> str | None:
    """``value`` when it is a valid source, else None."""
    if isinstance(value, str) and value in SOURCES:
        return value
    return None


def source_from_process(exe_name: str | None) -> str | None:
    """Source for a capture-target exe name, or None if unmappable.

    ``None`` (no target app picked) maps to ``solo`` only in
    :func:`detect_source` — this function returns None for it so
    callers can tell "no target" apart from "unknown app".
    """
    if not exe_name:
        return None
    return _PROCESS_SOURCE.get(exe_name.strip().lower())


def source_from_app_name(app_name: str | None) -> str | None:
    """Source for a stored ``app_name``, or None if unmappable."""
    if not app_name:
        return None
    return _APP_NAME_SOURCE.get(app_name.strip().lower())


def detect_source(exe_name: str | None) -> tuple[str, str]:
    """``(source, source_detail)`` for a capture-target exe.

    * A known exe maps to its source; the detail is its friendly
      display name (``"Google Chrome"``).
    * ``None`` — no target app picked (mic-only / system-wide
      with no subject) — is ``solo`` with an empty detail.
    * An unknown exe is ``other``, never a guess; the detail
      keeps the prettified exe name so the source can be
      corrected afterwards.
    """
    if not exe_name:
        return SOURCE_SOLO, ""
    mapped = source_from_process(exe_name)
    detail = display_name_for_exe(exe_name)
    if mapped is not None:
        return mapped, detail
    return SOURCE_OTHER, detail


def resolve_recording_source(*, override: str | None = None,
                             exe_name: str | None = None
                             ) -> tuple[str, str]:
    """``(source, source_detail)`` for a recording about to start.

    An explicit ``override`` (the Record card's Source dropdown)
    always wins over detection — this is the only way to record a
    source no process identifies, e.g. a Claude Code session
    (mic + system audio while working with the agent). With no
    override, detection from the target exe applies
    (:func:`detect_source`). Raises ValueError for an override
    outside :data:`SOURCES`.
    """
    if override is not None:
        normalized = normalize_source(override)
        if normalized is None:
            raise ValueError(
                f"unknown recording source {override!r}; expected "
                f"one of {list(SOURCES)}")
        detail = (display_name_for_exe(exe_name)
                  if exe_name else "")
        return normalized, detail
    return detect_source(exe_name)


def backfill_source(metadata: dict) -> dict | None:
    """Fields to write for a session whose ``source`` is missing.

    Returns ``{"source": ..., "source_detail": ...}`` when the
    session has no valid source (derived from its stored
    ``app_name`` where that maps cleanly, else ``other`` — never
    guessed), or None when the stored source is already valid
    and is left alone. Idempotent: applying the result once makes
    a second call return None. ``source_detail`` is the stored
    ``app_name`` when present, so the evidence survives even
    when the source itself is ``other``.
    """
    if normalize_source(metadata.get("source")) is not None:
        return None
    app_name = metadata.get("app_name") or ""
    mapped = source_from_app_name(app_name)
    return {
        "source": mapped if mapped is not None else SOURCE_OTHER,
        "source_detail": app_name,
    }


__all__ = [
    "SOURCES",
    "SOURCE_BROWSER",
    "SOURCE_CLAUDE_CODE",
    "SOURCE_LABELS",
    "SOURCE_MEET",
    "SOURCE_OTHER",
    "SOURCE_SOLO",
    "SOURCE_TEAMS",
    "SOURCE_WEBEX",
    "SOURCE_ZOOM",
    "backfill_source",
    "detect_source",
    "normalize_source",
    "resolve_recording_source",
    "source_from_app_name",
    "source_from_process",
]
