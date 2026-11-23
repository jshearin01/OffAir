"""Outlook calendar linking — READ-ONLY. (Feature 3, Nov 10 2026.)

Links a recorded session to the Outlook calendar event it belongs
to, by matching the recording window against the default Calendar
folder. The link is stored in the session's ``metadata.json`` under
``calendar_event`` (subject, start, end, attendees, organiser,
entry_id — or ``null`` when no event matched).

READ-ONLY GUARANTEE — this module never writes anything into
Outlook. It only *reads* calendar items (``GetNamespace`` /
``GetDefaultFolder`` / ``Items`` / ``Restrict`` / property gets)
and *displays* an item on request (``GetItemFromID`` + ``Display``,
which opens the event in Outlook for the user to look at). There is
no code path in this module — not even a helper "for later" — that
edits an appointment's body, adds an attachment, creates an item,
or persists an appointment object. ``tests/test_outlook_linking.py``
enforces this two ways: a grep-level test asserting no persist /
body-assignment code exists in this package (or in any file this
feature touches), and mock-COM tests asserting the link path only
ever calls read operations. The user forbade Outlook writes
because attendees with access to the meeting would see them.

Platform: Windows only, via pywin32 (``win32com.client``). On any
other platform — or when pywin32 / Outlook is missing —
:func:`outlook_available` returns ``(False, reason)`` and every
other function degrades to "no link" without raising, so recording
and transcription are never blocked by Outlook being absent.

Timezones — read this before touching the matching code. Outlook
returns appointment times through COM as **naive datetimes in the
machine's local time**. OffAir session metadata stores
``start_time`` / ``end_time`` as **UTC ISO strings** (see
``app/storage/sessions.py``). Matching therefore converts the
session window to local time first
(``astimezone()`` → drop the tzinfo) and compares naive-local
against naive-local; event times stored in ``metadata.json`` are
written back as local ISO strings *with* their UTC offset, so the
UI can display them without re-deriving the zone. The folder-name
stamp (``YYYY-MM-DD_HH-MM-SS``) is local time too, consistent with
the COM values. The timezone test in ``tests/test_outlook_linking.py``
covers a UTC-stored session time converted to local before matching.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("offair.calendar.outlook")

#: Outlook MAPI folder id for the default Calendar folder.
OL_FOLDER_CALENDAR = 9

#: An event counts as the session's meeting when its start is this
#: close to the recording start (either direction).
START_WINDOW = timedelta(minutes=15)

#: Or when the recording overlaps the event by more than this
#: fraction of the shorter of the two durations.
OVERLAP_FRACTION = 0.5

#: How far back the calendar query reaches before the recording
#: start, so a long event that started earlier (and overlaps the
#: recording) is still seen by the Restrict query.
QUERY_LOOKBACK = timedelta(hours=24)

#: COM date format for Restrict strings ("2026-11-10 14:30").
#: Outlook interprets these in local time.
_RESTRICT_FMT = "%Y-%m-%d %H:%M"

#: Methods the link path is allowed to call on COM objects. The
#: mock-COM tests assert no call outside this set ever happens —
#: anything that could mutate an item is absent by construction.
_ALLOWED_COM_CALLS = frozenset({
    "Dispatch", "GetNamespace", "GetDefaultFolder", "Sort",
    "Restrict", "GetItemFromID", "Display",
})

#: Properties the link path is allowed to read on COM objects.
#: (``Recipients`` iteration reads ``Name`` per recipient.)
_ALLOWED_COM_PROPERTIES = frozenset({
    "Items", "IncludeRecurrences", "Count",
    "Subject", "Start", "End", "Organizer", "EntryID",
    "Recipients", "Name",
})


# ------------------------------------------------------------ availability --
def outlook_available() -> tuple[bool, str | None]:
    """``(True, None)`` when Outlook linking can run here.

    Needs Windows + pywin32 + a registered ``Outlook.Application``
    ProgID (i.e. Outlook installed). The ProgID check is a registry
    read — it does not launch Outlook. Never raises.
    """
    if sys.platform != "win32":
        return False, (
            "Outlook linking needs Windows with Outlook installed; "
            f"this machine is {sys.platform}. Sessions still record "
            "and transcribe normally — they just have no calendar link.")
    try:
        import win32com.client  # noqa: F401  (import check only)
    except Exception as exc:  # noqa: BLE001 — any import failure = absent
        return False, (
            "Outlook linking needs pywin32 "
            "(pip install -r requirements.txt); "
            f"import failed: {exc}")
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT,
                            "Outlook.Application"):
            pass
    except Exception:  # noqa: BLE001 — no ProgID = Outlook not installed
        return False, (
            "Outlook is not installed on this machine "
            "(no Outlook.Application registration found).")
    return True, None


def _get_outlook_namespace(client: Any = None) -> Any:
    """The MAPI namespace (``client`` injectable for tests).

    ``client`` is a ``win32com.client`` module (real or fake). When
    omitted, ``win32com.client.Dispatch("Outlook.Application")``
    is called — which starts Outlook if it is not running.
    Raises on failure; callers convert that to "no link".
    """
    if client is None:
        import win32com.client
        client = win32com.client
    app = client.Dispatch("Outlook.Application")
    return app.GetNamespace("MAPI")


# ------------------------------------------------------------ matching ------
def _to_local_naive(when: datetime) -> datetime:
    """``when`` as a naive datetime in local time.

    Aware values (UTC-stored session times) convert via
    ``astimezone()``; naive values are already local (old metadata,
    COM values) and pass through unchanged.
    """
    if when.tzinfo is not None:
        when = when.astimezone()  # local zone, per the machine
    return when.replace(tzinfo=None)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _matches(rec_start: datetime, rec_end: datetime,
             ev_start: datetime, ev_end: datetime) -> bool:
    """Do the recording window and the event match?

    Either rule suffices: the event starts within
    :data:`START_WINDOW` of the recording start, or the recording
    overlaps the event by more than :data:`OVERLAP_FRACTION` of the
    shorter of the two durations. All four values are naive local
    datetimes (see the module docstring).
    """
    if abs(ev_start - rec_start) <= START_WINDOW:
        return True
    overlap_start = max(rec_start, ev_start)
    overlap_end = min(rec_end, ev_end)
    if overlap_end <= overlap_start:
        return False
    overlap = overlap_end - overlap_start
    shorter = min(rec_end - rec_start, ev_end - ev_start)
    return shorter > timedelta(0) and overlap > OVERLAP_FRACTION * shorter


def _event_to_dict(item: Any) -> dict | None:
    """Read an appointment item into a plain dict. None on bad data.

    Reads only: Subject / Start / End / Organizer / EntryID /
    Recipients' names. A missing EntryID means the item cannot be
    re-opened later, so it is not a usable link.
    """
    try:
        subject = str(item.Subject or "")
        start = item.Start
        end = item.End
        organiser = str(item.Organizer or "")
        entry_id = str(item.EntryID or "")
        attendees: list[str] = []
        try:
            for recipient in item.Recipients:
                name = getattr(recipient, "Name", None)
                if name:
                    attendees.append(str(name))
        except Exception:  # noqa: BLE001 — attendees are best-effort
            logger.debug("Could not read event recipients", exc_info=True)
    except Exception:  # noqa: BLE001 — a broken item is not a link
        logger.debug("Could not read calendar event", exc_info=True)
        return None
    if not entry_id or not isinstance(start, datetime) \
            or not isinstance(end, datetime):
        return None
    # Store local times with their offset, so the UI displays them
    # as the user saw them in Outlook.
    start_local = _to_local_naive(start).astimezone()
    end_local = _to_local_naive(end).astimezone()
    return {
        "subject": subject,
        "start": start_local.isoformat(timespec="minutes"),
        "end": end_local.isoformat(timespec="minutes"),
        "attendees": attendees,
        "organiser": organiser,
        "entry_id": entry_id,
    }


def find_event_for_session(start_dt: datetime, end_dt: datetime, *,
                           client: Any = None) -> dict | None:
    """The Outlook event for a recording window, or None.

    ``start_dt`` / ``end_dt`` are the recording start/end — aware
    (UTC, as stored in metadata.json) or naive local; both are
    converted to naive local before comparing with Outlook's
    local-time values. ``client`` injects a fake ``win32com.client``
    for tests; omit it on a real machine. Never raises: any COM
    failure (Outlook absent, not running, MAPI error) logs and
    returns None — the session simply has no calendar link.
    """
    rec_start = _to_local_naive(start_dt)
    rec_end = _to_local_naive(end_dt)
    if rec_end <= rec_start:
        return None
    try:
        namespace = _get_outlook_namespace(client)
        folder = namespace.GetDefaultFolder(OL_FOLDER_CALENDAR)
        items = folder.Items
        items.IncludeRecurrences = True
        items.Sort("[Start]")
        window_start = rec_start - QUERY_LOOKBACK
        window_end = rec_end + START_WINDOW
        restrict = (
            "[Start] <= '" + window_end.strftime(_RESTRICT_FMT) + "'"
            " AND [End] >= '" + window_start.strftime(_RESTRICT_FMT) + "'")
        candidates = items.Restrict(restrict)
    except Exception:  # noqa: BLE001 — Outlook absent/broken = no link
        logger.info("Outlook calendar query failed; no link for session",
                    exc_info=True)
        return None
    best: dict | None = None
    best_key: tuple | None = None
    try:
        iterable = list(candidates)
    except Exception:  # noqa: BLE001 — Restrict result unusable = no link
        logger.info("Could not enumerate Outlook events", exc_info=True)
        return None
    for item in iterable:
        event = _event_to_dict(item)
        if event is None:
            continue
        ev_start = _to_local_naive(_parse_iso(event["start"])
                                   or item.Start)
        ev_end = _to_local_naive(_parse_iso(event["end"]) or item.End)
        if not _matches(rec_start, rec_end, ev_start, ev_end):
            continue
        # Closest start wins; ties break towards the longest overlap.
        overlap = (min(rec_end, ev_end) - max(rec_start, ev_start))
        key = (abs(ev_start - rec_start), -overlap)
        if best_key is None or key < best_key:
            best, best_key = event, key
    return best


def open_event_in_outlook(entry_id: str, *, client: Any = None) -> bool:
    """Open a linked event in Outlook (read-only view), or False.

    ``GetItemFromID`` + ``Display`` — the item opens for the user
    to look at; nothing about it is changed. Never raises.
    """
    if not entry_id:
        return False
    try:
        namespace = _get_outlook_namespace(client)
        item = namespace.GetItemFromID(entry_id)
        item.Display()
    except Exception:  # noqa: BLE001 — Outlook absent = cannot open
        logger.info("Could not open event %s in Outlook", entry_id,
                    exc_info=True)
        return False
    return True


# ------------------------------------------------------------ session wiring --
def link_session(session_dir: str | Path, *, enabled: bool = True,
                 client: Any = None) -> dict | None:
    """Match a session to its Outlook event and store the result.

    Writes the ``calendar_event`` key into the session's
    ``metadata.json`` (the event dict, or ``null`` when no event
    matched). Nothing is written when linking is disabled or
    Outlook is unavailable — a previously stored link is left
    untouched in that case. Never raises: any failure leaves the
    metadata as it was and returns None (or the stored value).
    Returns the stored value (event dict, None, or the previously
    stored link when no attempt was made).
    """
    from app.storage.sessions import load_metadata, update_metadata

    session_dir = Path(session_dir)
    try:
        # Pure read: linking must never mutate the session except to
        # store the calendar_event below, so the source backfill
        # (which writes metadata.json on load) stays off here.
        metadata = load_metadata(session_dir, backfill=False)
    except Exception:  # noqa: BLE001 — no metadata = nothing to link
        logger.debug("No metadata.json in %s; skipping Outlook link",
                     session_dir)
        return None
    stored = metadata.get("calendar_event")
    if not enabled:
        return stored
    available, _reason = outlook_available()
    if not available:
        return stored
    start = _parse_iso(metadata.get("start_time"))
    if start is None:
        return stored
    end = _parse_iso(metadata.get("end_time"))
    if end is None:
        duration = metadata.get("duration_seconds")
        if isinstance(duration, (int, float)) and duration > 0:
            end = start + timedelta(seconds=float(duration))
        else:
            end = datetime.now(timezone.utc)
    try:
        event = find_event_for_session(start, end, client=client)
    except Exception:  # noqa: BLE001 — find_event_for_session is
        # guarded already; this is belt-and-braces so linking can
        # never break a session save.
        logger.exception("Outlook linking failed for %s", session_dir)
        return stored
    try:
        update_metadata(session_dir, calendar_event=event)
    except Exception:  # noqa: BLE001 — metadata write failure is not fatal
        logger.warning("Could not store calendar link for %s",
                       session_dir, exc_info=True)
        return stored
    return event


def format_event_label(event: dict | None) -> str:
    """The main-window line for a linked event.

    ``"Linked to: <subject>, <time>"``; the time is the event start
    in local time (stored with its offset, so it displays as seen
    in Outlook). Empty string when there is no event.
    """
    if not event:
        return ""
    subject = event.get("subject") or "(untitled event)"
    start = _parse_iso(event.get("start"))
    when = (start.astimezone().strftime("%b %d, %Y, %H:%M")
            if start is not None else "")
    return f"Linked to: {subject}, {when}" if when else f"Linked to: {subject}"
