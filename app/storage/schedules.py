"""Scheduled recordings: one-off timers that start a recording.

A schedule is a fire time, a target app, and optional calendar
details. The app's scheduler (a timer in the main window) fires
due schedules while the app process is running; the MCP server —
a separate stdio process — creates and cancels them. Both sides
share this one file, so the store is the contract:

    ~/.offair/schedules.json:
        {"schedules": [
            {"id": "9f2c...",
             "fire_at": "2026-11-23T09:00:00-05:00",
             "target_app": "ms-teams.exe",
             "subject": "Weekly sync",
             "attendees": ["Priya"],
             "organiser": "John",
             "created_by": "mcp",
             "status": "pending",
             "created": "2026-11-22T15:00:00+00:00",
             "fired_at": null,
             "session_id": null,
             "note": null},
            ...
        ]}

``fire_at`` is ISO-8601 with an offset (local wall time, as the
user or the agent gave it). ``status`` is ``pending``,
``fired``, ``skipped``, ``expired`` or ``cancelled`` — one-off
only, no recurrence. ``created_by`` is ``"mcp"`` or ``"gui"``.

Concurrency: the app and the MCP server are separate processes
that both read and write this file. Writes are atomic (temp file
in the same directory + ``os.replace``), so a reader never sees a
torn file, and every write merges against a fresh re-read (by
id) so one side's mark-fired cannot clobber the other side's
create that landed in between. There is no cross-process lock:
per-record last-writer-wins is the documented v1 semantic, and
the file is tiny (one record per schedule), so the window is a
single atomic replace.

Corruption: a missing file reads as empty; an unreadable or
wrong-shaped file logs and reads as empty — the app and the MCP
server must never crash on a bad schedules file. Malformed
*entries* are skipped individually (logged) so one bad record
never hides the valid schedules. Nothing here ever silently
discards a valid schedule.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("offair.storage.schedules")

SCHEDULES_FILE_NAME = "schedules.json"
SCHEDULES_DIR_NAME = ".offair"

#: Schedule statuses. Terminal: fired, skipped, expired,
#: cancelled. Only "pending" schedules can fire or be cancelled.
SCHEDULE_STATUSES: tuple[str, ...] = (
    "pending", "fired", "skipped", "expired", "cancelled")
TERMINAL_STATUSES: tuple[str, ...] = (
    "fired", "skipped", "expired", "cancelled")

#: How late a due schedule may still fire while the app is
#: running (covers sleep/wake and timer stalls). Older than this,
#: a due schedule expires instead of firing late. Startup expiry
#: (a time that passed while the app was closed) has no grace:
#: it always expires. See _due_action / expire_past_due.
FIRE_GRACE_SECONDS: float = 300.0


def default_schedules_path() -> Path:
    """Where schedules live: ``~/.offair/schedules.json``."""
    return Path.home() / SCHEDULES_DIR_NAME / SCHEDULES_FILE_NAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(value) -> datetime | None:
    """An ISO-8601 datetime, or None when unparsable.

    Naive values are read as local wall time (the user typed
    "09:00" meaning their 09:00) and localised with the machine's
    offset, so every comparison below is aware-vs-aware.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        when = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.astimezone()
    return when


@dataclass
class Schedule:
    """One one-off scheduled recording."""

    id: str
    fire_at: datetime
    target_app: str
    subject: str | None = None
    attendees: list[str] = field(default_factory=list)
    organiser: str | None = None
    created_by: str = "gui"
    status: str = "pending"
    created: datetime | None = None
    fired_at: datetime | None = None
    session_id: str | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        """The stored record (ISO strings with offsets)."""
        return {
            "id": self.id,
            "fire_at": self.fire_at.isoformat(timespec="seconds"),
            "target_app": self.target_app,
            "subject": self.subject,
            "attendees": list(self.attendees),
            "organiser": self.organiser,
            "created_by": self.created_by,
            "status": self.status,
            "created": (self.created.isoformat(timespec="seconds")
                        if self.created is not None else None),
            "fired_at": (self.fired_at.isoformat(timespec="seconds")
                         if self.fired_at is not None else None),
            "session_id": self.session_id,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Schedule | None":
        """A stored record as a Schedule, or None when malformed.

        Malformed means: not a dict, no usable id, no parseable
        fire time, a blank target app, or an unknown status.
        Anything else degrades (bad attendees become [], a bad
        created_by becomes "gui") rather than dropping the record
        — a schedule the user can see is better than one lost to
        a strict parse.
        """
        if not isinstance(data, dict):
            return None
        schedule_id = data.get("id")
        if not isinstance(schedule_id, str) or not schedule_id.strip():
            return None
        fire_at = _parse_iso(data.get("fire_at"))
        if fire_at is None:
            return None
        target_app = data.get("target_app")
        if not isinstance(target_app, str) or not target_app.strip():
            return None
        status = data.get("status", "pending")
        if status not in SCHEDULE_STATUSES:
            return None
        created_by = data.get("created_by")
        if created_by not in ("mcp", "gui"):
            created_by = "gui"
        attendees = data.get("attendees")
        if not isinstance(attendees, list):
            attendees = []
        attendees = [str(a).strip() for a in attendees
                     if str(a).strip()]
        subject = data.get("subject")
        subject = str(subject).strip() if isinstance(subject, str) \
            and subject.strip() else None
        organiser = data.get("organiser")
        organiser = str(organiser).strip() \
            if isinstance(organiser, str) and organiser.strip() else None
        return cls(
            id=schedule_id.strip(),
            fire_at=fire_at,
            target_app=target_app.strip(),
            subject=subject,
            attendees=attendees,
            organiser=organiser,
            created_by=created_by,
            status=status,
            created=_parse_iso(data.get("created")),
            fired_at=_parse_iso(data.get("fired_at")),
            session_id=(str(data["session_id"]).strip()
                        if isinstance(data.get("session_id"), str)
                        and str(data["session_id"]).strip() else None),
            note=(str(data["note"]).strip()
                  if isinstance(data.get("note"), str)
                  and str(data["note"]).strip() else None),
        )


def _read_records(path: Path) -> tuple[list[dict], list[dict]]:
    """(valid, malformed) raw records from the store file.

    A missing file is empty, not an error. An unreadable file, bad
    JSON, or a wrong-shaped payload logs and reads as empty — the
    caller must never crash on this file. Malformed entries are
    returned separately so the caller can log them once.
    """
    if not path.exists():
        return [], []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning("Schedules file %s unreadable (%s); "
                       "treating as empty", path, exc)
        return [], []
    if not isinstance(payload, dict):
        logger.warning("Schedules file %s has wrong shape "
                       "(expected an object); treating as empty", path)
        return [], []
    raw = payload.get("schedules", [])
    if not isinstance(raw, list):
        logger.warning("Schedules file %s has wrong shape "
                       "(expected a list under 'schedules'); "
                       "treating as empty", path)
        return [], []
    valid: list[dict] = []
    malformed: list[dict] = []
    for item in raw:
        schedule = Schedule.from_dict(item)
        if schedule is None:
            malformed.append(item)
        else:
            valid.append(schedule.to_dict())
    return valid, malformed


def load_schedules(path: str | Path | None = None) -> list[Schedule]:
    """All schedules in the store, in stored order.

    Never raises for a missing, unreadable, or malformed file
    (see :func:`_read_records`).
    """
    target = Path(path) if path is not None else default_schedules_path()
    valid, malformed = _read_records(target)
    if malformed:
        logger.warning("Skipping %d malformed schedule record(s) "
                       "in %s", len(malformed), target)
    return [Schedule.from_dict(item) for item in valid  # type: ignore[misc]
            if Schedule.from_dict(item) is not None]


def _write_atomic(path: Path, records: list[dict]) -> None:
    """Write the store atomically (temp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"schedules": records}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


def _save_merged(path: Path, updated: list[Schedule]) -> None:
    """Save ``updated`` records, merged against a fresh re-read.

    Records in ``updated`` replace same-id records from the
    current file; records the caller never saw are preserved.
    This is what makes an app-side mark-fired safe against an
    MCP-side create that landed in between.
    """
    current, _ = _read_records(path)
    by_id = {item["id"]: item for item in current}
    for schedule in updated:
        by_id[schedule.id] = schedule.to_dict()
    # Stored order: creation order (the file's order) with new
    # records appended — a re-read must not reorder the list.
    ordered = list(current)
    seen = {item["id"] for item in ordered}
    for schedule in updated:
        if schedule.id not in seen:
            ordered.append(schedule.to_dict())
            seen.add(schedule.id)
    for index, item in enumerate(ordered):
        if item["id"] in by_id:
            ordered[index] = by_id[item["id"]]
    _write_atomic(path, ordered)


def _save_all(path: Path, schedules: list[Schedule]) -> None:
    """Replace the whole store (used only by tests/helpers)."""
    _write_atomic(path, [s.to_dict() for s in schedules])


def create_schedule(fire_at: datetime, target_app: str, *,
                    subject: str | None = None,
                    attendees: list[str] | None = None,
                    organiser: str | None = None,
                    created_by: str = "gui",
                    path: str | Path | None = None) -> Schedule:
    """Create a pending schedule; raises ValueError on bad input.

    ``fire_at`` must be an aware datetime in the future (naive
    values are read as local wall time). ``target_app`` must be a
    non-empty exe name or display name — the caller resolves
    display names (see the MCP tool); the store keeps the value
    as given.
    """
    if not isinstance(fire_at, datetime):
        raise ValueError(
            f"fire_at must be a datetime, got {fire_at!r}")
    if fire_at.tzinfo is None:
        fire_at = fire_at.astimezone()
    now = datetime.now(timezone.utc)
    if fire_at <= now:
        raise ValueError(
            f"fire_at {fire_at.isoformat(timespec='seconds')} is not "
            f"in the future")
    if not isinstance(target_app, str) or not target_app.strip():
        raise ValueError("target_app must not be blank")
    if created_by not in ("mcp", "gui"):
        raise ValueError(
            f"created_by must be 'mcp' or 'gui', got {created_by!r}")
    schedule = Schedule(
        id=uuid.uuid4().hex,
        fire_at=fire_at,
        target_app=target_app.strip(),
        subject=str(subject).strip()
        if isinstance(subject, str) and subject.strip() else None,
        attendees=[str(a).strip() for a in (attendees or [])
                   if str(a).strip()],
        organiser=str(organiser).strip()
        if isinstance(organiser, str) and organiser.strip() else None,
        created_by=created_by,
        status="pending",
        created=datetime.now(timezone.utc),
    )
    target = Path(path) if path is not None else default_schedules_path()
    _save_merged(target, [schedule])
    logger.info("Schedule created: %s at %s for %s (%s)",
                schedule.id, schedule.fire_at.isoformat(),
                schedule.target_app, created_by)
    return schedule


def cancel_schedule(schedule_id: str, *,
                    path: str | Path | None = None) -> Schedule:
    """Cancel a pending schedule; raises ValueError otherwise.

    Unknown ids and non-pending schedules raise ValueError — a
    cancel that silently did nothing would leave the caller
    believing a recording will not start.
    """
    target = Path(path) if path is not None else default_schedules_path()
    for schedule in load_schedules(target):
        if schedule.id == schedule_id:
            if schedule.status != "pending":
                raise ValueError(
                    f"Schedule {schedule_id!r} is {schedule.status}, "
                    f"not pending — it cannot be cancelled")
            schedule.status = "cancelled"
            _save_merged(target, [schedule])
            logger.info("Schedule cancelled: %s", schedule_id)
            return schedule
    raise ValueError(f"No schedule {schedule_id!r}")


def mark_fired(schedule: Schedule, *,
               session_id: str | None = None,
               path: str | Path | None = None) -> Schedule:
    """Mark a schedule fired (optionally with its session id)."""
    target = Path(path) if path is not None else default_schedules_path()
    schedule.status = "fired"
    schedule.fired_at = datetime.now(timezone.utc)
    if session_id:
        schedule.session_id = session_id
    _save_merged(target, [schedule])
    return schedule


def mark_skipped(schedule: Schedule, note: str, *,
                 path: str | Path | None = None) -> Schedule:
    """Mark a schedule skipped (already recording / start failed)."""
    target = Path(path) if path is not None else default_schedules_path()
    schedule.status = "skipped"
    schedule.fired_at = datetime.now(timezone.utc)
    schedule.note = note
    _save_merged(target, [schedule])
    logger.info("Schedule skipped: %s (%s)", schedule.id, note)
    return schedule


def expire_past_due(now: datetime | None = None, *,
                    path: str | Path | None = None) -> list[Schedule]:
    """Expire pending schedules whose fire time has passed.

    Called at app startup: a schedule whose time passed while the
    app was closed never fires late. Returns the schedules it
    expired (empty when none). No grace here — see
    :func:`due_schedules` for the running-app grace.
    """
    target = Path(path) if path is not None else default_schedules_path()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.astimezone()
    expired = []
    for schedule in load_schedules(target):
        if schedule.status == "pending" and schedule.fire_at <= now:
            schedule.status = "expired"
            schedule.note = ("Fire time passed while the app was "
                             "not running")
            expired.append(schedule)
    if expired:
        _save_merged(target, expired)
        logger.info("Expired %d past-due schedule(s)", len(expired))
    return expired


def visible_schedules(schedules: list[Schedule],
                      now: datetime | None = None) -> list[Schedule]:
    """Display order: pending earliest-first, then recent history.

    History is terminal records (fired / skipped / expired /
    cancelled) from the last 7 days, newest first. Older terminal
    records stay in the store file but are not listed — the file
    is the archive, the list is the working view. Shared by the
    GUI list and the MCP ``list_schedules`` tool so both show the
    same schedules in the same order.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.astimezone()
    cutoff = now - timedelta(days=7)
    pending = sorted(
        (s for s in schedules if s.status == "pending"),
        key=lambda s: s.fire_at)
    history = sorted(
        (s for s in schedules
         if s.status != "pending"
         and (s.fired_at is None or s.fired_at >= cutoff)),
        key=lambda s: s.fired_at or s.fire_at, reverse=True)
    return pending + history


def due_schedules(now: datetime | None = None, *,
                  path: str | Path | None = None) -> list[Schedule]:
    """Pending schedules due to fire now, earliest first.

    Due means ``fire_at <= now <= fire_at + FIRE_GRACE_SECONDS``:
    the grace covers sleep/wake and timer stalls. Schedules older
    than the grace are NOT returned here — the caller expires them
    via :func:`expire_stale_due` instead of firing hours late.
    """
    target = Path(path) if path is not None else default_schedules_path()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.astimezone()
    due = [s for s in load_schedules(target)
           if s.status == "pending" and s.fire_at <= now
           and now - s.fire_at <= timedelta(seconds=FIRE_GRACE_SECONDS)]
    return sorted(due, key=lambda s: s.fire_at)


def expire_stale_due(now: datetime | None = None, *,
                     path: str | Path | None = None) -> list[Schedule]:
    """Expire due-but-stale schedules (past the firing grace).

    For the running app: a schedule whose fire time passed more
    than FIRE_GRACE_SECONDS ago is expired, not fired late.
    Returns the schedules it expired.
    """
    target = Path(path) if path is not None else default_schedules_path()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.astimezone()
    stale = []
    for schedule in load_schedules(target):
        if (schedule.status == "pending" and schedule.fire_at <= now
                and now - schedule.fire_at
                > timedelta(seconds=FIRE_GRACE_SECONDS)):
            schedule.status = "expired"
            schedule.note = ("Fire time passed too long ago to "
                             "start late")
            stale.append(schedule)
    if stale:
        _save_merged(target, stale)
    return stale


__all__ = [
    "FIRE_GRACE_SECONDS",
    "SCHEDULES_DIR_NAME",
    "SCHEDULES_FILE_NAME",
    "SCHEDULE_STATUSES",
    "TERMINAL_STATUSES",
    "Schedule",
    "cancel_schedule",
    "create_schedule",
    "default_schedules_path",
    "due_schedules",
    "expire_past_due",
    "expire_stale_due",
    "load_schedules",
    "mark_fired",
    "mark_skipped",
    "visible_schedules",
]
