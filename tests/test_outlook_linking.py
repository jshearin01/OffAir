"""Outlook calendar linking — mock-COM tests + read-only enforcement.

This Linux VM has no Outlook, so every COM interaction here runs
against the fake ``win32com.client`` below (injected into
``sys.modules``). The fakes record every method call and every
property read/write, so the tests prove two things the real machine
cannot yet: the match rules, and that the link path only ever
*reads* Outlook — it never mutates an appointment item. The open
gate is a real end-to-end against actual Outlook on the user's
work PC; see docs/OUTLOOK_LINKING.md.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import app.calendar.outlook as outlook
from app.calendar.outlook import (
    _ALLOWED_COM_CALLS,
    _ALLOWED_COM_PROPERTIES,
    find_event_for_session,
    format_event_label,
    link_session,
    open_event_in_outlook,
    outlook_available,
)

REPO = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------ fake COM ------
# The fakes mirror the COM surface outlook.py touches, and record
# everything: method calls, property reads, and — critically — any
# property *write* on an appointment item (a write would be a bug:
# this feature is read-only by design).


class FakeRecipient:
    """One Outlook Recipient — Name is read-only here."""

    _WATCHED = {"Name"}

    def __init__(self, log, name):
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "Name", name)

    def __getattribute__(self, name):
        if name in FakeRecipient._WATCHED:
            object.__getattribute__(self, "_log").append(f"GET:{name}")
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_log").append(f"SET:{name}")
        object.__setattr__(self, name, value)


class FakeAppointment:
    """One Outlook AppointmentItem. Any write is recorded (and fails
    the read-only test)."""

    _WATCHED = {"Subject", "Start", "End", "Organizer", "EntryID",
                "Recipients"}

    def __init__(self, log, subject, start, end, organiser, entry_id,
                 attendees=()):
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "Subject", subject)
        object.__setattr__(self, "Start", start)
        object.__setattr__(self, "End", end)
        object.__setattr__(self, "Organizer", organiser)
        object.__setattr__(self, "EntryID", entry_id)
        object.__setattr__(
            self, "Recipients",
            [FakeRecipient(log, name) for name in attendees])

    def __getattribute__(self, name):
        if name in FakeAppointment._WATCHED:
            object.__getattribute__(self, "_log").append(f"GET:{name}")
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_log").append(f"SET:{name}")
        object.__setattr__(self, name, value)


class FakeItems:
    """An Outlook Items collection. Restrict returns the same events
    (the Python-side match rules do the filtering); the query string
    is recorded so tests can assert the window is sane."""

    def __init__(self, log, events):
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_events", list(events))
        object.__setattr__(self, "IncludeRecurrences", False)

    def __setattr__(self, name, value):
        # IncludeRecurrences is a query setting on the collection,
        # not a mutation of an item — allowed, but still recorded.
        object.__getattribute__(self, "_log").append(f"SET:Items.{name}")
        object.__setattr__(self, name, value)

    def Sort(self, key):
        object.__getattribute__(self, "_log").append(f"CALL:Sort:{key}")
        return self

    def Restrict(self, query):
        object.__getattribute__(self, "_log").append(f"CALL:Restrict:{query}")
        return FakeItems(object.__getattribute__(self, "_log"),
                         object.__getattribute__(self, "_events"))

    def __iter__(self):
        return iter(object.__getattribute__(self, "_events"))

    def __len__(self):
        return len(object.__getattribute__(self, "_events"))


class FakeFolder:
    def __init__(self, log, events):
        self._log = log
        self.Items = FakeItems(log, events)


class FakeNamespace:
    def __init__(self, log, folder):
        self._log = log
        self._folder = folder

    def GetDefaultFolder(self, folder_id):
        self._log.append(f"CALL:GetDefaultFolder:{folder_id}")
        assert folder_id == 9, "must query the default Calendar folder"
        return self._folder

    def GetItemFromID(self, entry_id):
        self._log.append(f"CALL:GetItemFromID:{entry_id}")
        for item in self._folder.Items:
            if item.EntryID == entry_id:
                return item
        raise ValueError(f"no such item: {entry_id}")


class FakeOutlookApp:
    def __init__(self, log, namespace):
        self._log = log
        self._namespace = namespace

    def GetNamespace(self, name):
        self._log.append(f"CALL:GetNamespace:{name}")
        assert name == "MAPI"
        return self._namespace


class FakeClient:
    """Stand-in for the win32com.client module."""

    def __init__(self, log, events, *, dispatch_raises=False):
        self._log = log
        self._events = events
        self._dispatch_raises = dispatch_raises

    def Dispatch(self, prog_id):
        self._log.append(f"CALL:Dispatch:{prog_id}")
        if self._dispatch_raises:
            raise RuntimeError("Outlook is not running")
        assert prog_id == "Outlook.Application"
        folder = FakeFolder(self._log, self._events)
        return FakeOutlookApp(self._log, FakeNamespace(self._log, folder))


def _event(log, subject, start, end, *, entry_id="ID-1",
           organiser="Priya", attendees=("John", "Priya")):
    return FakeAppointment(log, subject, start, end, organiser,
                           entry_id, attendees)


def _window(day="2026-11-10", start="10:00", minutes=30):
    """(recording start, recording end) as aware UTC datetimes."""
    base = datetime.fromisoformat(f"{day}T{start}:00+00:00")
    return base, base + timedelta(minutes=minutes)


# ------------------------------------------------------------ availability ---
@pytest.fixture(autouse=True)
def utc_local_zone(monkeypatch):
    """Pin the local zone to UTC for every test in this module.

    The match/link tests below assume local == UTC (naive Outlook
    times are local times; the session times are UTC). Without this
    they pass or fail depending on the machine's zone and on which
    tests ran before them: time.tzset() state leaks across tests
    (this VM is America/New_York via TZ with /etc/localtime = UTC,
    and an earlier test's tzset leaves the process on UTC). The
    new_york_tz test overrides this within its own scope.
    """
    old_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    if old_tz is None:
        monkeypatch.delenv("TZ", raising=False)
    else:
        monkeypatch.setenv("TZ", old_tz)
    time.tzset()


def test_outlook_available_false_on_linux_with_reason():
    available, reason = outlook_available()
    assert available is False
    assert reason and "Windows" in reason


def test_find_event_outlook_absent_returns_none_never_raises():
    log: list[str] = []
    start, end = _window()
    # Dispatch itself fails (Outlook not running).
    assert find_event_for_session(
        start, end, client=FakeClient(log, [], dispatch_raises=True)) is None
    assert "CALL:Dispatch:Outlook.Application" in log
    # And a client that raises on any attribute access is tolerated.
    assert find_event_for_session(start, end, client=object()) is None


# ------------------------------------------------------------ match rules ----
def test_match_by_15_minute_rule():
    log: list[str] = []
    start, end = _window()  # 10:00–10:30 UTC; VM local zone is UTC
    event = _event(log, "Standup", datetime(2026, 11, 10, 10, 10),
                   datetime(2026, 11, 10, 10, 40))
    found = find_event_for_session(
        start, end, client=FakeClient(log, [event]))
    assert found is not None
    assert found["subject"] == "Standup"
    assert found["entry_id"] == "ID-1"
    assert found["organiser"] == "Priya"
    assert found["attendees"] == ["John", "Priya"]


def test_match_by_50_percent_overlap_rule():
    log: list[str] = []
    start, end = _window()  # 10:00–10:30
    # Starts 20 min before the recording (outside the 15-min rule)
    # but overlaps 20 of the 30 recording minutes: 20 > 50% of 30.
    event = _event(log, "Long planning", datetime(2026, 11, 10, 9, 40),
                   datetime(2026, 11, 10, 10, 20))
    found = find_event_for_session(
        start, end, client=FakeClient(log, [event]))
    assert found is not None
    assert found["subject"] == "Long planning"


def test_no_match_when_event_far_away():
    log: list[str] = []
    start, end = _window()  # 10:00–10:30
    event = _event(log, "Later sync", datetime(2026, 11, 10, 14, 0),
                   datetime(2026, 11, 10, 14, 30))
    assert find_event_for_session(
        start, end, client=FakeClient(log, [event])) is None


def test_no_match_at_exactly_50_percent_overlap():
    log: list[str] = []
    start, end = _window()  # 10:00–10:30
    # Overlap is exactly 15 of 30 min — the rule needs MORE than 50%.
    # Start is 20 min before the recording, so the 15-min rule does
    # not rescue it either.
    event = _event(log, "Edge case", datetime(2026, 11, 10, 9, 40),
                   datetime(2026, 11, 10, 10, 15))
    assert find_event_for_session(
        start, end, client=FakeClient(log, [event])) is None


def test_closest_start_wins_when_two_events_match():
    log: list[str] = []
    start, end = _window()  # 10:00–10:30
    far = _event(log, "Earlier", datetime(2026, 11, 10, 9, 55),
                 datetime(2026, 11, 10, 10, 25), entry_id="ID-EARLY")
    near = _event(log, "Right on time", datetime(2026, 11, 10, 10, 2),
                  datetime(2026, 11, 10, 10, 32), entry_id="ID-NEAR")
    found = find_event_for_session(
        start, end, client=FakeClient(log, [far, near]))
    assert found is not None
    assert found["entry_id"] == "ID-NEAR"


def test_restrict_window_covers_lookback_and_15_min_after():
    log: list[str] = []
    start, end = _window()
    find_event_for_session(start, end, client=FakeClient(log, []))
    restrict = next(c for c in log if c.startswith("CALL:Restrict:"))
    # Lookback: events starting up to 24 h before the recording.
    assert "2026-11-09 10:00" in restrict
    # 15 min past the recording end (the start-window rule).
    assert "2026-11-10 10:45" in restrict


# ------------------------------------------------------------ timezones ------
@pytest.fixture
def new_york_tz(monkeypatch):
    """Run one test with America/New_York as the local zone."""
    old_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    assert datetime.now().astimezone().utcoffset() == timedelta(hours=-5), (
        "expected EST (-05:00) in November")
    yield
    if old_tz is None:
        monkeypatch.delenv("TZ", raising=False)
    else:
        monkeypatch.setenv("TZ", old_tz)
    time.tzset()


def test_utc_stored_session_time_converted_to_local(new_york_tz):
    log: list[str] = []
    # Session stored as UTC in metadata.json: 15:00 UTC = 10:00 EST.
    start = datetime(2026, 11, 10, 15, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=30)
    # Outlook returns local time: 10:05 EST, 5 min after the start.
    event = _event(log, "Standup", datetime(2026, 11, 10, 10, 5),
                   datetime(2026, 11, 10, 10, 35))
    found = find_event_for_session(
        start, end, client=FakeClient(log, [event]))
    assert found is not None
    assert found["subject"] == "Standup"
    # Stored back with its local offset, so the UI shows Outlook time.
    assert found["start"].endswith("-05:00")
    assert found["start"].startswith("2026-11-10T10:05")


def test_format_event_label():
    event = {"subject": "Standup",
             "start": "2026-11-10T10:05:00-05:00",
             "end": "2026-11-10T10:35:00-05:00",
             "attendees": [], "organiser": "Priya", "entry_id": "ID-1"}
    # The label shows the event start in the machine's local zone
    # (UTC on this VM), whatever offset was stored.
    when = datetime.fromisoformat(
        event["start"]).astimezone().strftime("%b %d, %Y, %H:%M")
    assert format_event_label(event) == f"Linked to: Standup, {when}"
    assert format_event_label(None) == ""
    assert format_event_label({}) == ""


# ------------------------------------------------------------ open in Outlook -
def test_open_event_calls_only_read_operations():
    log: list[str] = []
    event = _event(log, "Standup", datetime(2026, 11, 10, 10, 5),
                   datetime(2026, 11, 10, 10, 35), entry_id="ID-OPEN")
    client = FakeClient(log, [event])
    # Display() is the only COM call the fake appointment does not
    # have — record it via a small wrapper.
    displayed: list[str] = []

    def _display(self):
        displayed.append(self.EntryID)

    FakeAppointment.Display = _display  # type: ignore[attr-defined]
    try:
        assert open_event_in_outlook("ID-OPEN", client=client) is True
    finally:
        del FakeAppointment.Display  # type: ignore[attr-defined]
    assert displayed == ["ID-OPEN"]
    assert "CALL:GetItemFromID:ID-OPEN" in log


def test_open_event_missing_returns_false():
    log: list[str] = []
    assert open_event_in_outlook("", client=FakeClient(log, [])) is False
    assert open_event_in_outlook(
        "ID-NOPE", client=FakeClient(log, [])) is False


# ------------------------------------------------------------ session wiring --
def _make_session(tmp_path, *, start="2026-11-10T10:00:00+00:00",
                  end="2026-11-10T10:30:00+00:00", calendar_event="absent"):
    session = tmp_path / "2026-11-10_10-00-00_teams"
    session.mkdir(parents=True)
    metadata = {
        "session_id": session.name, "app_name": "Teams",
        "mode": "system-loopback-fallback", "status": "completed",
        "start_time": start, "end_time": end,
        "duration_seconds": 1800.0, "sample_rate": 48000,
        "channels": 1, "devices": {}, "warning": None, "files": {},
    }
    if calendar_event != "absent":
        metadata["calendar_event"] = calendar_event
    (session / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return session


def _stored_calendar_event(session):
    return json.loads(
        (session / "metadata.json").read_text(encoding="utf-8")
    ).get("calendar_event", "absent")


@pytest.fixture
def outlook_present(monkeypatch):
    """Pretend Outlook is available (Linux VM: it is not)."""
    monkeypatch.setattr(outlook, "outlook_available",
                        lambda: (True, None))


def test_link_session_stores_match_in_metadata(tmp_path, outlook_present):
    log: list[str] = []
    session = _make_session(tmp_path)
    event = _event(log, "Standup", datetime(2026, 11, 10, 10, 5),
                   datetime(2026, 11, 10, 10, 35))
    found = link_session(session, client=FakeClient(log, [event]))
    assert found is not None and found["subject"] == "Standup"
    stored = _stored_calendar_event(session)
    assert stored["subject"] == "Standup"
    assert stored["entry_id"] == "ID-1"


def test_link_session_stores_null_when_no_match(tmp_path, outlook_present):
    log: list[str] = []
    session = _make_session(tmp_path)
    event = _event(log, "Later sync", datetime(2026, 11, 10, 14, 0),
                   datetime(2026, 11, 10, 14, 30))
    assert link_session(session, client=FakeClient(log, [event])) is None
    assert _stored_calendar_event(session) is None


def test_link_session_disabled_leaves_metadata_alone(tmp_path,
                                                     outlook_present):
    log: list[str] = []
    session = _make_session(tmp_path)
    before = (session / "metadata.json").read_text(encoding="utf-8")
    assert link_session(session, enabled=False,
                        client=FakeClient(log, [])) is None
    assert (session / "metadata.json").read_text(encoding="utf-8") == before
    assert not log, "disabled linking must not touch Outlook at all"


def test_link_session_outlook_absent_leaves_metadata_alone(tmp_path):
    # No outlook_present fixture: on this VM outlook_available()
    # is (False, reason) — nothing is written, nothing is called.
    log: list[str] = []
    session = _make_session(tmp_path)
    before = (session / "metadata.json").read_text(encoding="utf-8")
    assert link_session(session, client=FakeClient(log, [])) is None
    assert (session / "metadata.json").read_text(encoding="utf-8") == before
    assert not log


def test_link_session_missing_metadata_never_raises(tmp_path):
    assert link_session(tmp_path / "no-such-session") is None


# ------------------------------------------------------------ read-only ------
def test_link_path_only_calls_read_operations():
    """Every COM call made while linking must be a read.

    The fake records method calls (CALL:), property reads (GET:)
    and property writes (SET:). Anything outside the allowlisted
    read sets in app/calendar/outlook.py fails here — in
    particular any write to an appointment item.
    """
    log: list[str] = []
    start, end = _window()
    event = _event(log, "Standup", datetime(2026, 11, 10, 10, 5),
                   datetime(2026, 11, 10, 10, 35))
    assert find_event_for_session(
        start, end, client=FakeClient(log, [event])) is not None
    calls = {entry.split(":", 1)[1].split(":", 1)[0]
             for entry in log if entry.startswith("CALL:")}
    assert calls <= _ALLOWED_COM_CALLS, f"unexpected COM calls: {calls}"
    reads = {entry.split(":", 1)[1]
             for entry in log if entry.startswith("GET:")}
    assert reads <= _ALLOWED_COM_PROPERTIES, f"unexpected reads: {reads}"
    writes = [entry for entry in log if entry.startswith("SET:")]
    # The only write allowed anywhere is IncludeRecurrences on the
    # Items *collection* (a query setting, not an item mutation).
    assert writes == ["SET:Items.IncludeRecurrences"], (
        f"unexpected COM writes: {writes}")


def test_no_write_path_in_code():
    """Grep-level guard: no persist / body-assignment / item-creation
    code anywhere in this feature.

    Scans app/calendar/ plus every file this feature touches.
    ``.Save(`` would persist an appointment, ``.Body =`` would
    rewrite its body, ``CreateItem`` would create a new item, and
    ``.Delete(`` would delete one — all forbidden (attendees with
    access to the meeting would see the change). If this test
    fails, the feature has grown a write path: remove it.
    """
    targets = [REPO / "app" / "calendar"]
    touched = [
        REPO / "app" / "config.py",
        REPO / "app" / "platform_info.py",
        REPO / "app" / "ui" / "main_window.py",
        REPO / "app" / "ui" / "settings_dialog.py",
    ]
    py_files: list[Path] = []
    for target in targets:
        py_files.extend(sorted(target.rglob("*.py")))
    py_files.extend(touched)
    assert py_files, "expected Outlook module files to exist"
    forbidden = [
        (re.compile(r"\.Save\s*\("), "appointment persist call"),
        (re.compile(r"\.\s*Body\s*="), "appointment body assignment"),
        (re.compile(r"CreateItem\s*\("), "Outlook item creation"),
        (re.compile(r"\.Delete\s*\("), "Outlook item deletion"),
    ]
    violations = []
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        for pattern, what in forbidden:
            for match in pattern.finditer(text):
                line_no = text[:match.start()].count("\n") + 1
                violations.append(
                    f"{path.relative_to(REPO)}:{line_no}: {what}: "
                    f"{match.group(0)!r}")
    assert not violations, (
        "read-only violation — Outlook write path found:\n"
        + "\n".join(violations))


# ------------------------------------------------------------ Qt wiring ------
@pytest.fixture
def qapp():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/offair-demo/runtime")
    Path("/tmp/offair-demo/runtime").mkdir(parents=True, exist_ok=True)
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_settings_checkbox_disabled_without_outlook(qapp):
    pytest.importorskip("PyQt6")
    from app.config import AppConfig
    from app.ui.settings_dialog import SettingsDialog
    dialog = SettingsDialog(AppConfig())
    try:
        # This Linux VM has no Outlook: the toggle is unavailable
        # with the reason, not silently pretending to work.
        assert dialog.outlook_link_check.isEnabled() is False
        assert dialog.outlook_link_check.isChecked() is True
    finally:
        dialog.close()


def test_main_window_calendar_label_states(qapp, tmp_path):
    pytest.importorskip("PyQt6")
    from app.config import AppConfig
    from app.ui.main_window import create_main_window
    linked = _make_session(
        tmp_path, calendar_event={
            "subject": "Standup",
            "start": "2026-11-10T10:05:00+00:00",
            "end": "2026-11-10T10:35:00+00:00",
            "attendees": ["John"], "organiser": "Priya",
            "entry_id": "ID-1"})
    win = create_main_window(AppConfig(output_root=tmp_path))
    win.show()  # offscreen: needed for isVisible() to report True
    # The Outlook line lives on the Session view — show it, or
    # isVisible() reports False for a hidden page's children.
    win.show_view("Session")
    try:
        win.load_session(linked)
        when = datetime.fromisoformat(
            "2026-11-10T10:05:00+00:00").astimezone().strftime(
                "%b %d, %Y, %H:%M")
        assert win.calendar_event_label.text() == (
            f"Linked to: Standup, {when}")
        assert win.open_outlook_btn.isVisible() is True
        # A session checked with no match shows the plain label.
        nomatch = tmp_path / "2026-11-09_10-00-00_teams"
        nomatch.mkdir()
        (nomatch / "metadata.json").write_text(json.dumps({
            "session_id": nomatch.name, "app_name": "Teams",
            "status": "completed",
            "start_time": "2026-11-09T10:00:00+00:00",
            "end_time": "2026-11-09T10:30:00+00:00",
            "duration_seconds": 1800.0, "calendar_event": None,
        }, indent=2), encoding="utf-8")
        win.load_session(nomatch)
        assert win.calendar_event_label.text() == "No calendar event found"
        assert win.open_outlook_btn.isVisible() is False
    finally:
        win.close()
