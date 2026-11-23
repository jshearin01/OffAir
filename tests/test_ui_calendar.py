"""Calendar view — month grid of recordings (Feature 4, Nov 11 2026).

Two halves, like test_ui_visual.py:

* Qt-free: day derivation (start_time, folder-name fallback,
  neither -> skipped), badge text (time + app + source label +
  Outlook subject when present, omitted cleanly when absent),
  grouping across two months incl. a multi-recording day, and
  corrupt/missing metadata never crashing the collection.
* Qt-gated (``pytest.importorskip("PyQt6")`` per test — run with
  the gui-shots venv, offscreen): nav order (the calendar is its
  own view now, Help still last), grid placement, badge
  contents,
  clicking a badge loads the session, prev/next/Today incl. the
  Dec -> Jan year boundary, corrupt metadata in a live window,
  and geometry-clean at 1280x800 in both themes.

Run the Qt half with::

    QT_QPA_PLATFORM=offscreen \\
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \\
      -m pytest tests/test_ui_calendar.py -q
"""
from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

import pytest

from app.storage.browser import SessionSummary
from app.ui.calendar_view import (
    badge_text,
    calendar_event_subject,
    collect_calendar_entries,
    group_sessions_by_day,
    session_day,
    session_time_text,
)

REPO = Path(__file__).resolve().parent.parent


def _make_session(root: Path, folder: str, *,
                  start: str | None = "2026-11-07T10:00:00+00:00",
                  app_name: str = "Teams", source: str = "teams",
                  calendar_event=None, corrupt: bool = False,
                  no_metadata: bool = False) -> Path:
    """A session folder under ``root`` (metadata only, no audio)."""
    session = root / folder
    session.mkdir(parents=True)
    if no_metadata:
        return session
    if corrupt:
        (session / "metadata.json").write_text(
            "{not json", encoding="utf-8")
        return session
    metadata = {
        "session_id": folder, "app_name": app_name,
        "mode": "system-loopback-fallback", "status": "completed",
        "start_time": start, "end_time": None,
        "duration_seconds": 1800.0, "sample_rate": 48000,
        "channels": 1, "devices": {}, "warning": None, "files": {},
        "source": source, "source_detail": app_name,
    }
    if calendar_event is not None:
        metadata["calendar_event"] = calendar_event
    (session / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return session


def _summary(folder: str, **kwargs) -> SessionSummary:
    return SessionSummary(session_id=folder, path=Path(folder),
                          **kwargs)


@pytest.fixture(autouse=True)
def utc_local_zone(monkeypatch):
    """Pin the local zone to UTC for every test in this module.

    The badge tests seed 10:00 UTC sessions and expect the badge
    to read "10:00" — the badge shows local time (``astimezone()``),
    so without this they pass or fail depending on the machine's
    zone (this VM is America/New_York via TZ with /etc/localtime =
    UTC). Same pattern as tests/test_outlook_linking.py.
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


# ------------------------------------------------------------ Qt-free -------
def test_session_day_from_start_time():
    # Naive start_time is already local time.
    assert session_day(_summary(
        "2026-11-07_10-00-00_teams",
        start_time="2026-11-07T10:00:00")) == date(2026, 11, 7)
    # Aware UTC value converts to local time first.
    assert session_day(_summary(
        "2026-11-07_10-00-00_teams",
        start_time="2026-11-07T10:00:00+00:00")) is not None


def test_session_day_falls_back_to_folder_name():
    assert session_day(_summary(
        "2026-12-03_09-30-00_zoom")) == date(2026, 12, 3)
    assert session_time_text(_summary(
        "2026-12-03_09-30-00_zoom")) == "09:30"


def test_session_day_none_when_no_date_at_all():
    assert session_day(_summary("random-folder")) is None
    assert session_time_text(_summary("random-folder")) == "--:--"


def test_badge_text_shows_time_app_source_and_subject():
    summary = _summary("2026-11-07_10-00-00_teams",
                       start_time="2026-11-07T10:00:00",
                       app_name="Microsoft Teams", source="teams")
    text = badge_text(summary, "Weekly sync")
    assert "10:00" in text
    assert "Microsoft Teams" in text
    assert "Teams" in text  # the source label
    assert "Weekly sync" in text


def test_badge_text_omits_subject_cleanly_when_absent():
    summary = _summary("2026-11-07_10-00-00_teams",
                       start_time="2026-11-07T10:00:00",
                       app_name="Teams", source="teams")
    for subject in (None, "", "   "):
        text = badge_text(summary, subject)
        assert "None" not in text
        assert not text.rstrip().endswith("—"), text
        assert text == "10:00 · Teams · Teams"


def test_group_sessions_by_day_two_months_and_multi_day(tmp_path):
    nov7a = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    nov7b = _make_session(tmp_path, "2026-11-07_14-30-00_zoom",
                          start="2026-11-07T14:30:00+00:00",
                          app_name="Zoom", source="zoom")
    dec3 = _make_session(tmp_path, "2026-12-03_09-00-00_teams",
                         start="2026-12-03T09:00:00+00:00")
    grouped = collect_calendar_entries(tmp_path)
    assert sorted(grouped) == [date(2026, 11, 7), date(2026, 12, 3)]
    nov7 = grouped[date(2026, 11, 7)]
    assert len(nov7) == 2
    # Time order within the day, earliest first.
    assert [e.summary.path for e in nov7] == [nov7a, nov7b]
    assert len(grouped[date(2026, 12, 3)]) == 1
    assert grouped[date(2026, 12, 3)][0].summary.path == dec3


def test_group_sessions_by_day_skips_undated(tmp_path):
    _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    undated = tmp_path / "random-folder"
    undated.mkdir()
    (undated / "metadata.json").write_text(
        json.dumps({"session_id": "random-folder"}), encoding="utf-8")
    grouped = collect_calendar_entries(tmp_path)
    assert sorted(grouped) == [date(2026, 11, 7)]
    assert all(e.summary.path.name != "random-folder"
               for entries in grouped.values() for e in entries)


def test_collect_keeps_corrupt_metadata_sessions(tmp_path):
    corrupt = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            corrupt=True)
    # A folder with no metadata.json at all is not a session in the
    # browser's scan (list_sessions requires metadata.json) — the
    # calendar follows the same scan, so it is skipped, not a crash.
    _make_session(tmp_path, "2026-11-08_11-00-00_zoom",
                  no_metadata=True)
    grouped = collect_calendar_entries(tmp_path)
    # The corrupt session still appears, on its folder-name day — a
    # bad file degrades the badge, it never empties the month.
    assert grouped[date(2026, 11, 7)][0].summary.path == corrupt
    assert grouped[date(2026, 11, 7)][0].subject is None
    assert date(2026, 11, 8) not in grouped


def test_calendar_event_subject_variants(tmp_path):
    event = {"subject": "Weekly sync",
             "start": "2026-11-07T10:00:00+00:00",
             "end": "2026-11-07T10:30:00+00:00",
             "attendees": [], "organiser": "",
             "entry_id": "ID-1"}
    linked = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                           calendar_event=event)
    assert calendar_event_subject(linked) == "Weekly sync"
    null = _make_session(tmp_path, "2026-11-08_10-00-00_teams",
                         calendar_event=None)
    # calendar_event=None means "key absent" in the helper: no link.
    assert calendar_event_subject(null) is None
    stored_null = tmp_path / "2026-11-09_10-00-00_teams"
    stored_null.mkdir()
    (stored_null / "metadata.json").write_text(json.dumps(
        {"session_id": stored_null.name, "calendar_event": None}),
        encoding="utf-8")
    assert calendar_event_subject(stored_null) is None
    blank = _make_session(tmp_path, "2026-11-10_10-00-00_teams",
                          calendar_event={"subject": "  "})
    assert calendar_event_subject(blank) is None
    corrupt = _make_session(tmp_path, "2026-11-11_10-00-00_teams",
                            corrupt=True)
    assert calendar_event_subject(corrupt) is None
    assert calendar_event_subject(tmp_path / "no-such-folder") is None


def test_grouping_uses_subjects_from_metadata(tmp_path):
    event = {"subject": "Budget review",
             "start": "2026-11-07T10:00:00+00:00",
             "end": "2026-11-07T10:30:00+00:00",
             "attendees": [], "organiser": "",
             "entry_id": "ID-2"}
    _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                  calendar_event=event)
    _make_session(tmp_path, "2026-11-08_10-00-00_teams",
                  start="2026-11-08T10:00:00+00:00")
    grouped = collect_calendar_entries(tmp_path)
    assert grouped[date(2026, 11, 7)][0].subject == "Budget review"
    assert grouped[date(2026, 11, 8)][0].subject is None


# ------------------------------------------------------------ Qt-gated ------
@pytest.fixture(scope="module")
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


def _seeded_root(tmp_path):
    """Nov 7 (two sessions, one linked), Dec 3 (one), Nov 8 (one)."""
    event = {"subject": "Weekly sync",
             "start": "2026-11-07T10:00:00+00:00",
             "end": "2026-11-07T10:30:00+00:00",
             "attendees": [], "organiser": "",
             "entry_id": "ID-1"}
    _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                  calendar_event=event)
    _make_session(tmp_path, "2026-11-07_14-30-00_zoom",
                  start="2026-11-07T14:30:00+00:00",
                  app_name="Zoom", source="zoom")
    _make_session(tmp_path, "2026-12-03_09-00-00_teams",
                  start="2026-12-03T09:00:00+00:00")
    _make_session(tmp_path, "2026-11-08_11-00-00_teams",
                  start="2026-11-08T11:00:00+00:00")
    return tmp_path


def _open_window(qapp, tmp_path, theme: str = "light"):
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    win = create_main_window(
        AppConfig(output_root=tmp_path, theme=theme))
    win.show()
    qapp.processEvents()
    return win


def _show_calendar(win) -> None:
    """Show the Calendar view."""
    win.show_view("Calendar")


def test_nav_order_puts_calendar_between_library_and_session(
        qapp, tmp_path):
    win = _open_window(qapp, tmp_path)
    try:
        assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == [
            "Record", "Library", "Calendar", "Session", "People",
            "Settings", "MCP setup", "Help"]
        assert hasattr(win, "calendar_widget")
        # The calendar is a child of the Calendar view.
        assert win.calendar_widget in win._views["Calendar"].findChildren(
            type(win.calendar_widget))
    finally:
        win.close()


def test_grid_places_recordings_on_the_right_days(qapp, tmp_path):
    _seeded_root(tmp_path)
    win = _open_window(qapp, tmp_path)
    try:
        cal = win.calendar_widget
        _show_calendar(win)
        qapp.processEvents()
        cal.set_month(2026, 11)
        qapp.processEvents()
        assert len(cal.sessions_for_day(date(2026, 11, 7))) == 2
        assert len(cal.sessions_for_day(date(2026, 11, 8))) == 1
        assert cal.sessions_for_day(date(2026, 11, 9)) == []
        assert cal.month_label.text() == "November 2026"
        # December: the one session there, and only one badge shown.
        cal.set_month(2026, 12)
        qapp.processEvents()
        assert len(cal.sessions_for_day(date(2026, 12, 3))) == 1
        assert len(cal._badges) == 1
        # A month with no recordings at all: plain empty line.
        cal.set_month(2027, 1)
        qapp.processEvents()
        assert "No recordings in January 2027" in cal.empty_label.text()
    finally:
        win.close()


def test_badges_show_time_app_source_and_subject(qapp, tmp_path):
    _seeded_root(tmp_path)
    win = _open_window(qapp, tmp_path)
    try:
        cal = win.calendar_widget
        _show_calendar(win)
        cal.set_month(2026, 11)
        qapp.processEvents()
        badges = {b.text(): b for b in cal._badges}
        linked = next(t for t in badges if "Weekly sync" in t)
        assert "10:00" in linked
        assert "Teams" in linked
        zoom = next(t for t in badges if "14:30" in t)
        assert "Zoom" in zoom
        assert "Weekly sync" not in zoom
        # The unlinked badge omits the subject cleanly.
        assert not zoom.rstrip().endswith("—")
        assert "None" not in zoom
    finally:
        win.close()


def test_clicking_a_badge_loads_that_session(qapp, tmp_path):
    target = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    _make_session(tmp_path, "2026-11-08_11-00-00_teams",
                  start="2026-11-08T11:00:00+00:00")
    win = _open_window(qapp, tmp_path)
    try:
        cal = win.calendar_widget
        _show_calendar(win)
        cal.set_month(2026, 11)
        qapp.processEvents()
        badge = next(b for b in cal._badges
                     if b.entry.session_path == target)
        badge.click()
        qapp.processEvents()
        assert win.session_dir == target
        assert target.name in win.loaded_session_label.text()
    finally:
        win.close()


def test_prev_next_today_navigate_including_year_boundary(
        qapp, tmp_path):
    _seeded_root(tmp_path)
    win = _open_window(qapp, tmp_path)
    try:
        cal = win.calendar_widget
        cal.set_month(2026, 12)
        assert cal.visible_month == (2026, 12)
        cal.show_next_month()
        assert cal.visible_month == (2027, 1)
        assert cal.month_label.text() == "January 2027"
        cal.show_previous_month()
        assert cal.visible_month == (2026, 12)
        cal.show_previous_month()
        assert cal.visible_month == (2026, 11)
        assert cal.month_label.text() == "November 2026"
        # Today jumps back to the current month from anywhere.
        cal.set_month(2025, 3)
        cal.show_today()
        today = date.today()
        assert cal.visible_month == (today.year, today.month)
    finally:
        win.close()


def test_corrupt_metadata_does_not_crash_the_live_view(
        qapp, tmp_path):
    _make_session(tmp_path, "2026-11-07_10-00-00_teams", corrupt=True)
    _make_session(tmp_path, "2026-11-08_11-00-00_teams",
                  start="2026-11-08T11:00:00+00:00")
    win = _open_window(qapp, tmp_path)
    try:
        cal = win.calendar_widget
        _show_calendar(win)
        cal.set_month(2026, 11)
        qapp.processEvents()
        # The corrupt session still has a badge, on its folder day.
        assert len(cal.sessions_for_day(date(2026, 11, 7))) == 1
        assert len(cal._badges) == 2
    finally:
        win.close()


def test_calendar_geometry_clean_at_1280x800_both_themes(
        qapp, tmp_path):
    """Zero overlaps/clips on the Calendar view,
    both themes, with a busy day (three badges) and a
    multi-recording month."""
    from PyQt6.QtWidgets import QApplication

    from app.config import AppConfig
    from app.ui.main_window import create_main_window
    from app.ui.theme import apply_theme_to_app
    from test_ui_visual import _geometry_problems

    _seeded_root(tmp_path)
    _make_session(tmp_path, "2026-11-07_18-00-00_teams",
                  start="2026-11-07T18:00:00+00:00")
    app = QApplication.instance()
    problems: list[str] = []
    try:
        for theme in ("light", "dark"):
            apply_theme_to_app(app, theme)
            win = create_main_window(
                AppConfig(output_root=tmp_path, theme=theme))
            try:
                win.show()
                qapp.processEvents()
                assert win.width() == 1280 and win.height() == 800
                _show_calendar(win)
                qapp.processEvents()
                cal = win.calendar_widget
                cal.set_month(2026, 11)
                # Two event passes: rebuilding the grid posts a
                # layout request the scroll area only honours on
                # the second pass — after one pass the badges are
                # still squeezed to the viewport height (a one-frame
                # transient in the real app too). The settled
                # geometry is what the audit checks.
                qapp.processEvents()
                qapp.processEvents()
                problems.extend(_geometry_problems(
                    win, label=f"{theme}/Calendar"))
                # And the empty month: no badges, empty line only.
                cal.set_month(2027, 1)
                qapp.processEvents()
                qapp.processEvents()
                problems.extend(_geometry_problems(
                    win, label=f"{theme}/Calendar/empty"))
            finally:
                win.close()
    finally:
        apply_theme_to_app(app, "light")
    assert problems == [], "\n".join(problems)


# ------------------------------------------- Item 5: week + day modes -------
def test_monday_of_week_and_work_week_days():
    # Qt-free week helpers (Nov 20 2026): the work week is Mon–Fri,
    # never seven days, never a weekend day.
    from datetime import timedelta

    from app.ui.calendar_view import monday_of_week, work_week_days

    assert monday_of_week(date(2026, 11, 20)) == date(2026, 11, 16)
    assert monday_of_week(date(2026, 11, 16)) == date(2026, 11, 16)
    assert monday_of_week(date(2026, 11, 22)) == date(2026, 11, 16)
    assert work_week_days(date(2026, 11, 20)) == [
        date(2026, 11, 16) + timedelta(days=i) for i in range(5)]
    # A week crossing the year boundary: Mon Dec 28 – Fri Jan 1.
    days = work_week_days(date(2026, 12, 30))
    assert days[0] == date(2026, 12, 28)
    assert days[-1] == date(2027, 1, 1)
    assert len(days) == 5
    assert all(d.weekday() < 5 for d in days)


def test_work_week_mode_shows_mon_to_fri_badges(qapp, tmp_path):
    _seeded_root(tmp_path)
    win = _open_window(qapp, tmp_path)
    try:
        cal = win.calendar_widget
        _show_calendar(win)
        cal.set_month(2026, 12)  # anchor: Tue Dec 1
        cal.set_mode("workweek")
        qapp.processEvents()
        assert cal.visible_mode == "workweek"
        assert cal.visible_anchor == date(2026, 12, 1)
        # The week of Dec 1 is Mon Nov 30 – Fri Dec 4, and the Dec 3
        # session is the only badge in it.
        assert cal.month_label.text() == "30 November 2026 – 4 December 2026"
        assert len(cal._badges) == 1
        assert "09:00" in cal._badges[0].text()
        assert "Teams" in cal._badges[0].text()
        assert len(cal.sessions_for_day(date(2026, 12, 3))) == 1
        # Day headers are Mon–Fri with dates, never Sat/Sun.
        from PyQt6.QtWidgets import QLabel
        headers = [lbl.text() for lbl in cal.findChildren(QLabel)
                   if lbl.text().startswith(("Mon ", "Tue ", "Wed ",
                                             "Thu ", "Fri ", "Sat ",
                                             "Sun "))]
        assert headers == ["Mon 30", "Tue 01", "Wed 02", "Thu 03",
                           "Fri 04"]
        # Prev/next step a whole week; Today jumps to this week.
        cal.show_previous_month()
        qapp.processEvents()
        assert cal.visible_anchor == date(2026, 11, 24)
        assert cal._badges == []
        assert "No recordings in the week of 23 Nov 2026" in \
            cal.empty_label.text()
        cal.show_next_month()
        qapp.processEvents()
        assert cal.visible_anchor == date(2026, 12, 1)
        assert len(cal._badges) == 1
        cal.show_today()
        assert cal.visible_anchor == date.today()
    finally:
        win.close()


def test_day_mode_shows_one_day_badges_in_time_order(
        qapp, tmp_path):
    _seeded_root(tmp_path)
    win = _open_window(qapp, tmp_path)
    try:
        cal = win.calendar_widget
        _show_calendar(win)
        cal.set_month(2026, 11)  # anchor: Sun Nov 1
        # Anchor on the busy day instead: two badges, time order.
        cal.set_day(date(2026, 11, 7))
        qapp.processEvents()
        assert cal.visible_mode == "day"
        assert cal.month_label.text() == "Saturday 07 November 2026"
        assert len(cal._badges) == 2
        # Same badge content as month mode, in time order.
        assert cal._badges[0].text().startswith("10:00 · Teams")
        assert "Weekly sync" in cal._badges[0].text()
        assert cal._badges[1].text().startswith("14:30 · Zoom")
        # Prev/next step one day; an empty day says so plainly.
        cal.show_previous_month()
        qapp.processEvents()
        assert cal.visible_anchor == date(2026, 11, 6)
        assert cal._badges == []
        assert "No recordings on Friday 06 November 2026" in \
            cal.empty_label.text()
        cal.show_next_month()
        qapp.processEvents()
        assert cal.visible_anchor == date(2026, 11, 7)
        assert len(cal._badges) == 2
        # Clicking a badge in day mode loads that session, as in
        # month mode.
        target = tmp_path / "2026-11-07_10-00-00_teams"
        badge = next(b for b in cal._badges
                     if b.entry.session_path == target)
        badge.click()
        qapp.processEvents()
        assert win.session_dir == target
        # The mode switcher buttons switch modes too.
        from PyQt6.QtWidgets import QPushButton
        month_btn = next(b for b in cal.findChildren(QPushButton)
                         if b.text() == "Month")
        month_btn.click()
        qapp.processEvents()
        assert cal.visible_mode == "month"
        assert cal.month_label.text() == "November 2026"
        assert len(cal._badges) == 3
        with pytest.raises(ValueError):
            cal.set_mode("fortnight")
    finally:
        win.close()
