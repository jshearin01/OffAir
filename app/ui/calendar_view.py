"""Calendar view — a month grid of recordings (Feature 4, Nov 11 2026).

One badge per recording, on the day it was recorded: time, app name,
recording source (via :data:`app.storage.source.SOURCE_LABELS`) and,
when the session is linked to an Outlook event, that event's subject
(read from the session's ``metadata.json`` ``calendar_event`` — the
Outlook link is read-only, the calendar never writes anything).
Clicking a badge loads that session through the main window's
:meth:`app.ui.main_window.MainWindow.load_session` — the same path
as opening a session from the Recording Browser.

Three modes (Nov 20 2026): Month (the original 7-column grid),
Work week (Mon–Fri columns for the week containing the anchor
date — never a 7-day week) and Day (one column for the anchor
date, badges in time order). Badge content and click-to-open are
identical in all three modes; prev/next step by one period in
the current mode, Today jumps to the current period.

Data comes from the same recordings-root scan the browser uses
(:func:`app.storage.browser.list_session_summaries`, via
:func:`app.storage.sessions.list_sessions`); sessions whose
``start_time`` is missing or unparsable fall back to the folder-name
stamp (``YYYY-MM-DD_HH-MM-SS_…``), and sessions with neither are
skipped, never a crash. A session whose ``metadata.json`` is corrupt
still appears on its folder-name day (with no app/source detail
beyond the folder name) — a bad file must not empty the month.

The top half of this module is Qt-free (date/badge grouping pure
functions, unit-tested without PyQt6); the widget at the bottom is
Qt-gated on :data:`app.ui.level_meter.QT_AVAILABLE`, like the rest
of ``app.ui``.
"""
from __future__ import annotations

import calendar as _calendar
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from app.storage.browser import SessionSummary, summarize_session
from app.storage.sessions import METADATA_FILE, list_sessions
from app.storage.source import SOURCE_LABELS, SOURCE_OTHER
from app.ui.level_meter import QT_AVAILABLE

logger = logging.getLogger("offair.ui.calendar")

#: Folder-name stamp: ``YYYY-MM-DD_HH-MM-SS_<slug>`` (local time).
_FOLDER_STAMP_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})")


@dataclass
class CalendarEntry:
    """One recording on the calendar: its summary, its day, and the
    linked Outlook event subject (None when the session has no link)."""

    summary: SessionSummary
    day: date
    subject: str | None = None

    @property
    def session_path(self) -> Path:
        return self.summary.path


def session_local_datetime(summary: SessionSummary) -> datetime | None:
    """The session's start as a local datetime, or None.

    ``start_time`` is stored as UTC in metadata.json; an aware value
    is converted to local time, a naive value is already local (old
    sessions). Unparsable / missing values return None — the caller
    falls back to the folder-name stamp.
    """
    if not summary.start_time:
        return None
    try:
        when = datetime.fromisoformat(summary.start_time)
    except ValueError:
        return None
    if when.tzinfo is not None:
        when = when.astimezone()
    return when


def session_day(summary: SessionSummary) -> date | None:
    """The calendar day a session belongs on, or None when unknown.

    ``start_time`` (local date) first; the folder-name stamp second.
    A session with neither is skipped by the view, never a crash.
    """
    when = session_local_datetime(summary)
    if when is not None:
        return when.date()
    match = _FOLDER_STAMP_RE.match(summary.path.name)
    if match is None:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)),
                    int(match.group(3)))
    except ValueError:
        return None


def session_time_text(summary: SessionSummary) -> str:
    """``"HH:MM"`` for the badge, from ``start_time`` or the folder
    stamp; ``"--:--"`` when neither yields a time."""
    when = session_local_datetime(summary)
    if when is not None:
        return when.strftime("%H:%M")
    match = _FOLDER_STAMP_RE.match(summary.path.name)
    if match is not None:
        return f"{match.group(4)}:{match.group(5)}"
    return "--:--"


def badge_text(summary: SessionSummary,
               subject: str | None = None) -> str:
    """One-line badge text: time, app name, source label, and the
    Outlook subject when present.

    The app name falls back to the source detail (the stored app
    display name) and then to "Unknown" — a badge never shows a
    blank where the app should be. A missing/blank subject is
    omitted cleanly: no trailing separator, no "None".
    """
    app_name = summary.app_name or summary.source_detail or "Unknown"
    source_label = SOURCE_LABELS.get(summary.source, "Other")
    text = (f"{session_time_text(summary)} · {app_name} · "
            f"{source_label}")
    if subject and subject.strip():
        text += f" — {subject.strip()}"
    return text


def calendar_event_subject(session_dir: str | Path) -> str | None:
    """The linked Outlook event's subject for a session, or None.

    Pure read of ``metadata.json`` (``backfill=False`` — reading the
    calendar never writes anything back). Missing file, corrupt
    JSON, missing key, stored null, or a blank subject all mean
    None: the badge simply shows no subject.
    """
    from app.storage.sessions import load_metadata

    try:
        metadata = load_metadata(session_dir, backfill=False)
    except Exception:  # noqa: BLE001 — a bad file means "no subject"
        return None
    event = metadata.get("calendar_event")
    if not isinstance(event, dict):
        return None
    subject = event.get("subject")
    if isinstance(subject, str) and subject.strip():
        return subject.strip()
    return None


def _fallback_summary(session_dir: Path) -> SessionSummary:
    """A minimal summary for a session whose metadata cannot be read.

    The session still appears on its folder-name day; the badge
    shows the folder time and "Unknown · Other" for app/source.
    """
    return SessionSummary(session_id=session_dir.name,
                          path=session_dir, source=SOURCE_OTHER)


def group_sessions_by_day(
        summaries: list[SessionSummary],
        subjects: dict[Path, str | None] | None = None
        ) -> dict[date, list[CalendarEntry]]:
    """Summaries grouped by calendar day (ascending days, entries
    in time order within a day). Sessions with no derivable day
    are skipped. ``subjects`` maps a session path to its linked
    Outlook subject (None/absent = no subject)."""
    subjects = subjects or {}
    grouped: dict[date, list[CalendarEntry]] = {}
    for summary in summaries:
        day = session_day(summary)
        if day is None:
            logger.debug("Calendar: skipping %s — no usable date",
                         summary.path)
            continue
        grouped.setdefault(day, []).append(CalendarEntry(
            summary=summary, day=day,
            subject=subjects.get(summary.path)))
    for entries in grouped.values():
        entries.sort(key=lambda e: (
            session_time_text(e.summary), e.summary.session_id))
    return dict(sorted(grouped.items()))


def monday_of_week(day: date) -> date:
    """The Monday of the week containing ``day``."""
    return day - timedelta(days=day.weekday())


def work_week_days(day: date) -> list[date]:
    """The Mon–Fri days of the week containing ``day``."""
    monday = monday_of_week(day)
    return [monday + timedelta(days=i) for i in range(5)]


def collect_calendar_entries(
        output_root: str | Path) -> dict[date, list[CalendarEntry]]:
    """All sessions under ``output_root``, grouped by day.

    The same scan the Recording Browser uses
    (:func:`app.storage.sessions.list_sessions`), summarised
    one session at a time so a single corrupt ``metadata.json``
    degrades to a fallback summary instead of emptying the
    month (see :func:`_fallback_summary`). Never raises for
    bad session data — a missing output root is an empty month.
    """
    root = Path(output_root)
    summaries: list[SessionSummary] = []
    subjects: dict[Path, str | None] = {}
    try:
        session_dirs = list_sessions(root)
    except Exception:  # noqa: BLE001 — no sessions beats a crash
        logger.exception("Calendar: could not list sessions in %s", root)
        return {}
    for session_dir in session_dirs:
        try:
            summaries.append(summarize_session(session_dir))
        except Exception:  # noqa: BLE001 — corrupt metadata, keep the day
            logger.warning("Calendar: could not summarise %s — "
                           "showing it from its folder name",
                           session_dir, exc_info=True)
            summaries.append(_fallback_summary(session_dir))
        subjects[session_dir] = calendar_event_subject(session_dir)
    return group_sessions_by_day(summaries, subjects)


if QT_AVAILABLE:
    from PyQt6.QtCore import QSize, Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QButtonGroup,
        QComboBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )

    class _BadgeButton(QFrame):
        """One recording as a clickable badge.

        A frame (not a QPushButton): the badge text is two short
        lines — "time · app" / "source — subject" — and a button
        cannot word-wrap, so a long app or subject name would clip
        in a ~120 px day cell. The word-wrapped label grows the
        badge instead, and the frame carries the click (mouse or
        Enter/Space when focused). ``clicked`` carries the session
        path as a string. ``text()`` is the single-line badge text
        (see :func:`badge_text`), for tests and screen readers.
        """

        clicked = pyqtSignal(str)

        def __init__(self, entry: CalendarEntry,
                     parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.entry = entry
            self._text = badge_text(entry.summary, entry.subject)
            self.setObjectName("calendarBadge")
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setToolTip(self._text)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(6, 4, 6, 4)
            layout.setSpacing(0)
            self.label = QLabel(self._display_lines(), self)
            self.label.setObjectName("calendarBadgeText")
            self.label.setWordWrap(True)
            layout.addWidget(self.label)

        def _display_lines(self) -> str:
            """Two lines: "time · app" over "source [— subject]"."""
            app_name = (self.entry.summary.app_name
                        or self.entry.summary.source_detail
                        or "Unknown")
            source_label = SOURCE_LABELS.get(
                self.entry.summary.source, "Other")
            first = (f"{session_time_text(self.entry.summary)} · "
                     f"{app_name}")
            second = source_label
            if self.entry.subject:
                second += f" — {self.entry.subject}"
            return f"{first}\n{second}"

        def text(self) -> str:
            """The single-line badge text (time · app · source …)."""
            return self._text

        def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt name
            """Height for the badge text wrapped at a narrow width.

            The grid host lives in a widgetResizable scroll area,
            which squeezes rows to the viewport when the content's
            minimum size is small — and a word-wrapped QLabel
            reports a ~0 minimum height, so busy-day badges
            collapsed to height 0 and their text clipped (geometry
            audit, Nov 19 2026, when the calendar moved into the
            Library view). Measuring the label at a conservative
            80 px — narrower than any real day cell at 1280x800 —
            makes the row minimum honest, so the scroll area
            scrolls instead of squeezing the badges.
            """
            return QSize(0, self.label.heightForWidth(80) + 8)

        def click(self) -> None:
            """Emit :attr:`clicked` as a mouse click would."""
            self.clicked.emit(str(self.entry.session_path))

        def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt name
            self.clicked.emit(str(self.entry.session_path))
            super().mousePressEvent(event)

        def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt name
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Space):
                self.clicked.emit(str(self.entry.session_path))
                return
            super().keyPressEvent(event)

    class CalendarWidget(QWidget):
        """Month / work-week / day grid of recordings.

        One slim control row: Previous / title / Next / Today,
        the Month | Work week | Day switcher, and the speaker
        filter — then the grid takes the rest of the page.
        Month mode is Monday-first with one cell per day,
        work-week mode is five Mon–Fri columns for the anchor
        date's week, day mode is one column for the anchor date.
        One badge per recording that day, in time order. Clicking
        a badge emits :attr:`session_requested` with the session
        path — the main window loads it via ``load_session``.
        Periods with no recordings show a plain "No recordings"
        line instead of an empty grid. Refreshes on demand
        (:meth:`refresh`); the main window calls it when the
        Calendar view is shown and when the output folder changes.
        """

        session_requested = pyqtSignal(str)

        def __init__(self, output_root: str | Path,
                     parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._output_root = Path(output_root)
            today = date.today()
            self._year = today.year
            self._month = today.month
            # Week/day anchor: the date the work-week and day
            # views are centred on (today at startup).
            self._anchor = today
            # "month" | "workweek" | "day".
            self._mode = "month"
            self._entries: dict[date, list[CalendarEntry]] = {}
            self._badges: list[_BadgeButton] = []

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 4, 0, 0)
            layout.setSpacing(6)

            def _separator() -> QFrame:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.VLine)
                line.setFrameShadow(QFrame.Shadow.Sunken)
                return line

            # One slim control row (Nov 24 2026): navigation, the
            # mode switcher and the speaker filter share a single
            # row so the grid fills the page. The speaker combo
            # lives here; calendar_page.py aliases it onto the
            # main window and connects its signal.
            controls = QWidget()
            controls.setObjectName("slimControls")
            control_row = QHBoxLayout(controls)
            control_row.setContentsMargins(0, 0, 0, 0)
            control_row.setSpacing(8)
            self.prev_btn = QPushButton("‹")
            self.prev_btn.setToolTip("Previous month")
            self.prev_btn.setMaximumWidth(36)
            self.prev_btn.clicked.connect(self.show_previous_month)
            self.next_btn = QPushButton("›")
            self.next_btn.setToolTip("Next month")
            self.next_btn.setMaximumWidth(36)
            self.next_btn.clicked.connect(self.show_next_month)
            self.today_btn = QPushButton("Today")
            self.today_btn.setToolTip("Jump back to the current month")
            self.today_btn.clicked.connect(self.show_today)
            self.month_label = QLabel("")
            self.month_label.setObjectName("sectionTitle")
            self.month_label.setAlignment(
                Qt.AlignmentFlag.AlignCenter)
            control_row.addWidget(self.prev_btn)
            control_row.addWidget(self.month_label, 1)
            control_row.addWidget(self.next_btn)
            control_row.addWidget(self.today_btn)
            control_row.addWidget(_separator())
            self._mode_group = QButtonGroup(self)
            self._mode_buttons: list[tuple[QPushButton, str]] = []
            for mode, text, tip in (
                    ("month", "Month",
                     "Month grid — one badge per recording"),
                    ("workweek", "Work week",
                     "Monday to Friday of the anchor date's week"),
                    ("day", "Day",
                     "One day, badges in time order")):
                button = QPushButton(text)
                button.setCheckable(True)
                button.setChecked(mode == "month")
                button.setToolTip(tip)
                button.clicked.connect(
                    lambda _checked, _mode=mode: self.set_mode(_mode))
                self._mode_group.addButton(button)
                self._mode_buttons.append((button, mode))
                control_row.addWidget(button)
            control_row.addWidget(_separator())
            speaker_title = QLabel("Speaker:")
            speaker_title.setObjectName("fieldLabel")
            control_row.addWidget(speaker_title)
            self.speaker_combo = QComboBox()
            self.speaker_combo.setToolTip(
                "Dim days with no session where this speaker spoke. "
                "Matching badges stay clickable.")
            self.speaker_combo.addItem("All speakers", None)
            control_row.addWidget(self.speaker_combo)
            layout.addWidget(controls)

            self.empty_label = QLabel("")
            self.empty_label.setObjectName("hint")
            self.empty_label.setWordWrap(True)
            layout.addWidget(self.empty_label)

            self._scroll = QScrollArea(self)
            self._scroll.setToolTip(
                "One badge per recording, on the day it was "
                "recorded. Click a badge to load that session.")
            self._scroll.setWidgetResizable(True)
            self._scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._grid_host = QWidget(self._scroll)
            self._grid = QGridLayout(self._grid_host)
            self._grid.setContentsMargins(0, 0, 0, 0)
            self._grid.setSpacing(6)
            self._scroll.setWidget(self._grid_host)
            layout.addWidget(self._scroll, 1)

            self.refresh()

        # -------------------------------------------------- navigation ---
        @property
        def visible_month(self) -> tuple[int, int]:
            """``(year, month)`` currently shown — the anchor's
            month in work-week/day mode."""
            return self._year, self._month

        @property
        def visible_mode(self) -> str:
            """``"month"``, ``"workweek"`` or ``"day"``."""
            return self._mode

        @property
        def visible_anchor(self) -> date:
            """The anchor date the work-week/day view is centred on."""
            return self._anchor

        def set_mode(self, mode: str) -> None:
            """Switch Month | Work week | Day (raises ValueError on
            anything else). The anchor is kept, so switching modes
            stays on the same week/day."""
            if mode not in ("month", "workweek", "day"):
                raise ValueError(f"unknown calendar mode: {mode!r}")
            self._mode = mode
            for button, name in self._mode_buttons:
                button.setChecked(name == mode)
            self._update_nav_tooltips()
            self.refresh()

        def set_month(self, year: int, month: int) -> None:
            """Show ``year``/``month`` (1–12; raises ValueError
            outside it — callers use the prev/next helpers)."""
            if not 1 <= month <= 12:
                raise ValueError(f"month out of range: {month}")
            self._year, self._month = year, month
            self._anchor = date(year, month, 1)
            self.refresh()

        def set_anchor(self, day: date) -> None:
            """Move the anchor to ``day`` (week/day views centre
            on it; the month view follows the anchor's month).
            The public way to point a mode at a day — tests and
            callers should use this, never ``_anchor`` directly
            (N5, Nov 20 2026 review fix)."""
            self._anchor = day
            self._year, self._month = day.year, day.month
            self.refresh()

        def set_day(self, day: date) -> None:
            """Switch to Day mode showing ``day``."""
            self.set_anchor(day)
            self.set_mode("day")

        def _step(self, delta: int) -> None:
            """Step prev/next by one period in the current mode."""
            if self._mode == "month":
                month = self._month + delta
                year = self._year
                while month < 1:
                    month += 12
                    year -= 1
                while month > 12:
                    month -= 12
                    year += 1
                self._year, self._month = year, month
                self._anchor = date(year, month, 1)
            elif self._mode == "workweek":
                self._anchor += timedelta(days=7 * delta)
                self._year, self._month = (self._anchor.year,
                                           self._anchor.month)
            else:  # day
                self._anchor += timedelta(days=delta)
                self._year, self._month = (self._anchor.year,
                                           self._anchor.month)
            self.refresh()

        def show_previous_month(self) -> None:
            self._step(-1)

        def show_next_month(self) -> None:
            self._step(1)

        def show_today(self) -> None:
            today = date.today()
            self._year, self._month = today.year, today.month
            self._anchor = today
            self.refresh()

        def _update_nav_tooltips(self) -> None:
            """Prev/Next/Today tooltips name the current mode's step."""
            period = {"month": "month", "workweek": "week",
                      "day": "day"}[self._mode]
            self.prev_btn.setToolTip(f"Previous {period}")
            self.next_btn.setToolTip(f"Next {period}")
            self.today_btn.setToolTip(
                f"Jump back to the current {period}")

        # -------------------------------------------------------- data ---
        def set_output_root(self, output_root: str | Path) -> None:
            """Follow an output-folder change, then re-list."""
            self._output_root = Path(output_root)
            self.refresh()

        def sessions_for_day(self, day: date) -> list[CalendarEntry]:
            """Entries for ``day`` (empty list when none)."""
            return list(self._entries.get(day, []))

        def refresh(self) -> None:
            """Re-list sessions and rebuild the grid. Never raises
            for bad session data — a failed scan shows the empty
            state instead of crashing the tab."""
            try:
                self._entries = collect_calendar_entries(
                    self._output_root)
            except Exception:  # noqa: BLE001 — the tab must stay up
                logger.exception("Calendar refresh failed")
                self._entries = {}
            self._rebuild()

        # ------------------------------------------------------- layout ---
        def _rebuild(self) -> None:
            """Rebuild the grid for the current mode."""
            if self._mode == "month":
                self._rebuild_month()
            elif self._mode == "workweek":
                self._rebuild_week()
            else:
                self._rebuild_day()

        def _clear_grid(self) -> None:
            """Drop the old grid widgets.

            Layouts own them; deleting the widgets is what frees
            them. Hide each one first: deleteLater() only destroys
            them on the next event loop turn, and until then a stale
            cell would still paint over the fresh grid (seen in a
            screenshot grab taken right after a rebuild).
            """
            while self._grid.count():
                item = self._grid.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.hide()
                    widget.deleteLater()
            self._badges = []
            # N4 (Nov 20 2026 review fix): month mode stretches
            # rows 1–6 and columns 0–6, week/day fewer — clear
            # them all here, or a mode switch inherits stale
            # stretches (harmless in practice — empty rows/cols
            # take no space — but wrong by construction).
            for row in range(7):
                self._grid.setRowStretch(row, 0)
            for col in range(7):
                self._grid.setColumnStretch(col, 0)

        def _make_day_cell(self, day: date) -> QFrame:
            """One day cell: day number + that day's badges.

            No fixed minimum height: the row's height is the
            tallest cell's contents (day label + badges). A fixed
            minimum would override the layout's own minimum and
            squeeze badges on busy days — the grid row would stay
            at the minimum while the badges need more. Row stretch
            (set by the caller) evens the rows out when there is
            slack, and the scroll area takes over when there is not.
            """
            cell = QFrame()
            cell.setObjectName("card")
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(6, 4, 6, 4)
            cell_layout.setSpacing(2)
            day_label = QLabel(str(day.day))
            day_label.setObjectName("fieldLabel")
            day_label.setAlignment(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignTop)
            cell_layout.addWidget(day_label)
            for entry in self._entries.get(day, []):
                badge = _BadgeButton(entry, cell)
                badge.clicked.connect(self.session_requested)
                cell_layout.addWidget(badge)
                self._badges.append(badge)
            cell_layout.addStretch(1)
            return cell

        def _rebuild_month(self) -> None:
            """The original month grid (Monday-first, 7 columns)."""
            month_name = _calendar.month_name[self._month]
            self.month_label.setText(f"{month_name} {self._year}")
            month_days = {day for day in self._entries
                          if day.year == self._year
                          and day.month == self._month}
            if month_days:
                self.empty_label.setText("")
            else:
                self.empty_label.setText(
                    f"No recordings in {month_name} {self._year}.")
            self._clear_grid()
            for col, name in enumerate(
                    ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
                header = QLabel(name)
                header.setObjectName("fieldLabel")
                header.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._grid.addWidget(header, 0, col)
            for col in range(7):
                self._grid.setColumnStretch(col, 1)
            weeks = _calendar.Calendar(
                firstweekday=_calendar.MONDAY).monthdayscalendar(
                    self._year, self._month)
            for row, week in enumerate(weeks, start=1):
                self._grid.setRowStretch(row, 1)
                for col, day_num in enumerate(week):
                    if day_num == 0:
                        continue
                    day = date(self._year, self._month, day_num)
                    self._grid.addWidget(
                        self._make_day_cell(day), row, col)

        def _rebuild_week(self) -> None:
            """Work week: Mon–Fri columns for the anchor's week."""
            days = work_week_days(self._anchor)
            first, last = days[0], days[-1]
            if first.month == last.month:
                title = (f"{first.day} \u2013 {last.day} "
                         f"{_calendar.month_name[first.month]} "
                         f"{first.year}")
            else:
                title = (f"{first.day} "
                         f"{_calendar.month_name[first.month]} "
                         f"{first.year} \u2013 {last.day} "
                         f"{_calendar.month_name[last.month]} "
                         f"{last.year}")
            self.month_label.setText(title)
            week_days = {day for day in self._entries if day in days}
            if week_days:
                self.empty_label.setText("")
            else:
                self.empty_label.setText(
                    f"No recordings in the week of {first:%d %b %Y}.")
            self._clear_grid()
            for col, day in enumerate(days):
                header = QLabel(day.strftime("%a %d"))
                header.setObjectName("fieldLabel")
                header.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._grid.addWidget(header, 0, col)
                self._grid.setColumnStretch(col, 1)
            self._grid.setRowStretch(1, 1)
            for col, day in enumerate(days):
                self._grid.addWidget(self._make_day_cell(day), 1, col)

        def _rebuild_day(self) -> None:
            """Day: one column for the anchor date, badges in time
            order."""
            day = self._anchor
            self.month_label.setText(day.strftime("%A %d %B %Y"))
            if self._entries.get(day):
                self.empty_label.setText("")
            else:
                self.empty_label.setText(
                    f"No recordings on {day:%A %d %B %Y}.")
            self._clear_grid()
            header = QLabel(day.strftime("%A %d %B"))
            header.setObjectName("fieldLabel")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid.addWidget(header, 0, 0)
            self._grid.setColumnStretch(0, 1)
            self._grid.setRowStretch(1, 1)
            self._grid.addWidget(self._make_day_cell(day), 1, 0)

    __all__ = [
        "CalendarEntry",
        "CalendarWidget",
        "badge_text",
        "calendar_event_subject",
        "collect_calendar_entries",
        "group_sessions_by_day",
        "monday_of_week",
        "session_day",
        "work_week_days",
        "session_local_datetime",
        "session_time_text",
    ]
else:
    # No PyQt6: the widget does not exist, but the name must
    # still import — main_window.py imports it unconditionally,
    # and its __init__ raises before building any tab when Qt
    # is absent (see test_main_window_imports_without_qt…).
    CalendarWidget = None  # type: ignore[assignment,misc]

    __all__ = [
        "CalendarEntry",
        "CalendarWidget",
        "badge_text",
        "calendar_event_subject",
        "collect_calendar_entries",
        "group_sessions_by_day",
        "monday_of_week",
        "session_day",
        "work_week_days",
        "session_local_datetime",
        "session_time_text",
    ]
