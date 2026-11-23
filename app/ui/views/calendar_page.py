"""Calendar view — the recordings calendar, full-page.

One badge per recording, on the day it was recorded (Month,
Work week or Day — see :mod:`app.ui.calendar_view`): time, app
name, source and the linked Outlook event's subject when there
is one. Clicking a badge loads that session in the Session view
(the same path as opening from the Library). Split out of the
Library view Nov 20 2026: the month grid needs the whole page
width, and the Library list below it was capped to make room.
"""
from __future__ import annotations

from app.ui.level_meter import QT_AVAILABLE

if QT_AVAILABLE:
    from PyQt6.QtWidgets import (
        QComboBox,
        QHBoxLayout,
        QLabel,
        QVBoxLayout,
        QWidget,
    )

from app.ui.calendar_view import CalendarWidget


def build_calendar_view(win) -> QWidget:
    """Build the Calendar page; the widget lives on ``win``."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)

    title = QLabel("Calendar")
    title.setObjectName("sectionTitle")
    layout.addWidget(title)
    hint = QLabel(
        "One badge per recording, on the day it was recorded. "
        "Switch between Month, Work week and Day below; click a "
        "badge to load that session.")
    hint.setObjectName("hint")
    hint.setWordWrap(True)
    layout.addWidget(hint)

    # Speaker filter: days with no matching session are dimmed,
    # matching badges stay full-strength and clickable (see
    # MainWindow._apply_calendar_speaker_filter). The combo is
    # rebuilt from the library's speakers whenever the view is
    # shown; the filter itself is re-applied after every grid
    # rebuild, so it survives refreshes.
    filter_row = QHBoxLayout()
    filter_row.setSpacing(8)
    filter_label = QLabel("Speaker:")
    filter_label.setObjectName("fieldLabel")
    filter_row.addWidget(filter_label)
    win.calendar_speaker_combo = QComboBox()
    win.calendar_speaker_combo.setToolTip(
        "Dim days with no session where this speaker spoke. "
        "Matching badges stay clickable.")
    win.calendar_speaker_combo.addItem("All speakers", None)
    win.calendar_speaker_combo.currentIndexChanged.connect(
        win._on_calendar_speaker_changed)
    filter_row.addWidget(win.calendar_speaker_combo)
    filter_row.addStretch(1)
    layout.addLayout(filter_row)

    win.calendar_widget = CalendarWidget(win.config.output_root)
    win.calendar_widget.session_requested.connect(
        win._on_calendar_open_session)
    _original_refresh = win.calendar_widget.refresh

    def _refresh_with_filter() -> None:
        _original_refresh()
        win._apply_calendar_speaker_filter()

    win.calendar_widget.refresh = _refresh_with_filter
    layout.addWidget(win.calendar_widget, 1)
    return page
