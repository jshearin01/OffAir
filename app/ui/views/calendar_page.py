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
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

from app.ui.calendar_view import CalendarWidget


def build_calendar_view(win) -> QWidget:
    """Build the Calendar page; the widget lives on ``win``.

    No page title or hint paragraph: the left nav already names
    the page, and the widget's single control row plus the grid
    tooltip carry the guidance, so the grid fills the page.
    """
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    win.calendar_widget = CalendarWidget(win.config.output_root)
    win.calendar_widget.session_requested.connect(
        win._on_calendar_open_session)
    # Speaker filter: the combo lives in the widget's single
    # control row (Nov 24 2026). The main window aliases it and
    # rebuilds its options whenever the view is shown; the
    # filter itself is re-applied after every grid rebuild, so
    # it survives refreshes. Days with no matching session are
    # dimmed, matching badges stay full-strength and clickable
    # (see MainWindow._apply_calendar_speaker_filter).
    win.calendar_speaker_combo = win.calendar_widget.speaker_combo
    win.calendar_speaker_combo.currentIndexChanged.connect(
        win._on_calendar_speaker_changed)
    _original_refresh = win.calendar_widget.refresh

    def _refresh_with_filter() -> None:
        _original_refresh()
        win._apply_calendar_speaker_filter()

    win.calendar_widget.refresh = _refresh_with_filter
    layout.addWidget(win.calendar_widget, 1)
    return page
