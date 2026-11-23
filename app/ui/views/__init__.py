"""Main-window views (Nov 19 2026 redesign).

Each module builds one focused view for :class:`MainWindow` from
the widgets the window already owns — the builders do layout only,
no behaviour: all signals, workers and state stay in
``app/ui/main_window.py``. A builder takes the window (``win``) and
returns the page widget; widgets a view needs that no other view
owns (the Library search box, the Session empty state) are created
by that view's builder and stored on ``win`` like the rest.
"""
from __future__ import annotations

from app.ui.views.calendar_page import build_calendar_view
from app.ui.views.library_view import build_library_view
from app.ui.views.more_view import build_help_view, build_mcp_view
from app.ui.views.people_view import build_people_view
from app.ui.views.record_view import build_record_view
from app.ui.views.session_view import build_session_view
from app.ui.views.settings_view import build_settings_view

# Left-navigation order (also the stacked-widget order). Flat
# since Nov 20 2026: every row is a view with a self-explanatory
# name — the old "More" section header is gone, so MCP setup and
# Help are rows like the rest, not grouped under a heading.
VIEW_NAMES = ("Record", "Library", "Calendar", "Session", "People",
              "Settings", "MCP setup", "Help")

__all__ = [
    "VIEW_NAMES",
    "build_calendar_view",
    "build_help_view",
    "build_library_view",
    "build_mcp_view",
    "build_people_view",
    "build_record_view",
    "build_session_view",
    "build_settings_view",
]
