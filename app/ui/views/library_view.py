"""Library view — the session list, search and source filter.

The in-window library: search (text / semantic / hybrid, same
modes as the Recording Browser), a source filter, the session list
(newest first). Double-click a session — or Open — loads it in the
Session view. The standalone Recording Browser dialog (header
"Recordings" button) is still there for the transcript-preview
side of the same library. The calendar lives on its own Calendar
page now (Nov 20 2026) — the month grid needs the whole page
width, so the list is no longer capped to make room for it.
"""
from __future__ import annotations

from app.ui.level_meter import QT_AVAILABLE

if QT_AVAILABLE:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QComboBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

from app.config import DEFAULT_SEARCH_MODE, SEARCH_MODES


def build_library_view(win) -> QWidget:
    """Build the Library page; the search/list widgets live on ``win``."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)

    title = QLabel("Library")
    title.setObjectName("sectionTitle")
    layout.addWidget(title)
    hint = QLabel(
        "Every recording, starred first, then newest first, grouped "
        "by week. Search by exact words or by meaning; filter by "
        "what a recording is of, or by who spoke in it. Open a "
        "session to work on it — transcript, speakers and "
        "processing all follow the opened session.")
    hint.setObjectName("hint")
    hint.setWordWrap(True)
    layout.addWidget(hint)

    search_row = QHBoxLayout()
    search_row.setSpacing(8)
    win.library_search_edit = QLineEdit()
    win.library_search_edit.setPlaceholderText("Search recordings…")
    win.library_search_edit.setClearButtonEnabled(True)
    win.library_search_edit.setMinimumHeight(40)
    win.library_search_edit.textChanged.connect(
        win._on_library_search_changed)
    search_row.addWidget(win.library_search_edit, 1)
    win.library_mode_combo = QComboBox()
    win.library_mode_combo.setMinimumHeight(40)
    win.library_mode_combo.addItem(
        "Text — exact words (fast, always works offline)", "text")
    win.library_mode_combo.addItem(
        "Semantic — meaning (downloads the MiniLM model once, "
        "~22 MB)", "semantic")
    win.library_mode_combo.addItem(
        "Hybrid — exact matches first, then meaning (recommended)",
        "hybrid")
    win.library_mode_combo.setCurrentIndex(
        win.library_mode_combo.findData(
            win.config.search_mode
            if win.config.search_mode in SEARCH_MODES
            else DEFAULT_SEARCH_MODE))
    win.library_mode_combo.currentIndexChanged.connect(
        win._on_library_mode_changed)
    search_row.addWidget(win.library_mode_combo)
    win.library_source_combo = QComboBox()
    win.library_source_combo.setMinimumHeight(40)
    win.library_source_combo.setToolTip(
        "Show only sessions recorded from one source — Teams, "
        "Zoom, a browser, a Claude Code session…")
    win.library_source_combo.addItem("All sources", None)
    win.library_source_combo.currentIndexChanged.connect(
        win._on_library_source_changed)
    search_row.addWidget(win.library_source_combo)
    win.library_speaker_combo = QComboBox()
    win.library_speaker_combo.setMinimumHeight(40)
    win.library_speaker_combo.setToolTip(
        "Show only sessions where one speaker spoke — display "
        "name, exact match.")
    win.library_speaker_combo.addItem("All speakers", None)
    win.library_speaker_combo.currentIndexChanged.connect(
        win._on_library_speaker_changed)
    search_row.addWidget(win.library_speaker_combo)
    layout.addLayout(search_row)

    # The list fills the page now that the calendar has its own
    # page (Nov 20 2026): no maximum height any more.
    sessions_label = QLabel("Sessions:")
    sessions_label.setObjectName("fieldLabel")
    layout.addWidget(sessions_label)
    win.library_list = QListWidget()
    win.library_list.setWordWrap(True)
    win.library_list.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    win.library_list.setMinimumHeight(160)
    win.library_list.itemClicked.connect(win._on_library_clicked)
    win.library_list.itemDoubleClicked.connect(win._on_library_open)
    layout.addWidget(win.library_list, 1)
    open_row = QHBoxLayout()
    open_row.setSpacing(8)
    win.library_status_label = QLabel("")
    win.library_status_label.setObjectName("hint")
    win.library_status_label.setWordWrap(True)
    open_row.addWidget(win.library_status_label, 1)
    win.library_star_btn = QPushButton("☆ Star")
    win.library_star_btn.setMinimumHeight(40)
    win.library_star_btn.setToolTip(
        "Star the selected session — starred sessions sort "
        "first in the Library. Press again to unstar.")
    win.library_star_btn.setEnabled(False)
    win.library_star_btn.clicked.connect(win._on_library_star_toggle)
    open_row.addWidget(win.library_star_btn)
    win.library_open_btn = QPushButton("Open in Session view")
    win.library_open_btn.setObjectName("primaryButton")
    win.library_open_btn.setMinimumHeight(40)
    win.library_open_btn.setToolTip(
        "Load the selected session into the Session view — its "
        "transcript, speakers and processing all follow it.")
    win.library_open_btn.setEnabled(False)
    win.library_open_btn.clicked.connect(win._on_library_open)
    open_row.addWidget(win.library_open_btn)
    layout.addLayout(open_row)
    # Disk usage: free space on the recordings volume + what
    # retention would delete on the next prune (same helpers the
    # low-disk warning polls while recording).
    win.disk_status_label = QLabel("")
    win.disk_status_label.setObjectName("hint")
    win.disk_status_label.setWordWrap(True)
    layout.addWidget(win.disk_status_label)
    return page
