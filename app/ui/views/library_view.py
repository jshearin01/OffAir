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
    from PyQt6.QtCore import QSize, Qt
    from PyQt6.QtGui import (
        QFont,
        QFontMetrics,
        QKeySequence,
        QShortcut,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QPushButton,
        QStyledItemDelegate,
        QStyle,
        QStyleOptionViewItem,
        QVBoxLayout,
        QWidget,
    )

    class LibraryRowDelegate(QStyledItemDelegate):
        """Two-line Library rows: bold title, muted meta line.

        Session rows carry ``"title\\nmeta"`` text; the delegate
        draws line 1 bold in the theme text colour and line 2 in
        the muted placeholder colour, both elided at the row edge
        (the full text stays in the item tooltip). Single-line
        rows (week headers) fall through to the default paint.
        Colours come from the widget palette, so the stylesheet's
        light/dark themes and the accent selection colour apply
        without the delegate knowing the theme.
        """

        def paint(self, painter, option, index) -> None:
            text = index.data(Qt.ItemDataRole.DisplayRole) or ""
            lines = str(text).split("\n")
            if len(lines) < 2:
                super().paint(painter, option, index)
                return
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            widget = opt.widget
            style = (widget.style() if widget is not None
                     else QApplication.style())
            style.drawPrimitive(
                QStyle.PrimitiveElement.PE_PanelItemViewItem,
                opt, painter, widget)
            selected = bool(
                opt.state & QStyle.StateFlag.State_Selected)
            if selected:
                title_color = opt.palette.highlightedText().color()
                meta_color = title_color
            else:
                title_color = opt.palette.text().color()
                meta_color = opt.palette.placeholderText().color()
            rect = opt.rect.adjusted(12, 7, -12, -7)
            title_font = QFont(opt.font)
            title_font.setBold(True)
            title_fm = QFontMetrics(title_font)
            meta_fm = QFontMetrics(opt.font)
            title = title_fm.elidedText(
                lines[0], Qt.TextElideMode.ElideRight, rect.width())
            meta = meta_fm.elidedText(
                lines[1], Qt.TextElideMode.ElideRight, rect.width())
            painter.save()
            painter.setFont(title_font)
            painter.setPen(title_color)
            painter.drawText(
                rect.x(), rect.y(), rect.width(),
                title_fm.lineSpacing(),
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignVCenter, title)
            painter.setFont(opt.font)
            painter.setPen(meta_color)
            painter.drawText(
                rect.x(), rect.y() + title_fm.lineSpacing() + 2,
                rect.width(), meta_fm.lineSpacing(),
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignVCenter, meta)
            painter.restore()
            if opt.state & QStyle.StateFlag.State_HasFocus:
                focus_opt = QStyleOptionViewItem(opt)
                style.drawPrimitive(
                    QStyle.PrimitiveElement.PE_FrameFocusRect,
                    focus_opt, painter, widget)

        def sizeHint(self, option, index) -> QSize:
            text = index.data(Qt.ItemDataRole.DisplayRole) or ""
            if "\n" not in str(text):
                return super().sizeHint(option, index)
            title_font = QFont(option.font)
            title_font.setBold(True)
            height = (QFontMetrics(title_font).lineSpacing()
                      + QFontMetrics(option.font).lineSpacing() + 16)
            return QSize(200, height)

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
    win.library_list.setItemDelegate(LibraryRowDelegate(win.library_list))
    win.library_list.setContextMenuPolicy(
        Qt.ContextMenuPolicy.CustomContextMenu)
    win.library_list.customContextMenuRequested.connect(
        win._on_library_context_menu)
    win.library_delete_shortcut = QShortcut(
        QKeySequence(Qt.Key.Key_Delete), win.library_list)
    win.library_delete_shortcut.setContext(
        Qt.ShortcutContext.WidgetWithChildrenShortcut)
    win.library_delete_shortcut.activated.connect(win._on_library_delete)
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
    win.library_rename_btn = QPushButton("Rename")
    win.library_rename_btn.setMinimumHeight(40)
    win.library_rename_btn.setToolTip(
        "Rename the selected session — a custom name for sessions "
        "not tied to an Outlook calendar event.")
    win.library_rename_btn.setEnabled(False)
    win.library_rename_btn.clicked.connect(win._on_library_rename)
    open_row.addWidget(win.library_rename_btn)
    win.library_delete_btn = QPushButton("Delete")
    win.library_delete_btn.setObjectName("dangerButton")
    win.library_delete_btn.setMinimumHeight(40)
    win.library_delete_btn.setToolTip(
        "Delete the selected session permanently — its audio, "
        "transcript, search entries and session record are "
        "removed. A confirmation is asked first.")
    win.library_delete_btn.setEnabled(False)
    win.library_delete_btn.clicked.connect(win._on_library_delete)
    open_row.addWidget(win.library_delete_btn)
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
