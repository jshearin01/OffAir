"""People view — known speakers, learned in earlier calls.

Two layers in one view (no new navigation entry): the list of
known speakers, and a person page for the selected speaker
(see :mod:`app.ui.views.person_view`). List, play a voice
sample, rename, or open a person's page. Nothing session-specific
here: the suggestions for one session live on the Session view,
and the rename panel lives there too — Rename here fills that
panel in (and switches to it), so the identity path (database +
every past session) is unchanged.
"""
from __future__ import annotations

from app.ui.level_meter import QT_AVAILABLE

if QT_AVAILABLE:
    from PyQt6.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QPushButton,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )

from app.ui.views.person_view import build_person_view


def build_people_view(win) -> QWidget:
    """Build the People page from the window's known-speaker widgets.

    Returns a stacked widget: index 0 is the speaker list, index 1
    is the person page. The window switches layers with
    :meth:`MainWindow._show_person_page` / ``_show_people_list``;
    the left navigation always lands on the list layer.
    """
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)

    list_page = QWidget()
    list_layout = QVBoxLayout(list_page)
    list_layout.setContentsMargins(0, 0, 0, 0)
    list_layout.setSpacing(12)
    known_title = QLabel("Known speakers (learned in earlier calls)")
    known_title.setObjectName("sectionTitle")
    list_layout.addWidget(known_title)
    list_layout.addWidget(win.known_speakers_list)
    known_row = QHBoxLayout()
    known_row.addWidget(win.play_known_btn)
    known_row.addWidget(win.rename_known_btn)
    win.open_person_btn = QPushButton("Open person page")
    win.open_person_btn.setToolTip(
        "Open this speaker's page — every meeting with them, "
        "everything they said, and their voice sample.")
    win.open_person_btn.clicked.connect(win._on_open_person_page)
    known_row.addWidget(win.open_person_btn)
    list_layout.addLayout(known_row)
    known_hint = QLabel(
        "Select a speaker above, then Play sample to hear their "
        "voice, Rename to change their name in every session, or "
        "Open person page to see their meetings and everything "
        "they said. Rename opens the rename panel on the Session "
        "view, filled in for the speaker you picked.")
    known_hint.setObjectName("hint")
    known_hint.setWordWrap(True)
    list_layout.addWidget(known_hint)
    list_layout.addStretch(1)

    stack = QStackedWidget()
    stack.addWidget(list_page)
    stack.addWidget(build_person_view(win))
    win.people_stack = stack
    layout.addWidget(stack)
    return page
