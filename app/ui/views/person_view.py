"""Person page — one speaker's meetings, words, and voice sample.

Reached from the People view (never from the left navigation):
select a speaker there and open their page. The page shows every
meeting they spoke in (with utterance counts — clicking one loads
that session), everything they said (click an utterance to hear
that moment, where the audio still exists), their voice sample,
and Rename. There is no commitments section: no commitments data
exists, and no placeholder stands in for one. Back returns to the
People list. Layout only, like the other view builders — all
behaviour lives on the main window.
"""
from __future__ import annotations

from app.ui.level_meter import QT_AVAILABLE

if QT_AVAILABLE:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QListWidget,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )


def build_person_view(win) -> QWidget:
    """Build the person page; its widgets live on ``win``."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)

    top_row = QHBoxLayout()
    top_row.setSpacing(8)
    win.person_back_btn = QPushButton("← Back to People")
    win.person_back_btn.setToolTip(
        "Return to the list of known speakers.")
    win.person_back_btn.clicked.connect(win._show_people_list)
    top_row.addWidget(win.person_back_btn)
    top_row.addStretch(1)
    layout.addLayout(top_row)

    win.person_title_label = QLabel("")
    win.person_title_label.setObjectName("sectionTitle")
    layout.addWidget(win.person_title_label)
    win.person_summary_label = QLabel("")
    win.person_summary_label.setObjectName("hint")
    win.person_summary_label.setWordWrap(True)
    layout.addWidget(win.person_summary_label)

    sample_row = QHBoxLayout()
    sample_row.setSpacing(8)
    win.person_sample_btn = QPushButton("▶  Play voice sample")
    win.person_sample_btn.setToolTip(
        "Play the speaker's longest turn, so you can hear who "
        "they are.")
    win.person_sample_btn.clicked.connect(win._on_person_play_sample)
    sample_row.addWidget(win.person_sample_btn)
    win.person_rename_btn = QPushButton("Rename…")
    win.person_rename_btn.setToolTip(
        "Rename this speaker — the rename panel on the Session "
        "view is filled in for you.")
    win.person_rename_btn.clicked.connect(win._on_person_rename)
    sample_row.addWidget(win.person_rename_btn)
    sample_row.addStretch(1)
    layout.addLayout(sample_row)
    win.person_sample_label = QLabel("")
    win.person_sample_label.setObjectName("hint")
    win.person_sample_label.setWordWrap(True)
    layout.addWidget(win.person_sample_label)

    meetings_title = QLabel("Meetings with this speaker")
    meetings_title.setObjectName("sectionTitle")
    layout.addWidget(meetings_title)
    win.person_sessions_list = QListWidget()
    win.person_sessions_list.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    win.person_sessions_list.setToolTip(
        "Every session this speaker spoke in, newest first. "
        "Click one to load it in the Session view.")
    win.person_sessions_list.setMinimumHeight(120)
    win.person_sessions_list.itemClicked.connect(
        win._on_person_session_clicked)
    layout.addWidget(win.person_sessions_list)

    utterances_title = QLabel("Everything they said (click to play)")
    utterances_title.setObjectName("sectionTitle")
    layout.addWidget(utterances_title)
    win.person_utterances_list = QListWidget()
    win.person_utterances_list.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    win.person_utterances_list.setToolTip(
        "Every segment this speaker spoke, newest session first. "
        "Click one to hear that moment, where the audio still "
        "exists.")
    win.person_utterances_list.itemClicked.connect(
        win._on_person_utterance_clicked)
    layout.addWidget(win.person_utterances_list, 1)
    return page
