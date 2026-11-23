"""Record view — only recording, plus the live transcript.

Left: the Record card — the target-app picker, the capture-mode
callout, live level meters, transport, and the status/session
labels — and under it the compact "Scheduled recordings" card
(Nov 22 2026): the pending-schedule list and the one-off add
form. Nothing else: microphones, system-audio devices, record
mode, source and the output folder are settings now (Nov 20 2026
simplification) — devices live in the Settings view's "Recording
devices" section, mode/source/output folder in the Settings form.
The one control added since (Nov 21 2026): the Dictation toggle
beside Record — mic-only, spoken side only — and the Bookmark
action button in the transport row, which is an action button
like Pause/Resume/Stop. Right: the live transcript list —
segments land here as the 5 s poller transcribes them while
recording — with the active-model line, and the manual near-live
run for a recorded session underneath it, so the list and its
button stay together. Nothing about diarization, speakers or
settings lives here.
"""
from __future__ import annotations

from app.ui.level_meter import QT_AVAILABLE

if QT_AVAILABLE:
    from PyQt6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QVBoxLayout,
        QWidget,
    )


def build_schedules_card(win) -> QWidget:
    """The "Scheduled recordings" card (Nov 22 2026).

    Compact by design — the 1.0.4 simplification removed the
    settings clutter from this view, and this card must not
    bring it back: a short list (pending + last week's history),
    two form rows, one consent hint. The form shares rows
    (date/time beside the app combo, subject beside Schedule)
    and Cancel lives in the title row, because the Record card
    above plus this card must fit the 1280x800 target together
    (see the geometry audit — Qt compresses the Record card's
    own contents when the column is taller than the view).
    All behaviour lives on the window; this builder only lays
    the widgets out.
    """
    card = QFrame()
    card.setObjectName("card")
    card.setFixedWidth(380)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 8, 16, 8)
    layout.setSpacing(3)
    title_row = QHBoxLayout()
    title = QLabel("Scheduled recordings")
    title.setObjectName("sectionTitle")
    title_row.addWidget(title)
    title_row.addStretch(1)
    title_row.addWidget(win.schedule_cancel_btn)
    layout.addLayout(title_row)
    layout.addWidget(win.schedule_list)
    when_row = QHBoxLayout()
    when_row.addWidget(win.schedule_datetime, 1)
    when_row.addWidget(win.schedule_app_combo, 1)
    layout.addLayout(when_row)
    subject_row = QHBoxLayout()
    subject_row.addWidget(win.schedule_subject_edit, 1)
    subject_row.addWidget(win.schedule_add_btn)
    layout.addLayout(subject_row)
    layout.addWidget(win.schedule_hint_label)
    return card


def build_record_view(win) -> QWidget:
    """Build the Record page from the window's recording widgets."""
    page = QWidget()
    layout = QHBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(16)

    # Left: the Record card — the same controls as the old layout,
    # 6 px gaps: the card's content is tall, and at 800 px height
    # the layout runs out of room and clips the word-wrapped status
    # label if the gaps grow (see the old Phase 6 comments).
    record_card = QFrame()
    record_card.setObjectName("card")
    record_card.setFixedWidth(380)
    record_layout = QVBoxLayout(record_card)
    record_layout.setContentsMargins(16, 12, 16, 12)
    record_layout.setSpacing(4)
    record_title_row = QHBoxLayout()
    record_title = QLabel("Record")
    record_title.setObjectName("sectionTitle")
    record_title_row.addWidget(record_title)
    record_title_row.addStretch(1)
    record_title_row.addWidget(win.auto_badge)
    record_layout.addLayout(record_title_row)
    record_layout.addWidget(win.mode_label)
    app_picker_row = QHBoxLayout()
    app_picker_label = QLabel("App:")
    app_picker_label.setObjectName("fieldLabel")
    app_picker_label.setToolTip(
        "The app whose audio is recorded in “Record one app” mode. "
        "The list refreshes when you open it.")
    app_picker_row.addWidget(app_picker_label)
    app_picker_row.addWidget(win.process_combo, 1)
    record_layout.addLayout(app_picker_row)
    # Stored on the window so the Dictation toggle can hide the
    # picker + system meter (mic-only) and show them again.
    win.app_picker_label = app_picker_label
    mic_level_label = QLabel("Mic level:")
    mic_level_label.setObjectName("fieldLabel")
    record_layout.addWidget(mic_level_label)
    record_layout.addWidget(win.mic_meter)
    system_level_label = QLabel("System level:")
    system_level_label.setObjectName("fieldLabel")
    record_layout.addWidget(system_level_label)
    record_layout.addWidget(win.system_meter)
    win.system_level_label = system_level_label
    record_row = QHBoxLayout()
    record_row.addWidget(win.record_btn, 1)
    record_row.addWidget(win.dictation_btn)
    record_layout.addLayout(record_row)
    record_layout.addWidget(win.dictation_note_label)
    transport_row = QHBoxLayout()
    transport_row.addWidget(win.pause_btn)
    transport_row.addWidget(win.resume_btn)
    transport_row.addWidget(win.stop_btn)
    record_layout.addLayout(transport_row)
    # Bookmark on its own row: four transport buttons do not fit
    # the 380 px card side by side — the row compressed them below
    # their size hints and clipped their labels ("Paus", "ookmar").
    bookmark_row = QHBoxLayout()
    bookmark_row.addWidget(win.bookmark_btn, 1)
    record_layout.addLayout(bookmark_row)
    record_layout.addWidget(win.session_label)
    record_layout.addStretch(1)
    # Left column: the Record card above, the schedules card
    # below it (Nov 22 2026). The column is a layout, not one
    # tall card, so the schedules card stays a separate compact
    # card and the Record card keeps its own height.
    left_column = QVBoxLayout()
    left_column.setContentsMargins(0, 0, 0, 0)
    left_column.setSpacing(8)
    left_column.addWidget(record_card)
    left_column.addWidget(build_schedules_card(win))
    left_column.addStretch(1)
    layout.addLayout(left_column)

    # Right: the live transcript as it lands.
    live_column = QVBoxLayout()
    live_column.setSpacing(8)
    live_title = QLabel("Live transcript")
    live_title.setObjectName("sectionTitle")
    live_column.addWidget(live_title)
    live_hint = QLabel(
        "Segments appear here while you record — transcribed in "
        "25 s chunks as the meeting runs, and saved after every "
        "chunk, so a crash loses at most the chunk in flight.")
    live_hint.setObjectName("hint")
    live_hint.setWordWrap(True)
    live_column.addWidget(live_hint)
    # Active-model line (Nov 20 2026, Item 7): which Whisper
    # model is transcribing — it follows the configured model.
    live_column.addWidget(win.live_model_label)
    live_column.addWidget(win.nearlive_list, 1)
    # Manual near-live run for a recorded session (the old
    # Near-live tab's button): it fills this same list, so the
    # button lives with the list, not on the Session view.
    live_column.addWidget(win.nearlive_btn)
    live_column.addWidget(win.nearlive_progress)
    live_column.addWidget(win.nearlive_label)
    layout.addLayout(live_column, 1)
    return page
