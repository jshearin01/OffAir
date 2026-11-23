"""Session view — the one open session, and everything done to it.

Transcript (Original / Cleaned / Enterprise-cleaned toggle,
Re-clean, Polish, enterprise cleanup, click a segment to play
it), the waveform timeline, bookmarks, speaker suggestions +
the rename panel for this session, and the Process actions
(Transcribe, Diarize, Export & local LLM) with their progress.
Scrollable: the groups together need more than the ~640 px a
1280x800 window gives the view. When nothing is loaded, a real
empty state shows instead of a blank transcript — the actions
stay disabled until a session is opened from the Library.
"""
from __future__ import annotations

from app.ui.level_meter import QT_AVAILABLE

if QT_AVAILABLE:
    from PyQt6.QtWidgets import (
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )


def build_session_view(win) -> QWidget:
    """Build the Session page (a scroll area over the session work)."""
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)

    win.session_empty_label = QLabel(
        "No session loaded — record a meeting, or open one from "
        "the Library.")
    win.session_empty_label.setObjectName("hint")
    win.session_empty_label.setWordWrap(True)
    layout.addWidget(win.session_empty_label)
    layout.addWidget(win.loaded_session_label)
    # Dictation sessions say so plainly, with the spoken-side-only
    # wording (hidden for every other session).
    layout.addWidget(win.session_dictation_note_label)
    calendar_row = QHBoxLayout()
    calendar_row.addWidget(win.calendar_event_label, 1)
    calendar_row.addWidget(win.open_outlook_btn)
    layout.addLayout(calendar_row)

    # Session chooser: every action on this view works on the
    # session picked here (default: the newest). Choosing one loads
    # it — same path as opening from the Library.
    chooser_row = QHBoxLayout()
    chooser_row.setSpacing(8)
    chooser_label = QLabel("Session:")
    chooser_label.setObjectName("fieldLabel")
    chooser_row.addWidget(chooser_label)
    chooser_row.addWidget(win.session_combo, 1)
    layout.addLayout(chooser_row)

    # Source editor: correct the loaded session's recording source
    # + detail; both save to metadata.json on change.
    source_edit_row = QHBoxLayout()
    source_edit_row.setSpacing(8)
    source_edit_label = QLabel("Source:")
    source_edit_label.setObjectName("fieldLabel")
    source_edit_row.addWidget(source_edit_label)
    source_edit_row.addWidget(win.session_source_combo)
    source_edit_row.addWidget(win.session_source_detail_edit, 1)
    layout.addLayout(source_edit_row)

    # Transcript: click-to-play segments, Original/Cleaned toggle.
    transcript_title = QLabel("Transcript segments (click to play)")
    transcript_title.setObjectName("sectionTitle")
    layout.addWidget(transcript_title)
    view_row = QHBoxLayout()
    view_row.setSpacing(8)
    view_label = QLabel("Show:")
    view_label.setObjectName("fieldLabel")
    view_row.addWidget(view_label)
    view_row.addWidget(win.transcript_view_combo, 1)
    view_row.addWidget(win.reclean_btn)
    view_row.addWidget(win.polish_btn)
    view_row.addWidget(win.enterprise_btn)
    view_row.addWidget(win.enterprise_cancel_btn)
    layout.addLayout(view_row)
    waveform_title = QLabel("Waveform — click to hear that moment")
    waveform_title.setObjectName("fieldLabel")
    layout.addWidget(waveform_title)
    layout.addWidget(win.waveform_widget)
    layout.addWidget(win.segments_list)
    layout.addWidget(win.cleanup_label)

    # Bookmarks for the loaded session: click one to hear that
    # moment and highlight the segment; add one at the moment
    # currently playing (or the selected segment, or the start).
    bookmarks_title = QLabel("Bookmarks")
    bookmarks_title.setObjectName("sectionTitle")
    layout.addWidget(bookmarks_title)
    layout.addWidget(win.bookmarks_list)
    bookmarks_row = QHBoxLayout()
    bookmarks_row.addWidget(win.add_bookmark_btn)
    layout.addLayout(bookmarks_row)
    bookmarks_hint = QLabel(
        "Add bookmark marks the moment now playing, the selected "
        "segment, or the start of the session when nothing is "
        "playing. A bookmark whose audio was deleted by retention "
        "says so and cannot be played.")
    bookmarks_hint.setObjectName("hint")
    bookmarks_hint.setWordWrap(True)
    layout.addWidget(bookmarks_hint)

    # Speaker suggestions for this session + the rename panel.
    sugg_title = QLabel("Suggestions for this session")
    sugg_title.setObjectName("sectionTitle")
    layout.addWidget(sugg_title)
    layout.addWidget(win.suggest_btn)
    layout.addWidget(win.suggestions_list)
    sugg_row = QHBoxLayout()
    sugg_row.addWidget(win.play_suggestion_btn)
    sugg_row.addWidget(win.confirm_btn)
    sugg_row.addWidget(win.correct_btn)
    layout.addLayout(sugg_row)
    sugg_hint = QLabel(
        "Not sure who a suggestion is? Select it and Play sample "
        "to hear their voice before confirming. Correct uses the "
        "name in the rename panel below.")
    sugg_hint.setObjectName("hint")
    sugg_hint.setWordWrap(True)
    layout.addWidget(sugg_hint)
    rename_title = QLabel("Rename a speaker or label")
    rename_title.setObjectName("sectionTitle")
    layout.addWidget(rename_title)
    layout.addWidget(win.rename_label_combo)
    layout.addWidget(win.rename_name_edit)
    layout.addWidget(win.rename_btn)
    rename_hint = QLabel(
        "Pick a speaker above, or a label from the list; type the "
        "new name and press Rename speaker. Renaming a learned "
        "speaker updates every past session. The name field is "
        "also used to Correct a suggestion above.")
    rename_hint.setObjectName("hint")
    rename_hint.setWordWrap(True)
    layout.addWidget(rename_hint)

    transcribe_group = QGroupBox("Transcription")
    transcribe_layout = QVBoxLayout(transcribe_group)
    model_label = QLabel("Model (CPU, int8):")
    model_label.setObjectName("fieldLabel")
    transcribe_layout.addWidget(model_label)
    transcribe_layout.addWidget(win.model_combo)
    transcribe_layout.addWidget(win.transcribe_btn)
    transcribe_layout.addWidget(win.cancel_transcribe_btn)
    transcribe_layout.addWidget(win.transcribe_progress)
    transcribe_layout.addWidget(win.transcribe_label)
    layout.addWidget(transcribe_group)

    diarize_group = QGroupBox("Diarization")
    diarize_layout = QVBoxLayout(diarize_group)
    diarize_mode_label = QLabel("Mode:")
    diarize_mode_label.setObjectName("fieldLabel")
    diarize_layout.addWidget(diarize_mode_label)
    diarize_layout.addWidget(win.diarize_mode_combo)
    diarize_layout.addWidget(win.diarize_warning_label)
    diarize_layout.addWidget(win.sherpa_status_label)
    diarize_layout.addWidget(win.hf_token_edit)
    speaker_hint_row = QHBoxLayout()
    speaker_hint_row.addWidget(win.min_speakers_edit)
    speaker_hint_row.addWidget(win.max_speakers_edit)
    diarize_layout.addLayout(speaker_hint_row)
    diarize_layout.addWidget(win.diarize_btn)
    diarize_layout.addWidget(win.diarize_progress)
    diarize_layout.addWidget(win.diarize_label)
    layout.addWidget(diarize_group)

    export_group = QGroupBox("Export & local LLM")
    export_layout = QVBoxLayout(export_group)
    export_layout.addWidget(win.export_btn)
    export_layout.addWidget(win.export_label)
    export_layout.addWidget(win.summarize_btn)
    export_layout.addWidget(win.summarize_label)
    layout.addWidget(export_group)
    layout.addStretch(1)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    return scroll
