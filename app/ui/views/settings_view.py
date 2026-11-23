"""Settings view — the Settings dialog's form, as a view.

The form itself is built by :class:`app.ui.settings_dialog.
SettingsDialog` (``dialog.form``); this page wraps it in a scroll
area with Save / Discard buttons instead of the dialog's OK /
Cancel. The window rebuilds the form from the live config every
time the view is shown (``MainWindow._refresh_settings_view``),
so the fields never go stale after an output-folder change. The
dialog itself is unchanged and still exists for standalone use
(and its tests).

Above the form sit the "Recording devices" section (Nov 20 2026)
and the "Pre-roll & enterprise cleanup" section (Nov 21 2026): the
live microphone / system-audio pickers and the opt-in settings
whose config fields the Settings dialog does not edit. They share
the page's single scroll area with the form, in a holder whose
only child the form rebuilds replace — so the device widgets and
advanced controls survive rebuilds untouched. Device choice is
NOT persisted (as before): the combos are read at record time,
and the config's device-index fields are only what a session's
metadata records.
"""
from __future__ import annotations

from app.config import PREROLL_CONSENT_NOTE
from app.ui.level_meter import QT_AVAILABLE

if QT_AVAILABLE:
    from PyQt6.QtCore import QSize
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    class _HintLabel(QLabel):
        """Word-wrapped hint text that refuses to be clipped.

        A plain word-wrapped QLabel reports a one-line minimum
        height, so when the Settings page (taller than 800 px)
        squeezes, the layout clips hint text to a single line
        instead of shrinking the scroll area. This label reports
        its wrapped height at the current width as its minimum,
        so the scroll area shrinks instead. The width floor keeps
        the pre-layout minimum sane.
        """

        def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt name
            hint = super().minimumSizeHint()
            if self.wordWrap():
                width = max(self.width(), 600)
                hint.setHeight(max(hint.height(),
                                   self.heightForWidth(width)))
            return hint


def build_settings_view(win) -> QWidget:
    """Build the Settings page; the form is filled in by ``win``."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)
    # Recording devices (live widgets, not form fields — see the
    # module docstring): the combos are MainWindow-owned and
    # populated by refresh_devices(); the form rebuilds below
    # never touch them.
    devices_group = QGroupBox("Recording devices")
    devices_layout = QFormLayout(devices_group)
    devices_layout.addRow("Microphone:", win.mic_combo)
    devices_layout.addRow("System audio (loopback):",
                          win.loopback_combo)
    devices_hint = QLabel(
        "Defaults apply when nothing is picked. Devices are read "
        "when a recording starts; the choice is not saved.")
    devices_hint.setObjectName("hint")
    devices_hint.setWordWrap(True)
    devices_layout.addRow(devices_hint)
    devices_layout.addRow("", win.refresh_btn)
    # (devices_group joins the scroll content below, with the
    # advanced group and the dialog form.)
    # Pre-roll & enterprise cleanup (Nov 21 2026): the two opt-in
    # settings whose config fields the Settings dialog does not
    # edit. They share the page's scroll area with the form, in a
    # holder whose only child the rebuilds replace, so form
    # rebuilds never touch them; the window overlays their values
    # onto a saved config (see MainWindow._on_settings_save) and
    # syncs them from the live config on every form rebuild. The
    # consent note below is the exact text from app.config, shown
    # verbatim.
    advanced_group = QGroupBox("Pre-roll && enterprise cleanup")
    advanced_layout = QVBoxLayout(advanced_group)
    win.preroll_check = QCheckBox(
        "Pre-roll: capture audio before Record is pressed")
    win.preroll_check.setToolTip(
        "When on, OffAir keeps the last few seconds of audio in "
        "memory before you press Record, and adds them to the "
        "start of the recording.")
    win.preroll_check.toggled.connect(win._on_preroll_toggled)
    advanced_layout.addWidget(win.preroll_check)
    preroll_row = QHBoxLayout()
    preroll_label = QLabel("Pre-roll seconds (10–120):")
    preroll_label.setObjectName("fieldLabel")
    preroll_row.addWidget(preroll_label)
    win.preroll_seconds_spin = QSpinBox()
    win.preroll_seconds_spin.setRange(10, 120)
    win.preroll_seconds_spin.setSingleStep(5)
    win.preroll_seconds_spin.setValue(30)
    preroll_row.addWidget(win.preroll_seconds_spin)
    preroll_row.addStretch(1)
    advanced_layout.addLayout(preroll_row)
    win.preroll_consent_label = _HintLabel(PREROLL_CONSENT_NOTE)
    win.preroll_consent_label.setObjectName("hint")
    win.preroll_consent_label.setWordWrap(True)
    advanced_layout.addWidget(win.preroll_consent_label)
    win.enterprise_check = QCheckBox(
        "Enterprise cleanup: clean transcripts with the Claude "
        "Code CLI (transcript text only leaves this machine)")
    win.enterprise_check.setToolTip(
        "Off by default. When on, the Session view can send "
        "transcript TEXT ONLY to the Claude Code CLI (`claude "
        "-p`) for cleanup. Audio never leaves this machine.")
    advanced_layout.addWidget(win.enterprise_check)
    enterprise_hint = _HintLabel(
        "Enterprise cleanup needs the Claude Code CLI installed "
        "and on PATH. The verbatim transcript is never changed; "
        "the result is a separate, clearly labelled file.")
    enterprise_hint.setObjectName("hint")
    enterprise_hint.setWordWrap(True)
    advanced_layout.addWidget(enterprise_hint)
    win.tinyfix_check = QCheckBox(
        "tinyfix: fix real words in the wrong place "
        "(“form” → “from”, “their” → “there”)")
    win.tinyfix_check.setToolTip(
        "Off by default — opt in to run it. A small on-device "
        "model that fixes the wrong-word errors Whisper makes, "
        "as a stage of transcript cleanup — after the rules and "
        "the names dictionary, which always win. English only; "
        "the model downloads once (~3.5 MB) on first use. The "
        "verbatim transcript is never changed.")
    win.tinyfix_check.toggled.connect(win._on_tinyfix_toggled)
    win.tinyfix_check.clicked.connect(win._on_tinyfix_clicked)
    advanced_layout.addWidget(win.tinyfix_check)
    tinyfix_row = QHBoxLayout()
    tinyfix_label = QLabel("tinyfix mode:")
    tinyfix_label.setObjectName("fieldLabel")
    tinyfix_row.addWidget(tinyfix_label)
    win.tinyfix_mode_combo = QComboBox()
    win.tinyfix_mode_combo.addItem("balanced", "balanced")
    win.tinyfix_mode_combo.addItem("careful", "careful")
    win.tinyfix_mode_combo.setToolTip(
        "Balanced fixes more wrong words; careful makes fewer "
        "wrong changes. Careful is the default.")
    tinyfix_row.addWidget(win.tinyfix_mode_combo)
    tinyfix_row.addStretch(1)
    advanced_layout.addLayout(tinyfix_row)
    tinyfix_hint = _HintLabel(
        "Careful is the default: on the fixture transcript it made "
        "no wrong changes, where balanced made one. Balanced fixes "
        "a few more wrong words; switch to it if a transcript has "
        "many obvious mix-ups.")
    tinyfix_hint.setObjectName("hint")
    tinyfix_hint.setWordWrap(True)
    advanced_layout.addWidget(tinyfix_hint)
    win.settings_disk_label = QLabel("")
    win.settings_disk_label.setObjectName("hint")
    win.settings_disk_label.setWordWrap(True)
    advanced_layout.addWidget(win.settings_disk_label)
    # One scroll area for the whole page (Nov 21 2026): devices +
    # advanced + the dialog form together are taller than 800 px,
    # and a squeezing layout clipped the wrapped hint labels to a
    # single line instead of shrinking the form's scroll area.
    # The form rebuild replaces only the holder's child, so the
    # device / advanced widgets survive rebuilds untouched.
    scroll_content = QWidget()
    scroll_layout = QVBoxLayout(scroll_content)
    scroll_layout.setContentsMargins(0, 0, 0, 0)
    scroll_layout.setSpacing(12)
    scroll_layout.addWidget(devices_group)
    scroll_layout.addWidget(advanced_group)
    win._settings_form_holder = QWidget()
    holder_layout = QVBoxLayout(win._settings_form_holder)
    holder_layout.setContentsMargins(0, 0, 0, 0)
    holder_layout.setSpacing(0)
    scroll_layout.addWidget(win._settings_form_holder)
    win._settings_scroll = QScrollArea()
    win._settings_scroll.setWidgetResizable(True)
    win._settings_scroll.setWidget(scroll_content)
    layout.addWidget(win._settings_scroll, 1)
    buttons_row = QHBoxLayout()
    buttons_row.setSpacing(8)
    win.settings_save_btn = QPushButton("Save settings")
    win.settings_save_btn.setObjectName("primaryButton")
    win.settings_save_btn.setMinimumHeight(40)
    win.settings_save_btn.setToolTip(
        "Save these settings — output folder, model, diarization, "
        "auto-detect, retention, names dictionary and the rest.")
    win.settings_save_btn.clicked.connect(win._on_settings_save)
    buttons_row.addWidget(win.settings_save_btn, 1)
    win.settings_cancel_btn = QPushButton("Discard changes")
    win.settings_cancel_btn.setMinimumHeight(40)
    win.settings_cancel_btn.setToolTip(
        "Throw away unsaved changes and reload the saved settings.")
    win.settings_cancel_btn.clicked.connect(win._refresh_settings_view)
    buttons_row.addWidget(win.settings_cancel_btn, 1)
    layout.addLayout(buttons_row)
    return page
