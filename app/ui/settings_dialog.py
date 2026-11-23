"""Settings panel (PRD Phase 5).

A dialog for the settings persisted by :mod:`app.config`
(output folder, Whisper model, diarization mode, speaker
embedding backend, near-live, theme, local LLM, sherpa
diarization settings, search mode, default recording source).
The Hugging Face token is NEVER entered or stored here: the
panel shows only a checkbox stating whether a token is
configured (via the ``HF_TOKEN`` environment variable or the
token field on the main window, both in-memory only).

The form lives in a scroll area so the dialog fits a 1280x800
window even with the sherpa section added (Phase 6). Since the
Nov 20 2026 redesign the form is grouped into labelled sections
(Recording, Transcription, Speakers & diarization, Automation,
Storage & search, Local AI, Names, Appearance) with one
consistent field pattern — label above the control, a one-line
hint below only where the label does not explain itself — so
nothing clips or overlaps at 1280x800 in either theme.

Importing this module never requires PyQt6; constructing the
dialog raises a clear RuntimeError when Qt is absent.
"""
from __future__ import annotations

from pathlib import Path

from app.config import (
    AppConfig,
    AUDIO_RETENTION_DEFAULT_HOURS,
    AUDIO_RETENTION_OPTIONS,
    DIARIZATION_DEFAULT_MODE,
    EMBEDDING_DEFAULT_BACKEND,
    SHERPA_MODEL_DIR,
    WHISPER_DEFAULT_MODEL,
    WHISPER_MODELS,
    load_settings,
    save_settings,
)
from app.diarization import DIARIZATION_MODES
from app.platform_info import auto_detect_status, outlook_status
from app.storage.source import SOURCES, SOURCE_LABELS
from app.transcription.names_dictionary import (
    format_dictionary_text,
    load_names_dictionary,
    parse_dictionary_text,
    save_names_dictionary,
)
from app.ui.level_meter import QT_AVAILABLE

if QT_AVAILABLE:
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )


def _field_label(text: str) -> "QLabel":
    """A field heading: small, semibold, secondary colour."""
    label = QLabel(text)
    label.setObjectName("fieldLabel")
    return label


def _hint(text: str) -> "QLabel":
    """A one-line hint under a field (word-wrapped, muted)."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("hint")
    return label


def _section(title: str) -> tuple["QGroupBox", "QVBoxLayout"]:
    """A labelled form section with uniform margins/spacing.

    One consistent field pattern inside: label above the control,
    a hint below only where the label does not explain itself.
    """
    group = QGroupBox(title)
    layout = QVBoxLayout(group)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)
    return group, layout


def _retention_data(hours: float | None) -> str:
    """Combo data for a retention age: "never", or "%g" hours."""
    return "never" if hours is None else "%g" % float(hours)


def _retention_hours(data) -> float | None:
    """Retention age back from combo data ("never" -> None).

    A missing selection (None) falls back to the default — only an
    explicit "Never" choice disables pruning.
    """
    if data is None:
        return AUDIO_RETENTION_DEFAULT_HOURS
    if str(data) == "never":
        return None
    return float(data)


class SettingsDialog(QDialog if QT_AVAILABLE else object):  # type: ignore[misc]
    """Edit a copy of the settings; Apply writes them to disk."""

    def __init__(self, config: AppConfig | None = None,
                 parent=None) -> None:
        if not QT_AVAILABLE:
            raise RuntimeError(
                "PyQt6 is not installed — install it with "
                "`pip install -r requirements.txt` to use the GUI.")
        super().__init__(parent)
        self.setWindowTitle("OffAir — Settings")
        self.resize(720, 760)
        self.setMinimumSize(560, 480)
        self._config = config or load_settings()
        cfg = self._config

        # The form is scrollable (Phase 6): with the sherpa section
        # it is taller than a 1280x800 window's dialog area.
        form = QWidget()
        # The form is a separate widget so the main window can embed
        # the same settings as a view (Nov 19 2026 redesign): the
        # dialog wraps it in a scroll area + OK/Cancel, the Settings
        # view takes the form out of that scroll area and wraps it
        # in its own, with Save / Discard buttons instead.
        self.form = form
        layout = QVBoxLayout(form)
        # 16 px like every other window: the old 4 px left the form
        # hard against the scroll area's edge.
        layout.setContentsMargins(16, 16, 16, 24)
        layout.setSpacing(12)

        title = QLabel("Settings")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        subtitle = QLabel(
            "Changes are saved when you confirm them — OK in the "
            "dialog, Save settings in the Settings view. Your "
            "Hugging Face token is never stored here.")
        subtitle.setObjectName("hint")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # --- Recording ---
        recording_group, recording = _section("Recording")
        layout.addWidget(recording_group)
        recording.addWidget(_field_label("Output folder (recordings):"))
        self.output_edit = QLineEdit(str(cfg.output_root))
        recording.addWidget(self.output_edit)
        browse_btn = QPushButton("Choose output folder…")
        browse_btn.clicked.connect(self._choose_output_folder)
        recording.addWidget(browse_btn)

        # System-audio capture mode (Nov 8 2026, per-app capture):
        # "auto" tries Teams-only capture and falls back to
        # system-wide loopback; "per-app" fails loudly instead;
        # "loopback" is always system-wide. Linux ignores this —
        # it always records the PulseAudio/PipeWire monitor source.
        recording.addWidget(_field_label("System audio capture (Windows):"))
        self.capture_mode_combo = QComboBox()
        self.capture_mode_combo.addItem(
            "Auto — Teams-only when possible, else system-wide "
            "(recommended)", "auto")
        self.capture_mode_combo.addItem(
            "Teams only — per-app capture, fail if unavailable",
            "per-app")
        self.capture_mode_combo.addItem(
            "System-wide loopback — always (pre-Nov 8 behaviour)",
            "loopback")
        self.capture_mode_combo.setCurrentIndex(
            self.capture_mode_combo.findData(cfg.capture_system_mode))
        recording.addWidget(self.capture_mode_combo)
        recording.addWidget(_hint(
            "Per-app capture records only the target app's audio "
            "(WASAPI process loopback, Windows 10 build 20348+). "
            "Auto falls back to system-wide loopback with a warning "
            "when per-app is unavailable; Teams only fails loudly "
            "instead, so a bad setup is never a silent recording."))

        # Default recording source (Nov 20 2026, D4): the Record
        # view's Source dropdown is gone — this is the persisted
        # default instead. "Auto" detects the source from the
        # picked app at record time; an explicit pick wins over
        # detection (the only way to name a source no process
        # identifies, e.g. a Claude Code session).
        recording.addWidget(_field_label("Default recording source:"))
        self.default_source_combo = QComboBox()
        self.default_source_combo.addItem("Auto (detect from app)", None)
        for value in SOURCES:
            self.default_source_combo.addItem(SOURCE_LABELS[value], value)
        self.default_source_combo.setCurrentIndex(
            self.default_source_combo.findData(cfg.default_source))
        self.default_source_combo.setToolTip(
            "What a recording is of. Auto detects it from the "
            "app picked above; pick one explicitly when detection "
            "cannot know — e.g. “Claude Code” for a working session "
            "with the agent, or “Google Meet” for a Meet call in a "
            "browser. The pick is stored with the session and can "
            "be corrected later from the Session view.")
        recording.addWidget(self.default_source_combo)

        # Auto-process (Nov 8 2026): transcribe → diarize → identify
        # speakers automatically after each recording stops. The
        # recording is saved first, so a failed auto-step can never
        # lose it; every step stays re-runnable by hand.
        self.auto_process_check = QCheckBox(
            "Automatically transcribe, diarize and identify speakers "
            "after each recording (recommended)")
        self.auto_process_check.setChecked(cfg.auto_process_recordings)
        self.auto_process_check.setToolTip(
            "Runs Whisper (base), sherpa diarization and speaker "
            "identification in the background when a recording stops. "
            "First use downloads the models (Whisper base ~142 MB, "
            "sherpa ~33 MB). Untick to go back to the manual buttons "
            "on the Session view.")
        recording.addWidget(self.auto_process_check)
        recording.addWidget(_hint(
            "Auto-processing runs in the background — the window stays "
            "responsive — and a failed step is reported loudly instead "
            "of failing silently. Base takes ~4 min per 30 min of "
            "meeting on a 4-core laptop."))

        # --- Transcription ---
        transcription_group, transcription = _section("Transcription")
        layout.addWidget(transcription_group)
        transcription.addWidget(_field_label("Transcription model (CPU, int8):"))
        self.model_combo = QComboBox()
        for name, info in WHISPER_MODELS.items():
            # The current build's default is marked, so the user
            # can tell at a glance which item is the default.
            suffix = " (default)" if name == WHISPER_DEFAULT_MODEL else ""
            self.model_combo.addItem(
                f"{name} — {info['speed']} ({info['size_mb']} MB)"
                f"{suffix}", name)
        # Initialise from the EFFECTIVE model (Nov 20 2026, Item 8):
        # a stored "small" from 1.0.2 that was never explicitly
        # picked shows as the current default, base — the stored
        # value alone is not the model in use.
        self.model_combo.setCurrentIndex(
            self.model_combo.findData(cfg.effective_whisper_model))
        # Explicit-choice tracking (Item 8): only a user-driven
        # pick sets the flag. `activated` fires only when the user
        # picks an item (mouse or keyboard); programmatic
        # setCurrentIndex/setCurrentText — construction, refresh,
        # a plain save — do not emit it, so they never set it.
        # _model_explicit is the running state: the incoming
        # flag, or True once the user picks in this dialog.
        self._model_explicit = bool(cfg.whisper_model_explicit)
        self._model_touched = False
        self.model_combo.activated.connect(self._on_model_activated)
        transcription.addWidget(self.model_combo)
        # Hint line under the combo, stating the state at a glance
        # (Item 8): default vs an explicitly picked model.
        self.model_hint = _hint("")
        self._refresh_model_hint()
        transcription.addWidget(self.model_hint)

        self.nearlive_check = QCheckBox(
            "Near-live chunked transcription enabled")
        self.nearlive_check.setChecked(cfg.near_live_enabled)
        transcription.addWidget(self.nearlive_check)
        transcription.addWidget(_field_label(
            "Near-live chunk length (seconds, 5–60):"))
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(5, 60)
        self.chunk_spin.setValue(int(cfg.nearlive_chunk_seconds))
        transcription.addWidget(self.chunk_spin)

        # --- Speakers & diarization ---
        speakers_group, speakers = _section("Speakers & diarization")
        layout.addWidget(speakers_group)
        # Diarization mode — from the DIARIZATION_MODES registry
        # (simple / full / sherpa), never hard-coded.
        speakers.addWidget(_field_label("Diarization mode:"))
        self.diarize_combo = QComboBox()
        for mode_key, mode_info in DIARIZATION_MODES.items():
            self.diarize_combo.addItem(mode_info["label"], mode_key)
        self.diarize_combo.setCurrentIndex(
            self.diarize_combo.findData(cfg.diarization_mode))
        speakers.addWidget(self.diarize_combo)

        # Sherpa diarization settings (Phase 6 / Track 1): model
        # folder, expected-speaker hint (0 = auto, threshold
        # clustering discovers the count) and the clustering
        # threshold for auto mode. Defaults come from app.config.
        sherpa_group = QGroupBox("Sherpa diarization (no token, CPU)")
        sherpa_layout = QFormLayout(sherpa_group)
        self.sherpa_model_dir_edit = QLineEdit(
            str(cfg.sherpa_model_dir) if cfg.sherpa_model_dir is not None
            else str(SHERPA_MODEL_DIR))
        self.sherpa_model_dir_edit.setPlaceholderText(str(SHERPA_MODEL_DIR))
        sherpa_dir_btn = QPushButton("Choose…")
        sherpa_dir_btn.clicked.connect(self._choose_sherpa_model_dir)
        sherpa_layout.addRow("Model folder:", self.sherpa_model_dir_edit)
        sherpa_layout.addRow("", sherpa_dir_btn)
        self.sherpa_expected_spin = QSpinBox()
        self.sherpa_expected_spin.setRange(0, 10)
        self.sherpa_expected_spin.setSpecialValueText("0 — auto")
        self.sherpa_expected_spin.setValue(int(cfg.sherpa_expected_speakers))
        self.sherpa_expected_spin.setToolTip(
            "0 = auto: threshold clustering discovers the speaker "
            "count. 1–10 pins the cluster count instead.")
        sherpa_layout.addRow("Expected speakers (0 = auto):",
                             self.sherpa_expected_spin)
        self.sherpa_threshold_spin = QDoubleSpinBox()
        self.sherpa_threshold_spin.setRange(0.05, 0.95)
        self.sherpa_threshold_spin.setSingleStep(0.05)
        self.sherpa_threshold_spin.setDecimals(2)
        self.sherpa_threshold_spin.setValue(
            float(cfg.sherpa_clustering_threshold))
        self.sherpa_threshold_spin.setToolTip(
            "Cosine-similarity merge threshold for auto mode. "
            "0.5 is sherpa-onnx's default (see app/diarization/"
            "sherpa.py for how it was measured).")
        sherpa_layout.addRow("Clustering threshold:",
                             self.sherpa_threshold_spin)
        sherpa_layout.addRow(_hint(
            "Models (~33 MB) download once on first use from public "
            "GitHub releases — no account, no token — then work "
            "offline. Install sherpa-onnx with: pip install sherpa-onnx"))
        speakers.addWidget(sherpa_group)

        speakers.addWidget(_field_label("Speaker embedding backend:"))
        self.embedding_combo = QComboBox()
        self.embedding_combo.addItem(
            "Auto — best available (SpeechBrain ECAPA, else ONNX "
            "ECAPA, else fallback)", "auto")
        self.embedding_combo.addItem(
            "SpeechBrain ECAPA (needs torch + speechbrain)",
            "speechbrain-ecapa")
        self.embedding_combo.addItem(
            "ONNX ECAPA (no torch — the verified default here)",
            "onnx-ecapa")
        self.embedding_combo.addItem(
            "Sherpa ERes2Net (needs sherpa-onnx + its models)",
            "sherpa-eres2net")
        self.embedding_combo.addItem(
            "Pyannote (opt-in, needs HF token)", "pyannote")
        self.embedding_combo.addItem(
            "Fallback spectral (test-only, not a voice fingerprint)",
            "fallback-spectral-v1")
        self.embedding_combo.setCurrentIndex(
            self.embedding_combo.findData(cfg.embedding_backend))
        speakers.addWidget(self.embedding_combo)

        self.hf_configured_check = QCheckBox(
            "Hugging Face token is configured (in-memory only — "
            "OffAir never stores a token)")
        self.hf_configured_check.setChecked(cfg.hf_token_configured)
        self.hf_configured_check.setToolTip(
            "Tick this only to record that you have set HF_TOKEN in "
            "your environment. The token itself is never written to "
            "disk by OffAir.")
        speakers.addWidget(self.hf_configured_check)

        # --- Automation ---
        automation_group, automation = _section("Automation")
        layout.addWidget(automation_group)
        # Auto-detect + auto-record (Nov 8 2026): start recording
        # when a Teams meeting starts, stop when it ends. OPT-IN,
        # default OFF — the first time it is enabled the user gets
        # an explicit consent dialog (see _on_auto_detect_toggled).
        # Windows only: the watcher needs WASAPI audio sessions, so
        # on Linux/macOS the checkbox is disabled with the reason
        # instead of pretending the feature works.
        self.auto_detect_check = QCheckBox(
            "Start recording automatically when a Teams meeting starts")
        self.auto_detect_check.setChecked(cfg.auto_detect_teams)
        self.auto_detect_check.setToolTip(
            "Watches Teams' audio activity: recording starts when a "
            "meeting starts (session Active with sustained audio) and "
            "stops when it ends (the Teams session going inactive, or "
            "Teams exiting — silence during a meeting never stops it).")
        self._auto_detect_was_enabled = bool(cfg.auto_detect_teams)
        self._auto_detect_consent_shown = False
        self.auto_detect_check.toggled.connect(
            self._on_auto_detect_toggled)
        automation.addWidget(self.auto_detect_check)
        auto_detect_available, auto_detect_reason = auto_detect_status()
        if not auto_detect_available:
            self.auto_detect_check.setEnabled(False)
        automation.addWidget(_hint(
            "OffAir does NOT trigger Teams' own recording banner — "
            "tell everyone in the meeting that you are recording. "
            "Recording laws vary by jurisdiction. "
            + ("" if auto_detect_available
               else f"Not available here: {auto_detect_reason}")))

        # Outlook calendar linking (Nov 10 2026, Feature 3): match
        # each recording to its Outlook calendar event and store
        # the link in the session's metadata.json. Default ON where
        # Outlook is available. READ-ONLY — OffAir never writes
        # anything into Outlook (see app/calendar/outlook.py and
        # docs/OUTLOOK_LINKING.md). Windows + Outlook only: on
        # other machines the checkbox is disabled with the reason
        # instead of pretending the feature works.
        self.outlook_link_check = QCheckBox(
            "Link recordings to their Outlook calendar event")
        self.outlook_link_check.setChecked(cfg.outlook_link_enabled)
        self.outlook_link_check.setToolTip(
            "When a recording stops (and when an old session loads), "
            "OffAir looks up the Outlook event overlapping the "
            "recording and stores the link in the session's "
            "metadata.json, so the main window can say which meeting "
            "a recording was. Read-only: nothing is ever written "
            "into Outlook.")
        automation.addWidget(self.outlook_link_check)
        outlook_available, outlook_reason = outlook_status()
        if not outlook_available:
            self.outlook_link_check.setEnabled(False)
        automation.addWidget(_hint(
            "The linked event shows under the loaded session, with "
            "an “Open in Outlook” button. "
            + ("" if outlook_available
               else f"Not available here: {outlook_reason}")))

        # --- Storage & search ---
        storage_group, storage = _section("Storage & search")
        layout.addWidget(storage_group)
        # Audio retention (Nov 8 2026): auto-delete old recording
        # audio; transcripts are always kept. Recordings are WAVs
        # (mic/system/combined), ~1 GB per hour of meeting.
        storage.addWidget(_field_label(
            "Delete recording audio older than:"))
        self.retention_combo = QComboBox()
        for label, hours in AUDIO_RETENTION_OPTIONS:
            self.retention_combo.addItem(label, _retention_data(hours))
        self.retention_combo.setCurrentIndex(
            self.retention_combo.findData(
                _retention_data(cfg.audio_retention_hours)))
        self.retention_combo.setToolTip(
            "Only sessions with a transcript are pruned, and never "
            "a recording in progress.")
        storage.addWidget(self.retention_combo)
        storage.addWidget(_hint(
            "Audio older than this is deleted automatically; "
            "transcripts are always kept."))

        storage.addWidget(_field_label("Recording Browser search mode:"))
        self.search_combo = QComboBox()
        self.search_combo.addItem(
            "Text — exact words (fast, always works offline)", "text")
        self.search_combo.addItem(
            "Semantic — meaning (downloads the MiniLM model once, "
            "~22 MB)", "semantic")
        self.search_combo.addItem(
            "Hybrid — exact matches first, then meaning (recommended)",
            "hybrid")
        self.search_combo.setCurrentIndex(
            self.search_combo.findData(cfg.search_mode))
        storage.addWidget(self.search_combo)

        # --- Local AI ---
        ai_group, ai = _section("Local AI")
        layout.addWidget(ai_group)
        self.llm_check = QCheckBox(
            "Local LLM summaries enabled (Ollama on localhost only — "
            "never a cloud LLM)")
        self.llm_check.setChecked(cfg.ollama_enabled)
        ai.addWidget(self.llm_check)
        ai.addWidget(_field_label("Ollama model:"))
        self.ollama_model_edit = QLineEdit(cfg.ollama_model)
        ai.addWidget(self.ollama_model_edit)
        ai.addWidget(_field_label("Ollama URL (localhost only):"))
        self.ollama_url_edit = QLineEdit(cfg.ollama_base_url)
        ai.addWidget(self.ollama_url_edit)

        # Transcript AI polish (Nov 10 2026): when ticked, the
        # auto-process chain also runs the local-LLM polish pass
        # after cleaning (transcript-polished.md, clearly marked as
        # an AI rewrite — the verbatim transcript stays the record).
        # OFF by default. The Session view's Polish button runs it
        # on demand regardless of this box.
        self.polish_check = QCheckBox(
            "AI-polish transcripts with the local LLM after "
            "auto-processing (Ollama on localhost only)")
        self.polish_check.setChecked(cfg.transcript_polish_enabled)
        self.polish_check.setToolTip(
            "Writes transcript-polished.md, an AI rewrite of the "
            "cleaned transcript. The verbatim transcript is never "
            "touched. Needs Ollama running; a failed polish is "
            "reported and the rule-cleaned version stays in place.")
        ai.addWidget(self.polish_check)

        # --- Names ---
        names_group, names = _section("Names")
        layout.addWidget(names_group)
        # Names dictionary (Nov 10 2026): names, products and jargon
        # Whisper should hear, plus known mishearings to correct.
        # One entry per line — a bare line is a name, "wrong ->
        # Correct" is a correction. Stored as plain JSON in
        # ~/.offair/names_dictionary.json. Renaming a speaker adds
        # their name here automatically.
        names.addWidget(_field_label("Names dictionary:"))
        self.names_edit = QPlainTextEdit()
        self.names_edit.setPlainText(
            format_dictionary_text(load_names_dictionary()))
        self.names_edit.setPlaceholderText(
            "One entry per line, e.g.:\nJohn\nPriya\n"
            "Meat Trace -> MeetTrace")
        self.names_edit.setToolTip(
            "Names Whisper should bias toward at decode time, and "
            "mishearings to correct afterwards. A bare line is a "
            "name; \"wrong -> Correct\" maps a mishearing to its "
            "correct spelling (whole words only).")
        names.addWidget(self.names_edit)
        names.addWidget(_hint(
            "Names bias Whisper at decode time; corrections are "
            "applied to the cleaned transcript (whole words only — "
            "“Marcus” is never rewritten inside another word). The "
            "verbatim transcript is never changed."))

        # --- Appearance ---
        appearance_group, appearance = _section("Appearance")
        layout.addWidget(appearance_group)
        appearance.addWidget(_field_label("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Light — clean and airy (default)",
                                 "light")
        self.theme_combo.addItem("Dark — easy on the eyes", "dark")
        self.theme_combo.setCurrentIndex(
            self.theme_combo.findData(cfg.theme))
        appearance.addWidget(self.theme_combo)

        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form)
        self._form_scroll = scroll
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.addWidget(scroll)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setObjectName("primaryButton")
        outer.addWidget(buttons)

    def _on_model_activated(self, _index: int) -> None:
        """A user-driven pick in the model combo (Nov 20 2026, Item 8).

        `activated` fires only on a user pick — never on a
        programmatic set — so this is the one place the explicit
        flag is ever set True. The hint updates live to show the
        choice will be kept across updates.
        """
        self._model_touched = True
        self._model_explicit = True
        self._refresh_model_hint()

    def _refresh_model_hint(self) -> None:
        """Hint under the model combo: default vs explicit (Item 8)."""
        if self._model_explicit or self._model_touched:
            model = self.model_combo.currentData() or ""
            self.model_hint.setText(
                f"You picked {model} — this choice is kept across "
                "updates.")
        else:
            self.model_hint.setText(
                f"Using the default model ({WHISPER_DEFAULT_MODEL}). "
                "Pick a model to keep it across updates.")

    def _on_auto_detect_toggled(self, checked: bool) -> None:
        """First-enable consent for auto-detect (opt-in, default off).

        The first time the box is ticked in a session where the
        setting was off, ask explicitly — auto-recording is a
        privacy decision, and OffAir does not trigger Teams' own
        recording banner. Declining leaves the box unticked.
        """
        if (not checked or self._auto_detect_was_enabled
                or self._auto_detect_consent_shown):
            return
        self._auto_detect_consent_shown = True
        answer = QMessageBox.question(
            self, "Record meetings automatically?",
            "OffAir will start recording automatically when a Teams "
            "meeting starts, and stop when it ends.\n\n"
            "OffAir does NOT trigger Teams' own recording banner — "
            "participants will not see that they are being recorded "
            "unless you tell them. Only enable this if you will tell "
            "everyone in the meeting that you are recording.\n\n"
            "Enable auto-detect?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            # Block the signal: unchecking must not re-enter this
            # handler (the consent was already shown).
            self.auto_detect_check.blockSignals(True)
            try:
                self.auto_detect_check.setChecked(False)
            finally:
                self.auto_detect_check.blockSignals(False)

    def _choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.output_edit.text())
        if folder:
            self.output_edit.setText(folder)

    def _choose_sherpa_model_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose sherpa model folder",
            self.sherpa_model_dir_edit.text())
        if folder:
            self.sherpa_model_dir_edit.setText(folder)

    def config_from_dialog(self) -> AppConfig:
        """Build an AppConfig from the dialog fields (validates)."""
        sherpa_dir_text = self.sherpa_model_dir_edit.text().strip()
        cfg = AppConfig(
            output_root=Path(self.output_edit.text().strip()
                             or str(self._config.output_root)),
            whisper_model=self.model_combo.currentData()
            or self._config.effective_whisper_model,
            # Untouched save: the incoming flag (False unless the
            # user explicitly picked before) — the combo already
            # shows the effective model, so saving writes it with
            # explicit False and completes a pre-flag migration.
            # Touched save: the picked model, explicit True.
            whisper_model_explicit=(
                self._model_explicit or self._model_touched),
            diarization_mode=self.diarize_combo.currentData()
            or DIARIZATION_DEFAULT_MODE,
            sherpa_model_dir=(
                Path(sherpa_dir_text) if sherpa_dir_text else None),
            sherpa_expected_speakers=int(self.sherpa_expected_spin.value()),
            sherpa_clustering_threshold=float(
                self.sherpa_threshold_spin.value()),
            embedding_backend=self.embedding_combo.currentData()
            or EMBEDDING_DEFAULT_BACKEND,
            hf_token_configured=self.hf_configured_check.isChecked(),
            near_live_enabled=self.nearlive_check.isChecked(),
            nearlive_chunk_seconds=float(self.chunk_spin.value()),
            theme=self.theme_combo.currentData() or "light",
            search_mode=self.search_combo.currentData() or "hybrid",
            semantic_model_dir=self._config.semantic_model_dir,
            ollama_enabled=self.llm_check.isChecked(),
            ollama_model=self.ollama_model_edit.text().strip()
            or self._config.ollama_model,
            ollama_base_url=self.ollama_url_edit.text().strip()
            or self._config.ollama_base_url,
            transcript_polish_enabled=self.polish_check.isChecked(),
            capture_system_mode=self.capture_mode_combo.currentData()
            or "auto",
            default_source=self.default_source_combo.currentData(),
            auto_process_recordings=self.auto_process_check.isChecked(),
            auto_detect_teams=self.auto_detect_check.isChecked(),
            outlook_link_enabled=self.outlook_link_check.isChecked(),
            audio_retention_hours=_retention_hours(
                self.retention_combo.currentData()),
            # tinyfix is edited in the Settings view's advanced
            # section, not by this dialog — preserve the incoming
            # values (a plain dialog save must never reset them
            # to the dataclass defaults).
            tinyfix_enabled=self._config.tinyfix_enabled,
            tinyfix_mode=self._config.tinyfix_mode,
            tinyfix_enabled_explicit=(
                self._config.tinyfix_enabled_explicit),
        )
        return cfg

    def apply_and_save(self) -> AppConfig:
        """Validate the dialog, persist the settings, return the config.

        The names dictionary is saved alongside the settings (plain
        JSON in ~/.offair/names_dictionary.json) — it is not part
        of settings.json, so a bad dictionary line can never break
        settings loading.
        """
        cfg = self.config_from_dialog()
        save_settings(cfg)
        save_names_dictionary(
            parse_dictionary_text(self.names_edit.toPlainText()))
        return cfg


def edit_settings(config: AppConfig | None = None,
                  parent=None) -> AppConfig | None:
    """Open the Settings panel. Returns the saved config, or None.

    Returns None when the user cancels, and raises RuntimeError
    when PyQt6 is not installed.
    """
    dialog = SettingsDialog(config, parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.apply_and_save()
