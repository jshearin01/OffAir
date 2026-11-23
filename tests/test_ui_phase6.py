"""Phase 6 tests — redesigned UI, sherpa wiring, app-level theme.

Two halves:

* Qt-free (run in the plain test env): the diarization-mode
  registry, the new ``"sherpa"`` branch of ``diarize_session``,
  sherpa availability, the extended theme stylesheets, the
  per-platform capture messages, the main.py smoke check, and
  source-level guards that the stale Windows-only strings are gone.
* Qt-gated (``pytest.importorskip("PyQt6")`` per test — run with
  the gui-shots venv): the new main-window layout, the app-level
  theme reaching a second window, the sherpa picker + settings,
  and the browser layout. Run them with::

      QT_QPA_PLATFORM=offscreen \
        ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \
        -m pytest tests/test_ui_phase6.py -q
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from app.config import AppConfig
from app.diarization import DIARIZATION_MODES
from app.diarization.session import diarize_session
from app.diarization.sherpa import (
    SherpaDiarizationResult,
    check_availability,
)
from app.ui.theme import (
    apply_theme,
    apply_theme_to_app,
    stylesheet_for,
)

REPO = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------ Qt-free -------
def test_diarization_modes_registry_has_all_three_modes():
    assert set(DIARIZATION_MODES) == {"simple", "full", "sherpa"}
    for mode, info in DIARIZATION_MODES.items():
        assert info["label"] and info["description"], mode
    assert "no token" in DIARIZATION_MODES["sherpa"]["label"].lower()
    assert "Sherpa" in DIARIZATION_MODES["sherpa"]["label"]


def _make_transcribed_session(root: Path) -> Path:
    """A session folder with transcript.json + an (empty) combined.wav."""
    session = root / "2026-11-07_10-00-00_teams"
    session.mkdir(parents=True)
    (session / "metadata.json").write_text(json.dumps({
        "session_id": session.name, "app_name": "Teams",
        "mode": "system-loopback-fallback", "status": "stopped",
        "start_time": "2026-11-07T10:00:00+00:00",
        "end_time": "2026-11-07T10:01:01+00:00",
        "duration_seconds": 61.3, "sample_rate": 48000, "channels": 1,
        "devices": {}, "warning": None, "files": {}}, indent=2),
        encoding="utf-8")
    (session / "transcript.json").write_text(json.dumps({
        "model": "tiny",
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Hello everyone."},
            {"start": 5.0, "end": 9.0, "text": "Budget is twelve thousand."},
        ]}, indent=2), encoding="utf-8")
    (session / "combined.wav").write_bytes(b"")
    return session


class _FakeSherpaDiarizer:
    """Stand-in for SherpaDiarizer (injected ``diarizer`` path)."""

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs

    def diarize(self, audio_path, progress_callback=None):
        if progress_callback is not None:
            progress_callback(1.0, "done")
        return SherpaDiarizationResult(
            turns=[(0.0, 5.0, "SPEAKER_00"), (5.0, 9.0, "SPEAKER_01")],
            speakers=["SPEAKER_00", "SPEAKER_01"])


def test_diarize_session_sherpa_branch_with_fake_diarizer(tmp_path):
    session = _make_transcribed_session(tmp_path)
    payload = diarize_session(
        session, mode="sherpa", diarizer=_FakeSherpaDiarizer())
    assert payload["mode"] == "sherpa"
    assert payload["method"].startswith("sherpa-onnx")
    assert payload["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert len(payload["segments"]) == 2
    assert payload["segments"][0]["speaker"] == "SPEAKER_00"
    assert payload["segments"][1]["speaker"] == "SPEAKER_01"
    assert (session / "diarization.json").exists()
    transcript = json.loads(
        (session / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["segments"][0]["speaker"] == "SPEAKER_00"
    assert transcript["diarization"]["mode"] == "sherpa"
    assert "John" not in (session / "transcript.txt").read_text(
        encoding="utf-8")  # labels only, no invented names


def test_diarize_session_sherpa_passes_config_settings(monkeypatch, tmp_path):
    captured: dict = {}

    class _RecordingSherpa:
        def __init__(self, model_dir=None, expected_speakers=0,
                     clustering_threshold=0.5, **kwargs):
            captured.update(
                model_dir=model_dir,
                expected_speakers=expected_speakers,
                clustering_threshold=clustering_threshold)

        def diarize(self, audio_path, progress_callback=None):
            return SherpaDiarizationResult(
                turns=[(0.0, 5.0, "SPEAKER_00")],
                speakers=["SPEAKER_00"])

    monkeypatch.setattr("app.diarization.sherpa.SherpaDiarizer",
                        _RecordingSherpa)
    session = _make_transcribed_session(tmp_path)
    model_dir = tmp_path / "sherpa-models"
    payload = diarize_session(
        session, mode="sherpa",
        sherpa_model_dir=model_dir,
        sherpa_expected_speakers=3,
        sherpa_clustering_threshold=0.6)
    assert payload["mode"] == "sherpa"
    assert captured["model_dir"] == model_dir
    assert captured["expected_speakers"] == 3
    assert captured["clustering_threshold"] == 0.6


def test_diarize_session_sherpa_needs_audio(tmp_path):
    session = tmp_path / "2026-11-07_10-00-00_teams"
    session.mkdir()
    (session / "transcript.json").write_text(json.dumps({
        "model": "tiny",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hi."}]},
        indent=2), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="combined.wav"):
        diarize_session(session, mode="sherpa",
                        diarizer=_FakeSherpaDiarizer())


def test_diarize_session_unknown_mode_names_all_three(tmp_path):
    session = _make_transcribed_session(tmp_path)
    with pytest.raises(ValueError, match="sherpa"):
        diarize_session(session, mode="bogus")


def test_check_availability_shape_and_ready_means_both():
    info = check_availability()
    assert set(info) == {"installed", "models_present", "model_dir",
                         "label", "ready"}
    assert info["ready"] == (info["installed"] and info["models_present"])
    assert info["label"] == DIARIZATION_MODES["sherpa"]["label"]


def test_theme_stylesheets_cover_new_widgets():
    for theme in ("dark", "light"):
        qss = stylesheet_for(theme)
        assert "QTabWidget::pane" in qss
        assert "QTabBar::tab:selected" in qss
        assert "QFrame#card" in qss
        assert "QPushButton#recordButton" in qss
        assert "QPushButton#primaryButton" in qss
        assert "QSplitter::handle" in qss
        assert "QLabel#appTitle" in qss
        assert "QLabel#timerLabel" in qss
    assert stylesheet_for("dark") != stylesheet_for("light")


def test_apply_theme_to_app_sets_stylesheet_on_fake_app():
    class _FakeApp:
        def __init__(self):
            self.stylesheet = None

        def setStyleSheet(self, qss):
            self.stylesheet = qss

    app = _FakeApp()
    returned = apply_theme_to_app(app, "dark")
    assert returned == stylesheet_for("dark")
    assert app.stylesheet == returned
    with pytest.raises(ValueError, match="Unknown theme"):
        apply_theme_to_app(app, "neon")
    # apply_theme itself still works on a plain widget (Phase 5 API).
    assert apply_theme(app, "light") == stylesheet_for("light")


def test_platform_messages_name_the_right_backend(monkeypatch, no_sd):
    from app.platform_info import (
        loopback_empty_message,
        mic_empty_message,
    )
    # Linux without sounddevice: the message must name sounddevice /
    # PortAudio, never pyaudiowpatch. (``no_sd`` blocks the import —
    # the package IS installed in a full Linux venv.)
    monkeypatch.setattr(sys, "platform", "linux")
    assert "sounddevice" in mic_empty_message()
    assert "sounddevice" in loopback_empty_message()
    assert "pyaudiowpatch" not in mic_empty_message()
    assert "pyaudiowpatch" not in loopback_empty_message()
    # Windows: pyaudiowpatch is the right name there.
    monkeypatch.setattr(sys, "platform", "win32")
    assert "pyaudiowpatch" in mic_empty_message()
    assert "pyaudiowpatch" in loopback_empty_message()


def test_smoke_check_uses_platform_capture_state():
    from main import smoke_check
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert smoke_check(AppConfig()) == 0
    out = buf.getvalue()
    assert "Capture" in out
    assert "Run on Windows 10/11" not in out
    assert "audio capture is unavailable here" not in out


def test_stale_windows_only_strings_are_gone():
    main_window = (REPO / "app" / "ui" / "main_window.py").read_text(
        encoding="utf-8")
    assert "pyaudiowpatch required" not in main_window
    assert "System loopback fallback — pyaudiowpatch" not in main_window
    main_py = (REPO / "main.py").read_text(encoding="utf-8")
    assert "Run on Windows 10/11" not in main_py
    assert "describe_capture_state" in main_py
    assert "apply_theme_to_app" in main_py


def test_requirements_lists_sherpa_as_optional_only():
    text = (REPO / "requirements.txt").read_text(encoding="utf-8")
    assert "sherpa-onnx>=2.0" in text
    # The Track 3 Linux line is untouched.
    assert "sounddevice>=0.5; sys_platform == \"linux\"" in text
    # sherpa-onnx must not be pinned to a platform (it is cross-platform).
    for line in text.splitlines():
        if line.strip().startswith("sherpa-onnx"):
            assert "sys_platform" not in line


def test_config_sherpa_settings_round_trip(tmp_path):
    from app.config import load_settings, save_settings
    cfg = AppConfig(sherpa_expected_speakers=3,
                    sherpa_clustering_threshold=0.6,
                    sherpa_model_dir=tmp_path / "models")
    path = save_settings(cfg, tmp_path / "settings.json")
    loaded = load_settings(path)
    assert loaded.sherpa_expected_speakers == 3
    assert loaded.sherpa_clustering_threshold == 0.6
    assert loaded.sherpa_model_dir == tmp_path / "models"


# ------------------------------------------------------------ Qt-gated ------
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/offair-demo/runtime")
    Path("/tmp/offair-demo/runtime").mkdir(parents=True, exist_ok=True)
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _all_widget_texts(widget) -> list[str]:
    """Text of every QLabel/QPushButton/QGroupBox under ``widget``."""
    from PyQt6.QtWidgets import QGroupBox, QLabel, QPushButton
    texts = []
    for child in widget.findChildren(QLabel):
        texts.append(child.text())
    for child in widget.findChildren(QPushButton):
        texts.append(child.text())
    for child in widget.findChildren(QGroupBox):
        texts.append(child.title())
    return texts


def test_main_window_new_layout_has_every_control(qapp, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QListWidget, QStackedWidget
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        # Fits a 1280x800 window (the old stack needed ~2500 px).
        assert win.width() == 1280 and win.height() == 800
        # Header: app name + timer.
        assert any("OffAir" in t for t in _all_widget_texts(win))
        assert win.elapsed_label.text() == "00:00"
        # Left navigation + stacked views, not one endless column
        # (Nov 19 2026 redesign — the old 7 tabs are gone).
        nav = win.findChild(QListWidget, "navList")
        assert nav is not None
        assert win.tabs is not None
        assert isinstance(win.tabs, QStackedWidget)
        assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == [
            "Record", "Library", "Calendar", "Session", "People",
            "Settings",
            # MCP setup and Help are flat nav rows now (Nov 20
            # 2026 — the "More" header is gone), not grouped.
            # Help is last: reference material, not part of the
            # record → process → review flow.
            "MCP setup", "Help"]
        # Record is the landing view.
        assert win.tabs.tabText(win.tabs.currentIndex()) == "Record"
        # Every control from the old UI is reachable by attribute.
        # (Nov 20 2026: the Record view no longer has a capture
        # mode combo, a refresh-apps button or a browse button —
        # capture mode and the default source live in the Settings
        # form, the app picker refreshes on popup — and the
        # mic/system combos live in the Settings view, still as
        # MainWindow attributes.)
        for name in (
                "mic_combo", "loopback_combo", "process_combo",
                "mode_label", "status_label", "elapsed_label",
                "session_label", "mic_meter", "system_meter",
                "record_btn", "pause_btn", "resume_btn", "stop_btn",
                "refresh_btn", "model_combo",
                "transcribe_btn", "transcribe_progress",
                "transcribe_label", "diarize_mode_combo",
                "hf_token_edit", "diarize_warning_label",
                "min_speakers_edit", "max_speakers_edit",
                "diarize_btn", "diarize_progress", "diarize_label",
                "segments_list", "rename_label_combo",
                "rename_name_edit", "rename_btn",
                "known_speakers_list", "suggestions_list",
                "suggest_btn", "confirm_btn", "correct_btn",
                "nearlive_list", "nearlive_btn", "nearlive_progress",
                "nearlive_label", "live_model_label",
                "export_btn", "export_label",
                "summarize_btn", "summarize_label", "settings_btn",
                "browser_btn", "sherpa_status_label",
                # New navigation / views (Nov 19 2026).
                "nav_list", "session_empty_label",
                "library_search_edit", "library_mode_combo",
                "library_source_combo", "library_list",
                "library_status_label", "library_open_btn",
                "settings_save_btn", "settings_cancel_btn",
                "calendar_widget", "mcp_setup_widget", "help_tab"):
            assert hasattr(win, name), name
        # Record is the prominent control; Stop is styled as danger.
        assert win.record_btn.objectName() == "recordButton"
        assert win.stop_btn.objectName() == "dangerButton"
        # Capture state comes from platform_info, not hard-coded text.
        assert "Capture:" in win.mode_label.text()
        assert "pyaudiowpatch required" not in win.mode_label.text()
        # Target-app picker (Nov 10 2026): rows are live audio
        # sessions, not a hard-coded exe list. On this Linux VM
        # there are no WASAPI sessions, so the picker shows its
        # empty state plus the "Other app…" manual entry. On real
        # Windows the first row is a live session instead.
        _first = win.process_combo.itemText(0)
        if sys.platform == "win32":
            assert _first == "(no apps with audio sessions)" or _first.endswith(
                ("— playing sound", "— idle"))
        else:
            assert _first == "(no apps with audio sessions)"
        assert win.process_combo.itemText(
            win.process_combo.count() - 1) == "Other app…"
        # Device combos reflect the real capture state on this machine.
        from app import platform_info

        info = platform_info.describe_capture_state()
        if info.microphones:
            assert win.mic_combo.itemText(0) == (
                info.microphones[0].display_name())
        else:
            # Device placeholders are short so they never truncate in the
            # fixed-width card; the full cause lives in the tooltip.
            assert win.mic_combo.itemText(0) == "(no microphone found)"
            assert win.mic_combo.toolTip()
        # Device names are longer than the fixed-width Record card, so
        # the combo elides them — the full name must be in the tooltip
        # (Nov 10 2026: "Speakers (High Definition Audio Device)
        # [Loopba…" with no way to see the rest). Per-item tooltips do
        # the same in the dropdown.
        if info.loopback_devices:
            assert win.loopback_combo.toolTip() == (
                win.loopback_combo.currentText())
            assert win.loopback_combo.toolTip()
            for _i in range(win.loopback_combo.count()):
                assert win.loopback_combo.itemData(
                    _i, Qt.ItemDataRole.ToolTipRole)
    finally:
        win.close()


def test_main_window_diarize_picker_offers_sherpa(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.show()
        # The sherpa status label lives on the Session view — show it.
        win.show_view("Session")
        qapp.processEvents()
        combo = win.diarize_mode_combo
        assert [combo.itemData(i) for i in range(combo.count())] == [
            "simple", "full", "sherpa"]
        # Selecting sherpa shows its availability state, with a
        # guided message (sherpa-onnx is not installed in this env).
        combo.setCurrentIndex(combo.findData("sherpa"))
        qapp.processEvents()
        assert win.sherpa_status_label.isVisible()
        text = win.sherpa_status_label.text()
        assert ("sherpa-onnx is not installed" in text
                or "will download once" in text
                or "Sherpa ready" in text)
        assert win.hf_token_edit.isEnabled() is False  # sherpa: no token
        # Full mode re-enables the token field.
        combo.setCurrentIndex(combo.findData("full"))
        qapp.processEvents()
        assert win.hf_token_edit.isEnabled() is True
        assert win.sherpa_status_label.isVisible() is False
    finally:
        win.close()


def test_theme_applies_at_app_level_to_second_window(qapp, tmp_path):
    from PyQt6.QtWidgets import QApplication
    from app.config import AppConfig
    from app.ui.browser import create_browser
    from app.ui.theme import apply_theme_to_app, stylesheet_for

    app = QApplication.instance()
    apply_theme_to_app(app, "dark")
    assert app.styleSheet() == stylesheet_for("dark")
    browser = create_browser(AppConfig(output_root=tmp_path))
    try:
        # The browser sets no stylesheet of its own: it follows the
        # app-level theme (the Phase 5 bug was a per-window apply).
        assert browser.styleSheet() == ""
        # Switching the app theme switches what every window follows.
        apply_theme_to_app(app, "light")
        assert app.styleSheet() == stylesheet_for("light")
    finally:
        browser.close()
        apply_theme_to_app(app, "dark")


def test_settings_dialog_has_sherpa_fields_and_round_trips(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.settings_dialog import SettingsDialog

    cfg = AppConfig(output_root=tmp_path, diarization_mode="sherpa",
                    sherpa_expected_speakers=3,
                    sherpa_clustering_threshold=0.6)
    dlg = SettingsDialog(cfg)
    try:
        assert [dlg.diarize_combo.itemData(i)
                for i in range(dlg.diarize_combo.count())] == [
                    "simple", "full", "sherpa"]
        assert dlg.diarize_combo.currentData() == "sherpa"
        assert dlg.sherpa_expected_spin.value() == 3
        assert dlg.sherpa_threshold_spin.value() == pytest.approx(0.6)
        assert dlg.sherpa_model_dir_edit.text() != ""
        # The search-mode default is kept (hybrid since Nov 8 2026).
        assert dlg.search_combo.currentData() == "hybrid"
        # Round-trip through config_from_dialog.
        dlg.sherpa_expected_spin.setValue(0)
        dlg.sherpa_threshold_spin.setValue(0.5)
        out = dlg.config_from_dialog()
        assert out.diarization_mode == "sherpa"
        assert out.sherpa_expected_speakers == 0
        assert out.sherpa_clustering_threshold == pytest.approx(0.5)
        assert out.search_mode == "hybrid"
    finally:
        dlg.close()


def test_browser_side_by_side_layout_and_search(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.browser import create_browser

    session = _make_transcribed_session(tmp_path)
    (session / "transcript.txt").write_text(
        "[00:00] John: Hello everyone.\n"
        "[00:05] Marcus: The budget is twelve thousand dollars, "
        "deadline the fifteenth of December.\n", encoding="utf-8")
    browser = create_browser(AppConfig(output_root=tmp_path))
    try:
        browser.show()
        qapp.processEvents()
        assert browser.session_list.count() == 1
        # Search filters to the matching session (hybrid mode
        # default; in this venv the MiniLM model is unavailable,
        # so hybrid degrades to the text search — same results).
        assert browser.mode_combo.currentData() == "hybrid"
        browser.search_edit.setText("December")
        qapp.processEvents()
        assert browser.session_list.count() == 1
        browser.search_edit.setText("nothing matches this")
        qapp.processEvents()
        assert browser.session_list.count() == 0
        browser.search_edit.setText("")
        qapp.processEvents()
        assert browser.session_list.count() == 1
        # Clicking a session shows its transcript beside the list.
        browser.session_list.setCurrentRow(0)
        browser._on_session_clicked(None)
        assert "twelve thousand" in browser.transcript_view.toPlainText()
    finally:
        browser.close()


# ------------------------------------------- Nov 19 2026 redesign: nav ------
def test_nav_lists_views_in_order_and_switches(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.show()
        qapp.processEvents()
        nav = win.nav_list
        assert [nav.item(i).text() for i in range(nav.count())] == [
            "Record", "Library", "Calendar", "Session", "People",
            "Settings", "MCP setup", "Help"]
        # Flat nav (Nov 20 2026): every row is a view, every row
        # is selectable, and each row maps to exactly one view.
        from PyQt6.QtCore import Qt
        for i in range(nav.count()):
            assert nav.item(i).flags() & Qt.ItemFlag.ItemIsSelectable
            assert nav.item(i).data(Qt.ItemDataRole.UserRole) == \
                nav.item(i).text()
        # Picking a nav row switches the stack; show_view syncs
        # the nav back.
        nav.setCurrentRow(4)
        qapp.processEvents()
        assert win.tabs.currentIndex() == 4
        assert win.tabs.tabText(4) == "People"
        win.show_view("MCP setup")
        qapp.processEvents()
        assert win.tabs.currentIndex() == 6
        assert nav.currentRow() == 6
        with pytest.raises(ValueError, match="Unknown view"):
            win.show_view("Nope")
    finally:
        win.close()


def test_library_search_filters_and_open_loads_session_view(
        qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    first = _make_transcribed_session(tmp_path)
    (first / "transcript.txt").write_text(
        "[00:00] John: Hello everyone.\n"
        "[00:05] Marcus: The budget is twelve thousand dollars.\n",
        encoding="utf-8")
    second = tmp_path / "2026-11-08_10-00-00_teams"
    second.mkdir()
    (second / "metadata.json").write_text(json.dumps({
        "session_id": second.name, "app_name": "Teams",
        "mode": "system-loopback-fallback", "status": "completed",
        "start_time": "2026-11-08T10:00:00+00:00",
        "end_time": "2026-11-08T10:01:01+00:00",
        "duration_seconds": 61.3, "sample_rate": 48000, "channels": 1,
        "devices": {}, "warning": None, "files": {}}, indent=2),
        encoding="utf-8")
    (second / "transcript.json").write_text(json.dumps({
        "model": "tiny",
        "segments": [{"start": 0.0, "end": 5.0,
                      "text": "Quarterly review notes."}]}, indent=2),
        encoding="utf-8")
    (second / "transcript.txt").write_text(
        "[00:00] Priya: Quarterly review notes.\n", encoding="utf-8")
    (second / "combined.wav").write_bytes(b"")
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.show()
        win.show_view("Library")
        qapp.processEvents()
        # Text mode: hybrid would also surface the semantic-only
        # "Quarterly review" hit for "Budget", which is correct
        # hybrid behaviour, not a filtering failure.
        win.library_mode_combo.setCurrentIndex(
            win.library_mode_combo.findData("text"))
        qapp.processEvents()
        # Nov 21 2026: the unfiltered list has a week-group
        # header row above the two sessions (both in the same
        # ISO week), so 3 rows; search results stay flat.
        assert win.library_list.count() == 3
        assert "2 session(s)" in win.library_status_label.text()
        win.library_search_edit.setText("Budget")
        qapp.processEvents()
        assert win.library_list.count() == 1
        assert "Nov 07, 2026" in win.library_list.item(0).text()
        win.library_search_edit.setText("nothing matches this")
        qapp.processEvents()
        assert win.library_list.count() == 0
        win.library_search_edit.setText("")
        qapp.processEvents()
        assert win.library_list.count() == 3
        # Open the older session: it loads and the Session view shows.
        win.library_list.setCurrentRow(2)
        win._on_library_clicked(None)
        qapp.processEvents()
        assert win.library_open_btn.isEnabled()
        win.library_open_btn.click()
        qapp.processEvents()
        assert win.session_dir == first
        assert win.tabs.tabText(win.tabs.currentIndex()) == "Session"
        assert "Teams recording" in win.loaded_session_label.text()
        assert win.loaded_session_label.toolTip() == str(first)
    finally:
        win.close()


def test_session_view_empty_state_with_nothing_loaded(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.show()
        win.show_view("Session")
        qapp.processEvents()
        assert win.session_dir is None
        assert win.session_empty_label.isVisible()
        assert "No session loaded" in win.session_empty_label.text()
        assert not win.transcribe_btn.isEnabled()
    finally:
        win.close()


def test_settings_view_save_round_trips(qapp, tmp_path, monkeypatch):
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    captured: dict = {}

    def _fake_save(cfg, path=None):
        captured["cfg"] = cfg
        return tmp_path / "settings.json"

    monkeypatch.setattr("app.ui.settings_dialog.save_settings",
                        _fake_save)
    monkeypatch.setattr("app.config.save_settings", _fake_save)
    monkeypatch.setattr(
        "app.ui.settings_dialog.save_names_dictionary",
        lambda _d: None)
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.show()
        win.show_view("Settings")
        qapp.processEvents()
        dlg = win._settings_dialog
        assert dlg.model_combo.currentData() == "base"
        assert dlg.theme_combo.currentData() == "light"
        dlg.model_combo.setCurrentIndex(
            dlg.model_combo.findData("tiny"))
        # A real user pick emits `activated`; a programmatic set
        # alone does not (Nov 20 2026, Item 8) — emit it, so the
        # save records an explicit pick.
        dlg.model_combo.activated.emit(dlg.model_combo.currentIndex())
        win.settings_save_btn.click()
        qapp.processEvents()
        assert captured["cfg"].whisper_model == "tiny"
        assert captured["cfg"].whisper_model_explicit is True
        assert win.config.whisper_model == "tiny"
        assert win.model_combo.currentData() == "tiny"
        # Saving rebuilds the form from the saved config.
        assert win._settings_dialog.model_combo.currentData() == "tiny"
    finally:
        win.close()


def test_settings_view_tinyfix_controls_round_trip(
        qapp, tmp_path, monkeypatch):
    """tinyfix (Nov 21 2026): the advanced Settings checkbox +
    mode combo sync from the config, and saving persists them —
    including the mode combo disabling with the checkbox.

    Nov 22 2026: the default flipped to OFF (opt-in), and a
    user click on the checkbox marks the choice explicit, so a
    plain save never promotes a carried-over default — mirrors
    the whisper_model_explicit tracking."""
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    captured: dict = {}

    def _fake_save(cfg, path=None):
        captured["cfg"] = cfg
        return tmp_path / "settings.json"

    monkeypatch.setattr("app.ui.settings_dialog.save_settings",
                        _fake_save)
    monkeypatch.setattr("app.config.save_settings", _fake_save)
    monkeypatch.setattr(
        "app.ui.settings_dialog.save_names_dictionary",
        lambda _d: None)
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.show()
        win.show_view("Settings")
        qapp.processEvents()
        # Defaults: OFF, careful mode, combo disabled, nothing
        # explicit.
        assert win.tinyfix_check.isChecked() is False
        assert win.tinyfix_mode_combo.currentData() == "careful"
        assert win.tinyfix_mode_combo.isEnabled() is False
        assert win._tinyfix_explicit is False
        # A user click enables it + picks balanced; save persists
        # the opt-in AND the explicit flag.
        win.tinyfix_check.click()
        qapp.processEvents()
        assert win.tinyfix_check.isChecked() is True
        assert win.tinyfix_mode_combo.isEnabled() is True
        assert win._tinyfix_explicit is True
        win.tinyfix_mode_combo.setCurrentIndex(
            win.tinyfix_mode_combo.findData("balanced"))
        win.settings_save_btn.click()
        qapp.processEvents()
        assert captured["cfg"].tinyfix_enabled is True
        assert captured["cfg"].tinyfix_enabled_explicit is True
        assert captured["cfg"].tinyfix_mode == "balanced"
        assert win.config.tinyfix_enabled is True
        assert win.config.tinyfix_enabled_explicit is True
        # The rebuilt form reflects the saved config.
        assert win.tinyfix_check.isChecked() is True
        assert win.tinyfix_mode_combo.currentData() == "balanced"
        assert win.tinyfix_mode_combo.isEnabled() is True
    finally:
        win.close()


def test_settings_view_keeps_unsaved_edits_on_navigation(
        qapp, tmp_path, monkeypatch):
    """S4: navigating away from Settings and back must not discard
    unsaved edits; Discard still reloads the saved settings."""
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.show()
        win.show_view("Settings")
        qapp.processEvents()
        dlg = win._settings_dialog
        assert dlg.model_combo.currentData() == "base"
        dlg.model_combo.setCurrentIndex(
            dlg.model_combo.findData("tiny"))
        assert dlg.model_combo.currentData() == "tiny"
        win.show_view("Library")
        qapp.processEvents()
        win.show_view("Settings")
        qapp.processEvents()
        # Same dialog, edits intact — the form was not rebuilt.
        assert win._settings_dialog is dlg
        assert win._settings_dialog.model_combo.currentData() == "tiny"
        # Discard throws the edits away and reloads saved settings.
        win.settings_cancel_btn.click()
        qapp.processEvents()
        assert win._settings_dialog is not dlg
        assert win._settings_dialog.model_combo.currentData() == "base"
    finally:
        win.close()


# ------------------------------------------- Item 2: simplified Record view --
def test_record_view_has_no_device_mode_or_source_controls(qapp, tmp_path):
    # Nov 20 2026: the Record view keeps only the app picker, the
    # mode callout, meters, transport and the live transcript.
    # Device combos moved to the Settings view; capture mode and
    # source live in the Settings form; the refresh-apps and
    # browse buttons are deleted.
    from PyQt6.QtWidgets import QComboBox, QLabel

    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        for name in ("capture_mode_combo", "source_combo",
                     "refresh_apps_btn", "browse_btn"):
            assert not hasattr(win, name), name
        record_page = win._views["Record"]
        settings_page = win._views["Settings"]
        record_combos = record_page.findChildren(QComboBox)
        assert win.process_combo in record_combos
        assert win.mic_combo not in record_combos
        assert win.loopback_combo not in record_combos
        settings_combos = settings_page.findChildren(QComboBox)
        assert win.mic_combo in settings_combos
        assert win.loopback_combo in settings_combos
        assert win.live_model_label in record_page.findChildren(QLabel)
        assert win.live_model_label.text().startswith(
            "Transcribing with Whisper ")
    finally:
        win.close()


def test_app_picker_refreshes_when_opened(qapp, tmp_path, monkeypatch):
    # Nov 20 2026: the ↻ button next to the app picker is gone —
    # the picker refreshes when it is opened instead.
    import app.ui.main_window as mw
    from app.audio.app_sessions import AudioSessionInfo
    from app.ui.main_window import create_main_window

    sessions = [AudioSessionInfo("ms-teams.exe", "Microsoft Teams",
                                 100, "active", 0.4)]
    monkeypatch.setattr(mw, "list_audio_sessions",
                        lambda **kwargs: list(sessions))
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        assert win.process_combo.findData("ms-teams.exe") >= 0
        assert win.process_combo.findData("chrome.exe") < 0
        # A new app starts after the window opened.
        sessions.append(AudioSessionInfo(
            "chrome.exe", "Google Chrome", 200, "active", 0.3))
        win.process_combo.showPopup()
        qapp.processEvents()
        assert win.process_combo.findData("chrome.exe") >= 0
    finally:
        win.close()
