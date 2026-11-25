"""Qt-gated tests — target-app picker, capture-mode control, and the
silence warnings (Nov 10 2026).

The picker lists mocked audio sessions (no real WASAPI stack on
this Linux VM); the capture is faked the same way
tests/test_auto_process.py fakes the workers. Run with the
gui-shots venv::

    QT_QPA_PLATFORM=offscreen \
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \
      -m pytest tests/test_ui_app_picker.py -q
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from app.audio.app_sessions import AudioSessionInfo
from app.audio.capture import CaptureState
from app.config import AppConfig


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


def _make_window(monkeypatch, tmp_path, sessions=None, **config_kwargs):
    """MainWindow with a mocked session list.

    ``sessions`` is the list list_audio_sessions() returns (None =
    the real call, which is [] on this Linux VM).
    """
    import app.ui.main_window as mw

    if sessions is not None:
        monkeypatch.setattr(
            mw, "list_audio_sessions", lambda **kwargs: list(sessions))
    from app.ui.main_window import create_main_window

    win = create_main_window(
        AppConfig(output_root=tmp_path, **config_kwargs))
    return win


class _FakeCapture:
    """A recording capture: start() succeeds, mode/warning recorded."""

    def __init__(self, session_dir, *, process_name="ms-teams.exe",
                 capture_system_mode="auto", **kwargs):
        self.session_dir = Path(session_dir)
        self.process_name = process_name
        self.capture_system_mode = capture_system_mode
        self.state = CaptureState.RECORDING
        self.mode = ("per-app" if capture_system_mode == "per-app"
                     else "system-loopback-fallback")
        self.warning = None
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.state = CaptureState.STOPPED
        return {}


def _chrome_and_teams():
    return [
        AudioSessionInfo("chrome.exe", "Google Chrome", 100,
                         "active", 0.4),
        AudioSessionInfo("ms-teams.exe", "Microsoft Teams", 200,
                         "idle", 0.0),
    ]


# ------------------------------------------------------------ picker -------
def test_picker_lists_mocked_sessions(qapp, tmp_path, monkeypatch):
    win = _make_window(monkeypatch, tmp_path,
                       sessions=_chrome_and_teams())
    try:
        texts = [win.process_combo.itemText(i)
                 for i in range(win.process_combo.count())]
        assert texts == [
            "Google Chrome — playing sound",
            "Microsoft Teams — idle",
            "Other app…",
        ]
        # Teams is preferred: it is the configured target.
        assert win.selected_process_name() == "ms-teams.exe"
        assert "Microsoft Teams" in win.mode_label.text()
        assert "one app" in win.mode_label.text()
    finally:
        win.close()


def test_picker_prefers_active_when_no_teams(qapp, tmp_path, monkeypatch):
    sessions = [
        AudioSessionInfo("spotify.exe", "Spotify", 300, "idle", 0.0),
        AudioSessionInfo("chrome.exe", "Google Chrome", 100,
                         "active", 0.4),
    ]
    win = _make_window(monkeypatch, tmp_path, sessions=sessions)
    try:
        # Active sessions sort first, so Chrome is picked.
        assert win.selected_process_name() == "chrome.exe"
    finally:
        win.close()


def test_picker_empty_state(qapp, tmp_path, monkeypatch):
    win = _make_window(monkeypatch, tmp_path, sessions=[])
    try:
        assert win.process_combo.itemText(0) == (
            "(no apps with audio sessions)")
        assert win.process_combo.itemText(
            win.process_combo.count() - 1) == "Other app…"
        assert win.selected_process_name() is None
        # No app picked: the explanation says so honestly.
        assert "the chosen app" in win.mode_label.text()
    finally:
        win.close()


def test_picker_refreshes_when_dropdown_opens(
        qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw

    sessions: list = []
    monkeypatch.setattr(
        mw, "list_audio_sessions", lambda **kwargs: list(sessions))
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        assert win.process_combo.itemText(0) == (
            "(no apps with audio sessions)")
        sessions.extend(_chrome_and_teams())
        win.process_combo.showPopup()
        qapp.processEvents()
        win.process_combo.hidePopup()
        texts = [win.process_combo.itemText(i)
                 for i in range(win.process_combo.count())]
        assert texts[0] == "Google Chrome — playing sound"
        assert texts[-1] == "Other app…"
    finally:
        win.close()


def test_other_app_entry_sets_manual_name(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    import app.ui.main_window as mw

    monkeypatch.setattr(
        QInputDialog, "getText",
        classmethod(lambda cls, *a, **k: ("chrome.exe", True)))
    win = _make_window(monkeypatch, tmp_path, sessions=[])
    try:
        other = win.process_combo.findData(
            mw.OTHER_APP_SENTINEL)
        win.process_combo.setCurrentIndex(other)
        qapp.processEvents()
        assert win.selected_process_name() == "chrome.exe"
        texts = [win.process_combo.itemText(i)
                 for i in range(win.process_combo.count())]
        assert "Google Chrome — manual entry" in texts
    finally:
        win.close()


def test_other_app_entry_cancel_reverts(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    import app.ui.main_window as mw

    monkeypatch.setattr(
        QInputDialog, "getText",
        classmethod(lambda cls, *a, **k: ("", False)))
    win = _make_window(monkeypatch, tmp_path,
                       sessions=_chrome_and_teams())
    try:
        # Teams was the preferred pick; cancelling "Other app…"
        # must leave it selected, not the sentinel row.
        assert win.selected_process_name() == "ms-teams.exe"
        other = win.process_combo.findData(mw.OTHER_APP_SENTINEL)
        win.process_combo.setCurrentIndex(other)
        qapp.processEvents()
        assert win.selected_process_name() == "ms-teams.exe"
    finally:
        win.close()


# ------------------------------------------------------- capture mode ------
def test_capture_mode_setting_writes_and_explains(
        qapp, tmp_path, monkeypatch):
    # Nov 20 2026: the Record view's capture-mode combo is gone —
    # the Settings dialog's combo is the only editor, and saving
    # updates the Record view's mode callout.
    from app.ui.settings_dialog import SettingsDialog

    win = _make_window(monkeypatch, tmp_path, sessions=[])
    try:
        assert win.config.capture_system_mode == "auto"
        assert "one app" in win.mode_label.text()
        dlg = SettingsDialog(win.config)
        try:
            assert dlg.capture_mode_combo.currentData() == "auto"
            dlg.capture_mode_combo.setCurrentIndex(2)  # loopback
            qapp.processEvents()
            out = dlg.config_from_dialog()
            assert out.capture_system_mode == "loopback"
        finally:
            dlg.close()
        win._apply_saved_config(out)
        assert win.config.capture_system_mode == "loopback"
        assert "everything you hear" in win.mode_label.text()
        assert "every app at once" in win.mode_label.text()
        dlg = SettingsDialog(win.config)
        try:
            dlg.capture_mode_combo.setCurrentIndex(0)  # auto
            out = dlg.config_from_dialog()
            assert out.capture_system_mode == "auto"
        finally:
            dlg.close()
        win._apply_saved_config(out)
        assert win.config.capture_system_mode == "auto"
        assert "one app" in win.mode_label.text()
    finally:
        win.close()


def test_capture_mode_shows_per_app_as_record_one_app(
        qapp, tmp_path, monkeypatch):
    # "per-app" is set in Settings; the Record view's mode callout
    # shows it as "Record one app" (it is only downgraded to
    # "auto" if the user re-picks it).
    win = _make_window(monkeypatch, tmp_path, sessions=[],
                       capture_system_mode="per-app")
    try:
        assert win.config.capture_system_mode == "per-app"
        assert "one app" in win.mode_label.text()
        assert not hasattr(win, "capture_mode_combo")
    finally:
        win.close()


def test_settings_dialog_round_trips_capture_mode(
        qapp, tmp_path, monkeypatch):
    from app.ui.settings_dialog import SettingsDialog

    cfg = AppConfig(output_root=tmp_path, capture_system_mode="loopback")
    dlg = SettingsDialog(cfg)
    try:
        assert dlg.capture_mode_combo.currentData() == "loopback"
        out = dlg.config_from_dialog()
        assert out.capture_system_mode == "loopback"
    finally:
        dlg.close()


# ------------------------------------------------------------ on_record ----
def test_record_uses_picked_exe(qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw

    monkeypatch.setattr(mw, "DualCapture", _FakeCapture)
    win = _make_window(monkeypatch, tmp_path,
                       sessions=_chrome_and_teams())
    try:
        chrome = win.process_combo.findData("chrome.exe")
        win.process_combo.setCurrentIndex(chrome)
        qapp.processEvents()
        win.on_record()
        qapp.processEvents()
        assert isinstance(win.capture, _FakeCapture)
        assert win.capture.process_name == "chrome.exe"
        assert win.capture.capture_system_mode == "auto"
        assert win.capture.started is True
        # The session folder is named for the app actually recorded.
        assert "chrome" in win.session_dir.name
    finally:
        win.close()


def test_auto_record_uses_configured_target_not_picker(
        qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw

    monkeypatch.setattr(mw, "DualCapture", _FakeCapture)
    win = _make_window(monkeypatch, tmp_path,
                       sessions=_chrome_and_teams())
    try:
        # The picker is on Chrome, but an auto-started recording is
        # for the meeting the watcher detected (Teams).
        chrome = win.process_combo.findData("chrome.exe")
        win.process_combo.setCurrentIndex(chrome)
        assert win.selected_process_name() == "chrome.exe"
        win.on_record(auto=True)
        qapp.processEvents()
        assert isinstance(win.capture, _FakeCapture)
        assert win.capture.process_name == "ms-teams.exe"
    finally:
        win.close()


def test_record_without_session_warns_and_cancels(
        qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw

    # Windows-only gate: pretend to be Windows with a usable probe.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(mw, "audio_sessions_available",
                        lambda: (True, None))
    monkeypatch.setattr(mw, "DualCapture", _FakeCapture)
    win = _make_window(monkeypatch, tmp_path, sessions=[])
    try:
        win._manual_process_name = "ms-teams.exe"
        win._refresh_app_picker()
        qapp.processEvents()
        assert win.selected_process_name() == "ms-teams.exe"
        asked: list = []

        def _cancel(name, **kwargs):
            asked.append((name, kwargs))
            return False

        # No waiting in this test: the probe reports no session
        # once, the dialog is reached immediately. The process
        # probe is pinned (not running) so the test is hermetic.
        monkeypatch.setattr(mw, "_TARGET_SESSION_WAIT_SECONDS", 0)
        monkeypatch.setattr(win, "_target_process_running",
                            lambda exe: False)
        monkeypatch.setattr(
            win, "_confirm_record_without_session", _cancel)
        win.on_record()
        qapp.processEvents()
        assert asked == [("Microsoft Teams", {"target_running": False})]
        # Cancelled: no capture started, no session folder created.
        assert win.capture is None
        assert win.session_dir is None
        assert list(Path(tmp_path).iterdir()) == []
    finally:
        win.close()


def test_record_without_session_records_anyway_system_wide(
        qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(mw, "audio_sessions_available",
                        lambda: (True, None))
    monkeypatch.setattr(mw, "DualCapture", _FakeCapture)
    win = _make_window(monkeypatch, tmp_path, sessions=[])
    try:
        win._manual_process_name = "ms-teams.exe"
        win._refresh_app_picker()
        qapp.processEvents()
        monkeypatch.setattr(mw, "_TARGET_SESSION_WAIT_SECONDS", 0)
        monkeypatch.setattr(
            win, "_confirm_record_without_session",
            lambda name, **kwargs: True)
        win.on_record()
        qapp.processEvents()
        assert isinstance(win.capture, _FakeCapture)
        # The user's explicit choice: system-wide for this recording.
        assert win.capture.capture_system_mode == "loopback"
        assert win.capture.process_name == "ms-teams.exe"
    finally:
        win.close()


def test_record_waits_for_session_then_records_per_app(
        qapp, tmp_path, monkeypatch):
    """A session appearing during the wait means no dialog at all.

    Teams' session is transient: it appears when the call
    connects, which may be just after Record is pressed. The
    probe reports nothing, then a session on the next poll —
    recording starts in the requested mode, never system-wide.
    """
    import app.ui.main_window as mw

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(mw, "audio_sessions_available",
                        lambda: (True, None))
    monkeypatch.setattr(mw, "DualCapture", _FakeCapture)
    win = _make_window(monkeypatch, tmp_path, sessions=[])
    try:
        win._manual_process_name = "ms-teams.exe"
        win._refresh_app_picker()
        qapp.processEvents()
        probes = iter([False, True])
        monkeypatch.setattr(win, "_target_has_audio_session",
                            lambda exe: next(probes))

        def _must_not_ask(name, **kwargs):
            raise AssertionError("dialog must not appear: "
                                 "the session arrived during the wait")

        monkeypatch.setattr(
            win, "_confirm_record_without_session", _must_not_ask)
        win.on_record()
        qapp.processEvents()
        assert isinstance(win.capture, _FakeCapture)
        # The requested mode is kept: per-app is tried, not loopback.
        assert win.capture.capture_system_mode == "auto"
        assert win.capture.process_name == "ms-teams.exe"
    finally:
        win.close()


def test_no_session_dialog_text_names_the_cause():
    """Running-but-silent vs not-running get their own wording."""
    from app.ui.main_window import MainWindow

    title, text, informative = MainWindow._no_session_dialog_text(
        "Microsoft Teams", target_running=True)
    assert title == "No audio from Microsoft Teams"
    assert "is running but isn't playing" in text
    assert "Join the call first" in informative
    title, text, informative = MainWindow._no_session_dialog_text(
        "Microsoft Teams", target_running=False)
    assert title == "No audio from Microsoft Teams"
    assert "isn't running" in text
    assert "Start Microsoft Teams" in informative


def test_record_without_session_running_target_names_cause(
        qapp, tmp_path, monkeypatch):
    """Dialog is told the app is running but silent (not absent)."""
    import app.ui.main_window as mw

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(mw, "audio_sessions_available",
                        lambda: (True, None))
    monkeypatch.setattr(mw, "DualCapture", _FakeCapture)
    win = _make_window(monkeypatch, tmp_path, sessions=[])
    try:
        win._manual_process_name = "ms-teams.exe"
        win._refresh_app_picker()
        qapp.processEvents()
        monkeypatch.setattr(win, "_wait_for_target_session",
                            lambda exe: False)
        monkeypatch.setattr(win, "_target_process_running",
                            lambda exe: True)
        asked: list = []

        def _cancel(name, **kwargs):
            asked.append(kwargs)
            return False

        monkeypatch.setattr(
            win, "_confirm_record_without_session", _cancel)
        win.on_record()
        qapp.processEvents()
        assert asked == [{"target_running": True}]
        assert win.capture is None
    finally:
        win.close()


def test_record_finds_session_in_app_process_tree(
        qapp, tmp_path, monkeypatch):
    """The rendering session belongs to a child process with a
    different exe (Teams' WebView2 host): the probe must still
    find it, with no dialog and per-app capture attempted.

    The plain exe-name match reports nothing here — only the
    process-tree search finds the session. This is the Nov 24
    2026 live-call case (session active, sound playing).
    """
    import app.ui.main_window as mw

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(mw, "audio_sessions_available",
                        lambda: (True, None))
    monkeypatch.setattr(mw, "DualCapture", _FakeCapture)
    renderer = AudioSessionInfo("msedgewebview2.exe", "Teams", 300,
                                "active", 0.5)
    win = _make_window(monkeypatch, tmp_path, sessions=[renderer])
    try:
        win._manual_process_name = "ms-teams.exe"
        win._refresh_app_picker()
        qapp.processEvents()
        # Pick the Teams row explicitly: the first session row is
        # the renderer child, and the default selection keeps it.
        win.process_combo.setCurrentIndex(
            win._picker_index_for("ms-teams.exe"))
        qapp.processEvents()
        assert win.selected_process_name() == "ms-teams.exe"

        class _FakeProc:
            def __init__(self, pid, name, ppid):
                self.info = {"pid": pid, "name": name, "ppid": ppid}

        def _fake_process_iter(attrs):
            return [
                _FakeProc(1, "explorer.exe", 0),
                _FakeProc(100, "ms-teams.exe", 1),
                _FakeProc(300, "msedgewebview2.exe", 100),
            ]

        monkeypatch.setattr("psutil.process_iter", _fake_process_iter)

        def _must_not_ask(name, **kwargs):
            raise AssertionError(
                "dialog must not appear: the renderer child holds "
                "the session for the picked app")

        monkeypatch.setattr(
            win, "_confirm_record_without_session", _must_not_ask)
        win.on_record()
        qapp.processEvents()
        assert isinstance(win.capture, _FakeCapture)
        assert win.capture.capture_system_mode == "auto"
        assert win.capture.process_name == "ms-teams.exe"
    finally:
        win.close()


def test_broken_session_probe_fails_open(qapp, tmp_path, monkeypatch):
    """A probe exception never blocks a recording (no wait, no dialog)."""
    import app.ui.main_window as mw

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(mw, "audio_sessions_available",
                        lambda: (True, None))
    monkeypatch.setattr(mw, "DualCapture", _FakeCapture)
    win = _make_window(monkeypatch, tmp_path, sessions=[])
    try:
        win._manual_process_name = "ms-teams.exe"
        win._refresh_app_picker()
        qapp.processEvents()

        def _boom(**kwargs):
            raise RuntimeError("enumerator broken")

        monkeypatch.setattr(mw, "list_audio_sessions", _boom)
        assert win._target_has_audio_session("ms-teams.exe") is True
        assert win._wait_for_target_session("ms-teams.exe") is True
        def _must_not_ask(name, **kwargs):
            raise AssertionError("dialog must not appear: "
                                 "a broken probe fails open")

        monkeypatch.setattr(
            win, "_confirm_record_without_session", _must_not_ask)
        win.on_record()
        qapp.processEvents()
        assert isinstance(win.capture, _FakeCapture)
    finally:
        win.close()


# ------------------------------------------------------ silence warnings ---
def test_system_silent_while_mic_live_helper():
    from app.ui.main_window import system_silent_while_mic_live

    assert system_silent_while_mic_live([0.0, 0.001], [0.2]) is True
    assert system_silent_while_mic_live([0.005], [0.01]) is True
    assert system_silent_while_mic_live([0.0], [0.0]) is False
    assert system_silent_while_mic_live([0.5], [0.2]) is False
    assert system_silent_while_mic_live([], [0.2]) is False
    assert system_silent_while_mic_live([0.0], []) is False


def test_silence_timer_warns_and_clears(qapp, tmp_path, monkeypatch):
    win = _make_window(monkeypatch, tmp_path, sessions=[])
    try:
        win.capture = _FakeCapture(tmp_path)
        win.capture.state = CaptureState.RECORDING
        win.status_label.setText("● Recording… ")
        for _ in range(5):
            win._note_level("system", 0.0)
            win._note_level("mic", 0.3)
        win._check_system_silence()
        assert "No system audio detected" in win.status_label.text()
        assert "target app is playing sound" in win.status_label.text()
        # System audio comes back: the warning clears itself.
        win._note_level("system", 0.5)
        win._check_system_silence()
        assert "No system audio detected" not in win.status_label.text()
        assert win.status_label.text().startswith("● Recording")
    finally:
        win.close()


def test_silence_check_ignores_quiet_meeting(qapp, tmp_path, monkeypatch):
    # Mic silent too (nobody speaking): no warning — a quiet
    # meeting is not a broken recording.
    win = _make_window(monkeypatch, tmp_path, sessions=[])
    try:
        win.capture = _FakeCapture(tmp_path)
        win.capture.state = CaptureState.RECORDING
        win.status_label.setText("● Recording… ")
        for _ in range(5):
            win._note_level("system", 0.0)
            win._note_level("mic", 0.0)
        win._check_system_silence()
        assert win.status_label.text() == "● Recording… "
        # And a non-recording capture never warns either.
        win.capture.state = CaptureState.IDLE
        for _ in range(5):
            win._note_level("mic", 0.3)
        win._check_system_silence()
        assert win.status_label.text() == "● Recording… "
    finally:
        win.close()
