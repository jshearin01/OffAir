"""Qt-gated tests — sherpa is the diarization default, actually
(Nov 10 2026), and the finished progress bars hide (the "purple
bar that looks like a button but does nothing").

The two silent-simple fallbacks
(``diarize_mode_combo.currentData() or "simple"``) are covered by
test_diarize_fallback_uses_configured_default_not_simple: with the
combo at index -1 the mode that runs must be the configured
default, never "simple". Run with the gui-shots venv::

    QT_QPA_PLATFORM=offscreen \
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \
      -m pytest tests/test_ui_sherpa_default.py -q
"""
from __future__ import annotations

import inspect
import os
import time
from pathlib import Path

import pytest

from app.config import AppConfig
from app.diarization.session import diarize_session


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


def _pump_until(qapp, condition, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def _make_fake_diarize_worker(monkeypatch, seen: list):
    """Replace _DiarizeWorker with a fake that records its mode."""
    from PyQt6.QtCore import QObject, pyqtSignal
    import app.ui.main_window as mw

    class FakeWorker(QObject):
        progress = pyqtSignal(float, str)
        finished = pyqtSignal(dict)
        error = pyqtSignal(str)

        def __init__(self, *args, **kwargs):
            super().__init__()
            # _DiarizeWorker(session_dir, mode, min, max, token, ...)
            seen.append(args[1] if len(args) > 1 else kwargs.get("mode"))

        def run(self):
            self.finished.emit({
                "segments": 2, "speakers": ["SPEAKER_00"],
                "mode": seen[-1], "method": "fake",
                "wall_seconds": 1.0, "session_dir": ""})

    monkeypatch.setattr(mw, "_DiarizeWorker", FakeWorker)


# ------------------------------------------------------------ sherpa -------
def test_fresh_config_combo_defaults_to_sherpa(qapp, tmp_path):
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        assert AppConfig().diarization_mode == "sherpa"
        assert win.diarize_mode_combo.currentData() == "sherpa"
        # The description names the mode that will run.
        assert "sherpa-onnx" in win.diarize_warning_label.text()
        win.show()
        # The sherpa status label lives on the Session view — show it.
        win.show_view("Session")
        qapp.processEvents()
        assert win.sherpa_status_label.isVisible()
    finally:
        win.close()


def test_settings_dialog_round_trip_preserves_sherpa(
        qapp, tmp_path):
    from app.ui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(AppConfig(output_root=tmp_path))
    try:
        assert dlg.diarize_combo.currentData() == "sherpa"
        assert dlg.config_from_dialog().diarization_mode == "sherpa"
    finally:
        dlg.close()
    # And an explicit non-default survives the round trip too —
    # the dialog preserves choices, it does not force sherpa.
    cfg = AppConfig(output_root=tmp_path, diarization_mode="simple")
    dlg = SettingsDialog(cfg)
    try:
        assert dlg.diarize_combo.currentData() == "simple"
        assert dlg.config_from_dialog().diarization_mode == "simple"
    finally:
        dlg.close()


def test_diarize_session_default_mode_is_configured_default():
    from app.config import DIARIZATION_DEFAULT_MODE

    assert DIARIZATION_DEFAULT_MODE == "sherpa"
    param = inspect.signature(diarize_session).parameters["mode"]
    assert param.default == "sherpa"


def test_on_diarize_passes_sherpa_to_worker(qapp, tmp_path, monkeypatch):
    from app.ui.main_window import create_main_window

    seen: list = []
    _make_fake_diarize_worker(monkeypatch, seen)
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.session_dir = tmp_path
        win.on_diarize()
        assert seen == ["sherpa"]
        # Wait for the worker to finish (not just to be built) —
        # closing the window with its thread still running aborts.
        assert _pump_until(
            qapp,
            lambda: win.diarize_label.text().startswith("Done (sherpa/"))
        # The status line names the mode that actually ran.
        assert "sherpa" in win.diarize_label.text()
        assert win.diarize_label.text().startswith("Done (sherpa/")
    finally:
        win.close()


def test_diarize_fallback_uses_configured_default_not_simple(
        qapp, tmp_path, monkeypatch):
    from app.ui.main_window import create_main_window

    seen: list = []
    _make_fake_diarize_worker(monkeypatch, seen)
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        # No selection at all (index -1, currentData() None): the
        # mode that runs must be the configured default, never a
        # silent downgrade to "simple".
        win.diarize_mode_combo.setCurrentIndex(-1)
        qapp.processEvents()
        assert win.diarize_mode_combo.currentData() is None
        assert "sherpa-onnx" in win.diarize_warning_label.text()
        win.session_dir = tmp_path
        win.on_diarize()
        assert seen == ["sherpa"]
        assert "simple" not in seen
        assert _pump_until(
            qapp,
            lambda: win.diarize_label.text().startswith("Done (sherpa/"))
    finally:
        win.close()


def test_diarize_fallback_follows_config_when_simple_configured(
        qapp, tmp_path, monkeypatch):
    # The fallback is the *configured* default: with simple
    # configured, index -1 must run simple (not sherpa).
    from app.ui.main_window import create_main_window

    seen: list = []
    _make_fake_diarize_worker(monkeypatch, seen)
    win = create_main_window(
        AppConfig(output_root=tmp_path, diarization_mode="simple"))
    try:
        win.diarize_mode_combo.setCurrentIndex(-1)
        qapp.processEvents()
        win.session_dir = tmp_path
        win.on_diarize()
        assert seen == ["simple"]
        assert _pump_until(
            qapp,
            lambda: win.diarize_label.text().startswith("Done (simple/"))
    finally:
        win.close()


# -------------------------------------------------------- purple bar -------
def _show_session_view(qapp, win):
    win.show()
    win.show_view("Session")
    qapp.processEvents()


def test_progress_bars_hide_after_finish(qapp, tmp_path):
    """The finished bars hide: a full-width indigo bar left visible
    at 100% is the "purple bar that looks like a button but does
    nothing" (Nov 10 2026). Verified by rendering the Session view
    offscreen after simulated finishes."""
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        _show_session_view(qapp, win)
        # Transcribe.
        win.transcribe_progress.setVisible(True)
        win.transcribe_progress.setValue(50)
        win._on_transcribe_finished({
            "segments": 3, "wall_seconds": 1.0,
            "real_time_factor": 0.1, "model": "small",
            "session_dir": str(tmp_path)})
        qapp.processEvents()
        assert win.transcribe_progress.isVisible() is False
        assert "Done:" in win.transcribe_label.text()
        # Diarize.
        win.diarize_progress.setVisible(True)
        win.diarize_progress.setValue(50)
        win._on_diarize_finished({
            "segments": 2, "speakers": ["SPEAKER_00"], "mode": "sherpa",
            "method": "sherpa-onnx", "wall_seconds": 1.0,
            "session_dir": str(tmp_path)})
        qapp.processEvents()
        assert win.diarize_progress.isVisible() is False
        assert win.diarize_label.text().startswith("Done (sherpa/")
        # Near-live.
        win._nearlive_worker = None
        win.nearlive_progress.setVisible(True)
        win.nearlive_progress.setValue(50)
        win._on_nearlive_finished({
            "chunks": 2, "segments": 3, "wall_seconds": 1.0,
            "session_dir": str(tmp_path)})
        qapp.processEvents()
        assert win.nearlive_progress.isVisible() is False
        assert "Done:" in win.nearlive_label.text()
    finally:
        win.close()
