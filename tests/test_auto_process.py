"""Qt-gated tests — auto-process pipeline wiring (Nov 8 2026).

After a recording stops, the window must run transcribe →
diarize → suggest/identify in order, in background workers, and
a failed step must surface loudly without losing the recording.
The workers are faked here (no Whisper / sherpa models); the
wiring — chain order, auto flag, error stop, recording saved —
is what is under test. Run with the gui-shots venv::

    QT_QPA_PLATFORM=offscreen \
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \
      -m pytest tests/test_auto_process.py -q
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

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


def _make_fake_workers(monkeypatch, calls: list, fail_step: str | None = None):
    """Replace the three workers with fakes that finish immediately.

    ``calls`` records run() order. ``fail_step`` ("transcribe" /
    "diarize" / "suggest") makes that worker emit error instead.
    """
    from PyQt6.QtCore import QObject, pyqtSignal
    import app.ui.main_window as mw

    def _fake_worker(name, finished_payload):
        class FakeWorker(QObject):
            progress = pyqtSignal(float, str)
            finished = pyqtSignal(dict)
            error = pyqtSignal(str)

            def __init__(self, *args, **kwargs):
                super().__init__()
                self._args = args
                self._kwargs = kwargs
                # The real _SuggestWorker sets .suggestions in run();
                # _on_suggest_finished reads it, so the fake must too
                # (a missing attribute raises in the slot, and in
                # PyQt6 that aborts the process).
                self.suggestions: list = []

            def run(self):
                calls.append(name)
                if fail_step == name:
                    self.error.emit(f"fake {name} failure")
                else:
                    self.finished.emit(dict(finished_payload))

        return FakeWorker

    monkeypatch.setattr(
        mw, "_TranscribeWorker",
        _fake_worker("transcribe", {
            "segments": 3, "wall_seconds": 1.0,
            "real_time_factor": 0.1, "model": "small",
            "session_dir": ""}))
    monkeypatch.setattr(
        mw, "_DiarizeWorker",
        _fake_worker("diarize", {
            "segments": 3, "speakers": ["SPEAKER_00"], "mode": "sherpa",
            "method": "sherpa-onnx", "wall_seconds": 1.0,
            "session_dir": ""}))
    monkeypatch.setattr(
        mw, "_SuggestWorker",
        _fake_worker("suggest", {
            "suggestions": [],
            "auto_identify": {"applied": [],
                              "enrolled": [{"label": "SPEAKER_00",
                                            "name": "Speaker A"}],
                              "left_unknown": []},
            "session_dir": ""}))


class _StubCapture:
    """A stopped capture: stop() returns the saved files."""

    def __init__(self, session_dir: Path, duration: float = 30.0):
        self.session_dir = session_dir
        self.state = CaptureState.STOPPED
        self.duration_seconds = duration

    def stop(self):
        return {"combined": self.session_dir / "combined.wav",
                "mic": self.session_dir / "mic.wav",
                "system": self.session_dir / "system.wav"}


def _make_stopped_session(tmp_path: Path) -> Path:
    session = tmp_path / "2026-11-08_10-00-00_teams"
    session.mkdir(parents=True)
    (session / "metadata.json").write_text(json.dumps({
        "session_id": session.name, "app_name": "Teams",
        "mode": "system-loopback-fallback", "status": "recording",
        "start_time": "2026-11-08T10:00:00+00:00",
        "sample_rate": 48000, "channels": 1,
        "devices": {}, "warning": None, "files": {}}), encoding="utf-8")
    for name in ("combined.wav", "mic.wav", "system.wav"):
        (session / name).write_bytes(b"RIFF" + b"\x00" * 100)
    return session


def _pump_until(qapp, condition, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_stop_runs_auto_pipeline_in_order(qapp, tmp_path, monkeypatch):
    from app.ui.main_window import create_main_window

    calls: list = []
    _make_fake_workers(monkeypatch, calls)
    session = _make_stopped_session(tmp_path)
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        assert win.config.auto_process_recordings is True
        win.session_dir = session
        win.capture = _StubCapture(session)
        win.on_stop()
        assert _pump_until(
            qapp, lambda: calls == ["transcribe", "diarize", "suggest"]
                         and not win._auto_processing)
        assert calls == ["transcribe", "diarize", "suggest"]
        assert not win._auto_processing
        # The auto summary names the enrolled identity.
        assert "Auto-processing done" in win.status_label.text()
        assert "Speaker A" in win.status_label.text()
        # The recording itself is untouched by the pipeline.
        assert (session / "combined.wav").exists()
        assert (session / "metadata.json").exists()
        assert json.loads(
            (session / "metadata.json").read_text(
                encoding="utf-8"))["status"] == "completed"
    finally:
        win.close()


def test_auto_pipeline_failure_stops_loudly_and_keeps_recording(
        qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from app.ui.main_window import create_main_window

    calls: list = []
    _make_fake_workers(monkeypatch, calls, fail_step="transcribe")
    errors: list = []
    monkeypatch.setattr(
        QMessageBox, "critical",
        lambda *args, **kwargs: errors.append(args[2] if len(args) > 2
                                             else ""))
    session = _make_stopped_session(tmp_path)
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.session_dir = session
        win.capture = _StubCapture(session)
        win.on_stop()
        assert _pump_until(
            qapp, lambda: calls == ["transcribe"]
                         and not win._auto_processing and len(errors) == 1)
        # The chain stopped at the failed step — no diarize/suggest.
        assert calls == ["transcribe"]
        assert not win._auto_processing
        assert len(errors) == 1 and "fake transcribe failure" in errors[0]
        assert "Transcription failed" in win.transcribe_label.text()
        # The recording was saved before the pipeline started.
        assert (session / "combined.wav").exists()
        assert json.loads(
            (session / "metadata.json").read_text(
                encoding="utf-8"))["status"] == "completed"
    finally:
        win.close()


def test_auto_pipeline_off_when_setting_disabled(
        qapp, tmp_path, monkeypatch):
    from app.ui.main_window import create_main_window

    calls: list = []
    _make_fake_workers(monkeypatch, calls)
    session = _make_stopped_session(tmp_path)
    win = create_main_window(
        AppConfig(output_root=tmp_path, auto_process_recordings=False))
    try:
        win.session_dir = session
        win.capture = _StubCapture(session)
        win.on_stop()
        # S1: Stop still finalizes the transcript (here the
        # full-file fallback — there is no live state), but the
        # diarize → suggest chain stays gated on the setting.
        assert _pump_until(
            qapp, lambda: calls == ["transcribe"]
                         and not win._transcribing)
        assert calls == ["transcribe"]
        assert not win._auto_processing
        # The manual button is still there for the user.
        assert win.transcribe_btn.isEnabled()
    finally:
        win.close()


def test_settings_dialog_model_combo_defaults_to_base(qapp, tmp_path):
    """Settings dialog shows base for a fresh config (Nov 19 2026).

    Nov 20 2026 (Item 8): a stored "small" that was never
    explicitly picked is the 1.0.2 default, so the dialog shows
    the current default (base); only an explicit small still
    loads as small.
    """
    from app.config import AppConfig
    from app.ui.settings_dialog import SettingsDialog

    fresh = SettingsDialog(AppConfig(output_root=tmp_path))
    try:
        assert fresh.model_combo.currentData() == "base"
        assert fresh.config_from_dialog().whisper_model == "base"
    finally:
        fresh.close()
    saved_small = SettingsDialog(
        AppConfig(output_root=tmp_path, whisper_model="small"))
    try:
        assert saved_small.model_combo.currentData() == "base"
        assert saved_small.config_from_dialog().whisper_model == "base"
        assert (saved_small.config_from_dialog()
                .whisper_model_explicit is False)
    finally:
        saved_small.close()
    explicit_small = SettingsDialog(
        AppConfig(output_root=tmp_path, whisper_model="small",
                  whisper_model_explicit=True))
    try:
        assert explicit_small.model_combo.currentData() == "small"
        assert (explicit_small.config_from_dialog()
                .whisper_model_explicit is True)
    finally:
        explicit_small.close()


def test_settings_dialog_has_auto_process_checkbox(
        qapp, tmp_path, monkeypatch):
    from app.config import AppConfig
    from app.ui.settings_dialog import SettingsDialog

    cfg = AppConfig(output_root=tmp_path, auto_process_recordings=False)
    dlg = SettingsDialog(cfg)
    try:
        assert hasattr(dlg, "auto_process_check")
        assert dlg.auto_process_check.isChecked() is False
        dlg.auto_process_check.setChecked(True)
        out = dlg.config_from_dialog()
        assert out.auto_process_recordings is True
        assert out.diarization_mode == "sherpa"
        assert out.search_mode == "hybrid"
    finally:
        dlg.close()
