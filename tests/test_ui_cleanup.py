"""Qt-gated tests — Transcript tab toggle / Re-clean / Polish wiring
and the auto-process chain producing a cleaned artifact (Nov 10 2026).

Run with the gui-shots venv::

    QT_QPA_PLATFORM=offscreen \
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \
      -m pytest tests/test_ui_cleanup.py -q
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from app.audio.capture import CaptureState
from app.config import AppConfig
from app.transcription.names_dictionary import NamesDictionary


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


def _write_session_with_transcript(session: Path) -> None:
    """A session with a verbatim transcript that cleanup changes."""
    session.mkdir(parents=True)
    (session / "metadata.json").write_text(json.dumps({
        "session_id": session.name, "app_name": "Teams",
        "mode": "system-loopback-fallback", "status": "completed",
        "start_time": "2026-11-10T10:00:00+00:00",
        "sample_rate": 48000, "channels": 1,
        "devices": {}, "warning": None, "files": {}}), encoding="utf-8")
    payload = {"model": "tiny", "device": "cpu", "compute_type": "int8",
               "language": "en", "audio_file": "combined.wav",
               "audio_duration_seconds": 9.0, "wall_seconds": 1.0,
               "real_time_factor": 0.1,
               "segments": [
                   {"start": 0.0, "end": 2.5,
                    "text": "Um, good morning everyone.",
                    "speaker": "SPEAKER_00"},
                   {"start": 2.5, "end": 5.0,
                    "text": "I think— I think the the Meat Trace "
                            "demo was good.",
                    "speaker": "SPEAKER_01"},
                   {"start": 5.0, "end": 7.5,
                    "text": "Uh, Marcus, please confirm the list.",
                    "speaker": "SPEAKER_00"},
               ]}
    (session / "transcript.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (session / "transcript.txt").write_text(
        "[00:00] Um, good morning everyone.\n"
        "[00:02] I think— I think the the Meat Trace demo was good.\n"
        "[00:05] Uh, Marcus, please confirm the list.\n",
        encoding="utf-8")
    for name in ("combined.wav", "mic.wav", "system.wav"):
        (session / name).write_bytes(b"RIFF" + b"\x00" * 100)


def _list_texts(win) -> list[str]:
    return [win.segments_list.item(i).text()
            for i in range(win.segments_list.count())]


def _seeded_dictionary(monkeypatch):
    """Point the cleanup module's dictionary load at the seed."""
    import app.transcription.cleanup as cleanup
    monkeypatch.setattr(
        cleanup, "load_names_dictionary",
        lambda path=None: NamesDictionary(
            names=["John", "Priya", "Marcus"],
            corrections={"Meat Trace": "MeetTrace"}))


# ------------------------------------------------------- toggle + re-clean --
def test_transcript_toggle_and_reclean(qapp, tmp_path, monkeypatch):
    from app.ui.main_window import create_main_window

    _seeded_dictionary(monkeypatch)
    session = tmp_path / "2026-11-10_10-00-00_teams"
    _write_session_with_transcript(session)
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.load_session(session)
        # Default view is Original (verbatim).
        assert win.transcript_view_combo.currentData() == "original"
        texts = _list_texts(win)
        assert any("Um, good morning" in t for t in texts)
        assert any("Meat Trace" in t for t in texts)
        # Cleaned view with no cleaned artifact yet: placeholder.
        win.transcript_view_combo.setCurrentIndex(
            win.transcript_view_combo.findData("cleaned"))
        qapp.processEvents()
        texts = _list_texts(win)
        assert len(texts) == 1 and "Re-clean" in texts[0]
        # Re-clean (Transcript tab button): writes the artifact and
        # the list now shows the cleaned text.
        assert win.reclean_btn.isEnabled()
        win.reclean_btn.click()
        qapp.processEvents()
        assert (session / "transcript-clean.json").exists()
        assert (session / "transcript-clean.txt").exists()
        texts = _list_texts(win)
        assert any("Good morning everyone." in t for t in texts)
        assert any("MeetTrace" in t for t in texts)
        assert not any("Meat Trace" in t for t in texts)
        assert not any("Um," in t or "Uh," in t for t in texts)
        assert "2 filler" in win.cleanup_label.text()
        # Back to Original: the verbatim text is back, untouched.
        win.transcript_view_combo.setCurrentIndex(
            win.transcript_view_combo.findData("original"))
        qapp.processEvents()
        texts = _list_texts(win)
        assert any("Um, good morning" in t for t in texts)
        assert any("Meat Trace" in t for t in texts)
        # And the verbatim file is byte-identical to what we wrote.
        payload = json.loads(
            (session / "transcript.json").read_text(encoding="utf-8"))
        assert payload["segments"][1]["text"] == (
            "I think— I think the the Meat Trace demo was good.")
    finally:
        win.close()


def test_polish_button_needs_ollama_and_reports_clearly(
        qapp, tmp_path, monkeypatch):
    """Polish with no Ollama running fails with a clear message and
    leaves the cleaned version in place (no modal swallowing: the
    dialog is stubbed, the label carries the message)."""
    from PyQt6.QtWidgets import QMessageBox
    from app.ui.main_window import create_main_window

    _seeded_dictionary(monkeypatch)
    session = tmp_path / "2026-11-10_10-00-00_teams"
    _write_session_with_transcript(session)
    win = create_main_window(AppConfig(output_root=tmp_path))
    errors: list = []
    monkeypatch.setattr(
        QMessageBox, "critical",
        lambda *args, **kwargs: errors.append(args[2] if len(args) > 2
                                             else ""))
    try:
        win.load_session(session)
        win.reclean_btn.click()
        qapp.processEvents()
        assert (session / "transcript-clean.txt").exists()
        win.polish_btn.click()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            qapp.processEvents()
            if errors or "Polish failed" in win.cleanup_label.text():
                break
            time.sleep(0.05)
        assert not (session / "transcript-polished.md").exists()
        assert (session / "transcript-clean.txt").exists()
        assert len(errors) == 1
        assert "Ollama is not running" in errors[0]
        assert "Ollama is not running" in win.cleanup_label.text()
    finally:
        win.close()


# ------------------------------------------------- settings dialog wiring --
def test_settings_dialog_names_dictionary_and_polish(
        qapp, tmp_path, monkeypatch):
    import app.ui.settings_dialog as sd
    from app.ui.settings_dialog import SettingsDialog

    saved: dict = {}
    monkeypatch.setattr(
        sd, "load_names_dictionary",
        lambda: NamesDictionary(
            names=["John"], corrections={"Meat Trace": "MeetTrace"}))
    monkeypatch.setattr(
        sd, "save_names_dictionary",
        lambda d, path=None: saved.setdefault("dictionary", d))
    monkeypatch.setattr(
        sd, "save_settings",
        lambda cfg, path=None: saved.setdefault("config", cfg))
    cfg = AppConfig(output_root=tmp_path, transcript_polish_enabled=False)
    dlg = SettingsDialog(cfg)
    try:
        # The editor shows the loaded dictionary in editor format.
        assert "John" in dlg.names_edit.toPlainText()
        assert "Meat Trace -> MeetTrace" in dlg.names_edit.toPlainText()
        assert hasattr(dlg, "polish_check")
        assert dlg.polish_check.isChecked() is False
        # Edit both, apply: config + dictionary are saved.
        dlg.names_edit.setPlainText("Priya\nMarcus\nMeat Trace -> MeetTrace\n")
        dlg.polish_check.setChecked(True)
        out = dlg.apply_and_save()
        assert out.transcript_polish_enabled is True
        assert saved["config"].transcript_polish_enabled is True
        assert saved["dictionary"].names == ["Priya", "Marcus"]
        assert saved["dictionary"].corrections == {
            "Meat Trace": "MeetTrace"}
    finally:
        dlg.close()


# ------------------------------------------- auto chain produces the artifact --
def _make_fake_workers(monkeypatch, calls: list):
    from PyQt6.QtCore import QObject, pyqtSignal
    import app.ui.main_window as mw

    def _fake_worker(name, finished_payload):
        class FakeWorker(QObject):
            progress = pyqtSignal(float, str)
            finished = pyqtSignal(dict)
            error = pyqtSignal(str)

            def __init__(self, *args, **kwargs):
                super().__init__()
                self.suggestions: list = []

            def run(self):
                calls.append(name)
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
            "auto_identify": {"applied": [], "enrolled": [],
                              "left_unknown": []},
            "session_dir": ""}))


class _StubCapture:
    def __init__(self, session_dir: Path, duration: float = 30.0):
        self.session_dir = session_dir
        self.state = CaptureState.STOPPED
        self.duration_seconds = duration

    def stop(self):
        return {"combined": self.session_dir / "combined.wav",
                "mic": self.session_dir / "mic.wav",
                "system": self.session_dir / "system.wav"}


def _pump_until(qapp, condition, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_auto_pipeline_produces_cleaned_transcript(
        qapp, tmp_path, monkeypatch):
    """The auto-process chain (transcribe → diarize → suggest)
    produces transcript-clean.json/.txt for the new session, with
    the dictionary correction applied — without user action."""
    from app.ui.main_window import create_main_window

    _seeded_dictionary(monkeypatch)
    calls: list = []
    _make_fake_workers(monkeypatch, calls)
    session = tmp_path / "2026-11-10_10-00-00_teams"
    _write_session_with_transcript(session)
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.session_dir = session
        win.capture = _StubCapture(session)
        win.on_stop()
        assert _pump_until(
            qapp, lambda: calls == ["transcribe", "diarize", "suggest"]
                         and not win._auto_processing)
        assert calls == ["transcribe", "diarize", "suggest"]
        clean = session / "transcript-clean.json"
        assert clean.exists()
        assert (session / "transcript-clean.txt").exists()
        payload = json.loads(clean.read_text(encoding="utf-8"))
        texts = [s["text"] for s in payload["segments"]]
        assert any("MeetTrace" in t for t in texts)
        assert not any("Meat Trace" in t for t in texts)
        assert not any("Um," in t or "Uh," in t for t in texts)
        assert payload["cleaning"]["name_corrections_applied"] == 1
        assert payload["cleaning"]["fillers_removed"] == 2
        # Verbatim untouched by the whole chain.
        verbatim = json.loads(
            (session / "transcript.json").read_text(encoding="utf-8"))
        assert verbatim["segments"][1]["text"] == (
            "I think— I think the the Meat Trace demo was good.")
    finally:
        win.close()
