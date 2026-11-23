"""Qt tests for scheduled recordings (Nov 22 2026).

Covers the GUI card (render / add / cancel), the live reload of
MCP-written schedules, expire-on-startup, the real 2 s timer
firing a due schedule through the Record path, the
already-recording skip, and the invariant that a scheduled
recording is never auto-stopped (the watcher must not end it —
only the user's Stop does). Store/MCP behaviour is covered in
tests/test_schedules.py and tests/test_mcp.py; these tests
exercise the window wiring only. No dialog is ever shown from
the firing path, so QMessageBox is stubbed out of caution: a
modal dialog in an offscreen run would hang the suite.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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


class _FakeDualCapture:
    """Capture stand-in: start/stop flip state, no audio hardware."""

    def __init__(self, *args, **kwargs) -> None:
        from app.audio.capture import CaptureState
        self.state = CaptureState.IDLE
        self.mode = "system-loopback-fallback"
        self.warning = None
        self.duration_seconds = 5.0
        self.init_kwargs = dict(kwargs)

    def start(self) -> None:
        from app.audio.capture import CaptureState
        self.state = CaptureState.RECORDING

    def stop(self):
        from app.audio.capture import CaptureState
        self.state = CaptureState.STOPPED
        return {}


def _make_window(qapp, tmp_path, monkeypatch, *,
                 capture_system_mode: str = "auto"):
    import app.ui.main_window as mw
    monkeypatch.setattr(mw, "DualCapture", _FakeDualCapture)
    # No modal dialog may ever appear in these tests — a dialog
    # in an offscreen run hangs the suite. Record the calls.
    calls = []
    monkeypatch.setattr(
        mw.QMessageBox, "warning",
        lambda *args, **kwargs: calls.append(("warning", args)))
    monkeypatch.setattr(
        mw.QMessageBox, "critical",
        lambda *args, **kwargs: calls.append(("critical", args)))
    config = AppConfig(output_root=tmp_path,
                       auto_process_recordings=False,
                       capture_system_mode=capture_system_mode)
    win = mw.MainWindow(
        config, schedules_path=tmp_path / "schedules.json")
    win._dialog_calls = calls
    return win


def _row_texts(win):
    return [win.schedule_list.item(i).text()
            for i in range(win.schedule_list.count())]


def _pump_until(qapp, predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.1)
    qapp.processEvents()
    return predicate()


# ------------------------------------------------------------ rendering ----
def test_schedule_card_renders_empty_state(qapp, tmp_path, monkeypatch):
    win = _make_window(qapp, tmp_path, monkeypatch)
    try:
        win.show()
        qapp.processEvents()
        assert _row_texts(win) == ["No recordings scheduled"]
        assert win.schedule_cancel_btn.isEnabled() is False
        assert win.schedule_list.isVisible()
        assert "Teams' recording banner" in win.schedule_hint_label.text()
        assert win._schedule_timer.isActive()
        assert win._schedule_timer.interval() == 2000
    finally:
        win.close()


def test_schedule_row_text_shows_time_app_subject(
        qapp, tmp_path, monkeypatch):
    from app.storage.schedules import create_schedule
    store = tmp_path / "schedules.json"
    when = datetime.now(timezone.utc) + timedelta(hours=2)
    create_schedule(when, "ms-teams.exe", subject="Standup",
                    created_by="mcp", path=store)
    win = _make_window(qapp, tmp_path, monkeypatch)
    try:
        rows = _row_texts(win)
        assert len(rows) == 1
        assert "Standup" in rows[0]
        assert "Teams" in rows[0]
        assert when.astimezone().strftime("%H:%M") in rows[0]
    finally:
        win.close()


# ------------------------------------------------------------ add / cancel -
def test_add_and_cancel_flow(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QDateTime
    from app.storage.schedules import load_schedules
    win = _make_window(qapp, tmp_path, monkeypatch)
    try:
        win.schedule_datetime.setDateTime(
            QDateTime.currentDateTime().addSecs(7200))
        win.schedule_subject_edit.setText("Planning")
        win._on_schedule_add()
        schedules = load_schedules(tmp_path / "schedules.json")
        assert len(schedules) == 1
        assert schedules[0].status == "pending"
        assert schedules[0].created_by == "gui"
        rows = _row_texts(win)
        assert len(rows) == 1 and "Planning" in rows[0]
        # Cancel: select the row, press Cancel.
        win.schedule_list.setCurrentRow(0)
        qapp.processEvents()
        assert win.schedule_cancel_btn.isEnabled() is True
        win._on_schedule_cancel()
        schedules = load_schedules(tmp_path / "schedules.json")
        assert schedules[0].status == "cancelled"
        assert "cancelled" in _row_texts(win)[0]
        assert win.schedule_cancel_btn.isEnabled() is False
    finally:
        win.close()


def test_add_past_time_is_rejected_without_creating(
        qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QDateTime
    from app.storage.schedules import load_schedules
    win = _make_window(qapp, tmp_path, monkeypatch)
    try:
        win.schedule_datetime.setDateTime(
            QDateTime.currentDateTime().addSecs(-3600))
        win._on_schedule_add()
        assert load_schedules(tmp_path / "schedules.json") == []
        assert win._dialog_calls, "a warning dialog should have fired"
        assert win._dialog_calls[0][0] == "warning"
    finally:
        win.close()


# ------------------------------------------------------------ live reload ---
def test_mcp_written_schedule_appears_without_restart(
        qapp, tmp_path, monkeypatch):
    from app.storage.schedules import create_schedule
    win = _make_window(qapp, tmp_path, monkeypatch)
    try:
        assert _row_texts(win) == ["No recordings scheduled"]
        # Simulate the MCP server process writing the shared file.
        create_schedule(datetime.now(timezone.utc) + timedelta(hours=3),
                        "ms-teams.exe", subject="MCP one",
                        created_by="mcp",
                        path=tmp_path / "schedules.json")
        assert _pump_until(
            qapp, lambda: any("MCP one" in t for t in _row_texts(win)),
            timeout=6.0)
    finally:
        win.close()


def test_past_due_schedule_expires_on_startup(
        qapp, tmp_path, monkeypatch):
    from app.storage.schedules import create_schedule, load_schedules
    store = tmp_path / "schedules.json"
    # create_schedule refuses the past, so create a future one and
    # then move its fire time back: this is the "the app was
    # closed when the time passed" case.
    create_schedule(datetime.now(timezone.utc) + timedelta(hours=1),
                    "ms-teams.exe", created_by="mcp", path=store)
    import json
    records = json.loads(store.read_text(encoding="utf-8"))
    records["schedules"][0]["fire_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    records["schedules"][0]["status"] = "pending"
    store.write_text(json.dumps(records), encoding="utf-8")
    win = _make_window(qapp, tmp_path, monkeypatch)
    try:
        assert load_schedules(store)[0].status == "expired"
        assert "expired" in _row_texts(win)[0]
    finally:
        win.close()


# ------------------------------------------------------------ firing -------
def test_due_schedule_fires_through_record_path(
        qapp, tmp_path, monkeypatch):
    from app.storage.schedules import create_schedule, load_schedules
    from app.storage.sessions import load_metadata
    win = _make_window(qapp, tmp_path, monkeypatch)
    try:
        create_schedule(datetime.now(timezone.utc) + timedelta(seconds=3),
                        "ms-teams.exe", subject="Fired one",
                        created_by="mcp",
                        path=tmp_path / "schedules.json")
        from app.audio.capture import CaptureState
        assert _pump_until(
            qapp,
            lambda: (win.capture is not None
                     and win.capture.state == CaptureState.RECORDING),
            timeout=10.0), "schedule did not fire via the timer"
        # The watcher must never own it: auto flag stays False.
        assert win._auto_recording is False
        assert win._scheduled_recording_id is not None
        assert "Scheduled recording" in win.status_label.text()
        # The session metadata carries the schedule tag.
        metadata = load_metadata(win.session_dir)
        assert metadata["scheduled"]["schedule_id"] == \
            win._scheduled_recording_id
        assert metadata["scheduled"]["subject"] == "Fired one"
        # The store says fired, with the session id attached.
        schedules = load_schedules(tmp_path / "schedules.json")
        assert schedules[0].status == "fired"
        assert schedules[0].session_id == win.session_dir.name
        assert win._dialog_calls == [], \
            f"firing showed a dialog: {win._dialog_calls}"
        # Stop it manually (the only way a scheduled recording
        # ends) and confirm the stop path completes: the worker
        # finishes, the session is saved, and the capture sits in
        # STOPPED (a successful stop keeps the stopped object —
        # on_record builds a fresh capture next time).
        win.on_stop()
        assert _pump_until(
            qapp,
            lambda: (not win._stopping
                     and win.session_label.text().startswith("Saved:")),
            timeout=10.0)
        assert win.capture.state == CaptureState.STOPPED
        assert win._scheduled_recording_id is None
    finally:
        win.close()


def test_due_schedule_skipped_when_already_recording(
        qapp, tmp_path, monkeypatch):
    from app.audio.capture import CaptureState
    from app.storage.schedules import create_schedule, load_schedules
    from app.storage.sessions import create_session
    win = _make_window(qapp, tmp_path, monkeypatch)
    try:
        session = create_session(tmp_path, app_name="Teams")
        capture = _FakeDualCapture()
        capture.start()
        win.capture = capture
        win.session_dir = session
        create_schedule(datetime.now(timezone.utc) + timedelta(seconds=1),
                        "ms-teams.exe", created_by="mcp",
                        path=tmp_path / "schedules.json")
        assert _pump_until(
            qapp,
            lambda: load_schedules(
                tmp_path / "schedules.json")[0].status == "skipped",
            timeout=8.0)
        # The running recording was never touched.
        assert win.capture is capture
        assert capture.state == CaptureState.RECORDING
        assert win._dialog_calls == []
    finally:
        win.capture = None
        win.close()


def test_meeting_ended_never_stops_scheduled_recording(
        qapp, tmp_path, monkeypatch):
    from app.audio.capture import CaptureState
    win = _make_window(qapp, tmp_path, monkeypatch)
    try:
        win.capture = _FakeDualCapture()
        win.capture.start()
        win._auto_recording = False
        win._scheduled_recording_id = "sched-123"
        win._on_meeting_ended()
        assert win.capture.state == CaptureState.RECORDING
    finally:
        win.capture = None
        win.close()


def test_scheduled_start_forces_auto_capture_mode(
        qapp, tmp_path, monkeypatch):
    """A strict per-app setting must not silence a scheduled
    recording: the firing path uses "auto" semantics (try
    per-app, fall back to loopback with the warning)."""
    from app.audio.capture import CaptureState
    from app.storage.schedules import create_schedule
    win = _make_window(qapp, tmp_path, monkeypatch,
                       capture_system_mode="per-app")
    try:
        assert win.config.capture_system_mode == "per-app"
        create_schedule(datetime.now(timezone.utc) + timedelta(seconds=1),
                        "ms-teams.exe", created_by="mcp",
                        path=tmp_path / "schedules.json")
        assert _pump_until(
            qapp,
            lambda: (win.capture is not None
                     and win.capture.state == CaptureState.RECORDING),
            timeout=8.0)
        assert win.capture.init_kwargs["capture_system_mode"] == "auto"
    finally:
        win.capture = None
        win.close()


def test_failed_scheduled_start_marks_skipped_without_dialog(
        qapp, tmp_path, monkeypatch):
    """A start failure from the firing path never blocks on a
    dialog: the schedule is marked skipped and the failure is
    logged + status-lined."""
    import app.ui.main_window as mw
    from app.storage.schedules import create_schedule, load_schedules

    class _RaisingCapture:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("no audio device")

    win = _make_window(qapp, tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(mw, "DualCapture", _RaisingCapture)
        create_schedule(datetime.now(timezone.utc) + timedelta(seconds=1),
                        "ms-teams.exe", created_by="mcp",
                        path=tmp_path / "schedules.json")
        assert _pump_until(
            qapp,
            lambda: load_schedules(
                tmp_path / "schedules.json")[0].status == "skipped",
            timeout=8.0)
        assert win._dialog_calls == [], \
            f"firing showed a dialog: {win._dialog_calls}"
        assert "failed to start" in (
            load_schedules(tmp_path / "schedules.json")[0].note or "")
    finally:
        win.close()
