"""Tests for meeting auto-detect + auto-record (Nov 8 2026).

The watcher state machine is tested without Qt and without COM: a
fake probe returns scripted :class:`SessionSnapshot` values and a
fake clock controls time, so every gate is deterministic. The real
Windows probe never runs on this Linux VM — a real Teams meeting
start/stop is the open gate (see docs/PER_APP_CAPTURE.md).

Qt-gated tests (Settings dialog consent, AUTO badge, watcher
wiring) run offscreen with the gui-shots venv, following the
pattern in tests/test_auto_process.py.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from app.audio.auto_detect import (
    AUTO_DETECT_ACTIVE_MIC_SECONDS,
    AUTO_DETECT_INACTIVE_GRACE_SECONDS,
    AUTO_DETECT_PEAK_THRESHOLD,
    AUTO_DETECT_SILENCE_FUSE_SECONDS,
    AUTO_DETECT_SILENCE_PEAK,
    AUTO_DETECT_SUSTAIN_SECONDS,
    MeetingWatcher,
    SessionSnapshot,
    auto_detect_available,
)
from app.config import AppConfig, load_settings, save_settings


# ------------------------------------------------------------ fakes --------
class _FakeClock:
    """Deterministic clock: ``clock()`` returns ``now``; tests advance it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ScriptedProbe:
    """A probe returning whatever snapshot the test sets."""

    def __init__(self, snapshot: SessionSnapshot) -> None:
        self.snapshot = snapshot

    def __call__(self) -> SessionSnapshot:
        return self.snapshot


def _active_meeting(peak: float = 0.5, mic: bool = False) -> SessionSnapshot:
    return SessionSnapshot(process_running=True, session_state="active",
                           peak=peak, mic_claimed=mic)


def _watcher(probe, clock, **overrides):
    started: list = []
    ended: list = []
    watcher = MeetingWatcher(
        probe=probe, clock=clock,
        on_meeting_started=lambda: started.append(True),
        on_meeting_ended=lambda: ended.append(True),
        **overrides)
    return watcher, started, ended


def _drive_to_started(watcher, probe, clock) -> None:
    """Poll until the watcher fires meeting_started (or fail loudly)."""
    probe.snapshot = _active_meeting(peak=0.5)
    for _ in range(10):
        watcher.poll_once()
        clock.advance(1.0)
        if watcher.state == "in_meeting":
            return
    raise AssertionError("watcher never entered in_meeting")


# ------------------------------------------------------------ start gate ---
def test_start_gate_needs_sustained_peak():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, started, _ = _watcher(probe, clock)
    watcher.poll_once()  # t=0: first peak seen
    assert started == []
    clock.advance(1.0)
    watcher.poll_once()  # t=1: 1 s sustained < 2.5 s
    assert started == []
    clock.advance(1.0)
    watcher.poll_once()  # t=2: 2 s sustained < 2.5 s
    assert started == []
    clock.advance(1.0)
    watcher.poll_once()  # t=3: 3 s sustained >= 2.5 s
    assert started == [True]
    assert watcher.state == "in_meeting"


def test_process_running_only_never_starts():
    clock = _FakeClock()
    # Teams runs all day: process up, session inactive, no audio.
    probe = _ScriptedProbe(SessionSnapshot(
        process_running=True, session_state="inactive", peak=None))
    watcher, started, _ = _watcher(probe, clock)
    for _ in range(120):  # two minutes of "Teams is running"
        watcher.poll_once()
        clock.advance(1.0)
    assert started == []
    assert watcher.state == "idle"


def test_brief_blip_does_not_start():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.6))
    watcher, started, _ = _watcher(probe, clock)
    watcher.poll_once()  # one loud poll (a notification blip)
    probe.snapshot = _active_meeting(peak=0.0)  # then silence
    for _ in range(10):
        clock.advance(1.0)
        watcher.poll_once()
    assert started == []
    assert watcher.state == "idle"


def test_peak_below_threshold_never_starts():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=AUTO_DETECT_PEAK_THRESHOLD
                                           / 2))
    watcher, started, _ = _watcher(probe, clock)
    for _ in range(30):
        watcher.poll_once()
        clock.advance(1.0)
    assert started == []


# ------------------------------------------------------------- stop gate ---
# Revised Nov 8 2026 (user correction): silence mid-meeting NEVER
# stops a recording — the old 20 s silence rule is gone. Stop only
# on (a) the session going Inactive/Expired and STAYING that way for
# the 60 s grace, (b) the process exiting, or (c) the 10-minute
# long-silence fuse (Active, peak ≈ 0, mic unclaimed).
def test_sixty_seconds_of_silence_does_not_stop():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, _, ended = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    probe.snapshot = _active_meeting(peak=0.0)  # everyone goes quiet
    for _ in range(60):  # a full minute of silence…
        clock.advance(1.0)
        watcher.poll_once()
    assert ended == []
    assert watcher.state == "in_meeting"


def test_five_minutes_of_silence_does_not_stop():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, _, ended = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    probe.snapshot = _active_meeting(peak=0.0)
    for _ in range(300):  # five minutes, silent presenting
        clock.advance(1.0)
        watcher.poll_once()
    assert ended == []
    assert watcher.state == "in_meeting"


def test_ten_minute_silence_fuse_stops():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, _, ended = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    probe.snapshot = _active_meeting(peak=0.0)  # silent, mic unclaimed
    for _ in range(599):  # 599 s of silence < 600 s fuse
        clock.advance(1.0)
        watcher.poll_once()
    assert ended == []
    clock.advance(2.0)
    watcher.poll_once()  # 601 s of silence >= 600 s fuse
    assert ended == [True]
    assert watcher.state == "idle"


def test_silence_fuse_needs_mic_unclaimed():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, _, ended = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    # Silent render, but the mic is claimed (the user is talking
    # into a call where everyone else is muted) — still a meeting.
    probe.snapshot = _active_meeting(peak=0.0, mic=True)
    for _ in range(700):  # well past the 10-min fuse
        clock.advance(1.0)
        watcher.poll_once()
    assert ended == []
    assert watcher.state == "in_meeting"


def test_silence_fuse_resets_when_audio_resumes():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, _, ended = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    probe.snapshot = _active_meeting(peak=0.0)
    for _ in range(540):  # 9 min of silence…
        clock.advance(1.0)
        watcher.poll_once()
    probe.snapshot = _active_meeting(peak=0.4)  # …someone speaks…
    watcher.poll_once()
    probe.snapshot = _active_meeting(peak=0.0)  # …then silence again
    for _ in range(540):  # another 9 min — still under 10 continuous
        clock.advance(1.0)
        watcher.poll_once()
    assert ended == []
    for _ in range(61):
        clock.advance(1.0)
        watcher.poll_once()
    assert ended == [True]  # 10 min continuous silence since the speech


def test_single_inactive_poll_does_not_stop():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, _, ended = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    # One transient inactive poll (stream renegotiation)…
    probe.snapshot = SessionSnapshot(
        process_running=True, session_state="inactive", peak=0.0)
    watcher.poll_once()
    assert ended == []
    # …then the session is Active again: grace cancelled, still going.
    probe.snapshot = _active_meeting(peak=0.5)
    watcher.poll_once()
    assert ended == []
    assert watcher.state == "in_meeting"


def test_inactive_session_ends_meeting_after_60s_grace():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, _, ended = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    probe.snapshot = SessionSnapshot(
        process_running=True, session_state="inactive", peak=0.0)
    watcher.poll_once()  # grace starts
    for _ in range(59):
        clock.advance(1.0)
        watcher.poll_once()
    assert ended == []  # 59 s inactive < 60 s grace
    clock.advance(2.0)
    watcher.poll_once()  # 61 s inactive >= 60 s grace
    assert ended == [True]
    assert watcher.state == "idle"


def test_inactive_grace_resets_when_session_reactivates():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, _, ended = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    probe.snapshot = SessionSnapshot(
        process_running=True, session_state="inactive", peak=0.0)
    for _ in range(30):  # 30 s inactive…
        clock.advance(1.0)
        watcher.poll_once()
    probe.snapshot = _active_meeting(peak=0.5)  # …session comes back…
    watcher.poll_once()
    probe.snapshot = SessionSnapshot(
        process_running=True, session_state="inactive", peak=0.0)
    for _ in range(59):  # …goes inactive again: needs a full 60 s
        clock.advance(1.0)
        watcher.poll_once()
    assert ended == []
    clock.advance(2.0)
    watcher.poll_once()
    assert ended == [True]


def test_expired_session_ends_meeting_after_60s_grace():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, _, ended = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    probe.snapshot = SessionSnapshot(
        process_running=True, session_state="expired", peak=None)
    watcher.poll_once()  # grace starts
    for _ in range(59):
        clock.advance(1.0)
        watcher.poll_once()
    assert ended == []
    clock.advance(2.0)
    watcher.poll_once()
    assert ended == [True]


def test_process_exit_ends_meeting():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, _, ended = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    probe.snapshot = SessionSnapshot(process_running=False)
    watcher.poll_once()
    assert ended == [True]


# -------------------------------------------------------- secondary rule ---
def test_secondary_rule_active_30s_with_mic_starts():
    clock = _FakeClock()
    # All-muted meeting: session Active, peak 0, mic claimed.
    probe = _ScriptedProbe(_active_meeting(peak=0.0, mic=True))
    watcher, started, _ = _watcher(probe, clock)
    for _ in range(30):
        watcher.poll_once()
        clock.advance(1.0)
    assert started == []  # polls at t=0..29; 29 s Active < 30 s
    clock.advance(1.0)
    watcher.poll_once()  # t=30: 30 s Active with mic claimed
    assert started == [True]


def test_secondary_rule_needs_the_mic_claim():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.0, mic=False))
    watcher, started, _ = _watcher(probe, clock)
    for _ in range(120):  # two minutes, Active, silent, mic unclaimed
        watcher.poll_once()
        clock.advance(1.0)
    assert started == []  # the documented false negative


# ------------------------------------------------------- manual suppression -
def test_manual_stop_suppresses_retrigger_until_session_inactive():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, started, _ = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    assert started == [True]
    watcher.notify_manual_stop()  # the user stopped the recording
    assert watcher.state == "idle"
    # The meeting is still going — sustained peak must NOT re-trigger.
    for _ in range(30):
        watcher.poll_once()
        clock.advance(1.0)
    assert started == [True]
    # The meeting ends: session goes inactive → suppression clears.
    probe.snapshot = SessionSnapshot(
        process_running=True, session_state="inactive", peak=0.0)
    watcher.poll_once()
    # A new meeting starts later: sustained peak fires again.
    probe.snapshot = _active_meeting(peak=0.5)
    for _ in range(5):
        watcher.poll_once()
        clock.advance(1.0)
    assert started == [True, True]


def test_auto_stop_rearms_for_the_next_meeting():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, started, ended = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    # The meeting ends: session goes inactive for the full 60 s grace.
    probe.snapshot = SessionSnapshot(
        process_running=True, session_state="inactive", peak=0.0)
    for _ in range(62):
        clock.advance(1.0)
        watcher.poll_once()
    assert ended == [True]
    # Talking resumes (same meeting, new session): it re-triggers.
    probe.snapshot = _active_meeting(peak=0.5)
    for _ in range(5):
        watcher.poll_once()
        clock.advance(1.0)
    assert started == [True, True]


# ------------------------------------------------------- pre-roll hook -----
# The watcher drains an attached PreRollBuffer when a meeting starts
# and hands the buffered audio to the UI layer via take_preroll().
# The buffer itself is fed by a future always-on feeder (no feeder
# exists yet); these tests prove the drain/take contract only.
def test_watcher_drains_preroll_on_meeting_start():
    import numpy as np

    from app.audio.preroll import PreRollBuffer

    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    buf = PreRollBuffer(seconds=30.0, sample_rate=48000)
    buf.feed("mic", np.full(48000, 7, dtype=np.int16))
    watcher, started, _ = _watcher(probe, clock, preroll_buffer=buf)
    _drive_to_started(watcher, probe, clock)
    assert started == [True]
    pending = watcher.take_preroll()
    assert set(pending) == {"mic"}
    assert len(pending["mic"]) == 48000
    # Drained: the buffer is empty and a second take is empty.
    assert buf.duration_seconds("mic") == 0.0
    assert watcher.take_preroll() == {}


def test_watcher_without_preroll_buffer_takes_nothing():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, started, _ = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    assert watcher.take_preroll() == {}


def test_preroll_buffer_not_drained_before_a_meeting():
    import numpy as np

    from app.audio.preroll import PreRollBuffer

    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.0))
    buf = PreRollBuffer(seconds=30.0, sample_rate=48000)
    buf.feed("mic", np.full(48000, 7, dtype=np.int16))
    watcher, started, _ = _watcher(probe, clock, preroll_buffer=buf)
    for _ in range(10):  # silent, no meeting: nothing fires, buffer kept
        watcher.poll_once()
        clock.advance(1.0)
    assert started == []
    assert buf.duration_seconds("mic") == pytest.approx(1.0)
    assert watcher.take_preroll() == {}


# ------------------------------------------------------------- robustness --
def test_probe_error_tick_is_skipped():
    clock = _FakeClock()
    probe = _ScriptedProbe(_active_meeting(peak=0.5))
    watcher, started, ended = _watcher(probe, clock)
    _drive_to_started(watcher, probe, clock)
    # A COM hiccup mid-meeting must not end the recording.
    probe.snapshot = SessionSnapshot(
        process_running=True, session_state="active", peak=0.5,
        probe_error="enumeration failed")
    for _ in range(30):
        watcher.poll_once()
        clock.advance(1.0)
    assert ended == []
    assert watcher.state == "in_meeting"


def test_probe_exception_is_skipped():
    clock = _FakeClock()

    def _raising_probe():
        raise RuntimeError("COM exploded")

    watcher, started, ended = _watcher(_raising_probe, clock)
    for _ in range(5):
        watcher.poll_once()
        clock.advance(1.0)
    assert started == [] and ended == []


# -------------------------------------------------------------- platform ---
def test_auto_detect_unavailable_on_linux():
    import sys
    if sys.platform == "win32":
        pytest.skip("Windows-only assertion")
    available, reason = auto_detect_available()
    assert available is False
    assert "Windows" in (reason or "")


def test_watcher_start_raises_when_unavailable():
    import sys
    if sys.platform == "win32":
        pytest.skip("Windows-only assertion")
    watcher = MeetingWatcher(probe=lambda: SessionSnapshot())
    with pytest.raises(RuntimeError, match="unavailable"):
        watcher.start()


def test_gate_constants_match_the_design():
    assert AUTO_DETECT_SUSTAIN_SECONDS == pytest.approx(2.5)
    assert AUTO_DETECT_INACTIVE_GRACE_SECONDS == 60.0
    assert AUTO_DETECT_SILENCE_FUSE_SECONDS == 600.0
    assert AUTO_DETECT_ACTIVE_MIC_SECONDS == 30.0
    assert AUTO_DETECT_PEAK_THRESHOLD == 0.01
    assert AUTO_DETECT_SILENCE_PEAK == 0.001


# ------------------------------------------------------------ config -------
def test_auto_detect_defaults_off():
    assert AppConfig().auto_detect_teams is False


def test_auto_detect_round_trips(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(AppConfig(auto_detect_teams=True), path)
    assert load_settings(path).auto_detect_teams is True
    save_settings(AppConfig(auto_detect_teams=False), path)
    assert load_settings(path).auto_detect_teams is False


def test_auto_detect_missing_key_defaults_off(tmp_path):
    import json
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
    assert load_settings(path).auto_detect_teams is False


# ------------------------------------------------------------ Qt tests -----
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


def _patch_availability(monkeypatch, available: bool = True):
    """Pretend the platform supports (or lacks) auto-detect."""
    import app.ui.main_window as mw
    import app.ui.settings_dialog as sd
    reason = None if available else "Auto-detect needs Windows."
    monkeypatch.setattr(mw, "auto_detect_status",
                        lambda: (available, reason))
    monkeypatch.setattr(sd, "auto_detect_status",
                        lambda: (available, reason))


def test_settings_auto_detect_disabled_on_linux(qapp):
    import sys
    if sys.platform == "win32":
        pytest.skip("Linux-only assertion")
    from app.ui.settings_dialog import SettingsDialog
    dialog = SettingsDialog(AppConfig())
    try:
        assert dialog.auto_detect_check.isChecked() is False
        assert dialog.auto_detect_check.isEnabled() is False
    finally:
        dialog.close()


def test_settings_consent_decline_leaves_off(qapp, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    import app.ui.settings_dialog as sd
    _patch_availability(monkeypatch, available=True)
    monkeypatch.setattr(
        sd.QMessageBox, "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.No)
    dialog = sd.SettingsDialog(AppConfig(auto_detect_teams=False))
    try:
        assert dialog.auto_detect_check.isEnabled()
        dialog.auto_detect_check.setChecked(True)
        assert dialog.auto_detect_check.isChecked() is False
        assert dialog.config_from_dialog().auto_detect_teams is False
    finally:
        dialog.close()


def test_settings_consent_accept_enables(qapp, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    import app.ui.settings_dialog as sd
    _patch_availability(monkeypatch, available=True)
    asked: list = []

    def _ask(*args, **kwargs):
        asked.append(True)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(sd.QMessageBox, "question", _ask)
    dialog = sd.SettingsDialog(AppConfig(auto_detect_teams=False))
    try:
        dialog.auto_detect_check.setChecked(True)
        assert asked == [True]
        assert dialog.auto_detect_check.isChecked() is True
        assert dialog.config_from_dialog().auto_detect_teams is True
    finally:
        dialog.close()


def test_settings_consent_not_shown_when_already_enabled(
        qapp, monkeypatch):
    import app.ui.settings_dialog as sd
    _patch_availability(monkeypatch, available=True)
    monkeypatch.setattr(
        sd.QMessageBox, "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("consent must not be asked again")))
    dialog = sd.SettingsDialog(AppConfig(auto_detect_teams=True))
    try:
        assert dialog.auto_detect_check.isChecked() is True
        dialog.auto_detect_check.setChecked(False)
        dialog.auto_detect_check.setChecked(True)
        assert dialog.config_from_dialog().auto_detect_teams is True
    finally:
        dialog.close()


class _FakeWatcher:
    """Watcher stand-in: the test fires the callbacks by hand."""

    def __init__(self, *, process_names, on_meeting_started,
                 on_meeting_ended) -> None:
        self.process_names = process_names
        self.on_meeting_started = on_meeting_started
        self.on_meeting_ended = on_meeting_ended
        self.is_running = False
        self.started = False
        self.stopped = False
        self.manual_stop_calls = 0

    def start(self) -> None:
        self.is_running = True
        self.started = True

    def stop(self) -> None:
        self.is_running = False
        self.stopped = True

    def notify_manual_stop(self) -> None:
        self.manual_stop_calls += 1


class _FakeDualCapture:
    """Capture stand-in: start/stop flip state, no audio hardware."""

    def __init__(self, *args, **kwargs) -> None:
        from app.audio.capture import CaptureState
        self.state = CaptureState.IDLE
        self.mode = "system-loopback-fallback"
        self.warning = None
        self.duration_seconds = 5.0

    def start(self) -> None:
        from app.audio.capture import CaptureState
        self.state = CaptureState.RECORDING

    def stop(self):
        from app.audio.capture import CaptureState
        self.state = CaptureState.STOPPED
        return {}


def _make_window(qapp, tmp_path, monkeypatch, *, auto_detect: bool):
    import app.ui.main_window as mw
    _patch_availability(monkeypatch, available=True)
    monkeypatch.setattr(mw, "DualCapture", _FakeDualCapture)
    config = AppConfig(output_root=tmp_path,
                       auto_detect_teams=auto_detect,
                       auto_process_recordings=False)
    win = mw.MainWindow(
        config,
        auto_watcher_factory=lambda **kw: _FakeWatcher(**kw))
    return win


def test_window_no_watcher_when_setting_off(qapp, tmp_path, monkeypatch):
    win = _make_window(qapp, tmp_path, monkeypatch, auto_detect=False)
    try:
        assert win._auto_watcher is None
        assert win.auto_badge.isHidden() is True
    finally:
        win.close()


def test_window_watcher_starts_when_setting_on(qapp, tmp_path, monkeypatch):
    win = _make_window(qapp, tmp_path, monkeypatch, auto_detect=True)
    try:
        assert win._auto_watcher is not None
        assert win._auto_watcher.started is True
        assert win._auto_watcher.is_running is True
    finally:
        win.close()
        # closeEvent stops the watcher.
        assert win._auto_watcher is None


def test_window_auto_badge_and_auto_stop(qapp, tmp_path, monkeypatch):
    import time
    from app.audio.capture import CaptureState
    win = _make_window(qapp, tmp_path, monkeypatch, auto_detect=True)
    try:
        watcher = win._auto_watcher
        assert watcher is not None
        # Meeting starts → auto-recording, badge visible, status says so.
        win._auto_bridge.meeting_started.emit()
        qapp.processEvents()
        assert win._auto_recording is True
        assert win.auto_badge.isHidden() is False
        assert "Auto-recording" in win.status_label.text()
        assert win.capture.state == CaptureState.RECORDING
        # Meeting ends → auto-stop, badge hidden, recording saved.
        win._auto_bridge.meeting_ended.emit()
        qapp.processEvents()
        time.sleep(0.1)
        qapp.processEvents()
        assert win._auto_recording is False
        assert win.auto_badge.isHidden() is True
        assert win.capture.state == CaptureState.STOPPED
        # Auto-stop does NOT suppress the watcher (it already fired).
        assert watcher.manual_stop_calls == 0
    finally:
        win.close()


def test_low_disk_warning_is_non_modal(qapp, tmp_path, monkeypatch):
    # Regression (Nov 21 2026): the low-disk warning was a modal
    # QMessageBox, which hangs headless/offscreen runs forever —
    # pytest's tmp dir lives on a 512 MB tmpfs, always below the
    # 1 GiB threshold, so the dialog could never be dismissed and
    # the suite wedged at the first auto-record test. The box must
    # be non-modal: the warning shows, the recording runs on.
    from app.audio.capture import CaptureState
    win = _make_window(qapp, tmp_path, monkeypatch, auto_detect=True)
    try:
        win._auto_bridge.meeting_started.emit()
        qapp.processEvents()
        assert win.capture.state == CaptureState.RECORDING
        box = win._disk_warning_box
        assert box is not None
        assert box.isModal() is False
    finally:
        win.close()


def test_window_manual_stop_suppresses_retrigger(
        qapp, tmp_path, monkeypatch):
    from app.audio.capture import CaptureState
    win = _make_window(qapp, tmp_path, monkeypatch, auto_detect=True)
    try:
        watcher = win._auto_watcher
        win._auto_bridge.meeting_started.emit()
        qapp.processEvents()
        assert win._auto_recording is True
        # The user stops it by hand. Stop runs off the UI thread
        # now (Nov 20 2026 Stop-freeze fix), so the capture flips
        # to STOPPED once the worker reports back — pump, don't
        # assert synchronously.
        win.on_stop()
        assert win._auto_recording is False
        assert win.auto_badge.isHidden() is True
        assert watcher.manual_stop_calls == 1
        deadline = time.monotonic() + 10.0
        while (win.capture.state != CaptureState.STOPPED
               and time.monotonic() < deadline):
            qapp.processEvents()
            time.sleep(0.02)
        assert win.capture.state == CaptureState.STOPPED
        # Wait for the stop to settle fully (finished slot ran,
        # stop thread exited) before closing the window — the
        # capture flips to STOPPED in the worker thread before
        # the finished slot runs, and closing on state alone
        # destroys the still-running QThread and aborts the
        # process (S3/N7: same wait the live-thread tests use).
        deadline = time.monotonic() + 10.0
        while win._stopping and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.02)
        assert not win._stopping
    finally:
        win.close()


def test_window_manual_recording_not_stopped_by_meeting_end(
        qapp, tmp_path, monkeypatch):
    from app.audio.capture import CaptureState
    win = _make_window(qapp, tmp_path, monkeypatch, auto_detect=True)
    try:
        win.on_record()  # manual record
        assert win._auto_recording is False
        assert win.auto_badge.isHidden() is True
        win._auto_bridge.meeting_ended.emit()
        qapp.processEvents()
        # The watcher never stops a manual recording.
        assert win.capture.state == CaptureState.RECORDING
    finally:
        win.close()


def test_window_meeting_start_ignored_while_recording(
        qapp, tmp_path, monkeypatch):
    from app.audio.capture import CaptureState
    win = _make_window(qapp, tmp_path, monkeypatch, auto_detect=True)
    try:
        win.on_record()  # manual record first
        assert win._auto_recording is False
        win._auto_bridge.meeting_started.emit()
        qapp.processEvents()
        # Still the manual recording — no double start, no badge.
        assert win._auto_recording is False
        assert win.auto_badge.isHidden() is True
        assert win.capture.state == CaptureState.RECORDING
    finally:
        win.close()
