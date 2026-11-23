"""Tests for the Linux sounddevice capture backend.

No real audio hardware and no sounddevice install needed: the fake
sounddevice module is injected into sys.modules, and the capture
state machine is exercised with fake chunk sources exactly like the
WASAPI tests in test_capture.py.
"""
import sys
import time
import types

import numpy as np
import pytest

import app.audio.devices as devices
from app.audio.capture import (
    CaptureState,
    MicCapture,
    create_dual_capture,
)

#: Tests that drive the Linux backend (fake sounddevice, monitor
#: sources, Linux backend selection) cannot run on Windows: the code
#: under test branches on platform and the fakes only stand in for
#: the Linux audio stack. Linux-only tests get this marker; the
#: platform-agnostic ones in this file (pure helpers, the chunk-source
#: state machine, the guided sounddevice error) keep running
#: everywhere.
needs_linux = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Linux sounddevice backend")
from app.audio.linux_capture import (
    LINUX_MODE_MIC_ONLY,
    LINUX_MODE_SYSTEM_MONITOR,
    LinuxDualCapture,
    LinuxMicCapture,
    LinuxSystemCapture,
)
from app.audio.mixer import _read_wav


# ------------------------------------------------------------ fakes --------
class FakeInputStream:
    """Stand-in for sounddevice.InputStream (blocking read loop).

    Returns silence forever; the capture thread stops when the stop
    event is set. ``opened`` records the kwargs of the last open so
    tests can assert the device / format selection.
    """

    opened: dict | None = None

    def __init__(self, *args, **kwargs):
        type(self).opened = dict(kwargs)
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, frames):
        return np.zeros(frames, dtype=np.int16), False


def _fake_sounddevice(*, mic=True, monitor=True):
    sd = types.ModuleType("sounddevice")
    infos = []
    if mic:
        infos.append({
            "index": 0, "name": "Built-in Microphone", "hostapi": 0,
            "max_input_channels": 2, "max_output_channels": 0,
            "default_samplerate": 48000.0,
        })
    if monitor:
        infos.append({
            "index": 1,
            "name": "Monitor of Built-in Audio Analog Stereo",
            "hostapi": 0,
            "max_input_channels": 2, "max_output_channels": 0,
            "default_samplerate": 48000.0,
        })

    def query_devices(kind=None, **_kwargs):
        if kind == "input":
            for info in infos:
                if info["max_input_channels"] > 0:
                    return dict(info)
            return {"index": 0}
        return [dict(info) for info in infos]

    sd.query_devices = query_devices
    sd.default = types.SimpleNamespace(device=(0, 1))
    sd.InputStream = FakeInputStream
    return sd


@pytest.fixture
def fake_sd(monkeypatch):
    sd = _fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


# ``no_sd`` lives in tests/conftest.py (a real import blocker — see
# the note there).


def _chunks(n=20, frames=1024, value=1000):
    chunk = np.full(frames, value, dtype=np.int16)
    for _ in range(n):
        yield chunk.copy()


def _factory(kind):
    return lambda: _chunks(20)


# ------------------------------------------------------------ enumeration --
@needs_linux
def test_linux_enumeration_splits_mic_and_monitor(fake_sd):
    mics = devices.list_microphones()
    loops = devices.list_loopback_devices()
    assert [m.name for m in mics] == ["Built-in Microphone"]
    assert all(not m.is_loopback for m in mics)
    assert len(loops) == 1
    assert loops[0].is_loopback is True
    assert "Monitor" in loops[0].name
    assert devices.default_microphone() is not None
    assert devices.default_loopback() is not None
    assert devices.default_loopback().index == 1


@needs_linux
def test_linux_enumeration_no_monitor_source(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice",
                        _fake_sounddevice(monitor=False))
    assert devices.list_microphones() != []
    assert devices.list_loopback_devices() == []
    assert devices.default_loopback() is None


@needs_linux
def test_linux_enumeration_without_sounddevice(no_sd):
    assert devices.list_microphones() == []
    assert devices.list_loopback_devices() == []
    assert "sounddevice" in devices.sounddevice_status().lower()


def test_monitor_name_detection():
    assert devices._is_monitor_source_name(
        "Monitor of Built-in Audio Analog Stereo")
    assert devices._is_monitor_source_name("monitor of hdmi")
    assert not devices._is_monitor_source_name("Built-in Microphone")


# ------------------------------------------------------------ mic capture --
def test_linux_mic_capture_state_machine_with_chunk_source(tmp_path):
    levels = []
    cap = LinuxMicCapture(tmp_path / "mic.wav",
                          chunk_source=lambda: _chunks(50),
                          level_callback=levels.append)
    assert cap.state == CaptureState.IDLE
    cap.start()
    assert cap.state == CaptureState.RECORDING
    time.sleep(0.1)
    cap.pause()
    assert cap.state == CaptureState.PAUSED
    cap.resume()
    out = cap.stop()
    assert cap.state == CaptureState.STOPPED
    samples, rate = _read_wav(out)
    assert rate == 48000 and len(samples) > 0
    assert levels and all(0.0 <= lv <= 1.0 for lv in levels)


@needs_linux
def test_linux_mic_capture_hardware_path_uses_fake_stream(tmp_path, fake_sd):
    FakeInputStream.opened = None
    levels = []
    cap = LinuxMicCapture(tmp_path / "mic.wav", device_index=0,
                          level_callback=levels.append)
    cap.start()
    time.sleep(0.2)
    cap.stop()
    assert cap.state == CaptureState.STOPPED
    assert cap.frames_written > 0
    assert levels  # silence -> 0.0 levels still flow through the callback
    assert FakeInputStream.opened is not None
    assert FakeInputStream.opened["device"] == 0
    assert FakeInputStream.opened["samplerate"] == 48000
    assert FakeInputStream.opened["dtype"] == "int16"
    assert FakeInputStream.opened["channels"] == 1


def test_linux_mic_capture_without_sounddevice_raises_guided(tmp_path, no_sd):
    cap = LinuxMicCapture(tmp_path / "mic.wav", device_index=0)
    with pytest.raises(RuntimeError, match="sounddevice"):
        cap.start()


@needs_linux
def test_linux_system_capture_requires_monitor_index(tmp_path, fake_sd):
    # device_index=None for the system kind would silently record the
    # default microphone again — it must refuse instead.
    cap = LinuxSystemCapture(tmp_path / "system.wav")
    with pytest.raises(RuntimeError, match="monitor"):
        cap.start()


def test_linux_system_capture_mode_and_warning():
    assert LinuxSystemCapture.mode == LINUX_MODE_SYSTEM_MONITOR
    assert "system-wide" in LinuxSystemCapture.warning


# ------------------------------------------------------------ dual capture -
@needs_linux
def test_linux_dual_capture_with_monitor(tmp_path, fake_sd):
    dual = LinuxDualCapture(tmp_path, chunk_source_factory=_factory)
    assert dual.mode == LINUX_MODE_SYSTEM_MONITOR
    assert dual.system is not None
    dual.start()
    time.sleep(0.1)
    result = dual.stop()
    for key in ("mic", "system", "combined"):
        assert result[key].exists(), key
    mic_samples, _ = _read_wav(result["mic"])
    sys_samples, _ = _read_wav(result["system"])
    assert len(mic_samples) > 0 and len(sys_samples) > 0
    assert dual.state == CaptureState.STOPPED


@needs_linux
def test_linux_dual_capture_mic_only_without_monitor(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice",
                        _fake_sounddevice(monitor=False))
    dual = LinuxDualCapture(tmp_path, chunk_source_factory=_factory)
    assert dual.mode == LINUX_MODE_MIC_ONLY
    assert dual.system is None
    assert "monitor" in (dual.warning or "").lower()
    dual.start()
    time.sleep(0.1)
    result = dual.stop()
    # Session folder keeps its shape: empty system.wav + combined mix.
    assert result["mic"].exists()
    assert result["system"].exists()
    assert result["combined"].exists()
    mic_samples, _ = _read_wav(result["mic"])
    sys_samples, _ = _read_wav(result["system"])
    assert len(mic_samples) > 0
    assert len(sys_samples) == 0
    assert dual.state == CaptureState.STOPPED


def test_linux_dual_capture_pause_resume_mic_only(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice",
                        _fake_sounddevice(monitor=False))
    dual = LinuxDualCapture(tmp_path, chunk_source_factory=_factory)
    dual.start()
    dual.pause()
    assert dual.state == CaptureState.PAUSED
    dual.resume()
    assert dual.state == CaptureState.RECORDING
    dual.stop()


# ------------------------------------------------------------ factory ------
@needs_linux
def test_create_dual_capture_picks_linux_backend(tmp_path, fake_sd):
    dual = create_dual_capture(tmp_path, chunk_source_factory=_factory)
    # LinuxDualCapture subclasses DualCapture, so check the exact class.
    assert type(dual) is LinuxDualCapture
    assert dual.mode == LINUX_MODE_SYSTEM_MONITOR


@needs_linux
def test_create_dual_capture_raises_without_sounddevice(tmp_path, no_sd):
    with pytest.raises(RuntimeError, match="sounddevice"):
        create_dual_capture(tmp_path)


@needs_linux
def test_wasapi_capture_points_linux_users_to_linux_backend(tmp_path, no_sd):
    """The Windows-only class must not fail obscurely on Linux."""
    cap = MicCapture(tmp_path / "mic.wav")
    with pytest.raises(RuntimeError, match="Linux"):
        cap.start()
