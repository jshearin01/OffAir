"""Tests for app.platform_info — backend selection + honest status strings.

No real audio hardware, no sounddevice install needed: the fake
sounddevice module is injected into sys.modules, and sys.platform is
monkeypatched where a platform other than this Linux VM is needed.
"""
import sys
import types

import pytest

import app.audio.devices as devices
import app.platform_info as platform_info


# ------------------------------------------------------------ fakes --------
def _fake_sounddevice(*, mic=True, monitor=True):
    """A minimal sounddevice stand-in with PortAudio-style devices."""
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
    infos.append({  # output-only: must never appear as mic or loopback
        "index": 2, "name": "Built-in Audio Analog Stereo", "hostapi": 0,
        "max_input_channels": 0, "max_output_channels": 2,
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
    sd.default = types.SimpleNamespace(device=(0, 2))
    return sd


@pytest.fixture
def fake_sd(monkeypatch):
    """Install the fake sounddevice (mic + monitor) for one test."""
    sd = _fake_sounddevice()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


# ``no_sd`` lives in tests/conftest.py (a real import blocker — the
# old per-file version only deleted sys.modules entries, which cannot
# block re-import once sounddevice is installed on disk).

# ------------------------------------------------------------ backend ------
def test_backend_is_linux_sounddevice_with_sounddevice(fake_sd, monkeypatch):
    # The backend is a platform branch: force the Linux branch so the
    # test means the same thing on a Windows host.
    monkeypatch.setattr(sys, "platform", "linux")
    assert platform_info.get_capture_backend() == "linux-sounddevice"


def test_backend_is_unavailable_without_sounddevice(no_sd, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert platform_info.get_capture_backend() == "unavailable"


def test_backend_is_wasapi_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert platform_info.get_capture_backend() == "wasapi"


def test_backend_is_unavailable_on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert platform_info.get_capture_backend() == "unavailable"


def test_platform_name_helpers(monkeypatch):
    # Set the platform explicitly per case — the helpers read
    # sys.platform, so asserting the host's platform here would make
    # the test fail on any non-Linux machine.
    monkeypatch.setattr(sys, "platform", "linux")
    assert platform_info.is_linux()
    assert not platform_info.is_windows()
    assert not platform_info.is_macos()
    assert platform_info.platform_name() == "linux"
    monkeypatch.setattr(sys, "platform", "win32")
    assert platform_info.platform_name() == "windows"
    assert platform_info.is_windows()


# ------------------------------------------------------------ imports ------
def test_windows_only_imports_do_not_break_on_linux():
    """Importing every audio/platform module must be safe on Linux,
    with and without sounddevice — nothing may import pyaudiowpatch
    or sounddevice at module load."""
    import app.audio.capture
    import app.audio.devices
    import app.audio.linux_capture
    import app.audio.mixer
    import app.config
    import app.platform_info
    import main
    assert app.audio.capture.DualCapture is not None
    assert app.audio.linux_capture.LinuxDualCapture is not None
    assert main.build_parser() is not None


# ------------------------------------------------------------ state --------
def test_describe_state_linux_no_sounddevice(no_sd, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    info = platform_info.describe_capture_state()
    assert info.platform == "linux"
    assert info.backend == "unavailable"
    assert info.ready is False
    assert info.mic_only is False
    assert info.microphones == [] and info.loopback_devices == []
    assert "sounddevice" in info.status.lower()
    assert info.install_hint is not None
    assert "libportaudio2" in info.install_hint
    assert "pip install sounddevice" in info.install_hint


def test_describe_state_linux_ready_with_fake_devices(fake_sd, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    info = platform_info.describe_capture_state()
    assert info.backend == "linux-sounddevice"
    assert info.ready is True
    assert info.mic_only is False
    assert [m.name for m in info.microphones] == ["Built-in Microphone"]
    assert len(info.loopback_devices) == 1
    assert info.loopback_devices[0].is_loopback is True
    assert "Monitor" in info.loopback_devices[0].name
    assert "Ready" in info.status


def test_describe_state_linux_mic_only_without_monitor(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "sounddevice",
                        _fake_sounddevice(monitor=False))
    info = platform_info.describe_capture_state()
    assert info.backend == "linux-sounddevice"
    assert info.ready is False
    assert info.mic_only is True
    assert len(info.microphones) == 1
    assert info.loopback_devices == []
    assert "monitor" in info.status.lower()
    assert "monitor" in info.detail.lower()


def test_describe_state_linux_sounddevice_but_no_devices(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "sounddevice",
                        _fake_sounddevice(mic=False, monitor=False))
    info = platform_info.describe_capture_state()
    assert info.backend == "linux-sounddevice"
    assert info.ready is False
    assert "no audio devices" in info.status.lower()


def test_describe_state_windows_missing_pyaudiowpatch(
        monkeypatch, no_sd, no_pyaudiowpatch):
    # Block the import for real (conftest.py): deleting the name from
    # sys.modules cannot block re-import on a machine where
    # pyaudiowpatch is installed, so the old version of this test
    # enumerated the real Windows devices and reported ready=True.
    monkeypatch.setattr(sys, "platform", "win32")
    info = platform_info.describe_capture_state()
    assert info.backend == "wasapi"
    assert info.ready is False
    assert "pyaudiowpatch" in info.status.lower()
    assert info.install_hint is not None
    assert "requirements.txt" in info.install_hint


def test_describe_state_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    info = platform_info.describe_capture_state()
    assert info.backend == "unavailable"
    assert "macos" in info.status.lower()


def test_sounddevice_status_names_portaudio_when_library_missing(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    status = devices.sounddevice_status()
    assert "libportaudio2" in status
    assert "PortAudio" in status


# ------------------------------------------------------------ messages -----
def test_mic_empty_message_differs_by_platform(no_sd, monkeypatch):
    # Force the Linux branch first: on a Windows host the message
    # would otherwise name pyaudiowpatch from the start.
    monkeypatch.setattr(sys, "platform", "linux")
    # Linux without sounddevice: names sounddevice, never pyaudiowpatch.
    msg = platform_info.mic_empty_message()
    assert "sounddevice" in msg.lower()
    assert "pyaudiowpatch" not in msg.lower()
    # Linux with sounddevice but no devices: names PortAudio.
    monkeypatch.setitem(sys.modules, "sounddevice",
                        _fake_sounddevice(mic=False, monitor=False))
    msg = platform_info.mic_empty_message()
    assert "portaudio" in msg.lower()
    # Windows: the old message is still the right one there.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    assert "pyaudiowpatch" in platform_info.mic_empty_message().lower()


def test_loopback_empty_message_names_monitor_mode(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "sounddevice",
                        _fake_sounddevice(monitor=False))
    msg = platform_info.loopback_empty_message()
    assert "monitor" in msg.lower()
    assert "microphone only" in msg.lower()


def test_loopback_empty_message_sounddevice_missing(no_sd, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    msg = platform_info.loopback_empty_message()
    assert "sounddevice" in msg.lower()


# ------------------------------------------------------------ processes ----
def test_default_process_names_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    names = platform_info.default_process_names()
    assert "teams" in names and "ms-teams" in names
    assert "ms-teams.exe" in names  # Windows names kept, harmless on Linux


def test_default_process_names_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    names = platform_info.default_process_names()
    assert names == ["ms-teams.exe", "Teams.exe", "ms-teamsupdate.exe"]
    assert "teams" not in names
