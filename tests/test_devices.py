"""Tests for app.audio.devices — graceful degradation, no hardware."""
import sys

import app.audio.devices as devices


def test_list_microphones_degrades_on_non_windows(monkeypatch):
    # On this Linux VM pyaudiowpatch is unavailable by design. The
    # Linux backend enumerates via sounddevice, so pin "no backend"
    # here — otherwise the test flips with ambient PulseAudio state
    # (Nov 10 2026: a leftover RDP session keeps the PulseAudio
    # daemon up, so real 'pulse'/'default' devices appear).
    monkeypatch.setattr(devices, "load_sounddevice", lambda: None)
    result = devices.list_microphones()
    assert result == [] if sys.platform != "win32" else isinstance(result, list)


def test_list_loopback_degrades_on_non_windows():
    result = devices.list_loopback_devices()
    assert result == [] if sys.platform != "win32" else isinstance(result, list)


def test_defaults_are_none_when_no_devices(monkeypatch):
    if sys.platform == "win32":
        return  # On Windows real devices may exist; nothing to assert here.
    # Pin "no backend" — see the note on the microphone test above.
    monkeypatch.setattr(devices, "load_sounddevice", lambda: None)
    assert devices.default_microphone() is None
    assert devices.default_loopback() is None


def test_find_teams_process_never_raises_and_returns_none_or_dict(monkeypatch):
    result = devices.find_teams_process(["definitely-not-a-real-process.exe"])
    assert result is None


def test_find_teams_process_without_psutil(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert devices.find_teams_process() is None


def test_pyaudiowpatch_status_is_a_string():
    assert isinstance(devices.pyaudiowpatch_status(), str)


def test_audio_device_display_name():
    info = devices.AudioDeviceInfo(index=0, name="Mic", is_default=True)
    assert info.display_name() == "Mic (default)"
    plain = devices.AudioDeviceInfo(index=1, name="Loop")
    assert plain.display_name() == "Loop"
