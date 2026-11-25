"""Tests for app.audio.perapp_capture — true per-app capture (Windows).

The COM path cannot run on this Linux VM (no WASAPI), so the tests
exercise everything that *can* run here and mark the boundary honestly:

* pure ctypes structs + PROPVARIANT packing (importable anywhere),
* the verified GUID constants,
* target-PID resolution (fake process lists),
* availability reporting (False + reason on Linux),
* the capture class with injected fake streams (the real COM stream
  is swapped out via ``stream_factory``),
* mix-format conversion (pure numpy),
* DualCapture's auto-fallback swap (the fallback track itself needs
  WASAPI, so on Linux the test asserts the swap happened and the
  fallback failure surfaced — it does NOT claim a working fallback
  track here),
* config round-trip for ``capture_system_mode``,
* the Settings dialog picker (Qt-gated, offscreen).

A real Windows run is the open gate — see docs/PER_APP_CAPTURE.md.
"""
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from app.audio.capture import DualCapture, PerAppCapture
from app.audio.mixer import _read_wav
from app.audio.perapp_capture import (
    ACTIVATION_TYPE_PROCESS_LOOPBACK,
    AUDCLNT_BUFFERFLAGS_SILENT,
    AUDCLNT_E_UNSUPPORTED_FORMAT,
    AUDCLNT_STREAMFLAGS_LOOPBACK,
    E_ILLEGAL_METHOD_CALL,
    IID_IAGILE_OBJECT,
    IID_IACTIVATE_AUDIO_INTERFACE_ASYNC_OPERATION,
    IID_IACTIVATE_AUDIO_INTERFACE_COMPLETION_HANDLER,
    IID_IAUDIO_CAPTURE_CLIENT,
    LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
    MIN_WINDOWS_BUILD_PER_APP,
    MODE_LOOPBACK_FALLBACK,
    MODE_PER_APP,
    VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
    VT_BLOB,
    PerAppCaptureError,
    ProcessLoopbackCapture,
    TEAMS_DESKTOP_EXES,
    _ActivationParams,
    _PropVariantBlob,
    _convert_mix_chunk,
    _resample_linear,
    build_activation_params,
    describe_target_session,
    per_app_capture_available,
    resolve_capture_pid,
    resolve_target_pid,
    select_system_capture_class,
)
from app.config import (
    CAPTURE_SYSTEM_MODE_DEFAULT,
    CAPTURE_SYSTEM_MODES,
)


def _chunks(n=20, frames=1024, value=1000):
    chunk = np.full(frames, value, dtype=np.int16)
    for _ in range(n):
        yield chunk.copy()


# ------------------------------------------------------------ constants -----
def test_guids_match_the_verified_sdk_values():
    # Cross-checked against the winapi crate (generated from the Windows
    # SDK headers) and the ReactOS/Wine IDL mirrors.
    assert IID_IACTIVATE_AUDIO_INTERFACE_COMPLETION_HANDLER == (
        "{41D949AB-9862-444A-80F6-C261334DA5EB}")
    assert IID_IACTIVATE_AUDIO_INTERFACE_ASYNC_OPERATION == (
        "{72A22D78-CDE4-431D-B8CC-843A71199B6D}")
    assert IID_IAGILE_OBJECT == "{94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90}"
    assert IID_IAUDIO_CAPTURE_CLIENT == (
        "{C8ADBD64-E71E-48A0-A4DE-185C395CD317}")
    assert VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK == "VAD\\Process_Loopback"
    assert MIN_WINDOWS_BUILD_PER_APP == 20348
    assert AUDCLNT_E_UNSUPPORTED_FORMAT == 0x88890008
    assert E_ILLEGAL_METHOD_CALL == 0x8000000E


def test_audclnt_streamflags_loopback_matches_microsoft_header():
    # audioclient.h: CROSSPROCESS 0x00010000, LOOPBACK 0x00020000,
    # EVENTCALLBACK 0x00040000. 0x00000002 is AUDCLNT_BUFFERFLAGS_SILENT
    # (a GetBuffer flag), NOT a stream flag — passing it to
    # IAudioClient::Initialize fails on real Windows with
    # AUDCLNT_E_INVALID_STREAM_FLAG (0x88890021). This was the live
    # value until Nov 8 2026, when the first real Windows probe in the
    # VM failed on it; the two constants must never be confused again.
    assert AUDCLNT_STREAMFLAGS_LOOPBACK == 0x00020000
    assert AUDCLNT_BUFFERFLAGS_SILENT == 0x00000002
    assert AUDCLNT_STREAMFLAGS_LOOPBACK != AUDCLNT_BUFFERFLAGS_SILENT


def test_activation_params_struct_layout():
    # AUDIOCLIENT_ACTIVATION_PARAMS for the loopback member: 4-byte
    # ActivationType + 8-byte ProcessLoopbackParams = 12 bytes.
    assert ctypes.sizeof(_ActivationParams) == 12
    assert _ActivationParams.ActivationType.offset == 0
    assert _ActivationParams.u.offset == 4
    # PROPVARIANT with a VT_BLOB: 8-byte header + 4-byte cbSize +
    # padding + 8-byte pointer = 24 bytes on 64-bit.
    assert ctypes.sizeof(_PropVariantBlob) == 24
    assert _PropVariantBlob.vt.offset == 0
    assert _PropVariantBlob.cbSize.offset == 8
    assert _PropVariantBlob.pBlobData.offset == 16


def test_build_activation_params_packs_pid_and_mode():
    params, pv = build_activation_params(
        1234, LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE)
    assert params.ActivationType == ACTIVATION_TYPE_PROCESS_LOOPBACK
    assert params.u.ProcessLoopbackParams.TargetProcessId == 1234
    assert params.u.ProcessLoopbackParams.ProcessLoopbackMode == 0
    assert pv.vt == VT_BLOB
    assert pv.cbSize == ctypes.sizeof(params) == 12
    # The blob pointer really points at the params struct: read the
    # PID back through it.
    pid_back = ctypes.cast(pv.pBlobData,
                           ctypes.POINTER(ctypes.c_uint32))
    assert pid_back[1] == 1234  # TargetProcessId is the 2nd DWORD


def test_modes_are_the_documented_values():
    assert CAPTURE_SYSTEM_MODES == ("auto", "per-app", "loopback")
    assert CAPTURE_SYSTEM_MODE_DEFAULT == "auto"
    assert MODE_PER_APP == "per-app"
    assert MODE_LOOPBACK_FALLBACK == "system-loopback-fallback"


# ------------------------------------------------------------ availability --
def test_per_app_capture_unavailable_off_windows():
    if sys.platform == "win32":
        pytest.skip("Windows-only assertion")
    available, reason = per_app_capture_available()
    assert available is False
    assert reason and "Windows" in reason


def test_describe_target_session_returns_none_off_windows():
    if sys.platform == "win32":
        pytest.skip("Windows-only probe")
    assert describe_target_session(1234) is None


# ------------------------------------------------------------ pid lookup ----
class _FakeProc:
    def __init__(self, pid, name, ppid):
        self.info = {"pid": pid, "name": name, "ppid": ppid}


def _fake_iter(procs):
    def _iter(attrs):
        return list(procs)
    return _iter


def test_resolve_target_pid_returns_none_when_absent():
    assert resolve_target_pid(
        ["ms-teams.exe"], _fake_iter([_FakeProc(1, "notepad.exe", 0)])) is None


def test_resolve_target_pid_prefers_the_tree_root():
    # Teams main process (pid 100, parent is explorer) + a helper
    # child (pid 101, parent 100) under the same exe name: the root
    # (100) is the one INCLUDE_TARGET_PROCESS_TREE wants.
    procs = [_FakeProc(101, "ms-teams.exe", 100),
             _FakeProc(100, "ms-teams.exe", 4)]
    assert resolve_target_pid(["ms-teams.exe"], _fake_iter(procs)) == {
        "pid": 100, "name": "ms-teams.exe"}


def test_resolve_target_pid_falls_back_to_first_when_no_root():
    procs = [_FakeProc(7, "ms-teams.exe", 8),
             _FakeProc(8, "ms-teams.exe", 7)]
    assert resolve_target_pid(["ms-teams.exe"], _fake_iter(procs)) == {
        "pid": 7, "name": "ms-teams.exe"}


def test_resolve_target_pid_is_case_insensitive():
    procs = [_FakeProc(9, "MS-TEAMS.EXE", 0)]
    assert resolve_target_pid(["ms-teams.exe"], _fake_iter(procs)) == {
        "pid": 9, "name": "MS-TEAMS.EXE"}


# ------------------------------------------------------------ class select --
def test_select_system_capture_class():
    assert select_system_capture_class("loopback") is PerAppCapture
    assert select_system_capture_class("per-app") is ProcessLoopbackCapture
    # Unknown modes behave as "auto".
    assert select_system_capture_class("bogus") is (
        ProcessLoopbackCapture
        if per_app_capture_available()[0] else PerAppCapture)
    if sys.platform != "win32":
        assert select_system_capture_class("auto") is PerAppCapture


def test_select_system_capture_class_teams_desktop_auto_is_loopback(
        monkeypatch):
    """Auto mode never trusts per-app capture for the Teams desktop
    client: process loopback returns silence for it even mid-call
    (Microsoft windows-classic-samples#414), while device loopback
    captures it. Explicit per-app mode still tries per-app."""
    import app.audio.perapp_capture as perapp

    assert "ms-teams.exe" in TEAMS_DESKTOP_EXES
    monkeypatch.setattr(perapp, "per_app_capture_available",
                        lambda: (True, None))
    assert select_system_capture_class(
        "auto", "ms-teams.exe") is PerAppCapture
    assert select_system_capture_class(
        "auto", "MS-TEAMS.EXE") is PerAppCapture
    assert select_system_capture_class(
        "auto", "chrome.exe") is ProcessLoopbackCapture
    assert select_system_capture_class(
        "per-app", "ms-teams.exe") is ProcessLoopbackCapture
    assert select_system_capture_class(
        "loopback", "ms-teams.exe") is PerAppCapture


# ------------------------------------------------- session-pid resolution --
def _teams_tree_procs():
    return [
        _FakeProc(1, "explorer.exe", 0),
        _FakeProc(100, "ms-teams.exe", 1),
        _FakeProc(200, "ms-teams.exe", 1),
        _FakeProc(300, "msedgewebview2.exe", 100),
    ]


def test_resolve_capture_pid_prefers_the_rendering_session():
    from app.audio.app_sessions import AudioSessionInfo

    # Two ms-teams.exe trees (roots 100 and 200); the call renders
    # in tree 200. The root heuristic would pick 100; the session
    # pid (200) is the one actually rendering.
    sessions = [AudioSessionInfo("ms-teams.exe", "Microsoft Teams",
                                 200, "active", 0.5)]
    assert resolve_capture_pid(
        ["ms-teams.exe"], sessions=sessions,
        process_iter=_fake_iter(_teams_tree_procs())) == {
            "pid": 200, "name": "ms-teams.exe"}


def test_resolve_capture_pid_prefers_child_tree_session():
    from app.audio.app_sessions import AudioSessionInfo

    sessions = [AudioSessionInfo("msedgewebview2.exe", "Teams", 300,
                                 "active", 0.5)]
    assert resolve_capture_pid(
        ["ms-teams.exe"], sessions=sessions,
        process_iter=_fake_iter(_teams_tree_procs())) == {
            "pid": 300, "name": "msedgewebview2.exe"}


def test_resolve_capture_pid_falls_back_to_tree_root():
    # No session known: the tree-root heuristic, unchanged.
    assert resolve_capture_pid(
        ["ms-teams.exe"], sessions=[],
        process_iter=_fake_iter(_teams_tree_procs())) == {
            "pid": 100, "name": "ms-teams.exe"}
    # A listing failure (sessions=None on this Linux VM lists [])
    # degrades to the same fallback instead of raising.
    assert resolve_capture_pid(
        ["definitely-not-a-real-app.exe"], sessions=None) is None


# ------------------------------------------------------------ fake streams --
class _FakeStream:
    """A canned loopback stream: yields the given int16 packets."""

    def __init__(self, packets, **kwargs):
        self._packets = list(packets)
        self._index = 0
        self.started = False
        self.stopped = False
        self.closed = False
        self.kwargs = kwargs

    def start(self):
        self.started = True

    def read_chunk(self, timeout=0.1):
        if self._index < len(self._packets):
            packet = self._packets[self._index]
            self._index += 1
            return packet
        time.sleep(0.005)
        return None

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _RaisingStream(_FakeStream):
    def __init__(self, **kwargs):
        super().__init__([], **kwargs)

    def start(self):
        raise PerAppCaptureError("boom (test activation failure)")


def _packet(frames=1024, value=1000):
    return np.full(frames, value, dtype=np.int16).tobytes()


def test_process_loopback_capture_with_fake_stream(tmp_path):
    stream = _FakeStream([_packet(), _packet(), _packet()])
    cap = ProcessLoopbackCapture(
        tmp_path / "system.wav", process_name="ms-teams.exe",
        stream_factory=lambda **kw: stream)
    assert cap.mode == "per-app"
    assert cap.warning is None
    cap.start()
    assert stream.started
    time.sleep(0.3)
    cap.stop()
    assert stream.stopped and stream.closed
    samples, rate = _read_wav(tmp_path / "system.wav")
    assert rate == 48000
    assert len(samples) == 3 * 1024
    assert set(samples.tolist()) == {1000}


def test_process_loopback_capture_with_chunk_source(tmp_path):
    # The chunk_source seam (like the base class) skips hardware.
    cap = ProcessLoopbackCapture(
        tmp_path / "system.wav", process_name="ms-teams.exe",
        chunk_source=lambda: _chunks(5))
    assert cap.mode == "per-app"
    cap.start()
    time.sleep(0.3)  # let the thread consume the finite source
    cap.stop()
    samples, _ = _read_wav(tmp_path / "system.wav")
    assert len(samples) == 5 * 1024


def test_process_loopback_capture_start_fails_without_stream_on_linux(tmp_path):
    if sys.platform == "win32":
        pytest.skip("Linux-only: no WASAPI here")
    cap = ProcessLoopbackCapture(tmp_path / "system.wav")
    with pytest.raises(PerAppCaptureError, match="unavailable|not running"):
        cap.start()
    # A failed start leaves no stream behind and stop() still yields
    # a valid (empty) WAV.
    assert cap._stream is None
    out = cap.stop()
    assert out.exists()
    samples, _ = _read_wav(out)
    assert len(samples) == 0


def test_process_loopback_capture_start_propagates_stream_failure(tmp_path):
    cap = ProcessLoopbackCapture(
        tmp_path / "system.wav",
        stream_factory=lambda **kw: _RaisingStream())
    with pytest.raises(PerAppCaptureError, match="boom"):
        cap.start()
    assert cap._stream is None


# ------------------------------------------------------------ conversion ----
def test_resample_linear_identity_and_ratio():
    samples = np.arange(100, dtype=np.int16)
    assert _resample_linear(samples, 48000, 48000) is samples
    out = _resample_linear(samples, 44100, 48000)
    assert len(out) == round(100 * 48000 / 44100)
    assert out.dtype == np.int16


def test_convert_mix_chunk_int16_passthrough():
    data = np.full(1024, 2000, dtype=np.int16).tobytes()
    out = _convert_mix_chunk(data, rate=48000, channels=1,
                             format_tag=1, bits=16, target_rate=48000)
    assert np.frombuffer(out, dtype=np.int16).tolist() == [2000] * 1024


def test_convert_mix_chunk_float32_stereo_44100():
    # Stereo float32 sine at 44.1 kHz → mono int16 at 48 kHz.
    t = np.linspace(0, 0.1, 4410, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.column_stack([sine, sine]).astype(np.float32)
    out = _convert_mix_chunk(stereo.tobytes(), rate=44100, channels=2,
                             format_tag=3, bits=32, target_rate=48000)
    samples = np.frombuffer(out, dtype=np.int16)
    assert len(samples) == round(4410 * 48000 / 44100)
    assert samples.max() > 10000  # ~0.5 * 32767
    assert samples.min() < -10000


def test_convert_mix_chunk_int32():
    data = (np.full(100, 1 << 30, dtype=np.int32)).tobytes()
    out = _convert_mix_chunk(data, rate=48000, channels=1,
                             format_tag=1, bits=32, target_rate=48000)
    samples = np.frombuffer(out, dtype=np.int16)
    assert samples.tolist() == [16384] * 100  # 2**30 >> 16


def test_convert_mix_chunk_rejects_unknown_format():
    with pytest.raises(PerAppCaptureError, match="Unsupported mix format"):
        _convert_mix_chunk(b"\x00" * 16, rate=48000, channels=1,
                           format_tag=99, bits=8, target_rate=48000)


# ------------------------------------------------------------ DualCapture ---
def test_dual_capture_auto_falls_back_when_per_app_fails(tmp_path):
    """Per-app start failure in "auto" swaps in the loopback track.

    On Linux the fallback track itself cannot start (no WASAPI), so
    the test asserts the swap happened — mode updated, system object
    replaced — and the fallback's own failure surfaced instead of the
    original PerAppCaptureError. On real Windows the fallback *can*
    start (given a loopback device), so there the test asserts the
    swap and a working fallback track instead. Neither branch claims
    a working fallback track on Linux; that needs Windows.
    """
    class _FailingPerApp(ProcessLoopbackCapture):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._stream_factory = lambda **kw: _RaisingStream()

    # On real Windows the fallback needs a loopback device index (the
    # WASAPI gate refuses device_index=None for the system kind, so it
    # can never silently record the microphone as "system" audio).
    loop_index = None
    if sys.platform == "win32":
        from app.audio.devices import default_loopback

        loop = default_loopback()
        if loop is None:
            pytest.skip("No WASAPI loopback device on this Windows machine")
        loop_index = loop.index

    session = tmp_path / "session"
    dual = DualCapture(
        session, process_name="ms-teams.exe",
        capture_system_mode="auto", system_cls=_FailingPerApp,
        loopback_device_index=loop_index,
        chunk_source_factory=lambda kind: (
            _chunks(10) if kind == "mic" else None))
    if sys.platform == "win32":
        dual.start()
        try:
            assert isinstance(dual.system, PerAppCapture)
            assert dual.mode == "system-loopback-fallback"
            assert dual.warning  # the fallback's honest loopback warning
            time.sleep(0.5)  # let the fallback track start
        finally:
            result = dual.stop()
        # The fallback track itself is verified end-to-end (real audio)
        # in the Windows VM manual test, not here: when nothing is
        # playing, WASAPI loopback yields no packets and the file stays
        # a header, so a frame-count assertion would be environment
        # noise. This test pins the swap + a startable fallback track.
        assert result["system"].exists()
        return
    with pytest.raises(RuntimeError, match="Windows-only|pyaudiowpatch"):
        dual.start()
    assert isinstance(dual.system, PerAppCapture)
    assert dual.mode == "system-loopback-fallback"
    assert dual.warning  # the fallback's honest loopback warning


def test_dual_capture_per_app_mode_does_not_fall_back(tmp_path):
    class _FailingPerApp(ProcessLoopbackCapture):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._stream_factory = lambda **kw: _RaisingStream()

    session = tmp_path / "session"
    dual = DualCapture(
        session, process_name="ms-teams.exe",
        capture_system_mode="per-app", system_cls=_FailingPerApp,
        chunk_source_factory=lambda kind: (
            _chunks(10) if kind == "mic" else None))
    with pytest.raises(PerAppCaptureError, match="boom"):
        dual.start()
    assert isinstance(dual.system, ProcessLoopbackCapture)
    assert dual.mode == "per-app"


def test_dual_capture_loopback_mode_selects_old_class(tmp_path):
    session = tmp_path / "session"
    dual = DualCapture(
        session, process_name="ms-teams.exe",
        capture_system_mode="loopback",
        chunk_source_factory=lambda kind: _chunks(10))
    assert isinstance(dual.system, PerAppCapture)
    assert dual.mode == "system-loopback-fallback"
    dual.start()
    time.sleep(0.2)
    files = dual.stop()
    assert files["system"].exists()


# ------------------------------------------------------------ config --------
def test_capture_system_mode_round_trips_through_settings(tmp_path):
    from app.config import (
        AppConfig,
        config_from_settings,
        load_settings,
        save_settings,
        settings_to_dict,
    )

    cfg = AppConfig(output_root=tmp_path, capture_system_mode="per-app")
    assert settings_to_dict(cfg)["capture_system_mode"] == "per-app"
    loaded = config_from_settings(
        {"capture_system_mode": "per-app", "output_root": str(tmp_path)})
    assert loaded.capture_system_mode == "per-app"
    path = save_settings(cfg, tmp_path / "settings.json")
    assert load_settings(path).capture_system_mode == "per-app"
    # A bad stored value falls back to the default, never breaks startup.
    bad = config_from_settings(
        {"capture_system_mode": "bogus", "output_root": str(tmp_path)})
    assert bad.capture_system_mode == "auto"
    with pytest.raises(ValueError, match="unknown capture system mode"):
        AppConfig(capture_system_mode="bogus")


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


def test_settings_dialog_offers_capture_mode_and_round_trips(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.settings_dialog import SettingsDialog

    cfg = AppConfig(output_root=tmp_path, capture_system_mode="loopback")
    dlg = SettingsDialog(cfg)
    try:
        assert [dlg.capture_mode_combo.itemData(i)
                for i in range(dlg.capture_mode_combo.count())] == [
                    "auto", "per-app", "loopback"]
        assert dlg.capture_mode_combo.currentData() == "loopback"
        dlg.capture_mode_combo.setCurrentIndex(
            dlg.capture_mode_combo.findData("per-app"))
        out = dlg.config_from_dialog()
        assert out.capture_system_mode == "per-app"
    finally:
        dlg.close()
