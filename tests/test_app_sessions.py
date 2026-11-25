"""Tests for the target-app picker backend (app.audio.app_sessions).

The Windows COM path is exercised with injected fake enumerators —
no real WASAPI stack exists on this Linux VM — following the same
pattern as tests/test_auto_detect.py (fake probe, deterministic).
"""
from __future__ import annotations

import sys
import types

import pytest

from app.audio.app_sessions import (
    AudioSessionInfo,
    audio_sessions_available,
    display_name_for_exe,
    find_session,
    find_session_for_app,
    list_audio_sessions,
    prettify_exe_name,
)


# ------------------------------------------------------------ naming -------
def test_prettify_exe_name_known_apps():
    assert prettify_exe_name("chrome.exe") == "Google Chrome"
    assert prettify_exe_name("ms-teams.exe") == "Microsoft Teams"
    assert prettify_exe_name("ms-teamsupdate.exe") == "Microsoft Teams"
    assert prettify_exe_name("spotify.exe") == "Spotify"
    assert prettify_exe_name("CHROME.EXE") == "Google Chrome"


def test_prettify_exe_name_generic_fallback():
    assert prettify_exe_name("my-app.exe") == "My App"
    assert prettify_exe_name("some_tool.exe") == "Some Tool"
    assert prettify_exe_name("notepad++.exe") == "Notepad++"


def test_display_name_prefers_session_name():
    # A session display name set by the app wins over the exe map.
    assert display_name_for_exe(
        "chrome.exe", "Chrome — YouTube") == "Chrome — YouTube"
    assert display_name_for_exe("chrome.exe", "  ") == "Google Chrome"
    assert display_name_for_exe("chrome.exe", None) == "Google Chrome"
    assert display_name_for_exe("unknown-app.exe", None) == "Unknown App"


def test_find_session_case_insensitive():
    sessions = [AudioSessionInfo("chrome.exe", "Google Chrome", 100,
                                 "active", 0.4)]
    assert find_session(sessions, "CHROME.EXE") is sessions[0]
    assert find_session(sessions, "ms-teams.exe") is None


# ------------------------------------------------- tree-aware search -------
class _FakeProc:
    """One process-table row, psutil protocol (``.info`` dict)."""

    def __init__(self, pid, name, ppid):
        self.info = {"pid": pid, "name": name, "ppid": ppid}


def _fake_process_iter(procs):
    def _iter(attrs):
        return list(procs)

    return _iter


def _teams_tree():
    """ms-teams.exe root (100) + WebView2 renderer child (300)."""
    return [
        _FakeProc(1, "explorer.exe", 0),
        _FakeProc(100, "ms-teams.exe", 1),
        _FakeProc(300, "msedgewebview2.exe", 100),
    ]


def test_find_session_for_app_direct_match_wins():
    child = AudioSessionInfo("msedgewebview2.exe", "Teams", 300,
                             "active", 0.5)
    direct = AudioSessionInfo("ms-teams.exe", "Microsoft Teams", 100,
                              "idle", 0.0)
    found = find_session_for_app(
        [child, direct], "ms-teams.exe",
        process_iter=_fake_process_iter(_teams_tree()))
    assert found is direct


def test_find_session_for_app_finds_child_process_session():
    # The only session belongs to the renderer child (different
    # exe): the plain exe match reports nothing, the tree search
    # finds it. This is the Nov 24 2026 live-call case.
    child = AudioSessionInfo("msedgewebview2.exe", "Teams", 300,
                             "active", 0.5)
    found = find_session_for_app(
        [child], "ms-teams.exe",
        process_iter=_fake_process_iter(_teams_tree()))
    assert found is child


def test_find_session_for_app_prefers_active_in_tree():
    idle = AudioSessionInfo("msedgewebview2.exe", "Teams", 300,
                            "idle", 0.0)
    active = AudioSessionInfo("msedgewebview2.exe", "Teams", 301,
                              "active", 0.4)
    procs = _teams_tree() + [_FakeProc(301, "msedgewebview2.exe", 100)]
    found = find_session_for_app(
        [idle, active], "ms-teams.exe",
        process_iter=_fake_process_iter(procs))
    assert found is active


def test_find_session_for_app_no_match():
    other = AudioSessionInfo("chrome.exe", "Google Chrome", 200,
                             "active", 0.4)
    assert find_session_for_app(
        [other], "ms-teams.exe",
        process_iter=_fake_process_iter(_teams_tree())) is None
    # No process with the exe name at all: no tree to search.
    assert find_session_for_app(
        [other], "zoom.exe",
        process_iter=_fake_process_iter(_teams_tree())) is None


def test_find_session_for_app_process_table_failure_falls_back():
    direct = AudioSessionInfo("ms-teams.exe", "Microsoft Teams", 100,
                              "active", 0.4)

    def _boom(attrs):
        raise RuntimeError("process table unreadable")

    # Direct match still works; a missing direct match degrades
    # to None instead of raising.
    assert find_session_for_app(
        [direct], "ms-teams.exe", process_iter=_boom) is direct
    assert find_session_for_app(
        [], "ms-teams.exe", process_iter=_boom) is None


@pytest.mark.skipif(sys.platform == "win32",
                    reason="non-Windows behaviour; on Windows sessions exist")
def test_audio_sessions_available_non_windows():
    available, reason = audio_sessions_available()
    assert available is False
    assert reason and "Windows" in reason


@pytest.mark.skipif(sys.platform == "win32",
                    reason="non-Windows behaviour; on Windows sessions exist")
def test_list_audio_sessions_non_windows_returns_empty():
    # Never raises, even where there is no WASAPI stack at all.
    assert list_audio_sessions() == []


# ------------------------------------------------------------ fake COM -----
class _FakeMeter:
    """IAudioMeterInformation: GetPeakValue() -> the scripted peak."""

    def __init__(self, peak: float) -> None:
        self._peak = peak

    def GetPeakValue(self):
        return self._peak


class _FakeSession:
    """One IAudioSessionControl(2): pid, state, peak, display name."""

    def __init__(self, pid: int, raw_state: int, peak: float,
                 display_name: str | None) -> None:
        self._pid = pid
        self._raw_state = raw_state
        self._peak = peak
        self._display_name = display_name

    def QueryInterface(self, cls):
        if cls is _FakeMeterCls:
            return _FakeMeter(self._peak)
        return self

    def GetProcessId(self):
        return self._pid

    def GetState(self):
        return self._raw_state

    def GetDisplayName(self):
        return self._display_name


class _FakeMeterCls:
    """Stand-in for IAudioMeterInformation (identity matters only)."""


class _FakeControl2Cls:
    """Stand-in for IAudioSessionControl2 (identity matters only)."""


class _FakeSessionEnum:
    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def GetCount(self):
        return len(self._sessions)

    def GetSession(self, i):
        return self._sessions[i]


class _FakeManager:
    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def GetSessionEnumerator(self):
        return _FakeSessionEnum(self._sessions)


class _FakeActivated:
    def __init__(self, manager) -> None:
        self._manager = manager

    def QueryInterface(self, cls):
        return self._manager


class _FakeDevice:
    def __init__(self, sessions) -> None:
        self._sessions = sessions

    def Activate(self, iid, clsctx, _none):
        return _FakeActivated(_FakeManager(self._sessions))


class _FakeCollection:
    def __init__(self, devices) -> None:
        self._devices = devices

    def GetCount(self):
        return len(self._devices)

    def Item(self, i):
        return self._devices[i]


class _FakeEnumerator:
    """IMMDeviceEnumerator over scripted render endpoints."""

    def __init__(self, endpoints) -> None:
        # endpoints: list of session lists, one per render device.
        self._endpoints = endpoints

    def EnumAudioEndpoints(self, data_flow, state):
        assert data_flow == 0  # render
        return _FakeCollection(
            [_FakeDevice(sessions) for sessions in self._endpoints])


class _FakeManagerCls:
    _iid_ = "fake-manager-iid"


def _install_fake_pycaw(monkeypatch):
    """Put fake pycaw interface classes in sys.modules.

    list_audio_sessions imports them lazily, so the fakes only need
    to exist under the pycaw.api.* module names.
    """
    pycaw = types.ModuleType("pycaw")
    api = types.ModuleType("pycaw.api")
    audiopolicy = types.ModuleType("pycaw.api.audiopolicy")
    audiopolicy.IAudioSessionControl2 = _FakeControl2Cls
    audiopolicy.IAudioSessionManager2 = _FakeManagerCls
    endpointvolume = types.ModuleType("pycaw.api.endpointvolume")
    endpointvolume.IAudioMeterInformation = _FakeMeterCls
    mmdeviceapi = types.ModuleType("pycaw.api.mmdeviceapi")
    mmdeviceapi.IMMDeviceEnumerator = object
    monkeypatch.setitem(sys.modules, "pycaw", pycaw)
    monkeypatch.setitem(sys.modules, "pycaw.api", api)
    monkeypatch.setitem(sys.modules, "pycaw.api.audiopolicy", audiopolicy)
    monkeypatch.setitem(sys.modules, "pycaw.api.endpointvolume",
                        endpointvolume)
    monkeypatch.setitem(sys.modules, "pycaw.api.mmdeviceapi", mmdeviceapi)


def test_list_audio_sessions_windows_path_with_fakes(monkeypatch):
    _install_fake_pycaw(monkeypatch)
    monkeypatch.setattr(sys, "platform", "win32")
    # Endpoint 1: Chrome active, Teams idle. Endpoint 2: Spotify
    # active, plus a session whose pid has no process (skipped).
    enumerator = _FakeEnumerator([
        [_FakeSession(100, 1, 0.4, "Google Chrome"),
         _FakeSession(200, 0, 0.0, None)],
        [_FakeSession(300, 1, 0.1, None),
         _FakeSession(999, 1, 0.5, None)],
    ])
    names = {100: "chrome.exe", 200: "ms-teams.exe",
             300: "spotify.exe", 999: None}
    sessions = list_audio_sessions(
        enumerator_factory=lambda: enumerator,
        process_name_for_pid=lambda pid: names.get(pid))
    # Active first, then by display name; the unknown pid is skipped.
    assert [(s.exe_name, s.state, s.peak) for s in sessions] == [
        ("chrome.exe", "active", 0.4),
        ("spotify.exe", "active", 0.1),
        ("ms-teams.exe", "idle", 0.0),
    ]
    assert sessions[0].display_name == "Google Chrome"
    assert sessions[1].display_name == "Spotify"  # prettified exe
    assert sessions[2].display_name == "Microsoft Teams"


def test_list_audio_sessions_dedupes_pid_across_endpoints(monkeypatch):
    _install_fake_pycaw(monkeypatch)
    monkeypatch.setattr(sys, "platform", "win32")
    # Same pid idle on endpoint 1, active on endpoint 2: one entry,
    # and it is the Active one (the one making sound).
    enumerator = _FakeEnumerator([
        [_FakeSession(100, 0, 0.0, None)],
        [_FakeSession(100, 1, 0.3, None)],
    ])
    sessions = list_audio_sessions(
        enumerator_factory=lambda: enumerator,
        process_name_for_pid=lambda pid: "chrome.exe")
    assert len(sessions) == 1
    assert sessions[0].state == "active"
    assert sessions[0].peak == 0.3


def test_list_audio_sessions_enumerator_failure_returns_empty(monkeypatch):
    _install_fake_pycaw(monkeypatch)
    monkeypatch.setattr(sys, "platform", "win32")

    def _boom():
        raise RuntimeError("COM exploded")

    assert list_audio_sessions(enumerator_factory=_boom) == []
    # A failing pid lookup must not raise either — the picker just
    # shows nothing instead of breaking startup.
    enumerator = _FakeEnumerator([[_FakeSession(100, 1, 0.5, None)]])
    assert list_audio_sessions(
        enumerator_factory=lambda: enumerator,
        process_name_for_pid=lambda pid: (_ for _ in ()).throw(
            RuntimeError("psutil exploded"))) == []
