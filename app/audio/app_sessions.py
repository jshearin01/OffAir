"""List the apps that currently have audio sessions (Windows).

This is the target-app picker backend: it enumerates **all** render
endpoints' WASAPI audio sessions and reports, per process, the exe
name, a friendly display name, the pid, whether the session is
actively rendering sound, and the current peak level. The main
window's app picker is built from this list, so the user picks a
real app that is actually producing (or able to produce) sound
instead of typing an exe name blind — the work-PC failure (Nov 10
2026: mic recorded, system track recorded nothing, and no way to
tell which app was being captured) is exactly what this prevents.

The COM enumeration helpers (``_enumerate_endpoints``,
``_session_manager_for``, ``_read_session``) live here and are
shared with :mod:`app.audio.auto_detect` — the watcher needs the
same session reads for one target pid, and the two modules must
not drift into separate COM implementations. ``app.audio.auto_detect``
imports them from here.

Platform: **Windows only.** On any other platform
:func:`list_audio_sessions` returns ``[]`` and
:func:`audio_sessions_available` explains why — never raises, so
the picker degrades to its "Other app…" manual entry instead of
breaking startup. No fake Linux detection is attempted.

What is **not** verified here: no real Windows run exists yet —
this Linux VM has no Windows audio stack, so the COM path is
exercised only through injected fake enumerators (see
``tests/test_app_sessions.py``). The first real Windows run — a
Teams call plus a Chrome video playing — is the open gate.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: ``AudioSessionState`` (audiopolicy.h): 0 inactive, 1 active, 2 expired.
SESSION_STATE_INACTIVE = "inactive"
SESSION_STATE_ACTIVE = "active"
SESSION_STATE_EXPIRED = "expired"
#: ``EDataFlow`` (mmdeviceapi.h): 0 render, 1 capture.
EDATAFLOW_RENDER = 0
EDATAFLOW_CAPTURE = 1
#: ``DEVICE_STATE_ACTIVE`` (mmdeviceapi.h).
DEVICE_STATE_ACTIVE = 0x00000001
#: ``CLSCTX_ALL`` (comtypes) — the activation context for COM objects.
CLSCTX_ALL = 23
#: ``CLSID_MMDeviceEnumerator`` (mmdeviceapi.h).
CLSID_MMDEVICE_ENUMERATOR = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"

#: Sentinel data value for the picker's "Other app…" row (the UI
#: stores the manually typed exe name separately).
OTHER_APP_SENTINEL = "__other_app__"


@dataclass
class AudioSessionInfo:
    """One process with at least one render audio session.

    ``exe_name`` is the process exe (``"chrome.exe"``) — the value
    per-app capture needs. ``display_name`` is what the picker
    shows (the session's display name when the app set one, else a
    friendly name for the exe). ``state`` is ``"active"`` when the
    session is Active (rendering sound right now) and ``"idle"``
    otherwise (session exists but is inactive/expired — the app is
    open but silent, or the session is stale). ``peak`` is the
    session's current peak level (0.0–1.0) or ``None`` when it
    could not be read.
    """

    exe_name: str
    display_name: str
    pid: int
    state: str
    peak: float | None = None


#: Friendly names for the apps people actually record. Anything not
#: listed here falls back to :func:`prettify_exe_name` (strip
#: ``.exe``, title-case). Keys are lower-case exe names.
KNOWN_APP_NAMES: dict[str, str] = {
    "ms-teams.exe": "Microsoft Teams",
    "teams.exe": "Microsoft Teams",
    "ms-teamsupdate.exe": "Microsoft Teams",
    "chrome.exe": "Google Chrome",
    "msedge.exe": "Microsoft Edge",
    "firefox.exe": "Mozilla Firefox",
    "spotify.exe": "Spotify",
    "zoom.exe": "Zoom",
    "outlook.exe": "Microsoft Outlook",
    "powerpnt.exe": "Microsoft PowerPoint",
    "winword.exe": "Microsoft Word",
    "excel.exe": "Microsoft Excel",
    "vlc.exe": "VLC media player",
    "wmplayer.exe": "Windows Media Player",
    "discord.exe": "Discord",
    "slack.exe": "Slack",
    "obs64.exe": "OBS Studio",
    "code.exe": "Visual Studio Code",
    "notepad.exe": "Notepad",
}


def prettify_exe_name(exe_name: str) -> str:
    """Turn an exe name into a readable name: ``"chrome.exe"`` →
    ``"Chrome"``, ``"my-app.exe"`` → ``"My App"``.

    The known-app map (:data:`KNOWN_APP_NAMES`) wins first — this
    is only the generic fallback for unlisted exes.
    """
    known = KNOWN_APP_NAMES.get(exe_name.lower())
    if known is not None:
        return known
    stem = exe_name
    if stem.lower().endswith(".exe"):
        stem = stem[:-4]
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem.title() if stem else exe_name


def display_name_for_exe(exe_name: str,
                         session_display_name: str | None = None) -> str:
    """Friendly name for an exe, preferring the session's own name.

    WASAPI sessions sometimes carry a display name the app set
    itself (e.g. a window title); when present and non-blank it is
    the most accurate label. Otherwise the known-app map /
    prettified exe name is used.
    """
    if session_display_name and session_display_name.strip():
        return session_display_name.strip()
    return prettify_exe_name(exe_name)


def audio_sessions_available() -> tuple[bool, str | None]:
    """Is audio-session listing usable on this machine?

    Returns ``(True, None)`` or ``(False, reason)``. Never raises.
    Needs Windows (WASAPI session enumeration) + pycaw.
    """
    if sys.platform != "win32":
        return False, (
            "Audio-session listing needs Windows (WASAPI audio "
            f"sessions); this machine is {sys.platform}. Type the "
            "app's exe name instead.")
    try:
        import pycaw  # noqa: F401  # type: ignore[import-not-found]
        import comtypes  # noqa: F401
    except Exception as exc:  # pragma: no cover - Windows-only branch
        return False, (
            "Audio-session listing needs the 'pycaw' and 'comtypes' "
            f"packages ({exc}); install them with "
            "`pip install -r requirements.txt`.")
    return True, None


# --------------------------------------------------------------------------
# Shared COM enumeration (also used by app.audio.auto_detect)
# --------------------------------------------------------------------------
def _enumerate_endpoints(enumerator, data_flow: int) -> list:
    """All active endpoints for ``data_flow`` (render/capture)."""
    collection = enumerator.EnumAudioEndpoints(
        data_flow, DEVICE_STATE_ACTIVE)
    # comtypes returns ["out"] params; they are not passed.
    count = collection.GetCount()
    if isinstance(count, tuple):
        count = count[0]
    devices = []
    for i in range(count):
        device = collection.Item(i)
        devices.append(device)
    return devices


def _session_manager_for(device, manager_cls):
    """The IAudioSessionManager2 for one endpoint device."""
    # Activate returns POINTER(IUnknown) — the manager interface must
    # be queried explicitly (verified on real Windows Nov 8 2026:
    # calling GetSessionEnumerator on the raw pointer raises
    # AttributeError, so the probe never saw any session).
    activated = device.Activate(manager_cls._iid_, CLSCTX_ALL, None)
    return activated.QueryInterface(manager_cls)


def _read_session(ctl, meter_cls) -> tuple[str | None, float | None]:
    """(state, peak) for one session control, best-effort."""
    state = None
    try:
        # comtypes returns ["out"] params; they are not passed.
        raw_state = ctl.GetState()
        if isinstance(raw_state, tuple):
            raw_state = raw_state[0]
        state = {0: SESSION_STATE_INACTIVE,
                 1: SESSION_STATE_ACTIVE,
                 2: SESSION_STATE_EXPIRED}.get(raw_state)
    except Exception:
        state = None
    peak = None
    try:
        # QueryInterface takes the interface CLASS, not its iid —
        # passing the iid raises TypeError and peak silently stayed
        # None on real Windows.
        meter = ctl.QueryInterface(meter_cls)
        res = meter.GetPeakValue()
        if isinstance(res, tuple):
            res = res[0]
        peak = float(res)
    except Exception:
        peak = None
    return state, peak


def _iter_render_sessions(enumerator, manager_cls, meter_cls,
                          control2_cls):
    """Yield ``{pid, state, peak, display_name}`` for every render
    session on every active endpoint.

    Sessions the probe cannot read (a control that raises, a pid
    that cannot be queried) are skipped, never fatal — one broken
    session must not hide every other app.
    """
    for device in _enumerate_endpoints(enumerator, EDATAFLOW_RENDER):
        try:
            manager = _session_manager_for(device, manager_cls)
            session_enum = manager.GetSessionEnumerator()
            # comtypes returns ["out"] params; they are not passed.
            count = session_enum.GetCount()
            if isinstance(count, tuple):
                count = count[0]
            for i in range(count):
                try:
                    ctl = session_enum.GetSession(i)
                    # GetSession returns IAudioSessionControl; the pid
                    # lives on IAudioSessionControl2 (pycaw's own
                    # GetAllSessions does the same QueryInterface).
                    ctl2 = ctl.QueryInterface(control2_cls)
                    proc = ctl2.GetProcessId()
                    if isinstance(proc, tuple):
                        proc = proc[0]
                    state, peak = _read_session(ctl2, meter_cls)
                    name = None
                    try:
                        name = ctl.GetDisplayName() or None
                    except Exception:
                        name = None
                    yield {"pid": proc, "state": state, "peak": peak,
                           "display_name": name}
                except Exception:
                    continue
        except Exception:
            continue


def _default_enumerator_factory():
    """Create the real IMMDeviceEnumerator (Windows only)."""
    import comtypes
    from pycaw.api.mmdeviceapi import IMMDeviceEnumerator

    return comtypes.CoCreateInstance(
        comtypes.GUID(CLSID_MMDEVICE_ENUMERATOR),
        IMMDeviceEnumerator, CLSCTX_ALL)


def _default_process_name_for_pid(pid: int) -> str | None:
    """Exe name for a pid via psutil, or None when unknowable."""
    try:
        import psutil

        return psutil.Process(pid).name() or None
    except Exception:
        return None


def list_audio_sessions(*, enumerator_factory=None,
                        process_name_for_pid=None,
                        ) -> list[AudioSessionInfo]:
    """Every process with a render audio session, best first.

    Windows: enumerates **all** render endpoints (a Bluetooth
    headset or second output must not hide an app). One entry per
    pid — a process with sessions on several endpoints appears
    once, preferring its Active session. Sorted Active first, then
    by display name, so the app currently making sound is at the
    top of the picker. Non-Windows, or any enumeration failure:
    returns ``[]`` (the picker falls back to its manual entry) —
    never raises.
    """
    if sys.platform != "win32":
        return []
    try:
        from pycaw.api.audiopolicy import (
            IAudioSessionControl2,
            IAudioSessionManager2,
        )
        from pycaw.api.endpointvolume import IAudioMeterInformation
    except Exception as exc:  # pragma: no cover - Windows-only branch
        logger.warning("Audio-session listing unavailable: %s", exc)
        return []
    factory = enumerator_factory or _default_enumerator_factory
    name_for_pid = process_name_for_pid or _default_process_name_for_pid
    try:
        enumerator = factory()
    except Exception as exc:  # noqa: BLE001 — listing must not raise
        logger.warning("Audio-session enumeration failed: %s", exc)
        return []
    by_pid: dict[int, AudioSessionInfo] = {}
    try:
        rows = _iter_render_sessions(
            enumerator, IAudioSessionManager2, IAudioMeterInformation,
            IAudioSessionControl2)
        for row in rows:
            pid = row["pid"]
            exe = name_for_pid(pid)
            if not exe:
                continue
            state = ("active" if row["state"] == SESSION_STATE_ACTIVE
                     else "idle")
            info = AudioSessionInfo(
                exe_name=exe,
                display_name=display_name_for_exe(
                    exe, row["display_name"]),
                pid=pid, state=state, peak=row["peak"])
            previous = by_pid.get(pid)
            if previous is None:
                by_pid[pid] = info
            elif previous.state != "active" and state == "active":
                # Prefer the Active session when a pid renders on
                # several endpoints — it is the one making sound.
                by_pid[pid] = info
    except Exception as exc:  # noqa: BLE001 — listing must not raise
        logger.warning("Audio-session enumeration failed: %s", exc)
        return []
    sessions = sorted(
        by_pid.values(),
        key=lambda s: (s.state != "active",
                       s.display_name.lower()))
    return sessions


def find_session(sessions: list[AudioSessionInfo],
                 exe_name: str) -> AudioSessionInfo | None:
    """The session for ``exe_name`` (case-insensitive), or None."""
    wanted = exe_name.lower()
    for session in sessions:
        if session.exe_name.lower() == wanted:
            return session
    return None


def _process_rows(process_iter=None) -> list[tuple[int, str, int | None]]:
    """``[(pid, name, ppid)]`` for the process table, ``[]`` on failure.

    ``process_iter`` is the ``psutil`` protocol used by
    :func:`app.audio.perapp_capture.resolve_target_pid` (a callable
    taking an attrs list, yielding objects with ``.info`` holding
    ``pid`` / ``name`` / ``ppid``); defaults to
    ``psutil.process_iter``. Never raises: an unreadable table
    degrades the tree search to a plain exe-name match.
    """
    try:
        import psutil

        iterator = (process_iter if process_iter is not None
                    else psutil.process_iter)
        rows = []
        for proc in iterator(["pid", "name", "ppid"]):
            try:
                info = proc.info
            except Exception:
                continue
            rows.append((info["pid"], info.get("name") or "",
                         info.get("ppid")))
        return rows
    except Exception:
        return []


def _tree_pids(root_pids, rows) -> set[int]:
    """Root pids plus all descendants, from ``rows`` (see ``_process_rows``).

    Pure function over a snapshot; cycles cannot loop (``seen``).
    """
    children: dict[int | None, list[int]] = {}
    for pid, _name, ppid in rows:
        children.setdefault(ppid, []).append(pid)
    seen = set(root_pids)
    stack = list(root_pids)
    while stack:
        pid = stack.pop()
        for child in children.get(pid, ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def find_session_for_app(sessions: list[AudioSessionInfo],
                         exe_name: str, *,
                         process_iter=None) -> AudioSessionInfo | None:
    """The audio session for an app: the session actually rendering.

    A direct exe-name match wins first. Otherwise, any session
    whose pid is in the process tree of a process with that exe
    name: the renderer may be a child with a different exe (the
    new Teams client's WebView2 host, a browser's audio service),
    and the plain exe match would report "no session" for an app
    that is playing sound — the Nov 24 2026 live-call report.
    Active sessions win over idle ones in both passes. Never
    raises; without a readable process table it is exactly
    :func:`find_session`.
    """
    ordered = sorted(sessions, key=lambda s: s.state != "active")
    direct = find_session(ordered, exe_name)
    if direct is not None:
        return direct
    rows = _process_rows(process_iter)
    if not rows:
        return None
    wanted = exe_name.lower()
    roots = {pid for pid, name, _ppid in rows
             if name.lower() == wanted}
    if not roots:
        return None
    tree = _tree_pids(roots, rows)
    for session in ordered:
        if session.pid in tree:
            return session
    return None


__all__ = [
    "AudioSessionInfo",
    "KNOWN_APP_NAMES",
    "OTHER_APP_SENTINEL",
    "audio_sessions_available",
    "display_name_for_exe",
    "find_session",
    "find_session_for_app",
    "list_audio_sessions",
    "prettify_exe_name",
]
