"""Platform detection + honest capture status (Track 3: Linux support).

This is the API the UI track consumes. It answers, per platform, three
questions the old code got wrong on Linux:

1. **Which capture backend runs here?** :func:`get_capture_backend`
   returns ``"wasapi"`` (Windows), ``"linux-sounddevice"`` (Linux with
   sounddevice + PortAudio present) or ``"unavailable"``. The old UI
   message "no microphone found — Windows + pyaudiowpatch required"
   is wrong on Linux; use :func:`mic_empty_message` /
   :func:`loopback_empty_message` instead — they return the correct
   per-platform string.
2. **What exactly is missing?** :func:`describe_capture_state`
   returns a :class:`CaptureStateInfo` with the enumerated devices,
   whether the machine is in mic-only mode (Linux, no monitor source),
   and an install hint when a dependency is missing. Nothing here
   raises: on an unsupported platform every field degrades to an
   honest "unavailable" state.
3. **What should the target-app lookup match?** The Teams process
   names in ``app.config`` are Windows ``.exe`` names; on Linux the
   client runs as ``teams`` / ``ms-teams``. :func:`default_process_names`
   returns the platform-appropriate list to pass to
   ``app.audio.devices.find_teams_process``.

Backend facts (do not re-derive them in the UI):

* Windows capture is WASAPI system-wide loopback via pyaudiowpatch
  by default — but since Nov 8 2026 the system track can also be
  true per-app capture (WASAPI process loopback, Windows 10 build
  20348+) via :mod:`app.audio.perapp_capture`, selected by the
  ``capture_system_mode`` setting (``"auto"`` default). Use
  :func:`per_app_capture_status` to say which one applies here.
* Linux capture is sounddevice / PortAudio. System audio is a
  PulseAudio/PipeWire *monitor source* — also system-wide, not
  per-app, and only present when the sound server exposes one.
  Without a monitor source the app records the microphone only
  (``mic_only == True``) and says so; it never pretends the system
  track exists.
* ``sounddevice`` is an *optional* dependency: lazy import, and if
  it (or the system PortAudio library) is missing the app still runs
  — :func:`describe_capture_state` carries the guided install
  message instead of the app crashing on import.

Example (UI track)::

    from app.platform_info import describe_capture_state, get_capture_backend

    info = describe_capture_state()
    if info.backend == "unavailable":
        show_message(info.status + "\\n" + (info.install_hint or ""))
    elif info.mic_only:
        show_warning(info.status)  # mic records; system track will be empty
    mic_combo.set_empty_message(mic_empty_message(info))
    loopback_combo.set_empty_message(loopback_empty_message(info))
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from app.audio.devices import (
    AudioDeviceInfo,
    list_loopback_devices,
    list_microphones,
    load_sounddevice,
    pyaudiowpatch_status,
    sounddevice_status,
)
from app.config import TEAMS_PROCESS_NAMES

#: Backend identifiers returned by :func:`get_capture_backend`.
CAPTURE_BACKEND_WASAPI = "wasapi"
CAPTURE_BACKEND_LINUX = "linux-sounddevice"
CAPTURE_BACKEND_UNAVAILABLE = "unavailable"
CAPTURE_BACKENDS: tuple[str, ...] = (
    CAPTURE_BACKEND_WASAPI,
    CAPTURE_BACKEND_LINUX,
    CAPTURE_BACKEND_UNAVAILABLE,
)

#: Linux Teams client process names (the classic client runs as
#: ``teams``; the newer client as ``ms-teams``). The web/PWA client
#: runs inside a browser and will not match any of these — that is
#: a known gap, not a bug in the lookup.
LINUX_TEAMS_PROCESS_NAMES: tuple[str, ...] = ("teams", "ms-teams")

INSTALL_SOUNDDEVICE_LINUX = (
    "To record on Linux, install PortAudio and sounddevice:\n"
    "  Debian/Ubuntu: sudo apt install libportaudio2 && pip install sounddevice\n"
    "  Fedora:        sudo dnf install portaudio && pip install sounddevice\n"
    "Then restart OffAir. The rest of the app (transcription, "
    "diarization, browsing) works without it."
)
INSTALL_PYAUDIOWPATCH_WINDOWS = (
    "To record on Windows, install pyaudiowpatch:\n"
    "  pip install -r requirements.txt\n"
    "Then restart OffAir."
)


# ------------------------------------------------------------ platform ------
def is_windows() -> bool:
    return sys.platform == "win32"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_macos() -> bool:
    return sys.platform == "darwin"


def platform_name() -> str:
    """``"windows"`` / ``"linux"`` / ``"macos"`` / the raw sys.platform."""
    if is_windows():
        return "windows"
    if is_linux():
        return "linux"
    if is_macos():
        return "macos"
    return sys.platform


def get_capture_backend() -> str:
    """Which capture backend runs on this machine.

    ``"wasapi"`` on Windows, ``"linux-sounddevice"`` on Linux when
    sounddevice (and its PortAudio library) import, otherwise
    ``"unavailable"``. This answers *identity* only — whether the
    backend can actually record (devices present, monitor source
    found) is :func:`describe_capture_state`.
    """
    if is_windows():
        return CAPTURE_BACKEND_WASAPI
    if is_linux():
        return (CAPTURE_BACKEND_LINUX if load_sounddevice() is not None
                else CAPTURE_BACKEND_UNAVAILABLE)
    return CAPTURE_BACKEND_UNAVAILABLE


def default_process_names() -> list[str]:
    """Teams process names to look for on this platform.

    Windows: the ``.exe`` names from ``app.config``. Linux: the
    native client names first, then the Windows names (harmless —
    they simply never match on Linux).
    """
    if is_linux():
        return list(LINUX_TEAMS_PROCESS_NAMES) + list(TEAMS_PROCESS_NAMES)
    return list(TEAMS_PROCESS_NAMES)


# ------------------------------------------------------------ status --------
@dataclass
class CaptureStateInfo:
    """Everything the UI needs to say about capture on this machine.

    ``ready`` means both tracks can be recorded. ``mic_only`` means
    only the microphone can be recorded (Linux without a monitor
    source) — the system track will be an empty file and the session
    metadata records ``mode="linux-mic-only"``. ``status`` is the
    one-line summary for a label; ``detail`` is the longer
    explanation for a dialog/tooltip; ``install_hint`` is None when
    nothing needs installing.
    """

    platform: str
    backend: str
    sounddevice_available: bool
    sounddevice_error: str | None
    microphones: list[AudioDeviceInfo] = field(default_factory=list)
    loopback_devices: list[AudioDeviceInfo] = field(default_factory=list)
    mic_only: bool = False
    ready: bool = False
    status: str = ""
    detail: str = ""
    install_hint: str | None = None


def per_app_capture_status() -> tuple[bool, str | None]:
    """``(available, reason)`` for WASAPI per-app capture here.

    Thin wrapper over
    :func:`app.audio.perapp_capture.per_app_capture_available` —
    never raises; on non-Windows the reason says so. The UI uses it
    to explain which system-audio mode applies (per-app vs the
    system-wide loopback fallback).
    """
    try:
        from app.audio.perapp_capture import per_app_capture_available
        return per_app_capture_available()
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Per-app capture status check failed: {exc}"


def auto_detect_status() -> tuple[bool, str | None]:
    """``(available, reason)`` for meeting auto-detect here.

    Thin wrapper over
    :func:`app.audio.auto_detect.auto_detect_available` — never
    raises. Auto-detect is Windows-only (WASAPI session
    enumeration); on Linux/macOS the Settings checkbox is disabled
    with this reason instead of pretending the feature works.
    """
    try:
        from app.audio.auto_detect import auto_detect_available
        return auto_detect_available()
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Auto-detect status check failed: {exc}"


def outlook_status() -> tuple[bool, str | None]:
    """``(available, reason)`` for Outlook calendar linking here.

    Thin wrapper over
    :func:`app.calendar.outlook.outlook_available` — never raises.
    Linking is Windows-only (Outlook COM via pywin32); on other
    machines the Settings checkbox is disabled with this reason
    instead of pretending the feature works.
    """
    try:
        from app.calendar.outlook import outlook_available
        return outlook_available()
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Outlook status check failed: {exc}"


def describe_capture_state() -> CaptureStateInfo:
    """Describe capture on this machine, honestly, without raising."""
    backend = get_capture_backend()
    sd = load_sounddevice() if is_linux() else None
    sd_error: str | None = None
    if is_linux() and sd is None:
        sd_error = sounddevice_status()

    info = CaptureStateInfo(
        platform=platform_name(),
        backend=backend,
        sounddevice_available=sd is not None,
        sounddevice_error=sd_error,
    )

    if backend == CAPTURE_BACKEND_UNAVAILABLE:
        if is_linux():
            info.status = ("Linux capture unavailable — sounddevice / "
                           "PortAudio not installed")
            info.detail = (sd_error or "sounddevice is not available.")
            info.install_hint = INSTALL_SOUNDDEVICE_LINUX
        elif is_macos():
            info.status = "Capture is not supported on macOS yet"
            info.detail = ("OffAir records on Windows (WASAPI) and "
                           "Linux (sounddevice / PortAudio). macOS has no "
                           "capture backend in this build — the rest of "
                           "the app still works on recordings made "
                           "elsewhere.")
        else:
            info.status = f"Capture is not supported on {sys.platform}"
            info.detail = ("OffAir records on Windows (WASAPI) and "
                           "Linux (sounddevice / PortAudio).")
        return info

    # Backend exists — enumerate what it can actually see.
    info.microphones = list_microphones()
    info.loopback_devices = list_loopback_devices()

    if backend == CAPTURE_BACKEND_WASAPI:
        if pyaudiowpatch_status() != "pyaudiowpatch available":
            info.status = ("Windows capture unavailable — pyaudiowpatch "
                           "not installed")
            info.detail = pyaudiowpatch_status()
            info.install_hint = INSTALL_PYAUDIOWPATCH_WINDOWS
            return info
        if info.microphones and info.loopback_devices:
            info.ready = True
            info.status = "Ready — microphone + system loopback (WASAPI)"
            info.detail = ("System audio is system-wide WASAPI loopback, "
                           "not Teams-only: mute other apps for a clean "
                           "remote track.")
        elif info.microphones:
            info.mic_only = True
            info.status = "Microphone only — no loopback device found"
            info.detail = ("No WASAPI loopback device was found, so only "
                           "the microphone can be recorded. System audio "
                           "needs a playback device with loopback support.")
        else:
            info.status = "No audio devices found"
            info.detail = ("WASAPI is available but no microphone or "
                           "loopback device was enumerated.")
        return info

    # Linux sounddevice backend.
    if info.microphones and info.loopback_devices:
        info.ready = True
        info.status = "Ready — microphone + system monitor (sounddevice)"
        info.detail = ("System audio comes from a PulseAudio/PipeWire "
                       "monitor source, which is system-wide like WASAPI "
                       "loopback — not Teams-only. Mute other apps for a "
                       "clean remote track.")
    elif info.microphones:
        info.mic_only = True
        info.status = ("Microphone only — no PulseAudio/PipeWire monitor "
                       "source found")
        info.detail = ("sounddevice found a microphone but no monitor "
                       "source, so only the microphone can be recorded "
                       "and the system track will be empty. System audio "
                       "needs a monitor source: in PulseAudio/PipeWire "
                       "make sure a playback device exists (its monitor "
                       "is exposed automatically) and that no other app "
                       "has taken exclusive control of it.")
    elif info.loopback_devices:
        info.status = "Monitor source found, but no microphone"
        info.detail = ("A system-audio monitor source is available but "
                       "no microphone was found, so only the system track "
                       "can be recorded.")
    else:
        info.status = ("sounddevice is installed but no audio devices "
                       "were found")
        info.detail = ("PortAudio enumerated no input devices. Check the "
                       "microphone is connected and not exclusively held "
                       "by another app, and that PulseAudio/PipeWire is "
                       "running.")
    return info


def mic_empty_message(info: CaptureStateInfo | None = None) -> str:
    """Dropdown placeholder when no microphone is listed.

    Replaces the old hardcoded Windows message — on Linux it names
    the Linux cause (sounddevice missing vs no devices at all).
    Pass a :func:`describe_capture_state` result to avoid enumerating
    devices twice; otherwise one is computed here. Note the branch is
    on *platform*, not backend: on Linux without sounddevice the
    backend is "unavailable", and the message still has to name
    sounddevice, not the generic fallback.
    """
    info = info or describe_capture_state()
    if is_windows():
        return "(no microphone found — Windows + pyaudiowpatch required)"
    if is_linux():
        if not info.sounddevice_available:
            return ("(no microphone found — sounddevice / PortAudio not "
                    "installed; see README Linux section)")
        return "(no microphone found — no input device seen by PortAudio)"
    if is_macos():
        return "(no microphone found — capture is not supported on macOS yet)"
    return "(no microphone found — capture is unavailable on this platform)"


def loopback_empty_message(info: CaptureStateInfo | None = None) -> str:
    """Dropdown placeholder when no system-audio device is listed.

    On Linux a missing monitor source means mic-only mode, not a
    missing library — the message says that, so the user is not sent
    to install something that would not help. Branch is on platform,
    not backend (see :func:`mic_empty_message`).
    """
    info = info or describe_capture_state()
    if is_windows():
        return ("(no loopback device found — Windows + pyaudiowpatch "
                "required)")
    if is_linux():
        if not info.sounddevice_available:
            return ("(no system audio — sounddevice / PortAudio not "
                    "installed; see README Linux section)")
        if info.microphones:
            return ("(no monitor source found — microphone only; system "
                    "audio needs a PulseAudio/PipeWire monitor source)")
        return ("(no system audio — no monitor source seen by PortAudio)")
    if is_macos():
        return "(no system audio — capture is not supported on macOS yet)"
    return "(no system audio — capture is unavailable on this platform)"
