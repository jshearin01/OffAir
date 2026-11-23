"""Audio device enumeration (microphones + WASAPI loopback / Linux monitor).

Platforms:

* **Windows** — WASAPI loopback comes from ``pyaudiowpatch``.
* **Linux** — microphones and system audio come from ``sounddevice`` /
  PortAudio. System audio is a PulseAudio/PipeWire *monitor source*
  (an input device whose name contains "Monitor"), which — like WASAPI
  loopback — is system-wide, not per-app. When no monitor source
  exists, only the microphone is available (mic-only mode).
* **Anything else** — no capture backend; every function degrades
  gracefully and returns an empty list / ``None``, never raising on
  import or on call. The UI uses that to show a clear "no devices"
  message instead of crashing.

``sounddevice`` is an *optional* dependency: it is imported lazily,
and when it (or the system PortAudio library it needs) is missing the
functions below return ``[]`` and :func:`sounddevice_status` explains
how to install it. See :mod:`app.platform_info` for the per-platform
status strings the UI should show.
"""
from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass

from app.config import TEAMS_PROCESS_NAMES

logger = logging.getLogger("offair.devices")

_pyaudiowpatch_error: str | None = None
_sounddevice_error: str | None = None

#: Serialises PortAudio initialisation/termination across threads.
#: ``PyAudio()`` (``Pa_Initialize``) is not thread-safe: two threads
#: constructing ``PyAudio`` at once crash the process with an access
#: violation (0xc0000005 — Nov 10 2026, reproduced twice on real
#: Windows: the mic capture thread inside ``PyAudio()`` in
#: ``_run_hardware`` while the calling thread negotiated the system
#: format in ``_require_hardware``). Every ``PyAudio()`` construction
#: and ``terminate()`` in this package goes through this lock — here
#: and in :mod:`app.audio.capture`. It is held only for the brief
#: construct (+ open) / terminate, never across a recording.
PYAUDIO_INIT_LOCK = threading.Lock()


def _load_pyaudiowpatch():  # type: ignore[no-untyped-def]
    """Return the pyaudiowpatch module, or None (with the reason recorded)."""
    global _pyaudiowpatch_error
    if sys.platform != "win32":
        _pyaudiowpatch_error = (
            f"pyaudiowpatch / WASAPI is Windows-only (platform={sys.platform})")
        return None
    try:
        import pyaudiowpatch as p  # type: ignore[import-not-found]
    except Exception as exc:  # ImportError and friends — never crash here.
        _pyaudiowpatch_error = f"pyaudiowpatch unavailable: {exc}"
        return None
    _pyaudiowpatch_error = None
    return p


def pyaudiowpatch_status() -> str:
    """Human-readable reason why WASAPI capture is / is not available."""
    _load_pyaudiowpatch()
    return _pyaudiowpatch_error or "pyaudiowpatch available"


def load_sounddevice():  # type: ignore[no-untyped-def]
    """Return the sounddevice module, or None (with the reason recorded).

    Lazy import — sounddevice is an *optional* dependency, so importing
    this module must never fail when it is missing. On Linux the import
    itself raises ``OSError: PortAudio library not found`` when the
    system PortAudio library (``libportaudio2`` on Debian/Ubuntu) is
    not installed, even if the Python package is present; that case
    is reported distinctly so the install guidance can name the
    system package instead of the pip package.
    """
    global _sounddevice_error
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except Exception as exc:  # ImportError, OSError, ... — never crash.
        if "PortAudio library not found" in str(exc):
            _sounddevice_error = (
                "sounddevice is installed but the PortAudio library is "
                "missing — install the system package "
                "(Debian/Ubuntu: `sudo apt install libportaudio2`; "
                "Fedora: `sudo dnf install portaudio`)")
        else:
            _sounddevice_error = (
                "sounddevice is not installed — install it with "
                "`pip install sounddevice` (and the PortAudio system "
                "library, e.g. `sudo apt install libportaudio2` on "
                "Debian/Ubuntu). "
                f"Underlying error: {exc}")
        return None
    _sounddevice_error = None
    return sd


def sounddevice_status() -> str:
    """Human-readable reason why Linux (sounddevice) capture is / is not available."""
    load_sounddevice()
    return _sounddevice_error or "sounddevice available"


def _is_monitor_source_name(name: str) -> bool:
    """True for a PulseAudio/PipeWire monitor source.

    PortAudio exposes the monitor of an output device as an *input*
    device whose name contains "Monitor" (e.g. "Monitor of Built-in
    Audio Analog Stereo"). That is the Linux equivalent of a WASAPI
    loopback device: it carries everything currently playing, not
    one app's audio.
    """
    return "monitor" in name.lower()


def _list_linux_devices() -> tuple[list[AudioDeviceInfo], list[AudioDeviceInfo]]:
    """(microphones, monitor/loopback devices) via sounddevice/PortAudio."""
    sd = load_sounddevice()
    if sd is None:
        logger.debug("Linux device enumeration: %s", _sounddevice_error)
        return [], []
    try:
        infos = sd.query_devices()
    except Exception:
        logger.exception("Failed to enumerate Linux audio devices")
        return [], []
    default_input: int | None = None
    try:
        default_input = int(sd.query_devices(kind="input")["index"])
    except Exception:
        pass
    mics: list[AudioDeviceInfo] = []
    loops: list[AudioDeviceInfo] = []
    for info in infos:
        try:
            if int(info.get("max_input_channels", 0)) <= 0:
                continue
            index = int(info["index"])
            name = str(info.get("name", f"Input {index}"))
        except Exception:
            continue
        entry = AudioDeviceInfo(
            index=index,
            name=name,
            is_loopback=_is_monitor_source_name(name),
            is_default=(default_input is not None and index == default_input),
            max_input_channels=int(info.get("max_input_channels", 0)),
            default_sample_rate=float(info.get("default_samplerate", 0) or 0),
        )
        (loops if entry.is_loopback else mics).append(entry)
    return mics, loops


@dataclass(frozen=True)
class AudioDeviceInfo:
    """One audio device, as shown in the UI dropdowns."""

    index: int
    name: str
    is_loopback: bool = False
    is_default: bool = False
    max_input_channels: int = 0
    default_sample_rate: float = 0.0

    def display_name(self) -> str:
        label = self.name
        if self.is_default:
            label += " (default)"
        return label


def list_microphones() -> list[AudioDeviceInfo]:
    """All input (microphone) devices. Empty list when capture is unavailable.

    Windows: WASAPI via pyaudiowpatch. Linux: PortAudio inputs via
    sounddevice (monitor sources are excluded — see
    :func:`list_loopback_devices`). Anything else: ``[]``.
    """
    if sys.platform.startswith("linux"):
        return _list_linux_devices()[0]
    if sys.platform != "win32":
        return []
    p = _load_pyaudiowpatch()
    if p is None:
        logger.debug("list_microphones: %s", _pyaudiowpatch_error)
        return []
    devices: list[AudioDeviceInfo] = []
    try:
        # Serialised: a device refresh while a capture is starting
        # must not construct PyAudio on this thread at the same time
        # as a capture thread (see PYAUDIO_INIT_LOCK).
        with PYAUDIO_INIT_LOCK, p.PyAudio() as audio:
            default_index = None
            try:
                default_index = audio.get_default_input_device_info()["index"]
            except Exception:
                pass
            for index in range(audio.get_device_count()):
                try:
                    info = audio.get_device_info_by_index(index)
                except Exception:
                    continue
                if int(info.get("maxInputChannels", 0)) <= 0:
                    continue
                devices.append(AudioDeviceInfo(
                    index=int(index),
                    name=str(info.get("name", f"Input {index}")),
                    is_loopback=False,
                    is_default=(index == default_index),
                    max_input_channels=int(info.get("maxInputChannels", 0)),
                    default_sample_rate=float(info.get("defaultSampleRate", 0) or 0),
                ))
    except Exception:
        logger.exception("Failed to enumerate microphones")
        return []
    return devices


def list_loopback_devices() -> list[AudioDeviceInfo]:
    """All system-audio (loopback / monitor) devices. Empty when unavailable.

    Windows: WASAPI loopback via pyaudiowpatch. Linux: PulseAudio /
    PipeWire monitor sources via sounddevice. Anything else: ``[]``.
    """
    if sys.platform.startswith("linux"):
        return _list_linux_devices()[1]
    if sys.platform != "win32":
        return []
    p = _load_pyaudiowpatch()
    if p is None:
        logger.debug("list_loopback_devices: %s", _pyaudiowpatch_error)
        return []
    devices: list[AudioDeviceInfo] = []
    try:
        # Serialised — see the note on the microphone list above.
        with PYAUDIO_INIT_LOCK, p.PyAudio() as audio:
            try:
                generator = audio.get_loopback_device_info_generator()
            except Exception:
                logger.exception("Loopback enumeration not supported")
                return []
            for info in generator:
                devices.append(AudioDeviceInfo(
                    index=int(info["index"]),
                    name=str(info.get("name", f"Loopback {info['index']}")),
                    is_loopback=True,
                    is_default=bool(info.get("isDefault", False)),
                    max_input_channels=int(info.get("maxInputChannels", 0)),
                    default_sample_rate=float(info.get("defaultSampleRate", 0) or 0),
                ))
    except Exception:
        logger.exception("Failed to enumerate loopback devices")
        return []
    return devices


def default_microphone() -> AudioDeviceInfo | None:
    """The default microphone, or the first one found, or None."""
    mics = list_microphones()
    if not mics:
        return None
    for mic in mics:
        if mic.is_default:
            return mic
    return mics[0]


def default_loopback() -> AudioDeviceInfo | None:
    """The default loopback device, or the first one found, or None."""
    devices = list_loopback_devices()
    if not devices:
        return None
    for device in devices:
        if device.is_default:
            return device
    return devices[0]


def find_teams_process(
    process_names: list[str] | None = None,
) -> dict[str, object] | None:
    """Return ``{"pid", "name", "exe"}`` for a running Teams process, else None.

    Uses psutil when available; returns None (never raises) when psutil is
    missing or no matching process is running. Matching is by exact
    executable name, case-insensitive.
    """
    names = {n.lower() for n in (process_names or TEAMS_PROCESS_NAMES)}
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        logger.debug("find_teams_process: psutil unavailable")
        return None
    try:
        for proc in psutil.process_iter(["pid", "name", "exe"]):
            try:
                info = proc.info
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if str(info.get("name") or "").lower() in names:
                return {
                    "pid": int(info["pid"]),
                    "name": str(info["name"]),
                    "exe": str(info.get("exe") or ""),
                }
    except Exception:
        logger.exception("Failed to scan processes")
        return None
    return None
