"""Linux capture backend — microphone + system audio via sounddevice.

This is the Linux counterpart to the WASAPI backend in
:mod:`app.audio.capture`. It reuses the whole state machine, the
incremental WAV writer, pause-discard policy and level callbacks from
:class:`~app.audio.capture.WASAPICapture` — only the hardware stream
open and the availability checks differ.

* **Microphone** — any PortAudio input device (default when no device
  index is given).
* **System audio** — a PulseAudio/PipeWire *monitor source* (an input
  device whose name contains "Monitor"; see
  :func:`app.audio.devices.list_loopback_devices`). Like WASAPI
  loopback it is **system-wide, not per-app**: it carries everything
  currently playing, including Teams and any other app making noise.
  There is no per-app isolation on Linux either.
* **Mic-only mode** — when no monitor source exists,
  :class:`LinuxDualCapture` still records the microphone and writes an
  (empty) ``system.wav`` so the session folder keeps its shape, with
  ``mode="linux-mic-only"`` and a warning in the session metadata.
  It never pretends the system track has audio in it.

``sounddevice`` is an *optional* dependency: it is imported lazily
through :func:`app.audio.devices.load_sounddevice`, and when it (or
the system PortAudio library) is missing, ``start()`` raises a
``RuntimeError`` with the install steps instead of the app crashing
on import. ``import app.audio.linux_capture`` is always safe.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from app.audio.capture import DualCapture, WASAPICapture
from app.audio.devices import default_loopback, load_sounddevice, sounddevice_status
from app.config import SAMPLE_RATE

logger = logging.getLogger("offair.linux_capture")

LINUX_MODE_SYSTEM_MONITOR = "linux-system-monitor"
LINUX_MODE_MIC_ONLY = "linux-mic-only"

MONITOR_WARNING = (
    "System audio comes from a PulseAudio/PipeWire monitor source, "
    "which is system-wide like WASAPI loopback — NOT Teams-only. "
    "Mute other apps for a clean remote track.")
MIC_ONLY_WARNING = (
    "No PulseAudio/PipeWire monitor source was found, so only the "
    "microphone is being recorded and system.wav will be empty. "
    "System audio needs a monitor source (see README, Linux section).")


class LinuxSounddeviceCapture(WASAPICapture):
    """One audio channel (mic *or* system monitor) via sounddevice.

    ``kind`` is ``"mic"`` or ``"system"`` (inherited from
    :class:`WASAPICapture`). For ``kind == "system"``,
    ``device_index`` must be a monitor-source index — passing None
    would silently record the default *microphone* again, so
    :meth:`_require_hardware` refuses it with a mic-only message.
    """

    def _require_hardware(self) -> None:
        """Raise a clear, Linux-specific error when capture cannot run."""
        if sys.platform != "linux" and not sys.platform.startswith("linux"):
            raise RuntimeError(
                "The Linux sounddevice backend runs on Linux only "
                f"(this machine is {sys.platform}). On Windows use the "
                "WASAPI backend (app.audio.capture.DualCapture).")
        if load_sounddevice() is None:
            raise RuntimeError(
                "Linux capture needs sounddevice + PortAudio. "
                f"{sounddevice_status()} "
                "Install: Debian/Ubuntu `sudo apt install libportaudio2 "
                "&& pip install sounddevice` (Fedora: `sudo dnf install "
                "portaudio && pip install sounddevice`), then restart.")
        if self.kind == "system" and self.device_index is None:
            raise RuntimeError(
                "No PulseAudio/PipeWire monitor source selected — "
                "system audio cannot be recorded on this machine. "
                "Record microphone-only (LinuxDualCapture does this "
                "automatically), or add a playback device so its "
                "monitor source appears.")

    def _run_hardware(self) -> None:
        """Live PortAudio input stream. Linux + sounddevice only.

        Opened lazily here (not at import) so the module imports
        cleanly on Windows/macOS and on Linux without sounddevice.
        """
        sd = load_sounddevice()
        if sd is None:  # _require_hardware already checked; be safe anyway.
            raise RuntimeError(
                "sounddevice is not available: " + sounddevice_status())
        logger.info("Linux capture (%s): opening PortAudio input stream "
                    "(device=%s, %d Hz)", self.kind, self.device_index,
                    self.sample_rate)
        with sd.InputStream(device=self.device_index,
                            channels=self.channels,
                            samplerate=self.sample_rate,
                            dtype="int16",
                            blocksize=self.chunk_frames) as stream:
            while not self._stop_event.is_set():
                try:
                    data, overflowed = stream.read(self.chunk_frames)
                except Exception:
                    if self._stop_event.is_set():
                        break
                    raise
                if overflowed:
                    logger.warning("Linux capture (%s): PortAudio input "
                                   "overflow — audio was dropped",
                                   self.kind)
                self._handle_chunk(data)


class LinuxMicCapture(LinuxSounddeviceCapture):
    kind = "mic"


class LinuxSystemCapture(LinuxSounddeviceCapture):
    """System audio from a PulseAudio/PipeWire monitor source.

    Honest limitation (see module docstring): the monitor source is
    system-wide, so this records everything playing — not just the
    target app. ``mode`` / ``warning`` record that, mirroring
    :class:`~app.audio.capture.PerAppCapture` on Windows.
    """

    kind = "system"
    mode = LINUX_MODE_SYSTEM_MONITOR
    warning = MONITOR_WARNING

    def __init__(self, output_path: str | Path, *,
                 process_name: str = "ms-teams.exe",
                 process_names: list[str] | None = None, **kwargs) -> None:
        # Accepted for signature compatibility with PerAppCapture /
        # DualCapture. Linux has no per-app capture to target, so the
        # names are recorded only — the monitor source records every
        # app's audio regardless (see the module docstring).
        super().__init__(output_path, **kwargs)
        self.process_name = process_name
        self.process_names = process_names or [process_name]


class LinuxDualCapture(DualCapture):
    """Mic + system-monitor capture together, plus the combined mix.

    Picks the default monitor source when the caller does not pass
    one; when no monitor source exists at all it degrades to
    mic-only mode (``mode == "linux-mic-only"``): the microphone
    still records, ``system.wav`` is written empty, and the warning
    explains why. ``process_name`` is accepted for signature
    compatibility with :class:`DualCapture` and recorded in metadata
    only — Linux has no per-app capture to target.
    """

    def __init__(self, session_dir: str | Path, *,
                 mic_device_index: int | None = None,
                 loopback_device_index: int | None = None,
                 process_name: str = "ms-teams.exe",
                 sample_rate: int = SAMPLE_RATE,
                 mic_level_callback=None,
                 system_level_callback=None,
                 chunk_source_factory=None) -> None:
        if loopback_device_index is None:
            monitor = default_loopback()
            if monitor is not None:
                loopback_device_index = monitor.index
        if loopback_device_index is None:
            super().__init__(
                session_dir,
                mic_device_index=mic_device_index,
                loopback_device_index=None,
                process_name=process_name,
                sample_rate=sample_rate,
                mic_level_callback=mic_level_callback,
                system_level_callback=system_level_callback,
                chunk_source_factory=chunk_source_factory,
                mic_cls=LinuxMicCapture,
                system_cls=None,
                mode=LINUX_MODE_MIC_ONLY,
                warning=MIC_ONLY_WARNING)
            logger.warning("LinuxDualCapture: %s", MIC_ONLY_WARNING)
        else:
            super().__init__(
                session_dir,
                mic_device_index=mic_device_index,
                loopback_device_index=loopback_device_index,
                process_name=process_name,
                sample_rate=sample_rate,
                mic_level_callback=mic_level_callback,
                system_level_callback=system_level_callback,
                chunk_source_factory=chunk_source_factory,
                mic_cls=LinuxMicCapture,
                system_cls=LinuxSystemCapture,
                mode=LINUX_MODE_SYSTEM_MONITOR,
                warning=MONITOR_WARNING)
