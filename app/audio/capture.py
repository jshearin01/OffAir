"""WASAPI audio capture (Phase 1).

Design notes — read before assuming per-app isolation works:

* ``pyaudiowpatch`` exposes **system-wide** WASAPI loopback only. There is
  no public API in it to capture a single process's audio stream. So
  :class:`PerAppCapture` is honest about that: it finds the target process
  (so the UI can show it is running and which mode is in effect), then
  records system loopback and records ``mode="system-loopback-fallback"``
  plus a warning in the session metadata. It never claims isolation it
  does not have. True per-app capture now lives in
  :mod:`app.audio.perapp_capture` (WASAPI process loopback,
  Windows 10 build 20348+); :class:`DualCapture` selects it by default
  in ``"auto"`` mode and falls back to this class when it cannot start.
  See ``docs/PER_APP_CAPTURE.md``.
* Pause **discards** audio received while paused (the recording timeline
  has a gap, rather than a time-shifted track). This keeps mic and system
  tracks aligned with each other and with wall-clock time. The choice is
  recorded in the session metadata.
* WAV files are written incrementally, chunk by chunk, so a crash loses
  at most the in-flight chunk — the file on disk is always a valid WAV.
  The writer prefers ``soundfile`` and falls back to stdlib ``wave``.
* The capture thread never touches the UI directly; levels go out through
  the ``level_callback`` (the UI marshals them onto the Qt thread).
* ``PyAudio()`` construction / ``terminate()`` is serialised through
  ``app.audio.devices.PYAUDIO_INIT_LOCK`` — PortAudio's ``Pa_Initialize``
  is not thread-safe, and two threads constructing ``PyAudio`` at once
  (the mic capture thread in ``_run_hardware`` while the calling thread
  negotiated the system format in ``_require_hardware``) crashed the
  process with an access violation (0xc0000005, Nov 10 2026, reproduced
  twice on real Windows). The lock is held only for the brief construct
  (+ open) / terminate, never across a recording.
* Native-rate streams (Nov 10 2026): a device is opened at a rate
  PortAudio actually supports — pre-flighted with
  ``is_format_supported`` before ``open`` is ever called — and the
  stream is resampled to the 48 kHz pipeline rate in ``_handle_chunk``,
  so the WAVs stay 48 kHz mono even when the device is 44.1 kHz (an
  RDP "Remote Audio" endpoint rejects a 48 kHz open outright, and on
  the frozen app that failed open is adjacent to a native heap
  corruption crash). The native rate is recorded in the session
  metadata (``mic_stream_sample_rate`` / ``system_stream_sample_rate``).
* ``chunk_source`` is a test / development seam: a callable returning an
  iterable of int16 mono PCM chunks (bytes or numpy arrays). When given,
  no audio hardware is touched — this is how the tests exercise the whole
  state machine on a machine with no microphone at all.
* Pre-roll (Nov 21 2026): :meth:`WASAPICapture.set_preroll` queues
  buffered audio to lead the next recording, and
  :meth:`DualCapture.start` accepts ``preroll_samples`` (the drained
  :class:`app.audio.preroll.PreRollBuffer`). With no pre-roll queued,
  capture is byte-identical to before. The always-on feeder that
  fills the buffer is :class:`app.audio.preroll_feeder.PreRollFeeder`
  (UI layer driven) — see ``app/audio/preroll.py``.
"""
from __future__ import annotations

import logging
import threading
import time
import wave
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from app.audio.devices import PYAUDIO_INIT_LOCK, find_teams_process
from app.audio.mixer import rms_level
from app.config import (
    CAPTURE_SYSTEM_MODE_DEFAULT,
    CHANNELS,
    CHUNK_FRAMES,
    SAMPLE_RATE,
)

logger = logging.getLogger("offair.capture")

LevelCallback = Callable[[float], None]
ChunkSource = Callable[[], Iterable[bytes | np.ndarray]]


class CaptureState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class _IncrementalWavWriter:
    """Write int16 mono PCM incrementally; always a valid WAV on disk."""

    def __init__(self, path: Path, sample_rate: int, channels: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames_written = 0
        self._sf = None
        self._wav = None
        try:
            import soundfile as sf  # type: ignore[import-not-found]
            self._sf = sf.SoundFile(
                str(self.path), mode="w", samplerate=sample_rate,
                channels=channels, subtype="PCM_16")
        except Exception:
            self._sf = None
            self._wav = wave.open(str(self.path), "wb")
            self._wav.setnchannels(channels)
            self._wav.setsampwidth(2)
            self._wav.setframerate(sample_rate)

    def write(self, samples: np.ndarray) -> None:
        arr = np.asarray(samples, dtype=np.int16).ravel()
        if arr.size == 0:
            return
        if self._sf is not None:
            self._sf.write(arr)
        else:
            assert self._wav is not None
            self._wav.writeframes(arr.tobytes())
        self.frames_written += int(arr.size // self.channels)

    def close(self) -> None:
        if self._sf is not None:
            self._sf.close()
            self._sf = None
        if self._wav is not None:
            self._wav.close()
            self._wav = None

    @property
    def duration_seconds(self) -> float:
        return self.frames_written / self.sample_rate if self.sample_rate else 0.0


def _to_int16_mono(chunk: bytes | np.ndarray, channels: int = 1) -> np.ndarray:
    if isinstance(chunk, (bytes, bytearray)):
        arr = np.frombuffer(bytes(chunk), dtype=np.int16)
    else:
        arr = np.asarray(chunk)
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, -1.0, 1.0)
            arr = (arr * 32767).astype(np.int16)
        else:
            arr = arr.astype(np.int16)
    if channels > 1 and arr.size:
        arr = arr.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return arr.ravel().astype(np.int16)


class _StreamingResampler:
    """Linear-interpolation resampler that keeps its phase across chunks.

    Feed int16 mono samples at ``src_rate``; get int16 mono samples at
    ``dst_rate``. The fractional output position and the un-consumed
    input tail are carried between :meth:`push` calls, so a long
    stream resamples without drift or boundary clicks — resampling
    each chunk independently would restart the grid every chunk and
    drift by up to a sample per chunk. Speech-grade, not studio-grade
    (same trade as ``mixer._resample_linear`` and
    ``perapp_capture._resample_linear``); good enough for a meeting
    track. The tail left buffered at any moment is under ~2 input
    samples (< 0.05 ms), which is dropped at the end of the stream.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._step = src_rate / dst_rate  # input samples per output sample
        self._pos = 0.0  # input position of the next output sample
        self._buf = np.zeros(0, dtype=np.float64)  # un-consumed input

    def push(self, samples: np.ndarray) -> np.ndarray:
        arr = np.asarray(samples, dtype=np.float64).ravel()
        buf = (np.concatenate([self._buf, arr]) if self._buf.size
               else arr)
        if buf.size < 2:
            self._buf = buf
            return np.zeros(0, dtype=np.int16)
        # Output positions _pos, _pos+step, ... — each needs the sample
        # after it for interpolation, so the last usable position is
        # len(buf) - 1 (exclusive of the final sample itself).
        n_out = int((len(buf) - 1 - self._pos) // self._step) + 1
        if n_out <= 0:
            self._buf = buf
            return np.zeros(0, dtype=np.int16)
        x = self._pos + self._step * np.arange(n_out)
        out = np.interp(x, np.arange(len(buf)), buf).astype(np.int16)
        next_pos = self._pos + n_out * self._step
        drop = int(next_pos // 1)  # whole input samples fully consumed
        self._buf = buf[drop:]
        self._pos = next_pos - drop
        return out


#: Sample rates tried (after the requested and native rates) when
#: negotiating a stream format — see :func:`negotiate_stream_format`.
_COMMON_RATES: tuple[int, ...] = (48000, 44100, 32000, 22050, 16000, 8000)


def negotiate_stream_format(audio, *, device_index: int | None,
                            requested_rate: int, requested_channels: int,
                            kind: str) -> tuple[int, int]:
    """Choose ``(rate, channels)`` PortAudio will accept for this device.

    The requested rate comes first, then the device's native rate
    (``defaultSampleRate``), then :data:`_COMMON_RATES`. Every
    candidate is pre-flighted with ``audio.is_format_supported`` —
    which raises ``ValueError`` for an unsupported format — so
    ``audio.open`` is never handed a format PortAudio already refused.

    Why this exists (Nov 10 2026, reproduced twice on real Windows):
    the RDP "Remote Audio [Loopback]" endpoint's mix format is
    2ch/44.1 kHz, and opening it at the app's fixed 48 kHz fails with
    ``OSError: [Errno -9998] Invalid number of channels`` / -9997
    invalid sample rate — and the frozen app then dies with native
    heap corruption (0xc0000374 in ntdll.dll). The pre-flight removes
    the trigger: a 44.1 kHz-only device opens at 44.1 kHz and the
    stream is resampled to the 48 kHz pipeline rate instead.

    Channel counts: the system kind opens with the device's native
    channel count (loopback delivers its native count even when mono
    is requested — the Nov 8 stereo fix); the mic kind tries the
    requested count first, then the native count (downmixed in
    ``_handle_chunk``).

    Raises ``RuntimeError`` when no candidate is accepted — the caller
    must surface that as a clean error instead of calling
    ``audio.open`` anyway.
    """
    if device_index is None:
        info = audio.get_default_input_device_info()
        device_index = int(info["index"])
    else:
        info = audio.get_device_info_by_index(device_index)
    native_rate = int(info.get("defaultSampleRate") or requested_rate)
    native_channels = max(
        1, int(info.get("maxInputChannels") or requested_channels))
    if kind == "system":
        channel_options = [native_channels]
    else:
        channel_options = []
        for channels in (requested_channels, native_channels):
            if channels not in channel_options:
                channel_options.append(channels)
    rates: list[int] = []
    for rate in (requested_rate, native_rate, *_COMMON_RATES):
        if rate not in rates:
            rates.append(rate)
    fmt = audio.get_format_from_width(2)  # int16
    tried: list[tuple[int, int]] = []
    for rate in rates:
        for channels in channel_options:
            tried.append((rate, channels))
            try:
                supported = audio.is_format_supported(
                    rate, input_device=device_index,
                    input_channels=channels, input_format=fmt)
            except ValueError:
                continue  # PyAudio raises ValueError for unsupported formats
            except Exception:
                logger.debug(
                    "is_format_supported(%d Hz, %d ch) raised; treating "
                    "as unsupported", rate, channels, exc_info=True)
                continue
            if supported:
                if (rate, channels) != (requested_rate, requested_channels):
                    logger.info(
                        "capture (%s): device %s (%r) does not support "
                        "%d Hz / %d ch; opening at %d Hz / %d ch and "
                        "resampling to %d Hz",
                        kind, device_index, info.get("name"),
                        requested_rate, requested_channels,
                        rate, channels, requested_rate)
                return rate, channels
    raise RuntimeError(
        f"No supported input format for {kind} device {device_index} "
        f"({info.get('name')!r}): tried {tried} — the device may be "
        "in use by another app, or its driver may not expose a "
        "standard sample rate. Try another device.")


class WASAPICapture:
    """One audio channel (mic *or* system loopback) recorded to one WAV.

    State machine: idle -> recording <-> paused -> stopped.
    ``start()`` while recording, ``pause()`` while not recording, and
    ``resume()`` while not paused all raise RuntimeError.
    """

    kind = "capture"

    def __init__(self, output_path: str | Path, *,
                 device_index: int | None = None,
                 sample_rate: int = SAMPLE_RATE,
                 channels: int = CHANNELS,
                 chunk_frames: int = CHUNK_FRAMES,
                 level_callback: LevelCallback | None = None,
                 chunk_source: ChunkSource | None = None) -> None:
        self.output_path = Path(output_path)
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_frames = chunk_frames
        self.level_callback = level_callback
        self.chunk_source = chunk_source
        self.state = CaptureState.IDLE
        self.error: str | None = None
        self.frames_written = 0
        self.started_at: float | None = None
        # Channel count the live stream actually delivers (set by
        # _run_hardware for the loopback stream — see there). None
        # means "use self.channels" (chunk sources, mic, tests).
        self._stream_channels: int | None = None
        # Native sample rate the live stream actually runs at, and the
        # resampler that converts it to self.sample_rate (both set by
        # _require_hardware / _run_hardware; None means "the stream
        # runs at self.sample_rate" — chunk sources, tests, and
        # devices whose native rate already matches).
        self._stream_rate: int | None = None
        self._resampler: _StreamingResampler | None = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: _IncrementalWavWriter | None = None
        self._lock = threading.Lock()
        # Pre-roll audio (int16 mono, pipeline rate) to write ahead of
        # the first live chunk when start() runs — set by
        # set_preroll() / DualCapture.start(preroll_samples=...).
        # Consumed exactly once, at the next start(); None (the
        # default) leaves capture byte-identical to before.
        self._preroll_samples: np.ndarray | None = None

    def set_preroll(self, samples: bytes | np.ndarray) -> None:
        """Queue pre-roll audio to lead the next recording.

        ``samples`` is int16 mono PCM at ``self.sample_rate`` (the
        drained :class:`app.audio.preroll.PreRollBuffer` output for
        this track). It is written to the WAV before any live chunk,
        so the recording starts with the buffered seconds. Calling
        this while recording raises — pre-roll only ever leads a
        recording that has not started yet. An empty array is a
        no-op. The queued audio is consumed by the next start() and
        never retained for a later one.
        """
        if self.state == CaptureState.RECORDING:
            raise RuntimeError(
                "Cannot set pre-roll while recording — pre-roll "
                "only leads a recording that has not started")
        arr = _to_int16_mono(samples, self.channels)
        self._preroll_samples = arr if arr.size else None

    # ------------------------------------------------------------ control ---
    def start(self) -> None:
        with self._lock:
            if self.state == CaptureState.RECORDING:
                raise RuntimeError(f"{self.kind} capture is already recording")
            if self.state == CaptureState.PAUSED:
                raise RuntimeError(f"{self.kind} capture is paused — call resume()")
            if self.chunk_source is None:
                self._require_hardware()
            self._stop_event.clear()
            self._paused.clear()
            self._writer = _IncrementalWavWriter(
                self.output_path, self.sample_rate, self.channels)
            # Pre-roll (if any) is written synchronously here, before
            # the capture thread starts, so it leads the WAV ahead of
            # every live chunk. Consumed once, by this start() only.
            preroll, self._preroll_samples = self._preroll_samples, None
            if preroll is not None and preroll.size:
                self._writer.write(preroll)
            self.started_at = time.time()
            self.state = CaptureState.RECORDING
            self._thread = threading.Thread(
                target=self._run, name=f"offair-{self.kind}", daemon=True)
            self._thread.start()
            logger.info("%s capture started -> %s", self.kind, self.output_path)

    def _require_hardware(self) -> None:
        """Raise a clear error when real WASAPI capture cannot run here.

        Mirrors the Linux backend's gate (``LinuxCapture``): platform,
        then the capture library, then — for the system kind only — a
        real loopback device index. ``device_index=None`` for the
        system kind would silently record the default *microphone*
        again, so it is refused instead. For the mic kind, None means
        the default input device. The live stream itself is opened in
        :meth:`_run_hardware`; this gate only checks that the attempt
        can succeed. (Before Nov 8 2026 this gate raised unconditionally
        after the import check, so no hardware capture could ever start
        on real Windows — found by the per-app fallback test in the VM.)
        """
        import sys
        if sys.platform != "win32":
            if sys.platform.startswith("linux"):
                raise RuntimeError(
                    "WASAPI capture is Windows-only "
                    f"(this machine is {sys.platform}). On Linux use the "
                    "sounddevice backend: "
                    "app.audio.linux_capture.LinuxDualCapture (or "
                    "create_dual_capture(), which picks the backend "
                    "automatically).")
            raise RuntimeError(
                "WASAPI capture is Windows-only "
                f"(this machine is {sys.platform}). No audio hardware was used; "
                "run on Windows 10/11 with pyaudiowpatch installed.")
        try:
            import pyaudiowpatch  # noqa: F401  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                "pyaudiowpatch is not installed — install it with "
                "`pip install -r requirements.txt`. "
                f"Underlying error: {exc}") from exc
        if self.kind == "system" and self.device_index is None:
            raise RuntimeError(
                "No WASAPI loopback device selected — system audio cannot "
                "be recorded. Pass the loopback device index (from "
                "app.audio.devices.list_loopback_devices()); "
                "device_index=None would silently record the default "
                "microphone again.")
        # Negotiate the stream format NOW, synchronously on the calling
        # thread — before the capture thread exists. Two reasons:
        # * A device whose native rate is 44.1 kHz (e.g. the RDP
        #   "Remote Audio" endpoint) rejects a 48 kHz open, and on the
        #   frozen app that failed open is adjacent to a native heap
        #   corruption crash (Nov 10 2026). Pre-flighting here means
        #   audio.open is never handed a refused format, and a device
        #   with no usable format fails here — as a clean Python error
        #   the UI can show — instead of dying inside the thread.
        # * The negotiated rate/channels are needed before the writer
        #   starts, so the resampler (if any) is in place from the
        #   first chunk.
        # The construction + negotiation + terminate are serialised
        # through PYAUDIO_INIT_LOCK: DualCapture.start() calls
        # mic.start() (which spawns the mic thread) and then
        # system.start() on this thread, so without the lock this
        # PyAudio() would race the mic thread's own PyAudio() in
        # _run_hardware — concurrent Pa_Initialize crashes the
        # process (0xc0000005, Nov 10 2026).
        import pyaudiowpatch as pyaudio  # noqa: F401  # type: ignore[import-not-found]
        with PYAUDIO_INIT_LOCK:
            audio = pyaudio.PyAudio()
            try:
                rate, channels = negotiate_stream_format(
                    audio, device_index=self.device_index,
                    requested_rate=self.sample_rate,
                    requested_channels=self.channels, kind=self.kind)
            finally:
                audio.terminate()
        self._stream_rate = rate
        self._stream_channels = channels
        self._resampler = (
            _StreamingResampler(rate, self.sample_rate)
            if rate != self.sample_rate else None)

    def pause(self) -> None:
        with self._lock:
            if self.state != CaptureState.RECORDING:
                raise RuntimeError(
                    f"Cannot pause {self.kind} capture while {self.state.value}")
            self._paused.set()
            self.state = CaptureState.PAUSED
            logger.info("%s capture paused (audio while paused is discarded)",
                        self.kind)

    def resume(self) -> None:
        with self._lock:
            if self.state != CaptureState.PAUSED:
                raise RuntimeError(
                    f"Cannot resume {self.kind} capture while {self.state.value}")
            self._paused.clear()
            self.state = CaptureState.RECORDING
            logger.info("%s capture resumed", self.kind)

    def stop(self) -> Path:
        with self._lock:
            if self.state in (CaptureState.IDLE, CaptureState.STOPPED):
                # Idempotent stop from idle: still produce an (empty) valid WAV
                # so a session folder is never left without its tracks.
                if self.state == CaptureState.IDLE and not self.output_path.exists():
                    writer = _IncrementalWavWriter(
                        self.output_path, self.sample_rate, self.channels)
                    writer.close()
                self.state = CaptureState.STOPPED
                return self.output_path
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        with self._lock:
            self._finalise()
            self.state = CaptureState.STOPPED
            logger.info("%s capture stopped: %d frames (%.1fs) -> %s",
                        self.kind, self.frames_written,
                        self.duration_seconds, self.output_path)
            return self.output_path

    # ------------------------------------------------------------ internals -
    def _finalise(self) -> None:
        if self._writer is not None:
            self.frames_written = self._writer.frames_written
            self._writer.close()
            self._writer = None

    def _handle_chunk(self, chunk: bytes | np.ndarray) -> None:
        """Write one chunk unless paused (paused audio is discarded)."""
        if self._paused.is_set():
            return  # documented pause policy: discard, do not buffer
        # _stream_channels is the live loopback stream's real channel
        # count (see _run_hardware); otherwise the configured count
        # (chunk sources, mic, tests).
        samples = _to_int16_mono(
            chunk, self._stream_channels or self.channels)
        # The stream may run at the device's native rate (e.g. 44.1 kHz
        # on an RDP endpoint) while the pipeline — and the WAV writer —
        # is 48 kHz: resample here so every written frame is 48 kHz.
        if self._resampler is not None:
            samples = self._resampler.push(samples)
        if self.level_callback is not None:
            try:
                self.level_callback(rms_level(samples))
            except Exception:
                logger.exception("level_callback failed")
        if self._writer is not None:
            self._writer.write(samples)

    def _run(self) -> None:
        try:
            if self.chunk_source is not None:
                source = (self.chunk_source()
                          if callable(self.chunk_source) else self.chunk_source)
                for chunk in source:
                    if self._stop_event.is_set():
                        break
                    self._handle_chunk(chunk)
                # Source exhausted (e.g. a finite test feed): wait for stop()
                # so the file stays open and valid until the user stops.
                while not self._stop_event.wait(timeout=0.05):
                    pass
            else:
                self._run_hardware()
        except Exception as exc:
            self.error = str(exc)
            self.state = CaptureState.ERROR
            logger.exception("%s capture failed", self.kind)
        finally:
            # If stop() already finalised, _writer is None — safe to call.
            if self.state != CaptureState.STOPPED:
                with self._lock:
                    self._finalise()

    def _run_hardware(self) -> None:
        """Live WASAPI stream. Windows + pyaudiowpatch only.

        Opened lazily here (not at import) so the module imports cleanly
        on Linux/macOS. On Windows this opens the input stream for a mic
        or the loopback stream for system audio and feeds chunks to
        ``_handle_chunk`` until ``stop()`` is called.
        """
        import pyaudiowpatch as pyaudio  # type: ignore[import-not-found]

        # Construction + open are serialised through PYAUDIO_INIT_LOCK:
        # this thread's PyAudio() must never overlap another capture's
        # PyAudio() (or terminate()) — concurrent Pa_Initialize crashes
        # the process (0xc0000005, Nov 10 2026). The lock is released
        # once the stream is open; the read loop runs without it.
        audio = None
        stream = None
        try:
            with PYAUDIO_INIT_LOCK:
                audio = pyaudio.PyAudio()
                try:
                    if self._stream_rate is None:
                        # Defensive only: _require_hardware negotiates
                        # synchronously in start(), so this runs solely
                        # when the capture was driven without it (tests,
                        # direct _run_hardware calls). The pre-flight
                        # is what keeps a refused format out of
                        # audio.open — see there.
                        rate, channels = negotiate_stream_format(
                            audio, device_index=self.device_index,
                            requested_rate=self.sample_rate,
                            requested_channels=self.channels, kind=self.kind)
                        self._stream_rate = rate
                        self._stream_channels = channels
                        self._resampler = (
                            _StreamingResampler(rate, self.sample_rate)
                            if rate != self.sample_rate else None)
                    # The stream opens at the device's native rate (see the
                    # Nov 8 channel fix and the Nov 10 rate fix above); the
                    # WAV writer stays at self.sample_rate and _handle_chunk
                    # resamples into it, so mic.wav / system.wav /
                    # combined.wav are always 48 kHz mono int16.
                    logger.info(
                        "%s capture: opening stream (device=%s, %d Hz, %d ch; "
                        "writing %d Hz mono)", self.kind, self.device_index,
                        self._stream_rate, self._stream_channels or self.channels,
                        self.sample_rate)
                    stream = audio.open(
                        format=pyaudio.paInt16,
                        channels=self._stream_channels or self.channels,
                        rate=self._stream_rate,
                        input=True,
                        input_device_index=self.device_index,
                        frames_per_buffer=self.chunk_frames)
                except Exception:
                    # Open/negotiate failed: terminate this instance
                    # before releasing the lock, then re-raise so _run()
                    # records the clean ERROR state.
                    try:
                        audio.terminate()
                    except Exception:
                        logger.exception(
                            "%s capture: PyAudio terminate failed",
                            self.kind)
                    audio = None
                    raise
            try:
                while not self._stop_event.is_set():
                    try:
                        data = stream.read(self.chunk_frames,
                                           exception_on_overflow=False)
                    except Exception:
                        if self._stop_event.is_set():
                            break
                        raise
                    self._handle_chunk(data)
            finally:
                # Guard each teardown step: if open succeeded but the
                # stream is half-torn-down, a failing stop/close must
                # not mask the original error — or stop terminate()
                # from running.
                try:
                    stream.stop_stream()
                except Exception:
                    logger.exception("%s capture: stop_stream failed",
                                     self.kind)
                try:
                    stream.close()
                except Exception:
                    logger.exception("%s capture: stream close failed",
                                     self.kind)
        finally:
            # Terminate even when open failed: PyAudio holds WASAPI /
            # COM references that must be released, and a failed open
            # leaves no stream behind to double-close (stream is None
            # until open returns). If the native crash is inside
            # PortAudio's failed-open path itself, only avoiding the
            # failed open — the pre-flight above — removes it; nothing
            # Python-side can guard a heap corruption that already
            # happened. See the Nov 10 2026 note in
            # negotiate_stream_format. Terminate is serialised too:
            # Pa_Terminate racing another thread's Pa_Initialize is
            # the same class of crash.
            if audio is not None:
                with PYAUDIO_INIT_LOCK:
                    try:
                        audio.terminate()
                    except Exception:
                        logger.exception(
                            "%s capture: PyAudio terminate failed",
                            self.kind)

    @property
    def duration_seconds(self) -> float:
        frames = self.frames_written
        if self._writer is not None:
            frames = self._writer.frames_written
        return frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def stream_sample_rate(self) -> int:
        """Native rate the live stream runs at.

        Usually ``self.sample_rate`` (48 kHz); a device whose native
        rate differs (e.g. a 44.1 kHz RDP endpoint) streams at its
        native rate and is resampled to the pipeline rate — the WAV
        is always ``self.sample_rate``. Recorded in metadata.json so
        a session says what the hardware actually did.
        """
        return self._stream_rate if self._stream_rate is not None \
            else self.sample_rate

    @property
    def stream_channels(self) -> int:
        """Channel count the live stream delivers."""
        return self._stream_channels if self._stream_channels is not None \
            else self.channels

    @property
    def is_recording(self) -> bool:
        return self.state == CaptureState.RECORDING


class MicCapture(WASAPICapture):
    kind = "mic"


class SystemCapture(WASAPICapture):
    kind = "system"


class PerAppCapture(SystemCapture):
    """System-loopback capture *targeting* one process (e.g. Teams).

    Honest limitation (see module docstring): pyaudiowpatch only exposes
    system-wide loopback, so this class records the system loopback and
    reports ``mode == "system-loopback-fallback"`` with a warning, rather
    than pretending the Teams stream is isolated. ``process_info`` is the
    psutil record for the target process when it is running, else None.
    """

    kind = "system"

    def __init__(self, output_path: str | Path, *,
                 process_name: str = "ms-teams.exe",
                 process_names: list[str] | None = None, **kwargs) -> None:
        super().__init__(output_path, **kwargs)
        self.process_name = process_name
        self.process_info = find_teams_process(
            process_names or [process_name])
        self.mode = "system-loopback-fallback"
        if self.process_info is None:
            self.warning = (
                f"Process '{process_name}' is not running; recording "
                "system-wide loopback. Start Teams and re-select it to "
                "confirm the target app is active.")
        else:
            self.warning = (
                "Per-app isolation is NOT available via pyaudiowpatch: "
                "this records system-wide WASAPI loopback, which includes "
                f"audio from '{process_name}' (pid "
                f"{self.process_info.get('pid')}) and any other app playing "
                "sound. Mute other apps for a clean remote track.")
        logger.warning("PerAppCapture: %s", self.warning)


# Sentinel distinguishing "system_cls not passed" (default PerAppCapture)
# from "system_cls=None passed explicitly" (mic-only mode) in DualCapture.
_DEFAULT_SYSTEM_CLS: object = object()


class DualCapture:
    """Mic + system capture together, plus the combined mix on stop.

    Owns the two capture objects for one session folder and
    guarantees the mic/system tracks stay in lock-step for pause/resume.

    ``mic_cls`` / ``system_cls`` select the backend: the defaults are
    the WASAPI classes (:class:`MicCapture` /
    :class:`PerAppCapture`). The Linux backend passes its own classes
    (see :class:`app.audio.linux_capture.LinuxDualCapture`).
    ``system_cls=None`` means mic-only mode: no system capture object
    exists, ``stop()`` writes an empty ``system.wav`` so the session
    folder keeps its shape, and ``mode`` / ``warning`` (explicit
    arguments in that case) record why.

    ``capture_system_mode`` (``"auto"`` / ``"per-app"`` / ``"loopback"``,
    default ``"auto"``) decides the default ``system_cls`` when none is
    passed explicitly: per-app capture
    (:class:`app.audio.perapp_capture.ProcessLoopbackCapture`) when
    available, else :class:`PerAppCapture`. Exception: in ``"auto"``
    mode a Teams desktop target (``ms-teams.exe``) is recorded
    system-wide from the start — process loopback returns silence
    for it even mid-call — with :attr:`warning` saying why. If the
    per-app capture fails at ``start()`` in ``"auto"`` mode, ``start()`` falls back to
    the system-wide loopback track (the mic keeps running) and
    ``mode`` / ``warning`` are updated to the fallback — the recording
    is never lost to an activation failure. In ``"per-app"`` mode the
    failure propagates instead.
    """

    def __init__(self, session_dir: str | Path, *,
                 mic_device_index: int | None = None,
                 loopback_device_index: int | None = None,
                 process_name: str = "ms-teams.exe",
                 sample_rate: int = SAMPLE_RATE,
                 mic_level_callback: LevelCallback | None = None,
                 system_level_callback: LevelCallback | None = None,
                 chunk_source_factory=None,
                 mic_cls=None,
                 system_cls=_DEFAULT_SYSTEM_CLS,
                 capture_system_mode: str | None = None,
                 mode: str | None = None,
                 warning: str | None = None) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.process_name = process_name
        # chunk_source_factory(kind) -> chunk source or None; test seam.
        def _source(kind: str):
            if chunk_source_factory is None:
                return None
            result = chunk_source_factory(kind)
            if result is None or callable(result):
                return result
            # Factory returned an iterable (e.g. a generator) directly.
            return lambda _result=result: _result
        mic_cls = mic_cls if mic_cls is not None else MicCapture
        # Default system capture follows capture_system_mode ("auto" =
        # per-app when available, else system-wide loopback). Lazy
        # import: perapp_capture imports this module at module level.
        self._capture_system_mode = (
            capture_system_mode or CAPTURE_SYSTEM_MODE_DEFAULT)
        # Teams-desktop guard: in "auto" mode the selector records
        # the Teams desktop client system-wide (process loopback
        # returns silence for it) — remember that routing happened
        # so the warning below can say why (only when per-app was
        # otherwise available: an unavailable per-app path already
        # explains a loopback recording on its own).
        teams_routed = False
        if system_cls is _DEFAULT_SYSTEM_CLS:
            from app.audio.perapp_capture import (
                TEAMS_DESKTOP_EXES,
                per_app_capture_available,
                select_system_capture_class,
            )
            system_cls = select_system_capture_class(
                self._capture_system_mode, process_name)
            if (self._capture_system_mode == "auto"
                    and system_cls is PerAppCapture
                    and (process_name or "").lower() in TEAMS_DESKTOP_EXES
                    and per_app_capture_available()[0]):
                teams_routed = True
        # Construction kwargs for the system capture, kept so start()
        # can rebuild it as the loopback fallback if per-app fails.
        self._system_kwargs = {
            "process_name": process_name,
            "device_index": loopback_device_index,
            "sample_rate": sample_rate,
            "level_callback": system_level_callback,
            "chunk_source": _source("system"),
        }
        self.mic = mic_cls(
            self.session_dir / "mic.wav",
            device_index=mic_device_index, sample_rate=sample_rate,
            level_callback=mic_level_callback, chunk_source=_source("mic"))
        if system_cls is None:
            self.system = None
            self.mode = mode if mode is not None else "mic-only"
            self.warning = warning
        else:
            self.system = system_cls(
                self.session_dir / "system.wav", **self._system_kwargs)
            self.mode = (mode if mode is not None
                         else getattr(self.system, "mode", "system-loopback-fallback"))
            self.warning = (warning if warning is not None
                            else getattr(self.system, "warning", None))
        if teams_routed and warning is None:
            # The recording is system-wide loopback, but the user
            # asked for one app — the metadata must say this was
            # deliberate (Teams desktop + process loopback = silence),
            # not the generic loopback warning. An explicit caller
            # warning still wins.
            self.warning = (
                "Teams desktop app: per-app capture returns silence "
                "for it on Windows, so this recording is system-wide "
                "loopback instead (everything you hear).")
        self._started = False

    def start(self, preroll_samples: dict | None = None) -> None:
        """Start both tracks.

        ``preroll_samples`` (optional) is ``{"mic": samples,
        "system": samples}`` — int16 mono arrays at the pipeline
        rate, as drained from a
        :class:`app.audio.preroll.PreRollBuffer` — written ahead of
        the first live chunk on each track. Omit it (or pass None)
        and capture is byte-identical to before pre-roll existed.
        A "system" entry is dropped with a warning in mic-only mode
        (there is no system track to lead). The pre-roll is handed
        to each track via :meth:`WASAPICapture.set_preroll` before
        its start(), so the per-app fallback rebuild in "auto" mode
        receives it too.
        """
        if preroll_samples:
            for track, samples in preroll_samples.items():
                if track not in ("mic", "system"):
                    raise ValueError(
                        f"unknown pre-roll track {track!r}; "
                        "expected 'mic' and/or 'system'")
                capture = self.mic if track == "mic" else self.system
                if capture is None:
                    logger.warning(
                        "Pre-roll for %r dropped: no %s track "
                        "(mic-only mode)", track, track)
                    continue
                capture.set_preroll(samples)
        self.mic.start()
        try:
            if self.system is not None:
                self.system.start()
        except Exception as exc:
            # Per-app capture failed at start (activation, missing
            # process, ...). In "auto" mode fall back to the
            # system-wide loopback track — the mic is already running,
            # so the fallback track just starts a fraction of a second
            # late. Any other failure (or "per-app" mode) stops the
            # mic and propagates, as before.
            from app.audio.perapp_capture import (
                PerAppCaptureError, ProcessLoopbackCapture)
            if (self._capture_system_mode == "auto"
                    and isinstance(self.system, ProcessLoopbackCapture)
                    and isinstance(exc, PerAppCaptureError)):
                logger.warning(
                    "Per-app capture failed (%s); falling back to "
                    "system-wide loopback", exc)
                self.system = PerAppCapture(
                    self.session_dir / "system.wav", **self._system_kwargs)
                self.mode = self.system.mode
                self.warning = self.system.warning
                # The fallback object is new, so the pre-roll queued
                # on the failed per-app object must be queued again
                # (its writer — same path — is recreated from empty).
                if preroll_samples and "system" in preroll_samples:
                    self.system.set_preroll(preroll_samples["system"])
                try:
                    self.system.start()
                except Exception:
                    self.mic.stop()
                    raise
                self._started = True
                return
            self.mic.stop()
            raise
        self._started = True

    def pause(self) -> None:
        self.mic.pause()
        if self.system is not None:
            self.system.pause()

    def resume(self) -> None:
        self.mic.resume()
        if self.system is not None:
            self.system.resume()

    def stop(self) -> dict[str, Path]:
        """Stop both tracks and mix them to ``combined.wav``.

        Returns ``{"mic": ..., "system": ..., "combined": ...}``.
        In mic-only mode ``system.wav`` is an empty valid WAV (the
        system track was never recorded) and ``combined.wav`` is the
        mix of the mic track with silence. If mixing fails (e.g. a
        track is missing) the error is logged and the two raw tracks
        are still returned — a failed mix must never lose a recording.
        """
        from app.audio.mixer import mix_files

        mic_path = self.mic.stop()
        if self.system is not None:
            system_path = self.system.stop()
        else:
            system_path = self.session_dir / "system.wav"
            if not system_path.exists():
                writer = _IncrementalWavWriter(
                    system_path, self.mic.sample_rate, self.mic.channels)
                writer.close()
        combined_path = self.session_dir / "combined.wav"
        result: dict[str, Path] = {"mic": mic_path, "system": system_path}
        try:
            mix_files(mic_path, system_path, combined_path)
            result["combined"] = combined_path
        except Exception:
            logger.exception("Failed to mix combined.wav — raw tracks kept")
        self._started = False
        return result

    @property
    def state(self) -> CaptureState:
        if self.system is not None:
            if self.system.state == CaptureState.PAUSED:
                return CaptureState.PAUSED
            return self.system.state
        return self.mic.state

    @property
    def duration_seconds(self) -> float:
        if self.system is not None:
            return max(self.mic.duration_seconds,
                       self.system.duration_seconds)
        return self.mic.duration_seconds


def create_dual_capture(session_dir: str | Path, **kwargs) -> DualCapture:
    """Build the platform-appropriate :class:`DualCapture`.

    Windows → WASAPI :class:`DualCapture`; Linux with sounddevice /
    PortAudio → :class:`app.audio.linux_capture.LinuxDualCapture`
    (mic + monitor, or mic-only when no monitor source exists).
    Raises ``RuntimeError`` with install guidance when no capture
    backend is available on this machine (e.g. Linux without
    sounddevice) — the app itself never crashes on import for that.
    """
    from app.platform_info import describe_capture_state, get_capture_backend

    backend = get_capture_backend()
    if backend == "wasapi":
        return DualCapture(session_dir, **kwargs)
    if backend == "linux-sounddevice":
        from app.audio.linux_capture import LinuxDualCapture
        return LinuxDualCapture(session_dir, **kwargs)
    info = describe_capture_state()
    raise RuntimeError(
        f"Audio capture is unavailable on this machine: {info.status}. "
        + (info.install_hint or info.detail or ""))
