"""Pre-roll feeder: the always-on capture loop that fills PreRollBuffer.

Opt-in, default OFF: the UI layer creates a :class:`PreRollFeeder`
only when ``preroll_enabled`` is True (see ``app.config``) and drives
it with explicit :meth:`PreRollFeeder.start` / :meth:`PreRollFeeder.stop`
calls — nothing in this module reads settings, touches the UI, or
starts itself. While running, one background thread per track (mic /
system) opens that track's input stream in shared mode and feeds
chunks into the :class:`~app.audio.preroll.PreRollBuffer` the UI
layer owns. Feeding runs only while no recording is in progress:
the UI layer stops the feeder *before* starting capture (see the
lifecycle contract below), and starts it again after the recording
stops, if pre-roll is still enabled.

**Memory only, never disk.** This module performs no file I/O at
all: chunks flow from the audio stream straight into the in-memory
buffer and are discarded when they age out, when the buffer is
drained, or when the process ends. (The word ``open`` below is
always an audio-stream open — ``PyAudio.open`` / a sounddevice
``InputStream`` — never a file.)

**Device conflict / stop-before-start (mandatory).** Whether
WASAPI shared mode would allow the feeder's loopback stream and a
recording's loopback stream to be open at the same time was NOT
verified on real Windows hardware from this VM, and no concurrent
open is relied on: :meth:`PreRollFeeder.stop` is synchronous — it
returns only after every feeder thread has exited and its stream
has been closed — and capture must start only after it returns.
The codebase already opens mic + system concurrently inside one
recording (:class:`app.audio.capture.DualCapture`), but feeder-vs-
recording concurrency is a different pair of opens and is
unproven, so the stop-before-start contract stands regardless.

**Failure modes — silent degradation, never a failed recording.**
A device that cannot be opened, a missing device, a stream that
dies mid-feed, or an unsupported platform is logged (warning) and
that track simply stays empty; the other track keeps feeding, and
:meth:`start` never raises. A feeder failure must never fail a
recording start — the UI layer's ``feeder.stop()`` is safe to call
whether or not the feeder ever ran, and capture works exactly as
before when the buffer is empty.

**Backends — no new device layer.** Device indices come from the
same enumeration the app already uses
(:func:`app.audio.devices.default_microphone` /
:func:`app.audio.devices.default_loopback`, overridable with the
explicit indices the UI already passes to ``DualCapture``), and
the streams use the same backends as recording: WASAPI via
``pyaudiowpatch`` on Windows (format negotiation + resampling
reused from :mod:`app.audio.capture`), sounddevice / PortAudio on
Linux (the same ``InputStream`` shape as
:class:`app.audio.linux_capture.LinuxSounddeviceCapture`). The
system track is system-wide loopback / a monitor source — the
same honest limitation as the loopback fallback track — not
per-app isolated audio. On a platform with neither backend the
feeder starts nothing and logs why.

**Threading.** Feed runs on the feeder threads; drain runs on the
UI thread. :class:`~app.audio.preroll.PreRollBuffer` is already
thread-safe and stays that way — this module only calls
``feed``. Start/stop are idempotent and race-free: they are
serialised through one lock, a second ``start()`` while running
opens nothing new, ``stop()`` without ``start()`` is a no-op, and
``stop()`` joins the feeder threads (bounded) so a stream cannot
still be open when ``stop()`` returns.

Lifecycle contract for the UI layer (U4) — call order matters::

    buffer = PreRollBuffer(seconds=config.preroll_seconds,
                           sample_rate=SAMPLE_RATE)
    feeder = PreRollFeeder(
        buffer, enabled=config.preroll_enabled,
        sample_rate=SAMPLE_RATE,
        mic_device_index=<mic combo currentData() or None>,
        loopback_device_index=<loopback combo currentData() or None>)
    # When pre-roll is on (app start / setting turned on):
    feeder.start()  # no-op returning False when enabled=False or
                    # no usable device; never raises.
    # Manual Record handler, in this exact order:
    feeder.stop()                       # synchronous: streams closed
    preroll = buffer.drain_all()        # {} when nothing buffered
    capture.start(preroll_samples=preroll)
    # Auto-detect meeting-started handler, in this exact order:
    feeder.stop()                       # synchronous: streams closed
    preroll = watcher.take_preroll()    # drained by the watcher
                                        # at the start gate
    buffer.clear()  # discard anything fed between the gate's drain
                    # and this stop(); it is meeting audio, not
                    # pre-roll for the next session
    capture.start(preroll_samples=preroll)
    # After DualCapture.stop() returns (and if a start failed):
    feeder.start()  # resume feeding only if still enabled;
                    # leave stopped when pre-roll was turned off.
    # Pre-roll turned off / app shutdown:
    feeder.stop()
    buffer.clear()

If the selected devices change while the feeder is running, stop
it and start it again with the new indices (the indices are read
at ``start()`` time). Do not start the feeder while a recording
is in progress, and do not restart it until the recording has
stopped.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Callable, Iterable

import numpy as np

from app.audio.devices import default_loopback, default_microphone
from app.audio.preroll import PREROLL_TRACKS, PreRollBuffer
from app.config import CHUNK_FRAMES, SAMPLE_RATE

logger = logging.getLogger("offair.preroll_feeder")

#: How long stop() waits for each feeder thread to exit and close
#: its stream. A live read is one chunk long (~21 ms at 1024 frames
#: / 48 kHz), so this is generous; a thread still alive after it is
#: logged and left (daemon) rather than hanging the UI thread that
#: is about to start a recording.
_STOP_JOIN_TIMEOUT_SECONDS = 5.0

#: A stream factory: given a track and the stop event, return an
#: iterable of int16 mono PCM chunks (bytes or numpy arrays) at
#: ``sample_rate``. The iterable is consumed on the feeder thread
#: for that track; its ``close()`` (if any) is called on the same
#: thread when feeding stops or the iterable is exhausted, so a
#: factory may return a generator whose ``finally`` closes the
#: hardware stream. Raising from the factory (or from iteration)
#: degrades that track silently — it is logged, never propagated.
#: The default factory opens the platform backend; tests inject a
#: fake with this same signature.
StreamFactory = Callable[..., Iterable[bytes | np.ndarray]]


class PreRollFeeder:
    """Feed a :class:`PreRollBuffer` from live audio streams.

    ``enabled`` is the ``preroll_enabled`` setting as read by the
    UI layer at construction time: when False, :meth:`start` is a
    no-op that returns False and no device is ever opened. Changing
    the setting means constructing a new feeder (or stopping this
    one) — this class never reads settings itself.
    ``mic_device_index`` / ``loopback_device_index`` are the
    explicit device indices (or None for the defaults, resolved at
    :meth:`start` time from the shared enumeration). ``sample_rate``
    is the pipeline rate the buffer holds (48 kHz); streams running
    at a device-native rate are resampled to it, as in recording.
    ``stream_factory`` is the test seam (see :data:`StreamFactory`);
    omit it to use the platform backend.
    """

    def __init__(self, buffer: PreRollBuffer, *,
                 enabled: bool = True,
                 sample_rate: int = SAMPLE_RATE,
                 chunk_frames: int = CHUNK_FRAMES,
                 mic_device_index: int | None = None,
                 loopback_device_index: int | None = None,
                 stream_factory: StreamFactory | None = None) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        self._buffer = buffer
        self.enabled = bool(enabled)
        self.sample_rate = int(sample_rate)
        self.chunk_frames = int(chunk_frames)
        self.mic_device_index = mic_device_index
        self.loopback_device_index = loopback_device_index
        self._stream_factory = stream_factory or _default_stream_factory
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._stop_event: threading.Event | None = None

    # ------------------------------------------------------------ state ---
    @property
    def is_running(self) -> bool:
        """True while at least one feeder thread is alive."""
        with self._lock:
            self._threads = {track: thread for track, thread in
                             self._threads.items() if thread.is_alive()}
            return bool(self._threads)

    @property
    def active_tracks(self) -> tuple[str, ...]:
        """Tracks with a live feeder thread, in buffer order."""
        with self._lock:
            alive = {track for track, thread in self._threads.items()
                     if thread.is_alive()}
        return tuple(track for track in PREROLL_TRACKS if track in alive)

    # ---------------------------------------------------------- control ---
    def start(self) -> bool:
        """Start feeding the buffer (idempotent, never raises).

        Returns True when at least one track thread started (or was
        already running); False when disabled or when no usable
        device exists for any track. A stream that fails to open
        after the thread starts is async degradation — the thread
        ends, ``is_running`` / ``active_tracks`` reflect it, and
        pre-roll for that track is simply empty; recording is
        unaffected either way. A second call while running opens
        nothing new.
        """
        with self._lock:
            if not self.enabled:
                logger.info("Pre-roll feeder not started: pre-roll "
                            "is disabled")
                return False
            self._threads = {track: thread for track, thread in
                             self._threads.items() if thread.is_alive()}
            if self._threads:
                return True
            stop_event = threading.Event()
            self._stop_event = stop_event
            tracks = self._resolve_tracks()
            if not tracks:
                self._stop_event = None
                return False
            for track, device_index in tracks:
                thread = threading.Thread(
                    target=self._run_track,
                    args=(track, device_index, stop_event),
                    name=f"offair-preroll-{track}", daemon=True)
                self._threads[track] = thread
                thread.start()
            return True

    def stop(self) -> None:
        """Stop feeding and release every stream, synchronously.

        Returns only after the feeder threads have exited (bounded
        join) and their streams have been closed, so capture may
        start immediately afterwards without a device conflict.
        Idempotent: safe to call without ``start()``, twice in a
        row, or after the threads already ended on their own. Never
        raises. Does not clear the buffer — the UI layer drains it
        after stopping (see the module docstring contract).
        """
        with self._lock:
            threads = list(self._threads.values())
            stop_event = self._stop_event
            self._threads = {}
            self._stop_event = None
            if stop_event is not None:
                stop_event.set()
            for thread in threads:
                thread.join(timeout=_STOP_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    logger.warning(
                        "Pre-roll feeder thread %s did not exit "
                        "within %.1f s of stop(); leaving it (daemon) "
                        "rather than blocking the recording start",
                        thread.name, _STOP_JOIN_TIMEOUT_SECONDS)

    # ---------------------------------------------------------- internals -
    def _resolve_tracks(self) -> list[tuple[str, int | None]]:
        """(track, device_index) pairs that have a device to open.

        Indices are resolved from the shared enumeration at
        ``start()`` time: an explicit index wins, otherwise the
        default device for that kind, otherwise the track is
        skipped with a logged reason. Never raises — enumeration
        itself is guarded, because a failed lookup must degrade,
        not break ``start()``.
        """
        resolved: list[tuple[str, int | None]] = []
        for track, explicit_index, default_fn in (
                ("mic", self.mic_device_index, default_microphone),
                ("system", self.loopback_device_index, default_loopback)):
            if explicit_index is not None:
                resolved.append((track, int(explicit_index)))
                continue
            try:
                default = default_fn()
            except Exception:  # noqa: BLE001 — enumeration must degrade
                logger.warning("Pre-roll feeder (%s): device "
                               "enumeration failed; track skipped",
                               track, exc_info=True)
                continue
            if default is None:
                logger.warning(
                    "Pre-roll feeder (%s): no device available; "
                    "that track will have no pre-roll", track)
                continue
            resolved.append((track, int(default.index)))
        return resolved

    def _run_track(self, track: str, device_index: int | None,
                   stop_event: threading.Event) -> None:
        """Feed one track until stopped, the stream ends, or it fails.

        Runs on the feeder thread for ``track``. Every failure is
        logged and ends this track only — the sibling track is
        untouched, and nothing propagates to the UI thread.
        """
        try:
            stream = self._stream_factory(
                track, device_index=device_index,
                chunk_frames=self.chunk_frames,
                sample_rate=self.sample_rate, stop_event=stop_event)
        except Exception:  # noqa: BLE001 — open failure degrades
            logger.warning("Pre-roll feeder (%s): could not open "
                           "device %s; pre-roll for that track will "
                           "be empty", track, device_index,
                           exc_info=True)
            return
        try:
            for chunk in stream:
                if stop_event.is_set():
                    break
                try:
                    self._buffer.feed(track, chunk)
                except Exception:  # noqa: BLE001 — a bad chunk ends it
                    logger.warning("Pre-roll feeder (%s): feeding "
                                   "the buffer failed; stopping that "
                                   "track", track, exc_info=True)
                    break
        except Exception:  # noqa: BLE001 — stream error degrades
            logger.warning("Pre-roll feeder (%s): stream error; "
                           "pre-roll for that track ends here",
                           track, exc_info=True)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001 — close must not raise
                    logger.warning("Pre-roll feeder (%s): closing "
                                   "the stream failed", track,
                                   exc_info=True)


# --------------------------------------------------------------------------
# Default (platform) stream factories — same backends as recording.
# Each returns a generator: iterating it reads the hardware stream
# until the stop event is set, and its finally block closes the
# stream (and terminates its PortAudio instance) on the feeder
# thread that consumed it. No file I/O anywhere in this module.
# --------------------------------------------------------------------------
def _default_stream_factory(track: str, *, device_index: int | None,
                            chunk_frames: int, sample_rate: int,
                            stop_event: threading.Event,
                            ) -> Iterable[bytes | np.ndarray]:
    """Open ``track``'s platform stream; raise when unavailable.

    Dispatch happens here, at ``start()`` time, so importing this
    module is safe on every platform and a missing backend is a
    per-track degradation (logged by the caller), not an import
    error. The caller iterates the returned generator on the
    feeder thread.
    """
    if sys.platform == "win32":
        return _windows_stream(track, device_index=device_index,
                               chunk_frames=chunk_frames,
                               sample_rate=sample_rate,
                               stop_event=stop_event)
    if sys.platform.startswith("linux"):
        return _linux_stream(track, device_index=device_index,
                             chunk_frames=chunk_frames,
                             sample_rate=sample_rate,
                             stop_event=stop_event)
    raise RuntimeError(
        f"Pre-roll capture is not supported on {sys.platform}: "
        "no WASAPI (Windows) or sounddevice (Linux) backend")


def _windows_stream(track: str, *, device_index: int | None,
                    chunk_frames: int, sample_rate: int,
                    stop_event: threading.Event,
                    ) -> Iterable[np.ndarray]:
    """WASAPI mic / loopback stream via pyaudiowpatch (Windows only).

    Mirrors :meth:`app.audio.capture.WASAPICapture._run_hardware`:
    the format is pre-flighted with
    :func:`app.audio.capture.negotiate_stream_format` under
    ``PYAUDIO_INIT_LOCK`` (never hand ``open`` a refused format),
    the stream opens at the device-native rate and is resampled to
    the pipeline rate, and construction / open / terminate are
    serialised through the same lock. A ``system`` track with no
    device index is refused — opening it would silently record the
    default microphone (the same guard as the recording path).
    """
    import pyaudiowpatch as pyaudio  # type: ignore[import-not-found]

    from app.audio.capture import (
        _StreamingResampler,
        _to_int16_mono,
        negotiate_stream_format,
    )
    from app.audio.devices import PYAUDIO_INIT_LOCK

    if track == "system" and device_index is None:
        raise RuntimeError(
            "No WASAPI loopback device selected — system pre-roll "
            "cannot be captured. Pass the loopback device index "
            "(from app.audio.devices.list_loopback_devices()).")
    audio = None
    stream = None
    try:
        with PYAUDIO_INIT_LOCK:
            audio = pyaudio.PyAudio()
            try:
                rate, channels = negotiate_stream_format(
                    audio, device_index=device_index,
                    requested_rate=sample_rate, requested_channels=1,
                    kind=track)
                resampler = (
                    _StreamingResampler(rate, sample_rate)
                    if rate != sample_rate else None)
                logger.info(
                    "Pre-roll feeder (%s): opening stream (device=%s, "
                    "%d Hz, %d ch; feeding %d Hz mono)",
                    track, device_index, rate, channels, sample_rate)
                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=chunk_frames)
            except Exception:
                try:
                    audio.terminate()
                except Exception:
                    logger.exception(
                        "Pre-roll feeder (%s): PyAudio terminate "
                        "failed", track)
                audio = None
                raise
        while not stop_event.is_set():
            data = stream.read(chunk_frames, exception_on_overflow=False)
            samples = _to_int16_mono(data, channels)
            if resampler is not None:
                samples = resampler.push(samples)
            yield samples
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                logger.exception(
                    "Pre-roll feeder (%s): stop_stream failed", track)
            try:
                stream.close()
            except Exception:
                logger.exception(
                    "Pre-roll feeder (%s): stream close failed", track)
        if audio is not None:
            with PYAUDIO_INIT_LOCK:
                try:
                    audio.terminate()
                except Exception:
                    logger.exception(
                        "Pre-roll feeder (%s): PyAudio terminate "
                        "failed", track)


def _linux_stream(track: str, *, device_index: int | None,
                  chunk_frames: int, sample_rate: int,
                  stop_event: threading.Event,
                  ) -> Iterable[np.ndarray]:
    """PortAudio input stream via sounddevice (Linux only).

    Mirrors :meth:`app.audio.linux_capture.LinuxSounddeviceCapture.
    _run_hardware`: a ``sounddevice.InputStream`` at the pipeline
    rate, one channel, int16. A ``system`` track with no device
    index is refused — it would silently record the default
    microphone instead of a monitor source.
    """
    from app.audio.capture import _to_int16_mono
    from app.audio.devices import load_sounddevice, sounddevice_status

    sd = load_sounddevice()
    if sd is None:
        raise RuntimeError(
            "Linux pre-roll needs sounddevice + PortAudio. "
            f"{sounddevice_status()}")
    if track == "system" and device_index is None:
        raise RuntimeError(
            "No PulseAudio/PipeWire monitor source selected — "
            "system pre-roll cannot be captured on this machine.")
    logger.info("Pre-roll feeder (%s): opening PortAudio input "
                "stream (device=%s, %d Hz)", track, device_index,
                sample_rate)
    with sd.InputStream(device=device_index,
                        channels=1,
                        samplerate=sample_rate,
                        dtype="int16",
                        blocksize=chunk_frames) as stream:
        while not stop_event.is_set():
            try:
                data, overflowed = stream.read(chunk_frames)
            except Exception:
                if stop_event.is_set():
                    break
                raise
            if overflowed:
                logger.warning("Pre-roll feeder (%s): PortAudio "
                               "input overflow — audio was dropped",
                               track)
            yield _to_int16_mono(data, 1)


__all__ = ["PreRollFeeder", "StreamFactory"]
