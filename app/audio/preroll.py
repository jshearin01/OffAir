"""Pre-roll ring buffer: audio captured before Record is pressed.

Opt-in, default OFF (see ``app.config`` ``preroll_enabled`` /
``preroll_seconds``). When pre-roll is on, a feeder keeps the most
recent seconds of each track (mic / system) in this buffer; when a
recording starts, the UI layer drains the buffer
(:meth:`PreRollBuffer.drain_all`) and hands the samples to
:meth:`app.audio.capture.DualCapture.start` as ``preroll_samples``,
so the buffered seconds lead the new session's WAVs.

**Memory only, never disk.** This module performs no file I/O at
all: chunks live in a ``collections.deque`` and are discarded when
they age out, when the buffer is drained, or when the process ends.
Buffered audio reaches a file only via the drain-into-capture path
above — and only if a recording actually starts. If no recording
starts, the buffered audio is never saved anywhere.

**Feeder:** :class:`app.audio.preroll_feeder.PreRollFeeder` is the
always-on capture loop that calls :meth:`PreRollBuffer.feed`
while pre-roll is enabled and no recording is in progress; the UI
layer drives it with explicit start/stop calls and stops it
synchronously before capture starts. The feeder's Windows stream
path has not been run on real Windows hardware — do not claim
pre-roll captures opening syllables on a real machine until that
run exists (see ``docs/PER_APP_CAPTURE.md``).

Chunks are int16 mono PCM at the pipeline sample rate (48 kHz), as
``bytes`` or numpy arrays — the same shapes
:meth:`app.audio.capture.WASAPICapture._handle_chunk` accepts.
"""
from __future__ import annotations

import threading
from collections import deque

import numpy as np

#: Tracks the buffer holds. Mirrors the two WAV tracks a session
#: records (``mic.wav`` / ``system.wav``).
PREROLL_TRACKS: tuple[str, ...] = ("mic", "system")


def _to_int16_mono(chunk: bytes | np.ndarray) -> np.ndarray:
    """Normalise one chunk to a flat int16 mono array (no disk, no I/O)."""
    if isinstance(chunk, (bytes, bytearray)):
        arr = np.frombuffer(bytes(chunk), dtype=np.int16)
    else:
        arr = np.asarray(chunk)
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, -1.0, 1.0)
            arr = (arr * 32767).astype(np.int16)
        else:
            arr = arr.astype(np.int16)
    return arr.ravel().astype(np.int16)


class PreRollBuffer:
    """A bounded, memory-only ring buffer of recent audio per track.

    ``seconds`` is how much audio is kept (the config default is 30,
    range 10–120 — validated in ``app.config``; this class only
    requires a positive value). ``sample_rate`` is the pipeline rate
    (48 kHz). Thread-safe: ``feed`` may run on a capture thread while
    ``snapshot`` / ``drain_all`` run on the UI thread.
    """

    def __init__(self, seconds: float = 30.0,
                 sample_rate: int = 48000,
                 tracks: tuple[str, ...] = PREROLL_TRACKS) -> None:
        if not seconds > 0:
            raise ValueError("pre-roll seconds must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.seconds = float(seconds)
        self.sample_rate = int(sample_rate)
        self.tracks = tuple(tracks)
        self._max_samples = int(self.seconds * self.sample_rate)
        self._lock = threading.Lock()
        # Per track: deque of (sequence, samples) chunks plus a
        # running sample count, so trimming is O(chunks), not O(audio).
        self._chunks: dict[str, deque] = {t: deque() for t in self.tracks}
        self._counts: dict[str, int] = {t: 0 for t in self.tracks}

    # -------------------------------------------------------------- feed ---
    def feed(self, track: str, chunk: bytes | np.ndarray) -> None:
        """Append one chunk to ``track``'s buffer, dropping the oldest.

        Only the most recent ``seconds`` of audio is kept; older
        chunks are discarded as new ones arrive. Never touches disk.
        """
        samples = _to_int16_mono(chunk)
        if samples.size == 0:
            return
        with self._lock:
            if track not in self._chunks:
                raise ValueError(
                    f"unknown pre-roll track {track!r}; "
                    f"expected one of {list(self.tracks)}")
            buf = self._chunks[track]
            buf.append(samples)
            self._counts[track] += int(samples.size)
            # Trim to exactly the last ``seconds``: drop whole chunks
            # from the front, then trim the head of the chunk that
            # straddles the boundary (a single chunk larger than the
            # whole buffer keeps only its tail).
            excess = self._counts[track] - self._max_samples
            while excess > 0 and buf:
                head = buf[0]
                if len(head) <= excess:
                    buf.popleft()
                    self._counts[track] -= len(head)
                    excess -= len(head)
                else:
                    buf[0] = head[excess:]
                    self._counts[track] -= excess
                    excess = 0

    # ------------------------------------------------------------ access ---
    def snapshot(self, track: str) -> np.ndarray:
        """A copy of ``track``'s buffered audio, oldest first.

        Does not consume the buffer — call :meth:`drain_all` (or
        :meth:`drain`) when the audio has been handed to a recording.
        """
        with self._lock:
            if track not in self._chunks:
                raise ValueError(
                    f"unknown pre-roll track {track!r}; "
                    f"expected one of {list(self.tracks)}")
            chunks = list(self._chunks[track])
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks).astype(np.int16, copy=True)

    def drain(self, track: str) -> np.ndarray:
        """Snapshot ``track`` and clear it (one-shot handoff)."""
        samples = self.snapshot(track)
        with self._lock:
            if track in self._chunks:
                self._chunks[track].clear()
                self._counts[track] = 0
        return samples

    def drain_all(self) -> dict[str, np.ndarray]:
        """Snapshot every track that has audio, and clear the buffer.

        Returns ``{track: samples}`` for tracks with at least one
        buffered sample; empty tracks are omitted, so an empty buffer
        drains to ``{}``. This is the handoff shape
        :meth:`app.audio.capture.DualCapture.start` accepts as
        ``preroll_samples``.
        """
        result: dict[str, np.ndarray] = {}
        for track in self.tracks:
            samples = self.drain(track)
            if samples.size:
                result[track] = samples
        return result

    def duration_seconds(self, track: str) -> float:
        """Buffered seconds for ``track`` (0.0 when empty)."""
        with self._lock:
            if track not in self._chunks:
                raise ValueError(
                    f"unknown pre-roll track {track!r}; "
                    f"expected one of {list(self.tracks)}")
            return self._counts[track] / self.sample_rate

    def clear(self) -> None:
        """Discard all buffered audio (e.g. pre-roll turned off)."""
        with self._lock:
            for track in self.tracks:
                self._chunks[track].clear()
                self._counts[track] = 0


__all__ = ["PREROLL_TRACKS", "PreRollBuffer"]
