"""Waveform timeline for the Session view.

A custom-painted widget: RMS buckets (0.25 s) from the session's
``combined.wav`` (fallback ``mic.wav``), speaker-coloured segment
spans along the top, bookmark markers, click to hear that moment.
Audio decoding never runs on the UI thread — :func:`load_rms_buckets`
is pure (stdlib ``wave`` + numpy) and the main window calls it from
a worker, caching the result in memory per session.

Importing this module never requires PyQt6: the bucket loader and
the speaker-colour palette are plain Python, so they are
unit-testable without Qt; constructing :class:`WaveformWidget`
raises a clear RuntimeError when Qt is absent.
"""
from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path

import numpy as np

from app.ui.level_meter import QT_AVAILABLE

#: One RMS bucket per 0.25 s of audio.
BUCKET_SECONDS: float = 0.25

#: Fixed speaker palette: a label always gets the same colour, in
#: every view, in both themes. Colours are dark enough to read on
#: white and light enough to read on near-black.
SPEAKER_COLORS: tuple[str, ...] = (
    "#4F46E5", "#0891B2", "#059669", "#D97706",
    "#DC2626", "#7C3AED", "#DB2777", "#65A30D",
)

#: Bookmark marker colour (the same red as the Record button).
BOOKMARK_COLOR = "#DC2626"


def color_for_speaker(label: str | None) -> str:
    """The palette colour for a speaker label (deterministic).

    The same label always maps to the same colour: the palette
    index is the first byte of the label's MD5, so renaming a
    speaker re-colours their segments (the label changed), but
    nothing else ever shifts a colour. An empty label gets the
    first colour.
    """
    label = str(label or "")
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return SPEAKER_COLORS[digest[0] % len(SPEAKER_COLORS)]


def load_rms_buckets(audio_path: str | Path, *,
                     bucket_seconds: float = BUCKET_SECONDS,
                     ) -> tuple[list[float], float]:
    """RMS buckets for a WAV file, plus its duration in seconds.

    Returns ``(buckets, duration)``: one RMS value per
    ``bucket_seconds`` of audio, normalised 0.0–1.0 (full scale),
    mono (channels averaged). A missing, unreadable, or
    zero-length file returns ``([], 0.0)`` — the widget then
    shows its empty state instead of failing. No Qt, no disk
    writes: safe to call from a worker thread.
    """
    path = Path(audio_path)
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be positive")
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
    except (OSError, wave.Error, EOFError):
        return [], 0.0
    if frames == 0 or rate <= 0:
        return [], 0.0
    duration = frames / rate
    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        samples /= 32768.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
                   - 128.0) / 128.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        samples /= 2147483648.0
    else:
        return [], 0.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    bucket_frames = max(1, int(round(rate * bucket_seconds)))
    buckets: list[float] = []
    for start in range(0, len(samples), bucket_frames):
        chunk = samples[start:start + bucket_frames]
        if len(chunk) == 0:
            continue
        buckets.append(float(math.sqrt(float(np.mean(chunk ** 2)))))
    return buckets, duration


if QT_AVAILABLE:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QColor, QPainter, QPen
    from PyQt6.QtWidgets import QWidget

    class WaveformWidget(QWidget):
        """Painted waveform: bars, speaker spans, bookmark markers.

        :meth:`set_data` replaces everything shown (buckets,
        duration, segments, bookmarks). A click emits
        :attr:`time_clicked` with the clicked time in seconds —
        the main window plays from there through its single
        playback path. The widget paints only; it never decodes
        audio and never plays anything itself.
        """

        time_clicked = pyqtSignal(float)

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self._buckets: list[float] = []
            self._duration: float = 0.0
            # Segments: (start, end, speaker label).
            self._segments: list[tuple[float, float, str]] = []
            self._bookmarks: list[float] = []
            self.setMinimumHeight(64)
            self.setToolTip(
                "Waveform — click to hear that moment. Coloured "
                "spans are speaker segments; red lines are bookmarks.")

        def set_data(self, buckets: list[float], duration: float, *,
                     segments=None, bookmarks=None) -> None:
            """Show new waveform data (from the worker / cache)."""
            self._buckets = list(buckets)
            self._duration = float(duration)
            self._segments = [(float(s), float(e), str(sp))
                              for s, e, sp in (segments or [])]
            self._bookmarks = [float(t) for t in (bookmarks or [])]
            self.update()

        def clear(self) -> None:
            """Empty state (no session / no audio yet)."""
            self.set_data([], 0.0)

        def set_segments(self, segments) -> None:
            """Replace only the speaker spans (buckets unchanged)."""
            self._segments = [(float(s), float(e), str(sp))
                              for s, e, sp in segments]
            self.update()

        def set_bookmarks(self, bookmarks) -> None:
            """Replace only the bookmark markers (buckets unchanged)."""
            self._bookmarks = [float(t) for t in bookmarks]
            self.update()

        def _time_at(self, x: int) -> float:
            """Audio time for a horizontal pixel (clamped)."""
            width = max(1, self.width())
            if self._duration <= 0:
                return 0.0
            return max(0.0, min(self._duration,
                                x / width * self._duration))

        def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt name
            if event.button() == Qt.MouseButton.LeftButton:
                self.time_clicked.emit(
                    self._time_at(event.position().x()))
            super().mousePressEvent(event)

        def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
            painter = QPainter(self)
            rect = self.rect()
            palette = self.palette()
            base = palette.color(palette.ColorRole.Base)
            painter.fillRect(rect, base)
            if not self._buckets or self._duration <= 0:
                painter.setPen(palette.color(
                    palette.ColorRole.PlaceholderText))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                                 "No waveform yet — no audio in this "
                                 "session.")
                return
            width, height = rect.width(), rect.height()
            # Speaker spans: a coloured band along the top.
            strip_height = 8
            for start, end, speaker in self._segments:
                x0 = int(start / self._duration * width)
                x1 = max(x0 + 1, int(end / self._duration * width))
                painter.fillRect(x0, 0, x1 - x0, strip_height,
                                 QColor(color_for_speaker(speaker)))
            # RMS bars below the strip.
            bar_area_top = strip_height + 4
            bar_area_height = max(1, height - bar_area_top - 4)
            bar_color = palette.color(palette.ColorRole.Text)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bar_color)
            count = len(self._buckets)
            bar_width = width / count
            for index, rms in enumerate(self._buckets):
                bar_height = max(1.0, rms * bar_area_height)
                top = (bar_area_top
                       + (bar_area_height - bar_height) / 2.0)
                painter.drawRect(int(index * bar_width), int(top),
                                 max(1, int(bar_width) - 1),
                                 int(bar_height))
            # Bookmark markers: red lines over the bars.
            pen = QPen(QColor(BOOKMARK_COLOR))
            pen.setWidth(2)
            painter.setPen(pen)
            for mark in self._bookmarks:
                x = int(mark / self._duration * width)
                painter.drawLine(x, strip_height, x, height)

    __all__ = [
        "BUCKET_SECONDS",
        "BOOKMARK_COLOR",
        "SPEAKER_COLORS",
        "WaveformWidget",
        "color_for_speaker",
        "load_rms_buckets",
    ]
else:
    class WaveformWidget:  # type: ignore[no-redef]
        """Placeholder when PyQt6 is absent: construction fails loudly."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyQt6 is not installed — install it with "
                "`pip install -r requirements.txt` to use the GUI.")

    __all__ = [
        "BUCKET_SECONDS",
        "BOOKMARK_COLOR",
        "SPEAKER_COLORS",
        "WaveformWidget",
        "color_for_speaker",
        "load_rms_buckets",
    ]
