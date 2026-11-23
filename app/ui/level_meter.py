"""Live audio level meter widget — no audio dependencies.

Importing this module must never fail when PyQt6 is absent (e.g. on a
machine that has only run ``py_compile`` or the test suite): in that case
``QT_AVAILABLE`` is False and instantiating :class:`LevelMeter` raises a
clear RuntimeError. All Qt code lives behind the import guard.

The meter is custom-painted: a slim rounded track with a rounded fill,
green below 0.6, amber to 0.85, red above — the red zone is a visual
clipping warning only; the recorder itself hard-clips in the mixer.
Track/border colours follow the active theme via :meth:`set_theme`
(the main window calls it when the theme is applied or changed);
QSS cannot reach custom paint, so the theme is passed in code.
"""
from __future__ import annotations

try:
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor, QPainter, QPainterPath
    from PyQt6.QtWidgets import QWidget

    QT_AVAILABLE = True
except Exception:  # ImportError on machines without PyQt6 — by design.
    QT_AVAILABLE = False
    QRectF = None  # type: ignore[assignment]
    Qt = None  # type: ignore[assignment]
    QColor = None  # type: ignore[assignment]
    QPainter = None  # type: ignore[assignment]
    QPainterPath = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]

# Track + border colours per theme (must stay in sync with the
# palettes in app/ui/theme.py — the track is the border token).
_THEME_COLORS = {
    "light": {"track": "#E6E2DA", "border": "#D5D0C6"},
    "dark": {"track": "#2E2E38", "border": "#3F3F4C"},
}


class LevelMeter(QWidget):  # type: ignore[misc]
    """A slim rounded level bar driven by a 0..1 float via :meth:`set_level`."""

    def __init__(self, orientation: str = "horizontal",
                 parent=None) -> None:
        if not QT_AVAILABLE:
            raise RuntimeError(
                "PyQt6 is not installed — install it with "
                "`pip install -r requirements.txt` to use the GUI.")
        super().__init__(parent)
        self._level = 0.0
        self._orientation = orientation
        self._theme = "light"
        if orientation == "vertical":
            self.setMinimumSize(10, 120)
        else:
            self.setMinimumSize(120, 10)

    def set_theme(self, theme: str) -> None:
        """Track/border colours follow ``theme`` ("light" or "dark")."""
        self._theme = theme if theme in _THEME_COLORS else "light"
        self.update()

    @property
    def theme(self) -> str:
        return self._theme

    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, float(level)))
        self.update()

    @property
    def level(self) -> float:
        return self._level

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not QT_AVAILABLE:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        colours = _THEME_COLORS[self._theme]
        radius = rect.height() / 2 if self._orientation == "horizontal" \
            else rect.width() / 2
        track = QPainterPath()
        track.addRoundedRect(
            QRectF(rect.adjusted(0, 0, -1, -1)), radius, radius)
        painter.fillPath(track, QColor(colours["track"]))
        painter.setPen(QColor(colours["border"]))
        painter.drawPath(track)
        if self._level <= 0:
            painter.end()
            return
        if self._orientation == "vertical":
            height = max(int(rect.height() * self._level), 2)
            bar_rect = rect.adjusted(
                0, rect.height() - height, -1, -1)
        else:
            width = max(int(rect.width() * self._level), 2)
            bar_rect = rect.adjusted(0, 0, width - rect.width() - 1, -1)
        bar = QPainterPath()
        bar.addRoundedRect(QRectF(bar_rect), radius, radius)
        painter.fillPath(bar, QColor(self._colour_for_level()))
        painter.end()

    def _colour_for_level(self) -> str:
        if self._level < 0.6:
            return "#22C55E"
        if self._level < 0.85:
            return "#F59E0B"
        return "#EF4444"
