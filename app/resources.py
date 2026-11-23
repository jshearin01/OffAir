"""Resolve bundled resource files in source runs and frozen builds.

The app ships two bundled data entries — the application icon,
``assets/offair.ico``, and the ``docs/`` Markdown folder rendered by
the in-app Help view (both added to the spec's ``datas``, see
docs/PACKAGING.md). In a source checkout they live next to the repo
root; in a PyInstaller onedir build they live under ``sys._MEIPASS``,
the bundle root. :func:`resource_path` hides that difference.

Importing this module never requires PyQt6 (same pattern as
:mod:`app.ui.level_meter`): when Qt is absent, :func:`app_icon` and
:func:`set_app_icon` report that no icon is available instead of
raising. A missing icon file is likewise never fatal — the app
starts without an icon and logs a warning.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("offair.ui.resources")

try:
    from PyQt6.QtGui import QIcon

    QT_AVAILABLE = True
except Exception:  # ImportError on machines without PyQt6 — by design.
    QT_AVAILABLE = False
    QIcon = None  # type: ignore[assignment]

APP_ICON_RELATIVE = "assets/offair.ico"
DOCS_RELATIVE = "docs"


def docs_dir() -> Path:
    """Absolute path to the bundled docs/ folder, or the source one.

    In a frozen build this is only present when docs/ is in the
    spec's ``datas``; callers must tolerate it missing (see
    app/ui/help_view.py — a missing docs/ shows a plain message,
    never a crash).
    """
    return resource_path(DOCS_RELATIVE)


def resource_path(relative: str) -> Path:
    """Absolute path to a repo resource, frozen or source.

    Frozen (PyInstaller): under ``sys._MEIPASS`` (the bundle root,
    where the spec's ``datas`` entries land). Source checkout: under
    the repo root (the parent of this file's ``app/`` package).
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent.parent
    return base / relative


def app_icon_path() -> Path | None:
    """Path to the OffAir .ico, or None when it is not present."""
    path = resource_path(APP_ICON_RELATIVE)
    return path if path.is_file() else None


def app_icon():
    """The OffAir application icon, or None when Qt or the file is missing.

    Returns a ``QIcon`` built from the multi-size .ico, so Qt picks
    the right size for the title bar, the taskbar and dialogs.
    """
    if not QT_AVAILABLE:
        return None
    path = app_icon_path()
    if path is None:
        return None
    icon = QIcon(str(path))
    return icon if not icon.isNull() else None


def set_app_icon(target) -> bool:
    """Set the OffAir icon on a QApplication or a window.

    Returns True when the icon was set. A missing icon file (or no
    Qt) logs a warning and returns False — the app must still start,
    just without its mark. ``target`` needs a ``setWindowIcon`` method
    (QApplication and QMainWindow both have one).
    """
    icon = app_icon()
    if icon is None:
        logger.warning(
            "Application icon not found at %s; starting without it",
            resource_path(APP_ICON_RELATIVE))
        return False
    target.setWindowIcon(icon)
    return True
