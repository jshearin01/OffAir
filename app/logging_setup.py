"""Logging setup: a rotating log file in the output root, plus a
console handler only when a console exists.

Kept deliberately small. Calling setup_logging() twice does not add
duplicate handlers. The log file rotates at ~5 MB (one backup file),
so logging can never fill the disk. :func:`install_exception_hook`
logs uncaught exceptions and, when a Qt application is running,
shows the friendly crash dialog from :mod:`app.ui.crash_dialog`
— no telemetry, no network, the error stays on this machine.

Windowed builds (PyInstaller ``console=False``, see
docs/PACKAGING.md): ``sys.stderr`` is None, so a StreamHandler
attached to it would have nowhere to write — every emit would
raise inside the logging machinery. setup_logging() therefore
attaches the console handler ONLY when ``sys.stderr`` is not None;
in a windowed build the log file is the only sink, which is also
where the crash dialog points the user.
"""
from __future__ import annotations

import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "offair"
_LOG_FILE_NAME = "offair.log"
LOG_MAX_BYTES: int = 5 * 1024 * 1024  # ~5 MB per log file
LOG_BACKUP_COUNT: int = 1             # one rotated backup alongside it


def log_file_path(output_root: Path | str | None) -> Path | None:
    """The log file path for an output root, or None."""
    if output_root is None:
        return None
    return Path(output_root) / _LOG_FILE_NAME


def setup_logging(output_root: Path | str | None = None,
                  level: int = logging.INFO) -> logging.Logger:
    """Return the ``offair`` logger, wired to ``<output_root>/offair.log``
    (rotating, ~5 MB) when an output root is given, and to the console
    only when one exists (``sys.stderr`` is not None).

    In a PyInstaller windowed build (``console=False``) there is no
    console: ``sys.stderr`` is None, so no StreamHandler is attached
    and the log file is the only sink. With a console (dev runs, the
    ``--no-gui`` smoke check) the console handler is attached as
    before.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # Console handler only when there is a console to write to. In a
    # windowed frozen build sys.stderr is None; StreamHandler() would
    # bind to None and every emit would fail inside logging.
    if sys.stderr is not None and not any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, RotatingFileHandler)
            for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
        logger.addHandler(console)

    if output_root is not None:
        root = Path(output_root)
        try:
            root.mkdir(parents=True, exist_ok=True)
            log_file = root / _LOG_FILE_NAME
            if not any(isinstance(h, RotatingFileHandler)
                       and getattr(h, "baseFilename", "") == str(log_file)
                       for h in logger.handlers):
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=LOG_MAX_BYTES,
                    backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
                logger.addHandler(file_handler)
        except OSError:
            logger.warning("Could not create log file under %s", root,
                           exc_info=True)
    return logger


def format_crash_report(exc_type, exc_value, tb) -> str:
    """One text block describing an uncaught exception."""
    lines = traceback.format_exception(exc_type, exc_value, tb)
    return "".join(lines).strip() or f"{exc_type.__name__}: {exc_value}"


def install_exception_hook(log_path: str | Path | None = None,
                           show_dialog: bool = True):
    """Install a sys.excepthook that logs uncaught exceptions.

    The exception is always written to the log (critical level).
    When ``show_dialog`` is True and a Qt application is running,
    the friendly crash dialog is shown as well; if Qt is absent
    (or no application exists yet) the hook logs only and never
    raises a second exception. Returns the previous excepthook.
    """
    logger = logging.getLogger(LOGGER_NAME)
    previous_hook = sys.excepthook

    def _hook(exc_type, exc_value, tb) -> None:
        report = format_crash_report(exc_type, exc_value, tb)
        logger.critical("Uncaught exception:\n%s", report)
        # Flush the log so the crash is on disk before any dialog.
        for handler in logger.handlers:
            try:
                handler.flush()
            except Exception:  # noqa: BLE001 — logging must not crash
                pass
        if show_dialog:
            try:
                from PyQt6.QtWidgets import QApplication
                if QApplication.instance() is not None:
                    from app.ui.crash_dialog import show_crash_dialog
                    show_crash_dialog(report, log_path)
            except Exception:  # noqa: BLE001 — dialog is best-effort
                logger.warning("Could not show crash dialog", exc_info=True)
        # KeyboardInterrupt etc. still go to the previous hook so the
        # process can exit normally; everything else is handled above.
        if issubclass(exc_type, KeyboardInterrupt):
            previous_hook(exc_type, exc_value, tb)

    sys.excepthook = _hook
    return previous_hook
