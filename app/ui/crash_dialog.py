"""Crash dialog (PRD Phase 5).

When an uncaught exception reaches the UI, the user gets a friendly
dialog — not a silent exit — with the log file path and a "Copy
error" button. No telemetry, no network: the error text stays on
the machine, in the log file.

Importing this module never requires PyQt6; :func:`show_crash_dialog`
raises a clear RuntimeError when Qt is absent.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

from app.ui.level_meter import QT_AVAILABLE

logger = logging.getLogger("offair.ui.crash_dialog")


def format_crash_report(exc_type, exc_value, tb) -> str:
    """One text block describing an exception (for the dialog + log)."""
    lines = traceback.format_exception(exc_type, exc_value, tb)
    return "".join(lines).strip() or f"{exc_type.__name__}: {exc_value}"


def format_crash_message(error_text: str, log_path: str | Path | None) -> str:
    """The user-facing message shown above the error details."""
    message = (
        "OffAir hit an unexpected error. Your recordings are safe — "
        "they are written to disk as they are captured, so at most the "
        "in-flight chunk was lost.")
    if log_path:
        message += (f"\n\nDetails were written to the log file:\n{log_path}\n"
                    "Please include that file if you report the problem.")
    else:
        message += "\n\nNo log file path is known for this error."
    return message


if QT_AVAILABLE:
    from PyQt6.QtWidgets import (
        QApplication,
        QDialog,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
    )

    class CrashDialog(QDialog):
        """Dialog with the error details and a Copy button."""

        def __init__(self, error_text: str,
                     log_path: str | Path | None = None,
                     parent=None) -> None:
            super().__init__(parent)
            self.setWindowTitle("OffAir — unexpected error")
            self._error_text = error_text
            layout = QVBoxLayout(self)
            from PyQt6.QtWidgets import QLabel
            message = QLabel(format_crash_message(error_text, log_path))
            message.setWordWrap(True)
            layout.addWidget(message)
            details = QTextEdit()
            details.setReadOnly(True)
            details.setPlainText(error_text)
            layout.addWidget(details)
            copy_btn = QPushButton("Copy error")
            copy_btn.clicked.connect(self._copy_error)
            layout.addWidget(copy_btn)
            ok_btn = QPushButton("Close")
            ok_btn.clicked.connect(self.accept)
            layout.addWidget(ok_btn)
            self._copy_btn = copy_btn

        def _copy_error(self) -> None:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(self._error_text)


def show_crash_dialog(error_text: str,
                      log_path: str | Path | None = None,
                      parent=None) -> bool:
    """Show the crash dialog. Returns True if it was shown.

    Raises RuntimeError when PyQt6 is not installed — the caller
    (the exception hook) must fall back to logging only in that case.
    """
    if not QT_AVAILABLE:
        raise RuntimeError(
            "PyQt6 is not installed — cannot show a crash dialog. "
            f"The error was logged instead: {error_text[:200]}")
    dialog = CrashDialog(error_text, log_path, parent)
    dialog.exec()
    return True
