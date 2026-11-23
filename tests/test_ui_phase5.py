"""Phase 5 tests — theme stylesheets, crash dialog/report, log rotation.

No Qt is installed in this environment, so these tests cover the
pure parts (QSS text, crash-report formatting, rotation config)
and the import-guard behaviour of the Qt-dependent helpers.
"""
from __future__ import annotations

import logging
import sys

import pytest

from app.config import AppConfig
from app.logging_setup import (
    LOG_BACKUP_COUNT,
    LOG_MAX_BYTES,
    format_crash_report,
    install_exception_hook,
    log_file_path,
    setup_logging,
)
from app.ui.crash_dialog import format_crash_message, show_crash_dialog
from app.ui.theme import (
    DEFAULT_THEME,
    apply_theme,
    available_themes,
    stylesheet_for,
)


def test_theme_stylesheets_are_light_by_default_and_distinct():
    assert DEFAULT_THEME == "light"
    assert available_themes() == ["dark", "light"]
    dark = stylesheet_for("dark")
    light = stylesheet_for("light")
    assert "#17171B" in dark
    assert "#F4F4F5" in dark
    assert "#F7F6F3" in light
    assert "#ffffff" in light
    assert dark != light
    assert "QPushButton" in dark and "QProgressBar" in dark


def test_stylesheet_for_unknown_theme_rejected():
    with pytest.raises(ValueError, match="Unknown theme"):
        stylesheet_for("neon")


def test_apply_theme_sets_stylesheet_on_a_fake_widget():
    class _FakeWidget:
        def __init__(self):
            self.stylesheet = None

        def setStyleSheet(self, qss):
            self.stylesheet = qss

    widget = _FakeWidget()
    returned = apply_theme(widget, "dark")
    assert returned == stylesheet_for("dark")
    assert widget.stylesheet == returned
    with pytest.raises(ValueError, match="Unknown theme"):
        apply_theme(widget, "neon")


def test_apply_theme_without_qt_raises_clear_error():
    with pytest.raises(RuntimeError, match="setStyleSheet"):
        apply_theme(object(), "dark")


def test_crash_report_formats_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        report = format_crash_report(*sys.exc_info())
    assert "ValueError" in report
    assert "boom" in report
    assert "Traceback" in report


def test_crash_message_points_at_log_and_says_recordings_are_safe(tmp_path):
    message = format_crash_message("ValueError: boom",
                                   tmp_path / "offair.log")
    assert "recordings are safe" in message
    assert str(tmp_path / "offair.log") in message
    message2 = format_crash_message("err", None)
    assert "No log file path" in message2


def test_show_crash_dialog_without_qt_raises_clear_error():
    # PyQt6 is not installed here, so this must fail cleanly, not crash.
    with pytest.raises(RuntimeError, match="PyQt6 is not installed"):
        show_crash_dialog("ValueError: boom", None)


def test_log_rotation_config_is_about_5mb_with_one_backup():
    assert LOG_MAX_BYTES == 5 * 1024 * 1024
    assert LOG_BACKUP_COUNT == 1
    assert log_file_path("/tmp/out") is not None
    assert log_file_path(None) is None


def test_setup_logging_uses_rotating_handler(tmp_path):
    logger = logging.getLogger("offair")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    try:
        setup_logging(tmp_path)
        from logging.handlers import RotatingFileHandler
        rotating = [h for h in logger.handlers
                    if isinstance(h, RotatingFileHandler)]
        assert len(rotating) == 1
        assert rotating[0].maxBytes == 5 * 1024 * 1024
        assert rotating[0].backupCount == 1
        # Calling twice does not duplicate handlers.
        setup_logging(tmp_path)
        assert len([h for h in logger.handlers
                    if isinstance(h, RotatingFileHandler)]) == 1
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def test_install_exception_hook_logs_and_restores(tmp_path):
    logger = logging.getLogger("offair")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    log_file = tmp_path / "offair.log"
    setup_logging(tmp_path)
    previous = install_exception_hook(log_path=log_file, show_dialog=False)
    try:
        try:
            raise RuntimeError("test crash")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())
        # Flush handlers, then check the log captured the crash.
        for handler in logger.handlers:
            handler.flush()
        text = log_file.read_text(encoding="utf-8")
        assert "Uncaught exception" in text
        assert "test crash" in text
    finally:
        sys.excepthook = previous
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def test_main_window_imports_without_qt_and_settings_dialog_guards():
    from app.ui.main_window import create_main_window
    with pytest.raises(RuntimeError, match="PyQt6 is not installed"):
        create_main_window(AppConfig())
    from app.ui.settings_dialog import SettingsDialog
    with pytest.raises(RuntimeError, match="PyQt6 is not installed"):
        SettingsDialog(AppConfig())
    from app.ui.browser import create_browser
    with pytest.raises(RuntimeError, match="PyQt6 is not installed"):
        create_browser(AppConfig())
