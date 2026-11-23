"""Qt-gated tests — model default must not carry over between
versions (Nov 20 2026, Item 8).

The Settings dialog's model combo is the only place an explicit
model pick is recorded, and only a user-driven pick (the
``activated`` signal) records one. Run with the gui-shots venv::

    QT_QPA_PLATFORM=offscreen \
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \
      -m pytest tests/test_settings_dialog_model.py -q
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import AppConfig


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/offair-demo/runtime")
    Path("/tmp/offair-demo/runtime").mkdir(parents=True, exist_ok=True)
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _dialog(qapp, config):
    from app.ui.settings_dialog import SettingsDialog

    return SettingsDialog(config)


def test_untouched_save_does_not_set_explicit_flag(qapp, tmp_path):
    """A plain Settings save never records an explicit pick: a
    pre-flag "small" migrates to base, explicit False."""
    dlg = _dialog(qapp, AppConfig(output_root=tmp_path,
                                  whisper_model="small"))
    try:
        assert dlg.model_combo.currentData() == "base"
        saved = dlg.config_from_dialog()
        assert saved.whisper_model == "base"
        assert saved.whisper_model_explicit is False
    finally:
        dlg.close()


def test_untouched_save_preserves_explicit_pick(qapp, tmp_path):
    """An untouched save keeps an earlier explicit pick as-is."""
    dlg = _dialog(qapp, AppConfig(
        output_root=tmp_path, whisper_model="tiny",
        whisper_model_explicit=True))
    try:
        assert dlg.model_combo.currentData() == "tiny"
        saved = dlg.config_from_dialog()
        assert saved.whisper_model == "tiny"
        assert saved.whisper_model_explicit is True
    finally:
        dlg.close()


def test_programmatic_combo_set_does_not_set_flag(qapp, tmp_path):
    """setCurrentIndex alone (construction/refresh/sync) is not a
    user pick: the flag stays False even though the combo moved."""
    dlg = _dialog(qapp, AppConfig(output_root=tmp_path))
    try:
        dlg.model_combo.setCurrentIndex(
            dlg.model_combo.findData("tiny"))
        assert dlg.model_combo.currentData() == "tiny"
        saved = dlg.config_from_dialog()
        assert saved.whisper_model == "tiny"
        assert saved.whisper_model_explicit is False
        # …so the effective model is still the default, not tiny.
        assert saved.effective_whisper_model == "base"
    finally:
        dlg.close()


def test_activated_combo_pick_sets_flag(qapp, tmp_path):
    """A user pick — `activated`, which programmatic sets never
    emit — records the explicit choice."""
    dlg = _dialog(qapp, AppConfig(output_root=tmp_path))
    try:
        dlg.model_combo.setCurrentIndex(
            dlg.model_combo.findData("small"))
        dlg.model_combo.activated.emit(dlg.model_combo.currentIndex())
        saved = dlg.config_from_dialog()
        assert saved.whisper_model == "small"
        assert saved.whisper_model_explicit is True
        assert saved.effective_whisper_model == "small"
    finally:
        dlg.close()


def test_default_model_item_is_marked(qapp, tmp_path):
    """The current default reads as the default in the combo."""
    dlg = _dialog(qapp, AppConfig(output_root=tmp_path))
    try:
        base_text = dlg.model_combo.itemText(
            dlg.model_combo.findData("base"))
        assert "(default)" in base_text
        small_text = dlg.model_combo.itemText(
            dlg.model_combo.findData("small"))
        assert "(default)" not in small_text
    finally:
        dlg.close()


def test_model_hint_shows_default_then_explicit(qapp, tmp_path):
    """The hint line states the state at a glance, live."""
    dlg = _dialog(qapp, AppConfig(output_root=tmp_path))
    try:
        assert "default model (base)" in dlg.model_hint.text()
        assert "Pick a model" in dlg.model_hint.text()
        dlg.model_combo.setCurrentIndex(
            dlg.model_combo.findData("small"))
        dlg.model_combo.activated.emit(dlg.model_combo.currentIndex())
        assert "You picked small" in dlg.model_hint.text()
        assert "kept across updates" in dlg.model_hint.text()
    finally:
        dlg.close()


def test_model_hint_shows_explicit_on_open(qapp, tmp_path):
    """A dialog opened on an explicit pick shows the explicit
    hint from the start, not the default one."""
    dlg = _dialog(qapp, AppConfig(
        output_root=tmp_path, whisper_model="tiny",
        whisper_model_explicit=True))
    try:
        assert "You picked tiny" in dlg.model_hint.text()
        assert "kept across updates" in dlg.model_hint.text()
    finally:
        dlg.close()


def test_session_combo_change_does_not_set_persistent_flag(
        qapp, tmp_path):
    """The Session view's model combo stays an ephemeral
    per-session override: changing it never touches the
    persistent explicit flag."""
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        assert win.config.whisper_model_explicit is False
        win.model_combo.setCurrentIndex(
            win.model_combo.findData("small"))
        # The override works for this session…
        assert win._configured_model() == "small"
        # …but nothing persistent changed.
        assert win.config.whisper_model == "base"
        assert win.config.whisper_model_explicit is False
    finally:
        win.close()
