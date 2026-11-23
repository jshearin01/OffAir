"""Visual-design tests — the consumer-grade redesign (post-Phase 6).

Two halves, like test_ui_phase6.py:

* Qt-free: the two DEFAULT_THEME definitions agree and are "light";
  both stylesheets share the same selector set (same structure in
  both themes — only colours change) and carry the new design-system
  selectors (recording pill states, field labels, page title).
* Qt-gated (``pytest.importorskip("PyQt6")`` per test — run with the
  gui-shots venv, offscreen): the light theme is the default applied
  to a new window, the recording pill/timer state transitions, the
  empty-state placeholders, the Settings theme picker, and the
  browser's empty-state copy.

Run the Qt half with::

    QT_QPA_PLATFORM=offscreen \
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \
      -m pytest tests/test_ui_visual.py -q
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import DEFAULT_THEME as CONFIG_DEFAULT_THEME
from app.ui.theme import (
    DEFAULT_THEME,
    ACCENT_DARK,
    ACCENT_LIGHT,
    apply_theme,
    stylesheet_for,
)


# ------------------------------------------------------------ Qt-free -------
def test_light_is_the_default_in_both_places():
    # config.py and theme.py each define DEFAULT_THEME — they must
    # agree, or the app would default one way and the stylesheet
    # another. Light is the default: the clean, airy consumer look.
    assert DEFAULT_THEME == "light"
    assert CONFIG_DEFAULT_THEME == "light"
    assert CONFIG_DEFAULT_THEME == DEFAULT_THEME


def test_both_themes_share_the_design_system_selectors():
    """Same structure in both themes — only colours change."""
    required = (
        "QLabel#appTitle",
        "QLabel#appSubtitle",
        "QLabel#pageTitle",
        "QLabel#timerLabel",
        'QLabel#timerLabel[recording="true"]',
        "QLabel#sectionTitle",
        "QLabel#fieldLabel",
        "QLabel#modeLabel",
        "QLabel#hint",
        "QLabel#loadedSessionLabel",
        "QLabel#statusPill",
        "QLabel#statusPillRecording",
        "QLabel#statusPillPaused",
        "QPushButton#primaryButton",
        "QPushButton#recordButton",
        "QPushButton#dangerButton",
        "QFrame#card",
        "QFrame#headerBar",
        "QListWidget#navList",
        "QTabWidget::pane",
        "QTabBar::tab:selected",
        "QSplitter::handle",
    )
    for theme in ("dark", "light"):
        qss = stylesheet_for(theme)
        for selector in required:
            assert selector in qss, (theme, selector)


def test_design_tokens_are_one_radius_per_family():
    """One corner radius per family (Nov 10 2026 — the ugly-UI fix).

    Cards 12, controls 8, the status-pill family 10. The only other
    radii in the stylesheets are the thin chrome (progress bar 3 =
    half its 6 px height; scrollbar handle 4 = half its 8 px width),
    which are fully rounded by construction — the same rule, not a
    fourth family. Both themes carry the same set.
    """
    import re

    from app.ui.theme import (
        CARD_RADIUS,
        CONTROL_RADIUS,
        PILL_RADIUS,
    )

    assert (CARD_RADIUS, CONTROL_RADIUS, PILL_RADIUS) == (12, 8, 10)
    for theme in ("dark", "light"):
        qss = stylesheet_for(theme)
        radii = sorted({int(v) for v in
                        re.findall(r"border-radius:\s*(\d+)px", qss)})
        assert radii == [3, 4, 8, 10, 12], (theme, radii)


def test_radius_tokens_applied_to_the_right_selectors():
    """Each selector's radius is its family's token, not a one-off."""
    import re

    for theme in ("dark", "light"):
        qss = stylesheet_for(theme)
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", qss):
            sel = " ".join(selector.split())
            match = re.search(r"border-radius:\s*(\d+)px", body)
            if not match:
                continue
            radius = int(match.group(1))
            if "QFrame#card" in sel or sel in ("QGroupBox",):
                assert radius == 12, (theme, sel, radius)
            elif any(key in sel for key in
                     ("statusPill", "modeLabel", "autoBadge")):
                assert radius == 10, (theme, sel, radius)
            elif radius in (3, 4):
                assert ("QProgressBar" in sel
                        or "QScrollBar" in sel), (theme, sel, radius)
            else:
                assert radius == 8, (theme, sel, radius)


def test_accent_is_one_indigo_used_in_both_themes():
    light = stylesheet_for("light")
    dark = stylesheet_for("dark")
    # Light primary: indigo-600; dark primary: indigo-500 (white text
    # stays legible on the dark surface).
    assert ACCENT_LIGHT in light
    assert ACCENT_DARK in dark
    assert ACCENT_LIGHT == "#4F46E5"
    assert ACCENT_DARK == "#6366F1"
    # Warm off-white light background, layered near-black dark one.
    assert "#F7F6F3" in light
    assert "#17171B" in dark
    assert light != dark


def test_level_meter_theme_colours_match_the_palettes():
    from app.ui.level_meter import _THEME_COLORS

    light = stylesheet_for("light")
    dark = stylesheet_for("dark")
    # The meter's track colour is the same token as the light/dark
    # border family — if the palette moves, this test says so.
    assert _THEME_COLORS["light"]["track"] in light
    assert _THEME_COLORS["dark"]["track"] in dark


# ------------------------------------------------------------ Qt-gated ------
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


def test_main_window_defaults_to_light_theme(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        assert win.config.theme == "light"
        assert win.styleSheet() == stylesheet_for("light")
        assert win.mic_meter.theme == "light"
        assert win.system_meter.theme == "light"
        # Record is the unmistakable primary action: tall, red.
        assert win.record_btn.minimumHeight() >= 40
        assert win.record_btn.objectName() == "recordButton"
    finally:
        win.close()


def test_recording_pill_and_timer_state_transitions(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        assert win.status_label.objectName() == "statusPill"
        assert win.elapsed_label.property("recording") in (False, None)
        win._set_recording_ui("recording")
        assert win.status_label.objectName() == "statusPillRecording"
        assert win.elapsed_label.property("recording") is True
        win._set_recording_ui("paused")
        assert win.status_label.objectName() == "statusPillPaused"
        assert win.elapsed_label.property("recording") is False
        win._set_recording_ui("idle")
        assert win.status_label.objectName() == "statusPill"
    finally:
        win.close()


def test_empty_state_placeholders_are_present_and_unselectable(
        qapp, tmp_path):
    from PyQt6.QtCore import Qt
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        for lst, needle in (
                (win.segments_list, "No transcript yet"),
                (win.suggestions_list, "No suggestions yet"),
                (win.nearlive_list, "Nothing here yet")):
            assert lst.count() == 1, needle
            item = lst.item(0)
            assert needle in item.text()
            assert not (item.flags() & Qt.ItemFlag.ItemIsSelectable)
            assert not (item.flags() & Qt.ItemFlag.ItemIsEnabled)
        # A placeholder row must not enable Confirm/Correct.
        assert win.suggestions_list.currentRow() == -1
        assert win.confirm_btn.isEnabled() is False
    finally:
        win.close()


def test_settings_theme_picker_round_trips_light(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(AppConfig(output_root=tmp_path))
    try:
        assert dlg.theme_combo.currentData() == "light"
        assert "Light" in dlg.theme_combo.currentText()
        dlg.theme_combo.setCurrentIndex(
            dlg.theme_combo.findData("dark"))
        out = dlg.config_from_dialog()
        assert out.theme == "dark"
    finally:
        dlg.close()


def test_browser_empty_state_copy(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.browser import create_browser

    browser = create_browser(AppConfig(output_root=tmp_path))
    try:
        browser.show()
        qapp.processEvents()
        assert browser.session_list.count() == 0
        assert "No sessions yet" in browser.status_label.text()
        browser.search_edit.setText("December")
        qapp.processEvents()
        # No sessions exist, so any search reports no matches and
        # echoes the query. The exact wording depends on the mode:
        # Hybrid with a working index says "No matches for …",
        # while the offline fallback says "No sessions match …".
        _text = browser.status_label.text()
        assert "match" in _text.lower() and "December" in _text
    finally:
        browser.close()


def test_theme_switch_reaches_every_window(qapp, tmp_path):
    """Switching theme in Settings re-themes the app, not one window.

    (The old code applied the new theme to the main window only, so
    the Browser/Settings kept the old theme until restart.)
    """
    from PyQt6.QtWidgets import QApplication
    from app.config import AppConfig
    from app.ui.browser import create_browser
    from app.ui.main_window import create_main_window
    from app.ui.theme import apply_theme_to_app

    app = QApplication.instance()
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        apply_theme_to_app(app, "light")
        assert app.styleSheet() == stylesheet_for("light")
        browser = create_browser(AppConfig(output_root=tmp_path))
        try:
            assert browser.styleSheet() == ""
            apply_theme_to_app(app, "dark")
            assert app.styleSheet() == stylesheet_for("dark")
        finally:
            browser.close()
            apply_theme_to_app(app, "light")
    finally:
        win.close()


# ------------------------------------------- geometry: no overlap/clipping ---
def _geometry_problems(widget, *, label: str) -> list[str]:
    """Visible-widget geometry audit for one shown top-level widget.

    Every visible widget must sit fully inside its parent, no two
    visible siblings may overlap, and no QLabel's text may be
    clipped (wrapped labels: heightForWidth fits the allocated
    height; single-line labels: sizeHint fits the allocated width).
    Widgets inside a scroll area are exempt from the inside-parent
    and overlap rules — scrolled content legitimately overflows its
    viewport — but their labels are still checked for clipping.
    Returns a list of human-readable problems (empty = clean).
    """
    from PyQt6.QtWidgets import (
        QAbstractScrollArea,
        QLabel,
        QSplitterHandle,
        QWidget,
    )

    problems: list[str] = []

    def _in_scroll_area(w) -> bool:
        parent = w.parentWidget()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea):
                return True
            parent = parent.parentWidget()
        return False

    def _describe(w) -> str:
        name = type(w).__name__
        obj = w.objectName()
        text = ""
        if isinstance(w, QLabel):
            text = f" {w.text()[:40]!r}"
        elif hasattr(w, "text") and callable(w.text):
            try:
                text = f" {w.text()[:40]!r}"
            except Exception:  # noqa: BLE001 — describe is best-effort
                text = ""
        return f"{label}: {name}#{obj}{text} geo={w.geometry().getRect()}"

    visible = [w for w in [widget] + widget.findChildren(QWidget)
               if w.isVisible()]
    # Inside-parent + label clipping.
    for w in visible:
        parent = w.parentWidget()
        if parent is not None and not _in_scroll_area(w) \
                and not isinstance(w, QSplitterHandle):
            if not parent.rect().contains(w.geometry()):
                problems.append(f"OUTSIDE PARENT: {_describe(w)} "
                                f"parent={parent.rect().getRect()}")
        if isinstance(w, QLabel) and w.text():
            if w.wordWrap():
                needed = w.heightForWidth(w.width())
                if needed > w.height() + 1:
                    problems.append(
                        f"CLIPPED WRAPPED LABEL: {_describe(w)} "
                        f"needs height {needed}, has {w.height()}")
            else:
                needed = w.sizeHint().width()
                if needed > w.width() + 1:
                    problems.append(
                        f"CLIPPED LABEL: {_describe(w)} "
                        f"needs width {needed}, has {w.width()}")
    # Sibling overlap (same parent, both visible, not in a scroll area).
    by_parent: dict = {}
    for w in visible:
        parent = w.parentWidget()
        if parent is None or _in_scroll_area(w):
            continue
        by_parent.setdefault(parent, []).append(w)
    for _parent, sibs in by_parent.items():
        for i, first in enumerate(sibs):
            for second in sibs[i + 1:]:
                inter = first.geometry().intersected(second.geometry())
                if inter.width() > 1 and inter.height() > 1:
                    problems.append(
                        f"OVERLAP: {_describe(first)} x "
                        f"{_describe(second)} "
                        f"intersection={inter.getRect()}")
    return problems


def test_main_window_geometry_clean_at_1280x800_both_themes(
        qapp, tmp_path):
    """No overlaps, nothing outside its parent, no clipped labels —
    every view, both themes, at the 1280x800 target size."""
    from PyQt6.QtWidgets import QApplication
    from app.config import AppConfig
    from app.ui.main_window import create_main_window
    from app.ui.theme import apply_theme_to_app

    app = QApplication.instance()
    problems: list[str] = []
    for theme in ("light", "dark"):
        apply_theme_to_app(app, theme)
        win = create_main_window(
            AppConfig(output_root=tmp_path, theme=theme))
        try:
            win.show()
            qapp.processEvents()
            assert win.width() == 1280 and win.height() == 800
            for index in range(win.tabs.count()):
                win.show_view(win.tabs.tabText(index))
                # A scroll area settles its content on a second
                # event-loop turn (first pass sizes it to the
                # viewport, second to its full height) — audit the
                # settled layout, which is what the user sees.
                qapp.processEvents()
                qapp.processEvents()
                problems.extend(_geometry_problems(
                    win, label=f"{theme}/{win.tabs.tabText(index)}"))
        finally:
            win.close()
    apply_theme_to_app(app, "light")
    assert problems == [], "\n".join(problems)


def test_recording_state_geometry_clean(qapp, tmp_path):
    """The recording-in-progress state (red pill, timer, AUTO badge
    off, meters live) is geometry-clean too — it is the state the
    user stares at for the whole meeting."""
    import time
    from types import SimpleNamespace

    from PyQt6.QtWidgets import QApplication
    from app.audio.capture import CaptureState
    from app.config import AppConfig
    from app.storage.sessions import create_session
    from app.ui.main_window import create_main_window
    from app.ui.theme import apply_theme_to_app

    app = QApplication.instance()
    apply_theme_to_app(app, "light")
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        session = create_session(tmp_path, app_name="Teams")
        win.session_dir = session
        win.capture = SimpleNamespace(state=CaptureState.RECORDING,
                                      mode="loopback", warning="")
        # The recording state, as on_record leaves it: the short
        # capture label plus the one-line session line.
        win.session_label.setText(f"Recording to: {session.name}")
        win.status_label.setText("● Recording… ")
        win.mode_label.setText(
            "Capture: system-wide loopback "
            "(per-app capture unavailable — see warning)")
        win._record_started_at = time.time() - 277
        win._elapsed_paused_total = 0.0
        win._pause_started_at = None
        win._update_elapsed()
        win._set_recording_ui("recording")
        win.mic_meter.set_level(0.62)
        win.system_meter.set_level(0.41)
        win._update_buttons()
        win.show()
        qapp.processEvents()
        problems = _geometry_problems(win, label="recording")
        assert problems == [], "\n".join(problems)
        # And the stopped state, as on_stop leaves it: the
        # three-line capture explanation is back, with the one-line
        # "Saved: …" line. (The old three-line session line pushed
        # the card past 800 px and clipped the explanation — the
        # geometry audit is what caught it.)
        win.capture = SimpleNamespace(state=CaptureState.STOPPED,
                                      mode="loopback", warning="")
        win.session_label.setText(f"Saved: {session.name} · 61.3s")
        win._update_capture_mode_ui()
        win._set_recording_ui("idle")
        win._update_buttons()
        # Two event-loop turns, as in the main geometry test: the
        # Record view's left column is two cards plus a stretch
        # since the schedules card landed (Nov 22 2026), and the
        # column settles on the second turn — one pump audits a
        # transitional layout and falsely flags the mode label.
        qapp.processEvents()
        qapp.processEvents()
        problems = _geometry_problems(win, label="stopped")
        assert problems == [], "\n".join(problems)
    finally:
        win.close()


def test_browser_and_settings_geometry_clean(qapp, tmp_path):
    """Recording Browser (1100x700) and Settings (760x780): same
    audit — nothing outside its parent, no overlaps, no clipped
    labels, in the light theme (the dark theme shares the layout)."""
    from app.config import AppConfig
    from app.ui.browser import create_browser
    from app.ui.settings_dialog import SettingsDialog

    browser = create_browser(AppConfig(output_root=tmp_path))
    try:
        browser.resize(1100, 700)
        browser.show()
        qapp.processEvents()
        problems = _geometry_problems(browser, label="browser")
        assert problems == [], "\n".join(problems)
    finally:
        browser.close()
    dlg = SettingsDialog(AppConfig(output_root=tmp_path))
    try:
        dlg.resize(760, 780)
        dlg.show()
        qapp.processEvents()
        problems = _geometry_problems(dlg, label="settings")
        assert problems == [], "\n".join(problems)
    finally:
        dlg.close()
