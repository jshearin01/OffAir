"""UI layer for the Nov 21 2026 feature round (U4).

People pages, bookmarks UI, dictation entry, enterprise-cleanup UI,
and the nine UI improvements (tray, shortcuts, search by person,
provisional styling, onboarding, pins + week groups, follow-along,
theme toggle, disk line). Qt-free tests first (waveform loader,
colour palette, byte/week/onboarding helpers), then Qt-gated tests
in the gui-shots venv, offscreen.

Run them with::

    TMPDIR=~/workspace/.pytest-tmp QT_QPA_PLATFORM=offscreen \\
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \\
      -m pytest tests/test_ui_feature_round.py -q
"""
from __future__ import annotations

import json
import math
import os
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import AppConfig

REPO = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------ Qt-free -------
def test_load_rms_buckets_from_real_wav(tmp_path):
    """RMS buckets: 0.25 s buckets, duration, silence ~ 0."""
    pytest.importorskip("numpy")
    from app.ui.waveform import BUCKET_SECONDS, load_rms_buckets

    path = tmp_path / "tone.wav"
    rate = 48000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(rate))  # 1.0 s of tone
        wav.writeframes(frames)
    buckets, duration = load_rms_buckets(path)
    assert duration == pytest.approx(1.0, abs=0.01)
    assert len(buckets) == 4  # 1.0 s / 0.25 s
    assert all(0.0 < b <= 1.0 for b in buckets)
    assert BUCKET_SECONDS == 0.25


def test_load_rms_buckets_unreadable_returns_empty(tmp_path):
    from app.ui.waveform import load_rms_buckets

    buckets, duration = load_rms_buckets(tmp_path / "missing.wav")
    assert buckets == [] and duration == 0.0
    garbage = tmp_path / "garbage.wav"
    garbage.write_bytes(b"not a wav file")
    buckets, duration = load_rms_buckets(garbage)
    assert buckets == [] and duration == 0.0


def test_color_for_speaker_is_stable_and_palette_bounded():
    from app.ui.waveform import SPEAKER_COLORS, color_for_speaker

    assert color_for_speaker("Alice") == color_for_speaker("Alice")
    assert color_for_speaker("Alice") in SPEAKER_COLORS
    assert color_for_speaker("Bob") in SPEAKER_COLORS
    assert len(SPEAKER_COLORS) == 8


def test_format_bytes_units():
    from app.ui.main_window import _format_bytes

    assert _format_bytes(0) == "0 B"
    assert _format_bytes(512) == "512 B"
    assert _format_bytes(2048) == "2.0 KB"
    assert _format_bytes(3 * 1024 * 1024) == "3.0 MB"
    assert _format_bytes(int(1.5 * 1024 ** 3)) == "1.5 GB"


def test_week_header_for_this_week_and_older():
    from datetime import date, timedelta

    from app.ui.main_window import _week_header_for

    today = date.today()
    assert _week_header_for(_Summary(today)) == "This week"
    old = today - timedelta(days=21)
    monday = old - timedelta(days=old.weekday())
    assert _week_header_for(_Summary(old)) == (
        f"Week of {monday:%b %d, %Y}")
    assert _week_header_for(_Summary(None)) == "Other sessions"


class _Summary:
    """Minimal stand-in for SessionSummary (session_day needs a date)."""

    def __init__(self, day):
        self._day = day
        self.start_time = None if day is None else f"{day.isoformat()}T10:00:00"
        self.session_id = (
            "2026-11-07_10-00-00_teams" if day is None
            else f"{day:%Y-%m-%d}_10-00-00_teams")
        self.path = Path("/nonexistent")


def test_onboarding_marker_helpers_use_explicit_path(tmp_path):
    from app.ui.main_window import (
        _mark_onboarding_shown,
        _onboarding_shown,
    )

    marker = tmp_path / "onboarding_shown"
    assert not _onboarding_shown(marker)
    _mark_onboarding_shown(marker)
    assert _onboarding_shown(marker)
    assert marker.exists()


def test_onboarding_text_names_downloads_and_local_only():
    from app.ui.main_window import ONBOARDING_TEXT

    assert "~142 MB" in ONBOARDING_TEXT
    assert "~33 MB" in ONBOARDING_TEXT
    assert "stays on this machine" in ONBOARDING_TEXT


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


def _make_session(root: Path, name: str, *, speakers=(),
                  pinned: bool = False, start: str = "2026-11-07T10:00:00+00:00",
                  segments=None, with_audio: bool = True) -> Path:
    """A session folder with metadata + transcript.json (+ empty WAV)."""
    session = root / name
    session.mkdir(parents=True)
    metadata = {
        "session_id": name, "app_name": "Teams",
        "mode": "system-loopback-fallback", "status": "stopped",
        "start_time": start, "end_time": start,
        "duration_seconds": 61.3, "sample_rate": 48000, "channels": 1,
        "devices": {}, "warning": None, "files": {},
    }
    if pinned:
        metadata["pinned"] = True
    (session / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    if segments is None:
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello everyone.",
             "speaker": speakers[0] if speakers else "Speaker A"},
        ]
    (session / "transcript.json").write_text(json.dumps({
        "model": "base", "segments": segments}, indent=2), encoding="utf-8")
    if with_audio:
        (session / "combined.wav").write_bytes(b"")
    return session


def _make_window(qapp, tmp_path, **config_kwargs):
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path, **config_kwargs))
    win.show()
    qapp.processEvents()
    return win


def test_nav_stays_flat_eight_entries(qapp, tmp_path):
    from PyQt6.QtWidgets import QListWidget

    win = _make_window(qapp, tmp_path)
    try:
        nav = win.findChild(QListWidget, "navList")
        assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == [
            "Record", "Library", "Calendar", "Session", "People",
            "Settings", "MCP setup", "Help"]
        assert nav is not None
        assert nav.count() == 8
        assert not any("More" in nav.item(i).text()
                       for i in range(nav.count()))
    finally:
        win.close()


def test_dictation_toggle_hides_app_side(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    try:
        win.dictation_btn.setChecked(True)
        qapp.processEvents()
        assert not win.process_combo.isVisible()
        assert not win.app_picker_label.isVisible()
        assert not win.system_meter.isVisible()
        assert not win.system_level_label.isVisible()
        assert win.dictation_note_label.isVisible()
        assert "spoken side only" in win.dictation_note_label.text()
        assert win.mode_label.text() == "Dictation — mic only"
        win.dictation_btn.setChecked(False)
        qapp.processEvents()
        assert win.process_combo.isVisible()
        assert win.system_meter.isVisible()
        assert not win.dictation_note_label.isVisible()
    finally:
        win.close()


def test_person_page_lists_meetings_and_utterances(qapp, tmp_path):
    from PyQt6.QtCore import Qt

    _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                  speakers=["Alice"], with_audio=False,
                  segments=[
                      {"start": 0.0, "end": 5.0, "text": "Hello everyone.",
                       "speaker": "Alice"},
                      {"start": 5.0, "end": 9.0, "text": "Budget talk.",
                       "speaker": "Alice"},
                      {"start": 9.0, "end": 12.0, "text": "Not Alice.",
                       "speaker": "Bob"},
                  ])
    win = _make_window(qapp, tmp_path)
    try:
        win._show_person_page("Alice")
        qapp.processEvents()
        assert win.people_stack.currentIndex() == 1
        assert win.person_title_label.text() == "Alice"
        # Matching is display-name, case-insensitive exact.
        win._show_person_page("aLiCe")
        qapp.processEvents()
        assert win.person_utterances_list.count() == 2
        assert "2 utterance" in win.person_summary_label.text()
        assert win.person_sessions_list.count() == 1
        assert "2 utterance" in win.person_sessions_list.item(0).text()
        assert win.person_utterances_list.count() == 2
        assert "Hello everyone" in win.person_utterances_list.item(0).text()
        # No voice sample without audio: the button is disabled,
        # and the label says so in plain words.
        assert not win.person_sample_btn.isEnabled()
        assert "No voice sample" in win.person_sample_label.text()
        # No commitments UI anywhere on the page.
        texts = [win.person_title_label.text(),
                 win.person_summary_label.text()]
        assert not any("commitment" in t.lower() for t in texts)
        # Back returns to the list layer.
        win.person_back_btn.click()
        qapp.processEvents()
        assert win.people_stack.currentIndex() == 0
    finally:
        win.close()


def test_person_utterance_click_plays_via_play_audio(qapp, tmp_path, monkeypatch):
    _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                  speakers=["Alice"],
                  segments=[{"start": 3.0, "end": 7.0,
                             "text": "Play me.", "speaker": "Alice"}])
    win = _make_window(qapp, tmp_path)
    calls = []
    monkeypatch.setattr(win, "_play_audio",
                        lambda audio, start, end: calls.append((audio, start, end)))
    try:
        win._show_person_page("Alice")
        qapp.processEvents()
        item = win.person_utterances_list.item(0)
        win._on_person_utterance_clicked(item)
        qapp.processEvents()
        assert len(calls) == 1
        assert calls[0][1] == 3.0 and calls[0][2] == 7.0
        assert calls[0][0].name == "combined.wav"
    finally:
        win.close()


def test_bookmark_add_list_and_click_to_seek(qapp, tmp_path, monkeypatch):
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    win = _make_window(qapp, tmp_path)
    calls = []
    monkeypatch.setattr(win, "_play_audio",
                        lambda audio, start, end: calls.append((audio, start, end)))
    try:
        win.load_session(session)
        qapp.processEvents()
        assert "No bookmarks yet" in win.bookmarks_list.item(0).text()
        win._add_bookmark_at(session, 5.0, label="Decision")
        assert win.bookmarks_list.count() == 1
        assert "00:05" in win.bookmarks_list.item(0).text()
        assert "Decision" in win.bookmarks_list.item(0).text()
        win._on_bookmark_item_clicked(win.bookmarks_list.item(0))
        assert len(calls) == 1
        assert calls[0][1] == 5.0
        # The clicked bookmark highlights its segment.
        assert win.segments_list.currentRow() == 0
    finally:
        win.close()


def test_bookmark_with_deleted_audio_renders_disabled(qapp, tmp_path, monkeypatch):
    """Retention deleted the WAVs → the entry is disabled, with reason."""
    from app.storage.bookmarks import add_bookmark

    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    (session / "combined.wav").unlink()
    add_bookmark(session, 5.0, label="Gone")
    win = _make_window(qapp, tmp_path)
    try:
        win.load_session(session)
        qapp.processEvents()
        assert win.bookmarks_list.count() == 1
        item = win.bookmarks_list.item(0)
        assert "audio deleted" in item.text()
        from PyQt6.QtCore import Qt
        assert not (item.flags() & Qt.ItemFlag.ItemIsEnabled)
    finally:
        win.close()


def test_shortcuts_exist_with_expected_chords(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    try:
        assert win._bookmark_shortcut.key().toString() == "Ctrl+B"
        assert win._pause_shortcut.key().toString() == "Ctrl+P"
        assert win._search_shortcut.key().toString() == "Ctrl+K"
    finally:
        win.close()


def test_ctrl_k_jumps_to_library_search(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    try:
        win._on_jump_to_search()
        qapp.processEvents()
        assert win.tabs.tabText(win.tabs.currentIndex()) == "Library"
        assert win.library_search_edit.hasFocus()
    finally:
        win.close()


def test_library_pin_sort_star_prefix_and_toggle(qapp, tmp_path):
    first = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    _make_session(tmp_path, "2026-11-08_10-00-00_teams", pinned=True)
    win = _make_window(qapp, tmp_path)
    try:
        win.set_view("Library")
        qapp.processEvents()
        # Pinned session first (row 1, after the week header).
        assert "★" in win.library_list.item(1).text()
        assert "2026-11-08" in win.library_list.item(1).text()
        win.library_list.setCurrentRow(1)
        win._on_library_clicked(None)
        qapp.processEvents()
        assert win.library_star_btn.text() == "★ Unstar"
        win._on_library_star_toggle()
        qapp.processEvents()
        metadata = json.loads((tmp_path / "2026-11-08_10-00-00_teams"
                               / "metadata.json").read_text())
        assert metadata.get("pinned") is False
        # Unpinned, the newer session still leads (newest first)
        # but without the star.
        assert "★" not in win.library_list.item(1).text()
    finally:
        win.close()


def test_library_week_headers_only_unfiltered(qapp, tmp_path):
    _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                  start="2026-11-07T10:00:00+00:00")
    _make_session(tmp_path, "2026-10-24_10-00-00_teams",
                  start="2026-10-24T10:00:00+00:00")
    win = _make_window(qapp, tmp_path)
    try:
        win.set_view("Library")
        qapp.processEvents()
        headers = [win.library_list.item(i).text()
                   for i in range(win.library_list.count())
                   if win._library_summaries[i] is None]
        assert len(headers) == 2
        assert any(h.startswith("Week of") for h in headers)
        # A header row selects no session.
        win.library_list.setCurrentRow(0)
        assert win._selected_library_summary() is None
        assert not win.library_open_btn.isEnabled()
        # Search results stay flat: no header rows.
        win.library_search_edit.setText("Hello")
        qapp.processEvents()
        assert all(s is not None for s in win._library_summaries)
    finally:
        win.close()


def test_library_speaker_filter(qapp, tmp_path):
    _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                  speakers=["Alice"])
    _make_session(tmp_path, "2026-11-08_10-00-00_teams",
                  speakers=["Bob"],
                  segments=[{"start": 0.0, "end": 5.0,
                             "text": "Hello from Bob.", "speaker": "Bob"}])
    win = _make_window(qapp, tmp_path)
    try:
        win.set_view("Library")
        qapp.processEvents()
        assert "Alice" in [win.library_speaker_combo.itemText(i)
                           for i in range(win.library_speaker_combo.count())]
        win.library_speaker_combo.setCurrentIndex(
            win.library_speaker_combo.findData("Alice"))
        qapp.processEvents()
        summaries = [s for s in win._library_summaries if s is not None]
        assert len(summaries) == 1
        assert summaries[0].session_id.startswith("2026-11-07")
    finally:
        win.close()


def test_calendar_speaker_filter_dims_non_matching(qapp, tmp_path):
    from PyQt6.QtWidgets import QGraphicsOpacityEffect

    _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                  speakers=["Alice"])
    _make_session(tmp_path, "2026-11-08_10-00-00_teams",
                  speakers=["Bob"],
                  segments=[{"start": 0.0, "end": 5.0,
                             "text": "Hello from Bob.", "speaker": "Bob"}])
    win = _make_window(qapp, tmp_path)
    try:
        win.set_view("Calendar")
        qapp.processEvents()
        assert win.calendar_widget is not None
        win.calendar_widget.set_month(2026, 11)
        qapp.processEvents()
        win.calendar_speaker_combo.setCurrentIndex(
            win.calendar_speaker_combo.findData("Alice"))
        qapp.processEvents()
        win._apply_calendar_speaker_filter()
        seen = {}
        for badge in win.calendar_widget._badges:
            speakers = badge.entry.summary.speakers or []
            effect = badge.graphicsEffect()
            opacity = effect.opacity() if isinstance(
                effect, QGraphicsOpacityEffect) else 1.0
            seen[tuple(speakers)] = (opacity, badge.isEnabled())
        assert seen[("Alice",)] == (1.0, True)
        assert seen[("Bob",)] == (0.35, False)
    finally:
        win.close()


def test_live_items_are_provisional_styled(qapp, tmp_path):
    from PyQt6.QtGui import QColor

    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    win = _make_window(qapp, tmp_path)
    try:
        win.load_session(session)
        win._live_chunker = SimpleNamespace(
            model="base", chunks_done=1, segments=[object()],
            watermark_seconds=25.0,
            save_live=lambda _d: None)
        chunk = SimpleNamespace(segments=[
            SimpleNamespace(start=0.0, text="hello world", speaker="Speaker A"),
        ])
        win._on_live_chunks_finished([chunk])
        qapp.processEvents()
        assert win.nearlive_list.count() == 1
        item = win.nearlive_list.item(0)
        assert "(provisional)" in item.text()
        assert item.font().italic()
        assert item.foreground().color() == QColor("#6B7280")
        assert "provisional" in win.live_model_label.text()
        assert "Whisper base" in win.live_model_label.text()
    finally:
        win.close()


def test_theme_toggle_persists(qapp, tmp_path, monkeypatch):
    import app.config as config_mod
    import app.ui.main_window as mw

    saved = []
    fake_save = lambda cfg: saved.append(cfg)
    monkeypatch.setattr(mw, "save_settings", fake_save)
    monkeypatch.setattr(config_mod, "save_settings", fake_save)
    win = _make_window(qapp, tmp_path, theme="light")
    try:
        assert win.theme_toggle_btn.text() == "Dark theme"
        win._on_theme_toggle()
        qapp.processEvents()
        assert win.config.theme == "dark"
        assert saved and saved[-1].theme == "dark"
        assert win.theme_toggle_btn.text() == "Light theme"
    finally:
        win.close()


def test_disk_line_renders_in_library(qapp, tmp_path):
    _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    win = _make_window(qapp, tmp_path)
    try:
        win.set_view("Library")
        qapp.processEvents()
        assert "Free space:" in win.disk_status_label.text()
        assert "Retention would delete" in win.disk_status_label.text()
    finally:
        win.close()


def test_enterprise_cleanup_refuses_when_disabled(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    win = _make_window(qapp, tmp_path, enterprise_cleanup_enabled=False)
    shown = []
    monkeypatch.setattr(
        QMessageBox, "information",
        lambda *a, **k: shown.append(a) or QMessageBox.StandardButton.Ok)
    try:
        win.load_session(session)
        assert "leaves this machine" in win.enterprise_btn.text()
        win.on_enterprise_cleanup()
        assert shown, "expected the enable-in-Settings message"
        assert "Settings" in shown[0][2]
        assert not win._enterprise_running
    finally:
        win.close()


def test_enterprise_variant_renders_text_lines_without_timings(qapp, tmp_path):
    from app.transcription.enterprise_cleanup import ENTERPRISE_CLEAN_HEADER

    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    (session / "transcript-enterprise-clean.txt").write_text(
        ENTERPRISE_CLEAN_HEADER
        + "[00:00] Alice: Cleaned line one.\n"
        + "[00:05] Alice: Cleaned line two.\n", encoding="utf-8")
    win = _make_window(qapp, tmp_path)
    try:
        win.load_session(session)
        index = win.transcript_view_combo.findData("enterprise")
        assert index >= 0
        assert "Enterprise-cleaned" in win.transcript_view_combo.itemText(index)
        win.transcript_view_combo.setCurrentIndex(index)
        qapp.processEvents()
        assert win.segments_list.count() == 2
        assert "Cleaned line one" in win.segments_list.item(0).text()
        # No header lines leak into the list, and no timings are
        # attached to enterprise lines (there are none to attach).
        from PyQt6.QtCore import Qt

        assert "#" not in win.segments_list.item(0).text()
        assert win.segments_list.item(0).data(
            Qt.ItemDataRole.UserRole) is None
    finally:
        win.close()


def test_waveform_loads_and_click_seeks(qapp, tmp_path, monkeypatch):
    import numpy as np
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest

    pytest.importorskip("numpy")
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    rate = 48000
    samples = (12000 * np.sin(
        2 * np.pi * 440 * np.arange(rate) / rate)).astype(np.int16)
    with wave.open(str(session / "combined.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    win = _make_window(qapp, tmp_path)
    calls = []
    monkeypatch.setattr(win, "_play_audio",
                        lambda audio, start, end: calls.append((audio, start, end)))
    try:
        win.load_session(session)
        # Wait for the worker (off-thread RMS) to land in the cache.
        for _ in range(100):
            qapp.processEvents()
            if str(session) in win._waveform_cache:
                break
            QTest.qWait(50)
        assert str(session) in win._waveform_cache
        buckets, duration = win._waveform_cache[str(session)]
        assert duration == pytest.approx(1.0, abs=0.01)
        assert len(buckets) == 4
        widget = win.waveform_widget
        assert widget._buckets == buckets
        # Click the middle of the waveform: ~0.5 s in.
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton,
                         pos=QPoint(widget.width() // 2, widget.height() // 2))
        qapp.processEvents()
        assert len(calls) == 1
        assert calls[0][1] == pytest.approx(0.5, abs=0.05)
    finally:
        win.close()


def test_tray_degrades_silently_without_tray(qapp, tmp_path):
    from PyQt6.QtWidgets import QSystemTrayIcon

    win = _make_window(qapp, tmp_path)
    try:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            assert win._tray is None
        else:
            pytest.skip("a system tray is available in this environment")
    finally:
        win.close()


def test_geometry_clean_with_person_page_and_dictation(qapp, tmp_path):
    """The 1280x800 geometry audit, extended to the new layers."""
    try:
        from test_ui_visual import _geometry_problems
    except ImportError:
        pytest.skip("geometry audit helper unavailable")
    _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                  speakers=["Alice"])
    win = _make_window(qapp, tmp_path)
    try:
        win.dictation_btn.setChecked(True)
        qapp.processEvents()
        problems = _geometry_problems(win, label="record/dictation")
        win._show_person_page("Alice")
        qapp.processEvents()
        problems += _geometry_problems(win, label="person-page")
        assert problems == [], "\n".join(problems)
    finally:
        win.close()


def test_transport_buttons_are_not_compressed(qapp, tmp_path):
    """Regression: four transport buttons in one row compressed
    below their size hints and clipped their labels ("Paus",
    "ookmar") in the 380 px Record card. The Bookmark button
    moved to its own row; every transport button must render at
    or above its size hint so its label is fully visible."""
    win = _make_window(qapp, tmp_path)
    try:
        win.show_view("Record")
        qapp.processEvents()
        for btn in (win.pause_btn, win.resume_btn, win.stop_btn,
                    win.bookmark_btn):
            assert btn.width() >= btn.sizeHint().width(), (
                f"{btn.text()!r} compressed: width {btn.width()} "
                f"< sizeHint {btn.sizeHint().width()}")
    finally:
        win.close()
