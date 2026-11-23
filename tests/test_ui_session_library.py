"""Session library: open a past session, Process-tab chooser (Qt-gated).

Part 2 of the UI/UX overhaul (Nov 10 2026): the Recording Browser
gains "Open in main window", the main window gains ``load_session``
plus a "Loaded session" line and a Process-tab session chooser, and
no UI string says "last session" any more — every action works on
the loaded session. Run with the gui-shots venv (PyQt6 present)::

    QT_QPA_PLATFORM=offscreen \
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \
      -m pytest tests/test_ui_session_library.py -q
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parent.parent

SEGMENTS = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00",
     "text": "Hello everyone."},
    {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01",
     "text": "The budget is twelve thousand."},
]


@pytest.fixture
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


def _make_session(root: Path, name: str, *, transcript: bool = False,
                  diarized: bool = False, suggestions: bool = False,
                  start_time: str | None = None) -> Path:
    """A session folder with metadata.json (so list_sessions sees it)."""
    session = root / name
    session.mkdir(parents=True)
    (session / "metadata.json").write_text(json.dumps({
        "session_id": name, "app_name": "Teams",
        "mode": "system-loopback-fallback", "status": "completed",
        "start_time": start_time or f"{name[:10]}T10:00:00",
        "end_time": f"{name[:10]}T10:01:01",
        "duration_seconds": 61.3, "sample_rate": 48000, "channels": 1,
        "devices": {}, "warning": None, "files": {}}, indent=2),
        encoding="utf-8")
    (session / "combined.wav").write_bytes(b"")
    if transcript:
        (session / "transcript.json").write_text(json.dumps({
            "model": "tiny", "segments": SEGMENTS}, indent=2),
            encoding="utf-8")
        (session / "transcript.txt").write_text(
            "[00:00] SPEAKER_00: Hello everyone.\n"
            "[00:05] SPEAKER_01: The budget is twelve thousand.\n",
            encoding="utf-8")
    if diarized:
        (session / "diarization.json").write_text(json.dumps({
            "mode": "sherpa", "method": "sherpa-onnx",
            "segments": SEGMENTS}, indent=2), encoding="utf-8")
    if suggestions:
        (session / "speaker_suggestions.json").write_text(json.dumps({
            "version": 1, "suggestions": [
                {"label": "SPEAKER_00", "suggested_name": "John",
                 "confidence": 0.92, "best_name": "John",
                 "best_score": 0.92, "second_best_name": None,
                 "second_best_score": 0.0, "margin": 0.3,
                 "ambiguous": False, "unknown": False,
                 "threshold": 0.55, "backend": "fallback",
                 "seconds": 5.0},
                {"label": "SPEAKER_01", "suggested_name": "Priya",
                 "confidence": 0.88, "best_name": "Priya",
                 "best_score": 0.88, "second_best_name": None,
                 "second_best_score": 0.0, "margin": 0.25,
                 "ambiguous": False, "unknown": False,
                 "threshold": 0.55, "backend": "fallback",
                 "seconds": 4.0},
            ]}, indent=2), encoding="utf-8")
    return session


def _make_window(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.main_window import create_main_window
    return create_main_window(AppConfig(output_root=tmp_path))


def _chooser_rows(win):
    return [win.session_combo.itemText(i)
            for i in range(win.session_combo.count())]


# ------------------------------------------------------------- chooser -------
def test_chooser_lists_newest_first_with_status_and_defaults_to_newest(
        qapp, tmp_path):
    recorded = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    transcribed = _make_session(
        tmp_path, "2026-11-08_10-00-00_teams", transcript=True)
    full = _make_session(
        tmp_path, "2026-11-09_10-00-00_teams", transcript=True,
        diarized=True, suggestions=True)
    win = _make_window(qapp, tmp_path)
    try:
        rows = _chooser_rows(win)
        # Rows lead with the display name (same for all three
        # Teams sessions here); order is newest-first by path.
        assert [r.split(" — ")[0] for r in rows] == [
            "Teams recording"] * 3
        assert [win.session_combo.itemData(i)
                for i in range(win.session_combo.count())] == [
            str(full), str(transcribed), str(recorded)]
        assert "Transcribed · Diarized · Speakers identified" in rows[0]
        assert "Transcribed" in rows[1] and "Diarized" not in rows[1]
        assert "not transcribed yet" in rows[2]
        # Default: the most recent session is loaded, not just shown.
        assert win.session_combo.currentData() == str(full)
        assert win.session_dir == full
        assert "Teams recording" in win.loaded_session_label.text()
        assert win.loaded_session_label.toolTip() == str(full)
        assert win.loaded_session_label.text().startswith(
            "Loaded session:")
    finally:
        win.close()


def test_chooser_pick_loads_session_and_buttons_follow_it(qapp, tmp_path):
    _make_session(tmp_path, "2026-11-08_10-00-00_teams", transcript=True)
    older = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    win = _make_window(qapp, tmp_path)
    try:
        # Startup loaded the newest (transcribed) session.
        assert win.diarize_btn.isEnabled()
        # Pick the recorded-only session: transcript empty state,
        # Transcribe on, Diarize/Suggest off.
        win.session_combo.setCurrentIndex(1)
        qapp.processEvents()
        assert win.session_dir == older
        assert "Teams recording" in win.loaded_session_label.text()
        assert win.loaded_session_label.toolTip() == str(older)
        assert "not transcribed yet" in win.loaded_session_label.text()
        assert win.segments_list.count() == 1
        assert "No transcript yet" in win.segments_list.item(0).text()
        assert win.transcribe_btn.isEnabled()
        assert not win.diarize_btn.isEnabled()
        assert not win.suggest_btn.isEnabled()
        assert not win.export_btn.isEnabled()
        # And back: the chooser and the loaded session stay in sync.
        win.session_combo.setCurrentIndex(0)
        qapp.processEvents()
        assert win.session_dir.name == "2026-11-08_10-00-00_teams"
        assert win.diarize_btn.isEnabled()
        assert win.export_btn.isEnabled()
    finally:
        win.close()


def test_process_action_runs_on_the_chosen_session(qapp, tmp_path,
                                                  monkeypatch):
    import app.ui.main_window as mw_mod
    newest = _make_session(tmp_path, "2026-11-09_10-00-00_teams",
                           transcript=True)
    older = _make_session(tmp_path, "2026-11-08_10-00-00_teams",
                          transcript=True)
    win = _make_window(qapp, tmp_path)
    try:
        win.session_combo.setCurrentIndex(1)
        qapp.processEvents()
        assert win.session_dir == older
        # Transcribe with a stub worker: the session it is handed
        # must be the chosen one, not the newest.
        captured: dict = {}

        def _fake_worker(session_dir, model_size):
            captured["session_dir"] = Path(session_dir)
            return MagicMock()

        monkeypatch.setattr(mw_mod, "_TranscribeWorker", _fake_worker)
        win.on_transcribe()
        assert captured["session_dir"] == older
        assert captured["session_dir"] != newest
        if win._transcribe_thread is not None:
            win._transcribe_thread.quit()
            win._transcribe_thread.wait(2000)
        # Export is the real code path: the LLM Markdown must land
        # in the chosen session's folder.
        win.export_btn.click()
        qapp.processEvents()
        assert (older / "transcript-llm.md").exists()
        assert not (newest / "transcript-llm.md").exists()
    finally:
        win.close()


# ---------------------------------------------------------- load_session -----
def test_load_session_with_no_transcript_opens_cleanly(qapp, tmp_path):
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    win = _make_window(qapp, tmp_path)
    try:
        # Startup already loaded it (only session); reload
        # explicitly to exercise load_session on the empty state.
        win.load_session(session)
        assert win.session_dir == session
        assert "Teams recording" in win.loaded_session_label.text()
        assert win.loaded_session_label.toolTip() == str(session)
        assert "not transcribed yet" in win.loaded_session_label.text()
        assert win.segments_list.count() == 1
        assert "No transcript yet" in win.segments_list.item(0).text()
        assert win.transcribe_btn.isEnabled()
        assert not win.diarize_btn.isEnabled()
        assert not win.suggest_btn.isEnabled()
        assert win.rename_label_combo.count() == 0
        assert not win.rename_btn.isEnabled()
    finally:
        win.close()


def test_load_session_with_transcript_but_no_diarization(qapp, tmp_path):
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            transcript=True)
    win = _make_window(qapp, tmp_path)
    try:
        win.load_session(session)
        assert win.segments_list.count() == 2
        assert "Hello everyone." in win.segments_list.item(0).text()
        assert "budget" in win.segments_list.item(1).text()
        assert win.diarize_btn.isEnabled()
        assert win.export_btn.isEnabled()
        assert not win.suggest_btn.isEnabled()
        assert [win.rename_label_combo.itemText(i)
                for i in range(win.rename_label_combo.count())] == [
                    "SPEAKER_00", "SPEAKER_01"]
        # No stored suggestions: the empty state, not a crash.
        assert win.suggestions_list.count() == 1
        assert "No suggestions yet" in win.suggestions_list.item(0).text()
    finally:
        win.close()


def test_load_session_fully_processed_shows_speakers(qapp, tmp_path):
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            transcript=True, diarized=True,
                            suggestions=True)
    win = _make_window(qapp, tmp_path)
    try:
        win.load_session(session)
        assert "Speakers identified" in win.loaded_session_label.text()
        assert win.suggest_btn.isEnabled()
        assert win.segments_list.count() == 2
        # Stored suggestions are shown as a read-back (names only,
        # no embeddings): Play works from the row's label, Confirm
        # stays disabled until Suggest regenerates live objects.
        assert win.suggestions_list.count() == 2
        assert "SPEAKER_00: John" in win.suggestions_list.item(0).text()
        assert "SPEAKER_01: Priya" in win.suggestions_list.item(1).text()
        win.suggestions_list.setCurrentRow(0)
        win._update_buttons()
        assert win.play_suggestion_btn.isEnabled()
        assert not win.confirm_btn.isEnabled()
        assert not win.correct_btn.isEnabled()
        assert [win.rename_label_combo.itemText(i)
                for i in range(win.rename_label_combo.count())] == [
                    "SPEAKER_00", "SPEAKER_01"]
    finally:
        win.close()


def test_load_session_missing_folder_raises(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    try:
        with pytest.raises(FileNotFoundError):
            win.load_session(tmp_path / "2026-11-07_10-00-00_nope")
    finally:
        win.close()


def test_loaded_session_label_shows_name_and_date(qapp, tmp_path):
    session = _make_session(
        tmp_path, "2026-11-07_10-00-00_teams", transcript=True,
        start_time="2026-11-07T10:00:00")
    win = _make_window(qapp, tmp_path)
    try:
        win.load_session(session)
        text = win.loaded_session_label.text()
        assert text.startswith(
            "Loaded session: Teams recording — Nov 07, 2026, 10:00")
        assert "Nov 07, 2026" in text
        assert "10:00" in text
        assert win.loaded_session_label.toolTip() == str(session)
    finally:
        win.close()


def test_empty_root_loads_nothing_and_chooser_says_so(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    try:
        assert win.session_dir is None
        assert win.session_combo.count() == 1
        assert win.session_combo.itemText(0) == "(no sessions yet)"
        assert "No session loaded" in win.loaded_session_label.text()
        assert not win.transcribe_btn.isEnabled()
    finally:
        win.close()


# -------------------------------------------------------------- browser ------
def test_browser_rows_are_tall_enough_for_the_status_line(qapp, tmp_path):
    """Wrapped browser rows are sized to their text (Nov 10 2026).

    Qt's item delegate under-measures wrapped rows: its size hint
    counted 3 lines for a row that paints 5, so the status line
    elided ("Speakers …"). The browser now sizes rows explicitly
    (see _row_height_for) — this pins the behaviour at the shown
    width: the demo-shaped row (3 display lines + 2 status lines)
    gets at least the 118 px a 101 px row proved too short for.
    """
    from app.config import AppConfig
    from app.ui.browser import create_browser

    _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                  transcript=True, diarized=True, suggestions=True)
    win = _make_window(qapp, tmp_path)
    try:
        browser = create_browser(AppConfig(output_root=tmp_path), parent=win)
        try:
            browser.resize(1100, 700)
            browser.show()
            qapp.processEvents()
            assert browser.session_list.count() == 1
            item = browser.session_list.item(0)
            assert item.sizeHint().height() == (
                browser._row_height_for(item.text()))
            assert item.sizeHint().height() >= 118
            assert browser.session_list.visualRect(
                browser.session_list.indexFromItem(item)).height() >= 118
        finally:
            browser.close()
    finally:
        win.close()


def test_record_start_clears_previous_sessions_views(qapp, tmp_path,
                                                     monkeypatch):
    """Starting a recording loads the fresh session's (empty) views.

    Before this, the Transcript tab kept showing the previous
    session's segments under the new session's name. on_record now
    loads the fresh session first (see load_session).
    """
    import app.ui.main_window as mw_mod
    from app.audio.capture import CaptureState

    old = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                        transcript=True)
    win = _make_window(qapp, tmp_path)
    try:
        assert win.session_dir == old
        assert win.segments_list.count() == 2
        fake = MagicMock()
        fake.mode = "loopback"
        fake.warning = ""
        fake.state = CaptureState.RECORDING
        monkeypatch.setattr(mw_mod, "DualCapture",
                            lambda *args, **kwargs: fake)
        win.on_record()
        qapp.processEvents()
        assert win.capture is fake
        assert win.session_dir != old
        assert win.session_dir.is_dir()
        assert win.segments_list.count() == 1
        assert "No transcript yet" in win.segments_list.item(0).text()
        assert win.session_dir.name in win.loaded_session_label.toolTip()
        assert "Microsoft Teams recording" in win.loaded_session_label.text()
        assert "not transcribed yet" in win.loaded_session_label.text()
    finally:
        win.close()


def test_browser_open_in_main_window_loads_session(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.browser import create_browser

    _make_session(tmp_path, "2026-11-08_10-00-00_teams", transcript=True)
    older = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                          transcript=True)
    win = _make_window(qapp, tmp_path)
    try:
        assert win.session_dir.name == "2026-11-08_10-00-00_teams"
        browser = create_browser(AppConfig(output_root=tmp_path), parent=win)
        try:
            browser.show()
            qapp.processEvents()
            assert browser.session_list.count() == 2
            # Rows carry the display name + date, then the meta line.
            row_text = browser.session_list.item(0).text()
            assert row_text.startswith("Teams recording — Nov 08, 2026")
            assert "SPEAKER_00" in row_text.split("\n")[1]
            # Select the older session and open it.
            browser.session_list.setCurrentRow(1)
            browser._on_session_clicked(None)
            qapp.processEvents()
            assert browser.open_btn.isEnabled()
            browser.open_btn.click()
            qapp.processEvents()
            # The browser closed; the main window stayed on the
            # opened session — transcript, chooser and label follow.
            assert not browser.isVisible()
            assert win.session_dir == older
            assert win.session_combo.currentData() == str(older)
            assert "Teams recording" in win.loaded_session_label.text()
            assert win.loaded_session_label.toolTip() == str(older)
            assert win.segments_list.count() == 2
        finally:
            browser.close()
    finally:
        win.close()


def test_browser_open_button_disabled_without_parent_or_selection(
        qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.browser import create_browser

    _make_session(tmp_path, "2026-11-07_10-00-00_teams", transcript=True)
    # Standalone browser: nowhere to open into.
    browser = create_browser(AppConfig(output_root=tmp_path))
    try:
        browser.show()
        qapp.processEvents()
        assert browser.session_list.count() == 1
        assert not browser.open_btn.isEnabled()
        browser.session_list.setCurrentRow(0)
        browser._on_session_clicked(None)
        qapp.processEvents()
        assert not browser.open_btn.isEnabled()
    finally:
        browser.close()
    # With a main-window parent: disabled until a session is picked.
    win = _make_window(qapp, tmp_path)
    try:
        browser = create_browser(AppConfig(output_root=tmp_path), parent=win)
        try:
            browser.show()
            qapp.processEvents()
            assert not browser.open_btn.isEnabled()
            browser.session_list.setCurrentRow(0)
            browser._on_session_clicked(None)
            qapp.processEvents()
            assert browser.open_btn.isEnabled()
        finally:
            browser.close()
    finally:
        win.close()


def test_browser_double_click_opens_session(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.browser import create_browser

    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            transcript=True)
    win = _make_window(qapp, tmp_path)
    try:
        # Point the window elsewhere first: load a fresh empty root
        # session by clearing the loaded one via a second window is
        # overkill — instead load the same session through the
        # double-click path from a *different* loaded state: pick
        # the row after moving the chooser to the empty placeholder
        # is impossible with one session, so assert the double-click
        # handler itself loads and closes.
        win.session_dir = None
        browser = create_browser(AppConfig(output_root=tmp_path), parent=win)
        try:
            browser.show()
            qapp.processEvents()
            item = browser.session_list.item(0)
            browser.session_list.setCurrentRow(0)
            browser.session_list.itemDoubleClicked.emit(item)
            qapp.processEvents()
            assert not browser.isVisible()
            assert win.session_dir == session
        finally:
            browser.close()
    finally:
        win.close()


# ------------------------------------------------------- no "last session" ---
def test_no_last_session_text_in_source_or_ui(qapp, tmp_path):
    # Source: the app never says "last session" any more — every
    # action works on the loaded session. (docs/archive keeps the
    # historical plans, and the tests name the rule itself; neither
    # is the app.)
    hits = []
    for path in sorted((REPO / "app").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "last session" in text.lower():
            hits.append(str(path.relative_to(REPO)))
    for rel in ("main.py", "README.md", "docs/SCREENSHOTS.md"):
        text = (REPO / rel).read_text(encoding="utf-8")
        if "last session" in text.lower():
            hits.append(rel)
    assert hits == []
    # Constructed UI: buttons, labels, tooltips, placeholders.
    _make_session(tmp_path, "2026-11-07_10-00-00_teams", transcript=True)
    win = _make_window(qapp, tmp_path)
    try:
        from PyQt6.QtWidgets import QLabel, QPushButton, QComboBox
        strings = []
        for child in win.findChildren(QLabel):
            strings.append(child.text())
            strings.append(child.toolTip())
        for child in win.findChildren(QPushButton):
            strings.append(child.text())
            strings.append(child.toolTip())
        for child in win.findChildren(QComboBox):
            strings.append(child.toolTip())
            for i in range(child.count()):
                strings.append(child.itemText(i))
        bad = [s for s in strings if "last session" in s.lower()]
        assert bad == []
        for button in (win.transcribe_btn, win.diarize_btn,
                       win.suggest_btn, win.nearlive_btn, win.export_btn,
                       win.summarize_btn):
            assert "last session" not in button.text().lower()
            assert "session" in button.text().lower()
    finally:
        win.close()


# ------------------------------------------------- delete / rename (Nov 23) ---

def _show_library(win, qapp):
    win.show_view("Library")
    qapp.processEvents()


def _select_library_row(win, row):
    win.library_list.setCurrentRow(row)
    win._on_library_clicked(None)


def _fake_message_box(monkeypatch, answer):
    """Replace QMessageBox in main_window with a recording fake."""
    import app.ui.main_window as mw_mod
    from PyQt6.QtWidgets import QMessageBox as RealBox

    calls = {"question": [], "warning": []}

    class FakeBox:
        StandardButton = RealBox.StandardButton

        @staticmethod
        def question(*args, **kwargs):
            calls["question"].append(args)
            return answer

        @staticmethod
        def warning(*args, **kwargs):
            calls["warning"].append(args)
            return RealBox.StandardButton.Ok

    monkeypatch.setattr(mw_mod, "QMessageBox", FakeBox)
    return calls


def test_library_rows_show_display_name_date_and_meta(qapp, tmp_path):
    _make_session(tmp_path, "2026-11-07_10-00-00_teams", transcript=True)
    win = _make_window(qapp, tmp_path)
    try:
        _show_library(win, qapp)
        assert win.library_list.count() == 2  # header + session
        text = win.library_list.item(1).text()
        title, meta = text.split("\n")
        assert title == "Teams recording — Nov 07, 2026, 10:00"
        assert "1m 01s" in meta and "SPEAKER_00" in meta
        assert "Teams" in meta
        assert win.library_list.item(0).text() == "Week of Nov 02, 2026"
    finally:
        win.close()


def test_library_row_delegate_sizes_two_line_rows(qapp, tmp_path):
    from PyQt6.QtGui import QPainter, QPixmap
    from PyQt6.QtWidgets import QStyleOptionViewItem

    _make_session(tmp_path, "2026-11-07_10-00-00_teams", transcript=True)
    win = _make_window(qapp, tmp_path)
    try:
        _show_library(win, qapp)
        delegate = win.library_list.itemDelegate()
        model = win.library_list.model()
        header_hint = delegate.sizeHint(
            QStyleOptionViewItem(), model.index(0, 0))
        row_hint = delegate.sizeHint(
            QStyleOptionViewItem(), model.index(1, 0))
        assert row_hint.height() > header_hint.height()
        assert row_hint.height() >= 48
        # Painting a row offscreen must not crash.
        pixmap = QPixmap(600, row_hint.height())
        painter = QPainter(pixmap)
        option = QStyleOptionViewItem()
        option.rect.setRect(0, 0, 600, row_hint.height())
        delegate.paint(painter, option, model.index(1, 0))
        painter.end()
        assert not pixmap.isNull()
    finally:
        win.close()


def test_library_delete_confirmed_removes_session_and_unloads_it(
        qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox as RealBox

    doomed = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                           transcript=True)
    _make_session(tmp_path, "2026-11-08_10-00-00_teams")
    calls = _fake_message_box(monkeypatch, RealBox.StandardButton.Yes)
    win = _make_window(qapp, tmp_path)
    try:
        _show_library(win, qapp)
        # The newest session (Nov 8) is loaded at startup; delete it.
        newest = tmp_path / "2026-11-08_10-00-00_teams"
        assert win.session_dir == newest
        _select_library_row(win, 1)  # header is row 0, newest first
        assert win.library_delete_btn.isEnabled()
        win.library_delete_btn.click()
        qapp.processEvents()
        assert len(calls["question"]) == 1
        assert "Teams recording" in calls["question"][0][2]
        assert "permanently removed" in calls["question"][0][2]
        assert not newest.exists()
        assert doomed.exists()
        assert win.session_dir is None
        assert "No session loaded" in win.loaded_session_label.text()
        assert win.library_list.count() == 2  # header + remaining
        assert "Deleted session" in win.statusBar().currentMessage()
    finally:
        win.close()


def test_library_delete_cancelled_keeps_session(
        qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox as RealBox

    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    _fake_message_box(monkeypatch, RealBox.StandardButton.No)
    win = _make_window(qapp, tmp_path)
    try:
        _show_library(win, qapp)
        _select_library_row(win, 1)
        win.library_delete_btn.click()
        qapp.processEvents()
        assert session.exists()
        assert win.session_dir == session
        assert win.library_list.count() == 2
    finally:
        win.close()


def test_library_delete_live_session_is_refused(
        qapp, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from PyQt6.QtWidgets import QMessageBox as RealBox

    from app.audio.capture import CaptureState

    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    calls = _fake_message_box(monkeypatch, RealBox.StandardButton.Yes)
    win = _make_window(qapp, tmp_path)
    original_capture = win.capture
    try:
        _show_library(win, qapp)
        _select_library_row(win, 1)
        win.capture = SimpleNamespace(state=CaptureState.RECORDING)
        win.library_delete_btn.click()
        qapp.processEvents()
        assert len(calls["warning"]) == 1
        assert len(calls["question"]) == 0
        assert session.exists()
    finally:
        win.capture = original_capture
        win.close()


def test_library_delete_other_session_keeps_loaded_session(
        qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox as RealBox

    older = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    newer = _make_session(tmp_path, "2026-11-08_10-00-00_teams")
    _fake_message_box(monkeypatch, RealBox.StandardButton.Yes)
    win = _make_window(qapp, tmp_path)
    try:
        _show_library(win, qapp)
        assert win.session_dir == newer
        _select_library_row(win, 2)  # older session
        win.library_delete_btn.click()
        qapp.processEvents()
        assert not older.exists()
        assert win.session_dir == newer
        assert "Teams recording" in win.loaded_session_label.text()
    finally:
        win.close()


def test_library_delete_with_search_filter_active(
        qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox as RealBox

    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            transcript=True)
    _make_session(tmp_path, "2026-11-08_10-00-00_teams")
    _fake_message_box(monkeypatch, RealBox.StandardButton.Yes)
    win = _make_window(qapp, tmp_path)
    try:
        _show_library(win, qapp)
        win.library_search_edit.setText("budget")
        qapp.processEvents()
        assert win.library_list.count() == 1
        _select_library_row(win, 0)
        win.library_delete_btn.click()
        qapp.processEvents()
        assert not session.exists()
        assert win.library_search_edit.text() == "budget"
        assert win.library_list.count() == 0
    finally:
        win.close()


def test_library_rename_persists_and_updates_views(
        qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw_mod

    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            transcript=True)

    class FakeDialog:
        @staticmethod
        def getText(*args, **kwargs):
            return "Chat with Priya", True

    monkeypatch.setattr(mw_mod, "QInputDialog", FakeDialog)
    win = _make_window(qapp, tmp_path)
    try:
        _show_library(win, qapp)
        _select_library_row(win, 1)
        win.library_rename_btn.click()
        qapp.processEvents()
        from app.storage.sessions import load_metadata
        assert load_metadata(session)["custom_name"] == "Chat with Priya"
        assert "Chat with Priya" in win.library_list.item(1).text()
        assert "Chat with Priya" in win.loaded_session_label.text()
        assert win.session_combo.itemText(0).startswith("Chat with Priya")
    finally:
        win.close()


def test_library_rename_disabled_for_outlook_linked(
        qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw_mod

    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    from app.storage.sessions import update_metadata
    update_metadata(session, calendar_event={
        "subject": "Budget review",
        "start": "2026-11-07T10:00:00+00:00",
        "end": "2026-11-07T11:00:00+00:00", "attendees": [],
        "organiser": "John", "entry_id": "abc"})
    called = []

    class FakeDialog:
        @staticmethod
        def getText(*args, **kwargs):
            called.append(True)
            return "New name", True

    monkeypatch.setattr(mw_mod, "QInputDialog", FakeDialog)
    win = _make_window(qapp, tmp_path)
    try:
        _show_library(win, qapp)
        _select_library_row(win, 1)
        assert not win.library_rename_btn.isEnabled()
        assert win.library_rename_btn.toolTip() == (
            "Named by its Outlook calendar event")
        assert "Budget review" in win.library_list.item(1).text()
        win._on_library_rename()
        assert called == []
    finally:
        win.close()


def test_library_rename_blank_is_refused(
        qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw_mod

    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")

    class FakeDialog:
        @staticmethod
        def getText(*args, **kwargs):
            return "   ", True

    monkeypatch.setattr(mw_mod, "QInputDialog", FakeDialog)
    from PyQt6.QtWidgets import QMessageBox as RealBox

    calls = _fake_message_box(monkeypatch, RealBox.StandardButton.Yes)
    win = _make_window(qapp, tmp_path)
    try:
        _show_library(win, qapp)
        _select_library_row(win, 1)
        win.library_rename_btn.click()
        qapp.processEvents()
        assert len(calls["warning"]) == 1
        from app.storage.sessions import load_metadata
        assert "custom_name" not in load_metadata(session)
    finally:
        win.close()


def test_library_delete_shortcut_exists(qapp, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeySequence

    _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    win = _make_window(qapp, tmp_path)
    try:
        shortcut = win.library_delete_shortcut
        assert shortcut.key().toString() == QKeySequence(
            Qt.Key.Key_Delete).toString()
        assert shortcut.context() == (
            Qt.ShortcutContext.WidgetWithChildrenShortcut)
    finally:
        win.close()


def test_library_context_menu_selects_row(qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw_mod

    _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    _make_session(tmp_path, "2026-11-08_10-00-00_teams")

    class FakeMenu:
        instances = []

        def __init__(self, *args, **kwargs):
            self.actions = []
            FakeMenu.instances.append(self)

        def addAction(self, text):
            action = text
            self.actions.append(action)
            return action

        def exec(self, *args, **kwargs):
            return None

    monkeypatch.setattr(mw_mod, "QMenu", FakeMenu)
    win = _make_window(qapp, tmp_path)
    try:
        _show_library(win, qapp)
        rect = win.library_list.visualItemRect(
            win.library_list.item(2))
        win._on_library_context_menu(rect.center())
        assert win.library_list.currentRow() == 2
        assert FakeMenu.instances[-1].actions == ["Delete session…"]
        assert win.library_delete_btn.isEnabled()
    finally:
        win.close()


def test_library_action_buttons_disabled_with_no_selection(
        qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    try:
        _show_library(win, qapp)
        assert win.library_list.count() == 0
        assert not win.library_open_btn.isEnabled()
        assert not win.library_star_btn.isEnabled()
        assert not win.library_rename_btn.isEnabled()
        assert not win.library_delete_btn.isEnabled()
    finally:
        win.close()
