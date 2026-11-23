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
        assert [r.split(" — ")[0] for r in rows] == [
            full.name, transcribed.name, recorded.name]
        assert "Transcribed · Diarized · Speakers identified" in rows[0]
        assert "Transcribed" in rows[1] and "Diarized" not in rows[1]
        assert "not transcribed yet" in rows[2]
        # Default: the most recent session is loaded, not just shown.
        assert win.session_combo.currentData() == str(full)
        assert win.session_dir == full
        assert full.name in win.loaded_session_label.text()
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
        assert older.name in win.loaded_session_label.text()
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
        assert session.name in win.loaded_session_label.text()
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
        assert text.startswith("Loaded session: 2026-11-07_10-00-00_teams")
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
        assert win.session_dir.name in win.loaded_session_label.text()
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
            # Rows carry the human status line, not just flags.
            assert "Recorded · Transcribed" in \
                browser.session_list.item(0).text()
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
            assert older.name in win.loaded_session_label.text()
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
