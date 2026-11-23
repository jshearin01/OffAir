"""Recording source (Nov 10 2026) — detection, override, backfill,
filtering, and the edit path.

Pure tests run everywhere; the Qt tests (browser filter dropdown,
Record-card override, Process-tab editor) need PyQt6 and run with
the gui-shots venv::

    QT_QPA_PLATFORM=offscreen \\
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \\
      -m pytest tests/test_recording_source.py -q

The foreground-window-title capture for browser targets was
deliberately NOT built (see docs/RECORDING_SOURCE.md), so there
is no test for it — browser sessions carry the app name only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.storage.browser import (
    list_session_summaries,
    list_sources,
    search_sessions,
    summarize_session,
)
from app.storage.sessions import (
    create_session,
    load_metadata,
    set_session_source,
    update_metadata,
)
from app.storage.source import (
    SOURCES,
    SOURCE_LABELS,
    backfill_source,
    detect_source,
    normalize_source,
    resolve_recording_source,
    source_from_app_name,
    source_from_process,
)


def _raw_session(root: Path, name: str, *, app_name: str,
                 source: str | None = None) -> Path:
    """A session folder whose metadata.json is written by hand —
    i.e. a session recorded before the ``source`` field existed.
    ``source=None`` means the key is absent entirely.
    """
    session = root / name
    session.mkdir(parents=True)
    metadata = {
        "session_id": name, "app_name": app_name,
        "mode": "system-loopback-fallback", "status": "completed",
        "start_time": f"{name[:10]}T10:00:00",
        "end_time": f"{name[:10]}T10:01:01",
        "duration_seconds": 61.3, "sample_rate": 48000, "channels": 1,
        "devices": {}, "warning": None, "files": {}}
    if source is not None:
        metadata["source"] = source
    (session / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return session


# ------------------------------------------------------- process mapping ----
@pytest.mark.parametrize("exe,expected", [
    ("ms-teams.exe", "teams"),
    ("Teams.exe", "teams"),
    ("ms-teamsupdate.exe", "teams"),
    ("teams", "teams"),          # Linux client
    ("ms-teams", "teams"),       # Linux client
    ("MS-TEAMS.EXE", "teams"),   # case-insensitive
    ("zoom.exe", "zoom"),
    ("zoom.us", "zoom"),
    ("webex.exe", "webex"),
    ("atmgr.exe", "webex"),      # Webex Meetings manager, best-effort
    ("chrome.exe", "browser"),
    ("msedge.exe", "browser"),
    ("firefox.exe", "browser"),
    ("brave.exe", "browser"),
    ("opera.exe", "browser"),
])
def test_process_mapping_covers_every_detectable_source(exe, expected):
    assert source_from_process(exe) == expected
    source, detail = detect_source(exe)
    assert source == expected
    assert detail  # the friendly app name rides along as the detail


def test_process_mapping_unknown_exe_is_other_never_a_guess():
    assert source_from_process("notepad.exe") is None
    assert source_from_process("foobar.exe") is None
    source, detail = detect_source("foobar.exe")
    assert source == "other"
    assert detail == "Foobar"  # evidence kept, so it can be corrected


def test_no_target_app_is_solo():
    assert source_from_process(None) is None
    assert detect_source(None) == ("solo", "")
    assert detect_source("") == ("solo", "")


def test_meet_has_no_process_mapping_only_the_override():
    # Google Meet runs inside a browser: no exe identifies it.
    assert source_from_process("meet.exe") is None
    assert detect_source("meet.exe")[0] == "other"
    assert normalize_source("meet") == "meet"


@pytest.mark.parametrize("value", SOURCES)
def test_every_enum_value_is_a_valid_source(value):
    assert normalize_source(value) == value
    assert value in SOURCE_LABELS


def test_normalize_source_rejects_unknown_values():
    assert normalize_source("skype") is None
    assert normalize_source(None) is None
    assert normalize_source("") is None


# ------------------------------------------------------------- resolution ----
def test_manual_override_wins_over_detection():
    # The named case: a Claude Code session with Chrome picked —
    # Chrome is where the audio comes from, but the session is
    # a Claude Code session.
    source, detail = resolve_recording_source(
        override="claude-code", exe_name="chrome.exe")
    assert source == "claude-code"
    assert detail == "Google Chrome"


def test_manual_override_wins_even_when_detection_disagrees():
    source, _ = resolve_recording_source(
        override="teams", exe_name="chrome.exe")
    assert source == "teams"
    source, _ = resolve_recording_source(
        override="meet", exe_name="chrome.exe")
    assert source == "meet"


def test_override_with_no_target_app_still_wins():
    source, detail = resolve_recording_source(
        override="claude-code", exe_name=None)
    assert source == "claude-code"
    assert detail == ""


def test_no_override_means_detection():
    assert resolve_recording_source(exe_name="zoom.exe")[0] == "zoom"
    assert resolve_recording_source(exe_name=None)[0] == "solo"
    assert resolve_recording_source(exe_name="weird.exe")[0] == "other"


def test_invalid_override_raises():
    with pytest.raises(ValueError, match="unknown recording source"):
        resolve_recording_source(override="skype",
                                 exe_name="chrome.exe")


# ------------------------------------------------------------ create + edit --
def test_create_session_stores_source_and_detail(tmp_path):
    session = create_session(tmp_path, app_name="Zoom",
                             source="zoom",
                             source_detail="Zoom")
    metadata = load_metadata(session)
    assert metadata["source"] == "zoom"
    assert metadata["source_detail"] == "Zoom"


def test_create_session_defaults_to_other_never_a_guess(tmp_path):
    session = create_session(tmp_path, app_name="Teams")
    metadata = load_metadata(session)
    assert metadata["source"] == "other"
    assert metadata["source_detail"] == ""


def test_create_session_rejects_invalid_source(tmp_path):
    with pytest.raises(ValueError, match="unknown recording source"):
        create_session(tmp_path, source="skype")


def test_set_session_source_persists_to_metadata(tmp_path):
    session = create_session(tmp_path, app_name="Teams",
                             source="teams", source_detail="Microsoft Teams")
    updated = set_session_source(session, "claude-code",
                                 source_detail="Claude Code session")
    assert updated["source"] == "claude-code"
    assert updated["source_detail"] == "Claude Code session"
    assert load_metadata(session)["source"] == "claude-code"
    # Detail is optional: omitting it keeps the old detail.
    set_session_source(session, "zoom")
    assert load_metadata(session)["source"] == "zoom"
    assert load_metadata(session)["source_detail"] == "Claude Code session"


def test_set_session_source_rejects_invalid_source(tmp_path):
    session = create_session(tmp_path)
    with pytest.raises(ValueError, match="unknown recording source"):
        set_session_source(session, "skype")


# ------------------------------------------------------------------ backfill --
def test_backfill_derives_teams_from_app_name_and_writes_it(tmp_path):
    session = _raw_session(tmp_path, "2026-11-07_10-00-00_teams",
                           app_name="Microsoft Teams")
    assert "source" not in json.loads(
        (session / "metadata.json").read_text())
    metadata = load_metadata(session)
    assert metadata["source"] == "teams"
    assert metadata["source_detail"] == "Microsoft Teams"
    # Written back: a raw read sees it too.
    raw = json.loads((session / "metadata.json").read_text())
    assert raw["source"] == "teams"


@pytest.mark.parametrize("app_name,expected", [
    ("Microsoft Teams", "teams"),
    ("Teams", "teams"),
    ("Zoom", "zoom"),
    ("Webex", "webex"),
    ("Google Chrome", "browser"),
    ("Microsoft Edge", "browser"),
    ("Mozilla Firefox", "browser"),
    ("Google Meet", "meet"),
])
def test_backfill_app_name_mapping(tmp_path, app_name, expected):
    assert source_from_app_name(app_name) == expected
    session = _raw_session(tmp_path, "2026-11-07_10-00-00_x",
                           app_name=app_name)
    assert load_metadata(session)["source"] == expected


def test_backfill_unmappable_app_name_is_other(tmp_path):
    assert source_from_app_name("Spotify") is None
    assert source_from_app_name("") is None
    session = _raw_session(tmp_path, "2026-11-07_10-00-00_spotify",
                           app_name="Spotify")
    metadata = load_metadata(session)
    assert metadata["source"] == "other"
    assert metadata["source_detail"] == "Spotify"


def test_backfill_leaves_an_existing_source_alone(tmp_path):
    session = _raw_session(tmp_path, "2026-11-07_10-00-00_teams",
                           app_name="Microsoft Teams", source="zoom")
    assert load_metadata(session)["source"] == "zoom"
    assert backfill_source({"source": "zoom",
                            "app_name": "Microsoft Teams"}) is None


def test_backfill_is_idempotent(tmp_path):
    session = _raw_session(tmp_path, "2026-11-07_10-00-00_teams",
                           app_name="Microsoft Teams")
    first = load_metadata(session)
    second = load_metadata(session)
    assert first == second
    assert backfill_source(second) is None


def test_backfill_can_be_skipped_for_a_pure_read(tmp_path):
    session = _raw_session(tmp_path, "2026-11-07_10-00-00_teams",
                           app_name="Microsoft Teams")
    metadata = load_metadata(session, backfill=False)
    assert "source" not in metadata
    raw = json.loads((session / "metadata.json").read_text())
    assert "source" not in raw


# --------------------------------------------------------------- summaries ---
def test_summarize_session_exposes_source(tmp_path):
    session = create_session(tmp_path, app_name="Zoom", source="zoom",
                             source_detail="Zoom")
    summary = summarize_session(session)
    assert summary.source == "zoom"
    assert summary.source_detail == "Zoom"
    assert summary.source_label == "Zoom"
    assert "Zoom" in summary.display


def test_summarize_session_backfills_a_legacy_session(tmp_path):
    session = _raw_session(tmp_path, "2026-11-07_10-00-00_teams",
                           app_name="Microsoft Teams")
    summary = summarize_session(session)
    assert summary.source == "teams"
    assert "Teams" in summary.display


# ---------------------------------------------------------------- filtering --
def _library(tmp_path):
    teams = create_session(tmp_path, app_name="Microsoft Teams",
                           source="teams",
                           source_detail="Microsoft Teams")
    update_metadata(teams, duration_seconds=61.3)
    (teams / "transcript.txt").write_text(
        "The budget is twelve thousand dollars.", encoding="utf-8")
    browser = create_session(tmp_path, app_name="Google Chrome",
                             source="browser",
                             source_detail="Google Chrome")
    update_metadata(browser, duration_seconds=42.0)
    (browser / "transcript.txt").write_text(
        "The budget is twelve thousand dollars.", encoding="utf-8")
    solo = create_session(tmp_path, app_name="Teams", source="solo")
    update_metadata(solo, duration_seconds=10.0)
    return teams, browser, solo


def test_list_session_summaries_filters_by_source(tmp_path):
    teams, browser, solo = _library(tmp_path)
    assert [s.path for s in list_session_summaries(
        tmp_path, source="teams")] == [teams]
    assert [s.path for s in list_session_summaries(
        tmp_path, source="browser")] == [browser]
    assert [s.path for s in list_session_summaries(
        tmp_path, source="solo")] == [solo]
    assert len(list_session_summaries(tmp_path)) == 3
    assert list_session_summaries(tmp_path, source=None) is not None
    assert len(list_session_summaries(
        tmp_path, source=None)) == 3
    # A source with no sessions matches nothing, not everything.
    assert list_session_summaries(tmp_path, source="zoom") == []


def test_search_sessions_filters_by_source(tmp_path):
    teams, browser, _solo = _library(tmp_path)
    hits = search_sessions(tmp_path, "budget", source="teams")
    assert [h.path for h in hits] == [teams]
    hits = search_sessions(tmp_path, "budget", source="browser")
    assert [h.path for h in hits] == [browser]
    # Empty query + filter: the "return everything" path narrows too.
    assert [h.path for h in search_sessions(tmp_path, "",
                                            source="teams")] == [teams]
    assert len(search_sessions(tmp_path, "")) == 3


def test_list_sources_returns_only_sources_present_in_enum_order(tmp_path):
    _library(tmp_path)
    assert list_sources(tmp_path) == ["teams", "browser", "solo"]
    assert list_sources(tmp_path / "missing") == []


# ------------------------------------------------------------------ Qt UI ----
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


def _make_window(monkeypatch, tmp_path, sessions=None):
    """MainWindow with a mocked audio-session list (None = real
    call, which is [] on this Linux VM)."""
    import app.ui.main_window as mw

    if sessions is not None:
        monkeypatch.setattr(
            mw, "list_audio_sessions", lambda **kwargs: list(sessions))
    from app.ui.main_window import create_main_window
    from app.config import AppConfig

    return create_main_window(AppConfig(output_root=tmp_path))


class _FakeCapture:
    """A recording capture: start() succeeds, mode/warning recorded."""

    def __init__(self, session_dir, *, process_name="ms-teams.exe",
                 capture_system_mode="auto", **kwargs):
        from app.audio.capture import CaptureState

        self.session_dir = Path(session_dir)
        self.process_name = process_name
        self.capture_system_mode = capture_system_mode
        self.state = CaptureState.RECORDING
        self.mode = "system-loopback-fallback"
        self.warning = None
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        from app.audio.capture import CaptureState
        self.state = CaptureState.STOPPED
        return {}


def test_browser_source_filter_dropdown(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.browser import create_browser

    _library(tmp_path)
    browser = create_browser(AppConfig(output_root=tmp_path))
    try:
        browser.show()
        qapp.processEvents()
        assert browser.session_list.count() == 3
        options = [browser.source_combo.itemData(i)
                   for i in range(browser.source_combo.count())]
        assert options[0] is None  # "All sources"
        assert options[1:] == ["teams", "browser", "solo"]
        # Filter to Teams: only the Teams row remains.
        browser.source_combo.setCurrentIndex(
            browser.source_combo.findData("teams"))
        qapp.processEvents()
        assert browser.session_list.count() == 1
        assert "Teams" in browser.session_list.item(0).text()
        assert "source: Teams" in browser.status_label.text()
        # Back to All: everything returns.
        browser.source_combo.setCurrentIndex(0)
        qapp.processEvents()
        assert browser.session_list.count() == 3
    finally:
        browser.close()


def test_default_source_defaults_to_auto_and_dialog_lists_sources(
        qapp, tmp_path, monkeypatch):
    # Nov 20 2026: the Record view's Source dropdown is gone — the
    # default lives in the config and is edited in the Settings
    # dialog's "Default recording source" field.
    from app.ui.settings_dialog import SettingsDialog

    win = _make_window(monkeypatch, tmp_path)
    try:
        assert win.config.default_source is None
        assert not hasattr(win, "source_combo")
        dlg = SettingsDialog(win.config)
        try:
            assert dlg.default_source_combo.currentData() is None
            assert dlg.default_source_combo.itemText(0).startswith("Auto")
            values = [dlg.default_source_combo.itemData(i)
                      for i in range(dlg.default_source_combo.count())]
            assert values[1:] == list(SOURCES)
            dlg.default_source_combo.setCurrentIndex(
                dlg.default_source_combo.findData("claude-code"))
            out = dlg.config_from_dialog()
            assert out.default_source == "claude-code"
        finally:
            dlg.close()
    finally:
        win.close()


def test_on_record_stores_detected_source(qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw
    from app.audio.app_sessions import AudioSessionInfo

    monkeypatch.setattr(mw, "DualCapture", _FakeCapture)
    win = _make_window(monkeypatch, tmp_path, sessions=[
        AudioSessionInfo("chrome.exe", "Google Chrome", 100,
                         "active", 0.4),
        AudioSessionInfo("ms-teams.exe", "Microsoft Teams", 200,
                         "idle", 0.0),
    ])
    try:
        chrome = win.process_combo.findData("chrome.exe")
        win.process_combo.setCurrentIndex(chrome)
        qapp.processEvents()
        win.on_record()
        qapp.processEvents()
        metadata = load_metadata(win.session_dir)
        assert metadata["source"] == "browser"
        assert metadata["source_detail"] == "Google Chrome"
        assert "Source: Browser" in win.loaded_session_label.text()
    finally:
        win.close()


def test_on_record_override_wins_over_detection(qapp, tmp_path,
                                                monkeypatch):
    import app.ui.main_window as mw
    from app.audio.app_sessions import AudioSessionInfo

    monkeypatch.setattr(mw, "DualCapture", _FakeCapture)
    win = _make_window(monkeypatch, tmp_path, sessions=[
        AudioSessionInfo("chrome.exe", "Google Chrome", 100,
                         "active", 0.4),
    ])
    try:
        # Chrome picked, but the session is a Claude Code session:
        # the persisted default source wins over detection.
        win.process_combo.setCurrentIndex(
            win.process_combo.findData("chrome.exe"))
        win.config.default_source = "claude-code"
        qapp.processEvents()
        win.on_record()
        qapp.processEvents()
        metadata = load_metadata(win.session_dir)
        assert metadata["source"] == "claude-code"
        assert metadata["source_detail"] == "Google Chrome"
    finally:
        win.close()


def test_on_record_with_no_app_picked_is_solo(qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw

    monkeypatch.setattr(mw, "DualCapture", _FakeCapture)
    win = _make_window(monkeypatch, tmp_path, sessions=[])
    try:
        # No audio sessions: the picker has no target, so the
        # recording is solo — not Teams via the ms-teams.exe
        # capture fallback.
        assert win.selected_process_name() is None
        win.on_record()
        qapp.processEvents()
        assert load_metadata(win.session_dir)["source"] == "solo"
    finally:
        win.close()


def test_process_tab_source_editor_persists(qapp, tmp_path, monkeypatch):
    session = create_session(tmp_path, app_name="Microsoft Teams",
                             source="teams",
                             source_detail="Microsoft Teams")
    win = _make_window(monkeypatch, tmp_path)
    try:
        assert win.session_dir == session
        assert win.session_source_combo.currentData() == "teams"
        assert (win.session_source_detail_edit.text()
                == "Microsoft Teams")
        # Correct the source: the pick writes to metadata.json.
        win.session_source_combo.setCurrentIndex(
            win.session_source_combo.findData("claude-code"))
        qapp.processEvents()
        assert load_metadata(session)["source"] == "claude-code"
        assert "Source: Claude Code" in win.loaded_session_label.text()
        # Edit the detail: Enter / focus-out saves it.
        win.session_source_detail_edit.setText("Claude Code session")
        win.session_source_detail_edit.editingFinished.emit()
        qapp.processEvents()
        assert (load_metadata(session)["source_detail"]
                == "Claude Code session")
        assert load_metadata(session)["source"] == "claude-code"
    finally:
        win.close()
