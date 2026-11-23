"""Speakers tab: rename discoverability + voice samples (Qt-gated).

Run with the gui-shots venv (PyQt6 present)::

    QT_QPA_PLATFORM=offscreen \\
        ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \\
        -m pytest tests/test_ui_speakers.py -q
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURE_WAV = REPO / "tests" / "fixtures" / "meeting-3speakers-16k.wav"

SEGMENTS = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "hello"},
    {"start": 10.0, "end": 13.0, "speaker": "SPEAKER_00", "text": "again"},
    {"start": 30.0, "end": 38.0, "speaker": "SPEAKER_01", "text": "other"},
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


def _make_session(root: Path, name: str, *, with_audio: bool = True) -> Path:
    session = root / name
    session.mkdir(parents=True)
    (session / "transcript.json").write_text(
        json.dumps({"segments": SEGMENTS}), encoding="utf-8")
    if with_audio:
        shutil.copyfile(FIXTURE_WAV, session / "combined.wav")
    return session


def _seed_known_speaker(db_path: Path, name: str, session: str, label: str):
    from app.speakers.database import SpeakerDatabase
    db = SpeakerDatabase(db_path)
    try:
        db.add_embedding(name, np.array([1.0, 0.0, 0.0], dtype=np.float32),
                         backend="test-backend", source_session=session,
                         source_label=label)
    finally:
        db.close()


def _make_window(qapp, tmp_path):
    from app.config import AppConfig
    from app.ui.main_window import create_main_window
    return create_main_window(AppConfig(output_root=tmp_path))


def _label_texts(win) -> list[str]:
    from PyQt6.QtWidgets import QLabel
    return [child.text() for child in win.findChildren(QLabel)]


class _FakePlayer:
    """Stand-in for QMediaPlayer: records what would be played."""

    def __init__(self):
        self.source = None
        self.position = None
        self.played = False

    def setSource(self, url):
        self.source = url.toLocalFile()

    def setPosition(self, ms):
        self.position = ms

    def play(self):
        self.played = True

    def stop(self):
        self.played = False


# ------------------------------------------------------------------ controls ---
def test_speakers_tab_has_play_and_rename_controls(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    for name in ("play_known_btn", "rename_known_btn",
                 "play_suggestion_btn"):
        assert hasattr(win, name), name
    # Nothing selected yet: all three stay disabled, never a dead click.
    assert not win.play_known_btn.isEnabled()
    assert not win.rename_known_btn.isEnabled()
    assert not win.play_suggestion_btn.isEnabled()
    # The buttons say what they do; the hints say how to use them.
    assert "Play sample" in win.play_known_btn.text()
    assert "Rename" in win.rename_known_btn.text()
    assert "Play sample" in win.play_suggestion_btn.text()
    assert any("Play sample to hear" in t for t in _label_texts(win))
    assert any("hear their voice before confirming" in t
               for t in _label_texts(win))


def test_play_and_rename_stay_disabled_for_placeholder_rows(
        qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    # No speakers learned yet: the known list holds only the
    # placeholder row, which must never enable Rename / Play.
    win.known_speakers_list.setCurrentRow(0)
    win._update_buttons()
    assert not win.play_known_btn.isEnabled()
    assert not win.rename_known_btn.isEnabled()
    assert win._selected_known_speaker() is None


# -------------------------------------------------------------------- rename ---
def test_rename_known_speaker_prefills_the_panel(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    _seed_known_speaker(win._speakers_db_path(), "John",
                        session.name, "SPEAKER_00")
    win.session_dir = session
    win._refresh_rename_labels(["SPEAKER_00", "SPEAKER_01"])
    win._refresh_known_speakers()
    win.known_speakers_list.setCurrentRow(0)
    win._update_buttons()
    assert win.play_known_btn.isEnabled()
    assert win.rename_known_btn.isEnabled()

    win.on_rename_known_speaker()
    assert win.rename_label_combo.currentText() == "John"
    assert win.rename_name_edit.placeholderText() == "New name for John"
    assert "Renaming John" in win.status_label.text()


def test_rename_known_speaker_end_to_end(qapp, tmp_path):
    from app.speakers.database import SpeakerDatabase
    win = _make_window(qapp, tmp_path)
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    _seed_known_speaker(win._speakers_db_path(), "John",
                        session.name, "SPEAKER_00")
    win.session_dir = session
    win._refresh_rename_labels(["SPEAKER_00", "SPEAKER_01"])
    win._refresh_known_speakers()
    win.known_speakers_list.setCurrentRow(0)
    win.on_rename_known_speaker()

    win.rename_name_edit.setText("Priya")
    win.on_rename_speaker()
    db = SpeakerDatabase(win._speakers_db_path())
    try:
        assert db.get_speaker("Priya") is not None
        assert db.get_speaker("John") is None
    finally:
        db.close()
    assert "Priya" in win.known_speakers_list.item(0).text()
    assert "Renamed identity John" in win.status_label.text()


def test_double_click_known_speaker_starts_rename(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    _seed_known_speaker(win._speakers_db_path(), "John",
                        session.name, "SPEAKER_00")
    win.session_dir = session
    win._refresh_rename_labels(["SPEAKER_00", "SPEAKER_01"])
    win._refresh_known_speakers()
    item = win.known_speakers_list.item(0)
    win.known_speakers_list.setCurrentItem(item)
    win.known_speakers_list.itemDoubleClicked.emit(item)
    assert win.rename_label_combo.currentText() == "John"
    assert "Renaming John" in win.status_label.text()


# --------------------------------------------------------------- play sample ---
def test_play_suggestion_sample_uses_longest_turn(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    win.session_dir = session
    win._suggestion_objects = [
        SimpleNamespace(label="SPEAKER_00", suggested_name="John")]
    win.suggestions_list.addItem("SPEAKER_00: John (confidence 1.00)")
    win.suggestions_list.setCurrentRow(0)
    win._update_buttons()
    assert win.play_suggestion_btn.isEnabled()
    fake = _FakePlayer()
    win._player = fake

    win.on_play_suggestion_sample()
    assert fake.played
    # QUrl.toLocalFile() returns forward slashes on Windows; compare
    # as Paths so the assertion holds on every platform.
    assert Path(fake.source) == session / "combined.wav"
    assert fake.position == 0  # longest turn starts at 0 s
    assert "Playing a 5s sample" in win.status_label.text()


def test_play_suggestion_sample_reports_missing_audio(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            with_audio=False)
    win.session_dir = session
    win._suggestion_objects = [
        SimpleNamespace(label="SPEAKER_00", suggested_name="John")]
    win.suggestions_list.addItem("SPEAKER_00: John (confidence 1.00)")
    win.suggestions_list.setCurrentRow(0)
    win.on_play_suggestion_sample()
    assert "No voice sample" in win.status_label.text()


def test_play_known_sample_reports_missing_audio(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    # The speaker was learned from a session folder that is gone.
    _seed_known_speaker(win._speakers_db_path(), "John",
                        "2026-11-01_10-00-00_teams", "SPEAKER_00")
    win._refresh_known_speakers()
    win.known_speakers_list.setCurrentRow(0)
    win.on_play_known_sample()
    assert "No voice sample for John" in win.status_label.text()


def test_play_known_sample_plays_newest_session(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    _seed_known_speaker(win._speakers_db_path(), "John",
                        "2026-11-07_10-00-00_teams", "SPEAKER_01")
    win._refresh_known_speakers()
    win.known_speakers_list.setCurrentRow(0)
    fake = _FakePlayer()
    win._player = fake
    win.on_play_known_sample()
    assert fake.played
    assert fake.source.endswith("combined.wav")
    assert fake.position == 30_000  # SPEAKER_01's longest turn: 30–38 s
    assert "Playing a 8s sample of John" in win.status_label.text()


def test_double_click_suggestion_plays_sample(qapp, tmp_path):
    win = _make_window(qapp, tmp_path)
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams")
    win.session_dir = session
    win._suggestion_objects = [
        SimpleNamespace(label="SPEAKER_00", suggested_name="John")]
    win.suggestions_list.addItem("SPEAKER_00: John (confidence 1.00)")
    item = win.suggestions_list.item(0)
    win.suggestions_list.setCurrentItem(item)
    fake = _FakePlayer()
    win._player = fake
    win.suggestions_list.itemDoubleClicked.emit(item)
    assert fake.played
    assert "Playing a 5s sample" in win.status_label.text()
