"""Audio retention tests — auto-delete old WAVs, always keep transcripts.

Rules under test (see app/storage/retention.py):
* a session's WAVs are deleted only when it HAS a transcript AND the
  audio is older than the retention age;
* transcripts / diarization / speaker names / metadata are never
  deleted, and a whole session folder is never removed;
* a session still recording (metadata status "recording") is never
  touched, nor is a session passed in as excluded (the live one);
* retention None ("Never") disables pruning entirely.

Pure-function tests run anywhere; the Settings-dialog and main-window
wiring tests are Qt-gated (same offscreen pattern as
test_auto_process.py).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from app.config import (
    AppConfig,
    config_from_settings,
    load_settings,
    save_settings,
)
from app.storage.retention import (
    DISK_WARNING_FREE_BYTES,
    disk_status,
    prune_old_audio,
    scan_prunable_audio,
)

HOUR = 3600.0


def _make_session(root: Path, name: str, *, status: str = "completed",
                  transcript: bool = True, age_hours: float = 25.0,
                  now: float | None = None) -> Path:
    """A fake session folder with metadata + WAVs at a given age."""
    now = time.time() if now is None else now
    session = root / name
    session.mkdir(parents=True)
    metadata = {"session_id": name, "app_name": "Teams",
                "status": status,
                "start_time": "2026-11-08T10:00:00+00:00",
                "end_time": ("2026-11-08T11:00:00+00:00"
                             if status == "completed" else None),
                "duration_seconds": 3600.0 if status == "completed"
                else None}
    (session / "metadata.json").write_text(json.dumps(metadata),
                                           encoding="utf-8")
    if transcript:
        (session / "transcript.json").write_text(
            json.dumps({"segments": []}), encoding="utf-8")
        (session / "transcript.txt").write_text("hello\n", encoding="utf-8")
        (session / "transcript.srt").write_text("", encoding="utf-8")
        (session / "diarization.json").write_text("{}", encoding="utf-8")
        (session / "speakers.json").write_text("{}", encoding="utf-8")
    for wav in ("mic.wav", "system.wav", "combined.wav"):
        path = session / wav
        path.write_bytes(b"\x00\x00" * 48000)  # 1 s of silence, 96 KB
        mtime = now - age_hours * HOUR
        os.utime(path, (mtime, mtime))
    return session


def test_prunes_transcribed_old_audio(tmp_path):
    now = time.time()
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            now=now)
    result = prune_old_audio(tmp_path, 24.0, now=now)
    assert result == {"sessions_checked": 1, "sessions_pruned": 1,
                      "files_deleted": 3, "bytes_freed": 3 * 96000,
                      "errors": 0}
    for wav in ("mic.wav", "system.wav", "combined.wav"):
        assert not (session / wav).exists()
    # Transcripts, diarization, names, metadata all survive; the
    # folder itself is never removed.
    assert session.is_dir()
    for kept in ("transcript.json", "transcript.txt", "transcript.srt",
                 "diarization.json", "speakers.json", "metadata.json"):
        assert (session / kept).exists()


def test_keeps_untranscribed_audio(tmp_path):
    now = time.time()
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            transcript=False, now=now)
    result = prune_old_audio(tmp_path, 24.0, now=now)
    assert result["files_deleted"] == 0
    assert result["sessions_pruned"] == 0
    for wav in ("mic.wav", "system.wav", "combined.wav"):
        assert (session / wav).exists()


def test_keeps_recent_audio(tmp_path):
    now = time.time()
    session = _make_session(tmp_path, "2026-11-08_10-00-00_teams",
                            age_hours=1.0, now=now)
    result = prune_old_audio(tmp_path, 24.0, now=now)
    assert result["files_deleted"] == 0
    assert (session / "mic.wav").exists()


def test_never_disables_pruning(tmp_path):
    now = time.time()
    _make_session(tmp_path, "2026-11-07_10-00-00_teams", now=now)
    result = prune_old_audio(tmp_path, None, now=now)
    assert result == {"sessions_checked": 0, "sessions_pruned": 0,
                      "files_deleted": 0, "bytes_freed": 0, "errors": 0}


def test_custom_hours_respected(tmp_path):
    now = time.time()
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            age_hours=25.0, now=now)
    # 25 h old is inside a 7-day window…
    assert prune_old_audio(tmp_path, 168.0, now=now)["files_deleted"] == 0
    # …but outside a 1-hour window.
    result = prune_old_audio(tmp_path, 1.0, now=now)
    assert result["files_deleted"] == 3
    assert not (session / "mic.wav").exists()


def test_recording_session_untouched(tmp_path):
    now = time.time()
    session = _make_session(tmp_path, "2026-11-08_10-00-00_teams",
                            status="recording", age_hours=25.0, now=now)
    result = prune_old_audio(tmp_path, 24.0, now=now)
    assert result["files_deleted"] == 0
    for wav in ("mic.wav", "system.wav", "combined.wav"):
        assert (session / wav).exists()


def test_live_session_excluded(tmp_path):
    now = time.time()
    live = _make_session(tmp_path, "2026-11-08_10-00-00_teams",
                         age_hours=25.0, now=now)
    result = prune_old_audio(tmp_path, 24.0, now=now,
                             exclude_session_dirs=(live,))
    assert result["files_deleted"] == 0
    assert (live / "mic.wav").exists()


def test_prunes_only_old_files_in_a_session(tmp_path):
    now = time.time()
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            age_hours=25.0, now=now)
    recent = session / "system.wav"
    os.utime(recent, (now - HOUR, now - HOUR))
    result = prune_old_audio(tmp_path, 24.0, now=now)
    assert result["files_deleted"] == 2
    assert not (session / "mic.wav").exists()
    assert (session / "system.wav").exists()
    assert not (session / "combined.wav").exists()


def test_unreadable_metadata_skipped_not_crashed(tmp_path):
    now = time.time()
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            now=now)
    (session / "metadata.json").write_text("{not json", encoding="utf-8")
    result = prune_old_audio(tmp_path, 24.0, now=now)
    assert result["files_deleted"] == 0
    assert result["sessions_checked"] == 1
    assert (session / "mic.wav").exists()


def test_missing_or_empty_root_is_a_noop(tmp_path):
    assert prune_old_audio(tmp_path / "no-such-dir", 24.0)[
        "files_deleted"] == 0
    assert prune_old_audio(tmp_path, 24.0)["sessions_checked"] == 0


def test_invalid_retention_rejected():
    with pytest.raises(ValueError):
        prune_old_audio(Path("."), 0)
    with pytest.raises(ValueError):
        prune_old_audio(Path("."), -5.0)


# --------------------------------------------------------- disk warning ----
# The low-disk helpers are pure (shutil.disk_usage on the recordings
# volume): they only report. Nothing is ever stopped for disk reasons
# by this layer — the UI layer polls disk_status() and warns.

def test_disk_warning_threshold_is_one_gib():
    assert DISK_WARNING_FREE_BYTES == 1024 ** 3


def test_disk_status_reports_free_space(tmp_path):
    status = disk_status(tmp_path)
    assert status.total_bytes >= status.free_bytes > 0
    # used + free need not equal total (reserved blocks), but used
    # is a positive part of the volume.
    assert 0 < status.used_bytes <= status.total_bytes
    assert status.threshold_bytes == DISK_WARNING_FREE_BYTES
    # low_disk is derived, not assumed: on this VM /tmp is a 512 MB
    # tmpfs, so a tmp_path volume can genuinely be below 1 GiB free.
    assert status.low_disk == (status.free_bytes < status.threshold_bytes)


def test_disk_status_low_disk_predicate():
    # A threshold larger than any real volume is low; a tiny one is not.
    assert disk_status(Path("."), threshold_bytes=10 ** 30).low_disk is True
    assert disk_status(Path("."), threshold_bytes=1).low_disk is False


def test_disk_status_missing_subpath_uses_existing_ancestor(tmp_path):
    # A recordings root that does not exist yet is still on a volume.
    status = disk_status(tmp_path / "not-yet-created" / "deeper")
    assert status.free_bytes > 0
    assert status.path == str(tmp_path / "not-yet-created" / "deeper")


def test_disk_status_is_frozen_and_threshold_named():
    status = disk_status(Path("."))
    assert status.free_gb > 0
    with pytest.raises(AttributeError):
        status.free_bytes = 0


# ------------------------------------------------------------ dry-run scan -
def test_scan_prunable_audio_counts_without_deleting(tmp_path):
    now = time.time()
    session = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                            now=now)
    result = scan_prunable_audio(tmp_path, 24.0, now=now)
    assert result == {"sessions_checked": 1, "sessions_prunable": 1,
                      "files_prunable": 3, "bytes_prunable": 3 * 96000}
    # Dry run: nothing was deleted.
    for wav in ("mic.wav", "system.wav", "combined.wav"):
        assert (session / wav).exists()


def test_scan_prunable_audio_respects_the_same_rules(tmp_path):
    now = time.time()
    _make_session(tmp_path, "2026-11-08_10-00-00_teams",
                  age_hours=1.0, now=now)  # recent: not prunable
    _make_session(tmp_path, "2026-11-07_09-00-00_teams",
                  transcript=False, now=now)  # untranscribed: not prunable
    _make_session(tmp_path, "2026-11-07_08-00-00_teams",
                  status="recording", now=now)  # recording: not prunable
    result = scan_prunable_audio(tmp_path, 24.0, now=now)
    assert result["sessions_checked"] == 3
    assert result["files_prunable"] == 0
    assert result["bytes_prunable"] == 0


def test_scan_prunable_audio_never_and_invalid(tmp_path):
    now = time.time()
    _make_session(tmp_path, "2026-11-07_10-00-00_teams", now=now)
    assert scan_prunable_audio(tmp_path, None, now=now) == {
        "sessions_checked": 0, "sessions_prunable": 0,
        "files_prunable": 0, "bytes_prunable": 0}
    with pytest.raises(ValueError):
        scan_prunable_audio(tmp_path, 0, now=now)


def test_scan_prunable_audio_excludes_live_session(tmp_path):
    now = time.time()
    live = _make_session(tmp_path, "2026-11-07_10-00-00_teams",
                         now=now)
    result = scan_prunable_audio(tmp_path, 24.0, now=now,
                                 exclude_session_dirs=(live,))
    assert result["files_prunable"] == 0


def test_disk_warning_setting_round_trips(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(AppConfig(disk_warning_threshold_bytes=2 * 1024 ** 3),
                  path)
    assert (load_settings(path).disk_warning_threshold_bytes
            == 2 * 1024 ** 3)
    save_settings(AppConfig(), path)
    assert (load_settings(path).disk_warning_threshold_bytes
            == DISK_WARNING_FREE_BYTES)


def test_disk_warning_defaults_to_one_gib():
    assert AppConfig().disk_warning_threshold_bytes == DISK_WARNING_FREE_BYTES


def test_disk_warning_bad_values_fall_back_to_default():
    assert (config_from_settings(
        {"disk_warning_threshold_bytes": 0}).disk_warning_threshold_bytes
        == DISK_WARNING_FREE_BYTES)
    assert (config_from_settings(
        {"disk_warning_threshold_bytes": "junk"})
        .disk_warning_threshold_bytes == DISK_WARNING_FREE_BYTES)


def test_retention_setting_round_trips(tmp_path):
    path = tmp_path / "settings.json"
    for hours in (24.0, 168.0, 720.0, None):
        save_settings(AppConfig(audio_retention_hours=hours), path)
        assert load_settings(path).audio_retention_hours == hours


def test_retention_defaults_to_24_hours():
    assert AppConfig().audio_retention_hours == 24.0


def test_retention_bad_values_fall_back_to_default():
    assert (config_from_settings(
        {"audio_retention_hours": -5.0}).audio_retention_hours == 24.0)
    assert (config_from_settings(
        {"audio_retention_hours": "junk"}).audio_retention_hours == 24.0)
    assert (config_from_settings(
        {"audio_retention_hours": None}).audio_retention_hours is None)


# ------------------------------------------------------------- Qt wiring ---

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


def test_settings_dialog_retention_dropdown(qapp, tmp_path):
    from app.ui.settings_dialog import SettingsDialog

    dialog = SettingsDialog(AppConfig(audio_retention_hours=168.0))
    assert dialog.retention_combo.currentData() == "168"
    # Pick "Never" → config gets None; pick "30 days" → 720.0.
    dialog.retention_combo.setCurrentIndex(
        dialog.retention_combo.findData("never"))
    assert dialog.config_from_dialog().audio_retention_hours is None
    dialog.retention_combo.setCurrentIndex(
        dialog.retention_combo.findData("720"))
    assert dialog.config_from_dialog().audio_retention_hours == 720.0
    dialog.close()


def test_settings_dialog_retention_hint_text(qapp):
    from PyQt6.QtWidgets import QLabel

    from app.ui.settings_dialog import SettingsDialog

    dialog = SettingsDialog(AppConfig())
    hints = [w.text() for w in dialog.findChildren(QLabel)
             if "transcripts are always kept" in w.text()]
    assert len(hints) == 1
    dialog.close()


def test_main_window_prunes_on_startup_and_runs_hourly(qapp, tmp_path):
    from app.ui.main_window import create_main_window

    now = time.time()
    old = _make_session(tmp_path, "2026-11-07_10-00-00_teams", now=now)
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        # Startup prune ran in __init__: the old session's WAVs are gone.
        for wav in ("mic.wav", "system.wav", "combined.wav"):
            assert not (old / wav).exists()
        assert (old / "transcript.json").exists()
        # The hourly timer exists, fires hourly, and is running.
        assert win._retention_timer.interval() == 60 * 60 * 1000
        assert win._retention_timer.isActive()
    finally:
        win.close()


def test_main_window_prune_never_touches_live_session(qapp, tmp_path):
    from app.ui.main_window import create_main_window

    # Build the window FIRST (empty root, so the startup prune has
    # nothing to do), then add the old session and mark it live:
    # constructing the window first is the only way to test the
    # exclusion, since __init__ already runs a prune.
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        now = time.time()
        live = _make_session(tmp_path, "2026-11-08_10-00-00_teams",
                             age_hours=25.0, now=now)
        win.session_dir = live
        win._run_retention_prune()
        for wav in ("mic.wav", "system.wav", "combined.wav"):
            assert (live / wav).exists()
    finally:
        win.close()
