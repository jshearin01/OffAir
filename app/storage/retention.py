"""Audio retention: auto-delete old recording audio, keep transcripts.

Recordings are WAVs, not MP3s — each session folder holds ``mic.wav``,
``system.wav`` and ``combined.wav`` (48 kHz mono int16, ~5.7 MB/min per
track, so ~1 GB per hour of meeting across the three tracks). Left
alone they accumulate forever, so OffAir prunes them:

* a session's WAVs are deleted only when BOTH hold: the session has a
  transcript (``transcript.json`` exists — a failed or unfinished
  transcription must stay retryable, its audio is never deleted) AND
  every WAV is older than the retention age (file mtime);
* never deleted: ``transcript.json``/``.txt``/``.srt``,
  ``diarization.json``, ``speakers.json``, ``speaker_suggestions.json``,
  ``metadata.json``, ``speakers.db``, or anything else — pruning only
  ever removes the three WAV files, never a whole session folder;
* a session whose ``metadata.json`` still says ``status: "recording"``
  (or has no readable metadata at all) is never touched — a crashed
  session keeps its audio, because it may be the only copy of a
  meeting that never got transcribed. The main window additionally
  excludes its own live session (see ``MainWindow._run_retention_prune``).

``retention_hours=None`` disables pruning entirely ("Never" in
Settings). This module is pure (no Qt) so the rules are unit-testable.

Disk warning (Nov 21 2026): :func:`disk_status` reports free space on
the recordings volume for the UI layer to poll while recording — the
threshold is :data:`DISK_WARNING_FREE_BYTES` (1 GiB free). The helper
only reports; nothing is ever stopped for disk reasons by this
module. :func:`scan_prunable_audio` is the dry run of
:func:`prune_old_audio`: the bytes retention *would* delete on the
next prune, with nothing deleted.
"""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from app.storage.sessions import (
    COMBINED_FILE,
    METADATA_FILE,
    MIC_FILE,
    SYSTEM_FILE,
    list_sessions,
    load_metadata,
)

logger = logging.getLogger("offair.storage.retention")

# Must match app.transcription.engine.TRANSCRIPT_JSON; kept local so
# this module never imports the transcription stack.
TRANSCRIPT_JSON = "transcript.json"

# The only files pruning ever deletes. Everything else in a session
# folder survives by construction, whatever its age.
AUDIO_FILES: tuple[str, ...] = (MIC_FILE, SYSTEM_FILE, COMBINED_FILE)

# Low-disk warning threshold (Nov 21 2026): warn while recording when
# free space on the recordings volume drops below 1 GiB. One hour of
# meeting is ~1 GB of WAV across the three tracks, so 1 GiB free is
# roughly one more hour of recording headroom. The threshold is a
# warning line only — a recording is never stopped for disk reasons.
# The persisted setting (app.config disk_warning_threshold_bytes)
# defaults to this value; keep the two equal (tests pin it).
DISK_WARNING_FREE_BYTES: int = 1024 ** 3


@dataclass(frozen=True)
class DiskStatus:
    """Free-space snapshot for one volume, for the UI layer to poll.

    ``path`` is the path asked about (not the resolved ancestor);
    ``low_disk`` is True when ``free_bytes`` is below
    ``threshold_bytes``. Frozen: a snapshot never changes under the
    caller — poll again for a fresh one.
    """

    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    threshold_bytes: int = DISK_WARNING_FREE_BYTES

    @property
    def low_disk(self) -> bool:
        """True when free space has dropped below the warning threshold."""
        return self.free_bytes < self.threshold_bytes

    @property
    def free_gb(self) -> float:
        """Free space in GiB, for display."""
        return self.free_bytes / (1024 ** 3)


def _existing_ancestor(path: Path) -> Path:
    """The path itself, or the nearest ancestor that exists.

    A recordings root that has never been created yet is still on a
    volume — ``shutil.disk_usage`` on the missing path itself would
    raise, so walk up to the first existing directory (same volume
    for any path under it).
    """
    candidate = Path(path)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:  # pragma: no cover — filesystem root
            break
        candidate = parent
    return candidate


def disk_status(path: str | Path, *,
                threshold_bytes: int = DISK_WARNING_FREE_BYTES,
                ) -> DiskStatus:
    """Free space on the volume holding ``path`` (pure, pollable).

    ``path`` may be a recordings root that does not exist yet (the
    volume of its nearest existing ancestor is reported). Raises
    ``OSError`` only when no ancestor exists at all. Never deletes
    or stops anything — the caller decides what to do with a
    ``low_disk`` result (warn once per session; never stop a
    recording silently for disk reasons).
    """
    if threshold_bytes < 0:
        raise ValueError("threshold_bytes must not be negative")
    usage = shutil.disk_usage(_existing_ancestor(Path(path)))
    return DiskStatus(
        path=str(path), total_bytes=usage.total,
        used_bytes=usage.used, free_bytes=usage.free,
        threshold_bytes=int(threshold_bytes))


def _cutoff_for(retention_hours: float | None,
                now: float | None) -> float | None:
    """Epoch cutoff for ``retention_hours`` (None = pruning disabled)."""
    if retention_hours is None:
        return None
    if retention_hours <= 0:
        raise ValueError("retention_hours must be positive or None")
    now_epoch = time.time() if now is None else float(now)
    return now_epoch - float(retention_hours) * 3600.0


def _prunable_files(session_dir: Path, cutoff: float,
                    excluded: set[Path]) -> list[tuple[Path, int]]:
    """(path, size) of the WAVs pruning would delete in one session.

    The shared eligibility rule for :func:`prune_old_audio` and
    :func:`scan_prunable_audio`: a session qualifies only when its
    metadata says ``"completed"`` and it has a transcript, and only
    its WAVs older than ``cutoff`` are listed. Sessions with
    unreadable metadata, still-recording sessions, untranscribed
    sessions, and excluded (live) sessions yield nothing.
    """
    if session_dir.resolve() in excluded:
        return []
    try:
        metadata = load_metadata(session_dir)
    except Exception:  # noqa: BLE001 — unreadable metadata: skip
        logger.debug("Skipping %s: metadata unreadable", session_dir)
        return []
    # Still recording (or crashed mid-recording): never touch it.
    if metadata.get("status") != "completed":
        return []
    # No transcript yet: the audio is the only way to retry.
    if not (session_dir / TRANSCRIPT_JSON).exists():
        return []
    prunable: list[tuple[Path, int]] = []
    for name in AUDIO_FILES:
        path = session_dir / name
        try:
            if not path.exists():
                continue
            stat = path.stat()
        except OSError:
            logger.debug("Skipping %s: cannot stat", path)
            continue
        if stat.st_mtime < cutoff:
            prunable.append((path, stat.st_size))
    return prunable


def prune_old_audio(output_root: str | Path,
                    retention_hours: float | None,
                    *, now: float | None = None,
                    exclude_session_dirs=()) -> dict:
    """Delete old transcribed sessions' WAVs under ``output_root``.

    ``retention_hours=None`` disables pruning (returns all zeros).
    ``now`` is epoch seconds (defaults to ``time.time()`` — injectable
    for tests). ``exclude_session_dirs`` are session folders to skip
    even if they otherwise qualify (the live recording).

    Returns a dict with ``sessions_checked``, ``sessions_pruned``
    (sessions where at least one file was deleted), ``files_deleted``,
    ``bytes_freed`` and ``errors`` (files that could not be deleted).
    A run that deletes nothing logs at debug level only.
    """
    result = {"sessions_checked": 0, "sessions_pruned": 0,
              "files_deleted": 0, "bytes_freed": 0, "errors": 0}
    cutoff = _cutoff_for(retention_hours, now)
    if cutoff is None:
        return result
    excluded = {Path(p).resolve() for p in exclude_session_dirs}
    for session_dir in list_sessions(output_root):
        result["sessions_checked"] += 1
        pruned_here = 0
        for path, size in _prunable_files(session_dir, cutoff, excluded):
            try:
                path.unlink()
            except OSError:
                result["errors"] += 1
                logger.warning("Could not delete old audio %s", path,
                               exc_info=True)
                continue
            result["files_deleted"] += 1
            result["bytes_freed"] += size
            pruned_here += 1
        if pruned_here:
            result["sessions_pruned"] += 1
    if result["files_deleted"]:
        freed_mb = result["bytes_freed"] / (1024 * 1024)
        logger.info(
            "Audio retention: deleted %d file(s) in %d session(s), "
            "freed %.1f MB — transcripts kept",
            result["files_deleted"], result["sessions_pruned"], freed_mb)
    else:
        logger.debug("Audio retention: nothing to prune "
                     "(%d session(s) checked)",
                     result["sessions_checked"])
    return result


def scan_prunable_audio(output_root: str | Path,
                        retention_hours: float | None,
                        *, now: float | None = None,
                        exclude_session_dirs=()) -> dict:
    """Dry run of :func:`prune_old_audio`: what the next prune WOULD delete.

    Same eligibility rules, same arguments — but nothing is deleted.
    Returns ``sessions_checked``, ``sessions_prunable`` (sessions
    with at least one file that would be deleted), ``files_prunable``
    and ``bytes_prunable``. The UI layer uses this for the
    disk-usage line ("free space / what retention is about to
    delete"). ``retention_hours=None`` disables pruning (all zeros);
    non-positive hours raise ValueError, as in :func:`prune_old_audio`.
    """
    result = {"sessions_checked": 0, "sessions_prunable": 0,
              "files_prunable": 0, "bytes_prunable": 0}
    cutoff = _cutoff_for(retention_hours, now)
    if cutoff is None:
        return result
    excluded = {Path(p).resolve() for p in exclude_session_dirs}
    for session_dir in list_sessions(output_root):
        result["sessions_checked"] += 1
        files = _prunable_files(session_dir, cutoff, excluded)
        if files:
            result["sessions_prunable"] += 1
            result["files_prunable"] += len(files)
            result["bytes_prunable"] += sum(size for _, size in files)
    return result
