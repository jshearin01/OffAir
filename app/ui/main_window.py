"""Main window — PyQt6 UI for OffAir (Phases 1–5, redesigned Phase 6,
left navigation Nov 19 2026).

Importing this module must not fail when PyQt6 is absent: ``QT_AVAILABLE``
is False and :func:`create_main_window` raises a clear RuntimeError.
Level callbacks from the capture threads are marshalled onto the Qt
thread via a small QObject signal bridge — capture code never calls Qt
directly.

Nov 19 2026 layout (replacing the Phase 6 left Record card + 7 tabs):
a header (app name, status, recording timer), a left navigation list
and one focused view per task — Record (recording + live transcript),
Library (sessions, search), Calendar (the recordings calendar),
Session (the open session: transcript, suggestions, rename, process
actions), People (known speakers), Settings (the settings form as a
view), MCP setup and Help. The nav is flat since Nov 20 2026: every
row is a view, no "More" grouping. It fits a 1280x800 window. Every
control from the old layout is still present — moved into a view,
never removed. The views themselves live in ``app/ui/views/``; this
module keeps the shared state, the workers and all the behaviour.
"""
from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.audio.app_sessions import (
    OTHER_APP_SENTINEL,
    audio_sessions_available,
    display_name_for_exe,
    find_session,
    list_audio_sessions,
)
from app.audio.capture import CaptureState, DualCapture
from app.audio.mixer import mix_files
from app.audio.auto_detect import MeetingWatcher, auto_detect_available
from app.audio.devices import (
    default_loopback,
    default_microphone,
    find_teams_process,
)
from app.audio.preroll import PreRollBuffer
from app.audio.preroll_feeder import PreRollFeeder
from app.config import (
    AppConfig,
    NEARLIVE_CHUNK_SECONDS,
    SAMPLE_RATE,
    SPEAKERS_DB_FILE,
    WHISPER_MODELS,
    default_settings_path,
    save_settings,
)
from app.diarization import DIARIZATION_MODES
from app.diarization.names import (
    load_speaker_names,
    rename_speaker,
)
from app.diarization.session import diarize_session, reapply_names
from app.diarization.sherpa import check_availability, sherpa_install_guide
from app.platform_info import (
    auto_detect_status,
    default_process_names,
    describe_capture_state,
    loopback_empty_message,
    mic_empty_message,
    per_app_capture_status,
)
from app.nearlive.chunker import ChunkedTranscriber, load_live_transcript
from app.speakers.database import SpeakerDatabase
from app.speakers.embeddings import get_extractor
from app.speakers.people import get_speaker_detail, speaker_utterances
from app.speakers.session import (
    auto_identify_session,
    confirm_suggestion as confirm_session_suggestion,
    correct_suggestion as correct_session_suggestion,
    find_sample_for_speaker,
    load_suggestions,
    rename_identity,
    sample_window_for_label,
    suggest_for_session,
    write_suggestions,
)
from app.storage.bookmarks import add_bookmark, list_bookmarks
from app.storage.browser import (
    format_local_time,
    hybrid_search,
    library_row_lines,
    list_session_summaries,
    list_sources,
    row_date_text,
    search_sessions,
    summarize_session,
)
from app.storage.semantic_search import (
    ModelNotAvailableError,
    SemanticSearchIndex,
)
from app.transcription.cleanup import (
    CLEAN_JSON,
    TRANSCRIPT_VARIANT_LABELS,
    TRANSCRIPT_VARIANTS,
    clean_session,
    polish_session,
    transcript_variant_path,
)
from app.transcription.enterprise_cleanup import (
    EnterpriseCleanupError,
    EnterpriseCleanupRunner,
    enterprise_clean_session,
)
from app.transcription.engine import (
    TRANSCRIPT_JSON,
    TranscriptionEngine,
    TranscriptionResult,
    audio_duration_seconds,
    whisper_model_cached,
)
from app.transcription.exports import load_transcript_json
from app.transcription.names_dictionary import (
    add_name,
    build_initial_prompt,
    load_names_dictionary,
)
from app.storage.retention import (
    disk_status,
    prune_old_audio,
    scan_prunable_audio,
)
from app.storage.schedules import (
    Schedule,
    cancel_schedule,
    create_schedule,
    default_schedules_path,
    due_schedules,
    expire_past_due,
    expire_stale_due,
    load_schedules,
    mark_fired,
    mark_skipped,
    visible_schedules,
)
from app.storage.sessions import (
    COMBINED_FILE,
    MIC_FILE,
    SYSTEM_FILE,
    TRANSCRIPT_LIVE_FILE,
    create_dictation_session,
    create_session,
    delete_session,
    finalize_session,
    is_dictation_session,
    load_metadata,
    rename_session,
    set_session_source,
    update_metadata,
)
from app.calendar.outlook import (
    format_event_label,
    link_session,
    open_event_in_outlook,
)
from app.storage.source import (
    SOURCES,
    SOURCE_LABELS,
    SOURCE_OTHER,
    resolve_recording_source,
)
from app.resources import set_app_icon
from app.ui.calendar_view import CalendarWidget, session_day
from app.ui.help_view import create_help_tab
from app.ui.level_meter import QT_AVAILABLE, LevelMeter
from app.ui.theme import apply_theme, apply_theme_to_app
from app.ui.views import VIEW_NAMES
from app.ui.waveform import WaveformWidget, color_for_speaker, load_rms_buckets

logger = logging.getLogger("offair.ui")

# Stop-time finalize: how long to wait for an in-flight live chunk
# to finish after Stop before abandoning it and finalizing from the
# saved live transcript instead (see _begin_stop_finalize). A 25 s
# chunk transcribes in seconds at base speed, so 60 s is generous
# headroom for a slow chunk or a stalled first-run download — and
# bounded, so Stop can never wait forever on a stuck worker.
LIVE_FINALIZE_WAIT_SECONDS = 60.0

if QT_AVAILABLE:
    from PyQt6.QtCore import (
        QDateTime,
        QObject,
        Qt,
        QThread,
        QTimer,
        QUrl,
        pyqtSignal,
    )
    from PyQt6.QtGui import QAction, QColor, QFont, QKeySequence, QShortcut
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QDateTimeEdit,
        QDialog,
        QFrame,
        QGraphicsOpacityEffect,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QStackedWidget,
        QSystemTrayIcon,
        QVBoxLayout,
        QWidget,
    )
    try:
        from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
        MULTIMEDIA_AVAILABLE = True
    except ImportError:  # QtMultimedia is an optional PyQt6 component
        QAudioOutput = None  # type: ignore[assignment]
        QMediaPlayer = None  # type: ignore[assignment]
        MULTIMEDIA_AVAILABLE = False
else:
    QObject = object  # type: ignore[assignment,misc]
    QTimer = None  # type: ignore[assignment]
    pyqtSignal = None  # type: ignore[assignment]


if QT_AVAILABLE:
    class _LevelBridge(QObject):
        """Thread-safe bridge: capture thread -> Qt signal -> meter widget."""

        level_changed = pyqtSignal(float)

        def __init__(self, meter: LevelMeter) -> None:
            super().__init__()
            self.level_changed.connect(meter.set_level)

        def emit(self, level: float) -> None:
            self.level_changed.emit(level)

    class _AppPickerCombo(QComboBox):
        """Target-app picker: refreshes its list when opened.

        ``refresh_callback`` is called (if set) just before the
        dropdown opens, so the list shows the apps that have audio
        sessions *right now* — an app started a minute ago appears,
        one that closed disappears. Plain QComboBox has no
        about-to-open signal, hence the subclass.
        """

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.refresh_callback = None

        def showPopup(self) -> None:  # noqa: N802 - Qt name
            if self.refresh_callback is not None:
                try:
                    self.refresh_callback()
                except Exception:  # noqa: BLE001 — a stale list beats a crash
                    logger.exception("App picker refresh failed")
            super().showPopup()

    class _NavStack(QStackedWidget):
        """The view stack behind the left navigation.

        A plain QStackedWidget plus :meth:`tabText`, so code (and
        tests) that used to address the old QTabWidget by tab name
        can address views by name the same way: ``win.tabs`` is
        this stack, ``win.tabs.tabText(i)`` is the view name.
        ``_nav_names`` is set by MainWindow after construction.
        """

        _nav_names: tuple = ()

        def tabText(self, index: int) -> str:  # noqa: N802 - Qt name
            if 0 <= index < len(self._nav_names):
                return self._nav_names[index]
            return ""


if QT_AVAILABLE:
    class _AutoDetectBridge(QObject):
        """Thread-safe bridge: watcher thread -> Qt signal -> main window.

        The meeting watcher polls on its own thread and must never
        call Qt directly — its callbacks emit these signals instead,
        so start/stop run on the UI thread.
        """

        meeting_started = pyqtSignal()
        meeting_ended = pyqtSignal()

    class _TranscribeWorker(QObject):
        """Runs transcription off the UI thread (Phase 2).

        Emits progress as (fraction 0..1, status text); finished carries
        a small summary dict; error carries the exception message.
        ``cancelled`` carries the partial segment count: when
        ``is_cancelled`` fires, :meth:`TranscriptionEngine.
        transcribe_session` returns its partial segments and writes
        no files, so cancel is safe and fast.
        """

        progress = pyqtSignal(float, str)
        finished = pyqtSignal(dict)
        error = pyqtSignal(str)
        cancelled = pyqtSignal(dict)

        def __init__(self, session_dir: Path, model_size: str,
                     is_cancelled=None) -> None:
            super().__init__()
            self.session_dir = session_dir
            self.model_size = model_size
            self.is_cancelled = is_cancelled

        def run(self) -> None:
            engine = None
            try:
                engine = TranscriptionEngine(model_size=self.model_size)

                def on_progress(frac: float, elapsed: float,
                                remaining: float | None) -> None:
                    text = f"Transcribing… {frac:.0%} elapsed {elapsed:.0f}s"
                    if remaining is not None:
                        text += f" — ~{remaining:.0f}s left"
                    self.progress.emit(frac, text)

                # Names-dictionary biasing (Nov 10 2026): the
                # dictionary's names/correct spellings become
                # faster-whisper's initial_prompt at decode time.
                # A missing/corrupt dictionary just means no prompt
                # — transcription proceeds exactly as before.
                try:
                    initial_prompt = build_initial_prompt(
                        load_names_dictionary())
                except Exception:  # noqa: BLE001 — prompt is best-effort
                    logger.warning(
                        "Could not load the names dictionary — "
                        "transcribing without it", exc_info=True)
                    initial_prompt = None
                result = engine.transcribe_session(
                    self.session_dir, initial_prompt=initial_prompt,
                    progress_callback=on_progress,
                    is_cancelled=self.is_cancelled)
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception("Transcription failed")
                self.error.emit(str(exc))
                return
            finally:
                # Unload on every path, not just success: the old
                # code called unload() only after a successful
                # transcribe_session, so a failed run left the model
                # resident (MainWindow keeps the worker until the
                # next run) and a retry piled a second load onto
                # the first — see Item 3 findings.
                if engine is not None:
                    engine.unload()
            if result.cancelled:
                # No transcript.json was written (see
                # transcribe_session) — report the cancel, not a
                # finish, so the chain never diarizes a partial file.
                self.cancelled.emit({
                    "segments": len(result.segments),
                    "session_dir": str(self.session_dir),
                })
                return
            self.finished.emit({
                "segments": len(result.segments),
                "wall_seconds": round(result.wall_seconds, 1),
                "real_time_factor": (round(result.real_time_factor, 2)
                                     if result.real_time_factor else None),
                "model": result.model,
                "session_dir": str(self.session_dir),
            })

    class _FinalizeWorker(QObject):
        """Runs the Stop/finalize tail pass off the UI thread.

        :meth:`ChunkedTranscriber.finish` over the tail can take
        minutes when live transcription died early (the tail is
        then the bulk of the meeting), and a first-use model
        download happens inside it too — so it runs here, never
        inline in a UI slot, with progress from the chunker's
        ``on_chunk`` and the same cancel flag the manual path
        uses. ``finished`` carries the same summary dict as
        :class:`_TranscribeWorker`. ``error`` carries a dict with
        ``message`` / ``phase`` / ``session_dir``: ``"tail"`` means
        the chunker pass failed (the caller falls back to the
        full-file path), ``"write"`` means the canonical files
        could not be written (the caller surfaces the error —
        retrying the same write would fail the same way).
        ``cancelled`` carries ``session_dir``: the live file is
        left on disk and no ``transcript.json`` is written, so the
        session stays transcribable by hand (Transcribe in the
        Session view).
        """

        progress = pyqtSignal(float, str)
        finished = pyqtSignal(dict)
        error = pyqtSignal(dict)
        cancelled = pyqtSignal(dict)

        def __init__(self, session_dir: Path, chunker: ChunkedTranscriber,
                     model: str, is_cancelled=None) -> None:
            super().__init__()
            self.session_dir = session_dir
            self.chunker = chunker
            self.model = model
            self.is_cancelled = is_cancelled

        def run(self) -> None:
            session_dir = self.session_dir
            combined = session_dir / COMBINED_FILE
            mic_path = session_dir / MIC_FILE
            system_path = session_dir / SYSTEM_FILE
            duration = audio_duration_seconds(combined)
            prev_on_chunk = self.chunker.on_chunk
            prev_is_cancelled = self.chunker.is_cancelled
            try:
                def on_chunk(chunk) -> None:
                    if duration:
                        frac = min(1.0, chunk.end / duration)
                        self.progress.emit(
                            frac,
                            f"Finalizing transcript… {chunk.end:.0f}s "
                            f"of {duration:.0f}s")
                    if prev_on_chunk is not None:
                        prev_on_chunk(chunk)

                self.chunker.on_chunk = on_chunk
                if self.is_cancelled is not None:
                    self.chunker.is_cancelled = self.is_cancelled
                self.chunker.finish(
                    combined,
                    mic_path=mic_path if mic_path.exists() else None,
                    system_path=system_path if system_path.exists() else None)
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception("Live + tail finalize failed")
                self.error.emit({"message": str(exc), "phase": "tail",
                                 "session_dir": str(session_dir)})
                return
            finally:
                self.chunker.on_chunk = prev_on_chunk
                self.chunker.is_cancelled = prev_is_cancelled
                try:
                    self.chunker.unload()
                except Exception:  # noqa: BLE001 — unload is best-effort
                    logger.warning("Could not unload finalize chunker",
                                   exc_info=True)
            if self.is_cancelled is not None and self.is_cancelled():
                self.cancelled.emit({"session_dir": str(session_dir)})
                return
            try:
                # No model is loaded here: write_transcript_files
                # only formats the merged segments, so this step
                # cannot download anything.
                engine = TranscriptionEngine(model_size=self.model)
                result = TranscriptionResult(
                    segments=list(self.chunker.segments), model=self.model,
                    audio_path=combined,
                    audio_duration_seconds=duration,
                    wall_seconds=self.chunker.total_wall_seconds)
                engine.write_transcript_files(session_dir, result)
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception("Final transcript write failed")
                self.error.emit({"message": str(exc), "phase": "write",
                                 "session_dir": str(session_dir)})
                return
            try:
                (session_dir / TRANSCRIPT_LIVE_FILE).unlink(missing_ok=True)
            except OSError:  # noqa: BLE001 — a leftover live file is harmless
                logger.warning("Could not delete %s after finalize",
                               TRANSCRIPT_LIVE_FILE, exc_info=True)
            self.finished.emit({
                "segments": len(result.segments),
                "wall_seconds": round(result.wall_seconds, 1),
                "real_time_factor": (round(result.real_time_factor, 2)
                                     if result.real_time_factor else None),
                "model": self.model,
                "session_dir": str(session_dir),
            })

    class _DiarizeWorker(QObject):
        """Runs diarization off the UI thread (Phase 3).

        Same signal shape as _TranscribeWorker. Full mode can take
        minutes on CPU; the progress text says so, and the worker
        never blocks the UI.
        """

        progress = pyqtSignal(float, str)
        finished = pyqtSignal(dict)
        error = pyqtSignal(str)

        def __init__(self, session_dir: Path, mode: str,
                     min_speakers: int | None, max_speakers: int | None,
                     token: str | None,
                     sherpa_model_dir=None,
                     sherpa_expected_speakers: int | None = None,
                     sherpa_clustering_threshold: float | None = None,
                     ) -> None:
            super().__init__()
            self.session_dir = session_dir
            self.mode = mode
            self.min_speakers = min_speakers
            self.max_speakers = max_speakers
            self.token = token
            self.sherpa_model_dir = sherpa_model_dir
            self.sherpa_expected_speakers = sherpa_expected_speakers
            self.sherpa_clustering_threshold = sherpa_clustering_threshold

        def run(self) -> None:
            try:
                def on_progress(frac: float, text: str) -> None:
                    self.progress.emit(frac, text)

                payload = diarize_session(
                    self.session_dir, mode=self.mode,
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers,
                    token=self.token or None,
                    sherpa_model_dir=self.sherpa_model_dir,
                    sherpa_expected_speakers=self.sherpa_expected_speakers,
                    sherpa_clustering_threshold=(
                        self.sherpa_clustering_threshold),
                    progress_callback=on_progress)
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception("Diarization failed")
                self.error.emit(str(exc))
                return
            speakers = sorted({s["speaker"]
                               for s in payload.get("segments", [])})
            self.finished.emit({
                "segments": len(payload.get("segments", [])),
                "speakers": speakers,
                "mode": payload.get("mode"),
                "method": payload.get("method"),
                "wall_seconds": payload.get("wall_seconds"),
                "session_dir": str(self.session_dir),
            })

    class _SuggestWorker(QObject):
        """Suggests known names for a diarized session (Phase 4).

        Fast with the fallback extractor (<1 s); the ECAPA
        backends can take longer (embedding model load on first
        use), so it still runs off the UI thread. ``finished``
        carries the JSON form of the suggestions; the full objects
        (with embeddings) are kept on the worker for confirm/correct.
        ``embedding_backend`` is the configured backend name
        ("auto" = best available); the extractor is built in run()
        so a bad backend name fails in the worker, not at click.
        With ``auto_identify`` (the auto-pipeline chain only) the
        worker also runs :func:`auto_identify_session` right after
        suggesting — confident matches are applied and clean
        unknown clusters are enrolled as "Speaker A/B/…" identities
        — and ``finished`` carries the result under "auto_identify".
        """

        progress = pyqtSignal(float, str)
        finished = pyqtSignal(dict)
        error = pyqtSignal(str)

        def __init__(self, session_dir: Path, db_path: Path,
                     embedding_backend: str = "auto",
                     auto_identify: bool = False) -> None:
            super().__init__()
            self.session_dir = session_dir
            self.db_path = db_path
            self.embedding_backend = embedding_backend
            self.auto_identify = auto_identify
            self.suggestions: list = []
            self.auto_result: dict | None = None

        def run(self) -> None:
            try:
                from app.speakers.embeddings import get_extractor
                db = SpeakerDatabase(self.db_path)
                try:
                    extractor = get_extractor(self.embedding_backend)
                    self.suggestions = suggest_for_session(
                        self.session_dir, db, extractor=extractor)
                    write_suggestions(self.session_dir, self.suggestions)
                    if self.auto_identify:
                        self.auto_result = auto_identify_session(
                            self.session_dir, db, extractor=extractor,
                            suggestions=self.suggestions)
                finally:
                    db.close()
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception("Speaker suggestion failed")
                self.error.emit(str(exc))
                return
            self.finished.emit({
                "suggestions": [s.to_dict() for s in self.suggestions],
                "auto_identify": self.auto_result,
                "session_dir": str(self.session_dir),
            })

    class _StopWorker(QObject):
        """Stops capture off the UI thread (Nov 20 2026 Stop-freeze fix).

        :meth:`DualCapture.stop` joins the capture threads (up to
        10 s each) and mixes the tracks to ``combined.wav`` — on
        the UI thread that froze the app for seconds on a long
        meeting (measured 1.3–4.1 s of mix alone at 10 min, plus
        the joins). The worker owns the capture for the stop;
        ``finished`` carries ``(files, duration_seconds)`` and the
        window then finalizes the session on the UI thread.
        ``error`` carries the exception message: the recording
        itself is untouched (the raw tracks are already on disk).
        """

        finished = pyqtSignal(dict, float)
        error = pyqtSignal(str)

        def __init__(self, capture) -> None:
            super().__init__()
            self.capture = capture

        def run(self) -> None:
            started = time.monotonic()
            try:
                files = self.capture.stop()
                duration = self.capture.duration_seconds
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception("Capture stop failed")
                self.error.emit(str(exc))
                return
            logger.info("Capture stop + mix took %.2fs",
                        time.monotonic() - started)
            self.finished.emit(dict(files), float(duration))

    class _NearLiveWorker(QObject):
        """Chunked (near-live) transcription off the UI thread (Phase 4).

        Processes the loaded session's combined.wav in 25 s windows
        with the configured Whisper model and reports each chunk as
        it finishes, so the near-live pane grows progressively.
        This is the post-stop simulation path; the same chunker
        (process_new_audio) is what a live-during-recording poll
        would call — see PHASE4.md.
        """

        progress = pyqtSignal(float, str)
        finished = pyqtSignal(dict)
        error = pyqtSignal(str)

        def __init__(self, session_dir: Path, model_size: str) -> None:
            super().__init__()
            self.session_dir = session_dir
            self.model_size = model_size
            self.segments: list = []

        def run(self) -> None:
            try:
                from app.storage.sessions import COMBINED_FILE
                audio = self.session_dir / COMBINED_FILE
                if not audio.exists():
                    raise FileNotFoundError(
                        f"No {COMBINED_FILE} in {self.session_dir}")

                from app.diarization.simple import load_wav_mono
                _samples, _rate = load_wav_mono(audio)
                duration = len(_samples) / _rate if _rate else 0.0

                def on_chunk(chunk) -> None:
                    frac = (chunk.end / duration) if duration else 0.0
                    self.progress.emit(
                        min(1.0, frac),
                        f"Chunk {chunk.index} [{chunk.start:.0f}–"
                        f"{chunk.end:.0f}s]: {len(chunk.segments)} segments")

                chunker = ChunkedTranscriber(
                    model=self.model_size, on_chunk=on_chunk)
                chunks = chunker.process_file(
                    audio,
                    mic_path=self.session_dir / "mic.wav",
                    system_path=self.session_dir / "system.wav")
                total_wall = sum(c.wall_seconds for c in chunks)
                segments = chunker.segments
                self.segments = list(segments)
                chunker.unload()
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception("Near-live transcription failed")
                self.error.emit(str(exc))
                return
            self.finished.emit({
                "chunks": len(chunks),
                "segments": len(segments),
                "wall_seconds": round(total_wall, 1),
                "session_dir": str(self.session_dir),
            })

    class _LiveChunkWorker(QObject):
        """One live poll tick off the UI thread (Nov 19 2026).

        Snapshots the growing mic/system tracks to temp copies
        (never the live files — capture is still writing them),
        mixes the snapshots, and transcribes newly completed 25 s
        windows from the mix. The snapshot copy + mix used to run
        synchronously in :meth:`_on_live_poll` on the UI thread —
        0.7–1.9 s per tick at 10 minutes, growing with meeting
        length, and a tick in flight delayed Stop itself — so
        since Nov 20 2026 the worker owns the whole tick (Stop-
        freeze fix). The chunker (owned by the main window) holds
        the watermark and the segments, so a tick that finds
        nothing due emits an empty list and the next tick retries.
        The snapshot dir is deleted when the tick ends, whatever
        happened; the UI slot saves the progressive live transcript
        from the chunker's in-memory segments.
        """

        finished = pyqtSignal(list)
        error = pyqtSignal(str)

        def __init__(self, chunker: ChunkedTranscriber,
                     mic_path: Path, system_path: Path,
                     snapshot_dir: Path) -> None:
            super().__init__()
            self.chunker = chunker
            self.mic_path = mic_path
            self.system_path = system_path
            self.snapshot_dir = snapshot_dir
            # S6 (Nov 20 2026 review fix): True once the snapshot
            # copy + mix are done and only transcription remains.
            # Before that the tick is still reading the live
            # tracks, so a Stop landing now must abandon it
            # up-front (see on_stop) instead of letting the torn
            # copy fail into the live-error path.
            self.snapshot_done = False

        def run(self) -> None:
            try:
                mic_snap = self.snapshot_dir / MIC_FILE
                sys_snap = self.snapshot_dir / SYSTEM_FILE
                combined_snap = self.snapshot_dir / COMBINED_FILE
                shutil.copyfile(self.mic_path, mic_snap)
                shutil.copyfile(self.system_path, sys_snap)
                mix_files(mic_snap, sys_snap, combined_snap)
                self.snapshot_done = True
                chunks = self.chunker.process_new_audio(
                    combined_snap, mic_path=mic_snap,
                    system_path=sys_snap)
                self.finished.emit(chunks)
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception("Live chunk transcription failed")
                self.error.emit(str(exc))
            finally:
                shutil.rmtree(self.snapshot_dir, ignore_errors=True)

    class _SummarizeWorker(QObject):
        """Summarises the loaded session with the local LLM (Phase 5).

        Local only (Ollama on localhost, or llama-cpp with a local
        model file). Runs off the UI thread; Ollama's generation can
        take a while on CPU. ``finished`` carries the summary text;
        ``error`` carries the friendly message (e.g. Ollama not
        running). Never calls a cloud API.
        """

        progress = pyqtSignal(float, str)
        finished = pyqtSignal(str)
        error = pyqtSignal(str)

        def __init__(self, session_dir: Path, config) -> None:
            super().__init__()
            self.session_dir = session_dir
            self.config = config

        def run(self) -> None:
            try:
                from app.exports.local_llm import (
                    LocalLLMConfig,
                    summarize_locally,
                )
                from app.transcription.exports import load_transcript_json
                segments = load_transcript_json(
                    self.session_dir / TRANSCRIPT_JSON)
                text = "\n".join(
                    f"[{s.start:.0f}s] {s.speaker or ''}: {s.text}"
                    for s in segments)
                self.progress.emit(0.5, "Asking the local LLM…")
                cfg = LocalLLMConfig(
                    enabled=True, provider="ollama",
                    ollama_base_url=self.config.ollama_base_url,
                    ollama_model=self.config.ollama_model)
                summary = summarize_locally(text, config=cfg)
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception("Local LLM summary failed")
                self.error.emit(str(exc))
                return
            self.finished.emit(summary)

    class _PolishWorker(QObject):
        """AI-polishes the loaded session's cleaned transcript (Nov 10).

        Local Ollama only (see app/transcription/cleanup.py):
        rewrites transcript-clean.txt into transcript-polished.md,
        clearly marked as an AI rewrite. Runs off the UI thread;
        Ollama generation can take a while on CPU. ``finished``
        carries the polished file's path; ``error`` carries the
        friendly message (e.g. Ollama not running) — the
        rule-cleaned version stays in place either way.
        """

        progress = pyqtSignal(float, str)
        finished = pyqtSignal(str)
        error = pyqtSignal(str)

        def __init__(self, session_dir: Path, config) -> None:
            super().__init__()
            self.session_dir = session_dir
            self.config = config

        def run(self) -> None:
            try:
                self.progress.emit(0.5, "Asking the local LLM to polish…")
                path = polish_session(self.session_dir,
                                      config=self.config)
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception("Transcript polish failed")
                self.error.emit(str(exc))
                return
            self.finished.emit(str(path))


    class _EnterpriseCleanupWorker(QObject):
        """Enterprise-cleanup run off the UI thread (Nov 21 2026).

        Runs the headless ``claude -p`` subprocess through
        :func:`enterprise_clean_session` with the window's config
        (which enforces the opt-in in core) and a cancel event.
        ``finished`` carries the artifact path; ``error`` carries
        the exception message — a missing CLI raises the guided
        install error, which the window shows as a guided dialog.
        """

        progress = pyqtSignal(float, str)
        finished = pyqtSignal(str)
        error = pyqtSignal(str)

        def __init__(self, session_dir: Path, config,
                     is_cancelled=None) -> None:
            super().__init__()
            self.session_dir = session_dir
            self.config = config
            self.is_cancelled = is_cancelled
            self.runner: EnterpriseCleanupRunner | None = None

        def cancel(self) -> None:
            """Kill the running CLI process (from the UI thread)."""
            if self.runner is not None:
                self.runner.cancel()

        def run(self) -> None:
            try:
                self.progress.emit(
                    0.5, "Sending transcript text to the Claude Code "
                    "CLI… (audio never leaves this machine)")
                self.runner = EnterpriseCleanupRunner(
                    claude_command=self.config.enterprise_cleanup_claude_command,
                    timeout=self.config.enterprise_cleanup_timeout_seconds)
                path = enterprise_clean_session(
                    self.session_dir, config=self.config,
                    runner=self.runner, is_cancelled=self.is_cancelled)
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception("Enterprise cleanup failed")
                self.error.emit(str(exc))
                return
            self.finished.emit(str(path))

    class _WaveformWorker(QObject):
        """RMS buckets for the Session view waveform, off the UI thread.

        :func:`load_rms_buckets` reads the whole WAV to compute the
        buckets — that is file I/O + decoding, never on the UI
        thread. ``finished`` carries ``(buckets, duration)``; the
        window caches the result per session, so a session's
        waveform is computed once.
        """

        finished = pyqtSignal(list, float)
        error = pyqtSignal(str)

        def __init__(self, audio_path: Path, key: str = "") -> None:
            super().__init__()
            self.audio_path = audio_path
            # The session this computation is for: a result that
            # lands after the user switched sessions is stale and
            # must never paint another session's waveform.
            self.key = key

        def run(self) -> None:
            try:
                buckets, duration = load_rms_buckets(self.audio_path)
            except Exception as exc:  # noqa: BLE001 — surfaced as empty
                logger.exception("Waveform computation failed")
                self.error.emit(str(exc))
                return
            self.finished.emit(buckets, duration)


#: Silence-watch window (seconds): the system track is judged on its
#: recent peaks only — a quiet intro is not "silence", ten seconds
#: of flatline while the mic is live is.
_SILENCE_WINDOW_SECONDS = 10.0
#: Peak at/below this counts as silence on the system track.
_SILENCE_PEAK_THRESHOLD = 0.005
#: Peak at/above this counts as the mic being live.
_MIC_LIVE_THRESHOLD = 0.01

#: Hint under the near-live pane (restored when a new session loads,
#: so a previous run's "Done: …" line never claims to be about the
#: newly loaded session). Names the model actually in use — built
#: by :meth:`MainWindow._nearlive_hint_text`, never hard-coded:
#: the manual path and the live path share the configured model
#: (Nov 20 2026, Item 7).
def _nearlive_hint_text(model: str) -> str:
    return (
        "Live transcription runs during recording in 25 s windows with "
        f"Whisper {model}, and the pane fills as chunks land; "
        "labels are provisional (simple You/Remote when both tracks "
        "exist). The button below re-runs chunked transcription on a "
        f"finished session with Whisper {model}.")


#: Spoken-side-only wording for dictation (verbatim, wherever it
#: appears — the Record view note and the Session view note).
DICTATION_NOTE_TEXT = (
    "Dictation captures the spoken side only — what you say into "
    "the microphone. It does not capture the other side of a "
    "conversation.")

#: Consent note for scheduled recordings (Nov 22 2026): shown
#: under the schedule form, in docs/SCHEDULED_RECORDING.md, and in
#: the MCP schedule_recording description. Scheduled auto-start
#: does NOT trigger Teams' own recording banner — the same stance
#: as the auto-detect consent note. Kept to two lines at card
#: width so the left column still fits the 1280x800 geometry
#: audit; the "only while OffAir is running" limit is in the
#: docs and the list tooltip.
SCHEDULE_CONSENT_NOTE = (
    "Starts automatically and does not trigger Teams' recording "
    "banner — tell participants you are recording.")

#: Button label for the enterprise cleanup: the text-leaves-machine
#: warning is part of the label itself, not hidden in a tooltip.
ENTERPRISE_BUTTON_TEXT = (
    "Enterprise-clean — transcript text leaves this machine")

#: Short diarization-mode names for the live status line.
_DIARIZATION_SHORT_LABELS = {
    "simple": "Simple", "full": "Full", "sherpa": "Sherpa"}


def _format_mmss(seconds: float) -> str:
    """``[MM:SS]`` for a bookmark / utterance time, as the
    segments list already formats segment starts."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def _format_bytes(num_bytes: int) -> str:
    """Human size for the disk line: ``"42.3 GB"``, ``"120 MB"``."""
    value = float(max(0, int(num_bytes)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"  # pragma: no cover — unreachable


def _week_header_for(summary) -> str:
    """Week-group header for a Library row's session.

    ``"This week"`` for the current ISO week, else
    ``"Week of <Monday>"``. Sessions with no derivable day group
    under ``"Other sessions"``.
    """
    from datetime import date as _date

    day = session_day(summary)
    if day is None:
        return "Other sessions"
    monday = day - timedelta(days=day.weekday())
    today = _date.today()
    this_monday = today - timedelta(days=today.weekday())
    if monday == this_monday:
        return "This week"
    return f"Week of {monday:%b %d, %Y}"


#: First-run onboarding text (also in docs/OVERVIEW.md, the
#: "Getting started" Help section — the modal says where to find
#: it again, so the Help copy is the durable one).
ONBOARDING_TEXT = (
    "Welcome to OffAir.\n\n"
    "How it works: Record a meeting, OffAir transcribes it, labels "
    "who said what, and learns to recognise the people you meet "
    "with.\n\n"
    "Everything stays on this machine. Recordings, transcripts and "
    "voice profiles live in your recordings folder and settings "
    "folder only — nothing is uploaded, except when you explicitly "
    "opt in to a feature that says so (enterprise cleanup sends "
    "transcript text only, and only when you enable it in "
    "Settings).\n\n"
    "First run only: the Whisper base model (~142 MB) and the "
    "sherpa diarization models (~33 MB) download once, then work "
    "offline.\n\n"
    "You can read this again any time in Help → Getting started.")


def _onboarding_marker_path() -> Path:
    """Marker file recording that onboarding was shown once."""
    return default_settings_path().parent / "onboarding_shown"


def _onboarding_shown(marker: Path | None = None) -> bool:
    """Whether the first-run onboarding has already been shown."""
    try:
        return (marker or _onboarding_marker_path()).exists()
    except OSError:
        return False


def _mark_onboarding_shown(marker: Path | None = None) -> None:
    """Record that onboarding was shown (best-effort, never fatal)."""
    path = marker or _onboarding_marker_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError:  # noqa: BLE001 — onboarding must never break startup
        logger.warning("Could not write the onboarding marker",
                       exc_info=True)


def system_silent_while_mic_live(
        system_peaks, mic_peaks, *,
        silence_threshold: float = _SILENCE_PEAK_THRESHOLD,
        mic_live_threshold: float = _MIC_LIVE_THRESHOLD) -> bool:
    """True when the system track is silent while the mic is live.

    ``system_peaks`` / ``mic_peaks`` are the recent peak levels
    (0.0–1.0, last :data:`_SILENCE_WINDOW_SECONDS`). The system
    track is silent when *every* recent peak is at/below the
    silence threshold; the mic is live when *any* recent peak
    clears the mic threshold. Empty history is not evidence —
    either list empty returns False, so the warning only ever
    fires on observed silence, never on missing data.
    """
    if not system_peaks or not mic_peaks:
        return False
    return (max(system_peaks) <= silence_threshold
            and max(mic_peaks) >= mic_live_threshold)


class MainWindow(QMainWindow if QT_AVAILABLE else object):  # type: ignore[misc]
    """Record / Pause / Resume / Stop with two live level meters."""

    def __init__(self, config: AppConfig | None = None,
                 auto_watcher_factory=None,
                 schedules_path: str | Path | None = None) -> None:
        if not QT_AVAILABLE:
            raise RuntimeError(
                "PyQt6 is not installed — install it with "
                "`pip install -r requirements.txt` to use the GUI.")
        super().__init__()
        self.config = config or AppConfig()
        # tinyfix explicit-choice tracking (Nov 22 2026 — mirrors
        # whisper_model_explicit): True once the user clicks the
        # Settings checkbox, so a plain save never promotes a
        # carried-over default into an explicit opt-in. Reset to
        # the config's flag whenever the form is rebuilt from the
        # config (see _sync_advanced_settings_controls).
        self._tinyfix_explicit = bool(
            self.config.tinyfix_enabled_explicit)
        self.capture: DualCapture | None = None
        self.session_dir: Path | None = None
        self._record_started_at: float | None = None
        self._elapsed_paused_total = 0.0
        self._pause_started_at: float | None = None
        # Auto-detect (Nov 8 2026): the meeting watcher + whether the
        # current recording was started by it. ``auto_watcher_factory``
        # is the test seam — ``factory(process_names=...,
        # on_meeting_started=..., on_meeting_ended=...)`` returning a
        # watcher with ``start`` / ``stop`` / ``notify_manual_stop``;
        # None means the real MeetingWatcher.
        self._auto_watcher_factory = auto_watcher_factory
        self._auto_watcher = None
        self._auto_bridge = None
        self._auto_recording = False
        # Scheduled recordings (Nov 22 2026): one-off timers in
        # the shared store file (~/.offair/schedules.json — the
        # MCP server writes the same file from its own process).
        # ``schedules_path`` is the test seam; None means the
        # default store. ``_schedules_mtime_ns`` is the last file
        # mtime the list was loaded from (None = never loaded).
        self._schedules_path = (
            Path(schedules_path) if schedules_path is not None
            else default_schedules_path())
        self._schedules: list[Schedule] = []
        self._schedules_mtime_ns: int | None = None
        # The schedule id that started the current recording, if
        # it was schedule-fired (status text only — the watcher
        # never stops it, and Stop is always manual).
        self._scheduled_recording_id: str | None = None

        self.setWindowTitle("OffAir — Recorder & Transcription")
        # App icon (1.0.1): the main window carries the OffAir mark so
        # the title bar and the taskbar show it. Missing icon file is
        # a warning, never a startup failure (see app/resources.py);
        # main.py sets the same icon on the QApplication so dialogs
        # inherit it too.
        set_app_icon(self)
        # Phase 6: the redesign fits a 1280x800 window (the old
        # single-column stack needed ~2500 px of vertical scrolling).
        self.resize(1280, 800)
        self.setMinimumSize(960, 600)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        self.mic_combo = QComboBox()
        self.loopback_combo = QComboBox()
        # Device names are longer than the fixed-width Record card,
        # so the combo elides them — the full name lives in the
        # tooltip (see _update_device_tooltips).
        self.mic_combo.currentIndexChanged.connect(
            self._update_device_tooltips)
        self.loopback_combo.currentIndexChanged.connect(
            self._update_device_tooltips)
        # Target-app picker (Nov 10 2026): a real picker over the
        # apps that currently have audio sessions — friendly name +
        # live state — not a hand-typed exe name. The combo still
        # lives in ``process_combo`` (tests reference it). The
        # chosen exe name drives per-app capture (see on_record);
        # "Other app…" is the manual escape hatch for an app with
        # no session (or no session listing on this machine).
        self.process_combo = _AppPickerCombo()
        self.process_combo.refresh_callback = self._refresh_app_picker
        self.process_combo.setToolTip(
            "The app whose audio is recorded in “Record one app” "
            "mode. The list refreshes when you open it.")
        self._manual_process_name: str | None = None
        self._picker_last_exe: str | None = None
        # Capture mode has ONE editor after the Nov 20 2026
        # simplification: the Settings dialog's 3-option combo.
        # The Record view's 2-option combo is gone; the mode
        # label below still explains what will be recorded.
        self.mode_label = QLabel("Capture: —")
        # Callout style (QLabel#modeLabel): a soft info box so the
        # capture-mode explanation reads deliberately, not like a
        # stray grey strip.
        self.mode_label.setObjectName("modeLabel")
        self.mode_label.setWordWrap(True)
        # Word-wrapped status lines must never be compressed below
        # their wrapped height (the Record card is nearly full, and
        # the layout would otherwise shrink this label and clip its
        # text). Minimum vertical policy pins it to its sizeHint.
        self.mode_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.status_label = QLabel("Ready. Pick an app, then Record.")
        self.status_label.setObjectName("statusPill")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.elapsed_label = QLabel("00:00")
        self.elapsed_label.setObjectName("timerLabel")
        self.session_label = QLabel("")
        self.session_label.setWordWrap(True)
        self.session_label.setObjectName("hint")
        self.session_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.session_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)

        self.mic_meter = LevelMeter("horizontal")
        self.system_meter = LevelMeter("horizontal")
        self._mic_bridge = _LevelBridge(self.mic_meter)
        self._system_bridge = _LevelBridge(self.system_meter)
        # Silence watch (Nov 10 2026): the level bridges feed the
        # meters; these taps remember recent peaks so a timer can
        # warn when the system track stays silent while the mic is
        # live (the work-PC failure: a recording of nothing, found
        # only afterwards). See _check_system_silence.
        self._mic_levels: list[tuple[float, float]] = []
        self._system_levels: list[tuple[float, float]] = []
        self._mic_bridge.level_changed.connect(
            lambda v: self._note_level("mic", v))
        self._system_bridge.level_changed.connect(
            lambda v: self._note_level("system", v))
        self._system_silence_warned = False
        self._silence_timer = QTimer(self)
        self._silence_timer.setInterval(2000)
        self._silence_timer.timeout.connect(self._check_system_silence)

        self.record_btn = QPushButton("●  Record")
        self.record_btn.setObjectName("recordButton")
        self.record_btn.setMinimumHeight(48)
        self.pause_btn = QPushButton("❚❚  Pause")
        self.resume_btn = QPushButton("▶  Resume")
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("dangerButton")
        # Dictation (Nov 21 2026): the ONE control the Record view
        # gains — a checkable toggle beside Record. When on, the
        # app picker and system meter hide, capture is mic-only,
        # and the session is flagged dictation (spoken side only).
        self.dictation_btn = QPushButton("Dictation")
        self.dictation_btn.setCheckable(True)
        self.dictation_btn.setMinimumHeight(48)
        self.dictation_btn.setToolTip(
            "Dictation — mic only. " + DICTATION_NOTE_TEXT)
        self.dictation_note_label = QLabel(DICTATION_NOTE_TEXT)
        self.dictation_note_label.setObjectName("hint")
        self.dictation_note_label.setWordWrap(True)
        self.dictation_note_label.setVisible(False)
        # A second label instance for the Session view — one
        # widget cannot live in two layouts at once.
        self.session_dictation_note_label = QLabel(DICTATION_NOTE_TEXT)
        self.session_dictation_note_label.setObjectName("hint")
        self.session_dictation_note_label.setWordWrap(True)
        self.session_dictation_note_label.setVisible(False)
        # Bookmark (Nov 21 2026): an action button in the transport
        # row — marks the current moment while recording (Ctrl+B
        # does the same). Enabled only while recording.
        self.bookmark_btn = QPushButton("Bookmark")
        self.bookmark_btn.setToolTip(
            "Mark this moment in the recording (Ctrl+B).")
        self.bookmark_btn.setEnabled(False)
        # Scheduled recordings (Nov 22 2026): the add-form + list
        # widgets. Layout lives in build_record_view (the card
        # under the Record card); behaviour lives here. The app
        # combo is an _AppPickerCombo like the main picker — its
        # list refreshes when opened, so an app started a minute
        # ago can be scheduled.
        self.schedule_list = QListWidget()
        self.schedule_list.setMaximumHeight(48)
        self.schedule_list.setToolTip(
            "One-off scheduled recordings. Pending schedules fire "
            "automatically while OffAir is running; finished ones "
            "stay visible for a week.")
        self.schedule_list.itemSelectionChanged.connect(
            self._update_schedule_cancel_button)
        self.schedule_datetime = QDateTimeEdit()
        self.schedule_datetime.setCalendarPopup(True)
        # Short format (no year — the popup shows it): the
        # datetime edit shares a row with the app combo, so the
        # full "ddd MMM d yyyy, HH:mm" text would clip.
        self.schedule_datetime.setDisplayFormat("ddd d MMM HH:mm")
        self.schedule_datetime.setDateTime(
            QDateTime.currentDateTime().addSecs(3600))
        self.schedule_datetime.setToolTip(
            "When the recording starts (your local time).")
        self.schedule_app_combo = _AppPickerCombo()
        self.schedule_app_combo.refresh_callback = \
            self._refresh_schedule_app_combo
        self.schedule_app_combo.currentIndexChanged.connect(
            self._on_schedule_app_changed)
        self.schedule_app_combo.setToolTip(
            "The app whose audio is recorded. The list refreshes "
            "when you open it.")
        self._manual_schedule_app: str | None = None
        self._schedule_picker_last_exe: str | None = None
        self.schedule_subject_edit = QLineEdit()
        self.schedule_subject_edit.setPlaceholderText(
            "Meeting subject (optional)")
        self.schedule_add_btn = QPushButton("Schedule")
        self.schedule_add_btn.setToolTip(
            "Schedule a one-off recording at the time above.")
        self.schedule_add_btn.clicked.connect(self._on_schedule_add)
        self.schedule_cancel_btn = QPushButton("Cancel selected")
        self.schedule_cancel_btn.setToolTip(
            "Cancel the selected pending schedule.")
        self.schedule_cancel_btn.setEnabled(False)
        self.schedule_cancel_btn.clicked.connect(self._on_schedule_cancel)
        self.schedule_hint_label = QLabel(SCHEDULE_CONSENT_NOTE)
        self.schedule_hint_label.setObjectName("hint")
        self.schedule_hint_label.setWordWrap(True)
        self.schedule_hint_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.refresh_btn = QPushButton("Refresh devices")
        # Phase 2 — transcription (enabled only after a session is stopped).
        self.model_combo = QComboBox()
        for name, info in WHISPER_MODELS.items():
            self.model_combo.addItem(
                f"{name} — {info['speed']} ({info['size_mb']} MB)", name)
        self.model_combo.setCurrentText(
            self.model_combo.itemText(
                list(WHISPER_MODELS).index(
                    self.config.effective_whisper_model)))
        # The active-model line follows this combo (Item 7).
        self.model_combo.currentIndexChanged.connect(
            self._refresh_live_model_label)
        self.transcribe_btn = QPushButton("Transcribe session")
        self.transcribe_btn.setEnabled(False)
        self.cancel_transcribe_btn = QPushButton("Cancel")
        self.cancel_transcribe_btn.setEnabled(False)
        self.cancel_transcribe_btn.setToolTip(
            "Stop transcription now — partial segments are discarded "
            "and no transcript.json is written.")
        self.transcribe_progress = QProgressBar()
        self.transcribe_progress.setRange(0, 100)
        self.transcribe_progress.setValue(0)
        self.transcribe_progress.setVisible(False)
        self.transcribe_progress.setTextVisible(False)
        self.transcribe_label = QLabel("")
        self._transcribing = False
        self._transcribe_thread: QThread | None = None
        self._transcribe_worker: _TranscribeWorker | None = None
        self._transcribe_cancel_event: threading.Event | None = None
        # Live transcription (Nov 19 2026): the poller + chunker for
        # the recording in progress. ``_live_engine`` is the test
        # seam — an injected engine for the live chunker; None means
        # the chunker builds a real TranscriptionEngine lazily.
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(5000)
        self._live_timer.timeout.connect(self._on_live_poll)
        self._live_chunker: ChunkedTranscriber | None = None
        self._live_engine = None
        self._live_busy = False
        self._live_failed = False
        self._live_abandoned = False
        self._live_finalize_timeout_timer = QTimer(self)
        self._live_finalize_timeout_timer.setSingleShot(True)
        self._live_finalize_timeout_timer.timeout.connect(
            self._on_live_finalize_timeout)
        self._live_finalize_wait_seconds = LIVE_FINALIZE_WAIT_SECONDS
        # Live-wait countdown (Nov 20 2026 Stop-freeze fix): while
        # Stop waits for an in-flight live chunk, a 1 s timer
        # refreshes the "Waiting for live chunk… Ns / 60s" line so
        # the wait is visible — and Cancel is enabled, so the user
        # is never stuck watching a static label.
        self._live_wait_timer = QTimer(self)
        self._live_wait_timer.setInterval(1000)
        self._live_wait_timer.timeout.connect(
            self._update_live_wait_label)
        self._live_wait_started_at: float | None = None
        # Stop itself (Nov 20 2026): True from Stop until the stop
        # worker reports back — Record stays disabled while the
        # capture threads join and the tracks mix, so a second
        # Record can never start into a half-stopped capture.
        self._stopping = False
        self._stop_thread: QThread | None = None
        self._stop_worker: _StopWorker | None = None
        # Pre-roll (Nov 21 2026): the memory-only ring buffer and
        # its feeder, owned by the window. Created by
        # _setup_preroll_feeder() after the device combos are
        # populated; never started while recording.
        self._preroll_buffer: PreRollBuffer | None = None
        self._preroll_feeder: PreRollFeeder | None = None
        self._preroll_ready = False
        # Low-disk warning (Nov 21 2026): polled every 30 s while
        # recording; _disk_warned resets per recording, so the
        # dialog fires at most once per session.
        self._disk_timer = QTimer(self)
        self._disk_timer.setInterval(30_000)
        self._disk_timer.timeout.connect(
            self._check_disk_while_recording)
        self._disk_warned = False
        self._disk_warning_box = None
        # Tray icon (Nov 21 2026): created only when a system tray
        # is available; None otherwise (silent degrade — the main
        # window remains the stop path).
        self._tray = None
        self._tray_stop_action = None
        self._tray_show_action = None
        # Person page (Nov 21 2026): the speaker whose page is
        # open, or None when the People list layer is showing.
        self._person_name: str | None = None
        self._person_sample: dict | None = None
        # Onboarding dialog (non-modal, parentless — see
        # _maybe_show_onboarding).
        self._onboarding_dialog = None
        # The session the in-flight stop belongs to (S2, Nov 20
        # 2026 review fix): snapshotted in on_stop, so a
        # Library/Calendar click in the post-flip window cannot
        # redirect the finalize onto another session.
        self._stopping_session_dir = None
        # A close asked for while the stop is settling (S3, Nov 20
        # 2026 review fix): deferred until the stop + finalize
        # settle, then honoured by _maybe_close_pending.
        self._close_pending = False
        # Stop/finalize tail pass (runs in _FinalizeWorker, never on
        # the UI thread — see _finalize_transcription).
        self._finalizing = False
        self._finalize_thread: QThread | None = None
        self._finalize_worker: _FinalizeWorker | None = None
        self._finalize_cancel_event: threading.Event | None = None
        # Settings view: the embedded dialog + a snapshot of the
        # form as last built, so navigation never discards unsaved
        # edits (see _settings_form_dirty).
        self._settings_dialog = None
        self._settings_form_snapshot = None
        self._live_thread: QThread | None = None
        self._live_worker: _LiveChunkWorker | None = None
        self._live_list_started = False
        self._live_download_pending = False
        self._live_download_started_at: float | None = None
        self._stop_finalize_pending = False
        # Auto-process (Nov 8 2026): True while the
        # transcribe → diarize → suggest/identify chain started by
        # on_stop is still running. The recording is saved before
        # the chain starts, so a failed step can never lose it;
        # each error handler clears the flag so the chain stops
        # loudly instead of carrying on from a broken step.
        self._auto_processing = False
        # Phase 3 — diarization (enabled only after a transcript exists).
        # Phase 6: the mode list comes from DIARIZATION_MODES (the
        # registry), never hard-coded here — "sherpa" is offered
        # alongside "simple" and "full", with its availability shown
        # below the picker (see _update_diarize_mode_ui).
        self.diarize_mode_combo = QComboBox()
        for _mode_key, _mode_info in DIARIZATION_MODES.items():
            self.diarize_mode_combo.addItem(_mode_info["label"], _mode_key)
        _mode_idx = self.diarize_mode_combo.findData(
            self.config.diarization_mode)
        if _mode_idx >= 0:
            self.diarize_mode_combo.setCurrentIndex(_mode_idx)
        self.diarize_mode_combo.currentIndexChanged.connect(
            self._update_diarize_mode_ui)
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_edit.setPlaceholderText(
            "Hugging Face token (only for Full mode — never stored)")
        self.diarize_warning_label = QLabel("")
        self.diarize_warning_label.setWordWrap(True)
        # Sherpa availability (Track 1): shown only when the sherpa
        # mode is selected — installed? models present? — with a
        # guided message when something still needs downloading.
        self.sherpa_status_label = QLabel("")
        self.sherpa_status_label.setWordWrap(True)
        self.sherpa_status_label.setVisible(False)
        self.min_speakers_edit = QLineEdit()
        self.min_speakers_edit.setPlaceholderText("Min speakers (optional, 1–10)")
        self.max_speakers_edit = QLineEdit()
        self.max_speakers_edit.setPlaceholderText("Max speakers (optional, 1–10)")
        self.diarize_btn = QPushButton("Diarize session")
        self.diarize_btn.setEnabled(False)
        self.diarize_progress = QProgressBar()
        self.diarize_progress.setRange(0, 100)
        self.diarize_progress.setValue(0)
        self.diarize_progress.setVisible(False)
        self.diarize_progress.setTextVisible(False)
        self.diarize_label = QLabel("")
        self._diarizing = False
        self._diarize_thread: QThread | None = None
        self._diarize_worker: _DiarizeWorker | None = None
        # Phase 4 — known speakers + suggestions + near-live pane.
        self.known_speakers_list = QListWidget()
        # Long names wrap inside the list; a horizontal scrollbar
        # would only ever appear for a single unbroken word.
        self.known_speakers_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.known_speakers_list.setToolTip(
            "Speakers learned in earlier calls (speakers.db). "
            "Select one to play a voice sample or rename it; "
            "double-click to rename.")
        # Minimum height for 3 full rows at 1280x800: without it the
        # layout squeezes the lists and the 3rd row is cut mid-glyph.
        # Rows are ~40 px tall (13 px font + 8 px item padding top
        # and bottom), so 3 rows need ~124 px with the frame.
        self.known_speakers_list.setMinimumHeight(132)
        self.play_known_btn = QPushButton("▶  Play sample")
        self.play_known_btn.setToolTip(
            "Play a voice sample of the selected known speaker, so "
            "you can hear who it is.")
        self.play_known_btn.setEnabled(False)
        self.rename_known_btn = QPushButton("Rename…")
        self.rename_known_btn.setToolTip(
            "Rename the selected known speaker — the rename panel "
            "below is filled in for you.")
        self.rename_known_btn.setEnabled(False)
        self.suggestions_list = QListWidget()
        self.suggestions_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.suggestions_list.setToolTip(
            "Auto-suggestions for this session's speakers — select one "
            "to Confirm or Correct it. Double-click to hear a voice "
            "sample of that speaker.")
        self.suggestions_list.setMinimumHeight(132)
        self.play_suggestion_btn = QPushButton("▶  Play sample")
        self.play_suggestion_btn.setToolTip(
            "Play a voice sample of the selected suggestion, so you "
            "can tell who it is before confirming it.")
        self.play_suggestion_btn.setEnabled(False)
        self.suggest_btn = QPushButton("Suggest speakers for session")
        self.suggest_btn.setEnabled(False)
        self.confirm_btn = QPushButton("✓  Confirm selected suggestion")
        self.confirm_btn.setEnabled(False)
        self.correct_btn = QPushButton(
            "Correct selected suggestion (use the name field above)")
        self.correct_btn.setEnabled(False)
        self._suggesting = False
        self._suggest_thread: QThread | None = None
        self._suggest_worker: _SuggestWorker | None = None
        self._suggestion_objects: list = []
        # Stored suggestion dicts for the loaded session (from
        # speaker_suggestions.json — names only, no embeddings):
        # shown as a read-back by load_session; Confirm/Correct need
        # the live objects above, regenerated by Suggest.
        self._suggestion_dicts: list[dict] = []
        self.nearlive_list = QListWidget()
        self.nearlive_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nearlive_btn = QPushButton(
            "Run near-live (chunked) on session")
        self.nearlive_btn.setEnabled(False)
        self.nearlive_progress = QProgressBar()
        self.nearlive_progress.setRange(0, 100)
        self.nearlive_progress.setValue(0)
        self.nearlive_progress.setVisible(False)
        self.nearlive_progress.setTextVisible(False)
        self.nearlive_label = QLabel(self._nearlive_hint_text())
        self.nearlive_label.setWordWrap(True)
        # Active-model line (Nov 20 2026, Item 7): the Whisper
        # model actually in use, in the live-transcript area —
        # updated whenever the configured model changes (see
        # _refresh_live_model_label). The near-live tooltip names
        # it too; both are set there, never hard-coded.
        self.live_model_label = QLabel("")
        self.live_model_label.setObjectName("hint")
        self.live_model_label.setWordWrap(True)
        self._refresh_live_model_label()
        self.nearlive_label.setWordWrap(True)
        self._nearlive_running = False
        self._nearlive_thread: QThread | None = None
        self._nearlive_worker: _NearLiveWorker | None = None
        # Phase 5 — LLM export, local LLM summary, browser, settings.
        self.export_btn = QPushButton("Export session as LLM Markdown")
        self.export_btn.setEnabled(False)
        self.export_btn.setObjectName("primaryButton")
        self.export_label = QLabel("")
        self.summarize_btn = QPushButton(
            "Summarise session with local LLM")
        self.summarize_btn.setEnabled(False)
        self.summarize_label = QLabel(
            "Local LLM summaries are off unless enabled in Settings. "
            "Ollama must be running on localhost; otherwise this "
            "button explains that instead of failing.")
        self.summarize_label.setWordWrap(True)
        self._summarizing = False
        self._summarize_thread: QThread | None = None
        self._summarize_worker: _SummarizeWorker | None = None
        self.settings_btn = QPushButton("Settings")
        self.browser_btn = QPushButton("Recordings")
        # Theme toggle (Nov 21 2026): one button in the header,
        # toggling light/dark through the existing theme machinery
        # and persisting the choice. Its text names the theme it
        # will switch TO.
        self.theme_toggle_btn = QPushButton("")
        self.theme_toggle_btn.setToolTip(
            "Switch between the light and dark themes.")
        self._refresh_theme_toggle_text()
        # Session chooser + loaded-session line (Nov 10 2026): every
        # Process action works on the session picked here (default:
        # the newest), never on an implicit most-recent session. The
        # chooser and the Recording Browser's Open action stay in
        # sync through load_session().
        self.session_combo = QComboBox()
        self.session_combo.setToolTip(
            "The session the Process actions work on, and the "
            "session the Session view shows. Newest "
            "first; each row shows the session id and its status.")
        self.session_combo.currentIndexChanged.connect(
            self._on_session_chooser_changed)
        # Source editor (Nov 10 2026): the loaded session's
        # recording source, correctable after the fact. The combo
        # holds the explicit sources only (no Auto — an existing
        # session stores a concrete value); the detail edit holds
        # the free-text detail (app name, window title, note).
        # Both save to metadata.json on change (see
        # _on_session_source_changed / _on_session_source_detail_edited).
        self.session_source_combo = QComboBox()
        for value in SOURCES:
            self.session_source_combo.addItem(SOURCE_LABELS[value], value)
        self.session_source_combo.setToolTip(
            "What this session is a recording of. Correct it here "
            "when detection got it wrong — the change is saved to "
            "the session's metadata.json immediately.")
        self.session_source_combo.currentIndexChanged.connect(
            self._on_session_source_changed)
        self.session_source_detail_edit = QLineEdit()
        self.session_source_detail_edit.setPlaceholderText(
            "Detail — app name, window title, or note")
        self.session_source_detail_edit.setToolTip(
            "Free-text detail for the source — e.g. the app name "
            "or the video title. Saved when you press Enter or "
            "move to another field.")
        self.session_source_detail_edit.editingFinished.connect(
            self._on_session_source_detail_edited)
        self.loaded_session_label = QLabel("No session loaded yet.")
        self.loaded_session_label.setObjectName("loadedSessionLabel")
        self.loaded_session_label.setWordWrap(True)
        self.loaded_session_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.loaded_session_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        # Outlook calendar link (Nov 10 2026, Feature 3): the line
        # under the loaded-session line. Shows "Linked to:
        # <subject>, <time>" with an "Open in Outlook" button when
        # the session matched a calendar event, "No calendar event
        # found" when the match was attempted and found nothing, and
        # stays empty (button hidden) when no match was ever
        # attempted (linking disabled / Outlook unavailable / old
        # session not yet loaded through the link path).
        self.calendar_event_label = QLabel("")
        self.calendar_event_label.setObjectName("calendarEventLabel")
        self.calendar_event_label.setWordWrap(True)
        self.calendar_event_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.calendar_event_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.open_outlook_btn = QPushButton("Open in Outlook")
        self.open_outlook_btn.setToolTip(
            "Open the linked calendar event in Outlook (read-only — "
            "OffAir never writes anything into Outlook).")
        self.open_outlook_btn.setVisible(False)
        self.open_outlook_btn.clicked.connect(self._on_open_in_outlook)
        # Phase 3 — Speaker Name Panel + click-to-play segment list.
        self.segments_list = QListWidget()
        self.segments_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.segments_list.setToolTip(
            "Click a segment to play that part of the recording.")
        # Transcript cleanup (Nov 10 2026): Original shows the
        # verbatim transcript; Cleaned shows the rule-cleaned
        # derived artifact (transcript-clean.json). The verbatim
        # files are never modified — Re-clean regenerates the
        # derived artifact from them.
        # Transcript variants, from the cleanup registry — the
        # third variant (enterprise-cleaned, text only) appears
        # here explicitly labelled, never merged with the others.
        self.transcript_view_combo = QComboBox()
        for _variant in TRANSCRIPT_VARIANTS:
            self.transcript_view_combo.addItem(
                TRANSCRIPT_VARIANT_LABELS[_variant], _variant)
        self.transcript_view_combo.setToolTip(
            "Original shows the verbatim transcript; Cleaned shows "
            "the rule-cleaned version (fillers removed, names "
            "corrected, repeats kept verbatim). The verbatim "
            "transcript is never changed.")
        self.reclean_btn = QPushButton("Re-clean transcript")
        self.reclean_btn.setEnabled(False)
        self.reclean_btn.setToolTip(
            "Re-run the rule cleanup on the verbatim transcript "
            "(e.g. after a rename, or after editing the names "
            "dictionary). The verbatim transcript is untouched.")
        self.polish_btn = QPushButton("Polish with local LLM")
        self.polish_btn.setEnabled(False)
        self.polish_btn.setToolTip(
            "AI-rewrite the cleaned transcript with the local LLM "
            "(Ollama) into transcript-polished.md, clearly marked "
            "as an AI rewrite. Needs Ollama running; the verbatim "
            "and rule-cleaned transcripts stay in place.")
        self.cleanup_label = QLabel("")
        self.cleanup_label.setWordWrap(True)
        # Enterprise cleanup (Nov 21 2026): the button label itself
        # carries the text-leaves-machine warning; Cancel stops
        # the CLI run (no artifact is written on cancel).
        self.enterprise_btn = QPushButton(ENTERPRISE_BUTTON_TEXT)
        self.enterprise_btn.setEnabled(False)
        self.enterprise_btn.setToolTip(
            "Clean the transcript with the Claude Code CLI "
            "(`claude -p`). Only the transcript TEXT is sent — "
            "audio never leaves this machine. Opt-in: enable it "
            "in Settings first. The verbatim transcript is never "
            "changed; the result is a separate, clearly labelled "
            "file.")
        self.enterprise_cancel_btn = QPushButton("Cancel")
        self.enterprise_cancel_btn.setEnabled(False)
        self.enterprise_cancel_btn.setToolTip(
            "Stop the enterprise cleanup now — no artifact is "
            "written.")
        self._enterprise_running = False
        self._enterprise_thread: QThread | None = None
        self._enterprise_worker: _EnterpriseCleanupWorker | None = None
        self._enterprise_cancel_event: threading.Event | None = None
        # Waveform timeline (Nov 21 2026): RMS buckets computed in
        # a worker, cached per session (see _waveform_cache).
        self.waveform_widget = WaveformWidget()
        self._waveform_cache: dict[str, tuple[list, float]] = {}
        self._waveform_thread: QThread | None = None
        self._waveform_worker: _WaveformWorker | None = None
        # Every live waveform thread: several can be in flight
        # at once (startup load + an explicit reload), and ALL
        # of them must be waited for on close — destroying any
        # running QThread aborts the process.
        self._waveform_threads: set = set()
        # Bookmarks for the loaded session (Nov 21 2026).
        self.bookmarks_list = QListWidget()
        self.bookmarks_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.bookmarks_list.setToolTip(
            "Bookmarks for the loaded session — click one to hear "
            "that moment.")
        self.bookmarks_list.setMinimumHeight(80)
        self.add_bookmark_btn = QPushButton(
            "Add bookmark at current moment")
        self.add_bookmark_btn.setEnabled(False)
        self.add_bookmark_btn.setToolTip(
            "Mark the moment now playing, the selected segment, "
            "or the start of the session when nothing is playing.")
        self._polishing = False
        self._polish_quiet = False
        self._polish_thread: QThread | None = None
        self._polish_worker: _PolishWorker | None = None
        self.rename_label_combo = QComboBox()
        self.rename_name_edit = QLineEdit()
        self.rename_name_edit.setPlaceholderText("New name (blank clears it)")
        self.rename_btn = QPushButton("Rename speaker")
        self.rename_btn.setEnabled(False)
        self._player = None
        if MULTIMEDIA_AVAILABLE:
            self._player = QMediaPlayer(self)
            self._player.setAudioOutput(QAudioOutput(self))
        # Follow-along (Nov 21 2026): while playback is active the
        # segment being played is highlighted in the transcript
        # list and scrolled into view. positionChanged is the
        # primary signal; the poll timer is the fallback for
        # players that do not emit it reliably.
        self._follow_segment_index: int | None = None
        self._follow_timer = QTimer(self)
        self._follow_timer.setInterval(250)
        self._follow_timer.timeout.connect(self._poll_follow_along)
        if self._player is not None:
            self._player.positionChanged.connect(
                self._on_playback_position)
            self._player.playbackStateChanged.connect(
                self._on_playback_state_changed)
        # AUTO badge — visible only while an auto-started recording
        # runs, so an auto-recording is unmissable (the status pill
        # also turns red; this says *why* it started).
        self.auto_badge = QLabel("AUTO")
        self.auto_badge.setObjectName("autoBadge")
        self.auto_badge.setVisible(False)
        self.auto_badge.setToolTip(
            "This recording was started automatically — a Teams "
            "meeting was detected. Stop ends it; auto-detect will "
            "not restart it until the meeting ends.")

        # ------------------------------------------------ layout (Nov 19 2026)
        # Header: app name, status, recording timer, Settings/Recordings.
        header = QFrame()
        header.setObjectName("headerBar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(12)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        app_title = QLabel("OffAir")
        app_title.setObjectName("appTitle")
        app_subtitle = QLabel(
            "Local meeting recorder — CPU-only, nothing leaves this machine")
        app_subtitle.setObjectName("appSubtitle")
        app_subtitle.setWordWrap(True)
        title_box.addWidget(app_title)
        title_box.addWidget(app_subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch(1)
        header_layout.addWidget(self.status_label)
        header_layout.addWidget(self.elapsed_label)
        header_layout.addWidget(self.theme_toggle_btn)
        header_layout.addWidget(self.settings_btn)
        header_layout.addWidget(self.browser_btn)
        layout.addWidget(header)

        # Body: left navigation + one focused view per task. The
        # views are built by app/ui/views/ from the widgets above;
        # this shell owns the nav, the stack and the shared state.
        from app.ui.views import (
            build_calendar_view,
            build_help_view,
            build_library_view,
            build_mcp_view,
            build_people_view,
            build_record_view,
            build_session_view,
            build_settings_view,
        )

        body = QHBoxLayout()
        body.setSpacing(16)
        self.nav_list = QListWidget()
        self.nav_list.setObjectName("navList")
        self.nav_list.setFixedWidth(176)
        self.nav_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The nav rows are the view names, in stack order — one row
        # per view, every row selectable (Nov 20 2026: the "More"
        # section header is gone, so MCP setup and Help are rows
        # like the rest).
        for _name in VIEW_NAMES:
            _item = QListWidgetItem(_name)
            _item.setData(Qt.ItemDataRole.UserRole, _name)
            self.nav_list.addItem(_item)
        body.addWidget(self.nav_list)

        self._views: dict[str, QWidget] = {}
        self._views["Record"] = build_record_view(self)
        self._views["Library"] = build_library_view(self)
        self._views["Calendar"] = build_calendar_view(self)
        self._views["Session"] = build_session_view(self)
        self._views["People"] = build_people_view(self)
        self._views["Settings"] = build_settings_view(self)
        self._views["MCP setup"] = build_mcp_view(self)
        self._views["Help"] = build_help_view(self)

        stack = _NavStack()
        stack._nav_names = VIEW_NAMES
        # ``tabs`` stays the public name for the view container:
        # tests and screenshots address views by name through it
        # (``win.tabs.tabText(i)`` / ``setCurrentIndex``), as they
        # did the old QTabWidget.
        self.tabs = stack
        for _name in VIEW_NAMES:
            stack.addWidget(self._views[_name])
        body.addWidget(stack, 1)
        layout.addLayout(body, 1)
        self.setCentralWidget(central)

        self.nav_list.currentRowChanged.connect(
            self._on_nav_row_changed)
        self.tabs.currentChanged.connect(self._on_view_changed)
        # Record is the landing view: recording is the job the
        # window opens for.
        self.nav_list.setCurrentRow(0)
        self.tabs.setCurrentIndex(0)

        self.process_combo.currentIndexChanged.connect(
            self._on_app_picker_changed)
        self.record_btn.clicked.connect(self.on_record)
        self.pause_btn.clicked.connect(self.on_pause)
        self.resume_btn.clicked.connect(self.on_resume)
        self.stop_btn.clicked.connect(self.on_stop)
        self.refresh_btn.clicked.connect(self.refresh_devices)
        self.transcribe_btn.clicked.connect(self.on_transcribe)
        self.cancel_transcribe_btn.clicked.connect(
            self.on_cancel_transcribe)
        self.diarize_btn.clicked.connect(self.on_diarize)
        self.rename_btn.clicked.connect(self.on_rename_speaker)
        self.play_known_btn.clicked.connect(self.on_play_known_sample)
        self.rename_known_btn.clicked.connect(self.on_rename_known_speaker)
        self.play_suggestion_btn.clicked.connect(
            self.on_play_suggestion_sample)
        self.known_speakers_list.itemDoubleClicked.connect(
            self.on_rename_known_speaker)
        self.suggestions_list.itemDoubleClicked.connect(
            self.on_play_suggestion_sample)
        self.known_speakers_list.itemSelectionChanged.connect(
            self._update_buttons)
        self.suggestions_list.itemSelectionChanged.connect(
            self._update_buttons)
        self.suggest_btn.clicked.connect(self.on_suggest)
        self.confirm_btn.clicked.connect(self.on_confirm_suggestion)
        self.correct_btn.clicked.connect(self.on_correct_suggestion)
        self.nearlive_btn.clicked.connect(self.on_nearlive)
        self.segments_list.itemClicked.connect(self.on_segment_clicked)
        self.transcript_view_combo.currentIndexChanged.connect(
            self._on_transcript_view_changed)
        self.reclean_btn.clicked.connect(self.on_reclean)
        self.polish_btn.clicked.connect(self.on_polish)
        self.enterprise_btn.clicked.connect(self.on_enterprise_cleanup)
        self.enterprise_cancel_btn.clicked.connect(
            self._on_enterprise_cancel)
        self.export_btn.clicked.connect(self.on_export_llm)
        self.summarize_btn.clicked.connect(self.on_summarize_local)
        self.settings_btn.clicked.connect(self.on_open_settings)
        self.browser_btn.clicked.connect(self.on_open_browser)
        self.theme_toggle_btn.clicked.connect(self._on_theme_toggle)
        self.dictation_btn.toggled.connect(self._on_dictation_toggled)
        self.bookmark_btn.clicked.connect(self._on_bookmark_button)
        self.bookmarks_list.itemClicked.connect(
            self._on_bookmark_item_clicked)
        self.add_bookmark_btn.clicked.connect(
            self._on_add_bookmark_button)
        self.waveform_widget.time_clicked.connect(
            self._on_waveform_clicked)
        # Device changes restart the pre-roll feeder with the new
        # indices (debounced — refresh_devices fires this signal
        # repeatedly while repopulating). Guarded by
        # _preroll_ready, which is False until the feeder is first
        # set up, so startup population never restarts it.
        self.mic_combo.currentIndexChanged.connect(
            self._on_preroll_device_changed)
        self.loopback_combo.currentIndexChanged.connect(
            self._on_preroll_device_changed)

        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._update_elapsed)

        # Light is the default (the clean, airy consumer look);
        # main.py also applies it at the QApplication level (Phase 6
        # fix) so the Browser and Settings windows follow it; this
        # keeps a standalone-created window themed too.
        try:
            apply_theme(self, self.config.theme)
        except Exception:  # noqa: BLE001 — a bad theme must not block startup
            logger.warning("Could not apply theme %r", self.config.theme,
                           exc_info=True)
        self.mic_meter.set_theme(self.config.theme)
        self.system_meter.set_theme(self.config.theme)
        self.refresh_devices()
        # Pre-roll feeder: built after the device combos are
        # populated (its indices come from them) and before the
        # auto-detect watcher, which needs the same buffer.
        self._setup_preroll_feeder()
        self._refresh_known_speakers()
        self._update_diarize_mode_ui()
        self._update_buttons()
        self._show_empty_states()
        self.statusBar().showMessage(
            f"Output: {self.config.output_root}")
        # Auto-detect (opt-in, default off): start the watcher after
        # the window loads when the setting is on.
        self._apply_auto_detect_setting()
        # Audio retention (Nov 8 2026): prune transcribed sessions'
        # WAVs older than the retention age — on startup, after a
        # session finishes auto-processing, and hourly. Cheap and
        # synchronous (a directory walk); a failure is logged, never
        # raised into the UI.
        self._retention_timer = QTimer(self)
        self._retention_timer.setInterval(60 * 60 * 1000)
        self._retention_timer.timeout.connect(self._run_retention_prune)
        self._retention_timer.start()
        self._run_retention_prune()
        # Scheduled recordings (Nov 22 2026): anything whose time
        # passed while the app was closed expires now — it never
        # fires late — then the 2 s scheduler timer takes over
        # (see _check_schedules). A store failure here is logged,
        # never raised into startup.
        self._refresh_schedule_app_combo()
        try:
            expire_past_due(path=self._schedules_path)
        except Exception:  # noqa: BLE001 — schedules must not break startup
            logger.exception("Expiring past-due schedules failed")
        self._reload_schedules(force=True)
        self._refresh_schedule_list()
        self._schedule_timer = QTimer(self)
        self._schedule_timer.setInterval(2000)
        self._schedule_timer.timeout.connect(self._check_schedules)
        self._schedule_timer.start()
        # Session library default: load the most recent session, so
        # the window always shows which session its views are on.
        # (A fresh install has no sessions, and nothing is loaded.)
        # This runs AFTER the startup retention prune on purpose:
        # the prune excludes the loaded ("live") session, so loading
        # first would shield the newest session's old WAVs from it.
        self._refresh_settings_view()
        self._refresh_session_chooser()
        _newest = self.session_combo.currentData()
        if isinstance(_newest, str) and _newest:
            self.load_session(Path(_newest))
        # Tray, shortcuts, onboarding: all degrade silently when
        # their platform support is absent; none of them may
        # block startup.
        self._setup_tray()
        self._setup_shortcuts()
        QTimer.singleShot(0, self._maybe_show_onboarding)

    def _run_retention_prune(self) -> None:
        """Delete old transcribed sessions' WAVs (transcripts kept).

        The live session is always excluded, and retention.py itself
        refuses sessions that are still recording or untranscribed —
        so this is safe to run on a timer and after processing.
        """
        try:
            result = prune_old_audio(
                self.config.output_root,
                self.config.audio_retention_hours,
                exclude_session_dirs=(
                    (self.session_dir,) if self.session_dir else ()))
        except Exception:  # noqa: BLE001 — pruning must never break the UI
            logger.exception("Audio retention prune failed")
            return
        if result["files_deleted"]:
            freed_mb = result["bytes_freed"] / (1024 * 1024)
            self.statusBar().showMessage(
                f"Deleted {result['files_deleted']} old audio file(s) "
                f"({freed_mb:.0f} MB freed) — transcripts kept.")

    # ------------------------------------------------------------ devices ---
    def _update_diarize_mode_ui(self) -> None:
        """Show the right guidance for the selected diarization mode.

        Sherpa is the default and needs nothing but its models
        (downloaded once on first use). Full: the pyannote warning
        + HF token field. Simple: no model, no download.

        The fallback is the configured diarization mode — never a
        hard-coded "simple": if the combo ever has no selection
        (index -1, currentData() None), the mode that runs must
        still be the configured default, not a silent downgrade.
        """
        mode = (self.diarize_mode_combo.currentData()
                or self.config.diarization_mode)
        info = DIARIZATION_MODES.get(mode, DIARIZATION_MODES["simple"])
        self.diarize_warning_label.setText(info["description"])
        if mode == "full":
            self.diarize_warning_label.setText(
                "Full mode loads pyannote on CPU and is much slower — "
                "minutes for a 30-minute meeting is normal — and needs "
                "a Hugging Face token plus accepted model licences "
                "(see README, Phase 3).")
            self.hf_token_edit.setEnabled(True)
            self.sherpa_status_label.setVisible(False)
            return
        self.hf_token_edit.setEnabled(False)
        if mode != "sherpa":
            self.sherpa_status_label.setVisible(False)
            return
        try:
            avail = check_availability(self.config.sherpa_model_dir)
        except Exception:  # noqa: BLE001 — availability must not crash UI
            avail = {"installed": False, "models_present": False,
                     "ready": False}
        if avail.get("ready"):
            text = ("✓ Sherpa ready — sherpa-onnx is installed and the "
                    "diarization models are present. No token needed.")
        elif avail.get("installed"):
            text = ("Sherpa models (~33 MB) will download once on first "
                    "use from public GitHub releases — no account, no "
                    "token, no login — then diarization works offline.")
        else:
            text = sherpa_install_guide()
        self.sherpa_status_label.setText(text)
        self.sherpa_status_label.setVisible(True)

    def refresh_devices(self) -> None:
        # One honest per-platform capture description (Track 3):
        # device lists + status + install hint all come from
        # describe_capture_state(), so Linux never sees the old
        # Windows-only placeholder text.
        info = describe_capture_state()
        self.mic_combo.clear()
        self.loopback_combo.clear()
        for mic in info.microphones:
            self.mic_combo.addItem(mic.display_name(), mic.index)
            self.mic_combo.setItemData(
                self.mic_combo.count() - 1, mic.display_name(),
                Qt.ItemDataRole.ToolTipRole)
        for device in info.loopback_devices:
            self.loopback_combo.addItem(device.display_name(), device.index)
            self.loopback_combo.setItemData(
                self.loopback_combo.count() - 1, device.display_name(),
                Qt.ItemDataRole.ToolTipRole)
        if not info.microphones:
            # Short placeholder in the (fixed-width) combo — the full
            # cause lives in the tooltip, so the text never truncates.
            self.mic_combo.addItem("(no microphone found)", None)
            self.mic_combo.setToolTip(mic_empty_message(info))
        if not info.loopback_devices:
            self.loopback_combo.addItem("(no system audio found)", None)
            self.loopback_combo.setToolTip(loopback_empty_message(info))
        default_mic = default_microphone()
        default_loop = default_loopback()
        if default_mic is not None:
            self.mic_combo.setCurrentIndex(
                max(0, self._index_of(self.mic_combo, default_mic.index)))
        if default_loop is not None:
            self.loopback_combo.setCurrentIndex(
                max(0, self._index_of(self.loopback_combo, default_loop.index)))
        self._update_device_tooltips()
        # The target-app picker refreshes from live audio sessions
        # (the hand-typed exe-name list is gone — see
        # _refresh_app_picker).
        self._refresh_app_picker()
        process = find_teams_process(default_process_names())
        teams_note = (f"Teams running (pid {process['pid']})."
                      if process else "Teams is not currently running.")
        status = info.status
        if not status.endswith((".", "!", "?")):
            status += "."
        # Device status + Teams note live in the mode label's
        # tooltip now — the label itself is the capture-mode
        # explanation (see _update_capture_mode_ui).
        detail = f"{status} {teams_note}"
        if info.detail:
            detail += "\n" + info.detail
        if info.install_hint:
            detail = detail + "\n" + info.install_hint if detail else \
                info.install_hint
        # Per-app capture availability (Windows 10 build 20348+,
        # pycaw installed): one line in the tooltip so the user knows
        # which system-audio mode "auto" will pick.
        per_app_available, per_app_reason = per_app_capture_status()
        if per_app_available:
            detail += ("\nPer-app Teams capture: available — the system "
                       "track records Teams only (Settings → System "
                       "audio capture).")
        else:
            detail += ("\nPer-app Teams capture unavailable: "
                       f"{per_app_reason}")
        self.mode_label.setToolTip(detail)
        self._update_capture_mode_ui()

    def _update_device_tooltips(self) -> None:
        """Device combos show the full device name on hover.

        The Record card is fixed-width, so a long device name (e.g.
        "Speakers (High Definition Audio Device) [Loopback]") is
        elided in the combo itself — the tooltip carries the full
        name (per-item tooltips do the same in the dropdown; see
        refresh_devices). The "(no device found)" placeholders keep
        their explanatory tooltip instead.
        """
        for combo in (self.mic_combo, self.loopback_combo):
            text = combo.currentText()
            if text and not text.startswith("("):
                combo.setToolTip(text)

    # --------------------------------------------------- app picker ---
    def _refresh_app_picker(self) -> None:
        """Rebuild the target-app picker from live audio sessions.

        Rows are ``"<Friendly name> — playing sound"`` / ``"— idle"``
        with the exe name in UserRole; the previous selection is
        kept when the app is still listed, otherwise the picker
        prefers a Teams session, then the first session (Active
        ones sort first). With no sessions at all the list is a
        placeholder plus "Other app…" — the picker never pretends
        an app is there. A manually typed exe (from "Other app…")
        is kept as its own row until it gains a session of its own.
        Never raises: a failed enumeration leaves the previous list
        in place.
        """
        try:
            sessions = list_audio_sessions()
        except Exception:  # noqa: BLE001 — picker must not break startup
            logger.exception("App picker refresh failed")
            return
        # Active sessions first, then by name — the lister's order
        # is not trusted here: the picker itself guarantees the
        # app currently making sound is at the top.
        sessions = sorted(
            sessions,
            key=lambda s: (s.state != "active",
                           s.display_name.lower()))
        previous_exe = self.selected_process_name()
        self.process_combo.blockSignals(True)
        try:
            self.process_combo.clear()
            for session in sessions:
                state_text = ("playing sound"
                              if session.state == "active" else "idle")
                self.process_combo.addItem(
                    f"{session.display_name} — {state_text}",
                    session.exe_name)
            if (self._manual_process_name
                    and not any(s.exe_name.lower()
                                == self._manual_process_name.lower()
                                for s in sessions)):
                self.process_combo.addItem(
                    f"{display_name_for_exe(self._manual_process_name)}"
                    " — manual entry",
                    self._manual_process_name)
            if not sessions and not self._manual_process_name:
                self.process_combo.addItem(
                    "(no apps with audio sessions)", None)
            self.process_combo.addItem("Other app…", OTHER_APP_SENTINEL)
            target = self._picker_index_for(previous_exe)
            if target is None and sessions:
                for wanted in self.config.process_names:
                    target = self._picker_index_for(wanted)
                    if target is not None:
                        break
            if target is None:
                target = 0
            self.process_combo.setCurrentIndex(target)
            self._picker_last_exe = self.selected_process_name()
        finally:
            self.process_combo.blockSignals(False)
        self._update_capture_mode_ui()

    def _picker_index_for(self, exe_name: str | None) -> int | None:
        """Index of the picker row for ``exe_name``, or None."""
        if not exe_name:
            return None
        for i in range(self.process_combo.count()):
            data = self.process_combo.itemData(i)
            if (isinstance(data, str)
                    and data.lower() == exe_name.lower()):
                return i
        return None

    def selected_process_name(self) -> str | None:
        """Exe name the picker currently targets, or None.

        A session row returns its exe name; "Other app…" returns
        the manually typed exe name (if any); the empty-state
        placeholder returns None (no target picked).
        """
        data = self.process_combo.currentData()
        if data == OTHER_APP_SENTINEL:
            return self._manual_process_name
        if isinstance(data, str) and data:
            return data
        return None

    def _on_app_picker_changed(self, index: int) -> None:
        """Handle a picker change — incl. the "Other app…" dialog.

        Picking "Other app…" asks for an exe name; cancelling (or
        an empty name) reverts to the previous row instead of
        leaving the picker on a row with no target. Any other
        change updates the capture-mode explanation to name the
        new app.
        """
        data = self.process_combo.itemData(index)
        if data == OTHER_APP_SENTINEL:
            text, ok = QInputDialog.getText(
                self, "Other app",
                "Exe name of the app to record (e.g. chrome.exe):")
            if ok and text.strip():
                self._manual_process_name = text.strip()
                # Rebuild: the manual entry becomes a row of its own
                # and the restore logic selects it.
                self._refresh_app_picker()
            else:
                self.process_combo.blockSignals(True)
                try:
                    restored = self._picker_index_for(
                        self._picker_last_exe)
                    self.process_combo.setCurrentIndex(
                        restored if restored is not None else 0)
                finally:
                    self.process_combo.blockSignals(False)
        else:
            self._picker_last_exe = (
                data if isinstance(data, str) else None)
        self._update_capture_mode_ui()

    # -------------------------------------------------- pre-roll (Nov 21) --
    def _preroll_device_indices(self) -> tuple[int | None, int | None]:
        """Device indices for the feeder, from the Settings combos.

        ``currentData()`` is the device index the combos were
        populated with; anything else (``None`` or a non-integer)
        means "the defaults" for that track.
        """
        def _index(combo) -> int | None:
            try:
                value = combo.currentData()
            except Exception:  # noqa: BLE001 — defaults are safe
                return None
            return value if isinstance(value, int) else None

        return (_index(self.mic_combo),
                _index(self.loopback_combo))

    def _setup_preroll_feeder(self) -> None:
        """(Re)build the pre-roll buffer + feeder from the config.

        Called once after the device combos are populated, and
        again after a settings save or a device change. A feeder
        that is already running is stopped first (synchronously)
        and never restarted while a recording is in progress.
        Never raises: a pre-roll failure degrades silently (log
        only) and recording works exactly as before.
        """
        try:
            self._stop_preroll_feeder(clear=False)
            mic_index, loopback_index = self._preroll_device_indices()
            self._preroll_buffer = PreRollBuffer(
                seconds=self.config.preroll_seconds,
                sample_rate=SAMPLE_RATE)
            self._preroll_feeder = PreRollFeeder(
                self._preroll_buffer,
                enabled=self.config.preroll_enabled,
                sample_rate=SAMPLE_RATE,
                mic_device_index=mic_index,
                loopback_device_index=loopback_index)
            # The watcher drains this same buffer at the start
            # gate — keep its reference current when the buffer
            # is rebuilt (only the real watcher takes one; the
            # test factory takes no such argument).
            watcher = getattr(self, "_auto_watcher", None)
            if isinstance(watcher, MeetingWatcher):
                watcher._preroll_buffer = self._preroll_buffer
            self._preroll_ready = True
            if not self.capture and self._preroll_feeder is not None:
                self._preroll_feeder.start()
        except Exception:  # noqa: BLE001 — silent degradation
            logger.exception("Pre-roll feeder setup failed")
            self._preroll_buffer = None
            self._preroll_feeder = None

    def _stop_preroll_feeder(self, clear: bool = True) -> None:
        """Stop the feeder and (by default) discard buffered audio.

        Used on toggle-off / shutdown. ``clear=False`` keeps the
        buffer for a rebuild that replaces it immediately anyway.
        """
        feeder = self._preroll_feeder
        if feeder is not None:
            try:
                feeder.stop()
            except Exception:  # noqa: BLE001 — stop never raises
                logger.exception("Pre-roll feeder stop failed")
        if clear and self._preroll_buffer is not None:
            try:
                self._preroll_buffer.clear()
            except Exception:  # noqa: BLE001 — clearing is best-effort
                logger.exception("Pre-roll buffer clear failed")

    def _restart_preroll_feeder(self) -> None:
        """Restart the feeder after capture stopped / failed start.

        Contract: after ``DualCapture.stop()`` (or a failed start)
        the feeder starts again if still enabled. Never while a
        recording is running, and never raising.
        """
        if self.capture is not None:
            return
        feeder = self._preroll_feeder
        if feeder is None or not self.config.preroll_enabled:
            return
        try:
            feeder.start()
        except Exception:  # noqa: BLE001 — silent degradation
            logger.exception("Pre-roll feeder restart failed")

    def _take_preroll_samples(self, auto: bool = False) -> dict:
        """Pre-roll handoff, in the contract's exact order.

        The feeder is stopped first (synchronously — its streams
        must be closed before capture opens its own). Manual
        Record drains the buffer directly; an auto-detect start
        takes what the watcher drained at the start gate, then
        clears the buffer. Returns ``{}`` when pre-roll is off,
        the buffer is empty, or anything fails.
        """
        try:
            feeder = self._preroll_feeder
            if feeder is not None:
                feeder.stop()
            buffer = self._preroll_buffer
            if auto:
                watcher = getattr(self, "_auto_watcher", None)
                take = getattr(watcher, "take_preroll", None)
                samples = take() if callable(take) else {}
                if buffer is not None:
                    buffer.clear()
                return dict(samples or {})
            if buffer is None:
                return {}
            return dict(buffer.drain_all())
        except Exception:  # noqa: BLE001 — no pre-roll, no failure
            logger.exception("Pre-roll handoff failed")
            return {}

    def _on_preroll_device_changed(self, *_args) -> None:
        """A device combo changed: restart the feeder (debounced).

        Guarded by ``_preroll_ready`` so the population of the
        combos during startup never restarts anything, and never
        while recording. The debounce absorbs the flurry of
        signals ``refresh_devices`` fires while repopulating.
        """
        if not self._preroll_ready:
            return
        if self.capture is not None:
            return
        try:
            self._preroll_device_timer.stop()
        except AttributeError:
            self._preroll_device_timer = QTimer(self)
            self._preroll_device_timer.setSingleShot(True)
            self._preroll_device_timer.setInterval(500)
            self._preroll_device_timer.timeout.connect(
                self._setup_preroll_feeder)
        self._preroll_device_timer.start()

    def _on_preroll_toggled(self, checked: bool) -> None:
        """Pre-roll checkbox in the advanced Settings section.

        Enables the seconds spin; the feeder itself is rebuilt
        when the form is saved (the dirty-check snapshot carries
        the checkbox + spin values — see _settings_form_dirty).
        """
        self.preroll_seconds_spin.setEnabled(checked)

    # ------------------------------------------------------ disk (Nov 21) --
    def _check_disk_while_recording(self) -> None:
        """30 s low-disk poll while recording (never stops one).

        First crossing fires a status message + ONE dialog per
        recording (``_disk_warned`` resets in on_record). The
        dialog says the recording continues; the user decides
        what to do.
        """
        if self.capture is None or self._disk_warned:
            return
        try:
            status = disk_status(
                self.config.output_root,
                threshold_bytes=self.config.disk_warning_threshold_bytes)
        except Exception:  # noqa: BLE001 — a failed poll is silent
            logger.exception("Disk status poll failed")
            return
        if not status.low_disk:
            return
        self._disk_warned = True
        message = (
            f"Low disk space: {status.free_gb:.1f} GB free on the "
            "recordings volume. The recording continues — stop it "
            "when you are ready, or free space. Old audio is only "
            "deleted by the retention rule in Settings.")
        self.statusBar().showMessage(message)
        # Non-modal on purpose: a modal dialog here blocks the UI
        # thread while the recording runs, and hangs headless /
        # offscreen runs forever — the dialog can never be
        # dismissed there, so the first low-disk recording in a
        # test run wedged the whole suite. The warning still
        # appears once per recording; the user dismisses it at
        # leisure and the recording is never stopped by it.
        if self._disk_warning_box is not None:
            self._disk_warning_box.close()
        box = QMessageBox(
            QMessageBox.Icon.Warning, "Low disk space", message,
            QMessageBox.StandardButton.Ok, self)
        box.setModal(False)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.show()
        self._disk_warning_box = box

    def _refresh_disk_line(self) -> None:
        """Library footer + Settings line: free space + prunable bytes.

        Both labels say the same thing in the same words; the
        scan deletes nothing. A failed scan leaves the previous
        text (or a plain "unavailable" on first run) — the disk
        line must never break the Library.
        """
        try:
            status = disk_status(
                self.config.output_root,
                threshold_bytes=self.config.disk_warning_threshold_bytes)
            session_dir = getattr(self, "session_dir", None)
            scan = scan_prunable_audio(
                self.config.output_root,
                self.config.audio_retention_hours,
                exclude_session_dirs=(
                    (session_dir,) if session_dir else ()))
            threshold = _format_bytes(
                self.config.disk_warning_threshold_bytes)
            text = (
                f"Free space: {_format_bytes(status.free_bytes)} "
                f"(warning below {threshold}). Retention would "
                f"delete {_format_bytes(scan['bytes_prunable'])} in "
                f"{scan['files_prunable']} file(s) from "
                f"{scan['sessions_prunable']} session(s).")
        except Exception:  # noqa: BLE001 — disk line is best-effort
            logger.exception("Disk line refresh failed")
            text = "Free space: unavailable."
        if getattr(self, "disk_status_label", None) is not None:
            self.disk_status_label.setText(text)
        if getattr(self, "settings_disk_label", None) is not None:
            self.settings_disk_label.setText(text)

    # ------------------------------------------------------ tray (Nov 21) --
    def _setup_tray(self) -> None:
        """System tray icon, only when a tray is available.

        Silent degrade otherwise: ``self._tray`` stays None and
        the main window remains the only stop path. Stop in the
        menu calls the same :meth:`on_stop` path as the window's
        Stop button — never a second implementation.
        """
        try:
            if not QSystemTrayIcon.isSystemTrayAvailable():
                return
            icon = self.windowIcon()
            tray = QSystemTrayIcon(icon, self)
            menu = QMenu()
            show_action = QAction("Show OffAir", self)
            show_action.triggered.connect(self._on_tray_show)
            stop_action = QAction("Stop recording", self)
            stop_action.triggered.connect(self._on_tray_stop)
            quit_action = QAction("Quit", self)
            quit_action.triggered.connect(self.close)
            menu.addAction(show_action)
            menu.addAction(stop_action)
            menu.addSeparator()
            menu.addAction(quit_action)
            tray.setContextMenu(menu)
            tray.activated.connect(self._on_tray_activated)
            tray.setToolTip("OffAir")
            tray.show()
            self._tray = tray
            self._tray_stop_action = stop_action
            self._tray_show_action = show_action
        except Exception:  # noqa: BLE001 — tray is best-effort
            logger.exception("System tray setup failed")
            self._tray = None

    def _on_tray_show(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_tray_stop(self) -> None:
        self.on_stop()

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._on_tray_show()

    # ------------------------------------------------- shortcuts (Nov 21) --
    def _setup_shortcuts(self) -> None:
        """In-app keyboard shortcuts (see docs/KEYBOARD_SHORTCUTS.md).

        Kept on ``self`` so they stay alive. In-app only — no
        global hotkey (that would need new dependencies). The
        chords were verified against the codebase before they
        were chosen: no existing QShortcut / Ctrl shortcut in
        ``app/`` or the UI tests.
        """
        try:
            self._bookmark_shortcut = QShortcut(
                QKeySequence("Ctrl+B"), self)
            self._bookmark_shortcut.activated.connect(
                self._on_bookmark_shortcut)
            self._pause_shortcut = QShortcut(
                QKeySequence("Ctrl+P"), self)
            self._pause_shortcut.activated.connect(
                self._on_pause_resume_shortcut)
            self._search_shortcut = QShortcut(
                QKeySequence("Ctrl+K"), self)
            self._search_shortcut.activated.connect(
                self._on_jump_to_search)
        except Exception:  # noqa: BLE001 — shortcuts are best-effort
            logger.exception("Keyboard shortcut setup failed")

    def _on_bookmark_shortcut(self) -> None:
        """Ctrl+B: same as the Bookmark button, when it applies.

        While recording it marks the live moment; with a session
        loaded and stopped it marks the loaded session's current
        moment; otherwise it does nothing (no dialog, no error).
        """
        if self.bookmark_btn.isEnabled():
            self._on_bookmark_button()
        elif self.add_bookmark_btn.isEnabled():
            self._on_add_bookmark_button()

    def _on_pause_resume_shortcut(self) -> None:
        """Ctrl+P: pause when recording, resume when paused."""
        if self.pause_btn.isEnabled():
            self.on_pause()
        elif self.resume_btn.isEnabled():
            self.on_resume()

    def _on_jump_to_search(self) -> None:
        """Ctrl+K: open the Library and focus its search box."""
        self.set_view("Library")
        self.library_search_edit.setFocus()
        self.library_search_edit.selectAll()

    # ------------------------------------------------ onboarding (Nov 21) --
    def _maybe_show_onboarding(self) -> None:
        """First-launch modal, once, deferred past startup.

        The marker is written BEFORE the dialog shows, so a
        crash or a dismissed dialog can never re-show it. The
        dialog is non-modal and parentless: a modal dialog owned
        by the window would block the offscreen Qt tests (and
        any automation) on first launch. The same content lives
        in Help → Getting started (docs/OVERVIEW.md), which is
        the durable, re-readable copy.
        """
        if _onboarding_shown():
            return
        _mark_onboarding_shown()
        self._show_onboarding_dialog()

    def _show_onboarding_dialog(self) -> None:
        from PyQt6.QtWidgets import QDialogButtonBox, QPlainTextEdit

        dialog = QDialog()
        dialog.setWindowTitle("Welcome to OffAir")
        dialog.setMinimumSize(520, 420)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit(ONBOARDING_TEXT)
        text.setReadOnly(True)
        layout.addWidget(text)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.close)
        buttons.accepted.connect(dialog.close)
        layout.addWidget(buttons)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._onboarding_dialog = dialog
        dialog.show()

    # ----------------------------------------------------- theme (Nov 21) --
    def _refresh_theme_toggle_text(self) -> None:
        """The toggle's text names the theme it will switch TO."""
        target = "Light" if self.config.theme == "dark" else "Dark"
        self.theme_toggle_btn.setText(f"{target} theme")

    def _on_theme_toggle(self) -> None:
        """One button, light ↔ dark, through the existing machinery.

        The choice persists in settings like any other: the new
        config is saved, then applied (window + app + meters),
        exactly as a Settings save would. No theme code changed.
        """
        from dataclasses import replace

        new_theme = "light" if self.config.theme == "dark" else "dark"
        new_config = replace(self.config, theme=new_theme)
        try:
            save_settings(new_config)
        except Exception:  # noqa: BLE001 — a failed save is reported
            logger.exception("Failed to save the theme choice")
            QMessageBox.warning(
                self, "Theme not saved",
                "The theme changed for this session, but the choice "
                "could not be saved — it will revert next launch.")
        self._apply_saved_config(new_config)

    # ------------------------------------------------- dictation (Nov 21) --
    def _on_dictation_toggled(self, checked: bool) -> None:
        """Dictation toggle: hide the app side, mic only.

        When on, the app picker and the system meter (and its
        label) hide — there is no other side to capture — the
        mode label says dictation, and the spoken-side-only note
        shows. When off, everything returns. The toggle itself
        is disabled while recording (see _update_buttons).
        """
        self.app_picker_label.setVisible(not checked)
        self.process_combo.setVisible(not checked)
        self.system_level_label.setVisible(not checked)
        self.system_meter.setVisible(not checked)
        self.dictation_note_label.setVisible(checked)
        if checked:
            self.mode_label.setText("Dictation — mic only")
        self._update_capture_mode_ui()
        self._update_buttons()

    def _refresh_dictation_note(self) -> None:
        """Session-view note: shown only for dictation sessions."""
        is_dictation = False
        if self.session_dir is not None:
            try:
                is_dictation = is_dictation_session(self.session_dir)
            except Exception:  # noqa: BLE001 — a bad read hides it
                logger.exception("Dictation check failed")
        self.session_dictation_note_label.setVisible(is_dictation)

    def _update_capture_mode_ui(self) -> None:
        """Update the capture-mode explanation label.

        The mode label is the one-line explanation of what the
        capture setting does, naming the picked app for "Record
        one app". There is no mode control on the Record view
        anymore (Nov 20 2026 simplification — the Settings
        dialog's combo is the only editor); this is feedback, not
        a setting. A stored "per-app" value (set in Settings)
        shows as "Record one app" — the stricter fail-loudly
        variant stays available in Settings. Called after
        refreshes and picker changes; a recording overwrites the
        label with what actually ran (see on_record / on_stop).
        Dictation (Nov 21 2026): while the toggle is on and
        nothing is recording, the label says dictation — the
        capture-mode explanation does not apply to a mic-only
        session.
        """
        if (self.dictation_btn.isChecked()
                and self.capture is None):
            self.mode_label.setText("Dictation — mic only")
            return
        mode = self.config.capture_system_mode
        index = 0 if mode in ("auto", "per-app") else 1
        if index == 0:
            exe = self.selected_process_name()
            app_name = (display_name_for_exe(exe) if exe
                        else "the chosen app")
            # Two lines at the card's width — a third line would be
            # clipped: the card has ~11 px of vertical slack.
            self.mode_label.setText(
                f"Capture: one app — records only {app_name}'s audio. "
                "Falls back to everything, with a warning, if it "
                "can't.")
        else:
            self.mode_label.setText(
                "Capture: everything you hear — records all sound "
                "from your speakers, every app at once.")

    # ------------------------------------------------- silence warnings ---
    def _target_has_audio_session(self, exe_name: str) -> bool:
        """True when ``exe_name`` has a render audio session now.

        A probe failure returns True (fail open — a broken probe
        is not evidence the app is silent, and must never block a
        recording the user asked for).
        """
        try:
            sessions = list_audio_sessions()
        except Exception:  # noqa: BLE001 — probe failure fails open
            logger.exception("Target-session probe failed")
            return True
        return find_session(sessions, exe_name) is not None

    def _confirm_record_without_session(self, display_name: str) -> bool:
        """Loud warning: the target app has no audio session.

        Returns True to record anyway (system-wide) or False to
        cancel. This is the dialog the work-PC failure needed: a
        target with no session records digital silence in per-app
        mode, so the user chooses explicitly instead of finding
        out afterwards.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(f"No audio from {display_name}")
        box.setText(
            f"{display_name} has no audio session right now — it "
            "may not be running, or not playing any sound.")
        box.setInformativeText(
            "Recording now would capture silence on the system "
            "track in “Record one app” mode. Record anyway to "
            "capture everything you hear instead, or Cancel and "
            "start the app / play sound first.")
        record_anyway = box.addButton(
            "Record anyway (system-wide)",
            QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(
            "Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is record_anyway

    def _note_level(self, kind: str, value: float) -> None:
        """Remember a recent level sample for the silence watch.

        Only the last :data:`_SILENCE_WINDOW_SECONDS` are kept;
        old samples are pruned on every append, so the lists stay
        a few hundred entries at most.
        """
        now = time.monotonic()
        buf = (self._mic_levels if kind == "mic"
               else self._system_levels)
        buf.append((now, float(value)))
        cutoff = now - _SILENCE_WINDOW_SECONDS
        while buf and buf[0][0] < cutoff:
            buf.pop(0)

    def _check_system_silence(self) -> None:
        """Warn once when the system track is silent but the mic is live.

        Runs on a timer while recording. Fires on *observed*
        silence (ten seconds of flat system peaks with mic
        activity) — a quiet meeting is not silence, and the warning
        clears itself when system audio comes back, so a pause in
        the call does not leave a stale warning on screen.
        """
        if (self.capture is None
                or self.capture.state != CaptureState.RECORDING):
            return
        now = time.monotonic()
        system_peaks = [v for t, v in self._system_levels
                        if now - t <= _SILENCE_WINDOW_SECONDS]
        mic_peaks = [v for t, v in self._mic_levels
                     if now - t <= _SILENCE_WINDOW_SECONDS]
        if system_silent_while_mic_live(system_peaks, mic_peaks):
            if not self._system_silence_warned:
                self._system_silence_warned = True
                self.status_label.setText(
                    "⚠ No system audio detected — check the target "
                    "app is playing sound and the right output "
                    "device is selected.")
                self.statusBar().showMessage(
                    "No system audio detected — the system track "
                    "is silent while the mic is live.")
        elif self._system_silence_warned:
            self._system_silence_warned = False
            self.status_label.setText(
                "● Recording… " + (self.capture.warning or ""))

    @staticmethod
    def _index_of(combo, data) -> int:
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                return i
        return 0

    @staticmethod
    def _placeholder_item(text: str):
        """A greyed, non-selectable empty-state row for a list widget."""
        item = QListWidgetItem(text)
        # Centered like a real empty state instead of top-left.
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        return item

    def _show_empty_states(self) -> None:
        """Helpful empty states for the panes that start out blank."""
        if self.segments_list.count() == 0:
            self.segments_list.addItem(self._placeholder_item(
                "No transcript yet — record a meeting, then "
                "transcribe it."))
        if self.suggestions_list.count() == 0:
            self.suggestions_list.addItem(self._placeholder_item(
                "No suggestions yet — they appear automatically after "
                "a recording is processed, or tap “Suggest speakers” "
                "for this session."))
        if self.nearlive_list.count() == 0:
            self.nearlive_list.addItem(self._placeholder_item(
                "Nothing here yet — run near-live on a recorded "
                "session."))

    def _set_recording_ui(self, state: str) -> None:
        """Make the recording state unmistakable: the status pill
        turns red while recording (amber while paused) and the timer
        goes red. ``state`` is "idle", "recording" or "paused". The
        Stop button is already the danger-styled action; the pill +
        timer are the at-a-glance signal."""
        pill = {"idle": "statusPill", "recording": "statusPillRecording",
                "paused": "statusPillPaused"}[state]
        self.status_label.setObjectName(pill)
        self.elapsed_label.setProperty("recording", state == "recording")
        for widget in (self.status_label, self.elapsed_label):
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.update()

    # ------------------------------------------------------- auto-detect ---
    def _apply_auto_detect_setting(self) -> None:
        """Start/stop the meeting watcher to match the setting.

        Called once at startup and again after Settings are saved.
        A watcher that cannot start (Linux, pycaw missing) leaves a
        status note instead of crashing — the Settings checkbox is
        disabled on those platforms, but a settings file carried
        over from another machine could still have the flag on.
        """
        if not self.config.auto_detect_teams:
            self._stop_auto_watcher()
            return
        if (self._auto_watcher is not None
                and getattr(self._auto_watcher, "is_running", False)):
            return
        available, reason = auto_detect_status()
        if not available:
            logger.warning("Auto-detect enabled but unavailable: %s",
                           reason)
            self.status_label.setText(
                f"Auto-detect is on but unavailable here: {reason}")
            return
        bridge = _AutoDetectBridge(self)
        bridge.meeting_started.connect(self._on_meeting_started)
        bridge.meeting_ended.connect(self._on_meeting_ended)
        # Keep the bridge alive: Qt signals need their QObject.
        self._auto_bridge = bridge
        factory = self._auto_watcher_factory
        if factory is not None:
            watcher = factory(
                process_names=list(self.config.process_names),
                on_meeting_started=bridge.meeting_started.emit,
                on_meeting_ended=bridge.meeting_ended.emit)
        else:
            # The real watcher drains the pre-roll buffer at the
            # start gate (see _take_preroll_samples). The test
            # factory takes no such argument, so it is only passed
            # here, never to the factory.
            watcher = MeetingWatcher(
                process_names=list(self.config.process_names),
                on_meeting_started=bridge.meeting_started.emit,
                on_meeting_ended=bridge.meeting_ended.emit,
                preroll_buffer=self._preroll_buffer)
        self._auto_watcher = watcher
        try:
            watcher.start()
        except Exception as exc:  # noqa: BLE001 — watcher must not block UI
            logger.exception("Failed to start the meeting watcher")
            self._auto_watcher = None
            self.status_label.setText(
                f"Auto-detect failed to start: {exc}")
            return
        self.status_label.setText(
            "Auto-detect on — watching for a Teams meeting to start.")

    def _stop_auto_watcher(self) -> None:
        watcher, self._auto_watcher = self._auto_watcher, None
        if watcher is None:
            return
        try:
            watcher.stop()
        except Exception:  # noqa: BLE001 — stopping must not raise
            logger.exception("Failed to stop the meeting watcher")

    def _on_meeting_started(self) -> None:
        """Watcher event, on the UI thread via the signal bridge."""
        if (self.capture is not None
                and self.capture.state == CaptureState.RECORDING):
            logger.info("Auto-detect: meeting started but already "
                        "recording — ignored")
            return
        self.on_record(auto=True)

    def _on_meeting_ended(self) -> None:
        """Watcher event, on the UI thread via the signal bridge."""
        if not self._auto_recording:
            # A manual recording belongs to the user — the watcher
            # never stops it.
            return
        if self.capture is None:
            self._auto_recording = False
            self.auto_badge.setVisible(False)
            return
        self.on_stop(auto=True)

    # ------------------------------------------- scheduled recordings ---
    def _reload_schedules(self, force: bool = False) -> bool:
        """Reload the schedule list when the store file changed.

        The MCP server writes the same file from its own process,
        so the mtime check is what makes an MCP-created schedule
        appear without a restart. Returns True when the list was
        reloaded. Never raises — a store failure leaves the last
        good list in place and is logged.
        """
        try:
            mtime = self._schedules_path.stat().st_mtime_ns
        except OSError:
            mtime = None
        if not force and mtime == self._schedules_mtime_ns:
            return False
        self._schedules_mtime_ns = mtime
        try:
            self._schedules = load_schedules(self._schedules_path)
        except Exception:  # noqa: BLE001 — schedules never break the UI
            logger.exception("Reloading schedules failed")
            self._schedules = []
        return True

    def _schedule_row_text(self, schedule: Schedule) -> str:
        """One list row: local time, app, subject, terminal status."""
        local = schedule.fire_at.astimezone()
        now = datetime.now().astimezone()
        if local.date() == now.date():
            when = f"Today {local:%H:%M}"
        elif local.date() == (now + timedelta(days=1)).date():
            when = f"Tomorrow {local:%H:%M}"
        else:
            when = local.strftime("%a %d %b %H:%M")
        text = f"{when} · {display_name_for_exe(schedule.target_app)}"
        if schedule.subject:
            text += f" — {schedule.subject}"
        if schedule.status != "pending":
            text += f" · {schedule.status}"
        return text

    def _refresh_schedule_list(self) -> None:
        """Rebuild the schedule list from ``self._schedules``.

        Pending + recent-terminal order comes from
        ``visible_schedules`` (the same order the MCP tool lists).
        The selection is preserved by schedule id across rebuilds.
        """
        selected_id = None
        current = self.schedule_list.currentItem()
        if current is not None:
            selected_id = current.data(Qt.ItemDataRole.UserRole)
        self.schedule_list.blockSignals(True)
        try:
            self.schedule_list.clear()
            visible = visible_schedules(self._schedules)
            if not visible:
                placeholder = QListWidgetItem("No recordings scheduled")
                placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
                self.schedule_list.addItem(placeholder)
            for schedule in visible:
                item = QListWidgetItem(
                    self._schedule_row_text(schedule))
                item.setData(Qt.ItemDataRole.UserRole, schedule.id)
                detail = (
                    f"Starts {schedule.fire_at.astimezone():%a %d %b %H:%M}"
                    f" · {schedule.target_app}")
                if schedule.subject:
                    detail += f" · {schedule.subject}"
                if schedule.attendees:
                    detail += " · with " + ", ".join(schedule.attendees)
                detail += (f" · created via {schedule.created_by} "
                           f"· {schedule.status}")
                if schedule.note:
                    detail += f" — {schedule.note}"
                item.setToolTip(detail)
                self.schedule_list.addItem(item)
                if schedule.id == selected_id:
                    self.schedule_list.setCurrentItem(item)
        finally:
            self.schedule_list.blockSignals(False)
        self._update_schedule_cancel_button()

    def _update_schedule_cancel_button(self) -> None:
        """Cancel is enabled only for a selected pending schedule."""
        current = self.schedule_list.currentItem()
        schedule_id = (current.data(Qt.ItemDataRole.UserRole)
                       if current is not None else None)
        enabled = False
        if isinstance(schedule_id, str):
            for schedule in self._schedules:
                if (schedule.id == schedule_id
                        and schedule.status == "pending"):
                    enabled = True
                    break
        self.schedule_cancel_btn.setEnabled(enabled)

    def _refresh_schedule_app_combo(self) -> None:
        """Rebuild the schedule form's app combo (picker rows).

        Same rows as the main picker — live audio sessions first,
        then the configured targets (Teams et al.) even when not
        running, so a meeting app can always be scheduled — plus
        the manual "Other app…" entry. The previous selection is
        preserved by exe name; a fresh combo defaults to the
        configured target.
        """
        sessions = list_audio_sessions()
        sessions = sorted(
            sessions,
            key=lambda s: (s.state != "active",
                           s.display_name.lower()))
        previous_exe = self.selected_schedule_app()
        self.schedule_app_combo.blockSignals(True)
        try:
            self.schedule_app_combo.clear()
            seen: set[str] = set()
            for session in sessions:
                self.schedule_app_combo.addItem(
                    f"{session.display_name} — {session.state}",
                    session.exe_name)
                seen.add(session.exe_name.lower())
            for exe in self.config.process_names:
                if exe.lower() not in seen:
                    self.schedule_app_combo.addItem(
                        f"{display_name_for_exe(exe)} — not running",
                        exe)
                    seen.add(exe.lower())
            if (self._manual_schedule_app
                    and self._manual_schedule_app.lower() not in seen):
                self.schedule_app_combo.addItem(
                    f"{display_name_for_exe(self._manual_schedule_app)}"
                    " — manual entry",
                    self._manual_schedule_app)
            if self.schedule_app_combo.count() == 0:
                self.schedule_app_combo.addItem(
                    "(no apps with audio sessions)", None)
            self.schedule_app_combo.addItem(
                "Other app…", OTHER_APP_SENTINEL)
            target = None
            if previous_exe:
                for i in range(self.schedule_app_combo.count()):
                    data = self.schedule_app_combo.itemData(i)
                    if (isinstance(data, str)
                            and data.lower() == previous_exe.lower()):
                        target = i
                        break
            if target is None:
                for wanted in self.config.process_names:
                    for i in range(self.schedule_app_combo.count()):
                        data = self.schedule_app_combo.itemData(i)
                        if (isinstance(data, str)
                                and data.lower() == wanted.lower()):
                            target = i
                            break
                    if target is not None:
                        break
            if target is None:
                target = 0
            self.schedule_app_combo.setCurrentIndex(target)
            self._schedule_picker_last_exe = self.selected_schedule_app()
        finally:
            self.schedule_app_combo.blockSignals(False)

    def selected_schedule_app(self) -> str | None:
        """Exe name the schedule form targets, or None.

        Mirrors :meth:`selected_process_name` for the schedule
        form's combo: a session row returns its exe name, "Other
        app…" returns the manually typed exe name, the
        empty-state placeholder returns None.
        """
        data = self.schedule_app_combo.currentData()
        if data == OTHER_APP_SENTINEL:
            return self._manual_schedule_app
        if isinstance(data, str) and data:
            return data
        return None

    def _ask_manual_exe_name(self) -> str | None:
        """Ask for an exe name (the "Other app…" escape hatch)."""
        text, ok = QInputDialog.getText(
            self, "Other app",
            "Exe name of the app to record (e.g. chrome.exe):")
        if ok and text.strip():
            return text.strip()
        return None

    def _on_schedule_app_changed(self, index: int) -> None:
        """Handle a schedule-form picker change ("Other app…")."""
        data = self.schedule_app_combo.itemData(index)
        if data == OTHER_APP_SENTINEL:
            typed = self._ask_manual_exe_name()
            if typed:
                self._manual_schedule_app = typed
                self._refresh_schedule_app_combo()
            else:
                self.schedule_app_combo.blockSignals(True)
                try:
                    restored = None
                    last = getattr(self, "_schedule_picker_last_exe",
                                   None)
                    if last:
                        for i in range(
                                self.schedule_app_combo.count()):
                            item_data = \
                                self.schedule_app_combo.itemData(i)
                            if (isinstance(item_data, str)
                                    and item_data.lower()
                                    == last.lower()):
                                restored = i
                                break
                    self.schedule_app_combo.setCurrentIndex(
                        restored if restored is not None else 0)
                finally:
                    self.schedule_app_combo.blockSignals(False)
        else:
            self._schedule_picker_last_exe = (
                data if isinstance(data, str) else None)

    def _add_schedule_from_form(self) -> tuple[Schedule | None, str | None]:
        """Validate the add form and create the schedule.

        Returns ``(schedule, None)`` on success or ``(None, error)``
        — the error is plain text for the caller to show. Kept
        dialog-free so tests can exercise validation without a
        modal dialog.
        """
        when = self.schedule_datetime.dateTime().toPyDateTime()
        target = self.selected_schedule_app()
        if target is None:
            return None, "Pick an app to record first."
        subject = self.schedule_subject_edit.text().strip() or None
        try:
            schedule = create_schedule(
                when, target, subject=subject, created_by="gui",
                path=self._schedules_path)
        except ValueError as exc:
            return None, str(exc)
        return schedule, None

    def _on_schedule_add(self) -> None:
        """Add button: create the schedule or show why it failed."""
        schedule, error = self._add_schedule_from_form()
        if error is not None:
            QMessageBox.warning(self, "Cannot schedule", error)
            return
        self._reload_schedules(force=True)
        self._refresh_schedule_list()
        self.statusBar().showMessage(
            "Scheduled recording for "
            f"{schedule.fire_at.astimezone():%a %d %b %H:%M} — "
            f"{display_name_for_exe(schedule.target_app)}")

    def _on_schedule_cancel(self) -> None:
        """Cancel button: cancel the selected pending schedule."""
        current = self.schedule_list.currentItem()
        if current is None:
            return
        schedule_id = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(schedule_id, str):
            return
        try:
            cancel_schedule(schedule_id, path=self._schedules_path)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot cancel", str(exc))
            return
        self._reload_schedules(force=True)
        self._refresh_schedule_list()
        self.statusBar().showMessage("Scheduled recording cancelled")

    def _check_schedules(self) -> None:
        """Scheduler tick (every 2 s): reload, expire, fire.

        Reloads when the store file's mtime moved (an MCP-created
        schedule appears without a restart), expires stale due
        schedules instead of firing them late, and fires due ones.
        Never shows a dialog and never raises: a firing failure
        marks the schedule skipped and is logged — the timer must
        keep running.
        """
        try:
            changed = self._reload_schedules()
            now = datetime.now(timezone.utc)
            stale = expire_stale_due(now, path=self._schedules_path)
            due = due_schedules(now, path=self._schedules_path)
            for schedule in due:
                self._fire_schedule(schedule)
            if changed or stale or due:
                self._reload_schedules(force=True)
                self._refresh_schedule_list()
        except Exception:  # noqa: BLE001 — the timer must keep running
            logger.exception("Schedule check failed")

    def _fire_schedule(self, schedule: Schedule) -> None:
        """Fire one due schedule through the Record path.

        A recording already running (or a stop/finalize in flight)
        means skip, never double-start. The schedule is marked
        fired BEFORE starting so a crash mid-start cannot refire
        it; a failed start re-marks it skipped with the reason.
        No dialog is ever shown from here — firing is automatic
        and must never block on the user.
        """
        recording = (
            self.capture is not None
            and self.capture.state in (CaptureState.RECORDING,
                                       CaptureState.PAUSED))
        if recording or self._stopping or self._finalizing:
            mark_skipped(schedule, "A recording was already running",
                         path=self._schedules_path)
            logger.info("Scheduled recording %s skipped: already "
                        "recording", schedule.id)
            self.statusBar().showMessage(
                "Scheduled recording skipped — already recording")
            return
        mark_fired(schedule, path=self._schedules_path)
        self.on_record(schedule=schedule)
        if (self.capture is not None
                and self.capture.state == CaptureState.RECORDING):
            mark_fired(
                schedule,
                session_id=(self.session_dir.name
                            if self.session_dir is not None else None),
                path=self._schedules_path)
            logger.info("Scheduled recording %s started (%s)",
                        schedule.id, schedule.target_app)
        else:
            mark_skipped(schedule, "Recording failed to start",
                         path=self._schedules_path)
            logger.warning("Scheduled recording %s failed to start",
                           schedule.id)
            self.statusBar().showMessage(
                "Scheduled recording failed to start — see the log")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name
        # S3 (Nov 20 2026 review fix): closing while the stop
        # worker runs destroys the QThread mid-flight and aborts
        # the process. Defer the close until the stop settles —
        # _maybe_close_pending() honours it once the stop worker,
        # the live-chunk wait, and the finalize/transcribe tail
        # are all done. The same deferral covers the wait for an
        # in-flight live chunk (_stop_finalize_pending): its
        # worker thread is equally live.
        if self._stopping or self._stop_finalize_pending:
            self._close_pending = True
            self.status_label.setText(
                "Finishing the stop before closing…")
            event.ignore()
            return
        self._stop_auto_watcher()
        self._stop_preroll_feeder()
        # Worker threads (Nov 21 2026): destroying a QThread
        # while it is still running aborts the process (Qt
        # refuses it outright). The waveform thread is a short
        # file read — wait for it bounded. An enterprise run is
        # cancelled first (kills the CLI), then waited for
        # bounded; a run that still has not returned is left
        # running rather than aborting the close.
        self._wait_for_waveform_thread()
        self._cancel_and_wait_for_enterprise_thread()
        if self._tray is not None:
            self._tray.hide()
        if self._onboarding_dialog is not None:
            self._onboarding_dialog.close()
        super().closeEvent(event)

    def _wait_for_waveform_thread(self) -> None:
        for thread in list(getattr(self, "_waveform_threads", ())):
            if thread.isRunning():
                thread.quit()
                thread.wait(3000)

    def _cancel_and_wait_for_enterprise_thread(self) -> None:
        worker = getattr(self, "_enterprise_worker", None)
        if worker is not None:
            worker.cancel()
        event = getattr(self, "_enterprise_cancel_event", None)
        if event is not None:
            event.set()
        thread = getattr(self, "_enterprise_thread", None)
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(5000)

    def _maybe_close_pending(self) -> None:
        """Honour a close deferred by :meth:`closeEvent`.

        Only fires once nothing stop-related is still running:
        the stop worker, the live-chunk wait, the finalize tail,
        and the full-file transcribe fallback it can fall back
        to. Deliberately NOT gated on the later auto-process
        steps (diarize/suggest): closing during those was always
        possible, and waiting them out would hold the window
        open for minutes after the user asked to close.
        """
        if not self._close_pending:
            return
        if (self._stopping or self._stop_finalize_pending
                or self._finalizing or self._transcribing):
            return
        self._close_pending = False
        self.close()

    # ------------------------------------------------------------ buttons ---
    def _update_buttons(self) -> None:
        state = self.capture.state if self.capture else CaptureState.IDLE
        # "Stopped" includes no capture at all: a session loaded
        # from the library was stopped long ago, and its Process
        # actions must be enabled for it.
        stopped = self.capture is None or state == CaptureState.STOPPED
        # Record stays disabled until Stop fully settles (Nov 20
        # 2026 Stop-freeze fix): while the stop worker runs, while
        # Stop waits for an in-flight live chunk, and while the
        # finalize tail runs — a second Record must never start
        # into a half-stopped capture or a pending finalize.
        self.record_btn.setEnabled(
            state in (CaptureState.IDLE, CaptureState.STOPPED)
            and not self._stopping
            and not self._stop_finalize_pending
            and not self._finalizing)
        # Pause/Resume are gated on _stopping too (S1, Nov 20
        # 2026 review fix): capture.state stays RECORDING until
        # the stop worker calls capture.stop(), so without this
        # the buttons stayed live through the stop window and a
        # Pause then raised "Pause failed" mid-Stop.
        self.pause_btn.setEnabled(
            state == CaptureState.RECORDING and not self._stopping)
        self.resume_btn.setEnabled(
            state == CaptureState.PAUSED and not self._stopping)
        self.stop_btn.setEnabled(
            state in (CaptureState.RECORDING, CaptureState.PAUSED)
            and not self._stopping)
        # Phase 2: transcribe only once a session exists and is stopped,
        # and never while a transcription is already running.
        self.transcribe_btn.setEnabled(
            not self._transcribing
            and not self._finalizing
            and self.session_dir is not None
            and stopped)
        self.cancel_transcribe_btn.setEnabled(
            self._transcribing or self._finalizing
            or self._stop_finalize_pending)
        # Phase 3: diarize only once a transcript exists for the session.
        has_transcript = (
            self.session_dir is not None
            and (self.session_dir / TRANSCRIPT_JSON).exists())
        self.diarize_btn.setEnabled(
            not self._diarizing and not self._transcribing
            and has_transcript)
        self.rename_btn.setEnabled(
            has_transcript and self.rename_label_combo.count() > 0)
        # Phase 4: suggest only once the session is diarized (clusters
        # need speaker labels); near-live only once audio exists.
        from app.storage.sessions import COMBINED_FILE as _COMBINED
        has_diarization = (
            self.session_dir is not None
            and (self.session_dir / "diarization.json").exists())
        self.suggest_btn.setEnabled(
            not self._suggesting and has_diarization)
        has_audio = (
            self.session_dir is not None
            and (self.session_dir / _COMBINED).exists())
        self.nearlive_btn.setEnabled(
            not self._nearlive_running and has_audio
            and stopped)
        # Confirm/Correct need a live suggestion object (with its
        # embedding) — a stored row from speaker_suggestions.json
        # is a read-back only (see _refresh_suggestions_from_disk),
        # so they stay disabled for it until Suggest regenerates
        # live suggestions. Play sample works from the row's label
        # either way.
        self.confirm_btn.setEnabled(
            self._selected_suggestion() is not None)
        self.correct_btn.setEnabled(
            self._selected_suggestion() is not None
            and bool(self.rename_name_edit.text().strip()))
        # Speaker samples + rename: only for a real selection (the
        # placeholder rows carry no UserRole data, so they never
        # enable these — a suggestion with no object behind it
        # cannot be played either).
        self.play_known_btn.setEnabled(
            self._selected_known_speaker() is not None)
        self.rename_known_btn.setEnabled(
            self._selected_known_speaker() is not None)
        self.play_suggestion_btn.setEnabled(
            self._selected_suggestion_label() is not None)
        # Phase 5: LLM export + local summary need a transcript.
        self.export_btn.setEnabled(has_transcript)
        self.summarize_btn.setEnabled(
            has_transcript and not self._summarizing)
        # Transcript cleanup (Nov 10 2026): re-clean and polish
        # need a transcript; polish never runs twice at once.
        self.reclean_btn.setEnabled(
            has_transcript and not self._transcribing)
        self.polish_btn.setEnabled(
            has_transcript and not self._polishing)
        self.transcript_view_combo.setEnabled(has_transcript)
        # Enterprise cleanup: needs a transcript, and never runs
        # while transcription/finalize is writing one.
        self.enterprise_btn.setEnabled(
            has_transcript and not self._transcribing
            and not self._finalizing and not self._enterprise_running)
        self.enterprise_cancel_btn.setEnabled(self._enterprise_running)
        # Bookmarks: the Record-view button marks the live
        # recording; the Session-view button marks the loaded
        # session's current moment.
        self.bookmark_btn.setEnabled(
            state in (CaptureState.RECORDING, CaptureState.PAUSED)
            and not self._stopping)
        self.add_bookmark_btn.setEnabled(
            self.session_dir is not None and stopped
            and self.bookmarks_list is not None)
        # The Dictation toggle must not flip mid-recording: it
        # describes the session being recorded.
        self.dictation_btn.setEnabled(
            state in (CaptureState.IDLE, CaptureState.STOPPED)
            and not self._stopping
            and not self._stop_finalize_pending
            and not self._finalizing)
        # Person page: only for a real selection.
        open_person = getattr(self, "open_person_btn", None)
        if open_person is not None:
            open_person.setEnabled(
                self._selected_known_speaker() is not None)
        # Tray Stop follows the window's Stop button.
        if self._tray_stop_action is not None:
            self._tray_stop_action.setEnabled(self.stop_btn.isEnabled())
        self._update_library_star_button()

    def on_record(self, auto: bool = False,
                  schedule: Schedule | None = None) -> None:
        """Start a recording. ``auto=True`` marks it auto-started:
        the AUTO badge shows and the meeting-ended event stops it.
        A failed start never leaves the auto flags set.

        The target is the app picker's chosen exe name — never a
        hard-coded fallback. The exception is an auto-started
        recording (``auto=True``): the watcher fired for the
        *configured* target (Teams), so that is what is captured —
        the picker's choice must not hijack a meeting the watcher
        detected. In "Record one app" mode on Windows, a target
        with no audio session at all gets a loud warning dialog
        first (record anyway, system-wide, or cancel) — a per-app
        recording of a silent target is a recording of nothing,
        and the user must choose that explicitly.

        The session's recording source is resolved the same way:
        the persisted default source (Settings) overrides
        detection (the only way to name a source no process
        identifies, e.g. a Claude Code session); otherwise it is
        detected from the app actually picked — or the configured
        target for an auto-started recording. No app picked at all
        means ``solo``: the ``process_name`` fallback to ms-teams.exe
        is capture bookkeeping, not evidence of a Teams meeting.

        ``schedule`` is a fired :class:`Schedule` (Nov 22 2026):
        the target app comes from the schedule (never the
        picker), the no-audio-session dialog is skipped (a
        meeting app that launches a minute late must still be
        captured — ``capture_system_mode="auto"`` falls back to
        loopback with the warning recorded, as manual "record
        anyway" does), dictation never applies, and the recording
        is NEVER auto-stopped: ``_auto_recording`` stays False, so
        the watcher cannot end it — only the user's Stop does.
        The session metadata is tagged with the schedule id and
        any calendar details under a ``scheduled`` key (the
        ``calendar_event`` key belongs to Outlook linking).
        """
        if schedule is not None:
            # A fired schedule names its own target — the picker's
            # choice must not hijack it.
            process_name = schedule.target_app
        elif auto:
            process_name = (self.config.process_names[0]
                            if self.config.process_names
                            else "ms-teams.exe")
        else:
            process_name = self.selected_process_name()
        if not process_name:
            # No app picked (empty picker): fall back to the
            # configured target, as the old exe field did.
            process_name = (self.config.process_names[0]
                            if self.config.process_names
                            else "ms-teams.exe")
        capture_mode = self.config.capture_system_mode
        if schedule is not None and capture_mode == "per-app":
            # Scheduled recordings use "auto" capture-mode
            # semantics (Nov 22 2026): the target may launch a
            # minute late, so per-app is tried and loopback is
            # fallen back to with the warning recorded — never a
            # silent per-app recording, and never a dialog. Local
            # only: the persisted setting is untouched.
            capture_mode = "auto"
        if (schedule is None and not auto
                and capture_mode in ("auto", "per-app")
                and sys.platform == "win32"):
            # The probe has to be usable for the check to mean
            # anything — on a machine where it is not, recording
            # proceeds as before (DualCapture still falls back to
            # loopback with a warning when per-app fails).
            available, _reason = audio_sessions_available()
            if available and not self._target_has_audio_session(
                    process_name):
                if not self._confirm_record_without_session(
                        display_name_for_exe(process_name)):
                    return
                # The user chose to record anyway: system-wide for
                # this recording, loudly (the label below says so).
                capture_mode = "loopback"
        # Recording source (see the docstring): the persisted
        # default override wins; otherwise detect from the app
        # actually picked (auto: the configured target; manual
        # with nothing picked: solo — the ms-teams.exe fallback
        # is not evidence).
        detect_exe = (process_name if (auto or schedule is not None)
                      else self.selected_process_name())
        source, source_detail = resolve_recording_source(
            override=self.config.default_source,
            exe_name=detect_exe)
        # Dictation (Nov 21 2026): only a manual Record with the
        # toggle on. An auto-started recording is a meeting the
        # watcher detected — never dictation. A scheduled
        # recording is a meeting on a timer — never dictation.
        dictation = (schedule is None and not auto
                     and self.dictation_btn.isChecked())
        try:
            devices = {
                "microphone_index": self.mic_combo.currentData(),
                "loopback_index": self.loopback_combo.currentData(),
                "microphone": self.mic_combo.currentText(),
                "loopback": self.loopback_combo.currentText(),
            }
            if dictation:
                self.session_dir = create_dictation_session(
                    self.config.output_root,
                    sample_rate=self.config.sample_rate,
                    channels=self.config.channels,
                    devices=devices,
                    source_detail=source_detail)
            else:
                self.session_dir = create_session(
                    self.config.output_root,
                    app_name=display_name_for_exe(process_name),
                    sample_rate=self.config.sample_rate,
                    channels=self.config.channels,
                    source=source,
                    source_detail=source_detail,
                    devices=devices)
            # The fresh session is the loaded one: clear the previous
            # session's transcript / suggestions from the views now,
            # or the Session view would keep showing the old
            # session's segments under the new session's name.
            self.load_session(self.session_dir)
            capture_kwargs: dict = dict(
                mic_device_index=self.mic_combo.currentData(),
                loopback_device_index=self.loopback_combo.currentData(),
                process_name=process_name,
                sample_rate=self.config.sample_rate,
                capture_system_mode=capture_mode,
                mic_level_callback=self._mic_bridge.emit,
                system_level_callback=self._system_bridge.emit)
            if dictation:
                # Mic-only: no system capture object at all
                # (DualCapture's system_cls=None mode).
                capture_kwargs["system_cls"] = None
            self.capture = DualCapture(self.session_dir, **capture_kwargs)
            # Pre-roll handoff, in the contract's exact order:
            # feeder stopped first (synchronously, so no device
            # conflict), then the buffered audio drained into
            # capture.start(). An empty buffer means no pre-roll.
            preroll_samples = self._take_preroll_samples(auto=auto)
            # Only pass the argument when the buffer actually
            # drained audio: an empty dict is identical to no
            # pre-roll, and capture stand-ins whose start() takes
            # no arguments must keep working.
            if preroll_samples:
                self.capture.start(preroll_samples=preroll_samples)
            else:
                self.capture.start()
            # Record the honest capture mode + warning in the metadata
            # AFTER start(): in "auto" mode a failed per-app capture
            # falls back to system-wide loopback inside start(), and
            # the metadata must say what actually ran. The native
            # stream rates go with it: a device whose native rate is
            # 44.1 kHz (e.g. an RDP endpoint) streams at 44.1 kHz and
            # is resampled to the 48 kHz pipeline rate — the WAVs are
            # always 48 kHz, but the session should say what the
            # hardware actually did. This is bookkeeping only: it must
            # never fail a recording that already started (an
            # AttributeError here used to surface "Recording failed"
            # while the mic kept running orphaned), so it is guarded
            # and best-effort. ``getattr`` because a capture backend
            # need not expose ``mic`` / ``system`` attributes.
            from app.storage.sessions import update_metadata
            try:
                mic = getattr(self.capture, "mic", None)
                system = getattr(self.capture, "system", None)
                update_metadata(
                    self.session_dir, mode=self.capture.mode,
                    warning=self.capture.warning,
                    mic_stream_sample_rate=(
                        mic.stream_sample_rate
                        if mic is not None else None),
                    system_stream_sample_rate=(
                        system.stream_sample_rate
                        if system is not None else None))
            except Exception:
                logger.exception(
                    "Failed to update session metadata after start")
            if schedule is not None:
                # Tag the session with its schedule (Nov 22 2026):
                # id + any calendar details, under a ``scheduled``
                # key — the ``calendar_event`` key belongs to
                # Outlook linking and must not be overwritten.
                # Guarded and best-effort like the block above: a
                # tagging failure must never fail a recording that
                # already started.
                self._scheduled_recording_id = schedule.id
                scheduled = {
                    "schedule_id": schedule.id,
                    "fire_at": schedule.fire_at.isoformat(),
                }
                if schedule.subject:
                    scheduled["subject"] = schedule.subject
                if schedule.attendees:
                    scheduled["attendees"] = list(schedule.attendees)
                if schedule.organiser:
                    scheduled["organiser"] = schedule.organiser
                try:
                    update_metadata(self.session_dir,
                                    scheduled=scheduled)
                except Exception:
                    logger.exception(
                        "Failed to tag session metadata as scheduled")
            if self.capture.mode == "per-app":
                self.mode_label.setText(
                    "Capture: "
                    f"{display_name_for_exe(process_name)}-only audio "
                    "(per-app capture)")
            else:
                self.mode_label.setText(
                    "Capture: system-wide loopback"
                    + (" (per-app capture unavailable — see warning)"
                       if capture_mode == "auto"
                       else ""))
        except Exception as exc:
            logger.exception("Failed to start recording")
            self.capture = None
            self._auto_recording = False
            self._scheduled_recording_id = None
            self.auto_badge.setVisible(False)
            if schedule is not None:
                # A scheduled start must NEVER block on a dialog:
                # firing is automatic and unattended. Log it, say
                # so in the status line, and let _fire_schedule
                # mark the schedule skipped (capture is None).
                self.statusBar().showMessage(
                    "Scheduled recording failed to start — see the log")
            else:
                QMessageBox.critical(self, "Recording failed", str(exc))
            # Failed start: the pre-roll feeder restarts if still
            # enabled (contract) — the next Record must still be
            # able to drain it.
            self._restart_preroll_feeder()
            self._update_buttons()
            return
        self._record_started_at = time.time()
        self._elapsed_paused_total = 0.0
        self._pause_started_at = None
        self.timer.start()
        # Low-disk poll: every 30 s while recording, first
        # crossing only (see _check_disk_while_recording). Never
        # stops a recording.
        self._disk_warned = False
        self._check_disk_while_recording()
        self._disk_timer.start()
        # Silence watch: warn if the system track stays silent while
        # the mic is live (see _check_system_silence).
        self._system_silence_warned = False
        self._silence_timer.start()
        # Live transcription: the 5 s poller + chunker for this
        # recording (see _start_live_transcription). Auto-started
        # recordings take the same path — on_record is the one entry.
        self._start_live_transcription()
        self.session_label.setText(
            f"Recording to: {self.session_dir.name}")
        self.session_label.setToolTip(str(self.session_dir))
        self._auto_recording = auto
        self.auto_badge.setVisible(auto)
        if auto:
            self.status_label.setText(
                "● Auto-recording — Teams meeting detected. "
                + (self.capture.warning or ""))
            self.statusBar().showMessage(
                "Auto-recording — Teams meeting detected")
        elif schedule is not None:
            # Scheduled recording: say so in the status line (the
            # AUTO badge stays hidden — the watcher must never
            # stop it, and the badge implies watcher control).
            self.status_label.setText(
                "● Scheduled recording… "
                + (self.capture.warning or ""))
            self.statusBar().showMessage(
                f"Scheduled recording started — "
                f"{display_name_for_exe(schedule.target_app)}"
                + (f" · {schedule.subject}"
                   if schedule.subject else ""))
        else:
            self.status_label.setText(
                "● Recording… " + (self.capture.warning or ""))
            self.statusBar().showMessage(
                f"Recording to {self.session_dir}")
        self._set_recording_ui("recording")
        self._update_buttons()
        # The new session is the loaded one: list it in the chooser
        # and say so in the Session view.
        self._refresh_session_chooser(select=self.session_dir)
        self._update_loaded_session_label()

    def on_pause(self) -> None:
        if not self.capture:
            return
        try:
            self.capture.pause()
        except Exception as exc:
            QMessageBox.warning(self, "Pause failed", str(exc))
            return
        self._pause_started_at = time.time()
        self._live_timer.stop()
        self.status_label.setText(
            "❚❚ Paused — audio received while paused is discarded.")
        self._set_recording_ui("paused")
        self._update_buttons()

    def on_resume(self) -> None:
        if not self.capture:
            return
        try:
            self.capture.resume()
        except Exception as exc:
            QMessageBox.warning(self, "Resume failed", str(exc))
            return
        if self._pause_started_at is not None:
            self._elapsed_paused_total += time.time() - self._pause_started_at
            self._pause_started_at = None
        # Capture pause discards audio, so the tracks stay aligned
        # and the chunker's watermark (file-duration based) simply
        # does not advance while paused — restart the poller, which
        # then finds nothing new until audio grows again.
        if self._live_chunker is not None and not self._live_failed:
            self._live_timer.start()
        self.status_label.setText("● Recording…")
        self._set_recording_ui("recording")
        self._update_buttons()

    def on_stop(self, auto: bool = False) -> None:
        """Stop the recording and save the session.

        ``auto=True`` is the watcher ending an auto-started
        recording. A manual stop (``auto=False``) of an auto-started
        recording tells the watcher not to re-trigger on the
        still-active meeting — it stays suppressed until the
        session goes inactive first.

        The stop itself runs off the UI thread (Nov 20 2026
        Stop-freeze fix): capture stop joins the capture threads
        and mixes the tracks, which froze the app for seconds on
        a long meeting. This slot only shows "Stopping…",
        disables Stop/Record, and starts a :class:`_StopWorker`;
        :meth:`_on_stop_finished` does the save + finalize once
        the worker reports back. A second Stop while one is in
        flight is ignored.
        """
        if not self.capture or not self.session_dir:
            return
        if self._stopping:
            return
        started = time.monotonic()
        was_auto = self._auto_recording
        if not auto and was_auto and self._auto_watcher is not None:
            try:
                self._auto_watcher.notify_manual_stop()
            except Exception:  # noqa: BLE001 — suppression is best-effort
                logger.exception("notify_manual_stop failed")
        self._auto_recording = False
        self._scheduled_recording_id = None
        self.auto_badge.setVisible(False)
        self.timer.stop()
        self._silence_timer.stop()
        self._live_timer.stop()
        self._disk_timer.stop()
        self._stopping = True
        # S2: snapshot the session being stopped — _on_stop_finished
        # finalizes this directory, whatever session_dir becomes
        # before the slot runs.
        self._stopping_session_dir = self.session_dir
        # S6: a live tick still copying the growing tracks races
        # the stop worker closing those same files — a torn copy
        # would fail into the live-error path and spuriously kill
        # live transcription. Abandon such a tick up-front: its
        # late signals are ignored by design (see
        # _live_abandoned), the chunker is detached so the tail
        # pass cannot race the worker on it, and the finalize
        # proceeds from the saved live transcript — the tail
        # re-transcribes the tick's windows from the watermark,
        # so nothing is lost. A tick whose snapshot is already
        # done only reads its own temp copies, so it is safe to
        # wait for (see _begin_stop_finalize) and is left alone.
        if self._live_busy:
            worker = self._live_worker
            snapshot_done = getattr(worker, "snapshot_done", True)
            if not snapshot_done:
                self._live_abandoned = True
                self._live_busy = False
                self._live_chunker = None
        self.status_label.setText("Stopping…")
        self._set_recording_ui("idle")
        self._update_buttons()
        self._stop_thread = QThread(self)
        self._stop_worker = _StopWorker(self.capture)
        self._stop_worker.moveToThread(self._stop_thread)
        self._stop_thread.started.connect(self._stop_worker.run)
        self._stop_worker.finished.connect(self._on_stop_finished)
        self._stop_worker.error.connect(self._on_stop_error)
        self._stop_worker.finished.connect(self._stop_thread.quit)
        self._stop_worker.error.connect(self._stop_thread.quit)
        self._stop_worker.finished.connect(
            self._stop_worker.deleteLater)
        self._stop_worker.error.connect(
            self._stop_worker.deleteLater)
        self._stop_thread.finished.connect(
            self._stop_thread.deleteLater)
        self._stop_thread.start()
        # Cheap always-on signal (Nov 20 2026): the synchronous
        # section of on_stop must stay milliseconds — if a future
        # freeze report lands, this number says whether Stop
        # itself or the worker was slow.
        logger.info("on_stop synchronous section took %.3fs",
                    time.monotonic() - started)

    def _on_stop_finished(self, files: dict, duration: float) -> None:
        """A stop worker finished: save the session, then finalize.

        Runs on the UI thread. The capture is already stopped
        (state STOPPED) and ``combined.wav`` is mixed; what
        remains is metadata, the Outlook link, the status lines,
        and the Stop-time transcription (live + tail) via
        :meth:`_begin_stop_finalize` — exactly the tail of the
        old synchronous on_stop, moved here so none of it waits
        on the capture threads.

        Finalizes the session snapshotted in :meth:`on_stop`
        (``_stopping_session_dir``), not whatever
        ``self.session_dir`` happens to be now: after the
        worker flipped the capture state to STOPPED but before
        this slot ran, a Library/Calendar click could have
        loaded another session, and finalizing that one would
        write the wrong metadata duration and transcribe the
        wrong session. The stopped session is restored as the
        loaded one before finalizing, so the live + tail pass
        (which reads ``self.session_dir``) targets it too —
        its in-memory chunker is gone (load_session cleared
        it), but the saved live transcript on disk still stands
        and the tail re-covers the rest from the watermark.
        """
        self._stopping = False
        self._stop_worker = None
        self._stop_thread = None
        # Pre-roll: capture is stopped, so the feeder restarts if
        # still enabled (contract) — the next Record drains it.
        self._restart_preroll_feeder()
        stopping_dir = (self._stopping_session_dir
                        if self._stopping_session_dir is not None
                        else self.session_dir)
        self._stopping_session_dir = None
        try:
            finalize_session(stopping_dir, duration_seconds=duration)
            # Outlook calendar linking (Nov 10 2026, Feature 3):
            # match the just-finished session to its calendar event
            # and store the link in metadata.json. Best-effort and
            # fully guarded inside link_session() — a missing Outlook
            # can never fail the stop. The label refresh below shows
            # the link (or "No calendar event found").
            try:
                link_session(stopping_dir,
                             enabled=self.config.outlook_link_enabled)
            except Exception:  # noqa: BLE001 — linking never fails a stop
                logger.exception("Outlook linking failed after stop")
        except Exception as exc:
            logger.exception("Failed to stop recording cleanly")
            QMessageBox.critical(self, "Stop failed", str(exc))
            self._update_buttons()
            self._maybe_close_pending()
            return
        # Restore the stopped session as the loaded one (see the
        # docstring): everything below — labels, chooser, and the
        # Stop-time finalize — works on it.
        self.session_dir = stopping_dir
        names = "\n".join(f"• {p.name}" for p in files.values())
        # One line: the old three-line form ("Saved to:\n<name>\nDuration:
        # …") pushed the Record card past 800 px height and clipped
        # the capture-mode explanation above it. The full path and
        # file list stay in the tooltip.
        self.session_label.setText(
            f"Saved: {stopping_dir.name} · {duration:.1f}s")
        self.session_label.setToolTip(
            f"{stopping_dir}\n{names}")
        combined_state = ("present" if (stopping_dir / COMBINED_FILE).exists()
                          else "MISSING — see log")
        self.status_label.setText(
            f"Stopped. Combined track: {COMBINED_FILE} ({combined_state}).")
        self._set_recording_ui("idle")
        self.statusBar().showMessage(f"Saved to {stopping_dir}")
        self._update_buttons()
        # The stopped session is now in the library: list it in the
        # chooser (selected) and refresh its status line.
        self._refresh_session_chooser(select=stopping_dir)
        self._update_loaded_session_label()
        # Stop-time transcription (Nov 19 2026, live + tail): the
        # live transcript is finalised on every Stop of a recording
        # of 2 s or more, whether or not auto-processing is on —
        # the transcript is the record of the meeting, not an
        # auto-process extra. Only the diarize → suggest/identify
        # chain stays gated on auto_process_recordings (via
        # ``_auto_processing``, read by
        # :meth:`_after_transcribe_finished`). The recording is
        # saved and finalised above, so a failed auto-step can never
        # lose it. Recordings under 2 s are skipped — almost
        # certainly an accidental tap — with a status note, not an
        # error dialog; any stale live file is removed, so a short
        # tap leaves nothing behind.
        if duration >= 2.0:
            self._auto_processing = self.config.auto_process_recordings
            self.status_label.setText(
                "Saved. Auto-processing: transcribing…"
                if self._auto_processing else
                "Saved. Finalizing the live transcript…")
            self._begin_stop_finalize()
        else:
            try:
                (self.session_dir / TRANSCRIPT_LIVE_FILE).unlink(
                    missing_ok=True)
            except OSError:  # noqa: BLE001 — a leftover live file is harmless
                logger.warning("Could not delete %s after a short stop",
                               TRANSCRIPT_LIVE_FILE, exc_info=True)
            self._live_chunker = None
            self.status_label.setText(
                "Stopped. Recording too short to transcribe — "
                "use the Session view to run the steps by hand.")
        # Back to the capture-mode explanation (on_record overwrote
        # the label with what actually ran).
        self._update_capture_mode_ui()
        self._maybe_close_pending()

    def _on_stop_error(self, message: str) -> None:
        """A stop worker failed: the raw tracks are already on disk.

        The recording is untouched (capture.stop() failed after the
        tracks were written, or while joining a thread) — report
        it and leave the session loadable by hand. The failed
        capture is discarded (S5, Nov 20 2026 review fix): its
        state is unknown — ``capture.stop()`` may have raised
        before flipping it to STOPPED, which left Record disabled
        with no reset path — and :meth:`on_record` builds a fresh
        ``DualCapture`` anyway, so dropping the stale object is
        what makes Record usable again as this docstring claims.
        The capture threads are daemons and the tracks stay on
        disk in the session folder, so nothing is lost by
        dropping it. The session stays loadable by hand.
        """
        self._stopping = False
        self._stop_worker = None
        self._stop_thread = None
        self._stopping_session_dir = None
        self.capture = None
        # Pre-roll: the failed stop still ended capture, so the
        # feeder restarts if still enabled (contract).
        self._restart_preroll_feeder()
        logger.error("Stop failed: %s", message)
        self.status_label.setText(f"Stop failed: {message}")
        QMessageBox.critical(
            self, "Stop failed",
            f"{message}\n\nThe recording itself is untouched — the "
            "raw tracks are on disk in the session folder. The "
            "failed capture was discarded; press Record to start "
            "a new session.")
        self._update_buttons()
        self._maybe_close_pending()

    # ------------------------------------------------------- transcription ---
    def on_transcribe(self) -> None:
        """Transcribe the loaded session in a background thread."""
        if not self.session_dir or self._transcribing or self._finalizing:
            return
        model_size = self._configured_model()
        self._transcribing = True
        self._transcribe_cancel_event = threading.Event()
        self.transcribe_progress.setVisible(True)
        self.transcribe_progress.setValue(0)
        self.transcribe_label.setText(
            f"Transcribing with Whisper {model_size} (CPU, int8)… "
            "first run downloads the model.")
        self._update_buttons()
        self._transcribe_thread = QThread(self)
        self._transcribe_worker = _TranscribeWorker(
            self.session_dir, model_size)
        # Set after construction (not a constructor argument) so
        # test fakes that stub _TranscribeWorker with a narrower
        # signature keep working; the real worker reads this in run().
        self._transcribe_worker.is_cancelled = (
            self._transcribe_cancel_event.is_set)
        self._transcribe_worker.moveToThread(self._transcribe_thread)
        self._transcribe_thread.started.connect(self._transcribe_worker.run)
        self._transcribe_worker.progress.connect(
            self._on_transcribe_progress)
        self._transcribe_worker.finished.connect(
            self._on_transcribe_finished)
        self._transcribe_worker.error.connect(self._on_transcribe_error)
        # ``cancelled`` exists on the real worker; test fakes that
        # stub _TranscribeWorker may not have it — connect only
        # when present, so a stubbed worker still runs.
        cancelled = getattr(self._transcribe_worker, "cancelled", None)
        if cancelled is not None:
            cancelled.connect(self._on_transcribe_cancelled)
        self._transcribe_worker.finished.connect(
            self._transcribe_thread.quit)
        self._transcribe_worker.error.connect(self._transcribe_thread.quit)
        if cancelled is not None:
            cancelled.connect(self._transcribe_thread.quit)
        self._transcribe_thread.start()

    def on_cancel_transcribe(self) -> None:
        """Cancel the running transcription (manual or Stop finalize).

        The engine checks the flag per segment, so the worker stops
        after the current segment and reports the cancel — the
        partial segments are discarded and no transcript.json is
        written (see _TranscribeWorker / transcribe_session). A
        cancel requested while the model is still downloading
        (first run) takes effect once the model has loaded: the
        blocking download inside ``WhisperModel(...)`` cannot be
        interrupted mid-download, so the label says so honestly
        instead of promising an instant stop. Cancelling the Stop
        finalize leaves the saved live transcript on disk — the
        session stays transcribable by hand (Transcribe in the
        Session view) — and stops the auto-process chain quietly.
        Cancelling while Stop waits for an in-flight live chunk
        abandons that chunk immediately and finalizes from the
        saved live transcript instead of waiting out the 60 s
        (Nov 20 2026 Stop-freeze fix — Cancel is enabled during
        the wait, and the status line counts down to show it).
        """
        if self._stop_finalize_pending:
            self.status_label.setText(
                "Finalizing from the saved live transcript…")
            self._abandon_live_and_finalize()
            return
        if self._finalizing and self._finalize_cancel_event is not None:
            self._finalize_cancel_event.set()
            self.transcribe_label.setText(
                "Cancelling the final transcript… the live "
                "transcript is kept; run Transcribe in the Session "
                "view to transcribe the full recording by hand.")
            return
        if self._transcribe_cancel_event is not None:
            self._transcribe_cancel_event.set()
        self.transcribe_label.setText(
            "Cancelling transcription… takes effect once the model "
            "has loaded — a first-run download cannot be "
            "interrupted mid-download.")

    def _on_transcribe_cancelled(self, summary: dict) -> None:
        # A user-initiated cancel stops the auto-process chain too,
        # quietly (no error dialog — the user asked for this), and
        # the recording is untouched.
        self._auto_processing = False
        self._transcribing = False
        self._transcribe_cancel_event = None
        self.transcribe_progress.setVisible(False)
        self.transcribe_label.setText(
            f"Transcription cancelled after {summary['segments']} "
            "segment(s) — no transcript.json written.")
        self.statusBar().showMessage(
            "Transcription cancelled — no transcript written.")
        self._update_buttons()
        self._maybe_close_pending()

    def _on_transcribe_progress(self, fraction: float, text: str) -> None:
        self.transcribe_progress.setValue(int(fraction * 100))
        self.transcribe_label.setText(text)

    def _on_transcribe_finished(self, summary: dict) -> None:
        self._transcribing = False
        self._transcribe_cancel_event = None
        self._after_transcribe_finished(summary)
        self._maybe_close_pending()

    def _after_transcribe_finished(self, summary: dict) -> None:
        """Shared finish path for both transcribe routes.

        The full-file worker (via _on_transcribe_finished) and the
        live + tail finalize both land here with the same summary,
        so cleanup → diarize → suggest runs identically either way.
        """
        self.transcribe_progress.setValue(100)
        # Hide the bar when the step finishes: a full-width indigo
        # bar left visible at 100% looks like a button that does
        # nothing (the "purple bar" report, Nov 10 2026). The Done
        # line below is the record of the run.
        self.transcribe_progress.setVisible(False)
        self.transcribe_label.setText(
            f"Done: {summary['segments']} segments in "
            f"{summary['wall_seconds']}s (RTF {summary['real_time_factor']}, "
            f"model {summary['model']}). "
            "transcript.json / .txt / .srt written next to the WAVs.")
        self.statusBar().showMessage(
            f"Transcript saved in {summary['session_dir']}")
        # Rule cleanup (Nov 10 2026): every new transcript gets a
        # cleaned derived artifact automatically — the verbatim
        # files are only read, never written. Diarization rewrites
        # transcript.json with speaker labels, so the chain
        # re-cleans after that step too (see _on_diarize_finished
        # and the end of _on_suggest_finished).
        self._run_cleanup()
        self._refresh_segments_list()
        self._refresh_session_chooser()
        self._update_loaded_session_label()
        self._update_buttons()
        # Auto-process chain: transcription done → diarize next.
        if self._auto_processing:
            self.status_label.setText(
                "Auto-processing: diarizing…")
            self.on_diarize()

    # -------------------------------------------------------- diarization ---
    def on_diarize(self) -> None:
        """Diarize the loaded session in a background thread."""
        if not self.session_dir or self._diarizing:
            return
        # Same rule as _update_diarize_mode_ui: the fallback is the
        # configured mode, never a hard-coded "simple".
        mode = (self.diarize_mode_combo.currentData()
                or self.config.diarization_mode)
        try:
            min_sp = int(self.min_speakers_edit.text()) \
                if self.min_speakers_edit.text().strip() else None
            max_sp = int(self.max_speakers_edit.text()) \
                if self.max_speakers_edit.text().strip() else None
        except ValueError:
            QMessageBox.warning(
                self, "Invalid speaker hint",
                "Min/max speakers must be whole numbers between 1 and 10.")
            return
        self._diarizing = True
        self.diarize_progress.setVisible(True)
        self.diarize_progress.setValue(0)
        if mode == "full":
            self.diarize_label.setText(
                "Full diarization on CPU — this can take minutes. "
                "The UI stays responsive; please wait.")
        elif mode == "sherpa":
            self.diarize_label.setText(
                "Sherpa diarization on CPU — models download once "
                "(~33 MB) on first use, then it runs offline. "
                "This can take a while; the UI stays responsive.")
        else:
            self.diarize_label.setText(
                "Simple diarization (mic vs system energy)…")
        self._update_buttons()
        self._diarize_thread = QThread(self)
        self._diarize_worker = _DiarizeWorker(
            self.session_dir, mode, min_sp, max_sp,
            self.hf_token_edit.text().strip() or None,
            sherpa_model_dir=self.config.sherpa_model_dir,
            sherpa_expected_speakers=self.config.sherpa_expected_speakers,
            sherpa_clustering_threshold=(
                self.config.sherpa_clustering_threshold))
        self._diarize_worker.moveToThread(self._diarize_thread)
        self._diarize_thread.started.connect(self._diarize_worker.run)
        self._diarize_worker.progress.connect(self._on_diarize_progress)
        self._diarize_worker.finished.connect(self._on_diarize_finished)
        self._diarize_worker.error.connect(self._on_diarize_error)
        self._diarize_worker.finished.connect(self._diarize_thread.quit)
        self._diarize_worker.error.connect(self._diarize_thread.quit)
        self._diarize_thread.start()

    def _on_diarize_progress(self, fraction: float, text: str) -> None:
        self.diarize_progress.setValue(int(fraction * 100))
        self.diarize_label.setText(text)

    def _on_diarize_finished(self, summary: dict) -> None:
        self._diarizing = False
        self.diarize_progress.setValue(100)
        # Hide the bar when the step finishes (see
        # _on_transcribe_finished — a bar left visible at 100%
        # looks like a dead button).
        self.diarize_progress.setVisible(False)
        self.diarize_label.setText(
            f"Done ({summary['mode']}/{summary['method']}): "
            f"{summary['segments']} segments, speakers: "
            f"{', '.join(summary['speakers']) or '—'}. "
            "diarization.json written; transcript.txt/.srt updated.")
        self.statusBar().showMessage(
            f"Diarization saved in {summary['session_dir']}")
        # Diarization rewrote transcript.json with speaker labels —
        # regenerate the cleaned artifact so it carries them too.
        self._run_cleanup()
        self._refresh_segments_list()
        self._refresh_rename_labels(summary["speakers"])
        self._refresh_session_chooser()
        self._update_loaded_session_label()
        self._update_buttons()
        # Auto-process chain: diarization done → suggest + identify.
        if self._auto_processing:
            self.status_label.setText(
                "Auto-processing: identifying speakers…")
            self._start_suggest(auto_identify=True)

    def _on_diarize_error(self, message: str) -> None:
        # Auto-process chain stops here, loudly (the dialog below);
        # the recording was saved before the chain started.
        self._auto_processing = False
        self._diarizing = False
        self.diarize_progress.setVisible(False)
        self.diarize_label.setText(f"Diarization failed: {message}")
        QMessageBox.critical(self, "Diarization failed", message)
        self._update_buttons()

    def _refresh_segments_list(self) -> None:
        """Fill the clickable segment list from the transcript.

        Which file depends on the Original / Cleaned /
        Enterprise-cleaned toggle: the verbatim transcript.json,
        the rule-cleaned derived artifact transcript-clean.json
        (Nov 10 2026), or the text-only enterprise artifact
        transcript-enterprise-clean.txt (Nov 21 2026). A derived
        view with no artifact yet shows a placeholder pointing
        at the button that makes it — the verbatim transcript is
        never hidden behind a missing derived file. The
        enterprise variant has no timings (it is plain text),
        so its lines are shown as non-clickable text — timings
        are never invented for it.
        """
        self.segments_list.clear()
        if not self.session_dir:
            self.segments_list.addItem(self._placeholder_item(
                "No transcript yet — record a meeting, then "
                "transcribe it."))
            return
        view = self.transcript_view_combo.currentData() or "original"
        if view == "enterprise":
            path = transcript_variant_path(self.session_dir, "enterprise")
            if not path.exists():
                self.segments_list.addItem(self._placeholder_item(
                    "No enterprise-cleaned transcript yet — press "
                    "\"Enterprise-clean\" to make one. The transcript "
                    "text leaves this machine for that step; audio "
                    "never does."))
                return
            try:
                lines = path.read_text(
                    encoding="utf-8").splitlines()
            except Exception:  # noqa: BLE001 — a bad file is not fatal
                return
            shown = 0
            for line in lines:
                stripped = line.strip()
                # Header lines (# / >) are the artifact's own
                # provenance note, not transcript content.
                if not stripped or stripped.startswith(("#", ">")):
                    continue
                item = QListWidgetItem(stripped)
                item.setFlags(item.flags()
                              & ~Qt.ItemFlag.ItemIsSelectable)
                self.segments_list.addItem(item)
                shown += 1
            if not shown:
                self.segments_list.addItem(self._placeholder_item(
                    "The enterprise-cleaned transcript is empty."))
            return
        if view == "cleaned":
            path = self.session_dir / CLEAN_JSON
            if not path.exists():
                self.segments_list.addItem(self._placeholder_item(
                    "No cleaned transcript yet — press Re-clean to "
                    "make one from the verbatim transcript."))
                return
        else:
            path = self.session_dir / TRANSCRIPT_JSON
            if not path.exists():
                self.segments_list.addItem(self._placeholder_item(
                    "No transcript yet — transcribe this session to see "
                    "it here."))
                return
        try:
            import json as _json
            payload = _json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a bad transcript is not fatal
            return
        names = load_speaker_names(self.session_dir)
        for seg in payload.get("segments", []):
            speaker = seg.get("speaker")
            display = names.get(speaker, speaker) if speaker else "—"
            minutes, seconds = divmod(int(seg["start"]), 60)
            self.segments_list.addItem(
                f"[{minutes:02d}:{seconds:02d}] {display}: {seg['text']}")
            item = self.segments_list.item(
                self.segments_list.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole,
                         (float(seg["start"]), float(seg["end"])))
            # Speaker colour chip (Nov 21 2026): the segment's
            # speaker colours its text, from the fixed palette.
            item.setForeground(
                QColor(color_for_speaker(str(display))))

    def _refresh_rename_labels(self, speakers: list[str]) -> None:
        self.rename_label_combo.clear()
        self.rename_label_combo.addItems(sorted(speakers))
        self._update_buttons()

    # ------------------------------------------- session loading (Nov 10) ---
    def load_session(self, session_dir) -> None:
        """Load ``session_dir`` as the session every view works on.

        Sets ``self.session_dir`` and refreshes the Transcript
        segments, the rename labels, the known speakers, the stored
        suggestions, the near-live pane, the Session-view chooser
        (kept in sync — the library-open and the chooser are the
        same path) and the "Loaded session" line, plus the button
        states for that session. A session with no transcript yet
        opens cleanly: empty transcript state, Transcribe enabled.
        Raises FileNotFoundError for a missing folder and
        RuntimeError while a recording is in progress (opening
        another session mid-recording would strand the stop logic,
        which finalises ``self.session_dir``).
        """
        session_dir = Path(session_dir)
        if not session_dir.is_dir():
            raise FileNotFoundError(
                f"No such session folder: {session_dir}")
        state = self.capture.state if self.capture else CaptureState.IDLE
        if state in (CaptureState.RECORDING, CaptureState.PAUSED):
            raise RuntimeError(
                "Cannot open another session while recording — "
                "stop the recording first.")
        self.session_dir = session_dir
        # A newly loaded session has no live state: the previous
        # recording's chunker / watermark must never leak into the
        # next recording's finalize (on_record builds a fresh one).
        self._live_chunker = None
        self._live_busy = False
        self._live_failed = False
        self._live_list_started = False
        # Outlook calendar linking (Nov 10 2026, Feature 3): an old
        # session with no stored link gets one attempt here — a
        # session that was already checked (linked, or stored null)
        # is not re-queried on every load. Fully guarded inside
        # link_session(); a missing Outlook never blocks opening.
        try:
            metadata = load_metadata(session_dir)
        except Exception:  # noqa: BLE001 — a bad metadata file is not fatal
            metadata = {}
        if "calendar_event" not in metadata:
            try:
                link_session(session_dir,
                             enabled=self.config.outlook_link_enabled)
            except Exception:  # noqa: BLE001 — linking never fails a load
                logger.exception("Outlook linking failed on load")
        self._suggestion_objects = []
        self._suggestion_dicts = []
        # Reset the per-run status lines: they describe a run on the
        # previous session, not this one.
        self.transcribe_label.setText("")
        self.diarize_label.setText("")
        self.export_label.setText("")
        self.summarize_label.setText("")
        self.cleanup_label.setText("")
        self.nearlive_label.setText(self._nearlive_hint_text())
        self._refresh_live_model_label()
        for bar in (self.transcribe_progress, self.diarize_progress,
                    self.nearlive_progress):
            bar.setVisible(False)
            bar.setValue(0)
        self._refresh_segments_list()
        self._refresh_rename_labels(self._transcript_speakers())
        self._refresh_known_speakers()
        self._refresh_suggestions_from_disk()
        # Session-view extras (Nov 21 2026): bookmarks, the
        # waveform timeline, and the dictation note for this
        # session. Follow-along state resets with the session.
        self._follow_segment_index = None
        self._refresh_bookmarks_list()
        self._refresh_waveform()
        self._refresh_dictation_note()
        self.nearlive_list.clear()
        self.nearlive_list.addItem(self._placeholder_item(
            "Nothing here yet — run near-live on a recorded "
            "session."))
        self._refresh_session_chooser(select=session_dir)
        self._refresh_session_source_editor()
        self._update_loaded_session_label()
        self._update_buttons()
        # The calendar grid shows the loaded session's month state
        # too (a newly recorded session appears on its day).
        calendar = getattr(self, "calendar_widget", None)
        if calendar is not None:
            calendar.refresh()
        self.statusBar().showMessage(
            f"Loaded session {session_dir.name}")

    # ---------------------------------------------------------- navigation ---
    def show_view(self, name: str) -> None:
        """Show the view called ``name`` (a VIEW_NAMES entry)."""
        if name not in VIEW_NAMES:
            raise ValueError(
                f"Unknown view {name!r}; views: {list(VIEW_NAMES)}")
        self.tabs.setCurrentIndex(VIEW_NAMES.index(name))

    # ``set_view`` is the same call under the name some callers use.
    set_view = show_view

    def _on_nav_row_changed(self, row: int) -> None:
        """A nav row was picked — switch the stack to that view.

        Every nav row is a view (flat nav since Nov 20 2026 — no
        "More" header); the guard is for safety (a row with no
        name leaves the view alone).
        """
        item = self.nav_list.item(row)
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(name, str) and name in VIEW_NAMES:
            self.tabs.setCurrentIndex(VIEW_NAMES.index(name))

    def _on_view_changed(self, index: int) -> None:
        """A view was shown — keep the nav in sync and refresh it.

        Library: refresh the session list, so a session recorded
        (or a folder moved) since the view was last visible
        appears. Calendar: refresh the grid (the refresh Library
        used to do before the split). Settings: rebuild the form
        from the live config, so the fields never go stale.
        """
        if not 0 <= index < len(VIEW_NAMES):
            return
        name = VIEW_NAMES[index]
        for row in range(self.nav_list.count()):
            item = self.nav_list.item(row)
            if (item is not None
                    and item.data(Qt.ItemDataRole.UserRole) == name):
                self.nav_list.blockSignals(True)
                try:
                    self.nav_list.setCurrentRow(row)
                finally:
                    self.nav_list.blockSignals(False)
                break
        if name == "Library":
            self._refresh_library()
            self._refresh_disk_line()
        elif name == "Calendar":
            calendar = getattr(self, "calendar_widget", None)
            if calendar is not None:
                calendar.refresh()
            # The filter wraps refresh (calendar_page), but a
            # direct refresh call needs the filter re-applied.
            self._refresh_calendar_speaker_options()
            self._apply_calendar_speaker_filter()
        elif name == "People":
            # Always land on the list layer, never a stale person
            # page from a previous visit.
            self._show_people_list()
        elif name == "Settings":
            # Do not rebuild a dirty form: navigating away and back
            # must keep the user's unsaved edits (see
            # _settings_form_dirty). Discard / Save rebuild
            # explicitly via _refresh_settings_view.
            if not self._settings_form_dirty():
                self._refresh_settings_view()

    def _on_tab_changed(self, index: int) -> None:
        """Back-compat alias for :meth:`_on_view_changed`."""
        self._on_view_changed(index)

    # ------------------------------------------------------------- Library ---
    def _refresh_library_source_options(self) -> None:
        """Rebuild the Library source filter from the library's sources."""
        previous = self.library_source_combo.currentData()
        try:
            options = list_sources(Path(self.config.output_root))
        except Exception:  # noqa: BLE001 — filter must not break the view
            options = []
        self.library_source_combo.blockSignals(True)
        try:
            self.library_source_combo.clear()
            self.library_source_combo.addItem("All sources", None)
            for value in options:
                self.library_source_combo.addItem(
                    SOURCE_LABELS.get(value, value), value)
            index = self.library_source_combo.findData(previous)
            self.library_source_combo.setCurrentIndex(
                index if index >= 0 else 0)
        finally:
            self.library_source_combo.blockSignals(False)

    def _refresh_library(self, query: str | None = None) -> None:
        """Re-list the Library for the current search / filter.

        Text search matches transcript text, speaker names, session
        id and app name; Semantic / Hybrid go through the MiniLM
        index and fall back to text when the model is unavailable
        (same behaviour as the Recording Browser).
        """
        if not hasattr(self, "library_list"):
            return
        root = Path(self.config.output_root)
        if query is None:
            query = self.library_search_edit.text()
        mode = self.library_mode_combo.currentData() or "text"
        self._refresh_library_source_options()
        self._refresh_library_speaker_options()
        selected_source = self.library_source_combo.currentData()
        selected_speaker = self._selected_library_speaker()
        self._library_summaries: list = []
        self._library_hits: list = []
        self.library_list.clear()
        if not query:
            try:
                summaries = list_session_summaries(
                    root, source=selected_source,
                    speaker=selected_speaker)
            except Exception:  # noqa: BLE001 — list must not break the view
                summaries = []
            # Pinned-first, then newest, is the core's order
            # (list_session_summaries). Week-group headers are
            # added here, only on the unfiltered list: a header
            # row carries a None parallel entry, so selection
            # maps back to the session rows exactly.
            current_header = None
            for summary in summaries:
                header = _week_header_for(summary)
                if header != current_header:
                    current_header = header
                    header_item = QListWidgetItem(header)
                    header_item.setFlags(Qt.ItemFlag.NoItemFlags)
                    header_font = QFont()
                    header_font.setBold(True)
                    header_item.setFont(header_font)
                    self.library_list.addItem(header_item)
                    self._library_summaries.append(None)
                self._library_summaries.append(summary)
                title, meta = library_row_lines(summary)
                line = f"{title}\n{meta}"
                item = QListWidgetItem(line)
                item.setToolTip(line)
                self.library_list.addItem(item)
            if summaries:
                self.library_status_label.setText(
                    f"{len(summaries)} session(s) in {root}")
            else:
                self.library_status_label.setText(
                    "No sessions yet — record your first meeting "
                    "and it will appear here.")
            self._update_library_action_buttons()
            return
        if mode == "text":
            summaries = search_sessions(
                root, query, mode="text", source=selected_source,
                speaker=selected_speaker)
            self._library_summaries = list(summaries)
            for summary in summaries:
                title, meta = library_row_lines(summary)
                line = f"{title}\n{meta}"
                item = QListWidgetItem(line)
                item.setToolTip(line)
                self.library_list.addItem(item)
            self.library_status_label.setText(
                f"{len(summaries)} session(s) match “{query}”"
                if summaries else
                f"No sessions match “{query}” — try different words, "
                "or switch to Semantic search.")
            self._update_library_action_buttons()
            return
        # Semantic / hybrid: first use downloads the MiniLM model
        # (~22 MB) and builds the index — say so first.
        self.library_status_label.setText(
            "Building search index — first use downloads the MiniLM "
            "model (~22 MB)…")
        try:
            index = SemanticSearchIndex(
                root, model_dir=self.config.semantic_model_dir)
            if mode == "hybrid":
                hits = hybrid_search(root, query, index=index)
            else:
                hits = index.semantic_search(query)
        except ModelNotAvailableError as exc:
            summaries = search_sessions(
                root, query, mode="text", source=selected_source,
                speaker=selected_speaker)
            self._library_summaries = list(summaries)
            for summary in summaries:
                title, meta = library_row_lines(summary)
                line = f"{title}\n{meta}"
                item = QListWidgetItem(line)
                item.setToolTip(line)
                self.library_list.addItem(item)
            self.library_status_label.setText(
                f"Semantic search unavailable: {exc} Showing "
                f"{len(summaries)} text match(es) instead."
                if summaries else
                f"No sessions match “{query}” — Semantic search "
                f"unavailable: {exc}")
            self._update_library_action_buttons()
            return
        if selected_source is not None or selected_speaker is not None:
            by_id = {s.session_id: s
                     for s in list_session_summaries(root)}
            hits = [h for h in hits
                    if by_id.get(h.session_id) is not None
                    and (selected_source is None
                         or by_id[h.session_id].source == selected_source)
                    and (selected_speaker is None
                         or any(str(s).lower()
                                == str(selected_speaker).lower()
                                for s in by_id[h.session_id].speakers))]
        self._library_hits = hits
        summaries = {s.session_id: s
                     for s in list_session_summaries(root)}
        from app.storage.browser import SessionSummary
        from app.transcription.exports import format_txt_timestamp
        for hit in hits:
            hit_summary = summaries.get(hit.session_id) or SessionSummary(
                session_id=hit.session_id,
                path=root / hit.session_id)
            title, _meta = library_row_lines(hit_summary)
            span = (f"{format_txt_timestamp(hit.start_s)}–"
                    f"{format_txt_timestamp(hit.end_s)}")
            line = f"{title}\n{span} {hit.snippet}"
            item = QListWidgetItem(line)
            item.setToolTip(line)
            self.library_list.addItem(item)
            self._library_summaries.append(hit_summary)
        sessions = {h.session_id for h in hits}
        self.library_status_label.setText(
            f"{len(hits)} chunk(s) in {len(sessions)} session(s) "
            f"— {mode} search" if hits else
            f"No matches for “{query}” in {mode.capitalize()} "
            "search — try different words, or switch to Text search.")
        self._update_library_action_buttons()

    def _selected_library_summary(self):
        """The session the selected Library row belongs to, or None."""
        row = self.library_list.currentRow()
        if row < 0 or row >= len(getattr(
                self, "_library_summaries", [])):
            return None
        return self._library_summaries[row]

    def _update_library_open_button(self) -> None:
        """Enable Open only when a Library session is selected."""
        enabled = self._selected_library_summary() is not None
        self.library_open_btn.setEnabled(enabled)
        self.library_open_btn.setToolTip(
            "Load the selected session into the Session view."
            if enabled else "Select a session first.")

    def _update_library_action_buttons(self) -> None:
        """Enable/disable the Library row buttons for the selection.

        Rename stays disabled for Outlook-linked sessions (their
        name is the calendar event's subject); Delete needs only a
        selected session row.
        """
        self._update_library_open_button()
        self._update_library_star_button()
        summary = self._selected_library_summary()
        rename_btn = getattr(self, "library_rename_btn", None)
        if rename_btn is not None:
            if summary is None:
                rename_btn.setEnabled(False)
                rename_btn.setToolTip("Select a session first.")
            elif summary.calendar_linked:
                rename_btn.setEnabled(False)
                rename_btn.setToolTip(
                    "Named by its Outlook calendar event")
            else:
                rename_btn.setEnabled(True)
                rename_btn.setToolTip("Rename the selected session.")
        delete_btn = getattr(self, "library_delete_btn", None)
        if delete_btn is not None:
            delete_btn.setEnabled(summary is not None)
            delete_btn.setToolTip(
                "Delete the selected session permanently."
                if summary is not None else "Select a session first.")

    def _live_session_dir(self) -> Path | None:
        """The directory being recorded right now, or None."""
        state = self.capture.state if self.capture else CaptureState.IDLE
        if state in (CaptureState.RECORDING, CaptureState.PAUSED):
            return self.session_dir
        return None

    def _session_has_active_work(self) -> bool:
        """Whether a processing worker thread is still running."""
        names = ("_transcribe_thread", "_diarize_thread",
                 "_suggest_thread", "_nearlive_thread",
                 "_summarize_thread", "_enterprise_thread",
                 "_polish_thread", "_finalize_thread",
                 "_stop_thread", "_waveform_thread", "_live_thread")
        for name in names:
            thread = getattr(self, name, None)
            if thread is not None and thread.isRunning():
                return True
        return False

    def _on_library_context_menu(self, pos) -> None:
        """Right-click menu on a Library row: Delete the session."""
        item = self.library_list.itemAt(pos)
        if item is None:
            return
        row = self.library_list.row(item)
        if row < 0 or row >= len(getattr(self, "_library_summaries", [])):
            return
        if self._library_summaries[row] is None:
            return  # week header, not a session
        self.library_list.setCurrentRow(row)
        self._update_library_action_buttons()
        menu = QMenu(self)
        delete_action = menu.addAction("Delete session…")
        if menu.exec(self.library_list.mapToGlobal(pos)) == delete_action:
            self._on_library_delete()

    def _on_library_delete(self) -> None:
        """Delete the selected Library session, after confirmation."""
        summary = self._selected_library_summary()
        if summary is None:
            return
        session_dir = Path(summary.path)
        live_dir = self._live_session_dir()
        if live_dir is not None and Path(live_dir) == session_dir:
            QMessageBox.warning(
                self, "Cannot delete this session",
                "This session is being recorded — stop the recording "
                "first.")
            return
        if (self.session_dir is not None
                and Path(self.session_dir) == session_dir
                and self._session_has_active_work()):
            QMessageBox.warning(
                self, "Cannot delete this session",
                "This session is still being processed — wait for "
                "the processing to finish, then delete it.")
            return
        answer = QMessageBox.question(
            self, "Delete session?",
            f"Delete “{summary.display_name}” "
            f"({row_date_text(summary)}) permanently?\n\n"
            "Its audio, transcript, search entries and session "
            "record will be permanently removed. This cannot be "
            "undone.",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_session(session_dir, live_dir=live_dir)
        except (RuntimeError, FileNotFoundError, OSError) as exc:
            QMessageBox.warning(
                self, "Could not delete session", str(exc))
            return
        was_loaded = (self.session_dir is not None
                      and Path(self.session_dir) == session_dir)
        if was_loaded:
            self._unload_session()
        self._refresh_library()
        self._refresh_disk_line()
        self.statusBar().showMessage(
            f"Deleted session {session_dir.name}")

    def _unload_session(self) -> None:
        """Clear the loaded session (its folder was deleted).

        Mirrors the reset half of :meth:`load_session`: every view
        returns to the no-session empty state, and the Session
        chooser falls back to the newest remaining session.
        """
        self.session_dir = None
        self._live_chunker = None
        self._live_busy = False
        self._live_failed = False
        self._live_list_started = False
        self._suggestion_objects = []
        self._suggestion_dicts = []
        self.transcribe_label.setText("")
        self.diarize_label.setText("")
        self.export_label.setText("")
        self.summarize_label.setText("")
        self.cleanup_label.setText("")
        self.nearlive_label.setText(self._nearlive_hint_text())
        self._refresh_live_model_label()
        for bar in (self.transcribe_progress, self.diarize_progress,
                    self.nearlive_progress):
            bar.setVisible(False)
            bar.setValue(0)
        self._follow_segment_index = None
        self._refresh_segments_list()
        self._refresh_rename_labels([])
        self._refresh_known_speakers()
        self._refresh_suggestions_from_disk()
        self._refresh_bookmarks_list()
        self._refresh_waveform()
        self._refresh_dictation_note()
        self.nearlive_list.clear()
        self.nearlive_list.addItem(self._placeholder_item(
            "Nothing here yet — run near-live on a recorded "
            "session."))
        self._refresh_session_chooser()
        self._refresh_session_source_editor()
        self._update_loaded_session_label()
        self._update_buttons()
        calendar = getattr(self, "calendar_widget", None)
        if calendar is not None:
            calendar.refresh()

    def _on_library_rename(self) -> None:
        """Rename the selected Library session (custom name)."""
        summary = self._selected_library_summary()
        if summary is None or summary.calendar_linked:
            return
        text, ok = QInputDialog.getText(
            self, "Rename session", "Session name:",
            text=summary.display_name)
        if not ok:
            return
        try:
            rename_session(summary.path, text)
        except ValueError as exc:
            QMessageBox.warning(
                self, "Cannot rename session", str(exc))
            return
        self._refresh_library()
        if (self.session_dir is not None
                and Path(self.session_dir) == Path(summary.path)):
            self._update_loaded_session_label()
            self._refresh_session_chooser(select=self.session_dir)
        self.statusBar().showMessage(
            f"Renamed session to “{text.strip()}”")

    def _on_library_search_changed(self, text: str) -> None:
        self._refresh_library(query=text)

    def _on_library_mode_changed(self, _index: int) -> None:
        self._refresh_library()

    def _on_library_source_changed(self, _index: int) -> None:
        self._refresh_library()

    def _on_library_clicked(self, _item) -> None:
        self._update_library_action_buttons()

    def _on_library_open(self, _item=None) -> None:
        """Load the selected Library session in the Session view."""
        summary = self._selected_library_summary()
        if summary is None:
            self.library_status_label.setText("Select a session first.")
            return
        try:
            self.load_session(summary.path)
        except (RuntimeError, FileNotFoundError) as exc:
            self.library_status_label.setText(
                f"Could not open session: {exc}")
            return
        self.show_view("Session")

    # ------------------------------------------------------------ Settings ---
    def _settings_form_dirty(self) -> bool:
        """True when the embedded Settings form has unsaved edits.

        Compares the form against the snapshot taken when it was
        last built (config fields plus the names-dictionary text,
        which is not part of AppConfig). A form that cannot be
        read back counts as dirty — never discard what cannot be
        compared. No dialog yet (first build) counts as clean.
        """
        dialog = getattr(self, "_settings_dialog", None)
        snapshot = getattr(self, "_settings_form_snapshot", None)
        if dialog is None or snapshot is None:
            return False
        saved_config, saved_names_text = snapshot[0], snapshot[1]
        try:
            if dialog.config_from_dialog() != saved_config:
                return True
        except Exception:  # noqa: BLE001 — an unreadable form is dirty
            return True
        names_edit = getattr(dialog, "names_edit", None)
        if names_edit is not None:
            try:
                if names_edit.toPlainText() != saved_names_text:
                    return True
            except Exception:  # noqa: BLE001 — an unreadable form is dirty
                return True
        # Advanced section (Nov 21 2026): the pre-roll /
        # enterprise controls live outside the dialog form, so
        # their values join the snapshot as a plain tuple.
        if len(snapshot) > 2:
            if self._advanced_settings_values() != snapshot[2]:
                return True
        return False

    def _advanced_settings_values(self) -> tuple:
        """Current values of the advanced Settings controls."""
        return (
            self.preroll_check.isChecked(),
            self.preroll_seconds_spin.value(),
            self.enterprise_check.isChecked(),
            self.tinyfix_check.isChecked(),
            self.tinyfix_mode_combo.currentData(),
        )

    def _sync_advanced_settings_controls(self) -> None:
        """Set the advanced controls from the live config.

        Called after the form rebuild (never while the user is
        editing — the dirty check keeps a dirty form, and with
        it these controls, untouched on navigation).
        """
        self.preroll_check.blockSignals(True)
        self.enterprise_check.blockSignals(True)
        self.tinyfix_check.blockSignals(True)
        try:
            self.preroll_check.setChecked(self.config.preroll_enabled)
            self.preroll_seconds_spin.setValue(
                int(self.config.preroll_seconds))
            self.enterprise_check.setChecked(
                self.config.enterprise_cleanup_enabled)
            self.tinyfix_check.setChecked(self.config.tinyfix_enabled)
            mode_index = self.tinyfix_mode_combo.findData(
                self.config.tinyfix_mode)
            if mode_index >= 0:
                self.tinyfix_mode_combo.setCurrentIndex(mode_index)
        finally:
            self.preroll_check.blockSignals(False)
            self.enterprise_check.blockSignals(False)
            self.tinyfix_check.blockSignals(False)
        self.preroll_seconds_spin.setEnabled(
            self.config.preroll_enabled)
        self.tinyfix_mode_combo.setEnabled(self.config.tinyfix_enabled)
        # The form now shows the config's values, so a click
        # recorded against the previous form is spent: the
        # explicit flag resets to the config's flag (a later
        # click sets it again — see _on_tinyfix_clicked).
        self._tinyfix_explicit = bool(
            self.config.tinyfix_enabled_explicit)

    def _on_tinyfix_toggled(self, checked: bool) -> None:
        """tinyfix checkbox in the advanced Settings section.

        Enables the mode combo; the setting itself is applied
        when the form is saved (the dirty-check snapshot carries
        the checkbox + combo values — see _settings_form_dirty).
        """
        self.tinyfix_mode_combo.setEnabled(checked)

    def _on_tinyfix_clicked(self, _checked: bool) -> None:
        """A user-driven click on the tinyfix checkbox (Nov 22 2026).

        ``clicked`` fires only on a user click — never on a
        programmatic setChecked — so this is the one place the
        explicit flag is ever set True (mirrors the model combo's
        ``activated`` tracking for whisper_model_explicit). The
        flag is written on save; a plain save without a click
        keeps the incoming flag.
        """
        self._tinyfix_explicit = True

    def _refresh_settings_view(self) -> None:
        """Rebuild the embedded Settings form from the live config.

        A fresh SettingsDialog is built and its form taken out of
        the dialog's scroll area into the Settings view's form
        holder — so the fields always show the saved settings,
        including after the output folder moved. The dialog itself
        is never shown; it just owns the form widgets until Save.
        The previous dialog is deleted (it was leaking one per
        rebuild); the form it owned was already taken out, so
        deleting it cannot take the visible form with it. This
        rebuild is unconditional — navigation guards it with
        :meth:`_settings_form_dirty`, while Discard and Save call
        it directly because they mean "throw the form away".
        """
        from app.ui.settings_dialog import SettingsDialog

        old_dialog = getattr(self, "_settings_dialog", None)
        holder_layout = self._settings_form_holder.layout()
        old_form = None
        if holder_layout.count():
            old_form = holder_layout.takeAt(0).widget()
        if old_form is not None:
            old_form.setParent(None)
        dialog = SettingsDialog(self.config, parent=self)
        self._settings_dialog = dialog
        try:
            snapshot_config = dialog.config_from_dialog()
        except Exception:  # noqa: BLE001 — a fresh form always reads back
            snapshot_config = None
        names_edit = getattr(dialog, "names_edit", None)
        try:
            snapshot_names = (names_edit.toPlainText()
                              if names_edit is not None else "")
        except Exception:  # noqa: BLE001 — snapshot is best-effort
            snapshot_names = ""
        # Sync the advanced controls from the live config BEFORE
        # snapshotting them: the controls persist across form
        # rebuilds, so the snapshot must be the config's values,
        # not whatever edits the previous form left behind.
        self._sync_advanced_settings_controls()
        self._settings_form_snapshot = (
            snapshot_config, snapshot_names,
            self._advanced_settings_values())
        form = dialog._form_scroll.takeWidget()
        self._settings_form_holder.layout().addWidget(form)
        if old_dialog is not None:
            old_dialog.deleteLater()
        # Disk line (Nov 21 2026): refreshed with the form.
        self._refresh_disk_line()

    def _on_settings_save(self) -> None:
        """Save the Settings view's form and apply it, like OK."""
        dialog = getattr(self, "_settings_dialog", None)
        if dialog is None:
            return
        try:
            new_config = dialog.apply_and_save()
        except Exception as exc:  # noqa: BLE001 — surfaced to the user
            QMessageBox.critical(self, "Settings failed", str(exc))
            return
        # Advanced section overlay (Nov 21 2026): the dialog
        # form does not edit pre-roll / enterprise / disk fields,
        # and config_from_dialog() constructs them from defaults
        # — a plain save would reset them. Overlay the advanced
        # controls' values onto the dialog-derived config, and
        # preserve the disk threshold from the running config
        # (no control edits it). _apply_saved_config re-saves.
        from dataclasses import replace

        preroll_on, preroll_seconds, enterprise_on, tinyfix_on, \
            tinyfix_mode = self._advanced_settings_values()
        # The mode combo always has a selection, but never let a
        # missing one poison the saved config.
        if tinyfix_mode not in ("balanced", "careful"):
            tinyfix_mode = "careful"
        new_config = replace(
            new_config,
            preroll_enabled=preroll_on,
            preroll_seconds=preroll_seconds,
            enterprise_cleanup_enabled=enterprise_on,
            tinyfix_enabled=tinyfix_on,
            tinyfix_mode=tinyfix_mode,
            tinyfix_enabled_explicit=self._tinyfix_explicit,
            disk_warning_threshold_bytes=(
                self.config.disk_warning_threshold_bytes))
        self._apply_saved_config(new_config)
        self._refresh_settings_view()

    def _apply_saved_config(self, new_config) -> None:
        """Apply a saved config to the running window (post-save).

        Shared by the Settings view's Save and the old dialog path:
        theme at the app level, meters, the model / diarization /
        capture-mode dropdowns, the known speakers, the output
        folder follow-up and the auto-detect watcher.
        """
        from app.config import save_settings

        self.config = new_config
        self._tinyfix_explicit = bool(
            new_config.tinyfix_enabled_explicit)
        save_settings(new_config)
        try:
            app = QApplication.instance()
            if app is not None:
                apply_theme_to_app(app, new_config.theme)
            apply_theme(self, new_config.theme)
        except Exception:  # noqa: BLE001 — theme must not block settings
            logger.warning("Could not apply theme %r", new_config.theme,
                           exc_info=True)
        self.mic_meter.set_theme(new_config.theme)
        self.system_meter.set_theme(new_config.theme)
        try:
            self.model_combo.setCurrentText(
                self.model_combo.itemText(
                    list(WHISPER_MODELS).index(
                        new_config.effective_whisper_model)))
            mode_idx = self.diarize_mode_combo.findData(
                new_config.diarization_mode)
            if mode_idx >= 0:
                self.diarize_mode_combo.setCurrentIndex(mode_idx)
            self._update_diarize_mode_ui()
            self._update_capture_mode_ui()
            self._refresh_live_model_label()
        except Exception:  # noqa: BLE001 — dropdown sync is best-effort
            pass
        self.statusBar().showMessage(
            f"Settings saved. Output: {new_config.output_root}")
        self.status_label.setText(
            "Settings saved (Hugging Face token is never stored — "
            "only whether one is configured).")
        self._refresh_known_speakers()
        self._refresh_after_output_root_change()
        self._apply_auto_detect_setting()
        # Nov 21 2026: the pre-roll feeder follows the saved
        # settings (rebuilt, never started while recording);
        # the theme toggle text and the disk line follow the
        # saved theme / output root.
        self._setup_preroll_feeder()
        self._refresh_theme_toggle_text()
        self._refresh_disk_line()
        self._update_buttons()

    def _on_calendar_open_session(self, path: str) -> None:
        """A calendar badge was clicked: load that session.

        The same :meth:`load_session` path as opening from the
        Recording Browser, then switch to the Session view so the
        loaded session is what the user sees. Mid-recording (or a
        folder that vanished since the grid was built) shows a
        status message instead of raising into the click handler.
        """
        try:
            self.load_session(Path(path))
        except (RuntimeError, FileNotFoundError) as exc:
            self.statusBar().showMessage(
                f"Could not open session: {exc}")
            return
        self.show_view("Session")

    def _transcript_speakers(self) -> list[str]:
        """Sorted speaker labels in the loaded session's transcript."""
        if not self.session_dir:
            return []
        path = self.session_dir / TRANSCRIPT_JSON
        if not path.exists():
            return []
        try:
            segments = load_transcript_json(path)
        except Exception:  # noqa: BLE001 — a bad transcript is not fatal
            logger.warning("Could not read transcript in %s",
                           self.session_dir, exc_info=True)
            return []
        return sorted({str(seg.speaker) for seg in segments
                       if getattr(seg, "speaker", None)})

    def _refresh_suggestions_from_disk(self) -> None:
        """Show the loaded session's stored suggestions, if any.

        ``speaker_suggestions.json`` holds names only — no
        embeddings — so stored rows are a read-back: they show who
        the session was matched to, and Play sample works from the
        row's label, but Confirm/Correct need live suggestion
        objects, so they stay disabled until "Suggest speakers for
        session" regenerates them (fast with the fallback
        extractor). A session with no stored suggestions gets the
        usual empty state.
        """
        self._suggestion_dicts = []
        if not self.session_dir:
            self.suggestions_list.clear()
            self.suggestions_list.addItem(self._placeholder_item(
                "No suggestions yet — they appear automatically after "
                "a recording is processed, or tap “Suggest speakers” "
                "for this session."))
            return
        dicts = load_suggestions(self.session_dir)
        if not dicts:
            self.suggestions_list.clear()
            self.suggestions_list.addItem(self._placeholder_item(
                "No suggestions yet — they appear automatically after "
                "a recording is processed, or tap “Suggest speakers” "
                "for this session."))
            return
        self._suggestion_dicts = list(dicts)
        self._render_suggestion_dicts(dicts)

    def _render_suggestion_dicts(self, dicts) -> None:
        """Fill the suggestions list from suggestion dicts.

        Shared by the live Suggest run and the stored read-back,
        so both render identically (confidence, best match +
        margin, backend, ambiguous flag). Each row carries its
        label in UserRole, so Play sample works from the row alone.
        """
        self.suggestions_list.clear()
        for item in dicts:
            name = item.get("suggested_name") or "Unknown"
            extra = (f" — {item.get('best_name')} "
                     f"{item.get('best_score')} (margin "
                     f"{item.get('margin')})" if item.get("best_name")
                     else "")
            # Phase 5 (Phase 4 follow-up): show which embedding backend
            # produced the suggestion — the fallback is a spectral
            # proxy, not a voice fingerprint (see PHASE4.md).
            backend = item.get("backend")
            backend_text = f" [{backend}]" if backend else ""
            ambiguous = " ⚠ ambiguous" if item.get("ambiguous") else ""
            self.suggestions_list.addItem(
                f"{item['label']}: {name} "
                f"(confidence {item.get('confidence', 0):.2f})"
                f"{extra}{backend_text}{ambiguous}")
            row_item = self.suggestions_list.item(
                self.suggestions_list.count() - 1)
            row_item.setData(Qt.ItemDataRole.UserRole, item["label"])

    def _selected_suggestion_label(self) -> str | None:
        """Label of the selected suggestion row (live or stored).

        The live object wins when there is one; otherwise the row's
        UserRole label — so Play sample works for stored
        suggestions too, even though Confirm/Correct need a live
        object (see _refresh_suggestions_from_disk).
        """
        suggestion = self._selected_suggestion()
        if suggestion is not None:
            return suggestion.label
        item = self.suggestions_list.currentItem()
        if item is None:
            return None
        label = item.data(Qt.ItemDataRole.UserRole)
        return str(label) if label else None

    def _refresh_session_chooser(self, select: Path | None = None) -> None:
        """Rebuild the Session-view session chooser, newest first.

        Rows are ``"<display name> — <status>"`` from
        :func:`list_session_summaries`. ``select`` (a session path)
        is picked when present; otherwise the previous selection is
        kept, else the newest session. Called at startup, after a
        recording stops, after a session is processed, and by
        :meth:`load_session` (so the chooser and the library-open
        stay in sync). Never raises: a failed listing leaves the
        previous list in place.
        """
        try:
            summaries = list_session_summaries(self.config.output_root)
        except Exception:  # noqa: BLE001 — chooser must not break the UI
            logger.exception("Session chooser refresh failed")
            return
        previous: Path | None = None
        if select is not None:
            previous = Path(select)
        else:
            data = self.session_combo.currentData()
            if isinstance(data, str) and data:
                previous = Path(data)
        self.session_combo.blockSignals(True)
        try:
            self.session_combo.clear()
            for summary in summaries:
                self.session_combo.addItem(
                    f"{summary.display_name} — {summary.status_text}",
                    str(summary.path))
            if not summaries:
                self.session_combo.addItem("(no sessions yet)", None)
            target: int | None = None
            if previous is not None:
                for i in range(self.session_combo.count()):
                    item_data = self.session_combo.itemData(i)
                    if (isinstance(item_data, str)
                            and Path(item_data) == previous):
                        target = i
                        break
            if target is None and summaries:
                target = 0
            if target is not None:
                self.session_combo.setCurrentIndex(target)
        finally:
            self.session_combo.blockSignals(False)

    def _on_session_chooser_changed(self, index: int) -> None:
        """A chooser pick loads that session (same path as library-open)."""
        data = self.session_combo.itemData(index)
        if not isinstance(data, str) or not data:
            return
        path = Path(data)
        if self.session_dir is not None and path == self.session_dir:
            return
        state = self.capture.state if self.capture else CaptureState.IDLE
        if state in (CaptureState.RECORDING, CaptureState.PAUSED):
            # Mid-recording the live session is the loaded one —
            # revert the pick instead of stranding the stop logic.
            self._refresh_session_chooser(select=self.session_dir)
            return
        self.load_session(path)

    def _update_loaded_session_label(self) -> None:
        """The Session view's line: which session the view shows.

        Also toggles the Session view's empty state (shown when
        nothing is loaded)."""
        empty = getattr(self, "session_empty_label", None)
        if empty is not None:
            empty.setVisible(self.session_dir is None)
        if self.session_dir is None:
            self.loaded_session_label.setText(
                "No session loaded — record a meeting, or open one "
                "from Recordings.")
            self.loaded_session_label.setToolTip("")
            return
        try:
            summary = summarize_session(self.session_dir)
        except Exception:  # noqa: BLE001 — label must not crash the UI
            summary = None
        name = (summary.display_name if summary is not None
                else self.session_dir.name)
        # start_time is stored as UTC in metadata.json; the folder
        # name is local time — format_local_time converts so the two
        # agree (Nov 10 2026: the raw UTC value showed here before).
        date_text = ""
        if summary is not None:
            date_text = format_local_time(summary.start_time)
        if not date_text:
            # Fall back to the folder-name stamp (YYYY-MM-DD_HH-MM-SS_…).
            date_text = name[:10]
        status = (summary.status_text if summary is not None
                  else "Recorded")
        source_text = ""
        if summary is not None:
            source_text = f" · Source: {summary.source_label}"
        self.loaded_session_label.setText(
            f"Loaded session: {name} — {date_text} · {status}"
            f"{source_text}")
        self.loaded_session_label.setToolTip(str(self.session_dir))
        self._refresh_calendar_link_label()

    def _refresh_calendar_link_label(self) -> None:
        """The Outlook link line under the loaded-session line.

        Reads ``calendar_event`` from the loaded session's
        metadata.json (written by :func:`app.calendar.outlook.
        link_session` at finalise / load time). A stored event
        shows "Linked to: <subject>, <time>" with the Open button;
        a stored ``null`` (match attempted, nothing found) shows
        "No calendar event found" — a plain label, not an error;
        a missing key (never attempted) leaves the line empty.
        Never raises — a bad metadata file just hides the line.
        """
        self.calendar_event_label.setText("")
        self.open_outlook_btn.setVisible(False)
        if self.session_dir is None:
            return
        try:
            metadata = load_metadata(self.session_dir)
        except Exception:  # noqa: BLE001 — label must not crash the UI
            return
        if "calendar_event" not in metadata:
            return
        event = metadata.get("calendar_event")
        if event:
            self.calendar_event_label.setText(format_event_label(event))
            self.open_outlook_btn.setVisible(True)
        else:
            self.calendar_event_label.setText("No calendar event found")

    def _on_open_in_outlook(self) -> None:
        """Open the linked event in Outlook (read-only view)."""
        if self.session_dir is None:
            return
        try:
            metadata = load_metadata(self.session_dir)
            event = metadata.get("calendar_event") or {}
            entry_id = event.get("entry_id", "")
        except Exception:  # noqa: BLE001 — a bad metadata file is not fatal
            entry_id = ""
        if not entry_id:
            return
        if not open_event_in_outlook(entry_id):
            QMessageBox.warning(
                self, "Could not open in Outlook",
                "Outlook could not open the linked event — it may "
                "have been deleted, or Outlook is not running.")

    def _refresh_session_source_editor(self) -> None:
        """Fill the Session-view source editor for the loaded session.

        Signals are blocked while populating, so filling the
        fields never writes back to the session just read. With
        no session loaded the editor shows "Other" with an empty
        detail and stays disabled — there is nothing to edit.
        Called by :meth:`load_session` (the single path every
        session change goes through).
        """
        enabled = self.session_dir is not None
        self.session_source_combo.setEnabled(enabled)
        self.session_source_detail_edit.setEnabled(enabled)
        if not enabled:
            return
        try:
            summary = summarize_session(self.session_dir)
        except Exception:  # noqa: BLE001 — editor must not crash load
            logger.exception("Session source editor refresh failed")
            return
        self.session_source_combo.blockSignals(True)
        try:
            index = self.session_source_combo.findData(summary.source)
            if index < 0:
                index = self.session_source_combo.findData(SOURCE_OTHER)
            self.session_source_combo.setCurrentIndex(index)
        finally:
            self.session_source_combo.blockSignals(False)
        self.session_source_detail_edit.setText(
            summary.source_detail or "")

    def _on_session_source_changed(self, _index: int) -> None:
        """Save a source pick for the loaded session, immediately.

        The editor is the correction path for a wrongly detected
        source — the pick writes straight to metadata.json, so
        the browser filter and the "Loaded session" line follow
        it without a reload.
        """
        if self.session_dir is None:
            return
        source = self.session_source_combo.currentData()
        if not isinstance(source, str) or not source:
            return
        try:
            set_session_source(self.session_dir, source)
        except Exception as exc:  # noqa: BLE001 — surfaced in the UI
            logger.exception("Saving session source failed")
            self.statusBar().showMessage(f"Could not save source: {exc}")
            return
        self._update_loaded_session_label()
        self.statusBar().showMessage(
            f"Source set to {SOURCE_LABELS.get(source, source)} for "
            f"{self.session_dir.name}")

    def _on_session_source_detail_edited(self) -> None:
        """Save the source detail edit (Enter / focus out).

        Writes the detail alongside the editor's current source
        value — the detail alone never changes the source.
        """
        if self.session_dir is None:
            return
        source = self.session_source_combo.currentData()
        if not isinstance(source, str) or not source:
            return
        detail = self.session_source_detail_edit.text().strip()
        try:
            set_session_source(self.session_dir, source,
                               source_detail=detail)
        except Exception as exc:  # noqa: BLE001 — surfaced in the UI
            logger.exception("Saving session source detail failed")
            self.statusBar().showMessage(
                f"Could not save source detail: {exc}")
            return
        self.statusBar().showMessage(
            f"Source detail saved for {self.session_dir.name}")

    def _refresh_after_output_root_change(self) -> None:
        """Output root moved: re-list sessions and follow the chooser.

        The loaded session is kept when it is still in the new
        folder; otherwise the newest session there is loaded (or
        nothing, when the folder is empty — the loaded session
        stays visible until the user picks another). Called after
        the output folder is picked on the main window and after
        Settings are saved. Mid-recording it does nothing: the
        live session stays loaded, and on_stop refreshes the list.
        """
        state = self.capture.state if self.capture else CaptureState.IDLE
        if state in (CaptureState.RECORDING, CaptureState.PAUSED):
            return
        self._refresh_session_chooser()
        # The MCP setup view's status check follows the output
        # folder too (Feature 6).
        mcp_widget = getattr(self, "mcp_setup_widget", None)
        if mcp_widget is not None:
            mcp_widget.set_output_root(self.config.output_root)
        # The calendar grid follows the output folder as well.
        calendar = getattr(self, "calendar_widget", None)
        if calendar is not None:
            calendar.set_output_root(self.config.output_root)
        data = self.session_combo.currentData()
        if isinstance(data, str) and data:
            path = Path(data)
            if path != self.session_dir:
                self.load_session(path)
        self._update_loaded_session_label()

    def on_rename_speaker(self) -> None:
        """Rename the selected label and refresh the exports + list.

        Renaming a label whose current name is a known speaker
        identity (in speakers.db — e.g. "Speaker A" → "Priya")
        renames the identity everywhere it is used: the database
        and every past session. Renaming a plain label
        ("SPEAKER_00" → "Priya") is per-session only, as before.
        """
        if not self.session_dir:
            return
        label = self.rename_label_combo.currentText().strip()
        if not label:
            return
        new_name = self.rename_name_edit.text().strip()
        try:
            names = load_speaker_names(self.session_dir)
            old_display = names.get(label, label)
            if old_display == new_name:
                self.status_label.setText(
                    f"{label} is already called {new_name or old_display}.")
                self.rename_name_edit.clear()
                return
            db = SpeakerDatabase(self._speakers_db_path())
            try:
                is_identity = db.get_speaker(old_display) is not None
            finally:
                db.close()
            if is_identity and new_name:
                db = SpeakerDatabase(self._speakers_db_path())
                try:
                    result = rename_identity(
                        db, self.config.output_root, old_display, new_name)
                finally:
                    db.close()
                updated = result["sessions_updated"]
                skipped = result["sessions_skipped"]
                note = (f" ({len(skipped)} session(s) could not be "
                        "re-exported)" if skipped else "")
                self.status_label.setText(
                    f"Renamed identity {old_display} → {new_name} in "
                    f"{len(updated)} session(s){note} — transcript.txt/"
                    ".srt updated everywhere it was used.")
            else:
                rename_speaker(self.session_dir, label,
                               self.rename_name_edit.text())
                reapply_names(self.session_dir)
                # A rename is a real name Whisper should hear next
                # time — add it to the names dictionary (the
                # identity branch does this inside rename_identity).
                if new_name:
                    try:
                        add_name(new_name)
                    except Exception:  # noqa: BLE001 — never break a rename
                        logger.warning(
                            "Could not add %r to the names dictionary",
                            new_name, exc_info=True)
                self.status_label.setText(
                    f"Renamed {label} — transcript.txt/.srt updated.")
        except Exception as exc:  # noqa: BLE001 — surfaced to the user
            QMessageBox.critical(self, "Rename failed", str(exc))
            return
        self.rename_name_edit.clear()
        self._refresh_segments_list()
        self._refresh_known_speakers()
        self._update_buttons()

    def _play_audio(self, audio: Path, start: float, end: float) -> bool:
        """Play one stretch of audio through the shared player.

        Returns False (with a status reason) when playback is
        unavailable, so a caller never has to guess. The player
        stops at the end of the stretch (best-effort; a few ms of
        drift is fine). Shared by segment click-to-play and the
        speaker voice samples.
        """
        if self._player is None:
            self.status_label.setText(
                "Playback needs QtMultimedia — not installed with this "
                "PyQt6 build. The transcript files are unaffected.")
            return False
        self._player.setSource(QUrl.fromLocalFile(str(audio)))
        self._player.setPosition(int(start * 1000))
        self._player.play()
        QTimer.singleShot(int(max(0.0, (end - start)) * 1000),
                          self._player.stop)
        return True

    def on_segment_clicked(self, item) -> None:
        """Play the clicked segment's audio (best-effort)."""
        if not self.session_dir:
            return
        timing = item.data(Qt.ItemDataRole.UserRole)
        if not timing:
            return
        start, end = timing
        audio = self.session_dir / COMBINED_FILE
        if not audio.exists():
            self.status_label.setText(
                f"Cannot play segment: {COMBINED_FILE} is missing.")
            return
        self._play_audio(audio, start, end)

    def _on_transcribe_error(self, message: str) -> None:
        # Auto-process chain stops here, loudly (the dialog below);
        # the recording was saved before the chain started.
        self._auto_processing = False
        self._transcribing = False
        self._transcribe_cancel_event = None
        self.transcribe_progress.setVisible(False)
        self.transcribe_label.setText(f"Transcription failed: {message}")
        QMessageBox.critical(self, "Transcription failed", message)
        self._update_buttons()
        self._maybe_close_pending()

    # --------------------------------------------- transcript cleanup (Nov 10) ---
    def _run_cleanup(self) -> dict | None:
        """Re-clean the loaded session's transcript (rule layer).

        Best-effort by design: a failure is logged and shown in the
        cleanup label, never raised — cleanup must never break the
        auto-process chain that called it, and the verbatim
        transcript is only ever read, never written. Returns the
        stats dict, or None when there is no transcript / it failed.
        """
        if not self.session_dir:
            return None

        def _tinyfix_status(message: str) -> None:
            # First-use model download: show status text while it
            # runs, never a silent hang. This runs on the UI
            # thread, so repaint the label directly.
            self.cleanup_label.setText(message)
            self.cleanup_label.repaint()

        try:
            stats = clean_session(
                self.session_dir,
                tinyfix_enabled=self.config.tinyfix_enabled,
                tinyfix_mode=self.config.tinyfix_mode,
                tinyfix_status_callback=_tinyfix_status)
        except FileNotFoundError:
            return None
        except Exception:  # noqa: BLE001 — cleanup is best-effort
            logger.exception("Transcript cleanup failed")
            self.cleanup_label.setText(
                "Cleanup failed — see the log. The verbatim "
                "transcript is untouched.")
            return None
        tinyfix_note = ""
        if stats["tinyfix_corrections_applied"]:
            tinyfix_note = (f", {stats['tinyfix_corrections_applied']} "
                            f"tinyfix correction(s)")
        elif stats["tinyfix_skipped"]:
            tinyfix_note = f" (tinyfix skipped: {stats['tinyfix_skipped']})"
        self.cleanup_label.setText(
            f"Cleaned: {stats['fillers_removed']} filler(s), "
            f"{stats['false_starts_repaired']} false start(s), "
            f"{stats['repeated_words_removed']} repeat(s), "
            f"{stats['name_corrections_applied']} name correction(s)"
            f"{tinyfix_note} "
            "— transcript-clean.json/.txt (verbatim kept).")
        return stats

    def _on_transcript_view_changed(self) -> None:
        """Original / Cleaned toggle: re-fill the segment list."""
        self._refresh_segments_list()

    def on_reclean(self) -> None:
        """Re-run the rule cleanup on the loaded session (manual).

        The sensible spot for this is the Session view, next to
        the Original / Cleaned toggle: it is the view that shows
        the result, and it is what you press after a rename or a
        names-dictionary edit.
        """
        if not self.session_dir or self._transcribing:
            return
        stats = self._run_cleanup()
        if stats is not None:
            self._refresh_segments_list()
            self.status_label.setText(
                f"Re-cleaned the transcript — "
                f"{stats['fillers_removed']} filler(s), "
                f"{stats['false_starts_repaired']} false start(s), "
                f"{stats['name_corrections_applied']} name "
                "correction(s). Verbatim transcript untouched.")

    def on_polish(self) -> None:
        """Polish the loaded session with the local LLM (manual).

        Works regardless of the auto-polish setting — clicking it
        is an explicit request. Needs Ollama running on localhost;
        a failure reports clearly and leaves the rule-cleaned
        version in place.
        """
        self._start_polish(quiet=False)

    def _start_polish(self, *, quiet: bool) -> None:
        """Run the polish worker. ``quiet`` (the auto-chain) reports
        failures in the status line only — no modal for a background
        step the user did not ask for."""
        if not self.session_dir or self._polishing:
            return
        self._polishing = True
        self._polish_quiet = quiet
        self.cleanup_label.setText("Polishing with the local LLM…")
        self._update_buttons()
        self._polish_thread = QThread(self)
        self._polish_worker = _PolishWorker(self.session_dir, self.config)
        self._polish_worker.moveToThread(self._polish_thread)
        self._polish_thread.started.connect(self._polish_worker.run)
        self._polish_worker.progress.connect(
            lambda _frac, text: self.cleanup_label.setText(text))
        self._polish_worker.finished.connect(self._on_polish_finished)
        self._polish_worker.error.connect(self._on_polish_error)
        self._polish_worker.finished.connect(self._polish_thread.quit)
        self._polish_worker.error.connect(self._polish_thread.quit)
        self._polish_thread.start()

    def _on_polish_finished(self, path: str) -> None:
        self._polishing = False
        self.cleanup_label.setText(
            f"AI-polished transcript written: {Path(path).name} — "
            "clearly marked as an AI rewrite; the verbatim "
            "transcript is the record.")
        self.status_label.setText(
            f"Polished transcript saved in {self.session_dir}")
        self._update_buttons()

    def _on_polish_error(self, message: str) -> None:
        self._polishing = False
        self.cleanup_label.setText(f"Polish failed: {message}")
        if not self._polish_quiet:
            QMessageBox.critical(self, "Transcript polish failed", message)
        else:
            self.status_label.setText(
                "Auto-polish failed — the rule-cleaned transcript is "
                "still there.")
        self._update_buttons()

    # -------------------------------------------------- speaker learning ---
    def _speakers_db_path(self) -> Path:
        """speakers.db lives in the recordings root (local only)."""
        return Path(self.config.output_root) / SPEAKERS_DB_FILE

    # ---------------------------------------------- person pages (Nov 21) --
    def _show_people_list(self) -> None:
        """Back to the People list layer (from a person page)."""
        self._person_name = None
        self._person_sample = None
        self.people_stack.setCurrentIndex(0)
        self._update_buttons()

    def _on_open_person_page(self) -> None:
        """Open the person page for the selected known speaker."""
        name = self._selected_known_speaker()
        if name is not None:
            self._show_person_page(name)

    def _show_person_page(self, name: str) -> None:
        """Load a speaker's detail into the person page and show it."""
        self._person_name = name
        self._refresh_person_view()
        self.people_stack.setCurrentIndex(1)
        self._update_buttons()

    def _refresh_person_view(self) -> None:
        """Fill the person page from get_speaker_detail (core, U2).

        No commitments section exists anywhere in the product —
        there is no commitments data, so nothing is shown for it
        (no placeholder either). A speaker with no sessions and
        no utterances still gets a page: the lists say so plainly.
        """
        name = self._person_name
        if name is None:
            return
        try:
            # The detail query takes a SpeakerDatabase (or None
            # to skip the database lookup); the database is
            # opened for this one query and closed again. A
            # missing speakers.db is fine — SpeakerDatabase
            # creates an empty one, and the session scan is the
            # fallback for the voice sample either way.
            db = SpeakerDatabase(self._speakers_db_path())
            try:
                detail = get_speaker_detail(
                    self.config.output_root, db, name)
            finally:
                db.close()
        except Exception:  # noqa: BLE001 — a failed load is shown
            logger.exception("Person page load failed")
            self.person_title_label.setText(name)
            self.person_summary_label.setText(
                "Could not load this person's meetings.")
            self.person_sessions_list.clear()
            self.person_utterances_list.clear()
            return
        sample = detail.get("sample")
        self._person_sample = sample
        self.person_title_label.setText(detail["name"])
        known = "Known speaker" if detail["known_in_database"] else (
            "Speaker in transcripts — not yet enrolled as a known "
            "speaker")
        self.person_summary_label.setText(
            f"{known}. {detail['session_count']} meeting(s), "
            f"{detail['utterance_count']} utterance(s).")
        if sample is not None and detail["sample_available"]:
            self.person_sample_btn.setEnabled(True)
            self.person_sample_label.setText(
                "Voice sample available — play it to hear how this "
                "speaker sounds.")
        else:
            self.person_sample_btn.setEnabled(False)
            self.person_sample_label.setText(
                "No voice sample stored for this speaker yet.")
        sessions_list = self.person_sessions_list
        sessions_list.clear()
        sessions = detail["sessions"]
        if not sessions:
            item = QListWidgetItem("No meetings yet.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            sessions_list.addItem(item)
        for session in sessions:
            started = session.get("start_time") or ""
            label = (f"{session['session_id']} — "
                     f"{session['utterance_count']} utterance(s)")
            if started:
                label = f"{started}  {label}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, session["path"])
            sessions_list.addItem(item)
        utterances_list = self.person_utterances_list
        utterances_list.clear()
        utterances = detail["utterances"]
        if not utterances:
            item = QListWidgetItem("No utterances yet.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            utterances_list.addItem(item)
        for utterance in utterances:
            label = (
                f"[{_format_mmss(utterance['start'])}] "
                f"{utterance['text']}")
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, (
                utterance["session_id"], utterance["start"],
                utterance["end"]))
            utterances_list.addItem(item)

    def _on_person_session_clicked(self, item) -> None:
        """A meeting on a person page loads that session."""
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        session_dir = Path(path)
        if session_dir.is_dir():
            self.set_view("Session")
            self.load_session(session_dir)

    def _on_person_utterance_clicked(self, item) -> None:
        """An utterance plays its moment, via _play_audio only."""
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        session_id, start, end = data
        session_dir = self._session_dir_for_id(session_id)
        if session_dir is None:
            return
        audio = self._session_audio_path(session_dir)
        if audio is None:
            return
        self.set_view("Session")
        self.load_session(session_dir)
        self._play_audio(audio, start, end)

    def _on_person_play_sample(self) -> None:
        """Play the speaker's enrolled voice sample (if any)."""
        sample = self._person_sample
        if not sample:
            return
        audio_path = sample.get("audio_path")
        if not audio_path:
            return
        self._play_audio(Path(audio_path), sample.get("start", 0.0),
                         sample.get("end", 0.0))

    def _on_person_rename(self) -> None:
        """Rename from a person page: the existing rename path."""
        name = self._person_name
        if name is None:
            return
        self._rename_known_speaker_named(name)

    def _rename_known_speaker_named(self, name: str) -> None:
        """Rename ``name`` via the rename panel (shared by People).

        :meth:`on_rename_known_speaker` fills the rename panel
        from the current list selection; this variant does the
        same work for an explicit name (the person page), then
        refreshes the page under the new name when the rename
        lands.
        """
        found = False
        for index in range(self.known_speakers_list.count()):
            if self.known_speakers_list.item(index).text() == name:
                self.known_speakers_list.setCurrentRow(index)
                found = True
                break
        if not found:
            return
        # Fills the rename panel and switches to the Session
        # view, exactly as renaming from the list does — the
        # rename itself still goes through the Rename speaker
        # button there.
        self.on_rename_known_speaker()

    def _session_dir_for_id(self, session_id: str) -> Path | None:
        """Session folder for a ``YYYY-MM-DD_HH-MM-SS`` id, if any."""
        root = Path(self.config.output_root)
        candidate = root / session_id
        return candidate if candidate.is_dir() else None

    def _session_audio_path(self, session_dir: Path) -> Path | None:
        """Preferred playback file for a session, or None."""
        for name in (COMBINED_FILE, MIC_FILE, SYSTEM_FILE):
            candidate = session_dir / name
            if candidate.is_file():
                return candidate
        return None

    # -------------------------------------------------- bookmarks (Nov 21) --
    def _record_elapsed_seconds(self) -> float:
        """Seconds since Record, minus paused time (never negative)."""
        if self._record_started_at is None:
            return 0.0
        elapsed = time.time() - self._record_started_at
        elapsed -= self._elapsed_paused_total
        if self._pause_started_at is not None:
            elapsed -= time.time() - self._pause_started_at
        return max(0.0, elapsed)

    def _add_bookmark_at(self, session_dir: Path, at: float,
                         label: str = "") -> None:
        """Persist a bookmark, refresh the list, say it landed."""
        try:
            bookmark = add_bookmark(session_dir, at, label=label)
        except Exception as exc:  # noqa: BLE001 — reported, not fatal
            logger.exception("Bookmark failed")
            QMessageBox.warning(self, "Bookmark failed", str(exc))
            return
        if session_dir == self.session_dir:
            self._refresh_bookmarks_list()
        self.statusBar().showMessage(
            f"Bookmarked {_format_mmss(bookmark['t'])}"
            + (f" — {label}" if label else ""))

    def _on_bookmark_button(self) -> None:
        """Record-view Bookmark: mark the current elapsed time.

        No label prompt while recording — stopping to type would
        lose the moment. The bookmark is labelled with the time
        only (an empty label); labels can be added later from the
        Session view's list by re-adding at the same moment.
        """
        if self.session_dir is None:
            return
        self._add_bookmark_at(
            self.session_dir, self._record_elapsed_seconds())

    def _on_add_bookmark_button(self) -> None:
        """Session-view add: the moment now playing / selected / 0.

        The honest choice (documented in docs/OVERVIEW.md): the
        player position when playing, else the selected segment's
        start, else the start of the session. A label prompt is
        offered here — unlike during recording, nothing is lost
        by typing one.
        """
        if self.session_dir is None:
            return
        at = 0.0
        if (self._player is not None
                and self._player.playbackState()
                == QMediaPlayer.PlaybackState.PlayingState):
            at = self._player.position() / 1000.0
        else:
            current = self.segments_list.currentItem()
            if current is not None:
                timing = current.data(Qt.ItemDataRole.UserRole)
                if timing:
                    at = float(timing[0])
        label, ok = QInputDialog.getText(
            self, "Add bookmark",
            f"Label for the bookmark at {_format_mmss(at)} "
            "(optional):")
        if not ok:
            return
        self._add_bookmark_at(self.session_dir, at, label=label)

    def _refresh_bookmarks_list(self) -> None:
        """Bookmarks for the loaded session, earliest first.

        A bookmark whose audio was deleted by retention renders
        disabled with the reason in plain words — it can never
        be played. Clicking an available one seeks to it.
        """
        self.bookmarks_list.clear()
        if self.session_dir is None:
            return
        try:
            bookmarks = list_bookmarks(self.session_dir)
        except Exception:  # noqa: BLE001 — an empty list is shown
            logger.exception("Bookmark list failed")
            return
        if not bookmarks:
            item = QListWidgetItem(
                "No bookmarks yet — press Bookmark while recording, "
                "or add one at the current moment.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.bookmarks_list.addItem(item)
            return
        for bookmark in bookmarks:
            label = bookmark.label or "Bookmark"
            text = f"{_format_mmss(bookmark.t)} — {label}"
            if not bookmark.audio_available:
                text += " (audio deleted)"
            item = QListWidgetItem(text)
            if bookmark.audio_available:
                item.setData(Qt.ItemDataRole.UserRole, bookmark.t)
            else:
                item.setFlags(
                    item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip(
                    "The audio for this moment was deleted by the "
                    "retention rule, so it cannot be played.")
            self.bookmarks_list.addItem(item)

    def _on_bookmark_item_clicked(self, item) -> None:
        """Click-to-seek: play from the bookmark, highlight it."""
        at = item.data(Qt.ItemDataRole.UserRole)
        if at is None or self.session_dir is None:
            return
        audio = self._session_audio_path(self.session_dir)
        if audio is None:
            return
        self._play_audio(audio, float(at), None)
        self._highlight_segment_at(float(at))

    # ------------------------------------------ enterprise cleanup (Nov 21) --
    def on_enterprise_cleanup(self) -> None:
        """Enterprise-clean the loaded session's transcript.

        Refuses unless the opt-in is enabled in Settings, with a
        plain message saying where to enable it — the config
        enforces the same opt-in in core, but the button must
        not look like it ran and silently did nothing. Runs in
        a worker thread with progress + Cancel; the verbatim
        transcript is never touched, and the result appears as
        the explicitly labelled "Enterprise-cleaned" variant.
        """
        if self.session_dir is None:
            return
        if not self.config.enterprise_cleanup_enabled:
            QMessageBox.information(
                self, "Enterprise cleanup is off",
                "Enterprise cleanup is turned off. It sends the "
                "transcript TEXT to the Claude Code CLI on this "
                "machine (audio never leaves the machine).\n\n"
                "To enable it, open Settings and turn on "
                "\"Enterprise cleanup\" in the \"Pre-roll & "
                "enterprise cleanup\" section.")
            return
        if self._enterprise_running:
            return
        self._start_enterprise_cleanup()

    def _start_enterprise_cleanup(self) -> None:
        self._enterprise_running = True
        self._enterprise_cancel_event = threading.Event()
        self._enterprise_thread = QThread(self)
        self._enterprise_worker = _EnterpriseCleanupWorker(
            self.session_dir, self.config,
            is_cancelled=self._enterprise_cancel_event.is_set)
        self._enterprise_worker.moveToThread(self._enterprise_thread)
        self._enterprise_thread.started.connect(
            self._enterprise_worker.run)
        self._enterprise_worker.progress.connect(
            self._on_enterprise_progress)
        self._enterprise_worker.finished.connect(
            self._on_enterprise_finished)
        self._enterprise_worker.error.connect(
            self._on_enterprise_error)
        self._enterprise_worker.finished.connect(
            self._enterprise_thread.quit)
        self._enterprise_worker.error.connect(
            self._enterprise_thread.quit)
        self._enterprise_worker.finished.connect(
            self._enterprise_worker.deleteLater)
        self._enterprise_worker.error.connect(
            self._enterprise_worker.deleteLater)
        self._enterprise_thread.finished.connect(
            self._enterprise_thread.deleteLater)
        self._update_buttons()
        self._enterprise_thread.start()
        self.statusBar().showMessage(
            "Enterprise cleanup running — transcript text only…")

    def _on_enterprise_progress(self, _fraction: float, text: str) -> None:
        self.statusBar().showMessage(text)

    def _on_enterprise_finished(self, path: str) -> None:
        self._enterprise_running = False
        self._enterprise_worker = None
        self._enterprise_thread = None
        self._enterprise_cancel_event = None
        self.statusBar().showMessage(
            f"Enterprise-cleaned transcript saved: {Path(path).name}")
        self._refresh_segments_list()
        self._update_buttons()

    def _on_enterprise_error(self, message: str) -> None:
        self._enterprise_running = False
        self._enterprise_worker = None
        self._enterprise_thread = None
        self._enterprise_cancel_event = None
        self.statusBar().clearMessage()
        if "claude" in message.lower() or "not available" in message.lower():
            QMessageBox.warning(
                self, "Claude Code CLI not available",
                "Enterprise cleanup needs the Claude Code CLI "
                "(`claude -p`) on this machine, and it could not "
                "be run.\n\n"
                f"Detail: {message}\n\n"
                "Install it from https://claude.com/product/"
                "claude-code (or your package manager), make sure "
                "`claude` is on PATH, then try again. Nothing was "
                "changed — the verbatim transcript is untouched.")
        else:
            QMessageBox.warning(
                self, "Enterprise cleanup failed",
                f"{message}\n\nNothing was changed — the verbatim "
                "transcript is untouched.")
        self._update_buttons()

    def _on_enterprise_cancel(self) -> None:
        """Cancel: kill the CLI run; no artifact is written."""
        if self._enterprise_worker is not None:
            self._enterprise_worker.cancel()
        if self._enterprise_cancel_event is not None:
            self._enterprise_cancel_event.set()
        self.statusBar().showMessage("Enterprise cleanup cancelled.")

    # -------------------------------------------------- waveform (Nov 21) --
    def _refresh_waveform(self) -> None:
        """(Re)compute or reuse the loaded session's waveform.

        RMS buckets come from a worker and are cached in memory
        per session (``_waveform_cache``) — switching away and
        back never re-decodes. Speaker colours come from the
        transcript's segments; bookmarks draw as red markers.
        No audio is decoded on the UI thread.
        """
        widget = self.waveform_widget
        if self.session_dir is None:
            widget.clear()
            widget.set_segments([])
            widget.set_bookmarks([])
            return
        key = str(self.session_dir)
        audio = self._session_audio_path(self.session_dir)
        if audio is None:
            widget.clear()
            widget.set_segments([])
            widget.set_bookmarks([])
            return
        if key in self._waveform_cache:
            buckets, duration = self._waveform_cache[key]
            self._paint_waveform(buckets, duration)
            return
        widget.clear()
        self._apply_waveform_segments_and_bookmarks()
        self._waveform_thread = QThread(self)
        self._waveform_threads.add(self._waveform_thread)
        self._waveform_thread.finished.connect(
            lambda _t=self._waveform_thread:
            self._waveform_threads.discard(_t))
        self._waveform_worker = _WaveformWorker(audio, key=key)
        self._waveform_worker.moveToThread(self._waveform_thread)
        self._waveform_thread.started.connect(self._waveform_worker.run)
        # The worker is captured in the slot: when several
        # sessions are loaded in quick succession, each result
        # is attributed to its own worker, not to whichever
        # worker happens to be current when it lands.
        worker = self._waveform_worker
        self._waveform_worker.finished.connect(
            lambda buckets, duration, _w=worker:
            self._on_waveform_loaded(_w, buckets, duration))
        self._waveform_worker.error.connect(self._on_waveform_error)
        self._waveform_worker.finished.connect(
            self._waveform_thread.quit)
        self._waveform_worker.error.connect(self._waveform_thread.quit)
        self._waveform_worker.finished.connect(
            self._waveform_worker.deleteLater)
        self._waveform_worker.error.connect(
            self._waveform_worker.deleteLater)
        self._waveform_thread.finished.connect(
            self._waveform_thread.deleteLater)
        self._waveform_thread.start()

    def _apply_waveform_segments_and_bookmarks(self) -> None:
        """Speaker strip + bookmark markers for the loaded session.

        Speaker colours come from the verbatim transcript's
        segments (the segments list itself carries timings only,
        no speaker labels). A missing or unreadable transcript
        simply leaves the strip empty.
        """
        segments = []
        if self.session_dir is not None:
            try:
                import json as _json

                payload = _json.loads(
                    (self.session_dir / TRANSCRIPT_JSON).read_text(
                        encoding="utf-8"))
                names = load_speaker_names(self.session_dir)
                for segment in payload.get("segments", []):
                    speaker = segment.get("speaker")
                    display = (names.get(speaker, speaker)
                               if speaker else "Speaker")
                    segments.append((
                        float(segment.get("start", 0.0)),
                        float(segment.get("end", 0.0)),
                        str(display)))
            except Exception:  # noqa: BLE001 — strip is optional
                logger.exception("Waveform speaker strip failed")
        self.waveform_widget.set_segments(segments)
        marks = []
        if self.session_dir is not None:
            try:
                marks = [b.t for b in list_bookmarks(self.session_dir)
                         if b.audio_available]
            except Exception:  # noqa: BLE001 — markers are optional
                logger.exception("Waveform bookmark markers failed")
        self.waveform_widget.set_bookmarks(marks)

    def _on_waveform_loaded(self, worker, buckets: list,
                            duration: float) -> None:
        # Only the current worker clears the current refs: a
        # stale worker finishing after a newer one started must
        # not clear the newer one's thread.
        if worker is self._waveform_worker:
            self._waveform_worker = None
            self._waveform_thread = None
        # Stale result (the user switched sessions while this
        # worker ran): cache it under its own session, but never
        # paint it — the current session has its own worker.
        key = getattr(worker, "key", "")
        if key:
            self._waveform_cache[key] = (buckets, duration)
        if (self.session_dir is None
                or (key and str(self.session_dir) != key)):
            return
        self._paint_waveform(buckets, duration)

    def _paint_waveform(self, buckets: list, duration: float) -> None:
        """Paint cached/computed buckets + the speaker strip."""
        self.waveform_widget.set_data(buckets, duration)
        self._apply_waveform_segments_and_bookmarks()

    def _on_waveform_error(self, _message: str) -> None:
        self._waveform_worker = None
        self._waveform_thread = None
        self.waveform_widget.clear()

    def _on_waveform_clicked(self, at: float) -> None:
        """Waveform click: play from there, highlight the segment."""
        if self.session_dir is None:
            return
        audio = self._session_audio_path(self.session_dir)
        if audio is None:
            return
        self._play_audio(audio, at, None)
        self._highlight_segment_at(at)

    # ----------------------------------------------- follow-along (Nov 21) --
    def _on_playback_state_changed(self, state) -> None:
        """Start/stop the follow-along poll with playback."""
        playing = (state == QMediaPlayer.PlaybackState.PlayingState)
        if playing:
            self._follow_timer.start()
        else:
            self._follow_timer.stop()
            self._follow_segment_index = None

    def _on_playback_position(self, position_ms: int) -> None:
        """Highlight the segment containing the playhead (seconds)."""
        self._highlight_playing_segment(position_ms / 1000.0)

    def _poll_follow_along(self) -> None:
        """Fallback poll for players that rarely emit positionChanged."""
        if self._player is None:
            return
        if (self._player.playbackState()
                != QMediaPlayer.PlaybackState.PlayingState):
            return
        self._highlight_playing_segment(self._player.position() / 1000.0)

    def _highlight_playing_segment(self, at: float) -> None:
        """Select + scroll to the segment at ``at`` — playing only.

        Only while playback is active: a stopped player must
        never move the user's selection. Re-selecting the same
        row is skipped so the list does not flicker.
        """
        if self._player is None:
            return
        if (self._player.playbackState()
                != QMediaPlayer.PlaybackState.PlayingState):
            return
        index = self._segment_index_at(at)
        if index is None or index == self._follow_segment_index:
            return
        self._follow_segment_index = index
        self.segments_list.setCurrentRow(index)
        self.segments_list.scrollToItem(
            self.segments_list.item(index))

    def _highlight_segment_at(self, at: float) -> None:
        """Highlight the segment at ``at`` (bookmark / waveform click).

        Unlike follow-along this is a direct user action, so it
        applies even when playback has not started yet.
        """
        index = self._segment_index_at(at)
        if index is None:
            return
        self.segments_list.setCurrentRow(index)
        self.segments_list.scrollToItem(
            self.segments_list.item(index))

    def _segment_index_at(self, at: float) -> int | None:
        """Row of the segment containing ``at``, or None.

        A time exactly on a boundary belongs to the segment
        starting there; a time past the last segment's end (a
        bookmark at the very end of the session) highlights the
        last segment that started at or before it.
        """
        fallback = None
        for index in range(self.segments_list.count()):
            timing = self.segments_list.item(index).data(
                Qt.ItemDataRole.UserRole)
            if not timing:
                continue
            start, end = float(timing[0]), float(timing[1])
            if start <= at < end:
                return index
            if start <= at:
                fallback = index
        return fallback

    # -------------------------------------------- calendar filter (Nov 21) --
    def _on_calendar_speaker_changed(self, _index: int) -> None:
        self._apply_calendar_speaker_filter()

    def _refresh_calendar_speaker_options(self) -> None:
        """Speaker combo for the Calendar filter (same speakers as
        the Library filter — one query function, one list)."""
        combo = getattr(self, "calendar_speaker_combo", None)
        if combo is None:
            return
        current = combo.currentData()
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem("All speakers", None)
            for name in self._library_speaker_names():
                combo.addItem(name, name)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)

    def _apply_calendar_speaker_filter(self) -> None:
        """Dim (never remove) days that hold no matching session.

        The choice, documented in docs/OVERVIEW.md: non-matching
        badges get 35% opacity and are disabled, so the calendar
        keeps its shape and the filter is visibly a filter, not
        missing data. Matching badges stay fully opaque and
        clickable. Days with no badge at all are untouched.
        """
        combo = getattr(self, "calendar_speaker_combo", None)
        widget = getattr(self, "calendar_widget", None)
        if combo is None or widget is None:
            return
        speaker = combo.currentData()
        badges = getattr(widget, "_badges", [])
        for badge in badges:
            speakers = getattr(badge.entry.summary, "speakers", None) or []
            match = (speaker is None or any(
                str(s).lower() == str(speaker).lower()
                for s in speakers))
            effect = badge.graphicsEffect()
            if effect is None:
                effect = QGraphicsOpacityEffect(badge)
                badge.setGraphicsEffect(effect)
            effect.setOpacity(1.0 if match else 0.35)
            badge.setEnabled(match)

    # ---------------------------------------------- library stars (Nov 21) --
    def _library_speaker_names(self) -> list[str]:
        """Speakers with at least one session, alphabetical."""
        names: set[str] = set()
        try:
            for summary in list_session_summaries(self.config.output_root):
                names.update(summary.speakers)
        except Exception:  # noqa: BLE001 — an empty list is shown
            logger.exception("Library speaker list failed")
        return sorted(names, key=str.lower)

    def _refresh_library_speaker_options(self) -> None:
        combo = getattr(self, "library_speaker_combo", None)
        if combo is None:
            return
        current = combo.currentData()
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem("All speakers", None)
            for name in self._library_speaker_names():
                combo.addItem(name, name)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)

    def _on_library_speaker_changed(self, _index: int) -> None:
        self._refresh_library()

    def _selected_library_speaker(self) -> str | None:
        combo = getattr(self, "library_speaker_combo", None)
        if combo is None:
            return None
        return combo.currentData()

    def _update_library_star_button(self) -> None:
        """Star / Unstar label for the selected Library row."""
        button = getattr(self, "library_star_btn", None)
        if button is None:
            return
        summary = self._selected_library_summary()
        if summary is None:
            button.setText("☆ Star")
            button.setEnabled(False)
            return
        button.setEnabled(True)
        button.setText("★ Unstar" if summary.pinned else "☆ Star")

    def _on_library_star_toggle(self) -> None:
        """Pin / unpin the selected Library session, then refresh."""
        summary = self._selected_library_summary()
        if summary is None:
            return
        session_dir = Path(summary.path)
        try:
            update_metadata(session_dir, pinned=not summary.pinned)
        except Exception as exc:  # noqa: BLE001 — reported, not fatal
            logger.exception("Pin toggle failed")
            QMessageBox.warning(self, "Could not change the pin",
                                str(exc))
            return
        self._refresh_library()
        self.statusBar().showMessage(
            ("Pinned " if not summary.pinned else "Unpinned ")
            + session_dir.name)

    def _refresh_known_speakers(self) -> None:
        """Fill the known-speakers list from speakers.db (if any)."""
        self.known_speakers_list.clear()
        db_path = self._speakers_db_path()
        if not db_path.exists():
            self.known_speakers_list.addItem(
                "(no speakers learned yet — diarize a call, rename a "
                "speaker, then Confirm a suggestion)")
            return
        try:
            db = SpeakerDatabase(db_path)
            try:
                speakers = db.list_speakers()
            finally:
                db.close()
        except Exception:  # noqa: BLE001 — a bad DB is not fatal
            return
        for speaker in speakers:
            item = QListWidgetItem(
                f"{speaker['name']} — {speaker['sample_count']} sample(s), "
                f"{speaker['backend']}")
            # The name rides in UserRole so Rename / Play sample
            # never have to parse it back out of the display text.
            item.setData(Qt.ItemDataRole.UserRole, speaker["name"])
            self.known_speakers_list.addItem(item)

    def _selected_known_speaker(self) -> str | None:
        """Name of the selected known speaker, or None.

        The placeholder row ("(no speakers learned yet …)") carries
        no UserRole data, so it never counts as a selection.
        """
        item = self.known_speakers_list.currentItem()
        if item is None:
            return None
        name = item.data(Qt.ItemDataRole.UserRole)
        return str(name) if name else None

    def on_play_known_sample(self) -> None:
        """Play a voice sample of the selected known speaker."""
        name = self._selected_known_speaker()
        if name is None:
            return
        db = SpeakerDatabase(self._speakers_db_path())
        try:
            found = find_sample_for_speaker(
                self.config.output_root, db, name)
        finally:
            db.close()
        if found is None:
            self.status_label.setText(
                f"No voice sample for {name} — the recording it was "
                "learned from is gone (audio older than 24 h is "
                "deleted, or the session folder is missing).")
            return
        audio, start, end, session_name = found
        if self._play_audio(audio, start, end):
            self.status_label.setText(
                f"Playing a {end - start:.0f}s sample of {name} "
                f"from {session_name}…")

    def on_play_suggestion_sample(self) -> None:
        """Play a voice sample of the selected suggestion.

        The sample is that speaker's longest turn in this session,
        so it is one voice — hear it before confirming the name.
        Works for stored suggestions too (the row's label is all
        it needs — see _selected_suggestion_label).
        """
        label = self._selected_suggestion_label()
        if label is None or not self.session_dir:
            return
        found = sample_window_for_label(self.session_dir, label)
        if found is None:
            self.status_label.setText(
                f"No voice sample for {label} in this "
                "session — no audio for that speaker.")
            return
        audio, start, end = found
        if self._play_audio(audio, start, end):
            self.status_label.setText(
                f"Playing a {end - start:.0f}s sample of "
                f"{label}…")

    def on_rename_known_speaker(self) -> None:
        """Start renaming the selected known speaker.

        Fills in the rename panel on the Session view with the
        speaker's name and switches to it, putting the cursor in
        the name field — the rename itself still goes through the
        Rename speaker button, so the identity path (database +
        every past session) is unchanged. Double-clicking a known
        speaker does the same thing.
        """
        name = self._selected_known_speaker()
        if name is None:
            return
        if not self.session_dir:
            self.status_label.setText(
                "Rename needs a session — record or process one first; "
                "the rename panel on the Session view works on the "
                "loaded session.")
            return
        if self.rename_label_combo.findText(name) < 0:
            self.rename_label_combo.addItem(name)
        self.rename_label_combo.setCurrentText(name)
        self.rename_name_edit.setPlaceholderText(f"New name for {name}")
        self.rename_name_edit.clear()
        self.show_view("Session")
        self.rename_name_edit.setFocus()
        self.status_label.setText(
            f"Renaming {name} — type the new name below and press "
            "“Rename speaker”. A learned speaker is renamed in every "
            "past session, not just this one.")
        self._update_buttons()

    def on_suggest(self) -> None:
        """Suggest known names for the loaded diarized session (manual)."""
        self._start_suggest(auto_identify=False)

    def _start_suggest(self, *, auto_identify: bool = False) -> None:
        """Run the suggest worker; ``auto_identify`` is the auto-chain.

        With auto_identify the worker also applies confident
        matches and enrols clean unknown clusters as unnamed
        identities (see app.speakers.session.auto_identify_session).
        The manual button never auto-applies — Confirm/Correct
        stays the user's job there.
        """
        if not self.session_dir or self._suggesting:
            return
        self._suggesting = True
        self.suggestions_list.clear()
        self._update_buttons()
        self.status_label.setText("Matching speakers against known voices…")
        self._suggest_thread = QThread(self)
        self._suggest_worker = _SuggestWorker(
            self.session_dir, self._speakers_db_path(),
            embedding_backend=self.config.embedding_backend,
            auto_identify=auto_identify)
        self._suggest_worker.moveToThread(self._suggest_thread)
        self._suggest_thread.started.connect(self._suggest_worker.run)
        self._suggest_worker.finished.connect(self._on_suggest_finished)
        self._suggest_worker.error.connect(self._on_suggest_error)
        self._suggest_worker.finished.connect(self._suggest_thread.quit)
        self._suggest_worker.error.connect(self._suggest_thread.quit)
        self._suggest_thread.start()

    def _on_suggest_finished(self, summary: dict) -> None:
        self._suggesting = False
        # Defensive: a worker that finished without setting
        # .suggestions (e.g. a future refactor) must not crash this
        # slot — in PyQt6 an exception here aborts the process.
        self._suggestion_objects = (
            getattr(self._suggest_worker, "suggestions", None)
            or [])
        self._suggestion_dicts = list(summary.get("suggestions", []))
        self._render_suggestion_dicts(self._suggestion_dicts)
        if not summary.get("suggestions"):
            self.suggestions_list.addItem(self._placeholder_item(
                "No suggestions — no known speakers matched. Confirm "
                "a speaker to teach OffAir their voice."))
        self._refresh_known_speakers()
        # Phase 5 (Phase 4 follow-up): say out loud that Remote is
        # not suggested in simple mode, instead of silently returning
        # only You.
        remote_note = ""
        if self.session_dir is not None:
            try:
                import json as _json
                diar_path = self.session_dir / "diarization.json"
                if diar_path.exists():
                    method = _json.loads(
                        diar_path.read_text(encoding="utf-8")).get("method", "")
                    if "dual-channel" in str(method):
                        remote_note = (" Note: Remote is not suggested — "
                                       "simple mode mixes all remote "
                                       "speakers into one cluster.")
            except Exception:  # noqa: BLE001 — the note is best-effort
                pass
        self.status_label.setText(
            f"{len(summary.get('suggestions', []))} suggestion(s) — "
            "select one to Confirm or Correct it. speaker_suggestions.json "
            f"written in the session folder.{remote_note}")
        self._update_buttons()
        # Auto-process chain ends here: summarise what the
        # auto-identification did (the recording was saved before
        # the chain started, and each step reported as it went).
        if self._auto_processing:
            self._auto_processing = False
            auto = summary.get("auto_identify") or {}
            bits: list[str] = []
            applied = auto.get("applied", [])
            enrolled = auto.get("enrolled", [])
            left = auto.get("left_unknown", [])
            if applied:
                bits.append(f"{len(applied)} matched: " + ", ".join(
                    f"{a['label']}→{a['name']}" for a in applied))
            if enrolled:
                bits.append(f"{len(enrolled)} new identity(ies): " + ", ".join(
                    f"{a['label']}→{a['name']}" for a in enrolled))
            if left:
                bits.append(f"{len(left)} left unknown: " + ", ".join(
                    f"{a['label']} ({a['reason']})" for a in left))
            detail = "; ".join(bits) if bits else "no speakers to identify"
            self.status_label.setText(
                f"Auto-processing done — {detail}. The transcript, "
                "speaker names and speaker_suggestions.json are in the "
                "session folder.")
            self._refresh_segments_list()
            self._refresh_known_speakers()
            # Final re-clean: auto-identification renamed labels,
            # so the cleaned artifact picks up the final names.
            self._run_cleanup()
            # Auto-polish (Nov 10 2026): only when the setting is
            # on — OFF by default. Quiet: a missing Ollama is a
            # status note, not a modal; the cleaned version stays.
            if self.config.transcript_polish_enabled:
                self._start_polish(quiet=True)
            # The session now has a transcript, so its audio is
            # eligible for retention pruning on the next run.
            self._run_retention_prune()
        # Suggestions were (re)written: the chooser row's status
        # ("Speakers identified") and the loaded-session line follow.
        self._refresh_session_chooser()
        self._update_loaded_session_label()

    def _on_suggest_error(self, message: str) -> None:
        # Auto-process chain stops here, loudly (the dialog below);
        # the recording was saved before the chain started.
        self._auto_processing = False
        self._suggesting = False
        QMessageBox.critical(self, "Speaker suggestion failed", message)
        self._update_buttons()

    def _selected_suggestion(self):
        row = self.suggestions_list.currentRow()
        if row < 0 or row >= len(self._suggestion_objects):
            return None
        return self._suggestion_objects[row]

    def on_confirm_suggestion(self) -> None:
        """Confirm the selected suggestion: learn it + rename here."""
        suggestion = self._selected_suggestion()
        if suggestion is None or not self.session_dir:
            return
        try:
            db = SpeakerDatabase(self._speakers_db_path())
            try:
                # Same configured backend as the suggestion, so the
                # confirmed sample lands under that backend.
                confirm_session_suggestion(
                    self.session_dir, db, suggestion,
                    extractor=get_extractor(self.config.embedding_backend))
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001 — surfaced to the user
            QMessageBox.critical(self, "Confirm failed", str(exc))
            return
        self.status_label.setText(
            f"Confirmed {suggestion.label} as "
            f"{suggestion.suggested_name} — learned for next time, "
            "transcript.txt/.srt updated.")
        self._refresh_segments_list()
        self._refresh_known_speakers()
        self._update_buttons()

    def on_correct_suggestion(self) -> None:
        """Correct the selected suggestion to the name in the name field."""
        suggestion = self._selected_suggestion()
        if suggestion is None or not self.session_dir:
            return
        correct_name = self.rename_name_edit.text().strip()
        if not correct_name:
            QMessageBox.warning(
                self, "No correction",
                "Type the correct name in the name field first.")
            return
        try:
            db = SpeakerDatabase(self._speakers_db_path())
            try:
                correct_session_suggestion(
                    self.session_dir, db, suggestion, correct_name,
                    extractor=get_extractor(self.config.embedding_backend))
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001 — surfaced to the user
            QMessageBox.critical(self, "Correction failed", str(exc))
            return
        self.rename_name_edit.clear()
        # A corrected name is a real name Whisper should hear next
        # time — add it to the names dictionary.
        try:
            add_name(correct_name)
        except Exception:  # noqa: BLE001 — never break a correction
            logger.warning("Could not add %r to the names dictionary",
                           correct_name, exc_info=True)
        self.status_label.setText(
            f"Corrected {suggestion.label} to {correct_name} — learned "
            "for next time, transcript.txt/.srt updated.")
        self._refresh_segments_list()
        self._refresh_known_speakers()
        self._update_buttons()

    # --------------------------------------- live transcription (Nov 19) ---
    def _configured_model(self) -> str:
        """The Whisper model the live path and the manual path share.

        The session combo is an ephemeral per-session override; its
        fallback is the config's EFFECTIVE model (Nov 20 2026,
        Item 8) — the current build's default unless the user
        explicitly picked a model in Settings — never the raw
        stored value, which may be a stale pre-upgrade default.
        """
        return (self.model_combo.currentData()
                or self.config.effective_whisper_model)

    def _nearlive_hint_text(self) -> str:
        """The hint under the near-live pane, naming the model in use."""
        return _nearlive_hint_text(self._configured_model())

    def _refresh_live_model_label(self) -> None:
        """Show the Whisper model actually in use (Nov 20 2026, Item 7).

        The Record view's "Transcribing with Whisper base" line and
        the live-list tooltip both name the model the live path and
        the manual path share — never a hard-coded one. While a
        recording runs, the live chunker's model is the truth (it
        was built when recording started); otherwise the configured
        model is. Called at startup, when the model combo changes,
        when Settings are saved, when live transcription starts,
        and when a session loads.
        """
        chunker = getattr(self, "_live_chunker", None)
        model = (chunker.model if chunker is not None
                 else self._configured_model())
        # Diarization mode (Nov 21 2026): the live status line
        # names the model AND the diarization mode, and says
        # "provisional" — live speaker labels are provisional
        # until finalize confirms them, and no confidence number
        # is ever shown (the engine reports none).
        mode_key = self.config.diarization_mode
        mode_short = _DIARIZATION_SHORT_LABELS.get(
            mode_key, mode_key)
        self.live_model_label.setText(
            f"Transcribing with Whisper {model} · {mode_short} "
            "diarization · provisional")
        self.nearlive_list.setToolTip(
            f"Near-live transcript: 25 s chunks, Whisper {model}, "
            f"{mode_short} diarization, provisional speaker labels.")

    def _start_live_transcription(self) -> None:
        """Start the live poller for the recording just started.

        A fresh chunker on the *configured* model (base by default),
        so live chunks and the final tail agree — there is no
        longer a separate near-live default; every path uses the
        configured model. The 5 s QTimer
        fires only while recording (stopped in on_stop, paused in
        on_pause). Before the first chunk, an uncached model shows
        the download status: faster-whisper gives no byte-level
        download-progress hook, so there is no progress bar — only
        the status line with elapsed time while the first chunk
        waits for the download on the worker thread.
        """
        if not self.session_dir:
            return
        self._live_chunker = ChunkedTranscriber(
            model=self._configured_model(), engine=self._live_engine)
        self._live_busy = False
        self._live_failed = False
        self._live_abandoned = False
        self._live_list_started = False
        self._stop_finalize_pending = False
        self._live_download_pending = False
        self._live_download_started_at = None
        model = self._live_chunker.model
        try:
            cached = whisper_model_cached(model)
        except Exception:  # noqa: BLE001 — a cache check never fails a run
            cached = False
        if not cached:
            self._live_download_pending = True
            self._live_download_started_at = time.monotonic()
            size = WHISPER_MODELS.get(model, {}).get("size_mb")
            size_text = f" (~{size} MB)" if size else ""
            text = (f"Downloading Whisper {model}{size_text} — "
                    "first run only · 0s elapsed")
            self.nearlive_label.setText(text)
            self.statusBar().showMessage(text)
        self._refresh_live_model_label()
        self._live_timer.start()

    def _on_live_poll(self) -> None:
        """One live-transcription tick (every 5 s while recording).

        Starts a worker for the tick and returns immediately: the
        worker snapshots the growing mic/system tracks, mixes the
        snapshots, and transcribes the newly completed windows
        (see :class:`_LiveChunkWorker`). ``combined.wav`` does NOT
        grow during recording — it is created only at stop — so
        the tracks are the only thing a poll tick can read. A tick
        skipped while a chunk is still transcribing, or while
        paused, loses nothing: the watermark is unchanged and the
        next tick picks the window up.
        """
        if self._live_download_pending:
            started = self._live_download_started_at
            if started is not None:
                elapsed = time.monotonic() - started
                model = (self._live_chunker.model
                         if self._live_chunker is not None else "?")
                size = WHISPER_MODELS.get(model, {}).get("size_mb")
                size_text = f" (~{size} MB)" if size else ""
                self.nearlive_label.setText(
                    f"Downloading Whisper {model}{size_text} — "
                    f"first run only · {elapsed:.0f}s elapsed")
        if (self._live_failed or self._live_busy
                or self._live_chunker is None):
            return
        if self.capture is None or self.capture.state != CaptureState.RECORDING:
            return
        if not self.session_dir:
            return
        mic = self.session_dir / MIC_FILE
        system = self.session_dir / SYSTEM_FILE
        if not mic.exists() or not system.exists():
            return
        snapshot_dir = Path(tempfile.mkdtemp(prefix="offair-live-"))
        self._live_busy = True
        self._live_thread = QThread(self)
        self._live_worker = _LiveChunkWorker(
            self._live_chunker, mic, system, snapshot_dir)
        self._live_worker.moveToThread(self._live_thread)
        self._live_thread.started.connect(self._live_worker.run)
        self._live_worker.finished.connect(
            self._on_live_chunks_finished)
        self._live_worker.error.connect(self._on_live_chunk_error)
        self._live_worker.finished.connect(self._live_thread.quit)
        self._live_worker.error.connect(self._live_thread.quit)
        self._live_worker.finished.connect(
            self._live_worker.deleteLater)
        self._live_worker.error.connect(
            self._live_worker.deleteLater)
        self._live_thread.finished.connect(
            self._live_thread.deleteLater)
        self._live_thread.start()

    def _on_live_chunks_finished(self, chunks) -> None:
        """A live tick landed: append its segments, save the live file."""
        self._live_busy = False
        self._live_worker = None
        if self._live_abandoned:
            # Stop already gave up on this worker (see
            # _on_live_finalize_timeout): its late result belongs
            # to a chunker the finalize no longer uses — ignore it
            # instead of appending segments or re-finalizing.
            return
        if self._live_download_pending:
            self._live_download_pending = False
            self._live_download_started_at = None
        if chunks:
            if not self._live_list_started:
                self._live_list_started = True
                self.nearlive_list.clear()
            for chunk in chunks:
                for seg in chunk.segments:
                    minutes, seconds = divmod(int(seg.start), 60)
                    speaker = getattr(seg, "speaker", None) or "Provisional"
                    # Provisional styling (Nov 21 2026): live
                    # items are grey + italic until finalize
                    # confirms them — they are never presented as
                    # final, and no confidence number is shown.
                    item = QListWidgetItem(
                        f"[{minutes:02d}:{seconds:02d}] {speaker}: "
                        f"{seg.text} (provisional)")
                    font = item.font()
                    font.setItalic(True)
                    item.setFont(font)
                    item.setForeground(QColor("#6B7280"))
                    self.nearlive_list.addItem(item)
            # Progressive save — the crash-recovery record, written
            # after every chunk so a crash loses at most the chunk
            # in flight.
            if (self.session_dir is not None
                    and self._live_chunker is not None):
                try:
                    self._live_chunker.save_live(self.session_dir)
                except Exception:  # noqa: BLE001 — a failed save is logged, not fatal
                    logger.exception("Failed to save live transcript")
            chunker = self._live_chunker
            if chunker is not None:
                self.nearlive_label.setText(
                    f"Live: {chunker.chunks_done} chunk(s), "
                    f"{len(chunker.segments)} segments — watermark "
                    f"{chunker.watermark_seconds:.1f}s.")
        if self._stop_finalize_pending:
            self._stop_finalize_pending = False
            self._live_finalize_timeout_timer.stop()
            self._live_wait_timer.stop()
            self._live_wait_started_at = None
            self._finalize_transcription()
        self._maybe_close_pending()

    def _on_live_chunk_error(self, message: str) -> None:
        """A live chunk failed: stop live loudly, keep everything.

        The recording is untouched and the saved live transcript
        (transcript-live.json) is kept — Stop still runs, using the
        live chunks plus the tail, or the full-file path when there
        are none.
        """
        self._live_busy = False
        self._live_worker = None
        if self._live_abandoned:
            # See _on_live_chunks_finished: a late error from an
            # abandoned worker must not touch finalize state.
            return
        self._live_failed = True
        self._live_timer.stop()
        self.nearlive_label.setText(
            f"Live transcription stopped: {message} — recording "
            "continues; the saved live transcript is kept and Stop "
            "will use it plus the tail (or a full transcription "
            "when there is none).")
        logger.error("Live transcription stopped: %s", message)
        self.statusBar().showMessage(
            f"Live transcription stopped: {message}")
        if self._stop_finalize_pending:
            self._stop_finalize_pending = False
            self._live_finalize_timeout_timer.stop()
            self._live_wait_timer.stop()
            self._live_wait_started_at = None
            self._finalize_transcription()
        self._maybe_close_pending()

    def _update_live_wait_label(self) -> None:
        """Refresh the "Waiting for live chunk…" countdown line.

        Only meaningful while :attr:`_stop_finalize_pending` — the
        1 s wait timer is stopped everywhere the pending state
        clears, so a stale tick can never overwrite a later status.
        """
        if not self._stop_finalize_pending:
            return
        started = self._live_wait_started_at
        elapsed = (time.monotonic() - started) if started is not None else 0.0
        cap = self._live_finalize_wait_seconds
        self.status_label.setText(
            f"Waiting for live chunk… {elapsed:.0f}s / {cap:.0f}s — "
            "Cancel to finalize now")

    def _abandon_live_and_finalize(self) -> None:
        """Give up on the in-flight live chunk and finalize now.

        Shared by the 60 s timeout and by Cancel during the wait:
        the in-memory chunker is detached (the stuck worker still
        owns it and may mutate it later — its late finished/error
        signals are ignored, see ``_live_abandoned``), and the
        finalize proceeds from the saved live transcript on disk,
        falling back to the full-file path when there is none. The
        abandoned worker's thread is left to finish on its own; it
        touches nothing the finalize uses.
        """
        if not self._stop_finalize_pending:
            return
        self._stop_finalize_pending = False
        self._live_abandoned = True
        self._live_busy = False
        self._live_finalize_timeout_timer.stop()
        self._live_wait_timer.stop()
        self._live_wait_started_at = None
        self._live_chunker = None
        self._update_buttons()
        self._finalize_transcription()
        self._maybe_close_pending()

    def _on_live_finalize_timeout(self) -> None:
        """Stop waited too long for an in-flight live chunk.

        The live worker may be stuck in a first-run model download
        or a pathological chunk; Stop must not wait forever.
        """
        if not self._stop_finalize_pending:
            return
        logger.warning(
            "Live transcription did not finish within %.0fs of Stop — "
            "abandoning the in-flight chunk and finalizing from the "
            "saved live transcript",
            self._live_finalize_wait_seconds)
        self.status_label.setText(
            "Live transcription did not finish in time — finalizing "
            "from the saved live transcript…")
        self._abandon_live_and_finalize()

    def _begin_stop_finalize(self) -> None:
        """Stop-time transcription: live + tail, else the full file.

        When a live chunk is still transcribing its snapshot, the
        finalize waits for it — the chunker state must never be
        raced by the tail pass — and runs from the chunk slot. The
        wait is bounded by ``_live_finalize_wait_seconds``: if the
        worker is stuck (e.g. in a model download),
        :meth:`_on_live_finalize_timeout` abandons it and the
        finalize proceeds from the saved live transcript instead.
        The wait is visible (a live countdown in the status line,
        refreshed every second) and cancellable: Cancel is enabled
        while waiting, and pressing it abandons immediately (see
        :meth:`on_cancel_transcribe`) — the user is never left
        watching a static label with no way out (Nov 20 2026
        Stop-freeze fix).
        """
        if self._live_busy:
            self._stop_finalize_pending = True
            self._live_abandoned = False
            self._live_wait_started_at = time.monotonic()
            self._update_live_wait_label()
            self._live_wait_timer.start()
            self._live_finalize_timeout_timer.start(
                int(self._live_finalize_wait_seconds * 1000))
            self._update_buttons()
            return
        self._finalize_transcription()

    def _finalize_transcription(self) -> None:
        """Write the canonical transcript from live chunks + tail.

        The live chunker (or the saved live transcript, when the
        in-memory chunker is gone) supplies the completed windows.
        The tail pass itself — :meth:`ChunkedTranscriber.finish`
        over every remaining window plus the final partial window,
        from the final ``combined.wav`` now that it exists — runs
        in a :class:`_FinalizeWorker` thread, never on the UI
        thread: when live died early the tail is the bulk of the
        meeting, and a first-use model download happens inside it
        too, so running it inline would freeze the UI for minutes
        with no progress and no cancel. The worker merges live +
        tail with the chunker's dedup (no word twice), writes the
        canonical transcript.json / .txt / .srt, deletes the live
        file, and its finished slot runs the existing auto-process
        chain (cleanup → diarize → suggest) exactly as the
        full-file path does. With no live state at all (live failed
        before the first chunk, or the app restarted
        mid-recording), the existing full-file transcribe_session
        path runs instead — unless there is no audio at all, in
        which case there is nothing to transcribe and Stop just
        notes that (no worker, no error dialog).
        """
        session_dir = self.session_dir
        if session_dir is None or self._finalizing:
            return
        chunker = self._live_chunker
        if chunker is None or (chunker.chunks_done == 0
                               and not chunker.segments):
            live_state = None
            try:
                live_state = load_live_transcript(session_dir)
            except Exception:  # noqa: BLE001 — a corrupt live file falls back, loudly in the log
                logger.exception(
                    "Could not load live transcript — falling back "
                    "to full transcription")
            if live_state is not None and live_state["segments"]:
                # The saved live transcript names the model its
                # chunks were transcribed with — the tail must use
                # that model, not whatever is configured now.
                chunker = ChunkedTranscriber(
                    model=live_state.get("model")
                    or self._configured_model(),
                    engine=self._live_engine)
                chunker.restore_live_state(
                    segments=live_state["segments"],
                    chunks_done=live_state["chunks_done"],
                    watermark_seconds=live_state["watermark_seconds"])
        if chunker is None or (chunker.chunks_done == 0
                               and not chunker.segments):
            # No live state: the existing full-file path — unless
            # there is no audio to transcribe at all.
            self._live_chunker = None
            combined = session_dir / COMBINED_FILE
            mic_path = session_dir / MIC_FILE
            if not combined.exists() and not mic_path.exists():
                self._auto_processing = False
                self.status_label.setText(
                    "Stopped. No audio was recorded — nothing to "
                    "transcribe.")
                return
            self.on_transcribe()
            return
        # Hand the chunker to the worker: the window keeps no
        # reference to it while the thread owns it, so no other
        # slot can race the tail pass.
        model = chunker.model
        self._live_chunker = None
        self._finalizing = True
        self._finalize_cancel_event = threading.Event()
        self.transcribe_progress.setVisible(True)
        self.transcribe_progress.setValue(0)
        self.transcribe_label.setText(
            f"Finalizing transcript with Whisper {model} (CPU, "
            "int8)… first run downloads the model.")
        self._update_buttons()
        self._finalize_thread = QThread(self)
        self._finalize_worker = _FinalizeWorker(
            session_dir, chunker, model,
            is_cancelled=self._finalize_cancel_event.is_set)
        self._finalize_worker.moveToThread(self._finalize_thread)
        self._finalize_thread.started.connect(self._finalize_worker.run)
        self._finalize_worker.progress.connect(
            self._on_finalize_progress)
        self._finalize_worker.finished.connect(
            self._on_finalize_finished)
        self._finalize_worker.error.connect(self._on_finalize_error)
        self._finalize_worker.cancelled.connect(
            self._on_finalize_cancelled)
        self._finalize_worker.finished.connect(
            self._finalize_thread.quit)
        self._finalize_worker.error.connect(self._finalize_thread.quit)
        self._finalize_worker.cancelled.connect(
            self._finalize_thread.quit)
        self._finalize_worker.finished.connect(
            self._finalize_worker.deleteLater)
        self._finalize_worker.error.connect(
            self._finalize_worker.deleteLater)
        self._finalize_worker.cancelled.connect(
            self._finalize_worker.deleteLater)
        self._finalize_thread.finished.connect(
            self._finalize_thread.deleteLater)
        self._finalize_thread.start()

    def _on_finalize_progress(self, fraction: float, text: str) -> None:
        self.transcribe_progress.setValue(int(fraction * 100))
        self.transcribe_label.setText(text)

    def _on_finalize_finished(self, summary: dict) -> None:
        self._finalizing = False
        self._finalize_cancel_event = None
        self._finalize_worker = None
        self._finalize_thread = None
        self._after_transcribe_finished(summary)
        self._maybe_close_pending()

    def _on_finalize_error(self, info: dict) -> None:
        """A finalize worker failed: tail failure falls back to the
        full file; a write failure is surfaced and stops the chain.

        Either way the recording is untouched. On a write failure
        the saved live transcript is left on disk — the session
        stays transcribable by hand (Transcribe in the Session
        view) — and ``_auto_processing`` is cleared so the chain
        never diarizes a transcript that was never written.
        """
        self._finalizing = False
        self._finalize_cancel_event = None
        self._finalize_worker = None
        self._finalize_thread = None
        self.transcribe_progress.setVisible(False)
        message = info.get("message", "unknown error")
        if info.get("phase") == "tail":
            logger.warning(
                "Live + tail finalize failed — falling back to full "
                "transcription: %s", message)
            self.status_label.setText(
                f"Live finalize failed ({message}) — falling back "
                "to full transcription…")
            self.on_transcribe()
            return
        self._auto_processing = False
        self.transcribe_label.setText(
            f"Final transcript failed: {message} — the live "
            "transcript is kept; run Transcribe in the Session "
            "view to transcribe the full recording by hand.")
        self.statusBar().showMessage("Final transcript failed.")
        QMessageBox.critical(
            self, "Final transcript failed",
            f"Could not write the final transcript: {message}\n\n"
            "The recording is untouched and the live transcript "
            "(transcript-live.json) is kept — run Transcribe in "
            "the Session view to transcribe the full recording "
            "by hand.")
        self._update_buttons()
        self._maybe_close_pending()

    def _on_finalize_cancelled(self, info: dict) -> None:
        """The user cancelled the Stop finalize.

        No transcript.json is written and the saved live
        transcript stays on disk, so the session can still be
        transcribed by hand (Transcribe in the Session view).
        The auto-process chain stops quietly — the user asked
        for this — and the recording is untouched.
        """
        self._finalizing = False
        self._auto_processing = False
        self._finalize_cancel_event = None
        self._finalize_worker = None
        self._finalize_thread = None
        self.transcribe_progress.setVisible(False)
        self.transcribe_label.setText(
            "Finalize cancelled — no transcript.json written. The "
            "live transcript is kept; run Transcribe in the "
            "Session view to transcribe the full recording by hand.")
        self.statusBar().showMessage(
            "Finalize cancelled — no transcript written.")
        self._update_buttons()
        self._maybe_close_pending()

    # ------------------------------------------------------------ near-live ---
    def on_nearlive(self) -> None:
        """Chunked (near-live) transcription of the loaded session.

        The model is resolved ONCE here, via :meth:`_configured_model`,
        and that one value goes to the worker and to both status
        strings — the label can never disagree with the worker
        (Nov 20 2026, Item 7: the manual path used to hard-code
        tiny while the live path used the configured model).
        """
        if not self.session_dir or self._nearlive_running:
            return
        model = self._configured_model()
        self._nearlive_running = True
        self.nearlive_progress.setVisible(True)
        self.nearlive_progress.setValue(0)
        self.nearlive_list.clear()
        self.nearlive_label.setText(
            f"Near-live: {NEARLIVE_CHUNK_SECONDS:.0f} s windows, Whisper "
            f"{model} (CPU)…")
        self._update_buttons()
        self._nearlive_thread = QThread(self)
        self._nearlive_worker = _NearLiveWorker(
            self.session_dir, model)
        self._nearlive_worker.moveToThread(self._nearlive_thread)
        self._nearlive_thread.started.connect(self._nearlive_worker.run)
        self._nearlive_worker.progress.connect(
            self._on_nearlive_progress)
        self._nearlive_worker.finished.connect(
            self._on_nearlive_finished)
        self._nearlive_worker.error.connect(self._on_nearlive_error)
        self._nearlive_worker.finished.connect(self._nearlive_thread.quit)
        self._nearlive_worker.error.connect(self._nearlive_thread.quit)
        self._nearlive_thread.start()

    def _on_nearlive_progress(self, fraction: float, text: str) -> None:
        self.nearlive_progress.setValue(int(fraction * 100))
        self.nearlive_label.setText(text)
        # The pane itself is filled when the run finishes, from the
        # worker's segments (see _on_nearlive_finished).

    def _on_nearlive_finished(self, summary: dict) -> None:
        self._nearlive_running = False
        self.nearlive_progress.setValue(100)
        # Hide the bar when the step finishes (see
        # _on_transcribe_finished — a bar left visible at 100%
        # looks like a dead button).
        self.nearlive_progress.setVisible(False)
        worker = self._nearlive_worker
        self.nearlive_label.setText(
            f"Done: {summary['chunks']} chunks, {summary['segments']} "
            f"segments in {summary['wall_seconds']}s (Whisper "
            f"{worker.model_size if worker is not None else '?'}, "
            "provisional labels). Chunk "
            "boundaries can split Whisper segments — see PHASE4.md.")
        self.nearlive_list.clear()
        for seg in (worker.segments if worker is not None else []):
            minutes, seconds = divmod(int(seg.start), 60)
            speaker = getattr(seg, "speaker", None) or "Provisional"
            self.nearlive_list.addItem(
                f"[{minutes:02d}:{seconds:02d}] {speaker}: {seg.text}")
        self.statusBar().showMessage(
            f"Near-live done for {summary['session_dir']}")
        self._update_buttons()

    def _on_nearlive_error(self, message: str) -> None:
        self._nearlive_running = False
        self.nearlive_progress.setVisible(False)
        self.nearlive_label.setText(f"Near-live failed: {message}")
        QMessageBox.critical(self, "Near-live failed", message)
        self._update_buttons()

    # ------------------------------------------------ Phase 5: export/LLM ---
    def on_export_llm(self) -> None:
        """Write the LLM-ready Markdown export for the loaded session."""
        if not self.session_dir:
            return
        try:
            from app.exports.llm_export import (
                export_llm_markdown_from_session,
            )
            path = export_llm_markdown_from_session(self.session_dir)
        except Exception as exc:  # noqa: BLE001 — surfaced to the user
            logger.exception("LLM export failed")
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.export_label.setText(
            f"LLM Markdown written to {path.name} — paste it into an "
            "LLM for a summary / action items. No action items are "
            "invented by OffAir.")
        self.statusBar().showMessage(f"LLM export saved: {path}")
        # The export flag changed: the chooser row and the
        # loaded-session line follow.
        self._refresh_session_chooser()
        self._update_loaded_session_label()

    def on_summarize_local(self) -> None:
        """Summarise the loaded session with the local LLM (if enabled)."""
        if not self.session_dir or self._summarizing:
            return
        if not self.config.ollama_enabled:
            QMessageBox.information(
                self, "Local LLM summaries are off",
                "Local LLM summaries are off by default. Enable them "
                "in Settings (Ollama on localhost), start Ollama, and "
                "try again — or paste the LLM Markdown export into an "
                "LLM yourself. OffAir never uses a cloud LLM.")
            return
        self._summarizing = True
        self.summarize_label.setText(
            "Asking the local LLM (Ollama on localhost)… this can "
            "take a while on CPU.")
        self._update_buttons()
        self._summarize_thread = QThread(self)
        self._summarize_worker = _SummarizeWorker(
            self.session_dir, self.config)
        self._summarize_worker.moveToThread(self._summarize_thread)
        self._summarize_thread.started.connect(self._summarize_worker.run)
        self._summarize_worker.progress.connect(
            lambda _frac, text: self.summarize_label.setText(text))
        self._summarize_worker.finished.connect(
            self._on_summarize_finished)
        self._summarize_worker.error.connect(self._on_summarize_error)
        self._summarize_worker.finished.connect(
            self._summarize_thread.quit)
        self._summarize_worker.error.connect(self._summarize_thread.quit)
        self._summarize_thread.start()

    def _on_summarize_finished(self, summary: str) -> None:
        self._summarizing = False
        self.summarize_label.setText(
            "Summary (local LLM — Ollama, localhost only):\n" + summary)
        self._update_buttons()

    def _on_summarize_error(self, message: str) -> None:
        self._summarizing = False
        self.summarize_label.setText(f"Local LLM summary failed: {message}")
        QMessageBox.critical(self, "Local LLM summary failed", message)
        self._update_buttons()

    def on_open_settings(self) -> None:
        """Open Settings — since the Nov 19 2026 redesign the
        settings live in the Settings view, not a dialog: the
        header button switches to it. The dialog itself
        (``edit_settings``) still exists for standalone use, and
        the view's Save path is :meth:`_on_settings_save`."""
        self.show_view("Settings")

    def on_open_browser(self) -> None:
        """Open the Recording Browser window."""
        try:
            from app.ui.browser import create_browser
            browser = create_browser(self.config, parent=self)
            browser.exec()
        except Exception as exc:  # noqa: BLE001 — surfaced to the user
            QMessageBox.critical(self, "Recording Browser failed", str(exc))
            return
        # The browser may have exported (or opened) a session: the
        # chooser's status flags follow.
        self._refresh_session_chooser()
        self._update_loaded_session_label()
        self._update_buttons()

    # ------------------------------------------------------------ helpers ---
    def _update_elapsed(self) -> None:
        if self._record_started_at is None:
            return
        paused = 0.0
        if self._pause_started_at is not None:
            paused = time.time() - self._pause_started_at
        elapsed = (time.time() - self._record_started_at
                   - self._elapsed_paused_total - paused)
        minutes, seconds = divmod(max(0, int(elapsed)), 60)
        self.elapsed_label.setText(f"{minutes:02d}:{seconds:02d}")
        # Tray tooltip (Nov 21 2026): elapsed recording time.
        if self._tray is not None:
            self._tray.setToolTip(
                f"OffAir — recording {minutes:02d}:{seconds:02d}")


def create_main_window(config: AppConfig | None = None,
                       schedules_path: str | Path | None = None) -> MainWindow:
    """Build the main window, or raise a clear error if PyQt6 is missing.

    ``schedules_path`` is the test seam for the schedule store
    (Nov 22 2026) — None means the default ``~/.offair`` store.
    """
    if not QT_AVAILABLE:
        raise RuntimeError(
            "PyQt6 is not installed — install it with "
            "`pip install -r requirements.txt` to use the GUI.")
    return MainWindow(config, schedules_path=schedules_path)
