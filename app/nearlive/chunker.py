"""Chunked (near-live) transcription — PRD Phase 4.

:class:`ChunkedTranscriber` slices audio into 20–30 s windows and
transcribes one window at a time with the Phase 2 engine, so a
transcript grows progressively instead of appearing only at the
end. Speaker labels are *provisional*: when mic + system chunk
files are available they come from the Phase 3 simple mode
(dual-channel energy); otherwise the label is ``"Provisional"``.
A chunk's segments are offset to absolute session time before
they are appended, and the 0.5 s overlap between consecutive
chunks is de-duplicated (a segment whose start falls inside the
previous chunk's overlap and whose text duplicates an already
emitted segment is dropped), so no audio is transcribed twice and
no word is lost at a boundary.

Two entry points:

* :meth:`ChunkedTranscriber.process_file` — chunk a finished WAV
  (the post-stop simulation / long-recording path).
* :meth:`ChunkedTranscriber.process_new_audio` — near-live during
  recording: given a growing WAV file, transcribe every newly
  completed window and advance the watermark. Callers pass a
  *snapshot* copy of the file (capture may still be writing the
  original), and the method never re-processes a window.

The engine is injectable (tests pass a fake); the default is a
real :class:`TranscriptionEngine` with the code-fallback model
(``NEARLIVE_DEFAULT_MODEL``) — callers pass the configured model
explicitly, and base keeps up with real time too (RTF ~0.135).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from app.diarization.simple import label_segments_dual, load_wav_mono
from app.storage.sessions import TRANSCRIPT_LIVE_FILE
from app.transcription.engine import Segment, TranscriptionEngine

logger = logging.getLogger("offair.nearlive")

DEFAULT_CHUNK_SECONDS = 25.0
DEFAULT_OVERLAP_SECONDS = 0.5
MIN_CHUNK_SECONDS = 5.0
MAX_CHUNK_SECONDS = 60.0
NEARLIVE_DEFAULT_MODEL = "tiny"
PROVISIONAL_LABEL = "Provisional"


@dataclass
class ChunkResult:
    """One transcribed chunk, with absolute (session) timestamps."""

    index: int
    start: float
    end: float
    segments: list[Segment] = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def real_time_factor(self) -> float | None:
        duration = self.end - self.start
        return self.wall_seconds / duration if duration else None


def split_wav_into_chunks(wav_path: str | Path, *,
                          chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
                          overlap_seconds: float = DEFAULT_OVERLAP_SECONDS
                          ) -> list[tuple[float, float]]:
    """[(start, end), ...] windows covering a WAV file.

    Windows are ``chunk_seconds`` long, overlapping by
    ``overlap_seconds``; the last window is shortened to the file
    end (and dropped if it would be under 1 s and is not the only
    window). Raises FileNotFoundError / ValueError for bad input.
    """
    if not MIN_CHUNK_SECONDS <= chunk_seconds <= MAX_CHUNK_SECONDS:
        raise ValueError(
            f"chunk_seconds must be {MIN_CHUNK_SECONDS}–{MAX_CHUNK_SECONDS}, "
            f"got {chunk_seconds}")
    if not 0.0 <= overlap_seconds < chunk_seconds:
        raise ValueError(
            f"overlap_seconds must be 0–chunk_seconds, got {overlap_seconds}")
    samples, rate = load_wav_mono(wav_path)
    duration = len(samples) / rate if rate else 0.0
    if duration <= 0:
        raise ValueError(f"No audio in {wav_path}")
    step = chunk_seconds - overlap_seconds
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + chunk_seconds)
        windows.append((round(start, 3), round(end, 3)))
        if end >= duration:
            break
        start += step
    # Drop a trailing sliver (<1 s) — its words are already covered
    # by the previous window's overlap.
    if len(windows) > 1 and windows[-1][1] - windows[-1][0] < 1.0:
        windows.pop()
    return windows


def _write_chunk_wav(samples: np.ndarray, rate: int, start: float,
                     end: float, out_path: Path) -> Path:
    """Write one window of samples to a WAV file (soundfile, int16)."""
    import soundfile as sf

    i0 = max(0, int(start * rate))
    i1 = min(len(samples), int(end * rate))
    sf.write(str(out_path), samples[i0:i1], rate, subtype="PCM_16")
    return out_path


def save_live_transcript(session_dir: str | Path, *, model: str,
                         segments: list[Segment], chunks_done: int,
                         watermark_seconds: float) -> Path:
    """Write the progressive live transcript (``transcript-live.json``).

    Same payload shape as ``transcript.json`` plus ``"live": true``,
    ``chunks_done`` and ``watermark_seconds``. Written after every
    live chunk during recording; it is the crash-recovery record,
    not a substitute for the final ``transcript.json``.
    """
    payload = {
        "live": True,
        "model": model,
        "chunks_done": chunks_done,
        "watermark_seconds": round(float(watermark_seconds), 3),
        "segments": [s.to_dict() for s in segments],
    }
    path = Path(session_dir) / TRANSCRIPT_LIVE_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_live_transcript(session_dir: str | Path) -> dict | None:
    """Load a saved live transcript, or None when there is none.

    Returns the payload dict with ``"segments"`` as a list of
    :class:`Segment`. Raises on a corrupt file — the caller falls
    back to the full-file path then.
    """
    path = Path(session_dir) / TRANSCRIPT_LIVE_FILE
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "live": bool(payload.get("live", True)),
        "model": payload.get("model"),
        "chunks_done": int(payload.get("chunks_done", 0)),
        "watermark_seconds": float(payload.get("watermark_seconds", 0.0)),
        "segments": [Segment.from_dict(d)
                     for d in payload.get("segments", [])],
    }


class ChunkedTranscriber:
    """Transcribe audio in 20–30 s windows, progressively.

    ``on_chunk`` is called after each window with the
    :class:`ChunkResult` (absolute timestamps); ``on_segment`` is
    called per segment as it is appended. ``engine`` is an injected
    :class:`TranscriptionEngine` (or a fake with ``transcribe_file``
    returning a ``TranscriptionResult``); with ``None`` a real engine
    is created lazily with ``model``.
    """

    def __init__(self, *, chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
                 overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
                 model: str = NEARLIVE_DEFAULT_MODEL,
                 engine: TranscriptionEngine | None = None,
                 on_chunk=None, on_segment=None,
                 is_cancelled: Callable[[], bool] | None = None) -> None:
        if not MIN_CHUNK_SECONDS <= chunk_seconds <= MAX_CHUNK_SECONDS:
            raise ValueError(
                f"chunk_seconds must be {MIN_CHUNK_SECONDS}–"
                f"{MAX_CHUNK_SECONDS}, got {chunk_seconds}")
        if not 0.0 <= overlap_seconds < chunk_seconds:
            raise ValueError("overlap_seconds must be < chunk_seconds")
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.model = model
        self._engine = engine
        self.on_chunk = on_chunk
        self.on_segment = on_segment
        # Passed through to the engine per chunk (see
        # TranscriptionEngine.transcribe_file): a cancelled chunk
        # returns its partial segments instead of raising.
        self.is_cancelled = is_cancelled
        self.segments: list[Segment] = []
        self.chunks_done = 0
        self.total_wall_seconds = 0.0
        self._watermark = 0.0  # end of the last fully processed window
        self._seen: set[tuple[float, str]] = set()

    @property
    def watermark_seconds(self) -> float:
        """End of the last fully processed window (seconds of audio)."""
        return self._watermark

    # ------------------------------------------------------------ live -----
    def save_live(self, session_dir: str | Path) -> Path:
        """Write this chunker's progressive live transcript."""
        return save_live_transcript(
            session_dir, model=self.model, segments=self.segments,
            chunks_done=self.chunks_done,
            watermark_seconds=self._watermark)

    def restore_live_state(self, *, segments: list[Segment],
                           chunks_done: int,
                           watermark_seconds: float) -> None:
        """Seed this chunker from a saved live transcript.

        The dedup set is rebuilt from the live segments, so a
        finalize tail pass never re-emits a word the live pass
        already produced.
        """
        self.segments = list(segments)
        self.chunks_done = chunks_done
        self._watermark = float(watermark_seconds)
        self._seen = {(round(s.start, 1), s.text.strip())
                      for s in segments}

    # ------------------------------------------------------------ engine ---
    def _get_engine(self) -> TranscriptionEngine:
        if self._engine is None:
            self._engine = TranscriptionEngine(model_size=self.model)
        return self._engine

    def unload(self) -> None:
        """Drop the Whisper model and reclaim memory."""
        if self._engine is not None:
            self._engine.unload()

    # --------------------------------------------------------- deduping ---
    def _append(self, segment: Segment) -> bool:
        """Append a segment unless it duplicates one already emitted.

        A duplicate is a segment whose start (to 0.1 s) and text
        match an earlier one — the overlap region's second copy.
        Returns True if the segment was appended.
        """
        key = (round(segment.start, 1), segment.text.strip())
        if key in self._seen:
            return False
        self._seen.add(key)
        self.segments.append(segment)
        if self.on_segment is not None:
            self.on_segment(segment)
        return True

    # ------------------------------------------------------------ chunks ---
    def _transcribe_window(self, samples: np.ndarray, rate: int,
                           start: float, end: float, index: int,
                           work_dir: Path,
                           mic_samples: np.ndarray | None = None,
                           system_samples: np.ndarray | None = None
                           ) -> ChunkResult:
        """Transcribe one window; offset segments to absolute time."""
        chunk_path = work_dir / f"chunk-{index:03d}.wav"
        _write_chunk_wav(samples, rate, start, end, chunk_path)
        engine = self._get_engine()
        started = time.monotonic()
        transcribe_kwargs: dict = {}
        if self.is_cancelled is not None:
            transcribe_kwargs["is_cancelled"] = self.is_cancelled
        result = engine.transcribe_file(chunk_path, **transcribe_kwargs)
        wall = time.monotonic() - started
        self.total_wall_seconds += wall
        segments = [Segment(start=s.start + start, end=s.end + start,
                            text=s.text, speaker=PROVISIONAL_LABEL)
                    for s in result.segments]
        # Provisional speaker labels from dual-channel energy, when
        # both tracks are available for this window.
        if mic_samples is not None and system_samples is not None:
            labelled = label_segments_dual(
                segments, mic_samples, rate, system_samples, rate)
            for seg, lab in zip(segments, labelled):
                seg.speaker = lab.speaker
        kept = [s for s in segments if self._append(s)]
        chunk = ChunkResult(index=index, start=start, end=end,
                            segments=kept, wall_seconds=wall)
        self.chunks_done += 1
        if self.on_chunk is not None:
            self.on_chunk(chunk)
        logger.info("Near-live chunk %d [%.1f–%.1f]: %d segments in %.1fs "
                    "(RTF %.2f)", index, start, end, len(kept), wall,
                    chunk.real_time_factor or 0)
        return chunk

    def process_file(self, wav_path: str | Path, *,
                     work_dir: str | Path | None = None,
                     mic_path: str | Path | None = None,
                     system_path: str | Path | None = None
                     ) -> list[ChunkResult]:
        """Chunk a finished WAV and transcribe it progressively.

        Returns one ChunkResult per window. ``work_dir`` (default:
        a ``nearlive-chunks`` folder next to the audio) holds the
        temporary chunk WAVs; they are removed afterwards. With
        ``mic_path`` / ``system_path`` given, provisional labels
        come from dual-channel energy instead of ``"Provisional"``.
        """
        wav_path = Path(wav_path)
        samples, rate = load_wav_mono(wav_path)
        mic_samples = mic_rate = None
        sys_samples = sys_rate = None
        if mic_path is not None and system_path is not None:
            mic_samples, mic_rate = load_wav_mono(mic_path)
            sys_samples, sys_rate = load_wav_mono(system_path)
        windows = split_wav_into_chunks(
            wav_path, chunk_seconds=self.chunk_seconds,
            overlap_seconds=self.overlap_seconds)
        work = Path(work_dir) if work_dir else wav_path.parent / "nearlive-chunks"
        work.mkdir(parents=True, exist_ok=True)
        results: list[ChunkResult] = []
        try:
            for i, (start, end) in enumerate(windows):
                if self.is_cancelled is not None and self.is_cancelled():
                    break
                results.append(self._transcribe_window(
                    samples, rate, start, end, i, work,
                    mic_samples=mic_samples, system_samples=sys_samples))
                self._watermark = end
        finally:
            for tmp in work.glob("chunk-*.wav"):
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return results

    def process_new_audio(self, wav_path: str | Path, *,
                          work_dir: str | Path | None = None,
                          mic_path: str | Path | None = None,
                          system_path: str | Path | None = None
                          ) -> list[ChunkResult]:
        """Near-live during recording: transcribe newly completed windows.

        Reads the (possibly still growing) WAV, transcribes every
        full window ending before the current watermark, and advances
        the watermark. Windows that are not yet complete are left
        for the next call. Returns the chunks transcribed in this
        call (possibly empty). Callers should pass a snapshot copy
        of the file if capture is still writing the original.
        """
        wav_path = Path(wav_path)
        samples, rate = load_wav_mono(wav_path)
        duration = len(samples) / rate if rate else 0.0
        # Windows are anchored at the watermark (minus the overlap),
        # not at zero: precomputed windows anchored at 0 would skip
        # the audio between the watermark and the next window start.
        # Only *full* windows are transcribed live — a partial tail
        # is left for the next call (or for process_file at stop).
        step = self.chunk_seconds - self.overlap_seconds
        anchor = max(0.0, self._watermark - self.overlap_seconds)
        due: list[tuple[float, float]] = []
        start = anchor
        while start + self.chunk_seconds <= duration + 1e-6:
            due.append((round(start, 3),
                        round(start + self.chunk_seconds, 3)))
            start += step
        if not due:
            return []
        mic_samples = sys_samples = None
        if mic_path is not None and system_path is not None:
            mic_samples, _ = load_wav_mono(mic_path)
            sys_samples, _ = load_wav_mono(system_path)
        work = Path(work_dir) if work_dir else wav_path.parent / "nearlive-chunks"
        work.mkdir(parents=True, exist_ok=True)
        results: list[ChunkResult] = []
        base_index = self.chunks_done
        try:
            for i, (start, end) in enumerate(due):
                if self.is_cancelled is not None and self.is_cancelled():
                    break
                index = base_index + i
                results.append(self._transcribe_window(
                    samples, rate, start, end, index, work,
                    mic_samples=mic_samples, system_samples=sys_samples))
                self._watermark = end
        finally:
            for tmp in work.glob("chunk-*.wav"):
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return results

    def finish(self, wav_path: str | Path, *,
               work_dir: str | Path | None = None,
               mic_path: str | Path | None = None,
               system_path: str | Path | None = None
               ) -> list[ChunkResult]:
        """Transcribe everything after the watermark to the end of file.

        The Stop/finalize tail pass: first any full windows still
        due (e.g. live transcription died early, or the last poll
        tick was minutes ago), then the final partial window —
        audio from ``watermark − overlap`` to the file end — as one
        chunk, however short it is: a sub-1 s sliver is still
        transcribed when it holds audio past the watermark (its
        words are new — only the overlap part was covered before),
        and only a tail with no new audio at all is skipped.
        Segments go through the same dedup as live chunks, so
        merging live + tail never duplicates a word. After this
        call the watermark is the file duration. When
        ``is_cancelled`` fires the remaining windows are skipped
        and the chunks done so far are returned.
        """
        wav_path = Path(wav_path)
        results = self.process_new_audio(
            wav_path, work_dir=work_dir, mic_path=mic_path,
            system_path=system_path)
        samples, rate = load_wav_mono(wav_path)
        duration = len(samples) / rate if rate else 0.0
        start = max(0.0, self._watermark - self.overlap_seconds)
        if duration <= self._watermark:
            # Nothing past the watermark: no new audio to transcribe.
            return results
        if self.is_cancelled is not None and self.is_cancelled():
            return results
        mic_samples = sys_samples = None
        if mic_path is not None and system_path is not None:
            mic_samples, _ = load_wav_mono(mic_path)
            sys_samples, _ = load_wav_mono(system_path)
        work = (Path(work_dir) if work_dir
                else wav_path.parent / "nearlive-chunks")
        work.mkdir(parents=True, exist_ok=True)
        try:
            results.append(self._transcribe_window(
                samples, rate, start, duration, self.chunks_done, work,
                mic_samples=mic_samples, system_samples=sys_samples))
            self._watermark = duration
        finally:
            for tmp in work.glob("chunk-*.wav"):
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return results
