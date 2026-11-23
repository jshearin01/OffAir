"""Transcription engine — faster-whisper on CPU with int8 quantization.

PRD Phase 2. CPU-only by construction: ``device`` is forced to ``"cpu"``
and ``compute_type`` defaults to ``"int8"``; any attempt to request a GPU
device raises :class:`ValueError`. Model loading is lazy (nothing is
downloaded or loaded at import time) and :meth:`TranscriptionEngine.unload`
frees the model so a low-RAM machine can reclaim memory between runs.

Progress: :meth:`TranscriptionEngine.transcribe_file` calls
``progress_callback(fraction, elapsed_seconds, estimated_remaining_seconds)``
as segments arrive. ``fraction`` is the fraction of the audio duration
transcribed so far (0..1); the remaining-time estimate is
``elapsed / fraction - elapsed`` and is ``None`` until the first segment.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEFAULT_MODEL,
    WHISPER_DEVICE,
    WHISPER_MODELS,
)
from app.storage.sessions import COMBINED_FILE, MIC_FILE

logger = logging.getLogger("offair.transcription")

ProgressCallback = Callable[[float, float, float | None], None]

TRANSCRIPT_JSON = "transcript.json"
TRANSCRIPT_TXT = "transcript.txt"
TRANSCRIPT_SRT = "transcript.srt"


@dataclass
class Segment:
    """One timestamped transcript segment.

    ``speaker`` is None until Phase 3 diarization runs; it is omitted
    from :meth:`to_dict` while None so Phase 2 transcripts are unchanged.
    """

    start: float
    end: float
    text: str
    speaker: str | None = None

    def to_dict(self) -> dict:
        data = {"start": round(self.start, 3), "end": round(self.end, 3),
                "text": self.text}
        if self.speaker is not None:
            data["speaker"] = self.speaker
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        return cls(start=float(data["start"]), end=float(data["end"]),
                   text=str(data["text"]),
                   speaker=data.get("speaker"))


@dataclass
class TranscriptionResult:
    """What a transcription run produced, plus how long it took.

    ``cancelled`` is True when the run stopped early because
    ``is_cancelled`` fired — ``segments`` then holds only the part
    transcribed so far, and :meth:`TranscriptionEngine.
    transcribe_session` writes no files for a cancelled run.
    """

    segments: list[Segment] = field(default_factory=list)
    model: str = WHISPER_DEFAULT_MODEL
    audio_path: Path | None = None
    audio_duration_seconds: float | None = None
    wall_seconds: float = 0.0
    cancelled: bool = False

    @property
    def real_time_factor(self) -> float | None:
        """Wall time / audio duration (<1 means faster than real time)."""
        if not self.audio_duration_seconds:
            return None
        return self.wall_seconds / self.audio_duration_seconds

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


def audio_duration_seconds(path: str | Path) -> float | None:
    """Duration of a WAV file in seconds (stdlib only), or None if unknown.

    MP3 / other containers return None — the caller then reports progress
    by segment count instead of by audio time.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / rate if rate else None
    except Exception:
        return None


def whisper_model_cached(model_size: str,
                         download_root: str | Path | None = None) -> bool:
    """True when a faster-whisper model is already downloaded.

    faster-whisper (via the Hugging Face hub) caches a model under
    ``<cache>/models--Systran--faster-whisper-<size>/snapshots/<rev>/``,
    and the model file itself is ``model.bin`` in that snapshot —
    a snapshot dir with no ``model.bin`` (empty, or holding only
    metadata) is not a downloaded model. ``download_root`` (when
    given) is the cache faster-whisper will use, so it is the only
    root checked; otherwise the hub cache is resolved the way the
    hub library does: ``HF_HUB_CACHE`` (or its alias
    ``HUGGINGFACE_HUB_CACHE``), then ``HF_HOME/hub``, then
    ``XDG_CACHE_HOME/huggingface/hub``, then
    ``~/.cache/huggingface/hub``. Any error means "not cached" —
    the caller then shows the download status and the model
    construction downloads it, exactly as before. Never raises,
    never touches the network.
    """
    name = f"models--Systran--faster-whisper-{model_size}"
    # An explicit download_root is the cache faster-whisper will
    # use, so it is the only root checked; otherwise resolve the
    # hub cache from the environment (see docstring).
    if download_root:
        roots = [Path(download_root)]
    else:
        roots = []
        hub_cache = (os.environ.get("HF_HUB_CACHE")
                     or os.environ.get("HUGGINGFACE_HUB_CACHE"))
        if hub_cache:
            roots.append(Path(hub_cache))
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            roots.append(Path(hf_home) / "hub")
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            roots.append(Path(xdg_cache) / "huggingface" / "hub")
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen: set[Path] = set()
    try:
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            snapshots = root / name / "snapshots"
            if not snapshots.is_dir():
                continue
            for rev in snapshots.iterdir():
                if rev.is_dir() and (rev / "model.bin").is_file():
                    return True
    except Exception:  # noqa: BLE001 — a cache check never fails a run
        return False
    return False


class TranscriptionEngine:
    """faster-whisper transcription, CPU + int8 only.

    ``model`` is an injectable faster-whisper ``WhisperModel`` (or a fake
    with the same ``transcribe()`` interface) — tests pass a fake so no
    model download or hardware is needed.
    """

    def __init__(self, model_size: str = WHISPER_DEFAULT_MODEL, *,
                 device: str = WHISPER_DEVICE,
                 compute_type: str = WHISPER_COMPUTE_TYPE,
                 download_root: str | Path | None = None,
                 model=None) -> None:
        if model_size not in WHISPER_MODELS:
            raise ValueError(
                f"unknown Whisper model {model_size!r}; supported: "
                f"{sorted(WHISPER_MODELS)}")
        if device != "cpu":
            raise ValueError(
                f"OffAir is CPU-only: device must be 'cpu', got {device!r}")
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = str(download_root) if download_root else None
        self._model = model  # injected (tests) or None -> lazy-load real one

    # ------------------------------------------------------------ model ---
    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed — install it with "
                "`pip install -r requirements.txt` (CPU-only, no CUDA)."
            ) from exc
        logger.info("Loading Whisper model %r (cpu, %s) …",
                    self.model_size, self.compute_type)
        self._model = WhisperModel(
            self.model_size, device="cpu",
            compute_type=self.compute_type,
            download_root=self.download_root)
        return self._model

    def unload(self) -> None:
        """Drop the loaded model and reclaim memory."""
        self._model = None
        gc.collect()

    # ------------------------------------------------------- transcribe ---
    def transcribe_file(self, audio_path: str | Path, *,
                        language: str = "en",
                        initial_prompt: str | None = None,
                        progress_callback: ProgressCallback | None = None,
                        is_cancelled: Callable[[], bool] | None = None
                        ) -> TranscriptionResult:
        """Transcribe one audio file to timestamped segments.

        ``initial_prompt`` is faster-whisper's decode-time context
        (a short string of names/terms that may come up — built from
        the names dictionary by :func:`app.transcription.
        names_dictionary.build_initial_prompt`). It is only passed
        to the model when non-empty, so callers without a dictionary
        behave exactly as before. Raises FileNotFoundError if the
        file does not exist, ValueError for an unsupported model,
        RuntimeError if faster-whisper is missing.

        ``is_cancelled`` is checked before every segment: when it
        fires the run stops and returns the segments transcribed so
        far (``result.cancelled`` is True) instead of raising, so a
        Cancel button is safe and fast. It is also checked before
        the model is loaded and immediately after the load returns:
        a cancel requested while the model is still downloading
        (first run) takes effect as soon as the load finishes — the
        run then unloads the model and returns cancelled without
        transcribing anything. The blocking ``WhisperModel(...)``
        constructor itself cannot be interrupted mid-download, so a
        cancel during a first-run download waits for the load to
        return; the UI says so instead of promising an instant stop.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if is_cancelled is not None and is_cancelled():
            return TranscriptionResult(
                segments=[], model=self.model_size, audio_path=audio_path,
                audio_duration_seconds=audio_duration_seconds(audio_path),
                wall_seconds=0.0, cancelled=True)
        model = self._load_model()
        if is_cancelled is not None and is_cancelled():
            # Cancelled while the model was loading (e.g. during a
            # first-run download): unload and return cancelled
            # without transcribing — no partial work is done.
            self.unload()
            return TranscriptionResult(
                segments=[], model=self.model_size, audio_path=audio_path,
                audio_duration_seconds=audio_duration_seconds(audio_path),
                wall_seconds=0.0, cancelled=True)
        duration = audio_duration_seconds(audio_path)

        started = time.monotonic()
        segments: list[Segment] = []
        cancelled = False
        # faster-whisper returns (generator-of-segments, info); iterating
        # the generator does the work, so progress is reported per segment.
        transcribe_kwargs: dict = {"language": language, "beam_size": 5}
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        raw_segments, _info = model.transcribe(str(audio_path),
                                               **transcribe_kwargs)
        for raw in raw_segments:
            if is_cancelled is not None and is_cancelled():
                cancelled = True
                break
            segments.append(Segment(start=float(raw.start),
                                    end=float(raw.end),
                                    text=str(raw.text).strip()))
            if progress_callback is not None and duration:
                fraction = min(1.0, segments[-1].end / duration)
                elapsed = time.monotonic() - started
                remaining = (elapsed / fraction - elapsed
                             if fraction > 0 else None)
                progress_callback(fraction, elapsed, remaining)
        wall = time.monotonic() - started
        if not cancelled and progress_callback is not None:
            progress_callback(1.0, wall, 0.0)
        result = TranscriptionResult(
            segments=segments, model=self.model_size,
            audio_path=audio_path, audio_duration_seconds=duration,
            wall_seconds=wall, cancelled=cancelled)
        logger.info("Transcribed %s: %d segments in %.1fs (RTF %.2f)%s",
                    audio_path.name, len(segments), wall,
                    result.real_time_factor or 0,
                    " — cancelled" if cancelled else "")
        return result

    def write_transcript_files(
            self, session_dir: str | Path,
            result: TranscriptionResult, *, language: str = "en") -> dict:
        """Write ``transcript.json`` / ``.txt`` / ``.srt`` for ``result``.

        The single writer for a canonical transcript — used both by
        :meth:`transcribe_session` (full-file path) and by the
        live + tail finalize, which builds its ``result`` from the
        chunker's merged segments instead of a single pass. Returns
        the ``transcript.json`` payload. Does not load the model.
        """
        from app.transcription.exports import export_srt, export_txt

        session_dir = Path(session_dir)
        audio_path = result.audio_path
        payload = {
            "model": result.model,
            "device": "cpu",
            "compute_type": self.compute_type,
            "language": language,
            "audio_file": audio_path.name if audio_path is not None else None,
            "audio_duration_seconds": result.audio_duration_seconds,
            "wall_seconds": round(result.wall_seconds, 2),
            "real_time_factor": (round(result.real_time_factor, 3)
                                 if result.real_time_factor is not None
                                 else None),
            "segments": [s.to_dict() for s in result.segments],
        }
        (session_dir / TRANSCRIPT_JSON).write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        export_txt(result.segments, session_dir / TRANSCRIPT_TXT)
        export_srt(result.segments, session_dir / TRANSCRIPT_SRT)
        logger.info("Wrote %s (+ .txt/.srt) in %s",
                    TRANSCRIPT_JSON, session_dir)
        return payload

    def transcribe_session(self, session_dir: str | Path, *,
                           language: str = "en",
                           initial_prompt: str | None = None,
                           progress_callback: ProgressCallback | None = None,
                           is_cancelled: Callable[[], bool] | None = None
                           ) -> TranscriptionResult:
        """Transcribe a Phase 1 session folder.

        Uses ``combined.wav`` if present, else ``mic.wav``. Writes
        ``transcript.json`` (segments + run metadata), ``transcript.txt``
        and ``transcript.srt`` next to the WAVs, and returns the result.
        ``initial_prompt`` is forwarded to :meth:`transcribe_file`
        (names-dictionary biasing — see :mod:`app.transcription.
        names_dictionary`).

        A cancelled run (``is_cancelled`` fired) returns the partial
        segments with ``result.cancelled`` True and writes NO files —
        a cancelled manual transcription never leaves a partial
        ``transcript.json`` behind.
        """
        session_dir = Path(session_dir)
        audio_path = session_dir / COMBINED_FILE
        if not audio_path.exists():
            audio_path = session_dir / MIC_FILE
        if not audio_path.exists():
            raise FileNotFoundError(
                f"No {COMBINED_FILE} or {MIC_FILE} in {session_dir}")
        result = self.transcribe_file(
            audio_path, language=language, initial_prompt=initial_prompt,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled)
        if result.cancelled:
            return result
        self.write_transcript_files(session_dir, result,
                                    language=language)
        return result
