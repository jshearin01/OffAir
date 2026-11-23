"""Full multi-speaker diarization via pyannote.audio — CPU only.

PRD Phase 3 full mode. This module never imports pyannote or torch at
module import time: both are imported lazily inside
:class:`PyannoteDiarizer`, so importing OffAir, running the simple
mode, and running the unit tests all work on a machine without them.

Setup the user must do first (guided — see :func:`token_setup_guide`
and README § "Phase 3"):

1. Create a free Hugging Face account and a read token at
   https://huggingface.co/settings/tokens
2. Accept the licences for the two gated models:
   * https://huggingface.co/pyannote/speaker-diarization-3.1
   * https://huggingface.co/pyannote/segmentation-3.0
   (the diarization pipeline also pulls pyannote/embedding — accepting
   the two above is what gates access in practice; the pipeline will
   name any further gated model in its error if one appears)
3. Make the token available as the ``HF_TOKEN`` environment variable
   (or ``HUGGING_FACE_HUB_TOKEN``), or pass ``token=...`` directly.
   Never commit a token, and never paste one into a file in this repo.

If the token is missing or the licences are not accepted, diarization
raises :class:`HuggingFaceTokenError` with those steps in the message —
it never silently downloads a gated model, and never embeds a token.

CPU warning: the full pipeline loads a segmentation model plus a
speaker-embedding model and runs them over the whole recording. On a
CPU-only laptop this is much slower than the simple mode (minutes for
a 30-minute meeting is normal) and uses ~1–2 GB of RAM. That is why
the UI shows a warning and the simple mode stays the default.

Min/max speaker hints: ``min_speakers`` / ``max_speakers`` are passed
to the pipeline as ``min_speakers`` / ``max_speakers``. They are
validated here (integers, 1–10, min <= max) so a bad hint fails fast
with a clear message instead of inside the pipeline.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("offair.diarization.pyannote")

PIPELINE_MODEL = "pyannote/speaker-diarization-3.1"
SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
MIN_SPEAKERS_LIMIT = 1
MAX_SPEAKERS_LIMIT = 10

TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def token_setup_guide() -> str:
    """The guided setup text shown when full diarization cannot run."""
    return (
        "Full diarization needs a Hugging Face token and accepted model "
        "licences (free, no payment):\n"
        "  1. Create a free account at https://huggingface.co/join\n"
        "  2. Create a read token at "
        "https://huggingface.co/settings/tokens\n"
        "  3. Open and accept the licences for:\n"
        f"       {PIPELINE_MODEL}\n"
        f"       {SEGMENTATION_MODEL}\n"
        "  4. Set the token before starting OffAir, e.g. in PowerShell:\n"
        "       $env:HF_TOKEN = \"hf_...\"   (then run: python main.py)\n"
        "     or pass token=... to PyannoteDiarizer directly.\n"
        "  5. Run full diarization once — the models download (~1 GB) "
        "and are then cached locally; later runs are fully offline.\n"
        "No token is stored by OffAir, and no audio ever leaves the "
        "machine: the models run locally on CPU after download."
    )


class HuggingFaceTokenError(RuntimeError):
    """Raised when full diarization cannot run for token/licence reasons."""


class PyannoteNotInstalledError(RuntimeError):
    """Raised when pyannote.audio (or torch) is not installed."""


def resolve_hf_token(token: str | None = None) -> str:
    """Return a HF token from an argument or the environment, or raise.

    The error message is the guided setup (see :func:`token_setup_guide`)
    — the user is told exactly what to do, not just that it failed.
    """
    if token and token.strip():
        return token.strip()
    for var in TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value and value.strip():
            return value.strip()
    raise HuggingFaceTokenError(
        "No Hugging Face token found (checked the token argument and "
        "the HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variables).\n\n"
        + token_setup_guide())


def validate_speaker_hints(min_speakers: int | None,
                           max_speakers: int | None) -> tuple[int | None, int | None]:
    """Validate optional min/max speaker count hints.

    Returns the (possibly None) hints unchanged. Raises ValueError for
    non-integers, values outside 1–10, or min > max.
    """
    for name, value in (("min_speakers", min_speakers),
                        ("max_speakers", max_speakers)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if not (MIN_SPEAKERS_LIMIT <= value <= MAX_SPEAKERS_LIMIT):
            raise ValueError(
                f"{name} must be between {MIN_SPEAKERS_LIMIT} and "
                f"{MAX_SPEAKERS_LIMIT}, got {value}")
    if (min_speakers is not None and max_speakers is not None
            and min_speakers > max_speakers):
        raise ValueError(
            f"min_speakers ({min_speakers}) must not exceed "
            f"max_speakers ({max_speakers})")
    return min_speakers, max_speakers


@dataclass
class FullDiarizationResult:
    """What a full pyannote run produced."""

    # (start, end, "SPEAKER_00") turns, sorted by start time.
    turns: list[tuple[float, float, str]] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    model: str = PIPELINE_MODEL
    audio_duration_seconds: float | None = None
    wall_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {"mode": "full", "method": "pyannote",
                "model": self.model,
                "audio_duration_seconds": self.audio_duration_seconds,
                "wall_seconds": round(self.wall_seconds, 2),
                "speakers": self.speakers,
                "turns": [{"start": round(s, 3), "end": round(e, 3),
                           "speaker": sp} for s, e, sp in self.turns]}


class PyannoteDiarizer:
    """pyannote.audio speaker-diarization pipeline, CPU only.

    ``pipeline`` is injectable (tests pass a fake with the same
    call interface) so the class is testable without torch or a token.
    With a real pipeline, :meth:`diarize` runs it on CPU and returns
    generic ``SPEAKER_00``/``SPEAKER_01``… labels — no names are ever
    inferred; renaming is Phase 3's Speaker Name Panel (names.py).
    """

    def __init__(self, token: str | None = None, *,
                 model: str = PIPELINE_MODEL,
                 pipeline=None) -> None:
        self.model = model
        self._pipeline = pipeline  # injected (tests) or None -> lazy-load
        self._token = token  # resolved lazily, only when loading the pipeline

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        token = resolve_hf_token(self._token)
        try:
            from pyannote.audio import Pipeline
            import torch
        except ImportError as exc:
            raise PyannoteNotInstalledError(
                "pyannote.audio (and torch, CPU) is not installed — "
                "install it with `pip install -r requirements.txt`. "
                "It is a large CPU-only install (~1 GB) and is only "
                "needed for full diarization, not for the simple mode."
            ) from exc
        logger.info("Loading pyannote pipeline %r on CPU …", self.model)
        try:
            pipeline = Pipeline.from_pretrained(self.model, token=token)
        except Exception as exc:
            message = str(exc)
            if "401" in message or "gated" in message.lower() \
                    or "licen" in message.lower():
                raise HuggingFaceTokenError(
                    "Hugging Face refused access to the diarization "
                    f"model ({self.model}). This usually means the token "
                    "is wrong, or the model licences have not been "
                    "accepted.\n\n" + token_setup_guide()
                ) from exc
            raise
        # CPU-only by construction — never .to("cuda").
        pipeline.to(torch.device("cpu"))
        self._pipeline = pipeline
        return pipeline

    def diarize(self, audio_path: str | Path, *,
                min_speakers: int | None = None,
                max_speakers: int | None = None,
                progress_callback=None) -> FullDiarizationResult:
        """Diarize one audio file. Returns speaker turns (start/end/label).

        ``progress_callback`` (if given) is called as ``(fraction, text)``
        — the pipeline itself is mostly one long call, so the callback
        reports start (0.0), running (0.5), and done (1.0); it is a
        progress *indicator*, not a precise meter, and the UI says so.
        """
        import time

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        min_speakers, max_speakers = validate_speaker_hints(
            min_speakers, max_speakers)
        pipeline = self._load_pipeline()
        if progress_callback is not None:
            progress_callback(0.0, "Loading diarization model…")
        started = time.monotonic()
        kwargs: dict = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        if progress_callback is not None:
            progress_callback(0.5, "Diarizing on CPU — this takes a while…")
        annotation = pipeline(str(audio_path), **kwargs)
        turns: list[tuple[float, float, str]] = []
        speakers: list[str] = []
        # pyannote Annotation iterates (segment, track, label); the
        # label is already SPEAKER_00-style for this pipeline.
        for segment, _track, label in annotation.itertracks(yield_label=True):
            label = str(label)
            turns.append((float(segment.start), float(segment.end), label))
            if label not in speakers:
                speakers.append(label)
        turns.sort()
        speakers.sort()
        wall = time.monotonic() - started
        if progress_callback is not None:
            progress_callback(1.0, "Diarization complete.")
        result = FullDiarizationResult(
            turns=turns, speakers=speakers, model=self.model,
            wall_seconds=wall)
        logger.info("Diarized %s: %d turns, %d speakers, %.1fs",
                    audio_path.name, len(turns), len(speakers), wall)
        return result
