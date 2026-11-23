"""Multi-speaker diarization via sherpa-onnx — CPU only, no token.

This is the third diarization mode (after ``simple`` and ``full`` /
pyannote). It runs ``sherpa_onnx.OfflineSpeakerDiarization``: the
pyannote segmentation model exported to ONNX, plus 3D-Speaker
ERes2Net speaker embeddings, plus threshold-based clustering.

Why this mode exists: it is the only full diarization path that has
actually run on this VM (Nov 8 2026 tryouts, see TRYOUTS.md) —
pyannote needs a Hugging Face account, accepted licences and a
token, and torch never installed here; sherpa needs none of that.
Its models come from public GitHub releases (no account, no token,
no login) and are downloaded once on first use into
``~/.offair/models/sherpa-diarization/``; after that it works
fully offline.

Models (all CPU-only ONNX):

* Segmentation: ``sherpa-onnx-pyannote-segmentation-3-0.tar.bz2``
  (7.0 MB tarball; ``model.onnx`` 6.0 MB inside) — the pyannote
  segmentation-3.0 weights exported to ONNX by the sherpa-onnx
  project. The weights are pyannote-derived, but the ONNX export
  is hosted on GitHub releases and needs no Hugging Face account.
* Embeddings: ``3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx``
  (26.5 MB) — 3D-Speaker ERes2Net, 192-dim, Apache 2.0.

Measured on the fixture (Nov 8 2026, 61.3 s, 3 TTS voices, 9
turns — see ``docs/CPU_TIMES.md``): 9/9 turns correct, RTF ~0.35,
peak RSS ~262 MB.

Clustering — threshold, not a fixed count: the tryout hardcoded
``num_clusters=3`` because the fixture's speaker count was known.
Production code must not do that, so the default is
``FastClusteringConfig(num_clusters=-1, threshold=0.5)``: clusters
merge while their cosine similarity is above the threshold, and
the speaker count is discovered from the audio. An optional
``expected_speakers`` hint (1–10, 0 = auto) pins the cluster
count instead, for meetings where the count is known.

Threshold choice (documented, measured): on the fixture,
ERes2Net same-speaker cosine is 0.82–0.89 and cross-speaker
cosine is ~0.0–0.02 (margin 0.810 — the best of everything
tested, see TRYOUTS.md). 0.5 sits midway between those bands.
Thresholds 0.5, 0.6 and 0.7 were all run against the fixture and
all produced the same 9 segments, 3 speakers, 9/9 identity, so
the result is not knife-edge sensitive to the exact value; 0.5
is sherpa-onnx's own default and the value shipped here.
Re-measure on real meeting audio before trusting it blindly —
noisy rooms and short turns will narrow the margin.

Lazy import: importing this module never imports sherpa-onnx.
``SherpaDiarizer`` loads it (and the models) on first use, so
importing OffAir, running simple mode, and running the unit
tests all work on a machine without sherpa-onnx installed.

Output shape: :meth:`SherpaDiarizer.diarize` returns a
:class:`SherpaDiarizationResult` whose :meth:`to_dict` has the
same keys as the pyannote full mode (``mode`` / ``method`` /
``model`` / ``turns`` / ``speakers`` / ``wall_seconds``), with
generic ``SPEAKER_00`` labels — renames are the Speaker Name
Panel's job (``app.diarization.names``), exactly as for full
mode. One difference from raw sherpa output: cluster ids are
renumbered to contiguous ``SPEAKER_00``/``01``… in order of
first appearance, because threshold clustering can leave gaps
(e.g. speakers 00, 01, 03 with no 02).
"""
from __future__ import annotations

import logging
import ssl
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.diarization.simple import load_wav_mono

logger = logging.getLogger("offair.diarization.sherpa")

# --- Models ---------------------------------------------------------------
# GitHub release tags (note: sherpa-onnx's own tag really is spelled
# "speaker-recongition-models" — that is their typo, not ours).
SHERPA_RELEASE_BASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download")
SEGMENTATION_RELEASE_TAG = "speaker-segmentation-models"
SEGMENTATION_TARBALL = "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEGMENTATION_MODEL_FILE = "model.onnx"  # inside the extracted tarball dir
SEGMENTATION_TARBALL_BYTES = 6958444
SEGMENTATION_MODEL_BYTES = 5992913
EMBEDDING_RELEASE_TAG = "speaker-recongition-models"
EMBEDDING_MODEL_FILE = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"
EMBEDDING_MODEL_BYTES = 26485263
# 3D-Speaker ERes2Net models are Apache 2.0 (see TRYOUTS.md).
EMBEDDING_MODEL_LICENSE = "Apache-2.0"

SHERPA_DEFAULT_MODEL_DIR: Path = (
    Path.home() / ".offair" / "models" / "sherpa-diarization")

# Clustering defaults — see the module docstring for how 0.5 was picked.
SHERPA_DEFAULT_CLUSTERING_THRESHOLD: float = 0.5
SHERPA_MIN_CLUSTERING_THRESHOLD: float = 0.05
SHERPA_MAX_CLUSTERING_THRESHOLD: float = 0.95
# 0 = auto (threshold clustering); 1–10 pins the cluster count.
SHERPA_DEFAULT_EXPECTED_SPEAKERS: int = 0
SHERPA_MIN_EXPECTED_SPEAKERS: int = 1
SHERPA_MAX_EXPECTED_SPEAKERS: int = 10

SHERPA_SAMPLE_RATE: int = 16000  # both models expect 16 kHz
SHERPA_NUM_THREADS: int = 2
SHERPA_MIN_DURATION_ON: float = 0.3
SHERPA_MIN_DURATION_OFF: float = 0.5

MODE_NAME = "sherpa"
MODE_LABEL = "Sherpa — multi-speaker (no token, CPU) (default)"
MODE_METHOD = ("sherpa-onnx (pyannote segmentation ONNX + "
               "ERes2Net embeddings, threshold clustering)")


class SherpaNotInstalledError(RuntimeError):
    """Raised when sherpa-onnx is not installed."""


class SherpaModelError(RuntimeError):
    """Raised when the sherpa models cannot be obtained.

    The message is always a guided one: what to check (network,
    disk space) and that no account or token is needed — the
    models are public GitHub releases.
    """


def sherpa_install_guide() -> str:
    """How to install sherpa-onnx (shown when it is missing)."""
    return (
        "sherpa-onnx is not installed. It is a small CPU-only install "
        "(no torch, no GPU):\n"
        "    pip install sherpa-onnx\n"
        "The diarization models (~33 MB) download once on first use "
        "from public GitHub releases — no account, no token, no login. "
        "Simple mode keeps working without it."
    )


def model_download_guide() -> str:
    """Guided error text for a failed model download."""
    return (
        "Could not download the sherpa diarization models. They are "
        "public files on GitHub releases — no account, no token and "
        "no login are needed — so this is a network or disk problem, "
        "not a permissions problem:\n"
        "  1. Check the machine is online, then retry diarization.\n"
        "  2. Check there is ~50 MB free under "
        f"{SHERPA_DEFAULT_MODEL_DIR}\n"
        "  3. Or download the two files by hand into that folder:\n"
        f"       {SEGMENTATION_TARBALL}\n"
        f"       {EMBEDDING_MODEL_FILE}\n"
        "     from:\n"
        f"       {SHERPA_RELEASE_BASE}/{SEGMENTATION_RELEASE_TAG}/\n"
        f"       {SHERPA_RELEASE_BASE}/{EMBEDDING_RELEASE_TAG}/\n"
        "  Once the files are in place, diarization works fully offline."
    )


def sherpa_importable() -> bool:
    """Best-effort probe: sherpa_onnx imports cleanly.

    Never raises — availability checks must not crash the app if
    an optional dependency is half-installed. The strict error
    lives in :meth:`SherpaDiarizer._load`.
    """
    try:
        import sherpa_onnx  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means "no"
        return False
    return True


def validate_sherpa_settings(expected_speakers: int | None,
                             clustering_threshold: float | None
                             ) -> tuple[int, float]:
    """Validate the sherpa clustering settings.

    ``expected_speakers``: 0 or None = auto (threshold clustering);
    otherwise an integer 1–10. ``clustering_threshold``: a float
    strictly between 0 and 1 (the cosine-similarity merge
    threshold). Returns ``(expected_speakers, threshold)`` with
    defaults filled in. Raises ValueError on a bad value.
    """
    if expected_speakers is None:
        expected_speakers = SHERPA_DEFAULT_EXPECTED_SPEAKERS
    if isinstance(expected_speakers, bool) \
            or not isinstance(expected_speakers, int):
        raise ValueError(
            f"expected_speakers must be an integer, got "
            f"{expected_speakers!r}")
    if expected_speakers != 0 and not (
            SHERPA_MIN_EXPECTED_SPEAKERS <= expected_speakers
            <= SHERPA_MAX_EXPECTED_SPEAKERS):
        raise ValueError(
            f"expected_speakers must be 0 (auto) or between "
            f"{SHERPA_MIN_EXPECTED_SPEAKERS} and "
            f"{SHERPA_MAX_EXPECTED_SPEAKERS}, got {expected_speakers}")
    if clustering_threshold is None:
        clustering_threshold = SHERPA_DEFAULT_CLUSTERING_THRESHOLD
    if isinstance(clustering_threshold, bool) or not isinstance(
            clustering_threshold, (int, float)):
        raise ValueError(
            f"clustering_threshold must be a number, got "
            f"{clustering_threshold!r}")
    clustering_threshold = float(clustering_threshold)
    if not (SHERPA_MIN_CLUSTERING_THRESHOLD <= clustering_threshold
            <= SHERPA_MAX_CLUSTERING_THRESHOLD):
        raise ValueError(
            f"clustering_threshold must be between "
            f"{SHERPA_MIN_CLUSTERING_THRESHOLD} and "
            f"{SHERPA_MAX_CLUSTERING_THRESHOLD}, got "
            f"{clustering_threshold}")
    return expected_speakers, clustering_threshold


def _ssl_context() -> ssl.SSLContext:
    """SSL context for model downloads.

    Uses certifi's CA bundle when it is installed (it is a transitive
    dependency via huggingface_hub/requests, and PyInstaller collects
    it into the frozen app). This matters on machines whose system
    cert store is stale or incomplete — e.g. a bare Windows eval
    install, where urllib's default verification fails for
    github.com's chain with CERTIFICATE_VERIFY_FAILED ("unable to get
    local issuer certificate") while the certifi bundle verifies
    fine (verified Nov 9 2026 on the Windows VM). Falls back to the
    default context when certifi is absent.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 — fall back to the default context
        return ssl.create_default_context()


def _download(url: str, dest: Path, expected_bytes: int | None = None,
              progress_callback=None) -> None:
    """Download ``url`` to ``dest`` (atomic-ish: temp file + rename).

    Raises :class:`SherpaModelError` with the guided message on
    any network failure, and on a size mismatch when the expected
    size is known.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=_ssl_context()))
        with opener.open(url) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress_callback is not None and total > 0:
                    progress_callback(
                        min(1.0, done / total),
                        f"Downloading {dest.name}…")
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise SherpaModelError(
            f"Failed to download {url}: {exc}\n\n"
            + model_download_guide()) from exc
    if expected_bytes is not None and tmp.stat().st_size != expected_bytes:
        tmp.unlink(missing_ok=True)
        raise SherpaModelError(
            f"Downloaded {dest.name} has an unexpected size "
            f"(got {tmp.stat().st_size if tmp.exists() else '?'} bytes, "
            f"expected {expected_bytes}) — the download may be "
            f"incomplete or the release may have changed.\n\n"
            + model_download_guide())
    tmp.replace(dest)


def ensure_sherpa_models(model_dir: str | Path | None = None,
                         progress_callback=None
                         ) -> tuple[Path, Path]:
    """Return ``(segmentation_model_path, embedding_model_path)``.

    Downloads any missing model into ``model_dir`` (default
    ``~/.offair/models/sherpa-diarization/``) from the public
    GitHub releases — no token, no login. A file already present
    at the expected size is reused and never re-downloaded. The
    segmentation tarball is extracted in place; the extracted
    ``model.onnx`` is what the pipeline loads.

    Raises :class:`SherpaModelError` (guided message) when a
    download fails — e.g. the machine is offline on first use.
    """
    model_dir = Path(model_dir) if model_dir is not None \
        else SHERPA_DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = model_dir / "sherpa-onnx-pyannote-segmentation-3-0"
    seg_model = seg_dir / SEGMENTATION_MODEL_FILE
    emb_model = model_dir / EMBEDDING_MODEL_FILE

    if not emb_model.exists() \
            or emb_model.stat().st_size != EMBEDDING_MODEL_BYTES:
        logger.info("Downloading sherpa ERes2Net model (%s MB) …",
                    EMBEDDING_MODEL_BYTES // 1_000_000)
        _download(
            f"{SHERPA_RELEASE_BASE}/{EMBEDDING_RELEASE_TAG}/"
            f"{EMBEDDING_MODEL_FILE}",
            emb_model, expected_bytes=EMBEDDING_MODEL_BYTES,
            progress_callback=progress_callback)
    if not seg_model.exists() \
            or seg_model.stat().st_size != SEGMENTATION_MODEL_BYTES:
        tarball = model_dir / SEGMENTATION_TARBALL
        if not tarball.exists() \
                or tarball.stat().st_size != SEGMENTATION_TARBALL_BYTES:
            logger.info("Downloading sherpa segmentation model (%s MB) …",
                        SEGMENTATION_TARBALL_BYTES // 1_000_000)
            _download(
                f"{SHERPA_RELEASE_BASE}/{SEGMENTATION_RELEASE_TAG}/"
                f"{SEGMENTATION_TARBALL}",
                tarball, expected_bytes=SEGMENTATION_TARBALL_BYTES,
                progress_callback=progress_callback)
        logger.info("Extracting segmentation model …")
        try:
            with tarfile.open(tarball, "r:bz2") as tf:
                # filter="data": only regular files/dirs, no
                # absolute paths or device nodes (also silences
                # the Python 3.14 extraction-filter warning).
                tf.extractall(model_dir, filter="data")
        except (tarfile.TarError, OSError) as exc:
            raise SherpaModelError(
                f"Could not extract {tarball}: {exc}\n\n"
                + model_download_guide()) from exc
        if not seg_model.exists():
            raise SherpaModelError(
                f"Downloaded {SEGMENTATION_TARBALL} but it did not "
                f"contain {SEGMENTATION_MODEL_FILE} where expected.\n\n"
                + model_download_guide())
    return seg_model, emb_model


def check_availability(model_dir: str | Path | None = None) -> dict:
    """Availability info for the mode picker (UI calls this).

    Returns ``{"installed": bool, "models_present": bool,
    "model_dir": str, "label": str, "ready": bool}``. ``ready``
    means diarization can run right now (installed + models
    present); otherwise the first run downloads the models.
    Never raises and never downloads anything.
    """
    model_dir = Path(model_dir) if model_dir is not None \
        else SHERPA_DEFAULT_MODEL_DIR
    seg = (model_dir / "sherpa-onnx-pyannote-segmentation-3-0"
           / SEGMENTATION_MODEL_FILE)
    emb = model_dir / EMBEDDING_MODEL_FILE
    models_present = (
        seg.exists() and seg.stat().st_size == SEGMENTATION_MODEL_BYTES
        and emb.exists() and emb.stat().st_size == EMBEDDING_MODEL_BYTES)
    installed = sherpa_importable()
    return {"installed": installed, "models_present": models_present,
            "model_dir": str(model_dir), "label": MODE_LABEL,
            "ready": installed and models_present}


def _resample_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample mono audio to 16 kHz (linear interpolation).

    The sherpa segmentation + embedding models expect 16 kHz.
    Same crude-but-adequate approach as the ECAPA extractors
    (``app.speakers.embeddings._to_16k``).
    """
    if sample_rate == SHERPA_SAMPLE_RATE:
        return samples.astype(np.float32)
    n_out = int(len(samples) * SHERPA_SAMPLE_RATE / sample_rate)
    idx = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(idx, np.arange(len(samples)),
                     samples.astype(np.float64)).astype(np.float32)


def _renumber_speakers(turns: list[tuple[float, float, int]]
                       ) -> list[tuple[float, float, str]]:
    """Map raw cluster ids to contiguous ``SPEAKER_00`` labels.

    Threshold clustering can leave gaps in the cluster numbering
    (e.g. speakers 00, 01, 03 with no 02); the labels handed to
    the transcript must be contiguous, in order of first
    appearance — the same convention as the pyannote full mode.
    """
    mapping: dict[int, str] = {}
    renumbered: list[tuple[float, float, str]] = []
    for start, end, cluster in turns:
        if cluster not in mapping:
            mapping[cluster] = f"SPEAKER_{len(mapping):02d}"
        renumbered.append((start, end, mapping[cluster]))
    return renumbered


@dataclass
class SherpaDiarizationResult:
    """What a sherpa diarization run produced."""

    # (start, end, "SPEAKER_00") turns, sorted by start time.
    turns: list[tuple[float, float, str]] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    model: str = EMBEDDING_MODEL_FILE
    segmentation_model: str = SEGMENTATION_TARBALL
    # "threshold" (auto) or "expected-count" (hint was given).
    clustering: str = "threshold"
    clustering_threshold: float = SHERPA_DEFAULT_CLUSTERING_THRESHOLD
    expected_speakers: int = SHERPA_DEFAULT_EXPECTED_SPEAKERS
    audio_duration_seconds: float | None = None
    wall_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {"mode": MODE_NAME, "method": MODE_METHOD,
                "model": self.model,
                "segmentation_model": self.segmentation_model,
                "clustering": self.clustering,
                "clustering_threshold": self.clustering_threshold,
                "expected_speakers": self.expected_speakers,
                "audio_duration_seconds": self.audio_duration_seconds,
                "wall_seconds": round(self.wall_seconds, 2),
                "speakers": self.speakers,
                "turns": [{"start": round(s, 3), "end": round(e, 3),
                           "speaker": sp} for s, e, sp in self.turns]}


class SherpaDiarizer:
    """sherpa-onnx offline speaker diarization, CPU only.

    ``diarizer`` is injectable (tests pass a fake with the same
    ``process(samples)`` interface returning objects with
    ``.start`` / ``.end`` / ``.speaker``) so the class is testable
    without sherpa-onnx or the models. With a real pipeline,
    :meth:`diarize` runs it on CPU and returns generic
    ``SPEAKER_00``/``SPEAKER_01``… labels — no names are ever
    inferred; renaming is the Speaker Name Panel's job
    (``app.diarization.names``).
    """

    def __init__(self, model_dir: str | Path | None = None, *,
                 expected_speakers: int = SHERPA_DEFAULT_EXPECTED_SPEAKERS,
                 clustering_threshold: float
                 = SHERPA_DEFAULT_CLUSTERING_THRESHOLD,
                 num_threads: int = SHERPA_NUM_THREADS,
                 diarizer=None) -> None:
        self.model_dir = (Path(model_dir) if model_dir is not None
                          else SHERPA_DEFAULT_MODEL_DIR)
        (self.expected_speakers,
         self.clustering_threshold) = validate_sherpa_settings(
             expected_speakers, clustering_threshold)
        self.num_threads = num_threads
        self._diarizer = diarizer  # injected (tests) or None -> lazy-load

    def _load_pipeline(self):
        if self._diarizer is not None:
            return self._diarizer
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise SherpaNotInstalledError(sherpa_install_guide()) from exc
        seg_model, emb_model = ensure_sherpa_models(self.model_dir)
        if self.expected_speakers:
            clustering = sherpa_onnx.FastClusteringConfig(
                num_clusters=self.expected_speakers)
        else:
            clustering = sherpa_onnx.FastClusteringConfig(
                num_clusters=-1, threshold=self.clustering_threshold)
        logger.info("Building sherpa diarization pipeline "
                    "(threshold=%.2f, expected_speakers=%d) …",
                    self.clustering_threshold, self.expected_speakers)
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(seg_model)),
                num_threads=self.num_threads),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb_model), num_threads=self.num_threads,
                debug=0),
            clustering=clustering,
            min_duration_on=SHERPA_MIN_DURATION_ON,
            min_duration_off=SHERPA_MIN_DURATION_OFF)
        if not config.validate():
            raise SherpaModelError(
                "sherpa-onnx rejected the diarization config — "
                "this is a bug, not a user error; please report it.")
        self._diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
        return self._diarizer

    def diarize(self, audio_path: str | Path, *,
                progress_callback=None) -> SherpaDiarizationResult:
        """Diarize one audio file. Returns speaker turns (start/end/label).

        The audio is resampled to 16 kHz (the models' rate) and
        passed to the pipeline as float32 samples. ``progress_callback``
        (if given) is called as ``(fraction, text)`` — the pipeline
        itself is mostly one long call, so the callback reports
        start (0.0), running (0.5) and done (1.0); it is a progress
        *indicator*, not a precise meter, and the UI says so.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        samples, rate = load_wav_mono(audio_path)
        audio16k = _resample_16k(samples, rate)
        duration = len(audio16k) / SHERPA_SAMPLE_RATE
        diarizer = self._load_pipeline()
        if progress_callback is not None:
            progress_callback(0.0, "Loading sherpa diarization models…")
        started = time.monotonic()
        if progress_callback is not None:
            progress_callback(0.5, "Diarizing on CPU — this takes a while…")
        segments = diarizer.process(audio16k).sort_by_start_time()
        wall = time.monotonic() - started
        raw_turns = [(float(s.start), float(s.end), int(s.speaker))
                     for s in segments]
        raw_turns.sort()
        turns = _renumber_speakers(raw_turns)
        speakers = sorted({label for _s, _e, label in turns})
        if progress_callback is not None:
            progress_callback(1.0, "Diarization complete.")
        result = SherpaDiarizationResult(
            turns=turns, speakers=speakers,
            clustering=("expected-count" if self.expected_speakers
                        else "threshold"),
            clustering_threshold=self.clustering_threshold,
            expected_speakers=self.expected_speakers,
            audio_duration_seconds=round(duration, 2),
            wall_seconds=wall)
        logger.info("Diarized %s (sherpa): %d turns, %d speakers, %.1fs",
                    audio_path.name, len(turns), len(speakers), wall)
        return result
