"""Speaker embedding extraction — PRD Phase 4, ECAPA swap (Nov 8 2026).

Four backends, one interface. Every backend returns an L2-normalised
float32 vector, so cosine similarity is a dot product. Embeddings
from different backends live in different vector spaces and must
never be mixed — the backend name is stored with every embedding
(see :mod:`app.speakers.database`).

* :class:`SpeechBrainEcapaEmbeddingExtractor` — production backend
  (``"speechbrain-ecapa-v1"``, 192-dim). SpeechBrain ECAPA-TDNN
  ``speechbrain/spkrec-ecapa-voxceleb`` on CPU. Public weights,
  Apache 2.0, no Hugging Face token / licence click-through.
  Needs torch + speechbrain installed; lazy import, so importing
  this module never requires torch.
* :class:`OnnxEcapaEmbeddingExtractor` — production backend
  (``"onnx-ecapa-v1"``, 192-dim). WeSpeaker ECAPA-TDNN-512 run
  through ONNX Runtime via ``speakeronnx`` (model
  ``wespeaker-ecapa512``, source
  ``Wespeaker/wespeaker-ecapa-tdnn512-LM``, CC-BY-4.0, public,
  no token). Needs only onnxruntime + numpy + soundfile — no
  torch at all. **This is the only production backend verified
  working on this VM** (Nov 8 2026: model loads in ~1.7 s,
  ~0.2–0.3 s per ~9 s file, peak RSS ~169 MB; John–Marcus 0.064,
  John–Priya 0.120, Priya–Marcus 0.046; half-vs-half self
  0.88 / 0.81 / 0.88 — clean separation, unlike the fallback).
* :class:`SherpaEres2NetEmbeddingExtractor` — production backend
  (``"sherpa-eres2net-v1"``, 192-dim). sherpa-onnx 3D-Speaker
  ERes2Net (``3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx``,
  Apache 2.0, public GitHub release — no token). Best separation
  of everything tested (Nov 8 2026: margin 0.810, cross-speaker
  ~0.0; see TRYOUTS.md). Same models as sherpa diarization
  (``app.diarization.sherpa``), downloaded once on first use.
  Opt-in via ``get_extractor("sherpa-eres2net")``; not in the
  ``"auto"`` chain (that chain is unchanged).
* :class:`PyannoteEmbeddingExtractor` — opt-in backend
  (``"pyannote-embedding-v1"``, 512-dim). The ``pyannote/embedding``
  model on CPU via :class:`pyannote.audio.Inference`. Needs torch
  + pyannote.audio and a Hugging Face token (gated model). It was
  the production backend before the ECAPA swap; it is kept for
  users who already have the pyannote setup, but it is no longer
  the default. Never ran on this VM (torch does not install here).
* :class:`FallbackEmbeddingExtractor` — test-only backend
  (``"fallback-spectral-v1"``, 38-dim). Features: mean/std of
  MFCCs plus a log-pitch histogram and pitch statistics. It is a
  **spectral proxy, not a voice fingerprint** — see the package
  docstring and PHASE4.md for the measured limits (it confuses
  two similar male voices). Use it for development / testing on
  machines where neither ECAPA backend installs.

:func:`get_default_extractor` returns the best available
production backend — SpeechBrain ECAPA when torch + speechbrain
import, otherwise ONNX ECAPA when speakeronnx imports, otherwise
the fallback — and always says which one it returned, so a caller
can never mistake one for the other. :func:`get_extractor` pins
one backend by name (``"auto"`` = the default chain); that is how
the opt-in pyannote backend is selected.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.diarization.simple import load_wav_mono

logger = logging.getLogger("offair.speakers.embeddings")

# Fallback extractor: at most this many seconds of audio per speaker
# go into one embedding (windows are taken in order until the cap).
FALLBACK_MAX_SECONDS = 30.0
FALLBACK_N_MELS = 16
FALLBACK_DIM = 38  # 12 MFCC means + 12 MFCC stds + 12 pitch-hist bins + 2 pitch stats
PYANNOTE_EMBEDDING_MODEL = "pyannote/embedding"
PYANNOTE_EMBEDDING_RATE = 16000
PYANNOTE_EMBEDDING_CHUNK_SECONDS = 5.0

# ECAPA backends (both 192-dim, both 16 kHz in).
ECAPA_DIM = 192
ECAPA_SAMPLE_RATE = 16000
# At most this many seconds of audio per speaker go into one ECAPA
# embedding (windows are taken in order until the cap); chunked
# inside the extractor so long turns do not blow up memory.
ECAPA_MAX_SECONDS = 60.0
ECAPA_CHUNK_SECONDS = 10.0
# ONNX ECAPA: WeSpeaker ECAPA-TDNN-512 via speakeronnx. Public HF
# repo, CC-BY-4.0, no token / login. The model file is downloaded
# to the HF cache on first use (~20 MB).
ONNX_ECAPA_MODEL = "wespeaker-ecapa512"
ONNX_ECAPA_SOURCE = "Wespeaker/wespeaker-ecapa-tdnn512-LM"
# SpeechBrain ECAPA: public weights, Apache 2.0, no token / login.
# The model is downloaded to the SpeechBrain cache on first use.
SPEECHBRAIN_ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
# Sherpa ERes2Net: 3D-Speaker ERes2Net, 192-dim, Apache 2.0. Same
# model file as sherpa diarization — see app.diarization.sherpa
# for the download URL and the measured separation (margin 0.810,
# cross-speaker ~0.0 on the fixture, best of everything tested).
SHERPA_ERES2NET_MODEL_FILE = (
    "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx")


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_fft: int, sample_rate: int, n_mels: int = FALLBACK_N_MELS,
                    fmin: float = 80.0, fmax: float = 8000.0) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    mel_points = np.linspace(_hz_to_mel(np.array([fmin]))[0],
                             _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bank = np.zeros((n_mels, len(freqs)))
    for m in range(1, n_mels + 1):
        f0, f1, f2 = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f0, f1):
            bank[m - 1, k] = (k - f0) / (f1 - f0) if f1 > f0 else 0.0
        for k in range(f1, f2):
            bank[m - 1, k] = (f2 - k) / (f2 - f1) if f2 > f1 else 0.0
    return bank


def _mfcc_stats(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mean/std of MFCCs 1..12 (content- and timbre-sensitive)."""
    n_fft, hop = 1024, 512
    window = np.hanning(n_fft)
    frames = []
    for i in range(0, max(1, len(samples) - n_fft), hop):
        frames.append(np.abs(np.fft.rfft(samples[i:i + n_fft] * window,
                                         n=n_fft)))
    spec = np.asarray(frames, dtype=np.float64)
    if spec.size == 0:
        return np.zeros(24)
    bank = _mel_filterbank(n_fft, sample_rate)
    log_mel = np.log10(spec @ bank.T + 1e-6)
    n_mels = log_mel.shape[1]
    mfcc = np.zeros((len(log_mel), 13))
    for k in range(13):
        mfcc[:, k] = np.sum(
            log_mel * np.cos(np.pi / n_mels
                             * (np.arange(n_mels) + 0.5) * k), axis=1)
    return np.concatenate([mfcc[:, 1:].mean(axis=0) / 10.0,
                           mfcc[:, 1:].std(axis=0) / 5.0])


def _pitch_histogram(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Log-pitch histogram (12 bins, 60–400 Hz) + median/std pitch.

    Pitch per frame by autocorrelation; frames with low energy or a
    weak autocorrelation peak are skipped (unvoiced / silence).
    """
    f0s: list[float] = []
    for i in range(0, max(0, len(samples) - 2048), 512):
        frame = samples[i:i + 2048].astype(np.float64)
        frame = frame - frame.mean()
        if np.sqrt((frame ** 2).mean()) < 0.01:
            continue
        autocorr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        if autocorr[0] <= 0:
            continue
        autocorr = autocorr / autocorr[0]
        lo, hi = int(sample_rate / 400), int(sample_rate / 60)
        segment = autocorr[lo:hi]
        if len(segment) == 0 or segment.max() < 0.3:
            continue
        f0s.append(sample_rate / (lo + int(np.argmax(segment))))
    f0 = np.asarray(f0s, dtype=np.float64)
    bins = np.linspace(np.log(60.0), np.log(400.0), 13)
    hist, _ = np.histogram(np.log(np.clip(f0, 60.0, 400.0)), bins=bins)
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-9)
    median = float(np.median(f0)) if f0.size else 0.0
    std = float(np.std(f0)) if f0.size > 1 else 0.0
    vec = np.concatenate([hist * 3.0, [median / 200.0, std / 80.0]])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _select_windows(samples: np.ndarray, sample_rate: int,
                    windows: list[tuple[float, float]] | None,
                    max_seconds: float = FALLBACK_MAX_SECONDS) -> np.ndarray:
    """Concatenate the requested time windows, capped at ``max_seconds``."""
    if not windows:
        total = int(max_seconds * sample_rate)
        return samples[:total]
    parts: list[np.ndarray] = []
    budget = int(max_seconds * sample_rate)
    for start, end in windows:
        if budget <= 0:
            break
        i0 = max(0, int(start * sample_rate))
        i1 = min(len(samples), int(end * sample_rate))
        if i1 <= i0:
            continue
        chunk = samples[i0:i1][:budget]
        parts.append(chunk)
        budget -= len(chunk)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def _to_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample mono audio to 16 kHz (linear interpolation).

    The ECAPA models expect 16 kHz. Linear interpolation is
    crude but adequate for an embedding frontend — the same
    approach the pyannote extractor already uses.
    """
    if sample_rate == ECAPA_SAMPLE_RATE:
        return samples.astype(np.float32)
    n_out = int(len(samples) * ECAPA_SAMPLE_RATE / sample_rate)
    idx = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(idx, np.arange(len(samples)),
                     samples.astype(np.float64)).astype(np.float32)


def _normalise(vec: np.ndarray) -> np.ndarray:
    """L2-normalise to float32; raises ValueError on an empty vector."""
    vec = np.asarray(vec, dtype=np.float64).ravel()
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Empty embedding — audio has no usable signal")
    return (vec / norm).astype(np.float32)


class OnnxEcapaEmbeddingExtractor:
    """Production embedding backend — WeSpeaker ECAPA via ONNX.

    ``speakeronnx`` (ONNX Runtime, CPU) with the
    ``wespeaker-ecapa512`` model: ECAPA-TDNN-512 trained on
    VoxCeleb2 Dev, 192-dim embeddings. Model source
    ``Wespeaker/wespeaker-ecapa-tdnn512-LM`` (CC-BY-4.0, public
    Hugging Face repo — no token, no login, ~20 MB download to
    the HF cache on first use). Needs only onnxruntime + numpy +
    soundfile: no torch, no GPU. This is the only production
    backend verified working on this VM (Nov 8 2026 — see the
    module docstring for the measured numbers).

    Lazy import: constructing this class never imports
    speakeronnx; :meth:`extract` loads the model on first use.
    """

    backend = "onnx-ecapa-v1"
    dim = ECAPA_DIM

    def __init__(self, model: str = ONNX_ECAPA_MODEL) -> None:
        # ``model`` is a speakeronnx registry alias (default) or an
        # absolute path to a .onnx file.
        self.model = model
        self._embedder = None

    def _load(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from speakeronnx import SpeakerEmbedder
        except ImportError as exc:
            raise RuntimeError(
                "speakeronnx is not installed — install it with "
                "`pip install -r requirements.txt` (onnxruntime + "
                "speakeronnx, no torch needed). The fallback extractor "
                "is used until then."
            ) from exc
        logger.info("Loading ONNX ECAPA model %s (source %s) …",
                    self.model, ONNX_ECAPA_SOURCE)
        self._embedder = SpeakerEmbedder(model=self.model)
        return self._embedder

    def extract(self, wav_path: str | Path,
                windows: list[tuple[float, float]] | None = None
                ) -> np.ndarray:
        """Embedding for ``wav_path`` (optionally only ``windows``)."""
        samples, rate = load_wav_mono(wav_path)
        return self.extract_samples(samples, rate, windows)

    def extract_samples(self, samples: np.ndarray, sample_rate: int,
                        windows: list[tuple[float, float]] | None = None
                        ) -> np.ndarray:
        embedder = self._load()
        selected = _select_windows(samples, sample_rate, windows,
                                   max_seconds=ECAPA_MAX_SECONDS)
        if selected.size < int(0.5 * sample_rate):
            raise ValueError(
                "Need at least 0.5 s of audio to extract an embedding; "
                f"got {selected.size / max(sample_rate, 1):.2f} s")
        audio16k = _to_16k(selected, sample_rate)
        # speakeronnx accepts a float32 array at the model rate and
        # returns an already L2-normalised embedding; renormalise
        # defensively so the contract holds whatever it returns.
        return _normalise(embedder.embed(audio16k.astype(np.float32)))


class SpeechBrainEcapaEmbeddingExtractor:
    """Production embedding backend — SpeechBrain ECAPA-TDNN.

    ``speechbrain/spkrec-ecapa-voxceleb`` (ECAPA-TDNN, 192-dim,
    Apache 2.0, public weights — no Hugging Face token / licence
    click-through). Needs torch + speechbrain installed; the model
    is downloaded to the SpeechBrain cache on first use. This is
    the backend John chose (Nov 8 2026) as the pyannote
    replacement, but it has NOT run on this VM: torch does not
    install here (PyPI stalls at 261.6 MB into the 539 MB wheel
    and ~600 MB free RAM would likely OOM). First real run is on
    the target laptop — see PHASE4.md. Until then the ONNX ECAPA
    backend is the working production default.

    Lazy import: constructing this class never imports torch or
    speechbrain; :meth:`extract` loads the model on first use.
    Audio is chunked into 10 s pieces and the chunk embeddings
    are averaged, so long turns do not blow up memory.
    """

    backend = "speechbrain-ecapa-v1"
    dim = ECAPA_DIM

    def __init__(self,
                 model: str = SPEECHBRAIN_ECAPA_MODEL) -> None:
        self.model = model
        self._classifier = None
        self._torch = None

    def _load(self):
        if self._classifier is not None:
            return self._classifier
        try:
            import torch
            from speechbrain.inference import EncoderClassifier
        except ImportError as exc:
            raise RuntimeError(
                "speechbrain (and torch, CPU) is not installed — "
                "install it with `pip install -r "
                "requirements-speechbrain.txt`. The ONNX ECAPA backend "
                "is the working default until then."
            ) from exc
        logger.info("Loading SpeechBrain ECAPA model %s on CPU …",
                    self.model)
        self._torch = torch
        self._classifier = EncoderClassifier.from_hparams(
            source=self.model, run_opts={"device": "cpu"})
        return self._classifier

    def extract(self, wav_path: str | Path,
                windows: list[tuple[float, float]] | None = None
                ) -> np.ndarray:
        """Embedding for ``wav_path`` (optionally only ``windows``)."""
        samples, rate = load_wav_mono(wav_path)
        return self.extract_samples(samples, rate, windows)

    def extract_samples(self, samples: np.ndarray, sample_rate: int,
                        windows: list[tuple[float, float]] | None = None
                        ) -> np.ndarray:
        classifier = self._load()
        selected = _select_windows(samples, sample_rate, windows,
                                   max_seconds=ECAPA_MAX_SECONDS)
        if selected.size < int(0.5 * sample_rate):
            raise ValueError(
                "Need at least 0.5 s of audio to extract an embedding; "
                f"got {selected.size / max(sample_rate, 1):.2f} s")
        audio16k = _to_16k(selected, sample_rate)
        chunk = int(ECAPA_CHUNK_SECONDS * ECAPA_SAMPLE_RATE)
        vectors: list[np.ndarray] = []
        for i in range(0, max(1, len(audio16k)), chunk):
            piece = audio16k[i:i + chunk]
            if len(piece) < int(0.5 * ECAPA_SAMPLE_RATE):
                continue
            waveform = self._torch.from_numpy(
                piece.astype(np.float32)).unsqueeze(0)
            # encode_batch returns [batch, 1, 192]; np.asarray works
            # on a CPU torch tensor.
            encoding = classifier.encode_batch(waveform)
            vectors.append(np.asarray(encoding, dtype=np.float64).ravel())
        if not vectors:
            raise ValueError("No embedding could be extracted from the audio")
        return _normalise(np.mean(vectors, axis=0))


class SherpaEres2NetEmbeddingExtractor:
    """Production embedding backend — sherpa-onnx 3D-Speaker ERes2Net.

    ``3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx`` (192-dim,
    Apache 2.0, public GitHub release — no Hugging Face token /
    licence click-through). Best separation of everything tested
    (Nov 8 2026 tryouts: margin 0.810, cross-speaker cosine ~0.0,
    same-speaker half-vs-half 0.82–0.89; see TRYOUTS.md). Needs
    sherpa-onnx installed; the model is the same file sherpa
    diarization uses, downloaded once on first use into
    ``~/.offair/models/sherpa-diarization/``.

    Lazy import: constructing this class never imports sherpa-onnx;
    :meth:`extract` loads the model on first use. Audio is chunked
    into 10 s pieces and the chunk embeddings are averaged, so long
    turns do not blow up memory.
    """

    backend = "sherpa-eres2net-v1"
    dim = ECAPA_DIM

    def __init__(self, model_path: str | Path | None = None) -> None:
        # Explicit .onnx path, or None = the shared sherpa model dir
        # (downloaded on first use if missing).
        self.model_path = Path(model_path) if model_path else None
        self._extractor = None

    def _load(self):
        if self._extractor is not None:
            return self._extractor
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "sherpa-onnx is not installed — install it with "
                "`pip install sherpa-onnx` (small CPU-only install, "
                "no torch). The ONNX ECAPA backend is the working "
                "default until then."
            ) from exc
        if self.model_path is not None:
            model = self.model_path
        else:
            from app.diarization.sherpa import ensure_sherpa_models
            _seg, model = ensure_sherpa_models()
        logger.info("Loading sherpa ERes2Net model %s …", model)
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model), num_threads=2, debug=0)
        if not config.validate():
            raise RuntimeError(
                "sherpa-onnx rejected the ERes2Net extractor config — "
                "this is a bug, not a user error; please report it.")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        return self._extractor

    def extract(self, wav_path: str | Path,
                windows: list[tuple[float, float]] | None = None
                ) -> np.ndarray:
        """Embedding for ``wav_path`` (optionally only ``windows``)."""
        samples, rate = load_wav_mono(wav_path)
        return self.extract_samples(samples, rate, windows)

    def extract_samples(self, samples: np.ndarray, sample_rate: int,
                        windows: list[tuple[float, float]] | None = None
                        ) -> np.ndarray:
        extractor = self._load()
        selected = _select_windows(samples, sample_rate, windows,
                                   max_seconds=ECAPA_MAX_SECONDS)
        if selected.size < int(0.5 * sample_rate):
            raise ValueError(
                "Need at least 0.5 s of audio to extract an embedding; "
                f"got {selected.size / max(sample_rate, 1):.2f} s")
        audio16k = _to_16k(selected, sample_rate)
        chunk = int(ECAPA_CHUNK_SECONDS * ECAPA_SAMPLE_RATE)
        vectors: list[np.ndarray] = []
        for i in range(0, max(1, len(audio16k)), chunk):
            piece = audio16k[i:i + chunk]
            if len(piece) < int(0.5 * ECAPA_SAMPLE_RATE):
                continue
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=ECAPA_SAMPLE_RATE,
                                   waveform=piece.astype(np.float32))
            stream.input_finished()
            vectors.append(np.asarray(extractor.compute(stream),
                                      dtype=np.float64).ravel())
        if not vectors:
            raise ValueError("No embedding could be extracted from the audio")
        return _normalise(np.mean(vectors, axis=0))


class FallbackEmbeddingExtractor:
    """Lightweight spectral embedding — NOT a production voice fingerprint.

    ``backend`` is ``"fallback-spectral-v1"`` and is stored with every
    embedding produced, so a database built with the fallback can
    never be mistaken for one built with pyannote embeddings (the
    two live in different vector spaces and must not be mixed — see
    :func:`app.speakers.matching.match_embedding`).
    """

    backend = "fallback-spectral-v1"
    dim = FALLBACK_DIM

    def extract(self, wav_path: str | Path,
                windows: list[tuple[float, float]] | None = None
                ) -> np.ndarray:
        """Embedding for ``wav_path`` (optionally only ``windows``).

        Raises FileNotFoundError / ValueError for missing or
        unreadable audio, ValueError if the selected audio is empty.
        """
        samples, rate = load_wav_mono(wav_path)
        return self.extract_samples(samples, rate, windows)

    def extract_samples(self, samples: np.ndarray, sample_rate: int,
                        windows: list[tuple[float, float]] | None = None
                        ) -> np.ndarray:
        selected = _select_windows(samples, sample_rate, windows)
        if selected.size < int(0.5 * sample_rate):
            raise ValueError(
                "Need at least 0.5 s of audio to extract an embedding; "
                f"got {selected.size / max(sample_rate, 1):.2f} s")
        # MFCCs are computed at the file rate; pitch needs 16 kHz-ish
        # resolution, so resample by simple decimation only when the
        # rate is an exact multiple of 16 kHz, else use the raw rate
        # (autocorrelation pitch works at any rate).
        mfcc = _mfcc_stats(selected.astype(np.float64), sample_rate)
        pitch = _pitch_histogram(selected.astype(np.float64), sample_rate)
        # Pitch block weight 1.0: measured best on the fixture
        # (w=1.0: Priya separates cleanly; w=2/4 hurt John's
        # same-speaker stability). See PHASE4.md.
        vec = np.concatenate([mfcc, pitch * 1.0]).astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Empty embedding — audio has no usable signal")
        return (vec / norm).astype(np.float32)


class PyannoteEmbeddingExtractor:
    """Opt-in embedding backend — pyannote/embedding on CPU.

    Was the production backend before the Nov 8 2026 ECAPA swap;
    kept for users who already have the pyannote setup (torch +
    pyannote.audio + Hugging Face token for the gated model). It
    is NOT in the default chain any more — select it explicitly
    with ``get_extractor("pyannote")`` or the
    ``embedding_backend`` setting.

    Lazy import: constructing this class never imports torch or
    pyannote; :meth:`extract` loads the model on first use. Raises
    :class:`RuntimeError` with a clear message if pyannote/torch is
    missing, or if no Hugging Face token / licence lets the model
    download. Audio is chunked into 5 s pieces and the chunk
    embeddings are averaged, so long turns do not blow up memory.
    """

    backend = "pyannote-embedding-v1"
    dim = 512  # pyannote/embedding output dimension

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self._inference = None

    def _load(self):
        if self._inference is not None:
            return self._inference
        try:
            import torch
            from pyannote.audio import Inference
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio (and torch, CPU) is not installed — "
                "install it with `pip install -r requirements.txt`. "
                "The fallback extractor is used until then."
            ) from exc
        logger.info("Loading embedding model %s on CPU …",
                    PYANNOTE_EMBEDDING_MODEL)
        try:
            self._inference = Inference(
                PYANNOTE_EMBEDDING_MODEL, device=torch.device("cpu"))
        except Exception as exc:
            message = str(exc)
            if "401" in message or "gated" in message.lower() \
                    or "licen" in message.lower():
                raise RuntimeError(
                    "Hugging Face refused access to "
                    f"{PYANNOTE_EMBEDDING_MODEL}. This usually means the "
                    "token is wrong or the model licence has not been "
                    "accepted — see README, Phase 3, for the setup steps."
                ) from exc
            raise
        return self._inference

    def extract(self, wav_path: str | Path,
                windows: list[tuple[float, float]] | None = None
                ) -> np.ndarray:
        import torch

        inference = self._load()
        samples, rate = load_wav_mono(wav_path)
        selected = _select_windows(samples, rate, windows,
                                   max_seconds=60.0)
        if selected.size < int(0.5 * rate):
            raise ValueError("Need at least 0.5 s of audio for an embedding")
        # Resample to 16 kHz by linear interpolation (pyannote rate).
        if rate != PYANNOTE_EMBEDDING_RATE:
            n_out = int(len(selected) * PYANNOTE_EMBEDDING_RATE / rate)
            idx = np.linspace(0, len(selected) - 1, n_out)
            selected = np.interp(idx, np.arange(len(selected)),
                                 selected.astype(np.float64)).astype(np.float32)
            rate = PYANNOTE_EMBEDDING_RATE
        chunk = int(PYANNOTE_EMBEDDING_CHUNK_SECONDS * rate)
        vectors: list[np.ndarray] = []
        for i in range(0, max(1, len(selected)), chunk):
            piece = selected[i:i + chunk]
            if len(piece) < int(0.5 * rate):
                continue
            waveform = torch.from_numpy(
                piece.astype(np.float32)).unsqueeze(0)
            embedding = inference({"waveform": waveform,
                                   "sample_rate": rate})
            vectors.append(np.asarray(embedding, dtype=np.float64).ravel())
        if not vectors:
            raise ValueError("No embedding could be extracted from the audio")
        vec = np.mean(vectors, axis=0)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Empty embedding — audio has no usable signal")
        return (vec / norm).astype(np.float32)


# Backend names accepted by get_extractor() / the
# ``embedding_backend`` setting ("auto" = the default chain below).
EMBEDDING_BACKEND_NAMES: tuple[str, ...] = (
    "auto",
    "speechbrain-ecapa",
    "onnx-ecapa",
    "pyannote",
    "sherpa-eres2net",
    "fallback-spectral-v1",
)


def get_extractor(backend: str = "auto", token: str | None = None):
    """Return the extractor for ``backend`` (lazy — no model loads).

    * ``"auto"`` — :func:`get_default_extractor`.
    * ``"speechbrain-ecapa"`` — SpeechBrain ECAPA (needs torch).
    * ``"onnx-ecapa"`` — ONNX ECAPA via speakeronnx (no torch).
    * ``"sherpa-eres2net"`` — sherpa-onnx ERes2Net (needs
      sherpa-onnx; best measured separation).
    * ``"pyannote"`` — opt-in pyannote/embedding (needs torch +
      token; kept for users with that setup).
    * ``"fallback-spectral-v1"`` — test-only spectral proxy.

    Raises ValueError for an unknown name. The returned object
    always has ``.backend`` and ``.dim``; callers must store
    ``backend`` with the embedding.
    """
    if backend == "auto":
        return get_default_extractor(token=token)
    if backend == "speechbrain-ecapa":
        return SpeechBrainEcapaEmbeddingExtractor()
    if backend == "onnx-ecapa":
        return OnnxEcapaEmbeddingExtractor()
    if backend == "sherpa-eres2net":
        return SherpaEres2NetEmbeddingExtractor()
    if backend == "pyannote":
        return PyannoteEmbeddingExtractor(token=token)
    if backend == "fallback-spectral-v1":
        return FallbackEmbeddingExtractor()
    raise ValueError(
        f"unknown embedding backend {backend!r}; expected one of "
        f"{list(EMBEDDING_BACKEND_NAMES)}")


def _speechbrain_importable() -> bool:
    """Best-effort probe: torch + speechbrain import cleanly.

    Never raises — auto-detection must not crash the app if an
    optional dependency is half-installed. The strict error
    lives in SpeechBrainEcapaEmbeddingExtractor._load().
    """
    try:
        import torch  # noqa: F401
        import speechbrain.inference  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means "no"
        return False
    return True


def _speakeronnx_importable() -> bool:
    """Best-effort probe: speakeronnx imports cleanly (see above)."""
    try:
        import speakeronnx  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means "no"
        return False
    return True


def get_default_extractor(token: str | None = None):
    """Return the best available extractor, and say which one it is.

    SpeechBrain ECAPA when torch + speechbrain import, otherwise
    ONNX ECAPA when speakeronnx imports, otherwise the fallback
    spectral extractor. Pyannote is NOT in this chain any more —
    it is opt-in via ``get_extractor("pyannote")`` (it needs a
    gated model + token). The returned object always has
    ``.backend`` and ``.dim``; callers must store ``backend``
    with the embedding.
    """
    if _speechbrain_importable():
        return SpeechBrainEcapaEmbeddingExtractor()
    if _speakeronnx_importable():
        return OnnxEcapaEmbeddingExtractor()
    logger.info("No ECAPA backend importable — using the fallback "
                "spectral extractor (not a voice fingerprint)")
    return FallbackEmbeddingExtractor()
