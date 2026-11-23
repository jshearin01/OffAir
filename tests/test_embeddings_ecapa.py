"""Unit tests — ECAPA swap (Nov 8 2026): ONNX + SpeechBrain backends.

Fake models throughout (no torch, no speakeronnx, no downloads):
the real ONNX backend is verified separately in the manual run
(see PHASE4.md — measured cosine matrix on the 3-voice fixture).
"""
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import SPEAKER_MATCH_THRESHOLDS
from app.speakers.database import SpeakerDatabase
from app.speakers.embeddings import (
    FallbackEmbeddingExtractor,
    OnnxEcapaEmbeddingExtractor,
    PyannoteEmbeddingExtractor,
    SpeechBrainEcapaEmbeddingExtractor,
    get_default_extractor,
    get_extractor,
)
from app.speakers.matching import match_embedding, threshold_for_backend
from app.speakers.session import suggest_for_session
from app.transcription.engine import TRANSCRIPT_JSON


def unit192(angle: float) -> np.ndarray:
    """192-dim unit vector at ``angle`` from e0 (for threshold tests)."""
    vec = np.zeros(192, dtype=np.float32)
    vec[0] = math.cos(angle)
    vec[1] = math.sin(angle)
    return vec


def _write_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    samples = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(path), samples, rate)


# ------------------------------------------------------------- fake models ---
class FakeEmbedder:
    """Fake speakeronnx.SpeakerEmbedder: records input, fixed vector."""

    def __init__(self, model="wespeaker-ecapa512", providers=None):
        self.model = model
        self.input_lengths: list[int] = []
        self.input_dtypes: list = []

    def embed(self, source):
        arr = np.asarray(source, dtype=np.float32)
        self.input_lengths.append(len(arr))
        self.input_dtypes.append(arr.dtype)
        vec = np.zeros(192, dtype=np.float32)
        vec[0] = 0.6
        vec[1] = 0.8  # norm 1.0 already
        return vec


def _fake_speakeronnx(monkeypatch):
    module = types.ModuleType("speakeronnx")
    module.SpeakerEmbedder = FakeEmbedder
    monkeypatch.setitem(sys.modules, "speakeronnx", module)
    return module


class FakeTensor:
    """Minimal torch-tensor stand-in (from_numpy / unsqueeze / array)."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def unsqueeze(self, dim):
        return self

    def __array__(self, dtype=None):
        return np.asarray(self._arr, dtype=dtype) if dtype else self._arr


class FakeTorch:
    """Fake torch module: from_numpy only, records calls."""

    def __init__(self):
        self.from_numpy_calls: list[tuple] = []

    def from_numpy(self, arr):
        self.from_numpy_calls.append((arr.shape, arr.dtype))
        return FakeTensor(arr)


class FakeClassifier:
    """Fake speechbrain EncoderClassifier: fixed 192-dim vector."""

    instances: list = []

    def __init__(self):
        self.encode_calls = 0
        FakeClassifier.instances.append(self)

    @classmethod
    def from_hparams(cls, source, run_opts=None):
        obj = cls()
        obj.source = source
        obj.run_opts = run_opts
        return obj

    def encode_batch(self, waveform):
        self.encode_calls += 1
        # Deliberately non-unit (norm sqrt(96)) — extractor must normalise.
        return np.tile(np.array([0.6, 0.8]), 96).astype(np.float64)


def _fake_speechbrain(monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    inference = types.ModuleType("speechbrain.inference")
    inference.EncoderClassifier = FakeClassifier
    parent = types.ModuleType("speechbrain")
    parent.inference = inference
    monkeypatch.setitem(sys.modules, "speechbrain", parent)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", inference)
    FakeClassifier.instances.clear()
    return fake_torch


# ------------------------------------------------------------ ONNX backend ---
def test_onnx_extractor_returns_192_dim_unit_vector(tmp_path, monkeypatch):
    _fake_speakeronnx(monkeypatch)
    wav = tmp_path / "a.wav"
    _write_wav(wav, 2.0)
    ext = OnnxEcapaEmbeddingExtractor()
    assert ext.backend == "onnx-ecapa-v1" and ext.dim == 192
    vec = ext.extract(wav)
    assert vec.shape == (192,) and vec.dtype == np.float32
    assert np.linalg.norm(vec) == pytest.approx(1.0)
    assert vec[0] == pytest.approx(0.6) and vec[1] == pytest.approx(0.8)


def test_onnx_extractor_resamples_and_selects_windows(tmp_path, monkeypatch):
    module = _fake_speakeronnx(monkeypatch)
    wav = tmp_path / "a.wav"
    _write_wav(wav, 4.0, rate=48000)  # 48 kHz in, 16 kHz expected
    ext = OnnxEcapaEmbeddingExtractor()
    ext.extract(wav, windows=[(1.0, 2.0)])  # 1 s window -> 16000 samples
    embedder = ext._embedder
    assert embedder.input_lengths == [16000]
    assert embedder.input_dtypes[0] == np.float32
    assert embedder.model == "wespeaker-ecapa512"


def test_onnx_extractor_short_audio_raises(tmp_path, monkeypatch):
    _fake_speakeronnx(monkeypatch)
    wav = tmp_path / "a.wav"
    _write_wav(wav, 0.2)
    with pytest.raises(ValueError, match="at least 0.5 s"):
        OnnxEcapaEmbeddingExtractor().extract(wav)


def test_onnx_extractor_missing_package_raises_helpful_error(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "speakeronnx", None)
    wav = tmp_path / "a.wav"
    _write_wav(wav, 2.0)
    with pytest.raises(RuntimeError, match="speakeronnx is not installed"):
        OnnxEcapaEmbeddingExtractor().extract(wav)


# ------------------------------------------------------ SpeechBrain backend ---
def test_speechbrain_extractor_returns_192_dim_unit_vector(
        tmp_path, monkeypatch):
    _fake_speechbrain(monkeypatch)
    wav = tmp_path / "a.wav"
    _write_wav(wav, 2.0)
    ext = SpeechBrainEcapaEmbeddingExtractor()
    assert ext.backend == "speechbrain-ecapa-v1" and ext.dim == 192
    vec = ext.extract(wav)
    assert vec.shape == (192,) and vec.dtype == np.float32
    assert np.linalg.norm(vec) == pytest.approx(1.0)
    # Fake returns tiled (0.6, 0.8) with norm sqrt(96): normalising
    # divides every element by sqrt(96).
    assert vec[0] == pytest.approx(0.6 / math.sqrt(96))
    assert vec[1] == pytest.approx(0.8 / math.sqrt(96))


def test_speechbrain_extractor_chunks_long_audio(tmp_path, monkeypatch):
    _fake_speechbrain(monkeypatch)
    wav = tmp_path / "a.wav"
    _write_wav(wav, 25.0)  # 10 s chunks -> 3 encode_batch calls
    ext = SpeechBrainEcapaEmbeddingExtractor()
    vec = ext.extract(wav)
    assert vec.shape == (192,)
    assert FakeClassifier.instances[0].encode_calls == 3
    assert FakeClassifier.instances[0].source == \
        "speechbrain/spkrec-ecapa-voxceleb"
    assert FakeClassifier.instances[0].run_opts == {"device": "cpu"}


def test_speechbrain_extractor_short_audio_raises(tmp_path, monkeypatch):
    _fake_speechbrain(monkeypatch)
    wav = tmp_path / "a.wav"
    _write_wav(wav, 0.2)
    with pytest.raises(ValueError, match="at least 0.5 s"):
        SpeechBrainEcapaEmbeddingExtractor().extract(wav)


def test_speechbrain_extractor_missing_torch_raises_helpful_error(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "speechbrain", None)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", None)
    wav = tmp_path / "a.wav"
    _write_wav(wav, 2.0)
    with pytest.raises(RuntimeError, match="speechbrain .* is not installed"):
        SpeechBrainEcapaEmbeddingExtractor().extract(wav)


def test_speechbrain_extractor_missing_speechbrain_raises_helpful_error(
        tmp_path, monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "speechbrain", None)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", None)
    wav = tmp_path / "a.wav"
    _write_wav(wav, 2.0)
    with pytest.raises(RuntimeError, match="speechbrain .* is not installed"):
        SpeechBrainEcapaEmbeddingExtractor().extract(wav)


# ------------------------------------------------------- factory / selection ---
def test_get_extractor_names():
    assert get_extractor("onnx-ecapa").backend == "onnx-ecapa-v1"
    assert get_extractor("speechbrain-ecapa").backend == \
        "speechbrain-ecapa-v1"
    assert get_extractor("pyannote").backend == "pyannote-embedding-v1"
    assert isinstance(get_extractor("pyannote"), PyannoteEmbeddingExtractor)
    assert get_extractor("fallback-spectral-v1").backend == \
        "fallback-spectral-v1"
    # Construction is lazy: no model loads, no imports required.
    assert get_extractor("onnx-ecapa").dim == 192
    assert get_extractor("speechbrain-ecapa").dim == 192


def test_get_extractor_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown embedding backend"):
        get_extractor("pyannote-embedding-v1")  # backend id, not a name
    with pytest.raises(ValueError, match="unknown embedding backend"):
        get_extractor("resnet34")


def test_get_default_extractor_falls_back_without_ecapa_deps(monkeypatch):
    # This VM's test env has neither torch/speechbrain nor
    # speakeronnx — auto must degrade to the fallback, never crash.
    for name in ("torch", "speechbrain", "speechbrain.inference",
                 "speakeronnx"):
        monkeypatch.setitem(sys.modules, name, None)
    ext = get_default_extractor()
    assert isinstance(ext, FallbackEmbeddingExtractor)
    assert get_extractor("auto").backend == "fallback-spectral-v1"


def test_get_default_extractor_prefers_speechbrain(monkeypatch):
    _fake_speechbrain(monkeypatch)
    _fake_speakeronnx(monkeypatch)
    ext = get_default_extractor()
    assert isinstance(ext, SpeechBrainEcapaEmbeddingExtractor)


def test_get_default_extractor_uses_onnx_when_no_torch(monkeypatch):
    _fake_speakeronnx(monkeypatch)
    for name in ("torch", "speechbrain", "speechbrain.inference"):
        monkeypatch.setitem(sys.modules, name, None)
    ext = get_default_extractor()
    assert isinstance(ext, OnnxEcapaEmbeddingExtractor)


# ---------------------------------------------------- per-backend thresholds ---
def test_threshold_for_backend():
    assert threshold_for_backend("onnx-ecapa-v1") == 0.55
    assert threshold_for_backend("speechbrain-ecapa-v1") == 0.55
    assert threshold_for_backend("sherpa-eres2net-v1") == 0.55
    assert threshold_for_backend("fallback-spectral-v1") == 0.90
    assert threshold_for_backend("pyannote-embedding-v1") == 0.90
    assert threshold_for_backend("something-else") == 0.90
    assert set(SPEAKER_MATCH_THRESHOLDS) == {
        "fallback-spectral-v1", "pyannote-embedding-v1",
        "onnx-ecapa-v1", "speechbrain-ecapa-v1",
        "sherpa-eres2net-v1"}


def test_match_embedding_uses_per_backend_threshold(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Priya", unit192(0.0), backend="onnx-ecapa-v1")
    # cos(0.9 rad) = 0.62 clears the ECAPA threshold (0.55) …
    hit = match_embedding(unit192(0.9), db, backend="onnx-ecapa-v1")
    assert hit.suggested_name == "Priya"
    assert hit.threshold == 0.55
    # … but cos(1.1 rad) = 0.45 does not.
    miss = match_embedding(unit192(1.1), db, backend="onnx-ecapa-v1")
    assert miss.is_unknown
    # Same scores against the fallback backend would be unknown
    # (0.62 < 0.90) — the threshold really is per-backend.
    db.add_embedding("Marcus", unit192(0.0), backend="fallback-spectral-v1")
    assert match_embedding(unit192(0.9), db,
                           backend="fallback-spectral-v1").is_unknown
    db.close()


def test_match_embedding_explicit_threshold_overrides(tmp_path):
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Priya", unit192(0.0), backend="onnx-ecapa-v1")
    assert match_embedding(unit192(0.9), db, backend="onnx-ecapa-v1",
                           threshold=0.90).is_unknown
    assert match_embedding(unit192(0.9), db, backend="onnx-ecapa-v1",
                           threshold=0.50).suggested_name == "Priya"
    db.close()


class _EcapaExtractor:
    """Fake extractor with the ONNX ECAPA backend id."""

    backend = "onnx-ecapa-v1"
    dim = 192

    def __init__(self, vector):
        self._vector = vector

    def extract(self, wav_path, windows=None):
        return self._vector


def _make_session(tmp_path, labels):
    session = tmp_path / "session"
    session.mkdir()
    segments = [{"start": float(i * 5), "end": float(i * 5 + 4),
                 "text": f"segment {i}", "speaker": label}
                for i, label in enumerate(labels)]
    (session / TRANSCRIPT_JSON).write_text(json.dumps(
        {"segments": segments}), encoding="utf-8")
    (session / "combined.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    return session


def test_suggest_for_session_uses_ecapa_threshold_by_default(tmp_path):
    session = _make_session(tmp_path, ["SPEAKER_00"])
    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Priya", unit192(0.0), backend="onnx-ecapa-v1")
    # 0.62 would be "Unknown" at the old 0.90 default; with the
    # per-backend default it suggests.
    suggestions = suggest_for_session(
        session, db, extractor=_EcapaExtractor(unit192(0.9)))
    assert len(suggestions) == 1
    assert suggestions[0].suggested_name == "Priya"
    assert suggestions[0].match.threshold == 0.55
    db.close()
