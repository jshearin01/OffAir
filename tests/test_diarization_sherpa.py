"""Sherpa diarization tests — fake pipeline + settings only.

No sherpa-onnx, no model download, no network, no hardware. The
real-model test at the bottom is gated on
``OFFAIR_TEST_REAL_SHERPA=1`` (same pattern as the MiniLM gated
test) and skips gracefully when sherpa-onnx or the models are not
available.
"""
from __future__ import annotations

import os
import sys
import urllib.error
import wave
from pathlib import Path

import numpy as np
import pytest

from app.config import (
    AppConfig,
    config_from_settings,
    settings_to_dict,
)
from app.diarization import DIARIZATION_MODES
from app.diarization.sherpa import (
    EMBEDDING_MODEL_BYTES,
    EMBEDDING_MODEL_FILE,
    MODE_LABEL,
    SEGMENTATION_MODEL_BYTES,
    SEGMENTATION_MODEL_FILE,
    SEGMENTATION_TARBALL,
    SEGMENTATION_TARBALL_BYTES,
    SherpaDiarizationResult,
    SherpaDiarizer,
    SherpaModelError,
    SherpaNotInstalledError,
    _download,
    _renumber_speakers,
    _ssl_context,
    check_availability,
    ensure_sherpa_models,
    sherpa_importable,
    validate_sherpa_settings,
)
from app.speakers.embeddings import get_extractor

FIXTURE = Path(__file__).parent / "fixtures" / "meeting-3speakers-16k.wav"
SR = 16000


def _write_wav(path: Path, seconds: float = 2.0) -> Path:
    t = np.arange(int(seconds * SR)) / SR
    samples = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


class _FakeSegment:
    def __init__(self, start, end, speaker):
        self.start, self.end, self.speaker = start, end, speaker


class _FakeResult:
    def __init__(self, segments):
        self._segments = segments

    def sort_by_start_time(self):
        return sorted(self._segments, key=lambda s: s.start)


class _FakeDiarizer:
    """Stand-in for sherpa_onnx.OfflineSpeakerDiarization."""

    def __init__(self, segments):
        self._segments = segments
        self.seen_samples = None

    def process(self, samples):
        self.seen_samples = samples
        return _FakeResult(self._segments)


# --- mode registration ------------------------------------------------------

def test_sherpa_registered_in_mode_list():
    assert "sherpa" in DIARIZATION_MODES
    assert DIARIZATION_MODES["sherpa"]["label"] == MODE_LABEL
    assert "no token" in DIARIZATION_MODES["sherpa"]["label"].lower()
    # The other modes are still there.
    assert set(DIARIZATION_MODES) == {"simple", "full", "sherpa"}


def test_config_accepts_sherpa_mode():
    cfg = AppConfig(diarization_mode="sherpa")
    assert cfg.diarization_mode == "sherpa"
    assert cfg.sherpa_expected_speakers == 0
    assert cfg.sherpa_clustering_threshold == 0.5
    with pytest.raises(ValueError, match="unknown diarization mode"):
        AppConfig(diarization_mode="weird")


# --- settings validation ----------------------------------------------------

def test_validate_sherpa_settings_defaults():
    assert validate_sherpa_settings(None, None) == (0, 0.5)


def test_validate_sherpa_settings_hint():
    assert validate_sherpa_settings(3, 0.6) == (3, 0.6)
    assert validate_sherpa_settings(1, 0.05) == (1, 0.05)
    assert validate_sherpa_settings(10, 0.95) == (10, 0.95)


def test_validate_sherpa_settings_rejects_bad_values():
    with pytest.raises(ValueError, match="expected_speakers"):
        validate_sherpa_settings(11, 0.5)
    with pytest.raises(ValueError, match="expected_speakers"):
        validate_sherpa_settings(-1, 0.5)
    with pytest.raises(ValueError, match="expected_speakers"):
        validate_sherpa_settings("3", 0.5)
    with pytest.raises(ValueError, match="clustering_threshold"):
        validate_sherpa_settings(0, 0.0)
    with pytest.raises(ValueError, match="clustering_threshold"):
        validate_sherpa_settings(0, 1.0)
    with pytest.raises(ValueError, match="clustering_threshold"):
        validate_sherpa_settings(0, "0.5")
    with pytest.raises(ValueError):
        AppConfig(sherpa_expected_speakers=11)
    with pytest.raises(ValueError):
        AppConfig(sherpa_clustering_threshold=2.0)


def test_sherpa_settings_round_trip(tmp_path):
    cfg = AppConfig(diarization_mode="sherpa",
                    sherpa_model_dir=tmp_path / "models",
                    sherpa_expected_speakers=3,
                    sherpa_clustering_threshold=0.6)
    data = settings_to_dict(cfg)
    assert data["diarization_mode"] == "sherpa"
    assert data["sherpa_expected_speakers"] == 3
    assert data["sherpa_clustering_threshold"] == 0.6
    assert data["sherpa_model_dir"] == str(tmp_path / "models")
    restored = config_from_settings(data)
    assert restored.diarization_mode == "sherpa"
    assert restored.sherpa_expected_speakers == 3
    assert restored.sherpa_clustering_threshold == 0.6
    assert restored.sherpa_model_dir == tmp_path / "models"
    # A bad value in a settings file falls back to the default.
    restored = config_from_settings({"sherpa_expected_speakers": 99,
                                     "sherpa_clustering_threshold": 5.0})
    assert restored.sherpa_expected_speakers == 0
    assert restored.sherpa_clustering_threshold == 0.5


# --- label renumbering ------------------------------------------------------

def test_renumber_speakers_closes_threshold_clustering_gaps():
    # Threshold clustering can number clusters 00, 01, 03 (no 02).
    turns = [(0.0, 5.0, 0), (6.0, 9.0, 1), (10.0, 15.0, 3),
             (16.0, 20.0, 0), (21.0, 25.0, 3)]
    assert _renumber_speakers(turns) == [
        (0.0, 5.0, "SPEAKER_00"), (6.0, 9.0, "SPEAKER_01"),
        (10.0, 15.0, "SPEAKER_02"), (16.0, 20.0, "SPEAKER_00"),
        (21.0, 25.0, "SPEAKER_02")]


def test_renumber_speakers_empty():
    assert _renumber_speakers([]) == []


# --- fake-pipeline diarization ----------------------------------------------

def test_fake_diarizer_returns_full_mode_shaped_result(tmp_path):
    wav = _write_wav(tmp_path / "meeting.wav", seconds=6.0)
    fake = _FakeDiarizer([
        _FakeSegment(0.0, 2.0, 0), _FakeSegment(2.0, 4.0, 1),
        _FakeSegment(4.0, 6.0, 3),  # gap in cluster ids, on purpose
    ])
    calls = []
    result = SherpaDiarizer(diarizer=fake).diarize(
        wav, progress_callback=lambda f, t: calls.append(f))
    assert isinstance(result, SherpaDiarizationResult)
    assert result.turns == [
        (0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01"),
        (4.0, 6.0, "SPEAKER_02")]
    assert result.speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    assert result.audio_duration_seconds == 6.0
    assert result.clustering == "threshold"
    assert result.expected_speakers == 0
    assert result.wall_seconds >= 0.0
    # The samples handed to the pipeline are 16 kHz float32.
    assert fake.seen_samples.dtype == np.float32
    assert len(fake.seen_samples) == 6 * SR
    # Progress indicator: start, running, done.
    assert calls[0] == 0.0 and calls[-1] == 1.0
    # Same output shape as the pyannote full mode.
    payload = result.to_dict()
    assert payload["mode"] == "sherpa"
    assert "sherpa-onnx" in payload["method"]
    assert payload["speakers"] == result.speakers
    assert payload["turns"] == [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
        {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_02"}]


def test_expected_speakers_hint_recorded_in_result(tmp_path):
    wav = _write_wav(tmp_path / "meeting.wav", seconds=2.0)
    fake = _FakeDiarizer([_FakeSegment(0.0, 2.0, 0)])
    result = SherpaDiarizer(expected_speakers=3,
                            diarizer=fake).diarize(wav)
    assert result.clustering == "expected-count"
    assert result.expected_speakers == 3
    assert result.clustering_threshold == 0.5


def test_diarize_missing_audio_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SherpaDiarizer(diarizer=_FakeDiarizer([])).diarize(
            tmp_path / "nope.wav")


def test_diarize_resamples_non_16k_audio(tmp_path):
    # 48 kHz input must reach the pipeline at 16 kHz.
    t = np.arange(48000) / 48000
    samples = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    wav = tmp_path / "meeting48.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(pcm.tobytes())
    fake = _FakeDiarizer([_FakeSegment(0.0, 1.0, 0)])
    result = SherpaDiarizer(diarizer=fake).diarize(wav)
    assert len(fake.seen_samples) == SR
    assert result.audio_duration_seconds == 1.0


# --- missing sherpa / offline paths -----------------------------------------

def test_missing_sherpa_gives_guided_error(monkeypatch, tmp_path):
    # Force the import to fail even on a machine that has sherpa-onnx.
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    wav = _write_wav(tmp_path / "meeting.wav", seconds=1.0)
    with pytest.raises(SherpaNotInstalledError,
                       match="pip install sherpa-onnx"):
        SherpaDiarizer(model_dir=tmp_path).diarize(wav)


def test_check_availability_never_raises_and_names_label(tmp_path):
    info = check_availability(model_dir=tmp_path)
    assert info["label"] == MODE_LABEL
    assert info["model_dir"] == str(tmp_path)
    assert info["ready"] is False  # nothing installed / no models here
    assert set(info) == {"installed", "models_present", "model_dir",
                         "label", "ready"}


def test_sherpa_importable_probe_is_bool():
    assert sherpa_importable() is False or sherpa_importable() is True


def test_ensure_models_reuses_files_at_expected_size(tmp_path):
    # Pre-place both models at exactly the expected sizes (sparse
    # files — no real download, no network).
    seg_dir = tmp_path / "sherpa-onnx-pyannote-segmentation-3-0"
    seg_dir.mkdir()
    seg = seg_dir / SEGMENTATION_MODEL_FILE
    emb = tmp_path / EMBEDDING_MODEL_FILE
    with open(seg, "wb") as f:
        f.truncate(SEGMENTATION_MODEL_BYTES)
    with open(emb, "wb") as f:
        f.truncate(EMBEDDING_MODEL_BYTES)
    found_seg, found_emb = ensure_sherpa_models(model_dir=tmp_path)
    assert found_seg == seg and found_emb == emb


def test_ensure_models_offline_failure_is_guided(monkeypatch, tmp_path):
    class _BoomOpener:
        def open(self, url):
            raise urllib.error.URLError("network unreachable (test)")

    monkeypatch.setattr(urllib.request, "build_opener",
                        lambda *a, **k: _BoomOpener())
    with pytest.raises(SherpaModelError, match="no account"):
        ensure_sherpa_models(model_dir=tmp_path)


# --- ERes2Net embedding backend registration --------------------------------

def test_sherpa_eres2net_extractor_registered():
    ext = get_extractor("sherpa-eres2net")
    assert ext.backend == "sherpa-eres2net-v1"
    assert ext.dim == 192
    with pytest.raises(ValueError, match="unknown embedding backend"):
        get_extractor("sherpa-nope")


def test_sherpa_eres2net_extractor_missing_sherpa(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    ext = get_extractor("sherpa-eres2net")
    wav = _write_wav(tmp_path / "spk.wav", seconds=1.0)
    with pytest.raises(RuntimeError, match="sherpa-onnx is not installed"):
        ext.extract(wav)


# --- real model (gated, skips offline) --------------------------------------
@pytest.mark.skipif(
    os.environ.get("OFFAIR_TEST_REAL_SHERPA") != "1",
    reason="Real sherpa diarization test: set "
           "OFFAIR_TEST_REAL_SHERPA=1 to run it.")
def test_real_sherpa_diarizes_fixture_9_turns(tmp_path):
    """The real sherpa pipeline on the 3-speaker fixture.

    Downloads ~33 MB of models on first run (public GitHub
    releases, no token); skips (never fails) when sherpa-onnx
    is missing or the models cannot be fetched.
    """
    if not sherpa_importable():
        pytest.skip("sherpa-onnx is not installed here")
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    model_dir = tmp_path / "models"
    try:
        diarizer = SherpaDiarizer(model_dir=model_dir)
        result = diarizer.diarize(FIXTURE)
    except SherpaModelError:
        pytest.skip("sherpa models could not be downloaded here")
    assert len(result.turns) == 9, f"expected 9 turns, got {result.turns}"
    assert len(result.speakers) == 3, f"expected 3 speakers, got {result.speakers}"
    rtf = result.wall_seconds / (result.audio_duration_seconds or 1)
    assert rtf < 1.0, f"RTF {rtf:.2f} — slower than real time"
    # Identity-check each turn with the ERes2Net extractor against
    # the three isolated references (same method as the tryouts).
    from app.speakers.embeddings import SherpaEres2NetEmbeddingExtractor
    try:
        extractor = SherpaEres2NetEmbeddingExtractor()
        refs = {}
        for name in ("john", "priya", "marcus"):
            refs[name] = extractor.extract(
                FIXTURE.parent / f"speaker-{name}-16k.wav")
    except (SherpaModelError, RuntimeError):
        pytest.skip("ERes2Net extractor could not load here")
    truth = ["john", "priya", "marcus"] * 3
    correct = 0
    for (start, end, _label), expected in zip(result.turns, truth):
        vec = extractor.extract(FIXTURE, windows=[(start, end)])
        scores = {n: float(vec @ r) for n, r in refs.items()}
        if max(scores, key=scores.get) == expected:
            correct += 1
    assert correct == 9, f"identity {correct}/9 — turns: {result.turns}"


# --- Model-download SSL context (Nov 9 2026) ------------------------------
# A bare Windows eval install fails urllib's default verification for
# github.com's chain (CERTIFICATE_VERIFY_FAILED) while certifi's bundle
# verifies fine — _download must use certifi when it is installed.


def test_ssl_context_uses_certifi_when_available(monkeypatch):
    import ssl as _ssl

    certifi = pytest.importorskip("certifi")
    orig_create = _ssl.create_default_context
    seen: dict = {}

    def _fake_create_default_context(*args, **kwargs):
        seen.update(kwargs)
        return orig_create()

    monkeypatch.setattr(_ssl, "create_default_context",
                        _fake_create_default_context)
    ctx = _ssl_context()
    assert isinstance(ctx, _ssl.SSLContext)
    assert seen.get("cafile") == certifi.where()


def test_ssl_context_falls_back_without_certifi(monkeypatch):
    import ssl as _ssl

    orig_create = _ssl.create_default_context
    seen: dict = {}

    def _fake_create_default_context(*args, **kwargs):
        seen.update(kwargs)
        return orig_create()

    monkeypatch.setattr(_ssl, "create_default_context",
                        _fake_create_default_context)
    # Make `import certifi` raise ImportError inside _ssl_context.
    monkeypatch.setitem(sys.modules, "certifi", None)
    ctx = _ssl_context()
    assert isinstance(ctx, _ssl.SSLContext)
    assert "cafile" not in seen


class _FakeDownloadResponse:
    """Minimal stand-in for the urllib response _download reads."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        if not self._payload:
            return b""
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk


def test_download_uses_ssl_context_and_reports_progress(monkeypatch,
                                                        tmp_path):
    import ssl as _ssl
    import urllib.request as _urlreq

    captured: dict = {}

    class _FakeOpener:
        def open(self, url):
            return _FakeDownloadResponse(b"fake-model-bytes")

    def _fake_build_opener(*handlers):
        for h in handlers:
            if isinstance(h, _urlreq.HTTPSHandler):
                captured["context"] = h._context
        return _FakeOpener()

    monkeypatch.setattr(_urlreq, "build_opener", _fake_build_opener)
    progress: list = []
    dest = tmp_path / "model.onnx"
    _download("https://example.invalid/model.onnx", dest,
              expected_bytes=len(b"fake-model-bytes"),
              progress_callback=lambda f, m: progress.append(f))
    assert dest.read_bytes() == b"fake-model-bytes"
    assert progress and progress[-1] == 1.0
    assert isinstance(captured.get("context"), _ssl.SSLContext)
