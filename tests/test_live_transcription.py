"""Live transcription tests — chunked recording + crash resilience.

Nov 19 2026 (items 2 + 3). Two layers:

* Qt-free: engine cancel, chunker finish(), live save/load, and the
  crash simulation (a fake engine raising mid-chunk leaves exactly
  the completed chunks in transcript-live.json, and the session
  finalizes from it). No model download, no hardware, no network.
* Qt-gated (``qapp`` fixture, ``pytest.importorskip("PyQt6")`` per
  the other UI test files — skipped in the phase2 venv): the live
  poller, pause/resume, the model-download status, Stop finalize
  (live + tail, and the full-file fallback), and manual-transcribe
  Cancel. Fake engines only — a real model is never downloaded.

Long-audio tests (10-minute tracks) write ~100 MB of WAVs per
test into ``tmp_path`` — and ``/tmp`` on this machine is a 512 MB
tmpfs shared by every test's ``tmp_path`` dir. A test that writes
big WAVs must delete them in a ``finally`` block (not just on
success), and a full-suite run that fails with a soundfile
``System error`` on write is usually ``/tmp`` full, not a product
bug — clear ``/tmp/pytest-of-*`` and re-run.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import AppConfig
from app.nearlive.chunker import (
    ChunkedTranscriber,
    load_live_transcript,
    save_live_transcript,
)
from app.storage.sessions import TRANSCRIPT_LIVE_FILE, create_session
from app.transcription.engine import (
    Segment,
    TranscriptionEngine,
    TranscriptionResult,
    audio_duration_seconds,
    whisper_model_cached,
)


def _write_tone_wav(path, duration=61.0, rate=16000, freq=440.0):
    t = np.arange(int(duration * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), samples, rate, subtype="PCM_16")
    return path


class _FakeChunkEngine:
    """Fake Whisper: two 2 s segments per chunk, text names the chunk file.

    Plus the overlap words (S5): every full window also emits the
    same "overlap words" segment at the end of its 0.5 s overlap
    (relative 24.5–25.0 s) and, from the second window on, at the
    start of its overlap (relative 0.0–0.5 s) — the same words at
    the same absolute start, so the chunker's dedup must drop the
    second copy. Dedup is exact-match only (N7): a segment is
    dropped only when (start, text) matches an earlier one.
    """

    def __init__(self):
        self.calls = []
        self.kwargs = []

    def transcribe_file(self, audio_path, **kwargs):
        self.calls.append(Path(audio_path).name)
        self.kwargs.append(kwargs)
        name = Path(audio_path).name
        index = 0
        if name.startswith("chunk-"):
            try:
                index = int(name[6:9])
            except ValueError:
                index = 0
        segments = [Segment(0.0, 2.0, f"hello from {name}"),
                    Segment(2.0, 4.0, "world")]
        if index >= 1:
            # Re-transcription of the 0.5 s overlap at the window
            # start: same words, same absolute start as the
            # previous window's trailing overlap segment.
            segments.append(Segment(0.0, 0.5, "overlap words"))
        duration = audio_duration_seconds(Path(audio_path))
        if duration is not None and duration >= 25.0:
            # Trailing overlap segment (relative 24.5–25.0 s).
            segments.append(Segment(24.5, 25.0, "overlap words"))
        return TranscriptionResult(
            segments=segments,
            model="tiny", audio_duration_seconds=4.0, wall_seconds=0.1)

    def unload(self):
        pass


class _CancelAfter:
    """is_cancelled that fires after ``n`` checks."""

    def __init__(self, n):
        self.n = n
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.calls > self.n


class _FakeRaw:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _FakeModel:
    """Mimics faster-whisper's WhisperModel.transcribe() interface."""

    def __init__(self, raws):
        self.raws = raws

    def transcribe(self, path, language="en", beam_size=5):
        return iter(self.raws), {"language": language}


def _silent_wav(path, seconds=6, rate=16000):
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * rate * seconds)
    return path


# ------------------------------------------------------------------ engine --
def test_transcribe_file_cancel_returns_partial_segments(tmp_path):
    raws = [_FakeRaw(float(i), float(i + 1), f"word {i}") for i in range(5)]
    engine = TranscriptionEngine(model_size="tiny", model=_FakeModel(raws))
    wav = _silent_wav(tmp_path / "a.wav")
    # _CancelAfter(4): the first two checks are the pre-load and
    # post-load checks (S3), so the cancel fires on the third
    # segment — two segments transcribed, then stop.
    result = engine.transcribe_file(wav, is_cancelled=_CancelAfter(4))
    assert result.cancelled is True
    assert [s.text for s in result.segments] == ["word 0", "word 1"]
    # No exception, and no final 100% progress on a cancelled run.
    seen = []
    engine2 = TranscriptionEngine(model_size="tiny", model=_FakeModel(raws))
    engine2.transcribe_file(
        wav, is_cancelled=_CancelAfter(4),
        progress_callback=lambda f, e, r: seen.append(f))
    assert not seen or seen[-1] < 1.0


def test_transcribe_session_cancel_writes_no_files(tmp_path):
    raws = [_FakeRaw(float(i), float(i + 1), f"word {i}") for i in range(5)]
    engine = TranscriptionEngine(model_size="tiny", model=_FakeModel(raws))
    session = tmp_path / "s"
    session.mkdir()
    _silent_wav(session / "combined.wav")
    # _CancelAfter(4) — see test_transcribe_file_cancel_returns_
    # partial_segments: two checks are the pre/post-load checks.
    result = engine.transcribe_session(session, is_cancelled=_CancelAfter(4))
    assert result.cancelled is True
    assert len(result.segments) == 2
    assert not (session / "transcript.json").exists()
    assert not (session / "transcript.txt").exists()
    assert not (session / "transcript.srt").exists()


def test_whisper_model_cached_checks_snapshot_dir(tmp_path):
    root = tmp_path / "hub"
    assert whisper_model_cached("base", download_root=root) is False
    snap = (root / "models--Systran--faster-whisper-base"
            / "snapshots" / "abc123")
    snap.mkdir(parents=True)
    # An empty snapshot dir is not a downloaded model.
    assert whisper_model_cached("base", download_root=root) is False
    (snap / "model.bin").write_bytes(b"x")
    assert whisper_model_cached("base", download_root=root) is True
    assert whisper_model_cached("tiny", download_root=root) is False


def test_whisper_model_cached_honors_cache_env_vars(tmp_path, monkeypatch):
    """N4: HF_HOME / HUGGINGFACE_HUB_CACHE / XDG_CACHE_HOME are
    honoured, and a snapshot dir without model.bin is not a model."""
    # Isolate the ~/.cache fallback so the real home cannot leak in.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME",
                "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)

    def _snap(root, model):
        snap = (root / f"models--Systran--faster-whisper-{model}"
                / "snapshots" / "abc123")
        snap.mkdir(parents=True)
        return snap

    # HF_HOME: a snapshot holding only metadata is not a model.
    home = tmp_path / "hf-home"
    snap = _snap(home / "hub", "base")
    (snap / "config.json").write_bytes(b"{}")
    monkeypatch.setenv("HF_HOME", str(home))
    assert whisper_model_cached("base") is False
    (snap / "model.bin").write_bytes(b"x")
    assert whisper_model_cached("base") is True
    assert whisper_model_cached("tiny") is False

    # XDG_CACHE_HOME.
    monkeypatch.delenv("HF_HOME", raising=False)
    xdg = tmp_path / "xdg"
    snap = _snap(xdg / "huggingface" / "hub", "tiny")
    (snap / "model.bin").write_bytes(b"x")
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    assert whisper_model_cached("tiny") is True
    assert whisper_model_cached("base") is False

    # HUGGINGFACE_HUB_CACHE (the HF_HUB_CACHE alias).
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    hub = tmp_path / "hub2"
    snap = _snap(hub, "small")
    (snap / "model.bin").write_bytes(b"x")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    assert whisper_model_cached("small") is True
    assert whisper_model_cached("tiny") is False


def test_transcribe_file_cancel_before_load_skips_model_load(tmp_path):
    """S3: a cancel already firing means the model is never loaded."""
    raws = [_FakeRaw(float(i), float(i + 1), f"word {i}") for i in range(3)]

    class _LoadCountingEngine(TranscriptionEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loads = 0

        def _load_model(self):
            self.loads += 1
            self._model = _FakeModel(raws)
            return self._model

    engine = _LoadCountingEngine(model_size="tiny")
    wav = _silent_wav(tmp_path / "a.wav")
    result = engine.transcribe_file(wav, is_cancelled=lambda: True)
    assert result.cancelled is True
    assert result.segments == []
    assert engine.loads == 0


def test_transcribe_file_cancel_during_load_unloads_and_skips(
        tmp_path):
    """S3: a cancel that fires while the model loads takes effect
    as soon as the load returns — the model is unloaded and
    nothing is transcribed."""
    raws = [_FakeRaw(float(i), float(i + 1), f"word {i}") for i in range(3)]

    class _LoadCountingEngine(TranscriptionEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loads = 0
            self.checks = 0

        def _load_model(self):
            self.loads += 1
            self._model = _FakeModel(raws)
            return self._model

    engine = _LoadCountingEngine(model_size="tiny")

    def _cancel_after_load():
        # First check is before _load_model (False), second is
        # immediately after it returns (True).
        engine.checks += 1
        return engine.checks > 1

    wav = _silent_wav(tmp_path / "a.wav")
    result = engine.transcribe_file(wav, is_cancelled=_cancel_after_load)
    assert engine.loads == 1
    assert result.cancelled is True
    assert result.segments == []
    assert engine._model is None  # unloaded, not left resident


# ----------------------------------------------------------------- chunker --
def test_finish_transcribes_partial_tail(tmp_path):
    wav = _write_tone_wav(tmp_path / "a.wav", duration=61.0)
    transcriber = ChunkedTranscriber(engine=_FakeChunkEngine())
    first = transcriber.process_new_audio(wav, work_dir=tmp_path / "work")
    # Both full windows are due at once: [0, 25] and [24.5, 49.5].
    assert len(first) == 2
    tail = transcriber.finish(wav, work_dir=tmp_path / "work")
    # Only the partial tail [49.0, 61.0] is left.
    assert [round(c.start, 1) for c in tail] == [49.0]
    assert tail[-1].end == pytest.approx(61.0)
    assert transcriber.watermark_seconds == pytest.approx(61.0)
    assert transcriber.chunks_done == 3
    starts = [s.start for s in transcriber.segments]
    assert starts == sorted(starts)
    keys = [(round(s.start, 1), s.text) for s in transcriber.segments]
    assert len(keys) == len(set(keys))  # no duplicates across the merge


def test_finish_short_audio_is_one_window(tmp_path):
    wav = _write_tone_wav(tmp_path / "a.wav", duration=10.0)
    transcriber = ChunkedTranscriber(engine=_FakeChunkEngine())
    tail = transcriber.finish(wav, work_dir=tmp_path / "work")
    assert len(tail) == 1
    assert tail[0].start == 0.0 and tail[0].end == pytest.approx(10.0)
    assert transcriber.watermark_seconds == pytest.approx(10.0)


def test_finish_after_complete_file_is_noop(tmp_path):
    wav = _write_tone_wav(tmp_path / "a.wav", duration=61.0)
    transcriber = ChunkedTranscriber(engine=_FakeChunkEngine())
    transcriber.process_file(wav, work_dir=tmp_path / "work")
    assert transcriber.finish(wav, work_dir=tmp_path / "work") == []
    assert transcriber.watermark_seconds == pytest.approx(61.0)


def test_overlap_duplicate_words_are_deduped(tmp_path):
    """S5: the same words re-transcribed in the 0.5 s overlap of
    consecutive windows appear once in the merged transcript.

    The fake emits "overlap words" at the end of each full window
    and again at the start of the next one (same absolute start,
    same text); the chunker dedups exact (start, text) matches.
    Dedup is exact-match only (N7): two segments sharing a start
    but with different text are both kept.
    """
    wav = _write_tone_wav(tmp_path / "a.wav", duration=61.0)
    transcriber = ChunkedTranscriber(engine=_FakeChunkEngine())
    transcriber.process_new_audio(wav, work_dir=tmp_path / "work")
    transcriber.finish(wav, work_dir=tmp_path / "work")
    overlaps = [s for s in transcriber.segments
                if s.text == "overlap words"]
    # Emitted four times (trailing of windows 0/1, leading of
    # window 1 and of the tail) at two absolute starts — kept once
    # at each.
    assert [round(s.start, 1) for s in overlaps] == [24.5, 49.0]
    assert len(transcriber.segments) == 8
    # Exact-match rule: the two segments starting at 24.5 s with
    # different text are both kept.
    at_24_5 = sorted(s.text for s in transcriber.segments
                     if round(s.start, 1) == 24.5)
    assert at_24_5 == ["hello from chunk-001.wav", "overlap words"]


def test_finish_transcribes_sub_second_tail_sliver(tmp_path):
    """N6: a tail with less than 1 s of new audio past the
    watermark is still transcribed as a short window, not dropped."""
    wav = _write_tone_wav(tmp_path / "a.wav", duration=74.5)
    transcriber = ChunkedTranscriber(engine=_FakeChunkEngine())
    transcriber.process_new_audio(wav, work_dir=tmp_path / "work")
    assert transcriber.watermark_seconds == pytest.approx(74.0)
    _write_tone_wav(wav, duration=74.9)  # 0.9 s of new audio arrives
    tail = transcriber.finish(wav, work_dir=tmp_path / "work")
    assert len(tail) == 1
    assert tail[0].start == pytest.approx(73.5)  # watermark - overlap
    assert tail[0].end == pytest.approx(74.9)
    assert transcriber.watermark_seconds == pytest.approx(74.9)


def test_save_load_live_transcript_round_trip(tmp_path):
    wav = _write_tone_wav(tmp_path / "a.wav", duration=30.0)
    transcriber = ChunkedTranscriber(engine=_FakeChunkEngine(), model="base")
    transcriber.process_new_audio(wav, work_dir=tmp_path / "work")
    path = transcriber.save_live(tmp_path)
    assert path.name == TRANSCRIPT_LIVE_FILE
    payload = json.loads(path.read_text())
    assert payload["live"] is True
    assert payload["model"] == "base"
    assert payload["chunks_done"] == 1
    assert payload["watermark_seconds"] == pytest.approx(25.0)
    assert len(payload["segments"]) == 3  # hello, world, overlap words
    loaded = load_live_transcript(tmp_path)
    assert loaded is not None
    assert loaded["segments"] == transcriber.segments
    assert loaded["chunks_done"] == 1
    assert loaded["watermark_seconds"] == pytest.approx(25.0)
    assert load_live_transcript(tmp_path / "empty") is None


def test_chunker_passes_is_cancelled_to_engine(tmp_path):
    wav = _write_tone_wav(tmp_path / "a.wav", duration=30.0)
    engine = _FakeChunkEngine()
    transcriber = ChunkedTranscriber(engine=engine,
                                     is_cancelled=lambda: False)
    transcriber.process_file(wav, work_dir=tmp_path / "work")
    assert engine.kwargs and all(
        "is_cancelled" in kw for kw in engine.kwargs)


def test_crash_mid_transcription_finalizes_from_live_file(tmp_path):
    """Raise in the fake engine mid-chunk: the live file holds exactly
    the completed chunks, and a fresh chunker restored from it plus
    finish() produces a complete, duplicate-free transcript."""
    session = tmp_path / "session"
    session.mkdir()
    wav = _write_tone_wav(session / "combined.wav", duration=30.0)

    class FlakyEngine(_FakeChunkEngine):
        def transcribe_file(self, audio_path, **kwargs):
            if len(self.calls) >= 1:
                raise RuntimeError("simulated crash mid-transcription")
            return super().transcribe_file(audio_path, **kwargs)

    flaky = FlakyEngine()
    transcriber = ChunkedTranscriber(engine=flaky)
    first = transcriber.process_new_audio(wav, work_dir=tmp_path / "work")
    assert len(first) == 1
    transcriber.save_live(session)
    # The file grows to 61 s (the meeting went on); the next tick
    # crashes on its first window — as if the worker was killed.
    _write_tone_wav(session / "combined.wav", duration=61.0)
    with pytest.raises(RuntimeError, match="simulated crash"):
        transcriber.process_new_audio(wav, work_dir=tmp_path / "work")
    # The live file on disk still holds exactly the completed chunk.
    live = load_live_transcript(session)
    assert live is not None
    assert live["chunks_done"] == 1
    assert len(live["segments"]) == 3
    assert live["watermark_seconds"] == pytest.approx(25.0)
    # Finalize from it with a working engine: live + tail, no dups.
    restorer = ChunkedTranscriber(engine=_FakeChunkEngine())
    restorer.restore_live_state(
        segments=live["segments"], chunks_done=live["chunks_done"],
        watermark_seconds=live["watermark_seconds"])
    tail = restorer.finish(wav, work_dir=tmp_path / "work")
    assert [round(c.start, 1) for c in tail] == [24.5, 49.0]
    merged = restorer.segments
    assert len(merged) == 8
    keys = [(round(s.start, 1), s.text) for s in merged]
    assert len(keys) == len(set(keys))
    assert [s.start for s in merged] == sorted(s.start for s in merged)
    # The merged segments write as a canonical transcript.
    engine = TranscriptionEngine(model_size="tiny",
                                 model=_FakeModel([]))
    result = TranscriptionResult(
        segments=list(merged), model="tiny", audio_path=wav,
        audio_duration_seconds=61.0, wall_seconds=1.0)
    engine.write_transcript_files(session, result)
    payload = json.loads((session / "transcript.json").read_text())
    assert len(payload["segments"]) == 8
    assert (session / "transcript.txt").exists()
    assert (session / "transcript.srt").exists()


# ---------------------------------------------------------------------- Qt --
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


def _wait_until(qapp, condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    qapp.processEvents()
    return condition()


def _make_session(tmp_path, duration=61.0):
    session = tmp_path / "session"
    session.mkdir(exist_ok=True)
    _write_tone_wav(session / "mic.wav", duration=duration)
    _write_tone_wav(session / "system.wav", duration=duration)
    _write_tone_wav(session / "combined.wav", duration=duration)
    return session


class _FakeCapture:
    """Minimal capture stub: state + pause/resume only."""

    def __init__(self):
        from app.audio.capture import CaptureState
        self.state = CaptureState.RECORDING

    def pause(self):
        from app.audio.capture import CaptureState
        self.state = CaptureState.PAUSED

    def resume(self):
        from app.audio.capture import CaptureState
        self.state = CaptureState.RECORDING


def _make_window(qapp, tmp_path, monkeypatch, session):
    from app.ui.main_window import create_main_window
    monkeypatch.setattr(
        "app.ui.main_window.whisper_model_cached", lambda model: True)
    win = create_main_window(AppConfig(output_root=tmp_path))
    win.session_dir = session
    win.capture = _FakeCapture()
    win._live_engine = _FakeChunkEngine()
    return win


def test_live_poll_appends_segments_and_saves_live_file(
        qapp, tmp_path, monkeypatch):
    session = _make_session(tmp_path)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    try:
        win._start_live_transcription()
        assert win._live_timer.isActive()
        assert win._live_chunker.model == "base"  # configured, not tiny
        win._on_live_poll()
        assert _wait_until(qapp, lambda: not win._live_busy)
        # Two full windows are due at once ([0, 25], [24.5, 49.5]):
        # six segments in the pane (hello/world/overlap per window,
        # the second window's leading overlap copy deduped),
        # watermark at 49.5 s.
        assert win.nearlive_list.count() == 6
        assert "hello" in win.nearlive_list.item(0).text()
        live = load_live_transcript(session)
        assert live is not None
        assert live["chunks_done"] == 2
        assert live["watermark_seconds"] == pytest.approx(49.5)
        assert len(live["segments"]) == 6
        # A second tick with no new audio transcribes nothing.
        win._on_live_poll()
        assert _wait_until(qapp, lambda: not win._live_busy)
        assert load_live_transcript(session)["chunks_done"] == 2
    finally:
        win._live_timer.stop()
        win.close()


def test_live_poller_does_not_advance_while_paused(
        qapp, tmp_path, monkeypatch):
    session = _make_session(tmp_path)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    try:
        win._start_live_transcription()
        win._on_live_poll()
        assert _wait_until(qapp, lambda: not win._live_busy)
        watermark = win._live_chunker.watermark_seconds
        assert watermark == pytest.approx(49.5)
        win.on_pause()
        assert not win._live_timer.isActive()
        win._on_live_poll()  # paused capture: the tick is a no-op
        assert win._live_chunker.watermark_seconds == watermark
        win.on_resume()
        assert win._live_timer.isActive()
        assert win._live_chunker.watermark_seconds == watermark
    finally:
        win._live_timer.stop()
        win.close()


def test_live_download_status_shown_when_model_uncached(
        qapp, tmp_path, monkeypatch):
    session = _make_session(tmp_path)
    from app.ui.main_window import create_main_window
    monkeypatch.setattr(
        "app.ui.main_window.whisper_model_cached", lambda model: False)
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.session_dir = session
        win.capture = _FakeCapture()
        win._live_engine = _FakeChunkEngine()
        win._start_live_transcription()
        label = win.nearlive_label.text()
        assert "Downloading Whisper base" in label
        assert "(~142 MB)" in label
        assert "first run only" in label
    finally:
        win._live_timer.stop()
        win.close()


def test_stop_finalize_merges_live_and_tail(
        qapp, tmp_path, monkeypatch):
    session = _make_session(tmp_path)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    try:
        win._start_live_transcription()
        win._on_live_poll()
        assert _wait_until(qapp, lambda: not win._live_busy)
        assert load_live_transcript(session)["chunks_done"] == 2
        win._finalize_transcription()
        # The tail pass runs in a worker thread now (B1): wait for
        # it instead of reading the files synchronously.
        assert _wait_until(
            qapp, lambda: (session / "transcript.json").exists())
        assert _wait_until(qapp, lambda: not win._finalizing)
        payload = json.loads((session / "transcript.json").read_text())
        # Live chunks (6 segs) + tail window [49.0, 61.0] (2 segs):
        # 8 segments, no duplicates.
        assert len(payload["segments"]) == 8
        keys = [(round(s["start"], 1), s["text"])
                for s in payload["segments"]]
        assert len(keys) == len(set(keys))
        assert [s["start"] for s in payload["segments"]] == sorted(
            s["start"] for s in payload["segments"])
        assert (session / "transcript.txt").exists()
        assert (session / "transcript.srt").exists()
        # The live file served its purpose and is gone.
        assert not (session / TRANSCRIPT_LIVE_FILE).exists()
        assert "Done" in win.transcribe_label.text()
    finally:
        win._live_timer.stop()
        win.close()


def test_stop_finalize_timeout_abandons_stuck_live_worker(
        qapp, tmp_path, monkeypatch):
    """B1: if a live chunk is still transcribing past the Stop
    wait, finalize proceeds from the saved live transcript
    instead of waiting forever."""
    session = _make_session(tmp_path)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    try:
        saved = ChunkedTranscriber(engine=_FakeChunkEngine(), model="base")
        saved.process_new_audio(session / "combined.wav",
                                work_dir=tmp_path / "work")
        saved.save_live(session)
        win._live_finalize_wait_seconds = 0.2
        win._live_busy = True  # a worker stuck in a download/chunk
        win._begin_stop_finalize()
        assert win._stop_finalize_pending is True
        assert _wait_until(
            qapp, lambda: (session / "transcript.json").exists(),
            timeout=10.0)
        assert win._live_abandoned is True
        assert _wait_until(qapp, lambda: not win._finalizing)
        payload = json.loads((session / "transcript.json").read_text())
        assert len(payload["segments"]) == 8
    finally:
        win._live_timer.stop()
        win.close()


def test_stop_finalize_cancel_keeps_live_file(
        qapp, tmp_path, monkeypatch):
    """B1: cancelling the tail finalize writes no transcript.json
    and leaves the live file on disk, so the session stays
    transcribable by hand."""
    session = _make_session(tmp_path)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    try:
        class _SlowChunkEngine(_FakeChunkEngine):
            def transcribe_file(self, audio_path, **kwargs):
                time.sleep(0.5)
                return super().transcribe_file(audio_path, **kwargs)

        win._live_engine = _SlowChunkEngine()
        saved = ChunkedTranscriber(engine=_FakeChunkEngine(), model="base")
        saved.process_new_audio(session / "combined.wav",
                                work_dir=tmp_path / "work")
        saved.save_live(session)
        win._finalize_transcription()
        assert _wait_until(qapp, lambda: win._finalizing)
        assert win.cancel_transcribe_btn.isEnabled()
        time.sleep(0.1)
        qapp.processEvents()
        win.on_cancel_transcribe()
        assert _wait_until(qapp, lambda: not win._finalizing,
                           timeout=10.0)
        assert not (session / "transcript.json").exists()
        assert (session / TRANSCRIPT_LIVE_FILE).exists()
        assert "cancelled" in win.transcribe_label.text().lower()
        assert not win._auto_processing
    finally:
        win._live_timer.stop()
        win.close()


def test_stop_finalize_restores_from_live_file_without_live_chunker(
        qapp, tmp_path, monkeypatch):
    """S6: the crash-recovery restore branch at window level —
    _finalize_transcription with a live file on disk and no
    in-memory chunker finalizes from the saved state."""
    session = _make_session(tmp_path)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    try:
        # A previous run saved its progress, then the in-memory
        # chunker was lost (e.g. the app restarted mid-recording).
        saved = ChunkedTranscriber(engine=_FakeChunkEngine(), model="base")
        saved.process_new_audio(session / "combined.wav",
                                work_dir=tmp_path / "work")
        saved.save_live(session)
        assert win._live_chunker is None
        assert load_live_transcript(session)["chunks_done"] == 2
        win._finalize_transcription()
        assert _wait_until(
            qapp, lambda: (session / "transcript.json").exists())
        assert _wait_until(qapp, lambda: not win._finalizing)
        payload = json.loads((session / "transcript.json").read_text())
        # Live chunks (6 segs) + tail window [49.0, 61.0] (2 segs).
        assert len(payload["segments"]) == 8
        keys = [(round(s["start"], 1), s["text"])
                for s in payload["segments"]]
        assert len(keys) == len(set(keys))
        assert [s["start"] for s in payload["segments"]] == sorted(
            s["start"] for s in payload["segments"])
        assert (session / "transcript.txt").exists()
        assert (session / "transcript.srt").exists()
        # The live file served its purpose and is gone.
        assert not (session / TRANSCRIPT_LIVE_FILE).exists()
        assert "Done" in win.transcribe_label.text()
    finally:
        win._live_timer.stop()
        win.close()


def test_stop_finalize_falls_back_to_full_file_without_live_state(
        qapp, tmp_path, monkeypatch):
    session = _make_session(tmp_path)
    from app.ui.main_window import create_main_window
    import app.ui.main_window as mw

    class FakeFullEngine:
        def __init__(self, model_size="base", **kwargs):
            self.model_size = model_size

        def transcribe_session(self, session_dir, *, initial_prompt=None,
                               progress_callback=None, is_cancelled=None):
            from app.transcription.exports import export_srt, export_txt
            session_dir = Path(session_dir)
            segments = [Segment(0.0, 2.0, "full file hello"),
                        Segment(2.0, 4.0, "world")]
            payload = {"model": self.model_size, "segments": [
                s.to_dict() for s in segments]}
            (session_dir / "transcript.json").write_text(
                json.dumps(payload), encoding="utf-8")
            export_txt(segments, session_dir / "transcript.txt")
            export_srt(segments, session_dir / "transcript.srt")
            return TranscriptionResult(
                segments=segments, model=self.model_size,
                audio_path=session_dir / "combined.wav",
                audio_duration_seconds=61.0, wall_seconds=0.5)

        def unload(self):
            pass

    monkeypatch.setattr(mw, "TranscriptionEngine", FakeFullEngine)
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.session_dir = session
        assert win._live_chunker is None
        assert load_live_transcript(session) is None
        win._finalize_transcription()
        assert _wait_until(
            qapp, lambda: (session / "transcript.json").exists())
        assert _wait_until(qapp, lambda: not win._transcribing)
        payload = json.loads((session / "transcript.json").read_text())
        assert [s["text"] for s in payload["segments"]] == [
            "full file hello", "world"]
    finally:
        win.close()


def test_failed_transcription_unloads_the_model(
        qapp, tmp_path, monkeypatch):
    """Item 3 fix: a failed run must not leave the model resident —
    _TranscribeWorker unloads in a finally, not only on success."""
    import app.ui.main_window as mw
    session = _make_session(tmp_path)
    unloaded: list = []

    class FailingEngine:
        def __init__(self, model_size="base", **kwargs):
            self.model_size = model_size

        def transcribe_session(self, *args, **kwargs):
            raise RuntimeError("simulated model failure")

        def unload(self):
            unloaded.append(True)

    monkeypatch.setattr(mw, "TranscriptionEngine", FailingEngine)
    worker = mw._TranscribeWorker(session, "base")
    errors: list = []
    worker.error.connect(errors.append)
    worker.run()
    assert unloaded == [True]
    assert errors and "simulated model failure" in errors[0]


def test_manual_transcribe_cancel_writes_no_transcript(
        qapp, tmp_path, monkeypatch):
    session = _make_session(tmp_path)
    from app.ui.main_window import create_main_window
    import app.ui.main_window as mw

    class SlowCancellableEngine:
        def __init__(self, model_size="base", **kwargs):
            self.model_size = model_size

        def transcribe_session(self, session_dir, *, initial_prompt=None,
                               progress_callback=None, is_cancelled=None):
            for _ in range(200):
                if is_cancelled is not None and is_cancelled():
                    return TranscriptionResult(
                        segments=[Segment(0.0, 1.0, "partial")],
                        model=self.model_size, cancelled=True)
                time.sleep(0.02)
            raise AssertionError("cancel was never requested")

        def unload(self):
            pass

    monkeypatch.setattr(mw, "TranscriptionEngine", SlowCancellableEngine)
    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        win.session_dir = session
        win.on_transcribe()
        assert win._transcribing
        assert win.cancel_transcribe_btn.isEnabled()
        time.sleep(0.2)
        qapp.processEvents()
        win.on_cancel_transcribe()
        assert _wait_until(qapp, lambda: not win._transcribing)
        assert "cancelled" in win.transcribe_label.text().lower()
        assert not (session / "transcript.json").exists()
    finally:
        win.close()


# ------------------------------------------------- Nov 20 2026 Stop freeze --
# Item 1 regression tests: Stop must never freeze the UI — capture
# stop + mix run in a worker, the abandon wait is cancellable and
# visible, the live-poll snapshot runs in the worker, and Record
# stays disabled until Stop fully settles. Fake engines only.
def _live_thread_stopped(win) -> bool:
    """True when the window's live worker thread has exited (or
    never existed / was already deleted). Closing the window
    while its live thread is still running destroys the QThread
    mid-flight and aborts the process — tests that release a
    stuck worker must wait for this before win.close()."""
    thread = getattr(win, "_live_thread", None)
    if thread is None:
        return True
    try:
        return not thread.isRunning()
    except RuntimeError:  # C++ object already deleted
        return True


def _wait_live_thread_stopped(qapp, win, timeout=10.0) -> None:
    _wait_until(qapp, lambda: _live_thread_stopped(win),
                timeout=timeout)


def _write_minimal_metadata(session: Path) -> None:
    """metadata.json so finalize_session() can mark the session done."""
    (session / "metadata.json").write_text(json.dumps({
        "session_id": session.name, "app_name": "Teams",
        "mode": "system-loopback-fallback", "status": "recording",
        "start_time": "2026-11-08T10:00:00+00:00",
        "sample_rate": 48000, "channels": 1,
        "devices": {}, "warning": None, "files": {}}), encoding="utf-8")


def _write_silence_wav_48k(path: Path, seconds: int) -> None:
    """48 kHz mono silence, written in 1 MB blocks (fast, no numpy)."""
    import wave
    frames = seconds * 48000
    block = b"\x00\x00" * (512 * 1024)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        remaining = frames
        while remaining > 0:
            n = min(remaining, 512 * 1024)
            w.writeframes(block[:n * 2])
            remaining -= n


class _StopFakeCapture:
    """Capture stub with a real stop(): flips state, returns files."""

    def __init__(self, session_dir, duration=61.0):
        from app.audio.capture import CaptureState
        self.session_dir = session_dir
        self.state = CaptureState.RECORDING
        self.duration_seconds = duration

    def stop(self):
        from app.audio.capture import CaptureState
        self.state = CaptureState.STOPPED
        return {"combined": self.session_dir / "combined.wav",
                "mic": self.session_dir / "mic.wav",
                "system": self.session_dir / "system.wav"}


class _BlockingChunkEngine(_FakeChunkEngine):
    """Fake engine that blocks in transcribe_file until released —
    the stuck live worker (first-run download / pathological
    chunk). ``entered`` fires on entry; ``release`` unblocks it
    (the test's finally must always release, or the worker thread
    never exits)."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def transcribe_file(self, audio_path, **kwargs):
        self.entered.set()
        self.release.wait(timeout=60)
        return super().transcribe_file(audio_path, **kwargs)


def _save_live_state(session, tmp_path):
    """A saved live transcript (2 chunks, 6 segments) on disk."""
    saved = ChunkedTranscriber(engine=_FakeChunkEngine(), model="base")
    saved.process_new_audio(session / "combined.wav",
                            work_dir=tmp_path / "work")
    saved.save_live(session)


def test_stop_with_stuck_live_worker_stays_responsive_and_cancel_abandons(
        qapp, tmp_path, monkeypatch):
    """Item 1: with a live chunk stuck, on_stop() returns promptly,
    the 50 ms UI probe keeps firing, Cancel is enabled during the
    wait with a live countdown, and pressing Cancel abandons
    immediately and finalizes from the saved live transcript."""
    from PyQt6.QtCore import QTimer

    session = _make_session(tmp_path)
    _write_minimal_metadata(session)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    blocking = _BlockingChunkEngine()
    probe: list = []
    timer = QTimer()
    timer.setInterval(50)
    timer.timeout.connect(lambda: probe.append(time.monotonic()))
    try:
        win.config.auto_process_recordings = False
        _save_live_state(session, tmp_path)
        win._live_engine = blocking
        win._start_live_transcription()
        win._on_live_poll()
        assert _wait_until(qapp, blocking.entered.is_set, timeout=10.0)
        assert win._live_busy
        timer.start()
        capture = _StopFakeCapture(session)
        win.capture = capture
        started = time.monotonic()
        win.on_stop()
        wall = time.monotonic() - started
        assert wall < 3.0, f"on_stop took {wall:.2f}s with a stuck worker"
        assert "Stopping" in win.status_label.text()
        assert _wait_until(qapp, lambda: win._stop_finalize_pending,
                           timeout=10.0)
        # The UI thread is alive during the wait: the probe fires.
        before = len(probe)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.02)
        assert len(probe) - before >= 10
        # The wait is visible and cancellable.
        assert "Waiting for live chunk" in win.status_label.text()
        assert "/ 60s" in win.status_label.text()
        assert win.cancel_transcribe_btn.isEnabled()
        assert not win.record_btn.isEnabled()
        # The tail must not use the blocking engine: swap in the
        # fast fake before abandoning (the stuck worker still owns
        # the old chunker; its late signals are ignored).
        win._live_engine = _FakeChunkEngine()
        win.on_cancel_transcribe()
        assert not win._stop_finalize_pending
        assert _wait_until(
            qapp, lambda: (session / "transcript.json").exists(),
            timeout=10.0)
        assert _wait_until(qapp, lambda: not win._finalizing,
                           timeout=10.0)
        payload = json.loads((session / "transcript.json").read_text())
        assert len(payload["segments"]) == 8
    finally:
        timer.stop()
        blocking.release.set()
        # The released worker lands (its late signals are ignored)
        # — wait for its thread to exit before destroying the
        # window, or the close aborts mid-flight (see
        # _live_thread_stopped).
        _wait_live_thread_stopped(qapp, win)
        win._live_timer.stop()
        win.close()


def test_stop_returns_promptly_with_long_real_capture(
        qapp, tmp_path, monkeypatch):
    """Item 1: on_stop() returns in milliseconds even for a
    10-minute meeting — the real DualCapture stop (thread joins +
    mix) runs in a worker. A real DualCapture with silence chunk
    sources writes the tracks; the post-stop finalize uses a fake
    full-file engine so no model is ever downloaded."""
    import app.ui.main_window as mw
    from app.audio.capture import CaptureState, DualCapture
    from app.transcription.exports import export_srt, export_txt

    class FakeFullEngine:
        def __init__(self, model_size="base", **kwargs):
            self.model_size = model_size

        def transcribe_session(self, session_dir, *, initial_prompt=None,
                               progress_callback=None, is_cancelled=None):
            session_dir = Path(session_dir)
            segments = [Segment(0.0, 2.0, "full file hello"),
                        Segment(2.0, 4.0, "world")]
            payload = {"model": self.model_size, "segments": [
                s.to_dict() for s in segments]}
            (session_dir / "transcript.json").write_text(
                json.dumps(payload), encoding="utf-8")
            export_txt(segments, session_dir / "transcript.txt")
            export_srt(segments, session_dir / "transcript.srt")
            return TranscriptionResult(
                segments=segments, model=self.model_size,
                audio_path=session_dir / "combined.wav",
                audio_duration_seconds=600.0, wall_seconds=0.5)

        def unload(self):
            pass

    monkeypatch.setattr(mw, "TranscriptionEngine", FakeFullEngine)
    session = create_session(tmp_path, app_name="Teams")

    def silence_chunks():
        chunk = np.zeros(1024, dtype=np.int16)
        for _ in range(28125):  # 10 minutes at 48 kHz
            yield chunk.copy()

    dual = DualCapture(
        session, process_name="ms-teams.exe",
        capture_system_mode="loopback",
        chunk_source_factory=lambda kind: silence_chunks())
    dual.start()
    # Let the capture threads write the 10 minutes of silence.
    assert _wait_until(
        qapp,
        lambda: (session / "mic.wav").stat().st_size > 57_000_000,
        timeout=30.0)
    from app.ui.main_window import create_main_window
    win = create_main_window(AppConfig(
        output_root=tmp_path, auto_process_recordings=False))
    try:
        win.session_dir = session
        win.capture = dual
        started = time.monotonic()
        win.on_stop()
        wall = time.monotonic() - started
        assert wall < 1.0, f"on_stop took {wall:.2f}s for a 10-min meeting"
        assert "Stopping" in win.status_label.text()
        assert _wait_until(
            qapp, lambda: (session / "combined.wav").exists(),
            timeout=10.0)
        assert dual.state == CaptureState.STOPPED
        assert _wait_until(
            qapp, lambda: (session / "transcript.json").exists(),
            timeout=10.0)
    finally:
        # N7: wait for the stop thread to exit before destroying
        # the window, like the live-thread tests do — the
        # transcript landing does not mean the thread has quit.
        _wait_stop_thread_stopped(qapp, win)
        win.close()
        # ~170 MB of 10-minute WAVs: free the 512 MB /tmp tmpfs
        # for the rest of the suite.
        shutil.rmtree(session, ignore_errors=True)


def test_live_poll_snapshot_returns_promptly_on_long_tracks(
        qapp, tmp_path, monkeypatch):
    """Item 1: one _on_live_poll() on 10-minute tracks returns in
    milliseconds — the snapshot copy + mix now run in the worker,
    not synchronously on the UI thread (where a tick cost 0.7–1.9 s
    at 10 min). The tick still lands: the live file is written."""
    session = tmp_path / "session"
    session.mkdir()
    _write_silence_wav_48k(session / "mic.wav", seconds=600)
    _write_silence_wav_48k(session / "system.wav", seconds=600)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    try:
        win._start_live_transcription()
        started = time.monotonic()
        win._on_live_poll()
        wall = time.monotonic() - started
        assert wall < 0.5, f"_on_live_poll took {wall:.2f}s on 10-min tracks"
        assert _wait_until(qapp, lambda: not win._live_busy,
                           timeout=30.0)
        assert load_live_transcript(session) is not None
    finally:
        # The worker's finished slot already ran, but its thread
        # may still be quitting (the snapshot rmtree) — wait for
        # it before destroying the window.
        _wait_live_thread_stopped(qapp, win)
        win._live_timer.stop()
        win.close()
        # ~115 MB of 10-minute WAVs: free the 512 MB /tmp tmpfs.
        shutil.rmtree(session, ignore_errors=True)


def test_record_disabled_until_stop_finalize_settles(
        qapp, tmp_path, monkeypatch):
    """Item 1: Record stays disabled while Stop waits for an
    in-flight live chunk, and is back once the finalize settles
    (here via the short abandon timeout, then the tail)."""
    session = _make_session(tmp_path)
    _write_minimal_metadata(session)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    blocking = _BlockingChunkEngine()
    try:
        win.config.auto_process_recordings = False
        _save_live_state(session, tmp_path)
        win._live_engine = blocking
        win._start_live_transcription()
        win._on_live_poll()
        assert _wait_until(qapp, blocking.entered.is_set, timeout=10.0)
        win.capture = _StopFakeCapture(session)
        win._live_finalize_wait_seconds = 0.5
        win.on_stop()
        assert _wait_until(qapp, lambda: win._stop_finalize_pending,
                           timeout=10.0)
        assert not win.record_btn.isEnabled()
        # The tail must not use the blocking engine (see the Cancel
        # test above): swap in the fast fake while waiting.
        win._live_engine = _FakeChunkEngine()
        assert _wait_until(
            qapp, lambda: (session / "transcript.json").exists(),
            timeout=10.0)
        assert _wait_until(
            qapp,
            lambda: not win._finalizing and win.record_btn.isEnabled(),
            timeout=10.0)
    finally:
        blocking.release.set()
        _wait_live_thread_stopped(qapp, win)
        win._live_timer.stop()
        win.close()


# --------------------------- Nov 20 2026 review fixes (S1/S2/S3/S5/S6) ----
class _BlockingStopCapture(_StopFakeCapture):
    """Stop stub whose stop() blocks until released — the stop
    window stays open long enough for the test to prod it."""

    def __init__(self, session_dir, duration=61.0):
        super().__init__(session_dir, duration=duration)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.pause_calls = 0
        self.resume_calls = 0

    def stop(self):
        self.entered.set()
        self.release.wait(timeout=60)
        return super().stop()

    def pause(self):
        self.pause_calls += 1
        raise RuntimeError("track stopped")

    def resume(self):
        self.resume_calls += 1
        raise RuntimeError("track stopped")


def _stop_thread_stopped(win) -> bool:
    """True when the window's stop worker thread has exited (or
    never existed / was already deleted) — the _stop_thread
    mirror of _live_thread_stopped (N7)."""
    thread = getattr(win, "_stop_thread", None)
    if thread is None:
        return True
    try:
        return not thread.isRunning()
    except RuntimeError:  # C++ object already deleted
        return True


def _wait_stop_thread_stopped(qapp, win, timeout=10.0) -> None:
    _wait_until(qapp, lambda: _stop_thread_stopped(win),
                timeout=timeout)


def test_pause_resume_disabled_during_stop_window(
        qapp, tmp_path, monkeypatch):
    """S1: Pause/Resume stay enabled while _stopping without the
    fix (capture.state is still RECORDING until the worker calls
    capture.stop()), so pressing Pause mid-Stop raised a spurious
    "Pause failed" dialog. With the fix the buttons are dead
    through the stop window and neither handler runs."""
    from PyQt6.QtWidgets import QMessageBox

    session = _make_session(tmp_path)
    _write_minimal_metadata(session)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    capture = _BlockingStopCapture(session, duration=1.0)
    warnings: list = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        lambda *args, **kwargs: warnings.append(args[2] if len(args) > 2
                                               else ""))
    try:
        win.capture = capture
        win.on_stop()
        assert _wait_until(qapp, capture.entered.is_set, timeout=10.0)
        assert win._stopping
        assert not win.pause_btn.isEnabled()
        assert not win.resume_btn.isEnabled()
        win.pause_btn.click()
        win.resume_btn.click()
        qapp.processEvents()
        assert capture.pause_calls == 0
        assert capture.resume_calls == 0
        assert warnings == []
        assert capture.state.name == "RECORDING"
    finally:
        capture.release.set()
        assert _wait_until(qapp, lambda: not win._stopping, timeout=10.0)
        win.close()


def test_stop_finalizes_snapshot_session_despite_swap(
        qapp, tmp_path, monkeypatch):
    """S2: a session swapped in between Stop and the finished
    slot must not redirect the finalize — metadata duration and
    transcript land on the stopped session, and it is restored
    as the loaded one."""
    session = _make_session(tmp_path)
    _write_minimal_metadata(session)
    other = tmp_path / "other"
    other.mkdir()
    _write_minimal_metadata(other)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    capture = _BlockingStopCapture(session, duration=61.0)
    try:
        win.config.auto_process_recordings = False
        _save_live_state(session, tmp_path)
        win.capture = capture
        win.on_stop()
        assert _wait_until(qapp, capture.entered.is_set, timeout=10.0)
        # The post-flip window: another session is loaded before
        # the finished slot runs.
        win.session_dir = other
        capture.release.set()
        assert _wait_until(
            qapp, lambda: (session / "transcript.json").exists(),
            timeout=10.0)
        assert _wait_until(qapp, lambda: not win._finalizing,
                           timeout=10.0)
        assert win.session_dir == session
        metadata = json.loads((session / "metadata.json").read_text())
        assert metadata["status"] == "completed"
        assert metadata["duration_seconds"] == pytest.approx(61.0)
        other_metadata = json.loads(
            (other / "metadata.json").read_text())
        assert other_metadata["status"] == "recording"
        assert not (other / "transcript.json").exists()
        payload = json.loads((session / "transcript.json").read_text())
        assert len(payload["segments"]) == 8
    finally:
        capture.release.set()
        _wait_until(qapp, lambda: not win._finalizing, timeout=10.0)
        win._live_timer.stop()
        win.close()


def test_close_during_stop_window_is_deferred_until_settle(
        qapp, tmp_path, monkeypatch):
    """S3: closing the window while the stop worker runs must not
    destroy the running QThread — the close is deferred, the
    stop settles first (metadata completed), and the window
    closes after."""
    session = _make_session(tmp_path)
    _write_minimal_metadata(session)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    capture = _BlockingStopCapture(session, duration=1.0)
    try:
        win.capture = capture
        win.show()
        qapp.processEvents()
        assert win.isVisible()
        win.on_stop()
        assert _wait_until(qapp, capture.entered.is_set, timeout=10.0)
        assert win._stopping
        win.close()
        qapp.processEvents()
        assert win._close_pending
        assert win.isVisible()
        assert win._stop_thread is not None
        assert win._stop_thread.isRunning()
        capture.release.set()
        assert _wait_until(qapp, lambda: not win.isVisible(),
                           timeout=10.0)
        assert _stop_thread_stopped(win)
        metadata = json.loads((session / "metadata.json").read_text())
        assert metadata["status"] == "completed"
    finally:
        capture.release.set()
        _wait_until(qapp, lambda: not win._stopping, timeout=10.0)
        if win.isVisible():
            win.close()


class _FailingStopCapture(_StopFakeCapture):
    """Stop stub whose stop() raises before flipping state —
    the pre-fix _on_stop_error left Record disabled."""

    def stop(self):
        raise RuntimeError("simulated stop failure")


def test_stop_error_discards_capture_and_record_stays_usable(
        qapp, tmp_path, monkeypatch):
    """S5: on a stop error the failed capture is discarded, so
    Record is usable again (on_record builds a fresh
    DualCapture) — instead of leaving a RECORDING-state capture
    behind with Record disabled and no reset path. The dialog
    says the capture was discarded; the tracks stay on disk."""
    from PyQt6.QtWidgets import QMessageBox

    session = _make_session(tmp_path)
    _write_minimal_metadata(session)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    errors: list = []
    monkeypatch.setattr(
        QMessageBox, "critical",
        lambda *args, **kwargs: errors.append(args[2] if len(args) > 2
                                             else ""))
    try:
        win.capture = _FailingStopCapture(session)
        win.on_stop()
        assert _wait_until(qapp, lambda: win.capture is None,
                           timeout=10.0)
        assert not win._stopping
        assert len(errors) == 1
        assert "discarded" in errors[0]
        assert "untouched" in errors[0]
        assert win.record_btn.isEnabled()
        assert (session / "mic.wav").exists()
        assert (session / "system.wav").exists()
    finally:
        win.close()


def test_stop_abandons_tick_still_copying_tracks(
        qapp, tmp_path, monkeypatch):
    """S6: a live tick started just before Stop — still copying
    the growing tracks when the stop worker closes them — is
    abandoned up-front instead of failing into the live-error
    path: no wait for it, live is not marked failed, and the
    finalize proceeds from the saved live transcript (the tail
    re-covers the tick's windows from the watermark)."""
    import shutil as _shutil

    import app.ui.main_window as mw

    session = _make_session(tmp_path)
    _write_minimal_metadata(session)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    entered = threading.Event()
    release = threading.Event()
    real_copyfile = _shutil.copyfile
    calls: list = []

    def blocking_copyfile(src, dst):
        calls.append(src)
        if len(calls) == 1:
            entered.set()
            release.wait(timeout=60)
        return real_copyfile(src, dst)

    monkeypatch.setattr(mw.shutil, "copyfile", blocking_copyfile)
    try:
        win.config.auto_process_recordings = False
        _save_live_state(session, tmp_path)
        win._start_live_transcription()
        win._on_live_poll()
        assert _wait_until(qapp, entered.is_set, timeout=10.0)
        assert win._live_busy
        assert not win._live_worker.snapshot_done
        win.capture = _StopFakeCapture(session)
        win.on_stop()
        # Abandoned up-front: no wait for the tick, chunker
        # detached, late signals ignored by design.
        assert win._live_abandoned
        assert not win._live_busy
        assert win._live_chunker is None
        assert _wait_until(
            qapp, lambda: (session / "transcript.json").exists(),
            timeout=10.0)
        assert _wait_until(qapp, lambda: not win._finalizing,
                           timeout=10.0)
        assert not win._stop_finalize_pending
        assert not win._live_failed
        payload = json.loads((session / "transcript.json").read_text())
        assert len(payload["segments"]) == 8
    finally:
        release.set()
        # The released tick lands late; its signals are ignored —
        # wait for its thread to exit before destroying the window.
        _wait_live_thread_stopped(qapp, win)
        win._live_timer.stop()
        win.close()


# ------------------------------------------------- Item 7: one model -------
class _RecordingChunker:
    """Fake ChunkedTranscriber that records the model it was given."""

    instances: list = []

    def __init__(self, model=None, on_chunk=None, **kwargs):
        self.model = model
        self.on_chunk = on_chunk
        self.segments = []
        _RecordingChunker.instances.append(self)

    def process_file(self, audio, mic_path=None, system_path=None):
        return []

    def unload(self):
        pass


def test_manual_nearlive_uses_configured_model(qapp, tmp_path, monkeypatch):
    # Nov 20 2026, Item 7: the manual near-live run, its status
    # strings and the Record view's active-model line all name the
    # model actually in use — the configured one, never a hard-coded
    # tiny. Fake chunker only; no model is downloaded.
    import app.ui.main_window as mw

    monkeypatch.setattr(mw, "ChunkedTranscriber", _RecordingChunker)
    _RecordingChunker.instances.clear()
    session = _make_session(tmp_path)
    win = _make_window(qapp, tmp_path, monkeypatch, session)
    try:
        # Fresh install default: base.
        assert win._configured_model() == "base"
        assert (win.live_model_label.text()
                .startswith("Transcribing with Whisper base"))
        assert "provisional" in win.live_model_label.text()
        assert "Whisper base" in win.nearlive_list.toolTip()
        win.on_nearlive()
        qapp.processEvents()
        assert "Whisper base" in win.nearlive_label.text()
        assert _wait_until(qapp, lambda: not win._nearlive_running)
        assert _RecordingChunker.instances[-1].model == "base"
        assert "Whisper base" in win.nearlive_label.text()
        # Switch the configured model to small: the worker, the
        # status strings and the active-model line all follow.
        win.model_combo.setCurrentIndex(
            win.model_combo.findData("small"))
        qapp.processEvents()
        assert win._configured_model() == "small"
        assert (win.live_model_label.text()
                .startswith("Transcribing with Whisper small"))
        assert "Whisper small" in win.nearlive_list.toolTip()
        win.on_nearlive()
        qapp.processEvents()
        assert "Whisper small" in win.nearlive_label.text()
        assert _wait_until(qapp, lambda: not win._nearlive_running)
        assert _RecordingChunker.instances[-1].model == "small"
        assert "Whisper small" in win.nearlive_label.text()
    finally:
        win.close()
