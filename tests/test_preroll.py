"""Pre-roll ring buffer + capture prepend tests.

Pre-roll is opt-in and default OFF (see app/config.py). The buffer is
memory-only: it never writes to disk, and buffered audio reaches a
file only when a recording starts and the buffer is drained into it.
No always-on feeder exists yet — the buffer's ``feed`` seam is what a
feeder will call; these tests prove the buffer, the drain-on-start
hook, and the capture prepend, not live pre-meeting audio.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from app.audio.capture import CaptureState, DualCapture, MicCapture
from app.audio.mixer import _read_wav
from app.audio.preroll import PreRollBuffer
from app.config import (
    PREROLL_CONSENT_NOTE,
    AppConfig,
    config_from_settings,
    load_settings,
    save_settings,
)

RATE = 48000


def _tone(value: int, seconds: float) -> np.ndarray:
    return np.full(int(RATE * seconds), value, dtype=np.int16)


# ---------------------------------------------------------------- buffer ---
def test_buffer_keeps_only_the_last_n_seconds():
    buf = PreRollBuffer(seconds=30.0, sample_rate=RATE)
    # Feed 40 s of distinct values, 1 s at a time.
    for i in range(40):
        buf.feed("mic", _tone(100 + i, 1.0))
    snap = buf.snapshot("mic")
    assert len(snap) == 30 * RATE
    # The surviving audio is the last 30 s: values 110..139.
    assert snap[0] == 110
    assert snap[-1] == 139
    assert buf.duration_seconds("mic") == pytest.approx(30.0)


def test_buffer_snapshot_does_not_consume():
    buf = PreRollBuffer(seconds=30.0, sample_rate=RATE)
    buf.feed("mic", _tone(7, 1.0))
    first = buf.snapshot("mic")
    second = buf.snapshot("mic")
    assert np.array_equal(first, second)
    assert buf.duration_seconds("mic") == pytest.approx(1.0)


def test_buffer_drain_clears_that_track_only():
    buf = PreRollBuffer(seconds=30.0, sample_rate=RATE)
    buf.feed("mic", _tone(7, 1.0))
    buf.feed("system", _tone(9, 1.0))
    drained = buf.drain_all()
    assert set(drained) == {"mic", "system"}
    assert len(drained["mic"]) == RATE
    assert len(drained["system"]) == RATE
    assert buf.duration_seconds("mic") == 0.0
    assert buf.duration_seconds("system") == 0.0
    # A second drain is empty.
    assert buf.drain_all() == {}


def test_buffer_is_memory_only(tmp_path):
    buf = PreRollBuffer(seconds=30.0, sample_rate=RATE)
    before = {p.name for p in tmp_path.iterdir()}
    buf.feed("mic", _tone(7, 5.0))
    buf.feed("system", _tone(9, 5.0))
    _ = buf.snapshot("mic")
    _ = buf.drain_all()
    assert {p.name for p in tmp_path.iterdir()} == before


def test_buffer_accepts_bytes_and_float_chunks():
    buf = PreRollBuffer(seconds=30.0, sample_rate=RATE)
    buf.feed("mic", _tone(1000, 0.5).tobytes())
    buf.feed("mic", np.full(RATE // 2, 0.5, dtype=np.float64))
    snap = buf.snapshot("mic")
    assert len(snap) == RATE
    assert snap[0] == 1000
    # 0.5 float -> 16383 int16 (0.5 * 32767, truncated).
    assert snap[-1] == 16383


def test_buffer_rejects_unknown_track_and_bad_seconds():
    buf = PreRollBuffer(seconds=30.0, sample_rate=RATE)
    with pytest.raises(ValueError, match="unknown pre-roll track"):
        buf.feed("nope", _tone(1, 0.1))
    with pytest.raises(ValueError, match="unknown pre-roll track"):
        buf.snapshot("nope")
    with pytest.raises(ValueError):
        PreRollBuffer(seconds=0)
    with pytest.raises(ValueError):
        PreRollBuffer(seconds=-5)


def test_buffer_empty_snapshot_is_empty_array():
    buf = PreRollBuffer(seconds=30.0, sample_rate=RATE)
    snap = buf.snapshot("mic")
    assert isinstance(snap, np.ndarray)
    assert snap.dtype == np.int16
    assert len(snap) == 0


def test_buffer_clear_and_duration():
    buf = PreRollBuffer(seconds=30.0, sample_rate=RATE)
    buf.feed("mic", _tone(1, 2.0))
    assert buf.duration_seconds("mic") == pytest.approx(2.0)
    buf.clear()
    assert buf.duration_seconds("mic") == 0.0
    assert buf.snapshot("mic").size == 0


# ------------------------------------------------------- capture prepend ---
def _chunks(value: int, n: int = 5, frames: int = 1024):
    chunk = np.full(frames, value, dtype=np.int16)
    for _ in range(n):
        yield chunk.copy()


def test_preroll_prepended_to_wav(tmp_path):
    cap = MicCapture(
        tmp_path / "mic.wav",
        chunk_source=lambda: _chunks(2000))
    cap.set_preroll(_tone(1000, 1.0))
    cap.start()
    time.sleep(0.2)
    out = cap.stop()
    samples, rate = _read_wav(out)
    assert rate == RATE
    # Pre-roll first (48000 samples of 1000), then live chunks (2000).
    assert len(samples) == RATE + 5 * 1024
    assert np.all(samples[:RATE] == 1000)
    assert np.all(samples[RATE:] == 2000)


def test_preroll_off_leaves_capture_byte_identical(tmp_path):
    def _run(name: str) -> bytes:
        cap = MicCapture(
            tmp_path / name,
            chunk_source=lambda: _chunks(2000))
        cap.start()
        # Wait until the capture thread has written every chunk:
        # stopping immediately would race the thread and truncate
        # the file at a timing-dependent length.
        deadline = time.monotonic() + 5.0
        while (cap._writer is None
               or cap._writer.frames_written < 5 * 1024):
            if time.monotonic() > deadline:
                raise AssertionError("capture thread did not drain chunks")
            time.sleep(0.01)
        out = cap.stop()
        return out.read_bytes()

    first = _run("a.wav")
    second = _run("b.wav")
    assert first == second
    # And an explicit empty pre-roll is the same bytes too.
    cap = MicCapture(tmp_path / "c.wav",
                     chunk_source=lambda: _chunks(2000))
    cap.set_preroll(np.zeros(0, dtype=np.int16))
    cap.start()
    deadline = time.monotonic() + 5.0
    while (cap._writer is None
           or cap._writer.frames_written < 5 * 1024):
        if time.monotonic() > deadline:
            raise AssertionError("capture thread did not drain chunks")
        time.sleep(0.01)
    out = cap.stop()
    assert out.read_bytes() == first


def test_preroll_consumed_once(tmp_path):
    cap = MicCapture(tmp_path / "mic.wav",
                     chunk_source=lambda: _chunks(2000))
    cap.set_preroll(_tone(1000, 1.0))
    cap.start()
    time.sleep(0.2)
    cap.stop()
    # Restarting the same capture without setting pre-roll again:
    # the pre-roll was consumed by the first start, not retained.
    cap.start()
    time.sleep(0.2)
    out = cap.stop()
    samples, _ = _read_wav(out)
    assert len(samples) == 5 * 1024
    assert np.all(samples == 2000)
    # A fresh capture with no pre-roll set records no pre-roll.
    cap2 = MicCapture(tmp_path / "mic2.wav",
                      chunk_source=lambda: _chunks(2000))
    cap2.start()
    time.sleep(0.2)
    out2 = cap2.stop()
    samples2, _ = _read_wav(out2)
    assert len(samples2) == 5 * 1024
    assert np.all(samples2 == 2000)


def test_dual_capture_start_preroll(tmp_path):
    capture = DualCapture(
        tmp_path,
        chunk_source_factory=lambda kind: (
            _chunks(2000) if kind == "mic" else _chunks(4000)),
        capture_system_mode="loopback")
    capture.start(preroll_samples={
        "mic": _tone(1000, 1.0),
        "system": _tone(3000, 0.5),
    })
    time.sleep(0.2)
    result = capture.stop()
    mic_samples, _ = _read_wav(result["mic"])
    system_samples, _ = _read_wav(result["system"])
    assert np.all(mic_samples[:RATE] == 1000)
    assert np.all(mic_samples[RATE:] == 2000)
    assert np.all(system_samples[:RATE // 2] == 3000)
    assert np.all(system_samples[RATE // 2:] == 4000)
    assert capture.state == CaptureState.STOPPED


def test_dual_capture_start_without_preroll_unchanged(tmp_path):
    capture = DualCapture(
        tmp_path,
        chunk_source_factory=lambda kind: _chunks(2000),
        capture_system_mode="loopback")
    capture.start()
    time.sleep(0.2)
    result = capture.stop()
    samples, _ = _read_wav(result["mic"])
    assert len(samples) == 5 * 1024
    assert np.all(samples == 2000)


# ---------------------------------------------------------------- config ---
def test_preroll_defaults_off_30_seconds():
    cfg = AppConfig()
    assert cfg.preroll_enabled is False
    assert cfg.preroll_seconds == 30.0


def test_preroll_round_trips(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(AppConfig(preroll_enabled=True, preroll_seconds=60.0),
                  path)
    loaded = load_settings(path)
    assert loaded.preroll_enabled is True
    assert loaded.preroll_seconds == 60.0
    save_settings(AppConfig(), path)
    assert load_settings(path).preroll_enabled is False


def test_preroll_bad_values_fall_back_to_default():
    assert config_from_settings(
        {"preroll_seconds": 5.0}).preroll_seconds == 30.0
    assert config_from_settings(
        {"preroll_seconds": 121.0}).preroll_seconds == 30.0
    assert config_from_settings(
        {"preroll_seconds": "junk"}).preroll_seconds == 30.0
    with pytest.raises(ValueError):
        AppConfig(preroll_seconds=5.0)
    with pytest.raises(ValueError):
        AppConfig(preroll_seconds=121.0)


def test_preroll_consent_note_says_it_captures_before_record():
    note = PREROLL_CONSENT_NOTE
    assert "before you press Record" in note
    assert "memory" in note.lower()
    assert "never written to disk" in note
    assert "discarded" in note


# ---------------------------------------------------------------- feeder ---
# The feeder (app/audio/preroll_feeder.py) is what makes pre-roll
# capture anything: it opens the mic + system streams in shared
# mode on a background thread and feeds PreRollBuffer until the UI
# layer stops it (synchronously, before capture starts). All feeder
# tests use fake stream factories + synthetic audio — no real
# hardware is claimed. The real Windows/sounddevice stream factories
# are exercised only through the Linux fake below; the Windows
# factory has no run on this VM (see docs/PER_APP_CAPTURE.md).

import sys as _sys

from app.audio.preroll_feeder import PreRollFeeder

FEED_RATE = 8000  # small rate keeps the synthetic feeder tests fast
FEED_FRAMES = 800  # 0.1 s per chunk at FEED_RATE


def _feed_tone(value: int, frames: int = FEED_FRAMES) -> np.ndarray:
    return np.full(frames, value, dtype=np.int16)


class _FakeStream:
    """A fake audio stream: an iterable of chunks with a close().

    ``chunks`` may be a finite list (the stream ends) or the string
    ``"infinite"`` (chunks keep coming, distinct values, until the
    stop event is set). ``closed`` records that the feeder released
    the stream; ``opened`` records the factory kwargs it was opened
    with. Both live on the factory's record dict, keyed by track.
    """

    def __init__(self, chunks, stop_event, record):
        self._chunks = chunks
        self._stop_event = stop_event
        self._record = record

    def __iter__(self):
        if self._chunks == "infinite":
            value = 0
            while not self._stop_event.is_set():
                yield _feed_tone(value)
                value += 1
                self._stop_event.wait(0.001)
            return
        yield from self._chunks

    def close(self):
        self._record["closed"] = True


def _fake_factory(records, *, chunks="infinite", fail_tracks=()):
    """Build a stream_factory recording opens/closes per track.

    ``records`` maps track -> {"opened": kwargs, "closed": bool,
    "opens": int}. ``fail_tracks`` names tracks whose open raises.
    """

    def factory(track, *, device_index, chunk_frames, sample_rate,
                stop_event):
        record = records.setdefault(
            track, {"opened": None, "closed": False, "opens": 0})
        record["opens"] += 1
        record["opened"] = {
            "device_index": device_index,
            "chunk_frames": chunk_frames,
            "sample_rate": sample_rate,
        }
        if track in fail_tracks:
            raise RuntimeError(f"fake device open failed ({track})")
        chunks_for_track = (
            [_feed_tone(100 + i) for i in range(10)]
            if chunks == "finite" else chunks)
        return _FakeStream(chunks_for_track, stop_event, record)

    return factory


def _wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _make_feeder(buf, records, **kwargs):
    kwargs.setdefault("sample_rate", FEED_RATE)
    kwargs.setdefault("chunk_frames", FEED_FRAMES)
    # Explicit indices, as the UI passes combo currentData(): this
    # VM enumerates no devices, so the defaults would resolve to
    # nothing and the fake factory would never be called.
    kwargs.setdefault("mic_device_index", 0)
    kwargs.setdefault("loopback_device_index", 1)
    kwargs.setdefault("stream_factory", _fake_factory(records, **{
        k: v for k, v in kwargs.pop("factory_kwargs", {}).items()}))
    return PreRollFeeder(buf, **kwargs)


def test_feeder_feeds_synthetic_chunks_in_order():
    # 0.5 s buffer, 0.1 s chunks: 10 finite chunks keep only the last 5.
    buf = PreRollBuffer(seconds=0.5, sample_rate=FEED_RATE)
    records = {}
    feeder = _make_feeder(buf, records, factory_kwargs={"chunks": "finite"})
    assert feeder.start() is True
    assert _wait_until(lambda: not feeder.is_running)
    snap = buf.snapshot("mic")
    assert len(snap) == FEED_RATE // 2  # 0.5 s kept
    # Surviving chunks are values 105..109, in feed order.
    for i, value in enumerate(range(105, 110)):
        block = snap[i * FEED_FRAMES:(i + 1) * FEED_FRAMES]
        assert np.all(block == value)
    # The system track fed the same finite sequence.
    assert np.array_equal(buf.snapshot("system"), snap)


def test_feeder_stop_before_start_releases_device_and_capture_gets_preroll(tmp_path):
    buf = PreRollBuffer(seconds=30.0, sample_rate=FEED_RATE)
    records = {}
    feeder = _make_feeder(buf, records)
    assert feeder.start() is True
    assert _wait_until(lambda: buf.duration_seconds("mic") >= 0.5)
    feeder.stop()
    assert feeder.is_running is False
    # Both (fake) devices were released synchronously by stop().
    assert records["mic"]["closed"] is True
    assert records["system"]["closed"] is True
    # Drain the buffer and prepend it through DualCapture: the WAV
    # starts with the pre-roll, then the live chunks.
    preroll = buf.drain_all()
    assert set(preroll) == {"mic", "system"}
    capture = DualCapture(
        tmp_path,
        chunk_source_factory=lambda kind: _chunks(2000),
        capture_system_mode="loopback")
    # DualCapture writes at 48 kHz; resample the 8 kHz feeder audio
    # to the pipeline rate for this synthetic end-to-end check.
    from app.audio.mixer import _resample_linear
    preroll_48k = {
        track: _resample_linear(samples, FEED_RATE, RATE).astype(np.int16)
        for track, samples in preroll.items()}
    capture.start(preroll_samples=preroll_48k)
    time.sleep(0.2)
    result = capture.stop()
    mic_samples, _ = _read_wav(result["mic"])
    assert len(mic_samples) > len(preroll_48k["mic"])
    assert np.array_equal(
        mic_samples[:len(preroll_48k["mic"])], preroll_48k["mic"])
    assert np.all(mic_samples[len(preroll_48k["mic"]):] == 2000)


def test_feeder_device_open_failure_leaves_recording_unaffected(tmp_path):
    # Mic open fails, system feeds: pre-roll degrades per track,
    # start() never raises, and a recording still works exactly.
    buf = PreRollBuffer(seconds=30.0, sample_rate=FEED_RATE)
    records = {}
    feeder = _make_feeder(
        buf, records, factory_kwargs={"fail_tracks": ("mic",)})
    assert feeder.start() is True  # system track still feeding
    assert _wait_until(lambda: buf.duration_seconds("system") >= 0.3)
    assert buf.duration_seconds("mic") == 0.0
    feeder.stop()
    capture = DualCapture(
        tmp_path,
        chunk_source_factory=lambda kind: _chunks(2000),
        capture_system_mode="loopback")
    capture.start()
    time.sleep(0.2)
    result = capture.stop()
    samples, _ = _read_wav(result["mic"])
    assert len(samples) == 5 * 1024
    assert np.all(samples == 2000)


def test_feeder_all_devices_fail_degrades_silently(tmp_path):
    buf = PreRollBuffer(seconds=30.0, sample_rate=FEED_RATE)
    records = {}
    feeder = _make_feeder(
        buf, records, factory_kwargs={"fail_tracks": ("mic", "system")})
    # Threads start, then end at once on the failed opens; the
    # failure is async degradation, visible via is_running.
    assert feeder.start() is True
    assert _wait_until(lambda: not feeder.is_running)
    assert buf.drain_all() == {}
    # Recording still works exactly as before.
    cap = MicCapture(tmp_path / "mic.wav",
                     chunk_source=lambda: _chunks(2000))
    cap.start()
    time.sleep(0.2)
    out = cap.stop()
    samples, _ = _read_wav(out)
    assert len(samples) == 5 * 1024
    assert np.all(samples == 2000)


def test_feeder_start_stop_idempotent_and_stop_without_start():
    buf = PreRollBuffer(seconds=30.0, sample_rate=FEED_RATE)
    records = {}
    feeder = _make_feeder(buf, records)
    # stop() without start() is a safe no-op.
    feeder.stop()
    assert feeder.is_running is False
    assert feeder.start() is True
    # A second start() does not open the devices a second time.
    assert feeder.start() is True
    assert records["mic"]["opens"] == 1
    assert records["system"]["opens"] == 1
    # stop() during feeding releases, and is itself idempotent.
    feeder.stop()
    feeder.stop()
    assert records["mic"]["closed"] is True
    assert feeder.is_running is False
    # start() after stop() reopens the devices.
    assert feeder.start() is True
    assert records["mic"]["opens"] == 2
    feeder.stop()
    assert feeder.is_running is False


def test_feeder_stream_error_mid_feed_degrades_silently():
    buf = PreRollBuffer(seconds=30.0, sample_rate=FEED_RATE)

    def flaky_factory(track, *, device_index, chunk_frames, sample_rate,
                     stop_event):
        yield _feed_tone(7)
        yield _feed_tone(8)
        raise RuntimeError("fake stream died mid-feed")

    feeder = PreRollFeeder(buf, sample_rate=FEED_RATE,
                           chunk_frames=FEED_FRAMES,
                           mic_device_index=0, loopback_device_index=1,
                           stream_factory=flaky_factory)
    assert feeder.start() is True
    # The feeder thread ends on its own; the two fed chunks stay.
    assert _wait_until(lambda: not feeder.is_running)
    snap = buf.snapshot("mic")
    assert len(snap) == 2 * FEED_FRAMES
    assert np.all(snap[:FEED_FRAMES] == 7)
    assert np.all(snap[FEED_FRAMES:] == 8)
    feeder.stop()  # safe after the thread already ended


def test_feeder_disabled_leaves_buffer_empty():
    buf = PreRollBuffer(seconds=30.0, sample_rate=FEED_RATE)
    records = {}
    feeder = _make_feeder(buf, records, enabled=False)
    assert feeder.start() is False
    assert feeder.is_running is False
    time.sleep(0.05)
    # The factory was never called: no device was ever opened.
    assert records == {}
    assert buf.duration_seconds("mic") == 0.0
    assert buf.duration_seconds("system") == 0.0
    feeder.stop()


def test_feeder_no_devices_leaves_buffer_empty(monkeypatch):
    import app.audio.preroll_feeder as feeder_mod

    monkeypatch.setattr(feeder_mod, "default_microphone", lambda: None)
    monkeypatch.setattr(feeder_mod, "default_loopback", lambda: None)
    buf = PreRollBuffer(seconds=30.0, sample_rate=FEED_RATE)
    feeder = PreRollFeeder(buf, sample_rate=FEED_RATE,
                           chunk_frames=FEED_FRAMES)
    assert feeder.start() is False
    assert feeder.is_running is False
    assert buf.drain_all() == {}


def test_feeder_uses_explicit_device_indices():
    buf = PreRollBuffer(seconds=30.0, sample_rate=FEED_RATE)
    records = {}
    feeder = _make_feeder(buf, records, mic_device_index=3,
                          loopback_device_index=7)
    assert feeder.start() is True
    assert records["mic"]["opened"]["device_index"] == 3
    assert records["system"]["opened"]["device_index"] == 7
    feeder.stop()


def test_feeder_module_performs_no_file_io():
    import pathlib
    import re

    source = pathlib.Path(
        __import__("app.audio.preroll_feeder", fromlist=["__file__"]).__file__
    ).read_text(encoding="utf-8")
    assert "pathlib" not in source
    assert "import os" not in source
    # No builtin open() call (attribute opens such as audio.open /
    # stream.read are hardware, not files).
    assert re.search(r"(?<![\w.])open\s*\(", source) is None
    assert "write_text" not in source and "write_bytes" not in source


# ------------------------------------------------- Linux default factory --
needs_linux = pytest.mark.skipif(
    _sys.platform == "win32", reason="Linux sounddevice backend")


class _FakeLinuxInputStream:
    """Stand-in for sounddevice.InputStream; records opens/closes."""

    opened: list = []
    closed: int = 0

    def __init__(self, *args, **kwargs):
        type(self).opened.append(dict(kwargs))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        type(self).closed += 1
        return False

    def read(self, frames):
        return np.zeros(frames, dtype=np.int16), False


def _fake_linux_sounddevice():
    import types

    sd = types.ModuleType("sounddevice")
    infos = [
        {"index": 0, "name": "Built-in Microphone", "hostapi": 0,
         "max_input_channels": 2, "max_output_channels": 0,
         "default_samplerate": 48000.0},
        {"index": 1, "name": "Monitor of Built-in Audio Analog Stereo",
         "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0,
         "default_samplerate": 48000.0},
    ]

    def query_devices(kind=None, **_kwargs):
        if kind == "input":
            return dict(infos[0])
        return [dict(info) for info in infos]

    sd.query_devices = query_devices
    sd.InputStream = _FakeLinuxInputStream
    return sd


@needs_linux
def test_feeder_linux_path_uses_sounddevice_enumeration(monkeypatch):
    monkeypatch.setitem(_sys.modules, "sounddevice",
                        _fake_linux_sounddevice())
    _FakeLinuxInputStream.opened = []
    _FakeLinuxInputStream.closed = 0
    buf = PreRollBuffer(seconds=30.0, sample_rate=RATE)
    feeder = PreRollFeeder(buf, sample_rate=RATE, chunk_frames=1024)
    assert feeder.start() is True
    assert _wait_until(lambda: buf.duration_seconds("mic") >= 0.1)
    feeder.stop()
    assert feeder.is_running is False
    # Both tracks opened through the same enumeration the app uses:
    # default mic (index 0) and monitor source (index 1).
    devices_opened = sorted(
        entry["device"] for entry in _FakeLinuxInputStream.opened)
    assert devices_opened == [0, 1]
    assert all(entry["samplerate"] == RATE
               for entry in _FakeLinuxInputStream.opened)
    assert _FakeLinuxInputStream.closed == 2
