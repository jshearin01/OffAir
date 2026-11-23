"""Regression tests for the Nov 10 2026 native-rate capture fix.

Bug: "Record everything you hear" (system-wide WASAPI loopback) crashed
the frozen 1.0.1 app ~0.2 s after Record on devices whose mix format is
44.1 kHz (the RDP "Remote Audio [Loopback]" endpoint: 2ch / 44100 Hz /
32-bit). The capture opened the stream at a fixed 48 kHz, PortAudio
refused with ``OSError: [Errno -9998] Invalid number of channels``
(-9997 invalid sample rate), and the process then died with native heap
corruption (0xc0000374 in ntdll.dll).

Fix (see ``app/audio/capture.py``): negotiate the stream format with
``is_format_supported`` *before* ``open`` — synchronously in
``start()``, so a failure is a clean Python error — open at the
device's native rate, and resample to the 48 kHz pipeline rate in
``_handle_chunk``. The WAVs stay 48 kHz mono; the native rate is
recorded in the session metadata.

These tests run on Linux: a fake ``pyaudiowpatch`` module is injected
into ``sys.modules`` and the Windows-only gate in ``_require_hardware``
is run with ``sys.platform`` patched to ``"win32"`` (scoped to that
one call — see the ``fake_pyaudio`` fixture). No real audio hardware
is touched.
"""
import sys
import time
import types

import numpy as np
import pytest

from app.audio.capture import (
    CaptureState,
    DualCapture,
    MicCapture,
    SystemCapture,
    _StreamingResampler,
    negotiate_stream_format,
)
from app.audio.mixer import _read_wav

#: Fake device indices: 0 = microphone, 5 = loopback ("Remote Audio").
MIC_INDEX = 0
LOOPBACK_INDEX = 5


class FakeStream:
    """Stand-in for a PortAudio stream.

    Returns a continuous 440 Hz sine at the stream's own rate,
    stereo-interleaved when ``channels == 2``. ``read`` is throttled
    (~10 ms) so the capture thread runs near real time instead of
    spinning through gigabytes of fake audio.
    """

    def __init__(self, channels: int, rate: int, frames_per_buffer: int):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.read_calls = 0
        self.stopped = False
        self.closed = False
        self._phase = 0  # samples generated so far

    def read(self, frames, exception_on_overflow=False):
        t = (np.arange(frames) + self._phase) / self.rate
        mono = (0.5 * 32767
                * np.sin(2 * np.pi * 440.0 * t)).astype(np.int16)
        self._phase += frames
        self.read_calls += 1
        time.sleep(0.01)
        if self.channels == 2:
            return np.column_stack([mono, mono]).astype(np.int16).tobytes()
        return mono.tobytes()

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakePyAudio:
    """Stand-in for ``pyaudiowpatch.PyAudio``.

    ``devices`` maps an index to an info dict with ``maxInputChannels``,
    ``defaultSampleRate``, ``supported_rates`` and
    ``supported_channels``. ``open`` records its kwargs and either
    returns a :class:`FakeStream` or raises ``open_raises``.
    """

    instances: list = []

    def __init__(self, devices, open_raises=None):
        self.devices = devices
        self.open_raises = open_raises
        self.opened_kwargs: dict | None = None
        self.stream: FakeStream | None = None
        self.terminated = False
        self.is_format_supported_calls: list = []
        type(self).instances.append(self)

    def get_device_info_by_index(self, index):
        return dict(self.devices[index])

    def get_default_input_device_info(self):
        return dict(self.devices["default"])

    def get_format_from_width(self, width):
        assert width == 2
        return 8  # paInt16

    def is_format_supported(self, rate, input_device=None,
                            input_channels=None, input_format=None):
        self.is_format_supported_calls.append(
            (rate, input_device, input_channels))
        info = self.devices[input_device]
        if (rate in info["supported_rates"]
                and input_channels in info["supported_channels"]):
            return True
        raise ValueError(f"Invalid sample rate ({rate})")

    def open(self, **kwargs):
        self.opened_kwargs = dict(kwargs)
        if self.open_raises is not None:
            raise self.open_raises
        self.stream = FakeStream(kwargs["channels"], kwargs["rate"],
                                 kwargs["frames_per_buffer"])
        return self.stream

    def terminate(self):
        self.terminated = True


def _devices(*, loopback_rate=44100, loopback_rates=(44100,),
             loopback_channels=2, mic_rate=48000, mic_rates=(48000, 44100)):
    """Fake device table: a mic (index 0) and a loopback device (index 5)."""
    return {
        MIC_INDEX: {
            "index": MIC_INDEX, "name": "Built-in Microphone",
            "maxInputChannels": 2, "defaultSampleRate": float(mic_rate),
            "supported_rates": set(mic_rates),
            "supported_channels": {1, 2},
        },
        LOOPBACK_INDEX: {
            "index": LOOPBACK_INDEX, "name": "Remote Audio [Loopback]",
            "maxInputChannels": loopback_channels,
            "defaultSampleRate": float(loopback_rate),
            "supported_rates": set(loopback_rates),
            "supported_channels": {1, 2},
        },
        "default": {
            "index": MIC_INDEX, "name": "Built-in Microphone",
            "maxInputChannels": 2, "defaultSampleRate": float(mic_rate),
            "supported_rates": set(mic_rates),
            "supported_channels": {1, 2},
        },
    }


@pytest.fixture
def fake_pyaudio(monkeypatch):
    """Inject a fake pyaudiowpatch module; return a configurator.

    ``configure(devices, open_raises=None)`` (re)builds the fake for
    one test. ``WASAPICapture._require_hardware`` is wrapped so the
    Windows-only gate inside it sees ``sys.platform == "win32"`` —
    scoped to that one call, because patching ``sys.platform`` for the
    whole test would also make ``soundfile`` take its Windows
    (``sf_wchar_open``) path, which this Linux libsndfile lacks.
    The capture thread itself never checks the platform.
    """
    from unittest import mock

    import app.audio.capture as capture_mod

    orig_require = capture_mod.WASAPICapture._require_hardware

    def _require_win32(self):
        with mock.patch.object(sys, "platform", "win32"):
            return orig_require(self)

    monkeypatch.setattr(capture_mod.WASAPICapture, "_require_hardware",
                        _require_win32)
    state = {"devices": _devices(), "open_raises": None}
    module = types.ModuleType("pyaudiowpatch")
    module.paInt16 = 8

    def _factory():
        return FakePyAudio(state["devices"],
                           open_raises=state["open_raises"])

    module.PyAudio = _factory
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", module)

    def configure(devices=None, open_raises=None):
        FakePyAudio.instances.clear()
        if devices is not None:
            state["devices"] = devices
        state["open_raises"] = open_raises

    configure._state = state
    return configure


def _zero_crossing_hz(samples, rate):
    """Estimate a sine's frequency from its zero crossings."""
    s = np.asarray(samples)
    if s.size < 10:
        return 0.0
    crossings = int(np.sum((s[:-1] < 0) != (s[1:] < 0)))
    return crossings / (2 * (len(s) / rate))


# ------------------------------------------------------- negotiation ------
def test_negotiate_prefers_requested_rate_when_supported(fake_pyaudio):
    audio = FakePyAudio(_devices())
    rate, channels = negotiate_stream_format(
        audio, device_index=LOOPBACK_INDEX, requested_rate=48000,
        requested_channels=1, kind="system")
    # The loopback fake only supports 44100, so the requested 48 kHz
    # must be refused and the native rate picked — never the reverse.
    assert (rate, channels) == (44100, 2)


def test_negotiate_keeps_48000_on_48k_device(fake_pyaudio):
    audio = FakePyAudio(_devices(loopback_rate=48000,
                                 loopback_rates=(48000, 44100)))
    rate, channels = negotiate_stream_format(
        audio, device_index=LOOPBACK_INDEX, requested_rate=48000,
        requested_channels=1, kind="system")
    assert (rate, channels) == (48000, 2)


def test_negotiate_raises_when_nothing_supported(fake_pyaudio):
    audio = FakePyAudio(_devices(loopback_rates=()))
    with pytest.raises(RuntimeError, match="No supported input format"):
        negotiate_stream_format(
            audio, device_index=LOOPBACK_INDEX, requested_rate=48000,
            requested_channels=1, kind="system")
    assert audio.opened_kwargs is None  # open must never be attempted


# ------------------------------------------------------------ resampler ----
def test_streaming_resampler_frame_counts_and_pitch():
    """44100 Hz in -> 48000 Hz out, same 440 Hz tone, no drift."""
    resampler = _StreamingResampler(44100, 48000)
    t = np.arange(44100) / 44100
    sine = (0.5 * 32767 * np.sin(2 * np.pi * 440.0 * t)).astype(np.int16)
    out_chunks = []
    for i in range(0, len(sine), 1024):
        out_chunks.append(resampler.push(sine[i:i + 1024]))
    out = np.concatenate(out_chunks)
    # One second of 44.1 kHz in must come out as one second of 48 kHz.
    assert abs(len(out) - 48000) <= 3
    assert abs(_zero_crossing_hz(out, 48000) - 440.0) < 5.0


def test_streaming_resampler_downsample_and_empty_push():
    resampler = _StreamingResampler(48000, 44100)
    assert resampler.push(np.zeros(0, dtype=np.int16)).size == 0
    # A single sample cannot produce output yet — it is buffered, not lost.
    assert resampler.push(np.full(1, 1000, dtype=np.int16)).size == 0
    out = resampler.push(np.full(48000, 1000, dtype=np.int16))
    assert abs(len(out) - 44100) <= 3


# ------------------------------------------------- loopback capture (44.1k) --
def test_loopback_44100_device_opens_at_native_rate_and_wav_stays_48k(
        tmp_path, fake_pyaudio):
    """The reproduced bug: a 44.1 kHz-only loopback device must open at
    44100 (never 48000) and the WAV must still be 48 kHz mono with the
    pitch intact (440 Hz in -> 440 Hz out)."""
    fake_pyaudio()
    cap = SystemCapture(tmp_path / "system.wav",
                        device_index=LOOPBACK_INDEX)
    cap.start()
    time.sleep(0.3)
    out = cap.stop()
    assert cap.state == CaptureState.STOPPED
    assert cap.stream_sample_rate == 44100
    assert cap._resampler is not None
    opened = [inst.opened_kwargs for inst in FakePyAudio.instances
              if inst.opened_kwargs is not None]
    assert len(opened) == 1
    assert opened[0]["rate"] == 44100
    assert opened[0]["channels"] == 2  # native channel count (Nov 8 fix)
    assert opened[0]["input_device_index"] == LOOPBACK_INDEX
    samples, rate = _read_wav(out)
    assert rate == 48000
    assert len(samples) > 0
    # Frame counts: the fake runs faster than real time (10 ms per
    # 1024-frame read), so compare against what the stream actually
    # delivered, resampled 44100 -> 48000.
    stream = next(inst.stream for inst in FakePyAudio.instances
                  if inst.stream is not None)
    expected = stream.read_calls * 1024 * 48000 / 44100
    assert abs(len(samples) - expected) <= 4
    # And the tone survives the resample at its original pitch.
    assert abs(_zero_crossing_hz(samples, 48000) - 440.0) < 25.0


def test_loopback_48k_device_opens_at_48000_without_resampler(
        tmp_path, fake_pyaudio):
    fake_pyaudio(_devices(loopback_rate=48000,
                          loopback_rates=(48000, 44100)))
    cap = SystemCapture(tmp_path / "system.wav",
                        device_index=LOOPBACK_INDEX)
    cap.start()
    time.sleep(0.2)
    cap.stop()
    assert cap.stream_sample_rate == 48000
    assert cap._resampler is None
    opened = [inst.opened_kwargs for inst in FakePyAudio.instances
              if inst.opened_kwargs is not None]
    assert opened[0]["rate"] == 48000


def test_no_supported_format_raises_cleanly_before_open(
        tmp_path, fake_pyaudio):
    """Pre-flight failure is a synchronous RuntimeError from start();
    open is never called and the capture stays IDLE."""
    fake_pyaudio(_devices(loopback_rates=()))
    cap = SystemCapture(tmp_path / "system.wav",
                        device_index=LOOPBACK_INDEX)
    with pytest.raises(RuntimeError, match="No supported input format"):
        cap.start()
    assert cap.state == CaptureState.IDLE
    assert all(inst.opened_kwargs is None
               for inst in FakePyAudio.instances)


def test_open_failure_after_preflight_is_clean_python_error(
        tmp_path, fake_pyaudio):
    """If open still fails after a passing pre-flight (e.g. the device
    vanished between the two calls), the capture thread records a
    clean ERROR state, tears down, and the process survives — no
    exception escapes the thread."""
    fake_pyaudio(open_raises=OSError(
        "[Errno -9998] Invalid number of channels"))
    cap = SystemCapture(tmp_path / "system.wav",
                        device_index=LOOPBACK_INDEX)
    cap.start()  # pre-flight passes (44100 is supported); open fails
    time.sleep(0.3)
    assert cap.state == CaptureState.ERROR
    assert "Invalid number of channels" in (cap.error or "")
    out = cap.stop()
    assert cap.state == CaptureState.STOPPED
    assert out.exists()
    samples, rate = _read_wav(out)
    assert rate == 48000 and len(samples) == 0  # valid empty WAV
    # Teardown ran: the stream was never created, terminate() was called.
    assert any(inst.terminated for inst in FakePyAudio.instances)
    # The failing open was attempted exactly once (by the capture
    # thread's PyAudio instance); the pre-flight instance never opens.
    attempted = [inst.opened_kwargs for inst in FakePyAudio.instances
                 if inst.opened_kwargs is not None]
    assert len(attempted) == 1 and attempted[0]["rate"] == 44100


# ------------------------------------------------------------- mic path ----
def test_mic_44100_device_negotiates_and_resamples(tmp_path, fake_pyaudio):
    """The mic path had the same fixed-rate assumption: a 44.1 kHz-only
    default mic must open at 44100 and still produce a 48 kHz WAV."""
    fake_pyaudio(_devices(mic_rate=44100, mic_rates=(44100,)))
    cap = MicCapture(tmp_path / "mic.wav", device_index=None)
    cap.start()
    time.sleep(0.3)
    out = cap.stop()
    assert cap.state == CaptureState.STOPPED
    assert cap.stream_sample_rate == 44100
    opened = [inst.opened_kwargs for inst in FakePyAudio.instances
              if inst.opened_kwargs is not None]
    assert opened[0]["rate"] == 44100
    assert opened[0]["channels"] == 1  # mic keeps the requested mono
    samples, rate = _read_wav(out)
    assert rate == 48000 and len(samples) > 0
    stream = next(inst.stream for inst in FakePyAudio.instances
                  if inst.stream is not None)
    expected = stream.read_calls * 1024 * 48000 / 44100
    assert abs(len(samples) - expected) <= 4
    assert abs(_zero_crossing_hz(samples, 48000) - 440.0) < 25.0


# ----------------------------------------------------------- dual capture --
def test_dual_capture_loopback_failure_stops_mic_and_raises(
        tmp_path, fake_pyaudio):
    """A loopback device with no usable format fails DualCapture.start()
    synchronously: the mic is stopped and the error propagates (the UI
    shows it) instead of recording on with a dead system track."""
    fake_pyaudio(_devices(loopback_rates=()))
    dual = DualCapture(tmp_path, process_name="ms-teams.exe",
                       loopback_device_index=LOOPBACK_INDEX,
                       capture_system_mode="loopback")
    with pytest.raises(RuntimeError, match="No supported input format"):
        dual.start()
    assert dual.mic.state == CaptureState.STOPPED
    assert dual.mic._thread is None or not dual.mic._thread.is_alive()
    assert dual.system.state == CaptureState.IDLE
    # The mic opened (and was stopped); the loopback device was never
    # opened — the pre-flight refused it before open was attempted.
    assert all(kwargs is None
               or kwargs["input_device_index"] != LOOPBACK_INDEX
               for inst in FakePyAudio.instances
               for kwargs in [inst.opened_kwargs])

# ------------------------------------------- concurrent PyAudio init (race) --
def test_dual_capture_pyaudio_construction_is_serialized(
        tmp_path, fake_pyaudio, monkeypatch):
    """Regression: Nov 10 2026 native crash (0xc0000005).

    ``DualCapture.start()`` calls ``mic.start()`` (which spawns the
    mic thread and returns at once) and then ``system.start()``, whose
    ``_require_hardware`` constructs a ``PyAudio`` on the calling
    thread. Without serialisation the mic thread's ``PyAudio()`` in
    ``_run_hardware`` and the calling thread's ``PyAudio()`` run
    concurrently inside ``Pa_Initialize`` — which is not thread-safe
    and crashes the whole process with an access violation (reproduced
    twice on real Windows; the fake's ``__init__`` sleeps 0.1 s to
    widen the race window the same way the real initialiser does).

    The fix serialises every ``PyAudio()`` construction through
    ``PYAUDIO_INIT_LOCK``: at most one ``__init__`` may be in flight
    at any time. This test fails on the old code (max 2) and passes
    on the fixed code (max 1) — while still proving both tracks
    actually recorded.
    """
    import threading

    state = {"in_init": 0, "max_in_init": 0, "lock": threading.Lock()}
    orig_init = FakePyAudio.__init__

    def _tracked_init(self, *args, **kwargs):
        with state["lock"]:
            state["in_init"] += 1
            state["max_in_init"] = max(state["max_in_init"],
                                       state["in_init"])
        time.sleep(0.1)  # widen the race window
        try:
            orig_init(self, *args, **kwargs)
        finally:
            with state["lock"]:
                state["in_init"] -= 1

    monkeypatch.setattr(FakePyAudio, "__init__", _tracked_init)
    fake_pyaudio()
    dual = DualCapture(tmp_path, process_name="ms-teams.exe",
                       mic_device_index=MIC_INDEX,
                       loopback_device_index=LOOPBACK_INDEX,
                       capture_system_mode="loopback")
    dual.start()
    time.sleep(0.3)
    result = dual.stop()
    assert dual.mic.state == CaptureState.STOPPED
    assert dual.system.state == CaptureState.STOPPED
    # Both tracks really recorded (the test is not vacuous: a start
    # that raised before either thread ran would also show max 1).
    assert result["mic"].exists() and result["system"].exists()
    mic_samples, _ = _read_wav(result["mic"])
    system_samples, _ = _read_wav(result["system"])
    assert len(mic_samples) > 0 and len(system_samples) > 0
    assert state["max_in_init"] == 1
