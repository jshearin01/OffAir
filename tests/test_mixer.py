"""Tests for app.audio.mixer — pure functions, no hardware, no network."""
import wave

import numpy as np
import pytest

from app.audio.mixer import mix_arrays, mix_files, peak_level, rms_level


def _sine(freq=440.0, seconds=0.1, rate=48000, amplitude=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * freq * t) * amplitude * 32767).astype(np.int16)


def _write_wav(path, samples, rate=48000, channels=1):
    if channels == 2:
        stereo = np.column_stack([samples, samples])
        data = stereo.astype(np.int16)
    else:
        data = np.asarray(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(data.tobytes())


def test_rms_silence_is_zero():
    assert rms_level(np.zeros(1000, dtype=np.int16)) == 0.0
    assert rms_level(b"\x00\x00" * 100) == 0.0
    assert rms_level(np.array([], dtype=np.int16)) == 0.0


def test_rms_full_scale_square_wave_is_one():
    assert rms_level(np.full(100, 32767, dtype=np.int16)) == pytest.approx(1.0, abs=0.01)
    assert rms_level(np.full(100, -32768, dtype=np.int16)) == pytest.approx(1.0, abs=0.01)


def test_rms_sine_half_amplitude():
    # RMS of a sine at amplitude A is A / sqrt(2).
    assert rms_level(_sine(amplitude=0.5)) == pytest.approx(0.5 / np.sqrt(2), abs=0.02)


def test_peak_level():
    assert peak_level(np.zeros(10, dtype=np.int16)) == 0.0
    assert peak_level(np.array([0, 16384, -16384], dtype=np.int16)) == pytest.approx(0.5, abs=0.01)


def test_mix_clipping_protection():
    loud = np.full(100, 32767, dtype=np.int16)
    mixed = mix_arrays(loud, loud)  # 0.5 + 0.5 of full scale
    assert mixed.dtype == np.int16
    assert mixed.max() <= 32767
    assert mixed.min() >= -32768
    # 32767*0.5 + 32767*0.5 == 32767 (no overflow wrap-around to negative)
    assert mixed[0] == 32767


def test_mix_unequal_lengths_zero_pads():
    short = np.full(10, 1000, dtype=np.int16)
    long_ = np.full(30, 2000, dtype=np.int16)
    mixed = mix_arrays(short, long_)
    assert len(mixed) == 30
    assert mixed[0] == 1500  # 1000*0.5 + 2000*0.5
    assert mixed[20] == 1000  # 0*0.5 + 2000*0.5 (short side padded with 0)
    assert len(mix_arrays(long_, short)) == 30


def test_mix_empty_inputs():
    assert len(mix_arrays(np.array([], dtype=np.int16),
                          np.array([], dtype=np.int16))) == 0


def test_mix_files_round_trip(tmp_path):
    mic = tmp_path / "mic.wav"
    system = tmp_path / "system.wav"
    out = tmp_path / "combined.wav"
    _write_wav(mic, _sine(amplitude=0.4))
    _write_wav(system, _sine(freq=880.0, amplitude=0.4))
    result = mix_files(mic, system, out)
    assert result == out and out.exists()
    from app.audio.mixer import _read_wav
    samples, rate = _read_wav(out)
    assert rate == 48000
    assert len(samples) == len(_sine(amplitude=0.4))
    assert rms_level(samples) > 0


def test_mix_files_normalises_sample_rate_and_channels(tmp_path):
    mic = tmp_path / "mic.wav"
    system = tmp_path / "system.wav"
    out = tmp_path / "combined.wav"
    _write_wav(mic, _sine(seconds=0.1, rate=24000), rate=24000)
    _write_wav(system, _sine(seconds=0.1, rate=48000), rate=48000, channels=2)
    mix_files(mic, system, out)
    from app.audio.mixer import _read_wav
    samples, rate = _read_wav(out)
    assert rate == 48000
    assert len(samples) == pytest.approx(4800, abs=2)


def test_mix_files_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mix_files(tmp_path / "nope.wav", tmp_path / "also-nope.wav",
                  tmp_path / "out.wav")


def test_mix_files_streams_long_meeting_fast(tmp_path):
    """Nov 20 2026 Stop-freeze fix: mixing a 5-minute meeting must
    stream in blocks, not load whole tracks as float64 — wall time
    well under the old 0.8–1.8 s, and the output is sample-exact
    against the block mix (spot-checked at the head, the middle
    and the tail). Five minutes, not ten, so the test fits the
    512 MB /tmp tmpfs alongside the rest of the suite."""
    import time

    mic = tmp_path / "mic.wav"
    system = tmp_path / "system.wav"
    out = tmp_path / "combined.wav"
    seconds = 300
    rate = 48000
    # Constant tones: exact expected values, no float noise.
    _write_wav(mic, np.full(seconds * rate, 10000, dtype=np.int16))
    _write_wav(system, np.full(seconds * rate, 20000, dtype=np.int16))
    try:
        started = time.monotonic()
        mix_files(mic, system, out)
        wall = time.monotonic() - started
        assert wall < 1.0, f"5-min mix took {wall:.2f}s — not streaming"
        from app.audio.mixer import _read_wav
        samples, out_rate = _read_wav(out)
        assert out_rate == 48000
        assert len(samples) == seconds * rate
        assert samples[0] == 15000
        assert samples[len(samples) // 2] == 15000
        assert samples[-1] == 15000
    finally:
        # ~86 MB of WAVs: free the 512 MB /tmp tmpfs for the rest
        # of the suite instead of leaving them in the tmpdir.
        for path in (mic, system, out):
            path.unlink(missing_ok=True)


def test_mix_files_streams_unequal_lengths(tmp_path):
    """Streaming mix zero-pads the shorter track, like mix_arrays."""
    mic = tmp_path / "mic.wav"
    system = tmp_path / "system.wav"
    out = tmp_path / "combined.wav"
    _write_wav(mic, np.full(48000, 10000, dtype=np.int16))
    _write_wav(system, np.full(96000, 20000, dtype=np.int16))
    mix_files(mic, system, out)
    from app.audio.mixer import _read_wav
    samples, _rate = _read_wav(out)
    assert len(samples) == 96000
    assert samples[0] == 15000
    assert samples[48000] == 10000  # mic ended: 0*0.5 + 20000*0.5
    assert samples[-1] == 10000


def test_mix_files_failure_leaves_no_partial_output(tmp_path, monkeypatch):
    """Nov 20 2026 review fix (S4): a mid-mix failure must not
    leave a truncated combined.wav — the Stop path reports the
    mix "present" from the file existing, and would then try to
    tail-transcribe the partial file. The mix goes to a temp
    file and is only moved into place on success, so a failure
    leaves no output at all (and a previous complete output,
    if one exists, is left untouched)."""
    import app.audio.mixer as mixer

    mic = tmp_path / "mic.wav"
    system = tmp_path / "system.wav"
    out = tmp_path / "combined.wav"
    _write_wav(mic, _sine(seconds=2.0))
    _write_wav(system, _sine(seconds=2.0, freq=880.0))

    real_frombuffer = np.frombuffer
    calls: list = []

    def failing_frombuffer(*args, **kwargs):
        calls.append(1)
        if len(calls) >= 2:
            raise RuntimeError("simulated mid-mix failure")
        return real_frombuffer(*args, **kwargs)

    monkeypatch.setattr(mixer.np, "frombuffer", failing_frombuffer)
    # A previous complete output survives the failed mix.
    sentinel = b"previous complete mix"
    out.write_bytes(sentinel)
    with pytest.raises(RuntimeError,
                       match="simulated mid-mix failure"):
        mix_files(mic, system, out)
    assert out.read_bytes() == sentinel
    # No temp file is left behind either.
    assert list(tmp_path.glob(".combined-*")) == []
    # And with no previous output, a failed mix leaves nothing.
    out.unlink()
    with pytest.raises(RuntimeError,
                       match="simulated mid-mix failure"):
        mix_files(mic, system, out)
    assert not out.exists()
    assert list(tmp_path.glob(".combined-*")) == []

    # The whole-file path (resample / downmix) is atomic the same
    # way: a failure there leaves no partial output either.
    monkeypatch.setattr(mixer.np, "frombuffer", real_frombuffer)
    monkeypatch.setattr(
        mixer, "_resample_linear",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated resample failure")))
    _write_wav(mic, _sine(seconds=0.1, rate=24000), rate=24000)
    with pytest.raises(RuntimeError,
                       match="simulated resample failure"):
        mix_files(mic, system, out)
    assert not out.exists()
    assert list(tmp_path.glob(".combined-*")) == []
