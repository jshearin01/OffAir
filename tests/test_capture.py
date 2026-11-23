"""Tests for app.audio.capture — fake chunk sources only, no hardware."""
import time

import numpy as np
import pytest

from app.audio.capture import (
    CaptureState,
    DualCapture,
    MicCapture,
    PerAppCapture,
    WASAPICapture,
)
from app.audio.mixer import _read_wav


def _chunks(n=20, frames=1024, value=1000):
    """A finite fake source: n chunks of constant int16 samples."""
    chunk = np.full(frames, value, dtype=np.int16)
    for _ in range(n):
        yield chunk.copy()


def test_state_machine_start_pause_resume_stop(tmp_path):
    levels = []
    cap = MicCapture(tmp_path / "mic.wav",
                     chunk_source=lambda: _chunks(50),
                     level_callback=levels.append)
    assert cap.state == CaptureState.IDLE
    with pytest.raises(RuntimeError):
        cap.pause()
    with pytest.raises(RuntimeError):
        cap.resume()
    cap.start()
    assert cap.state == CaptureState.RECORDING
    with pytest.raises(RuntimeError):
        cap.start()  # already recording
    time.sleep(0.2)
    cap.pause()
    assert cap.state == CaptureState.PAUSED
    with pytest.raises(RuntimeError):
        cap.pause()
    cap.resume()
    assert cap.state == CaptureState.RECORDING
    out = cap.stop()
    assert cap.state == CaptureState.STOPPED
    assert out.exists()
    samples, rate = _read_wav(out)
    assert rate == 48000
    assert len(samples) > 0
    assert levels and all(0.0 <= lv <= 1.0 for lv in levels)


def test_pause_discards_audio(tmp_path):
    cap = MicCapture(tmp_path / "mic.wav",
                     chunk_source=lambda: _chunks(100))
    cap.start()
    time.sleep(0.1)
    cap.pause()
    # Chunks arriving while paused must not be written: the writer's
    # frame count does not move while paused.
    assert cap._writer is not None
    writer_frames = cap._writer.frames_written
    cap._handle_chunk(np.full(1024, 20000, dtype=np.int16))
    cap._handle_chunk(np.full(1024, 20000, dtype=np.int16))
    assert cap._writer.frames_written == writer_frames
    cap.resume()
    cap.stop()
    assert cap.state == CaptureState.STOPPED


def test_stop_from_idle_produces_valid_empty_wav(tmp_path):
    cap = MicCapture(tmp_path / "mic.wav",
                     chunk_source=lambda: _chunks(0))
    out = cap.stop()
    assert out.exists()
    samples, rate = _read_wav(out)
    assert len(samples) == 0 and rate == 48000


def test_capture_without_source_fails_cleanly_on_non_windows(tmp_path):
    import sys
    if sys.platform == "win32":
        pytest.skip("Hardware path only meaningful on Windows")
    cap = MicCapture(tmp_path / "mic.wav", chunk_source=None)
    with pytest.raises(RuntimeError, match="Windows-only|pyaudiowpatch"):
        cap.start()


def test_per_app_capture_is_honest_fallback(tmp_path):
    cap = PerAppCapture(tmp_path / "system.wav",
                        process_name="ms-teams.exe",
                        chunk_source=lambda: _chunks(5))
    assert cap.mode == "system-loopback-fallback"
    assert "NOT available" in cap.warning or "not running" in cap.warning
    assert "system-wide" in cap.warning
    cap.start()
    cap.stop()
    assert (tmp_path / "system.wav").exists()


def test_dual_capture_produces_three_files(tmp_path):
    session = tmp_path / "session"
    dual = DualCapture(
        session, process_name="ms-teams.exe",
        # Pin the old system-wide loopback path: the default ("auto")
        # selects per-app capture where available (Windows), which
        # needs real WASAPI — see tests/test_perapp_capture.py.
        capture_system_mode="loopback",
        chunk_source_factory=lambda kind: _chunks(10))
    assert dual.mode == "system-loopback-fallback"
    assert dual.warning
    dual.start()
    time.sleep(0.2)
    dual.pause()
    assert dual.state == CaptureState.PAUSED
    dual.resume()
    files = dual.stop()
    assert files["mic"].exists()
    assert files["system"].exists()
    assert files["combined"].exists()
    mic_samples, _ = _read_wav(files["mic"])
    combined_samples, _ = _read_wav(files["combined"])
    assert len(combined_samples) == len(mic_samples)
    assert len(mic_samples) > 0


def test_capture_import_does_not_require_windows_or_pyqt(tmp_path):
    import app.audio.capture as capture_module
    assert capture_module.WASAPICapture is not None
    assert capture_module.MicCapture is not None
