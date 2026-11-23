"""Mixing and level helpers — pure functions, no audio hardware needed.

All mixing normalises to 48 kHz mono int16 (the Phase 1 recording format).
Levels are floats in 0..1 (RMS / peak of a full-scale-normalised signal).
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger("offair.audio.mixer")

TARGET_SAMPLE_RATE = 48000
INT16_MAX = 32767
INT16_MIN = -32768

#: Samples per streaming block in :func:`mix_arrays` / :func:`mix_files`
#: (~21 s at 48 kHz): small enough that the transient buffers stay
#: a few MB even on hour-long meetings.
_MIX_BLOCK_SAMPLES = 1 << 20


# ---------------------------------------------------------------- levels ----
def _as_float_array(samples) -> np.ndarray:
    """Accept int16 / float numpy arrays, lists, or raw int16 bytes."""
    if isinstance(samples, (bytes, bytearray)):
        arr = np.frombuffer(bytes(samples), dtype=np.int16)
        return arr.astype(np.float64) / 32768.0
    arr = np.asarray(samples)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float64) / 32768.0
    return arr.astype(np.float64)


def rms_level(samples) -> float:
    """RMS level in 0..1. Silence -> 0.0, full-scale square wave -> 1.0."""
    arr = _as_float_array(samples)
    if arr.size == 0:
        return 0.0
    value = float(np.sqrt(np.mean(arr ** 2)))
    return max(0.0, min(1.0, value))


def peak_level(samples) -> float:
    """Peak (max absolute) level in 0..1."""
    arr = _as_float_array(samples)
    if arr.size == 0:
        return 0.0
    return max(0.0, min(1.0, float(np.max(np.abs(arr)))))


def rms_level_bytes(data: bytes) -> float:
    """RMS level of raw int16 mono PCM bytes."""
    return rms_level(data)


# ---------------------------------------------------------------- mixing ----
def mix_arrays(mic, system, mic_gain: float = 0.5,
               system_gain: float = 0.5) -> np.ndarray:
    """Mix two int16 mono arrays.

    Shorter input is zero-padded to the longer length; the sum is scaled
    by the per-channel gains (0.5 each by default) and hard-clipped to
    int16 range, so two loud channels cannot overflow.

    Mixed in blocks (float32, never a whole-array float64 copy):
    a 30-minute meeting is ~86M samples per track, and the old
    whole-array float64 path held several GB transiently at Stop
    (Nov 20 2026 Stop-freeze fix — see docs/TRANSCRIPTION.md).
    """
    a = np.asarray(mic).ravel()
    b = np.asarray(system).ravel()
    length = max(a.size, b.size)
    if length == 0:
        return np.zeros(0, dtype=np.int16)
    out = np.empty(length, dtype=np.int16)
    for start in range(0, length, _MIX_BLOCK_SAMPLES):
        stop = min(start + _MIX_BLOCK_SAMPLES, length)
        n = stop - start
        a_block = np.zeros(n, dtype=np.float32)
        b_block = np.zeros(n, dtype=np.float32)
        a_end = min(stop, a.size)
        if start < a_end:
            a_block[:a_end - start] = a[start:a_end].astype(np.float32)
        b_end = min(stop, b.size)
        if start < b_end:
            b_block[:b_end - start] = b[start:b_end].astype(np.float32)
        mixed = a_block * mic_gain + b_block * system_gain
        out[start:stop] = np.clip(mixed, INT16_MIN, INT16_MAX).astype(
            np.int16)
    return out


def _resample_linear(samples: np.ndarray, src_rate: int,
                     dst_rate: int) -> np.ndarray:
    """Cheap linear-interpolation resample (mono float/int array)."""
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / src_rate
    new_size = max(1, int(round(duration * dst_rate)))
    old_x = np.linspace(0.0, 1.0, samples.size)
    new_x = np.linspace(0.0, 1.0, new_size)
    return np.interp(new_x, old_x, samples.astype(np.float64))


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as (mono int16 samples, sample_rate)."""
    try:
        import soundfile as sf  # type: ignore[import-not-found]
    except Exception:
        sf = None  # noqa: F841  (fallback below)
    if sf is not None:
        data, rate = sf.read(str(path), dtype="int16", always_2d=True)
        mono = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)
        return np.asarray(mono, dtype=np.int16), int(rate)
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        if width != 2:
            raise ValueError(f"{path}: only 16-bit WAV supported by fallback reader")
        arr = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            arr = arr.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return arr, int(rate)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf  # type: ignore[import-not-found]
    except Exception:
        sf = None  # noqa: F841
    if sf is not None:
        sf.write(str(path), np.asarray(samples, dtype=np.int16),
                 sample_rate, subtype="PCM_16")
        return
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


def mix_files(mic_path: str | Path, system_path: str | Path,
              output_path: str | Path,
              sample_rate: int = TARGET_SAMPLE_RATE,
              mic_gain: float = 0.5, system_gain: float = 0.5) -> Path:
    """Mix ``mic.wav`` + ``system.wav`` -> ``combined.wav``.

    Each input is normalised to ``sample_rate`` mono int16; unequal
    lengths are zero-padded (see :func:`mix_arrays`). Raises
    FileNotFoundError if either input is missing.

    The normal case (both inputs already ``sample_rate`` mono)
    streams in blocks — int16 in, float32 mix, int16 out — so a
    30-minute meeting mixes in seconds with a few MB of transient
    memory instead of the old whole-file float64 path (~4–12 s
    wall, ~3.3 GB transient RSS at 30 min; Nov 20 2026
    Stop-freeze fix). Inputs that need resampling or
    stereo-to-mono still take the whole-file path below: those
    are rare (a 44.1 kHz RDP endpoint), and correctness beats
    speed there. Wall time is logged either way, so the next
    Stop-freeze report carries a number instead of a feeling.

    Atomic output (Nov 20 2026 review fix): the mix is written
    to a temp file in the same directory and ``os.replace()``d
    into place only on success — a mid-mix failure (disk full,
    corrupt frame) leaves no truncated ``combined.wav`` behind
    for the Stop path to mistake for a complete one, and an
    existing complete output is left untouched.
    """
    started = time.monotonic()
    mic_path, system_path, output_path = Path(mic_path), Path(system_path), Path(output_path)
    if not mic_path.exists():
        raise FileNotFoundError(f"Microphone track not found: {mic_path}")
    if not system_path.exists():
        raise FileNotFoundError(f"System track not found: {system_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-", suffix=".wav",
        dir=str(output_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        if _streamable(mic_path, system_path, sample_rate):
            _mix_files_streaming(mic_path, system_path, tmp_path,
                                 mic_gain=mic_gain,
                                 system_gain=system_gain)
        else:
            _mix_files_whole(mic_path, system_path, tmp_path,
                             sample_rate=sample_rate, mic_gain=mic_gain,
                             system_gain=system_gain)
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        logger.info("mix_files %s + %s -> %s in %.2fs",
                    mic_path.name, system_path.name, output_path.name,
                    time.monotonic() - started)
    return output_path


def _wav_info(path: Path) -> tuple[int, int, int]:
    """``(channels, sample_rate, sampwidth)`` from a WAV header."""
    with wave.open(str(path), "rb") as wav:
        return (wav.getnchannels(), wav.getframerate(),
                wav.getsampwidth())


def _streamable(mic_path: Path, system_path: Path,
                sample_rate: int) -> bool:
    """True when both inputs are already ``sample_rate`` mono WAVs.

    Anything else (a 44.1 kHz RDP endpoint, a stereo track, a
    non-16-bit track) needs resampling / downmixing / conversion,
    which the streaming path does not do.
    """
    try:
        mic_ch, mic_rate, mic_width = _wav_info(mic_path)
        sys_ch, sys_rate, sys_width = _wav_info(system_path)
    except Exception:  # noqa: BLE001 — an unreadable header is not streamable
        return False
    return (mic_ch == 1 and sys_ch == 1
            and mic_width == 2 and sys_width == 2
            and mic_rate == sample_rate and sys_rate == sample_rate)


def _mix_files_streaming(mic_path: Path, system_path: Path,
                         output_path: Path, mic_gain: float,
                         system_gain: float) -> None:
    """Block-streaming mix for 48 kHz mono inputs (see :func:`mix_files`)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (wave.open(str(mic_path), "rb") as mic_wav,
          wave.open(str(system_path), "rb") as sys_wav,
          wave.open(str(output_path), "wb") as out_wav):
        out_wav.setnchannels(1)
        out_wav.setsampwidth(2)
        out_wav.setframerate(mic_wav.getframerate())
        while True:
            mic_frames = mic_wav.readframes(_MIX_BLOCK_SAMPLES)
            sys_frames = sys_wav.readframes(_MIX_BLOCK_SAMPLES)
            if not mic_frames and not sys_frames:
                break
            n = max(len(mic_frames), len(sys_frames)) // 2
            mic_block = np.zeros(n, dtype=np.float32)
            sys_block = np.zeros(n, dtype=np.float32)
            if mic_frames:
                mic_block[:len(mic_frames) // 2] = np.frombuffer(
                    mic_frames, dtype=np.int16).astype(np.float32)
            if sys_frames:
                sys_block[:len(sys_frames) // 2] = np.frombuffer(
                    sys_frames, dtype=np.int16).astype(np.float32)
            mixed = mic_block * mic_gain + sys_block * system_gain
            out_wav.writeframes(np.clip(
                mixed, INT16_MIN, INT16_MAX).astype(np.int16).tobytes())


def _mix_files_whole(mic_path: Path, system_path: Path,
                     output_path: Path, sample_rate: int,
                     mic_gain: float, system_gain: float) -> None:
    """Whole-file mix (resample / downmix inputs — see :func:`mix_files`)."""
    mic, mic_rate = _read_wav(mic_path)
    system, system_rate = _read_wav(system_path)
    mic = _resample_linear(mic, mic_rate, sample_rate).astype(np.int16) \
        if mic_rate != sample_rate else mic
    system = _resample_linear(system, system_rate, sample_rate).astype(np.int16) \
        if system_rate != sample_rate else system
    combined = mix_arrays(mic, system, mic_gain=mic_gain, system_gain=system_gain)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(output_path, combined, sample_rate)
