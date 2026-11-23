"""Simple "You vs Remote" diarization — no token, no model, fast.

PRD Phase 3 simple mode. The idea is the one the dual-channel Phase 1
capture makes possible: in a real call ``mic.wav`` holds (almost) only
your voice and ``system.wav`` holds (almost) only the remote side, so
which track has energy during a transcript segment tells you who spoke.

For each transcript segment we compute the mean RMS energy of each
track over the segment's time window and compare:

* mic clearly louder  -> ``"You"``
* system clearly louder -> ``"Remote"``
* both near silence -> ``"Silence"`` (kept, so a segment is never
  silently mislabelled as a speaker)
* neither clearly louder -> ``"Mixed"`` (both tracks active — e.g.
  crosstalk, or the system track carrying your own voice back)

"Clearly louder" means a dominance ratio of 1.5x by default
(``dominance_ratio``). Everything is a pure function over sample
arrays, so it is testable without audio hardware, pyannote, or torch.

Single-track fallback (:func:`label_segments_single_track`): when a
session has only one audio track there is nothing to compare, so we
split the track into speech turns at silence gaps (>= ``gap_seconds``)
and alternate generic labels ``SPEAKER_00`` / ``SPEAKER_01``. This is
explicitly approximate — it cannot know who is who, it only knows
that the speaker changed after a pause — and the returned method
string says so. It exists so a single-track recording still gets
some structure, not so it can be mistaken for real diarization.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("offair.diarization.simple")

YOU = "You"
REMOTE = "Remote"
SILENCE = "Silence"
MIXED = "Mixed"

# RMS below this (on a 0..1 float scale) counts as "no speech in this
# track". 0.005 is well above digital silence and well below normal
# conversational speech at 48 kHz int16 (typical RMS 0.02–0.2).
DEFAULT_SILENCE_RMS = 0.005
DEFAULT_DOMINANCE_RATIO = 1.5
DEFAULT_FRAME_SECONDS = 0.05
DEFAULT_GAP_SECONDS = 0.8


@dataclass
class DiarizedSegment:
    """One transcript segment with a speaker label attached."""

    start: float
    end: float
    speaker: str
    text: str = ""
    # Mean RMS of each track over the segment window (0..1). Useful for
    # debugging a label and for the manual test report.
    mic_rms: float | None = None
    system_rms: float | None = None

    def to_dict(self) -> dict:
        data = {"start": round(self.start, 3), "end": round(self.end, 3),
                "speaker": self.speaker, "text": self.text}
        if self.mic_rms is not None:
            data["mic_rms"] = round(self.mic_rms, 5)
        if self.system_rms is not None:
            data["system_rms"] = round(self.system_rms, 5)
        return data


@dataclass
class SimpleDiarizationResult:
    """Result of a simple-mode run."""

    segments: list[DiarizedSegment] = field(default_factory=list)
    mode: str = "simple"
    # e.g. "dual-channel-energy" or "single-track-energy (approximate)".
    method: str = "dual-channel-energy"
    audio_duration_seconds: float | None = None

    def to_dict(self) -> dict:
        return {"mode": self.mode, "method": self.method,
                "audio_duration_seconds": self.audio_duration_seconds,
                "segments": [s.to_dict() for s in self.segments]}


def load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV as float32 mono in -1..1, plus its sample rate.

    Uses soundfile when available; falls back to stdlib ``wave`` for
    plain PCM16 WAVs. Multi-channel files are mixed down to mono.
    Raises FileNotFoundError / ValueError on missing or unreadable files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        import soundfile as sf
        data, rate = sf.read(str(path), always_2d=True, dtype="float32")
        return data.mean(axis=1).astype(np.float32), int(rate)
    except ImportError:
        pass
    except Exception as exc:  # soundfile present but cannot read it
        raise ValueError(f"Cannot read audio file {path}: {exc}") from exc
    import wave
    try:
        with wave.open(str(path), "rb") as wav:
            n_channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            rate = wav.getframerate()
            raw = wav.readframes(wav.getnframes())
    except Exception as exc:
        raise ValueError(f"Cannot read audio file {path}: {exc}") from exc
    if sampwidth != 2:
        raise ValueError(f"Unsupported sample width {sampwidth} in {path} "
                         "(stdlib fallback handles PCM16 only)")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples.astype(np.float32), int(rate)


def frame_rms(samples: np.ndarray, sample_rate: int,
              frame_seconds: float = DEFAULT_FRAME_SECONDS) -> np.ndarray:
    """RMS energy per frame (one float per ``frame_seconds`` of audio)."""
    if samples.size == 0 or sample_rate <= 0:
        return np.zeros(0, dtype=np.float32)
    frame_len = max(1, int(sample_rate * frame_seconds))
    n_frames = int(np.ceil(samples.size / frame_len))
    padded = np.zeros(n_frames * frame_len, dtype=np.float32)
    padded[:samples.size] = samples
    frames = padded.reshape(n_frames, frame_len)
    return np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)).astype(np.float32)


def mean_rms_in_window(samples: np.ndarray, sample_rate: int,
                       start: float, end: float) -> float:
    """Mean RMS of ``samples`` between ``start`` and ``end`` seconds."""
    if samples.size == 0 or end <= start:
        return 0.0
    i0 = max(0, int(start * sample_rate))
    i1 = min(samples.size, int(end * sample_rate))
    if i1 <= i0:
        return 0.0
    window = samples[i0:i1].astype(np.float64)
    return float(np.sqrt((window ** 2).mean()))


def label_window(mic_rms: float, system_rms: float, *,
                 silence_rms: float = DEFAULT_SILENCE_RMS,
                 dominance_ratio: float = DEFAULT_DOMINANCE_RATIO) -> str:
    """Label one time window from the two tracks' RMS energies.

    The dominance comparison uses a ratio with a small tolerance, so
    a window exactly at the boundary (e.g. 0.15 vs 0.10 at a 1.5x
    ratio) counts as dominant instead of falling through on float
    rounding (1.5 * 0.10 is 0.15000000000000002 in binary).
    """
    if mic_rms < silence_rms and system_rms < silence_rms:
        return SILENCE
    eps = 1e-9
    if mic_rms / max(system_rms, eps) >= dominance_ratio - 1e-9:
        return YOU
    if system_rms / max(mic_rms, eps) >= dominance_ratio - 1e-9:
        return REMOTE
    return MIXED


def label_segments_dual(segments: list, mic_samples: np.ndarray,
                        mic_rate: int, system_samples: np.ndarray,
                        system_rate: int, *,
                        silence_rms: float = DEFAULT_SILENCE_RMS,
                        dominance_ratio: float = DEFAULT_DOMINANCE_RATIO
                        ) -> list[DiarizedSegment]:
    """Label transcript segments (``.start``/``.end``/``.text``) by track energy."""
    labelled: list[DiarizedSegment] = []
    for seg in segments:
        mic = mean_rms_in_window(mic_samples, mic_rate, seg.start, seg.end)
        sys = mean_rms_in_window(system_samples, system_rate,
                                 seg.start, seg.end)
        labelled.append(DiarizedSegment(
            start=float(seg.start), end=float(seg.end),
            speaker=label_window(mic, sys, silence_rms=silence_rms,
                                 dominance_ratio=dominance_ratio),
            text=getattr(seg, "text", ""), mic_rms=mic, system_rms=sys))
    return labelled


def _speech_intervals(rms: np.ndarray, frame_seconds: float,
                      silence_rms: float, gap_seconds: float
                      ) -> list[tuple[float, float]]:
    """Contiguous speech intervals from per-frame RMS.

    Frames above ``silence_rms`` are speech; gaps shorter than
    ``gap_seconds`` are bridged (a breath is not a turn change).
    """
    speech = rms >= silence_rms
    intervals: list[tuple[float, float]] = []
    start: int | None = None
    last_speech = -1
    for i, is_speech in enumerate(speech):
        if is_speech:
            if start is None:
                start = i
            last_speech = i
        elif start is not None and (i - last_speech) * frame_seconds >= gap_seconds:
            intervals.append((start * frame_seconds,
                              (last_speech + 1) * frame_seconds))
            start = None
    if start is not None:
        intervals.append((start * frame_seconds,
                          (last_speech + 1) * frame_seconds))
    return intervals


def label_segments_single_track(segments: list, samples: np.ndarray,
                                sample_rate: int, *,
                                silence_rms: float = DEFAULT_SILENCE_RMS,
                                frame_seconds: float = DEFAULT_FRAME_SECONDS,
                                gap_seconds: float = DEFAULT_GAP_SECONDS
                                ) -> list[DiarizedSegment]:
    """Approximate fallback: split one track at silence gaps, alternate labels.

    Labels are generic ``SPEAKER_00`` / ``SPEAKER_01`` — this mode
    cannot know who is speaking, only that the speaker changed after
    a pause. A segment is assigned the label of the speech interval
    it overlaps most; a segment with no speech overlap is ``Silence``.
    """
    rms = frame_rms(samples, sample_rate, frame_seconds)
    intervals = _speech_intervals(rms, frame_seconds, silence_rms,
                                  gap_seconds)
    labelled: list[DiarizedSegment] = []
    for seg in segments:
        best_label = SILENCE
        best_overlap = 0.0
        for idx, (istart, iend) in enumerate(intervals):
            overlap = max(0.0, min(seg.end, iend) - max(seg.start, istart))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = f"SPEAKER_{idx % 2:02d}"
        labelled.append(DiarizedSegment(
            start=float(seg.start), end=float(seg.end), speaker=best_label,
            text=getattr(seg, "text", "")))
    return labelled
