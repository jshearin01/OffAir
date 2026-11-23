"""Unit tests — Phase 4 near-live chunked transcription (fake engine)."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.nearlive.chunker import (
    ChunkedTranscriber,
    split_wav_into_chunks,
)
from app.transcription.engine import Segment, TranscriptionResult


def _write_tone_wav(path, duration=61.0, rate=16000, freq=440.0):
    t = np.arange(int(duration * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), samples, rate, subtype="PCM_16")
    return path


class FakeEngine:
    """Fake Whisper: two 2 s segments per chunk, text names the chunk file."""

    def __init__(self):
        self.calls = []

    def transcribe_file(self, audio_path, **kwargs):
        self.calls.append(Path(audio_path).name)
        return TranscriptionResult(
            segments=[Segment(0.0, 2.0, f"hello from {Path(audio_path).name}"),
                      Segment(2.0, 4.0, "world")],
            model="tiny", audio_duration_seconds=4.0, wall_seconds=0.1)

    def unload(self):
        pass


# ------------------------------------------------------------ window split ---
def test_split_wav_into_chunks_covers_audio(tmp_path):
    wav = _write_tone_wav(tmp_path / "a.wav", duration=61.0)
    windows = split_wav_into_chunks(wav, chunk_seconds=25.0,
                                    overlap_seconds=0.5)
    assert windows[0] == (0.0, 25.0)
    assert windows[1] == (24.5, 49.5)
    assert windows[2][0] == 49.0 and windows[2][1] == pytest.approx(61.0)
    # Coverage: every second of audio is in at least one window.
    covered = np.zeros(61, dtype=bool)
    for start, end in windows:
        covered[int(start):int(end) + 1] = True
    assert covered.all()


def test_split_short_audio_is_one_window(tmp_path):
    wav = _write_tone_wav(tmp_path / "a.wav", duration=10.0)
    assert split_wav_into_chunks(wav) == [(0.0, 10.0)]


def test_split_validation(tmp_path):
    wav = _write_tone_wav(tmp_path / "a.wav", duration=10.0)
    with pytest.raises(ValueError, match="chunk_seconds"):
        split_wav_into_chunks(wav, chunk_seconds=2.0)
    with pytest.raises(ValueError, match="chunk_seconds"):
        split_wav_into_chunks(wav, chunk_seconds=120.0)
    with pytest.raises(ValueError, match="overlap_seconds"):
        split_wav_into_chunks(wav, overlap_seconds=25.0)
    with pytest.raises(FileNotFoundError):
        split_wav_into_chunks(tmp_path / "missing.wav")


# ------------------------------------------------------- chunked transcriber ---
def test_process_file_offsets_segments_and_appends_in_order(tmp_path):
    wav = _write_tone_wav(tmp_path / "a.wav", duration=61.0)
    engine = FakeEngine()
    transcriber = ChunkedTranscriber(engine=engine)
    chunks = transcriber.process_file(wav, work_dir=tmp_path / "work")
    assert len(chunks) == 3
    assert transcriber.chunks_done == 3
    # Chunk 1's segments are offset by its 24.5 s start.
    assert chunks[1].segments[0].start == pytest.approx(24.5)
    assert chunks[1].segments[0].end == pytest.approx(26.5)
    # All segments are in time order and carry the provisional label.
    starts = [s.start for s in transcriber.segments]
    assert starts == sorted(starts)
    assert all(s.speaker == "Provisional" for s in transcriber.segments)
    # No chunk WAVs are left behind.
    assert list((tmp_path / "work").glob("chunk-*.wav")) == []
    assert len(engine.calls) == 3


def test_chunk_overlap_dedupes_second_copy(tmp_path):
    # Fake engine returns the same "world" segment in every chunk;
    # the overlap copy (same start to 0.1 s + same text) is dropped.
    wav = _write_tone_wav(tmp_path / "a.wav", duration=61.0)
    engine = FakeEngine()
    transcriber = ChunkedTranscriber(engine=engine)
    transcriber.process_file(wav, work_dir=tmp_path / "work")
    texts = [s.text for s in transcriber.segments]
    # "world" appears once per chunk (different starts), not twice
    # per overlap — and the first chunk's hello is kept once.
    assert texts.count("world") == 3
    assert len(texts) == len(set(
        (round(s.start, 1), s.text) for s in transcriber.segments))


def test_provisional_labels_from_dual_channel_energy(tmp_path):
    # Mic loud in the first half, system loud in the second half.
    rate = 16000
    t = np.arange(int(61 * rate)) / rate
    mic = np.where(t < 30, 0.3 * np.sin(2 * np.pi * 440 * t), 0.001)
    system = np.where(t < 30, 0.001, 0.3 * np.sin(2 * np.pi * 440 * t))
    mic_path = tmp_path / "mic.wav"
    sys_path = tmp_path / "system.wav"
    combined = tmp_path / "combined.wav"
    sf.write(str(mic_path), mic.astype(np.float32), rate, subtype="PCM_16")
    sf.write(str(sys_path), system.astype(np.float32), rate, subtype="PCM_16")
    sf.write(str(combined), (mic + system).astype(np.float32), rate,
             subtype="PCM_16")
    transcriber = ChunkedTranscriber(engine=FakeEngine())
    transcriber.process_file(combined, work_dir=tmp_path / "work",
                             mic_path=mic_path, system_path=sys_path)
    labels = {s.speaker for s in transcriber.segments}
    assert "You" in labels and "Remote" in labels
    early = [s for s in transcriber.segments if s.start < 25]
    assert all(s.speaker == "You" for s in early)


def test_process_new_audio_advances_watermark(tmp_path):
    wav = _write_tone_wav(tmp_path / "a.wav", duration=30.0)
    engine = FakeEngine()
    transcriber = ChunkedTranscriber(engine=engine)
    first = transcriber.process_new_audio(wav, work_dir=tmp_path / "work")
    # Only the one full 25 s window is due; the 5 s tail waits.
    assert len(first) == 1 and first[0].start == 0.0
    # Nothing new: the same file again transcribes nothing.
    assert transcriber.process_new_audio(wav, work_dir=tmp_path / "work") == []
    # Grow the file to 61 s: the next call picks up the window
    # anchored at the watermark (25 s) minus the overlap (24.5 s).
    _write_tone_wav(tmp_path / "a.wav", duration=61.0)
    second = transcriber.process_new_audio(wav, work_dir=tmp_path / "work")
    assert len(second) == 1
    assert second[0].start == pytest.approx(24.5)
    assert second[0].end == pytest.approx(49.5)
    starts = [s.start for s in transcriber.segments]
    assert starts == sorted(starts)
    # The 49.5–61 s tail is partial live; process_file finishes it.
    rest = transcriber.process_file(wav, work_dir=tmp_path / "work")
    assert rest[-1].end == pytest.approx(61.0)


def test_chunked_transcriber_validation():
    with pytest.raises(ValueError, match="chunk_seconds"):
        ChunkedTranscriber(chunk_seconds=2.0)
    with pytest.raises(ValueError, match="overlap_seconds"):
        ChunkedTranscriber(overlap_seconds=25.0)


def test_process_file_missing_audio_raises(tmp_path):
    transcriber = ChunkedTranscriber(engine=FakeEngine())
    with pytest.raises(FileNotFoundError):
        transcriber.process_file(tmp_path / "missing.wav")
