"""Phase 3 diarization tests — fake pipelines + synthetic audio only.

No pyannote, no torch, no Hugging Face token, no network, no hardware.
Synthetic WAVs are written with the stdlib ``wave`` module; the
simple-mode tests use tone bursts at known times so the expected
"You"/"Remote" labels are exact by construction.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from app.diarization.names import (
    apply_names,
    display_name,
    load_speaker_names,
    rename_speaker,
    save_speaker_names,
)
from app.diarization.pyannote_diarize import (
    HuggingFaceTokenError,
    PyannoteDiarizer,
    resolve_hf_token,
    token_setup_guide,
    validate_speaker_hints,
)
from app.diarization.session import (
    DIARIZATION_JSON,
    PersistedLabelMismatchError,
    _label_segments_from_turns,
    _write_outputs,
    diarize_session,
    reapply_names,
    verify_persisted_labels,
)
from app.diarization.simple import (
    DiarizedSegment,
    frame_rms,
    label_segments_dual,
    label_segments_single_track,
    label_window,
    load_wav_mono,
    mean_rms_in_window,
)
from app.transcription.engine import Segment, TRANSCRIPT_JSON
from app.transcription.exports import export_srt, export_txt

SR = 16000


def _tone(seconds: float, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray) -> Path:
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def _burst_track(speech_windows: list[tuple[float, float]]) -> np.ndarray:
    """Silence with a tone burst in each (start, end) window."""
    total = max((e for _, e in speech_windows), default=0.0)
    samples = np.zeros(int(total * SR) + SR, dtype=np.float32)
    for start, end in speech_windows:
        i0, i1 = int(start * SR), int(end * SR)
        samples[i0:i1] = _tone(end - start)[: i1 - i0]
    return samples


def _session_with_transcript(tmp_path: Path, segments: list[Segment],
                             mic_windows=None, system_windows=None,
                             combined_windows=None) -> Path:
    session = tmp_path / "session"
    session.mkdir()
    if mic_windows is not None:
        _write_wav(session / "mic.wav", _burst_track(mic_windows))
    if system_windows is not None:
        _write_wav(session / "system.wav", _burst_track(system_windows))
    if combined_windows is not None:
        _write_wav(session / "combined.wav", _burst_track(combined_windows))
    payload = {"model": "tiny", "segments": [s.to_dict() for s in segments]}
    (session / TRANSCRIPT_JSON).write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return session


# -------------------------------------------------------------- label_window --
def test_label_window_thresholds():
    assert label_window(0.10, 0.01) == "You"
    assert label_window(0.01, 0.10) == "Remote"
    assert label_window(0.001, 0.002) == "Silence"
    # Neither dominates by 1.5x -> Mixed.
    assert label_window(0.10, 0.09) == "Mixed"
    # Exactly at the dominance boundary counts as dominant.
    assert label_window(0.15, 0.10) == "You"


def test_frame_rms_and_window_rms():
    silence = np.zeros(SR, dtype=np.float32)
    assert frame_rms(silence, SR).max() == 0.0
    assert mean_rms_in_window(silence, SR, 0.0, 1.0) == 0.0
    tone = _tone(1.0, amp=0.5)
    # Sine RMS = amplitude / sqrt(2).
    assert mean_rms_in_window(tone, SR, 0.0, 1.0) == pytest.approx(
        0.5 / np.sqrt(2), rel=0.01)
    assert mean_rms_in_window(tone, SR, 2.0, 1.0) == 0.0  # empty window


def test_load_wav_mono_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_wav_mono(tmp_path / "nope.wav")


# ------------------------------------------------------------ dual-channel --
def test_dual_mode_labels_you_remote_silence_mixed():
    segments = [Segment(0.0, 2.0, "Hello, this is John."),
                Segment(2.0, 4.0, "Hi John, this is Priya."),
                Segment(4.0, 5.0, ""),
                Segment(5.0, 7.0, "Both talking at once.")]
    mic = _burst_track([(0.0, 2.0), (5.0, 7.0)])
    system = _burst_track([(2.0, 4.0), (5.0, 7.0)])
    labelled = label_segments_dual(segments, mic, SR, system, SR)
    assert [s.speaker for s in labelled] == ["You", "Remote", "Silence", "Mixed"]
    assert labelled[0].mic_rms > 0.1 and labelled[0].system_rms == 0.0
    assert labelled[1].system_rms > 0.1 and labelled[1].mic_rms == 0.0


def test_dual_mode_tolerates_unequal_track_lengths():
    segments = [Segment(0.0, 1.0, "Short.")]
    mic = _burst_track([(0.0, 1.0)])
    system = np.zeros(SR // 2, dtype=np.float32)  # only 0.5 s long
    labelled = label_segments_dual(segments, mic, SR, system, SR)
    assert labelled[0].speaker == "You"


# --------------------------------------------------------- single-track -----
def test_single_track_fallback_alternates_at_gaps():
    segments = [Segment(0.0, 1.0, "First turn."),
                Segment(1.0, 2.0, "Still first."),
                Segment(3.0, 4.0, "Second turn after a pause."),
                Segment(5.0, 6.0, "Silence here, no speech.")]
    samples = _burst_track([(0.0, 2.0), (3.0, 4.0)])
    labelled = label_segments_single_track(segments, samples, SR)
    assert [s.speaker for s in labelled] == [
        "SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "Silence"]


def test_single_track_all_silence_is_silence():
    segments = [Segment(0.0, 1.0, "Nothing said.")]
    samples = np.zeros(SR, dtype=np.float32)
    assert label_segments_single_track(
        segments, samples, SR)[0].speaker == "Silence"


# ------------------------------------------------------------------ names ---
def test_names_save_load_rename_and_clear(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    assert load_speaker_names(session) == {}
    rename_speaker(session, "SPEAKER_00", "Priya")
    rename_speaker(session, "You", "John")
    assert load_speaker_names(session) == {
        "SPEAKER_00": "Priya", "You": "John"}
    assert (session / "speakers.json").exists()
    # Blank name removes the rename.
    rename_speaker(session, "You", "   ")
    assert load_speaker_names(session) == {"SPEAKER_00": "Priya"}
    with pytest.raises(ValueError, match="must not be blank"):
        rename_speaker(session, "   ", "Nobody")


def test_names_apply_to_segments_and_dicts():
    segs = [Segment(0.0, 1.0, "Hi.", speaker="SPEAKER_00"),
            Segment(1.0, 2.0, "Hello.", speaker="You"),
            Segment(2.0, 3.0, "No label.")]
    out = apply_names(segs, {"SPEAKER_00": "Priya", "You": "John"})
    assert [s.speaker for s in out] == ["Priya", "John", None]
    # Inputs untouched, dicts supported.
    assert segs[0].speaker == "SPEAKER_00"
    dicts = apply_names([{"speaker": "SPEAKER_00", "text": "Hi."}],
                        {"SPEAKER_00": "Priya"})
    assert dicts[0]["speaker"] == "Priya"
    assert display_name("SPEAKER_01", {}) == "SPEAKER_01"


def test_save_speaker_names_drops_blanks(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    save_speaker_names(session, {"A": "  ", "B": "Bea"})
    assert load_speaker_names(session) == {"B": "Bea"}


# ------------------------------------------------------- pyannote wrapper ---
def test_validate_speaker_hints():
    assert validate_speaker_hints(None, None) == (None, None)
    assert validate_speaker_hints(1, 3) == (1, 3)
    assert validate_speaker_hints(10, 10) == (10, 10)
    for bad in (0, 11, -1, 2.5, "3", True):
        with pytest.raises(ValueError):
            validate_speaker_hints(bad, None)
        with pytest.raises(ValueError):
            validate_speaker_hints(None, bad)
    with pytest.raises(ValueError, match="must not exceed"):
        validate_speaker_hints(4, 2)


def test_missing_token_raises_guided_error(monkeypatch):
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(HuggingFaceTokenError) as excinfo:
        resolve_hf_token()
    message = str(excinfo.value)
    assert "huggingface.co/settings/tokens" in message
    assert "speaker-diarization-3.1" in message
    assert "segmentation-3.0" in message
    assert "HF_TOKEN" in message
    # The guide itself is the same text the UI shows.
    assert token_setup_guide() in message
    # No real token is ever invented or embedded (the guide only shows
    # the "hf_..." placeholder for the user to replace).
    assert "hf_test123" not in message


def test_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "  hf_test123  ")
    assert resolve_hf_token() == "hf_test123"
    assert resolve_hf_token("hf_arg") == "hf_arg"


class _FakeSegment:
    def __init__(self, start, end):
        self.start, self.end = start, end


class _FakeAnnotation:
    def __init__(self, turns):
        self._turns = turns

    def itertracks(self, yield_label=False):
        for start, end, label in self._turns:
            yield _FakeSegment(start, end), 0, label


class _FakePipeline:
    def __init__(self, turns):
        self.turns = turns
        self.calls = []

    def __call__(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return _FakeAnnotation(self.turns)

    def to(self, device):  # pipeline.to(torch.device("cpu")) in real code
        self.device = str(device)
        return self


def test_full_diarizer_with_fake_pipeline(tmp_path):
    turns = [(5.0, 7.0, "SPEAKER_01"), (0.0, 2.0, "SPEAKER_00"),
             (2.0, 4.0, "SPEAKER_00")]
    fake = _FakePipeline(turns)
    diarizer = PyannoteDiarizer(token="hf_x", pipeline=fake)
    audio = _write_wav(tmp_path / "a.wav", _tone(8.0))
    result = diarizer.diarize(audio, min_speakers=1, max_speakers=3)
    assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert result.turns == sorted(turns)
    assert fake.calls[0][1] == {"min_speakers": 1, "max_speakers": 3}
    # Note: an injected pipeline is assumed already configured — the
    # CPU-only `.to(torch.device("cpu"))` call lives in the real
    # _load_pipeline path, not in the injected-pipeline path.
    # Progress callback is called at start and end.
    marks = []
    diarizer.diarize(tmp_path / "a.wav",
                     progress_callback=lambda f, t: marks.append(f))
    assert marks[0] == 0.0 and marks[-1] == 1.0


def test_full_diarizer_missing_audio_raises(tmp_path):
    diarizer = PyannoteDiarizer(token="hf_x", pipeline=_FakePipeline([]))
    with pytest.raises(FileNotFoundError):
        diarizer.diarize(tmp_path / "nope.wav")


def test_full_diarizer_with_token_but_no_library_raises(tmp_path):
    # Only meaningful where pyannote is NOT installed (this VM): with
    # a token present, the failure must be the "not installed" error,
    # not a token error and not a crash. Skipped where pyannote exists.
    import importlib.util
    try:
        _installed = importlib.util.find_spec("pyannote.audio") is not None
    except ModuleNotFoundError:
        _installed = False  # no parent 'pyannote' package at all
    if _installed:
        pytest.skip("pyannote.audio is installed here")
    from app.diarization.pyannote_diarize import PyannoteNotInstalledError
    audio = _write_wav(tmp_path / "a.wav", _tone(1.0))
    diarizer = PyannoteDiarizer(token="hf_x")
    with pytest.raises(PyannoteNotInstalledError, match="not installed"):
        diarizer.diarize(audio)


def test_full_diarizer_without_token_or_library_raises(tmp_path):
    # No pipeline injected and no token -> guided token error (before any
    # import attempt). If pyannote were installed this would instead be
    # a token error from resolve_hf_token — either way it is guided.
    import os
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    diarizer = PyannoteDiarizer()
    audio = _write_wav(tmp_path / "a.wav", _tone(1.0))
    with pytest.raises(HuggingFaceTokenError, match="huggingface.co"):
        diarizer.diarize(audio)


# ----------------------------------------------------------------- session --
def test_diarize_session_simple_writes_files_and_updates_transcript(tmp_path):
    segments = [Segment(0.0, 2.0, "Hello, this is John."),
                Segment(2.0, 4.0, "Hi John, this is Priya.")]
    session = _session_with_transcript(
        tmp_path, segments, mic_windows=[(0.0, 2.0)],
        system_windows=[(2.0, 4.0)], combined_windows=[(0.0, 4.0)])
    payload = diarize_session(session, mode="simple")
    assert payload["mode"] == "simple"
    assert payload["method"] == "dual-channel-energy"
    assert [s["speaker"] for s in payload["segments"]] == ["You", "Remote"]
    assert (session / DIARIZATION_JSON).exists()
    transcript = json.loads((session / TRANSCRIPT_JSON).read_text())
    assert [s["speaker"] for s in transcript["segments"]] == ["You", "Remote"]
    assert transcript["model"] == "tiny"  # other transcript fields kept
    txt = (session / "transcript.txt").read_text()
    assert "[00:00] You: Hello, this is John." in txt
    assert "[00:02] Remote: Hi John, this is Priya." in txt
    srt = (session / "transcript.srt").read_text()
    assert "You: Hello, this is John." in srt


def test_diarize_session_simple_single_track_fallback(tmp_path):
    segments = [Segment(0.0, 1.0, "First."), Segment(2.0, 3.0, "Second.")]
    session = _session_with_transcript(
        tmp_path, segments, combined_windows=[(0.0, 1.0), (2.0, 3.0)])
    payload = diarize_session(session, mode="simple")
    assert payload["method"] == "single-track-energy (approximate)"
    assert [s["speaker"] for s in payload["segments"]] == [
        "SPEAKER_00", "SPEAKER_01"]


def test_diarize_session_full_with_fake_diarizer(tmp_path):
    segments = [Segment(0.0, 2.0, "Hello."), Segment(2.0, 4.0, "Hi.")]
    session = _session_with_transcript(
        tmp_path, segments, combined_windows=[(0.0, 4.0)])
    fake = _FakePipeline([(0.0, 2.0, "SPEAKER_00"),
                          (2.0, 4.0, "SPEAKER_01")])
    diarizer = PyannoteDiarizer(token="hf_x", pipeline=fake)
    payload = diarize_session(session, mode="full", diarizer=diarizer,
                              min_speakers=2, max_speakers=2)
    assert payload["mode"] == "full"
    assert payload["method"] == "pyannote"
    assert [s["speaker"] for s in payload["segments"]] == [
        "SPEAKER_00", "SPEAKER_01"]
    assert (session / "transcript.txt").read_text().startswith(
        "[00:00] SPEAKER_00: Hello.")


def test_diarize_session_full_no_overlap_is_unknown(tmp_path):
    segments = [Segment(10.0, 11.0, "After the turns.")]
    session = _session_with_transcript(
        tmp_path, segments, combined_windows=[(0.0, 12.0)])
    fake = _FakePipeline([(0.0, 2.0, "SPEAKER_00")])
    payload = diarize_session(session, mode="full",
                              diarizer=PyannoteDiarizer(token="hf_x",
                                                        pipeline=fake))
    assert payload["segments"][0]["speaker"] == "Unknown"


def test_rename_persists_and_appears_in_exports(tmp_path):
    segments = [Segment(0.0, 2.0, "Hello, this is John."),
                Segment(2.0, 4.0, "Hi John, this is Priya.")]
    session = _session_with_transcript(
        tmp_path, segments, mic_windows=[(0.0, 2.0)],
        system_windows=[(2.0, 4.0)], combined_windows=[(0.0, 4.0)])
    diarize_session(session, mode="simple")
    rename_speaker(session, "You", "John")
    rename_speaker(session, "Remote", "Priya")
    reapply_names(session)
    txt = (session / "transcript.txt").read_text()
    assert "[00:00] John: Hello, this is John." in txt
    assert "[00:02] Priya: Hi John, this is Priya." in txt
    assert "You:" not in txt and "Remote:" not in txt
    srt = (session / "transcript.srt").read_text()
    assert "John: Hello" in srt and "Priya: Hi" in srt
    # Renames are stored per recording, not baked into the labels.
    transcript = json.loads((session / TRANSCRIPT_JSON).read_text())
    assert transcript["segments"][0]["speaker"] == "You"


def test_reapply_names_without_labels_raises(tmp_path):
    segments = [Segment(0.0, 1.0, "Plain.")]
    session = _session_with_transcript(tmp_path, segments)
    with pytest.raises(ValueError, match="no speaker labels"):
        reapply_names(session)


def test_diarize_session_errors(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="transcript.json"):
        diarize_session(empty, mode="simple")
    segments = [Segment(0.0, 1.0, "Hi.")]
    session = _session_with_transcript(tmp_path, segments)
    with pytest.raises(FileNotFoundError, match="mic.wav"):
        diarize_session(session, mode="simple")
    with pytest.raises(ValueError, match="Unknown diarization mode"):
        diarize_session(session, mode="weird")


# ----------------------------------------------------------------- exports --
def test_exports_show_speakers_and_names(tmp_path):
    segs = [Segment(0.0, 2.5, "Hello.", speaker="SPEAKER_00"),
            Segment(2.5, 5.0, "Hi.", speaker=None)]
    path = export_txt(segs, tmp_path / "t.txt",
                      speaker_names={"SPEAKER_00": "Priya"})
    assert path.read_text() == (
        "[00:00] Priya: Hello.\n[00:02] Hi.\n")
    path = export_srt(segs, tmp_path / "t.srt",
                      speaker_names={"SPEAKER_00": "Priya"})
    assert "Priya: Hello." in path.read_text()
    assert "Hi." in path.read_text()


# ------------------------------------------- persisted-label agreement --
def test_returned_labels_agree_with_persisted_turns_at_a_boundary(tmp_path):
    """A segment straddling a turn boundary by <1 ms must label the
    same way from the persisted (rounded) turns as from the turns
    diarization used in memory.

    The boundary sits at 4.9996 s: on the raw turns SPEAKER_01
    overlaps the 10 s segment by 5.0004 s (more than SPEAKER_00's
    4.9996 s), but once the turns are rounded to 3 decimals for
    diarization.json the overlaps tie at 5.0 s and the first turn
    wins. Before the fix the returned payload said SPEAKER_01
    while every later reader re-deriving from the persisted file
    (including the learning guard) got SPEAKER_00 — the saved
    file disagreed with the returned payload, and learning read
    the file. Labels must be computed from the rounded turns, so
    the returned list, diarization.json and transcript.json are
    one source of truth.
    """
    segments = [Segment(0.0, 10.0, "Ten seconds of talk.")]
    session = _session_with_transcript(
        tmp_path, segments, combined_windows=[(0.0, 10.0)])
    fake = _FakePipeline([(0.0, 4.9996, "SPEAKER_00"),
                          (4.9996, 10.0, "SPEAKER_01")])
    payload = diarize_session(
        session, mode="full",
        diarizer=PyannoteDiarizer(token="hf_x", pipeline=fake))
    persisted = json.loads((session / DIARIZATION_JSON).read_text())
    persisted_turns = [(t["start"], t["end"], t["speaker"])
                       for t in persisted["turns"]]
    rederived = _label_segments_from_turns(segments, persisted_turns)
    assert [s.speaker for s in rederived] == [
        s["speaker"] for s in payload["segments"]]
    transcript = json.loads((session / TRANSCRIPT_JSON).read_text())
    assert [s["speaker"] for s in transcript["segments"]] == [
        s["speaker"] for s in payload["segments"]]
    assert [s["speaker"] for s in persisted["segments"]] == [
        s["speaker"] for s in payload["segments"]]


def test_write_outputs_refuses_segment_count_mismatch_without_writing(tmp_path):
    """A transcript with more segments than diarization returned must
    never be partially labelled: the old zip-based write silently
    left the extra segments unlabelled while the payload claimed
    the session was diarized."""
    session = _session_with_transcript(
        tmp_path, [Segment(0.0, 1.0, "One."), Segment(1.0, 2.0, "Two.")],
        combined_windows=[(0.0, 2.0)])
    payload = {"mode": "simple", "method": "dual-channel-energy"}
    labelled = [DiarizedSegment(0.0, 1.0, "You", "One.")]
    with pytest.raises(PersistedLabelMismatchError, match="2 segments"):
        _write_outputs(session, payload, labelled)
    assert not (session / DIARIZATION_JSON).exists()


def test_verify_persisted_labels_raises_on_tampered_transcript(tmp_path):
    """Reconciliation reads the files back: if transcript.json is
    edited after diarization so its labels no longer match the
    returned list, verification raises loudly."""
    segments = [Segment(0.0, 2.0, "Hello."), Segment(2.0, 4.0, "Hi.")]
    session = _session_with_transcript(
        tmp_path, segments, mic_windows=[(0.0, 2.0)],
        system_windows=[(2.0, 4.0)], combined_windows=[(0.0, 4.0)])
    payload = diarize_session(session, mode="simple")
    labelled = [DiarizedSegment(s["start"], s["end"], s["speaker"], s["text"])
                for s in payload["segments"]]
    verify_persisted_labels(session, labelled)  # agrees as written
    transcript = json.loads((session / TRANSCRIPT_JSON).read_text())
    transcript["segments"][0]["speaker"] = "SPEAKER_99"
    (session / TRANSCRIPT_JSON).write_text(
        json.dumps(transcript, indent=2), encoding="utf-8")
    with pytest.raises(PersistedLabelMismatchError, match="SPEAKER_99"):
        verify_persisted_labels(session, labelled)
