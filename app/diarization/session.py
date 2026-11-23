"""Wire diarization into a transcribed session — PRD Phase 3.

:func:`diarize_session` is the entry point the UI and the manual test
call. It:

1. Loads ``transcript.json`` (Phase 2 output) from a session folder.
2. Runs the requested mode — ``"simple"`` (dual-channel energy, no
   token), ``"full"`` (pyannote, needs a token; see
   :mod:`app.diarization.pyannote_diarize`) or ``"sherpa"``
   (sherpa-onnx, no token; see :mod:`app.diarization.sherpa`).
3. Writes ``diarization.json`` (mode, method, per-segment labels).
4. Updates ``transcript.json`` in place, adding a ``"speaker"`` key
   to each segment (other transcript fields are preserved).
5. Re-exports ``transcript.txt`` / ``transcript.srt`` with the
   speaker labels — and any renames from ``speakers.json`` — applied.

Renames are applied at export time, never baked into the stored
labels: re-running :func:`rename_speaker` and re-exporting (or
re-running :func:`diarize_session`) picks the new names up. Use
:func:`reapply_names` to refresh the exports after a rename without
re-running diarization.

Single source of truth: the segment list :func:`diarize_session`
returns. The write path serializes exactly that list — the same
``DiarizedSegment`` objects, no re-derivation — and labels are
computed from the *rounded* turns that are persisted in
``diarization.json`` (3 decimals, the same rounding the result
``to_dict`` methods apply). Labelling from the engines' raw
full-precision turns once let a <1 ms boundary segment come back
labelled one speaker while every later reader re-deriving from
the persisted file got the other speaker; learning reads the
file, so that disagreement could poison a profile. After the
write, :func:`verify_persisted_labels` re-reads both files and
compares the persisted speaker labels to the returned list; a
mismatch raises :class:`PersistedLabelMismatchError` loudly
instead of returning success with wrong labels on disk.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from app.config import DIARIZATION_DEFAULT_MODE
from app.diarization.names import load_speaker_names
from app.diarization.simple import (
    DiarizedSegment,
    SimpleDiarizationResult,
    label_segments_dual,
    label_segments_single_track,
    load_wav_mono,
)
from app.storage.sessions import COMBINED_FILE, MIC_FILE, SYSTEM_FILE
from app.transcription.engine import TRANSCRIPT_JSON, Segment
from app.transcription.exports import (
    export_srt,
    export_txt,
    load_transcript_json,
)

logger = logging.getLogger("offair.diarization.session")

DIARIZATION_JSON = "diarization.json"
TRANSCRIPT_TXT = "transcript.txt"
TRANSCRIPT_SRT = "transcript.srt"


class PersistedLabelMismatchError(RuntimeError):
    """Persisted speaker labels disagree with the returned segment list.

    Raised by the write path (and by :func:`verify_persisted_labels`)
    when ``diarization.json`` / ``transcript.json`` on disk do not
    carry exactly the speaker labels :func:`diarize_session`
    returned. Never silently persist a disagreement: callers must
    treat the run as failed and re-run diarization rather than
    letting learning read the wrong labels.
    """


def _rounded_turns(turns: list[tuple[float, float, str]]
                   ) -> list[tuple[float, float, str]]:
    """Turns rounded to the precision persisted in diarization.json.

    The result ``to_dict`` methods round turn boundaries to 3
    decimals; labelling must use those exact values, so the labels
    in the returned payload, in ``diarization.json`` and in
    ``transcript.json`` are all computed from one turn list, and a
    later reader re-deriving labels from the persisted file gets
    the same answer. Turn order is preserved (ties in overlap are
    broken by order, so re-derivation must see the same order).
    """
    return [(round(float(start), 3), round(float(end), 3), label)
            for start, end, label in turns]


def _label_segments_from_turns(
        segments: list[Segment],
        turns: list[tuple[float, float, str]]) -> list[DiarizedSegment]:
    """Assign each segment the turn label it overlaps most (full mode).

    A segment with no overlapping turn gets ``"Unknown"`` — it is
    never silently given a speaker it did not overlap.
    """
    labelled: list[DiarizedSegment] = []
    for seg in segments:
        best_label = "Unknown"
        best_overlap = 0.0
        for start, end, label in turns:
            overlap = max(0.0, min(seg.end, end) - max(seg.start, start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        labelled.append(DiarizedSegment(
            start=float(seg.start), end=float(seg.end), speaker=best_label,
            text=seg.text))
    return labelled


def verify_persisted_labels(session_dir: str | Path,
                            labelled: list[DiarizedSegment]) -> None:
    """Re-read the written files and compare labels to ``labelled``.

    ``diarization.json`` (when it carries a ``"segments"`` list)
    and ``transcript.json`` must both carry exactly the speaker
    labels of the returned segment list, in order. Raises
    :class:`PersistedLabelMismatchError` on any count or label
    disagreement, and FileNotFoundError when a file is missing.
    """
    session_dir = Path(session_dir)
    expected = [lab.speaker for lab in labelled]
    diarization_path = session_dir / DIARIZATION_JSON
    if not diarization_path.exists():
        raise FileNotFoundError(f"No {DIARIZATION_JSON} in {session_dir}")
    diarization = json.loads(diarization_path.read_text(encoding="utf-8"))
    persisted_segments = diarization.get("segments")
    if isinstance(persisted_segments, list):
        persisted = [str(seg.get("speaker")) for seg in persisted_segments]
        if persisted != expected:
            raise PersistedLabelMismatchError(
                f"Persisted {DIARIZATION_JSON} labels {persisted} "
                f"disagree with the returned labels {expected}")
    transcript_path = session_dir / TRANSCRIPT_JSON
    if not transcript_path.exists():
        raise FileNotFoundError(f"No {TRANSCRIPT_JSON} in {session_dir}")
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        raise PersistedLabelMismatchError(
            f"{TRANSCRIPT_JSON} in {session_dir} has no segments list")
    persisted = [str(seg.get("speaker")) for seg in segments]
    if persisted != expected:
        raise PersistedLabelMismatchError(
            f"Persisted {TRANSCRIPT_JSON} labels {persisted} disagree "
            f"with the returned labels {expected}")


def _write_outputs(session_dir: Path, payload: dict,
                   labelled: list[DiarizedSegment]) -> None:
    """Write diarization.json, update transcript.json, re-export txt/srt.

    Serializes exactly ``labelled``: the payload's ``"segments"``
    are the returned segment list's dicts, and ``transcript.json``
    gets those speakers, in order, with no re-derivation. The
    transcript is read and its segment count validated *before*
    anything is written, so a count disagreement raises
    :class:`PersistedLabelMismatchError` instead of silently
    truncating via ``zip``. After both files are written, the
    persisted labels are re-read and reconciled against
    ``labelled`` (see :func:`verify_persisted_labels`).
    """
    transcript_path = session_dir / TRANSCRIPT_JSON
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript_segments = transcript.get("segments")
    if not isinstance(transcript_segments, list):
        raise PersistedLabelMismatchError(
            f"{TRANSCRIPT_JSON} in {session_dir} has no segments list")
    if len(transcript_segments) != len(labelled):
        raise PersistedLabelMismatchError(
            f"{TRANSCRIPT_JSON} has {len(transcript_segments)} segments "
            f"but diarization returned {len(labelled)} — refusing to "
            "persist a partial label assignment")
    payload["segments"] = [s.to_dict() for s in labelled]
    (session_dir / DIARIZATION_JSON).write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    for seg_dict, lab in zip(transcript_segments, labelled):
        seg_dict["speaker"] = lab.speaker
    transcript["diarization"] = {
        "mode": payload["mode"], "method": payload["method"]}
    transcript_path.write_text(json.dumps(transcript, indent=2),
                               encoding="utf-8")
    verify_persisted_labels(session_dir, labelled)
    names = load_speaker_names(session_dir)
    segments = [Segment(s.start, s.end, s.text, speaker=s.speaker)
                for s in labelled]
    export_txt(segments, session_dir / TRANSCRIPT_TXT,
               speaker_names=names)
    export_srt(segments, session_dir / TRANSCRIPT_SRT,
               speaker_names=names)


def diarize_session(session_dir: str | Path, *,
                    mode: str = DIARIZATION_DEFAULT_MODE,
                    min_speakers: int | None = None,
                    max_speakers: int | None = None,
                    token: str | None = None,
                    progress_callback=None,
                    diarizer=None,
                    sherpa_model_dir: str | Path | None = None,
                    sherpa_expected_speakers: int | None = None,
                    sherpa_clustering_threshold: float | None = None,
                    ) -> dict:
    """Diarize a transcribed session folder. Returns the diarization payload.

    ``mode`` is ``"sherpa"`` (default — the configured default,
    :data:`app.config.DIARIZATION_DEFAULT_MODE`), ``"simple"`` or
    ``"full"``. ``diarizer`` is an injectable diarizer (tests pass a
    fake); with ``None`` the real diarizer for the mode is used
    (``PyannoteDiarizer`` for "full", ``SherpaDiarizer`` for
    "sherpa"). The sherpa settings (``sherpa_model_dir`` /
    ``sherpa_expected_speakers`` / ``sherpa_clustering_threshold``)
    come from the app config — the UI passes them; ``None`` means
    the diarizer's own defaults. Raises FileNotFoundError if there
    is no transcript or no usable audio, and
    :class:`HuggingFaceTokenError` from full mode when no token is set.
    """
    session_dir = Path(session_dir)
    transcript_path = session_dir / TRANSCRIPT_JSON
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"No {TRANSCRIPT_JSON} in {session_dir} — transcribe the "
            "session first (Phase 2).")
    segments = load_transcript_json(transcript_path)
    started = time.monotonic()

    if mode == "simple":
        mic_path = session_dir / MIC_FILE
        system_path = session_dir / SYSTEM_FILE
        if mic_path.exists() and system_path.exists():
            mic_samples, mic_rate = load_wav_mono(mic_path)
            sys_samples, sys_rate = load_wav_mono(system_path)
            labelled = label_segments_dual(
                segments, mic_samples, mic_rate, sys_samples, sys_rate)
            method = "dual-channel-energy"
            duration = max(len(mic_samples) / mic_rate if mic_rate else 0,
                           len(sys_samples) / sys_rate if sys_rate else 0)
        else:
            # Single-track fallback — clearly labelled approximate.
            audio_path = session_dir / COMBINED_FILE
            if not audio_path.exists():
                audio_path = mic_path
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"No {MIC_FILE}, {SYSTEM_FILE} or {COMBINED_FILE} in "
                    f"{session_dir} to diarize.")
            samples, rate = load_wav_mono(audio_path)
            labelled = label_segments_single_track(segments, samples, rate)
            method = "single-track-energy (approximate)"
            duration = len(samples) / rate if rate else 0
        result = SimpleDiarizationResult(
            segments=labelled, mode="simple", method=method,
            audio_duration_seconds=round(duration, 2) if duration else None)
        payload = result.to_dict()
        payload["wall_seconds"] = round(time.monotonic() - started, 2)
        if progress_callback is not None:
            progress_callback(1.0, "Simple diarization complete.")
    elif mode == "full":
        from app.diarization.pyannote_diarize import PyannoteDiarizer
        audio_path = session_dir / COMBINED_FILE
        if not audio_path.exists():
            audio_path = session_dir / MIC_FILE
        if not audio_path.exists():
            raise FileNotFoundError(
                f"No {COMBINED_FILE} or {MIC_FILE} in {session_dir} "
                "to diarize.")
        engine = diarizer if diarizer is not None else PyannoteDiarizer(token=token)
        full = engine.diarize(
            audio_path, min_speakers=min_speakers,
            max_speakers=max_speakers,
            progress_callback=progress_callback)
        # Label from the rounded turns that are persisted, so the
        # returned labels and any later re-derivation agree exactly.
        turns = _rounded_turns(full.turns)
        labelled = _label_segments_from_turns(segments, turns)
        payload = full.to_dict()
        payload["turns"] = [{"start": s, "end": e, "speaker": sp}
                            for s, e, sp in turns]
        payload["segments"] = [s.to_dict() for s in labelled]
        payload["wall_seconds"] = round(time.monotonic() - started, 2)
    elif mode == "sherpa":
        # Mirror of the "full" branch: SherpaDiarizer returns the same
        # to_dict() shape (mode / method / turns / speakers /
        # wall_seconds), so the same _label_segments_from_turns +
        # _write_outputs path applies. The sherpa settings come from
        # the app config (UI passes them); None = diarizer defaults.
        from app.diarization.sherpa import SherpaDiarizer
        audio_path = session_dir / COMBINED_FILE
        if not audio_path.exists():
            audio_path = session_dir / MIC_FILE
        if not audio_path.exists():
            raise FileNotFoundError(
                f"No {COMBINED_FILE} or {MIC_FILE} in {session_dir} "
                "to diarize.")
        if diarizer is not None:
            engine = diarizer
        else:
            kwargs: dict = {}
            if sherpa_model_dir is not None:
                kwargs["model_dir"] = sherpa_model_dir
            if sherpa_expected_speakers is not None:
                kwargs["expected_speakers"] = sherpa_expected_speakers
            if sherpa_clustering_threshold is not None:
                kwargs["clustering_threshold"] = sherpa_clustering_threshold
            engine = SherpaDiarizer(**kwargs)
        sherpa_result = engine.diarize(
            audio_path, progress_callback=progress_callback)
        # Label from the rounded turns that are persisted, so the
        # returned labels and any later re-derivation agree exactly
        # (same rule as the full branch above).
        turns = _rounded_turns(sherpa_result.turns)
        labelled = _label_segments_from_turns(segments, turns)
        payload = sherpa_result.to_dict()
        payload["turns"] = [{"start": s, "end": e, "speaker": sp}
                            for s, e, sp in turns]
        payload["segments"] = [s.to_dict() for s in labelled]
        payload["wall_seconds"] = round(time.monotonic() - started, 2)
    else:
        raise ValueError(f"Unknown diarization mode {mode!r}; "
                         "expected 'simple', 'full' or 'sherpa'")

    _write_outputs(session_dir, payload, labelled)
    logger.info("Diarized %s (%s/%s): %d segments in %.2fs",
                session_dir.name, payload["mode"], payload["method"],
                len(labelled), payload["wall_seconds"])
    return payload


def reapply_names(session_dir: str | Path) -> Path:
    """Re-export transcript.txt/.srt after a rename, without re-diarizing.

    Reads the speaker labels from transcript.json and the current
    renames from speakers.json. Raises FileNotFoundError if the
    session has not been diarized yet.
    """
    session_dir = Path(session_dir)
    transcript_path = session_dir / TRANSCRIPT_JSON
    if not transcript_path.exists():
        raise FileNotFoundError(f"No {TRANSCRIPT_JSON} in {session_dir}")
    segments = load_transcript_json(transcript_path)
    if not any(getattr(s, "speaker", None) for s in segments):
        raise ValueError(
            "Transcript has no speaker labels yet — run diarization first.")
    names = load_speaker_names(session_dir)
    export_txt(segments, session_dir / TRANSCRIPT_TXT,
               speaker_names=names)
    export_srt(segments, session_dir / TRANSCRIPT_SRT,
               speaker_names=names)
    return session_dir / TRANSCRIPT_TXT
