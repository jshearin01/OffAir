"""Transcript exports — TXT and SRT. Pure functions, no Whisper needed.

TXT format (one line per segment)::

    [00:03] Good morning everyone, thanks for joining.

SRT format is the standard SubRip layout with ``HH:MM:SS,mmm`` timestamps.
Both writers are also used by :meth:`TranscriptionEngine.transcribe_session`.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.transcription.engine import Segment


def format_srt_timestamp(seconds: float) -> str:
    """Seconds -> ``HH:MM:SS,mmm`` (SRT)."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_txt_timestamp(seconds: float) -> str:
    """Seconds -> ``[MM:SS]`` (``[HH:MM:SS]`` once past an hour)."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def segments_to_dicts(segments: list[Segment]) -> list[dict]:
    return [s.to_dict() for s in segments]


def segments_from_dicts(data: list[dict]) -> list[Segment]:
    return [Segment.from_dict(d) for d in data]


def load_transcript_json(path: str | Path) -> list[Segment]:
    """Load segments from a transcript.json written by transcribe_session."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return segments_from_dicts(payload["segments"])


def _display_speaker(segment: Segment,
                     speaker_names: dict[str, str] | None) -> str | None:
    """The speaker label to print for a segment, or None if unlabelled."""
    speaker = getattr(segment, "speaker", None)
    if not speaker:
        return None
    if speaker_names:
        speaker = speaker_names.get(speaker, speaker)
    return speaker


def export_txt(segments: list[Segment], path: str | Path, *,
               speaker_names: dict[str, str] | None = None) -> Path:
    """Write ``[MM:SS] text`` lines, one per segment.

    Phase 3: a diarized segment is written as
    ``[MM:SS] Speaker: text`` (renames from ``speaker_names`` applied).
    Segments without a speaker keep the Phase 2 format, so existing
    callers and tests are unaffected.
    """
    path = Path(path)
    lines = []
    for s in segments:
        speaker = _display_speaker(s, speaker_names)
        prefix = f"{speaker}: " if speaker else ""
        lines.append(f"{format_txt_timestamp(s.start)} {prefix}{s.text}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8")
    return path


def export_srt(segments: list[Segment], path: str | Path, *,
               speaker_names: dict[str, str] | None = None) -> Path:
    """Write standard SRT blocks, numbered from 1.

    Phase 3: a diarized segment's text is prefixed ``Speaker: ``
    (renames applied), matching export_txt. Unlabelled segments are
    written exactly as in Phase 2.
    """
    path = Path(path)
    blocks = []
    for i, s in enumerate(segments, start=1):
        speaker = _display_speaker(s, speaker_names)
        prefix = f"{speaker}: " if speaker else ""
        blocks.append(
            f"{i}\n"
            f"{format_srt_timestamp(s.start)} --> "
            f"{format_srt_timestamp(s.end)}\n"
            f"{prefix}{s.text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path
