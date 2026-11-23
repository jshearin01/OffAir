"""Clean Markdown export optimised for LLM consumption (PRD Phase 5).

The export is meant to be pasted straight into an LLM for a summary
or action items. It contains only things OffAir actually knows:

* a short "Context for summariser" block (date, duration, speakers,
  transcription model, diarization method),
* a participants list (display names from speakers.json),
* the timestamped, speaker-labelled transcript.

It never invents content. In particular, there is NO auto-generated
"Action items" section: that section appears only when the caller
passes an explicit ``action_items`` list (e.g. items the user typed,
or items a local LLM produced and the user approved). With no such
list the section is omitted entirely — an LLM reading the export is
free to derive action items from the transcript itself, which is
exactly what the PRD asks for ("exports clean transcripts suitable
for LLM summarization / todo generation").
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.diarization.names import load_speaker_names
from app.storage.sessions import METADATA_FILE, load_metadata
from app.transcription.engine import TRANSCRIPT_JSON, Segment
from app.transcription.exports import (
    format_txt_timestamp,
    load_transcript_json,
)

logger = logging.getLogger("offair.exports")

LLM_EXPORT_FILE = "transcript-llm.md"


def _display_speaker(segment: Segment,
                     speaker_names: dict[str, str] | None) -> str | None:
    speaker = getattr(segment, "speaker", None)
    if not speaker:
        return None
    if speaker_names:
        speaker = speaker_names.get(speaker, speaker)
    return speaker


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def export_llm_markdown(segments: list[Segment], path: str | Path, *,
                        speaker_names: dict[str, str] | None = None,
                        title: str | None = None,
                        date: str | None = None,
                        duration_seconds: float | None = None,
                        app_name: str | None = None,
                        model: str | None = None,
                        diarization: dict | None = None,
                        capture_mode: str | None = None,
                        action_items: list[str] | None = None) -> Path:
    """Write the LLM-ready Markdown export. Returns the path written.

    ``action_items`` is an explicit, caller-supplied list only. When
    it is None or empty, no "Action items" section is written at all
    — OffAir does not invent action items from the transcript.
    """
    path = Path(path)
    names = speaker_names or {}
    # Participants: every labelled speaker, display name first.
    participants: list[str] = []
    for seg in segments:
        speaker = _display_speaker(seg, names)
        if speaker and speaker not in participants:
            participants.append(speaker)

    lines: list[str] = []
    lines.append(f"# {title or 'Meeting transcript'}")
    lines.append("")
    lines.append("## Context for summariser")
    lines.append("")
    lines.append("This is a transcript of a recorded meeting, produced "
                 "entirely on-device by OffAir (local recording, "
                 "local transcription, local diarization). Summarise it "
                 "and extract action items from the transcript below. "
                 "Do not invent facts that are not in the transcript.")
    lines.append("")
    if date:
        lines.append(f"- Date: {date}")
    lines.append(f"- Duration: {_format_duration(duration_seconds)}")
    if app_name:
        lines.append(f"- Recorded from: {app_name}")
    if participants:
        lines.append(f"- Participants: {', '.join(participants)}")
    else:
        lines.append("- Participants: unknown (no speaker labels)")
    if model:
        lines.append(f"- Transcription model: {model} "
                     "(faster-whisper, CPU, int8)")
    if diarization:
        method = diarization.get("method") or diarization.get("mode") or "?"
        lines.append(f"- Speaker labelling: {method}")
        if "dual-channel" in str(method):
            lines.append("  - Note: simple mode labels speakers as "
                         "\"You\" (microphone) vs \"Remote\" (system "
                         "audio); it cannot tell remote speakers apart.")
    if capture_mode:
        lines.append(f"- Capture mode: {capture_mode}")
    lines.append("")
    lines.append("## Participants")
    lines.append("")
    if participants:
        for name in participants:
            lines.append(f"- {name}")
    else:
        lines.append("- Unknown")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")
    if not segments:
        lines.append("(No transcript segments.)")
    for seg in segments:
        speaker = _display_speaker(seg, names)
        prefix = f"**{speaker}:** " if speaker else ""
        lines.append(f"{format_txt_timestamp(seg.start)} {prefix}{seg.text}")
    lines.append("")
    # Action items: explicit caller-supplied list only. Never invented.
    if action_items:
        lines.append("## Action items")
        lines.append("")
        for item in action_items:
            item = str(item).strip()
            if item:
                lines.append(f"- [ ] {item}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote LLM Markdown export: %s (%d segments)",
                path, len(segments))
    return path


def export_llm_markdown_from_session(session_dir: str | Path, *,
                                    output_name: str = LLM_EXPORT_FILE,
                                    action_items: list[str] | None = None
                                    ) -> Path:
    """Build the LLM Markdown export for a transcribed session folder.

    Reads transcript.json, speakers.json renames, metadata.json and
    diarization.json from the session. Raises FileNotFoundError if
    there is no transcript yet.
    """
    session_dir = Path(session_dir)
    transcript_path = session_dir / TRANSCRIPT_JSON
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"No {TRANSCRIPT_JSON} in {session_dir} — transcribe the "
            "session first (Phase 2).")
    segments = load_transcript_json(transcript_path)
    names = load_speaker_names(session_dir)
    try:
        metadata = load_metadata(session_dir)
    except FileNotFoundError:
        metadata = {}
    diarization: dict | None = None
    diar_path = session_dir / "diarization.json"
    if diar_path.exists():
        try:
            diarization = json.loads(diar_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            diarization = None
    # Transcription model, if the transcript payload recorded one.
    model = None
    try:
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        model = payload.get("model")
    except (json.JSONDecodeError, OSError):
        model = None
    title = f"Meeting transcript — {metadata.get('session_id', session_dir.name)}"
    return export_llm_markdown(
        segments, session_dir / output_name,
        speaker_names=names, title=title,
        date=(metadata.get("start_time") or ""),
        duration_seconds=metadata.get("duration_seconds"),
        app_name=metadata.get("app_name"),
        model=model, diarization=diarization,
        capture_mode=metadata.get("mode"),
        action_items=action_items)
