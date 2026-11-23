"""Dictation pipeline shape.

Dictation reuses the existing session / transcribe / cleanup path:
a session with source "solo" and metadata dictation=true, whose
primary readable artifact is the deterministic cleaned note
(transcript-clean.txt). No Whisper run happens here — the tests
write a verbatim transcript and run the deterministic cleanup,
which is the layer that produces the note.
"""
from __future__ import annotations

import json

from app.storage.sessions import (
    create_dictation_session,
    create_session,
    is_dictation_session,
    load_metadata,
    update_metadata,
)
from app.transcription.cleanup import CLEAN_TXT, clean_session

SEGMENTS = [
    {"start": 0.0, "end": 3.0,
     "text": "Um, remember to call Priya about the budget."},
    {"start": 4.0, "end": 7.0,
     "text": "The the deadline is Friday."},
]


def _write_verbatim(session):
    (session / "transcript.json").write_text(json.dumps(
        {"model": "base", "segments": SEGMENTS}, indent=2),
        encoding="utf-8")


def test_dictation_session_shape(tmp_path):
    session = create_dictation_session(tmp_path)
    metadata = load_metadata(session)
    assert metadata["source"] == "solo"
    assert metadata["dictation"] is True
    assert metadata["app_name"] == "Dictation"
    assert metadata["status"] == "recording"
    assert metadata["session_id"] == session.name
    # Standard session keys are all present, unchanged.
    for key in ("start_time", "sample_rate", "channels", "files"):
        assert key in metadata
    assert is_dictation_session(session) is True


def test_normal_session_is_not_dictation(tmp_path):
    session = create_session(tmp_path, app_name="Teams")
    assert is_dictation_session(session) is False
    update_metadata(session, source="solo")
    # Solo source alone is not dictation: the flag is what marks it.
    assert is_dictation_session(session) is False


def test_dictation_cleaned_note_is_the_primary_artifact(tmp_path):
    session = create_dictation_session(tmp_path)
    _write_verbatim(session)
    stats = clean_session(session, tinyfix_enabled=False)
    assert stats["cleaned_segments"] == 2
    note = (session / CLEAN_TXT).read_text(encoding="utf-8")
    # Only the spoken words, cleaned: filler gone, the repeated
    # "the" kept verbatim (repeat dedup was removed Nov 22 2026
    # — see docs/notes/cleanup-repeat-rule-2026-11-22.md),
    # nothing added.
    assert note == ("[00:00] Remember to call Priya about the budget.\n"
                    "[00:04] The the deadline is Friday.\n")
    # The verbatim transcript is untouched by the clean pass.
    verbatim = json.loads((session / "transcript.json").read_text(
        encoding="utf-8"))
    assert verbatim["segments"][0]["text"].startswith("Um, remember")
    # No LLM artifact is produced by the deterministic layer.
    assert not (session / "transcript-polished.md").exists()
    assert not (session / "transcript-enterprise-clean.txt").exists()
