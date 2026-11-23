"""Speaker diarization — PRD Phase 3.

Two modes:

* ``simple`` ("You vs Remote", :mod:`app.diarization.simple`): no token,
  no model download, fast. Labels each transcript segment by comparing
  energy in the Phase 1 ``mic.wav`` (you) against ``system.wav``
  (remote participants). A single-track fallback exists for sessions
  with only one track, and is clearly labelled approximate.
* ``full`` (pyannote.audio, :mod:`app.diarization.pyannote_diarize`):
  multi-speaker diarization on CPU with generic ``SPEAKER_00`` labels.
  Needs a Hugging Face token and accepted model licences (see
  :mod:`app.diarization.pyannote_diarize` and README § Phase 3).
* ``sherpa`` (sherpa-onnx, :mod:`app.diarization.sherpa`):
  multi-speaker diarization on CPU with generic ``SPEAKER_00``
  labels — pyannote segmentation exported to ONNX plus 3D-Speaker
  ERes2Net embeddings, threshold-based clustering (no fixed speaker
  count). Needs no Hugging Face token: the models download once
  from public GitHub releases. This is the mode the UI offers as
  the token-free full diarization option.

Speaker names (renames) live in :mod:`app.diarization.names` and are
persisted per recording in ``speakers.json``. The session entry point
is :func:`app.diarization.session.diarize_session`.
"""
from __future__ import annotations

from app.diarization.names import (
    apply_names,
    load_speaker_names,
    rename_speaker,
    save_speaker_names,
)
from app.diarization.session import diarize_session
from app.diarization.sherpa import MODE_LABEL as SHERPA_MODE_LABEL

# Diarization modes the UI mode picker can offer. Each entry has a
# human label and a short description; "sherpa" additionally has an
# availability check (see :func:`app.diarization.sherpa.check_availability`)
# because it needs the sherpa-onnx package and its models. The UI
# track reads this table — it must not hard-code the mode list.
DIARIZATION_MODES: dict[str, dict] = {
    "simple": {
        "label": "Simple — You vs Remote (no token, fast)",
        "description": "Labels segments by mic vs system track energy. "
                       "No model, no download, instant.",
    },
    "full": {
        "label": "Full — pyannote (needs HF token)",
        "description": "Multi-speaker diarization via pyannote.audio. "
                       "Needs a Hugging Face token and accepted licences.",
    },
    "sherpa": {
        "label": SHERPA_MODE_LABEL,
        "description": "Multi-speaker diarization via sherpa-onnx "
                       "(pyannote segmentation ONNX + ERes2Net). "
                       "No token; models (~33 MB) download once on "
                       "first use from GitHub releases.",
    },
}

__all__ = [
    "DIARIZATION_MODES",
    "apply_names",
    "diarize_session",
    "load_speaker_names",
    "rename_speaker",
    "save_speaker_names",
]
