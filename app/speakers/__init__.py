"""Persistent speaker learning — PRD Phase 4.

Cross-session speaker memory: extract a voice embedding per
diarized speaker, store it in a local SQLite database
(:mod:`app.speakers.database`), and match new speakers against
known ones by cosine similarity (:mod:`app.speakers.matching`).

Four embedding backends exist (:mod:`app.speakers.embeddings`):

* ``speechbrain-ecapa`` — the production backend since the Nov 8
  2026 ECAPA swap: SpeechBrain ECAPA-TDNN
  (``speechbrain/spkrec-ecapa-voxceleb``, 192-dim, Apache 2.0,
  public weights, no token). Needs torch + speechbrain; not yet
  run on this VM (torch does not install here).
* ``onnx-ecapa`` — the other production backend: WeSpeaker
  ECAPA-TDNN-512 via ONNX Runtime (``speakeronnx``, 192-dim,
  CC-BY-4.0, no token, no torch). The only production backend
  verified working on this VM — it separates John from Marcus
  cleanly (cross-speaker 0.064 vs same-speaker 0.81–0.88).
* ``pyannote`` — opt-in only: the embedding model from the
  pyannote diarization pipeline. Needs torch + pyannote.audio
  and a Hugging Face token (same setup as Phase 3 full mode).
* ``fallback-spectral`` — a clearly-labelled test-only backend
  built from numpy/soundfile only (log-mel / MFCC statistics +
  pitch histogram). It is NOT a voice fingerprint: on the 3-voice
  TTS fixture it separates a female voice from male voices well,
  but it cannot reliably separate two similar male voices
  (John vs Marcus: cross-speaker cosine 0.95, above John's
  same-speaker half-vs-half score of 0.88 — see PHASE4.md).

``get_default_extractor()`` picks the best available production
backend (SpeechBrain ECAPA, else ONNX ECAPA, else the fallback);
``get_extractor(name)`` pins one backend by name. Matching
thresholds are per-backend (ECAPA backends score same-speaker
pairs around 0.8, so they use 0.55, not the fallback's 0.90) —
see ``app.config.SPEAKER_MATCH_THRESHOLDS``.
"""
from __future__ import annotations

from app.speakers.database import SpeakerDatabase
from app.speakers.embeddings import (
    FallbackEmbeddingExtractor,
    OnnxEcapaEmbeddingExtractor,
    SpeechBrainEcapaEmbeddingExtractor,
    get_default_extractor,
    get_extractor,
)
from app.speakers.matching import match_embedding

__all__ = [
    "FallbackEmbeddingExtractor",
    "OnnxEcapaEmbeddingExtractor",
    "SpeechBrainEcapaEmbeddingExtractor",
    "SpeakerDatabase",
    "get_default_extractor",
    "get_extractor",
    "match_embedding",
]
