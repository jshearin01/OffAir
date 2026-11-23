"""Near-live / chunked processing — PRD Phase 4.

Not true real-time streaming: audio is processed in 20–30 second
windows (the PRD's near-live shape), each chunk transcribed on CPU
with the Phase 2 engine, and the transcript grows chunk by chunk
with *provisional* speaker labels. The primary workflow stays
"process after stop"; near-live is a progressive enhancement for
watching a transcript build up, and a chunked re-run is also useful
for long recordings on low-RAM machines (one chunk's model pass at
a time, then unload).

CPU-stability guidance (see :mod:`app.nearlive.chunker` and
PHASE4.md): 25 s chunks, 0.5 s overlap so words are not lost at a
boundary, the configured Whisper model for near-live — ``base``
on a fresh install, which keeps up with real time (RTF ~0.135)
— and the engine is unloaded between chunks when memory is tight.
"""
from __future__ import annotations

from app.nearlive.chunker import ChunkedTranscriber, split_wav_into_chunks

__all__ = ["ChunkedTranscriber", "split_wav_into_chunks"]
