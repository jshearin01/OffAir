# PRD: OffAir — Local Meeting Recorder with Transcription, Diarization, and Persistent Speaker Learning

## Executive Summary

**Project:** OffAir  
**Purpose:** Build a fully local Windows desktop application that records Microsoft Teams (and other) calls via system/per-app audio capture, performs transcription and speaker diarization, and learns individual speaker voices over time for automatic identification in future calls.  
**Hardware Constraint:** Must run acceptably on a typical HP laptop with **no dedicated GPU and no CUDA** (Intel/AMD CPU only, 8–16 GB RAM).  
**Success Criteria:**  
- Records Teams calls without joining as a bot  
- Produces timestamped, speaker-labeled transcripts entirely on-device  
- Stores and matches speaker embeddings across sessions  
- All processing stays local  
- Exports clean transcripts suitable for LLM summarization / todo generation  
- Runs on CPU-only hardware without requiring NVIDIA GPU  
**Timeline:** Phased delivery (core capture → transcription → diarization → speaker memory)  
**Strategic Alignment:** Privacy-first local alternative to cloud meeting tools for knowledge workers who want structured meeting notes.

## Strategic Context

**Business Value:** Enables private, offline meeting capture + structured output for personal productivity and LLM-powered follow-up.  
**User Pain Point:** Cloud tools send audio off-device; free local tools lack persistent speaker recognition across calls.  
**Competitive Advantage:** Dual-channel capture + local pyannote diarization + cross-session speaker embedding memory, all running on ordinary laptop CPUs.  
**Constraints:** Windows-first (Windows 10/11), fully local, open-source friendly stack, **CPU-only**.

## Hardware Constraint (Mandatory)

The application MUST run acceptably on a typical HP laptop with:
- No dedicated NVIDIA GPU
- No CUDA
- Intel or AMD CPU only
- 8–16 GB RAM

Requirements:
- Default to CPU + int8 quantization for faster-whisper
- Do NOT require or assume CUDA
- Provide clear progress indicators because processing will be slower
- Prefer smaller Whisper models as the default
- Optimize for memory usage (avoid loading multiple large models simultaneously if possible)
- Document expected processing time for a 30- and 60-minute meeting on CPU

## Phased Implementation Plan

### Phase 1: Audio Capture & Recording Foundation (Estimated: 12–16 hours)

## Scope
What WILL be implemented:
- Dual-channel recording: microphone + system/app audio (WASAPI loopback)
- Windows 11 per-app capture targeting Teams (or user-selected process)
- Fallback system-wide loopback for Windows 10
- Record / Pause / Resume / Stop controls with live audio level meters
- Save separate mic.wav, system.wav, and combined.wav per session
- Basic recording metadata (start time, duration, selected devices/apps)
- Local folder structure for each recording

## Explicit Exclusions
- ❌ Transcription
- ❌ Diarization
- ❌ Speaker learning
- ❌ Live UI transcript
- ❌ AI summaries

## Prerequisites
- [ ] Python 3.10+ environment
- [ ] Windows machine with microphone and Teams installed for testing

## Dependencies
- pyaudiowpatch or equivalent WASAPI capture
- soundfile / pydub
- PyQt6 (recommended)

## Success Criteria
- [ ] Can record a Teams call and produce separate mic + system tracks
- [ ] Per-app capture isolates Teams audio on Windows 11
- [ ] Recordings are saved cleanly to disk
- [ ] No audio is sent off-device

## Testing Requirements
- Manual test: 5-minute Teams call
- Verify both channels contain expected audio
- Unit tests for device enumeration and file writing

## Verification Steps
1. Launch app, select Teams + mic
2. Record 2–3 minutes of call
3. Inspect output folder for mic.wav, system.wav, combined.wav
4. Confirm level meters respond

—

### Phase 2: Transcription Engine (CPU-Only) (Estimated: 10–14 hours)

## Scope
What WILL be implemented:
- Post-recording transcription using faster-whisper on **CPU with int8 quantization**
- Support for model sizes: tiny, base, small, medium (large models optional but not recommended as default)
- Default model = `small` (best speed/accuracy balance on CPU)
- Timestamped segments written to transcript.json
- Simple TXT and SRT export
- Clear progress indicator with estimated remaining time
- Settings panel showing relative speed expectations for each model on CPU

## Explicit Exclusions
- ❌ CUDA / GPU path as a requirement or primary path
- ❌ Live/streaming transcription (Phase 4)
- ❌ Speaker labels
- ❌ Diarization

## Prerequisites
- [ ] Phase 1 complete and recording reliably

## Dependencies
- faster-whisper ≥1.1 (CPU/int8)
- No CUDA dependency

## Success Criteria
- [ ] Full recording produces accurate timestamped transcript on CPU
- [ ] Completes a 30-minute recording in reasonable time on a typical laptop CPU without crashing or excessive memory use
- [ ] Exports work
- [ ] Default model is CPU-friendly

## Testing Requirements
- Transcribe a known 10-minute recording on CPU
- Test memory usage during a longer recording
- Confirm progress indicator is useful

## Verification Steps
1. Load a Phase 1 recording
2. Run transcription with default `small` model on CPU
3. Export TXT and SRT
4. Confirm timestamps roughly match speech

—

### Phase 3: Speaker Diarization (CPU-Only) (Estimated: 14–18 hours)

## Scope
What WILL be implemented:
- Integration with pyannote.audio for full multi-speaker diarization running on CPU
- Simple “You vs Remote” mode from dual-channel energy (no token required, very fast)
- Guided setup for Hugging Face token (optional for full mode)
- Generic labels (SPEAKER_00, SPEAKER_01…) assigned to segments
- Speaker Name Panel: user can rename labels; names saved per recording
- Interactive transcript: click segment to play corresponding audio
- Optional min/max speaker count hints
- Clear UI warning that full multi-speaker diarization is slower on CPU

## Explicit Exclusions
- ❌ Persistent voice learning across sessions (Phase 4)
- ❌ Automatic name suggestion from past meetings
- ❌ GPU acceleration requirement

## Prerequisites
- [ ] Phase 2 transcription working

## Dependencies
- pyannote.audio (CPU)
- Hugging Face token for full diarization models

## Success Criteria
- [ ] Multi-speaker calls receive distinct labels on CPU
- [ ] User can rename speakers and see updates in transcript
- [ ] Simple mode works without any external token and is fast
- [ ] Segment playback is accurate
- [ ] App remains stable on 8–16 GB RAM laptop

## Testing Requirements
- Diarize a 3+ person call on CPU
- Verify rename persists and appears in export
- Test simple mode on 1-on-1 call
- Monitor memory during diarization

## Verification Steps
1. Transcribe + diarize a multi-person recording
2. Rename speakers
3. Export and confirm names appear
4. Play individual segments

—

### Phase 4: Persistent Speaker Learning + Near-Live Mode (CPU-Optimized) (Estimated: 16–22 hours)

## Scope
What WILL be implemented:
- Extract speaker embeddings (voice fingerprints) after each call using CPU
- Local speaker database (SQLite + embedding storage)
- On new calls, match new speaker clusters against known embeddings (cosine similarity)
- Suggest previously learned names with confidence score
- User can confirm / correct suggestions; confirmed embeddings are updated
- Near-live / chunked processing: process audio in 20–30 second windows while recording for progressive transcript (optimized for CPU stability)
- UI shows growing transcript with provisional speaker labels
- Primary workflow remains “process after stop” with near-live as progressive enhancement

## Explicit Exclusions
- ❌ Cloud sync of voice profiles
- ❌ True real-time streaming (near-live with larger chunks is acceptable)
- ❌ Mobile app
- ❌ Calendar integration
- ❌ GPU-dependent features

## Prerequisites
- [ ] Phase 3 diarization stable on CPU

## Dependencies
- pyannote embedding models or SpeechBrain (CPU)
- SQLite
- Vector similarity (numpy)

## Success Criteria
- [ ] After naming speakers in 2–3 calls, subsequent calls auto-suggest correct names with reasonable accuracy
- [ ] Embeddings stay fully local
- [ ] Near-live transcript appears while recording without crashing on CPU
- [ ] User can always override any auto-label
- [ ] Memory usage stays acceptable on 8–16 GB laptop

## Testing Requirements
- Train on two known speakers, then test recognition on a third call
- Measure suggestion accuracy
- Confirm no data leaves the machine
- Test near-live mode on CPU for stability

## Verification Steps
1. Complete 2 calls and name speakers
2. Start a new call with the same people
3. Confirm auto-suggestions appear
4. Export final named transcript

—

### Phase 5: Polish, Export & LLM Handoff (Estimated: 8–12 hours)

## Scope
What WILL be implemented:
- Clean Markdown / TXT export optimized for LLM consumption (with speaker names and timestamps)
- Optional local LLM summary / action-item generation (Ollama or llama-cpp-python on CPU)
- Recording browser and basic search
- Settings persistence
- Basic error logging and crash dialog
- Dark theme UI
- Clear documentation of expected CPU processing times

## Explicit Exclusions
- ❌ Team/shared voice profiles
- ❌ Cloud LLM by default
- ❌ GPU-dependent features

## Success Criteria
- [ ] Exported transcript is immediately usable by an LLM for todos and summary
- [ ] App is usable for daily work on a CPU-only HP laptop without major friction
- [ ] Processing times are clearly communicated to the user

## Testing Requirements
- End-to-end: record → transcribe → diarize → name/learn → export → feed to LLM on CPU-only machine

—

## Technical Architecture

### Technology Stack
- Language: Python 3.10+
- UI: PyQt6
- Audio: pyaudiowpatch / WASAPI
- Transcription: faster-whisper (CPU + int8, default model `small`)
- Diarization + Embeddings: pyannote.audio (CPU)
- Storage: SQLite + local files
- Optional AI: Ollama or llama-cpp-python (CPU)

### File Structure (suggested)
```
app/
  audio/          # Capture and device management
  transcription/  # Whisper pipeline (CPU-focused)
  diarization/    # pyannote + embedding logic
  speakers/       # Persistent speaker database & matching
  ui/             # PyQt windows and widgets
  storage/        # Recording folders + SQLite
  exports/
main.py
```

### Key Constraints for the Agent
- All processing MUST remain local. No audio or embeddings may be uploaded.
- The application MUST run on CPU-only hardware (no CUDA required or assumed).
- Default to smaller, faster models and int8 quantization.
- Prefer extending patterns from existing open-source tools (especially TalkTrack) where possible.
- Every phase must be independently testable.
- Explicitly document any Hugging Face model licenses the user must accept.
- Provide clear progress indicators and realistic time estimates for CPU processing.
- Optimize memory usage for 8–16 GB RAM laptops.

## Non-Functional Requirements

- **Privacy:** Zero network calls for core recording/transcription/diarization.
- **Performance (CPU-only):** Transcription of a 60-minute call should complete in a practical timeframe on a mid-range laptop CPU (document actual expectations). Full diarization will be slower and should be clearly indicated.
- **Reliability:** Recordings must not be lost on crash; write incrementally where practical.
- **Usability:** First-time setup should guide the user through Hugging Face token creation if they want full diarization. Progress and expected wait times must be visible.
- **Memory:** Must remain stable on 8–16 GB RAM systems.

—

**Instructions for the coding agent:**

Implement one phase at a time. After completing a phase, stop and report:
1. What was built
2. How to test it
3. Any deviations from this PRD and why
4. Observed CPU performance and memory usage
5. Recommended next phase adjustments

Start with Phase 1. Do not implement later phases until the current phase’s success criteria are met.

The application must be fully usable on a typical HP laptop with no GPU/CUDA.
