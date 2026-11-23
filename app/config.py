"""OffAir Phase 1 configuration — defaults and paths.

CPU-only, Windows 10/11, Python 3.10+. No CUDA / GPU anywhere in this
project. All values here are overridable; nothing is hard-coded in the
capture or UI code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Audio defaults (PRD Phase 1) -------------------------------------------
SAMPLE_RATE: int = 48000          # Hz
CHANNELS: int = 1                 # mono
BIT_DEPTH: int = 16               # PCM16
CHUNK_FRAMES: int = 1024          # frames per capture callback chunk
CHUNK_FRAMES_LARGE: int = 2048    # alternate chunk size (lower overhead)
SAMPLE_FORMAT: str = "int16"

# --- System-audio capture mode (Nov 8 2026, per-app capture) --------------
# How the system track is captured on Windows ("auto" tries Teams-only
# per-app capture — app/audio/perapp_capture.py — and falls back to
# system-wide loopback when per-app is unavailable or fails; "per-app"
# is Teams-only and fails loudly instead of falling back; "loopback"
# is always system-wide, the pre-Nov-8 behaviour). Linux always uses
# its monitor source regardless of this setting.
CAPTURE_SYSTEM_MODES: tuple[str, ...] = ("auto", "per-app", "loopback")
CAPTURE_SYSTEM_MODE_DEFAULT: str = "auto"

# --- Transcription (PRD Phase 2) -------------------------------------------
# faster-whisper, CPU-only, int8. Default model is "base" since
# Nov 19 2026 (was "small" per the PRD): John's report on 1.0.2 —
# transcription of a 30-minute meeting crashed / never finished —
# settled the re-check the PRD had left open. On the 61.3 s fixture
# base matched small's accuracy (both 7.5% WER) at 2.5x less time
# (RTF 0.135 vs 0.333), and needs ~550 MB less RAM while running.
# "expected_30min_seconds" figures are extrapolations from measured
# RTFs on this Linux VM (4 vCPU, ~650 MB RAM free) — see PHASE2.md.
# Medium is still an unmeasured estimate. Measured numbers are
# recorded in PHASE2.md. The "expected_30min_seconds" figures below
# are EXTRAPOLATIONS from the measured real-time factors on this
# Linux VM (61.3 s fixture, 4 vCPU, ~650 MB RAM free) — i.e.
# expected = RTF x 1800 s — not measurements of a 30-minute meeting,
# and a modern 8-16 GB laptop should beat them. Medium was not
# measured here.
#   tiny  RTF 0.070 -> ~130 s per 30 min, ~260 s per 60 min
#   base  RTF 0.135 -> ~240 s per 30 min, ~490 s per 60 min
#   small RTF 0.333 -> ~600 s per 30 min, ~1200 s per 60 min
WHISPER_DEFAULT_MODEL: str = "base"
# Past defaults that differ from the current one, oldest first.
# Used only by the pre-flag migration in config_from_settings():
# a settings file saved before whisper_model_explicit existed
# cannot tell an explicit pick from the then-default, so a stored
# model that WAS a past default is treated as "never explicitly
# picked" and the current default wins. "small" was the default
# under 1.0.2 (and the PRD before Nov 19 2026); no other past
# default differs from "base".
PAST_WHISPER_DEFAULTS: tuple[str, ...] = ("small",)
WHISPER_DEVICE: str = "cpu"
WHISPER_COMPUTE_TYPE: str = "int8"

# Per-model notes. "speed" is relative on the same CPU (tiny = fastest).
# "params" / "size_mb" are the published Whisper figures.
WHISPER_MODELS: dict[str, dict] = {
    "tiny": {
        "params": "39M", "size_mb": 75,
        "speed": "fastest (~4x faster than small)",
        "accuracy": "lowest — usable for drafts / search, expect word errors",
        "expected_30min_seconds": 130,  # extrapolated from measured RTF 0.070
        "note": "Safest on low-RAM machines; default fallback if base OOMs.",
    },
    "base": {
        "params": "74M", "size_mb": 142,
        "speed": "fast (~2x faster than small)",
        "accuracy": "good for clear speech; noticeably better than tiny",
        "expected_30min_seconds": 240,  # extrapolated from measured RTF 0.135
        "note": "Default since Nov 19 2026 — same accuracy as small on "
                "clean audio (both 7.5% WER on the 61.3 s fixture) at "
                "2.5x less time.",
    },
    "small": {
        "params": "244M", "size_mb": 466,
        "speed": "moderate — former default, now opt-in",
        "accuracy": "same as base on clean audio (both 7.5% WER), 2.5x slower",
        "expected_30min_seconds": 600,  # extrapolated from measured RTF 0.333
        "note": "Former PRD default; opt-in since Nov 19 2026. "
                "Needs ~1 GB free RAM while running.",
    },
    "medium": {
        "params": "769M", "size_mb": 1500,
        "speed": "slow (~3-4x slower than small)",
        "accuracy": "better than small, rarely worth the wait on CPU",
        "expected_30min_seconds": 2700,  # ESTIMATE, see PHASE2.md
        "note": "Not recommended as a default; opt-in only, needs 8+ GB free.",
    },
}
WHISPER_SUPPORTED_MODELS: tuple[str, ...] = tuple(WHISPER_MODELS)

# --- Diarization (PRD Phase 3) ------------------------------------------------
# Simple "You vs Remote" mode needs no model and no token. Full mode uses
# pyannote/speaker-diarization-3.1 on CPU (see
# app/diarization/pyannote_diarize.py for the HF token + licence setup).
# Sherpa mode (sherpa-onnx, no token) is the default since Nov 8 2026:
# multi-speaker labels for every call, not just You vs Remote — that is
# what speaker learning across recordings needs. A segment is labelled
# "You" / "Remote" when one track's RMS is at least
# DIARIZATION_DOMINANCE_RATIO times the other's; below
# DIARIZATION_SILENCE_RMS on both tracks it is "Silence".
DIARIZATION_DEFAULT_MODE: str = "sherpa"
DIARIZATION_SILENCE_RMS: float = 0.005
DIARIZATION_DOMINANCE_RATIO: float = 1.5
DIARIZATION_PYANNOTE_MODEL: str = "pyannote/speaker-diarization-3.1"

# --- Diarization, sherpa mode (Nov 8 2026) ----------------------------------
# sherpa-onnx OfflineSpeakerDiarization: pyannote segmentation ONNX +
# 3D-Speaker ERes2Net embeddings, CPU only, no Hugging Face token.
# Models download once on first use from public GitHub releases
# into SHERPA_MODEL_DIR (see app/diarization/sherpa.py for the URLs
# and the measured numbers). Settings:
# * sherpa_expected_speakers: 0 = auto (threshold clustering
#   discovers the speaker count); 1–10 pins the cluster count.
# * sherpa_clustering_threshold: cosine-similarity merge threshold
#   for auto mode. 0.5 is sherpa-onnx's default and sits midway
#   between the measured same-speaker (0.82–0.89) and cross-speaker
#   (~0.0–0.02) ERes2Net scores on the fixture; 0.5/0.6/0.7 all
#   gave identical results there (see app/diarization/sherpa.py).
SHERPA_MODEL_DIR: Path = (Path.home() / ".offair" / "models"
                          / "sherpa-diarization")
SHERPA_DEFAULT_EXPECTED_SPEAKERS: int = 0
SHERPA_DEFAULT_CLUSTERING_THRESHOLD: float = 0.5

# --- Speaker learning (PRD Phase 4) -----------------------------------------
# Persistent speaker memory lives in speakers.db in the recordings
# root. A suggestion is made only when the best cosine similarity
# clears the threshold for that embedding backend (see
# SPEAKER_MATCH_THRESHOLDS). Thresholds are per-backend because the
# backends live in different vector spaces with different
# same-speaker scores:
# * fallback-spectral-v1: 0.90 — measured on the fixture (Priya
#   separates cleanly, but John vs Marcus cross-speaker cosine is
#   0.95, above John's same-speaker half-vs-half score of 0.88).
#   The threshold is deliberately high and the UI must show the
#   margin — a small margin means "nearly tied", even at a high
#   absolute score.
# * onnx-ecapa-v1: 0.55 — measured Nov 8 2026 on the fixture:
#   same-speaker half-vs-half 0.81–0.88, cross-speaker 0.05–0.12.
#   0.55 sits between them, favouring precision (a wrong name is
#   worse than "Unknown"). Re-measure on real meeting audio.
# * speechbrain-ecapa-v1: 0.55 — PROVISIONAL, same ECAPA-TDNN
#   family / VoxCeleb training as the ONNX model; re-measure when
#   it first runs on the target laptop.
# * pyannote-embedding-v1: 0.90 — UNMEASURED (never ran here);
#   re-measure before trusting suggestions from it.
SPEAKER_MATCH_THRESHOLD: float = 0.90  # default for unknown backends
SPEAKER_MATCH_THRESHOLDS: dict[str, float] = {
    "fallback-spectral-v1": 0.90,
    "pyannote-embedding-v1": 0.90,
    "onnx-ecapa-v1": 0.55,
    "speechbrain-ecapa-v1": 0.55,
    # sherpa-eres2net-v1: 0.55 — measured Nov 8 2026 on the fixture:
    # same-speaker half-vs-half 0.82–0.89, cross-speaker ~0.0–0.02.
    # 0.55 sits between them, favouring precision (a wrong name is
    # worse than "Unknown"). Re-measure on real meeting audio.
    "sherpa-eres2net-v1": 0.55,
}
SPEAKER_MATCH_AMBIGUOUS_MARGIN: float = 0.05
SPEAKERS_DB_FILE: str = "speakers.db"

# Embedding backend for speaker learning (ECAPA swap, Nov 8 2026):
# "auto" (default) picks the best available production backend —
# SpeechBrain ECAPA when torch + speechbrain import, otherwise
# ONNX ECAPA when speakeronnx imports, otherwise the fallback
# spectral extractor. The other names pin one backend:
# "speechbrain-ecapa" (needs torch + speechbrain),
# "onnx-ecapa" (needs speakeronnx + onnxruntime, no torch),
# "pyannote" (opt-in; needs torch + pyannote.audio + HF token),
# "fallback-spectral-v1" (test-only spectral proxy).
# "sherpa-eres2net" (Nov 8 2026): sherpa-onnx 3D-Speaker ERes2Net,
# 192-dim — the best separation of everything tested (margin 0.810,
# cross-speaker ~0); needs sherpa-onnx + its models (same models as
# sherpa diarization, downloaded once on first use).
EMBEDDING_BACKENDS: tuple[str, ...] = (
    "auto",
    "speechbrain-ecapa",
    "onnx-ecapa",
    "pyannote",
    "sherpa-eres2net",
    "fallback-spectral-v1",
)
EMBEDDING_DEFAULT_BACKEND: str = "auto"

# --- Near-live mode (PRD Phase 4) -------------------------------------------
# 25 s windows, 0.5 s overlap (words at a boundary are covered by
# both windows; the chunker de-duplicates the second copy). The
# near-live model is the CONFIGURED model (Nov 20 2026, Item 7):
# live chunks and the manual near-live run both use
# ``_configured_model()`` — base on a fresh install. Base keeps up
# with real time too (RTF ~0.135: a 25 s chunk transcribes in
# ~3.4 s) with small-level accuracy, so there is no reason to pin
# near-live to tiny. ``NEARLIVE_DEFAULT_MODEL`` stays only as the
# code fallback (the ChunkedTranscriber default parameter, tests)
# and is never the user-facing default.
NEARLIVE_CHUNK_SECONDS: float = 25.0
NEARLIVE_CHUNK_OVERLAP_SECONDS: float = 0.5
NEARLIVE_DEFAULT_MODEL: str = "tiny"

# --- Settings persistence (PRD Phase 5) --------------------------------------
# Settings are stored as JSON in ~/.offair/settings.json (the user
# config dir). Only the allowlisted keys in _SETTINGS_KEYS are ever
# written — a Hugging Face token is NEVER stored: only the boolean
# ``hf_token_configured`` flag ("a token is configured in-memory /
# in the environment") is persisted. Unknown keys in a settings file
# are ignored; invalid values fall back to defaults.
SETTINGS_FILE_NAME: str = "settings.json"
SETTINGS_DIR_NAME: str = ".offair"
THEMES: tuple[str, ...] = ("dark", "light")
# Must match DEFAULT_THEME in app/ui/theme.py (tests assert they
# agree). Light is the default — the clean, airy consumer look.
DEFAULT_THEME: str = "light"
OLLAMA_DEFAULT_URL: str = "http://localhost:11434"
# Default Ollama model (Nov 8 2026, research verdict — see README):
# qwen3.5:9b, not llama3.1. Qwen3.5-9B (Alibaba, Mar 2026, Apache
# 2.0) is a generation newer than Llama 3.1/3.3, has a 256K
# context window (vs 128K), beats the Llama 8B options on the
# general benchmarks that exist, and is a permissive Apache 2.0
# licence rather than Meta's Llama Community License. 6.6 GB
# download; needs ~8 GB free RAM, so on an 8 GB machine use
# qwen3.5:4b instead. Ollama stays OFF by default — this only
# picks the model once the user enables it.
OLLAMA_DEFAULT_MODEL: str = "qwen3.5:9b"

# --- Enterprise-LLM cleanup (Nov 21 2026) -------------------------------------
# Cleanup via the headless Claude Code CLI (`claude -p`, subprocess
# only — never ACP/A2A). Opt-in, OFF by default: transcript TEXT
# ONLY leaves the machine, audio never does (see
# app/transcription/enterprise_cleanup.py). The timeout bounds one
# run; ``claude_command`` is the CLI name on PATH or an absolute
# path override. No Ollama default is changed by this block.
ENTERPRISE_CLEANUP_DEFAULT_ENABLED: bool = False
ENTERPRISE_CLEANUP_DEFAULT_TIMEOUT_SECONDS: float = 120.0
ENTERPRISE_CLEANUP_DEFAULT_COMMAND: str = "claude"

# --- tinyfix (Nov 21 2026) --------------------------------------------------
# Real-word cleanup (paperweight/tinyfix, MIT) as a stage of
# transcription cleanup, after the deterministic rules and the
# names dictionary. Default OFF since Nov 22 2026 (was ON):
# on disfluent real speech the stage changed correct words
# ("lace" -> "place", "hate" -> "have", both at 0.9998
# confidence) while fixing nothing — see
# docs/notes/cleanup-repeat-rule-2026-11-22.md. It stays
# available as an explicit opt-in. English only: the stage is
# skipped when the transcript language is known non-English.
# The model (2.1 MB ONNX) is downloaded once on first use from
# public Hugging Face into ~/.offair/models/tinyfix/<revision/>
# and verified by hash; after that it works fully offline.
# onnxruntime only, no torch.
TINYFIX_DEFAULT_ENABLED: bool = False
TINYFIX_MODES: tuple[str, ...] = ("balanced", "careful")
#: Measured Nov 21 2026 (docs/notes/tinyfix-2026-11-21.md):
#: careful made zero false changes on the clean fixture script
#: where balanced made one ("clock" -> "click"), while keeping
#: the core mix-up fixes. The package default is balanced; the
#: shipped default here is careful because a default-on stage
#: must do no harm to correct text first.
TINYFIX_DEFAULT_MODE: str = "careful"

# --- Semantic search (Nov 8 2026) -------------------------------------------
# Transcript search modes: "text" (Phase 5 substring search), "semantic"
# (MiniLM vector search) and "hybrid" (substring hits boosted first +
# semantic hits). "hybrid" is the default since Nov 8 2026 — exact
# matches still come first, and meaning-based matches fill in what the
# exact words miss. The embedding model is sentence-transformers/all-MiniLM-L6-v2
# (384-dim), ONNX int8 (~22 MB), downloaded once on first
# semantic use from public Hugging Face (no token / login) into
# SEMANTIC_MODEL_DIR; after that it works fully offline. No
# torch / transformers anywhere — onnxruntime only.
SEARCH_MODES: tuple[str, ...] = ("text", "semantic", "hybrid")
DEFAULT_SEARCH_MODE: str = "hybrid"
SEMANTIC_MODEL_REPO: str = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MODEL_DIR: Path = (Path.home() / ".offair" / "models"
                            / "all-MiniLM-L6-v2")
SEMANTIC_DEFAULT_THRESHOLD: float = 0.15
SEMANTIC_CHUNK_MAX_WORDS: int = 40
SEMANTIC_INDEX_FILE: str = "search_index.db"

# --- Target application ------------------------------------------------------
DEFAULT_APP_NAME: str = "Teams"
TEAMS_PROCESS_NAMES: list[str] = [
    "ms-teams.exe",
    "Teams.exe",
    "ms-teamsupdate.exe",
]

# --- Auto-detect + auto-record (Nov 8 2026) ----------------------------------
# Meeting watcher (app/audio/auto_detect.py): 1 Hz polling of the
# target app's WASAPI audio sessions. Start gate = session Active
# AND peak above threshold sustained for AUTO_DETECT_SUSTAIN_SECONDS
# (2–3 s debounce — a ringtone or notification blip never starts a
# recording). Stop gate (revised Nov 8 2026 — the old 20 s silence
# rule was wrong: real meetings pause longer than 20 s): silence
# mid-meeting NEVER stops a recording. Stop only on (a) the session
# going Inactive/Expired and staying that way for
# AUTO_DETECT_INACTIVE_GRACE_SECONDS (60 s grace covers transient
# stream renegotiation mid-call), (b) the target process exiting,
# or (c) the long-silence fuse — session Active with peak at/below
# silence AND the mic unclaimed continuously for
# AUTO_DETECT_SILENCE_FUSE_SECONDS (10 min; a stuck/leftover
# session, not a real pause). Secondary start rule: session Active
# for AUTO_DETECT_ACTIVE_MIC_SECONDS (30 s) with the mic claimed
# counts as a meeting even with no peak (all-muted meeting). The
# stop gate rests on Teams keeping its session Active across quiet
# stretches — that assumption MUST be verified on a real Windows
# Teams call (see docs/PER_APP_CAPTURE.md). All values are tunable
# here and injectable in the watcher constructor for tests.
# auto_detect_teams is OPT-IN and default OFF — a privacy decision,
# not a default; see docs/PER_APP_CAPTURE.md.
AUTO_DETECT_POLL_INTERVAL_SECONDS: float = 1.0
AUTO_DETECT_SUSTAIN_SECONDS: float = 2.5
AUTO_DETECT_INACTIVE_GRACE_SECONDS: float = 60.0
AUTO_DETECT_SILENCE_FUSE_SECONDS: float = 600.0
AUTO_DETECT_ACTIVE_MIC_SECONDS: float = 30.0
AUTO_DETECT_PEAK_THRESHOLD: float = 0.01
AUTO_DETECT_SILENCE_PEAK: float = 0.001

# --- Storage -----------------------------------------------------------------
DEFAULT_OUTPUT_ROOT: Path = Path.home() / "OffAir" / "recordings"


def get_default_output_root() -> Path:
    """Default folder recordings are saved under (overridable via CLI/UI)."""
    return DEFAULT_OUTPUT_ROOT


# --- Audio retention (Nov 8 2026) -------------------------------------------
# Recording audio (mic.wav / system.wav / combined.wav) is deleted
# automatically once it is older than the retention age — but ONLY
# for sessions that have a transcript (transcript.json), and never
# for a session that is still recording; transcripts, diarization,
# speaker names and metadata.json are always kept (see
# app/storage/retention.py). A 1-hour meeting is ~1 GB of WAV
# (3 tracks x ~5.7 MB/min), so a day of meetings is several GB if
# nothing is ever pruned. None = never delete audio ("Never" in
# Settings). The value is hours; the dialog offers 24 h / 7 d /
# 30 d / Never.
AUDIO_RETENTION_DEFAULT_HOURS: float = 24.0
AUDIO_RETENTION_OPTIONS: tuple[tuple[str, float | None], ...] = (
    ("24 hours", 24.0),
    ("7 days", 7 * 24.0),
    ("30 days", 30 * 24.0),
    ("Never — keep all audio", None),
)

# --- Low-disk warning (Nov 21 2026) -----------------------------------------
# While recording, free space on the recordings volume is checked
# against this threshold and a warning fires before the disk fills
# (see app/storage/retention.py disk_status — the check only
# reports; a recording is never stopped for disk reasons). 1 GiB
# free is roughly one more hour of recording headroom (a 1-hour
# meeting is ~1 GB of WAV across the three tracks). Must equal
# app.storage.retention.DISK_WARNING_FREE_BYTES (tests pin it).
DISK_WARNING_THRESHOLD_BYTES: int = 1024 ** 3

# --- Pre-roll (Nov 21 2026) -------------------------------------------------
# Opt-in, default OFF everywhere: when enabled, audio is captured
# into a memory-only ring buffer before Record is pressed, and the
# buffered seconds lead a recording that starts (manual or
# auto-detect). The buffer never touches disk unless a recording
# starts (see app/audio/preroll.py). preroll_seconds is the buffer
# length, 10–120 s. The consent note below is the exact text the
# Settings UI shows when the user turns pre-roll on — it must say
# plainly that the app captures before Record is pressed.
PREROLL_DEFAULT_ENABLED: bool = False
PREROLL_DEFAULT_SECONDS: float = 30.0
PREROLL_MIN_SECONDS: float = 10.0
PREROLL_MAX_SECONDS: float = 120.0
PREROLL_CONSENT_NOTE: str = (
    "Pre-roll captures audio before you press Record. When pre-roll "
    "is on, OffAir keeps recording the microphone and the selected "
    "app's audio into a short buffer held only in memory "
    "(30 seconds by default). The buffered audio is never written "
    "to disk unless a recording starts: if you press Record, or "
    "auto-detect starts a recording, the buffered audio is added "
    "to the start of that recording. If no recording starts, the "
    "buffered audio is discarded and never saved. When pre-roll "
    "is off, nothing is captured before you press Record.")


@dataclass
class AppConfig:
    """Runtime configuration for one OffAir run."""

    output_root: Path = field(default_factory=get_default_output_root)
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    chunk_frames: int = CHUNK_FRAMES
    app_name: str = DEFAULT_APP_NAME
    process_names: list[str] = field(default_factory=lambda: list(TEAMS_PROCESS_NAMES))
    # Device indices as reported by app.audio.devices; None = system default.
    microphone_device_index: int | None = None
    loopback_device_index: int | None = None
    # System-audio capture mode (Nov 8 2026): "auto" / "per-app" /
    # "loopback" — see CAPTURE_SYSTEM_MODES above.
    capture_system_mode: str = CAPTURE_SYSTEM_MODE_DEFAULT
    # Default recording source (Nov 20 2026): the persisted
    # default for "what this recording is of", edited in Settings
    # (the Record view's Source dropdown is gone). None = Auto:
    # detect from the picked app at record time (Teams / Zoom /
    # browser / solo — see app.storage.source); an explicit value
    # wins over detection (the only way to name a source no
    # process identifies, e.g. a Claude Code session).
    default_source: str | None = None
    # Transcription (Phase 2): faster-whisper model, CPU/int8 only.
    whisper_model: str = WHISPER_DEFAULT_MODEL
    # Whether the user explicitly picked whisper_model in Settings
    # (Nov 20 2026, Item 8). The model default must NOT carry over
    # between versions: a setting saved under 1.0.2 ("small", the
    # old default) would otherwise keep the user on small forever
    # and they would never get the new base default. So the
    # effective model is the current build's default unless this
    # flag is True — see effective_whisper_model. The flag is set
    # ONLY by a user-driven pick in the Settings dialog's model
    # combo (its `activated` signal); never by a plain Settings
    # save, by loading, or by programmatic combo syncs. Honest
    # caveat: an explicit "small" pick made under 1.0.2/1.0.3 is
    # indistinguishable from the old default and resets to base
    # once on upgrade; re-picking small in Settings sets the flag
    # and persists forever after.
    whisper_model_explicit: bool = False
    # Diarization (Phase 3): "sherpa" (default — multi-speaker, no
    # token), "simple" (You vs Remote, no model) or "full" (pyannote).
    diarization_mode: str = DIARIZATION_DEFAULT_MODE
    # Sherpa diarization settings (Nov 8 2026): model dir override
    # (None = the default ~/.offair/models/sherpa-diarization),
    # expected speaker count hint (0 = auto, threshold clustering),
    # and the clustering threshold for auto mode.
    sherpa_model_dir: Path | None = None
    sherpa_expected_speakers: int = SHERPA_DEFAULT_EXPECTED_SPEAKERS
    sherpa_clustering_threshold: float = SHERPA_DEFAULT_CLUSTERING_THRESHOLD
    # Speaker embedding backend (Phase 4, ECAPA swap): "auto" picks
    # the best available backend (SpeechBrain ECAPA, else ONNX
    # ECAPA, else the fallback); see EMBEDDING_BACKENDS above.
    embedding_backend: str = EMBEDDING_DEFAULT_BACKEND
    # NEVER a token value: only whether one is configured in-memory /
    # in the environment (HF_TOKEN). OffAir never writes a token
    # to disk — see save_settings() (allowlisted keys only).
    hf_token_configured: bool = False
    # Near-live (Phase 4): enabled flag + chunk length (5–60 s).
    near_live_enabled: bool = False
    nearlive_chunk_seconds: float = NEARLIVE_CHUNK_SECONDS
    # UI theme: "light" (default — the clean, airy consumer look)
    # or "dark" (equally finished alternative).
    theme: str = DEFAULT_THEME
    # Local LLM summaries (Phase 5): off by default, localhost only.
    ollama_enabled: bool = False
    ollama_model: str = OLLAMA_DEFAULT_MODEL
    ollama_base_url: str = OLLAMA_DEFAULT_URL
    # Transcript AI polish (Nov 10 2026): when True, the auto-process
    # chain also runs the local-LLM polish pass after cleaning (see
    # app/transcription/cleanup.py). OFF by default — the polish is
    # an AI rewrite, clearly marked as such, and the verbatim
    # transcript stays the record. The Session view's "Polish"
    # button runs it on demand regardless of this setting.
    transcript_polish_enabled: bool = False
    # Enterprise-LLM cleanup (Nov 21 2026): headless `claude -p`
    # subprocess, text only, audio never leaves the machine.
    # OFF by default — an explicit opt-in. See
    # app/transcription/enterprise_cleanup.py.
    enterprise_cleanup_enabled: bool = ENTERPRISE_CLEANUP_DEFAULT_ENABLED
    enterprise_cleanup_timeout_seconds: float = (
        ENTERPRISE_CLEANUP_DEFAULT_TIMEOUT_SECONDS)
    enterprise_cleanup_claude_command: str = ENTERPRISE_CLEANUP_DEFAULT_COMMAND
    # tinyfix (Nov 21 2026): real-word cleanup stage in
    # transcription cleanup, after the deterministic rules and
    # the names dictionary. OFF by default since Nov 22 2026 —
    # an explicit opt-in; English only. Mode "balanced" fixes
    # more wrong words; "careful" makes fewer wrong changes.
    # See app/transcription/tinyfix.py.
    tinyfix_enabled: bool = TINYFIX_DEFAULT_ENABLED
    tinyfix_mode: str = TINYFIX_DEFAULT_MODE
    # Whether the user explicitly chose the tinyfix setting
    # (Nov 22 2026 — mirrors whisper_model_explicit, Nov 20
    # 2026): set only by a user-driven toggle of the Settings
    # checkbox, never by a plain save/load. A pre-flag settings
    # file with tinyfix_enabled=True cannot be told apart from
    # the old ON default, so it migrates to OFF — see
    # config_from_settings(). Re-enabling in Settings sets the
    # flag and the choice persists.
    tinyfix_enabled_explicit: bool = False
    # Semantic transcript search (Nov 8 2026): default search mode
    # for the Recording Browser ("hybrid" = exact matches first,
    # then meaning; "text" = Phase 5 substring search; "semantic" /
    # "hybrid" need the MiniLM model, downloaded once on first use).
    # Model dir override: None means the default
    # (~/.offair/models/all-MiniLM-L6-v2).
    search_mode: str = DEFAULT_SEARCH_MODE
    semantic_model_dir: Path | None = None
    # Auto-process recordings (Nov 8 2026): when True (the default),
    # stopping a recording automatically runs transcribe → diarize →
    # suggest + identify speakers in the background (see
    # app/ui/main_window.py). The recording is saved first, so a
    # failed auto-step can never lose it; every step can still be
    # re-run manually from the Session view. Turn it off in Settings
    # to go back to fully manual processing.
    auto_process_recordings: bool = True
    # Auto-detect + auto-record (Nov 8 2026): when True, the meeting
    # watcher (app/audio/auto_detect.py) starts a recording when a
    # Teams meeting starts and stops it when the meeting ends.
    # OPT-IN, default OFF — auto-recording is a privacy decision:
    # OffAir does NOT trigger Teams' own recording banner, so the
    # user must tell participants they are recording. Windows only;
    # on Linux the setting is shown disabled in Settings.
    auto_detect_teams: bool = False
    # Audio retention (Nov 8 2026): hours after which a transcribed
    # session's WAVs are auto-deleted; transcripts are always kept.
    # None = never delete audio. See app/storage/retention.py.
    audio_retention_hours: float | None = AUDIO_RETENTION_DEFAULT_HOURS
    # Low-disk warning (Nov 21 2026): warn while recording when free
    # space on the recordings volume drops below this many bytes
    # (default 1 GiB). Reporting only — never stops a recording.
    # See app/storage/retention.py disk_status.
    disk_warning_threshold_bytes: int = DISK_WARNING_THRESHOLD_BYTES
    # Pre-roll (Nov 21 2026): opt-in, default OFF. When enabled, a
    # memory-only ring buffer holds the last preroll_seconds of
    # audio before Record is pressed, and the buffered audio leads
    # a recording that starts. See app/audio/preroll.py and
    # PREROLL_CONSENT_NOTE (the exact Settings consent text).
    preroll_enabled: bool = PREROLL_DEFAULT_ENABLED
    preroll_seconds: float = PREROLL_DEFAULT_SECONDS
    # Outlook calendar linking (Nov 10 2026, Feature 3): when True
    # (the default), a session is matched to its Outlook calendar
    # event when it finalises and when an old session loads, and the
    # result is stored in metadata.json under "calendar_event".
    # READ-ONLY — nothing is ever written into Outlook (see
    # app/calendar/outlook.py). Windows + Outlook only; on other
    # machines the setting is shown disabled in Settings and the
    # link is simply never attempted.
    outlook_link_enabled: bool = True

    @property
    def effective_whisper_model(self) -> str:
        """The Whisper model actually in use (Nov 20 2026, Item 8).

        Single source of truth for every reader (Settings dialog,
        main window, live/transcribe paths): an explicitly picked
        model wins; otherwise the current build's default
        (WHISPER_DEFAULT_MODEL) wins, so an upgrade can move the
        default without the user ever touching Settings. Reads the
        module global at call time, so a simulated version change
        (monkeypatching WHISPER_DEFAULT_MODEL) is honoured.
        """
        if self.whisper_model_explicit:
            return self.whisper_model
        return WHISPER_DEFAULT_MODEL

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        if self.semantic_model_dir is not None:
            self.semantic_model_dir = Path(self.semantic_model_dir)
        if self.sherpa_model_dir is not None:
            self.sherpa_model_dir = Path(self.sherpa_model_dir)
        if self.search_mode not in SEARCH_MODES:
            raise ValueError(
                f"unknown search mode {self.search_mode!r}; "
                f"supported: {list(SEARCH_MODES)}")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if self.whisper_model not in WHISPER_MODELS:
            raise ValueError(
                f"unknown Whisper model {self.whisper_model!r}; "
                f"supported: {sorted(WHISPER_MODELS)}")
        if self.diarization_mode not in ("simple", "full", "sherpa"):
            raise ValueError(
                f"unknown diarization mode {self.diarization_mode!r}; "
                "expected 'simple', 'full' or 'sherpa'")
        # Sherpa settings validated by the diarization module
        # (lazy import: config is a leaf, the module pulls numpy).
        from app.diarization.sherpa import validate_sherpa_settings
        (self.sherpa_expected_speakers,
         self.sherpa_clustering_threshold) = validate_sherpa_settings(
             self.sherpa_expected_speakers,
             self.sherpa_clustering_threshold)
        if self.embedding_backend not in EMBEDDING_BACKENDS:
            raise ValueError(
                f"unknown embedding backend {self.embedding_backend!r}; "
                f"supported: {list(EMBEDDING_BACKENDS)}")
        if self.capture_system_mode not in CAPTURE_SYSTEM_MODES:
            raise ValueError(
                f"unknown capture system mode "
                f"{self.capture_system_mode!r}; supported: "
                f"{list(CAPTURE_SYSTEM_MODES)}")
        if self.default_source is not None:
            from app.storage.source import SOURCES

            if self.default_source not in SOURCES:
                raise ValueError(
                    f"unknown recording source "
                    f"{self.default_source!r}; supported: "
                    f"{list(SOURCES)}")
        if self.theme not in THEMES:
            raise ValueError(
                f"unknown theme {self.theme!r}; supported: {list(THEMES)}")
        if not 5.0 <= float(self.nearlive_chunk_seconds) <= 60.0:
            raise ValueError(
                "nearlive_chunk_seconds must be between 5 and 60")
        if (self.audio_retention_hours is not None
                and not float(self.audio_retention_hours) > 0):
            raise ValueError(
                "audio_retention_hours must be positive or None")
        if not int(self.disk_warning_threshold_bytes) > 0:
            raise ValueError(
                "disk_warning_threshold_bytes must be positive")
        if not float(self.enterprise_cleanup_timeout_seconds) > 0:
            raise ValueError(
                "enterprise_cleanup_timeout_seconds must be positive")
        if not str(self.enterprise_cleanup_claude_command).strip():
            raise ValueError(
                "enterprise_cleanup_claude_command must not be blank")
        if self.tinyfix_mode not in TINYFIX_MODES:
            raise ValueError(
                f"unknown tinyfix_mode {self.tinyfix_mode!r}; "
                f"supported: {list(TINYFIX_MODES)}")
        if not PREROLL_MIN_SECONDS <= float(self.preroll_seconds) \
                <= PREROLL_MAX_SECONDS:
            raise ValueError(
                f"preroll_seconds must be between {PREROLL_MIN_SECONDS} "
                f"and {PREROLL_MAX_SECONDS}")


# --- Settings file I/O ------------------------------------------------------
_SETTINGS_KEYS: tuple[str, ...] = (
    "output_root", "whisper_model", "whisper_model_explicit",
    "diarization_mode",
    "sherpa_model_dir", "sherpa_expected_speakers",
    "sherpa_clustering_threshold",
    "embedding_backend",
    "hf_token_configured", "near_live_enabled",
    "nearlive_chunk_seconds", "theme",
    "ollama_enabled", "ollama_model", "ollama_base_url",
    "transcript_polish_enabled",
    "enterprise_cleanup_enabled", "enterprise_cleanup_timeout_seconds",
    "enterprise_cleanup_claude_command",
    "search_mode", "semantic_model_dir",
    "capture_system_mode", "auto_process_recordings",
    "auto_detect_teams", "audio_retention_hours",
    "outlook_link_enabled", "default_source",
    "disk_warning_threshold_bytes", "preroll_enabled", "preroll_seconds",
    "tinyfix_enabled", "tinyfix_mode", "tinyfix_enabled_explicit",
)


def default_settings_path() -> Path:
    """Where settings live: ``~/.offair/settings.json``."""
    return Path.home() / SETTINGS_DIR_NAME / SETTINGS_FILE_NAME


def settings_to_dict(config: AppConfig) -> dict:
    """The allowlisted settings for ``config`` — no token, ever."""
    return {
        "output_root": str(config.output_root),
        "whisper_model": config.whisper_model,
        "whisper_model_explicit": bool(config.whisper_model_explicit),
        "diarization_mode": config.diarization_mode,
        "sherpa_model_dir": (str(config.sherpa_model_dir)
                             if config.sherpa_model_dir is not None
                             else None),
        "sherpa_expected_speakers": int(config.sherpa_expected_speakers),
        "sherpa_clustering_threshold": float(
            config.sherpa_clustering_threshold),
        "embedding_backend": config.embedding_backend,
        "hf_token_configured": bool(config.hf_token_configured),
        "near_live_enabled": bool(config.near_live_enabled),
        "nearlive_chunk_seconds": float(config.nearlive_chunk_seconds),
        "theme": config.theme,
        "ollama_enabled": bool(config.ollama_enabled),
        "ollama_model": config.ollama_model,
        "ollama_base_url": config.ollama_base_url,
        "transcript_polish_enabled": bool(
            config.transcript_polish_enabled),
        "enterprise_cleanup_enabled": bool(
            config.enterprise_cleanup_enabled),
        "enterprise_cleanup_timeout_seconds": float(
            config.enterprise_cleanup_timeout_seconds),
        "enterprise_cleanup_claude_command": (
            config.enterprise_cleanup_claude_command),
        "search_mode": config.search_mode,
        "semantic_model_dir": (str(config.semantic_model_dir)
                               if config.semantic_model_dir is not None
                               else None),
        "capture_system_mode": config.capture_system_mode,
        "default_source": config.default_source,
        "auto_process_recordings": bool(config.auto_process_recordings),
        "auto_detect_teams": bool(config.auto_detect_teams),
        "outlook_link_enabled": bool(config.outlook_link_enabled),
        "audio_retention_hours": (
            float(config.audio_retention_hours)
            if config.audio_retention_hours is not None else None),
        "disk_warning_threshold_bytes": int(
            config.disk_warning_threshold_bytes),
        "preroll_enabled": bool(config.preroll_enabled),
        "preroll_seconds": float(config.preroll_seconds),
        "tinyfix_enabled": bool(config.tinyfix_enabled),
        "tinyfix_mode": config.tinyfix_mode,
        "tinyfix_enabled_explicit": bool(config.tinyfix_enabled_explicit),
    }


def config_from_settings(data: dict) -> AppConfig:
    """Build an AppConfig from a settings dict.

    Unknown keys are ignored; any invalid value falls back to the
    default for that key (a single bad value never breaks startup).
    """
    data = data if isinstance(data, dict) else {}
    kwargs: dict = {}
    for key in _SETTINGS_KEYS:
        if key not in data:
            continue
        value = data[key]
        try:
            if key == "output_root":
                kwargs[key] = Path(str(value))
            elif key in ("semantic_model_dir", "sherpa_model_dir"):
                kwargs[key] = Path(str(value)) if value else None
            elif key == "sherpa_expected_speakers":
                kwargs[key] = int(value)
            elif key == "sherpa_clustering_threshold":
                kwargs[key] = float(value)
            elif key in ("hf_token_configured", "near_live_enabled",
                         "ollama_enabled", "auto_process_recordings",
                         "auto_detect_teams", "outlook_link_enabled",
                         "transcript_polish_enabled",
                         "enterprise_cleanup_enabled",
                         "whisper_model_explicit", "preroll_enabled",
                         "tinyfix_enabled", "tinyfix_enabled_explicit"):
                kwargs[key] = bool(value)
            elif key == "enterprise_cleanup_timeout_seconds":
                kwargs[key] = float(value)
            elif key == "audio_retention_hours":
                kwargs[key] = None if value is None else float(value)
            elif key == "disk_warning_threshold_bytes":
                kwargs[key] = int(value)
            elif key == "preroll_seconds":
                kwargs[key] = float(value)
            elif key == "nearlive_chunk_seconds":
                kwargs[key] = float(value)
            else:
                kwargs[key] = value
            # Validate one key at a time so a bad value falls back.
            AppConfig(**kwargs)
        except (ValueError, TypeError, OSError):
            kwargs.pop(key, None)
    if "whisper_model_explicit" not in data:
        # Pre-flag settings file (before Nov 20 2026, Item 8):
        # infer explicitness from the stored model. A stored model
        # that was a past default ("small" under 1.0.2) or is the
        # current default — or is missing/invalid, so nothing was
        # stored — cannot have been an explicit pick: the current
        # default wins (effective_whisper_model). Anything else
        # ("tiny"/"medium") could only have been picked, so it is
        # kept. Going forward the flag is always written, so this
        # runs only for pre-flag files. Honest caveat: an explicit
        # "small" pick made under 1.0.2/1.0.3 is indistinguishable
        # from the old default and resets to base once on upgrade;
        # re-picking small in Settings sets the flag and persists.
        stored = kwargs.get("whisper_model")
        kwargs["whisper_model_explicit"] = (
            stored is not None
            and stored != WHISPER_DEFAULT_MODEL
            and stored not in PAST_WHISPER_DEFAULTS)
    if "tinyfix_enabled_explicit" not in data:
        # Pre-flag settings file (before Nov 22 2026): a stored
        # tinyfix_enabled=True cannot be told apart from the old
        # ON default — every settings save wrote the checkbox
        # state, checked by default — so it is NOT an explicit
        # opt-in and the current default (OFF) wins. A stored
        # False was never a default either way and stays False.
        # Going forward the flag is always written, so this runs
        # only for pre-flag files. Honest caveat: an explicit
        # opt-in saved under 1.0.5 is indistinguishable from the
        # old default and resets to OFF once on upgrade;
        # re-enabling in Settings sets the flag and persists.
        kwargs["tinyfix_enabled_explicit"] = False
        if kwargs.get("tinyfix_enabled") is True:
            kwargs["tinyfix_enabled"] = False
    return AppConfig(**kwargs)


def save_settings(config: AppConfig,
                  path: str | Path | None = None) -> Path:
    """Persist ``config`` to the settings file (allowlisted keys only).

    The file never contains a token: settings_to_dict() only writes
    the boolean ``hf_token_configured`` flag. Returns the path used.
    """
    import json

    target = Path(path) if path is not None else default_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(settings_to_dict(config), indent=2),
                      encoding="utf-8")
    return target


def load_settings(path: str | Path | None = None) -> AppConfig:
    """Load settings from disk, or defaults if none / unreadable.

    A missing file, corrupt JSON, or wrong-typed payload all fall
    back to defaults — startup must never fail on settings.
    """
    import json

    target = Path(path) if path is not None else default_settings_path()
    if not target.exists():
        return AppConfig()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return AppConfig()
    if not isinstance(data, dict):
        return AppConfig()
    return config_from_settings(data)
