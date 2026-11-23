"""Transcript cleanup — rule-based, deterministic, no LLM.

The verbatim transcript is NEVER modified. Cleanup reads
``transcript.json`` and writes derived artifacts only::

    transcript-clean.json   cleaned segments (same shape as
                            transcript.json, plus a "cleaning" block
                            with the counts and the source file)
    transcript-clean.txt    the cleaned segments as [MM:SS] lines,
                            speaker display names applied like the
                            verbatim export

The original ``transcript.json`` / ``transcript.txt`` / ``.srt``
stay byte-identical after any clean pass — a test proves it
(see tests/test_transcript_cleanup.py). Both artifacts are
reversible: delete them and re-run :func:`clean_session`.

Rules, in order (the names dictionary runs first — see
:mod:`app.transcription.names_dictionary`):

1. **Dictionary corrections** — known mishearings -> correct
   spellings, whole-word only.
2. **Filler removal** — um, uh, hmm, er, erm, mmm, ah. A filler
   at the very start of a segment capitalises the new first word
   ("Um, let's go" -> "Let's go").
3. **Punctuation / spacing tidy** — collapse whitespace, drop the
   space before ``.,!?;:)``, collapse doubled punctuation
   ("word.." -> "word."), ensure a space after ``,;:!?`` when a
   letter follows.

There is deliberately NO repeat-removal rule. An earlier version
had false-start repair ("I think— I think we should" -> "I think
we should") and repeated-word dedup ("the the budget" ->
"the budget"); both were REMOVED Nov 22 2026 because on real
disfluent speech every removal they made was wrong: against the
published transcript of a 9-minute interview (see
docs/notes/cleanup-repeat-rule-2026-11-22.md) they deleted 6
words the speaker actually said, including a deliberate
"technique, technique, technique" chant. No text-only signal
distinguishes a stutter repeat from intentional emphasis.
Repeats are kept verbatim; the ``false_starts_repaired`` /
``repeated_words_removed`` counts in the cleaning block stay
(forever 0) so old artifacts keep their shape.

Segments that clean down to nothing (a segment that was only a
filler, e.g. "Um.") are dropped from the cleaned artifact and
counted as ``dropped_segments``.

The tinyfix layer lives here too (Nov 21 2026): after the rules
and the names dictionary, a small ONNX model fixes real words in
the wrong place ("form" -> "from", "their" -> "there") that no
rule can see. OFF by default since Nov 22 2026 (opt-in in
Settings) — on disfluent real speech it changed correct words
(see docs/notes/cleanup-repeat-rule-2026-11-22.md). English
only; the names dictionary wins — every dictionary word is on
tinyfix's ignore list. A failed model download degrades to
deterministic-only cleanup and is recorded, never a failure. See
:mod:`app.transcription.tinyfix`.

The LLM polish layer lives here too (:func:`polish_session`):
optional, OFF by default, local Ollama only. It rewrites the
*cleaned* transcript into ``transcript-polished.md``, a separate
artifact whose header states plainly that it is an AI rewrite of
the verbatim transcript — never a replacement for it. If Ollama
is not running it raises
:class:`app.exports.local_llm.OllamaNotAvailableError` with a clear
message and leaves the rule-cleaned version in place.

A third variant, the enterprise-LLM cleanup, is REGISTERED here
only (:data:`TRANSCRIPT_VARIANTS` / :data:`TRANSCRIPT_VARIANT_FILES`
/ :func:`transcript_variant_path`); the runner itself lives in
:mod:`app.transcription.enterprise_cleanup` and writes
``transcript-enterprise-clean.txt``. Verbatim, deterministic-clean,
and enterprise-clean are three distinct artifacts — never merged.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.diarization.names import load_speaker_names
from app.transcription.engine import Segment, TRANSCRIPT_JSON
from app.transcription.exports import export_txt, load_transcript_json
from app.transcription.names_dictionary import (
    NamesDictionary,
    apply_corrections,
    load_names_dictionary,
)

logger = logging.getLogger("offair.transcription.cleanup")

CLEAN_JSON = "transcript-clean.json"
CLEAN_TXT = "transcript-clean.txt"
POLISHED_MD = "transcript-polished.md"

#: Third transcript variant: the enterprise-LLM cleanup (see
#: :mod:`app.transcription.enterprise_cleanup`). It is a TEXT
#: artifact only — ``transcript-enterprise-clean.txt`` — because
#: the headless LLM returns unstructured text and its segment
#: timings cannot be trusted, so no
#: ``transcript-enterprise-clean.json`` exists. The verbatim,
#: deterministic-clean, and enterprise-clean artifacts are three
#: distinct files; the transcript-view toggle offers all three
#: explicitly, by the keys in :data:`TRANSCRIPT_VARIANTS`.
ENTERPRISE_CLEAN_TXT = "transcript-enterprise-clean.txt"
TRANSCRIPT_VARIANTS: tuple[str, ...] = ("original", "cleaned",
                                        "enterprise")
TRANSCRIPT_VARIANT_LABELS: dict[str, str] = {
    "original": "Original (verbatim)",
    "cleaned": "Cleaned",
    "enterprise": "Enterprise-cleaned",
}
#: Readable text file per variant (the file the toggle shows).
#: "original" is the verbatim ``transcript.txt``; "cleaned" is
#: the rule-cleaned ``transcript-clean.txt``.
TRANSCRIPT_VARIANT_FILES: dict[str, str] = {
    "original": "transcript.txt",
    "cleaned": CLEAN_TXT,
    "enterprise": ENTERPRISE_CLEAN_TXT,
}


def transcript_variant_path(session_dir: str | Path,
                            variant: str) -> Path:
    """Text-file path for one transcript variant.

    Raises ValueError for an unknown variant. The file may not
    exist yet (a variant is only written when it is produced) —
    callers check existence and say so, never invent content.
    """
    try:
        filename = TRANSCRIPT_VARIANT_FILES[variant]
    except KeyError as exc:
        raise ValueError(
            f"unknown transcript variant {variant!r}; expected one of "
            f"{list(TRANSCRIPT_VARIANTS)}") from exc
    return Path(session_dir) / filename

#: Fillers Whisper leaves in spontaneous speech. Whole-word only.
FILLERS: frozenset[str] = frozenset(
    {"um", "uh", "hmm", "er", "erm", "mmm", "ah"})

_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(sorted(FILLERS)) + r")\b[,.]?",
    re.IGNORECASE)
# NOTE (Nov 22 2026): the false-start and repeated-word patterns
# lived here and were removed — see the module docstring. There
# is no text-only signal that distinguishes a stutter repeat
# from intentional emphasis, and on real disfluent speech every
# removal they made was wrong
# (docs/notes/cleanup-repeat-rule-2026-11-22.md). Do not re-add
# a repeat-removal rule without real-speech evidence that it
# helps more than it harms.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:%)])")
_SPACE_AFTER_PUNCT_RE = re.compile(r"([,;:!?])(?=[A-Za-z])")
_DOUBLED_PUNCT_RE = re.compile(r"([.,!?])\1+")

_POLISH_PROMPT = (
    "You are cleaning up a meeting transcript. Rewrite the transcript "
    "below to fix grammar, punctuation and obvious transcription "
    "errors, and remove filler words and false starts. Keep every "
    "speaker label and timestamp exactly as written. Do not add "
    "facts, names, dates or commitments that are not in the "
    "transcript, and do not remove any spoken content except fillers "
    "and false starts. Return only the rewritten transcript, in the "
    "same [MM:SS] Speaker: text format.\n\nTranscript:\n{transcript}"
)

POLISHED_HEADER = (
    "# Transcript (AI-polished)\n\n"
    "> This is an AI rewrite of the verbatim transcript, produced by "
    "a local LLM (Ollama on this machine). It may contain errors — "
    "the verbatim transcript in transcript.json / transcript.txt is "
    "the record. This file is a readability aid only, never a "
    "replacement for the verbatim transcript.\n\n"
)


@dataclass
class CleanStats:
    """What one clean pass did, per rule."""

    fillers: int = 0
    #: Removed Nov 22 2026 (see the module docstring): the
    #: false-start and repeat rules no longer run, so these stay
    #: 0. The fields remain so the "cleaning" block keeps its
    #: shape for old artifacts and readers.
    false_starts: int = 0
    repeats: int = 0
    corrections: int = 0
    dropped_segments: int = 0
    tinyfix: int = 0
    #: Why the tinyfix stage did not run: "disabled" when turned
    #: off, "non-English transcript (language=…)" for a known
    #: non-English transcript, "unavailable: …" when the model
    #: could not be loaded. "" when the stage ran.
    tinyfix_skipped: str = ""

    def as_dict(self) -> dict:
        return {"fillers_removed": self.fillers,
                "false_starts_repaired": self.false_starts,
                "repeated_words_removed": self.repeats,
                "name_corrections_applied": self.corrections,
                "dropped_segments": self.dropped_segments,
                "tinyfix_corrections_applied": self.tinyfix,
                "tinyfix_skipped": self.tinyfix_skipped}

    def __add__(self, other: "CleanStats") -> "CleanStats":
        return CleanStats(
            fillers=self.fillers + other.fillers,
            false_starts=self.false_starts + other.false_starts,
            repeats=self.repeats + other.repeats,
            corrections=self.corrections + other.corrections,
            dropped_segments=self.dropped_segments + other.dropped_segments,
            tinyfix=self.tinyfix + other.tinyfix,
            tinyfix_skipped=self.tinyfix_skipped or other.tinyfix_skipped)


def _tidy(text: str) -> str:
    """Punctuation / spacing tidy (rule 3)."""
    text = _DOUBLED_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_PUNCT_RE.sub(r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str,
               dictionary: NamesDictionary | None = None) -> tuple[str, CleanStats]:
    """Clean one segment's text. Returns ``(cleaned, stats)``.

    Pure function — no I/O. The input is never mutated (strings
    are immutable; the return is a new string).
    """
    stats = CleanStats()
    if not text or not text.strip():
        return "", stats
    out = text.strip()

    # 1. Dictionary corrections first (whole-word, canonical spelling).
    if dictionary is not None:
        out, made = apply_corrections(out, dictionary.corrections)
        stats.corrections += made

    # No repeat removal (Nov 22 2026): false-start repair and
    # repeated-word dedup were removed — repeats are kept
    # verbatim. See the module docstring.

    # 2. Filler removal. A filler at the very start capitalises the
    # new first word ("Um, let's go" -> "Let's go").
    leading_filler = bool(_FILLER_RE.match(out.lstrip()))
    out, made = _FILLER_RE.subn("", out)
    stats.fillers += made
    out = _tidy(out)
    if leading_filler and out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out, stats


def clean_segment(segment: Segment,
                  dictionary: NamesDictionary | None = None
                  ) -> tuple[Segment | None, CleanStats]:
    """Clean one segment. Returns ``(segment, stats)`` — or
    ``(None, stats)`` when the segment cleaned down to nothing."""
    text, stats = clean_text(segment.text, dictionary)
    if not text:
        stats.dropped_segments = 1
        return None, stats
    cleaned = Segment(start=segment.start, end=segment.end, text=text,
                      speaker=segment.speaker)
    return cleaned, stats


def _transcript_language(payload: dict) -> str:
    """Normalised language tag from a transcript payload ("" when
    absent — older transcripts carry no language)."""
    language = payload.get("language")
    if not language:
        return ""
    return str(language).lower().replace("_", "-").split("-")[0]


def _run_tinyfix_stage(payload: dict, cleaned_segments: list[Segment],
                       dictionary: NamesDictionary, mode: str,
                       checker, status_callback,
                       total: CleanStats) -> str:
    """Runs the tinyfix stage in place over ``cleaned_segments``.

    Returns "" when the stage ran, or the reason it was skipped.
    Never raises: a missing model or failed download degrades to
    deterministic-only cleanup."""
    language = _transcript_language(payload)
    if language and language != "en":
        reason = f"non-English transcript (language={language})"
        logger.info("tinyfix skipped: %s", reason)
        return reason
    try:
        if checker is None:
            from app.transcription.tinyfix import (
                TinyfixUnavailableError, build_checker)

            try:
                checker = build_checker(
                    mode=mode, status_callback=status_callback)
            except TinyfixUnavailableError as exc:
                reason = f"unavailable: {exc}"
                logger.warning("tinyfix skipped: %s", reason)
                return reason
        # Names win: every dictionary word is ignored, so a known
        # name or product term is never "fixed".
        ignore_words = list(dictionary.names)
        for mishearing, correct in dictionary.corrections.items():
            ignore_words.extend((mishearing, correct))
        checker.ignore_words(ignore_words)
        dicts = [segment.to_dict() for segment in cleaned_segments]
        fixed, corrections = checker.fix_segments(dicts)
        total.tinyfix = len(corrections)
        for segment, fixed_dict in zip(cleaned_segments, fixed):
            segment.text = fixed_dict["text"]
    except Exception as exc:  # noqa: BLE001 — cleanup must not fail
        reason = f"unavailable: {exc}"
        logger.warning("tinyfix skipped after error: %s", reason,
                       exc_info=True)
        return reason
    return ""


def clean_session(session_dir: str | Path,
                  dictionary: NamesDictionary | None = None,
                  speaker_names: dict[str, str] | None = None,
                  *, tinyfix_enabled: bool | None = None,
                  tinyfix_mode: str = "careful",
                  tinyfix_checker=None,
                  tinyfix_status_callback=None,
                  ) -> dict:
    """Clean a session's verbatim transcript into derived artifacts.

    Reads ``transcript.json`` (and ``speakers.json`` for display
    names, unless ``speaker_names`` is given), writes
    ``transcript-clean.json`` + ``transcript-clean.txt``. The
    verbatim files are only read, never written — a clean pass
    cannot change them. Returns the stats dict (the same shape as
    :meth:`CleanStats.as_dict`, plus ``source_segments`` /
    ``cleaned_segments``). Raises FileNotFoundError when the
    session has no transcript yet.

    The tinyfix stage runs after the deterministic rules and the
    names dictionary, unless ``tinyfix_enabled`` is False.
    ``None`` means the shipped default: OFF since Nov 22 2026
    (``TINYFIX_DEFAULT_ENABLED``). An injected
    ``tinyfix_checker`` (tests) skips the model download; a
    failed download degrades to deterministic-only cleanup and
    is recorded in ``tinyfix_skipped``, never raised. The stage
    is English only: it is skipped when the transcript's
    ``language`` is known non-English, and runs when the language
    is absent (older transcripts) or English.
    """
    session_dir = Path(session_dir)
    source = session_dir / TRANSCRIPT_JSON
    if not source.exists():
        raise FileNotFoundError(f"No {TRANSCRIPT_JSON} in {session_dir}")
    if dictionary is None:
        dictionary = load_names_dictionary()
    if speaker_names is None:
        try:
            speaker_names = load_speaker_names(session_dir)
        except ValueError:
            speaker_names = {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    segments = [Segment.from_dict(d) for d in payload["segments"]]
    total = CleanStats()
    cleaned_segments: list[Segment] = []
    for segment in segments:
        cleaned, stats = clean_segment(segment, dictionary)
        total += stats
        if cleaned is not None:
            cleaned_segments.append(cleaned)
    if tinyfix_enabled is None:
        from app.config import TINYFIX_DEFAULT_ENABLED

        tinyfix_enabled = TINYFIX_DEFAULT_ENABLED
    if tinyfix_enabled:
        total.tinyfix_skipped = _run_tinyfix_stage(
            payload, cleaned_segments, dictionary, tinyfix_mode,
            tinyfix_checker, tinyfix_status_callback, total)
    else:
        total.tinyfix_skipped = "disabled"
    clean_payload = dict(payload)
    clean_payload["cleaned_from"] = TRANSCRIPT_JSON
    clean_payload["cleaning"] = total.as_dict()
    clean_payload["source_segments"] = len(segments)
    clean_payload["cleaned_segments"] = len(cleaned_segments)
    clean_payload["segments"] = [s.to_dict() for s in cleaned_segments]
    (session_dir / CLEAN_JSON).write_text(
        json.dumps(clean_payload, indent=2), encoding="utf-8")
    export_txt(cleaned_segments, session_dir / CLEAN_TXT,
               speaker_names=speaker_names)
    logger.info("Cleaned %s: %d -> %d segments "
                "(fillers %d, false starts %d, repeats %d, "
                "corrections %d, tinyfix %d%s)", session_dir.name,
                len(segments), len(cleaned_segments), total.fillers,
                total.false_starts, total.repeats, total.corrections,
                total.tinyfix,
                f", tinyfix skipped: {total.tinyfix_skipped}"
                if total.tinyfix_skipped else "")
    result = total.as_dict()
    result["source_segments"] = len(segments)
    result["cleaned_segments"] = len(cleaned_segments)
    return result


def load_clean_transcript_json(path: str | Path) -> list[Segment]:
    """Load segments from a transcript-clean.json (same shape as
    the verbatim loader)."""
    return load_transcript_json(path)


def polish_session(session_dir: str | Path, *,
                   config=None, ollama_client=None) -> Path:
    """AI-polish a session's cleaned transcript (local Ollama only).

    Reads ``transcript-clean.txt`` when it exists (else the verbatim
    ``transcript.txt``), asks the local Ollama model to rewrite it
    for readability, and writes ``transcript-polished.md`` with a
    header stating it is an AI rewrite. The verbatim and
    rule-cleaned files are untouched. ``config`` is an
    :class:`app.config.AppConfig` (its Ollama URL/model are used
    when no client is given); ``ollama_client`` is injectable for
    tests. Raises
    :class:`app.exports.local_llm.OllamaNotAvailableError` when
    Ollama is not running — the caller turns that into a clear
    message; the rule-cleaned version stays in place.
    """
    from app.exports.local_llm import (
        OllamaClient,
        OllamaNotAvailableError,
    )

    session_dir = Path(session_dir)
    clean_txt = session_dir / CLEAN_TXT
    source = clean_txt if clean_txt.exists() else session_dir / "transcript.txt"
    if not source.exists():
        raise FileNotFoundError(f"No transcript to polish in {session_dir}")
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Cannot polish an empty transcript")
    if ollama_client is None:
        ollama_client = OllamaClient(
            base_url=(config.ollama_base_url if config is not None
                      else "http://localhost:11434"),
            model=(config.ollama_model if config is not None
                   else "qwen3.5:9b"))
    if not ollama_client.is_available():
        raise OllamaNotAvailableError(
            "Ollama is not running on localhost:11434. Start it "
            "(install from https://ollama.com, then `ollama serve`, "
            "and `ollama pull qwen3.5:9b`) to AI-polish this "
            "transcript — the rule-cleaned version is still in "
            "transcript-clean.txt.")
    polished = ollama_client.summarize(
        text, prompt_template=_POLISH_PROMPT)
    target = session_dir / POLISHED_MD
    target.write_text(POLISHED_HEADER + polished.strip() + "\n",
                      encoding="utf-8")
    logger.info("Wrote AI-polished transcript to %s", target)
    return target
