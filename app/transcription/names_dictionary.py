"""Names dictionary — user-editable names and mishearing corrections.

The dictionary does two jobs, both reversible because the verbatim
transcript is never modified (see :mod:`app.transcription.cleanup`):

1. **Decode-time biasing.** :func:`build_initial_prompt` turns the
   names (and the correct spellings from the corrections map) into a
   faster-whisper ``initial_prompt`` string, passed at decode time by
   :meth:`TranscriptionEngine.transcribe_file
   <app.transcription.engine.TranscriptionEngine.transcribe_file>`.
   Whisper then hears "MeetTrace" instead of "Meat Trace" more often.
2. **Post-process correction.** :func:`apply_corrections` maps known
   mishearings to their correct spellings with whole-word matching
   only — "Marcus" is never rewritten inside another word.

Storage: plain JSON in ``~/.offair/names_dictionary.json``::

    {"version": 1,
     "names": ["John", "Priya", "Marcus"],
     "corrections": {"Meat Trace": "MeetTrace"}}

The seed correction is real: Whisper tiny mishears the product name
"MeetTrace" as "Meat Trace" in the project's demo fixture. Renaming
a speaker (see :func:`app.speakers.session.rename_identity`) adds
their name to the dictionary automatically.

A missing file loads the seed defaults; a corrupt file also falls
back to them (logged, never raised — a bad dictionary must not
break transcription). The file is only written by
:func:`save_names_dictionary` / :func:`add_name` / the Settings
dialog, so a user who deletes the seed entry is never re-seeded.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("offair.transcription.names_dictionary")

NAMES_DICTIONARY_FILE = "names_dictionary.json"

#: Seed corrections — real mishearings observed in this project's data.
#: Whisper tiny hears the product name "MeetTrace" as "Meat Trace".
SEED_CORRECTIONS: dict[str, str] = {"Meat Trace": "MeetTrace"}


def default_dictionary_path() -> Path:
    """Where the dictionary lives: ``~/.offair/names_dictionary.json``."""
    return Path.home() / ".offair" / NAMES_DICTIONARY_FILE


@dataclass
class NamesDictionary:
    """User-editable names + mishearing → correct-spelling map."""

    names: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)

    def prompt_terms(self) -> list[str]:
        """Terms to bias Whisper with: names + correct spellings."""
        terms = list(self.names)
        for correct in self.corrections.values():
            if correct not in terms:
                terms.append(correct)
        return terms


def _seeded() -> NamesDictionary:
    return NamesDictionary(names=[], corrections=dict(SEED_CORRECTIONS))


def load_names_dictionary(path: str | Path | None = None) -> NamesDictionary:
    """Load the dictionary; missing/corrupt file -> seed defaults."""
    target = Path(path) if path is not None else default_dictionary_path()
    if not target.exists():
        return _seeded()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning("Cannot read names dictionary %s (%s) — "
                       "using defaults", target, exc)
        return _seeded()
    if not isinstance(payload, dict):
        logger.warning("Names dictionary %s is not an object — "
                       "using defaults", target)
        return _seeded()
    names = payload.get("names", [])
    corrections = payload.get("corrections", {})
    if not isinstance(names, list) or not isinstance(corrections, dict):
        logger.warning("Names dictionary %s has wrong-typed fields — "
                       "using defaults", target)
        return _seeded()
    clean_names = []
    for name in names:
        name = str(name).strip()
        if name and name not in clean_names:
            clean_names.append(name)
    clean_corrections = {}
    for wrong, right in corrections.items():
        wrong, right = str(wrong).strip(), str(right).strip()
        if wrong and right:
            clean_corrections[wrong] = right
    return NamesDictionary(names=clean_names,
                           corrections=clean_corrections)


def save_names_dictionary(dictionary: NamesDictionary,
                          path: str | Path | None = None) -> Path:
    """Persist the dictionary as JSON. Returns the path used."""
    target = Path(path) if path is not None else default_dictionary_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "names": list(dictionary.names),
               "corrections": dict(dictionary.corrections)}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def add_name(name: str, path: str | Path | None = None) -> bool:
    """Add ``name`` to the dictionary (case-insensitive dedup).

    Returns True when the name was added, False when it was already
    there (or blank). Used by the speaker-rename path so a renamed
    speaker is biased for in future transcriptions.
    """
    name = str(name).strip()
    if not name:
        return False
    dictionary = load_names_dictionary(path)
    if any(existing.lower() == name.lower()
           for existing in dictionary.names):
        return False
    dictionary.names.append(name)
    save_names_dictionary(dictionary, path)
    logger.info("Added %r to the names dictionary", name)
    return True


def build_initial_prompt(
        dictionary: NamesDictionary | None) -> str | None:
    """The faster-whisper ``initial_prompt`` for ``dictionary``.

    None when the dictionary has nothing to bias with (no names and
    no corrections) — the caller then passes no prompt at all, so
    transcription behaves exactly as before.
    """
    if dictionary is None:
        return None
    terms = dictionary.prompt_terms()
    if not terms:
        return None
    return ("In this meeting, these names and terms may be mentioned — "
            "spell them exactly as shown: " + ", ".join(terms) + ".")


def _correction_pattern(wrong: str) -> "re.Pattern[str]":
    """Whole-word, case-insensitive pattern for a mishearing.

    The key's internal whitespace becomes ``\\s+`` so "Meat Trace"
    also matches a line-broken "Meat  Trace". ``\\b`` on both ends
    is what makes the match whole-word: "Marcus" never matches
    inside "Marcuson" or "Marcusville".
    """
    parts = [re.escape(part) for part in str(wrong).split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b",
                      re.IGNORECASE)


def apply_corrections(text: str,
                      corrections: dict[str, str] | None) -> tuple[str, int]:
    """Replace known mishearings with their correct spellings.

    Whole-word, case-insensitive matching; the replacement always
    uses the dictionary's canonical spelling. Returns
    ``(new_text, replacements_made)``.
    """
    if not corrections or not text:
        return text, 0
    total = 0
    for wrong, right in corrections.items():
        pattern = _correction_pattern(wrong)
        text, made = pattern.subn(right, text)
        total += made
    return text, total


def parse_dictionary_text(text: str) -> NamesDictionary:
    """Parse the Settings editor format into a dictionary.

    One entry per line; ``#`` lines are comments and blank lines are
    skipped::

        John
        Priya
        Meat Trace -> MeetTrace

    A line with ``->`` is a correction (mishearing -> correct
    spelling); a bare line is a name.
    """
    names: list[str] = []
    corrections: dict[str, str] = {}
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "->" in line:
            wrong, _, right = line.partition("->")
            wrong, right = wrong.strip(), right.strip()
            if wrong and right:
                corrections[wrong] = right
        elif line not in names:
            names.append(line)
    return NamesDictionary(names=names, corrections=corrections)


def format_dictionary_text(dictionary: NamesDictionary) -> str:
    """The Settings editor format for ``dictionary`` (round-trips
    through :func:`parse_dictionary_text`)."""
    lines = list(dictionary.names)
    for wrong, right in dictionary.corrections.items():
        lines.append(f"{wrong} -> {right}")
    return "\n".join(lines) + ("\n" if lines else "")
