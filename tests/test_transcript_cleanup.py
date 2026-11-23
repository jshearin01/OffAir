"""Transcript cleanup + names dictionary tests (Nov 10 2026).

No Qt, no model download (except the one fixture test, which uses
the cached Whisper tiny model — the same fixture the Phase 2
manual tests use). The hard rule under test throughout: the
verbatim transcript is NEVER modified — cleanup writes derived
artifacts only, and the original files stay byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from app.exports.local_llm import OllamaNotAvailableError
from app.transcription.cleanup import (
    CLEAN_JSON,
    CLEAN_TXT,
    POLISHED_MD,
    clean_session,
    clean_text,
    polish_session,
)
from app.transcription.engine import (
    TRANSCRIPT_JSON,
    TRANSCRIPT_TXT,
    Segment,
    TranscriptionEngine,
)
from app.transcription.names_dictionary import (
    SEED_CORRECTIONS,
    NamesDictionary,
    add_name,
    apply_corrections,
    build_initial_prompt,
    format_dictionary_text,
    load_names_dictionary,
    parse_dictionary_text,
    save_names_dictionary,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_verbatim_session(session: Path, texts: list[str],
                            speakers: list[str | None] | None = None
                            ) -> None:
    """A transcript.json/.txt pair like transcribe_session writes."""
    session.mkdir(parents=True, exist_ok=True)
    segments = []
    for i, text in enumerate(texts):
        seg = {"start": round(i * 3.0, 3), "end": round(i * 3.0 + 2.5, 3),
               "text": text}
        if speakers and speakers[i]:
            seg["speaker"] = speakers[i]
        segments.append(seg)
    payload = {"model": "tiny", "device": "cpu", "compute_type": "int8",
               "language": "en", "audio_file": "combined.wav",
               "audio_duration_seconds": len(texts) * 3.0,
               "wall_seconds": 1.0, "real_time_factor": 0.1,
               "segments": segments}
    (session / TRANSCRIPT_JSON).write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (session / TRANSCRIPT_TXT).write_text(
        "".join(f"[{i * 3 // 60:02d}:{i * 3 % 60:02d}] {t}\n"
                for i, t in enumerate(texts)), encoding="utf-8")


# ------------------------------------------------------------------ rules --
def test_filler_removal():
    text, stats = clean_text("Um, let's start the review.")
    assert text == "Let's start the review."
    assert stats.fillers == 1
    text, stats = clean_text("Well uh we should uh check the budget.")
    assert text == "Well we should check the budget."
    assert stats.fillers == 2
    text, stats = clean_text("Hmm, er, I don't know.")
    assert text == "I don't know."
    assert stats.fillers == 2


def test_filler_inside_word_not_removed():
    # "umbrella" contains "um" but is not a filler; "hmm" in "hmmph" no.
    text, stats = clean_text("Bring the umbrella tomorrow.")
    assert text == "Bring the umbrella tomorrow."
    assert stats.fillers == 0


def test_false_start_dash_repetition_is_kept():
    # Nov 22 2026 (test 2, disfluent real speech): the old
    # false-start rule deleted repeats the published transcript
    # keeps ("feeding — feeding that —"), so the rule was
    # removed outright — repeats are kept verbatim. See
    # docs/notes/cleanup-repeat-rule-2026-11-22.md.
    text, stats = clean_text("I think— I think we should ship it.")
    assert text == "I think— I think we should ship it."
    assert stats.false_starts == 0
    text, stats = clean_text("So the— the budget is twelve thousand.")
    assert text == "So the— the budget is twelve thousand."
    assert stats.false_starts == 0


def test_false_start_comma_repetition_is_kept():
    # As above: the comma false-start rule is removed.
    text, stats = clean_text("I think, I think we should ship it.")
    assert text == "I think, I think we should ship it."
    assert stats.false_starts == 0


def test_false_start_bare_repetition_is_kept():
    # As above: the bare-repetition rule is removed.
    text, stats = clean_text("I think I think we should ship it.")
    assert text == "I think I think we should ship it."
    assert stats.false_starts == 0


def test_comma_repetition_is_emphasis_not_false_start():
    # One word repeated with a comma is emphasis, not a false start.
    text, stats = clean_text("That was very, very good.")
    assert text == "That was very, very good."
    assert stats.false_starts == 0


def test_repeated_words_are_kept():
    # Nov 22 2026 (test 2): repeat dedup deleted words the
    # published transcript keeps, so the rule was removed —
    # repeats stay verbatim. See
    # docs/notes/cleanup-repeat-rule-2026-11-22.md.
    text, stats = clean_text("Please check the the budget.")
    assert text == "Please check the the budget."
    assert stats.repeats == 0
    text, stats = clean_text("The the the plan is ready.")
    assert text == "The the the plan is ready."
    assert stats.repeats == 0


def test_test2_repeat_cases_are_kept_verbatim():
    """Regression: the exact segments whose repeats the old
    rules deleted in the Nov 22 2026 test-2 measurement (Lex
    Fridman / Tom Brands interview, official transcript) must
    now clean to themselves. The ground truth keeps every
    repeat below, including the deliberate triple chant."""
    cases = [
        # Dash false-start repair deleted "feeding —".
        "The ingredient of lost feeding — feeding that — "
        "the drive of heart training?",
        # Dedup deleted the second "maybe".
        "So if you look back, you probably spent tens of "
        "thousands of hours on the mat spilled sweat, blood, "
        "even tears, maybe maybe a few times.",
        # Dedup deleted the second "that".
        "And massage could be that. Stretching could be that "
        "thing, things like that that are more fluffy.",
        # Dedup deleted one "technique" from the chant.
        "And you said repetition. Do you mean technique? I'm "
        "talking repetition technique technique, technique "
        "drill, drill, drill, hit, hit, hit, drive, finish, "
        "hit, hit, hit, drive, finish.",
        # Bare-repetition repair deleted "in the".
        "So go back to your point or your question, you know, "
        "do you believe in the in the 10,000 reps?",
    ]
    for text_in in cases:
        text, stats = clean_text(text_in)
        assert text == text_in
        assert stats.false_starts == 0
        assert stats.repeats == 0


def test_punctuation_spacing_tidy():
    text, stats = clean_text("Hello  .  World ,  how are you ?")
    assert text == "Hello. World, how are you?"
    text, stats = clean_text("Wait.. what happened?")
    assert text == "Wait. what happened?"
    text, stats = clean_text("Yes,ok let's go")
    assert text == "Yes, ok let's go"


def test_clean_text_counts_all_rules():
    dictionary = NamesDictionary(
        names=["MeetTrace"],
        corrections={"Meat Trace": "MeetTrace"})
    text, stats = clean_text(
        "Um, the the Meat Trace demo — the Meat Trace demo was, "
        "uh, I think— I think it was good.",
        dictionary)
    # Repeats kept verbatim (Nov 22 2026 rule removal); only
    # the dictionary + fillers change anything.
    assert text == ("The the MeetTrace demo — the MeetTrace demo "
                    "was, I think— I think it was good.")
    assert stats.fillers == 2
    assert stats.false_starts == 0
    assert stats.repeats == 0
    assert stats.corrections == 2


def test_segment_cleaned_to_nothing_is_dropped(tmp_path):
    from app.transcription.cleanup import clean_segment
    seg = Segment(0.0, 1.0, "Um.")
    cleaned, stats = clean_segment(seg)
    assert cleaned is None
    assert stats.dropped_segments == 1


# ------------------------------------------------------- names dictionary --
def test_dictionary_seed_has_meat_trace(tmp_path):
    dictionary = load_names_dictionary(tmp_path / "missing.json")
    assert dictionary.corrections == SEED_CORRECTIONS
    assert dictionary.corrections["Meat Trace"] == "MeetTrace"


def test_seed_correction_applied():
    text, made = apply_corrections(
        "Welcome to the Meat Trace review.", SEED_CORRECTIONS)
    assert text == "Welcome to the MeetTrace review."
    assert made == 1


def test_whole_word_replacement_only():
    corrections = {"Marcus": "Markus"}
    # Inside another word: untouched.
    text, made = apply_corrections("Marcuson called.", corrections)
    assert text == "Marcuson called."
    assert made == 0
    text, made = apply_corrections("Marcusville is far.", corrections)
    assert text == "Marcusville is far."
    assert made == 0
    # Whole word (and possessive): replaced.
    text, made = apply_corrections(
        "Marcus, please confirm. Marcus's list is ready.", corrections)
    assert text == "Markus, please confirm. Markus's list is ready."
    assert made == 2


def test_correction_is_case_insensitive_but_spelling_is_canonical():
    text, made = apply_corrections(
        "the meat trace demo and MEAT TRACE slides", SEED_CORRECTIONS)
    assert text == "the MeetTrace demo and MeetTrace slides"
    assert made == 2


def test_multiword_correction_tolerates_spacing():
    text, made = apply_corrections(
        "The Meat  Trace demo.", SEED_CORRECTIONS)
    assert text == "The MeetTrace demo."
    assert made == 1


def test_build_initial_prompt():
    dictionary = NamesDictionary(
        names=["John", "Priya"],
        corrections={"Meat Trace": "MeetTrace"})
    prompt = build_initial_prompt(dictionary)
    assert prompt is not None
    assert "John" in prompt and "Priya" in prompt
    assert "MeetTrace" in prompt  # correct spelling, not the mishearing
    assert "Meat Trace" not in prompt
    assert build_initial_prompt(NamesDictionary()) is None
    assert build_initial_prompt(None) is None


def test_dictionary_round_trip_and_editor_format(tmp_path):
    path = tmp_path / "names_dictionary.json"
    dictionary = NamesDictionary(
        names=["John", "Priya"],
        corrections={"Meat Trace": "MeetTrace"})
    save_names_dictionary(dictionary, path)
    loaded = load_names_dictionary(path)
    assert loaded == dictionary
    text = format_dictionary_text(dictionary)
    assert parse_dictionary_text(text) == dictionary
    # Comments and blank lines are ignored by the editor format.
    parsed = parse_dictionary_text(
        "# names\nJohn\nPriya\n\nMeat Trace -> MeetTrace\n")
    assert parsed == dictionary


def test_add_name_dedups_case_insensitively(tmp_path):
    path = tmp_path / "names_dictionary.json"
    assert add_name("Priya", path) is True
    assert add_name("priya", path) is False
    assert add_name("  ", path) is False
    assert load_names_dictionary(path).names == ["Priya"]


def test_corrupt_dictionary_falls_back_to_seed(tmp_path):
    path = tmp_path / "names_dictionary.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_names_dictionary(path).corrections == SEED_CORRECTIONS


# ------------------------------------------------------------------ engine --
class _FakeRaw:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _FakeModel:
    """Records the kwargs transcribe() was called with."""

    def __init__(self, raws):
        self.raws = raws
        self.calls: list[dict] = []

    def transcribe(self, path, **kwargs):
        self.calls.append(dict(kwargs))
        return iter(self.raws), {"language": "en"}


def test_engine_passes_initial_prompt_to_model(tmp_path):
    import wave
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 2)
    model = _FakeModel([_FakeRaw(0.0, 1.0, "Hello.")])
    engine = TranscriptionEngine(model_size="tiny", model=model)
    engine.transcribe_file(wav, initial_prompt="Names: Priya, MeetTrace.")
    assert model.calls[0].get("initial_prompt") == "Names: Priya, MeetTrace."
    assert model.calls[0]["language"] == "en"


def test_engine_prompt_is_constructed_from_dictionary(tmp_path):
    """The dictionary's names + correct spellings become the prompt
    that reaches faster-whisper's transcribe()."""
    import wave
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 2)
    model = _FakeModel([_FakeRaw(0.0, 1.0, "Hello.")])
    engine = TranscriptionEngine(model_size="tiny", model=model)
    dictionary = NamesDictionary(
        names=["John", "Priya"],
        corrections={"Meat Trace": "MeetTrace"})
    engine.transcribe_file(
        wav, initial_prompt=build_initial_prompt(dictionary))
    prompt = model.calls[0]["initial_prompt"]
    assert "John" in prompt and "Priya" in prompt
    assert "MeetTrace" in prompt and "Meat Trace" not in prompt


def test_engine_without_prompt_passes_no_initial_prompt(tmp_path):
    """No dictionary -> no initial_prompt kwarg at all, so old fake
    models (and old behaviour) are unaffected."""
    import wave
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 2)

    class _OldFakeModel:
        def transcribe(self, path, language="en", beam_size=5):
            return iter([_FakeRaw(0.0, 1.0, "Hello.")]), {}

    engine = TranscriptionEngine(model_size="tiny", model=_OldFakeModel())
    result = engine.transcribe_file(wav)
    assert [s.text for s in result.segments] == ["Hello."]


# ------------------------------------------------------------- clean_session --
def test_clean_session_writes_derived_and_keeps_verbatim(tmp_path):
    session = tmp_path / "2026-11-10_10-00-00_teams"
    _write_verbatim_session(session, [
        "Um, good morning everyone.",
        "I think— I think the the Meat Trace demo was good.",
        "Uh, Marcus, please confirm the hardware list.",
    ], speakers=["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])
    (session / "speakers.json").write_text(json.dumps(
        {"version": 1,
         "names": {"SPEAKER_00": "John", "SPEAKER_01": "Priya"}}),
        encoding="utf-8")
    before_json = _hash(session / TRANSCRIPT_JSON)
    before_txt = _hash(session / TRANSCRIPT_TXT)
    dictionary = NamesDictionary(
        names=["John", "Priya", "Marcus"],
        corrections={"Meat Trace": "MeetTrace"})
    stats = clean_session(session, dictionary, tinyfix_enabled=False)
    # Verbatim files byte-identical afterwards (the hard rule).
    assert _hash(session / TRANSCRIPT_JSON) == before_json
    assert _hash(session / TRANSCRIPT_TXT) == before_txt
    # Derived artifacts exist with the cleaned content.
    assert (session / CLEAN_JSON).exists()
    assert (session / CLEAN_TXT).exists()
    clean_txt = (session / CLEAN_TXT).read_text(encoding="utf-8")
    assert "MeetTrace" in clean_txt
    assert "Meat Trace" not in clean_txt
    assert "Um," not in clean_txt and "Uh," not in clean_txt
    # Repeats kept verbatim (Nov 22 2026 rule removal — the old
    # rules deleted repeats real speech keeps; see
    # docs/notes/cleanup-repeat-rule-2026-11-22.md).
    assert "the the" in clean_txt
    assert "I think— I think" in clean_txt
    # Speaker display names applied, like the verbatim export.
    assert "John:" in clean_txt and "Priya:" in clean_txt
    assert stats["fillers_removed"] == 2
    assert stats["false_starts_repaired"] == 0
    assert stats["repeated_words_removed"] == 0
    assert stats["name_corrections_applied"] == 1
    assert stats["source_segments"] == 3
    assert stats["cleaned_segments"] == 3
    payload = json.loads((session / CLEAN_JSON).read_text())
    assert payload["cleaned_from"] == TRANSCRIPT_JSON
    assert payload["cleaning"]["fillers_removed"] == 2


def test_clean_session_without_transcript_raises(tmp_path):
    session = tmp_path / "empty"
    session.mkdir()
    with pytest.raises(FileNotFoundError):
        clean_session(session, NamesDictionary(), tinyfix_enabled=False)


def test_clean_session_idempotent(tmp_path):
    session = tmp_path / "s"
    _write_verbatim_session(session, ["Um, the the budget."])
    dictionary = NamesDictionary()
    first = clean_session(session, dictionary, tinyfix_enabled=False)
    first_txt = (session / CLEAN_TXT).read_text(encoding="utf-8")
    second = clean_session(session, dictionary, tinyfix_enabled=False)
    assert (session / CLEAN_TXT).read_text(encoding="utf-8") == first_txt
    assert second["fillers_removed"] == first["fillers_removed"]


# ------------------------------------------------------------------- polish --
class _StubOllama:
    """OllamaClient stand-in: canned rewrite, or unavailable."""

    def __init__(self, available=True, response="[00:00] Polished text."):
        self._available = available
        self._response = response
        self.prompts: list[str] = []

    def is_available(self):
        return self._available

    def summarize(self, text, prompt_template=None):
        self.prompts.append(prompt_template or "")
        return self._response


def test_polish_writes_marked_artifact(tmp_path):
    session = tmp_path / "s"
    _write_verbatim_session(session, ["Um, good morning."])
    clean_session(session, NamesDictionary(), tinyfix_enabled=False)
    stub = _StubOllama(response="[00:00] John: Good morning.")
    path = polish_session(session, ollama_client=stub)
    assert path.name == POLISHED_MD
    text = path.read_text(encoding="utf-8")
    assert "AI rewrite" in text
    assert "verbatim transcript" in text.lower()
    assert "Good morning." in text
    # The polish prompt went to the client, and it polished the
    # *cleaned* text (no "Um," in the input it saw).
    assert stub.prompts and "Um," not in stub.prompts[0].split(
        "Transcript:")[-1]
    # Verbatim + cleaned artifacts untouched by polishing.
    assert (session / TRANSCRIPT_JSON).exists()
    assert (session / CLEAN_TXT).exists()


def test_polish_ollama_unavailable_leaves_cleaned_in_place(tmp_path):
    session = tmp_path / "s"
    _write_verbatim_session(session, ["Good morning."])
    clean_session(session, NamesDictionary(), tinyfix_enabled=False)
    with pytest.raises(OllamaNotAvailableError, match="Ollama is not running"):
        polish_session(session, ollama_client=_StubOllama(available=False))
    assert not (session / POLISHED_MD).exists()
    assert (session / CLEAN_TXT).exists()


# ------------------------------------------------------- rename -> dictionary --
def test_rename_identity_adds_name_to_dictionary(tmp_path):
    import numpy as np

    from app.speakers.database import SpeakerDatabase
    from app.speakers.session import rename_identity

    db = SpeakerDatabase(tmp_path / "speakers.db")
    db.add_embedding("Speaker A", np.array([1.0, 0.0, 0.0, 0.0]),
                     backend="fallback-spectral-v1")
    dictionary_path = tmp_path / "names_dictionary.json"
    result = rename_identity(
        db, tmp_path, "Speaker A", "Priya",
        dictionary_path=dictionary_path)
    db.close()
    assert result["name"] == "Priya"
    dictionary = load_names_dictionary(dictionary_path)
    assert "Priya" in dictionary.names
    assert dictionary.corrections["Meat Trace"] == "MeetTrace"


# ------------------------------------------------------------------- fixture --
def test_fixture_clean_pass_keeps_verbatim_byte_identical(tmp_path):
    """Real transcribe (Whisper tiny, cached model) of the 61.3 s
    3-speaker fixture, then a clean pass: report the counts and
    prove the verbatim files are byte-identical afterwards.

    Skipped when faster-whisper is not importable (the shared
    phase2 venv has it, with the tiny model cached).
    """
    pytest.importorskip("faster_whisper")
    fixture = FIXTURES / "meeting-3speakers.mp3"
    if not fixture.exists():
        pytest.skip("fixture missing")
    session = tmp_path / "2026-11-07_10-00-00_teams"
    session.mkdir()
    shutil.copy(fixture, session / "combined.wav")
    engine = TranscriptionEngine(model_size="tiny")
    try:
        result = engine.transcribe_session(session)
    except Exception as exc:  # noqa: BLE001 — model/download failure
        pytest.skip(f"Whisper tiny could not run here: {exc}")
    assert len(result.segments) > 0
    before_json = _hash(session / TRANSCRIPT_JSON)
    before_txt = _hash(session / TRANSCRIPT_TXT)
    dictionary = load_names_dictionary(tmp_path / "names_dictionary.json")
    stats = clean_session(session, dictionary, tinyfix_enabled=False)
    assert _hash(session / TRANSCRIPT_JSON) == before_json
    assert _hash(session / TRANSCRIPT_TXT) == before_txt
    assert (session / CLEAN_JSON).exists()
    assert (session / CLEAN_TXT).exists()
    # The counts are the report — print them for the run log.
    print(f"\nfixture clean pass: {len(result.segments)} segments, "
          f"fillers={stats['fillers_removed']} "
          f"false_starts={stats['false_starts_repaired']} "
          f"repeats={stats['repeated_words_removed']} "
          f"corrections={stats['name_corrections_applied']} "
          f"dropped={stats['dropped_segments']}")
