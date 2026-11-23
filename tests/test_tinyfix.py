"""tinyfix stage tests: engine rules with a fake scorer, real-model
behaviour when the pinned files are available, and the cleanup
integration (names win, toggle-off identity, verbatim untouched,
download failure, English-only guard).

Real-model tests skip cleanly when the model files cannot be
downloaded — they prove behaviour against the real thing, not
against a fake, and never fail a suite for lack of network.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error

import pytest

from app.transcription.cleanup import (
    CLEAN_JSON,
    CLEAN_TXT,
    clean_session,
)
from app.transcription.engine import TRANSCRIPT_JSON, TRANSCRIPT_TXT
from app.transcription.names_dictionary import NamesDictionary
from app.transcription.tinyfix import (
    MODEL_FILES,
    TinyfixChecker,
    TinyfixUnavailableError,
    build_checker,
    ensure_model_files,
)
from app.transcription.tinyfix_engine import (
    Fixer,
    WordTrie,
    WrongWordChecker,
    parse_word_list,
    typo_cost,
)


# ------------------------------------------------------------ fake scorer ---

class _FakeModel:
    """Scores candidate words from a table, so engine rules are
    tested deterministically without the ONNX model."""

    def __init__(self, vocabulary, scores_by_word):
        self._vocabulary = vocabulary
        self._scores = scores_by_word

    def run(self, *, left, right, candidates):
        out = []
        for row in candidates:
            for token_id in row:
                word = self._vocabulary[token_id]
                out.append(self._scores.get(word, 0.0))
        return out


def _mini_checker(scores_by_word, mode="balanced"):
    """A TinyfixChecker over a toy vocab, with a fake model."""
    import tempfile
    from pathlib import Path

    vocab = ["<pad>", "<unk>", "<num>", "<s>", "<name>",
             "the", "their", "there", "they're", "form", "from",
             "to", "too", "your", "you're", "went", "store",
             "house", "going", "books", "seperate", "separate",
             "recieve", "receive", "offair", "meetings"]
    confusions = ["their there they're", "there their they're",
                  "form from", "to too", "your you're"]
    words = "\n".join(
        f"{word} {count}" for word, count in
        (("the", 1000), ("their", 100), ("there", 100),
         ("they're", 50), ("form", 80), ("from", 200),
         ("to", 900), ("too", 90), ("your", 120), ("you're", 60),
         ("went", 90), ("store", 70), ("house", 80),
         ("going", 110), ("books", 60), ("separate", 70),
         ("receive", 70), ("offair", 10), ("meetings", 40)))
    tmp = Path(tempfile.mkdtemp())
    (tmp / "wrong-word-vocab.txt").write_text(
        "\n".join(vocab), encoding="utf-8")
    (tmp / "wrong-word-confusions.txt").write_text(
        "\n".join(confusions), encoding="utf-8")
    (tmp / "words.txt").write_text(words, encoding="utf-8")
    (tmp / "wrong-word.onnx").write_bytes(b"")
    model = _FakeModel(vocab, scores_by_word)
    return TinyfixChecker(tmp, mode=mode, model=model)


# ------------------------------------------------------------- engine rules ---

def test_trie_finds_one_edit_neighbours():
    trie = WordTrie(["from", "form", "farm", "fort"])
    found = {m.index for m in trie.search("form", 1)}
    assert found == {0, 1, 2, 3}


def test_typo_cost_prefers_plausible_slips():
    # An adjacent-key slip costs less than a random wrong key.
    assert typo_cost("form", "from") < typo_cost("form", "fork")
    assert typo_cost("teh", "the") < typo_cost("teh", "ten")


def test_fixer_corrects_unknown_word_with_fake_scorer():
    words = parse_word_list("the 1000\nseparate 70\nreceive 70\n")
    checker = WrongWordChecker(
        _FakeModel(["<pad>", "<unk>", "<num>", "<s>", "<name>",
                    "the", "separate", "receive"],
                   {"separate": 5.0, "receive": 5.0}),
        ["<pad>", "<unk>", "<num>", "<s>", "<name>",
         "the", "separate", "receive"])
    fixer = Fixer(words, scorer=checker)
    correction = fixer.correct("We need to ", "seperate")
    assert correction is not None
    assert correction.replacement == "separate"
    assert correction.used_context is False


def test_fixer_leaves_known_words_and_names_alone():
    words = parse_word_list("the 1000\nfrom 200\n")
    fixer = Fixer(words)
    assert fixer.correct("I went ", "from") is None
    # Capitalised mid-sentence: a name, not a typo.
    assert fixer.correct("I saw ", "Marcus") is None
    # ALL-CAPS: an acronym, not a typo.
    assert fixer.correct("The ", "NASA") is None


def test_context_check_swaps_known_mixup_with_fake_scorer():
    # Margin 5 clears the balanced threshold (3) after the
    # cost weight and mix-up bonus: 5 - 1.5*1.5 + 1 = 3.75.
    scores = {"there": 0.0, "their": 5.0, "they're": 0.5}
    checker = _mini_checker(scores)
    fixed, corrections = checker.fix_text(
        "I went to there house.")
    assert fixed == "I went to their house."
    assert [c.original for c in corrections] == ["there"]
    assert corrections[0].used_context is True


def test_ignore_list_wins_over_context_check():
    scores = {"there": 0.0, "their": 4.0}
    checker = _mini_checker(scores)
    checker.ignore_words(["there"])
    fixed, corrections = checker.fix_text("I went to there house.")
    assert fixed == "I went to there house."
    assert corrections == []


def test_fix_segments_keeps_timestamps_and_maps_fixes_back():
    scores = {"there": 0.0, "their": 5.0, "they're": 0.5}
    checker = _mini_checker(scores)
    segments = [
        {"start": 0.0, "end": 2.0, "text": "I went to there"},
        {"start": 2.0, "end": 4.0, "text": "house yesterday."},
    ]
    fixed, corrections = checker.fix_segments(segments)
    assert fixed[0]["text"] == "I went to their"
    assert fixed[1]["text"] == "house yesterday."
    assert fixed[0]["start"] == 0.0 and fixed[0]["end"] == 2.0
    assert fixed[1]["start"] == 2.0 and fixed[1]["end"] == 4.0
    assert len(corrections) == 1


def test_fix_segments_lengthening_fix_keeps_later_segments_whole():
    """Regression: a fix that lengthens a word ("Their" -> "They're")
    must not eat characters off the following segment when the
    fixed text is split back."""
    scores = {"their": 0.0, "they're": 10.0, "there": 0.0}
    checker = _mini_checker(scores)
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Their going"},
        {"start": 2.0, "end": 4.0, "text": "to the store."},
    ]
    fixed, corrections = checker.fix_segments(segments)
    assert [c.original for c in corrections] == ["Their"]
    assert fixed[0]["text"] == "They're going"
    assert fixed[1]["text"] == "to the store."


# -------------------------------------------------------------- real model ---

def _model_dir_or_skip(tmp_path, monkeypatch):
    """Downloads the pinned files into a temp dir; skips the test
    when onnxruntime is missing or the network / Hugging Face is
    unreachable."""
    pytest.importorskip("onnxruntime")
    target = tmp_path / "tinyfix-models"
    monkeypatch.setattr(
        "app.transcription.tinyfix._models_root", lambda: target)
    try:
        ensure_model_files()
    except TinyfixUnavailableError as exc:
        pytest.skip(f"tinyfix model unavailable: {exc}")
    for filename, expected in MODEL_FILES:
        digest = hashlib.sha256(
            (target / filename).read_bytes()).hexdigest()
        assert digest == expected, filename
    return target


def _write_session(session, texts, language="en"):
    payload = {"model": "base", "language": language,
               "segments": [{"start": float(i * 3), "end": float(i * 3 + 2),
                             "text": text} for i, text in enumerate(texts)]}
    (session / TRANSCRIPT_JSON).write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (session / TRANSCRIPT_TXT).write_text(
        "\n".join(texts), encoding="utf-8")


@pytest.mark.parametrize("mode,expected_their", [
    ("balanced", "They're going to bring their books."),
    # Careful's higher bar (threshold 5) leaves the sentence-start
    # "Their" alone in isolation — measured behaviour, not a bug:
    # with fuller left context (see the clean_session test) it
    # does fire.
    ("careful", "Their going to bring their books."),
])
def test_real_model_fixes_planted_errors(tmp_path, monkeypatch, mode,
                                         expected_their):
    _model_dir_or_skip(tmp_path, monkeypatch)
    checker = build_checker(mode=mode)
    cases = {
        "I went form the store to there house.":
            "I went from the store to their house.",
        "Their going to bring there books.": expected_their,
        "Your going to love this.": "You're going to love this.",
        "We need to seperate the wheat form the chaff.":
            "We need to separate the wheat from the chaff.",
    }
    for source, expected in cases.items():
        fixed, _ = checker.fix_text(source)
        assert fixed == expected, (mode, source, fixed)


def test_real_model_leaves_correct_text_alone(tmp_path, monkeypatch):
    _model_dir_or_skip(tmp_path, monkeypatch)
    checker = build_checker(mode="balanced")
    text = ("The meeting is on Tuesday and the notes are in the "
            "shared folder. We discussed the budget and the "
            "deadline is Friday.")
    fixed, corrections = checker.fix_text(text)
    assert fixed == text
    assert corrections == []


def test_clean_session_with_real_model(tmp_path, monkeypatch):
    _model_dir_or_skip(tmp_path, monkeypatch)
    session = tmp_path / "s"
    session.mkdir()
    _write_session(session, ["Um, I went form the store.",
                             "Their going to bring there books."])
    before = (session / TRANSCRIPT_JSON).read_bytes()
    stats = clean_session(session, NamesDictionary(),
                          tinyfix_enabled=True, tinyfix_mode="balanced")
    assert stats["tinyfix_corrections_applied"] >= 2
    assert stats["tinyfix_skipped"] == ""
    # Verbatim untouched; derived artifact carries the fixes.
    assert (session / TRANSCRIPT_JSON).read_bytes() == before
    cleaned = json.loads((session / CLEAN_JSON).read_text(
        encoding="utf-8"))
    assert cleaned["segments"][0]["text"] == "I went from the store."
    assert cleaned["segments"][1]["text"] == \
        "They're going to bring their books."
    assert cleaned["cleaning"]["tinyfix_corrections_applied"] >= 2


def test_names_dictionary_words_untouched_by_real_model(
        tmp_path, monkeypatch):
    _model_dir_or_skip(tmp_path, monkeypatch)
    session = tmp_path / "s"
    session.mkdir()
    _write_session(session, ["We talked about OffAir and MeetTrace."])
    dictionary = NamesDictionary(
        names=["OffAir"], corrections={"Meat Trace": "OffAir"})
    stats = clean_session(session, dictionary, tinyfix_enabled=True,
                          tinyfix_mode="balanced")
    cleaned = json.loads((session / CLEAN_JSON).read_text(
        encoding="utf-8"))
    assert cleaned["segments"][0]["text"] == \
        "We talked about OffAir and MeetTrace."


# ------------------------------------------------------- integration guards ---

def test_toggle_off_is_deterministic_only(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    _write_session(session, ["Um, I went form the store."])
    stats = clean_session(session, NamesDictionary(),
                          tinyfix_enabled=False)
    assert stats["tinyfix_corrections_applied"] == 0
    assert stats["tinyfix_skipped"] == "disabled"
    cleaned = json.loads((session / CLEAN_JSON).read_text(
        encoding="utf-8"))
    # Deterministic rules ran (filler gone); tinyfix did not.
    assert cleaned["segments"][0]["text"] == "I went form the store."
    first_bytes = (session / CLEAN_JSON).read_bytes()
    clean_session(session, NamesDictionary(), tinyfix_enabled=False)
    assert (session / CLEAN_JSON).read_bytes() == first_bytes


def test_clean_session_default_is_tinyfix_off(tmp_path):
    """Nov 22 2026: the shipped default flipped from ON to OFF
    (opt-in) after test 2 showed the stage changing correct
    words on disfluent real speech. Omitting the argument must
    behave exactly like tinyfix_enabled=False — the checker is
    never built."""
    session = tmp_path / "s"
    session.mkdir()
    _write_session(session, ["Um, I went form the store."])

    class _BoomChecker:
        def ignore_words(self, words):
            raise AssertionError("checker must not be built")

        def fix_segments(self, segments):
            raise AssertionError("checker must not run")

    stats = clean_session(session, NamesDictionary(),
                          tinyfix_checker=_BoomChecker())
    assert stats["tinyfix_corrections_applied"] == 0
    assert stats["tinyfix_skipped"] == "disabled"
    cleaned = json.loads((session / CLEAN_JSON).read_text(
        encoding="utf-8"))
    assert cleaned["segments"][0]["text"] == "I went form the store."
    from app.config import TINYFIX_DEFAULT_ENABLED
    assert TINYFIX_DEFAULT_ENABLED is False


def test_no_change_checker_matches_toggle_off(tmp_path):
    """A checker that finds nothing must produce byte-identical
    output to the toggle-off run."""
    session = tmp_path / "s"
    session.mkdir()
    _write_session(session, ["Um, the the budget."])
    clean_session(session, NamesDictionary(), tinyfix_enabled=False)
    off_bytes = (session / CLEAN_JSON).read_bytes()
    off_txt = (session / CLEAN_TXT).read_bytes()

    class _NoopChecker:
        def ignore_words(self, words):
            pass

        def fix_segments(self, segments):
            return segments, []

    clean_session(session, NamesDictionary(), tinyfix_enabled=True,
                  tinyfix_checker=_NoopChecker())
    # The cleaned text is identical; only the recorded
    # tinyfix_skipped reason differs ("disabled" vs ran clean).
    assert (session / CLEAN_TXT).read_bytes() == off_txt
    cleaned = json.loads((session / CLEAN_JSON).read_text(
        encoding="utf-8"))
    off = json.loads(off_bytes)
    assert cleaned["cleaning"]["tinyfix_skipped"] == ""
    cleaned["cleaning"]["tinyfix_skipped"] = \
        off["cleaning"]["tinyfix_skipped"]
    assert cleaned == off


def test_download_failure_degrades_to_deterministic_only(
        tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    _write_session(session, ["Um, I went form the store."])
    empty_dir = tmp_path / "empty-models"
    monkeypatch.setattr(
        "app.transcription.tinyfix._models_root", lambda: empty_dir)

    def _boom(*args, **kwargs):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    stats = clean_session(session, NamesDictionary(),
                          tinyfix_enabled=True)
    assert stats["tinyfix_skipped"].startswith("unavailable:")
    assert stats["tinyfix_corrections_applied"] == 0
    cleaned = json.loads((session / CLEAN_JSON).read_text(
        encoding="utf-8"))
    assert cleaned["segments"][0]["text"] == "I went form the store."


def test_non_english_transcript_skips_tinyfix(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    _write_session(session, ["Je suis allé form le magasin."],
                   language="fr")

    class _BoomChecker:
        def ignore_words(self, words):
            raise AssertionError("checker must not be built")

        def fix_segments(self, segments):
            raise AssertionError("checker must not run")

    stats = clean_session(session, NamesDictionary(),
                          tinyfix_enabled=True,
                          tinyfix_checker=_BoomChecker())
    assert stats["tinyfix_skipped"].startswith("non-English transcript")
    assert stats["tinyfix_corrections_applied"] == 0


def test_unknown_language_runs_tinyfix(tmp_path):
    """Older transcripts carry no language key: the stage runs
    (documented behaviour — English is the app's only supported
    transcription language)."""
    session = tmp_path / "s"
    session.mkdir()
    payload = {"model": "base",
               "segments": [{"start": 0.0, "end": 2.0,
                             "text": "I went form the store."}]}
    (session / TRANSCRIPT_JSON).write_text(
        json.dumps(payload), encoding="utf-8")
    (session / TRANSCRIPT_TXT).write_text(
        "I went form the store.", encoding="utf-8")

    seen = {}

    class _RecordingChecker:
        def ignore_words(self, words):
            seen["ignored"] = list(words)

        def fix_segments(self, segments):
            seen["ran"] = True
            return segments, []

    stats = clean_session(session, NamesDictionary(),
                          tinyfix_enabled=True,
                          tinyfix_checker=_RecordingChecker())
    assert seen.get("ran") is True
    assert stats["tinyfix_skipped"] == ""


def test_checker_gets_dictionary_words_on_ignore_list(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    _write_session(session, ["Hello."])
    seen = {}

    class _RecordingChecker:
        def ignore_words(self, words):
            seen["ignored"] = list(words)

        def fix_segments(self, segments):
            return segments, []

    dictionary = NamesDictionary(
        names=["Priya Sharma"],
        corrections={"Meat Trace": "OffAir"})
    clean_session(session, dictionary, tinyfix_enabled=True,
                  tinyfix_checker=_RecordingChecker())
    import re

    ignored = {piece.lower() for phrase in seen["ignored"]
               for piece in re.findall(r"[A-Za-z']+", phrase)}
    assert {"priya", "sharma", "meat", "trace", "offair"} <= ignored
