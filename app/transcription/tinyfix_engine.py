"""Faithful Python port of the tinyfix engine (paperweight/tinyfix, MIT).

Source: github.com/cfahlgren1/tinyfix, ``src/engine/{rules,trie,
typoCost,fixer,wrongWord}.ts`` plus the ``check()`` orchestration in
``src/index.ts``. This module is a pure port: no I/O, no onnxruntime
import — the model is injected as a callable (see
:mod:`app.transcription.tinyfix`), so the rules can be tested with a
fake scorer. Behaviour notes that matter for the port are kept as
comments citing the TS original.

Two fix kinds, exactly as the package defines them:

* ``spelling`` — the word is not a known word; candidates come from
  a trie search within 1–2 edits, ranked by typo cost and word
  frequency, optionally re-ranked by the model.
* ``context`` — the word is correctly spelled but does not fit the
  words around it; nearby real words (known mix-ups first, then
  one-edit neighbours) are scored by the model against 16 words
  before and 4 after, and swapped only when the best beats the
  typed word by a clear margin.

The swap decision lives here, in these rules — the model only
scores words.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- rules.ts ---

_CONTRACTIONS: dict[str, str] = {
    "arent": "aren't", "cant": "can't", "couldnt": "couldn't",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't",
    "im": "I'm", "isnt": "isn't", "ive": "I've",
    "shouldnt": "shouldn't", "thats": "that's", "theres": "there's",
    "theyre": "they're", "theyve": "they've", "wasnt": "wasn't",
    "werent": "weren't", "weve": "we've", "whats": "what's",
    "wont": "won't", "wouldnt": "wouldn't", "youre": "you're",
    "youve": "you've",
}

_BRITISH_OUR_ENDING = re.compile(
    r"(.{2,})our(s|ed|ing|ite|ites|ful|able|ably|less|hood|hoods|ly|ist|ists|er|ers)?$")
_BRITISH_ISE_ENDING = re.compile(r"(is|ys)(e|es|ed|ing|er|ers|ation|ations|able)$")
_BRITISH_DOUBLED_L = re.compile(
    r"^(un|re|mis|pre)?(travel|cancel|model|label|signal|fuel|level|total|marvel|counsel|jewel|tunnel|channel|shovel|dial|equal|quarrel|rival|panel|funnel|pencil|spiral|trial)l(ed|ing|er|ers|or|ors)$")
_MIN_LENGTH_FOR_ISE_RULE = 7
_BRITISH_DOUBLED_L_ENDING = re.compile(r"ll(ed|ing|er|ers|or|ors|ist|ists|ous)$")
_BRITISH_RE_ENDING = re.compile(r"([^aeiou])re(s|d)?$")
_MIN_LENGTH_FOR_RE_RULE = 5
_BRITISH_DIGRAPH = re.compile(r"(ae|oe)")
_MIN_LENGTH_FOR_DIGRAPH_RULE = 7

_BRITISH_WORDS: frozenset[str] = frozenset(
    word
    for base in (
        "theatre centre centred metre litre fibre calibre sombre spectre "
        "defence offence licence pretence catalogue analogue programme "
        "judgement ageing aluminium fulfil enrol skilful wilful jewellery "
        "tyre grey mould moulding plough sceptical sceptic cheque kerb "
        "draught manoeuvre paediatric anaesthetic foetus oestrogen "
        "moustache pyjamas storey acknowledgement instalment distil "
        "instil appal whisky enquiry amongst whilst").split()
    # The TS source adds each word and its plural (flatMap).
    for word in (base, base + "s")
)

#: Everyday words newer than the word list, which would otherwise
#: be "corrected" (emoji -> meiji, todo -> too). Singular + plural,
#: exactly as the TS source builds the set (flatMap word + "s").
_NEWER_WORDS: frozenset[str] = frozenset(
    word
    for base in (
        "https emoji todo selfie hashtag livestream chatbot vlog vlogger "
        "influencer fintech covid podcaster blogger unfollow unfriend "
        "retweet screenshare webinar onboarding offboarding skincare "
        "smartwatch earbud airpod wifi bluetooth async runtime frontend "
        "backend dataset username login logout signup checkout touchscreen "
        "spreadsheet workflow timeline toolbar dropdown popup tooltip "
        "codebase repo config dev devops cybersecurity crypto "
        "kombucha matcha ramen udon pho bao boba poke quinoa kale hummus "
        "falafel tahini sriracha kimchi gochujang tofu tempeh seitan acai "
        "chia granola brunch latte cappuccino espresso macchiato "
        "americano frappe smoothie barista crema robusta arabica "
        "croissant focaccia ciabatta brioche sourdough gelato prosecco "
        "charcuterie hygge wellness mindfulness hangry yeet meme binge "
        "rewatch preorder crowdfund carpool rideshare staycation "
        "glamping bae vibe vibes ghosting doomscroll doomscrolling "
        "cringe adulting").split()
    for word in (base, base + "s")
)


def is_newer_word(lower: str) -> bool:
    return lower in _NEWER_WORDS


def is_british_spelling(lower: str, is_known) -> bool:
    """Recognizes British spellings, so "colour" isn't treated as a
    misspelling of "color". ``is_known`` answers whether a word is
    in the word list."""
    if lower in _BRITISH_WORDS:
        return True
    if _BRITISH_DOUBLED_L.match(lower):
        return True
    with_or = _BRITISH_OUR_ENDING.sub(r"\1or\2", lower)
    if with_or != lower and is_known(with_or):
        return True
    single_l = _BRITISH_DOUBLED_L_ENDING.sub(r"l\1", lower)
    if single_l != lower and is_known(single_l):
        return True
    if len(lower) >= _MIN_LENGTH_FOR_RE_RULE:
        with_er = _BRITISH_RE_ENDING.sub(
            lambda m: f"{m.group(1)}er"
            + ("ed" if m.group(2) == "d" else (m.group(2) or "")),
            lower)
        if with_er != lower and is_known(with_er):
            return True
    if (len(lower) >= _MIN_LENGTH_FOR_DIGRAPH_RULE
            and _BRITISH_DIGRAPH.search(lower)
            and is_known(_BRITISH_DIGRAPH.sub("e", lower, count=1))):
        return True
    if len(lower) < _MIN_LENGTH_FOR_ISE_RULE:
        return False

    def _ise_replace(m: re.Match) -> str:
        stem, ending = m.group(1), m.group(2)
        return ("iz" if stem == "is" else "yz") + ending

    with_z = _BRITISH_ISE_ENDING.sub(_ise_replace, lower)
    return with_z != lower and is_known(with_z)


def is_sentence_start(text_before: str) -> bool:
    return re.search(r"(^|[.!?:\n]\s*|^\s*[-*•]\s+)$", text_before) is not None


def match_case(typed: str, replacement: str) -> str:
    """Applies typed casing to a replacement, so "Teh" becomes "The"."""
    if not typed or not replacement:
        return replacement
    if typed[0] != typed[0].lower():
        return replacement[0].upper() + replacement[1:]
    return replacement


def apply_fixed_rule(word: str) -> str | None:
    if word == "i":
        return "I"
    contraction = _CONTRACTIONS.get(word.lower())
    if contraction is None:
        return None
    return contraction if contraction.startswith("I") else match_case(word, contraction)


def is_checkable(word: str, text_before: str) -> bool:
    """Words the engine must leave alone: names, acronyms,
    identifiers and anything not plainly a word."""
    if len(word) < 3 or not re.fullmatch(r"[A-Za-z]+", word):
        return False
    if word == word.upper():
        return False
    if re.search(r"[A-Z]", word[1:]):
        return False
    is_capitalized = word[0] != word[0].lower()
    if is_capitalized and not is_sentence_start(text_before):
        return False
    previous_token = text_before.rstrip().split()[-1] if text_before.rstrip() else ""
    return re.search(r"[/@\\_#]|\.\w", previous_token) is None


# ----------------------------------------------------------------- trie.ts ---

@dataclass
class TrieMatch:
    index: int
    distance: int


class WordTrie:
    """Compact trie that finds dictionary words within a small edit
    distance of a query. Same array layout as the TS original: new
    siblings are prepended, so children iterate in reverse
    insertion order (outcomes only depend on this for exact ties,
    where both sorts are stable)."""

    def __init__(self, words) -> None:
        self._character: list[int] = [0]
        self._first_child: list[int] = [-1]
        self._next_sibling: list[int] = [-1]
        self._word_index: list[int] = [-1]
        for index, word in enumerate(words):
            self._insert(word, index)

    def _insert(self, word: str, index: int) -> None:
        node = 0
        for position in range(len(word)):
            node = self._find_or_create_child(node, ord(word[position]))
        self._word_index[node] = index

    def _find_or_create_child(self, parent: int, code: int) -> int:
        child = self._first_child[parent]
        while child != -1:
            if self._character[child] == code:
                return child
            child = self._next_sibling[child]
        created = len(self._character)
        self._character.append(code)
        self._first_child.append(-1)
        self._next_sibling.append(self._first_child[parent])
        self._word_index.append(-1)
        self._first_child[parent] = created
        return created

    def search(self, query: str, max_edits: int) -> list[TrieMatch]:
        """Every word within ``max_edits`` insertions, deletions,
        substitutions or adjacent swaps."""
        width = len(query) + 1
        matches: list[TrieMatch] = []
        typed = [ord(query[position]) for position in range(len(query))]

        def walk(node: int, previous_code: int, row: list[int],
                 row_before: list[int] | None) -> None:
            child = self._first_child[node]
            while child != -1:
                code = self._character[child]
                nxt = [0] * width
                nxt[0] = row[0] + 1
                smallest = nxt[0]
                for column in range(1, width):
                    substitution = row[column - 1] + (
                        0 if typed[column - 1] == code else 1)
                    value = min(row[column] + 1, nxt[column - 1] + 1,
                                substitution)
                    is_swap = (row_before is not None and column > 1
                               and typed[column - 1] == previous_code
                               and typed[column - 2] == code)
                    if is_swap:
                        value = min(value, row_before[column - 2] + 1)
                    nxt[column] = value
                    smallest = min(smallest, value)
                distance = nxt[width - 1]
                if self._word_index[child] != -1 and distance <= max_edits:
                    matches.append(TrieMatch(
                        index=self._word_index[child], distance=distance))
                if smallest <= max_edits:
                    walk(child, code, nxt, row)
                child = self._next_sibling[child]

        walk(0, -1, list(range(width)), None)
        return matches


# ------------------------------------------------------------- typoCost.ts ---

_KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_VOWELS = frozenset("aeiou")

_COST = {
    "swap": 1.2,
    "doubled_key": 1.2,
    "adjacent_key": 1.6,
    "vowel_for_vowel": 2.0,
    "extra_nearby_key": 1.8,
    "missed_key": 2.4,
    "extra_key": 3.0,
    "wrong_key": 3.4,
    "first_letter_penalty": 1.5,
}


def _build_neighbors() -> dict[str, frozenset[str]]:
    neighbors: dict[str, set[str]] = {}
    for row_index, row in enumerate(_KEYBOARD_ROWS):
        for key_index, key in enumerate(row):
            near: set[str] = set()
            for other_row, shift in ((row_index, -1), (row_index, 1),
                                    (row_index - 1, 0), (row_index - 1, 1),
                                    (row_index + 1, -1), (row_index + 1, 0)):
                if 0 <= other_row < len(_KEYBOARD_ROWS):
                    other = _KEYBOARD_ROWS[other_row][
                        key_index + shift] if 0 <= key_index + shift < len(
                            _KEYBOARD_ROWS[other_row]) else None
                    if other:
                        near.add(other)
            neighbors[key] = frozenset(near)
    return neighbors


_NEIGHBORS = _build_neighbors()


def _is_nearby(first: str | None, second: str | None) -> bool:
    return bool(first and second and second in _NEIGHBORS.get(first, frozenset()))


def _substitution_cost(typed: str, intended: str) -> float:
    if _is_nearby(typed, intended):
        return _COST["adjacent_key"]
    if typed in _VOWELS and intended in _VOWELS:
        return _COST["vowel_for_vowel"]
    return _COST["wrong_key"]


def _extra_key_cost(typed: str, position: int) -> float:
    key = typed[position]
    prev_key = typed[position - 1] if position > 0 else None
    next_key = typed[position + 1] if position + 1 < len(typed) else None
    if key == prev_key or key == next_key:
        return _COST["doubled_key"]
    if _is_nearby(key, prev_key) or _is_nearby(key, next_key):
        return _COST["extra_nearby_key"]
    return _COST["extra_key"]


def _missed_key_cost(intended: str, position: int) -> float:
    if position > 0 and intended[position] == intended[position - 1]:
        return _COST["doubled_key"]
    return _COST["missed_key"]


def _start(position: int) -> float:
    """A slip on a word's first letter is less likely, so it costs more."""
    return _COST["first_letter_penalty"] if position == 0 else 0.0


def typo_cost(typed: str, intended: str) -> float:
    """Weighted edit distance: a lower cost means the typo is a more
    believable slip of the fingers."""
    rows = len(typed) + 1
    columns = len(intended) + 1
    table = [[0.0] * columns for _ in range(rows)]
    for row in range(1, rows):
        table[row][0] = (table[row - 1][0]
                         + _extra_key_cost(typed, row - 1) + _start(row - 1))
    for column in range(1, columns):
        table[0][column] = (table[0][column - 1]
                            + _missed_key_cost(intended, column - 1)
                            + _start(column - 1))
    for row in range(1, rows):
        for column in range(1, columns):
            typed_key = typed[row - 1]
            intended_key = intended[column - 1]
            first_letter = _start(min(row, column) - 1)
            replace = (0.0 if typed_key == intended_key
                       else _substitution_cost(typed_key, intended_key)
                       + first_letter)
            best = min(
                table[row - 1][column] + _extra_key_cost(typed, row - 1)
                + first_letter,
                table[row][column - 1]
                + _missed_key_cost(intended, column - 1) + first_letter,
                table[row - 1][column - 1] + replace,
            )
            swapped = (row > 1 and column > 1
                       and typed_key == intended[column - 2]
                       and typed[row - 2] == intended_key)
            if swapped:
                best = min(best, table[row - 2][column - 2] + _COST["swap"])
            table[row][column] = best
    return table[rows - 1][columns - 1]


# ---------------------------------------------------------------- fixer.ts ---

@dataclass
class FixerWeights:
    frequency: float
    context: float
    keep_cost: float
    min_confidence: float


DEFAULT_WEIGHTS = FixerWeights(frequency=0.5, context=0.0, keep_cost=10.0,
                               min_confidence=0.7)
#: Use with a context scorer, so the model helps choose between
#: spelling candidates.
WITH_MODEL_WEIGHTS = FixerWeights(frequency=0.5, context=0.25,
                                  keep_cost=10.0, min_confidence=0.7)

#: Sure enough to change a misspelled word, with the model helping
#: to choose — the bar index.ts uses when a scorer is present.
SPELLING_MIN_CONFIDENCE = 0.55


@dataclass
class WordEntry:
    word: str
    count: int


@dataclass
class Candidate:
    word: str
    cost: float
    log_frequency: float
    context_gain: float = 0.0


@dataclass
class Correction:
    original: str
    replacement: str
    confidence: float
    #: True when the word was spelled correctly but didn't fit the
    #: sentence (a "context" fix, not a "spelling" fix).
    used_context: bool = False


def parse_word_list(text: str) -> list[WordEntry]:
    entries: list[WordEntry] = []
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2:
            entries.append(WordEntry(word=parts[0], count=int(parts[1])))
    return entries


def pick_candidate(candidates, weights: FixerWeights):
    """Returns ``(word, confidence)`` or None. The "keep the word"
    option competes with a score of ``-keep_cost``."""
    if not candidates:
        return None
    scored = [
        (c.word,
         -c.cost + weights.frequency * c.log_frequency
         + weights.context * c.context_gain)
        for c in candidates
    ]
    best_word, best_score = max(scored, key=lambda item: item[1])
    total = sum(math.exp(score - best_score) for _, score in scored)
    total += math.exp(-weights.keep_cost - best_score)
    confidence = 1.0 / total
    if confidence >= weights.min_confidence:
        return best_word, confidence
    return None


class Fixer:
    """Spelling corrections: unknown words ranked by typo cost and
    frequency, optionally re-ranked by a context scorer."""

    def __init__(self, words, trie: WordTrie | None = None,
                 scorer=None, weights: FixerWeights | None = None,
                 max_candidates: int = 12) -> None:
        self._words = list(words)
        self._known = {entry.word for entry in self._words}
        self._trie = trie if trie is not None else WordTrie(
            [entry.word for entry in self._words])
        self._scorer = scorer
        self._weights = FixerWeights(
            **{**DEFAULT_WEIGHTS.__dict__, **(weights.__dict__ if weights else {})})
        self._max_candidates = max_candidates
        self._total_count = sum(entry.count for entry in self._words)
        self._undone: set[str] = set()

    def learn(self, word: str) -> None:
        """Treats a word as correct from now on, for example after
        the user undoes a fix."""
        self._known.add(word.lower())

    def is_known(self, word: str) -> bool:
        lower = word.lower()
        return (lower in self._known or is_newer_word(lower)
                or is_british_spelling(lower, lambda w: w in self._known))

    def was_undone(self, word: str) -> bool:
        return word.lower() in self._undone

    def remember_undo(self, word: str) -> None:
        """Stops a fixed rule from firing again for a word the user
        restored."""
        self._undone.add(word.lower())
        self.learn(word)

    def analyze(self, text_before: str, word: str):
        """Ranks spelling candidates for an unknown word, or returns
        None when the word should be left alone."""
        if not is_checkable(word, text_before) or self.is_known(word):
            return None
        lower = word.lower()
        max_edits = 1 if len(lower) <= 4 else 2
        ranked = []
        for match in self._trie.search(lower, max_edits):
            entry = self._words[match.index]
            ranked.append(Candidate(
                word=entry.word,
                cost=typo_cost(lower, entry.word),
                log_frequency=math.log(entry.count / self._total_count)))
        ranked.sort(key=lambda c: c.log_frequency - c.cost, reverse=True)
        ranked = ranked[:self._max_candidates]
        if self._scorer is None or not ranked:
            return ranked
        scores = self._scorer.score(text_before, [
            word, *[match_case(word, c.word) for c in ranked]])
        typed_score = scores[0]
        return [Candidate(word=c.word, cost=c.cost,
                          log_frequency=c.log_frequency,
                          context_gain=scores[index + 1] - typed_score)
                for index, c in enumerate(ranked)]

    def correct(self, text_before: str, word: str):
        """Returns a correction for the word just finished after
        ``text_before``, or None to keep it."""
        ruled = apply_fixed_rule(word)
        if ruled is not None and ruled != word and not self.was_undone(word):
            return Correction(original=word, replacement=ruled, confidence=1.0)
        candidates = self.analyze(text_before, word)
        if not candidates:
            return None
        pick = pick_candidate(candidates, self._weights)
        if pick is None:
            return None
        picked_word, confidence = pick
        return Correction(original=word,
                          replacement=match_case(word, picked_word),
                          confidence=confidence)


# ------------------------------------------------------------ wrongWord.ts ---

LEFT_WINDOW = 16
RIGHT_WINDOW = 4
MAX_NEIGHBORS = 8
CANDIDATES = MAX_NEIGHBORS + 1

PAD = 0
UNKNOWN = 1
NUMBER = 2
START = 3
NAME = 4
SPECIAL_TOKENS = 5

#: Known mix-ups like "their" and "there" are likely even when
#: they're far apart on the keyboard.
MAX_KNOWN_MIXUP_COST = 1.5
#: A known mix-up needs to clear a lower bar than other nearby
#: words, which are mostly slips of the fingers.
KNOWN_MIXUP_BONUS = 1.0
#: For a word with known mix-ups, any other nearby word needs to
#: clear a higher bar: "your" is rarely meant as "you".
OTHER_WORD_PENALTY = 1.0
#: A word that only differs by an ending ("examples" and "example")
#: needs to clear a higher bar, unless it's a known mix-up.
INFLECTION_PENALTY = 1.0
INFLECTION_ENDINGS = frozenset(["", "s", "es", "d", "ed", "ing", "r", "er", "ly"])
MIN_STEM = 3

_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|\n|[^\s\w]", re.ASCII)
_SENTENCE_OPENERS = frozenset([".", "!", "?", ":", '"'])
_WORD_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")
_LISTED_WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)?$")


@dataclass
class WrongWordSettings:
    #: A nearby word must beat the typed word by at least this much.
    threshold: float
    #: How strongly an unlikely slip of the fingers counts against a
    #: nearby word.
    cost_weight: float


#: Tuned on a dev split to wrongly change at most about 1 correct
#: word in 2,000.
CAREFUL_WRONG_WORD = WrongWordSettings(threshold=5.0, cost_weight=0.5)
#: The "balanced" mode: fixes about 43% of wrong words and wrongly
#: changes about 1 correct word in 1,750 on the lab bench's test
#: sets.
EAGER_WRONG_WORD = WrongWordSettings(threshold=3.0, cost_weight=1.5)

#: Mode name -> settings, mirroring ``loadTinyfix``: "balanced" is
#: the package default (what offlinepage.app uses); "careful" makes
#: fewer wrong changes and fixes fewer wrong words.
WRONG_WORD_MODES: dict[str, WrongWordSettings] = {
    "balanced": EAGER_WRONG_WORD,
    "careful": CAREFUL_WRONG_WORD,
}


def differs_only_in_ending(first: str, second: str) -> bool:
    """True for pairs like "services"/"service" or
    "continued"/"continues"."""
    shared = 0
    limit = min(len(first), len(second))
    while shared < limit and first[shared] == second[shared]:
        shared += 1
    for stem in range(shared, MIN_STEM - 1, -1):
        if (first[stem:] in INFLECTION_ENDINGS
                and second[stem:] in INFLECTION_ENDINGS):
            return True
    return False


def _tokenize(text: str, starts_sentence: bool) -> list[str]:
    """Lowercases words, except capitalized words mid-sentence,
    which become a single name token. A line break ends a sentence,
    so the end of the paragraph before reads like the sentence
    before."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.replace("’", "'")):
        if raw == "\n":
            if tokens and tokens[-1] not in _SENTENCE_OPENERS:
                tokens.append(".")
            continue
        previous = tokens[-1] if tokens else None
        at_sentence_start = (starts_sentence if previous is None
                             else previous in _SENTENCE_OPENERS)
        is_pronoun = raw == "I" or raw.startswith("I'")
        if raw[0].isupper() and not at_sentence_start and not is_pronoun:
            tokens.append("<name>")
        else:
            tokens.append(raw.lower())
    return tokens


@dataclass
class NearbyWord:
    word: str
    #: How much better this word fits than the typed word, as a
    #: log-odds difference.
    margin: float
    #: How unlikely it is to type the given word when meaning this one.
    cost: float
    #: True when the word is a known mix-up for the typed word.
    is_known_mixup: bool


class WrongWordChecker:
    """Compares a correctly spelled word with nearby real words,
    using the words on both sides of it. Also serves as the
    context scorer for :class:`Fixer`."""

    def __init__(self, model, vocabulary, confusions=None,
                 settings: WrongWordSettings | None = None) -> None:
        self._model = model
        self.settings = settings if settings is not None else CAREFUL_WRONG_WORD
        self._vocabulary = list(vocabulary)
        self._ids: dict[str, int] = {}
        self._targets: list[int] = []
        for index, token in enumerate(self._vocabulary):
            self._ids[token] = index
            if (index >= SPECIAL_TOKENS and len(token) >= 2
                    and _WORD_RE.match(token)):
                self._targets.append(index)
        self._trie = WordTrie(
            [self._vocabulary[index] for index in self._targets])
        self._neighbors: dict[int, list[int]] = {}
        self._known_mixups: dict[int, list[int]] = {}
        for line in confusions or []:
            parts = line.split(" ")
            typed = self._ids.get(parts[0]) if parts else None
            if typed is not None:
                self._known_mixups[typed] = [
                    self._ids[token] for token in parts[1:]
                    if token in self._ids]

    def _token_id(self, token: str) -> int:
        if token == "<name>":
            return NAME
        if token and token[0].isdigit():
            return NUMBER
        return self._ids.get(token, UNKNOWN)

    def _nearby_ids(self, word_id: int) -> list[int]:
        cached = self._neighbors.get(word_id)
        if cached is not None:
            return cached
        one_edit = sorted(
            self._targets[match.index]
            for match in self._trie.search(self._vocabulary[word_id], 1))
        one_edit = [other for other in one_edit if other != word_id]
        found = list(dict.fromkeys(
            [*self._known_mixups.get(word_id, []), *one_edit]))[:MAX_NEIGHBORS]
        self._neighbors[word_id] = found
        return found

    def _run_model(self, text_before: str, text_after: str,
                   rows: list[list[int]]) -> list[float]:
        before = [self._token_id(t) for t in _tokenize(text_before, True)]
        if len(before) >= LEFT_WINDOW:
            left = before[-LEFT_WINDOW:]
        else:
            left = [PAD] * (LEFT_WINDOW - len(before) - 1) + [START] + before
        right_tokens = [self._token_id(t)
                        for t in _tokenize(text_after, False)[:RIGHT_WINDOW]]
        right_tokens.reverse()
        right = [PAD] * (RIGHT_WINDOW - len(right_tokens)) + right_tokens
        tiled_left = left * len(rows)
        tiled_right = right * len(rows)
        candidates: list[int] = []
        for row in rows:
            padded = list(row) + [PAD] * (CANDIDATES - len(row))
            candidates.extend(padded)
        return self._model.run(
            left=[tiled_left[i * LEFT_WINDOW:(i + 1) * LEFT_WINDOW]
                  for i in range(len(rows))],
            right=[tiled_right[i * RIGHT_WINDOW:(i + 1) * RIGHT_WINDOW]
                   for i in range(len(rows))],
            candidates=[candidates[i * CANDIDATES:(i + 1) * CANDIDATES]
                        for i in range(len(rows))])

    def compare(self, text_before: str, word: str, text_after: str,
                only_known_mixups: bool = False):
        """Returns nearby words with how well each fits, or None when
        the word isn't one the model can judge."""
        typed = self._ids.get(word.lower())
        if typed is None or typed < SPECIAL_TOKENS:
            return None
        nearby = (self._known_mixups.get(typed, []) if only_known_mixups
                  else self._nearby_ids(typed))
        if not nearby:
            return None
        scores = self._run_model(text_before, text_after, [[typed, *nearby]])
        mixups = self._known_mixups.get(typed, [])
        result = []
        for index, word_id in enumerate(nearby):
            nearby_word = self._vocabulary[word_id]
            cost = typo_cost(word.lower(), nearby_word)
            is_mixup = word_id in mixups
            result.append(NearbyWord(
                word=nearby_word,
                margin=scores[index + 1] - scores[0],
                cost=min(cost, MAX_KNOWN_MIXUP_COST) if is_mixup else cost,
                is_known_mixup=is_mixup))
        return result

    def score(self, text_before: str, words):
        """Scores spelling candidates by how well they fit the text
        before them. The first word is the misspelling itself (scored
        as unknown)."""
        ids = [UNKNOWN, *[self._ids.get(w.lower(), UNKNOWN)
                          for w in list(words)[1:]]]
        rows: list[list[int]] = []
        for start in range(0, len(ids), MAX_NEIGHBORS):
            rows.append([PAD, *ids[start:start + MAX_NEIGHBORS]])
        scores = self._run_model(text_before, "", rows)
        return [scores[(index // MAX_NEIGHBORS) * CANDIDATES
                       + (index % MAX_NEIGHBORS) + 1]
                for index in range(len(ids))]

    def _has_known_mixups(self, word: str, text_before: str) -> bool:
        word_id = self._ids.get(word.lower())
        if word_id is None or word_id not in self._known_mixups:
            return False
        if not _LISTED_WORD_RE.match(word):
            return False
        if len(word) > 1 and word == word.upper():
            return False
        is_capitalized = word[0] != word[0].lower()
        return not is_capitalized or is_sentence_start(text_before)

    def judged_against(self, word: str, text_before: str):
        """Whether a word is judged at all, and if so whether only
        against its known mix-ups. None means leave it alone."""
        if is_checkable(word, text_before):
            return "nearby"
        if self._has_known_mixups(word, text_before):
            return "knownMixups"
        return None

    def _value(self, item: NearbyWord, typed: str,
               typed_has_mixups: bool) -> float:
        value = item.margin - self.settings.cost_weight * item.cost
        if item.is_known_mixup:
            return value + KNOWN_MIXUP_BONUS
        penalty = OTHER_WORD_PENALTY if typed_has_mixups else 0.0
        if differs_only_in_ending(typed, item.word):
            penalty += INFLECTION_PENALTY
        return value - penalty

    def check(self, text_before: str, word: str, text_after: str):
        """Returns a fix when a nearby word fits the sentence clearly
        better than the one typed."""
        judged = self.judged_against(word, text_before)
        if judged is None:
            return None
        nearby = self.compare(text_before, word, text_after,
                              judged == "knownMixups")
        if not nearby:
            return None
        typed_has_mixups = any(item.is_known_mixup for item in nearby)
        ranked = sorted(
            nearby,
            key=lambda item: self._value(item, word.lower(), typed_has_mixups),
            reverse=True)
        best = ranked[0]
        best_value = self._value(best, word.lower(), typed_has_mixups)
        if best_value < self.settings.threshold:
            return None
        return Correction(
            original=word,
            replacement=match_case(word, best.word),
            confidence=1.0 / (1.0 + math.exp(-best.margin)),
            used_context=True)


# ------------------------------------------------------------- index.ts ------
# The check() orchestration (word loop, code ranges, NOT_PROSE guard,
# progressive corrected context) lives in TinyfixChecker in
# app/transcription/tinyfix.py, which drives Fixer + WrongWordChecker
# above exactly as index.ts does.
