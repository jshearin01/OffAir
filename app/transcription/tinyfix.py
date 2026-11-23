"""tinyfix stage for transcription cleanup (paperweight/tinyfix, MIT).

The engine itself lives in :mod:`app.transcription.tinyfix_engine`
(a faithful port of the TypeScript source). This module owns the
parts that touch the machine: downloading the pinned model files
from Hugging Face, verifying their hashes, running them with
onnxruntime, and applying fixes to a session's cleaned segments.

Placement in the cleanup pipeline: AFTER the deterministic rules
and AFTER the names dictionary. Names win — every dictionary word
is fed to the checker's ignore list, so a known name or product
term is never "fixed".

The verbatim ``transcript.json`` is never touched; fixes land only
in the derived ``transcript-clean`` artifacts, like every other
cleanup layer.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.request
from pathlib import Path

from app.transcription.tinyfix_engine import (
    SPELLING_MIN_CONFIDENCE,
    WITH_MODEL_WEIGHTS,
    WRONG_WORD_MODES,
    Correction,
    Fixer,
    WrongWordChecker,
    parse_word_list,
)

logger = logging.getLogger(__name__)

#: Hugging Face revision the package pins per version, so behaviour
#: is reproducible. Recorded here and in docs/notes/tinyfix-2026-11-21.md.
MODEL_REVISION = "8efdf5cd91b32584f4077aa0e8ecaf48f7651bef"
_MODEL_REPO = "paperweight/tinyfix"

#: (filename, sha256) for every file, verified on download and on
#: every load from cache. Measured 2026-11-22 from the pinned
#: revision above.
MODEL_FILES: tuple[tuple[str, str], ...] = (
    ("wrong-word.onnx",
     "35d76c0f2cc1fbb5450188a749f4c10b0d569d8806bb939bd66fa926561b47e9"),
    ("wrong-word-vocab.txt",
     "14ddd048052cf547d570c5c75b18843527916f892d2650fd0c3009b783b976a1"),
    ("wrong-word-confusions.txt",
     "9924ff7ed577595ae1b632adbbf3324779a6857d3eb102715e63abf5981d06d3"),
    ("words.txt",
     "f84bfae717ff3a4a3b90c824ed06ea08e4b2ed2746f7bba63d4e52f2c8bf85c3"),
)

#: Model filenames on disk, as the package expects them.
MODEL_FILE = "wrong-word.onnx"
VOCAB_FILE = "wrong-word-vocab.txt"
CONFUSIONS_FILE = "wrong-word-confusions.txt"
WORDS_FILE = "words.txt"

#: Window of surrounding text the context check sees, in
#: characters — the same 600-before / 200-after window index.ts
#: uses when it slices the text around each word.
CONTEXT_BEFORE = 600
CONTEXT_AFTER = 200

_WORD_RE = re.compile(r"[A-Za-z']+")
_NOT_PROSE = re.compile(r"[@/\\_#\d]|\.\w")
_FENCED = re.compile(r"```[\s\S]*?(?:```|$)", re.MULTILINE)
_INLINE_CODE = re.compile(r"`[^`\n]*`")


def _models_root() -> Path:
    """Cache root, following the sherpa/Whisper pattern."""
    return Path.home() / ".offair" / "models" / "tinyfix" / MODEL_REVISION


def model_dir() -> Path:
    return _models_root()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TinyfixUnavailableError(RuntimeError):
    """Raised when the model cannot be loaded (no onnxruntime, no
    network for the first download, or a hash mismatch). Callers
    degrade to deterministic-only cleanup; this is never fatal."""


def _download_file(url: str, dest: Path, expected_sha256: str,
                   status_callback=None) -> None:
    """Downloads one file with the certifi CA bundle (sherpa
    pattern), verifying its hash before it is trusted."""

    def _report(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    request = urllib.request.Request(url, headers={"User-Agent": "OffAir"})
    try:
        import ssl

        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 — no certifi, or no CA bundle:
        # fall back to the default context; the hash check below
        # is what actually trusts the file.
        context = None
    try:
        with urllib.request.urlopen(request, context=context,
                                    timeout=120) as response, open(
                                        dest, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                if total:
                    _report(f"Downloading tinyfix model… "
                            f"{received * 100 // total}%")
                else:
                    _report("Downloading tinyfix model…")
    except Exception as exc:  # noqa: BLE001 — any failure degrades
        dest.unlink(missing_ok=True)
        raise TinyfixUnavailableError(
            f"tinyfix model download failed: {exc}") from exc
    if _sha256(dest) != expected_sha256:
        dest.unlink(missing_ok=True)
        raise TinyfixUnavailableError(
            f"tinyfix model file {dest.name} failed its hash check")


def ensure_model_files(status_callback=None) -> Path:
    """Returns the directory holding the verified model files,
    downloading them on first use. Raises
    :class:`TinyfixUnavailableError` when the download fails or a
    cached file no longer matches its pinned hash."""
    target = _models_root()
    target.mkdir(parents=True, exist_ok=True)
    for filename, expected in MODEL_FILES:
        path = target / filename
        if path.exists() and _sha256(path) == expected:
            continue
        if status_callback is not None:
            status_callback(f"Downloading tinyfix model ({filename})…")
        else:
            logger.info("Downloading tinyfix model file %s", filename)
        url = (f"https://huggingface.co/{_MODEL_REPO}/resolve/"
               f"{MODEL_REVISION}/{filename}")
        _download_file(url, path, expected, status_callback)
    return target


def _load_text(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


class OnnxModel:
    """Thin onnxruntime wrapper with the exact I/O the engine
    expects: int64 [batch,16] / [batch,4] / [batch,9] in, float32
    [batch,9] scores out."""

    def __init__(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise TinyfixUnavailableError(
                "onnxruntime is not installed; tinyfix is unavailable"
            ) from exc
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])

    def run(self, *, left, right, candidates):
        import numpy as np

        outputs = self._session.run(
            None,
            {
                "left": np.asarray(left, dtype=np.int64),
                "right": np.asarray(right, dtype=np.int64),
                "candidates": np.asarray(candidates, dtype=np.int64),
            },
        )
        return [float(value) for value in outputs[0].reshape(-1)]


class TinyfixChecker:
    """Drives :class:`Fixer` + :class:`WrongWordChecker` over a
    whole text, exactly as ``check()`` in index.ts does: word by
    word, left to right, with later words judged against the
    corrected text so far, and fixes never overlapping."""

    def __init__(self, directory: Path, *, mode: str = "careful",
                 model=None) -> None:
        directory = Path(directory)
        vocab = _load_text(directory / VOCAB_FILE)
        confusions = _load_text(directory / CONFUSIONS_FILE)
        words = parse_word_list(
            (directory / WORDS_FILE).read_text(encoding="utf-8"))
        if model is None:
            model = OnnxModel(directory / MODEL_FILE)
        wrong_word = WrongWordChecker(
            model, vocab, confusions,
            settings=WRONG_WORD_MODES[mode])
        # The package runs the spelling fixer with the model scorer
        # at SPELLING_MIN_CONFIDENCE when a scorer is present: the
        # scorer helps choose between candidates, and the lower bar
        # applies because the model has already weighed in.
        weights = WITH_MODEL_WEIGHTS
        self._fixer = Fixer(words, scorer=wrong_word, weights=weights)
        self._fixer._weights.min_confidence = SPELLING_MIN_CONFIDENCE
        self._wrong_word = wrong_word
        self._ignored: set[str] = set()
        self._settings_mode = mode

    # -- ignore list ---------------------------------------------------

    def ignore(self, word: str) -> None:
        """Never fix this word (names-dictionary entries land here),
        and treat it as a known word from now on."""
        lower = word.lower()
        if lower:
            self._ignored.add(lower)
            self._fixer.learn(lower)

    def ignore_words(self, words) -> None:
        for word in words:
            for piece in _WORD_RE.findall(word):
                self.ignore(piece)

    # -- the check loop -------------------------------------------------

    def _is_ignored(self, word: str) -> bool:
        return word.lower() in self._ignored

    def fix_text(self, text: str):
        """Returns ``(fixed_text, corrections)`` with the original
        casing preserved in each correction."""
        fixed, corrections, _spans = self._fix_with_offsets(text)
        return fixed, corrections

    def _fix_with_offsets(self, text: str):
        """The index.ts loop: word by word, left to right, later
        words judged against the corrected text so far, fixes
        never overlapping.

        Returns ``(fixed_text, corrections, spans)`` where spans
        are ``(start, end, replacement)`` in the ORIGINAL text's
        coordinates, so callers can map fixes back exactly."""
        fixed_parts: list[str] = []
        corrections: list[Correction] = []
        spans: list[tuple[int, int, str]] = []
        fixed_so_far = ""
        position = 0
        length = len(text)

        # Code ranges are skipped, as in the package (transcripts
        # rarely contain them, but the rule is part of the engine).
        skip_ranges = [(m.start(), m.end())
                       for m in _FENCED.finditer(text)]
        skip_ranges += [(m.start(), m.end())
                        for m in _INLINE_CODE.finditer(text)]

        def in_skip_range(start: int, end: int) -> bool:
            # Overlap, as in the package: a word touching a code
            # range is skipped, not just one contained in it.
            return any(start < e and end > s for s, e in skip_ranges)

        while position < length:
            match = _WORD_RE.search(text, position)
            if match is None:
                fixed_parts.append(text[position:])
                fixed_so_far += text[position:]
                break
            start, end = match.span()
            fixed_parts.append(text[position:start])
            fixed_so_far += text[position:start]
            word = match.group(0)
            if self._is_ignored(word) or in_skip_range(start, end):
                fixed_parts.append(word)
                fixed_so_far += word
                position = end
                continue
            token_around = text[max(0, start - 1):end + 1]
            if _NOT_PROSE.search(token_around):
                fixed_parts.append(word)
                fixed_so_far += word
                position = end
                continue
            # Fixed rules first (confidence 1), then spelling, then
            # context — the package tries the fixer, and only when
            # it returns nothing asks the context checker. The
            # after-context stops at the line end, as in the
            # package: context never crosses a line break.
            before = fixed_so_far[-CONTEXT_BEFORE:]
            line_end = text.find("\n", end)
            after_end = (len(text) if line_end == -1 else line_end)
            after = text[end:min(after_end, end + CONTEXT_AFTER)]
            correction = self._fixer.correct(before, word)
            if correction is None:
                correction = self._wrong_word.check(before, word, after)
            if correction is not None and \
                    correction.replacement == word:
                # A no-op fix is not a fix (the package skips it).
                correction = None
            if correction is not None:
                fixed_parts.append(correction.replacement)
                fixed_so_far += correction.replacement
                corrections.append(correction)
                spans.append((start, end, correction.replacement))
            else:
                fixed_parts.append(word)
                fixed_so_far += word
            position = end
        return "".join(fixed_parts), corrections, spans

    def fix_segments(self, segments: list[dict]):
        """Applies fixes to cleaned segments with cross-segment
        context: the 16-before/4-after window never resets at a
        segment boundary. Returns ``(new_segments, corrections)``
        with timestamps and every other field untouched."""
        joined = " ".join(seg.get("text", "") for seg in segments)
        fixed, corrections, spans = self._fix_with_offsets(joined)
        if not corrections:
            return segments, []
        # Split the fixed text back into segments on the same
        # single-space joins. A fix changes its segment's length,
        # so each segment's slice end moves by the net length
        # change of the fixes inside that segment (spans are in
        # original coordinates; fixes never cross the separator
        # space because a fix is always one word).
        new_segments = [dict(seg) for seg in segments]
        drift = 0
        seg_start = 0
        for index, seg in enumerate(segments):
            seg_text = seg.get("text", "")
            seg_end = seg_start + len(seg_text)
            delta = sum(len(rep) - (end - start)
                        for start, end, rep in spans
                        if seg_start <= start and end <= seg_end)
            piece = fixed[seg_start + drift:seg_end + drift + delta]
            new_segments[index]["text"] = piece
            drift += delta
            seg_start = seg_end + 1
        return new_segments, corrections


def build_checker(*, mode: str = "careful", model_dir_override=None,
                  status_callback=None, model=None) -> TinyfixChecker:
    """Builds a checker, downloading the model on first use.
    Raises :class:`TinyfixUnavailableError` on any failure,
    including missing or unreadable cached files."""
    directory = (Path(model_dir_override) if model_dir_override
                 else ensure_model_files(status_callback))
    try:
        return TinyfixChecker(directory, mode=mode, model=model)
    except OSError as exc:
        raise TinyfixUnavailableError(
            f"tinyfix model files unreadable in {directory}: {exc}"
        ) from exc
