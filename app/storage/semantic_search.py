"""Semantic transcript search — vector embeddings, fully local.

Search in OffAir used to be substring-only (see
:mod:`app.storage.browser`): typing ``"budget"`` found the line
about the budget, but ``"money"`` found nothing and ``"12000"``
missed Whisper's spelled-out ``"twelve thousand"``. This module
adds a second, meaning-based search on top of it.

How it works
------------
1. **Chunking** (:func:`build_chunks`): a session's timestamped
   transcript segments are merged into chunks of up to
   ``SEMANTIC_CHUNK_MAX_WORDS`` words (default 40 — measured,
   see below), with a one-segment overlap so a thought split
   across a boundary is still found whole. Each chunk keeps its
   session id, start/end times and text (speaker labels
   included, e.g. ``"Priya: ..."``).
2. **Embedding** (:class:`MiniLMEmbedder`): each chunk is embedded
   with ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim),
   exported to ONNX and run with **onnxruntime on CPU** — the
   same runtime the ONNX ECAPA speaker backend already uses.
   No torch, no transformers, no cloud, no Hugging Face token.
   The int8 model (``onnx/model_quint8_avx2.onnx``, ~22 MB) and
   the WordPiece vocabulary are downloaded once, on first
   semantic use, from public Hugging Face into
   ``~/.offair/models/all-MiniLM-L6-v2/``. Tokenization is a
   small pure-Python BERT WordPiece implementation in this file
   (no extra dependency); embeddings are mean-pooled with the
   attention mask and L2-normalised, so cosine similarity is a
   dot product.
3. **Storage** (:class:`SemanticSearchIndex`): a SQLite index
   (``search_index.db`` in the recordings root, next to
   ``speakers.db``) holding one row per chunk — session id,
   chunk id, start/end seconds, text, embedding BLOB (float32).
   Indexing is incremental: :meth:`SemanticSearchIndex.index_session`
   hashes the session's ``transcript.json`` and skips sessions
   whose transcript has not changed since they were indexed.
4. **Search**: :meth:`SemanticSearchIndex.semantic_search` embeds
   the query and ranks chunks by cosine similarity (default
   threshold 0.30, ``top_k`` 10). :func:`hybrid_search` in
   :mod:`app.storage.browser` merges those with the substring
   hits — exact matches are boosted so they rank first — and
   de-duplicates by (session, chunk).

Honest limits
-------------
* Whisper wording errors still hurt matching: if the transcript
  says "marked down" instead of "markdown", the embedding sees
  the wrong word too.
* The first semantic search needs the internet once (model
  download, ~22 MB); after that everything is offline. If the
  download fails, :class:`ModelNotAvailableError` is raised with
  a guided message — the UI falls back to substring search.
* Single-word queries are noisy: a genuine weak match
  ("money" -> the budget discussion) scores ~0.20, but an
  unrelated word can occasionally score ~0.22. Multi-word
  queries ("how much money", "send the timeline") score
  0.28–0.60 and rank correctly. The 0.15 default threshold is
  a measured trade-off — see docs/SEMANTIC_SEARCH.md.
* The index must be rebuilt after a re-transcription (the
  transcript hash changes, so the session is re-indexed
  automatically on the next semantic search; use
  :meth:`SemanticSearchIndex.rebuild_index` to force a full
  rebuild).
* All chunk embeddings are loaded into memory for a search
  (~1.5 KB per chunk). A personal library of even a thousand
  hour-long meetings is ~100 MB of embeddings at most — fine on
  the 8–16 GB target laptop, but this is not a vector database
  and is not meant for millions of chunks.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.storage.sessions import METADATA_FILE, list_sessions
from app.transcription.engine import Segment
from app.transcription.exports import load_transcript_json

logger = logging.getLogger("offair.storage.semantic_search")

# --- Model ------------------------------------------------------------------
SEMANTIC_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MODEL_DIM = 384
SEMANTIC_MAX_SEQ_LEN = 256
# ONNX candidates, in preference order. The int8 AVX2 model is ~22 MB
# and is the right size/speed trade-off for a CPU-only laptop; the
# fp32 model (~90 MB) is the fallback if the quantized file ever
# disappears from the repo. (Verified against the HF file list
# Nov 8 2026.)
SEMANTIC_MODEL_FILES: tuple[str, ...] = (
    "onnx/model_quint8_avx2.onnx",
    "onnx/model.onnx",
)
SEMANTIC_VOCAB_FILE = "vocab.txt"
SEMANTIC_MODEL_DIR_NAME = "all-MiniLM-L6-v2"
SEMANTIC_INDEX_FILE = "search_index.db"

# --- Chunking / search defaults ----------------------------------------------
# Measured on the 61 s fixture (Nov 8 2026) — these defaults are
# NOT the first guesses, they are what the numbers forced:
# * Chunks are ~40 words, not 150. Mean-pooled embeddings dilute
#   fast: at 150 words the query "money" scored 0.073 against
#   the chunk holding the budget sentence (useless); at 40 words
#   it scores 0.199. Multi-word queries score 0.28–0.60.
# * Threshold is 0.15, not 0.30. Genuine weak matches ("money"
#   -> budget) score 0.17–0.20; unrelated words mostly score
#   < 0.15. Single-word queries are noisy by nature — an
#   unrelated word can occasionally reach ~0.22 — so results
#   are always ranked, and the threshold is a trade-off, not a
#   guarantee. See docs/SEMANTIC_SEARCH.md for the full table.
SEMANTIC_CHUNK_MAX_WORDS = 40
SEMANTIC_CHUNK_OVERLAP_SEGMENTS = 1
SEMANTIC_DEFAULT_THRESHOLD = 0.15
SEMANTIC_DEFAULT_TOP_K = 10
# Exact (substring) matches in hybrid search get this added to their
# cosine score, so they rank above purely semantic hits.
HYBRID_TEXT_BOOST = 1.0
# How many chunks to embed in one ONNX call.
EMBED_BATCH_SIZE = 32


def default_model_dir() -> Path:
    """Where the MiniLM model lives: ``~/.offair/models/...``."""
    return (Path.home() / ".offair" / "models"
            / SEMANTIC_MODEL_DIR_NAME)


class ModelDownloadError(Exception):
    """The embedding model could not be downloaded."""


class ModelNotAvailableError(Exception):
    """Semantic search cannot run (model missing / not downloadable).

    The message is written for the user: it says what is missing
    and what to do, and the UI shows it instead of a traceback.
    """


# --- WordPiece tokenizer (pure Python, BERT uncased) -------------------------
# all-MiniLM-L6-v2 is a BERT-style uncased model. Its tokenizer is
# WordPiece over a ~30k vocabulary (vocab.txt in the model repo).
# This is a small, dependency-free re-implementation of the
# tokenization the model was trained with: lowercase, strip
# accents, split on whitespace/punctuation, then greedy
# longest-match WordPiece. It matches the Hugging Face fast
# tokenizer closely enough for English meeting transcripts that
# ranking is unaffected; it is not a general-purpose tokenizer.
_PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _basic_tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, split on whitespace/punctuation."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text
                   if unicodedata.category(c) != "Mn")
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for char in text:
        if char.isspace():
            flush()
        elif char in _PUNCTUATION:
            flush()
            tokens.append(char)
        else:
            current.append(char)
    flush()
    return tokens


class WordPieceTokenizer:
    """Minimal BERT WordPiece tokenizer backed by a vocab.txt."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.unk_id = vocab.get("[UNK]", 100)
        self.cls_id = vocab.get("[CLS]", 101)
        self.sep_id = vocab.get("[SEP]", 102)
        self.pad_id = vocab.get("[PAD]", 0)

    @classmethod
    def from_vocab_file(cls, path: str | Path) -> "WordPieceTokenizer":
        vocab: dict[str, int] = {}
        with open(path, encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                token = line.rstrip("\n")
                if token:
                    vocab[token] = index
        return cls(vocab)

    def _wordpiece(self, token: str) -> list[str]:
        """Greedy longest-match WordPiece for one basic token."""
        if token in self.vocab:
            return [token]
        pieces: list[str] = []
        start = 0
        while start < len(token):
            end = len(token)
            found = None
            while end > start:
                piece = token[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    found = piece
                    break
                end -= 1
            if found is None:
                return ["[UNK]"]
            pieces.append(found)
            start = end
        return pieces

    def encode(self, text: str,
               max_len: int = SEMANTIC_MAX_SEQ_LEN) -> tuple[list[int],
                                                            list[int]]:
        """Text -> (input_ids, attention_mask), with [CLS]/[SEP].

        Truncates to ``max_len`` tokens (a 150-word chunk is ~200
        tokens, so truncation is rare).
        """
        pieces: list[str] = []
        for token in _basic_tokenize(text):
            pieces.extend(self._wordpiece(token))
        pieces = pieces[:max_len - 2]
        ids = [self.cls_id]
        ids.extend(self.vocab.get(p, self.unk_id) for p in pieces)
        ids.append(self.sep_id)
        mask = [1] * len(ids)
        return ids, mask


def _mean_pool(last_hidden_state: np.ndarray,
               attention_mask: np.ndarray) -> np.ndarray:
    """Mean-pool token vectors with the attention mask, L2-normalise.

    ``last_hidden_state`` is (batch, seq, dim); ``attention_mask``
    is (batch, seq). Returns (batch, dim) float32, unit length.
    """
    mask = attention_mask.astype(np.float32)[..., None]
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = summed / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    return (pooled / norms).astype(np.float32)


# --- MiniLM embedder ---------------------------------------------------------
class MiniLMEmbedder:
    """all-MiniLM-L6-v2 via onnxruntime (CPU), lazily loaded.

    The model + vocabulary are downloaded on first use from
    public Hugging Face (no token / login) into ``model_dir``.
    Constructing this class downloads nothing and imports
    nothing heavy; :meth:`embed` triggers the lazy load.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None \
            else default_model_dir()
        self._session = None
        self._tokenizer: WordPieceTokenizer | None = None
        self._model_path: Path | None = None

    # ------------------------------------------------------------ download ---
    def _download(self, remote_path: str, local_path: Path,
                  progress=None) -> None:
        """Download one file from the public HF repo (atomic).

        Raises ModelDownloadError on any failure (offline, 404,
        timeout) — the caller decides whether to try the next
        candidate file.
        """
        url = (f"https://huggingface.co/{SEMANTIC_MODEL_REPO}"
               f"/resolve/main/{remote_path}")
        tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        request = urllib.request.Request(
            url, headers={"User-Agent": "OffAir/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as resp, \
                    open(tmp_path, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress is not None and total:
                        progress(done, total)
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError) as exc:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ModelDownloadError(
                f"Could not download {remote_path} from Hugging Face "
                f"({exc}). Check the internet connection and retry — "
                f"the model is only needed once; after that semantic "
                f"search works fully offline.") from exc
        if tmp_path.stat().st_size < 1024:
            tmp_path.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"Downloaded {remote_path} looks truncated "
                f"(< 1 KB) — retry the download.")
        os.replace(tmp_path, local_path)

    def ensure_downloaded(self, progress=None) -> Path:
        """Make sure model + vocab exist locally; download if not.

        Returns the model file path. The int8 model is tried
        first, then the fp32 model.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = self.model_dir / SEMANTIC_VOCAB_FILE
        if not vocab_path.exists():
            self._download(SEMANTIC_VOCAB_FILE, vocab_path, progress)
        for candidate in SEMANTIC_MODEL_FILES:
            local = self.model_dir / Path(candidate).name
            if local.exists() and local.stat().st_size > 1024 * 1024:
                return local
        errors: list[str] = []
        for candidate in SEMANTIC_MODEL_FILES:
            local = self.model_dir / Path(candidate).name
            try:
                self._download(candidate, local, progress)
                return local
            except ModelDownloadError as exc:
                errors.append(str(exc))
        raise ModelDownloadError(
            "Could not download the MiniLM embedding model from "
            "Hugging Face. " + " ".join(errors))

    # ------------------------------------------------------------ loading ----
    def _load(self, progress=None) -> None:
        """Lazy load: download if needed, then build the session."""
        if self._session is not None:
            return
        try:
            self._model_path = self.ensure_downloaded(progress)
        except ModelDownloadError as exc:
            raise ModelNotAvailableError(str(exc)) from exc
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelNotAvailableError(
                "onnxruntime is not installed — run "
                "`pip install -r requirements.txt`.") from exc
        vocab_path = self.model_dir / SEMANTIC_VOCAB_FILE
        try:
            self._tokenizer = WordPieceTokenizer.from_vocab_file(
                vocab_path)
            options = ort.SessionOptions()
            self._session = ort.InferenceSession(
                str(self._model_path), options,
                providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001 — surfaced as guidance
            raise ModelNotAvailableError(
                f"Could not load the MiniLM model from "
                f"{self.model_dir} ({exc}). Delete that folder and "
                f"retry — it will be re-downloaded.") from exc

    # ------------------------------------------------------------ embed ------
    def embed(self, texts: list[str],
              progress=None) -> np.ndarray:
        """Embed ``texts`` -> (n, 384) float32, L2-normalised rows."""
        if not texts:
            return np.zeros((0, SEMANTIC_MODEL_DIM), dtype=np.float32)
        self._load(progress)
        assert self._tokenizer is not None and self._session is not None
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            encoded = [self._tokenizer.encode(t) for t in batch]
            max_len = max(len(ids) for ids, _ in encoded)
            input_ids = np.zeros((len(batch), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(batch), max_len),
                                      dtype=np.int64)
            token_type_ids = np.zeros((len(batch), max_len),
                                      dtype=np.int64)
            for i, (ids, mask) in enumerate(encoded):
                input_ids[i, :len(ids)] = ids
                attention_mask[i, :len(mask)] = mask
            feed: dict[str, np.ndarray] = {"input_ids": input_ids,
                                           "attention_mask": attention_mask}
            input_names = {inp.name
                           for inp in self._session.get_inputs()}
            if "token_type_ids" in input_names:
                feed["token_type_ids"] = token_type_ids
            result = self._session.run(None, feed)
            output_names = [out.name
                            for out in self._session.get_outputs()]
            hidden = result[output_names.index("last_hidden_state")] \
                if "last_hidden_state" in output_names else result[0]
            outputs.append(_mean_pool(
                np.asarray(hidden, dtype=np.float32), attention_mask))
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed one string -> (384,) float32, unit length."""
        return self.embed([text])[0]


# --- Chunking ----------------------------------------------------------------
@dataclass
class Chunk:
    """One searchable piece of a transcript."""

    session_id: str
    chunk_id: int
    start_s: float
    end_s: float
    text: str


def _segment_text(segment: Segment) -> str:
    """One segment's text, with its speaker label when present."""
    text = (segment.text or "").strip()
    if not text:
        return ""
    if getattr(segment, "speaker", None):
        return f"{segment.speaker}: {text}"
    return text


def build_chunks(session_id: str, segments: list[Segment], *,
                 max_words: int = SEMANTIC_CHUNK_MAX_WORDS,
                 overlap_segments: int = SEMANTIC_CHUNK_OVERLAP_SEGMENTS
                 ) -> list[Chunk]:
    """Merge timestamped segments into ~``max_words``-word chunks.

    Segments are accumulated until the chunk reaches ``max_words``;
    the next chunk then starts ``overlap_segments`` segments back,
    so a thought split across a boundary is found whole in at
    least one chunk. A single segment longer than ``max_words``
    becomes a chunk on its own. Empty segments are skipped.
    """
    texts = [(_segment_text(s), s) for s in segments]
    texts = [(t, s) for t, s in texts if t]
    chunks: list[Chunk] = []
    start_idx = 0
    chunk_id = 0
    while start_idx < len(texts):
        words = 0
        end_idx = start_idx
        while end_idx < len(texts):
            seg_words = len(texts[end_idx][0].split())
            if words > 0 and words + seg_words > max_words:
                break
            words += seg_words
            end_idx += 1
        if end_idx == start_idx:  # one segment alone exceeds the cap
            end_idx = start_idx + 1
        piece = texts[start_idx:end_idx]
        chunks.append(Chunk(
            session_id=session_id,
            chunk_id=chunk_id,
            start_s=float(piece[0][1].start),
            end_s=float(piece[-1][1].end),
            text=" ".join(t for t, _ in piece)))
        chunk_id += 1
        if end_idx >= len(texts):
            break
        # Next chunk starts `overlap_segments` segments before the
        # end of this one (at least one segment of progress, so
        # the loop always terminates).
        start_idx = max(end_idx - overlap_segments, start_idx + 1)
    return chunks


# --- Search results ----------------------------------------------------------
@dataclass
class SemanticHit:
    """One ranked chunk hit from a semantic / hybrid search."""

    session_id: str
    chunk_id: int
    start_s: float
    end_s: float
    text: str
    score: float
    match_type: str = "semantic"  # "semantic" | "text" | "both"

    @property
    def snippet(self) -> str:
        """Chunk text, truncated for display (~300 chars)."""
        text = self.text.strip()
        return text if len(text) <= 300 else text[:297].rstrip() + "…"

    def display(self) -> str:
        """One-line hit for the browser list."""
        from app.transcription.exports import format_txt_timestamp

        span = (f"{format_txt_timestamp(self.start_s)}–"
                f"{format_txt_timestamp(self.end_s)}")
        return (f"[{self.score:.2f} {self.match_type}] "
                f"{self.session_id} {span} {self.snippet}")


# --- SQLite index ------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    session_id TEXT NOT NULL,
    chunk_id INTEGER NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (session_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS index_state (
    session_id TEXT PRIMARY KEY,
    transcript_hash TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);
"""


class SemanticSearchIndex:
    """Chunk embeddings for every session, in ``search_index.db``.

    The index lives in the recordings root (next to
    ``speakers.db``). ``embedder`` is injectable — tests pass a
    deterministic fake; production passes nothing and the real
    MiniLM model is loaded lazily on first use.
    """

    def __init__(self, output_root: str | Path,
                 embedder: MiniLMEmbedder | None = None,
                 model_dir: str | Path | None = None,
                 db_path: str | Path | None = None) -> None:
        self.output_root = Path(output_root)
        self._embedder = embedder
        self._model_dir = model_dir
        self.db_path = (Path(db_path) if db_path is not None
                        else self.output_root / SEMANTIC_INDEX_FILE)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------ internals --
    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        return self._conn

    def _embedder_or_default(self) -> MiniLMEmbedder:
        if self._embedder is None:
            self._embedder = MiniLMEmbedder(model_dir=self._model_dir)
        return self._embedder

    @staticmethod
    def _transcript_hash(session_dir: Path) -> str | None:
        """SHA-256 of transcript.json (or .txt), or None if absent."""
        for name in ("transcript.json", "transcript.txt"):
            path = session_dir / name
            if path.exists():
                try:
                    return hashlib.sha256(
                        path.read_bytes()).hexdigest()
                except OSError:
                    return None
        return None

    def _load_segments(self, session_dir: Path) -> list[Segment] | None:
        """Segments for a session, or None if it has no transcript."""
        path = session_dir / "transcript.json"
        if not path.exists():
            return None
        try:
            return load_transcript_json(path)
        except Exception:  # noqa: BLE001 — a bad transcript is skipped
            logger.warning("Could not read transcript in %s",
                           session_dir, exc_info=True)
            return None

    def remove_session(self, session_id: str) -> int:
        """Delete every index row for ``session_id`` (chunks + state).

        Used by session delete: a deleted session must leave no
        search entries behind. Returns the rows removed. A missing
        index file is a no-op (0) — no empty database is created
        just to delete from it.
        """
        if not self.db_path.exists():
            return 0
        conn = self._connect()
        chunks = conn.execute(
            "DELETE FROM chunks WHERE session_id = ?",
            (session_id,)).rowcount
        state = conn.execute(
            "DELETE FROM index_state WHERE session_id = ?",
            (session_id,)).rowcount
        conn.commit()
        return chunks + state

    # ------------------------------------------------------------ indexing ---
    def index_session(self, session_dir: str | Path) -> int:
        """Index one session. Returns chunks indexed (0 = skipped).

        Sessions with no transcript, or whose transcript hash
        matches the last index run, are skipped without touching
        the model — a search over an up-to-date library never
        downloads or loads anything.
        """
        session_dir = Path(session_dir)
        session_id = session_dir.name
        segments = self._load_segments(session_dir)
        if not segments:
            return 0
        transcript_hash = self._transcript_hash(session_dir)
        conn = self._connect()
        row = conn.execute(
            "SELECT transcript_hash FROM index_state "
            "WHERE session_id = ?", (session_id,)).fetchone()
        if (row is not None and transcript_hash is not None
                and row[0] == transcript_hash):
            return 0
        chunks = build_chunks(session_id, segments)
        if not chunks:
            return 0
        # The model is only loaded when there is actually work.
        embeddings = self._embedder_or_default().embed(
            [c.text for c in chunks])
        conn.execute("DELETE FROM chunks WHERE session_id = ?",
                     (session_id,))
        for chunk, vector in zip(chunks, embeddings):
            conn.execute(
                "INSERT INTO chunks "
                "(session_id, chunk_id, start_s, end_s, text, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (chunk.session_id, chunk.chunk_id, chunk.start_s,
                 chunk.end_s, chunk.text,
                 np.asarray(vector, dtype=np.float32).tobytes()))
        conn.execute(
            "INSERT OR REPLACE INTO index_state "
            "(session_id, transcript_hash, indexed_at) "
            "VALUES (?, ?, ?)",
            (session_id, transcript_hash or "",
             datetime.now(timezone.utc).isoformat(timespec="seconds")))
        conn.commit()
        logger.info("Indexed %s: %d chunks", session_id, len(chunks))
        return len(chunks)

    def ensure_indexed(self) -> dict[str, int]:
        """Index every session that is new or changed.

        Returns {session_id: chunks indexed} (0 for skipped).
        """
        counts: dict[str, int] = {}
        for session_dir in list_sessions(self.output_root):
            counts[session_dir.name] = self.index_session(session_dir)
        return counts

    def rebuild_index(self) -> dict[str, int]:
        """Drop the whole index and rebuild it from scratch."""
        conn = self._connect()
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("DROP TABLE IF EXISTS index_state")
        conn.executescript(_SCHEMA)
        conn.commit()
        return self.ensure_indexed()

    def indexed_sessions(self) -> list[str]:
        """Session ids currently in the index."""
        conn = self._connect()
        return [r[0] for r in conn.execute(
            "SELECT session_id FROM index_state ORDER BY session_id")]

    def chunk_count(self) -> int:
        """Total chunks in the index."""
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def find_text_chunk(self, session_id: str,
                        query: str) -> SemanticHit | None:
        """First chunk of ``session_id`` containing ``query``.

        Used by hybrid search for sessions that matched by
        substring but had no semantic hit. Returns None when the
        session has no indexed chunk containing the query.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT chunk_id, start_s, end_s, text FROM chunks "
            "WHERE session_id = ? AND lower(text) LIKE ? "
            "ORDER BY chunk_id LIMIT 1",
            (session_id, f"%{query.lower()}%")).fetchone()
        if row is None:
            return None
        return SemanticHit(session_id=session_id, chunk_id=row[0],
                           start_s=row[1], end_s=row[2], text=row[3],
                           score=HYBRID_TEXT_BOOST, match_type="text")

    # ------------------------------------------------------------ searching --
    def semantic_search(self, query: str, *, top_k: int = SEMANTIC_DEFAULT_TOP_K,
                        threshold: float = SEMANTIC_DEFAULT_THRESHOLD,
                        ) -> list[SemanticHit]:
        """Rank chunks by cosine similarity to ``query``.

        The index is brought up to date first (incremental — only
        new or changed sessions are embedded). An empty query
        returns []. Raises ModelNotAvailableError when the model
        cannot be downloaded/loaded.
        """
        query = (query or "").strip()
        if not query:
            return []
        self.ensure_indexed()
        query_vec = self._embedder_or_default().embed_one(query)
        conn = self._connect()
        rows = conn.execute(
            "SELECT session_id, chunk_id, start_s, end_s, text, "
            "embedding FROM chunks").fetchall()
        hits: list[SemanticHit] = []
        for session_id, chunk_id, start_s, end_s, text, blob in rows:
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.shape[0] != query_vec.shape[0]:
                continue  # stale chunk from another embedder — skip
            score = float(np.dot(vector, query_vec))
            if score >= threshold:
                hits.append(SemanticHit(
                    session_id=session_id, chunk_id=chunk_id,
                    start_s=start_s, end_s=end_s, text=text,
                    score=score, match_type="semantic"))
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    def close(self) -> None:
        """Close the SQLite connection (safe to call repeatedly)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


__all__ = [
    "Chunk",
    "MiniLMEmbedder",
    "ModelDownloadError",
    "ModelNotAvailableError",
    "SemanticHit",
    "SemanticSearchIndex",
    "WordPieceTokenizer",
    "build_chunks",
    "default_model_dir",
    "SEMANTIC_CHUNK_MAX_WORDS",
    "SEMANTIC_DEFAULT_THRESHOLD",
    "SEMANTIC_DEFAULT_TOP_K",
    "SEMANTIC_INDEX_FILE",
    "SEMANTIC_MODEL_DIM",
    "SEMANTIC_MODEL_REPO",
]
