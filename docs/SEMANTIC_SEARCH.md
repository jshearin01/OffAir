# Semantic transcript search

Added Nov 8 2026. Search defaults to Hybrid (exact substring
hits first, then semantic matches); Text and Semantic are
selectable modes in the Library view and in
`search_sessions(..., mode="text" | "semantic" | "hybrid")`.

## How it works

```
transcript segments  →  chunks (~40 words, 1-segment overlap)
                     →  MiniLM-L6-v2 ONNX embeddings (384-dim, L2-normalised)
                     →  search_index.db (SQLite, in the recordings root)
query                →  same embedding → cosine similarity → ranked hits
```

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, ONNX int8
  (`onnx/model_quint8_avx2.onnx`, 23.0 MB) + `vocab.txt`
  (232 KB), downloaded once from public Hugging Face (no token /
  login) into `~/.offair/models/all-MiniLM-L6-v2/`. Run with
  onnxruntime on CPU. No torch, no transformers, no cloud.
- **Tokenizer:** a small pure-Python BERT WordPiece implementation
  (`WordPieceTokenizer` in `app/storage/semantic_search.py`) —
  lowercase, strip accents, split on whitespace/punctuation, greedy
  longest-match WordPiece. No extra dependency.
- **Pooling:** mean-pooling of `last_hidden_state` with the
  attention mask, then L2-normalisation (per the model's
  `1_Pooling/config.json`), so cosine similarity is a dot product.
- **Chunking:** segments are merged until a chunk reaches ~40
  words; the next chunk starts one segment back (overlap), so a
  thought split across a boundary is found whole. A single segment
  longer than the cap becomes a chunk on its own. Speaker labels
  are kept in the chunk text (`"Priya: ..."`).
- **Storage:** `search_index.db` in the recordings root (next to
  `speakers.db`). Tables: `chunks(session_id, chunk_id, start_s,
  end_s, text, embedding BLOB)` and `index_state(session_id,
  transcript_hash, indexed_at)`. Indexing is incremental:
  `index_session()` hashes `transcript.json` and skips sessions
  whose transcript has not changed. `rebuild_index()` drops
  everything and re-indexes.
- **Search:** `semantic_search(query, top_k=10, threshold=0.15)`
  returns ranked `SemanticHit`s (session, chunk, start/end,
  text, cosine score, snippet). `hybrid_search()` (in
  `app/storage/browser.py`) merges substring hits with semantic
  hits: exact matches get +1.0 so they rank first, a chunk
  matching both ways appears once as `match_type="both"`, and
  substring-only sessions contribute their first matching chunk.
- **Offline behaviour:** if the model cannot be downloaded or
  loaded, `ModelNotAvailableError` is raised with a guided
  message. The Recording Browser catches it, shows the message,
  and falls back to text search. Text search never needs the
  model at all.

## Measured numbers (61 s fixture, this Linux VM, Nov 8 2026)

Model file 23.0 MB, vocab 232 KB. First-use index build 28.7 s
(incl. ~23 MB download + model load); with the model cached,
indexing 8 chunks takes 0.4 s. Query ~2–7 ms after the first
(67 ms — ONNX warm-up). Peak RSS ~115 MB with the model loaded.
A 1-hour meeting is ~300 chunks at 40 words (~450 KB of
embeddings); even 100 hours of meetings is ~45 MB in memory
during a search.

Top hits on the fixture (8 chunks, 40-word cap):

| query | top hit | score | correct chunk? |
|---|---|---|---|
| `money` | budget / deadline passage | 0.199 | yes |
| `budget` | same | 0.445 | yes |
| `deadline` | same | 0.427 | yes |
| `timeline` | timeline / Monday passage | 0.401 | yes |
| `project budget` | budget passage | 0.422 | yes |
| `how much money` | budget passage | 0.283 | yes |
| `when is the deadline` | deadline passage | 0.470 | yes |
| `send the timeline` | timeline passage | 0.602 | yes |
| `confirm the hardware list` | hardware passage | 0.583 | yes |
| `what did Priya record` | recording passage | 0.489 | yes |
| `meeting notes export` | export passage | 0.587 | yes |

## Why 40-word chunks and a 0.15 threshold (not 150 / 0.30)

The first implementation used 150-word chunks and a 0.30
threshold. It failed the flagship case: at 150 words the query
`"money"` scored **0.073** against the chunk holding the budget
sentence (mean pooling dilutes a single topic across a long
chunk), and even against the budget sentence alone `"money"`
scores only 0.212 — so 0.30 could never work for it. Measured
dilution of `"money"` vs the budget chunk by chunk size:

| chunk cap (words) | "money" score |
|---|---|
| 150 | 0.073 |
| 75 | 0.108 |
| 50 | 0.118 |
| 40 | 0.199 |
| 30 | 0.174 |

40 words was the sweet spot on this fixture (9/10 related
queries found at threshold 0.15, vs 8/10 at 25–30 words).
Unrelated words mostly score < 0.15, so 0.15 is the default
threshold.

## Limitations (honest)

- **Single-word queries are noisy.** A genuine weak match
  (`"money"` → budget, 0.20) and an unrelated word
  (`"lasagna"` → a greeting chunk, 0.22) can score similarly.
  Results are always ranked best-first, and multi-word queries
  (0.28–0.60) are far more reliable than single words. If a
  search returns junk, rephrase with more words.
- **Whisper errors hurt matching.** If the transcript says
  "marked down" instead of "markdown", the embedding sees the wrong word too.
- **Chunking parameters are not part of the index hash.** If
  `SEMANTIC_CHUNK_MAX_WORDS` changes in a future version, run
  `rebuild_index()` once — the transcript hash alone will not
  trigger a re-index.
- **Not a vector database.** All chunk embeddings are loaded
  into memory per search. Fine for a personal library (see
  sizes above); not meant for millions of chunks.
- **First use needs the internet once** (model download). After
  that, fully offline.
- The gated test `test_real_minilm_model_ranks_meaning` only runs
  with `OFFAIR_TEST_REAL_MINILM=1` and skips gracefully when
  the model cannot be downloaded, so the suite stays green
  offline.
