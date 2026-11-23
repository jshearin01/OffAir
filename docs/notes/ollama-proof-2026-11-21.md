# Ollama proof — 21 November 2026

**Status: NOT proven end-to-end.** No real Ollama summary has ever
run — not in this round, and not before it. What exists is the
wiring (code + fake-client tests) plus the probe behaviour verified
below. Do not claim a `qwen3.5:9b` summary ran, or any quality
judgement about one, until the real-machine procedure below has
been followed on a machine with Ollama installed.

## What was checked on this VM (21 Nov 2026)

- No Ollama binary: `which ollama` → not found.
- No Ollama server: `GET http://localhost:11434/api/tags` →
  connection refused.
- Real (unfaked) `OllamaClient` against this VM:
  `is_available()` → `False` (0.05 s), `list_models()` → `[]`,
  `summarize()` on the fixture transcript
  (`tests/fixtures/meeting-script.txt`, 1,228 characters) →
  `OllamaNotAvailableError` with the install/start guidance. The
  probe path behaves as documented: the app says Ollama is not
  running instead of failing obscurely.
- Fake-client tests in `tests/test_exports.py` still pass: the
  client sends the transcript only to localhost, requests
  `num_ctx: 16384` with `"think": false`, and rejects
  non-localhost URLs.

## Why no install was attempted

Installing Ollama and pulling `qwen3.5:9b` (~6.6 GB download, ~8 GB
free RAM needed per the README) was not quick or CPU-feasible here:
this VM has 7.7 GB of RAM total (about 0.5 GB free at check time)
and 4 shared CPUs. A 9B model would not fit in RAM alongside the
rest of the system, and downloading 6.6 GB to prove a path that is
already covered by the probe + fake-client tests would have risked
the VM for no new evidence about the wiring. The install was
deliberately skipped; the procedure below is what proves it on a
real machine.

## Exact real-machine procedure (Windows, the user's PC)

1. Install Ollama from https://ollama.com (the installer also
   starts the server; otherwise run `ollama serve`). Optionally
   set `OLLAMA_NO_CLOUD=1` and `OLLAMA_KEEP_HISTORY=false` so it
   stays fully local and does not store chat history in plaintext
   (see README).
2. Pull the default model: `ollama pull qwen3.5:9b` (~6.6 GB).
   On an 8 GB machine use the fallback instead:
   `ollama pull qwen3.5:4b` (~3.3 GB).
3. In OffAir Settings, enable local-LLM summaries (OFF by
   default — summaries stay off until the user turns them on) and
   leave the model at the default (`qwen3.5:9b`) and the URL at
   `http://localhost:11434`. OffAir talks only to localhost: any
   other URL is rejected in code.
4. Verify the probe from a Python prompt in the repo:
   `OllamaClient().is_available()` → `True`, and
   `OllamaClient().list_models()` lists the pulled model.
5. In the app, open a transcribed session and click **Summarise
   session with local LLM**. Record: wall time from click to
   summary, the summary's shape (short paragraphs + bullet action
   items, facts only from the transcript), and whether the UI
   stayed responsive. A CPU summary takes minutes — that is
   expected, not a freeze.
6. If the summary fails, the error message says what to do
   (Ollama not running, or the model not pulled). Report the
   exact message, the model pulled, and the wall time.

## Bug found by reading the code during this proof

`OllamaClient` used one 10-second timeout for every request,
including `/api/generate`. A real CPU summary takes minutes, so
the old timeout would have killed every real summary with a
timeout error before the model answered — the wiring could never
have produced a summary on a real machine as written. Fixed
minimally in `app/exports/local_llm.py`: generation now uses a
separate 600-second timeout (`OLLAMA_GENERATE_TIMEOUT_SECONDS`);
the availability probe (2 s) and model listing (10 s) keep the
short timeouts. This fix is itself unproven on hardware — it was
verified only with a fake HTTP client that records the timeout
passed to `/api/generate` (600.0 s) versus `/api/tags` (2.0 s).
The first real summary run must confirm a summary actually
completes within 600 s on the user's PC; if it does not, the
timeout — not the model — is the next suspect.
