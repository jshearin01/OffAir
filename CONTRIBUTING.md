# Contributing to OffAir

OffAir is a local-first meeting recorder: mic + system audio capture,
on-device transcription and diarization, speaker learning, and semantic
search. Everything runs on CPU with no network calls in the core paths.

## Getting started

Requires Python 3.10+ and Windows 10/11 (Linux capture is supported;
macOS is not). Install dependencies from `requirements.txt`, then run
the app with `python main.py`.

```sh
pip install -r requirements.txt
python main.py
```

Optional extras live in their own requirement files:
`requirements-speechbrain.txt` (SpeechBrain ECAPA backend) and
`requirements-dev.txt` (development tools).

## Tests

```sh
pytest tests/
```

Qt UI tests need a display or an offscreen platform
(`QT_QPA_PLATFORM=offscreen`). A small number of tests are skipped when
optional dependencies (e.g. `onnxruntime`, `torch`) are not installed;
that is expected — the app degrades gracefully without them.

## Ground rules

- **Fully local stays fully local.** No new network calls in the
  recording, transcription, diarization, or search paths. Downloads of
  models on first use are the only exception, and they must be
  explicit, verified, and cached.
- **Never modify the verbatim transcript.** Cleanup stages write
  derived artifacts (`transcript-clean.txt`); `transcript.json` is the
  record of what was said.
- **Keep the README current.** Any change to behavior, defaults,
  settings, or requirements must update `README.md` (and the relevant
  doc under `docs/`) in the same change.
- **Keep recordings out of the repo.** Audio fixtures, installer
  binaries, and local environment state are gitignored; do not force-add
  them.
- **Verify against the real thing.** New capture, transcription, or
  diarization behavior needs a measured run (timings, accuracy, or a
  real-device check), not just unit tests. Say plainly what was not
  verified.

## Pull requests

Small, focused changes with tests are easiest to review. Describe what
you changed, how you tested it, what you could not verify, and any
behavior or documentation deviation from the plan you followed.
