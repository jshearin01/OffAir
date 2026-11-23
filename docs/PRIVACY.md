# Privacy — what, if anything, leaves the machine

OffAir is fully local. Recordings, transcripts, and voice profiles
live only in the output folder on your machine, and `*.wav` is in
`.gitignore` — recordings are never committed anywhere.

## Core features: zero network calls

Recording, transcription, diarization, and speaker learning make
no network calls at all. There is no account, no telemetry, and no
analytics.

## The only downloads, ever

Models download once each, from public sources, on first use — and
after that everything works fully offline:

- Whisper models from Hugging Face (transcription),
- sherpa diarization models from GitHub releases,
- the MiniLM search model from Hugging Face (semantic search),
- the ECAPA speaker-embedding weights (speaker learning).

No audio, transcript, or voice profile is ever uploaded as part of
a download.

## The one local-network exception: Ollama

The optional local-LLM features (transcript polish and summary)
talk to an Ollama server on `http://localhost:11434` — on your own
machine only, and only when you enable them. They are **off by default**,
and any non-localhost URL is rejected in code. If Ollama
is not running, the button says so instead of failing silently.
Nothing else in the app makes any network call.

## Enterprise cleanup: transcript text leaves the machine

The optional enterprise cleanup (Session view → **Enterprise-clean**,
opt-in, **off by default**) sends the transcript *text* — never
audio — to the Claude Code CLI (`claude -p`) on this machine,
which forwards it to the user's enterprise LLM and returns a
cleaned variant. The verbatim transcript is never changed, and
the cleaned text is stored as a separate, clearly labelled
artifact. Everything else in the app stays fully local; this is
the one feature where text leaves the machine by design, and only
when the user enables it in Settings → "Pre-roll & enterprise
cleanup" (see [Transcript cleanup](TRANSCRIPT_CLEANUP.md)).

## MCP server: local, but clients can read everything

The [MCP server](MCP.md) is read-only — except the three
scheduling tools, which write only the shared schedule store
file — and local (stdio, no socket, no network). But any client
connected to it can read every transcript in the output folder
— only connect clients you trust.

## Outlook linking: read-only

OffAir reads your Outlook calendar to match a session to its
meeting, and stores that link in the session's own `metadata.json`.
**OffAir never writes anything into Outlook** — no event is
created, edited, or deleted, and no attendee ever sees anything
(see [Outlook linking](OUTLOOK_LINKING.md)).
