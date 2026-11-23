# Transcription

OffAir transcribes with **faster-whisper on CPU (int8)** — no GPU,
no CUDA, no cloud. Stopping a recording always finalizes the
transcript (live chunks + tail, or the whole file when there is no
live state) — that happens whether or not auto-processing is on.
The auto-process setting gates only what follows the transcript:
the chain transcribe → diarize → suggest/identify speakers runs
after a stop only when it is on. A transcript can also be run
manually from the Session view's "Transcribe session" button.

## Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny  | 75 MB  | fastest (~4x faster than small) | lowest — drafts / search, expect word errors |
| base  | 142 MB | fast (~2x faster than small)    | good for clear speech — the default |
| small | 466 MB | ~2.5x slower than base | same accuracy as base on clean audio; opt-in (the former default) |

Pick the model on the Session view before transcribing. The model
downloads once on first use (from Hugging Face) into
`~/.offair/models` and is reused afterwards. The default follows the current build: unless you explicitly pick a
model in Settings, an upgrade can move the default (base since
1.0.3) and you get the new one. A pick you make in Settings is kept
across updates. Settings saved under 1.0.2 with the old default
("small") migrate to base once on upgrade — a "small" explicitly
picked back then is indistinguishable from that old default, so it
migrates too; re-pick small in Settings and it sticks.

Near-live mode (the "Run near-live (chunked) on session" button
on the Record view) uses the **configured** model — the same one
live transcription uses (`base` by default; the Record view says
which one, "Transcribing with Whisper base"): it
transcribes the loaded session in chunks while you watch, and its
output is provisional — the full transcription above is the record.

## Live transcription (while recording)

While a recording runs, the transcript already grows in the
Near-live pane in 25-second windows using the **configured**
model (`base` by default — a 25 s chunk takes ~3.4 s at its
measured RTF of 0.135, comfortably real-time), with provisional
You/Remote labels from the two tracks. After every chunk the
progress is saved to `transcript-live.json` in the session folder
(same shape as `transcript.json`, plus `"live": true`,
`chunks_done`, and `watermark_seconds`), so a crash loses at most
the chunk in flight. Pausing the recording pauses live
transcription too; it resumes with the recording.

When you stop, only the tail — the audio after the last completed
window — is transcribed, then merged with the live chunks (the
0.5 s overlaps are de-duplicated) into the canonical
`transcript.json` / `.txt` / `.srt`, and `transcript-live.json`
is deleted. The tail pass runs in the background (the UI stays
responsive, with progress and a Cancel button): when live
transcription died early the tail is most of the meeting, so
running it on the UI thread would freeze the app for minutes.
If a live chunk is still transcribing when you stop, the finalize
waits for it — but only up to 60 s; after that the in-flight
chunk is abandoned and the finalize proceeds from the saved live
transcript instead, so Stop can never wait forever on a stuck
worker. The diarization → suggest/identify chain then runs
exactly as after a full-file transcription, when auto-processing
is on. If there is no live state (live failed before the first
chunk, or the app restarted mid-recording), Stop falls back to
transcribing the whole file. `transcript-live.json` is a
crash-recovery record only: audio retention still requires the
final `transcript.json`, and search reads the final file.

On a first run the model download is shown as "Downloading
Whisper base (~142 MB) — first run only" with elapsed time.
faster-whisper gives no byte-level download-progress hook, so
there is no progress bar for it — the first chunk simply waits.

## Cancelling a transcription

The Session view's **Cancel** button (next to "Transcribe
session") stops a running transcription after the current
segment — manual, or the Stop finalize. The partial segments are
discarded and no `transcript.json` is written — a cancelled run
never leaves a partial transcript behind. Cancelling the Stop
finalize leaves `transcript-live.json` on disk, so the session
stays transcribable by hand (Transcribe in the Session view).
Cancelling also stops the auto-process chain; the recording
itself is untouched. A cancel requested while the model is still
downloading (first run) takes effect once the model has loaded:
the download inside the model constructor cannot be
interrupted mid-download, so Cancel says so instead of
promising an instant stop.

## Output files

Every transcription writes three files into the session folder:

- `transcript.json` — timestamped segments (start/end, text,
  speaker label once diarized). This is the record everything else
  is derived from.
- `transcript.txt` — plain text.
- `transcript.srt` — subtitles.

Re-transcribing a session overwrites these three files; the
derived cleaned/polished copies are separate files (see
[Transcript cleanup & names](TRANSCRIPT_CLEANUP.md)).

## How long it takes

CPU-only, so a meeting takes a fraction of its own length to
transcribe — roughly 13% of real time on a modern laptop for
`base` (the default since Nov 19 2026; measured real-time factors
and per-model timings are in [CPU_TIMES.md](CPU_TIMES.md)). The progress bar on the Session view
shows where it is; the recording itself is saved before
transcription starts, so a failed transcription can never lose it.
