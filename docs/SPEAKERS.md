# Speakers & renaming — learning who is who

OffAir learns voices across calls. Each diarized speaker's voice is
turned into an embedding (ECAPA speaker embedding, computed locally)
and stored in `speakers.db` in the recordings root. The next time
that voice appears, OffAir suggests the match — with a confidence
and a margin, so you can see how sure it is.

## Unnamed speakers

A voice OffAir has never seen before is auto-enrolled as a
persistent identity: "Speaker A", "Speaker B", … The name sticks
to that voice from call to call, so you can rename it later and the
rename still lands everywhere.

## The People view

- **Known speakers** — everyone OffAir has learned. Select one and
  **Play sample** to hear their voice, or **Rename** to change
  their name.
- **Suggestions for this session** — the matches for the loaded
  session, each with confidence and margin. **Confirm** accepts a
  suggestion; **Correct** (with a name typed in the field below)
  tells OffAir it got that voice wrong. Not sure? Play the sample
  first.
- **Rename a speaker or label** — pick a speaker (or a label such
  as "Speaker A") from the list, type the new name, press **Rename
  speaker**.

## Renaming propagates

Renaming a learned speaker updates **every past session**: the
speaker's name in `speakers.json` and the segment labels in
`diarization.json` / `transcript.json` are rewritten wherever that
speaker appears. Renaming a one-off label only affects that session.
A rename is never destructive — the audio and the verbatim
transcript text are untouched; only the labels change.

## Person pages

Clicking a person in People opens their page: meetings, utterances,
the voice sample, and rename only. There is no commitments section
and no placeholder for one.

* **Sessions** — every session where the speaker has at least one
  transcript segment, newest first, with an utterance count each.
  The speaker is matched by display name: the stored segment label
  after the session's `speakers.json` renames are applied. A rename
  in one session and a display name already stored in another both
  count, so renames propagate here with no extra bookkeeping.
  A speaker need not be in `speakers.db` to have sessions here.
  Clicking a meeting loads it.
* **Utterances** — every segment the speaker spoke, sessions newest
  first and chronological within a session, each with session,
  start, end, and text. Click-to-play uses the segment's
  timestamps against the session audio, where audio exists.
* **Voice sample** — the speaker's longest turn, with the same
  audio-file rules as embedding extraction — `mic.wav`
  for "You", `system.wav` for "Remote", `combined.wav` otherwise.
  The database path is tried first (newest sample session that
  still has audio); otherwise sessions are scanned newest-first,
  so a speaker with no stored embedding still gets a sample
  while their audio survives. When no playable sample exists
  (retention deleted the WAVs), the page says so plainly.

The page reuses the same speaker queries as the MCP
`get_speaker` tool. `speakers.db` itself is unchanged: one SQLite
file in the recordings root (`speakers.db`), names + embeddings
local only.

## When the app gets it wrong

A diarization boundary error can poison a new speaker's first
sample (the sample is taken from the start of their first segment,
and a wrong boundary mixes two voices). If a suggestion looks wrong,
Correct it with the right name (or Play the sample first) — the
correction teaches the stored profile, it does not just relabel
one session.

## Mixed clusters are never learned from

One guard stands in front of every way a voice can be learned —
automatic identification, learning a whole session, Confirm, and
Correct. A cluster is only learned from its clean audio: segments
that sit inside their own diarization turns, enough of it, with
both halves sounding like the same voice. A cluster that is
mostly boundary spill, or whose halves sound like different
voices, is mixed: automatic identification leaves it Unknown,
and Confirm/Correct refuse it rather than storing a blended
sample. Short but clean clusters can still be confirmed by hand;
automatic enrolment additionally needs at least 5 seconds of
clean audio.
