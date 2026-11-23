# OffAir — Getting started

OffAir records a meeting on your own machine, transcribes it,
labels who said what, and learns to recognise the people you meet
with — all locally. No cloud, no account, no GPU.

## First run

The first time OffAir starts it shows this page's content once, in
a welcome dialog: the flow above, the local-only promise, and the
one-time downloads. The Whisper base model (~142 MB) and the
sherpa diarization models (~33 MB) download once, on first use,
then everything works offline. The dialog is shown once (a marker
file, `onboarding_shown`, next to `settings.json`); you can read
this page again any time in Help → Getting started.

## The normal flow

1. **Record.** Pick the app (or "Record
   everything you hear"), press Record, and press Stop at the end.
   The microphone and system-audio devices are picked in
   Settings → "Recording devices" (the choice is read when a
   recording starts and is not saved).
   Each session is a folder with `mic.wav`, `system.wav`,
   `combined.wav` and `metadata.json`. See [Recording](PER_APP_CAPTURE.md).
2. **Transcribe.** After you stop, OffAir transcribes the session
   automatically (or press "Transcribe session" on the Session view
   to pick a different model). See [Transcription](TRANSCRIPTION.md).
3. **Diarize.** The transcript is labelled with speakers — sherpa
   mode by default, no account or token needed. See
   [Diarization](DIARIZATION.md).
4. **Name speakers.** Unknown voices become "Speaker A",
   "Speaker B", … until you name them on the People view; a name
   then applies to every past session. See
   [Speakers & renaming](SPEAKERS.md).
5. **Find it later.** The Recording Browser searches every
   transcript by exact text, by meaning, or both. See
   [Search](SEMANTIC_SEARCH.md).

## Where things live

- **Sessions:** the output folder you chose (default
  `~/OffAir/recordings`), one folder per session.
- **Settings:** `~/.offair/settings.json` (retention, theme,
  auto-detect, models).
- **Voice profiles:** `speakers.db` in the recordings root.

## People pages

On the People view, select a speaker and press **Open person
page**. The page lists every meeting that speaker took part in
(with how many utterances they had in each — clicking a meeting
loads it), everything they said (click an utterance to hear that
moment), and their enrolled voice sample, with a Play button. The
**Rename** button fills in the rename panel on the Session view,
exactly as renaming from the list does. There is no commitments
section: OffAir does not track commitments.

## Bookmarks

While recording, press **Bookmark** (on its own row below the
transport buttons, or Ctrl+B) to mark the current moment. On the Session view the
bookmark list sits under the waveform: click a bookmark to hear
that moment, and **Add bookmark at current moment** marks the
moment now playing, the selected segment's start, or the start of
the session when nothing is playing (a label prompt is offered
there — none while recording, so the moment is never lost to
typing). A bookmark whose audio was deleted by the retention rule
renders disabled, with "audio deleted" in plain words. Bookmarks
are stored in the session's `metadata.json`.

## Dictation

Press the **Dictation** toggle beside Record for a mic-only note:
the app picker and system meter hide, and the session is recorded
from the microphone alone. Dictation captures the spoken side
only — what you say into the microphone. It does not capture the
other side of a conversation. After Stop, the session is
transcribed and cleaned like any other, and the Session view
shows the same spoken-side-only note above the transcript; the
cleaned note (`transcript-clean.txt`) is the primary readable
artifact. Live transcription during a dictation recording may
show nothing until Stop (the live poll reads the system track,
which a mic-only recording has none of) — the finalize pass
still transcribes the whole session.

## Enterprise cleanup (opt-in)

On the Session view, **Enterprise-clean — transcript text leaves
this machine** sends the transcript *text* to the Claude Code CLI
(`claude -p`) on this machine and writes a third, explicitly
labelled variant, **Enterprise-cleaned**, next to Original and
Cleaned. Audio never leaves the machine, the verbatim transcript
is never changed, and the run has progress and Cancel. It refuses
to run unless enabled in Settings → "Pre-roll & enterprise
cleanup" — the button says where to enable it. When the CLI is
missing, a guided dialog says how to install it.

## The waveform

Under the transcript on the Session view, the waveform shows the
session's loudness (0.25 s buckets, computed in a background
thread and cached per session), coloured by speaker along the
top, with bookmark markers in red. Click anywhere on it to hear
that moment and highlight the segment being played.

## The other pieces

- The [Calendar view](CALENDAR_VIEW.md) shows your recordings on a
  month grid. [Outlook linking](OUTLOOK_LINKING.md) matches each
  session to its calendar event — read-only, OffAir never writes
  anything into Outlook. The speaker filter above the grid dims
  days with no matching session (they stay visible, at reduced
  opacity, and cannot be clicked) instead of hiding them.
- Old recording audio is pruned automatically once transcribed
  (default: after 24 hours); transcripts are always kept. See
  [Retention](RETENTION.md). The Library footer and the Settings
  page show the free space on the recordings volume and how much
  audio the next retention run would delete; while recording, a
  low-disk warning (below 1 GB free) appears once per session and
  never stops the recording.
- **Pre-roll** (Settings → "Pre-roll & enterprise cleanup",
  off by default) keeps a short memory-only buffer of the
  microphone and the selected app's audio before you press
  Record, and adds it to the start of the recording when one
  starts. The consent note in Settings says exactly what is
  kept and when it is discarded; nothing is written to disk
  unless a recording starts.
- The Library lists pinned sessions first (star a row with the
  **☆ Star** button), grouped under week headers — "This week"
  for the current week, "Week of …" for earlier ones. Search
  results stay in relevance order, without headers. The speaker
  filter in the Library search narrows by who spoke.
- Live transcript items are grey and italic, marked
  "provisional", until the finalize pass confirms them, and the
  live status line names the Whisper model and the diarization
  mode in use. No confidence number is ever shown — the engine
  reports none.
- The **theme toggle** in the header switches light ↔ dark and
  remembers the choice. The tray icon (when the system has a
  tray) shows recording time in its tooltip, and its menu offers
  Show, Stop recording (the same stop as the window's Stop
  button), and Quit. Without a tray, everything still works from
  the main window.
- [Keyboard shortcuts](KEYBOARD_SHORTCUTS.md): Ctrl+B bookmark,
  Ctrl+P pause/resume, Ctrl+K jump to search — in-app only.
- Transcripts are cleaned automatically (fillers removed into a
  derived copy; the verbatim transcript is never edited) and a
  names dictionary keeps product names spelled right. See
  [Transcript cleanup & names](TRANSCRIPT_CLEANUP.md).
- The [MCP server](MCP.md) exposes the archive to MCP
  clients such as Claude Code, over local stdio — read-only,
  except the three scheduling tools, which write only the shared
  schedule store file.
- [Privacy](PRIVACY.md) covers what, if anything, ever leaves the
  machine: almost nothing — see that page for the full story.
