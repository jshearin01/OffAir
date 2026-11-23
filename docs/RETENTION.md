# Retention — old recording audio is pruned, transcripts are kept

Recordings are WAVs: each session folder holds `mic.wav`,
`system.wav` and `combined.wav` (48 kHz mono, ~5.7 MB/min per
track — about 1 GB per hour of meeting across the three tracks).
Left alone they would accumulate forever, so OffAir prunes them
automatically.

## The rule

A session's WAVs are deleted only when **all three** hold:

- the session has a transcript (`transcript.json` exists — a failed
  or unfinished transcription must stay retryable, so its audio is
  never deleted), **and**
- the session's `metadata.json` says `status: "completed"` (a
  still-recording or crashed session keeps its audio), **and**
- every WAV is older than the retention age (file modification time).

The default retention age is **24 hours**; change it in Settings
(24 hours / 7 days / 30 days / Never). "Never" disables pruning
entirely. Pruning runs at startup, after a session finishes
auto-processing, and once an hour; the main window always
excludes its own live session.

## What is never deleted

Pruning only ever removes the three WAV files — never a whole
session folder, and never anything else:

- `transcript.json` / `.txt` / `.srt`, `diarization.json`,
  `speakers.json`, `speaker_suggestions.json`, `metadata.json` —
  always kept;
- `speakers.db` (the learned voice profiles) — always kept;
- a session whose `metadata.json` still says `status: "recording"`
  (or has no readable metadata at all) is never touched — a crashed
  session keeps its audio, because it may be the only copy of a
  meeting that never got transcribed.

In short: once a meeting is transcribed, its audio is a large file
you no longer need — OffAir deletes the audio and keeps everything
you can still search, read, or re-play the transcript from.

## Low-disk warning

While recording, free space on the recordings volume is checked
every 30 seconds against a **1 GiB** threshold (`DISK_WARNING_FREE_BYTES` in
`app/storage/retention.py`; the persisted setting
`disk_warning_threshold_bytes` in `app/config.py` defaults to the
same value). One hour of meeting is ~1 GB of WAV, so 1 GiB free is
roughly one more hour of headroom.

When free space drops below the threshold, the app warns once per
recording — a status-bar message plus one non-modal dialog — and
never stops the recording for disk reasons. The check itself is
the pure helper `disk_status(path)` in `app/storage/retention.py`,
which returns a `DiskStatus` (total / used / free bytes, threshold,
`low_disk`).

## Dry-run scan

`scan_prunable_audio(...)` in `app/storage/retention.py` is the
dry run of the prune above: same eligibility rules, same
arguments, but nothing is deleted. It returns how many sessions /
files / bytes the next prune *would* delete, so the UI can show a
disk-usage line ("free space / what retention is about to
delete") without deleting anything.
