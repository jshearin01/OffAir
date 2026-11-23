# Scheduled recordings

One-off timers that start a recording automatically at a set
time, for a set app — for example, Teams at 14:30. Built Nov 22
2026.

## How to use

**By hand, in the app.** The Record view has a "Scheduled
recordings" card under the Record card: pick a date and time
(your local time), pick the app (the same app-picker values as
manual recording — live audio sessions first, then the
configured targets even when they are not running yet), add an
optional meeting subject, and press **Schedule**. The list shows
pending schedules and finished ones from the last week, each
with its status. **Cancel selected** cancels a pending one.

**By an agent, through MCP.** The MCP server has three
scheduling tools — `schedule_recording`, `list_schedules`,
`cancel_schedule` (see `docs/MCP.md`). The intended workflow: an
agent reads your Outlook calendar for the day (read-only — the
app never writes into Outlook) and sets one schedule per
meeting, carrying the subject, attendees, and organiser into
the session metadata. `list_schedules` afterwards confirms
what was set.

## Consent — read this first

A scheduled recording starts automatically and **does not
trigger Teams' own recording banner** — the same stance as
auto-detect. Tell everyone in the meeting that you are
recording. The note is on the GUI add-form, in the MCP
`schedule_recording` tool description, and here, so whichever
way a schedule is created, the person creating it sees it.

## What happens at the fire time

* The schedule fires through the exact manual Record path, with
  the schedule's target app. A meeting app that launches a
  minute late is still captured: recording starts even with no
  live audio session yet, falling back to system-wide loopback
  with the warning recorded, exactly as manual "record anyway"
  does.
* The session metadata is tagged `scheduled: {schedule_id,
  fire_at, subject?, attendees?, organiser?}` — the
  `calendar_event` key belongs to Outlook linking and is never
  overwritten.
* If a recording is already running (or a stop is in flight),
  the schedule is **skipped** and logged — never double-started.
* A failed start marks the schedule skipped with the reason, so
  it is never retried into a dialog loop. Firing never shows a
  dialog: it is automatic and must never block on the user.

## V1 limits

* **The app must be running.** There is no background service in
  v1: if OffAir is fully closed, schedules do not fire.
  Minimised to the tray counts as running — the process is
  alive. A schedule whose time passed while the app was closed
  **expires on next startup** — it never fires late.
* **No end time.** A scheduled recording stops only when you
  press Stop. The auto-detect stop gates (inactive grace,
  silence fuse) never end one, exactly as manual recordings are
  never stopped by the watcher.
* **One-off only.** No recurring schedules in v1 — set each
  meeting's timer, or let the agent set the day's from your
  calendar.
* **5-minute grace.** A due schedule up to 5 minutes past its
  time still fires (covers sleep/wake); older than that, it
  expires instead of firing late.
* **No pre-roll arming.** Pre-roll is whatever the manual start
  path already does — nothing extra is armed for schedules.

## How it is built

Schedules live in a shared store file,
`~/.offair/schedules.json` — one file the app and the separate
MCP server process both read and write, atomically (temp file +
`os.replace`), with per-record merge on write so a schedule
created in one process is never clobbered by a write from the
other. The app reloads the list whenever the file's mtime
moves (checked by its 2-second scheduler timer), so an
MCP-created schedule appears in the GUI without a restart, and
an MCP cancel is picked up the same way. A corrupt store file
never crashes the app or the server: it is logged and treated
as empty, and malformed entries are skipped individually —
valid schedules are never silently discarded.

The store module (`app/storage/schedules.py`) is pure Python,
no Qt: create / list / cancel / expire-past-due / due
detection. The scheduler itself is a 2-second `QTimer` in the
main window (`_check_schedules` → `_fire_schedule` → the manual
`on_record` path). Statuses: `pending`, `fired`, `skipped`,
`expired`, `cancelled`. Terminal records are kept in the file
indefinitely; the GUI and `list_schedules` show pending ones
plus terminal ones from the last 7 days.

## Still to verify (needs a real PC)

* A real Teams call fired by a schedule on a real Windows PC, stopped
  by hand — the whole pipeline, plus the tray-minimised case.
* A real Outlook-driven agent run: calendar → schedules →
  `list_schedules` confirmation.
* The 5-minute grace and expire-on-startup paths on real
  Windows sleep/wake behaviour.
