# Outlook calendar linking (Feature 3, Nov 10 2026)

Links each recorded session to the Outlook calendar event it
belongs to, so the main window can say which meeting a recording
was. The link lives in the session's `metadata.json` under
`calendar_event`:

```json
"calendar_event": {
  "subject": "Weekly sync",
  "start": "2026-11-10T10:05:00-05:00",
  "end": "2026-11-10T10:35:00-05:00",
  "attendees": ["John", "Priya"],
  "organiser": "Priya",
  "entry_id": "00000000A1B2..."
}
```

…or `null` when the match was attempted and no event matched. A
session that was never checked has no `calendar_event` key at all.

## Read-only guarantee

**OffAir never writes anything into Outlook.** This is a hard
constraint, not a preference: by design, attendees with access
to the meeting would see any change.

What the code does — the only COM operations in
`app/calendar/outlook.py`:

- `Dispatch("Outlook.Application")` + `GetNamespace("MAPI")` —
  connect to the running Outlook (starts it if it is not running).
- `GetDefaultFolder(9)` — the default Calendar folder.
- `Items`, `IncludeRecurrences = True`, `Sort("[Start]")`,
  `Restrict(...)` — read the events in the recording's window.
- Property reads only: `Subject`, `Start`, `End`, `Organizer`,
  `EntryID`, `Recipients` / `Name`.
- `GetItemFromID(entry_id)` + `Display()` — opens the linked event
  in Outlook for the user to look at (the "Open in Outlook" button).

What the code never does — anywhere, not even in a helper "for
later": no body edits, no attachments, no new items, no persist
call on an appointment object. `tests/test_outlook_linking.py`
enforces this two ways:

1. **Grep-level test** (`test_no_write_path_in_code`): scans
   `app/calendar/` plus every file this feature touches
   (`app/config.py`, `app/platform_info.py`,
   `app/ui/main_window.py`, `app/ui/settings_dialog.py`) and fails
   on any persist call, body assignment, item creation, or item
   deletion.
2. **Mock-COM read-only test** (`test_link_path_only_calls_read_operations`):
   the fake COM objects record every method call, property read,
   and property write; the test asserts calls stay inside the
   allowlisted read set and the only write anywhere is
   `IncludeRecurrences` on the `Items` *collection* (a query
   setting, not an item mutation).

## Match rules

A calendar event matches a recording when **either** rule holds:

1. **15-minute rule** — the event starts within 15 minutes of the
   recording start (either direction). Covers the usual case: you
   hit Record a few minutes after the meeting started, or the
   meeting started a few minutes after you did.
2. **50%-overlap rule** — the recording overlaps the event by more
   than 50% of the shorter of the two durations. Covers long
   events that started well before the recording (e.g. you joined
   a 2-hour workshop 40 minutes in and recorded the last hour).

If several events match, the one whose start is closest to the
recording start wins (ties break towards the longest overlap).
Only one event is stored — the best match.

The calendar query (`Restrict`) asks for events with
`[Start] <= recording_end + 15 min AND [End] >= recording_start
- 24 h`, so a long event that started the previous day and is
still running is still seen. Recurring events are included
(`IncludeRecurrences = True`).

## Timezones

Outlook returns appointment times through COM as **naive
datetimes in the machine's local time**. OffAir session metadata
stores `start_time` / `end_time` as **UTC ISO strings**. Matching
converts the session window to local time first (`astimezone()`,
then drop the tzinfo) and compares naive-local against
naive-local. Event times stored in `metadata.json` are written
back as local ISO strings **with** their UTC offset, so the UI
displays them as the user saw them in Outlook. The folder-name
stamp (`YYYY-MM-DD_HH-MM-SS`) is local time too, consistent with
the COM values.

`tests/test_outlook_linking.py::test_utc_stored_session_time_converted_to_local`
covers this: a session stored as `15:00 UTC` with the local zone
set to `America/New_York` (EST, −05:00) matches a 10:05 local
event, and the stored start carries the `-05:00` offset.

## When matching runs

- **At finalise** (`MainWindow.on_stop`, right after
  `finalize_session`): the just-finished session is matched and
  the result stored immediately, so the "Linked to:" line is
  already there when the session lands in the library.
- **On load** (`MainWindow.load_session`): an old session with no
  `calendar_event` key gets one attempt. A session that was
  already checked (linked, or stored `null`) is **not** re-queried
  on every load — re-querying would be slow and could overwrite a
  good link with `null` if the event was later deleted.

Both call sites are wrapped so linking can never fail a stop or a
load: any COM failure logs and the session works exactly as
before, just without a calendar link.

## Settings

`outlook_link_enabled` (Settings → "Link recordings to their
Outlook calendar event"), default **ON** where Outlook is
available. On machines without Outlook the checkbox is disabled
with the reason from `outlook_available()` — the same pattern as
the auto-detect toggle. Turning it off leaves any previously
stored link untouched (nothing is written, nothing is queried).

## Graceful degradation

| Situation | Behaviour |
|---|---|
| Not Windows | `outlook_available()` → `(False, reason)`; no link attempted |
| pywin32 missing | Same, with an install hint |
| Outlook not installed | Same, registry ProgID check fails |
| Outlook not running / MAPI error | `find_event_for_session` logs and returns `None` |
| No event matches | `calendar_event: null`, UI shows "No calendar event found" (plain label, not an error) |
| COM fails mid-query | Session saves/loads normally, no link |

Every COM call is inside `try/except` and logs; nothing in this
feature raises to a caller.

## Verification — open gate

**The COM code has mock tests only; no run against real Outlook
has been done yet.** What is verified here:

- All 21 tests in `tests/test_outlook_linking.py` pass (mock COM):
  both match rules, the exactly-50%-overlap boundary, no-match,
  Outlook-absent, UTC→local timezone conversion, open-in-Outlook
  read-only calls, session-metadata writes, the grep-level
  no-write-path test, and the Qt wiring (Settings toggle disabled
  without Outlook; main-window label states).
- `outlook_available()` on this machine returns
  `(False, "Outlook linking needs Windows with Outlook installed;
  this machine is linux. …")`.
- Full suite stays green.

**Not verified — open gate:** a real end-to-end against actual
Outlook on a real Windows PC with Outlook. Do not claim it passed until
someone runs it. Procedure on the Windows machine:

1. Install the build with pywin32 (`pip install -r
   requirements.txt` — `pywin32>=306; sys_platform == "win32"`).
2. With Outlook running and a meeting on the calendar, record a
   short session overlapping it, stop, and check the main window
   shows "Linked to: \<subject\>, \<time\>" and `metadata.json`
   carries the `calendar_event` dict (subject, start/end with the
   local offset, attendees, organiser, entry_id).
3. Click **Open in Outlook** — the event opens in Outlook.
4. Record a session with no calendar event — the label reads "No
   calendar event found" and the session is otherwise normal.
5. Load an old session recorded before this feature — it links on
   load (one attempt).
6. In Outlook, confirm the event was **not modified**: no body
   change, no new attachment, no new item, last-modified time
   unchanged.
7. With Outlook closed, record and stop — the session saves
   normally with no error dialog and no link.
8. Untick the Settings toggle — stopping a recording performs no
   Outlook query (no Outlook process starts).

Known assumptions to confirm on that run: Outlook's `Restrict`
date format (`YYYY-MM-DD HH:MM` in local time) is accepted;
`IncludeRecurrences` + `Sort("[Start]")` behave on the real
`Items` collection; `GetItemFromID` + `Display` opens the item
without a security prompt. If `Dispatch` prompts or is blocked
by Outlook's object-model guard on a real Windows PC, that is a real
finding to report — the feature must then degrade to "no link"
rather than show a prompt.
