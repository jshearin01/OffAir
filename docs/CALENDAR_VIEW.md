# Calendar view

## What it is

The Calendar view is its own page in the left navigation — a
calendar of your recordings. Each day that has recordings shows
one badge per recording, so you can see at a glance which days you
recorded and what each session was. (A badge wraps onto two lines —
time and app on the first, source and event subject on the second —
so long app names and subjects fit the narrow day cells.)

## Modes

- **Month** — the month grid, 7 columns, Monday-first, one cell
  per day.
- **Work week** — five Monday-to-Friday columns for the week
  containing the anchor date. Never a 7-day week: Saturday and
  Sunday have no column. Day headers show the date ("Mon 30").
- **Day** — one column for the anchor date, its badges in time
  order, with the date as the header.

Switch modes with the **Month | Work week | Day** buttons under
the navigation row. The anchor is kept when you switch, so
switching modes stays on the same week or day.

## What each badge shows

- the recording's **time**,
- the **app name** it was recorded from (e.g. Teams),
- the recording **source** (e.g. per-app capture, system-wide
  loopback, auto-detected),
- the **linked Outlook event subject** when the session is linked
  to a calendar event (see [Outlook linking](OUTLOOK_LINKING.md)).

A session with no linked event shows no subject — nothing extra,
no placeholder. A session whose start time is missing is placed
by its folder-name date instead; a session with no usable date at
all is skipped, and a session with a damaged `metadata.json`
still shows on its folder-name day.

Badge content is identical in all three modes.

## Using it

- **Click a recording** to load that session — the same as opening
  it from the Recording Browser: the Session view then shows that
  session.
- **Previous / next** step one period at a time in the current
  mode — one month, one week, or one day; **Today** jumps back to
  the current period.
- A period with no recordings says so plainly ("No recordings in
  …") instead of showing an empty grid.

## What it does not do

The calendar is a way to find and open recordings. It does not
edit, move, or delete anything — and it never touches your Outlook
calendar (the Outlook link is read-only; see
[Outlook linking](OUTLOOK_LINKING.md)).
