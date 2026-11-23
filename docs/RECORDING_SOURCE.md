# Recording source

Every OffAir session records *what it is a recording of*, in
`metadata.json`, under two keys. The Recording Browser filter,
the calendar view, and the MCP server all read the same keys —
there is one definition, in `app/storage/source.py`.

## Metadata contract (MCP + calendar: read this)

```json
{
  "source": "teams",
  "source_detail": "Microsoft Teams"
}
```

* **`source`** — exactly one of:

  | value | label | meaning |
  |---|---|---|
  | `teams` | Teams | Microsoft Teams meeting |
  | `zoom` | Zoom | Zoom meeting |
  | `webex` | Webex | Cisco Webex meeting |
  | `meet` | Google Meet | Google Meet call (browser tab) |
  | `browser` | Browser | Browser audio — a video, a stream, a web call that is not Meet |
  | `solo` | Solo (mic only) | No target app — mic-only / system-wide with no subject |
  | `claude-code` | Claude Code | Working session with the Claude Code agent |
  | `other` | Other | Source could not be determined — never a guess |

  The values are stable API. Do not rename them; the MCP
  `list_sessions` / `search_meetings` filters match on these
  exact strings, and the calendar badges show `source_label`
  (the Label column above).

* **`source_detail`** — free text: the app display name
  (`"Google Chrome"`), a window title, or a user note. It is
  *evidence*, not a second guess: for an unknown exe it holds
  the prettified exe name (`"Foobar"`), so the source can be
  corrected later instead of the information being lost.

## How the source is set

1. **At record start** — `MainWindow.on_record` resolves it with
   `app.storage.source.resolve_recording_source()` and passes it
   to `create_session()`:
   * The **Default recording source** in Settings overrides detection.
     `Auto (detect from app)` is the default; an explicit pick
     always wins. The override exists because some sources have
     no identifying process — a Claude Code session is mic +
     system audio while working with the agent, and nothing
     running identifies it.
   * With Auto, detection runs on the capture target process
     (`detect_source`): the exe name maps through
     `_PROCESS_SOURCE` in `app/storage/source.py`.
   * An auto-started recording (meeting watcher) detects from
     the configured target (Teams), not the picker's choice —
     same rule as the capture target itself.
   * No app picked at all → `solo`. Note the capture code falls
     back to `ms-teams.exe` as the *capture* target in that case;
     the fallback is capture bookkeeping, not evidence of a
     Teams meeting, so detection deliberately uses the picked
     app (`selected_process_name()`), not the fallback.

2. **Correction afterwards** — the Session view's **Source** row
   (under the Session chooser) edits the loaded session's source
   + detail; both save to that session's `metadata.json`
   immediately via `set_session_source()`. The "Loaded session"
   line at the top of the view shows the source.

3. **Backfill** — sessions recorded before this feature have no
   `source` key. `load_metadata()` derives one on load from the
   stored `app_name` where it maps cleanly (`"Microsoft Teams"`
   → `teams`, `"Google Chrome"` → `browser`, `"Zoom"` → `zoom`,
   `"Webex"` → `webex`, `"Google Meet"` → `meet` — the full map
   is `_APP_NAME_SOURCE` in `app/storage/source.py`), else
   `other`, and writes the derived value back into
   `metadata.json`. The write happens only when the source is
   missing or invalid, so it is idempotent: a session that
   already has a valid source is left alone, and a second load
   writes nothing. `load_metadata(..., backfill=False)` is the
   pure read with no side effects.

## Dictation sessions

Dictation is a mic-only flow producing a cleaned transcript
note. It reuses the ordinary session / transcribe / cleanup
path — no separate storage, no separate transcription code.
A dictation session is an ordinary session whose `metadata.json`
carries exactly these distinguishing fields (created by
`create_dictation_session()` in `app/storage/sessions.py`,
recognised by `is_dictation_session()`):

```json
{
  "app_name": "Dictation",
  "source": "solo",
  "source_detail": "",
  "dictation": true
}
```

All other `metadata.json` keys are the standard session keys
(`session_id`, `mode`, `status`, `start_time`, `end_time`,
`duration_seconds`, `sample_rate`, `channels`, `devices`,
`warning`, `pause_policy`, `files`). A `solo` source alone does
not make a session dictation — only `"dictation": true` does.

The primary readable artifact is the deterministic cleaned
note, `transcript-clean.txt`, produced by `clean_session()`
(the rule layer in `app/transcription/cleanup.py`) from the
verbatim `transcript.json` — the same artifacts every session
produces, with the cleaned note treated as the note itself.
LLM polish runs only if already enabled (the existing
`transcript_polish_enabled` setting / manual Polish action);
dictation changes no Ollama default and adds no automatic LLM
step.

Wording for the UI and docs (verbatim): **"Dictation captures
the spoken side only — what you say into the microphone. It
does not capture the other side of a conversation."** The
transcript captures the spoken side only: it does not include
anything another person said, anything on screen, or anything
that was not spoken aloud into the microphone.

## Bookmarks and pins (metadata contract)

Bookmarks live in `metadata.json` under `"bookmarks"`, written
via `update_metadata` (`app/storage/bookmarks.py`):

```json
{
  "bookmarks": [
    {"t": 42.5, "label": "Budget decision",
     "created": "2026-11-21T20:00:00+00:00"}
  ]
}
```

`t` is seconds (float) from the start of the session, `label`
is free text (empty when unnamed), `created` is an ISO-8601
UTC timestamp. Whether a bookmark's audio still exists is
computed when listing (any of `mic.wav` / `system.wav` /
`combined.wav` present) — it is never stored, so retention
deleting the WAVs is reflected immediately and a bookmark
pointing at deleted audio says so plainly.

Pinning is a single metadata flag, also via `update_metadata`:

```json
{
  "pinned": true
}
```

Absent (or `false`) means unpinned. The library sorts pinned
sessions first, then newest (`SessionSummary.pinned` in
`app/storage/browser.py`); search accepts `speaker=...` to
filter by who spoke (display name after renames, exact
case-insensitive match).

## Process → source mapping

| Process (exe, case-insensitive) | Source |
|---|---|
| `ms-teams.exe`, `Teams.exe`, `ms-teamsupdate.exe`, `teams`, `ms-teams` | `teams` |
| `zoom.exe`, `zoom`, `zoom.us` | `zoom` |
| `webex.exe`, `atmgr.exe` | `webex` |
| `chrome.exe`, `msedge.exe`, `firefox.exe`, `brave.exe`, `opera.exe`, `chromium.exe` (+ the bare Linux names) | `browser` |
| anything else / no target app | `other` / `solo` |

Notes:

* **Google Meet has no process mapping.** It runs inside a
  browser, so a Meet call is `browser` by detection and `meet`
  only via the manual override (or a correction afterwards).
* **Webex names are best-effort.** `webex.exe` is the Webex
  app; `atmgr.exe` is the Webex Meetings desktop-app manager
  (its meeting process). If a real Webex run shows a different
  process name, add it to `_PROCESS_SOURCE` — the mapping is
  data, not logic.
* **Zoom on macOS** (`zoom.us`) is mapped even though OffAir
  has no macOS capture backend yet; the mapping is harmless
  there.

## What was deliberately NOT built

**Foreground-window-title capture.** The plan allowed capturing
the foreground window title at record start into
`source_detail` for browser targets (a YouTube page title
identifies the video; a Meet window title would identify a
Meet call). It was skipped: it is Windows-only
(`GetForegroundWindow` / `GetWindowTextW` via ctypes), it
cannot be verified on this Linux VM beyond a mock, and the
value is nice-to-have — browser sessions already carry the app
name, and Meet / specific videos are reachable through the
manual override and the post-hoc editor. If it is ever added,
it must stay best-effort behind a `try/except` that can never
block or crash record start, and `source_detail` is the place
for the title.

## Filtering

* **Recording Browser** — the Source dropdown next to the
  search box offers "All sources" plus one entry per source
  actually present in the library (`list_sources()`, in enum
  order). It narrows the list in every search mode (text,
  semantic, hybrid).
* **Data layer** — `list_session_summaries(root, source=...)`
  and `search_sessions(root, query, source=...)` take an
  optional `source`; `None` returns everything. An unknown
  source value matches nothing.
* **Calendar view / MCP** — read `source` / `source_detail`
  straight from `metadata.json` (see the contract above).
  `SessionSummary.source` / `.source_detail` / `.source_label`
  carry the same values in-process.

## Tests

`tests/test_recording_source.py`: process mapping for every
enum value (including unknown exe → `other`), override-wins
(including `claude-code` over Chrome), backfill (Teams app name
→ `teams`, unmappable → `other`, existing source untouched,
idempotent), browser source filter (matching sessions only,
"All" returns everything), the edit path (change persists to
`metadata.json`), and the Record-card flow end to end
(detected `browser` for Chrome, override `claude-code`,
no-app-picked → `solo`). `tests/test_dictation.py` covers the
dictation session shape and the cleaned-note artifact;
`tests/test_bookmarks.py` covers bookmark storage and the
deleted-audio listing state; `tests/test_browser.py` covers
the speaker filter and pinned sort.
