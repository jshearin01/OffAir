# MCP server — OffAir over the Model Context Protocol

OffAir exposes its meeting archive to MCP clients (Claude Code,
Claude Desktop, ...) as a **read-only, local, stdio** server —
with ONE scoped exception: the three scheduling tools (Nov 22
2026, see below), which write only the shared schedule store
file. There is no HTTP, SSE or streamable-HTTP transport — the
server listens on no socket — and no network calls, ever: it
reads local session folders, `speakers.db` and `search_index.db`
only. The one exception is `ask_meetings`, which may talk to a
local Ollama server on `http://localhost:11434` when one is
running; when it is not, it falls back to plain search results
and says so (see below). The server works with the network
unplugged.

## Privacy — read this first

**Any client connected to this server can read every transcript in
the output folder.** Only connect clients you trust — a connected
Claude Code session, a Claude Desktop config, or any other MCP
client has the same read access to your meetings as you do. There
is no per-client scoping and no approval step per query: connecting
is the approval.

**A connected client can also schedule recordings** (the
scheduling tools below). A scheduled recording starts
automatically at its time and does not trigger Teams' recording
banner — tell participants you are recording. Scheduling is the
only write the server can do: it cannot record, transcribe,
rename, delete, or change a setting.

## Run it

```powershell
# Frozen build (Windows):
OffAir.exe --mcp
# Dev install (from the repo root):
python main.py --mcp
python main.py --mcp --output-dir "D:\Meetings\OffAir"
# Console script (after `pip install -e .`):
offair-mcp
offair-mcp --output-dir ~/OffAir/recordings
```

`--output-dir` overrides the recordings folder for that run;
otherwise the saved setting (or `~/OffAir/recordings`) is used.
The server runs until the client disconnects (Ctrl-C / closing the
client). Logs go to `offair.log` in the output folder, never to
stdout — stdout is the MCP wire.

`--mcp` needs the `mcp` package (`pip install -r requirements.txt`
includes it). Without it you get a plain error and exit code 2,
not a traceback.

## In-app setup page

The main window has an **MCP setup** view (its own row in the left
navigation) with everything on this page in
one place, computed for the install actually running:

* One plain sentence saying what the server is, plus the privacy
  note above.
* The exact command to run the server: `<install-dir>\OffAir.exe
  --mcp` for the installed app (the real exe path when frozen),
  `python -m app.mcp` from the repo root — or `offair-mcp` — for a
  dev install. Each snippet's `command` / `cwd` are the real paths
  for the current run, never a placeholder.
* Copy-paste config snippets for Claude Code, Claude Desktop,
  and Codex, each with a Copy button.
* A **Check status** button that reports, on demand: whether the
  MCP tools module imports, whether the `mcp` SDK package is
  installed (if not: "mcp package not installed — run pip install
  -r requirements.txt"), whether the recordings folder was found,
  and how many sessions would be exposed. A problem says what it
  is; nothing fails silently.
* The twelve tools in plain language, one line each — the nine
  read-only archive tools plus the three scheduling tools, with
  the scheduling exception and its consent note stated on the
  page.

The page follows the app theme (light + dark) and fits the
1280x800 window (the view is scrollable, like Session and
People). When Settings move the output folder, the status
check follows the new folder.

## Connect a client

### Claude Code

```json
{
  "mcpServers": {
    "offair": {
      "command": "C:\\OffAir\\OffAir.exe",
      "args": ["--mcp"]
    }
  }
}
```

(Dev install: `"command": "python", "args": ["main.py", "--mcp"]`
with `"cwd"` set to the repo root, or `"command": "offair-mcp"`.)

### Codex

Add the server to the `[mcp_servers]` section of the Codex config
file (`~/.codex/config.toml`):

```toml
[mcp_servers.offair]
command = "C:\\OffAir\\OffAir.exe"
args = ["--mcp"]
```

(Dev install: `command = "python"`, `args = ["-m", "app.mcp"]`
with `cwd = "<repo root>"`.)

### Claude Desktop

Add the same server to the `mcpServers` object in the Claude
Desktop config file (`%APPDATA%\Claude\claude_desktop_config.json`
on Windows, `~/Library/Application Support/Claude/
claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "offair": {
      "command": "C:\\OffAir\\OffAir.exe",
      "args": ["--mcp"]
    }
  }
}
```

## Read-only tools (nine)

Every read-only tool returns one JSON object. All nine are
annotated `read_only_hint: true`, and there is no write, delete,
rename, record, transcribe or summarise-and-save operation among
them.

| Tool | What it does |
| --- | --- |
| `search_meetings(query, mode, limit, source, speaker)` | Search sessions. `mode`: `text` (substring, always offline), `semantic` (meaning-based), `hybrid` (exact hits first, then semantic — default). `source`: optional recording-source filter (see below). `speaker`: optional speaker filter — only sessions that speaker spoke in (display name, case-insensitive exact match). Each result is a session plus `score`, `match_type`, `snippet`. When the semantic model is unavailable it falls back to text search and says so in `note` — it never fails outright. An empty query lists every session. |
| `list_sessions(limit, since, source, pinned_only)` | Sessions newest first. `since`: ISO date (`2026-11-01`) or datetime. `source`: optional recording-source filter. `pinned_only`: keep only pinned (starred) sessions. Every entry carries `pinned` and `dictation`. |
| `get_transcript(session_id, format, variant)` | `format="text"`: one `[MM:SS] Speaker: text` line per segment. `format="segments"`: `[{start, end, speaker, text}]`. Speaker renames from the session are applied. `variant`: `verbatim` (the record — default), `clean` (the deterministic cleaned note) or `enterprise` (the external-LLM cleanup, labelled as such in the result — see below). Clean and enterprise variants are text only. Errors when the session has no transcript yet, or no artifact for the requested variant. |
| `get_session(session_id)` | Status, speakers, app, duration, `source` / `source_detail`, `dictation` (true only for dictation sessions), `pinned`, `bookmarks` (earliest first — see below), and `calendar_event` (always `null` in v1 — the Outlook linking feature will fill it later). |
| `list_bookmarks(session_id)` | A session's bookmarks, earliest first: each marked moment, its label, and whether its audio still exists. Each entry is `{t, label, created, audio_available}`; `audio_available` is false when retention has deleted the session's audio. Empty when the session has no bookmarks. Bookmark creation stays in the app only. |
| `list_speakers()` | Learned speakers from `speakers.db`: `{name, backend, sample_count, sessions}`. Empty (not an error) when there is no `speakers.db` yet — and it does not create one. |
| `get_speaker(name)` | Everything about one speaker: their sessions, everything they said, and their voice sample, if one still exists. Matches the display name case-insensitively and exactly. Returns `{name, known_in_database, sessions, utterances, sample, sample_available, session_count, utterance_count}`. A speaker need not be in `speakers.db` to have sessions and utterances. An unknown speaker returns empty sessions / utterances, not an error. `sample` is a pointer back into a session folder (audio path + timestamps), never audio bytes. |
| `what_did_speaker_say(speaker, query, limit)` | Segments spoken by one speaker across all sessions, newest first. Matches the display name (renames applied) case-insensitively; `query` narrows to segments containing that text. Unknown speaker: empty results, not an error. |
| `ask_meetings(question)` | RAG over the archive: retrieves the most relevant transcript excerpts with hybrid search, then asks the local Ollama model to answer **from those excerpts only**. When Ollama is not running there is no generated answer — the response says so plainly (`source: "search-fallback"`, `answer: null`) and returns the excerpts instead. Nothing is ever invented: with no matching meetings the answer is null and the note says so. |

`limit` is 1–100 (default 10). Unknown sessions, modes, formats,
variants and sources raise a tool error with a plain message.

## Scheduling tools — the ONE write exception (Nov 22 2026)

> **⚠️ Exception to the read-only rule, explicitly authorised.**
> The three tools below are the only write operations in the
> server. They read and write ONLY the shared schedule store
> file (`~/.offair/schedules.json`) — they never touch a
> transcript, a session folder, `speakers.db`, the search index,
> or any setting. They cannot start a recording themselves: a
> schedule fires only when the OffAir app is running. If this
> exception is ever widened, this section is where it must be
> declared first.

| Tool | What it does |
| --- | --- |
| `schedule_recording(fire_at, target_app, subject, attendees, organiser)` | Create a one-off scheduled recording. `fire_at`: ISO 8601 with a UTC offset (e.g. `2026-11-23T14:30:00-05:00`); a naive value is read as local time. Must be in the future. `target_app`: an exe name from the app picker (`ms-teams.exe`, `chrome.exe`, …) or a display name the picker knows (`Teams`, `Chrome`); unknown names are rejected. `subject` / `attendees` / `organiser` are optional calendar details carried into the session metadata. Returns the schedule id. Annotated `read_only_hint: false`. |
| `list_schedules()` | Pending schedules plus finished ones from the last 7 days, with status (`pending` / `fired` / `skipped` / `expired` / `cancelled`), local fire time, app, subject, and who created them. Annotated `read_only_hint: true` — it only reads the store. |
| `cancel_schedule(schedule_id)` | Cancel one pending schedule. Fired / skipped / expired / cancelled ones cannot be cancelled (tool error). Annotated `read_only_hint: false`. |

### Consent

A scheduled recording starts automatically and **does not trigger
Teams' recording banner** — the same stance as auto-detect.
Tell participants you are recording. The consent note is in the
`schedule_recording` tool description itself, in the GUI
add-form, and in `docs/SCHEDULED_RECORDING.md`.

### Agent workflow (Outlook calendar → schedules)

1. Read today's Outlook calendar (read-only — the app never
   writes into Outlook).
2. For each meeting, call `schedule_recording` with the meeting's
   start time, `ms-teams.exe`, and the subject / attendees /
   organiser from the event.
3. Call `list_schedules` to confirm what was set, and report the
   list back — including the consent reminder.

### V1 limits

* The app must be running for a schedule to fire (minimised to
  the tray counts — the process is alive). A schedule whose time
  passed while the app was closed expires on next startup; it
  never fires late.
* No end time: a scheduled recording stops only when the user
  presses Stop. The auto-detect stop gates never end one.
* If a recording is already running when a schedule fires, the
  schedule is skipped — never double-started.
* One-off only: no recurring schedules in v1.
* A due schedule whose time is up to 5 minutes past still fires
  (covers sleep/wake); older than that, it expires instead.

### Transcript variants

Three artifacts, never merged: the verbatim transcript
(`transcript.json` / `transcript.txt`), the deterministic cleaned
note (`transcript-clean.txt`) and the enterprise-cleaned text
(`transcript-enterprise-clean.txt`). `get_transcript` picks
between them with `variant`: `verbatim` (default), `clean` or
`enterprise`. The registry spellings `original` and `cleaned` are
accepted as aliases for `verbatim` and `clean`. Every result
reports `variant` (the MCP name), `variant_label` and `note`; the
enterprise result's note states it was produced by an external
LLM (Claude Code CLI, `claude -p`) from transcript text only —
no audio ever leaves the machine — and that the verbatim
transcript is the record.

### Dictation notes

There is no dictation tool: dictation sessions are ordinary
sessions with `source: "solo"` and `dictation: true` in their
metadata, so the existing tools already reach them. Find them
with `list_sessions(source="solo")` or
`search_meetings(source="solo")` (every session entry carries
`dictation`, so a `solo` session that is not dictation is told
apart from one that is), and read the cleaned note — the primary
readable artifact, capturing the spoken side only — with
`get_transcript(session_id, variant="clean")`.

### Bookmarks and pins

`list_bookmarks` reads the session's `metadata.json` bookmarks
(`{t, label, created}`) through the same code the Session view
uses, with `audio_available` computed at list time; `get_session`
embeds the same list. Pinned (starred) sessions are exposed in
every session entry (`pinned: true/false`) and with the
`pinned_only` filter on `list_sessions`.

### Recording source

Every session carries `source` (enum: `teams`, `zoom`, `webex`,
`meet`, `browser`, `solo`, `claude-code`, `other`) and
`source_detail` (app display name, window title, or user note) in
its `metadata.json`. When undetermined the source is `other`,
never a guess. `list_sessions` and `search_meetings` accept an
optional `source` filter (exact enum value; omit or null = no
filtering); `get_session` returns both fields. Sessions written
before the source field existed have no `source` key and read as
`other` with `source_detail: null` — they are not crashed on and
no value is invented for them.

## How it is built

* `app/mcp/tools.py` — the twelve tools as plain functions over an
  output root. No `mcp` import here, so the logic is testable
  without the SDK and reuses the existing code: `search_sessions` /
  `hybrid_search` (`app/storage/browser.py`), `list_session_summaries`
  / `summarize_session`, `load_transcript_json`,
  `SpeakerDatabase`, `get_speaker_detail`
  (`app/speakers/people.py`), `list_bookmarks`
  (`app/storage/bookmarks.py`), the transcript-variant registry
  (`app/transcription/cleanup.py`), and `OllamaClient`
  (`app/exports/local_llm.py`).
  Nothing in this module writes a file, a database row or a
  setting — except the three scheduling tools, which write only
  the shared schedule store file. The one nuance: a semantic search maintains
  `search_index.db` (the same index maintenance the Recording
  Browser already does on the same code path) — index maintenance
  for a read, not a change to any recording, transcript or speaker.
* `app/mcp/server.py` — wires the tools into the official MCP
  Python SDK (`mcp` 2.x, `MCPServer`, stdio transport) plus the
  `offair-mcp` console-script entry point. This is the only module
  that imports `mcp`, and it does so lazily, so the rest of OffAir
  never requires the SDK.
* `main.py --mcp` — the frozen-build entry point (`OffAir.exe --mcp`).
* `pyproject.toml` — dev-install metadata for the `offair-mcp`
  console script only; the distributed app is still the PyInstaller
  bundle (`OffAir.spec` collects the `mcp` package for `--mcp`).

### SDK choice

The official MCP Python SDK (`mcp` 2.3.0) — **not** hand-rolled
JSON-RPC. It installed cleanly on Python 3.12 (`pip install mcp`,
~34 s, all wheels) and the v2 API is small: `MCPServer`,
`@server.tool()`, `server.run(transport="stdio")`. Note the v2
rename: `FastMCP` became `MCPServer` (`from
mcp.server.mcpserver import MCPServer`); v1 `FastMCP` examples
need adapting, and result fields are snake_case in v2
(`result.is_error`, not `isError`). The SDK is pure Python plus
binary wheels for `pydantic-core` and `cryptography` (both ship
Windows + manylinux wheels — no compiler needed on the build
machine); there is no native code of the SDK's own to bundle.
`OffAir.spec` adds `collect_all("mcp")` plus the SDK submodules as
hidden imports. The frozen `--mcp` path is verified in every
release build (see "Still to verify" for what remains open).

### Threading note (a real bug found in verification)

The SDK runs each sync tool on a worker thread
(`anyio.to_thread`), and the thread varies per call. The first
version cached one `SemanticSearchIndex` on the tools object; its
single `sqlite3` connection is bound to the thread that created
it, so the second semantic search that landed on another thread
crashed with `sqlite3.ProgrammingError`. The fix is fresh-per-call
indexes (`_new_index` / `_release_index` in `app/mcp/tools.py`):
one connection per call, closed afterwards — cheap and always
thread-correct. Covered by
`test_semantic_tools_survive_different_calling_threads`.

## Verify it

Run the tool tests from a checkout with the SDK installed:

```bash
# Unit tests (the stdio test skips without `mcp`):
PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_mcp.py -q

# Manual check against demo sessions (all tools, real SDK client):
PYTHONPATH=$PWD <venv>/bin/python tests/manual_mcp.py \
    --output-dir <recordings-folder>
```

Or point any MCP client at `python -m app.mcp --output-dir
<recordings>` and call the tools by hand. Demo sessions are
built by the repo's screenshot tooling from the synthetic
fixture (see `docs/SCREENSHOTS.md`); the seed scripts live
outside the repo, so a fresh checkout has no demo data until
you record or transcribe a session of your own.

MCP Inspector (needs node/npx): launch it and add the server
with command `python`, args `-m app.mcp --output-dir
<recordings>` (run from the repo root so the `app` package
resolves), transport stdio:

```bash
npx @modelcontextprotocol/inspector
```

## Measured (this VM, 61.3 s 3-speaker demo session)

* Server idle: ~49 MB RSS, ~0 CPU, zero sockets, no listening socket.
* After a hybrid search (MiniLM model load + onnxruntime): ~82 MB
  RSS, ~1 s CPU total for the search.
* Tool latency on the demo session: hybrid search ~0.85 s
  (first call, includes model load), `ask_meetings` fallback
  ~0.17 s, `get_transcript` ~0.004 s.

## Still to verify

* **Frozen build**: `OffAir.exe --mcp` on Windows is verified in
  every release build — the frozen MCP smoke test lists all
  twelve tools and exercises the scheduling tools against a
  temporary schedule store — but a run against a large real
  library on a user's machine has not been done.
* **`ask_meetings` with Ollama actually running**: verified only
  with Ollama down (fallback path) and with a fake client in unit
  tests. Needs one run with `ollama serve` + `qwen3.5:9b` to
  confirm the generated-answer path end to end.
* **Larger libraries**: timings above are for one 61 s session;
  semantic search loads all chunk embeddings into memory (~1.5 KB
  per chunk — see `docs/SEMANTIC_SEARCH.md`).
