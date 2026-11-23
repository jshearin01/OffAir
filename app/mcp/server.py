"""MCP transport wiring — official MCP Python SDK, stdio only.

This is the only module in :mod:`app.mcp` that imports ``mcp``.
The import is done lazily inside :func:`build_server`, so importing
:mod:`app.mcp.tools` (and the rest of OffAir) never requires the SDK
— a machine without ``mcp`` installed gets a clear error only when
it actually tries to start the server (``pip install -r
requirements.txt``).

Transport: stdio, via ``MCPServer.run(transport="stdio")``. No
HTTP, SSE or streamable-HTTP transport is ever started — the server
listens on nothing. The SDK's stdio transport re-points stray
stdout at stderr while serving, but the tools themselves never
print; logging goes to ``offair.log`` in the output root (see
:mod:`app.logging_setup`) and, when a console exists, to stderr.

PyInstaller: the frozen build needs the SDK bundled — see
``OffAir.spec`` (``collect_all("mcp")`` plus the SDK's pure-Python
dependencies). The SDK is pure Python plus binary wheels for
``pydantic-core`` and ``cryptography`` (both ship manylinux /
Windows wheels on PyPI, so pip installs them without a compiler);
there is no native code of the SDK's own to bundle.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import get_default_output_root, load_settings
from app.mcp.tools import OffAirMCPTools

logger = logging.getLogger("offair.mcp.server")

SERVER_NAME = "offair"
SERVER_INSTRUCTIONS = (
    "OffAir — access to the local meeting archive, plus scheduled "
    "recording. Nine tools are read-only: search sessions, read "
    "transcripts (including the cleaned and enterprise-cleaned "
    "variants), list learned speakers and per-speaker detail, "
    "read bookmarks, and ask questions about past meetings. "
    "Three scheduling tools WRITE, and only to the schedule store "
    "file (~/.offair/schedules.json): schedule_recording creates "
    "a one-off timer that starts a recording automatically while "
    "the app is running, list_schedules reads schedules, and "
    "cancel_schedule cancels a pending one. Nothing else here "
    "writes: no tool records, transcribes, renames, deletes or "
    "edits a session, transcript or setting. Scheduled auto-start "
    "does NOT trigger Teams' recording banner — participants must "
    "be told. ask_meetings answers from the local Ollama model "
    "when it is running; otherwise it returns the most relevant "
    "transcript excerpts and says so."
)

# Tool schemas are derived from the function signatures by the SDK.
# All twelve tools return a JSON string (one JSON object per call)
# so every client — including ones without structured-output
# support — gets the same parseable payload.


def build_server(output_root: str | Path,
                 tools: OffAirMCPTools | None = None):
    """Build the MCP server bound to ``output_root``.

    Raises RuntimeError with install guidance when the ``mcp``
    package is not installed.
    """
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise RuntimeError(
            "The MCP Python SDK is not installed — run "
            "`pip install -r requirements.txt` (it includes `mcp`) "
            "and retry.") from exc

    tools = tools or OffAirMCPTools(output_root)
    server = MCPServer(name=SERVER_NAME, instructions=SERVER_INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True)

    @server.tool(annotations=read_only)
    def search_meetings(query: str, mode: str = "hybrid",
                        limit: int = 10,
                        source: str | None = None,
                        speaker: str | None = None) -> str:
        """Search recorded meetings by text, meaning, or both.

        mode: "text" (substring, always offline), "semantic"
        (meaning-based), or "hybrid" (exact hits first, then
        semantic — the default). source: optional recording-source
        filter (teams, zoom, webex, meet, browser, solo,
        claude-code, other; omit for all). speaker: optional
        speaker filter — only sessions that speaker spoke in
        (display name, case-insensitive exact match; omit for
        all). Returns {mode, note, results} with one entry per
        matching session (session_id, start_time, speakers,
        source, source_detail, pinned, dictation, score,
        match_type, snippet). Falls back to text search, and says
        so in "note", when the semantic model is unavailable.
        """
        return json.dumps(tools.search_meetings(query, mode=mode,
                                                limit=limit,
                                                source=source,
                                                speaker=speaker))

    @server.tool(annotations=read_only)
    def list_sessions(limit: int = 10, since: str | None = None,
                      source: str | None = None,
                      pinned_only: bool = False) -> str:
        """List recorded sessions, newest first.

        since: optional ISO date ("2026-11-01") or datetime to
        filter by session start time. source: optional
        recording-source filter (teams, zoom, webex, meet, browser,
        solo, claude-code, other; omit for all). pinned_only:
        keep only pinned (starred) sessions. Returns {results}
        with one entry per session (session_id, start_time,
        duration_seconds, app_name, status, speakers, source,
        source_detail, pinned, dictation, flags).
        """
        return json.dumps(tools.list_sessions(limit=limit, since=since,
                                              source=source,
                                              pinned_only=pinned_only))

    @server.tool(annotations=read_only)
    def get_transcript(session_id: str, format: str = "text",
                       variant: str = "verbatim") -> str:
        """Read one session's transcript.

        format "text": one "[MM:SS] Speaker: text" line per segment.
        format "segments": [{start, end, speaker, text}]. Speaker
        renames from the session are applied. variant: "verbatim"
        (the record — the default), "clean" (the deterministic
        cleaned note) or "enterprise" (the external-LLM cleanup,
        labelled as such in the result). Clean and enterprise
        variants are text only. The result reports which variant
        was returned.
        """
        return json.dumps(tools.get_transcript(session_id,
                                               format=format,
                                               variant=variant))

    @server.tool(annotations=read_only)
    def get_session(session_id: str) -> str:
        """One session's status, speakers, app, duration, source and calendar event.

        source / source_detail come from the session's metadata
        ("other" / null when the session predates the source field).
        dictation is true only for dictation sessions. bookmarks
        lists the session's bookmarks, earliest first.
        calendar_event is the linked Outlook event stored in the
        session's metadata, or null when no event matched.
        """
        return json.dumps(tools.get_session(session_id))

    @server.tool(annotations=read_only)
    def list_bookmarks(session_id: str) -> str:
        """A session's bookmarks, earliest first: each marked moment, its label, and whether its audio still exists.

        Each entry is {t, label, created, audio_available}, where
        t is seconds from the start of the session and
        audio_available is false when retention has deleted the
        session's audio. Empty when the session has no bookmarks.
        """
        return json.dumps(tools.list_bookmarks(session_id))

    @server.tool(annotations=read_only)
    def list_speakers() -> str:
        """Learned speakers from speakers.db: {name, backend, sample_count, sessions}."""
        return json.dumps(tools.list_speakers())

    @server.tool(annotations=read_only)
    def get_speaker(name: str) -> str:
        """Everything about one speaker: their sessions, everything they said, and their voice sample, if one still exists.

        Matches the speaker's display name (case-insensitive
        exact match). Returns {name, known_in_database, sessions,
        utterances, sample, sample_available, session_count,
        utterance_count}. An unknown speaker returns empty
        sessions / utterances, not an error. sample is a pointer
        back into a session folder (audio path + timestamps),
        never audio bytes.
        """
        return json.dumps(tools.get_speaker(name))

    @server.tool(annotations=read_only)
    def what_did_speaker_say(speaker: str, query: str = "",
                             limit: int = 10) -> str:
        """Everything one speaker said, across all sessions.

        Matches the speaker's display name (case-insensitive),
        newest session first. query narrows to segments containing
        that text. Returns {speaker, results} with
        {session_id, start, end, text} per segment.
        """
        return json.dumps(tools.what_did_speaker_say(
            speaker, query=query, limit=limit))

    @server.tool(annotations=read_only)
    def ask_meetings(question: str) -> str:
        """Answer a question from the meeting archive (local RAG).

        Retrieves the most relevant transcript excerpts and asks the
        local Ollama model to answer from them only. When Ollama is
        not running there is no generated answer — the response says
        so plainly (source "search-fallback", answer null) and
        returns the excerpts instead. Nothing is ever invented.
        """
        return json.dumps(tools.ask_meetings(question))

    # Scheduling tools (Nov 22 2026): the scoped write exception.
    # These are the ONLY tools that write anything, and they write
    # only the schedule store file. They are annotated
    # read_only_hint=False so clients can see the difference.
    write = ToolAnnotations(read_only_hint=False)

    @server.tool(annotations=write)
    def schedule_recording(fire_at: str, target_app: str,
                           subject: str | None = None,
                           attendees: list[str] | None = None,
                           organiser: str | None = None) -> str:
        """Schedule a one-off recording. WRITE — see server instructions.

        The app starts recording target_app automatically at
        fire_at while it is running, and stops only when the user
        presses Stop. fire_at: ISO datetime with an offset, in the
        future (naive = local wall time). target_app: exe name
        ("ms-teams.exe") or display name ("Microsoft Teams").
        subject / attendees / organiser: optional calendar details
        carried into the session's metadata. Consent: scheduled
        auto-start does NOT trigger Teams' recording banner —
        participants must be told. Returns the created schedule
        with its id. A schedule whose time passes while the app is
        closed expires instead of firing late.
        """
        return json.dumps(tools.schedule_recording(
            fire_at, target_app, subject=subject, attendees=attendees,
            organiser=organiser))

    @server.tool(annotations=read_only)
    def list_schedules(limit: int = 20) -> str:
        """Schedules: pending first (earliest first), then recent history.

        Read-only. Each entry is the stored schedule record (id,
        fire_at, target_app, subject, attendees, organiser,
        created_by, status, created, fired_at, session_id, note).
        """
        return json.dumps(tools.list_schedules(limit=limit))

    @server.tool(annotations=write)
    def cancel_schedule(schedule_id: str) -> str:
        """Cancel a pending schedule. WRITE — see server instructions.

        Errors for an unknown id or a schedule that is no longer
        pending. Returns the cancelled schedule.
        """
        return json.dumps(tools.cancel_schedule(schedule_id))

    return server


def run_mcp_server(output_root: str | Path,
                   tools: OffAirMCPTools | None = None) -> int:
    """Run the MCP server on stdio until the client disconnects."""
    output_root = Path(output_root)
    logger.info("OffAir MCP server starting (stdio) on %s",
                output_root)
    server = build_server(output_root, tools=tools)
    server.run(transport="stdio")
    return 0


def main(argv: list[str] | None = None) -> int:
    """``offair-mcp`` console-script entry point (dev installs)."""
    parser = argparse.ArgumentParser(
        prog="offair-mcp",
        description="OffAir MCP server — meeting archive (read-only) "
                    "plus scheduled recording (scheduling tools only) "
                    "over stdio.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Folder recordings are saved under "
             f"(default: {get_default_output_root()})")
    parser.add_argument(
        "--schedules-file", type=Path, default=None,
        help="Schedule store file for the scheduling tools "
             "(default: ~/.offair/schedules.json)")
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        output_root = args.output_dir
    else:
        output_root = load_settings().output_root
    # File logging only: setup_logging() also adds a console handler
    # when stderr exists, which is fine on stdio (stderr is not the
    # MCP wire) and is where a startup failure should be visible.
    from app.logging_setup import setup_logging

    setup_logging(output_root)
    try:
        tools = OffAirMCPTools(
            output_root, schedules_path=args.schedules_file)
        return run_mcp_server(output_root, tools=tools)
    except RuntimeError as exc:
        # Missing SDK: a plain message on stderr + non-zero exit,
        # not a traceback — the user just needs the install line.
        logger.error("%s", exc)
        print(f"offair-mcp: {exc}", file=sys.stderr, flush=True)
        return 2


__all__ = ["SERVER_NAME", "build_server", "main", "run_mcp_server"]
