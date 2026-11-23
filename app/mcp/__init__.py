"""MCP server for OffAir — local, stdio (Feature 1), read-only
except scheduling (Nov 22 2026).

Exposes the meeting archive to MCP clients (Claude Code, Claude
Desktop, ...) over stdio. Nine archive tools are read-only: there
is no write, delete, rename, record, transcribe or
summarise-and-save operation among them, and the server opens no
network socket — it reads local session folders, ``speakers.db``
and ``search_index.db`` only. (``ask_meetings`` may talk to a
local Ollama server on ``http://localhost:11434`` when one is
running; when it is not, it falls back to plain search results
and says so — see :mod:`app.mcp.tools`.) The three scheduling
tools are the ONE scoped exception: they write only the shared
schedule store file (``~/.offair/schedules.json``) — see
``docs/MCP.md`` for the flagged exception and the consent note.

Privacy: any client connected to this server can read every
transcript in the output folder. Only connect clients you trust —
see ``docs/MCP.md``.

Structure (the ``mcp`` SDK is only needed for the transport):

* :mod:`app.mcp.tools` — the twelve tools as plain functions over
  an output root. No ``mcp`` import here, so the logic is testable
  without the SDK and reusable by any other caller.
* :mod:`app.mcp.server` — wires the tools into the official MCP
  Python SDK (``mcp`` package, ``MCPServer``) on stdio, plus the
  ``offair-mcp`` console-script entry point.
* ``python main.py --mcp`` — the frozen-build entry point
  (``OffAir.exe --mcp``).
"""
from __future__ import annotations

from app.mcp.tools import OffAirMCPTools

__all__ = ["OffAirMCPTools"]
