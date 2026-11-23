"""``python -m app.mcp`` — run the MCP server on stdio (read-only
except the scheduling tools — see docs/MCP.md)."""
from __future__ import annotations

from app.mcp.server import main

if __name__ == "__main__":
    raise SystemExit(main())
