"""More views — MCP setup and Help, out of the daily path.

Both stay reachable from the left navigation as their own rows
(flat nav since Nov 20 2026 — no "More" grouping), not hidden:
MCP setup is its own nav entry with the exact server command and
the Codex snippet, and Help renders the repo docs in-app. The
module name is historical; the builders are unchanged.
"""
from __future__ import annotations

from app.ui.level_meter import QT_AVAILABLE

if QT_AVAILABLE:
    from PyQt6.QtWidgets import QScrollArea, QWidget

from app.ui.help_view import create_help_tab


def build_mcp_view(win) -> QWidget:
    """Build the MCP setup page (scrollable, like the Session view)."""
    from app.ui.mcp_setup import McpSetupWidget

    win.mcp_setup_widget = McpSetupWidget(win.config)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(win.mcp_setup_widget)
    return scroll


def build_help_view(win) -> QWidget:
    """Build the Help page (in-app docs, rendered from docs/)."""
    win.help_tab = create_help_tab()
    return win.help_tab
