"""MCP setup page (Feature 6, Nov 10 2026).

A view in the main window ("MCP setup") that tells the user, in plain
language, what the MCP server is (read-only, except the three
scheduling tools — Nov 22 2026), shows the exact command
to run it for THIS install, gives copy-paste config snippets for
Claude Code, Claude Desktop, and Codex, runs an on-demand status
check, and lists the twelve tools the server exposes.

Why a view, not a Settings section: the Settings dialog is a long
scrolling form for persisted settings; MCP setup is not a setting —
it is a setup/status page (copy buttons, a check that runs on
demand, a tool list). A nav entry sits one click away while the app
is running, and the existing geometry test already audits every
view at 1280x800.

The Qt-free helpers (command computation, snippet building, status
check) live here so they are unit-testable without PyQt6; the
widget is a thin view over them. Importing this module never
requires Qt; constructing :class:`McpSetupWidget` raises a clear
RuntimeError when Qt is absent.
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import AppConfig, load_settings
from app.ui.level_meter import QT_AVAILABLE

if QT_AVAILABLE:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )


# ------------------------------------------------------------ plain data ----
INTRO_TEXT = (
    "The MCP server lets AI assistants like Claude read your recorded "
    "meetings — search transcripts, list sessions, and answer questions "
    "about past meetings — and schedule future recordings. It runs on "
    "this machine only, talks to the client over a local connection, "
    "not the network, and is read-only EXCEPT for scheduling: the "
    "three schedule tools create, list and cancel recording "
    "schedules, and nothing else. They never record, change or "
    "delete a transcript, session, or setting."
)

PRIVACY_TEXT = (
    "Privacy: any client connected to this server can read every "
    "transcript, so only connect clients you trust. A connected "
    "client can also schedule recordings: a scheduled recording "
    "starts automatically and does not trigger Teams' recording "
    "banner, so tell participants you are recording."
)

# The twelve tools, one plain-language line each. The wording here
# is the user-facing version of docs/MCP.md's table — keep them in
# sync. The three schedule tools are the ONE exception to the
# read-only rule (Nov 22 2026): they write only the shared
# schedule store file, never transcripts, sessions, or settings.
TOOL_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("search_meetings",
     "Search your meetings by words, by meaning, or both."),
    ("list_sessions",
     "List recorded sessions, newest first."),
    ("get_transcript",
     "Read one session's full transcript."),
    ("get_session",
     "Show one session's details — when it was, how long, who spoke."),
    ("list_bookmarks",
     "A session's bookmarks, earliest first: each marked moment, "
     "its label, and whether its audio still exists."),
    ("list_speakers",
     "List the speakers OffAir has learned to recognise."),
    ("get_speaker",
     "Everything about one speaker: their sessions, everything "
     "they said, and their voice sample, if one still exists."),
    ("what_did_speaker_say",
     "Find everything one speaker said, across all sessions."),
    ("ask_meetings",
     "Answer a question from your meetings, using the local AI "
     "when it is running — otherwise it returns the matching "
     "excerpts and says so."),
    ("schedule_recording",
     "SCHEDULES a recording (the one write tool): a future time, "
     "an app, and optional meeting details. It starts "
     "automatically while OffAir is running."),
    ("list_schedules",
     "List scheduled recordings — pending ones and recent "
     "history — with their status."),
    ("cancel_schedule",
     "Cancel one pending scheduled recording."),
)

# Default install folder for the frozen app (OffAir.iss:
# DefaultDirName={autopf}\OffAir). Used only to NAME the folder in
# the hint line when running from source — never as a path the app
# trusts without checking it exists first.
DEFAULT_INSTALL_DIR = r"C:\Program Files\OffAir"


@dataclass(frozen=True)
class ServerCommand:
    """The real commands for this install — computed, never guessed.

    ``mode`` is "frozen" (running from the installed OffAir.exe) or
    "source" (running from a repo checkout). ``run_command`` is the
    exact command line for this run; ``claude_code`` /
    ``claude_desktop`` are the ready-to-paste JSON config objects.
    """

    mode: str
    run_command: str
    # Frozen only: the path to the running exe.
    exe_path: str | None
    # Source only: the repo root (cwd for `python -m app.mcp`).
    repo_root: str | None
    # Dev-install console script, when one is on PATH.
    offair_mcp: str | None
    claude_code: dict
    claude_desktop: dict


def _quote(path: str) -> str:
    """Quote a path for a command line when it contains a space."""
    return f'"{path}"' if " " in path else path


def compute_server_command() -> ServerCommand:
    """The MCP server commands for the current run.

    Frozen (``sys.frozen`` set by PyInstaller): the running exe +
    ``--mcp`` — the exe path is real, taken from ``sys.executable``.
    From source: ``python -m app.mcp`` with the real repo root as
    the working directory. The ``offair-mcp`` console script is
    reported when it is actually installed (``shutil.which``).
    """
    repo_root = str(Path(__file__).resolve().parents[2])
    # Best-effort: the console-script lookup must never break window
    # construction — e.g. tests that fake sys.platform as "win32"
    # make shutil.which raise (it dispatches on the platform).
    try:
        offair_mcp = shutil.which("offair-mcp")
    except Exception:  # noqa: BLE001 — "not found" is the fallback
        offair_mcp = None
    if getattr(sys, "frozen", False):
        exe = str(Path(sys.executable).resolve())
        config = {
            "mcpServers": {
                "offair": {"command": exe, "args": ["--mcp"]},
            },
        }
        return ServerCommand(
            mode="frozen",
            run_command=f"{_quote(exe)} --mcp",
            exe_path=exe,
            repo_root=None,
            offair_mcp=offair_mcp,
            claude_code=config,
            claude_desktop=config,
        )
    python = sys.executable
    config = {
        "mcpServers": {
            "offair": {
                "command": python,
                "args": ["-m", "app.mcp"],
                "cwd": repo_root,
            },
        },
    }
    return ServerCommand(
        mode="source",
        run_command=f"{_quote(python)} -m app.mcp",
        exe_path=None,
        repo_root=repo_root,
        offair_mcp=offair_mcp,
        claude_code=config,
        claude_desktop=config,
    )


def claude_code_snippet(command: ServerCommand) -> str:
    """The Claude Code ``mcpServers`` JSON for this install."""
    return json.dumps(command.claude_code, indent=2)


def claude_desktop_snippet(command: ServerCommand) -> str:
    """The Claude Desktop ``mcpServers`` JSON for this install."""
    return json.dumps(command.claude_desktop, indent=2)


def _toml_str(value: str) -> str:
    """A TOML basic string: escape backslashes and double quotes."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def codex_snippet(command: ServerCommand) -> str:
    """The Codex ``[mcp_servers.offair]`` TOML for this install.

    Same server as the Claude snippets, in Codex's config format
    (``~/.codex/config.toml``). Frozen: the running exe + ``--mcp``;
    from source: ``python -m app.mcp`` with the repo root as ``cwd``.
    """
    if command.mode == "frozen":
        return "\n".join([
            "[mcp_servers.offair]",
            f"command = {_toml_str(command.exe_path or '')}",
            'args = ["--mcp"]',
        ])
    return "\n".join([
        "[mcp_servers.offair]",
        f"command = {_toml_str(sys.executable)}",
        'args = ["-m", "app.mcp"]',
        f"cwd = {_toml_str(command.repo_root or '')}",
    ])


def alternative_commands_hint(command: ServerCommand) -> str:
    """The other ways to run the server, with real paths where known.

    The installed-app variant is named with the documented default
    install folder when the exe path is not known on this machine
    (source run) — never a bare "PATH" placeholder.
    """
    if command.mode == "frozen":
        parts = [
            "Dev install instead: run `python -m app.mcp` from the "
            "repo root",
        ]
        if command.offair_mcp:
            parts.append(f"or `{command.offair_mcp}`")
        else:
            parts.append("or `offair-mcp` (after `pip install -e .`)")
        return " ".join(parts) + "."
    installed = (
        f"{DEFAULT_INSTALL_DIR}\\OffAir.exe --mcp"
        if command.exe_path is None
        else f"{_quote(command.exe_path)} --mcp"
    )
    parts = [
        "Installed app instead: run "
        f"`{installed}`"
        f" (default install folder: {DEFAULT_INSTALL_DIR}).",
    ]
    if command.offair_mcp:
        parts.append(f"Console script: `{command.offair_mcp}`.")
    else:
        parts.append("Console script: `offair-mcp` "
                     "(after `pip install -e .`).")
    return " ".join(parts)


# ------------------------------------------------------------ status check --
@dataclass(frozen=True)
class StatusLine:
    """One status-check line: did it pass, and what it says."""

    ok: bool
    text: str


def check_mcp_status(output_root: str | Path,
                     import_fn=None) -> list[StatusLine]:
    """Check the MCP server can run against ``output_root``.

    Four checks, each saying WHAT is wrong instead of failing
    silently: the tools module imports, the ``mcp`` SDK is
    installed, the recordings folder exists, and how many sessions
    would be exposed. ``import_fn`` is injectable for tests (a fake
    that raises ImportError for "mcp" simulates the SDK-missing
    state); production passes nothing.
    """
    import_fn = import_fn or importlib.import_module
    root = Path(output_root)
    lines: list[StatusLine] = []

    try:
        import_fn("app.mcp.tools")
        lines.append(StatusLine(
            True, "MCP tools module imports cleanly."))
    except Exception as exc:  # noqa: BLE001 — the message IS the report
        lines.append(StatusLine(
            False, f"Cannot import the MCP tools module: {exc}"))
    try:
        import_fn("mcp")
        lines.append(StatusLine(
            True, "MCP Python SDK (mcp) is installed."))
    except Exception:  # noqa: BLE001 — any import failure is "missing"
        lines.append(StatusLine(
            False, "mcp package not installed — run "
                   "pip install -r requirements.txt"))
    if root.is_dir():
        lines.append(StatusLine(
            True, f"Recordings folder found at {root}."))
    else:
        lines.append(StatusLine(
            False, f"Recordings folder not found at {root} — check "
                   "the output folder in Settings."))
        return lines
    try:
        from app.storage.sessions import list_sessions

        count = len(list_sessions(root))
        lines.append(StatusLine(
            True,
            f"{count} session(s) would be exposed to a connected "
            "client." if count else
            "No sessions yet — nothing would be exposed until you "
            "record a meeting."))
    except Exception as exc:  # noqa: BLE001 — the message IS the report
        lines.append(StatusLine(
            False, f"Could not list sessions in {root}: {exc}"))
    return lines


def status_text(lines: list[StatusLine]) -> str:
    """The status lines as one display string (✓ / ✗ per line)."""
    return "\n".join(
        f"{'✓' if line.ok else '✗'} {line.text}" for line in lines)


# ------------------------------------------------------------ the widget ----
class McpSetupWidget(QWidget if QT_AVAILABLE else object):  # type: ignore[misc]
    """The MCP setup page (a view's content — the view wraps it in a
    scroll area, like the Session view)."""

    def __init__(self, config: AppConfig | None = None,
                 parent=None) -> None:
        if not QT_AVAILABLE:
            raise RuntimeError(
                "PyQt6 is not installed — install it with "
                "`pip install -r requirements.txt` to use the GUI.")
        super().__init__(parent)
        self._config = config or load_settings()
        self._command = compute_server_command()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("MCP setup — let AI assistants read your meetings")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        intro = QLabel(INTRO_TEXT)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        privacy = QLabel(PRIVACY_TEXT)
        privacy.setObjectName("modeLabel")
        privacy.setWordWrap(True)
        layout.addWidget(privacy)

        layout.addWidget(self._field_label("Run the server:"))
        command_row = QHBoxLayout()
        command_row.setSpacing(8)
        self.command_edit = QLineEdit(self._command.run_command)
        self.command_edit.setReadOnly(True)
        self.command_edit.setToolTip(
            "The exact command that starts the server for this "
            "install. The client runs it for you — you only need "
            "this if you start the server by hand.")
        command_row.addWidget(self.command_edit, 1)
        self.copy_command_btn = QPushButton("Copy")
        self.copy_command_btn.clicked.connect(
            lambda: self._copy(self.command_edit.text(),
                               self.copy_command_btn))
        command_row.addWidget(self.copy_command_btn)
        layout.addLayout(command_row)

        alt_hint = QLabel(alternative_commands_hint(self._command))
        alt_hint.setObjectName("hint")
        alt_hint.setWordWrap(True)
        layout.addWidget(alt_hint)

        layout.addWidget(self._field_label(
            "Claude Code — paste into your MCP settings:"))
        code_hint = QLabel(
            "Add this to the \"mcpServers\" object in your Claude "
            "Code settings (or run `claude mcp add-json offair "
            "<this JSON>`).")
        code_hint.setObjectName("hint")
        code_hint.setWordWrap(True)
        layout.addWidget(code_hint)
        self.claude_code_edit = QPlainTextEdit(
            claude_code_snippet(self._command))
        self.claude_code_edit.setReadOnly(True)
        self._size_snippet_edit(self.claude_code_edit)
        layout.addWidget(self.claude_code_edit)
        self.copy_code_btn = QPushButton("Copy Claude Code config")
        self.copy_code_btn.clicked.connect(
            lambda: self._copy(self.claude_code_edit.toPlainText(),
                               self.copy_code_btn))
        layout.addWidget(self.copy_code_btn)

        layout.addWidget(self._field_label(
            "Claude Desktop — paste into claude_desktop_config.json:"))
        desktop_hint = QLabel(
            "Add this to the \"mcpServers\" object in the Claude "
            "Desktop config file (Windows: "
            "%APPDATA%\\Claude\\claude_desktop_config.json), then "
            "restart Claude Desktop.")
        desktop_hint.setObjectName("hint")
        desktop_hint.setWordWrap(True)
        layout.addWidget(desktop_hint)
        self.claude_desktop_edit = QPlainTextEdit(
            claude_desktop_snippet(self._command))
        self.claude_desktop_edit.setReadOnly(True)
        self._size_snippet_edit(self.claude_desktop_edit)
        layout.addWidget(self.claude_desktop_edit)
        self.copy_desktop_btn = QPushButton("Copy Claude Desktop config")
        self.copy_desktop_btn.clicked.connect(
            lambda: self._copy(
                self.claude_desktop_edit.toPlainText(),
                self.copy_desktop_btn))
        layout.addWidget(self.copy_desktop_btn)

        layout.addWidget(self._field_label(
            "Codex — add to your Codex config:"))
        codex_hint = QLabel(
            "Add this to the \"[mcp_servers]\" section of your Codex "
            "config file (`~/.codex/config.toml`), then restart Codex.")
        codex_hint.setObjectName("hint")
        codex_hint.setWordWrap(True)
        layout.addWidget(codex_hint)
        self.codex_edit = QPlainTextEdit(
            codex_snippet(self._command))
        self.codex_edit.setReadOnly(True)
        self._size_snippet_edit(self.codex_edit)
        layout.addWidget(self.codex_edit)
        self.copy_codex_btn = QPushButton("Copy Codex config")
        self.copy_codex_btn.clicked.connect(
            lambda: self._copy(self.codex_edit.toPlainText(),
                               self.copy_codex_btn))
        layout.addWidget(self.copy_codex_btn)

        layout.addWidget(self._field_label("Status:"))
        self.check_btn = QPushButton("Check status")
        self.check_btn.setObjectName("primaryButton")
        self.check_btn.clicked.connect(self.refresh_status)
        layout.addWidget(self.check_btn)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        layout.addWidget(self._field_label(
            "What a connected client can do (read-only, except the "
            "three scheduling tools — see docs/MCP.md):"))
        for name, description in TOOL_DESCRIPTIONS:
            tool_label = QLabel(f"• {name} — {description}")
            tool_label.setWordWrap(True)
            layout.addWidget(tool_label)

        docs_hint = QLabel(
            "Full documentation, including every tool's inputs and "
            "outputs: docs/MCP.md in the OffAir repo.")
        docs_hint.setObjectName("hint")
        docs_hint.setWordWrap(True)
        layout.addWidget(docs_hint)

        layout.addStretch(1)
        # Run the check once so the page opens with an answer, not a
        # blank — the button re-runs it on demand.
        self.refresh_status()

    @staticmethod
    def _field_label(text: str) -> "QLabel":
        label = QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    @staticmethod
    def _size_snippet_edit(edit: "QPlainTextEdit") -> None:
        """Size a read-only snippet box to its content (no scrolling
        inside the page's own scroll area)."""
        lines = edit.document().lineCount()
        height = (edit.fontMetrics().lineSpacing() * lines
                  + edit.contentsMargins().top()
                  + edit.contentsMargins().bottom() + 16)
        edit.setFixedHeight(height)

    def set_output_root(self, output_root: str | Path) -> None:
        """Point the status check at a new recordings folder (called
        when Settings move the output folder)."""
        self._config.output_root = Path(output_root)
        self.refresh_status()

    def refresh_status(self) -> None:
        """Run the status check on demand and show the result."""
        lines = check_mcp_status(self._config.output_root)
        self.status_label.setText(status_text(lines))

    def _copy(self, text: str, button: "QPushButton") -> None:
        """Copy ``text`` to the clipboard; confirm on the button."""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        button.setText("Copied!")
        QTimer.singleShot(1500, lambda: button.setText(
            "Copy" if button is self.copy_command_btn
            else ("Copy Claude Code config"
                  if button is self.copy_code_btn
                  else ("Copy Codex config"
                        if button is self.copy_codex_btn
                        else "Copy Claude Desktop config"))))


__all__ = [
    "INTRO_TEXT",
    "PRIVACY_TEXT",
    "TOOL_DESCRIPTIONS",
    "ServerCommand",
    "StatusLine",
    "alternative_commands_hint",
    "check_mcp_status",
    "claude_code_snippet",
    "claude_desktop_snippet",
    "codex_snippet",
    "compute_server_command",
    "status_text",
    "McpSetupWidget",
]
