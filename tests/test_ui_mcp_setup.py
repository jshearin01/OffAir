"""MCP setup page tests (Feature 6, Nov 10 2026).

Qt-free: the snippets carry the real computed command (no
placeholders), the status check reports correctly with a
recordings root present / missing / empty, and with the ``mcp``
SDK missing.

Qt-gated (``pytest.importorskip("PyQt6")`` per test — run with the
gui-shots venv, offscreen): the Copy buttons put the snippet on
the clipboard, the status label follows the output folder, and
the new "MCP setup" view is geometry-clean at 1280x800 in both
themes (reusing the audit from test_ui_visual.py).

Run the Qt half with::

    QT_QPA_PLATFORM=offscreen \
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \
      -m pytest tests/test_ui_mcp_setup.py -q
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

from app.ui.mcp_setup import (
    TOOL_DESCRIPTIONS,
    ServerCommand,
    alternative_commands_hint,
    check_mcp_status,
    claude_code_snippet,
    claude_desktop_snippet,
    codex_snippet,
    compute_server_command,
    status_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TOOLS = (
    "search_meetings",
    "list_sessions",
    "get_transcript",
    "get_session",
    "list_bookmarks",
    "list_speakers",
    "get_speaker",
    "what_did_speaker_say",
    "ask_meetings",
    "schedule_recording",
    "list_schedules",
    "cancel_schedule",
)


# ------------------------------------------------------------ Qt-free -------
def test_snippets_carry_the_real_command_no_placeholders():
    """Both snippets contain the computed command for THIS run —
    the real python/exe path and the real repo root — and no
    placeholder tokens anywhere."""
    command = compute_server_command()
    for snippet in (claude_code_snippet(command),
                    claude_desktop_snippet(command)):
        assert "TODO" not in snippet
        assert "PLACEHOLDER" not in snippet
        assert "<" not in snippet and ">" not in snippet
        assert "PATH" not in snippet
        if command.mode == "frozen":
            assert command.exe_path is not None
            assert command.exe_path in snippet
            assert "--mcp" in snippet
        else:
            assert command.repo_root is not None
            assert command.repo_root in snippet
            assert str(Path(sys.executable)) in snippet
            assert "-m" in snippet and "app.mcp" in snippet
    # The run command shown on the page is the same real command.
    assert command.run_command
    assert "TODO" not in command.run_command
    assert "<" not in command.run_command


def test_codex_snippet_is_valid_toml():
    """The Codex snippet parses as TOML and names the same server
    as the Claude snippets — same command, same args, no
    placeholders."""
    tomllib = pytest.importorskip("tomllib")

    command = compute_server_command()
    snippet = codex_snippet(command)
    assert "TODO" not in snippet
    assert "PLACEHOLDER" not in snippet
    data = tomllib.loads(snippet)
    server = data["mcp_servers"]["offair"]
    if command.mode == "frozen":
        assert server["command"] == command.exe_path
        assert server["args"] == ["--mcp"]
    else:
        assert server["command"] == sys.executable
        assert server["args"] == ["-m", "app.mcp"]
        assert server["cwd"] == command.repo_root


def test_codex_snippet_escapes_windows_paths():
    """A Windows exe path round-trips through the TOML: the
    backslashes are escaped in the text and parse back to the
    original path."""
    tomllib = pytest.importorskip("tomllib")

    exe = (r"C:\Users\john\AppData\Local\Programs\OffAir"
           r"\OffAir.exe")
    fake = ServerCommand(
        mode="frozen",
        run_command=f'"{exe}" --mcp',
        exe_path=exe,
        repo_root=None,
        offair_mcp=None,
        claude_code={},
        claude_desktop={},
    )
    snippet = codex_snippet(fake)
    assert "\\\\" in snippet  # the path is escaped, not bare
    data = tomllib.loads(snippet)
    assert data["mcp_servers"]["offair"]["command"] == exe
    assert data["mcp_servers"]["offair"]["args"] == ["--mcp"]


def test_alternative_commands_hint_names_real_paths():
    """The hint line never shows a bare placeholder: the installed
    app is named with its documented default folder, and the dev
    equivalents are the real commands."""
    hint = alternative_commands_hint(compute_server_command())
    assert "TODO" not in hint
    assert "PLACEHOLDER" not in hint
    assert "<install-dir>" not in hint
    assert "PATH" not in hint
    assert "OffAir.exe --mcp" in hint


def test_tool_list_is_the_twelve_server_tools():
    """The page's plain-language list names exactly the twelve tools
    the server exposes (see app/mcp/server.py) — the nine
    read-only tools plus the three scheduling tools, the one
    scoped write exception (Nov 22 2026)."""
    assert [name for name, _ in TOOL_DESCRIPTIONS] == list(EXPECTED_TOOLS)
    for _name, description in TOOL_DESCRIPTIONS:
        assert description and description.strip() == description
        assert len(description) > 10  # a real line, not a stub


def test_status_check_with_present_empty_root(tmp_path):
    lines = check_mcp_status(tmp_path)
    text = status_text(lines)
    assert "MCP tools module imports cleanly" in text
    assert "Recordings folder found at" in text
    assert "No sessions yet" in text
    assert "not found" not in text


def test_status_check_counts_sessions(tmp_path):
    from app.storage.sessions import create_session

    create_session(tmp_path, app_name="Teams")
    create_session(tmp_path, app_name="Teams")
    text = status_text(check_mcp_status(tmp_path))
    assert "2 session(s) would be exposed" in text


def test_status_check_with_missing_root(tmp_path):
    missing = tmp_path / "no-such-folder"
    lines = check_mcp_status(missing)
    text = status_text(lines)
    assert "Recordings folder not found at" in text
    assert str(missing) in text
    # A missing root is reported, not silently skipped — and the
    # session count is not attempted after it.
    assert "would be exposed" not in text
    assert any(not line.ok for line in lines)


def test_status_check_reports_sdk_missing(tmp_path):
    """The SDK-missing state says what to install (simulated by a
    fake importer that raises ImportError for "mcp")."""
    def _fake_import(name):
        if name == "mcp":
            raise ImportError("No module named 'mcp'")
        return importlib.import_module(name)

    text = status_text(check_mcp_status(tmp_path, import_fn=_fake_import))
    assert "mcp package not installed" in text
    assert "pip install -r requirements.txt" in text


def test_status_check_reports_broken_tools_module(tmp_path):
    def _fake_import(name):
        if name == "app.mcp.tools":
            raise ImportError("boom")
        return importlib.import_module(name)

    text = status_text(check_mcp_status(tmp_path, import_fn=_fake_import))
    assert "Cannot import the MCP tools module" in text
    assert "boom" in text


# ------------------------------------------------------------ Qt-gated ------
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/offair-demo/runtime")
    Path("/tmp/offair-demo/runtime").mkdir(parents=True, exist_ok=True)
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _make_widget(tmp_path, qapp):
    from app.config import AppConfig
    from app.ui.mcp_setup import McpSetupWidget

    widget = McpSetupWidget(AppConfig(output_root=tmp_path))
    widget.show()
    qapp.processEvents()
    return widget


def test_copy_buttons_put_snippets_on_clipboard(qapp, tmp_path):
    from PyQt6.QtWidgets import QApplication

    from app.ui.mcp_setup import (
        claude_code_snippet,
        claude_desktop_snippet,
        codex_snippet,
        compute_server_command,
    )

    widget = _make_widget(tmp_path, qapp)
    try:
        command = compute_server_command()
        clipboard = QApplication.clipboard()

        widget.copy_code_btn.click()
        qapp.processEvents()
        assert clipboard.text() == claude_code_snippet(command)
        assert clipboard.text() == widget.claude_code_edit.toPlainText()

        widget.copy_desktop_btn.click()
        qapp.processEvents()
        assert clipboard.text() == claude_desktop_snippet(command)
        assert clipboard.text() == widget.claude_desktop_edit.toPlainText()

        widget.copy_codex_btn.click()
        qapp.processEvents()
        assert clipboard.text() == codex_snippet(command)
        assert clipboard.text() == widget.codex_edit.toPlainText()

        widget.copy_command_btn.click()
        qapp.processEvents()
        assert clipboard.text() == command.run_command
        assert clipboard.text() == widget.command_edit.text()
    finally:
        widget.close()


def test_status_label_follows_output_folder(qapp, tmp_path):
    from app.storage.sessions import create_session

    widget = _make_widget(tmp_path, qapp)
    try:
        assert "No sessions yet" in widget.status_label.text()
        create_session(tmp_path, app_name="Teams")
        widget.check_btn.click()
        qapp.processEvents()
        assert "1 session(s) would be exposed" in widget.status_label.text()

        widget.set_output_root(tmp_path / "gone")
        qapp.processEvents()
        assert "Recordings folder not found" in widget.status_label.text()
        assert str(tmp_path / "gone") in widget.status_label.text()
    finally:
        widget.close()


def test_mcp_view_present_in_main_window(qapp, tmp_path):
    from PyQt6.QtWidgets import QLabel

    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        labels = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        assert "MCP setup" in labels
        win.show_view("MCP setup")
        qapp.processEvents()
        assert win.mcp_setup_widget is not None
        texts = [lbl.text() for lbl in
                 win.mcp_setup_widget.findChildren(QLabel)]
        assert any("only connect clients you trust" in t for t in texts)
    finally:
        win.close()


def test_mcp_view_geometry_clean_at_1280x800_both_themes(
        qapp, tmp_path):
    """Zero overlaps/clips on the new view, both themes, at the
    1280x800 target size (same audit as test_ui_visual.py)."""
    from PyQt6.QtWidgets import QApplication

    from app.config import AppConfig
    from app.ui.main_window import create_main_window
    from app.ui.theme import apply_theme_to_app
    from test_ui_visual import _geometry_problems

    app = QApplication.instance()
    problems: list[str] = []
    for theme in ("light", "dark"):
        apply_theme_to_app(app, theme)
        win = create_main_window(
            AppConfig(output_root=tmp_path, theme=theme))
        try:
            win.show()
            qapp.processEvents()
            assert win.width() == 1280 and win.height() == 800
            win.show_view("MCP setup")
            qapp.processEvents()
            problems.extend(_geometry_problems(
                win, label=f"{theme}/MCP setup"))
        finally:
            win.close()
    apply_theme_to_app(app, "light")
    assert problems == [], "\n".join(problems)
