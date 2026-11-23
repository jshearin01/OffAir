"""In-app Help / Documentation view (Feature 7, Nov 10 2026).

A view in the main window: a section list on the left, the rendered
docs/ Markdown file for the selected section on the right. The docs/
files are the single source of truth — this view renders them, it
does not duplicate their content — so the in-app help cannot drift
from the repo docs.

Docs path: resolved the same way as every other bundled file, via
:app.func:`app.resources.resource_path` (``docs/`` next to the repo
root in a source checkout, under ``sys._MEIPASS`` in a frozen build —
``docs/`` is in the spec's ``datas``, see ``OffAir.spec`` and
``docs/PACKAGING.md``). A missing file never crashes the app: the
pane shows a plain "not included in this install" message instead.
That matters for frozen installs where docs/ may not be bundled, and
for a future docs deletion — which the test suite also catches
loudly (``tests/test_ui_help.py`` asserts every listed section has
a docs/ source file).

The view needs no Qt at import time for its data: :data:`HELP_SECTIONS`
and :func:`docs_dir` are plain Python, so the section inventory is
unit-testable without PyQt6.
"""
from __future__ import annotations

from pathlib import Path

from app.resources import docs_dir as _bundled_docs_dir

# (section title, docs/ filename) — one section per topic, in the
# order a new user meets the app. The filename is the single source
# of truth for that section's content; every file must exist under
# docs/ (tests/test_ui_help.py enforces this).
HELP_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Getting started", "OVERVIEW.md"),
    ("Recording", "PER_APP_CAPTURE.md"),
    ("Scheduled recordings", "SCHEDULED_RECORDING.md"),
    ("Transcription", "TRANSCRIPTION.md"),
    ("Diarization", "DIARIZATION.md"),
    ("Speakers & renaming", "SPEAKERS.md"),
    ("Search", "SEMANTIC_SEARCH.md"),
    ("Calendar view", "CALENDAR_VIEW.md"),
    ("Outlook linking", "OUTLOOK_LINKING.md"),
    ("Retention", "RETENTION.md"),
    ("Transcript cleanup & names", "TRANSCRIPT_CLEANUP.md"),
    ("Keyboard shortcuts", "KEYBOARD_SHORTCUTS.md"),
    ("MCP server", "MCP.md"),
    ("Privacy", "PRIVACY.md"),
)

#: Shown when a section's docs/ file is missing (frozen install
#: without docs/ bundled, or a file deleted after the section list
#: was written). Plain text, never a crash, never an empty page.
MISSING_DOC_MESSAGE = (
    "This help page is not included in this install.\n\n"
    "The documentation file {filename} could not be found under "
    "{docs_dir}. The full documentation is in the docs/ folder of "
    "the OffAir repository."
)


def docs_dir() -> Path:
    """Absolute path to the bundled docs/ folder (frozen or source).

    Re-exported from :mod:`app.resources` so help-view callers have
    one obvious name for it.
    """
    return _bundled_docs_dir()


def section_source(filename: str, base: Path | None = None) -> Path:
    """Path to one section's docs/ source file."""
    return (base if base is not None else docs_dir()) / filename


def render_section(filename: str, base: Path | None = None) -> str:
    """Markdown text for a section, or the graceful missing-doc message.

    Never raises for a missing/unreadable file — the caller renders
    whatever this returns.
    """
    path = section_source(filename, base)
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return MISSING_DOC_MESSAGE.format(
            filename=filename, docs_dir=path.parent)


def create_help_tab(docs_base: Path | None = None):
    """Build the Help view widget (section list + rendered Markdown).

    ``docs_base`` is the test seam: a folder holding the docs/ files
    to render (defaults to the bundled docs/ — see :func:`docs_dir`).
    Raises RuntimeError when PyQt6 is absent, like the other UI
    builders. The returned widget exposes ``section_list`` (the
    QListWidget) and ``doc_browser`` (the QTextBrowser) for tests.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QHBoxLayout,
        QListWidget,
        QSplitter,
        QTextBrowser,
        QWidget,
    )

    base = docs_base if docs_base is not None else docs_dir()

    tab = QWidget()
    layout = QHBoxLayout(tab)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(12)

    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.setChildrenCollapsible(False)

    section_list = QListWidget()
    section_list.setObjectName("helpSectionList")
    section_list.setMinimumWidth(180)
    section_list.setMaximumWidth(260)
    for title, filename in HELP_SECTIONS:
        section_list.addItem(title)
        section_list.item(section_list.count() - 1).setData(
            Qt.ItemDataRole.UserRole, filename)
    splitter.addWidget(section_list)

    doc_browser = QTextBrowser()
    doc_browser.setObjectName("helpDocBrowser")
    doc_browser.setReadOnly(True)
    # Links inside the docs point at other repo files (e.g.
    # docs/MCP.md) — they are not openable from inside the app, so
    # never hand them to a browser; a click does nothing.
    doc_browser.setOpenLinks(False)
    splitter.addWidget(doc_browser)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)

    def _show_section(row: int) -> None:
        item = section_list.item(row)
        if item is None:
            return
        filename = item.data(Qt.ItemDataRole.UserRole)
        doc_browser.setMarkdown(render_section(filename, base))

    section_list.currentRowChanged.connect(_show_section)
    # Select the first section so the tab is never an empty pane.
    section_list.setCurrentRow(0)

    layout.addWidget(splitter)
    tab.section_list = section_list  # type: ignore[attr-defined]
    tab.doc_browser = doc_browser  # type: ignore[attr-defined]
    tab.docs_base = base  # type: ignore[attr-defined]
    return tab
