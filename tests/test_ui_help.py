"""In-app Help / Documentation view (Feature 7, Nov 10 2026).

Two halves, like test_ui_visual.py:

* Qt-free: every listed section has a docs/ source file (so a
  future docs deletion fails loudly instead of silently emptying
  the page); a missing file renders the graceful message, never a
  crash; the spec really bundles docs/ for the frozen build; and
  the two promises the Help tab makes in words — Outlook linking
  is read-only, Ollama is localhost-only — are actually in the
  docs they render.
* Qt-gated (``pytest.importorskip("PyQt6")`` per test — run with
  the gui-shots venv, offscreen): every section renders non-empty
  Markdown in both themes, a missing docs/ folder shows the
  graceful message with no crash, and the Help tab is
  geometry-clean at 1280x800 in both themes.

Run the Qt half with::

    QT_QPA_PLATFORM=offscreen \\
      ~/workspace/meettrace-tryouts/gui-shots/.venv/bin/python \\
      -m pytest tests/test_ui_help.py -q
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.resources import docs_dir as bundled_docs_dir
from app.ui.help_view import (
    HELP_SECTIONS,
    MISSING_DOC_MESSAGE,
    docs_dir,
    render_section,
    section_source,
)

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "OffAir.spec"


# ------------------------------------------------------------ Qt-free -------
def test_every_section_has_a_docs_source_file():
    """A deleted docs/ file must fail loudly, not empty the page.

    HELP_SECTIONS is the single source of truth for what the Help
    tab lists; each entry must point at a real file under docs/.
    """
    assert HELP_SECTIONS, "no Help sections listed at all"
    titles = [title for title, _ in HELP_SECTIONS]
    assert len(titles) == len(set(titles)), "duplicate section title"
    filenames = [name for _, name in HELP_SECTIONS]
    assert len(filenames) == len(set(filenames)), "duplicate source file"
    for title, filename in HELP_SECTIONS:
        path = section_source(filename)
        assert path.is_file(), (
            f"Help section {title!r} has no docs/ source: {path} "
            "(restore the file or remove the section from "
            "app/ui/help_view.py HELP_SECTIONS)")
        assert path.read_text(encoding="utf-8").strip(), (
            f"Help section {title!r} source is empty: {path}")


def test_docs_dir_resolves_to_the_repo_docs_folder():
    # Source checkout: resource_path("docs") is <repo>/docs.
    assert docs_dir() == REPO / "docs"
    assert bundled_docs_dir() == docs_dir()


def test_missing_file_renders_the_graceful_message_not_a_crash():
    text = render_section("DOES_NOT_EXIST.md")
    assert "not included in this install" in text
    assert "DOES_NOT_EXIST.md" in text
    # The message template itself, not a traceback or an empty page.
    assert text.startswith(MISSING_DOC_MESSAGE.split("{filename}")[0])


def test_spec_bundles_docs_for_the_frozen_build():
    """OffAir.spec must carry docs/ in datas, or the frozen Help tab
    is an empty page with a polite apology instead of the docs."""
    spec = SPEC.read_text(encoding="utf-8")
    assert '("docs", "docs")' in spec


def test_outlook_section_says_explicitly_it_never_writes():
    """The read-only Outlook promise is the point of the section —
    assert the words are in the source file, not just the intent."""
    text = (REPO / "docs" / "OUTLOOK_LINKING.md").read_text(
        encoding="utf-8")
    assert "never writes anything into Outlook" in text


def test_privacy_section_states_ollama_is_localhost_only():
    text = (REPO / "docs" / "PRIVACY.md").read_text(encoding="utf-8")
    assert "localhost" in text
    assert "off by default" in text


def test_calendar_doc_describes_the_shipped_view():
    """CALENDAR_VIEW.md describes the shipped calendar view (the
    view landed Nov 11 2026): no "as approved" banner any more,
    and the behaviours it lists — month grid, per-recording badge,
    click to load, prev/next + Today — are what the app does."""
    text = (REPO / "docs" / "CALENDAR_VIEW.md").read_text(
        encoding="utf-8")
    assert "as approved" not in text
    for required in ("month grid", "Click a recording", "Today"):
        assert required in text, f"calendar doc missing: {required}"


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


def _select_section(tab, row: int, qapp) -> str:
    tab.section_list.setCurrentRow(row)
    qapp.processEvents()
    return tab.doc_browser.toPlainText()


def test_every_section_renders_nonempty_in_both_themes(qapp):
    """Each section renders its docs/ Markdown (non-empty, not the
    missing-file message) in the light and dark themes."""
    from PyQt6.QtWidgets import QApplication
    from app.ui.help_view import create_help_tab
    from app.ui.theme import apply_theme_to_app

    app = QApplication.instance()
    try:
        for theme in ("light", "dark"):
            apply_theme_to_app(app, theme)
            tab = create_help_tab()
            try:
                for row, (title, _filename) in enumerate(HELP_SECTIONS):
                    text = _select_section(tab, row, qapp)
                    assert text.strip(), f"{theme}: {title!r} rendered empty"
                    assert "not included in this install" not in text, (
                        f"{theme}: {title!r} rendered the missing-doc "
                        "message despite its file existing")
                    # Markdown actually rendered: the file's first
                    # heading shows as plain text, not "# ..." raw.
                    assert not text.lstrip().startswith("#"), (
                        f"{theme}: {title!r} shows raw Markdown")
            finally:
                tab.close()
    finally:
        apply_theme_to_app(app, "light")


def test_missing_docs_folder_shows_graceful_message_no_crash(
        qapp, tmp_path):
    """A frozen install without docs/ bundled: every section shows
    the plain message, the tab stays usable, nothing raises."""
    from app.ui.help_view import create_help_tab

    empty = tmp_path / "no-docs-here"
    empty.mkdir()
    tab = create_help_tab(docs_base=empty)
    try:
        for row, (title, filename) in enumerate(HELP_SECTIONS):
            text = _select_section(tab, row, qapp)
            assert "not included in this install" in text, (
                f"{title!r} did not show the graceful message")
            assert filename in text
    finally:
        tab.close()


def test_help_tab_geometry_clean_at_1280x800_both_themes(
        qapp, tmp_path):
    """The Help view inside the main window: zero overlaps/clips
    at the 1280x800 target size, both themes — same audit as
    test_ui_visual.py."""
    from PyQt6.QtWidgets import QApplication
    from app.config import AppConfig
    from app.ui.main_window import create_main_window
    from app.ui.theme import apply_theme_to_app
    from test_ui_visual import _geometry_problems

    app = QApplication.instance()
    problems: list[str] = []
    try:
        for theme in ("light", "dark"):
            apply_theme_to_app(app, theme)
            win = create_main_window(
                AppConfig(output_root=tmp_path, theme=theme))
            try:
                win.show()
                qapp.processEvents()
                assert win.width() == 1280 and win.height() == 800
                win.show_view("Help")
                qapp.processEvents()
                problems.extend(_geometry_problems(
                    win, label=f"{theme}/Help"))
                # And after selecting the longest section (Recording):
                tab = win.help_tab
                tab.section_list.setCurrentRow(1)
                qapp.processEvents()
                problems.extend(_geometry_problems(
                    win, label=f"{theme}/Help/Recording"))
            finally:
                win.close()
    finally:
        apply_theme_to_app(app, "light")
    assert problems == [], "\n".join(problems)


def test_main_window_has_help_view_last(qapp, tmp_path):
    """The Help view exists and is the last nav entry — reference
    material after the record → process → review flow, not in the
    middle of it."""
    from app.config import AppConfig
    from app.ui.main_window import create_main_window

    win = create_main_window(AppConfig(output_root=tmp_path))
    try:
        assert win.tabs.tabText(win.tabs.count() - 1) == "Help"
        assert hasattr(win, "help_tab")
    finally:
        win.close()
