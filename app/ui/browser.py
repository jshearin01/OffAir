"""Recording Browser UI (PRD Phase 5).

A window listing every recorded session (newest first), with a
search box over transcript text and speaker names. Clicking a
session shows its transcript; buttons export the LLM-ready
Markdown and open the session folder.

Importing this module never requires PyQt6; constructing the
browser raises a clear RuntimeError when Qt is absent.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.config import AppConfig, DEFAULT_SEARCH_MODE, SEARCH_MODES
from app.exports.llm_export import export_llm_markdown_from_session
from app.storage.browser import (
    SessionSummary,
    hybrid_search,
    list_session_summaries,
    list_sources,
    search_sessions,
    session_transcript_text,
)
from app.storage.source import SOURCE_LABELS
from app.storage.semantic_search import (
    ModelNotAvailableError,
    SemanticHit,
    SemanticSearchIndex,
)
from app.transcription.engine import TRANSCRIPT_JSON
from app.ui.level_meter import QT_AVAILABLE

logger = logging.getLogger("offair.ui.browser")


def format_session_line(summary: SessionSummary) -> str:
    """Browser entry (pure function, no Qt needed).

    Two lines from :func:`app.storage.browser.library_row_lines`:
    the display name plus date, then duration / speakers / source
    with a status note only when it matters. The list widget shows
    both lines; the Library delegate draws the first bold and the
    second muted.
    """
    from app.storage.browser import library_row_lines
    title, meta = library_row_lines(summary)
    return f"{title}\n{meta}"


def format_search_hit_line(hit: SemanticHit) -> str:
    """One-line semantic/hybrid hit (pure function, no Qt needed)."""
    return hit.display()


if QT_AVAILABLE:
    from PyQt6.QtCore import QSize, Qt
    from PyQt6.QtWidgets import (
        QComboBox,
        QDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QPushButton,
        QSplitter,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    class RecordingBrowser(QDialog):
        """List sessions, search them, view a transcript, export it.

        Phase 6 layout: a header (title + search + mode), then a
        horizontal splitter — sessions on the left, the selected
        session's transcript on the right — with export + status
        in a footer. The search behaviour (text default, semantic
        / hybrid via the mode picker) is unchanged.
        """

        def __init__(self, config: AppConfig | None = None,
                     parent=None) -> None:
            super().__init__(parent)
            self.setWindowTitle("OffAir — Recording Browser")
            self.setMinimumSize(900, 560)
            self.resize(1100, 700)
            self.config = config or AppConfig()
            self._summaries: list[SessionSummary] = []
            # One entry per list row: the session that row belongs
            # to (text mode: the row IS the session; semantic /
            # hybrid: one row per chunk hit, all pointing at their
            # session). Kept parallel to session_list rows.
            self._row_summaries: list[SessionSummary] = []
            self._hits: list[SemanticHit] = []

            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)
            title = QLabel("Recordings")
            title.setObjectName("pageTitle")
            layout.addWidget(title)
            hint = QLabel(
                "Sessions (newest first). Text search matches "
                "transcript text, speaker names, session id and app "
                "name; Semantic finds meaning (\"money\" finds the "
                "budget discussion); Hybrid puts exact matches first. "
                "Select a session and open it in the main window to "
                "work on it — transcript, speakers and processing "
                "all follow the opened session.")
            hint.setObjectName("hint")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            search_row = QHBoxLayout()
            search_row.setSpacing(10)
            self.search_edit = QLineEdit()
            self.search_edit.setPlaceholderText("Search recordings…")
            self.search_edit.setClearButtonEnabled(True)
            self.search_edit.setMinimumHeight(40)
            self.search_edit.textChanged.connect(self._on_search_changed)
            search_row.addWidget(self.search_edit, 1)
            self.mode_combo = QComboBox()
            self.mode_combo.setMinimumHeight(40)
            self.mode_combo.addItem(
                "Text — exact words (fast, always works offline)",
                "text")
            self.mode_combo.addItem(
                "Semantic — meaning (downloads the MiniLM model "
                "once, ~22 MB)", "semantic")
            self.mode_combo.addItem(
                "Hybrid — exact matches first, then meaning "
                "(recommended)", "hybrid")
            self.mode_combo.setCurrentIndex(
                self.mode_combo.findData(
                    self.config.search_mode
                    if self.config.search_mode in SEARCH_MODES
                    else DEFAULT_SEARCH_MODE))
            self.mode_combo.currentIndexChanged.connect(
                self._on_mode_changed)
            search_row.addWidget(self.mode_combo)
            # Source filter (Nov 10 2026): "All sources" plus one
            # entry per source actually present in the library
            # (rebuilt on every refresh — see
            # _refresh_source_options). It narrows the list in
            # every search mode; the calendar view and the MCP
            # server read the same metadata keys.
            self.source_combo = QComboBox()
            self.source_combo.setMinimumHeight(40)
            self.source_combo.setToolTip(
                "Show only sessions recorded from one source — "
                "Teams, Zoom, a browser, a Claude Code session… "
                "The source is set at record time (auto-detected "
                "from the app, or picked on the Record card) and "
                "can be corrected per session from the main "
                "window's Session view.")
            self.source_combo.addItem("All sources", None)
            self.source_combo.currentIndexChanged.connect(
                self._on_source_changed)
            search_row.addWidget(self.source_combo)
            # Speaker filter (Nov 21 2026): "All speakers" plus
            # one entry per speaker who spoke in the library
            # (rebuilt on every refresh — see
            # _refresh_speaker_options). It narrows the list in
            # every search mode, like the source filter.
            self.speaker_combo = QComboBox()
            self.speaker_combo.setMinimumHeight(40)
            self.speaker_combo.setToolTip(
                "Show only sessions where one speaker spoke — "
                "display name, exact match.")
            self.speaker_combo.addItem("All speakers", None)
            self.speaker_combo.currentIndexChanged.connect(
                self._on_speaker_changed)
            search_row.addWidget(self.speaker_combo)
            layout.addLayout(search_row)

            splitter = QSplitter(Qt.Orientation.Horizontal)
            left = QWidget()
            left_layout = QVBoxLayout(left)
            # Breathing room around the splitter handle — the two
            # panes used to sit hard against it.
            left_layout.setContentsMargins(0, 0, 8, 0)
            sessions_label = QLabel("Sessions:")
            sessions_label.setObjectName("fieldLabel")
            left_layout.addWidget(sessions_label)
            self.session_list = QListWidget()
            # Long session lines wrap instead of clipping behind a
            # horizontal scrollbar.
            self.session_list.setWordWrap(True)
            self.session_list.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.session_list.itemClicked.connect(self._on_session_clicked)
            self.session_list.itemSelectionChanged.connect(
                self._update_open_button)
            self.session_list.itemDoubleClicked.connect(
                self._on_open_in_main_window)
            left_layout.addWidget(self.session_list)
            splitter.addWidget(left)
            right = QWidget()
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(8, 0, 0, 0)
            transcript_label = QLabel("Transcript:")
            transcript_label.setObjectName("fieldLabel")
            right_layout.addWidget(transcript_label)
            self.transcript_view = QTextEdit()
            self.transcript_view.setReadOnly(True)
            right_layout.addWidget(self.transcript_view)
            splitter.addWidget(right)
            splitter.setSizes([380, 720])
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 2)
            layout.addWidget(splitter, 1)

            self.status_label = QLabel("")
            self.status_label.setObjectName("hint")
            layout.addWidget(self.status_label)
            # Footer actions, side by side: opening the selected
            # session in the main window is the primary action
            # (the library's job is jumping into a recording);
            # exporting its LLM Markdown is the secondary one.
            footer_row = QHBoxLayout()
            footer_row.setSpacing(12)
            self.open_btn = QPushButton("Open in main window")
            self.open_btn.setObjectName("primaryButton")
            self.open_btn.setMinimumHeight(40)
            self.open_btn.setToolTip(
                "Load the selected session into the main window — "
                "its transcript, speakers and Process actions all "
                "follow that session.")
            self.open_btn.clicked.connect(self._on_open_in_main_window)
            footer_row.addWidget(self.open_btn, 1)
            self.export_btn = QPushButton(
                "Export selected session as LLM Markdown")
            self.export_btn.setMinimumHeight(40)
            self.export_btn.clicked.connect(self._on_export_llm)
            footer_row.addWidget(self.export_btn, 1)
            layout.addLayout(footer_row)
            self._refresh()
            self._update_open_button()

        def showEvent(self, event) -> None:  # noqa: N802 - Qt name
            super().showEvent(event)
            # Rows are sized at the real (shown) width — see
            # _relayout_session_rows.
            self._relayout_session_rows()

        def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
            super().resizeEvent(event)
            # Splitter drags change the list width, so wrapped rows
            # need re-measuring (a row sized for a wide pane elides
            # when the pane narrows).
            self._relayout_session_rows()

        def _row_height_for(self, text: str) -> int:
            """Row height that fits ``text`` wrapped at the list width.

            Qt's item delegate under-measures wrapped rows: its size
            hint counts fewer lines than it paints (the Nov 10
            render showed the status line elided — "Speakers …" —
            because the row was sized for 3 lines and the text needs
            5). So rows get an explicit height from a greedy
            word-wrap count: at a conservative width (the paint
            wraps ~60 px earlier than the viewport suggests) plus
            two spare lines, so a row is never clipped by a
            one-line miscount.
            """
            fm = self.session_list.fontMetrics()
            width = max(120, self.session_list.viewport().width() - 100)
            lines = 0
            for paragraph in text.split("\n"):
                lines += 1
                line_width = 0
                for word in paragraph.split(" "):
                    word_width = fm.horizontalAdvance(word + " ")
                    if line_width + word_width > width and line_width > 0:
                        lines += 1
                        line_width = word_width
                    else:
                        line_width += word_width
            # 16 px item padding (QSS) + one line per wrapped line +
            # two spare lines (see the docstring).
            return 16 + (lines + 2) * fm.lineSpacing()

        def _relayout_session_rows(self) -> None:
            """Apply :meth:`_row_height_for` to every session row."""
            for row in range(self.session_list.count()):
                item = self.session_list.item(row)
                item.setSizeHint(QSize(
                    self.session_list.viewport().width(),
                    self._row_height_for(item.text())))

        # ------------------------------------------------------------ data ---
        def _selected_source(self) -> str | None:
            """The source filter's value, or None for "All sources"."""
            return self.source_combo.currentData()

        def _selected_speaker(self) -> str | None:
            """The speaker filter's value, or None for "All speakers"."""
            return self.speaker_combo.currentData()

        def _format_row(self, summary: SessionSummary) -> str:
            """A session row: starred sessions carry a star prefix."""
            text = format_session_line(summary)
            return f"★ {text}" if summary.pinned else text

        def _source_status_note(self) -> str:
            """``" — source: Teams"`` when a filter is active, else ""."""
            selected = self._selected_source()
            if selected is None:
                return ""
            return f" — source: {SOURCE_LABELS.get(selected, selected)}"

        def _refresh_source_options(self) -> None:
            """Rebuild the filter dropdown from the library's sources.

            Only sources that actually occur are offered (plus
            "All sources"). The current pick is kept when it is
            still present; otherwise the filter resets to All
            instead of silently filtering to nothing.
            """
            previous = self._selected_source()
            options = list_sources(Path(self.config.output_root))
            self.source_combo.blockSignals(True)
            try:
                self.source_combo.clear()
                self.source_combo.addItem("All sources", None)
                for value in options:
                    self.source_combo.addItem(
                        SOURCE_LABELS.get(value, value), value)
                index = self.source_combo.findData(previous)
                self.source_combo.setCurrentIndex(
                    index if index >= 0 else 0)
            finally:
                self.source_combo.blockSignals(False)

        def _refresh_speaker_options(self) -> None:
            """Rebuild the speaker filter from the library's speakers.

            Only speakers who spoke in at least one session are
            offered (plus "All speakers"). The current pick is
            kept when it is still present; otherwise the filter
            resets to All instead of silently filtering to nothing.
            """
            previous = self._selected_speaker()
            try:
                summaries = list_session_summaries(
                    Path(self.config.output_root))
            except Exception:  # noqa: BLE001 — filter must not break
                summaries = []
            speakers: list[str] = []
            for summary in summaries:
                for name in summary.speakers:
                    if name not in speakers:
                        speakers.append(name)
            self.speaker_combo.blockSignals(True)
            try:
                self.speaker_combo.clear()
                self.speaker_combo.addItem("All speakers", None)
                for name in sorted(speakers):
                    self.speaker_combo.addItem(name, name)
                index = self.speaker_combo.findData(previous)
                self.speaker_combo.setCurrentIndex(
                    index if index >= 0 else 0)
            finally:
                self.speaker_combo.blockSignals(False)

        def _refresh(self, query: str = "") -> None:
            root = Path(self.config.output_root)
            mode = self.mode_combo.currentData() or DEFAULT_SEARCH_MODE
            self._refresh_source_options()
            self._refresh_speaker_options()
            selected_source = self._selected_source()
            selected_speaker = self._selected_speaker()
            self._summaries = []
            self._row_summaries = []
            self._hits = []
            self.session_list.clear()
            if not query:
                self._summaries = list_session_summaries(
                    root, source=selected_source,
                    speaker=selected_speaker)
                self._row_summaries = list(self._summaries)
                for summary in self._summaries:
                    self.session_list.addItem(self._format_row(summary))
                if self._summaries:
                    self.status_label.setText(
                        f"{len(self._summaries)} session(s) in {root}"
                        f"{self._source_status_note()}")
                else:
                    self.status_label.setText(
                        "No sessions yet — record your first meeting "
                        "and it will appear here."
                        if selected_source is None else
                        f"No {SOURCE_LABELS.get(selected_source, '')} "
                        "sessions yet.")
                return
            if mode == "text":
                self._summaries = search_sessions(
                    root, query, mode="text",
                    source=selected_source,
                    speaker=selected_speaker)
                self._row_summaries = list(self._summaries)
                for summary in self._summaries:
                    self.session_list.addItem(self._format_row(summary))
                if self._summaries:
                    self.status_label.setText(
                        f"{len(self._summaries)} session(s) match in {root}"
                        f"{self._source_status_note()}")
                else:
                    self.status_label.setText(
                        f"No sessions match “{query}” — try different "
                        "words, or switch to Semantic search.")
                return
            # Semantic / hybrid: first use downloads the MiniLM
            # model (~22 MB) and builds the index — say so first,
            # so the window does not look frozen.
            self.status_label.setText(
                "Building search index — first use downloads the "
                "MiniLM model (~22 MB)…")
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            try:
                index = SemanticSearchIndex(
                    root, model_dir=self.config.semantic_model_dir)
                if mode == "hybrid":
                    hits = hybrid_search(root, query, index=index)
                else:
                    hits = index.semantic_search(query)
            except ModelNotAvailableError as exc:
                # Offline (or download failed): say so plainly and
                # fall back to the substring search that always
                # works, instead of showing an empty list.
                # (search_sessions in hybrid mode falls back to
                # text internally, so this cannot raise.)
                self._summaries = search_sessions(
                    root, query, mode="text",
                    source=selected_source,
                    speaker=selected_speaker)
                self._row_summaries = list(self._summaries)
                for summary in self._summaries:
                    self.session_list.addItem(self._format_row(summary))
                if self._summaries:
                    self.status_label.setText(
                        f"Semantic search unavailable: {exc} "
                        f"Showing {len(self._summaries)} text "
                        "match(es) instead.")
                else:
                    self.status_label.setText(
                        f"No sessions match “{query}” — Semantic "
                        f"search unavailable: {exc}")
                return
            if selected_source is not None or selected_speaker is not None:
                # Semantic / hybrid hits are per-chunk, not
                # per-session, so the source/speaker filters apply
                # here — a chunk only counts when its session
                # matches. Speaker matching is the display-name
                # exact match the search core uses.
                by_id = {s.session_id: s
                         for s in list_session_summaries(root)}
                wanted = (selected_speaker or "").strip().lower()
                hits = [h for h in hits
                        if by_id.get(h.session_id) is not None
                        and (selected_source is None
                             or by_id[h.session_id].source == selected_source)
                        and (not wanted or any(
                            name.strip().lower() == wanted
                            for name in by_id[h.session_id].speakers))]
            self._hits = hits
            summaries = {s.session_id: s
                         for s in list_session_summaries(root)}
            for hit in hits:
                hit_text = format_search_hit_line(hit)
                hit_summary = summaries.get(hit.session_id)
                if hit_summary is not None and hit_summary.pinned:
                    hit_text = f"★ {hit_text}"
                self.session_list.addItem(hit_text)
                # Keep one row-summary per list row, even if the
                # session folder vanished mid-search.
                self._row_summaries.append(
                    summaries.get(hit.session_id) or SessionSummary(
                        session_id=hit.session_id,
                        path=root / hit.session_id))
            sessions = {h.session_id for h in hits}
            if hits:
                self.status_label.setText(
                    f"{len(hits)} chunk(s) in {len(sessions)} session(s) "
                    f"— {mode} search{self._source_status_note()}")
            else:
                self.status_label.setText(
                    f"No matches for “{query}” in {mode.capitalize()} "
                    "search — try different words, or switch to Text "
                    "search.")

        def _on_search_changed(self, text: str) -> None:
            self._refresh(query=text)

        def _on_source_changed(self, _index: int) -> None:
            self._refresh(query=self.search_edit.text())

        def _on_speaker_changed(self, _index: int) -> None:
            self._refresh(query=self.search_edit.text())

        def _on_mode_changed(self, _index: int) -> None:
            self._refresh(query=self.search_edit.text())

        def _selected_summary(self) -> SessionSummary | None:
            row = self.session_list.currentRow()
            if row < 0 or row >= len(self._row_summaries):
                return None
            return self._row_summaries[row]

        def _can_open_in_main_window(self) -> bool:
            """True when a session is selected AND the parent window
            can load one (the main window's ``load_session``).

            A standalone browser (no parent, or a parent without
            ``load_session``) has nowhere to open into, so the
            button stays disabled with the reason in its tooltip
            instead of failing on click.
            """
            if self._selected_summary() is None:
                return False
            parent = self.parent()
            return callable(getattr(parent, "load_session", None))

        def _update_open_button(self) -> None:
            """Enable Open only for a selected session + a loadable parent."""
            self.open_btn.setEnabled(self._can_open_in_main_window())
            if self._selected_summary() is None:
                self.open_btn.setToolTip("Select a session first.")
            elif not callable(getattr(self.parent(), "load_session",
                                      None)):
                self.open_btn.setToolTip(
                    "Open the Recordings browser from the main window "
                    "to load a session into it.")
            else:
                self.open_btn.setToolTip(
                    "Load the selected session into the main window — "
                    "its transcript, speakers and Process actions all "
                    "follow that session.")

        def _on_open_in_main_window(self, _item=None) -> None:
            """Load the selected session into the main window.

            The browser closes; the main window stays on the opened
            session (its ``load_session`` sets the session and
            refreshes every view for it). Double-clicking a session
            row does the same thing.
            """
            summary = self._selected_summary()
            if summary is None:
                self.status_label.setText("Select a session first.")
                return
            parent = self.parent()
            load = getattr(parent, "load_session", None)
            if not callable(load):
                self.status_label.setText(
                    "Cannot open here — this browser was not opened "
                    "from the main window.")
                return
            try:
                load(summary.path)
            except Exception as exc:  # noqa: BLE001 — surfaced in-browser
                logger.exception("Opening session in main window failed")
                self.status_label.setText(f"Could not open session: {exc}")
                return
            self.accept()

        def _on_session_clicked(self, _item) -> None:
            summary = self._selected_summary()
            if summary is None:
                return
            text = session_transcript_text(summary.path)
            if not text:
                has_json = (summary.path / TRANSCRIPT_JSON).exists()
                text = ("(No transcript yet — transcribe this session "
                        "first.)" if not has_json else
                        "(Transcript file exists but is empty.)")
            self.transcript_view.setPlainText(text)
            self.status_label.setText(f"Showing {summary.session_id}")
            self._update_open_button()

        def _on_export_llm(self) -> None:
            summary = self._selected_summary()
            if summary is None:
                self.status_label.setText("Select a session first.")
                return
            try:
                path = export_llm_markdown_from_session(summary.path)
            except Exception as exc:  # noqa: BLE001 — surfaced to the user
                logger.exception("LLM export failed")
                self.status_label.setText(f"Export failed: {exc}")
                return
            self.status_label.setText(f"LLM Markdown written to {path}")
            self._refresh(query=self.search_edit.text())


def create_browser(config: AppConfig | None = None,
                   parent=None):
    """Build the Recording Browser, or raise if PyQt6 is missing."""
    if not QT_AVAILABLE:
        raise RuntimeError(
            "PyQt6 is not installed — install it with "
            "`pip install -r requirements.txt` to use the GUI.")
    return RecordingBrowser(config, parent)
