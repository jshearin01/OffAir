"""Recording browser data layer (PRD Phase 5).

Lists sessions newest-first with their metadata, and searches them.
No session format changes: this reads the Phase 1 layout
(``metadata.json`` + ``transcript.txt`` / ``transcript.json`` +
``speakers.json`` / ``diarization.json``) exactly as written by
Phases 1–4. Search is a plain case-insensitive substring match
over transcript text, speaker names, session id, display name
and app name.
:func:`hybrid_search` adds a meaning-based layer on top of it
(see :mod:`app.storage.semantic_search`): exact substring hits
are boosted so they rank first, and semantic hits catch
queries whose words never appear in the transcript
("money" finds the budget discussion). The default search mode is hybrid: exact substring hits first,
semantic hits after (see :func:`hybrid_search`). ``mode="text"``
is the Phase 5 behaviour, unchanged; ``mode="semantic"`` needs
the MiniLM model and raises ModelNotAvailableError from
:mod:`app.storage.semantic_search` when the model cannot be
fetched. Hybrid is best-effort: when the model is unavailable
(not installed, offline, first-use download failed) it degrades
to the text search instead of raising, so the default search
path never breaks — callers that need to *know* the model is
missing should call :func:`hybrid_search` directly.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.diarization.names import load_speaker_names
from app.storage.semantic_search import (
    HYBRID_TEXT_BOOST,
    SEMANTIC_DEFAULT_THRESHOLD,
    SEMANTIC_DEFAULT_TOP_K,
    ModelNotAvailableError,
    SemanticHit,
    SemanticSearchIndex,
)
from app.storage.sessions import (
    METADATA_FILE,
    is_calendar_linked,
    list_sessions,
    load_metadata,
    session_display_name,
)
from app.storage.source import (
    SOURCES,
    SOURCE_LABELS,
    SOURCE_OTHER,
    normalize_source,
)
from app.transcription.engine import TRANSCRIPT_JSON
from app.transcription.exports import load_transcript_json

logger = logging.getLogger("offair.storage.browser")

# Search modes for search_sessions(). "text" is the Phase 5
# behaviour, unchanged; "semantic" and "hybrid" need the MiniLM
# model (downloaded once, ~22 MB) and raise
# ModelNotAvailableError from app.storage.semantic_search when
# the model cannot be fetched.
SEARCH_MODES: tuple[str, ...] = ("text", "semantic", "hybrid")


@dataclass
class SessionSummary:
    """One session, as the Recording Browser shows it."""

    session_id: str
    path: Path
    start_time: str | None = None
    duration_seconds: float | None = None
    app_name: str | None = None
    mode: str | None = None
    status: str | None = None
    # Recording source (app.storage.source): what the session is
    # a recording of. Always present after load_metadata's
    # backfill; "other" when it could not be determined.
    source: str = SOURCE_OTHER
    source_detail: str = ""
    speakers: list[str] = field(default_factory=list)
    has_transcript: bool = False
    has_diarization: bool = False
    has_llm_export: bool = False
    has_suggestions: bool = False
    # Pinned (starred) sessions sort first in the library
    # (metadata.json "pinned": true, via update_metadata).
    pinned: bool = False
    # Display name, resolved by the one rule in
    # app.storage.sessions.session_display_name: the Outlook
    # event's subject when linked, else the custom name, else the
    # derived "<app> recording" default. calendar_event itself is
    # NOT stored here — summaries stay light; the link is read
    # from metadata.json when needed.
    display_name: str = ""
    # Whether the session is tied to an Outlook item (its name is
    # the event subject and it cannot be renamed).
    calendar_linked: bool = False

    @property
    def source_label(self) -> str:
        """Human label for :attr:`source`, e.g. ``"Teams"``."""
        return SOURCE_LABELS.get(self.source, "Other")

    @property
    def status_text(self) -> str:
        """Human status line, e.g. "Recorded · Transcribed · Diarized".

        Built only from the ``has_*`` flags (see
        :func:`summarize_session`) — "Recorded" is always true, and
        a session with no transcript yet says so plainly instead of
        showing an empty flag list. The Recording Browser shows
        this as the second line of each row and the main window's
        Session-view session chooser shows it after the session id.
        """
        if not self.has_transcript:
            return "Recorded — not transcribed yet"
        bits = ["Recorded", "Transcribed"]
        if self.has_diarization:
            bits.append("Diarized")
        if self.has_suggestions:
            bits.append("Speakers identified")
        if self.has_llm_export:
            bits.append("LLM export ready")
        return " · ".join(bits)

    @property
    def display(self) -> str:
        """One-line summary for the browser list."""
        duration = ""
        if self.duration_seconds is not None:
            total = int(self.duration_seconds)
            minutes, secs = divmod(total, 60)
            duration = f" — {minutes}m {secs:02d}s" if minutes else f" — {secs}s"
        speakers = f" — {', '.join(self.speakers)}" if self.speakers else ""
        flags = []
        if self.has_transcript:
            flags.append("transcript")
        if self.has_diarization:
            flags.append("diarized")
        if self.has_llm_export:
            flags.append("LLM export")
        flag_text = f" [{', '.join(flags)}]" if flags else ""
        name = self.display_name or self.session_id
        return (f"{name}{duration}{speakers}{flag_text}"
                f" — {self.source_label}")


def summarize_session(session_dir: str | Path) -> SessionSummary:
    """Build a SessionSummary for one session folder."""
    session_dir = Path(session_dir)
    try:
        metadata = load_metadata(session_dir)
    except FileNotFoundError:
        metadata = {}
    names = load_speaker_names(session_dir)
    speakers: list[str] = []
    transcript_path = session_dir / TRANSCRIPT_JSON
    if transcript_path.exists():
        try:
            segments = load_transcript_json(transcript_path)
            for seg in segments:
                speaker = getattr(seg, "speaker", None)
                if speaker:
                    display = names.get(speaker, speaker)
                    if display not in speakers:
                        speakers.append(display)
        except Exception:  # noqa: BLE001 — a bad transcript is not fatal
            logger.warning("Could not read transcript in %s",
                           session_dir, exc_info=True)
    # Renames with no transcript segment still count as known speakers.
    for display in names.values():
        if display not in speakers:
            speakers.append(display)
    return SessionSummary(
        session_id=metadata.get("session_id", session_dir.name),
        path=session_dir,
        start_time=metadata.get("start_time"),
        duration_seconds=metadata.get("duration_seconds"),
        app_name=metadata.get("app_name"),
        mode=metadata.get("mode"),
        status=metadata.get("status"),
        source=(normalize_source(metadata.get("source"))
                or SOURCE_OTHER),
        source_detail=metadata.get("source_detail") or "",
        speakers=speakers,
        has_transcript=transcript_path.exists(),
        has_diarization=(session_dir / "diarization.json").exists(),
        has_llm_export=(session_dir / "transcript-llm.md").exists(),
        has_suggestions=(session_dir / "speaker_suggestions.json").exists(),
        pinned=bool(metadata.get("pinned", False)),
        display_name=session_display_name(metadata),
        calendar_linked=is_calendar_linked(metadata),
    )


def row_date_text(summary: SessionSummary) -> str:
    """The date line-1 shows for a session, in local time.

    The metadata start time (UTC) formatted locally, falling back
    to the folder-name stamp (``YYYY-MM-DD``) when the start time
    is missing or unparsable.
    """
    date_text = format_local_time(summary.start_time)
    if date_text:
        return date_text
    stamp = summary.session_id[:10]
    try:
        day = datetime.strptime(stamp, "%Y-%m-%d")
    except ValueError:
        return stamp
    return day.strftime("%b %d, %Y")


def library_row_lines(summary: SessionSummary) -> tuple[str, str]:
    """The Library row's two lines: title, then muted meta.

    Line 1 is the display name plus the date (a star prefix marks
    pinned rows). Line 2 is duration, speakers and source, with a
    status note only when it matters ("Recording…" for a live
    session, "No transcript yet" for an untranscribed one) — the
    full status line lives in the Session view. Pure function, no
    Qt needed; the Library delegate draws the two lines.
    """
    title = summary.display_name or summary.session_id
    if summary.pinned:
        title = "★ " + title
    title = f"{title} — {row_date_text(summary)}"
    bits: list[str] = []
    if summary.duration_seconds is not None:
        total = int(summary.duration_seconds)
        minutes, secs = divmod(total, 60)
        bits.append(f"{minutes}m {secs:02d}s" if minutes else f"{secs}s")
    if summary.speakers:
        bits.append(", ".join(summary.speakers))
    bits.append(summary.source_label)
    if summary.status == "recording":
        bits.append("Recording…")
    elif not summary.has_transcript:
        bits.append("No transcript yet")
    return title, " • ".join(bits)


def format_local_time(iso_text: str | None) -> str:
    """Format a metadata ISO timestamp in the user's local time.

    Session folder names use local time, but ``metadata.json`` stores
    UTC (``start_time`` / ``end_time``) — displaying the UTC value
    as-is put the "Loaded session" line hours ahead of the folder
    name on non-UTC machines (Nov 10 2026). Naive values (old
    sessions written without an offset) are already local time and
    are left alone. Returns "" when the value is missing or
    unparsable, so callers can fall back to the folder-name stamp.
    """
    if not iso_text:
        return ""
    try:
        when = datetime.fromisoformat(iso_text)
    except ValueError:
        return ""
    if when.tzinfo is not None:
        when = when.astimezone()  # UTC stored -> local display
    return when.strftime("%b %d, %Y, %H:%M")


def list_session_summaries(output_root: str | Path,
                           source: str | None = None,
                           speaker: str | None = None
                           ) -> list[SessionSummary]:
    """All sessions under ``output_root``, pinned first, then newest.

    ``source`` narrows to one recording source (see
    :mod:`app.storage.source`); None (the default) returns
    everything. An unknown source value matches nothing.
    ``speaker`` narrows to sessions where that speaker spoke
    (display name after renames, case-insensitive exact match —
    see :func:`_filter_by_speaker`); None (the default) returns
    everyone. Pinned sessions (``pinned: true`` in metadata.json)
    come first, newest-first within each group.
    """
    summaries = [summarize_session(p) for p in list_sessions(output_root)]
    return _sort_pinned_first(
        _filter_by_speaker(_filter_by_source(summaries, source), speaker))


def list_sources(output_root: str | Path) -> list[str]:
    """Recording sources present in the library, in enum order.

    Drives the Recording Browser's source filter dropdown: only
    sources that actually occur are offered (plus "All").
    """
    present = {s.source for s in list_session_summaries(output_root)}
    return [value for value in SOURCES if value in present]


def _filter_by_source(summaries: list[SessionSummary],
                      source: str | None) -> list[SessionSummary]:
    """Summaries whose ``source`` is ``source``; all when None."""
    if source is None:
        return summaries
    return [s for s in summaries if s.source == source]


def _filter_by_speaker(summaries: list[SessionSummary],
                       speaker: str | None) -> list[SessionSummary]:
    """Summaries where ``speaker`` spoke; all when None / blank.

    Matches the speaker's display name (transcript speaker labels
    after speakers.json renames, plus rename-only names — see
    :func:`summarize_session`) case-insensitively and exactly:
    "priya" matches "Priya" but not "Priyanka". A name that
    appears in speakers.db but in no session's transcript or
    renames matches nothing here — there is no session to show.
    """
    if speaker is None or not str(speaker).strip():
        return summaries
    wanted = str(speaker).strip().lower()
    return [s for s in summaries
            if any(name.strip().lower() == wanted
                   for name in s.speakers)]


def _sort_pinned_first(summaries: list[SessionSummary]
                       ) -> list[SessionSummary]:
    """Pinned summaries first, otherwise keeping the given order.

    The input is already newest-first, and the sort is stable, so
    pinned sessions stay newest-first among themselves and so do
    the rest.
    """
    return sorted(summaries, key=lambda s: not s.pinned)


def session_transcript_text(session_dir: str | Path) -> str:
    """Plain transcript text for one session (for search), or ""."""
    session_dir = Path(session_dir)
    txt_path = session_dir / "transcript.txt"
    if txt_path.exists():
        try:
            return txt_path.read_text(encoding="utf-8")
        except OSError:
            return ""
    transcript_path = session_dir / TRANSCRIPT_JSON
    if transcript_path.exists():
        try:
            segments = load_transcript_json(transcript_path)
            return "\n".join(seg.text for seg in segments)
        except Exception:  # noqa: BLE001 — search must not crash
            return ""
    return ""


def search_sessions(output_root: str | Path,
                    query: str, mode: str = "hybrid",
                    index: SemanticSearchIndex | None = None,
                    top_k: int = SEMANTIC_DEFAULT_TOP_K,
                    threshold: float = SEMANTIC_DEFAULT_THRESHOLD,
                    source: str | None = None,
                    speaker: str | None = None,
                    ) -> list[SessionSummary]:
    """Sessions matching ``query``.

    ``source`` narrows the matches to one recording source (see
    :mod:`app.storage.source`); None returns every source. The
    filter applies in every mode, including the empty-query
    "return everything" path. ``speaker`` narrows to sessions
    where that speaker spoke (display name after renames, see
    :func:`_filter_by_speaker`); None returns everyone, in every
    mode. Text / empty-query results are pinned-first, then
    newest; semantic and hybrid results keep relevance order
    (the pinned sort is the library list's order, not a
    relevance signal).

    ``mode="hybrid"`` (default since Nov 8 2026) merges exact and
    meaning matches — exact matches boosted first, semantic matches
    after. ``mode="text"`` is the Phase 5 behaviour, unchanged:
    case-insensitive substring over transcript text, speaker
    names, session id, display name and app name. ``mode="semantic"`` ranks
    sessions by their best chunk's cosine similarity to the
    query (see :mod:`app.storage.semantic_search`).
    Hybrid is best-effort: when the MiniLM model cannot be fetched
    (not installed, offline, first-use download failed) it falls
    back to the text search instead of raising, so the default
    search path never breaks. ``mode="semantic"`` still raises
    ``ModelNotAvailableError`` in that case — the caller asked
    for meaning matches explicitly. An empty / blank query returns
    every session, newest first, in every mode. ``index`` is
    injectable for tests; pass None to build one for
    ``output_root``.
    """
    if mode not in SEARCH_MODES:
        raise ValueError(
            f"unknown search mode {mode!r}; expected one of "
            f"{list(SEARCH_MODES)}")
    query = (query or "").strip()
    if mode == "text" or not query:
        return _filter_by_speaker(
            _filter_by_source(
                _search_sessions_text(output_root, query), source),
            speaker)
    if mode == "hybrid":
        try:
            hits = hybrid_search(output_root, query, index=index,
                                 top_k=top_k, threshold=threshold)
        except ModelNotAvailableError:
            # No embedding model (onnxruntime missing, offline,
            # download failed): degrade to the substring search
            # that always works. Hybrid is the default mode, so
            # raising here would break the Recording Browser on
            # any machine without the model — the UI layer tells
            # the user the semantic part is unavailable.
            logger.info("hybrid search: MiniLM unavailable, "
                        "falling back to text search")
            return _filter_by_speaker(
                _filter_by_source(
                    _search_sessions_text(output_root, query), source),
                speaker)
    else:
        hits = _semantic_hits_to_sessions(output_root, query,
                                          index=index, top_k=top_k,
                                          threshold=threshold)
    summaries = {s.session_id: s
                 for s in list_session_summaries(output_root)}
    ordered: list[SessionSummary] = []
    for hit in hits:
        summary = summaries.get(hit.session_id)
        if summary is not None and summary not in ordered:
            ordered.append(summary)
    return _filter_by_speaker(
        _filter_by_source(ordered, source), speaker)


def _semantic_hits_to_sessions(output_root: str | Path, query: str,
                               index: SemanticSearchIndex | None = None,
                               top_k: int = SEMANTIC_DEFAULT_TOP_K,
                               threshold: float = SEMANTIC_DEFAULT_THRESHOLD
                               ) -> list[SemanticHit]:
    """Semantic chunk hits for a query (index built on demand)."""
    search_index = index or SemanticSearchIndex(output_root)
    return search_index.semantic_search(query, top_k=top_k,
                                        threshold=threshold)


def hybrid_search(output_root: str | Path, query: str, *,
                  index: SemanticSearchIndex | None = None,
                  top_k: int = SEMANTIC_DEFAULT_TOP_K,
                  threshold: float = SEMANTIC_DEFAULT_THRESHOLD
                  ) -> list[SemanticHit]:
    """Merge substring hits with semantic hits, de-duplicated.

    Exact (substring) matches are boosted by
    ``HYBRID_TEXT_BOOST`` so they rank first; a chunk that
    matches both ways is returned once, as ``match_type="both"``.
    Sessions that matched by substring but had no semantic hit
    contribute their first chunk containing the query
    (``match_type="text"``); if the session has no indexed chunk
    at all, its transcript snippet is returned as a text hit.
    Raises ModelNotAvailableError when the embedding model
    cannot be fetched — callers that want an offline fallback
    should catch it and use the text mode instead.
    """
    query = (query or "").strip()
    if not query:
        return []
    search_index = index or SemanticSearchIndex(output_root)
    sem_hits = search_index.semantic_search(query, top_k=top_k,
                                            threshold=threshold)
    text_ids = {s.session_id for s in
                _search_sessions_text(output_root, query)}
    merged: dict[tuple[str, int], SemanticHit] = {}
    for hit in sem_hits:
        if hit.session_id in text_ids:
            hit = SemanticHit(
                session_id=hit.session_id, chunk_id=hit.chunk_id,
                start_s=hit.start_s, end_s=hit.end_s, text=hit.text,
                score=hit.score + HYBRID_TEXT_BOOST,
                match_type="both")
        merged[(hit.session_id, hit.chunk_id)] = hit
    for session_id in text_ids:
        if any(h.session_id == session_id for h in merged.values()):
            continue
        chunk = search_index.find_text_chunk(session_id, query)
        if chunk is not None:
            merged[(session_id, chunk.chunk_id)] = chunk
            continue
        summary = next(
            (s for s in list_session_summaries(output_root)
             if s.session_id == session_id), None)
        if summary is None:
            continue
        text = session_transcript_text(summary.path)
        snippet = _snippet_around(text, query)
        merged[(session_id, -1)] = SemanticHit(
            session_id=session_id, chunk_id=-1, start_s=0.0,
            end_s=0.0, text=snippet, score=HYBRID_TEXT_BOOST,
            match_type="text")
    hits = sorted(merged.values(), key=lambda h: -h.score)
    return hits[:top_k]


def _snippet_around(text: str, query: str, radius: int = 140) -> str:
    """~300 chars of ``text`` centred on the first ``query`` hit."""
    lower = text.lower()
    pos = lower.find(query.lower())
    if pos < 0:
        snippet = text[:300]
    else:
        start = max(0, pos - radius)
        snippet = text[start:start + 2 * radius]
    snippet = " ".join(snippet.split())
    return snippet if len(snippet) <= 300 else snippet[:297] + "…"


def _search_sessions_text(output_root: str | Path,
                          query: str) -> list[SessionSummary]:
    """The Phase 5 substring search (unchanged behaviour)."""
    query = (query or "").strip().lower()
    summaries = list_session_summaries(output_root)
    if not query:
        return summaries
    hits: list[SessionSummary] = []
    for summary in summaries:
        haystacks = [
            summary.session_id,
            summary.display_name,
            summary.app_name or "",
            " ".join(summary.speakers),
            session_transcript_text(summary.path),
        ]
        if any(query in (h or "").lower() for h in haystacks):
            hits.append(summary)
    return hits


def load_session_transcript(session_dir: str | Path):
    """Segments for a session (Phase 2 loader, re-exported here)."""
    return load_transcript_json(Path(session_dir) / TRANSCRIPT_JSON)


__all__ = [
    "SEARCH_MODES",
    "SessionSummary",
    "hybrid_search",
    "library_row_lines",
    "list_session_summaries",
    "list_sources",
    "load_session_transcript",
    "row_date_text",
    "search_sessions",
    "session_transcript_text",
    "summarize_session",
]
