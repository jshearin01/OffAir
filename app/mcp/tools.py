"""The MCP tools, as plain functions (no SDK import).

Nine read-only tools plus three scheduling tools — see the
scheduling exception below. Every tool here is a method of
:class:`OffAirMCPTools`, which only needs an output root — the
folder recordings live under (``~/OffAir/recordings`` by default).
All search / session loading is delegated to the existing code:
:func:`app.storage.browser.search_sessions` /
:func:`app.storage.browser.hybrid_search` for search,
:func:`app.storage.browser.list_session_summaries` /
:func:`app.storage.browser.summarize_session` for sessions,
:func:`app.transcription.exports.load_transcript_json` for
transcripts, :class:`app.speakers.database.SpeakerDatabase` for
learned speakers, :func:`app.speakers.people.get_speaker_detail`
for per-speaker detail, :func:`app.storage.bookmarks.
list_bookmarks` for bookmarks, the transcript-variant registry
in :mod:`app.transcription.cleanup` for the cleaned variants, and
:class:`app.exports.local_llm.OllamaClient`
for the optional RAG answer in :meth:`ask_meetings`.

Read-only guarantee, with ONE scoped exception: the nine archive
tools write nothing (the semantic-search index exception below
is index maintenance for a read, not a change to any recording,
transcript or speaker). The three scheduling tools
(:meth:`schedule_recording`, :meth:`list_schedules`,
:meth:`cancel_schedule`) are the deliberate exception John
ordered Nov 22 2026: they write ONLY the schedule store file
(``~/.offair/schedules.json``) — never a transcript, session,
setting, or database row. An agent connected to this server can
therefore schedule (and cancel) future recordings, and nothing
else. The exception is flagged in the server instructions, in
``docs/MCP.md``, and on the MCP setup page — a client must never
have to guess which tools write.

Errors: bad input (unknown mode, bad ``since``, unknown session)
raises :class:`ValueError`. The MCP server surface turns that
into a tool error; direct callers catch it the same way. A
missing semantic model is NOT an error for ``search_meetings``
in ``hybrid`` mode (it falls back to text, like the browser) or
for ``semantic`` mode (it falls back to text and says so in the
returned ``note``) — search must never fail outright just because
a model download is unavailable offline.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from app.diarization.names import load_speaker_names
from app.exports.local_llm import OllamaClient
from app.speakers.people import get_speaker_detail
from app.storage.bookmarks import list_bookmarks as list_session_bookmarks
from app.storage.browser import (
    hybrid_search,
    list_session_summaries,
    load_session_transcript,
    search_sessions,
    session_transcript_text,
    summarize_session,
)
from app.storage.semantic_search import (
    ModelNotAvailableError,
    SemanticSearchIndex,
)
from app.storage.sessions import list_sessions, load_metadata
from app.storage.source import SOURCE_OTHER, SOURCES
from app.transcription.cleanup import (
    TRANSCRIPT_VARIANT_LABELS,
    transcript_variant_path,
)
from app.transcription.exports import format_txt_timestamp

logger = logging.getLogger("offair.mcp.tools")

# Recording source — the single definition lives in
# app/storage/source.py (the source feature owns the enum); this
# module aliases it so the MCP contract and the app cannot drift
# apart. Every session's metadata.json carries ``source`` (enum
# below) and ``source_detail`` (app display name, window title, or
# user note). When undetermined the source is ``other``, never a
# guess; sessions written before the field existed have no
# ``source`` key at all and are treated as ``other`` here.
SOURCE_VALUES = SOURCES
SOURCE_DEFAULT = SOURCE_OTHER


def _calendar_event(session_dir: str | Path) -> dict | None:
    """The session's linked Outlook event, or None.

    A pure read of the ``calendar_event`` key Outlook linking
    stored in ``metadata.json`` (see :mod:`app.calendar.outlook`),
    returned verbatim: ``{subject, start, end, attendees,
    organiser, entry_id}``. None when the key is absent, null
    (no event matched), or not a dict — a corrupt value reads as
    no link, never as a crash.
    """
    try:
        # backfill=False: a pure read, matching _source_fields —
        # the MCP server is read-only and must never trigger the
        # backfill write load_metadata() performs by default.
        metadata = load_metadata(session_dir, backfill=False)
    except (FileNotFoundError, ValueError, OSError):
        return None
    event = metadata.get("calendar_event")
    return event if isinstance(event, dict) else None


def _source_fields(session_dir: str | Path) -> dict:
    """``{source, source_detail, dictation}`` for a session folder.

    Never crashes and never invents a value: a missing
    metadata.json, a missing ``source`` key (pre-backfill sessions)
    or an unrecognised value all read as ``other`` with
    ``source_detail`` null. ``dictation`` is true only when the
    metadata carries ``dictation: true`` — a ``solo`` source alone
    does not make a session dictation.
    """
    try:
        # backfill=False: a pure read. The MCP server is read-only,
        # so it must never trigger the backfill write that
        # load_metadata() performs by default on legacy sessions.
        metadata = load_metadata(session_dir, backfill=False)
    except (FileNotFoundError, ValueError, OSError):
        return {"source": SOURCE_DEFAULT, "source_detail": None,
                "dictation": False}
    raw = metadata.get("source")
    source = (str(raw).strip().lower() if raw is not None else "")
    if source not in SOURCE_VALUES:
        source = SOURCE_DEFAULT
    detail = metadata.get("source_detail")
    detail = str(detail) if isinstance(detail, str) and detail.strip() \
        else None
    return {"source": source, "source_detail": detail,
            "dictation": metadata.get("dictation") is True}


def _validate_source_filter(source: str | None) -> str | None:
    """A ``source`` filter value, or None for no filtering.

    Raises ValueError on anything outside the enum — a typo'd
    filter fails loudly instead of silently matching nothing.
    """
    if source is None:
        return None
    value = str(source).strip().lower()
    if value not in SOURCE_VALUES:
        raise ValueError(
            f"unknown source {source!r}; expected one of "
            f"{list(SOURCE_VALUES)}")
    return value


def _validate_speaker_filter(speaker: str | None) -> str | None:
    """A ``speaker`` filter value, or None for no filtering.

    A blank value means no filtering, matching the browser's
    speaker filter. Matching itself is an exact case-insensitive
    display-name match (see :func:`_speaker_matches`).
    """
    if speaker is None:
        return None
    value = str(speaker).strip()
    return value or None


def _speaker_matches(speakers: list[str], speaker: str | None) -> bool:
    """Whether ``speaker`` is among ``speakers`` (display names).

    No filter (None) matches everything; otherwise the match is
    exact and case-insensitive — "priya" matches "Priya" but not
    "Priyanka" — the same semantics as the browser's speaker
    filter over the same speaker list.
    """
    if speaker is None:
        return True
    wanted = speaker.strip().lower()
    return any(str(name).strip().lower() == wanted for name in speakers)


# Transcript variants, as the MCP caller names them. The registry
# in app.transcription.cleanup keys the variants "original" /
# "cleaned" / "enterprise"; the MCP names are "verbatim" /
# "clean" / "enterprise", and both spellings are accepted — the
# result always reports the MCP name plus the registry label.
_VARIANT_ALIASES = {
    "verbatim": "original",
    "original": "original",
    "clean": "cleaned",
    "cleaned": "cleaned",
    "enterprise": "enterprise",
}
_VARIANT_NAMES = {
    "original": "verbatim",
    "cleaned": "clean",
    "enterprise": "enterprise",
}
# The enterprise variant was produced by an external LLM from
# transcript text only; the note says so in every enterprise
# result, so a client never mistakes it for the verbatim record.
ENTERPRISE_VARIANT_NOTE = (
    "This enterprise-cleaned transcript was produced by an "
    "external LLM (Claude Code CLI, `claude -p`) from transcript "
    "text only — no audio ever leaves this machine. It may "
    "contain errors: the verbatim transcript is the record.")


def _resolve_variant(variant: str) -> str:
    """The registry key for an MCP ``variant`` value.

    Raises ValueError on anything outside "verbatim" / "clean" /
    "enterprise" (plus the registry spellings "original" /
    "cleaned" as aliases).
    """
    key = _VARIANT_ALIASES.get(str(variant).strip().lower())
    if key is None:
        raise ValueError(
            f"unknown transcript variant {variant!r}; expected one "
            f"of 'verbatim', 'clean', 'enterprise'")
    return key

# ask_meetings: how many retrieved excerpts go into the prompt /
# the search fallback. Five 40-word chunks is ~200 words of
# context — enough to answer a question about one meeting without
# stuffing a whole transcript into the model.
ASK_RETRIEVAL_TOP_K = 5

_ASK_PROMPT_TEMPLATE = (
    "You are answering a question about the user's recorded meetings. "
    "Answer ONLY from the transcript excerpts below — do not invent "
    "names, dates, numbers or commitments that do not appear in them. "
    "If the excerpts do not contain the answer, say so plainly "
    "instead of guessing.\n\n"
    "Question: {question}\n\n"
    "Excerpts:\n{transcript}"
)


def _summary_to_dict(summary) -> dict:
    """A SessionSummary as plain JSON-serialisable data.

    Includes the recording-source fields (``source`` /
    ``source_detail`` / ``dictation``) — see :func:`_source_fields`
    — and the pinned flag from the summary itself.
    """
    result = {
        "session_id": summary.session_id,
        "start_time": summary.start_time,
        "duration_seconds": summary.duration_seconds,
        "app_name": summary.app_name,
        "mode": summary.mode,
        "status": summary.status,
        "speakers": list(summary.speakers),
        "has_transcript": summary.has_transcript,
        "has_diarization": summary.has_diarization,
        "has_llm_export": summary.has_llm_export,
        "has_suggestions": summary.has_suggestions,
        "pinned": bool(summary.pinned),
    }
    result.update(_source_fields(summary.path))
    return result


def _parse_since(value: str) -> datetime:
    """Parse ``since`` (ISO date or datetime) to an aware datetime.

    A bare date means midnight at the start of that day; a naive
    datetime is read as UTC, matching metadata.json (which stores
    UTC). Raises ValueError on anything unparsable.
    """
    text = str(value).strip()
    try:
        when = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse 'since' {value!r} — use an ISO date "
            f"or datetime, e.g. '2026-11-01' or "
            f"'2026-11-01T00:00:00'.") from exc
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when


def _parse_start(value: str | None) -> datetime | None:
    """A session's metadata start_time as an aware datetime, or None."""
    if not value:
        return None
    try:
        when = datetime.fromisoformat(value)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when


class OffAirMCPTools:
    """The MCP tools, bound to one output root.

    Nine read-only archive tools, plus three scheduling tools —
    the scoped write exception (see the module docstring):
    scheduling writes ONLY the schedule store file. ``index`` /
    ``ollama_client`` are injectable for tests (a fake
    embedder-backed index, a fake Ollama client); production passes
    neither and the real implementations are built lazily.
    ``schedules_path`` is injectable for tests; production uses
    ``~/.offair/schedules.json``.
    """

    def __init__(self, output_root: str | Path,
                 index: SemanticSearchIndex | None = None,
                 ollama_client: OllamaClient | None = None,
                 schedules_path: str | Path | None = None) -> None:
        self.output_root = Path(output_root)
        self._index = index
        self._ollama_client = ollama_client
        self._schedules_path = (
            Path(schedules_path) if schedules_path is not None
            else None)

    # ------------------------------------------------------------ helpers --
    def _find_session_dir(self, session_id: str) -> Path:
        """Session folder for ``session_id``; ValueError if unknown."""
        session_id = str(session_id).strip()
        if not session_id:
            raise ValueError("session_id must not be blank")
        for session_dir in list_sessions(self.output_root):
            if session_dir.name == session_id:
                return session_dir
        raise ValueError(f"No session {session_id!r} under "
                         f"{self.output_root}")

    def _new_index(self) -> SemanticSearchIndex:
        """A search index for one call (see :meth:`_release_index`).

        When an index was injected (tests) it is reused; otherwise a
        fresh one is built per call.
        """
        if self._index is not None:
            return self._index
        return SemanticSearchIndex(self.output_root)

    def _release_index(self, index: SemanticSearchIndex) -> None:
        """Close a per-call index; an injected one is left alone.

        WHY fresh-per-call: the MCP SDK runs each sync tool in a
        worker thread (``anyio.to_thread``), and the thread varies
        from call to call. :class:`SemanticSearchIndex` holds one
        ``sqlite3`` connection, and SQLite objects may only be used
        from the thread that created them — caching one index on
        ``self`` crashed the second semantic search that landed on
        a different thread (``sqlite3.ProgrammingError``, found by
        re-running the manual check). A fresh connection per call
        is cheap (one file open) and always thread-correct; the
        injected test index is single-threaded, so reusing it is
        safe and keeps the fake embedder shared across calls.
        """
        if index is not self._index:
            index.close()

    def _ollama(self) -> OllamaClient:
        if self._ollama_client is None:
            self._ollama_client = OllamaClient()
        return self._ollama_client

    @staticmethod
    def _validate_limit(limit: int) -> int:
        try:
            limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"limit must be an integer, got "
                             f"{limit!r}") from exc
        if limit < 1 or limit > 100:
            raise ValueError(f"limit must be between 1 and 100, got "
                             f"{limit}")
        return limit

    # ------------------------------------------------------------ tools ----
    def search_meetings(self, query: str, mode: str = "hybrid",
                        limit: int = 10,
                        source: str | None = None,
                        speaker: str | None = None) -> dict:
        """Search sessions. Returns ``{mode, note, results}``.

        ``mode`` is ``text`` (substring, always offline), ``semantic``
        (meaning-based, needs the MiniLM model) or ``hybrid``
        (exact hits boosted first, then semantic — the default).
        ``source`` filters to one recording source (exact enum value;
        omit or null = no filtering). ``speaker`` filters to sessions
        that speaker spoke in (display name, case-insensitive exact
        match; omit, null or blank = no filtering). Each result is a
        session dict (see :func:`_summary_to_dict`) plus ``score``,
        ``match_type`` and ``snippet`` when the mode produced them.
        When the semantic model is unavailable the search falls back
        to text and ``note`` says so — it never fails outright. An
        empty query lists every session.
        """
        limit = self._validate_limit(limit)
        source_filter = _validate_source_filter(source)
        speaker_filter = _validate_speaker_filter(speaker)
        query = (query or "").strip()
        mode = str(mode).strip().lower()
        if mode not in ("text", "semantic", "hybrid"):
            raise ValueError(f"unknown search mode {mode!r}; expected "
                             f"'text', 'semantic' or 'hybrid'")
        summaries = {s.session_id: s
                     for s in list_session_summaries(self.output_root)}
        if not query:
            results = [_summary_to_dict(s) for s in
                       list_session_summaries(self.output_root)]
            if source_filter is not None:
                results = [r for r in results
                           if r["source"] == source_filter]
            if speaker_filter is not None:
                results = [r for r in results
                           if _speaker_matches(r["speakers"],
                                               speaker_filter)]
            for result in results[:limit]:
                result.update({"score": None, "match_type": None,
                               "snippet": None})
            return {"mode": mode, "note": None,
                    "results": results[:limit]}
        if mode == "text":
            hits = search_sessions(self.output_root, query, mode="text",
                                   speaker=speaker_filter)
            results = []
            for summary in hits:
                entry = _summary_to_dict(summary)
                if (source_filter is not None
                        and entry["source"] != source_filter):
                    continue
                text = session_transcript_text(summary.path)
                pos = text.lower().find(query.lower())
                snippet = (" ".join(text[max(0, pos - 140):
                                         pos + 160].split())
                           if pos >= 0 else text[:300])
                entry.update({"score": None, "match_type": "text",
                              "snippet": snippet or None})
                results.append(entry)
                if len(results) >= limit:
                    break
            return {"mode": "text", "note": None, "results": results}
        # semantic / hybrid: chunk hits carry score + snippet.
        # With a source or speaker filter the hit pool must be
        # larger than the limit, or filtering after ranking starves
        # the results.
        pool = 100 if (source_filter is not None
                       or speaker_filter is not None) else limit
        index = self._new_index()
        try:
            try:
                if mode == "hybrid":
                    hits = hybrid_search(
                        self.output_root, query, index=index,
                        top_k=pool)
                else:
                    hits = index.semantic_search(query, top_k=pool)
            except ModelNotAvailableError as exc:
                logger.info("search_meetings: semantic model unavailable "
                            "(%s); falling back to text search", exc)
                fallback = search_sessions(self.output_root, query,
                                           mode="text",
                                           speaker=speaker_filter)
                results = []
                for summary in fallback:
                    entry = _summary_to_dict(summary)
                    if (source_filter is not None
                            and entry["source"] != source_filter):
                        continue
                    entry.update({"score": None, "match_type": "text",
                                  "snippet": None})
                    results.append(entry)
                    if len(results) >= limit:
                        break
                return {"mode": "text",
                        "note": ("Semantic search is unavailable "
                                 f"({exc}); fell back to text search."),
                        "results": results}
        finally:
            self._release_index(index)
        seen: set[str] = set()
        results = []
        for hit in hits:
            if hit.session_id in seen:
                continue
            seen.add(hit.session_id)
            summary = summaries.get(hit.session_id)
            if summary is None:
                continue
            entry = _summary_to_dict(summary)
            if (source_filter is not None
                    and entry["source"] != source_filter):
                continue
            if not _speaker_matches(entry["speakers"], speaker_filter):
                continue
            entry.update({"score": round(float(hit.score), 3),
                          "match_type": hit.match_type,
                          "snippet": hit.snippet})
            results.append(entry)
            if len(results) >= limit:
                break
        return {"mode": mode, "note": None, "results": results}

    def list_sessions(self, limit: int = 10,
                      since: str | None = None,
                      source: str | None = None,
                      pinned_only: bool = False) -> dict:
        """Sessions newest first; ``since`` filters by start time.

        ``since`` is an ISO date (``2026-11-01``) or datetime.
        ``source`` filters to one recording source (exact enum value;
        omit or null = no filtering). ``pinned_only`` keeps only
        pinned (starred) sessions. Returns ``{results}`` — session
        dicts, no search metadata. Every entry carries ``pinned``
        and ``dictation``.
        """
        limit = self._validate_limit(limit)
        cutoff = _parse_since(since) if since is not None else None
        source_filter = _validate_source_filter(source)
        results = []
        for summary in list_session_summaries(self.output_root):
            if cutoff is not None:
                started = _parse_start(summary.start_time)
                if started is not None and started < cutoff:
                    continue
            entry = _summary_to_dict(summary)
            if (source_filter is not None
                    and entry["source"] != source_filter):
                continue
            if pinned_only and not entry["pinned"]:
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return {"results": results}

    def get_session(self, session_id: str) -> dict:
        """One session: status, speakers, app, duration, source, calendar event.

        ``source`` / ``source_detail`` come from the session's
        metadata (``other`` / null when the session predates the
        source field); ``dictation`` is true only for dictation
        sessions. ``bookmarks`` lists the session's bookmarks,
        earliest first (see :meth:`list_bookmarks`).
        ``calendar_event`` is the linked Outlook event stored in
        the session's metadata (``{subject, start, end, attendees,
        organiser, entry_id}``), or null when no event matched.
        """
        session_dir = self._find_session_dir(session_id)
        summary = summarize_session(session_dir)
        result = _summary_to_dict(summary)
        result["calendar_event"] = _calendar_event(session_dir)
        result["bookmarks"] = self.list_bookmarks(
            session_dir.name)["results"]
        return result

    def list_bookmarks(self, session_id: str) -> dict:
        """A session's bookmarks, earliest first: each marked moment, its label, and whether its audio still exists.

        Each entry is ``{t, label, created, audio_available}``,
        where ``t`` is seconds from the start of the session and
        ``audio_available`` is computed at list time — false when
        retention has deleted the session's audio, in which case
        the bookmark is still listed and says so plainly. Empty
        (not an error) when the session has no bookmarks. Bookmark
        creation stays in the app only; this tool only reads.
        """
        session_dir = self._find_session_dir(session_id)
        bookmarks = list_session_bookmarks(session_dir)
        return {
            "session_id": session_dir.name,
            "results": [{
                "t": bookmark.t,
                "label": bookmark.label,
                "created": bookmark.created,
                "audio_available": bookmark.audio_available,
            } for bookmark in bookmarks],
        }

    def get_transcript(self, session_id: str,
                       format: str = "text",
                       variant: str = "verbatim") -> dict:
        """A session's transcript as text lines or segments.

        ``format="text"`` returns one ``[MM:SS] Speaker: text`` line
        per segment; ``format="segments"`` returns
        ``[{start, end, speaker, text}]``. Speaker renames from the
        session's ``speakers.json`` are applied. ``variant`` picks
        which transcript artifact is returned: ``"verbatim"`` (the
        default — the record, unchanged by every existing caller),
        ``"clean"`` (the deterministic cleaned note, e.g. a
        dictation session's primary readable artifact) or
        ``"enterprise"`` (the external-LLM cleanup). The registry
        spellings ``"original"`` and ``"cleaned"`` are accepted as
        aliases for ``"verbatim"`` and ``"clean"``. The result
        always reports ``variant`` (the MCP name), ``variant_label``
        and ``note``; the enterprise result's note states it was
        produced by an external LLM from transcript text only.
        Clean and enterprise variants are text only
        (``format="segments"`` with either raises ValueError).
        Raises ValueError when the session has no transcript yet,
        or no artifact for the requested variant.
        """
        session_dir = self._find_session_dir(session_id)
        fmt = str(format).strip().lower()
        if fmt not in ("text", "segments"):
            raise ValueError(f"unknown transcript format {fmt!r}; "
                             f"expected 'text' or 'segments'")
        variant_key = _resolve_variant(variant)
        variant_name = _VARIANT_NAMES[variant_key]
        variant_label = TRANSCRIPT_VARIANT_LABELS[variant_key]
        if variant_key != "original":
            if fmt != "text":
                raise ValueError(
                    f"transcript variant {variant_name!r} is text "
                    f"only; use format='text'")
            path = transcript_variant_path(session_dir, variant_key)
            if not path.exists():
                raise ValueError(
                    f"Session {session_id!r} has no "
                    f"{variant_label} transcript yet")
            text = path.read_text(encoding="utf-8")
            return {
                "session_id": session_dir.name, "format": "text",
                "variant": variant_name,
                "variant_label": variant_label,
                "text": text, "segments": None,
                "note": (ENTERPRISE_VARIANT_NOTE
                         if variant_key == "enterprise" else None),
            }
        names = load_speaker_names(session_dir)
        transcript_path = session_dir / "transcript.json"
        if transcript_path.exists():
            segments = load_session_transcript(session_dir)
        else:
            txt_path = session_dir / "transcript.txt"
            if not txt_path.exists():
                raise ValueError(
                    f"Session {session_id!r} has no transcript yet")
            text = txt_path.read_text(encoding="utf-8")
            return {"session_id": session_dir.name, "format": "text",
                    "variant": "verbatim",
                    "variant_label": variant_label,
                    "text": text,
                    "segments": None, "note": None}
        display = []
        for seg in segments:
            speaker = getattr(seg, "speaker", None)
            if speaker:
                speaker = names.get(speaker, speaker)
            display.append({"start": round(float(seg.start), 3),
                            "end": round(float(seg.end), 3),
                            "speaker": speaker,
                            "text": seg.text})
        if fmt == "segments":
            return {"session_id": session_dir.name, "format": "segments",
                    "variant": "verbatim",
                    "variant_label": variant_label,
                    "text": None, "segments": display, "note": None}
        lines = []
        for seg in display:
            prefix = f"{seg['speaker']}: " if seg["speaker"] else ""
            lines.append(f"{format_txt_timestamp(seg['start'])} "
                         f"{prefix}{seg['text']}")
        return {"session_id": session_dir.name, "format": "text",
                "variant": "verbatim",
                "variant_label": variant_label,
                "text": "\n".join(lines) + ("\n" if lines else ""),
                "segments": None, "note": None}

    def list_speakers(self) -> dict:
        """Learned speakers from ``speakers.db`` (empty when none).

        Each entry: ``{name, backend, sample_count, sessions}`` where
        ``sessions`` is the list of source sessions its voice samples
        came from, newest first. Speakers known only as a per-session
        rename (``speakers.json``) with no learned voice are not
        listed — this is the persistent speaker memory, not the
        per-recording label list.
        """
        from app.speakers.database import SPEAKERS_DB_FILE, SpeakerDatabase

        db_path = self.output_root / SPEAKERS_DB_FILE
        if not db_path.exists():
            return {"results": []}
        db = SpeakerDatabase(db_path)
        try:
            results = []
            for speaker in db.list_speakers():
                samples = db.speaker_samples(speaker["name"])
                sessions = []
                for sample in samples:
                    source = sample.get("source_session")
                    if source and source not in sessions:
                        sessions.append(source)
                results.append({
                    "name": speaker["name"],
                    "backend": speaker["backend"],
                    "sample_count": speaker["sample_count"],
                    "sessions": sessions,
                })
            return {"results": results}
        finally:
            db.close()

    def get_speaker(self, name: str) -> dict:
        """Everything about one speaker: their sessions, everything they said, and their voice sample, if one still exists.

        Matches the display name (renames applied)
        case-insensitively and exactly — the same semantics as the
        person pages, whose query this reuses directly (see
        :func:`app.speakers.people.get_speaker_detail`). A speaker
        need not be in ``speakers.db`` to have sessions and
        utterances; ``known_in_database`` says whether a learned
        voice exists. An unknown speaker returns empty sessions /
        utterances, not an error. ``sample`` is a pointer back into
        a session folder (audio path + timestamps), never audio
        bytes; ``sample_available`` is false when the audio is gone.
        """
        wanted = str(name).strip() if name is not None else ""
        if not wanted:
            raise ValueError("Speaker name must not be blank")
        from app.speakers.database import (
            SPEAKERS_DB_FILE,
            SpeakerDatabase,
        )

        db_path = self.output_root / SPEAKERS_DB_FILE
        db = SpeakerDatabase(db_path) if db_path.exists() else None
        try:
            return get_speaker_detail(self.output_root, db, wanted)
        finally:
            if db is not None:
                db.close()

    def what_did_speaker_say(self, speaker: str, query: str = "",
                             limit: int = 10) -> dict:
        """Segments spoken by ``speaker`` across all sessions.

        Matches the display name (renames applied) case-insensitively,
        newest session first. ``query`` narrows to segments containing
        that substring (case-insensitive); empty means every segment
        by that speaker. Returns ``{speaker, results}`` with
        ``{session_id, start, end, text}`` per segment.
        """
        limit = self._validate_limit(limit)
        wanted = str(speaker).strip()
        if not wanted:
            raise ValueError("speaker must not be blank")
        needle = (query or "").strip().lower()
        results = []
        for summary in list_session_summaries(self.output_root):
            session_dir = summary.path
            transcript_path = session_dir / "transcript.json"
            if not transcript_path.exists():
                continue
            names = load_speaker_names(session_dir)
            try:
                segments = load_session_transcript(session_dir)
            except Exception:  # noqa: BLE001 — a bad transcript is skipped
                logger.warning("what_did_speaker_say: could not read "
                               "transcript in %s", session_dir,
                               exc_info=True)
                continue
            for seg in segments:
                label = getattr(seg, "speaker", None)
                display = names.get(label, label) if label else None
                if display is None or display.lower() != wanted.lower():
                    continue
                if needle and needle not in (seg.text or "").lower():
                    continue
                results.append({
                    "session_id": summary.session_id,
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "speaker": display,
                    "text": seg.text,
                })
                if len(results) >= limit:
                    return {"speaker": wanted, "results": results}
        return {"speaker": wanted, "results": results}

    def ask_meetings(self, question: str) -> dict:
        """Answer ``question`` from the archive (RAG, local Ollama).

        Retrieves the most relevant transcript excerpts with hybrid
        search, then asks the local Ollama model to answer from
        those excerpts only. When Ollama is not running there is no
        generated answer — the response says so plainly and returns
        the retrieved excerpts instead (``source``:
        ``"search-fallback"``, ``answer`` null). Nothing is ever
        invented: with no excerpts and no Ollama, the answer is
        null and the note says no matching meetings were found.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be blank")
        excerpts: list[dict] = []
        index = self._new_index()
        try:
            try:
                hits = hybrid_search(self.output_root, question,
                                     index=index,
                                     top_k=ASK_RETRIEVAL_TOP_K)
                for hit in hits:
                    excerpts.append({
                        "session_id": hit.session_id,
                        "start": round(float(hit.start_s), 3),
                        "end": round(float(hit.end_s), 3),
                        "text": hit.text,
                        "score": round(float(hit.score), 3),
                    })
            except ModelNotAvailableError:
                logger.info("ask_meetings: semantic model unavailable; "
                            "retrieving by text search")
                for summary in search_sessions(
                        self.output_root, question,
                        mode="text")[:ASK_RETRIEVAL_TOP_K]:
                    text = session_transcript_text(summary.path)
                    pos = text.lower().find(question.lower())
                    snippet = (" ".join(text[max(0, pos - 140):
                                             pos + 160].split())
                               if pos >= 0 else text[:300])
                    excerpts.append({"session_id": summary.session_id,
                                     "start": 0.0, "end": 0.0,
                                     "text": snippet, "score": None})
        finally:
            self._release_index(index)
        if not excerpts:
            return {"answer": None, "source": "search-fallback",
                    "model": None, "sessions": [],
                    "note": ("No meetings matched that question, so "
                             "there is nothing to answer from."),
                    "results": []}
        sessions = []
        for excerpt in excerpts:
            if excerpt["session_id"] not in sessions:
                sessions.append(excerpt["session_id"])
        client = self._ollama()
        if not client.is_available():
            return {
                "answer": None,
                "source": "search-fallback",
                "model": None,
                "sessions": sessions,
                "note": ("Ollama is not running on localhost:11434, so "
                         "no generated answer is available — here are "
                         "the most relevant transcript excerpts instead. "
                         "Start Ollama (`ollama serve`, then "
                         "`ollama pull qwen3.5:9b`) to get answers."),
                "results": excerpts,
            }
        context = "\n\n".join(
            f"[{e['session_id']} {format_txt_timestamp(e['start'])}] "
            f"{e['text']}" for e in excerpts)
        # The template is formatted twice (once here for the
        # question, once inside summarize() for {transcript}), so
        # braces in the question must be escaped or the second
        # .format() would try to substitute them.
        safe_question = question.replace("{", "{{").replace("}", "}}")
        answer = client.summarize(
            context, prompt_template=_ASK_PROMPT_TEMPLATE.format(
                question=safe_question, transcript="{transcript}"))
        return {"answer": answer, "source": "ollama",
                "model": client.model, "sessions": sessions,
                "note": None, "results": excerpts}

    # ------------------------------------------------ scheduling tools --
    # The scoped write exception (Nov 22 2026): these three tools
    # write ONLY the schedule store file (~/.offair/schedules.json)
    # via app.storage.schedules — never a transcript, session,
    # setting, or database row. The app picks schedules up live
    # (mtime reload) and fires them while it is running; a
    # schedule whose time passes while the app is closed expires
    # instead of firing late.
    def _schedules_file(self) -> Path:
        from app.storage.schedules import default_schedules_path

        return (self._schedules_path
                if self._schedules_path is not None
                else default_schedules_path())

    @staticmethod
    def _resolve_target_app(target_app: str) -> str:
        """An exe name for a target-app value.

        The agent names an app the way the calendar names it
        ("Microsoft Teams", "Zoom", "chrome.exe"). A known
        display name maps to its exe (the known-app map in
        app.audio.app_sessions, reversed); anything else is kept
        as given — per-app capture fails open to loopback for an
        unknown exe, so an unrecognised name degrades instead of
        refusing the schedule.
        """
        from app.audio.app_sessions import KNOWN_APP_NAMES

        value = str(target_app).strip()
        if not value:
            raise ValueError("target_app must not be blank")
        # First exe wins: several exes share one display name
        # ("Microsoft Teams" is ms-teams.exe, teams.exe AND the
        # ms-teamsupdate.exe updater) and the meeting app — the
        # first listed — is the one to record, never the updater.
        by_display: dict[str, str] = {}
        for exe, display in KNOWN_APP_NAMES.items():
            by_display.setdefault(display.lower(), exe)
        return by_display.get(value.lower(), value)

    @staticmethod
    def _parse_fire_at(fire_at: str) -> datetime:
        """An aware future datetime for a ``fire_at`` value.

        ISO-8601 with an offset ("2026-11-23T09:00:00-05:00"). A
        naive value is read as local wall time. Raises ValueError
        for anything unparsable or not in the future — a schedule
        in the past would expire on sight, so refusing it loudly
        is kinder than accepting it.
        """
        text = str(fire_at).strip() if fire_at is not None else ""
        try:
            when = datetime.fromisoformat(text)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Could not parse fire_at {fire_at!r} — use an ISO "
                f"datetime with an offset, e.g. "
                f"'2026-11-23T09:00:00-05:00'.") from exc
        if when.tzinfo is None:
            when = when.astimezone()
        if when <= datetime.now(timezone.utc):
            raise ValueError(
                f"fire_at {when.isoformat(timespec='seconds')} is not "
                f"in the future")
        return when

    @staticmethod
    def _schedule_to_dict(schedule) -> dict:
        """A stored schedule as plain JSON-serialisable data."""
        return schedule.to_dict()

    def schedule_recording(self, fire_at: str, target_app: str,
                           subject: str | None = None,
                           attendees: list[str] | str | None = None,
                           organiser: str | None = None) -> dict:
        """Schedule a one-off recording (WRITE — see module docstring).

        The app starts recording ``target_app`` automatically at
        ``fire_at`` while it is running, and stops only when the
        user presses Stop. ``fire_at`` is an ISO datetime with an
        offset (naive = local wall time) and must be in the
        future. ``target_app`` is an exe name ("ms-teams.exe") or
        a display name ("Microsoft Teams"). ``subject`` /
        ``attendees`` / ``organiser`` are optional calendar
        details carried into the session's metadata. Consent:
        scheduled auto-start does NOT trigger Teams' recording
        banner — participants must be told. Returns the created
        schedule, including its id.
        """
        from app.storage.schedules import create_schedule

        when = self._parse_fire_at(fire_at)
        exe = self._resolve_target_app(target_app)
        if isinstance(attendees, str):
            attendees_list = [attendees]
        elif attendees is None:
            attendees_list = []
        else:
            attendees_list = [str(a) for a in attendees]
        schedule = create_schedule(
            when, exe, subject=subject, attendees=attendees_list,
            organiser=organiser, created_by="mcp",
            path=self._schedules_file())
        return {"schedule": self._schedule_to_dict(schedule)}

    def list_schedules(self, limit: int = 20) -> dict:
        """Schedules: pending first (earliest first), then recent history.

        Read-only. Each entry is the stored schedule record
        (``id``, ``fire_at`` with offset, ``target_app``,
        ``subject``, ``attendees``, ``organiser``, ``created_by``,
        ``status``, ``created``, ``fired_at``, ``session_id``,
        ``note``). Terminal records (fired / skipped / expired /
        cancelled) from the last 7 days are included; older ones
        stay in the store file but are not listed.
        """
        from app.storage.schedules import (
            load_schedules,
            visible_schedules,
        )

        limit = self._validate_limit(limit)
        schedules = load_schedules(self._schedules_file())
        results = [self._schedule_to_dict(s)
                   for s in visible_schedules(schedules)]
        return {"results": results[:limit]}

    def cancel_schedule(self, schedule_id: str) -> dict:
        """Cancel a pending schedule (WRITE — see module docstring).

        Raises ValueError for an unknown id or a schedule that is
        no longer pending. Returns the cancelled schedule.
        """
        from app.storage.schedules import cancel_schedule

        schedule_id = str(schedule_id).strip()
        if not schedule_id:
            raise ValueError("schedule_id must not be blank")
        schedule = cancel_schedule(schedule_id,
                                   path=self._schedules_file())
        return {"schedule": self._schedule_to_dict(schedule)}


__all__ = ["OffAirMCPTools", "ASK_RETRIEVAL_TOP_K", "SOURCE_VALUES",
           "SOURCE_DEFAULT"]
