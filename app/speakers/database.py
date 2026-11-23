"""Local speaker database — SQLite + embedding storage (PRD Phase 4).

One file, ``speakers.db``, in the recordings root (default
``~/OffAir/recordings/speakers.db``). Nothing leaves the machine:
names and embeddings are local only, and there is no cloud sync.

Schema::

    speakers   (id, name UNIQUE, backend, created_at, updated_at,
                sample_count)
    embeddings (id, speaker_id -> speakers.id, embedding BLOB float32,
                dim, backend, source_session, source_label, created_at)

``backend`` is stored on both tables: embeddings from different
backends live in different vector spaces and must never be compared
with each other. :func:`SpeakerDatabase.speaker_embedding` averages
only the embeddings of one backend (the speaker's backend), and
matching refuses to compare across backends.

A speaker's stored embedding is the mean of its samples, renormalised
to unit length; every confirmed sample adds a row (a running average
is kept implicitly, so a correction can be audited back to the
session it came from). ``sample_count`` is a denormalised counter.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("offair.speakers.database")

SPEAKERS_DB_FILE = "speakers.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    backend TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    backend TEXT NOT NULL,
    source_session TEXT,
    source_label TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_speaker
    ON embeddings(speaker_id);
"""


def _now() -> float:
    return time.time()


class SpeakerDatabase:
    """SQLite store for known speakers and their voice embeddings."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SpeakerDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ speakers ---
    def create_speaker(self, name: str, backend: str) -> int:
        """Create a speaker (or return the existing id for ``name``).

        A speaker's backend is fixed at creation: adding an embedding
        from a different backend to the same name raises ValueError,
        because the vectors would not be comparable.
        """
        name = str(name).strip()
        if not name:
            raise ValueError("Speaker name must not be blank")
        row = self._conn.execute(
            "SELECT id, backend FROM speakers WHERE name = ?", (name,)
        ).fetchone()
        if row is not None:
            if row[1] != backend:
                raise ValueError(
                    f"Speaker {name!r} was created with backend "
                    f"{row[1]!r}, not {backend!r} — embeddings from "
                    "different backends live in different vector spaces "
                    "and must not be mixed. Delete the speaker first if "
                    "you really want to switch backends.")
            return int(row[0])
        now = _now()
        cur = self._conn.execute(
            "INSERT INTO speakers (name, backend, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)", (name, backend, now, now))
        self._conn.commit()
        logger.info("Created speaker %r (backend %s)", name, backend)
        return int(cur.lastrowid)

    def get_speaker(self, name: str) -> dict | None:
        row = self._conn.execute(
            "SELECT id, name, backend, created_at, updated_at, sample_count "
            "FROM speakers WHERE name = ?", (str(name).strip(),)
        ).fetchone()
        if row is None:
            return None
        return {"id": int(row[0]), "name": row[1], "backend": row[2],
                "created_at": row[3], "updated_at": row[4],
                "sample_count": int(row[5])}

    def list_speakers(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, name, backend, created_at, updated_at, sample_count "
            "FROM speakers ORDER BY name").fetchall()
        return [{"id": int(r[0]), "name": r[1], "backend": r[2],
                 "created_at": r[3], "updated_at": r[4],
                 "sample_count": int(r[5])} for r in rows]

    def speaker_samples(self, name: str) -> list[dict]:
        """Samples for a speaker, newest first.

        Each entry: {"source_session", "source_label", "created_at"}.
        Only the session + label are stored — no audio (see
        ``app.speakers.session.find_sample_for_speaker``: a sample
        is played back from its source session's audio file, so a
        speaker whose source recordings are gone has no sample
        left to play).
        """
        speaker = self.get_speaker(name)
        if speaker is None:
            return []
        rows = self._conn.execute(
            "SELECT source_session, source_label, created_at "
            "FROM embeddings WHERE speaker_id = ? "
            "ORDER BY created_at DESC, rowid DESC",
            (speaker["id"],)).fetchall()
        return [{"source_session": r[0], "source_label": r[1],
                 "created_at": r[2]} for r in rows]

    def rename_speaker(self, old_name: str, new_name: str) -> dict:
        """Rename a known speaker. Raises ValueError on a blank / taken name."""
        old_name, new_name = str(old_name).strip(), str(new_name).strip()
        if not new_name:
            raise ValueError("New speaker name must not be blank")
        if self.get_speaker(old_name) is None:
            raise ValueError(f"No known speaker {old_name!r}")
        if self.get_speaker(new_name) is not None:
            raise ValueError(f"A speaker named {new_name!r} already exists")
        self._conn.execute(
            "UPDATE speakers SET name = ?, updated_at = ? WHERE name = ?",
            (new_name, _now(), old_name))
        self._conn.commit()
        return self.get_speaker(new_name)  # type: ignore[return-value]

    def delete_speaker(self, name: str) -> bool:
        """Delete a speaker and all its embeddings. Returns True if one existed."""
        speaker = self.get_speaker(name)
        if speaker is None:
            return False
        self._conn.execute("DELETE FROM embeddings WHERE speaker_id = ?",
                           (speaker["id"],))
        self._conn.execute("DELETE FROM speakers WHERE id = ?",
                           (speaker["id"],))
        self._conn.commit()
        logger.info("Deleted speaker %r", name)
        return True

    # ----------------------------------------------------------- embeddings ---
    def add_embedding(self, name: str, embedding: np.ndarray, *,
                      backend: str, source_session: str | None = None,
                      source_label: str | None = None) -> int:
        """Add one embedding sample for ``name`` (creating it if needed).

        Returns the speaker id. The embedding is stored as float32
        bytes; it is normalised to unit length on the way in.
        """
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if vec.size == 0:
            raise ValueError("Cannot store an empty embedding")
        norm = float(np.linalg.norm(vec))
        if norm == 0:
            raise ValueError("Cannot store a zero embedding")
        vec = (vec / norm).astype(np.float32)
        speaker_id = self.create_speaker(name, backend)
        self._conn.execute(
            "INSERT INTO embeddings (speaker_id, embedding, dim, backend, "
            "source_session, source_label, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (speaker_id, vec.tobytes(), int(vec.size), backend,
             source_session, source_label, _now()))
        self._conn.execute(
            "UPDATE speakers SET sample_count = sample_count + 1, "
            "updated_at = ? WHERE id = ?", (_now(), speaker_id))
        self._conn.commit()
        return speaker_id

    def speaker_embedding(self, name: str) -> np.ndarray | None:
        """Mean embedding for ``name`` (unit length), or None if unknown."""
        speaker = self.get_speaker(name)
        if speaker is None:
            return None
        rows = self._conn.execute(
            "SELECT embedding, dim FROM embeddings WHERE speaker_id = ? "
            "AND backend = ?", (speaker["id"], speaker["backend"])
        ).fetchall()
        if not rows:
            return None
        vecs = [np.frombuffer(r[0], dtype=np.float32) for r in rows]
        mean = np.mean(vecs, axis=0).astype(np.float64)
        norm = float(np.linalg.norm(mean))
        if norm == 0:
            return None
        return (mean / norm).astype(np.float32)

    def all_speaker_embeddings(self, backend: str | None = None
                               ) -> dict[str, np.ndarray]:
        """{name: mean embedding} for every speaker with samples.

        With ``backend`` set, only speakers of that backend are
        returned — matching must never compare across backends.
        """
        out: dict[str, np.ndarray] = {}
        for speaker in self.list_speakers():
            if backend is not None and speaker["backend"] != backend:
                continue
            vec = self.speaker_embedding(speaker["name"])
            if vec is not None:
                out[speaker["name"]] = vec
        return out
