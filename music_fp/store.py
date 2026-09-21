"""SQLite persistence for tracks and their feature vectors."""

import io
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .config import DB_PATH, ensure_dirs


def _connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    _migrate(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source          TEXT NOT NULL UNIQUE,
            title           TEXT,
            artist          TEXT,
            year            INTEGER,
            notes           TEXT,
            duration        REAL,
            label           TEXT NOT NULL DEFAULT 'untagged',
            features        BLOB,
            feature_version INTEGER NOT NULL DEFAULT 1,
            added_at        TEXT NOT NULL
        )
    """)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that didn't exist in v1 schema."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracks)").fetchall()}
    for col, typedef in [
        ("artist", "TEXT"),
        ("year", "INTEGER"),
        ("notes", "TEXT"),
        ("feature_version", "INTEGER NOT NULL DEFAULT 1"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE tracks ADD COLUMN {col} {typedef}")
    conn.commit()


def _to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


def upsert_track(
    source: str,
    title: Optional[str],
    duration: Optional[float],
    label: str,
    features: np.ndarray,
    artist: Optional[str] = None,
    year: Optional[int] = None,
    notes: Optional[str] = None,
    feature_version: int = 2,
) -> int:
    """Insert or update a track; return its row id."""
    from .fingerprint import FEATURE_VERSION
    conn = _connect()
    blob = _to_blob(features)
    existing = conn.execute("SELECT id FROM tracks WHERE source = ?", (source,)).fetchone()
    if existing:
        conn.execute(
            """UPDATE tracks SET title=?, artist=?, year=?, notes=?,
               duration=?, label=?, features=?, feature_version=? WHERE id=?""",
            (title, artist, year, notes, duration, label, blob, feature_version, existing["id"]),
        )
        conn.commit()
        return existing["id"]
    cur = conn.execute(
        """INSERT INTO tracks
           (source, title, artist, year, notes, duration, label, features, feature_version, added_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (source, title, artist, year, notes, duration, label, blob, feature_version,
         datetime.utcnow().isoformat()),
    )
    conn.commit()
    return cur.lastrowid


def get_all_tracks(labeled_only: bool = False, label_filter: Optional[list[str]] = None) -> list[dict]:
    conn = _connect()
    query = "SELECT id, source, title, artist, year, notes, duration, label, features, feature_version, added_at FROM tracks"
    params: list = []
    if label_filter:
        placeholders = ",".join("?" * len(label_filter))
        query += f" WHERE label IN ({placeholders})"
        params = label_filter
    elif labeled_only:
        query += " WHERE label NOT IN ('untagged')"
    rows = conn.execute(query, params).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["features"] = _from_blob(d["features"]) if d["features"] else None
        result.append(d)
    return result


def get_track_by_source(source: str) -> Optional[dict]:
    conn = _connect()
    row = conn.execute("SELECT * FROM tracks WHERE source = ?", (source,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["features"] = _from_blob(d["features"]) if d["features"] else None
    return d


def update_label(track_id: int, label: str) -> None:
    conn = _connect()
    conn.execute("UPDATE tracks SET label=? WHERE id=?", (label, track_id))
    conn.commit()


def count_by_label() -> dict[str, int]:
    conn = _connect()
    rows = conn.execute("SELECT label, COUNT(*) as n FROM tracks GROUP BY label").fetchall()
    return {r["label"]: r["n"] for r in rows}
