"""SQLite persistence — tracks, votes, candidates."""

import io
import re
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
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_schema(conn)
    _migrate(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
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
            feature_version INTEGER NOT NULL DEFAULT 2,
            audio_path      TEXT,
            thumbnail_url   TEXT,
            video_id        TEXT,
            added_at        TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS votes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id    INTEGER NOT NULL REFERENCES tracks(id),
            vote        TEXT NOT NULL,   -- liked / disliked / skipped
            voted_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS candidates (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id    INTEGER NOT NULL REFERENCES tracks(id),
            score       REAL DEFAULT 0.5,
            seed_artist TEXT,
            source      TEXT DEFAULT 'youtube_search',
            status      TEXT NOT NULL DEFAULT 'pending',  -- pending/served/voted
            queued_at   TEXT NOT NULL,
            served_at   TEXT,
            voted_at    TEXT
        );
    """)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracks)").fetchall()}
    for col, typedef in [
        ("artist",          "TEXT"),
        ("year",            "INTEGER"),
        ("notes",           "TEXT"),
        ("feature_version", "INTEGER NOT NULL DEFAULT 2"),
        ("audio_path",      "TEXT"),
        ("thumbnail_url",   "TEXT"),
        ("video_id",        "TEXT"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE tracks ADD COLUMN {col} {typedef}")
    conn.commit()


# ── Serialisation ────────────────────────────────────────────────────────────

def _to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _from_blob(blob: bytes) -> Optional[np.ndarray]:
    if not blob:
        return None
    return np.load(io.BytesIO(blob))


def _extract_video_id(source: str) -> Optional[str]:
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", source)
    return m.group(1) if m else None


def _thumbnail(video_id: Optional[str]) -> Optional[str]:
    if not video_id:
        return None
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


# ── Tracks ───────────────────────────────────────────────────────────────────

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
    audio_path: Optional[str] = None,
) -> int:
    conn = _connect()
    blob = _to_blob(features)
    video_id = _extract_video_id(source)
    thumbnail = _thumbnail(video_id)

    existing = conn.execute("SELECT id FROM tracks WHERE source = ?", (source,)).fetchone()
    if existing:
        conn.execute(
            """UPDATE tracks SET title=?, artist=?, year=?, notes=?,
               duration=?, label=?, features=?, feature_version=?,
               audio_path=COALESCE(?, audio_path),
               thumbnail_url=COALESCE(thumbnail_url, ?),
               video_id=COALESCE(video_id, ?)
               WHERE id=?""",
            (title, artist, year, notes, duration, label, blob, feature_version,
             audio_path, thumbnail, video_id, existing["id"]),
        )
        conn.commit()
        return existing["id"]

    cur = conn.execute(
        """INSERT INTO tracks
           (source, title, artist, year, notes, duration, label, features,
            feature_version, audio_path, thumbnail_url, video_id, added_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (source, title, artist, year, notes, duration, label, blob,
         feature_version, audio_path, thumbnail, video_id,
         datetime.utcnow().isoformat()),
    )
    conn.commit()
    return cur.lastrowid


def get_track_by_id(track_id: int) -> Optional[dict]:
    conn = _connect()
    row = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["features"] = _from_blob(d["features"])
    return d


def get_track_by_source(source: str) -> Optional[dict]:
    conn = _connect()
    row = conn.execute("SELECT * FROM tracks WHERE source = ?", (source,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["features"] = _from_blob(d["features"])
    return d


def get_all_tracks(
    labeled_only: bool = False,
    label_filter: Optional[list] = None,
) -> list[dict]:
    conn = _connect()
    q = "SELECT * FROM tracks"
    params: list = []
    if label_filter:
        placeholders = ",".join("?" * len(label_filter))
        q += f" WHERE label IN ({placeholders})"
        params = list(label_filter)
    elif labeled_only:
        q += " WHERE label NOT IN ('untagged')"
    rows = conn.execute(q, params).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["features"] = _from_blob(d["features"])
        result.append(d)
    return result


def update_label(track_id: int, label: str) -> None:
    conn = _connect()
    conn.execute("UPDATE tracks SET label=? WHERE id=?", (label, track_id))
    conn.commit()


def set_audio_path(track_id: int, path: str) -> None:
    conn = _connect()
    conn.execute("UPDATE tracks SET audio_path=? WHERE id=?", (path, track_id))
    conn.commit()


def count_by_label() -> dict:
    conn = _connect()
    rows = conn.execute("SELECT label, COUNT(*) as n FROM tracks GROUP BY label").fetchall()
    return {r["label"]: r["n"] for r in rows}


# ── Votes ────────────────────────────────────────────────────────────────────

def record_vote(track_id: int, vote: str) -> None:
    """Record a user vote and update the track label accordingly."""
    label_map = {"liked": "liked", "disliked": "disliked", "skipped": "untagged"}
    conn = _connect()
    conn.execute(
        "INSERT INTO votes (track_id, vote, voted_at) VALUES (?,?,?)",
        (track_id, vote, datetime.utcnow().isoformat()),
    )
    if vote in label_map:
        conn.execute("UPDATE tracks SET label=? WHERE id=?", (label_map[vote], track_id))
    conn.execute(
        "UPDATE candidates SET status='voted', voted_at=? WHERE track_id=? AND status='served'",
        (datetime.utcnow().isoformat(), track_id),
    )
    conn.commit()


def get_vote_counts() -> dict:
    conn = _connect()
    rows = conn.execute("SELECT vote, COUNT(*) n FROM votes GROUP BY vote").fetchall()
    return {r["vote"]: r["n"] for r in rows}


def get_labeled_for_training() -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where y: 1=liked, 0=disliked. Excludes skipped/untagged."""
    tracks = get_all_tracks(label_filter=["liked", "top10", "disliked", "hors_sujet"])
    X, y = [], []
    for t in tracks:
        if t["features"] is None:
            continue
        X.append(t["features"])
        y.append(1 if t["label"] in ("liked", "top10") else 0)
    if not X:
        return np.empty((0, 0)), np.empty(0)
    return np.stack(X), np.array(y, dtype=int)


# ── Candidates ───────────────────────────────────────────────────────────────

def enqueue_candidate(track_id: int, score: float, seed_artist: str, source: str = "youtube_search") -> None:
    conn = _connect()
    existing = conn.execute(
        "SELECT id FROM candidates WHERE track_id=? AND status='pending'", (track_id,)
    ).fetchone()
    if existing:
        return
    conn.execute(
        "INSERT INTO candidates (track_id, score, seed_artist, source, queued_at) VALUES (?,?,?,?,?)",
        (track_id, score, seed_artist, source, datetime.utcnow().isoformat()),
    )
    conn.commit()


def pop_next_candidate() -> Optional[dict]:
    """Return the highest-scoring pending candidate and mark it served."""
    conn = _connect()
    row = conn.execute(
        """SELECT c.id as cid, t.* FROM candidates c
           JOIN tracks t ON c.track_id = t.id
           WHERE c.status = 'pending'
           ORDER BY c.score DESC
           LIMIT 1"""
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["features"] = _from_blob(d["features"])
    conn.execute(
        "UPDATE candidates SET status='served', served_at=? WHERE id=?",
        (datetime.utcnow().isoformat(), d["cid"]),
    )
    conn.commit()
    return d


def pending_candidate_count() -> int:
    conn = _connect()
    return conn.execute("SELECT COUNT(*) FROM candidates WHERE status='pending'").fetchone()[0]


def already_seen(source: str) -> bool:
    """True if this source was already ingested or is in the candidate queue."""
    conn = _connect()
    row = conn.execute("SELECT id FROM tracks WHERE source=?", (source,)).fetchone()
    return row is not None
