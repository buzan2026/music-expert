"""SQLite persistence — verdicts + candidate queue."""
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.environ.get("DB_PATH", "verdicts.db"))

BOOTSTRAP_LIKED = [
    {"artist": "Jamie Woon",   "title": "Sharpness",                           "year": 2010, "yt_id": "iVawIrs5-fs"},
    {"artist": "RAYE",         "title": "Worth It",                            "year": 2023, "yt_id": "Ojz-mAn6TDo"},
    {"artist": "Jason Piccioni","title": "Cos It's Love That Really Matters",  "year": 2026, "yt_id": None},
    {"artist": "Robin Thicke", "title": "Ain't No Hat 4 That",                 "year": 2013, "yt_id": None},
    {"artist": "KINGH",        "title": "RIDE (feat. Kojey Radical)",          "year": 2026, "yt_id": None},
    {"artist": "Amber Mark",   "title": "Foreign Things",                      "year": 2022, "yt_id": None},
]

BOOTSTRAP_REJECTED = [
    {"artist": "Michael Kiwanuka", "title": "You Ain't The Problem", "year": 2019},
    {"artist": "Hiatus Kaiyote",   "title": "Get Sun",               "year": 2024},
    {"artist": "The Internet",     "title": "Come Together",         "year": 2018},
    {"artist": "Kadhja Bonet",     "title": "Mother Maybe",          "year": 2018},
    {"artist": "Cleo Sol",         "title": "Sunshine",              "year": 2020},
]


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS verdicts (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                artist  TEXT    NOT NULL,
                title   TEXT    NOT NULL,
                yt_id   TEXT,
                year    INTEGER,
                verdict TEXT    NOT NULL CHECK(verdict IN ('LIKED','REJECTED','SKIPPED')),
                ts      REAL    NOT NULL DEFAULT (unixepoch())
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_verdicts_track
                ON verdicts(lower(artist), lower(title));
            CREATE INDEX IF NOT EXISTS idx_verdicts_verdict
                ON verdicts(verdict);

            CREATE TABLE IF NOT EXISTS candidate_queue (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                artist           TEXT    NOT NULL,
                title            TEXT    NOT NULL,
                yt_id            TEXT    NOT NULL,
                year             INTEGER,
                playcount        INTEGER DEFAULT 0,
                listeners        INTEGER DEFAULT 0,
                tags_json        TEXT    DEFAULT '[]',
                seed_artist      TEXT,
                similarity_score REAL    DEFAULT 0.0,
                score            REAL    DEFAULT 0.5,
                added_at         REAL    NOT NULL DEFAULT (unixepoch())
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_queue_track
                ON candidate_queue(lower(artist), lower(title));
        """)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── verdicts ────────────────────────────────────────────────────────────────

def insert_verdict(artist: str, title: str, verdict: str,
                   yt_id: Optional[str] = None, year: Optional[int] = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO verdicts(artist, title, yt_id, year, verdict)
               VALUES(?,?,?,?,?)""",
            (artist, title, yt_id, year, verdict),
        )


def get_verdict(artist: str, title: str) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT verdict FROM verdicts WHERE lower(artist)=lower(?) AND lower(title)=lower(?)",
            (artist, title),
        ).fetchone()
    return row["verdict"] if row else None


def get_liked_artists() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT artist FROM verdicts WHERE verdict='LIKED'"
        ).fetchall()
    return [r["artist"] for r in rows]


def get_all_seen_permanent() -> set[tuple[str, str]]:
    """LIKED + REJECTED — never show again."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT artist, title FROM verdicts WHERE verdict IN ('LIKED','REJECTED')"
        ).fetchall()
    return {(r["artist"].lower(), r["title"].lower()) for r in rows}


def get_recent_skipped(days: int = 7) -> set[tuple[str, str]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT artist, title FROM verdicts "
            "WHERE verdict='SKIPPED' AND ts > unixepoch() - ?",
            (days * 86400,),
        ).fetchall()
    return {(r["artist"].lower(), r["title"].lower()) for r in rows}


def get_old_skipped(days: int = 7) -> list[dict]:
    """Skipped > `days` ago — eligible for 30% re-proposal."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM verdicts "
            "WHERE verdict='SKIPPED' AND ts <= unixepoch() - ?",
            (days * 86400,),
        ).fetchall()
    return [dict(r) for r in rows]


def count_verdicts() -> dict[str, int]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT verdict, count(*) as n FROM verdicts GROUP BY verdict"
        ).fetchall()
    return {r["verdict"]: r["n"] for r in rows}


# ── candidate queue ──────────────────────────────────────────────────────────

def pool_size() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT count(*) as n FROM candidate_queue").fetchone()
    return row["n"]


def enqueue_candidate(artist: str, title: str, yt_id: str, year: Optional[int],
                      playcount: int, listeners: int, tags: list[str],
                      seed_artist: str, similarity_score: float, score: float) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO candidate_queue
               (artist,title,yt_id,year,playcount,listeners,tags_json,
                seed_artist,similarity_score,score)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (artist, title, yt_id, year, playcount, listeners,
             json.dumps(tags), seed_artist, similarity_score, score),
        )


def remove_candidate(artist: str, title: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM candidate_queue "
            "WHERE lower(artist)=lower(?) AND lower(title)=lower(?)",
            (artist, title),
        )


def get_top_candidates(limit: int = 200) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM candidate_queue ORDER BY score DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── bootstrap ────────────────────────────────────────────────────────────────

def seed_bootstrap(get_yt_id_fn) -> None:
    """Run once: insert bootstrap verdicts, looking up missing yt_ids."""
    for track in BOOTSTRAP_LIKED:
        if get_verdict(track["artist"], track["title"]):
            continue
        yt_id = track.get("yt_id") or get_yt_id_fn(track["artist"], track["title"])
        insert_verdict(track["artist"], track["title"], "LIKED",
                       yt_id=yt_id, year=track.get("year"))

    for track in BOOTSTRAP_REJECTED:
        if get_verdict(track["artist"], track["title"]):
            continue
        insert_verdict(track["artist"], track["title"], "REJECTED",
                       year=track.get("year"))
