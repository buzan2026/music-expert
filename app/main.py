"""FastAPI app — routes, lifespan, static files."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.db import (
    init_db, seed_bootstrap, count_verdicts, pool_size,
    get_top_candidates, remove_candidate, insert_verdict,
    get_old_skipped,
)
from app.notion import load_exclusion_set, write_liked_track
from app.youtube import get_yt_id_sync
from app.recommender import fill_pool, pick_next, update_model, POOL_MIN

STATIC_DIR = Path(__file__).parent.parent / "static"

_exclusion_set: set[str] = set()
_bg_task: asyncio.Task | None = None


async def _pool_monitor() -> None:
    while True:
        try:
            if pool_size() < POOL_MIN:
                await fill_pool(_exclusion_set)
        except Exception as e:
            log.error("Pool monitor: %s", e)
        await asyncio.sleep(60)


async def _check_connections() -> dict:
    """Vérifie Last.fm et Notion au démarrage, affiche un résumé clair."""
    status = {}

    # Last.fm
    try:
        import pylast
        network = pylast.LastFMNetwork(api_key=os.environ["LASTFM_API_KEY"])
        network.get_artist("RAYE").get_similar(limit=1)
        status["lastfm"] = "✓ Last.fm OK"
    except KeyError:
        status["lastfm"] = "✗ Last.fm : LASTFM_API_KEY manquant dans .env"
    except Exception as e:
        status["lastfm"] = f"✗ Last.fm : {e}"

    # Notion
    try:
        from app.notion import _client, ARTISTES_DB
        client = _client()
        await client.databases.retrieve(database_id=ARTISTES_DB)
        status["notion"] = "✓ Notion OK"
    except KeyError:
        status["notion"] = "✗ Notion : NOTION_TOKEN manquant dans .env"
    except Exception as e:
        status["notion"] = f"✗ Notion : {e}"

    return status


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _exclusion_set, _bg_task

    init_db()

    print("\n" + "="*50)
    print("  music-boris — vérification des connexions")
    print("="*50)
    checks = await _check_connections()
    for v in checks.values():
        print(" ", v)
    print("="*50 + "\n")

    log.info("Bootstrap en cours…")
    seed_bootstrap(get_yt_id_sync)
    log.info("Bootstrap OK (%s)", count_verdicts())

    if "✓" in checks.get("notion", ""):
        try:
            _exclusion_set = await load_exclusion_set()
        except Exception as e:
            log.warning("Notion exclusion set : %s", e)

    _bg_task = asyncio.create_task(_pool_monitor())
    log.info("Prêt sur http://localhost:8000")

    yield

    if _bg_task:
        _bg_task.cancel()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ── GET /api/next ─────────────────────────────────────────────────────────────

@app.get("/api/next")
async def api_next():
    counts = count_verdicts()
    total = sum(counts.values())

    candidates = get_top_candidates()
    old_skipped = get_old_skipped()
    chosen = pick_next(candidates, old_skipped, total)

    if not chosen:
        return {"ready": False, "pool_size": pool_size()}

    return {
        "ready":    True,
        "id":       chosen.get("id"),
        "artist":   chosen["artist"],
        "title":    chosen["title"],
        "yt_id":    chosen["yt_id"],
        "year":     chosen.get("year"),
        "seed":     chosen.get("seed_artist"),
        "playcount": chosen.get("playcount", 0),
        "tags_json": chosen.get("tags_json", "[]"),
        "similarity_score": chosen.get("similarity_score", 0.0),
    }


# ── POST /api/verdict ─────────────────────────────────────────────────────────

class VerdictIn(BaseModel):
    verdict:   str            # LIKED | REJECTED | SKIPPED
    artist:    str
    title:     str
    yt_id:     Optional[str] = None
    year:      Optional[int] = None
    playcount: Optional[int] = None
    tags_json: Optional[str] = None
    similarity_score: Optional[float] = None


@app.post("/api/verdict")
async def api_verdict(req: VerdictIn):
    if req.verdict not in ("LIKED", "REJECTED", "SKIPPED"):
        raise HTTPException(400, "verdict must be LIKED, REJECTED or SKIPPED")

    insert_verdict(req.artist, req.title, req.verdict,
                   yt_id=req.yt_id, year=req.year)
    remove_candidate(req.artist, req.title)

    if req.verdict == "LIKED":
        asyncio.create_task(
            write_liked_track(req.artist, req.title, req.yt_id, req.year)
        )

    # Incremental model update (non-blocking)
    if req.verdict in ("LIKED", "REJECTED"):
        track = {
            "artist": req.artist, "title": req.title,
            "playcount": req.playcount or 0,
            "tags_json": req.tags_json or "[]",
            "similarity_score": req.similarity_score or 0.0,
        }
        asyncio.create_task(
            asyncio.to_thread(_sync_model_update, track, req.verdict)
        )

    return {"ok": True}


def _sync_model_update(track: dict, verdict: str) -> None:
    liked    = [track] if verdict == "LIKED"    else []
    rejected = [track] if verdict == "REJECTED" else []
    update_model(liked, rejected)


# ── GET /api/status ───────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    counts = count_verdicts()
    return {
        "pool_size": pool_size(),
        "verdicts":  counts,
        "total":     sum(counts.values()),
    }
