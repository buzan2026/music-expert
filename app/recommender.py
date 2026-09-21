"""Recommendation engine — bootstrap scoring + SGDClassifier online learning."""
import asyncio
import json
import logging
import math
import random
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

POOL_MIN = 20
_fill_lock = asyncio.Lock()

_model = None          # SGDClassifier, None until >= 20 verdicts
_top_tags: list[str] = []
_seed_artists: list[str] = []


# ── scoring ──────────────────────────────────────────────────────────────────

def _bootstrap_score(playcount: int) -> float:
    rarity = 1.0 / math.log(max(playcount, 1) + 2)
    return rarity * (0.7 + 0.3 * random.random())


def _features(candidates: list[dict]) -> np.ndarray:
    rows = []
    for c in candidates:
        tags = [t.lower() for t in json.loads(c.get("tags_json") or "[]")]
        row = [
            math.log(max(c.get("listeners", 0), 1) + 1),
            math.log(max(c.get("playcount",  0), 1) + 1),
            (c.get("year") or 0) / 2000.0,
        ]
        row += [1.0 if t.lower() in tags else 0.0 for t in _top_tags]
        row += [1.0 if c.get("seed_artist", "") == s else 0.0 for s in _seed_artists]
        row.append(float(c.get("similarity_score", 0.0)))
        rows.append(row)
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 1))


# ── pick ─────────────────────────────────────────────────────────────────────

def pick_next(candidates: list[dict], old_skipped: list[dict],
              total_verdicts: int) -> Optional[dict]:
    if not candidates and not old_skipped:
        return None

    # 30% chance to re-surface an old skipped track
    if old_skipped and random.random() < 0.3:
        return random.choice(old_skipped)

    if not candidates:
        return None

    if total_verdicts < 20 or _model is None:
        for c in candidates:
            if "_score" not in c:
                c["_score"] = _bootstrap_score(c.get("playcount", 0))
        candidates.sort(key=lambda x: x.get("_score", 0), reverse=True)
        # 70% exploit top, 30% explore
        if random.random() < 0.7 or len(candidates) == 1:
            return candidates[0]
        pool = candidates[len(candidates) // 2:] or candidates
        return random.choice(pool)

    X = _features(candidates)
    if X.shape[0] == 0:
        return candidates[0]

    try:
        probs = _model.predict_proba(X)[:, 1]
    except Exception:
        return candidates[0]

    if random.random() < 0.7:
        return candidates[int(np.argmax(probs))]
    uncertainty = np.abs(probs - 0.5)
    return candidates[int(np.argmin(uncertainty))]


# ── model update ─────────────────────────────────────────────────────────────

def update_model(liked: list[dict], rejected: list[dict]) -> None:
    global _model, _top_tags, _seed_artists
    from sklearn.linear_model import SGDClassifier

    all_c = liked + rejected
    if not all_c:
        return

    # Rebuild top tags from liked
    tag_counts: dict[str, int] = {}
    for c in liked:
        for t in json.loads(c.get("tags_json") or "[]"):
            tag_counts[t.lower()] = tag_counts.get(t.lower(), 0) + 1
    _top_tags = sorted(tag_counts, key=lambda k: tag_counts[k], reverse=True)[:50]
    _seed_artists = list({c.get("seed_artist", "") for c in all_c})

    X = _features(all_c)
    y = [1] * len(liked) + [0] * len(rejected)
    if X.shape[0] == 0:
        return

    if _model is None:
        _model = SGDClassifier(loss="log_loss", warm_start=True, random_state=42)
    try:
        _model.partial_fit(X, y, classes=[0, 1])
    except Exception as e:
        log.warning("Model partial_fit failed: %s", e)


# ── pool fill ─────────────────────────────────────────────────────────────────

async def fill_pool(exclusion_set: set[str]) -> None:
    async with _fill_lock:
        from app.db import (pool_size, get_liked_artists, get_all_seen_permanent,
                             get_recent_skipped, enqueue_candidate)
        from app.lastfm import get_similar_artists, get_top_tracks, get_artist_info
        from app.youtube import get_yt_id

        if pool_size() >= POOL_MIN:
            return

        seen = get_all_seen_permanent() | get_recent_skipped()
        liked_artists = get_liked_artists()
        if not liked_artists:
            log.info("Pool fill: no liked artists yet")
            return

        added = 0
        for seed in liked_artists:
            if added >= 60:
                break

            similar = await get_similar_artists(seed, limit=30)
            if len(similar) < 3:
                log.info("Pool fill: too few similar for %s, skipping seed", seed)
                continue

            for sim in similar:
                name = sim["name"]
                if name.lower() in exclusion_set:
                    continue

                info = await get_artist_info(name)
                tracks = await get_top_tracks(name, limit=5)

                for track in tracks:
                    key = (name.lower(), track["title"].lower())
                    if key in seen:
                        continue

                    yt_id = await get_yt_id(name, track["title"])
                    if not yt_id:
                        continue

                    score = _bootstrap_score(info.get("playcount", 0))
                    enqueue_candidate(
                        artist=name,
                        title=track["title"],
                        yt_id=yt_id,
                        year=None,
                        playcount=info.get("playcount", 0),
                        listeners=info.get("listeners", 0),
                        tags=info.get("tags", []),
                        seed_artist=seed,
                        similarity_score=sim["match"],
                        score=score,
                    )
                    seen.add(key)
                    added += 1
                    log.info("Pool: +%s — %s (seed=%s)", name, track["title"], seed)

        log.info("Pool fill done: added=%d pool_size=%d", added, pool_size())
