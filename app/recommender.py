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

_tag_counts: dict[str, int] = {}


def update_model(liked: list[dict], rejected: list[dict]) -> None:
    global _model, _top_tags, _seed_artists, _tag_counts
    from sklearn.linear_model import SGDClassifier

    all_c = liked + rejected
    if not all_c:
        return

    # Accumulate tag counts from liked (never shrink)
    for c in liked:
        for t in json.loads(c.get("tags_json") or "[]"):
            _tag_counts[t.lower()] = _tag_counts.get(t.lower(), 0) + 1

    new_top_tags = sorted(_tag_counts, key=lambda k: _tag_counts[k], reverse=True)[:50]
    new_seed_artists = list(set(_seed_artists) | {c.get("seed_artist", "") for c in all_c})

    # If feature dimensions changed, reset the model to avoid mismatch
    if new_top_tags != _top_tags or set(new_seed_artists) != set(_seed_artists):
        if _model is not None:
            log.info("Feature dimensions changed — resetting model")
            _model = None
        _top_tags = new_top_tags
        _seed_artists = new_seed_artists

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
        from app.db import (pool_size, get_liked_tracks, get_all_seen_permanent,
                             get_recent_skipped, enqueue_candidate)
        from app.lastfm import get_similar_tracks, get_similar_artists, get_top_tracks, get_artist_info
        from app.youtube import get_yt_id

        if pool_size() >= POOL_MIN:
            return

        seen = get_all_seen_permanent() | get_recent_skipped()
        liked_tracks = get_liked_tracks()
        if not liked_tracks:
            log.info("Pool fill: no liked tracks yet")
            return

        added = 0
        for liked in liked_tracks:
            if added >= 60:
                break

            # Primary: track-level similarity
            candidates_to_add = []
            similar_tracks = await get_similar_tracks(liked["artist"], liked["title"], limit=20)
            if similar_tracks:
                candidates_to_add = [
                    {"artist": s["artist"], "title": s["title"], "match": s["match"]}
                    for s in similar_tracks
                ]
                log.info("Track.getSimilar: %d results for %s — %s", len(candidates_to_add), liked["artist"], liked["title"])
            else:
                # Fallback: find similar artists, take their top tracks
                log.info("Pool fill: no track-similar for %s — %s, falling back to artist-similar", liked["artist"], liked["title"])
                similar_artists = await get_similar_artists(liked["artist"], limit=15)
                for sim_artist in similar_artists[:10]:
                    name = sim_artist["name"]
                    if name.lower() in exclusion_set:
                        continue
                    tracks = await get_top_tracks(name, limit=3)
                    for t in tracks:
                        candidates_to_add.append({"artist": name, "title": t["title"], "match": sim_artist["match"]})

            for cand in candidates_to_add:
                if added >= 60:
                    break
                artist_name = cand["artist"]
                title = cand["title"]

                if artist_name.lower() in exclusion_set:
                    continue
                key = (artist_name.lower(), title.lower())
                if key in seen:
                    continue

                info = await get_artist_info(artist_name)
                yt_id = await get_yt_id(artist_name, title)
                if not yt_id:
                    continue

                score = _bootstrap_score(info.get("playcount", 0))
                enqueue_candidate(
                    artist=artist_name,
                    title=title,
                    yt_id=yt_id,
                    year=None,
                    playcount=info.get("playcount", 0),
                    listeners=info.get("listeners", 0),
                    tags=info.get("tags", []),
                    seed_artist=liked["artist"],
                    similarity_score=cand["match"],
                    score=score,
                )
                seen.add(key)
                added += 1
                log.info("Pool: +%s — %s (seed=%s — %s)", artist_name, title, liked["artist"], liked["title"])

        log.info("Pool fill done: added=%d pool_size=%d", added, pool_size())
