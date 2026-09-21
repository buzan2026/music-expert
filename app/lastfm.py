"""Last.fm client via pylast — similar artists + top tracks."""
import asyncio
import logging
import os
from functools import lru_cache
from typing import Optional

import pylast

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _network() -> pylast.LastFMNetwork:
    return pylast.LastFMNetwork(api_key=os.environ["LASTFM_API_KEY"])


def _run(fn):
    return asyncio.get_event_loop().run_in_executor(None, fn)


async def get_similar_artists(artist_name: str, limit: int = 30) -> list[dict]:
    """Returns [{name, match}] or [] on failure."""
    def _fetch():
        try:
            artist = _network().get_artist(artist_name)
            return [{"name": s.item.name, "match": float(s.match or 0)}
                    for s in artist.get_similar(limit=limit)]
        except Exception as e:
            log.warning("Last.fm getSimilar(%s): %s", artist_name, e)
            return []
    return await _run(_fetch)


async def get_top_tracks(artist_name: str, limit: int = 5) -> list[dict]:
    """Returns [{title, playcount}] or [] on failure."""
    def _fetch():
        results = []
        try:
            for t in _network().get_artist(artist_name).get_top_tracks(limit=limit):
                try:
                    results.append({"title": t.item.title, "playcount": int(t.weight or 0)})
                except Exception:
                    pass
        except Exception as e:
            log.warning("Last.fm getTopTracks(%s): %s", artist_name, e)
        return results
    return await _run(_fetch)


async def get_artist_info(artist_name: str) -> dict:
    """Returns {listeners, playcount, tags[]}."""
    def _fetch():
        try:
            a = _network().get_artist(artist_name)
            return {
                "listeners": int(a.get_listener_count() or 0),
                "playcount":  int(a.get_playcount() or 0),
                "tags":       [t.item.name for t in (a.get_top_tags(limit=5) or [])],
            }
        except Exception as e:
            log.warning("Last.fm getArtistInfo(%s): %s", artist_name, e)
            return {"listeners": 0, "playcount": 0, "tags": []}
    return await _run(_fetch)
