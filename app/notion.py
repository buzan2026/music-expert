"""Notion client — exclusion set loading + LIKED track writing."""
import asyncio
import logging
import os
from datetime import date
from functools import lru_cache
from typing import Optional

from notion_client import AsyncClient

log = logging.getLogger(__name__)

MORCEAUX_DB = "a5ca6315-9d38-4246-85ba-344204308304"
ARTISTES_DB = "6c426920-4e3d-465b-a80a-f642da7374ef"

_exclusion_set: set[str] = set()


def _client() -> AsyncClient:
    return AsyncClient(auth=os.environ["NOTION_TOKEN"])


async def load_exclusion_set() -> set[str]:
    global _exclusion_set
    client = _client()
    artists: set[str] = set()
    cursor = None

    while True:
        kwargs: dict = {"database_id": ARTISTES_DB, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        try:
            resp = await client.databases.query(**kwargs)
        except Exception as e:
            log.error("Notion exclusion set load failed: %s", e)
            break

        for page in resp["results"]:
            parts = page["properties"]["Nom"]["title"]
            name = "".join(p.get("plain_text", "") for p in parts).strip()
            if name and not name.startswith("🗑️ DOUBLON"):
                artists.add(name.lower())

        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    _exclusion_set = artists
    log.info("Notion: exclusion set loaded (%d artists)", len(_exclusion_set))
    return _exclusion_set


def is_excluded(artist_name: str) -> bool:
    return artist_name.lower() in _exclusion_set


async def _get_or_create_artist(client: AsyncClient, artist_name: str) -> str:
    resp = await client.databases.query(
        database_id=ARTISTES_DB,
        filter={"property": "Nom", "title": {"equals": artist_name}},
    )
    for page in resp["results"]:
        parts = page["properties"]["Nom"]["title"]
        name = "".join(p.get("plain_text", "") for p in parts).strip()
        if not name.startswith("🗑️ DOUBLON"):
            return page["id"]

    new_page = await client.pages.create(
        parent={"database_id": ARTISTES_DB},
        properties={"Nom": {"title": [{"text": {"content": artist_name}}]}},
    )
    return new_page["id"]


async def write_liked_track(artist: str, title: str,
                            yt_id: Optional[str], year: Optional[int]) -> None:
    for attempt in range(3):
        try:
            client = _client()
            artist_page_id = await _get_or_create_artist(client, artist)

            today = date.today().isoformat()
            yt_url = (f"https://www.youtube.com/watch?v={yt_id}" if yt_id else "inconnue")
            year_str = str(year) if year else "inconnue"

            await client.pages.create(
                parent={"database_id": MORCEAUX_DB},
                properties={
                    "Titre":       {"title":        [{"text": {"content": title}}]},
                    "Artiste":     {"relation":      [{"id": artist_page_id}]},
                    "Genre":       {"multi_select":  [{"name": "Soul/Funk"}]},
                    "Coup de cœur": {"checkbox":     False},
                },
                children=[
                    _para("Statut : 📥 Boîte de réception"),
                    _para(f"Source : Découverte {today}"),
                    _para(f"Lien YouTube : {yt_url}"),
                    _para(f"Année : {year_str}"),
                ],
            )
            log.info("Notion: wrote %s — %s", artist, title)
            return
        except Exception as e:
            if attempt < 2:
                delay = 4 ** attempt  # 1s, 4s
                log.warning("Notion write retry %d for %s — %s: %s", attempt + 1, artist, title, e)
                await asyncio.sleep(delay)
            else:
                log.error("Notion write failed (3 attempts) for %s — %s: %s", artist, title, e)


def _para(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }
