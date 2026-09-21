"""
Candidate discovery worker.

Algorithm:
  1. Collect liked artists from DB (top10 + liked labels)
  2. For each artist: yt-dlp metadata search (no download) → candidate URLs
  3. Filter already-seen
  4. Download + fast fingerprint (mix only, no stems)
  5. Score with current model (or cosine heuristic)
  6. Enqueue top-N

Also supports Last.fm artist.getSimilar when LASTFM_API_KEY is set
(env var or ~/.music-fp/lastfm_key).
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from .config import DATA_DIR, ensure_dirs

MIN_QUEUE_SIZE = 5       # refill when queue drops below this
REFILL_TARGET = 10       # fill up to this many pending candidates
SEARCH_PER_ARTIST = 8    # yt-dlp ytsearch results per artist
SCORE_THRESHOLD = 0.35   # discard candidates predicted clearly disliked


def _lastfm_key() -> Optional[str]:
    key = os.environ.get("LASTFM_API_KEY")
    if key:
        return key
    p = DATA_DIR / "lastfm_key"
    if p.exists():
        return p.read_text().strip()
    return None


def _lastfm_similar_artists(artist: str, limit: int = 5) -> list[str]:
    key = _lastfm_key()
    if not key:
        return []
    try:
        import requests
        resp = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={"method": "artist.getSimilar", "artist": artist,
                    "api_key": key, "format": "json", "limit": limit},
            timeout=10,
        )
        data = resp.json()
        return [a["name"] for a in data.get("similarartists", {}).get("artist", [])]
    except Exception:
        return []


def _yt_search_metadata(query: str, n: int = SEARCH_PER_ARTIST) -> list[dict]:
    """Run yt-dlp metadata search without downloading; return list of info dicts."""
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--dump-json",
        "--quiet",
        "--no-warnings",
        "--flat-playlist",
        f"ytsearch{n}:{query}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    items = []
    for line in result.stdout.splitlines():
        try:
            d = json.loads(line)
            items.append(d)
        except json.JSONDecodeError:
            pass
    return items


def _liked_artists() -> list[str]:
    from .store import get_all_tracks
    tracks = get_all_tracks(label_filter=["top10", "liked"])
    seen, artists = set(), []
    for t in tracks:
        a = t.get("artist") or ""
        if a and a.lower() not in seen:
            seen.add(a.lower())
            artists.append(a)
    return artists


def _fingerprint_and_enqueue(url: str, seed_artist: str) -> bool:
    """Download, fingerprint (mix only), score, and enqueue. Returns True on success."""
    from .store import already_seen, upsert_track, enqueue_candidate
    from .download import fetch
    from .fingerprint import extract
    from .model import get_model

    if already_seen(url):
        return False

    try:
        audio_path, title = fetch(url, progress=False)
    except Exception:
        return False

    try:
        features = extract(audio_path, use_stems=False)
    except Exception:
        return False

    try:
        import librosa
        duration = librosa.get_duration(path=str(audio_path))
    except Exception:
        duration = None

    model = get_model()
    liked = []
    if not model.is_trained:
        from .store import get_all_tracks
        liked = get_all_tracks(label_filter=["top10", "liked"])

    score = model.score(features, liked_tracks=liked)

    if score < SCORE_THRESHOLD:
        return False

    track_id = upsert_track(
        source=url,
        title=title or Path(audio_path).stem,
        duration=duration,
        label="untagged",
        features=features,
        artist=seed_artist,
        audio_path=str(audio_path),
    )

    enqueue_candidate(track_id, score=score, seed_artist=seed_artist)
    return True


def run_discovery_cycle(max_new: int = REFILL_TARGET) -> int:
    """
    Run one discovery cycle. Returns number of candidates added.
    Blocks until done — call from a background thread.
    """
    from .store import pending_candidate_count

    already = pending_candidate_count()
    if already >= REFILL_TARGET:
        return 0

    need = max_new - already
    artists = _liked_artists()
    if not artists:
        return 0

    # Optionally expand with Last.fm similar artists
    expanded = list(artists)
    for a in artists[:3]:
        expanded.extend(_lastfm_similar_artists(a, limit=3))
    # deduplicate while preserving order
    seen_a: set = set()
    unique_artists = []
    for a in expanded:
        if a.lower() not in seen_a:
            seen_a.add(a.lower())
            unique_artists.append(a)

    added = 0
    for artist in unique_artists:
        if added >= need:
            break
        query = f"{artist} official audio"
        candidates = _yt_search_metadata(query, n=SEARCH_PER_ARTIST)
        for info in candidates:
            if added >= need:
                break
            url = info.get("url") or info.get("webpage_url")
            if not url:
                # flat-playlist gives 'id' → reconstruct URL
                vid = info.get("id")
                if vid:
                    url = f"https://www.youtube.com/watch?v={vid}"
                else:
                    continue
            if _fingerprint_and_enqueue(url, seed_artist=artist):
                added += 1

    return added


class DiscoveryWorker:
    """Background thread that keeps the candidate queue topped up."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="discovery")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def _loop(self) -> None:
        from .store import pending_candidate_count
        while not self._stop.is_set():
            try:
                if pending_candidate_count() < MIN_QUEUE_SIZE:
                    self._running.set()
                    run_discovery_cycle()
                    self._running.clear()
            except Exception:
                self._running.clear()
            # Sleep 60s between checks, but wake immediately if stopped
            self._stop.wait(timeout=60)
