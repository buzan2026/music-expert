"""
Candidate discovery worker.

Algorithm:
  1. Collect liked artists from DB (top10 + liked labels).
     Falls back to SEED_ARTISTS when no liked artists yet.
  2. For each artist: yt-dlp metadata search → candidate metadata
  3. Filter already-seen
  4. Enqueue (metadata-only, no download by default)
  5. Optionally: download + fingerprint for ML scoring
     (enabled by MUSIC_FP_FINGERPRINT=1 env var)

Also supports Last.fm artist.getSimilar when LASTFM_API_KEY is set.
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

from .config import DATA_DIR, ensure_dirs

MIN_QUEUE_SIZE = 5
REFILL_TARGET = 15
SEARCH_PER_ARTIST = 8
SCORE_THRESHOLD = 0.35

# Seed artists that match Boris's taste profile (funky bass, dry drums,
# sensual vocals, mid-up tempo, readable mix).
SEED_ARTISTS = [
    "Jamie Woon",
    "RAYE",
    "Robin Thicke",
    "Amber Mark",
    "Tom Misch",
    "Cleo Sol",
    "Mahalia",
    "Joy Crookes",
    "Masego",
    "Daniel Caesar",
    "Jorja Smith",
    "Leon Bridges",
    "Sault",
    "Yebba",
    "H.E.R.",
    "Lucky Daye",
    "SiR",
    "Ari Lennox",
    "Sudan Archives",
    "Samm Henshaw",
    "Kojey Radical",
    "Pa Salieu",
    "Knucks",
    "Greentea Peng",
    "Biig Piig",
]

# Title fragments that indicate unwanted content
_SKIP_KEYWORDS = frozenset([
    "remix", "cover", "tribute", "karaoke", "instrumental",
    "lyrics", "lyric video", "reaction", "tutorial",
])


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


def _yt_search_metadata(query: str, n: int = SEARCH_PER_ARTIST) -> List[dict]:
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


def _liked_artists() -> List[str]:
    from .store import get_all_tracks
    tracks = get_all_tracks(label_filter=["top10", "liked"])
    seen, artists = set(), []
    for t in tracks:
        a = t.get("artist") or ""
        if a and a.lower() not in seen:
            seen.add(a.lower())
            artists.append(a)
    return artists


def _enqueue_from_metadata(info: dict, seed_artist: str) -> bool:
    """
    Enqueue a YouTube track from yt-dlp flat-playlist metadata only.
    No audio download or fingerprinting needed.
    """
    from .store import already_seen, upsert_track, enqueue_candidate

    vid = info.get("id")
    if not vid or len(vid) != 11:
        return False

    url = f"https://www.youtube.com/watch?v={vid}"
    if already_seen(url):
        return False

    title = info.get("title") or info.get("fulltitle") or ""
    low = title.lower()
    if any(kw in low for kw in _SKIP_KEYWORDS):
        return False

    duration = info.get("duration")
    if duration and (duration < 60 or duration > 600):
        return False

    try:
        track_id = upsert_track(
            source=url,
            title=title,
            duration=float(duration) if duration else None,
            label="untagged",
            features=None,
            artist=seed_artist,
        )
        enqueue_candidate(track_id, score=0.5, seed_artist=seed_artist, source="yt_metadata")
        return True
    except Exception:
        return False


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

    Uses metadata-only enqueueing by default (fast, no download).
    Set MUSIC_FP_FINGERPRINT=1 to enable audio fingerprinting for scoring.
    """
    from .store import pending_candidate_count

    already = pending_candidate_count()
    if already >= REFILL_TARGET:
        return 0

    need = max_new - already
    use_fingerprint = os.environ.get("MUSIC_FP_FINGERPRINT") == "1"

    artists = _liked_artists()
    if not artists:
        artists = list(SEED_ARTISTS)

    # Expand with Last.fm similar artists
    expanded = list(artists)
    for a in artists[:3]:
        expanded.extend(_lastfm_similar_artists(a, limit=3))
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
        query = f"{artist} official"
        candidates = _yt_search_metadata(query, n=SEARCH_PER_ARTIST)
        for info in candidates:
            if added >= need:
                break
            if use_fingerprint:
                vid = info.get("id")
                url = (info.get("url") or info.get("webpage_url") or
                       (f"https://www.youtube.com/watch?v={vid}" if vid else None))
                if url and _fingerprint_and_enqueue(url, seed_artist=artist):
                    added += 1
            else:
                if _enqueue_from_metadata(info, seed_artist=artist):
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
            self._stop.wait(timeout=60)
