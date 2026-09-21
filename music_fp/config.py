"""Paths and runtime configuration."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("MUSIC_FP_DATA", Path.home() / ".music-fp"))
DB_PATH = DATA_DIR / "tracks.db"
AUDIO_CACHE_DIR = DATA_DIR / "cache"

SAMPLE_RATE = 22050
AUDIO_DURATION = 90  # seconds analysed (middle 90s of track)

LABEL_LIKED = "liked"
LABEL_DISLIKED = "disliked"
LABEL_UNTAGGED = "untagged"
VALID_LABELS = {LABEL_LIKED, LABEL_DISLIKED, LABEL_UNTAGGED}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
