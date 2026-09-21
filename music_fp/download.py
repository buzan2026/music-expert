"""Resolve a source (YouTube URL or local path) to a local audio file."""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .config import AUDIO_CACHE_DIR, ensure_dirs

_YOUTUBE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+"
)


def is_youtube_url(source: str) -> bool:
    return bool(_YOUTUBE_RE.match(source))


def is_ytdlp_source(source: str) -> bool:
    """True for YouTube URLs and yt-dlp search strings (ytsearch:...)."""
    return is_youtube_url(source) or source.lower().startswith("ytsearch")


def _sanitize(name: str) -> str:
    return re.sub(r"[^\w\-. ]", "_", name)[:80]


def fetch(source: str, progress: bool = True) -> tuple[Path, Optional[str]]:
    """
    Return (local_audio_path, title).

    For a local file: validate existence and supported format.
    For a YouTube URL: download best-quality audio to the cache dir (skip if cached).
    """
    if not is_ytdlp_source(source):
        p = Path(source).resolve()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        suffix = p.suffix.lower()
        if suffix not in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm"}:
            raise ValueError(f"Unsupported audio format: {suffix}")
        return p, p.stem

    ensure_dirs()

    # Resolve search query to a stable video_id first
    video_id = _extract_video_id(source)
    cached = list(AUDIO_CACHE_DIR.glob(f"{video_id}.*"))
    if cached:
        return cached[0], None  # title unknown without re-fetching metadata

    out_template = str(AUDIO_CACHE_DIR / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-x",                          # extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0",        # best quality
        "-o", out_template,
        "--print", "title",            # print title to stdout
        "--no-warnings",
    ]
    if not progress:
        cmd.append("--quiet")
    cmd.append(source)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr.strip()}")

    title = result.stdout.strip().splitlines()[0] if result.stdout.strip() else None

    downloaded = list(AUDIO_CACHE_DIR.glob(f"{video_id}.*"))
    if not downloaded:
        raise RuntimeError("yt-dlp ran but no output file found")

    return downloaded[0], title


def _extract_video_id(url: str) -> str:
    """Extract the YouTube video id from a URL."""
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    if m:
        return m.group(1)
    # Fallback: use yt-dlp to get the id
    result = subprocess.run(
        ["yt-dlp", "--get-id", "--no-playlist", "--quiet", url],
        capture_output=True, text=True,
    )
    vid = result.stdout.strip()
    if not vid:
        raise ValueError(f"Cannot extract video id from URL: {url}")
    return vid
