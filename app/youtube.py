"""YouTube ID lookup via yt-dlp (no download, no ffmpeg needed)."""
import asyncio
import logging
import subprocess
from typing import Optional

log = logging.getLogger(__name__)

_YT_DLP_ARGS = [
    "--get-id", "--skip-download", "--no-playlist",
    "--quiet", "--no-warnings",
]


async def get_yt_id(artist: str, title: str) -> Optional[str]:
    query = f"{artist} - {title}"
    try:
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp", f"ytsearch1:{query}", *_YT_DLP_ARGS,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        yt_id = stdout.decode().strip().split("\n")[0].strip()
        return yt_id or None
    except asyncio.TimeoutError:
        log.warning("yt-dlp timeout: %s — %s", artist, title)
        return None
    except FileNotFoundError:
        log.error("yt-dlp not found — install via requirements.txt")
        return None
    except Exception as e:
        log.warning("yt-dlp error for %s — %s: %s", artist, title, e)
        return None


def get_yt_id_sync(artist: str, title: str) -> Optional[str]:
    """Sync version used at bootstrap."""
    query = f"{artist} - {title}"
    try:
        result = subprocess.run(
            ["yt-dlp", f"ytsearch1:{query}", *_YT_DLP_ARGS],
            capture_output=True, text=True, timeout=30,
        )
        yt_id = result.stdout.strip().split("\n")[0].strip()
        return yt_id or None
    except FileNotFoundError:
        log.error("yt-dlp not found")
        return None
    except Exception as e:
        log.warning("yt-dlp sync error for %s — %s: %s", artist, title, e)
        return None
