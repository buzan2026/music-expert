"""Demucs-based stem separation: vocals, bass, drums, other."""

import subprocess
import shutil
from pathlib import Path
from typing import Optional

from .config import AUDIO_CACHE_DIR, ensure_dirs

STEM_NAMES = ["vocals", "bass", "drums", "other"]


def separate(audio_path: Path, force: bool = False) -> dict[str, Path]:
    """
    Run htdemucs on audio_path and return {stem_name: stem_path}.

    Results are cached next to the source file; pass force=True to re-run.
    """
    ensure_dirs()

    stem_dir = AUDIO_CACHE_DIR / "stems" / audio_path.stem
    stem_paths = {s: stem_dir / f"{s}.mp3" for s in STEM_NAMES}

    if not force and all(p.exists() for p in stem_paths.values()):
        return stem_paths

    stem_dir.mkdir(parents=True, exist_ok=True)

    # demucs writes: <out_dir>/htdemucs/<track_stem>/{vocals,bass,drums,other}.mp3
    out_base = AUDIO_CACHE_DIR / "stems_raw"
    out_base.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", "-m", "demucs",
        "--name", "htdemucs",
        "--mp3",
        "--out", str(out_base),
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"demucs failed:\n{result.stderr[-2000:]}")

    raw_dir = out_base / "htdemucs" / audio_path.stem
    if not raw_dir.exists():
        # demucs strips the extension from the track name
        name_no_ext = audio_path.stem
        candidates = list((out_base / "htdemucs").glob(f"{name_no_ext}*"))
        if not candidates:
            raise RuntimeError(f"demucs output dir not found under {out_base}/htdemucs")
        raw_dir = candidates[0]

    for stem in STEM_NAMES:
        src = raw_dir / f"{stem}.mp3"
        if not src.exists():
            raise RuntimeError(f"Missing demucs output stem: {src}")
        shutil.copy2(src, stem_paths[stem])

    return stem_paths
