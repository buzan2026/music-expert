"""Audio feature extraction — produces a fixed-size fingerprint vector."""

from pathlib import Path
from typing import Optional

import numpy as np

from .config import SAMPLE_RATE, AUDIO_DURATION

# Feature vector layout (documented here because order matters for comparison):
#   [0:40]   MFCCs mean×20 + std×20
#   [40:80]  ΔMFCCs mean×20 + std×20
#   [80:104] Chroma mean×12 + std×12
#   [104:106] Spectral centroid mean+std
#   [106:108] Spectral bandwidth mean+std
#   [108:110] Spectral rolloff mean+std
#   [110:124] Spectral contrast mean×7 + std×7
#   [124:126] Zero crossing rate mean+std
#   [126:128] RMS energy mean+std
#   [128]    Tempo (BPM)
#   [129:141] Tonnetz mean×6 + std×6
# Total: 141 dimensions


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    import librosa  # deferred: slow import

    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)

    # Keep the middle segment up to AUDIO_DURATION seconds to avoid silence at ends
    target_len = AUDIO_DURATION * sr
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start : start + target_len]

    return y, sr


def _stat(x: np.ndarray) -> np.ndarray:
    """Return [mean, std] for each row (feature × frames)."""
    return np.concatenate([x.mean(axis=-1), x.std(axis=-1)])


def extract(path: Path) -> np.ndarray:
    """Return the 141-dim fingerprint vector for an audio file."""
    import librosa

    y, sr = _load_audio(path)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta_mfcc = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(np.atleast_1d(tempo)[0])
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    features = np.concatenate([
        _stat(mfcc),               # 40
        _stat(delta_mfcc),         # 40
        _stat(chroma),             # 24
        _stat(centroid),           # 2
        _stat(bandwidth),          # 2
        _stat(rolloff),            # 2
        _stat(contrast),           # 14
        _stat(zcr),                # 2
        _stat(rms),                # 2
        np.array([tempo_val]),     # 1
        _stat(tonnetz),            # 12
    ])

    return features.astype(np.float32)


def feature_dim() -> int:
    return 141


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def predict_label(
    query_vec: np.ndarray,
    reference_tracks: list[dict],
    top_k: int = 5,
) -> dict:
    """
    Heuristic prediction without a trained classifier.

    Computes cosine similarity against all labeled reference tracks,
    then returns a weighted vote: mean-sim-to-liked vs mean-sim-to-disliked.
    """
    liked_sims: list[float] = []
    disliked_sims: list[float] = []
    neighbors: list[tuple[float, dict]] = []

    for track in reference_tracks:
        if track["features"] is None:
            continue
        sim = cosine_similarity(query_vec, track["features"])
        neighbors.append((sim, track))
        if track["label"] == "liked":
            liked_sims.append(sim)
        elif track["label"] == "disliked":
            disliked_sims.append(sim)

    neighbors.sort(key=lambda x: x[0], reverse=True)

    avg_liked = float(np.mean(liked_sims)) if liked_sims else 0.0
    avg_disliked = float(np.mean(disliked_sims)) if disliked_sims else 0.0

    if liked_sims and disliked_sims:
        margin = avg_liked - avg_disliked
        confidence = abs(margin) / (avg_liked + avg_disliked + 1e-9)
        prediction = "liked" if margin > 0 else "disliked"
    elif liked_sims:
        prediction = "liked"
        confidence = avg_liked
        margin = avg_liked
    elif disliked_sims:
        prediction = "disliked"
        confidence = avg_disliked
        margin = -avg_disliked
    else:
        prediction = "unknown"
        confidence = 0.0
        margin = 0.0

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "avg_sim_liked": round(avg_liked, 4),
        "avg_sim_disliked": round(avg_disliked, 4),
        "top_neighbors": neighbors[:top_k],
    }
