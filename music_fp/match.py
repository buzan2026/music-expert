"""Matcher v0 — distance euclidienne + cosine d'un candidat vs étalons top10."""

from pathlib import Path
from typing import Optional

import numpy as np


FEATURE_NAMES = [
    # Mix features
    *[f"mfcc_{i}_mean" for i in range(20)],
    *[f"mfcc_{i}_std" for i in range(20)],
    *[f"delta_mfcc_{i}_mean" for i in range(20)],
    *[f"delta_mfcc_{i}_std" for i in range(20)],
    *[f"chroma_{i}_mean" for i in range(12)],
    *[f"chroma_{i}_std" for i in range(12)],
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    *[f"spectral_contrast_{i}_mean" for i in range(7)],
    *[f"spectral_contrast_{i}_std" for i in range(7)],
    "zcr_mean", "zcr_std",
    "rms_mean", "rms_std",
    "tempo_bpm",
    *[f"tonnetz_{i}_mean" for i in range(6)],
    *[f"tonnetz_{i}_std" for i in range(6)],
    # Stem features
    "rms_rel_vocals", "rms_rel_bass", "rms_rel_drums", "rms_rel_other",
    "f0_min", "f0_max", "f0_median", "f0_std",
    "vocal_dynamics",
    "bass_harmonic_ratio",
    "bass_spectral_centroid",
    "drums_spectral_centroid",
    "drums_spectral_flatness",
    *[f"drums_mfcc_{i}" for i in range(20)],
    "drums_mfcc_std",
    # LUFS
    "lufs_norm",
]


def _normalise(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalise; return (normalised, mean, std)."""
    mu = vectors.mean(axis=0)
    sigma = vectors.std(axis=0)
    sigma[sigma < 1e-10] = 1.0
    return (vectors - mu) / sigma, mu, sigma


def match_report(
    query_vec: np.ndarray,
    query_title: str,
    top10_tracks: list[dict],
) -> dict:
    """
    Compute distances from query to each top10 etalon.

    Returns a dict with:
      - per_etalon: list of {title, artist, euclidean, cosine}
      - nearest: the closest etalon
      - top3_diverging_features: features where query deviates most from top10 mean
    """
    if not top10_tracks:
        return {"error": "No top10 tracks in database"}

    valid = [t for t in top10_tracks if t["features"] is not None]
    if not valid:
        return {"error": "top10 tracks have no features"}

    # Align all to common dimension (min across query + all refs)
    dim = min(len(query_vec), *(len(t["features"]) for t in valid))
    refs = np.stack([t["features"][:dim] for t in valid])  # (N, dim)
    q = query_vec[:dim]

    # Z-score normalise across top10 + query together
    all_vecs = np.vstack([refs, q.reshape(1, -1)])
    all_norm, mu, sigma = _normalise(all_vecs)
    refs_norm = all_norm[:-1]
    q_norm = all_norm[-1]

    per_etalon = []
    for i, track in enumerate(valid):
        r = refs_norm[i]
        euclid = float(np.linalg.norm(q_norm - r))
        norm_a = np.linalg.norm(q_norm)
        norm_b = np.linalg.norm(r)
        cosine_dist = 1.0 - float(np.dot(q_norm, r) / (norm_a * norm_b + 1e-10))
        per_etalon.append({
            "id": track["id"],
            "title": track["title"] or track["source"],
            "artist": track.get("artist") or "",
            "euclidean": round(euclid, 3),
            "cosine_dist": round(cosine_dist, 3),
        })

    per_etalon.sort(key=lambda x: x["euclidean"])
    nearest = per_etalon[0]

    # Top 3 features where |query - mean(top10)| is largest
    top10_mean = refs_norm.mean(axis=0)
    deviations = np.abs(q_norm - top10_mean)
    top3_idx = np.argsort(deviations)[::-1][:3]

    feat_names = FEATURE_NAMES[:dim] if dim <= len(FEATURE_NAMES) else [f"f{i}" for i in range(dim)]
    top3_features = []
    for idx in top3_idx:
        top3_features.append({
            "feature": feat_names[idx],
            "candidate_norm": round(float(q_norm[idx]), 3),
            "top10_mean_norm": round(float(top10_mean[idx]), 3),
            "deviation": round(float(deviations[idx]), 3),
            "candidate_raw": round(float(q[idx]), 4),
            "top10_mean_raw": round(float(refs[:, idx].mean()), 4),
        })

    return {
        "query_title": query_title,
        "per_etalon": per_etalon,
        "nearest": nearest,
        "top3_diverging_features": top3_features,
    }
