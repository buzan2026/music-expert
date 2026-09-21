"""
Audio feature extraction — mix-level + stem-level fingerprint.

Feature vector layout (v2, 175 dimensions):
  Mix features (141 dims):
    [0:40]   MFCCs mean×20 + std×20
    [40:80]  ΔMFCCs mean×20 + std×20
    [80:104] Chroma mean×12 + std×12
    [104:106] Spectral centroid mean+std
    [106:108] Spectral bandwidth mean+std
    [108:110] Spectral rolloff mean+std
    [110:124] Spectral contrast mean×7 + std×7
    [124:126] Zero crossing rate mean+std
    [126:128] RMS energy mean+std
    [128]    Tempo (BPM)
    [129:141] Tonnetz mean×6 + std×6
  Stem features (34 dims):
    [141]    vocals RMS (relative to mix)
    [142]    bass   RMS (relative)
    [143]    drums  RMS (relative)
    [144]    other  RMS (relative)
    [145]    vocals F0 min (Hz, normalised /1000)
    [146]    vocals F0 max
    [147]    vocals F0 median
    [148]    vocals F0 std
    [149]    vocals RMS dynamics (std of RMS frames)
    [150]    bass   harmonic ratio
    [151]    bass   spectral centroid mean (normalised /10000)
    [152]    drums  spectral centroid mean (normalised /10000)
    [153]    drums  spectral flatness mean
    [154:175] drums MFCCs mean×20 + std×1 (timbre proxy: 21 values)
  LUFS (1 dim):
    [175]    integrated loudness (LUFS, normalised: clamp to [-40,0]/40)
Total: 176 dimensions
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .config import SAMPLE_RATE, AUDIO_DURATION

FEATURE_VERSION = 2
FEATURE_DIM = 176


def _load(path: Path, sr: int = SAMPLE_RATE, duration: Optional[float] = None) -> tuple[np.ndarray, int]:
    import librosa
    y, sr_out = librosa.load(str(path), sr=sr, mono=True, duration=duration)
    if duration is None:
        target_len = AUDIO_DURATION * sr_out
        if len(y) > target_len:
            start = (len(y) - target_len) // 2
            y = y[start: start + target_len]
    return y, sr_out


def _stat(x: np.ndarray) -> np.ndarray:
    """[mean…, std…] across last axis."""
    return np.concatenate([x.mean(axis=-1), x.std(axis=-1)])


def _mix_features(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
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

    return np.concatenate([
        _stat(mfcc),
        _stat(delta_mfcc),
        _stat(chroma),
        _stat(centroid),
        _stat(bandwidth),
        _stat(rolloff),
        _stat(contrast),
        _stat(zcr),
        _stat(rms),
        np.array([tempo_val]),
        _stat(tonnetz),
    ])  # 141 dims


def _rms_val(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2)))


def _f0_stats(y: np.ndarray, sr: int) -> np.ndarray:
    """F0 min/max/median/std in Hz using pyin on up to 30s of audio."""
    import librosa
    seg = y[:sr * 30] if len(y) > sr * 30 else y
    try:
        f0, voiced_flag, _ = librosa.pyin(
            seg, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        voiced = f0[voiced_flag > 0.5]
        if len(voiced) < 5:
            return np.zeros(4, dtype=np.float32)
        return np.array([voiced.min(), voiced.max(), np.median(voiced), voiced.std()], dtype=np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)


def _harmonic_ratio(y: np.ndarray) -> float:
    import librosa
    harmonic = librosa.effects.harmonic(y)
    h_energy = float(np.mean(harmonic ** 2))
    t_energy = float(np.mean(y ** 2)) + 1e-10
    return min(h_energy / t_energy, 1.0)


def _drums_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.concatenate([mfcc.mean(axis=-1), [mfcc.std()]])  # 21


def _lufs(y: np.ndarray, sr: int) -> float:
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        mono = y.reshape(1, -1).T  # (samples, 1)
        loudness = meter.integrated_loudness(mono)
        if np.isinf(loudness) or np.isnan(loudness):
            return 0.0
        clamped = max(-40.0, min(0.0, loudness))
        return float((clamped + 40.0) / 40.0)  # normalise to [0,1]
    except Exception:
        return 0.0


def _stem_features(stems: dict[str, Path], mix_rms: float, sr: int) -> np.ndarray:
    import librosa

    def load_stem(name: str) -> np.ndarray:
        p = stems.get(name)
        if p is None or not Path(p).exists():
            return np.zeros(sr * 10, dtype=np.float32)
        y, _ = _load(Path(p), sr=sr)
        return y

    yv = load_stem("vocals")
    yb = load_stem("bass")
    yd = load_stem("drums")
    yo = load_stem("other")

    # Relative RMS per stem (normalised by mix RMS; clamp to avoid inf)
    mix_rms = max(mix_rms, 1e-10)
    rms_rel = np.array([
        _rms_val(yv) / mix_rms,
        _rms_val(yb) / mix_rms,
        _rms_val(yd) / mix_rms,
        _rms_val(yo) / mix_rms,
    ], dtype=np.float32)

    # Vocals F0 stats (normalised /1000 Hz)
    f0 = _f0_stats(yv, sr) / 1000.0

    # Vocals RMS dynamics (std of RMS frames)
    rms_frames = librosa.feature.rms(y=yv)[0]
    vocal_dynamics = np.array([float(rms_frames.std())], dtype=np.float32)

    # Bass harmonic ratio + spectral centroid
    bass_harm = np.array([_harmonic_ratio(yb)], dtype=np.float32)
    bass_cent = librosa.feature.spectral_centroid(y=yb, sr=sr).mean() / 10000.0
    bass_cent = np.array([float(bass_cent)], dtype=np.float32)

    # Drums spectral centroid + flatness
    drums_cent = librosa.feature.spectral_centroid(y=yd, sr=sr).mean() / 10000.0
    drums_flat = librosa.feature.spectral_flatness(y=yd).mean()
    drums_feat = np.array([float(drums_cent), float(drums_flat)], dtype=np.float32)

    # Drums MFCC (21)
    drums_mfcc = _drums_mfcc(yd, sr).astype(np.float32)

    return np.concatenate([rms_rel, f0, vocal_dynamics, bass_harm, bass_cent, drums_feat, drums_mfcc])
    # 4 + 4 + 1 + 1 + 1 + 2 + 21 = 34 dims


def extract(path: Path, use_stems: bool = True) -> np.ndarray:
    """Return the 176-dim fingerprint vector (141 mix + 34 stem + 1 LUFS)."""
    y, sr = _load(path)

    mix_feat = _mix_features(y, sr)  # 141

    if use_stems:
        try:
            from .stems import separate
            stems = separate(path)
            stem_feat = _stem_features(stems, _rms_val(y), sr)  # 34
        except Exception as e:
            # Demucs unavailable or failed — fill with zeros
            stem_feat = np.zeros(34, dtype=np.float32)
    else:
        stem_feat = np.zeros(34, dtype=np.float32)

    lufs_feat = np.array([_lufs(y, sr)], dtype=np.float32)  # 1

    return np.concatenate([mix_feat, stem_feat, lufs_feat]).astype(np.float32)


def feature_dim() -> int:
    return FEATURE_DIM


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
    liked_sims: list[float] = []
    disliked_sims: list[float] = []
    neighbors: list[tuple[float, dict]] = []

    for track in reference_tracks:
        if track["features"] is None:
            continue
        ref = track["features"]
        # align dimensions if there's a mismatch (legacy tracks)
        dim = min(len(query_vec), len(ref))
        sim = cosine_similarity(query_vec[:dim], ref[:dim])
        neighbors.append((sim, track))
        if track["label"] in ("liked", "top10", "valide"):
            liked_sims.append(sim)
        elif track["label"] in ("disliked", "hors_sujet"):
            disliked_sims.append(sim)

    neighbors.sort(key=lambda x: x[0], reverse=True)
    avg_liked = float(np.mean(liked_sims)) if liked_sims else 0.0
    avg_disliked = float(np.mean(disliked_sims)) if disliked_sims else 0.0

    if liked_sims and disliked_sims:
        margin = avg_liked - avg_disliked
        confidence = abs(margin) / (avg_liked + avg_disliked + 1e-9)
        prediction = "liked" if margin > 0 else "disliked"
    elif liked_sims:
        prediction, confidence, margin = "liked", avg_liked, avg_liked
    elif disliked_sims:
        prediction, confidence, margin = "disliked", avg_disliked, -avg_disliked
    else:
        prediction, confidence, margin = "unknown", 0.0, 0.0

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "avg_sim_liked": round(avg_liked, 4),
        "avg_sim_disliked": round(avg_disliked, 4),
        "top_neighbors": neighbors[:top_k],
    }


# ── Sanity report ────────────────────────────────────────────────────────────

def sanity_report(path: Path) -> dict:
    """
    Return a human-readable dict of diagnostic features for a single track.
    Runs Demucs stem separation when the model is available; gracefully
    falls back to mix-only features (stem values will be None) otherwise.
    """
    import librosa

    y, sr = _load(path)

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    # LUFS (un-normalised)
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        mono = y.reshape(1, -1).T
        lufs_raw = meter.integrated_loudness(mono)
        lufs_str = f"{lufs_raw:.1f}" if not np.isinf(lufs_raw) else "−∞"
    except Exception as e:
        lufs_str = f"N/A ({e})"

    mix_rms = max(_rms_val(y), 1e-10)

    # Stem separation (optional — requires Demucs + model download)
    stems_available = False
    yv = yb = yd = yo = np.zeros(sr * 5, dtype=np.float32)
    try:
        from .stems import separate
        stems = separate(path)

        def load_s(name: str) -> np.ndarray:
            y_s, _ = _load(Path(stems[name]), sr=sr)
            return y_s

        yv, yb, yd, yo = load_s("vocals"), load_s("bass"), load_s("drums"), load_s("other")
        stems_available = True
    except Exception as e:
        stems_error = str(e)[:120]

    # F0 on vocals
    f0_raw = _f0_stats(yv, sr) * 1000.0  # back to Hz
    f0_min, f0_max, f0_med, f0_std = (
        round(float(f0_raw[0]), 1),
        round(float(f0_raw[1]), 1),
        round(float(f0_raw[2]), 1),
        round(float(f0_raw[3]), 1),
    )

    # Vocal RMS dynamics
    rms_frames_v = librosa.feature.rms(y=yv)[0]
    vocal_dynamics = round(float(rms_frames_v.std()), 4)

    # Bass harmonic ratio
    bass_harm = round(_harmonic_ratio(yb), 3)

    # Spectral centroid per stem (Hz)
    def cent(y_s: np.ndarray) -> float:
        c = librosa.feature.spectral_centroid(y=y_s, sr=sr)
        return round(float(c.mean()), 0)

    # Relative RMS
    def rrms(y_s: np.ndarray) -> float:
        return round(_rms_val(y_s) / mix_rms, 3)

    stem_note = "" if stems_available else "[dim](stems N/A — Demucs model not downloaded)[/dim]"

    return {
        "bpm": round(bpm, 1),
        "lufs": lufs_str,
        "stems_available": stems_available,
        "stem_note": stem_note,
        "f0_min_hz": f0_min if stems_available else None,
        "f0_max_hz": f0_max if stems_available else None,
        "f0_median_hz": f0_med if stems_available else None,
        "f0_std_hz": f0_std if stems_available else None,
        "vocal_dynamics_rms_std": vocal_dynamics if stems_available else None,
        "bass_harmonic_ratio": bass_harm if stems_available else None,
        "spectral_centroid_drums_hz": cent(yd) if stems_available else None,
        "spectral_centroid_bass_hz": cent(yb) if stems_available else None,
        "rms_rel_vocals": rrms(yv) if stems_available else None,
        "rms_rel_bass": rrms(yb) if stems_available else None,
        "rms_rel_drums": rrms(yd) if stems_available else None,
        "rms_rel_other": rrms(yo) if stems_available else None,
        # Mix-level features always available
        "spectral_centroid_mix_hz": cent(y),
        "rms_mix": round(float(_rms_val(y)), 5),
    }
