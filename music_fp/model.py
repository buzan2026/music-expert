"""
Preference model — ranks candidates as liked/disliked.

Two modes depending on labeled data count:
  < MIN_LABELED : cosine distance heuristic (no training needed)
  ≥ MIN_LABELED : logistic regression trained on all labeled tracks

Auto-retrain triggered after RETRAIN_EVERY new votes.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .config import DATA_DIR

MODEL_PATH = DATA_DIR / "model.pkl"
MIN_LABELED = 20
RETRAIN_EVERY = 5


class PreferenceModel:
    def __init__(self) -> None:
        self._clf = None         # sklearn LogisticRegression
        self._scaler = None      # sklearn StandardScaler
        self._version: int = 0
        self._n_labeled: int = 0
        self._load()

    # ── Public API ────────────────────────────────────────────────────────

    def score(self, features: np.ndarray, liked_tracks: Optional[list] = None) -> float:
        """
        Return a score in [0, 1] where 1 = very likely liked.

        Falls back to cosine-distance heuristic when no trained model yet.
        """
        if self._clf is not None:
            return self._clf_score(features)
        if liked_tracks:
            return self._heuristic_score(features, liked_tracks)
        return 0.5

    def maybe_retrain(self) -> bool:
        """Retrain if enough new votes have accumulated. Returns True if retrained."""
        from .store import get_labeled_for_training, get_vote_counts
        X, y = get_labeled_for_training()
        n = len(y)
        if n < MIN_LABELED:
            return False
        if n == self._n_labeled:
            return False

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ])
        pipe.fit(X, y)
        self._clf = pipe
        self._n_labeled = n
        self._version += 1
        self._save()
        return True

    @property
    def is_trained(self) -> bool:
        return self._clf is not None

    @property
    def version(self) -> int:
        return self._version

    @property
    def n_labeled(self) -> int:
        return self._n_labeled

    # ── Private ───────────────────────────────────────────────────────────

    def _clf_score(self, features: np.ndarray) -> float:
        try:
            proba = self._clf.predict_proba(features.reshape(1, -1))[0]
            # class order: 0=disliked, 1=liked
            classes = list(self._clf.classes_)
            liked_idx = classes.index(1) if 1 in classes else -1
            return float(proba[liked_idx]) if liked_idx >= 0 else 0.5
        except Exception:
            return 0.5

    @staticmethod
    def _heuristic_score(features: np.ndarray, liked_tracks: list) -> float:
        from .fingerprint import cosine_similarity
        sims = []
        for t in liked_tracks:
            if t.get("features") is None:
                continue
            ref = t["features"]
            dim = min(len(features), len(ref))
            sims.append(cosine_similarity(features[:dim], ref[:dim]))
        return float(np.mean(sims)) if sims else 0.5

    def _save(self) -> None:
        from .config import ensure_dirs
        ensure_dirs()
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "clf": self._clf,
                "version": self._version,
                "n_labeled": self._n_labeled,
            }, f)

    def _load(self) -> None:
        if not MODEL_PATH.exists():
            return
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self._clf = data.get("clf")
            self._version = data.get("version", 0)
            self._n_labeled = data.get("n_labeled", 0)
        except Exception:
            pass


# Module-level singleton
_model: Optional[PreferenceModel] = None


def get_model() -> PreferenceModel:
    global _model
    if _model is None:
        _model = PreferenceModel()
    return _model
