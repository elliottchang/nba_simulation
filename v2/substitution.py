"""Learned substitution-exit policy.

Replaces the hand-tuned energy thresholds: a logistic regression fitted on
real (deadball, on-court player) -> exited-within-10s decisions. The model
imitates actual coach behavior — stint length, period, clock, score margin,
foul trouble, and starter status drive exits, with weights estimated from
data instead of guessed.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

FEATURES = ["stint_sec", "cum_sec", "period", "sec_remaining",
            "margin_abs", "fouls", "is_starter"]


class SubstitutionModel:
    def __init__(self, pipeline, base_rate: float):
        self.pipeline = pipeline
        self.base_rate = base_rate

    def exit_probability(self, stint_sec, cum_sec, period, sec_remaining,
                         margin_abs, fouls, is_starter) -> float:
        x = np.array([[stint_sec, cum_sec, period, sec_remaining,
                       margin_abs, fouls, is_starter]], dtype=np.float64)
        return float(self.pipeline.predict_proba(x)[0, 1])

    def exit_probabilities(self, rows: np.ndarray) -> np.ndarray:
        """Vectorized: rows is (n, 7) in FEATURES order."""
        return self.pipeline.predict_proba(rows)[:, 1]

    def save(self, path=config.SUB_MODEL_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=config.SUB_MODEL_FILE) -> "SubstitutionModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_substitution_model(sub_decisions: pd.DataFrame) -> SubstitutionModel:
    df = sub_decisions.dropna(subset=FEATURES + ["exited"])
    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df["exited"].to_numpy(dtype=np.int64)

    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0),
    )
    pipeline.fit(X, y)
    base_rate = float(y.mean())
    model = SubstitutionModel(pipeline, base_rate)
    model.save()

    from sklearn.metrics import log_loss
    ll = log_loss(y, pipeline.predict_proba(X)[:, 1])
    print(f"Substitution model: {len(df):,} decisions, base exit rate "
          f"{base_rate:.3f}, train log-loss {ll:.4f}")
    return model
