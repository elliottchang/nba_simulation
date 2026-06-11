"""Empirical chance-duration model.

Pace is not a constant: it is the distribution of observed chance durations,
bucketed by action type and clock situation. The engine samples real
durations, so simulated pace inherits real pace (including faster late-clock
play) without any tuned parameter.
"""

import pickle

import numpy as np
import pandas as pd

from . import config

MAX_SAMPLES_PER_BUCKET = 8000


def _bucket_key(action: str, sec_remaining: float) -> tuple:
    clock = "late" if sec_remaining < 35 else "normal"
    return (action, clock)


class DurationModel:
    def __init__(self, buckets: dict, fallback: np.ndarray):
        self.buckets = buckets        # (action, clock) -> np.array of seconds
        self.fallback = fallback

    def sample(self, action: str, sec_remaining: float, rng=None) -> float:
        rng = rng or np.random
        arr = self.buckets.get(_bucket_key(action, sec_remaining), self.fallback)
        d = float(rng.choice(arr))
        # a chance can never take longer than the remaining clock
        return min(d, max(sec_remaining, 0.1))

    def save(self, path=config.DURATION_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=config.DURATION_FILE) -> "DurationModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_duration_model(chances: pd.DataFrame) -> DurationModel:
    rng = np.random.default_rng(11)
    buckets = {}
    for (action, clock), grp in chances.groupby(
            [chances["action"], np.where(chances["sec_remaining"] < 35, "late", "normal")]):
        arr = grp["duration"].to_numpy(dtype=np.float32)
        if len(arr) > MAX_SAMPLES_PER_BUCKET:
            arr = rng.choice(arr, MAX_SAMPLES_PER_BUCKET, replace=False)
        if len(arr):
            buckets[(action, clock)] = arr
    fallback = chances["duration"].to_numpy(dtype=np.float32)
    if len(fallback) > MAX_SAMPLES_PER_BUCKET:
        fallback = rng.choice(fallback, MAX_SAMPLES_PER_BUCKET, replace=False)
    model = DurationModel(buckets, fallback)
    model.save()
    print(f"Duration model: {len(buckets)} buckets, "
          f"median normal-clock duration "
          f"{np.median(model.fallback):.1f}s")
    return model
