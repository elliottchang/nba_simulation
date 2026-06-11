"""Sampling interface over the trained factorized model.

The engine asks stage-by-stage questions (who ends it? what do they do? does
it go in? who rebounds?) and this class answers them by sampling the fitted
heads. All tensor assembly and caching lives here so the engine stays
readable.
"""

import numpy as np
import torch

from . import config
from .dataset import build_context
from .features import FeatureStore
from .model import PossessionModelV2

ACTIONS = config.ACTIONS
SHOT_RESULTS = config.SHOT_RESULTS


class FactorizedPredictor:
    def __init__(self, fs: FeatureStore = None, model: PossessionModelV2 = None,
                 device: str = "cpu"):
        self.device = device
        self.fs = fs or FeatureStore.load()
        if model is None:
            ckpt = torch.load(config.MODEL_FILE, map_location=device, weights_only=False)
            model = PossessionModelV2(ckpt["num_players"], ckpt["num_seasons"])
            model.load_state_dict(ckpt["model_state_dict"])
        self.model = model.to(device).eval()
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------ tensors
    def _lineup_tensors(self, lineup, season, stints, cums):
        idx = np.array([self.fs.player_idx(p) for p in lineup], dtype=np.int64)
        feat = np.zeros((5, config.STAT_DIM + 2), dtype=np.float32)
        for j, pid in enumerate(lineup):
            feat[j, :config.STAT_DIM] = self.fs.vector(pid, season)
            feat[j, config.STAT_DIM] = stints[j] / 720.0
            feat[j, config.STAT_DIM + 1] = cums[j] / 2880.0
        return (torch.from_numpy(idx).unsqueeze(0),
                torch.from_numpy(feat).unsqueeze(0))

    def encode_state(self, off_lineup, def_lineup, season, ctx_kwargs,
                     off_stints, off_cums, def_stints, def_cums):
        """Encode the current floor state once; reuse across stage samples."""
        off_idx, off_feat = self._lineup_tensors(off_lineup, season, off_stints, off_cums)
        def_idx, def_feat = self._lineup_tensors(def_lineup, season, def_stints, def_cums)
        season_idx = torch.tensor([self.fs.season_idx(season)])
        ctx = torch.from_numpy(build_context(**ctx_kwargs)).unsqueeze(0)
        with torch.no_grad():
            off = self.model.encoder(off_idx, off_feat, season_idx)
            deff = self.model.encoder(def_idx, def_feat, season_idx)
        return {"off": off, "deff": deff, "ctx": ctx}

    @staticmethod
    def _sample(probs: np.ndarray, rng) -> int:
        return int(rng.choice(len(probs), p=probs / probs.sum()))

    # ------------------------------------------------------------ stages
    def sample_ball_ender(self, state) -> int:
        """Returns offensive slot index 0-4."""
        with torch.no_grad():
            logits = self.model.ender_logits(state["off"], state["deff"], state["ctx"])
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        return self._sample(probs, self.rng)

    def sample_action(self, state, slot: int) -> str:
        slot_t = torch.tensor([slot])
        with torch.no_grad():
            logits = self.model.action_logits(state["off"], state["deff"],
                                              state["ctx"], slot_t)
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        return ACTIONS[self._sample(probs, self.rng)]

    def sample_shot_result(self, state, slot: int, is_three: bool) -> str:
        slot_t = torch.tensor([slot])
        flag = torch.tensor([1.0 if is_three else 0.0])
        with torch.no_grad():
            logits = self.model.shot_logits(state["off"], state["deff"],
                                            state["ctx"], slot_t, flag)
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        return SHOT_RESULTS[self._sample(probs, self.rng)]

    def sample_oreb(self, state, is_three: bool) -> bool:
        flag = torch.tensor([1.0 if is_three else 0.0])
        with torch.no_grad():
            logit = self.model.oreb_logit(state["off"], state["deff"],
                                          state["ctx"], flag)
        return bool(self.rng.random() < torch.sigmoid(logit).item())

    def sample_rebounder(self, state, offensive: bool) -> int:
        """Returns slot index on the rebounding side."""
        side = state["off"] if offensive else state["deff"]
        flag = torch.tensor([1 if offensive else 0])
        with torch.no_grad():
            logits = self.model.rebounder_logits(side, flag)
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        return self._sample(probs, self.rng)

    # ------------------------------------------------------------ stat helpers
    def ft_make(self, player_id, season) -> bool:
        return bool(self.rng.random() < self.fs.ft_pct(player_id, season))

    def sample_assister(self, off_lineup, season, shooter, action) -> int | None:
        """None = unassisted. Weighted by teammates' real assist rates."""
        if self.rng.random() > self.fs.assisted_rate.get(action, 0.6):
            return None
        mates = [p for p in off_lineup if p != shooter]
        weights = np.array([max(self.fs.raw(p, season)[11], 1e-4) for p in mates])
        return int(self.rng.choice(mates, p=weights / weights.sum()))

    def sample_stealer(self, def_lineup, season) -> int | None:
        if self.rng.random() > self.fs.steal_share_of_to:
            return None
        weights = np.array([max(self.fs.raw(p, season)[12], 1e-4) for p in def_lineup])
        return int(self.rng.choice(def_lineup, p=weights / weights.sum()))

    def sample_blocker(self, def_lineup, season, action) -> int | None:
        if self.rng.random() > self.fs.block_share_of_miss.get(action, 0.05):
            return None
        weights = np.array([max(self.fs.raw(p, season)[13], 1e-4) for p in def_lineup])
        return int(self.rng.choice(def_lineup, p=weights / weights.sum()))
