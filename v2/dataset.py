"""PyTorch dataset for the factorized possession model.

Splits are by GAME, never by row: random row splits leak same-game lineups
into validation and flatter every modeling decision.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config
from .features import FeatureStore, UNK_IDX

ACTION_TO_IDX = {a: i for i, a in enumerate(config.ACTIONS)}
SHOT_TO_IDX = {s: i for i, s in enumerate(config.SHOT_RESULTS)}


def build_context(margin, period, sec_remaining, def_fouls_before,
                  is_home_offense) -> np.ndarray:
    """Shared between training and the live engine — keep in sync with CONTEXT_DIM."""
    m = np.clip(margin / 15.0, -2.0, 2.0)
    return np.array([
        m,
        (period - 1) / 4.0,
        sec_remaining / 720.0,
        1.0 if def_fouls_before >= config.TEAM_FOULS_FOR_BONUS - 1 else 0.0,
        float(is_home_offense),
        1.0 if period > config.REGULATION_PERIODS else 0.0,
        1.0 if sec_remaining < 60 else 0.0,
        abs(m),
    ], dtype=np.float32)


def split_by_game(chances: pd.DataFrame, val_fraction=config.VAL_FRACTION, seed=7):
    games = chances["game_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_fraction))
    val_games = set(games[:n_val])
    is_val = chances["game_id"].isin(val_games)
    return chances[~is_val].reset_index(drop=True), chances[is_val].reset_index(drop=True)


class ChanceDataset(Dataset):
    def __init__(self, chances: pd.DataFrame, fs: FeatureStore, training: bool = False):
        self.fs = fs
        self.training = training
        self.n = len(chances)

        # Precompute everything into dense arrays (fast __getitem__).
        off_idx = np.zeros((self.n, 5), dtype=np.int64)
        def_idx = np.zeros((self.n, 5), dtype=np.int64)
        off_feat = np.zeros((self.n, 5, config.STAT_DIM + 2), dtype=np.float32)
        def_feat = np.zeros((self.n, 5, config.STAT_DIM + 2), dtype=np.float32)
        season_idx = np.zeros(self.n, dtype=np.int64)
        ctx = np.zeros((self.n, config.CONTEXT_DIM), dtype=np.float32)

        slot = np.full(self.n, -1, dtype=np.int64)
        action = np.zeros(self.n, dtype=np.int64)
        shot = np.full(self.n, -1, dtype=np.int64)
        oreb = np.full(self.n, -1, dtype=np.int64)
        reb_slot = np.full(self.n, -1, dtype=np.int64)
        reb_side = np.zeros(self.n, dtype=np.int64)   # 1 = offensive board

        for i, row in enumerate(chances.itertuples()):
            season = row.season
            season_idx[i] = fs.season_idx(season)
            for j, pid in enumerate(row.off_lineup):
                off_idx[i, j] = fs.player_idx(pid)
                off_feat[i, j, :config.STAT_DIM] = fs.vector(pid, season)
                off_feat[i, j, config.STAT_DIM] = row.off_stint[j] / 720.0
                off_feat[i, j, config.STAT_DIM + 1] = row.off_cum[j] / 2880.0
            for j, pid in enumerate(row.def_lineup):
                def_idx[i, j] = fs.player_idx(pid)
                def_feat[i, j, :config.STAT_DIM] = fs.vector(pid, season)
                def_feat[i, j, config.STAT_DIM] = row.def_stint[j] / 720.0
                def_feat[i, j, config.STAT_DIM + 1] = row.def_cum[j] / 2880.0

            ctx[i] = build_context(row.margin, row.period, row.sec_remaining,
                                   row.def_fouls_before, row.is_home_offense)
            slot[i] = row.ball_ender_slot
            action[i] = ACTION_TO_IDX[row.action]
            if row.shot_result is not None and row.shot_result in SHOT_TO_IDX:
                shot[i] = SHOT_TO_IDX[row.shot_result]
            if row.oreb in (0, 1) and row.shot_result == "miss":
                oreb[i] = row.oreb
                if row.rebounder > 0:
                    lineup = row.off_lineup if row.oreb == 1 else row.def_lineup
                    if row.rebounder in lineup:
                        reb_slot[i] = list(lineup).index(row.rebounder)
                        reb_side[i] = row.oreb

        self.off_idx, self.def_idx = off_idx, def_idx
        self.off_feat, self.def_feat = off_feat, def_feat
        self.season_idx, self.ctx = season_idx, ctx
        self.slot, self.action, self.shot = slot, action, shot
        self.oreb, self.reb_slot, self.reb_side = oreb, reb_slot, reb_side

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        off_idx = self.off_idx[i].copy()
        def_idx = self.def_idx[i].copy()
        if self.training and config.PLAYER_DROPOUT > 0:
            mask = np.random.random(5) < config.PLAYER_DROPOUT
            off_idx[mask] = UNK_IDX
            mask = np.random.random(5) < config.PLAYER_DROPOUT
            def_idx[mask] = UNK_IDX
        return {
            "off_idx": torch.from_numpy(off_idx),
            "def_idx": torch.from_numpy(def_idx),
            "off_feat": torch.from_numpy(self.off_feat[i]),
            "def_feat": torch.from_numpy(self.def_feat[i]),
            "season_idx": torch.tensor(self.season_idx[i]),
            "ctx": torch.from_numpy(self.ctx[i]),
            "slot": torch.tensor(self.slot[i]),
            "action": torch.tensor(self.action[i]),
            "shot": torch.tensor(self.shot[i]),
            "oreb": torch.tensor(self.oreb[i]),
            "reb_slot": torch.tensor(self.reb_slot[i]),
            "reb_side": torch.tensor(self.reb_side[i]),
        }
