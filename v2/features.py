"""Player feature store: shrunken stat vectors + vocabularies + rosters.

Design decisions (see project discussion):
  * Usage is computed correctly: chances ended / team chances while on floor.
  * Every rate is shrunk toward the league mean with pseudo-counts
    (empirical Bayes), so low-sample players get stable features instead of
    noise.
  * The vocabulary is PLAYER-level (cross-season pooling); the season enters
    the model as a separate small embedding. (LeBron 2022-23 and 2023-24
    share one embedding.)
  * Stat vectors are the anchor of the player representation; the learned
    embedding is only a residual on top of them.

Stat vector layout (config.STAT_DIM = 15):
   0 usage          ended chances / team chances on floor
   1 share_2pt      of own ended chances
   2 share_3pt
   3 share_to       (turnover + off_foul)
   4 share_drawn    (non-shooting fouls drawn)
   5 pct_2pt        resolved 2pt FG%
   6 pct_3pt        resolved 3pt FG%
   7 pct_ft
   8 foul_draw_rate (andone + shooting_foul) / shot attempts
   9 oreb_pct       off. rebounds / off. rebound chances on floor
  10 dreb_pct
  11 ast_rate       assists / chances on floor (offense)
  12 stl_rate       steals / chances on floor (defense)
  13 blk_rate       blocks / chances on floor (defense)
  14 log_volume     log1p(chances on floor) / 10  (reliability signal)
"""

import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config

PAD_IDX, UNK_IDX = 0, 1


class FeatureStore:
    def __init__(self):
        self.player_to_idx = {}      # player_id -> embedding index (cross-season)
        self.season_to_idx = {}      # season str -> index
        self.stats = {}              # (player_id, season) -> raw stat vector (np.float32)
        self.league_mean = None      # raw league-average stat vector
        self.norm_mean = None        # for z-scoring
        self.norm_std = None
        self.rosters = None          # DataFrame: season, team_id, player_id, games, starter_rate, floor_share
        self.assisted_rate = {"2pt_attempt": 0.5, "3pt_attempt": 0.8}
        self.steal_share_of_to = 0.5
        self.block_share_of_miss = {"2pt_attempt": 0.1, "3pt_attempt": 0.01}
        self.names = {}              # player_id -> display name (best effort)

    # ------------------------------------------------------------ access
    def player_idx(self, player_id) -> int:
        return self.player_to_idx.get(int(player_id), UNK_IDX)

    def season_idx(self, season) -> int:
        return self.season_to_idx.get(season, 0)

    def vector(self, player_id, season) -> np.ndarray:
        """Z-scored stat vector; league mean for unknown players."""
        raw = self.stats.get((int(player_id), season), self.league_mean)
        return ((raw - self.norm_mean) / self.norm_std).astype(np.float32)

    def raw(self, player_id, season) -> np.ndarray:
        return self.stats.get((int(player_id), season), self.league_mean)

    def ft_pct(self, player_id, season) -> float:
        return float(self.raw(player_id, season)[7])

    def name(self, player_id) -> str:
        return self.names.get(int(player_id), f"Player {player_id}")

    @property
    def num_players(self) -> int:
        return len(self.player_to_idx) + 2  # + PAD/UNK

    @property
    def num_seasons(self) -> int:
        return max(len(self.season_to_idx), 1)

    def save(self, path=config.FEATURE_STORE_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=config.FEATURE_STORE_FILE) -> "FeatureStore":
        with open(path, "rb") as f:
            return pickle.load(f)


def _shrink(successes, attempts, prior_rate, pseudo):
    return (successes + prior_rate * pseudo) / (attempts + pseudo)


def build_feature_store(chances: pd.DataFrame, fts: pd.DataFrame,
                        subs: pd.DataFrame = None) -> FeatureStore:
    fs = FeatureStore()

    seasons = sorted(chances["season"].unique())
    fs.season_to_idx = {s: i for i, s in enumerate(seasons)}

    # ------------------------------------------------------------ accumulate counts
    c = defaultdict(lambda: defaultdict(float))   # (pid, season) -> counter dict

    def bump(pid, season, key, amt=1.0):
        if pid is not None and pid > 0:
            c[(int(pid), season)][key] += amt

    is_shot = chances["action"].isin(["2pt_attempt", "3pt_attempt"])
    for row in chances.itertuples():
        season = row.season
        for pid in row.off_lineup:
            bump(pid, season, "on_floor_off")
        for pid in row.def_lineup:
            bump(pid, season, "on_floor_def")

        pid = row.ball_ender
        if pid > 0:
            bump(pid, season, "ended")
            a = row.action
            if a == "2pt_attempt":
                bump(pid, season, "att2" if row.shot_result != "shooting_foul" else "sf2")
                if row.shot_result in ("make", "andone"):
                    bump(pid, season, "make2")
                if row.shot_result in ("andone", "shooting_foul"):
                    bump(pid, season, "fdraw")
            elif a == "3pt_attempt":
                bump(pid, season, "att3" if row.shot_result != "shooting_foul" else "sf3")
                if row.shot_result in ("make", "andone"):
                    bump(pid, season, "make3")
                if row.shot_result in ("andone", "shooting_foul"):
                    bump(pid, season, "fdraw")
            elif a in ("turnover", "off_foul"):
                bump(pid, season, "to")
            elif a == "drawn_foul":
                bump(pid, season, "drawn")

        if row.oreb != -1 and row.rebounder > 0:
            side = "oreb" if row.oreb == 1 else "dreb"
            bump(row.rebounder, season, side)
        if row.oreb != -1:
            for pid in row.off_lineup:
                bump(pid, season, "oreb_chance")
            for pid in row.def_lineup:
                bump(pid, season, "dreb_chance")
        if row.assist_id > 0:
            bump(row.assist_id, season, "ast")
        if row.steal_id > 0:
            bump(row.steal_id, season, "stl")
        if row.block_id > 0:
            bump(row.block_id, season, "blk")

    ft_counts = defaultdict(lambda: [0.0, 0.0])
    if len(fts):
        for row in fts.itertuples():
            key = (int(row.player_id), row.season)
            ft_counts[key][0] += row.made
            ft_counts[key][1] += 1

    # ------------------------------------------------------------ league priors
    tot = defaultdict(float)
    for d in c.values():
        for k, v in d.items():
            tot[k] += v
    ended = max(tot["ended"], 1.0)
    on_floor = max(tot["on_floor_off"], 1.0)
    league = {
        "usage": ended / on_floor,
        "share_2pt": (tot["att2"] + tot["sf2"]) / ended,
        "share_3pt": (tot["att3"] + tot["sf3"]) / ended,
        "share_to": tot["to"] / ended,
        "share_drawn": tot["drawn"] / ended,
        "pct_2pt": tot["make2"] / max(tot["att2"], 1.0),
        "pct_3pt": tot["make3"] / max(tot["att3"], 1.0),
        "pct_ft": (sum(v[0] for v in ft_counts.values())
                   / max(sum(v[1] for v in ft_counts.values()), 1.0)) if ft_counts else 0.78,
        "fdraw": tot["fdraw"] / max(tot["att2"] + tot["att3"] + tot["sf2"] + tot["sf3"], 1.0),
        "oreb": tot["oreb"] / max(tot["oreb_chance"], 1.0),
        "dreb": tot["dreb"] / max(tot["dreb_chance"], 1.0),
        "ast": tot["ast"] / on_floor,
        "stl": tot["stl"] / max(tot["on_floor_def"], 1.0),
        "blk": tot["blk"] / max(tot["on_floor_def"], 1.0),
    }

    # league-level helper rates for the engine
    makes = chances[chances["shot_result"].isin(["make", "andone"])]
    for act in ("2pt_attempt", "3pt_attempt"):
        sub = makes[makes["action"] == act]
        if len(sub):
            fs.assisted_rate[act] = float((sub["assist_id"] > 0).mean())
    tos = chances[chances["action"] == "turnover"]
    if len(tos):
        fs.steal_share_of_to = float((tos["steal_id"] > 0).mean())
    misses = chances[chances["shot_result"] == "miss"]
    for act in ("2pt_attempt", "3pt_attempt"):
        sub = misses[misses["action"] == act]
        if len(sub):
            fs.block_share_of_miss[act] = float((sub["block_id"] > 0).mean())

    # ------------------------------------------------------------ per-player vectors
    sp, sa = config.SHRINK_POSSESSIONS, config.SHRINK_ATTEMPTS
    for key, d in c.items():
        floor = d["on_floor_off"]
        floor_def = d["on_floor_def"]
        end = d["ended"]
        att2, att3 = d["att2"], d["att3"]
        shots = att2 + att3 + d["sf2"] + d["sf3"]
        ftm, fta = ft_counts.get(key, [0.0, 0.0])
        vec = np.array([
            _shrink(end, floor, league["usage"], sp),
            _shrink(att2 + d["sf2"], end, league["share_2pt"], sa),
            _shrink(att3 + d["sf3"], end, league["share_3pt"], sa),
            _shrink(d["to"], end, league["share_to"], sa),
            _shrink(d["drawn"], end, league["share_drawn"], sa),
            _shrink(d["make2"], att2, league["pct_2pt"], sa),
            _shrink(d["make3"], att3, league["pct_3pt"], sa),
            _shrink(ftm, fta, league["pct_ft"], sa),
            _shrink(d["fdraw"], shots, league["fdraw"], sa),
            _shrink(d["oreb"], d["oreb_chance"], league["oreb"], sp),
            _shrink(d["dreb"], d["dreb_chance"], league["dreb"], sp),
            _shrink(d["ast"], floor, league["ast"], sp),
            _shrink(d["stl"], floor_def, league["stl"], sp),
            _shrink(d["blk"], floor_def, league["blk"], sp),
            np.log1p(floor) / 10.0,
        ], dtype=np.float32)
        fs.stats[key] = vec

    fs.league_mean = np.array([
        league["usage"], league["share_2pt"], league["share_3pt"],
        league["share_to"], league["share_drawn"], league["pct_2pt"],
        league["pct_3pt"], league["pct_ft"], league["fdraw"],
        league["oreb"], league["dreb"], league["ast"], league["stl"],
        league["blk"], np.log1p(200.0) / 10.0,
    ], dtype=np.float32)

    mat = np.stack(list(fs.stats.values())) if fs.stats else fs.league_mean.reshape(1, -1)
    fs.norm_mean = mat.mean(axis=0)
    fs.norm_std = np.maximum(mat.std(axis=0), 1e-4)

    # ------------------------------------------------------------ vocab (player level)
    players = sorted({pid for pid, _ in fs.stats.keys()})
    fs.player_to_idx = {pid: i + 2 for i, pid in enumerate(players)}

    # ------------------------------------------------------------ rosters
    if subs is not None and len(subs):
        agg = (subs.groupby(["season", "team_id", "player_id"])
               .agg(games=("game_id", "nunique"), starter_rate=("is_starter", "mean"))
               .reset_index())
        floor_share = []
        for r in agg.itertuples():
            d = c.get((int(r.player_id), r.season), {})
            floor_share.append(d.get("on_floor_off", 0.0))
        agg["floor_share"] = floor_share
        total = agg.groupby(["season", "team_id"])["floor_share"].transform("sum")
        agg["floor_share"] = agg["floor_share"] / total.clip(lower=1.0)
        fs.rosters = agg

    print(f"Feature store: {len(fs.stats)} player-seasons, "
          f"{len(fs.player_to_idx)} unique players, {len(seasons)} seasons")
    return fs


def resolve_names(fs: FeatureStore, chances: pd.DataFrame = None):
    """Best-effort player names from the static nba_api index (offline file)."""
    try:
        from nba_api.stats.static import players as static_players
        for p in static_players.get_players():
            if p["id"] in fs.player_to_idx:
                fs.names[p["id"]] = p["full_name"]
    except Exception:
        pass


def main():
    chances = pd.read_parquet(config.CHANCES_FILE)
    fts = pd.read_parquet(config.FT_FILE)
    subs = pd.read_parquet(config.SUB_EVENTS_FILE)
    fs = build_feature_store(chances, fts, subs)
    resolve_names(fs)
    fs.save()
    print(f"Saved {config.FEATURE_STORE_FILE}")


if __name__ == "__main__":
    main()
