"""Calibration harness: simulated games vs. real held-out games.

Realism is a measurable objective. For a sample of real games, simulate the
same matchup K times and compare distributions:

    team points, total points, pace (chances/game), FG% / 3PA share / FT rate,
    OREB%, home win rate, assist & turnover counts, minutes concentration.

Run after training:
    python -m v2.calibrate --games 40 --sims 5
"""

import argparse

import numpy as np
import pandas as pd

from . import config
from .dataset import split_by_game
from .duration import DurationModel
from .engine import GameEngineV2
from .predictor import FactorizedPredictor
from .substitution import SubstitutionModel


def real_game_stats(chances: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Aggregate reference stats from the real chance table."""
    shots = chances[chances["action"].isin(["2pt_attempt", "3pt_attempt"])]
    resolved = shots[shots["shot_result"].isin(["make", "miss", "andone"])]
    return {
        "ppg_team": (games["home_score"].mean() + games["away_score"].mean()) / 2,
        "total_points": (games["home_score"] + games["away_score"]).mean(),
        "home_win_rate": (games["home_score"] > games["away_score"]).mean(),
        "chances_per_game": len(chances) / max(len(games), 1),
        "fg_pct": resolved["shot_result"].isin(["make", "andone"]).mean(),
        "share_3pa": (shots["action"] == "3pt_attempt").mean(),
        "to_rate": chances["action"].isin(["turnover", "off_foul"]).mean(),
        "oreb_pct": (chances.loc[chances["oreb"] >= 0, "oreb"]).mean(),
        "ft_per_game": chances["n_ft"].sum() / max(len(games), 1),
    }


def simulate_sample(games: pd.DataFrame, n_games: int, sims_per_game: int,
                    seed: int = 3) -> tuple:
    predictor = FactorizedPredictor()
    duration = DurationModel.load()
    subs = SubstitutionModel.load()
    rng = np.random.default_rng(seed)

    sample = games.sample(min(n_games, len(games)), random_state=seed)
    sim_rows, real_rows = [], []
    box_frames = []
    for g in sample.itertuples():
        real_rows.append({"home": g.home_score, "away": g.away_score})
        for k in range(sims_per_game):
            try:
                eng = GameEngineV2(g.home_team, g.away_team, g.season,
                                   predictor=predictor, duration_model=duration,
                                   sub_model=subs, seed=int(rng.integers(1 << 30)))
                res = eng.simulate_game()
            except ValueError as e:
                print(f"  skipping {g.game_id}: {e}")
                break
            sim_rows.append({
                "game_id": g.game_id, "sim": k,
                "home": res["home_score"], "away": res["away_score"],
                "real_home": g.home_score, "real_away": g.away_score,
            })
            box_frames.append(res["home_box"].assign(team="home"))
            box_frames.append(res["away_box"].assign(team="away"))
    return (pd.DataFrame(sim_rows), pd.DataFrame(real_rows),
            pd.concat(box_frames, ignore_index=True) if box_frames else pd.DataFrame())


def report(sim: pd.DataFrame, real: pd.DataFrame, boxes: pd.DataFrame,
           reference: dict) -> pd.DataFrame:
    n_sims = len(sim)
    sim_total = sim["home"] + sim["away"]

    rows = [
        ("team points/game", (sim["home"].mean() + sim["away"].mean()) / 2,
         reference["ppg_team"]),
        ("total points/game", sim_total.mean(), reference["total_points"]),
        ("total points (sd)", sim_total.std(), None),
        ("home win rate", (sim["home"] > sim["away"]).mean(),
         reference["home_win_rate"]),
        ("score MAE vs real", (abs(sim["home"] - sim["real_home"])
                               + abs(sim["away"] - sim["real_away"])).mean() / 2, None),
    ]
    if len(boxes):
        games_in_box = n_sims * 2
        rows += [
            ("FGA/team/game", boxes["FGA"].sum() / games_in_box, None),
            ("FG%", boxes["FGM"].sum() / max(boxes["FGA"].sum(), 1),
             reference["fg_pct"]),
            ("3PA share", boxes["3PA"].sum() / max(boxes["FGA"].sum(), 1),
             reference["share_3pa"]),
            ("FT/team/game", boxes["FTA"].sum() / games_in_box,
             reference["ft_per_game"] / 2),
            ("AST/team/game", boxes["AST"].sum() / games_in_box, None),
            ("TO/team/game", boxes["TO"].sum() / games_in_box, None),
            ("REB/team/game", boxes["REB"].sum() / games_in_box, None),
            ("max player MIN", boxes.groupby(["team"])["MIN"].max().mean(), None),
        ]
    df = pd.DataFrame(rows, columns=["metric", "simulated", "real"])
    df["simulated"] = df["simulated"].astype(float).round(3)
    df["real"] = df["real"].map(lambda v: round(v, 3) if v is not None else "")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--sims", type=int, default=5)
    args = parser.parse_args()

    chances = pd.read_parquet(config.CHANCES_FILE)
    games = pd.read_parquet(config.GAMES_FILE)

    # evaluate on validation games only (same split as training)
    _, val_chances = split_by_game(chances)
    val_games = games[games["game_id"].isin(val_chances["game_id"].unique())]
    reference = real_game_stats(val_chances, val_games)

    print(f"Simulating {args.games} held-out games x {args.sims} runs...")
    sim, real, boxes = simulate_sample(val_games, args.games, args.sims)
    if sim.empty:
        print("No games simulated (insufficient roster data?)")
        return
    print()
    print(report(sim, real, boxes, reference).to_string(index=False))


if __name__ == "__main__":
    main()
