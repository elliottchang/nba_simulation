"""End-to-end smoke test on synthetic data (no network, no real data needed).

Part 1: hand-crafted event stream -> GameTransformer; asserts the attribution
        rules that were broken in the legacy pipeline (fouled player as
        ball-ender, assist/steal/block capture, oreb linking).
Part 2: synthetic chance/FT/sub tables -> feature store -> 2-epoch training ->
        duration + substitution models -> full game simulation -> calibration
        report. Proves every pipeline stage is wired correctly.

Run:  python -m v2.smoke_test
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Redirect all artifact/table paths to a temp dir BEFORE importing modules
# that bind these paths as default arguments.
_TMP = Path(tempfile.mkdtemp(prefix="nba_v2_smoke_"))
for name in ("DATA_DIR", "EVENTS_DIR", "ROTATIONS_DIR", "TABLES_DIR", "ARTIFACTS_DIR"):
    setattr(config, name, _TMP / name.lower())
config.CHANCES_FILE = _TMP / "tables/chances.parquet"
config.GAMES_FILE = _TMP / "tables/games.parquet"
config.SUB_EVENTS_FILE = _TMP / "tables/sub_decisions.parquet"
config.FT_FILE = _TMP / "tables/free_throws.parquet"
config.FEATURE_STORE_FILE = _TMP / "artifacts/feature_store.pkl"
config.MODEL_FILE = _TMP / "artifacts/possession_model_v2.pt"
config.DURATION_FILE = _TMP / "artifacts/duration_model.pkl"
config.SUB_MODEL_FILE = _TMP / "artifacts/substitution_model.pkl"

from .transform import GameTransformer  # noqa: E402

HOME, AWAY = 1610612700, 1610612799
HOME_PLAYERS = {1: "Smith", 2: "Jones", 3: "Brown", 4: "Lee", 5: "Park"}
AWAY_PLAYERS = {11: "Doe", 12: "Roe", 13: "Poe", 14: "Moe", 15: "Joe"}


def _event(num, action_type, sub_type, elapsed, team=0, pid=0, shot_value=0,
           desc="", score_home=0, score_away=0):
    return {
        "GAME_ID": "SMOKE01", "SEASON": "2023-24", "EVENTNUM": num,
        "ACTION_TYPE": action_type, "SUB_TYPE": sub_type, "PERIOD": 1,
        "ELAPSED": elapsed, "TEAM_ID": team, "PERSON_ID": pid,
        "SHOT_VALUE": shot_value, "SHOT_RESULT": "",
        "SCORE_HOME": score_home, "SCORE_AWAY": score_away,
        "DESCRIPTION": desc,
    }


def test_transform_attribution():
    events = pd.DataFrame([
        # assisted home 2pt make
        _event(1, "Made Shot", "Jump Shot", 10.0, team=HOME, pid=1, shot_value=2,
               desc="Smith 18' Jump Shot (2 PTS) (Jones 1 AST)",
               score_home=2, score_away=0),
        # shooting foul BY home player 3 ON away player 11 -> 2 FTs (1/2) -> home dreb
        _event(2, "Foul", "Shooting", 30.0, team=HOME, pid=3,
               desc="Brown S.FOUL (P1.T1) (R.Doe)"),
        _event(3, "Free Throw", "Free Throw 1 of 2", 31.0, team=AWAY, pid=11,
               desc="Doe Free Throw 1 of 2", score_home=2, score_away=1),
        _event(4, "Free Throw", "Free Throw 2 of 2", 32.0, team=AWAY, pid=11,
               desc="MISS Doe Free Throw 2 of 2"),
        _event(5, "Rebound", "", 33.0, team=HOME, pid=4, desc="Lee REBOUND"),
        # away turnover, stolen by home player 5 (adjacent blank-actionType row)
        _event(6, "Turnover", "Bad Pass", 50.0, team=AWAY, pid=12,
               desc="Roe Bad Pass Turnover (P1.T2)"),
        _event(7, "", "", 50.0, team=HOME, pid=5, desc="Park STEAL (1 STL)"),
        # home missed 3, offensive rebound by home player 1
        _event(8, "Missed Shot", "Jump Shot", 70.0, team=HOME, pid=2, shot_value=3,
               desc="MISS Jones 26' 3PT Jump Shot"),
        _event(9, "Rebound", "", 71.0, team=HOME, pid=1, desc="Smith REBOUND"),
    ])
    players = pd.DataFrame(
        [{"GAME_ID": "SMOKE01", "SEASON": "2023-24", "TEAM_ID": t,
          "PERSON_ID": p, "NAME": f"X {fam}", "NAME_I": f"{'R' if t == AWAY else 'X'}. {fam}",
          "FAMILY_NAME": fam, "STARTER": 1, "MINUTES": 24.0}
         for t, roster in ((HOME, HOME_PLAYERS), (AWAY, AWAY_PLAYERS))
         for p, fam in roster.items()])

    result = GameTransformer("SMOKE01", "2023-24", events, players).run()
    ch = pd.DataFrame(result["chances"])

    assert len(ch) == 4, f"expected 4 chances, got {len(ch)}"
    assert result["game_meta"]["home_team"] == HOME, "home team inference failed"

    make = ch.iloc[0]
    assert make["ball_ender"] == 1 and make["action"] == "2pt_attempt"
    assert make["shot_result"] == "make" and make["assist_id"] == 2
    assert make["points"] == 2

    foul = ch.iloc[1]
    assert foul["offense_team"] == AWAY, "shooting foul: offense should be the fouled team"
    assert foul["ball_ender"] == 11, "shooting foul must attribute the FOULED player, not the fouler"
    assert foul["shot_result"] == "shooting_foul" and foul["n_ft"] == 2 and foul["ft_made"] == 1
    assert foul["oreb"] == 0 and foul["rebounder"] == 4, "missed last FT should link the rebound"

    to = ch.iloc[2]
    assert to["action"] == "turnover" and to["ball_ender"] == 12 and to["steal_id"] == 5

    miss = ch.iloc[3]
    assert miss["action"] == "3pt_attempt" and miss["shot_result"] == "miss"
    assert miss["oreb"] == 1 and miss["rebounder"] == 1

    fts = pd.DataFrame(result["fts"])
    assert len(fts) == 2 and fts["made"].sum() == 1
    assert result["game_meta"]["home_score"] == 2
    assert result["game_meta"]["away_score"] == 1
    print("PART 1 OK: transform attribution correct "
          f"({len(ch)} chances, scores {result['game_meta']['home_score']}-"
          f"{result['game_meta']['away_score']})")


# ---------------------------------------------------------------- synthetic league
def synth_tables(n_games=60, seed=5):
    """Synthetic chance/FT/sub/game tables with realistic-ish distributions."""
    rng = np.random.default_rng(seed)
    teams = [1610612700 + i for i in range(4)]
    rosters = {t: [t % 1000 * 100 + j for j in range(1, 13)] for t in teams}
    season = "2023-24"

    # player archetypes: usage weight, 3pt lean, skill
    arch = {p: {"usage": rng.uniform(0.5, 2.0), "three": rng.uniform(0.1, 0.6),
                "skill": rng.uniform(-0.1, 0.1)}
            for ps in rosters.values() for p in ps}

    chances, fts, subs, games = [], [], [], []
    for gi in range(n_games):
        home, away = rng.choice(teams, 2, replace=False)
        gid = f"SYN{gi:04d}"
        scores = {home: 0, away: 0}
        elapsed = 0.0
        offense, defense = (home, away) if rng.random() < 0.5 else (away, home)
        team_fouls = {home: 0, away: 0}
        period = 1
        while period <= 4:
            dur = float(rng.uniform(4, 24))
            elapsed += dur
            if elapsed > period * 720:
                period += 1
                team_fouls = {home: 0, away: 0}
                continue
            off_lineup = sorted(rng.choice(rosters[offense][:9], 5, replace=False).tolist())
            def_lineup = sorted(rng.choice(rosters[defense][:9], 5, replace=False).tolist())
            w = np.array([arch[p]["usage"] for p in off_lineup])
            ender = int(rng.choice(off_lineup, p=w / w.sum()))
            a = arch[ender]
            r = rng.random()
            n_ft = ft_made = 0
            shot_result, oreb, rebounder = None, -1, -1
            assist_id = steal_id = block_id = -1
            points = 0
            if r < 0.13:
                action = "turnover"
                if rng.random() < 0.5:
                    steal_id = int(rng.choice(def_lineup))
            elif r < 0.16:
                action = "off_foul"
            elif r < 0.24:
                action = "drawn_foul"
                team_fouls[defense] += 1
                if team_fouls[defense] >= 5:
                    n_ft = 2
                    ft_made = int(rng.binomial(2, 0.78))
                    points = ft_made
            else:
                is3 = rng.random() < a["three"]
                action = "3pt_attempt" if is3 else "2pt_attempt"
                p_make = (0.36 if is3 else 0.54) + a["skill"]
                rr = rng.random()
                if rr < 0.06:
                    shot_result = "shooting_foul"
                    n_ft = 3 if is3 else 2
                    ft_made = int(rng.binomial(n_ft, 0.78))
                    points = ft_made
                    team_fouls[defense] += 1
                elif rng.random() < p_make:
                    if rng.random() < 0.07:
                        shot_result = "andone"
                        n_ft, ft_made = 1, int(rng.random() < 0.78)
                        team_fouls[defense] += 1
                    else:
                        shot_result = "make"
                    points = (3 if is3 else 2) + ft_made
                    if rng.random() < (0.8 if is3 else 0.5):
                        assist_id = int(rng.choice([p for p in off_lineup if p != ender]))
                else:
                    shot_result = "miss"
                    oreb = int(rng.random() < 0.27)
                    side = off_lineup if oreb else def_lineup
                    rebounder = int(rng.choice(side))
                    if rng.random() < 0.06:
                        block_id = int(rng.choice(def_lineup))
            for _ in range(n_ft):
                fts.append({"game_id": gid, "season": season, "player_id": ender,
                            "made": int(rng.random() < 0.78)})
            scores[offense] += points
            sec_rem = max(period * 720 - elapsed, 0.0)
            chances.append({
                "game_id": gid, "season": season, "period": period,
                "elapsed": elapsed, "sec_remaining": sec_rem,
                "offense_team": offense, "defense_team": defense,
                "is_home_offense": int(offense == home),
                "off_lineup": off_lineup, "def_lineup": def_lineup,
                "ball_ender": ender, "ball_ender_slot": off_lineup.index(ender),
                "action": action, "shot_result": shot_result,
                "points": points, "n_ft": n_ft, "ft_made": ft_made,
                "oreb": oreb, "rebounder": rebounder,
                "assist_id": assist_id, "steal_id": steal_id, "block_id": block_id,
                "margin": scores[offense] - scores[defense],
                "def_fouls_before": team_fouls[defense],
                "duration": dur,
                "off_stint": rng.uniform(0, 500, 5).tolist(),
                "off_cum": rng.uniform(0, 2000, 5).tolist(),
                "def_stint": rng.uniform(0, 500, 5).tolist(),
                "def_cum": rng.uniform(0, 2000, 5).tolist(),
            })
            if action != "drawn_foul" or n_ft > 0:
                offense, defense = defense, offense
        # substitution decisions
        for t in (home, away):
            for p in rosters[t]:
                for _ in range(6):
                    stint = float(rng.uniform(0, 700))
                    subs.append({
                        "game_id": gid, "season": season, "player_id": p,
                        "team_id": t, "period": int(rng.integers(1, 5)),
                        "sec_remaining": float(rng.uniform(0, 720)),
                        "stint_sec": stint, "cum_sec": float(rng.uniform(0, 2400)),
                        "margin_abs": float(rng.integers(0, 25)),
                        "fouls": int(rng.integers(0, 5)),
                        "is_starter": int(p in rosters[t][:5]),
                        "exited": int(rng.random() < min(0.05 + stint / 2000, 0.9)),
                    })
        games.append({"game_id": gid, "season": season, "home_team": home,
                      "away_team": away, "home_score": scores[home],
                      "away_score": scores[away],
                      "n_chances": 0, "n_skipped": 0})
    return (pd.DataFrame(chances), pd.DataFrame(fts),
            pd.DataFrame(subs), pd.DataFrame(games))


def test_full_pipeline():
    config.CHANCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    chances, fts, subs, games = synth_tables()
    chances.to_parquet(config.CHANCES_FILE, index=False)
    fts.to_parquet(config.FT_FILE, index=False)
    subs.to_parquet(config.SUB_EVENTS_FILE, index=False)
    games.to_parquet(config.GAMES_FILE, index=False)
    print(f"\nSynthetic league: {len(chances):,} chances, {len(games)} games")

    from .train import train
    train(epochs=2, patience=10)

    # usage sanity: high-usage archetypes should get higher fitted usage
    from .features import FeatureStore
    fs = FeatureStore.load()
    usages = [v[0] for v in fs.stats.values()]
    assert 0.05 < np.mean(usages) < 0.5, f"mean usage {np.mean(usages):.3f} not plausible"
    assert np.std(usages) > 0.005, "usage shows no variation between players"

    from .engine import GameEngineV2
    eng = GameEngineV2(games.iloc[0]["home_team"], games.iloc[0]["away_team"],
                       "2023-24", seed=1)
    res = eng.simulate_game()
    hb, ab = res["home_box"], res["away_box"]
    print(f"\nSimulated game: {res['home_score']}-{res['away_score']} "
          f"({res['periods']} periods, {len(eng.sub_log)} substitutions)")
    print(hb.head(8).to_string(index=False))

    assert res["home_score"] != res["away_score"], "game should not end tied"
    assert 40 < res["home_score"] < 220, "score out of plausible range"
    team_min = hb["MIN"].sum()
    expected_min = res["periods"] * 60.0 if res["periods"] <= 4 else 240 + (res["periods"] - 4) * 25
    assert abs(team_min - expected_min) < 2, f"team minutes {team_min:.0f} != {expected_min}"
    assert (hb["PTS"] == hb["FGM"] * 2 + hb["3PM"] + hb["FTM"]).all(), "points identity violated"
    assert hb["PF"].max() <= config.FOUL_OUT_LIMIT, "player exceeded foul limit"
    assert hb["AST"].sum() > 0 and (hb["REB"].sum() + ab["REB"].sum()) > 0

    from .calibrate import simulate_sample, real_game_stats, report
    sim, real, boxes = simulate_sample(games, n_games=4, sims_per_game=2)
    print("\nCalibration report (synthetic):")
    print(report(sim, real, boxes, real_game_stats(chances, games)).to_string(index=False))
    print("\nPART 2 OK: full pipeline runs end to end")


if __name__ == "__main__":
    test_transform_attribution()
    test_full_pipeline()
    print("\nALL SMOKE TESTS PASSED")
    sys.exit(0)
