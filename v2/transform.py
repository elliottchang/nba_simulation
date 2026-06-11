"""Transform raw events + rotations into model-ready training tables.

Outputs (all under data_v2/tables/):
  chances.parquet       one row per scoring chance, with correct attribution:
                          - ball-ender chosen from the event that defines the
                            chance (shooter / turnover committer / FOULED
                            player on defensive fouls -- PLAYER2, never the
                            defender)
                          - offense-perspective score margin from accumulated
                            scores (not the home-perspective SCOREMARGIN string)
                          - lineups from rotation intervals (no drift)
                          - linked rebound outcome for misses
                          - ball-ender stint/cumulative minutes (fatigue signal)
  free_throws.parquet   one row per FT attempt (shooter, made)
  sub_decisions.parquet (deadball moment x on-court player) -> exited or not
  games.parquet         per-game metadata + final scores for calibration

Re-runnable without network access whenever labeling logic changes.
"""

import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .config import (
    EVT_MADE_SHOT, EVT_MISSED_SHOT, EVT_FREE_THROW, EVT_REBOUND,
    EVT_TURNOVER, EVT_FOUL, EVT_TIMEOUT, EVT_PERIOD_END,
)

# Foul EVENTMSGACTIONTYPEs (NBA pbp)
FOUL_SHOOTING = {2, 9, 29}          # shooting, shooting-block, shooting away-from-play
FOUL_OFFENSIVE = {4, 26}            # offensive, charge
FOUL_TECHNICAL = {11, 12, 13, 18, 19, 25, 30}

MADE_SHOT_DEADBALL_SAMPLE = 0.3     # sub-decision sampling rate at made-shot deadballs
TEAM_ID_FLOOR = 1_600_000_000       # PLAYER*_ID values >= this are team ids


def is_player_id(v) -> bool:
    try:
        v = int(v)
    except (TypeError, ValueError):
        return False
    return 0 < v < TEAM_ID_FLOOR


def is_team_id(v) -> bool:
    try:
        v = int(v)
    except (TypeError, ValueError):
        return False
    return v >= TEAM_ID_FLOOR


def event_desc(row) -> str:
    parts = [row.get("HOMEDESCRIPTION"), row.get("NEUTRALDESCRIPTION"), row.get("VISITORDESCRIPTION")]
    return " ".join(str(p) for p in parts if pd.notna(p)).upper()


def is_3pt(row) -> bool:
    return "3PT" in event_desc(row)


class RotationIndex:
    """Fast on-floor lookups from GameRotation intervals for one game."""

    def __init__(self, rot_df: pd.DataFrame):
        self.intervals = defaultdict(list)   # (team_id, player_id) -> [(in, out)]
        self.team_players = defaultdict(set)
        for r in rot_df.itertuples():
            key = (int(r.TEAM_ID), int(r.PERSON_ID))
            self.intervals[key].append((float(r.IN_SEC), float(r.OUT_SEC)))
            self.team_players[int(r.TEAM_ID)].add(int(r.PERSON_ID))
        self.team_ids = sorted(self.team_players.keys())

    def on_floor(self, team_id: int, t: float) -> list:
        out = []
        for pid in self.team_players[team_id]:
            for lo, hi in self.intervals[(team_id, pid)]:
                if lo <= t < hi:
                    out.append(pid)
                    break
        return sorted(out)

    def lineup_at(self, team_id: int, t: float, must_include: int = None) -> list:
        """5-man lineup at time t; nudges t if it lands on a sub boundary."""
        for probe in (t, t - 0.5, t + 0.5):
            lineup = self.on_floor(team_id, probe)
            if len(lineup) == 5 and (must_include is None or must_include in lineup):
                return lineup
        lineup = self.on_floor(team_id, t)
        return lineup if len(lineup) == 5 else None

    def stint_features(self, team_id: int, player_id: int, t: float):
        """(seconds into current stint, cumulative seconds played) at time t."""
        stint, cum = 0.0, 0.0
        for lo, hi in self.intervals[(team_id, player_id)]:
            if lo <= t:
                cum += min(hi, t) - lo
                if t < hi:
                    stint = t - lo
        return stint, cum

    def starters(self, team_id: int) -> set:
        return {pid for pid in self.team_players[team_id]
                if any(lo == 0.0 for lo, _ in self.intervals[(team_id, pid)])}


def infer_home_team(events: pd.DataFrame, team_ids: list) -> int:
    """Home team = team whose shots appear in HOMEDESCRIPTION."""
    shots = events[events["EVENTMSGTYPE"].isin([EVT_MADE_SHOT, EVT_MISSED_SHOT])]
    votes = defaultdict(int)
    for r in shots.itertuples():
        if pd.notna(r.HOMEDESCRIPTION) and is_team_id(r.PLAYER1_TEAM_ID):
            votes[int(r.PLAYER1_TEAM_ID)] += 1
        elif pd.notna(r.VISITORDESCRIPTION) and is_team_id(r.PLAYER1_TEAM_ID):
            votes[int(r.PLAYER1_TEAM_ID)] -= 1
    if not votes:
        return team_ids[0]
    return max(votes, key=votes.get)


class GameTransformer:
    """Walks one game's events and emits training rows."""

    def __init__(self, game_id, season, events, rotation):
        self.game_id = game_id
        self.season = season
        self.events = events.sort_values("EVENTNUM").reset_index(drop=True)
        self.rot = RotationIndex(rotation)
        if len(self.rot.team_ids) != 2:
            raise ValueError("expected exactly 2 teams in rotation data")
        self.home_team = infer_home_team(self.events, self.rot.team_ids)
        self.away_team = [t for t in self.rot.team_ids if t != self.home_team][0]

        self.scores = {self.home_team: 0, self.away_team: 0}
        self.team_fouls_period = defaultdict(int)   # (team, period) -> fouls
        self.player_fouls = defaultdict(int)
        self.chance_start = 0.0                      # elapsed secs when current chance began

        self.chances, self.fts, self.sub_rows = [], [], []
        self.skipped = 0

    # ------------------------------------------------------------ utilities
    def other(self, team_id: int) -> int:
        return self.away_team if team_id == self.home_team else self.home_team

    def margin_for(self, offense_team: int) -> int:
        return self.scores[offense_team] - self.scores[self.other(offense_team)]

    def period_seconds_remaining(self, period: int, elapsed: float) -> float:
        length = config.PERIOD_SECONDS if period <= config.REGULATION_PERIODS else config.OT_SECONDS
        from .collect import period_start_elapsed
        return max(0.0, period_start_elapsed(period) + length - elapsed)

    def _ft_run(self, start_idx: int, shooter: int):
        """Consecutive FT events by `shooter` starting at start_idx. Returns (n, made, last_missed, end_idx)."""
        n = made = 0
        last_missed = False
        i = start_idx
        while i < len(self.events):
            row = self.events.iloc[i]
            if row["EVENTMSGTYPE"] != EVT_FREE_THROW or not is_player_id(row["PLAYER1_ID"]) \
                    or int(row["PLAYER1_ID"]) != shooter:
                # allow interleaved subs/timeouts inside a FT sequence
                if row["EVENTMSGTYPE"] in (EVT_FREE_THROW, EVT_MADE_SHOT, EVT_MISSED_SHOT,
                                           EVT_TURNOVER, EVT_REBOUND, EVT_PERIOD_END):
                    break
                i += 1
                continue
            n += 1
            missed = "MISS" in event_desc(row)
            last_missed = missed
            if not missed:
                made += 1
                self.scores[int(row["PLAYER1_TEAM_ID"])] += 1
            self.fts.append({
                "game_id": self.game_id, "season": self.season,
                "player_id": shooter, "made": int(not missed),
            })
            i += 1
        return n, made, last_missed, i

    def _find_rebound(self, start_idx: int):
        """Next rebound event at/after start_idx (skipping non-action rows)."""
        for i in range(start_idx, min(start_idx + 6, len(self.events))):
            row = self.events.iloc[i]
            if row["EVENTMSGTYPE"] == EVT_REBOUND:
                team = row["PLAYER1_TEAM_ID"]
                pid = row["PLAYER1_ID"]
                if is_player_id(pid) and is_team_id(team):
                    return int(team), int(pid)
                # team rebound: PLAYER1_ID holds the team id
                if is_team_id(pid):
                    return int(pid), None
                if is_team_id(team):
                    return int(team), None
            if row["EVENTMSGTYPE"] in (EVT_MADE_SHOT, EVT_MISSED_SHOT, EVT_TURNOVER,
                                       EVT_FOUL, EVT_PERIOD_END):
                break
        return None, None

    # ------------------------------------------------------------ row emission
    def emit_chance(self, *, elapsed, period, offense_team, ball_ender, action,
                    shot_result=None, points=0, n_ft=0, ft_made=0,
                    oreb=None, rebounder=None, margin_before=None,
                    def_fouls_before=0, assist_id=None, steal_id=None,
                    block_id=None):
        defense_team = self.other(offense_team)
        off_lineup = self.rot.lineup_at(offense_team, elapsed, must_include=ball_ender)
        def_lineup = self.rot.lineup_at(defense_team, elapsed)
        if off_lineup is None or def_lineup is None:
            self.skipped += 1
            self.chance_start = elapsed
            return
        slot = off_lineup.index(ball_ender) if ball_ender in off_lineup else -1
        off_stints = [self.rot.stint_features(offense_team, p, elapsed) for p in off_lineup]
        def_stints = [self.rot.stint_features(defense_team, p, elapsed) for p in def_lineup]
        duration = float(np.clip(elapsed - self.chance_start, 0.5, 35.0))
        self.chance_start = elapsed

        self.chances.append({
            "game_id": self.game_id, "season": self.season, "period": int(period),
            "elapsed": float(elapsed),
            "sec_remaining": self.period_seconds_remaining(period, elapsed),
            "offense_team": offense_team, "defense_team": defense_team,
            "is_home_offense": int(offense_team == self.home_team),
            "off_lineup": off_lineup, "def_lineup": def_lineup,
            "ball_ender": ball_ender if ball_ender is not None else -1,
            "ball_ender_slot": slot,
            "action": action, "shot_result": shot_result,
            "points": int(points), "n_ft": int(n_ft), "ft_made": int(ft_made),
            "oreb": oreb if oreb is not None else -1,        # 1/0, -1 = n/a
            "rebounder": rebounder if rebounder is not None else -1,
            "assist_id": assist_id if assist_id is not None else -1,
            "steal_id": steal_id if steal_id is not None else -1,
            "block_id": block_id if block_id is not None else -1,
            "margin": int(margin_before),
            "def_fouls_before": int(def_fouls_before),
            "duration": duration,
            "off_stint": [s for s, _ in off_stints],
            "off_cum": [c for _, c in off_stints],
            "def_stint": [s for s, _ in def_stints],
            "def_cum": [c for _, c in def_stints],
        })

    def emit_sub_decisions(self, elapsed, period, force: bool = False):
        """At a deadball, record (player, exited-within-10s?) for everyone on floor."""
        if not force and random.random() > MADE_SHOT_DEADBALL_SAMPLE:
            return
        margin = abs(self.scores[self.home_team] - self.scores[self.away_team])
        sec_rem = self.period_seconds_remaining(period, elapsed)
        for team in self.rot.team_ids:
            starters = self.rot.starters(team)
            for pid in self.rot.on_floor(team, elapsed - 0.5):
                stint, cum = self.rot.stint_features(team, pid, elapsed)
                exited = any(elapsed - 1.0 < hi <= elapsed + 10.0
                             for _, hi in self.rot.intervals[(team, pid)])
                self.sub_rows.append({
                    "game_id": self.game_id, "season": self.season,
                    "player_id": pid, "team_id": team,
                    "period": int(period), "sec_remaining": sec_rem,
                    "stint_sec": stint, "cum_sec": cum,
                    "margin_abs": margin,
                    "fouls": self.player_fouls[pid],
                    "is_starter": int(pid in starters),
                    "exited": int(exited),
                })

    # ------------------------------------------------------------ main walk
    def run(self):
        ev = self.events
        i = 0
        while i < len(ev):
            row = ev.iloc[i]
            etype = row["EVENTMSGTYPE"]
            elapsed = float(row["ELAPSED"])
            period = int(row["PERIOD"])

            if etype == EVT_MADE_SHOT and is_player_id(row["PLAYER1_ID"]):
                shooter = int(row["PLAYER1_ID"])
                team = int(row["PLAYER1_TEAM_ID"])
                three = is_3pt(row)
                margin_before = self.margin_for(team)
                def_fouls = self.team_fouls_period[(self.other(team), period)]
                pts = 3 if three else 2
                self.scores[team] += pts

                # and-one: a shooting/personal foul on the shooter at ~the same instant
                shot_result, n_ft, ft_made = "make", 0, 0
                j = i + 1
                while j < len(ev) and abs(float(ev.iloc[j]["ELAPSED"]) - elapsed) < 1.5:
                    nrow = ev.iloc[j]
                    if nrow["EVENTMSGTYPE"] == EVT_FOUL \
                            and nrow["EVENTMSGACTIONTYPE"] not in FOUL_TECHNICAL \
                            and is_player_id(nrow["PLAYER2_ID"]) \
                            and int(nrow["PLAYER2_ID"]) == shooter:
                        self.team_fouls_period[(self.other(team), period)] += 1
                        if is_player_id(nrow["PLAYER1_ID"]):
                            self.player_fouls[int(nrow["PLAYER1_ID"])] += 1
                        n_ft, ft_made, _, j = self._ft_run(j + 1, shooter)
                        shot_result = "andone"
                        break
                    j += 1

                assister = (int(row["PLAYER2_ID"])
                            if is_player_id(row["PLAYER2_ID"])
                            and is_team_id(row["PLAYER2_TEAM_ID"])
                            and int(row["PLAYER2_TEAM_ID"]) == team else None)
                self.emit_chance(
                    elapsed=elapsed, period=period, offense_team=team,
                    ball_ender=shooter,
                    action="3pt_attempt" if three else "2pt_attempt",
                    shot_result=shot_result, points=pts + ft_made,
                    n_ft=n_ft, ft_made=ft_made, assist_id=assister,
                    margin_before=margin_before, def_fouls_before=def_fouls)
                self.emit_sub_decisions(elapsed, period)
                i = j if shot_result == "andone" else i + 1
                continue

            if etype == EVT_MISSED_SHOT and is_player_id(row["PLAYER1_ID"]):
                shooter = int(row["PLAYER1_ID"])
                team = int(row["PLAYER1_TEAM_ID"])
                three = is_3pt(row)
                reb_team, reb_player = self._find_rebound(i + 1)
                oreb = int(reb_team == team) if reb_team is not None else None
                blocker = (int(row["PLAYER3_ID"])
                           if is_player_id(row["PLAYER3_ID"])
                           and is_team_id(row["PLAYER3_TEAM_ID"])
                           and int(row["PLAYER3_TEAM_ID"]) != team else None)
                self.emit_chance(
                    elapsed=elapsed, period=period, offense_team=team,
                    ball_ender=shooter,
                    action="3pt_attempt" if three else "2pt_attempt",
                    shot_result="miss", oreb=oreb, rebounder=reb_player,
                    block_id=blocker,
                    margin_before=self.margin_for(team),
                    def_fouls_before=self.team_fouls_period[(self.other(team), period)])
                i += 1
                continue

            if etype == EVT_TURNOVER:
                if not is_team_id(row["PLAYER1_TEAM_ID"]):
                    i += 1
                    continue
                team = int(row["PLAYER1_TEAM_ID"])
                pid = int(row["PLAYER1_ID"]) if is_player_id(row["PLAYER1_ID"]) else None
                desc = event_desc(row)
                action = "off_foul" if ("OFF.FOUL" in desc or "OFFENSIVE FOUL" in desc
                                        or "CHARGE" in desc) else "turnover"
                if action == "off_foul" and pid is not None:
                    self.player_fouls[pid] += 1
                stealer = (int(row["PLAYER2_ID"])
                           if is_player_id(row["PLAYER2_ID"])
                           and is_team_id(row["PLAYER2_TEAM_ID"])
                           and int(row["PLAYER2_TEAM_ID"]) != team else None)
                self.emit_chance(
                    elapsed=elapsed, period=period, offense_team=team,
                    ball_ender=pid, action=action, steal_id=stealer,
                    margin_before=self.margin_for(team),
                    def_fouls_before=self.team_fouls_period[(self.other(team), period)])
                i += 1
                continue

            if etype == EVT_FOUL:
                atype = row["EVENTMSGACTIONTYPE"]
                fouler_team = int(row["PLAYER1_TEAM_ID"]) if is_team_id(row["PLAYER1_TEAM_ID"]) else None
                if is_player_id(row["PLAYER1_ID"]):
                    self.player_fouls[int(row["PLAYER1_ID"])] += 1
                if atype in FOUL_TECHNICAL or fouler_team is None:
                    i += 1
                    continue
                if atype in FOUL_OFFENSIVE:
                    # handled by the paired turnover event
                    self.team_fouls_period[(fouler_team, period)] += 1
                    i += 1
                    continue

                self.team_fouls_period[(fouler_team, period)] += 1
                offense_team = self.other(fouler_team)
                fouled = int(row["PLAYER2_ID"]) if is_player_id(row["PLAYER2_ID"]) else None

                if atype in FOUL_SHOOTING and fouled is not None:
                    n_ft, ft_made, last_missed, j = self._ft_run(i + 1, fouled)
                    oreb, rebounder = (None, None)
                    if last_missed:
                        reb_team, reb_player = self._find_rebound(j)
                        oreb = int(reb_team == offense_team) if reb_team is not None else None
                        rebounder = reb_player
                    self.emit_chance(
                        elapsed=elapsed, period=period, offense_team=offense_team,
                        ball_ender=fouled,
                        action="3pt_attempt" if n_ft >= 3 else "2pt_attempt",
                        shot_result="shooting_foul", points=ft_made,
                        n_ft=n_ft, ft_made=ft_made, oreb=oreb, rebounder=rebounder,
                        margin_before=self.margin_for(offense_team),
                        def_fouls_before=self.team_fouls_period[(fouler_team, period)] - 1)
                    self.emit_sub_decisions(elapsed, period, force=True)
                    i = max(j, i + 1)
                    continue

                # non-shooting defensive foul: offense keeps the ball (or shoots
                # bonus FTs); ball-ender = the FOULED player, never the defender
                if fouled is not None:
                    n_ft, ft_made, last_missed, j = self._ft_run(i + 1, fouled)
                    self.emit_chance(
                        elapsed=elapsed, period=period, offense_team=offense_team,
                        ball_ender=fouled, action="drawn_foul",
                        points=ft_made, n_ft=n_ft, ft_made=ft_made,
                        margin_before=self.margin_for(offense_team),
                        def_fouls_before=self.team_fouls_period[(fouler_team, period)] - 1)
                    self.emit_sub_decisions(elapsed, period, force=True)
                    i = max(j, i + 1)
                    continue
                i += 1
                continue

            if etype == EVT_FREE_THROW:
                # FTs not consumed by a foul handler (e.g. technicals): score only
                if "MISS" not in event_desc(row) and is_team_id(row["PLAYER1_TEAM_ID"]):
                    self.scores[int(row["PLAYER1_TEAM_ID"])] += 1
                i += 1
                continue

            if etype == EVT_TIMEOUT:
                self.emit_sub_decisions(elapsed, period, force=True)
                i += 1
                continue

            if etype == EVT_PERIOD_END:
                self.chance_start = elapsed
                self.emit_sub_decisions(elapsed, period, force=True)
                i += 1
                continue

            i += 1

        return {
            "chances": self.chances, "fts": self.fts, "subs": self.sub_rows,
            "game_meta": {
                "game_id": self.game_id, "season": self.season,
                "home_team": self.home_team, "away_team": self.away_team,
                "home_score": self.scores[self.home_team],
                "away_score": self.scores[self.away_team],
                "n_chances": len(self.chances), "n_skipped": self.skipped,
            },
        }


def transform_all(seasons=None) -> dict:
    """Run the transformer over every collected game; write tables; return paths."""
    seasons = seasons or config.SEASONS
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_chances, all_fts, all_subs, all_games = [], [], [], []
    for season in seasons:
        tag = season.replace("-", "_")
        events_path = config.EVENTS_DIR / f"events_{tag}.parquet"
        rot_path = config.ROTATIONS_DIR / f"rotations_{tag}.parquet"
        if not events_path.exists() or not rot_path.exists():
            print(f"[{season}] no collected data, skipping")
            continue
        events = pd.read_parquet(events_path)
        rotations = pd.read_parquet(rot_path)
        game_ids = sorted(set(events["GAME_ID"]) & set(rotations["GAME_ID"]))
        print(f"[{season}] transforming {len(game_ids)} games")

        rot_by_game = dict(tuple(rotations.groupby("GAME_ID")))
        ev_by_game = dict(tuple(events.groupby("GAME_ID")))
        for n, gid in enumerate(game_ids, 1):
            try:
                result = GameTransformer(gid, season, ev_by_game[gid], rot_by_game[gid]).run()
            except Exception as e:
                print(f"  {gid}: transform failed ({str(e)[:80]})")
                continue
            all_chances.extend(result["chances"])
            all_fts.extend(result["fts"])
            all_subs.extend(result["subs"])
            all_games.append(result["game_meta"])
            if n % 200 == 0:
                print(f"  [{season}] {n}/{len(game_ids)}")

    chances = pd.DataFrame(all_chances)
    chances.to_parquet(config.CHANCES_FILE, index=False)
    pd.DataFrame(all_fts).to_parquet(config.FT_FILE, index=False)
    pd.DataFrame(all_subs).to_parquet(config.SUB_EVENTS_FILE, index=False)
    games = pd.DataFrame(all_games)
    games.to_parquet(config.GAMES_FILE, index=False)

    if len(games):
        cpg = len(chances) / len(games)
        skip_rate = games["n_skipped"].sum() / max(games["n_skipped"].sum() + len(chances), 1)
        print(f"\n{len(chances):,} chances from {len(games):,} games "
              f"({cpg:.0f} per game, {skip_rate:.1%} skipped on lineup issues)")
    return {"chances": config.CHANCES_FILE, "games": config.GAMES_FILE}


if __name__ == "__main__":
    transform_all()
