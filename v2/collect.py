"""Event-level data collection (PlayByPlayV3 + BoxScoreTraditionalV3).

The NBA retired PlayByPlayV2 and GameRotation (empty JSON as of 2025), so:
  * events come from PlayByPlayV3 — structured shotValue/shotResult/subType,
    adjacent STEAL/BLOCK rows with their own personId, scoreHome/scoreAway;
  * game starters + rosters come from BoxScoreTraditionalV3 (position field);
  * on-floor lineups are reconstructed offline in transform.py from
    substitution rows with period-start inference.

Two API calls per game. Checkpointed and resumable.

Usage:
    python -m v2.collect                  # all seasons in config.SEASONS
    python -m v2.collect --season 2023-24 --max-games 50
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

EVENT_COLUMNS = ["GAME_ID", "SEASON", "EVENTNUM", "ACTION_TYPE", "SUB_TYPE",
                 "PERIOD", "ELAPSED", "TEAM_ID", "PERSON_ID", "SHOT_VALUE",
                 "SHOT_RESULT", "SCORE_HOME", "SCORE_AWAY", "DESCRIPTION"]

PLAYER_COLUMNS = ["GAME_ID", "SEASON", "TEAM_ID", "PERSON_ID", "NAME",
                  "NAME_I", "FAMILY_NAME", "STARTER", "MINUTES"]

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


def period_start_elapsed(period: int) -> float:
    """Seconds of game time elapsed when `period` begins."""
    reg = min(period - 1, config.REGULATION_PERIODS) * config.PERIOD_SECONDS
    ot = max(period - 1 - config.REGULATION_PERIODS, 0) * config.OT_SECONDS
    return reg + ot


def parse_clock(clock: str) -> float:
    """'PT11M38.00S' -> seconds remaining in period."""
    m = _CLOCK_RE.match(str(clock))
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))


def clock_to_elapsed(period: int, clock: str) -> float:
    length = config.PERIOD_SECONDS if period <= config.REGULATION_PERIODS else config.OT_SECONDS
    return period_start_elapsed(period) + (length - parse_clock(clock))


def minutes_to_float(v) -> float:
    s = str(v)
    if ":" in s:
        m, sec = s.split(":")
        try:
            return float(m) + float(sec) / 60.0
        except ValueError:
            return 0.0
    return 0.0


def retry(func, max_retries: int = 3, base_delay: float = 2.0):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


class EventCollector:
    def __init__(self, seasons=None, request_delay: float = 0.8):
        self.seasons = seasons or config.SEASONS
        self.request_delay = request_delay
        config.EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        config.PLAYERS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _season_paths(season: str):
        tag = season.replace("-", "_")
        return (config.EVENTS_DIR / f"events_{tag}.parquet",
                config.PLAYERS_DIR / f"players_{tag}.parquet")

    @staticmethod
    def _done_game_ids(path: Path) -> set:
        if path.exists():
            return set(pd.read_parquet(path, columns=["GAME_ID"])["GAME_ID"].unique())
        return set()

    def list_game_ids(self, season: str) -> list:
        from nba_api.stats.endpoints import leaguegamefinder

        games = retry(lambda: leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable="Regular Season",
            league_id_nullable="00",
        ).get_data_frames()[0])
        return sorted(games["GAME_ID"].unique())

    def fetch_game(self, game_id: str, season: str):
        """Returns (events_df, players_df) or None."""
        from nba_api.stats.endpoints import boxscoretraditionalv3, playbyplayv3

        time.sleep(self.request_delay)
        pbp = retry(lambda: playbyplayv3.PlayByPlayV3(
            game_id=game_id, timeout=60).get_data_frames()[0])
        if pbp.empty:
            return None

        time.sleep(self.request_delay)
        box = retry(lambda: boxscoretraditionalv3.BoxScoreTraditionalV3(
            game_id=game_id, timeout=60).get_data_frames()[0])
        if box.empty:
            return None

        events = pd.DataFrame({
            "GAME_ID": game_id,
            "SEASON": season,
            "EVENTNUM": pbp["actionNumber"],
            "ACTION_TYPE": pbp["actionType"].fillna(""),
            "SUB_TYPE": pbp["subType"].fillna(""),
            "PERIOD": pbp["period"].astype(int),
            "ELAPSED": [clock_to_elapsed(p, c)
                        for p, c in zip(pbp["period"], pbp["clock"])],
            "TEAM_ID": pd.to_numeric(pbp["teamId"], errors="coerce").fillna(0).astype(np.int64),
            "PERSON_ID": pd.to_numeric(pbp["personId"], errors="coerce").fillna(0).astype(np.int64),
            "SHOT_VALUE": pd.to_numeric(pbp["shotValue"], errors="coerce").fillna(0).astype(int),
            "SHOT_RESULT": pbp["shotResult"].fillna(""),
            "SCORE_HOME": pd.to_numeric(pbp["scoreHome"], errors="coerce").fillna(0).astype(int),
            "SCORE_AWAY": pd.to_numeric(pbp["scoreAway"], errors="coerce").fillna(0).astype(int),
            "DESCRIPTION": pbp["description"].fillna(""),
        })

        players = pd.DataFrame({
            "GAME_ID": game_id,
            "SEASON": season,
            "TEAM_ID": box["teamId"].astype(np.int64),
            "PERSON_ID": box["personId"].astype(np.int64),
            "NAME": box["firstName"].fillna("") + " " + box["familyName"].fillna(""),
            "NAME_I": box["nameI"].fillna(""),
            "FAMILY_NAME": box["familyName"].fillna(""),
            "STARTER": (box["position"].fillna("") != "").astype(int),
            "MINUTES": box["minutes"].fillna("").map(minutes_to_float),
        })
        if players["STARTER"].sum() != 10:
            # fall back to top-5 minutes per team
            players["STARTER"] = 0
            for tid, grp in players.groupby("TEAM_ID"):
                top5 = grp.nlargest(5, "MINUTES").index
                players.loc[top5, "STARTER"] = 1
        return events, players

    def collect_season(self, season: str, max_games: int = None):
        events_path, players_path = self._season_paths(season)
        done = self._done_game_ids(events_path)
        game_ids = [g for g in self.list_game_ids(season) if g not in done]
        if max_games is not None:
            game_ids = game_ids[:max_games]
        print(f"[{season}] {len(done)} games already collected, {len(game_ids)} to go")

        events_buf, players_buf = [], []
        for i, game_id in enumerate(game_ids, 1):
            try:
                result = self.fetch_game(game_id, season)
            except Exception as e:
                print(f"  {game_id}: failed after retries ({str(e)[:80]})")
                continue
            if result is None:
                print(f"  {game_id}: empty data, skipped")
                continue
            ev, pl = result
            events_buf.append(ev)
            players_buf.append(pl)

            if i % 25 == 0 or i == len(game_ids):
                self._flush(events_buf, events_path)
                self._flush(players_buf, players_path)
                events_buf, players_buf = [], []
                print(f"  [{season}] {i}/{len(game_ids)} games collected")

    @staticmethod
    def _flush(frames: list, path: Path):
        if not frames:
            return
        new = pd.concat(frames, ignore_index=True)
        if path.exists():
            new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
        new.to_parquet(path, index=False)

    def collect_all(self, max_games: int = None):
        for season in self.seasons:
            self.collect_season(season, max_games=max_games)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=None, help="single season, e.g. 2023-24")
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()
    seasons = [args.season] if args.season else None
    EventCollector(seasons=seasons).collect_all(max_games=args.max_games)


if __name__ == "__main__":
    main()
