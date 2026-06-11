"""Event-level data collection with rotation-anchored lineups.

Fixes versus the legacy collector:
  * Keeps ALL play-by-play events (rebounds, FTs, subs, timeouts), not just
    possession-ending ones.
  * Lineups come from the GameRotation endpoint (exact on-floor intervals),
    so they cannot drift across period boundaries the way substitution
    tracking does.
  * No outcome parsing happens here; labeling lives in transform.py and can
    be re-run without touching the network.

Usage:
    python -m v2.collect                  # all seasons in config.SEASONS
    python -m v2.collect --season 2023-24 --max-games 50
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

EVENT_COLUMNS = [
    "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD",
    "PCTIMESTRING", "HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION",
    "SCORE", "SCOREMARGIN",
    "PLAYER1_ID", "PLAYER1_NAME", "PLAYER1_TEAM_ID",
    "PLAYER2_ID", "PLAYER2_NAME", "PLAYER2_TEAM_ID",
    "PLAYER3_ID", "PLAYER3_NAME", "PLAYER3_TEAM_ID",
]


def period_start_elapsed(period: int) -> float:
    """Seconds of game time elapsed when `period` begins."""
    reg = min(period - 1, config.REGULATION_PERIODS) * config.PERIOD_SECONDS
    ot = max(period - 1 - config.REGULATION_PERIODS, 0) * config.OT_SECONDS
    return reg + ot


def clock_to_elapsed(period: int, clock_str: str) -> float:
    """Convert (period, 'MM:SS' remaining) to total seconds elapsed."""
    try:
        m, s = str(clock_str).split(":")
        remaining = int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        remaining = 0
    length = config.PERIOD_SECONDS if period <= config.REGULATION_PERIODS else config.OT_SECONDS
    return period_start_elapsed(period) + (length - remaining)


def retry(func, max_retries: int = 3, base_delay: float = 2.0):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


class EventCollector:
    """Collects raw events + rotations for a set of seasons, with checkpointing."""

    def __init__(self, seasons=None, request_delay: float = 0.8):
        self.seasons = seasons or config.SEASONS
        self.request_delay = request_delay
        config.EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        config.ROTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _season_paths(season: str):
        tag = season.replace("-", "_")
        return (
            config.EVENTS_DIR / f"events_{tag}.parquet",
            config.ROTATIONS_DIR / f"rotations_{tag}.parquet",
        )

    @staticmethod
    def _done_game_ids(path: Path) -> set:
        if path.exists():
            return set(pd.read_parquet(path, columns=["GAME_ID"])["GAME_ID"].unique())
        return set()

    def list_game_ids(self, season: str) -> list:
        """Regular-season game ids for a season, oldest first."""
        from nba_api.stats.endpoints import leaguegamefinder

        games = retry(lambda: leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable="Regular Season",
            league_id_nullable="00",
        ).get_data_frames()[0])
        ids = sorted(games["GAME_ID"].unique())
        return ids

    def fetch_game(self, game_id: str):
        """Return (events_df, rotation_df) for one game, or None on failure."""
        from nba_api.stats.endpoints import gamerotation, playbyplayv2

        time.sleep(self.request_delay)
        events = retry(lambda: playbyplayv2.PlayByPlayV2(
            game_id=game_id, timeout=60).get_data_frames()[0])
        if events.empty:
            return None

        time.sleep(self.request_delay)
        rot_frames = retry(lambda: gamerotation.GameRotation(
            game_id=game_id, timeout=60).get_data_frames())
        rotation = pd.concat(rot_frames, ignore_index=True)
        if rotation.empty:
            return None

        events = events.reindex(columns=EVENT_COLUMNS)
        events["GAME_ID"] = game_id
        events["ELAPSED"] = [
            clock_to_elapsed(p, c) for p, c in zip(events["PERIOD"], events["PCTIMESTRING"])
        ]

        # GameRotation reports IN/OUT in tenths of a second of game time.
        rotation = rotation.rename(columns=str.upper)
        keep = ["GAME_ID", "TEAM_ID", "PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"]
        rotation = rotation[keep].copy()
        rotation["IN_SEC"] = rotation["IN_TIME_REAL"] / 10.0
        rotation["OUT_SEC"] = rotation["OUT_TIME_REAL"] / 10.0
        rotation["GAME_ID"] = game_id
        return events, rotation

    # ------------------------------------------------------------ main loop
    def collect_season(self, season: str, max_games: int = None):
        events_path, rot_path = self._season_paths(season)
        done = self._done_game_ids(events_path)
        game_ids = [g for g in self.list_game_ids(season) if g not in done]
        if max_games is not None:
            game_ids = game_ids[:max_games]
        print(f"[{season}] {len(done)} games already collected, {len(game_ids)} to go")

        events_buf, rot_buf = [], []
        for i, game_id in enumerate(game_ids, 1):
            try:
                result = self.fetch_game(game_id)
            except Exception as e:
                print(f"  {game_id}: failed after retries ({str(e)[:80]})")
                continue
            if result is None:
                print(f"  {game_id}: empty data, skipped")
                continue
            ev, rot = result
            ev["SEASON"] = season
            rot["SEASON"] = season
            events_buf.append(ev)
            rot_buf.append(rot)

            if i % 25 == 0 or i == len(game_ids):
                self._flush(events_buf, events_path)
                self._flush(rot_buf, rot_path)
                events_buf, rot_buf = [], []
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
