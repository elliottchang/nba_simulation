"""Transform raw V3 events + game rosters into model-ready training tables.

Attribution rules (all fixes vs. legacy):
  * ball-ender = shooter / turnover committer / FOULED player on defensive
    fouls (parsed from the foul description, resolved against the opposing
    roster) — never the defender
  * offense-perspective score margin from accumulated scores
  * lineups reconstructed from substitution rows with period-start
    inference; chances whose lineups can't be established are skipped and
    counted (the skip rate is a data-quality metric, printed at the end)
  * misses linked to their rebound; assists/steals/blocks from the
    structured adjacent rows
  * per-lineup stint/cumulative-minutes features for fatigue learning

Outputs (data_v2/tables/): chances, free_throws, sub_decisions, games.
Re-runnable offline whenever labeling logic changes.
"""

import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config
from .config import (
    ACT_MADE, ACT_MISSED, ACT_FT, ACT_REBOUND, ACT_TURNOVER, ACT_FOUL,
    ACT_SUB, ACT_TIMEOUT, ACT_PERIOD,
)
from .collect import period_start_elapsed

MADE_SHOT_DEADBALL_SAMPLE = 0.3   # sub-decision sampling rate at made-shot deadballs

CHANCE_ENDERS = {ACT_MADE, ACT_MISSED, ACT_TURNOVER, ACT_FOUL}
PRESENCE_ACTIONS = {ACT_MADE, ACT_MISSED, ACT_FT, ACT_REBOUND, ACT_TURNOVER,
                    ACT_FOUL, ACT_SUB, "Jump Ball", ""}

_SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)$", re.IGNORECASE)
_AST_RE = re.compile(r"\(([\w .'-]+?)\s+\d+\s+AST\)")
_INITIAL_RE = re.compile(r"^([A-Z])\.\s*(.+)$")


def norm_name(s: str) -> str:
    return re.sub(r"[\s.]", "", str(s)).lower()


class GameRoster:
    """Name-resolution and starter info for one game from the boxscore."""

    def __init__(self, players_df: pd.DataFrame):
        self.team_ids = sorted(players_df["TEAM_ID"].unique().tolist())
        self.players = defaultdict(set)        # team -> pids
        self.starters = defaultdict(set)
        self.by_family = defaultdict(list)     # (team, norm family) -> [pids]
        self.by_initial = defaultdict(list)    # (team, norm nameI) -> [pids]
        for r in players_df.itertuples():
            t, p = int(r.TEAM_ID), int(r.PERSON_ID)
            self.players[t].add(p)
            if r.STARTER:
                self.starters[t].add(p)
            self.by_family[(t, norm_name(r.FAMILY_NAME))].append(p)
            self.by_initial[(t, norm_name(r.NAME_I))].append(p)

    def other(self, team_id: int) -> int:
        return self.team_ids[1] if team_id == self.team_ids[0] else self.team_ids[0]

    def resolve_family(self, team_id: int, name: str, prefer_not_in: set = None) -> int:
        """Resolve 'Wagner', 'Nance Jr.' or initial-disambiguated 'M. Wagner'."""
        cands = self.by_family.get((int(team_id), norm_name(name)), [])
        if not cands:
            m = _INITIAL_RE.match(str(name).strip())
            if m:
                cands = self.by_initial.get(
                    (int(team_id), norm_name(f"{m.group(1)}.{m.group(2)}")), [])
        if prefer_not_in and len(cands) > 1:
            filtered = [c for c in cands if c not in prefer_not_in]
            if filtered:
                cands = filtered
        return cands[0] if cands else None


class LineupTracker:
    """Reconstructs on-floor intervals from substitution rows.

    Q1 starts from boxscore starters; later periods are inferred: a player
    is on at period start if he registers any event before being subbed IN
    during that period; gaps are filled by carry-over from the previous
    period's closing lineup.
    """

    def __init__(self, events: pd.DataFrame, roster: GameRoster):
        self.events = events
        self.roster = roster
        self.intervals = defaultdict(list)     # (team, pid) -> [(in, out)]
        self.fill_periods = 0                  # diagnostics
        self._build()

    def _parse_sub(self, row):
        m = _SUB_RE.match(str(row.DESCRIPTION))
        if not m:
            return None, None
        team = int(row.TEAM_ID)
        out_pid = int(row.PERSON_ID)
        in_pid = self.roster.resolve_family(team, m.group(1))
        return out_pid, in_pid

    def _infer_period_starters(self, period_events, team: int, carry: set) -> set:
        evidence, subbed_in = set(), set()
        for r in period_events.itertuples():
            if int(r.TEAM_ID) != team:
                continue
            if r.ACTION_TYPE == ACT_SUB:
                out_pid, in_pid = self._parse_sub(r)
                if out_pid and out_pid not in subbed_in:
                    evidence.add(out_pid)
                if in_pid:
                    subbed_in.add(in_pid)
            elif r.ACTION_TYPE in PRESENCE_ACTIONS:
                pid = int(r.PERSON_ID)
                if pid > 0 and pid in self.roster.players[team] and pid not in subbed_in:
                    evidence.add(pid)
        starters = set(list(evidence)[:5])
        if len(starters) < 5:
            self.fill_periods += 1
            for pid in carry:
                if len(starters) == 5:
                    break
                if pid not in subbed_in and pid not in starters:
                    starters.add(pid)
        return starters

    def _build(self):
        ev = self.events
        periods = sorted(ev["PERIOD"].unique())
        game_end = float(ev["ELAPSED"].max())
        on_floor = {t: set() for t in self.roster.team_ids}
        entry = {}

        def come_on(team, pid, t):
            if pid not in on_floor[team]:
                on_floor[team].add(pid)
                entry[(team, pid)] = t

        def go_off(team, pid, t):
            if pid in on_floor[team]:
                on_floor[team].discard(pid)
                t0 = entry.pop((team, pid), t)
                if t > t0:
                    self.intervals[(team, pid)].append((t0, t))

        for period in periods:
            pstart = period_start_elapsed(int(period))
            pev = ev[ev["PERIOD"] == period]
            for team in self.roster.team_ids:
                if period == 1:
                    starters = set(self.roster.starters[team]) or set(
                        list(self.roster.players[team])[:5])
                else:
                    starters = self._infer_period_starters(pev, team, on_floor[team])
                for pid in list(on_floor[team]):
                    if pid not in starters:
                        go_off(team, pid, pstart)
                for pid in starters:
                    come_on(team, pid, pstart)
            for r in pev.itertuples():
                if r.ACTION_TYPE == ACT_SUB:
                    team = int(r.TEAM_ID)
                    out_pid, in_pid = self._parse_sub(r)
                    if out_pid:
                        go_off(team, out_pid, float(r.ELAPSED))
                    if in_pid:
                        come_on(team, in_pid, float(r.ELAPSED))
                elif r.ACTION_TYPE in PRESENCE_ACTIONS:
                    # self-heal: an actor we lost track of is evidently on the
                    # floor — re-anchor him if there is room
                    team, pid = int(r.TEAM_ID), int(r.PERSON_ID)
                    if (pid > 0 and pid in self.roster.players.get(team, ())
                            and pid not in on_floor[team]
                            and len(on_floor[team]) < 5):
                        come_on(team, pid, float(r.ELAPSED))
        for team in self.roster.team_ids:
            for pid in list(on_floor[team]):
                go_off(team, pid, game_end)

    # ----- same query interface the transformer used with GameRotation data
    def on_floor_at(self, team_id: int, t: float) -> list:
        out = []
        for (team, pid), spans in self.intervals.items():
            if team != team_id:
                continue
            if any(lo <= t < hi for lo, hi in spans):
                out.append(pid)
        return sorted(out)

    def lineup_at(self, team_id: int, t: float, must_include: int = None) -> list:
        for probe in (t, t - 0.5, t + 0.5):
            lineup = self.on_floor_at(team_id, probe)
            if len(lineup) == 5 and (must_include is None or must_include in lineup):
                return lineup
        lineup = self.on_floor_at(team_id, t)
        return lineup if len(lineup) == 5 else None

    def stint_features(self, team_id: int, player_id: int, t: float):
        stint, cum = 0.0, 0.0
        for lo, hi in self.intervals.get((team_id, player_id), []):
            if lo <= t:
                cum += min(hi, t) - lo
                if t < hi:
                    stint = t - lo
        return stint, cum


def infer_home_team(events: pd.DataFrame, team_ids: list) -> int:
    """Home team = team whose made shots move SCORE_HOME."""
    votes = defaultdict(int)
    last_h = last_a = 0
    scoring = events[(events["SCORE_HOME"] > 0) | (events["SCORE_AWAY"] > 0)]
    for r in scoring.itertuples():
        h, a = int(r.SCORE_HOME), int(r.SCORE_AWAY)
        team = int(r.TEAM_ID)
        if team > 0:
            if h > last_h and a == last_a:
                votes[team] += 1
            elif a > last_a and h == last_h:
                votes[team] -= 1
        last_h, last_a = h, a
    if not votes:
        return team_ids[0]
    return max(votes, key=votes.get)


class GameTransformer:
    """Walks one game's canonical V3 events and emits training rows."""

    def __init__(self, game_id, season, events, players_df):
        self.game_id = game_id
        self.season = season
        self.events = events.sort_values("EVENTNUM").reset_index(drop=True)
        self.roster = GameRoster(players_df)
        if len(self.roster.team_ids) != 2:
            raise ValueError("expected exactly 2 teams in boxscore data")
        self.tracker = LineupTracker(self.events, self.roster)
        self.home_team = infer_home_team(self.events, self.roster.team_ids)
        self.away_team = self.roster.other(self.home_team)

        self.scores = {self.home_team: 0, self.away_team: 0}
        self.team_fouls_period = defaultdict(int)
        self.player_fouls = defaultdict(int)
        self.chance_start = 0.0
        self.consumed_ft = set()      # row indices already absorbed by a foul

        self.chances, self.fts, self.sub_rows = [], [], []
        self.skipped = 0

    # ------------------------------------------------------------ utilities
    def other(self, team_id: int) -> int:
        return self.away_team if team_id == self.home_team else self.home_team

    def margin_for(self, offense_team: int) -> int:
        return self.scores[offense_team] - self.scores[self.other(offense_team)]

    def period_seconds_remaining(self, period: int, elapsed: float) -> float:
        length = (config.PERIOD_SECONDS if period <= config.REGULATION_PERIODS
                  else config.OT_SECONDS)
        return max(0.0, period_start_elapsed(period) + length - elapsed)

    def _ft_run(self, start_idx: int, shooter: int):
        """Consecutive FTs by `shooter`. Returns (n, made, last_missed, end_idx)."""
        n = made = 0
        last_missed = False
        i = start_idx
        while i < len(self.events):
            row = self.events.iloc[i]
            at = row["ACTION_TYPE"]
            if at == ACT_FT and int(row["PERSON_ID"]) == shooter:
                n += 1
                self.consumed_ft.add(i)
                missed = str(row["DESCRIPTION"]).upper().startswith("MISS")
                last_missed = missed
                if not missed:
                    made += 1
                    self.scores[int(row["TEAM_ID"])] += 1
                self.fts.append({"game_id": self.game_id, "season": self.season,
                                 "player_id": shooter, "made": int(not missed)})
                i += 1
            elif at in ("", ACT_SUB, ACT_TIMEOUT):   # interleaved non-action rows
                i += 1
            else:
                break
        return n, made, last_missed, i

    def _find_rebound(self, start_idx: int):
        for i in range(start_idx, min(start_idx + 6, len(self.events))):
            row = self.events.iloc[i]
            at = row["ACTION_TYPE"]
            if at == ACT_REBOUND:
                team = int(row["TEAM_ID"])
                pid = int(row["PERSON_ID"])
                if team > 0:
                    return team, (pid if pid > 0 else None)
            if at in CHANCE_ENDERS or (at == ACT_PERIOD and row["SUB_TYPE"] == "end"):
                break
        return None, None

    def _adjacent_marker(self, idx: int, marker: str):
        """personId of an adjacent blank-actionType row like 'X STEAL (1 STL)'.
        Marker rows share their parent event's EVENTNUM."""
        parent_num = self.events.iloc[idx]["EVENTNUM"]
        for j in (idx + 1, idx - 1):
            if 0 <= j < len(self.events):
                row = self.events.iloc[j]
                if row["ACTION_TYPE"] == "" and row["EVENTNUM"] == parent_num \
                        and marker in str(row["DESCRIPTION"]).upper():
                    pid = int(row["PERSON_ID"])
                    if pid > 0:
                        return pid
        return None

    def _parse_assister(self, row, team: int, shooter: int):
        m = _AST_RE.search(str(row["DESCRIPTION"]))
        if not m:
            return None
        pid = self.roster.resolve_family(team, m.group(1), prefer_not_in={shooter})
        return pid if pid and pid != shooter else None

    def _next_ft_shooter(self, start_idx: int, offense_team: int):
        """The fouled player is identified by who shoots the ensuing FTs
        (V3 foul descriptions name the referee, not the fouled player)."""
        for j in range(start_idx, min(start_idx + 6, len(self.events))):
            row = self.events.iloc[j]
            at = row["ACTION_TYPE"]
            if at == ACT_FT:
                pid = int(row["PERSON_ID"])
                if int(row["TEAM_ID"]) == offense_team and pid > 0:
                    return pid
                return None
            if at not in ("", ACT_SUB, ACT_TIMEOUT):
                return None
        return None

    # ------------------------------------------------------------ row emission
    def emit_chance(self, *, elapsed, period, offense_team, ball_ender, action,
                    shot_result=None, points=0, n_ft=0, ft_made=0,
                    oreb=None, rebounder=None, margin_before=None,
                    def_fouls_before=0, assist_id=None, steal_id=None,
                    block_id=None):
        defense_team = self.other(offense_team)
        off_lineup = self.tracker.lineup_at(offense_team, elapsed, must_include=ball_ender)
        def_lineup = self.tracker.lineup_at(defense_team, elapsed)
        if off_lineup is None or def_lineup is None:
            self.skipped += 1
            self.chance_start = elapsed
            return
        slot = off_lineup.index(ball_ender) if ball_ender in off_lineup else -1
        off_stints = [self.tracker.stint_features(offense_team, p, elapsed) for p in off_lineup]
        def_stints = [self.tracker.stint_features(defense_team, p, elapsed) for p in def_lineup]
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
            "oreb": oreb if oreb is not None else -1,
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
        if not force and random.random() > MADE_SHOT_DEADBALL_SAMPLE:
            return
        margin = abs(self.scores[self.home_team] - self.scores[self.away_team])
        sec_rem = self.period_seconds_remaining(period, elapsed)
        for team in self.roster.team_ids:
            starters = self.roster.starters[team]
            for pid in self.tracker.on_floor_at(team, elapsed - 0.5):
                stint, cum = self.tracker.stint_features(team, pid, elapsed)
                exited = any(elapsed - 1.0 < hi <= elapsed + 10.0
                             for _, hi in self.tracker.intervals[(team, pid)])
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
            at = row["ACTION_TYPE"]
            sub_type = str(row["SUB_TYPE"])
            elapsed = float(row["ELAPSED"])
            period = int(row["PERIOD"])
            team = int(row["TEAM_ID"])
            pid = int(row["PERSON_ID"])

            if at == ACT_MADE and pid > 0:
                three = int(row["SHOT_VALUE"]) == 3
                margin_before = self.margin_for(team)
                def_fouls = self.team_fouls_period[(self.other(team), period)]
                pts = 3 if three else 2
                self.scores[team] += pts
                assister = self._parse_assister(row, team, pid)

                # and-one: defensive foul at ~the same instant whose FTs go
                # to the shooter
                shot_result, n_ft, ft_made = "make", 0, 0
                j = i + 1
                while j < len(ev) and abs(float(ev.iloc[j]["ELAPSED"]) - elapsed) < 1.5:
                    nrow = ev.iloc[j]
                    nsub = str(nrow["SUB_TYPE"])
                    if nrow["ACTION_TYPE"] == ACT_FOUL \
                            and "Offensive" not in nsub and "Technical" not in nsub \
                            and int(nrow["TEAM_ID"]) == self.other(team):
                        if self._next_ft_shooter(j + 1, team) == pid:
                            self.team_fouls_period[(self.other(team), period)] += 1
                            if int(nrow["PERSON_ID"]) > 0:
                                self.player_fouls[int(nrow["PERSON_ID"])] += 1
                            n_ft, ft_made, _, j = self._ft_run(j + 1, pid)
                            shot_result = "andone"
                        break
                    j += 1

                self.emit_chance(
                    elapsed=elapsed, period=period, offense_team=team,
                    ball_ender=pid,
                    action="3pt_attempt" if three else "2pt_attempt",
                    shot_result=shot_result, points=pts + ft_made,
                    n_ft=n_ft, ft_made=ft_made, assist_id=assister,
                    margin_before=margin_before, def_fouls_before=def_fouls)
                self.emit_sub_decisions(elapsed, period)
                i = j if shot_result == "andone" else i + 1
                continue

            if at == ACT_MISSED and pid > 0:
                three = int(row["SHOT_VALUE"]) == 3
                blocker = self._adjacent_marker(i, "BLOCK")
                reb_team, reb_player = self._find_rebound(i + 1)
                oreb = int(reb_team == team) if reb_team is not None else None
                self.emit_chance(
                    elapsed=elapsed, period=period, offense_team=team,
                    ball_ender=pid,
                    action="3pt_attempt" if three else "2pt_attempt",
                    shot_result="miss", oreb=oreb, rebounder=reb_player,
                    block_id=blocker,
                    margin_before=self.margin_for(team),
                    def_fouls_before=self.team_fouls_period[(self.other(team), period)])
                i += 1
                continue

            if at == ACT_TURNOVER and team > 0:
                action = ("off_foul" if "offensive foul" in sub_type.lower()
                          else "turnover")
                ball_ender = pid if pid > 0 else None
                if action == "off_foul" and ball_ender is not None:
                    self.player_fouls[ball_ender] += 1
                stealer = self._adjacent_marker(i, "STEAL")
                self.emit_chance(
                    elapsed=elapsed, period=period, offense_team=team,
                    ball_ender=ball_ender, action=action,
                    steal_id=stealer if action == "turnover" else None,
                    margin_before=self.margin_for(team),
                    def_fouls_before=self.team_fouls_period[(self.other(team), period)])
                i += 1
                continue

            if at == ACT_FOUL and team > 0:
                if pid > 0:
                    self.player_fouls[pid] += 1
                if "Technical" in sub_type:
                    i += 1
                    continue
                if "Offensive" in sub_type:
                    # the paired Turnover row carries the chance
                    i += 1
                    continue

                self.team_fouls_period[(team, period)] += 1
                offense_team = self.other(team)
                fouled = self._next_ft_shooter(i + 1, offense_team)

                if "Shooting" in sub_type and fouled is not None:
                    n_ft, ft_made, last_missed, j = self._ft_run(i + 1, fouled)
                    oreb, rebounder = None, None
                    if last_missed:
                        reb_team, reb_player = self._find_rebound(j)
                        oreb = (int(reb_team == offense_team)
                                if reb_team is not None else None)
                        rebounder = reb_player
                    self.emit_chance(
                        elapsed=elapsed, period=period, offense_team=offense_team,
                        ball_ender=fouled,
                        action="3pt_attempt" if n_ft >= 3 else "2pt_attempt",
                        shot_result="shooting_foul", points=ft_made,
                        n_ft=n_ft, ft_made=ft_made, oreb=oreb, rebounder=rebounder,
                        margin_before=self.margin_for(offense_team),
                        def_fouls_before=self.team_fouls_period[(team, period)] - 1)
                    self.emit_sub_decisions(elapsed, period, force=True)
                    i = max(j, i + 1)
                    continue

                # non-shooting defensive foul: if it produced (bonus) FTs the
                # shooter is the fouled player; otherwise the drawn foul is
                # kept at lineup level (ball_ender=None -> slot -1)
                n_ft, ft_made, last_missed, j = 0, 0, False, i + 1
                oreb, rebounder = None, None
                if fouled is not None:
                    n_ft, ft_made, last_missed, j = self._ft_run(i + 1, fouled)
                    if last_missed:
                        reb_team, reb_player = self._find_rebound(j)
                        oreb = (int(reb_team == offense_team)
                                if reb_team is not None else None)
                        rebounder = reb_player
                self.emit_chance(
                    elapsed=elapsed, period=period, offense_team=offense_team,
                    ball_ender=fouled, action="drawn_foul",
                    points=ft_made, n_ft=n_ft, ft_made=ft_made,
                    oreb=oreb, rebounder=rebounder,
                    margin_before=self.margin_for(offense_team),
                    def_fouls_before=self.team_fouls_period[(team, period)] - 1)
                self.emit_sub_decisions(elapsed, period, force=True)
                i = max(j, i + 1)
                continue

            if at == ACT_FT:
                # stray FTs not consumed by a foul handler (e.g. technicals)
                if i not in self.consumed_ft and team > 0 \
                        and not str(row["DESCRIPTION"]).upper().startswith("MISS"):
                    self.scores[team] += 1
                i += 1
                continue

            if at == ACT_TIMEOUT:
                self.emit_sub_decisions(elapsed, period, force=True)
                i += 1
                continue

            if at == ACT_PERIOD and sub_type == "end":
                self.chance_start = period_start_elapsed(period + 1)
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
                "fill_periods": self.tracker.fill_periods,
            },
        }


def transform_all(seasons=None) -> dict:
    seasons = seasons or config.SEASONS
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_chances, all_fts, all_subs, all_games = [], [], [], []
    for season in seasons:
        tag = season.replace("-", "_")
        events_path = config.EVENTS_DIR / f"events_{tag}.parquet"
        players_path = config.PLAYERS_DIR / f"players_{tag}.parquet"
        if not events_path.exists() or not players_path.exists():
            print(f"[{season}] no collected data, skipping")
            continue
        events = pd.read_parquet(events_path)
        players = pd.read_parquet(players_path)
        game_ids = sorted(set(events["GAME_ID"]) & set(players["GAME_ID"]))
        print(f"[{season}] transforming {len(game_ids)} games")

        ev_by_game = dict(tuple(events.groupby("GAME_ID")))
        pl_by_game = dict(tuple(players.groupby("GAME_ID")))
        for n, gid in enumerate(game_ids, 1):
            try:
                result = GameTransformer(gid, season, ev_by_game[gid],
                                         pl_by_game[gid]).run()
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
        n_skip = games["n_skipped"].sum()
        skip_rate = n_skip / max(n_skip + len(chances), 1)
        print(f"\n{len(chances):,} chances from {len(games):,} games "
              f"({cpg:.0f} per game, {skip_rate:.1%} skipped on lineup issues, "
              f"{games['fill_periods'].mean():.1f} carry-over periods/game)")
    return {"chances": config.CHANCES_FILE, "games": config.GAMES_FILE}


if __name__ == "__main__":
    transform_all()
