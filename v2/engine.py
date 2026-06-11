"""Game engine v2.

Everything behavioral is fitted, not tuned:
  * outcomes/attribution  -> factorized possession model
  * pace                  -> empirical duration model
  * substitutions         -> learned exit policy on real coach decisions
  * fatigue               -> stint/cumulative-minutes features flowing into
                             the possession model (its effect was estimated
                             from real data at training time)
  * free throws           -> real (shrunken) FT%
  * bonus / team fouls / foul-outs -> actual NBA rules, tracked exactly

Rosters and starters come from the collected data (no live API calls), so a
full season simulates without network access.
"""

import numpy as np
import pandas as pd

from . import config
from .duration import DurationModel
from .features import FeatureStore
from .predictor import FactorizedPredictor
from .substitution import SubstitutionModel, FEATURES as SUB_FEATURES

BOX_COLS = ["MIN", "PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK",
            "FGA", "FGM", "3PA", "3PM", "FTA", "FTM", "TO", "PF", "+/-"]


class TeamState:
    def __init__(self, team_id: int, season: str, fs: FeatureStore):
        self.team_id = int(team_id)
        self.season = season
        roster = fs.rosters
        roster = roster[(roster["team_id"] == team_id) & (roster["season"] == season)]
        roster = roster.sort_values("floor_share", ascending=False).head(13)
        if len(roster) < 8:
            raise ValueError(f"team {team_id} {season}: only {len(roster)} players in data")
        self.players = roster["player_id"].astype(int).tolist()
        self.floor_share = dict(zip(roster["player_id"].astype(int), roster["floor_share"]))
        starters = roster.sort_values(["starter_rate", "floor_share"],
                                      ascending=False).head(5)
        self.starters = starters["player_id"].astype(int).tolist()
        self.lineup = list(self.starters)

        self.score = 0
        self.fouls_this_period = 0
        self.box = {p: dict.fromkeys(BOX_COLS, 0.0) for p in self.players}
        self.entry_time = {p: (0.0 if p in self.lineup else None) for p in self.players}
        self.exit_time = dict.fromkeys(self.players, 0.0)
        self.cum_sec = dict.fromkeys(self.players, 0.0)

    def stint_sec(self, pid, now) -> float:
        t = self.entry_time.get(pid)
        return (now - t) if t is not None else 0.0

    def fouled_out(self, pid) -> bool:
        return self.box[pid]["PF"] >= config.FOUL_OUT_LIMIT

    def bench(self) -> list:
        return [p for p in self.players if p not in self.lineup and not self.fouled_out(p)]


class GameEngineV2:
    def __init__(self, home_team_id: int, away_team_id: int, season: str,
                 predictor: FactorizedPredictor = None,
                 duration_model: DurationModel = None,
                 sub_model: SubstitutionModel = None,
                 verbose: bool = False, seed: int = None):
        self.predictor = predictor or FactorizedPredictor()
        self.fs = self.predictor.fs
        self.duration = duration_model or DurationModel.load()
        self.subs = sub_model or SubstitutionModel.load()
        self.rng = np.random.default_rng(seed)
        self.predictor.rng = self.rng
        self.verbose = verbose

        self.season = season
        self.home = TeamState(home_team_id, season, self.fs)
        self.away = TeamState(away_team_id, season, self.fs)

        self.period = 1
        self.elapsed_in_period = 0.0
        self.offense = self.home if self.rng.random() < 0.5 else self.away
        self.game_over = False
        self.sub_log = []

    # ------------------------------------------------------------ helpers
    def defense(self) -> TeamState:
        return self.away if self.offense is self.home else self.home

    def period_length(self) -> float:
        return (config.PERIOD_SECONDS if self.period <= config.REGULATION_PERIODS
                else config.OT_SECONDS)

    def sec_remaining(self) -> float:
        return max(self.period_length() - self.elapsed_in_period, 0.0)

    def now(self) -> float:
        from .collect import period_start_elapsed
        return period_start_elapsed(self.period) + self.elapsed_in_period

    def log(self, msg):
        if self.verbose:
            clock = self.sec_remaining()
            print(f"  Q{self.period} {int(clock//60)}:{int(clock%60):02d} "
                  f"{self.home.score}-{self.away.score} | {msg}")

    def name(self, pid) -> str:
        return self.fs.name(pid)

    # ------------------------------------------------------------ state encoding
    def encode_state(self):
        off, deff = self.offense, self.defense()
        now = self.now()
        return self.predictor.encode_state(
            off.lineup, deff.lineup, self.season,
            ctx_kwargs={
                "margin": off.score - deff.score,
                "period": self.period,
                "sec_remaining": self.sec_remaining(),
                "def_fouls_before": deff.fouls_this_period,
                "is_home_offense": int(off is self.home),
            },
            off_stints=[off.stint_sec(p, now) for p in off.lineup],
            off_cums=[off.cum_sec[p] + off.stint_sec(p, now) for p in off.lineup],
            def_stints=[deff.stint_sec(p, now) for p in deff.lineup],
            def_cums=[deff.cum_sec[p] + deff.stint_sec(p, now) for p in deff.lineup],
        )

    # ------------------------------------------------------------ stat updates
    def add(self, team: TeamState, pid: int, stat: str, amt=1):
        team.box[pid][stat] += amt

    def score_points(self, team: TeamState, pid: int, pts: int):
        team.score += pts
        self.add(team, pid, "PTS", pts)

    def charge_defensive_foul(self) -> int:
        """Random defender commits the foul (defender attribution is not in
        the chance table; an explicit who-fouls model is a future upgrade)."""
        deff = self.defense()
        fouler = int(self.rng.choice(deff.lineup))
        self.add(deff, fouler, "PF")
        deff.fouls_this_period += 1
        if deff.fouled_out(fouler):
            self.log(f"{self.name(fouler)} fouls out")
            self.force_replace(deff, fouler)
        return fouler

    def shoot_fts(self, shooter: int, n: int):
        """Returns (made, last_missed)."""
        off = self.offense
        made = 0
        last_missed = False
        for _ in range(n):
            self.add(off, shooter, "FTA")
            if self.predictor.ft_make(shooter, self.season):
                self.add(off, shooter, "FTM")
                self.score_points(off, shooter, 1)
                made += 1
                last_missed = False
            else:
                last_missed = True
        return made, last_missed

    def resolve_rebound(self, state, is_three: bool) -> bool:
        """Sample rebound after a live miss. Returns True if offense keeps ball."""
        offensive = self.predictor.sample_oreb(state, is_three)
        side = self.offense if offensive else self.defense()
        slot = self.predictor.sample_rebounder(state, offensive)
        rebounder = side.lineup[slot]
        self.add(side, rebounder, "REB")
        self.add(side, rebounder, "OREB" if offensive else "DREB")
        self.log(f"{'OFF' if offensive else 'def'} rebound {self.name(rebounder)}")
        return offensive

    # ------------------------------------------------------------ one chance
    def simulate_chance(self):
        off, deff = self.offense, self.defense()
        state = self.encode_state()
        slot = self.predictor.sample_ball_ender(state)
        pid = off.lineup[slot]
        action = self.predictor.sample_action(state, slot)
        points_before = off.score
        possession_flips = True
        deadball = True

        if action in ("2pt_attempt", "3pt_attempt"):
            is_three = action == "3pt_attempt"
            result = self.predictor.sample_shot_result(state, slot, is_three)
            self.add(off, pid, "FGA")
            if is_three:
                self.add(off, pid, "3PA")

            if result in ("make", "andone"):
                self.add(off, pid, "FGM")
                if is_three:
                    self.add(off, pid, "3PM")
                self.score_points(off, pid, 3 if is_three else 2)
                assister = self.predictor.sample_assister(off.lineup, self.season, pid, action)
                if assister is not None:
                    self.add(off, assister, "AST")
                self.log(f"{self.name(pid)} {'3PT' if is_three else '2PT'} make"
                         + (" AND-ONE" if result == "andone" else ""))
                if result == "andone":
                    self.charge_defensive_foul()
                    _, last_missed = self.shoot_fts(pid, 1)
                    if last_missed:
                        deadball = False
                        possession_flips = not self.resolve_rebound(state, False)

            elif result == "shooting_foul":
                self.charge_defensive_foul()
                n = 3 if is_three else 2
                self.log(f"{self.name(pid)} fouled shooting ({n} FTs)")
                _, last_missed = self.shoot_fts(pid, n)
                if last_missed:
                    deadball = False
                    possession_flips = not self.resolve_rebound(state, False)

            else:  # miss
                blocker = self.predictor.sample_blocker(deff.lineup, self.season, action)
                if blocker is not None:
                    self.add(deff, blocker, "BLK")
                self.log(f"{self.name(pid)} {'3PT' if is_three else '2PT'} miss"
                         + (f" (blocked by {self.name(blocker)})" if blocker else ""))
                deadball = False
                possession_flips = not self.resolve_rebound(state, is_three)

        elif action == "turnover":
            self.add(off, pid, "TO")
            stealer = self.predictor.sample_stealer(deff.lineup, self.season)
            if stealer is not None:
                self.add(deff, stealer, "STL")
                deadball = False    # live-ball steal
            self.log(f"{self.name(pid)} turnover"
                     + (f" (steal {self.name(stealer)})" if stealer else ""))

        elif action == "off_foul":
            self.add(off, pid, "TO")
            self.add(off, pid, "PF")
            if off.fouled_out(pid):
                self.force_replace(off, pid)
            self.log(f"{self.name(pid)} offensive foul")

        elif action == "drawn_foul":
            self.charge_defensive_foul()
            in_bonus = deff.fouls_this_period >= config.TEAM_FOULS_FOR_BONUS
            self.log(f"{self.name(pid)} draws foul"
                     + (" (bonus)" if in_bonus else ""))
            if in_bonus:
                _, last_missed = self.shoot_fts(pid, 2)
                if last_missed:
                    deadball = False
                    possession_flips = not self.resolve_rebound(state, False)
            else:
                possession_flips = False   # side-out, offense keeps the ball

        # clock, minutes, plus-minus
        dt = self.duration.sample(action, self.sec_remaining(), self.rng)
        self.elapsed_in_period += dt
        pts = off.score - points_before
        for team, sign in ((off, 1), (deff, -1)):
            for p in team.lineup:
                team.box[p]["MIN"] += dt / 60.0
                team.box[p]["+/-"] += sign * pts

        if possession_flips:
            self.offense = deff
        return deadball

    # ------------------------------------------------------------ substitutions
    def force_replace(self, team: TeamState, pid: int):
        bench = team.bench()
        if not bench:
            return
        self.swap(team, pid, self.pick_replacement(team, bench))

    def pick_replacement(self, team: TeamState, bench: list) -> int:
        """Real-minutes prior x freshness; small noise breaks ties."""
        now = self.now()
        scores = []
        for p in bench:
            rest = min((now - team.exit_time.get(p, 0.0)) / 600.0, 1.0)
            share = team.floor_share.get(p, 0.01)
            scores.append(share * (0.5 + 0.5 * rest) + self.rng.normal(0, 0.005))
        return bench[int(np.argmax(scores))]

    def swap(self, team: TeamState, out_pid: int, in_pid: int):
        now = self.now()
        i = team.lineup.index(out_pid)
        team.cum_sec[out_pid] += team.stint_sec(out_pid, now)
        team.entry_time[out_pid] = None
        team.exit_time[out_pid] = now
        team.lineup[i] = in_pid
        team.entry_time[in_pid] = now
        self.sub_log.append({"period": self.period, "elapsed": now,
                             "team": team.team_id, "out": out_pid, "in": in_pid})

    def run_substitutions(self):
        """At a deadball: learned exit policy for everyone on the floor."""
        now = self.now()
        margin_abs = abs(self.home.score - self.away.score)
        for team in (self.home, self.away):
            rows, pids = [], []
            for pid in team.lineup:
                rows.append([
                    team.stint_sec(pid, now),
                    team.cum_sec[pid] + team.stint_sec(pid, now),
                    self.period, self.sec_remaining(), margin_abs,
                    team.box[pid]["PF"], int(pid in team.starters),
                ])
                pids.append(pid)
            probs = self.subs.exit_probabilities(np.array(rows, dtype=np.float64))
            for pid, p in zip(pids, probs):
                if self.rng.random() < p:
                    bench = team.bench()
                    if bench:
                        self.swap(team, pid, self.pick_replacement(team, bench))

    # ------------------------------------------------------------ game loop
    def end_period(self):
        # close out stints at the period boundary so MIN bookkeeping is exact
        self.log(f"end of period {self.period}")
        if self.period >= config.REGULATION_PERIODS and self.home.score != self.away.score:
            self.game_over = True
            return
        self.period += 1
        self.elapsed_in_period = 0.0
        self.home.fouls_this_period = 0
        self.away.fouls_this_period = 0
        self.run_substitutions()

    def simulate_game(self) -> dict:
        while not self.game_over:
            deadball = self.simulate_chance()
            if self.elapsed_in_period >= self.period_length() - 1e-9:
                self.end_period()
                continue
            if deadball:
                self.run_substitutions()
            if self.period > config.REGULATION_PERIODS + 10:   # safety valve
                break
        # flush open stints into cumulative seconds
        now = self.now()
        for team in (self.home, self.away):
            for pid in list(team.lineup):
                team.cum_sec[pid] += team.stint_sec(pid, now)
                team.entry_time[pid] = None
        return {
            "home_team": self.home.team_id, "away_team": self.away.team_id,
            "home_score": self.home.score, "away_score": self.away.score,
            "periods": self.period,
            "home_box": self.boxscore(self.home),
            "away_box": self.boxscore(self.away),
        }

    def boxscore(self, team: TeamState) -> pd.DataFrame:
        rows = []
        for pid, stats in team.box.items():
            row = {"PLAYER_ID": pid, "PLAYER": self.name(pid)}
            row.update(stats)
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("PTS", ascending=False)
        for col in ("FGA", "FGM", "3PA", "3PM", "FTA", "FTM", "PTS", "REB",
                    "OREB", "DREB", "AST", "STL", "BLK", "TO", "PF"):
            df[col] = df[col].astype(int)
        df["MIN"] = df["MIN"].round(1)
        df["FG%"] = np.where(df["FGA"] > 0, (df["FGM"] / df["FGA"].clip(lower=1) * 100), 0).round(1)
        return df.reset_index(drop=True)
