"""
NBA Game Simulation Engine
Main classes for simulating NBA games possession-by-possession.
"""

"""
TODO:
- model substitutions using ML instead of hardcoded rules
- include auxiliary stats like defensive rebound, offensive rebound, etc.
"""

# ====================================================================
# IMPORTS
# ====================================================================

from nba_api.stats.endpoints import commonteamroster, playergamelog
from nba_api.stats.static import teams, players
from player_aware_predictor import PlayerAwarePredictor
from train_possession_model import FeatureEngineer  # Needed for unpickling
import numpy as np
import pandas as pd


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

# Load dataframe of NBA teams
nba_teams = pd.DataFrame(teams.get_teams())

# Boxscore column definitions
BOXCOLS = ["MIN", "PTS", "REB", "AST", "FGA", "FGM", "3PA", "3PM", "FTA", "FTM", "STL", "BLK", "TO", "PF", "+/-"]


def init_boxscore_from_roster(roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize empty boxscore from team roster.
    
    Args:
        roster_df: Team roster DataFrame
    
    Returns:
        DataFrame with player stats initialized to 0
    """
    box = roster_df[["PLAYER_ID", "PLAYER", "POSITION", "NUM"]].copy()
    for c in BOXCOLS:
        if c == "MIN":
            box[c] = 0.0  # Minutes stored as float for precision
        else:
            box[c] = 0    # Other stats as integers
    return box.set_index("PLAYER_ID")


def _mins_to_float(v):
    """Convert minutes from 'MM:SS' format to float."""
    if pd.isna(v):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    if ":" in s:
        m, sec = s.split(":")
        return float(m) + float(sec) / 60.0
    return float(s)


def team_resolver(id=None, abbr=None, name=None) -> dict:
    """
    Resolve team information from ID, abbreviation, or name.
    
    Args:
        id: NBA team ID (int)
        abbr: Team abbreviation (str)
        name: Team name (str)
    
    Returns:
        Dict with team_id, team_abbr, team_name, location
    """
    # Check that exactly one parameter is provided
    if sum(x is not None for x in (id, abbr, name)) != 1:
        raise ValueError("Provide exactly one of team_id, abbr, or name.")
    
    # Resolve by ID
    if id is not None:
        team_name = nba_teams[nba_teams['id'] == id]['nickname'].iloc[0]
        team_abbr = nba_teams[nba_teams['id'] == id]['abbreviation'].iloc[0]
        team_id = id
    
    # Resolve by name
    elif name is not None:
        team_id = nba_teams[nba_teams['nickname'] == name]['id'].iloc[0]
        team_abbr = nba_teams[nba_teams['id'] == team_id]['abbreviation'].iloc[0]
        team_name = name
    
    # Resolve by abbreviation
    else:
        team_id = nba_teams[nba_teams['abbreviation'] == abbr]['id'].iloc[0]
        team_name = nba_teams[nba_teams['id'] == team_id]['nickname'].iloc[0]
        team_abbr = abbr
    
    location = nba_teams[nba_teams['id'] == team_id]['city'].iloc[0]
    
    return {
        'team_id': team_id,
        'team_abbr': team_abbr,
        'team_name': team_name,
        'location': location
    }


# ====================================================================
# CLASSES
# ====================================================================

class Team:
    """
    NBA Team class.
    Manages team identity, roster, lineup, and game state.
    """
    
    def __init__(self, season: str, team_id: int = None, team_abbr: str = None, team_name: str = None):
        """
        Initialize Team.
        
        Args:
            season: Season string (e.g., '2023-24')
            team_id: Team ID (integer) OR
            team_abbr: Team abbreviation (string) OR
            team_name: Team name (string)
        """
        # Resolve team information
        team_info = team_resolver(team_id, team_abbr, team_name)
        
        # Team identity
        self.team_id = int(team_info['team_id'])
        self.team_name = str(team_info['team_name'])
        self.team_abbr = str(team_info['team_abbr'])
        self.location = str(team_info['location'])
        self.season = str(season)
        
        # Get roster from NBA API
        endpoint = commonteamroster.CommonTeamRoster(team_id=self.team_id, season=self.season)
        self.roster_df = endpoint.get_data_frames()[0]
        
        # Game state (updated during game)
        self.starters = self.pick_starters(self.roster_df, season)
        self.lineup = self.starters.copy()  # Current 5 players on court
        self.score = 0
        self.fouls = 0
        self.timeouts = 7
        self.boxscore = init_boxscore_from_roster(self.roster_df)
    
    def pick_starters(self, roster_df: pd.DataFrame, season: str) -> list:
        """
        Pick the 5 starting players based on games started and average minutes.
        
        Returns:
            list: List of 5 player IDs (integers)
        """
        records = []
        for pid in roster_df["PLAYER_ID"].tolist():
            try:
                p = Player(player_id=int(pid), season=season)
                df = p.gamelog
                if df.empty:
                    continue
                starts = int(df.get("GS", pd.Series([0] * len(df))).fillna(0).astype(int).sum())
                avg_min = float(pd.Series(df.get("MIN", [])).apply(_mins_to_float).mean()) if "MIN" in df.columns else 0.0
                records.append((int(pid), starts, avg_min))
            except Exception:
                pass  # Skip players with no logs
        
        # Sort by games started, then average minutes
        records.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [pid for pid, _, _ in records[:5]]
    
    def __str__(self) -> str:
        header = f"{self.season} {self.location} {self.team_name} Summary:"
        meta = f"{self.team_abbr} | Team ID: {self.team_id}"
        roster_preview = self.roster_df[['PLAYER_ID', 'PLAYER', 'POSITION', 'NUM']].to_string(index=False)
        return f"{header}\n\n{meta}\n\nRoster:\n{roster_preview}"


class Player:
    """
    NBA Player class.
    Contains player identity and historical game log data.
    """
    
    def __init__(self, player_id: int, season: str):
        player_info = players.find_player_by_id(player_id)
        
        self.id = player_id
        self.name = player_info['full_name']
        self.gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]


class GameEngine:
    """
    NBA Game Simulation Engine.
    Coordinates game flow, manages fatigue, handles substitutions, and simulates possessions.
    """
    
    def __init__(self, hometeam_id: int, awayteam_id: int, season: str, predictor: PlayerAwarePredictor = None):
        """
        Initialize game between two teams.
        
        Args:
            hometeam_id: Home team NBA ID
            awayteam_id: Away team NBA ID
            season: Season string (e.g., '2023-24')
            predictor: Optional pre-initialized predictor
        """
        # Teams
        self.season = season
        self.hometeam = Team(team_id=hometeam_id, season=season)
        self.awayteam = Team(team_id=awayteam_id, season=season)
        
        # ML Predictor (shared resource)
        self.predictor = predictor if predictor else PlayerAwarePredictor()
        
        # ====== Game State ======
        self.quarter = 1
        self.poss = np.random.choice([1, -1])  # 1 = home, -1 = away
        self.play = 1
        self.quarter_duration = 12 * 60  # seconds
        self.ot_duration = 5 * 60
        self.deadball = False
        
        # ====== Energy System ======
        all_players = (self.hometeam.roster_df['PLAYER_ID'].tolist() + 
                      self.awayteam.roster_df['PLAYER_ID'].tolist())
        self.player_energy = {pid: 100.0 for pid in all_players}
        self.player_minutes = {pid: 0.0 for pid in all_players}
        
        # ====== Logs ======
        self.substitution_log = []
    
    # ================================================================
    # FATIGUE & ENERGY MANAGEMENT
    # ================================================================
    
    def get_fatigue_multiplier(self, player_id: int) -> float:
        """
        Calculate performance multiplier based on player's energy (0.5 to 1.0).
        """
        energy = self.player_energy.get(player_id, 100.0)
        
        if energy > 80:
            return 1.0
        elif energy > 50:
            return 0.85 + (energy - 50) * 0.005
        elif energy > 20:
            return 0.70 + (energy - 20) * 0.005
        else:
            return 0.50 + energy * 0.01
    
    def deplete_energy(self, player_id: int, base_cost: float, intensity_bonus: float = 0.0):
        """Deplete player's energy based on activity level."""
        total_cost = base_cost + intensity_bonus
        self.player_energy[player_id] -= total_cost
        self.player_energy[player_id] = max(0, self.player_energy[player_id])
    
    def rest_player(self, player_id: int):
        """Regenerate energy for bench player (+2.0 per possession)."""
        recovery_rate = 2.0
        self.player_energy[player_id] = min(100, self.player_energy[player_id] + recovery_rate)
    
    # ================================================================
    # SUBSTITUTION MANAGEMENT
    # ================================================================
    
    def substitute_player(self, team: str, position: int, new_player_id: int) -> bool:
        """
        Substitute a player in the lineup.
        
        Args:
            team: 'home' or 'away'
            position: Index in lineup (0-4)
            new_player_id: Player ID coming into the game
        
        Returns:
            bool: True if substitution successful
        """
        # Validate inputs
        if position < 0 or position > 4 or new_player_id is None:
            return False
        
        # Get team references
        team_obj = self.hometeam if team == 'home' else self.awayteam
        
        # Validate new player
        if new_player_id not in team_obj.roster_df['PLAYER_ID'].values:
            return False
        if new_player_id in team_obj.lineup:
            return False
        
        # Execute substitution
        old_player_id = team_obj.lineup[position]
        team_obj.lineup[position] = new_player_id
        
        # Log substitution
        old_name = team_obj.roster_df[team_obj.roster_df['PLAYER_ID'] == old_player_id]['PLAYER'].values[0] if old_player_id in team_obj.roster_df['PLAYER_ID'].values else f"Player {old_player_id}"
        new_name = team_obj.roster_df[team_obj.roster_df['PLAYER_ID'] == new_player_id]['PLAYER'].values[0]
        
        self.substitution_log.append({
            'quarter': self.quarter,
            'time_remaining': self.quarter_duration,
            'team': team,
            'player_out': old_player_id,
            'player_out_name': old_name,
            'player_in': new_player_id,
            'player_in_name': new_name,
            'energy_out': self.player_energy[old_player_id],
            'energy_in': self.player_energy[new_player_id]
        })
        
        return True
    
    def get_player_quality(self, player_id: int) -> float:
        """Calculate player quality score (usage + efficiency)."""
        key = (int(player_id), self.season)
        
        if key not in self.predictor.fe.player_season_stats:
            return 0.1
        
        stats = self.predictor.fe.player_season_stats[key]
        quality = (
            stats['usage_rate'] * 0.5 +
            stats['2pt_pct'] * 0.25 +
            stats['3pt_pct'] * 0.25
        )
        
        return quality
    
    def find_best_bench_player(self, team: str, energy_weight: float = 0.6, quality_weight: float = 0.4) -> int:
        """
        Find best bench player balancing freshness (60%) and quality (40%).
        
        Returns:
            int: Player ID of best available bench player
        """
        team_obj = self.hometeam if team == 'home' else self.awayteam
        
        # Get bench players
        bench_ids = [pid for pid in team_obj.roster_df['PLAYER_ID'].tolist() 
                    if pid not in team_obj.lineup]
        
        if not bench_ids:
            return None
        
        # Score each bench player
        scores = []
        for player_id in bench_ids:
            energy_score = self.player_energy[player_id] / 100.0
            quality_score = self.get_player_quality(player_id)
            quality_normalized = min(1.0, quality_score / 0.4)
            
            combined_score = (energy_score * energy_weight) + (quality_normalized * quality_weight)
            scores.append((player_id, combined_score))
        
        # Return player with highest combined score
        best = max(scores, key=lambda x: x[1])
        return int(best[0])
    
    # ================================================================
    # GAME STATE & DECISION LOGIC
    # ================================================================
    
    def is_close_game(self) -> bool:
        """Check if game is competitive (close games → keep stars in)."""
        margin = abs(self.hometeam.score - self.awayteam.score)
        
        if self.quarter == 4 and self.quarter_duration < 360 and margin <= 5:
            return True
        if self.quarter == 4 and self.quarter_duration < 120 and margin <= 8:
            return True
        
        return False
    
    def is_blowout(self) -> bool:
        """Check if game is decided (blowouts → rest starters)."""
        margin = abs(self.hometeam.score - self.awayteam.score)
        
        if self.quarter >= 3 and margin >= 20:
            return True
        if self.quarter == 4 and self.quarter_duration < 180 and margin >= 15:
            return True
        
        return False
    
    def check_substitutions(self):
        """
        Hybrid substitution strategy:
        - Time-based (6-minute mark, between quarters)
        - Fatigue-based (emergency subs)
        - Game situation aware (close vs blowout)
        """
        if not self.deadball:
            return
        
        # Between quarters - bring back starters
        if self.quarter_duration <= 0:
            self.hometeam.lineup = self.hometeam.starters.copy()
            self.awayteam.lineup = self.awayteam.starters.copy()
            return
        
        # Make substitutions for both teams
        close_game = self.is_close_game()
        blowout = self.is_blowout()
        
        self._make_team_substitutions('home', close_game, blowout)
        self._make_team_substitutions('away', close_game, blowout)
    
    def _make_team_substitutions(self, team: str, close_game: bool, blowout: bool):
        """Execute substitutions for one team based on strategy."""
        team_obj = self.hometeam if team == 'home' else self.awayteam
        
        for i, player_id in enumerate(team_obj.lineup):
            energy = self.player_energy[player_id]
            fouls = team_obj.boxscore.loc[player_id, 'PF']
            
            should_sub = False
            reason = ""
            
            # Critical situations (always sub)
            if fouls >= 5:
                should_sub = True
                reason = "foul_trouble"
            elif energy < 20:
                should_sub = True
                reason = "exhausted"
            
            # Blowout (rest starters)
            elif blowout:
                if player_id in team_obj.starters:
                    should_sub = True
                    reason = "blowout_rest"
            
            # Regular fatigue (skip if close game)
            elif not close_game:
                if energy < 40:
                    should_sub = True
                    reason = "very_tired"
                elif 5.5*60 < self.quarter_duration < 6.5*60 and energy < 70:
                    should_sub = True
                    reason = "rotation"
            
            # Execute substitution if needed
            if should_sub:
                bench_player = self.find_best_bench_player(team)
                if bench_player and (self.player_energy[bench_player] > 60 or reason in ['foul_trouble', 'exhausted']):
                    self.substitute_player(team, i, bench_player)
    
    def call_timeout(self, team: str) -> bool:
        """Call timeout - forces deadball and allows substitutions."""
        team_obj = self.hometeam if team == 'home' else self.awayteam
        
        if team_obj.timeouts <= 0:
            return False
        
        team_obj.timeouts -= 1
        self.deadball = True
        
        # Small energy recovery during timeout
        for player_id in self.player_energy.keys():
            self.player_energy[player_id] = min(100, self.player_energy[player_id] + 2.0)
        
        self.check_substitutions()
        return True
    
    def should_call_timeout(self, team: str) -> bool:
        """Decide if team should call timeout (for forced substitutions)."""
        team_obj = self.hometeam if team == 'home' else self.awayteam
        
        if team_obj.timeouts <= 0 or self.is_blowout():
            return False
        
        # Save last timeout for Q4 final minute
        if self.quarter == 4 and self.quarter_duration < 60 and team_obj.timeouts <= 1:
            return False
        
        # Call timeout if team exhausted
        avg_energy = sum(self.player_energy[pid] for pid in team_obj.lineup) / 5
        if avg_energy < 45 and not self.deadball:
            return True
        
        # Call if multiple players need rest
        tired_count = sum(1 for pid in team_obj.lineup if self.player_energy[pid] < 40)
        if tired_count >= 3 and not self.deadball:
            return True
        
        return False
    
    # ================================================================
    # MAIN SIMULATION LOOP
    # ================================================================

    def get_player_ft_pct(self, player_id: int) -> float:
        """Get player's free throw percentage from stats."""
        key = (int(player_id), self.season)
        
        # Try to get from feature engineer stats
        if key in self.predictor.fe.player_season_stats:
            stats = self.predictor.fe.player_season_stats[key]
            # Estimate FT% as weighted average of 2PT% and 3PT%
            # In reality, you'd want actual FT% from NBA API
            # For now, use a reasonable estimate
            ft_pct = (stats.get('2pt_pct', 0.5) * 0.6 + 0.35) / 1.0
            return min(0.95, max(0.50, ft_pct))  # Clamp between 50-95%
        
        # Default league average
        return 0.75
    
    def shooting_foul(self, outcome: str, player_id: int, offensive_team, defensive_team) -> dict:
        """
        Handle shooting foul outcome with free throw simulation.
        
        Args:
            outcome: One of '2pt_foul', '3pt_foul', '2pt_andone', '3pt_andone'
            player_id: ID of player shooting free throws
            offensive_team: Team object
            defensive_team: Team object
        
        Returns:
            dict with 'points', 'fta', 'ftm', and play-by-play description
        """
        # Determine number of free throws
        if outcome == '2pt_foul':
            num_fts = 2
        elif outcome == '3pt_foul':
            num_fts = 3
        elif outcome in ['2pt_andone', '3pt_andone']:
            num_fts = 1
        else:
            return {'points': 0, 'fta': 0, 'ftm': 0, 'description': ''}
        
        # Get player info
        player_name = offensive_team.roster_df[offensive_team.roster_df['PLAYER_ID'] == player_id]['PLAYER'].values[0]
        ft_pct = self.get_player_ft_pct(player_id)
        
        # Simulate free throws
        fts_made = 0
        ft_descriptions = []
        for i in range(num_fts):
            ft_result = np.random.binomial(n=1, p=ft_pct)
            fts_made += ft_result
            
            if ft_result == 1:
                ft_descriptions.append(f"  FT {i+1}/{num_fts}: GOOD")
            else:
                ft_descriptions.append(f"  FT {i+1}/{num_fts}: MISS")
        
        description = f"{player_name} shoots {num_fts} FTs ({fts_made}/{num_fts} made)\n" + "\n".join(ft_descriptions)
        
        return {
            'points': fts_made,
            'fta': num_fts,
            'ftm': fts_made,
            'description': description
        }
    
    def outcome_ends_possession(self, outcome: str) -> str:
        """
        Determine if outcome ends possession.
        
        Returns:
            'yes' - possession definitely changes
            'miss' - possession changes only if defensive rebound
            'no' - possession continues (offensive keeps ball)
        """
        # Made baskets always end possession
        if outcome in ['2pt_make', '3pt_make', '2pt_andone', '3pt_andone']:
            return 'yes'
        
        # Turnovers and offensive fouls end possession
        if outcome in ['TO', 'off_foul']:
            return 'yes'
        
        # Shooting fouls end possession (after free throws are shot)
        if outcome in ['2pt_foul', '3pt_foul']:
            return 'yes'
        
        # Misses might end possession (depends on rebound)
        if outcome in ['2pt_miss', '3pt_miss']:
            return 'miss'
        
        # Non-shooting defensive fouls keep possession
        if outcome == 'def_foul':
            return 'no'
        
        return 'yes'  # Default: end possession
    
    def get_defensive_rebound_probability(self, outcome: str) -> float:
        """Get probability of defensive rebound after a miss."""
        # 3PT misses have higher defensive rebound rate (longer rebounds)
        if outcome == '3pt_miss':
            return 0.75  # ~75% defensive rebound rate
        elif outcome == '2pt_miss':
            return 0.68  # ~68% defensive rebound rate
        return 0.70  # Default
    
    def step_possession(self) -> dict:
        """
        Simulate ONE offensive outcome (e.g., one shot attempt, one turnover).
        
        Returns:
            dict with outcome details including:
                - outcome: str (one of 11 categories)
                - player_id: int
                - player_name: str
                - points: int
                - possession_ends: bool
                - stats: dict (fga, fgm, 3pa, 3pm, fta, ftm, to, pf)
        """
        # Determine offensive/defensive teams
        if self.poss == 1:
            offensive_team = self.hometeam
            defensive_team = self.awayteam
        else:
            offensive_team = self.awayteam
            defensive_team = self.hometeam
        
        # Get fatigue multipliers
        fatigue_multipliers = {
            pid: self.get_fatigue_multiplier(pid)
            for pid in offensive_team.lineup + defensive_team.lineup
        }
        
        # Predict possession outcome
        result = self.predictor.sample_possession_detailed(
            offensive_lineup=offensive_team.lineup,
            defensive_lineup=defensive_team.lineup,
            season=self.season,
            score_margin=offensive_team.score - defensive_team.score,
            period=self.quarter,
            fatigue_multipliers=fatigue_multipliers
        )
        
        outcome = result['outcome']
        player_id = result['player_id']
        player_name = offensive_team.roster_df[offensive_team.roster_df['PLAYER_ID'] == player_id]['PLAYER'].values[0]
        
        # Initialize stats tracking
        points = 0
        fga = 0
        fgm = 0
        three_pa = 0
        three_pm = 0
        fta = 0
        ftm = 0
        to = 0
        pf = 0
        
        # Process outcome and update stats
        if outcome == '2pt_make':
            points = 2
            fga = 1
            fgm = 1
            description = f"{player_name} makes 2PT shot!"
            
        elif outcome == '3pt_make':
            points = 3
            fga = 1
            fgm = 1
            three_pa = 1
            three_pm = 1
            description = f"{player_name} makes 3PT shot!"
            
        elif outcome == '2pt_miss':
            fga = 1
            description = f"{player_name} misses 2PT shot"
            
        elif outcome == '3pt_miss':
            fga = 1
            three_pa = 1
            description = f"{player_name} misses 3PT shot"
            
        elif outcome in ['2pt_andone', '3pt_andone']:
            # Made basket + foul
            if outcome == '2pt_andone':
                points = 2
                fga = 1
                fgm = 1
                description = f"{player_name} makes 2PT AND-ONE!"
            else:
                points = 3
                fga = 1
                fgm = 1
                three_pa = 1
                three_pm = 1
                description = f"{player_name} makes 3PT AND-ONE!"
            
            # Shoot free throw
            ft_result = self.shooting_foul(outcome, player_id, offensive_team, defensive_team)
            points += ft_result['points']
            fta = ft_result['fta']
            ftm = ft_result['ftm']
            description += f"\n{ft_result['description']}"
            
        elif outcome in ['2pt_foul', '3pt_foul']:
            # Shooting foul - no basket made
            if outcome == '2pt_foul':
                fga = 1  # Attempted shot
                description = f"{player_name} fouled on 2PT attempt"
            else:
                fga = 1
                three_pa = 1
                description = f"{player_name} fouled on 3PT attempt"
            
            # Shoot free throws
            ft_result = self.shooting_foul(outcome, player_id, offensive_team, defensive_team)
            points = ft_result['points']
            fta = ft_result['fta']
            ftm = ft_result['ftm']
            description += f"\n{ft_result['description']}"
            
        elif outcome == 'TO':
            to = 1
            description = f"{player_name} TURNOVER"
            
        elif outcome == 'off_foul':
            to = 1  # Offensive foul counts as turnover
            pf = 1
            description = f"{player_name} OFFENSIVE FOUL (turnover)"
            
        elif outcome == 'def_foul':
            # Defensive foul (non-shooting) - possession continues
            description = f"Defensive foul on {defensive_team.team_abbr}"
        
        # Determine if possession ends
        poss_end_type = self.outcome_ends_possession(outcome)
        
        if poss_end_type == 'yes':
            possession_ends = True
        elif poss_end_type == 'miss':
            # Check for defensive rebound
            def_reb_prob = self.get_defensive_rebound_probability(outcome)
            is_def_rebound = np.random.random() < def_reb_prob
            possession_ends = is_def_rebound
            
            if is_def_rebound:
                description += " → Defensive rebound"
            else:
                description += " → OFFENSIVE REBOUND!"
        else:
            possession_ends = False
        
        return {
            'outcome': outcome,
            'player_id': player_id,
            'player_name': player_name,
            'points': points,
            'possession_ends': possession_ends,
            'description': description,
            'stats': {
                'fga': fga,
                'fgm': fgm,
                '3pa': three_pa,
                '3pm': three_pm,
                'fta': fta,
                'ftm': ftm,
                'to': to,
                'pf': pf
            }
        }
    
    def update_boxscore(self, result: dict, offensive_team, defensive_team):
        """Update boxscore with result from step_possession."""
        player_id = result['player_id']
        stats = result['stats']
        
        # Update offensive player stats
        offensive_team.boxscore.loc[player_id, 'PTS'] += result['points']
        offensive_team.boxscore.loc[player_id, 'FGA'] += stats['fga']
        offensive_team.boxscore.loc[player_id, 'FGM'] += stats['fgm']
        offensive_team.boxscore.loc[player_id, '3PA'] += stats['3pa']
        offensive_team.boxscore.loc[player_id, '3PM'] += stats['3pm']
        offensive_team.boxscore.loc[player_id, 'FTA'] += stats['fta']
        offensive_team.boxscore.loc[player_id, 'FTM'] += stats['ftm']
        offensive_team.boxscore.loc[player_id, 'TO'] += stats['to']
        offensive_team.boxscore.loc[player_id, 'PF'] += stats['pf']
    

    def step(self):
        """
        Simulate one FULL possession (can include multiple outcomes).
        A possession continues until it changes (e.g., miss → offensive rebound → make).
        """
        self.play += 1
        
        # Determine offensive/defensive teams
        if self.poss == 1:
            offensive_team = self.hometeam
            defensive_team = self.awayteam
        else:
            offensive_team = self.awayteam
            defensive_team = self.hometeam
        
        total_possession_points = 0
        possession_ended = False
        possession_outcomes = []  # Track all outcomes in this possession
        
        # Loop until possession ends
        while not possession_ended:
            # Simulate one outcome (shot attempt, turnover, etc.)
            result = self.step_possession()
            possession_outcomes.append(result)
            
            # Print play-by-play
            print(f"  {result['description']}")
            
            # Update boxscore
            self.update_boxscore(result, offensive_team, defensive_team)
            
            # Track points for plus/minus
            total_possession_points += result['points']
            
            # Update team score
            offensive_team.score += result['points']
            
            # Check if possession ends
            if result['possession_ends']:
                possession_ended = True
                self.poss *= -1  # Switch possession
        
        # Update plus/minus for all players on court (for the full possession)
        for pid in offensive_team.lineup:
            offensive_team.boxscore.loc[pid, '+/-'] += total_possession_points
        for pid in defensive_team.lineup:
            defensive_team.boxscore.loc[pid, '+/-'] -= total_possession_points
        
        # Calculate possession time and update clock/minutes
        # Base time + extra time for offensive rebounds/fouls
        num_outcomes = len(possession_outcomes)
        base_time = np.random.binomial(n=24, p=0.7)  # seconds
        extra_time = max(0, (num_outcomes - 1) * 3)  # +3 seconds per extra outcome
        total_time_seconds = min(base_time + extra_time, 30)  # Cap at 30 seconds
        possession_time_minutes = total_time_seconds / 60.0
        
        self.quarter_duration -= total_time_seconds
        
        # Update minutes for all players on court
        for player_id in offensive_team.lineup + defensive_team.lineup:
            self.player_minutes[player_id] += possession_time_minutes
            if player_id in offensive_team.boxscore.index:
                offensive_team.boxscore.loc[player_id, 'MIN'] += possession_time_minutes
            if player_id in defensive_team.boxscore.index:
                defensive_team.boxscore.loc[player_id, 'MIN'] += possession_time_minutes
        
        # Energy depletion for players on court
        for player_id in offensive_team.lineup + defensive_team.lineup:
            base_cost = 1.0
            intensity_bonus = 0.0
            
            # Extra energy cost for players who were involved
            for outcome in possession_outcomes:
                if player_id == outcome['player_id']:
                    intensity_bonus += 0.5
                    if outcome['outcome'] in ['2pt_make', '2pt_miss', '2pt_andone']:
                        intensity_bonus += 0.3
                    elif outcome['stats']['to'] > 0:
                        intensity_bonus += 0.2
            
            self.deplete_energy(player_id, base_cost, intensity_bonus)
        
        # Energy recovery for bench players
        all_players = list(self.player_energy.keys())
        on_court = offensive_team.lineup + defensive_team.lineup
        bench_players = [pid for pid in all_players if pid not in on_court]
        
        for player_id in bench_players:
            self.rest_player(player_id)
        
        # Check if timeouts needed
        if self.should_call_timeout('home'):
            self.call_timeout('home')
        if self.should_call_timeout('away'):
            self.call_timeout('away')
        
        # Make substitutions if deadball
        self.check_substitutions()
    
    # ================================================================
    # GAME UTILITIES & DISPLAY
    # ================================================================
    
    def simulate_game(self) -> None:
        """Simulate a complete 4-quarter game."""
        while self.quarter <= 4:
            self.step()
            if self.quarter_duration <= 0:
                self.quarter += 1
                self.quarter_duration = 12 * 60
    
    def get_boxscore_display(self, team: str) -> pd.DataFrame:
        """
        Get boxscore with minutes rounded for display.
        
        Args:
            team: 'home' or 'away'
        
        Returns:
            DataFrame with integer minutes (original unchanged)
        """
        boxscore = self.hometeam.boxscore if team == 'home' else self.awayteam.boxscore
        display_box = boxscore.copy()
        display_box['MIN'] = display_box['MIN'].round(0).astype(int)
        return display_box
    
    def restart_game(self):
        """Restart game with same teams."""
        self.__init__(self.hometeam.team_id, self.awayteam.team_id, self.season)
    
    def __str__(self) -> str:
        return f"Q{self.quarter} - {self.hometeam.team_abbr} {self.hometeam.score} vs {self.awayteam.team_abbr} {self.awayteam.score}"

