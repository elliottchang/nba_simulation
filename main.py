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

from nba_api.stats.endpoints import commonteamroster, playergamelog, leaguegamefinder
from nba_api.stats.static import teams, players
from player_aware_predictor import PlayerAwarePredictor
from train_possession_model import FeatureEngineer  # Needed for unpickling
import numpy as np
import pandas as pd
import random
import time
from typing import Dict, List, Tuple


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def retry_api_call(func, max_retries: int = 3, base_delay: float = 2.0, timeout: int = 60):
    """
    Retry an API call with exponential backoff.
    
    Args:
        func: Function to call (should be a lambda or callable with no args)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be doubled for each retry)
        timeout: Timeout for the API call in seconds
    
    Returns:
        Result of the function call, or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt
            
            delay = base_delay * (2 ** attempt)
            print(f"  API retry {attempt + 1}/{max_retries}: {str(e)[:100]}...")
            time.sleep(delay)
    
    return None

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
        
        # Get roster from NBA API with retry logic
        try:
            self.roster_df = retry_api_call(
                lambda: commonteamroster.CommonTeamRoster(team_id=self.team_id, season=self.season).get_data_frames()[0],
                max_retries=3,
                base_delay=2.0
            )
        except Exception as e:
            print(f"⚠️  Failed to get roster for team {self.team_id} after retries: {e}")
            # Create a minimal roster to prevent crashes
            self.roster_df = pd.DataFrame({
                'PLAYER_ID': [999999 + i for i in range(5)],
                'PLAYER': [f'Player_{i+1}' for i in range(5)],
                'POSITION': ['G', 'G', 'F', 'F', 'C'],
                'NUM': [i+1 for i in range(5)]
            })
        
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
                    # Add player with default stats if no gamelog
                    records.append((int(pid), 0, 0.0))
                    continue
                starts = int(df.get("GS", pd.Series([0] * len(df))).fillna(0).astype(int).sum())
                avg_min = float(pd.Series(df.get("MIN", [])).apply(_mins_to_float).mean()) if "MIN" in df.columns else 0.0
                records.append((int(pid), starts, avg_min))
            except Exception:
                # Add player with default stats if API fails
                records.append((int(pid), 0, 0.0))
        
        # Sort by games started, then average minutes
        records.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Ensure we return exactly 5 starters, pad with available players if needed
        selected_starters = [pid for pid, _, _ in records[:5]]
        
        # If we don't have 5 players, fill with remaining roster players
        if len(selected_starters) < 5:
            available_pids = [pid for pid in roster_df["PLAYER_ID"].tolist() if pid not in selected_starters]
            selected_starters.extend(available_pids[:5 - len(selected_starters)])
        
        return selected_starters[:5]  # Ensure we return exactly 5
    
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
        try:
            player_info = players.find_player_by_id(player_id)
            self.name = player_info['full_name']
        except Exception:
            self.name = f"Player_{player_id}"
        
        self.id = player_id
        
        # Get player gamelog with retry logic
        try:
            self.gamelog = retry_api_call(
                lambda: playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    season_type_all_star='Regular Season'
                ).get_data_frames()[0],
                max_retries=2,  # Fewer retries for individual players
                base_delay=1.0
            )
            if self.gamelog is None:
                self.gamelog = pd.DataFrame()  # Empty dataframe if API fails
        except Exception as e:
            # If all retries fail, create empty gamelog to prevent crashes
            self.gamelog = pd.DataFrame()


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
        
        # Lineup resets are now handled in the quarter transition logic
        # to ensure proper timing relative to game end conditions
        
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
    
    def select_rebounder(self, team_lineup: list, team_roster_df: pd.DataFrame) -> int:
        """Select a random player from the lineup to get the rebound."""
        return np.random.choice(team_lineup)
    
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
        description = ""  # Initialize description variable
        
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
        
        else:
            # Fallback for unexpected outcomes
            description = f"{player_name} {outcome}"
        
        # Determine if possession ends
        poss_end_type = self.outcome_ends_possession(outcome)
        
        if poss_end_type == 'yes':
            possession_ends = True
            rebound_info = None
        elif poss_end_type == 'miss':
            # Check for defensive rebound
            def_reb_prob = self.get_defensive_rebound_probability(outcome)
            is_def_rebound = np.random.random() < def_reb_prob
            possession_ends = is_def_rebound
            
            # Select rebounder and create rebound info
            if is_def_rebound:
                rebounder_id = self.select_rebounder(defensive_team.lineup, defensive_team.roster_df)
                rebounder_name = defensive_team.roster_df[defensive_team.roster_df['PLAYER_ID'] == rebounder_id]['PLAYER'].values[0]
                description += f" → Defensive rebound by {rebounder_name}"
                rebound_info = {
                    'rebounder_id': rebounder_id,
                    'rebounder_team': defensive_team,
                    'is_offensive': False
                }
            else:
                rebounder_id = self.select_rebounder(offensive_team.lineup, offensive_team.roster_df)
                rebounder_name = offensive_team.roster_df[offensive_team.roster_df['PLAYER_ID'] == rebounder_id]['PLAYER'].values[0]
                description += f" → OFFENSIVE REBOUND by {rebounder_name}!"
                rebound_info = {
                    'rebounder_id': rebounder_id,
                    'rebounder_team': offensive_team,
                    'is_offensive': True
                }
        else:
            possession_ends = False
            rebound_info = None
        
        return {
            'outcome': outcome,
            'player_id': player_id,
            'player_name': player_name,
            'points': points,
            'possession_ends': possession_ends,
            'description': description,
            'rebound_info': rebound_info,
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
        if player_id in offensive_team.boxscore.index:
            offensive_team.boxscore.loc[player_id, 'PTS'] += result['points']
            offensive_team.boxscore.loc[player_id, 'FGA'] += stats['fga']
            offensive_team.boxscore.loc[player_id, 'FGM'] += stats['fgm']
            offensive_team.boxscore.loc[player_id, '3PA'] += stats['3pa']
            offensive_team.boxscore.loc[player_id, '3PM'] += stats['3pm']
            offensive_team.boxscore.loc[player_id, 'FTA'] += stats['fta']
            offensive_team.boxscore.loc[player_id, 'FTM'] += stats['ftm']
            offensive_team.boxscore.loc[player_id, 'TO'] += stats['to']
            offensive_team.boxscore.loc[player_id, 'PF'] += stats['pf']
        
        # Update rebound stats if rebound occurred
        if result.get('rebound_info'):
            rebound_info = result['rebound_info']
            rebounder_id = rebound_info['rebounder_id']
            rebounder_team = rebound_info['rebounder_team']
            
            if rebounder_id in rebounder_team.boxscore.index:
                rebounder_team.boxscore.loc[rebounder_id, 'REB'] += 1

        offensive_team.boxscore = offensive_team.boxscore.sort_values(by='PTS', ascending=False)

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
            if pid in offensive_team.boxscore.index:
                offensive_team.boxscore.loc[pid, '+/-'] += total_possession_points
        for pid in defensive_team.lineup:
            if pid in defensive_team.boxscore.index:
                defensive_team.boxscore.loc[pid, '+/-'] -= total_possession_points
        
        # Calculate possession time and update clock/minutes
        # Base time + extra time for offensive rebounds/fouls
        num_outcomes = len(possession_outcomes)
        base_time = np.random.binomial(n=24, p=0.7)  # seconds
        extra_time = max(0, (num_outcomes - 1) * 3)  # +3 seconds per extra outcome
        total_time_seconds = min(base_time + extra_time, 30)  # Cap at 30 seconds
        possession_time_minutes = total_time_seconds / 60.0
        
        self.quarter_duration -= total_time_seconds
        
        
        # Check if quarter/period has ended
        if self.quarter_duration <= 0:
            if self.quarter <= 4:
                print(f"\n🎯 END OF QUARTER {self.quarter}")
                print(f"Score: {self.hometeam.team_abbr} {self.hometeam.score} vs {self.awayteam.team_abbr} {self.awayteam.score}")
                self.quarter += 1
                if self.quarter <= 4:
                    self.quarter_duration = 12 * 60  # Reset to 12 minutes for next quarter
                    # Reset lineups to starters when starting new quarter
                    self.hometeam.lineup = self.hometeam.starters.copy()
                    self.awayteam.lineup = self.awayteam.starters.copy()
                    print(f"Starting Quarter {self.quarter}")
                else:
                    # Regular time is over - check for overtime
                    if self.hometeam.score == self.awayteam.score:
                        print("⏰ END OF REGULATION - TIE GAME!")
                        print("🏀 OVERTIME NEEDED!")
                        self.quarter = 5  # First overtime
                        self.quarter_duration = self.ot_duration  # 5 minutes
                        # Reset lineups to starters for overtime
                        self.hometeam.lineup = self.hometeam.starters.copy()
                        self.awayteam.lineup = self.awayteam.starters.copy()
                        print(f"Starting Overtime {self.quarter - 4}")
                    else:
                        winner = self.hometeam.team_abbr if self.hometeam.score > self.awayteam.score else self.awayteam.team_abbr
                        print(f"🏆 GAME OVER! {winner} wins!")
                        self.quarter = 999  # Mark game as completely finished
            else:
                # This is an overtime period (quarter 5+)
                ot_num = self.quarter - 4
                print(f"\n⏰ END OF OVERTIME {ot_num}")
                print(f"Score: {self.hometeam.team_abbr} {self.hometeam.score} vs {self.awayteam.team_abbr} {self.awayteam.score}")
                
                if self.hometeam.score == self.awayteam.score:
                    # Still tied - another overtime needed
                    self.quarter += 1
                    ot_num = self.quarter - 4
                    self.quarter_duration = self.ot_duration  # 5 minutes
                    # Reset lineups to starters for additional overtime
                    self.hometeam.lineup = self.hometeam.starters.copy()
                    self.awayteam.lineup = self.awayteam.starters.copy()
                    print(f"🏀 Another Overtime needed! Starting Overtime {ot_num}")
                else:
                    # Overtime winner found
                    winner = self.hometeam.team_abbr if self.hometeam.score > self.awayteam.score else self.awayteam.team_abbr
                    print(f"🏆 OVERTIME WINNER: {winner}!")
                    print("🏆 GAME OVER!")
                    self.quarter = 999  # Mark game as completely finished
        
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
        """Simulate a complete game with possible overtime."""
        while self.quarter < 10 and self.quarter != 999:  # Allow up to 10 periods (4 quarters + 6 overtimes)
            self.step()
            # Quarter advancement is handled in step() method now
    
    def get_boxscore_display(self, team: str) -> pd.DataFrame:
        """
        Get boxscore with minutes rounded and shooting percentages added.
        
        Args:
            team: 'home' or 'away'
        
        Returns:
            DataFrame with integer minutes and calculated percentages
        """
        boxscore = self.hometeam.boxscore if team == 'home' else self.awayteam.boxscore
        display_box = boxscore.copy()
        
        # Round minutes to integers for display
        display_box['MIN'] = display_box['MIN'].round(0).astype(int)
        
        # Calculate shooting percentages
        # FG% = FGM / FGA (avoid division by zero)
        display_box['FG%'] = display_box.apply(lambda row: 
            f"{(row['FGM'] / row['FGA'] * 100):.1f}%" if row['FGA'] > 0 else "0.0%", axis=1)
        
        # 3P% = 3PM / 3PA
        display_box['3P%'] = display_box.apply(lambda row: 
            f"{(row['3PM'] / row['3PA'] * 100):.1f}%" if row['3PA'] > 0 else "0.0%", axis=1)
        
        # FT% = FTM / FTA
        display_box['FT%'] = display_box.apply(lambda row: 
            f"{(row['FTM'] / row['FTA'] * 100):.1f}%" if row['FTA'] > 0 else "0.0%", axis=1)
        
        # Reorder columns to put percentages after their respective attempt columns
        cols = list(display_box.columns)
        
        # Remove percentage columns first if they exist to avoid duplicates
        for pct_col in ['FG%', '3P%', 'FT%']:
            if pct_col in cols:
                cols.remove(pct_col)
        
        # Insert percentages after their respective attempt columns
        if 'FGA' in cols:
            fga_idx = cols.index('FGA') + 1
            cols.insert(fga_idx, 'FG%')
        
        if '3PA' in cols:
            threep_idx = cols.index('3PA') + 1
            # Adjust index if FG% was added before 3PA
            if 'FG%' in cols and cols.index('FG%') < threep_idx:
                threep_idx += 1
            cols.insert(threep_idx, '3P%')
        
        if 'FTA' in cols:
            fta_idx = cols.index('FTA') + 1
            # Adjust index if FG% or 3P% were added before FTA
            adjustments = sum(1 for pct in ['FG%', '3P%'] if pct in cols and cols.index(pct) < fta_idx)
            fta_idx += adjustments
            cols.insert(fta_idx, 'FT%')
        
        return display_box[cols]
    
    def restart_game(self):
        """Restart game with same teams."""
        self.__init__(self.hometeam.team_id, self.awayteam.team_id, self.season)
    
    def __str__(self) -> str:
        if self.quarter <= 4:
            period = f"Q{self.quarter}"
        else:
            ot_num = self.quarter - 4
            period = f"OT{ot_num}"
        return f"{period} - {self.hometeam.team_abbr} {self.hometeam.score} vs {self.awayteam.team_abbr} {self.awayteam.score}"


class Season:
    """
    NBA Season Simulation System.
    Simulates complete seasons with standings, player/team stats, and schedules.
    """
    
    def __init__(self, season: str, max_games_per_team: int = 82):
        """
        Initialize season simulation.
        
        Args:
            season: NBA season string (e.g., '2023-24')
            max_games_per_team: Maximum regular season games (default 82)
        """
        self.season = season
        self.max_games_per_team = max_games_per_team
        
        # Initialize predictor once for efficiency
        self.predictor = PlayerAwarePredictor()
        
        # Get all NBA teams
        self.teams_df = nba_teams
        self.team_ids = self.teams_df['id'].tolist()
        self.num_teams = len(self.team_ids)
        
        # Initialize tracking structures
        self.team_records = {}  # team_id -> {'W': wins, 'L': losses, 'PF': points_for, 'PA': points_against}
        self.team_stats = {}    # team_id -> aggregated stats
        self.player_season_stats = {}  # player_id -> season totals
        self.schedule = {}      # team_id -> list of game tuples (opponent_id, home/away, result)
        self.completed_games = []  # List of completed game results
        
        # Initialize team records and stats
        for team_id in self.team_ids:
            self.team_records[team_id] = {'W': 0, 'L': 0, 'PF': 0, 'PA': 0}
            self.team_stats[team_id] = self._init_team_stats()
            self.schedule[team_id] = []
        
        print(f"🏀 Initialized {self.season} season simulation")
        print(f"   Teams: {self.num_teams}")
        print(f"   Max games per team: {self.max_games_per_team}")
    
    def _init_team_stats(self) -> Dict:
        """Initialize empty team stats."""
        return {
            'games_played': 0,
            'total_minutes': 0.0,
            'total_points': 0,
            'total_rebounds': 0,
            'total_assists': 0,
            'total_fgm': 0,
            'total_fga': 0,
            'total_3pm': 0,
            'total_3pa': 0,
            'total_ftm': 0,
            'total_fta': 0,
            'total_turnovers': 0,
            'total_fouls': 0
        }
    
    def generate_schedule(self):
        """Generate a coordinated NBA schedule ensuring each team plays exactly max_games_per_team games."""
        print("📅 Generating schedule...")
        
        # Create a global list of all games to ensure no duplicates
        all_games = []
        games_per_team = {team_id: 0 for team_id in self.team_ids}
        
        # Continue adding games until each team has played the required number
        while min(games_per_team.values()) < self.max_games_per_team:
            # Shuffle team order for randomness
            teams_to_consider = [tid for tid in self.team_ids if games_per_team[tid] < self.max_games_per_team]
            
            if len(teams_to_consider) < 2:
                break  # Can't create more games if fewer than 2 teams need games
                
            random.shuffle(teams_to_consider)
            
            for team_id in teams_to_consider:
                if games_per_team[team_id] >= self.max_games_per_team:
                    continue
                    
                # Find potential opponents who also need games
                potential_opponents = [tid for tid in self.team_ids 
                                     if tid != team_id and games_per_team[tid] < self.max_games_per_team]
                
                if not potential_opponents:
                    continue
                    
                opponent_id = random.choice(potential_opponents)
                
                # Check if this matchup already exists recently (avoid immediate rematches)
                recent_games = all_games[-10:] if len(all_games) >= 10 else all_games
                recent_matchup = any(
                    (g['home_id'] == team_id and g['away_id'] == opponent_id) or
                    (g['home_id'] == opponent_id and g['away_id'] == team_id)
                    for g in recent_games
                )
                
                # If too recent, try a different opponent
                if len(potential_opponents) > 1 and recent_matchup:
                    other_opponents = [tid for tid in potential_opponents if tid != opponent_id]
                    if other_opponents:
                        opponent_id = random.choice(other_opponents)
                
                # Randomly assign home/away (but prefer to balance home/away for each team)
                home_away_balance = sum(1 for g in all_games if g['home_id'] == team_id) - \
                                  sum(1 for g in all_games if g['away_id'] == team_id)
                
                if abs(home_away_balance) < 2 and random.random() < 0.5:
                    # Try to balance home/away
                    is_home = home_away_balance <= 0
                else:
                    is_home = random.choice([True, False])
                
                # Create the game
                game_id = f"game_{len(all_games)}_{team_id}_{opponent_id}"
                game = {
                    'home_id': team_id if is_home else opponent_id,
                    'away_id': opponent_id if is_home else team_id,
                    'result': None,
                    'game_id': game_id
                }
                
                all_games.append(game)
                games_per_team[team_id] += 1
                games_per_team[opponent_id] += 1
                
                # Stop if we've reached the limit for all teams
                if min(games_per_team.values()) >= self.max_games_per_team:
                    break
        
        # Convert global game list to per-team schedules
        for game in all_games:
            home_id = game['home_id']
            away_id = game['away_id']
            
            # Add to home team's schedule
            self.schedule[home_id].append({
                'opponent_id': away_id,
                'is_home': True,
                'result': None,
                'game_id': game['game_id']
            })
            
            # Add to away team's schedule
            self.schedule[away_id].append({
                'opponent_id': home_id,
                'is_home': False,
                'result': None,
                'game_id': game['game_id']
            })
        
        # Verify each team has the correct number of games
        for team_id in self.team_ids:
            actual_games = len(self.schedule[team_id])
            if actual_games != self.max_games_per_team:
                print(f"⚠️  Warning: Team {team_id} has {actual_games} games instead of {self.max_games_per_team}")
        
        total_games = len(all_games)
        print(f"   Generated {total_games} total games")
        print(f"   Each team should play exactly {self.max_games_per_team} games")
        
        # Verify schedule integrity
        self._verify_schedule_integrity()
    
    def _verify_schedule_integrity(self):
        """Verify that the schedule is properly coordinated between teams."""
        # Collect all games by ID to check for consistency
        all_games = {}
        
        for team_id in self.team_ids:
            for game in self.schedule[team_id]:
                game_id = game['game_id']
                if game_id not in all_games:
                    all_games[game_id] = {
                        'home_team_id': game['opponent_id'] if not game['is_home'] else team_id,
                        'away_team_id': team_id if not game['is_home'] else game['opponent_id'],
                        'teams_seen': set()
                    }
                
                all_games[game_id]['teams_seen'].add(team_id)
        
        # Check that each game has exactly 2 teams involved
        for game_id, game_info in all_games.items():
            if len(game_info['teams_seen']) != 2:
                print(f"⚠️  Warning: Game {game_id} has {len(game_info['teams_seen'])} teams instead of 2")
            
            # Check that home/away is consistent
            expected_home = game_info['home_team_id']
            expected_away = game_info['away_team_id']
            
            # Verify this in both teams' schedules
            home_schedule = self.schedule.get(expected_home, [])
            away_schedule = self.schedule.get(expected_away, [])
            
            home_game_found = any(g['game_id'] == game_id and g['is_home'] for g in home_schedule)
            away_game_found = any(g['game_id'] == game_id and not g['is_home'] for g in away_schedule)
            
            if not home_game_found or not away_game_found:
                print(f"⚠️  Warning: Game {game_id} home/away assignment inconsistent")
    
    def simulate_game(self, home_team_id: int, away_team_id: int) -> Dict:
        """
        Simulate a single game between two teams.
        
        Returns:
            Dict with game result including scores, player stats, etc.
        """
        try:
            # Add small delay to prevent API overload
            time.sleep(0.5)
            
            # Create game engine
            game = GameEngine(home_team_id, away_team_id, self.season, self.predictor)
            
            # Simulate the full game
            while game.quarter < 10 and game.quarter != 999:  # Allow overtime
                game.step()
            
            # Extract results
            home_score = game.hometeam.score
            away_score = game.awayteam.score
            
            result = {
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_score': home_score,
                'away_score': away_score,
                'home_boxscore': game.get_boxscore_display('home'),
                'away_boxscore': game.get_boxscore_display('away'),
                'game_completed': True
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error simulating game {home_team_id} vs {away_team_id}: {e}")
            return {
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_score': 0,
                'away_score': 0,
                'game_completed': False,
                'error': str(e)
            }
    
    def update_records_and_stats(self, game_result: Dict):
        """Update team records and stats after a game."""
        if not game_result['game_completed']:
            return
            
        home_id = game_result['home_team_id']
        away_id = game_result['away_team_id']
        home_score = game_result['home_score']
        away_score = game_result['away_score']
        
        # Update win/loss records
        if home_score > away_score:
            self.team_records[home_id]['W'] += 1
            self.team_records[away_id]['L'] += 1
        else:
            self.team_records[home_id]['L'] += 1
            self.team_records[away_id]['W'] += 1
        
        # Update points for/against
        self.team_records[home_id]['PF'] += home_score
        self.team_records[home_id]['PA'] += away_score
        self.team_records[away_id]['PF'] += away_score
        self.team_records[away_id]['PA'] += home_score
        
        # Update team aggregate stats
        self._update_team_stats_from_boxscore(home_id, game_result['home_boxscore'])
        self._update_team_stats_from_boxscore(away_id, game_result['away_boxscore'])
        
        # Update player season stats
        self._update_player_stats_from_boxscore(home_id, game_result['home_boxscore'])
        self._update_player_stats_from_boxscore(away_id, game_result['away_boxscore'])
    
    def _update_team_stats_from_boxscore(self, team_id: int, boxscore: pd.DataFrame):
        """Update team aggregate stats from a boxscore."""
        stats = self.team_stats[team_id]
        stats['games_played'] += 1
        
        # Sum up stats from all players
        for _, row in boxscore.iterrows():
            stats['total_minutes'] += row.get('MIN', 0)
            stats['total_points'] += row.get('PTS', 0)
            stats['total_rebounds'] += row.get('REB', 0)
            stats['total_assists'] += row.get('AST', 0)
            stats['total_fgm'] += row.get('FGM', 0)
            stats['total_fga'] += row.get('FGA', 0)
            stats['total_3pm'] += row.get('3PM', 0)
            stats['total_3pa'] += row.get('3PA', 0)
            stats['total_ftm'] += row.get('FTM', 0)
            stats['total_fta'] += row.get('FTA', 0)
            stats['total_turnovers'] += row.get('TO', 0)
            stats['total_fouls'] += row.get('PF', 0)
    
    def _update_player_stats_from_boxscore(self, team_id: int, boxscore: pd.DataFrame):
        """Update player season stats from a boxscore."""
        for player_id, row in boxscore.iterrows():
            if player_id not in self.player_season_stats:
                player_name = row.get('PLAYER', f'Player_{player_id}')
                self.player_season_stats[player_id] = {
                    'player_name': player_name,
                    'team_id': team_id,
                    'games_played': 0,
                    'total_minutes': 0.0,
                    'total_points': 0,
                    'total_rebounds': 0,
                    'total_assists': 0,
                    'total_fgm': 0,
                    'total_fga': 0,
                    'total_3pm': 0,
                    'total_3pa': 0,
                    'total_ftm': 0,
                    'total_fta': 0,
                    'total_turnovers': 0,
                    'total_fouls': 0,
                    'total_plus_minus': 0
                }
            
            player_stats = self.player_season_stats[player_id]
            player_stats['games_played'] += 1
            player_stats['total_minutes'] += row.get('MIN', 0)
            player_stats['total_points'] += row.get('PTS', 0)
            player_stats['total_rebounds'] += row.get('REB', 0)
            player_stats['total_assists'] += row.get('AST', 0)
            player_stats['total_fgm'] += row.get('FGM', 0)
            player_stats['total_fga'] += row.get('FGA', 0)
            player_stats['total_3pm'] += row.get('3PM', 0)
            player_stats['total_3pa'] += row.get('3PA', 0)
            player_stats['total_ftm'] += row.get('FTM', 0)
            player_stats['total_fta'] += row.get('FTA', 0)
            player_stats['total_turnovers'] += row.get('TO', 0)
            player_stats['total_fouls'] += row.get('PF', 0)
            player_stats['total_plus_minus'] += row.get('+/-', 0)
    
    def simulate_season(self, progress_callback=None):
        """Simulate the entire season."""
        print(f"\n🏀 Starting {self.season} season simulation...")
        
        # Generate schedule if not already done
        if not any(self.schedule.values()):
            self.generate_schedule()
        
        # Collect all unique games by game_id to avoid duplicates
        all_games_by_id = {}
        for team_id in self.team_ids:
            for game in self.schedule[team_id]:
                if game['result'] is None:  # Only process unsimulated games
                    game_id = game['game_id']
                    if game_id not in all_games_by_id:
                        # Determine home/away teams for this game
                        if game['is_home']:
                            home_id = team_id
                            away_id = game['opponent_id']
                        else:
                            home_id = game['opponent_id']
                            away_id = team_id
                        
                        all_games_by_id[game_id] = {
                            'game_id': game_id,
                            'home_id': home_id,
                            'away_id': away_id,
                            'team_schedules': [(team_id, game)]  # Track which team schedules contain this game
                        }
                    else:
                        # Add this team's schedule entry to the existing game
                        all_games_by_id[game_id]['team_schedules'].append((team_id, game))
        
        total_games = len(all_games_by_id)
        games_completed = 0
        
        print(f"   Found {total_games} unique games to simulate")
        
        # Simulate each unique game once
        for game_id, game_info in all_games_by_id.items():
            home_id = game_info['home_id']
            away_id = game_info['away_id']
            
            # Simulate the game
            game_result = self.simulate_game(home_id, away_id)
            
            # Update the result in all team schedules that contain this game
            for team_id, schedule_entry in game_info['team_schedules']:
                schedule_entry['result'] = game_result
            
            # Update records and stats
            self.update_records_and_stats(game_result)
            self.completed_games.append(game_result)
            
            games_completed += 1
            
            if games_completed % 10 == 0:
                print(f"   Completed {games_completed}/{total_games} games...")
            
            if progress_callback:
                progress_callback(games_completed, total_games)
        
        print(f"\n✅ Season simulation complete! {games_completed} games played.")
        
        # Verify each team played the expected number of games
        for team_id in self.team_ids:
            completed_games_for_team = len([g for g in self.schedule[team_id] if g['result'] is not None])
            if completed_games_for_team != self.max_games_per_team:
                team_name = self.teams_df[self.teams_df['id'] == team_id]['nickname'].iloc[0]
                print(f"⚠️  Warning: {team_name} played {completed_games_for_team} games instead of {self.max_games_per_team}")
    
    def get_standings(self) -> pd.DataFrame:
        """Get current league standings."""
        standings_data = []
        
        for team_id in self.team_ids:
            record = self.team_records[team_id]
            team_name = self.teams_df[self.teams_df['id'] == team_id]['nickname'].iloc[0]
            team_city = self.teams_df[self.teams_df['id'] == team_id]['city'].iloc[0]
            
            wins = record['W']
            losses = record['L']
            win_pct = wins / max(wins + losses, 1)
            
            standings_data.append({
                'Team': f"{team_city} {team_name}",
                'W': wins,
                'L': losses,
                'PCT': f"{win_pct:.3f}",
                'PF': record['PF'],
                'PA': record['PA'],
                'DIFF': record['PF'] - record['PA']
            })
        
        standings_df = pd.DataFrame(standings_data)
        standings_df = standings_df.sort_values(['PCT', 'DIFF'], ascending=[False, False])
        standings_df.index = range(1, len(standings_df) + 1)
        
        return standings_df
    
    def get_team_averages(self, team_id: int) -> Dict:
        """Get per-game averages for a team."""
        stats = self.team_stats[team_id]
        record = self.team_records[team_id]
        
        if stats['games_played'] == 0:
            return {}
        
        return {
            'games_played': stats['games_played'],
            'wins': record['W'],
            'losses': record['L'],
            'win_pct': record['W'] / max(record['W'] + record['L'], 1),
            'points_per_game': stats['total_points'] / stats['games_played'],
            'rebounds_per_game': stats['total_rebounds'] / stats['games_played'],
            'assists_per_game': stats['total_assists'] / stats['games_played'],
            'fg_pct': stats['total_fgm'] / max(stats['total_fga'], 1) * 100,
            'three_pct': stats['total_3pm'] / max(stats['total_3pa'], 1) * 100,
            'ft_pct': stats['total_ftm'] / max(stats['total_fta'], 1) * 100,
            'turnovers_per_game': stats['total_turnovers'] / stats['games_played'],
            'fouls_per_game': stats['total_fouls'] / stats['games_played']
        }
    
    def get_player_averages(self, player_id: int = None) -> pd.DataFrame:
        """Get per-game averages for players."""
        if player_id:
            # Return specific player
            if player_id not in self.player_season_stats:
                return pd.DataFrame()
            
            stats = self.player_season_stats[player_id]
            if stats['games_played'] == 0:
                return pd.DataFrame()
            
            return pd.DataFrame([{
                'Player': stats['player_name'],
                'Team': self.teams_df[self.teams_df['id'] == stats['team_id']]['nickname'].iloc[0],
                'GP': stats['games_played'],
                'MIN': stats['total_minutes'] / stats['games_played'],
                'PTS': stats['total_points'] / stats['games_played'],
                'REB': stats['total_rebounds'] / stats['games_played'],
                'AST': stats['total_assists'] / stats['games_played'],
                'FG%': stats['total_fgm'] / max(stats['total_fga'], 1) * 100,
                '3P%': stats['total_3pm'] / max(stats['total_3pa'], 1) * 100,
                'FT%': stats['total_ftm'] / max(stats['total_fta'], 1) * 100,
                'TO': stats['total_turnovers'] / stats['games_played'],
                'PF': stats['total_fouls'] / stats['games_played'],
                '+/-': stats['total_plus_minus'] / stats['games_played']
            }])
        
        else:
            # Return all players with minimum games played
            min_games = 10  # Minimum games to appear in leaderboards
            
            player_data = []
            for pid, stats in self.player_season_stats.items():
                if stats['games_played'] >= min_games:
                    team_name = self.teams_df[self.teams_df['id'] == stats['team_id']]['nickname'].iloc[0]
                    
                    player_data.append({
                        'Player': stats['player_name'],
                        'Team': team_name,
                        'GP': stats['games_played'],
                        'MIN': stats['total_minutes'] / stats['games_played'],
                        'PTS': stats['total_points'] / stats['games_played'],
                        'REB': stats['total_rebounds'] / stats['games_played'],
                        'AST': stats['total_assists'] / stats['games_played'],
                        'FG%': stats['total_fgm'] / max(stats['total_fga'], 1) * 100,
                        '3P%': stats['total_3pm'] / max(stats['total_3pa'], 1) * 100,
                        'FT%': stats['total_ftm'] / max(stats['total_fta'], 1) * 100,
                        'TO': stats['total_turnovers'] / stats['games_played'],
                        'PF': stats['total_fouls'] / stats['games_played'],
                        '+/-': stats['total_plus_minus'] / stats['games_played']
                    })
            
            df = pd.DataFrame(player_data)
            return df.sort_values('PTS', ascending=False)
    
    def get_leaders(self, stat: str, limit: int = 10) -> pd.DataFrame:
        """Get league leaders for a specific stat."""
        player_averages = self.get_player_averages()
        
        if player_averages.empty:
            return pd.DataFrame()
        
        if stat not in player_averages.columns:
            print(f"❌ Stat '{stat}' not found. Available stats: {list(player_averages.columns)}")
            return pd.DataFrame()
        
        # Sort by the requested stat (descending for most stats, ascending for turnovers/fouls)
        ascending = stat in ['TO', 'PF']
        
        leaders = player_averages.sort_values(stat, ascending=ascending).head(limit)
        leaders = leaders.reset_index(drop=True)
        leaders.index = range(1, len(leaders) + 1)
        
        return leaders[['Player', 'Team', 'GP', stat]]
    
    def get_team_name(self, team_id: int) -> str:
        """Get full team name from team ID."""
        team_row = self.teams_df[self.teams_df['id'] == team_id]
        if not team_row.empty:
            return f"{team_row.iloc[0]['city']} {team_row.iloc[0]['nickname']}"
        return f"Team_{team_id}"
    
    def print_season_summary(self):
        """Print a comprehensive season summary."""
        print(f"\n🏆 {self.season} NBA Season Summary")
        print("=" * 50)
        
        # Standings
        standings = self.get_standings()
        print(f"\n📊 Final Standings (Top 10):")
        print(standings.head(10).to_string())
        
        # Scoring leaders
        print(f"\n🔥 Scoring Leaders:")
        scoring_leaders = self.get_leaders('PTS', 5)
        print(scoring_leaders.to_string())
        
        # Rebounding leaders
        print(f"\n🏀 Rebounding Leaders:")
        reb_leaders = self.get_leaders('REB', 5)
        print(reb_leaders.to_string())
        
        # Assist leaders
        print(f"\n🎯 Assist Leaders:")
        ast_leaders = self.get_leaders('AST', 5)
        print(ast_leaders.to_string())
        
        # League-wide stats
        total_games = len(self.completed_games)
        print(f"\n📈 League Statistics:")
        print(f"   Total Games Played: {total_games}")
        print(f"   Teams: {self.num_teams}")
        
        # Calculate average points per game across all teams
        total_points = 0
        games_with_scores = 0
        
        for game in self.completed_games:
            if game['game_completed']:
                total_points += game['home_score'] + game['away_score']
                games_with_scores += 1
        
        if games_with_scores > 0:
            avg_ppg = total_points / games_with_scores / 2  # Average per team per game
            print(f"   Average Points Per Game: {avg_ppg:.1f}")


# Example usage function
def run_season_simulation(season: str = '2024-25', max_games: int = 10):
    """
    Example function to run a season simulation.
    
    Args:
        season: NBA season to simulate
        max_games: Maximum games per team (use 10 for quick testing, 82 for full season)
    """
    print(f"🚀 Starting NBA Season Simulation")
    print(f"   Season: {season}")
    print(f"   Games per team: {max_games}")
    
    # Initialize season
    season_sim = Season(season=season, max_games_per_team=max_games)
    
    # Run simulation
    season_sim.simulate_season()
    
    # Print results
    season_sim.print_season_summary()
    
    return season_sim


if __name__ == "__main__":
    # Example usage
    season = run_season_simulation(max_games=5)  # Quick test with 5 games per team