# imports
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog
from player_aware_predictor import PlayerAwarePredictor
from train_possession_model import FeatureEngineer  # Needed for unpickling
import numpy as np
import pandas as pd


# ====================================================================
# helper functions
# ====================================================================

# load dataframe of nba teams
nba_teams = pd.DataFrame(teams.get_teams())

# box score variables
BOXCOLS = ["MIN", "PTS","REB","AST","FGA","FGM","3PA","3PM","FTA","FTM","STL","BLK","TO","PF","+/-"]

# function to initialize boxscore from roster 
def init_boxscore_from_roster(roster_df: pd.DataFrame) -> pd.DataFrame:
    box = roster_df[["PLAYER_ID","PLAYER","POSITION","NUM"]].copy()
    for c in BOXCOLS:
        if c == "MIN":
            box[c] = 0.0  # Minutes stored as float for precision
        else:
            box[c] = 0    # Other stats as integers
    return box.set_index("PLAYER_ID")

# minutes to float function
def _mins_to_float(v):
    # "MM:SS" -> minutes as float; also handles NaN or numeric
    if pd.isna(v): return 0.0
    if isinstance(v, (int, float)): return float(v)
    s = str(v)
    if ":" in s:
        m, sec = s.split(":")
        return float(m) + float(sec)/60.0
    return float(s)


# nba team id, name, abbr resolver
def team_resolver(id=None, abbr=None, name=None) -> dict:

    # check to make sure only one value is present
    if sum(x is not None for x in (id, abbr, name)) != 1:
        raise ValueError("Provide exactly one of team_id, abbr, or name.")
    
    # by id
    if id is not None:
        team_name = nba_teams[nba_teams['id'] == id]['nickname'].iloc[0] # returns the name of the team
        team_abbr = nba_teams[nba_teams['id'] == id]['abbreviation'].iloc[0] # returns abbreviation
        team_id = id
    
    # by name
    elif name is not None:
        team_id = nba_teams[nba_teams['nickname'] == name]['id'].iloc[0] # returns the id of the team
        team_abbr = nba_teams[nba_teams['id'] == team_id]['abbreviation'].iloc[0] # returns abbreviation
        team_name = name

    # by abbreviation
    else:
        team_id = nba_teams[nba_teams['abbreviation'] == abbr]['id'].iloc[0] # returns the name of the team
        team_name = nba_teams[nba_teams['id'] == team_id]['nickname'].iloc[0] # returns abbreviation
        team_abbr = abbr

    location = nba_teams[nba_teams['id'] == team_id]['city'].iloc[0]
    
    return {'team_id': team_id, 'team_abbr': team_abbr, 'team_name': team_name, 'location': location}
    

# ====================================================
# game simmulation classes
# ====================================================

# team class

class Team:
    """
    Team class for NBA game simulation.
    Manages team identification, roster, and starters.
    """
    def __init__(
            self, 
            season: str, 
            team_id: int = None, 
            team_abbr: str = None, 
            team_name: str = None
        ):
        """
        Initialize Team class.
        Args:
            season: Season string (e.g., '2023-24')
            team_id: Team ID (integer) OR
            team_abbr: Team abbreviation (string) OR
            team_name: Team name (string)
        """

        team_info = team_resolver(team_id, team_abbr, team_name)

        self.team_id = int(team_info['team_id'])
        self.team_name = str(team_info['team_name'])
        self.team_abbr = str(team_info['team_abbr'])
        self.location = str(team_info['location'])
        self.season = str(season)
        
        endpoint = commonteamroster.CommonTeamRoster(team_id=self.team_id, season=self.season)
        self.roster_df = endpoint.get_data_frames()[0]
        self.starters = self.pick_starters(self.roster_df, season)
        self.lineup = self.starters.copy()
        self.score = 0
        self.fouls = 0
        self.timeouts = 7
        self.boxscore = init_boxscore_from_roster(self.roster_df)


    def __str__(self) -> str:
        header = f"{self.season} {self.location} {self.team_name} Summary:"
        meta = f"{self.team_abbr} | Team ID: {self.team_id}"
        roster_preview = self.roster_df[['PLAYER_ID', 'PLAYER', 'POSITION', 'NUM']].to_string(index=False)
        return f"{header}\n\n{meta}\n\nRoster:\n{roster_preview}"
    
    def pick_starters(self, roster_df: pd.DataFrame, season: str) -> list:
        """
        Pick the 5 starting players based on games started and average minutes.
        
        Returns:
            list: List of 5 player IDs (integers)
        """
        records = []
        for pid in roster_df["PLAYER_ID"].tolist():
            try:
                # reuse your Player class
                p = Player(player_id=int(pid), season=season)
                df = p.gamelog
                if df.empty:
                    continue
                starts = int(df.get("GS", pd.Series([0]*len(df))).fillna(0).astype(int).sum())
                avg_min = float(pd.Series(df.get("MIN", [])).apply(_mins_to_float).mean()) if "MIN" in df.columns else 0.0
                records.append((int(pid), starts, avg_min))
            except Exception:
                # player may have no logs; skip quietly
                pass

        # sort by (starts desc, avg_min desc) and take top 5
        records.sort(key=lambda x: (x[1], x[2]), reverse=True)
        # Return only player IDs, not the stats
        return [pid for pid, _, _ in records[:5]]
    

class Player:
    def __init__(self, player_id: int, season: str):
        player_info = players.find_player_by_id(player_id)

        self.id = player_id
        self.name = player_info['full_name']
        self.gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Regular Season').get_data_frames()[0]
    

class GameEngine:
    def __init__(
            self, 
            hometeam_id: int, 
            awayteam_id: int, 
            season: str,
            predictor: PlayerAwarePredictor = None
        ):
        
        self.season = season
        self.hometeam = Team(team_id = hometeam_id, season = season)
        self.awayteam = Team(team_id = awayteam_id, season=season)

        # initialize predictor
        self.predictor = PlayerAwarePredictor()

        # ====== Game State Variables ======
        self.quarter = 1
        self.poss = np.random.choice([1, -1])  # 1 = home has ball, -1 = away has ball
        self.play = 1  # Play counter (increments each possession)
        self.quarter_duration = 12 * 60  # Seconds remaining in quarter
        self.ot_duration = 5 * 60  # OT duration (seconds)
        self.deadball = False  # True when ball is not in play
        
        # ====== Energy & Fatigue System ======
        all_home_players = self.hometeam.roster_df['PLAYER_ID'].tolist()
        all_away_players = self.awayteam.roster_df['PLAYER_ID'].tolist()
        self.player_energy = {pid: 100.0 for pid in all_home_players + all_away_players}
        self.player_minutes = {pid: 0.0 for pid in all_home_players + all_away_players}
        
        # ====== Game Logs ======
        self.substitution_log = []  # Track all substitutions
    
    # ================================================================
    # FATIGUE & ENERGY MANAGEMENT
    # ================================================================
    
    def get_fatigue_multiplier(self, player_id: int) -> float:
        """
        Calculate performance multiplier based on player's energy.
        Energy Pool System (Option C)
        
        Returns:
            float: Multiplier from 0.5 (exhausted) to 1.0 (fresh)
        """
        energy = self.player_energy.get(player_id, 100.0)
        
        if energy > 80:
            return 1.0  # Fresh (100-80 energy)
        elif energy > 50:
            # 80-50 energy → 1.0 to 0.85 multiplier
            return 0.85 + (energy - 50) * 0.005
        elif energy > 20:
            # 50-20 energy → 0.85 to 0.70 multiplier
            return 0.70 + (energy - 20) * 0.005
        else:
            # 20-0 energy → 0.70 to 0.50 multiplier
            return 0.50 + energy * 0.01
    
    def deplete_energy(self, player_id: int, base_cost: float, intensity_bonus: float = 0.0):
        """
        Deplete a player's energy based on activity level.
        
        Args:
            player_id: Player whose energy to deplete
            base_cost: Base energy cost (1.0 per possession on court)
            intensity_bonus: Extra cost for high-intensity actions
        """
        total_cost = base_cost + intensity_bonus
        self.player_energy[player_id] -= total_cost
        self.player_energy[player_id] = max(0, self.player_energy[player_id])
    
    def rest_player(self, player_id: int):
        """
        Regenerate energy for a player on the bench.
        Called each possession for players NOT on court.
        """
        recovery_rate = 2.0  # Recover 2 energy per possession
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
            bool: True if substitution successful, False otherwise
        """
        # Validate inputs
        if position < 0 or position > 4:
            return False
        
        if new_player_id is None:
            return False
        
        # Get current lineup and roster
        if team == 'home':
            lineup = self.home_lineup
            roster = self.hometeam.roster_df
            team_name = self.hometeam.team_abbr
        else:
            lineup = self.away_lineup
            roster = self.awayteam.roster_df
            team_name = self.awayteam.team_abbr
        
        # Validate new player is on the roster
        if new_player_id not in roster['PLAYER_ID'].values:
            return False
        
        # Validate new player isn't already in the game
        if new_player_id in lineup:
            return False
        
        # Get player being subbed out
        old_player_id = lineup[position]
        
        # Get player names for logging
        old_player_name = roster[roster['PLAYER_ID'] == old_player_id]['PLAYER'].values[0] if old_player_id in roster['PLAYER_ID'].values else f"Player {old_player_id}"
        new_player_name = roster[roster['PLAYER_ID'] == new_player_id]['PLAYER'].values[0]
        
        # Make the substitution
        if team == 'home':
            self.home_lineup[position] = new_player_id
        else:
            self.away_lineup[position] = new_player_id
        
        # Log the substitution (optional - for game recap/debugging)
        if not hasattr(self, 'substitution_log'):
            self.substitution_log = []
        
        self.substitution_log.append({
            'quarter': self.quarter,
            'time_remaining': self.quarter_duration,
            'team': team,
            'player_out': old_player_id,
            'player_out_name': old_player_name,
            'player_in': new_player_id,
            'player_in_name': new_player_name,
            'energy_out': self.player_energy[old_player_id],
            'energy_in': self.player_energy[new_player_id]
        })
        
        # Note: Energy regeneration happens automatically in step() for bench players
        
        return True
    
    def get_player_quality(self, player_id: int) -> float:
        """
        Calculate player quality score for substitution decisions.
        Combines usage rate with efficiency.
        
        Returns:
            float: Quality score (0.0 to ~0.5, higher is better)
        """
        key = (int(player_id), self.season)
        
        if key not in self.predictor.fe.player_season_stats:
            return 0.1  # Unknown player → low quality
        
        stats = self.predictor.fe.player_season_stats[key]
        
        # Quality = weighted combination of usage and efficiency
        quality = (
            stats['usage_rate'] * 0.5 +      # 50% weight: how much they're used
            stats['2pt_pct'] * 0.25 +        # 25% weight: 2PT shooting
            stats['3pt_pct'] * 0.25          # 25% weight: 3PT shooting
        )
        
        return quality
    
    def find_best_bench_player(self, team: str, energy_weight: float = 0.6, quality_weight: float = 0.4) -> int:
        """
        Find best bench player balancing freshness and quality.
        
        Hybrid approach: Fresh legs + skill level
        
        Args:
            team: 'home' or 'away'
            energy_weight: How much to weight energy (0.6 = 60% importance)
            quality_weight: How much to weight quality (0.4 = 40% importance)
        
        Returns:
            player_id: Best available bench player
        """
        roster = self.hometeam.roster_df if team == 'home' else self.awayteam.roster_df
        lineup = self.hometeam.lineup if team == 'home' else self.awayteam.lineup
        
        # Get all bench players
        bench_ids = [pid for pid in roster['PLAYER_ID'].tolist() if pid not in lineup]
        
        if not bench_ids:
            return None
        
        # Score each bench player
        scores = []
        for player_id in bench_ids:
            # Normalize energy to 0-1 scale
            energy_score = self.player_energy[player_id] / 100.0
            
            # Get quality score
            quality_score = self.get_player_quality(player_id)
            
            # Normalize quality to 0-1 scale (typical max ~0.4)
            quality_normalized = min(1.0, quality_score / 0.4)
            
            # Combined score
            combined_score = (energy_score * energy_weight) + (quality_normalized * quality_weight)
            
            scores.append((player_id, combined_score, energy_score, quality_normalized))
        
        # Return player with highest combined score
        best = max(scores, key=lambda x: x[1])
        return int(best[0])
    
    # ================================================================
    # GAME STATE & DECISION LOGIC
    # ================================================================
    
    def is_close_game(self) -> bool:
        """
        Check if game is competitive (affects substitution strategy).
        Close games → keep stars in longer.
        """
        margin = abs(self.hometeam.score - self.awayteam.score)
        
        # Q4 last 6 minutes within 5 points = close
        if self.quarter == 4 and self.quarter_duration < 360 and margin <= 5:
            return True
        
        # Q4 last 2 minutes within 8 points = close
        if self.quarter == 4 and self.quarter_duration < 120 and margin <= 8:
            return True
        
        return False
    
    def is_blowout(self) -> bool:
        """
        Check if game is decided (affects substitution strategy).
        Blowouts → rest starters completely.
        """
        margin = abs(self.hometeam.score - self.awayteam.score)
        
        # Q3 or Q4 with 20+ point lead
        if self.quarter >= 3 and margin >= 20:
            return True
        
        # Q4 with <3 minutes and 15+ point lead
        if self.quarter == 4 and self.quarter_duration < 180 and margin >= 15:
            return True
        
        return False
    
    def check_substitutions(self):
        """
        Hybrid substitution strategy:
        - Time-based windows (6-minute mark, between quarters)
        - Fatigue-based triggers (emergency subs)
        - Game situation aware (close vs blowout)
        """
        if not self.deadball:
            return
        
        # Between quarters - always bring back starters
        if self.quarter_duration <= 0:
            self.hometeam.lineup = self.hometeam.starters.copy()
            self.awayteam.lineup = self.awayteam.starters.copy()
            return
        
        # Check game situation
        close_game = self.is_close_game()
        blowout = self.is_blowout()
        
        # Process substitutions for both teams
        for team in ['home', 'away']:
            self._make_team_substitutions(team, close_game, blowout)
    
    def _make_team_substitutions(self, team: str, close_game: bool, blowout: bool):
        """
        Make substitutions for one team based on hybrid strategy.
        """
        lineup = self.hometeam.lineup if team == 'home' else self.awayteam.lineup
        boxscore = self.hometeam.boxscore if team == 'home' else self.awayteam.boxscore
        starters = self.hometeam.starters if team == 'home' else self.awayteam.starters
        
        for i, player_id in enumerate(lineup):
            energy = self.player_energy[player_id]
            fatigue_mult = self.get_fatigue_multiplier(player_id)
            fouls = boxscore.loc[player_id, 'PF']
            
            should_sub = False
            reason = ""
            
            # PRIORITY 1: Critical situations (always sub, even in close games)
            if fouls >= 5:
                should_sub = True
                reason = "foul_trouble"
            elif energy < 20:
                should_sub = True
                reason = "exhausted"
            
            # PRIORITY 2: Blowout (rest all starters)
            elif blowout:
                if player_id in starters:
                    should_sub = True
                    reason = "blowout_rest"
            
            # PRIORITY 3: Regular fatigue (skip if close game)
            elif not close_game:
                # Very tired
                if energy < 40:
                    should_sub = True
                    reason = "very_tired"
                
                # 6-minute rotation window
                elif 5.5*60 < self.quarter_duration < 6.5*60:
                    if energy < 70:  # Only sub if they're getting tired
                        should_sub = True
                        reason = "rotation"
            
            # Execute substitution
            if should_sub:
                # Find best bench replacement (balances energy + quality)
                bench_player = self.find_best_bench_player(team)
                
                if bench_player:
                    # Only sub if bench player is reasonably fresh
                    if self.player_energy[bench_player] > 60 or reason in ['foul_trouble', 'exhausted']:
                        self.substitute_player(team, i, bench_player)
    
    def call_timeout(self, team: str) -> bool:
        """
        Call a timeout - forces deadball and allows substitutions.
        
        Args:
            team: 'home' or 'away'
        
        Returns:
            bool: True if timeout called, False if none available
        """
        if team == 'home':
            if self.hometeam.timeouts <= 0:
                return False
            self.hometeam.timeouts -= 1
        else:
            if self.awayteam.timeouts <= 0:
                return False
            self.awayteam.timeouts -= 1
        
        # Force deadball
        self.deadball = True
        
        # Players get small energy recovery during timeout
        for player_id in self.player_energy.keys():
            self.player_energy[player_id] = min(100, self.player_energy[player_id] + 2.0)
        
        # Now make substitutions
        self.check_substitutions()
        
        return True
    
    def should_call_timeout(self, team: str) -> bool:
        """
        Decide if team should call timeout.
        
        Timeouts used for:
        1. Force deadball when team exhausted
        2. Multiple subs needed at once
        3. (Future: stop opponent runs, advance ball)
        """
        if team == 'home':
            timeouts = self.hometeam.timeouts
            lineup = self.hometeam.lineup
        else:
            timeouts = self.awayteam.timeouts
            lineup = self.awayteam.lineup
        
        if timeouts <= 0:
            return False
        
        # Don't use timeouts in blowouts
        if self.is_blowout():
            return False
        
        # Save last timeout for Q4 final minute
        if self.quarter == 4 and self.quarter_duration < 60 and timeouts <= 1:
            return False
        
        # Reason 1: Team is exhausted but no natural deadball
        avg_energy = sum(self.player_energy[pid] for pid in lineup) / 5
        if avg_energy < 45 and not self.deadball:
            return True
        
        # Reason 2: Multiple players need rest
        tired_count = sum(1 for pid in lineup if self.player_energy[pid] < 40)
        if tired_count >= 3 and not self.deadball:
            return True
        
        return False
    
    # ================================================================
    # MAIN SIMULATION LOOP
    # ================================================================

        return int(best[0])
    
    # ================================================================
    # GAME STATE & DECISION LOGIC
    # ================================================================
    
    def is_close_game(self) -> bool:
        
        # STEP 5b: Handle secondary stats (assists, rebounds, steals, blocks)
        # Why: Basketball is a team sport - other players contribute to each possession
        # 
        # Assists: When a player makes a shot, a teammate may have assisted
        # TODO: Implement if using EnhancedPossessionResult
        # if 'assist_player_id' in result and result['assist_player_id'] is not None:
        #     offensive_boxscore.loc[result['assist_player_id'], 'AST'] += 1
        
        # Rebounds: When a shot is missed, someone gets the rebound
        # Why: Offensive rebounds extend possession, defensive rebounds end it
        # TODO: Implement rebound logic
        # if 'rebound_player_id' in result and result['rebound_player_id'] is not None:
        #     rebound_player = result['rebound_player_id']
        #     
        #     if result['rebound_type'] == 'offensive':
        #         # Offensive rebound: offensive team keeps possession
        #         offensive_boxscore.loc[rebound_player, 'REB'] += 1
        #         result['possession_changes'] = False  # Override: keep possession!
        #     
        #     elif result['rebound_type'] == 'defensive':
        #         # Defensive rebound: defensive team gets the ball
        #         defensive_boxscore.loc[rebound_player, 'REB'] += 1
        #         result['possession_changes'] = True  # Confirm possession change
        
        # Steals: When turnover occurs, defender may have stolen the ball
        # TODO: Implement if using EnhancedPossessionResult
        # if 'steal_player_id' in result and result['steal_player_id'] is not None:
        #     defensive_boxscore.loc[result['steal_player_id'], 'STL'] += 1
        
        # Blocks: When shot is missed, defender may have blocked it
        # TODO: Implement if using EnhancedPossessionResult
        # if 'block_player_id' in result and result['block_player_id'] is not None:
        #     defensive_boxscore.loc[result['block_player_id'], 'BLK'] += 1
        
        # STEP 6: Determine if possession changes teams
        # Why: Not all plays end a possession (offensive rebounds, some fouls)
        # 
        # Possession changes when:
        # - Made shot (2PT, 3PT, FT) → Other team gets ball
        # - Missed shot with defensive rebound → Other team gets ball
        # - Turnover → Other team gets ball
        # 
        # Possession does NOT change when:
        # - Missed shot with offensive rebound → Same team keeps ball
        # - Shooting foul (FT situation) → Same team shoots FTs then ball changes
        # - Non-shooting foul before bonus → Same team retains possession
        # 
        # The 'possession_changes' flag from the model tells us what to do
        if result['possession_changes']:
            # Switch possession to other team
            self.poss *= -1
            # Why multiply by -1? Because:
            # - If self.poss was 1 (home), it becomes -1 (away)
            # - If self.poss was -1 (away), it becomes 1 (home)
        else:
            # Keep possession with same team
            # This happens after offensive rebounds, non-shooting fouls, etc.
            pass
        
        # STEP 7: Handle fouls (future implementation)
        # TODO: Track fouls per team and player
        # TODO: Implement bonus/penalty situations (free throws)
        # TODO: Handle flagrant fouls, technical fouls
        # 
        # if 'foul' in result:
        #     if self.poss == 1:
        #         self.away_fouls += 1  # Defensive team committed foul
        #         defensive_boxscore.loc[result['foul_player_id'], 'PF'] += 1
        #     else:
        #         self.home_fouls += 1
        #         defensive_boxscore.loc[result['foul_player_id'], 'PF'] += 1
        #     
        #     # Check bonus situation (4+ fouls in quarter = free throws)
        #     if self.away_fouls >= 5 or self.home_fouls >= 5:
        #         # Award free throws (handled by predictor in next possession)
        #         pass
        
        # STEP 8: Check if timeouts needed (for forced substitutions)
        # If team is exhausted but no natural deadball, call timeout
        if self.should_call_timeout('home'):
            self.call_timeout('home')
        if self.should_call_timeout('away'):
            self.call_timeout('away')
        
        # STEP 9: Make substitutions (if deadball)
        self.check_substitutions()
    
    # ================================================================
    # GAME UTILITIES & DISPLAY
    # ================================================================

    def simulate_game(self) -> None:
        """
        Simulate a complete 4-quarter game.
        TODO: Add overtime to simulation
        """
        while self.quarter <= 4:
            self.step()
            if self.quarter_duration <= 0:
                self.quarter += 1
                self.quarter_duration = 12*60
    
    def get_boxscore_display(self, team: str) -> pd.DataFrame:
        """
        Get boxscore with minutes rounded for display.
        
        Minutes are stored as floats during simulation for accuracy,
        but displayed as integers for readability.
        
        Args:
            team: 'home' or 'away'
        
        Returns:
            DataFrame with rounded minutes (original boxscore unchanged)
        
        Usage:
            # Display rounded:
            print(game.get_boxscore_display('home'))
            
            # Access raw data (floats):
            print(game.hometeam_boxscore)
        """
        boxscore = self.hometeam.boxscore if team == 'home' else self.awayteam.boxscore
        
        # Create a copy for display
        display_box = boxscore.copy()
        
        # Round minutes to integers for display
        display_box['MIN'] = display_box['MIN'].round(0).astype(int)
        
        return display_box

    def restart_game(self):
        self.__init__(self.hometeam.team_id, self.awayteam.team_id, self.season)

    def __str__(self) -> str:
        return f"Q{self.quarter} - {self.hometeam.team_abbr} {self.hometeam.score} vs {self.awayteam.team_abbr} {self.awayteam.score}"
