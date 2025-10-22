"""
Data transformation pipeline that processes raw NBA data into features for ML models.
This runs independently of data ingestion and can be re-run with different transformation logic.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from .config import DataCollectionConfig, SnowflakeConfig
from .snowflake_client import SnowflakeClient

logger = logging.getLogger(__name__)

class NBADataTransformation:
    """Transforms raw NBA data into processed features for ML models."""
    
    def __init__(
        self,
        collection_config: DataCollectionConfig,
        snowflake_config: SnowflakeConfig
    ):
        """Initialize transformation pipeline."""
        self.collection_config = collection_config
        self.snowflake_config = snowflake_config
        self.snowflake_client = SnowflakeClient(snowflake_config)
        
        # Feature engineering state
        self.player_season_stats = {}
        self.player_to_idx = {}
        self.outcome_to_idx = {}
    
    def load_raw_data(self, seasons: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw data from Snowflake for processing."""
        logger.info(f"Loading raw data for seasons: {seasons}")
        
        seasons_str = ','.join([f"'{s}'" for s in seasons])
        
        # Load events data
        events_sql = f"""
            SELECT * FROM {self.collection_config.raw_events_table}
            WHERE season IN ({seasons_str})
            ORDER BY game_id, period, period_time_remaining
        """
        events_df = self.snowflake_client.query_data(events_sql)
        logger.info(f"Loaded {len(events_df)} raw events")
        
        # Load lineup data
        lineups_sql = f"""
            SELECT * FROM {self.collection_config.raw_lineups_table}
            WHERE season IN ({seasons_str})
        """
        lineups_df = self.snowflake_client.query_data(lineups_sql)
        logger.info(f"Loaded {len(lineups_df)} lineup records")
        
        # Load game metadata
        games_sql = f"""
            SELECT * FROM {self.collection_config.raw_games_table}
            WHERE season IN ({seasons_str})
        """
        games_df = self.snowflake_client.query_data(games_sql)
        logger.info(f"Loaded {len(games_df)} game records")
        
        return events_df, lineups_df, games_df
    
    def parse_possession_outcome(self, events: pd.DataFrame, poss_events: pd.DataFrame) -> Optional[Dict]:
        """
        Parse a possession to determine outcome and player who ended it.
        This is the same logic as in the original trainer, but now works with DataFrame inputs.
        """
        if len(poss_events) == 0:
            return None
        
        # Get the last event and check for fouls in the possession
        last_event = poss_events.iloc[-1]
        event_type = last_event['event_type']
        
        # Helper function to check if shot is 3PT
        def is_3pt(event):
            desc = str(event.get('home_description', '')) + str(event.get('visitor_description', ''))
            return '3PT' in desc
        
        # Check for fouls in the possession
        has_foul = any(poss_events['event_type'] == 6)
        
        # Check for made shots in the possession  
        made_shot_events = poss_events[poss_events['event_type'] == 1]
        has_made_shot = len(made_shot_events) > 0
        
        # Special case: Foul after made shot = And-one
        if event_type == 6 and has_made_shot:
            made_shot_event = made_shot_events.iloc[-1]
            is_three = is_3pt(made_shot_event)
            
            # Check if this is a defensive foul (not offensive)
            foul_desc = str(last_event.get('home_description', '')) + str(last_event.get('visitor_description', ''))
            if 'OFF.FOUL' not in foul_desc.upper() and 'OFFENSIVE' not in foul_desc.upper():
                outcome = '3pt_andone' if is_three else '2pt_andone'
                return {
                    'outcome': outcome,
                    'player_id': made_shot_event.get('player1_id'),
                    'player_name': made_shot_event.get('player1_name', ''),
                    'team_id': made_shot_event.get('player1_team_id')
                }
        
        # Determine outcome based on event sequence
        if event_type == 1:  # Made shot (no foul after)
            is_three = is_3pt(last_event)
            outcome = '3pt_make' if is_three else '2pt_make'
                
        elif event_type == 2:  # Missed shot
            is_three = is_3pt(last_event)
            
            if has_foul:
                # Check if it's a shooting foul
                foul_events = poss_events[poss_events['event_type'] == 6]
                for _, foul_event in foul_events.iterrows():
                    foul_desc = str(foul_event.get('home_description', '')) + str(foul_event.get('visitor_description', ''))
                    if 'SHOOTING' in foul_desc.upper() or 'S.FOUL' in foul_desc.upper():
                        outcome = '3pt_foul' if is_three else '2pt_foul'
                        break
                else:
                    outcome = '3pt_miss' if is_three else '2pt_miss'
            else:
                outcome = '3pt_miss' if is_three else '2pt_miss'
                
        elif event_type == 5:  # Turnover
            to_desc = str(last_event.get('home_description', '')) + str(last_event.get('visitor_description', ''))
            if 'OFF.FOUL' in to_desc.upper() or 'OFFENSIVE FOUL' in to_desc.upper() or 'CHARGE' in to_desc.upper():
                outcome = 'off_foul'
            else:
                outcome = 'TO'
                
        elif event_type == 6:  # Foul
            foul_desc = str(last_event.get('home_description', '')) + str(last_event.get('visitor_description', ''))
            
            if 'OFF.FOUL' in foul_desc.upper() or 'OFFENSIVE' in foul_desc.upper() or 'CHARGE' in foul_desc.upper():
                outcome = 'off_foul'
            elif 'SHOOTING' in foul_desc.upper() or 'S.FOUL' in foul_desc.upper():
                if len(poss_events) > 1:
                    prev_event = poss_events.iloc[-2]
                    if prev_event['event_type'] == 2:
                        is_three = is_3pt(prev_event)
                        outcome = '3pt_foul' if is_three else '2pt_foul'
                    else:
                        outcome = '2pt_foul'
                else:
                    outcome = '2pt_foul'
            else:
                outcome = 'def_foul'
        else:
            return None
        
        return {
            'outcome': outcome,
            'player_id': last_event.get('player1_id'),
            'player_name': last_event.get('player1_name', ''),
            'team_id': last_event.get('player1_team_id')
        }
    
    def extract_possessions_from_events(
        self, 
        events_df: pd.DataFrame, 
        lineups_df: pd.DataFrame
    ) -> List[Dict]:
        """Extract possession-level data from raw events."""
        logger.info("Extracting possessions from raw events data")
        
        possessions = []
        
        # Group events by game
        for game_id, game_events in events_df.groupby('game_id'):
            logger.debug(f"Processing possessions for game {game_id}")
            
            # Get lineup data for this game
            game_lineups = lineups_df[lineups_df['game_id'] == game_id]
            if game_lineups.empty:
                logger.warning(f"No lineup data found for game {game_id}")
                continue
            
            lineup_row = game_lineups.iloc[0]
            
            # Parse JSON lineup data
            try:
                home_starters = json.loads(lineup_row['home_starters'])
                away_starters = json.loads(lineup_row['away_starters'])
                home_team_id = lineup_row['home_team_id']
                away_team_id = lineup_row['away_team_id']
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not parse lineup data for game {game_id}: {e}")
                continue
            
            # Initialize lineups - convert to sets for easy modification
            home_lineup = set(home_starters)
            away_lineup = set(away_starters)
            
            current_possession = []
            
            # Sort events by period and time
            game_events = game_events.sort_values(['period', 'period_time_remaining'])
            
            for idx, event in game_events.iterrows():
                event_type = event['event_type']
                
                # Track substitutions (event type 8)
                if event_type == 8:
                    player_in = event.get('player1_id')
                    player_out = event.get('player2_id')
                    team_id = event.get('player1_team_id')
                    
                    if pd.notna(player_in) and pd.notna(player_out) and pd.notna(team_id):
                        if team_id == home_team_id:
                            if player_out in home_lineup:
                                home_lineup.remove(player_out)
                            home_lineup.add(player_in)
                        elif team_id == away_team_id:
                            if player_out in away_lineup:
                                away_lineup.remove(player_out)
                            away_lineup.add(player_in)
                
                # Events that end possessions
                if event_type in [1, 2, 5, 6]:  # Made shot, missed shot, turnover, foul
                    current_possession.append(event)
                    
                    # Special case: If this is a made shot, check if next event is a foul (and-one)
                    should_continue = False
                    if (event_type == 1 and 
                        idx < len(game_events) - 1 and 
                        game_events.iloc[idx + 1]['event_type'] == 6):
                        should_continue = True
                    
                    if should_continue:
                        continue
                    
                    # Process the possession
                    if len(current_possession) > 0:
                        poss_df = pd.DataFrame(current_possession)
                        outcome = self.parse_possession_outcome(game_events, poss_df)
                        
                        if outcome and outcome['player_id'] is not None and outcome['outcome'] is not None:
                            # Determine offensive/defensive teams
                            is_home_offense = outcome['team_id'] == home_team_id
                            
                            # Get current lineups (ensure exactly 5 players)
                            current_home = list(home_lineup)[:5]
                            current_away = list(away_lineup)[:5]
                            
                            # Skip if lineups are incomplete
                            if len(current_home) < 5 or len(current_away) < 5:
                                current_possession = []
                                continue
                            
                            offensive_lineup = current_home if is_home_offense else current_away
                            defensive_lineup = current_away if is_home_offense else current_home
                            
                            # Get score differential
                            score_margin = event.get('score_margin', '0')
                            try:
                                score_margin = int(score_margin) if score_margin not in ['TIE', None, ''] else 0
                            except:
                                score_margin = 0
                            
                            possession_record = {
                                'possession_id': f"{game_id}_{len(possessions)}",
                                'game_id': game_id,
                                'season': event['season'],
                                'period': event['period'],
                                'time_remaining': event['period_time_remaining'],
                                'score_margin': score_margin,
                                'offensive_team_id': outcome['team_id'],
                                'outcome': outcome['outcome'],
                                'player_id': outcome['player_id'],
                                'player_name': outcome['player_name'],
                                'offensive_lineup': json.dumps(offensive_lineup),
                                'defensive_lineup': json.dumps(defensive_lineup),
                                'processed_at': datetime.now()
                            }
                            possessions.append(possession_record)
                    
                    # Reset for next possession
                    current_possession = []
        
        logger.info(f"Extracted {len(possessions)} possessions")
        return possessions
    
    def compute_player_season_stats(self, possessions_df: pd.DataFrame) -> Dict:
        """Compute aggregate statistics for each player-season pair."""
        logger.info("Computing player season statistics")
        
        stats = defaultdict(lambda: {
            'possessions': 0,
            '2pt_attempts': 0,
            '2pt_makes': 0,
            '3pt_attempts': 0,
            '3pt_makes': 0,
            'turnovers': 0,
            'fouls_drawn': 0,
            'fouls_committed': 0,
            'andones': 0,
        })
        
        for _, row in possessions_df.iterrows():
            key = (int(row['player_id']), row['season'])
            stats[key]['possessions'] += 1
            
            outcome = row['outcome']
            
            # 2-point attempts and makes
            if outcome in ['2pt_make', '2pt_andone']:
                stats[key]['2pt_attempts'] += 1
                stats[key]['2pt_makes'] += 1
                if outcome == '2pt_andone':
                    stats[key]['andones'] += 1
                    stats[key]['fouls_drawn'] += 1
            elif outcome in ['2pt_miss', '2pt_foul']:
                stats[key]['2pt_attempts'] += 1
                if outcome == '2pt_foul':
                    stats[key]['fouls_drawn'] += 1
            
            # 3-point attempts and makes
            elif outcome in ['3pt_make', '3pt_andone']:
                stats[key]['3pt_attempts'] += 1
                stats[key]['3pt_makes'] += 1
                if outcome == '3pt_andone':
                    stats[key]['andones'] += 1
                    stats[key]['fouls_drawn'] += 1
            elif outcome in ['3pt_miss', '3pt_foul']:
                stats[key]['3pt_attempts'] += 1
                if outcome == '3pt_foul':
                    stats[key]['fouls_drawn'] += 1
            
            # Turnovers
            elif outcome == 'TO':
                stats[key]['turnovers'] += 1
            
            # Fouls
            elif outcome == 'off_foul':
                stats[key]['fouls_committed'] += 1
                stats[key]['turnovers'] += 1  # Offensive fouls are turnovers
        
        # Compute rates and percentages
        for key, stat_dict in stats.items():
            poss = max(stat_dict['possessions'], 1)
            stat_dict['usage_rate'] = stat_dict['possessions'] / poss
            stat_dict['2pt_rate'] = stat_dict['2pt_attempts'] / poss
            stat_dict['3pt_rate'] = stat_dict['3pt_attempts'] / poss
            stat_dict['to_rate'] = stat_dict['turnovers'] / poss
            
            # Shooting percentages
            if stat_dict['2pt_attempts'] > 0:
                stat_dict['2pt_pct'] = stat_dict['2pt_makes'] / stat_dict['2pt_attempts']
            else:
                stat_dict['2pt_pct'] = 0.0
                
            if stat_dict['3pt_attempts'] > 0:
                stat_dict['3pt_pct'] = stat_dict['3pt_makes'] / stat_dict['3pt_attempts']
            else:
                stat_dict['3pt_pct'] = 0.0
            
            # Additional rates
            stat_dict['foul_drawn_rate'] = stat_dict['fouls_drawn'] / poss
            stat_dict['foul_commit_rate'] = stat_dict['fouls_committed'] / poss
            stat_dict['andone_rate'] = stat_dict['andones'] / poss
        
        self.player_season_stats = dict(stats)
        logger.info(f"Computed stats for {len(self.player_season_stats)} player-season pairs")
        return self.player_season_stats
    
    def run_transformation(self, seasons: List[str]) -> bool:
        """Run the complete data transformation pipeline."""
        logger.info(f"Starting data transformation for seasons: {seasons}")
        
        try:
            # 1. Load raw data from Snowflake
            events_df, lineups_df, games_df = self.load_raw_data(seasons)
            
            if events_df.empty:
                logger.warning("No events data found to transform")
                return False
            
            # 2. Extract possessions from raw events
            possessions_data = self.extract_possessions_from_events(events_df, lineups_df)
            
            if not possessions_data:
                logger.warning("No possessions extracted from events data")
                return False
            
            # 3. Convert to DataFrame and store
            possessions_df = pd.DataFrame(possessions_data)
            
            # Store processed possessions back to Snowflake
            self.snowflake_client.upsert_data(
                possessions_df,
                self.collection_config.processed_possessions_table,
                ['possession_id']
            )
            
            logger.info(f"Stored {len(possessions_df)} processed possessions to Snowflake")
            
            # 4. Compute and store player statistics (could be stored in separate table)
            player_stats = self.compute_player_season_stats(possessions_df)
            
            # You could store player_stats to a separate table here if needed
            # For now, we'll just log the completion
            
            logger.info("Data transformation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            return False
    
    def get_transformed_data(self, seasons: List[str]) -> pd.DataFrame:
        """Load processed possessions data from Snowflake."""
        seasons_str = ','.join([f"'{s}'" for s in seasons])
        
        sql = f"""
            SELECT * FROM {self.collection_config.processed_possessions_table}
            WHERE season IN ({seasons_str})
            ORDER BY game_id, period, time_remaining
        """
        
        return self.snowflake_client.query_data(sql)
