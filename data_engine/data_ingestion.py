"""
Raw data ingestion from NBA API to Snowflake data warehouse.
This module handles only data collection and storage, no transformation.
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
import json

from nba_api.stats.endpoints import (
    playbyplayv2, 
    leaguegamefinder, 
    boxscoretraditionalv2
)

from .config import DataCollectionConfig, SnowflakeConfig
from .snowflake_client import SnowflakeClient

logger = logging.getLogger(__name__)

class NBADataIngestion:
    """Handles raw data collection from NBA API and storage to Snowflake."""
    
    def __init__(
        self, 
        collection_config: DataCollectionConfig,
        snowflake_config: SnowflakeConfig
    ):
        """Initialize data ingestion with configuration."""
        self.collection_config = collection_config
        self.snowflake_config = snowflake_config
        self.snowflake_client = SnowflakeClient(snowflake_config)
        
        # Rate limiting and error handling
        self.base_delay = collection_config.base_delay
        self.current_delay = self.base_delay
        self.consecutive_errors = 0
        self.error_threshold = 3
    
    def retry_api_call(self, func, max_retries: Optional[int] = None, base_delay: Optional[float] = None):
        """Retry an API call with exponential backoff."""
        max_retries = max_retries or self.collection_config.max_retries
        base_delay = base_delay or self.base_delay
        
        for attempt in range(max_retries):
            try:
                result = func()
                self.consecutive_errors = 0
                self.current_delay = max(self.current_delay * 0.9, self.base_delay)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= self.error_threshold:
                        cooldown = 30 + (self.consecutive_errors - self.error_threshold) * 15
                        logger.warning(f"Entering cooldown period: {cooldown} seconds")
                        time.sleep(cooldown)
                        self.consecutive_errors = 0
                    raise
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"API retry {attempt + 1}/{max_retries}: {str(e)[:100]}...")
                time.sleep(delay)
        
        return None
    
    def collect_games_for_season(self, season: str) -> List[str]:
        """Collect all game IDs for a given season."""
        logger.info(f"Collecting games for {season} season...")
        
        try:
            games_df = self.retry_api_call(
                lambda: leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable='Regular Season',
                    timeout=self.collection_config.timeout
                ).get_data_frames()[0]
            )
            
            game_ids = games_df['GAME_ID'].unique().tolist()
            
            if self.collection_config.max_games_per_season:
                game_ids = game_ids[:self.collection_config.max_games_per_season]
                logger.info(f"Limited to {len(game_ids)} games for {season}")
            else:
                logger.info(f"Found {len(game_ids)} games for {season}")
            
            return game_ids
            
        except Exception as e:
            logger.error(f"Error collecting games for {season}: {e}")
            return []
    
    def get_game_metadata(self, game_id: str, season: str) -> Optional[Dict]:
        """Collect game metadata."""
        try:
            # Get basic game info from league game finder
            games_df = self.retry_api_call(
                lambda: leaguegamefinder.LeagueGameFinder(
                    game_id_nullable=game_id,
                    timeout=self.collection_config.timeout
                ).get_data_frames()[0]
            )
            
            if games_df.empty:
                return None
            
            # LeagueGameFinder returns 2 rows per game (one for each team)
            # Need to identify home vs away by MATCHUP format (e.g., "LAL vs. GSW" vs "GSW @ LAL")
            home_row = games_df[games_df['MATCHUP'].str.contains('vs.', case=False, na=False)]
            away_row = games_df[games_df['MATCHUP'].str.contains('@', na=False)]
            
            if home_row.empty or away_row.empty:
                # Fallback: just use the first two rows
                home_row = games_df.iloc[[0]]
                away_row = games_df.iloc[[1]] if len(games_df) > 1 else games_df.iloc[[0]]
            
            home = home_row.iloc[0]
            away = away_row.iloc[0]
            
            return {
                'game_id': game_id,
                'season': season,
                'game_date': pd.to_datetime(home['GAME_DATE']).date(),
                'home_team_id': int(home['TEAM_ID']),
                'home_team_name': home['TEAM_NAME'],
                'away_team_id': int(away['TEAM_ID']),
                'away_team_name': away['TEAM_NAME'],
                'status': home.get('GAME_STATUS_TEXT', 'Unknown')
            }
            
        except Exception as e:
            logger.error(f"Error collecting metadata for game {game_id}: {e}")
            return None
    
    def get_starting_lineups(self, game_id: str) -> Optional[Dict]:
        """Get starting lineups and rosters for a game."""
        try:
            time.sleep(self.current_delay)
            
            player_stats = self.retry_api_call(
                lambda: boxscoretraditionalv2.BoxScoreTraditionalV2(
                    game_id=game_id, 
                    timeout=self.collection_config.timeout
                ).get_data_frames()[0]
            )
            
            if player_stats.empty or 'START_POSITION' not in player_stats.columns:
                return None
            
            # Get unique team IDs
            team_ids = player_stats['TEAM_ID'].unique()
            if len(team_ids) != 2:
                return None
            
            home_team_id = team_ids[0]
            away_team_id = team_ids[1]
            
            # Get starters (players with non-empty START_POSITION)
            starters = player_stats[player_stats['START_POSITION'].notna()]
            
            home_starters = starters[starters['TEAM_ID'] == home_team_id]['PLAYER_ID'].tolist()
            away_starters = starters[starters['TEAM_ID'] == away_team_id]['PLAYER_ID'].tolist()
            
            # Get all players (for roster info)
            home_roster = player_stats[player_stats['TEAM_ID'] == home_team_id]['PLAYER_ID'].tolist()
            away_roster = player_stats[player_stats['TEAM_ID'] == away_team_id]['PLAYER_ID'].tolist()
            
            return {
                'game_id': game_id,
                'home_team_id': int(home_team_id),
                'away_team_id': int(away_team_id),
                'home_starters': home_starters[:5],  # Ensure exactly 5
                'away_starters': away_starters[:5],
                'home_roster': home_roster,
                'away_roster': away_roster
            }
            
        except Exception as e:
            logger.error(f"Error getting lineups for game {game_id}: {e}")
            return None
    
    def get_play_by_play_events(self, game_id: str, season: str) -> List[Dict]:
        """Get raw play-by-play events for a game."""
        try:
            time.sleep(self.current_delay)
            
            events_df = self.retry_api_call(
                lambda: playbyplayv2.PlayByPlayV2(
                    game_id=game_id, 
                    timeout=self.collection_config.timeout
                ).get_data_frames()[0]
            )
            
            if events_df.empty:
                return []
            
            # Convert to list of dictionaries with proper data types
            events = []
            for _, event in events_df.iterrows():
                event_dict = {
                    'game_id': game_id,
                    'season': season,
                    'event_id': f"{game_id}_{event.name}",  # Use pandas index as event_id
                    'event_type': int(event['EVENTMSGTYPE']) if pd.notna(event['EVENTMSGTYPE']) else None,
                    'period': int(event['PERIOD']) if pd.notna(event['PERIOD']) else None,
                    'period_time_remaining': str(event.get('PCTIMESTRING', '')),
                    'event_description': str(event.get('EVENTMSGACTIONTYPE', '')),
                    'home_description': str(event.get('HOMEDESCRIPTION', '')),
                    'visitor_description': str(event.get('VISITORDESCRIPTION', '')),
                    'player1_id': int(event['PLAYER1_ID']) if pd.notna(event['PLAYER1_ID']) else None,
                    'player1_name': str(event.get('PLAYER1_NAME', '')),
                    'player1_team_id': int(event['PLAYER1_TEAM_ID']) if pd.notna(event['PLAYER1_TEAM_ID']) else None,
                    'player2_id': int(event['PLAYER2_ID']) if pd.notna(event['PLAYER2_ID']) else None,
                    'player2_name': str(event.get('PLAYER2_NAME', '')),
                    'player2_team_id': int(event['PLAYER2_TEAM_ID']) if pd.notna(event['PLAYER2_TEAM_ID']) else None,
                    'player3_id': int(event['PLAYER3_ID']) if pd.notna(event['PLAYER3_ID']) else None,
                    'player3_name': str(event.get('PLAYER3_NAME', '')),
                    'player3_team_id': int(event['PLAYER3_TEAM_ID']) if pd.notna(event['PLAYER3_TEAM_ID']) else None,
                    'score_margin': str(event.get('SCOREMARGIN', '0'))
                }
                events.append(event_dict)
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting play-by-play for game {game_id}: {e}")
            return []
    
    def ingest_game_data(self, game_id: str, season: str) -> bool:
        """Ingest all data for a single game."""
        logger.info(f"Ingesting data for game {game_id} ({season})")
        
        try:
            # 1. Get game metadata
            game_metadata = self.get_game_metadata(game_id, season)
            if game_metadata:
                metadata_df = pd.DataFrame([game_metadata])
                self.snowflake_client.upsert_data(
                    metadata_df, 
                    self.collection_config.raw_games_table,
                    ['game_id']
                )
            
            # 2. Get starting lineups
            lineups_data = self.get_starting_lineups(game_id)
            if lineups_data:
                # Convert lists to JSON strings for Snowflake VARIANT columns
                lineups_data['season'] = season
                lineups_data['home_starters'] = json.dumps(lineups_data['home_starters'])
                lineups_data['away_starters'] = json.dumps(lineups_data['away_starters'])
                lineups_data['home_roster'] = json.dumps(lineups_data['home_roster'])
                lineups_data['away_roster'] = json.dumps(lineups_data['away_roster'])
                
                lineups_df = pd.DataFrame([lineups_data])
                self.snowflake_client.upsert_data(
                    lineups_df,
                    self.collection_config.raw_lineups_table,
                    ['game_id']
                )
            
            # 3. Get play-by-play events
            events = self.get_play_by_play_events(game_id, season)
            if events:
                events_df = pd.DataFrame(events)
                self.snowflake_client.upsert_data(
                    events_df,
                    self.collection_config.raw_events_table,
                    ['game_id', 'event_id']
                )
                
                logger.info(f"Successfully ingested {len(events)} events for game {game_id}")
            else:
                logger.warning(f"No events found for game {game_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest data for game {game_id}: {e}")
            return False
    
    def run_ingestion(self, seasons: Optional[List[str]] = None) -> Dict[str, int]:
        """Run the complete data ingestion process."""
        seasons = seasons or self.collection_config.seasons
        
        logger.info(f"Starting data ingestion for seasons: {seasons}")
        
        # Ensure tables exist
        self.snowflake_client.create_raw_tables()
        
        # Get already processed games to avoid duplicates
        processed_games = self.snowflake_client.get_processed_games(seasons)
        logger.info(f"Found {len(processed_games)} already processed games")
        
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        
        for season in seasons:
            logger.info(f"Processing season {season}")
            game_ids = self.collect_games_for_season(season)
            
            for i, game_id in enumerate(game_ids):
                if game_id in processed_games:
                    logger.debug(f"Skipping already processed game {game_id}")
                    results['skipped'] += 1
                    continue
                
                logger.info(f"Processing game {i+1}/{len(game_ids)}: {game_id}")
                
                if self.ingest_game_data(game_id, season):
                    results['success'] += 1
                else:
                    results['failed'] += 1
                
                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i+1}/{len(game_ids)} games for {season}")
        
        logger.info(f"Ingestion complete. Success: {results['success']}, Failed: {results['failed']}, Skipped: {results['skipped']}")
        return results
    
    def get_ingestion_status(self) -> Dict[str, Dict]:
        """Get current ingestion status."""
        return self.snowflake_client.get_collection_status(self.collection_config.seasons)
