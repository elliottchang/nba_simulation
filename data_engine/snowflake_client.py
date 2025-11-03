"""
Snowflake data warehouse client for NBA data storage and retrieval.
"""

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from typing import Dict, List, Optional, Union
import logging
from contextlib import contextmanager

from .config import SnowflakeConfig

logger = logging.getLogger(__name__)

class SnowflakeClient:
    """Client for interacting with Snowflake data warehouse."""
    
    def __init__(self, config: SnowflakeConfig):
        """Initialize Snowflake client with connection config."""
        self.config = config
        self._connection = None
    
    def _get_connection(self):
        """Get or create Snowflake connection."""
        if self._connection is None or self._connection.is_closed():
            try:
                # Build connection parameters based on available auth methods
                conn_params = {
                    'account': self.config.account,
                    'user': self.config.user,
                    'warehouse': self.config.warehouse,
                    'database': self.config.database,
                    'schema': self.config.schema
                }
                
                # Add role if specified
                if self.config.role:
                    conn_params['role'] = self.config.role
                
                # Handle different authentication methods
                if self.config.authenticator:
                    conn_params['authenticator'] = self.config.authenticator
                    if self.config.authenticator == "externalbrowser":
                        logger.info("Using external browser authentication - a browser window will open")
                elif self.config.private_key_path:
                    # Key pair authentication
                    from cryptography.hazmat.primitives import serialization
                    with open(self.config.private_key_path, "rb") as key:
                        private_key = serialization.load_pem_private_key(
                            key.read(),
                            password=self.config.private_key_passphrase.encode() if self.config.private_key_passphrase else None
                        )
                    
                    conn_params['private_key'] = private_key
                    logger.info("Using key pair authentication")
                elif self.config.password:
                    # Password authentication
                    conn_params['password'] = self.config.password
                    logger.info("Using password authentication")
                else:
                    raise ValueError("No authentication method configured. Set password, private_key_path, or authenticator.")
                
                self._connection = snowflake.connector.connect(**conn_params)
                logger.info(f"Connected to Snowflake: {self.config.database}.{self.config.schema}")
            except Exception as e:
                logger.error(f"Failed to connect to Snowflake: {e}")
                raise
        return self._connection
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()
    
    def create_raw_tables(self) -> None:
        """Create raw data tables if they don't exist."""
        tables_sql = {
            'PLAY_BY_PLAY_EVENTS': """
                CREATE TABLE IF NOT EXISTS PLAY_BY_PLAY_EVENTS (
                    game_id STRING,
                    season STRING,
                    event_id STRING,
                    event_type INTEGER,
                    period INTEGER,
                    period_time_remaining STRING,
                    event_description STRING,
                    home_description STRING,
                    visitor_description STRING,
                    player1_id INTEGER,
                    player1_name STRING,
                    player1_team_id INTEGER,
                    player2_id INTEGER,
                    player2_name STRING,
                    player2_team_id INTEGER,
                    player3_id INTEGER,
                    player3_name STRING,
                    player3_team_id INTEGER,
                    score_margin STRING,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (game_id, event_id)
                )
            """,
            
            'GAME_LINEUPS': """
                CREATE TABLE IF NOT EXISTS GAME_LINEUPS (
                    game_id STRING,
                    season STRING,
                    home_team_id INTEGER,
                    away_team_id INTEGER,
                    home_starters VARIANT,  -- JSON array of player IDs
                    away_starters VARIANT,  -- JSON array of player IDs
                    home_roster VARIANT,    -- JSON array of all player IDs
                    away_roster VARIANT,    -- JSON array of all player IDs
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (game_id)
                )
            """,
            
            'GAME_METADATA': """
                CREATE TABLE IF NOT EXISTS GAME_METADATA (
                    game_id STRING,
                    season STRING,
                    game_date DATE,
                    home_team_id INTEGER,
                    home_team_name STRING,
                    away_team_id INTEGER,
                    away_team_name STRING,
                    status STRING,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (game_id)
                )
            """,
            
            'PROCESSED_POSSESSIONS': """
                CREATE TABLE IF NOT EXISTS PROCESSED_POSSESSIONS (
                    possession_id STRING,
                    game_id STRING,
                    season STRING,
                    period INTEGER,
                    time_remaining STRING,
                    score_margin INTEGER,
                    offensive_team_id INTEGER,
                    outcome STRING,
                    player_id INTEGER,
                    player_name STRING,
                    offensive_lineup VARIANT,  -- JSON array of player IDs
                    defensive_lineup VARIANT,  -- JSON array of player IDs
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (possession_id)
                )
            """
        }
        
        with self.get_cursor() as cursor:
            for table_name, create_sql in tables_sql.items():
                try:
                    cursor.execute(create_sql)
                    logger.info(f"Created/verified table: {table_name}")
                except Exception as e:
                    logger.error(f"Failed to create table {table_name}: {e}")
                    raise
    
    def upsert_data(
        self, 
        df: pd.DataFrame, 
        table_name: str, 
        primary_key_columns: List[str]
    ) -> None:
        """
        Upsert data to Snowflake table using MERGE to avoid duplicates.
        This properly handles the case where data already exists in the table.
        
        Args:
            df: DataFrame to upsert
            table_name: Target table name
            primary_key_columns: Columns that form the primary key
        """
        if df.empty:
            logger.warning(f"DataFrame is empty, skipping upsert to {table_name}")
            return
        
        try:
            conn = self._get_connection()
            
            # Ensure we're using the correct schema
            with conn.cursor() as cursor:
                cursor.execute(f"USE SCHEMA {self.config.database}.{self.config.schema}")
            
            # Use MERGE for proper upsert behavior
            with conn.cursor() as cursor:
                import json
                
                # Get all column names
                all_columns = df.columns.tolist()
                # Exclude collected_at from updates (it should remain as the original insert time)
                update_columns = [col for col in all_columns if col != 'collected_at']
                
                rows_processed = 0
                
                for _, row in df.iterrows():
                    # Build values for the MERGE statement
                    select_parts = []
                    
                    for col in all_columns:
                        val = row[col]
                        
                        if col.endswith(('_starters', '_roster', '_lineup')):
                            # Handle VARIANT columns with PARSE_JSON
                            if isinstance(val, (list, dict)):
                                json_str = json.dumps(val)
                            elif isinstance(val, str) and val.startswith(('{', '[')):
                                json_str = val
                            else:
                                json_str = '[]'
                            
                            # Escape single quotes for SQL
                            escaped_json = json_str.replace("'", "''")
                            select_parts.append(f"PARSE_JSON('{escaped_json}') AS {col}")
                        else:
                            # Regular columns
                            if pd.isna(val):
                                select_parts.append(f"NULL AS {col}")
                            elif isinstance(val, str):
                                escaped_val = val.replace("'", "''")
                                select_parts.append(f"'{escaped_val}' AS {col}")
                            elif isinstance(val, (int, float)):
                                select_parts.append(f"{val} AS {col}")
                            else:
                                select_parts.append(f"'{str(val)}' AS {col}")
                    
                    # Build merge conditions (match on primary key)
                    merge_conditions = []
                    for pk_col in primary_key_columns:
                        merge_conditions.append(f"t.{pk_col} = s.{pk_col}")
                    
                    # Build update assignments (exclude collected_at and primary key columns)
                    update_assignments = []
                    for col in update_columns:
                        if col not in primary_key_columns:
                            update_assignments.append(f"t.{col} = s.{col}")
                    
                    # Build columns string for INSERT
                    columns_str = ', '.join(all_columns)
                    select_str = ', '.join(select_parts)
                    conditions_str = ' AND '.join(merge_conditions)
                    updates_str = ', '.join(update_assignments) if update_assignments else '1=1'
                    values_str = ', '.join([f"s.{col}" for col in all_columns])
                    
                    # Build the MERGE statement
                    merge_sql = f"""
                        MERGE INTO {table_name} AS t
                        USING (SELECT {select_str}) AS s
                        ON {conditions_str}
                        WHEN MATCHED THEN
                            UPDATE SET {updates_str}
                        WHEN NOT MATCHED THEN
                            INSERT ({columns_str})
                            VALUES ({values_str})
                    """
                    
                    cursor.execute(merge_sql)
                    rows_processed += 1
                
                logger.info(f"Successfully upserted {rows_processed} rows to {table_name} using MERGE")
                
        except Exception as e:
            logger.error(f"Error upserting data to {table_name}: {e}")
            raise
    
    def query_data(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        try:
            conn = self._get_connection()
            return pd.read_sql(sql, conn)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def get_collection_status(self, seasons: List[str]) -> Dict[str, int]:
        """Get data collection status for specified seasons."""
        status_sql = f"""
            SELECT 
                season,
                COUNT(DISTINCT game_id) as games_collected,
                COUNT(*) as total_events
            FROM PLAY_BY_PLAY_EVENTS 
            WHERE season IN ({','.join([f"'{s}'" for s in seasons])})
            GROUP BY season
            ORDER BY season
        """
        
        try:
            result_df = self.query_data(status_sql)
            # Snowflake returns column names in uppercase, normalize to lowercase
            result_df.columns = result_df.columns.str.lower()
            return result_df.set_index('season').to_dict('index')
        except Exception as e:
            logger.warning(f"Could not get collection status: {e}")
            return {}
    
    def get_processed_games(self, seasons: List[str]) -> set:
        """Get set of game IDs that have already been processed."""
        processed_sql = f"""
            SELECT DISTINCT game_id 
            FROM GAME_METADATA 
            WHERE season IN ({','.join([f"'{s}'" for s in seasons])})
        """
        
        try:
            result_df = self.query_data(processed_sql)
            if result_df.empty:
                return set()
            # Snowflake returns column names in uppercase, normalize to lowercase
            result_df.columns = result_df.columns.str.lower()
            return set(result_df['game_id'].tolist())
        except Exception as e:
            logger.warning(f"Could not get processed games: {e}")
            # Return empty set if table doesn't exist or has no data
            return set()
    
    def get_processed_games_by_season(self, seasons: List[str]) -> Dict[str, set]:
        """Get set of game IDs per season that have already been processed.
        
        Returns:
            Dictionary mapping season to set of processed game IDs
        """
        processed_sql = f"""
            SELECT DISTINCT season, game_id 
            FROM GAME_METADATA 
            WHERE season IN ({','.join([f"'{s}'" for s in seasons])})
        """
        
        try:
            result_df = self.query_data(processed_sql)
            if result_df.empty:
                return {season: set() for season in seasons}
            
            # Snowflake returns column names in uppercase by default, so normalize them
            # Convert to lowercase for consistent access
            result_df.columns = result_df.columns.str.lower()
            
            # Group by season
            processed_by_season = {}
            for season in seasons:
                season_games = result_df[result_df['season'] == season]['game_id'].tolist()
                processed_by_season[season] = set(season_games)
            
            # Ensure all seasons are in the dict
            for season in seasons:
                if season not in processed_by_season:
                    processed_by_season[season] = set()
            
            return processed_by_season
        except Exception as e:
            logger.warning(f"Could not get processed games by season: {e}")
            # Return empty sets for all seasons if query fails
            return {season: set() for season in seasons}
    
    def close(self) -> None:
        """Close the database connection."""
        if self._connection and not self._connection.is_closed():
            self._connection.close()
            logger.info("Snowflake connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
