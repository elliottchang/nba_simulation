"""
Configuration settings for the NBA data engineering pipeline.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class SnowflakeConfig:
    """Snowflake connection configuration."""
    account: str
    user: str
    warehouse: str = "COMPUTE_WH"
    database: str = "NBA_ANALYTICS"
    schema: str = "RAW"
    role: Optional[str] = None
    
    # Password-based authentication
    password: Optional[str] = None
    
    # Key pair authentication
    private_key_path: Optional[str] = None
    private_key_passphrase: Optional[str] = None
    
    # Alternative authentication methods
    authenticator: Optional[str] = None  # e.g., "externalbrowser"
    
    @classmethod
    def from_env(cls) -> "SnowflakeConfig":
        """Load configuration from environment variables."""
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        user = os.getenv("SNOWFLAKE_USER")
        
        if not account or not user:
            raise ValueError("SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER are required")
        
        return cls(
            account=account,
            user=user,
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            private_key_path=os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH"),
            private_key_passphrase=os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", ""),
            authenticator=os.getenv("SNOWFLAKE_AUTHENTICATOR"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            database=os.getenv("SNOWFLAKE_DATABASE", "NBA_ANALYTICS"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "RAW"),
            role=os.getenv("SNOWFLAKE_ROLE")
        )

@dataclass
class DataCollectionConfig:
    """Configuration for NBA API data collection."""
    seasons: List[str]
    max_games_per_season: Optional[int] = None
    base_delay: float = 1.5
    max_retries: int = 3
    timeout: int = 60
    
    # Table names for different data types (must match Snowflake's actual table names)
    raw_events_table: str = "PLAY_BY_PLAY_EVENTS"
    raw_lineups_table: str = "GAME_LINEUPS" 
    raw_games_table: str = "GAME_METADATA"
    processed_possessions_table: str = "PROCESSED_POSSESSIONS"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.seasons:
            raise ValueError("At least one season must be specified")

# Default configuration
DEFAULT_SEASONS = ['2022-23', '2023-24', '2024-25']
