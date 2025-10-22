"""
NBA Data Engineering Package

This package provides modern data engineering capabilities for the NBA simulation project,
including Snowflake integration, separate data ingestion and transformation pipelines.
"""

from .config import DataCollectionConfig, SnowflakeConfig, DEFAULT_SEASONS
from .snowflake_client import SnowflakeClient
from .data_ingestion import NBADataIngestion
from .data_transformation import NBADataTransformation

__all__ = [
    'DataCollectionConfig',
    'SnowflakeConfig', 
    'SnowflakeClient',
    'NBADataIngestion',
    'NBADataTransformation',
    'DEFAULT_SEASONS'
]
