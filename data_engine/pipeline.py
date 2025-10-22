"""
Main pipeline orchestrator for NBA data engineering operations.
"""

import logging
import os
from typing import List, Optional

from .config import DataCollectionConfig, SnowflakeConfig, DEFAULT_SEASONS
from .data_ingestion import NBADataIngestion
from .data_transformation import NBADataTransformation

logger = logging.getLogger(__name__)

class NBADataPipeline:
    """Orchestrates the complete NBA data engineering pipeline."""
    
    def __init__(
        self,
        seasons: Optional[List[str]] = None,
        max_games_per_season: Optional[int] = None,
        snowflake_config: Optional[SnowflakeConfig] = None
    ):
        """Initialize the data pipeline."""
        self.seasons = seasons or DEFAULT_SEASONS
        
        # Initialize configurations
        self.collection_config = DataCollectionConfig(
            seasons=self.seasons,
            max_games_per_season=max_games_per_season
        )
        
        self.snowflake_config = snowflake_config or SnowflakeConfig.from_env()
        
        # Initialize pipeline components
        self.ingestion = NBADataIngestion(self.collection_config, self.snowflake_config)
        self.transformation = NBADataTransformation(self.collection_config, self.snowflake_config)
    
    def run_full_pipeline(self, skip_ingestion: bool = False) -> dict:
        """
        Run the complete data pipeline: ingestion -> transformation.
        
        Args:
            skip_ingestion: If True, skip data ingestion and only run transformation
            
        Returns:
            Dictionary with pipeline results
        """
        results = {
            'ingestion': None,
            'transformation': None,
            'success': False
        }
        
        try:
            # Step 1: Data Ingestion (if not skipped)
            if not skip_ingestion:
                logger.info("Starting data ingestion phase...")
                ingestion_results = self.ingestion.run_ingestion(self.seasons)
                results['ingestion'] = ingestion_results
                logger.info(f"Ingestion completed: {ingestion_results}")
            else:
                logger.info("Skipping data ingestion phase")
            
            # Step 2: Data Transformation
            logger.info("Starting data transformation phase...")
            transformation_success = self.transformation.run_transformation(self.seasons)
            results['transformation'] = transformation_success
            logger.info(f"Transformation completed: {transformation_success}")
            
            results['success'] = transformation_success
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_ingestion_only(self) -> dict:
        """Run only the data ingestion phase."""
        logger.info("Running ingestion-only pipeline...")
        
        try:
            results = self.ingestion.run_ingestion(self.seasons)
            logger.info(f"Ingestion completed: {results}")
            return {'ingestion': results, 'success': True}
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {'ingestion': None, 'success': False, 'error': str(e)}
    
    def run_transformation_only(self) -> dict:
        """Run only the data transformation phase."""
        logger.info("Running transformation-only pipeline...")
        
        try:
            success = self.transformation.run_transformation(self.seasons)
            logger.info(f"Transformation completed: {success}")
            return {'transformation': success, 'success': success}
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            return {'transformation': False, 'success': False, 'error': str(e)}
    
    def get_pipeline_status(self) -> dict:
        """Get current status of the data pipeline."""
        try:
            ingestion_status = self.ingestion.get_ingestion_status()
            
            # Check if we have processed data
            seasons_list = [f"'{s}'" for s in self.seasons]
            seasons_str = ','.join(seasons_list)
            processed_data = self.transformation.snowflake_client.query_data(
                f"SELECT COUNT(*) as count FROM {self.collection_config.processed_possessions_table}"
                f" WHERE season IN ({seasons_str})"
            )
            
            processed_count = processed_data['count'].iloc[0] if not processed_data.empty else 0
            
            return {
                'ingestion_status': ingestion_status,
                'processed_possessions': processed_count,
                'seasons': self.seasons
            }
        except Exception as e:
            logger.error(f"Could not get pipeline status: {e}")
            return {'error': str(e)}


def setup_logging():
    """Set up logging for the data pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Example usage functions
def run_ingestion_cli():
    """Command line interface for running data ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Data Ingestion')
    parser.add_argument('--seasons', nargs='+', default=DEFAULT_SEASONS,
                       help='Seasons to process (e.g., 2022-23 2023-24)')
    parser.add_argument('--max-games', type=int, default=None,
                       help='Maximum games per season (for testing)')
    
    args = parser.parse_args()
    
    setup_logging()
    
    pipeline = NBADataPipeline(seasons=args.seasons, max_games_per_season=args.max_games)
    results = pipeline.run_ingestion_only()
    
    if results['success']:
        print(f"✅ Ingestion successful: {results['ingestion']}")
    else:
        print(f"❌ Ingestion failed: {results.get('error', 'Unknown error')}")


def run_transformation_cli():
    """Command line interface for running data transformation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Data Transformation')
    parser.add_argument('--seasons', nargs='+', default=DEFAULT_SEASONS,
                       help='Seasons to process (e.g., 2022-23 2023-24)')
    
    args = parser.parse_args()
    
    setup_logging()
    
    pipeline = NBADataPipeline(seasons=args.seasons)
    results = pipeline.run_transformation_only()
    
    if results['success']:
        print(f"✅ Transformation successful")
    else:
        print(f"❌ Transformation failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    # Load environment variables first
    from dotenv import load_dotenv
    load_dotenv()
    
    # Default: run full pipeline
    setup_logging()
    
    pipeline = NBADataPipeline()
    results = pipeline.run_full_pipeline()
    
    if results['success']:
        print("✅ Full pipeline completed successfully!")
        print(f"Results: {results}")
    else:
        print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
