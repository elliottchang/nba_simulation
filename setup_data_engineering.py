"""
Setup script for the NBA simulation data engineering infrastructure.
This script helps users set up Snowflake connection and run the initial data pipeline.
"""

import os
import sys
import logging
from typing import Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_engine.config import SnowflakeConfig, DataCollectionConfig, DEFAULT_SEASONS
from data_engine.pipeline import NBADataPipeline

def check_environment_setup() -> bool:
    """Check if required environment variables are set."""
    required_vars = ['SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease create a .env file with your Snowflake credentials:")
        print("   SNOWFLAKE_ACCOUNT=your_account.region"
        print("   SNOWFLAKE_USER=your_username")
        print("   SNOWFLAKE_PASSWORD=your_password")
        print("\nSee env_example.txt for the complete template.")
        return False
    
    print("✅ Environment variables are set")
    return True

def test_snowflake_connection() -> bool:
    """Test connection to Snowflake."""
    try:
        from data_engine.snowflake_client import SnowflakeClient
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        config = SnowflakeConfig.from_env()
        
        print("🔌 Testing Snowflake connection...")
        with SnowflakeClient(config) as client:
            # Simple query to test connection
            result = client.query_data("SELECT CURRENT_VERSION() as version")
            if not result.empty:
                print(f"✅ Connected to Snowflake successfully")
                print(f"   Version: {result['version'].iloc[0]}")
                return True
            else:
                print("❌ Snowflake connection failed - no response")
                return False
                
    except Exception as e:
        print(f"❌ Snowflake connection failed: {e}")
        return False

def setup_snowflake_tables() -> bool:
    """Set up required Snowflake tables."""
    try:
        from data_engine.snowflake_client import SnowflakeClient
        from dotenv import load_dotenv
        
        load_dotenv()
        config = SnowflakeConfig.from_env()
        
        print("📋 Setting up Snowflake tables...")
        with SnowflakeClient(config) as client:
            client.create_raw_tables()
            print("✅ Snowflake tables created successfully")
            return True
            
    except Exception as e:
        print(f"❌ Failed to create Snowflake tables: {e}")
        return False

def run_initial_data_pipeline(
    seasons: Optional[list] = None,
    max_games_per_season: Optional[int] = 5  # Conservative default for testing
) -> bool:
    """Run the initial data pipeline to populate Snowflake with test data."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        seasons = seasons or ['2023-24']  # Use just one recent season for initial setup
        
        print(f"🚀 Running initial data pipeline for seasons: {seasons}")
        print(f"   Max games per season: {max_games_per_season}")
        
        pipeline = NBADataPipeline(
            seasons=seasons,
            max_games_per_season=max_games_per_season
        )
        
        results = pipeline.run_full_pipeline()
        
        if results['success']:
            print("✅ Initial data pipeline completed successfully!")
            print(f"   Ingestion: {results.get('ingestion', 'skipped')}")
            print(f"   Transformation: {results.get('transformation', False)}")
            return True
        else:
            print(f"❌ Data pipeline failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run data pipeline: {e}")
        return False

def install_dependencies() -> bool:
    """Check and install required Python packages."""
    try:
        import subprocess
        import importlib
        
        required_packages = [
            'torch',
            'pandas', 
            'numpy',
            'nba-api',
            'snowflake-connector-python',
            'python-dotenv'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"📦 Installing missing packages: {missing_packages}")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', *missing_packages
            ])
            print("✅ Dependencies installed successfully")
        else:
            print("✅ All required dependencies are already installed")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main setup function."""
    print("="*80)
    print("NBA Simulation - Data Engineering Setup")
    print("="*80)
    
    # Step 1: Install dependencies
    print("\n1. Checking dependencies...")
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return
    
    # Step 2: Check environment setup
    print("\n2. Checking environment setup...")
    if not check_environment_setup():
        print("❌ Setup failed - please configure environment variables")
        return
    
    # Step 3: Test Snowflake connection
    print("\n3. Testing Snowflake connection...")
    if not test_snowflake_connection():
        print("❌ Setup failed - could not connect to Snowflake")
        return
    
    # Step 4: Set up tables
    print("\n4. Setting up Snowflake tables...")
    if not setup_snowflake_tables():
        print("❌ Setup failed - could not create tables")
        return
    
    # Step 5: Run initial data pipeline (optional)
    print("\n5. Initial data pipeline...")
    user_input = input("Would you like to run a small test data collection? (y/n): ").lower().strip()
    
    if user_input in ['y', 'yes']:
        try:
            max_games = input("Max games per season for testing (default 5): ").strip()
            max_games = int(max_games) if max_games else 5
        except ValueError:
            max_games = 5
        
        if not run_initial_data_pipeline(max_games_per_season=max_games):
            print("⚠️  Data pipeline failed, but setup is complete. You can run it manually later.")
    else:
        print("Skipping data pipeline. You can run it manually with:")
        print("   python -m data_engine.pipeline")
    
    print("\n" + "="*80)
    print("🎉 Setup complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run data ingestion:     python -m data_engine.pipeline")
    print("2. Run ML training:        python train_with_data_engine.py")
    print("3. Use modern predictor:   from modern_player_aware_predictor import create_predictor")
    print("\nFor more information, see the updated README.md")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
