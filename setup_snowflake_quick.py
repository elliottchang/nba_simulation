"""
Quick Snowflake setup script to help you get started immediately.
Run this after you've created your Snowflake account and have your credentials.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_env_vars():
    """Check if required environment variables are set."""
    # Basic required variables
    required = ['SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_USER']
    missing = []
    
    for var in required:
        if not os.getenv(var):
            missing.append(var)
    
    # Check authentication method
    password = os.getenv('SNOWFLAKE_PASSWORD')
    authenticator = os.getenv('SNOWFLAKE_AUTHENTICATOR')
    private_key = os.getenv('SNOWFLAKE_PRIVATE_KEY_PATH')
    
    if not password and not authenticator and not private_key:
        missing.append('SNOWFLAKE_PASSWORD (or SNOWFLAKE_AUTHENTICATOR, or SNOWFLAKE_PRIVATE_KEY_PATH)')
    
    if missing:
        print("❌ Missing environment variables:")
        for var in missing:
            print(f"   - {var}")
        print("\nPlease set these in your .env file or environment.")
        print("\nAuthentication options:")
        print("   - SNOWFLAKE_PASSWORD=your_password")
        print("   - SNOWFLAKE_AUTHENTICATOR=externalbrowser")
        print("   - SNOWFLAKE_PRIVATE_KEY_PATH=/path/to/private/key")
        return False
    
    return True

def create_snowflake_resources():
    """Create the necessary Snowflake resources."""
    try:
        from data_engine.snowflake_client import SnowflakeClient
        from data_engine.config import SnowflakeConfig
        
        print("🔌 Connecting to Snowflake...")
        config = SnowflakeConfig.from_env()
        
        with SnowflakeClient(config) as client:
            print("✅ Connected successfully!")
            
            # Create the required resources if they don't exist
            print("\n📋 Setting up Snowflake resources...")
            
            setup_queries = [
                "CREATE DATABASE IF NOT EXISTS NBA_ANALYTICS",
                "USE DATABASE NBA_ANALYTICS",
                """CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH
                   WITH 
                       WAREHOUSE_SIZE = 'XSMALL'
                       AUTO_SUSPEND = 60
                       AUTO_RESUME = TRUE
                       INITIALLY_SUSPENDED = TRUE""",
                "CREATE SCHEMA IF NOT EXISTS RAW",
                "USE SCHEMA NBA_ANALYTICS.RAW"
            ]
            
            with client.get_cursor() as cursor:
                for query in setup_queries:
                    try:
                        cursor.execute(query)
                        query_name = query.split()[2] if len(query.split()) > 2 else "Resource"
                        print(f"   ✅ Created/verified: {query_name}")
                    except Exception as e:
                        if "already exists" in str(e).lower() or "object does not exist" in str(e).lower():
                            print(f"   ✅ Already exists")
                        else:
                            print(f"   ⚠️  Warning: {str(e)[:100]}...")
            
            # Create tables
            print("\n📊 Creating data tables...")
            client.create_raw_tables()
            
            # Test a simple query
            try:
                result = client.query_data("SELECT CURRENT_VERSION() as version")
                if not result.empty and 'version' in result.columns:
                    version_value = result['version'].iloc[0]
                    print(f"   ✅ Snowflake version: {version_value}")
                elif not result.empty:
                    # Try accessing by index if column name doesn't match
                    version_value = result.iloc[0, 0]
                    print(f"   ✅ Snowflake version: {version_value}")
                else:
                    print("   ✅ Connection test successful (version query returned empty)")
            except Exception as e:
                print(f"   ⚠️  Connection test had minor issue: {str(e)[:80]}...")
            
            print("\n🎉 Snowflake setup complete!")
            return True
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

def test_data_pipeline():
    """Test the data pipeline with minimal data."""
    try:
        from data_engine.pipeline import NBADataPipeline
        from dotenv import load_dotenv
        
        load_dotenv()
        
        print("\n🧪 Testing data pipeline with minimal data...")
        
        # Very small test - just 2 games from one season
        pipeline = NBADataPipeline(
            seasons=['2024-25'],
            max_games_per_season=2  # Very small for testing
        )
        
        print("   Starting small data collection test...")
        results = pipeline.run_ingestion_only()
        
        if results['success']:
            print(f"   ✅ Test successful! {results['ingestion']}")
            return True
        else:
            print(f"   ⚠️  Test had issues: {results}")
            return False
            
    except Exception as e:
        print(f"   ⚠️  Test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("="*60)
    print("NBA Simulation - Quick Snowflake Setup")
    print("="*60)
    
    # Step 1: Check environment
    print("\n1. Checking environment variables...")
    if not check_env_vars():
        print("\nPlease create a .env file with your Snowflake credentials:")
        print("SNOWFLAKE_ACCOUNT=your_account.region")
        print("SNOWFLAKE_USER=your_username")
        print("SNOWFLAKE_PASSWORD=your_password")
        print("\nSee SNOWFLAKE_SETUP_GUIDE.md for detailed instructions.")
        return
    
    # Step 2: Set up Snowflake resources
    print("\n2. Setting up Snowflake resources...")
    if not create_snowflake_resources():
        print("❌ Failed to set up Snowflake resources. Check your credentials and network connection.")
        return
    
    # Step 3: Test pipeline (optional)
    print("\n3. Testing data pipeline...")
    user_input = input("Run a small data collection test? (y/n): ").lower().strip()
    
    if user_input in ['y', 'yes']:
        test_success = test_data_pipeline()
        if test_success:
            print("\n🎉 Everything is working! You can now run the full pipeline.")
        else:
            print("\n⚠️  Setup complete, but data test had issues. You can still try the full pipeline.")
    else:
        print("\n⏭️  Skipping data test.")
    
    print("\n" + "="*60)
    print("🚀 Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run full data collection: python -m data_engine.pipeline")
    print("2. Train your model: python train_with_data_engine.py")
    print("3. See SNOWFLAKE_SETUP_GUIDE.md for more details")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check SNOWFLAKE_SETUP_GUIDE.md for troubleshooting help.")
