"""
Simple Snowflake connection test to diagnose MFA issues.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_basic_connection():
    """Test basic Snowflake connection with different methods."""
    try:
        import snowflake.connector
        print("✅ Snowflake connector imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import snowflake connector: {e}")
        return False
    
    # Get connection parameters
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    
    print(f"\n📋 Connection parameters:")
    print(f"   Account: {account}")
    print(f"   User: {user}")
    print(f"   Password: {'*' * len(password) if password else 'NOT SET'}")
    
    # Test 1: Basic password authentication
    print(f"\n🔐 Testing password authentication...")
    try:
        conn = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse='COMPUTE_WH',
            database='NBA_ANALYTICS'
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT CURRENT_VERSION()")
        result = cursor.fetchone()
        print(f"✅ Connection successful! Snowflake version: {result[0]}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Password authentication failed: {error_msg}")
        
        # Check if it's an MFA error
        if "Multi-factor authentication" in error_msg:
            print("\n🔍 MFA Error detected. Trying alternative approaches...")
            
            # Test 2: Try with authenticator parameter
            print("\n🌐 Testing external browser authentication...")
            try:
                conn = snowflake.connector.connect(
                    account=account,
                    user=user,
                    authenticator='externalbrowser',
                    warehouse='COMPUTE_WH',
                    database='NBA_ANALYTICS'
                )
                
                cursor = conn.cursor()
                cursor.execute("SELECT CURRENT_VERSION()")
                result = cursor.fetchone()
                print(f"✅ External browser authentication successful! Version: {result[0]}")
                
                cursor.close()
                conn.close()
                return True
                
            except Exception as e2:
                print(f"❌ External browser auth failed: {e2}")
        
        return False

def test_account_info():
    """Test if we can get account information."""
    print(f"\n🔍 Account troubleshooting suggestions:")
    print(f"1. Verify your account identifier format:")
    print(f"   - Current: {os.getenv('SNOWFLAKE_ACCOUNT')}")
    print(f"   - Expected format: abc12345.region.cloud_provider")
    
    print(f"\n2. Check MFA settings in Snowflake:")
    print(f"   - Log into https://app.snowflake.com")
    print(f"   - Go to: Account > Admin > Users & Roles")
    print(f"   - Click on user: {os.getenv('SNOWFLAKE_USER')}")
    print(f"   - Verify 'Enable Multi-Factor Authentication' is UNCHECKED")
    
    print(f"\n3. Alternative - Use Snowsight directly:")
    print(f"   - Your account URL might be: https://{os.getenv('SNOWFLAKE_ACCOUNT')}.snowflakecomputing.com")

if __name__ == "__main__":
    print("="*60)
    print("Snowflake Connection Diagnostic")
    print("="*60)
    
    success = test_basic_connection()
    test_account_info()
    
    if success:
        print(f"\n🎉 Connection test passed! You can proceed with:")
        print(f"   python setup_snowflake_quick.py")
    else:
        print(f"\n❌ Connection test failed. Please check the suggestions above.")
        print(f"\n💡 Quick fix options:")
        print(f"   1. Double-check MFA is disabled in Snowflake web interface")
        print(f"   2. Try logging into Snowsight first: https://app.snowflake.com")
        print(f"   3. Contact Snowflake support if MFA persists")
