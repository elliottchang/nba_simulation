"""
Test external browser authentication with passkey MFA.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_external_browser_auth():
    """Test external browser authentication."""
    try:
        import snowflake.connector
        
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        user = os.getenv("SNOWFLAKE_USER")
        
        print("🌐 Testing external browser authentication...")
        print("A browser window should open for you to authenticate with your passkey.")
        print("Please authenticate when the browser opens.")
        
        # Try external browser authentication
        conn = snowflake.connector.connect(
            account=account,
            user=user,
            authenticator='externalbrowser'
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT CURRENT_USER(), CURRENT_ROLE()")
        result = cursor.fetchone()
        
        print(f"✅ Authentication successful!")
        print(f"   User: {result[0]}")
        print(f"   Role: {result[1]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ External browser auth failed: {error_msg}")
        
        if "SAML Identity Provider" in error_msg:
            print("\n💡 SAML issue detected. Let's try a different approach...")
            return try_alternative_auth()
        
        return False

def try_alternative_auth():
    """Try alternative authentication methods."""
    print("\n🔧 Alternative authentication options:")
    print("1. Contact your Snowflake admin to set up application-specific authentication")
    print("2. Use key pair authentication (more complex but works with MFA)")
    print("3. Ask admin to create a service user without MFA for this application")
    
    return False

if __name__ == "__main__":
    print("="*60)
    print("Testing Passkey MFA Authentication")
    print("="*60)
    
    success = test_external_browser_auth()
    
    if success:
        print(f"\n🎉 Authentication successful! You can now run:")
        print(f"   python setup_snowflake_quick.py")
    else:
        print(f"\n⚠️  Authentication failed. See alternative options above.")
