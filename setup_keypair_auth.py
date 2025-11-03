"""
Setup script for key pair authentication with Snowflake.
This bypasses MFA for programmatic connections.
"""

import os
import subprocess
import sys
from pathlib import Path

def generate_key_pair():
    """Generate a key pair for Snowflake authentication."""
    home = Path.home()
    snowflake_dir = home / ".snowflake"
    
    # Create .snowflake directory if it doesn't exist
    snowflake_dir.mkdir(exist_ok=True)
    
    private_key_path = snowflake_dir / "rsa_key.p8"
    public_key_path = snowflake_dir / "rsa_key.pub"
    
    if private_key_path.exists():
        print(f"⚠️  Key pair already exists at {snowflake_dir}")
        return str(private_key_path), str(public_key_path)
    
    try:
        print("🔑 Generating RSA key pair for Snowflake authentication...")
        
        # Generate private key
        subprocess.run([
            "openssl", "genrsa", "2048"
        ], stdout=subprocess.PIPE, check=True)
        
        result = subprocess.run([
            "openssl", "genrsa", "2048"
        ], stdout=subprocess.PIPE, check=True)
        
        with open(private_key_path, "wb") as f:
            f.write(result.stdout)
        
        # Convert to PKCS8 format
        result = subprocess.run([
            "openssl", "pkcs8", "-topk8", "-inform", "PEM", "-outform", "PEM", "-nocrypt"
        ], input=result.stdout, stdout=subprocess.PIPE, check=True)
        
        with open(private_key_path, "wb") as f:
            f.write(result.stdout)
        
        # Generate public key
        result = subprocess.run([
            "openssl", "rsa", "-in", str(private_key_path), "-pubout"
        ], stdout=subprocess.PIPE, check=True)
        
        with open(public_key_path, "wb") as f:
            f.write(result.stdout)
        
        print(f"✅ Key pair generated successfully!")
        print(f"   Private key: {private_key_path}")
        print(f"   Public key:  {public_key_path}")
        
        return str(private_key_path), str(public_key_path)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate key pair: {e}")
        print("Make sure OpenSSL is installed on your system.")
        return None, None
    except FileNotFoundError:
        print("❌ OpenSSL not found. Please install OpenSSL or use Homebrew:")
        print("   brew install openssl")
        return None, None

def get_public_key_content(public_key_path):
    """Extract public key content for Snowflake."""
    try:
        with open(public_key_path, 'r') as f:
            content = f.read()
        
        # Remove header and footer lines and newlines
        lines = content.strip().split('\n')
        key_lines = [line for line in lines if not line.startswith('-----')]
        return ''.join(key_lines)
        
    except Exception as e:
        print(f"❌ Failed to read public key: {e}")
        return None

def update_env_file(private_key_path):
    """Update .env file with key pair authentication."""
    try:
        env_path = Path(".env")
        if not env_path.exists():
            print("❌ .env file not found")
            return False
        
        # Read current .env
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Remove old auth methods and add key pair auth
        updated_lines = []
        skip_next = False
        
        for line in lines:
            line = line.strip()
            
            # Skip password and authenticator lines
            if line.startswith('SNOWFLAKE_PASSWORD=') or line.startswith('SNOWFLAKE_AUTHENTICATOR='):
                continue
            
            # Add key pair path after user line
            if line.startswith('SNOWFLAKE_USER='):
                updated_lines.append(line + '\n')
                updated_lines.append(f'SNOWFLAKE_PRIVATE_KEY_PATH={private_key_path}\n')
                updated_lines.append('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE=\n')
            else:
                updated_lines.append(line + '\n' if not line.endswith('\n') else line)
        
        # Write back
        with open(env_path, 'w') as f:
            f.writelines(updated_lines)
        
        print(f"✅ Updated .env file with key pair authentication")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update .env file: {e}")
        return False

def main():
    print("="*60)
    print("Snowflake Key Pair Authentication Setup")
    print("="*60)
    
    # Generate key pair
    private_key_path, public_key_path = generate_key_pair()
    
    if not private_key_path:
        print("❌ Setup failed - could not generate key pair")
        return
    
    # Get public key content
    public_key_content = get_public_key_content(public_key_path)
    if not public_key_content:
        print("❌ Setup failed - could not read public key")
        return
    
    print(f"\n📋 Next steps:")
    print(f"1. Copy this public key content:")
    print(f"   {public_key_content}")
    print(f"\n2. In Snowflake, run this SQL command:")
    print(f"   ALTER USER {os.getenv('SNOWFLAKE_USER', 'YOUR_USERNAME')} SET RSA_PUBLIC_KEY='{public_key_content}';")
    
    # Update .env file
    if update_env_file(private_key_path):
        print(f"\n3. Test the connection:")
        print(f"   python setup_snowflake_quick.py")
    
    print(f"\n⚠️  Important: Keep your private key secure!")
    print(f"   Never commit it to version control.")

if __name__ == "__main__":
    main()
