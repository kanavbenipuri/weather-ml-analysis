import os
import json
import webbrowser
from pathlib import Path

def setup_kaggle_api():
    """Set up Kaggle API credentials"""
    print("\n⚙️ Setting up Kaggle API credentials...")
    
    # Check if credentials file exists
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Check if credentials file exists
    cred_file = kaggle_dir / 'kaggle.json'
    
    if not cred_file.exists():
        print("\n⚠️ Kaggle API credentials not found!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/ and sign in")
        print("2. Go to Account Settings")
        print("3. Scroll down to API")
        print("4. Click 'Create New API Token'")
        print("5. A file named 'kaggle.json' will be downloaded")
        print("6. Place this file in your ~/.kaggle/ directory")
        
        # Open browser to Kaggle
        webbrowser.open('https://www.kaggle.com/account')
        
        # Wait for user to complete setup
        input("\nPress Enter after placing the kaggle.json file...")
    
    # Verify credentials
    try:
        with open(cred_file, 'r') as f:
            creds = json.load(f)
            if 'username' in creds and 'key' in creds:
                print("✅ Kaggle API credentials verified!")
                return True
            else:
                print("❌ Invalid kaggle.json file format")
                return False
    except Exception as e:
        print(f"❌ Error reading kaggle.json: {e}")
        return False

def download_dataset():
    """Download the historical weather dataset using Kaggle API"""
    print("\n📥 Downloading dataset...")
    
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Install kaggle package if not installed
        import kaggle
    except ImportError:
        print("Installing Kaggle package...")
        os.system('pip install kaggle')
    
    try:
        # Download dataset
        os.system('kaggle datasets download vanvalkenberg/historicalweatherdataforindiancities -p data/')
        print("✅ Dataset downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Starting dataset download process...")
    
    if setup_kaggle_api():
        if download_dataset():
            print("\n🎉 Dataset download completed!")
        else:
            print("❌ Failed to download dataset")
    else:
        print("❌ Failed to set up Kaggle API credentials")
