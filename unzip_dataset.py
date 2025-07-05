import zipfile
import os
from pathlib import Path

def unzip_dataset():
    """Unzip the downloaded dataset"""
    print("\n📂 Unzipping dataset...")
    
    # Define paths
    zip_path = Path('data/historicalweatherdataforindiancities.zip')
    extract_path = Path('data')
    
    try:
        # Open and extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print("✅ Dataset unzipped successfully!")
            
            # List extracted files
            print("\nExtracted files:")
            for file in extract_path.glob('*'):
                if not file.is_dir():
                    print(f"- {file.name}")
    except Exception as e:
        print(f"❌ Error unzipping dataset: {e}")

if __name__ == "__main__":
    unzip_dataset()
