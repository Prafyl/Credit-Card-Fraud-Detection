



import os
import json
import sys
import pandas as pd

# ==========================================
# 1. PASTE YOUR KAGGLE CREDENTIALS HERE
# ==========================================
KAGGLE_USERNAME = "tchutcutchutcu"   # e.g., "johndoe"
KAGGLE_KEY = "KGAT_6bc7f972013f5c9e8e937d45988375ba"         # e.g., "a1b2c3d4e5f6g7h8i9j0"
# ==========================================
def setup_kaggle_credentials():
    """Creates the kaggle.json file in the correct hidden folder."""
    if not KAGGLE_USERNAME or KAGGLE_USERNAME == "paste_your_username_here":
        print("❌ ERROR: You forgot to paste your Kaggle Username and Key at the top of the script!")
        sys.exit(1)

    # Create the ~/.kaggle directory
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)

    # Write the credentials
    credentials = {"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}
    cred_path = os.path.join(kaggle_dir, 'kaggle.json')
    
    with open(cred_path, 'w') as f:
        json.dump(credentials, f)

    print("✅ Kaggle credentials configured successfully!")

def download_and_load_data():
    """Downloads the dataset and loads it into a DataFrame."""
    dataset_url = "mlg-ulb/creditcardfraud"
    data_dir = "creditcardfraud"
    csv_file = os.path.join(data_dir, "creditcard.csv")

    if os.path.exists(csv_file):
        print("✅ Dataset already exists! Loading from local file...")
    else:
        print("⬇️ Downloading dataset from Kaggle (this might take a minute)...")
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Download and unzip
        api.dataset_download_files(dataset_url, path=data_dir, unzip=True)
        print("✅ Download complete!")

    # Load the data
    print("\n📊 Loading data into Pandas DataFrame...")
    df = pd.read_csv(csv_file)
    
    # Display basic info
    print("\n" + "="*50)
    print("          DATASET SUMMARY")
    print("="*50)
    print(f"Total transactions : {len(df):,}")
    print(f"Normal transactions: {(df['Class'] == 0).sum():,}")
    print(f"Fraud transactions : {(df['Class'] == 1).sum():,}")
    print(f"Fraud percentage   : {(df['Class'].mean() * 100):.3f}%")
    print("\nColumns available  :", list(df.columns))
    print("="*50)
    print("\n🔎 First 3 rows of the dataset:")
    print(df.head(3))
    
    return df

if __name__ == "__main__":
    setup_kaggle_credentials()
    df = download_and_load_data()