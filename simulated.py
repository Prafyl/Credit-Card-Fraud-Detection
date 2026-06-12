import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def download_paysim():
    dataset_url = "ealaxi/paysim1"
    data_dir = "paysim_data"
    csv_file = os.path.join(data_dir, "PS_20174392719_1491204439457_log.csv")

    if os.path.exists(csv_file):
        print("✅ PaySim dataset already exists!")
    else:
        print("⬇️ Downloading PaySim (6.3 million rows... this might take a few minutes)...")
        os.makedirs(data_dir, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_url, path=data_dir, unzip=True)
        print("✅ Download complete!")

    print("\n📊 Loading PaySim data...")
    df = pd.read_csv(csv_file)
    
    print("\n" + "="*50)
    print("          PAYSIM DATASET SUMMARY")
    print("="*50)
    print(f"Total rows   : {len(df):,}")
    print(f"Fraud cases  : {(df['isFraud'] == 1).sum():,}")
    print(f"Columns      : {list(df.columns)}")
    print("="*50)
    
    return df

if __name__ == "__main__":
    df_paysim = download_paysim()