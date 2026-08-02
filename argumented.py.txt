import pandas as pd
import numpy as np

def augment_dataset(input_path="creditcardfraud/creditcard.csv", output_path="creditcard_1M.csv", target_rows=1_000_000):
    print(f"📊 Loading original data from {input_path}...")
    df = pd.read_csv(input_path)
    
    original_count = len(df)
    print(f"Original size: {original_count:,} rows")
    
    if original_count >= target_rows:
        print("Already hit target!")
        return df

    rows_needed = target_rows - original_count
    print(f"🛠️ Generating {rows_needed:,} new augmented rows...")

    df_normal = df[df['Class'] == 0]
    df_fraud = df[df['Class'] == 1]
    
    def augment_data(dataframe, num_new_rows):
        new_rows = []
        features = dataframe.drop('Class', axis=1)
        
        for _ in range(num_new_rows):
            base_row = features.sample(1).iloc[0]
            noise = np.random.normal(0, 0.01, size=len(base_row) - 1)
            new_row = base_row.copy()
            new_row[1:] = new_row[1:] + noise
            new_rows.append(new_row)
            
        new_df = pd.DataFrame(new_rows, columns=features.columns)
        new_df['Class'] = dataframe['Class'].values[0]
        return new_df

    frac_normal = len(df_normal) / original_count
    frac_fraud = len(df_fraud) / original_count
    
    new_normal = augment_data(df_normal, int(rows_needed * frac_normal))
    new_fraud = augment_data(df_fraud, int(rows_needed * frac_fraud))
    
    df_augmented = pd.concat([df, new_normal, new_fraud], ignore_index=True)
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    df_augmented.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("          AUGMENTED DATASET SUMMARY")
    print("="*50)
    print(f"Total rows   : {len(df_augmented):,}")
    print(f"Normal       : {(df_augmented['Class'] == 0).sum():,}")
    print(f"Fraud        : {(df_augmented['Class'] == 1).sum():,}")
    print(f"Saved to     : {output_path}")
    print("="*50)
    
    return df_augmented

if __name__ == "__main__":
    df_large = augment_dataset()
