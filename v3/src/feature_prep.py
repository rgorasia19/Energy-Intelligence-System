import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_data(pca_path, raw_path):
    """
    Load PCA features and merge with core demand signals.
    """
    # Load PCA data
    df_pca = pd.read_parquet(pca_path)
    df_pca.index = pd.to_datetime(df_pca.index)
    
    # Load raw data for demand signals
    df_raw = pd.read_csv(raw_path)
    df_raw['DATETIME'] = pd.to_datetime(df_raw['DATETIME'])
    df_raw.set_index('DATETIME', inplace=True)
    
    # Merge
    demand_cols = ['ND', 'TSD', 'ENGLAND_WALES_DEMAND']
    available_cols = [c for c in demand_cols if c in df_raw.columns]
    df_demand = df_raw[available_cols]
    
    # Join on index (DATETIME)
    df = df_pca.join(df_demand, how='inner')
    df.sort_index(inplace=True)
    
    return df, available_cols

def create_features(df, demand_cols):
    """
    Create temporal encodings, lags, and rolling statistics.
    """
    # 2. Temporal encoding
    df['HOUR'] = df.index.hour
    df['DAY_OF_WEEK'] = df.index.dayofweek
    df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
    df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
    df['DOW_SIN'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
    df['DOW_COS'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)
    df.drop(columns=['HOUR', 'DAY_OF_WEEK'], inplace=True)
    
    # 3. Lag structure
    # Applying to all features currently in the dataframe (PCA + demand + temporal)
    lags = [1, 2, 3, 24, 48]
    base_features = df.columns.tolist()
    
    for lag in lags:
        shifted = df[base_features].shift(lag)
        shifted.columns = [f"{c}_LAG_{lag}" for c in base_features]
        df = pd.concat([df, shifted], axis=1)
        
    # 4. Rolling statistics
    # 24h = 48 half-hour periods. Shift(1) ensures no target leakage
    for col in demand_cols:
        df[f"{col}_ROLL_MEAN_24h"] = df[col].shift(1).rolling(window=48).mean()
        df[f"{col}_ROLL_STD_24h"] = df[col].shift(1).rolling(window=48).std()
        
    # Define aligned target (next-step horizon)
    df['TARGET_ND'] = df['ND'].shift(-1)
    
    # Handle NaNs explicitly
    df.dropna(inplace=True)
    
    # Cast to float16 to save memory and file size
    return df.astype(np.float16)

def train_val_test_split(df, train_ratio=0.7, val_ratio=0.15):
    """
    Time-based split only. No shuffling.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def normalize_data(train_df, val_df, test_df, scaler_path):
    """
    Normalize using StandardScaler fit only on the training split.
    """
    target_col = 'TARGET_ND'
    feature_cols = [c for c in train_df.columns if c != target_col]
    
    scaler = StandardScaler()
    
    # Fit and transform on train
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    
    # Transform only on val and test
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    joblib.dump(scaler, scaler_path)
    
    return train_df, val_df, test_df, feature_cols

import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset that lazily creates sliding windows (3D tensors) from flat 2D data.
    This entirely avoids the massive memory overhead of saving 3D overlapping sequences!
    """
    def __init__(self, X_2d, y_1d, seq_length=48):
        self.X = torch.tensor(X_2d, dtype=torch.float32)
        self.y = torch.tensor(y_1d, dtype=torch.float32)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X) - self.seq_length + 1
        
    def __getitem__(self, idx):
        # Return (seq_len, num_features) and scalar target
        return self.X[idx:idx+self.seq_length], self.y[idx+self.seq_length-1]

def main():
    # Paths
    pca_path = '../../datalake/pca/df_pca.parquet'
    raw_path = '../../datalake/raw_data/full_data.csv'
    out_dir = '../../datalake/hmm_tensors/'
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("1. Loading data...")
    df, demand_cols = load_data(pca_path, raw_path)
    print(f"Data shape after load: {df.shape}")
    
    print("2. Creating features...")
    df = create_features(df, demand_cols)
    print(f"Data shape after feature engineering: {df.shape}")
    
    print("3. Splitting dataset...")
    train_df, val_df, test_df = train_val_test_split(df)
    
    print("4. Normalizing features...")
    scaler_path = os.path.join(out_dir, 'scaler.pkl')
    train_df, val_df, test_df, feature_cols = normalize_data(train_df, val_df, test_df, scaler_path)
    
    print("5. Saving data to compressed Parquet format...")
    # Parquet provides excellent columnar compression and preserves feature names
    # Casting to float16 strictly ensures the file sizes are tiny and comfortably under GitHub's 100MB limit
    train_df = train_df.astype(np.float16)
    val_df = val_df.astype(np.float16)
    test_df = test_df.astype(np.float16)
    
    train_df.to_parquet(os.path.join(out_dir, 'train.parquet'), index=True, compression='brotli')
    val_df.to_parquet(os.path.join(out_dir, 'val.parquet'), index=True, compression='brotli')
    test_df.to_parquet(os.path.join(out_dir, 'test.parquet'), index=True, compression='brotli')
    
    print(f"Train Parquet shape: {train_df.shape}")
    print(f"Val Parquet shape: {val_df.shape}")
    print(f"Test Parquet shape: {test_df.shape}")
    
    print(f"Success! Compressed parquet artifacts saved to {out_dir}")
    print(f"NOTE: Use TimeSeriesDataset to lazily load these dataframes into 3D sequence tensors during training.")

if __name__ == "__main__":
    main()
