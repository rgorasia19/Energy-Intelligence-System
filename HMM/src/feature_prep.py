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
    Create clean low-dimensional features for HMM (Curated macro set, no lags).
    """
    hmm_features = ['ND', 'INTER_PC0', 'GEN_PC0', 'CAP_PC0']
    
    # Select only these features
    df_feat = df[hmm_features].copy()
    
    # Define aligned target (next-step horizon)
    df_feat['TARGET_ND'] = df['ND'].shift(-1)
    
    # Handle NaNs explicitly
    df_feat.dropna(inplace=True)
    
    # Compute rank and condition number to validate structure
    cov_matrix = np.cov(StandardScaler().fit_transform(df_feat[hmm_features].values), rowvar=False)
    rank = np.linalg.matrix_rank(cov_matrix)
    cond_num = np.linalg.cond(cov_matrix)
    print(f"[HYGIENE CHECK] Feature columns: {hmm_features}")
    print(f"[HYGIENE CHECK] Covariance rank: {rank}/{len(hmm_features)}")
    print(f"[HYGIENE CHECK] Condition number: {cond_num:.2f}")
    if rank < len(hmm_features):
        print("[WARNING] Collinear feature space! Covariance rank is less than feature count.")
        
    # Cast to float32 for numerical stability (no float16)
    return df_feat.astype(np.float32)

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
    # Use float32 to ensure numerical stability and avoid EM collapse
    train_df = train_df.astype(np.float32)
    val_df = val_df.astype(np.float32)
    test_df = test_df.astype(np.float32)
    
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
