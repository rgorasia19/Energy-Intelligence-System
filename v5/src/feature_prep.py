import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_data(pca_path, raw_path):
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
    
    df = df_pca.join(df_demand, how='inner')
    df.sort_index(inplace=True)
    
    return df, available_cols

def compute_rolling_fft_mag(series, window=48, freq_idx=1):
    def extract_fft(x):
        return np.abs(np.fft.rfft(x))[freq_idx]
    return series.rolling(window).apply(extract_fft, raw=True)

def create_features(df):
    df_feat = pd.DataFrame(index=df.index)
    
    # Base columns
    base_cols = ['ND', 'INTER_PC0', 'GEN_PC0', 'CAP_PC0']
    for c in base_cols:
        df_feat[f'raw_{c}'] = df[c]
        
    # Temporal derivatives (raw only)
    for c in base_cols:
        df_feat[f'raw_{c}_diff1'] = df[c].diff(1)
        df_feat[f'raw_{c}_diff2'] = df[c].diff(2)
        
    # Volatility (raw only)
    df_feat['raw_ND_vol_12'] = df['ND'].rolling(12).std()
    df_feat['raw_ND_vol_48'] = df['ND'].rolling(48).std()
    
    # Spectral (raw only)
    df_feat['raw_ND_fft_daily'] = compute_rolling_fft_mag(df['ND'], window=48, freq_idx=1)
    df_feat['raw_ND_fft_12h'] = compute_rolling_fft_mag(df['ND'], window=48, freq_idx=2)
    
    # Calendar features
    hour_sin = np.sin(2 * np.pi * df.index.hour / 24.0)
    hour_cos = np.cos(2 * np.pi * df.index.hour / 24.0)
    dow_sin = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    dow_cos = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
    
    df_feat['raw_cal_hour_sin'] = hour_sin
    df_feat['raw_cal_hour_cos'] = hour_cos
    df_feat['raw_cal_dow_sin'] = dow_sin
    df_feat['raw_cal_dow_cos'] = dow_cos
    
    df_feat['gate_cal_hour_sin'] = hour_sin
    df_feat['gate_cal_hour_cos'] = hour_cos
    df_feat['gate_cal_dow_sin'] = dow_sin
    df_feat['gate_cal_dow_cos'] = dow_cos
    
    # Regime Priors for Gating (Smoothed Base)
    df_feat['gate_ND_roll24_mean'] = df['ND'].rolling(24).mean()
    df_feat['gate_ND_roll48_mean'] = df['ND'].rolling(48).mean()
    df_feat['gate_INTER_roll48'] = df['INTER_PC0'].rolling(48).mean()
    
    # NEW IN V5: Explicit Regime Features (Volatility and Returns)
    df_feat['gate_ND_vol_12'] = df['ND'].rolling(12).std()
    df_feat['gate_ND_vol_48'] = df['ND'].rolling(48).std()
    df_feat['gate_ND_ret_12'] = df['ND'].pct_change(12)
    df_feat['gate_ND_ret_48'] = df['ND'].pct_change(48)
    
    # Targets (next step)
    df_feat['TARGET_ND'] = df['ND'].shift(-1)
    
    # Auxiliary Targets for Regime Interpretability (predicting next step's state)
    # We will predict the 48-step rolling standard deviation (volatility) and 48-step rolling return (trend)
    df_feat['TARGET_VOL'] = df['ND'].rolling(48).std().shift(-1)
    df_feat['TARGET_TREND'] = df['ND'].pct_change(48).shift(-1)
    
    # Drop NaNs
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_feat.dropna(inplace=True)
    
    return df_feat.astype(np.float32)

def train_val_test_split(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def normalize_data(train_df, val_df, test_df, scaler_path):
    target_cols = ['TARGET_ND', 'TARGET_VOL', 'TARGET_TREND']
    feature_cols = [c for c in train_df.columns if c not in target_cols]
    
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    joblib.dump(scaler, scaler_path)
    
    raw_cols = [c for c in feature_cols if c.startswith('raw_')]
    gate_cols = [c for c in feature_cols if c.startswith('gate_')]
    
    # We also need to scale the auxiliary targets so they have reasonable magnitudes for MSE loss!
    # Unlike TARGET_ND which we left raw, TARGET_VOL and TARGET_TREND can have very small or large scales.
    aux_scaler = StandardScaler()
    aux_cols = ['TARGET_VOL', 'TARGET_TREND']
    train_df[aux_cols] = aux_scaler.fit_transform(train_df[aux_cols])
    val_df[aux_cols] = aux_scaler.transform(val_df[aux_cols])
    test_df[aux_cols] = aux_scaler.transform(test_df[aux_cols])
    
    joblib.dump(aux_scaler, scaler_path.replace('scaler.pkl', 'aux_scaler.pkl'))
    
    return train_df, val_df, test_df, raw_cols, gate_cols

def main():
    pca_path = '../../datalake/pca/df_pca.parquet'
    raw_path = '../../datalake/raw_data/full_data.csv'
    out_dir = '../../datalake/moe_tensors/'
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("1. Loading data...")
    df, _ = load_data(pca_path, raw_path)
    
    print("2. Creating features...")
    df = create_features(df)
    
    print("3. Splitting dataset...")
    train_df, val_df, test_df = train_val_test_split(df)
    
    print("4. Normalizing features...")
    scaler_path = os.path.join(out_dir, 'scaler.pkl')
    train_df, val_df, test_df, raw_cols, gate_cols = normalize_data(train_df, val_df, test_df, scaler_path)
    
    joblib.dump({'raw_cols': raw_cols, 'gate_cols': gate_cols}, os.path.join(out_dir, 'feature_groups.pkl'))
    
    print("5. Saving data to compressed Parquet format...")
    train_df.to_parquet(os.path.join(out_dir, 'train.parquet'), index=True, compression='brotli')
    val_df.to_parquet(os.path.join(out_dir, 'val.parquet'), index=True, compression='brotli')
    test_df.to_parquet(os.path.join(out_dir, 'test.parquet'), index=True, compression='brotli')
    
    print(f"Train Parquet shape: {train_df.shape}")
    print(f"Raw features: {len(raw_cols)}, Gate features: {len(gate_cols)}")
    print(f"Success! Compressed parquet artifacts saved to {out_dir}")

if __name__ == "__main__":
    main()
