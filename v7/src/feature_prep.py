import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_data(pca_path, raw_path, weather_path):
    # Load PCA data
    df_pca = pd.read_parquet(pca_path)
    df_pca.index = pd.to_datetime(df_pca.index)
    
    # Load raw data for demand signals
    df_raw = pd.read_csv(raw_path)
    df_raw['DATETIME'] = pd.to_datetime(df_raw['DATETIME'])
    df_raw.set_index('DATETIME', inplace=True)
    
    # Load weather data
    df_weather = pd.read_csv(weather_path)
    df_weather['DATETIME'] = pd.to_datetime(df_weather['DATETIME'])
    df_weather.set_index('DATETIME', inplace=True)
    
    # Merge
    demand_cols = ['ND', 'TSD', 'ENGLAND_WALES_DEMAND']
    available_cols = [c for c in demand_cols if c in df_raw.columns]
    df_demand = df_raw[available_cols]
    
    df = df_pca.join(df_demand, how='inner').join(df_weather, how='inner')
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
    
    # Observed past (we will add 'obs_' prefix)
    # STRIPPED: We only keep the raw signals. We remove all diffs, volatilities, and FFTs
    # so the TFT can learn these features natively.
    for c in base_cols:
        df_feat[f'obs_{c}'] = df[c]
    
    # Known future (we will add 'known_' prefix) with Fourier Features
    # Daily Fourier (k=1..4)
    for k in range(1, 5):
        df_feat[f'known_cal_hour_sin_{k}'] = np.sin(2 * np.pi * k * df.index.hour / 24.0)
        df_feat[f'known_cal_hour_cos_{k}'] = np.cos(2 * np.pi * k * df.index.hour / 24.0)
    # Weekly Fourier (k=1..3)
    for k in range(1, 4):
        df_feat[f'known_cal_dow_sin_{k}'] = np.sin(2 * np.pi * k * df.index.dayofweek / 7.0)
        df_feat[f'known_cal_dow_cos_{k}'] = np.cos(2 * np.pi * k * df.index.dayofweek / 7.0)
        
    # Weather Forecasts (using actuals as a proxy for perfectly accurate forecasts)
    weather_cols = ['temperature_2m', 'cloudcover', 'windspeed_10m', 'shortwave_radiation']
    for wc in weather_cols:
        if wc in df.columns:
            df_feat[f'known_{wc}'] = df[wc]
    
    # Static (we will add 'static_' prefix)
    df_feat['static_dummy'] = 1.0
    
    # Targets (UNSHIFTED - the Dataset will slice [t : t+H] from these)
    df_feat['TARGET_ND_DIFF'] = df['ND'].diff(1)
    df_feat['TARGET_VOL'] = df['ND'].rolling(48).std()
    df_feat['TARGET_TREND'] = df['ND'].pct_change(48)
    
    # Keep track of absolute current value for reconstruction in eval
    df_feat['ND_CURRENT'] = df['ND']
    
    # Drop NaNs created by rolling and diff
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
    target_cols = ['TARGET_ND_DIFF', 'TARGET_VOL', 'TARGET_TREND', 'ND_CURRENT']
    feature_cols = [c for c in train_df.columns if c not in target_cols and not c.startswith('static_')]
    
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    joblib.dump(scaler, scaler_path)
    
    obs_cols = [c for c in feature_cols if c.startswith('obs_')]
    known_cols = [c for c in feature_cols if c.startswith('known_')]
    static_cols = ['static_dummy']
    
    aux_scaler = StandardScaler()
    # Normalize ALL targets for multi-step to make loss balanced
    train_df[target_cols] = aux_scaler.fit_transform(train_df[target_cols])
    val_df[target_cols] = aux_scaler.transform(val_df[target_cols])
    test_df[target_cols] = aux_scaler.transform(test_df[target_cols])
    
    joblib.dump(aux_scaler, scaler_path.replace('scaler.pkl', 'target_scaler.pkl'))
    
    return train_df, val_df, test_df, obs_cols, known_cols, static_cols

def main():
    pca_path = '../../datalake/pca/df_pca.parquet'
    raw_path = '../../datalake/raw_data/full_data.csv'
    weather_path = '../../datalake/raw_data/weather_data.csv'
    out_dir = '../../datalake/v7_tensors/'
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("1. Loading data...")
    df, _ = load_data(pca_path, raw_path, weather_path)
    
    print("2. Creating features...")
    df = create_features(df)
    
    print("3. Splitting dataset...")
    train_df, val_df, test_df = train_val_test_split(df)
    
    print("4. Normalizing features...")
    scaler_path = os.path.join(out_dir, 'scaler.pkl')
    train_df, val_df, test_df, obs_cols, known_cols, static_cols = normalize_data(train_df, val_df, test_df, scaler_path)
    
    joblib.dump({
        'obs_cols': obs_cols, 
        'known_cols': known_cols, 
        'static_cols': static_cols
    }, os.path.join(out_dir, 'feature_groups.pkl'))
    
    print("5. Saving data to compressed Parquet format...")
    train_df.to_parquet(os.path.join(out_dir, 'train.parquet'), index=True, compression='brotli')
    val_df.to_parquet(os.path.join(out_dir, 'val.parquet'), index=True, compression='brotli')
    test_df.to_parquet(os.path.join(out_dir, 'test.parquet'), index=True, compression='brotli')
    
    print(f"Train Parquet shape: {train_df.shape}")
    print(f"Observed: {len(obs_cols)}, Known: {len(known_cols)}, Static: {len(static_cols)}")
    print(f"Success! Compressed parquet artifacts saved to {out_dir}")

if __name__ == "__main__":
    main()
