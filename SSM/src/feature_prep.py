import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib

def standardise_timeseries(df: pd.DataFrame, time_col: str, continuous_cols: list, flow_cols: list) -> pd.DataFrame:
    """
    Converts timestamps to timezone-aware UTC, resamples to daily frequency,
    and ensures a complete daily index.
    """
    df = df.copy()
    
    if time_col not in df.columns:
        # If it's an index, reset it
        if df.index.name == time_col:
            df = df.reset_index()
        else:
            raise KeyError(f"Time column '{time_col}' not found in dataframe.")

    # Convert to datetime, make timezone aware (UTC)
    df[time_col] = pd.to_datetime(df[time_col])
    if df[time_col].dt.tz is None:
        df[time_col] = df[time_col].dt.tz_localize('UTC')
    else:
        df[time_col] = df[time_col].dt.tz_convert('UTC')
        
    df.set_index(time_col, inplace=True)
    
    # Keep only relevant columns that actually exist
    valid_continuous = [c for c in continuous_cols if c in df.columns]
    valid_flow = [c for c in flow_cols if c in df.columns]
    
    df = df[valid_continuous + valid_flow]
    
    # Resample
    # mean for continuous, sum for flow
    agg_dict = {col: 'mean' for col in valid_continuous}
    agg_dict.update({col: 'sum' for col in valid_flow})
    
    df_daily = df.resample('D').agg(agg_dict)
    
    return df_daily

def merge_features(dfs: dict, start_date: str = '2001-01-01') -> pd.DataFrame:
    """
    Outer joins dataframes on the date index, filters from start_date,
    and adds missingness masks.
    dfs: dict mapping name -> dataframe
    """
    if not dfs:
        return pd.DataFrame()
        
    # Start with the first dataframe
    names = list(dfs.keys())
    merged_df = dfs[names[0]].copy()
    
    # Outer join the rest
    for name in names[1:]:
        merged_df = merged_df.join(dfs[name], how='outer')
        
    # Filter from start_date
    start_dt = pd.to_datetime(start_date).tz_localize('UTC')
    merged_df = merged_df[merged_df.index >= start_dt]
    
    # Add missingness masks
    for col in merged_df.columns:
        if not col.endswith('_available'):
            merged_df[f"{col}_available"] = merged_df[col].notna().astype(int)
            
    return merged_df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar features and rolling statistics.
    """
    df = df.copy()
    
    # Calendar features
    day_of_week = df.index.dayofweek
    day_of_year = df.index.dayofyear
    
    df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
    
    df['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # Rolling stats for non-mask columns
    base_cols = [c for c in df.columns if not c.endswith('_available') and not c.endswith('_sin') and not c.endswith('_cos')]
    
    for col in base_cols:
        # 7-day rolling
        df[f"{col}_roll7_mean"] = df[col].rolling(window=7, min_periods=1).mean()
        df[f"{col}_roll7_std"] = df[col].rolling(window=7, min_periods=1).std().fillna(0)
        
        # 30-day rolling
        df[f"{col}_roll30_mean"] = df[col].rolling(window=30, min_periods=1).mean()
        df[f"{col}_roll30_std"] = df[col].rolling(window=30, min_periods=1).std().fillna(0)
        
        # Lags
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag7"] = df[col].shift(7)
        
    # Important: Drop the first 7 rows as they will have NaNs from lag7
    # Note: Rolling with min_periods=1 avoids dropping 30 rows.
    df = df.iloc[7:]
        
    return df

def fit_scaler(df: pd.DataFrame, feature_cols: list) -> RobustScaler:
    """
    Fits a robust scaler on the training dataframe.
    """
    import warnings
    scaler = RobustScaler()
    # We only scale non-mask features
    cols_to_scale = [c for c in feature_cols if not c.endswith('_available')]
    
    # RobustScaler handles NaNs, but throws a warning if a column is all NaN.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        scaler.fit(df[cols_to_scale])
        
    # If any feature was entirely NaN during training, its center and scale will be NaN.
    # We replace those with 0 and 1 respectively to act as a pass-through.
    if hasattr(scaler, 'center_') and scaler.center_ is not None:
        scaler.center_ = np.nan_to_num(scaler.center_, nan=0.0)
    if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
        scaler.scale_ = np.nan_to_num(scaler.scale_, nan=1.0)
    
    return scaler

def transform_data(df: pd.DataFrame, scaler: RobustScaler, feature_cols: list) -> pd.DataFrame:
    """
    Transforms the dataframe using the fitted scaler.
    """
    df = df.copy()
    cols_to_scale = [c for c in feature_cols if not c.endswith('_available')]
    
    # To handle NaNs during transform, we temporarily fill them, transform, then put NaNs back
    temp_df = df[cols_to_scale].copy()
    mask = temp_df.isna()
    temp_df = temp_df.fillna(0)
    
    scaled_values = scaler.transform(temp_df)
    scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=cols_to_scale)
    
    # Put NaNs back
    scaled_df[mask] = np.nan
    
    # Replace in original df
    df[cols_to_scale] = scaled_df
    
    return df

class SSMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, horizon: int, feature_columns: list, target_columns: list, max_missing_pct: float = 0.1):
        self.seq_len = seq_len
        self.horizon = horizon
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.max_missing_pct = max_missing_pct
        
        self.features = df[feature_columns].values
        self.targets = df[target_columns].values
        
        # Pre-calculate masks: 1 if present (not NaN), 0 if missing (NaN)
        self.feature_masks = (~np.isnan(self.features)).astype(np.float32)
        self.target_masks = (~np.isnan(self.targets)).astype(np.float32)
        
        # Zero-fill NaNs for tensor conversion (the mask will tell the model to ignore them)
        self.features = np.nan_to_num(self.features, nan=0.0)
        self.targets = np.nan_to_num(self.targets, nan=0.0)
        
        self.valid_indices = self._get_valid_indices()
        
    def _get_valid_indices(self):
        valid = []
        total_len = len(self.features)
        window_size = self.seq_len + self.horizon
        
        for i in range(total_len - window_size + 1):
            # Check sparsity
            seq_features = self.features[i : i + self.seq_len]
            seq_mask = self.feature_masks[i : i + self.seq_len]
            
            missing_pct = 1.0 - (seq_mask.sum() / seq_mask.size)
            if missing_pct <= self.max_missing_pct:
                valid.append(i)
                
        return valid

    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        enc_end = start_idx + self.seq_len
        dec_end = enc_end + self.horizon
        
        encoder_inputs = torch.tensor(self.features[start_idx:enc_end], dtype=torch.float32)
        encoder_mask = torch.tensor(self.feature_masks[start_idx:enc_end], dtype=torch.float32)
        
        decoder_targets = torch.tensor(self.targets[enc_end:dec_end], dtype=torch.float32)
        decoder_mask = torch.tensor(self.target_masks[enc_end:dec_end], dtype=torch.float32)
        
        return {
            "encoder_inputs": encoder_inputs,
            "encoder_mask": encoder_mask,
            "decoder_targets": decoder_targets,
            "decoder_mask": decoder_mask
        }

def time_split(df: pd.DataFrame, train_end: str, val_end: str):
    """
    Strictly time-based split.
    """
    train_dt = pd.to_datetime(train_end).tz_localize('UTC')
    val_dt = pd.to_datetime(val_end).tz_localize('UTC')
    
    train_df = df[df.index <= train_dt]
    val_df = df[(df.index > train_dt) & (df.index <= val_dt)]
    test_df = df[df.index > val_dt]
    
    return train_df, val_df, test_df

def create_dataloaders(train_ds, val_ds, test_ds, batch_size: int, num_workers: int = 0, pin_memory: bool = True):
    """
    Creates dataloaders for the datasets.
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Simple test to verify the pipeline runs end-to-end
    print("Testing SSM data pipeline...")
    
    # 1. Create dummy data
    dates = pd.date_range(start='2001-01-01', periods=3000, freq='D') # 3000 days is ~8 years
    df_demand = pd.DataFrame({
        'SETTLEMENT_DATE': dates,
        'demand': np.random.rand(3000) * 100
    })
    
    gen_dates = pd.date_range(start='2001-01-01', periods=3000, freq='D')
    df_gen = pd.DataFrame({
        'DATETIME': gen_dates,
        'generation': np.random.rand(3000) * 50
    })
    
    # 2. Standardise
    df_demand_daily = standardise_timeseries(df_demand, 'SETTLEMENT_DATE', continuous_cols=[], flow_cols=['demand'])
    df_gen_daily = standardise_timeseries(df_gen, 'DATETIME', continuous_cols=[], flow_cols=['generation'])
    
    # 3. Merge
    dfs = {'demand': df_demand_daily, 'gen': df_gen_daily}
    merged = merge_features(dfs)
    
    # 4. Engineer
    engineered = engineer_features(merged)
    
    # 5. Split
    train, val, test = time_split(engineered, '2001-12-31', '2002-12-31')
    
    # 6. Scale
    feature_cols = [c for c in engineered.columns]
    scaler = fit_scaler(train, feature_cols)
    train_scaled = transform_data(train, scaler, feature_cols)
    val_scaled = transform_data(val, scaler, feature_cols)
    test_scaled = transform_data(test, scaler, feature_cols)
    
    # 7. Dataset
    target_cols = ['demand']
    train_ds = SSMDataset(train_scaled, seq_len=30, horizon=7, feature_columns=feature_cols, target_columns=target_cols)
    
    print(f"Dataset length: {len(train_ds)}")
    
    if len(train_ds) > 0:
        sample = train_ds[0]
        print(f"Encoder Inputs: {sample['encoder_inputs'].shape}")
        print(f"Encoder Mask: {sample['encoder_mask'].shape}")
        print(f"Decoder Targets: {sample['decoder_targets'].shape}")
        print(f"Decoder Mask: {sample['decoder_mask'].shape}")
        
    print("Pipeline tested successfully.")
