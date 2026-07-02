import os
import sys

if sys.platform == 'win32' and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import dagshub
import mlflow
import mlflow.pytorch
from torch.utils.data import Dataset, DataLoader

from models.tft import TemporalFusionTransformer

class TFTDataset(Dataset):
    def __init__(self, df, obs_cols, known_cols, static_cols, seq_length=48, horizon=48):
        self.X_obs = torch.tensor(df[obs_cols].values, dtype=torch.float32)
        self.X_known = torch.tensor(df[known_cols].values, dtype=torch.float32)
        self.X_static = torch.tensor(df[static_cols].values, dtype=torch.float32)
        
        self.y_abs = torch.tensor(df['ND_CURRENT'].values, dtype=torch.float32)
        self.nd_current = torch.tensor(df['ND_CURRENT'].values, dtype=torch.float32)
        self.y_vol = torch.tensor(df['TARGET_VOL'].values, dtype=torch.float32)
        self.y_trend = torch.tensor(df['TARGET_TREND'].values, dtype=torch.float32)
        
        self.seq_length = seq_length
        self.horizon = horizon
        
    def __len__(self):
        return len(self.X_obs) - self.seq_length - self.horizon + 1
        
    def __getitem__(self, idx):
        # Static is constant across sequence, just take the first step
        static = self.X_static[idx]
        
        # Past observed (only available up to seq_length)
        past_obs = self.X_obs[idx : idx + self.seq_length]
        
        # Future known (available for both seq_length AND horizon)
        future_known = self.X_known[idx : idx + self.seq_length + self.horizon]
        
        # Targets are for the horizon period
        target_idx_start = idx + self.seq_length
        target_idx_end = idx + self.seq_length + self.horizon
        
        y_abs = self.y_abs[target_idx_start : target_idx_end]
        y_vol = self.y_vol[target_idx_start : target_idx_end]
        y_trend = self.y_trend[target_idx_start : target_idx_end]
        
        # The last known value before the prediction horizon starts
        nd_current = self.nd_current[idx + self.seq_length - 1]
        
        return static, past_obs, future_known, y_abs, y_vol, y_trend, nd_current

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_dir = '../../datalake/v7_tensors/'
    train_data = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    val_data = pd.read_parquet(os.path.join(data_dir, 'val.parquet'))
    
    feature_groups = joblib.load(os.path.join(data_dir, 'feature_groups.pkl'))
    obs_cols = feature_groups['obs_cols']
    known_cols = feature_groups['known_cols']
    static_cols = feature_groups['static_cols']
    
    seq_len = 48
    horizon = 48
    batch_size = 512
    d_model = 64
    num_heads = 4

    train_dataset = TFTDataset(train_data, obs_cols, known_cols, static_cols, seq_length=seq_len, horizon=horizon)
    val_dataset = TFTDataset(val_data, obs_cols, known_cols, static_cols, seq_length=seq_len, horizon=horizon)
    
    num_workers = 4 if device == "cuda" else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device=="cuda"))

    target_idx = None
    if 'obs_ND' in obs_cols:
        target_idx = obs_cols.index('obs_ND') if isinstance(obs_cols, list) else obs_cols.tolist().index('obs_ND')

    model = TemporalFusionTransformer(
        num_static_vars=len(static_cols),
        num_future_vars=len(known_cols),
        num_past_vars=len(obs_cols),
        d_model=d_model,
        num_heads=num_heads,
        seq_len=seq_len,
        horizon=horizon,
        target_in_past_idx=target_idx,
        dropout=0.1
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.SmoothL1Loss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=8, min_delta=1e-4)
    
    epochs = 40
    
    # DagsHub Auth
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rgorasia19"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "26b84ad06cab16667533b56a5cebe4ebf27d8f9a"
    
    dagshub.init(repo_owner="rgorasia19", repo_name="Energy-Intelligence-System", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow")
    mlflow.set_experiment("v7_tft_multistep")
    
    with mlflow.start_run():
        mlflow.log_params({
            "seq_len": seq_len,
            "horizon": horizon,
            "d_model": d_model,
            "num_heads": num_heads,
            "batch_size": batch_size,
            "learning_rate": 1e-3,
        })
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_idx, (static, past, known, y_abs, y_vol, y_trend, _) in enumerate(train_loader):
                static = static.to(device)
                past = past.to(device)
                known = known.to(device)
                y_abs = y_abs.to(device)
                y_vol = y_vol.to(device)
                y_trend = y_trend.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                pred_abs, pred_vol, pred_trend, _ = model(static, past, known)
                
                # Joint loss over H steps for all targets
                loss_nd = criterion(pred_abs, y_abs)
                loss_vol = criterion(pred_vol, y_vol)
                loss_trend = criterion(pred_trend, y_trend)
                
                # Equal weighting for now
                loss = loss_nd + loss_vol + loss_trend
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            val_nd_loss = 0.0
            
            with torch.no_grad():
                for static, past, known, y_abs, y_vol, y_trend, _ in val_loader:
                    static = static.to(device)
                    past = past.to(device)
                    known = known.to(device)
                    y_abs = y_abs.to(device)
                    y_vol = y_vol.to(device)
                    y_trend = y_trend.to(device)
                    
                    pred_abs, pred_vol, pred_trend, _ = model(static, past, known)
                    
                    loss_nd = criterion(pred_abs, y_abs)
                    loss_vol = criterion(pred_vol, y_vol)
                    loss_trend = criterion(pred_trend, y_trend)
                    
                    loss = loss_nd + loss_vol + loss_trend
                    
                    val_loss += loss.item()
                    val_nd_loss += loss_nd.item()
                    
            val_loss /= len(val_loader)
            val_nd_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} (Val ND Loss: {val_nd_loss:.4f})")
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_nd_loss": val_nd_loss
            }, step=epoch)
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
                
        # Save model and artifacts
        os.makedirs('saved_models', exist_ok=True)
        model_path = "saved_models/tft_v7.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(os.path.join(data_dir, 'scaler.pkl'))
        mlflow.log_artifact(os.path.join(data_dir, 'target_scaler.pkl'))
        mlflow.log_artifact(os.path.join(data_dir, 'feature_groups.pkl'))
        print("Training complete and model saved.")

if __name__ == "__main__":
    train()
