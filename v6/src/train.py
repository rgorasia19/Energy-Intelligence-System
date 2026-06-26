import os
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

from models.end_to_end import ContinuousStateForecaster

class TimeSeriesDataset(Dataset):
    def __init__(self, df, raw_cols, gate_cols, seq_length=48):
        self.X_raw = torch.tensor(df[raw_cols].values, dtype=torch.float32)
        self.X_gate = torch.tensor(df[gate_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df['TARGET_ND'].values, dtype=torch.float32)
        self.y_vol = torch.tensor(df['TARGET_VOL'].values, dtype=torch.float32)
        self.y_trend = torch.tensor(df['TARGET_TREND'].values, dtype=torch.float32)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X_raw) - self.seq_length + 1
        
    def __getitem__(self, idx):
        return (
            self.X_raw[idx:idx+self.seq_length], 
            self.X_gate[idx:idx+self.seq_length], 
            self.y[idx+self.seq_length-1],
            self.y_vol[idx+self.seq_length-1],
            self.y_trend[idx+self.seq_length-1]
        )

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

def compute_continuous_losses(z_seq, lambda_smooth, lambda_ib):
    """
    z_seq: (B, seq_len, latent_dim)
    """
    # 1. Temporal Smoothness (MSE between consecutive latent states)
    if z_seq.size(1) > 1:
        l_smooth = lambda_smooth * torch.mean((z_seq[:, 1:, :] - z_seq[:, :-1, :]) ** 2)
    else:
        l_smooth = torch.tensor(0.0, device=z_seq.device)
        
    # 2. Information Bottleneck (L2 Penalty on latent states)
    l_ib = lambda_ib * torch.mean(z_seq ** 2)
    
    return l_smooth, l_ib

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_dir = '../../datalake/moe_tensors/'
    train_data = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    val_data = pd.read_parquet(os.path.join(data_dir, 'val.parquet'))
    
    feature_groups = joblib.load(os.path.join(data_dir, 'feature_groups.pkl'))
    raw_cols = feature_groups['raw_cols']
    gate_cols = feature_groups['gate_cols']
    
    seq_len = 48
    batch_size = 2048
    latent_dim = 32

    train_dataset = TimeSeriesDataset(train_data, raw_cols, gate_cols, seq_length=seq_len)
    val_dataset = TimeSeriesDataset(val_data, raw_cols, gate_cols, seq_length=seq_len)
    
    num_workers = 4 if device == "cuda" else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device=="cuda"))

    model = ContinuousStateForecaster(
        raw_feature_dim=len(raw_cols), 
        gate_feature_dim=len(gate_cols), 
        seq_len=seq_len,
        d_model=64,
        latent_dim=latent_dim
    ).to(device)

    # L2 weight decay added to AdamW - Increased for stronger regularization
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.SmoothL1Loss()
    aux_criterion = nn.MSELoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=8, min_delta=1e-4)
    
    epochs = 40
    
    # Loss weights for Continuous Regularization
    lambda_smooth = 0.1 # Weight for ||z_t - z_{t-1}||^2
    lambda_ib = 1e-2    # Increased weight for ||z_t||^2 (regularization)
    
    # DagsHub Auth
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rgorasia19"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "26b84ad06cab16667533b56a5cebe4ebf27d8f9a"
    
    dagshub.init(repo_owner="rgorasia19", repo_name="Energy-Intelligence-System", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow")
    mlflow.set_experiment("v6_continuous_latent_state")
    
    with mlflow.start_run() as run:
        print(f"Run started: {run.info.run_id}")
        
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "latent_dim": latent_dim,
            "lambda_smooth": lambda_smooth,
            "lambda_ib": lambda_ib
        })
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for x_raw, x_gate, y, y_vol, y_trend in train_loader:
                x_raw, x_gate, y = x_raw.to(device), x_gate.to(device), y.to(device)
                y_vol, y_trend = y_vol.to(device), y_trend.to(device)
                
                optimizer.zero_grad()
                
                y_hat, z_seq, aux_preds = model(x_raw, x_gate)
                
                l_pred = criterion(y_hat, y)
                l_smooth, l_ib = compute_continuous_losses(z_seq, lambda_smooth, lambda_ib)
                
                # Auxiliary Loss: predict vol and trend for the sequence
                # aux_preds is (B, seq_len, 2)
                l_aux_vol = aux_criterion(aux_preds[:, -1, 0], y_vol)
                l_aux_trend = aux_criterion(aux_preds[:, -1, 1], y_trend)
                l_aux = l_aux_vol + l_aux_trend
                
                loss = l_pred + l_aux + l_smooth + l_ib
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * x_raw.size(0)
                
            train_loss /= len(train_dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_pred_loss = 0.0
            val_aux_loss = 0.0
            val_smooth_loss = 0.0
            val_ib_loss = 0.0
            
            with torch.no_grad():
                for x_raw, x_gate, y, y_vol, y_trend in val_loader:
                    x_raw, x_gate, y = x_raw.to(device), x_gate.to(device), y.to(device)
                    y_vol, y_trend = y_vol.to(device), y_trend.to(device)
                    
                    y_hat, z_seq, aux_preds = model(x_raw, x_gate)
                    
                    l_pred = criterion(y_hat, y)
                    l_smooth, l_ib = compute_continuous_losses(z_seq, lambda_smooth, lambda_ib)
                    
                    l_aux_vol = aux_criterion(aux_preds[:, -1, 0], y_vol)
                    l_aux_trend = aux_criterion(aux_preds[:, -1, 1], y_trend)
                    l_aux = l_aux_vol + l_aux_trend
                    
                    loss = l_pred + l_aux + l_smooth + l_ib
                    
                    val_loss += loss.item() * x_raw.size(0)
                    val_pred_loss += l_pred.item() * x_raw.size(0)
                    val_aux_loss += l_aux.item() * x_raw.size(0)
                    val_smooth_loss += l_smooth.item() * x_raw.size(0)
                    val_ib_loss += l_ib.item() * x_raw.size(0)
                    
            val_loss /= len(val_dataset)
            val_pred_loss /= len(val_dataset)
            val_aux_loss /= len(val_dataset)
            val_smooth_loss /= len(val_dataset)
            val_ib_loss /= len(val_dataset)
            
            mlflow.log_metric("train_loss_total", train_loss, step=epoch)
            mlflow.log_metric("val_loss_total", val_loss, step=epoch)
            mlflow.log_metric("val_loss_pred", val_pred_loss, step=epoch)
            mlflow.log_metric("val_loss_aux", val_aux_loss, step=epoch)
            mlflow.log_metric("val_loss_smooth", val_smooth_loss, step=epoch)
            mlflow.log_metric("val_loss_ib", val_ib_loss, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss Total: {val_loss:.4f}")
            
            scheduler.step(val_loss)
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break
                
        print("Training complete. Logging robust artifacts...")
        
        mlflow.log_artifact(os.path.join(data_dir, 'feature_groups.pkl'))
        mlflow.log_artifact(os.path.join(data_dir, 'scaler.pkl'))
        mlflow.log_artifact(os.path.join(data_dir, 'aux_scaler.pkl'))
        
        torch.save(model.state_dict(), "continuous_latent.pth")
        mlflow.log_artifact("continuous_latent.pth")

if __name__ == "__main__":
    train()
