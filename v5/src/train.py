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

from models.predictor import UnifiedRegimeModel

class MoEDataset(Dataset):
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

def compute_regime_losses(p_regime, lambda1, lambda2, lambda3):
    """
    p_regime: (B, seq_len, K)
    """
    B, seq_len, K = p_regime.size()
    
    # 1. Entropy Loss (Maximize entropy to encourage diversity early, per user request)
    entropy = -torch.sum(p_regime * torch.log(p_regime + 1e-8), dim=-1).mean()
    l_entropy = -lambda1 * entropy
    
    # 2. Load Balancing (KL divergence between global mean and uniform distribution)
    global_mean_p = p_regime.mean(dim=(0, 1)) # (K,)
    uniform_p = torch.ones(K, device=p_regime.device) / K
    l_balance = lambda2 * torch.sum(global_mean_p * torch.log((global_mean_p + 1e-8) / uniform_p))
    
    # 3. Temporal Smoothness (MSE between consecutive timesteps)
    if seq_len > 1:
        l_smooth = lambda3 * torch.mean((p_regime[:, 1:, :] - p_regime[:, :-1, :]) ** 2)
    else:
        l_smooth = torch.tensor(0.0, device=p_regime.device)
        
    return l_entropy, l_balance, l_smooth, entropy.item()

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

    train_dataset = MoEDataset(train_data, raw_cols, gate_cols, seq_length=seq_len)
    val_dataset = MoEDataset(val_data, raw_cols, gate_cols, seq_length=seq_len)
    
    num_workers = 4 if device == "cuda" else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device=="cuda"))

    num_regimes = 2
    embed_dim = 32
    model = UnifiedRegimeModel(
        raw_feature_dim=len(raw_cols), 
        gate_feature_dim=len(gate_cols), 
        seq_len=seq_len,
        num_regimes=num_regimes,
        d_model=64,
        embed_dim=embed_dim
    ).to(device)

    # L2 weight decay added to AdamW
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.SmoothL1Loss()
    aux_criterion = nn.MSELoss()
    
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)
    
    epochs = 40
    
    # Loss weights
    lambda1 = 0.1 # Entropy
    lambda2 = 1.0 # Load balancing
    lambda3 = 0.5 # Smoothness
    
    # DagsHub Auth
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rgorasia19"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "26b84ad06cab16667533b56a5cebe4ebf27d8f9a"
    
    dagshub.init(repo_owner="rgorasia19", repo_name="Energy-Intelligence-System", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow")
    mlflow.set_experiment("v5_attention_regime_2state")
    
    with mlflow.start_run() as run:
        print(f"Run started: {run.info.run_id}")
        
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_regimes": num_regimes,
            "lambda1_entropy": lambda1,
            "lambda2_balance": lambda2,
            "lambda3_smooth": lambda3
        })
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            # Temperature Annealing: 2.5 -> 0.8
            progress = epoch / max(1, epochs - 1)
            tau = 2.5 - progress * (2.5 - 0.8)
            
            for x_raw, x_gate, y, y_vol, y_trend in train_loader:
                x_raw, x_gate, y = x_raw.to(device), x_gate.to(device), y.to(device)
                y_vol, y_trend = y_vol.to(device), y_trend.to(device)
                
                optimizer.zero_grad()
                
                y_hat, p_regime, _, aux_preds, _ = model(x_raw, x_gate, tau=tau)
                
                l_pred = criterion(y_hat, y)
                l_entropy, l_balance, l_smooth, _ = compute_regime_losses(p_regime, lambda1, lambda2, lambda3)
                
                # Auxiliary Loss: predict vol and trend for the sequence
                # aux_preds is (B, seq_len, 2)
                # We want to match the target at the end of the sequence
                # (or we could match sequence targets if we output sequence aux, but y_vol is scalar for the window)
                # Let's take the last step of the sequence for aux_preds
                l_aux_vol = aux_criterion(aux_preds[:, -1, 0], y_vol)
                l_aux_trend = aux_criterion(aux_preds[:, -1, 1], y_trend)
                l_aux = l_aux_vol + l_aux_trend
                
                loss = l_pred + l_aux + l_entropy + l_balance + l_smooth
                
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
            val_entropy_raw = 0.0
            global_p_sum = torch.zeros(num_regimes, device=device)
            
            with torch.no_grad():
                for x_raw, x_gate, y, y_vol, y_trend in val_loader:
                    x_raw, x_gate, y = x_raw.to(device), x_gate.to(device), y.to(device)
                    y_vol, y_trend = y_vol.to(device), y_trend.to(device)
                    
                    y_hat, p_regime, _, aux_preds, _ = model(x_raw, x_gate, tau=tau)
                    
                    l_pred = criterion(y_hat, y)
                    l_entropy, l_balance, l_smooth, raw_entropy = compute_regime_losses(p_regime, lambda1, lambda2, lambda3)
                    
                    l_aux_vol = aux_criterion(aux_preds[:, -1, 0], y_vol)
                    l_aux_trend = aux_criterion(aux_preds[:, -1, 1], y_trend)
                    l_aux = l_aux_vol + l_aux_trend
                    
                    loss = l_pred + l_aux + l_entropy + l_balance + l_smooth
                    
                    val_loss += loss.item() * x_raw.size(0)
                    val_pred_loss += l_pred.item() * x_raw.size(0)
                    val_aux_loss += l_aux.item() * x_raw.size(0)
                    val_entropy_raw += raw_entropy * x_raw.size(0)
                    
                    # Accumulate for global load balancing check
                    global_p_sum += p_regime.sum(dim=(0, 1))
                    
            val_loss /= len(val_dataset)
            val_pred_loss /= len(val_dataset)
            val_aux_loss /= len(val_dataset)
            val_entropy_raw /= len(val_dataset)
            
            total_elements = len(val_dataset) * seq_len
            occupancy = (global_p_sum / total_elements).cpu().numpy()
            
            mlflow.log_metric("train_loss_total", train_loss, step=epoch)
            mlflow.log_metric("val_loss_total", val_loss, step=epoch)
            mlflow.log_metric("val_loss_pred", val_pred_loss, step=epoch)
            mlflow.log_metric("val_loss_aux", val_aux_loss, step=epoch)
            mlflow.log_metric("val_entropy", val_entropy_raw, step=epoch)
            mlflow.log_metric("tau", tau, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} - Tau: {tau:.2f} - Train Loss: {train_loss:.4f} - Val Loss Total: {val_loss:.4f}")
            print(f"  --> Occupancy: {[f'{x*100:.1f}%' for x in occupancy]}")
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break
                
        print("Training complete. Logging robust artifacts...")
        
        mlflow.log_artifact(os.path.join(data_dir, 'feature_groups.pkl'))
        mlflow.log_artifact(os.path.join(data_dir, 'scaler.pkl'))
        mlflow.log_artifact(os.path.join(data_dir, 'aux_scaler.pkl'))
        
        torch.save(model.state_dict(), "regime_attention.pth")
        mlflow.log_artifact("regime_attention.pth")

if __name__ == "__main__":
    train()
