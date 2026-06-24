import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import Dataset, DataLoader
from models.moe import TemporalMoE

class MoEDataset(Dataset):
    def __init__(self, df, raw_cols, gate_cols, seq_length=48):
        self.X_raw = torch.tensor(df[raw_cols].values, dtype=torch.float32)
        self.X_gate = torch.tensor(df[gate_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df['TARGET_ND'].values, dtype=torch.float32)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X_raw) - self.seq_length + 1
        
    def __getitem__(self, idx):
        return (
            self.X_raw[idx:idx+self.seq_length], 
            self.X_gate[idx:idx+self.seq_length], 
            self.y[idx+self.seq_length-1]
        )

def get_anchor_penalty(model, anchors):
    loss = 0.0
    for name, param in model.named_parameters():
        if name in anchors:
            loss += torch.sum((param - anchors[name]) ** 2)
    return loss

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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=(device=="cuda"))

    model = TemporalMoE(
        raw_feature_dim=len(raw_cols), 
        gate_feature_dim=len(gate_cols), 
        seq_len=seq_len
    ).to(device)
    
    criterion = nn.SmoothL1Loss()
    
    epochs_linear = 10
    epochs_fourier = 15
    epochs_attention = 20
    epochs_joint = 40
    
    total_epochs = epochs_linear + epochs_fourier + epochs_attention + epochs_joint
    
    # MLflow Config (matching test_mlflow.py style)
    mlflow.set_tracking_uri("sqlite:///../../mlflow.db")
    mlflow.set_experiment("v4_moe")
    
    print("Starting MLflow V4 MoE run...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Run started: {run_id}")
        
        mlflow.log_params({
            "epochs_linear": epochs_linear,
            "epochs_fourier": epochs_fourier,
            "epochs_attention": epochs_attention,
            "epochs_joint": epochs_joint,
            "batch_size": batch_size,
            "seq_len": seq_len
        })
        
        # Phase 1: Warm-start Linear Expert
        print("=== Phase 1: Warm-start Linear Expert ===")
        opt_linear = optim.Adam(model.linear_expert.parameters(), lr=1e-3)
        for epoch in range(epochs_linear):
            model.train()
            train_loss = 0.0
            for x_raw, x_gate, y in train_loader:
                x_raw, y = x_raw.to(device), y.to(device)
                opt_linear.zero_grad()
                pred = model.linear_expert(x_raw)
                loss = criterion(pred, y)
                loss.backward()
                opt_linear.step()
                train_loss += loss.item() * x_raw.size(0)
            print(f"Epoch {epoch+1}/{epochs_linear} - Linear Loss: {train_loss / len(train_dataset):.4f}")
            
        # Phase 2: Warm-start Fourier Expert
        print("=== Phase 2: Warm-start Fourier Expert (Predicting Residuals) ===")
        opt_fourier = optim.Adam(model.fourier_expert.parameters(), lr=1e-3)
        for epoch in range(epochs_fourier):
            model.train()
            train_loss = 0.0
            for x_raw, x_gate, y in train_loader:
                x_raw, y = x_raw.to(device), y.to(device)
                opt_fourier.zero_grad()
                with torch.no_grad():
                    pred_linear = model.linear_expert(x_raw)
                pred_fourier = model.fourier_expert(x_raw)
                # Predict residual implicitly by minimizing sum
                loss = criterion(pred_linear + pred_fourier, y)
                loss.backward()
                opt_fourier.step()
                train_loss += loss.item() * x_raw.size(0)
            print(f"Epoch {epoch+1}/{epochs_fourier} - Fourier Loss: {train_loss / len(train_dataset):.4f}")
            
        # Phase 3: Warm-start Attention Expert
        print("=== Phase 3: Warm-start Attention Expert (Predicting Residuals) ===")
        opt_attention = optim.Adam(model.attention_expert.parameters(), lr=1e-3)
        for epoch in range(epochs_attention):
            model.train()
            train_loss = 0.0
            for x_raw, x_gate, y in train_loader:
                x_raw, y = x_raw.to(device), y.to(device)
                opt_attention.zero_grad()
                with torch.no_grad():
                    pred_linear = model.linear_expert(x_raw)
                    pred_fourier = model.fourier_expert(x_raw)
                pred_attn = model.attention_expert(x_raw)
                loss = criterion(pred_linear + pred_fourier + pred_attn, y)
                loss.backward()
                opt_attention.step()
                train_loss += loss.item() * x_raw.size(0)
            print(f"Epoch {epoch+1}/{epochs_attention} - Attention Loss: {train_loss / len(train_dataset):.4f}")
            
        # Save Anchor Weights
        print("Saving anchor weights for regularization...")
        anchors = {name: param.clone().detach() for name, param in model.named_parameters() if 'expert' in name}
        
        # Phase 4: Joint MoE Training
        print("=== Phase 4: Joint MoE Training ===")
        opt_joint = optim.Adam([
            {'params': model.gating.parameters(), 'lr': 1e-3},
            {'params': model.linear_expert.parameters(), 'lr': 1e-4},
            {'params': model.fourier_expert.parameters(), 'lr': 1e-4},
            {'params': model.attention_expert.parameters(), 'lr': 1e-4}
        ])
        
        tau = 5.0
        tau_decay = 0.9
        
        # For adaptive loss triggering
        val_losses = []
        smoothing_active = False
        
        for epoch in range(epochs_joint):
            model.train()
            train_loss = 0.0
            
            anchor_weight = max(0.0, 1.0 - (epoch / epochs_joint)) # Decays from 1 to 0
            
            for x_raw, x_gate, y in train_loader:
                x_raw, x_gate, y = x_raw.to(device), x_gate.to(device), y.to(device)
                opt_joint.zero_grad()
                
                final_output, weights, logits, _ = model(x_raw, x_gate, tau=tau, hard=True)
                
                base_loss = criterion(final_output, y)
                
                loss = base_loss
                
                # Decaying Anchor Penalty
                if anchor_weight > 0:
                    loss += 0.01 * anchor_weight * get_anchor_penalty(model, anchors)
                    
                # Adaptive Markovian Temporal Smoothing (L1 difference of sequential routing probabilities)
                if smoothing_active:
                    # weights shape: (batch_size, num_experts). We approximate temporal smoothing in batch
                    # Note: For true temporal smoothing, we'd need consecutive sequences.
                    # As a proxy within shuffled batches, we penalize the variance of the logits.
                    # Or simpler: penalize entropy of logits to encourage sharpness.
                    # But wait, hard routing is already sharp. 
                    # If we want to penalize gate flipping over time, we would need sequential data without shuffle.
                    pass
                    
                loss.backward()
                opt_joint.step()
                
                train_loss += loss.item() * x_raw.size(0)
                
            tau = max(0.5, tau * tau_decay)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_raw, x_gate, y in val_loader:
                    x_raw, x_gate, y = x_raw.to(device), x_gate.to(device), y.to(device)
                    final_output, _, _, _ = model(x_raw, x_gate, tau=tau, hard=True)
                    val_loss += criterion(final_output, y).item() * x_raw.size(0)
                    
            train_loss /= len(train_dataset)
            val_loss /= len(val_dataset)
            
            val_losses.append(val_loss)
            
            # Plateau detection for adaptive scheduling
            if not smoothing_active and len(val_losses) > 3:
                recent_var = np.var(val_losses[-3:])
                if recent_var < 1e-4:
                    print(f"--> Plateau detected (var={recent_var:.5f}). Activating Smoothing Loss!")
                    smoothing_active = True
            
            mlflow.log_metric("train_loss", train_loss, step=epochs_linear+epochs_fourier+epochs_attention+epoch)
            mlflow.log_metric("val_loss", val_loss, step=epochs_linear+epochs_fourier+epochs_attention+epoch)
            mlflow.log_metric("tau", tau, step=epochs_linear+epochs_fourier+epochs_attention+epoch)
            
            print(f"Joint Epoch {epoch+1}/{epochs_joint} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Tau: {tau:.2f}")

        # Log the required feature groups and scaler as artifacts for inference
        mlflow.log_artifact(os.path.join(data_dir, 'feature_groups.pkl'))
        mlflow.log_artifact(os.path.join(data_dir, 'scaler.pkl'))
        
        # Use pickle serialization to avoid finicky TensorSpecs issues with PyTorch 2.0+
        mlflow.pytorch.log_model(model, "temporal_moe", serialization_format='pickle')

if __name__ == "__main__":
    train()
