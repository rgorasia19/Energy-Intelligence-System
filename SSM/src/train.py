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
from torch.utils.data import DataLoader

from feature_prep import SSMDataset
from models.ssm import LatentSSM, SSMLoss

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

    data_dir = '../../datalake/ssm_tensors/'
    train_data = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    val_data = pd.read_parquet(os.path.join(data_dir, 'val.parquet'))
    
    col_info = joblib.load(os.path.join(data_dir, 'columns.pkl'))
    original_feature_cols = col_info['feature_columns']
    target_cols = col_info['target_columns']
    
    demand_cols = ['ND', 'TSD', 'ENGLAND_WALES_DEMAND']
    gen_cols = ['GAS', 'COAL', 'NUCLEAR', 'WIND', 'GENERATION']
    
    demand_idx = [target_cols.index(c) for c in demand_cols]
    gen_idx = [target_cols.index(c) for c in gen_cols]
    
    seq_len = 60
    horizon = 30
    batch_size = 256
    latent_dim = 8
    hidden_dim = 64
    num_regimes = 4

    weather_cols = ['temperature_2m', 'cloudcover', 'windspeed_10m', 'shortwave_radiation']
    fourier_cols = [c for c in original_feature_cols if '_sin_k' in c or '_cos_k' in c]
    calendar_cols = ['is_bank_holiday'] if 'is_bank_holiday' in original_feature_cols else []
    
    # Restrict input features strictly to the 5 specified categories
    feature_cols = demand_cols + gen_cols + fourier_cols + calendar_cols + weather_cols
    known_columns = fourier_cols + calendar_cols + weather_cols
    known_dim = len(known_columns)

    # Load scaler and extract scales/centers for demand and generation targets
    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    cols_to_scale = [c for c in original_feature_cols if not c.endswith('_available')]
    
    demand_scales = np.array([scaler.scale_[cols_to_scale.index(c)] for c in demand_cols])
    demand_centers = np.array([scaler.center_[cols_to_scale.index(c)] for c in demand_cols])
    
    gen_scales = np.array([scaler.scale_[cols_to_scale.index(c)] for c in gen_cols])
    gen_centers = np.array([scaler.center_[cols_to_scale.index(c)] for c in gen_cols])

    train_dataset = SSMDataset(train_data, seq_len=seq_len, horizon=horizon, 
                               feature_columns=feature_cols, target_columns=target_cols,
                               known_columns=known_columns)
    val_dataset = SSMDataset(val_data, seq_len=seq_len, horizon=horizon, 
                             feature_columns=feature_cols, target_columns=target_cols,
                             known_columns=known_columns)
    
    num_workers = 4 if device == "cuda" else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                               num_workers=num_workers, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=(device=="cuda"))

    model = LatentSSM(
        input_dim=len(feature_cols),
        demand_dim=len(demand_cols),
        gen_dim=len(gen_cols),
        known_dim=known_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_regimes=num_regimes,
        dropout=0.2,
        fourier_dim=len(fourier_cols),
        demand_scale=demand_scales,
        demand_center=demand_centers,
        gen_scale=gen_scales,
        gen_center=gen_centers
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = SSMLoss(
        kl_z_weight=1.0, 
        kl_r_weight=1.0, 
        entropy_weight=0.1,
        demand_scale=demand_scales,
        demand_center=demand_centers,
        gen_scale=gen_scales,
        gen_center=gen_centers
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    
    epochs = 100 # increased epochs for complex H-SSM training
    
    # DagsHub Auth
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rgorasia19"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "26b84ad06cab16667533b56a5cebe4ebf27d8f9a"
    
    dagshub.init(repo_owner="rgorasia19", repo_name="Energy-Intelligence-System", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow")
    mlflow.set_experiment("hssm_probabilistic")
    
    with mlflow.start_run():
        mlflow.log_params({
            "seq_len": seq_len,
            "horizon": horizon,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "num_regimes": num_regimes,
            "batch_size": batch_size,
            "learning_rate": 1e-3,
        })
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_metrics_sum = {'loss_demand': 0, 'loss_gen': 0, 'kl_z': 0, 'kl_r': 0, 'entropy_r': 0}
            
            # Curriculum: grow horizon linearly over first 50% of epochs
            curriculum_frac = min(1.0, (epoch + 1) / max(1, epochs / 2))
            max_horizon = max(1, int(np.ceil(horizon * curriculum_frac)))
                
            for batch in train_loader:
                enc_inputs = batch['encoder_inputs'].to(device)
                dec_inputs = batch['decoder_inputs'].to(device)
                dec_targets = batch['decoder_targets'].to(device)
                dec_masks = batch['decoder_mask'].to(device)
                
                # Noise Injection and Shocks
                enc_inputs = enc_inputs + torch.randn_like(enc_inputs) * 0.05
                dec_inputs = dec_inputs + torch.randn_like(dec_inputs) * 0.05
                
                # Inject synthetic target spike into a random segment (5% probability)
                if torch.rand(1).item() < 0.05:
                    spike_len = min(max_horizon, torch.randint(1, 4, (1,)).item())
                    start_idx = torch.randint(0, max_horizon - spike_len + 1, (1,)).item()
                    shock = torch.empty(dec_targets.shape[0], 1, 1, device=device).uniform_(1.2, 2.0)
                    dec_targets[:, start_idx:start_idx+spike_len, :] = dec_targets[:, start_idx:start_idx+spike_len, :] * shock
                
                # Curriculum horizon
                current_horizon = max_horizon
                
                dec_inputs_trunc = dec_inputs[:, :current_horizon, :]
                dec_targets_trunc = dec_targets[:, :current_horizon, :]
                dec_masks_trunc = dec_masks[:, :current_horizon, :]
                
                optimizer.zero_grad(set_to_none=True)
                
                tau = max(0.5, 2.0 * (0.95 ** epoch))
                
                K = 5
                enc_inputs_k = enc_inputs.repeat_interleave(K, dim=0)
                dec_inputs_trunc_k = dec_inputs_trunc.repeat_interleave(K, dim=0)
                dec_targets_trunc_k = dec_targets_trunc.repeat_interleave(K, dim=0)
                dec_masks_trunc_k = dec_masks_trunc.repeat_interleave(K, dim=0)
                
                outputs = model(enc_inputs_k, dec_inputs_trunc_k, current_horizon, target_seq=dec_targets_trunc_k, tau=tau)
                
                loss, metrics = criterion(outputs, dec_targets_trunc_k, dec_masks_trunc_k, demand_idx, gen_idx, epoch, epochs, free_bits_z=0.5, free_bits_r=1.0, k_samples=K)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                for k in train_metrics_sum:
                    train_metrics_sum[k] += metrics[k]
                
            train_loss /= len(train_loader)
            for k in train_metrics_sum:
                train_metrics_sum[k] /= len(train_loader)
            
            model.eval()
            val_loss = 0.0
            val_metrics_sum = {'loss_demand': 0, 'loss_gen': 0, 'kl_z': 0, 'kl_r': 0, 'entropy_r': 0}
            
            with torch.no_grad():
                for batch in val_loader:
                    enc_inputs = batch['encoder_inputs'].to(device)
                    dec_inputs = batch['decoder_inputs'].to(device)
                    dec_targets = batch['decoder_targets'].to(device)
                    dec_masks = batch['decoder_mask'].to(device)
                    
                    outputs = model(enc_inputs, dec_inputs, horizon, target_seq=None, tau=0.5) # use 0.5 in eval for harder choices
                    
                    loss, metrics = criterion(outputs, dec_targets, dec_masks, demand_idx, gen_idx, epochs, epochs, free_bits_z=0.0, free_bits_r=0.0)
                    
                    val_loss += loss.item()
                    for k in val_metrics_sum:
                        val_metrics_sum[k] += metrics[k]
                        
            val_loss /= len(val_loader)
            for k in val_metrics_sum:
                val_metrics_sum[k] /= len(val_loader)
            
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} (Demand NLL: {val_metrics_sum['loss_demand']:.4f})")
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_demand_nll": val_metrics_sum['loss_demand'],
                "val_gen_nll": val_metrics_sum['loss_gen'],
                "train_kl_z": train_metrics_sum['kl_z'],
                "train_kl_r": train_metrics_sum['kl_r'],
                "train_entropy_r": train_metrics_sum['entropy_r']
            }, step=epoch)
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
                
        # Save model and artifacts
        os.makedirs('saved_models', exist_ok=True)
        model_path = "saved_models/ssm_v1.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        
        print("Training complete and model saved.")

if __name__ == "__main__":
    train()
