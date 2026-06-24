import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import dagshub
import mlflow
import mlflow.pytorch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader

from models.predictor import UnifiedRegimeModel

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

def evaluate():
    print("--- Connecting to DagsHub MLflow ---")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rgorasia19"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "26b84ad06cab16667533b56a5cebe4ebf27d8f9a"
    
    dagshub.init(repo_owner="rgorasia19", repo_name="Energy-Intelligence-System", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow")
    
    experiment = mlflow.get_experiment_by_name("v5_attention_regime")
    if experiment is None:
        raise ValueError("Experiment 'v5_attention_regime' not found.")
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    if runs.empty:
        raise ValueError("No runs found in experiment 'v5_attention_regime'.")
        
    run_id = runs.iloc[0].run_id
    short_run_id = run_id[:8]
    print(f"Latest Run ID: {run_id}")
    
    print("--- Downloading Model and Artifacts ---")
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
    scaler_path = os.path.join(local_dir, "scaler.pkl")
    feature_groups_path = os.path.join(local_dir, "feature_groups.pkl")
    model_path = os.path.join(local_dir, "regime_attention.pth")
    
    target_scaler = joblib.load(scaler_path)
    feature_groups = joblib.load(feature_groups_path)
    raw_cols = feature_groups['raw_cols']
    gate_cols = feature_groups['gate_cols']
    
    seq_len = 48
    num_regimes = 3
    model = UnifiedRegimeModel(
        raw_feature_dim=len(raw_cols), 
        gate_feature_dim=len(gate_cols), 
        seq_len=seq_len,
        num_regimes=num_regimes,
        d_model=64
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    print("--- Loading Test Data ---")
    test_data = pd.read_parquet('../../datalake/moe_tensors/test.parquet')
    
    batch_size = 2048
    test_dataset = MoEDataset(test_data, raw_cols, gate_cols, seq_length=seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    test_predictions = []
    actual_targets = []
    all_weights = []
    all_attention = []
    
    print("--- Generating Predictions ---")
    with torch.no_grad():
        for x_raw, x_gate, y_label in test_loader:
            x_raw, x_gate = x_raw.to(device), x_gate.to(device)
            
            # tau=0.8 at inference based on final annealed value
            y_hat, p_regime, _, attention_maps = model(x_raw, x_gate, tau=0.8, return_attention=True)
            
            test_predictions.append(y_hat.cpu().numpy())
            actual_targets.append(y_label.numpy())
            
            # Take just the last step of the sequence for transition plotting
            all_weights.append(p_regime[:, -1, :].cpu().numpy())
            
            # Take the attention map from the LAST block, for the LAST sequence step
            # attention_maps is a list of [B, seq_len+1, seq_len+1] tensors across blocks
            if attention_maps:
                last_block_attn = attention_maps[-1] # (B, seq_len+1, seq_len+1)
                # We want the attention the LAST sequence token pays to previous tokens
                last_token_attn = last_block_attn[:, -1, 1:] # (B, seq_len)
                all_attention.append(last_token_attn.cpu().numpy())
            
    test_pred_flat = np.concatenate(test_predictions).flatten()
    actual_targets_flat = np.concatenate(actual_targets).flatten()
    all_weights_flat = np.concatenate(all_weights)
    all_attention_flat = np.concatenate(all_attention) if len(all_attention) > 0 else None
    
    print("--- Evaluating Metrics ---")
    mae = mean_absolute_error(actual_targets_flat, test_pred_flat)
    rmse = np.sqrt(mean_squared_error(actual_targets_flat, test_pred_flat))
    r2 = r2_score(actual_targets_flat, test_pred_flat)
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")
    
    print("\n--- Diagnostics ---")
    expected_occupancy = all_weights_flat.mean(axis=0)
    print("Expected State Occupancy:")
    for i, mass in enumerate(expected_occupancy):
        print(f"  Regime {i}: {mass*100:.2f}%")
        
    entropies = -np.sum(all_weights_flat * np.log(all_weights_flat + 1e-8), axis=-1)
    print(f"Mean Per-Timestep Entropy: {entropies.mean():.4f}")
    print("-------------------\n")
    
    print("--- Generating Plots ---")
    
    # 1. Prediction Plot
    plot_filename = f"eval_pred_{short_run_id}.png"
    plt.figure(figsize=(12, 6))
    plt.plot(actual_targets_flat[:2000], label='Actual Demand', color='blue', alpha=0.5)
    plt.plot(test_pred_flat[:2000], label='Predicted Demand', color='red', alpha=0.5)
    plt.title(f'Actual vs Predicted Demand (First 2000 samples) - Run: {short_run_id}')
    plt.xlabel('Time Step')
    plt.ylabel('Demand')
    plt.legend()
    plt.savefig(plot_filename, dpi=600)
    
    # 2. Regime Transitions Plot (Argmax)
    trans_filename = f"eval_regimes_{short_run_id}.png"
    plt.figure(figsize=(12, 3))
    regime_argmax = np.argmax(all_weights_flat[:2000], axis=1)
    plt.plot(regime_argmax, label='Regime (Hardened)', color='purple')
    plt.title('Regime Transitions over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Regime ID')
    plt.yticks(range(num_regimes))
    plt.savefig(trans_filename, dpi=600)
    
    # 3. Attention Heatmap (Averaged over first 100 samples)
    if all_attention_flat is not None:
        attn_filename = f"eval_attention_{short_run_id}.png"
        plt.figure(figsize=(10, 2))
        avg_attn = all_attention_flat[:100].mean(axis=0) # (seq_len,)
        plt.imshow(avg_attn.reshape(1, -1), cmap='hot', aspect='auto')
        plt.title('Average Attention Weights over Context Window (Last step looking back)')
        plt.xlabel('Lookback step')
        plt.yticks([])
        plt.colorbar()
        plt.savefig(attn_filename, dpi=600)
        
    print(f"Saved plots to current directory.")

if __name__ == "__main__":
    evaluate()
