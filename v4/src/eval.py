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

# Re-import model components to satisfy pickle deserialization
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

def evaluate():
    print("--- Connecting to DagsHub MLflow ---")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rgorasia19"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "26b84ad06cab16667533b56a5cebe4ebf27d8f9a"
    
    dagshub.init(repo_owner="rgorasia19", repo_name="Energy-Intelligence-System", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow")
    
    # Get latest run
    experiment = mlflow.get_experiment_by_name("v4_moe")
    if experiment is None:
        raise ValueError("Experiment 'v4_moe' not found. Have you trained a model yet?")
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    if runs.empty:
        raise ValueError("No runs found in experiment 'v4_moe'.")
        
    run_id = runs.iloc[0].run_id
    short_run_id = run_id[:8]
    print(f"Latest Run ID: {run_id}")
    
    print("--- Downloading Model and Artifacts ---")
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
    scaler_path = os.path.join(local_dir, "scaler.pkl")
    feature_groups_path = os.path.join(local_dir, "feature_groups.pkl")
    model_path = os.path.join(local_dir, "temporal_moe.pth")
    
    target_scaler = joblib.load(scaler_path)
    feature_groups = joblib.load(feature_groups_path)
    raw_cols = feature_groups['raw_cols']
    gate_cols = feature_groups['gate_cols']
    
    # Instantiate architecture and load robust state_dict
    seq_len = 48
    model = TemporalMoE(raw_feature_dim=len(raw_cols), gate_feature_dim=len(gate_cols), seq_len=seq_len)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    print("--- Loading Test Data ---")
    test_data = pd.read_parquet('../../datalake/moe_tensors/test.parquet')
    
    seq_len = 48
    batch_size = 2048
    
    test_dataset = MoEDataset(test_data, raw_cols, gate_cols, seq_length=seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    test_predictions = []
    actual_targets = []
    all_weights = []
    
    print("--- Generating Predictions ---")
    with torch.no_grad():
        for x_raw, x_gate, y_label in test_loader:
            x_raw, x_gate = x_raw.to(device), x_gate.to(device)
            
            # hard=False means we use soft routing (gate_probs) for probabilistic decoding
            final_output, weights, logits, _ = model(x_raw, x_gate, tau=1.0, hard=False)
            
            test_predictions.append(final_output.cpu().numpy())
            actual_targets.append(y_label.numpy())
            all_weights.append(weights.cpu().numpy())
            
    test_pred_scaled = np.concatenate(test_predictions).flatten()
    actual_targets_scaled = np.concatenate(actual_targets).flatten()
    all_weights_flat = np.concatenate(all_weights)
    
    # Inverse transform
    test_pred_flat = target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    actual_targets_flat = target_scaler.inverse_transform(actual_targets_scaled.reshape(-1, 1)).flatten()
    
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
    expert_names = ["Linear (Base)", "Fourier (Seasonal)", "Attention (Volatile)"]
    for i, mass in enumerate(expected_occupancy):
        name = expert_names[i] if i < len(expert_names) else f"Expert {i}"
        print(f"  {name}: {mass*100:.2f}%")
    print("-------------------\n")
    
    print("--- Generating Plot ---")
    plot_filename = f"eval_plot_{short_run_id}.png"
    plt.figure(figsize=(12, 6))
    plt.plot(actual_targets_flat[:2000], label='Actual Demand', color='blue', alpha=0.5)
    plt.plot(test_pred_flat[:2000], label='Predicted Demand', color='red', alpha=0.5)
    plt.title(f'Actual vs Predicted Demand (First 2000 samples) - Run: {short_run_id}')
    plt.xlabel('Time Step')
    plt.ylabel('Demand')
    plt.legend()
    plt.savefig(plot_filename, dpi=600)
    print(f"Saved plot to {plot_filename}")

if __name__ == "__main__":
    evaluate()
