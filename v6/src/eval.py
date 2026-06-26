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

from models.end_to_end import ContinuousStateForecaster
from diagnostics import (
    calculate_latent_drift,
    calculate_temporal_smoothness,
    plot_forecast_error_by_volatility,
    plot_latent_trajectory,
    feature_influence_analysis
)

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

def evaluate():
    print("--- Connecting to DagsHub MLflow ---")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rgorasia19"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "26b84ad06cab16667533b56a5cebe4ebf27d8f9a"
    
    dagshub.init(repo_owner="rgorasia19", repo_name="Energy-Intelligence-System", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow")
    
    experiment = mlflow.get_experiment_by_name("v6_continuous_latent_state")
    if experiment is None:
        raise ValueError("Experiment 'v6_continuous_latent_state' not found.")
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    if runs.empty:
        raise ValueError("No runs found in experiment 'v6_continuous_latent_state'.")
        
    run_id = runs.iloc[0].run_id
    short_run_id = run_id[:8]
    print(f"Latest Run ID: {run_id}")
    
    print("--- Downloading Model and Artifacts ---")
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
    scaler_path = os.path.join(local_dir, "scaler.pkl")
    aux_scaler_path = os.path.join(local_dir, "aux_scaler.pkl")
    feature_groups_path = os.path.join(local_dir, "feature_groups.pkl")
    model_path = os.path.join(local_dir, "continuous_latent.pth")
    
    target_scaler = joblib.load(scaler_path)
    aux_scaler = joblib.load(aux_scaler_path)
    feature_groups = joblib.load(feature_groups_path)
    raw_cols = feature_groups['raw_cols']
    gate_cols = feature_groups['gate_cols']
    
    seq_len = 48
    latent_dim = 32
    
    model = ContinuousStateForecaster(
        raw_feature_dim=len(raw_cols), 
        gate_feature_dim=len(gate_cols), 
        seq_len=seq_len,
        d_model=64,
        latent_dim=latent_dim
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    print("--- Loading Test Data ---")
    test_data = pd.read_parquet('../../datalake/moe_tensors/test.parquet')
    
    batch_size = 2048
    test_dataset = TimeSeriesDataset(test_data, raw_cols, gate_cols, seq_length=seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    test_predictions = []
    actual_targets = []
    
    all_z_seq = []
    actual_aux_targets = []
    pred_aux_targets = []
    
    print("--- Generating Predictions ---")
    with torch.no_grad():
        for x_raw, x_gate, y_label, y_vol, y_trend in test_loader:
            x_raw, x_gate = x_raw.to(device), x_gate.to(device)
            
            y_hat, z_seq, aux_preds = model(x_raw, x_gate)
            
            test_predictions.append(y_hat.cpu().numpy())
            actual_targets.append(y_label.numpy())
            
            actual_aux_targets.append(torch.stack([y_vol, y_trend], dim=1).numpy())
            pred_aux_targets.append(aux_preds[:, -1, :].cpu().numpy())
            
            # Save latent state trajectories
            all_z_seq.append(z_seq.cpu().numpy())
            
    test_pred_flat = np.concatenate(test_predictions).flatten()
    actual_targets_flat = np.concatenate(actual_targets).flatten()
    
    all_z_seq = np.concatenate(all_z_seq, axis=0) # (N, seq_len, latent_dim)
    z_seq_flat = all_z_seq[:, -1, :] # Take the last timestep for flat diagnostics
    
    actual_aux_flat = np.concatenate(actual_aux_targets)
    pred_aux_flat = np.concatenate(pred_aux_targets)
    
    print("--- Evaluating Metrics ---")
    mae = mean_absolute_error(actual_targets_flat, test_pred_flat)
    rmse = np.sqrt(mean_squared_error(actual_targets_flat, test_pred_flat))
    r2 = r2_score(actual_targets_flat, test_pred_flat)
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")
    
    print("\n--- Diagnostics (v6 Continuous State Space) ---")
    
    mean_drift, var_drift = calculate_latent_drift(z_seq_flat)
    print(f"Mean Latent Drift (norm): {np.linalg.norm(mean_drift):.4f}")
    print(f"Mean Latent Variance: {np.mean(var_drift):.4f}")
    
    smoothness = calculate_temporal_smoothness(all_z_seq)
    print(f"Temporal Smoothness (||z_t - z_{{t-1}}||^2): {smoothness:.4f}")
    
    # Inverse transform aux targets to raw units
    actual_aux_raw = aux_scaler.inverse_transform(actual_aux_flat)
    actual_vol_raw = actual_aux_raw[:, 0]
    actual_trend_raw = actual_aux_raw[:, 1]
    
    vol_corr, trend_corr = feature_influence_analysis(z_seq_flat, actual_vol_raw, actual_trend_raw)
    top_vol_dim = np.argmax(np.abs(vol_corr))
    top_trend_dim = np.argmax(np.abs(trend_corr))
    print(f"Highest Volatility Correlation in Latent Dim {top_vol_dim}: {vol_corr[top_vol_dim]:.4f}")
    print(f"Highest Trend Correlation in Latent Dim {top_trend_dim}: {trend_corr[top_trend_dim]:.4f}")
    
    print("-------------------\n")
    
    print("--- Generating Plots ---")
    
    # 1. Prediction Plot
    plot_filename = f"eval_pred_v6_{short_run_id}.png"
    plt.figure(figsize=(12, 6))
    plt.plot(actual_targets_flat[:2000], label='Actual Demand', color='blue', alpha=0.5)
    plt.plot(test_pred_flat[:2000], label='Predicted Demand', color='red', alpha=0.5)
    plt.title(f'Actual vs Predicted Demand (v6) - Run: {short_run_id}')
    plt.xlabel('Time Step')
    plt.ylabel('Demand')
    plt.legend()
    plt.savefig(plot_filename, dpi=600)
    plt.close()
    
    # 2. PCA Latent Trajectory
    pca_filename = f"eval_latent_pca_v6_{short_run_id}.png"
    plot_latent_trajectory(z_seq_flat, save_path=pca_filename, method='pca')
    
    # 3. Forecast Error by Volatility
    abs_errors = np.abs(actual_targets_flat - test_pred_flat)
    error_vol_filename = f"eval_error_by_vol_v6_{short_run_id}.png"
    plot_forecast_error_by_volatility(abs_errors, actual_vol_raw, save_path=error_vol_filename)
        
    print(f"Saved plots to current directory.")

if __name__ == "__main__":
    evaluate()
