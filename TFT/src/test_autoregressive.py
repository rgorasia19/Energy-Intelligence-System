import os
import sys

if sys.platform == 'win32' and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import dagshub
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.tft import TemporalFusionTransformer

def evaluate_autoregressive():
    print("--- Connecting to DagsHub MLflow ---")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rgorasia19"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "26b84ad06cab16667533b56a5cebe4ebf27d8f9a"
    
    dagshub.init(repo_owner="rgorasia19", repo_name="Energy-Intelligence-System", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow")
    
    experiment = mlflow.get_experiment_by_name("v7_tft_multistep")
    if experiment is None:
        raise ValueError("Experiment 'v7_tft_multistep' not found.")
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    if runs.empty:
        raise ValueError("No runs found.")
        
    run_id = runs.iloc[0].run_id
    short_run_id = run_id[:8]
    print(f"Latest Run ID: {run_id}")
    
    print("--- Downloading Model and Artifacts ---")
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
    model_path = os.path.join(local_dir, "tft_v7.pth")
    
    feature_groups_path = "../../datalake/v7_tensors/feature_groups.pkl"
    feature_groups = joblib.load(feature_groups_path)
    obs_cols = feature_groups['obs_cols']
    known_cols = feature_groups['known_cols']
    static_cols = feature_groups['static_cols']
    
    seq_len = 48
    horizon = 48
    d_model = 32
    num_heads = 4
    
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
        dropout=0.3
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    print("--- Loading Test Data ---")
    test_data = pd.read_parquet('../../datalake/v7_tensors/test.parquet')
    
    X_obs = torch.tensor(test_data[obs_cols].values, dtype=torch.float32)
    X_known = torch.tensor(test_data[known_cols].values, dtype=torch.float32)
    X_static = torch.tensor(test_data[static_cols].values, dtype=torch.float32)
    y_abs = torch.tensor(test_data['ND_CURRENT'].values, dtype=torch.float32)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    X_obs_ar = X_obs.clone().to(device)
    X_known = X_known.to(device)
    X_static = X_static.to(device)
    y_abs = y_abs.to(device)
    
    # Identify weather variables to simulate forecast degradation
    weather_vars = ['known_temperature_2m', 'known_cloudcover', 'known_windspeed_10m', 'known_shortwave_radiation']
    weather_indices = [known_cols.index(col) for col in weather_vars if col in known_cols]
    
    # Linearly scale noise standard deviation from 0.0 to 0.5 over the 48-step horizon
    noise_stds = torch.linspace(0.0, 0.5, horizon, device=device).unsqueeze(1)
    
    ar_predictions = []
    actuals = []
    
    print("--- Running Autoregressive Loop ---")
    num_steps = len(X_obs) - seq_len - horizon + 1
    
    # Step by horizon to do non-overlapping consecutive predictions
    for idx in range(0, num_steps, horizon):
        static = X_static[idx].unsqueeze(0)
        past_obs = X_obs_ar[idx : idx + seq_len].unsqueeze(0)
        future_known = X_known[idx : idx + seq_len + horizon].unsqueeze(0)
        
        noisy_future_known = future_known.clone()
        if len(weather_indices) > 0:
            noise = torch.randn(horizon, len(weather_indices), device=device) * noise_stds
            for i, w_idx in enumerate(weather_indices):
                noisy_future_known[0, seq_len:, w_idx] += noise[:, i]
                
            # Black Swan Event Injection: ~2% chance of a massive sustained anomaly in the forecast
            if torch.rand(1).item() < 0.02:
                # Pick a random weather variable to spike
                spike_var_idx = torch.randint(0, len(weather_indices), (1,)).item()
                w_idx = weather_indices[spike_var_idx]
                
                # Create a massive spike (+4 to +6 stds or -4 to -6 stds)
                spike_magnitude = (torch.rand(1).item() * 2 + 4) * (1 if torch.rand(1).item() > 0.5 else -1)
                
                # Apply it to a block of 12-24 steps (6-12 hours)
                spike_start = torch.randint(0, max(1, horizon - 12), (1,)).item()
                spike_duration = torch.randint(12, 24, (1,)).item()
                spike_end = min(horizon, spike_start + spike_duration)
                
                noisy_future_known[0, seq_len + spike_start : seq_len + spike_end, w_idx] += spike_magnitude
                
        with torch.no_grad():
            pred_abs, _, _, _ = model(static, past_obs, noisy_future_known)
            
        # Overwrite the future sequence in X_obs_ar with our predictions.
        # obs_ND is column 0 based on target_idx.
        # The true exogenous variables (INTER_PC0, GEN_PC0, CAP_PC0) are untouched!
        X_obs_ar[idx + seq_len : idx + seq_len + horizon, target_idx] = pred_abs[0]
        
        ar_predictions.append(pred_abs[0].cpu().numpy())
        actuals.append(y_abs[idx + seq_len : idx + seq_len + horizon].cpu().numpy())
        
        if (idx // horizon) % 50 == 0:
            print(f"Processed step {idx} / {num_steps}")

    print("--- Compiling Trajectories ---")
    pred_nd_arr = np.concatenate(ar_predictions, axis=0)
    actual_nd_arr = np.concatenate(actuals, axis=0)
    
    mae = mean_absolute_error(actual_nd_arr, pred_nd_arr)
    rmse = np.sqrt(mean_squared_error(actual_nd_arr, pred_nd_arr))
    r2 = r2_score(actual_nd_arr, pred_nd_arr)
    
    print(f"Long-term Autoregressive MAE: {mae:.4f}")
    print(f"Long-term Autoregressive RMSE: {rmse:.4f}")
    print(f"Long-term Autoregressive R2: {r2:.4f}")
    
    print("--- Generating Plot ---")
    plot_filename = f"../plots/run10/autoregressive_black_swan_v7_{short_run_id}.png"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    
    # Plotting the whole trajectory is messy, let's plot a few months (e.g. 10000 steps)
    viz_steps = min(10000, len(actual_nd_arr))
    
    plt.figure(figsize=(24, 6))
    plt.plot(actual_nd_arr[:viz_steps], label='Actual Demand (Scaled)', color='blue', alpha=0.5, linewidth=1)
    plt.plot(pred_nd_arr[:viz_steps], label='Black Swan AR Forecast (Scaled)', color='red', alpha=0.8, linewidth=1)
    plt.title(f'Autoregressive Forecast with Black Swan Outliers - First {viz_steps} Steps (v7) - Run: {short_run_id}')
    plt.xlabel('Time Step (30min intervals)')
    plt.ylabel('Demand (Scaled)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=600)
    plt.close()
    
    print(f"Saved plot to {plot_filename}")

if __name__ == "__main__":
    evaluate_autoregressive()
