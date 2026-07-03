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

from models.tft import TemporalFusionTransformer

def test_shock_recovery():
    print("--- Connecting to DagsHub MLflow ---")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rgorasia19"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "26b84ad06cab16667533b56a5cebe4ebf27d8f9a"
    
    dagshub.init(repo_owner="rgorasia19", repo_name="Energy-Intelligence-System", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow")
    
    experiment = mlflow.get_experiment_by_name("v7_tft_multistep")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    run_id = runs.iloc[0].run_id
    print(f"Latest Run ID: {run_id}")
    
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
    model_path = os.path.join(local_dir, "tft_v7.pth")
    
    feature_groups_path = "../../datalake/v7_tensors/feature_groups.pkl"
    feature_groups = joblib.load(feature_groups_path)
    obs_cols = list(feature_groups['obs_cols'])
    known_cols = list(feature_groups['known_cols'])
    static_cols = list(feature_groups['static_cols'])
    
    seq_len = 48
    horizon = 48
    
    target_idx = obs_cols.index('obs_ND')

    model = TemporalFusionTransformer(
        num_static_vars=len(static_cols),
        num_future_vars=len(known_cols),
        num_past_vars=len(obs_cols),
        d_model=32,
        num_heads=4,
        seq_len=seq_len,
        horizon=horizon,
        target_in_past_idx=target_idx,
        dropout=0.3
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print("--- Loading Test Data ---")
    test_data = pd.read_parquet('../../datalake/v7_tensors/test.parquet')
    
    X_obs = torch.tensor(test_data[obs_cols].values, dtype=torch.float32)
    X_known = torch.tensor(test_data[known_cols].values, dtype=torch.float32)
    X_static = torch.tensor(test_data[static_cols].values, dtype=torch.float32)
    y_abs = torch.tensor(test_data['ND_CURRENT'].values, dtype=torch.float32)
    
    print("--- Injecting Synthetic Shock ---")
    start_idx = 1000 # Random stable window in test set
    eval_len = 336 # 1 week of half-hourly steps
    
    X_obs_shocked = X_obs.clone()
    y_abs_shocked = y_abs.clone()
    
    # Inject +5 std spike into target demand for 12 hours
    shock_start = start_idx + seq_len + 48 # Starts on Day 2
    shock_duration = 24 # 12 hours
    shock_end = shock_start + shock_duration
    spike_magnitude = 5.0
    
    X_obs_shocked[shock_start:shock_end, target_idx] += spike_magnitude
    y_abs_shocked[shock_start:shock_end] += spike_magnitude
    
    tft_preds = []
    naive_preds = []
    actuals = []
    
    print("--- Running Sliding Window Forecast ---")
    for i in range(eval_len):
        idx = start_idx + i
        static = X_static[idx].unsqueeze(0).to(device)
        past_obs = X_obs_shocked[idx : idx + seq_len].unsqueeze(0).to(device)
        future_known = X_known[idx : idx + seq_len + horizon].unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_abs, _, _, _ = model(static, past_obs, future_known)
            
        tft_preds.append(pred_abs[0, 0].item())
        
        # Seasonal Naive: Exactly 48 steps prior
        naive_val = past_obs[0, 0, target_idx].item()
        naive_preds.append(naive_val)
        
        actuals.append(y_abs_shocked[idx + seq_len].item())
        
    tft_preds = np.array(tft_preds)
    naive_preds = np.array(naive_preds)
    actuals = np.array(actuals)
    
    print("--- Computing Recovery Dynamics ---")
    tft_errors = np.abs(actuals - tft_preds)
    naive_errors = np.abs(actuals - naive_preds)
    
    # Pre-shock baseline (Day 1)
    pre_shock_errors = tft_errors[:48]
    pre_shock_mae = np.mean(pre_shock_errors)
    pre_shock_std = np.std(pre_shock_errors)
    recovery_threshold = pre_shock_mae + pre_shock_std
    print(f"Pre-Shock TFT MAE: {pre_shock_mae:.4f} (Threshold: {recovery_threshold:.4f})")
    
    # Post-shock recovery (After Day 2)
    post_shock_start = 48 + shock_duration
    post_shock_tft_errors = tft_errors[post_shock_start:]
    
    # Find TFT recovery step
    # We use a 3-step moving average to smooth the error curve for a stable threshold crossing
    tft_smoothed_errors = pd.Series(post_shock_tft_errors).rolling(3, min_periods=1).mean().values
    tft_recovery_step = -1
    for step, err in enumerate(tft_smoothed_errors):
        if err < recovery_threshold:
            tft_recovery_step = step
            break
            
    print(f"TFT Recovery Steps: {tft_recovery_step if tft_recovery_step != -1 else 'Did not recover'}")
    
    print("--- Generating Plots ---")
    plot_filename = f"../plots/run10/shock_recovery_analysis_v7_{run_id[:8]}.png"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Trajectory
    ax1.plot(actuals, label='Actual Demand (Shocked)', color='black', linewidth=2, alpha=0.8)
    ax1.plot(tft_preds, label='TFT Forecast (1-Step)', color='blue', linewidth=2)
    ax1.plot(naive_preds, label='Seasonal Naive (T-48)', color='red', linestyle='--', linewidth=1.5)
    
    ax1.axvspan(48, post_shock_start, color='orange', alpha=0.2, label='Exogenous Shock Window')
    ax1.axvspan(post_shock_start + 48, post_shock_start + 48 + shock_duration, color='red', alpha=0.1, label='Naive Echo Window')
    
    ax1.set_title('Synthetic Demand Shock (+5 Std): TFT vs Seasonal Naive')
    ax1.set_ylabel('Demand (Scaled)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error Decay
    ax2.plot(tft_errors, label='TFT Absolute Error', color='blue', alpha=0.7)
    ax2.plot(naive_errors, label='Naive Absolute Error', color='red', alpha=0.5, linestyle='--')
    ax2.axhline(recovery_threshold, color='green', linestyle=':', label='TFT Recovery Threshold')
    
    if tft_recovery_step != -1:
        ax2.axvline(post_shock_start + tft_recovery_step, color='green', linestyle='-', label=f'TFT Recovered (Step {tft_recovery_step})')
        
    ax2.axvspan(48, post_shock_start, color='orange', alpha=0.2)
    ax2.set_title('Error Decay & Recovery Horizon')
    ax2.set_xlabel('Time Step (30m intervals)')
    ax2.set_ylabel('Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=600)
    plt.close()
    print(f"Saved plot to {plot_filename}")

if __name__ == "__main__":
    test_shock_recovery()
