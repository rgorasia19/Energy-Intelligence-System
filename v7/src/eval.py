import os
import sys

if sys.platform == 'win32' and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import dagshub
import mlflow
import mlflow.pytorch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf
from torch.utils.data import DataLoader

from models.tft import TemporalFusionTransformer
from train import TFTDataset

def evaluate():
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
        raise ValueError("No runs found in experiment 'v7_tft_multistep'.")
        
    run_id = runs.iloc[0].run_id
    short_run_id = run_id[:8]
    print(f"Latest Run ID: {run_id}")
    
    print("--- Downloading Model and Artifacts ---")
    # For v7, train.py logs the artifact directly as 'tft_v7.pth'
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
    scaler_path = "../../datalake/v7_tensors/scaler.pkl"
    target_scaler_path = "../../datalake/v7_tensors/target_scaler.pkl"
    feature_groups_path = "../../datalake/v7_tensors/feature_groups.pkl"
    model_path = os.path.join(local_dir, "tft_v7.pth")
    
    target_scaler = joblib.load(target_scaler_path)
    feature_groups = joblib.load(feature_groups_path)
    obs_cols = feature_groups['obs_cols']
    known_cols = feature_groups['known_cols']
    static_cols = feature_groups['static_cols']
    
    seq_len = 48
    horizon = 48
    d_model = 64
    num_heads = 4
    
    model = TemporalFusionTransformer(
        num_static_vars=len(static_cols),
        num_future_vars=len(known_cols),
        num_past_vars=len(obs_cols),
        d_model=d_model,
        num_heads=num_heads,
        seq_len=seq_len,
        horizon=horizon,
        dropout=0.1
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    print("--- Loading Test Data ---")
    test_data = pd.read_parquet('../../datalake/v7_tensors/test.parquet')
    
    batch_size = 512
    test_dataset = TFTDataset(test_data, obs_cols, known_cols, static_cols, seq_length=seq_len, horizon=horizon)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    test_predictions_nd = []
    actual_targets_nd = []
    all_attn_weights = []
    
    print("--- Generating Predictions ---")
    with torch.no_grad():
        for static, past, known, y_diff, y_vol, y_trend, nd_current in test_loader:
            static, past, known = static.to(device), past.to(device), known.to(device)
            
            pred_diff, pred_vol, pred_trend, attn_weights = model(static, past, known)
            
            # Reconstruct absolute demand
            # pred_diff and y_diff are (batch, horizon)
            # nd_current is (batch,)
            
            # Cumulative sum across horizon:
            pred_diff_np = pred_diff.cpu().numpy()
            y_diff_np = y_diff.numpy()
            nd_curr_np = nd_current.numpy()[:, None] # (batch, 1)
            
            pred_nd_abs = nd_curr_np + np.cumsum(pred_diff_np, axis=1)
            actual_nd_abs = nd_curr_np + np.cumsum(y_diff_np, axis=1)
            
            test_predictions_nd.append(pred_nd_abs)
            actual_targets_nd.append(actual_nd_abs)
            all_attn_weights.append(attn_weights.cpu().numpy())
            
    # Each batch returns (batch_size, horizon)
    pred_nd_arr = np.concatenate(test_predictions_nd, axis=0)
    actual_nd_arr = np.concatenate(actual_targets_nd, axis=0)
    
    # We want to evaluate the overall multi-step performance, so we can flatten and compare
    test_pred_flat = pred_nd_arr.flatten()
    actual_targets_flat = actual_nd_arr.flatten()
    
    # Let's also extract the 1-step ahead prediction (horizon = 0) to plot a continuous line
    pred_1step = pred_nd_arr[:, 0]
    actual_1step = actual_nd_arr[:, 0]
    
    print("--- Evaluating Metrics (All Horizons) ---")
    mae = mean_absolute_error(actual_targets_flat, test_pred_flat)
    rmse = np.sqrt(mean_squared_error(actual_targets_flat, test_pred_flat))
    r2 = r2_score(actual_targets_flat, test_pred_flat)
    
    print(f"MAE (All Horizons): {mae:.4f}")
    print(f"RMSE (All Horizons): {rmse:.4f}")
    print(f"R2 (All Horizons): {r2:.4f}")
    
    print("\n--- Evaluating Metrics (1-Step Ahead) ---")
    mae_1step = mean_absolute_error(actual_1step, pred_1step)
    rmse_1step = np.sqrt(mean_squared_error(actual_1step, pred_1step))
    r2_1step = r2_score(actual_1step, pred_1step)
    
    print(f"MAE (1-Step): {mae_1step:.4f}")
    print(f"RMSE (1-Step): {rmse_1step:.4f}")
    print(f"R2 (1-Step): {r2_1step:.4f}")
    
    print("\n--- Generating Plots ---")
    
    # 1. 1-Step Ahead Prediction Plot
    plot_filename = f"eval_pred_1step_v7_{short_run_id}.png"
    plt.figure(figsize=(12, 6))
    plt.plot(actual_1step[:2000], label='Actual Demand (1-step)', color='blue', alpha=0.5)
    plt.plot(pred_1step[:2000], label='Predicted Demand (1-step)', color='red', alpha=0.5)
    plt.title(f'Actual vs Predicted Demand 1-Step (v7 TFT) - Run: {short_run_id}')
    plt.xlabel('Time Step')
    plt.ylabel('Demand (Scaled)')
    plt.legend()
    plt.savefig(plot_filename, dpi=600)
    plt.close()
    
    # 2. Multi-Step Horizon Plot for a specific sample
    sample_idx = 1000 # arbitrary sample
    plot_multi_filename = f"eval_pred_multistep_v7_{short_run_id}.png"
    plt.figure(figsize=(12, 6))
    plt.plot(actual_nd_arr[sample_idx, :], label='Actual Sequence', color='blue', marker='o')
    plt.plot(pred_nd_arr[sample_idx, :], label='Predicted Sequence', color='red', marker='x')
    plt.title(f'Multi-step Forecast (48 steps ahead) for Sample {sample_idx}')
    plt.xlabel('Horizon Step')
    plt.ylabel('Demand (Scaled)')
    plt.legend()
    plt.savefig(plot_multi_filename, dpi=600)
    plt.close()
        
    # 3. Horizon-Wide Degradation
    mae_per_step = []
    for h in range(horizon):
        mae_h = mean_absolute_error(actual_nd_arr[:, h], pred_nd_arr[:, h])
        mae_per_step.append(mae_h)
        
    plot_degrad_filename = f"eval_degradation_v7_{short_run_id}.png"
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, horizon + 1), mae_per_step, marker='o', color='purple')
    plt.title(f'Horizon-Wide MAE Degradation (v7 TFT) - Run: {short_run_id}')
    plt.xlabel('Horizon Step')
    plt.ylabel('Mean Absolute Error (Scaled)')
    plt.grid(True)
    plt.savefig(plot_degrad_filename, dpi=600)
    plt.close()

    # 4. Attention Weight Heatmap
    # all_attn_weights is a list of arrays of shape (batch, q_len, k_len)
    attn_arr = np.concatenate(all_attn_weights, axis=0)
    # Average across all test samples
    avg_attn = np.mean(attn_arr, axis=0) # (q_len, k_len)
    
    plot_attn_filename = f"eval_attention_heatmap_v7_{short_run_id}.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(avg_attn, cmap="viridis")
    plt.title(f'Average Temporal Attention Weights (v7 TFT)')
    plt.xlabel('Key (Attended Past/Future Sequence)')
    plt.ylabel('Query (Target Sequence)')
    plt.savefig(plot_attn_filename, dpi=600)
    plt.close()
    
    # 5. Residual Autocorrelation (ACF) on 1-Step Ahead
    residuals_1step = actual_1step - pred_1step
    plot_acf_filename = f"eval_residual_acf_v7_{short_run_id}.png"
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(residuals_1step, lags=100, ax=ax, title="Autocorrelation of 1-Step Ahead Residuals")
    plt.savefig(plot_acf_filename, dpi=600)
    plt.close()
        
    print(f"Saved plots to current directory.")

if __name__ == "__main__":
    evaluate()
