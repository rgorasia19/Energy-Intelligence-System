import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import dagshub
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
import matplotlib.patches as mpatches

# Import Dataset and Model exactly as defined in eval.py
from eval import MoEDataset
from models.predictor import UnifiedRegimeModel

def run_ablation_test():
    print("--- Connecting to DagsHub MLflow ---")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rgorasia19"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "26b84ad06cab16667533b56a5cebe4ebf27d8f9a"
    
    dagshub.init(repo_owner="rgorasia19", repo_name="Energy-Intelligence-System", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow")
    
    experiment = mlflow.get_experiment_by_name("v5_attention_regime_2state")
    if experiment is None:
        raise ValueError("Experiment 'v5_attention_regime_2state' not found.")
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    if runs.empty:
        raise ValueError("No runs found in experiment 'v5_attention_regime_2state'.")
        
    run_id = runs.iloc[0].run_id
    print(f"Loaded Run ID: {run_id}")
    
    print("--- Downloading Model and Artifacts ---")
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
    scaler_path = os.path.join(local_dir, "scaler.pkl")
    feature_groups_path = os.path.join(local_dir, "feature_groups.pkl")
    model_path = os.path.join(local_dir, "regime_attention.pth")
    
    feature_groups = joblib.load(feature_groups_path)
    raw_cols = feature_groups['raw_cols']
    gate_cols = feature_groups['gate_cols']
    
    seq_len = 48
    num_regimes = 2
    embed_dim = 32
    
    # Initialize the model
    model = UnifiedRegimeModel(
        raw_feature_dim=len(raw_cols), 
        gate_feature_dim=len(gate_cols), 
        seq_len=seq_len,
        num_regimes=num_regimes,
        d_model=64,
        embed_dim=embed_dim
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print("--- Loading Test Data ---")
    test_data = pd.read_parquet('../../datalake/moe_tensors/test.parquet')
    
    batch_size = 2048
    test_dataset = MoEDataset(test_data, raw_cols, gate_cols, seq_length=seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    baseline_preds = []
    ablated_preds = []
    actual_targets = []
    regime_probs = []
    
    print("--- Running Inference and Ablation ---")
    with torch.no_grad():
        for x_raw, x_gate, y_label, y_vol, y_trend in test_loader:
            x_raw, x_gate = x_raw.to(device), x_gate.to(device)
            
            # 1. Standard Forward Pass
            # Get regime probabilities and baseline prediction
            p_regime, logits, aux_preds, e_regime, attention_maps = model.regime_network(x_gate, tau=0.8)
            y_hat_base = model.predictor(x_raw, e_regime)
            
            # 2. Ablated Forward Pass (Zero out Regime 1)
            p_regime_ablated = p_regime.clone()
            p_regime_ablated[:, :, 1] = 0.0 # Force Regime 1 probability to 0
            
            # Renormalize across remaining regimes (just 0 now)
            # Add a small epsilon to prevent division by zero
            p_regime_ablated = p_regime_ablated / (p_regime_ablated.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Pass the ablated probabilities through the embedding projection
            e_regime_ablated = model.regime_network.head.embed_proj(p_regime_ablated)
            
            # Get ablated predictions
            y_hat_ablated = model.predictor(x_raw, e_regime_ablated)
            
            baseline_preds.append(y_hat_base.cpu().numpy())
            ablated_preds.append(y_hat_ablated.cpu().numpy())
            actual_targets.append(y_label.numpy())
            regime_probs.append(p_regime[:, -1, :].cpu().numpy()) # Take last step for conditional analysis

    base_pred_flat = np.concatenate(baseline_preds).flatten()
    ablated_pred_flat = np.concatenate(ablated_preds).flatten()
    actual_targets_flat = np.concatenate(actual_targets).flatten()
    regime_probs_flat = np.concatenate(regime_probs)
    
    # Hard assignment based on the baseline model
    hard_regimes = np.argmax(regime_probs_flat, axis=1)
    
    print("\n=============================================")
    print("  PHASE 1: CONDITIONAL PERFORMANCE BY REGIME ")
    print("=============================================")
    
    for i in range(num_regimes):
        mask = (hard_regimes == i)
        count = mask.sum()
        if count == 0:
            print(f"Regime {i}: No samples found.")
            continue
            
        y_true_regime = actual_targets_flat[mask]
        y_pred_regime = base_pred_flat[mask]
        
        mae = mean_absolute_error(y_true_regime, y_pred_regime)
        rmse = np.sqrt(mean_squared_error(y_true_regime, y_pred_regime))
        r2 = r2_score(y_true_regime, y_pred_regime)
        
        print(f"Regime {i} (N={count} | {count/len(actual_targets_flat)*100:.1f}%):")
        print(f"  -> MAE:  {mae:.2f}")
        print(f"  -> RMSE: {rmse:.2f}")
        print(f"  -> R2:   {r2:.4f}")

    print("\n=============================================")
    print("  PHASE 2: ABLATION OF REGIME 1              ")
    print("=============================================")
    
    # Global degradation
    base_mae_global = mean_absolute_error(actual_targets_flat, base_pred_flat)
    ablated_mae_global = mean_absolute_error(actual_targets_flat, ablated_pred_flat)
    
    print("Global Performance:")
    print(f"  -> Baseline MAE: {base_mae_global:.2f}")
    print(f"  -> Ablated MAE:  {ablated_mae_global:.2f}")
    print(f"  -> Degradation:  {ablated_mae_global - base_mae_global:+.2f} MW\n")
    
    # Degradation specifically on timesteps where Regime 1 WAS dominant
    r2_mask = (hard_regimes == 1)
    if r2_mask.sum() > 0:
        y_true_r2 = actual_targets_flat[r2_mask]
        y_pred_base_r2 = base_pred_flat[r2_mask]
        y_pred_ablated_r2 = ablated_pred_flat[r2_mask]
        
        base_mae_r2 = mean_absolute_error(y_true_r2, y_pred_base_r2)
        ablated_mae_r2 = mean_absolute_error(y_true_r2, y_pred_ablated_r2)
        
        print("Performance specifically on Regime 1 Timesteps:")
        print(f"  -> Baseline MAE: {base_mae_r2:.2f}")
        print(f"  -> Ablated MAE:  {ablated_mae_r2:.2f}")
        print(f"  -> Degradation:  {ablated_mae_r2 - base_mae_r2:+.2f} MW")
        print("  (If degradation is highly positive, Regime 1 is structurally vital.)")
    else:
        print("No Regime 1 dominant timesteps found to test specific degradation.")
        
    print("\n--- Generating Ablation Plot ---")
    
    if r2_mask.sum() > 0:
        # Plot the first 200 samples that belong to Regime 1
        idx_r2 = np.where(r2_mask)[0][:200]
        
        plt.figure(figsize=(12, 6))
        plt.plot(actual_targets_flat[idx_r2], label='Actual Demand', color='black', alpha=0.7)
        plt.plot(base_pred_flat[idx_r2], label='Baseline Prediction (K=2)', color='blue', alpha=0.7)
        plt.plot(ablated_pred_flat[idx_r2], label='Ablated Prediction (R1=0)', color='red', alpha=0.7)
        
        plt.title('Impact of Regime 1 Ablation on Target Timesteps')
        plt.xlabel('Sample Index (Filtered for Regime 1)')
        plt.ylabel('Demand')
        plt.legend()
        plt.savefig('ablation_regime1.png', dpi=600)
        print("Saved ablation plot to ablation_regime1.png")

if __name__ == "__main__":
    run_ablation_test()
