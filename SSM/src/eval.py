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
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from models.ssm import LatentSSM
from feature_prep import SSMDataset, transform_data

def evaluate():
    print("--- Connecting to DagsHub MLflow ---")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rgorasia19"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "26b84ad06cab16667533b56a5cebe4ebf27d8f9a"
    
    dagshub.init(repo_owner="rgorasia19", repo_name="Energy-Intelligence-System", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow")
    
    experiment = mlflow.get_experiment_by_name("ssm_long_horizon")
    if experiment is None:
        raise ValueError("Experiment 'ssm_long_horizon' not found.")
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    if runs.empty:
        raise ValueError("No runs found in experiment 'ssm_long_horizon'.")
        
    run_id = runs.iloc[0].run_id
    short_run_id = run_id[:8]
    print(f"Latest Run ID: {run_id}")
    
    print("--- Downloading Model and Artifacts ---")
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
    model_path = os.path.join(local_dir, "ssm_v1.pth")
    
    data_dir = '../../datalake/ssm_tensors/'
    col_info = joblib.load(os.path.join(data_dir, 'columns.pkl'))
    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    feature_cols = col_info['feature_columns']
    target_cols = col_info['target_columns']
    
    demand_cols = ['ND', 'TSD', 'ENGLAND_WALES_DEMAND']
    gen_cols = ['GAS', 'COAL', 'NUCLEAR', 'WIND', 'GENERATION']
    
    demand_idx = [target_cols.index(c) for c in demand_cols]
    gen_idx = [target_cols.index(c) for c in gen_cols]
    
    seq_len = 60
    horizon = 30
    latent_dim = 8
    hidden_dim = 64
    
    model = LatentSSM(
        input_dim=len(feature_cols),
        demand_dim=len(demand_cols),
        gen_dim=len(gen_cols),
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    print("--- Loading Test Data ---")
    test_data = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    
    batch_size = 256
    test_dataset = SSMDataset(test_data, seq_len=seq_len, horizon=horizon, 
                              feature_columns=feature_cols, target_columns=target_cols)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print("--- Running Deterministic Inference ---")
    model.eval()
    
    all_pred_demand = []
    all_true_demand = []
    all_mask_demand = []
    
    all_pred_gen = []
    all_true_gen = []
    all_mask_gen = []
    
    all_z_seq = []
    
    with torch.no_grad():
        for batch in test_loader:
            enc_inputs = batch['encoder_inputs'].to(device)
            dec_inputs = batch['decoder_inputs'].to(device)
            dec_targets = batch['decoder_targets'].numpy()
            dec_masks = batch['decoder_mask'].numpy()
            
            outputs = model(enc_inputs, dec_inputs, horizon)
            
            pred_demand = outputs['pred_demand'].cpu().numpy()
            pred_gen = outputs['pred_gen'].cpu().numpy()
            z_seq = outputs['z_seq'].cpu().numpy()
            
            all_pred_demand.append(pred_demand)
            all_true_demand.append(dec_targets[:, :, demand_idx])
            all_mask_demand.append(dec_masks[:, :, demand_idx])
            
            all_pred_gen.append(pred_gen)
            all_true_gen.append(dec_targets[:, :, gen_idx])
            all_mask_gen.append(dec_masks[:, :, gen_idx])
            
            all_z_seq.append(z_seq)
            
    # Concatenate all batches
    pred_demand = np.concatenate(all_pred_demand, axis=0)
    true_demand = np.concatenate(all_true_demand, axis=0)
    mask_demand = np.concatenate(all_mask_demand, axis=0)
    
    pred_gen = np.concatenate(all_pred_gen, axis=0)
    true_gen = np.concatenate(all_true_gen, axis=0)
    mask_gen = np.concatenate(all_mask_gen, axis=0)
    
    z_seq = np.concatenate(all_z_seq, axis=0)
    
    print("\n--- Evaluation Metrics ---")
    
    def calc_metrics(pred, true, mask, name):
        # Flatten and filter by mask
        p_flat = pred[mask == 1]
        t_flat = true[mask == 1]
        
        if len(p_flat) == 0:
            print(f"{name} - No valid masked targets found.")
            return
            
        mae = mean_absolute_error(t_flat, p_flat)
        rmse = np.sqrt(mean_squared_error(t_flat, p_flat))
        
        print(f"{name} MAE:  {mae:.4f}")
        print(f"{name} RMSE: {rmse:.4f}")
        
    calc_metrics(pred_demand, true_demand, mask_demand, "Demand")
    calc_metrics(pred_gen, true_gen, mask_gen, "Generation")
    
    # Smoothness evaluation
    z_diff = z_seq[:, 1:, :] - z_seq[:, :-1, :]
    smoothness = np.mean(z_diff ** 2)
    print(f"Latent State Smoothness (Var(z_t - z_t-1)): {smoothness:.4f}")
    
    print("\n--- Generating Plots ---")
    
    # 1. PCA Plot of Latent State
    print("Generating PCA Plot...")
    # Flatten across batch and horizon to get a collection of states
    z_flat = z_seq.reshape(-1, latent_dim)
    
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_flat)
    
    # Color by progress through the test set (time index proxy)
    time_color = np.linspace(0, 1, len(z_flat))
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=time_color, cmap='viridis', alpha=0.1, s=2)
    plt.colorbar(scatter, label='Normalized Time Index')
    plt.title(f"Latent Regime PCA - Run: {short_run_id}\nSmoothness: {smoothness:.4f}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(f"eval_latent_pca_{short_run_id}.png", bbox_inches='tight')
    plt.close()
    
    # 2. Fan Chart (Uncertainty via Monte Carlo)
    print("Generating Monte Carlo Fan Chart...")
    # Take the very first sequence in the test set
    sample_enc_inputs = test_dataset[0]['encoder_inputs'].unsqueeze(0).to(device)
    sample_dec_inputs = test_dataset[0]['decoder_inputs'].unsqueeze(0).to(device)
    sample_true_demand = test_dataset[0]['decoder_targets'][:, demand_idx[0]].numpy() # First demand target (e.g., ND)
    
    model.train() # Enable process noise sampling
    mc_passes = 100
    mc_preds = []
    
    with torch.no_grad():
        for _ in range(mc_passes):
            out = model(sample_enc_inputs, sample_dec_inputs, horizon)
            # Take the first demand output
            mc_preds.append(out['pred_demand'][0, :, 0].cpu().numpy())
            
    mc_preds = np.array(mc_preds) # [100, horizon]
    
    p10 = np.percentile(mc_preds, 10, axis=0)
    p50 = np.percentile(mc_preds, 50, axis=0)
    p90 = np.percentile(mc_preds, 90, axis=0)
    
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(horizon)
    plt.plot(time_axis, sample_true_demand, color='black', label='Actual', marker='o', linewidth=2)
    plt.plot(time_axis, p50, color='blue', label='Median Forecast')
    plt.fill_between(time_axis, p10, p90, color='blue', alpha=0.2, label='10th-90th Percentile')
    plt.title(f"30-Day Forecast Fan Chart (SSM Uncertainty) - Run: {short_run_id}")
    plt.xlabel("Days Ahead")
    plt.ylabel("Scaled Demand")
    plt.legend()
    plt.savefig(f"eval_fan_chart_{short_run_id}.png", bbox_inches='tight')
    plt.close()

    print("Evaluation complete. Artifacts saved locally.")

if __name__ == "__main__":
    evaluate()
