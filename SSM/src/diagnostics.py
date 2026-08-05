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
from statsmodels.graphics.tsaplots import plot_acf
from torch.utils.data import DataLoader

from models.ssm import LatentSSM
from feature_prep import SSMDataset

def run_diagnostics():
    print("--- Connecting to DagsHub MLflow ---")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "rgorasia19"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "26b84ad06cab16667533b56a5cebe4ebf27d8f9a"
    
    dagshub.init(repo_owner="rgorasia19", repo_name="Energy-Intelligence-System", mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow")
    
    experiment = mlflow.get_experiment_by_name("hssm_probabilistic")
    if experiment is None:
        raise ValueError("Experiment 'hssm_probabilistic' not found.")
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    if runs.empty:
        raise ValueError("No runs found in experiment 'hssm_probabilistic'.")
        
    run_id = runs.iloc[0].run_id
    short_run_id = run_id[:8]
    print(f"Latest Run ID: {run_id}")
    
    print("--- Downloading Model and Artifacts ---")
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
    model_path = os.path.join(local_dir, "ssm_v1.pth")
    
    data_dir = '../../datalake/ssm_tensors/'
    col_info = joblib.load(os.path.join(data_dir, 'columns.pkl'))
    feature_cols = col_info['feature_columns']
    target_cols = col_info['target_columns']
    
    demand_cols = ['ND', 'TSD', 'ENGLAND_WALES_DEMAND']
    gen_cols = ['GAS', 'COAL', 'NUCLEAR', 'WIND', 'SOLAR', 'IMPORTS', 'GENERATION']
    
    demand_idx = [target_cols.index(c) for c in demand_cols]
    gen_idx = [target_cols.index(c) for c in gen_cols]
    
    seq_len = 60
    horizon = 30
    latent_dim_demand = 16
    latent_dim_gen = 24
    latent_dim = latent_dim_demand + latent_dim_gen
    hidden_dim = 64
    num_regimes_demand = 4
    num_regimes_gen = 6
    
    weather_cols = ['temperature_2m', 'cloudcover', 'windspeed_10m', 'shortwave_radiation']
    fourier_cols = [c for c in feature_cols if '_sin_k' in c or '_cos_k' in c]
    calendar_cols = []
    if 'is_bank_holiday' in feature_cols:
        calendar_cols.append('is_bank_holiday')
    if 'is_weekend' in feature_cols:
        calendar_cols.append('is_weekend')
    physical_cols = []
    if 'theoretical_wind_generation' in feature_cols:
        physical_cols.append('theoretical_wind_generation')
    if 'theoretical_solar_generation' in feature_cols:
        physical_cols.append('theoretical_solar_generation')
    embedded_cols = ['EMBEDDED_WIND_CAPACITY', 'EMBEDDED_SOLAR_CAPACITY']
    macro_cols = ['uk_cpi', 'uk_gdp_index', 'bank_rate']
    price_cols = ['day_ahead_price']
    remit_cols = ['nuclear_available_capacity', 'gas_available_capacity', 'coal_available_capacity']
    
    advanced_gen_cols = [
        'nuclear_availability_gap', 'nuclear_forward_drop_3d', 'nuclear_inertia_proxy',
        'freq_excursion_flag', 'freq_p99_dev', 'curtailment_composite_index',
        'imbalance_volatility', 'system_stress_alert'
    ]
    
    known_demand_cols = fourier_cols + weather_cols + physical_cols + calendar_cols
    known_gen_cols = fourier_cols + weather_cols + physical_cols + calendar_cols + [c for c in embedded_cols + macro_cols + price_cols + remit_cols + advanced_gen_cols if c in feature_cols]
    
    known_dim_d = len(known_demand_cols)
    known_dim_g = len(known_gen_cols)
    
    demand_indices = [feature_cols.index(c) for c in demand_cols]
    model = LatentSSM(
        input_dim=len(feature_cols),
        demand_dim=len(demand_cols),
        gen_dim=len(gen_cols),
        known_dim_d=known_dim_d,
        known_dim_g=known_dim_g,
        latent_dim_demand=latent_dim_demand,
        latent_dim_gen=latent_dim_gen,
        hidden_dim=hidden_dim,
        num_regimes_demand=num_regimes_demand,
        num_regimes_gen=num_regimes_gen,
        fourier_dim=len(fourier_cols),
        bidirectional_d=False,
        bidirectional_g=True,
        demand_indices=demand_indices
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    print("--- Loading Test Data ---")
    test_data = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    
    batch_size = 256
    test_dataset = SSMDataset(test_data, seq_len=seq_len, horizon=horizon, 
                              feature_columns=feature_cols, target_columns=target_cols,
                              known_demand_columns=known_demand_cols, known_gen_columns=known_gen_cols)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    all_d_mean = []
    all_d_var = []
    all_g_mean = []
    all_g_var = []
    all_d_true = []
    all_g_true = []
    all_mask = []
    all_z_seq = []
    all_r_seq = []
    all_features = []

    print("--- Running Inference ---")
    for batch in test_loader:
        enc_inputs = batch['encoder_inputs'].to(device)
        decoder_inputs_d = batch['decoder_inputs_d'].to(device)
        decoder_inputs_g = batch['decoder_inputs_g'].to(device)
        dec_targets = batch['decoder_targets'].numpy()
        dec_masks = batch['decoder_mask'].numpy()
        
        N = 50
        batch_d_samples = []
        batch_g_samples = []
        with torch.no_grad():
            for _ in range(N):
                outputs = model(enc_inputs, decoder_inputs_d, decoder_inputs_g, horizon, target_seq=None, sample=True, tau=1.0)
                d_scale = outputs['demand_scale'].cpu().numpy()
                d_nu = outputs['demand_nu'].cpu().numpy()
                d_sample = outputs['demand_mean'].cpu().numpy() + np.random.standard_t(df=np.maximum(d_nu, 2.001)) * d_scale
                
                g_scale = outputs['gen_scale'].cpu().numpy()
                g_nu = outputs['gen_nu'].cpu().numpy()
                g_sample = outputs['gen_mean'].cpu().numpy() + np.random.standard_t(df=np.maximum(g_nu, 2.001)) * g_scale
                batch_d_samples.append(d_sample)
                batch_g_samples.append(g_sample)
            
            batch_d_samples = np.stack(batch_d_samples, axis=1)
            batch_g_samples = np.stack(batch_g_samples, axis=1)
            
            all_d_mean.append(np.mean(batch_d_samples, axis=1))
            all_d_var.append(np.var(batch_d_samples, axis=1))
            all_g_mean.append(np.mean(batch_g_samples, axis=1))
            all_g_var.append(np.var(batch_g_samples, axis=1))
            
            all_d_true.append(dec_targets[:, :, demand_idx])
            all_g_true.append(dec_targets[:, :, gen_idx])
            all_mask.append(dec_masks[:, :, demand_idx])
            
            sampled_z = torch.cat([outputs['sampled_z_d_seq'], outputs['sampled_z_g_seq']], dim=-1)
            all_z_seq.append(sampled_z.cpu().numpy())
            all_r_seq.append(outputs['sampled_r_d_seq'].cpu().numpy())
            all_features.append(dec_inputs[:, :, -known_dim:].cpu().numpy())
            
    d_mean = np.concatenate(all_d_mean, axis=0)
    d_var = np.concatenate(all_d_var, axis=0) + 1e-6
    g_mean = np.concatenate(all_g_mean, axis=0)
    g_var = np.concatenate(all_g_var, axis=0) + 1e-6
    d_true = np.concatenate(all_d_true, axis=0)
    g_true = np.concatenate(all_g_true, axis=0)
    mask = np.concatenate(all_mask, axis=0)
    z_seq = np.concatenate(all_z_seq, axis=0)
    r_seq = np.concatenate(all_r_seq, axis=0)
    features = np.concatenate(all_features, axis=0)

    valid_mask = mask == 1
    d0_mean_flat = d_mean[:, :, 0][valid_mask[:, :, 0]]
    d0_var_flat = d_var[:, :, 0][valid_mask[:, :, 0]]
    d0_true_flat = d_true[:, :, 0][valid_mask[:, :, 0]]
    d0_error_flat = np.abs(d0_mean_flat - d0_true_flat)

    g0_mean_flat = g_mean[:, :, 0][valid_mask[:, :, 0]]
    g0_var_flat = g_var[:, :, 0][valid_mask[:, :, 0]]
    g0_true_flat = g_true[:, :, 0][valid_mask[:, :, 0]]
    g0_error_flat = np.abs(g0_mean_flat - g0_true_flat)
    
    os.makedirs('diagnostics_output', exist_ok=True)
    print("\n--- Generating Diagnostics Plots ---")

    # 1. Predicted Sigma vs Absolute Error
    print("1. Plotting Sigma vs Absolute Error...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.scatterplot(x=np.sqrt(d0_var_flat), y=d0_error_flat, alpha=0.1, ax=axes[0])
    axes[0].set_xlabel('Predicted Sigma')
    axes[0].set_ylabel('Absolute Error')
    axes[0].set_title(f'Demand ({demand_cols[0]}): Sigma vs Abs Error')
    axes[0].plot([0, max(np.sqrt(d0_var_flat))], [0, max(np.sqrt(d0_var_flat))], 'r--')
    
    sns.scatterplot(x=np.sqrt(g0_var_flat), y=g0_error_flat, alpha=0.1, ax=axes[1])
    axes[1].set_xlabel('Predicted Sigma')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title(f'Generation ({gen_cols[0]}): Sigma vs Abs Error')
    axes[1].plot([0, max(np.sqrt(g0_var_flat))], [0, max(np.sqrt(g0_var_flat))], 'r--')
    
    plt.tight_layout()
    plt.savefig('diagnostics_output/1_sigma_vs_error.png')
    plt.close()

    # 2. Latent Trajectory Inspection
    print("2. Plotting Latent Trajectories...")
    sample_idx = 0
    sample_z = z_seq[sample_idx]
    plt.figure(figsize=(12, 6))
    for i in range(latent_dim):
        plt.plot(sample_z[:, i], label=f'z_{i}')
    plt.title(f'Latent State Trajectory for Sample {sample_idx}')
    plt.xlabel('Horizon Step')
    plt.ylabel('Latent Value')
    plt.legend()
    plt.savefig('diagnostics_output/2_latent_trajectory.png')
    plt.close()

    # 3. Autocorrelation of Residuals
    print("3. Plotting Autocorrelation of Residuals...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    d_resid_seq = d_mean[0, :, 0] - d_true[0, :, 0]
    g_resid_seq = g_mean[0, :, 0] - g_true[0, :, 0]
    
    plot_acf(d_resid_seq, ax=axes[0], title=f'Demand ({demand_cols[0]}): Residual ACF (Sample 0)', lags=horizon-1)
    plot_acf(g_resid_seq, ax=axes[1], title=f'Generation ({gen_cols[0]}): Residual ACF (Sample 0)', lags=horizon-1)
    
    plt.tight_layout()
    plt.savefig('diagnostics_output/3_residual_acf.png')
    plt.close()

    # 4. Regime Transition Matrix
    print("4. Plotting Regime Transition Matrix...")
    r_labels = np.argmax(r_seq, axis=-1)
    transitions = np.zeros((num_regimes_demand, num_regimes_demand))
    for b in range(r_labels.shape[0]):
        for t in range(horizon - 1):
            curr_r = r_labels[b, t]
            next_r = r_labels[b, t+1]
            transitions[curr_r, next_r] += 1
            
    row_sums = transitions.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums!=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_probs, annot=True, cmap='Blues', fmt=".2f")
    plt.title('Regime Transition Matrix')
    plt.xlabel('Next Regime')
    plt.ylabel('Current Regime')
    plt.savefig('diagnostics_output/4_regime_transitions.png')
    plt.close()

    # 5. Error vs Exogenous Features
    print("5. Plotting Error vs Exogenous Features...")
    temp_idx = known_columns.index('temperature_2m') if 'temperature_2m' in known_columns else None
    
    if temp_idx is not None:
        temp_flat = features[:, :, temp_idx][valid_mask[:, :, 0]]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.scatterplot(x=temp_flat, y=d0_error_flat, alpha=0.1, ax=axes[0])
        axes[0].set_xlabel('Scaled Temperature')
        axes[0].set_ylabel('Absolute Error')
        axes[0].set_title(f'Demand ({demand_cols[0]}): Error vs Temp')
        
        sns.scatterplot(x=temp_flat, y=g0_error_flat, alpha=0.1, ax=axes[1])
        axes[1].set_xlabel('Scaled Temperature')
        axes[1].set_ylabel('Absolute Error')
        axes[1].set_title(f'Generation ({gen_cols[0]}): Error vs Temp')
        
        plt.tight_layout()
        plt.savefig('diagnostics_output/5_error_vs_feature_temp.png')
        plt.close()

    print("\nAll diagnostics completed and saved to 'diagnostics_output/'.")

if __name__ == "__main__":
    run_diagnostics()
