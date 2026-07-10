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
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from models.ssm import LatentSSM
from feature_prep import SSMDataset, transform_data

def crps_gaussian(mu, sig_sq, y):
    sig = np.sqrt(sig_sq)
    loc = (y - mu) / (sig + 1e-8)
    pdf = stats.norm.pdf(loc)
    cdf = stats.norm.cdf(loc)
    crps = sig * (loc * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    return crps

def evaluate():
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
    num_regimes = 4
    
    weather_cols = ['temperature_2m', 'cloudcover', 'windspeed_10m', 'shortwave_radiation']
    calendar_cols = [c for c in feature_cols if c.endswith('_sin') or c.endswith('_cos') or c == 'is_bank_holiday']
    embedded_cols = ['EMBEDDED_WIND_CAPACITY', 'EMBEDDED_SOLAR_CAPACITY']
    macro_cols = ['uk_cpi', 'uk_gdp_index', 'bank_rate']
    known_columns = weather_cols + calendar_cols + [c for c in embedded_cols + macro_cols if c in feature_cols]
    known_dim = len(known_columns)
    
    model = LatentSSM(
        input_dim=len(feature_cols),
        demand_dim=len(demand_cols),
        gen_dim=len(gen_cols),
        known_dim=known_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_regimes=num_regimes
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    print("--- Loading Test Data ---")
    test_data = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    
    batch_size = 256
    test_dataset = SSMDataset(test_data, seq_len=seq_len, horizon=horizon, 
                              feature_columns=feature_cols, target_columns=target_cols,
                              known_columns=known_columns)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print("--- Running Deterministic Inference ---")
    model.eval()
    
    all_demand_mean = []
    all_demand_var = []
    all_true_demand = []
    all_mask_demand = []
    all_seas_demand = []
    
    all_gen_mean = []
    all_gen_var = []
    all_true_gen = []
    all_mask_gen = []
    all_seas_gen = []
    
    all_z_seq = []
    all_r_seq = []
    all_demand_pit = []
    
    with torch.no_grad():
        for batch in test_loader:
            enc_inputs = batch['encoder_inputs'].to(device)
            dec_inputs = batch['decoder_inputs'].to(device)
            dec_targets = batch['decoder_targets'].numpy()
            dec_masks = batch['decoder_mask'].numpy()
            
            # If decoder_seasonality exists, load it, otherwise zeros
            if 'decoder_seasonality' in batch:
                dec_seas = batch['decoder_seasonality'].numpy()
            else:
                dec_seas = np.zeros_like(dec_targets)
            
            N = 50
            batch_demand_samples = []
            batch_gen_samples = []
            
            for _ in range(N):
                outputs = model(enc_inputs, dec_inputs, horizon, target_seq=None, sample=True, tau=1.0)
                
                d_mean = outputs['demand_mean'].cpu().numpy()
                d_var = outputs['demand_var'].cpu().numpy()
                g_mean = outputs['gen_mean'].cpu().numpy()
                g_var = outputs['gen_var'].cpu().numpy()
                
                d_sample = np.random.normal(loc=d_mean, scale=np.sqrt(d_var))
                g_sample = np.random.normal(loc=g_mean, scale=np.sqrt(g_var))
                
                batch_demand_samples.append(d_sample)
                batch_gen_samples.append(g_sample)
                
            batch_demand_samples = np.stack(batch_demand_samples, axis=1) # [batch, N, horizon, dim]
            batch_gen_samples = np.stack(batch_gen_samples, axis=1)
            
            # Empirical mean and variance
            demand_mean = np.mean(batch_demand_samples, axis=1)
            demand_var = np.var(batch_demand_samples, axis=1) + 1e-6
            gen_mean = np.mean(batch_gen_samples, axis=1)
            gen_var = np.var(batch_gen_samples, axis=1) + 1e-6
            
            # Compute PIT for demand
            # True target needs to be broadcasted to [batch, 1, horizon, dim]
            t_demand = dec_targets[:, np.newaxis, :, demand_idx]
            pit = np.mean(batch_demand_samples < t_demand, axis=1)
            all_demand_pit.append(pit)
            
            all_demand_mean.append(demand_mean)
            all_demand_var.append(demand_var)
            all_true_demand.append(dec_targets[:, :, demand_idx])
            all_mask_demand.append(dec_masks[:, :, demand_idx])
            all_seas_demand.append(dec_seas[:, :, demand_idx])
            
            all_gen_mean.append(gen_mean)
            all_gen_var.append(gen_var)
            all_true_gen.append(dec_targets[:, :, gen_idx])
            all_mask_gen.append(dec_masks[:, :, gen_idx])
            all_seas_gen.append(dec_seas[:, :, gen_idx])
            
            # For simplicity, just append the last r_seq
            all_z_seq.append(outputs['sampled_z_seq'].cpu().numpy())
            all_r_seq.append(outputs['sampled_r_seq'].cpu().numpy())
            
    # Concatenate all batches
    demand_mean = np.concatenate(all_demand_mean, axis=0)
    demand_var = np.concatenate(all_demand_var, axis=0)
    true_demand = np.concatenate(all_true_demand, axis=0)
    mask_demand = np.concatenate(all_mask_demand, axis=0)
    seas_demand = np.concatenate(all_seas_demand, axis=0)
    
    gen_mean = np.concatenate(all_gen_mean, axis=0)
    gen_var = np.concatenate(all_gen_var, axis=0)
    true_gen = np.concatenate(all_true_gen, axis=0)
    mask_gen = np.concatenate(all_mask_gen, axis=0)
    seas_gen = np.concatenate(all_seas_gen, axis=0)
    
    z_seq = np.concatenate(all_z_seq, axis=0)
    r_seq = np.concatenate(all_r_seq, axis=0)
    
    print("\n--- Evaluation Metrics ---")
    
    def calc_metrics(mean, var, true, mask, name):
        p_flat = mean[mask == 1]
        v_flat = var[mask == 1]
        t_flat = true[mask == 1]
        
        if len(p_flat) == 0:
            return
            
        mae = mean_absolute_error(t_flat, p_flat)
        rmse = np.sqrt(mean_squared_error(t_flat, p_flat))
        
        nll = 0.5 * np.log(2 * np.pi * v_flat) + ((t_flat - p_flat)**2) / (2 * v_flat)
        mean_nll = np.mean(nll)
        
        crps = crps_gaussian(p_flat, v_flat, t_flat)
        mean_crps = np.mean(crps)
        
        # 90% coverage
        z_90 = 1.645
        lower = p_flat - z_90 * np.sqrt(v_flat)
        upper = p_flat + z_90 * np.sqrt(v_flat)
        coverage = np.mean((t_flat >= lower) & (t_flat <= upper))
        
        print(f"[{name}] MAE: {mae:.4f} | RMSE: {rmse:.4f} | NLL: {mean_nll:.4f} | CRPS: {mean_crps:.4f} | 90% Coverage: {coverage:.2%}")

    # Residual performance (Secondary)
    print("\n[SECONDARY] Residual Performance (Scaled)")
    calc_metrics(demand_mean, demand_var, true_demand, mask_demand, "Demand (Resid)")
    calc_metrics(gen_mean, gen_var, true_gen, mask_gen, "Generation (Resid)")
    
    # Reconstruct raw performance (Primary)
    print("\n[PRIMARY] Raw Reconstructed Performance")
    
    # Extract scales and centers for targets
    cols_to_scale = [c for c in feature_cols if not c.endswith('_available')]
    
    demand_scales = np.array([scaler.scale_[cols_to_scale.index(target_cols[i])] for i in demand_idx])
    demand_centers = np.array([scaler.center_[cols_to_scale.index(target_cols[i])] for i in demand_idx])
    
    gen_scales = np.array([scaler.scale_[cols_to_scale.index(target_cols[i])] for i in gen_idx])
    gen_centers = np.array([scaler.center_[cols_to_scale.index(target_cols[i])] for i in gen_idx])
    
    # Unscale and add seasonality
    raw_demand_mean = (demand_mean * demand_scales + demand_centers) + seas_demand
    raw_demand_var = demand_var * (demand_scales ** 2)
    raw_true_demand = (true_demand * demand_scales + demand_centers) + seas_demand
    
    raw_gen_mean = (gen_mean * gen_scales + gen_centers) + seas_gen
    raw_gen_var = gen_var * (gen_scales ** 2)
    raw_true_gen = (true_gen * gen_scales + gen_centers) + seas_gen
    
    calc_metrics(raw_demand_mean, raw_demand_var, raw_true_demand, mask_demand, "Demand (Raw)")
    calc_metrics(raw_gen_mean, raw_gen_var, raw_true_gen, mask_gen, "Generation (Raw)")
    
    # Regime Occupancy
    r_labels = np.argmax(r_seq, axis=-1)
    unique, counts = np.unique(r_labels, return_counts=True)
    occupancy = dict(zip(unique, counts / r_labels.size))
    print(f"Regime Occupancy: {occupancy}")
    
    print("\n--- Generating Plots ---")
    
    print("Generating Fan Chart...")
    sample_enc = test_dataset[0]['encoder_inputs'].unsqueeze(0).to(device)
    sample_dec = test_dataset[0]['decoder_inputs'].unsqueeze(0).to(device)
    sample_true = test_dataset[0]['decoder_targets'][:, demand_idx[0]].numpy()
    
    with torch.no_grad():
        N = 100
        samples = []
        r_outs = []
        for _ in range(N):
            out = model(sample_enc, sample_dec, horizon, sample=True, tau=0.5)
            mean = out['demand_mean'][0, :, 0].cpu().numpy()
            var = out['demand_var'][0, :, 0].cpu().numpy()
            std = np.sqrt(var)
            emissions = np.random.normal(loc=mean, scale=std)
            samples.append(emissions)
            r_outs.append(np.argmax(out['sampled_r_seq'][0].cpu().numpy(), axis=-1))
            
        samples = np.array(samples)
        p10 = np.percentile(samples, 10, axis=0)
        p90 = np.percentile(samples, 90, axis=0)
        mean_forecast = np.mean(samples, axis=0)
        
        # Mode regime
        r_outs = np.array(r_outs)
        r_out, _ = stats.mode(r_outs, axis=0)
        r_out = r_out.flatten() if hasattr(r_out, 'flatten') else (r_out[0] if isinstance(r_out, tuple) else np.array(r_out).flatten())
    
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(horizon)
    plt.plot(time_axis, sample_true, color='black', label='Actual', marker='o', linewidth=2)
    plt.plot(time_axis, mean_forecast, color='blue', label='Mean Forecast')
    plt.fill_between(time_axis, p10, p90, color='blue', alpha=0.2, label='10th-90th Percentile')
    
    # Add regime colors
    colors = ['red', 'green', 'orange', 'purple']
    for t in time_axis:
        plt.axvspan(t-0.5, t+0.5, color=colors[r_out[t] % 4], alpha=0.1)
        
    plt.title(f"30-Day Forecast Fan Chart (H-SSM) - Run: {short_run_id}")
    plt.xlabel("Days Ahead")
    plt.ylabel("Scaled Demand")
    plt.legend()
    plt.savefig(f"eval_fan_chart_{short_run_id}.png", bbox_inches='tight')
    plt.close()
    
    print("Generating PIT Histogram...")
    plt.figure(figsize=(8, 6))
    pit_flat = np.concatenate(all_demand_pit, axis=0)[mask_demand == 1]
    plt.hist(pit_flat, bins=20, density=True, alpha=0.7, color='purple', edgecolor='black')
    plt.axhline(1.0, color='r', linestyle='--', label='Ideal Uniform')
    plt.title("Probability Integral Transform (PIT) Histogram - Demand")
    plt.xlabel("PIT Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"eval_pit_hist_{short_run_id}.png", bbox_inches='tight')
    plt.close()

    print("Evaluation complete.")

if __name__ == "__main__":
    evaluate()
