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
    latent_dim = 16
    hidden_dim = 64
    num_regimes = 4
    
    weather_cols = ['temperature_2m', 'cloudcover', 'windspeed_10m', 'shortwave_radiation']
    fourier_cols = [c for c in feature_cols if '_sin_k' in c or '_cos_k' in c]
    calendar_cols = ['is_bank_holiday'] if 'is_bank_holiday' in feature_cols else []
    embedded_cols = ['EMBEDDED_WIND_CAPACITY', 'EMBEDDED_SOLAR_CAPACITY']
    macro_cols = ['uk_cpi', 'uk_gdp_index', 'bank_rate']
    known_columns = fourier_cols + weather_cols + calendar_cols + [c for c in embedded_cols + macro_cols if c in feature_cols]
    known_dim = len(known_columns)
    
    model = LatentSSM(
        input_dim=len(feature_cols),
        demand_dim=len(demand_cols),
        gen_dim=len(gen_cols),
        known_dim=known_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_regimes=num_regimes,
        fourier_dim=len(fourier_cols)
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
    
    # Print diagnostics at eval time
    with torch.no_grad():
        diag_batch = next(iter(test_loader))
        val_encoder = diag_batch['encoder_inputs'].to(device)
        val_decoder = diag_batch['decoder_inputs'].to(device)
        val_targets = diag_batch['decoder_targets'].to(device)
        
        outputs = model(val_encoder, val_decoder, horizon, sample=False)
        
        demand_var = (outputs['demand_nu'] / (outputs['demand_nu'] - 2.0)) * (outputs['demand_scale'] ** 2)
        target_demand_std = val_targets[:, :, demand_idx].std()
        
        print(f"Demand scale range: {outputs['demand_scale'].min():.3f} - {outputs['demand_scale'].max():.3f}")
        print(f"Demand nu range: {outputs['demand_nu'].min():.1f} - {outputs['demand_nu'].max():.1f}")
        print(f"Demand var range: {demand_var.min():.3f} - {demand_var.max():.3f}")
        
        scale_percentiles = torch.quantile(
            outputs['demand_scale'].flatten(), 
            torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99], device=device)
        )
        print(f"Demand Scale Percentiles (1%, 25%, 50%, 75%, 99%):")
        print(f"  {scale_percentiles.cpu().numpy()}")
        print(f"Target Demand Std: {target_demand_std:.4f}")
        print(f"Ratio scale/target_std: {(outputs['demand_scale'] / target_demand_std).mean():.2f}")
        print("-" * 40)

        demand_scale = outputs['demand_scale']
        target_demand = val_targets[:, :, demand_idx]
        demand_nu = outputs['demand_nu']
    
        var_factor = demand_nu / (demand_nu - 2.0)
        std = torch.sqrt(var_factor) * demand_scale
    
        z_score = 1.645
        ci_width = 2.0 * z_score * std
    
        print(f"90% CI width - Median: {ci_width.median():.3f}")
        print(f"90% CI width - Mean: {ci_width.mean():.3f}")
        print(f"Target demand std: {target_demand.std():.3f}")
        print(f"Target CI width (1.5x): {1.5 * target_demand.std():.3f}")
        print(f"\nRatio (actual / target): {ci_width.mean() / (1.5 * target_demand.std()):.2f}")
        print("-" * 40)
    
        demand_mean = outputs['demand_mean']
        target_demand = val_targets[:, :, demand_idx]
        
        print(f"Target demand range: [{target_demand.min():.1f}, {target_demand.max():.1f}]")
        print(f"Target demand mean: {target_demand.mean():.1f}, std: {target_demand.std():.4f}")
        print(f"Predicted mean range: [{demand_mean.min():.1f}, {demand_mean.max():.1f}]")
        print(f"Predicted mean mean: {demand_mean.mean():.1f}, std: {demand_mean.std():.4f}")
        
        # Compute mean error IN THE SAME SPACE
        mean_error = torch.abs(demand_mean - target_demand)
        print(f"Mean error range: [{mean_error.min():.1f}, {mean_error.max():.1f}]")
        print(f"Mean error mean: {mean_error.mean():.1f}")
        print(f"Mean error as multiple of target_std: {mean_error.mean() / target_demand.std():.1f}x")
        
        # Posterior vs Prior gap
        outputs_post = model(val_encoder, val_decoder, horizon, target_seq=val_targets, sample=False)
        post_mean_error = torch.abs(outputs_post['demand_mean'] - target_demand)
        print(f"\nPrior-only MAE: {mean_error.mean():.4f}")
        print(f"Posterior+Prior MAE: {post_mean_error.mean():.4f}")
        print(f"Prior-only vs Posterior+Prior MAE gap: {mean_error.mean() - post_mean_error.mean():.4f}")
        
        
        # Computed generation stats
        gen_idx = [target_cols.index(c) for c in gen_cols]
        gen_mean = outputs['gen_mean']
        target_gen = val_targets[:, :, gen_idx]
        gen_var = (outputs['gen_nu'] / (outputs['gen_nu'] - 2.0)) * (outputs['gen_scale'] ** 2)
        target_gen_std = target_gen.std()
        
        print(f"\nGeneration scale range: {outputs['gen_scale'].min():.3f} - {outputs['gen_scale'].max():.3f}")
        print(f"Target Generation Std: {target_gen_std:.4f}")
        print(f"Gen Ratio scale/target_std: {(outputs['gen_scale'] / target_gen_std).mean():.2f}")
        
        gen_mean_error = torch.abs(gen_mean - target_gen)
        print(f"Gen Mean error range: [{gen_mean_error.min():.1f}, {gen_mean_error.max():.1f}]")
        print(f"Gen Mean error mean: {gen_mean_error.mean():.1f}")
        print(f"Gen Mean error as multiple of target_std: {gen_mean_error.mean() / target_gen_std:.1f}x")
        print("-" * 40)
        
        # Check if targets/predictions are normalized vs raw
        print(f"\nAre targets normalized? {target_demand.max() <= 1.0 and target_demand.min() >= -1.0}")
        print(f"Are predictions normalized? {demand_mean.max() <= 1.0 and demand_mean.min() >= -1.0}")
    
    all_demand_mean = []
    all_demand_var = []
    all_true_demand = []
    all_mask_demand = []
    
    all_gen_mean = []
    all_gen_var = []
    all_true_gen = []
    all_mask_gen = []
    
    all_z_seq = []
    all_r_seq = []
    all_demand_pit = []
    
    with torch.no_grad():
        for batch in test_loader:
            enc_inputs = batch['encoder_inputs'].to(device)
            dec_inputs = batch['decoder_inputs'].to(device)
            dec_targets = batch['decoder_targets'].numpy()
            dec_masks = batch['decoder_mask'].numpy()
            

            
            N = 50
            batch_demand_samples = []
            batch_gen_samples = []
            
            for _ in range(N):
                outputs = model(enc_inputs, dec_inputs, horizon, target_seq=None, sample=True, tau=1.0)
                
                d_mean = outputs['demand_mean'].cpu().numpy()
                d_scale = outputs['demand_scale'].cpu().numpy()
                d_nu = outputs['demand_nu'].cpu().numpy()
                g_mean = outputs['gen_mean'].cpu().numpy()
                g_scale = outputs['gen_scale'].cpu().numpy()
                g_nu = outputs['gen_nu'].cpu().numpy()
                
                d_sample = d_mean + np.random.standard_t(df=np.maximum(d_nu, 2.001)) * d_scale
                g_sample = g_mean + np.random.standard_t(df=np.maximum(g_nu, 2.001)) * g_scale
                
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
            
            all_gen_mean.append(gen_mean)
            all_gen_var.append(gen_var)
            all_true_gen.append(dec_targets[:, :, gen_idx])
            all_mask_gen.append(dec_masks[:, :, gen_idx])
            
            # For simplicity, just append the last r_seq
            all_z_seq.append(outputs['sampled_z_seq'].cpu().numpy())
            all_r_seq.append(outputs['sampled_r_seq'].cpu().numpy())
            
    # Concatenate all batches
    demand_mean = np.concatenate(all_demand_mean, axis=0)
    demand_var = np.concatenate(all_demand_var, axis=0)
    true_demand = np.concatenate(all_true_demand, axis=0)
    mask_demand = np.concatenate(all_mask_demand, axis=0)
    
    gen_mean = np.concatenate(all_gen_mean, axis=0)
    gen_var = np.concatenate(all_gen_var, axis=0)
    true_gen = np.concatenate(all_true_gen, axis=0)
    mask_gen = np.concatenate(all_mask_gen, axis=0)
    
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

    # Reconstruct raw performance
    print("\n--- Performance Metrics ---")
    
    # Extract scales and centers for targets
    cols_to_scale = [c for c in feature_cols if not c.endswith('_available')]
    
    demand_scales = np.array([scaler.scale_[cols_to_scale.index(target_cols[i])] for i in demand_idx])
    demand_centers = np.array([scaler.center_[cols_to_scale.index(target_cols[i])] for i in demand_idx])
    
    gen_scales = np.array([scaler.scale_[cols_to_scale.index(target_cols[i])] for i in gen_idx])
    gen_centers = np.array([scaler.center_[cols_to_scale.index(target_cols[i])] for i in gen_idx])
    
    # Unscale predictions and targets
    raw_demand_mean = demand_mean * demand_scales + demand_centers
    raw_demand_var = demand_var * (demand_scales ** 2)
    raw_true_demand = true_demand * demand_scales + demand_centers
    
    raw_gen_mean = gen_mean * gen_scales + gen_centers
    raw_gen_var = gen_var * (gen_scales ** 2)
    raw_true_gen = true_gen * gen_scales + gen_centers
    
    calc_metrics(raw_demand_mean, raw_demand_var, raw_true_demand, mask_demand, "Demand")
    calc_metrics(raw_gen_mean, raw_gen_var, raw_true_gen, mask_gen, "Generation")
    
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
            scale = out['demand_scale'][0, :, 0].cpu().numpy()
            nu = out['demand_nu'][0, :, 0].cpu().numpy()
            emissions = mean + np.random.standard_t(df=np.maximum(nu, 2.001)) * scale
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
