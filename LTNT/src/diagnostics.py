import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def calculate_latent_drift(z_seq_flat):
    """
    z_seq_flat: (N, latent_dim)
    Returns mean and variance across the dataset
    """
    mean_drift = np.mean(z_seq_flat, axis=0)
    var_drift = np.var(z_seq_flat, axis=0)
    return mean_drift, var_drift

def calculate_temporal_smoothness(z_seq):
    """
    z_seq: (B, seq_len, latent_dim)
    """
    if z_seq.shape[1] > 1:
        diffs = z_seq[:, 1:, :] - z_seq[:, :-1, :]
        smoothness = np.mean(diffs ** 2)
    else:
        smoothness = 0.0
    return smoothness

def plot_forecast_error_by_volatility(errors, actual_vol, bins=5, save_path="error_by_vol.png"):
    """
    errors: (N,) absolute errors
    actual_vol: (N,) volatility values
    """
    vol_bins = np.percentile(actual_vol, np.linspace(0, 100, bins+1))
    bin_indices = np.digitize(actual_vol, vol_bins) - 1
    # Handle edge case where max value gets put in bin+1
    bin_indices = np.clip(bin_indices, 0, bins-1)
    
    mean_errors = []
    for i in range(bins):
        mask = (bin_indices == i)
        if np.any(mask):
            mean_errors.append(np.mean(errors[mask]))
        else:
            mean_errors.append(0)
            
    plt.figure(figsize=(8, 5))
    plt.bar(range(bins), mean_errors, color='skyblue')
    plt.title('Forecast Error (MAE) by Volatility Quintile')
    plt.xlabel('Volatility Quintile (0=Low, 4=High)')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(range(bins))
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_latent_trajectory(z_seq_flat, save_path="latent_pca.png", method='pca'):
    """
    z_seq_flat: (N, latent_dim)
    """
    # Sample if too large
    if len(z_seq_flat) > 5000:
        indices = np.random.choice(len(z_seq_flat), 5000, replace=False)
        z_sample = z_seq_flat[indices]
    else:
        z_sample = z_seq_flat
        
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, learning_rate='auto', init='random')
        
    z_2d = reducer.fit_transform(z_sample)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.5, s=10, c=np.arange(len(z_sample)), cmap='viridis')
    plt.colorbar(label='Time (Sample Index)')
    plt.title(f'Latent Trajectory ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(save_path, dpi=300)
    plt.close()

def feature_influence_analysis(z_seq_flat, actual_vol, actual_trend):
    """
    z_seq_flat: (N, latent_dim)
    actual_vol: (N,)
    actual_trend: (N,)
    """
    latent_dim = z_seq_flat.shape[1]
    vol_corr = []
    trend_corr = []
    
    for i in range(latent_dim):
        z_i = z_seq_flat[:, i]
        vol_corr.append(np.corrcoef(z_i, actual_vol)[0, 1])
        trend_corr.append(np.corrcoef(z_i, actual_trend)[0, 1])
        
    return vol_corr, trend_corr
