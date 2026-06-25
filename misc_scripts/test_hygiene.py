import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# 1. Load data
pca_path = 'datalake/pca/df_pca.parquet'
raw_path = 'datalake/raw_data/full_data.csv'

df_pca = pd.read_parquet(pca_path)
df_pca.index = pd.to_datetime(df_pca.index)

df_raw = pd.read_csv(raw_path)
df_raw['DATETIME'] = pd.to_datetime(df_raw['DATETIME'])
df_raw.set_index('DATETIME', inplace=True)

demand_cols = ['ND', 'TSD', 'ENGLAND_WALES_DEMAND']
df_demand = df_raw[demand_cols]

df = df_pca.join(df_demand, how='inner')
df.sort_index(inplace=True)

# Define candidates
candidates = {
    "PCA_only_top1": ['INTER_PC0', 'GEN_PC0', 'CAP_PC0'],
    "PCA_only_top2": ['INTER_PC0', 'INTER_PC1', 'GEN_PC0', 'GEN_PC1', 'CAP_PC0', 'CAP_PC1'],
    "Raw_demand_only": ['ND', 'TSD'],
    "Curated_macro": ['ND', 'INTER_PC0', 'GEN_PC0', 'CAP_PC0']
}

for name, base_feats in candidates.items():
    print(f"\n=================== Testing: {name} ===================")
    # Extract base features
    df_feat = df[base_feats].copy()
    
    # Add minimal lags (lag 1 and lag 24)
    lags = [1, 24]
    for lag in lags:
        for col in base_feats:
            df_feat[f"{col}_LAG_{lag}"] = df_feat[col].shift(lag)
            
    # Target
    df_feat['TARGET_ND'] = df['ND'].shift(-1)
    df_feat.dropna(inplace=True)
    
    # Check data type stability - cast to float32
    df_feat = df_feat.astype(np.float32)
    
    feature_cols = [c for c in df_feat.columns if c != 'TARGET_ND']
    X = df_feat[feature_cols].values
    
    # Fit StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Covariance Matrix Analysis
    cov_matrix = np.cov(X_scaled, rowvar=False)
    rank = np.linalg.matrix_rank(cov_matrix)
    cond_num = np.linalg.cond(cov_matrix)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    min_eig = np.min(eigenvalues)
    
    print(f"Number of features: {len(feature_cols)}")
    print(f"Covariance Matrix Rank: {rank}/{len(feature_cols)}")
    print(f"Condition Number: {cond_num:.2f}")
    print(f"Minimum Eigenvalue: {min_eig:.8f}")
    
    # Fit GMM
    gmm = GaussianMixture(n_components=4, covariance_type='diag', random_state=42)
    gmm.fit(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    labels = probs.argmax(axis=1)
    
    # Stats
    balance = pd.Series(labels).value_counts(normalize=True).sort_index()
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1).mean()
    persistence = np.mean(np.diff(labels) == 0)
    
    print("Regime Balance:")
    for regime, weight in balance.items():
        print(f"  Regime {regime}: {weight*100:.2f}%")
    print(f"Mean Posterior Entropy: {entropy:.4f}")
    print(f"State Persistence: {persistence*100:.2f}%")
