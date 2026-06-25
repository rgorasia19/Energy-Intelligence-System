import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

pca_path = 'datalake/pca/df_pca.parquet'
raw_path = 'datalake/raw_data/full_data.csv'

df_pca = pd.read_parquet(pca_path)
df_pca.index = pd.to_datetime(df_pca.index)

df_raw = pd.read_csv(raw_path)
df_raw['DATETIME'] = pd.to_datetime(df_raw['DATETIME'])
df_raw.set_index('DATETIME', inplace=True)

df = df_pca.join(df_raw[['ND']], how='inner')
df.sort_index(inplace=True)

feature_sets = {
    "PCA_top1": ['INTER_PC0', 'GEN_PC0', 'CAP_PC0'],
    "PCA_top2": ['INTER_PC0', 'INTER_PC1', 'GEN_PC0', 'GEN_PC1', 'CAP_PC0', 'CAP_PC1'],
    "Curated_macro": ['ND', 'INTER_PC0', 'GEN_PC0', 'CAP_PC0']
}

lag_configs = {
    "No_lags": [],
    "Lag_1": [1],
    "Lag_1_24": [1, 24]
}

for set_name, base_feats in feature_sets.items():
    for lag_name, lags in lag_configs.items():
        df_feat = df[base_feats].copy()
        for lag in lags:
            for col in base_feats:
                df_feat[f"{col}_LAG_{lag}"] = df_feat[col].shift(lag)
        
        df_feat['TARGET_ND'] = df['ND'].shift(-1)
        df_feat.dropna(inplace=True)
        df_feat = df_feat.astype(np.float32)
        
        feature_cols = [c for c in df_feat.columns if c != 'TARGET_ND']
        X = df_feat[feature_cols].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        cov_matrix = np.cov(X_scaled, rowvar=False)
        rank = np.linalg.matrix_rank(cov_matrix)
        cond_num = np.linalg.cond(cov_matrix)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        min_eig = np.min(eigenvalues)
        
        # GMM Fit
        gmm = GaussianMixture(n_components=4, covariance_type='diag', random_state=42)
        gmm.fit(X_scaled)
        probs = gmm.predict_proba(X_scaled)
        labels = probs.argmax(axis=1)
        
        balance = pd.Series(labels).value_counts(normalize=True).sort_index()
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1).mean()
        persistence = np.mean(np.diff(labels) == 0)
        
        print(f"{set_name} | {lag_name} | Feats: {len(feature_cols)} | Rank: {rank}/{len(feature_cols)} | Cond: {cond_num:.2f} | MinEig: {min_eig:.6f} | Entropy: {entropy:.4f} | Persistence: {persistence*100:.2f}%")
