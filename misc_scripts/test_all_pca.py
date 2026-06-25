import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

pca_path = 'datalake/pca/df_pca.parquet'
df_pca = pd.read_parquet(pca_path)
df_pca.index = pd.to_datetime(df_pca.index)

# Option A: All PCA components + lags [1, 24]
df_feat = df_pca.copy()
base_feats = list(df_pca.columns)
lags = [1, 24]
for lag in lags:
    for col in base_feats:
        df_feat[f"{col}_LAG_{lag}"] = df_feat[col].shift(lag)

df_feat.dropna(inplace=True)
df_feat = df_feat.astype(np.float32)

X = df_feat.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cov_matrix = np.cov(X_scaled, rowvar=False)
rank = np.linalg.matrix_rank(cov_matrix)
cond_num = np.linalg.cond(cov_matrix)
eigenvalues = np.linalg.eigvalsh(cov_matrix)
min_eig = np.min(eigenvalues)

print(f"Number of features (All PCA + lags): {len(base_feats) * 3}")
print(f"Covariance Matrix Rank: {rank}/{len(base_feats) * 3}")
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
