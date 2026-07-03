import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

df = pd.read_parquet('../datalake/pca/df_pca.parquet')

X = StandardScaler().fit_transform(df)

gmm = GaussianMixture(n_components=4, covariance_type='full')
gmm.fit(X)

regime_probs = gmm.predict_proba(X)
print(regime_probs.max(axis = 1).mean())

regime_labels = regime_probs.argmax(axis = 1)
print("Persistence:", np.mean(np.diff(regime_labels)==0))

print("\n--- Regime Balance ---")
balance = pd.Series(regime_labels).value_counts(normalize=True).sort_index()
print(balance)

print("\n--- Regime Interpretation (Mean Scaled PCA Features) ---")
df_scaled = pd.DataFrame(X, columns=df.columns)
df_scaled['Regime'] = regime_labels
regime_means = df_scaled.groupby('Regime').mean()

for regime in range(4):
    print(f"\nRegime {regime} ({(balance[regime]*100):.1f}% of data):")
    means = regime_means.loc[regime].sort_values()
    print("  Defining Negative Features (Below Average):")
    print(means.head(3).to_string())
    print("  Defining Positive Features (Above Average):")
    print(means.tail(3).sort_values(ascending=False).to_string())