import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pull_data(path:str):
    df = pd.read_csv(path)
    return df

df = pull_data("../data_lake/master_df.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Drop the Timestamp column for PCA
X = df.drop(columns=["Timestamp"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=[f"PC{i+1}" for i in range(len(X.columns))]
)

loadings.to_csv("PCA_Scores.csv")
# 1. Plot Explained Variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.savefig("PCA_Explained_Variance.jpg", dpi=600)
plt.show()

# 2. Plot First Two Principal Components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, cmap='viridis')
plt.title('PCA: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig("PCA_Scatter_Plot.jpg", dpi=600)
plt.show()
