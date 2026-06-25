import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../datalake/full_data.csv")
df["DATETIME"] = pd.to_datetime(df["DATETIME"])
df.index = df["DATETIME"]
df = df.drop(columns = ["DATETIME"])

def plot_ND():
  plt.plot(df["ND"])
  plt.title("National Demand over time")
  plt.ylabel("MW")
  plt.xlabel("Time")
  plt.savefig("ND.jpg",dpi=600)
  plt.show()

def plot_gen_mix():
  plt.plot(df["GAS"], label="gas")
  plt.plot(df["COAL"], label="coal")
  plt.plot(df["NUCLEAR"], label="nuclear")
  plt.plot(df["WIND"], label="wind")
  plt.plot(df["HYDRO"], label="hydro")
  plt.plot(df["WIND_EMB"], label="wind_emb")
  plt.plot(df["IMPORTS"], label="imported")
  plt.plot(df["BIOMASS"], label="biomass")
  plt.plot(df["SOLAR"], label="solar")
  plt.title("Generation Mix over time")
  plt.ylabel("MW")
  plt.xlabel("Time")
  plt.legend()
  plt.savefig("gen_mix.jpg",dpi=600)
  plt.show()

def global_correlation_matrix():
  from scipy.stats import pearsonr
  solar_x_carbon = pearsonr(df["SOLAR"],df["CARBON_INTENSITY"])
  wind_x_carbon = pearsonr(df["WIND"],df["CARBON_INTENSITY"])
  hydro_x_carbon = pearsonr(df["HYDRO"],df["CARBON_INTENSITY"])
  print(f"SOLAR X CARBON : ",solar_x_carbon.statistic)
  print(f"WIND X CARBON : ",wind_x_carbon.statistic)
  print(f"HYDRO X Carbon : ",hydro_x_carbon.statistic)

def seasonality():
  fig, axes = plt.subplots(4, 3, figsize=(18, 20))
  
  time_features = {
      'Hour': df.index.hour,
      'Day of Week': df.index.dayofweek,
      'Week of Year': df.index.isocalendar().week,
      'Month': df.index.month
  }
  
  metrics = ["ND", "SOLAR", "CARBON_INTENSITY"]
  
  for i, (time_name, time_feat) in enumerate(time_features.items()):
      for j, metric in enumerate(metrics):
          avg_data = df.groupby(time_feat)[metric].mean()
          axes[i, j].plot(avg_data.index, avg_data.values, marker='o')
          axes[i, j].set_title(f'Average {metric} by {time_name}')
          axes[i, j].set_xlabel(time_name)
          axes[i, j].set_ylabel(metric)
          axes[i, j].grid(True)
          
  plt.tight_layout()
  plt.savefig("seasonality.jpg", dpi=600)
  plt.show()

def pca():
  from sklearn.preprocessing import StandardScaler
  from sklearn.decomposition import PCA
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(df)
  
  pca = PCA()
  X_pca = pca.fit_transform(X_scaled)

  loadings = pd.DataFrame(
    pca.components_.T,
    index=df.columns,
    columns=[f"PC{i+1}" for i in range(len(df.columns))]
  )

  loadings.to_csv("PCA_loadings.csv")
  plt.figure(figsize=(8,5))
  plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
  plt.title('Cumulative Explained Variance by PCA Components')
  plt.xlabel('Number of Components')
  plt.ylabel('Cumulative Explained Variance')
  plt.grid(True)
  plt.savefig("PCA_Explained_Variance.jpg", dpi=600)
  plt.show()

  plt.figure(figsize=(8, 6))
  plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, cmap='viridis')
  plt.title('PCA: First Two Principal Components')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.grid(True)
  plt.savefig("PCA_Scatter_Plot.jpg", dpi=600)
  plt.show()

pca()