import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pull_data(path:str):
    df = pd.read_csv(path)
    return df

df = pull_data("../data_lake/master_df.csv")
df.index = pd.to_datetime(df["Timestamp"])
df.drop(columns=["Timestamp"], inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

pc_df = pd.DataFrame(
  X_pca[:,:2],
  index = df.index,
  columns = ["PC1","PC2"]
)

plot_df = pd.DataFrame(index=df.index)
plot_df["PC1"] = X_pca[:, 0]

#Find Total Demand Using Scaled
scaled_total_demand = X_scaled.sum(axis=1)

#Normalizing using MinMax
plot_df["PC1_normalized"] = (plot_df["PC1"] - plot_df["PC1"].min()) / (plot_df["PC1"].max() - plot_df["PC1"].min())
plot_df["Demand_normalized"] = (scaled_total_demand - scaled_total_demand.min()) / (scaled_total_demand.max() - scaled_total_demand.min())

#Plotting
def create_plots():
  plt.figure(figsize=(12, 6))
  plt.plot(plot_df.index, plot_df["PC1_normalized"], linestyle="-", linewidth=1, label="PC1 (Normalized)", alpha=0.8)
  plt.plot(plot_df.index, plot_df["Demand_normalized"], linestyle="--", linewidth=1, label="Total Scaled Demand", alpha=0.7)

  plt.title("Verification: PC1 Alignment with Global Network Demand")
  plt.ylabel("Normalized Scale (0-1)")
  plt.legend()
  plt.tight_layout()
  plt.savefig("PC1_Demand_Verification.jpg", dpi=600)
  plt.show()

  plt.figure(figsize=(12, 6))
  plt.plot(pc_df.index, pc_df["PC2"])
  plt.ylabel("PC2")
  plt.title("Principal Component 2 - System Structure")
  plt.savefig("PC2.jpg", dpi=600)
  plt.show()

  pc_df["Hour"] = pc_df.index.hour
  pc_df["Month"] = pc_df.index.month
  plt.figure(figsize=(12, 6))
  sns.boxplot(x="Hour", y="PC2", data=pc_df)
  plt.title("Principal Component 2 - System Structure by Hour")
  plt.ylabel("PC2")
  plt.xlabel("Hour")
  plt.tight_layout()
  plt.savefig("PC2_hour.jpg", dpi=600)
  plt.show()

  plt.figure(figsize=(12, 6))
  sns.boxplot(x="Month", y="PC2", data=pc_df)
  plt.title("Principal Component 2 - System Structure by Month")
  plt.ylabel("PC2")
  plt.xlabel("Month")
  plt.tight_layout()
  plt.savefig("PC2_month.jpg", dpi=600)
  plt.show()  

# Defining thresholds
#print(pc_df["PC2"].describe())
#plt.plot(pc_df["PC2"])
#plt.show()

pc2_z = (pc_df["PC2"] - pc_df["PC2"].mean()) / pc_df["PC2"].std()
pc_df["PC2_z"] = pc2_z

threshold = pc_df["PC2_z"].quantile(0.35)
pc_df["regime"] = (pc_df["PC2_z"] < threshold).astype(int)
print(pc_df["regime"].value_counts())

def visualise_regimes():
    plt.figure(figsize=(12,6))
    plt.plot(pc_df.index, pc_df["PC2_z"], label="PC2_z")
    plt.axhline(1, color='r', linestyle='--')
    plt.axhline(-1, color='r', linestyle='--')
    plt.legend()
    plt.savefig("PC2_z.jpg", dpi=600)
    plt.show()
    
    plt.figure(figsize=(12,4))
    plt.plot(pc_df["regime"], label="regime")
    plt.title("Regime (0=normal / 1=solar/export)")
    plt.legend()
    plt.savefig("regime.jpg", dpi=600)
    plt.show()

#visualise_regimes()

pc_df.to_csv("pc_df.csv")
