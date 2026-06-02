import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pull_data(path:str):
    df = pd.read_csv(path)
    return df

import random

df = pull_data("../data_lake/master_df.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

X = df.drop(columns=["Timestamp"])
corr_matrix = X.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation Matrix of All Stations")
plt.tight_layout()
plt.savefig("Correlation_Matrix.jpg",dpi=600)
plt.show()

plt.figure(figsize=(12,10))
sns.clustermap(corr_matrix,cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation Clustering of All Stations")
plt.savefig("Correlation_Clustering.jpg",dpi=600)
plt.show()