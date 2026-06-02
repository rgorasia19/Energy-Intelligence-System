import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

def pull_data(path:str):
    df = pd.read_csv(path)
    return df

df = pull_data("../data_lake/master_df.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

sum_cols = [col for col in df.columns if not col.startswith('Timestamp')]
df["Total_load"] = df[sum_cols].sum(axis=1)

plt.plot(df["Timestamp"],df["Total_load"])
plt.show()
plt.close()

plt.figure(figsize=(10,5))
plt.plot(df["Timestamp"], df["Total_load"].diff().abs().rolling(48).mean())
plt.title("Rolling 48-Period Volatility of Total Load using diff+abs")
plt.xlabel("Timestamp")
plt.ylabel("Standard Deviation")
plt.grid(True)
plt.savefig("Volatility_diff_abs.jpg", dpi=600)
plt.show()

plt.figure(figsize=(10,5))
autocorrelation_plot(df["Total_load"])
plt.title("Autocorrelation of Total Load")
plt.savefig("Autocorrelation.jpg", dpi=600)
plt.show()