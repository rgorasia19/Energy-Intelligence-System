import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pull_data(path:str):
    df = pd.read_csv(path)
    return df

import random

df = pull_data("../data_lake/master_df.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Pick a random date
dates = df["Timestamp"].dt.date.unique()
random_date = random.choice(dates)

# Filter the dataframe for only that day
df_day = df[df["Timestamp"].dt.date == random_date]

for col in df_day.columns:
    if "drakelow" in col:
        plt.figure(figsize=(10, 5))
        plt.grid(True, alpha = 0.3)
        plt.plot(df_day["Timestamp"], df_day[col])
        plt.title(f"Drakelow Substation Load - {random_date}")

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"Drakelow_substation_load_{random_date}.jpg",dpi=600)
plt.show()
