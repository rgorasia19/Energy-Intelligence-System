import pandas as pd
import numpy as np

data_dir = "../../datalake/raw_data/generation_mix.csv"

print(f"Reading {data_dir}...")
df = pd.read_csv(data_dir)

print("Formatting DATETIME column...")
# Convert DATETIME from 'YYYY-MM-DDTHH:MM:SS' to 'YYYY-MM-DD HH:MM:SS'
df['DATETIME'] = df['DATETIME'].str.replace('T', ' ')

print(f"Saving changes to {data_dir}...")
df.to_csv(data_dir, index=False)

print("Clean completed successfully.")
