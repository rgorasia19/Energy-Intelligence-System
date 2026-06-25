import pandas as pd
import numpy as np

df = pd.read_parquet('../../datalake/clean+features/feature_df.parquet')

train_split = int(len(df)*0.7)
val_split = int(len(df)*0.85)

train_df = df.iloc[:train_split]
val_df = df.iloc[train_split:val_split]
test_df = df.iloc[val_split:]

print(f"Training data shape: {train_df.shape}")
print(f"Validation data shape: {val_df.shape}")
print(f"Testing data shape: {test_df.shape}")

train_df.to_parquet('../../datalake/splits/v2/train_df.parquet', index=True)
val_df.to_parquet('../../datalake/splits/v2/val_df.parquet', index=True)
test_df.to_parquet('../../datalake/splits/v2/test_df.parquet', index=True)