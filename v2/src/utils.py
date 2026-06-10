import pandas as pd
import numpy as np

df = pd.read_csv('../../datalake/clean+features/feature_df.csv')
df['DATETIME'] = pd.to_datetime(df['DATETIME'])
df.set_index('DATETIME', inplace=True)

train_split = int(len(df)*0.7)
val_split = int(len(df)*0.85)

train_df = df.iloc[:train_split]
val_df = df.iloc[train_split:val_split]
test_df = df.iloc[val_split:]

print(f"Training data shape: {train_df.shape}")
print(f"Validation data shape: {val_df.shape}")
print(f"Testing data shape: {test_df.shape}")

train_df.to_csv('../../datalake/splits/train_df.csv', index=True)
val_df.to_csv('../../datalake/splits/val_df.csv', index=True)
test_df.to_csv('../../datalake/splits/test_df.csv', index=True)