import pandas as pd
import numpy as np
import os

pca_path = 'datalake/pca/df_pca.parquet'
raw_path = 'datalake/raw_data/full_data.csv'

if os.path.exists(pca_path):
    df_pca = pd.read_parquet(pca_path)
    print("df_pca shape:", df_pca.shape)
    print("df_pca columns:", list(df_pca.columns))
else:
    print("df_pca path not found")

if os.path.exists(raw_path):
    # Only load first 1000 rows to check columns
    df_raw = pd.read_csv(raw_path, nrows=1000)
    print("df_raw columns:", list(df_raw.columns))
else:
    print("df_raw path not found")
