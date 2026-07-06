import pandas as pd
import numpy as np
import os
import datetime as dt

file_path = "../../datalake/raw_data/"
files = [f for f in os.listdir(file_path) if f.startswith("demanddata_") and f.endswith(".csv")]

# Read the 2026 file to get target columns
df_2026 = pd.read_csv(os.path.join(file_path, "demanddata_2026.csv"))
target_columns = df_2026.columns.tolist()

for file in files:
    full_path = os.path.join(file_path, file)
    print(f"Cleaning {file}...")
    df = pd.read_csv(full_path)
    
    # Standardize the date format to YYYY-MM-DD
    df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE']).dt.strftime('%Y-%m-%d')
    
    # Convert SETTLEMENT_PERIOD to half-hour increments from 00:00, if not already
    if pd.api.types.is_numeric_dtype(df['SETTLEMENT_PERIOD']):
        times = pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 30, unit='m')
        base_time = pd.to_datetime('2000-01-01')
        df['SETTLEMENT_PERIOD'] = (base_time + times).dt.strftime('%H:%M:%S')
    
    # Add any missing columns and fill with 0
    for col in target_columns:
        if col not in df.columns:
            df[col] = 0
            
    # Reorder columns to match the 2026 file structure
    df = df[target_columns]
    
    # Save the cleaned data back to the file
    df.to_csv(full_path, index=False)

print("All files cleaned.")
