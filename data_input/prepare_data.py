import numpy as np
import pandas as pd
from pathlib import Path

def load_and_parse(path:Path):
    df_list = []
    timestamps = None
    for f in path.rglob("*.csv"):
        if f.name == "master_df.csv":
            continue
            
        df = pd.read_csv(f)
        if timestamps is None:
            timestamps = df['Power Units_Bus Number'].values
            
        mw_cols = [col for col in df.columns if col.startswith('MW_')]
        # Compute the total MW for this file
        total_mw = df[mw_cols].sum(axis=1)
        total_mw.name = f.stem # Optionally rename the column to the file's name
        df_list.append(total_mw)
        
    # Concatenate all series at once to avoid fragmentation
    master_df = pd.concat(df_list, axis=1)
    master_df['Total'] = master_df.sum(axis=1)
    master_df.index = timestamps
    master_df.index.name = "Timestamp"
    
    out_file = path / "master_df.csv"
    master_df.to_csv(out_file, index=True)
    print(master_df.head())
    return master_df

load_and_parse(Path("C:\\Users\\ronak\\OneDrive\\Desktop\\Random\\Energy-Intelligence-System\\data_lake"))