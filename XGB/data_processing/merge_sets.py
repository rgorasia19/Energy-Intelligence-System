import pandas as pd
import numpy as np

demand_df = pd.read_csv("../../datalake/raw_data/merged_demanddata.csv")
generation_df = pd.read_csv("../../datalake/raw_data/generation_mix.csv")


merged_df = pd.merge(demand_df, generation_df, on='DATETIME', how='inner')
merged_df.to_csv("../../datalake/raw_data/full_data.csv", index=False)

print("Merge completed successfully.")