import pandas as pd
import os

file_path = "../../datalake/raw_data/"
files = [f for f in os.listdir(file_path) if f.startswith("demanddata_") and f.endswith(".csv")]

all_data = []

for file in files:
    print(f"Reading {file}...")
    full_path = os.path.join(file_path, file)
    df = pd.read_csv(full_path)
    all_data.append(df)

print("Concatenating all files...")
merged_df = pd.concat(all_data, ignore_index=True)

print("Merging SETTLEMENT_DATE and SETTLEMENT_PERIOD into DATETIME...")
# Combine the date and time strings and convert to datetime
merged_df['DATETIME'] = pd.to_datetime(merged_df['SETTLEMENT_DATE'] + ' ' + merged_df['SETTLEMENT_PERIOD'])

# Drop the old columns
merged_df.drop(columns=['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD'], inplace=True)

# Reorder columns to put DATETIME first
cols = merged_df.columns.tolist()
cols = ['DATETIME'] + [c for c in cols if c != 'DATETIME']
merged_df = merged_df[cols]

print("Sorting chronologically...")
merged_df.sort_values(by='DATETIME', inplace=True)

output_file = os.path.join(file_path, "merged_demanddata.csv")
print(f"Saving merged data to {output_file}...")
merged_df.to_csv(output_file, index=False)

print("Merge completed successfully.")
