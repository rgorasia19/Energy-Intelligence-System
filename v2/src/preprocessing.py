import pandas as pd
import numpy as np

#1. Import dataset from CSV
df = pd.read_csv('../datalake/full_data.csv')

#2. Check for missing values
#print("\nMissing values:")
#print(df.isnull().sum())

#3. Check for duplicate values
#print("\nDuplicate values:")
#print(df.duplicated().sum())

#4. Drop unnecessary columns
for col in df.columns:
    if "perc" in col.lower():
        df.drop(col, axis=1, inplace=True)
        print(f"Dropped column: {col}")
    elif "flow" in col.lower():
        df.drop(col, axis=1, inplace=True)
        print(f"Dropped column: {col}")
    elif "embedded" in col.lower():
        df.drop(col, axis=1, inplace=True)
        print(f"Dropped column: {col}")

#5. Save processed data
df.to_csv('../datalake/processed_data.csv', index=False)
print("Processed data saved to ../datalake/processed_data.csv")