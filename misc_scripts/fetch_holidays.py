import pandas as pd
import requests
import os

def fetch_and_merge_holidays():
    url = "https://www.gov.uk/bank-holidays.json"
    print(f"Fetching bank holidays from {url}...")
    response = requests.get(url)
    data = response.json()
    
    events = data['england-and-wales']['events']
    holiday_dates = [event['date'] for event in events]
    holiday_dates = pd.to_datetime(holiday_dates).tz_localize(None)
    
    # Assume script is run from misc_scripts
    parquet_path = "../datalake/clean+features/feature_df.parquet"
    
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found.")
        return
        
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    print("Mapping holidays to DATETIME index...")
    # df index is DATETIME
    # Check if the date part of the index is in holiday_dates
    df['is_bank_holiday'] = df.index.tz_localize(None).normalize().isin(holiday_dates).astype(float)
    
    # Save back
    print(f"Saving updated DataFrame to {parquet_path}...")
    df.to_parquet(parquet_path)
    print("Successfully added is_bank_holiday feature!")
    
    # Also update features.txt
    features_txt_path = "../datalake/clean+features/features.txt"
    if os.path.exists(features_txt_path):
        with open(features_txt_path, "r") as f:
            features = f.read().splitlines()
        if "is_bank_holiday" not in features:
            features.append("is_bank_holiday")
            with open(features_txt_path, "w") as f:
                f.write("\n".join(features) + "\n")
            print("Added is_bank_holiday to features.txt")

if __name__ == "__main__":
    fetch_and_merge_holidays()
