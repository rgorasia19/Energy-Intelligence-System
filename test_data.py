import pandas as pd

# Load the filtered datasets
try:
    df = pd.read_csv("filtered_datasets.csv")
    print(f"Loaded {len(df)} datasets from filtered_datasets.csv")
    
    # Check if 'url' column exists, otherwise inform the user
    if 'url' not in df.columns:
        print("Error: The 'url' column is missing from filtered_datasets.csv.")
        print("Please close filtered_datasets.csv (if open in Excel/another app) and run 'py start.py' again to regenerate it with URLs.")
    else:
        # Grab the first URL to test
        test_url = df.iloc[0]["url"]
        print(f"\nFetching data from: {test_url}")

        # Read the CSV directly into a pandas dataframe
        data = pd.read_csv(test_url)

        # Inspect the data
        print("\n--- Data Preview ---")
        print(data.head())
        print("\n--- Data Info ---")
        print(data.info())

except FileNotFoundError:
    print("filtered_datasets.csv not found. Please run 'py start.py' to generate it.")
except PermissionError:
    print("Permission denied when reading filtered_datasets.csv. Please close it if it's open elsewhere.")
except Exception as e:
    print(f"An error occurred: {e}")
