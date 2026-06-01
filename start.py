import requests
import pandas as pd

base_url = "https://connecteddata.nationalgrid.co.uk/api/3/action/"
def get(endpoint, params=None, headers=None):
    headers = {}

    #if api_key:
    #    headers["Authorization"] = api_key
    
    response = requests.get(base_url + endpoint, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


search_query = "demand OR generation OR capacity"
print(f"Searching for datasets matching: {search_query}")

# Use package_search to find datasets matching the query
response = get("package_search", params={"q": search_query, "rows": 100})
datasets = response["result"]["results"]

url_ids = pd.DataFrame()

for dataset in datasets:
    dataset_name = dataset.get("name", "unknown")
    print(f"Checking dataset: {dataset_name}")
    
    resources = dataset.get("resources", [])
    
    for r in resources:
        if r.get("datastore_active") and r.get("format", "").upper() == "CSV" and r.get("url")!="redacted" and dataset_name=="substation-loading":
            org = dataset.get("groups") or {}
            org_desc = org[0].get("description", "") if isinstance(org, list) else ""
            url = r.get("url", "")
            df = pd.DataFrame({"label": dataset_name, "id": [r["id"]], "details": [org_desc], "url": [url]})
            url_ids = pd.concat([url_ids, df], ignore_index=True)            
        # print(
        #     r.get("id"),
        #     r.get("format"),
        #     r.get("datastore_active"),
        #     r.get("url")
        # )
    print("\n")
print(url_ids)
url_ids.to_csv("filtered_datasets.csv", index=False)
print("Saved results to filtered_datasets.csv")
