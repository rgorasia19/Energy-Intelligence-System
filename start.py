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


datapoint_labels = ["demand-data","generation-capacity-register","flexibility-forecasts","live-data"]
url_ids = pd.DataFrame()

for d in datapoint_labels:
    print(f"Checking endpoint: {d}")
    dataset = get("package_show",params={"id":d, })
    resources = dataset["result"]["resources"]
    
    for r in dataset["result"]["resources"]:
        if r["datastore_active"] and r["format"] == "CSV":
            df = pd.DataFrame({"label": d, "id": r["id"]})
            url_ids = pd.concat([url_ids, df])            
        print(
            r["id"],
            r["format"],
            r["datastore_active"],
            r["url"]
        )
    print("\n")
print(url_ids)
