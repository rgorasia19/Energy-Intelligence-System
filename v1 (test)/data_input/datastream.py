import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class DataStream():
    def __init__(self):
        self.base_url = "https://connecteddata.nationalgrid.co.uk/api/3/action/"
        self.root = Path("data_lake")
        self.data_dir = self.root / "datasets"
        self.processed_file = self.root / "processed_ids.txt"

        self.root.mkdir(exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

        self.retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=self.retries))



    def api(self, endpoint, params=None):
        try:
            r = self.session.get(self.base_url + endpoint, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[API FAIL] {endpoint}: {e}")
            return None


    def load_processed(self):
        if not self.processed_file.exists():
            return set()
        return set(self.processed_file.read_text().splitlines())


    def mark_processed(self, dataset_id):
        with open(self.processed_file, "a") as f:
            f.write(dataset_id + "\n")




    def get_all_datasets(self,limit=50):
        start = 0
        all_data = []

        while True:
            res = self.api("package_search", {"rows": limit, "start": start})

            if not res or not res.get("success"):
                print(f"[WARN] failed batch start={start}")
                time.sleep(2)
                continue

            batch = res["result"]["results"]
            if not batch:
                break

            all_data.extend(batch)

            print(f"[OK] fetched {len(batch)} datasets at start={start}")
            start += limit
            time.sleep(0.2)

        return all_data



    def get_resources(self, dataset_id):
        res = self.api("package_show", {"id": dataset_id})
        if not res or not res.get("success"):
            return []
        return res["result"].get("resources", [])


    def fetch_file(self, url, out_path):
        try:
            if url.endswith(".csv"):
                df = pd.read_csv(url)
            elif url.endswith(".xlsx") or url.endswith(".xls"):
                df = pd.read_excel(url)
            else:
                return False

            df.to_parquet(out_path, index=False)
            return True

        except Exception as e:
            print(f"[FILE FAIL] {url}: {e}")
            return False




    def fetch_datastore(self, resource_id):
        res = self.api("datastore_search", {
            "resource_id": resource_id,
            "limit": 5000
        })

        if not res or not res.get("success"):
            return None

        return res["result"].get("records")



    def process_dataset(self, dataset):
        dataset_id = dataset["id"]
        ds_dir = self.data_dir / dataset_id
        ds_dir.mkdir(exist_ok=True)

        # save metadata
        with open(ds_dir / "metadata.json", "w") as f:
            json.dump(dataset, f, indent=2)

        resources = get_resources(dataset_id)

        with open(ds_dir / "resources.json", "w") as f:
            json.dump(resources, f, indent=2)

        for i, r in enumerate(resources):
            if not r.get("url"):
                continue

            fmt = (r.get("format") or "").lower()
            url = r["url"]

            # FILE DATA
            if fmt in {"csv", "xlsx", "xls"}:
                out_file = ds_dir / f"data_{i}.parquet"
                if not out_file.exists():
                    success = fetch_file(url, out_file)
                    print(f"[FILE] {dataset_id} -> {success}")

            # API DATASTORE
            elif r.get("datastore_active") and fmt == "csv":
                data = fetch_datastore(r["id"])
                if data:
                    out_file = ds_dir / f"datastore_{i}.json"
                    with open(out_file, "w") as f:
                        json.dump(data, f)

        self.mark_processed(dataset_id)




    def run(self, max_datasets=100):
        processed = self.load_processed()
        datasets = self.get_all_datasets()

        print(f"[INFO] total datasets: {len(datasets)}")
        print(f"[INFO] already processed: {len(processed)}")

        count = 0

        for d in datasets:
            if count >= max_datasets:
                break

            dataset_id = d.get("id")
            if not dataset_id or dataset_id in processed:
                continue

            print(f"\n[RUNNING] {dataset_id}")

            try:
                process_dataset(d)
                count += 1

            except Exception as e:
                print(f"[ERROR] dataset {dataset_id}: {e}")
                continue

            time.sleep(0.2)

        print(f"\nDONE. processed {count} new datasets")


if __name__ == "__main__":
    ds = DataStream()
    ds.run(max_datasets=200)