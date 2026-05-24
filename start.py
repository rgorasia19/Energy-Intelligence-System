import csv
import io
import urllib.request
from ckanapi import RemoteCKAN
from ckanapi.errors import NotAuthorized, NotFound

# 1. Initialize the client pointing to the National Grid data portal
portal = RemoteCKAN('https://connecteddata.nationalgrid.co.uk')

# 2. Get the dataset/package metadata
dataset_id = 'demand-data'
print(f"Fetching dataset info for '{dataset_id}'...")
package = portal.action.package_show(id=dataset_id)

# 3. Retrieve the first resource
resource = package['resources'][0]
resource_id = resource['id']
resource_name = resource['name']
datastore_active = resource.get('datastore_active', False)

print(f"Dataset Title:  {package['title']}")
print(f"Resource Name:  {resource_name}")
print(f"Resource ID:    {resource_id}")
print(f"Datastore Active: {datastore_active}\n")

# 4. Check if the resource is in the active datastore or is a direct file download
if datastore_active:
    print("Using CKAN Datastore API to fetch records...")
    try:
        result = portal.action.datastore_search(
            resource_id=resource_id,
            limit=100
        )
        records = result['records']
    except NotAuthorized:
        print("This is a restricted dataset")
        print("May require your API key")
        records = []
    except NotFound:
        print("Resource not found in the Datastore.")
        records = []
else:
    print("Direct file upload.")
    print(f"Downloading file from: {resource['url']}...")
    try:
        req = urllib.request.Request(resource['url'], headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            csv_content = response.read().decode('utf-8')
            csv_file = io.StringIO(csv_content)
            reader = csv.DictReader(csv_file)
            records = list(reader)
    except Exception as e:
        print(f"Failed to download or parse CSV file: {e}")
        records = []

# 5. Output the results
if records:
    print(f"\nSuccessfully retrieved {len(records)} records. Preview (first 3):")
    for i, record in enumerate(records[:3], 1):
        print(f"Record {i}: {record}")
else:
    print("\nNo records retrieved.")

