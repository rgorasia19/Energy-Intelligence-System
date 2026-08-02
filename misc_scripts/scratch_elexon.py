import requests
import json

def test_remit():
    url = "https://data.elexon.co.uk/bmrs/api/v1/remit/list"
    params = {
        "eventStart": "2024-01-01",
        "eventEnd": "2024-01-31",
        "format": "json"
    }
    
    print("Testing REMIT API:", url)
    r = requests.get(url, params=params)
    print("Status:", r.status_code)
    if r.status_code == 200:
        data = r.json()
        print(f"Records: {len(data.get('data', []))}")
        if data.get('data'):
            print(json.dumps(data['data'][0], indent=2))
    else:
        print(r.text)

if __name__ == "__main__":
    test_remit()
