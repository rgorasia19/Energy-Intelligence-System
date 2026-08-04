import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

async def fetch_endpoint(session, name, url):
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                print(f"--- {name} (200 OK) ---")
                if 'data' in data and len(data['data']) > 0:
                    print(json.dumps(data['data'][0], indent=2))
                elif isinstance(data, list) and len(data) > 0:
                    print(json.dumps(data[0], indent=2))
                else:
                    print(f"Empty data: {data}")
            else:
                print(f"--- {name} ({response.status}) ---")
                print(await response.text())
    except Exception as e:
        print(f"--- {name} (ERROR) ---")
        print(e)

async def main():
    async with aiohttp.ClientSession() as session:
        # Nuclear Plant Status -> REMIT? or generation?
        await fetch_endpoint(session, "Generation By Fuel Type (Nuclear)", "https://data.elexon.co.uk/bmrs/api/v1/generation/outturn/summary?startTime=2024-01-01T00:00:00Z&endTime=2024-01-01T23:59:00Z")
        
        # Frequency Data
        await fetch_endpoint(session, "System Frequency", "https://data.elexon.co.uk/bmrs/api/v1/system/frequency?from=2024-01-01T00:00:00Z&to=2024-01-01T01:00:00Z")
        
        # Curtailment (BOA / Bid Offer Acceptances)
        await fetch_endpoint(session, "BOAs (Curtailment)", "https://data.elexon.co.uk/bmrs/api/v1/balancing/acceptances?settlementDate=2024-01-01&settlementPeriod=1")
        
        # Stress alerts (Loss of Load, EMN, CMW)
        await fetch_endpoint(session, "Loss of Load Probability", "https://data.elexon.co.uk/bmrs/api/v1/balancing/dynamic/loss-of-load?settlementDate=2024-01-01")
        await fetch_endpoint(session, "System Warnings", "https://data.elexon.co.uk/bmrs/api/v1/system/warnings?from=2023-01-01T00:00:00Z&to=2024-12-31T23:59:00Z")

if __name__ == "__main__":
    asyncio.run(main())
