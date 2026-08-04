import os
import pandas as pd
import asyncio
import aiohttp
from datetime import datetime

async def fetch_date(session, date_str, semaphore):
    # Elexon API for Outturn Generation by Fuel Type
    # startTime and endTime for a single day
    start_time = f"{date_str}T00:00:00Z"
    end_time = f"{date_str}T23:59:00Z"
    url = f"https://data.elexon.co.uk/bmrs/api/v1/generation/outturn/summary?startTime={start_time}&endTime={end_time}"
    
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        records = data.get('data', [])
                        
                        # We want the average nuclear generation for the day
                        nuclear_gen = []
                        for r in records:
                            if r.get('fuelType') == 'NUCLEAR':
                                nuclear_gen.append(r.get('generation', 0))
                        
                        if nuclear_gen:
                            avg_nuclear = sum(nuclear_gen) / len(nuclear_gen)
                            return {"DATETIME": date_str, "nuclear_outturn": avg_nuclear}
                        else:
                            return {"DATETIME": date_str, "nuclear_outturn": 0}
            except Exception:
                pass
            await asyncio.sleep(2)
        return {"DATETIME": date_str, "nuclear_outturn": None}

async def fetch_nuclear_async(start_date, end_date):
    print(f"Fetching nuclear outturn data from {start_date} to {end_date}...")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    
    semaphore = asyncio.Semaphore(20)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_date(session, d, semaphore) for d in date_strs]
        results = await asyncio.gather(*tasks)
        
    df = pd.DataFrame([r for r in results if r])
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.sort_values('DATETIME').reset_index(drop=True)
    
    # Forward fill missing values
    df['nuclear_outturn'] = df['nuclear_outturn'].ffill().bfill()
    
    return df

def main():
    out_dir = '../datalake/raw_data'
    os.makedirs(out_dir, exist_ok=True)
    
    # Fast test run dates
    start_date = '2023-01-01'
    end_date = '2024-01-31'
    
    df_nuc = asyncio.run(fetch_nuclear_async(start_date, end_date))
    out_path = os.path.join(out_dir, 'nuclear_outturn.csv')
    df_nuc.to_csv(out_path, index=False)
    print(f"Saved nuclear outturn to {out_path} ({len(df_nuc)} days)")

if __name__ == "__main__":
    main()
