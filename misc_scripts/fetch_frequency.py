import os
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta

async def fetch_frequency_day(session, date_str, semaphore):
    # API allows max 2 days. We fetch 1 day.
    start_time = f"{date_str}T00:00:00Z"
    end_time = f"{date_str}T23:59:59Z"
    url = f"https://data.elexon.co.uk/bmrs/api/v1/system/frequency?from={start_time}&to={end_time}"
    
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list) and len(data) > 0:
                            freqs = [r['frequency'] for r in data if r.get('frequency') is not None]
                        elif 'data' in data:
                            freqs = [r['frequency'] for r in data['data'] if r.get('frequency') is not None]
                        else:
                            freqs = []
                            
                        if freqs:
                            freq_arr = np.array(freqs)
                            # Excursions outside +/- 0.1Hz (Statutory limit in UK is 49.5 to 50.5, but operational limit is 49.9 to 50.1)
                            excursions = np.sum((freq_arr < 49.9) | (freq_arr > 50.1))
                            excursion_flag = 1.0 if excursions > 0 else 0.0
                            
                            # p99 absolute deviation from 50.0 Hz
                            abs_dev = np.abs(freq_arr - 50.0)
                            p99_dev = np.percentile(abs_dev, 99)
                            
                            return {
                                "DATETIME": date_str,
                                "freq_excursion_flag": excursion_flag,
                                "freq_p99_dev": p99_dev
                            }
                    elif response.status == 404:
                        return {"DATETIME": date_str, "freq_excursion_flag": 0.0, "freq_p99_dev": 0.0}
            except Exception:
                pass
            await asyncio.sleep(2)
        return {"DATETIME": date_str, "freq_excursion_flag": None, "freq_p99_dev": None}

async def fetch_frequency_async(start_date, end_date):
    print(f"Fetching 15-second frequency data and computing daily tail-risks from {start_date} to {end_date}...")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    
    # 20 concurrent requests to keep Elexon happy while being fast
    semaphore = asyncio.Semaphore(20)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_frequency_day(session, d, semaphore) for d in date_strs]
        results = await asyncio.gather(*tasks)
        
    df = pd.DataFrame([r for r in results if r])
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.sort_values('DATETIME').reset_index(drop=True)
    
    # Forward fill missing values
    df['freq_excursion_flag'] = df['freq_excursion_flag'].fillna(0.0)
    df['freq_p99_dev'] = df['freq_p99_dev'].fillna(0.0)
    
    return df

def main():
    out_dir = '../datalake/raw_data'
    os.makedirs(out_dir, exist_ok=True)
    
    # Fast test run dates
    start_date = '2023-01-01'
    end_date = '2024-01-31'
    
    df_freq = asyncio.run(fetch_frequency_async(start_date, end_date))
    out_path = os.path.join(out_dir, 'system_frequency_stats.csv')
    df_freq.to_csv(out_path, index=False)
    print(f"Saved frequency stats to {out_path} ({len(df_freq)} days)")

if __name__ == "__main__":
    main()
