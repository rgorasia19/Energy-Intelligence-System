import os
import pandas as pd
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

async def fetch_warnings_chunk(session, start_date, end_date):
    url = f"https://data.elexon.co.uk/bmrs/api/v1/system/warnings?from={start_date}T00:00:00Z&to={end_date}T23:59:00Z"
    try:
        async with session.get(url, timeout=30) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                print(f"Error {response.status}: {await response.text()}")
    except Exception as e:
        print(f"Error fetching warnings: {e}")
    return []

async def fetch_warnings_async(start_date, end_date):
    print(f"Fetching system warnings from {start_date} to {end_date}...")
    # System warnings endpoint allows wider ranges, let's chunk by 30 days
    dates = pd.date_range(start=start_date, end=end_date, freq='30D')
    
    semaphore = asyncio.Semaphore(5)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(len(dates)):
            chunk_start = dates[i].strftime("%Y-%m-%d")
            if i + 1 < len(dates):
                chunk_end = (dates[i+1] - timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                chunk_end = end_date
            
            async def bound_fetch(s, e):
                async with semaphore:
                    return await fetch_warnings_chunk(session, s, e)
                    
            tasks.append(bound_fetch(chunk_start, chunk_end))
            
        results = await asyncio.gather(*tasks)
        
    all_warnings = []
    for r in results:
        all_warnings.extend(r)
        
    # Process warnings to create a daily stress alert flag
    df = pd.DataFrame(all_warnings)
    
    dates_df = pd.DataFrame({"DATETIME": pd.date_range(start=start_date, end=end_date, freq='D')})
    dates_df['stress_alert_flag'] = 0.0
    
    if not df.empty and 'publishTime' in df.columns:
        df['DATETIME'] = pd.to_datetime(df['publishTime']).dt.normalize().dt.tz_localize(None)
        
        # Look for Electricity Margin Notice (EMN) or Capacity Market Warning (CMW)
        # or Loss of Load (LOL) in the warningText
        df['is_stress'] = df['warningText'].str.contains('Margin Notice|Capacity Market Warning|Loss of Load', case=False, na=False)
        df['is_stress'] = df['is_stress'] | df['warningType'].str.contains('EMN|CMW', case=False, na=False)
        
        stress_days = df[df['is_stress']]['DATETIME'].unique()
        
        dates_df.loc[dates_df['DATETIME'].isin(stress_days), 'stress_alert_flag'] = 1.0
        
    return dates_df

def main():
    out_dir = '../datalake/raw_data'
    os.makedirs(out_dir, exist_ok=True)
    
    # Fast test run dates
    start_date = '2023-01-01'
    end_date = '2024-01-31'
    
    df_warn = asyncio.run(fetch_warnings_async(start_date, end_date))
    out_path = os.path.join(out_dir, 'system_stress_alerts.csv')
    df_warn.to_csv(out_path, index=False)
    print(f"Saved stress alerts to {out_path} ({df_warn['stress_alert_flag'].sum()} stress days found)")

if __name__ == "__main__":
    main()
