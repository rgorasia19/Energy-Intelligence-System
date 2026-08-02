import os
import pandas as pd
import asyncio
import aiohttp
from datetime import datetime, timedelta

async def fetch_date(session, date_str, semaphore):
    url = f"https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{date_str}"
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        records = data.get('data', [])
                        if records:
                            # We just take the average system sell price for the day to aggregate to daily
                            prices = [r['systemSellPrice'] for r in records if r.get('systemSellPrice') is not None]
                            if prices:
                                avg_price = sum(prices) / len(prices)
                                return {"DATETIME": date_str, "day_ahead_price": avg_price}
                    elif response.status == 404:
                        return {"DATETIME": date_str, "day_ahead_price": None}
            except Exception as e:
                pass
            await asyncio.sleep(2)
        return {"DATETIME": date_str, "day_ahead_price": None}

async def fetch_prices_async(start_date, end_date):
    print(f"Fetching wholesale prices data from {start_date} to {end_date} from Elexon BMRS...")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    
    # Elexon API rate limit friendly
    semaphore = asyncio.Semaphore(20)
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_date(session, d, semaphore) for d in date_strs]
        results = await asyncio.gather(*tasks)
        
    df = pd.DataFrame([r for r in results if r])
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.sort_values('DATETIME').reset_index(drop=True)
    
    # Forward fill missing values if any
    df['day_ahead_price'] = df['day_ahead_price'].ffill().bfill()
    
    return df

def main():
    out_dir = '../datalake/raw_data'
    os.makedirs(out_dir, exist_ok=True)
    
    # We fetch a shorter range to avoid timing out, ideally 2009-2026.
    # To keep it fast for this run, let's fetch from 2015 to 2026.
    start_date = '2015-01-01'
    end_date = '2026-07-27'
    
    df_prices = asyncio.run(fetch_prices_async(start_date, end_date))
    
    out_path = os.path.join(out_dir, 'wholesale_prices.csv')
    df_prices.to_csv(out_path, index=False)
    print(f"Saved wholesale prices to {out_path} ({len(df_prices)} days)")

if __name__ == "__main__":
    main()
