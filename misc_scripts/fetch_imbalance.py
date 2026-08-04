import os
import pandas as pd
import asyncio
import aiohttp
from datetime import datetime

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
                            prices = [r['systemSellPrice'] for r in records if r.get('systemSellPrice') is not None]
                            if prices:
                                avg_price = sum(prices) / len(prices)
                                # Volatility: standard deviation of half-hourly prices
                                mean = avg_price
                                variance = sum((p - mean) ** 2 for p in prices) / len(prices)
                                vol = variance ** 0.5
                                # Duration of negative prices (count of half-hours < 0)
                                neg_count = sum(1 for p in prices if p < 0)
                                
                                return {
                                    "DATETIME": date_str, 
                                    "day_ahead_price": avg_price,
                                    "imbalance_volatility": vol,
                                    "negative_price_duration": neg_count
                                }
                    elif response.status == 404:
                        return {"DATETIME": date_str, "day_ahead_price": None, "imbalance_volatility": None, "negative_price_duration": 0}
            except Exception:
                pass
            await asyncio.sleep(2)
        return {"DATETIME": date_str, "day_ahead_price": None, "imbalance_volatility": None, "negative_price_duration": 0}

async def fetch_prices_async(start_date, end_date):
    print(f"Fetching wholesale/imbalance prices data from {start_date} to {end_date}...")
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
    df['day_ahead_price'] = df['day_ahead_price'].ffill().bfill()
    df['imbalance_volatility'] = df['imbalance_volatility'].fillna(0)
    df['negative_price_duration'] = df['negative_price_duration'].fillna(0)
    
    return df

def main():
    out_dir = '../datalake/raw_data'
    os.makedirs(out_dir, exist_ok=True)
    
    # Fast test run dates (e.g. 2023-2024). Change back for full history
    start_date = '2023-01-01'
    end_date = '2024-01-31'
    
    df_prices = asyncio.run(fetch_prices_async(start_date, end_date))
    out_path = os.path.join(out_dir, 'wholesale_prices_advanced.csv')
    df_prices.to_csv(out_path, index=False)
    print(f"Saved advanced prices to {out_path} ({len(df_prices)} days)")

if __name__ == "__main__":
    main()
