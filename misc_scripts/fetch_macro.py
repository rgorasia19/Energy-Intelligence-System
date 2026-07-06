import os
import pandas as pd
import urllib.request
import ssl
import io

import json

def fetch_ons_json(url, col_name):
    print(f"Fetching ONS data from {url}...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req, context=ctx)
    data = json.loads(response.read().decode('utf-8'))
    
    # Extract the 'months' array
    months_data = data.get('months', [])
    records = []
    for item in months_data:
        records.append({
            'Date': item['date'],
            col_name: item['value']
        })
        
    df = pd.DataFrame(records)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y %b')
    df = df.set_index('Date')
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    return df

def fetch_boe_rate():
    print("Fetching BoE Official Bank Rate...")
    url = 'https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp'
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req, context=ctx)
    html_content = response.read().decode('utf-8')
    df = pd.read_html(io.StringIO(html_content))[0]
    # Columns typically: 'Date Changed', 'Rate'
    df.columns = ['Date', 'bank_rate']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df['bank_rate'] = pd.to_numeric(df['bank_rate'], errors='coerce')
    return df

def main():
    # 1. Fetch Data
    # CPI (All Items) - D7BT
    cpi_url = "https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7bt/data"
    df_cpi = fetch_ons_json(cpi_url, 'uk_cpi')
    
    # Monthly GDP (Index) - ECY2
    gdp_url = "https://www.ons.gov.uk/economy/grossdomesticproductgdp/timeseries/ecy2/data"
    df_gdp = fetch_ons_json(gdp_url, 'uk_gdp_index')
    
    # BoE Rate
    df_boe = fetch_boe_rate()
    
    # 2. Merge all at their native frequencies
    df_macro = df_cpi.join(df_gdp, how='outer').join(df_boe, how='outer')
    df_macro = df_macro.sort_index()
    
    # 3. Create a daily date range from 2001-01-01 to present
    start_date = '2001-01-01'
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    daily_idx = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 4. Reindex to daily and interpolate
    # The user specifically requested linear interpolation instead of ffill
    print("Aligning to daily frequency with linear interpolation...")
    df_daily = df_macro.reindex(daily_idx)
    
    # BoE rate is a step function by nature, linear interpolation between steps doesn't make financial sense, 
    # BUT the user requested linear interpolation explicitly. 
    # To be safe, we'll linearly interpolate everything.
    df_daily = df_daily.interpolate(method='linear')
    
    # There may be NaNs at the start before the first value. We bfill them.
    df_daily = df_daily.bfill()
    
    df_daily.index.name = 'DATETIME'
    
    out_dir = '../datalake/raw_data'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'macro_data.csv')
    df_daily.to_csv(out_path)
    print(f"Saved aligned macro data to {out_path}")

if __name__ == "__main__":
    main()
