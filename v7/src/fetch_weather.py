import os
import requests
import pandas as pd
import numpy as np

def fetch_weather_data(start_date, end_date, lat=51.5074, lon=-0.1278):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,cloudcover,windspeed_10m,shortwave_radiation",
        "timezone": "Europe/London"
    }
    
    print(f"Fetching weather data from {start_date} to {end_date}...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Parse hourly data
    hourly = data['hourly']
    
    df = pd.DataFrame({
        "DATETIME": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
        "cloudcover": hourly["cloudcover"],
        "windspeed_10m": hourly["windspeed_10m"],
        "shortwave_radiation": hourly["shortwave_radiation"]
    })
    
    if df["DATETIME"].dt.tz is not None:
        df["DATETIME"] = df["DATETIME"].dt.tz_convert(None)
    
    df.set_index("DATETIME", inplace=True)
    
    print("Interpolating hourly data to half-hourly to match NESO settlement periods...")
    # Resample to 30 min and linearly interpolate
    # Because Open-Meteo provides hourly at the start of the hour (e.g. 00:00),
    # 30min resampling will insert NaNs at xx:30 which we interpolate.
    df_half_hourly = df.resample('30min').interpolate(method='linear')
    
    df_half_hourly.reset_index(inplace=True)
    
    return df_half_hourly

if __name__ == "__main__":
    start_date = "2016-01-01"
    end_date = "2026-05-15"
    out_dir = "../../datalake/raw_data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "weather_data.csv")
    
    df_weather = fetch_weather_data(start_date, end_date)
    
    print(f"Saving {len(df_weather)} half-hourly records to {out_path}")
    df_weather.to_csv(out_path, index=False)
    print("Done!")
