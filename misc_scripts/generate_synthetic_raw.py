import os
import pandas as pd
import numpy as np

def generate_synthetic_data(start_date, end_date):
    out_dir = '../datalake/raw_data'
    os.makedirs(out_dir, exist_ok=True)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    # 1. Nuclear Outturn
    df_nuc = pd.DataFrame({
        'DATETIME': dates,
        'nuclear_outturn': np.random.normal(5000, 500, n).clip(0)
    })
    df_nuc.to_csv(os.path.join(out_dir, 'nuclear_outturn.csv'), index=False)
    
    # 2. System Stress Alerts
    df_warn = pd.DataFrame({
        'DATETIME': dates,
        'stress_alert_flag': np.random.choice([0.0, 1.0], size=n, p=[0.99, 0.01])
    })
    df_warn.to_csv(os.path.join(out_dir, 'system_stress_alerts.csv'), index=False)
    
    # 3. System Frequency Stats
    df_freq = pd.DataFrame({
        'DATETIME': dates,
        'freq_excursion_flag': np.random.choice([0.0, 1.0], size=n, p=[0.95, 0.05]),
        'freq_p99_dev': np.random.uniform(0.05, 0.15, n)
    })
    df_freq.to_csv(os.path.join(out_dir, 'system_frequency_stats.csv'), index=False)
    
    # 4. Wholesale Prices Advanced
    df_prices = pd.DataFrame({
        'DATETIME': dates,
        'day_ahead_price': np.random.normal(50, 20, n),
        'imbalance_volatility': np.random.uniform(10, 100, n),
        'negative_price_duration': np.random.poisson(1, n)
    })
    df_prices.to_csv(os.path.join(out_dir, 'wholesale_prices_advanced.csv'), index=False)
    
    print(f"Successfully generated synthetic raw data for {n} days.")

if __name__ == "__main__":
    generate_synthetic_data('2001-01-01', '2026-12-31')
