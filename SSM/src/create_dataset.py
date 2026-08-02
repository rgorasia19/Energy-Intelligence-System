import os
import pandas as pd
import joblib
from feature_prep import standardise_timeseries, merge_features, engineer_features, fit_scaler, transform_data, time_split

def main():
    raw_dir = '../../datalake/raw_data'
    out_dir = '../../datalake/ssm_tensors'
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data
    # Use merged_demanddata and generation_mix directly to preserve the outer join logic
    # (full_data.csv was inner joined, which drops pre-2009 data)
    print("Loading raw data...")
    df_demand = pd.read_csv(os.path.join(raw_dir, 'merged_demanddata.csv'))
    df_gen = pd.read_csv(os.path.join(raw_dir, 'generation_mix.csv'))
    df_weather = pd.read_csv(os.path.join(raw_dir, 'weather_data.csv'))
    df_macro = pd.read_csv(os.path.join(raw_dir, 'macro_data.csv'))
    
    if os.path.exists(os.path.join(raw_dir, 'wholesale_prices.csv')):
        df_prices = pd.read_csv(os.path.join(raw_dir, 'wholesale_prices.csv'))
    else:
        df_prices = pd.DataFrame(columns=['DATETIME', 'day_ahead_price'])
        
    if os.path.exists(os.path.join(raw_dir, 'remit_capacity.csv')):
        df_remit = pd.read_csv(os.path.join(raw_dir, 'remit_capacity.csv'))
    else:
        df_remit = pd.DataFrame(columns=['DATETIME', 'nuclear_available_capacity', 'gas_available_capacity', 'coal_available_capacity'])
    
    # Define which columns are continuous vs flow
    demand_flow_cols = ['ND', 'TSD', 'ENGLAND_WALES_DEMAND']
    demand_cont_cols = ['EMBEDDED_WIND_CAPACITY', 'EMBEDDED_SOLAR_CAPACITY']
    gen_flow_cols = ['GAS', 'COAL', 'NUCLEAR', 'WIND', 'SOLAR', 'IMPORTS', 'GENERATION']
    weather_cont_cols = ['temperature_2m', 'cloudcover', 'windspeed_10m', 'shortwave_radiation']
    macro_cont_cols = ['uk_cpi', 'uk_gdp_index', 'bank_rate']
    price_cont_cols = ['day_ahead_price']
    remit_cont_cols = ['nuclear_available_capacity', 'gas_available_capacity', 'coal_available_capacity']
    
    print("Standardising timeseries...")
    df_demand_daily = standardise_timeseries(df_demand, 'DATETIME', continuous_cols=demand_cont_cols, flow_cols=demand_flow_cols)
    df_gen_daily = standardise_timeseries(df_gen, 'DATETIME', continuous_cols=[], flow_cols=gen_flow_cols)
    df_weather_daily = standardise_timeseries(df_weather, 'DATETIME', continuous_cols=weather_cont_cols, flow_cols=[])
    df_macro_daily = standardise_timeseries(df_macro, 'DATETIME', continuous_cols=macro_cont_cols, flow_cols=[])
    
    if not df_prices.empty:
        df_prices_daily = standardise_timeseries(df_prices, 'DATETIME', continuous_cols=price_cont_cols, flow_cols=[])
    else:
        df_prices_daily = pd.DataFrame()
        
    if not df_remit.empty:
        df_remit_daily = standardise_timeseries(df_remit, 'DATETIME', continuous_cols=remit_cont_cols, flow_cols=[])
    else:
        df_remit_daily = pd.DataFrame()
    
    # 2. Merge
    print("Merging features...")
    dfs = {
        'DEMAND_': df_demand_daily,
        'GEN_': df_gen_daily,
        'WEATHER_': df_weather_daily,
        'MACRO_': df_macro_daily
    }
    if not df_prices_daily.empty:
        dfs['PRICE_'] = df_prices_daily
    if not df_remit_daily.empty:
        dfs['REMIT_'] = df_remit_daily
    # This will outer join and fill missingness masks, starting from 2001-01-01
    merged = merge_features(dfs, start_date='2001-01-01')
    
    # 3. Engineer
    print("Engineering features...")
    engineered = engineer_features(merged)
    
    # 4. Split
    print("Splitting data...")
    train, val, test = time_split(engineered, '2016-12-31', '2019-12-31')
    # 5. Scale
    print("Scaling data...")
    feature_cols = list(engineered.columns)
    scaler = fit_scaler(train, feature_cols)
    
    train_scaled = transform_data(train, scaler, feature_cols)
    val_scaled = transform_data(val, scaler, feature_cols)
    test_scaled = transform_data(test, scaler, feature_cols)
    
    # 6. Save
    print("Saving to parquet...")
    train_scaled.to_parquet(os.path.join(out_dir, 'train.parquet'))
    val_scaled.to_parquet(os.path.join(out_dir, 'val.parquet'))
    test_scaled.to_parquet(os.path.join(out_dir, 'test.parquet'))
    
    # Save artifacts needed for inference
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.pkl'))
    
    # Target columns
    target_cols = demand_flow_cols + gen_flow_cols
    joblib.dump({
        'feature_columns': feature_cols,
        'target_columns': target_cols
    }, os.path.join(out_dir, 'columns.pkl'))
    
    print("Done!")

if __name__ == "__main__":
    main()
