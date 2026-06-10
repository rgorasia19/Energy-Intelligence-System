import pandas as pd
import numpy as np

#1. Import dataset from CSV
df = pd.read_csv('../../datalake/clean+features/processed_data.csv')
#2. Convert DATETIME to datetime objects
df['DATETIME'] = pd.to_datetime(df['DATETIME'])
feature_list = ['ND','DATETIME']

#3. TEMPORAL FEATURES
df['YEAR'] = df['DATETIME'].dt.year
df['MONTH'] = df['DATETIME'].dt.month
df['DAY'] = df['DATETIME'].dt.day
df['HOUR'] = df['DATETIME'].dt.hour
df['MINUTE'] = df['DATETIME'].dt.minute
df['DAY_OF_WEEK'] = df['DATETIME'].dt.dayofweek
df['WEEK_OF_YEAR'] = df['DATETIME'].dt.isocalendar().week.astype(int)
df['QUARTER'] = df['DATETIME'].dt.quarter
df['DAY_OF_WEEK_SIN'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
df['DAY_OF_WEEK_COS'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)
df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].apply(lambda x: 1 if x >= 5 else 0)

feature_list.extend(['YEAR','MONTH','DAY','HOUR','MINUTE','DAY_OF_WEEK','WEEK_OF_YEAR','QUARTER','DAY_OF_WEEK_SIN','DAY_OF_WEEK_COS','HOUR_SIN','HOUR_COS','MONTH_SIN','MONTH_COS','IS_WEEKEND'])
#4. CORE LOAD FEATURES
df['ENGLAND_WALES_DEMAND_SHARE'] = df['ENGLAND_WALES_DEMAND'] / df['ND']
df['TSD_SHARE'] = df['TSD'] / df['ND']
feature_list.extend(['ENGLAND_WALES_DEMAND_SHARE', 'TSD_SHARE'])

for col in ['TSD_SHARE', 'ND', 'ENGLAND_WALES_DEMAND_SHARE']:
  for lag in [1,2,3,48,336]:
    df[f'{col}_LAG_{lag}'] = df[col].shift(lag)
    feature_list.append(f'{col}_LAG_{lag}')
  df[f'{col}_MEAN_48'] = df[col].rolling(window=48).mean()
  feature_list.append(f'{col}_MEAN_48')
  df[f'{col}_STD_48'] = df[col].rolling(window=48).std()
  feature_list.append(f'{col}_STD_48')
  df[f'{col}_RAMP_48'] = df[col] - df[col].shift(48)
  feature_list.append(f'{col}_RAMP_48')
  df[f'{col}_ACCELERATION_48'] = df[f'{col}_RAMP_48'] - df[f'{col}_RAMP_48'].shift(48)
  feature_list.append(f'{col}_ACCELERATION_48')
    
#5. SUPPLY COMPOSITION
for col in ['GAS','COAL','NUCLEAR','HYDRO','WIND','SOLAR','BIOMASS']:
  df[f'{col}_SHARE'] = df[col] / (df['GENERATION'] + 1)
  feature_list.append(f'{col}_SHARE')

#6. FLEXIBILITY AND BALANCING
df["NET_STORAGE"] = df["STORAGE"] - df["PUMP_STORAGE_PUMPING"]
df["INTERCONNECTOR_NET"] = df[["IFA_FLOW","IFA2_FLOW","BRITNED_FLOW","MOYLE_FLOW","EAST_WEST_FLOW",
  "NEMO_FLOW","NSL_FLOW","ELECLINK_FLOW","VIKING_FLOW","GREENLINK_FLOW"]].sum(axis=1)
df["IMPORT_SHARE"] = df["IMPORTS"] / (df['GENERATION'] + 1)
df["SUPPLY_DEMAND_GAP"] = df["GENERATION"] + df["IMPORTS"] - df["ND"]
df = df.copy()

feature_list.extend(['NET_STORAGE', 'INTERCONNECTOR_NET', 'IMPORT_SHARE', 'SUPPLY_DEMAND_GAP'])
#7. CARBON FEATURES
for lag in [1,2,3,48,336]:
  df[f"CARBON_INTENSITY_LAG_{lag}"] = df["CARBON_INTENSITY"].shift(lag)
  feature_list.append(f"CARBON_INTENSITY_LAG_{lag}")
df["CARBON_ROLL_MEAN_48"] = df["CARBON_INTENSITY"].rolling(window=48).mean()
df["CARBON_PER_DEMAND"] = df["CARBON_INTENSITY"] / (df["ND"] + 1)
df["RENEWABLES_VS_CARBON"] = (df["RENEWABLE"] / (df["GENERATION"] + 1)) * df["CARBON_INTENSITY"]
feature_list.extend(['CARBON_ROLL_MEAN_48','CARBON_PER_DEMAND','RENEWABLES_VS_CARBON'])
#8. CONTEXTUALS

df["WIND_VS_RAMP"] = df["WIND_SHARE"] * df["ND_RAMP_48"]
df["GAS_VS_RAMP"] = df["GAS_SHARE"] * df["ND_RAMP_48"]
df["INTERCONNECTOR_VS_DEMAND"] = df["INTERCONNECTOR_NET"] * df["ND"]
df["STORAGE_VS_RAMP"] = df["STORAGE"] * df["ND_RAMP_48"]
df["GENERATION_VS_DEMAND"] = df["GENERATION"] * df["ND"]
feature_list.extend(['WIND_VS_RAMP','GAS_VS_RAMP','INTERCONNECTOR_VS_DEMAND','STORAGE_VS_RAMP','GENERATION_VS_DEMAND'])

#9. CREATE A .TXT FILE CONTAINING FEATURES AND WRITE A SEPARATE DATAFRAME

feature_df = pd.DataFrame()
with open('../../datalake/clean+features/features.txt', 'w') as f:
  for feature in feature_list:
    f.write(feature + '\n')
    feature_df[feature] = df[feature]

feature_df.dropna(inplace=True)
feature_df.set_index('DATETIME', inplace=True)

float_cols = feature_df.select_dtypes(include=['float64']).columns
feature_df[float_cols] = feature_df[float_cols].astype('float32')

feature_df.to_parquet('../../datalake/clean+features/feature_df.parquet', index=True)