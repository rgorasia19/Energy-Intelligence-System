import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

df = pd.read_csv('../datalake/raw_data/full_data.csv')

interconnect_df = df[['IFA_FLOW','IFA2_FLOW','BRITNED_FLOW','MOYLE_FLOW','EAST_WEST_FLOW','NEMO_FLOW','NSL_FLOW','ELECLINK_FLOW','VIKING_FLOW','GREENLINK_FLOW']]
interconnect_df.set_index(df['DATETIME'], inplace=True)

generation_df = df[["GAS","COAL","NUCLEAR","HYDRO","WIND","SOLAR","BIOMASS","OTHER"]]
generation_df.set_index(df['DATETIME'], inplace=True)

capacity_df = df[['EMBEDDED_WIND_GENERATION','EMBEDDED_WIND_CAPACITY','EMBEDDED_SOLAR_GENERATION','EMBEDDED_SOLAR_CAPACITY', 'NON_BM_STOR','PUMP_STORAGE_PUMPING','STORAGE','GENERATION']]
capacity_df.set_index(df['DATETIME'], inplace=True)

scaler_interconnect = StandardScaler()
scaler_generation = StandardScaler()
scaler_capacity = StandardScaler()

pca_interconnect = PCA(n_components=0.95)
pca_generation = PCA(n_components=0.95)
pca_capacity = PCA(n_components=0.95)

X_inter = scaler_interconnect.fit_transform(interconnect_df)
X_generation = scaler_generation.fit_transform(generation_df)
X_capacity = scaler_capacity.fit_transform(capacity_df)

inter_pcas = pca_interconnect.fit_transform(X_inter)
gen_pcas = pca_generation.fit_transform(X_generation)
cap_pcas = pca_capacity.fit_transform(X_capacity)

print(pca_interconnect.explained_variance_ratio_)
print(pca_generation.explained_variance_ratio_)
print(pca_capacity.explained_variance_ratio_)

inter_df = pd.DataFrame(inter_pcas, columns=[f"INTER_PC{i}" for i in range(inter_pcas.shape[1])])
gen_df = pd.DataFrame(gen_pcas, columns=[f"GEN_PC{i}" for i in range(gen_pcas.shape[1])])
cap_df = pd.DataFrame(cap_pcas, columns=[f"CAP_PC{i}" for i in range(cap_pcas.shape[1])])

df_final = pd.concat([inter_df, gen_df, cap_df],axis=1)
df_final.set_index(df['DATETIME'], inplace=True)

df_final.to_parquet('../datalake/pca/df_pca.parquet', index=True)