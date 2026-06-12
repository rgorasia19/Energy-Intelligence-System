import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

train_df = pd.read_parquet('../../../datalake/splits/v2/train_df.parquet')
with open ('../../../datalake/clean+features/features.txt','r') as f:
  feature_list = f.read().splitlines()

feature_list.remove('ND')
feature_list.remove('DATETIME')

X_train = train_df[feature_list]
Y_train = train_df['ND']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled.shape)
print(Y_train.shape)

val_df = pd.read_parquet('../../../datalake/splits/v2/val_df.parquet')
X_val = val_df[feature_list]
Y_val = val_df['ND']
X_val_scaled = scaler.transform(X_val)

model = XGBRegressor(n_estimators = 2000,
                     learning_rate = 0.025,
                     max_depth = 10,
                     n_jobs = -1,
                     tree_method="hist",
                     early_stopping_rounds=50,
                     device="cuda",
                     verbosity=1)

model.fit(X_train_scaled, Y_train,
          eval_set=[(X_train_scaled, Y_train), (X_val_scaled, Y_val)],
          verbose=100)

joblib.dump({"model": model,
            "scaler": scaler,
            "training": train_df,
            "validation": val_df,},
            "baseline_xgb_4.joblib",
            compress=('lz4',3))
print("Model trained and saved to baseline_xgb_4.joblib")

val_predictions = model.predict(X_val_scaled)
print("MAE:", mean_absolute_error(Y_val, val_predictions))
print("RMSE:", np.sqrt(mean_squared_error(Y_val, val_predictions)))
print("R2:", r2_score(Y_val, val_predictions))
