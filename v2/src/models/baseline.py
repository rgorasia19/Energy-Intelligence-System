import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, RocCurveDisplay
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

model = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, verbose = 1)
model.fit(X_train_scaled, Y_train)

joblib.dump({"model": model,
            "scaler": scaler},
            "baseline_rf.joblib")
print("Model trained and saved to baseline_rf.joblib")

val_df = pd.read_parquet('../../../datalake/splits/v2/val_df.parquet')
X_val = val_df[feature_list]
Y_val = val_df['ND']
X_val_scaled = scaler.transform(X_val)

val_predictions = model.predict(X_val_scaled)

nb_disp = RocCurveDisplay.from_estimator(model,X_val_scaled,Y_val)
plt.savefig("ROC_1.png", dpi = 600)
plt.show()
