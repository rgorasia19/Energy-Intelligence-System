import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

bundle = joblib.load('../../v2/src/models/baseline_xgb_5.joblib')

model = bundle['model']
scaler = bundle['scaler']

test_df = pd.read_parquet('../../datalake/splits/v2/test_df.parquet')
with open ('../../datalake/clean+features/features.txt','r') as f:
  feature_list = f.read().splitlines()

feature_list.remove('TARGET')
feature_list.remove('DATETIME')

X_test = test_df[feature_list]
Y_test = test_df['TARGET']
X_test_scaled = scaler.transform(X_test)

test_predictions = model.predict(X_test_scaled)

plt.figure(figsize=(12,6))
plt.plot(test_df.index[:2000], Y_test.values[:2000],color = 'blue', alpha = 0.5, label = 'Actual Demand')
plt.plot(test_df.index[:2000], test_predictions[:2000],color='red', alpha = 0.5,label='Predicted Demand')
plt.legend()
plt.title('Actual vs Predicted Demand')
plt.savefig('actual_vs_pred_5.png',dpi=600)
plt.show()

print("MAE:", mean_absolute_error(Y_test, test_predictions))
print("RMSE:", np.sqrt(mean_squared_error(Y_test, test_predictions)))
print("R2:", r2_score(Y_test, test_predictions))