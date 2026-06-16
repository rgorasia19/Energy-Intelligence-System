import mlflow
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from feature_prep import TimeSeriesDataset
from torch.utils.data import DataLoader

# 1. Load the best model
# Point directly to the MLflow artifact directory, NOT the raw .pth file!
model_uri = 'mlruns/0/models/m-95e6cf39fd7447e9a28bcabc4ae2e293/artifacts'
model = mlflow.pytorch.load_model(model_uri)
model.eval()

# Load the target scaler saved during training
scaler_path = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/target_scaler.pkl")
with open(scaler_path, "rb") as f:
    target_scaler = pickle.load(f)

# 2. Load Test Data
# Adjust this path to point to your test set
test_df = pd.read_parquet('../../datalake/hmm_tensors/test.parquet')
feature_cols = [c for c in test_df.columns if c != 'TARGET_ND']
X_test = test_df[feature_cols].values
Y_test = test_df['TARGET_ND'].values

# Convert to PyTorch Tensors and move to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

seq_length = 48
test_dataset = TimeSeriesDataset(X_test, Y_test, seq_length=seq_length)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

test_predictions = []
actual_targets = []

# 3. Generate Predictions
with torch.no_grad():
    for x, y_label in test_loader:
        x = x.to(device)
        
        with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.bfloat16):
            log_alpha = model(x)
            
        # Get final state log probabilities at timestep T
        log_alpha_T = log_alpha[:, -1, :]
        
        # Find the most likely hidden state
        _, states = torch.max(log_alpha_T, dim=1)
        
        # Decode the predicted target demand using the state's y_means parameter
        pred_means = model.y_means[states]
        
        test_predictions.append(pred_means.cpu().numpy())
        actual_targets.append(y_label.numpy())

# 4. Evaluate Metrics
Y_test_flat = np.concatenate(actual_targets)
test_pred_scaled = np.concatenate(test_predictions)

# Inverse transform the scaled predictions back to actual MW demand!
test_pred_flat = target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()

mae = mean_absolute_error(Y_test_flat, test_pred_flat)
rmse = np.sqrt(mean_squared_error(Y_test_flat, test_pred_flat))
r2 = r2_score(Y_test_flat, test_pred_flat)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# 5. Plot Results (First N samples)
plt.figure(figsize=(12, 6))
plt.plot(Y_test_flat[:2000], label='Actual Demand', color='blue', alpha=0.5)
plt.plot(test_pred_flat[:2000], label='Predicted Demand', color='red', alpha=0.5)
plt.title('Actual vs Predicted Demand (First 2000 samples)')
plt.xlabel('Time Step')
plt.ylabel('Demand')
plt.legend()
plt.savefig('eval_plot.png', dpi=600)
plt.show()