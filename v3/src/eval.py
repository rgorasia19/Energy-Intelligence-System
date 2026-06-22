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
model_uri = 'mlruns_new/0/models/m-75c9fcd2970041af823acdf7aac01182/artifacts'
model = mlflow.pytorch.load_model(model_uri, map_location=torch.device('cpu'))

# Extract the original module if it was saved as a torch.compile wrapper!
model = getattr(model, '_orig_mod', model)
model.eval()

# Load the target scaler saved during training
# Since you pulled it down via git, it's sitting right in your current directory!
with open("target_scaler.pkl", "rb") as f:
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
# Scale targets so evaluation dataset matches training space exactly
Y_test_scaled = target_scaler.transform(Y_test.reshape(-1, 1)).flatten()
test_dataset = TimeSeriesDataset(X_test, Y_test_scaled, seq_length=seq_length)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

test_predictions = []
actual_targets = []
all_states = []
all_entropies = []

# 3. Generate Predictions
with torch.no_grad():
    for x, y_label in test_loader:
        x = x.to(device)
        
        # Remove AMP for evaluation correctness
        log_alpha, gamma = model(x)
            
        # True smoothed posterior at timestep T (P(z_T | x_1:T))
        probs = gamma[:, -1, :]
        
        # Probabilistic Decoding: Expected value over all states E[y | x]
        pred_means = (probs * model.y_means).sum(dim=1)
        
        if len(test_predictions) == 0:  # Print for the very first batch
            print("--- Sanity Checks (Scaled Space) ---")
            print("Pred mean:", pred_means.mean().item())
            print("Target mean:", y_label.mean().item())
            print("------------------------------------")
            
        # --- Diagnostics ---
        # 1. State Occupancy (soft expectation, no argmax!)
        # 2. Posterior Entropy: -sum(p * log(p))
        entropies = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        
        test_predictions.append(pred_means.cpu().numpy())
        actual_targets.append(y_label.numpy())
        # Store full posterior distributions for export instead of integers
        all_states.append(probs.cpu().numpy())
        all_entropies.append(entropies.cpu().numpy())

# 4. Evaluate Metrics
Y_test_flat_scaled = np.concatenate(actual_targets)
test_pred_scaled = np.concatenate(test_predictions)
all_states_flat = np.concatenate(all_states)
all_entropies_flat = np.concatenate(all_entropies)

# Inverse transform the scaled predictions and targets back to actual MW demand!
test_pred_flat = target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
Y_test_flat = target_scaler.inverse_transform(Y_test_flat_scaled.reshape(-1, 1)).flatten()

mae = mean_absolute_error(Y_test_flat, test_pred_flat)
rmse = np.sqrt(mean_squared_error(Y_test_flat, test_pred_flat))
r2 = r2_score(Y_test_flat, test_pred_flat)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

print("\n--- Diagnostics ---")
# Expected Occupancy = average of probabilities across test set
expected_occupancy = all_states_flat.mean(axis=0)
print("Expected State Occupancy:")
for state, mass in enumerate(expected_occupancy):
    print(f"  Regime {state}: {mass*100:.2f}%")
print(f"Mean Posterior Entropy: {all_entropies_flat.mean():.4f}")
print("-------------------\n")

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