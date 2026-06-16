import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
from models.hmm import NeuralHMM
from feature_prep import TimeSeriesDataset

# Enable TensorFloat-32 (TF32) for massive speedups on Ampere/Ada/Blackwell GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train_model():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using device: {device}")

  train_data = pd.read_parquet('../../datalake/hmm_tensors/train.parquet')
  val_data = pd.read_parquet('../../datalake/hmm_tensors/val.parquet')

  # Dynamically get feature columns
  feature_cols = [c for c in train_data.columns if c != 'TARGET_ND']

  n_states = 4
  n_features = len(feature_cols)

  model = NeuralHMM(n_states=n_states, n_features=n_features).to(device)
  
  # Compile model to fuse GPU operations and reduce Python overhead (PyTorch 2.0+)
  try:
      model = torch.compile(model)
  except Exception as e:
      print("Could not compile model, continuing without torch.compile:", e)

  optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
  # GradScaler for Mixed Precision training
  scaler = torch.amp.GradScaler('cuda')

  # Massively increased batch size to saturate the RTX 4500 GPU
  batch_size = 2048
  num_epochs = 20
  seq_length = 48

  train_dataset = TimeSeriesDataset(train_data[feature_cols].values, train_data['TARGET_ND'].values, seq_length=seq_length)
  val_dataset = TimeSeriesDataset(val_data[feature_cols].values, val_data['TARGET_ND'].values, seq_length=seq_length)

  # Added multiprocessing (num_workers) and pin_memory to prevent the GPU from waiting on data
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

  with mlflow.start_run():
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("n_states", n_states)
    mlflow.log_param("n_features", n_features)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)

    mlflow.log_metric("initial_state_std", model.initial_state.std().item())
    mlflow.log_metric("means_std", model.means.std().item())
    mlflow.log_metric("log_vars_std", model.log_vars.std().item())
    mlflow.log_metric("transition_std", model.transition.std().item())

    for epoch in range(num_epochs):
      train_loss = 0
      for x, y_label in train_loader:
        x, y_label = x.to(device), y_label.to(device)
        optimizer.zero_grad()
        
        # Automatic Mixed Precision for high throughput
        with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.bfloat16):
            log_alpha = model(x)
            loss = model.compute_loss(log_alpha, y_label)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item() * x.size(0)
      
      val_loss = 0
      with torch.no_grad():
        for x, y_label in val_loader:
          x, y_label = x.to(device), y_label.to(device)
          with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.bfloat16):
              log_alpha = model(x)
              loss = model.compute_loss(log_alpha, y_label)
          val_loss += loss.item() * x.size(0)

      train_loss = train_loss / len(train_loader.dataset)
      val_loss = val_loss / len(val_loader.dataset)

      mlflow.log_metric("train_loss", train_loss, step=epoch)
      mlflow.log_metric("val_loss", val_loss, step=epoch)
      print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # MLflow's pt2 format is extremely finicky with TensorSpecs.
    # Falling back to default 'pickle' format. You will see a security warning, but it will save successfully!
    mlflow.pytorch.log_model(model, name='HMM_model')

if __name__ == '__main__':
    train_model()