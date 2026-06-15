import torch
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
from models.hmm import NeuralHMM


def train_model():
  device = "cuda" if torch.cuda.is_available() else "cpu"

  train_data = pd.read_parquet('../../datalake/hmm_tensors/train.parquet')
  val_data = pd.read_parquet('../../datalake/hmm_tensors/val.parquet')

  n_states = 4
  n_features = 20

  model = NeuralHMM(n_states=n_states, n_features=n_features).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

  loss_fn = nn.NLLLoss()

  batch_size = 64
  num_epochs = 20

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

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
        optimizer.zero_grad()
        log_alpha = model(x)
        loss = loss_fn(log_alpha, y_label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
      
      val_loss = 0
      with torch.no_grad():
        for x, y_label in val_loader:
          log_alpha = model(x)
          loss = loss_fn(log_alpha, y_label)
          val_loss += loss.item() * x.size(0)

      train_loss = train_loss / len(train_loader.dataset)
      val_loss = val_loss / len(val_loader.dataset)

      mlflow.log_metric("train_loss", train_loss, step=epoch)
      mlflow.log_metric("val_loss", val_loss, step=epoch)

    mlflow.pytorch.log_model(model, 'HMM_model')

train_model()