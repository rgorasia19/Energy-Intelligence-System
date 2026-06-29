#!/bin/bash

# Strict mode: fail on error, fail on unset vars, fail on pipe errors
set -euo pipefail

echo "==> Creating virtual environment..."
python3 -m venv venv

echo "==> Activating virtual environment..."
source venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing requirements..."
pip install -r requirements.txt

# Export DagsHub tracking URI
export MLFLOW_TRACKING_URI="https://dagshub.com/rgorasia19/Energy-Intelligence-System.mlflow"

echo "==> Safety Check: MLflow tracking URI:"
echo $MLFLOW_TRACKING_URI
if [[ "$MLFLOW_TRACKING_URI" != *"dagshub.com"* ]]; then
    echo "ERROR: MLFLOW_TRACKING_URI is not pointing to DagsHub. Failing fast."
    exit 1
fi

echo "==> Moving to v7/src and starting evaluation..."
cd v7/src
python eval.py

echo "==> Evaluation sequence completed successfully!"
