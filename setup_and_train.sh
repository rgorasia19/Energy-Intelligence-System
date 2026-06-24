#!/bin/bash

# Exit immediately if any command fails
set -e

echo "==> Creating virtual environment..."
python3 -m venv venv

echo "==> Activating virtual environment..."
source venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing requirements..."
pip install -r requirements.txt

echo "==> Moving to v4/src and starting training..."
cd v4/src
python train.py

echo "==> Training sequence completed successfully!"
