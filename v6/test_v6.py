import sys
import os

# Add v6/src to path
sys.path.append(os.path.abspath("v6/src"))

import torch
from models.end_to_end import ContinuousStateForecaster

def test_model():
    print("Testing ContinuousStateForecaster...")
    batch_size = 4
    seq_len = 48
    raw_dim = 10
    gate_dim = 5
    
    model = ContinuousStateForecaster(
        raw_feature_dim=raw_dim, 
        gate_feature_dim=gate_dim, 
        seq_len=seq_len,
        d_model=64,
        latent_dim=32
    )
    
    x_raw = torch.randn(batch_size, seq_len, raw_dim)
    x_gate = torch.randn(batch_size, seq_len, gate_dim)
    
    y_hat, z_seq, aux_preds = model(x_raw, x_gate)
    
    print(f"y_hat shape: {y_hat.shape} (Expected: {batch_size})")
    print(f"z_seq shape: {z_seq.shape} (Expected: {batch_size}, {seq_len}, 32)")
    print(f"aux_preds shape: {aux_preds.shape} (Expected: {batch_size}, {seq_len}, 2)")
    
    # Test continuous losses
    from train import compute_continuous_losses
    l_smooth, l_ib = compute_continuous_losses(z_seq, 0.1, 1e-3)
    
    print(f"l_smooth: {l_smooth.item()}")
    print(f"l_ib: {l_ib.item()}")
    print("Test passed successfully.")

if __name__ == "__main__":
    test_model()
