import torch
import torch.nn as nn
from .encoder import ContinuousEncoder
from .latent_state import ContinuousLatentState
from .decoder import HybridDecoder

class ContinuousStateForecaster(nn.Module):
    """
    End-to-End v6 model replacing MoE with a continuous latent state space.
    """
    def __init__(self, raw_feature_dim, gate_feature_dim, seq_len=48, d_model=64, latent_dim=32):
        super().__init__()
        
        # Transformer Backbone
        self.encoder = ContinuousEncoder(
            feature_dim=gate_feature_dim, 
            d_model=d_model, 
            seq_len=seq_len
        )
        
        # Core Latent State Transition
        self.latent_state_model = ContinuousLatentState(
            d_model=d_model,
            latent_dim=latent_dim
        )
        
        # Prediction Head
        self.decoder = HybridDecoder(
            raw_feature_dim=raw_feature_dim,
            seq_len=seq_len,
            d_model=d_model,
            latent_dim=latent_dim
        )
        
        # Auxiliary task (from v5): predict vol and trend to ensure latent state captures market dynamics
        self.aux_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 2)
        )
        
    def forward(self, x_raw, x_gate):
        # 1. Encode contextual features
        h_seq = self.encoder(x_gate) # (B, seq_len, d_model)
        
        # 2. Transition through continuous latent space
        z_seq = self.latent_state_model(h_seq) # (B, seq_len, latent_dim)
        
        # 3. Predict forecast from hybrid context
        y_hat = self.decoder(x_raw, z_seq, h_seq) # (B,)
        
        # 4. Predict auxiliary targets from latent state
        aux_preds = self.aux_head(z_seq) # (B, seq_len, 2)
        
        return y_hat, z_seq, aux_preds
