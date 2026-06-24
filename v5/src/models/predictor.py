import torch
import torch.nn as nn
from .regime_attention import RegimeAttentionModel

class PredictionHead(nn.Module):
    def __init__(self, raw_feature_dim, seq_len, num_regimes=3, hidden_dim=128):
        super().__init__()
        
        # We concatenate the regime probabilities to the raw features at each timestep before flattening
        input_dim = (raw_feature_dim + num_regimes) * seq_len
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_raw, p_regime):
        # x_raw: (B, seq_len, raw_feature_dim)
        # p_regime: (B, seq_len, num_regimes)
        
        B = x_raw.size(0)
        
        # Concatenate features and regimes at each timestep
        x_cond = torch.cat((x_raw, p_regime), dim=-1) # (B, seq_len, raw_feature_dim + num_regimes)
        
        # Flatten for the MLP prediction
        x_flat = x_cond.reshape(B, -1)
        
        out = self.mlp(x_flat) # (B, 1)
        return out.squeeze(-1)

class UnifiedRegimeModel(nn.Module):
    """
    End-to-End model combining the Regime Attention Network (discovery) 
    with the conditioned Downstream Predictor.
    """
    def __init__(self, raw_feature_dim, gate_feature_dim, seq_len=48, num_regimes=3, d_model=64):
        super().__init__()
        self.regime_network = RegimeAttentionModel(
            gate_feature_dim, num_regimes, d_model, seq_len=seq_len
        )
        self.predictor = PredictionHead(raw_feature_dim, seq_len, num_regimes)
        
    def forward(self, x_raw, x_gate, tau=1.0, return_attention=False):
        # 1. Discover regime probabilities
        p_regime, logits, attention_maps = self.regime_network(x_gate, tau, return_attention)
        
        # 2. Predict based on raw features and discovered regime
        y_hat = self.predictor(x_raw, p_regime)
        
        return y_hat, p_regime, logits, attention_maps
