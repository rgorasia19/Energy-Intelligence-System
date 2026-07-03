import torch
import torch.nn as nn
from .regime_attention import RegimeAttentionModel

class PredictionHead(nn.Module):
    def __init__(self, raw_feature_dim, seq_len, num_regimes=2, hidden_dim=128):
        super().__init__()
        
        flat_dim = raw_feature_dim * seq_len
        self.num_regimes = num_regimes
        
        # Instantiate separate MLPs for each regime
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_regimes)
        ])
        
    def forward(self, x_raw, p_regime):
        # x_raw: (B, seq_len, raw_feature_dim)
        # p_regime: (B, seq_len, num_regimes)
        
        B = x_raw.size(0)
        x_flat = x_raw.reshape(B, -1) # (B, flat_dim)
        
        # Calculate the global regime probability for the entire sequence (e.g. taking the last step)
        # We will use the last timestep's regime probability to gate the final sequence prediction
        p_global = p_regime[:, -1, :] # (B, num_regimes)
        
        # Get predictions from all experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x_flat)) # Each gives (B, 1)
            
        expert_tensor = torch.cat(expert_outputs, dim=-1) # (B, num_regimes)
        
        # Final prediction is the probability-weighted sum of expert predictions
        y_hat = torch.sum(p_global * expert_tensor, dim=-1) # (B,)
        
        return y_hat

class UnifiedRegimeModel(nn.Module):
    """
    End-to-End model combining the Regime Attention Network (discovery) 
    with the strict Mixture-of-Experts (MoE) Predictor.
    """
    def __init__(self, raw_feature_dim, gate_feature_dim, seq_len=48, num_regimes=2, d_model=64):
        super().__init__()
        self.regime_network = RegimeAttentionModel(
            gate_feature_dim, num_regimes, d_model, seq_len=seq_len
        )
        self.predictor = PredictionHead(raw_feature_dim, seq_len, num_regimes)
        
    def forward(self, x_raw, x_gate, tau=1.0, return_attention=False):
        # 1. Discover regime probabilities
        p_regime, logits, aux_preds, attention_maps = self.regime_network(x_gate, tau, return_attention)
        
        # 2. Predict based on raw features gated by regime probabilities
        y_hat = self.predictor(x_raw, p_regime)
        
        return y_hat, p_regime, logits, aux_preds, attention_maps
