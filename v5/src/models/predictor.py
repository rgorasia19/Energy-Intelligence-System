import torch
import torch.nn as nn
from .regime_attention import RegimeAttentionModel

class PredictionHead(nn.Module):
    def __init__(self, raw_feature_dim, seq_len, embed_dim=32, hidden_dim=128):
        super().__init__()
        
        # Initial encoding of raw features
        flat_dim = raw_feature_dim * seq_len
        self.feature_encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # FiLM Generators: map E_regime (averaged over seq) to gamma and beta
        self.film_gamma = nn.Linear(embed_dim, hidden_dim)
        self.film_beta = nn.Linear(embed_dim, hidden_dim)
        
        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_raw, e_regime):
        # x_raw: (B, seq_len, raw_feature_dim)
        # e_regime: (B, seq_len, embed_dim)
        
        B = x_raw.size(0)
        
        # Flatten raw features temporally
        x_flat = x_raw.reshape(B, -1)
        
        # Encode features
        h = self.feature_encoder(x_flat) # (B, hidden_dim)
        
        # Average the regime embedding over the sequence length for the global context
        e_global = e_regime.mean(dim=1) # (B, embed_dim)
        
        # Generate FiLM parameters
        gamma = self.film_gamma(e_global) # (B, hidden_dim)
        beta = self.film_beta(e_global) # (B, hidden_dim)
        
        # Apply FiLM modulation: h' = gamma * h + beta
        h_modulated = gamma * h + beta
        
        # Final prediction
        out = self.mlp(h_modulated) # (B, 1)
        return out.squeeze(-1)

class UnifiedRegimeModel(nn.Module):
    """
    End-to-End model combining the Regime Attention Network (discovery) 
    with the FiLM-Conditioned Downstream Predictor.
    """
    def __init__(self, raw_feature_dim, gate_feature_dim, seq_len=48, num_regimes=3, d_model=64, embed_dim=32):
        super().__init__()
        self.regime_network = RegimeAttentionModel(
            gate_feature_dim, num_regimes, d_model, seq_len=seq_len, embed_dim=embed_dim
        )
        self.predictor = PredictionHead(raw_feature_dim, seq_len, embed_dim)
        
    def forward(self, x_raw, x_gate, tau=1.0, return_attention=False):
        # 1. Discover regime probabilities and embeddings
        p_regime, logits, aux_preds, e_regime, attention_maps = self.regime_network(x_gate, tau, return_attention)
        
        # 2. Predict based on raw features and FiLM modulated regime embedding
        y_hat = self.predictor(x_raw, e_regime)
        
        return y_hat, p_regime, logits, aux_preds, attention_maps
