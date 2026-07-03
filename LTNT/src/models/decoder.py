import torch
import torch.nn as nn

class HybridDecoder(nn.Module):
    def __init__(self, raw_feature_dim, seq_len, d_model=64, latent_dim=32, hidden_dim=128):
        super().__init__()
        
        # We concatenate the flattened raw features, the continuous latent state z_t (last step), 
        # and the contextual sequence h_t (last step).
        # raw_feature_dim * seq_len corresponds to the unrolled raw sequence.
        
        flat_raw_dim = raw_feature_dim * seq_len
        input_dim = flat_raw_dim + latent_dim + d_model
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_raw, z_seq, h_seq):
        """
        x_raw: (B, seq_len, raw_feature_dim)
        z_seq: (B, seq_len, latent_dim)
        h_seq: (B, seq_len, d_model)
        """
        B = x_raw.size(0)
        
        x_raw_flat = x_raw.reshape(B, -1)
        
        # Take the latent state and sequence context at the final timestep
        z_last = z_seq[:, -1, :] # (B, latent_dim)
        h_last = h_seq[:, -1, :] # (B, d_model)
        
        # Hybrid Context
        combined_features = torch.cat([x_raw_flat, z_last, h_last], dim=-1)
        
        y_hat = self.mlp(combined_features) # (B, 1)
        
        return y_hat.squeeze(-1)
