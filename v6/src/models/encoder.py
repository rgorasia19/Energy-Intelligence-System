import torch
import torch.nn as nn

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class ContinuousEncoder(nn.Module):
    def __init__(self, feature_dim, d_model=64, n_heads=4, n_layers=2, seq_len=48):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # Positional encoding for sequence (no CLS token)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        self.layers = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
    def forward(self, x):
        # x: (B, seq_len, feature_dim)
        
        # Project inputs
        h = self.input_proj(x) # (B, seq_len, d_model)
        
        # Add positional embedding
        h = h + self.pos_embedding
        
        for layer in self.layers:
            h = layer(h)
                
        # output h: (B, seq_len, d_model)
        return h
