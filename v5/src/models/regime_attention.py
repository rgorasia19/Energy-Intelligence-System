import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
    def forward(self, x, need_weights=False):
        # Self-attention
        attn_out, attn_weights = self.attention(x, x, x, need_weights=need_weights)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights

class RegimeEncoder(nn.Module):
    def __init__(self, feature_dim, d_model=64, n_heads=4, n_layers=2, seq_len=48):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding for sequence + 1 (for CLS)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model))
        
        self.layers = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
    def forward(self, x_gate, return_attention=False):
        # x_gate: (B, seq_len, feature_dim)
        B = x_gate.size(0)
        
        # Project inputs
        x = self.input_proj(x_gate) # (B, seq_len, d_model)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1) # (B, seq_len + 1, d_model)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        attention_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x, need_weights=return_attention)
            if return_attention:
                attention_maps.append(attn_weights)
                
        # Extract sequence output (excluding CLS token)
        seq_out = x[:, 1:, :] # (B, seq_len, d_model)
        
        return seq_out, attention_maps

class RegimeHead(nn.Module):
    def __init__(self, d_model=64, num_regimes=3, dropout_prob=0.1, embed_dim=32):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, num_regimes)
        )
        
        # Auxiliary Task: Predict vol and trend (2 outputs) from the regime probabilities
        self.aux_head = nn.Sequential(
            nn.Linear(num_regimes, 16),
            nn.GELU(),
            nn.Linear(16, 2)
        )
        
        # Regime Embedding Projection
        self.embed_proj = nn.Linear(num_regimes, embed_dim)
        
    def forward(self, seq_out, tau=1.0):
        # seq_out: (B, seq_len, d_model)
        logits = self.mlp(seq_out) # (B, seq_len, num_regimes)
        
        # Regime Dropout (only during training)
        if self.training and self.dropout_prob > 0.0:
            mask = (torch.rand_like(logits) > self.dropout_prob).float()
            # We set dropped logits to a very large negative number so softmax -> 0
            logits = logits.masked_fill(mask == 0, -1e9)
        
        # Softmax with temperature
        probs = F.softmax(logits / tau, dim=-1)
        
        # Predict auxiliary targets from probabilities
        aux_preds = self.aux_head(probs) # (B, seq_len, 2)
        
        # Project probabilities into a continuous embedding vector
        e_regime = self.embed_proj(probs) # (B, seq_len, embed_dim)
        
        return probs, logits, aux_preds, e_regime

class RegimeAttentionModel(nn.Module):
    def __init__(self, gate_feature_dim, num_regimes=3, d_model=64, n_heads=4, n_layers=2, seq_len=48, embed_dim=32):
        super().__init__()
        self.encoder = RegimeEncoder(gate_feature_dim, d_model, n_heads, n_layers, seq_len)
        self.head = RegimeHead(d_model, num_regimes, dropout_prob=0.1, embed_dim=embed_dim)
        
    def forward(self, x_gate, tau=1.0, return_attention=False):
        seq_out, attention_maps = self.encoder(x_gate, return_attention)
        probs, logits, aux_preds, e_regime = self.head(seq_out, tau)
        return probs, logits, aux_preds, e_regime, attention_maps
