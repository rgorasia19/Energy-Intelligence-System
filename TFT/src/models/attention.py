import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.3, temperature=2.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.temperature = temperature
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        Q = self.q_proj(q)
        K = self.k_proj(k)
        V = self.v_proj(v)
        
        # Split into heads for Q and K: (batch, seq_len, num_heads, d_head) -> (batch, num_heads, seq_len, d_head)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        # TFT specific: V is shared across heads, so we don't split V by head dimension, 
        # instead we just expand V to match num_heads
        # But wait, standard TFT uses shared values. 
        # The equation is: Head_i = Attention(Q_i, K_i, V_proj(V))
        # V_proj is a linear layer mapping d_model to d_model, but standard TFT often maps to d_head.
        # Let's align with the standard implementation of Interpretable Multi-Head Attention.
        V = V.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        # Actually, in true interpretable MHA for TFT, V is passed through a single linear layer
        # and then attention scores are calculated per head, and we average them before multiplying with V.
        # Let's implement that:
        
        # 1. Attention scores per head: (batch, num_heads, q_len, k_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / ((self.d_head ** 0.5) * self.temperature)
        
        if mask is not None:
            # Mask is typically (batch, 1, q_len, k_len) or (q_len, k_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 2. Average attention weights over all heads: (batch, q_len, k_len)
        # This is the "interpretable" part
        avg_attn_weights = attn_weights.mean(dim=1) 
        
        # 3. Multiply with shared V:
        # We need V to be projected, but not split into heads. 
        # V is (batch, k_len, d_model)
        # avg_attn_weights is (batch, q_len, k_len)
        # out = (batch, q_len, d_model)
        V_shared = self.v_proj(v) # (batch, k_len, d_model)
        out = torch.bmm(avg_attn_weights, V_shared)
        
        # Final linear projection
        out = self.out_proj(out)
        
        return out, avg_attn_weights
