import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import LinearExpert, FourierExpert, AttentionExpert

class GatingNetwork(nn.Module):
    """
    Operates strictly on X_gate (restricted, low-frequency features).
    Uses Gumbel-Softmax to output a discrete one-hot vector (Top-1 hard routing).
    """
    def __init__(self, gate_feature_dim, num_experts=3, hidden_dim=32):
        super().__init__()
        # GRU to process the gating features temporally
        self.gru = nn.GRU(
            input_size=gate_feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_experts)
        )

    def forward(self, x_gate, tau=1.0, hard=True):
        # x_gate shape: (batch_size, seq_len, gate_feature_dim)
        _, h_n = self.gru(x_gate)
        h_n = h_n.squeeze(0)
        
        logits = self.fc(h_n) # shape: (batch_size, num_experts)
        
        # Gumbel-Softmax for hard routing with Straight-Through Estimator (STE)
        if self.training:
            weights = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        else:
            # During inference, act greedily
            indices = torch.argmax(logits, dim=-1)
            weights = F.one_hot(indices, num_classes=logits.size(-1)).float()
            
        return weights, logits

class TemporalMoE(nn.Module):
    def __init__(self, raw_feature_dim, gate_feature_dim, seq_len):
        super().__init__()
        self.gating = GatingNetwork(gate_feature_dim, num_experts=3)
        
        self.linear_expert = LinearExpert(raw_feature_dim)
        self.fourier_expert = FourierExpert(raw_feature_dim, seq_len)
        self.attention_expert = AttentionExpert(raw_feature_dim)

    def forward(self, x_raw, x_gate, tau=1.0, hard=True):
        # 1. Get routing weights (hard one-hot vectors)
        weights, logits = self.gating(x_gate, tau=tau, hard=hard)
        
        # 2. Compute expert outputs
        out_linear = self.linear_expert(x_raw)
        out_fourier = self.fourier_expert(x_raw)
        out_attention = self.attention_expert(x_raw)
        
        # Stack outputs: (batch_size, num_experts)
        expert_outputs = torch.stack([out_linear, out_fourier, out_attention], dim=1)
        
        # 3. Final prediction
        final_output = torch.sum(weights * expert_outputs, dim=1)
        
        return final_output, weights, logits, expert_outputs
