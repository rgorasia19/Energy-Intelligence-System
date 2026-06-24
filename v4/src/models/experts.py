import torch
import torch.nn as nn

class LinearExpert(nn.Module):
    """
    Operates directly on the raw final timestep features to capture trends.
    """
    def __init__(self, raw_feature_dim):
        super().__init__()
        self.linear = nn.Linear(raw_feature_dim, 1)

    def forward(self, x_raw):
        # x_raw shape: (batch_size, seq_len, raw_feature_dim)
        # Only use the last timestep
        x_last = x_raw[:, -1, :]
        return self.linear(x_last).squeeze(-1)


class FourierExpert(nn.Module):
    """
    Operates directly on the raw sequence to identify periodicities using FFT.
    """
    def __init__(self, raw_feature_dim, seq_len):
        super().__init__()
        freq_dim = (seq_len // 2) + 1
        
        # Flatten the FFT magnitudes across all raw features
        self.fc = nn.Sequential(
            nn.Linear(raw_feature_dim * freq_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x_raw):
        # x_raw shape: (batch_size, seq_len, raw_feature_dim)
        fft_mag = torch.abs(torch.fft.rfft(x_raw, dim=1))
        fft_flat = fft_mag.reshape(fft_mag.size(0), -1)
        return self.fc(fft_flat).squeeze(-1)


class AttentionExpert(nn.Module):
    """
    Uses a GRU encoder followed by multi-head self-attention to capture non-linear interactions.
    """
    def __init__(self, raw_feature_dim, hidden_dim=64, num_heads=4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=raw_feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x_raw):
        # x_raw shape: (batch_size, seq_len, raw_feature_dim)
        gru_out, _ = self.gru(x_raw)
        gru_out = self.layer_norm(gru_out)
        
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Take the final timestep's output from the attention layer
        out_last = attn_out[:, -1, :]
        return self.fc(out_last).squeeze(-1)
