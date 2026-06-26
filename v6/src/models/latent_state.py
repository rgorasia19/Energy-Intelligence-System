import torch
import torch.nn as nn

class ContinuousLatentState(nn.Module):
    def __init__(self, d_model=64, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # We use a GRU cell to act as the continuous state transition module
        # h_t is the input from the transformer (d_model), z_{t-1} is the hidden state (latent_dim)
        self.gru_cell = nn.GRUCell(input_size=d_model, hidden_size=latent_dim)
        
    def forward(self, h_seq):
        """
        h_seq: (B, seq_len, d_model) context from encoder
        Returns:
            z_seq: (B, seq_len, latent_dim) latent state at each timestep
        """
        B, seq_len, _ = h_seq.size()
        
        # Initialize z_0 (could be learned or zeros, using zeros for simplicity)
        z_t = torch.zeros(B, self.latent_dim, device=h_seq.device)
        
        z_seq = []
        for t in range(seq_len):
            h_t = h_seq[:, t, :]
            z_t = self.gru_cell(h_t, z_t)
            z_seq.append(z_t)
            
        z_seq = torch.stack(z_seq, dim=1) # (B, seq_len, latent_dim)
        return z_seq
