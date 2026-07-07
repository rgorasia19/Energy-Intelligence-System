import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentSSM(nn.Module):
    def __init__(self, input_dim, demand_dim, gen_dim, known_dim=4, latent_dim=8, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.demand_dim = demand_dim
        self.gen_dim = gen_dim
        self.known_dim = known_dim
        
        # 1. Encoder (Inference Net for z_0)
        # Takes past context (encoder_inputs) and outputs q(z_0 | x_{1:T})
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.z0_mean = nn.Linear(hidden_dim, latent_dim)
        self.z0_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 2. Latent Dynamics (Prior Net)
        # Gated residual dynamics: z_t = z_{t-1} + sigma(W z_{t-1}) * f(z_{t-1}) + eps
        self.f_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.gate_net = nn.Linear(latent_dim, latent_dim)
        
        # Prior variance (can be learned or fixed)
        # Using a small learned variance for process noise
        self.prior_logvar = nn.Parameter(torch.zeros(1, latent_dim))
        
        # 3. Emission Model (Decoder)
        # Maps z_t + calendar features to observations
        self.emission_shared = nn.Sequential(
            nn.Linear(latent_dim + known_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head emission
        self.demand_head = nn.Linear(hidden_dim, demand_dim)
        self.gen_head = nn.Linear(hidden_dim, gen_dim)
        
    def encode(self, x):
        """
        x: [batch, seq_len, input_dim]
        returns: z0_mean [batch, latent_dim], z0_logvar [batch, latent_dim]
        """
        _, h_n = self.encoder_gru(x)
        h_t = h_n[-1] # [batch, hidden_dim]
        
        z0_mean = self.z0_mean(h_t)
        z0_logvar = self.z0_logvar(h_t)
        
        return z0_mean, z0_logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def transition(self, z_prev):
        """
        z_t = z_{t-1} + sigma(W z_{t-1}) * f(z_{t-1})
        """
        f_z = self.f_net(z_prev)
        gate = torch.sigmoid(self.gate_net(z_prev))
        z_next_mean = z_prev + gate * f_z
        return z_next_mean
        
    def emit(self, z, decoder_inputs):
        """
        Maps latent state sequence + known futures to predictions.
        z: [batch, horizon, latent_dim]
        decoder_inputs: [batch, horizon, known_dim]
        """
        emission_input = torch.cat([z, decoder_inputs], dim=-1)
        shared_features = self.emission_shared(emission_input)
        pred_demand = self.demand_head(shared_features)
        pred_gen = self.gen_head(shared_features)
        return pred_demand, pred_gen
        
    def forward(self, encoder_inputs, decoder_inputs, horizon):
        """
        encoder_inputs: [batch, seq_len, input_dim]
        decoder_inputs: [batch, horizon, known_dim]
        horizon: int
        """
        batch_size = encoder_inputs.size(0)
        
        # 1. Infer initial state z0
        z0_mean, z0_logvar = self.encode(encoder_inputs)
        z0 = self.reparameterize(z0_mean, z0_logvar)
        
        # 2. Rollout latent dynamics over horizon
        z_seq = []
        z_curr = z0
        for _ in range(horizon):
            z_curr = self.transition(z_curr)
            # Add process noise during training (optional, or just use the mean for emission)
            if self.training:
                z_curr = self.reparameterize(z_curr, self.prior_logvar.expand(batch_size, -1))
            z_seq.append(z_curr)
            
        z_seq = torch.stack(z_seq, dim=1) # [batch, horizon, latent_dim]
        
        # 3. Emit predictions
        pred_demand, pred_gen = self.emit(z_seq, decoder_inputs)
        
        return {
            'z0_mean': z0_mean,
            'z0_logvar': z0_logvar,
            'z_seq': z_seq, # For smoothness penalty
            'pred_demand': pred_demand,
            'pred_gen': pred_gen
        }

class SSMLoss(nn.Module):
    def __init__(self, kl_weight=1.0, smooth_weight=0.1):
        super().__init__()
        self.kl_weight = kl_weight
        self.smooth_weight = smooth_weight
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, model_outputs, targets, masks, demand_idx, gen_idx, epoch, total_epochs):
        """
        model_outputs: dict from LatentSSM
        targets: [batch, horizon, target_dim]
        masks: [batch, horizon, target_dim]
        demand_idx: slice or list of indices for demand targets
        gen_idx: slice or list of indices for gen targets
        """
        pred_demand = model_outputs['pred_demand']
        pred_gen = model_outputs['pred_gen']
        
        target_demand = targets[:, :, demand_idx]
        target_gen = targets[:, :, gen_idx]
        
        mask_demand = masks[:, :, demand_idx]
        mask_gen = masks[:, :, gen_idx]
        
        # 1. Reconstruction Loss (Masked)
        loss_demand = (self.mse(pred_demand, target_demand) * mask_demand).sum() / (mask_demand.sum() + 1e-8)
        loss_gen = (self.mse(pred_gen, target_gen) * mask_gen).sum() / (mask_gen.sum() + 1e-8)
        
        recon_loss = loss_demand + loss_gen
        
        # 2. KL Divergence for initial state (Standard Normal Prior)
        # KL(q(z0 | x) || N(0, I))
        mu = model_outputs['z0_mean']
        logvar = model_outputs['z0_logvar']
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # KL Annealing
        anneal_factor = min(1.0, epoch / (total_epochs * 0.5))
        
        # 3. Temporal Smoothness Penalty
        # ||z_t - z_{t-1}||^2
        z_seq = model_outputs['z_seq'] # [batch, horizon, latent_dim]
        if z_seq.size(1) > 1:
            z_diff = z_seq[:, 1:, :] - z_seq[:, :-1, :]
            smoothness_loss = (z_diff ** 2).sum(dim=-1).mean()
        else:
            smoothness_loss = torch.tensor(0.0, device=z_seq.device)
            
        total_loss = recon_loss + (self.kl_weight * anneal_factor * kl_div) + (self.smooth_weight * smoothness_loss)
        
        return total_loss, {
            'loss_demand': loss_demand.item(),
            'loss_gen': loss_gen.item(),
            'kl_div': kl_div.item(),
            'smoothness': smoothness_loss.item()
        }
