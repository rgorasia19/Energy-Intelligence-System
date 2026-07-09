import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentSSM(nn.Module):
    def __init__(self, input_dim, demand_dim, gen_dim, known_dim=5, latent_dim=8, hidden_dim=64, num_regimes=4, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.demand_dim = demand_dim
        self.gen_dim = gen_dim
        self.known_dim = known_dim
        self.num_regimes = num_regimes
        
        # 1. Past Encoder (for z0)
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.z0_mean = nn.Linear(hidden_dim, latent_dim)
        self.z0_raw_var = nn.Linear(hidden_dim, latent_dim)
        
        # Initial regime prior
        self.r0_logits = nn.Linear(hidden_dim, num_regimes)
        
        # 2. Bidirectional Smoothing Posterior
        # Takes future targets + known futures to infer optimal z_t and r_t
        self.posterior_lstm = nn.LSTM(demand_dim + gen_dim + known_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.post_r_logits = nn.Linear(hidden_dim * 2, num_regimes)
        self.post_z_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.post_z_raw_var = nn.Linear(hidden_dim * 2, latent_dim)
        
        # 3. Prior Transition Dynamics
        # Regime transition: p(r_t | r_{t-1}, z_{t-1}, u_t)
        self.prior_r_net = nn.Sequential(
            nn.Linear(latent_dim + num_regimes + known_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_regimes)
        )
        
        # Continuous transition: p(z_t | z_{t-1}, r_t, u_t)
        # Mixture of Experts: K separate transition networks
        self.prior_z_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + known_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim * 2) # mean, raw_var
            ) for _ in range(num_regimes)
        ])
        
        # 4. Probabilistic Emission
        self.emit_shared = nn.Linear(latent_dim + known_dim, hidden_dim) # no r_t here to force z_t to carry the state
        
        self.demand_mean = nn.Linear(hidden_dim, demand_dim)
        self.demand_raw_var = nn.Linear(hidden_dim, demand_dim)
        self.gen_mean = nn.Linear(hidden_dim, gen_dim)
        self.gen_raw_var = nn.Linear(hidden_dim, gen_dim)
        
    def _get_var(self, raw_var):
        # Softplus with variance floor to prevent deterministic collapse
        return torch.clamp(F.softplus(raw_var) + 1e-2, max=10.0)
        
    def reparameterize_gaussian(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def sample_regime(self, logits, tau=1.0, hard=True):
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        
    def forward(self, encoder_inputs, decoder_inputs, horizon, target_seq=None, sample=False, tau=1.0):
        """
        encoder_inputs: [batch, seq_len, input_dim]
        decoder_inputs: [batch, horizon, known_dim]
        target_seq: [batch, horizon, demand_dim + gen_dim] (only during training for posterior)
        """
        batch_size = encoder_inputs.size(0)
        device = encoder_inputs.device
        
        # 1. Encode past for z0 and r0
        _, h_n = self.encoder_gru(encoder_inputs)
        h_t = h_n[-1]
        
        z0_mean = self.z0_mean(h_t)
        z0_var = self._get_var(self.z0_raw_var(h_t))
        
        r0_logits = self.r0_logits(h_t)
        
        if self.training or sample:
            z_curr = self.reparameterize_gaussian(z0_mean, z0_var)
            r_curr = self.sample_regime(r0_logits, tau=tau, hard=True)
        else:
            z_curr = z0_mean
            r_curr = F.one_hot(torch.argmax(r0_logits, dim=-1), num_classes=self.num_regimes).float()
            
        # 2. Infer Posterior (if training)
        post_r_logits_seq = None
        post_z_mean_seq = None
        post_z_var_seq = None
        
        if self.training and target_seq is not None:
            post_inputs = torch.cat([target_seq, decoder_inputs], dim=-1)
            post_out, _ = self.posterior_lstm(post_inputs)
            post_r_logits_seq = self.post_r_logits(post_out)
            post_z_mean_seq = self.post_z_mean(post_out)
            post_z_var_seq = self._get_var(self.post_z_raw_var(post_out))
            
        # 3. Rollout Dynamics
        prior_r_logits_seq = []
        prior_z_mean_seq = []
        prior_z_var_seq = []
        sampled_z_seq = []
        sampled_r_seq = []
        
        for t in range(horizon):
            u_t = decoder_inputs[:, t, :]
            
            # Prior Regime Transition
            r_prior_input = torch.cat([z_curr, r_curr, u_t], dim=-1)
            r_logits = self.prior_r_net(r_prior_input)
            r_logits = r_logits + r_curr # Residual connection for persistence
            prior_r_logits_seq.append(r_logits)
            
            if self.training and target_seq is not None:
                # Use posterior samples during training to ground the continuous transition
                r_logits_t = post_r_logits_seq[:, t, :]
            else:
                r_logits_t = r_logits
                
            if self.training or sample:
                r_next = self.sample_regime(r_logits_t, tau=tau, hard=True)
            else:
                r_next = F.one_hot(torch.argmax(r_logits_t, dim=-1), num_classes=self.num_regimes).float()
                
            # Prior Continuous Transition (Mixture of Experts)
            # z_curr: [batch, latent_dim], u_t: [batch, known_dim], r_next: [batch, num_regimes]
            z_prior_input = torch.cat([z_curr, u_t], dim=-1)
            
            # Compute outputs for all K experts
            expert_outputs = []
            for k in range(self.num_regimes):
                expert_outputs.append(self.prior_z_experts[k](z_prior_input)) # Each: [batch, latent_dim * 2]
            expert_outputs = torch.stack(expert_outputs, dim=1) # [batch, num_regimes, latent_dim * 2]
            
            # Weighted sum over experts using Gumbel-Softmax r_next
            z_out = torch.einsum('bk,bkd->bd', r_next, expert_outputs)
            
            z_mean, z_raw_var = torch.split(z_out, self.latent_dim, dim=-1)
            z_var = self._get_var(z_raw_var)
            
            prior_z_mean_seq.append(z_mean)
            prior_z_var_seq.append(z_var)
            
            if self.training and target_seq is not None:
                z_mean_t = post_z_mean_seq[:, t, :]
                z_var_t = post_z_var_seq[:, t, :]
            else:
                z_mean_t = z_mean
                z_var_t = z_var
                
            if self.training or sample:
                z_next = self.reparameterize_gaussian(z_mean_t, z_var_t)
            else:
                z_next = z_mean_t
                
            sampled_r_seq.append(r_next)
            sampled_z_seq.append(z_next)
            
            z_curr = z_next
            r_curr = r_next
            
        prior_r_logits_seq = torch.stack(prior_r_logits_seq, dim=1)
        prior_z_mean_seq = torch.stack(prior_z_mean_seq, dim=1)
        prior_z_var_seq = torch.stack(prior_z_var_seq, dim=1)
        sampled_z_seq = torch.stack(sampled_z_seq, dim=1)
        sampled_r_seq = torch.stack(sampled_r_seq, dim=1)
        
        # 4. Probabilistic Emission
        emit_input = torch.cat([sampled_z_seq, decoder_inputs], dim=-1)
        shared_features = self.emit_shared(emit_input)
        
        demand_mean = self.demand_mean(shared_features)
        demand_var = self._get_var(self.demand_raw_var(shared_features))
        
        gen_mean = self.gen_mean(shared_features)
        gen_var = self._get_var(self.gen_raw_var(shared_features))
        
        return {
            'z0_mean': z0_mean,
            'z0_var': z0_var,
            'r0_logits': r0_logits,
            'post_r_logits_seq': post_r_logits_seq,
            'post_z_mean_seq': post_z_mean_seq,
            'post_z_var_seq': post_z_var_seq,
            'prior_r_logits_seq': prior_r_logits_seq,
            'prior_z_mean_seq': prior_z_mean_seq,
            'prior_z_var_seq': prior_z_var_seq,
            'sampled_z_seq': sampled_z_seq,
            'sampled_r_seq': sampled_r_seq,
            'demand_mean': demand_mean,
            'demand_var': demand_var,
            'gen_mean': gen_mean,
            'gen_var': gen_var
        }

class SSMLoss(nn.Module):
    def __init__(self, kl_z_weight=1.0, kl_r_weight=1.0, entropy_weight=0.1):
        super().__init__()
        self.kl_z_weight = kl_z_weight
        self.kl_r_weight = kl_r_weight
        self.entropy_weight = entropy_weight
        
    def student_t_nll(self, mean, var, target, mask, df=3.0):
        # NLL of Student-T distribution
        nll = 0.5 * torch.log(var) + ((df + 1.0) / 2.0) * torch.log(1.0 + ((target - mean) ** 2) / (df * var))
        masked_nll = nll * mask
        return masked_nll.sum() / (mask.sum() + 1e-8)
        
    def crps_approx(self, mean, var, target, mask):
        std = torch.sqrt(var)
        z = (target - mean) / (std + 1e-8)
        normal = torch.distributions.Normal(0, 1)
        cdf = normal.cdf(z)
        pdf = torch.exp(normal.log_prob(z))
        crps = std * (z * (2 * cdf - 1) + 2 * pdf - 1 / torch.sqrt(torch.tensor(torch.pi)))
        masked_crps = crps * mask
        return masked_crps.sum() / (mask.sum() + 1e-8)
        
    def kl_divergence_gaussian(self, q_mu, q_var, p_mu, p_var):
        # KL(q || p) = 0.5 * [log(p_var/q_var) + (q_var + (q_mu - p_mu)^2)/p_var - 1]
        kl = 0.5 * (torch.log(p_var / q_var) + (q_var + (q_mu - p_mu)**2) / p_var - 1.0)
        return kl.sum(dim=-1)
        
    def kl_divergence_categorical(self, q_logits, p_logits):
        q_probs = F.softmax(q_logits, dim=-1)
        q_log_probs = F.log_softmax(q_logits, dim=-1)
        p_log_probs = F.log_softmax(p_logits, dim=-1)
        kl = torch.sum(q_probs * (q_log_probs - p_log_probs), dim=-1)
        return kl
        
    def entropy_categorical(self, logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
        
    def forward(self, model_outputs, targets, masks, demand_idx, gen_idx, epoch, total_epochs, free_bits_z=0.1, free_bits_r=1.0):
        demand_mean = model_outputs['demand_mean']
        demand_var = model_outputs['demand_var']
        gen_mean = model_outputs['gen_mean']
        gen_var = model_outputs['gen_var']
        
        target_demand = targets[:, :, demand_idx]
        target_gen = targets[:, :, gen_idx]
        
        mask_demand = masks[:, :, demand_idx]
        mask_gen = masks[:, :, gen_idx]
        
        # 1. NLL + CRPS Emission Loss
        nll_demand = self.student_t_nll(demand_mean, demand_var, target_demand, mask_demand)
        nll_gen = self.student_t_nll(gen_mean, gen_var, target_gen, mask_gen)
        
        crps_demand = self.crps_approx(demand_mean, demand_var, target_demand, mask_demand)
        crps_gen = self.crps_approx(gen_mean, gen_var, target_gen, mask_gen)
        
        recon_loss = nll_demand + nll_gen + crps_demand + crps_gen
        
        # 2. Latent Consistency (KL Divergence)
        post_z_mean = model_outputs['post_z_mean_seq']
        post_z_var = model_outputs['post_z_var_seq']
        prior_z_mean = model_outputs['prior_z_mean_seq']
        prior_z_var = model_outputs['prior_z_var_seq']
        
        post_r_logits = model_outputs['post_r_logits_seq']
        prior_r_logits = model_outputs['prior_r_logits_seq']
        
        kl_z = 0.0
        kl_r = 0.0
        entropy_r = 0.0
        
        if post_z_mean is not None:
            # KL for continuous state
            kl_z_raw = self.kl_divergence_gaussian(post_z_mean, post_z_var, prior_z_mean, prior_z_var)
            kl_z = torch.clamp(kl_z_raw - free_bits_z, min=0.0).mean()
            
            # KL for discrete regime
            kl_r_raw = self.kl_divergence_categorical(post_r_logits, prior_r_logits)
            kl_r = torch.clamp(kl_r_raw - free_bits_r, min=0.0).mean()
            
            # Entropy Regularization (Maximizing entropy of posterior to prevent regime collapse)
            entropy_r = self.entropy_categorical(post_r_logits).mean()
            
        # KL Annealing
        anneal_factor = min(1.0, epoch / (total_epochs * 0.5))
        
        total_loss = recon_loss + anneal_factor * (self.kl_z_weight * kl_z + self.kl_r_weight * kl_r) - self.entropy_weight * entropy_r
        
        return total_loss, {
            'loss_demand': nll_demand.item(),
            'loss_gen': nll_gen.item(),
            'kl_z': kl_z.item() if isinstance(kl_z, torch.Tensor) else kl_z,
            'kl_r': kl_r.item() if isinstance(kl_r, torch.Tensor) else kl_r,
            'entropy_r': entropy_r.item() if isinstance(entropy_r, torch.Tensor) else entropy_r
        }
