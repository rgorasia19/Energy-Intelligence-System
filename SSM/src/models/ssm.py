import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentSSM(nn.Module):
    def __init__(self, input_dim, demand_dim, gen_dim, known_dim=5, latent_dim=8, hidden_dim=64, num_regimes=4, dropout=0.2, fourier_dim=12, fourier_embed_dim=16,
                 demand_scale=None, demand_center=None, gen_scale=None, gen_center=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.demand_dim = demand_dim
        self.gen_dim = gen_dim
        self.known_dim = known_dim
        self.num_regimes = num_regimes
        self.fourier_dim = fourier_dim
        
        self.fourier_embed = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fourier_embed_dim)
        )
        self.processed_known_dim = known_dim - fourier_dim + fourier_embed_dim
        
        # Learnable variance scaling parameter (alpha) initialized to 2.0 (ln(2) approx 0.693)
        self.log_alpha = nn.Parameter(torch.tensor([0.693]))
        
        # Register scale/center values for demand and gen targets as buffers to handle unscaling to raw space
        if demand_scale is not None:
            self.register_buffer('demand_scale', torch.tensor(demand_scale, dtype=torch.float32))
        else:
            self.register_buffer('demand_scale', torch.ones(demand_dim, dtype=torch.float32))
            
        if demand_center is not None:
            self.register_buffer('demand_center', torch.tensor(demand_center, dtype=torch.float32))
        else:
            self.register_buffer('demand_center', torch.zeros(demand_dim, dtype=torch.float32))
            
        if gen_scale is not None:
            self.register_buffer('gen_scale', torch.tensor(gen_scale, dtype=torch.float32))
        else:
            self.register_buffer('gen_scale', torch.ones(gen_dim, dtype=torch.float32))
            
        if gen_center is not None:
            self.register_buffer('gen_center', torch.tensor(gen_center, dtype=torch.float32))
        else:
            self.register_buffer('gen_center', torch.zeros(gen_dim, dtype=torch.float32))
        
        # 1. Past Encoder (for z0)
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.z0_mean = nn.Linear(hidden_dim, latent_dim * 2) # demand + gen
        self.z0_raw_var = nn.Linear(hidden_dim, latent_dim * 2)
        
        # Initial regime prior
        self.r0_logits = nn.Linear(hidden_dim, num_regimes)
        
        # 2. Bidirectional Smoothing Posterior
        # Takes future targets + known futures to infer optimal z_t and r_t
        self.posterior_lstm = nn.LSTM(demand_dim + gen_dim + self.processed_known_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.post_r_logits = nn.Linear(hidden_dim * 2, num_regimes)
        self.post_z_mean = nn.Linear(hidden_dim * 2, latent_dim * 2)
        self.post_z_raw_var = nn.Linear(hidden_dim * 2, latent_dim * 2)
        
        # 3. Prior Transition Dynamics
        # Regime transition: p(r_t | r_{t-1}, z_{t-1}, u_t)
        self.prior_r_net = nn.Sequential(
            nn.Linear(latent_dim * 2 + num_regimes + self.processed_known_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_regimes)
        )
        
        # Continuous transition: p(z_t | z_{t-1}, r_t, u_t)
        # Separate experts for Demand and Gen
        self.prior_z_experts_demand = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + self.processed_known_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim * 2) # mean, raw_var for demand
            ) for _ in range(num_regimes)
        ])
        
        self.prior_z_experts_gen = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + self.processed_known_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim * 2) # mean, raw_var for gen
            ) for _ in range(num_regimes)
        ])
        
        # 4. Probabilistic Emission
        # +1 for horizon-aware variance feature (normalized horizon step h/horizon)
        self.emit_demand = nn.Sequential(
            nn.Linear(latent_dim + self.processed_known_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.emit_gen = nn.Sequential(
            nn.Linear(latent_dim + self.processed_known_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.demand_mean = nn.Linear(hidden_dim, demand_dim)
        self.demand_raw_var = nn.Linear(hidden_dim, demand_dim)
        self.gen_mean = nn.Linear(hidden_dim, gen_dim)
        self.gen_raw_var = nn.Linear(hidden_dim, gen_dim)
        
    def _get_var(self, raw_var):
        # Softplus with variance floor to prevent deterministic collapse, scaled by learned alpha
        base_var = torch.clamp(F.softplus(raw_var) + 1e-2, max=10.0)
        return torch.exp(self.log_alpha) * base_var
        
    def reparameterize_gaussian(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def sample_regime(self, logits, tau=1.0, hard=True):
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        
    def _process_known(self, u):
        fourier = u[..., :self.fourier_dim]
        other = u[..., self.fourier_dim:]
        f_emb = self.fourier_embed(fourier)
        return torch.cat([f_emb, other], dim=-1)
        
    def forward(self, encoder_inputs, decoder_inputs, horizon, target_seq=None, sample=False, tau=1.0):
        """
        encoder_inputs: [batch, seq_len, input_dim]
        decoder_inputs: [batch, horizon, known_dim]
        target_seq: [batch, horizon, demand_dim + gen_dim] (only during training for posterior)
        """
        batch_size = encoder_inputs.size(0)
        device = encoder_inputs.device
        
        processed_decoder_inputs = self._process_known(decoder_inputs)
        
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
            post_inputs = torch.cat([target_seq, processed_decoder_inputs], dim=-1)
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
            u_t = processed_decoder_inputs[:, t, :]
            
            # Prior Regime Transition
            r_prior_input = torch.cat([z_curr, r_curr, u_t], dim=-1)
            r_logits = self.prior_r_net(r_prior_input)
            r_logits = r_logits + 10.0 * r_curr # Strong residual connection for persistence
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
            # z_curr: [batch, latent_dim * 2]
            z_curr_d, z_curr_g = torch.split(z_curr, self.latent_dim, dim=-1)
            
            z_prior_input_d = torch.cat([z_curr_d, u_t], dim=-1)
            z_prior_input_g = torch.cat([z_curr_g, u_t], dim=-1)
            
            # Compute outputs for all K experts
            expert_outputs_d = []
            expert_outputs_g = []
            for k in range(self.num_regimes):
                expert_outputs_d.append(self.prior_z_experts_demand[k](z_prior_input_d))
                expert_outputs_g.append(self.prior_z_experts_gen[k](z_prior_input_g))
                
            expert_outputs_d = torch.stack(expert_outputs_d, dim=1) # [batch, num_regimes, latent_dim * 2]
            expert_outputs_g = torch.stack(expert_outputs_g, dim=1)
            
            # Weighted sum over experts using Gumbel-Softmax r_next
            z_out_d = torch.einsum('bk,bkd->bd', r_next, expert_outputs_d)
            z_out_g = torch.einsum('bk,bkd->bd', r_next, expert_outputs_g)
            
            z_mean_d, z_raw_var_d = torch.split(z_out_d, self.latent_dim, dim=-1)
            z_mean_g, z_raw_var_g = torch.split(z_out_g, self.latent_dim, dim=-1)
            
            # Learn residual dynamics: z_mean = z_curr + z_delta
            z_mean_d = z_curr_d + z_mean_d
            z_mean_g = z_curr_g + z_mean_g
            
            z_mean = torch.cat([z_mean_d, z_mean_g], dim=-1)
            z_var = torch.cat([self._get_var(z_raw_var_d), self._get_var(z_raw_var_g)], dim=-1)
            
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
        sampled_z_d, sampled_z_g = torch.split(sampled_z_seq, self.latent_dim, dim=-1)
        
        # Horizon feature h / horizon
        h_feature = torch.arange(horizon, device=device, dtype=torch.float32) / float(horizon)
        h_feature = h_feature.view(1, horizon, 1).expand(batch_size, horizon, 1)
        
        emit_input_d = torch.cat([sampled_z_d, processed_decoder_inputs, h_feature], dim=-1)
        emit_input_g = torch.cat([sampled_z_g, processed_decoder_inputs, h_feature], dim=-1)
        
        demand_features = self.emit_demand(emit_input_d)
        demand_mean = self.demand_mean(demand_features)
        demand_var = self._get_var(self.demand_raw_var(demand_features))
        
        gen_features = self.emit_gen(emit_input_g)
        gen_mean = self.gen_mean(gen_features)
        gen_var = self._get_var(self.gen_raw_var(gen_features))
        
        # Unscale predicted mean & variance to raw physical space
        demand_mean = demand_mean * self.demand_scale + self.demand_center
        demand_var = demand_var * (self.demand_scale ** 2)
        
        gen_mean = gen_mean * self.gen_scale + self.gen_center
        gen_var = gen_var * (self.gen_scale ** 2)
        
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
    def __init__(self, kl_z_weight=1.0, kl_r_weight=1.0, entropy_weight=0.1,
                 demand_scale=None, demand_center=None, gen_scale=None, gen_center=None):
        super().__init__()
        self.kl_z_weight = kl_z_weight
        self.kl_r_weight = kl_r_weight
        self.entropy_weight = entropy_weight
        
        if demand_scale is not None:
            self.register_buffer('demand_scale', torch.tensor(demand_scale, dtype=torch.float32))
        else:
            self.register_buffer('demand_scale', None)
            
        if demand_center is not None:
            self.register_buffer('demand_center', torch.tensor(demand_center, dtype=torch.float32))
        else:
            self.register_buffer('demand_center', None)
            
        if gen_scale is not None:
            self.register_buffer('gen_scale', torch.tensor(gen_scale, dtype=torch.float32))
        else:
            self.register_buffer('gen_scale', None)
            
        if gen_center is not None:
            self.register_buffer('gen_center', torch.tensor(gen_center, dtype=torch.float32))
        else:
            self.register_buffer('gen_center', None)
        

        
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
        
    def forward(self, model_outputs, targets, masks, demand_idx, gen_idx, epoch, total_epochs, free_bits_z=0.1, free_bits_r=1.0, k_samples=1):
        demand_mean = model_outputs['demand_mean']
        demand_var = model_outputs['demand_var']
        gen_mean = model_outputs['gen_mean']
        gen_var = model_outputs['gen_var']
        
        target_demand = targets[:, :, demand_idx]
        target_gen = targets[:, :, gen_idx]
        
        # Unscale targets to raw space before loss calculation
        if self.demand_scale is not None and self.demand_center is not None:
            target_demand = target_demand * self.demand_scale + self.demand_center
        if self.gen_scale is not None and self.gen_center is not None:
            target_gen = target_gen * self.gen_scale + self.gen_center
            
        mask_demand = masks[:, :, demand_idx]
        mask_gen = masks[:, :, gen_idx]
        
        # Calculate per-item NLL and CRPS
        def calc_per_item_loss(mean, var, target, mask, is_demand):
            nll = 0.5 * torch.log(var) + (4.0 / 2.0) * torch.log(1.0 + ((target - mean) ** 2) / (3.0 * var))
            std = torch.sqrt(var)
            z = (target - mean) / (std + 1e-8)
            normal = torch.distributions.Normal(0, 1)
            crps = std * (z * (2 * normal.cdf(z) - 1) + 2 * torch.exp(normal.log_prob(z)) - 1 / torch.sqrt(torch.tensor(torch.pi)))
            
            combined = (nll + crps) * mask
            per_item = combined.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
            
            # Variance penalty
            var_penalty = -0.1 * torch.log(var) * mask
            per_item += var_penalty.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
            
            return per_item
            
        def calc_coverage_penalty(mean, var, target, mask):
            std = torch.sqrt(var) + 1e-8
            z = (target - mean) / std
            # 90% interval corresponds to [-1.645, 1.645] in dimensionless z-score space
            scale = 5.0
            coverage_approx = torch.sigmoid(scale * (z + 1.645)) * torch.sigmoid(scale * (1.645 - z))
            coverage = (coverage_approx * mask).sum() / (mask.sum() + 1e-8)
            return (0.9 - coverage) ** 2
            
        loss_demand_per_item = calc_per_item_loss(demand_mean, demand_var, target_demand, mask_demand, True)
        loss_gen_per_item = calc_per_item_loss(gen_mean, gen_var, target_gen, mask_gen, False)
        
        total_recon_per_item = loss_demand_per_item + loss_gen_per_item
        
        if k_samples > 1:
            batch_size = total_recon_per_item.size(0) // k_samples
            recon_reshaped = total_recon_per_item.view(batch_size, k_samples)
            import math
            recon_loss = -torch.logsumexp(-recon_reshaped, dim=1) + math.log(k_samples)
            recon_loss = recon_loss.mean()
        else:
            recon_loss = total_recon_per_item.mean()
            
        coverage_demand = calc_coverage_penalty(demand_mean, demand_var, target_demand, mask_demand)
        coverage_gen = calc_coverage_penalty(gen_mean, gen_var, target_gen, mask_gen)
        # Weight coverage penalty relative to recon_loss magnitude so calibration pressure remains strong in raw space
        coverage_weight = torch.clamp(recon_loss.detach().abs() * 0.5, min=10.0)
        coverage_loss = coverage_weight * (coverage_demand + coverage_gen)
            
        # Macro Consistency Loss (Sum over horizon)
        sum_pred_demand = (demand_mean * mask_demand).sum(dim=1)
        sum_target_demand = (target_demand * mask_demand).sum(dim=1)
        consistency_loss_demand = F.l1_loss(sum_pred_demand, sum_target_demand, reduction='none').mean()
        
        sum_pred_gen = (gen_mean * mask_gen).sum(dim=1)
        sum_target_gen = (target_gen * mask_gen).sum(dim=1)
        consistency_loss_gen = F.l1_loss(sum_pred_gen, sum_target_gen, reduction='none').mean()
        
        consistency_loss = 0.1 * (consistency_loss_demand + consistency_loss_gen)
            
        recon_loss = recon_loss + coverage_loss + consistency_loss
        
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
        mi_loss = 0.0
        util_loss = 0.0
        semantic_anchor_loss = 0.0
        
        if post_z_mean is not None:
            # KL for continuous state
            kl_z_raw = self.kl_divergence_gaussian(post_z_mean, post_z_var, prior_z_mean, prior_z_var)
            kl_z = torch.clamp(kl_z_raw - free_bits_z, min=0.0).mean()
            
            # KL for discrete regime
            kl_r_raw = self.kl_divergence_categorical(post_r_logits, prior_r_logits)
            kl_r = torch.clamp(kl_r_raw - free_bits_r, min=0.0).mean()
            
            # Entropy Regularization
            entropy_r = self.entropy_categorical(post_r_logits).mean()
            
            # Mutual Information & Utilization
            q_r = F.softmax(post_r_logits, dim=-1)
            num_regimes = q_r.size(-1)
            q_r_flat = q_r.view(-1, num_regimes)
            
            # Target magnitude for MI (use scaled targets to keep MI loss invariant to target scale)
            target_mag = targets[:, :, demand_idx].norm(dim=-1, keepdim=True).view(-1, 1)
            regime_weights = q_r_flat.sum(dim=0) + 1e-8
            expected_mag_per_regime = (q_r_flat * target_mag).sum(dim=0) / regime_weights
            
            # Maximize variance of expected target magnitude across regimes
            mi_loss = -0.5 * torch.var(expected_mag_per_regime)
            
            # Penalize dead regimes (force utilization)
            avg_q_r = q_r_flat.mean(dim=0)
            util_loss = 1.0 * (num_regimes * avg_q_r * torch.log(num_regimes * avg_q_r + 1e-8)).mean()
            
            # Semantic Anchor: Force Regime 0 for >90th percentile demand
            demand_scalar = target_demand.mean(dim=-1)
            q90 = torch.quantile(demand_scalar, 0.9)
            extreme_mask = demand_scalar > q90
            if extreme_mask.any():
                extreme_logits = post_r_logits[extreme_mask]
                target_regime = torch.zeros(extreme_logits.size(0), dtype=torch.long, device=extreme_logits.device)
                semantic_anchor_loss = 1.0 * F.cross_entropy(extreme_logits, target_regime)
            else:
                semantic_anchor_loss = 0.0
            
        # KL Annealing
        # (Annealing factor is passed from train.py, or we can use the default if not provided)
        # We will assume train.py passes it in some form, but we'll re-calculate just in case:
        cycle_length = max(1, total_epochs // 2)
        cycle_frac = (epoch % cycle_length) / cycle_length
        anneal_factor = min(1.0, cycle_frac / 0.5)
        
        # Latent Smoothness Loss
        if prior_z_mean.size(1) > 1:
            diff = prior_z_mean[:, 1:, :] - prior_z_mean[:, :-1, :]
            smoothness_loss = 0.1 * (diff ** 2).sum(dim=-1).mean()
        else:
            smoothness_loss = 0.0
        
        total_loss = recon_loss + anneal_factor * (self.kl_z_weight * kl_z + self.kl_r_weight * kl_r) - self.entropy_weight * entropy_r + mi_loss + util_loss + semantic_anchor_loss + smoothness_loss
        
        return total_loss, {
            'loss_demand': loss_demand_per_item.mean().item(),
            'loss_gen': loss_gen_per_item.mean().item(),
            'kl_z': kl_z.item() if isinstance(kl_z, torch.Tensor) else kl_z,
            'kl_r': kl_r.item() if isinstance(kl_r, torch.Tensor) else kl_r,
            'entropy_r': entropy_r.item() if isinstance(entropy_r, torch.Tensor) else entropy_r,
            'coverage_loss': coverage_loss.item() if isinstance(coverage_loss, torch.Tensor) else coverage_loss,
            'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
            'semantic_anchor_loss': semantic_anchor_loss.item() if isinstance(semantic_anchor_loss, torch.Tensor) else semantic_anchor_loss,
            'smoothness_loss': smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else smoothness_loss
        }
