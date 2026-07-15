import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentSSM(nn.Module):
    def __init__(self, input_dim, demand_dim, gen_dim, known_dim=5, latent_dim=8, hidden_dim=64, num_regimes=4, dropout=0.2, fourier_dim=12, fourier_embed_dim=16):
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
        
        # Learnable variance scaling parameter (alpha) initialized to zeros per latent feature
        self.log_alpha = nn.Parameter(torch.zeros(latent_dim * 2))
        self.log_alpha_demand = nn.Parameter(torch.zeros(demand_dim))
        self.log_alpha_gen = nn.Parameter(torch.zeros(gen_dim))
        
        # 1. Past Encoder (for z0)
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.z0_mean = nn.Linear(hidden_dim, latent_dim * 2) # demand + gen
        self.z0_raw_var = nn.Linear(hidden_dim, latent_dim * 2)
        
        # Initial regime prior
        self.r0_logits = nn.Linear(hidden_dim, num_regimes)
        
        # 2. Bidirectional Smoothing Posterior
        # Takes future targets + known futures to infer optimal z_t and r_t
        self.posterior_lstm = nn.LSTM(demand_dim + gen_dim + self.processed_known_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.post_r_logits = nn.Linear(hidden_dim, num_regimes)
        self.post_z_mean = nn.Linear(hidden_dim, latent_dim * 2)
        self.post_z_raw_var = nn.Linear(hidden_dim, latent_dim * 2)
        
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
        self.demand_mean = nn.Linear(hidden_dim, demand_dim)
        self.demand_scale_heads = nn.ModuleList([nn.Linear(hidden_dim, demand_dim) for _ in range(num_regimes)])
        self.demand_nu_heads = nn.ModuleList([nn.Linear(hidden_dim, demand_dim) for _ in range(num_regimes)])
        
        self.emit_gen = nn.Sequential(
            nn.Linear(latent_dim + self.processed_known_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gen_mean = nn.Linear(hidden_dim, gen_dim)
        self.gen_scale_heads = nn.ModuleList([nn.Linear(hidden_dim, gen_dim) for _ in range(num_regimes)])
        self.gen_nu_heads = nn.ModuleList([nn.Linear(hidden_dim, gen_dim) for _ in range(num_regimes)])
        
        # Initialize scale heads to start with a strong baseline bias
        for head in self.demand_scale_heads:
            nn.init.constant_(head.bias, 0.3)
            nn.init.normal_(head.weight, 0.0, 0.01)
        for head in self.gen_scale_heads:
            nn.init.constant_(head.bias, 0.3)
            nn.init.normal_(head.weight, 0.0, 0.01)

        # Structural horizon scaling parameters beta (one per target feature)
        self.beta_demand = nn.Parameter(torch.tensor([0.1] * demand_dim, dtype=torch.float32))
        self.beta_gen = nn.Parameter(torch.tensor([0.1] * gen_dim, dtype=torch.float32))
        
    def _get_var(self, raw_var):
        # Softplus with variance floor to prevent deterministic collapse, scaled by learned alpha
        base_var = F.softplus(raw_var) + 1e-2
        clamped_alpha = torch.clamp(self.log_alpha, -2.0, 2.0)
        return torch.exp(clamped_alpha) * base_var
        
    def _get_scale(self, raw_scale, target='demand'):
        base_scale = F.softplus(raw_scale + 2.0) + 1e-2
        alpha = self.log_alpha_demand if target == 'demand' else self.log_alpha_gen
        clamped_alpha = torch.clamp(alpha, -0.5, 0.5)
        modulation = torch.exp(0.1 * clamped_alpha)
        return base_scale * modulation
        
    def _get_nu(self, raw_nu):
        return 10.0 + F.softplus(raw_nu)
        
    def reparameterize_gaussian(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def sample_regime(self, logits, tau=1.0, hard=True, k_blend=1):
        if self.training and k_blend > 1 and not hard:
            # Top-K=2 Soft Expert Blending during training:
            topk_vals, topk_idx = torch.topk(logits, k=min(k_blend, logits.size(-1)), dim=-1)
            mask = torch.full_like(logits, -float('inf'))
            mask.scatter_(-1, topk_idx, topk_vals)
            return F.gumbel_softmax(mask, tau=tau, hard=False, dim=-1)
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        
    def _process_known(self, u):
        fourier = u[..., :self.fourier_dim]
        other = u[..., self.fourier_dim:]
        f_emb = self.fourier_embed(fourier)
        return torch.cat([f_emb, other], dim=-1)
        
    def forward(self, encoder_inputs, decoder_inputs, horizon, target_seq=None, sample=False, tau=1.0, tf_ratio=1.0):
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
            prior_r_logits_seq.append(r_logits)
            
            # Generate shared per-sample teacher forcing mask between z and r
            if self.training and target_seq is not None:
                mask_tf = (torch.rand(batch_size, 1, device=device) < tf_ratio)
            else:
                mask_tf = None
                
            # Sample prior and posterior regime states, then select branch with torch.where
            if self.training:
                r_prior = self.sample_regime(r_logits, tau=tau, hard=False, k_blend=2)
            elif sample:
                r_prior = self.sample_regime(r_logits, tau=tau, hard=True)
            else:
                r_prior = F.one_hot(torch.argmax(r_logits, dim=-1), num_classes=self.num_regimes).float()
                
            if self.training and target_seq is not None and mask_tf is not None:
                r_post = self.sample_regime(post_r_logits_seq[:, t, :], tau=tau, hard=False, k_blend=2)
                mask_tf_r = mask_tf.expand_as(r_post)
                r_next = torch.where(mask_tf_r, r_post, r_prior)
            else:
                r_next = r_prior
                
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
            
            # Weighted sum over experts using r_next
            z_out_d = torch.einsum('bk,bkd->bd', r_next, expert_outputs_d)
            z_out_g = torch.einsum('bk,bkd->bd', r_next, expert_outputs_g)
            
            z_mean_d, z_raw_var_d = torch.split(z_out_d, self.latent_dim, dim=-1)
            z_mean_g, z_raw_var_g = torch.split(z_out_g, self.latent_dim, dim=-1)
            
            z_mean = torch.cat([z_mean_d, z_mean_g], dim=-1)
            z_raw_var = torch.cat([z_raw_var_d, z_raw_var_g], dim=-1)
            z_var = self._get_var(z_raw_var)
            
            prior_z_mean_seq.append(z_mean)
            prior_z_var_seq.append(z_var)
            
            # Avoid sampling prior during teacher-forced training steps (use z_mean for prior branch)
            if self.training and target_seq is not None and mask_tf is not None:
                z_prior = z_mean
                z_post = self.reparameterize_gaussian(post_z_mean_seq[:, t, :], post_z_var_seq[:, t, :])
                mask_tf_z = mask_tf.expand_as(z_post)
                z_next = torch.where(mask_tf_z, z_post, z_prior)
            elif sample:
                z_next = self.reparameterize_gaussian(z_mean, z_var)
            else:
                z_next = z_mean
                
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
        
        gen_features = self.emit_gen(emit_input_g)
        gen_mean = self.gen_mean(gen_features)
        
        demand_scale_experts = torch.stack([self._get_scale(head(demand_features), 'demand') for head in self.demand_scale_heads], dim=2)
        demand_nu_experts = torch.stack([self._get_nu(head(demand_features)) for head in self.demand_nu_heads], dim=2)
        
        gen_scale_experts = torch.stack([self._get_scale(head(gen_features), 'gen') for head in self.gen_scale_heads], dim=2)
        gen_nu_experts = torch.stack([self._get_nu(head(gen_features)) for head in self.gen_nu_heads], dim=2)
        
        # Mix across active regime vector sampled_r_seq [batch_size, horizon, num_regimes]
        demand_scale_net = torch.einsum('bhr,bhrd->bhd', sampled_r_seq, demand_scale_experts)
        demand_nu = torch.einsum('bhr,bhrd->bhd', sampled_r_seq, demand_nu_experts)
        
        gen_scale_net = torch.einsum('bhr,bhrd->bhd', sampled_r_seq, gen_scale_experts)
        gen_nu = torch.einsum('bhr,bhrd->bhd', sampled_r_seq, gen_nu_experts)
        
        # Structural Horizon Scaling Factor: 1 + Softplus(beta) * sqrt((h + 1) / H)
        h_steps = torch.arange(1, horizon + 1, device=device, dtype=torch.float32)
        horizon_factor_d = 1.0 + F.softplus(self.beta_demand.view(1, 1, -1)) * torch.sqrt(h_steps / float(horizon)).view(1, horizon, 1)
        horizon_factor_g = 1.0 + F.softplus(self.beta_gen.view(1, 1, -1)) * torch.sqrt(h_steps / float(horizon)).view(1, horizon, 1)
        
        demand_scale = demand_scale_net * horizon_factor_d
        gen_scale = gen_scale_net * horizon_factor_g
        
        # Exact analytical variance for Student-t: nu / (nu - 2) * scale^2
        demand_var = (demand_nu / (demand_nu - 2.0)) * (demand_scale ** 2)
        gen_var = (gen_nu / (gen_nu - 2.0)) * (gen_scale ** 2)
        
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
            'demand_scale': demand_scale,
            'demand_nu': demand_nu,
            'demand_var': demand_var,
            'gen_mean': gen_mean,
            'gen_scale': gen_scale,
            'gen_nu': gen_nu,
            'gen_var': gen_var
        }

class SSMLoss(nn.Module):
    def __init__(self, kl_z_weight=1.0, kl_r_weight=1.0, entropy_weight=0.2, pinball_weight=0.5, occ_weight=10.0, min_occupancy=0.15):
        super().__init__()
        self.kl_z_weight = kl_z_weight
        self.kl_r_weight = kl_r_weight
        self.entropy_weight = entropy_weight
        self.pinball_weight = pinball_weight
        self.occ_weight = occ_weight
        self.min_occupancy = min_occupancy
        
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
        demand_scale = model_outputs['demand_scale']
        demand_nu = model_outputs['demand_nu']
        demand_var = model_outputs['demand_var']
        
        gen_mean = model_outputs['gen_mean']
        gen_scale = model_outputs['gen_scale']
        gen_nu = model_outputs['gen_nu']
        gen_var = model_outputs['gen_var']
        
        target_demand = targets[:, :, demand_idx]
        target_gen = targets[:, :, gen_idx]
        
        mask_demand = masks[:, :, demand_idx]
        mask_gen = masks[:, :, gen_idx]
        
        # Calculate per-item Student-t NLL and Normal-CRPS approximation
        def calc_crps_loss(mean, scale, nu, target, mask):
            std = torch.sqrt((nu / (nu - 2.0)) * (scale ** 2)) + 1e-8
            z = (target - mean) / std
            normal = torch.distributions.Normal(0, 1)
            crps = std * (z * (2 * normal.cdf(z) - 1) + 2 * torch.exp(normal.log_prob(z)) - 1 / torch.sqrt(torch.tensor(torch.pi)))
            
            per_item = (crps * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
            return per_item
            
        def calc_pinball_loss(mean, scale, nu, target, mask):
            # Symmetric Cornish-Fisher expansion for quantiles q=0.05, 0.50, 0.95
            qs = [0.05, 0.50, 0.95]
            zs = [-1.64485, 0.0, 1.64485]
            total_pinball = 0.0
            for q, z in zip(qs, zs):
                if z == 0.0:
                    t_q = 0.0
                else:
                    t_q = z + (z**3 + z) / (4.0 * nu) + (5.0*z**5 + 16.0*z**3 + 3.0*z) / (96.0 * (nu ** 2))
                y_hat = mean + scale * t_q
                err = target - y_hat
                pinball_q = torch.max(q * err, (q - 1.0) * err) * mask
                total_pinball += pinball_q.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
            return total_pinball / len(qs)
            
        def calc_coverage_penalty(mean, var, target, mask):
            std = torch.sqrt(var)
            lower = mean - 1.645 * std
            upper = mean + 1.645 * std
            
            # 1. Per-timestep, per-item violation
            violation = torch.relu(lower - target) + torch.relu(target - upper)
            violation_penalty = (violation * mask).sum() / (mask.sum() + 1e-8)
            
            # 2. Width Loss (force minimum width for 90% interval, data std is ~1 so width should be ~3.29)
            width = upper - lower
            width_penalty = torch.relu(3.0 - width)
            width_loss = (width_penalty * mask).sum() / (mask.sum() + 1e-8)
            
            return violation_penalty, width_loss
            
        loss_demand_per_item = calc_crps_loss(demand_mean, demand_scale, demand_nu, target_demand, mask_demand)
        loss_gen_per_item = calc_crps_loss(gen_mean, gen_scale, gen_nu, target_gen, mask_gen)
        
        pinball_demand = calc_pinball_loss(demand_mean, demand_scale, demand_nu, target_demand, mask_demand)
        pinball_gen = calc_pinball_loss(gen_mean, gen_scale, gen_nu, target_gen, mask_gen)
        pinball_loss = self.pinball_weight * (pinball_demand + pinball_gen).mean()
        
        violation_demand, width_loss_demand = calc_coverage_penalty(demand_mean, demand_var, target_demand, mask_demand)
        violation_gen, width_loss_gen = calc_coverage_penalty(gen_mean, gen_var, target_gen, mask_gen)
        
        coverage_loss = 5.0 * (violation_demand + violation_gen)
        width_loss = 100.0 * (width_loss_demand + width_loss_gen)
        
        # Scale Floor Loss (Fix #1: hard constraint to prevent scale collapse)
        min_scale_demand = 0.05 * target_demand.std(dim=(0, 1), keepdim=True).clamp(min=0.1)
        min_scale_gen = 0.05 * target_gen.std(dim=(0, 1), keepdim=True).clamp(min=0.1)
        scale_floor_loss_d = torch.relu(min_scale_demand - demand_scale).mean()
        scale_floor_loss_g = torch.relu(min_scale_gen - gen_scale).mean()
        scale_floor_loss = 10.0 * (scale_floor_loss_d + scale_floor_loss_g)
        
        # Direct Mean Anchor Loss: prevents the mean from drifting when variance grows
        l1_demand = (torch.abs(demand_mean - target_demand) * mask_demand).sum() / (mask_demand.sum() + 1e-8)
        l1_gen = (torch.abs(gen_mean - target_gen) * mask_gen).sum() / (mask_gen.sum() + 1e-8)
        mean_anchor_loss = 2.0 * (l1_demand + l1_gen)
        
        total_recon_per_item = loss_demand_per_item + loss_gen_per_item
        
        if k_samples > 1:
            batch_size = total_recon_per_item.size(0) // k_samples
            recon_reshaped = total_recon_per_item.view(batch_size, k_samples)
            import math
            recon_loss = -torch.logsumexp(-recon_reshaped, dim=1) + math.log(k_samples)
            recon_loss = recon_loss.mean()
        else:
            recon_loss = total_recon_per_item.mean()
            
        # Macro Consistency Loss (Sum over horizon)
        sum_pred_demand = (demand_mean * mask_demand).sum(dim=1)
        sum_target_demand = (target_demand * mask_demand).sum(dim=1)
        consistency_loss_demand = F.l1_loss(sum_pred_demand, sum_target_demand, reduction='none').mean()
        
        sum_pred_gen = (gen_mean * mask_gen).sum(dim=1)
        sum_target_gen = (target_gen * mask_gen).sum(dim=1)
        consistency_loss_gen = F.l1_loss(sum_pred_gen, sum_target_gen, reduction='none').mean()
        
        consistency_loss = 0.1 * (consistency_loss_demand + consistency_loss_gen)
            
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
        
        latent_consistency_loss = 0.0
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
            
            # Target magnitude for MI
            target_mag = target_demand.norm(dim=-1, keepdim=True).view(-1, 1)
            regime_weights = q_r_flat.sum(dim=0) + 1e-8
            expected_mag_per_regime = (q_r_flat * target_mag).sum(dim=0) / regime_weights
            
            # Maximize variance of expected target magnitude across regimes
            mi_loss = -0.5 * torch.var(expected_mag_per_regime)
            
            # Penalize dead regimes via quadratic Minimum Occupancy Penalty
            avg_q_r = q_r_flat.mean(dim=0)
            util_loss = self.occ_weight * torch.sum(torch.clamp(self.min_occupancy - avg_q_r, min=0.0) ** 2)
            
            # Semantic Anchor: Force Regime 0 for >90th percentile demand
            demand_scalar = target_demand.mean(dim=-1)
            q90 = torch.quantile(demand_scalar, 0.9)
            extreme_mask = demand_scalar > q90
            if extreme_mask.any():
                extreme_logits = post_r_logits[extreme_mask]
                target_regime = torch.zeros(extreme_logits.size(0), dtype=torch.long, device=extreme_logits.device)
                semantic_anchor_loss = 10.0 * F.cross_entropy(extreme_logits, target_regime)
            else:
                semantic_anchor_loss = 0.0
            
        # KL Annealing: gradual monotonic warmup to avoid posterior/prior fighting early training
        warmup_kl = max(1, int(total_epochs * 0.2))
        if epoch < warmup_kl:
            anneal_factor = 0.01 + 0.99 * (epoch / warmup_kl)
        else:
            anneal_factor = 1.0
        
        # Latent Smoothness Loss
        if prior_z_mean.size(1) > 1:
            diff = prior_z_mean[:, 1:, :] - prior_z_mean[:, :-1, :]
            smoothness_loss = 0.1 * (diff ** 2).sum(dim=-1).mean()
        else:
            smoothness_loss = 0.0
            
        # Latent L2 Regularization (replacing clamp)
        latent_l2_reg = 0.01 * (prior_z_mean ** 2).mean()
        
        total_loss = recon_loss + mean_anchor_loss + pinball_loss + coverage_loss + width_loss + scale_floor_loss + consistency_loss + anneal_factor * (self.kl_z_weight * kl_z + self.kl_r_weight * kl_r) - self.entropy_weight * entropy_r + mi_loss + util_loss + semantic_anchor_loss + smoothness_loss + latent_l2_reg
        
        return total_loss, {
            'loss_demand': loss_demand_per_item.mean().item(),
            'loss_gen': loss_gen_per_item.mean().item(),
            'pinball_loss': pinball_loss.item() if isinstance(pinball_loss, torch.Tensor) else pinball_loss,
            'kl_z': kl_z.item() if isinstance(kl_z, torch.Tensor) else kl_z,
            'kl_r': kl_r.item() if isinstance(kl_r, torch.Tensor) else kl_r,
            'entropy_r': entropy_r.item() if isinstance(entropy_r, torch.Tensor) else entropy_r,
            'coverage_loss': coverage_loss.item() if isinstance(coverage_loss, torch.Tensor) else coverage_loss,
            'width_loss': width_loss.item() if isinstance(width_loss, torch.Tensor) else width_loss,
            'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
            'latent_consistency_loss': latent_consistency_loss.item() if isinstance(latent_consistency_loss, torch.Tensor) else latent_consistency_loss,
            'semantic_anchor_loss': semantic_anchor_loss.item() if isinstance(semantic_anchor_loss, torch.Tensor) else semantic_anchor_loss,
            'smoothness_loss': smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else smoothness_loss,
            'util_loss': util_loss.item() if isinstance(util_loss, torch.Tensor) else util_loss,
            'avg_nu_demand': demand_nu.mean().item(),
            'avg_nu_gen': gen_nu.mean().item(),
            'z_mean_norm': prior_z_mean.norm(dim=-1).mean().item(),
            'z_std_mean': torch.sqrt(prior_z_var).mean().item(),
            'demand_scale_mean': demand_scale.mean().item(),
            'gen_scale_mean': gen_scale.mean().item(),
            'scale_floor_loss': scale_floor_loss.item() if isinstance(scale_floor_loss, torch.Tensor) else scale_floor_loss,
            'anneal_factor': anneal_factor
        }
