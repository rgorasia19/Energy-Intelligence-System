import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NeuralHMM(nn.Module):
  def __init__(self, n_states, n_features):
    super().__init__()
    self.n_states = n_states
    self.n_features = n_features

    # Feature embedding network
    self.feature_net = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, n_features)
    )

    #Transition matrix
    self.transition = nn.Parameter(torch.randn(n_states, n_states))

    #Emission
    self.means = nn.Parameter(torch.randn(n_states, n_features))
    self.log_vars = nn.Parameter(torch.randn(n_states, n_features))

    #Initial State Distribution
    self.initial_state = nn.Parameter(torch.randn(n_states))  

    #Target Emission (Predicting demand y from final state)
    self.y_means = nn.Parameter(torch.randn(n_states))
    self.y_log_vars = nn.Parameter(torch.randn(n_states))
  
  def forward(self, x):
    # x shape: (batch_size, seq_len, num_features)
    batch_size, T, _ = x.shape
    
    # Apply feature network
    x_emb = self.feature_net(x)
    
    # Forward pass (log_alpha)
    log_alpha = torch.zeros(batch_size, T, self.n_states, device=x.device)
    log_trans = F.log_softmax(self.transition, dim=1) # (n_states, n_states)
    
    log_alpha[:, 0, :] = F.log_softmax(self.initial_state, dim=0) + self.emission_log_prob(x_emb[:, 0, :])
    
    for t in range(1, T):
        prev = log_alpha[:, t-1, :].unsqueeze(2) + log_trans.unsqueeze(0)
        log_alpha[:, t, :] = torch.logsumexp(prev, dim=1) + self.emission_log_prob(x_emb[:, t, :])

    # Backward pass (log_beta)
    log_beta = torch.zeros(batch_size, T, self.n_states, device=x.device)
    for t in range(T-2, -1, -1):
        obs_t1 = self.emission_log_prob(x_emb[:, t+1, :])
        beta_t1 = log_beta[:, t+1, :]
        val = log_trans.unsqueeze(0) + obs_t1.unsqueeze(1) + beta_t1.unsqueeze(1)
        log_beta[:, t, :] = torch.logsumexp(val, dim=2)
        
    # Gamma (smoothed posterior)
    log_gamma = log_alpha + log_beta
    # Normalize gamma at each timestep
    log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=2, keepdim=True)
    gamma = torch.exp(log_gamma)
    
    return log_alpha, gamma

  def emission_log_prob(self, x_t):
    # x_t shape: (batch_size, num_features)
    x_t = x_t.unsqueeze(1) # (batch_size, 1, num_features)
    means = self.means.unsqueeze(0) # (1, n_states, num_features)
    log_vars = self.log_vars.unsqueeze(0) # (1, n_states, num_features)
    
    diff = x_t - means
    return (-0.5 * math.log(2 * math.pi) - 0.5 * log_vars - 0.5 * ((diff ** 2) / torch.exp(log_vars))).sum(dim=2)

  def compute_loss(self, log_alpha, gamma, y):
    batch_size, seq_len, n_states = gamma.shape
    log_alpha_T = log_alpha[:, -1, :] 
    
    # 1. HMM Joint Likelihood
    y_expanded = y.unsqueeze(1) 
    y_means = self.y_means.unsqueeze(0) 
    y_log_vars = self.y_log_vars.unsqueeze(0) 
    
    log_prob_y = -0.5 * math.log(2 * math.pi) - 0.5 * y_log_vars - 0.5 * ((y_expanded - y_means) ** 2) / torch.exp(y_log_vars)
    joint_log_likelihood = torch.logsumexp(log_alpha_T + log_prob_y, dim=1)
    loss_hmm = -(joint_log_likelihood / seq_len).mean()
    
    # 2. Transition Matrix Regularisation
    # Penalize diagonal dominance, encourage cross-state mixing
    A = F.softmax(self.transition, dim=1)
    diag_penalty = A.diag().mean()
    # Maximize row entropy
    trans_entropy = -(A * torch.log(A + 1e-8)).sum(dim=1).mean()
    
    # 3. Posterior Entropy Control
    # Maximize H(gamma) to prevent regime collapse
    gamma_entropy = -(gamma * torch.log(gamma + 1e-8)).sum(dim=2).mean()
    
    # Total loss: L_hmm + lambda * diag - beta * H(gamma) - alpha * H(A)
    # Using small coefficients to not overpower the main likelihood
    lambda_diag = 0.1
    beta_ent = 0.05
    alpha_trans = 0.05
    
    total_loss = loss_hmm + (lambda_diag * diag_penalty) - (beta_ent * gamma_entropy) - (alpha_trans * trans_entropy)
    
    return total_loss