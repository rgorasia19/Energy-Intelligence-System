import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NeuralHMM(nn.Module):
  def __init__(self, n_states, n_features):
    super().__init__()
    self.n_states = n_states

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
    log_alpha = []

    #Initial step
    log_pi = F.log_softmax(self.initial_state, dim=0)
    log_emission = self.emission_log_prob(x[:, 0, :])
    # log_pi: (n_states,), log_emission: (batch_size, n_states)
    log_alpha_t = log_pi.unsqueeze(0) + log_emission
    log_alpha.append(log_alpha_t)

    #Forward algorithm
    for t in range(1, T):
        log_trans = F.log_softmax(self.transition, dim=1)
            
        # prev: (batch_size, n_states, 1)
        prev = log_alpha[-1].unsqueeze(2)
        # log_trans: (1, n_states, n_states)
        # We logsumexp over dim=1 (the previous state)
        log_alpha_t = torch.logsumexp(prev + log_trans.unsqueeze(0), dim=1)
            
        log_alpha_t = log_alpha_t + self.emission_log_prob(x[:, t, :])
        log_alpha.append(log_alpha_t)

    # Output shape: (batch_size, seq_len, n_states)
    return torch.stack(log_alpha, dim=1)

  def emission_log_prob(self, x_t):
    # x_t shape: (batch_size, num_features)
    # Means and vars shape: (n_states, num_features)
    x_t = x_t.unsqueeze(1) # (batch_size, 1, num_features)
    means = self.means.unsqueeze(0) # (1, n_states, num_features)
    log_vars = self.log_vars.unsqueeze(0) # (1, n_states, num_features)
    
    diff = x_t - means
    # Return shape: (batch_size, n_states)
    return (-0.5 * math.log(2 * math.pi) - 0.5 * log_vars - 0.5 * ((diff ** 2) / torch.exp(log_vars))).sum(dim=2)

  def compute_loss(self, log_alpha, y):
    # log_alpha shape: (batch_size, seq_len, n_states)
    # y shape: (batch_size,)
    
    # We take the log_alpha at the final timestep T
    log_alpha_T = log_alpha[:, -1, :] # (batch_size, n_states)
    
    # Compute log P(y | z_T) for each state
    y_expanded = y.unsqueeze(1) # (batch_size, 1)
    y_means = self.y_means.unsqueeze(0) # (1, n_states)
    y_log_vars = self.y_log_vars.unsqueeze(0) # (1, n_states)
    
    # log probability of Normal distribution
    log_prob_y = -0.5 * math.log(2 * math.pi) - 0.5 * y_log_vars - 0.5 * ((y_expanded - y_means) ** 2) / torch.exp(y_log_vars)
    
    # Joint log likelihood: log P(x_{1:T}, y) = logsumexp_{z_T} (log P(z_T, x_{1:T}) + log P(y | z_T))
    joint_log_likelihood = torch.logsumexp(log_alpha_T + log_prob_y, dim=1) # (batch_size,)
    
    # Minimize negative log likelihood
    return -joint_log_likelihood.mean()