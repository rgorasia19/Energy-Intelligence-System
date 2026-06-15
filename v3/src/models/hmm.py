import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralHMM(nn.Module):
  def __init__(self, n_states, n_features):
    super().__init__()
    self.n_states = n_states

    #Transition matrix
    self.transition = nn.Parameter(torch.randn(n_states, n_states))

    #Emission
    self.means = nn.Parameter(torch.randn(n_states, n_features))
    self.log_variances = nn.Parameter(torch.randn(n_states, n_features))

    #Initial State Distribution
    self.initial_state = nn.Parameter(torch.randn(n_states))  
  
  def forward(self, x):
    T = x.shape[0]
    log_alpha = []

    #Initial step
    log_pi = F.log_softmax(self.initial_state, dim=0)
    log_emission = self.emission_log_prob(x[0])
    log_alpha_t = log_pi + log_emission
    log_alpha.append(log_alpha_t)

    #Forward algorithm
    for t in range(1, T):
        log_trans = F.log_softmax(self.transition, dim=1)
            
        prev = log_alpha[-1].unsqueeze(1)
        log_alpha_t = torch.logsumexp(prev + log_trans, dim=0)
            
        log_alpha_t += self.emission_log_prob(x[t])
        log_alpha.append(log_alpha_t)

    return torch.stack(log_alpha)

  def emission_log_prob(self, x_t):
    diff = x_t - self.means
    return -0.5 * ((diff ** 2) / torch.exp(self.log_vars) + self.log_vars).sum(dim=1)


  
  