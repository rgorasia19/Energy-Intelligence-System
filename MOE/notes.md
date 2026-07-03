# V4 - TEMPORAL MIXTURE OF EXPERTS
Moving away from the Neural Hidden Markov Model, we will pursue a temporal mixture of experts to identify regimes. 
Architecture:

- Features : 
  - Raw features
  - Temporal derivatives
  - Volatility
  - Spectral, i.e short-window FFT magnitudes, dominant frequency index, energy in low vs high bands
  - Calendar/positional features
  - Regime priors

- Create a shared encoder where features are compressed
- Create a gating network that decides which experts determine the regime:
  - low capacity
  - add entropy + smoothing over time
- Create 3 experts:
  - Regime 1: Linear model to identify trends
  - Regime 2: Fourier Transform based model to identify periodicities
  - Regime 3: Attention based model to identify non-linearities

- Loss
  - Prediction is weighted sum of experts output by gating network
  - Add a penalty on entropy
  - Add a penalty on the L0 norm of the experts output
  - Smooth L1 loss for the experts
  - Expert diversity : encourage different outputs
  - load balancing : encourage usage of all experts

- Training:
  - Warm start experts separately:
    - Train Linear to predict target
    - Train Fourier to predict residual periodicities
    - Train Attention to predict residual non-linearities
  - Then plug into MoE

  - Gating temperature : use \pi = softmax(logits/\tau),
    - \tau is high initially for exploration
    - \tau lowers later to identifying sharp regimes

  - Expert dropout : Randomly disable experts during training for robustness
  - Normalize expert outputs to prevent scale dominance

  
