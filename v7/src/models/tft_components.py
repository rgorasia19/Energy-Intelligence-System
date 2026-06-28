import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return self.fc1(x) * torch.sigmoid(self.fc2(x))

class GRN(nn.Module):
    """Gated Residual Network"""
    def __init__(self, input_size, hidden_size, output_size, context_size=None, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        if context_size is not None:
            self.context_projection = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_projection = None
            
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.elu = nn.ELU()
        
        self.glu = GLU(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.add_norm = nn.LayerNorm(output_size)
        
        if input_size != output_size:
            self.skip_projection = nn.Linear(input_size, output_size)
        else:
            self.skip_projection = None

    def forward(self, x, context=None):
        residual = x
        if self.skip_projection is not None:
            residual = self.skip_projection(x)
            
        x = self.fc1(x)
        if self.context_projection is not None and context is not None:
            x = x + self.context_projection(context)
            
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        
        return self.add_norm(x + residual)

class VSN(nn.Module):
    """Variable Selection Network"""
    def __init__(self, num_vars, hidden_size, context_size=None, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size
        
        self.flattened_grn = GRN(
            input_size=num_vars * hidden_size,
            hidden_size=hidden_size,
            output_size=num_vars,
            context_size=context_size,
            dropout=dropout
        )
        self.softmax = nn.Softmax(dim=-1)
        
        self.var_grns = nn.ModuleList([
            GRN(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout
            ) for _ in range(num_vars)
        ])

    def forward(self, x, context=None):
        # x is of shape (batch, seq_len, num_vars, hidden_size)
        batch_size, seq_len, _, _ = x.size()
        
        # Flatten num_vars and hidden_size for calculating variable weights
        flat_x = x.view(batch_size, seq_len, -1)
        sparse_weights = self.flattened_grn(flat_x, context)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(-1) # (batch, seq_len, num_vars, 1)
        
        # Pass each variable through its own GRN
        processed_vars = []
        for i in range(self.num_vars):
            var_out = self.var_grns[i](x[:, :, i, :])
            processed_vars.append(var_out)
        
        processed_vars = torch.stack(processed_vars, dim=2) # (batch, seq_len, num_vars, hidden_size)
        
        # Weight by the sparse weights
        weighted_vars = processed_vars * sparse_weights
        
        # Sum across the variables dimension
        out = weighted_vars.sum(dim=2) # (batch, seq_len, hidden_size)
        
        return out, sparse_weights.squeeze(-1)
