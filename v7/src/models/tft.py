import torch
import torch.nn as nn
from .tft_components import GRN, VSN, GLU
from .attention import InterpretableMultiHeadAttention

class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        num_static_vars,
        num_future_vars,
        num_past_vars,
        d_model=64,
        num_heads=4,
        seq_len=48,
        horizon=48,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.horizon = horizon
        
        # 1. Feature Encoders (Mapping to d_model)
        self.static_encoders = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_static_vars)])
        self.past_encoders = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_past_vars)])
        self.future_encoders = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_future_vars)])
        
        # 2. Variable Selection Networks
        self.static_vsn = VSN(num_static_vars, d_model, dropout=dropout)
        # For past, we consider both past_vars and future_vars (which are available in the past)
        self.past_vsn = VSN(num_past_vars + num_future_vars, d_model, context_size=d_model, dropout=dropout)
        self.future_vsn = VSN(num_future_vars, d_model, context_size=d_model, dropout=dropout)
        
        # Static Context Encoders
        self.static_context_grn_cs = GRN(d_model, d_model, d_model, dropout=dropout)
        self.static_context_grn_ce = GRN(d_model, d_model, d_model, dropout=dropout)
        self.static_context_grn_ch = GRN(d_model, d_model, d_model, dropout=dropout)
        self.static_context_grn_cc = GRN(d_model, d_model, d_model, dropout=dropout)
        
        # 3. LSTM Encoders
        self.past_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.future_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)
        
        self.post_lstm_gate = GLU(d_model, d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)
        
        # 4. Interpretable Attention
        self.attention = InterpretableMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.post_attn_gate = GLU(d_model, d_model)
        self.post_attn_norm = nn.LayerNorm(d_model)
        
        # 5. Position-wise GRN
        self.pos_grn = GRN(d_model, d_model, d_model, dropout=dropout)
        self.final_gate = GLU(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)
        
        # 6. Output Layers (Joint prediction of ND, VOL, TREND for horizon steps)
        # We need to map from the decoder states to the H steps output
        self.out_nd = nn.Linear(d_model, 1)
        self.out_vol = nn.Linear(d_model, 1)
        self.out_trend = nn.Linear(d_model, 1)
        
    def forward(self, static_feat, past_feat, future_feat):
        """
        static_feat: (batch, num_static)
        past_feat: (batch, seq_len, num_past)
        future_feat_past: (batch, seq_len, num_future)
        future_feat_future: (batch, horizon, num_future) 
        Wait, we need the future features for the full sequence (seq_len + horizon).
        Let's assume future_feat is (batch, seq_len + horizon, num_future)
        """
        batch_size = static_feat.size(0)
        
        # Split future_feat into past part and future part
        future_feat_past = future_feat[:, :self.seq_len, :]
        future_feat_future = future_feat[:, self.seq_len:, :]
        
        # Encode static features
        encoded_static = [self.static_encoders[i](static_feat[:, i:i+1]) for i in range(static_feat.size(1))]
        encoded_static = torch.stack(encoded_static, dim=1) # (batch, num_static, d_model)
        
        # VSN for static
        static_out, _ = self.static_vsn(encoded_static.unsqueeze(1)) # add dummy seq_len dim
        static_context = static_out.squeeze(1) # (batch, d_model)
        
        # Create contexts
        cs = self.static_context_grn_cs(static_context)
        ce = self.static_context_grn_ce(static_context)
        ch = self.static_context_grn_ch(static_context)
        cc = self.static_context_grn_cc(static_context)
        
        # Encode past features
        encoded_past = [self.past_encoders[i](past_feat[:, :, i:i+1]) for i in range(past_feat.size(2))]
        encoded_future_past = [self.future_encoders[i](future_feat_past[:, :, i:i+1]) for i in range(future_feat_past.size(2))]
        
        # VSN for past
        all_past_vars = torch.stack(encoded_past + encoded_future_past, dim=2) # (batch, seq_len, num_vars, d_model)
        past_out, _ = self.past_vsn(all_past_vars, cs.unsqueeze(1).expand(-1, self.seq_len, -1))
        
        # Encode future features
        encoded_future_future = [self.future_encoders[i](future_feat_future[:, :, i:i+1]) for i in range(future_feat_future.size(2))]
        all_future_vars = torch.stack(encoded_future_future, dim=2)
        future_out, _ = self.future_vsn(all_future_vars, cs.unsqueeze(1).expand(-1, self.horizon, -1))
        
        # LSTMs
        # Init hidden states with static context
        h0 = ch.unsqueeze(0)
        c0 = cc.unsqueeze(0)
        
        past_lstm_out, (h_past, c_past) = self.past_lstm(past_out, (h0, c0))
        future_lstm_out, _ = self.future_lstm(future_out, (h_past, c_past))
        
        # Combine LSTM outputs
        lstm_out = torch.cat([past_lstm_out, future_lstm_out], dim=1) # (batch, seq_len + horizon, d_model)
        vsn_out = torch.cat([past_out, future_out], dim=1) # (batch, seq_len + horizon, d_model)
        
        # Gating and Residual
        lstm_gated = self.post_lstm_gate(lstm_out)
        lstm_res = self.post_lstm_norm(lstm_gated + vsn_out)
        
        # Static enrichment (add ce to all steps)
        enriched = lstm_res + ce.unsqueeze(1)
        
        # Attention
        # Provide causal mask or just standard mask for TFT
        # In TFT, the decoder can attend to all past steps
        attn_out, attn_weights = self.attention(q=enriched, k=enriched, v=enriched)
        
        attn_gated = self.post_attn_gate(attn_out)
        attn_res = self.post_attn_norm(attn_gated + lstm_res)
        
        # Position-wise GRN
        grn_out = self.pos_grn(attn_res)
        final_gated = self.final_gate(grn_out)
        final_out = self.final_norm(final_gated + attn_res) # (batch, seq_len + horizon, d_model)
        
        # We only care about the future steps for prediction
        future_final = final_out[:, self.seq_len:, :] # (batch, horizon, d_model)
        
        pred_nd = self.out_nd(future_final).squeeze(-1) # (batch, horizon)
        pred_vol = self.out_vol(future_final).squeeze(-1)
        pred_trend = self.out_trend(future_final).squeeze(-1)
        
        return pred_nd, pred_vol, pred_trend, attn_weights
