import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKScores(nn.Module):
    def __init__(self, input_dim=96, n_vars=7, top_k=2, noise_epsilon=1e-5):
        super(TopKScores, self).__init__()
        self.w_noise = nn.Parameter(torch.zeros(n_vars, n_vars), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        self.noise_epsilon = noise_epsilon
        self.top_k = top_k
        self.n_vars = n_vars
        self.input_dim = input_dim

    def forward(self, attn):
        # attn: [bs, num_features, seq_len]
        # TODO: 添加一个映射
        clean_logits = attn

        if self.training:
            raw_noise_stddev = attn @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.n_vars), dim=-1)
        top_k_logits = top_logits[:, :, : self.top_k]
        top_k_indices = top_indices[:, :, : self.top_k]

        top_k_gates = self.softmax(top_k_logits / (self.input_dim ** 0.5))

        zeros = torch.zeros_like(logits, requires_grad=True)
        sparse_attn = zeros.scatter(-1, top_k_indices, top_k_gates)

        return sparse_attn

class SparseSelfAttention(nn.Module):
    def __init__(self, attn_dopout=0.1, draw_attn_map=False, nst=False, gating=None):
        super(SparseSelfAttention, self).__init__()
        self.gating = gating
        self.dropout = nn.Dropout(attn_dopout)
        self.draw_attn_map = draw_attn_map
        self.nst = nst

    def forward(self, queries, keys, values):
        _, _, D = queries.shape

        scores = torch.einsum("bnd,bsd->bns", queries, keys)
        gates = self.gating(scores)

        A = self.dropout(gates)
        V = torch.einsum("bns,bsd->bnd", A, values)

        return V.contiguous(), A, gates
    
class FreqCointAttention(nn.Module):
    def __init__(self, attention, enc_in=7, s_len=12, seq_len=96, epsilon=1e-3, hidden_dim=256):
        super(FreqCointAttention, self).__init__()

        self.attention = attention
        self.enc_in = enc_in
        self.s_len = s_len
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Parameter(
            torch.ones(self.enc_in, self.s_len // 2 + 1)
        )
        self.k_proj = nn.Parameter(
            torch.ones(self.enc_in, self.s_len // 2 + 1)
        )
        self.v_proj = nn.Linear(self.s_len, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.s_len)

    def forward(self, x_enc, x_ori):
        # x_enc.shape [bs, n_var, s_len]

        # FFT Transformation
        x_freq = torch.fft.rfft(x_enc, dim=-1)
        x_amplitude = torch.abs(x_freq)

        quries = self.q_proj * x_amplitude
        keys = self.k_proj * x_amplitude
        values = self.v_proj(x_ori)

        out, attn, gates = self.attention(
            quries,
            keys,
            values
        )

        return self.out(out), attn, gates

class Encoder(nn.Module):
    def __init__(self, attention, dropout=0.1, activation="relu", s_len=96, d_ff=64):
        super(Encoder, self).__init__()
        self.s_len = s_len
        self.d_ff =d_ff

        self.fc1 = nn.Linear(self.s_len, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, self.s_len)

        self.attention = attention
        self.norm1 = nn.LayerNorm(self.s_len)
        self.norm2 = nn.LayerNorm(self.s_len)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)

        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x_enc, x):
        
        new_x, attn, gates = self.attention(x_enc, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        importance = gates.sum(0)
        loss = self.cv_squared(importance)

        return self.norm2(x + y), attn, loss
