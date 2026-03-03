import torch
import torch.nn as nn
import torch.nn.functional as F

class PDR(nn.Module):
    def __init__(self, s_len=96, enc_in=7, patch_len=12, n_history=1, dropout=0.1, layernorm=True):
        super(PDR, self).__init__()
        self.n_history = n_history
        self.patch = patch_len
        self.s_len = s_len
        self.enc_in = enc_in

        self.agg = nn.Linear(n_history * patch_len, patch_len)

        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.s_len * self.enc_in)
        self.norm1 = nn.BatchNorm1d(self.n_history * self.patch * self.enc_in)

        self.dropout_t = nn.Dropout(dropout)

    def forward(self, s):
        # s: [batch_size, features_num, seq_len]
        if self.layernorm:
            s = self.norm(torch.flatten(s, 1, -1)).reshape(s.shape)

        output = torch.zeros_like(s)
        output[:, :, :self.n_history * self.patch] = s[:, :, :self.n_history * self.patch].clone()
        for i in range(self.n_history * self.patch, self.s_len, self.patch):
            # input [batch_size, feature_num, self.n_history * patch]
            input = output[:, :, i - self.n_history * self.patch: i]
            # input [batch_size, feature_num, self.n_history * patch]
            input = self.norm1(torch.flatten(input, 1, -1)).reshape(input.shape)
            # aggregation
            # [batch_size, feature_num, patch]
            input = F.gelu(self.agg(input))  # self.n_history * patch -> patch
            input = self.dropout_t(input)
            # input [batch_size, feature_num, patch]
            # input = torch.squeeze(input, dim=-1)
            tmp = input + s[:, :, i: i + self.patch]

            output[:, :, i: i + self.patch] = tmp

        return output

class MPDR(nn.Module):
    def __init__(self, configs, s_len, k=2, c=1, d=2, dropout=0.1):
        super(MPDR, self).__init__()
        """
            Args:
                k(int): the number of block
                c(int): mininum length of patch
                d(int): the base of the exponent
        """
        self.k = k
        self.c = c
        self.s_len = s_len
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(s_len)

        self.model_season = nn.ModuleList(
            [
                PDR(s_len=s_len, patch_len=self.c * (d ** i), enc_in=configs.d_model if configs.channel_mixing else configs.enc_in, layernorm=configs.layernorm)
                for i in range(self.k-1, -1, -1)
            ]
        )

    def forward(self, s):

        for i in range(self.k):
            s = self.norm(s)
            s = self.dropout(self.model_season[i](s)) + s

        return s
