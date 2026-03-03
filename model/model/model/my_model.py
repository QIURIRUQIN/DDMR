import torch
import torch.nn as nn

from layers.revin import RevIN
from layers.decomp import DECOMP
from layers.mpdr import MPDR
from layers.tsmoe import LightMoE
from layers.Transformer import TopKScores, SparseSelfAttention, FreqCointAttention, Encoder
from layers.Embed import DataEmbedding, TokenEmbedding
from layers.down_sampling import process_multi_scale_series

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_ff = configs.d_ff
        self.down_sampling_layers = configs.down_sampling_layers

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(self.enc_in, affine=True, subtract_last=False)

        # Moving Average
        self.ma_type = configs.ma_type
        self.alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        self.beta = configs.beta         # smoothing factor for DEMA (Double Exponential Moving Average)
        self.decomp = DECOMP(self.ma_type, self.alpha, self.beta)
        self.channel_mixing = configs.channel_mixing
        self.loss_coef = 0.5

        if self.channel_mixing:
            self.embedding = DataEmbedding(c_in=configs.enc_in, d_model=configs.d_model)
            self.enc_in = configs.d_model
            self.inverse_dimen_red = TokenEmbedding(c_in=configs.d_model, d_model=configs.enc_in)

        # model time dependency
        self.model_season_list = []
        for i in range(configs.down_sampling_layers + 1):
            s_len = self.seq_len // (configs.down_sampling_window ** i)
            if i == 0:
                self.model_season_list.append(
                    MPDR(configs=configs, s_len=s_len, k=3, c=configs.c, d=2)
                )

            else:
                self.model_season_list.append(
                    MPDR(configs=configs, s_len=s_len, k=configs.seq_len // (configs.down_sampling_window ** (i+1)) // configs.c, c=configs.c, d=2)
                )   

        self.model_season_list = nn.ModuleList(self.model_season_list)

        self.model_trend_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim := configs.seq_len // (configs.down_sampling_window ** i), dim * 4),
                    nn.AvgPool1d(kernel_size=2),
                    nn.LayerNorm(dim * 2),
                    nn.Linear(dim * 2, dim),
                    nn.AvgPool1d(kernel_size=2),
                    nn.LayerNorm(dim // 2),
                    nn.Linear(dim // 2, dim)
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        # mixing multi-scale series
        self.mms_blocks1 = nn.ModuleList(
            [
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** (i+1)), configs.seq_len // (configs.down_sampling_window ** i))
                for i in range(self.down_sampling_layers)
            ]
        )

        self.mms_blocks2 = nn.ModuleList(
            [
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** (i+1)), configs.seq_len // (configs.down_sampling_window ** i))
                for i in range(self.down_sampling_layers)
            ]
        )

        # model channel dependency
        self.down_sampling = process_multi_scale_series(configs)
        self.Encoder_list = nn.ModuleList(
            Encoder(
                    FreqCointAttention(
                        SparseSelfAttention(
                            gating=TopKScores(input_dim=configs.seq_len // (configs.down_sampling_window ** i), n_vars=configs.d_model if configs.channel_mixing else configs.enc_in, top_k=int(0.6 * configs.enc_in))
                        ),
                        enc_in=configs.d_model if configs.channel_mixing else configs.enc_in,
                        s_len=self.seq_len // (configs.down_sampling_window ** i),
                        seq_len=self.seq_len
                    ),
                    s_len=self.seq_len // (configs.down_sampling_window ** i),
                    d_ff=self.d_ff
                )
            for i in range(configs.down_sampling_layers + 1)
        )
        
        # MoE
        self.moe = LightMoE(self.seq_len, self.pred_len, ff_dim=2048, dropout=0.1, \
                            num_experts=configs.num_experts, top_k=configs.top_k, base_alpha=configs.base_alpha)

        # Streams Concatination
        self.fc1_list = nn.ModuleList(
            [
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** i) * 2, configs.seq_len // (configs.down_sampling_window ** i))
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        self.fc2 = nn.Linear(configs.seq_len * 2, configs.seq_len)
    
    def _mixing_multi_scale_series1(self, x_list):

        n = self.down_sampling_layers - 1
        for i in range(self.down_sampling_layers, 0, -1):
            temp = self.mms_blocks1[n](x_list[i])
            x_list[i-1] = x_list[i-1] + temp
            n -= 1

        return x_list
    
    def _mixing_multi_scale_series2(self, x_dec_list):

        n = self.down_sampling_layers - 1
        for i in range(self.down_sampling_layers, 0, -1):
            temp = self.mms_blocks2[n](x_dec_list[i])
            x_dec_list[i-1] = x_dec_list[i-1] + temp
            n -= 1

        return x_dec_list

    def forward(self, x):
        # x: [Batch, Input, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        B, L, N = x.shape
        if self.channel_mixing:
            x = self.embedding(x, None)
        x = x.transpose(1, 2).contiguous()
        # down sampling
        x_enc_list = self.down_sampling(x)

        # decomposition
        seasonal_init_list = []
        trend_init_list = []
        for i in range(self.down_sampling_layers + 1):
            seasonal_init, trend_init = self.decomp(x_enc_list[i].transpose(1, 2))
            seasonal_init_list.append(seasonal_init)
            trend_init_list.append(trend_init)

        # modeling time denpendency
        x_new_list = []
        for i, seasonal_init, trend_init in zip(range(self.down_sampling_layers + 1), seasonal_init_list, trend_init_list):
            s = self.model_season_list[i](seasonal_init.transpose(1, 2))
            t = self.model_trend_list[i](trend_init.transpose(1, 2))
            # temp = torch.cat([s, t], dim=-1)
            # x_new = self.fc1_list[i](temp)
            x_new = s + t
            x_new_list.append(x_new)

        # mixing multi-scale series
        x_ = self._mixing_multi_scale_series1(x_new_list)[0] + x
        
        x_dec_list = []
        gates_loss = 0
        for i, x_enc in enumerate(x_enc_list):
            temp, attn, loss  = self.Encoder_list[i](x_enc, x_enc)
            gates_loss += self.loss_coef * loss
            x_dec_list.append(temp)

        # mixing multi-scale series
        x_dec = self._mixing_multi_scale_series2(x_dec_list)[0] + x

        x_dec = torch.cat((x_, x_dec), dim=-1)
        x_dec = self.fc2(x_dec) + x
        if self.channel_mixing:
            x_dec = self.inverse_dimen_red(x_dec.permute(0, 2, 1)).transpose(1, 2)

        output, moe_loss = self.moe(x_dec, x_dec)
        # output, moe_loss = self.moe(x_, x_)
        output = torch.transpose(output, 1, 2)

        # Denormalization
        if self.revin:
            output = self.revin_layer(output, 'denorm')

        return output, moe_loss, gates_loss
