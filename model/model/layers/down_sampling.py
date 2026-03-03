import torch
import torch.nn as nn

class process_multi_scale_series(nn.Module):
    def __init__(self, configs):
        super(process_multi_scale_series, self).__init__()
        self.down_sampling_layers = configs.down_sampling_layers
        self.down_sampling_method = configs.down_sampling_method
        self.down_sampling_window = configs.down_sampling_window
        self.enc_in = configs.d_model if configs.channel_mixing else configs.enc_in

        if self.down_sampling_method == 'max':
            self.down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            self.down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.down_pool = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in,
                                       kernel_size=3, padding=padding,
                                       stride=self.down_sampling_window,
                                       padding_mode='circular',
                                       bias=False)

    def forward(self, x):
        # x.shape [bs, n_vars, seq_len]
        x_sampling_list = []
        x_sampling_list.append(x)

        for i in range(self.down_sampling_layers):
            x_sampling = self.down_pool(x_sampling_list[i])
            x_sampling_list.append(x_sampling)
        
        return x_sampling_list
