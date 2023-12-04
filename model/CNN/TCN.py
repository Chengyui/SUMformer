import torch.nn.functional as F
import numpy as np
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from einops import reduce, rearrange


'''
Model
'''


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
        self.norm = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return self.norm(x + residual)

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, extract_layers=None):
        super().__init__()

        # if extract_layers is not None:
        #     assert len(channels) - 1 in extract_layers

        self.extract_layers = extract_layers
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
        ])
        self.norm = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        if self.extract_layers is not None:
            outputs = []
            for idx, mod in enumerate(self.net):
                x = mod(x)
                if idx in self.extract_layers:
                    outputs.append(x)
            return outputs
        return self.net(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(d_model, num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)


        x, _ = self.attention(x, x, x)


        x = x.permute(1, 0, 2)

        return x


class SimVP_Model(nn.Module):

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4,
                 mlp_ratio=8., drop=0.0, drop_path=0.01, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, dilat=True,**kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H_, W_ = int(H / 2 ** (N_S / 2)), int(W / 2 ** (N_S / 2))  # downsample 1 / 2**(N_S/2)

        self.time_dim = C*H*W #32*32*2
        self.N_T = N_T
        kernel = 1
        self.kernels = []
        while kernel<T:
            self.kernels.append(kernel)
            kernel *=2

        self.depth = 4
        self.dilat = True
        if self.dilat:
            self.time_enc = DilatedConvEncoder(C*H*W,[C*H*W]*self.depth,kernel_size=3) #可改造成指数型
        else:
            self.time_enc = nn.ModuleList(
                [nn.Conv1d(self.time_dim, self.time_dim, k, padding=k - 1) for k in self.kernels]
            )


        self.mode = 0
        self.att1 = MultiHeadSelfAttention(T,8)
        self.att2 = MultiHeadSelfAttention(T, 8)
        self.dec = nn.Linear(T,self.N_T)
        self.norm = nn.BatchNorm1d(C*H*W)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()



    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B ,T, C* H *W)
        x = x.transpose(1,2)


        embeds = self.time_enc(x)

        Y = embeds
        Y = self.dec(Y)

        Y = Y.transpose(1,2).view(B,self.N_T,C,H,W)

        return Y