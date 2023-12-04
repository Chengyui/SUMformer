from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(seg_num, dim, depth, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(seg_num, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim)
    )



class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1,dilat =True):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.dilat = dilat
        if self.dilat:
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups
            )
            self.kernel_size = 0
        else:
            self.conv = nn.Conv1d(
                in_channels, out_channels, dilation,
                padding=dilation-1,
                groups=groups
            )
            self.kernel_size = dilation
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        if self.kernel_size>1:
            out = out[:,:,:-(self.kernel_size-1)]
        return out
def DilatFeedForward(dim,seg_num, expansion_factor = 4, dropout = 0.,index=1):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        SamePadConv(seg_num,seg_num,3,dilation=2**index),
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    )

def DilatMixer(seg_num, dim, depth, index,expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(seg_num, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, DilatFeedForward(dim,seg_num, expansion_factor_token, dropout,index=index))
        ) for _ in range(depth)],
        nn.LayerNorm(dim)
    )