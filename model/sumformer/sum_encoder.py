import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .attn import FullAttention, AttentionLayer,\
                    SUMformer_AD_layer,SUMformer_MD_layer,SUMformer_AL_layer,SUMformer_AF_layer,\
                        SUMformer_AA_layer,SUMformer_TS

from math import ceil

class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    '''
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0: 
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x

class scale_block(nn.Module):
    '''
    Every phase has one patch merging layers. The ratio in our paper is 2.
    '''
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, \
                    seg_num = 10, factor=10,index=0,layer_scaler=1):
        super(scale_block, self).__init__()

        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        
        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            # Several variants for options
            self.encode_layers.append(SUMformer_AD_layer(seg_num, factor, d_model, n_heads, \
                                                         d_ff, dropout, index=index, layer_scaler=layer_scaler))
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)
        
        for layer in self.encode_layers:
            x = layer(x)
        
        return x

class Encoder(nn.Module):
    '''
    The TVF block for SUMformer
    '''
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout,
                in_seg_num = 10, factor=10,layer_scaler=1):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()
        self.block_depth = [1,1,1,1]
        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, self.block_depth[0], dropout,\
                                            in_seg_num, factor,layer_scaler=layer_scaler))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(win_size, d_model, n_heads, d_ff, self.block_depth[i], dropout,\
                                            ceil(in_seg_num/win_size**i), factor,index=i,layer_scaler=layer_scaler))

    def forward(self, x):
        encode_x = []
        encode_x.append(x)
        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x
