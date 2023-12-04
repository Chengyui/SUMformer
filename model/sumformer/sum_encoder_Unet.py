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


class SegSplit(nn.Module):
    '''
    Segment Merging Layer.
    '''
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(d_model, d_model*win_size)
        self.norm = norm_layer(d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        batch_size, ts_d, seg_num, d_model = x.shape

        x = self.linear_trans(x)

        x = rearrange(x, 'b ts_d seg_num (win_size d_model) -> b ts_d (seg_num win_size) d_model',
                      win_size=self.win_size)

        x = self.norm(x)

        return x


class SegMergingNew(nn.Module):
    '''
    Segment Merging Layer.
    '''
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(d_model,  d_model)
        self.norm = norm_layer(d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        # batch_size, ts_d, seg_num, d_model = x.shape
        # pad_num = seg_num % self.win_size
        # if pad_num != 0:
        #     pad_num = self.win_size - pad_num
        #     x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)
        #
        # seg_to_merge = []
        # for i in range(self.win_size):
        #     seg_to_merge.append(x[:, :, i::self.win_size, :])
        # x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]
        #
        # x = self.norm(x)
        # x = self.linear_trans(x)

        x = rearrange(x,'b ts_d (seg_num win_size) d_model -> b ts_d seg_num (win_size d_model)',win_size=self.win_size)
        x = self.norm(x)
        x = self.linear_trans(x)

        return x

class SegSplitNew(nn.Module):
    '''
    Segment split layer
    '''
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(d_model, d_model*win_size)
        self.norm = norm_layer(d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        # batch_size, ts_d, seg_num, d_model = x.shape
        #
        x = self.linear_trans(x)

        x = rearrange(x, 'b ts_d seg_num (win_size d_model) -> b ts_d (seg_num win_size) d_model',
                      win_size=self.win_size)

        x = self.norm(x)

        return x
class down_block(nn.Module):
    '''
        TVF block
    '''
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, \
                    seg_num = 10, factor=10,index=0):
        super(down_block, self).__init__()
        if (win_size > 1):
            self.merge_layer = SegMergingNew(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(SUMformer_AD_layer(seg_num, factor, d_model, n_heads, \
                                                         d_ff, dropout, index=index))

    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x


class up_block(nn.Module):
    '''
    TVF block
    '''

    def __init__(self, up_win_size, d_model, n_heads, d_ff, depth, dropout, \
                 seg_num=10, factor=10, index=0):
        super(up_block, self).__init__()
        if (up_win_size > 1):
            self.split_layer = SegSplitNew(d_model, up_win_size, nn.LayerNorm)
        else:
            self.split_layer = None

        # self.merge_layer = SegMerging(d_model, right_win_size, nn.LayerNorm)

        self.encode_layers = nn.ModuleList()
        self.win_size = up_win_size
        # self.concat_projection = nn.Linear(d_model*2,d_model)
        # self.reduction = nn.Linear(d_model*2,d_model)
        self.reduction = nn.Sequential(nn.Linear(d_model*2, d_model*8),
                      nn.GELU(),
                      nn.Linear(d_model*8, d_model*2),
                      nn.GELU(),nn.Linear(d_model*2, d_model))

        self.norm = nn.LayerNorm(d_model)

        for i in range(depth):
            self.encode_layers.append(
                SUMformer_AD_layer(seg_num, factor, d_model, n_heads, \
                                   d_ff, dropout, index=index))

    def forward(self, x_down,x_left): # left's seg_num is win_size times longer than down
        _, ts_dim, _, _ = x_down.shape

        if x_down.shape[2]!=x_left.shape[2]:
            x_down = self.split_layer(x_down)

        x = torch.cat([x_down,x_left],dim=-1)

        # x = self.concat_projection(x)
        x = self.reduction(x)
        x = self.norm(x)
        for layer in self.encode_layers:
            x = layer(x)

        return x

class Encoder(nn.Module):
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout,
                in_seg_num = 10, factor=10):
        super(Encoder, self).__init__()
        self.e_blocks = e_blocks
        self.encode_merge_blocks = nn.ModuleList()

        self.encode_merge_blocks.append(down_block(1, d_model, n_heads, d_ff, block_depth, dropout,\
                                            in_seg_num, factor))
        for i in range(1, e_blocks):
            self.encode_merge_blocks.append(down_block(win_size, d_model*2**i, n_heads, d_ff, block_depth, dropout,\
                                            ceil(in_seg_num/win_size**i), factor,index=i))

        self.encode_split_blocks = nn.ModuleList()

        for i in range(1,e_blocks):
            self.encode_split_blocks.append(up_block(win_size, d_model*2**(e_blocks-i-1), n_heads, d_ff, block_depth, dropout,\
                                            ceil(in_seg_num/win_size**(e_blocks-1))*win_size**i, factor,index=i))

    def forward(self, origin_x):
        encode_x_down = []
        encode_x_down.append(origin_x)
        x = origin_x
        for block in self.encode_merge_blocks:
            x = block(x)
            encode_x_down.append(x)

        encode_x_up = []
        encode_x_up.append(encode_x_down[-1])
        x = self.encode_split_blocks[0](encode_x_down[-1],encode_x_down[-2])

        encode_x_up.append(x)

        for i in range(1,self.e_blocks-1):
            x = self.encode_split_blocks[i](x,encode_x_down[-i-2])
            encode_x_up.append(x)

        encode_x_up.append(origin_x)

        encode_x_up.reverse()
        return encode_x_up