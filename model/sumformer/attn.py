import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import sys
sys.path.append("../")
from ..MLP.MLPmixer import MLPMixer,DilatMixer
from math import sqrt
from .FourierCorrelation import FourierCrossAttention,Cross_grid_fourierAttention,FourierBlock
from .Frequency_Enhanced_Net import series_decomp,series_decomp_multi
import math
from .AdditiveAttention import AdditiveAttention
from .gconv_standalone import GConv


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous()


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)



class SUMformer_AD_layer(nn.Module):
    '''
    MHSA for TSB
    Neural dictionary for ISSB :https://proceedings.mlr.press/v97/lee19d.html
    LFSB for filter the lower frequency band
    '''

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1, index=0,layer_scaler=1):
        super(SUMformer_AD_layer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.cross_head = 1
        self.seg_num = seg_num
        self.frequency_time_len = 128
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.neural_dictionary = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.frequency_cross = FourierBlock(400,400,self.frequency_time_len,modes=self.frequency_time_len//4,
                                                     num_head=self.cross_head,mode_select_method='lower')

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP3 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.time_projection = nn.Linear(seg_num*d_model,self.frequency_time_len)
        self.frequency_projection = nn.Linear(self.frequency_time_len, seg_num * d_model)
        self.layer_scale = layer_scaler
        self.FNO1 = True  # When in taxibj, it is False.

    def forward(self, x):
        # TSB
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc*self.layer_scale)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in)*self.layer_scale)
        dim_in = self.norm2(dim_in)

        # ISSB
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.neural_dictionary, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_out = dim_send + self.dropout(dim_receive*self.layer_scale)
        dim_out = self.norm3(dim_out)
        dim_out = dim_out + self.dropout(self.MLP2(dim_out)*self.layer_scale)
        dim_out = self.norm4(dim_out)


        # LFSB
        dim_out = rearrange(dim_out,'(b seg_num) ts_d d_model -> b ts_d seg_num  d_model',b=batch)
        frq_send = rearrange(dim_out, 'b ts_d seg_num  d_model-> b ts_d (seg_num d_model)', b=batch)

        frq_in_q = frq_send
        frq_in_q = self.time_projection(frq_in_q) #project seg_num*d_model to original time length

        frq_in_q = rearrange(frq_in_q,'b (h d) l ->b l h d',h=self.cross_head)


        frq_out = self.frequency_cross(frq_in_q)  #input B L H E output B H E L

        frq_out = rearrange(frq_out,'b h e l -> b (h e) l')
        frq_out = self.frequency_projection(frq_out) #b e seg_num*d_model
        frq_out = rearrange(frq_out,'b e (seg_num dmodel) -> b e seg_num dmodel',seg_num=self.seg_num)

        if self.FNO1:
            frq_enc = frq_out
            frq_enc = self.norm5(frq_enc)
            frq_enc = dim_out + self.MLP3(frq_enc)*self.layer_scale
            frq_enc = self.norm6(frq_enc)
        else:
            frq_enc = dim_out + self.dropout(frq_out * self.layer_scale)
            frq_enc = self.norm5(frq_enc)
            frq_enc = frq_enc + self.dropout(self.MLP3(frq_enc) * self.layer_scale)
            frq_enc = self.norm6(frq_enc)

        return frq_enc


class SUMformer_MD_layer(nn.Module):
    '''
    MLPmixer for TSB
    Neural dictionary for ISSB
    LFSB for filter the lower frequency band
    '''

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1, index=0,layer_scaler=1):
        super(SUMformer_MD_layer, self).__init__()
        d_ff = d_ff or 4 * d_model
        # self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.time_mlpmixer = MLPMixer(seg_num, d_model, 1)
        self.cross_head = 8
        self.seg_num = seg_num
        self.frequency_time_len = 128
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.neural_dictionary = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.frequency_cross = FourierBlock(400,400,self.frequency_time_len,modes=32,
                                                     num_head=self.cross_head,mode_select_method='lower')

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP3 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.time_projection = nn.Linear(seg_num*d_model,self.frequency_time_len)
        self.frequency_projection = nn.Linear(self.frequency_time_len, seg_num * d_model)
        self.layer_scale = layer_scaler
        self.FNO1 = True # When in taxibj, it is False.
    def forward(self, x):
        # TSB
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_mlpmixer(
            time_in
        )
        dim_in = time_in + self.dropout(time_enc*self.layer_scale)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in)*self.layer_scale)
        dim_in = self.norm2(dim_in)

        # ISSB
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.neural_dictionary, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_out = dim_send + self.dropout(dim_receive*self.layer_scale)
        dim_out = self.norm3(dim_out)
        dim_out = dim_out + self.dropout(self.MLP2(dim_out)*self.layer_scale)
        dim_out = self.norm4(dim_out)


        #LFSB
        dim_out = rearrange(dim_out, '(b seg_num) ts_d d_model -> b ts_d seg_num  d_model', b=batch)
        frq_send = rearrange(dim_out, 'b ts_d seg_num  d_model-> b ts_d (seg_num d_model)', b=batch)

        frq_in_q = frq_send
        frq_in_q = self.time_projection(frq_in_q) #将seg_num*d_model放缩到原来的时间长度

        frq_in_q = rearrange(frq_in_q,'b (h d) l ->b l h d',h=self.cross_head)

        frq_out = self.frequency_cross(frq_in_q)  #input B L H E output B H E L

        frq_out = rearrange(frq_out,'b h e l -> b (h e) l')
        frq_out = self.frequency_projection(frq_out) #b e seg_num*d_model
        frq_out = rearrange(frq_out,'b e (seg_num dmodel) -> b e seg_num dmodel',seg_num=self.seg_num)

        if self.FNO1:
            frq_enc = frq_out
            frq_enc = self.norm5(frq_enc)
            frq_enc = dim_out + self.MLP3(frq_enc)*self.layer_scale
            frq_enc = self.norm6(frq_enc)
        else:
            frq_enc = dim_out + self.dropout(frq_out * self.layer_scale)
            frq_enc = self.norm5(frq_enc)
            frq_enc = frq_enc + self.dropout(self.MLP3(frq_enc) * self.layer_scale)
            frq_enc = self.norm6(frq_enc)

        return frq_enc



class SUMformer_AL_layer(nn.Module):
    '''
    MHSA for TSB
    Low-rank projection for ISSB: https://arxiv.org/abs/2006.04768
    LFSB for filter the lower frequency band
    '''

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1, index=0,layer_scaler=1):
        super(SUMformer_AL_layer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.cross_head = 8
        self.seg_num = seg_num
        self.frequency_time_len = 128
        self.lin_att = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.k_project = nn.Parameter(init_(torch.zeros(2048, factor)))
        self.frequency_cross = FourierBlock(2048,2048,self.frequency_time_len,modes=self.frequency_time_len//4,
                                                     num_head=self.cross_head,mode_select_method='lower')

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP3 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.time_projection = nn.Linear(seg_num*d_model,self.frequency_time_len)
        self.frequency_projection = nn.Linear(self.frequency_time_len, seg_num * d_model)
        self.layer_scale = layer_scaler
        self.FNO1=True

    def forward(self, x):
        # TSB
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc*self.layer_scale)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in)*self.layer_scale)
        dim_in = self.norm2(dim_in)


        # ISSB
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        dim_k = torch.einsum('bnd,nk->bkd',dim_send,self.k_project)
        dim_receive = self.lin_att(dim_send,dim_k,dim_k)
        dim_out = dim_send + self.dropout(dim_receive*self.layer_scale)
        dim_out = self.norm3(dim_out)
        dim_out = dim_out + self.dropout(self.MLP2(dim_out)*self.layer_scale)
        dim_out = self.norm4(dim_out)


        # LFSB
        dim_out = rearrange(dim_out, '(b seg_num) ts_d d_model -> b ts_d seg_num  d_model', b=batch)
        frq_send = rearrange(dim_out, 'b ts_d seg_num  d_model-> b ts_d (seg_num d_model)', b=batch)

        frq_in_q = frq_send
        frq_in_q = self.time_projection(frq_in_q)

        frq_in_q = rearrange(frq_in_q,'b (h d) l ->b l h d',h=self.cross_head)


        frq_out = self.frequency_cross(frq_in_q)  #input B L H E output B H E L

        frq_out = rearrange(frq_out,'b h e l -> b (h e) l')
        frq_out = self.frequency_projection(frq_out) #b e seg_num*d_model
        frq_out = rearrange(frq_out,'b e (seg_num dmodel) -> b e seg_num dmodel',seg_num=self.seg_num)
        if self.FNO1:
            frq_enc = frq_out
            frq_enc = self.norm5(frq_enc)
            frq_enc = dim_out + self.MLP3(frq_enc)*self.layer_scale
            frq_enc = self.norm6(frq_enc)
        else:
            frq_enc = dim_out + self.dropout(frq_out * self.layer_scale)
            frq_enc = self.norm5(frq_enc)
            frq_enc = frq_enc + self.dropout(self.MLP3(frq_enc) * self.layer_scale)
            frq_enc = self.norm6(frq_enc)


        return frq_enc


class SUMformer_AF_layer(nn.Module):
    '''
    MHSA for TSB
    Full attention with quadratic computation burden for ISSB
    LFSB for filter the lower frequency band
    '''

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1, index=0,layer_scaler=1):
        super(SUMformer_AF_layer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.cross_head = 8
        self.seg_num = seg_num
        self.frequency_time_len = 128
        self.spatial_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.frequency_cross = FourierBlock(2048,2048,self.frequency_time_len,modes=self.frequency_time_len//4,
                                                     num_head=self.cross_head,mode_select_method='lower')

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP3 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.time_projection = nn.Linear(seg_num*d_model,self.frequency_time_len)
        self.frequency_projection = nn.Linear(self.frequency_time_len, seg_num * d_model)
        self.layer_scale = layer_scaler
        self.FNO1 = False

    def forward(self, x):
        # TSB
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc*self.layer_scale)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in)*self.layer_scale)
        dim_in = self.norm2(dim_in)

        # ISSB
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        dim_receive = self.spatial_attention(dim_send, dim_send, dim_send)
        dim_out = dim_send + self.dropout(dim_receive*self.layer_scale)
        dim_out = self.norm3(dim_out)
        dim_out = dim_out + self.dropout(self.MLP2(dim_out)*self.layer_scale)
        dim_out = self.norm4(dim_out)

        #LFSB
        dim_out = rearrange(dim_out, '(b seg_num) ts_d d_model -> b ts_d seg_num  d_model', b=batch)
        frq_send = rearrange(dim_out, 'b ts_d seg_num  d_model-> b ts_d (seg_num d_model)', b=batch)

        frq_in_q = frq_send
        frq_in_q = self.time_projection(frq_in_q) #将seg_num*d_model放缩到原来的时间长度

        frq_in_q = rearrange(frq_in_q,'b (h d) l ->b l h d',h=self.cross_head)


        frq_out = self.frequency_cross(frq_in_q)  #input B L H E output B H E L

        frq_out = rearrange(frq_out,'b h e l -> b (h e) l')
        frq_out = self.frequency_projection(frq_out) #b e seg_num*d_model
        frq_out = rearrange(frq_out,'b e (seg_num dmodel) -> b e seg_num dmodel',seg_num=self.seg_num)

        if self.FNO1:
            frq_enc = frq_out
            frq_enc = self.norm5(frq_enc)
            frq_enc = dim_out + self.MLP3(frq_enc) * self.layer_scale
            frq_enc = self.norm6(frq_enc)
        else:
            frq_enc = dim_out + self.dropout(frq_out * self.layer_scale)
            frq_enc = self.norm5(frq_enc)
            frq_enc = frq_enc + self.dropout(self.MLP3(frq_enc) * self.layer_scale)
            frq_enc = self.norm6(frq_enc)

        return frq_enc

class SUMformer_AA_layer(nn.Module):
    '''
    MHSA for TSB
    Additive attention for ISSB : from https://arxiv.org/abs/2108.09084
    LFSB for filter the lower frequency band
    '''

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1, index=0,layer_scaler=1):
        super(SUMformer_AA_layer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.cross_head = 8
        self.seg_num = seg_num
        self.frequency_time_len = 128
        self.spatial_attention = AdditiveAttention(d_model,n_heads)
        self.frequency_cross = FourierBlock(2048,2048,self.frequency_time_len,modes=self.frequency_time_len//4,
                                                     num_head=self.cross_head,mode_select_method='lower')

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP3 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.time_projection = nn.Linear(seg_num*d_model,self.frequency_time_len)
        self.frequency_projection = nn.Linear(self.frequency_time_len, seg_num * d_model)
        self.layer_scale = layer_scaler

    def forward(self, x):
        # TSB
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc*self.layer_scale)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in)*self.layer_scale)
        dim_in = self.norm2(dim_in)

        # ISSB
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        dim_receive = self.spatial_attention(dim_send)
        dim_out = dim_send + self.dropout(dim_receive*self.layer_scale)
        dim_out = self.norm3(dim_out)
        dim_out = dim_out + self.dropout(self.MLP2(dim_out)*self.layer_scale)
        dim_out = self.norm4(dim_out)



        # LFSB
        dim_out = rearrange(dim_out, '(b seg_num) ts_d d_model -> b ts_d seg_num  d_model', b=batch)
        frq_send = rearrange(dim_out, 'b ts_d seg_num  d_model-> b ts_d (seg_num d_model)', b=batch)

        frq_in_q = frq_send
        frq_in_q = self.time_projection(frq_in_q)

        frq_in_q = rearrange(frq_in_q,'b (h d) l ->b l h d',h=self.cross_head)

        frq_out = self.frequency_cross(frq_in_q)  #input B L H E output B H E L
        frq_out = rearrange(frq_out,'b h e l -> b (h e) l')
        frq_out = self.frequency_projection(frq_out) #b e seg_num*d_model
        frq_out = rearrange(frq_out,'b e (seg_num dmodel) -> b e seg_num dmodel',seg_num=self.seg_num)

        frq_enc = frq_out
        frq_enc = self.norm5(frq_enc)
        frq_enc = dim_out + self.MLP3(frq_enc)*self.layer_scale
        frq_enc = self.norm6(frq_enc)

        return frq_enc

class SUMformer_TS_layer(nn.Module):
    '''
    Mix the Spatial-Temporal tokens together using Neural Dictionary for correlation capture
    LFSB for filter the lower frequency band
    '''

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1, index=0,layer_scaler=1):
        super(SUMformer_TS_layer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_head = 8
        self.seg_num = seg_num
        self.frequency_time_len = 128
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.neural_dictionary = nn.Parameter(torch.randn(seg_num * factor, d_model))
        self.frequency_cross = FourierBlock(2048,2048,self.frequency_time_len,modes=self.frequency_time_len//4,
                                                     num_head=self.cross_head,mode_select_method='lower')

        self.dropout = nn.Dropout(dropout)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)
        self.norm6 = nn.LayerNorm(d_model)

        # self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
        #                           nn.GELU(),
        #                           nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP3 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.time_projection = nn.Linear(seg_num*d_model,self.frequency_time_len)
        self.frequency_projection = nn.Linear(self.frequency_time_len, seg_num * d_model)
        self.layer_scale = layer_scaler
        self.FNO1 = True
    def forward(self, x):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> b (ts_d seg_num) d_model')

        # ISSB
        neural_dictionary = repeat(self.neural_dictionary, 'token_num d_model ->  repeat token_num d_model', repeat=batch)
        dim_buffer = self.dim_sender(neural_dictionary, time_in, time_in)
        dim_receive = self.dim_receiver(time_in, dim_buffer, dim_buffer)
        dim_out = time_in + self.dropout(dim_receive * self.layer_scale)
        dim_out = self.norm3(dim_out)
        dim_out = dim_out + self.dropout(self.MLP2(dim_out) * self.layer_scale)
        dim_out = self.norm4(dim_out)

        dim_out = rearrange(dim_out, 'b (ts_d seg_num) d_model -> b ts_d seg_num d_model', seg_num=self.seg_num)

        # LFSB
        frq_send = rearrange(dim_out, 'b ts_d seg_num d_model -> b ts_d (seg_num d_model)', b=batch)

        frq_in_q = frq_send
        frq_in_q = self.time_projection(frq_in_q)

        frq_in_q = rearrange(frq_in_q, 'b (h d) l ->b l h d', h=self.cross_head)


        frq_out = self.frequency_cross(frq_in_q)  # input B L H E output B H E L

        frq_out = rearrange(frq_out, 'b h e l -> b (h e) l')
        frq_out = self.frequency_projection(frq_out)  # b e seg_num*d_model
        frq_out = rearrange(frq_out, 'b e (seg_num dmodel) -> b e seg_num dmodel', seg_num=self.seg_num)

        if self.FNO1:
            frq_enc = frq_out
            frq_enc = self.norm5(frq_enc)
            frq_enc = dim_out + self.MLP3(frq_enc) * self.layer_scale
            frq_enc = self.norm6(frq_enc)
        else:
            frq_enc = dim_out + self.dropout(frq_out * self.layer_scale)
            frq_enc = self.norm5(frq_enc)
            frq_enc = frq_enc + self.dropout(self.MLP3(frq_enc) * self.layer_scale)
            frq_enc = self.norm6(frq_enc)

        return frq_enc