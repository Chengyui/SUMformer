import math

import torch
import torch.nn as nn
from einops import rearrange, repeat,reduce

from .sum_encoder import Encoder
from .sum_embed import DSW_embedding
from math import ceil

class Sumformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3,
                dropout=0.0, baseline = False, device=torch.device('cuda:0'),layer_scaler=1):
        super(Sumformer, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.d_layers = 0

        self.baseline = baseline

        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 5, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor,layer_scaler=layer_scaler)

        self.predict_linear = nn.ModuleList([nn.Linear(ceil(in_len / (seg_len * win_size ** (a)))*d_model,self.out_len) for a in range(e_layers)])
        self.dropout = nn.Dropout(0.01)

        
    def forward(self, x_seq,mode="train",frozen=False):
        """
        mode:train,pretrain+finetune
        finetune:stage 1: Frozen backbone, train Linear
        stage 2: train end2end
        """
        B,T,C,H,W = x_seq.shape
        x_seq = x_seq.reshape(B,T,C*H*W)
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        if mode=="pretrain":
            sample_index,mask_inputs,predict_mask = self.density_sampling(x_seq)
            x_seq = rearrange(x_seq,'b (seg_num seg_len) d -> b seg_num d seg_len',seg_len = self.seg_len)
            mask = torch.zeros_like(x_seq,dtype=bool)
            mask.scatter_(2,sample_index.unsqueeze(-1).expand(-1,-1,-1,x_seq.shape[-1]),1)
            x_seq[mask] = 0 #mask必须与x_seq相同形状
            x_seq = rearrange(x_seq, 'b seg_num d seg_len -> b (seg_num seg_len) d')
            x_seq = self.enc_value_embedding(x_seq)
        elif mode=="train":
            x_seq = self.enc_value_embedding(x_seq)

        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        if frozen:
            with torch.no_grad():
                enc_out = self.encoder(x_seq)
        else:
            enc_out = self.encoder(x_seq)
        # enc_out = enc_out[-self.d_layers:]
        if self.d_layers>0:
            dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
            predict_y = self.decoder(dec_in, enc_out)
        elif self.d_layers==0:
            # enc_out = enc_out[-1]
            # predict_y = self.predict_linear(enc_out)
            # predict_y = rearrange(predict_y, 'b out_d seg_num (seg_len) -> b (seg_num seg_len) out_d')
            res = []
            if mode=="pretrain":
                for idx,mod in enumerate(self.pretrain_linear):
                    predict_y = mod(enc_out[idx+1])
                    predict_y = rearrange(predict_y, 'b out_d seg_num (seg_len) -> b (seg_num seg_len) out_d')
                    res.append(predict_y)
                predict_y = reduce(
                    rearrange(res, 'list b t d -> list b t d'),
                    'list b t d -> b t d', 'mean'
                )
            else:
                for idx,mod in enumerate(self.predict_linear):
                    enc_in = rearrange(enc_out[idx+1],'b d seg_num d_model -> b d (seg_num d_model)')
                    enc_in = self.dropout(enc_in)
                    predict_y = mod(enc_in)
                    predict_y = rearrange(predict_y, 'b out_d t -> b t out_d')
                    res.append(predict_y)
                predict_y = reduce(
                    rearrange(res, 'list b t d -> list b t d'),
                    'list b t d -> b t d', 'mean')


        else:
            enc_out = enc_out[-1]
            predict_y = self.predict_linear[-1](enc_out)
            predict_y = rearrange(predict_y, 'b out_d seg_num (seg_len) -> b (seg_num seg_len) out_d')

        predict_y = base + predict_y[:, :self.out_len, :]
        # predict_y = predict_y.reshape(B, self.out_len, C, H, W)
        if mode=="pretrain":
            mask_predict = rearrange(predict_y,'b (seg_num seg_len) out_d-> b seg_num out_d seg_len',seg_len=self.seg_len)
            mask_predict = mask_predict[predict_mask]

            return mask_inputs,mask_predict
        else:
            predict_y = predict_y.reshape(B, self.out_len, C, H, W)
            return predict_y

    def density_sampling(self,x_seq,mode="softmax"):
        """
        mode:
        1. uniform
        2.
        3. softmax
        """
        B, T, CHW = x_seq.shape

        x_seq = rearrange(x_seq,'b (seg_num seg_len) d -> b seg_num d seg_len',seg_len=self.seg_len)
        density = torch.sum(x_seq,dim=3) # b,seg_num,d



        density = rearrange(density,'b seg_num d -> (b seg_num) d')
        if mode=="uniform":
            val = 1/density.shape[-1]
            sample_index = torch.multinomial(torch.ones_like(density)*val, int(0.4 * density.shape[-1]),
                                             replacement=False)
        elif mode=="density":
            sample_index = torch.multinomial(density/density.sum(dim=-1),int(0.4*density.shape[-1]),replacement=False)
        elif mode=="softmax":
            sample_index = torch.multinomial(torch.softmax(density,dim=-1), int(0.4 * density.shape[-1]),
                                             replacement=False)
        else:
            raise "choose uniform density or softmax"

        sample_index = rearrange(sample_index, '(b seg_num) d -> b seg_num d', b=B)

        mask = torch.zeros_like(x_seq, dtype=bool)
        mask.scatter_(2, sample_index.unsqueeze(-1).expand(-1, -1, -1, x_seq.shape[-1]), 1)
        mask_inputs = x_seq[mask]
        return sample_index,mask_inputs,mask

