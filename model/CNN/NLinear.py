import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, shape,N_T=128,individual=False):
        super(Model, self).__init__()
        self.seq_len = shape[0]
        self.pred_len = N_T
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = shape[1]*shape[2]*shape[3]
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_raw):
        # x_raw (batch,T,Variable,H,W)
        B,T,C,H,W = x_raw.shape
        x = x_raw.reshape(B,T,C*H*W)
        # x: [Batch, Input length, Channel]

        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        x = x.reshape(B,self.pred_len,C,H,W)
        return x # [Batch, Output length, Channel]