import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat,reduce

from .FourierCorrelation import FourierCrossAttention,Cross_grid_fourierAttention,FourierBlock,SpectralConv3d
from math import ceil
import torch.nn.functional as F
from einops import rearrange, repeat
import math
class Frequency_Enhanced_Block(nn.Module):
    def __init__(self, data_dim,in_len,modes=32,depth=4,num_head=1,moving_avg=128,d_values = None):
        super(Frequency_Enhanced_Block, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.modes = 32
        self.depth = depth
        self.num_head = num_head
        self.d_values = d_values or (data_dim//num_head)
        self.net = FourierBlock(data_dim,data_dim,in_len,modes=modes,num_head=num_head,mode_select_method='lower')
        self.dropout =nn.Dropout(0.05)
        self.activation = F.gelu
        self.in_projection = nn.Linear(data_dim,num_head*self.d_values)
        self.out_projection = nn.Linear(num_head*self.d_values,data_dim)
        # self.norm = nn.LayerNorm(data_dim)
        self.norm = nn.BatchNorm1d(in_len)
        self.moving_avg = moving_avg
        # self.decomp1 = series_decomp_multi(self.moving_avg)
        # self.decomp2 = series_decomp_multi(self.moving_avg)
        self.decomp1 = series_decomp(self.moving_avg)
        self.decomp2 = series_decomp(self.moving_avg)
        self.conv1 = nn.Conv1d(in_channels=data_dim,out_channels=data_dim,kernel_size=1,bias=False)
        self.conv2 = nn.Conv1d(in_channels=data_dim,out_channels=data_dim,kernel_size=1,bias=False)


    def forward(self,x):
        # x shape(B,T,CHW)
        fre_in = self.in_projection(x)
        fre_in = rearrange(fre_in,'b l (h d) -> b l h d',h =self.num_head)

        fre_out = self.net(fre_in) #input shape (b l h d) output shape(b h d l)

        fre_out = rearrange(fre_out,'b h d l -> b l (h d)')
        fre_out = self.out_projection(fre_out)

        x = x+self.dropout(fre_out)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # res, _ = self.decomp2(x + y)
        res = x+y


        return self.norm(res)



class Frequency_Enhanced_Block3d(nn.Module):
    def __init__(self, data_dim,in_len,modes=32,depth=4,num_head=1,moving_avg=128,d_values = None):
        super(Frequency_Enhanced_Block3d, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.modes = 32
        self.depth = depth
        self.num_head = num_head
        self.d_values = d_values or (data_dim//num_head)
        self.net = SpectralConv3d(2,2,32,32,32)
        self.dropout =nn.Dropout(0.05)
        self.activation = F.gelu
        self.in_projection = nn.Linear(data_dim,num_head*self.d_values)
        self.out_projection = nn.Linear(num_head*self.d_values,data_dim)
        # self.norm = nn.LayerNorm(data_dim)
        self.norm = nn.BatchNorm1d(in_len)
        self.moving_avg = moving_avg
        # self.decomp1 = series_decomp_multi(self.moving_avg)
        # self.decomp2 = series_decomp_multi(self.moving_avg)
        self.decomp1 = series_decomp(self.moving_avg)
        self.decomp2 = series_decomp(self.moving_avg)
        self.conv1 = nn.Conv1d(in_channels=data_dim,out_channels=data_dim,kernel_size=1,bias=False)
        self.conv2 = nn.Conv1d(in_channels=data_dim,out_channels=data_dim,kernel_size=1,bias=False)


    def forward(self,x):
        # x shape(B,T,CHW)
        B,T,_ = x.shape
        fre_in = x
        fre_in = fre_in.reshape(B,T,2,32,32)
        fre_in = fre_in.permute(0,2,3,4,1)


        # input shape B C H W T
        fre_out = self.net(fre_in)

        fre_out = rearrange(fre_out,'b c h w t -> b t (c h w)')
        fre_out = self.out_projection(fre_out)

        x = x+self.dropout(fre_out)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # res, _ = self.decomp2(x + y)
        res = x+y


        return self.norm(res)


class Cross_Frequency_Enhanced_Block(nn.Module):
    def __init__(self, data_dim,in_len,modes=32,depth=4,num_head=1,moving_avg=128,d_values = None):
        super(Cross_Frequency_Enhanced_Block, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.modes = 32
        self.depth = depth
        self.num_head = num_head
        self.d_values = d_values or (data_dim//num_head)
        self.net = FourierCrossAttention(data_dim,data_dim,in_len,in_len,modes=modes,head_num=num_head,mode_select_method='lower')
        self.dropout =nn.Dropout(0.05)
        self.activation = F.gelu
        self.in_projection_Q = nn.Linear(data_dim,num_head*self.d_values)
        self.in_projection_K = nn.Linear(data_dim, num_head * self.d_values)
        self.out_projection = nn.Linear(num_head*self.d_values,data_dim)
        # self.norm = nn.LayerNorm(data_dim)
        self.norm = nn.BatchNorm1d(in_len)
        self.moving_avg = moving_avg
        # self.decomp1 = series_decomp_multi(self.moving_avg)
        # self.decomp2 = series_decomp_multi(self.moving_avg)
        self.decomp1 = series_decomp(self.moving_avg)
        self.decomp2 = series_decomp(self.moving_avg)
        self.conv1 = nn.Conv1d(in_channels=data_dim,out_channels=data_dim,kernel_size=1,bias=False)
        self.conv2 = nn.Conv1d(in_channels=data_dim,out_channels=data_dim,kernel_size=1,bias=False)


    def forward(self,x):
        # x shape(B,T,CHW)
        fre_in_q = self.in_projection_Q(x)
        fre_in_k = self.in_projection_K(x)
        fre_in_q = rearrange(fre_in_q,'b l (h d) -> b l h d',h =self.num_head)
        fre_in_k = rearrange(fre_in_k, 'b l (h d) -> b l h d', h=self.num_head)

        fre_out = self.net(fre_in_q,fre_in_k) #input shape (b l h d) output shape(b h d l)

        fre_out = rearrange(fre_out,'b h d l -> b l (h d)')
        fre_out = self.out_projection(fre_out)

        x = x+self.dropout(fre_out)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # res, _ = self.decomp2(x + y)
        res = x+y

        return self.norm(res)


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean
