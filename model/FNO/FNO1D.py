import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from ..FedFormer.layers.FourierCorrelation import FourierCrossAttention, FourierBlock
from ..sumformer.Frequency_Enhanced_Net import Frequency_Enhanced_Block, Cross_Frequency_Enhanced_Block
from math import ceil
from einops import rearrange, repeat

# class FNO1d(nn.Module):
#     def __init__(self,C,H,W,in_len,out_len):
#         super(FNO1d, self).__init__()
#         self.data_dim = C*H*W
#         self.in_len = in_len
#         self.out_len = out_len
#         self.f_layers = 4
#         # self.net = FourierBlock(data_dim, data_dim, in_len, modes=modes, num_head=num_head, mode_select_method='lower')
#         self.frequency_net = nn.ModuleList([FourierBlock(self.data_dim, self.data_dim, in_len, modes=32, num_head=8, mode_select_method='lower')
#                                             for a in range(self.f_layers)])  # output shape: series, not patch
#         # output shape: series, not patch
#         self.frequency_linear = nn.Linear(in_len, out_len)
#
#         # self.frequency_net = FourierBlock(2048,2048,in_len,32,mode_select_method='lower') #INPUT (B L H E )
#
#     def forward(self, x_seq, mode="train", frozen=False):
#
#         B, T, C, H, W = x_seq.shape
#         x_seq = x_seq.reshape(B, T, C * H * W)
#
#         # Frequency Enhanced Net
#         fre_in = x_seq
#         fre_out = list()
#         for i in range(self.f_layers):
#             fre_in = self.frequency_net[i](fre_in)
#             fre_out.append(fre_in)
#
#
#         predict_y = self.frequency_linear(fre_out[-1].transpose(1, -1)).transpose(1, -1)
#
#         return predict_y.reshape(B, self.out_len, C, H, W)

# class SpectralConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
#         super(SpectralConv3d, self).__init__()
#
#         """
#         3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
#         """
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
#         self.modes2 = modes2
#         self.modes3 = modes3
#
#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
#                                     dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
#                                     dtype=torch.cfloat))
#         self.weights3 = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
#                                     dtype=torch.cfloat))
#         self.weights4 = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
#                                     dtype=torch.cfloat))
#
#     # Complex multiplication
#     def compl_mul3d(self, input, weights):
#         # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
#         return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
#
#     def forward(self, x):
#         batchsize = x.shape[0]
#         # Compute Fourier coeffcients up to factor of e^(- something constant)
#         x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
#
#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
#                              dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
#         out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
#         out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
#         out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
#             self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
#
#         # Return to physical space
#         x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
#         return x
#
#
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO1d(nn.Module):
    def __init__(self, C,H,W,in_len,out_len):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = in_len//4
        self.width = C*H*W
        self.data_dim = self.width
         # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        # self.conv0 = SpectralConv1d(self.data_dim,in_len)
        # self.conv1 = SpectralConv1d(self.data_dim, in_len)
        # self.conv2 = SpectralConv1d(self.data_dim, in_len)
        # self.conv3 = SpectralConv1d(self.data_dim, in_len)
        self.conv0 = SpectralConv1dT(self.width, self.width//16, self.modes1)
        self.conv1 = SpectralConv1dT(self.width, self.width//16, self.modes1)
        self.conv2 = SpectralConv1dT(self.width, self.width//16, self.modes1)
        self.conv3 = SpectralConv1dT(self.width, self.width//16, self.modes1)

        self.mlp0 = MLP(self.width//16, self.width, self.width//16)
        self.mlp1 = MLP(self.width//16, self.width, self.width//16)
        self.mlp2 = MLP(self.width//16, self.width, self.width//16)
        self.mlp3 = MLP(self.width//16, self.width, self.width//16)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, self.width, self.width//16)  # output channel is 1: u(x, y)
        self.predict_linear = nn.Linear(in_len,out_len)
        self.out_len = out_len

    def forward(self, x):
        # x shape B T C H W
        B,T,C,H,W = x.shape
        x = x.permute(0,2,3,4,1) # B C H W T
        x = x.reshape(B,C*H*W,T)
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = self.q(x)
        x = self.predict_linear(x)
        x = x.reshape(B,C,H,W,self.out_len).permute(0,4,1,2,3)  # pad the domain if input is non-periodic
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels,in_len):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.head_num = 8

        self.frequency_cross = FourierBlock(in_channels, in_channels, in_len, modes=32, mode_select_method='lower')


    def forward(self,frq_in_q):
        "input : b d l "

        frq_in_q = rearrange(frq_in_q, 'b (h d) l ->b l h d', h=self.head_num)
        # frq_in_k = self.K_projection(frq_send)  # (b ts_dim (H L))
        # frq_in_k = rearrange(frq_in_k, 'b d (h l) ->b l h d',h=self.cross_head)

        frq_out,_ = self.frequency_cross(frq_in_q,0,0,0)  # input B L H E output B H E L
        # B H E L -> B E (H L) --linear-- B E (H L) -- B ts_d seg_num d_model
        # frequency_cross output get through: linear projection to transfer time domain to patch domain,droupout,res,norm,mlp,droupout,res,norm
        frq_out = rearrange(frq_out, 'b h e l -> b (h e) l')

        return frq_out

class SpectralConv1dT(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1dT, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        # out_ft[:, :, :self.modes1] =x_ft[:, :, :self.modes1]
        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x