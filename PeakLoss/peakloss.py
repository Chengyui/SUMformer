
import torch
import torch.nn as nn
from utils.metrics import MAE, RMSE, SMAPE, MSE
class peak_loss(nn.Module):
    def __init__(self, win_len,spatial=False):
        super(peak_loss, self).__init__()
        self.win_len = win_len
        self.avg_pool = nn.MaxPool1d(win_len, stride=win_len)
        self.spatial = spatial
        self.criterion = nn.MSELoss()

    def forward(self, output,target,Eval=False):
        #when eval,the input is list of torch tensor
        B, T, C, H, W = output.size()


        # reshape the input tensor into AvgPooling1D
        time_output = output.permute(0,2,3,4,1)
        time_output = time_output.reshape(B,C*H*W,T)
        # Employ the avgpooling on temporal dimension of output
        time_output = self.avg_pool(time_output)

        time_target = target.permute(0, 2, 3, 4, 1)
        time_target = time_target.reshape(B, C * H * W, T)
        # Employ the avgpooling on temporal dimension of target
        time_target = self.avg_pool(time_target)

        time_loss = self.criterion(time_output,time_target)

        if self.spatial:
            topk_value ,topk_indices = torch.topk(output.view(B,T,C,-1),k=H*W//10,dim=3)
            target_value = torch.gather(target.view(B,T,C,-1),index=topk_indices,dim=3)
            spatial_loss = self.criterion(topk_value,target_value)
            time_loss +=spatial_loss
        if Eval:
            return time_output,time_target,topk_value,target_value

        return time_loss

if __name__ == '__main__':
    peak = peak_loss(16)
    inputs = torch.rand((1,128,2,32,32))
    output = peak(inputs)