import torch
from torch import Tensor
from typing import Tuple

from . import path_soft_dtw
from . import soft_dtw
from einops import rearrange, repeat,reduce

class DTWShpTime(torch.nn.Module):
    def __init__(self, alpha: float, gamma: float = 0.01, reduction: str = "mean") -> None:
        """
        Batch-DILATE loss function, a batchwise extension of https://github.com/vincent-leguen/DILATE

        :param alpha: Weight of shape component of the loss versus the temporal component.
        :type alpha: float
        :param gamma: Weight of softmax component of DTW.
        :type gamma: float
        """
        super(DTWShpTime, self).__init__()
        assert 0 <= alpha <= 1
        assert 0 <= gamma <= 1
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tuple[Tensor]:
        """
        Pass through the loss function with input tensor (the prediction) and the target tensor.

        :param input: prediction, shape should be (batch, channels, num_timesteps_outputs). If the input does not match
            the target's shape, it will be broadcasted to it (if possible).
        :type input: torch.Tensor
        :param target: target with same shape as prediction.
        :type target: torch.Tensor
        :return: total_loss, shape_loss, temporal_loss, with first dimensions being the batch and second the channel
        :rtype: tuple
        """
        assert input.device == target.device, f"Device for input and target must be the same, but found {input.device} " \
                                              f"and {target.device}."

        assert input.shape == target.shape, f"Input shape ({input.shape}) and " \
                                            f"target shape ({target.shape}) must match exactly!"
        input = rearrange(input, 'B T C H W -> (B C H W) T 1')
        target = rearrange(target, 'B T C H W -> (B C H W) T 1')
        batch_size, N_channel, N_output = input.shape



        D = soft_dtw.pairwise_distances_with_channels_and_batches(
            target[:, :, :].reshape(batch_size * N_channel, N_output, 1).double(),
            input[:, :, :].reshape(batch_size * N_channel, N_output, 1).double()
        )

        D = D.reshape(batch_size, N_channel, N_output, N_output)

        softdtw_batch = soft_dtw.SoftDTWBatch.apply
        loss_shape = softdtw_batch(D, self.gamma)

        path_dtw = path_soft_dtw.PathDTWBatch.apply
        path = path_dtw(D, self.gamma)

        Omega = soft_dtw.pairwise_distances(torch.arange(1, N_output + 1).view(N_output, 1)).to(target.device)

        Omega = Omega.repeat(N_channel, 1, 1)
        loss_temporal = torch.sum(path * Omega, dim=(2, 3)) / (N_output * N_output)

        if self.reduction == "mean":
            loss_shape = torch.mean(loss_shape, dim=(0, 1))
            loss_temporal = torch.mean(loss_temporal, dim=(0, 1))
        elif self.reduction == "sum":
            loss_shape = torch.sum(loss_shape, dim=(0, 1))
            loss_temporal = torch.sum(loss_temporal, dim=(0, 1))

        loss = self.alpha * loss_shape + (1 - self.alpha) * loss_temporal

        return loss, loss_shape, loss_temporal
