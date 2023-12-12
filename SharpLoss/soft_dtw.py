import numpy as np
import torch
from numba import njit
from torch.autograd import Function


def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))


def pairwise_distances_with_channels_and_batches(x, y=None):
    batch_size = x.shape[0]
    N_output = x.shape[1]
    x_norm = (x ** 2).sum(2).view(batch_size, -1, 1)
    if y is not None:
        y_t = torch.transpose(y, 1, 2)
        y_norm = (y ** 2).sum(2).view(batch_size, 1, -1)

    dist = x_norm + y_norm - 2.0 * torch.bmm(x.view(batch_size, N_output, 1),
                                             y_t.contiguous().view(batch_size, 1, N_output)).view(batch_size, N_output,
                                                                                                  N_output)

    return torch.clamp(dist, 0.0, float('inf'))


@njit(cache=True)
def compute_softdtw_batch_channel(D, gamma):
    batch_size = D.shape[0]
    num_channels = D.shape[1]
    N = D.shape[2]
    M = D.shape[3]
    R = np.zeros((batch_size, num_channels, N + 2, M + 2), dtype=np.float32) + 1e8
    R[:, :, 0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            r0 = -R[:, :, i - 1, j - 1] / gamma
            r1 = -R[:, :, i - 1, j] / gamma
            r2 = -R[:, :, i, j - 1] / gamma
            rmax = np.maximum(np.maximum(r0, r1), r2)
            rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
            softmin = - gamma * (np.log(rsum) + rmax)
            R[:, :, i, j] = D[:, :, i - 1, j - 1] + softmin

    return R


@njit(cache=True)
def compute_softdtw_backward_batch_channel(D_, R, gamma):
    B = D_.shape[0]
    C = D_.shape[1]
    N = D_.shape[2]
    M = D_.shape[3]
    D = np.zeros((B, C, N + 2, M + 2), dtype=np.float32)
    E = np.zeros((B, C, N + 2, M + 2), dtype=np.float32)

    D[:, :, 1:N + 1, 1:M + 1] = D_
    E[:, :, -1, -1] = 1
    R[:, :, :, -1] = -1e8
    R[:, :, -1, :] = -1e8
    R[:, :, -1, -1] = R[:, :, -2, -2]

    R2 = R[:, :, ::-1, ::-1]
    D2 = D[:, :, ::-1, ::-1]
    E2 = E[:, :, ::-1, ::-1]
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            a0 = (R2[:, :, i - 1, j] - R2[:, :, i, j] - D2[:, :, i - 1, j]) / gamma
            b0 = (R2[:, :, i, j - 1] - R2[:, :, i, j] - D2[:, :, i, j - 1]) / gamma
            c0 = (R2[:, :, i - 1, j - 1] - R2[:, :, i, j] - D2[:, :, i - 1, j - 1]) / gamma
            a = np.exp(a0)
            b = np.exp(b0)
            c = np.exp(c0)
            E2[:, :, i, j] = E2[:, :, i - 1, j] * a + E2[:, :, i, j - 1] * b + E2[:, :, i - 1, j - 1] * c

    E3 = E2[:, :, ::-1, ::-1]

    return E3[:, :, 1:N + 1, 1:M + 1]


class SoftDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma=1.0):
        dev = D.device
        gamma = torch.FloatTensor([gamma]).to(dev)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()

        R = torch.FloatTensor(compute_softdtw_batch_channel(D_[:, :, :, :].astype(np.float32),
                                                            g_)).to(dev)

        total_loss = R[:, :, -2, -2]
        ctx.save_for_backward(D, R, gamma)
        return total_loss

    @staticmethod
    def backward(ctx, grad_output):
        """ accepts ctx as first arg, then as many outputs forward() returned.
        Each argument is the gradient wrt the given output; each returned value should be the gradient wrt the corresponding input
        """

        dev = grad_output.device
        D, R, gamma = ctx.saved_tensors
        batch_size, num_channels, N, N = D.shape
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()

        E = compute_softdtw_backward_batch_channel(D_[:, :, :, :].astype(np.float32),
                                                   R_[:, :, :, :].astype(np.float32),
                                                   g_)

        E = torch.FloatTensor(E).to(dev)

        out = torch.einsum("bc, bcxy-> bcxy", grad_output, E)

        return out, None
