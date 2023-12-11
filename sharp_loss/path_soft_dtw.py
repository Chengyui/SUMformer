import numpy as np
import torch
from numba import njit
from torch.autograd import Function

from .loss_utils import numba_min, numba_min_hessian_product


@njit(cache=True)
def dtw_grad(theta, gamma):
    b = theta.shape[0]
    c = theta.shape[1]
    m = theta.shape[2]
    n = theta.shape[3]

    V = np.zeros((b, c, m + 1, n + 1), dtype=np.float32)
    V[:, :, :, 0] = 1e10
    V[:, :, 0, :] = 1e10
    V[:, :, 0, 0] = 0

    Q = np.zeros((b, c, m + 2, n + 2, 3), dtype=np.float32)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            interm_v = np.zeros((b, c, 3), dtype=np.float32)
            interm_v[:, :, 0] = V[:, :, i, j - 1]
            interm_v[:, :, 1] = V[:, :, i - 1, j - 1]
            interm_v[:, :, 2] = V[:, :, i - 1, j]

            # here we use temporary intermediate variables because of numba reshape assignment inside loop has
            # array unification issues.
            # see https://github.com/numba/numba/issues/3931
            interm_v_reshaped = interm_v.reshape((b * c, 3))
            v, out = numba_min(interm_v_reshaped, gamma)
            v_reshaped = v.reshape((b, c))
            out_reshaped = out.reshape((b, c, 3))
            Q[:, :, i, j] = out_reshaped
            V[:, :, i, j] = theta[:, :, i - 1, j - 1] + v_reshaped

    E2 = np.zeros((b, c, m + 2, n + 2), dtype=np.float32)
    E2[:, :, m + 1, :] = 0
    E2[:, :, :, n + 1] = 0
    E2[:, :, m + 1, n + 1] = 1

    Q[:, :, m + 1, n + 1] = 1
    Q2 = Q[:, :, ::-1, ::-1, :]
    E3 = E2[:, :, ::-1, ::-1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            E3[:, :, i, j] = Q2[:, :, i, j - 1, 0] * E3[:, :, i, j - 1] + \
                             Q2[:, :, i - 1, j - 1, 1] * E3[:, :, i - 1, j - 1] + \
                             Q2[:, :, i - 1, j, 2] * E3[:, :, i - 1, j]

    E4 = E3[:, :, ::-1, ::-1]
    return V[:, :, m, n], E4[:, :, 1:m + 1, 1:n + 1], Q, E4


@njit(cache=True)
def dtw_hessian_prod(theta, Z, Q, E, gamma):
    b, num_ch, m, n = Z.shape

    V_dot = np.zeros((b, num_ch, m + 1, n + 1), dtype=np.float32)
    V_dot[:, :, 0, 0] = 0

    Q_dot = np.zeros((b, num_ch, m + 2, n + 2, 3), dtype=np.float32)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            V_dot[:, :, i, j] = Z[:, :, i - 1, j - 1] + \
                                Q[:, :, i, j, 0] * V_dot[:, :, i, j - 1] + \
                                Q[:, :, i, j, 1] * V_dot[:, :, i - 1, j - 1] + \
                                Q[:, :, i, j, 2] * V_dot[:, :, i - 1, j]

            v = np.zeros((b, num_ch, 3))
            v[:, :, 0] = V_dot[:, :, i, j - 1]
            v[:, :, 1] = V_dot[:, :, i - 1, j - 1]
            v[:, :, 2] = V_dot[:, :, i - 1, j]

            Q_dot[:, :, i, j] = numba_min_hessian_product(Q[:, :, i, j], v, gamma)

    E2_dot = np.zeros((b, num_ch, m + 2, n + 2), dtype=np.float32)

    Q2 = Q[:, :, ::-1, ::-1, :]
    E2 = E[:, :, ::-1, ::-1]
    Q2_dot = Q_dot[:, :, ::-1, ::-1, :]
    E3_dot = E2_dot[:, :, ::-1, ::-1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            E3_dot[:, :, i, j] = Q2_dot[:, :, i, j - 1, 0] * E2[:, :, i, j - 1] + \
                                 Q2[:, :, i, j - 1, 0] * E3_dot[:, :, i, j - 1] + \
                                 Q2_dot[:, :, i - 1, j - 1, 1] * E2[:, :, i - 1, j - 1] + \
                                 Q2[:, :, i - 1, j - 1, 1] * E3_dot[:, :, i - 1, j - 1] + \
                                 Q2_dot[:, :, i - 1, j, 2] * E2[:, :, i - 1, j] + \
                                 Q2[:, :, i - 1, j, 2] * E3_dot[:, :, i - 1, j]
    E4_dot = E3_dot[:, :, ::-1, ::-1]

    return V_dot[:, :, m, n], E4_dot[:, :, 1:m + 1, 1:n + 1]


class PathDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma):
        batch_size, num_channels, N, N = D.shape
        device = D.device
        D_cpu = D.detach().cpu().numpy()
        gamma_gpu = torch.FloatTensor([gamma]).to(device)

        grad_gpu = torch.zeros((batch_size, num_channels, N, N)).to(device)
        Q_gpu = torch.zeros((batch_size, num_channels, N + 2, N + 2, 3)).to(device)
        E_gpu = torch.zeros((batch_size, num_channels, N + 2, N + 2)).to(device)

        _, grad_cpu_k, Q_cpu_k, E_cpu_k = dtw_grad(D_cpu[:, :, :, :].astype(np.float32),
                                                   gamma)

        grad_gpu[:, :, :] = torch.FloatTensor(grad_cpu_k).to(device)
        Q_gpu[:, :, :, :] = torch.FloatTensor(Q_cpu_k).to(device)
        E_gpu[:, :, :] = torch.FloatTensor(E_cpu_k).to(device)
        ctx.save_for_backward(grad_gpu, D, Q_gpu, E_gpu, gamma_gpu)

        return grad_gpu

    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        grad_gpu, D_gpu, Q_gpu, E_gpu, gamma = ctx.saved_tensors
        D_cpu = D_gpu.detach().cpu().numpy()
        Q_cpu = Q_gpu.detach().cpu().numpy()

        E_cpu = E_gpu.detach().cpu().numpy()
        gamma = gamma.detach().cpu().numpy()[0]

        Z = grad_output.detach().cpu().numpy()

        batch_size, num_channels, N, N = D_cpu.shape
        Hessian = torch.zeros((batch_size, num_channels, N, N)).to(device)

        _, hess_k = dtw_hessian_prod(D_cpu[:, :, :, :].astype(np.float32),
                                     Z.astype(np.float32),
                                     Q_cpu[:, :, :, :, :].astype(np.float32),
                                     E_cpu[:, :, :, :].astype(np.float32),
                                     gamma)

        Hessian[:, :, :, :] = torch.FloatTensor(hess_k).to(device)

        return Hessian, None
