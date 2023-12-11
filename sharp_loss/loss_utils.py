import numpy as np
from numba import njit


@njit(cache=True)
def numba_apply_along_axis(func1d, axis, arr):
    """ credits to @joelrich : https://github.com/joelrich """
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.zeros(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.zeros(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@njit(cache=True)
def numba_max_along_axis(array, axis):
    return numba_apply_along_axis(np.max, axis, array)


@njit(cache=True)
def numba_sum_along_axis(array, axis):
    return numba_apply_along_axis(np.sum, axis, array)


@njit(cache=True)
def numba_softmax(x, gamma):
    max_x = numba_max_along_axis(x, 1).reshape(-1, 1)
    exp_x = np.exp((x - max_x) / gamma)
    Z = numba_sum_along_axis(exp_x, 1).reshape(-1, 1)
    return gamma * np.log(Z.reshape(-1)) + max_x.reshape(-1), exp_x / Z


@njit(cache=True)
def numba_min(x, gamma):
    min_x, argmax_x = numba_softmax(-x, gamma)
    return -min_x, argmax_x


@njit(cache=True)
def numba_max_hessian_product(p, z, gamma):
    b, c, _ = p.shape
    interm_sum = numba_sum_along_axis((p * z).reshape(-1, 3), 1)
    sum_recalculated = p * interm_sum.reshape(b, c, 1)
    return (p * z - sum_recalculated) / gamma


@njit(cache=True)
def numba_min_hessian_product(p, z, gamma):
    return -numba_max_hessian_product(p, z, gamma)
