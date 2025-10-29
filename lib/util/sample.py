from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np


def geometricBM(
    S0: float, steps_n: int, samples_n: int, mu: float, sigma: float, _dt: float, gBM_out=None, BM_out=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    generate one trajectory of geometric brownian motion with N steps with time step T
    :param steps_n: total steps
    :param samples_n: number of trajectories
    :param mu: drift term
    :param sigma: volatility term
    :return: a numpy array of length N+1
    s_t = s_0 * exp( (mu - sigma^2 / 2) t + sigma * W_t)
    """
    if BM_out is None:
        BM_out = np.empty((samples_n, steps_n + 1), dtype=float)
    if gBM_out is None:
        gBM_out = np.empty((samples_n, steps_n + 1), dtype=float)

    # generate the BM trajectories first
    r = np.random.randn(samples_n, steps_n) * np.sqrt(_dt)
    BM_out[:, 0] = 0
    np.cumsum(r, axis=1, out=BM_out[:, 1:])
    BM_out += np.log(S0) / sigma

    adjusted_mu = mu - sigma**2 / 2
    t_arr = np.broadcast_to(np.arange(steps_n + 1)[np.newaxis, :] * _dt, (samples_n, steps_n + 1))
    gBM_out[:, :] = np.exp(adjusted_mu * t_arr + sigma * BM_out)

    return gBM_out, BM_out


def geometricBM_parallel(
    S0: np.ndarray, parallel_n: int, steps_n: int, samples_n: int, mu: np.ndarray, sigma: np.ndarray, _dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    generate one trajectory of geometric brownian motion with N steps with time step T
    :param steps_n: total steps
    :param samples_n: number of trajectories
    :param mu: drift term
    :param sigma: volatility term
    :return: a numpy array of length N+1
    s_t = s_0 * exp( (mu - sigma^2 / 2) t + sigma * W_t)
    """

    SHAPE = (parallel_n, steps_n + 1, samples_n)
    BM_out = np.empty(SHAPE, dtype=float)
    gBM_out = np.empty(SHAPE, dtype=float)

    S0 = np.broadcast_to(S0[:, None, None], SHAPE)
    sigma = np.broadcast_to(sigma[:, None, None], SHAPE)
    mu = np.broadcast_to(mu[:, None, None], SHAPE)

    # generate the BM trajectories first
    r = np.random.randn(parallel_n, steps_n, samples_n) * np.sqrt(_dt)
    BM_out[:, 0, :] = 0
    np.cumsum(r, axis=1, out=BM_out[:, 1:, :])
    BM_out[:, :, :] += np.log(S0) / sigma

    adjusted_mu = mu - sigma**2 / 2
    t_arr = np.broadcast_to(np.arange(steps_n + 1)[np.newaxis, :, np.newaxis] * _dt, SHAPE)
    gBM_out[:, :, :] = np.exp(adjusted_mu * t_arr + sigma * BM_out)

    return gBM_out, BM_out


def discrete_OU_process(
    decay: float, jump: float | None = None, std: float | None = None, mu: float = 0.00, log10: bool = False
):
    if jump is None:
        assert std is not None
        jump = std * np.sqrt(2 * decay)

    def helper(x: float):
        if log10:
            x = np.log10(x)
        y = x + (mu - x) * decay + jump * np.random.randn(1)[0]
        if log10:
            y = np.pow(10.0, y)
        return y

    return helper
