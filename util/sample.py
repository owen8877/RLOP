from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np


def geometricBM(S0: float, steps_n: int, samples_n: int, mu: float, sigma: float, _dt: float, gBM_out=None,
                BM_out=None):
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

    adjusted_mu = mu - sigma ** 2 / 2
    t_arr = np.broadcast_to(np.arange(steps_n + 1)[np.newaxis, :] * _dt, (samples_n, steps_n + 1))
    gBM_out[:, :] = S0 * np.exp(adjusted_mu * t_arr + sigma * BM_out)

    return gBM_out, BM_out


class Test(TestCase):
    def test_geometricBM(self):
        a = geometricBM(10, 20, 50, 0.2, 0.1, 1)
        x = np.arange(21)

        for i in range(50):
            y = a[i, :]
            plt.plot(x, y)

        plt.show()
