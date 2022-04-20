from unittest import TestCase

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def geometricBM(S0: float, steps_n: int, samples_n: int, mu: float, sigma: float, dt: float):
    """
    generate one trajectory of geometric brownian motion with N steps with time step T
    :param steps_n: total steps
    :param samples_n: number of trajectories
    :param mu: drift term
    :param sigma: volatility term
    :return: a numpy array of length N+1
    s_t = s_0 * exp( (mu - sigma^2 / 2) t + sigma * W_t)
    """

    # generate the BM trajectories first
    r = np.random.randn(samples_n, steps_n) * np.sqrt(dt)
    BM = np.empty((samples_n, steps_n + 1), dtype=float)
    BM[:, 0] = 0
    np.cumsum(r, axis=1, out=BM[:, 1:])

    adjusted_mu = mu - sigma ** 2 / 2
    t_arr = np.broadcast_to(np.arange(steps_n + 1)[np.newaxis, :] * dt, (samples_n, steps_n + 1))
    gBM = S0 * np.exp(adjusted_mu * t_arr + sigma * BM)

    return gBM, BM


class Test(TestCase):
    def test_geometricBM(self):
        a = geometricBM(10, 20, 50, 0.2, 0.1, 1)
        x = np.arange(21)

        for i in range(50):
            y = a[i, :]
            plt.plot(x, y)

        plt.show()
