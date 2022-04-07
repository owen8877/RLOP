from unittest import TestCase

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def brownian(x0, n, dt, out=None):
    """
    :param x0: float
     the initial value of stock price
    :param n: int
     the number of steps to take
    :param dt: float
     the time step
    :param out: numpy array or None
     If `out` is not None, it specifies the array in which to put the
     result.  If `out` is None, a new numpy array is created and returned.
    :reutrns: a numpy array of length n+1 with the first element x0
    """
    r = norm.rvs(size=n, scale=np.sqrt(dt))
    if out is None:
        out = np.empty(n + 1)

    out[0] = x0

    for i in range(1, n + 1):
        out[i] = out[i - 1] + r[i - 1]

    return out


def geometricBM(x0: np.float, T: int, N: int, mu: np.float, sigma: np.float):
    """
    generate one trajectory of geometric brownian motion with N steps with time step T
    :param T: total steps
    :param N: number of trajectories
    :param mu: drift term
    :param sigma: volatility term
    :return: a numpy array of length N+1
    x_t = x_0 * exp( (mu - sigma^2 / 2) t + sigma * W_t)
    """
    GBM_trajectories = np.zeros([N, T + 1])
    BM_trajectories = np.zeros([N, T + 1])
    for i in range(N):
        samples = brownian(0, T, 1)
        GBM_trajectories[i, :] = x0 * np.exp((mu - sigma * sigma / 2) * np.arange(T + 1) + sigma * samples)
        BM_trajectories[i, :] = samples

    return GBM_trajectories, BM_trajectories


class Test(TestCase):
    def test_BM(self):
        # some tests showing that the function is valid...
        # use matplotlib to plot sampled trajectories
        k = 500
        dt = 0.05
        n = 20
        x = [0]
        bm_final = []
        var = 0

        for j in range(1, n + 1):
            x.append(j * dt)

        for i in range(k):
            bm = brownian(0, n, dt)
            bm_final.append(bm[n])

            bm_list = bm.tolist()
            plt.plot(x, bm_list)

        print(np.average(bm_final))
        print(np.var(bm_final))
        plt.show()

    def test_geometricBM(self):
        a = geometricBM(10, 20, 50, 0.2, 0.1)
        x = np.arange(21)

        for i in range(50):
            y = a[i, :]
            plt.plot(x,y)

        plt.show()




        pass
