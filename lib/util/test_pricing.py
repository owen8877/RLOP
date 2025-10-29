from unittest import TestCase

import numpy as np
import scipy.stats as si

from .sample import geometricBM_parallel

from .pricing import bs_euro_vanilla_call, bs_euro_vanilla_put


class Test(TestCase):
    def test_call(self):
        a = bs_euro_vanilla_call(100, 120, 1, 0.05, 0.25)
        print(a)

    def test_put(self):
        b = bs_euro_vanilla_put(120, 120, 1, 0.05, 0.25)
        print(b)

    def test_zero_handling(self):
        rows = 5
        cols = 5
        S = 1
        r = 0.1
        sigma = 2
        K = 3

        T_arr = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if np.random.randn(1)[0] < 0.5:
                    T_arr[i, j] = 0
                else:
                    T_arr[i, j] = np.random.randint(5, 10, 1)

        call_price_vectorized = bs_euro_vanilla_call(S, K, T_arr, r, sigma)
        call_price_ref = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if T_arr[i, j] == 0:
                    call_price_ref[i, j] = max(S - K, 0)
                else:
                    call_price_ref[i, j] = bs_euro_vanilla_call(S, K, T_arr[i, j], r, sigma)
        self.assertTrue(np.all(np.isclose(call_price_vectorized, call_price_ref)), "Price result doesn't match!")

    def test_inverse(self):
        S = 100
        K = 120
        T = 1
        r = 0.05
        sigma = 0.25

        call_price = bs_euro_vanilla_call(S, K, T, r, sigma)
        # sigma_recovered = bs_euro_vanilla_solve_volatility(call_price, K, T, r, sigma)
        # print(S)
        # print(call_price)
        # print(sigma_recovered)
        # self.assertTrue(np.isclose(S, sigma_recovered), "Inverse BS formula failed to recover S!")

    def test_geometricBM_parallel(self):
        parallel_N = 3
        S = np.array([10] * parallel_N)
        mu = np.array([0.1] * parallel_N)
        sigma = np.array([0.1] * parallel_N)
        a, _ = geometricBM_parallel(S, parallel_N, 50, 10, mu, sigma, 0.1)
        x = np.arange(51) * 0.1

        import matplotlib.pyplot as plt

        for i in range(parallel_N):
            for j in range(10):
                plt.plot(x, a[i, :, j])

        plt.show()
