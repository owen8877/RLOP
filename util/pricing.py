from unittest import TestCase

import numpy as np
import scipy.stats as si


def bs_euro_vanilla_call(S, K, T, r, sigma):
    """
    :param S: current stock price
    :param K: strike price
    :param T: maturity date
    :param r: risk-free rate
    :param sigma: volatility
    :return: option price
    """

    # Wrap the T variable as an ndarray if it's not
    if not isinstance(T, np.ndarray):
        T = np.array([T])
        T_scalar = True
    else:
        T_scalar = False

    # Mask the maturity matrix at entries with value 0
    T0_mask = np.isclose(T, 0)
    T_fake = T.copy()
    T_fake[T0_mask] = 1

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_fake) / (sigma * np.sqrt(T_fake))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T_fake) / (sigma * np.sqrt(T_fake))
    call = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T_fake) * si.norm.cdf(d2, 0.0, 1.0)
    call[T0_mask] = max(S - K, 0)
    return call[0] if T_scalar else call


def bs_euro_vanilla_put(S, K, T, r, sigma):
    # Wrap the T variable as an ndarray if it's not
    if not isinstance(T, np.ndarray):
        T = np.array([T])
        T_scalar = True
    else:
        T_scalar = False

    # Mask the maturity matrix at entries with value 0
    T0_mask = np.isclose(T, 0)
    T_fake = T.copy()
    T_fake[T0_mask] = 1

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T_fake) / (sigma * np.sqrt(T_fake))
    d2 = (np.log(S / K) + (r - 0.5 * sigma * sigma) * T_fake) / (sigma * np.sqrt(T_fake))
    put = K * np.exp(-r * T_fake) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)
    put[T0_mask] = max(K - S, 0)
    return put[0] if T_scalar else put


def delta_hedge_bs_euro_vanilla_call(S, K, T, r, sigma):
    if np.any(np.isclose(T, 0)):
        raise Exception('Shall not pass T=0 to delta hedge!')
    dplus = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = si.norm.cdf(dplus, 0, 1)
    return delta


def delta_hedge_bs_euro_vanilla_put(S, K, T, r, sigma):
    return delta_hedge_bs_euro_vanilla_call(S, K, T, r, sigma) - 1


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
        self.assertTrue(np.all(np.isclose(call_price_vectorized, call_price_ref)), 'Price result doesn\'t match!')
