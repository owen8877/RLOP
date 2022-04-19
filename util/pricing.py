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
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call


def bs_euro_vanilla_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    return put


def optimal_hedging_position(S, K, T, r, sigma, option_func, dS=1e-6):
    dC = option_func(S + dS, K, T, r, sigma) - option_func(S, K, T, r, sigma)
    return dC / dS


class Test(TestCase):
    def test_call(self):
        a = bs_euro_vanilla_call(100, 120, 1, 0.05, 0.25)
        print(a)

    def test_put(self):
        b = bs_euro_vanilla_put(120, 120, 1, 0.05, 0.25)
        print(b)
