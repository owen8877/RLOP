import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import matplotlib as mpl
from util.timer import Timer

import seaborn as sns

mpl.use('TkAgg')
sns.set_style('whitegrid')


def geometricBM(S0: float, steps_n: int, dt: float, samples_n: int, mu: float, sigma: float):
    # generate the BM trajectories first
    r = np.random.randn(samples_n, steps_n) * np.sqrt(dt)
    BM = np.empty((samples_n, steps_n + 1), dtype=float)
    BM[:, 0] = 0
    np.cumsum(r, axis=1, out=BM[:, 1:])

    adjusted_mu = mu - sigma ** 2 / 2
    t_arr = np.broadcast_to(np.arange(steps_n + 1)[np.newaxis, :] * dt, (samples_n, steps_n + 1))
    gBM = S0 * np.exp(adjusted_mu * t_arr + sigma * BM)

    return BM, gBM


def bs_euro_vanilla_call(S, K, T, r, sigma):
    dplus = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    dminus = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * si.norm.cdf(dplus, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(dminus, 0.0, 1.0))
    return call


def bs_euro_call_hedge(S, K, T, r, sigma):
    dplus = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = si.norm.cdf(dplus, 0, 1)
    return delta


S0 = 1
strike_price = 1
mu = 0e-3
r = 0e-3
sigma = 1e-2

T = 10  # terminal time in real time scale
dt = 0.01

steps_n = int(np.round(T / dt))
samples_n = 1000

t_arr = np.arange(steps_n + 1) * dt
t_arr_broad = np.broadcast_to(t_arr[np.newaxis, :], (samples_n, steps_n + 1))
X, S = geometricBM(S0, steps_n, dt, samples_n, mu, sigma)
bs_prices = bs_euro_vanilla_call(S, strike_price, T - t_arr_broad, r, sigma)
hedge = bs_euro_call_hedge(S, strike_price, T - t_arr_broad, r, sigma)[:, :-1]

value_change = hedge * (S[:, 1:] - S[:, :-1])
portfolio_value = np.empty((samples_n, steps_n + 1), dtype=float)
portfolio_value[:, 0] = 0
np.cumsum(value_change, axis=1, out=portfolio_value[:, 1:])
portfolio_value += bs_prices[:, :1]

# compare path-by-path bs price and portfolio value
for i in range(samples_n):
    fig, (ax_price, ax_option, ax_hedge) = plt.subplots(3, 1, figsize=(5, 5))
    ax_price.plot(t_arr, S[i, :])
    ax_option.plot(t_arr, bs_prices[i, :], '--')
    ax_option.plot(t_arr, portfolio_value[i, :], '-')
    ax_hedge.plot(t_arr[:-1], hedge[i, :])
    plt.show(block=True)

fig, ax = plt.subplots()
end_error = bs_prices[:, -1] - portfolio_value[:, -1]
sns.histplot(end_error, ax=ax)
plt.show(block=True)
