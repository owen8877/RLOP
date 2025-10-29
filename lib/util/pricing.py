from unittest import TestCase

import numpy as np
import scipy.stats as si
import torch


def bs_euro_vanilla_call_torch(S, K, T, r, sigma, _dt=1.0):
    T0_mask = np.isclose(T, 0)
    T_clean = T * _dt
    T_clean[T0_mask] = 1

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_clean) / (sigma * np.sqrt(T_clean))
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T_clean) / (sigma * np.sqrt(T_clean))
    call = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T_clean) * si.norm.cdf(d2, 0.0, 1.0)
    S_mask = S[T0_mask]
    K_mask = K[T0_mask]
    call[T0_mask] = torch.clip(S_mask - K_mask, 0).double()
    return call


def bs_euro_vanilla_call_numpy(S, K, T, r, sigma, _dt=1.0):
    # Wrap the T variable as an ndarray if it's not
    if not isinstance(T, np.ndarray):
        T = np.array([T])
        is_T_scalar = True
    else:
        is_T_scalar = False

    # Mask the maturity matrix at entries with value 0
    T0_mask = np.isclose(T, 0)
    T_clean = T.copy() * _dt
    T_clean[T0_mask] = 1

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_clean) / (sigma * np.sqrt(T_clean))
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T_clean) / (sigma * np.sqrt(T_clean))
    call = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T_clean) * si.norm.cdf(d2, 0.0, 1.0)
    S_mask = S[T0_mask] if isinstance(S, np.ndarray) else S
    K_mask = K[T0_mask] if isinstance(K, np.ndarray) else K
    call[T0_mask] = np.clip(S_mask - K_mask, a_max=None, a_min=0)
    return call[0] if is_T_scalar else call


def bs_euro_vanilla_call(S, K, T, r, sigma, _dt=1.0):
    """
    :param S: current stock price
    :param K: strike price
    :param T: maturity date, counted in the unit of _dt
    :param r: risk-free rate
    :param sigma: volatility
    :return: option price
    """
    if isinstance(S, torch.Tensor):
        return bs_euro_vanilla_call_torch(S, K, T, r, sigma, _dt=_dt)
    else:
        return bs_euro_vanilla_call_numpy(S, K, T, r, sigma, _dt=_dt)


def bs_euro_vanilla_put(S, K, T, r, sigma, _dt=1.0):
    # Wrap the T variable as an ndarray if it's not
    if not isinstance(T, np.ndarray):
        T = np.array([T])
        is_T_scalar = True
    else:
        is_T_scalar = False

    # Mask the maturity matrix at entries with value 0
    T0_mask = np.isclose(T, 0)
    T_clean = T.copy() * _dt
    T_clean[T0_mask] = 1

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T_clean) / (sigma * np.sqrt(T_clean))
    d2 = (np.log(S / K) + (r - 0.5 * sigma * sigma) * T_clean) / (sigma * np.sqrt(T_clean))
    put = K * np.exp(-r * T_clean) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)
    if isinstance(S, np.ndarray):
        put[T0_mask] = np.clip(K - S[T0_mask], a_max=None, a_min=0)
    else:
        put[T0_mask] = np.clip(K - S, a_max=None, a_min=0)
    return put[0] if is_T_scalar else put


def bs_euro_vanilla_vega(S, K, T, r, sigma, _dt=1.0):
    """Vega (derivative of price w.r.t sigma)."""
    T_clean = T.copy() * _dt
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_clean) / (sigma * np.sqrt(T_clean))
    return S * si.norm.pdf(d1) * np.sqrt(T_clean)


def bs_solve_implied_vol_from_call_price(
    price: np.ndarray, S: np.ndarray, K: np.ndarray, T, r, _dt=1.0, tol=1e-3, max_iter=100, sigma_init=0.2
):
    price, S, K, T, r = map(np.asarray, (price, S, K, T, r))
    sigma = np.full_like(price, sigma_init, dtype=float)

    for _ in range(max_iter):
        price_est = bs_euro_vanilla_call(S, K, T, r, sigma)
        vega = bs_euro_vanilla_vega(S, K, T, r, sigma)
        diff = price_est - price
        update = diff / np.maximum(vega, 1e-6)
        sigma_proposed = sigma - np.clip(update, -0.01, 0.01)
        sigma = np.clip(sigma_proposed, 1e-3, 1.0)
        # stop if all converged
        if np.all(np.abs(update) < tol):
            break

    # clean up unrealistic values
    sigma = np.clip(sigma, 1e-3, 1.0)
    return sigma


def delta_hedge_bs_euro_vanilla_call(S, K, T, r, sigma, _dt=1.0):
    if np.any(np.isclose(T, 0)):
        raise Exception("Shall not pass T=0 to delta hedge!")
    T_real = T * _dt
    dplus = (np.log(S / K) + (r + 0.5 * sigma**2) * T_real) / (sigma * np.sqrt(T_real))
    delta = si.norm.cdf(dplus, 0, 1)
    return delta


def delta_hedge_bs_euro_vanilla_put(S, K, T, r, sigma, _dt=1.0):
    return delta_hedge_bs_euro_vanilla_call(S, K, T, r, sigma, _dt=_dt) - 1
