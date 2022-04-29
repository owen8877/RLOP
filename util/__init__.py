from pathlib import Path

import numpy as np


def standard_to_normalized_price(standard_price: np.ndarray, mu: float, sigma: float, time: np.ndarray, _dt: float):
    return (-(mu - sigma ** 2 / 2) * time * _dt + np.log(standard_price)) / sigma


def normalized_to_standard_price(normalized_price: np.ndarray, mu: float, sigma: float, time: np.ndarray, _dt: float):
    return np.exp(sigma * normalized_price + (mu - sigma ** 2 / 2) * time * _dt)


def payoff_of_option(is_call_option: bool, asset_price: np.sctypes, strike_price: np.sctypes):
    if is_call_option:
        return np.clip(asset_price - strike_price, a_max=None, a_min=0)
    else:
        return np.clip(strike_price - asset_price, a_max=None, a_min=0)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
