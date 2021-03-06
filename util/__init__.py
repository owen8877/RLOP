import pickle
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

import util


def _prefix(simplified: bool, plan: str):
    return ('simplified_' if simplified else '') + plan


def standard_to_normalized_price(standard_price: np.ndarray, mu: float, sigma: float, time: np.ndarray, _dt: float):
    return (-(mu - sigma ** 2 / 2) * time * _dt + np.log(standard_price)) / sigma


def normalized_to_standard_price(normalized_price: np.ndarray, mu: float, sigma: float, time: np.ndarray, _dt: float):
    return np.exp(sigma * normalized_price + (mu - sigma ** 2 / 2) * time * _dt)


def payoff_of_option(is_call_option: bool, asset_price: np.sctypes, strike_price: np.sctypes):
    if is_call_option:
        return np.clip(asset_price - strike_price, a_max=None, a_min=0)
    else:
        return np.clip(strike_price - asset_price, a_max=None, a_min=0)


def ensure_dir(path: str, need_strip_end=False):
    if need_strip_end:
        path = '/'.join(path.split('/')[:-1])
    Path(path).mkdir(parents=True, exist_ok=True)


class EMACollector:
    def __init__(self, half_life: float, **kwargs):
        self.l = np.power(0.5, 1 / half_life)
        self.ema_dict = {key: [] for key in kwargs.keys()}
        self.emsq_dict = {key: [] for key in kwargs.keys()}

    def reset(self):
        self.ema_dict = {key: [] for key in self.ema_dict.keys()}
        self.emsq_dict = {key: [] for key in self.emsq_dict.keys()}

    def append(self, **kwargs):
        for key, val in kwargs.items():
            emas = self.ema_dict[key]
            emsqs = self.emsq_dict[key]
            if len(emas) == 0:
                ema = float(val)
                emsq = float(val ** 2)
            else:
                ema = self.l * emas[-1] + float(val) * (1 - self.l)
                emsq = self.l * emsqs[-1] + float(val ** 2) * (1 - self.l)
            emas.append(ema)
            emsqs.append(emsq)

    def plot(self, ax: plt.Axes):
        for (key, emas), emsqs in zip(self.ema_dict.items(), self.emsq_dict.values()):
            # ax.plot(range(len(emas)), np.array(emas), label=key)
            emas = np.array(emas)
            emsqs = np.array(emsqs)
            emstd = np.sqrt(emsqs - emas ** 2) * 2
            ax.plot(range(len(emas)), emas, label=key)
            ax.fill_between(range(len(emas)), emas + emstd, emas - emstd, alpha=0.5)
        ax.legend(loc='best')
        ax.set(xlabel='iteration', yscale='linear')


def abs(arr):
    if isinstance(arr, np.ndarray):
        return np.abs(arr)
    else:
        return torch.abs(arr)
