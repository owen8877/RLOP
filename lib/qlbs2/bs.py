from dataclasses import replace
from math import isnan
import stat
import numpy as np
import torch

from .env import Policy, Baseline, EnvSetup, QLBSEnv, Info, State
from unittest import TestCase
from lib.util.pricing import (
    bs_euro_vanilla_call,
    bs_euro_vanilla_put,
    delta_hedge_bs_euro_vanilla_put,
    delta_hedge_bs_euro_vanilla_call,
)


class BSPolicy(Policy):
    def __init__(self, is_call: bool = True):
        super().__init__()
        self.is_call = is_call

    def action(self, state, info):
        S = state.spots
        K = info.strike_prices
        _T = 0 * S + state.remaining_step
        _d = (delta_hedge_bs_euro_vanilla_call if self.is_call else delta_hedge_bs_euro_vanilla_put)(
            S, K, _T, info.r, info.sigmas, info._dt
        )
        return torch.tensor(_d)

    def batch_action(self, state_info_tensor: torch.Tensor, random: bool = True) -> torch.Tensor:
        """
        :param state_info_tensor: [[spot, passed_real_time, remaining_real_time, strike, r, mu, sigma, risk_lambda]]
        :return:
        """
        passed_real_time = state_info_tensor[:, 1]
        remaining_real_time = state_info_tensor[:, 2]
        r = state_info_tensor[:, 4]
        mu = state_info_tensor[:, 5]
        sigma = state_info_tensor[:, 6]
        S = state_info_tensor[:, 0]
        K = state_info_tensor[:, 3]

        _d = (delta_hedge_bs_euro_vanilla_call if self.is_call else delta_hedge_bs_euro_vanilla_put)(
            S, K, remaining_real_time, r, sigma
        )
        return torch.tensor(_d)

    def update(self, delta: float, action, state, info, *args):
        raise Exception("BS policy cannot be updated!")


class BSBaseline(Baseline):
    def __init__(self, is_call: bool):
        super().__init__()
        self.is_call = is_call

    def __call__(self, state, info):
        S = state.spots
        K = info.strike_prices
        _T = 0 * S + state.remaining_step
        _d = (bs_euro_vanilla_call if self.is_call else bs_euro_vanilla_put)(S, K, _T, info.r, info.sigmas, info._dt)
        return torch.tensor(_d)

    def batch_estimate(self, state_info_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param state_info_tensor: [[spot, passed_real_time, remaining_real_time, strike, r, mu, sigma, risk_lambda]]
        :return:
        """
        passed_real_time = state_info_tensor[:, 1]
        remaining_real_time = state_info_tensor[:, 2]
        r = state_info_tensor[:, 4]
        mu = state_info_tensor[:, 5]
        sigma = state_info_tensor[:, 6]
        S = state_info_tensor[:, 0]
        K = state_info_tensor[:, 3]

        _d = (bs_euro_vanilla_call if self.is_call else bs_euro_vanilla_put)(S, K, remaining_real_time, r, sigma, 1)

        return _d  # type: ignore

    def update(self, G: float, state, info):
        raise Exception("BS baseline cannot be updated!")
