import numpy as np

import lib.util
from .env import Policy, Baseline
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
        S = lib.util.normalized_to_standard_price(
            state.normalized_asset_price, info.mu, info.sigma, state.passed_step, info._dt
        )
        K = info.strike_price
        return (delta_hedge_bs_euro_vanilla_call if self.is_call else delta_hedge_bs_euro_vanilla_put)(
            S, K, state.remaining_step, info.r, info.sigma, info._dt
        )

    def batch_action(self, state_info_tensor, random: bool = True):
        """
        :param state_info_tensor: [[normal_price, passed_real_time, remaining_real_time, normal_strike_price, r, mu,
            sigma, risk_lambda]]
        :return:
        """
        passed_real_time = state_info_tensor[:, 1]
        remaining_real_time = state_info_tensor[:, 2]
        r = state_info_tensor[:, 4]
        mu = state_info_tensor[:, 5]
        sigma = state_info_tensor[:, 6]
        S = lib.util.normalized_to_standard_price(state_info_tensor[:, 0], mu, sigma, passed_real_time, 1)
        K = lib.util.normalized_to_standard_price(
            state_info_tensor[:, 3], mu, sigma, passed_real_time + remaining_real_time, 1
        )
        return (delta_hedge_bs_euro_vanilla_call if self.is_call else delta_hedge_bs_euro_vanilla_put)(
            S, K, remaining_real_time, r, sigma
        )

    def update(self, delta: float, action, state, info, *args):
        raise Exception("BS policy cannot be updated!")


class BSBaseline(Baseline):
    def __init__(self, is_call: bool):
        super().__init__()
        self.is_call = is_call

    def __call__(self, state, info):
        S = lib.util.normalized_to_standard_price(
            state.normalized_asset_price, info.mu, info.sigma, state.passed_step, info._dt
        )
        K = info.strike_price
        return (bs_euro_vanilla_call if self.is_call else bs_euro_vanilla_put)(
            S, K, state.remaining_step, info.r, info.sigma, info._dt
        )

    def batch_estimate(self, state_info_tensor):
        """
        :param state_info_tensor: [[normal_price, passed_real_time, remaining_real_time, normal_strike_price, r, mu,
            sigma, risk_lambda]]
        :return:
        """
        passed_real_time = state_info_tensor[:, 1]
        remaining_real_time = state_info_tensor[:, 2]
        r = state_info_tensor[:, 4]
        mu = state_info_tensor[:, 5]
        sigma = state_info_tensor[:, 6]
        S = lib.util.normalized_to_standard_price(state_info_tensor[:, 0], mu, sigma, passed_real_time, 1)
        K = lib.util.normalized_to_standard_price(
            state_info_tensor[:, 3], mu, sigma, passed_real_time + remaining_real_time, 1
        )
        return (bs_euro_vanilla_call if self.is_call else bs_euro_vanilla_put)(S, K, remaining_real_time, r, sigma, 1)

    def update(self, G: float, state, info):
        raise Exception("BS baseline cannot be updated!")


class BSInitialEstimator:
    def __init__(self, is_call_option: bool):
        self.is_call_option = is_call_option

    def __call__(
        self, initial_asset_price: float, strike_price: float, remaining_time: int, r: float, sigma: float, _dt: float
    ) -> float:
        return (bs_euro_vanilla_call if self.is_call_option else bs_euro_vanilla_put)(
            initial_asset_price, strike_price, remaining_time, r, sigma, _dt
        )
