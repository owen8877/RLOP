import numpy as np

import util
from qlbs.interface import Policy, InitialEstimator
from util.pricing import bs_euro_vanilla_call, bs_euro_vanilla_put, delta_hedge_bs_euro_vanilla_put, \
    delta_hedge_bs_euro_vanilla_call


class BSPolicy(Policy):
    def __init__(self, is_call: bool = True):
        super().__init__()
        self.is_call = is_call

    def action(self, state, info):
        S = util.normalized_to_standard_price(state.normalized_asset_price, info.mu, info.sigma,
                                              state.remaining_time, info._dt)
        K = info.strike_price
        return (delta_hedge_bs_euro_vanilla_call if self.is_call else delta_hedge_bs_euro_vanilla_put)(
            S, K, state.remaining_time, info.r, info.sigma, info._dt)

    def batch_action(self, state_info_tensor):
        """
        :param state_info_tensor: [[normal_price, strike_price, r, mu, sigma, remaining_real_time, risk_lambda]]
        :return:
        """
        S = state_info_tensor[:, 0]
        K = state_info_tensor[:, 1]
        r = state_info_tensor[:, 2]
        # mu = state_info_tensor[:, 3]
        sigma = state_info_tensor[:, 4]
        remaining_real_time = state_info_tensor[:, 5]
        return (delta_hedge_bs_euro_vanilla_call if self.is_call else delta_hedge_bs_euro_vanilla_put)(
            S, K, remaining_real_time, r, sigma)

    def update(self, delta: np.sctypes, action, state, info, *args):
        raise Exception('BS policy cannot be updated!')


class BSInitialEstimator(InitialEstimator):
    def __call__(self, initial_asset_price: float, strike_price: float, remaining_time: int, r: float,
                 sigma: float, _dt: float) -> float:
        return (bs_euro_vanilla_call if self.is_call_option else bs_euro_vanilla_put)(
            initial_asset_price, strike_price, remaining_time, r, sigma, _dt
        )
