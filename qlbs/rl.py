from typing import Tuple
from unittest import TestCase

import gym
import numpy as np
import torch

import util
from qlbs.interface import Policy, InitialEstimator
from util.pricing import bs_euro_vanilla_call, bs_euro_vanilla_put, delta_hedge_bs_euro_vanilla_put, \
    delta_hedge_bs_euro_vanilla_call
from util.sample import geometricBM


class Info:
    def __init__(self, strike_price: float, r: float, mu: float, sigma: float, _dt: float):
        self.strike_price = strike_price
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self._dt = _dt


class State:
    def __init__(self, normalized_asset_price: float, remaining_time: int):
        self.normalized_asset_price = normalized_asset_price
        self.remaining_time = remaining_time

    def to_tensor(self, info: Info):
        return torch.tensor((
            self.normalized_asset_price,
            self.remaining_time * info._dt,
            util.standard_to_normalized_price(info.strike_price, info.mu, info.sigma, self.remaining_time, info._dt),
            info.r,
            info.mu,
            info.sigma
        ))


class HedgeEnv:
    def __init__(self, remaining_till, is_call_option, _dt: float = 1):
        self.state = State(0, 0)
        self.remaining_till = remaining_till
        self.is_call_option = is_call_option
        self._dt = _dt

    def reset(self, info: Info, asset_normalized_prices: np.ndarray, asset_standard_prices: np.ndarray):
        self.info = info
        self.gamma = np.exp(-self.info.r * self._dt)
        self.asset_normalized_prices = asset_normalized_prices
        self.asset_standard_prices = asset_standard_prices

        self.portfolio_value = util.payoff_of_option(self.is_call_option, self.asset_standard_prices[-1],
                                                     self.info.strike_price)
        self.state.remaining_time = 1
        self.state.normalized_asset_price = self.asset_normalized_prices[-2]
        return self.state

    def step(self, hedge) -> Tuple[State, float, float, bool]:
        rt = self.state.remaining_time
        dS = self.asset_standard_prices[-rt] - self.asset_standard_prices[-rt - 1] / self.gamma
        in_position_change = self.gamma * hedge * dS
        self.portfolio_value *= self.gamma
        self.portfolio_value -= in_position_change

        self.state.remaining_time = rt + 1
        done = self.state.remaining_time > self.remaining_till
        if not done:
            self.state.normalized_asset_price = self.asset_normalized_prices[-rt - 2]
        return self.state, self.portfolio_value, in_position_change, done


class QLBSEnv(gym.Env):
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass


class BSPolicy(Policy):
    def __init__(self, is_call: bool = True):
        super().__init__()
        # self.option_func = bs_euro_vanilla_call if is_call else bs_euro_vanilla_put
        self.hedge_func = delta_hedge_bs_euro_vanilla_call if is_call else delta_hedge_bs_euro_vanilla_put

    def action(self, state, info):
        S = util.normalized_to_standard_price(state.normalized_asset_price, info.mu, info.sigma,
                                              state.remaining_time, info._dt)
        K = info.strike_price
        return self.hedge_func(S, K, state.remaining_time, info.r, info.sigma, info._dt)

    def update(self, delta: np.sctypes, action, state, info, *args):
        raise Exception('BS policy cannot be updated!')


class BSInitialEstimator(InitialEstimator):
    def __call__(self, initial_asset_price: float, strike_price: float, remaining_time: int, r: float,
                 sigma: float, _dt: float) -> float:
        return (bs_euro_vanilla_call if self.is_call_option else bs_euro_vanilla_put)(
            initial_asset_price, strike_price, remaining_time, r, sigma, _dt
        )


class Test(TestCase):
    def test_hedge_env(self):
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        from tqdm import trange
        import seaborn as sns
        import pandas as pd
        mpl.use('TkAgg')
        sns.set_style('whitegrid')

        is_call_option = True
        r = 0e-3
        mu = 0e-3
        sigma = 5e-3
        initial_price = 1
        strike_price = 1.001
        T = 10
        _dt = 0.01

        max_time = int(np.round(T / _dt))
        env = HedgeEnv(remaining_till=max_time, is_call_option=is_call_option)
        bs_pi = BSPolicy(is_call=is_call_option)
        bs_estimator = BSInitialEstimator(is_call_option)

        initial_errors = []
        linf_errors = []
        for _ in trange(10):
            standard_prices, normalized_prices = geometricBM(initial_price, max_time, 1, mu, sigma, _dt)
            standard_prices = standard_prices[0, :]
            normalized_prices = normalized_prices[0, :]
            info = Info(strike_price=strike_price, r=r, mu=mu, sigma=sigma, _dt=_dt)
            state = env.reset(info, normalized_prices, standard_prices)
            done = False

            bs_option_prices = np.array(
                [bs_estimator(standard_prices[t], strike_price, max_time - t, r, sigma, _dt) for t in
                 range(max_time + 1)])

            pvs = np.zeros(max_time + 1)
            hedges = np.zeros(max_time + 1)
            pvs[-1] = util.payoff_of_option(is_call_option, standard_prices[-1], strike_price)
            while not done:
                hedge = bs_pi.action(state, info)
                state, pv, in_position_change, done = env.step(hedge)
                pvs[-state.remaining_time] = pv
                hedges[-state.remaining_time] = hedge

            initial_errors.append(pvs[0] - bs_option_prices[0])
            linf_errors.append(np.linalg.norm(pvs - bs_option_prices, ord=np.inf))

            fig, (ax_price, ax_option, ax_hedge) = plt.subplots(3, 1, figsize=(4, 5))
            times = np.arange(0, max_time + 1)
            ax_price.plot(times, standard_prices)
            ax_price.set(ylabel='stock price')
            ax_option.plot(times, pvs, ls='--', label='portfolio')
            ax_option.plot(times, bs_option_prices, label='bs price')
            ax_option.legend(loc='best')
            ax_hedge.plot(times, hedges)

            plt.show(block=True)

        sns.histplot(pd.DataFrame({'initial': initial_errors, 'inf': linf_errors}))
        plt.show(block=True)
