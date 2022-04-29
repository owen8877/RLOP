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
    def __init__(self, strike_price: float, r: float, mu: float, sigma: float, risk_lambda: float, _dt: float):
        self.strike_price = strike_price
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.risk_lambda = risk_lambda
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
            info.sigma,
            info.risk_lambda,
        ))


class QLBSEnv(gym.Env):
    def __init__(self, is_call_option: bool, strike_price: float, max_time: int, mu: float, sigma: float, r: float,
                 risk_lambda: float, initial_asset_price: float, risk_simulation_paths: int, *, mutation: float = 0.1,
                 _dt: float = 1):
        # Setting of the system; stays constant for a long time
        self.is_call_option = is_call_option
        self.info = Info(strike_price=strike_price, r=r, mu=mu, sigma=sigma, risk_lambda=risk_lambda, _dt=_dt)
        self.gamma = np.exp(-r * _dt)
        self.max_time = max_time
        self.initial_asset_price = initial_asset_price
        self.mutation = mutation
        self.risk_simulation_paths = risk_simulation_paths

        # State variables
        self._normalized_price = None
        self._standard_price = None
        self.portfolio_value = 0
        self.current_time = 0

    def _describe(self):
        return State(self._normalized_price[self.current_time], self.max_time - self.current_time)

    def reset(self) -> State:
        self.mutate_parameters()
        GBM, BM = geometricBM(self.initial_asset_price, self.max_time, 1, self.info.mu, self.info.sigma, self.info._dt)
        self._normalized_price = BM[0, :]
        self._standard_price = GBM[0, :]
        self.portfolio_value = util.payoff_of_option(self.is_call_option, self._standard_price[-1],
                                                     self.info.strike_price)
        self.current_time = self.max_time - 1

        return self._describe()

    def step(self, action, pi: Policy) -> Tuple[State, float, bool, dict]:
        t = self.current_time
        S = self._standard_price[t]
        S_new = self._standard_price[t + 1]

        # base reward = change in portfolio value
        base_reward = (self.gamma * S_new - S) * action

        # simulate RS paths to compute Var[Pi_t|F_t]
        RS = self.risk_simulation_paths
        t_arr = np.arange(self.max_time - t + 1) * self.info._dt
        t_arr_broad = np.broadcast_to(t_arr[np.newaxis, :], (RS, len(t_arr)))
        GBM, _ = geometricBM(S, self.max_time - t, RS, self.info.mu, self.info.sigma, self.info._dt)

        # compute hedging positions
        hedge = np.empty((RS, self.max_time - t), dtype=float)
        for s in np.arange(t, self.max_time):
            # now at time s, ask the policy to generate batch actions
            sit = torch.empty((RS, 7))
            sit[:, 0] = torch.tensor(GBM[:, s - t])  # normal_price
            sit[:, 1] = self.info.strike_price  # strike_price
            sit[:, 2] = self.info.r  # r
            sit[:, 3] = self.info.mu  # mu
            sit[:, 4] = self.info.sigma  # sigma
            sit[:, 5] = (self.max_time - t) * self.info._dt  # remaining_real_time
            sit[:, 6] = self.info.risk_lambda  # risk_lambda
            hedge[:, s - t] = pi.batch_action(sit)

        discounted_S = np.exp(-self.info.r * t_arr_broad) * GBM
        value_change = hedge * (discounted_S[:, 1:] - discounted_S[:, :-1])
        tot_value_change = np.sum(value_change, axis=1)
        end_portfolio_value = util.payoff_of_option(self.is_call_option, GBM[:, -1], self.info.strike_price)
        t_portfolio_value = (end_portfolio_value * np.exp(-self.info.r * (self.max_time - t)) - tot_value_change)
        risk = np.std(t_portfolio_value, ddof=1)

        # clean up and return
        self.current_time = t - 1
        done = self.current_time < 0
        reward = base_reward - self.info.risk_lambda * risk
        return self._describe(), reward, done, {'risk': risk}

    def render(self, mode='human'):
        pass

    def mutate_parameters(self):
        if np.random.rand(1) < self.mutation:
            self.info.r = np.clip(self.info.r * (1 + 0.1 * np.random.randn(1)[0]), 0, 2e-3)
        if np.random.rand(1) < self.mutation:
            self.info.mu = np.clip(self.info.mu + 1e-3 * np.random.randn(1)[0] - 1 * self.info.mu, -1e-3, 1e-3)
        if np.random.rand(1) < self.mutation:
            self.info.sigma = np.clip(self.info.sigma * (1 + 0.1 * np.random.randn(1)[0]), 0, 2e-2)
        if np.random.rand(1) < self.mutation:
            self.info.strike_price = np.clip(
                self.info.strike_price + 1e-2 * np.random.randn(1)[0] - 0.1 * (self.info.strike_price - 1), 0.9, 1.1)


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


class Test(TestCase):
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

    def test_hedge_env_bs(self):
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
        risk_lambda = 1
        initial_price = 1
        strike_price = 1.001
        T = 10
        _dt = 0.01

        max_time = int(np.round(T / _dt))
        env = Test.HedgeEnv(remaining_till=max_time, is_call_option=is_call_option)
        bs_pi = BSPolicy(is_call=is_call_option)
        bs_estimator = BSInitialEstimator(is_call_option)

        initial_errors = []
        linf_errors = []
        for _ in trange(10):
            standard_prices, normalized_prices = geometricBM(initial_price, max_time, 1, mu, sigma, _dt)
            standard_prices = standard_prices[0, :]
            normalized_prices = normalized_prices[0, :]
            info = Info(strike_price=strike_price, r=r, mu=mu, sigma=sigma, risk_lambda=risk_lambda, _dt=_dt)
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

    def test_qlbs_env(self):
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        from tqdm import trange
        import seaborn as sns
        import pandas as pd
        mpl.use('TkAgg')
        sns.set_style('whitegrid')

        is_call_option = True
        r = 1e-1
        mu = 0e-3
        sigma = 2e-1
        risk_lambda = 1
        initial_price = 1
        strike_price = 1
        T = 10
        _dt = 1

        max_time = int(np.round(T / _dt))
        env = QLBSEnv(is_call_option=is_call_option, strike_price=strike_price, max_time=max_time, mu=mu, sigma=sigma,
                      r=r, risk_lambda=risk_lambda, initial_asset_price=initial_price, risk_simulation_paths=50,
                      _dt=_dt)
        bs_pi = BSPolicy(is_call=is_call_option)
        bs_estimator = BSInitialEstimator(is_call_option)

        while True:
            state, done = env.reset(), False
            risks = []
            rewards = []
            while not done:
                action = bs_pi.action(state, env.info)
                state, reward, done, additional = env.step(action, bs_pi)
                risks.append(additional['risk'])
                rewards.append(reward)
            print(rewards)
            print(risks)
