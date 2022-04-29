from typing import Tuple
from unittest import TestCase

import numpy as np
from gym import Env
from matplotlib import pyplot as plt
from tqdm import trange

import util
from qlbs.env import State, Info, QLBSEnv
from qlbs.interface import Policy
from util.sample import geometricBM


def policy_evaluation(env: Env, pi: Policy, episode_n: int, *, plot: bool = False):
    t0_returns = []
    t0_risks = []
    fig, ax = plt.subplots()
    pbar = trange(episode_n)
    for e in pbar:
        states = []
        actions = []
        rewards = []
        risks = []

        (state, info), done = env.reset(), False
        states.append(state)
        while not done:
            action = pi.action(state, info)
            state, reward, done, additional = env.step(action, pi)
            if e == episode_n - 1 and plot:
                env.render()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            risks.append(additional['risk'])

        t0_returns.append(np.sum(rewards))
        t0_risks.append(env.info.risk_lambda * np.dot(risks, np.power(env.gamma, np.arange(len(risks))[::-1])))
        pbar.set_description(
            f't0_return={t0_returns[-1]:.2e};t0_risks={t0_risks[-1]:.2e};r={info.r:.4f};mu={info.mu:.4f};sigma={info.sigma:.4f};K={info.strike_price:.4f}')
        if (e + 1) % 100 == 0 and plot:
            indices = np.arange(0, e + 1)
            ax.cla()
            option_price = np.array(t0_returns) * (-1)
            ax.plot(indices, option_price)
            ax.plot(indices, option_price + np.array(t0_risks), ls=':')
            ax.plot(indices, option_price - np.array(t0_risks), ls=':')
            ax.plot(indices, np.cumsum(t0_returns) / (1 + indices) * (-1), ls='--')
            ax.set(yscale='linear', ylabel='negative reward (=option price)')
            plt.show(block=False)
            plt.pause(0.001)
    return t0_returns


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
        from qlbs.bs import BSInitialEstimator, BSPolicy
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
        import matplotlib as mpl
        import seaborn as sns
        from qlbs.bs import BSPolicy
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
                      _dt=_dt, mutation=0)
        bs_pi = BSPolicy(is_call=is_call_option)

        policy_evaluation(env, bs_pi, episode_n=1000, plot=True)
        plt.show()
