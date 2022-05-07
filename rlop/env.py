from typing import Optional, Tuple, Iterable, Callable, Union
from unittest import TestCase

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from gym.envs.registration import register

import util
from rlop.interface import InitialEstimator
from util.sample import geometricBM


class Info:
    def __init__(self, strike_price: np.sctypes, r: np.sctypes, mu: np.sctypes, sigma: np.sctypes, _dt: np.sctypes,
                 friction: float):
        self.strike_price = strike_price
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self._dt = _dt
        self.friction = friction


class State:
    def __init__(self, normalized_asset_price: np.sctypes, remaining_step: int, portfolio_value: np.ndarray):
        self.normalized_asset_price = normalized_asset_price
        self.remaining_step = remaining_step  # a.k.a. rt
        self.portfolio_value = portfolio_value  # shape=(rt,)

    def to_tensors(self, info: Info):
        return [torch.tensor([
            self.normalized_asset_price,
            (t + 1) * info._dt,
            self.portfolio_value[t],
            util.standard_to_normalized_price(info.strike_price, info.mu, info.sigma, t + 1, info._dt),
            info.r,
            info.mu,
            info.sigma,
            info.friction,
        ]).float() for t in range(self.remaining_step)]


class SamplePool:
    def __init__(self, max_time: int, size: int = 20):
        self.size = size
        self.max_time = max_time
        self.counter = size
        self.gBM = np.empty((self.size, self.max_time + 1))
        self.BM = np.empty((self.size, self.max_time + 1))

    def update(self, initial_asset_price, mu, sigma, _dt):
        self.initial_asset_price = initial_asset_price
        self.mu = mu
        self.sigma = sigma
        self._dt = _dt

    def next(self):
        if self.counter >= self.size:
            self.counter = 0
            geometricBM(self.initial_asset_price, self.max_time, self.size, self.mu, self.sigma, self._dt,
                        gBM_out=self.gBM, BM_out=self.BM)
        GBM = self.gBM[self.counter, :]
        BM = self.BM[self.counter, :]
        self.counter += 1
        return GBM, BM


class RLOPEnv(gym.Env):
    def __init__(self, is_call_option: bool, strike_price: np.sctypes, max_step: int, mu: np.sctypes, sigma: np.sctypes,
                 r: np.sctypes, friction: float, initial_estimator: InitialEstimator, initial_asset_price: np.sctypes,
                 *, mutation: Union[float, Callable] = 0.01, _dt: float = 1):
        # Environment constants, or stay constant for quite a long time
        self.is_call_option = is_call_option
        self.max_step = max_step
        self.initial_estimator = initial_estimator
        self.initial_asset_price = initial_asset_price
        self.info = Info(strike_price, r, mu, sigma, _dt, friction)
        self.mutation = mutation
        self._dt = _dt
        self.sample_pool = SamplePool(self.max_step)

        # Episodic variables
        self.current_time = max_step
        self.portfolio_value_history = np.zeros((self.max_step + 1, self.max_step))
        self.old_action = np.zeros(self.max_step)

        # For rendering
        self.fig, self.axs = None, None

    def __init_render__(self):
        self.fig, self.axs = plt.subplots(2, 1, figsize=(7, 5))

    def describe_state(self):
        return State(
            normalized_asset_price=self._normalized_price[self.current_time],
            remaining_step=self.max_step - self.current_time,
            portfolio_value=self.portfolio_value,
        )

    def step(self, action):
        """

        :param action: An array, where the t-th entry represents the hedge position for the portfolio with terminal time t
        :return:
        """
        tau = self.current_time
        t = tau + 1
        S_tau, S_t = self._standard_price[tau], self._standard_price[t]

        # compute the cash position
        cash_tau = self.portfolio_value - action * S_tau
        cash_t = cash_tau * np.exp(self.info.r * self._dt)
        portfolio_value_t = cash_t + action * S_t - self.info.friction * np.abs(action - self.old_action) * S_t

        payoff = util.payoff_of_option(self.is_call_option, S_t, self.info.strike_price)
        reward = -np.abs(payoff - portfolio_value_t[0])
        # reward = -(payoff - portfolio_value_t[0]) ** 2

        self.portfolio_value = portfolio_value_t[1:]
        self.old_action = action[1:]
        self.portfolio_value_history[t, t - 1:] = portfolio_value_t
        self.current_time += 1
        done = self.current_time == self.max_step

        return self.describe_state(), reward, done, self.info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        self.mutate_parameters()
        self.sample_pool.update(self.initial_asset_price, self.info.mu, self.info.sigma, self._dt)
        GBM, BM = self.sample_pool.next()
        self._normalized_price = BM
        self._standard_price = GBM
        self.current_time = 0
        self.portfolio_value = np.array([
            self.initial_estimator(self.initial_asset_price, self.info.strike_price, t + 1, self.info.r,
                                   self.info.sigma, self._dt) for t in range(self.max_step)])
        self.portfolio_value_history[0, :] = self.portfolio_value
        self.old_action = np.zeros(self.max_step)

        return self.describe_state(), self.info

    def render(self, mode="human", axs=None):
        if axs is not None and isinstance(axs, Iterable):
            ax_stock, ax_option = axs
        else:
            if self.fig is None:
                self.__init_render__()
            ax_stock, ax_option = self.axs
        tau = self.current_time
        T = self.max_step

        ax_stock.cla()
        ax_stock.plot(np.arange(0, tau + 1), self._standard_price[0:tau + 1])
        ax_stock.set(xlim=[0, T], xlabel='time', ylabel='asset price')

        ax_option.cla()
        palette = sns.color_palette("hls", T)
        for i in range(T):
            if i < tau:
                payoff = util.payoff_of_option(self.is_call_option, self._standard_price[i + 1], self.info.strike_price)
                ax_option.plot(np.arange(0, i + 2), self.portfolio_value_history[0:i + 2, i], label=f'#{i:d}',
                               c=palette[i])
                ax_option.plot(i + 1, payoff, marker='+', c=palette[i])
            else:
                ax_option.plot(np.arange(0, tau + 1), self.portfolio_value_history[0:tau + 1, i], label=f'#{i:d}',
                               c=palette[i])
        ax_option.set(xlim=[0, T], xlabel='time', ylabel='asset price')

        plt.show(block=False)
        plt.pause(0.0001)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)

    def mutate_parameters(self):
        if callable(self.mutation):
            self.mutation(self)
            return

        if np.random.rand(1) < self.mutation:
            self.info.r = np.clip(self.info.r * (1 + 0.1 * np.random.randn(1)[0]), 0, 2e-3)
        if np.random.rand(1) < self.mutation:
            self.info.mu = np.clip(self.info.mu + 1e-3 * np.random.randn(1)[0] - 1 * self.info.mu, -1e-3, 1e-3)
        if np.random.rand(1) < self.mutation:
            self.info.sigma = np.clip(self.info.sigma * (1 + 0.1 * np.random.randn(1)[0]), 0, 2e-2)
        if np.random.rand(1) < self.mutation:
            self.info.strike_price = np.clip(
                self.info.strike_price + 1e-2 * np.random.randn(1)[0] - 0.1 * (self.info.strike_price - 1), 0.9, 1.1)


register(id='RLOP-v0', entry_point='rlop.env:RLOPEnv')


class Test(TestCase):
    def test_step(self):
        import matplotlib as mpl
        from rlop.rl import BSInitialEstimator
        mpl.use('TkAgg')
        env = RLOPEnv(is_call_option=True, strike_price=100, max_step=5, mu=0.5e-2, sigma=1e-2, r=0.025e-2, friction=0,
                      initial_estimator=BSInitialEstimator(True), initial_asset_price=100)

        while True:
            (state, info), done = env.reset(), False
            env.render()

            while not done:
                action = np.random.rand(state.remaining_time)
                state, reward, done, info = env.step(action)
                env.render()
