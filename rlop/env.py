from typing import Optional
from unittest import TestCase

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.envs.registration import register

import util

NAP = 'normalized_asset_price'
SP = 'strike_price'
RT = 'remaining_time'
Mu = 'mu'
Sigma = 'sigma'
PV = 'portfolio_value'


class RLOPEnv(gym.Env):
    def __init__(self, is_call_option: bool, strike_price: np.sctypes, max_time: int, mu: np.sctypes, sigma: np.sctypes,
                 gamma: np.sctypes, initial_portfolio_value: np.ndarray):
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(max_time,))
        self.observation_space = spaces.Dict({
            NAP: spaces.Box(low=-np.inf, high=np.inf, shape=(1, )),
            SP: spaces.Box(low=0, high=np.inf, shape=(1,)),
            RT: spaces.Box(low=1, high=max_time, shape=(1,)),
            Mu: spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            Sigma: spaces.Box(low=0, high=np.inf, shape=(1,)),
            PV: spaces.Box(low=-np.inf, high=np.inf, shape=(max_time,))
        })

        # Environment constants, or stay constant for quite a long time
        self.is_call_option = is_call_option
        self.strike_price = strike_price
        self.mu = mu
        self.sigma = sigma
        self.max_time = max_time
        self.gamma = gamma
        self.initial_portfolio_value = initial_portfolio_value

        # Episodic variables
        self.current_time = max_time
        self.portfolio_value_history = np.zeros((self.max_time + 1, self.max_time))

        # For rendering
        self.fig, self.axs = plt.subplots(2, 1, figsize=(7, 5))

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
        cash_t = cash_tau / self.gamma
        portfolio_value_t = cash_t + action * S_t

        payoff = util.payoff_of_option(self.is_call_option, S_t, self.strike_price)
        reward = -(payoff - portfolio_value_t[self.current_time - 1]) ** 2

        self.portfolio_value = portfolio_value_t
        self.portfolio_value_history[t, :] = self.portfolio_value
        self.current_time += 1
        done = self.current_time == self.max_time

        return {
                   NAP: self._normalized_price[self.current_time],
                   SP: self.strike_price,
                   RT: self.max_time - self.current_time,
                   Mu: self.mu,
                   Sigma: self.sigma,
                   PV: self.portfolio_value,
               }, reward, done, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        self._normalized_price = np.random.randn(self.max_time + 1).cumsum()
        self._standard_price = np.random.randint(10, 20) * util.normalized_to_standard_price(
            self._normalized_price, self.mu, self.sigma, np.arange(0, self.max_time + 1))
        self.current_time = 0
        self.portfolio_value = self.initial_portfolio_value
        self.portfolio_value_history[0, :] = self.portfolio_value

        return {
            NAP: self._normalized_price[self.current_time],
            SP: self.strike_price,
            RT: self.max_time - self.current_time,
            Mu: self.mu,
            Sigma: self.sigma,
            PV: self.portfolio_value,
        }

    def alter_initial_portfolio_value(self, initial_portfolio_value: np.ndarray):
        self.initial_portfolio_value = initial_portfolio_value

    def render(self, mode="human"):
        ax_stock, ax_option = self.axs
        tau = self.current_time
        T = self.max_time

        ax_stock.cla()
        ax_stock.plot(np.arange(0, tau + 1), self._standard_price[0:tau + 1])
        ax_stock.set(xlim=[0, T], xlabel='time', ylabel='asset price')

        ax_option.cla()
        for i in range(T):
            if i < tau:
                payoff = util.payoff_of_option(self.is_call_option, self._standard_price[i+1], self.strike_price)
                ax_option.plot(np.arange(0, i + 2), self.portfolio_value_history[0:i + 2, i], label=f'#{i:d}')
                ax_option.plot(i + 1, payoff, marker='+')
            else:
                ax_option.plot(np.arange(0, tau + 1), self.portfolio_value_history[0:tau + 1, i], label=f'#{i:d}')
        ax_option.set(xlim=[0, T], xlabel='time', ylabel='asset price')

        self.fig.canvas.draw_idle()
        plt.show(block=False)

    def close(self):
        pass


register(id='RLOP-v0', entry_point='rlop.env:RLOPEnv')


class Test(TestCase):
    def test_step(self):
        import matplotlib as mpl
        mpl.use('TkAgg')
        env = RLOPEnv(is_call_option=True, strike_price=11, max_time=5, mu=.1, sigma=1, gamma=0.99,
                      initial_portfolio_value=np.array([1, 2, 3, 4, 5]))

        while True:
            state, done = env.reset(), False
            env.render()

            while not done:
                action = np.random.rand(5)
                state, reward, done, _ = env.step(action)
                env.render()
