from typing import Tuple

import gym
import numpy as np
import torch

import util
from qlbs.interface import Policy
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

    def reset(self) -> Tuple[State, Info]:
        self.mutate_parameters()
        GBM, BM = geometricBM(self.initial_asset_price, self.max_time, 1, self.info.mu, self.info.sigma, self.info._dt)
        self._normalized_price = BM[0, :]
        self._standard_price = GBM[0, :]
        self.portfolio_value = util.payoff_of_option(self.is_call_option, self._standard_price[-1],
                                                     self.info.strike_price)
        self.current_time = self.max_time - 1

        return self._describe(), self.info

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
            self.info.r = np.clip(self.info.r * (1 + 0.1 * np.random.randn(1)[0]), 0, 2e-2)
        if np.random.rand(1) < self.mutation:
            self.info.mu = np.clip(self.info.mu + 1e-3 * np.random.randn(1)[0] - 1 * self.info.mu, -1e-3, 1e-3)
        if np.random.rand(1) < self.mutation:
            self.info.sigma = np.clip(self.info.sigma * (1 + 0.1 * np.random.randn(1)[0]), 0, 2e-1)
        if np.random.rand(1) < self.mutation:
            self.info.strike_price = np.clip(
                self.info.strike_price + 1e-2 * np.random.randn(1)[0] - 0.1 * (self.info.strike_price - 1), 0.9, 1.1)
