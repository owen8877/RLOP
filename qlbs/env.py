from typing import Tuple

import gym
import numpy as np
import torch

import util
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


class Policy:
    def __init__(self):
        pass

    def action(self, state: State, info: Info):
        raise NotImplementedError

    def batch_action(self, state_info_tensor):
        """
        :param state_info_tensor: [[normal_price, remaining_real_time, strike_price, r, mu, sigma, risk_lambda]]
        :return:
        """
        raise NotImplementedError

    def update(self, delta: float, action: float, state: State, info: Info):
        raise NotImplementedError


class Baseline:
    def __init__(self):
        pass

    def __call__(self, state: State, info: Info):
        raise NotImplementedError

    def update(self, delta: float, state: State, info: Info):
        raise NotImplementedError


class QLBSEnv(gym.Env):
    def __init__(self, is_call_option: bool, strike_price: float, max_step: int, mu: float, sigma: float, r: float,
                 risk_lambda: float, initial_asset_price: float, risk_simulation_paths: int, *, mutation: float = 0.1,
                 _dt: float = 1):
        # Setting of the system; stays constant for a long time
        self.is_call_option = is_call_option
        self.info = Info(strike_price=strike_price, r=r, mu=mu, sigma=sigma, risk_lambda=risk_lambda, _dt=_dt)
        self.gamma = np.exp(-r * _dt)
        self.max_step = max_step
        self.initial_asset_price = initial_asset_price
        self.mutation = mutation
        self.risk_simulation_paths = risk_simulation_paths

        # State variables
        self._normalized_price = None
        self._standard_price = None
        self.current_step = 0

    def _describe(self):
        return State(self._normalized_price[self.current_step], self.max_step - self.current_step)

    def reset(self) -> Tuple[State, Info]:
        self.mutate_parameters()
        GBM, BM = geometricBM(self.initial_asset_price, self.max_step, 1, self.info.mu, self.info.sigma, self.info._dt)
        self._normalized_price = BM[0, :]
        self._standard_price = GBM[0, :]
        self.current_step = 0

        return self._describe(), self.info

    def step(self, action, pi: Policy) -> Tuple[State, float, bool, dict]:
        t = self.current_step

        # simulate RS paths to compute E[Pi_t|F_t], E[Pi_(t+1)|F_(t+1)], and Var[Pi_t|F_t]
        RS = self.risk_simulation_paths
        t_arr = np.arange(self.max_step - t + 1)
        t_arr_broad = np.broadcast_to(t_arr[np.newaxis, :], (RS, len(t_arr)))
        GBM, _ = geometricBM(self._standard_price[t], self.max_step - t, RS, self.info.mu, self.info.sigma,
                             self.info._dt)

        # Compute hedge position in a batch fashion
        sits = []
        for s in np.arange(t, self.max_step):
            sit = torch.empty((RS, 7))
            sit[:, 0] = torch.tensor(GBM[:, s - t])  # normal_price
            sit[:, 1] = (self.max_step - s) * self.info._dt  # remaining_real_time
            sit[:, 2] = self.info.strike_price  # strike_price
            sit[:, 3] = self.info.r  # r
            sit[:, 4] = self.info.mu  # mu
            sit[:, 5] = self.info.sigma  # sigma
            sit[:, 6] = self.info.risk_lambda  # risk_lambda
            sits.append(sit)
        hedge_long = pi.batch_action(torch.cat(sits, dim=0))
        hedge = hedge_long.reshape(RS, self.max_step - t, order='F')
        hedge[:, 0] = action

        discounted_S = np.power(self.gamma, t_arr_broad) * GBM
        discounted_value_change = hedge * (discounted_S[:, 1:] - discounted_S[:, :-1])
        end_value = util.payoff_of_option(self.is_call_option, GBM[:, -1], self.info.strike_price)
        total_value_change = np.sum(discounted_value_change, axis=1)
        t_value = end_value * np.power(self.gamma, self.max_step - t) - total_value_change
        tp1_value = (t_value + discounted_value_change[:, 0]) / self.gamma

        base_reward = self.gamma * (1 - (t + 1) / self.max_step) * np.mean(tp1_value) - (
                1 - t / self.max_step) * np.mean(t_value)
        risk = np.std(t_value, ddof=1)

        # clean up and return
        self.current_step = t + 1
        done = self.current_step >= self.max_step
        reward = base_reward - self.info.risk_lambda * risk * self.info._dt
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
