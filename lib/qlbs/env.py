from typing import Tuple, Callable, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import register

import lib.util
from lib.util.sample import geometricBM


class Info:
    def __init__(
        self,
        strike_price: float,
        r: float,
        mu: float,
        sigma: float,
        risk_lambda: float,
        _dt: float,
        friction: float,
    ):
        self.strike_price = strike_price
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.risk_lambda = risk_lambda
        self._dt = _dt
        self.friction = friction


class State:
    def __init__(self, normalized_asset_price: float, passed_step: int, remaining_step: int):
        self.normalized_asset_price = normalized_asset_price
        self.passed_step = passed_step
        self.remaining_step = remaining_step

    def to_tensor(self, info: Info):
        return torch.tensor(
            (
                self.normalized_asset_price,
                self.passed_step * info._dt,
                self.remaining_step * info._dt,
                lib.util.standard_to_normalized_price(
                    info.strike_price,
                    info.mu,
                    info.sigma,
                    (self.passed_step + self.remaining_step),
                    1,
                ),
                info.r,
                info.mu,
                info.sigma,
                info.risk_lambda,
                info.friction,
            )
        )


class Policy:
    def __init__(self):
        pass

    def action(self, state: State, info: Info):
        raise NotImplementedError

    def batch_action(self, state_info_tensor, random: bool = True):
        """
        :param random:
        :param state_info_tensor: [[normal_price, passed_real_time, remaining_real_time, normal_strike_price, r, mu,
            sigma, risk_lambda, friction]]
        :return:
        """
        raise NotImplementedError

    def update(self, delta: float, action: float, state: State, info: Info):
        raise NotImplementedError

    def train_based_on(self, source, target, lr, itr_max):
        raise NotImplementedError


class Baseline:
    def __init__(self):
        pass

    def __call__(self, state: State, info: Info):
        raise NotImplementedError

    def batch_estimate(self, state_info_tensor):
        """
        :param state_info_tensor: [[normal_price, passed_real_time, remaining_real_time, normal_strike_price, r, mu,
            sigma, risk_lambda, friction]]
        :return:
        """
        raise NotImplementedError

    def update(self, G: float, state: State, info: Info):
        raise NotImplementedError

    def train_based_on(self, source, target, lr, itr_max):
        raise NotImplementedError


class QLBSEnv(gym.Env):
    def __init__(
        self,
        is_call_option: bool,
        strike_price: float,
        max_step: int,
        mu: float,
        sigma: float,
        r: float,
        risk_lambda: float,
        friction: float,
        initial_asset_price: float,
        risk_simulation_paths: int,
        *,
        mutation: Union[float, Callable] = 0.1,
        _dt: float = 1.0,
    ):
        # Setting of the system; stays constant for a long time
        self.is_call_option = is_call_option
        self.info = Info(
            strike_price=strike_price,
            r=r,
            mu=mu,
            sigma=sigma,
            risk_lambda=risk_lambda,
            _dt=_dt,
            friction=friction,
        )
        self.gamma = np.exp(-r * _dt)
        self.max_step = max_step
        self.initial_asset_price = initial_asset_price
        self.mutation = mutation
        self.risk_simulation_paths = risk_simulation_paths

        # State variables
        self._normalized_price: np.ndarray | None = None
        self._standard_price: np.ndarray | None = None
        self.current_step = 0

    def _describe(self):
        assert self._normalized_price is not None
        return State(
            self._normalized_price[self.current_step],
            self.current_step,
            self.max_step - self.current_step,
        )

    def reset(self) -> Tuple[State, Info]:
        self.mutate_parameters()
        GBM, BM = geometricBM(
            self.initial_asset_price,
            self.max_step,
            1,
            self.info.mu,
            self.info.sigma,
            self.info._dt,
        )
        self._normalized_price = BM[0, :]
        self._standard_price = GBM[0, :]
        self.current_step = 0

        return self._describe(), self.info

    def step(self, action, pi: Policy) -> Tuple[State, float, bool, dict]:
        assert self._standard_price is not None
        t = self.current_step

        # simulate RS paths to compute E[Pi_t|F_t], E[Pi_(t+1)|F_(t+1)], and Var[Pi_t|F_t]
        RS = self.risk_simulation_paths
        t_arr = np.arange(self.max_step - t + 1)
        t_arr_broad = np.broadcast_to(t_arr[np.newaxis, :], (RS, len(t_arr)))
        GBM, BM = geometricBM(
            self._standard_price[t],
            self.max_step - t,
            RS,
            self.info.mu,
            self.info.sigma,
            self.info._dt,
        )

        # Compute hedge position in a batch fashion
        sits = []
        for s in np.arange(t, self.max_step):
            sit = torch.empty((RS, 9))
            sit[:, 0] = torch.tensor(BM[:, s - t])  # normal_price
            sit[:, 1] = s * self.info._dt  # passed_real_time
            sit[:, 2] = (self.max_step - s) * self.info._dt  # remaining_real_time
            sit[:, 3] = lib.util.standard_to_normalized_price(  # type: ignore
                self.info.strike_price,
                self.info.mu,
                self.info.sigma,
                self.max_step,
                self.info._dt,
            )  # normal_strike_price
            sit[:, 4] = self.info.r  # r
            sit[:, 5] = self.info.mu  # mu
            sit[:, 6] = self.info.sigma  # sigma
            sit[:, 7] = self.info.risk_lambda  # risk_lambda
            sit[:, 8] = self.info.friction  # friction
            sits.append(sit)
        hedge_long = pi.batch_action(torch.cat(sits, dim=0))
        hedge = hedge_long.reshape(RS, self.max_step - t, order="F")
        hedge[:, 0] = action

        # compute discounted stock price and value change
        discount = np.power(self.gamma, t_arr_broad)
        discounted_S = discount * GBM
        discounted_cashflow = hedge * (discounted_S[:, 1:] - discounted_S[:, :-1])
        extended_hedge = np.concatenate((hedge, np.zeros((RS, 1))), axis=1)
        discounted_tc = (
            self.info.friction
            * discount[:, 1:]
            * lib.util.abs(extended_hedge[:, 1:] - extended_hedge[:, :-1])
            * GBM[:, 1:]
        )
        end_value = lib.util.payoff_of_option(self.is_call_option, GBM[:, -1], self.info.strike_price)  # type:ignore

        # sum up change and compute the expected portfolio value and its variance
        total_value_change = np.sum(discounted_cashflow, axis=1)
        total_tc = np.sum(discounted_tc, axis=1)
        t_value = end_value * np.power(self.gamma, self.max_step - t) - total_value_change + total_tc
        tp1_value = (t_value + discounted_cashflow[:, 0]) / self.gamma

        base_reward = self.gamma * (1 - (t + 1) / self.max_step) * np.mean(tp1_value) - (
            1 - t / self.max_step
        ) * np.mean(t_value)
        risk = np.std(t_value, ddof=1)

        # clean up and return
        self.current_step = t + 1
        done = self.current_step >= self.max_step
        reward = base_reward - self.info.risk_lambda * risk * self.info._dt
        return self._describe(), reward, done, {"risk": risk}

    def render(self, mode="human"):
        pass

    def mutate_parameters(self):
        if callable(self.mutation):
            self.mutation(self)
            return

        if np.random.rand(1) < self.mutation:
            self.info.r = np.clip(self.info.r * (1 + 0.1 * np.random.randn(1)[0]), 0, 2e-2)
        if np.random.rand(1) < self.mutation:
            self.info.mu = np.clip(
                self.info.mu + 1e-3 * np.random.randn(1)[0] - 1 * self.info.mu,
                -1e-3,
                1e-3,
            )
        if np.random.rand(1) < self.mutation:
            self.info.sigma = np.clip(self.info.sigma * (1 + 0.1 * np.random.randn(1)[0]), 0, 2e-1)
        if np.random.rand(1) < self.mutation:
            self.info.strike_price = np.clip(
                self.info.strike_price + 1e-2 * np.random.randn(1)[0] - 0.1 * (self.info.strike_price - 1),
                0.9,
                1.1,
            )
        if np.random.rand(1) < self.mutation:
            self.initial_asset_price = np.clip(
                self.initial_asset_price + 1e-1 * np.random.randn(1)[0] - 0.1 * (self.initial_asset_price - 1),
                0.7,
                1.3,
            )


register(id="QLBS-v0", entry_point="qlbs.env:QLBSEnv")
