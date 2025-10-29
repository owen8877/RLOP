from dataclasses import dataclass
from typing import Callable, Tuple, Union

import numpy as np
import torch

import lib.util
from lib.util.sample import geometricBM_parallel


@dataclass
class Info:
    strike_prices: np.ndarray
    r: float
    mus: np.ndarray
    sigmas: np.ndarray
    risk_lambdas: np.ndarray
    _dt: float
    frictions: np.ndarray

    def _assert_valid(self):
        assert all(s > 0 for s in self.strike_prices)
        assert self.r >= 0
        assert all(s > 0 for s in self.sigmas)
        assert self._dt > 0
        assert all(f >= 0 for f in self.frictions)
        assert all(m is not None for m in self.mus)
        assert all(l >= 0 for l in self.risk_lambdas)


@dataclass
class State:
    spots: np.ndarray
    passed_step: int
    remaining_step: int

    def to_tensor(self, info: Info):
        N = len(self.spots)
        sit = torch.empty((N, 9))
        sit[:, 0] = torch.tensor(self.spots)  # spot
        sit[:, 1] = self.passed_step * info._dt  # passed_real
        sit[:, 2] = self.remaining_step * info._dt  # remaining_real_time
        sit[:, 3] = torch.tensor(info.strike_prices)  # strike
        sit[:, 4] = info.r  # r
        sit[:, 5] = torch.tensor(info.mus)  # mu
        sit[:, 6] = torch.tensor(info.sigmas)  # sigma
        sit[:, 7] = torch.tensor(info.risk_lambdas)  # risk_lambda
        sit[:, 8] = torch.tensor(info.frictions)  # friction
        return sit


class Policy:
    def __init__(self):
        pass

    def action(self, state: State, info: Info, return_pre_action: bool = False):
        raise NotImplementedError

    def batch_action(self, state_info_tensor: torch.Tensor, random: bool = True) -> torch.Tensor:
        """
        :param random:
        :param state_info_tensor: [[spot, passed_real_time, remaining_real_time, strike, r, mu, sigma, risk_lambda, friction]]
        :return:
        """
        raise NotImplementedError

    def update(self, delta: float, action: float, state: State, info: Info):
        raise NotImplementedError

    def train_based_on(self, source: torch.Tensor, target: torch.Tensor, lr: float, itr_max: int):
        raise NotImplementedError


class Baseline:
    def __init__(self):
        pass

    def __call__(self, state: State, info: Info):
        raise NotImplementedError

    def batch_estimate(self, state_info_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param state_info_tensor: [[spot, passed_real_time, remaining_real_time, strike, r, mu, sigma, risk_lambda, friction]]
        :return:
        """
        raise NotImplementedError

    def update(self, G: float, state: State, info: Info):
        raise NotImplementedError

    def train_based_on(self, source: torch.Tensor, target: torch.Tensor, lr: float, itr_max: int):
        raise NotImplementedError


@dataclass
class EnvSetup(Info):
    is_call_option: bool
    parallel_n: int
    max_step: int
    initial_spots: np.ndarray
    risk_simulation_paths: int
    mutation: Union[float, Callable] = 0.1

    def _assert_valid(self):
        super()._assert_valid()
        assert self.parallel_n == len(self.initial_spots)
        assert self.risk_simulation_paths >= 1
        assert self.max_step >= 1
        assert all(s > 0 for s in self.initial_spots)


class QLBSEnv:
    def __init__(self, setup: EnvSetup):
        setup._assert_valid()
        self.is_call_option = setup.is_call_option
        self.info = Info(
            strike_prices=setup.strike_prices,
            r=setup.r,
            mus=setup.mus,
            sigmas=setup.sigmas,
            risk_lambdas=setup.risk_lambdas,
            _dt=setup._dt,
            frictions=setup.frictions,
        )
        self.gamma = np.exp(-self.info.r * self.info._dt)
        self.max_step = setup.max_step
        self.parallel_n = setup.parallel_n
        self.initial_spots = setup.initial_spots
        self.mutation = setup.mutation
        self.risk_simulation_paths = setup.risk_simulation_paths

        # State variables
        self._standard_price: np.ndarray | None = None  # shape: (parallel_n, max_step)
        self.current_step = 0

    def _describe(self):
        assert self._standard_price is not None
        return State(
            self._standard_price[:, self.current_step],
            self.current_step,
            self.max_step - self.current_step,
        )

    def reset(self) -> Tuple[State, Info]:
        self.mutate_parameters()
        GBM, _ = geometricBM_parallel(
            np.array(self.initial_spots),
            self.parallel_n,
            self.max_step,
            1,
            np.array(self.info.mus),
            np.array(self.info.sigmas),
            self.info._dt,
        )
        self._standard_price = GBM[:, :, 0]
        self.current_step = 0

        return self._describe(), self.info

    def step(self, action, pi: Policy) -> Tuple[State, np.ndarray, bool, dict]:
        assert self._standard_price is not None
        t = self.current_step

        # simulate RS paths to compute E[Pi_t|F_t], E[Pi_(t+1)|F_(t+1)], and Var[Pi_t|F_t]
        RS = self.risk_simulation_paths
        N_parallel = self.parallel_n
        t_arr = np.arange(self.max_step - t + 1)
        t_arr_broad = np.broadcast_to(t_arr[np.newaxis, :, np.newaxis], (N_parallel, len(t_arr), RS))
        GBM, _ = geometricBM_parallel(
            self._standard_price[:, t],
            N_parallel,
            self.max_step - t,
            RS,
            np.array(self.info.mus),
            np.array(self.info.sigmas),
            self.info._dt,
        )

        # Compute hedge position in a batch fashion
        sits = []
        for s in np.arange(t, self.max_step):
            sit = torch.empty((N_parallel, 9, RS))
            sit[:, 0, :] = torch.tensor(GBM[:, s - t, :])  # spot
            sit[:, 1, :] = s * self.info._dt  # passed_real_time
            sit[:, 2, :] = (self.max_step - s) * self.info._dt  # remaining_real_time
            sit[:, 3, :] = torch.tensor(self.info.strike_prices)[:, None]  # strike
            sit[:, 4, :] = self.info.r  # r
            sit[:, 5, :] = torch.tensor(self.info.mus)[:, None]  # mu
            sit[:, 6, :] = torch.tensor(self.info.sigmas)[:, None]  # sigma
            sit[:, 7, :] = torch.tensor(self.info.risk_lambdas)[:, None]  # risk_lambda
            sit[:, 8, :] = torch.tensor(self.info.frictions)[:, None]  # friction
            sits.append(sit)

        sits_stacked = torch.cat(sits, dim=0)
        sits_reshaped = sits_stacked.permute(0, 2, 1).reshape(-1, 9)
        hedge_long = pi.batch_action(sits_reshaped).cpu().numpy()
        hedge = np.transpose(hedge_long.reshape(self.max_step - t, N_parallel, RS), (1, 0, 2))
        hedge[:, 0, :] = action[:, None]

        # compute discounted stock price and value change
        discount = np.power(self.gamma, t_arr_broad)
        discounted_S = discount * GBM
        discounted_cashflow = hedge * (discounted_S[:, 1:, :] - discounted_S[:, :-1, :])
        extended_hedge = np.concatenate((hedge, np.zeros((N_parallel, 1, RS))), axis=1)
        discounted_tc = (
            np.array(self.info.frictions)[:, None, None]
            * discount[:, 1:, :]
            * lib.util.abs(extended_hedge[:, 1:, :] - extended_hedge[:, :-1, :])
            * GBM[:, 1:, :]
        )
        end_value = lib.util.payoff_of_option(
            self.is_call_option,
            GBM[:, -1, :],  # type:ignore
            np.array(self.info.strike_prices)[:, None],  # type:ignore
        )

        # sum up change and compute the expected portfolio value and its variance
        total_value_change = np.sum(discounted_cashflow, axis=1)
        total_tc = np.sum(discounted_tc, axis=1)
        t_value = end_value * np.power(self.gamma, self.max_step - t) - total_value_change + total_tc
        tp1_value = (t_value + discounted_cashflow[:, 0]) / self.gamma

        base_reward = self.gamma * (1 - (t + 1) / self.max_step) * np.mean(tp1_value, axis=1) - (
            1 - t / self.max_step
        ) * np.mean(t_value, axis=1)
        risk = np.std(t_value, axis=1, ddof=1)

        # clean up and return
        self.current_step = t + 1
        done = self.current_step >= self.max_step
        reward = base_reward - self.info.risk_lambdas * risk * self.info._dt
        return self._describe(), reward, done, {"risk": risk}

    def render(self, mode="human"):
        pass

    def mutate_parameters(self):
        if callable(self.mutation):
            self.mutation(self)
            return


# register(id="QLBS-v0", entry_point="qlbs.env:QLBSEnv")
