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
    parallel_n: int

    def to_tensor0(self, info: Info):
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

    def to_tensor(self, info: Info):
        sits = []
        for t in range(self.passed_step, self.passed_step + self.remaining_step):
            N = len(self.spots)
            sit = torch.empty((N, 9))
            sit[:, 0] = torch.tensor(self.spots)  # spot
            sit[:, 1] = t * info._dt  # passed_real
            sit[:, 2] = (self.remaining_step + self.passed_step - t) * info._dt  # remaining_real_time
            sit[:, 3] = torch.tensor(info.strike_prices)  # strike
            sit[:, 4] = info.r  # r
            sit[:, 5] = torch.tensor(info.mus)  # mu
            sit[:, 6] = torch.tensor(info.sigmas)  # sigma
            sit[:, 7] = torch.tensor(info.risk_lambdas)  # risk_lambda
            sit[:, 8] = torch.tensor(info.frictions)  # friction
            sits.append(sit)
        return torch.concat(sits, dim=0)


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


class PriceEstimator:
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

    def update(self, loss: np.ndarray, state: State, info: Info):
        raise NotImplementedError

    def train_based_on(self, source: torch.Tensor, target: torch.Tensor, lr: float, itr_max: int):
        raise NotImplementedError


@dataclass
class EnvSetup(Info):
    is_call_option: bool
    parallel_n: int
    max_step: int
    initial_spots: np.ndarray
    mutation: Union[float, Callable] = 0.1

    def _assert_valid(self):
        super()._assert_valid()
        assert self.parallel_n == len(self.initial_spots)
        assert self.max_step >= 1
        assert all(s > 0 for s in self.initial_spots)


class RLOPEnv:
    def __init__(self, setup: EnvSetup, price_estimator: PriceEstimator):
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
        self.price_estimator = price_estimator

        self.portfolio_value = np.zeros((self.parallel_n, self.max_step))

        # State variables
        self._standard_price: np.ndarray | None = None  # shape: (parallel_n, max_step)
        self.current_step = 0

    def _describe(self):
        assert self._standard_price is not None
        return State(
            self._standard_price[:, self.current_step],
            self.current_step,
            self.max_step - self.current_step,
            self.parallel_n,
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

        # values = self.price_estimator(self._describe(), self.info).cpu().numpy()
        # self.portfolio_value = np.repeat(values[:, None], self.max_step, axis=1)
        self.portfolio_value = self.price_estimator(self._describe(), self.info).cpu().numpy()  # type: ignore
        self.old_action = np.zeros((self.parallel_n, self.max_step))

        return self._describe(), self.info

    def step(self, action) -> Tuple[State, np.ndarray, bool]:
        assert self._standard_price is not None

        tau = self.current_step
        t = tau + 1

        pvs = []
        for expiry in range(tau, self.max_step):
            S_tau = self._standard_price[:, tau]
            S_t = self._standard_price[:, t]
            _action = action[self.parallel_n * (expiry - tau) : self.parallel_n * (expiry - tau + 1)]
            if tau > 0:
                _old_action = self.old_action[
                    self.parallel_n * (expiry - tau + 1) : self.parallel_n * (expiry - tau + 2)
                ]
            else:
                _old_action = np.zeros_like(_action)
            _pv = self.portfolio_value[self.parallel_n * (expiry - tau) : self.parallel_n * (expiry - tau + 1)]

            # compute the cash position
            cash_tau = _pv - _action * S_tau
            cash_t = cash_tau * np.exp(self.info.r * self.info._dt)
            portfolio_value_t = cash_t + _action * S_t - self.info.frictions * np.abs(_action - _old_action) * S_t

            if expiry == tau:
                payoff = lib.util.payoff_of_option(self.is_call_option, S_t, self.info.strike_prices)
                reward = -np.abs(payoff - portfolio_value_t)
                # reward = -(payoff - portfolio_value_t) ** 2
            else:
                pvs.append(portfolio_value_t)

        if len(pvs) > 0:
            self.portfolio_value = np.concatenate(pvs, axis=0)
        self.current_step += 1
        done = self.current_step >= self.max_step
        self.old_action = action

        return self._describe(), reward, done

    def render(self, mode="human"):
        pass

    def mutate_parameters(self):
        if callable(self.mutation):
            self.mutation(self)
            return


# register(id="QLBS-v0", entry_point="qlbs.env:QLBSEnv")
