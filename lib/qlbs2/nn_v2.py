from ast import Not
from itertools import chain
from math import isnan
from os import write
from typing import Tuple, TypeVar
from unittest import TestCase

import numpy as np
import scipy as sp
import torch
from gymnasium import Env
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import lib.util
from lib.util.net import StrictResNet, CombinedResNet
from lib.util.sample import geometricBM

from .bs import BSBaseline, BSPolicy
from .env import Baseline, Info, Policy, QLBSEnv, State


class NetHolder:
    def __init__(self, from_filename: str | None):
        def in_transform(x: torch.Tensor) -> torch.Tensor:
            # x == [[spot, passed_real_time, remaining_real_time, strike, r, mu, sigma, risk_lambda]]

            x[:, 0] = x[:, 0].log()  # spot
            x[:, 3] = (x[:, 0] - x[:, 3].log() + (x[:, 4]) * x[:, 2]) / (x[:, 6] * x[:, 2].sqrt())  # moneyness
            return x

        def out_transform(y: torch.Tensor) -> torch.Tensor:
            y[:, 0] = torch.exp(y[:, 0])  # price
            y[:, 1] = torch.sigmoid(y[:, 1])  # delta
            return y

        self.net = CombinedResNet(
            input_dim=9,
            hidden_dim=64,
            transform_pair=(in_transform, out_transform),
            activation="elu",
            groups=3,
            layer_per_group=3,
        ).cuda()
        if from_filename is not None:
            state_dict = torch.load(from_filename)
            self.net.load_state_dict(state_dict)
            self.net.cuda()
            self.net.eval()

    def save(self, filename: str):
        lib.util.ensure_dir(filename, need_strip_end=True)
        torch.save(self.net.state_dict(), filename)


class GaussianPolicy_v2(Policy):
    def __init__(self, holder: NetHolder, is_call_option: bool, from_filename: str | None = None):
        super().__init__()
        self.theta_mu_holder = holder
        self.is_call_option = is_call_option
        self.theta_sigma = StrictResNet(9, 10, groups=2, layer_per_group=2).cuda()
        if from_filename is not None:
            self.load(from_filename)

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def _gauss_param(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = tensor.float().cuda()
        _mu = self.theta_mu_holder.net(tensor)[:, 1]

        if torch.any(torch.isnan(_mu)):
            breakpoint()
            raise ValueError("NaN encountered in mu computation!")

        sigma = self.theta_sigma(tensor)[:, 0]
        sigma_c = torch.sigmoid(sigma) * 0.9 + 0.01
        return _mu, sigma_c

    def action(self, state, info):
        return self.batch_action(state.to_tensor(info), random=True)

    def update(self, delta, action, state, info):
        delta = torch.tensor(delta).float().cuda()  # type: ignore
        action = torch.tensor(action).float().cuda()  # type: ignore
        tensor = state.to_tensor(info)

        def loss_func():
            mu, sigma = self._gauss_param(tensor)
            _action, _mu = torch.logit(action, 1e-3), torch.logit(mu, 1e-3)
            log_pi = -((_action - _mu) ** 2) / (2 * sigma**2) - torch.log(sigma)
            loss = torch.sum(-delta * log_pi)
            loss.backward()
            return loss

        self.optimizer.zero_grad()
        loss = self.optimizer.step(loss_func)  # type: ignore
        return loss

    def batch_action(self, state_info_tensor, random: bool = True):
        """
        :param state_info_tensor: [[normal_price, remaining_real_time, normal_strike_price, r, mu, sigma, risk_lambda]]
        :return:
        """
        with torch.no_grad():
            mu, sigma = self._gauss_param(state_info_tensor)
            if random:
                _mu = torch.special.logit(mu, 1e-3)
                _action = torch.randn(mu.shape).cuda() * sigma + _mu
                action = torch.sigmoid(_action)
                return action
            else:
                return mu

    def save(self, filename: str):
        lib.util.ensure_dir(filename, need_strip_end=True)
        torch.save({"sigma_net": self.theta_sigma.state_dict()}, filename)

    def load(self, filename: str):
        state_dict = torch.load(filename)
        self.theta_sigma.load_state_dict(state_dict["sigma_net"])
        self.theta_sigma.cuda()
        self.theta_sigma.eval()


class NNBaseline_v2(Baseline):
    def __init__(self, holder: NetHolder):
        super().__init__()
        self.holder = holder

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def _predict(self, tensor):
        y = self.holder.net(tensor.float().cuda())
        return -y[:, 0]

    def __call__(self, state: State, info: Info):
        return self.batch_estimate(state.to_tensor(info))

    def update(self, G: np.ndarray, state: State, info: Info):
        G = torch.tensor(G).float().cuda()  # type: ignore
        tensor = state.to_tensor(info).cuda()

        def loss_func():
            tensor1 = self._predict(tensor)

            _G = torch.log(torch.clamp(-G, min=1e-3))  # type: ignore
            _tensor1 = torch.log(torch.clamp(-tensor1, min=1e-3))
            loss = torch.sum((_G - _tensor1) ** 2)

            # loss = torch.sum((G - tensor1) ** 2)
            loss.backward()
            return loss

        self.optimizer.zero_grad()
        loss = self.optimizer.step(loss_func)  # type: ignore
        return loss

    def batch_estimate(self, state_info_tensor):
        with torch.no_grad():
            return self._predict(state_info_tensor)

    def save(self, filename: str):
        raise NotImplementedError
        lib.util.ensure_dir(filename, need_strip_end=True)
        torch.save(self.holder.net.state_dict(), filename)
