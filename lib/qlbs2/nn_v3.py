from typing import Tuple

import numpy as np
import torch

import lib.util
from lib.util.net import StrictResNet, CombinedResNet

from .env import Baseline, Info, Policy, State
from torch.optim.adam import Adam
from itertools import chain


def in_transform(x: torch.Tensor) -> torch.Tensor:
    # x == [[spot, passed_real_time, remaining_real_time, strike, r, mu, sigma, risk_lambda]]

    x[:, 0] = x[:, 0].log()  # spot
    x[:, 3] = (x[:, 0] - x[:, 3].log() + (x[:, 4]) * x[:, 2]) / (x[:, 6] * x[:, 2].sqrt())  # moneyness
    return x


class GaussianPolicy_v3(Policy):
    def __init__(self, is_call_option: bool, from_filename: str | None = None, lr: float = 1e-3, reset_optimizer=False):
        super().__init__()

        def out_transform(y: torch.Tensor) -> torch.Tensor:
            y[:, 0] = torch.sigmoid(y[:, 0])  # delta
            return y

        self.is_call_option = is_call_option
        self.theta_mu = CombinedResNet(
            input_dim=9,
            hidden_dim=64,
            output_dim=1,
            transform_pair=(in_transform, out_transform),
            activation="elu",
            groups=3,
            layer_per_group=3,
        ).cuda()
        self.theta_sigma = StrictResNet(9, 10, groups=2, layer_per_group=2).cuda()
        self.optimizer = Adam(chain(self.theta_mu.parameters(), self.theta_sigma.parameters()), lr=lr)
        if from_filename is not None:
            self.load(from_filename, reset_optimizer=reset_optimizer)

    def _gauss_param(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = tensor.float().cuda()
        _mu = self.theta_mu(tensor)[:, 0]

        if torch.any(torch.isnan(_mu)):
            breakpoint()
            raise ValueError("NaN encountered in mu computation!")

        sigma = self.theta_sigma(tensor)[:, 0]
        sigma_c = torch.sigmoid(sigma) * 0.9 + 0.01
        return _mu, sigma_c

    def action(self, state, info, return_pre_action: bool = False):
        return self.batch_action(state.to_tensor(info), random=True, return_pre_action=return_pre_action)

    def update(self, delta, _action, state, info):
        delta = torch.tensor(delta).float().cuda()  # type: ignore
        _action = torch.tensor(_action).float().cuda()  # type: ignore
        tensor = state.to_tensor(info)

        def loss_func():
            mu, sigma = self._gauss_param(tensor)
            _mu = torch.logit(mu, 1e-3)
            log_pi = -((_action - _mu) ** 2) / (2 * sigma**2) - torch.log(sigma)
            loss = torch.sum(-delta * log_pi)
            loss.backward()
            return loss

        self.optimizer.zero_grad()
        loss = self.optimizer.step(loss_func)  # type: ignore
        return loss

    def batch_action(self, state_info_tensor, random: bool = True, return_pre_action: bool = False):
        with torch.no_grad():
            mu, sigma = self._gauss_param(state_info_tensor)
            if random:
                _mu = torch.special.logit(mu, 1e-3)
                _action = torch.randn(mu.shape).cuda() * sigma + _mu
                action = torch.sigmoid(_action)
                if return_pre_action:
                    return action, _action
                else:
                    return action
            else:
                if return_pre_action:
                    return mu, torch.special.logit(mu, 1e-3)
                else:
                    return mu

    def save(self, filename: str):
        lib.util.ensure_dir(filename, need_strip_end=True)
        torch.save(
            {
                "sigma_net": self.theta_sigma.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "mu_net": self.theta_mu.state_dict(),
            },
            filename,
        )

    def load(self, filename: str, reset_optimizer=False):
        state_dict = torch.load(filename)
        self.theta_sigma.load_state_dict(state_dict["sigma_net"])
        self.theta_sigma.cuda()
        self.theta_sigma.eval()

        self.theta_mu.load_state_dict(state_dict["mu_net"])
        self.theta_mu.cuda()
        self.theta_mu.eval()

        if not reset_optimizer:
            self.optimizer.load_state_dict(state_dict["optimizer"])


class NNBaseline_v3(Baseline):
    def __init__(self, from_filename: str | None = None, lr: float = 1e-3, reset_optimizer=False):
        super().__init__()

        def out_transform(y: torch.Tensor) -> torch.Tensor:
            y[:, 0] = torch.exp(y[:, 0])
            return y

        self.net = CombinedResNet(
            input_dim=9,
            hidden_dim=64,
            output_dim=1,
            transform_pair=(in_transform, out_transform),
            activation="elu",
            groups=3,
            layer_per_group=3,
        ).cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        if from_filename is not None:
            self.load(from_filename, reset_optimizer=reset_optimizer)

    def _predict(self, tensor):
        y = self.net(tensor.float().cuda())
        return -y[:, 0]

    def __call__(self, state: State, info: Info):
        return self.batch_estimate(state.to_tensor(info))

    def update(self, G: np.ndarray, state: State, info: Info):
        G = torch.tensor(G).float().cuda()  # type: ignore
        tensor = state.to_tensor(info).cuda()

        def loss_func():
            tensor1 = self._predict(tensor)

            # _G = torch.log(torch.clamp(-G, min=1e-3))  # type: ignore
            # _tensor1 = torch.log(torch.clamp(-tensor1, min=1e-3))
            # loss = torch.sum((_G - _tensor1) ** 2)

            loss = torch.sum((G - tensor1) ** 2)
            loss.backward()
            return loss

        self.optimizer.zero_grad()
        loss = self.optimizer.step(loss_func)  # type: ignore
        return loss

    def batch_estimate(self, state_info_tensor):
        with torch.no_grad():
            return self._predict(state_info_tensor)

    def save(self, filename: str):
        lib.util.ensure_dir(filename, need_strip_end=True)
        torch.save({"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()}, filename)

    def load(self, filename: str, reset_optimizer=False):
        state_dict = torch.load(filename)
        self.net.load_state_dict(state_dict["net"])
        self.net.cuda()
        self.net.eval()

        if not reset_optimizer:
            self.optimizer.load_state_dict(state_dict["optimizer"])
