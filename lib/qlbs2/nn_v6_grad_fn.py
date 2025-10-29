from typing import Tuple

import numpy as np
import torch

import lib.util
from lib.util.net import StrictResNet, CombinedResNet

import torch.nn.functional as Func
from .env import Baseline, Info, Policy, State
from torch.optim.adam import Adam
from itertools import chain


def in_transform(x: torch.Tensor) -> torch.Tensor:
    # x == [[spot, passed_real_time, remaining_real_time, strike, r, mu, sigma, risk_lambda, friction]]

    log_spot = x[:, 0].log()
    moneyness = (log_spot - x[:, 3].log() + (x[:, 4]) * x[:, 2]) / (x[:, 6] * x[:, 2].sqrt())

    return torch.stack(
        [
            log_spot,  # spot (logged)
            x[:, 1],  # passed_real_time
            x[:, 2],  # remaining_real_time
            moneyness,  # strike (transformed to moneyness)
            x[:, 4],  # r
            x[:, 5],  # mu
            x[:, 6],  # sigma
            x[:, 7],  # risk_lambda
            x[:, 8],  # friction
        ],
        dim=1,
    )


def out_transform(y: torch.Tensor) -> torch.Tensor:
    # return 2 * torch.sigmoid(y)
    return torch.exp(y)


def torch_norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))


def torch_price(S, _, T, K, r, __, sigma, ___, ____):
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
    d2 = (torch.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
    return S * torch_norm_cdf(d1) - K * torch.exp(-r * T) * torch_norm_cdf(d2)


def torch_delta(S, _, T, K, r, __, sigma, ___, ____):
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
    return torch_norm_cdf(d1)


class GaussianPolicy_v6(Policy):
    def __init__(self, is_call_option: bool, from_filename: str | None = None, lr: float = 1e-3, reset_optimizer=False):
        super().__init__()

        self.is_call_option = is_call_option
        self.theta_mu = CombinedResNet(
            input_dim=9,
            hidden_dim=64,
            output_dim=1,
            transform_pair=(in_transform, out_transform),
            activation="elu",
            groups=3,
            layer_per_group=3,
        )
        self.theta_sigma = StrictResNet(9, 10, groups=2, layer_per_group=2)
        self.optimizer = Adam(chain(self.theta_mu.parameters(), self.theta_sigma.parameters()), lr=lr)
        if from_filename is not None:
            self.load(from_filename, reset_optimizer=reset_optimizer)

    def _gauss_param(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = tensor.float()
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
        raise NotImplementedError

    def batch_action(self, state_info_tensor, random: bool = True, return_pre_action: bool = False):
        mu, sigma = self._gauss_param(state_info_tensor)
        if random:
            _action = torch.randn(mu.shape) * sigma + mu
        else:
            _action = mu
        S, _, T, K, r, __, discard, ___, ____ = [state_info_tensor[:, i] for i in range(9)]
        action = torch_delta(S, _, T, K, r, __, _action, ___, ____)
        if return_pre_action:
            return action, _action
        else:
            return action

    def save(self, filename: str):
        raise NotImplementedError

    def load(self, filename: str, reset_optimizer=False):
        state_dict = torch.load(filename)
        self.theta_sigma.load_state_dict(state_dict["sigma_net"])
        self.theta_sigma
        self.theta_sigma.eval()

        self.theta_mu.load_state_dict(state_dict["mu_net"])
        self.theta_mu
        self.theta_mu.eval()

        if not reset_optimizer:
            self.optimizer.load_state_dict(state_dict["optimizer"])


class NNBaseline_v6(Baseline):
    def __init__(self, net: CombinedResNet, optimizer: Adam):
        super().__init__()

        self.net = net
        self.optimizer = optimizer

    def _predict(self, tensor, return_latent_vol: bool = False):
        y = self.net(tensor)
        S, _, T, K, r, __, discard, ___, ____ = [tensor[:, i] for i in range(9)]
        price = torch_price(S, _, T, K, r, __, y[:, 0], ___, ____)
        if return_latent_vol:
            return -price, y[:, 0]
        else:
            return -price

    def __call__(self, state: State, info: Info):
        return self.batch_estimate(state.to_tensor(info))

    def update(self, G: np.ndarray, state: State, info: Info):
        raise NotImplementedError

    def batch_estimate(self, state_info_tensor, return_latent_vol: bool = False):
        state_info_tensor = state_info_tensor.float()
        return self._predict(state_info_tensor, return_latent_vol=return_latent_vol)

    def save(self, filename: str):
        raise NotImplementedError

    def load(self, filename: str, reset_optimizer=False):
        raise NotImplementedError
