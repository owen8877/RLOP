import os
import sys

# sys.path.append(os.getcwd())

from dataclasses import replace
from itertools import chain
from typing import Tuple
from unittest import TestCase

from attr import dataclass
import numpy as np
import scipy as sp
import seaborn as sns
import torch
import torch.nn.functional as Func
from matplotlib import pyplot as plt
from torch.optim.adam import Adam

import lib.util
from lib.util.net import CombinedResNet, StrictResNet

from .bs import BSBaseline, BSPolicy
from .env import Baseline, EnvSetup, Info, Policy, QLBSEnv, State
from .rl import policy_gradient


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


@dataclass
class QLBSFitResult:
    sigma: float
    mu: float
    estimated_prices: np.ndarray
    implied_vols: np.ndarray


class QLBSModel:
    def __init__(self, is_call_option: bool, checkpoint: str, anchor_T: float):
        assert is_call_option, "Only call option is supported in this model."
        self.nn_policy = GaussianPolicy_v6(is_call_option=is_call_option, from_filename=checkpoint)
        self.nn_baseline = NNBaseline_v6(self.nn_policy.theta_mu, self.nn_policy.optimizer)
        self.anchor_T = anchor_T

    def fit(
        self,
        spot: float,
        time_to_expiries: np.ndarray,
        strikes: np.ndarray,
        r: float,
        risk_lambda: float,
        friction: float,
        observed_prices: np.ndarray,
        weights: np.ndarray,
        sigma_guess: float,
        mu_guess: float,
        n_epochs: int = 200,
    ):
        N = len(strikes)
        assert len(observed_prices) == N, "Length of observed_prices must match strikes"
        assert len(weights) == N, "Length of weights must match strikes"

        assert sigma_guess > 0
        sigma = torch.tensor(sp.special.logit(sigma_guess), requires_grad=True, dtype=torch.float32)
        mu = torch.tensor(mu_guess, requires_grad=True, dtype=torch.float32)
        optimizer = torch.optim.Adam([sigma, mu], lr=0.5)
        # optimizer = torch.optim.LBFGS([sigma, mu], lr=0.1, max_iter=20, line_search_fn="strong_wolfe")

        _scaled_strikes = torch.tensor(strikes / spot, dtype=torch.float32)
        _scaled_prices = torch.tensor(observed_prices / spot, dtype=torch.float32)
        _scaled_tte = torch.tensor(time_to_expiries / self.anchor_T, dtype=torch.float32)
        _scaled_rs = r * _scaled_tte
        _scaled_risk_lambdas = torch.full((N,), risk_lambda, dtype=torch.float32)
        _scaled_frictions = torch.full((N,), friction, dtype=torch.float32)
        _scaled_weights = torch.tensor(weights * np.sqrt(spot), dtype=torch.float32)

        _zeros, _ones = torch.zeros(N, dtype=torch.float32), torch.ones(N, dtype=torch.float32)

        def get_state_info_tensor(mu, sigma):
            return torch.stack(
                [
                    _ones,  # spot
                    _zeros,  # passed_real_time
                    _ones * self.anchor_T,  # remaining_real_time
                    _scaled_strikes,  # strike
                    _scaled_rs,
                    (mu * _scaled_tte),  # mu
                    ((torch.sigmoid(sigma) + 0.01) * torch.sqrt(_scaled_tte)),  # sigma
                    _scaled_risk_lambdas,  # risk_lambda
                    _scaled_frictions,  # friction
                ],
                dim=1,
            )

        def loss():
            state_info_tensor = get_state_info_tensor(mu, sigma)
            estimated_price = -self.nn_baseline.batch_estimate(state_info_tensor)  # type: ignore
            loss = torch.mean(_scaled_weights * (estimated_price - _scaled_prices) ** 2)
            loss.backward()
            return loss

        loss_history = []
        for epoch in range(n_epochs + 1):
            optimizer.zero_grad()
            optimizer.step(loss)
            l = loss()
            if epoch % 50 == 0:
                print(
                    f"Epoch {epoch}: loss={l.item():.6f}, sigma={(torch.sigmoid(sigma) + 0.01).item():.6f}, mu={mu.item():.6f}"
                )

            if l < 1e-7:
                print("Early stopping as loss is sufficiently small.")
                break

            if epoch > 100 and l.item() > loss_history[-100] * 0.9:
                print("Early stopping as loss has not decreased sufficiently.")
                break
            loss_history.append(l.item())

        with torch.no_grad():
            state_info_tensor = get_state_info_tensor(mu, sigma)
            estimated_price, latent_vol = self.nn_baseline.batch_estimate(state_info_tensor, return_latent_vol=True)  # type: ignore

        # print(sigma, mu)
        return QLBSFitResult(
            sigma=(torch.sigmoid(sigma) + 0.01).item(),
            mu=mu.item(),
            estimated_prices=(-estimated_price.numpy() * spot),
            implied_vols=(latent_vol.numpy() / np.sqrt(time_to_expiries)),
        )

    def predict(
        self,
        spot: float,
        time_to_expiries: np.ndarray,
        strikes: np.ndarray,
        r: float,
        risk_lambda: float,
        friction: float,
        sigma_fit: float,
        mu_fit: float,
    ):
        sigma = torch.tensor(sigma_fit, requires_grad=True, dtype=torch.float32)
        mu = torch.tensor(mu_fit, requires_grad=True, dtype=torch.float32)

        N = len(strikes)
        _scaled_strikes = torch.tensor(strikes / spot, dtype=torch.float32)
        _scaled_rs = torch.tensor(r * time_to_expiries, dtype=torch.float32)
        _scaled_tte = torch.tensor(time_to_expiries / self.anchor_T, dtype=torch.float32)
        _scaled_risk_lambdas = torch.full((N,), risk_lambda, dtype=torch.float32)
        _scaled_frictions = torch.full((N,), friction * spot, dtype=torch.float32)

        _zeros, _ones = torch.zeros(N, dtype=torch.float32), torch.ones(N, dtype=torch.float32)

        def get_state_info_tensor(mu, sigma):
            return torch.stack(
                [
                    _ones,  # spot
                    _zeros,  # passed_real_time
                    _ones * self.anchor_T,  # remaining_real_time
                    _scaled_strikes,  # strike
                    _scaled_rs,
                    (mu * _scaled_tte),  # mu
                    ((torch.sigmoid(sigma) + 0.01) * torch.sqrt(_scaled_tte)),  # sigma
                    _scaled_risk_lambdas,  # risk_lambda
                    _scaled_frictions,  # friction
                ],
                dim=1,
            )

        with torch.no_grad():
            state_info_tensor = get_state_info_tensor(mu, sigma)
            estimated_price, latent_vol = self.nn_baseline.batch_estimate(state_info_tensor, return_latent_vol=True)  # type: ignore

        return QLBSFitResult(
            sigma=(torch.sigmoid(sigma) + 0.01).item(),
            mu=mu.item(),
            estimated_prices=(-estimated_price.numpy() * spot),
            implied_vols=(latent_vol.numpy() / np.sqrt(time_to_expiries)),
        )

    def fit_predict(
        self,
        spot: float,
        time_to_expiries: np.ndarray,
        strikes: np.ndarray,
        r: float,
        risk_lambda: float,
        friction: float,
        observed_prices: np.ndarray,
        weights: np.ndarray,
        sigma_guess: float,
        mu_guess: float,
        n_epochs: int = 200,
    ):
        result = self.fit(
            spot=spot,
            time_to_expiries=time_to_expiries,
            strikes=strikes,
            r=r,
            risk_lambda=risk_lambda,
            friction=friction,
            observed_prices=observed_prices,
            weights=weights,
            sigma_guess=sigma_guess,
            mu_guess=mu_guess,
            n_epochs=n_epochs,
        )
        return self.predict(
            spot=spot,
            time_to_expiries=time_to_expiries,
            strikes=strikes,
            r=r,
            risk_lambda=risk_lambda,
            friction=friction,
            sigma_fit=result.sigma,
            mu_fit=result.mu,
        )


class TestTrainedModel(TestCase):
    def test_load_models(self):
        # setup hyperparameters
        r = 0.02  # interest rate, annualized
        friction = 2e-3  # transaction cost per unit traded
        is_call_option = True  # has to be call option for this test
        risk_lambda = 0.5  # risk aversion parameter
        # for this time, time to maturity and strike price are fixed, but we can probably find a way to vectorize this
        T = 0.5  # time to maturity, in years
        spot = 10.0  # spot price

        sigma_guess = 0.5  # initial guess of volatility
        mu_guess = -1  # initial guess of drift

        def vol_curve_true(K):
            return 0.52 - 0.5 * (sp.special.expit(spot / K - 1) - 0.5)

        # synthesize test data
        N = 100
        Ks = np.linspace(0.85 * spot, 1.15 * spot, N)

        sigma_BS = vol_curve_true(Ks)
        bs_baseline = BSBaseline(is_call=is_call_option)
        target_price = bs_baseline.batch_estimate(
            torch.tensor(
                np.column_stack(
                    (
                        np.full(N, spot),  # spot
                        np.zeros(N),  # passed_real_time
                        np.full(N, T),  # remaining_real_time
                        Ks,  # strike
                        np.full(N, r),  # r
                        np.zeros(N),  # mu
                        sigma_BS,  # sigma
                        np.full(N, risk_lambda),  # risk_lambda
                        np.full(N, friction),  # friction
                    )
                )
            ).float()
        )  # type: ignore
        observed_prices = target_price.numpy() + np.random.randn(N) * 0.1

        # load trained models
        model = QLBSModel(
            is_call_option=is_call_option, checkpoint="trained_model/test7/risk_lambda=5.0e-01/policy_2.pt"
        )
        result = model.fit_predict(
            spot=spot,
            time_to_expiries=T,
            strikes=Ks,
            r=r,
            risk_lambda=risk_lambda,
            friction=friction,
            observed_prices=observed_prices,
            weights=np.ones(N),
            sigma_guess=sigma_guess,
            mu_guess=mu_guess,
            n_epochs=500,
        )

        # plot results
        fig, axs = plt.subplots(2, 1, figsize=(7, 5))

        axs[0].plot(Ks, observed_prices, label="Target Price with Noise", color="black", linestyle="--")
        axs[0].plot(Ks, result.estimated_prices, label="Estimated Price", color="blue", linestyle="-")
        axs[0].set_ylabel("Option Price")
        axs[0].legend(loc="best")

        axs[1].plot(Ks, vol_curve_true(Ks), label="True Vol Curve", color="green", linestyle="--")
        axs[1].plot(Ks, result.implied_vols, label="Inverted Latent Vol", color="red", linestyle="-")
        axs[1].set_ylabel("Volatility")
        axs[1].legend(loc="best")

        plt.xlabel("Strike Price")
        plt.tight_layout()
        plt.show()
