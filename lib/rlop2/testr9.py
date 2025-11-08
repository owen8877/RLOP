import os
from dataclasses import replace
from unittest import TestCase

import numpy as np
import scipy as sp
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from .bs import BSBaseline, BSPolicy
from .env import EnvSetup, RLOPEnv
from .nn_v6 import GaussianPolicy_v6, NNBaseline_v6, PriceEstimator_v6
from .rl import policy_gradient

exp_name = os.path.basename(__file__)[:-3]


class Training(TestCase):
    def _parameters(self):
        T = 1
        parallel_n = 1_000
        return EnvSetup(
            parallel_n=parallel_n,
            r=0.04,
            mus=np.random.randn(parallel_n) * 2.0,
            sigmas=np.random.rand(parallel_n) * 2.0 + 0.01,
            risk_lambdas=None,  # type: ignore
            strike_prices=np.random.rand(parallel_n) * 1.4 + 0.3,
            _dt=(_dt := 1 / 4),
            frictions=np.random.rand(parallel_n) * 1e-2,
            is_call_option=True,
            max_step=int(np.round(T / _dt)),
            initial_spots=np.random.rand(parallel_n) * 1.4 + 0.3,
            mutation=0.0,
        )

    def _path(self, exp_name=exp_name):
        return f"trained_model/{exp_name}"

    def test_learn_both_from_0_vol_latent(self):
        setup = self._parameters()
        N_parallel = setup.parallel_n
        check_point = None

        path = self._path()

        nn_policy = GaussianPolicy_v6(
            lr=1e-4,
            is_call_option=setup.is_call_option,
            from_filename=f"{path}/policy_{check_point}.pt"
            if check_point
            else None,  # f"{self._path(risk_lambda, exp_name='test2')}/policy_1.pt",
            # reset_optimizer=True,
        )
        nn_baseline = NNBaseline_v6(
            lr=1e-4,
            is_call_option=setup.is_call_option,
            from_filename=f"{path}/baseline_{check_point}.pt"
            if check_point
            else None,  # f"{self._path(risk_lambda, exp_name='test2')}/baseline_1.pt",
        )
        nn_price_estimator = PriceEstimator_v6(
            nn_policy.theta_mu,
            nn_policy.optimizer,
        )
        env = RLOPEnv(replace(setup, risk_lambdas=np.array([0] * N_parallel)), nn_price_estimator)

        policy_gradient(
            env,
            nn_policy,
            nn_baseline,
            nn_price_estimator,
            episode_n=200,
            tensorboard_label=f"rlop2-{exp_name}-test-8",
        )

        check_point_save = 1 if check_point is None else check_point + 1
        nn_policy.save(f"{path}/policy_{check_point_save}.pt")
        nn_baseline.save(f"{path}/baseline_{check_point_save}.pt")

    def test_draw_plot(self):
        check_point = 1

        setup = self._parameters()
        is_call_option = True
        path = self._path()

        T, K, r, sigma, friction = 28 / 252, 1.0, 0.04, 0.5, 1e-3
        initial_spots = np.linspace(0.2, 1.8, 50)

        _, axs = plt.subplots(4, 1, figsize=(7, 9), constrained_layout=True, sharex=True)
        palette = sns.color_palette()

        def build_state_info_tensor(mu):
            initial_spot_tensor = torch.tensor(initial_spots)
            state_info_tensor = torch.empty((len(initial_spot_tensor), 9))
            state_info_tensor[:, 0] = initial_spot_tensor  # spot
            state_info_tensor[:, 1] = 0  # passed_real_time
            state_info_tensor[:, 2] = T  # remaining_real_time
            state_info_tensor[:, 3] = K  # strike
            state_info_tensor[:, 4] = r  # r
            state_info_tensor[:, 5] = mu  # mu
            state_info_tensor[:, 6] = sigma  # sigma
            state_info_tensor[:, 7] = 0  # risk_lambda
            state_info_tensor[:, 8] = friction  # friction
            return state_info_tensor

        print(path)
        nn_policy = GaussianPolicy_v6(is_call_option=is_call_option, from_filename=f"{path}/policy_{check_point}.pt")
        nn_price_estimator = PriceEstimator_v6(nn_policy.theta_mu, nn_policy.optimizer)

        for i, mu in enumerate([-2, -1, 0, 1, 2]):
            tensor = build_state_info_tensor(mu)

            if i == 0:
                bs_baseline = BSBaseline(is_call=is_call_option)
                bs_price = bs_baseline.batch_estimate(tensor)
                axs[0].plot(initial_spots, bs_price, label="BS Baseline", color=palette[0], linestyle="--")

                bs_policy = BSPolicy(is_call=is_call_option)
                bs_price = bs_policy.batch_action(tensor, random=False)
                axs[2].plot(initial_spots, bs_price, label="BS Policy", color=palette[0], linestyle="--")

            nn_price, nn_latent_vol = nn_price_estimator.batch_estimate(tensor.cuda(), return_latent_vol=True)
            nn_price = nn_price.cpu().numpy()[: len(initial_spots)]
            nn_latent_vol = nn_latent_vol.cpu().numpy()[: len(initial_spots)]
            axs[0].plot(initial_spots, nn_price, label=f"mu={mu}", color=palette[i + 1], linestyle="-")
            axs[0].set_ylabel("Option Price")

            axs[1].plot(initial_spots, nn_latent_vol, label=f"mu={mu}", color=palette[i + 1], linestyle="-")
            axs[1].set_ylabel("Latent Vol")

            nn_action, _nn_action = nn_policy.batch_action(tensor.cuda(), random=False, return_pre_action=True)  # type: ignore
            nn_action = nn_action.cpu().numpy()[: len(initial_spots)]
            _nn_action = _nn_action.cpu().numpy()[: len(initial_spots)]
            axs[2].plot(initial_spots, nn_action, label=f"mu={mu}", color=palette[i + 1], linestyle="-")
            axs[2].set_ylabel("Delta Action")

            axs[3].plot(initial_spots, _nn_action, label=f"mu={mu}", color=palette[i + 1], linestyle="-")
            axs[3].set_ylabel("Delta net Latent Vol")

        axs[0].legend(loc="best")
        plt.show()

    def test_invert_price(self):
        from lib.rlop2.nn_v6 import PriceEstimator_v6_grad_fn

        # setup hyperparameters
        r = 0.04  # interest rate, annualized
        friction = 2e-3  # transaction cost per unit traded
        is_call_option = True  # has to be call option for this test
        # for this time, time to maturity and strike price are fixed, but we can probably find a way to vectorize this
        T = 28 / 252  # time to maturity, in years
        K = 10.0  # strike price

        initial_sigma_guess = 0.5  # initial guess of volatility
        initial_mu_guess = 0.00  # initial guess of drift; due to theoretic loop holes, call skew options have mu < 0

        def vol_curve_true(spots):
            return 0.52 + 3 * (sp.special.expit(spots / K - 1) - 0.5)

        # synthesize test data
        N = 100
        spots = np.linspace(0.85 * K, 1.15 * K, N)

        sigma_BS = vol_curve_true(spots)
        bs_baseline = BSBaseline(is_call=is_call_option)
        target_price = bs_baseline.batch_estimate(
            torch.tensor(
                np.column_stack(
                    (
                        spots,  # spot
                        np.zeros(N),  # passed_real_time
                        np.full(N, T),  # remaining_real_time
                        np.full(N, K),  # strike
                        np.full(N, r),  # r
                        np.zeros(N),  # mu
                        sigma_BS,  # sigma
                        np.full(N, 0),  # risk_lambda
                        np.full(N, friction),  # friction
                    )
                )
            ).float()
        )  # type: ignore
        target_price_with_noise = target_price.numpy() + np.random.randn(N) * 0.1

        # load trained models
        path = self._path()
        nn_policy = GaussianPolicy_v6(is_call_option=is_call_option, from_filename=f"{path}/policy_2.pt")
        nn_price_estimator = PriceEstimator_v6_grad_fn(nn_policy.theta_mu, nn_policy.optimizer)

        # invert prices
        sigma = torch.tensor(initial_sigma_guess, requires_grad=True, dtype=torch.float32)
        mu = torch.tensor(initial_mu_guess, requires_grad=True, dtype=torch.float32)
        optimizer = torch.optim.Adam([sigma, mu], lr=0.1)

        _scaled_spot = torch.tensor(spots, dtype=torch.float32) / K
        target_price_with_noise_cuda = torch.tensor(target_price_with_noise / K, dtype=torch.float32)
        _scaled_r = r * T
        _scaled_friction = friction * K

        def get_state_info_tensor(mu, sigma):
            return torch.stack(
                [
                    _scaled_spot,  # spot
                    torch.zeros(N, dtype=torch.float32),  # passed_real_time
                    torch.ones(N, dtype=torch.float32),  # remaining_real_time
                    torch.ones(N, dtype=torch.float32),  # strike
                    torch.full((N,), _scaled_r, dtype=torch.float32),  # r
                    (mu * T).expand(N),  # mu
                    (sigma * np.sqrt(T)).expand(N),  # sigma
                    torch.full((N,), 0, dtype=torch.float32),  # risk_lambda
                    torch.full((N,), _scaled_friction, dtype=torch.float32),  # friction
                ],
                dim=1,
            )

        def loss():
            state_info_tensor = get_state_info_tensor(mu, sigma)
            estimated_price = nn_price_estimator.batch_estimate(state_info_tensor)  # type: ignore
            return torch.mean((estimated_price - target_price_with_noise_cuda) ** 2)  # type: ignore

        for epoch in range(20):
            optimizer.zero_grad()
            l = loss()
            l.backward()
            optimizer.step()
            print(f"Epoch {epoch}: loss={l.item():.6f}, sigma={sigma.item():.6f}, mu={mu.item():.6f}")

        # plot results
        fig, axs = plt.subplots(2, 1, figsize=(7, 5))

        axs[0].plot(spots, target_price_with_noise, label="Target Price with Noise", color="black", linestyle="--")
        with torch.no_grad():
            state_info_tensor = get_state_info_tensor(mu, sigma)
            estimated_price, latent_vol = nn_price_estimator.batch_estimate(state_info_tensor, return_latent_vol=True)  # type: ignore
            axs[0].plot(spots, -estimated_price.numpy() * K, label="Estimated Price", color="blue", linestyle="-")
        axs[0].set_ylabel("Option Price")
        axs[0].legend(loc="best")

        axs[1].plot(spots, vol_curve_true(spots), label="True Vol Curve", color="green", linestyle="--")
        axs[1].plot(spots, latent_vol.numpy(), label="Inverted Latent Vol", color="red", linestyle="-")
        axs[1].set_ylabel("Volatility")
        axs[1].legend(loc="best")

        plt.xlabel("Spot Price")
        plt.tight_layout()
        plt.show()
