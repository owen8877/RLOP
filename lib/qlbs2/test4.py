import os
from dataclasses import replace
from unittest import TestCase

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from ..util.pricing import bs_solve_implied_vol_from_call_price
from .bs import BSBaseline, BSPolicy
from .env import EnvSetup, QLBSEnv
from .nn_v3 import GaussianPolicy_v3, NNBaseline_v3
from .rl import policy_gradient

exp_name = os.path.basename(__file__)[:-3]


class Training(TestCase):
    def _parameters(self):
        T = 1
        parallel_n = 500
        return (
            EnvSetup(
                parallel_n=parallel_n,
                r=0.02,
                mus=np.random.randn(parallel_n) * 0.2,
                sigmas=np.random.rand(parallel_n) * 0.3 + 0.05,
                risk_lambdas=None,  # type: ignore
                strike_prices=np.random.rand(parallel_n) * 0.6 + 0.7,
                _dt=(_dt := 1 / 4),
                frictions=np.random.rand(parallel_n) * 2e-3,
                is_call_option=True,
                max_step=int(np.round(T / _dt)),
                initial_spots=np.random.rand(parallel_n) * 0.6 + 0.7,
                risk_simulation_paths=200,
                mutation=0.0,
            ),
            (risk_lambdas := [0.5]),
        )

    def _path(self, risk_lambda, exp_name=exp_name):
        plan = f"risk_lambda={risk_lambda:.1e}"
        return f"trained_model/{exp_name}/{plan}"

    def test_learn_both_from_0(self):
        setup, risk_lambdas = self._parameters()
        N_parallel = setup.parallel_n
        check_point = 1

        for risk_lambda in risk_lambdas:
            path = self._path(risk_lambda)

            env = QLBSEnv(replace(setup, risk_lambdas=np.array([risk_lambda] * N_parallel)))
            nn_baseline = NNBaseline_v3(
                lr=1e-4,
                from_filename=f"{path}/baseline_{check_point}.pt"
                if check_point
                else None,  # f"{self._path(risk_lambda, exp_name='test1')}/baseline_1.pt",
                # reset_optimizer=True,
            )
            nn_policy = GaussianPolicy_v3(
                lr=1e-4,
                is_call_option=setup.is_call_option,
                from_filename=f"{path}/policy_{check_point}.pt"
                if check_point
                else None,  # f"{self._path(risk_lambda, exp_name='test2')}/policy_1.pt",
                # reset_optimizer=True,
            )

            policy_gradient(
                env,
                nn_policy,
                nn_baseline,
                episode_n=200,
                tensorboard_label=f"qlbs2-{exp_name}-learn-both-from-0",
            )

            check_point_save = 1 if check_point is None else check_point + 1
            nn_policy.save(f"{path}/policy_{check_point_save}.pt")
            nn_baseline.save(f"{path}/baseline_{check_point_save}.pt")

    def test_draw_plot(self):
        check_point = 2

        setup, risk_lambdas = self._parameters()
        risk_lambda = risk_lambdas[0]
        is_call_option = True
        path = self._path(risk_lambda)

        T, K, r, sigma, friction = 1.0, 1.0, 0.02, 0.2, 1e-3
        initial_spots = np.linspace(0.7, 1.3, 50)

        _, axs = plt.subplots(3, 1, figsize=(7, 9), constrained_layout=True, sharex=True)
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
            state_info_tensor[:, 7] = risk_lambdas[0]  # risk_lambda
            state_info_tensor[:, 8] = friction  # friction
            return state_info_tensor

        print(path)
        nn_policy = GaussianPolicy_v3(is_call_option=is_call_option, from_filename=f"{path}/policy_{check_point}.pt")
        nn_baseline = NNBaseline_v3(from_filename=f"{path}/baseline_{check_point}.pt")

        for i, mu in enumerate([-0.4, -0.2, 0.0, 0.2, 0.4]):
            tensor = build_state_info_tensor(mu)

            if i == 0:
                bs_baseline = BSBaseline(is_call=is_call_option)
                bs_price = bs_baseline.batch_estimate(tensor)
                axs[0].plot(initial_spots, bs_price, label="BS Baseline", color=palette[0], linestyle="--")

                bs_policy = BSPolicy(is_call=is_call_option)
                bs_price = bs_policy.batch_action(tensor, random=False)
                axs[2].plot(initial_spots, bs_price, label="BS Policy", color=palette[0], linestyle="--")

            nn_price = -nn_baseline.batch_estimate(tensor.cuda()).cpu().numpy()
            axs[0].plot(initial_spots, nn_price, label=f"mu={mu}", color=palette[i + 1], linestyle="-")
            axs[0].set_ylabel("Option Price")

            IV = bs_solve_implied_vol_from_call_price(
                nn_price,
                initial_spots,
                initial_spots * 0 + T,
                initial_spots * 0 + K,
                initial_spots * 0 + r,
                sigma_init=sigma,
            )
            axs[1].plot(initial_spots, IV, label=f"mu={mu}", color=palette[i + 1], linestyle="-")
            axs[1].set_ylabel("Implied Volatility")

            nn_action = nn_policy.batch_action(tensor.cuda(), random=False, return_pre_action=False).cpu().numpy()  # type: ignore
            axs[2].plot(initial_spots, nn_action, label=f"mu={mu}", color=palette[i + 1], linestyle="-")
            axs[2].set_ylabel("Delta Action")

        axs[0].legend(loc="best")
        plt.show()
