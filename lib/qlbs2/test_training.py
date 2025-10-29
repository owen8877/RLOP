import os
import pickle
from dataclasses import replace
from itertools import chain
from unittest import TestCase

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.optim.adam import Adam

from ..util.pricing import bs_solve_implied_vol_from_call_price
from ..util import standard_to_normalized_price
from ..util.sample import discrete_OU_process
from .bs import BSBaseline, BSPolicy
from .env import EnvSetup, QLBSEnv
from . import nn_v1
from .nn_v2 import GaussianPolicy_v2, NetHolder, NNBaseline_v2
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
                mus=np.random.randn(parallel_n) * 0.1,
                sigmas=np.random.rand(parallel_n) * 1.0 + 0.05,
                risk_lambdas=None,  # type: ignore
                strike_prices=np.random.rand(parallel_n) * 1.2 + 0.4,
                _dt=(_dt := 1 / 4),
                frictions=np.random.rand(parallel_n) * 1e-3,
                is_call_option=True,
                max_step=int(np.round(T / _dt)),
                initial_spots=np.random.rand(parallel_n) * 1.2 + 0.4,
                risk_simulation_paths=200,
                mutation=0.0,
            ),
            (risk_lambdas := [1]),
        )

    def _path(self, risk_lambda):
        plan = f"risk_lambda={risk_lambda:.1e}"
        return f"trained_model/{exp_name}/{plan}"

    def test_separate_net(self):
        setup, risk_lambdas = self._parameters()
        N_parallel = setup.parallel_n
        check_point = 1
        mutation_lambda = setup.mutation

        for risk_lambda in risk_lambdas:
            mutate_status = []

            def mutate(env: QLBSEnv):
                if np.random.rand(1)[0] < mutation_lambda:
                    env.initial_spots = np.random.rand(N_parallel) * 2 + 0.2
                    env.info.mus = np.random.randn(N_parallel) * 1.0
                    env.info.strike_prices = np.random.rand(N_parallel) * 2 + 0.2
                    env.info.sigmas = np.random.rand(N_parallel) * 0.4 + 0.05
                    env.info.frictions = np.random.rand(N_parallel) * 5e-3
                    env.info.r = discrete_OU_process(decay=3e-2, std=0.1, mu=-1.5, log10=True)(env.info.r)
                    mutated = True
                else:
                    mutated = False
                mutate_status.append((len(mutate_status), mutated))

            path = self._path(risk_lambda)

            env = QLBSEnv(replace(setup, risk_lambdas=np.array([risk_lambda] * N_parallel), mutation=mutate))
            holder1 = nn_v1.NetHolder(
                f"{path}/holder1_{check_point}.pt" if check_point else "trained_model/pretrained-qlbs.pt"
            )
            holder2 = nn_v1.NetHolder(
                f"{path}/holder2_{check_point}.pt" if check_point else "trained_model/pretrained-qlbs.pt"
            )
            nn_policy = nn_v1.GaussianPolicy_v1(
                holder1,
                is_call_option=True,
                alpha=1e-4,
                from_filename=f"{path}/policy_{check_point}.pt" if check_point else None,
            )
            bs_policy = BSPolicy(is_call=True)
            nn_baseline = nn_v1.NNBaseline_v1(holder2, alpha=1e-4)
            bs_baseline = BSBaseline(is_call=True)

            collector = policy_gradient(
                env,
                nn_policy,
                nn_baseline,
                episode_n=200,
                tensorboard_label=f"qlbs2-separate-{exp_name}",
                # V_frozen=True,
                # pi_frozen=True,
            )

            check_point_save = 1 if check_point is None else check_point + 1
            nn_policy.save(f"{path}/policy_{check_point_save}.pt")
            holder1.save(f"{path}/holder1_{check_point_save}.pt")
            holder2.save(f"{path}/holder2_{check_point_save}.pt")
            with open(f"{path}/additional_{check_point_save}.pickle", "wb") as f:
                pickle.dump({"mutate_status": mutate_status, "collector": collector}, f)

    def test_together_net(self):
        setup, risk_lambdas = self._parameters()
        N_parallel = setup.parallel_n
        check_point = 2
        mutation_lambda = setup.mutation

        for risk_lambda in risk_lambdas:
            mutate_status = []

            def mutate(env: QLBSEnv):
                if np.random.rand(1)[0] < mutation_lambda:
                    env.initial_spots = np.random.rand(N_parallel) * 2 + 0.2
                    env.info.mus = np.random.randn(N_parallel) * 1.0
                    env.info.strike_prices = np.random.rand(N_parallel) * 2 + 0.2
                    env.info.sigmas = np.random.rand(N_parallel) * 0.4 + 0.05
                    env.info.frictions = np.random.rand(N_parallel) * 5e-3
                    env.info.r = discrete_OU_process(decay=3e-2, std=0.1, mu=-1.5, log10=True)(env.info.r)
                    mutated = True
                else:
                    mutated = False
                mutate_status.append((len(mutate_status), mutated))

            path = self._path(risk_lambda)

            env = QLBSEnv(replace(setup, risk_lambdas=np.array([risk_lambda] * N_parallel), mutation=mutate))
            holder = NetHolder(
                f"{path}/holderT_{check_point}.pt" if check_point else "trained_model/pretrained-qlbs.pt"
            )

            nn_policy = GaussianPolicy_v2(
                holder,
                is_call_option=True,
                from_filename=f"{path}/policyT_{check_point}.pt" if check_point else None,
            )
            bs_policy = BSPolicy(is_call=True)
            nn_baseline = NNBaseline_v2(holder)
            bs_baseline = BSBaseline(is_call=True)

            optimizer = Adam(chain(holder.net.parameters(), nn_policy.theta_sigma.parameters()), lr=1e-4)
            nn_policy.set_optimizer(optimizer)
            nn_baseline.set_optimizer(optimizer)

            collector = policy_gradient(
                env,
                bs_policy,
                nn_baseline,
                episode_n=200,
                tensorboard_label=f"qlbs2-separate-{exp_name}",
                # V_frozen=True,
                pi_frozen=True,
            )

            check_point_save = 1 if check_point is None else check_point + 1
            nn_policy.save(f"{path}/policyT_{check_point_save}.pt")
            holder.save(f"{path}/holderT_{check_point_save}.pt")
            with open(f"{path}/additional_{check_point_save}.pickle", "wb") as f:
                pickle.dump({"mutate_status": mutate_status, "collector": collector}, f)

    def test_draw_plot(self):
        setup, risk_lambdas = self._parameters()
        N_parallel = setup.parallel_n
        risk_lambda = risk_lambdas[0]
        is_call_option = True
        path = self._path(risk_lambda)
        check_point = 2

        # Start to plot the fixed part
        fig1, axs1 = plt.subplots(3, 1, figsize=(7, 9), constrained_layout=True, sharex=True)
        palette = sns.color_palette()
        deep_palette = sns.color_palette("deep")

        # plot option price - strike plot
        def build_state_info_tensor(initial_spot, mu):
            initial_spot = torch.tensor(initial_spot)
            state_info_tensor = torch.empty((len(initial_spot), 9))
            state_info_tensor[:, 0] = initial_spot  # spot
            state_info_tensor[:, 1] = 0  # passed_real_time
            state_info_tensor[:, 2] = 1.0  # remaining_real_time
            state_info_tensor[:, 3] = 1.0  # strike
            state_info_tensor[:, 4] = 2e-2  # r
            state_info_tensor[:, 5] = mu  # mu
            state_info_tensor[:, 6] = 0.2  # sigma
            state_info_tensor[:, 7] = risk_lambdas[0]  # risk_lambda
            state_info_tensor[:, 8] = 1e-3  # friction
            return state_info_tensor

        initial_prices = np.linspace(0.6, 1.4, 50)

        holder1 = nn_v1.NetHolder(f"{path}/holder1_{check_point}.pt")
        holder2 = nn_v1.NetHolder(f"{path}/holder2_{check_point}.pt")
        nn_policy = nn_v1.GaussianPolicy_v1(
            holder1, is_call_option, alpha=1e-4, from_filename=f"{path}/policy_{check_point}.pt"
        )
        nn_baseline = nn_v1.NNBaseline_v1(holder2, alpha=1e-4)
        # Plot and compare delta learnt

        sigma_test = 0.4

        for i, mu in enumerate([-0.2, 0.0, 0.2]):
            tensor = build_state_info_tensor(initial_spot=initial_prices, mu=mu)

            if i == 0:
                bs_baseline = BSBaseline(is_call=is_call_option)
                bs_price = bs_baseline.batch_estimate(tensor)
                axs1[0].plot(initial_prices, bs_price, label="BS Baseline", color=palette[0], linestyle="--")
                bs_policy = BSPolicy(is_call=is_call_option)
                bs_price = bs_policy.batch_action(tensor, random=False)
                axs1[1].plot(initial_prices, bs_price, label="BS Policy", color=palette[0], linestyle="--")

            nn_price = -nn_baseline.batch_estimate(tensor.cuda()).cpu().numpy()
            axs1[0].plot(initial_prices, nn_price, label=f"NN Baseline (mu={mu})", color=palette[i + 1], linestyle="-")

            IV = bs_solve_implied_vol_from_call_price(
                nn_price,
                initial_prices,
                initial_prices * 0 + 1.0,
                initial_prices * 0 + 1.0,
                initial_prices * 0 + 0.02,
                sigma_init=sigma_test,
            )
            axs1[2].plot(initial_prices, IV, label=f"Implied Vol (mu={mu})", color=palette[i + 1], linestyle="-")

            path = self._path(risk_lambda)
            nn_price = nn_policy.batch_action(tensor.cuda(), random=False).cpu().numpy()
            axs1[1].plot(initial_prices, nn_price, label=f"NN Policy (mu={mu})", color=palette[i + 1], linestyle="-")
        ###################

        plt.legend(loc="best")
        plt.show()

    def test_draw_plotT(self):
        setup, risk_lambdas = self._parameters()
        N_parallel = setup.parallel_n
        risk_lambda = risk_lambdas[0]
        is_call_option = True
        path = self._path(risk_lambda)
        check_point = 3

        # Start to plot the fixed part
        fig1, axs1 = plt.subplots(3, 1, figsize=(7, 9), constrained_layout=True, sharex=True)
        palette = sns.color_palette()
        deep_palette = sns.color_palette("deep")

        # plot option price - strike plot
        def build_state_info_tensor(initial_spot, mu, sigma):
            initial_spot = torch.tensor(initial_spot)
            state_info_tensor = torch.empty((len(initial_spot), 9))
            state_info_tensor[:, 0] = initial_spot  # spot
            state_info_tensor[:, 1] = 0  # passed_real_time
            state_info_tensor[:, 2] = 1.0  # remaining_real_time
            state_info_tensor[:, 3] = 1.0  # strike
            state_info_tensor[:, 4] = 2e-2  # r
            state_info_tensor[:, 5] = mu  # mu
            state_info_tensor[:, 6] = sigma  # sigma
            state_info_tensor[:, 7] = risk_lambdas[0]  # risk_lambda
            state_info_tensor[:, 8] = 1e-3  # friction
            return state_info_tensor

        initial_prices = np.linspace(0.4, 1.6, 50)

        holder = NetHolder(f"{path}/holderT_{check_point}.pt")
        nn_policy = GaussianPolicy_v2(holder, is_call_option, from_filename=f"{path}/policyT_{check_point}.pt")
        nn_baseline = NNBaseline_v2(holder)

        sigma_test = 0.1

        # Plot and compare delta learnt
        for i, mu in enumerate([-0.2, 0.0, 0.2]):
            tensor = build_state_info_tensor(initial_spot=initial_prices, mu=mu, sigma=sigma_test)

            if i == 0:
                bs_baseline = BSBaseline(is_call=is_call_option)
                bs_price = bs_baseline.batch_estimate(tensor)
                axs1[0].plot(initial_prices, bs_price, label="BS Baseline", color=palette[0], linestyle="--")
                bs_policy = BSPolicy(is_call=is_call_option)
                bs_price = bs_policy.batch_action(tensor, random=False)
                axs1[1].plot(initial_prices, bs_price, label="BS Policy", color=palette[0], linestyle="--")

            nn_price = -nn_baseline.batch_estimate(tensor.cuda()).cpu().numpy()
            axs1[0].plot(initial_prices, nn_price, label=f"NN Baseline (mu={mu})", color=palette[i + 1], linestyle="-")

            IV = bs_solve_implied_vol_from_call_price(
                nn_price,
                initial_prices,
                initial_prices * 0 + 1.0,
                initial_prices * 0 + 1.0,
                initial_prices * 0 + 0.02,
                sigma_init=sigma_test,
            )
            axs1[2].plot(initial_prices, IV, label=f"Implied Vol (mu={mu})", color=palette[i + 1], linestyle="-")

            path = self._path(risk_lambda)
            nn_price = nn_policy.batch_action(tensor.cuda(), random=False).cpu().numpy()
            axs1[1].plot(initial_prices, nn_price, label=f"NN Policy (mu={mu})", color=palette[i + 1], linestyle="-")

        plt.legend(loc="best")
        plt.show()
