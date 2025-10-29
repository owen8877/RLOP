from curses.ascii import BS
from dataclasses import replace
import dis
import os
import pickle
from unittest import TestCase

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
import torch

from .bs import BSBaseline, BSPolicy

from ..util import standard_to_normalized_price, EMACollector
from ..util.sample import discrete_OU_process
from .env import QLBSEnv, Info, EnvSetup
from .rl import GaussianPolicy_v2, NNBaseline_v2, policy_gradient, NetHolder

exp_name = os.path.basename(__file__)[:-3]


class Experiment11(TestCase):
    def _parameters(self):
        T = 1
        return (
            EnvSetup(
                r=0.02,
                mu=0.05,
                sigma=0.1,
                risk_lambda=None,  # type: ignore
                strike_price=1,
                _dt=(_dt := 1 / 4),
                friction=0e-3,
                is_call_option=True,
                max_step=int(np.round(T / _dt)),
                initial_asset_price=1,
                risk_simulation_paths=200,
                mutation=0.5,
            ),
            (risk_lambdas := [0.5]),
        )

    def _path(self, risk_lambda):
        plan = f"risk_lambda={risk_lambda:.1e}"
        return f"trained_model/{exp_name}/{plan}"

    def test_ivbs(self):
        setup, risk_lambdas = self._parameters()
        check_point = 1
        mutation_lambda = setup.mutation

        for risk_lambda in risk_lambdas:
            mutate_status = []

            def mutate(env: QLBSEnv):
                if np.random.rand(1)[0] < mutation_lambda:
                    env.initial_asset_price = np.clip(
                        np.pow(10.0, discrete_OU_process(decay=3e-2, std=0.5, mu=1)(np.log10(env.initial_asset_price))),
                        5,
                        15,
                    )
                    env.info.mu = discrete_OU_process(decay=3e-1, std=0.2, mu=0)(env.info.mu)
                    env.info.strike_price = np.clip(
                        np.pow(10.0, discrete_OU_process(decay=3e-2, std=0.5, mu=1)(np.log10(env.info.strike_price))),
                        5,
                        15,
                    )
                    env.info.r = np.pow(10.0, discrete_OU_process(decay=3e-2, std=0.1, mu=-1.5)(np.log10(env.info.r)))
                    mutated = True
                else:
                    mutated = False
                mutate_status.append((len(mutate_status), mutated))

            path = self._path(risk_lambda)

            env = QLBSEnv(replace(setup, risk_lambda=risk_lambda, mutation=mutate))

            bs_policy = BSPolicy(is_call=True)
            bs_baseline = BSBaseline(is_call=True)

            collector = policy_gradient(
                env,
                bs_policy,
                bs_baseline,
                episode_n=500,
                tensorboard_label=f"qlbs-{exp_name}",
                V_frozen=True,
                pi_frozen=True,
            )

    def test_iv15(self):
        setup, risk_lambdas = self._parameters()
        check_point = 1
        mutation_lambda = setup.mutation

        for risk_lambda in risk_lambdas:
            mutate_status = []

            def mutate(env: QLBSEnv):
                if np.random.rand(1)[0] < mutation_lambda:
                    env.initial_asset_price = np.clip(
                        np.pow(10.0, discrete_OU_process(decay=3e-2, std=0.5, mu=1)(np.log10(env.initial_asset_price))),
                        5,
                        15,
                    )
                    env.info.mu = discrete_OU_process(decay=3e-1, std=0.2, mu=0)(env.info.mu)
                    env.info.strike_price = np.clip(
                        np.pow(10.0, discrete_OU_process(decay=3e-2, std=0.5, mu=1)(np.log10(env.info.strike_price))),
                        5,
                        15,
                    )
                    env.info.r = np.pow(10.0, discrete_OU_process(decay=3e-2, std=0.1, mu=-1.5)(np.log10(env.info.r)))
                    mutated = True
                else:
                    mutated = False
                mutate_status.append((len(mutate_status), mutated))

            path = self._path(risk_lambda)

            env = QLBSEnv(replace(setup, risk_lambda=risk_lambda, mutation=mutate))
            holder1 = NetHolder(
                None
                # f"{path}/holder1_{check_point}.pt" if check_point else "trained_model/pretrained-qlbs.pt"
            )
            holder2 = NetHolder(
                None
                # f"{path}/holder2_{check_point}.pt" if check_point else "trained_model/pretrained-qlbs.pt"
            )
            nn_policy = GaussianPolicy_v2(
                holder1,
                is_call_option=True,
                alpha=1e-4,
                from_filename=f"{path}/policy_{check_point}.pt" if check_point else None,
            )
            bs_policy = BSPolicy(is_call=True)
            nn_baseline = NNBaseline_v2(holder2, alpha=1e-4)
            # nn_baseline = BSBaseline(is_call=True)

            collector = policy_gradient(
                env,
                nn_policy,
                nn_baseline,
                episode_n=500,
                tensorboard_label=f"qlbs-{exp_name}",
                # V_frozen=True,
                # pi_frozen=True,
            )

            check_point_save = 1 if check_point is None else check_point + 1
            nn_policy.save(f"{path}/policy_{check_point_save}.pt")
            # nn_baseline.save(f"{path}/baseline_{check_point_save}.pt")
            holder1.save(f"{path}/holder1_{check_point_save}.pt")
            holder2.save(f"{path}/holder2_{check_point_save}.pt")
            with open(f"{path}/additional_{check_point_save}.pickle", "wb") as f:
                pickle.dump({"mutate_status": mutate_status, "collector": collector}, f)

    def test_iv(self):
        setup, risk_lambdas = self._parameters()
        check_point = 3
        mutation_lambda = setup.mutation

        for risk_lambda in risk_lambdas:
            mutate_status = []

            def mutate(env: QLBSEnv):
                if np.random.rand(1)[0] < mutation_lambda:
                    env.initial_asset_price = np.clip(
                        np.pow(10.0, discrete_OU_process(decay=4e-2, std=0.1, mu=0)(np.log10(env.initial_asset_price))),
                        0.6,
                        1.4,
                    )
                    env.info.mu = discrete_OU_process(decay=2e-1, std=0.2, mu=0)(env.info.mu)
                    env.info.strike_price = np.clip(
                        np.pow(10.0, discrete_OU_process(decay=4e-2, std=0.1, mu=0)(np.log10(env.info.strike_price))),
                        0.6,
                        1.4,
                    )
                    env.info.r = np.pow(10.0, discrete_OU_process(decay=2e-2, std=0.2, mu=-1.5)(np.log10(env.info.r)))
                    mutated = True
                else:
                    mutated = False
                mutate_status.append((len(mutate_status), mutated))

            path = self._path(risk_lambda)

            env = QLBSEnv(replace(setup, risk_lambda=risk_lambda, mutation=mutate))
            holder = NetHolder(
                f"{path}/holderA_{check_point}.pt" if check_point else "trained_model/pretrained-qlbs.pt"
            )
            nn_policy = GaussianPolicy_v2(
                holder,
                is_call_option=True,
                alpha=1e-4,
                from_filename=f"{path}/policyA_{check_point}.pt" if check_point else None,
            )
            nn_baseline = NNBaseline_v2(holder, alpha=1e-4)

            collector = policy_gradient(
                env,
                nn_policy,
                nn_baseline,
                episode_n=500,
                tensorboard_label=f"qlbs-{exp_name}",
            )

            check_point_save = 1 if check_point is None else check_point + 1
            nn_policy.save(f"{path}/policyA_{check_point_save}.pt")
            holder.save(f"{path}/holderA_{check_point_save}.pt")
            with open(f"{path}/additionalA_{check_point_save}.pickle", "wb") as f:
                pickle.dump({"mutate_status": mutate_status, "collector": collector}, f)

    def test_draw_plot(self):
        setup, risk_lambdas = self._parameters()
        risk_lambda = risk_lambdas[0]
        is_call_option = True

        # Start to plot the fixed part
        fig1, axs1 = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True, sharex=True)
        palette = sns.color_palette()
        deep_palette = sns.color_palette("deep")

        # plot option price - strike plot
        def build_state_info_tensor(initial_price, current_time):
            current_time = torch.tensor(current_time)
            initial_price = torch.tensor(initial_price)
            state_info_tensor = torch.empty((len(current_time), 9))
            state_info_tensor[:, 0] = torch.tensor(
                standard_to_normalized_price(
                    initial_price.numpy(), setup.mu, setup.sigma, current_time.numpy(), setup._dt
                )
            )  # normal_price
            state_info_tensor[:, 1] = current_time * setup._dt  # passed_real_time
            state_info_tensor[:, 2] = (setup.max_step - current_time) * setup._dt  # remaining_real_time
            state_info_tensor[:, 3] = torch.tensor(
                standard_to_normalized_price(setup.strike_price, setup.mu, setup.sigma, setup.max_step, setup._dt)
            )  # normal_strike_price
            state_info_tensor[:, 4] = setup.r  # r
            state_info_tensor[:, 5] = setup.mu  # mu
            state_info_tensor[:, 6] = setup.sigma  # sigma
            state_info_tensor[:, 7] = risk_lambda  # risk_lambda
            state_info_tensor[:, 8] = setup.friction  # friction
            return state_info_tensor

        initial_prices = np.linspace(6, 14, 100)

        # Plot and compare prices learnt
        tensor = build_state_info_tensor(initial_price=initial_prices, current_time=np.zeros(len(initial_prices)))
        bs_baseline = BSBaseline(is_call=is_call_option)
        bs_price = bs_baseline.batch_estimate(tensor.numpy())
        axs1[0].plot(initial_prices, bs_price, label="BS Baseline", color=palette[0], linestyle="--")

        path = self._path(risk_lambda)
        check_point = 1

        # holder = NetHolder("trained_model/pretrained-qlbs.pt")
        # nn_policy = GaussianPolicy_v2(holder, is_call_option, alpha=1e-4, from_filename=None)
        # nn_baseline = NNBaseline_v2(holder, alpha=1e-4)

        # holder = NetHolder(f"{path}/holder_{check_point}.pt")
        # nn_policy = GaussianPolicy_v2(
        #     holder, is_call_option, alpha=1e-4, from_filename=f"{path}/policy_{check_point}.pt"
        # )
        # nn_baseline = NNBaseline_v2(holder, alpha=1e-4)

        holder1 = NetHolder(f"{path}/holder1_{check_point}.pt")
        holder2 = NetHolder(f"{path}/holder2_{check_point}.pt")
        nn_policy = GaussianPolicy_v2(
            holder1, is_call_option, alpha=1e-4, from_filename=f"{path}/policy_{check_point}.pt"
        )
        nn_baseline = NNBaseline_v2(holder2, alpha=1e-4)

        nn_price = nn_baseline.batch_estimate(tensor.cuda()).cpu().numpy()
        axs1[0].plot(initial_prices, -nn_price, label="NN Baseline", color=palette[1], linestyle="-.")

        # Plot and compare delta learnt
        bs_policy = BSPolicy(is_call=is_call_option)
        bs_price = bs_policy.batch_action(tensor.numpy(), random=False)
        axs1[1].plot(initial_prices, bs_price, label="BS Policy", color=palette[0], linestyle="--")

        path = self._path(risk_lambda)
        nn_price = nn_policy.batch_action(tensor.cuda(), random=False).cpu().numpy()
        axs1[1].plot(initial_prices, nn_price, label="NN Policy", color=palette[1], linestyle="-.")

        ###################

        # holderA = NetHolder(f"{path}/holderA_{check_point}.pt")
        # nn_policyA = GaussianPolicy_v2(
        #     holderA, is_call_option, alpha=1e-4, from_filename=f"{path}/policyA_{check_point}.pt"
        # )
        # nn_baselineA = NNBaseline_v2(holderA, alpha=1e-4)

        # nn_price = nn_baselineA.batch_estimate(tensor.cuda()).cpu().numpy()
        # axs1[0].plot(initial_prices, nn_price, label="NN BaselineA", color=palette[2], linestyle="-")

        # check_point = 1
        # nn_price = nn_policyA.batch_action(tensor.cuda(), random=False).cpu().numpy()
        # axs1[1].plot(initial_prices, nn_price, label="NN PolicyA", color=palette[2], linestyle="-")

        plt.legend(loc="best")
        plt.show()

    def test_draw_plot2(self):
        setup, risk_lambdas = self._parameters()
        is_call_option = True

        # Start to plot the fixed part
        fig1, axs1 = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True, sharex=True)
        palette = sns.color_palette()
        deep_palette = sns.color_palette("deep")

        check_point = 4

        # plot option price - strike plot
        def build_state_info_tensor(initial_price, current_time, risk_lambda):
            current_time = torch.tensor(current_time)
            initial_price = torch.tensor(initial_price)
            state_info_tensor = torch.empty((len(current_time), 9))
            state_info_tensor[:, 0] = torch.tensor(
                standard_to_normalized_price(
                    initial_price.numpy(), setup.mu, setup.sigma, current_time.numpy(), setup._dt
                )
            )  # normal_price
            state_info_tensor[:, 1] = current_time * setup._dt  # passed_real_time
            state_info_tensor[:, 2] = (setup.max_step - current_time) * setup._dt  # remaining_real_time
            state_info_tensor[:, 3] = torch.tensor(
                standard_to_normalized_price(setup.strike_price, setup.mu, setup.sigma, setup.max_step, setup._dt)
            )  # normal_strike_price
            state_info_tensor[:, 4] = setup.r  # r
            state_info_tensor[:, 5] = setup.mu  # mu
            state_info_tensor[:, 6] = setup.sigma  # sigma
            state_info_tensor[:, 7] = risk_lambda  # risk_lambda
            state_info_tensor[:, 8] = setup.friction  # friction
            return state_info_tensor

        initial_prices = np.linspace(0.6, 1.4, 100)

        for i, risk_lambda in enumerate(risk_lambdas):
            # Plot and compare prices learnt
            tensor = build_state_info_tensor(
                initial_price=initial_prices, current_time=np.zeros(len(initial_prices)), risk_lambda=risk_lambda
            )
            if i == 0:
                bs_baseline = BSBaseline(is_call=is_call_option)
                bs_price = bs_baseline.batch_estimate(tensor.numpy())
                axs1[0].plot(initial_prices, bs_price, label=f"BS Baseline", color=palette[i * 2], linestyle="--")

                # Plot and compare delta learnt
                bs_policy = BSPolicy(is_call=is_call_option)
                bs_price = bs_policy.batch_action(tensor.numpy(), random=False)
                axs1[1].plot(initial_prices, bs_price, label=f"BS Policy", color=palette[i * 2], linestyle="--")

            ###################
            path = self._path(risk_lambda)

            holderA = NetHolder(f"{path}/holderA_{check_point}.pt")
            # holderA = NetHolder("trained_model/pretrained-qlbs.pt")
            nn_policyA = GaussianPolicy_v2(
                holderA, is_call_option, alpha=1e-4, from_filename=f"{path}/policyA_{check_point}.pt"
            )
            nn_baselineA = NNBaseline_v2(holderA, alpha=1e-4)

            nn_price = nn_baselineA.batch_estimate(tensor.cuda()).cpu().numpy()
            axs1[0].plot(
                initial_prices, -nn_price, label=f"NN Baseline {risk_lambda}", color=palette[i + 1], linestyle="-"
            )

            nn_delta = nn_policyA.batch_action(tensor.cuda(), random=False).cpu().numpy()
            axs1[1].plot(
                initial_prices, nn_delta, label=f"NN PolicyA {risk_lambda}", color=palette[i + 1], linestyle="-"
            )

        plt.legend(loc="best")
        plt.show()
