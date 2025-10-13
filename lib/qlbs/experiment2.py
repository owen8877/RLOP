import os
import pickle
from unittest import TestCase

import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from lib.util import _prefix, standard_to_normalized_price

from .bs import BSBaseline, BSPolicy
from .env import QLBSEnv
from .rl import NNBaseline, policy_gradient, NNBaseline_v2, NetHolder

mpl.use("TkAgg")
sns.set_style("whitegrid")
exp_name = os.path.basename(__file__)[:-3]


class Experiment2(TestCase):
    def _parameters(self):
        r = 1e-2
        mu = 0e-3
        sigma = 1e-1
        risk_lambdas = [0, 1, 2, 3]
        initial_price = 10
        strike_price = 10
        T = 5
        _dt = 0.2
        friction = 0e-1
        is_call_option = True
        max_step = int(np.round(T / _dt))
        mutation_lambda = 5e-2

        return (
            r,
            mu,
            sigma,
            risk_lambdas,
            initial_price,
            strike_price,
            T,
            _dt,
            friction,
            is_call_option,
            max_step,
            mutation_lambda,
        )

    def _path(self, risk_lambda, mutation_lambda, exp_name=exp_name):
        plan = f"risk_lambda={risk_lambda:d}, mutation_lambda={mutation_lambda:.2e}"
        return f"trained_model/{exp_name}/{plan}"

    def test_bs_value(self):
        (
            r,
            mu,
            sigma,
            risk_lambdas,
            initial_price,
            strike_price,
            T,
            _dt,
            friction,
            is_call_option,
            max_step,
            mutation_lambda,
        ) = self._parameters()

        for risk_lambda in risk_lambdas:
            mutate_status = []

            def mutate(env: QLBSEnv):
                if np.random.rand(1)[0] < mutation_lambda:
                    env.initial_asset_price = np.random.rand(1)[0] * 0.4 + 0.8
                    mutated = True
                else:
                    mutated = False
                mutate_status.append((len(mutate_status), mutated))

            path = self._path(risk_lambda, mutation_lambda)
            env = QLBSEnv(
                is_call_option=is_call_option,
                strike_price=strike_price,
                max_step=max_step,
                mu=mu,
                sigma=sigma,
                r=r,
                risk_lambda=risk_lambda,
                friction=friction,
                initial_asset_price=initial_price,
                risk_simulation_paths=200,
                _dt=_dt,
                mutation=mutate,
            )
            bs_policy = BSPolicy(is_call=is_call_option)
            holder = NetHolder("trained_model/pretrained-qlbs.pt")
            nn_baseline = NNBaseline_v2(holder, alpha=1e-4)

            collector = policy_gradient(
                env, bs_policy, nn_baseline, episode_n=1000, pi_frozen=True, tensorboard_label=f"qlbs-{exp_name}"
            )
            nn_baseline.save(f"{path}/baseline.pt")
            holder.save(f"{path}/holder.pt")
            with open(f"{path}/additional.pickle", "wb") as f:
                pickle.dump({"mutate_status": mutate_status, "collector": collector}, f)

    def test_draw_plot(self):
        (
            r,
            mu,
            sigma,
            risk_lambdas,
            initial_price,
            strike_price,
            T,
            _dt,
            friction,
            is_call_option,
            max_step,
            mutation_lambda,
        ) = self._parameters()

        baselines = dict()

        for exp in [2]:
            baselines[exp] = dict()
            for risk_lambda in risk_lambdas:
                path = self._path(risk_lambda, mutation_lambda, exp_name=exp_name.replace("2", str(exp)))
                baseline = NNBaseline_v2(alpha=0, from_filename=f"{path}/baseline.pt")
                baselines[exp][risk_lambda] = baseline

        def build_state_info_tensor(initial_price, current_time):
            current_time = torch.tensor(current_time)
            initial_price = torch.tensor(initial_price)
            state_info_tensor = torch.empty((len(current_time), 9))
            state_info_tensor[:, 0] = torch.tensor(
                standard_to_normalized_price(initial_price.numpy(), mu, sigma, current_time.numpy(), _dt)
            )  # normal_price
            state_info_tensor[:, 1] = current_time * _dt  # passed_real_time
            state_info_tensor[:, 2] = (max_step - current_time) * _dt  # remaining_real_time
            state_info_tensor[:, 3] = torch.tensor(
                standard_to_normalized_price(strike_price, mu, sigma, max_step, _dt)
            )  # normal_strike_price
            state_info_tensor[:, 4] = r  # r
            state_info_tensor[:, 5] = mu  # mu
            state_info_tensor[:, 6] = sigma  # sigma
            state_info_tensor[:, 7] = risk_lambda  # risk_lambda
            state_info_tensor[:, 8] = friction  # friction
            return state_info_tensor

        # Plot and compare prices learnt
        initial_prices = np.linspace(0.8, 1.2, 31)
        tensor = build_state_info_tensor(initial_price=initial_prices, current_time=np.zeros(len(initial_prices)))
        bs_baseline = BSBaseline(is_call=is_call_option)
        bs_price = bs_baseline.batch_estimate(tensor.numpy())
        bs_policy_prices, nn_prices = dict(), dict()

        for risk_lambda in risk_lambdas:
            bs_policy_prices[risk_lambda] = baselines[2][risk_lambda].batch_estimate(tensor)
            nn_prices[risk_lambda] = baselines[1][risk_lambda].batch_estimate(tensor)

        fig1 = plt.figure(figsize=(7, 4))
        palette = sns.color_palette()
        plt.plot(initial_prices, bs_price, label="BS", c=palette[-1], ls=":")
        for i, risk_lambda in enumerate(risk_lambdas):
            plt.plot(
                initial_prices,
                -bs_policy_prices[risk_lambda],
                c=palette[i],
                ls="--",
                label=rf"BS policy ($\lambda$={risk_lambda:d})",
            )
            plt.plot(
                initial_prices, -nn_prices[risk_lambda], c=palette[i], label=rf"NN policy ($\lambda$={risk_lambda:d})"
            )

        plt.legend(loc="best", title="method")
        plt.gca().set(xlabel=r"initial price $S_0$", ylabel="option price(negative return)")
        fig1.savefig(f"plot/{exp_name}/option-price.png")

        plt.show()
