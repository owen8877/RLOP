import os
import pickle
from unittest import TestCase

import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

import lib.util
from .bs import BSInitialEstimator, BSPolicy
from .env import RLOPEnv
from .rl import GaussianPolicy, policy_gradient_for_stacked

mpl.use("TkAgg")
sns.set_style("whitegrid")
exp_name = os.path.basename(__file__)[:-3]


class Experiment2(TestCase):
    def _parameters(self):
        r = 1e-2
        mu = 0e-3
        sigma = 1e-1
        initial_price = 1
        strike_price = 1
        T = 5
        _dt = 1
        frictions = [5e-2, 1e-1]
        is_call_option = True
        simplified = True
        max_step = int(np.round(T / _dt))
        mutation_lambda = 1e-2

        return (
            r,
            mu,
            sigma,
            initial_price,
            strike_price,
            T,
            _dt,
            frictions,
            is_call_option,
            simplified,
            max_step,
            mutation_lambda,
        )

    def _path(self, simplified, friction):
        plan = f"friction={friction:.2f}"
        return f"trained_model/{exp_name}/{lib.util._prefix(simplified, plan)}"

    def test_tc(self):
        (
            r,
            mu,
            sigma,
            initial_price,
            strike_price,
            T,
            _dt,
            frictions,
            is_call_option,
            simplified,
            max_step,
            mutation_lambda,
        ) = self._parameters()

        for friction in frictions:
            mutate_status = []

            def mutate(env: RLOPEnv):
                if np.random.rand(1)[0] < mutation_lambda:
                    env.initial_asset_price = np.random.rand(1)[0] * 0.4 + 0.8
                    mutated = True
                else:
                    mutated = False
                mutate_status.append((len(mutate_status), mutated))

            path = self._path(simplified, friction)
            env = RLOPEnv(
                is_call_option=is_call_option,
                strike_price=strike_price,
                max_step=max_step,
                mu=mu,
                sigma=sigma,
                r=r,
                friction=friction,
                initial_estimator=BSInitialEstimator(is_call_option),
                initial_asset_price=initial_price,
                _dt=_dt,
                mutation=mutate,
            )
            nn_policy = GaussianPolicy(
                simplified=simplified,
                alpha=1e-4,
                # from_filename='trained_model/experiment1/simplified_mutation_lambda=1.00e-02/policy.pt')
                from_filename=f"{path}/policy.pt",
            )

            collector = policy_gradient_for_stacked(env, nn_policy, batch=True, plot=True, episode_n=10000)
            nn_policy.save(f"{path}/policy.pt")
            with open(f"{path}/additional.pickle", "wb") as f:
                pickle.dump({"mutate_status": mutate_status, "collector": collector}, f)

    def test_draw_plot(self):
        (
            r,
            mu,
            sigma,
            initial_price,
            strike_price,
            T,
            _dt,
            frictions,
            is_call_option,
            simplified,
            max_step,
            mutation_lambda,
        ) = self._parameters()

        policies = dict()
        policies[0] = GaussianPolicy(
            simplified=simplified,
            alpha=1e-4,
            from_filename="trained_model/experiment1/simplified_mutation_lambda=1.00e-02/policy.pt",
        )

        for friction in frictions:
            path = self._path(simplified, friction)
            policies[friction] = GaussianPolicy(simplified=simplified, alpha=1e-4, from_filename=f"{path}/policy.pt")

        all_frictions = [0, *frictions]
        initial_estimator = BSInitialEstimator(is_call_option)

        def build_state_info_tensor(price, current_time):
            current_time = torch.tensor(current_time)
            price = torch.tensor(price)
            state_info_tensor = torch.empty((len(current_time), 9))
            S = lib.util.standard_to_normalized_price(
                price.numpy(), mu, sigma, current_time.numpy(), _dt
            )  # normal_price
            state_info_tensor[:, 0] = torch.tensor(S)
            state_info_tensor[:, 1] = (max_step - current_time) * _dt  # remaining_real_time
            state_info_tensor[:, 2] = torch.tensor(
                initial_estimator(price.numpy(), strike_price, max_step - current_time.numpy(), r, sigma, _dt)
            )  # portfolio value
            state_info_tensor[:, 3] = torch.tensor(
                lib.util.standard_to_normalized_price(strike_price, mu, sigma, max_step, _dt)
            )  # normal_strike_price
            state_info_tensor[:, 4] = r  # r
            state_info_tensor[:, 5] = mu  # mu
            state_info_tensor[:, 6] = sigma  # sigma
            state_info_tensor[:, 7] = friction  # friction
            return state_info_tensor

        # Plot and compare prices learnt
        initial_prices = np.linspace(0.8, 1.2, 31)
        times = np.arange(max_step) * _dt

        bs_hedges = dict()
        nn_hedges = dict()
        bs_policy = BSPolicy(is_call=is_call_option)

        for t in times:
            tensor = build_state_info_tensor(price=initial_prices, current_time=np.zeros(len(initial_prices)) + t)
            bs_hedges[t] = bs_policy.batch_action(tensor.numpy(), random=False, passed_real_time=t)
            nn_hedges[t] = dict()
            for friction in all_frictions:
                nn_hedges[t][friction] = policies[friction].batch_action(tensor, random=False).numpy()

        fig1, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        palette = sns.color_palette()
        for j, friction in enumerate(all_frictions):
            for i, t in enumerate(times):
                axs[j].plot(
                    initial_prices,
                    nn_hedges[t][friction],
                    label=f"{T - t:d}(NN)" if j == 0 else None,
                    c=palette[i],
                    alpha=1 - 0.0 * j,
                )
                axs[j].set_title(rf"$\epsilon={friction:.2f}$")
        axs[0].legend(loc="best", title="remaining time")
        for j in range(len(all_frictions)):
            axs[j].set(xlabel="asset price", ylabel="hedge position" if j == 0 else None)
        fig1.savefig(f"plot/{exp_name}/hedge_position.png")
        plt.show()
