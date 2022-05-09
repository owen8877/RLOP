import os
import pickle
from unittest import TestCase

import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

import util
from qlbs.bs import BSPolicy, BSBaseline
from qlbs.env import QLBSEnv
from qlbs.rl import GaussianPolicy, NNBaseline, policy_gradient

mpl.use('TkAgg')
sns.set_style('whitegrid')
exp_name = os.path.basename(__file__)[:-3]


class Experiment3(TestCase):
    def _parameters(self):
        r = 1e-2
        mu = 0e-3
        sigma = 1e-1
        risk_lambda = 0.5
        initial_price = 1
        strike_price = 1
        T = 5
        _dt = 1
        frictions = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        is_call_option = True
        simplified = True
        max_step = int(np.round(T / _dt))
        mutation_lambda = 1e-2

        return r, mu, sigma, risk_lambda, initial_price, strike_price, T, _dt, frictions, is_call_option, simplified, \
               max_step, mutation_lambda

    def _path(self, simplified, friction, exp_name=exp_name):
        plan = f'friction={friction:.2e}'
        return f'trained_model/{exp_name}/{util._prefix(simplified, plan)}'

    def test_tc(self):
        r, mu, sigma, risk_lambda, initial_price, strike_price, T, _dt, frictions, is_call_option, simplified, \
        max_step, mutation_lambda = self._parameters()
        reuse_training = True

        for friction in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:  # frictions:
            mutate_status = []

            def mutate(env: QLBSEnv):
                if np.random.rand(1)[0] < mutation_lambda:
                    env.initial_asset_price = np.random.rand(1)[0] * 0.4 + 0.8
                    mutated = True
                else:
                    mutated = False
                mutate_status.append((len(mutate_status), mutated))

            # load_path = self._path(simplified, friction)
            load_path = self._path(simplified, 0.3)
            env = QLBSEnv(is_call_option=is_call_option, strike_price=strike_price, max_step=max_step, mu=mu,
                          sigma=sigma, r=r, risk_lambda=risk_lambda, friction=friction,
                          initial_asset_price=initial_price, risk_simulation_paths=200, _dt=_dt, mutation=mutate)
            nn_policy = GaussianPolicy(simplified=simplified, alpha=1e-4,
                                       from_filename=f'{load_path}/policy.pt' if reuse_training else None)
            nn_baseline = NNBaseline(simplified=simplified, alpha=1e-4,
                                     from_filename=f'{load_path}/baseline.pt' if reuse_training else None)

            save_path = self._path(simplified, friction)
            collector = policy_gradient(env, nn_policy, nn_baseline, episode_n=4000, plot=True)
            nn_policy.save(f'{save_path}/policy.pt')
            nn_baseline.save(f'{save_path}/baseline.pt')
            with open(f'{save_path}/additional.pickle', 'wb') as f:
                pickle.dump({'mutate_status': mutate_status, 'collector': collector}, f)

    def test_draw_plot(self):
        r, mu, sigma, friction, initial_price, strike_price, T, _dt, frictions, is_call_option, simplified, \
        max_step, mutation_lambda = self._parameters()

        baselines = dict()
        policies = dict()

        for friction in frictions:
            path = self._path(simplified, friction)
            baselines[friction] = NNBaseline(simplified=simplified, alpha=0, from_filename=f'{path}/baseline.pt')
            policies[friction] = GaussianPolicy(simplified=simplified, alpha=0, from_filename=f'{path}/policy.pt')

        def build_state_info_tensor(initial_price, current_time):
            current_time = torch.tensor(current_time)
            initial_price = torch.tensor(initial_price)
            state_info_tensor = torch.empty((len(current_time), 9))
            state_info_tensor[:, 0] = torch.tensor(
                util.standard_to_normalized_price(initial_price.numpy(), mu, sigma, current_time.numpy(),
                                                  _dt))  # normal_price
            state_info_tensor[:, 1] = current_time * _dt  # passed_real_time
            state_info_tensor[:, 2] = (max_step - current_time) * _dt  # remaining_real_time
            state_info_tensor[:, 3] = torch.tensor(
                util.standard_to_normalized_price(strike_price, mu, sigma, max_step, _dt))  # normal_strike_price
            state_info_tensor[:, 4] = r  # r
            state_info_tensor[:, 5] = mu  # mu
            state_info_tensor[:, 6] = sigma  # sigma
            state_info_tensor[:, 7] = friction  # friction
            state_info_tensor[:, 8] = friction  # friction
            return state_info_tensor

        # Plot and compare prices learnt
        initial_prices = np.linspace(0.8, 1.2, 31)
        tensor = build_state_info_tensor(initial_price=initial_prices, current_time=np.zeros(len(initial_prices)))
        bs_baseline = BSBaseline(is_call=is_call_option)
        bs_policy = BSPolicy(is_call=is_call_option)
        bs_price = bs_baseline.batch_estimate(tensor.numpy())
        bs_hedge = bs_policy.batch_action(tensor.numpy(), random=False)
        nn_hedges, nn_prices = dict(), dict()

        for friction in frictions:
            nn_prices[friction] = baselines[friction].batch_estimate(tensor)
            nn_hedges[friction] = policies[friction].batch_action(tensor, random=False)

        fig1, (ax_price, ax_hedge) = plt.subplots(1, 2, figsize=(9, 4))
        palette = sns.color_palette()

        ax_price.plot(initial_prices, bs_price, label='BS', c=palette[-1], ls=':')
        ax_hedge.plot(initial_prices, bs_hedge, label='BS', c=palette[-1], ls=':')
        for i, friction in enumerate(frictions):
            ax_price.plot(initial_prices, -nn_prices[friction], c=palette[i],
                          label=rf'NN policy ($\epsilon$={friction:.1f})')
            ax_hedge.plot(initial_prices, nn_hedges[friction], c=palette[i],
                          label=rf'NN policy ($\epsilon$={friction:.1f})')

        ax_hedge.legend(loc='best', title='method')
        # ax_price.legend(loc='best', title='method')
        ax_hedge.set(xlabel=r'initial price $S_0$', ylabel='hedge position')
        ax_price.set(xlabel=r'initial price $S_0$', ylabel='option price(negative return)')
        fig1.savefig(f'plot/{exp_name}/price-hedge.png')

        plt.show()
