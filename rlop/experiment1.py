import os
import pickle
from unittest import TestCase

import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame, Series

import util
from rlop.bs import BSInitialEstimator, BSPolicy
from rlop.env import RLOPEnv
from rlop.rl import GaussianPolicy, policy_gradient_for_stacked

mpl.use('TkAgg')
sns.set_style('whitegrid')
exp_name = os.path.basename(__file__)[:-3]


class Experiment1(TestCase):
    def _parameters(self):
        r = 1e-2
        mu = 0e-3
        sigma = 1e-1
        initial_price = 1
        strike_price = 1
        T = 5
        _dt = 1
        friction = 0e-1
        is_call_option = True
        simplified = True
        max_step = int(np.round(T / _dt))
        mutation_lambdas = [0, 1e-2]

        return r, mu, sigma, initial_price, strike_price, T, _dt, friction, is_call_option, simplified, \
               max_step, mutation_lambdas

    def _path(self, simplified, mutation_lambda):
        plan = f'mutation_lambda={mutation_lambda:.2e}'
        return f'trained_model/{exp_name}/{util._prefix(simplified, plan)}'

    def test_mutation_or_not(self):
        r, mu, sigma, initial_price, strike_price, T, _dt, friction, is_call_option, simplified, \
        max_step, mutation_lambdas = self._parameters()
        reuse_training = True

        for mutation_lambda in mutation_lambdas:
            mutate_status = []

            def mutate(env: RLOPEnv):
                if np.random.rand(1)[0] < mutation_lambda:
                    env.initial_asset_price = np.random.rand(1)[0] * 0.4 + 0.8
                    mutated = True
                else:
                    mutated = False
                mutate_status.append((len(mutate_status), mutated))

            path = self._path(simplified, mutation_lambda)
            env = RLOPEnv(is_call_option=is_call_option, strike_price=strike_price, max_step=max_step, mu=mu,
                          sigma=sigma, r=r, friction=friction, initial_estimator=BSInitialEstimator(is_call_option),
                          initial_asset_price=initial_price, _dt=_dt, mutation=mutate)
            nn_policy = GaussianPolicy(simplified=simplified, alpha=1e-4,
                                       from_filename=f'{path}/policy.pt' if reuse_training else None)

            collector = policy_gradient_for_stacked(env, nn_policy, batch=True, plot=True, episode_n=10000)
            nn_policy.save(f'{path}/policy.pt')
            with open(f'{path}/additional.pickle', 'wb') as f:
                pickle.dump({'mutate_status': mutate_status, 'collector': collector}, f)

    def test_draw_plot(self):
        r, mu, sigma, initial_price, strike_price, T, _dt, friction, is_call_option, simplified, \
        max_step, mutation_lambdas = self._parameters()

        mutated_summary = dict()

        for mutation_lambda in mutation_lambdas:
            path = self._path(simplified, mutation_lambda)

            with open(f'{path}/additional.pickle', 'rb') as f:
                additional = pickle.load(f)
                collector: util.EMACollector = additional['collector']
                mutate_status = additional['mutate_status']

            t_return_ma = Series(collector.ema_dict['t_return'])
            t_return_sq = Series(collector.emsq_dict['t_return'])
            t_return_std = np.sqrt(t_return_sq - t_return_ma ** 2)

            mutated_summary[mutation_lambda] = DataFrame({
                't_return_ma': t_return_ma,
                't_return_std': t_return_std,
                'mutation': (m for (t, m) in mutate_status),
            })

        df_0 = mutated_summary[0]
        df_m = mutated_summary[1e-2]

        # Start to plot the fixed part
        fig1, (ax_0, ax_m) = plt.subplots(2, 1, figsize=(7, 6))
        ax_0.plot(df_0.index, df_0['t_return_ma'])
        ax_0.fill_between(df_0.index, df_0['t_return_ma'] - df_0['t_return_std'] * 2,
                          df_0['t_return_ma'] + df_0['t_return_std'] * 2, alpha=0.25)
        ax_0.set(ylabel='return (no adjustment)', ylim=[-0.7, 0.1])
        ax_m.plot(df_m.index, df_m['t_return_ma'])
        ax_m.fill_between(df_m.index, df_m['t_return_ma'] - df_m['t_return_std'] * 2,
                          df_m['t_return_ma'] + df_m['t_return_std'] * 2, alpha=0.25)
        for index in df_m.index[df_m['mutation'] == 1]:
            plt.plot([index], [df_m['t_return_ma'][index] - 5e-2], 'm^', alpha=0.5)
        ax_m.set(xlabel='iteration', ylabel='return (initial price adjustment)', ylim=[-0.7, 0.1])
        fig1.savefig(f'plot/{exp_name}/learning-curve.png')

        plt.show()

    def test_compared_with_bs(self):
        r, mu, sigma, initial_price, strike_price, T, _dt, friction, is_call_option, simplified, \
        max_step, mutation_lambdas = self._parameters()

        path = self._path(simplified, 1e-2)
        nn_policy = GaussianPolicy(simplified=simplified, alpha=0, from_filename=f'{path}/policy.pt')
        initial_estimator = BSInitialEstimator(is_call_option)

        def build_state_info_tensor(price, current_time):
            current_time = torch.tensor(current_time)
            price = torch.tensor(price)
            state_info_tensor = torch.empty((len(current_time), 9))
            S = util.standard_to_normalized_price(price.numpy(), mu, sigma, current_time.numpy(),
                                                  _dt)  # normal_price
            state_info_tensor[:, 0] = torch.tensor(S)
            state_info_tensor[:, 1] = (max_step - current_time) * _dt  # remaining_real_time
            state_info_tensor[:, 2] = torch.tensor(
                initial_estimator(price.numpy(), strike_price, max_step - current_time.numpy(), r, sigma,
                                  _dt))  # portfolio value
            state_info_tensor[:, 3] = torch.tensor(
                util.standard_to_normalized_price(strike_price, mu, sigma, max_step, _dt))  # normal_strike_price
            state_info_tensor[:, 4] = r  # r
            state_info_tensor[:, 5] = mu  # mu
            state_info_tensor[:, 6] = sigma  # sigma
            state_info_tensor[:, 7] = friction  # friction
            return state_info_tensor

        # plot and compare prices learnt
        initial_prices = np.linspace(0.8, 1.2, 31)
        times = np.arange(max_step) * _dt

        bs_hedges = dict()
        nn_hedges = dict()
        bs_policy = BSPolicy(is_call=is_call_option)

        for t in times:
            tensor = build_state_info_tensor(price=initial_prices,
                                             current_time=np.zeros(len(initial_prices)) + t)
            bs_hedges[t] = bs_policy.batch_action(tensor.numpy(), random=False, passed_real_time=t)
            nn_hedges[t] = nn_policy.batch_action(tensor, random=False).numpy()

        fig1, (ax_nn, ax_bs) = plt.subplots(2, 1, figsize=(7, 6))
        palette = sns.color_palette()
        for i, t in enumerate(times):
            ax_bs.plot(initial_prices, bs_hedges[t], label=f'{T - t:d}', ls='-', c=palette[i])
            ax_nn.plot(initial_prices, nn_hedges[t], label=f'{T - t:d}', ls='-', c=palette[i])
        ax_nn.legend(loc='best', title='remaining time')
        ax_bs.set(xlabel='asset price', ylabel='BS hedge position', ylim=[0, 1])
        ax_nn.set(xlabel='asset price', ylabel='NN hedge position', ylim=[0, 1])
        fig1.savefig(f'plot/{exp_name}/hedge_position.png')
        plt.show()
