import os
import pickle
from unittest import TestCase
import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, Series

import util
from qlbs.bs import BSPolicy, BSBaseline
from qlbs.env import QLBSEnv
from qlbs.rl import GaussianPolicy, NNBaseline, policy_gradient

mpl.use('TkAgg')
sns.set_style('whitegrid')
exp_name = os.path.basename(__file__)[:-3]


class Experiment1(TestCase):
    def _parameters(self):
        r = 1e-2
        mu = 0e-3
        sigma = 1e-1
        risk_lambdas = [0, 1, 2, 3]
        initial_price = 1
        strike_price = 1
        T = 5
        _dt = 1
        friction = 0e-1
        is_call_option = True
        simplified = True
        max_step = int(np.round(T / _dt))
        mutation_lambdas = [0, 5e-3]

        return r, mu, sigma, risk_lambdas, initial_price, strike_price, T, _dt, friction, is_call_option, simplified, \
               max_step, mutation_lambdas

    def _path(self, simplified, risk_lambda, mutation_lambda):
        plan = f'risk_lambda={risk_lambda:d}, mutation_lambda={mutation_lambda:.2e}'
        return f'trained_model/{exp_name}/{util._prefix(simplified, plan)}'

    def test_no_mutation(self):
        r, mu, sigma, risk_lambdas, initial_price, strike_price, T, _dt, friction, is_call_option, simplified, \
        max_step, mutation_lambdas = self._parameters()

        for mutation_lambda in [0]: #mutation_lambdas:
            for risk_lambda in [2]: #risk_lambdas:
                mutate_status = []

                def mutate(env: QLBSEnv):
                    if np.random.rand(1)[0] < mutation_lambda:
                        env.initial_asset_price = np.random.rand(1)[0] * 0.4 + 0.8
                        mutated = True
                    else:
                        mutated = False
                    mutate_status.append((len(mutate_status), mutated))

                env = QLBSEnv(is_call_option=is_call_option, strike_price=strike_price, max_step=max_step, mu=mu,
                              sigma=sigma, r=r, risk_lambda=risk_lambda, friction=friction,
                              initial_asset_price=initial_price, risk_simulation_paths=200, _dt=_dt, mutation=mutate)
                nn_policy = GaussianPolicy(simplified=simplified, alpha=1e-4)
                nn_baseline = NNBaseline(simplified=simplified, alpha=1e-4)

                collector = policy_gradient(env, nn_policy, nn_baseline, episode_n=4000)
                path = self._path(simplified, risk_lambda, mutation_lambda)
                nn_policy.save(f'{path}/policy.pt')
                nn_baseline.save(f'{path}/baseline.pt')
                with open(f'{path}/additional.pickle', 'wb') as f:
                    pickle.dump({'mutate_status': mutate_status, 'collector': collector}, f)

    def test_draw_plot(self):
        r, mu, sigma, risk_lambdas, initial_price, strike_price, T, _dt, friction, is_call_option, simplified, \
        max_step, mutation_lambdas = self._parameters()

        mutated_summary = dict()
        fixed_summary = dict()

        for mutation_lambda in mutation_lambdas:
            summary = fixed_summary if np.isclose(mutation_lambda, 0) else mutated_summary
            for risk_lambda in risk_lambdas:
                path = self._path(simplified, risk_lambda, mutation_lambda)

                with open(f'{path}/additional.pickle', 'rb') as f:
                    additional = pickle.load(f)
                    collector: util.EMACollector = additional['collector']
                    mutate_status = additional['mutate_status']

                t_return_ma = Series(collector.ema_dict['t_return'])
                t_return_sq = Series(collector.emsq_dict['t_return'])
                t_return_std = np.sqrt(t_return_sq - t_return_ma ** 2)

                t_base_return_ma = Series(collector.ema_dict['t_base_return'])
                t_base_return_sq = Series(collector.emsq_dict['t_base_return'])
                t_base_return_std = np.sqrt(t_base_return_sq - t_base_return_ma ** 2)

                summary[risk_lambda] = DataFrame({
                    't_return_ma': t_return_ma,
                    't_return_std': t_return_std,
                    't_base_return_ma': t_base_return_ma,
                    't_base_return_std': t_base_return_std,
                    'mutation': (m for (t, m) in mutate_status),
                })

        # Start to plot the fixed part
        fig1 = plt.figure(figsize=(7, 5))
        for risk_lambda in risk_lambdas:
            df: DataFrame = fixed_summary[risk_lambda]
            plt.plot(df.index, df['t_return_ma'], label=rf'return ($\lambda$={risk_lambda:d})')
            plt.fill_between(df.index, df['t_return_ma'] - df['t_return_std'] * 2,
                             df['t_return_ma'] + df['t_return_std'] * 2, alpha=0.1)
        for risk_lambda in risk_lambdas:
            if np.isclose(risk_lambda, 0):
                continue
            df: DataFrame = fixed_summary[risk_lambda]
            plt.plot(df.index, df['t_base_return_ma'], label=rf'cashflow ($\lambda$={risk_lambda:d})')
            plt.fill_between(df.index, df['t_base_return_ma'] - df['t_base_return_std'] * 2,
                             df['t_base_return_ma'] + df['t_base_return_std'] * 2, alpha=0.1)
        plt.legend(loc='best')
        plt.show()
