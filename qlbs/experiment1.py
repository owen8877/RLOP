import os
import pickle
from unittest import TestCase
import matplotlib as mpl
import numpy as np
import seaborn as sns

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

        for mutation_lambda in mutation_lambdas:
            for risk_lambda in risk_lambdas:
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

                collector = policy_gradient(env, nn_policy, nn_baseline, episode_n=400)
                path = self._path(simplified, risk_lambda, mutation_lambda)
                nn_policy.save(f'{path}/policy.pt')
                nn_baseline.save(f'{path}/baseline.pt')
                with open(f'{path}/additional.pickle', 'wb') as f:
                    pickle.dump({'mutate_status': mutate_status, 'collector': collector}, f)

    def test_draw_plot(self):
        r, mu, sigma, risk_lambdas, initial_price, strike_price, T, _dt, friction, is_call_option, simplified, \
        max_step, mutation_lambdas = self._parameters()

        for mutation_lambda in mutation_lambdas:
            for risk_lambda in risk_lambdas:
                path = self._path(simplified, risk_lambda, mutation_lambda)

                with open(f'{path}/additional.pickle', 'rb') as f:
                    additional = pickle.load(f)
                    collector = additional['collector']
                    mutate_status = additional['mutate_status']
                nn_policy = GaussianPolicy(simplified=simplified, alpha=1e-4, from_filename=f'{path}/policy.pt')
                nn_baseline = NNBaseline(simplified=simplified, alpha=1e-4, from_filename=f'{path}/baseline.pt')
                bs_pi = BSPolicy(is_call=is_call_option)
                bs_baseline = BSBaseline(is_call=is_call_option)
