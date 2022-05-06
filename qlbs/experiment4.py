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
from qlbs.bs import BSPolicy, BSBaseline
from qlbs.env import QLBSEnv
from qlbs.rl import GaussianPolicy, NNBaseline, policy_gradient

mpl.use('TkAgg')
sns.set_style('whitegrid')
exp_name = os.path.basename(__file__)[:-3]


class Experiment4(TestCase):
    def _parameters(self):
        r = 1e-2, 2e-2
        mu = 0e-3, 1e-2
        sigma = 1e-1, 2e-2
        risk_lambda = 0.5, 1.5
        initial_price = 1
        strike_price = 1
        T = 5
        _dt = 1
        friction = 0, 0.1
        is_call_option = True
        simplified = False
        max_step = int(np.round(T / _dt))
        mutation_lambda = 1e-2

        params = [(r[i], mu[i], sigma[i], risk_lambda[i], initial_price, strike_price, T, _dt, friction[i],
                   is_call_option, simplified, max_step, mutation_lambda) for i in range(2)]
        avg_param = tuple(
            (params[0][i] + params[1][i] if i in (0, 1, 2, 3, 8) else params[0][i] for i in range(len(params[0]))))

        return params[0], params[1], avg_param

    def _path(self, simplified, plan):
        return f'trained_model/{exp_name}/{util._prefix(simplified, plan)}'

    def test_mixed(self):
        low_param, high_param, avg_param = self._parameters()
        r, mu, sigma, risk_lambda, initial_price, strike_price, T, _dt, friction, is_call_option, simplified, \
        max_step, mutation_lambda = low_param
        reuse_training = True
        plan = 'mixed'

        def mutate(env: QLBSEnv):
            use_low = np.isclose(env.info.r, low_param[0])
            if np.random.rand(1)[0] < mutation_lambda:
                param = high_param if use_low else low_param
            else:
                param = low_param if use_low else high_param

            env.info.r = param[0]
            env.info.mu = param[1]
            env.info.sigma = param[2]
            env.info.risk_lambda = param[3]
            env.info.friction = param[8]

        load_path = self._path(simplified, plan)
        env = QLBSEnv(is_call_option=is_call_option, strike_price=strike_price, max_step=max_step, mu=mu,
                      sigma=sigma, r=r, risk_lambda=risk_lambda, friction=friction,
                      initial_asset_price=initial_price, risk_simulation_paths=200, _dt=_dt, mutation=mutate)
        nn_policy = GaussianPolicy(simplified=simplified, alpha=1e-4,
                                   from_filename=f'{load_path}/policy.pt' if reuse_training else None)
        nn_baseline = NNBaseline(simplified=simplified, alpha=1e-4,
                                 from_filename=f'{load_path}/baseline.pt' if reuse_training else None)

        save_path = self._path(simplified, plan)
        collector = policy_gradient(env, nn_policy, nn_baseline, episode_n=4000, plot=True)
        nn_policy.save(f'{save_path}/policy.pt')
        nn_baseline.save(f'{save_path}/baseline.pt')
        with open(f'{save_path}/additional.pickle', 'wb') as f:
            pickle.dump({'collector': collector}, f)

    def test_focused(self):
        low_param, high_param, avg_param = self._parameters()
        r, mu, sigma, risk_lambda, initial_price, strike_price, T, _dt, friction, is_call_option, simplified, \
        max_step, mutation_lambda = avg_param
        reuse_training = True
        plan = 'focused'

        def mutate(env: QLBSEnv):
            use_low = np.isclose(env.info.r, low_param[0])
            use_high = np.isclose(env.info.r, high_param[0])
            if np.random.rand(1)[0] < mutation_lambda:
                r = np.random.rand(1)[0]
                if r < 1/3:
                    param = low_param
                elif r < 2/3:
                    param = high_param
                else:
                    param = avg_param
            else:
                param = low_param if use_low else (high_param if use_high else avg_param)

            env.info.r = param[0]
            env.info.mu = param[1]
            env.info.sigma = param[2]
            env.info.risk_lambda = param[3]
            env.info.friction = param[8]

        load_path = self._path(simplified, 'mixed')
        env = QLBSEnv(is_call_option=is_call_option, strike_price=strike_price, max_step=max_step, mu=mu,
                      sigma=sigma, r=r, risk_lambda=risk_lambda, friction=friction,
                      initial_asset_price=initial_price, risk_simulation_paths=200, _dt=_dt, mutation=mutate)
        nn_policy = GaussianPolicy(simplified=simplified, alpha=1e-4,
                                   from_filename=f'{load_path}/policy.pt' if reuse_training else None)
        nn_baseline = NNBaseline(simplified=simplified, alpha=1e-4,
                                 from_filename=f'{load_path}/baseline.pt' if reuse_training else None)

        save_path = self._path(simplified, plan)
        collector = policy_gradient(env, nn_policy, nn_baseline, episode_n=2000, plot=True)
        nn_policy.save(f'{save_path}/policy.pt')
        nn_baseline.save(f'{save_path}/baseline.pt')
        with open(f'{save_path}/additional.pickle', 'wb') as f:
            pickle.dump({'collector': collector}, f)

    def test_draw_plot(self):
        low_param, high_param, avg_param = self._parameters()
        r, mu, sigma, risk_lambda, initial_price, strike_price, T, _dt, friction, is_call_option, simplified, \
        max_step, mutation_lambda = low_param

        mixed_path = self._path(simplified, 'mixed')
        mixed_baseline = NNBaseline(simplified=simplified, alpha=0, from_filename=f'{mixed_path}/baseline.pt')
        mixed_policy = GaussianPolicy(simplified=simplified, alpha=0, from_filename=f'{mixed_path}/policy.pt')

        focused_path = self._path(simplified, 'focused')
        focused_baseline = NNBaseline(simplified=simplified, alpha=0, from_filename=f'{focused_path}/baseline.pt')
        focused_policy = GaussianPolicy(simplified=simplified, alpha=0, from_filename=f'{focused_path}/policy.pt')

        def build_state_info_tensor(initial_price, current_time, param):
            current_time = torch.tensor(current_time)
            initial_price = torch.tensor(initial_price)
            state_info_tensor = torch.empty((len(current_time), 9))
            state_info_tensor[:, 0] = torch.tensor(
                util.standard_to_normalized_price(initial_price.numpy(), param[1], param[2], current_time.numpy(),
                                                  param[7]))  # normal_price
            state_info_tensor[:, 1] = current_time * param[7]  # passed_real_time
            state_info_tensor[:, 2] = (param[11] - current_time) * param[7]  # remaining_real_time
            state_info_tensor[:, 3] = torch.tensor(
                util.standard_to_normalized_price(param[5], param[1], param[2], param[11],
                                                  param[7]))  # normal_strike_price
            state_info_tensor[:, 4] = param[0]  # r
            state_info_tensor[:, 5] = param[1]  # mu
            state_info_tensor[:, 6] = param[2]  # sigma
            state_info_tensor[:, 7] = param[3]  # risk_lambda
            state_info_tensor[:, 8] = param[8]  # friction
            return state_info_tensor

        # Plot and compare prices learnt
        initial_prices = np.linspace(0.8, 1.2, 31)
        low_tensor = build_state_info_tensor(initial_price=initial_prices, current_time=np.zeros(len(initial_prices)),
                                             param=low_param)
        high_tensor = build_state_info_tensor(initial_price=initial_prices, current_time=np.zeros(len(initial_prices)),
                                              param=high_param)
        avg_tensor = build_state_info_tensor(initial_price=initial_prices, current_time=np.zeros(len(initial_prices)),
                                             param=avg_param)
        # bs_baseline = BSBaseline(is_call=is_call_option)
        # bs_policy = BSPolicy(is_call=is_call_option)
        # bs_price = bs_baseline.batch_estimate(tensor.numpy())
        # bs_hedge = bs_policy.batch_action(tensor.numpy(), random=False)

        low_price = mixed_baseline.batch_estimate(low_tensor)
        low_hedge = mixed_policy.batch_action(low_tensor, random=False)
        high_price = mixed_baseline.batch_estimate(high_tensor)
        high_hedge = mixed_policy.batch_action(high_tensor, random=False)
        avg_price = mixed_baseline.batch_estimate(avg_tensor)
        avg_hedge = mixed_policy.batch_action(avg_tensor, random=False)

        avg_focused_price = focused_baseline.batch_estimate(avg_tensor)
        avg_focused_hedge = focused_policy.batch_action(avg_tensor, random=False)

        fig1, (ax_price, ax_hedge) = plt.subplots(1, 2, figsize=(9, 4))
        palette = sns.color_palette("Paired")

        ax_price.plot(initial_prices, -low_price, label='condition 1', c=palette[1], alpha=0.25)
        ax_hedge.plot(initial_prices, low_hedge, label='condition 1', c=palette[1], alpha=0.25)
        ax_price.plot(initial_prices, -high_price, label='condition 2', c=palette[5], alpha=0.25)
        ax_hedge.plot(initial_prices, high_hedge, label='condition 2', c=palette[5], alpha=0.25)
        ax_price.plot(initial_prices, -avg_price, label='avg (no fine tuning)', c=palette[3], alpha=1)
        ax_hedge.plot(initial_prices, avg_hedge, label='avg (no fine tuning)', c=palette[3], alpha=1)
        ax_price.plot(initial_prices, -avg_focused_price, label='avg (fine tuning)', c=palette[3], alpha=1, ls='--')
        ax_hedge.plot(initial_prices, avg_focused_hedge, label='avg (fine tuning)', c=palette[3], alpha=1, ls='--')

        ax_hedge.legend(loc='best', title='method')
        ax_hedge.set(xlabel=r'initial price $S_0$', ylabel='hedge position')
        ax_price.set(xlabel=r'initial price $S_0$', ylabel='option price(negative return)')
        fig1.savefig(f'plot/{exp_name}/price-hedge.png')

        plt.show()
