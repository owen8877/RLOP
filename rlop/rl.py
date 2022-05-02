from itertools import chain
from typing import Tuple
from unittest import TestCase

import numpy as np
import seaborn as sns
import torch
from gym import Env
from matplotlib import pyplot as plt
from torch.optim.adam import Adam
from tqdm import trange

import util
from rlop.env import RLOPEnv
from rlop.interface import Policy, InitialEstimator
from util.net import ResNet
from util.pricing import bs_euro_vanilla_call, bs_euro_vanilla_put, delta_hedge_bs_euro_vanilla_call, \
    delta_hedge_bs_euro_vanilla_put


class GaussianPolicy(Policy):
    def __init__(self, alpha):
        super().__init__()
        self.theta_mu = ResNet(7, 10, groups=2, layer_per_group=2)
        self.theta_sigma = ResNet(7, 10, groups=2, layer_per_group=2)

        self.optimizer = Adam(chain(self.theta_mu.parameters(), self.theta_sigma.parameters()), lr=alpha)

    def _gauss_param(self, tensor):
        mu = self.theta_mu(tensor)
        sigma = self.theta_sigma(tensor)

        mu_c = torch.sigmoid(mu)
        sigma_s = torch.sigmoid(sigma)

        sigma_c = torch.clip(sigma_s, 1e-1, 1)

        return mu_c, sigma_c

    def action(self, state, info, state_info_tensors=None):
        rt = state.remaining_step
        positions = state.portfolio_value * 0
        tensors = state.to_tensors(info) if state_info_tensors is None else state_info_tensors
        with torch.no_grad():
            for t in range(rt):
                mu, sigma = self._gauss_param(tensors[t])
                positions[t] = np.random.randn(1) * float(sigma) + float(mu)
        return positions

    def update(self, delta, action, state, info, state_info_tensors=None, update_on: int = None):
        tensors = state.to_tensors(info) if state_info_tensors is None else state_info_tensors

        def loss(delta, a, tensor):
            mu, sigma = self._gauss_param(tensor)
            log_pi = - (a - mu) ** 2 / (2 * sigma ** 2) - torch.log(sigma)
            loss = - delta * log_pi
            loss.backward()
            return loss

        self.optimizer.zero_grad()
        self.optimizer.step(lambda: loss(delta, action[update_on], tensors[update_on]))

    class BatchUpdater:
        def __init__(self, policy):
            self.deltas = []
            self.actions = []
            self.tensors = []
            self.policy = policy

        def collect(self, delta, action, state_info_tensors, update_on):
            self.deltas.append(delta)
            self.actions.append(action[update_on])
            self.tensors.append(state_info_tensors[update_on])

        def update(self):
            tensors = torch.stack(self.tensors)
            actions = torch.tensor(self.actions)
            deltas = torch.tensor(self.deltas)

            def loss():
                mu, sigma = self.policy._gauss_param(tensors)
                log_pi = - (actions - mu[:, 0]) ** 2 / (2 * sigma[:, 0] ** 2) - torch.log(sigma[:, 0])
                loss = - (deltas * log_pi).sum()
                loss.backward()
                return loss

            self.policy.optimizer.zero_grad()
            self.policy.optimizer.step(loss)

    def batch_update(self):
        return self.BatchUpdater(self)

    def _save_path(self, filename: str):
        return f'data/{filename}.pt'

    def save(self, filename: str):
        util.ensure_dir(self._save_path(''))
        torch.save({
            'mu_net': self.theta_mu.state_dict(),
            'sigma_net': self.theta_sigma.state_dict(),
        }, self._save_path(filename))

    def load(self, filename: str):
        state_dict = torch.load(self._save_path(filename))
        self.theta_mu.load_state_dict(state_dict['mu_net'])
        self.theta_mu.eval()
        self.theta_sigma.load_state_dict(state_dict['sigma_net'])
        self.theta_sigma.eval()


def policy_gradient_for_stacked(env: Env, pi: GaussianPolicy, episode_n: int, *, ax: plt.Axes = None,
                                axs_env: Tuple[plt.Axes] = None, last_day_train_only: bool = False,
                                batch: bool = False):
    avg_rewards = []
    if ax is None:
        fig, ax = plt.subplots()
    pbar = trange(episode_n)
    for e in pbar:
        states = []
        actions = []
        rewards = []

        (state, info), done = env.reset(), False
        states.append(state)
        while not done:
            action = pi.action(state, info)
            state, reward, done, _ = env.step(action)
            if e == episode_n - 1 and axs_env is not None:
                env.render(axs=axs_env)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        T = len(actions)

        if batch:
            updater = pi.batch_update()
        for t in range(T):
            state_info_tensors = states[t].to_tensors(info)
            start_day = T - 1 if last_day_train_only else t
            for i in range(start_day, T):
                delta = rewards[i]
                if batch:
                    updater.collect(delta, actions[t], state_info_tensors, i - t)
                else:
                    pi.update(delta, actions[t], states[t], info, state_info_tensors=state_info_tensors,
                              update_on=i - t)
        if batch:
            updater.update()

        avg_rewards.append(np.average(rewards))
        pbar.set_description(
            f'avg_r={avg_rewards[-1]:.2e};r={info.r:.4f};mu={info.mu:.4f};sigma={info.sigma:.4f};K={info.strike_price:.4f}')
        if (e + 1) % 100 == 0 and ax is not None:
            indices = np.arange(0, e + 1)
            ax.cla()
            ax.plot(indices, np.array(avg_rewards) * (-1))
            ax.plot(indices, np.cumsum(avg_rewards) / (1 + indices) * (-1), ls='--')
            ax.set(yscale='log', ylabel='negative reward')
            plt.show(block=False)
            plt.pause(0.001)
    return avg_rewards


def policy_evaluation(env: Env, pi: Policy, episode_n: int, *, plot: bool = False):
    avg_rewards = []
    fig, ax = plt.subplots()
    pbar = trange(episode_n)
    for e in pbar:
        states = []
        actions = []
        rewards = []

        (state, info), done = env.reset(), False
        states.append(state)
        while not done:
            action = pi.action(state, info)
            state, reward, done, _ = env.step(action)
            if e == episode_n - 1 and plot:
                env.render()

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        avg_rewards.append(np.average(rewards))
        pbar.set_description(
            f'avg_r={avg_rewards[-1]:.2e};r={info.r:.4f};mu={info.mu:.4f};sigma={info.sigma:.4f};K={info.strike_price:.4f}')
        if (e + 1) % 100 == 0 and plot:
            indices = np.arange(0, e + 1)
            ax.cla()
            ax.plot(indices, np.array(avg_rewards) * (-1))
            ax.plot(indices, np.cumsum(avg_rewards) / (1 + indices) * (-1), ls='--')
            ax.set(yscale='log', ylabel='negative reward')
            plt.show(block=False)
            plt.pause(0.001)
    return avg_rewards


class BSPolicy(Policy):
    def __init__(self, is_call: bool = True):
        super().__init__()
        self.is_call = is_call

    def action(self, state, info):
        S = util.normalized_to_standard_price(state.normalized_asset_price, info.mu, info.sigma,
                                              state.remaining_step, info._dt)
        K = info.strike_price
        return (delta_hedge_bs_euro_vanilla_call if self.is_call else delta_hedge_bs_euro_vanilla_put)(
            S, K, np.arange(state.remaining_step) + 1, info.r, info.sigma, info._dt)

    def update(self, delta: np.sctypes, action, state, info, *args):
        raise Exception('BS policy cannot be updated!')


class BSInitialEstimator(InitialEstimator):
    def __call__(self, initial_asset_price: float, strike_price: float, remaining_time: int, r: float,
                 sigma: float, _dt: float) -> float:
        return (bs_euro_vanilla_call if self.is_call_option else bs_euro_vanilla_put)(
            initial_asset_price, strike_price, remaining_time, r, sigma, _dt
        )


class Test(TestCase):
    def test_policy_gradient(self):
        import matplotlib as mpl
        mpl.use('TkAgg')
        sns.set_style('whitegrid')

        is_call_option = True
        initial_asset_price = 1
        max_time = 3
        itr = 10000

        strike_price = 1.000
        mu = 1.0e-3
        sigma = 1e-2
        r = 1.0e-3

        env = RLOPEnv(is_call_option=is_call_option, strike_price=strike_price, max_time=max_time, mu=mu, sigma=sigma,
                      r=r, initial_estimator=BSInitialEstimator(is_call_option),
                      initial_asset_price=initial_asset_price)

        pi = GaussianPolicy(5e-3)
        avg_rewards = policy_gradient_for_stacked(env, pi, itr, axs_env=True, batch=True, last_day_train_only=False)
        print(f'mean: {np.mean(avg_rewards):.2e}, std: {np.std(avg_rewards):.2e}')

    def test_cumulative_training(self):
        import matplotlib as mpl
        from util.timer import Timer
        mpl.use('TkAgg')
        sns.set_style('whitegrid')

        is_call_option = True
        initial_asset_price = 1
        max_time = 3
        itr = 10000

        fig, ax = plt.subplots()
        _, axs_env = plt.subplots(2, 1)
        with Timer(desc=f'cumulative training for {max_time} levels'):
            pi = GaussianPolicy(1e-3)
            for sub_max_time in range(1, max_time + 1):
                strike_price = 1.000
                mu = 1.0e-3
                sigma = 1e-2
                r = 1.0e-3

                env = RLOPEnv(is_call_option=is_call_option, strike_price=strike_price, max_time=sub_max_time, mu=mu,
                              sigma=sigma, r=r, initial_estimator=BSInitialEstimator(is_call_option),
                              initial_asset_price=initial_asset_price)

                policy_gradient_for_stacked(env, pi, itr, ax=ax, axs_env=axs_env, last_day_train_only=True)
        plt.show(block=True)

    def test_BS_error(self):
        import matplotlib as mpl
        from util.timer import Timer
        mpl.use('TkAgg')
        sns.set_style('whitegrid')

        is_call_option = True
        initial_asset_price = 1

        itr = 200

        strike_price = 1.000
        mu = 1.0e-3
        sigma = 1e-2
        r = 1.0e-3
        _dt = 1
        T = 10

        max_time = int(np.round(T / _dt))

        env = RLOPEnv(is_call_option=is_call_option, strike_price=strike_price, max_time=max_time, mu=mu, sigma=sigma,
                      r=r, initial_estimator=BSInitialEstimator(is_call_option),
                      initial_asset_price=initial_asset_price, mutation=0, _dt=_dt)
        pi = BSPolicy(is_call=is_call_option)
        with Timer(desc=f'BS baseline for {itr} iterations'):
            avg_rewards = policy_evaluation(env, pi, itr, axs_env=False)
        print(f'mean: {np.mean(avg_rewards):.2e}, std: {np.std(avg_rewards):.2e}')
        plt.show(block=True)
