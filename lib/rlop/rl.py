from itertools import chain
from typing import Tuple
from unittest import TestCase

import numpy as np
import seaborn as sns
import torch
from gymnasium import Env
from matplotlib import pyplot as plt
from torch.optim.adam import Adam
from tqdm import trange

import lib.util
from .bs import BSInitialEstimator, BSPolicy
from .env import RLOPEnv
from .interface import Policy
from lib.util.net import ResNet


class GaussianPolicy(Policy):
    def __init__(self, simplified: bool, alpha=1e-2, from_filename: str = None):
        super().__init__()
        self.simplified = simplified
        self.theta_mu = ResNet(4 if simplified else 8, 10, groups=2, layer_per_group=2)
        self.theta_sigma = ResNet(4 if simplified else 8, 10, groups=2, layer_per_group=2)

        self.optimizer = Adam(chain(self.theta_mu.parameters(), self.theta_sigma.parameters()), lr=alpha)
        if from_filename is not None:
            self.load(from_filename)

    def _gauss_param(self, tensor):
        if self.simplified:
            if len(tensor.size()) == 1:
                tensor = tensor[None, :4].float()
            else:
                tensor = tensor[:, :4].float()
        else:
            tensor = tensor.float()
        mu = self.theta_mu(tensor)
        sigma = self.theta_sigma(tensor)

        mu_c = torch.sigmoid(mu)
        sigma_s = torch.sigmoid(sigma)

        sigma_c = torch.clip(sigma_s, 2e-2, 1)

        return mu_c, sigma_c

    def action(self, state, info, state_info_tensors=None, random: bool = True):
        rt = state.remaining_step
        positions = state.portfolio_value * 0
        tensors = state.to_tensors(info) if state_info_tensors is None else state_info_tensors
        with torch.no_grad():
            for t in range(rt):
                mu, sigma = self._gauss_param(tensors[t])
                if random:
                    positions[t] = np.random.randn(1) * float(sigma) + float(mu)
                else:
                    positions[t] = float(mu)
        return positions

    def batch_action(self, tensors, random: bool = True):
        with torch.no_grad():
            mu, sigma = self._gauss_param(tensors)
            if random:
                return torch.randn(len(mu)) * sigma[:, 0] + mu[:, 0]
            else:
                return mu[:, 0]

    def update(self, delta, action, state, info, state_info_tensors=None, update_on: int = None):
        tensors = state.to_tensors(info) if state_info_tensors is None else state_info_tensors

        def loss(delta, a, tensor):
            mu, sigma = self._gauss_param(tensor)
            log_pi = -((a - mu) ** 2) / (2 * sigma**2) - torch.log(sigma)
            loss = -delta * log_pi
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
                log_pi = -((actions - mu[:, 0]) ** 2) / (2 * sigma[:, 0] ** 2) - torch.log(sigma[:, 0])
                loss = -(deltas * log_pi).sum()
                loss.backward()
                return loss

            self.policy.optimizer.zero_grad()
            self.policy.optimizer.step(loss)

    def batch_update(self):
        return self.BatchUpdater(self)

    def save(self, filename: str):
        lib.util.ensure_dir(filename, need_strip_end=True)
        torch.save(
            {
                "mu_net": self.theta_mu.state_dict(),
                "sigma_net": self.theta_sigma.state_dict(),
            },
            filename,
        )

    def load(self, filename: str):
        state_dict = torch.load(filename)
        self.theta_mu.load_state_dict(state_dict["mu_net"])
        self.theta_mu.eval()
        self.theta_sigma.load_state_dict(state_dict["sigma_net"])
        self.theta_sigma.eval()


def policy_gradient_for_stacked(
    env: Env,
    pi: GaussianPolicy,
    episode_n: int,
    *,
    ax: plt.Axes = None,
    axs_env: Tuple[plt.Axes] = None,
    last_day_train_only: bool = False,
    batch: bool = False,
    plot: bool = True,
    pi_frozen: bool = False,
):
    collector = lib.util.EMACollector(half_life=100, t_return=None)
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

        if not pi_frozen:
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
                        pi.update(
                            delta,
                            actions[t],
                            states[t],
                            info,
                            state_info_tensors=state_info_tensors,
                            update_on=i - t,
                        )
            if batch:
                updater.update()

        t_return = np.sum(rewards)
        collector.append(t_return=t_return)
        pbar.set_description(
            f"avg_r={t_return:.2e};r={info.r:.4f};mu={info.mu:.4f};sigma={info.sigma:.4f};K={info.strike_price:.4f}"
        )
        if (e + 1) % 100 == 0 and ax is not None and plot:
            ax.cla()
            collector.plot(ax)
            # ax.set(yscale='log', ylabel='episodic return')
            plt.show(block=False)
            plt.pause(0.01)

    return collector


class Test(TestCase):
    def test_policy_gradient(self):
        import matplotlib as mpl

        mpl.use("TkAgg")
        sns.set_style("whitegrid")

        is_call_option = True
        initial_asset_price = 1
        max_time = 5
        itr = 10000

        strike_price = 1.000
        mu = 0.0e-3
        sigma = 1e-1
        r = 1.0e-2
        friction = 0
        simplified = True
        mutation_lambda = 1e-2

        def mutate(env: RLOPEnv):
            if np.random.rand(1)[0] < mutation_lambda:
                env.initial_asset_price = np.random.rand(1)[0] * 0.4 + 0.8

        env = RLOPEnv(
            is_call_option=is_call_option,
            strike_price=strike_price,
            max_step=max_time,
            mu=mu,
            sigma=sigma,
            r=r,
            friction=friction,
            initial_estimator=BSInitialEstimator(is_call_option),
            initial_asset_price=initial_asset_price,
            mutation=mutate,
        )

        pi = GaussianPolicy(simplified, 1e-3)
        collector = policy_gradient_for_stacked(env, pi, itr, axs_env=True, batch=True, last_day_train_only=False)

    def test_cumulative_training(self):
        import matplotlib as mpl
        from lib.util.timer import Timer

        mpl.use("TkAgg")
        sns.set_style("whitegrid")

        is_call_option = True
        initial_asset_price = 1
        max_time = 3
        itr = 10000
        simplified = True

        fig, ax = plt.subplots()
        _, axs_env = plt.subplots(2, 1)
        with Timer(desc=f"cumulative training for {max_time} levels"):
            pi = GaussianPolicy(simplified, 1e-3)
            for sub_max_time in range(1, max_time + 1):
                strike_price = 1.000
                mu = 1.0e-3
                sigma = 1e-2
                r = 1.0e-3
                friction = 0.0

                env = RLOPEnv(
                    is_call_option=is_call_option,
                    strike_price=strike_price,
                    max_step=sub_max_time,
                    mu=mu,
                    sigma=sigma,
                    r=r,
                    friction=friction,
                    initial_estimator=BSInitialEstimator(is_call_option),
                    initial_asset_price=initial_asset_price,
                )

                policy_gradient_for_stacked(env, pi, itr, ax=ax, axs_env=axs_env, last_day_train_only=True)
        plt.show(block=True)

    def test_BS_error(self):
        import matplotlib as mpl
        from lib.util.timer import Timer

        mpl.use("TkAgg")
        sns.set_style("whitegrid")

        is_call_option = True
        initial_asset_price = 1

        itr = 2000

        strike_price = 1.000
        mu = 1.0e-3
        sigma = 1e-2
        r = 1.0e-3
        _dt = 1
        friction = 0.01
        T = 10

        max_time = int(np.round(T / _dt))

        env = RLOPEnv(
            is_call_option=is_call_option,
            strike_price=strike_price,
            max_step=max_time,
            mu=mu,
            sigma=sigma,
            r=r,
            friction=friction,
            initial_estimator=BSInitialEstimator(is_call_option),
            initial_asset_price=initial_asset_price,
            mutation=0,
            _dt=_dt,
        )
        pi = BSPolicy(is_call=is_call_option)
        with Timer(desc=f"BS baseline for {itr} iterations"):
            policy_gradient_for_stacked(env, pi, itr, axs_env=False, pi_frozen=True)
        plt.show(block=True)
