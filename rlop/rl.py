from itertools import chain
from unittest import TestCase

import numpy as np
import torch
from gym import Env
from matplotlib import pyplot as plt
from torch.optim.adam import Adam
from tqdm import trange

import util
from rlop.env import RLOPEnv
from rlop.interface import Policy, Baseline
from util.net import ResNet
from util.pricing import bs_euro_vanilla_call, bs_euro_vanilla_put, optimal_hedging_position


# def state_info_to_tensor_short(state, info):
#     normalized_asset_price = state[NAP]
#     remaining_time = state[RT]
#     portfolio_value = state[PV]
#     strike_price = info[SP]
#     mu = info[Mu]
#     sigma = info[Sigma]
#
#     return torch.concat([torch.tensor([normalized_asset_price, remaining_time, strike_price, mu, sigma]),
#                          torch.tensor(portfolio_value)])


class GaussianPolicy(Policy):
    def __init__(self, alpha):
        super().__init__()
        self.theta_mu = ResNet(7, 20, groups=4, layer_per_group=2)
        self.theta_sigma = ResNet(7, 20, groups=4, layer_per_group=2)

        self.optimizer = Adam(chain(self.theta_mu.parameters(), self.theta_sigma.parameters()), lr=alpha)

    def action(self, state, info, state_info_tensors=None):
        rt = state.remaining_time
        positions = state.portfolio_value * 0
        tensors = state.to_tensors(info) if state_info_tensors is None else state_info_tensors
        for t in range(rt):
            theta_mu = self.theta_mu(tensors[t]).detach()
            theta_sigma = torch.exp(self.theta_sigma(tensors[t]).detach())
            positions[t] = np.random.randn(1) * float(theta_sigma) + float(theta_mu)
        return positions
        # return np.clip(positions, a_min=-1, a_max=1)

    def update(self, delta, action, state, info, state_info_tensors=None, update_on: int = None):
        """

        :param delta:
        :param action:
        :param state:
        :param info:
        :param state_info_tensors:
        :param update_on:
        :return:
        """
        tensors = state.to_tensors(info) if state_info_tensors is None else state_info_tensors

        def loss(delta, a, tensor):
            mu = self.theta_mu(tensor)
            sigma = torch.exp(self.theta_sigma(tensor))
            log_pi = (a - mu) ** 2 / (2 * sigma ** 2) - torch.log(sigma)
            return -delta * log_pi

        self.optimizer.zero_grad()
        self.optimizer.step(lambda: loss(delta, action[update_on], tensors[update_on]))


class FCNetBaseline(Baseline):
    def __init__(self, alpha):
        super().__init__()
        self.v_net = ResNet(6, 20, groups=4, layer_per_group=2)

        self.optimizer = Adam(self.v_net.parameters(), lr=alpha)

    def __call__(self, state, info):
        tensor = state_info_to_tensor_short(state, info)
        return float(self.v_net(tensor).detach())

    def update(self, delta, state, info):
        tensor = state_info_to_tensor_short(state, info)

        def loss(delta, tensor):
            return -delta * self.v_net(tensor)

        self.optimizer.zero_grad()
        self.optimizer.step(lambda: loss(delta, tensor))


def policy_gradient(env: Env, V: Baseline, pi: Policy, episode_n: int, gamma: np.sctypes):
    """

    :param env:
    :param V:
    :param pi:
    :param episode_n:
    :param gamma:
    :return:
    """
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
            if e == episode_n - 1:
                env.render()

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        T = len(actions)
        G_tmp = 0
        Gs_rev = []
        for t in range(T - 1, -1, -1):
            G_tmp = G_tmp * gamma + rewards[t]
            Gs_rev.append(G_tmp)
        Gs = Gs_rev[::-1]

        for t in range(T):
            G = Gs[t]
            delta = G
            if V is not None:
                delta -= V(states[t], info)
                V.update(delta, states[t], info)
            pi.update(delta * np.power(gamma, t), actions[t], state, info)

        avg_rewards.append(np.average(rewards))
        pbar.set_description(f'avg_reward={avg_rewards[-1]:.2f}')
        if (e + 1) % 100 == 0:
            indices = np.arange(0, e + 1)
            ax.cla()
            ax.plot(indices, avg_rewards)
            ax.plot(indices, np.cumsum(avg_rewards) / (1 + indices), ls='--')
            fig.canvas.draw_idle()
            plt.show(block=False)


def policy_gradient_for_stacked(env: Env, V: Baseline, pi: Policy, episode_n: int):
    """

    :param env:
    :param V:
    :param pi:
    :param episode_n:
    :return:
    """
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
            if e == episode_n - 1:
                env.render()

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        T = len(actions)

        for t in range(T):
            state_info_tensors = states[t].to_tensors(info)
            for i in range(t, T):
                delta = rewards[i]
                if V is not None:
                    delta -= V(states[t], info)
                    V.update(delta, states[t], info)  # TODO: ?
                pi.update(delta, actions[t], states[t], info, state_info_tensors=state_info_tensors, update_on=i - t)

        avg_rewards.append(np.average(rewards))
        pbar.set_description(f'avg_reward={avg_rewards[-1]:.2e}')
        if (e + 1) % 100 == 0:
            indices = np.arange(0, e + 1)
            ax.cla()
            ax.plot(indices, avg_rewards)
            ax.plot(indices, np.cumsum(avg_rewards) / (1 + indices), ls='--')
            fig.canvas.draw_idle()
            plt.show(block=False)


def policy_evaluation(env: Env, pi: Policy, episode_n: int):
    """

    :param env:
    :param pi:
    :param episode_n:
    :return:
    """
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
            env.render()

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        avg_rewards.append(np.average(rewards))
        pbar.set_description(f'avg_reward={avg_rewards[-1]:.2e}')
        if (e + 1) % 100 == 0:
            indices = np.arange(0, e + 1)
            ax.cla()
            ax.plot(indices, avg_rewards)
            ax.plot(indices, np.cumsum(avg_rewards) / (1 + indices), ls='--')
            fig.canvas.draw_idle()
            plt.show(block=False)


class BSPolicy(Policy):
    def __init__(self, is_call: bool = True):
        super().__init__()
        self.option_func = bs_euro_vanilla_call if is_call else bs_euro_vanilla_put

    def action(self, state, info):
        S = util.normalized_to_standard_price(state.normalized_asset_price, info.mu, info.sigma, state.remaining_time)
        K = info.strike_price
        return np.array([optimal_hedging_position(S, K, t + 1, info.r, info.sigma, self.option_func) for t in
                range(state.remaining_time)])

    def update(self, delta: np.sctypes, action, state, info, *args):
        raise Exception('BS policy cannot be updated!')


class Test(TestCase):
    def test_policy_gradient(self):
        import matplotlib as mpl
        mpl.use('TkAgg')

        is_call_option = True
        initial_asset_price = 1
        strike_price = 1.000
        max_time = 50
        mu = 0.0e-3
        sigma = 1e-2
        r = 0.0e-3
        gamma = np.exp(-r)

        initial_portfolio_value = [
            (bs_euro_vanilla_call if is_call_option else bs_euro_vanilla_put)(
                initial_asset_price, strike_price, t, r, sigma
            ) * np.exp((mu - r) * t) for t in range(1, max_time + 1)
        ]

        print(initial_portfolio_value)

        env = RLOPEnv(is_call_option=is_call_option, strike_price=strike_price, max_time=max_time, mu=mu, sigma=sigma,
                      r=r, initial_portfolio_value=np.array(initial_portfolio_value),
                      initial_asset_price=initial_asset_price)

        # V = FCNetBaseline(1e-2)
        # pi = GaussianPolicy(1e-2)
        # policy_gradient_for_stacked(env, None, pi, 2000)
        # plt.show(block=True)

        pi = BSPolicy(is_call=is_call_option)
        policy_evaluation(env, pi, 1000)
        plt.show(block=True)
