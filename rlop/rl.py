from itertools import chain
from unittest import TestCase

import numpy as np
import torch
from gym import Env
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.optim.adam import Adam

from rlop.env import NAP, RT, PV, SP, Mu, Sigma, RLOPEnv
from util.pricing import bs_euro_vanilla_call, bs_euro_vanilla_put


def state_info_to_tensor(state, info):
    normalized_asset_price = state[NAP]
    remaining_time = state[RT]
    portfolio_value = state[PV]
    strike_price = info[SP]
    mu = info[Mu]
    sigma = info[Sigma]

    return [torch.tensor(
        [normalized_asset_price, remaining_time - t, portfolio_value[-(t + 1)], strike_price, mu, sigma]
    ) for t in range(remaining_time)]


def state_info_to_tensor_short(state, info):
    normalized_asset_price = state[NAP]
    remaining_time = state[RT]
    portfolio_value = state[PV]
    strike_price = info[SP]
    mu = info[Mu]
    sigma = info[Sigma]

    return torch.concat([torch.tensor([normalized_asset_price, remaining_time, strike_price, mu, sigma]), torch.tensor(portfolio_value)])


class FCNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(FCNet, self).__init__()
        composite_dims = [input_dim, *hidden_dims, 1]
        self.hidden_layers = nn.ModuleList([
            torch.nn.Linear(composite_dims[i], composite_dims[i + 1]).double()
            for i in range(len(hidden_dims) + 1)])

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = F.relu(layer(x))
        return self.hidden_layers[-1](x)


class Policy:
    def __init__(self, alpha):
        self.theta_mu = FCNet(6, (50, 50))
        self.theta_sigma = FCNet(6, (50, 50))

        self.optimizer = Adam(chain(self.theta_mu.parameters(), self.theta_sigma.parameters()), lr=alpha)

    def action(self, state, info):
        remaining_time = state[RT]
        positions = np.zeros(len(state[PV]))
        tensors = state_info_to_tensor(state, info)
        for t in range(remaining_time):
            theta_mu = self.theta_mu(tensors[t]).detach()
            theta_sigma = self.theta_sigma(tensors[t]).detach()
            positions[-(t + 1)] = np.random.randn(1) * np.exp(float(theta_sigma)) + float(theta_mu)
        return np.clip(positions, a_min=-1, a_max=1)

    def update(self, delta, action, state, info):
        remaining_time = state[RT]
        tensors = state_info_to_tensor(state, info)

        def loss(delta, a, tensor):
            mu = self.theta_mu(tensor)
            sigma = self.theta_sigma(tensor)
            log_pi = (a - mu) ** 2 / (2 * sigma ** 2) - torch.log(sigma)
            return -delta * log_pi

        for t in range(remaining_time):
            self.optimizer.zero_grad()
            self.optimizer.step(lambda: loss(delta, action[-(t + 1)], tensors[t]))


class Value:
    def __init__(self, alpha):
        self.v_net = FCNet(6, (50, 50))

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


def policy_gradient(env: Env, V: Value, pi: Policy, episode_n: int, gamma: np.sctypes):
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
    for e in range(episode_n):
        states = []
        actions = []
        rewards = []

        (state, info), done = env.reset(), False
        states.append(state)
        while not done:
            action = pi.action(state, info)
            state, reward, done, _ = env.step(action)
            # env.render()

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
            # delta = G - V(states[t], info)
            delta = G
            # V.update(delta, states[t], info)
            pi.update(delta * np.power(gamma, t), actions[t], state, info)

        avg_rewards.append(np.average(rewards))
        if (e+1) % 100 == 0:
            indices = np.arange(0, e+1)
            ax.cla()
            ax.plot(indices, avg_rewards)
            ax.plot(indices, np.cumsum(avg_rewards) / (1+indices), ls='--')
            fig.canvas.draw_idle()
            plt.show(block=False)


class Test(TestCase):
    def test_policy_gradient(self):
        import matplotlib as mpl
        mpl.use('TkAgg')

        is_call_option = True
        strike_price = 95
        max_time = 10
        mu = 0.5e-2
        sigma = 1e-2
        r = 0.25e-3
        gamma = np.exp(-r)

        initial_portfolio_value = [
            (bs_euro_vanilla_call if is_call_option else bs_euro_vanilla_put)(
                100, strike_price, t, (r * t), sigma * np.sqrt(t)
            ) * np.exp((mu-r) * t) for t in range(1, max_time+1)
        ]

        print(initial_portfolio_value)

        env = RLOPEnv(is_call_option=is_call_option, strike_price=strike_price, max_time=max_time,
                      mu=mu, sigma=sigma, gamma=gamma,
                      initial_portfolio_value=np.array(initial_portfolio_value))

        V = Value(1e-3)
        pi = Policy(1e-3)
        policy_gradient(env, V, pi, 1000, gamma)
        plt.show(block=True)
