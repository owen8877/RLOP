from itertools import chain
from typing import Tuple
from unittest import TestCase

import numpy as np
import torch
from gym import Env
from matplotlib import pyplot as plt
from torch.optim.adam import Adam
from tqdm import trange

import util
from qlbs.env import State, Info, QLBSEnv, Policy, Baseline
from util.net import ResNet
from util.sample import geometricBM


class GaussianPolicy(Policy):
    def __init__(self, alpha):
        super().__init__()
        self.theta_mu = ResNet(7, 10, groups=2, layer_per_group=2)
        self.theta_sigma = ResNet(7, 10, groups=2, layer_per_group=2)

        self.optimizer = Adam(chain(self.theta_mu.parameters(), self.theta_sigma.parameters()), lr=alpha)

    def _gauss_param(self, tensor):
        tensor = tensor.float()
        mu = self.theta_mu(tensor)
        sigma = self.theta_sigma(tensor)

        mu_c = torch.sigmoid(mu)
        sigma_s = torch.sigmoid(sigma)

        sigma_c = torch.clip(sigma_s, 1e-1, 1)

        return mu_c, sigma_c

    def action(self, state, info):
        tensor = state.to_tensor(info)
        with torch.no_grad():
            mu, sigma = self._gauss_param(tensor)
            return float(np.random.randn(1)) * float(sigma) + float(mu)

    def update(self, delta, action, state, info):
        tensor = state.to_tensor(info)

        def loss(delta, a, tensor):
            mu, sigma = self._gauss_param(tensor)
            log_pi = - (a - mu) ** 2 / (2 * sigma ** 2) - torch.log(sigma)
            loss = - delta * log_pi
            loss.backward()
            return loss

        self.optimizer.zero_grad()
        self.optimizer.step(lambda: loss(delta, action, tensor))

    def batch_action(self, state_info_tensor):
        """
        :param state_info_tensor: [[normal_price, remaining_real_time, strike_price, r, mu, sigma, risk_lambda]]
        :return:
        """
        with torch.no_grad():
            mu, sigma = self._gauss_param(state_info_tensor)
            return torch.randn(len(mu)) * sigma[:, 0] + mu[:, 0]

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


class NNBaseline(Baseline):
    def __init__(self, alpha=1e-2):
        super().__init__()
        self.net = ResNet(7, 10, groups=2, layer_per_group=2)
        self.optimizer = Adam(self.net.parameters(), lr=alpha)

    def __call__(self, state: State, info: Info):
        tensor = state.to_tensor(info)
        with torch.no_grad():
            return float(self.net(tensor.float()))

    def update(self, delta: float, state: State, info: Info):
        tensor = state.to_tensor(info)

        def loss(delta, tensor):
            loss = -delta * self.net(tensor.float())
            loss.backward()
            return loss

        self.optimizer.zero_grad()
        self.optimizer.step(lambda: loss(delta, tensor))


def policy_gradient(env: Env, pi: Policy, V: Baseline, episode_n: int, *, ax: plt.Axes = None,
                    axs_env: Tuple[plt.Axes] = None):
    t0_returns = []
    t0_risks = []
    if ax is None:
        fig, ax = plt.subplots()
    pbar = trange(episode_n)
    for e in pbar:
        states = []
        actions = []
        rewards = []
        risks = []

        (state, info), done = env.reset(), False
        states.append(state)
        while not done:
            action = pi.action(state, info)
            state, reward, done, additional = env.step(action, pi)
            if e == episode_n - 1 and axs_env is not None:
                env.render(axs=axs_env)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            risks.append(additional['risk'])

        if V is not None:
            T = len(actions)
            G_tmp = 0
            Gs_rev = []
            for t in range(T - 1, -1, -1):
                G_tmp = G_tmp * env.gamma + rewards[t]
                Gs_rev.append(G_tmp)
            Gs = Gs_rev[::-1]

            for t in range(T):
                delta = Gs[t] - V(states[t], info)
                V.update(delta, states[t], info)
                pi.update(delta * np.power(env.gamma, t), actions[t], states[t], info)

        discount = np.power(env.gamma, np.arange(len(rewards))[::-1])
        t0_returns.append(np.dot(rewards, discount))
        t0_risks.append(env.info.risk_lambda * np.dot(risks, discount))
        pbar.set_description(
            f't0_return={t0_returns[-1]:.2e};t0_risks={t0_risks[-1]:.2e};r={info.r:.4f};mu={info.mu:.4f};sigma={info.sigma:.4f};K={info.strike_price:.4f}')
        if (e + 1) % 100 == 0 and ax is not None:
            indices = np.arange(0, e + 1)
            ax.cla()
            option_price = np.array(t0_returns) * (-1)
            ax.plot(indices, option_price)
            ax.plot(indices, option_price + np.array(t0_risks), ls=':')
            ax.plot(indices, option_price - np.array(t0_risks), ls=':')
            ax.plot(indices, np.cumsum(t0_returns) / (1 + indices) * (-1), ls='--')
            ax.set(yscale='linear', ylabel='negative reward (=option price)')
            plt.show(block=False)
            plt.pause(0.001)
    return t0_returns


class Test(TestCase):
    class HedgeEnv:
        def __init__(self, remaining_till, is_call_option, _dt: float = 1):
            self.state = State(0, 0)
            self.remaining_till = remaining_till
            self.is_call_option = is_call_option
            self._dt = _dt

        def reset(self, info: Info, asset_normalized_prices: np.ndarray, asset_standard_prices: np.ndarray):
            self.info = info
            self.gamma = np.exp(-self.info.r * self._dt)
            self.asset_normalized_prices = asset_normalized_prices
            self.asset_standard_prices = asset_standard_prices

            self.portfolio_value = util.payoff_of_option(self.is_call_option, self.asset_standard_prices[-1],
                                                         self.info.strike_price)
            self.state.remaining_time = 1
            self.state.normalized_asset_price = self.asset_normalized_prices[-2]
            return self.state

        def step(self, hedge) -> Tuple[State, float, float, bool]:
            rt = self.state.remaining_time
            dS = self.asset_standard_prices[-rt] - self.asset_standard_prices[-rt - 1] / self.gamma
            in_position_change = self.gamma * hedge * dS
            self.portfolio_value *= self.gamma
            self.portfolio_value -= in_position_change

            self.state.remaining_time = rt + 1
            done = self.state.remaining_time > self.remaining_till
            if not done:
                self.state.normalized_asset_price = self.asset_normalized_prices[-rt - 2]
            return self.state, self.portfolio_value, in_position_change, done

    def test_hedge_env_bs(self):
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        from tqdm import trange
        import seaborn as sns
        import pandas as pd
        from qlbs.bs import BSInitialEstimator, BSPolicy
        mpl.use('TkAgg')
        sns.set_style('whitegrid')

        is_call_option = True
        r = 0e-3
        mu = 0e-3
        sigma = 5e-3
        risk_lambda = 1
        initial_price = 1
        strike_price = 1.001
        T = 10
        _dt = 0.01

        max_time = int(np.round(T / _dt))
        env = Test.HedgeEnv(remaining_till=max_time, is_call_option=is_call_option)
        bs_pi = BSPolicy(is_call=is_call_option)
        bs_estimator = BSInitialEstimator(is_call_option)

        initial_errors = []
        linf_errors = []
        for _ in trange(10):
            standard_prices, normalized_prices = geometricBM(initial_price, max_time, 1, mu, sigma, _dt)
            standard_prices = standard_prices[0, :]
            normalized_prices = normalized_prices[0, :]
            info = Info(strike_price=strike_price, r=r, mu=mu, sigma=sigma, risk_lambda=risk_lambda, _dt=_dt)
            state = env.reset(info, normalized_prices, standard_prices)
            done = False

            bs_option_prices = np.array(
                [bs_estimator(standard_prices[t], strike_price, max_time - t, r, sigma, _dt) for t in
                 range(max_time + 1)])

            pvs = np.zeros(max_time + 1)
            hedges = np.zeros(max_time + 1)
            pvs[-1] = util.payoff_of_option(is_call_option, standard_prices[-1], strike_price)
            while not done:
                hedge = bs_pi.action(state, info)
                state, pv, in_position_change, done = env.step(hedge)
                pvs[-state.remaining_time] = pv
                hedges[-state.remaining_time] = hedge

            initial_errors.append(pvs[0] - bs_option_prices[0])
            linf_errors.append(np.linalg.norm(pvs - bs_option_prices, ord=np.inf))

            fig, (ax_price, ax_option, ax_hedge) = plt.subplots(3, 1, figsize=(4, 5))
            times = np.arange(0, max_time + 1)
            ax_price.plot(times, standard_prices)
            ax_price.set(ylabel='stock price')
            ax_option.plot(times, pvs, ls='--', label='portfolio')
            ax_option.plot(times, bs_option_prices, label='bs price')
            ax_option.legend(loc='best')
            ax_hedge.plot(times, hedges)

            plt.show(block=True)

        sns.histplot(pd.DataFrame({'initial': initial_errors, 'inf': linf_errors}))
        plt.show(block=True)

    def test_qlbs_env(self):
        import matplotlib as mpl
        import seaborn as sns
        from qlbs.bs import BSPolicy
        mpl.use('TkAgg')
        sns.set_style('whitegrid')

        is_call_option = True
        r = 1e-2
        mu = 0e-3
        sigma = 1e-1
        risk_lambda = 1
        initial_price = 1
        strike_price = 1
        T = 5
        _dt = 1

        max_time = int(np.round(T / _dt))
        env = QLBSEnv(is_call_option=is_call_option, strike_price=strike_price, max_time=max_time, mu=mu, sigma=sigma,
                      r=r, risk_lambda=risk_lambda, initial_asset_price=initial_price, risk_simulation_paths=50,
                      _dt=_dt, mutation=0)
        bs_pi = BSPolicy(is_call=is_call_option)

        policy_gradient(env, bs_pi, None, episode_n=10000)
        plt.show()

    def test_gaussian_policy_training(self):
        import matplotlib as mpl
        import seaborn as sns
        mpl.use('TkAgg')
        sns.set_style('whitegrid')

        is_call_option = True
        r = 1e-2
        mu = 0e-3
        sigma = 1e-1
        risk_lambda = 1
        initial_price = 1
        strike_price = 1
        T = 5
        _dt = 1

        max_time = int(np.round(T / _dt))
        env = QLBSEnv(is_call_option=is_call_option, strike_price=strike_price, max_time=max_time, mu=mu, sigma=sigma,
                      r=r, risk_lambda=risk_lambda, initial_asset_price=initial_price, risk_simulation_paths=50,
                      _dt=_dt, mutation=0)
        gaussian_pi = GaussianPolicy(alpha=1e-2)
        nnbaseline = NNBaseline(alpha=1e-2)

        policy_gradient(env, gaussian_pi, nnbaseline, episode_n=100000)
        plt.show()
