from itertools import chain
from math import isnan
from os import write
from typing import Tuple, TypeVar
from unittest import TestCase

import numpy as np
import scipy as sp
import torch
from gymnasium import Env
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import lib.util
from lib.util.net import StrictResNet, CombinedResNet
from lib.util.sample import geometricBM

from .bs import BSBaseline, BSPolicy
from .env import Baseline, Info, Policy, QLBSEnv, State


class NetHolder:
    def __init__(self, from_filename: str | None):
        def in_transform(x: torch.Tensor) -> torch.Tensor:
            # x[:, 0].log_()  # spot
            # x[:, 3].log_()  # strike
            return x

        def out_transform(y: torch.Tensor) -> torch.Tensor:
            y[:, 0] = -torch.exp(y[:, 0])  # price
            print(1, y[:, 1])
            y[:, 1] = torch.sigmoid(y[:, 1])  # delta
            print(2, y[:, 1])
            return y

        self.net = CombinedResNet(
            input_dim=9,
            hidden_dim=64,
            transform_pair=(in_transform, out_transform),
            activation="elu",
            groups=3,
            layer_per_group=3,
        ).cuda()
        if from_filename is not None:
            state_dict = torch.load(from_filename)
            self.net.load_state_dict(state_dict)
            self.net.cuda()
            self.net.eval()

    def save(self, filename: str):
        lib.util.ensure_dir(filename, need_strip_end=True)
        torch.save(self.net.state_dict(), filename)


T1 = TypeVar("T1", float, torch.Tensor, np.ndarray)


class GaussianPolicy_v2(Policy):
    def __init__(self, holder: NetHolder, is_call_option: bool, alpha=1e-2, from_filename: str | None = None):
        super().__init__()
        self.theta_mu_holder = holder
        self.is_call_option = is_call_option
        self.theta_sigma = StrictResNet(9, 10, groups=2, layer_per_group=2).cuda()
        self.optimizer = Adam(chain(self.theta_mu_holder.net.parameters(), self.theta_sigma.parameters()), lr=alpha)
        if from_filename is not None:
            self.load(from_filename)

    def _latent_to_delta(self, x: T1) -> T1:
        _sigmoid = torch.sigmoid if isinstance(x, torch.Tensor) else sp.special.expit
        return (1 if self.is_call_option else -1) * _sigmoid(x)  # type: ignore

    def _delta_to_latent(self, delta: T1, eps: float = 1e-3) -> T1:
        _logit = torch.special.logit if isinstance(delta, torch.Tensor) else sp.special.logit
        _cutoff = torch.clamp if isinstance(delta, torch.Tensor) else np.clip
        return _logit(_cutoff((1 if self.is_call_option else -1) * delta, eps, 1 - eps))  # type: ignore

    def _gauss_param(self, tensor):
        tensor = tensor.float().cuda()
        _mu = self.theta_mu_holder.net(tensor)
        print(3, _mu)
        _mu = self._delta_to_latent(_mu[:, 1:2])
        print(4, _mu)

        if torch.any(torch.isnan(_mu)):
            breakpoint()
            raise ValueError("NaN encountered in mu computation!")
        sigma = self.theta_sigma(tensor)

        sigma_c = torch.sigmoid(sigma) * 0.3 + 0.03

        return _mu, sigma_c

    def action(self, state, info):
        tensor = state.to_tensor(info)[None, :]
        with torch.no_grad():
            mu, sigma = self._gauss_param(tensor)
            print(5, mu, sigma)
            _r = self._latent_to_delta(float(np.random.randn(1)) * float(sigma) + float(mu))
            print(6, mu, sigma)
            return _r

    def update(self, delta, action, state, info):
        tensor = state.to_tensor(info)[None, :]

        def loss_func():
            mu, sigma = self._gauss_param(tensor)
            _action, _mu = self._delta_to_latent(action), mu
            print(_action, _mu, sigma, ((_action - _mu) ** 2) / (2 * sigma**2), torch.log(sigma))
            log_pi = -((_action - _mu) ** 2) / (2 * sigma**2) - torch.log(sigma)
            loss = -delta * log_pi
            loss.backward()
            print(delta, loss, log_pi)
            raise
            return loss

        self.optimizer.zero_grad()
        loss = self.optimizer.step(loss_func)  # type: ignore
        return loss

    def batch_action(self, state_info_tensor, random: bool = True):
        """
        :param state_info_tensor: [[normal_price, remaining_real_time, normal_strike_price, r, mu, sigma, risk_lambda]]
        :return:
        """
        with torch.no_grad():
            mu, sigma = self._gauss_param(state_info_tensor)
            if random:
                return self._latent_to_delta((torch.randn(len(mu)).cuda() * sigma[:, 0] + mu[:, 0]))  # .numpy()
            else:
                return self._latent_to_delta(mu[:, 0])  # .numpy()

    def save(self, filename: str):
        lib.util.ensure_dir(filename, need_strip_end=True)
        torch.save({"sigma_net": self.theta_sigma.state_dict()}, filename)

    def load(self, filename: str):
        state_dict = torch.load(filename)
        self.theta_sigma.load_state_dict(state_dict["sigma_net"])
        self.theta_sigma.cuda()
        self.theta_sigma.eval()

    def train_based_on(self, source, target, lr, itr_max):
        optimizer = Adam(chain(self.theta_mu_holder.net.parameters(), self.theta_sigma.parameters()), lr=lr)
        pbar = trange(int(itr_max))
        for _ in pbar:

            def loss_func():
                mu, sigma = self._gauss_param(source)
                prediction = torch.randn(len(mu)) * sigma[:, 0] + mu[:, 0]
                loss = torch.mean((prediction - target) ** 2)
                # loss = torch.mean((mu[:, 0] - target) ** 2) + torch.mean((sigma[:, 0] - 0.2) ** 2)
                loss.backward()
                return loss

            optimizer.zero_grad()
            loss = optimizer.step(loss_func)  # type: ignore
            pbar.set_description(desc=f"loss={loss:.5e}")


class NNBaseline_v2(Baseline):
    def __init__(self, holder: NetHolder, alpha=1e-2):
        super().__init__()

        self.holder = holder
        self.optimizer = Adam(self.holder.net.parameters(), lr=alpha)

    def _predict(self, tensor):
        y = self.holder.net(tensor.float().cuda())
        return y[:, 0]

    def __call__(self, state: State, info: Info):
        tensor = state.to_tensor(info)[None, :].cuda()
        with torch.no_grad():
            return float(self._predict(tensor))

    def update(self, G: float, state: State, info: Info):
        tensor = state.to_tensor(info)[None, :].cuda()

        def loss_func():
            _G = np.log(max(1e-6, -G))
            tensor1 = self._predict(tensor)
            _tensor1 = torch.log(torch.clamp(-tensor1, min=1e-6))
            loss = (_G - _tensor1) ** 2
            loss.backward()
            return loss

        self.optimizer.zero_grad()
        loss = self.optimizer.step(loss_func)
        return loss

    def batch_estimate(self, state_info_tensor):
        with torch.no_grad():
            return self._predict(state_info_tensor)

    def save(self, filename: str):
        lib.util.ensure_dir(filename, need_strip_end=True)
        torch.save(self.holder.net.state_dict(), filename)

    def train_based_on(self, source, target, lr, itr_max):
        optimizer = Adam(self.holder.net.parameters(), lr=lr)
        pbar = trange(int(itr_max))
        for _ in pbar:

            def loss_func():
                prediction = self._predict(source)[:, 0]
                loss = torch.mean((prediction - target) ** 2)
                loss.backward()
                return loss

            optimizer.zero_grad()
            loss = optimizer.step(loss_func)  # type: ignore
            pbar.set_description(desc=f"loss={loss:.5e}")


def policy_gradient(
    env: Env,
    pi: Policy,
    V: Baseline,
    episode_n: int,
    *,
    V_frozen: bool = False,
    pi_frozen: bool = False,
    tensorboard_label: str | None = None,
):
    collector = lib.util.EMACollector(half_life=100, t_return=None, t_base_return=None)
    V_loss, pi_loss = None, None
    if tensorboard_label:
        writer = SummaryWriter(f"runs/{tensorboard_label}")
        fig, ax = plt.subplots()

    for e in range(episode_n):
        states = []
        actions = []
        rewards = []
        risks = []

        (state, info), done = env.reset(), False  # type:ignore
        info: Info
        states.append(state)
        while not done:
            action = pi.action(state, info)
            state, reward, done, additional = env.step(action, pi)  # type:ignore

            if isnan(reward):
                raise ValueError("NaN encountered!")

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            risks.append(additional["risk"])

        if V is not None:
            T = len(actions)
            G_tmp = 0
            Gs_rev = []
            for t in range(T - 1, -1, -1):
                G_tmp = G_tmp * env.gamma + rewards[t]  # type: ignore
                Gs_rev.append(G_tmp)
            Gs = Gs_rev[::-1]

            for t in range(T):
                delta = Gs[t] - V(states[t], info)
                if not V_frozen:
                    V_loss = V.update(Gs[t], states[t], info)
                    # print(f"{V_loss=}")
                if not pi_frozen:
                    pi_loss = pi.update(delta * np.power(env.gamma, t), actions[t], states[t], info)  # type: ignore
                    # print(f"{pi_loss=}")

        discount = np.power(env.gamma, np.arange(len(rewards)))  # type: ignore
        t0_return = np.dot(rewards, discount)
        t0_risk = env.info.risk_lambda * np.dot(risks, discount) * info._dt  # type: ignore

        collector.append(t_return=t0_return, t_base_return=t0_return + t0_risk)
        collector.write(writer)

        writer.add_scalar("info.strike_price", env.info.strike_price, e + 1)  # type: ignore
        writer.add_scalar("info.mu", env.info.mu, e + 1)  # type: ignore
        writer.add_scalar("info.sigma", env.info.sigma, e + 1)  # type: ignore
        writer.add_scalar("info.r", env.info.r, e + 1)  # type: ignore
        writer.add_scalar("info.risk_lambda", env.info.risk_lambda, e + 1)  # type: ignore
        writer.add_scalar("info.friction", env.info.friction, e + 1)  # type: ignore
        writer.add_scalar("initial_asset_price", env.initial_asset_price, e + 1)  # type: ignore
        writer.add_scalar("pi_loss", pi_loss if not pi_frozen else np.nan, e + 1)  # type: ignore

        if (e + 1) % 100 == 0 and tensorboard_label:
            ax.cla()
            collector.plot(ax)
            ax.set(ylabel="return (=negative option price)")
            writer.add_figure("combined", fig, e + 1)

    return collector


class Test(TestCase):
    class HedgeEnv:
        def __init__(self, remaining_till, is_call_option, _dt: float = 1):
            self.state = State(0, 0, 1)
            self.remaining_till = remaining_till
            self.is_call_option = is_call_option
            self._dt = _dt

        def reset(
            self,
            info: Info,
            asset_normalized_prices: np.ndarray,
            asset_standard_prices: np.ndarray,
        ):
            self.info = info
            self.gamma = np.exp(-self.info.r * self._dt)
            self.asset_normalized_prices = asset_normalized_prices
            self.asset_standard_prices = asset_standard_prices

            self.portfolio_value = lib.util.payoff_of_option(
                self.is_call_option,
                self.asset_standard_prices[-1],
                self.info.strike_price,
            )
            self.state.remaining_step = 1
            self.state.normalized_asset_price = self.asset_normalized_prices[-2]
            return self.state

        def step(self, hedge) -> Tuple[State, float, float, bool]:
            rt = self.state.remaining_step
            dS = self.asset_standard_prices[-rt] - self.asset_standard_prices[-rt - 1] / self.gamma
            in_position_change = self.gamma * hedge * dS
            self.portfolio_value *= self.gamma
            self.portfolio_value -= in_position_change

            self.state.remaining_step = rt + 1
            done = self.state.remaining_step > self.remaining_till
            if not done:
                self.state.normalized_asset_price = self.asset_normalized_prices[-rt - 2]
            return self.state, self.portfolio_value, in_position_change, done

    def test_hedge_env_bs(self):
        import matplotlib as mpl
        import pandas as pd
        import seaborn as sns
        from matplotlib import pyplot as plt
        from qlbs.bs import BSInitialEstimator, BSPolicy
        from tqdm import trange

        mpl.use("TkAgg")
        sns.set_style("whitegrid")

        is_call_option = True
        r = 0e-3
        mu = 0e-3
        sigma = 5e-3
        risk_lambda = 1
        initial_price = 1
        strike_price = 1.001
        T = 10
        _dt = 0.01
        friction = 1e-2

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
            info = Info(
                strike_price=strike_price,
                r=r,
                mu=mu,
                sigma=sigma,
                risk_lambda=risk_lambda,
                _dt=_dt,
                friction=friction,
            )
            state = env.reset(info, normalized_prices, standard_prices)
            done = False

            bs_option_prices = np.array(
                [
                    bs_estimator(standard_prices[t], strike_price, max_time - t, r, sigma, _dt)
                    for t in range(max_time + 1)
                ]
            )

            pvs = np.zeros(max_time + 1)
            hedges = np.zeros(max_time + 1)
            pvs[-1] = lib.util.payoff_of_option(is_call_option, standard_prices[-1], strike_price)
            while not done:
                hedge = bs_pi.action(state, info)
                state, pv, in_position_change, done = env.step(hedge)
                pvs[-state.remaining_step] = pv
                hedges[-state.remaining_step] = hedge

            initial_errors.append(pvs[0] - bs_option_prices[0])
            linf_errors.append(np.linalg.norm(pvs - bs_option_prices, ord=np.inf))

            fig, (ax_price, ax_option, ax_hedge) = plt.subplots(3, 1, figsize=(4, 5))
            times = np.arange(0, max_time + 1)
            ax_price.plot(times, standard_prices)
            ax_price.set(ylabel="stock price")
            ax_option.plot(times, pvs, ls="--", label="portfolio")
            ax_option.plot(times, bs_option_prices, label="bs price")
            ax_option.legend(loc="best")
            ax_hedge.plot(times, hedges)

            plt.show(block=True)

        sns.histplot(pd.DataFrame({"initial": initial_errors, "inf": linf_errors}))
        plt.show(block=True)

    def test_qlbs_env(self):
        import matplotlib as mpl
        import seaborn as sns
        from qlbs.bs import BSPolicy

        mpl.use("TkAgg")
        sns.set_style("whitegrid")

        is_call_option = True
        r = 1e-2
        mu = 0e-3
        sigma = 1e-1
        risk_lambda = 1
        initial_price = 1
        strike_price = 1
        T = 3
        _dt = 1
        friction = 1e-2

        max_time = int(np.round(T / _dt))
        env = QLBSEnv(
            is_call_option=is_call_option,
            strike_price=strike_price,
            max_step=max_time,
            mu=mu,
            sigma=sigma,
            r=r,
            risk_lambda=risk_lambda,
            friction=friction,
            initial_asset_price=initial_price,
            risk_simulation_paths=200,
            _dt=_dt,
            mutation=0,
        )
        bs_pi = BSPolicy(is_call=is_call_option)

        policy_gradient(env, bs_pi, None, episode_n=2000)  # type:ignore
        plt.show()

    def test_gaussian_policy_training(self):
        import matplotlib as mpl
        import seaborn as sns

        mpl.use("TkAgg")
        sns.set_style("whitegrid")

        is_call_option = True
        r = 1e-2
        mu = 0e-3
        sigma = 1e-1
        risk_lambda = 0
        initial_price = 1
        strike_price = 1
        T = 5
        _dt = 1
        friction = 1e-1
        simplified = True

        max_time = int(np.round(T / _dt))
        env = QLBSEnv(
            is_call_option=is_call_option,
            strike_price=strike_price,
            max_step=max_time,
            mu=mu,
            sigma=sigma,
            r=r,
            risk_lambda=risk_lambda,
            friction=friction,
            initial_asset_price=initial_price,
            risk_simulation_paths=200,
            _dt=_dt,
            mutation=5e-3,
        )
        gaussian_pi = GaussianPolicy(simplified=simplified, alpha=1e-4)
        nnbaseline = NNBaseline(simplified=simplified, alpha=1e-4)
        bs_pi = BSPolicy(is_call=is_call_option)
        bs_baseline = BSBaseline(is_call=is_call_option)

        # load_plan = 'train_to_bs_test'
        load_plan = f"T{T:d}_test"
        nnbaseline.load(f"trained_model/test/{lib.util._prefix(simplified, load_plan)}/baseline.pt")
        gaussian_pi.load(f"trained_model/test/{lib.util._prefix(simplified, load_plan)}/policy.pt")
        policy_gradient(env, gaussian_pi, nnbaseline, episode_n=4000, pi_frozen=False)
        # policy_gradient(env, bs_pi, nnbaseline, episode_n=4000, pi_frozen=True)
        # save_plan = 'rl_after_pretrain'
        save_plan = f"T{T:d}_test"
        gaussian_pi.save(f"trained_model/test/{lib.util._prefix(simplified, save_plan)}/policy.pt")
        # save_plan = 'qlbs_bs_value'
        nnbaseline.save(f"trained_model/test/{lib.util._prefix(simplified, save_plan)}/baseline.pt")
        plt.show()

    def test_examine_trained_model(self):
        import matplotlib as mpl
        import seaborn as sns

        mpl.use("TkAgg")
        sns.set_style("whitegrid")

        is_call_option = True
        r = 1e-2
        mu = 0e-3
        sigma = 1e-1
        risk_lambda = 0
        strike_price = 1
        max_step = 5
        _dt = 1
        friction = 1e-1
        simplified = True

        plan = f"T{max_step:d}_test"
        # plan = 'train_to_bs_test'
        # plan = 'rl_after_pretrain'
        additional_plan = "qlbs_bs_value"

        gaussian_pi = GaussianPolicy(simplified=simplified, alpha=1e-4)
        gaussian_pi.load(f"trained_model/test/{lib.util._prefix(simplified, plan)}/policy.pt")
        nnbaseline = NNBaseline(simplified=simplified, alpha=1e-4)
        nnbaseline.load(f"trained_model/test/{lib.util._prefix(simplified, plan)}/baseline.pt")
        additional_nnbaseline = NNBaseline(simplified=simplified, alpha=1e-4)
        additional_nnbaseline.load(f"trained_model/test/{lib.util._prefix(simplified, additional_plan)}/baseline.pt")
        bs_pi = BSPolicy(is_call=is_call_option)
        bs_baseline = BSBaseline(is_call=is_call_option)

        T_grid, S_grid = np.meshgrid(
            np.arange(max_step + 1),
            np.linspace(strike_price * 0.8, strike_price * 1.2, 91),
        )
        T_long = T_grid.reshape(-1, 1)[:, 0]
        S_long = S_grid.reshape(-1, 1)[:, 0]

        T1_grid, S1_grid = np.meshgrid(np.arange(max_step), np.linspace(strike_price * 0.8, strike_price * 1.2, 91))
        T1_long = T1_grid.reshape(-1, 1)[:, 0]
        S1_long = S1_grid.reshape(-1, 1)[:, 0]

        def build_state_info_tensor(price, time):
            time = torch.tensor(time)
            price = torch.tensor(price)
            state_info_tensor = torch.empty((len(time), 9))
            state_info_tensor[:, 0] = torch.tensor(
                lib.util.standard_to_normalized_price(price.numpy(), mu, sigma, time.numpy(), _dt)
            )  # normal_price
            state_info_tensor[:, 1] = time * _dt  # passed_real_time
            state_info_tensor[:, 2] = (max_step - time) * _dt  # remaining_real_time
            state_info_tensor[:, 3] = torch.tensor(
                lib.util.standard_to_normalized_price(strike_price, mu, sigma, max_step, _dt)
            )  # normal_strike_price
            state_info_tensor[:, 4] = r  # r
            state_info_tensor[:, 5] = mu  # mu
            state_info_tensor[:, 6] = sigma  # sigma
            state_info_tensor[:, 7] = risk_lambda  # risk_lambda
            state_info_tensor[:, 8] = friction  # friction
            return state_info_tensor

        price_tensor = build_state_info_tensor(S_long, T_long)
        bs_price = bs_baseline.batch_estimate(price_tensor.numpy())
        nn_price = nnbaseline.batch_estimate(price_tensor)
        hedge_tensor = build_state_info_tensor(S1_long, T1_long)
        bs_hedge = bs_pi.batch_action(hedge_tensor, random=False)
        nn_hedge = gaussian_pi.batch_action(hedge_tensor, random=False)

        fig: Figure = plt.figure(figsize=(10, 8))
        ax11 = fig.add_subplot(2, 2, 1, projection="3d")
        ax11.plot_surface(T_grid * _dt, S_grid, bs_price.reshape(T_grid.shape))
        ax11.set_title("BS price")

        ax12 = fig.add_subplot(2, 2, 2, projection="3d")
        ax12.plot_surface(T_grid * _dt, S_grid, -nn_price.reshape(T_grid.shape))
        ax12.set_title("NN price")

        ax21 = fig.add_subplot(2, 2, 3, projection="3d")
        ax21.plot_surface(T1_grid * _dt, S1_grid, bs_hedge.reshape(T1_grid.shape))
        ax21.set_title("BS hedge")

        ax22 = fig.add_subplot(2, 2, 4, projection="3d")
        ax22.plot_surface(T1_grid * _dt, S1_grid, nn_hedge.reshape(T1_grid.shape))
        ax22.set_title("NN hedge")

        T_slice, S_slice = np.meshgrid(0, np.linspace(strike_price * 0.8, strike_price * 1.2, 61))
        T_slice = T_slice.reshape(-1, 1)[:, 0]
        S_slice = S_slice.reshape(-1, 1)[:, 0]

        fig2 = plt.figure(2, figsize=(7, 5))
        price_slice = build_state_info_tensor(S_slice, T_slice)
        plt.plot(
            S_slice,
            bs_baseline.batch_estimate(price_slice.numpy()),
            "--",
            label="vanilla BS",
        )
        plt.plot(
            S_slice,
            -additional_nnbaseline.batch_estimate(price_slice),
            "--",
            label="QLBS",
        )
        plt.plot(S_slice, -nnbaseline.batch_estimate(price_slice), label="NN")
        plt.legend(loc="best")

        plt.show()

    def test_train_nn_with_bs(self):
        import matplotlib as mpl
        import seaborn as sns

        mpl.use("TkAgg")
        sns.set_style("whitegrid")

        is_call_option = True
        max_step = 4
        _dt = 1
        simplified = True
        r, mu, sigma, risk_lambda, friction = 1e-2, 0, 1e-1, 0, 1e-2

        gaussian_pi = GaussianPolicy(simplified=simplified, alpha=1e-4)
        nnbaseline = NNBaseline(simplified=simplified, alpha=1e-4)
        bs_pi = BSPolicy(is_call=is_call_option)
        bs_baseline = BSBaseline(is_call=is_call_option)

        def build_random(RS, include_terminal: bool):
            time = np.random.randint(low=0, high=max_step + 1 if include_terminal else max_step, size=RS)
            price = np.exp(np.random.randn(RS) * 0.025)
            strike_price = np.exp(np.random.randn(RS) * 0.025)
            state_info_tensor = np.empty((RS, 9))

            state_info_tensor[:, 4] = np.abs(np.random.randn(RS) * 5e-3)  # r
            mu = np.random.randn(RS) * 3e-3
            state_info_tensor[:, 5] = mu  # mu
            sigma = np.abs(np.random.randn(RS)) * 1e-1
            state_info_tensor[:, 6] = sigma  # sigma
            state_info_tensor[:, 7] = np.abs(np.random.randn(RS))  # risk_lambda
            state_info_tensor[:, 8] = np.abs(np.random.randn(RS)) * 0.1  # friction

            state_info_tensor[:, 0] = lib.util.standard_to_normalized_price(
                price,
                mu,  # type:ignore
                sigma,  # type:ignore
                time,
                _dt,
            )  # normal_price
            state_info_tensor[:, 1] = time * _dt  # passed_real_time
            state_info_tensor[:, 2] = (max_step - time) * _dt  # remaining_real_time
            state_info_tensor[:, 3] = lib.util.standard_to_normalized_price(
                strike_price,
                mu,  # type:ignore
                sigma,  # type:ignore
                max_step,
                _dt,
            )  # normal_strike_price

            return torch.tensor(state_info_tensor)

        def build_simplified_random(RS, include_terminal: bool, r, mu, sigma, risk_lambda):
            time = np.random.randint(low=0, high=max_step + 1 if include_terminal else max_step, size=RS)
            price = np.exp(np.random.randn(RS) * 0.25)
            strike_price = np.exp(np.random.randn(RS) * 0.25)
            state_info_tensor = np.empty((RS, 9))

            state_info_tensor[:, 4] = r
            state_info_tensor[:, 5] = mu
            state_info_tensor[:, 6] = sigma
            state_info_tensor[:, 7] = risk_lambda
            state_info_tensor[:, 8] = friction

            state_info_tensor[:, 0] = lib.util.standard_to_normalized_price(price, mu, sigma, time, _dt)  # normal_price
            state_info_tensor[:, 1] = time * _dt  # passed_real_time
            state_info_tensor[:, 2] = (max_step - time) * _dt  # remaining_real_time
            state_info_tensor[:, 3] = lib.util.standard_to_normalized_price(
                strike_price, mu, sigma, max_step, _dt
            )  # normal_strike_price

            return torch.tensor(state_info_tensor)

        RS = 10000
        if simplified:
            source_price = build_simplified_random(RS, True, r, mu, sigma, risk_lambda)
            source_hedge = build_simplified_random(RS, False, r, mu, sigma, risk_lambda)
        else:
            source_price = build_random(RS, include_terminal=True)
            source_hedge = build_random(RS, include_terminal=False)
        target_price = torch.tensor(-bs_baseline.batch_estimate(source_price.numpy()))
        target_hedge = torch.tensor(bs_pi.batch_action(source_hedge))

        label = "train_to_bs_test"
        gaussian_pi.load(f"trained_model/test/{lib.util._prefix(simplified, label)}/policy.pt")
        gaussian_pi.train_based_on(source_hedge, target_hedge, lr=1e-3, itr_max=2e3)
        gaussian_pi.save(f"trained_model/test/{lib.util._prefix(simplified, label)}/policy.pt")

        nnbaseline.load(f"trained_model/test/{lib.util._prefix(simplified, label)}/baseline.pt")
        nnbaseline.train_based_on(source_price, target_price, lr=2e-4, itr_max=2e3)
        nnbaseline.save(f"trained_model/test/{lib.util._prefix(simplified, label)}/baseline.pt")
