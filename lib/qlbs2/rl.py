from math import isnan

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import lib.util

from .env import Baseline, Info, Policy, QLBSEnv


def policy_gradient(
    env: QLBSEnv,
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
        actions, _actions = [], []
        rewards = []
        risks = []

        (state, info), done = env.reset(), False
        info: Info
        states.append(state)
        while not done:
            action, _action = pi.action(state, info, return_pre_action=True)
            action = action.cpu().numpy()
            _action = _action.cpu().numpy()
            state, reward, done, additional = env.step(action, pi)

            if any(isnan(r) for r in reward):
                raise ValueError("NaN encountered!")

            states.append(state)
            actions.append(action)
            _actions.append(_action)
            rewards.append(reward)
            risks.append(additional["risk"])

        T = len(actions)
        G_tmp = 0
        Gs_rev = []
        for t in range(T - 1, -1, -1):
            G_tmp = G_tmp * env.gamma + rewards[t]
            Gs_rev.append(G_tmp)
        Gs = Gs_rev[::-1]

        for t in range(T):
            delta = Gs[t] - V(states[t], info).cpu().numpy()
            if not V_frozen:
                V_loss = V.update(Gs[t], states[t], info)
                # print(f"{V_loss=}")
            if not pi_frozen:
                pi_loss = pi.update(delta * np.power(env.gamma, t), _actions[t], states[t], info)  # type: ignore
                # print(f"{pi_loss=}")

        discount = np.power(env.gamma, np.arange(len(rewards)))  # type: ignore
        t0_return = np.sum(np.array(rewards) * discount[:, None], axis=0)
        t0_risk = env.info.risk_lambdas * np.sum(np.array(risks) * discount[:, None], axis=0) * info._dt

        collector.append(t_return=t0_return.mean(), t_base_return=(t0_return + t0_risk).mean())
        collector.write(writer)

        # writer.add_scalar("info.strike_price", env.info.strike_price, e + 1)  # type: ignore
        # writer.add_scalar("info.mu", env.info.mu, e + 1)  # type: ignore
        # writer.add_scalar("info.sigma", env.info.sigma, e + 1)  # type: ignore
        writer.add_scalar("info.r", env.info.r, e + 1)  # type: ignore
        # writer.add_scalar("info.risk_lambda", env.info.risk_lambda, e + 1)  # type: ignore
        # writer.add_scalar("info.friction", env.info.friction, e + 1)  # type: ignore
        # writer.add_scalar("initial_spot", env.initial_spot, e + 1)  # type: ignore
        writer.add_scalar("pi_loss", pi_loss if not pi_frozen else np.nan, e + 1)  # type: ignore
        writer.add_scalar("V_loss", V_loss if not V_frozen else np.nan, e + 1)  # type: ignore

        print(
            f"Episode {e + 1}/{episode_n}: "
            f"V_loss={V_loss if not V_frozen else 'frozen'}, "
            f"pi_loss={pi_loss if not pi_frozen else 'frozen'}"
        )

        if (e + 1) % 100 == 0 and tensorboard_label:
            ax.cla()
            collector.plot(ax)
            ax.set(ylabel="return (=negative option price)")
            writer.add_figure("combined", fig, e + 1)

    return collector
