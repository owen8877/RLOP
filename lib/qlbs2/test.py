from dataclasses import replace
from math import isnan
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .bs import BSBaseline, BSPolicy
from .env import EnvSetup, Info, QLBSEnv


class Test(TestCase):
    def _parameters(self):
        T = 1
        N_parallel = 100
        return (
            EnvSetup(
                r=0.02,
                parallel_n=N_parallel,
                mus=np.array([0.05] * N_parallel),
                sigmas=np.array([0.1] * N_parallel),
                risk_lambdas=None,  # type: ignore
                strike_prices=np.array([1] * N_parallel),
                _dt=(_dt := 1 / 12),
                frictions=np.array([0e-3] * N_parallel),
                is_call_option=True,
                max_step=int(np.round(T / _dt)),
                initial_spots=[1] * N_parallel,
                risk_simulation_paths=200,
                mutation=0.5,
            ),
            (risk_lambdas := [5]),
        )

    def test_qlbs_value_function(self):
        """Tests if the QLBS value function (BS baseline) matches the Black-Shorles formula under B-S hedging strategy."""
        setup, risk_lambdas = self._parameters()
        risk_lambda = risk_lambdas[0]
        N_parallel = setup.parallel_n

        pi = BSPolicy(is_call=True)
        V = BSBaseline(is_call=True)

        def sovler(spot: float):
            states = []
            actions = []
            rewards = []
            risks = []

            env = QLBSEnv(
                replace(
                    setup, risk_lambdas=[risk_lambdas[0]] * N_parallel, initial_spots=[spot] * N_parallel, mutation=0.0
                )
            )
            (state, info), done = env.reset(), False
            info: Info
            while not done:
                action = pi.action(state, info)
                state, reward, done, additional = env.step(action, pi)

                if any(isnan(r) for r in reward):
                    raise ValueError("NaN encountered!")

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                risks.append(additional["risk"])

            discount = np.power(env.gamma, np.arange(len(rewards)))
            t0_return = np.sum(np.array(rewards) * discount[:, None], axis=0)
            t0_risk = env.info.risk_lambdas * np.sum(np.array(risks) * discount[:, None], axis=0) * info._dt

            return t0_return, t0_risk

        data = []
        for spot in np.arange(0.1, 2.0, 0.05):
            sit = torch.empty((N_parallel, 9))
            sit[:, 0] = spot  # type: ignore
            sit[:, 2] = 0
            sit[:, 3] = torch.tensor(setup.strike_prices)  # strike
            sit[:, 4] = setup.r  # r
            sit[:, 5] = torch.tensor(setup.mus)  # mu
            sit[:, 6] = torch.tensor(setup.sigmas)  # sigma
            sit[:, 7] = torch.tensor([risk_lambdas[0]] * N_parallel)  # risk_lambda
            sit[:, 8] = torch.tensor(setup.frictions)  # friction
            estimated = V.batch_estimate(sit).numpy()
            simulated, simulated_risk = sovler(spot)  # type: ignore

            for trial in range(N_parallel):
                data += [
                    {
                        "trial": trial,
                        "spot": spot,
                        "estimated": estimated[trial],
                        "simulated_total": -simulated[trial],
                        "simulated_risk": simulated_risk[trial],
                    }
                ]

        dff = pd.DataFrame(data)
        dff["simulated_base"] = dff["simulated_total"] - dff["simulated_risk"]

        df = (
            dff.groupby(["spot"])
            .agg(
                estimated=("estimated", "mean"),
                simulated_base=("simulated_base", "mean"),
                simulated_risk=("simulated_risk", "mean"),
                simulated_total=("simulated_total", "mean"),
                simulated_base_std=("simulated_base", "std"),
                simulated_risk_std=("simulated_risk", "std"),
                simulated_total_std=("simulated_total", "std"),
            )
            .reset_index()
        )

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        df.plot.line(
            x="spot",
            y=["estimated", "simulated_base", "simulated_risk", "simulated_total"],  # type: ignore
            title="BS Value Function: Estimated vs Simulated",
            ax=axs[0],
        )
        df.plot.line(
            x="spot",
            y=["simulated_base_std", "simulated_risk_std", "simulated_total_std"],  # type: ignore
            title="BS Value Function: Simulated Std Dev",
            ax=axs[1],
        )
        plt.show()

    def test_reshape_order(self):
        t, max_step = 6, 9
        N_parallel = 2
        RS = 4
        GBM = np.random.rand(N_parallel, max_step - t + 1, RS)

        sits = []
        for s in np.arange(t, max_step):
            sit = torch.empty((N_parallel, 9, RS))
            sit[:, 0, :] = torch.tensor(GBM[:, s - t, :])  # spot
            sit[:, 1, :] = s + 0.0
            sit[:, 2, :] = 11
            sit[:, 3, :] = 12
            sit[:, 4, :] = 13
            sit[:, 5, :] = 14
            sit[:, 6, :] = 15
            sit[:, 7, :] = 16
            sit[:, 8, :] = 17
            sits.append(sit)

        def calc(tensor: torch.Tensor) -> torch.Tensor:
            return tensor[:, 0] * tensor[:, 1]

        sits_stacked = torch.cat(sits, dim=0)
        sits_reshaped = sits_stacked.permute(0, 2, 1).reshape(-1, 9)
        hedge_long = calc(sits_reshaped).numpy()
        # hedge = np.transpose(hedge_long.reshape(max_step - t, N_parallel, RS), (1, 0, 2))
        hedge = hedge_long.reshape(N_parallel, max_step - t, RS)

        GBM_expected = GBM[:, :-1, :].copy()
        GBM_expected[:, 0, :] *= 6
        GBM_expected[:, 1, :] *= 7
        GBM_expected[:, 2, :] *= 8
        print(hedge)
        print(GBM_expected)
