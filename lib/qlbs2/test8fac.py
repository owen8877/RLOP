import os
from re import I
import time
from dataclasses import replace
from unittest import TestCase

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.optimize import brentq
import joblib
from scipy.stats import norm

from ..util import ensure_dir
from .bs import BSBaseline, BSPolicy
from .env import EnvSetup, QLBSEnv
from .nn_v6 import GaussianPolicy_v6, NNBaseline_v6
from .rl import policy_gradient

exp_name = os.path.basename(__file__)[:-3]


def black_scholes_price(S, K, T, r, sigma, is_call=True):
    """
    Calculate the Black-Scholes price of a European option.
    """
    if T <= 0:
        return np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if is_call:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def calculate_implied_volatility(prices, strikes, S, T, r, is_call=True):
    """
    Calculate the implied volatility given market prices and strike prices.

    :param prices: np.ndarray or list of option market prices
    :param strikes: np.ndarray or list of option strike prices
    :param S: float, current asset spot price
    :param T: float, time to maturity in years
    :param r: float, risk-free interest rate
    :param is_call: bool, True for Call options, False for Put options
    :return: np.ndarray of implied volatilities (returns np.nan for failed conversions)
    """
    prices = np.asarray(prices)
    strikes = np.asarray(strikes)

    # Ensure inputs are the same length
    if prices.shape != strikes.shape:
        raise ValueError("Prices and strikes arrays must have the same shape.")

    implied_vols = np.zeros_like(strikes, dtype=float)

    for i, (price, K) in enumerate(zip(prices, strikes)):
        # Calculate the intrinsic value of the option
        intrinsic_value = max(S - K, 0.0) if is_call else max(K - S, 0.0)

        # If the option price is less than its intrinsic value, implied volatility is undefined
        if price <= intrinsic_value:
            implied_vols[i] = np.nan
            continue

        # Objective function: difference between model price and market price
        def objective(sigma):
            return black_scholes_price(S, K, T, r, sigma, is_call) - price

        try:
            # Brent's method is reliable for 1D root finding.
            # We search for a root between 1e-6 (approx. 0%) and 5.0 (500% volatility).
            iv = brentq(objective, 1e-6, 5.0)
            implied_vols[i] = iv
        except ValueError:
            # brentq raises a ValueError if the root is not bracketed in the given interval
            implied_vols[i] = np.nan

    return implied_vols


def _path(risk_lambda, network_depth: int, exp_name=exp_name):
    plan = f"risk_lambda={risk_lambda:.1e}/network_depth={network_depth:d}"
    return f"trained_model/{exp_name}/{plan}"


def invert_price_suit(
    risk_lambda=0.1,
    network_depth=2,
    replica=1,
    check_point=1,
    *,
    n_optimization: int = 40,
    numeric_only: bool = False,
):
    from lib.qlbs2.nn_v6_grad_fn import GaussianPolicy_v6, NNBaseline_v6

    # setup hyperparameters
    r = 0.04  # interest rate, annualized
    friction = 2e-3  # transaction cost per unit traded
    is_call_option = True  # has to be call option for this test
    # risk_lambda = 0.1  # risk aversion parameter
    # for this time, time to maturity and spot price are fixed, but we can probably find a way to vectorize this
    T = 28 / 252  # time to maturity, in years
    spot = 1.0  # spot price

    initial_sigma_guess = 0.5  # initial guess of volatility
    initial_mu_guess = 0.00  # initial guess of drift; due to theoretic loop holes, call skew options have mu < 0

    def vol_curve_true(strikes):
        return 0.52 - 2 * (sp.special.expit(spot / strikes - 1) - 0.5)

    # synthesize test data
    N = 100
    strikes = np.linspace(0.85 * spot, 1.15 * spot, N)

    sigma_BS = vol_curve_true(strikes)
    bs_baseline = BSBaseline(is_call=is_call_option)
    target_price = bs_baseline.batch_estimate(
        torch.tensor(
            np.column_stack(
                (
                    spot + (strikes * 0),  # spot
                    np.zeros(N),  # passed_real_time
                    np.full(N, T),  # remaining_real_time
                    strikes,  # strike
                    np.full(N, r),  # r
                    np.zeros(N),  # mu
                    sigma_BS,  # sigma
                    np.full(N, risk_lambda),  # risk_lambda
                    np.full(N, friction),  # friction
                )
            )
        ).float()
    )  # type: ignore
    target_price_with_noise = target_price.numpy() + np.random.randn(N) * 2e-2 * np.max(np.abs(target_price.numpy()))
    # load trained models
    path = _path(risk_lambda, network_depth)
    nn_policy = GaussianPolicy_v6(
        is_call_option=is_call_option,
        from_filename=f"{path}/policy_{check_point}_R{replica}.pt",
        network_depth=network_depth,
    )
    nn_baseline = NNBaseline_v6(nn_policy.theta_mu, nn_policy.optimizer)

    # invert prices
    sigma = torch.tensor(initial_sigma_guess, requires_grad=True, dtype=torch.float32)
    mu = torch.tensor(initial_mu_guess, requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.Adam([sigma, mu], lr=0.1)

    _scaled_strikes = torch.tensor(strikes / spot, dtype=torch.float32)
    _scaled_tte = torch.tensor((T + strikes * 0) / T, dtype=torch.float32)
    _scaled_rs = r * _scaled_tte
    _scaled_prices = torch.tensor(target_price_with_noise / spot, dtype=torch.float32)
    _scaled_risk_lambdas = torch.full((N,), risk_lambda, dtype=torch.float32)
    _scaled_frictions = torch.full((N,), friction, dtype=torch.float32)
    weights = np.exp(-(((strikes - spot) / (spot * np.sqrt(T))) ** 2))
    _scaled_weights = torch.tensor(weights * np.power(spot, 2), dtype=torch.float32)

    _zeros, _ones = torch.zeros(N, dtype=torch.float32), torch.ones(N, dtype=torch.float32)

    def get_state_info_tensor(mu, sigma):
        return torch.stack(
            [
                _ones,  # spot
                _zeros,  # passed_real_time
                _ones * T,  # remaining_real_time
                _scaled_strikes,  # strike
                _scaled_rs,  # r
                (mu * _ones),  # mu
                (sigma * _ones),  # sigma
                _scaled_risk_lambdas,  # risk_lambda
                _scaled_frictions,  # friction
            ],
            dim=1,
        )

    def loss():
        state_info_tensor = get_state_info_tensor(mu, sigma)
        estimated_price = -nn_baseline.batch_estimate(state_info_tensor)  # type: ignore
        loss = torch.mean(_scaled_weights * (estimated_price - _scaled_prices) ** 2)
        loss.backward()
        return loss

    for epoch in range(n_optimization):
        optimizer.zero_grad()
        optimizer.step(loss)
        l = loss()
        print(f"Epoch {epoch}: loss={l.item():.6f}, sigma={sigma.item():.6f}, mu={mu.item():.6f}")

    with torch.no_grad():
        state_info_tensor = get_state_info_tensor(mu, sigma)
        estimated_price, latent_vol = nn_baseline.batch_estimate(state_info_tensor, return_latent_vol=True)  # type: ignore

    final_price = -estimated_price.numpy() * spot
    final_vol = latent_vol.numpy() / _scaled_tte.numpy() ** 0.5

    if numeric_only:
        return (
            final_price,
            final_vol,
            target_price.numpy(),
            target_price_with_noise,
            vol_curve_true(strikes),
            strikes,
            weights,
        )

    # plot results
    fig, axs = plt.subplots(2, 1, figsize=(7, 5))

    axs[0].plot(strikes, target_price_with_noise, label="Target Price with Noise", color="black", linestyle="--")
    axs[0].plot(strikes, final_price, label="Estimated Price", color="blue", linestyle="-")
    axs[0].set_ylabel("Option Price")
    axs[0].legend(loc="best")

    axs[1].plot(strikes, vol_curve_true(strikes), label="True Vol Curve", color="green", linestyle="--")
    axs[1].plot(strikes, final_vol, label="Inverted Latent Vol", color="red", linestyle="-")

    # clean_inversion = calculate_implied_volatility(target_price, strikes, spot, T, r)
    # direct_inversion = calculate_implied_volatility(target_price_with_noise, strikes, spot, T, r)
    # nn_inversion = calculate_implied_volatility(final_price, strikes, spot, T, r)

    # axs[1].plot(strikes, clean_inversion, label="Clean Inversion", color="red", linestyle="--")
    # axs[1].plot(strikes, direct_inversion, label="Direct Inversion", color="orange", linestyle="-")
    # axs[1].plot(strikes, nn_inversion, label="NN Inversion", color="purple", linestyle="-")

    axs[1].set_ylabel("Volatility")
    axs[1].legend(loc="best")

    plt.xlabel("Strike Price")
    plt.tight_layout()
    plt.show()


class Training(TestCase):
    def _parameters(self):
        T = 28 / 252
        parallel_n = 2500
        return EnvSetup(
            parallel_n=parallel_n,
            r=0.04,
            mus=np.random.randn(parallel_n) * 0.2,
            sigmas=np.random.rand(parallel_n) * 0.2 + 0.4,
            risk_lambdas=None,  # type: ignore
            strike_prices=np.random.rand(parallel_n) * 0.6 + 0.7,
            _dt=(_dt := 7 / 252),
            frictions=np.random.rand(parallel_n) * 1e-2,
            is_call_option=True,
            max_step=int(np.round(T / _dt)),
            initial_spots=np.random.rand(parallel_n) * 0.6 + 0.7,
            risk_simulation_paths=200,
            mutation=0.0,
        )

    def learn_both_from_0_vol_latent_core(
        self, risk_lambda: float, network_depth: int, replica: int, check_point: int | None, *, episode_n=200
    ):
        setup = self._parameters()
        N_parallel = setup.parallel_n

        path = _path(risk_lambda, network_depth)
        env = QLBSEnv(replace(setup, risk_lambdas=np.array([risk_lambda] * N_parallel)))

        nn_policy = GaussianPolicy_v6(
            lr=1e-4,
            is_call_option=setup.is_call_option,
            from_filename=f"{path}/policy_{check_point}_R{replica}.pt" if check_point else None,
            groups=network_depth,
        )
        nn_baseline = NNBaseline_v6(nn_policy.theta_mu, nn_policy.optimizer)

        start_time = time.time()
        collector = policy_gradient(
            env,
            nn_policy,
            nn_baseline,
            episode_n=episode_n,
            tensorboard_label=f"qlbs2-{exp_name}-{risk_lambda=}-{network_depth=}",
        )

        time_used = time.time() - start_time
        value_t_return = float(collector.ema_dict["t_return"][-1])
        value_t_base_return = float(collector.ema_dict["t_base_return"][-1])

        check_point_save = 1 if check_point is None else check_point + 1
        ensure_dir(path)
        nn_policy.save(f"{path}/policy_{check_point_save}_R{replica}.pt")
        # nn_baseline.save(f"{path}/baseline_{check_point_save}.pt")
        joblib.dump(
            {
                "time_used": time_used,
                "value_t_return": value_t_return,
                "value_t_base_return": value_t_base_return,
            },
            f"{path}/info_{check_point_save}_R{replica}.joblib",
        )

    def test_learn_both_from_0_vol_latent(self):
        check_point = None
        for risk_lambda in 0.01, 0.1, 1.0:
            for network_depth in 2, 3, 4:
                for replica in range(5, 10):
                    print(f"Working on risk_lambda={risk_lambda}, network_depth={network_depth}, replica={replica}...")
                    while True:
                        try:
                            self.learn_both_from_0_vol_latent_core(
                                risk_lambda=risk_lambda,
                                network_depth=network_depth,
                                replica=replica,
                                episode_n=50,
                                check_point=check_point,
                            )
                            break
                        except Exception as e:
                            raise e

    def test_invert_price_set1(self):
        check_point = 1

        results = []

        for risk_lambda in 0.01, 0.1, 1.0:
            for network_depth in 2, 3, 4:
                for replica in range(10):
                    info = joblib.load(f"{_path(risk_lambda, network_depth)}/info_{check_point}_R{replica}.joblib")

                    while True:
                        try:
                            (
                                final_price,
                                final_vol,
                                target_price,
                                target_price_with_noise,
                                vol_curve_true,
                                strikes,
                                weights,
                            ) = invert_price_suit(
                                risk_lambda=risk_lambda,
                                network_depth=network_depth,
                                replica=replica,
                                check_point=check_point,
                                numeric_only=True,
                                n_optimization=10,
                            )  # type: ignore
                            results += [
                                {
                                    "risk_lambda": risk_lambda,
                                    "network_depth": network_depth,
                                    "replica": replica,
                                    "price_error": np.sqrt(np.mean((final_price - target_price) ** 2 * weights)),
                                    "vol_error": np.sqrt(np.mean((final_vol - vol_curve_true) ** 2 * weights)),
                                    **info,
                                }
                            ]
                            print(info)
                            break
                        except ValueError as e:
                            pass

        df = pd.DataFrame(results)
        ensure_dir(dir := "result/qlbs/test8fac")
        df.to_csv(f"{dir}/set1.csv")

        print(
            df.pivot_table(
                index=["risk_lambda"], columns="network_depth", values=["price_error", "vol_error"], aggfunc="mean"
            )
        )
        print(
            df.pivot_table(
                index=["risk_lambda"], columns="network_depth", values=["time_used", "value_t_return"], aggfunc="mean"
            )
        )

    def test_analyze_set1(self):
        df = pd.read_csv("result/qlbs/test8fac/set1.csv")
        output_dir = "result/qlbs/test8fac/set1"
        ensure_dir(output_dir)

        for value_col in ["price_error", "vol_error", "time_used", "value_t_return"]:
            pivot_stats = df.pivot_table(
                index="risk_lambda",
                columns="network_depth",
                values=value_col,
                aggfunc=["mean", "std"],
            )

            pivot_mean = pivot_stats["mean"]
            pivot_std = pivot_stats["std"]

            if value_col in ["price_error", "vol_error"]:
                pivot_mean *= 100
                pivot_std *= 100

            pivot_formatted = pivot_mean.applymap("{:.2f}".format) + r" $\pm$ " + pivot_std.applymap("{:.2f}".format)

            pivot_formatted.to_latex(f"{output_dir}/{value_col}.tex", escape=False)

    def test_analyze_set2(self):
        df = pd.read_csv("result/qlbs/test8fac/set1.csv")
        df_rlop = pd.read_csv("result/rlop/testr9fac/set1.csv")
        output_dir = "result/qlbs/test8fac/set2"
        ensure_dir(output_dir)

        df["Method"] = "QLBS"
        df_rlop["Method"] = "RLOP"

        df_combined = pd.concat([df, df_rlop], ignore_index=True)

        metrics = {"vol_error": "IV RMSE (%)", "time_used": "Time Used (s)"}

        formatted_dfs = []
        for value_col, metric_name in metrics.items():
            pivot_stats = df_combined.pivot_table(
                index="risk_lambda",
                columns=["network_depth", "Method"],
                values=value_col,
                aggfunc=["mean", "std"],
            )

            pivot_mean = pivot_stats["mean"]
            pivot_std = pivot_stats["std"]

            if value_col == "vol_error":
                pivot_mean *= 100
                pivot_std *= 100

            pivot_formatted = pivot_mean.applymap("{:.1f}".format) + r" $\pm$ " + pivot_std.applymap("{:.1f}".format)

            pivot_formatted = pd.concat({metric_name: pivot_formatted}, names=["Metric"])
            formatted_dfs.append(pivot_formatted)

        final_df = pd.concat(formatted_dfs)
        # Reorder columns: network depth 2, 3, 4 top tier, Method QLBS, RLOP second tier
        final_df = final_df.reindex(
            columns=pd.MultiIndex.from_product([[2, 3, 4], ["QLBS", "RLOP"]], names=["Network Depth", "Method"])
        )

        print(final_df)
        final_df.to_latex(f"{output_dir}/combined_table.tex", escape=False)
