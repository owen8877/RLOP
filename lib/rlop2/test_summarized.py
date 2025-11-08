from __future__ import annotations

import math
from unittest import TestCase
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Try SciPy for calibration; fall back to coarse grid if missing
try:
    from scipy.optimize import minimize

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ----------------------------
# Black–76 pricing & IV inversion
# ----------------------------


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def b76_price_call(F: float, K: float, tau: float, r: float, sigma: float) -> float:
    """Black–76 call price (also BS if F, DF inferred via parity)."""
    if tau <= 0 or K <= 0 or F <= 0:
        DF = math.exp(-r * max(tau, 0.0))
        return DF * max(F - K, 0.0)
    if sigma <= 0:
        DF = math.exp(-r * tau)
        return DF * max(F - K, 0.0)
    v = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / v
    d2 = d1 - v
    DF = math.exp(-r * tau)
    return DF * (F * _norm_cdf(d1) - K * _norm_cdf(d2))


def b76_iv_from_price(
    target: float,
    F: float,
    K: float,
    tau: float,
    r: float,
    tol: float = 1e-8,
    max_iter: int = 100,
    lo: float = 1e-4,
    hi: float = 5.0,
) -> Optional[float]:
    """Robust bisection with bracket expansion; clamps target to intrinsic."""
    if not (F > 0 and K > 0 and tau >= 0 and math.isfinite(target)):
        return None
    DF = math.exp(-r * tau)
    intrinsic = DF * max(F - K, 0.0)
    t = max(target, intrinsic)

    def f(sig: float) -> float:
        return b76_price_call(F, K, tau, r, sig) - t

    a, b = lo, hi
    fa, fb = f(a), f(b)
    k = 0
    while fa * fb > 0 and k < 25:
        b *= 1.5
        fb = f(b)
        k += 1
        if b > 50:
            break
    if fa * fb > 0:
        return None

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < tol or 0.5 * (b - a) < 1e-8:
            return c
        if fa * fc <= 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


# ----------------------------
# Put–call parity: infer F and DF
# ----------------------------


def parity_infer_F_DF(df_pairs: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """
    Infer forward F and discount DF from (C - P) vs K.
    y = C - P = DF*(F - K) = a + b*(-K)  ->  b = DF,  F = a/DF.
    """
    y = (df_pairs["C_mid"].values - df_pairs["P_mid"].values).astype(float)
    X = df_pairs["strike"].values.astype(float)
    if len(y) < 2:
        # fallback with single pair
        K0 = float(df_pairs["strike"].iloc[0])
        DF = 1.0
        F = float(y[0] + K0)
        return F, DF
    Xmat = np.column_stack([np.ones_like(X), -X])
    a, b = np.linalg.lstsq(Xmat, y, rcond=None)[0]
    DF = float(b)
    if not np.isfinite(DF) or DF <= 0:
        return None
    F = float(a / DF)
    if not np.isfinite(F) or F <= 0:
        return None
    return F, DF


# ----------------------------
# Maturity bucketing & σ fit
# ----------------------------


def assign_bucket(tau_years: float, centers_days: List[int] = [14, 28, 56]) -> str:
    """Map τ (years) to nearest maturity center in days: '14d'/'28d'/'56d'."""
    days = tau_years * 365.0
    idx = int(np.argmin([abs(days - c) for c in centers_days]))
    return f"{centers_days[idx]}d"


def fit_sigma_bucket(df_bucket: pd.DataFrame) -> float:
    """
    1-parameter BS/B76 baseline: choose σ minimizing price SSE.
    Coarse grid (geomspace) + golden-section refinement.
    """
    if len(df_bucket) == 0:
        return np.nan

    def sse(sig: float) -> float:
        if sig <= 0:
            return 1e18
        prices = df_bucket.apply(lambda r: b76_price_call(r["F"], r["strike"], r["tau"], r["r"], sig), axis=1).values
        err = prices - df_bucket["C_mid"].values
        return float(np.dot(err, err))

    grid = np.geomspace(0.05, 2.0, 40)
    best = min(grid, key=sse)
    a = max(best / 3, 1e-4)
    b = min(best * 3, 5.0)
    phi = (1 + np.sqrt(5)) / 2
    invphi = 1 / phi
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    fc, fd = sse(c), sse(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi
            fc = sse(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi
            fd = sse(d)
    return float((a + b) / 2)


# ============================================================
# Merton Jump–Diffusion (JD)
# ============================================================


def merton_price_call_b76(
    F: float,
    K: float,
    tau: float,
    r: float,
    sigma: float,
    lam: float,
    muJ: float,
    deltaJ: float,
    eps_tail: float = 1e-8,
    n_max: int = 80,
) -> float:
    """
    Merton JD via Poisson mixture of B76 prices.
    Compensation k = E[e^Y]-1 = exp(muJ + 0.5*deltaJ^2) - 1
    Use F_adj = F * exp(-lam * k * tau).
    For n jumps: F_n = F_adj * exp(n*muJ), sigma_n^2 = sigma^2 + n*deltaJ^2 / tau.
    """
    if tau <= 0:
        DF = math.exp(-r * max(tau, 0.0))
        return DF * max(F - K, 0.0)

    k = math.exp(muJ + 0.5 * deltaJ * deltaJ) - 1.0
    F_adj = F * math.exp(-lam * k * tau)
    L = lam * tau

    p = math.exp(-L)  # p0
    price = p * b76_price_call(F_adj, K, tau, r, sigma)
    cum = p
    n = 0
    while cum < 1 - eps_tail and n < n_max:
        n += 1
        p = p * (L / n)  # p_n
        sigma_n = math.sqrt(sigma * sigma + (n * deltaJ * deltaJ) / max(tau, 1e-12))
        F_n = F_adj * math.exp(n * muJ)
        price += p * b76_price_call(F_n, K, tau, r, sigma_n)
        cum += p
    return float(price)


# ============================================================
# Merton Jump–Diffusion (JD)
# ============================================================


def merton_price_call_b76(
    F: float,
    K: float,
    tau: float,
    r: float,
    sigma: float,
    lam: float,
    muJ: float,
    deltaJ: float,
    eps_tail: float = 1e-8,
    n_max: int = 80,
) -> float:
    """
    Merton JD via Poisson mixture of B76 prices.
    Compensation k = E[e^Y]-1 = exp(muJ + 0.5*deltaJ^2) - 1
    Use F_adj = F * exp(-lam * k * tau).
    For n jumps: F_n = F_adj * exp(n*muJ), sigma_n^2 = sigma^2 + n*deltaJ^2 / tau.
    """
    if tau <= 0:
        DF = math.exp(-r * max(tau, 0.0))
        return DF * max(F - K, 0.0)

    k = math.exp(muJ + 0.5 * deltaJ * deltaJ) - 1.0
    F_adj = F * math.exp(-lam * k * tau)
    L = lam * tau

    p = math.exp(-L)  # p0
    price = p * b76_price_call(F_adj, K, tau, r, sigma)
    cum = p
    n = 0
    while cum < 1 - eps_tail and n < n_max:
        n += 1
        p = p * (L / n)  # p_n
        sigma_n = math.sqrt(sigma * sigma + (n * deltaJ * deltaJ) / max(tau, 1e-12))
        F_n = F_adj * math.exp(n * muJ)
        price += p * b76_price_call(F_n, K, tau, r, sigma_n)
        cum += p
    return float(price)


# ============================================================
# Heston Stochastic Volatility (SV)
# ============================================================


def _heston_cf(
    u: np.ndarray, F: float, tau: float, kappa: float, theta: float, sigma_v: float, rho: float, v0: float
) -> np.ndarray:
    """
    Heston characteristic function φ(u) for log-price under forward measure
    (set S0 = F and q = r). Implements the "little Heston trap" form.
    Vectorized over u (real or complex).
    """
    x = math.log(F)
    iu = 1j * u
    d = np.sqrt((rho * sigma_v * iu - kappa) ** 2 + sigma_v**2 * (iu + u * u))
    g = (kappa - rho * sigma_v * iu - d) / (kappa - rho * sigma_v * iu + d)
    exp_dt = np.exp(-d * tau)
    C = (kappa * theta / (sigma_v**2)) * (
        (kappa - rho * sigma_v * iu - d) * tau - 2.0 * np.log((1 - g * exp_dt) / (1 - g))
    )
    D = ((kappa - rho * sigma_v * iu - d) / (sigma_v**2)) * ((1 - exp_dt) / (1 - g * exp_dt))
    return np.exp(C + D * v0 + iu * x)


def _simpson_integral(fx: np.ndarray, dx: float) -> float:
    """Simpson’s rule; falls back to trapz if <3 points. Ensures odd #points."""
    n = len(fx)
    if n < 3:
        return float(np.trapz(fx, dx=dx))
    if n % 2 == 0:
        fx = fx[:-1]
        n -= 1
    S = fx[0] + fx[-1] + 4.0 * fx[1:-1:2].sum() + 2.0 * fx[2:-2:2].sum()
    return float((dx / 3.0) * S)


def _heston_prob(
    F: float, K: float, tau: float, params: Dict[str, float], j: int, u_max: float = 100.0, n_points: int = 501
) -> float:
    """
    Risk-neutral probabilities P1 (j=1) and P2 (j=2):
      P2 = 1/2 + 1/π ∫_0^∞ Re[ e^{-i u ln K} φ(u) / (i u) ] du
      P1 = 1/2 + 1/π ∫_0^∞ Re[ e^{-i u ln K} φ(u - i) / (i u * φ(-i)) ] du
    """
    kappa, theta, sigma_v, rho, v0 = params["kappa"], params["theta"], params["sigma_v"], params["rho"], params["v0"]
    lnK = math.log(K)
    u = np.linspace(1e-6, u_max, n_points)
    du = u[1] - u[0]

    if j == 2:
        phi = _heston_cf(u, F, tau, kappa, theta, sigma_v, rho, v0)
        integrand = np.real(np.exp(-1j * u * lnK) * phi / (1j * u))
    else:
        phi_shift = _heston_cf(u - 1j, F, tau, kappa, theta, sigma_v, rho, v0)
        phi_mi = _heston_cf(np.array([-1j]), F, tau, kappa, theta, sigma_v, rho, v0)[0]
        integrand = np.real(np.exp(-1j * u * lnK) * (phi_shift / (1j * u * phi_mi)))

    integral = _simpson_integral(integrand, du)
    return float(0.5 + (1.0 / math.pi) * integral)


def heston_price_call(
    F: float, K: float, tau: float, r: float, params: Dict[str, float], u_max: float = 100.0, n_points: int = 501
) -> float:
    """Heston price via P1/P2 under forward measure."""
    DF = math.exp(-r * tau)
    P1 = _heston_prob(F, K, tau, params, j=1, u_max=u_max, n_points=n_points)
    P2 = _heston_prob(F, K, tau, params, j=2, u_max=u_max, n_points=n_points)
    return DF * (F * P1 - K * P2)


# ============================================================
# Data prep: one day & symbol -> calls with market IVs
# ============================================================


def prepare_calls_one_day_symbol(
    df_day_symbol: pd.DataFrame, min_parity_pairs: int = 2, tau_floor_days: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input (one trading day × one symbol quotes) →
      calls_df with: F, DF, r, tau, bucket, market B76 IV (calls only)
      parity_df with per-expiry inferred (F, DF, r).
    """
    df = df_day_symbol.copy()
    # normalize schema (accept 'act_symbol' or 'symbol')
    rename_map = {
        "date": "date",
        "expiration": "expiration",
        "strike": "strike",
        "call_put": "cp",
        "bid": "bid",
        "ask": "ask",
    }
    if "act_symbol" in df.columns:
        rename_map["act_symbol"] = "symbol"
    elif "symbol" in df.columns:
        rename_map["symbol"] = "symbol"
    else:
        raise ValueError("Expected 'act_symbol' or 'symbol' column.")
    df = df.rename(columns=rename_map)

    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["tau"] = (df["expiration"] - df["date"]).dt.days / 365.0

    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df["mid"] = (df["bid"].clip(lower=0) + df["ask"].clip(lower=0)) / 2.0

    # cleaning & optional τ floor
    df = df[df["mid"].notna() & (df["mid"] > 0) & (df["ask"] >= df["bid"]) & (df["bid"] >= 0) & (df["tau"] > 0)]
    if tau_floor_days and tau_floor_days > 0:
        df = df[df["tau"] * 365.0 >= tau_floor_days]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # pair calls & puts
    pvt = df.pivot_table(
        index=["date", "symbol", "expiration", "tau", "strike"],
        columns="cp",
        values="mid",
        aggfunc="first",
    ).reset_index()
    pvt = pvt.rename(columns={"Call": "C_mid", "Put": "P_mid"})
    pairs = pvt.dropna(subset=["C_mid", "P_mid"])
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()

    ##########################################################################
    # NOTE: add this hack to inject spot and SOFR_Rate into pvt
    ##########################################################################
    df_extra = df[["date", "symbol", "expiration", "tau", "strike", "spot", "SOFR_Rate"]].drop_duplicates()
    pvt = pvt.merge(df_extra, on=["date", "symbol", "expiration", "tau", "strike"], how="left")
    ##########################################################################

    # infer F, DF, r per expiry
    recs, parity_rows = [], []
    for (date, sym, exp), g in pairs.groupby(["date", "symbol", "expiration"]):
        if len(g) < min_parity_pairs:
            continue
        tau = float(g["tau"].iloc[0])
        res = parity_infer_F_DF(g[["strike", "C_mid", "P_mid"]])
        if res is None:
            continue
        F, DF = res
        r = -math.log(max(DF, 1e-12)) / max(tau, 1e-12)

        gg = pvt[(pvt["date"] == date) & (pvt["symbol"] == sym) & (pvt["expiration"] == exp)].copy()
        gg["F"], gg["DF"], gg["r"] = F, DF, r

        ##########################################################################
        # NOTE: hacking r and F
        ##########################################################################
        gg["F"] = gg["spot"]
        gg["r"] = gg["SOFR_Rate"] / 100
        recs.append(gg)
        parity_rows.append(
            {
                "date": date,
                "symbol": sym,
                "expiration": exp,
                "tau": tau,
                # "F": F,
                "DF": DF,
                # "r": r,
                "n_pairs": len(g),
                # "spot": gg["spot"].iloc[0],
                # "SOFR_Rate": gg["SOFR_Rate"].iloc[0],
                "F": gg["spot"].iloc[0],
                "r": gg["SOFR_Rate"].iloc[0] / 100,
            }
        )
        ##########################################################################

    if not recs:
        return pd.DataFrame(), pd.DataFrame()

    calls = pd.concat(recs, ignore_index=True)
    parity_df = pd.DataFrame(parity_rows)

    # market B76 IVs (calls only)
    calls = calls[calls["C_mid"].notna()].copy()
    calls["sigma_mkt_b76"] = calls.apply(
        lambda r: b76_iv_from_price(r["C_mid"], r["F"], r["strike"], r["tau"], r["r"]), axis=1
    )
    calls = calls.dropna(subset=["sigma_mkt_b76"]).copy()
    if calls.empty:
        return pd.DataFrame(), parity_df

    calls["moneyness_F"] = calls["strike"] / calls["F"]
    # bucket default; caller may overwrite with custom centers
    calls["bucket"] = calls["tau"].apply(assign_bucket)
    return calls, parity_df


# ============================================================
# Calibration helpers (JD & Heston)
# ============================================================


def _minimize(func, x0, bounds):
    """SciPy L-BFGS-B; fall back to a small local grid around x0 if SciPy missing."""
    if _HAVE_SCIPY:
        res = minimize(func, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
        return res.x if res.success else x0
    x0 = np.array(x0, dtype=float)
    grids = []
    for (lo, hi), xi in zip(bounds, x0):
        span = hi - lo
        g = np.linspace(max(lo, xi - 0.2 * span), min(hi, xi + 0.2 * span), 7)
        grids.append(g)
    mesh = np.meshgrid(*grids, indexing="ij")
    cand = np.stack([m.ravel() for m in mesh], axis=1)
    vals = np.array([func(p) for p in cand])
    return cand[int(np.argmin(vals))]


def calibrate_jd_bucket(calls_bucket: pd.DataFrame) -> Tuple[Dict[str, float], float]:
    """Fit JD params (sigma, lam, muJ, deltaJ) by price SSE on the bucket cross-section."""
    if calls_bucket.empty:
        return {}, np.inf
    atm_iv = float(np.median(calls_bucket["sigma_mkt_b76"].values))

    def sse_vec(p):
        sigma, lam, muJ, dJ = p
        if sigma <= 0 or lam < 0 or dJ <= 0:
            return 1e18
        prices = np.array(
            [
                merton_price_call_b76(
                    float(r0["F"]), float(r0["strike"]), float(r0["tau"]), float(r0["r"]), sigma, lam, muJ, dJ
                )
                for _, r0 in calls_bucket.iterrows()
            ],
            dtype=float,
        )
        err = prices - calls_bucket["C_mid"].values
        return float(np.dot(err, err))

    x0 = np.array([max(0.02, atm_iv), 0.1, -0.02, 0.10], dtype=float)
    bounds = [(0.01, 3.0), (0.0, 5.0), (-0.5, 0.5), (0.01, 1.0)]
    p = _minimize(sse_vec, x0, bounds)
    params = {"sigma": float(p[0]), "lam": float(p[1]), "muJ": float(p[2]), "deltaJ": float(p[3])}
    sse_final = sse_vec([params["sigma"], params["lam"], params["muJ"], params["deltaJ"]])
    return params, sse_final


def calibrate_heston_bucket(
    calls_bucket: pd.DataFrame, u_max: float = 100.0, n_points: int = 501
) -> Tuple[Dict[str, float], float]:
    """Fit Heston params (kappa, theta, sigma_v, rho, v0) by price SSE on the bucket cross-section."""
    if calls_bucket.empty:
        return {}, np.inf
    atm_iv = float(np.median(calls_bucket["sigma_mkt_b76"].values))

    def sse_vec(p):
        kappa, theta, sigma_v, rho, v0 = p
        if kappa <= 0 or theta <= 0 or sigma_v <= 0 or not (-0.999 <= rho <= 0.0) or v0 <= 0:
            return 1e18
        params = {"kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho, "v0": v0}
        prices = np.array(
            [
                heston_price_call(
                    float(r0["F"]),
                    float(r0["strike"]),
                    float(r0["tau"]),
                    float(r0["r"]),
                    params,
                    u_max=u_max,
                    n_points=n_points,
                )
                for _, r0 in calls_bucket.iterrows()
            ],
            dtype=float,
        )
        err = prices - calls_bucket["C_mid"].values
        return float(np.dot(err, err))

    v0_0 = max(1e-4, atm_iv * atm_iv)
    x0 = np.array([2.0, max(1e-4, v0_0), 0.5, -0.5, v0_0], dtype=float)
    bounds = [(0.05, 10.0), (1e-4, 2.0), (1e-3, 3.0), (-0.999, 0.0), (1e-4, 3.0)]
    p = _minimize(sse_vec, x0, bounds)
    params = {"kappa": float(p[0]), "theta": float(p[1]), "sigma_v": float(p[2]), "rho": float(p[3]), "v0": float(p[4])}
    sse_final = sse_vec([params["kappa"], params["theta"], params["sigma_v"], params["rho"], params["v0"]])
    return params, sse_final


# ============================================================
# IVRMSE bookkeeping
# ============================================================


def _ivrmse_vs_constant_sigma(calls_bucket: pd.DataFrame, sig_hat: float) -> Dict[str, float]:
    """IVRMSE slices when model IV is constant σ̂ across strikes in bucket."""

    def rmse(sub: pd.DataFrame) -> float:
        if len(sub) == 0:
            return np.nan
        return float(np.sqrt(np.mean((sub["sigma_mkt_b76"].values - sig_hat) ** 2)))

    return {
        "whole": rmse(calls_bucket),
        "<1": rmse(calls_bucket[calls_bucket["moneyness_F"] < 1.0]),
        ">1": rmse(calls_bucket[calls_bucket["moneyness_F"] > 1.0]),
        ">1.03": rmse(calls_bucket[calls_bucket["moneyness_F"] > 1.03]),
    }


def _ivrmse_vs_model_prices(calls_bucket: pd.DataFrame, price_fn) -> Dict[str, float]:
    """
    Price with a model → invert to B76 IV → compare to market IVs.
    Keep only rows where inversion succeeds.
    """
    sig_model, keep = [], []
    for _, r0 in calls_bucket.iterrows():
        pm = price_fn(F=float(r0["F"]), K=float(r0["strike"]), tau=float(r0["tau"]), r=float(r0["r"]))
        s = b76_iv_from_price(pm, float(r0["F"]), float(r0["strike"]), float(r0["tau"]), float(r0["r"]))
        if s is not None:
            sig_model.append(s)
            keep.append(True)
        else:
            keep.append(False)
    if not any(keep):
        return {"whole": np.nan, "<1": np.nan, ">1": np.nan, ">1.03": np.nan}
    cb = calls_bucket.loc[np.array(keep, dtype=bool)].copy()
    cb["sigma_model_b76"] = np.array(sig_model, dtype=float)

    def rmse(sub: pd.DataFrame) -> float:
        if len(sub) == 0:
            return np.nan
        return float(np.sqrt(np.mean((sub["sigma_model_b76"].values - sub["sigma_mkt_b76"].values) ** 2)))

    return {
        "whole": rmse(cb),
        "<1": rmse(cb[cb["moneyness_F"] < 1.0]),
        ">1": rmse(cb[cb["moneyness_F"] > 1.0]),
        ">1.03": rmse(cb[cb["moneyness_F"] > 1.03]),
    }


def _pooled_rmse_x1000(block: pd.DataFrame, iv_col: str, weight_col: str = "N") -> float:
    """
    Contract-weighted pooled RMSE across rows in `block`, ignoring NaNs.
    Returns NaN only if no valid (value, weight) pairs remain after masking
    or if total weight is zero.
    """
    import numpy as np

    x = block[iv_col].to_numpy(dtype=float)  # IVRMSE ×1000 (may contain NaN)
    w = block[weight_col].to_numpy(dtype=float)  # weights (contract counts)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)  # keep only valid pairs
    if not m.any():
        return np.nan
    x_raw = x[m] / 1000.0  # back to raw RMSE
    w_use = w[m]
    return float(np.sqrt((w_use * (x_raw**2)).sum() / w_use.sum()) * 1000.0)


# ============================================================
# Main entry: summarize symbol over a period with saving
# ============================================================


def summarize_symbol_period_ivrmse(
    df_all: pd.DataFrame,
    symbol: str = "SPY",
    start_date: str = "2025-04-01",
    end_date: str = "2025-06-30",
    buckets: List[int] = [14, 28, 56],
    min_parity_pairs: int = 4,
    tau_floor_days: int = 3,
    run_bs: bool = True,
    run_jd: bool = True,
    run_heston: bool = True,
    run_RLOP: bool = True,
    show_progress: bool = True,
    print_daily: bool = True,
    out_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    ONE summary for a symbol over a period (per bucket):
      • Primary  = equal-day mean IVRMSE (each day counts once)
      • Secondary= pooled/N-weighted IVRMSE (every contract counts)
    Saves: daily table, equal_day_mean, pooled, and a daily log (one line/day).
    """
    # Output folder & run config
    outp = None
    log_fp = None
    if out_dir is not None:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        log_fp = (outp / "daily_log.txt").open("w", encoding="utf-8")
        (outp / "run_config.txt").write_text(
            f"symbol={symbol}\\nperiod={start_date}..{end_date}\\n"
            f"buckets={buckets}\\nmin_parity_pairs={min_parity_pairs}\\n"
            f"tau_floor_days={tau_floor_days}\\nrun_bs={run_bs} run_jd={run_jd} run_heston={run_heston}\\n",
            encoding="utf-8",
        )

    # Filter to symbol & period
    df = df_all.copy()
    sym_col = "act_symbol" if "act_symbol" in df.columns else "symbol"
    if sym_col not in df.columns:
        if log_fp:
            log_fp.close()
        raise ValueError("Expected 'act_symbol' or 'symbol' in DataFrame.")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    mask = (df[sym_col] == symbol) & (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    df = df.loc[mask].copy()
    if df.empty:
        if log_fp:
            log_fp.close()
        raise ValueError(f"No rows for {symbol} in [{start_date}, {end_date}].")

    # Bucket mapper using requested centers
    def assign_bucket_centers(tau_years: float) -> str:
        days = tau_years * 365.0
        i = int(np.argmin([abs(days - c) for c in buckets]))
        return f"{buckets[i]}d"

    # Iterate by day with progress
    days = sorted(df["date"].unique())
    iterator = days
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(days, desc=f"{symbol} {pd.Timestamp(start_date).date()}→{pd.Timestamp(end_date).date()}")
        except Exception:
            pass

    daily_rows = []

    for day in iterator:
        df_day = df[df["date"] == day]
        calls, _ = prepare_calls_one_day_symbol(
            df_day, min_parity_pairs=min_parity_pairs, tau_floor_days=tau_floor_days
        )
        if calls.empty:
            msg = f"[{pd.Timestamp(day).date()}] no valid pairs/contracts after filters"
            if print_daily:
                print(msg)
            if log_fp:
                print(msg, file=log_fp)
            continue

        calls["bucket"] = calls["tau"].apply(assign_bucket_centers)

        day_rows = []
        for bucket, g in calls.groupby("bucket"):
            row = {"date": pd.Timestamp(day), "bucket": bucket, "N": int(len(g))}

            # BS/B76 baseline (1-σ)
            if run_bs:
                sig_hat = fit_sigma_bucket(g)
                ivs = _ivrmse_vs_constant_sigma(g, sig_hat)
                row.update(
                    {
                        "BS_IVRMSE_x1000_Whole": ivs["whole"] * 1000 if np.isfinite(ivs["whole"]) else np.nan,
                        "BS_IVRMSE_x1000_<1": ivs["<1"] * 1000 if np.isfinite(ivs["<1"]) else np.nan,
                        "BS_IVRMSE_x1000_>1": ivs[">1"] * 1000 if np.isfinite(ivs[">1"]) else np.nan,
                        "BS_IVRMSE_x1000_>1.03": ivs[">1.03"] * 1000 if np.isfinite(ivs[">1.03"]) else np.nan,
                    }
                )

            # JD baseline
            if run_jd:
                jd_params, _ = calibrate_jd_bucket(g)

                def jd_price(F, K, tau, r):
                    return merton_price_call_b76(
                        F, K, tau, r, jd_params["sigma"], jd_params["lam"], jd_params["muJ"], jd_params["deltaJ"]
                    )

                ivs = _ivrmse_vs_model_prices(g, jd_price)
                row.update(
                    {
                        "JD_IVRMSE_x1000_Whole": ivs["whole"] * 1000 if np.isfinite(ivs["whole"]) else np.nan,
                        "JD_IVRMSE_x1000_<1": ivs["<1"] * 1000 if np.isfinite(ivs["<1"]) else np.nan,
                        "JD_IVRMSE_x1000_>1": ivs[">1"] * 1000 if np.isfinite(ivs[">1"]) else np.nan,
                        "JD_IVRMSE_x1000_>1.03": ivs[">1.03"] * 1000 if np.isfinite(ivs[">1.03"]) else np.nan,
                    }
                )

            # Heston baseline
            if run_heston:
                h_params, _ = calibrate_heston_bucket(g, u_max=100.0, n_points=501)  # stable Simpson settings

                def h_price(F, K, tau, r):
                    return heston_price_call(F, K, tau, r, h_params, u_max=100.0, n_points=501)

                ivs = _ivrmse_vs_model_prices(g, h_price)
                row.update(
                    {
                        "Heston_IVRMSE_x1000_Whole": ivs["whole"] * 1000 if np.isfinite(ivs["whole"]) else np.nan,
                        "Heston_IVRMSE_x1000_<1": ivs["<1"] * 1000 if np.isfinite(ivs["<1"]) else np.nan,
                        "Heston_IVRMSE_x1000_>1": ivs[">1"] * 1000 if np.isfinite(ivs[">1"]) else np.nan,
                        "Heston_IVRMSE_x1000_>1.03": ivs[">1.03"] * 1000 if np.isfinite(ivs[">1.03"]) else np.nan,
                    }
                )

            # RLOP model
            ##########################################################################
            # NOTE: this part is the new RLOP model
            ##########################################################################
            if run_RLOP:
                from .test_trained_model import RLOPModel

                Rmodel = RLOPModel(
                    is_call_option=True,
                    checkpoint="trained_model/testr9/policy_1.pt",
                    anchor_T=28 / 252,
                )
                # spot = (g["F"] * np.exp(-g["r"] * g["tau"])).iloc[0]
                spot = g["F"].iloc[0]
                time_to_expiries = g["tau"].to_numpy()
                strikes = g["strike"].to_numpy()
                r = g["r"].iloc[0]
                risk_lambda = 0.1
                friction = 4e-3
                observed_prices = g["C_mid"].to_numpy()

                moneyness = np.log(g["strike"] / g["F"]) / (np.sqrt(g["tau"]) * 0.2)
                inv_price = 1 / np.power(np.clip(observed_prices, 1.0, None), 1.0)

                try:
                    result = Rmodel.fit(
                        spot=spot,
                        time_to_expiries=time_to_expiries,
                        strikes=strikes,
                        r=r,
                        risk_lambda=risk_lambda,
                        friction=friction,
                        observed_prices=observed_prices,
                        ##########################################################################
                        # NOTE: different combinations of weights, can try which is best
                        ##########################################################################
                        # weights=inv_price * np.exp(-(moneyness.to_numpy() ** 2) * 0.5),
                        weights=inv_price,
                        # weights=np.exp(-(moneyness.to_numpy() ** 2) * 0.5),
                        ##########################################################################
                        sigma_guess=0.3,
                        mu_guess=0,
                        n_epochs=2000,
                    )
                    # import pdb
                    # pdb.set_trace()
                    # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                    # for tau, _df in g.groupby("tau"):
                    #     print(tau)
                    #     axs[0].plot(_df["strike"], _df["C_mid"], label=f"Observed {tau}")
                    #     _indices = np.where(np.isclose(time_to_expiries, tau))[0]
                    #     axs[0].plot(
                    #         _df["strike"], result.estimated_prices[_indices], label=f"Model {tau}", linestyle="--"
                    #     )
                    # plt.legend()
                    # plt.show()
                    # if result.sigma > 3:
                    #     import pdb
                    #     pdb.set_trace()

                    def rlop_price(F, K, tau, r):
                        _result = Rmodel.predict(
                            spot=F,
                            time_to_expiries=np.array([tau]),
                            strikes=np.array([K]),
                            r=r,
                            risk_lambda=risk_lambda,
                            friction=friction,
                            sigma_fit=result.sigma,
                            mu_fit=result.mu,
                        )
                        return _result.estimated_prices[0]

                    ivs = _ivrmse_vs_model_prices(g, rlop_price)
                except KeyboardInterrupt:
                    ivs = {"whole": np.nan, "<1": np.nan, ">1": np.nan, ">1.03": np.nan}
                row.update(
                    {
                        "RLOP_IVRMSE_x1000_Whole": ivs["whole"] * 1000 if np.isfinite(ivs["whole"]) else np.nan,
                        "RLOP_IVRMSE_x1000_<1": ivs["<1"] * 1000 if np.isfinite(ivs["<1"]) else np.nan,
                        "RLOP_IVRMSE_x1000_>1": ivs[">1"] * 1000 if np.isfinite(ivs[">1"]) else np.nan,
                        "RLOP_IVRMSE_x1000_>1.03": ivs[">1.03"] * 1000 if np.isfinite(ivs[">1.03"]) else np.nan,
                    }
                )

                ##########################################################################

            day_rows.append(row)

        if not day_rows:
            msg = f"[{pd.Timestamp(day).date()}] no valid buckets"
            if print_daily:
                print(msg)
            if log_fp:
                print(msg, file=log_fp)
            continue

        daily_rows.extend(day_rows)

        # One compact line per day — pooled across buckets (Whole slice only)
        msg_parts = []
        day_df = pd.DataFrame(day_rows)
        for model, col in [
            ("BS", "BS_IVRMSE_x1000_Whole"),
            ("JD", "JD_IVRMSE_x1000_Whole"),
            ("Heston", "Heston_IVRMSE_x1000_Whole"),
            ("RLOP", "RLOP_IVRMSE_x1000_Whole"),
        ]:
            if col in day_df.columns:
                pooled = _pooled_rmse_x1000(day_df[["N", col]].rename(columns={col: "val"}), "val")
                msg_parts.append(f"{model}={round(float(pooled), 1) if pd.notna(pooled) else 'NA'}")
        msg = f"[{pd.Timestamp(day).date()}] pooled Whole x1000: " + ", ".join(msg_parts)
        if print_daily:
            print(msg)
        if log_fp:
            print(msg, file=log_fp)

    if log_fp:
        log_fp.close()

    if not daily_rows:
        raise RuntimeError("No valid (day,bucket) rows — check filters or min_parity_pairs/tau_floor_days.")

    daily = pd.DataFrame(daily_rows)

    # ---- Primary: equal-day mean per bucket (each day counts once) ----
    iv_cols = [
        c
        for c in daily.columns
        if c.endswith("_IVRMSE_x1000_Whole")
        or c.endswith("_IVRMSE_x1000_<1")
        or c.endswith("_IVRMSE_x1000_>1")
        or c.endswith("_IVRMSE_x1000_>1.03")
    ]
    equal_day_mean = daily.groupby("bucket", as_index=False)[iv_cols].mean()
    # Coverage diagnostics
    days_used = daily.groupby("bucket")["date"].nunique().rename("days_used").reset_index()
    N_total = daily.groupby("bucket")["N"].sum().rename("N_total").reset_index()
    equal_day_mean = equal_day_mean.merge(days_used, on="bucket").merge(N_total, on="bucket").sort_values("bucket")

    # ---- Secondary: pooled/N-weighted per bucket ----
    pooled_rows = []
    for b, g in daily.groupby("bucket"):
        row = {"bucket": b, "days_used": int(g["date"].nunique()), "N_total": int(g["N"].sum())}
        for col in iv_cols:
            row[col] = _pooled_rmse_x1000(g[["N", col]].rename(columns={col: "val"}), "val")
        pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows).sort_values("bucket")

    # Save outputs
    if outp is not None:
        daily.to_csv(outp / "daily_ivrmse.csv", index=False)
        equal_day_mean.to_csv(outp / "equal_day_mean.csv", index=False)
        pooled.to_csv(outp / "pooled.csv", index=False)

    return {"daily": daily, "equal_day_mean": equal_day_mean, "pooled": pooled}


# ============================================================
# Publication-style table (CSV + Markdown with bold minima)
# ============================================================


def make_publication_table(
    res: Dict[str, pd.DataFrame],
    symbol: str = "SPY",
    measure: str = "equal",  # "equal" (primary) or "pooled" (secondary)
    buckets: List[int] = (14, 28, 56),
    decimals: int = 2,
    out_dir: Optional[str] = None,
    basename: str = "table_ivrmse",
) -> pd.DataFrame:
    """
    Build a paper-style table matching the reference format:
      Sections: Whole sample / Moneyness <1 / >1 / >1.03
      Rows: one per bucket like 'SPY (τ=14d)'
      Columns: BS, JD, SV (Heston)
    Saves CSV + Markdown (bolds the row minimum) if out_dir is set.
    """
    if measure not in ("equal", "pooled"):
        raise ValueError("measure must be 'equal' or 'pooled'.")

    df_src = res["equal_day_mean"] if measure == "equal" else res["pooled"]
    if df_src is None or df_src.empty:
        raise ValueError("Source summary is empty.")

    present = set(df_src.columns)
    models = [
        m
        for m, prefix in [("BS", "BS"), ("JD", "JD"), ("SV", "Heston"), ("RLOP", "RLOP")]
        if any(col.startswith(f"{prefix}_IVRMSE_x1000") for col in present)
    ]
    model_to_prefix = {"BS": "BS", "JD": "JD", "SV": "Heston", "RLOP": "RLOP"}

    sections = [("Whole sample", "Whole"), ("Moneyness <1", "<1"), ("Moneyness >1", ">1"), ("Moneyness >1.03", ">1.03")]
    bucket_labels = [f"{d}d" for d in buckets]

    rows = []
    for section_name, suffix in sections:
        for d in bucket_labels:
            row = {"Moneyness": section_name, "Asset": f"{symbol} (τ={d})"}
            sub = df_src[df_src["bucket"] == d]
            if sub.empty:
                for m in models:
                    row[m] = np.nan
            else:
                for m in models:
                    prefix = model_to_prefix[m]
                    col = f"{prefix}_IVRMSE_x1000_{suffix}"
                    row[m] = float(sub[col].iloc[0]) if col in sub.columns and pd.notna(sub[col].iloc[0]) else np.nan
            rows.append(row)

    table = pd.DataFrame(rows)
    for m in models:
        table[m] = table[m].round(decimals)

    if out_dir:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        csv_path = outp / f"{basename}_{measure}.csv"
        table.to_csv(csv_path, index=False)

        # Markdown with bold minimum per row
        md_rows = []
        header = ["Moneyness", "Asset"] + models
        md_rows.append("| " + " | ".join(header) + " |")
        md_rows.append("| " + " | ".join(["---"] * len(header)) + " |")
        for _, r in table.iterrows():
            vals = [r[m] for m in models]
            not_nan = [i for i, v in enumerate(vals) if pd.notna(v)]
            best_idx = None
            if not_nan:
                best_idx = min(not_nan, key=lambda i: vals[i])
            cells = [str(r["Moneyness"]), str(r["Asset"])]
            for i, m in enumerate(models):
                v = r[m]
                if pd.isna(v):
                    cells.append("")
                else:
                    s = f"{v:.{decimals}f}"
                    cells.append(f"**{s}**" if i == best_idx else s)
            md_rows.append("| " + " | ".join(cells) + " |")
        md_text = "\\n".join(md_rows)
        md_path = outp / f"{basename}_{measure}.md"
        md_path.write_text(md_text, encoding="utf-8")
        print(f"Saved: {csv_path}")
        print(f"Saved: {md_path}")

    return table


class FullTest(TestCase):
    def test_SPY(self):
        df = pd.read_csv("data/SPY Options 2025.csv")
        ##########################################################################
        # NOTE: add this hack to inject SPY price
        ##########################################################################
        df_spot_sofr = pd.read_csv("data/spy_eod_and_box_rate.csv")
        df = df.merge(
            df_spot_sofr.rename(columns={"Date": "date", "SPY": "spot", "SOFR_Rate": "SOFR_Rate"}),
            on="date",
            how="left",
        )

        res = summarize_symbol_period_ivrmse(
            df_all=df,
            symbol="SPY",
            start_date="2025-04-01",
            ##########################################################################
            # NOTE: end date has been changed to reduce workload
            ##########################################################################
            end_date="2025-04-05",
            buckets=[14, 28, 56],
            min_parity_pairs=4,
            tau_floor_days=3,
            run_bs=True,
            run_jd=True,
            run_heston=False,
            run_RLOP=True,
            show_progress=True,
            print_daily=True,
            out_dir="SPY_25Q2_baseline",  # outputs saved here
        )

        # PRIMARY (paper): equal-day mean table
        tbl_equal = make_publication_table(
            res,
            symbol="SPY",
            measure="equal",
            buckets=[14, 28, 56],
            decimals=2,
            out_dir="SPY_25Q2_baseline",
            basename="table_ivrmse",
        )

        # SECONDARY (robustness): pooled table
        tbl_pooled = make_publication_table(
            res,
            symbol="SPY",
            measure="pooled",
            buckets=[14, 28, 56],
            decimals=2,
            out_dir="SPY_25Q2_baseline",
            basename="table_ivrmse",
        )

        print(tbl_equal)
        print(tbl_pooled)
