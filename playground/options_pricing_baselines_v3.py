from __future__ import annotations

import math
import pickle
from unittest import TestCase
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re
from datetime import datetime

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

from typing import Tuple
import math
import numpy as np
import pandas as pd


def prepare_calls_one_day_symbol(
    df_day_symbol: pd.DataFrame, min_parity_pairs: int = 2, tau_floor_days: int = 0, type: str = "american"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    type='american' : Fast-path. Use provided F/DF/(r); prefer C_eur as market mid if available.
    type='european' : Pair Calls & Puts, infer F/DF via parity (original fallback).
    Returns (calls_df, parity_df).
    """
    # ---- normalize schema (accept act_symbol or symbol; call_put or cp) ----
    df = df_day_symbol.copy()
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
        raise ValueError("Expected an 'act_symbol' or 'symbol' column.")
    df = df.rename(columns=rename_map)

    # ---- basic types/columns ----
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["tau"] = (df["expiration"] - df["date"]).dt.days / 365.0

    # If cp is missing in calls-only data, set it to Call
    if "cp" not in df.columns:
        df["cp"] = "Call"

    # numeric & mid
    if "mid" not in df.columns:
        df["bid"] = pd.to_numeric(df.get("bid", np.nan), errors="coerce")
        df["ask"] = pd.to_numeric(df.get("ask", np.nan), errors="coerce")
        df["mid"] = (df["bid"].clip(lower=0) + df["ask"].clip(lower=0)) / 2.0
    else:
        df["mid"] = pd.to_numeric(df["mid"], errors="coerce")

    # ---- if american fast-path and you have C_eur, use it as market mid ----
    if type.lower() == "american" and "C_eur" in df.columns:
        df["mid"] = pd.to_numeric(df["C_eur"], errors="coerce")

    # ---- cleaning ----
    df = df[df["strike"].notna() & df["mid"].notna() & (df["mid"] > 0) & (df["tau"] > 0)]
    if "ask" in df.columns and "bid" in df.columns:
        df = df[(df["ask"] >= df["bid"]) & (df["bid"] >= 0)]
    if tau_floor_days and tau_floor_days > 0:
        df = df[df["tau"] * 365.0 >= tau_floor_days]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # =========================================================================
    # FAST-PATH for 'american': Calls-only, trust supplied F/DF/(r)
    # =========================================================================
    if type.lower() == "american":
        # Keep Calls only (avoid accidentally treating Puts as Calls)
        df = df[df["cp"].str.upper().eq("CALL")]

        if not {"F", "DF"}.issubset(df.columns):
            raise ValueError("type='american' expects F and DF columns in the dataset.")
        df["F"] = pd.to_numeric(df["F"], errors="coerce")
        df["DF"] = pd.to_numeric(df["DF"], errors="coerce")

        has_r = "r" in df.columns
        if has_r:
            df["r"] = pd.to_numeric(df["r"], errors="coerce")

        # per-expiry header
        hdr = df.groupby(["date", "symbol", "expiration"], as_index=False).agg(
            F=("F", "first"),
            DF=("DF", "first"),
        )

        # tau per (date,exp) for r derivation if needed
        tau_hdr = df.groupby(["date", "symbol", "expiration"], as_index=False).agg(tau=("tau", "first"))
        hdr = hdr.merge(tau_hdr, on=["date", "symbol", "expiration"], how="left")

        if has_r:
            r_hdr = df.groupby(["date", "symbol", "expiration"], as_index=False).agg(r=("r", "first"))
            hdr = hdr.merge(r_hdr, on=["date", "symbol", "expiration"], how="left")
        else:
            hdr["r"] = -np.log(np.clip(hdr["DF"].to_numpy(), 1e-12, None)) / np.maximum(hdr["tau"].to_numpy(), 1e-12)

        # calls table shaped like downstream expects (C_mid etc.)
        calls = df.groupby(["date", "symbol", "expiration", "tau", "strike"], as_index=False).agg(
            C_mid=("mid", "first")
        )
        calls = calls.merge(
            hdr[["date", "symbol", "expiration", "F", "DF", "r"]], on=["date", "symbol", "expiration"], how="left"
        )

        # market B76 IVs (calls only)
        calls["sigma_mkt_b76"] = calls.apply(
            lambda r: b76_iv_from_price(r["C_mid"], r["F"], r["strike"], r["tau"], r["r"]), axis=1
        )
        calls = calls.dropna(subset=["sigma_mkt_b76"])
        if calls.empty:
            return pd.DataFrame(), pd.DataFrame()

        calls["moneyness_F"] = calls["strike"] / calls["F"]
        calls["bucket"] = calls["tau"].apply(assign_bucket)

        # parity_df: simple header + coverage proxy (#strikes)
        n_pairs = df.groupby(["date", "symbol", "expiration"])["strike"].nunique().rename("n_pairs").reset_index()
        parity_df = hdr.merge(n_pairs, on=["date", "symbol", "expiration"], how="left")
        parity_df["n_pairs"] = parity_df["n_pairs"].fillna(0).astype(int)

        return calls, parity_df

    # =========================================================================
    # 'european' path: Calls–Puts pairing + parity inference (original behavior)
    # =========================================================================
    pvt = df.pivot_table(
        index=["date", "symbol", "expiration", "tau", "strike"], columns="cp", values="mid", aggfunc="first"
    ).reset_index()
    pvt = pvt.rename(columns={"Call": "C_mid", "Put": "P_mid"})
    pairs = pvt.dropna(subset=["C_mid", "P_mid"])
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()

    # infer F, DF, r per expiry via parity
    recs, parity_rows = [], []
    for (date, sym, exp), g in pairs.groupby(["date", "symbol", "expiration"], sort=False):
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
        recs.append(gg)
        parity_rows.append(
            {"date": date, "symbol": sym, "expiration": exp, "tau": tau, "F": F, "DF": DF, "r": r, "n_pairs": len(g)}
        )

    if not recs:
        return pd.DataFrame(), pd.DataFrame()

    calls = pd.concat(recs, ignore_index=True)

    # market B76 IVs (calls only)
    calls = calls[calls["C_mid"].notna()].copy()
    calls["sigma_mkt_b76"] = calls.apply(
        lambda r: b76_iv_from_price(r["C_mid"], r["F"], r["strike"], r["tau"], r["r"]), axis=1
    )
    calls = calls.dropna(subset=["sigma_mkt_b76"])
    if calls.empty:
        return pd.DataFrame(), pd.DataFrame()

    calls["moneyness_F"] = calls["strike"] / calls["F"]
    calls["bucket"] = calls["tau"].apply(assign_bucket)

    parity_df = pd.DataFrame(parity_rows)
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


# Data Preprocessing


def adapter_eur_calls_to_summarizer(calls_out: pd.DataFrame) -> pd.DataFrame:
    """
    Input (from preprocess_american_to_european):
      required cols: date, act_symbol, expiration, strike, C_eur, F, DF
    Output for your summarizer (calls only, no parity step needed):
      date, act_symbol, expiration, strike, cp="Call", bid, ask, mid, F, DF
    """
    req = {"date", "act_symbol", "expiration", "strike", "C_eur", "F", "DF"}
    missing = req - set(calls_out.columns)
    if missing:
        raise ValueError(f"calls_out is missing required columns: {sorted(missing)}")

    df = calls_out.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()

    # Build calls-only frame
    calls = df[["date", "act_symbol", "expiration", "strike", "C_eur", "F", "DF"]].copy()
    calls["cp"] = "Call"
    calls["mid"] = pd.to_numeric(calls["C_eur"], errors="coerce")

    # Bid/ask = mid (keeps ask≥bid>0 filters happy)
    calls["bid"] = calls["mid"]
    calls["ask"] = calls["mid"]

    # Final columns (sorted)
    calls = (
        calls[(calls["mid"] > 0) & (calls["ask"] >= calls["bid"])][
            ["date", "act_symbol", "expiration", "strike", "cp", "bid", "ask", "mid", "F", "DF"]
        ]
        .sort_values(["date", "act_symbol", "expiration", "strike"])
        .reset_index(drop=True)
    )
    return calls


def preprocess_deribit(
    df: pd.DataFrame,
    *,
    instr_col: str = "instrument_name",  # e.g. "BTC-26DEC25-144000-C"
    ts_col: str = "creation_timestamp",  # ms since epoch (UTC)
    spot_col: str = "underlying_price",  # USD per BTC
    bid_btc_col: str = "bid_price",  # bid quoted in BTC
    ask_btc_col: str = "ask_price",  # ask quoted in BTC
) -> pd.DataFrame:
    """
    Minimal mapper for your summarizer. Outputs exactly the columns it expects:
      ['date','symbol','act_symbol','expiration','strike','cp','bid','ask','mid']
    - date: UTC date from creation_timestamp
    - symbol/act_symbol: parsed from instrument_name (e.g., "BTC")
    - expiration: parsed from instrument_name (naive UTC datetime)
    - strike: parsed from instrument_name (float)
    - cp: "Call"/"Put"
    - bid/ask/mid: USD premiums = BTC quote × underlying_price
    """

    def _parse(name: str):
        m = re.match(r"^([A-Z]+)-(\d{2}[A-Z]{3}\d{2})-([0-9]+)-(C|P)$", str(name).strip())
        if not m:
            return None
        sym, exp_s, k_s, cp = m.groups()
        try:
            exp_dt = datetime.strptime(exp_s, "%d%b%y")  # naive UTC
            strike = float(k_s)
            cp_full = "Call" if cp == "C" else "Put"
            return sym, exp_dt, strike, cp_full
        except Exception:
            return None

    # Parse instrument
    parsed = df[instr_col].apply(_parse)
    # print(parsed)
    ok = parsed.notna()
    if not ok.any():
        raise ValueError("No rows matched pattern like 'BTC-26DEC25-144000-C'.")

    base = df.loc[ok].copy()
    sym, exp, k, cp_full = zip(*parsed[ok])
    # base["symbol"]      = list(sym)
    base["act_symbol"] = list(sym)
    base["expiration"] = list(exp)
    base["strike"] = list(k)
    base["cp"] = list(cp_full)

    # Observation date from real timestamp (UTC → naive date)
    base["date"] = pd.to_datetime(base[ts_col], unit="ms", utc=True).dt.tz_convert(None).dt.date

    # BTC-quoted → USD premiums
    spot = pd.to_numeric(base[spot_col], errors="coerce")
    bid = pd.to_numeric(base[bid_btc_col], errors="coerce") * spot
    ask = pd.to_numeric(base[ask_btc_col], errors="coerce") * spot
    mid = (bid + ask) / 2.0

    # Minimal cleaning
    keep = (bid > 0) & (ask > 0) & (ask >= bid)
    base = base.loc[keep, ["date", "act_symbol", "expiration", "strike", "cp"]].copy()
    base["bid"] = bid.loc[base.index].values
    base["ask"] = ask.loc[base.index].values
    base["mid"] = mid.loc[base.index].values

    # Sorted output
    out = base.sort_values(["date", "act_symbol", "expiration", "strike", "cp"]).reset_index(drop=True)
    return out


def _predict_model_prices(g: pd.DataFrame, price_fn, model_tag: str, bucket_label: str) -> pd.DataFrame:
    """
    Return one row per option with the model price.
    Keeps: date, symbol, expiration, tau, bucket, strike, market (C_mid),
           sigma_mkt_b76, F, DF, r, moneyness_F, and adds: price_hat, model.
    """
    h = g[
        ["date", "symbol", "expiration", "tau", "strike", "C_mid", "sigma_mkt_b76", "F", "DF", "r", "moneyness_F"]
    ].copy()
    h["bucket"] = bucket_label
    h["price_hat"] = [price_fn(F, K, tau, r) for F, K, tau, r in zip(h["F"], h["strike"], h["tau"], h["r"])]
    h["model"] = model_tag
    return h[
        [
            "date",
            "symbol",
            "expiration",
            "tau",
            "bucket",
            "strike",
            "C_mid",
            "price_hat",
            "sigma_mkt_b76",
            "F",
            "DF",
            "r",
            "moneyness_F",
            "model",
        ]
    ]


# ============================================================
# Main entry: summarize symbol over a period with saving
# ============================================================


def summarize_symbol_period_ivrmse(
    df_all: pd.DataFrame,
    symbol: str = "SPY",
    type: str = "american",
    start_date: str = "2025-04-01",
    end_date: str = "2025-06-30",
    buckets: List[int] = [14, 28, 56],
    min_parity_pairs: int = 4,
    tau_floor_days: int = 3,
    run_bs: bool = True,
    run_jd: bool = True,
    run_heston: bool = True,
    run_qlbs: bool = False,
    run_rlop: bool = False,
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
    if type == "american":
        df_pre = adapter_eur_calls_to_summarizer(df_all)
    else:
        df_pre = preprocess_deribit(df_all)

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
    df = df_pre.copy()
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
    preds_rows = []

    for day in iterator:
        df_day = df[df["date"] == day]
        calls, _ = prepare_calls_one_day_symbol(
            df_day, min_parity_pairs=min_parity_pairs, tau_floor_days=tau_floor_days, type=type
        )
        # print(calls)
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

                def bs_price(F, K, tau, r):  # priced with fitted sigma
                    return b76_price_call(F, K, tau, r, sig_hat)

                ivs = _ivrmse_vs_constant_sigma(g, sig_hat)
                row.update(
                    {
                        "BS_IVRMSE_x1000_Whole": ivs["whole"] * 1000 if np.isfinite(ivs["whole"]) else np.nan,
                        "BS_IVRMSE_x1000_<1": ivs["<1"] * 1000 if np.isfinite(ivs["<1"]) else np.nan,
                        "BS_IVRMSE_x1000_>1": ivs[">1"] * 1000 if np.isfinite(ivs[">1"]) else np.nan,
                        "BS_IVRMSE_x1000_>1.03": ivs[">1.03"] * 1000 if np.isfinite(ivs[">1.03"]) else np.nan,
                    }
                )
                # NEW: stash BS per-row predictions
                preds_rows.append(_predict_model_prices(g, bs_price, "BS", bucket))

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
                # NEW: stash JD per-row predictions
                preds_rows.append(_predict_model_prices(g, jd_price, "JD", bucket))

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
                # NEW: stash SV per-row predictions
                preds_rows.append(_predict_model_prices(g, h_price, "SV", bucket))

            # QLBS model
            ##########################################################################
            # NOTE: this part is the new QLBS model
            ##########################################################################
            if run_qlbs:
                from lib.qlbs2.test_trained_model import QLBSModel

                risk_lambda = 0.01
                Qmodel = QLBSModel(
                    is_call_option=True,
                    checkpoint=f"trained_model/test8/risk_lambda={risk_lambda:.1e}/policy_1.pt",
                    anchor_T=28 / 252,
                )
                # spot = (g["F"] * np.exp(-g["r"] * g["tau"])).iloc[0]
                spot = g["F"].iloc[0]
                time_to_expiries = g["tau"].to_numpy()
                strikes = g["strike"].to_numpy()
                r = g["r"].iloc[0]
                friction = 4e-3
                observed_prices = g["C_mid"].to_numpy()

                moneyness = np.log(g["strike"] / g["F"]) / (np.sqrt(g["tau"]) * 0.2)
                inv_price = 1 / np.power(np.clip(observed_prices, 1.0, None), 1.0)

                try:
                    result = Qmodel.fit(
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

                    def qlbs_price(F, K, tau, r):
                        _result = Qmodel.predict(
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

                    ivs = _ivrmse_vs_model_prices(g, qlbs_price)
                except KeyboardInterrupt:
                    ivs = {"whole": np.nan, "<1": np.nan, ">1": np.nan, ">1.03": np.nan}
                row.update(
                    {
                        "QLBS_IVRMSE_x1000_Whole": ivs["whole"] * 1000 if np.isfinite(ivs["whole"]) else np.nan,
                        "QLBS_IVRMSE_x1000_<1": ivs["<1"] * 1000 if np.isfinite(ivs["<1"]) else np.nan,
                        "QLBS_IVRMSE_x1000_>1": ivs[">1"] * 1000 if np.isfinite(ivs[">1"]) else np.nan,
                        "QLBS_IVRMSE_x1000_>1.03": ivs[">1.03"] * 1000 if np.isfinite(ivs[">1.03"]) else np.nan,
                    }
                )
                # NEW: stash QLBS per-row predictions
                preds_rows.append(_predict_model_prices(g, qlbs_price, "QLBS", bucket))

                ##########################################################################

            # RLOP model
            ##########################################################################
            # NOTE: this part is the new RLOP model
            ##########################################################################
            if run_rlop:
                from lib.rlop2.test_trained_model import RLOPModel

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
                # NEW: stash RLOP per-row predictions
                preds_rows.append(_predict_model_prices(g, rlop_price, "RLOP", bucket))

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
            ("QLBS", "QLBS_IVRMSE_x1000_Whole"),
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

    print(f"========================{outp=}")
    # ---- Save per-option predictions (long + wide) ----
    if outp is not None and len(preds_rows) > 0:
        preds_all = pd.concat(preds_rows, ignore_index=True)

        keys = [
            "date",
            "symbol",
            "expiration",
            "tau",
            "bucket",
            "strike",
            "C_mid",
            "sigma_mkt_b76",
            "F",
            "DF",
            "r",
            "moneyness_F",
        ]
        preds_wide = preds_all.pivot_table(
            index=keys, columns="model", values="price_hat", aggfunc="first"
        ).reset_index()
        # Make column names friendly: BS→price_BS, JD→price_JD, SV→price_SV
        if "BS" in preds_wide.columns:
            preds_wide = preds_wide.rename(columns={"BS": "price_BS"})
        if "JD" in preds_wide.columns:
            preds_wide = preds_wide.rename(columns={"JD": "price_JD"})
        if "SV" in preds_wide.columns:
            preds_wide = preds_wide.rename(columns={"SV": "price_SV"})
        if "QLBS" in preds_wide.columns:
            preds_wide = preds_wide.rename(columns={"QLBS": "price_QLBS"})
        if "RLOP" in preds_wide.columns:
            preds_wide = preds_wide.rename(columns={"RLOP": "price_RLOP"})

        preds_wide_fp = outp / "predicted_prices_wide.csv"
        preds_wide.to_csv(preds_wide_fp, index=False)
        print(f"Saved: {preds_wide_fp}")

    res = {"daily": daily, "equal_day_mean": equal_day_mean, "pooled": pooled}
    if outp is not None:
        with open(outp / "summary_res.pkl", "wb") as f:
            pickle.dump(res, f)

    return res


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
        for m, prefix in [("BS", "BS"), ("JD", "JD"), ("SV", "Heston"), ("QLBS", "QLBS"), ("RLOP", "RLOP")]
        if any(col.startswith(f"{prefix}_IVRMSE_x1000") for col in present)
    ]
    model_to_prefix = {"BS": "BS", "JD": "JD", "SV": "Heston", "QLBS": "QLBS", "RLOP": "RLOP"}

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

# ============================================================
# Dynamic hedging helpers: GBM simulation + generic delta-hedge
# ============================================================

def _simulate_gbm_paths(
    S0: float,
    r: float,
    sigma_true: float,
    T: float,
    n_steps: int = 28,
    n_paths: int = 2000,
    seed: int = 123,
) -> np.ndarray:
    """
    Simple GBM under risk-neutral measure:
        dS/S = r dt + sigma_true dW
    Returns array of shape (n_paths, n_steps+1) with S_t along each path.
    """
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_paths, n_steps))
    S = np.empty((n_paths, n_steps + 1), dtype=float)
    S[:, 0] = S0
    for t in range(n_steps):
        S[:, t + 1] = S[:, t] * np.exp((r - 0.5 * sigma_true**2) * dt + sigma_true * math.sqrt(dt) * Z[:, t])
    return S


def _delta_from_pricer(price_fn, S: np.ndarray, K: float, tau: float, r: float, eps: float = 1e-4) -> np.ndarray:
    """
    Finite-difference delta: dC/dS using central difference.
    price_fn(S_vec, K, tau, r) must accept vector S and return vector prices.
    """
    S = np.asarray(S, dtype=float)
    up = price_fn(S * (1.0 + eps), K, tau, r)
    dn = price_fn(S * (1.0 - eps), K, tau, r)
    return (up - dn) / (2.0 * eps * S)


def _hedge_paths(
    S_paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    price_fn,
    delta_fn,
    friction: float = 0.0,
) -> Dict[str, float]:
    """
    Delta-hedge a short call using given price_fn & delta_fn.
    friction: proportional cost per $ traded (e.g. 4e-3).
    Returns hedging metrics:
        RMSE_hedge    = sqrt(E[(V_T - payoff)^2])
        avg_cost      = E[ total trading cost ]
        shortfall_prob= P(V_T - payoff < 0)
        shortfall_1pct= 1% quantile of (V_T - payoff)
    """
    n_paths, n_steps_plus_1 = S_paths.shape
    n_steps = n_steps_plus_1 - 1
    dt = T / n_steps

    S0 = S_paths[:, 0]
    tau0 = T

    ST = S_paths[:, -1]
    payoff = np.maximum(ST - K, 0.0)

    # Initial price and delta
    C0 = price_fn(S0, K, tau0, r)       # vector length n_paths
    delta = delta_fn(S0, K, tau0, r)

    # Short 1 option, receive C0; hold delta shares and cash in bank
    cash = C0 - delta * S0
    trading_cost = np.zeros_like(cash)

    for t in range(1, n_steps + 1):
        tau = max(T - t * dt, 0.0)
        St = S_paths[:, t]

        # Recompute delta
        new_delta = delta_fn(St, K, tau, r)
        trade = new_delta - delta  # shares to buy (>0) or sell (<0)

        # Trading cost
        if friction > 0.0:
            cost = friction * np.abs(trade) * St
            trading_cost += cost
            cash -= cost

        # Cash to fund underlying trade
        cash -= trade * St

        delta = new_delta

        # Risk-free growth between hedges (no growth after last step)
        if t < n_steps:
            cash *= math.exp(r * dt)

    V_T = cash + delta * ST
    error = V_T - payoff

    rmse = float(np.sqrt(np.mean(error**2)))
    avg_cost = float(np.mean(trading_cost))
    shortfall_prob = float(np.mean(error < 0.0))
    shortfall_1pct = float(np.quantile(error, 0.01))

    return {
        "RMSE_hedge": rmse,
        "avg_cost": avg_cost,
        "shortfall_prob": shortfall_prob,
        "shortfall_1pct": shortfall_1pct,
    }

# ============================================================
# Reusable dynamic hedging runner for a symbol/period
# ============================================================

def summarize_symbol_period_hedging(
    df_all: pd.DataFrame,
    symbol: str,
    type: str,
    start_date: str,
    end_date: str,
    buckets: List[int] = (14, 28, 56),
    min_parity_pairs: int = 4,
    tau_floor_days: int = 3,
    n_paths: int = 2000,
    friction: float = 4e-3,
    seed: int = 123,
    run_bs: bool = True,
    run_jd: bool = True,
    run_heston: bool = True,
    run_qlbs: bool = True,
    run_rlop: bool = True,
    show_progress: bool = True,
    print_daily: bool = True,
    out_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Dynamic hedging summary consistent with IVRMSE:

      • Same preprocessing (adapter_eur_calls_to_summarizer / preprocess_deribit)
      • Loop over ALL days in [start_date, end_date]
      • For each day & bucket:
          - use the bucket cross-section (same as IVRMSE)
          - calibrate BS / JD / SV / QLBS / RLOP on that bucket
          - pick representative strikes for each moneyness slice
          - simulate GBM with sigma_true = bucket median IV
          - delta-hedge a short call with each model

      • Returns:
          res["daily"]          : one row per day × bucket × moneyness × model
          res["equal_day_mean"]: per bucket × moneyness × model, averaged over days
    """
    # --- Preprocess, same entry as IVRMSE summariser ---
    if type == "american":
        df_pre = adapter_eur_calls_to_summarizer(df_all)
    else:
        df_pre = preprocess_deribit(df_all)

    sym_col = "act_symbol" if "act_symbol" in df_pre.columns else "symbol"
    df_pre["date"] = pd.to_datetime(df_pre["date"]).dt.normalize()

    mask = (
        (df_pre[sym_col] == symbol)
        & (df_pre["date"] >= pd.Timestamp(start_date))
        & (df_pre["date"] <= pd.Timestamp(end_date))
    )
    df = df_pre.loc[mask].copy()
    if df.empty:
        raise ValueError(f"No rows for {symbol} in [{start_date}, {end_date}] in hedging summariser.")

    # Bucket mapper (same style as IVRMSE)
    def assign_bucket_centers(tau_years: float) -> str:
        days = tau_years * 365.0
        i = int(np.argmin([abs(days - c) for c in buckets]))
        return f"{buckets[i]}d"

    # Moneyness “sections” parallel to IVRMSE
    # (not disjoint on purpose, to match your static table)
    moneyness_sections = [
        "Whole sample",
        "Moneyness <1",
        "Moneyness >1",
        "Moneyness >1.03"
    ]

    days = sorted(df["date"].unique())
    iterator = days
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(days, desc=f"hedge {symbol} {start_date}→{end_date}")
        except Exception:
            pass

    daily_rows: List[Dict[str, float]] = []

    # Import RL models once
    from lib.qlbs2.test_trained_model import QLBSModel
    from lib.rlop2.test_trained_model import RLOPModel

    for day in iterator:
        df_day = df[df["date"] == day]
        calls, _ = prepare_calls_one_day_symbol(
            df_day, min_parity_pairs=min_parity_pairs, tau_floor_days=tau_floor_days, type=type
        )
        if calls.empty:
            if print_daily:
                print(f"[{pd.Timestamp(day).date()}] no valid contracts after filters")
            continue

        calls["bucket"] = calls["tau"].apply(assign_bucket_centers)
        if "moneyness_F" not in calls.columns:
            calls["moneyness_F"] = calls["strike"] / calls["F"]

        for bucket_label, calls_b in calls.groupby("bucket"):
            calls_b = calls_b.copy()
            if calls_b.empty:
                continue

            # ----- world settings for this (day, bucket) -----
            # World volatility = median market IV in this bucket (same as before)
            sigma_true = float(calls_b["sigma_mkt_b76"].median())

            # Use the bucket centre as the hedging horizon: 14d / 28d / 56d
            center_days = int(bucket_label.rstrip("d"))
            T_world = center_days / 252.0
            n_steps_local = max(1, center_days)      # hedge once per trading day

            # Representative S0 and r for the underlying world
            row0 = calls_b.iloc[0]
            S0_world = float(row0["F"])
            r_world = float(row0["r"])

            # ----- simulate GBM ONCE per (day, bucket) -----
            S_paths = _simulate_gbm_paths(
                S0=S0_world,
                r=r_world,
                sigma_true=sigma_true,
                T=T_world,
                n_steps=n_steps_local,
                n_paths=n_paths,
                seed=seed,
            )

            # ===== calibrations: identical to your original code =====
            price_fns: Dict[str, callable] = {}
            delta_fns: Dict[str, callable] = {}

            # BS
            if run_bs:
                sigma_bs = fit_sigma_bucket(calls_b)

                def bs_price(F, K, tau, r):
                    return b76_price_call(F, K, tau, r, sigma_bs)

                price_fns["BS"] = lambda S_vec, K_, tau_, r_: np.array(
                    [bs_price(Si, K_, tau_, r_) for Si in np.atleast_1d(S_vec)], dtype=float
                )
                delta_fns["BS"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(
                    price_fns["BS"], S_vec, K_, tau_, r_
                )

            # JD
            if run_jd:
                jd_params, _ = calibrate_jd_bucket(calls_b)
                sigma_jd = jd_params["sigma"]
                lam_jd = jd_params["lam"]
                muJ_jd = jd_params["muJ"]
                dJ_jd = jd_params["deltaJ"]

                def jd_price(F, K, tau, r):
                    return merton_price_call_b76(F, K, tau, r, sigma_jd, lam_jd, muJ_jd, dJ_jd)

                price_fns["JD"] = lambda S_vec, K_, tau_, r_: np.array(
                    [jd_price(Si, K_, tau_, r_) for Si in np.atleast_1d(S_vec)], dtype=float
                )
                delta_fns["JD"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(
                    price_fns["JD"], S_vec, K_, tau_, r_
                )

            # Heston
            if run_heston:
                h_params, _ = calibrate_heston_bucket(calls_b, u_max=100.0, n_points=501)

                def sv_price(F, K, tau, r):
                    return heston_price_call(F, K, tau, r, h_params, u_max=100.0, n_points=501)

                price_fns["SV"] = lambda S_vec, K_, tau_, r_: np.array(
                    [sv_price(Si, K_, tau_, r_) for Si in np.atleast_1d(S_vec)], dtype=float
                )
                delta_fns["SV"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(
                    price_fns["SV"], S_vec, K_, tau_, r_
                )

            # QLBS
            Qmodel = None
            if run_qlbs:
                from lib.qlbs2.test_trained_model import QLBSModel

                risk_lambda_qlbs = 0.01
                time_to_expiries = calls_b["tau"].to_numpy()
                strikes = calls_b["strike"].to_numpy()
                observed_prices = calls_b["C_mid"].to_numpy()
                inv_price = 1.0 / np.power(np.clip(observed_prices, 1.0, None), 1.0)

                Qmodel = QLBSModel(
                    is_call_option=True,
                    checkpoint=f"trained_model/test8/risk_lambda={risk_lambda_qlbs:.1e}/policy_1.pt",
                    anchor_T=28 / 252,
                )
                q_result = Qmodel.fit(
                    spot=S0_world,
                    time_to_expiries=time_to_expiries,
                    strikes=strikes,
                    r=r_world,
                    risk_lambda=risk_lambda_qlbs,
                    friction=friction,
                    observed_prices=observed_prices,
                    weights=inv_price,
                    sigma_guess=0.3,
                    mu_guess=0.0,
                    n_epochs=2000,
                )
                sigma_q, mu_q = q_result.sigma, q_result.mu

                def qlbs_price(F, K, tau, r):
                    res = Qmodel.predict(
                        spot=F,
                        time_to_expiries=np.array([tau]),
                        strikes=np.array([K]),
                        r=r,
                        risk_lambda=risk_lambda_qlbs,
                        friction=friction,
                        sigma_fit=sigma_q,
                        mu_fit=mu_q,
                    )
                    return res.estimated_prices[0]

                price_fns["QLBS"] = lambda S_vec, K_, tau_, r_: np.array(
                    [qlbs_price(Si, K_, tau_, r_) for Si in np.atleast_1d(S_vec)], dtype=float
                )
                delta_fns["QLBS"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(
                    price_fns["QLBS"], S_vec, K_, tau_, r_
                )

            # RLOP
            Rmodel = None
            if run_rlop:
                from lib.rlop2.test_trained_model import RLOPModel

                risk_lambda_rlop = 0.10
                time_to_expiries = calls_b["tau"].to_numpy()
                strikes = calls_b["strike"].to_numpy()
                observed_prices = calls_b["C_mid"].to_numpy()
                inv_price = 1.0 / np.power(np.clip(observed_prices, 1.0, None), 1.0)

                Rmodel = RLOPModel(
                    is_call_option=True,
                    checkpoint="trained_model/testr9/policy_1.pt",
                    anchor_T=28 / 252,
                )
                r_result = Rmodel.fit(
                    spot=S0_world,
                    time_to_expiries=time_to_expiries,
                    strikes=strikes,
                    r=r_world,
                    risk_lambda=risk_lambda_rlop,
                    friction=friction,
                    observed_prices=observed_prices,
                    weights=inv_price,
                    sigma_guess=0.3,
                    mu_guess=0.0,
                    n_epochs=2000,
                )
                sigma_r, mu_r = r_result.sigma, r_result.mu

                def rlop_price(F, K, tau, r):
                    res = Rmodel.predict(
                        spot=F,
                        time_to_expiries=np.array([tau]),
                        strikes=np.array([K]),
                        r=r,
                        risk_lambda=risk_lambda_rlop,
                        friction=friction,
                        sigma_fit=sigma_r,
                        mu_fit=mu_r,
                    )
                    return res.estimated_prices[0]

                price_fns["RLOP"] = lambda S_vec, K_, tau_, r_: np.array(
                    [rlop_price(Si, K_, tau_, r_) for Si in np.atleast_1d(S_vec)], dtype=float
                )
                delta_fns["RLOP"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(
                    price_fns["RLOP"], S_vec, K_, tau_, r_
                )

            models = list(price_fns.keys())

            # ===== loop over moneyness sections, reusing S_paths =====
            for section_name in moneyness_sections:
                if section_name == "Whole sample":
                    sub = calls_b
                elif section_name == "Moneyness <1":
                    sub = calls_b[calls_b["moneyness_F"] < 1.0]
                elif section_name == "Moneyness >1":
                    sub = calls_b[calls_b["moneyness_F"] > 1.0]
                elif section_name == "Moneyness >1.03":
                    sub = calls_b[calls_b["moneyness_F"] > 1.03]
                else:
                    continue

                if sub.empty:
                    continue

                # Representative strike for this moneyness slice
                m_target = float(sub["moneyness_F"].median())
                idx = (sub["moneyness_F"] - m_target).abs().idxmin()
                row_rep = sub.loc[idx]
                K = float(row_rep["strike"])

                for model_name in models:
                    metrics = _hedge_paths(
                        S_paths=S_paths,
                        K=K,
                        r=r_world,
                        T=T_world,
                        price_fn=price_fns[model_name],
                        delta_fn=delta_fns[model_name],
                        friction=friction,
                    )
                    row_out = {
                        "date": pd.Timestamp(day),
                        "symbol": symbol,
                        "bucket": bucket_label,
                        "moneyness_section": section_name,
                        "model": model_name,
                        "S0": S0_world,
                        "K": K,
                        "T_days": T_world * 252.0,
                        "sigma_true": sigma_true,
                    }
                    row_out.update(metrics)
                    daily_rows.append(row_out)

        # optional per-day logging (only using Whole-sample, 28d if present)
        if print_daily and daily_rows:
            # just print something short so you can see progress
            pass

    if not daily_rows:
        raise RuntimeError("summarize_symbol_period_hedging: no (day,bucket) rows produced.")

    daily = pd.DataFrame(daily_rows)

    metric_cols = ["RMSE_hedge", "avg_cost", "shortfall_prob", "shortfall_1pct"]

    equal_day_mean = (
        daily
        .groupby(["bucket", "moneyness_section", "model"], as_index=False)[metric_cols]
        .mean()
    )
    days_used = (
        daily.groupby(["bucket", "moneyness_section"])["date"]
        .nunique()
        .rename("days_used")
        .reset_index()
    )
    equal_day_mean = equal_day_mean.merge(
        days_used, on=["bucket", "moneyness_section"], how="left"
    )

    if out_dir is not None:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        daily.to_csv(outp / "hedging_daily.csv", index=False)
        equal_day_mean.to_csv(outp / "hedging_equal_day_mean.csv", index=False)

    return {"daily": daily, "equal_day_mean": equal_day_mean}


# def run_dynamic_hedging_for_symbol_period(
#     df_all: pd.DataFrame,
#     symbol: str,
#     type: str,
#     start_date: str,
#     end_date: str,
#     buckets: List[int] = [28],
#     min_parity_pairs: int = 4,
#     tau_floor_days: int = 3,
#     n_steps: int = 28,
#     n_paths: int = 2000,
#     friction: float = 4e-3,
#     seed: int = 123,
#     run_bs: bool = True,
#     run_jd: bool = True,
#     run_heston: bool = True,
#     run_qlbs: bool = True,
#     run_rlop: bool = True,
# ) -> pd.DataFrame:
#     """
#     One-shot dynamic hedging summary for a symbol & period.

#     Strategy:
#       - Preprocess df_all via adapter_eur_calls_to_summarizer or preprocess_deribit (same as IVRMSE).
#       - Filter to [symbol, start_date..end_date].
#       - Pick a representative 'mid' day in the period that has valid calls.
#       - For each requested bucket (e.g., 14,28,56d):
#           * Build bucket cross-section (calls_bucket).
#           * sigma_true = median market IV in that bucket.
#           * Calibrate BS / JD / SV / QLBS / RLOP to that bucket (same as IVRMSE).
#           * Pick ATM-ish strike K and T from that bucket.
#           * Simulate GBM world with sigma_true.
#           * Delta-hedge short call with each model.
#       - Return long-form DataFrame with columns:
#           ['symbol','date','period','bucket','model',
#            'S0','K','T_days','sigma_true',
#            'RMSE_hedge','avg_cost','shortfall_prob','shortfall_1pct'].
#     """
#     # --- Preprocess calls, same entry logic as summarize_symbol_period_ivrmse ---
#     if type == "american":
#         df_pre = adapter_eur_calls_to_summarizer(df_all)
#     else:
#         df_pre = preprocess_deribit(df_all)

#     sym_col = "act_symbol" if "act_symbol" in df_pre.columns else "symbol"
#     df_pre["date"] = pd.to_datetime(df_pre["date"]).dt.normalize()

#     mask = (
#         (df_pre[sym_col] == symbol)
#         & (df_pre["date"] >= pd.Timestamp(start_date))
#         & (df_pre["date"] <= pd.Timestamp(end_date))
#     )
#     df = df_pre.loc[mask].copy()
#     if df.empty:
#         raise ValueError(f"No rows for {symbol} in [{start_date}, {end_date}] in dynamic hedging runner.")

#     # Pick a representative day (midpoint in time)
#     unique_days = sorted(df["date"].unique())
#     rep_day = unique_days[len(unique_days) // 2]
#     df_day = df[df["date"] == rep_day]

#     # Build calls for that day using your fast-path/european logic
#     calls, _ = prepare_calls_one_day_symbol(
#         df_day, min_parity_pairs=min_parity_pairs, tau_floor_days=tau_floor_days, type=type
#     )
#     if calls.empty:
#         raise RuntimeError(f"Dynamic hedging: no valid calls on representative day {rep_day.date()}.")

#     # Bucket mapper using requested centers
#     def assign_bucket_centers(tau_years: float) -> str:
#         days = tau_years * 365.0
#         i = int(np.argmin([abs(days - c) for c in buckets]))
#         return f"{buckets[i]}d"

#     calls["bucket"] = calls["tau"].apply(assign_bucket_centers)

#     bucket_labels = [f"{d}d" for d in buckets]
#     period_str = f"{start_date}..{end_date}"

#     results_rows = []

#     # Import RL models once
#     from lib.qlbs2.test_trained_model import QLBSModel
#     from lib.rlop2.test_trained_model import RLOPModel

#     for d in bucket_labels:
#         calls_b = calls[calls["bucket"] == d].copy()
#         if calls_b.empty:
#             print(f"[dynamic hedging] bucket {d} has no contracts on {rep_day.date()}, skipping.")
#             continue

#         # World vol = median market IV on this bucket
#         sigma_true = float(calls_b["sigma_mkt_b76"].median())

#         # Choose ATM-ish strike: moneyness_F ~= 1
#         calls_b["moneyness_F"] = calls_b["strike"] / calls_b["F"]
#         idx_atm = (calls_b["moneyness_F"] - 1.0).abs().idxmin()
#         row_atm = calls_b.loc[idx_atm]

#         F0 = float(row_atm["F"])
#         K = float(row_atm["strike"])
#         T = float(row_atm["tau"])
#         r = float(row_atm["r"])
#         S0 = F0  # treat forward ~ spot for hedging world

#         # Calibrate BS baseline
#         sigma_bs = fit_sigma_bucket(calls_b) if run_bs else np.nan

#         # Calibrate JD
#         jd_params = None
#         if run_jd:
#             jd_params, _ = calibrate_jd_bucket(calls_b)

#         # Calibrate Heston (SV)
#         h_params = None
#         if run_heston:
#             h_params, _ = calibrate_heston_bucket(calls_b, u_max=100.0, n_points=501)

#         # Calibrate QLBS on this bucket (same style as IVRMSE)
#         q_result = None
#         if run_qlbs:
#             risk_lambda_qlbs = 0.01
#             time_to_expiries = calls_b["tau"].to_numpy()
#             strikes = calls_b["strike"].to_numpy()
#             observed_prices = calls_b["C_mid"].to_numpy()
#             inv_price = 1.0 / np.power(np.clip(observed_prices, 1.0, None), 1.0)

#             Qmodel = QLBSModel(
#                 is_call_option=True,
#                 checkpoint=f"trained_model/test8/risk_lambda={risk_lambda_qlbs:.1e}/policy_1.pt",
#                 anchor_T=28 / 252,
#             )
#             q_result = Qmodel.fit(
#                 spot=S0,
#                 time_to_expiries=time_to_expiries,
#                 strikes=strikes,
#                 r=r,
#                 risk_lambda=risk_lambda_qlbs,
#                 friction=friction,
#                 observed_prices=observed_prices,
#                 weights=inv_price,
#                 sigma_guess=0.3,
#                 mu_guess=0.0,
#                 n_epochs=2000,
#             )

#         # Calibrate RLOP
#         r_result = None
#         if run_rlop:
#             risk_lambda_rlop = 0.10
#             time_to_expiries = calls_b["tau"].to_numpy()
#             strikes = calls_b["strike"].to_numpy()
#             observed_prices = calls_b["C_mid"].to_numpy()
#             inv_price = 1.0 / np.power(np.clip(observed_prices, 1.0, None), 1.0)

#             Rmodel = RLOPModel(
#                 is_call_option=True,
#                 checkpoint="trained_model/testr9/policy_1.pt",
#                 anchor_T=28 / 252,
#             )
#             r_result = Rmodel.fit(
#                 spot=S0,
#                 time_to_expiries=time_to_expiries,
#                 strikes=strikes,
#                 r=r,
#                 risk_lambda=risk_lambda_rlop,
#                 friction=friction,
#                 observed_prices=observed_prices,
#                 weights=inv_price,
#                 sigma_guess=0.3,
#                 mu_guess=0.0,
#                 n_epochs=2000,
#             )

#         # --------------------------------------------------------
#         # Wrap model-specific price functions price_fn(S_vec,K,tau,r)
#         # --------------------------------------------------------

#         price_fns = {}
#         delta_fns = {}

#         if run_bs and np.isfinite(sigma_bs):
#             def bs_price_fn(S_vec, K_, tau_, r_):
#                 S_vec = np.atleast_1d(S_vec)
#                 out = np.empty_like(S_vec, dtype=float)
#                 for i, Si in enumerate(S_vec):
#                     out[i] = b76_price_call(Si, K_, tau_, r_, sigma_bs)
#                 return out
#             price_fns["BS"] = bs_price_fn
#             delta_fns["BS"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(bs_price_fn, S_vec, K_, tau_, r_)

#         if run_jd and jd_params:
#             sigma_jd = jd_params["sigma"]
#             lam_jd = jd_params["lam"]
#             muJ_jd = jd_params["muJ"]
#             dJ_jd = jd_params["deltaJ"]

#             def jd_price_fn(S_vec, K_, tau_, r_):
#                 S_vec = np.atleast_1d(S_vec)
#                 out = np.empty_like(S_vec, dtype=float)
#                 for i, Si in enumerate(S_vec):
#                     out[i] = merton_price_call_b76(Si, K_, tau_, r_, sigma_jd, lam_jd, muJ_jd, dJ_jd)
#                 return out

#             price_fns["JD"] = jd_price_fn
#             delta_fns["JD"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(jd_price_fn, S_vec, K_, tau_, r_)

#         if run_heston and h_params:
#             def sv_price_fn(S_vec, K_, tau_, r_):
#                 S_vec = np.atleast_1d(S_vec)
#                 out = np.empty_like(S_vec, dtype=float)
#                 for i, Si in enumerate(S_vec):
#                     # Slightly lighter settings than calibration for speed
#                     out[i] = heston_price_call(Si, K_, tau_, r_, h_params, u_max=60.0, n_points=201)
#                 return out

#             price_fns["SV"] = sv_price_fn
#             delta_fns["SV"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(sv_price_fn, S_vec, K_, tau_, r_)

#         if run_qlbs and q_result:
#             risk_lambda_qlbs = 0.01
#             sigma_q, mu_q = q_result.sigma, q_result.mu

#             def qlbs_price_fn(S_vec, K_, tau_, r_):
#                 S_vec = np.atleast_1d(S_vec)
#                 out = np.empty_like(S_vec, dtype=float)
#                 for i, Si in enumerate(S_vec):
#                     res = Qmodel.predict(
#                         spot=Si,
#                         time_to_expiries=np.array([tau_]),
#                         strikes=np.array([K_]),
#                         r=r_,
#                         risk_lambda=risk_lambda_qlbs,
#                         friction=friction,
#                         sigma_fit=sigma_q,
#                         mu_fit=mu_q,
#                     )
#                     out[i] = res.estimated_prices[0]
#                 return out

#             price_fns["QLBS"] = qlbs_price_fn
#             delta_fns["QLBS"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(qlbs_price_fn, S_vec, K_, tau_, r_)

#         if run_rlop and r_result:
#             risk_lambda_rlop = 0.10
#             sigma_r, mu_r = r_result.sigma, r_result.mu

#             def rlop_price_fn(S_vec, K_, tau_, r_):
#                 S_vec = np.atleast_1d(S_vec)
#                 out = np.empty_like(S_vec, dtype=float)
#                 for i, Si in enumerate(S_vec):
#                     res = Rmodel.predict(
#                         spot=Si,
#                         time_to_expiries=np.array([tau_]),
#                         strikes=np.array([K_]),
#                         r=r_,
#                         risk_lambda=risk_lambda_rlop,
#                         friction=friction,
#                         sigma_fit=sigma_r,
#                         mu_fit=mu_r,
#                     )
#                     out[i] = res.estimated_prices[0]
#                 return out

#             price_fns["RLOP"] = rlop_price_fn
#             delta_fns["RLOP"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(rlop_price_fn, S_vec, K_, tau_, r_)

#         # --------------------------------------------------------
#         # Simulate GBM world and run hedging
#         # --------------------------------------------------------
#         S_paths = _simulate_gbm_paths(
#             S0=S0,
#             r=r,
#             sigma_true=sigma_true,
#             T=T,
#             n_steps=n_steps,
#             n_paths=n_paths,
#             seed=seed,
#         )

#         for model_name in ["BS", "JD", "SV", "QLBS", "RLOP"]:
#             if model_name not in price_fns:
#                 continue
#             metrics = _hedge_paths(
#                 S_paths=S_paths,
#                 K=K,
#                 r=r,
#                 T=T,
#                 price_fn=price_fns[model_name],
#                 delta_fn=delta_fns[model_name],
#                 friction=friction,
#             )
#             row = {
#                 "symbol": symbol,
#                 "date": rep_day,
#                 "period": period_str,
#                 "bucket": d,
#                 "model": model_name,
#                 "S0": S0,
#                 "K": K,
#                 "T_days": T * 252.0,
#                 "sigma_true": sigma_true,
#             }
#             row.update(metrics)
#             results_rows.append(row)

#     if not results_rows:
#         raise RuntimeError("run_dynamic_hedging_for_symbol_period: no bucket produced results.")

#     hedge_res = pd.DataFrame(results_rows)
#     return hedge_res

# ============================================================
# Publication-style hedging table (BS / JD / SV / QLBS / RLOP)
# ============================================================

def make_hedging_publication_table(
    hedge_res: Dict[str, pd.DataFrame],
    symbol: str,
    metric: str = "RMSE_hedge",   # "avg_cost", "shortfall_prob", "shortfall_1pct"
    buckets: List[int] = (14, 28, 56),
    decimals: int = 3,
    out_dir: Optional[str] = None,
    basename: str = "table_hedging",
) -> pd.DataFrame:
    """
    Build a hedging table parallel to the IVRMSE publication table.

      Sections (rows grouped by this):
        Whole sample / Moneyness <1 / Moneyness >1 / Moneyness >1.03

      Rows:
        one per maturity bucket, e.g. 'SPY (τ=28d)'

      Columns:
        BS, JD, SV, QLBS, RLOP   (whichever are present in hedge_res)

      Values:
        equal-day mean of the chosen hedging metric across the period.
    """
    if "equal_day_mean" not in hedge_res or hedge_res["equal_day_mean"].empty:
        raise ValueError("make_hedging_publication_table: equal_day_mean is empty.")

    df_src = hedge_res["equal_day_mean"].copy()

    models = sorted(df_src["model"].unique().tolist())
    sections = [
        "Whole sample",
        "Moneyness <1",
        "Moneyness >1",
        "Moneyness >1.03",
    ]
    bucket_labels = [f"{d}d" for d in buckets]

    rows = []
    for section_name in sections:
        for dlab in bucket_labels:
            row = {"Moneyness": section_name, "Asset": f"{symbol} (τ={dlab})"}
            sub = df_src[
                (df_src["bucket"] == dlab)
                & (df_src["moneyness_section"] == section_name)
            ]
            for m in models:
                val = np.nan
                if not sub.empty:
                    sm = sub[sub["model"] == m]
                    if not sm.empty and metric in sm.columns:
                        val = float(sm[metric].iloc[0])
                row[m] = val
            rows.append(row)

    table = pd.DataFrame(rows)
    for m in models:
        table[m] = table[m].round(decimals)

    if out_dir:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        csv_path = outp / f"{basename}_{metric}.csv"
        table.to_csv(csv_path, index=False)

        # Markdown with bold minima, same style as IVRMSE
        header = ["Moneyness", "Asset"] + models
        md_rows = []
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
        md_text = "\n".join(md_rows)
        md_path = outp / f"{basename}_{metric}.md"
        md_path.write_text(md_text, encoding="utf-8")
        print(f"Saved: {csv_path}")
        print(f"Saved: {md_path}")

    return table

# def make_hedging_table(
#     hedge_res: pd.DataFrame,
#     symbol: str,
#     metric: str = "RMSE_hedge",  # or "avg_cost", "shortfall_prob", "shortfall_1pct"
#     buckets: List[int] = (14, 28, 56),
#     decimals: int = 4,
#     out_dir: Optional[str] = None,
#     basename: str = "table_hedging",
# ) -> pd.DataFrame:
#     """
#     Build a paper-style hedging table analogous to IVRMSE tables:

#       Columns: BS, JD, SV, QLBS, RLOP (only those present in hedge_res)
#       Rows:    one per bucket, like 'SPY (τ=28d)'
#       Extra cols: 'Metric' (e.g., 'RMSE_hedge'), 'Asset'
#       Values:  chosen metric from hedge_res.

#     Saves CSV + Markdown (bolds row minimum) if out_dir is set.
#     """
#     if hedge_res is None or hedge_res.empty:
#         raise ValueError("hedge_res is empty in make_hedging_table.")

#     present_models = sorted(hedge_res["model"].unique().tolist())
#     # Only keep known models to stay consistent with IVRMSE layout
#     models = [m for m in ["BS", "JD", "SV", "QLBS", "RLOP"] if m in present_models]

#     bucket_labels = [f"{d}d" for d in buckets]
#     rows = []

#     for d in bucket_labels:
#         row = {"Metric": metric, "Asset": f"{symbol} (τ={d})"}
#         sub = hedge_res[hedge_res["bucket"] == d]
#         for m in models:
#             val = np.nan
#             if not sub.empty:
#                 sub_m = sub[sub["model"] == m]
#                 if not sub_m.empty and metric in sub_m.columns:
#                     val = float(sub_m[metric].iloc[0])
#             row[m] = val
#         rows.append(row)

#     table = pd.DataFrame(rows)
#     for m in models:
#         table[m] = table[m].round(decimals)

#     # Save CSV + Markdown with bold minima
#     if out_dir:
#         outp = Path(out_dir)
#         outp.mkdir(parents=True, exist_ok=True)
#         csv_path = outp / f"{basename}_{metric}.csv"
#         table.to_csv(csv_path, index=False)

#         header = ["Metric", "Asset"] + models
#         md_rows = []
#         md_rows.append("| " + " | ".join(header) + " |")
#         md_rows.append("| " + " | ".join(["---"] * len(header)) + " |")

#         for _, r in table.iterrows():
#             vals = [r[m] for m in models]
#             not_nan = [i for i, v in enumerate(vals) if pd.notna(v)]
#             best_idx = None
#             if not_nan:
#                 best_idx = min(not_nan, key=lambda i: vals[i])
#             cells = [str(r["Metric"]), str(r["Asset"])]
#             for i, m in enumerate(models):
#                 v = r[m]
#                 if pd.isna(v):
#                     cells.append("")
#                 else:
#                     s = f"{v:.{decimals}f}"
#                     cells.append(f"**{s}**" if i == best_idx else s)
#             md_rows.append("| " + " | ".join(cells) + " |")

#         md_text = "\n".join(md_rows)
#         md_path = outp / f"{basename}_{metric}.md"
#         md_path.write_text(md_text, encoding="utf-8")
#         print(f"Saved hedging CSV: {csv_path}")
#         print(f"Saved hedging Markdown: {md_path}")

#     return table

def hedging_spy20():
    df = pd.read_csv("data/spy_preprocessed_calls_20q1.csv")

    hedge_res = summarize_symbol_period_hedging(
        df_all=df,
        symbol="SPY",
        type="american",
        start_date="2020-01-06",
        end_date="2020-03-30",
        buckets=[14, 28, 56],   # or [28] if you prefer only 1M
        min_parity_pairs=4,
        tau_floor_days=3,
        n_paths=500,
        friction=4e-3,
        seed=123,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
    )

    # Primary dynamic-hedging table: RMSE_hedge
    tbl_rmse = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="RMSE_hedge",
        buckets=[14, 28, 56],
        decimals=4,
        out_dir="SPY_20Q1_baseline_v3",
        basename="table_hedging",
    )

    # Optional: cost and shortfall tables
    tbl_cost = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="avg_cost",
        buckets=[14, 28, 56],
        decimals=6,
        out_dir="SPY_20Q1_baseline_v3",
        basename="table_hedging",
    )
    tbl_shortfall = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="shortfall_prob",
        buckets=[14, 28, 56],
        decimals=3,
        out_dir="SPY_20Q1_baseline_v3",
        basename="table_hedging",
    )

    print("Dynamic hedging RMSE table (SPY 20Q1):")
    print(tbl_rmse)
    print(tbl_cost)
    print(tbl_shortfall)

def hedging_spy25():
    df = pd.read_csv("data/spy_preprocessed_calls_25.csv")

    hedge_res = summarize_symbol_period_hedging(
        df_all=df,
        symbol="SPY",
        type="american",
        start_date="2025-04-01",
        end_date="2025-06-30",
        buckets=[14, 28, 56],   # keep consistent with IVRMSE summary
        min_parity_pairs=4,
        tau_floor_days=3,
        n_paths=500,
        friction=4e-3,
        seed=123,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
    )

    # Primary dynamic-hedging table: RMSE_hedge
    tbl_rmse = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="RMSE_hedge",
        buckets=[14, 28, 56],
        decimals=4,
        out_dir="SPY_25Q2_baseline_v3",
        basename="table_hedging",
    )

    # Optional: cost and shortfall tables
    tbl_cost = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="avg_cost",
        buckets=[14, 28, 56],
        decimals=6,
        out_dir="SPY_25Q2_baseline_v3",
        basename="table_hedging",
    )
    tbl_shortfall = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="shortfall_prob",
        buckets=[14, 28, 56],
        decimals=3,
        out_dir="SPY_25Q2_baseline_v3",
        basename="table_hedging",
    )

    print("Dynamic hedging RMSE table (SPY 25Q2):")
    print(tbl_rmse)
    print(tbl_cost)
    print(tbl_shortfall)


def hedging_xop20():
    df = pd.read_csv("data/xop_preprocessed_calls_20q1.csv")

    hedge_res = summarize_symbol_period_hedging(
        df_all=df,
        symbol="XOP",
        type="american",
        start_date="2020-01-06",
        end_date="2020-03-30",
        buckets=[14, 28, 56],
        min_parity_pairs=4,
        tau_floor_days=3,
        n_paths=500,
        friction=4e-3,
        seed=123,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
    )

    # Primary dynamic-hedging table: RMSE_hedge
    tbl_rmse = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="RMSE_hedge",
        buckets=[14, 28, 56],
        decimals=4,
        out_dir="XOP_20Q1_baseline_v3",
        basename="table_hedging",
    )

    # Optional: cost and shortfall tables
    tbl_cost = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="avg_cost",
        buckets=[14, 28, 56],
        decimals=6,
        out_dir="XOP_20Q1_baseline_v3",
        basename="table_hedging",
    )
    tbl_shortfall = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="shortfall_prob",
        buckets=[14, 28, 56],
        decimals=3,
        out_dir="XOP_20Q1_baseline_v3",
        basename="table_hedging",
    )

    print("Dynamic hedging RMSE table (XOP 20Q1):")
    print(tbl_rmse)
    print(tbl_cost)
    print(tbl_shortfall)


def hedging_xop25():
    df = pd.read_csv("data/xop_preprocessed_calls_25.csv")

    hedge_res = summarize_symbol_period_hedging(
        df_all=df,
        symbol="XOP",
        type="american",
        start_date="2025-04-01",
        end_date="2025-06-30",
        buckets=[14, 28, 56],
        min_parity_pairs=4,
        tau_floor_days=3,
        n_paths=500,
        friction=4e-3,
        seed=123,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
    )

    # Primary dynamic-hedging table: RMSE_hedge
    tbl_rmse = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="RMSE_hedge",
        buckets=[14, 28, 56],
        decimals=4,
        out_dir="XOP_25Q2_baseline_v3",
        basename="table_hedging",
    )

    # Optional: cost and shortfall tables
    tbl_cost = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="avg_cost",
        buckets=[14, 28, 56],
        decimals=6,
        out_dir="XOP_25Q2_baseline_v3",
        basename="table_hedging",
    )
    tbl_shortfall = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="shortfall_prob",
        buckets=[14, 28, 56],
        decimals=3,
        out_dir="XOP_25Q2_baseline_v3",
        basename="table_hedging",
    )

    print("Dynamic hedging RMSE table (XOP 25Q2):")
    print(tbl_rmse)
    print(tbl_cost)
    print(tbl_shortfall)

def main_spy20():
    df = pd.read_csv("data/spy_preprocessed_calls_20q1.csv")
    # df_pre = adapter_eur_calls_to_summarizer(df)
    # df_pre.head()

    res = summarize_symbol_period_ivrmse(
        df_all=df,
        symbol="SPY",
        type="american",
        start_date="2020-01-06",
        end_date="2020-03-30",
        buckets=[14, 28, 56],
        min_parity_pairs=4,
        tau_floor_days=3,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        show_progress=True,
        print_daily=True,
        out_dir="SPY_20Q1_baseline_v3",  # outputs saved here
    )

    # PRIMARY (paper): equal-day mean table
    tbl_equal = make_publication_table(
        res,
        symbol="SPY",
        measure="equal",
        buckets=[14, 28, 56],
        decimals=2,
        out_dir="SPY_20Q1_baseline_v3",
        basename="table_ivrmse",
    )

    # SECONDARY (robustness): pooled table
    tbl_pooled = make_publication_table(
        res,
        symbol="SPY",
        measure="pooled",
        buckets=[14, 28, 56],
        decimals=2,
        out_dir="SPY_20Q1_baseline_v3",
        basename="table_ivrmse",
    )

    print(tbl_equal)
    print(tbl_pooled)


def main_spy25():
    df = pd.read_csv("data/spy_preprocessed_calls_25.csv")
    # df_pre = adapter_eur_calls_to_summarizer(df)
    # df_pre.head()

    res = summarize_symbol_period_ivrmse(
        df_all=df,
        symbol="SPY",
        type="american",
        start_date="2025-04-01",
        end_date="2025-06-30",
        buckets=[14, 28, 56],
        min_parity_pairs=4,
        tau_floor_days=3,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        show_progress=True,
        print_daily=True,
        out_dir="SPY_25Q2_baseline_v3",  # outputs saved here
    )

    # PRIMARY (paper): equal-day mean table
    tbl_equal = make_publication_table(
        res,
        symbol="SPY",
        measure="equal",
        buckets=[14, 28, 56],
        decimals=2,
        out_dir="SPY_25Q2_baseline_v3",
        basename="table_ivrmse",
    )

    # SECONDARY (robustness): pooled table
    tbl_pooled = make_publication_table(
        res,
        symbol="SPY",
        measure="pooled",
        buckets=[14, 28, 56],
        decimals=2,
        out_dir="SPY_25Q2_baseline_v3",
        basename="table_ivrmse",
    )

    print(tbl_equal)
    print(tbl_pooled)

def main_xop20():
    df = pd.read_csv("data/xop_preprocessed_calls_20q1.csv")
    # df_pre = adapter_eur_calls_to_summarizer(df)
    # df_pre.head()

    res = summarize_symbol_period_ivrmse(
        df_all=df,
        symbol="XOP",
        type="american",
        start_date="2020-01-06",
        end_date="2020-03-30",
        buckets=[14, 28, 56],
        min_parity_pairs=4,
        tau_floor_days=3,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        show_progress=True,
        print_daily=True,
        out_dir="XOP_20Q1_baseline_v3",  # outputs saved here
    )

    # PRIMARY (paper): equal-day mean table
    tbl_equal = make_publication_table(
        res,
        symbol="XOP",
        measure="equal",
        buckets=[14, 28, 56],
        decimals=2,
        out_dir="XOP_20Q1_baseline_v3",
        basename="table_ivrmse",
    )

    # SECONDARY (robustness): pooled table
    tbl_pooled = make_publication_table(
        res,
        symbol="XOP",
        measure="pooled",
        buckets=[14, 28, 56],
        decimals=2,
        out_dir="XOP_20Q1_baseline_v3",
        basename="table_ivrmse",
    )

    print(tbl_equal)
    print(tbl_pooled)


def main_xop25():
    df = pd.read_csv("data/xop_preprocessed_calls_25.csv")
    # df_pre = adapter_eur_calls_to_summarizer(df)
    # df_pre.head()

    res = summarize_symbol_period_ivrmse(
        df_all=df,
        symbol="XOP",
        type="american",
        start_date="2025-04-01",
        end_date="2025-06-30",
        buckets=[14, 28, 56],
        min_parity_pairs=4,
        tau_floor_days=3,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        show_progress=True,
        print_daily=True,
        out_dir="XOP_25Q2_baseline_v3",  # outputs saved here
    )

    # PRIMARY (paper): equal-day mean table
    tbl_equal = make_publication_table(
        res,
        symbol="XOP",
        measure="equal",
        buckets=[14, 28, 56],
        decimals=2,
        out_dir="XOP_25Q2_baseline_v3",
        basename="table_ivrmse",
    )

    # SECONDARY (robustness): pooled table
    tbl_pooled = make_publication_table(
        res,
        symbol="XOP",
        measure="pooled",
        buckets=[14, 28, 56],
        decimals=2,
        out_dir="XOP_25Q2_baseline_v3",
        basename="table_ivrmse",
    )

    print(tbl_equal)
    print(tbl_pooled)


def main_btc():
    df = pd.read_csv("data/BTC_E_Options_09NOV25.csv")
    # df_pre = preprocess_deribit(df)
    # df_pre.head()

    res = summarize_symbol_period_ivrmse(
        df_all=df,
        symbol="BTC",
        type="european",
        start_date="2025-11-09",
        end_date="2025-11-09",
        buckets=[7, 30, 90],
        min_parity_pairs=4,
        tau_floor_days=3,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        show_progress=True,
        print_daily=True,
        out_dir="BTC_09NOV25_v3",  # outputs saved here
    )

    # PRIMARY (paper): equal-day mean table
    tbl_equal = make_publication_table(
        res,
        symbol="BTC",
        measure="equal",
        buckets=[7, 30, 90],
        decimals=2,
        out_dir="BTC_09NOV25_v3",
        basename="table_ivrmse",
    )

    # SECONDARY (robustness): pooled table
    tbl_pooled = make_publication_table(
        res,
        symbol="BTC",
        measure="pooled",
        buckets=[7, 30, 90],
        decimals=2,
        out_dir="BTC_09NOV25_v3",
        basename="table_ivrmse",
    )

    print(tbl_equal)
    print(tbl_pooled)


class Test(TestCase):
    def test_main(self):
        #main_spy20()
        #main_spy25()
        #main_xop20()
        #main_xop25()
        #main_btc()
        #hedging_spy20()
        #hedging_spy25()
        hedging_xop20()
        hedging_xop25()


# python -m unittest playground/options_pricing_baselines_v3.py
