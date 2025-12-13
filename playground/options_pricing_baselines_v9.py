from __future__ import annotations

import math
import pickle
from unittest import TestCase
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd

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
        prices = df_bucket.apply(
            lambda r: b76_price_call(r["F"], r["strike"], r["tau"], r["r"], sig),
            axis=1
        ).values
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
    u: np.ndarray, F: float, tau: float, kappa: float, theta: float,
    sigma_v: float, rho: float, v0: float
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
        (kappa - rho * sigma_v * iu - d) * tau
        - 2.0 * np.log((1 - g * exp_dt) / (1 - g))
    )
    D = ((kappa - rho * sigma_v * iu - d) / (sigma_v**2)) * (
        (1 - exp_dt) / (1 - g * exp_dt)
    )
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
    F: float, K: float, tau: float, params: Dict[str, float], j: int,
    u_max: float = 100.0, n_points: int = 501
) -> float:
    """
    Risk-neutral probabilities P1 (j=1) and P2 (j=2):
      P2 = 1/2 + 1/π ∫_0^∞ Re[ e^{-i u ln K} φ(u) / (i u) ] du
      P1 = 1/2 + 1/π ∫_0^∞ Re[ e^{-i u ln K} φ(u - i) / (i u * φ(-i)) ] du
    """
    kappa = params["kappa"]
    theta = params["theta"]
    sigma_v = params["sigma_v"]
    rho = params["rho"]
    v0 = params["v0"]

    lnK = math.log(K)
    u = np.linspace(1e-6, u_max, n_points)
    du = u[1] - u[0]

    if j == 2:
        phi = _heston_cf(u, F, tau, kappa, theta, sigma_v, rho, v0)
        integrand = np.real(np.exp(-1j * u * lnK) * phi / (1j * u))
    else:
        phi_shift = _heston_cf(u - 1j, F, tau, kappa, theta, sigma_v, rho, v0)
        phi_mi = _heston_cf(np.array([-1j]), F, tau, kappa, theta, sigma_v, rho, v0)[0]
        integrand = np.real(
            np.exp(-1j * u * lnK) * (phi_shift / (1j * u * phi_mi))
        )

    integral = _simpson_integral(integrand, du)
    return float(0.5 + (1.0 / math.pi) * integral)


def heston_price_call(
    F: float, K: float, tau: float, r: float, params: Dict[str, float],
    u_max: float = 100.0, n_points: int = 501
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
    df_day_symbol: pd.DataFrame,
    min_parity_pairs: int = 2,
    tau_floor_days: int = 0,
    type: str = "american",
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
        tau_hdr = df.groupby(
            ["date", "symbol", "expiration"], as_index=False
        ).agg(tau=("tau", "first"))
        hdr = hdr.merge(tau_hdr, on=["date", "symbol", "expiration"], how="left")

        if has_r:
            r_hdr = df.groupby(
                ["date", "symbol", "expiration"], as_index=False
            ).agg(r=("r", "first"))
            hdr = hdr.merge(r_hdr, on=["date", "symbol", "expiration"], how="left")
        else:
            hdr["r"] = -np.log(
                np.clip(hdr["DF"].to_numpy(), 1e-12, None)
            ) / np.maximum(hdr["tau"].to_numpy(), 1e-12)

        # calls table shaped like downstream expects (C_mid etc.)
        calls = df.groupby(
            ["date", "symbol", "expiration", "tau", "strike"],
            as_index=False,
        ).agg(C_mid=("mid", "first"))
        calls = calls.merge(
            hdr[["date", "symbol", "expiration", "F", "DF", "r"]],
            on=["date", "symbol", "expiration"],
            how="left",
        )

        # market B76 IVs (calls only)
        calls["sigma_mkt_b76"] = calls.apply(
            lambda r: b76_iv_from_price(
                r["C_mid"], r["F"], r["strike"], r["tau"], r["r"]
            ),
            axis=1,
        )
        calls = calls.dropna(subset=["sigma_mkt_b76"])
        if calls.empty:
            return pd.DataFrame(), pd.DataFrame()

        calls["moneyness_F"] = calls["strike"] / calls["F"]
        calls["bucket"] = calls["tau"].apply(assign_bucket)

        # parity_df: simple header + coverage proxy (#strikes)
        n_pairs = (
            df.groupby(["date", "symbol", "expiration"])["strike"]
            .nunique()
            .rename("n_pairs")
            .reset_index()
        )
        parity_df = hdr.merge(
            n_pairs, on=["date", "symbol", "expiration"], how="left"
        )
        parity_df["n_pairs"] = parity_df["n_pairs"].fillna(0).astype(int)

        return calls, parity_df

    # =========================================================================
    # 'european' path: Calls–Puts pairing + parity inference (original behavior)
    # =========================================================================
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

    # infer F, DF, r per expiry via parity
    recs, parity_rows = [], []
    for (date, sym, exp), g in pairs.groupby(
        ["date", "symbol", "expiration"], sort=False
    ):
        if len(g) < min_parity_pairs:
            continue
        tau = float(g["tau"].iloc[0])
        res = parity_infer_F_DF(g[["strike", "C_mid", "P_mid"]])
        if res is None:
            continue
        F, DF = res
        r = -math.log(max(DF, 1e-12)) / max(tau, 1e-12)

        gg = pvt[
            (pvt["date"] == date)
            & (pvt["symbol"] == sym)
            & (pvt["expiration"] == exp)
        ].copy()
        gg["F"], gg["DF"], gg["r"] = F, DF, r
        recs.append(gg)
        parity_rows.append(
            {
                "date": date,
                "symbol": sym,
                "expiration": exp,
                "tau": tau,
                "F": F,
                "DF": DF,
                "r": r,
                "n_pairs": len(g),
            }
        )

    if not recs:
        return pd.DataFrame(), pd.DataFrame()

    calls = pd.concat(recs, ignore_index=True)

    # market B76 IVs (calls only)
    calls = calls[calls["C_mid"].notna()].copy()
    calls["sigma_mkt_b76"] = calls.apply(
        lambda r: b76_iv_from_price(
            r["C_mid"], r["F"], r["strike"], r["tau"], r["r"]
        ),
        axis=1,
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
        res = minimize(
            func,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200},
        )
        return res.x if res.success else x0
    x0 = np.array(x0, dtype=float)
    grids = []
    for (lo, hi), xi in zip(bounds, x0):
        span = hi - lo
        g = np.linspace(
            max(lo, xi - 0.2 * span), min(hi, xi + 0.2 * span), 7
        )
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
                    float(r0["F"]),
                    float(r0["strike"]),
                    float(r0["tau"]),
                    float(r0["r"]),
                    sigma,
                    lam,
                    muJ,
                    dJ,
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
    params = {
        "sigma": float(p[0]),
        "lam": float(p[1]),
        "muJ": float(p[2]),
        "deltaJ": float(p[3]),
    }
    sse_final = sse_vec(
        [params["sigma"], params["lam"], params["muJ"], params["deltaJ"]]
    )
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
        if (
            kappa <= 0
            or theta <= 0
            or sigma_v <= 0
            or not (-0.999 <= rho <= 0.0)
            or v0 <= 0
        ):
            return 1e18
        params = {
            "kappa": kappa,
            "theta": theta,
            "sigma_v": sigma_v,
            "rho": rho,
            "v0": v0,
        }
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
    bounds = [
        (0.05, 10.0),
        (1e-4, 2.0),
        (1e-3, 3.0),
        (-0.999, 0.0),
        (1e-4, 3.0),
    ]
    p = _minimize(sse_vec, x0, bounds)
    params = {
        "kappa": float(p[0]),
        "theta": float(p[1]),
        "sigma_v": float(p[2]),
        "rho": float(p[3]),
        "v0": float(p[4]),
    }
    sse_final = sse_vec(
        [
            params["kappa"],
            params["theta"],
            params["sigma_v"],
            params["rho"],
            params["v0"],
        ]
    )
    return params, sse_final


# ============================================================
# Data Preprocessing helpers
# ============================================================

def adapter_eur_calls_to_summarizer(calls_out: pd.DataFrame) -> pd.DataFrame:
    """
    Input (from preprocess_american_to_european):
      required cols: date, act_symbol, expiration, strike, C_eur, F, DF
    Output for summarizer (calls only, no parity step needed):
      date, act_symbol, expiration, strike, cp="Call", bid, ask, mid, F, DF
    """
    req = {"date", "act_symbol", "expiration", "strike", "C_eur", "F", "DF"}
    missing = req - set(calls_out.columns)
    if missing:
        raise ValueError(
            f"calls_out is missing required columns: {sorted(missing)}"
        )

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

    calls = (
        calls[(calls["mid"] > 0) & (calls["ask"] >= calls["bid"])][
            [
                "date",
                "act_symbol",
                "expiration",
                "strike",
                "cp",
                "bid",
                "ask",
                "mid",
                "F",
                "DF",
            ]
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
    bid_btc_col: str = "bid_price",      # bid quoted in BTC
    ask_btc_col: str = "ask_price",      # ask quoted in BTC
) -> pd.DataFrame:
    """
    Minimal mapper for summarizer. Outputs columns:
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
    ok = parsed.notna()
    if not ok.any():
        raise ValueError("No rows matched pattern like 'BTC-26DEC25-144000-C'.")

    base = df.loc[ok].copy()
    sym, exp, k, cp_full = zip(*parsed[ok])
    base["act_symbol"] = list(sym)
    base["expiration"] = list(exp)
    base["strike"] = list(k)
    base["cp"] = list(cp_full)

    # Observation date from real timestamp (UTC → naive date)
    base["date"] = (
        pd.to_datetime(base[ts_col], unit="ms", utc=True)
        .dt.tz_convert(None)
        .dt.date
    )

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

    out = base.sort_values(
        ["date", "act_symbol", "expiration", "strike", "cp"]
    ).reset_index(drop=True)
    return out


# ============================================================
# Dynamic hedging helpers: GBM simulation + generic delta-hedge
# (updated for ATM focus / period×bucket calibration / light SV)
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
        S[:, t + 1] = S[:, t] * np.exp(
            (r - 0.5 * sigma_true**2) * dt + sigma_true * math.sqrt(dt) * Z[:, t]
        )
    return S

def _simulate_bootstrap_paths_from_hist(
    S0: float,
    returns: np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 123,
) -> np.ndarray:
    """
    使用经验 log-return bootstrap 构造 world 路径：

      - returns: 1D 数组，为历史 log(S_{t+1}/S_t)。
      - 每条路径、每个时间步，从 returns 中有放回抽样一个 return。
      - 只用 returns 的历史形态塑造路径；T 仅用于下游贴现/剩余时间计算。

    如果 returns 为空，则退化为常数路径 S_t ≡ S0（极端兜底）。
    """
    rng = np.random.default_rng(seed)
    if returns is None or np.asarray(returns).size == 0:
        # 兜底：所有路径都是常数 S0
        S_flat = np.full((n_paths, n_steps + 1), float(S0), dtype=float)
        return S_flat

    returns = np.asarray(returns, dtype=float)
    idx = rng.integers(0, returns.size, size=(n_paths, n_steps))
    sampled = returns[idx]  # (n_paths, n_steps)
    logS = math.log(S0) + np.cumsum(sampled, axis=1)

    S_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    S_paths[:, 0] = S0
    S_paths[:, 1:] = np.exp(logS)
    return S_paths


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
    C0 = price_fn(S0, K, tau0, r)
    delta = delta_fn(S0, K, tau0, r)

    cash = C0 - delta * S0
    trading_cost = np.zeros_like(cash)

    for t in range(1, n_steps + 1):
        tau = max(T - t * dt, 0.0)
        St = S_paths[:, t]

        new_delta = delta_fn(St, K, tau, r)
        trade = new_delta - delta  # shares to buy (>0) or sell (<0)

        if friction > 0.0:
            cost = friction * np.abs(trade) * St
            trading_cost += cost
            cash -= cost

        cash -= trade * St
        delta = new_delta

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
# Dynamic hedging runner (ATM-focused, period×bucket calibration)
# ============================================================

def summarize_symbol_period_hedging(
    df_all: pd.DataFrame,
    symbol: str,
    type: str,
    start_date: str,
    end_date: str,
    buckets: List[int] = (28,),
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
    Dynamic hedging summary with:

      • Same preprocessing as IVRMSE (adapter_eur_calls_to_summarizer / preprocess_deribit)
      • ONE calibration per (period, bucket), using ATM-focused subset:
            moneyness_F in [0.9, 1.1] if available, else full bucket
      • Heston uses lighter numerical settings (u_max=60, n_points=201)
      • World paths are built from *historical* log-returns (bootstrap):
            – per (period, bucket), get daily F(t) series
            – compute log(F_{t+1}/F_t)
            – bootstrap these returns to generate S_paths
      • Metrics reported per day × bucket × moneyness-section × model:
            RMSE_hedge, avg_cost, shortfall_prob, shortfall_1pct
    """
    # 预处理：full period（例如整季），再筛选月份窗口
    if type == "american":
        df_pre_all = adapter_eur_calls_to_summarizer(df_all)
    else:
        df_pre_all = preprocess_deribit(df_all)

    sym_col = "act_symbol" if "act_symbol" in df_pre_all.columns else "symbol"
    df_pre_all["date"] = pd.to_datetime(df_pre_all["date"]).dt.normalize()

    mask = (
        (df_pre_all[sym_col] == symbol)
        & (df_pre_all["date"] >= pd.Timestamp(start_date))
        & (df_pre_all["date"] <= pd.Timestamp(end_date))
    )
    df = df_pre_all.loc[mask].copy()
    if df.empty:
        raise ValueError(
            f"No rows for {symbol} in [{start_date}, {end_date}] in hedging summariser."
        )

    # Bucket mapper
    def assign_bucket_centers(tau_years: float) -> str:
        days = tau_years * 365.0
        i = int(np.argmin([abs(days - c) for c in buckets]))
        return f"{buckets[i]}d"

    bucket_labels = [f"{d}d" for d in buckets]

    # 新的 moneyness sections（注意 whole sample / 子区间 一致）
    moneyness_sections = [
        "Whole sample",         # 0.8–1.2
        "Moneyness <1",         # 0.8–1.0
        "Moneyness >1",         # 1.0–1.2
        "Moneyness >1.03",      # 1.03–1.2
        "Moneyness [0.9,1.1]",  # 0.9–1.1 （额外看一眼）
    ]

    days = sorted(df["date"].unique())
    if not days:
        raise RuntimeError(
            "summarize_symbol_period_hedging: no trading days in filtered data."
        )

    # ---------------------------------------------------------------------
    # 第一遍：逐日生成 calls_by_day 和 period 内 pooled calls_all
    # ---------------------------------------------------------------------
    calls_by_day: Dict[pd.Timestamp, pd.DataFrame] = {}
    all_calls_list: List[pd.DataFrame] = []

    iterator = days
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(days, desc=f"prep hedging {symbol} {start_date}→{end_date}")
        except Exception:
            pass

    for day in iterator:
        df_day = df[df["date"] == day]
        calls, _ = prepare_calls_one_day_symbol(
            df_day,
            min_parity_pairs=min_parity_pairs,
            tau_floor_days=tau_floor_days,
            type=type,
        )
        if calls.empty:
            if print_daily:
                print(f"[{pd.Timestamp(day).date()}] no valid contracts after filters")
            continue

        calls["bucket"] = calls["tau"].apply(assign_bucket_centers)
        if "moneyness_F" not in calls.columns:
            calls["moneyness_F"] = calls["strike"] / calls["F"]

        calls_by_day[pd.Timestamp(day)] = calls
        all_calls_list.append(calls)

    if not all_calls_list:
        raise RuntimeError(
            "summarize_symbol_period_hedging: no (day,bucket) calls after preprocessing."
        )

    calls_all = pd.concat(all_calls_list, ignore_index=True)
    if "moneyness_F" not in calls_all.columns:
        calls_all["moneyness_F"] = calls_all["strike"] / calls_all["F"]

    # ---------------------------------------------------------------------
    # Period × bucket calibration with ATM band [0.9, 1.1]
    # （同之前，只是 world 后面改成历史 bootstrap）
    # ---------------------------------------------------------------------
    from lib.qlbs2.test_trained_model import QLBSModel
    from lib.rlop2.test_trained_model import RLOPModel

    calibrations_by_bucket: Dict[str, Dict[str, object]] = {}

    for bucket_label in bucket_labels:
        calls_b_all = calls_all[calls_all["bucket"] == bucket_label].copy()
        if calls_b_all.empty:
            continue

        atm_mask = (calls_b_all["moneyness_F"] >= 0.9) & (calls_b_all["moneyness_F"] <= 1.1)
        calls_b_atm = calls_b_all[atm_mask].copy()
        if len(calls_b_atm) >= 5:
            calib_set = calls_b_atm
        else:
            calib_set = calls_b_all

        calib_set = calib_set.dropna(
            subset=["sigma_mkt_b76", "C_mid", "F", "strike", "tau", "r"]
        )
        if calib_set.empty:
            continue

        sigma_true_period = float(calib_set["sigma_mkt_b76"].median())
        S0_ref = float(calib_set["F"].median())
        r_ref = float(calib_set["r"].median())

        # BS baseline
        sigma_bs = fit_sigma_bucket(calib_set) if run_bs else np.nan

        # JD baseline
        jd_params = None
        if run_jd:
            jd_params, _ = calibrate_jd_bucket(calib_set)

        # Heston baseline, lighter settings
        h_params = None
        if run_heston:
            h_params, _ = calibrate_heston_bucket(
                calib_set, u_max=60.0, n_points=201
            )

        # QLBS calibration
        Qmodel = None
        q_params = None
        if run_qlbs:
            risk_lambda_qlbs = 0.01
            time_to_expiries = calib_set["tau"].to_numpy()
            strikes = calib_set["strike"].to_numpy()
            observed_prices = calib_set["C_mid"].to_numpy()
            inv_price = 1.0 / np.power(np.clip(observed_prices, 1.0, None), 1.0)

            Qmodel = QLBSModel(
                is_call_option=True,
                checkpoint=f"trained_model/test8/risk_lambda={risk_lambda_qlbs:.1e}/policy_1.pt",
                anchor_T=28 / 252,
            )
            q_result = Qmodel.fit(
                spot=S0_ref,
                time_to_expiries=time_to_expiries,
                strikes=strikes,
                r=r_ref,
                risk_lambda=risk_lambda_qlbs,
                friction=friction,
                observed_prices=observed_prices,
                weights=inv_price,
                sigma_guess=0.3,
                mu_guess=0.0,
                n_epochs=2000,
            )
            q_params = {"sigma": q_result.sigma, "mu": q_result.mu}

        # RLOP calibration
        Rmodel = None
        rlop_params = None
        if run_rlop:
            risk_lambda_rlop = 0.10
            time_to_expiries = calib_set["tau"].to_numpy()
            strikes = calib_set["strike"].to_numpy()
            observed_prices = calib_set["C_mid"].to_numpy()
            inv_price = 1.0 / np.power(np.clip(observed_prices, 1.0, None), 1.0)

            Rmodel = RLOPModel(
                is_call_option=True,
                checkpoint="trained_model/testr9/policy_1.pt",
                anchor_T=28 / 252,
            )
            r_result = Rmodel.fit(
                spot=S0_ref,
                time_to_expiries=time_to_expiries,
                strikes=strikes,
                r=r_ref,
                risk_lambda=risk_lambda_rlop,
                friction=friction,
                observed_prices=observed_prices,
                weights=inv_price,
                sigma_guess=0.3,
                mu_guess=0.0,
                n_epochs=2000,
            )
            rlop_params = {"sigma": r_result.sigma, "mu": r_result.mu}

        calibrations_by_bucket[bucket_label] = {
            "sigma_true_period": sigma_true_period,
            "S0_ref": S0_ref,
            "r_ref": r_ref,
            "sigma_bs": sigma_bs,
            "jd_params": jd_params,
            "h_params": h_params,
            "Qmodel": Qmodel,
            "q_params": q_params,
            "Rmodel": Rmodel,
            "rlop_params": rlop_params,
        }

    if not calibrations_by_bucket:
        raise RuntimeError(
            "summarize_symbol_period_hedging: no bucket could be calibrated."
        )

    # ---------------------------------------------------------------------
    # 基于 period 内 calls_all，为每个 bucket 构造“历史 log-return”序列
    # 用 F(t) 的日度中位数作为 proxy 现货
    # ---------------------------------------------------------------------
    hist_returns_by_bucket: Dict[str, np.ndarray] = {}
    if "date" in calls_all.columns:
        calls_all_dates = calls_all.copy()
        calls_all_dates["date"] = pd.to_datetime(calls_all_dates["date"]).dt.normalize()
        for bucket_label in bucket_labels:
            sub = calls_all_dates[calls_all_dates["bucket"] == bucket_label]
            if sub.empty:
                continue
            g = (
                sub.groupby("date", as_index=False)["F"]
                .median()
                .sort_values("date")
            )
            S_series = g["F"].to_numpy(dtype=float)
            if S_series.size >= 2:
                rets = np.diff(np.log(S_series))
                hist_returns_by_bucket[bucket_label] = rets
            else:
                # 太短，先存一个空数组，后面 fallback 到 GBM world
                hist_returns_by_bucket[bucket_label] = np.array([], dtype=float)

    # ---------------------------------------------------------------------
    # 第二遍：使用 period×bucket calibration，跑基于历史 bootstrap world 的 hedging
    # ---------------------------------------------------------------------
    daily_rows: List[Dict[str, float]] = []

    iterator2 = list(enumerate(days))
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator2 = list(
                enumerate(
                    tqdm(days, desc=f"hedge {symbol} {start_date}→{end_date}")
                )
            )
        except Exception:
            pass

    for day_idx, day in iterator2:
        day_ts = pd.Timestamp(day)
        if day_ts not in calls_by_day:
            continue

        calls = calls_by_day[day_ts]

        for bucket_label in bucket_labels:
            if bucket_label not in calibrations_by_bucket:
                continue

            calls_b = calls[calls["bucket"] == bucket_label].copy()
            if calls_b.empty:
                continue

            calib = calibrations_by_bucket[bucket_label]

            if "moneyness_F" not in calls_b.columns:
                calls_b["moneyness_F"] = calls_b["strike"] / calls_b["F"]

            # 当日 ATM IV / F / r：用于 true sigma 和 world 起点
            atm_mask_day = (calls_b["moneyness_F"] >= 0.9) & (
                calls_b["moneyness_F"] <= 1.1
            )
            calls_b_atm_day = calls_b[atm_mask_day].copy()
            if not calls_b_atm_day.empty:
                sigma_true = float(calls_b_atm_day["sigma_mkt_b76"].median())
                S0_world = float(calls_b_atm_day["F"].median())
                r_world = float(calls_b_atm_day["r"].median())
            else:
                sigma_true = float(calib["sigma_true_period"])
                S0_world = float(calls_b["F"].iloc[0])
                r_world = float(calls_b["r"].iloc[0])

            center_days = int(bucket_label.rstrip("d"))
            T_world = center_days / 252.0
            n_steps_local = max(1, center_days)

            # --------- World: 历史 bootstrap vs GBM 兜底 ----------
            hist_rets = hist_returns_by_bucket.get(bucket_label, None)
            if hist_rets is not None and hist_rets.size > 0:
                S_paths = _simulate_bootstrap_paths_from_hist(
                    S0=S0_world,
                    returns=hist_rets,
                    T=T_world,
                    n_steps=n_steps_local,
                    n_paths=n_paths,
                    seed=seed + day_idx,
                )
            else:
                # 万一这个 bucket 历史太短，退回到 GBM world
                S_paths = _simulate_gbm_paths(
                    S0=S0_world,
                    r=r_world,
                    sigma_true=sigma_true,
                    T=T_world,
                    n_steps=n_steps_local,
                    n_paths=n_paths,
                    seed=seed + day_idx,
                )

            price_fns: Dict[str, callable] = {}
            delta_fns: Dict[str, callable] = {}

            # BS
            if run_bs and np.isfinite(calib["sigma_bs"]):
                sigma_bs = calib["sigma_bs"]

                def bs_price_vec(S_vec, K_, tau_, r_):
                    S_vec = np.atleast_1d(S_vec)
                    out = np.empty_like(S_vec, dtype=float)
                    for i, Si in enumerate(S_vec):
                        out[i] = b76_price_call(Si, K_, tau_, r_, sigma_bs)
                    return out

                price_fns["BS"] = bs_price_vec
                delta_fns["BS"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(
                    bs_price_vec, S_vec, K_, tau_, r_
                )

            # JD
            if run_jd and calib["jd_params"]:
                jp = calib["jd_params"]
                sigma_jd = jp["sigma"]
                lam_jd = jp["lam"]
                muJ_jd = jp["muJ"]
                dJ_jd = jp["deltaJ"]

                def jd_price_vec(S_vec, K_, tau_, r_):
                    S_vec = np.atleast_1d(S_vec)
                    out = np.empty_like(S_vec, dtype=float)
                    for i, Si in enumerate(S_vec):
                        out[i] = merton_price_call_b76(
                            Si, K_, tau_, r_, sigma_jd, lam_jd, muJ_jd, dJ_jd
                        )
                    return out

                price_fns["JD"] = jd_price_vec
                delta_fns["JD"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(
                    jd_price_vec, S_vec, K_, tau_, r_
                )

            # SV (Heston) with light numerical settings
            if run_heston and calib["h_params"]:
                h_params = calib["h_params"]

                def sv_price_vec(S_vec, K_, tau_, r_):
                    S_vec = np.atleast_1d(S_vec)
                    out = np.empty_like(S_vec, dtype=float)
                    for i, Si in enumerate(S_vec):
                        out[i] = heston_price_call(
                            Si, K_, tau_, r_, h_params, u_max=60.0, n_points=201
                        )
                    return out

                price_fns["SV"] = sv_price_vec
                delta_fns["SV"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(
                    sv_price_vec, S_vec, K_, tau_, r_
                )

            # QLBS
            if run_qlbs and calib["Qmodel"] is not None and calib["q_params"] is not None:
                Qmodel = calib["Qmodel"]
                risk_lambda_qlbs = 0.01
                sigma_q = calib["q_params"]["sigma"]
                mu_q = calib["q_params"]["mu"]

                def qlbs_price_vec(S_vec, K_, tau_, r_):
                    S_vec = np.atleast_1d(S_vec)
                    out = np.empty_like(S_vec, dtype=float)
                    for i, Si in enumerate(S_vec):
                        res = Qmodel.predict(
                            spot=Si,
                            time_to_expiries=np.array([tau_]),
                            strikes=np.array([K_]),
                            r=r_,
                            risk_lambda=risk_lambda_qlbs,
                            friction=friction,
                            sigma_fit=sigma_q,
                            mu_fit=mu_q,
                        )
                        out[i] = res.estimated_prices[0]
                    return out

                price_fns["QLBS"] = qlbs_price_vec
                delta_fns["QLBS"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(
                    qlbs_price_vec, S_vec, K_, tau_, r_
                )

            # RLOP
            if run_rlop and calib["Rmodel"] is not None and calib["rlop_params"] is not None:
                Rmodel = calib["Rmodel"]
                risk_lambda_rlop = 0.10
                sigma_r = calib["rlop_params"]["sigma"]
                mu_r = calib["rlop_params"]["mu"]

                def rlop_price_vec(S_vec, K_, tau_, r_):
                    S_vec = np.atleast_1d(S_vec)
                    out = np.empty_like(S_vec, dtype=float)
                    for i, Si in enumerate(S_vec):
                        res = Rmodel.predict(
                            spot=Si,
                            time_to_expiries=np.array([tau_]),
                            strikes=np.array([K_]),
                            r=r_,
                            risk_lambda=risk_lambda_rlop,
                            friction=friction,
                            sigma_fit=sigma_r,
                            mu_fit=mu_r,
                        )
                        out[i] = res.estimated_prices[0]
                    return out

                price_fns["RLOP"] = rlop_price_vec
                delta_fns["RLOP"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(
                    rlop_price_vec, S_vec, K_, tau_, r_
                )

            models = list(price_fns.keys())
            if not models:
                continue

            # -------------------------------------------------------------
            # Moneyness sections: 0.8–1.2 宽带 + 近 ATM [0.9,1.1] 额外一行
            # -------------------------------------------------------------
            for section_name in moneyness_sections:
                if section_name == "Whole sample":
                    sub = calls_b[
                        (calls_b["moneyness_F"] >= 0.8)
                        & (calls_b["moneyness_F"] <= 1.2)
                    ]
                elif section_name == "Moneyness <1":
                    sub = calls_b[
                        (calls_b["moneyness_F"] >= 0.8)
                        & (calls_b["moneyness_F"] < 1.0)
                    ]
                elif section_name == "Moneyness >1":
                    sub = calls_b[
                        (calls_b["moneyness_F"] > 1.0)
                        & (calls_b["moneyness_F"] <= 1.2)
                    ]
                elif section_name == "Moneyness >1.03":
                    sub = calls_b[
                        (calls_b["moneyness_F"] > 1.03)
                        & (calls_b["moneyness_F"] <= 1.2)
                    ]
                elif section_name == "Moneyness [0.9,1.1]":
                    sub = calls_b[
                        (calls_b["moneyness_F"] >= 0.9)
                        & (calls_b["moneyness_F"] <= 1.1)
                    ]
                else:
                    continue

                if sub.empty:
                    continue

                # 选该 section 中最靠近 median moneyness 的代表合约
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
                        "date": day_ts,
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

    if not daily_rows:
        raise RuntimeError(
            "summarize_symbol_period_hedging: no (day,bucket) rows produced."
        )

    daily = pd.DataFrame(daily_rows)

    metric_cols = ["RMSE_hedge", "avg_cost", "shortfall_prob", "shortfall_1pct"]

    equal_day_mean = (
        daily.groupby(["bucket", "moneyness_section", "model"], as_index=False)[
            metric_cols
        ].mean()
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

# ============================================================
# Publication-style hedging table
# ============================================================

def make_hedging_publication_table(
    hedge_res: Dict[str, pd.DataFrame],
    symbol: str,
    metric: str = "RMSE_hedge",   # "avg_cost", "shortfall_prob", "shortfall_1pct"
    buckets: List[int] = (28,),
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
        raise ValueError(
            "make_hedging_publication_table: equal_day_mean is empty."
        )

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


# ============================================================
# Wrappers for the four regimes (Feb-2020 & Jun-2025, 28d)
# ============================================================

def hedging_spy20():
    df = pd.read_csv("data/spy_preprocessed_calls_20q1.csv")

    hedge_res = summarize_symbol_period_hedging(
        df_all=df,
        symbol="SPY",
        type="american",
        start_date="2020-02-01",   # Feb 2020 only
        end_date="2020-02-29",
        buckets=[28],              # 1M bucket
        min_parity_pairs=4,
        tau_floor_days=3,
        n_paths=1000,
        friction=4e-3,
        seed=123,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        show_progress=True,
        print_daily=True,
        out_dir="SPY_20FEB_hedging_v3",
    )

    tbl_rmse = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="RMSE_hedge",
        buckets=[28],
        decimals=4,
        out_dir="SPY_20FEB_hedging_v3",
        basename="table_hedging",
    )
    tbl_cost = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="avg_cost",
        buckets=[28],
        decimals=6,
        out_dir="SPY_20FEB_hedging_v3",
        basename="table_hedging",
    )
    tbl_shortfall = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="shortfall_prob",
        buckets=[28],
        decimals=3,
        out_dir="SPY_20FEB_hedging_v3",
        basename="table_hedging",
    )
    tbl_shortfall_1pct = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="shortfall_1pct",
        buckets=[28],
        decimals=4,
        out_dir="SPY_20FEB_hedging_v3",
        basename="table_hedging",
    )

    print("Dynamic hedging RMSE table (SPY Feb 2020):")
    print(tbl_rmse)
    print(tbl_cost)
    print(tbl_shortfall)
    print(tbl_shortfall_1pct)

def hedging_spy25():
    df = pd.read_csv("data/spy_preprocessed_calls_25.csv")

    hedge_res = summarize_symbol_period_hedging(
        df_all=df,
        symbol="SPY",
        type="american",
        start_date="2025-06-01",   # Jun 2025 only
        end_date="2025-06-30",
        buckets=[28],
        min_parity_pairs=4,
        tau_floor_days=3,
        n_paths=1000,
        friction=4e-3,
        seed=123,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        show_progress=True,
        print_daily=True,
        out_dir="SPY_25JUN_hedging_v2",
    )

    tbl_rmse = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="RMSE_hedge",
        buckets=[28],
        decimals=4,
        out_dir="SPY_25JUN_hedging_v2",
        basename="table_hedging",
    )
    tbl_cost = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="avg_cost",
        buckets=[28],
        decimals=6,
        out_dir="SPY_25JUN_hedging_v2",
        basename="table_hedging",
    )
    tbl_shortfall = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="shortfall_prob",
        buckets=[28],
        decimals=3,
        out_dir="SPY_25JUN_hedging_v2",
        basename="table_hedging",
    )

    tbl_shortfall_1pct = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="shortfall_1pct",
        buckets=[28],
        decimals=3,
        out_dir="SPY_25JUN_hedging_v2",
        basename="table_hedging",
    )

    print("Dynamic hedging RMSE table (SPY Jun 2025):")
    print(tbl_rmse)
    print(tbl_cost)
    print(tbl_shortfall)
    print(tbl_shortfall_1pct)


def hedging_xop20():
    df = pd.read_csv("data/xop_preprocessed_calls_20q1.csv")

    hedge_res = summarize_symbol_period_hedging(
        df_all=df,
        symbol="XOP",
        type="american",
        start_date="2020-02-01",   # Feb 2020 only
        end_date="2020-02-29",
        buckets=[28],
        min_parity_pairs=4,
        tau_floor_days=3,
        n_paths=1000,
        friction=4e-3,
        seed=123,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        show_progress=True,
        print_daily=True,
        out_dir="XOP_20FEB_hedging_v2",
    )

    tbl_rmse = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="RMSE_hedge",
        buckets=[28],
        decimals=4,
        out_dir="XOP_20FEB_hedging_v2",
        basename="table_hedging",
    )
    tbl_cost = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="avg_cost",
        buckets=[28],
        decimals=6,
        out_dir="XOP_20FEB_hedging_v2",
        basename="table_hedging",
    )
    tbl_shortfall = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="shortfall_prob",
        buckets=[28],
        decimals=3,
        out_dir="XOP_20FEB_hedging_v2",
        basename="table_hedging",
    )
    tbl_shortfall_1pct = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="shortfall_1pct",
        buckets=[28],
        decimals=3,
        out_dir="XOP_20FEB_hedging_v2",
        basename="table_hedging",
    )

    print("Dynamic hedging RMSE table (XOP Feb 2020):")
    print(tbl_rmse)
    print(tbl_cost)
    print(tbl_shortfall)
    print(tbl_shortfall_1pct)


def hedging_xop25():
    df = pd.read_csv("data/xop_preprocessed_calls_25.csv")

    hedge_res = summarize_symbol_period_hedging(
        df_all=df,
        symbol="XOP",
        type="american",
        start_date="2025-06-01",   # Jun 2025 only
        end_date="2025-06-30",
        buckets=[28],
        min_parity_pairs=4,
        tau_floor_days=3,
        n_paths=1000,
        friction=4e-3,
        seed=123,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        show_progress=True,
        print_daily=True,
        out_dir="XOP_25JUN_hedging_v2",
    )

    tbl_rmse = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="RMSE_hedge",
        buckets=[28],
        decimals=4,
        out_dir="XOP_25JUN_hedging_v2",
        basename="table_hedging",
    )
    tbl_cost = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="avg_cost",
        buckets=[28],
        decimals=6,
        out_dir="XOP_25JUN_hedging_v2",
        basename="table_hedging",
    )
    tbl_shortfall = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="shortfall_prob",
        buckets=[28],
        decimals=3,
        out_dir="XOP_25JUN_hedging_v2",
        basename="table_hedging",
    )
    tbl_shortfall_1pct = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="shortfall_1pct",
        buckets=[28],
        decimals=3,
        out_dir="XOP_25JUN_hedging_v2",
        basename="table_hedging",
    )

    print("Dynamic hedging RMSE table (XOP Jun 2025):")
    print(tbl_rmse)
    print(tbl_cost)
    print(tbl_shortfall)
    print(tbl_shortfall_1pct)


# ============================================================
# Unit test harness
# ============================================================

class Test(TestCase):
    def test_hedging_runs(self):
        # Comment/uncomment as needed
        hedging_spy20()
        hedging_spy25()
        hedging_xop20()
        hedging_xop25()
