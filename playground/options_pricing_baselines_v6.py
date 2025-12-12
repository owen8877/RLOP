from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pathlib import Path
import re
from datetime import datetime

# ============================================================
# Basic Black–76 pricing and IV inversion
# ============================================================

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


# ============================================================
# Parity helper (for European data path if needed)
# ============================================================

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


# ============================================================
# Bucket mapping
# ============================================================

def assign_bucket_centers(tau_years: float, centers_days: List[int]) -> str:
    """Map τ (years) to nearest maturity center in days, e.g. '28d'."""
    days = tau_years * 365.0
    idx = int(np.argmin([abs(days - c) for c in centers_days]))
    return f"{centers_days[idx]}d"


# ============================================================
# Data preprocessing helpers (American & Deribit)
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
        raise ValueError(f"calls_out is missing required columns: {sorted(missing)}")

    df = calls_out.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()

    calls = df[["date", "act_symbol", "expiration", "strike", "C_eur", "F", "DF"]].copy()
    calls["cp"] = "Call"
    calls["mid"] = pd.to_numeric(calls["C_eur"], errors="coerce")

    calls["bid"] = calls["mid"]
    calls["ask"] = calls["mid"]

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
    bid_btc_col: str = "bid_price",      # bid quoted in BTC
    ask_btc_col: str = "ask_price",      # ask quoted in BTC
) -> pd.DataFrame:
    """
    Minimal mapper for summarizer. Outputs columns:
      ['date','symbol','act_symbol','expiration','strike','cp','bid','ask','mid']
    """

    def _parse(name: str):
        m = re.match(r"^([A-Z]+)-(\d{2}[A-Z]{3}\d{2})-([0-9]+)-(C|P)$", str(name).strip())
        if not m:
            return None
        sym, exp_s, k_s, cp = m.groups()
        try:
            exp_dt = datetime.strptime(exp_s, "%d%b%y")
            strike = float(k_s)
            cp_full = "Call" if cp == "C" else "Put"
            return sym, exp_dt, strike, cp_full
        except Exception:
            return None

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

    base["date"] = pd.to_datetime(base[ts_col], unit="ms", utc=True).dt.tz_convert(None).dt.date

    spot = pd.to_numeric(base[spot_col], errors="coerce")
    bid = pd.to_numeric(base[bid_btc_col], errors="coerce") * spot
    ask = pd.to_numeric(base[ask_btc_col], errors="coerce") * spot
    mid = (bid + ask) / 2.0

    keep = (bid > 0) & (ask > 0) & (ask >= bid)
    base = base.loc[keep, ["date", "act_symbol", "expiration", "strike", "cp"]].copy()
    base["bid"] = bid.loc[base.index].values
    base["ask"] = ask.loc[base.index].values
    base["mid"] = mid.loc[base.index].values

    out = base.sort_values(["date", "act_symbol", "expiration", "strike", "cp"]).reset_index(drop=True)
    return out


def prepare_calls_one_day_symbol(
    df_day_symbol: pd.DataFrame,
    min_parity_pairs: int = 2,
    tau_floor_days: int = 0,
    type: str = "american",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    type='american' : Fast-path. Use provided F/DF/(r); prefer C_eur as market mid if available.
    type='european' : Pair Calls & Puts, infer F/DF via parity.
    Returns (calls_df, parity_df).
    """
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

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["tau"] = (df["expiration"] - df["date"]).dt.days / 365.0

    if "cp" not in df.columns:
        df["cp"] = "Call"

    if "mid" not in df.columns:
        df["bid"] = pd.to_numeric(df.get("bid", np.nan), errors="coerce")
        df["ask"] = pd.to_numeric(df.get("ask", np.nan), errors="coerce")
        df["mid"] = (df["bid"].clip(lower=0) + df["ask"].clip(lower=0)) / 2.0
    else:
        df["mid"] = pd.to_numeric(df["mid"], errors="coerce")

    if type.lower() == "american" and "C_eur" in df.columns:
        df["mid"] = pd.to_numeric(df["C_eur"], errors="coerce")

    df = df[df["strike"].notna() & df["mid"].notna() & (df["mid"] > 0) & (df["tau"] > 0)]
    if "ask" in df.columns and "bid" in df.columns:
        df = df[(df["ask"] >= df["bid"]) & (df["bid"] >= 0)]
    if tau_floor_days and tau_floor_days > 0:
        df = df[df["tau"] * 365.0 >= tau_floor_days]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if type.lower() == "american":
        df = df[df["cp"].str.upper().eq("CALL")]

        if not {"F", "DF"}.issubset(df.columns):
            raise ValueError("type='american' expects F and DF columns in the dataset.")
        df["F"] = pd.to_numeric(df["F"], errors="coerce")
        df["DF"] = pd.to_numeric(df["DF"], errors="coerce")

        has_r = "r" in df.columns
        if has_r:
            df["r"] = pd.to_numeric(df["r"], errors="coerce")

        hdr = df.groupby(["date", "symbol", "expiration"], as_index=False).agg(
            F=("F", "first"),
            DF=("DF", "first"),
        )
        tau_hdr = df.groupby(["date", "symbol", "expiration"], as_index=False).agg(
            tau=("tau", "first")
        )
        hdr = hdr.merge(tau_hdr, on=["date", "symbol", "expiration"], how="left")

        if has_r:
            r_hdr = df.groupby(["date", "symbol", "expiration"], as_index=False).agg(
                r=("r", "first")
            )
            hdr = hdr.merge(r_hdr, on=["date", "symbol", "expiration"], how="left")
        else:
            hdr["r"] = -np.log(np.clip(hdr["DF"].to_numpy(), 1e-12, None)) / np.maximum(
                hdr["tau"].to_numpy(), 1e-12
            )

        calls = df.groupby(["date", "symbol", "expiration", "tau", "strike"], as_index=False).agg(
            C_mid=("mid", "first")
        )
        calls = calls.merge(
            hdr[["date", "symbol", "expiration", "F", "DF", "r"]],
            on=["date", "symbol", "expiration"],
            how="left",
        )

        # market B76 IVs
        calls["sigma_mkt_b76"] = calls.apply(
            lambda r: b76_iv_from_price(r["C_mid"], r["F"], r["strike"], r["tau"], r["r"]),
            axis=1,
        )
        calls = calls.dropna(subset=["sigma_mkt_b76"])
        if calls.empty:
            return pd.DataFrame(), pd.DataFrame()

        calls["moneyness_F"] = calls["strike"] / calls["F"]

        n_pairs = df.groupby(["date", "symbol", "expiration"])["strike"].nunique().rename("n_pairs").reset_index()
        parity_df = hdr.merge(n_pairs, on=["date", "symbol", "expiration"], how="left")
        parity_df["n_pairs"] = parity_df["n_pairs"].fillna(0).astype(int)

        return calls, parity_df

    # european path
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

    recs = []
    parity_rows = []
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
    calls = calls[calls["C_mid"].notna()].copy()

    calls["sigma_mkt_b76"] = calls.apply(
        lambda r: b76_iv_from_price(r["C_mid"], r["F"], r["strike"], r["tau"], r["r"]),
        axis=1,
    )
    calls = calls.dropna(subset=["sigma_mkt_b76"])
    if calls.empty:
        return pd.DataFrame(), pd.DataFrame()

    calls["moneyness_F"] = calls["strike"] / calls["F"]
    parity_df = pd.DataFrame(parity_rows)
    return calls, parity_df


# ============================================================
# JD and Heston pricing + calibration (weighted, period×bucket)
# ============================================================

from math import exp, sqrt, log

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
    """
    if tau <= 0:
        DF = exp(-r * max(tau, 0.0))
        return DF * max(F - K, 0.0)

    k = exp(muJ + 0.5 * deltaJ * deltaJ) - 1.0
    F_adj = F * exp(-lam * k * tau)
    L = lam * tau

    p = exp(-L)
    price = p * b76_price_call(F_adj, K, tau, r, sigma)
    cum = p
    n = 0
    while cum < 1 - eps_tail and n < n_max:
        n += 1
        p = p * (L / n)
        sigma_n = sqrt(sigma * sigma + (n * deltaJ * deltaJ) / max(tau, 1e-12))
        F_n = F_adj * exp(n * muJ)
        price += p * b76_price_call(F_n, K, tau, r, sigma_n)
        cum += p
    return float(price)


def _heston_cf(
    u: np.ndarray, F: float, tau: float, kappa: float, theta: float, sigma_v: float, rho: float, v0: float
) -> np.ndarray:
    """
    Heston characteristic function φ(u) for log-price under forward measure.
    """
    x = log(F)
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
    """Simpson’s rule; falls back to trapz if <3 points."""
    n = len(fx)
    if n < 3:
        return float(np.trapz(fx, dx=dx))
    if n % 2 == 0:
        fx = fx[:-1]
        n -= 1
    S = fx[0] + fx[-1] + 4.0 * fx[1:-1:2].sum() + 2.0 * fx[2:-2:2].sum()
    return float((dx / 3.0) * S)


def _heston_prob(
    F: float,
    K: float,
    tau: float,
    params: Dict[str, float],
    j: int,
    u_max: float = 60.0,
    n_points: int = 201,
) -> float:
    """
    Risk-neutral probabilities P1 (j=1) and P2 (j=2) under Heston.
    Low-resolution integration (u_max, n_points) for speed.
    """
    kappa, theta, sigma_v, rho, v0 = (
        params["kappa"],
        params["theta"],
        params["sigma_v"],
        params["rho"],
        params["v0"],
    )
    lnK = log(K)
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
    F: float,
    K: float,
    tau: float,
    r: float,
    params: Dict[str, float],
    u_max: float = 60.0,
    n_points: int = 201,
) -> float:
    DF = exp(-r * tau)
    P1 = _heston_prob(F, K, tau, params, j=1, u_max=u_max, n_points=n_points)
    P2 = _heston_prob(F, K, tau, params, j=2, u_max=u_max, n_points=n_points)
    return DF * (F * P1 - K * P2)


# ============================================================
# Optimization helper (SciPy if available, else small grid)
# ============================================================

try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _minimize(func, x0, bounds):
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


# ============================================================
# ATM weights (0.9–1.1) for period×bucket calibration
# ============================================================

def _compute_atm_weights(calls: pd.DataFrame) -> np.ndarray:
    """
    Compute near-ATM weights:
      - restrict to moneyness_F already in [0.9,1.1] outside
      - inv_price = 1 / max(C_mid, 1)
      - Gaussian in standardized moneyness.
    """
    C = np.asarray(calls["C_mid"].to_numpy(dtype=float))
    m = np.asarray(calls["moneyness_F"].to_numpy(dtype=float))

    inv_price = 1.0 / np.maximum(C, 1.0)

    # standardize log-moneyness; use 20% vol as scale
    tau = np.asarray(calls["tau"].to_numpy(dtype=float))
    z = np.log(np.maximum(m, 1e-12)) / (np.sqrt(np.maximum(tau, 1e-6)) * 0.2)
    gaussian = np.exp(-0.5 * z * z)

    w = inv_price * gaussian
    w = np.maximum(w, 1e-8)
    return w


# ============================================================
# Weighted calibrations
# ============================================================

def fit_sigma_bucket_weighted(calls_bucket: pd.DataFrame, weights: np.ndarray) -> float:
    """
    One-parameter BS/B76 baseline with weights:
      minimize sum_i w_i * (C_model_i - C_mid_i)^2
    """
    if calls_bucket.empty:
        return float("nan")
    w = np.asarray(weights, dtype=float)
    C_mid = calls_bucket["C_mid"].to_numpy(dtype=float)
    F = calls_bucket["F"].to_numpy(dtype=float)
    K = calls_bucket["strike"].to_numpy(dtype=float)
    tau = calls_bucket["tau"].to_numpy(dtype=float)
    r = calls_bucket["r"].to_numpy(dtype=float)

    def sse(sig: float) -> float:
        if sig <= 0:
            return 1e18
        prices = np.array(
            [b76_price_call(F[i], K[i], tau[i], r[i], sig) for i in range(len(C_mid))],
            dtype=float,
        )
        err = prices - C_mid
        return float(np.dot(w, err * err))

    grid = np.geomspace(0.05, 2.0, 40)
    best = min(grid, key=sse)
    a = max(best / 3, 1e-4)
    b = min(best * 3, 3.0)
    phi = (1 + math.sqrt(5)) / 2
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


def calibrate_jd_bucket_weighted(
    calls_bucket: pd.DataFrame,
    weights: np.ndarray,
    muJ_fixed: float = -0.02,
    deltaJ_fixed: float = 0.15,
) -> Tuple[Dict[str, float], float]:
    """
    Weighted JD calibration on bucket, with fixed jump size distribution
    (muJ, deltaJ) to avoid overfitting. Free params: sigma, lam.
    """
    if calls_bucket.empty:
        return {}, float("inf")

    w = np.asarray(weights, dtype=float)
    C_mid = calls_bucket["C_mid"].to_numpy(dtype=float)
    F = calls_bucket["F"].to_numpy(dtype=float)
    K = calls_bucket["strike"].to_numpy(dtype=float)
    tau = calls_bucket["tau"].to_numpy(dtype=float)
    r = calls_bucket["r"].to_numpy(dtype=float)

    atm_iv = float(np.median(calls_bucket["sigma_mkt_b76"].values))
    x0 = np.array([max(0.02, atm_iv), 0.1], dtype=float)  # sigma, lam
    bounds = [(0.01, 3.0), (0.0, 1.0)]

    def sse_vec(p):
        sigma, lam = p
        if sigma <= 0 or lam < 0:
            return 1e18
        prices = np.array(
            [
                merton_price_call_b76(F[i], K[i], tau[i], r[i], sigma, lam, muJ_fixed, deltaJ_fixed)
                for i in range(len(C_mid))
            ],
            dtype=float,
        )
        err = prices - C_mid
        return float(np.dot(w, err * err))

    p_opt = _minimize(sse_vec, x0, bounds)
    params = {
        "sigma": float(p_opt[0]),
        "lam": float(p_opt[1]),
        "muJ": float(muJ_fixed),
        "deltaJ": float(deltaJ_fixed),
    }
    sse_final = sse_vec([params["sigma"], params["lam"]])
    return params, sse_final


def calibrate_heston_bucket_weighted(
    calls_bucket: pd.DataFrame,
    weights: np.ndarray,
    u_max: float = 60.0,
    n_points: int = 201,
) -> Tuple[Dict[str, float], float]:
    """
    Weighted Heston calibration with reduced dimension:
      free params: kappa, theta, sigma_v, rho
      v0 is tied to theta (v0 = theta) to avoid over-parameterization.
    """
    if calls_bucket.empty:
        return {}, float("inf")

    w = np.asarray(weights, dtype=float)
    C_mid = calls_bucket["C_mid"].to_numpy(dtype=float)
    F = calls_bucket["F"].to_numpy(dtype=float)
    K = calls_bucket["strike"].to_numpy(dtype=float)
    tau = calls_bucket["tau"].to_numpy(dtype=float)
    r = calls_bucket["r"].to_numpy(dtype=float)

    atm_iv = float(np.median(calls_bucket["sigma_mkt_b76"].values))
    v0_0 = max(1e-4, atm_iv * atm_iv)
    # kappa, theta, sigma_v, rho
    x0 = np.array([2.0, v0_0, 0.5, -0.5], dtype=float)
    bounds = [
        (0.05, 10.0),   # kappa
        (1e-4, 2.0),    # theta
        (1e-3, 3.0),    # sigma_v
        (-0.8, 0.0),    # rho
    ]

    def sse_vec(p):
        kappa, theta, sigma_v, rho = p
        if kappa <= 0 or theta <= 0 or sigma_v <= 0 or not (-0.8 <= rho <= 0.0):
            return 1e18
        params = {"kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho, "v0": theta}
        prices = np.array(
            [
                heston_price_call(F[i], K[i], tau[i], r[i], params, u_max=u_max, n_points=n_points)
                for i in range(len(C_mid))
            ],
            dtype=float,
        )
        err = prices - C_mid
        return float(np.dot(w, err * err))

    p_opt = _minimize(sse_vec, x0, bounds)
    params = {
        "kappa": float(p_opt[0]),
        "theta": float(p_opt[1]),
        "sigma_v": float(p_opt[2]),
        "rho": float(p_opt[3]),
        "v0": float(p_opt[1]),  # v0 = theta
    }
    sse_final = sse_vec([params["kappa"], params["theta"], params["sigma_v"], params["rho"]])
    return params, sse_final


# ============================================================
# GBM simulation + generic delta-hedge
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

    C0 = price_fn(S0, K, tau0, r)
    delta = delta_fn(S0, K, tau0, r)

    cash = C0 - delta * S0
    trading_cost = np.zeros_like(cash)

    for t in range(1, n_steps + 1):
        tau = max(T - t * dt, 0.0)
        St = S_paths[:, t]

        new_delta = delta_fn(St, K, tau, r)
        trade = new_delta - delta

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
# Hedging summariser: period×bucket calibration + ATM weights
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
    print_info: bool = True,
    out_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Dynamic hedging summary with period×bucket calibration:

      - Preprocess to calls (adapter_eur_calls_to_summarizer / preprocess_deribit)
      - Filter to [symbol, start_date..end_date]
      - Build calls for each day, then pool by bucket over the entire period
      - For each bucket:
          * restrict to near-ATM: 0.9 <= K/F <= 1.1
          * world volatility = median market IV over this set
          * world S0, r      = medians of F, r over this set
          * calibrate BS / JD / SV on this set with ATM weights
          * calibrate QLBS / RLOP on a representative day within this set
          * simulate one GBM world and delta-hedge a short call for:
              - Whole sample
              - Moneyness <1
              - Moneyness >1
              - Moneyness >1.03
    """
    # --- Preprocess calls, same entry logic ---
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

    # --- Build calls_all with sigma_mkt_b76 etc. day by day ---
    centers = list(buckets)
    all_calls = []
    days = sorted(df["date"].unique())
    iterator = days
    if show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(days, desc=f"prep {symbol} {start_date}→{end_date}")
        except Exception:
            pass

    for day in iterator:
        df_day = df[df["date"] == day]
        calls_day, _ = prepare_calls_one_day_symbol(
            df_day,
            min_parity_pairs=min_parity_pairs,
            tau_floor_days=tau_floor_days,
            type=type,
        )
        if calls_day.empty:
            continue
        calls_day["bucket"] = calls_day["tau"].apply(lambda t: assign_bucket_centers(t, centers))
        all_calls.append(calls_day)

    if not all_calls:
        raise RuntimeError("No valid calls after preprocessing and filtering.")

    calls_all = pd.concat(all_calls, ignore_index=True)

    # restrict to near-ATM [0.9, 1.1] for hedging universe
    calls_all = calls_all[(calls_all["moneyness_F"] >= 0.9) & (calls_all["moneyness_F"] <= 1.1)].copy()
    if calls_all.empty:
        raise RuntimeError("No near-ATM contracts (0.9–1.1 moneyness) available for hedging.")

    bucket_labels = [f"{d}d" for d in buckets]

    # --- containers for world + params per bucket ---
    world_settings: Dict[str, Dict[str, float]] = {}
    bs_params: Dict[str, float] = {}
    jd_params: Dict[str, Dict[str, float]] = {}
    heston_params: Dict[str, Dict[str, float]] = {}
    qlbs_fits: Dict[str, Dict[str, object]] = {}
    rlop_fits: Dict[str, Dict[str, object]] = {}

    # Import RL models lazily (only if needed)
    QLBSModel = None
    RLOPModel = None
    if run_qlbs:
        from lib.qlbs2.test_trained_model import QLBSModel as _Q
        QLBSModel = _Q
    if run_rlop:
        from lib.rlop2.test_trained_model import RLOPModel as _R
        RLOPModel = _R

    # --- period×bucket calibrations ---
    for dlab in bucket_labels:
        calls_b = calls_all[calls_all["bucket"] == dlab].copy()
        if calls_b.empty:
            if print_info:
                print(f"[{symbol} {start_date}..{end_date}] bucket {dlab}: no near-ATM contracts.")
            continue

        # world parameters from this bucket
        sigma_true = float(np.median(calls_b["sigma_mkt_b76"].values))
        S0_world = float(np.median(calls_b["F"].values))
        r_world = float(np.median(calls_b["r"].values))
        center_days = int(dlab.rstrip("d"))
        T_world = center_days / 252.0
        n_steps_world = max(1, center_days)

        world_settings[dlab] = {
            "S0": S0_world,
            "r": r_world,
            "sigma_true": sigma_true,
            "T": T_world,
            "n_steps": n_steps_world,
        }

        # weights for calibration
        w = _compute_atm_weights(calls_b)

        # BS calibration
        if run_bs:
            sig_bs = fit_sigma_bucket_weighted(calls_b, w)
            bs_params[dlab] = sig_bs
            if print_info:
                print(f"[{symbol} {dlab}] BS sigma={sig_bs:.4f}")

        # JD calibration
        if run_jd:
            jd_p, _ = calibrate_jd_bucket_weighted(calls_b, w)
            jd_params[dlab] = jd_p
            if print_info:
                print(f"[{symbol} {dlab}] JD params={jd_p}")

        # Heston calibration
        if run_heston:
            h_p, _ = calibrate_heston_bucket_weighted(calls_b, w)
            heston_params[dlab] = h_p
            if print_info:
                print(f"[{symbol} {dlab}] Heston params={h_p}")

        # RL calibration on representative day (still period×bucket: once per bucket)
        days_b = sorted(calls_b["date"].unique())
        rep_day = days_b[len(days_b) // 2]
        calib = calls_b[calls_b["date"] == rep_day].copy()
        if calib.empty:
            calib = calls_b.copy()

        if run_qlbs and QLBSModel is not None:
            risk_lambda_q = 0.01
            spot_q = float(calib["F"].iloc[0])
            r_q = float(calib["r"].iloc[0])
            time_to_expiries = calib["tau"].to_numpy(dtype=float)
            strikes = calib["strike"].to_numpy(dtype=float)
            observed_prices = calib["C_mid"].to_numpy(dtype=float)
            w_q = _compute_atm_weights(calib)

            Qmodel = QLBSModel(
                is_call_option=True,
                checkpoint=f"trained_model/test8/risk_lambda={risk_lambda_q:.1e}/policy_1.pt",
                anchor_T=28 / 252,
            )
            q_res = Qmodel.fit(
                spot=spot_q,
                time_to_expiries=time_to_expiries,
                strikes=strikes,
                r=r_q,
                risk_lambda=risk_lambda_q,
                friction=friction,
                observed_prices=observed_prices,
                weights=w_q,
                sigma_guess=0.3,
                mu_guess=0.0,
                n_epochs=2000,
            )
            qlbs_fits[dlab] = {
                "model": Qmodel,
                "sigma": float(q_res.sigma),
                "mu": float(q_res.mu),
                "risk_lambda": risk_lambda_q,
                "friction": friction,
                "r": r_q,
            }
            if print_info:
                print(f"[{symbol} {dlab}] QLBS sigma={q_res.sigma:.4f}, mu={q_res.mu:.4f}")

        if run_rlop and RLOPModel is not None:
            risk_lambda_r = 0.10
            spot_r = float(calib["F"].iloc[0])
            r_r = float(calib["r"].iloc[0])
            time_to_expiries = calib["tau"].to_numpy(dtype=float)
            strikes = calib["strike"].to_numpy(dtype=float)
            observed_prices = calib["C_mid"].to_numpy(dtype=float)
            w_r = _compute_atm_weights(calib)

            Rmodel = RLOPModel(
                is_call_option=True,
                checkpoint="trained_model/testr9/policy_1.pt",
                anchor_T=28 / 252,
            )
            r_res = Rmodel.fit(
                spot=spot_r,
                time_to_expiries=time_to_expiries,
                strikes=strikes,
                r=r_r,
                risk_lambda=risk_lambda_r,
                friction=friction,
                observed_prices=observed_prices,
                weights=w_r,
                sigma_guess=0.3,
                mu_guess=0.0,
                n_epochs=2000,
            )
            rlop_fits[dlab] = {
                "model": Rmodel,
                "sigma": float(r_res.sigma),
                "mu": float(r_res.mu),
                "risk_lambda": risk_lambda_r,
                "friction": friction,
                "r": r_r,
            }
            if print_info:
                print(f"[{symbol} {dlab}] RLOP sigma={r_res.sigma:.4f}, mu={r_res.mu:.4f}")

    # --- Hedging metrics per bucket & moneyness section ---
    moneyness_sections = [
        "Whole sample",
        "Moneyness <1",
        "Moneyness >1",
        "Moneyness >1.03",
    ]

    daily_rows: List[Dict[str, float]] = []

    for dlab in bucket_labels:
        if dlab not in world_settings:
            continue
        calls_b = calls_all[calls_all["bucket"] == dlab].copy()
        if calls_b.empty:
            continue

        world = world_settings[dlab]
        S0_world = world["S0"]
        r_world = world["r"]
        sigma_true = world["sigma_true"]
        T_world = world["T"]
        n_steps_world = int(world["n_steps"])

        # simulate one GBM world per bucket
        S_paths = _simulate_gbm_paths(
            S0=S0_world,
            r=r_world,
            sigma_true=sigma_true,
            T=T_world,
            n_steps=n_steps_world,
            n_paths=n_paths,
            seed=seed,
        )

        # price functions per model (vectorized over S)
        price_fns = {}
        delta_fns = {}

        if run_bs and dlab in bs_params:
            sigma_bs = bs_params[dlab]

            def bs_price_vec(S_vec, K_, tau_, r_):
                S_vec = np.atleast_1d(S_vec)
                out = np.empty_like(S_vec, dtype=float)
                for i, Si in enumerate(S_vec):
                    out[i] = b76_price_call(Si, K_, tau_, r_, sigma_bs)
                return out

            price_fns["BS"] = bs_price_vec
            delta_fns["BS"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(bs_price_vec, S_vec, K_, tau_, r_)

        if run_jd and dlab in jd_params:
            jp = jd_params[dlab]
            sigma_jd = jp["sigma"]
            lam_jd = jp["lam"]
            muJ_jd = jp["muJ"]
            dJ_jd = jp["deltaJ"]

            def jd_price_vec(S_vec, K_, tau_, r_):
                S_vec = np.atleast_1d(S_vec)
                out = np.empty_like(S_vec, dtype=float)
                for i, Si in enumerate(S_vec):
                    out[i] = merton_price_call_b76(Si, K_, tau_, r_, sigma_jd, lam_jd, muJ_jd, dJ_jd)
                return out

            price_fns["JD"] = jd_price_vec
            delta_fns["JD"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(jd_price_vec, S_vec, K_, tau_, r_)

        if run_heston and dlab in heston_params:
            hp = heston_params[dlab]

            def sv_price_vec(S_vec, K_, tau_, r_):
                S_vec = np.atleast_1d(S_vec)
                out = np.empty_like(S_vec, dtype=float)
                for i, Si in enumerate(S_vec):
                    out[i] = heston_price_call(Si, K_, tau_, r_, hp, u_max=60.0, n_points=201)
                return out

            price_fns["SV"] = sv_price_vec
            delta_fns["SV"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(sv_price_vec, S_vec, K_, tau_, r_)

        if run_qlbs and dlab in qlbs_fits:
            qfit = qlbs_fits[dlab]
            Qmodel = qfit["model"]
            sigma_q = qfit["sigma"]
            mu_q = qfit["mu"]
            risk_lambda_q = qfit["risk_lambda"]
            friction_q = qfit["friction"]
            r_q = qfit["r"]

            def qlbs_price_vec(S_vec, K_, tau_, r_ignored):
                S_vec = np.atleast_1d(S_vec)
                out = np.empty_like(S_vec, dtype=float)
                for i, Si in enumerate(S_vec):
                    res = Qmodel.predict(
                        spot=Si,
                        time_to_expiries=np.array([tau_]),
                        strikes=np.array([K_]),
                        r=r_q,
                        risk_lambda=risk_lambda_q,
                        friction=friction_q,
                        sigma_fit=sigma_q,
                        mu_fit=mu_q,
                    )
                    out[i] = res.estimated_prices[0]
                return out

            price_fns["QLBS"] = qlbs_price_vec
            delta_fns["QLBS"] = lambda S_vec, K_, tau_, r_: _delta_from_pricer(
                qlbs_price_vec, S_vec, K_, tau_, r_
            )

        if run_rlop and dlab in rlop_fits:
            rfit = rlop_fits[dlab]
            Rmodel = rfit["model"]
            sigma_r = rfit["sigma"]
            mu_r = rfit["mu"]
            risk_lambda_r = rfit["risk_lambda"]
            friction_r = rfit["friction"]
            r_r = rfit["r"]

            def rlop_price_vec(S_vec, K_, tau_, r_ignored):
                S_vec = np.atleast_1d(S_vec)
                out = np.empty_like(S_vec, dtype=float)
                for i, Si in enumerate(S_vec):
                    res = Rmodel.predict(
                        spot=Si,
                        time_to_expiries=np.array([tau_]),
                        strikes=np.array([K_]),
                        r=r_r,
                        risk_lambda=risk_lambda_r,
                        friction=friction_r,
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

        # hedging sections
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

            # representative strike: median moneyness within this slice
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
                    "date": pd.Timestamp(start_date),  # single period, we tag start_date
                    "symbol": symbol,
                    "bucket": dlab,
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
        raise RuntimeError("summarize_symbol_period_hedging: no hedging rows produced.")

    daily = pd.DataFrame(daily_rows)
    metric_cols = ["RMSE_hedge", "avg_cost", "shortfall_prob", "shortfall_1pct"]

    equal_day_mean = (
        daily.groupby(["bucket", "moneyness_section", "model"], as_index=False)[metric_cols]
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


# ============================================================
# Publication-style hedging table (same interface as before)
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


# ============================================================
# Convenience runners for your four SPY/XOP regimes
# (using Feb-2020 and Jun-2025 single-month windows, bucket=28)
# ============================================================

def hedging_spy20_feb():
    df = pd.read_csv("data/spy_preprocessed_calls_20q1.csv")

    hedge_res = summarize_symbol_period_hedging(
        df_all=df,
        symbol="SPY",
        type="american",
        start_date="2020-02-01",
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
        out_dir="SPY_20FEB_hedging_v1",
    )

    tbl_rmse = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="RMSE_hedge",
        buckets=[28],
        decimals=4,
        out_dir="SPY_20FEB_hedging_v1",
        basename="table_hedging",
    )
    tbl_cost = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="avg_cost",
        buckets=[28],
        decimals=6,
        out_dir="SPY_20FEB_hedging_v1",
        basename="table_hedging",
    )
    tbl_shortfall = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="shortfall_prob",
        buckets=[28],
        decimals=3,
        out_dir="SPY_20FEB_hedging_v1",
        basename="table_hedging",
    )

    print("Dynamic hedging RMSE table (SPY Feb-2020):")
    print(tbl_rmse)
    print(tbl_cost)
    print(tbl_shortfall)


def hedging_spy25_jun():
    df = pd.read_csv("data/spy_preprocessed_calls_25.csv")

    hedge_res = summarize_symbol_period_hedging(
        df_all=df,
        symbol="SPY",
        type="american",
        start_date="2025-06-01",
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
        out_dir="SPY_25JUN_hedging_v1",
    )

    tbl_rmse = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="RMSE_hedge",
        buckets=[28],
        decimals=4,
        out_dir="SPY_25JUN_hedging_v1",
        basename="table_hedging",
    )
    tbl_cost = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="avg_cost",
        buckets=[28],
        decimals=6,
        out_dir="SPY_25JUN_hedging_v1",
        basename="table_hedging",
    )
    tbl_shortfall = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="shortfall_prob",
        buckets=[28],
        decimals=3,
        out_dir="SPY_25JUN_hedging_v1",
        basename="table_hedging",
    )

    print("Dynamic hedging RMSE table (SPY Jun-2025):")
    print(tbl_rmse)
    print(tbl_cost)
    print(tbl_shortfall)


def hedging_xop20_feb():
    df = pd.read_csv("data/xop_preprocessed_calls_20q1.csv")

    hedge_res = summarize_symbol_period_hedging(
        df_all=df,
        symbol="XOP",
        type="american",
        start_date="2020-02-01",
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
        out_dir="XOP_20FEB_hedging_v1",
    )

    tbl_rmse = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="RMSE_hedge",
        buckets=[28],
        decimals=4,
        out_dir="XOP_20FEB_hedging_v1",
        basename="table_hedging",
    )
    tbl_cost = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="avg_cost",
        buckets=[28],
        decimals=6,
        out_dir="XOP_20FEB_hedging_v1",
        basename="table_hedging",
    )
    tbl_shortfall = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="shortfall_prob",
        buckets=[28],
        decimals=3,
        out_dir="XOP_20FEB_hedging_v1",
        basename="table_hedging",
    )

    print("Dynamic hedging RMSE table (XOP Feb-2020):")
    print(tbl_rmse)
    print(tbl_cost)
    print(tbl_shortfall)


def hedging_xop25_jun():
    df = pd.read_csv("data/xop_preprocessed_calls_25.csv")

    hedge_res = summarize_symbol_period_hedging(
        df_all=df,
        symbol="XOP",
        type="american",
        start_date="2025-06-01",
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
        out_dir="XOP_25JUN_hedging_v1",
    )

    tbl_rmse = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="RMSE_hedge",
        buckets=[28],
        decimals=4,
        out_dir="XOP_25JUN_hedging_v1",
        basename="table_hedging",
    )
    tbl_cost = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="avg_cost",
        buckets=[28],
        decimals=6,
        out_dir="XOP_25JUN_hedging_v1",
        basename="table_hedging",
    )
    tbl_shortfall = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="shortfall_prob",
        buckets=[28],
        decimals=3,
        out_dir="XOP_25JUN_hedging_v1",
        basename="table_hedging",
    )

    print("Dynamic hedging RMSE table (XOP Jun-2025):")
    print(tbl_rmse)
    print(tbl_cost)
    print(tbl_shortfall)


if __name__ == "__main__":
    # Example: uncomment one of these to run:
    hedging_spy20_feb()
    hedging_spy25_jun()
    # hedging_xop20_feb()
    # hedging_xop25_jun()
    pass
