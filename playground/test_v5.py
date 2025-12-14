from __future__ import annotations

import math
import pickle
from unittest import TestCase
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try SciPy for calibration; fall back to local grid if missing
try:
    from scipy.optimize import minimize

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ============================================================
# Black–76 pricing & IV inversion
# ============================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def b76_price_call(F: float, K: float, tau: float, r: float, sigma: float) -> float:
    """Black–76 call price with forward F and continuous rate r."""
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
    *,
    clamp_to_intrinsic: bool,
    enforce_noarb_bounds: bool,
    tol_price: float = 1e-10,
    tol_iv: float = 1e-8,
    max_iter: int = 120,
    lo: float = 1e-5,
    hi: float = 5.0,
) -> Optional[float]:
    """
    B76 implied vol from price via robust bisection.

    IMPORTANT:
      - For MARKET prices: clamp_to_intrinsic=True is OK (robustness).
      - For MODEL prices: clamp_to_intrinsic=False and enforce_noarb_bounds=True
        so invalid prices are flagged rather than silently repaired.

    No-arb bounds for a call under B76:
      intrinsic = DF * max(F-K,0)
      upper     = DF * F
    """
    if not (F > 0 and K > 0 and tau >= 0 and math.isfinite(target) and math.isfinite(r)):
        return None

    DF = math.exp(-r * tau)
    intrinsic = DF * max(F - K, 0.0)
    upper = DF * F

    if enforce_noarb_bounds:
        if (target < intrinsic - tol_price) or (target > upper + tol_price):
            return None

    t = float(target)
    if clamp_to_intrinsic:
        t = max(t, intrinsic)

    def f(sig: float) -> float:
        return b76_price_call(F, K, tau, r, sig) - t

    a, b = lo, hi
    fa, fb = f(a), f(b)

    # Expand bracket if needed
    k = 0
    while fa * fb > 0 and k < 30:
        b *= 1.5
        fb = f(b)
        k += 1
        if b > 50.0:
            break
    if fa * fb > 0:
        return None

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < tol_iv or 0.5 * (b - a) < 1e-10:
            return float(c)
        if fa * fc <= 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return float(0.5 * (a + b))


# ============================================================
# Put–call parity: infer F and DF (european path)
# ============================================================

def parity_infer_F_DF(df_pairs: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """
    Infer forward F and discount DF from (C - P) vs K:
      y = C - P = DF*(F - K) = a + b*(-K) where b=DF and F=a/DF
    """
    y = (df_pairs["C_mid"].values - df_pairs["P_mid"].values).astype(float)
    X = df_pairs["strike"].values.astype(float)
    if len(y) < 2:
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
# Bucketing & baseline sigma fit
# ============================================================

def assign_bucket_nearest(tau_years: float, centers_days: List[int]) -> str:
    days = float(tau_years) * 365.0
    idx = int(np.argmin([abs(days - c) for c in centers_days]))
    return f"{centers_days[idx]}d"


def fit_sigma_bucket(df_bucket: pd.DataFrame) -> float:
    """
    1-parameter BS/B76 baseline: choose σ minimizing price SSE.
    Coarse geom-grid + golden-section refinement.
    """
    if df_bucket.empty:
        return float("nan")

    F = df_bucket["F"].to_numpy(dtype=float)
    K = df_bucket["strike"].to_numpy(dtype=float)
    tau = df_bucket["tau"].to_numpy(dtype=float)
    r = df_bucket["r"].to_numpy(dtype=float)
    C = df_bucket["C_mid"].to_numpy(dtype=float)

    def sse(sig: float) -> float:
        if sig <= 0:
            return 1e18
        prices = np.array([b76_price_call(Fi, Ki, ti, ri, sig) for Fi, Ki, ti, ri in zip(F, K, tau, r)], dtype=float)
        err = prices - C
        return float(np.dot(err, err))

    grid = np.geomspace(0.03, 2.5, 48)
    best = float(min(grid, key=sse))
    a = max(best / 3.0, 1e-5)
    b = min(best * 3.0, 5.0)

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    invphi = 1.0 / phi

    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    fc, fd = sse(c), sse(d)
    for _ in range(70):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi
            fc = sse(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi
            fd = sse(d)
    return float((a + b) / 2.0)


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
    while cum < 1.0 - eps_tail and n < n_max:
        n += 1
        p = p * (L / n)
        sigma_n = math.sqrt(sigma * sigma + (n * deltaJ * deltaJ) / max(tau, 1e-12))
        F_n = F_adj * math.exp(n * muJ)
        price += p * b76_price_call(F_n, K, tau, r, sigma_n)
        cum += p
    return float(price)


# ============================================================
# Heston SV (fast-but-reasonable default integration)
# ============================================================

def _heston_cf(
    u: np.ndarray, F: float, tau: float, kappa: float, theta: float, sigma_v: float, rho: float, v0: float
) -> np.ndarray:
    """Heston CF under forward measure (x=log(F)). 'Little Heston trap' form."""
    x = math.log(F)
    iu = 1j * u
    d = np.sqrt((rho * sigma_v * iu - kappa) ** 2 + sigma_v**2 * (iu + u * u))
    g = (kappa - rho * sigma_v * iu - d) / (kappa - rho * sigma_v * iu + d)
    exp_dt = np.exp(-d * tau)
    C = (kappa * theta / (sigma_v**2)) * (
        (kappa - rho * sigma_v * iu - d) * tau - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = ((kappa - rho * sigma_v * iu - d) / (sigma_v**2)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    return np.exp(C + D * v0 + iu * x)


def _simpson_integral(fx: np.ndarray, dx: float) -> float:
    n = len(fx)
    if n < 3:
        return float(np.trapz(fx, dx=dx))
    if n % 2 == 0:
        fx = fx[:-1]
        n -= 1
    S = fx[0] + fx[-1] + 4.0 * fx[1:-1:2].sum() + 2.0 * fx[2:-2:2].sum()
    return float((dx / 3.0) * S)


def _heston_prob(
    F: float, K: float, tau: float, params: Dict[str, float], j: int, u_max: float, n_points: int
) -> float:
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
    F: float, K: float, tau: float, r: float, params: Dict[str, float], u_max: float, n_points: int
) -> float:
    DF = math.exp(-r * tau)
    P1 = _heston_prob(F, K, tau, params, j=1, u_max=u_max, n_points=n_points)
    P2 = _heston_prob(F, K, tau, params, j=2, u_max=u_max, n_points=n_points)
    return float(DF * (F * P1 - K * P2))


# ============================================================
# Data prep: one day & symbol -> calls with market IVs
# ============================================================

def prepare_calls_one_day_symbol(
    df_day_symbol: pd.DataFrame,
    *,
    centers_days: List[int],
    min_parity_pairs: int = 2,
    tau_floor_days: int = 0,
    type: str = "american",
    r_from_df: bool = True,
    r_mismatch_tol: float = 5e-4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      calls_df with columns:
        date,symbol,expiration,tau,strike,C_mid,F,DF,r,(optional S,PV_div,q_hat),
        sigma_mkt_b76,moneyness_F,bucket
      parity_df:
        american: per-expiry header + n_strikes
        european: inferred per-expiry header + n_pairs

    american fast-path expects F and DF columns; uses C_eur as mid if present.
    """

    def _normalize_cp(x) -> str:
        if isinstance(x, str):
            y = x.strip().upper()
            if y in ("C", "CALL"):
                return "Call"
            if y in ("P", "PUT"):
                return "Put"
        return str(x)

    df = df_day_symbol.copy()

    # Normalize schema (tolerant)
    rename_map = {
        "act_symbol": "symbol",
        "symbol": "symbol",
        "date": "date",
        "expiration": "expiration",
        "expiry": "expiration",
        "strike": "strike",
        "call_put": "cp",
        "cp": "cp",
        "bid": "bid",
        "ask": "ask",
        "mid": "mid",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "symbol" not in df.columns:
        raise ValueError("Expected 'symbol' or 'act_symbol' column.")
    if "date" not in df.columns or "expiration" not in df.columns or "strike" not in df.columns:
        raise ValueError("Expected columns among: date, expiration, strike.")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

    # cp
    if "cp" not in df.columns:
        df["cp"] = "Call"
    df["cp"] = df["cp"].apply(_normalize_cp)

    # tau
    df["tau"] = (df["expiration"] - df["date"]).dt.days / 365.0

    # mid
    if "C_eur" in df.columns:
        # if you precomputed European call premium
        df["C_eur"] = pd.to_numeric(df["C_eur"], errors="coerce")
    if "mid" not in df.columns:
        df["bid"] = pd.to_numeric(df.get("bid", np.nan), errors="coerce")
        df["ask"] = pd.to_numeric(df.get("ask", np.nan), errors="coerce")
        df["mid"] = (df["bid"].clip(lower=0) + df["ask"].clip(lower=0)) / 2.0
    else:
        df["mid"] = pd.to_numeric(df["mid"], errors="coerce")

    # If american fast-path and C_eur present, use it as the market mid
    if type.lower() == "american" and "C_eur" in df.columns:
        df["mid"] = df["C_eur"]

    # hygiene filters
    df = df[df["strike"].notna() & df["mid"].notna() & (df["mid"] > 0) & (df["tau"] > 0)].copy()
    if "ask" in df.columns and "bid" in df.columns:
        df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
        df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
        df = df[(df["ask"] >= df["bid"]) & (df["bid"] >= 0)]
    if tau_floor_days and tau_floor_days > 0:
        df = df[df["tau"] * 365.0 >= tau_floor_days]

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # =========================
    # AMERICAN fast-path
    # =========================
    if type.lower() == "american":
        df = df[df["cp"] == "Call"].copy()

        if not {"F", "DF"}.issubset(df.columns):
            raise ValueError("type='american' expects F and DF columns in the dataset.")
        df["F"] = pd.to_numeric(df["F"], errors="coerce")
        df["DF"] = pd.to_numeric(df["DF"], errors="coerce")

        # optional
        if "S" in df.columns:
            df["S"] = pd.to_numeric(df["S"], errors="coerce")
        if "PV_div" in df.columns:
            df["PV_div"] = pd.to_numeric(df["PV_div"], errors="coerce")
        else:
            df["PV_div"] = 0.0
        if "r" in df.columns:
            df["r"] = pd.to_numeric(df["r"], errors="coerce")

        # per-expiry header (median is stable under duplicates)
        hdr = df.groupby(["date", "symbol", "expiration"], as_index=False).agg(
            F=("F", "median"),
            DF=("DF", "median"),
            tau=("tau", "median"),
            PV_div=("PV_div", "median"),
            S=("S", "median") if "S" in df.columns else ("F", "median"),
            r=("r", "median") if "r" in df.columns else ("DF", "median"),
        )

        # Always derive r from DF for pricing consistency (fix)
        DFv = np.clip(hdr["DF"].to_numpy(dtype=float), 1e-12, None)
        tauv = np.maximum(hdr["tau"].to_numpy(dtype=float), 1e-12)
        r_df = -np.log(DFv) / tauv

        if "r" in df.columns and hdr["r"].notna().any() and (not r_from_df):
            # optional: keep provided r, but warn by storing mismatch
            r_prov = hdr["r"].to_numpy(dtype=float)
            hdr["r_from_df"] = r_df
            hdr["r_mismatch"] = np.abs(r_prov - r_df)
        else:
            hdr["r"] = r_df
            hdr["r_from_df"] = r_df
            hdr["r_mismatch"] = 0.0

        # Infer S if missing: S ≈ F*DF + PV_div
        S_infer = hdr["F"].to_numpy(dtype=float) * hdr["DF"].to_numpy(dtype=float) + hdr["PV_div"].to_numpy(dtype=float)
        S0 = hdr["S"].to_numpy(dtype=float)
        hdr["S"] = np.where(np.isfinite(S0), S0, S_infer)

        # implied q_hat via F = S * exp((r-q)tau)
        Fh = np.maximum(hdr["F"].to_numpy(dtype=float), 1e-12)
        Sh = np.maximum(hdr["S"].to_numpy(dtype=float), 1e-12)
        rh = hdr["r"].to_numpy(dtype=float)
        hdr["q_hat"] = rh - np.log(Fh / Sh) / tauv

        # collapse calls per strike (median for stability)
        calls = df.groupby(["date", "symbol", "expiration", "tau", "strike"], as_index=False).agg(
            C_mid=("mid", "median")
        )
        calls = calls.merge(
            hdr[["date", "symbol", "expiration", "F", "DF", "r", "S", "PV_div", "q_hat", "r_from_df", "r_mismatch"]],
            on=["date", "symbol", "expiration"],
            how="left",
        )

        # market IVs: robust clamp to intrinsic is OK
        calls["sigma_mkt_b76"] = [
            b76_iv_from_price(
                float(Ci), float(Fi), float(Ki), float(ti), float(ri),
                clamp_to_intrinsic=True, enforce_noarb_bounds=False
            )
            for Ci, Fi, Ki, ti, ri in zip(calls["C_mid"], calls["F"], calls["strike"], calls["tau"], calls["r"])
        ]
        calls = calls.dropna(subset=["sigma_mkt_b76"]).copy()
        if calls.empty:
            return pd.DataFrame(), pd.DataFrame()

        calls["moneyness_F"] = calls["strike"] / calls["F"]
        calls["bucket"] = calls["tau"].apply(lambda t: assign_bucket_nearest(t, centers_days))

        # parity_df: header + coverage proxy
        n_strikes = calls.groupby(["date", "symbol", "expiration"])["strike"].nunique().rename("n_strikes").reset_index()
        parity_df = hdr.merge(n_strikes, on=["date", "symbol", "expiration"], how="left")
        parity_df["n_strikes"] = parity_df["n_strikes"].fillna(0).astype(int)

        # store mismatch flags for debugging
        parity_df["r_badflag"] = (parity_df["r_mismatch"] > r_mismatch_tol).astype(int)

        return calls, parity_df

    # =========================
    # EUROPEAN path
    # =========================
    pvt = df.pivot_table(
        index=["date", "symbol", "expiration", "tau", "strike"],
        columns="cp",
        values="mid",
        aggfunc="median",
    ).reset_index()
    pvt = pvt.rename(columns={"Call": "C_mid", "Put": "P_mid"})
    pairs = pvt.dropna(subset=["C_mid", "P_mid"])
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()

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
        parity_rows.append({"date": date, "symbol": sym, "expiration": exp, "tau": tau, "F": F, "DF": DF, "r": r, "n_pairs": len(g)})

    if not recs:
        return pd.DataFrame(), pd.DataFrame()

    calls = pd.concat(recs, ignore_index=True)
    calls = calls[calls["C_mid"].notna()].copy()
    calls["sigma_mkt_b76"] = [
        b76_iv_from_price(
            float(Ci), float(Fi), float(Ki), float(ti), float(ri),
            clamp_to_intrinsic=True, enforce_noarb_bounds=False
        )
        for Ci, Fi, Ki, ti, ri in zip(calls["C_mid"], calls["F"], calls["strike"], calls["tau"], calls["r"])
    ]
    calls = calls.dropna(subset=["sigma_mkt_b76"]).copy()
    if calls.empty:
        return pd.DataFrame(), pd.DataFrame()

    calls["moneyness_F"] = calls["strike"] / calls["F"]
    calls["bucket"] = calls["tau"].apply(lambda t: assign_bucket_nearest(t, centers_days))
    parity_df = pd.DataFrame(parity_rows)
    return calls, parity_df


# ============================================================
# Calibration helpers (JD & Heston)
# ============================================================

def _minimize(func, x0, bounds):
    """SciPy L-BFGS-B; fallback to small local grid if SciPy missing."""
    if _HAVE_SCIPY:
        res = minimize(func, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
        return res.x if res.success else np.array(x0, dtype=float)
    x0 = np.array(x0, dtype=float)
    grids = []
    for (lo, hi), xi in zip(bounds, x0):
        span = hi - lo
        g = np.linspace(max(lo, xi - 0.2 * span), min(hi, xi + 0.2 * span), 7)
        grids.append(g)
    mesh = np.meshgrid(*grids, indexing="ij")
    cand = np.stack([m.ravel() for m in mesh], axis=1)
    vals = np.array([func(p) for p in cand], dtype=float)
    return cand[int(np.argmin(vals))]


def calibrate_jd_bucket(calls_bucket: pd.DataFrame) -> Tuple[Dict[str, float], float]:
    """Fit JD params (sigma, lam, muJ, deltaJ) by price SSE on the bucket cross-section."""
    if calls_bucket.empty:
        return {}, float("inf")
    atm_iv = float(np.median(calls_bucket["sigma_mkt_b76"].values))

    F = calls_bucket["F"].to_numpy(dtype=float)
    K = calls_bucket["strike"].to_numpy(dtype=float)
    tau = calls_bucket["tau"].to_numpy(dtype=float)
    r = calls_bucket["r"].to_numpy(dtype=float)
    C = calls_bucket["C_mid"].to_numpy(dtype=float)

    def sse_vec(p):
        sigma, lam, muJ, dJ = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        if sigma <= 0 or lam < 0 or dJ <= 0:
            return 1e18
        prices = np.array(
            [merton_price_call_b76(Fi, Ki, ti, ri, sigma, lam, muJ, dJ) for Fi, Ki, ti, ri in zip(F, K, tau, r)],
            dtype=float,
        )
        err = prices - C
        return float(np.dot(err, err))

    x0 = np.array([max(0.02, atm_iv), 0.10, -0.02, 0.10], dtype=float)
    bounds = [(0.01, 3.0), (0.0, 5.0), (-0.5, 0.5), (0.01, 1.0)]
    p = _minimize(sse_vec, x0, bounds)
    params = {"sigma": float(p[0]), "lam": float(p[1]), "muJ": float(p[2]), "deltaJ": float(p[3])}
    sse_final = sse_vec([params["sigma"], params["lam"], params["muJ"], params["deltaJ"]])
    return params, float(sse_final)


def calibrate_heston_bucket(
    calls_bucket: pd.DataFrame, *, u_max: float, n_points: int
) -> Tuple[Dict[str, float], float]:
    """Fit Heston params (kappa, theta, sigma_v, rho, v0) by price SSE."""
    if calls_bucket.empty:
        return {}, float("inf")
    atm_iv = float(np.median(calls_bucket["sigma_mkt_b76"].values))
    v0_0 = max(1e-4, atm_iv * atm_iv)

    F = calls_bucket["F"].to_numpy(dtype=float)
    K = calls_bucket["strike"].to_numpy(dtype=float)
    tau = calls_bucket["tau"].to_numpy(dtype=float)
    r = calls_bucket["r"].to_numpy(dtype=float)
    C = calls_bucket["C_mid"].to_numpy(dtype=float)

    def sse_vec(p):
        kappa, theta, sigma_v, rho, v0 = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
        if kappa <= 0 or theta <= 0 or sigma_v <= 0 or not (-0.999 <= rho <= 0.999) or v0 <= 0:
            return 1e18
        params = {"kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho, "v0": v0}
        prices = np.array(
            [heston_price_call(Fi, Ki, ti, ri, params, u_max=u_max, n_points=n_points) for Fi, Ki, ti, ri in zip(F, K, tau, r)],
            dtype=float,
        )
        err = prices - C
        return float(np.dot(err, err))

    x0 = np.array([2.0, v0_0, 0.5, -0.5, v0_0], dtype=float)
    bounds = [(0.05, 10.0), (1e-4, 2.0), (1e-3, 3.0), (-0.999, 0.999), (1e-4, 3.0)]
    p = _minimize(sse_vec, x0, bounds)
    params = {"kappa": float(p[0]), "theta": float(p[1]), "sigma_v": float(p[2]), "rho": float(p[3]), "v0": float(p[4])}
    sse_final = sse_vec([params["kappa"], params["theta"], params["sigma_v"], params["rho"], params["v0"]])
    return params, float(sse_final)


# ============================================================
# IVRMSE computations (equal-day only)
# ============================================================

def _rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")


def _ivrmse_vs_constant_sigma(calls_bucket: pd.DataFrame, sig_hat: float) -> Dict[str, float]:
    """RMSE between market IV and constant σ̂, by moneyness slice."""
    if calls_bucket.empty or not np.isfinite(sig_hat):
        return {"whole": np.nan, "<1": np.nan, ">1": np.nan, ">1.03": np.nan}

    iv_mkt = calls_bucket["sigma_mkt_b76"].to_numpy(dtype=float)
    m = calls_bucket["moneyness_F"].to_numpy(dtype=float)
    diff = iv_mkt - float(sig_hat)

    out = {
        "whole": _rmse(diff),
        "<1": _rmse(diff[m < 1.0]),
        ">1": _rmse(diff[m > 1.0]),
        ">1.03": _rmse(diff[m > 1.03]),
    }
    return out


def _ivrmse_vs_model_prices(
    calls_bucket: pd.DataFrame,
    price_fn,  # price_fn(F,K,tau,r) -> price
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Price with a model → invert to B76 IV → compare to market IVs.
    Strict for model prices: no intrinsic clamp; enforce no-arb bounds.

    Returns:
      (rmse_dict, diag_dict)
      rmse_dict keys: whole, <1, >1, >1.03
      diag_dict keys: bad_frac_whole, n_valid_whole, n_total_whole
    """
    if calls_bucket.empty:
        rmse_dict = {"whole": np.nan, "<1": np.nan, ">1": np.nan, ">1.03": np.nan}
        diag = {"bad_frac_whole": np.nan, "n_valid_whole": 0.0, "n_total_whole": 0.0}
        return rmse_dict, diag

    F = calls_bucket["F"].to_numpy(dtype=float)
    K = calls_bucket["strike"].to_numpy(dtype=float)
    tau = calls_bucket["tau"].to_numpy(dtype=float)
    r = calls_bucket["r"].to_numpy(dtype=float)
    iv_mkt = calls_bucket["sigma_mkt_b76"].to_numpy(dtype=float)
    m = calls_bucket["moneyness_F"].to_numpy(dtype=float)

    iv_model = np.full_like(iv_mkt, np.nan, dtype=float)
    ok = np.zeros_like(iv_mkt, dtype=bool)

    for i in range(len(calls_bucket)):
        pm = float(price_fn(float(F[i]), float(K[i]), float(tau[i]), float(r[i])))

        s = b76_iv_from_price(
            pm, float(F[i]), float(K[i]), float(tau[i]), float(r[i]),
            clamp_to_intrinsic=False, enforce_noarb_bounds=True
        )
        if s is not None and np.isfinite(s):
            iv_model[i] = float(s)
            ok[i] = True

    n_total = float(len(ok))
    n_valid = float(ok.sum())
    bad_frac = float((n_total - n_valid) / n_total) if n_total > 0 else np.nan

    if ok.sum() == 0:
        rmse_dict = {"whole": np.nan, "<1": np.nan, ">1": np.nan, ">1.03": np.nan}
        diag = {"bad_frac_whole": bad_frac, "n_valid_whole": n_valid, "n_total_whole": n_total}
        return rmse_dict, diag

    diff = iv_model[ok] - iv_mkt[ok]
    rmse_dict = {
        "whole": _rmse(diff),
        "<1": _rmse((iv_model[(ok) & (m < 1.0)] - iv_mkt[(ok) & (m < 1.0)])),
        ">1": _rmse((iv_model[(ok) & (m > 1.0)] - iv_mkt[(ok) & (m > 1.0)])),
        ">1.03": _rmse((iv_model[(ok) & (m > 1.03)] - iv_mkt[(ok) & (m > 1.03)])),
    }
    diag = {"bad_frac_whole": bad_frac, "n_valid_whole": n_valid, "n_total_whole": n_total}
    return rmse_dict, diag


# ============================================================
# Optional Deribit preprocessor (if you still want BTC later)
# ============================================================

def preprocess_deribit(
    df: pd.DataFrame,
    *,
    instr_col: str = "instrument_name",
    ts_col: str = "creation_timestamp",
    spot_col: str = "underlying_price",
    bid_btc_col: str = "bid_price",
    ask_btc_col: str = "ask_price",
) -> pd.DataFrame:
    """
    Outputs columns for prepare_calls_one_day_symbol european path:
      date, symbol, expiration, strike, cp, bid, ask, mid
    """
    import re

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
    base["symbol"] = list(sym)
    base["expiration"] = list(exp)
    base["strike"] = list(k)
    base["cp"] = list(cp_full)

    base["date"] = pd.to_datetime(base[ts_col], unit="ms", utc=True).dt.tz_convert(None).dt.date

    spot = pd.to_numeric(base[spot_col], errors="coerce")
    bid = pd.to_numeric(base[bid_btc_col], errors="coerce") * spot
    ask = pd.to_numeric(base[ask_btc_col], errors="coerce") * spot
    mid = (bid + ask) / 2.0

    keep = (bid > 0) & (ask > 0) & (ask >= bid)
    out = base.loc[keep, ["date", "symbol", "expiration", "strike", "cp"]].copy()
    out["bid"] = bid.loc[out.index].values
    out["ask"] = ask.loc[out.index].values
    out["mid"] = mid.loc[out.index].values
    out = out.sort_values(["date", "symbol", "expiration", "strike", "cp"]).reset_index(drop=True)
    return out


# ============================================================
# Main IVRMSE summarizer (equal-day mean only)
# ============================================================

@dataclass
class IVRMSEConfig:
    symbol: str
    type: str  # "american" or "european"
    start_date: str
    end_date: str
    centers_days: List[int]
    min_parity_pairs: int = 4
    tau_floor_days: int = 3

    run_bs: bool = True
    run_jd: bool = True
    run_heston: bool = True
    run_qlbs: bool = True
    run_rlop: bool = True

    # Heston integration speed/accuracy
    heston_u_max: float = 60.0
    heston_n_points: int = 201

    # Optional day sampling for quick runs (None -> all days)
    n_rep_days: Optional[int] = None
    rep_day_mode: str = "even"  # "even" or "middle"

    # Saving
    out_dir: Optional[str] = None
    save_predictions: bool = True  # saves predicted_prices_long/wide if out_dir set
    print_daily: bool = True
    show_progress: bool = True

    # RL knobs (kept consistent with your earlier code)
    qlbs_risk_lambda: float = 0.01
    qlbs_checkpoint_tmpl: str = "trained_model/test8/risk_lambda={rl:.1e}/policy_1.pt"
    qlbs_epochs: int = 2000

    rlop_risk_lambda: float = 0.10
    rlop_checkpoint: str = "trained_model/testr9/policy_1.pt"
    rlop_epochs: int = 2000

    # RL state convention (important): "forward" reproduces your prior behavior; "spot" uses S if available.
    rl_state_mode: str = "forward"  # "forward" or "spot"
    rl_weight_mode: str = "inv_price"  # "inv_price" or "inv_price_atm"
    rl_atm_band_scale: float = 1.0  # only used when rl_weight_mode="inv_price_atm"


def _pick_rep_days(days: List[pd.Timestamp], n: int, mode: str) -> List[pd.Timestamp]:
    if n <= 0:
        return list(days)
    if mode == "middle":
        return [days[len(days) // 2]]
    if len(days) <= n:
        return list(days)
    idx = np.linspace(0, len(days) - 1, n)
    idx = np.unique(np.round(idx).astype(int))
    return [days[i] for i in idx]


def summarize_symbol_period_ivrmse(df_all: pd.DataFrame, cfg: IVRMSEConfig) -> Dict[str, pd.DataFrame]:
    # Prepare the input for the chosen type
    if cfg.type.lower() == "european":
        # If user already passed Deribit-like data, keep; else preprocess outside.
        df = df_all.copy()
    else:
        df = df_all.copy()

    # Normalize symbol column
    if "act_symbol" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"act_symbol": "symbol"})

    if "symbol" not in df.columns:
        raise ValueError("Expected a 'symbol' (or 'act_symbol') column in df_all.")
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in df_all.")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    mask = (df["symbol"] == cfg.symbol) & (df["date"] >= pd.Timestamp(cfg.start_date)) & (df["date"] <= pd.Timestamp(cfg.end_date))
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError(f"No rows for {cfg.symbol} in [{cfg.start_date}, {cfg.end_date}].")

    unique_days = sorted(df["date"].unique())
    if not unique_days:
        raise RuntimeError("No trading days after filtering.")

    # Representative day sampling (optional)
    days_used = unique_days
    if cfg.n_rep_days is not None:
        days_used = _pick_rep_days(unique_days, int(cfg.n_rep_days), cfg.rep_day_mode)

    iterator = days_used
    if cfg.show_progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(days_used, desc=f"{cfg.symbol} {pd.Timestamp(cfg.start_date).date()}→{pd.Timestamp(cfg.end_date).date()}")
        except Exception:
            pass

    outp = None
    log_fp = None
    if cfg.out_dir:
        outp = Path(cfg.out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        log_fp = (outp / "daily_log.txt").open("w", encoding="utf-8")
        (outp / "run_config.txt").write_text(str(cfg), encoding="utf-8")

    # Lazy imports for RL (only if requested)
    QLBSModel = None
    RLOPModel = None
    if cfg.run_qlbs:
        from lib.qlbs2.test_trained_model import QLBSModel as _QLBSModel
        QLBSModel = _QLBSModel
    if cfg.run_rlop:
        from lib.rlop2.test_trained_model import RLOPModel as _RLOPModel
        RLOPModel = _RLOPModel

    daily_rows: List[Dict[str, object]] = []
    preds_long_rows: List[pd.DataFrame] = []

    for day in iterator:
        df_day = df[df["date"] == day].copy()

        calls, parity_df = prepare_calls_one_day_symbol(
            df_day,
            centers_days=cfg.centers_days,
            min_parity_pairs=cfg.min_parity_pairs,
            tau_floor_days=cfg.tau_floor_days,
            type=cfg.type,
        )
        if calls.empty:
            msg = f"[{pd.Timestamp(day).date()}] no valid contracts after filters"
            if cfg.print_daily:
                print(msg)
            if log_fp:
                print(msg, file=log_fp)
            continue

        # Ensure bucket uses the configured centers
        calls["bucket"] = calls["tau"].apply(lambda t: assign_bucket_nearest(t, cfg.centers_days))

        day_rows = []
        for bucket, g in calls.groupby("bucket", sort=False):
            row: Dict[str, object] = {
                "date": pd.Timestamp(day),
                "bucket": bucket,
                "N": int(len(g)),
                "n_expiries": int(g["expiration"].nunique()),
            }

            # ==== BS (constant sigma) ====
            if cfg.run_bs:
                sig_hat = fit_sigma_bucket(g)
                row["BS_sigma_hat"] = float(sig_hat)
                ivs = _ivrmse_vs_constant_sigma(g, sig_hat)
                row.update({
                    "BS_IVRMSE_x1000_Whole": ivs["whole"] * 1000 if np.isfinite(ivs["whole"]) else np.nan,
                    "BS_IVRMSE_x1000_<1": ivs["<1"] * 1000 if np.isfinite(ivs["<1"]) else np.nan,
                    "BS_IVRMSE_x1000_>1": ivs[">1"] * 1000 if np.isfinite(ivs[">1"]) else np.nan,
                    "BS_IVRMSE_x1000_>1.03": ivs[">1.03"] * 1000 if np.isfinite(ivs[">1.03"]) else np.nan,
                })

            # ==== JD ====
            jd_params = None
            if cfg.run_jd:
                jd_params, jd_sse = calibrate_jd_bucket(g)
                row["JD_sse"] = float(jd_sse)
                row.update({f"JD_{k}": float(v) for k, v in jd_params.items()})

                def jd_price(F, K, tau, r):
                    return merton_price_call_b76(F, K, tau, r, jd_params["sigma"], jd_params["lam"], jd_params["muJ"], jd_params["deltaJ"])

                ivs, diag = _ivrmse_vs_model_prices(g, jd_price)
                row.update({
                    "JD_IVRMSE_x1000_Whole": ivs["whole"] * 1000 if np.isfinite(ivs["whole"]) else np.nan,
                    "JD_IVRMSE_x1000_<1": ivs["<1"] * 1000 if np.isfinite(ivs["<1"]) else np.nan,
                    "JD_IVRMSE_x1000_>1": ivs[">1"] * 1000 if np.isfinite(ivs[">1"]) else np.nan,
                    "JD_IVRMSE_x1000_>1.03": ivs[">1.03"] * 1000 if np.isfinite(ivs[">1.03"]) else np.nan,
                    "JD_bad_frac_whole": diag["bad_frac_whole"],
                    "JD_n_valid_whole": diag["n_valid_whole"],
                    "JD_n_total_whole": diag["n_total_whole"],
                })

            # ==== Heston SV ====
            h_params = None
            if cfg.run_heston:
                h_params, h_sse = calibrate_heston_bucket(g, u_max=cfg.heston_u_max, n_points=cfg.heston_n_points)
                row["SV_sse"] = float(h_sse)
                row.update({f"SV_{k}": float(v) for k, v in h_params.items()})

                def sv_price(F, K, tau, r):
                    return heston_price_call(F, K, tau, r, h_params, u_max=cfg.heston_u_max, n_points=cfg.heston_n_points)

                ivs, diag = _ivrmse_vs_model_prices(g, sv_price)
                row.update({
                    "Heston_IVRMSE_x1000_Whole": ivs["whole"] * 1000 if np.isfinite(ivs["whole"]) else np.nan,
                    "Heston_IVRMSE_x1000_<1": ivs["<1"] * 1000 if np.isfinite(ivs["<1"]) else np.nan,
                    "Heston_IVRMSE_x1000_>1": ivs[">1"] * 1000 if np.isfinite(ivs[">1"]) else np.nan,
                    "Heston_IVRMSE_x1000_>1.03": ivs[">1.03"] * 1000 if np.isfinite(ivs[">1.03"]) else np.nan,
                    "Heston_bad_frac_whole": diag["bad_frac_whole"],
                    "Heston_n_valid_whole": diag["n_valid_whole"],
                    "Heston_n_total_whole": diag["n_total_whole"],
                })

            # ==== QLBS ====
            if cfg.run_qlbs:
                Qmodel = QLBSModel(
                    is_call_option=True,
                    checkpoint=cfg.qlbs_checkpoint_tmpl.format(rl=cfg.qlbs_risk_lambda),
                    anchor_T=28 / 252,
                )

                # RL "underlying" convention
                if cfg.rl_state_mode == "spot" and "S" in g.columns and g["S"].notna().any():
                    underlying = g["S"].to_numpy(dtype=float)
                else:
                    underlying = g["F"].to_numpy(dtype=float)

                # fit uses a single scalar "spot" per their API; use median underlying
                spot0 = float(np.median(underlying))
                time_to_expiries = g["tau"].to_numpy(dtype=float)
                strikes = g["strike"].to_numpy(dtype=float)
                observed_prices = g["C_mid"].to_numpy(dtype=float)
                r0 = float(np.median(g["r"].to_numpy(dtype=float)))

                inv_price = 1.0 / np.power(np.clip(observed_prices, 1.0, None), 1.0)

                if cfg.rl_weight_mode == "inv_price_atm":
                    # Gaussian weight around ATM in log-moneyness units
                    m_z = np.log(strikes / np.maximum(g["F"].to_numpy(dtype=float), 1e-12))
                    w_atm = np.exp(-0.5 * (m_z / max(cfg.rl_atm_band_scale, 1e-6)) ** 2)
                    weights = inv_price * w_atm
                else:
                    weights = inv_price

                q_res = Qmodel.fit(
                    spot=spot0,
                    time_to_expiries=time_to_expiries,
                    strikes=strikes,
                    r=r0,
                    risk_lambda=cfg.qlbs_risk_lambda,
                    friction=4e-3,
                    observed_prices=observed_prices,
                    weights=weights,
                    sigma_guess=0.3,
                    mu_guess=0.0,
                    n_epochs=int(cfg.qlbs_epochs),
                )
                row["QLBS_sigma_fit"] = float(q_res.sigma)
                row["QLBS_mu_fit"] = float(q_res.mu)

                def qlbs_price(F, K, tau, r):
                    # Choose underlying consistently with the convention
                    spot_in = float(F) if cfg.rl_state_mode != "spot" else float(F)  # forward-mode: pass F
                    if cfg.rl_state_mode == "spot" and "S" in g.columns and g["S"].notna().any():
                        # For spot-mode, we *approximate* spot by F*DF+PV_div if present per-row,
                        # but if S is present in g, the safest is to map by expiry; keep simple here.
                        spot_in = float(F)  # user can customize if needed
                    pred = Qmodel.predict(
                        spot=spot_in,
                        time_to_expiries=np.array([float(tau)]),
                        strikes=np.array([float(K)]),
                        r=float(r),
                        risk_lambda=cfg.qlbs_risk_lambda,
                        friction=4e-3,
                        sigma_fit=float(q_res.sigma),
                        mu_fit=float(q_res.mu),
                    )
                    return float(pred.estimated_prices[0])

                ivs, diag = _ivrmse_vs_model_prices(g, qlbs_price)
                row.update({
                    "QLBS_IVRMSE_x1000_Whole": ivs["whole"] * 1000 if np.isfinite(ivs["whole"]) else np.nan,
                    "QLBS_IVRMSE_x1000_<1": ivs["<1"] * 1000 if np.isfinite(ivs["<1"]) else np.nan,
                    "QLBS_IVRMSE_x1000_>1": ivs[">1"] * 1000 if np.isfinite(ivs[">1"]) else np.nan,
                    "QLBS_IVRMSE_x1000_>1.03": ivs[">1.03"] * 1000 if np.isfinite(ivs[">1.03"]) else np.nan,
                    "QLBS_bad_frac_whole": diag["bad_frac_whole"],
                    "QLBS_n_valid_whole": diag["n_valid_whole"],
                    "QLBS_n_total_whole": diag["n_total_whole"],
                })

            # ==== RLOP ====
            if cfg.run_rlop:
                Rmodel = RLOPModel(
                    is_call_option=True,
                    checkpoint=cfg.rlop_checkpoint,
                    anchor_T=28 / 252,
                )

                if cfg.rl_state_mode == "spot" and "S" in g.columns and g["S"].notna().any():
                    underlying = g["S"].to_numpy(dtype=float)
                else:
                    underlying = g["F"].to_numpy(dtype=float)

                spot0 = float(np.median(underlying))
                time_to_expiries = g["tau"].to_numpy(dtype=float)
                strikes = g["strike"].to_numpy(dtype=float)
                observed_prices = g["C_mid"].to_numpy(dtype=float)
                r0 = float(np.median(g["r"].to_numpy(dtype=float)))

                inv_price = 1.0 / np.power(np.clip(observed_prices, 1.0, None), 1.0)
                if cfg.rl_weight_mode == "inv_price_atm":
                    m_z = np.log(strikes / np.maximum(g["F"].to_numpy(dtype=float), 1e-12))
                    w_atm = np.exp(-0.5 * (m_z / max(cfg.rl_atm_band_scale, 1e-6)) ** 2)
                    weights = inv_price * w_atm
                else:
                    weights = inv_price

                r_res = Rmodel.fit(
                    spot=spot0,
                    time_to_expiries=time_to_expiries,
                    strikes=strikes,
                    r=r0,
                    risk_lambda=cfg.rlop_risk_lambda,
                    friction=4e-3,
                    observed_prices=observed_prices,
                    weights=weights,
                    sigma_guess=0.3,
                    mu_guess=0.0,
                    n_epochs=int(cfg.rlop_epochs),
                )
                row["RLOP_sigma_fit"] = float(r_res.sigma)
                row["RLOP_mu_fit"] = float(r_res.mu)

                def rlop_price(F, K, tau, r):
                    spot_in = float(F)  # forward-mode by default
                    if cfg.rl_state_mode == "spot" and "S" in g.columns and g["S"].notna().any():
                        spot_in = float(F)  # same note as QLBS
                    pred = Rmodel.predict(
                        spot=spot_in,
                        time_to_expiries=np.array([float(tau)]),
                        strikes=np.array([float(K)]),
                        r=float(r),
                        risk_lambda=cfg.rlop_risk_lambda,
                        friction=4e-3,
                        sigma_fit=float(r_res.sigma),
                        mu_fit=float(r_res.mu),
                    )
                    return float(pred.estimated_prices[0])

                ivs, diag = _ivrmse_vs_model_prices(g, rlop_price)
                row.update({
                    "RLOP_IVRMSE_x1000_Whole": ivs["whole"] * 1000 if np.isfinite(ivs["whole"]) else np.nan,
                    "RLOP_IVRMSE_x1000_<1": ivs["<1"] * 1000 if np.isfinite(ivs["<1"]) else np.nan,
                    "RLOP_IVRMSE_x1000_>1": ivs[">1"] * 1000 if np.isfinite(ivs[">1"]) else np.nan,
                    "RLOP_IVRMSE_x1000_>1.03": ivs[">1.03"] * 1000 if np.isfinite(ivs[">1.03"]) else np.nan,
                    "RLOP_bad_frac_whole": diag["bad_frac_whole"],
                    "RLOP_n_valid_whole": diag["n_valid_whole"],
                    "RLOP_n_total_whole": diag["n_total_whole"],
                })

            day_rows.append(row)

        if not day_rows:
            msg = f"[{pd.Timestamp(day).date()}] no valid buckets"
            if cfg.print_daily:
                print(msg)
            if log_fp:
                print(msg, file=log_fp)
            continue

        daily_rows.extend(day_rows)

        if cfg.print_daily:
            # quick pooled-like status line (not used for paper, just a sanity check)
            dd = pd.DataFrame(day_rows)
            parts = []
            for model, col in [
                ("BS", "BS_IVRMSE_x1000_Whole"),
                ("JD", "JD_IVRMSE_x1000_Whole"),
                ("SV", "Heston_IVRMSE_x1000_Whole"),
                ("QLBS", "QLBS_IVRMSE_x1000_Whole"),
                ("RLOP", "RLOP_IVRMSE_x1000_Whole"),
            ]:
                if col in dd.columns:
                    v = dd[col].mean(skipna=True)
                    parts.append(f"{model}={v:.1f}" if np.isfinite(v) else f"{model}=NA")
            msg = f"[{pd.Timestamp(day).date()}] avg bucket Whole x1000: " + ", ".join(parts)
            print(msg)
            if log_fp:
                print(msg, file=log_fp)

    if log_fp:
        log_fp.close()

    if not daily_rows:
        raise RuntimeError("No valid (day,bucket) rows — check filters or inputs.")

    daily = pd.DataFrame(daily_rows)

    # Equal-day mean per bucket (PRIMARY)
    iv_cols = [c for c in daily.columns if c.endswith("_IVRMSE_x1000_Whole") or c.endswith("_IVRMSE_x1000_<1") or c.endswith("_IVRMSE_x1000_>1") or c.endswith("_IVRMSE_x1000_>1.03")]
    equal_day_mean = daily.groupby("bucket", as_index=False)[iv_cols].mean()

    # Coverage diagnostics
    days_used_df = daily.groupby("bucket")["date"].nunique().rename("days_used").reset_index()
    N_total = daily.groupby("bucket")["N"].sum().rename("N_total").reset_index()
    equal_day_mean = equal_day_mean.merge(days_used_df, on="bucket").merge(N_total, on="bucket").sort_values("bucket")

    # Save
    res = {"daily": daily, "equal_day_mean": equal_day_mean}
    if outp is not None:
        daily.to_csv(outp / "daily_ivrmse.csv", index=False)
        equal_day_mean.to_csv(outp / "equal_day_mean.csv", index=False)
        with open(outp / "summary_res.pkl", "wb") as f:
            pickle.dump(res, f)

    return res


# ============================================================
# Publication-style table (matches your IVRMSE layout)
# ============================================================

def make_publication_table_ivrmse(
    res: Dict[str, pd.DataFrame],
    *,
    symbol: str,
    buckets: List[int],
    decimals: int = 2,
    out_dir: Optional[str] = None,
    basename: str = "table_ivrmse_equal",
) -> pd.DataFrame:
    """
    Sections: Whole sample / Moneyness <1 / >1 / >1.03
    Rows: per bucket: '{symbol} (τ=14d)'
    Columns: BS, JD, SV(Heston), QLBS, RLOP (if present)
    """
    df_src = res["equal_day_mean"]
    if df_src is None or df_src.empty:
        raise ValueError("equal_day_mean is empty.")

    present = set(df_src.columns)
    models = []
    if any(c.startswith("BS_IVRMSE_x1000") for c in present):
        models.append("BS")
    if any(c.startswith("JD_IVRMSE_x1000") for c in present):
        models.append("JD")
    if any(c.startswith("Heston_IVRMSE_x1000") for c in present):
        models.append("SV")
    if any(c.startswith("QLBS_IVRMSE_x1000") for c in present):
        models.append("QLBS")
    if any(c.startswith("RLOP_IVRMSE_x1000") for c in present):
        models.append("RLOP")

    prefix = {"BS": "BS", "JD": "JD", "SV": "Heston", "QLBS": "QLBS", "RLOP": "RLOP"}
    sections = [("Whole sample", "Whole"), ("Moneyness <1", "<1"), ("Moneyness >1", ">1"), ("Moneyness >1.03", ">1.03")]

    rows = []
    for section_name, suffix in sections:
        for d in [f"{x}d" for x in buckets]:
            sub = df_src[df_src["bucket"] == d]
            row = {"Moneyness": section_name, "Asset": f"{symbol} (τ={d})"}
            for m in models:
                col = f"{prefix[m]}_IVRMSE_x1000_{suffix}"
                row[m] = float(sub[col].iloc[0]) if (not sub.empty and col in sub.columns and pd.notna(sub[col].iloc[0])) else np.nan
            rows.append(row)

    table = pd.DataFrame(rows)
    for m in models:
        table[m] = table[m].round(decimals)

    if out_dir:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        csv_path = outp / f"{basename}.csv"
        table.to_csv(csv_path, index=False)

        # Markdown with bold minimum per row
        header = ["Moneyness", "Asset"] + models
        md_rows = []
        md_rows.append("| " + " | ".join(header) + " |")
        md_rows.append("| " + " | ".join(["---"] * len(header)) + " |")
        for _, r in table.iterrows():
            vals = [r[m] for m in models]
            not_nan = [i for i, v in enumerate(vals) if pd.notna(v)]
            best_idx = min(not_nan, key=lambda i: vals[i]) if not_nan else None
            cells = [str(r["Moneyness"]), str(r["Asset"])]
            for i, m in enumerate(models):
                v = r[m]
                if pd.isna(v):
                    cells.append("")
                else:
                    s = f"{v:.{decimals}f}"
                    cells.append(f"**{s}**" if (best_idx is not None and i == best_idx) else s)
            md_rows.append("| " + " | ".join(cells) + " |")
        md_text = "\n".join(md_rows)
        md_path = outp / f"{basename}.md"
        md_path.write_text(md_text, encoding="utf-8")
        print(f"Saved: {csv_path}")
        print(f"Saved: {md_path}")

    return table


# ============================================================
# Example runners
# ============================================================

def run_spy_20q1_ivrmse():
    df = pd.read_csv("data/spy_preprocessed_calls_20q1.csv")

    cfg = IVRMSEConfig(
        symbol="SPY",
        type="american",
        start_date="2020-01-06",
        end_date="2020-03-30",
        centers_days=[14, 28, 56],
        min_parity_pairs=4,
        tau_floor_days=3,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        # speed knobs
        heston_u_max=60.0,
        heston_n_points=201,
        # full sample by default; for quick test set e.g. n_rep_days=5
        n_rep_days=None,
        rep_day_mode="even",
        out_dir="SPY_20Q1_baseline_ivrmse_v6",
        print_daily=True,
        show_progress=True,
        # RL config
        rl_state_mode="forward",       # matches your prior behavior; change to "spot" if you want to use S
        rl_weight_mode="inv_price",    # or "inv_price_atm"
    )

    res = summarize_symbol_period_ivrmse(df, cfg)
    tbl = make_publication_table_ivrmse(
        res,
        symbol="SPY",
        buckets=[14, 28, 56],
        decimals=2,
        out_dir=cfg.out_dir,
        basename="table_ivrmse_equal",
    )
    print(tbl)

class Test(TestCase):
    def test_main(self):
        #main_spy20()
        #main_spy25()
        #main_xop20()
        #main_xop25()
        #main_btc()
        run_spy_20q1_ivrmse()
        #hedging_spy25()
        #hedging_xop20()
        #hedging_xop25()
