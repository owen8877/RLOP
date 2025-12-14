
from __future__ import annotations

import json
import math
import re
import pickle
from unittest import TestCase
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try SciPy for calibration; fall back to coarse grid if missing
try:
    from scipy.optimize import minimize

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ============================================================
# Core math: Normal CDF (vector-friendly)
# ============================================================

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """
    Normal CDF using math.erf; vectorized via np.vectorize (fast enough for n~2000).
    """
    x = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


# ============================================================
# Black–76 call price (expects FORWARD F)
# ============================================================

def b76_price_call(F: float, K: float, tau: float, r: float, sigma: float) -> float:
    """Black–76 call: DF*(F N(d1) - K N(d2)), where F is forward."""
    if tau <= 0.0 or K <= 0.0 or F <= 0.0:
        DF = math.exp(-r * max(tau, 0.0))
        return DF * max(F - K, 0.0)
    if sigma <= 0.0:
        DF = math.exp(-r * tau)
        return DF * max(F - K, 0.0)

    v = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / v
    d2 = d1 - v
    DF = math.exp(-r * tau)
    return DF * (F * float(_norm_cdf(np.array([d1]))[0]) - K * float(_norm_cdf(np.array([d2]))[0]))


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
    """Bisection IV with bracket expansion; clamps to intrinsic."""
    if not (F > 0 and K > 0 and tau >= 0 and math.isfinite(target)):
        return None

    DF = math.exp(-r * tau)
    intrinsic = DF * max(F - K, 0.0)
    t = max(float(target), intrinsic)

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
            return float(c)
        if fa * fc <= 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return float(0.5 * (a + b))


# ============================================================
# Spot <-> Forward conversion (deterministic r,q)
# ============================================================

def spot_to_forward(S: np.ndarray, r: float, q: float, tau: float) -> np.ndarray:
    return np.asarray(S, dtype=float) * math.exp((r - q) * max(tau, 0.0))


def bs_call_from_spot(S: float, K: float, tau: float, r: float, q: float, sigma: float) -> float:
    """Black–Scholes call from SPOT (implemented via B76 forward conversion)."""
    F = float(spot_to_forward(np.array([S]), r, q, tau)[0])
    return b76_price_call(F, K, tau, r, sigma)


def bs_delta_from_spot(S: np.ndarray, K: float, tau: float, r: float, q: float, sigma: float) -> np.ndarray:
    """
    Delta w.r.t SPOT for a call:
      delta = exp(-q tau) * N(d1), where d1 uses forward F=S*exp((r-q)tau).
    """
    S = np.asarray(S, dtype=float)
    if tau <= 0.0:
        return (S > K).astype(float)

    F = spot_to_forward(S, r, q, tau)
    v = sigma * math.sqrt(tau)
    d1 = (np.log(np.maximum(F, 1e-300) / K) + 0.5 * sigma * sigma * tau) / v
    return np.exp(-q * tau) * _norm_cdf(d1)


# ============================================================
# GBM spot simulation under risk-neutral (r-q)
# ============================================================

def simulate_spot_paths(
    S0: float,
    r: float,
    q: float,
    sigma_true: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """
    Simulate SPOT:
      dS/S = (r-q) dt + sigma dW
    Returns shape (n_paths, n_steps+1).
    """
    dt = T / n_steps
    rng = np.random.default_rng(int(seed))
    Z = rng.normal(size=(n_paths, n_steps))

    S = np.empty((n_paths, n_steps + 1), dtype=float)
    S[:, 0] = float(S0)

    drift = (r - q - 0.5 * sigma_true * sigma_true) * dt
    vol = sigma_true * math.sqrt(dt)

    for t in range(n_steps):
        S[:, t + 1] = S[:, t] * np.exp(drift + vol * Z[:, t])
    return S


# ============================================================
# Bucket assignment
# ============================================================

def assign_bucket_label(tau_years: float, centers_days: List[int]) -> str:
    days = float(tau_years) * 365.0
    j = int(np.argmin([abs(days - c) for c in centers_days]))
    return f"{centers_days[j]}d"


# ============================================================
# Data adapter + one-day call surface (American fast-path)
# ============================================================

def adapter_eur_calls_to_summarizer(calls_out: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts preprocessed calls with at least:
      date, act_symbol (or symbol), expiration, strike, C_eur, F, DF
    Optional but strongly recommended:
      S, r, PV_div
    Returns the minimal schema used by prepare_calls_one_day_symbol.
    """
    df = calls_out.copy()

    if "act_symbol" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "act_symbol"})
    if "act_symbol" not in df.columns:
        raise ValueError("Expected 'act_symbol' (or 'symbol') in input.")

    req = {"date", "act_symbol", "expiration", "strike", "C_eur", "F", "DF"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["C_eur"] = pd.to_numeric(df["C_eur"], errors="coerce")
    df["F"] = pd.to_numeric(df["F"], errors="coerce")
    df["DF"] = pd.to_numeric(df["DF"], errors="coerce")

    df["S"] = pd.to_numeric(df["S"], errors="coerce") if "S" in df.columns else np.nan
    df["r"] = pd.to_numeric(df["r"], errors="coerce") if "r" in df.columns else np.nan
    df["PV_div"] = pd.to_numeric(df["PV_div"], errors="coerce") if "PV_div" in df.columns else 0.0

    out = df[["date", "act_symbol", "expiration", "strike", "C_eur", "S", "r", "PV_div", "F", "DF"]].copy()
    out["cp"] = "Call"
    out["mid"] = out["C_eur"]
    out["bid"] = out["mid"]
    out["ask"] = out["mid"]

    out = out.dropna(subset=["date", "act_symbol", "expiration", "strike", "mid", "F", "DF"])
    out = out[(out["mid"] > 0) & (out["DF"] > 0) & (out["ask"] >= out["bid"]) & (out["bid"] >= 0)]
    out = out.sort_values(["date", "act_symbol", "expiration", "strike"]).reset_index(drop=True)
    return out


def prepare_calls_one_day_symbol(
    df_day_symbol: pd.DataFrame,
    tau_floor_days: int = 3,
) -> pd.DataFrame:
    """
    American fast-path:
      - calls only
      - trusts supplied F/DF/(r) and uses C_eur as mid already
      - infers q_hat per expiry using F = S*exp((r-q)tau)
    Returns calls surface with:
      date,symbol,expiration,tau,strike,C_mid,F,DF,r,S,q_hat,
      sigma_mkt_b76, moneyness_F, moneyness_S
    """
    df = df_day_symbol.copy()
    df = df.rename(columns={"act_symbol": "symbol"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    df["F"] = pd.to_numeric(df["F"], errors="coerce")
    df["DF"] = pd.to_numeric(df["DF"], errors="coerce")
    df["S"] = pd.to_numeric(df["S"], errors="coerce") if "S" in df.columns else np.nan
    df["r"] = pd.to_numeric(df["r"], errors="coerce") if "r" in df.columns else np.nan
    df["PV_div"] = pd.to_numeric(df["PV_div"], errors="coerce") if "PV_div" in df.columns else 0.0

    df["tau"] = (df["expiration"] - df["date"]).dt.days / 365.0
    df = df[df["tau"].notna() & (df["tau"] > 0)].copy()
    df = df[df["strike"].notna() & df["mid"].notna() & (df["mid"] > 0)].copy()
    df = df[df["F"].notna() & (df["F"] > 0) & df["DF"].notna() & (df["DF"] > 0)].copy()

    if tau_floor_days and tau_floor_days > 0:
        df = df[df["tau"] * 365.0 >= tau_floor_days].copy()

    if df.empty:
        return pd.DataFrame()

    # Per-expiry header
    hdr = df.groupby(["date", "symbol", "expiration"], as_index=False).agg(
        F=("F", "first"),
        DF=("DF", "first"),
        S=("S", "first"),
        PV_div=("PV_div", "first"),
        tau=("tau", "first"),
        r=("r", "first"),
    )

    # Infer r if missing
    r_infer = -np.log(np.clip(hdr["DF"].to_numpy(), 1e-12, None)) / np.maximum(hdr["tau"].to_numpy(), 1e-12)
    hdr["r"] = np.where(np.isfinite(hdr["r"].to_numpy()), hdr["r"].to_numpy(), r_infer)

    # Infer spot if missing: S = DF*F + PV_div
    S_infer = hdr["DF"].to_numpy() * hdr["F"].to_numpy() + hdr["PV_div"].to_numpy()
    hdr["S"] = np.where(np.isfinite(hdr["S"].to_numpy()) & (hdr["S"].to_numpy() > 0), hdr["S"].to_numpy(), S_infer)

    # q_hat per expiry: F = S * exp((r-q)tau) => q = r - ln(F/S)/tau
    tau = np.maximum(hdr["tau"].to_numpy(), 1e-12)
    F = np.maximum(hdr["F"].to_numpy(), 1e-12)
    S = np.maximum(hdr["S"].to_numpy(), 1e-12)
    r = hdr["r"].to_numpy()
    hdr["q_hat"] = r - np.log(F / S) / tau

    calls = df.groupby(["date", "symbol", "expiration", "tau", "strike"], as_index=False).agg(C_mid=("mid", "first"))
    calls = calls.merge(hdr[["date", "symbol", "expiration", "F", "DF", "r", "S", "PV_div", "q_hat"]], on=["date", "symbol", "expiration"], how="left")

    # Market B76 IVs (using forward F)
    calls["sigma_mkt_b76"] = calls.apply(
        lambda rr: b76_iv_from_price(float(rr["C_mid"]), float(rr["F"]), float(rr["strike"]), float(rr["tau"]), float(rr["r"])),
        axis=1,
    )
    calls = calls.dropna(subset=["sigma_mkt_b76"]).copy()
    if calls.empty:
        return pd.DataFrame()

    calls["moneyness_F"] = calls["strike"] / calls["F"]
    calls["moneyness_S"] = calls["strike"] / calls["S"]
    return calls.reset_index(drop=True)


# ============================================================
# Calibration helpers
# ============================================================

def _minimize(func, x0: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    SciPy L-BFGS-B; fallback to a small local grid if SciPy missing.
    """
    x0 = np.asarray(x0, dtype=float)

    if _HAVE_SCIPY:
        res = minimize(func, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
        return res.x if res.success else x0

    grids = []
    for (lo, hi), xi in zip(bounds, x0):
        span = hi - lo
        g = np.linspace(max(lo, xi - 0.2 * span), min(hi, xi + 0.2 * span), 7)
        grids.append(g)

    mesh = np.meshgrid(*grids, indexing="ij")
    cand = np.stack([m.ravel() for m in mesh], axis=1)
    vals = np.array([func(p) for p in cand], dtype=float)
    return cand[int(np.argmin(vals))]


def fit_sigma_bucket(calls_bucket: pd.DataFrame) -> float:
    """
    1-parameter BS/B76: choose sigma minimizing price SSE under B76 (forward F).
    """
    if calls_bucket.empty:
        return float("nan")

    F = calls_bucket["F"].to_numpy(dtype=float)
    K = calls_bucket["strike"].to_numpy(dtype=float)
    tau = calls_bucket["tau"].to_numpy(dtype=float)
    r = calls_bucket["r"].to_numpy(dtype=float)
    C = calls_bucket["C_mid"].to_numpy(dtype=float)

    def sse(sig: float) -> float:
        if sig <= 0:
            return 1e18
        phat = np.array([b76_price_call(F[i], K[i], tau[i], r[i], sig) for i in range(len(C))], dtype=float)
        err = phat - C
        return float(np.dot(err, err))

    grid = np.geomspace(0.05, 2.0, 40)
    best = float(min(grid, key=sse))

    # Golden-section refinement
    a = max(best / 3.0, 1e-4)
    b = min(best * 3.0, 5.0)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    invphi = 1.0 / phi

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

    return float((a + b) / 2.0)


# ============================================================
# Merton Jump–Diffusion (JD) under forward measure (uses B76 mixture)
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
    Merton JD via Poisson mixture of B76 call prices (forward-based).
    k = E[e^Y]-1, Y~N(muJ,deltaJ^2)
    Use F_adj = F * exp(-lam*k*tau).
    For n jumps: F_n = F_adj * exp(n*muJ), sigma_n^2 = sigma^2 + n*deltaJ^2/tau.
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


def calibrate_jd_bucket(calls_bucket: pd.DataFrame) -> Tuple[Dict[str, float], float]:
    """
    Fit JD params by price SSE on the calibration surface.
    """
    if calls_bucket.empty:
        return {}, float("inf")

    atm_iv = float(np.median(calls_bucket["sigma_mkt_b76"].to_numpy(dtype=float)))
    F = calls_bucket["F"].to_numpy(dtype=float)
    K = calls_bucket["strike"].to_numpy(dtype=float)
    tau = calls_bucket["tau"].to_numpy(dtype=float)
    r = calls_bucket["r"].to_numpy(dtype=float)
    C = calls_bucket["C_mid"].to_numpy(dtype=float)

    def sse_vec(p: np.ndarray) -> float:
        sigma, lam, muJ, dJ = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        if sigma <= 0 or lam < 0 or dJ <= 0:
            return 1e18
        phat = np.array(
            [merton_price_call_b76(F[i], K[i], tau[i], r[i], sigma, lam, muJ, dJ) for i in range(len(C))],
            dtype=float,
        )
        err = phat - C
        return float(np.dot(err, err))

    x0 = np.array([max(0.02, atm_iv), 0.10, -0.02, 0.10], dtype=float)
    bounds = [(0.01, 3.0), (0.0, 5.0), (-0.5, 0.5), (0.01, 1.0)]
    p = _minimize(sse_vec, x0, bounds)
    params = {"sigma": float(p[0]), "lam": float(p[1]), "muJ": float(p[2]), "deltaJ": float(p[3])}
    return params, float(sse_vec(p))


# ============================================================
# Heston SV (Heston CF + integration) under forward measure
# (kept lightweight: defaults reduced for speed)
# ============================================================

def _heston_cf(u: np.ndarray, F: float, tau: float, kappa: float, theta: float, sigma_v: float, rho: float, v0: float) -> np.ndarray:
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


def _heston_prob(F: float, K: float, tau: float, params: Dict[str, float], j: int, u_max: float, n_points: int) -> float:
    kappa, theta, sigma_v, rho, v0 = params["kappa"], params["theta"], params["sigma_v"], params["rho"], params["v0"]
    lnK = math.log(K)
    u = np.linspace(1e-6, float(u_max), int(n_points))
    du = float(u[1] - u[0])

    if j == 2:
        phi = _heston_cf(u, F, tau, kappa, theta, sigma_v, rho, v0)
        integrand = np.real(np.exp(-1j * u * lnK) * phi / (1j * u))
    else:
        phi_shift = _heston_cf(u - 1j, F, tau, kappa, theta, sigma_v, rho, v0)
        phi_mi = _heston_cf(np.array([-1j]), F, tau, kappa, theta, sigma_v, rho, v0)[0]
        integrand = np.real(np.exp(-1j * u * lnK) * (phi_shift / (1j * u * phi_mi)))

    return float(0.5 + (1.0 / math.pi) * _simpson_integral(integrand, du))


def heston_price_call(
    F: float, K: float, tau: float, r: float, params: Dict[str, float], u_max: float = 60.0, n_points: int = 201
) -> float:
    DF = math.exp(-r * tau)
    P1 = _heston_prob(F, K, tau, params, j=1, u_max=u_max, n_points=n_points)
    P2 = _heston_prob(F, K, tau, params, j=2, u_max=u_max, n_points=n_points)
    return float(DF * (F * P1 - K * P2))


def calibrate_heston_bucket(calls_bucket: pd.DataFrame, u_max: float = 60.0, n_points: int = 201) -> Tuple[Dict[str, float], float]:
    if calls_bucket.empty:
        return {}, float("inf")

    atm_iv = float(np.median(calls_bucket["sigma_mkt_b76"].to_numpy(dtype=float)))
    F = calls_bucket["F"].to_numpy(dtype=float)
    K = calls_bucket["strike"].to_numpy(dtype=float)
    tau = calls_bucket["tau"].to_numpy(dtype=float)
    r = calls_bucket["r"].to_numpy(dtype=float)
    C = calls_bucket["C_mid"].to_numpy(dtype=float)

    def sse_vec(p: np.ndarray) -> float:
        kappa, theta, sigma_v, rho, v0 = map(float, p)
        if kappa <= 0 or theta <= 0 or sigma_v <= 0 or not (-0.999 <= rho <= 0.0) or v0 <= 0:
            return 1e18
        params = {"kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho, "v0": v0}
        phat = np.array([heston_price_call(F[i], K[i], tau[i], r[i], params, u_max=u_max, n_points=n_points) for i in range(len(C))], dtype=float)
        err = phat - C
        return float(np.dot(err, err))

    v0_0 = max(1e-4, atm_iv * atm_iv)
    x0 = np.array([2.0, v0_0, 0.5, -0.5, v0_0], dtype=float)
    bounds = [(0.05, 10.0), (1e-4, 2.0), (1e-3, 3.0), (-0.999, 0.0), (1e-4, 3.0)]
    p = _minimize(sse_vec, x0, bounds)
    params = {"kappa": float(p[0]), "theta": float(p[1]), "sigma_v": float(p[2]), "rho": float(p[3]), "v0": float(p[4])}
    return params, float(sse_vec(p))


# ============================================================
# Delta grid precompute (speedup)
# ============================================================

def build_S_grid(S0: float, sigma_true: float, T: float, grid_n: int = 61, n_std: float = 4.0) -> np.ndarray:
    """
    Log-space grid covering roughly exp(±n_std*sigma*sqrt(T)) around S0.
    """
    width = float(n_std) * float(sigma_true) * math.sqrt(max(T, 1e-12))
    lo = math.log(max(S0, 1e-12)) - width
    hi = math.log(max(S0, 1e-12)) + width
    return np.exp(np.linspace(lo, hi, int(grid_n))).astype(float)


def precompute_delta_grid(
    price_scalar_fn: Callable[[float, float, float, float, float], float],
    K: float,
    r: float,
    q: float,
    T: float,
    n_steps: int,
    S_grid: np.ndarray,
) -> np.ndarray:
    """
    Precompute delta(t, S) for hedge times t=0..n_steps-1 (no rebalance at maturity).
    price_scalar_fn signature: price(S, K, tau, r, q)
    Returns delta_grid shape (n_steps, grid_n).
    """
    Sg = np.asarray(S_grid, dtype=float)
    grid_n = Sg.size
    dt = T / n_steps

    prices = np.empty((n_steps, grid_n), dtype=float)
    for t in range(n_steps):
        tau = max(T - t * dt, 0.0)
        prices[t, :] = np.array([price_scalar_fn(float(Sg[i]), float(K), float(tau), float(r), float(q)) for i in range(grid_n)], dtype=float)

    # Numerical derivative dC/dS along S_grid
    deltas = np.gradient(prices, Sg, axis=1)
    return deltas.astype(float)


def interp_delta(delta_grid: np.ndarray, S_grid: np.ndarray, step_idx: int, S_vec: np.ndarray) -> np.ndarray:
    """
    Interpolate deltas at the current S for a given step index.
    """
    S_vec = np.asarray(S_vec, dtype=float)
    smin, smax = float(S_grid[0]), float(S_grid[-1])
    Sv = np.clip(S_vec, smin, smax)
    return np.interp(Sv, S_grid, delta_grid[int(step_idx), :]).astype(float)


# ============================================================
# Hedging engine (no rebalance at maturity)
# ============================================================

def hedge_short_call_paths(
    S_paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    premium0: float,
    friction: float,
    get_delta: Callable[[int, np.ndarray], np.ndarray],
) -> Dict[str, float]:
    """
    Delta hedge a SHORT call:
      - t=0: receive premium0, buy delta0 shares (pay friction)
      - rebalance at steps 1..n_steps-1 only
      - then accrue cash to maturity and settle payoff

    Returns:
      RMSE_hedge, avg_cost, shortfall_prob, shortfall_1pct, err_mean, err_std
    """
    S_paths = np.asarray(S_paths, dtype=float)
    n_paths, n_steps_plus_1 = S_paths.shape
    n_steps = n_steps_plus_1 - 1
    dt = T / n_steps

    S0 = S_paths[:, 0]
    ST = S_paths[:, -1]
    payoff = np.maximum(ST - float(K), 0.0)

    prem = float(premium0) * np.ones_like(S0, dtype=float)

    # Initial delta and trade
    delta = get_delta(0, S0)
    trade0 = delta  # from 0 to delta

    trading_cost = np.zeros_like(S0, dtype=float)
    cost0 = friction * np.abs(trade0) * S0 if friction > 0.0 else 0.0
    trading_cost += cost0
    cash = prem - trade0 * S0 - cost0

    # Rebalance at times 1..n_steps-1
    for t in range(1, n_steps):
        cash *= math.exp(r * dt)
        St = S_paths[:, t]
        new_delta = get_delta(t, St)
        trade = new_delta - delta

        if friction > 0.0:
            cost = friction * np.abs(trade) * St
            trading_cost += cost
            cash -= cost

        cash -= trade * St
        delta = new_delta

    # Accrue cash to maturity (last interval), hold delta fixed
    cash *= math.exp(r * dt)

    V_T = cash + delta * ST
    error = V_T - payoff

    rmse = float(np.sqrt(np.mean(error**2)))
    avg_cost = float(np.mean(trading_cost))
    shortfall_prob = float(np.mean(error < 0.0))
    shortfall_1pct = float(np.quantile(error, 0.01))
    err_mean = float(np.mean(error))
    err_std = float(np.std(error, ddof=0))

    return {
        "RMSE_hedge": rmse,
        "avg_cost": avg_cost,
        "shortfall_prob": shortfall_prob,
        "shortfall_1pct": shortfall_1pct,
        "err_mean": err_mean,
        "err_std": err_std,
    }


# ============================================================
# Selection helper: pick strikes closest to moneyness targets
# ============================================================

def pick_options_by_moneyness_targets(anchor_slice: pd.DataFrame, targets: List[float]) -> pd.DataFrame:
    """
    Picks one option per target using min |(K/F) - target| within anchor expiry slice.
    Returns a DataFrame with one row per target (may contain duplicate strikes if surface is sparse).
    """
    g = anchor_slice.copy()
    g["moneyness_F"] = g["strike"] / g["F"]
    rows = []
    for m in targets:
        j = (g["moneyness_F"] - float(m)).abs().idxmin()
        rows.append(g.loc[j])
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["moneyness_target"] = [float(x) for x in targets]
    out["moneyness_actual_F"] = out["strike"] / out["F"]
    out["moneyness_actual_S"] = out["strike"] / out["S"]
    return out


# ============================================================
# Output saving
# ============================================================

def _json_dump(path: Path, obj: Any) -> None:
    def _default(x: Any) -> Any:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, (pd.Timestamp,)):
            return x.isoformat()
        return str(x)

    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


# ============================================================
# Main runner: dynamic hedging for one symbol & period
# ============================================================

@dataclass
class HedgingRunConfig:
    symbol: str = "SPY"
    start_date: str = "2020-01-06"
    end_date: str = "2020-03-30"
    buckets: Tuple[int, ...] = (28,)
    n_rep_days: int = 5
    rep_day_mode: str = "even"  # "even" or "middle"
    moneyness_targets: Tuple[float, ...] = (0.90, 0.97, 1.00, 1.03, 1.10)

    calibrate_on: str = "expiry"  # "expiry" or "bucket"
    sigma_true_mode: str = "expiry_atm"  # "expiry_atm" or "expiry_median" or "bucket_median"
    n_steps: int = 28
    n_paths: int = 2000
    friction: float = 4e-3
    seed: int = 123

    run_bs: bool = True
    run_jd: bool = True
    run_heston: bool = True
    run_qlbs: bool = True
    run_rlop: bool = True

    # Delta grid controls (major speed lever)
    grid_n: int = 61
    grid_n_std: float = 4.0

    # Heston integration (speed/quality)
    heston_u_max: float = 60.0
    heston_n_points: int = 201

    # RL knobs
    qlbs_risk_lambda: float = 0.01
    qlbs_checkpoint_tmpl: str = "trained_model/test8/risk_lambda={rl:.1e}/policy_1.pt"
    rlop_risk_lambda: float = 0.10
    rlop_checkpoint: str = "trained_model/testr9/policy_1.pt"

    # Saving
    out_dir: Optional[str] = None
    run_tag: Optional[str] = None


def run_dynamic_hedging_spy_like(
    df_all: pd.DataFrame,
    cfg: HedgingRunConfig,
) -> pd.DataFrame:
    """
    One run producing long-form rows:
      (date, bucket, moneyness_target, model) -> hedging metrics
    Calibrates ONCE per (rep_day, bucket) and reuses for all targets.
    """
    # Preprocess to expected schema
    df_pre = adapter_eur_calls_to_summarizer(df_all)

    # Filter
    df_pre["date"] = pd.to_datetime(df_pre["date"]).dt.normalize()
    mask = (
        (df_pre["act_symbol"] == cfg.symbol)
        & (df_pre["date"] >= pd.Timestamp(cfg.start_date))
        & (df_pre["date"] <= pd.Timestamp(cfg.end_date))
    )
    df = df_pre.loc[mask].copy()
    if df.empty:
        raise ValueError(f"No rows for {cfg.symbol} in [{cfg.start_date}, {cfg.end_date}].")

    # Representative days
    unique_days = sorted(df["date"].unique())
    if cfg.rep_day_mode not in ("even", "middle"):
        raise ValueError("rep_day_mode must be 'even' or 'middle'.")

    if cfg.rep_day_mode == "middle":
        rep_days = [unique_days[len(unique_days) // 2]]
    else:
        n = max(1, int(cfg.n_rep_days))
        if len(unique_days) <= n:
            rep_days = list(unique_days)
        else:
            idx = np.linspace(0, len(unique_days) - 1, n)
            idx = np.unique(np.round(idx).astype(int))
            rep_days = [unique_days[i] for i in idx]

    # Output folder
    outp = None
    if cfg.out_dir is not None:
        outp = Path(cfg.out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        run_tag = cfg.run_tag or f"{cfg.symbol}_{cfg.start_date}_{cfg.end_date}_hedge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        outp = outp / run_tag
        outp.mkdir(parents=True, exist_ok=True)
        _json_dump(outp / "config.json", cfg.__dict__)
        pd.DataFrame({"rep_day": rep_days}).to_csv(outp / "rep_days.csv", index=False)

    # Lazy imports for RL
    QLBSModel = None
    RLOPModel = None
    if cfg.run_qlbs:
        from lib.qlbs2.test_trained_model import QLBSModel as _QLBSModel
        QLBSModel = _QLBSModel
    if cfg.run_rlop:
        from lib.rlop2.test_trained_model import RLOPModel as _RLOPModel
        RLOPModel = _RLOPModel

    bucket_centers = list(map(int, cfg.buckets))
    targets = list(map(float, cfg.moneyness_targets))

    results: List[Dict[str, Any]] = []

    for day_i, rep_day in enumerate(rep_days):
        df_day = df[df["date"] == rep_day].copy()
        calls = prepare_calls_one_day_symbol(df_day, tau_floor_days=3)
        if calls.empty:
            continue

        calls["bucket"] = calls["tau"].apply(lambda t: assign_bucket_label(float(t), bucket_centers))

        for b_center in bucket_centers:
            b_label = f"{b_center}d"
            calls_b = calls[calls["bucket"] == b_label].copy()
            if calls_b.empty:
                continue

            # Anchor expiry: nearest tau_days to bucket center
            calls_b["tau_days"] = calls_b["tau"] * 365.0
            idx_anchor = (calls_b["tau_days"] - float(b_center)).abs().idxmin()
            anchor_row = calls_b.loc[idx_anchor]
            anchor_exp = anchor_row["expiration"]

            anchor_slice = calls_b[calls_b["expiration"] == anchor_exp].copy()
            if anchor_slice.empty:
                anchor_slice = calls_b.copy()

            # Core market quantities for this (day,bucket)
            T = float(anchor_row["tau"])
            r = float(anchor_row["r"])
            q = float(anchor_row["q_hat"])
            S0 = float(anchor_row["S"])
            F0 = float(anchor_row["F"])

            # Selection: 5 strikes for targets (within anchor expiry)
            selected = pick_options_by_moneyness_targets(anchor_slice, targets)

            # Calibration surface: expiry-only or full bucket
            if cfg.calibrate_on not in ("expiry", "bucket"):
                raise ValueError("calibrate_on must be 'expiry' or 'bucket'.")
            cal_df = anchor_slice if cfg.calibrate_on == "expiry" else calls_b

            # World sigma_true
            if cfg.sigma_true_mode == "expiry_atm":
                # Use the strike closest to K/F=1.0 in anchor expiry
                tmp = anchor_slice.copy()
                tmp["moneyness_F"] = tmp["strike"] / tmp["F"]
                j_atm = (tmp["moneyness_F"] - 1.0).abs().idxmin()
                sigma_true = float(tmp.loc[j_atm, "sigma_mkt_b76"])
            elif cfg.sigma_true_mode == "expiry_median":
                sigma_true = float(anchor_slice["sigma_mkt_b76"].median())
            elif cfg.sigma_true_mode == "bucket_median":
                sigma_true = float(calls_b["sigma_mkt_b76"].median())
            else:
                raise ValueError("sigma_true_mode must be 'expiry_atm', 'expiry_median', or 'bucket_median'.")

            # --------------------------------------------------------
            # Calibrate once per (rep_day, bucket)
            # --------------------------------------------------------
            sigma_bs = fit_sigma_bucket(cal_df) if cfg.run_bs else float("nan")

            jd_params = None
            if cfg.run_jd:
                jd_params, _ = calibrate_jd_bucket(cal_df)

            h_params = None
            if cfg.run_heston:
                h_params, _ = calibrate_heston_bucket(cal_df, u_max=cfg.heston_u_max, n_points=cfg.heston_n_points)

            # RL fit (treated as forward-based: feed spot=F)
            Qmodel, q_fit = None, None
            if cfg.run_qlbs:
                Qmodel = QLBSModel(
                    is_call_option=True,
                    checkpoint=cfg.qlbs_checkpoint_tmpl.format(rl=cfg.qlbs_risk_lambda),
                    anchor_T=28 / 252,
                )
                time_to_exp = cal_df["tau"].to_numpy(dtype=float)
                strikes = cal_df["strike"].to_numpy(dtype=float)
                observed = cal_df["C_mid"].to_numpy(dtype=float)
                weights = 1.0 / np.power(np.clip(observed, 1.0, None), 1.0)
                q_fit = Qmodel.fit(
                    spot=float(F0),
                    time_to_expiries=time_to_exp,
                    strikes=strikes,
                    r=float(r),
                    risk_lambda=float(cfg.qlbs_risk_lambda),
                    friction=float(cfg.friction),
                    observed_prices=observed,
                    weights=weights,
                    sigma_guess=0.3,
                    mu_guess=0.0,
                    n_epochs=2000,
                )

            Rmodel, r_fit = None, None
            if cfg.run_rlop:
                Rmodel = RLOPModel(
                    is_call_option=True,
                    checkpoint=cfg.rlop_checkpoint,
                    anchor_T=28 / 252,
                )
                time_to_exp = cal_df["tau"].to_numpy(dtype=float)
                strikes = cal_df["strike"].to_numpy(dtype=float)
                observed = cal_df["C_mid"].to_numpy(dtype=float)
                weights = 1.0 / np.power(np.clip(observed, 1.0, None), 1.0)
                r_fit = Rmodel.fit(
                    spot=float(F0),
                    time_to_expiries=time_to_exp,
                    strikes=strikes,
                    r=float(r),
                    risk_lambda=float(cfg.rlop_risk_lambda),
                    friction=float(cfg.friction),
                    observed_prices=observed,
                    weights=weights,
                    sigma_guess=0.3,
                    mu_guess=0.0,
                    n_epochs=2000,
                )

            # --------------------------------------------------------
            # Simulate ONE SPOT path set for this (day,bucket)
            # --------------------------------------------------------
            seed_db = int(cfg.seed + day_i * 10007 + b_center * 101)
            S_paths = simulate_spot_paths(
                S0=S0,
                r=r,
                q=q,
                sigma_true=sigma_true,
                T=T,
                n_steps=cfg.n_steps,
                n_paths=cfg.n_paths,
                seed=seed_db,
            )

            # --------------------------------------------------------
            # Build model spot-price scalar functions (price(S,K,tau,r,q))
            # All B76-based models are called on forward F(S,tau).
            # --------------------------------------------------------
            def _F_of(S: float, tau: float) -> float:
                return float(spot_to_forward(np.array([S]), r, q, tau)[0])

            price_fns: Dict[str, Callable[[float, float, float, float, float], float]] = {}

            if cfg.run_bs and np.isfinite(sigma_bs):
                def bs_price_scalar(S: float, K_: float, tau_: float, r_: float, q_: float) -> float:
                    return bs_call_from_spot(float(S), float(K_), float(tau_), float(r_), float(q_), float(sigma_bs))
                price_fns["BS"] = bs_price_scalar

            if cfg.run_jd and jd_params is not None:
                sigma_jd, lam, muJ, dJ = float(jd_params["sigma"]), float(jd_params["lam"]), float(jd_params["muJ"]), float(jd_params["deltaJ"])
                def jd_price_scalar(S: float, K_: float, tau_: float, r_: float, q_: float) -> float:
                    F = _F_of(float(S), float(tau_))
                    return merton_price_call_b76(float(F), float(K_), float(tau_), float(r_), sigma_jd, lam, muJ, dJ)
                price_fns["JD"] = jd_price_scalar

            if cfg.run_heston and h_params is not None:
                def sv_price_scalar(S: float, K_: float, tau_: float, r_: float, q_: float) -> float:
                    F = _F_of(float(S), float(tau_))
                    return heston_price_call(float(F), float(K_), float(tau_), float(r_), h_params, u_max=cfg.heston_u_max, n_points=cfg.heston_n_points)
                price_fns["SV"] = sv_price_scalar

            if cfg.run_qlbs and q_fit is not None:
                sigma_q, mu_q = float(q_fit.sigma), float(q_fit.mu)
                def qlbs_price_scalar(S: float, K_: float, tau_: float, r_: float, q_: float) -> float:
                    F = _F_of(float(S), float(tau_))  # feed forward as "spot" to match your training convention
                    pred = Qmodel.predict(
                        spot=float(F),
                        time_to_expiries=np.array([float(tau_)]),
                        strikes=np.array([float(K_)]),
                        r=float(r_),
                        risk_lambda=float(cfg.qlbs_risk_lambda),
                        friction=float(cfg.friction),
                        sigma_fit=float(sigma_q),
                        mu_fit=float(mu_q),
                    )
                    return float(pred.estimated_prices[0])
                price_fns["QLBS"] = qlbs_price_scalar

            if cfg.run_rlop and r_fit is not None:
                sigma_r, mu_r = float(r_fit.sigma), float(r_fit.mu)
                def rlop_price_scalar(S: float, K_: float, tau_: float, r_: float, q_: float) -> float:
                    F = _F_of(float(S), float(tau_))
                    pred = Rmodel.predict(
                        spot=float(F),
                        time_to_expiries=np.array([float(tau_)]),
                        strikes=np.array([float(K_)]),
                        r=float(r_),
                        risk_lambda=float(cfg.rlop_risk_lambda),
                        friction=float(cfg.friction),
                        sigma_fit=float(sigma_r),
                        mu_fit=float(mu_r),
                    )
                    return float(pred.estimated_prices[0])
                price_fns["RLOP"] = rlop_price_scalar

            # --------------------------------------------------------
            # Precompute delta grids per option & model
            # BS uses analytic delta; others use grid interpolation.
            # --------------------------------------------------------
            S_grid = build_S_grid(S0=S0, sigma_true=sigma_true, T=T, grid_n=cfg.grid_n, n_std=cfg.grid_n_std)

            delta_grids: Dict[Tuple[str, int], np.ndarray] = {}  # (model, opt_idx) -> (n_steps, grid_n)

            for opt_idx in range(len(selected)):
                K = float(selected.loc[opt_idx, "strike"])
                for model in price_fns.keys():
                    if model == "BS":
                        continue
                    dg = precompute_delta_grid(
                        price_scalar_fn=price_fns[model],
                        K=K,
                        r=r,
                        q=q,
                        T=T,
                        n_steps=cfg.n_steps,
                        S_grid=S_grid,
                    )
                    delta_grids[(model, opt_idx)] = dg

            # --------------------------------------------------------
            # Hedge each selected strike and produce rows
            # --------------------------------------------------------
            period_str = f"{cfg.start_date}..{cfg.end_date}"

            for opt_idx in range(len(selected)):
                K = float(selected.loc[opt_idx, "strike"])
                premium0_mkt = float(selected.loc[opt_idx, "C_mid"])
                m_target = float(selected.loc[opt_idx, "moneyness_target"])
                m_actual_F = float(selected.loc[opt_idx, "moneyness_actual_F"])
                m_actual_S = float(selected.loc[opt_idx, "moneyness_actual_S"])
                sigma_mkt = float(selected.loc[opt_idx, "sigma_mkt_b76"])

                # Diagnostics for Issue B: "world-consistent" BS premium at t=0 using sigma_true
                premium0_world_bs = bs_call_from_spot(S0, K, T, r, q, sigma_true)
                prem_gap_mkt_minus_world = premium0_mkt - float(premium0_world_bs)

                for model in ["BS", "JD", "SV", "QLBS", "RLOP"]:
                    if model not in price_fns and model != "BS":
                        continue
                    if model == "BS" and (not cfg.run_bs or not np.isfinite(sigma_bs)):
                        continue

                    # Build get_delta(step_idx, S_vec)
                    if model == "BS":
                        dt = T / cfg.n_steps
                        def get_delta(step_idx: int, S_vec: np.ndarray) -> np.ndarray:
                            tau = max(T - step_idx * dt, 0.0)
                            return bs_delta_from_spot(S_vec, K, tau, r, q, float(sigma_bs))
                    else:
                        dg = delta_grids[(model, opt_idx)]
                        def get_delta(step_idx: int, S_vec: np.ndarray) -> np.ndarray:
                            return interp_delta(dg, S_grid, step_idx, S_vec)

                    metrics = hedge_short_call_paths(
                        S_paths=S_paths,
                        K=K,
                        r=r,
                        T=T,
                        premium0=premium0_mkt,  # FAIRNESS: always market premium across models
                        friction=float(cfg.friction),
                        get_delta=get_delta,
                    )

                    row = {
                        "symbol": cfg.symbol,
                        "date": pd.Timestamp(rep_day),
                        "period": period_str,
                        "bucket": b_label,
                        "anchor_expiration": pd.Timestamp(anchor_exp),
                        "T_years": float(T),
                        "T_days_365": float(T) * 365.0,
                        "S0": float(S0),
                        "F0": float(F0),
                        "r": float(r),
                        "q_hat": float(q),
                        "sigma_true": float(sigma_true),
                        "sigma_mkt_b76": float(sigma_mkt),
                        "K": float(K),
                        "premium0_mkt": float(premium0_mkt),
                        "premium0_world_bs": float(premium0_world_bs),
                        "premium_gap_mkt_minus_world": float(prem_gap_mkt_minus_world),
                        "moneyness_target": float(m_target),
                        "moneyness_actual_F": float(m_actual_F),
                        "moneyness_actual_S": float(m_actual_S),
                        "model": model,
                    }
                    row.update(metrics)
                    results.append(row)

            # --------------------------------------------------------
            # Save per (day,bucket) artifacts
            # --------------------------------------------------------
            if outp is not None:
                subdir = outp / f"day={pd.Timestamp(rep_day).date()}_bucket={b_label}"
                subdir.mkdir(parents=True, exist_ok=True)

                # Save calibration surface + selection
                cal_df_min = cal_df[["date","symbol","expiration","tau","strike","C_mid","F","DF","r","S","q_hat","sigma_mkt_b76","moneyness_F","moneyness_S"]].copy()
                cal_df_min.to_csv(subdir / "calibration_surface.csv", index=False)

                sel_min = selected[["expiration","tau","strike","C_mid","F","S","r","q_hat","sigma_mkt_b76","moneyness_target","moneyness_actual_F","moneyness_actual_S"]].copy()
                sel_min.to_csv(subdir / "selected_options.csv", index=False)

                # Save calibrated params
                calib = {
                    "sigma_bs": None if not np.isfinite(sigma_bs) else float(sigma_bs),
                    "jd_params": jd_params,
                    "heston_params": h_params,
                    "qlbs_fit": None if q_fit is None else {"sigma": float(q_fit.sigma), "mu": float(q_fit.mu), "risk_lambda": float(cfg.qlbs_risk_lambda)},
                    "rlop_fit": None if r_fit is None else {"sigma": float(r_fit.sigma), "mu": float(r_fit.mu), "risk_lambda": float(cfg.rlop_risk_lambda)},
                    "anchor": {
                        "rep_day": pd.Timestamp(rep_day).isoformat(),
                        "bucket": b_label,
                        "anchor_expiration": pd.Timestamp(anchor_exp).isoformat(),
                        "T_years": float(T),
                        "S0": float(S0),
                        "F0": float(F0),
                        "r": float(r),
                        "q_hat": float(q),
                        "sigma_true": float(sigma_true),
                        "seed": int(seed_db),
                    },
                    "delta_grid": {"grid_n": int(cfg.grid_n), "grid_n_std": float(cfg.grid_n_std)},
                    "heston_integration": {"u_max": float(cfg.heston_u_max), "n_points": int(cfg.heston_n_points)},
                }
                _json_dump(subdir / "calibrated_params.json", calib)

                # Save paths + grids
                np.savez_compressed(subdir / "paths.npz", S_paths=S_paths.astype(np.float32), S_grid=S_grid.astype(np.float32))
                # Save delta grids (keyed)
                dz = {}
                for (model, opt_idx), dg in delta_grids.items():
                    dz[f"delta_{model}_opt{opt_idx}"] = dg.astype(np.float32)
                np.savez_compressed(subdir / "delta_grids.npz", **dz)

    if not results:
        raise RuntimeError("No hedging results produced. Check filters or data coverage.")

    hedge_res = pd.DataFrame(results)

    if outp is not None:
        hedge_res.to_csv(outp / "hedge_res.csv", index=False)

    return hedge_res


# ============================================================
# Publication-style aggregation matching your IVRMSE layout
# (Whole / <1 / >1 / >1.03) using moneyness TARGETS
# ============================================================

def make_hedging_publication_table(
    hedge_res: pd.DataFrame,
    symbol: str,
    metric: str = "RMSE_hedge",
    buckets: List[int] = [28],
    decimals: int = 4,
    out_dir: Optional[str] = None,
    basename: str = "table_hedging_pub",
) -> pd.DataFrame:
    """
    Table layout aligned with your IVRMSE structure:
      Sections: Whole sample / Moneyness <1 / >1 / >1.03
      Rows: one per bucket like 'SPY (τ=28d)'
      Columns: BS, JD, SV, QLBS, RLOP (present only)

    Aggregation:
      mean over rep_days and targets in the bin.
    """
    if hedge_res.empty:
        raise ValueError("hedge_res is empty.")

    models = [m for m in ["BS", "JD", "SV", "QLBS", "RLOP"] if m in hedge_res["model"].unique()]
    bucket_labels = [f"{int(d)}d" for d in buckets]

    def _bin_mask(df: pd.DataFrame, bin_name: str) -> pd.Series:
        mt = df["moneyness_target"].astype(float)
        if bin_name == "Whole":
            return pd.Series(True, index=df.index)
        if bin_name == "<1":
            return mt < 1.0
        if bin_name == ">1":
            return mt > 1.0
        if bin_name == ">1.03":
            return mt > 1.03
        raise ValueError("unknown bin")

    sections = [("Whole sample", "Whole"), ("Moneyness <1", "<1"), ("Moneyness >1", ">1"), ("Moneyness >1.03", ">1.03")]
    rows = []

    for sec_name, sec_key in sections:
        for b in bucket_labels:
            row = {"Moneyness": sec_name, "Asset": f"{symbol} (τ={b})"}
            sub = hedge_res[(hedge_res["bucket"] == b) & _bin_mask(hedge_res, sec_key)]
            for m in models:
                v = sub.loc[sub["model"] == m, metric]
                row[m] = float(v.mean()) if len(v) else np.nan
            rows.append(row)

    table = pd.DataFrame(rows)
    for m in models:
        table[m] = table[m].round(decimals)

    if out_dir:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        csv_path = outp / f"{basename}_{metric}.csv"
        table.to_csv(csv_path, index=False)

    return table


# ============================================================
# Convenience: SPY 20Q1 run
# ============================================================

def hedging_spy20():
    df = pd.read_csv("data/spy_preprocessed_calls_20q1.csv")

    cfg = HedgingRunConfig(
        symbol="SPY",
        start_date="2020-01-06",
        end_date="2020-03-30",
        buckets=(28,),
        n_rep_days=60,
        rep_day_mode="even",
        moneyness_targets=(0.90, 0.97, 1.00, 1.03, 1.10),
        calibrate_on="expiry",
        sigma_true_mode="expiry_atm",  # good default for stability/fairness
        n_steps=28,
        n_paths=2000,
        friction=4e-3,
        seed=123,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        # Saving
        out_dir="SPY_20Q1_hedging_v7",
        run_tag=None,
    )

    hedge_res = run_dynamic_hedging_spy_like(df_all=df, cfg=cfg)

    print("\nDynamic hedging summary (SPY 20Q1):")
    summary = (
        hedge_res
        .groupby(["bucket", "moneyness_target", "model"], as_index=False)
        .agg(
            RMSE_mean=("RMSE_hedge", "mean"),
            RMSE_std=("RMSE_hedge", "std"),
            cost_mean=("avg_cost", "mean"),
            shortfall_mean=("shortfall_prob", "mean"),
            N=("RMSE_hedge", "size"),
        )
        .sort_values(["bucket", "moneyness_target", "RMSE_mean"])
    )
    print(summary.to_string(index=False))

    pub_tbl = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="SPY",
        metric="RMSE_hedge",
        buckets=[28],
        decimals=4,
        out_dir="SPY_20Q1_hedging_v7",
        basename="table_hedging_pub",
    )
    print("\nPublication-style hedging table (aligned with IVRMSE layout):")
    print(pub_tbl.to_string(index=False))


def hedging_xop20():
    df = pd.read_csv("data/xop_preprocessed_calls_20q1.csv")

    cfg = HedgingRunConfig(
        symbol="XOP",
        start_date="2020-01-06",
        end_date="2020-03-30",
        buckets=(28,),
        n_rep_days=60,
        rep_day_mode="even",
        moneyness_targets=(0.90, 0.97, 1.00, 1.03, 1.10),
        calibrate_on="expiry",
        sigma_true_mode="expiry_atm",  # good default for stability/fairness
        n_steps=28,
        n_paths=2000,
        friction=4e-3,
        seed=123,
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        # Saving
        out_dir="XOP_20Q1_hedging_v7",
        run_tag=None,
    )

    hedge_res = run_dynamic_hedging_spy_like(df_all=df, cfg=cfg)

    print("\nDynamic hedging summary (XOP 20Q1):")
    summary = (
        hedge_res
        .groupby(["bucket", "moneyness_target", "model"], as_index=False)
        .agg(
            RMSE_mean=("RMSE_hedge", "mean"),
            RMSE_std=("RMSE_hedge", "std"),
            cost_mean=("avg_cost", "mean"),
            shortfall_mean=("shortfall_prob", "mean"),
            N=("RMSE_hedge", "size"),
        )
        .sort_values(["bucket", "moneyness_target", "RMSE_mean"])
    )
    print(summary.to_string(index=False))

    pub_tbl = make_hedging_publication_table(
        hedge_res=hedge_res,
        symbol="XOP",
        metric="RMSE_hedge",
        buckets=[28],
        decimals=4,
        out_dir="XOP_20Q1_hedging_v7",
        basename="table_hedging_pub",
    )
    print("\nPublication-style hedging table (aligned with IVRMSE layout):")
    print(pub_tbl.to_string(index=False))

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
        #hedging_xop25()


# python -m unittest playground/options_pricing_baselines_v5.py
