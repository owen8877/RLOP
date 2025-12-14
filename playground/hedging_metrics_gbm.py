from __future__ import annotations

import json
import math
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
    x = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


# ============================================================
# Black–76 call price (expects FORWARD F)
# ============================================================

def b76_price_call(F: float, K: float, tau: float, r: float, sigma: float) -> float:
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
    Nd1 = float(_norm_cdf(np.array([d1]))[0])
    Nd2 = float(_norm_cdf(np.array([d2]))[0])
    return DF * (F * Nd1 - K * Nd2)


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
    F = float(spot_to_forward(np.array([S]), r, q, tau)[0])
    return b76_price_call(F, K, tau, r, sigma)


def bs_delta_from_spot(S: np.ndarray, K: float, tau: float, r: float, q: float, sigma: float) -> np.ndarray:
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
# Data adapter + one-day call surface (European fast-path)
# ============================================================

def adapter_eur_calls_to_summarizer(calls_out: pd.DataFrame) -> pd.DataFrame:
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
    q_clip_lo: float = -0.10,
    q_clip_hi: float = 0.20,
) -> pd.DataFrame:
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

    hdr = df.groupby(["date", "symbol", "expiration"], as_index=False).agg(
        F=("F", "first"),
        DF=("DF", "first"),
        S=("S", "first"),
        PV_div=("PV_div", "first"),
        tau=("tau", "first"),
        r=("r", "first"),
    )

    r_infer = -np.log(np.clip(hdr["DF"].to_numpy(), 1e-12, None)) / np.maximum(hdr["tau"].to_numpy(), 1e-12)
    hdr["r"] = np.where(np.isfinite(hdr["r"].to_numpy()), hdr["r"].to_numpy(), r_infer)

    # Prepaid forward consistency: S ≈ DF*F + PV_div
    S_infer = hdr["DF"].to_numpy() * hdr["F"].to_numpy() + hdr["PV_div"].to_numpy()
    hdr["S"] = np.where(
        np.isfinite(hdr["S"].to_numpy()) & (hdr["S"].to_numpy() > 0),
        hdr["S"].to_numpy(),
        S_infer,
    )

    tau = np.maximum(hdr["tau"].to_numpy(), 1e-12)
    F = np.maximum(hdr["F"].to_numpy(), 1e-12)
    S = np.maximum(hdr["S"].to_numpy(), 1e-12)
    r = hdr["r"].to_numpy()

    hdr["q_hat_raw"] = r - np.log(F / S) / tau
    hdr["q_hat"] = hdr["q_hat_raw"].clip(float(q_clip_lo), float(q_clip_hi))

    calls = df.groupby(["date", "symbol", "expiration", "tau", "strike"], as_index=False).agg(C_mid=("mid", "first"))
    calls = calls.merge(
        hdr[["date", "symbol", "expiration", "F", "DF", "r", "S", "PV_div", "q_hat_raw", "q_hat"]],
        on=["date", "symbol", "expiration"],
        how="left",
    )

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
    x0 = np.asarray(x0, dtype=float)

    if _HAVE_SCIPY:
        res = minimize(func, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
        return res.x if res.success else x0

    # lightweight local grid fallback
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
# Merton Jump–Diffusion (JD) under forward measure
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
    if tau <= 0:
        DF = math.exp(-r * max(tau, 0.0))
        return DF * max(F - K, 0.0)

    k = math.exp(muJ + 0.5 * deltaJ * deltaJ) - 1.0
    F_adj = F * math.exp(-lam * k * tau)
    L = lam * tau

    p = math.exp(-L)
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
# Heston SV (CF + integration) under forward measure
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
# Delta engines
# ============================================================

def build_S_grid(S0: float, sigma_ref: float, T: float, grid_n: int = 61, n_std: float = 4.0) -> np.ndarray:
    width = float(n_std) * float(sigma_ref) * math.sqrt(max(T, 1e-12))
    lo = math.log(max(S0, 1e-12)) - width
    hi = math.log(max(S0, 1e-12)) + width
    return np.exp(np.linspace(lo, hi, int(grid_n))).astype(float)


def precompute_delta_grid(
    price_scalar_fn: Callable[[float, float, float, float, float], float],
    K: float,
    r: float,
    q: float,
    tau_grid: np.ndarray,   # length n_steps (rebalance times)
    S_grid: np.ndarray,
) -> np.ndarray:
    Sg = np.asarray(S_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    n_steps = int(tau_grid.size)
    grid_n = int(Sg.size)

    prices = np.empty((n_steps, grid_n), dtype=float)
    for t in range(n_steps):
        tau = max(float(tau_grid[t]), 0.0)
        prices[t, :] = np.array(
            [price_scalar_fn(float(Sg[i]), float(K), tau, float(r), float(q)) for i in range(grid_n)],
            dtype=float,
        )

    deltas = np.gradient(prices, Sg, axis=1)
    return deltas.astype(float)


def interp_delta(delta_grid: np.ndarray, S_grid: np.ndarray, step_idx: int, S_vec: np.ndarray) -> np.ndarray:
    S_vec = np.asarray(S_vec, dtype=float)
    smin, smax = float(S_grid[0]), float(S_grid[-1])
    Sv = np.clip(S_vec, smin, smax)
    return np.interp(Sv, S_grid, delta_grid[int(step_idx), :]).astype(float)


def delta_local_fd(
    price_scalar_fn: Callable[[float, float, float, float, float], float],
    S_vec: np.ndarray,
    K: float,
    tau: float,
    r: float,
    q: float,
    rel_eps: float = 1e-3,
    abs_min: float = 1e-4,
) -> np.ndarray:
    S_vec = np.asarray(S_vec, dtype=float)
    h = np.maximum(abs_min, rel_eps * np.maximum(np.abs(S_vec), 1.0))
    out = np.empty_like(S_vec, dtype=float)
    for i in range(S_vec.size):
        Sp = float(S_vec[i] + h[i])
        Sm = float(max(1e-12, S_vec[i] - h[i]))
        Cp = float(price_scalar_fn(Sp, float(K), float(tau), float(r), float(q)))
        Cm = float(price_scalar_fn(Sm, float(K), float(tau), float(r), float(q)))
        out[i] = (Cp - Cm) / (Sp - Sm)
    return out


# ============================================================
# Hedging engines
#   - Paths engine: MC (many paths) or 1-path (historical)
#   - For historical: we EXTRACT (error_T, cost_total) from the 1-path run.
# ============================================================

def hedge_short_call_paths(
    S_paths: np.ndarray,
    K: float,
    r: float,
    q: float,
    T: float,
    premium0: float,
    friction: float,
    get_delta: Callable[[int, np.ndarray], np.ndarray],
    dt_years: Optional[np.ndarray] = None,
    return_pathwise: bool = False,
) -> Dict[str, Any]:
    """
    Delta hedge a SHORT call with:
      - cash accrues at r
      - shares receive dividend yield q (credited to cash)

    If dt_years is provided: length n_steps, variable time increments.
    Otherwise: constant dt = T/n_steps.

    return_pathwise:
      - if True, also returns 'error_vec' and 'cost_vec' (use ONLY for n_paths small).
    """
    S_paths = np.asarray(S_paths, dtype=float)
    n_paths, n_steps_plus_1 = S_paths.shape
    n_steps = n_steps_plus_1 - 1

    if dt_years is None:
        dt_years = np.full(n_steps, float(T) / n_steps, dtype=float)
    else:
        dt_years = np.asarray(dt_years, dtype=float)
        if dt_years.size != n_steps:
            raise ValueError(f"dt_years size {dt_years.size} != n_steps {n_steps}")
        T = float(dt_years.sum())

    S0 = S_paths[:, 0]
    ST = S_paths[:, -1]
    payoff = np.maximum(ST - float(K), 0.0)

    prem = float(premium0) * np.ones_like(S0, dtype=float)

    # Initial hedge
    delta = get_delta(0, S0)
    trade0 = delta

    trading_cost = np.zeros_like(S0, dtype=float)
    cost0 = friction * np.abs(trade0) * S0 if friction > 0.0 else 0.0
    trading_cost += cost0
    cash = prem - trade0 * S0 - cost0

    # Rebalance at times 1..n_steps-1
    for t in range(1, n_steps):
        dt = float(dt_years[t - 1])

        # cash accrual
        cash *= math.exp(r * dt)

        # dividend carry credit for shares held over (t-1, t)
        if q != 0.0:
            S_prev = S_paths[:, t - 1]
            cash += delta * S_prev * (math.exp(q * dt) - 1.0)

        St = S_paths[:, t]
        new_delta = get_delta(t, St)
        trade = new_delta - delta

        if friction > 0.0:
            cost = friction * np.abs(trade) * St
            trading_cost += cost
            cash -= cost

        cash -= trade * St
        delta = new_delta

    # Final interval (n_steps-1 -> n_steps): accrue cash + dividends, no rebalance
    dt_last = float(dt_years[-1])
    cash *= math.exp(r * dt_last)
    if q != 0.0:
        S_prev = S_paths[:, n_steps - 1]
        cash += delta * S_prev * (math.exp(q * dt_last) - 1.0)

    V_T = cash + delta * ST
    error = V_T - payoff

    err2_mean = float(np.mean(error**2))
    rmse = float(math.sqrt(err2_mean))
    avg_cost = float(np.mean(trading_cost))
    shortfall_prob = float(np.mean(error < 0.0))

    var_1pct = float(np.quantile(error, 0.01))
    tail = error[error <= var_1pct]
    cvar_1pct = float(np.mean(tail)) if tail.size else var_1pct

    out: Dict[str, Any] = {
        "RMSE_hedge": rmse,
        "err2_mean": err2_mean,
        "avg_cost": avg_cost,
        "shortfall_prob": shortfall_prob,
        "shortfall_1pct": var_1pct,
        "cvar_1pct": cvar_1pct,
        "err_mean": float(np.mean(error)),
        "err_std": float(np.std(error, ddof=0)),
    }
    if return_pathwise:
        out["error_vec"] = error
        out["cost_vec"] = trading_cost
    return out


# ============================================================
# Selection helper: pick strikes closest to moneyness targets
# ============================================================

def pick_options_by_moneyness_targets(anchor_slice: pd.DataFrame, targets: List[float]) -> pd.DataFrame:
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
# Historical SPOT extraction + path builder
#   IMPORTANT UPDATE:
#     - maturity_date_used = last available trading day in the path (<= contract expiry)
#     - we hedge/payoff on that maturity_date_used
#     - expiry_gap_days tracks how far it is from contract expiry (e.g., Saturday expiry -> Friday close gap=1)
# ============================================================

def build_spot_history_from_df(df_pre: pd.DataFrame, symbol: str) -> pd.Series:
    d = df_pre[df_pre["act_symbol"] == symbol].copy()
    if d.empty:
        return pd.Series(dtype=float)

    d["date"] = pd.to_datetime(d["date"]).dt.normalize()

    # primary: S
    if "S" in d.columns and d["S"].notna().any():
        s1 = d.groupby("date")["S"].median()
        s1 = s1[np.isfinite(s1) & (s1 > 0)]
    else:
        s1 = pd.Series(dtype=float)

    if len(s1) >= 5:
        return s1.sort_index()

    # fallback: DF*F + PV_div
    pv_div = d["PV_div"] if "PV_div" in d.columns else 0.0
    s2_raw = d["DF"].astype(float) * d["F"].astype(float) + pd.to_numeric(pv_div, errors="coerce").fillna(0.0)
    d["_S_infer"] = s2_raw
    s2 = d.groupby("date")["_S_infer"].median()
    s2 = s2[np.isfinite(s2) & (s2 > 0)]
    return s2.sort_index()


def build_historical_path(
    spot: pd.Series,
    start_date: pd.Timestamp,
    contract_expiry: pd.Timestamp,
    *,
    mode: str,
    fixed_steps: int,
    max_expiry_gap_days: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[pd.Timestamp]], str, Optional[pd.Timestamp], Optional[int]]:
    """
    Returns:
      S_path: (1, n_steps+1)
      dt_years: (n_steps,)
      dates: list length n_steps+1
      status
      maturity_date_used
      expiry_gap_days = (contract_expiry - maturity_date_used).days
    """
    if spot.empty:
        return None, None, None, "no_spot_history", None, None

    start_date = pd.Timestamp(start_date).normalize()
    contract_expiry = pd.Timestamp(contract_expiry).normalize()

    all_dates = spot.index

    # start alignment: next available trading date
    if start_date not in all_dates:
        pos = all_dates.searchsorted(start_date)
        if pos >= len(all_dates):
            return None, None, None, "start_date_after_spot_range", None, None
        start_date = all_dates[pos]

    if mode == "fixed_steps":
        pos = all_dates.get_loc(start_date)
        end_pos = pos + int(fixed_steps)
        if end_pos >= len(all_dates):
            return None, None, None, "insufficient_future_spot_for_fixed_steps", None, None
        dates = list(all_dates[pos:end_pos + 1])
        maturity_date_used = pd.Timestamp(dates[-1]).normalize()
        expiry_gap_days = int((contract_expiry - maturity_date_used).days)

    elif mode == "to_expiry":
        # all trading dates in [start_date, contract_expiry]
        sub = spot.loc[(spot.index >= start_date) & (spot.index <= contract_expiry)]
        if sub.empty or len(sub) < 2:
            return None, None, None, "insufficient_spot_between_start_and_expiry", None, None
        dates = list(sub.index)
        maturity_date_used = pd.Timestamp(dates[-1]).normalize()
        expiry_gap_days = int((contract_expiry - maturity_date_used).days)
        if expiry_gap_days > int(max_expiry_gap_days):
            return None, None, None, f"expiry_gap_too_large(gap_days={expiry_gap_days})", None, None

    else:
        return None, None, None, f"unknown_hist_mode({mode})", None, None

    Svec = spot.loc[dates].to_numpy(dtype=float)
    if not np.all(np.isfinite(Svec)) or np.any(Svec <= 0):
        return None, None, None, "nonfinite_or_nonpositive_spot_in_path", None, None

    dt_days = np.diff(pd.Index(dates).to_numpy(dtype="datetime64[D]")).astype(int)
    dt_years = (dt_days / 365.0).astype(float)
    if np.any(dt_years <= 0):
        return None, None, None, "nonpositive_dt_in_hist_path", None, None

    S_path = Svec.reshape(1, -1)
    return S_path, dt_years, [pd.Timestamp(x).normalize() for x in dates], "ok", maturity_date_used, expiry_gap_days


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
# Main runner
# ============================================================

@dataclass
class HedgingRunConfig:
    symbol: str = "SPY"
    start_date: str = "2020-01-06"
    end_date: str = "2020-03-30"
    buckets: Tuple[int, ...] = (28,)
    n_rep_days: int = 5
    rep_day_mode: str = "even"  # "even" or "middle" or "all"
    moneyness_targets: Tuple[float, ...] = (0.90, 0.97, 1.00, 1.03, 1.10)

    calibrate_on: str = "expiry"  # "expiry" or "bucket"
    sigma_true_mode: str = "expiry_atm"  # "expiry_atm" or "expiry_median" or "bucket_median"

    # World
    world: str = "gbm"  # "gbm" or "historical"

    # GBM params
    n_steps: int = 28
    n_paths: int = 2000
    friction: float = 4e-3
    seed: int = 123

    # Historical world params
    hist_mode: str = "to_expiry"         # "to_expiry" or "fixed_steps"
    hist_fixed_steps: int = 28
    hist_max_expiry_gap_days: int = 1    # allow Sat expiry (gap=1)
    hist_delta_method: str = "local_fd"  # "local_fd" or "grid"

    # Model toggles
    run_bs: bool = True
    run_jd: bool = True
    run_heston: bool = True
    run_qlbs: bool = True
    run_rlop: bool = True

    # Delta grid controls (speed lever)
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

    # q clipping safety
    q_clip_lo: float = -0.10
    q_clip_hi: float = 0.20

    # Saving
    out_dir: Optional[str] = None
    run_tag: Optional[str] = None


def run_dynamic_hedging_spy_like(df_all: pd.DataFrame, cfg: HedgingRunConfig) -> pd.DataFrame:
    df_pre = adapter_eur_calls_to_summarizer(df_all)

    # For calibration/selection: restrict to requested window
    df_pre["date"] = pd.to_datetime(df_pre["date"]).dt.normalize()
    mask = (
        (df_pre["act_symbol"] == cfg.symbol)
        & (df_pre["date"] >= pd.Timestamp(cfg.start_date))
        & (df_pre["date"] <= pd.Timestamp(cfg.end_date))
    )
    df = df_pre.loc[mask].copy()
    if df.empty:
        raise ValueError(f"No rows for {cfg.symbol} in [{cfg.start_date}, {cfg.end_date}].")

    # Historical spot series uses ALL available dates in df_pre for that symbol
    spot_hist = build_spot_history_from_df(df_pre, cfg.symbol)

    # Representative days
    unique_days = sorted(df["date"].unique())
    if cfg.rep_day_mode not in ("even", "middle", "all"):
        raise ValueError("rep_day_mode must be 'even', 'middle', or 'all'.")

    if cfg.rep_day_mode == "all":
        rep_days = list(unique_days)
    elif cfg.rep_day_mode == "middle":
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
        run_tag = cfg.run_tag or f"{cfg.symbol}_{cfg.start_date}_{cfg.end_date}_{cfg.world}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        calls = prepare_calls_one_day_symbol(
            df_day,
            tau_floor_days=3,
            q_clip_lo=float(cfg.q_clip_lo),
            q_clip_hi=float(cfg.q_clip_hi),
        )
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
            contract_exp = pd.Timestamp(anchor_row["expiration"]).normalize()

            anchor_slice = calls_b[calls_b["expiration"] == contract_exp].copy()
            if anchor_slice.empty:
                anchor_slice = calls_b.copy()

            # Market quantities for this (rep_day, bucket)
            T_mkt = float(anchor_row["tau"])
            r = float(anchor_row["r"])
            q_raw = float(anchor_row["q_hat_raw"]) if "q_hat_raw" in anchor_row.index else float(anchor_row["q_hat"])
            q = float(anchor_row["q_hat"]) if "q_hat" in anchor_row.index else float(q_raw)
            q = float(np.clip(q, cfg.q_clip_lo, cfg.q_clip_hi))

            S0_anchor = float(anchor_row["S"])
            F0 = float(anchor_row["F"])

            # Select strikes
            selected = pick_options_by_moneyness_targets(anchor_slice, targets)

            # Calibration surface
            if cfg.calibrate_on not in ("expiry", "bucket"):
                raise ValueError("calibrate_on must be 'expiry' or 'bucket'.")
            cal_df = anchor_slice if cfg.calibrate_on == "expiry" else calls_b

            # sigma_true (used for GBM world and for grid span)
            if cfg.sigma_true_mode == "expiry_atm":
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

            # RL fit
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
            # World paths + time grids
            # --------------------------------------------------------
            world_status = "ok"
            path_dates: Optional[List[pd.Timestamp]] = None
            dt_years: Optional[np.ndarray] = None
            maturity_date_used: Optional[pd.Timestamp] = None
            expiry_gap_days: Optional[int] = None

            if cfg.world == "gbm":
                T_used = float(T_mkt)
                n_steps_run = int(cfg.n_steps)
                tau_grid = np.array([max(T_used - (T_used / n_steps_run) * t, 0.0) for t in range(n_steps_run)], dtype=float)
                seed_db = int(cfg.seed + day_i * 10007 + b_center * 101)
                S_paths = simulate_spot_paths(
                    S0=S0_anchor,
                    r=r,
                    q=q,
                    sigma_true=sigma_true,
                    T=T_used,
                    n_steps=n_steps_run,
                    n_paths=int(cfg.n_paths),
                    seed=seed_db,
                )
                dt_years = None
                path_dates = None
                maturity_date_used = None
                expiry_gap_days = None
                T = T_used

            elif cfg.world == "historical":
                S_paths, dt_years, path_dates, world_status, maturity_date_used, expiry_gap_days = build_historical_path(
                    spot=spot_hist,
                    start_date=pd.Timestamp(rep_day).normalize(),
                    contract_expiry=contract_exp,
                    mode=str(cfg.hist_mode),
                    fixed_steps=int(cfg.hist_fixed_steps),
                    max_expiry_gap_days=int(cfg.hist_max_expiry_gap_days),
                )
                if world_status != "ok" or S_paths is None or dt_years is None or path_dates is None or maturity_date_used is None:
                    results.append({
                        "symbol": cfg.symbol,
                        "date": pd.Timestamp(rep_day),
                        "period": f"{cfg.start_date}..{cfg.end_date}",
                        "bucket": b_label,
                        "contract_expiration": pd.Timestamp(contract_exp),
                        "world": cfg.world,
                        "world_status": world_status,
                    })
                    continue

                n_steps_run = S_paths.shape[1] - 1

                # IMPORTANT: hedge/payoff to maturity_date_used (last trading day in path)
                tau_grid = np.array([(maturity_date_used - pd.Timestamp(path_dates[t])).days / 365.0 for t in range(n_steps_run)], dtype=float)
                tau_grid = np.clip(tau_grid, 0.0, None)

                T = float(dt_years.sum())

            else:
                raise ValueError("cfg.world must be 'gbm' or 'historical'.")

            # --------------------------------------------------------
            # Model price scalar fns: price(S,K,tau,r,q)
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
                    F = _F_of(float(S), float(tau_))
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
            # Delta grids (optional)
            # --------------------------------------------------------
            delta_grids: Dict[Tuple[str, int], np.ndarray] = {}
            S_grid = None

            use_grid = (cfg.world == "gbm") or (cfg.world == "historical" and cfg.hist_delta_method == "grid")
            if use_grid:
                S0_for_grid = float(S_paths[0, 0])
                S_grid = build_S_grid(S0=S0_for_grid, sigma_ref=sigma_true, T=float(max(T, 1e-12)), grid_n=cfg.grid_n, n_std=cfg.grid_n_std)
                for opt_idx in range(len(selected)):
                    K0 = float(selected.loc[opt_idx, "strike"])
                    for model in price_fns.keys():
                        if model == "BS":
                            continue
                        dg = precompute_delta_grid(
                            price_scalar_fn=price_fns[model],
                            K=K0,
                            r=r,
                            q=q,
                            tau_grid=tau_grid,
                            S_grid=S_grid,
                        )
                        delta_grids[(model, opt_idx)] = dg

            # --------------------------------------------------------
            # Hedge each selected strike
            # --------------------------------------------------------
            period_str = f"{cfg.start_date}..{cfg.end_date}"

            for opt_idx in range(len(selected)):
                K0 = float(selected.loc[opt_idx, "strike"])
                premium0_mkt = float(selected.loc[opt_idx, "C_mid"])
                m_target = float(selected.loc[opt_idx, "moneyness_target"])
                m_actual_F = float(selected.loc[opt_idx, "moneyness_actual_F"])
                m_actual_S = float(selected.loc[opt_idx, "moneyness_actual_S"])
                sigma_mkt = float(selected.loc[opt_idx, "sigma_mkt_b76"])

                # “World-consistent” BS premium diagnostic
                premium0_world_bs = bs_call_from_spot(float(S_paths[0, 0]), K0, float(max(T, 1e-12)), r, q, sigma_true)
                prem_gap_mkt_minus_world = premium0_mkt - float(premium0_world_bs)

                for model in ["BS", "JD", "SV", "QLBS", "RLOP"]:
                    if model != "BS" and model not in price_fns:
                        continue
                    if model == "BS" and (not cfg.run_bs or not np.isfinite(sigma_bs)):
                        continue

                    # delta(t,S): uses tau_grid[t]
                    if model == "BS":
                        def get_delta(step_idx: int, S_vec: np.ndarray) -> np.ndarray:
                            tau = float(tau_grid[int(step_idx)])
                            return bs_delta_from_spot(S_vec, K0, tau, r, q, float(sigma_bs))
                    else:
                        if use_grid:
                            dg = delta_grids[(model, opt_idx)]
                            def get_delta(step_idx: int, S_vec: np.ndarray) -> np.ndarray:
                                return interp_delta(dg, S_grid, step_idx, S_vec)  # type: ignore[arg-type]
                        else:
                            price_fn = price_fns[model]
                            def get_delta(step_idx: int, S_vec: np.ndarray) -> np.ndarray:
                                tau = float(tau_grid[int(step_idx)])
                                return delta_local_fd(price_fn, S_vec, K0, tau, r, q)

                    # IMPORTANT UPDATE:
                    #   - GBM world: keep aggregated metrics
                    #   - Historical world: extract the single-path error_T and cost_total
                    if cfg.world == "historical":
                        met = hedge_short_call_paths(
                            S_paths=S_paths,
                            K=K0,
                            r=r,
                            q=q,
                            T=float(T),
                            premium0=premium0_mkt,
                            friction=float(cfg.friction),
                            get_delta=get_delta,
                            dt_years=dt_years,
                            return_pathwise=True,  # n_paths=1 only
                        )
                        error_T = float(np.asarray(met["error_vec"])[0])
                        cost_total = float(np.asarray(met["cost_vec"])[0])
                        metrics = {"error_T": error_T, "cost_total": cost_total}
                    else:
                        metrics = hedge_short_call_paths(
                            S_paths=S_paths,
                            K=K0,
                            r=r,
                            q=q,
                            T=float(T),
                            premium0=premium0_mkt,
                            friction=float(cfg.friction),
                            get_delta=get_delta,
                            dt_years=dt_years,
                            return_pathwise=False,
                        )

                    row: Dict[str, Any] = {
                        "symbol": cfg.symbol,
                        "date": pd.Timestamp(rep_day),
                        "period": period_str,
                        "bucket": b_label,
                        "contract_expiration": pd.Timestamp(contract_exp),
                        "T_years_used": float(T),
                        "T_mkt_years": float(T_mkt),
                        "n_steps_run": int(S_paths.shape[1] - 1),
                        "S0_used": float(S_paths[0, 0]),
                        "S0_anchor": float(S0_anchor),
                        "F0": float(F0),
                        "r": float(r),
                        "q_hat_raw": float(q_raw),
                        "q_hat": float(q),
                        "sigma_true": float(sigma_true),
                        "sigma_mkt_b76": float(sigma_mkt),
                        "K": float(K0),
                        "premium0_mkt": float(premium0_mkt),
                        "premium0_world_bs": float(premium0_world_bs),
                        "premium_gap_mkt_minus_world": float(prem_gap_mkt_minus_world),
                        "moneyness_target": float(m_target),
                        "moneyness_actual_F": float(m_actual_F),
                        "moneyness_actual_S": float(m_actual_S),
                        "model": model,
                        "world": cfg.world,
                        "world_status": world_status,
                    }

                    if cfg.world == "historical":
                        row.update({
                            "path_start_date": pd.Timestamp(path_dates[0]).isoformat() if path_dates else None,
                            "maturity_date_used": pd.Timestamp(maturity_date_used).isoformat() if maturity_date_used is not None else None,
                            "expiry_gap_days": int(expiry_gap_days) if expiry_gap_days is not None else None,
                            "hist_mode": cfg.hist_mode,
                            "hist_delta_method": cfg.hist_delta_method,
                        })

                    row.update(metrics)
                    results.append(row)

            # --------------------------------------------------------
            # Save per (day,bucket) artifacts
            # --------------------------------------------------------
            if outp is not None:
                subdir = outp / f"day={pd.Timestamp(rep_day).date()}_bucket={b_label}"
                subdir.mkdir(parents=True, exist_ok=True)

                cal_cols = ["date","symbol","expiration","tau","strike","C_mid","F","DF","r","S","PV_div","q_hat_raw","q_hat","sigma_mkt_b76","moneyness_F","moneyness_S"]
                cal_df_min = cal_df[[c for c in cal_cols if c in cal_df.columns]].copy()
                cal_df_min.to_csv(subdir / "calibration_surface.csv", index=False)

                sel_cols = ["expiration","tau","strike","C_mid","F","S","r","q_hat_raw","q_hat","sigma_mkt_b76","moneyness_target","moneyness_actual_F","moneyness_actual_S"]
                sel_min = selected[[c for c in sel_cols if c in selected.columns]].copy()
                sel_min.to_csv(subdir / "selected_options.csv", index=False)

                calib = {
                    "world": cfg.world,
                    "world_status": world_status,
                    "sigma_bs": None if not np.isfinite(sigma_bs) else float(sigma_bs),
                    "jd_params": jd_params,
                    "heston_params": h_params,
                    "qlbs_fit": None if q_fit is None else {"sigma": float(q_fit.sigma), "mu": float(q_fit.mu), "risk_lambda": float(cfg.qlbs_risk_lambda)},
                    "rlop_fit": None if r_fit is None else {"sigma": float(r_fit.sigma), "mu": float(r_fit.mu), "risk_lambda": float(cfg.rlop_risk_lambda)},
                    "anchor": {
                        "rep_day": pd.Timestamp(rep_day).isoformat(),
                        "bucket": b_label,
                        "contract_expiration": pd.Timestamp(contract_exp).isoformat(),
                        "maturity_date_used": pd.Timestamp(maturity_date_used).isoformat() if maturity_date_used is not None else None,
                        "expiry_gap_days": int(expiry_gap_days) if expiry_gap_days is not None else None,
                        "T_years_used": float(T),
                        "T_mkt_years": float(T_mkt),
                        "S0_used": float(S_paths[0, 0]),
                        "S0_anchor": float(S0_anchor),
                        "F0": float(F0),
                        "r": float(r),
                        "q_hat_raw": float(q_raw),
                        "q_hat": float(q),
                        "sigma_true": float(sigma_true),
                    },
                    "timegrid": {
                        "n_steps_run": int(S_paths.shape[1] - 1),
                        "tau_grid_first_last": [float(tau_grid[0]), float(tau_grid[-1])] if len(tau_grid) else None,
                        "dt_years_sum": float(np.sum(dt_years)) if dt_years is not None else None,
                    },
                    "delta": {
                        "use_grid": bool(use_grid),
                        "grid_n": int(cfg.grid_n),
                        "grid_n_std": float(cfg.grid_n_std),
                        "hist_delta_method": cfg.hist_delta_method,
                    },
                    "hist": {
                        "hist_mode": cfg.hist_mode,
                        "hist_fixed_steps": int(cfg.hist_fixed_steps),
                        "hist_max_expiry_gap_days": int(cfg.hist_max_expiry_gap_days),
                    },
                    "heston_integration": {"u_max": float(cfg.heston_u_max), "n_points": int(cfg.heston_n_points)},
                    "q_clip": {"lo": float(cfg.q_clip_lo), "hi": float(cfg.q_clip_hi)},
                }
                _json_dump(subdir / "calibrated_params.json", calib)

                np.savez_compressed(subdir / "paths.npz", S_paths=S_paths.astype(np.float32))
                if cfg.world == "historical" and path_dates is not None and dt_years is not None:
                    _json_dump(subdir / "hist_path_dates.json", {"dates": [pd.Timestamp(x).isoformat() for x in path_dates]})
                    np.savez_compressed(subdir / "hist_dt_years.npz", dt_years=dt_years.astype(np.float32))

                if use_grid and S_grid is not None:
                    np.savez_compressed(subdir / "S_grid.npz", S_grid=S_grid.astype(np.float32))
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
# Publication table
#   - Historical world: aggregate from error_T / cost_total across hedges
#   - GBM world: use pooled RMSE via err2_mean; other metrics are means of per-(rep_day,option) metrics
# ============================================================

def make_hedging_publication_table(
    hedge_res: pd.DataFrame,
    symbol: str,
    metric: str = "RMSE_hedge",
    buckets: List[int] = [28],
    decimals: int = 4,
    out_dir: Optional[str] = None,
    basename: str = "table_hedging_pub",
    pooled_rmse_gbm: bool = True,
) -> pd.DataFrame:
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

    supported = {"RMSE_hedge", "avg_cost", "shortfall_prob", "shortfall_1pct", "cvar_1pct"}
    if metric not in supported:
        raise ValueError(f"Unsupported metric: {metric}")

    sections = [
        ("Whole sample", "Whole"),
        ("Moneyness <1", "<1"),
        ("Moneyness >1", ">1"),
        ("Moneyness >1.03", ">1.03"),
    ]

    def _agg_from_errors(sub: pd.DataFrame) -> float:
        err = sub["error_T"].to_numpy(dtype=float)
        if metric == "RMSE_hedge":
            return float(np.sqrt(np.mean(err * err)))
        if metric == "avg_cost":
            return float(sub["cost_total"].astype(float).mean())
        if metric == "shortfall_prob":
            return float(np.mean(err < 0.0))
        if metric == "shortfall_1pct":
            return float(np.quantile(err, 0.01))
        if metric == "cvar_1pct":
            var = float(np.quantile(err, 0.01))
            tail = err[err <= var]
            return float(np.mean(tail)) if tail.size else var
        raise ValueError("bad metric")

    rows = []
    for sec_name, sec_key in sections:
        for b in bucket_labels:
            row = {"Moneyness": sec_name, "Asset": f"{symbol} (τ={b})"}
            sub_all = hedge_res[(hedge_res["bucket"] == b) & _bin_mask(hedge_res, sec_key)].copy()

            for m in models:
                sub = sub_all[sub_all["model"] == m]
                if sub.empty:
                    row[m] = np.nan
                    continue

                # Historical rows contain error_T/cost_total; GBM rows contain RMSE_hedge/err2_mean/avg_cost...
                if "error_T" in sub.columns and sub["error_T"].notna().any():
                    row[m] = _agg_from_errors(sub.dropna(subset=["error_T"]))
                else:
                    if metric == "RMSE_hedge" and pooled_rmse_gbm and "err2_mean" in sub.columns:
                        row[m] = float(np.sqrt(np.mean(sub["err2_mean"].astype(float))))
                    elif metric == "avg_cost":
                        row[m] = float(np.mean(sub["avg_cost"].astype(float)))
                    else:
                        row[m] = float(np.mean(sub[metric].astype(float)))

            rows.append(row)

    table = pd.DataFrame(rows)
    for m in models:
        table[m] = table[m].round(decimals)

    if out_dir:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        table.to_csv(outp / f"{basename}_{metric}.csv", index=False)
    return table


# ============================================================
# Convenience runners
# ============================================================

def hedging_spy20(world: str = "gbm"):
    df = pd.read_csv("data/spy_preprocessed_calls_20q1.csv")
    out_dir = f"SPY_20Q1_hedging_v_{world}"

    cfg = HedgingRunConfig(
        symbol="SPY",
        start_date="2020-01-06",
        end_date="2020-03-30",
        buckets=(28,),
        n_rep_days=90,
        rep_day_mode="even",
        moneyness_targets=(0.90, 0.97, 1.00, 1.03, 1.10),
        calibrate_on="expiry",
        sigma_true_mode="expiry_atm",
        world=world,
        n_steps=28,
        n_paths=2000,
        friction=4e-3,
        seed=123,
        hist_mode="to_expiry",
        hist_fixed_steps=28,
        hist_max_expiry_gap_days=1,
        hist_delta_method="local_fd",
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        out_dir=out_dir,
        run_tag=None,
    )

    hedge_res = run_dynamic_hedging_spy_like(df_all=df, cfg=cfg)

    print(f"\nPublication tables (SPY 20Q1, world={world}):")
    for met in ["RMSE_hedge", "avg_cost", "shortfall_prob", "shortfall_1pct", "cvar_1pct"]:
        tbl = make_hedging_publication_table(
            hedge_res=hedge_res,
            symbol="SPY",
            metric=met,
            buckets=[28],
            decimals=4,
            out_dir=out_dir,
            basename="table_hedging_pub",
            pooled_rmse_gbm=True,
        )
        print(f"\n{met}:")
        print(tbl.to_string(index=False))


def hedging_xop20(world: str = "gbm"):
    df = pd.read_csv("data/xop_preprocessed_calls_20q1.csv")
    out_dir = f"XOP_20Q1_hedging_v_{world}"

    cfg = HedgingRunConfig(
        symbol="XOP",
        start_date="2020-01-06",
        end_date="2020-03-30",
        buckets=(28,),
        n_rep_days=90,
        rep_day_mode="even",
        moneyness_targets=(0.90, 0.97, 1.00, 1.03, 1.10),
        calibrate_on="expiry",
        sigma_true_mode="expiry_atm",
        world=world,
        n_steps=28,
        n_paths=2000,
        friction=4e-3,
        seed=123,
        hist_mode="to_expiry",
        hist_fixed_steps=28,
        hist_max_expiry_gap_days=1,
        hist_delta_method="local_fd",
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        out_dir=out_dir,
        run_tag=None,
    )

    hedge_res = run_dynamic_hedging_spy_like(df_all=df, cfg=cfg)

    print(f"\nPublication tables (XOP 20Q1, world={world}):")
    for met in ["RMSE_hedge", "avg_cost", "shortfall_prob", "shortfall_1pct", "cvar_1pct"]:
        tbl = make_hedging_publication_table(
            hedge_res=hedge_res,
            symbol="XOP",
            metric=met,
            buckets=[28],
            decimals=4,
            out_dir=out_dir,
            basename="table_hedging_pub",
            pooled_rmse_gbm=True,
        )
        print(f"\n{met}:")
        print(tbl.to_string(index=False))


def hedging_spy25(world: str = "gbm"):
    df = pd.read_csv("data/spy_preprocessed_calls_25.csv")
    out_dir = f"SPY_25Q2_hedging_v_{world}"

    cfg = HedgingRunConfig(
        symbol="SPY",
        start_date="2025-04-01",
        end_date="2025-06-30",
        buckets=(28,),
        n_rep_days=90,
        rep_day_mode="even",
        moneyness_targets=(0.90, 0.97, 1.00, 1.03, 1.10),
        calibrate_on="expiry",
        sigma_true_mode="expiry_atm",
        world=world,
        n_steps=28,
        n_paths=2000,
        friction=4e-3,
        seed=123,
        hist_mode="to_expiry",
        hist_fixed_steps=28,
        hist_max_expiry_gap_days=1,
        hist_delta_method="local_fd",
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        out_dir=out_dir,
        run_tag=None,
    )

    hedge_res = run_dynamic_hedging_spy_like(df_all=df, cfg=cfg)

    print(f"\nPublication tables (SPY 25Q2, world={world}):")
    for met in ["RMSE_hedge", "avg_cost", "shortfall_prob", "shortfall_1pct", "cvar_1pct"]:
        tbl = make_hedging_publication_table(
            hedge_res=hedge_res,
            symbol="SPY",
            metric=met,
            buckets=[28],
            decimals=4,
            out_dir=out_dir,
            basename="table_hedging_pub",
            pooled_rmse_gbm=True,
        )
        print(f"\n{met}:")
        print(tbl.to_string(index=False))


def hedging_xop25(world: str = "gbm"):
    df = pd.read_csv("data/xop_preprocessed_calls_25.csv")
    out_dir = f"XOP_25Q2_hedging_v_{world}"

    cfg = HedgingRunConfig(
        symbol="XOP",
        start_date="2025-04-01",
        end_date="2025-06-30",
        buckets=(28,),
        n_rep_days=90,
        rep_day_mode="even",
        moneyness_targets=(0.90, 0.97, 1.00, 1.03, 1.10),
        calibrate_on="expiry",
        sigma_true_mode="expiry_atm",
        world=world,
        n_steps=28,
        n_paths=2000,
        friction=4e-3,
        seed=123,
        hist_mode="to_expiry",
        hist_fixed_steps=28,
        hist_max_expiry_gap_days=1,
        hist_delta_method="local_fd",
        run_bs=True,
        run_jd=True,
        run_heston=True,
        run_qlbs=True,
        run_rlop=True,
        out_dir=out_dir,
        run_tag=None,
    )

    hedge_res = run_dynamic_hedging_spy_like(df_all=df, cfg=cfg)

    print(f"\nPublication tables (XOP 25Q2, world={world}):")
    for met in ["RMSE_hedge", "avg_cost", "shortfall_prob", "shortfall_1pct", "cvar_1pct"]:
        tbl = make_hedging_publication_table(
            hedge_res=hedge_res,
            symbol="XOP",
            metric=met,
            buckets=[28],
            decimals=4,
            out_dir=out_dir,
            basename="table_hedging_pub",
            pooled_rmse_gbm=True,
        )
        print(f"\n{met}:")
        print(tbl.to_string(index=False))


class Test(TestCase):
    def test_main(self):
        # GBM world
        hedging_spy20(world="gbm")
        hedging_xop20(world="gbm")

        # Historical world
        # hedging_spy20(world="historical")
        # hedging_xop20(world="historical")

        # Uncomment if your 25Q2 datasets include enough SPOT coverage:
        hedging_spy25(world="gbm")
        hedging_xop25(world="gbm")
        # hedging_spy25(world="historical")
        # hedging_xop25(world="historical")
