"""
AlgoForge — Options Pricing & Risk Engine
Python: Black-Scholes analytics, Monte Carlo simulation, IV surface, FastAPI
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Optional

# ── Option Parameters ──────────────────────────────────────────────────────────

@dataclass
class OptionParams:
    S: float        # Spot price
    K: float        # Strike
    T: float        # Time to expiry (years)
    r: float        # Risk-free rate (continuous)
    sigma: float    # Volatility (annualised)
    q: float = 0.0  # Dividend yield
    option_type: str = "C"   # "C" or "P"


# ── Black-Scholes Closed Form ──────────────────────────────────────────────────

def _d1d2(p: OptionParams):
    d1 = (np.log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma**2) * p.T) / \
         (p.sigma * np.sqrt(p.T))
    d2 = d1 - p.sigma * np.sqrt(p.T)
    return d1, d2


def bs_price(p: OptionParams) -> float:
    """Black-Scholes option price."""
    if p.T <= 0:
        intrinsic = max(p.S - p.K, 0) if p.option_type == "C" else max(p.K - p.S, 0)
        return intrinsic
    d1, d2 = _d1d2(p)
    disc_S = p.S * np.exp(-p.q * p.T)
    disc_K = p.K * np.exp(-p.r * p.T)
    if p.option_type == "C":
        return disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    else:
        return disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def bs_greeks(p: OptionParams) -> Greeks:
    """All Black-Scholes Greeks in one pass."""
    price = bs_price(p)
    if p.T <= 1e-6:
        return Greeks(price=price, delta=0, gamma=0, theta=0, vega=0, rho=0)

    d1, d2 = _d1d2(p)
    sq_T   = np.sqrt(p.T)
    nd1    = norm.pdf(d1)
    disc_q = np.exp(-p.q * p.T)
    disc_r = np.exp(-p.r * p.T)

    gamma = disc_q * nd1 / (p.S * p.sigma * sq_T)
    vega  = p.S * disc_q * nd1 * sq_T / 100   # per 1% vol move

    if p.option_type == "C":
        delta = disc_q * norm.cdf(d1)
        theta = (-(p.S * disc_q * nd1 * p.sigma) / (2 * sq_T)
                 - p.r * p.K * disc_r * norm.cdf(d2)
                 + p.q * p.S * disc_q * norm.cdf(d1)) / 365
        rho   = p.K * p.T * disc_r * norm.cdf(d2) / 100
    else:
        delta = disc_q * (norm.cdf(d1) - 1)
        theta = (-(p.S * disc_q * nd1 * p.sigma) / (2 * sq_T)
                 + p.r * p.K * disc_r * norm.cdf(-d2)
                 - p.q * p.S * disc_q * norm.cdf(-d1)) / 365
        rho   = -p.K * p.T * disc_r * norm.cdf(-d2) / 100

    return Greeks(
        price=round(price, 4),
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        theta=round(theta, 4),
        vega=round(vega, 4),
        rho=round(rho, 4)
    )


# ── Implied Volatility (Brent's Method) ───────────────────────────────────────

def implied_vol(
    market_price: float,
    p: OptionParams,
    vol_bounds: tuple[float, float] = (0.001, 10.0)
) -> Optional[float]:
    """Compute implied volatility via root-finding."""
    if market_price <= 0:
        return None

    def objective(sigma: float) -> float:
        pp = OptionParams(p.S, p.K, p.T, p.r, sigma, p.q, p.option_type)
        return bs_price(pp) - market_price

    try:
        lo_price = bs_price(OptionParams(p.S, p.K, p.T, p.r, vol_bounds[0], p.q, p.option_type))
        hi_price = bs_price(OptionParams(p.S, p.K, p.T, p.r, vol_bounds[1], p.q, p.option_type))
        if objective(vol_bounds[0]) * objective(vol_bounds[1]) > 0:
            return None
        iv = brentq(objective, vol_bounds[0], vol_bounds[1], xtol=1e-6, maxiter=100)
        return round(iv, 6)
    except Exception:
        return None


# ── Monte Carlo Pricer (Geometric Brownian Motion) ────────────────────────────

def mc_price_option(
    p: OptionParams,
    num_paths: int = 50_000,
    num_steps: int = 252,
    seed: int = 42
) -> dict:
    """
    Price European option via Monte Carlo with antithetic variates.
    Returns price, std error, and confidence interval.
    """
    np.random.seed(seed)
    dt = p.T / num_steps
    drift = (p.r - p.q - 0.5 * p.sigma**2) * dt
    diffusion = p.sigma * np.sqrt(dt)

    # Antithetic variates: use Z and -Z
    half = num_paths // 2
    Z = np.random.standard_normal((half, num_steps))
    Z_full = np.concatenate([Z, -Z], axis=0)

    log_returns = drift + diffusion * Z_full
    log_S_paths = np.log(p.S) + np.cumsum(log_returns, axis=1)
    S_T = np.exp(log_S_paths[:, -1])

    if p.option_type == "C":
        payoffs = np.maximum(S_T - p.K, 0)
    else:
        payoffs = np.maximum(p.K - S_T, 0)

    discounted = np.exp(-p.r * p.T) * payoffs
    price      = discounted.mean()
    std_err    = discounted.std() / np.sqrt(num_paths)

    return {
        "mc_price": round(price, 4),
        "std_error": round(std_err, 4),
        "ci_lower": round(price - 1.96 * std_err, 4),
        "ci_upper": round(price + 1.96 * std_err, 4),
        "bs_price": round(bs_price(p), 4),     # for comparison
        "num_paths": num_paths
    }


# ── Volatility Surface Builder ─────────────────────────────────────────────────

def build_iv_surface(
    S: float,
    r: float = 0.05,
    q: float = 0.0,
    strikes_pct: list = None,
    tenors_years: list = None
) -> pd.DataFrame:
    """
    Build a synthetic IV surface (smile + term structure).
    In production, replace with real market quotes.
    """
    if strikes_pct is None:
        strikes_pct = [0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20]
    if tenors_years is None:
        tenors_years = [1/12, 2/12, 3/12, 6/12, 1.0, 1.5, 2.0]

    rows = []
    for T in tenors_years:
        for m in strikes_pct:
            K = S * m
            # Synthetic smile: parabolic in moneyness + term structure premium
            atm_vol = 0.20 + 0.02 * np.sqrt(T)
            skew    = -0.10 * (m - 1.0)
            convex  = 0.15 * (m - 1.0)**2
            iv      = atm_vol + skew + convex + np.random.normal(0, 0.005)
            iv      = max(0.05, iv)
            rows.append({
                "tenor": round(T, 4),
                "strike": round(K, 2),
                "moneyness": round(m, 3),
                "implied_vol": round(iv, 4),
                "option_type": "C" if m >= 1.0 else "P"
            })

    return pd.DataFrame(rows)


# ── Scenario Analysis ──────────────────────────────────────────────────────────

def scenario_analysis(p: OptionParams) -> list[dict]:
    """
    Compute P&L for a range of spot moves and vol shocks.
    Useful for risk-managing a single option position.
    """
    base_price = bs_price(p)
    scenarios = []

    for spot_move in [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]:
        for vol_shock in [-0.05, 0, 0.05]:
            new_S     = p.S * (1 + spot_move)
            new_sigma = max(0.01, p.sigma + vol_shock)
            new_p     = OptionParams(new_S, p.K, p.T, p.r, new_sigma, p.q, p.option_type)
            new_price = bs_price(new_p)
            pnl       = new_price - base_price
            scenarios.append({
                "spot_move_pct":  round(spot_move * 100, 1),
                "vol_shock":      round(vol_shock * 100, 1),
                "new_spot":       round(new_S, 2),
                "new_vol":        round(new_sigma * 100, 1),
                "option_price":   round(new_price, 4),
                "pnl":            round(pnl, 4),
                "pnl_pct":        round(pnl / base_price * 100, 2) if base_price else 0
            })

    return scenarios


# ── FastAPI Application ────────────────────────────────────────────────────────

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AlgoForge Options API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/api/greeks")
def get_greeks(
    S: float = Query(100), K: float = Query(100),
    T: float = Query(0.25), r: float = Query(0.05),
    sigma: float = Query(0.20), q: float = Query(0.0),
    option_type: str = Query("C")
):
    p = OptionParams(S, K, T, r, sigma, q, option_type)
    g = bs_greeks(p)
    return {"price": g.price, "delta": g.delta, "gamma": g.gamma,
            "theta": g.theta, "vega": g.vega, "rho": g.rho}


@app.get("/api/monte-carlo")
def get_mc(
    S: float = 100, K: float = 100, T: float = 0.25,
    r: float = 0.05, sigma: float = 0.20, q: float = 0.0,
    option_type: str = "C", paths: int = 10000
):
    p = OptionParams(S, K, T, r, sigma, q, option_type)
    return mc_price_option(p, num_paths=min(paths, 100_000))


@app.get("/api/iv-surface")
def get_iv_surface(S: float = 100, r: float = 0.05):
    df = build_iv_surface(S, r)
    return df.to_dict(orient="records")


@app.get("/api/scenario")
def get_scenario(
    S: float = 100, K: float = 100, T: float = 0.25,
    r: float = 0.05, sigma: float = 0.20, option_type: str = "C"
):
    p = OptionParams(S, K, T, r, sigma, option_type=option_type)
    return scenario_analysis(p)


@app.get("/api/iv-smile")
def get_iv_smile(
    S: float = 100, T: float = 0.25, r: float = 0.05
):
    """Compute IV smile for a single tenor across strikes."""
    strikes = [S * m for m in [0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20]]
    result = []
    for K in strikes:
        # Synthetic market price with smile
        m = K / S
        atm_vol = 0.20
        skew = -0.10 * (m - 1.0)
        convex = 0.15 * (m - 1.0)**2
        true_iv = max(0.05, atm_vol + skew + convex)
        p = OptionParams(S, K, T, r, true_iv)
        mkt_price = bs_price(p)
        iv = implied_vol(mkt_price, OptionParams(S, K, T, r, 0.20))
        result.append({"strike": round(K, 2), "moneyness": round(m, 3),
                       "implied_vol": round(true_iv * 100, 2),
                       "option_price": round(mkt_price, 4)})
    return result


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AlgoForge"}
