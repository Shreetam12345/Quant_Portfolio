# AlgoForge — Options Pricing & Risk Platform

> **Real-time options analytics** with Black-Scholes, Monte Carlo simulation, IV surface visualization, and C++ pricing engine.
## Architecture

```
┌─────────────────────────────────────────────────┐
│                  React Dashboard                 │
│  Greeks · IV Surface · MC Simulator · Scenario  │
└──────────────────┬──────────────────────────────┘
                   │ REST
┌──────────────────▼──────────────────────────────┐
│              FastAPI (Python)                    │
│  ┌──────────────────────────────────────────┐   │
│  │  Black-Scholes Analytics    (scipy/numpy)│   │
│  │  Implied Vol (Brent's Method)            │   │
│  │  Monte Carlo (GBM + Antithetic)          │   │
│  │  IV Surface Builder                      │   │
│  │  Scenario / Stress-Test Matrix           │   │
│  └────────────────┬─────────────────────────┘   │
│  ┌────────────────▼─────────────────────────┐   │
│  │  C++ Module (pybind11)                   │   │
│  │  bs_price · bs_greeks                    │   │
│  │  binomial_american · monte_carlo_price   │   │
│  └──────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           PostgreSQL Database                    │
│  option_contracts · market_snapshots            │
│  option_greeks · positions · portfolio_risk     │
│  Views: v_greeks_latest · v_iv_surface          │
└─────────────────────────────────────────────────┘
```

---

## Mathematical Foundation

### Black-Scholes Formula

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$
$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

Where:
$$d_1 = \frac{\ln(S/K) + (r - q + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

### Greeks
| Greek | Formula | Interpretation |
|---|---|---|
| $\Delta$ | $e^{-qT}N(d_1)$ | Spot sensitivity |
| $\Gamma$ | $\frac{e^{-qT}N'(d_1)}{S\sigma\sqrt{T}}$ | Delta's rate of change |
| $\Theta$ | (time decay formula) | Daily P&L from time decay |
| $\mathcal{V}$ | $Se^{-qT}N'(d_1)\sqrt{T}/100$ | Vol sensitivity per 1% |
| $\rho$ | $KTe^{-rT}N(d_2)/100$ | Rate sensitivity per 1% |

### Monte Carlo (GBM + Antithetic Variates)
$$S_T = S_0 \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T} \cdot Z\right]$$

Antithetic variates: simulate pairs $(Z, -Z)$ to reduce variance by ~50%.

$$\hat{C} = e^{-rT} \cdot \frac{1}{N}\sum_{i=1}^{N}\max(S_T^{(i)} - K, 0)$$

### Binomial Tree (American Options — Cox-Ross-Rubinstein)
$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$

Backward induction with early exercise check at each node.

---

## SQL Layer

| Table | Purpose |
|---|---|
| `underlyings` | Instrument master |
| `option_contracts` | Calls/puts with strike & expiry |
| `market_snapshots` | Spot, IV, rate time series |
| `option_greeks` | Computed prices + Greeks per snapshot |
| `positions` | Portfolio holdings |
| `portfolio_risk` | Net Greeks & VaR aggregation |
| `mc_runs` | Monte Carlo run log |

Key views: `v_greeks_latest`, `v_iv_surface`, `v_portfolio_greeks`

---

## C++ Module (`options_engine.cpp`)

| Function | Description |
|---|---|
| `bs_price(S,K,T,r,σ,q,type)` | Closed-form BS price |
| `bs_greeks(...)` | Returns `GreeksResult` struct |
| `binomial_american(...)` | CRR tree, 200 steps |
| `monte_carlo_price(...)` | GBM + antithetic, returns `MCResult` |

**Compile:**
```bash
cd backend/cpp
g++ -O2 -shared -fPIC $(python3-config --includes --ldflags) \
    -I$(python3 -c "import pybind11; print(pybind11.get_include())") \
    options_engine.cpp -o options_engine.so

# QuantAlpha — Statistical Arbitrage Platform

> **Pairs trading engine** using cointegration, z-score signals, ML filtering, and C++ performance layer.
## Architecture

```
┌─────────────────────────────────────────────────┐
│                  React Dashboard                 │
│    Equity Curve · Signals · Pair Scanner        │
└──────────────────┬──────────────────────────────┘
                   │ REST API
┌──────────────────▼──────────────────────────────┐
│              FastAPI (Python)                    │
│  ┌──────────────────────────────────────────┐   │
│  │  Cointegration Analysis  (statsmodels)   │   │
│  │  Z-Score Signal Generator                │   │
│  │  ML Signal Filter  (sklearn LogReg)      │   │
│  │  Backtesting Engine  (vectorised NumPy)  │   │
│  └────────────────┬─────────────────────────┘   │
│  ┌────────────────▼─────────────────────────┐   │
│  │  C++ Module (pybind11)                   │   │
│  │  rolling_zscore · ou_halflife            │   │
│  │  historical_var · max_drawdown · sharpe  │   │
│  └──────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│            PostgreSQL Database                   │
│  raw_ohlcv · prices_clean · coint_pairs         │
│  signals · backtest_runs                         │
│  Views: v_rolling_vol · v_live_signals          │
└─────────────────────────────────────────────────┘
```

---

## Mathematical Foundation

### Cointegration (Engle-Granger)
Two price series $P_A$ and $P_B$ are cointegrated if a linear combination:
$$\epsilon_t = P_A(t) - \beta \cdot P_B(t)$$
is stationary (I(0)). $\beta$ is estimated via OLS.

### Ornstein-Uhlenbeck Half-Life
The spread follows:
$$d\epsilon_t = \kappa(\mu - \epsilon_t)dt + \sigma dW_t$$
Half-life: $\tau_{1/2} = \frac{\ln 2}{\kappa}$, where $\kappa$ is estimated from OLS regression of $\Delta\epsilon_t$ on $\epsilon_{t-1}$.

### Z-Score Signals
$$z_t = \frac{\epsilon_t - \mu_{t,w}}{\sigma_{t,w}}$$
- Entry Long:  $z_t < -2$
- Entry Short: $z_t > +2$  
- Exit:        $|z_t| < 0.5$

### Sharpe Ratio
$$SR = \frac{\bar{r} - r_f}{\sigma_r} \cdot \sqrt{252}$$

---

## SQL Layer

| Table | Purpose |
|---|---|
| `raw_ohlcv` | Raw price ingestion from data vendor |
| `prices_clean` | Outlier-removed, log-return enriched prices |
| `coint_pairs` | Discovered pairs with hedge ratios |
| `signals` | Real-time z-score signals |
| `backtest_runs` | JSONB-parameterised backtest results |

Key function: `flag_outliers(symbol, window)` — IQR-based outlier detection.

---

## Python Stack

| Module | Role |
|---|---|
| `statsmodels.coint` | Engle-Granger cointegration test |
| `statsmodels.OLS` | Hedge ratio & OU regression |
| `sklearn.LogisticRegression` | ML signal quality filter |
| `numpy` | Vectorised backtest loop |
| `FastAPI` | REST API layer |

---

## C++ Module (`spread_calc.cpp`)

Exposed via `pybind11`:

| Function | Description |
|---|---|
| `rolling_zscore(spread, window)` | O(n·w) rolling z-score |
| `ou_halflife(spread)` | OU process parameter estimation |
| `historical_var(pnl, confidence)` | Returns `(VaR, CVaR)` |
| `max_drawdown(equity)` | Maximum drawdown computation |
| `sharpe_ratio(returns, ann)` | Annualised Sharpe |

**Compile:**
```bash
cd backend/cpp
g++ -O2 -shared -fPIC $(python3-config --includes --ldflags) \
    -I$(python3 -c "import pybind11; print(pybind11.get_include())") \
    spread_calc.cpp -o spread_calc.so
