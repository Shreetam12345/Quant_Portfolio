"""
QuantAlpha — Statistical Arbitrage Engine
Python Layer: Data ingestion, cointegration analysis, ML signal filtering, backtesting
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import Optional
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── Data Fetching (yfinance shim) ──────────────────────────────────────────────

def fetch_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices. Falls back to synthetic data if yfinance unavailable."""
    try:
        import yfinance as yf
        raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
        return raw["Close"].dropna()
    except Exception:
        # Synthetic GBM data for demo / testing
        np.random.seed(42)
        dates = pd.bdate_range(start, end)
        n = len(dates)
        data = {}
        base = {"GS": 350, "MS": 85, "JPM": 140, "BAC": 35}
        for sym in symbols:
            s0 = base.get(sym, 100)
            log_ret = np.random.normal(0.0003, 0.015, n)
            data[sym] = s0 * np.exp(np.cumsum(log_ret))
        return pd.DataFrame(data, index=dates)


# ── Cointegration Discovery ────────────────────────────────────────────────────

@dataclass
class CointPair:
    sym_a: str
    sym_b: str
    pvalue: float
    hedge_ratio: float
    half_life: float
    spread: pd.Series = field(default_factory=pd.Series)

    def zscore(self, window: int = 20) -> pd.Series:
        mu = self.spread.rolling(window).mean()
        sigma = self.spread.rolling(window).std()
        return (self.spread - mu) / sigma


def compute_half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life from OLS regression on lagged spread."""
    s = spread.dropna()
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    reg = OLS(delta, sm.add_constant(lag)).fit()
    lam = reg.params.iloc[1]
    if lam >= 0:
        return np.nan
    return -np.log(2) / lam


def find_coint_pairs(
    prices: pd.DataFrame,
    pvalue_threshold: float = 0.05
) -> list[CointPair]:
    """Scan all symbol pairs for cointegration using Engle-Granger test."""
    syms = list(prices.columns)
    pairs: list[CointPair] = []

    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            series_a = prices[a].dropna()
            series_b = prices[b].dropna()
            idx = series_a.index.intersection(series_b.index)
            if len(idx) < 60:
                continue

            sa, sb = series_a[idx], series_b[idx]
            _, pval, _ = coint(sa, sb)

            if pval < pvalue_threshold:
                reg = OLS(sa, sm.add_constant(sb)).fit()
                hedge = reg.params.iloc[1]
                spread = sa - hedge * sb
                hl = compute_half_life(spread)
                if 1 < hl < 120:          # sensible mean-reversion window
                    pairs.append(CointPair(
                        sym_a=a, sym_b=b,
                        pvalue=round(pval, 4),
                        hedge_ratio=round(hedge, 4),
                        half_life=round(hl, 2),
                        spread=spread
                    ))

    pairs.sort(key=lambda p: p.pvalue)
    return pairs


# ── ADF Stationarity Check ─────────────────────────────────────────────────────

def is_stationary(series: pd.Series, significance: float = 0.05) -> bool:
    result = adfuller(series.dropna())
    return result[1] < significance


# ── Signal Generation ──────────────────────────────────────────────────────────

def generate_signals(
    pair: CointPair,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 20
) -> pd.DataFrame:
    """
    Generate long/short signals based on z-score thresholds.
    Signal  = +1 → spread will mean-revert UP  (long A, short B)
    Signal  = -1 → spread will mean-revert DOWN (short A, long B)
    Signal  =  0 → exit / flat
    """
    z = pair.zscore(window)
    signals = pd.Series(0, index=z.index)
    position = 0

    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            continue
        if position == 0:
            if zi < -entry_z:
                position = 1
            elif zi > entry_z:
                position = -1
        elif position == 1 and zi >= -exit_z:
            position = 0
        elif position == -1 and zi <= exit_z:
            position = 0
        signals.iloc[i] = position

    return pd.DataFrame({
        "zscore": z,
        "signal": signals,
        "spread": pair.spread
    })


# ── Backtesting Engine ─────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    sharpe: float
    max_drawdown: float
    total_return: float
    win_rate: float
    num_trades: int
    calmar: float


def backtest_pair(
    prices: pd.DataFrame,
    pair: CointPair,
    initial_capital: float = 100_000,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 20,
    transaction_cost_bps: float = 5
) -> BacktestResult:
    """
    Dollar-neutral backtest: allocate capital equally between legs.
    Each leg: capital/2 notional.
    """
    sig_df = generate_signals(pair, entry_z, exit_z, window)
    pa = prices[pair.sym_a].reindex(sig_df.index)
    pb = prices[pair.sym_b].reindex(sig_df.index)

    capital = initial_capital
    equity = []
    trades_log = []
    prev_signal = 0
    cost_factor = transaction_cost_bps / 10_000

    for i in range(1, len(sig_df)):
        row = sig_df.iloc[i]
        prev = sig_df.iloc[i - 1]
        signal = row["signal"]
        date = sig_df.index[i]

        # Daily PnL from holding position overnight
        ret_a = (pa.iloc[i] - pa.iloc[i - 1]) / pa.iloc[i - 1]
        ret_b = (pb.iloc[i] - pb.iloc[i - 1]) / pb.iloc[i - 1]

        leg_notional = capital / 2
        if prev["signal"] == 1:       # long A, short B
            pnl = leg_notional * ret_a - leg_notional * ret_b
        elif prev["signal"] == -1:    # short A, long B
            pnl = -leg_notional * ret_a + leg_notional * ret_b
        else:
            pnl = 0

        # Transaction costs on signal change
        if signal != prev_signal:
            pnl -= capital * cost_factor
            if signal != 0:
                trades_log.append({"date": date, "signal": signal,
                                   "zscore": row["zscore"], "capital": capital})
        capital += pnl
        equity.append(capital)
        prev_signal = signal

    equity_series = pd.Series(equity, index=sig_df.index[1:])
    daily_returns = equity_series.pct_change().dropna()

    # ── Performance Metrics ──
    total_return = (equity_series.iloc[-1] / initial_capital - 1) * 100
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0

    trades_df = pd.DataFrame(trades_log)
    win_rate = 0.0
    if not trades_df.empty:
        wins = (trades_df["zscore"].shift(-1).fillna(0) * trades_df["signal"] < 0).sum()
        win_rate = wins / len(trades_df)

    return BacktestResult(
        equity_curve=equity_series,
        trades=trades_df,
        sharpe=round(sharpe, 3),
        max_drawdown=round(max_dd, 2),
        total_return=round(total_return, 2),
        win_rate=round(win_rate, 3),
        num_trades=len(trades_df),
        calmar=round(calmar, 3)
    )


# ── ML Signal Filter (Logistic Regression) ────────────────────────────────────

def train_signal_filter(
    sig_df: pd.DataFrame,
    prices: pd.DataFrame,
    pair: CointPair,
    lookback: int = 5
) -> dict:
    """
    Train a logistic regression to filter out low-quality signals.
    Features: z-score, spread momentum, volatility regime.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import classification_report

    pa = prices[pair.sym_a].reindex(sig_df.index)
    pb = prices[pair.sym_b].reindex(sig_df.index)

    df = sig_df.copy()
    df["ret_a"] = pa.pct_change()
    df["ret_b"] = pb.pct_change()
    df["spread_mom"] = df["spread"].diff(lookback)
    df["spread_vol"] = df["spread"].rolling(lookback * 2).std()
    df["zscore_lag1"] = df["zscore"].shift(1)
    df["zscore_lag2"] = df["zscore"].shift(2)
    df["future_ret"] = df["spread"].shift(-5) - df["spread"]
    df["profitable"] = (df["future_ret"] * df["signal"] > 0).astype(int)
    df = df.dropna()
    df = df[df["signal"] != 0]

    if len(df) < 20:
        return {"model": None, "accuracy": 0.0, "report": "Insufficient data"}

    feat_cols = ["zscore", "zscore_lag1", "zscore_lag2",
                 "spread_mom", "spread_vol", "ret_a", "ret_b"]
    X = df[feat_cols].values
    y = df["profitable"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=3)
    model = LogisticRegression(max_iter=500, C=0.5)
    accs = []
    for train_idx, test_idx in tscv.split(X_scaled):
        model.fit(X_scaled[train_idx], y[train_idx])
        accs.append(model.score(X_scaled[test_idx], y[test_idx]))

    model.fit(X_scaled, y)
    report = classification_report(y, model.predict(X_scaled), output_dict=True)

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": round(np.mean(accs), 3),
        "feature_cols": feat_cols,
        "report": report
    }


# ── FastAPI App ────────────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI(title="QuantAlpha API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_cache: dict = {}

def get_data():
    if "prices" not in _cache:
        symbols = ["GS", "MS", "JPM", "BAC"]
        prices = fetch_prices(symbols, "2021-01-01", "2024-12-31")
        pairs = find_coint_pairs(prices)
        _cache["prices"] = prices
        _cache["pairs"] = pairs
    return _cache["prices"], _cache["pairs"]


@app.get("/api/pairs")
def list_pairs():
    _, pairs = get_data()
    return [
        {"sym_a": p.sym_a, "sym_b": p.sym_b,
         "pvalue": p.pvalue, "hedge_ratio": p.hedge_ratio,
         "half_life": p.half_life}
        for p in pairs[:5]
    ]


@app.get("/api/backtest/{sym_a}/{sym_b}")
def run_backtest(sym_a: str, sym_b: str,
                 entry_z: float = 2.0, exit_z: float = 0.5):
    prices, pairs = get_data()
    pair = next((p for p in pairs
                 if p.sym_a == sym_a and p.sym_b == sym_b), None)
    if not pair:
        pair = CointPair(sym_a=sym_a, sym_b=sym_b,
                         pvalue=0.05, hedge_ratio=1.0, half_life=20)
        sa, sb = prices[sym_a], prices[sym_b]
        pair.spread = sa - sb

    result = backtest_pair(prices, pair, entry_z=entry_z, exit_z=exit_z)
    equity = result.equity_curve.reset_index()
    equity.columns = ["date", "equity"]
    equity["date"] = equity["date"].astype(str)

    return {
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "total_return": result.total_return,
        "win_rate": result.win_rate,
        "num_trades": result.num_trades,
        "calmar": result.calmar,
        "equity_curve": equity.to_dict(orient="records")
    }


@app.get("/api/signals/{sym_a}/{sym_b}")
def get_signals(sym_a: str, sym_b: str):
    prices, pairs = get_data()
    pair = next((p for p in pairs
                 if p.sym_a == sym_a and p.sym_b == sym_b), None)
    if not pair:
        return {"error": "Pair not found"}

    sig_df = generate_signals(pair)
    tail = sig_df.tail(60).reset_index()
    tail.columns = ["date", "zscore", "signal", "spread"]
    tail["date"] = tail["date"].astype(str)
    return tail.to_dict(orient="records")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "QuantAlpha"}
