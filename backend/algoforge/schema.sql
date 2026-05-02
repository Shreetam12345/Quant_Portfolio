-- ============================================================
-- AlgoForge: Options Pricing & Risk Platform
-- SQL Schema — Trade Logs, Greeks, Risk Metrics
-- ============================================================

-- Underlying instruments
CREATE TABLE IF NOT EXISTS underlyings (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10) UNIQUE NOT NULL,
    name        VARCHAR(80),
    asset_class VARCHAR(20) DEFAULT 'equity'
);

-- Option contract definitions
CREATE TABLE IF NOT EXISTS option_contracts (
    id            SERIAL PRIMARY KEY,
    underlying_id INT REFERENCES underlyings(id),
    option_type   CHAR(1) CHECK (option_type IN ('C','P')),  -- C=Call, P=Put
    strike        NUMERIC(12,2) NOT NULL,
    expiry        DATE NOT NULL,
    style         CHAR(1) DEFAULT 'E',                        -- E=European, A=American
    UNIQUE (underlying_id, option_type, strike, expiry)
);

-- Market snapshots (spot price, IV, risk-free rate)
CREATE TABLE IF NOT EXISTS market_snapshots (
    id              SERIAL PRIMARY KEY,
    underlying_id   INT REFERENCES underlyings(id),
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    spot_price      NUMERIC(12,4) NOT NULL,
    implied_vol     NUMERIC(8,6),
    risk_free_rate  NUMERIC(6,4) DEFAULT 0.05,
    dividend_yield  NUMERIC(6,4) DEFAULT 0.0
);

-- Computed option prices and Greeks
CREATE TABLE IF NOT EXISTS option_greeks (
    id           SERIAL PRIMARY KEY,
    contract_id  INT REFERENCES option_contracts(id),
    snapshot_id  INT REFERENCES market_snapshots(id),
    model        VARCHAR(20) DEFAULT 'black_scholes',
    theo_price   NUMERIC(12,6),
    delta        NUMERIC(10,6),
    gamma        NUMERIC(10,6),
    theta        NUMERIC(10,6),
    vega         NUMERIC(10,6),
    rho          NUMERIC(10,6),
    computed_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Portfolio positions
CREATE TABLE IF NOT EXISTS positions (
    id           SERIAL PRIMARY KEY,
    portfolio_id VARCHAR(30) NOT NULL,
    contract_id  INT REFERENCES option_contracts(id),
    quantity     INT NOT NULL,              -- +ve = long, -ve = short
    avg_cost     NUMERIC(12,4),
    opened_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Portfolio aggregate risk (updated after each reprice)
CREATE TABLE IF NOT EXISTS portfolio_risk (
    id            SERIAL PRIMARY KEY,
    portfolio_id  VARCHAR(30) NOT NULL,
    ts            TIMESTAMPTZ DEFAULT NOW(),
    net_delta     NUMERIC(14,4),
    net_gamma     NUMERIC(14,4),
    net_theta     NUMERIC(14,4),
    net_vega      NUMERIC(14,4),
    portfolio_pnl NUMERIC(14,2),
    var_95        NUMERIC(14,2),
    params        JSONB
);

-- Monte Carlo simulation runs
CREATE TABLE IF NOT EXISTS mc_runs (
    id              SERIAL PRIMARY KEY,
    contract_id     INT REFERENCES option_contracts(id),
    snapshot_id     INT REFERENCES market_snapshots(id),
    num_paths       INT,
    num_steps       INT,
    mc_price        NUMERIC(12,6),
    mc_std_error    NUMERIC(12,6),
    ran_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- VIEWS
-- ============================================================

-- Full Greeks view with contract details
CREATE OR REPLACE VIEW v_greeks_latest AS
SELECT
    u.symbol,
    CONCAT(u.symbol, ' ', oc.strike, oc.option_type, ' ', oc.expiry) AS contract_label,
    oc.option_type,
    oc.strike,
    oc.expiry,
    oc.expiry - CURRENT_DATE AS days_to_expiry,
    og.theo_price,
    og.delta,
    og.gamma,
    og.theta,
    og.vega,
    og.rho,
    og.model,
    og.computed_at
FROM option_greeks og
JOIN option_contracts oc ON oc.id = og.contract_id
JOIN underlyings u ON u.id = oc.underlying_id
WHERE og.id IN (
    SELECT MAX(id) FROM option_greeks GROUP BY contract_id
);

-- IV surface summary
CREATE OR REPLACE VIEW v_iv_surface AS
SELECT
    u.symbol,
    oc.expiry,
    oc.strike,
    oc.option_type,
    ms.implied_vol,
    ms.spot_price,
    (oc.strike / ms.spot_price) AS moneyness,
    ms.ts
FROM option_contracts oc
JOIN underlyings u ON u.id = oc.underlying_id
JOIN market_snapshots ms ON ms.underlying_id = oc.underlying_id
WHERE ms.ts = (
    SELECT MAX(ts) FROM market_snapshots ms2
    WHERE ms2.underlying_id = oc.underlying_id
);

-- Portfolio net Greeks aggregation
CREATE OR REPLACE VIEW v_portfolio_greeks AS
SELECT
    p.portfolio_id,
    SUM(p.quantity * og.delta)  AS net_delta,
    SUM(p.quantity * og.gamma)  AS net_gamma,
    SUM(p.quantity * og.theta)  AS net_theta,
    SUM(p.quantity * og.vega)   AS net_vega,
    SUM(p.quantity * og.theo_price) AS portfolio_value
FROM positions p
JOIN option_greeks og ON og.contract_id = p.contract_id
WHERE og.id IN (
    SELECT MAX(id) FROM option_greeks GROUP BY contract_id
)
GROUP BY p.portfolio_id;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_greeks_contract ON option_greeks(contract_id, computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON market_snapshots(underlying_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_positions_portfolio ON positions(portfolio_id);
