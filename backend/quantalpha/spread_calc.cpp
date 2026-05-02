// ============================================================
// QuantAlpha — C++ Spread & Risk Calculator
// High-performance computations exposed via pybind11
// Compile: g++ -O2 -shared -fPIC $(python3-config --includes) 
//          spread_calc.cpp -o spread_calc.so -lpybind11
// ============================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

// ── Rolling Z-Score ────────────────────────────────────────────────────────────

/**
 * Compute rolling z-score on a price spread.
 * Far faster than pandas rolling for large datasets.
 */
std::vector<double> rolling_zscore(
    const std::vector<double>& spread,
    int window
) {
    int n = static_cast<int>(spread.size());
    std::vector<double> result(n, std::numeric_limits<double>::quiet_NaN());

    for (int i = window - 1; i < n; ++i) {
        double sum = 0.0, sq_sum = 0.0;
        for (int j = i - window + 1; j <= i; ++j) {
            sum    += spread[j];
            sq_sum += spread[j] * spread[j];
        }
        double mean = sum / window;
        double var  = sq_sum / window - mean * mean;
        double std  = (var > 1e-12) ? std::sqrt(var) : 1e-12;
        result[i]   = (spread[i] - mean) / std;
    }
    return result;
}

// ── Ornstein-Uhlenbeck Half-Life ───────────────────────────────────────────────

/**
 * Estimate OU process half-life via OLS on first differences.
 * Returns NaN if the process is not mean-reverting.
 */
double ou_halflife(const std::vector<double>& spread) {
    int n = static_cast<int>(spread.size());
    if (n < 3) return std::numeric_limits<double>::quiet_NaN();

    // delta_t = spread[t] - spread[t-1]
    // Regress delta_t on spread[t-1]: delta_t = lambda * spread[t-1] + epsilon
    double sx = 0, sy = 0, sxy = 0, sxx = 0;
    int m = n - 1;
    for (int i = 0; i < m; ++i) {
        double x = spread[i];          // spread[t-1]
        double y = spread[i+1] - x;    // delta_t
        sx  += x;
        sy  += y;
        sxy += x * y;
        sxx += x * x;
    }
    double denom = (m * sxx - sx * sx);
    if (std::abs(denom) < 1e-12) return std::numeric_limits<double>::quiet_NaN();

    double lambda = (m * sxy - sx * sy) / denom;
    if (lambda >= 0) return std::numeric_limits<double>::quiet_NaN();
    return -std::log(2.0) / lambda;
}

// ── Value at Risk (Historical Simulation) ─────────────────────────────────────

/**
 * Compute portfolio VaR at given confidence level using historical simulation.
 * Returns (VaR_dollar, CVaR_dollar).
 */
std::pair<double, double> historical_var(
    const std::vector<double>& pnl_series,
    double confidence = 0.95
) {
    if (pnl_series.empty()) return {0.0, 0.0};

    std::vector<double> sorted = pnl_series;
    std::sort(sorted.begin(), sorted.end());

    int n = static_cast<int>(sorted.size());
    int idx = static_cast<int>((1.0 - confidence) * n);
    idx = std::max(0, std::min(idx, n - 1));

    double var = -sorted[idx];
    // CVaR = average of losses beyond VaR
    double cvar_sum = 0;
    int cvar_count = 0;
    for (int i = 0; i <= idx; ++i) {
        cvar_sum += sorted[i];
        ++cvar_count;
    }
    double cvar = (cvar_count > 0) ? -(cvar_sum / cvar_count) : var;
    return {var, cvar};
}

// ── Maximum Drawdown ───────────────────────────────────────────────────────────

double max_drawdown(const std::vector<double>& equity_curve) {
    if (equity_curve.empty()) return 0.0;
    double peak = equity_curve[0];
    double max_dd = 0.0;
    for (double v : equity_curve) {
        if (v > peak) peak = v;
        double dd = (peak > 0) ? (v - peak) / peak : 0.0;
        if (dd < max_dd) max_dd = dd;
    }
    return max_dd;
}

// ── Sharpe Ratio ───────────────────────────────────────────────────────────────

double sharpe_ratio(const std::vector<double>& returns, double ann_factor = 252.0) {
    if (returns.size() < 2) return 0.0;
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double sq_sum = 0.0;
    for (double r : returns) sq_sum += (r - mean) * (r - mean);
    double std_dev = std::sqrt(sq_sum / (returns.size() - 1));
    if (std_dev < 1e-12) return 0.0;
    return (mean / std_dev) * std::sqrt(ann_factor);
}

// ── Spread Construction ────────────────────────────────────────────────────────

std::vector<double> compute_spread(
    const std::vector<double>& price_a,
    const std::vector<double>& price_b,
    double hedge_ratio
) {
    if (price_a.size() != price_b.size())
        throw std::runtime_error("Price series must have equal length");

    std::vector<double> spread(price_a.size());
    for (size_t i = 0; i < price_a.size(); ++i)
        spread[i] = price_a[i] - hedge_ratio * price_b[i];
    return spread;
}

// ── Pybind11 Module Registration ──────────────────────────────────────────────

PYBIND11_MODULE(spread_calc, m) {
    m.doc() = "QuantAlpha C++ spread calculation & risk module";

    m.def("rolling_zscore", &rolling_zscore,
          py::arg("spread"), py::arg("window"),
          "Compute rolling z-score of spread series");

    m.def("ou_halflife", &ou_halflife,
          py::arg("spread"),
          "Estimate Ornstein-Uhlenbeck half-life of spread");

    m.def("historical_var", &historical_var,
          py::arg("pnl_series"), py::arg("confidence") = 0.95,
          "Compute (VaR, CVaR) via historical simulation");

    m.def("max_drawdown", &max_drawdown,
          py::arg("equity_curve"),
          "Compute maximum drawdown of an equity curve");

    m.def("sharpe_ratio", &sharpe_ratio,
          py::arg("returns"), py::arg("ann_factor") = 252.0,
          "Compute annualised Sharpe ratio");

    m.def("compute_spread", &compute_spread,
          py::arg("price_a"), py::arg("price_b"), py::arg("hedge_ratio"),
          "Construct dollar-neutral spread: A - hedge_ratio * B");
}
