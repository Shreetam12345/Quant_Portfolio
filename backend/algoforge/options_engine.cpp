// ============================================================
// AlgoForge — C++ Options Pricing Engine
// High-speed Black-Scholes, Greeks, and Monte Carlo
// Exposed via pybind11 to Python FastAPI
// ============================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

// ── Standard Normal CDF & PDF ─────────────────────────────────────────────────

static inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

static inline double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

// ── d1 / d2 ──────────────────────────────────────────────────────────────────

static void compute_d1d2(
    double S, double K, double T, double r, double sigma, double q,
    double& d1, double& d2
) {
    d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
         (sigma * std::sqrt(T));
    d2 = d1 - sigma * std::sqrt(T);
}

// ── Black-Scholes Price ───────────────────────────────────────────────────────

double bs_price(
    double S, double K, double T, double r,
    double sigma, double q, char option_type
) {
    if (T <= 0.0) {
        double intrinsic = (option_type == 'C') ?
            std::max(S - K, 0.0) : std::max(K - S, 0.0);
        return intrinsic;
    }

    double d1, d2;
    compute_d1d2(S, K, T, r, sigma, q, d1, d2);

    double disc_S = S * std::exp(-q * T);
    double disc_K = K * std::exp(-r * T);

    if (option_type == 'C')
        return disc_S * norm_cdf(d1) - disc_K * norm_cdf(d2);
    else
        return disc_K * norm_cdf(-d2) - disc_S * norm_cdf(-d1);
}

// ── Greeks ────────────────────────────────────────────────────────────────────

struct GreeksResult {
    double price, delta, gamma, theta, vega, rho;
};

GreeksResult bs_greeks(
    double S, double K, double T, double r,
    double sigma, double q, char option_type
) {
    GreeksResult g{};
    g.price = bs_price(S, K, T, r, sigma, q, option_type);

    if (T < 1e-6) return g;

    double d1, d2;
    compute_d1d2(S, K, T, r, sigma, q, d1, d2);

    double sqrt_T  = std::sqrt(T);
    double nd1     = norm_pdf(d1);
    double disc_q  = std::exp(-q * T);
    double disc_r  = std::exp(-r * T);

    g.gamma = disc_q * nd1 / (S * sigma * sqrt_T);
    g.vega  = S * disc_q * nd1 * sqrt_T / 100.0;

    if (option_type == 'C') {
        g.delta = disc_q * norm_cdf(d1);
        g.theta = (-(S * disc_q * nd1 * sigma) / (2.0 * sqrt_T)
                   - r * K * disc_r * norm_cdf(d2)
                   + q * S * disc_q * norm_cdf(d1)) / 365.0;
        g.rho   = K * T * disc_r * norm_cdf(d2) / 100.0;
    } else {
        g.delta = disc_q * (norm_cdf(d1) - 1.0);
        g.theta = (-(S * disc_q * nd1 * sigma) / (2.0 * sqrt_T)
                   + r * K * disc_r * norm_cdf(-d2)
                   - q * S * disc_q * norm_cdf(-d1)) / 365.0;
        g.rho   = -K * T * disc_r * norm_cdf(-d2) / 100.0;
    }

    return g;
}

// ── Binomial Tree (American options) ─────────────────────────────────────────

double binomial_american(
    double S, double K, double T, double r,
    double sigma, double q, char option_type,
    int steps = 200
) {
    double dt   = T / steps;
    double u    = std::exp(sigma * std::sqrt(dt));
    double d    = 1.0 / u;
    double disc = std::exp(-r * dt);
    double pu   = (std::exp((r - q) * dt) - d) / (u - d);
    double pd   = 1.0 - pu;

    // Terminal node prices
    std::vector<double> prices(steps + 1);
    std::vector<double> values(steps + 1);

    for (int i = 0; i <= steps; ++i)
        prices[i] = S * std::pow(u, steps - 2 * i);

    for (int i = 0; i <= steps; ++i) {
        if (option_type == 'C')
            values[i] = std::max(prices[i] - K, 0.0);
        else
            values[i] = std::max(K - prices[i], 0.0);
    }

    // Backward induction with early exercise
    for (int step = steps - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            double node_price = S * std::pow(u, step - 2 * i);
            double hold_val   = disc * (pu * values[i] + pd * values[i + 1]);
            double exer_val   = (option_type == 'C') ?
                                std::max(node_price - K, 0.0) :
                                std::max(K - node_price, 0.0);
            values[i] = std::max(hold_val, exer_val);
        }
    }

    return values[0];
}

// ── Monte Carlo with Antithetic Variates ──────────────────────────────────────

struct MCResult {
    double price, std_error, ci_lower, ci_upper;
    int num_paths;
};

MCResult monte_carlo_price(
    double S, double K, double T, double r,
    double sigma, double q, char option_type,
    int num_paths = 50000, int num_steps = 252, int seed = 42
) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double dt       = T / num_steps;
    double drift    = (r - q - 0.5 * sigma * sigma) * dt;
    double diffusion = sigma * std::sqrt(dt);

    int half = num_paths / 2;
    std::vector<double> payoffs;
    payoffs.reserve(num_paths);

    for (int i = 0; i < half; ++i) {
        double log_S1 = std::log(S);
        double log_S2 = std::log(S);
        for (int j = 0; j < num_steps; ++j) {
            double z   = normal(rng);
            log_S1    += drift + diffusion * z;
            log_S2    += drift - diffusion * z;   // antithetic
        }
        double ST1 = std::exp(log_S1);
        double ST2 = std::exp(log_S2);

        double pay1 = (option_type == 'C') ? std::max(ST1 - K, 0.0) : std::max(K - ST1, 0.0);
        double pay2 = (option_type == 'C') ? std::max(ST2 - K, 0.0) : std::max(K - ST2, 0.0);
        double disc = std::exp(-r * T);
        payoffs.push_back(disc * pay1);
        payoffs.push_back(disc * pay2);
    }

    double mean = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / payoffs.size();
    double sq_sum = 0;
    for (double v : payoffs) sq_sum += (v - mean) * (v - mean);
    double std_dev = std::sqrt(sq_sum / (payoffs.size() - 1));
    double std_err = std_dev / std::sqrt(payoffs.size());

    return {mean, std_err, mean - 1.96 * std_err, mean + 1.96 * std_err,
            static_cast<int>(payoffs.size())};
}

// ── Pybind11 Module ───────────────────────────────────────────────────────────

PYBIND11_MODULE(options_engine, m) {
    m.doc() = "AlgoForge C++ Options Pricing Engine";

    m.def("bs_price", &bs_price,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("sigma"), py::arg("q") = 0.0,
          py::arg("option_type") = 'C',
          "Black-Scholes option price");

    py::class_<GreeksResult>(m, "GreeksResult")
        .def_readonly("price", &GreeksResult::price)
        .def_readonly("delta", &GreeksResult::delta)
        .def_readonly("gamma", &GreeksResult::gamma)
        .def_readonly("theta", &GreeksResult::theta)
        .def_readonly("vega",  &GreeksResult::vega)
        .def_readonly("rho",   &GreeksResult::rho);

    m.def("bs_greeks", &bs_greeks,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("sigma"), py::arg("q") = 0.0,
          py::arg("option_type") = 'C',
          "Compute all Black-Scholes Greeks");

    m.def("binomial_american", &binomial_american,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("sigma"), py::arg("q") = 0.0,
          py::arg("option_type") = 'C', py::arg("steps") = 200,
          "Price American option via binomial tree (CRR)");

    py::class_<MCResult>(m, "MCResult")
        .def_readonly("price",     &MCResult::price)
        .def_readonly("std_error", &MCResult::std_error)
        .def_readonly("ci_lower",  &MCResult::ci_lower)
        .def_readonly("ci_upper",  &MCResult::ci_upper)
        .def_readonly("num_paths", &MCResult::num_paths);

    m.def("monte_carlo_price", &monte_carlo_price,
          py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("sigma"), py::arg("q") = 0.0,
          py::arg("option_type") = 'C',
          py::arg("num_paths") = 50000,
          py::arg("num_steps") = 252,
          py::arg("seed") = 42,
          "Price European option via Monte Carlo with antithetic variates");
}
