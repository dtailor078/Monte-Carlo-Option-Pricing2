"""
Monte Carlo Option Pricing with Variance Reduction
Supports:
- European Call
- Asian Arithmetic Average Call
- Barrier Up-and-Out Call
With:
- Antithetic Variates
- Control Variates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp
from scipy.stats import norm

# ================================
# Parameters (you can edit these)
# ================================
S0 = 100.0       # initial stock price
K = 100.0        # strike
r = 0.05         # risk-free rate
sigma = 0.2      # volatility
T = 1.0          # maturity in years
n_paths = 50_000 # number of Monte Carlo paths
n_steps = 252    # time steps per path
barrier = 120.0  # barrier for up-and-out option
use_antithetic = True
use_control = True
seed = 42

np.random.seed(seed)


# ================================
# Black-Scholes analytic price
# ================================
def bs_call_price(S, K, r, sigma, T):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


# ================================
# Path simulation
# ================================
def simulate_paths(S0, r, sigma, T, n_steps, n_paths, antithetic=False):
    dt = T / n_steps
    Z = np.random.normal(size=(n_paths, n_steps))
    if antithetic:
        Z = np.vstack([Z, -Z])
    increments = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((Z.shape[0],1)), log_paths])
    ST_paths = S0 * np.exp(log_paths)
    return ST_paths


# ================================
# Monte Carlo Pricing
# ================================
def mc_pricing(S0, K, r, sigma, T, n_paths, n_steps, barrier,
               use_antithetic=False, use_control=False):

    # European
    Z = np.random.normal(size=n_paths)
    if use_antithetic:
        Z = np.concatenate([Z, -Z])
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    euro_payoffs = np.maximum(ST - K, 0.0) * np.exp(-r*T)

    # Asian
    paths = simulate_paths(S0, r, sigma, T, n_steps, n_paths, antithetic=use_antithetic)
    avg_price = paths.mean(axis=1)
    asian_payoffs = np.maximum(avg_price - K, 0.0) * np.exp(-r*T)

    # Barrier
    knocked = (paths.max(axis=1) >= barrier)
    barrier_payoffs = np.where(knocked, 0.0, np.maximum(paths[:,-1] - K, 0.0)) * np.exp(-r*T)

    # Control variates
    if use_control:
        discounted_ST = ST * np.exp(-r*T)
        var_X = discounted_ST.var()
        def apply_cv(Y):
            cov_YX = np.cov(Y, discounted_ST[:len(Y)])[0,1]
            b_opt = cov_YX / var_X
            return Y - b_opt*(discounted_ST[:len(Y)] - S0)
        euro_payoffs = apply_cv(euro_payoffs)
        asian_payoffs = apply_cv(asian_payoffs)
        barrier_payoffs = apply_cv(barrier_payoffs)

    def summarize(payoffs, name):
        price = payoffs.mean()
        se = payoffs.std(ddof=1)/np.sqrt(len(payoffs))
        return {
            'Option': name,
            'Price': price,
            'StdErr': se,
            '95% CI lower': price - 1.96*se,
            '95% CI upper': price + 1.96*se
        }

    results = [
        summarize(euro_payoffs, 'European Call (MC)'),
        summarize(asian_payoffs, 'Asian Arithmetic Call (MC)'),
        summarize(barrier_payoffs, 'Barrier Up-and-Out Call (MC)'),
        {
            'Option': 'European Call (BS)',
            'Price': bs_call_price(S0,K,r,sigma,T),
            'StdErr': 0.0,
            '95% CI lower': bs_call_price(S0,K,r,sigma,T),
            '95% CI upper': bs_call_price(S0,K,r,sigma,T)
        }
    ]
    return pd.DataFrame(results)


# ================================
# Run simulation
# ================================
results = mc_pricing(S0, K, r, sigma, T, n_paths, n_steps, barrier,
                     use_antithetic, use_control)

print("\n===== Results =====")
print(results)

results.plot(x='Option', y='Price', kind='bar', legend=False,
             title='Option Prices', figsize=(8,4))
plt.show()

