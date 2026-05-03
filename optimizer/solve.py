"""
optimizer/solve.py
Owned by Person B.

Minimizes portfolio variance subject to:
  - Portfolio annualized vol <= target_vol  (passed directly — from UI slider)
  - Weights sum to 1
  - No short positions (w >= min_weight)
  - min_weight = DIVERSITY_FACTOR / n_stocks  (scales with portfolio size)
  - No single position > max_position cap (dynamic — from buckets.dynamic_max_position)

L2 Regularization
-----------------
The optimizer objective includes an L2 penalty on weights:
    maximize  mu @ w  -  L2_REG * sum_squares(w)
This softly penalizes concentration — putting 40% in one stock incurs a larger
penalty than spreading to 20% each — without needing a hard re-run. The penalty
is continuous so it doesn't create discontinuities in the frontier curve.
L2_REG = 0.05 is a mild nudge. Increase toward 0.15 for stronger diversification.

Black-Litterman
---------------
BL expected returns are computed ONCE in app.py (_run_pipeline) via
black_litterman_mu() and passed in as exp_ret. This ensures the optimizer,
frontier, and return metrics all use the same mu vector.

Public API
----------
black_litterman_mu(cov, mu_hist, market_cap_weights, risk_aversion, blend)
    Computes BL posterior expected returns (no views).
    Call ONCE in app.py; pass result as exp_ret everywhere downstream.

compute_min_vol(cov, max_cap, diversity_factor)
    Returns annualized vol of the minimum-variance portfolio.
    Use to set the slider floor in the UI.

optimize(cov_df, current_weights, target_vol, max_position, diversity_factor,
         exp_ret, min_weights)
    High-level call. Expects BL-adjusted exp_ret from app.py.
    Returns the contracts/optimized_weights.csv schema.

_solve_at_target_vol(cov, target_vol, max_cap, diversity_factor, mu, min_weights)
    Low-level solve used internally by frontier.py.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
import numpy as np
import pandas as pd
import cvxpy as cp

from data.buckets import DIVERSITY_FACTOR as DEFAULT_DIVERSITY_FACTOR


# ---------------------------------------------------------------------------
# Black-Litterman constants
# ---------------------------------------------------------------------------

# 0.0 = pure market-anchored, 1.0 = pure historical.
# 0.10 = trust history 10%, trust market equilibrium 90%.
BL_HISTORICAL_BLEND = 0.10

BL_RISK_AVERSION = 2.5

# ---------------------------------------------------------------------------
# L2 regularization strength
# ---------------------------------------------------------------------------
# Penalty added to the optimizer objective: -L2_REG * sum_squares(w)
# This is minimized when weights are equal (1/n), so it continuously pushes
# against concentration without needing a hard cap to do all the work.
#
# Effect on a 10-stock portfolio (equal weight = 10%):
#   w=40% in one stock → penalty contribution: 0.05 * 0.16 = 0.008
#   w=20% in one stock → penalty contribution: 0.05 * 0.04 = 0.002
# The optimizer gives up 0.006 of expected return to avoid concentrating.
#
# Tune: 0.03 = mild nudge, 0.05 = moderate, 0.10 = strong diversification.
L2_REG: float = 0.1


# ---------------------------------------------------------------------------
# Black-Litterman — call from app.py, NOT from optimize()
# ---------------------------------------------------------------------------

def black_litterman_mu(
    cov: np.ndarray,
    mu_hist: np.ndarray,
    market_cap_weights: np.ndarray | None = None,
    risk_aversion: float = BL_RISK_AVERSION,
    blend: float = BL_HISTORICAL_BLEND,
) -> np.ndarray:
    """
    Compute Black-Litterman posterior expected returns (no views).

    Call ONCE in app.py and pass the result as exp_ret to both
    optimize() and generate_frontier() so every component uses the same mu.

    Parameters
    ----------
    cov : np.ndarray, shape (n, n)
    mu_hist : np.ndarray, shape (n,)
        Winsorized, James-Stein shrunk historical returns from app.py.
    market_cap_weights : np.ndarray | None
        Falls back to equal weights if None.
    risk_aversion : float
        Lambda. 2.5 is standard.
    blend : float
        0 = pure equilibrium, 1 = pure historical.

    Returns
    -------
    np.ndarray of shape (n,) — BL-blended expected returns.

    Notes
    -----
    pi    = lambda * Sigma * w_mkt     (reverse-optimized equilibrium returns)
    mu_BL = (1 - blend) * pi  +  blend * mu_hist
    """
    n = cov.shape[0]

    if market_cap_weights is None:
        w_mkt = np.ones(n) / n
    else:
        w_mkt = np.array(market_cap_weights, dtype=float)
        w_mkt = np.clip(w_mkt, 0, None)
        total = w_mkt.sum()
        w_mkt = np.ones(n) / n if total < 1e-8 else w_mkt / total

    pi = risk_aversion * cov @ w_mkt
    return (1.0 - blend) * pi + blend * mu_hist


# ---------------------------------------------------------------------------
# Min-vol solver
# ---------------------------------------------------------------------------

def compute_min_vol(
    cov: np.ndarray,
    max_cap: float,
    diversity_factor: float = DEFAULT_DIVERSITY_FACTOR,
) -> float:
    """
    Find the annualized vol of the minimum-variance portfolio.
    Use in app.py to set the slider floor.

    Returns 0.0 if solve fails.
    """
    n = cov.shape[0]
    min_weight = diversity_factor / n
    if n * min_weight > 1.0:
        min_weight = 1.0 / n

    w = cp.Variable(n)
    cov_psd = cp.psd_wrap(cov)
    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_cap]
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cov_psd)), constraints)
    problem.solve(solver=cp.CLARABEL)

    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return 0.0

    return float(np.sqrt(max(float(w.value @ cov @ w.value), 0.0)))


# ---------------------------------------------------------------------------
# Low-level solver (used by both optimize() and frontier.py)
# ---------------------------------------------------------------------------

def _solve_at_target_vol(
    cov: np.ndarray,
    target_vol: float,
    max_cap: float,
    diversity_factor: float = DEFAULT_DIVERSITY_FACTOR,
    mu: np.ndarray | None = None,
    min_weights: np.ndarray | None = None,
    l2_reg: float = L2_REG,
) -> np.ndarray | None:
    """
    Solve for optimal portfolio weights at a given vol ceiling.

    Objective (when mu provided):
        maximize  mu @ w  -  l2_reg * sum_squares(w)

    The L2 term penalizes concentration continuously — putting 40% in one
    stock costs more penalty than 20% each — so the optimizer naturally
    diversifies without relying solely on hard caps.

    Parameters
    ----------
    cov : np.ndarray, shape (n, n)
    target_vol : float
    max_cap : float
    diversity_factor : float
    mu : np.ndarray | None — BL-adjusted expected returns
    min_weights : np.ndarray | None — per-ticker floor
    l2_reg : float — L2 penalty strength (default L2_REG module constant)

    Returns
    -------
    np.ndarray of shape (n,) or None if infeasible.
    """
    n = cov.shape[0]

    if min_weights is not None:
        lb = min_weights
    else:
        min_weight = diversity_factor / n
        if n * min_weight > 1.0:
            min_weight = 1.0 / n
        lb = np.full(n, min_weight)

    w = cp.Variable(n)
    cov_psd = cp.psd_wrap(cov)
    constraints = [
        cp.sum(w) == 1,
        w >= lb,
        w <= max_cap,
        cp.quad_form(w, cov_psd) <= target_vol ** 2,
    ]

    if mu is not None:
        # L2 regularization: penalize concentration, encourage diversification.
        # sum_squares(w) is minimized at equal weights (1/n each).
        objective = cp.Maximize(mu @ w - l2_reg * cp.sum_squares(w))
    else:
        objective = cp.Minimize(cp.quad_form(w, cov_psd))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, warm_start=True)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None

    weights = w.value.copy()
    weights = np.where(weights < 0.001, 0.0, weights)
    total = weights.sum()
    if total < 1e-8:
        return None
    weights /= total
    return weights


# ---------------------------------------------------------------------------
# High-level public function
# ---------------------------------------------------------------------------

def optimize(
    cov_df: pd.DataFrame,
    current_weights: pd.Series,
    target_vol: float,
    max_position: float = 0.25,
    diversity_factor: float = DEFAULT_DIVERSITY_FACTOR,
    exp_ret: pd.Series | None = None,
    min_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Optimize portfolio weights at a given target volatility.

    exp_ret should be BL-adjusted — call black_litterman_mu() in app.py
    BEFORE this function. max_position should come from
    dynamic_max_position(bucket, n_stocks) in app.py, not bucket base cap.

    Infeasibility is handled by clamping target_vol to the min-variance
    floor + 1% buffer, so the solver is always feasible.

    Parameters
    ----------
    cov_df : pd.DataFrame
    current_weights : pd.Series
    target_vol : float
    max_position : float
        Use dynamic_max_position(bucket, n_stocks) from buckets.py.
    diversity_factor : float
    exp_ret : pd.Series | None — BL-adjusted expected returns.
    min_weights : np.ndarray | None — per-ticker floors for new-stocks mode.

    Returns
    -------
    pd.DataFrame: ticker, current_weight, optimized_weight.
    """
    tickers = cov_df.index.tolist()
    _validate_inputs(cov_df, current_weights, tickers)

    cov = cov_df.loc[tickers, tickers].values.astype(float)
    w_curr = current_weights[tickers].values.astype(float)
    mu = exp_ret[tickers].values.astype(float) if exp_ret is not None else None

    min_vol = compute_min_vol(cov, max_cap=max_position, diversity_factor=diversity_factor)
    vol_floor = min_vol * 1.01
    effective_vol = max(target_vol, vol_floor)

    if effective_vol > target_vol:
        warnings.warn(
            f"target_vol={target_vol:.1%} is below the minimum achievable "
            f"vol={min_vol:.1%}. Clamped to {effective_vol:.1%}.",
            UserWarning,
            stacklevel=2,
        )

    weights = _solve_at_target_vol(
        cov,
        target_vol=effective_vol,
        max_cap=max_position,
        diversity_factor=diversity_factor,
        mu=mu,
        min_weights=min_weights,
    )

    if weights is None:
        raise ValueError(
            f"Optimization infeasible even after clamping to floor vol={min_vol:.1%}. "
            "Check the covariance matrix for near-singular rows."
        )

    return pd.DataFrame({
        "ticker":           tickers,
        "current_weight":   np.round(w_curr, 4),
        "optimized_weight": np.round(weights, 4),
    })


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    cov_df: pd.DataFrame,
    current_weights: pd.Series,
    tickers: list[str],
) -> None:
    for t in tickers:
        if t not in current_weights.index:
            raise KeyError(f"Ticker '{t}' missing from current_weights.")

    assert cov_df.shape[0] == cov_df.shape[1], "Covariance matrix must be square."

    if not np.allclose(cov_df.values, cov_df.values.T, atol=1e-6):
        raise ValueError("Covariance matrix is not symmetric.")

    eigvals = np.linalg.eigvalsh(cov_df.values)
    if eigvals.min() < -1e-6:
        raise ValueError(
            f"Covariance matrix is not positive semi-definite "
            f"(min eigenvalue = {eigvals.min():.6f}). "
            "Check Person A's covariance output."
        )

    w_sum = current_weights[tickers].sum()
    if not np.isclose(w_sum, 1.0, atol=0.01):
        raise ValueError(
            f"current_weights sum to {w_sum:.4f}, expected ~1.0."
        )