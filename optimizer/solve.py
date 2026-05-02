"""
optimizer/solve.py
Owned by Person B.

Minimizes portfolio variance subject to:
  - Portfolio annualized vol <= target_vol  (passed directly — from UI slider)
  - Weights sum to 1
  - No short positions (w >= min_weight)
  - min_weight = DIVERSITY_FACTOR / n_stocks  (scales with portfolio size)
  - No single position > max_position cap

Public API
----------
optimize(cov_df, current_weights, target_vol, max_position, diversity_factor)
    High-level call. Returns the contracts/optimized_weights.csv schema.

_solve_at_target_vol(cov, target_vol, max_cap, diversity_factor)
    Low-level solve used internally by frontier.py.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import cvxpy as cp

from data.buckets import DIVERSITY_FACTOR as DEFAULT_DIVERSITY_FACTOR


# ---------------------------------------------------------------------------
# Low-level solver (used by both optimize() and frontier.py)
# ---------------------------------------------------------------------------

def _solve_at_target_vol(
    cov: np.ndarray,
    target_vol: float,
    max_cap: float,
    diversity_factor: float = DEFAULT_DIVERSITY_FACTOR,
) -> np.ndarray | None:
    """
    Solve the minimum-variance portfolio for a given vol constraint.

    Parameters
    ----------
    cov : np.ndarray, shape (n, n)
        Annualized covariance matrix. Must be positive semi-definite.
    target_vol : float
        Upper bound on annualized portfolio volatility (e.g., 0.25 for 25%).
    max_cap : float
        Maximum weight for any single ticker (e.g., 0.40).
    diversity_factor : float
        Controls minimum weight floor: min_weight = diversity_factor / n.
        With 5 stocks and factor=0.2: each position >= 4%.
        With 10 stocks and factor=0.2: each position >= 2%.
        Pass 0.0 to disable the floor (used in frontier sweeps).

    Returns
    -------
    np.ndarray of shape (n,) with optimal weights, or None if infeasible.
    Weights sum to 1. Dust below 0.1% is zeroed and renormalized.
    """
    n = cov.shape[0]
    min_weight = diversity_factor / n  # scales with portfolio size

    # Guard: if min_weight floor would over-constrain (n * min_weight > 1),
    # fall back to equal weight floor instead of crashing.
    if n * min_weight > 1.0:
        min_weight = 1.0 / n

    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, cov))
    constraints = [
        cp.sum(w) == 1,                           # fully invested
        w >= min_weight,                          # diversity floor (scales with n)
        w <= max_cap,                             # max position cap
        cp.quad_form(w, cov) <= target_vol ** 2, # vol constraint
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, warm_start=True)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None

    weights = w.value.copy()
    weights = np.where(weights < 0.001, 0.0, weights)  # zero out dust < 0.1%
    total = weights.sum()
    if total < 1e-8:
        return None
    weights /= total  # renormalize to exactly 1

    return weights


# ---------------------------------------------------------------------------
# High-level public function
# ---------------------------------------------------------------------------

def optimize(
    cov_df: pd.DataFrame,
    current_weights: pd.Series,
    target_vol: float,
    max_position: float = 0.40,
    diversity_factor: float = DEFAULT_DIVERSITY_FACTOR,
) -> pd.DataFrame:
    """
    Optimize portfolio weights for minimum variance at a given target volatility.

    Parameters
    ----------
    cov_df : pd.DataFrame
        Annualized covariance matrix. Index and columns are ticker strings.
    current_weights : pd.Series
        Current portfolio weights (by market value). Sums to ~1.0.
    target_vol : float
        User's chosen risk level — the vol ceiling for the optimizer.
        Comes directly from the UI slider (e.g. 0.25 for 25% annualized vol).
    max_position : float
        Maximum weight for any single ticker. Get this from the bucket's
        max_position field: get_bucket_params(bucket)['max_position'].
        Defaults to 0.40.
    diversity_factor : float
        Controls the minimum weight floor per position.
        min_weight = diversity_factor / n_stocks.
        Defaults to DIVERSITY_FACTOR from buckets.py (0.2).

    Returns
    -------
    pd.DataFrame matching the contracts/optimized_weights.csv schema:
        ticker            str     Ticker symbol
        current_weight    float   Current portfolio weight (rounded to 4dp)
        optimized_weight  float   Optimizer-recommended weight (rounded to 4dp)

    Raises
    ------
    ValueError if optimization is infeasible. This usually means target_vol
    is below the minimum-variance portfolio vol for these assets — tell the
    user to increase their risk tolerance on the slider.
    """
    tickers = cov_df.index.tolist()

    _validate_inputs(cov_df, current_weights, tickers)

    cov = cov_df.loc[tickers, tickers].values.astype(float)
    w_curr = current_weights[tickers].values.astype(float)

    weights = _solve_at_target_vol(
        cov,
        target_vol=target_vol,
        max_cap=max_position,
        diversity_factor=diversity_factor,
    )

    if weights is None:
        raise ValueError(
            f"Optimization infeasible at target_vol={target_vol:.1%}. "
            "The minimum-variance portfolio for these assets already exceeds "
            "this target. Try increasing the risk slider."
        )

    return pd.DataFrame({
        "ticker":            tickers,
        "current_weight":    np.round(w_curr, 4),
        "optimized_weight":  np.round(weights, 4),
    })


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    cov_df: pd.DataFrame,
    current_weights: pd.Series,
    tickers: list[str],
) -> None:
    """Raise informative errors on bad inputs before hitting the solver."""
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
