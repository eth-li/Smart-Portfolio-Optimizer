"""
optimizer/solve.py
Owned by Person B.

Maximizes expected return subject to:
  - Portfolio annualized vol <= target_vol
  - Weights sum to 1
  - No short positions (w >= 0)
  - No single position > max_position cap

Public API
----------
optimize(cov_df, expected_returns, current_weights, risk_bucket) -> pd.DataFrame
    High-level call. Reads bucket params and returns the contracts/optimized_weights.csv schema.

_solve_at_target_vol(cov, mu, target_vol, max_cap) -> np.ndarray | None
    Low-level solve used internally by frontier.py. Returns weight array or None if infeasible.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import cvxpy as cp

from data.buckets import get_bucket_params


# ---------------------------------------------------------------------------
# Low-level solver (used by both optimize() and frontier.py)
# ---------------------------------------------------------------------------

def _solve_at_target_vol(
    cov: np.ndarray,
    mu: np.ndarray,
    target_vol: float,
    max_cap: float,
) -> np.ndarray | None:
    """
    Solve the max-return portfolio for a given vol constraint.

    Parameters
    ----------
    cov : np.ndarray, shape (n, n)
        Annualized covariance matrix. Must be positive semi-definite.
    mu : np.ndarray, shape (n,)
        Annualized expected returns.
    target_vol : float
        Upper bound on annualized portfolio volatility (e.g., 0.15 for 15%).
    max_cap : float
        Maximum weight for any single ticker (e.g., 0.40).

    Returns
    -------
    np.ndarray of shape (n,) with optimal weights, or None if infeasible.
    Weights are non-negative, sum to 1, and have dust (< 0.5%) zeroed out.
    """
    n = len(mu)
    w = cp.Variable(n)

    objective = cp.Maximize(mu @ w)
    constraints = [
        cp.sum(w) == 1,                           # fully invested
        w >= 0.01,                                # minimum 1% per position — no stock goes to zero
        w <= max_cap,                             # max position cap
        cp.quad_form(w, cov) <= target_vol ** 2, # vol constraint
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, warm_start=True)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None  # caller decides how to handle

    weights = w.value.copy()
    weights = np.clip(weights, 0.01, None)  # enforce floor after solve
    weights /= weights.sum()               # renormalize to exactly 1

    return weights


# ---------------------------------------------------------------------------
# High-level public function
# ---------------------------------------------------------------------------

def optimize(
    cov_df: pd.DataFrame,
    expected_returns: pd.Series,
    current_weights: pd.Series,
    risk_bucket: str,
) -> pd.DataFrame:
    """
    Optimize portfolio weights for a given risk bucket.

    Parameters
    ----------
    cov_df : pd.DataFrame
        Annualized covariance matrix. Index and columns must be ticker strings.
        Shape: (n, n). Must match ticker order in expected_returns and current_weights.
    expected_returns : pd.Series
        Annualized expected returns. Index is ticker strings.
    current_weights : pd.Series
        Current portfolio weights (by market value). Index is ticker strings.
        Must sum to ~1.0.
    risk_bucket : str
        One of 'Conservative', 'Moderate', 'Aggressive'.

    Returns
    -------
    pd.DataFrame matching the contracts/optimized_weights.csv schema:
        ticker            str     Ticker symbol
        current_weight    float   Current portfolio weight
        optimized_weight  float   Optimizer-recommended weight

    Raises
    ------
    ValueError if the optimization is infeasible (target vol is below the
    minimum-variance portfolio vol for these assets and constraints).
    """
    params = get_bucket_params(risk_bucket)
    tickers = cov_df.index.tolist()

    # Validate inputs
    _validate_inputs(cov_df, expected_returns, current_weights, tickers)

    cov = cov_df.loc[tickers, tickers].values.astype(float)
    mu = expected_returns[tickers].values.astype(float)
    w_curr = current_weights[tickers].values.astype(float)

    weights = _solve_at_target_vol(
        cov,
        mu,
        target_vol=params["target_vol"],
        max_cap=params["max_position"],
    )

    if weights is None:
        raise ValueError(
            f"Optimization infeasible for bucket '{risk_bucket}' "
            f"(target_vol={params['target_vol']:.0%}). "
            "This usually means the minimum-variance portfolio for these "
            "assets already exceeds the target. Consider using the "
            "'Aggressive' bucket or check the covariance matrix."
        )

    return pd.DataFrame({
        "ticker": tickers,
        "current_weight": np.round(w_curr, 6),
        "optimized_weight": np.round(weights, 6),
    })


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    cov_df: pd.DataFrame,
    expected_returns: pd.Series,
    current_weights: pd.Series,
    tickers: list[str],
) -> None:
    """Raise informative errors on bad inputs before hitting the solver."""
    # All tickers present
    for t in tickers:
        if t not in expected_returns.index:
            raise KeyError(f"Ticker '{t}' missing from expected_returns.")
        if t not in current_weights.index:
            raise KeyError(f"Ticker '{t}' missing from current_weights.")

    # Covariance matrix is square and symmetric
    assert cov_df.shape[0] == cov_df.shape[1], "Covariance matrix must be square."
    if not np.allclose(cov_df.values, cov_df.values.T, atol=1e-6):
        raise ValueError("Covariance matrix is not symmetric.")

    # Covariance matrix is positive semi-definite
    eigvals = np.linalg.eigvalsh(cov_df.values)
    if eigvals.min() < -1e-6:
        raise ValueError(
            f"Covariance matrix is not positive semi-definite "
            f"(min eigenvalue = {eigvals.min():.6f}). "
            "Check Person A's covariance output."
        )

    # Current weights sum to ~1
    w_sum = current_weights[tickers].sum()
    if not np.isclose(w_sum, 1.0, atol=0.01):
        raise ValueError(
            f"current_weights sum to {w_sum:.4f}, expected ~1.0."
        )
