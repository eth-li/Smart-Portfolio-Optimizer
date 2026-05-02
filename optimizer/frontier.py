"""
optimizer/frontier.py
Owned by Person B.

Generates the efficient frontier by sweeping target_vol from the minimum-variance
portfolio vol to the bucket's vol_max, solving the optimizer at each step.

Public API
----------
generate_frontier(cov_df, expected_returns, current_weights, risk_bucket, n_points=20)
    Returns pd.DataFrame matching the contracts/frontier.csv schema.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import cvxpy as cp

from data.buckets import get_bucket_params
from optimizer.solve import _solve_at_target_vol


# ---------------------------------------------------------------------------
# Minimum variance portfolio helper
# ---------------------------------------------------------------------------

def _min_variance_vol(cov: np.ndarray, max_cap: float) -> float:
    """
    Find the annualized vol of the global minimum-variance portfolio.
    This is the leftmost point of the efficient frontier.

    Parameters
    ----------
    cov : np.ndarray, shape (n, n)
        Annualized covariance matrix.
    max_cap : float
        Max single-position weight (same cap as the optimizer).

    Returns
    -------
    float: annualized vol of the min-variance portfolio.
    """
    n = cov.shape[0]
    w = cp.Variable(n)
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(w, cov)),
        [cp.sum(w) == 1, w >= 0, w <= max_cap],
    )
    problem.solve(solver=cp.CLARABEL)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError("Could not find minimum variance portfolio.")

    return float(np.sqrt(max(problem.value, 0.0)))


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_frontier(
    cov_df: pd.DataFrame,
    expected_returns: pd.Series,
    current_weights: pd.Series,
    risk_bucket: str,
    n_points: int = 20,
) -> pd.DataFrame:
    """
    Generate efficient frontier data points for a given risk bucket.

    Sweeps target_vol from just above the minimum-variance vol to the bucket's
    vol_max ceiling, solving the max-return problem at each step.

    Parameters
    ----------
    cov_df : pd.DataFrame
        Annualized covariance matrix. Index and columns are ticker strings.
    expected_returns : pd.Series
        Annualized expected returns, indexed by ticker.
    current_weights : pd.Series
        Current portfolio weights (used for current-portfolio annotation by Person C).
        Not used in the frontier solve itself — just passed through for convenience.
    risk_bucket : str
        One of 'Conservative', 'Moderate', 'Aggressive'.
        Determines the vol sweep range and max_position cap.
    n_points : int
        Number of points to solve along the frontier (default 20).
        Infeasible points are skipped, so actual rows may be slightly fewer.

    Returns
    -------
    pd.DataFrame matching the contracts/frontier.csv schema:
        target_vol_annual       float   The vol constraint used for this solve
        expected_return_annual  float   Achieved expected return at this point
        sharpe_ratio            float   return / vol (risk-free rate = 0)
        <TICKER>                float   Optimal weight for each ticker (one col per ticker)

    Notes
    -----
    - Ticker columns appear in the same order as cov_df.index.
    - Rows where the optimizer fails (infeasible) are silently dropped.
    - Person C overlays the current portfolio and optimized portfolio
      as named points on top of this curve.
    """
    params = get_bucket_params(risk_bucket)
    max_cap = params["max_position"]
    tickers = cov_df.index.tolist()

    cov = cov_df.loc[tickers, tickers].values.astype(float)
    mu = expected_returns[tickers].values.astype(float)

    # Find the floor: minimum achievable vol with the position cap
    vol_floor = _min_variance_vol(cov, max_cap)
    vol_ceil = params["vol_max"]

    if vol_floor >= vol_ceil:
        raise ValueError(
            f"Minimum variance portfolio vol ({vol_floor:.2%}) already exceeds "
            f"the '{risk_bucket}' bucket ceiling ({vol_ceil:.2%}). "
            "No frontier can be generated for this bucket."
        )

    # Sweep from just above the floor to the ceiling
    # Add a small epsilon above floor so the solve is never on the exact boundary
    target_vols = np.linspace(vol_floor + 1e-4, vol_ceil, n_points)

    rows = []
    for tv in target_vols:
        weights = _solve_at_target_vol(cov, mu, target_vol=tv, max_cap=max_cap)
        if weights is None:
            continue  # skip infeasible points silently

        actual_vol = float(np.sqrt(weights @ cov @ weights))
        actual_ret = float(mu @ weights)
        sharpe = actual_ret / actual_vol if actual_vol > 1e-8 else 0.0

        row: dict = {
            "target_vol_annual": round(tv, 6),
            "expected_return_annual": round(actual_ret, 6),
            "sharpe_ratio": round(sharpe, 6),
        }
        for ticker, w in zip(tickers, weights):
            row[ticker] = round(float(w), 6)

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "Frontier generation produced zero feasible points. "
            "Check covariance matrix and bucket parameters."
        )

    df = pd.DataFrame(rows)

    # Deduplicate: if two points have essentially the same return (optimizer
    # hit the same solution), keep the one with lower vol.
    df = (
        df.sort_values("expected_return_annual")
        .drop_duplicates(subset=["expected_return_annual"], keep="first")
        .reset_index(drop=True)
    )

    return df
