"""
optimizer/frontier.py
Owned by Person B.

Generates the efficient frontier by sweeping target_vol from the minimum-variance
portfolio vol to a given vol ceiling, solving the optimizer at each step.

Public API
----------
generate_frontier(cov_df, expected_returns, vol_min, vol_max, max_position, diversity_factor, n_points)
    Returns pd.DataFrame matching the contracts/frontier.csv schema.

The frontier spans the full achievable range — Person C plots the whole curve
and marks the current portfolio + the user's chosen point on top of it.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import cvxpy as cp

from data.buckets import DIVERSITY_FACTOR as DEFAULT_DIVERSITY_FACTOR
from optimizer.solve import _solve_at_target_vol


# ---------------------------------------------------------------------------
# Minimum variance portfolio helper
# ---------------------------------------------------------------------------

def _min_variance_vol(
    cov: np.ndarray,
    max_cap: float,
    diversity_factor: float = DEFAULT_DIVERSITY_FACTOR,
) -> float:
    """
    Find the annualized vol of the global minimum-variance portfolio.
    Applies the same diversity floor and max cap as the main optimizer
    so the frontier floor is consistent with what optimize() can achieve.

    Parameters
    ----------
    cov : np.ndarray, shape (n, n)
    max_cap : float
    diversity_factor : float

    Returns
    -------
    float: annualized vol of the min-variance portfolio.
    """
    n = cov.shape[0]
    min_weight = diversity_factor / n

    w = cp.Variable(n)
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(w, cov)),
        [cp.sum(w) == 1, w >= min_weight, w <= max_cap],
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
    vol_max: float,
    max_position: float = 0.40,
    diversity_factor: float = DEFAULT_DIVERSITY_FACTOR,
    n_points: int = 30,
) -> pd.DataFrame:
    """
    Generate efficient frontier data points across the full achievable vol range.

    Sweeps target_vol from just above the minimum-variance floor to vol_max,
    solving the max-return problem at each step.

    Parameters
    ----------
    cov_df : pd.DataFrame
        Annualized covariance matrix. Index and columns are ticker strings.
    expected_returns : pd.Series
        Annualized expected returns, indexed by ticker.
    vol_max : float
        Upper bound for the frontier sweep — typically the Aggressive bucket's
        vol_max (0.40). Pass a higher value to extend the curve further right.
    max_position : float
        Per-ticker weight cap. Use the most permissive bucket's max_position
        (Aggressive = 0.50) so the full frontier is visible, not just one bucket.
    diversity_factor : float
        Controls minimum weight floor: min_weight = diversity_factor / n.
        Defaults to DIVERSITY_FACTOR from buckets.py.
    n_points : int
        Number of points to solve along the frontier (default 30).
        More points = smoother curve. Infeasible points are silently skipped.

    Returns
    -------
    pd.DataFrame matching the contracts/frontier.csv schema:
        target_vol_annual       float   The vol constraint used for this solve
        expected_return_annual  float   Achieved expected return at this point
        sharpe_ratio            float   return / vol (risk-free rate = 0)
        <TICKER>                float   Optimal weight for each ticker at this point

    Notes
    -----
    - The frontier spans the full asset universe range, not just one bucket.
      Person C draws the bucket zone markers (Conservative/Moderate/Aggressive)
      as shaded regions on top of this single curve.
    - Person C marks the current portfolio and the user's chosen point
      (determined by the slider) as labeled dots on the curve.
    - Ticker columns are in the same order as cov_df.index.
    """
    tickers = cov_df.index.tolist()
    cov = cov_df.loc[tickers, tickers].values.astype(float)
    mu = expected_returns[tickers].values.astype(float)

    # Floor: smallest achievable vol given the diversity and position constraints
    vol_floor = _min_variance_vol(cov, max_position, diversity_factor)

    if vol_floor >= vol_max:
        raise ValueError(
            f"Minimum variance portfolio vol ({vol_floor:.2%}) already exceeds "
            f"vol_max ({vol_max:.2%}). Increase vol_max or relax constraints."
        )

    target_vols = np.linspace(vol_floor + 1e-4, vol_max, n_points)

    rows = []
    for tv in target_vols:
        # Frontier uses diversity_factor=0 so it can freely explore the full shape.
        # The per-ticker floor is intentionally relaxed here — the floor only
        # applies when optimize() is called with the user's actual portfolio.
        weights = _solve_at_target_vol(
            cov, mu,
            target_vol=tv,
            max_cap=max_position,
            diversity_factor=0.0,   # no floor on frontier — show full curve shape
        )
        if weights is None:
            continue

        actual_vol = float(np.sqrt(weights @ cov @ weights))
        actual_ret = float(mu @ weights)
        sharpe = actual_ret / actual_vol if actual_vol > 1e-8 else 0.0

        row: dict = {
            "target_vol_annual":      round(tv, 6),
            "expected_return_annual": round(actual_ret, 6),
            "sharpe_ratio":           round(sharpe, 6),
        }
        for ticker, w in zip(tickers, weights):
            row[ticker] = round(float(w), 6)

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "Frontier generation produced zero feasible points. "
            "Check covariance matrix and vol_max."
        )

    df = pd.DataFrame(rows)
    df = (
        df.sort_values("expected_return_annual")
        .drop_duplicates(subset=["expected_return_annual"], keep="first")
        .reset_index(drop=True)
    )

    return df