"""
optimizer/frontier.py
Owned by Person B.

Generates the minimum variance frontier by sweeping target_vol from the minimum-
variance portfolio vol to a given vol ceiling, solving the optimizer at each step.

Public API
----------
generate_frontier(cov_df, vol_max, max_position, diversity_factor, n_points)
    Returns pd.DataFrame matching the contracts/frontier.csv schema.

The frontier spans the full achievable vol range — Person C plots the whole curve
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
    cov_psd = cp.psd_wrap(cov)
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(w, cov_psd)),
        [cp.sum(w) == 1, w >= min_weight, w <= max_cap],
    )
    problem.solve(solver=cp.CLARABEL)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError("Could not find minimum variance portfolio.")

    return float(np.sqrt(max(problem.value, 0.0)))


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def _solve_max_return_at_vol(
    cov: np.ndarray,
    mu: np.ndarray,
    target_vol: float,
    max_cap: float,
    diversity_factor: float = 0.0,
) -> np.ndarray | None:
    """
    Maximize expected return subject to portfolio vol <= target_vol.
    This is the classic Markowitz problem — gives a different portfolio at
    each vol level, producing a proper curved frontier.
    """
    n = cov.shape[0]
    min_weight = diversity_factor / n
    if n * min_weight > 1.0:
        min_weight = 1.0 / n

    w = cp.Variable(n)
    cov_psd = cp.psd_wrap(cov)
    problem = cp.Problem(
        cp.Maximize(mu @ w),
        [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_cap,
            cp.quad_form(w, cov_psd) <= target_vol ** 2,
        ],
    )
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


def generate_frontier(
    cov_df: pd.DataFrame,
    vol_max: float,
    max_position: float = 0.50,
    diversity_factor: float = 0.0,
    n_points: int = 30,
    exp_ret: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Generate the efficient frontier by sweeping target_vol from the minimum-
    variance floor to vol_max.

    If exp_ret is provided (recommended), solves Markowitz at each point:
    maximize expected return subject to vol <= target. This produces a genuine
    curve with different portfolios at each risk level.

    If exp_ret is None, falls back to pure min-variance (all points identical —
    only the target_vol column varies). Avoid this for chart display.

    Returns
    -------
    pd.DataFrame with columns:
        target_vol_annual     float   Vol constraint used for this solve
        realized_vol_annual   float   Actual portfolio vol (sqrt(w^T Σ w))
        expected_return_annual float  Expected return (w^T μ), NaN if no exp_ret
        <TICKER>              float   Optimal weight for each ticker
    """
    tickers = cov_df.index.tolist()
    cov = cov_df.loc[tickers, tickers].values.astype(float)

    vol_floor = _min_variance_vol(cov, max_position, diversity_factor)

    if vol_floor >= vol_max:
        raise ValueError(
            f"Minimum variance portfolio vol ({vol_floor:.2%}) already exceeds "
            f"vol_max ({vol_max:.2%}). Increase vol_max or relax constraints."
        )

    target_vols = np.linspace(vol_floor + 1e-4, vol_max, n_points)
    mu = exp_ret[tickers].values.astype(float) if exp_ret is not None else None

    rows = []
    for tv in target_vols:
        if mu is not None:
            weights = _solve_max_return_at_vol(cov, mu, tv, max_position, diversity_factor)
        else:
            weights = _solve_at_target_vol(cov, tv, max_position, diversity_factor)

        if weights is None:
            continue

        realized_vol = float(np.sqrt(max(float(weights @ cov @ weights), 0.0)))
        expected_ret = float(mu @ weights) if mu is not None else float("nan")

        row: dict = {
            "target_vol_annual":      round(tv, 6),
            "realized_vol_annual":    round(realized_vol, 6),
            "expected_return_annual": round(expected_ret, 6),
        }
        for ticker, w in zip(tickers, weights):
            row[ticker] = round(float(w), 6)

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "Frontier generation produced zero feasible points. "
            "Check covariance matrix and vol_max."
        )

    return (
        pd.DataFrame(rows)
        .sort_values("realized_vol_annual")
        .drop_duplicates(subset=["realized_vol_annual"], keep="first")
        .reset_index(drop=True)
    )
