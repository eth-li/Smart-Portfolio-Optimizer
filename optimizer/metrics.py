"""
optimizer/metrics.py
Owned by Person B.

Computes the headline portfolio statistics that Person C displays in the
summary metrics table at the top of the Streamlit app.

Public API
----------
compute_metrics(cov_df, current_weights, optimized_weights, target_vol)
    Returns single-row pd.DataFrame matching contracts/metrics.csv schema.

compute_portfolio_vol(cov_df, weights)
    Utility: annualized vol for an arbitrary weight vector. Used by Person C.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from data.buckets import classify_vol


# ---------------------------------------------------------------------------
# Utility functions (also useful for Person C's chart annotations)
# ---------------------------------------------------------------------------

def compute_portfolio_vol(cov_df: pd.DataFrame, weights: pd.Series) -> float:
    """
    Annualized portfolio volatility: sqrt(w^T * Sigma * w).

    Parameters
    ----------
    cov_df : pd.DataFrame
        Annualized covariance matrix.
    weights : pd.Series
        Portfolio weights, indexed by ticker. Must align with cov_df index.

    Returns
    -------
    float: annualized volatility (e.g. 0.15 for 15%).
    """
    tickers = cov_df.index.tolist()
    w = weights[tickers].values.astype(float)
    cov = cov_df.loc[tickers, tickers].values.astype(float)
    variance = float(w @ cov @ w)
    return float(np.sqrt(max(variance, 0.0)))


# ---------------------------------------------------------------------------
# Main metrics function
# ---------------------------------------------------------------------------

def compute_metrics(
    cov_df: pd.DataFrame,
    current_weights: pd.Series,
    optimized_weights: pd.Series,
    target_vol: float,
) -> pd.DataFrame:
    """
    Compute before/after portfolio statistics for the summary metrics table.

    Parameters
    ----------
    cov_df : pd.DataFrame
        Annualized covariance matrix. Index and columns are ticker strings.
    current_weights : pd.Series
        Current portfolio weights, indexed by ticker. Sums to ~1.
    optimized_weights : pd.Series
        Optimizer-recommended weights, indexed by ticker. Sums to ~1.
    target_vol : float
        The vol ceiling passed to the optimizer — comes from the UI slider.
        Stored in the output so Person C can display "optimized at X% vol".

    Returns
    -------
    Single-row pd.DataFrame matching the contracts/metrics.csv schema:

        current_vol_annual    float   Annualized vol of current portfolio
        optimized_vol_annual  float   Annualized vol of optimized portfolio
        vol_reduction_pct     float   % reduction: (curr-opt)/curr * 100
        vol_reduction_abs     float   Absolute reduction in vol (decimal, e.g. 0.05 = 5pp)
        target_vol            float   Vol ceiling used — from the slider
        current_bucket        str     Bucket label for current portfolio vol
        optimized_bucket      str     Bucket label for optimized portfolio vol
    """
    curr_vol = compute_portfolio_vol(cov_df, current_weights)
    opt_vol = compute_portfolio_vol(cov_df, optimized_weights)

    vol_reduction_pct = (curr_vol - opt_vol) / curr_vol * 100 if curr_vol > 1e-8 else 0.0
    vol_reduction_abs = curr_vol - opt_vol

    return pd.DataFrame([{
        "current_vol_annual":  round(curr_vol, 4),
        "optimized_vol_annual": round(opt_vol, 4),
        "vol_reduction_pct":   round(vol_reduction_pct, 2),
        "vol_reduction_abs":   round(vol_reduction_abs, 4),
        "target_vol":          round(target_vol, 4),
        "current_bucket":      classify_vol(curr_vol),
        "optimized_bucket":    classify_vol(opt_vol),
    }])
