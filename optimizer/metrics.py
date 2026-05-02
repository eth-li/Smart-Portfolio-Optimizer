"""
optimizer/metrics.py
Owned by Person B.

Computes the headline portfolio statistics that Person C displays in the
summary metrics table at the top of the Streamlit app.

Public API
----------
compute_metrics(cov_df, expected_returns, current_weights, optimized_weights, risk_bucket)
    Returns single-row pd.DataFrame matching contracts/metrics.csv schema.

compute_portfolio_vol(cov_df, weights)
    Utility: annualized vol for an arbitrary weight vector. Used by Person C.

compute_portfolio_return(expected_returns, weights)
    Utility: expected return for an arbitrary weight vector. Used by Person C.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from data.buckets import get_bucket_params, classify_vol


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


def compute_portfolio_return(expected_returns: pd.Series, weights: pd.Series) -> float:
    """
    Annualized expected portfolio return: mu^T * w.

    Parameters
    ----------
    expected_returns : pd.Series
        Annualized expected returns, indexed by ticker.
    weights : pd.Series
        Portfolio weights, indexed by ticker.

    Returns
    -------
    float: annualized expected return (e.g. 0.18 for 18%).
    """
    tickers = weights.index.tolist()
    mu = expected_returns[tickers].values.astype(float)
    w = weights[tickers].values.astype(float)
    return float(mu @ w)


def compute_sharpe(portfolio_return: float, portfolio_vol: float, risk_free: float = 0.0) -> float:
    """
    Sharpe ratio = (return - risk_free) / vol.
    Risk-free rate defaults to 0 for simplicity; document this assumption in the slides.
    """
    if portfolio_vol < 1e-8:
        return 0.0
    return (portfolio_return - risk_free) / portfolio_vol


# ---------------------------------------------------------------------------
# Main metrics function
# ---------------------------------------------------------------------------

def compute_metrics(
    cov_df: pd.DataFrame,
    expected_returns: pd.Series,
    current_weights: pd.Series,
    optimized_weights: pd.Series,
    risk_bucket: str,
) -> pd.DataFrame:
    """
    Compute before/after portfolio statistics for the summary metrics table.

    Parameters
    ----------
    cov_df : pd.DataFrame
        Annualized covariance matrix. Index and columns are ticker strings.
    expected_returns : pd.Series
        Annualized expected returns, indexed by ticker.
    current_weights : pd.Series
        Current portfolio weights (by market value), indexed by ticker. Sums to ~1.
    optimized_weights : pd.Series
        Optimizer-recommended weights, indexed by ticker. Sums to ~1.
    risk_bucket : str
        One of 'Conservative', 'Moderate', 'Aggressive' (as selected by the user).

    Returns
    -------
    Single-row pd.DataFrame matching the contracts/metrics.csv schema:

        current_vol_annual        float   Annualized vol of current portfolio
        optimized_vol_annual      float   Annualized vol of optimized portfolio
        current_return_annual     float   Expected annual return of current portfolio
        optimized_return_annual   float   Expected annual return of optimized portfolio
        current_sharpe            float   Sharpe of current portfolio (risk-free = 0)
        optimized_sharpe          float   Sharpe of optimized portfolio (risk-free = 0)
        risk_bucket               str     Bucket label as selected by user
        vol_reduction_pct         float   % reduction in vol: (curr-opt)/curr * 100
        return_improvement_pct    float   % improvement in return: (opt-curr)/curr * 100

    Notes
    -----
    - vol_reduction_pct is the headline stat for the demo — show this first.
    - Sharpe uses risk-free rate = 0; note this on the presentation slide.
    - A positive vol_reduction_pct AND positive return_improvement_pct is the
      "free lunch" story — diversification improved both simultaneously.
    """
    curr_vol = compute_portfolio_vol(cov_df, current_weights)
    opt_vol = compute_portfolio_vol(cov_df, optimized_weights)
    curr_ret = compute_portfolio_return(expected_returns, current_weights)
    opt_ret = compute_portfolio_return(expected_returns, optimized_weights)
    curr_sharpe = compute_sharpe(curr_ret, curr_vol)
    opt_sharpe = compute_sharpe(opt_ret, opt_vol)

    vol_reduction_pct = (curr_vol - opt_vol) / curr_vol * 100 if curr_vol > 1e-8 else 0.0
    ret_improvement_pct = (opt_ret - curr_ret) / curr_ret * 100 if abs(curr_ret) > 1e-8 else 0.0

    return pd.DataFrame([{
        "current_vol_annual":      round(curr_vol, 4),
        "optimized_vol_annual":    round(opt_vol, 4),
        "current_return_annual":   round(curr_ret, 4),
        "optimized_return_annual": round(opt_ret, 4),
        "current_sharpe":          round(curr_sharpe, 4),
        "optimized_sharpe":        round(opt_sharpe, 4),
        "risk_bucket":             risk_bucket,
        "vol_reduction_pct":       round(vol_reduction_pct, 2),
        "return_improvement_pct":  round(ret_improvement_pct, 2),
    }])
