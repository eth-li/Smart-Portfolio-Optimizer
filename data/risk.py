"""Builds IV-scaled covariance matrix and computes expected returns.

Covariance matrix construction:
  - Off-diagonal (correlations): exponentially weighted, halflife=60 trading days
    (~3 months). Recent co-movement counts more than data from 2 years ago.
  - Diagonal (individual variances): IV² from current options prices.
    Forward-looking — reflects what the market expects, not just history.
  - Formula: Σ_ij = ρ_ewm[i,j] × σ_IV[i] × σ_IV[j]

Outputs:
  contracts/covariance.csv        — IV-scaled, EWM-correlated covariance matrix
  contracts/expected_returns.csv  — annualized expected returns (mean log return × 252)
                                    (not used by optimizer, kept for reference)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Halflife for exponential weighting of correlations.
# 60 trading days ≈ 3 months. Data from 3 months ago has half the weight
# of today's data. Data from 6 months ago has one-quarter the weight.
# Increase for more stable/conservative estimates; decrease to react faster
# to regime changes.
# ---------------------------------------------------------------------------
EWM_HALFLIFE = 60


def _ewm_correlation(returns: pd.DataFrame, halflife: int) -> np.ndarray:
    """
    Compute the exponentially weighted correlation matrix from daily returns.

    Uses the most recent observation of the rolling EWM covariance — i.e.
    the covariance matrix where every past return is weighted by
    λ^(T-t) where λ = exp(-ln(2) / halflife).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns, shape (T, n). No date column — tickers only.
    halflife : int
        Halflife in trading days.

    Returns
    -------
    np.ndarray, shape (n, n): correlation matrix, values in [-1, 1].
    """
    # pandas ewm().cov() returns MultiIndex (date, ticker) × ticker.
    # The last n rows are the most recent covariance matrix.
    n = returns.shape[1]
    ewm_cov_full = returns.ewm(halflife=halflife).cov()

    # Extract the last date's n×n block
    last_date = ewm_cov_full.index.get_level_values(0)[-1]
    ewm_cov = ewm_cov_full.loc[last_date].values.astype(float)  # shape (n, n)

    # Convert covariance → correlation: corr[i,j] = cov[i,j] / (std[i] * std[j])
    stds = np.sqrt(np.diag(ewm_cov))
    # Guard against zero variance (shouldn't happen with real data)
    stds = np.where(stds < 1e-10, 1e-10, stds)
    ewm_corr = ewm_cov / np.outer(stds, stds)

    # Clip to [-1, 1] — numerical precision can push values slightly outside
    ewm_corr = np.clip(ewm_corr, -1.0, 1.0)

    # Force exact symmetry and unit diagonal
    ewm_corr = (ewm_corr + ewm_corr.T) / 2
    np.fill_diagonal(ewm_corr, 1.0)

    return ewm_corr


def compute_covariance(
    returns_df: pd.DataFrame,
    iv_df: pd.DataFrame,
    halflife: int = EWM_HALFLIFE,
) -> tuple:
    """
    Build the IV-scaled covariance matrix with EWM correlations.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily log returns with a 'date' column plus one column per ticker.
    iv_df : pd.DataFrame
        Implied volatility data matching contracts/iv.csv schema.
    halflife : int
        Halflife in trading days for EWM correlation weighting. Default 60.

    Returns
    -------
    (cov_df, iv_df_updated)
        cov_df         : pd.DataFrame matching contracts/covariance.csv schema
        iv_df_updated  : iv_df with historical_fallback vols filled in
    """
    tickers = [c for c in returns_df.columns if c != "date"]
    ret = returns_df[tickers]  # keep as DataFrame for ewm()

    # ------------------------------------------------------------------
    # Off-diagonal: exponentially weighted correlations
    # ------------------------------------------------------------------
    ewm_corr = _ewm_correlation(ret, halflife)

    # Historical annualized vol — used only as fallback for missing IV
    hist_cov_ann = np.cov(ret.values, rowvar=False) * 252
    hist_std_ann = np.sqrt(np.diag(hist_cov_ann))

    # ------------------------------------------------------------------
    # Diagonal: implied volatility from options (forward-looking)
    # Fall back to historical vol if IV unavailable for a ticker
    # ------------------------------------------------------------------
    iv_map = dict(zip(iv_df["ticker"], iv_df["iv_annualized"]))
    source_map = dict(zip(iv_df["ticker"], iv_df["data_source"]))

    iv_vols = []
    for i, ticker in enumerate(tickers):
        if source_map.get(ticker) == "historical_fallback" or iv_map.get(ticker) is None:
            vol = round(float(hist_std_ann[i]), 4)
            iv_df.loc[iv_df["ticker"] == ticker, "iv_annualized"] = vol
        else:
            vol = float(iv_map[ticker])
        iv_vols.append(vol)

    iv_vols = np.array(iv_vols)

    # ------------------------------------------------------------------
    # Combine: Σ_ij = ρ_ewm[i,j] × σ_IV[i] × σ_IV[j]
    # ------------------------------------------------------------------
    cov = np.outer(iv_vols, iv_vols) * ewm_corr

    # Validate: must be positive semi-definite
    min_eig = np.linalg.eigvalsh(cov).min()
    assert min_eig >= -1e-6, (
        f"Covariance matrix not PSD (min eigenvalue: {min_eig:.4e}). "
        "This can happen if EWM correlations are computed from very few observations. "
        f"Try increasing halflife (current: {halflife}) or the data window."
    )

    cov_df = pd.DataFrame(np.round(cov, 6), index=tickers, columns=tickers)
    cov_df.index.name = "ticker"
    cov_df = cov_df.reset_index()

    return cov_df, iv_df


def compute_expected_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annualized expected returns as mean daily log return × 252.

    Note: not used by the optimizer (which is pure minimum-variance).
    Kept for reference and slide deck transparency.
    """
    tickers = [c for c in returns_df.columns if c != "date"]
    means = returns_df[tickers].mean() * 252
    return pd.DataFrame({
        "ticker": tickers,
        "expected_return_annual": means.values.round(4),
    })


def run(
    returns_path: str = "contracts/returns.csv",
    iv_path: str = "contracts/iv.csv",
) -> tuple:
    returns_df = pd.read_csv(returns_path)
    iv_df = pd.read_csv(iv_path)

    print(f"Computing IV-scaled covariance matrix (EWM halflife={EWM_HALFLIFE} days)...")
    cov_df, iv_df = compute_covariance(returns_df, iv_df)
    cov_df.to_csv("contracts/covariance.csv", index=False)
    print("  Saved covariance.csv")

    # Fallback vols written back to iv.csv
    iv_df.to_csv(iv_path, index=False)

    print("Computing expected returns (reference only, not used by optimizer)...")
    er_df = compute_expected_returns(returns_df)
    er_df.to_csv("contracts/expected_returns.csv", index=False)
    print("  Saved expected_returns.csv")

    return cov_df, er_df


if __name__ == "__main__":
    run()