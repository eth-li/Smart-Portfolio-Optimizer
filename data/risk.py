"""Builds IV-scaled covariance matrix and computes expected returns.

Outputs:
  contracts/covariance.csv        — IV-scaled annualized covariance matrix
  contracts/expected_returns.csv  — annualized expected returns (mean log return × 252)
"""

import numpy as np
import pandas as pd


def compute_covariance(returns_df: pd.DataFrame, iv_df: pd.DataFrame) -> tuple:
    """Return (cov_df, iv_df_updated) where iv_df has fallback vols filled in."""
    tickers = [c for c in returns_df.columns if c != "date"]
    ret = returns_df[tickers].values  # shape: (T, N)

    # Annualized historical covariance and correlation
    hist_cov_ann = np.cov(ret, rowvar=False) * 252
    hist_std_ann = np.sqrt(np.diag(hist_cov_ann))
    hist_corr = np.corrcoef(ret, rowvar=False)

    # Resolve IV vols: use options_chain IV or fall back to historical vol
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

    # IV-scaled covariance: diagonal = IV²; off-diagonal = corr_hist * iv_i * iv_j
    cov = np.outer(iv_vols, iv_vols) * hist_corr

    # Validate: must be positive semi-definite
    min_eig = np.linalg.eigvalsh(cov).min()
    assert min_eig >= -1e-8, f"Covariance matrix not PSD (min eigenvalue: {min_eig:.4e})"

    cov_df = pd.DataFrame(np.round(cov, 6), index=tickers, columns=tickers)
    cov_df.index.name = "ticker"
    cov_df = cov_df.reset_index()

    return cov_df, iv_df


def compute_expected_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    tickers = [c for c in returns_df.columns if c != "date"]
    means = returns_df[tickers].mean() * 252  # mean daily log return × 252
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

    print("Computing IV-scaled covariance matrix...")
    cov_df, iv_df = compute_covariance(returns_df, iv_df)
    cov_df.to_csv("contracts/covariance.csv", index=False)
    print("  Saved covariance.csv")

    # Write back updated IV (historical fallback values filled in)
    iv_df.to_csv(iv_path, index=False)

    print("Computing expected returns...")
    er_df = compute_expected_returns(returns_df)
    er_df.to_csv("contracts/expected_returns.csv", index=False)
    print("  Saved expected_returns.csv")

    return cov_df, er_df


if __name__ == "__main__":
    run()
