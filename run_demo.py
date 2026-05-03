#!/usr/bin/env python3
"""
run_demo.py
Owned by Person B.

Integration test + contract CSV generator.

Usage
-----
    python run_demo.py                          # Medium bucket default (target vol = midpoint)
    python run_demo.py --bucket High            # different bucket
    python run_demo.py --target-vol 0.26        # exact vol from slider — bypasses bucket default

The --target-vol flag mirrors how the real app works: the user drags a slider
to a specific annualized vol and the optimizer runs against that number directly.
Bucket labels (Low / Medium-Low / Medium / Medium-High / High) are just zone
markers on the slider — they don't control the optimizer.
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from optimizer.solve import optimize, black_litterman_mu
from optimizer.frontier import generate_frontier
from optimizer.metrics import compute_metrics, compute_portfolio_vol
from data.buckets import (
    classify_vol, VALID_BUCKETS, get_bucket_params, dynamic_max_position,
    default_target_vol, DIVERSITY_FACTOR, BUCKET_PARAMS,
)


def load_contracts() -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio_df = pd.read_csv("contracts/portfolio_input.csv")
    if "current_weight" not in portfolio_df.columns:
        if "weight" in portfolio_df.columns:
            portfolio_df = portfolio_df.rename(columns={"weight": "current_weight"})
        elif "shares" in portfolio_df.columns:
            total_shares = portfolio_df["shares"].sum()
            if total_shares <= 0:
                raise ValueError("contracts/portfolio_input.csv has no positive shares to normalize.")
            portfolio_df["current_weight"] = portfolio_df["shares"] / total_shares
        else:
            raise ValueError(
                "contracts/portfolio_input.csv must contain one of: "
                "current_weight, weight, or shares."
            )
    cov_df = pd.read_csv("contracts/covariance.csv", index_col=0)

    portfolio_tickers = portfolio_df["ticker"].astype(str).str.upper().tolist()
    available_tickers = [ticker for ticker in portfolio_tickers if ticker in cov_df.index]
    missing_tickers = sorted(set(portfolio_tickers) - set(available_tickers))

    if missing_tickers:
        print(f"Warning: dropping tickers missing from covariance.csv: {', '.join(missing_tickers)}")

    if len(available_tickers) < 2:
        raise ValueError("Need at least 2 portfolio tickers present in contracts/covariance.csv.")

    portfolio_df = portfolio_df[portfolio_df["ticker"].isin(available_tickers)].copy()
    cov_df = cov_df.loc[available_tickers, available_tickers]
    return portfolio_df, cov_df


def load_expected_returns(tickers: list[str], cov_df: pd.DataFrame) -> pd.Series:
    """
    Build BL-adjusted expected returns for the demo universe from returns.csv.
    Falls back to contracts/expected_returns.csv if returns history is unavailable.
    """
    try:
        returns_df = pd.read_csv("contracts/returns.csv", index_col="date")
        annual_ret = returns_df[tickers].mean() * 252
        cross_sectional_mean = annual_ret.mean()
        shrunk = 0.6 * annual_ret + 0.4 * cross_sectional_mean
        winsorized = shrunk.clip(
            lower=shrunk.mean() - 2 * shrunk.std(),
            upper=shrunk.mean() + 2 * shrunk.std(),
        )
        bl_mu = black_litterman_mu(
            cov_df.loc[tickers, tickers].values.astype(float),
            winsorized[tickers].values.astype(float),
        )
        return pd.Series(bl_mu, index=tickers)
    except Exception:
        exp_ret_df = pd.read_csv("contracts/expected_returns.csv")
        exp_ret = exp_ret_df.set_index("ticker")["expected_return_annual"].reindex(tickers)
        if exp_ret.isna().any():
            raise ValueError("Missing expected returns for one or more demo tickers.")
        return exp_ret


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def main(bucket: str = "Medium", target_vol_override: float | None = None) -> None:
    # ------------------------------------------------------------------
    # Resolve target_vol and max_position from bucket + optional override
    # ------------------------------------------------------------------
    bucket_params = get_bucket_params(bucket)

    if target_vol_override is not None:
        target_vol = target_vol_override
        print(f"\nSlider override: target_vol={target_vol:.1%} (bucket label: {bucket})")
    else:
        target_vol = default_target_vol(bucket)
        print(f"\nBucket default: {bucket} -> target_vol={target_vol:.1%}")

    print_section(f"Demo Portfolio Optimizer — target_vol={target_vol:.1%}")

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    portfolio_df, cov_df = load_contracts()
    current_weights = portfolio_df.set_index("ticker")["current_weight"]

    tickers = cov_df.index.tolist()
    n = len(tickers)
    max_position = dynamic_max_position(bucket, n)
    exp_ret = load_expected_returns(tickers, cov_df)
    curr_vol = compute_portfolio_vol(cov_df, current_weights)
    min_weight_floor = DIVERSITY_FACTOR / n

    print(f"\nLoaded {n} tickers: {tickers}")
    print(f"    Diversity floor: {DIVERSITY_FACTOR} / {n} = {min_weight_floor:.1%} per position")
    print(f"\nCurrent portfolio:")
    for t in tickers:
        print(f"    {t:6s}  {current_weights[t]:.1%}")
    print(f"\n    Vol: {curr_vol:.1%}  ({classify_vol(curr_vol)} zone)")

    # ------------------------------------------------------------------
    # 2. Optimize
    # ------------------------------------------------------------------
    print_section("Step 1 — Optimize")
    result_df = optimize(
        cov_df, current_weights,
        target_vol=target_vol,
        max_position=max_position,
        diversity_factor=DIVERSITY_FACTOR,
        exp_ret=exp_ret,
    )
    result_df.to_csv("contracts/optimized_weights.csv", index=False)
    print(f"\n  Wrote contracts/optimized_weights.csv")
    print(result_df.to_string(index=False))

    opt_sum = result_df["optimized_weight"].sum()
    opt_max = result_df["optimized_weight"].max()
    opt_min = result_df["optimized_weight"].min()
    assert abs(opt_sum - 1.0) < 1e-4, f"Weights sum to {opt_sum:.6f}, expected 1.0"
    assert (result_df["optimized_weight"] >= min_weight_floor - 1e-4).all(), \
        f"Some weights below diversity floor {min_weight_floor:.1%}"
    assert (result_df["optimized_weight"] <= max_position + 1e-4).all(), \
        f"Some weights exceed max_position cap {max_position:.1%}"
    print(f"\n    Weights sum to {opt_sum:.6f}")
    print(f"    Max position: {opt_max:.1%}  (cap: {max_position:.0%})")
    print(f"    Min position: {opt_min:.1%}  (floor: {min_weight_floor:.1%})")

    # ------------------------------------------------------------------
    # 3. Frontier — spans full vol range across all buckets
    # ------------------------------------------------------------------
    print_section("Step 2 — Minimum Variance Frontier")
    full_vol_max = BUCKET_PARAMS["High"]["vol_max"]
    frontier_df = generate_frontier(
        cov_df,
        vol_max=full_vol_max,
        max_position=dynamic_max_position("High", n),
        diversity_factor=0.0,
        n_points=30,
        exp_ret=exp_ret,
    )
    frontier_df.to_csv("contracts/frontier.csv", index=False)
    print(f"\n  Wrote contracts/frontier.csv ({len(frontier_df)} points, vol range: "
          f"{frontier_df['target_vol_annual'].min():.1%}–{frontier_df['target_vol_annual'].max():.1%})")
    print(frontier_df[["target_vol_annual", "realized_vol_annual"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Metrics
    # ------------------------------------------------------------------
    print_section("Step 3 — Portfolio Metrics")
    opt_weights = result_df.set_index("ticker")["optimized_weight"]
    metrics_df = compute_metrics(cov_df, current_weights, opt_weights, target_vol)
    metrics_df.to_csv("contracts/metrics.csv", index=False)
    print(f"\n  Wrote contracts/metrics.csv")

    m = metrics_df.iloc[0]
    print(f"""
    Current portfolio:
      Vol: {m['current_vol_annual']:.1%}  ({m['current_bucket']} zone)

    Optimized (target_vol={target_vol:.1%}):
      Vol: {m['optimized_vol_annual']:.1%}  ({m['optimized_bucket']} zone)

    Headline stats:
      Vol reduction: {m['vol_reduction_pct']:+.1f}%  ({m['vol_reduction_abs']:.1%} abs)
    """)

    # ------------------------------------------------------------------
    # 5. Sanity checks
    # ------------------------------------------------------------------
    print_section("Sanity Checks")
    assert len(frontier_df) >= 5, f"Frontier has only {len(frontier_df)} feasible points (need >= 5)"
    assert frontier_df["realized_vol_annual"].is_monotonic_increasing, \
        "Frontier realized vols should be sorted ascending"
    assert m["optimized_vol_annual"] <= target_vol + 0.01, \
        f"Optimized vol {m['optimized_vol_annual']:.1%} exceeds target {target_vol:.1%} + 1pp"
    if m["optimized_vol_annual"] >= m["current_vol_annual"]:
        print(f"    NOTE: optimized vol ({m['optimized_vol_annual']:.1%}) >= current vol "
              f"({m['current_vol_annual']:.1%}) — current portfolio may already be near minimum variance.")
    assert opt_min >= min_weight_floor - 1e-4, \
        f"Min weight {opt_min:.1%} is below diversity floor {min_weight_floor:.1%}"
    assert opt_max <= max_position + 1e-4, \
        f"Max weight {opt_max:.1%} exceeds cap {max_position:.1%}"
    print("\n  All assertions passed")
    print("\n  Integration test complete. Contract CSVs ready for Person C.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the portfolio optimizer demo.")
    parser.add_argument(
        "--bucket", default="Medium", choices=VALID_BUCKETS,
        help="Risk bucket label — sets max_position cap and default target_vol (default: Medium)",
    )
    parser.add_argument(
        "--target-vol", type=float, default=None, dest="target_vol",
        help="Override target vol directly (e.g. 0.26 for 26%%). Mirrors the UI slider.",
    )
    args = parser.parse_args()
    main(bucket=args.bucket, target_vol_override=args.target_vol)
