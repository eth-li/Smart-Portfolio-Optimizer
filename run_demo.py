#!/usr/bin/env python3
"""
run_demo.py
Owned by Person B.

Integration test + contract CSV generator.

Usage
-----
    python run_demo.py                          # Moderate bucket default (target vol = midpoint)
    python run_demo.py --bucket Aggressive      # different bucket
    python run_demo.py --target-vol 0.26        # exact vol from slider — bypasses bucket default

The --target-vol flag mirrors how the real app works: the user drags a slider
to a specific annualized vol and the optimizer runs against that number directly.
Bucket labels (Conservative / Moderate / Aggressive) are just zone markers on
the slider — they don't control the optimizer.
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from optimizer.solve import optimize
from optimizer.frontier import generate_frontier
from optimizer.metrics import compute_metrics, compute_portfolio_vol, compute_portfolio_return
from data.buckets import (
    classify_vol, VALID_BUCKETS, get_bucket_params,
    default_target_vol, DIVERSITY_FACTOR, BUCKET_PARAMS,
)


# ---------------------------------------------------------------------------
# Demo portfolio prices — replace with Person A's fetch.py in production
# ---------------------------------------------------------------------------
DEMO_PRICES: dict[str, float] = {
    "AAPL":  185.0,
    "MSFT":  375.0,
    "NVDA":  875.0,
    "GOOGL": 140.0,
    "XOM":   105.0,
}


def load_contracts() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    portfolio_df = pd.read_csv("contracts/portfolio_input.csv")
    cov_df = pd.read_csv("contracts/covariance.csv", index_col=0)
    exp_ret = pd.read_csv("contracts/expected_returns.csv").set_index("ticker")["expected_return_annual"]
    return portfolio_df, cov_df, exp_ret


def compute_current_weights(portfolio_df: pd.DataFrame) -> pd.Series:
    df = portfolio_df.copy()
    df["price"] = df["ticker"].map(DEMO_PRICES)
    missing = df[df["price"].isna()]["ticker"].tolist()
    if missing:
        raise KeyError(f"No demo price for: {missing}. Add to DEMO_PRICES in run_demo.py.")
    df["value"] = df["shares"] * df["price"]
    df["weight"] = df["value"] / df["value"].sum()
    return df.set_index("ticker")["weight"]


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def main(bucket: str = "Moderate", target_vol_override: float | None = None) -> None:

    # ------------------------------------------------------------------
    # Resolve target_vol and max_position from bucket + optional override
    # ------------------------------------------------------------------
    bucket_params = get_bucket_params(bucket)
    max_position = bucket_params["max_position"]

    if target_vol_override is not None:
        target_vol = target_vol_override
        print(f"\n🎚️  Slider override: target_vol={target_vol:.1%} (bucket label: {bucket})")
    else:
        target_vol = default_target_vol(bucket)
        print(f"\n🪣  Bucket default: {bucket} → target_vol={target_vol:.1%}")

    print_section(f"Demo Portfolio Optimizer — target_vol={target_vol:.1%}")

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    portfolio_df, cov_df, exp_ret = load_contracts()
    current_weights = compute_current_weights(portfolio_df)

    tickers = cov_df.index.tolist()
    n = len(tickers)
    curr_vol = compute_portfolio_vol(cov_df, current_weights)
    curr_ret = compute_portfolio_return(exp_ret, current_weights)
    min_weight_floor = DIVERSITY_FACTOR / n

    print(f"\n📂  Loaded {n} tickers: {tickers}")
    print(f"    Diversity floor: {DIVERSITY_FACTOR} / {n} = {min_weight_floor:.1%} per position")
    print(f"\n💼  Current portfolio:")
    for t in tickers:
        print(f"    {t:6s}  {current_weights[t]:.1%}")
    print(f"\n    Vol:    {curr_vol:.1%}  ({classify_vol(curr_vol)} zone)")
    print(f"    Return: {curr_ret:.1%}")
    print(f"    Sharpe: {curr_ret/curr_vol:.2f}")

    # ------------------------------------------------------------------
    # 2. Optimize
    # ------------------------------------------------------------------
    print_section("Step 1 — Optimize")
    result_df = optimize(
        cov_df, exp_ret, current_weights,
        target_vol=target_vol,
        max_position=max_position,
        diversity_factor=DIVERSITY_FACTOR,
    )
    result_df.to_csv("contracts/optimized_weights.csv", index=False)
    print(f"\n✓  Wrote contracts/optimized_weights.csv")
    print(result_df.to_string(index=False))

    opt_sum = result_df["optimized_weight"].sum()
    opt_max = result_df["optimized_weight"].max()
    opt_min = result_df["optimized_weight"].min()
    assert abs(opt_sum - 1.0) < 1e-4, f"Weights sum to {opt_sum:.6f}, expected 1.0"
    assert (result_df["optimized_weight"] >= -1e-6).all(), "Negative weights found"
    print(f"\n    ✓ Weights sum to {opt_sum:.6f}")
    print(f"    ✓ Max position: {opt_max:.1%}  (cap: {max_position:.0%})")
    print(f"    ✓ Min position: {opt_min:.1%}  (floor: {min_weight_floor:.1%})")

    # ------------------------------------------------------------------
    # 3. Frontier — spans full vol range across all buckets
    # ------------------------------------------------------------------
    print_section("Step 2 — Efficient Frontier")
    full_vol_max = BUCKET_PARAMS["Aggressive"]["vol_max"]  # show the whole curve
    frontier_df = generate_frontier(
        cov_df, exp_ret,
        vol_max=full_vol_max,
        max_position=BUCKET_PARAMS["Aggressive"]["max_position"],
        diversity_factor=0.0,  # no floor on frontier — show unconstrained curve shape
        n_points=30,
    )
    frontier_df.to_csv("contracts/frontier.csv", index=False)
    print(f"\n✓  Wrote contracts/frontier.csv ({len(frontier_df)} points, vol range: "
          f"{frontier_df['target_vol_annual'].min():.1%}–{frontier_df['target_vol_annual'].max():.1%})")
    print(frontier_df[["target_vol_annual", "expected_return_annual", "sharpe_ratio"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Metrics
    # ------------------------------------------------------------------
    print_section("Step 3 — Portfolio Metrics")
    opt_weights = result_df.set_index("ticker")["optimized_weight"]
    metrics_df = compute_metrics(cov_df, exp_ret, current_weights, opt_weights, target_vol)
    metrics_df.to_csv("contracts/metrics.csv", index=False)
    print(f"\n✓  Wrote contracts/metrics.csv")

    m = metrics_df.iloc[0]
    print(f"""
    Current portfolio:
      Vol:    {m['current_vol_annual']:.1%}  ({classify_vol(m['current_vol_annual'])} zone)
      Return: {m['current_return_annual']:.1%}
      Sharpe: {m['current_sharpe']:.2f}

    Optimized (target_vol={target_vol:.1%}):
      Vol:    {m['optimized_vol_annual']:.1%}
      Return: {m['optimized_return_annual']:.1%}
      Sharpe: {m['optimized_sharpe']:.2f}

    ✨  Headline stats:
      Vol reduction:       {m['vol_reduction_pct']:+.1f}%
      Return improvement:  {m['return_improvement_pct']:+.1f}%
    """)

    # ------------------------------------------------------------------
    # 5. Sanity checks
    # ------------------------------------------------------------------
    print_section("Sanity Checks")
    assert len(frontier_df) >= 5, "Frontier has fewer than 5 points"
    assert m["optimized_vol_annual"] <= target_vol + 0.01, \
        f"Optimized vol {m['optimized_vol_annual']:.1%} exceeds target {target_vol:.1%}"
    assert opt_min >= min_weight_floor - 1e-4, \
        f"Min weight {opt_min:.1%} is below diversity floor {min_weight_floor:.1%}"
    print("\n  ✓ All assertions passed")
    print("\n  ✅  Integration test complete. Contract CSVs ready for Person C.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the portfolio optimizer demo.")
    parser.add_argument(
        "--bucket", default="Moderate", choices=VALID_BUCKETS,
        help="Risk bucket label — sets max_position cap and default target_vol (default: Moderate)",
    )
    parser.add_argument(
        "--target-vol", type=float, default=None, dest="target_vol",
        help="Override target vol directly (e.g. 0.26 for 26%%). Mirrors the UI slider.",
    )
    args = parser.parse_args()
    main(bucket=args.bucket, target_vol_override=args.target_vol)