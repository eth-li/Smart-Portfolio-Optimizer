#!/usr/bin/env python3
"""
run_demo.py
Owned by Person B.

Integration test + contract CSV generator.
Runs the full optimizer pipeline against the demo portfolio in contracts/portfolio_input.csv
and writes the output contract CSVs that Person C needs to build charts.

Usage
-----
    python run_demo.py                      # Moderate bucket (default)
    python run_demo.py --bucket Aggressive  # try a different bucket

What it does
------------
1. Loads mock covariance + expected returns from contracts/
2. Computes current weights from demo portfolio (shares × demo prices)
3. Runs optimize() → writes contracts/optimized_weights.csv
4. Runs generate_frontier() → writes contracts/frontier.csv
5. Runs compute_metrics() → writes contracts/metrics.csv
6. Prints a sanity-check summary to stdout

Swap in Person A's real data
-----------------------------
Replace contracts/covariance.csv and contracts/expected_returns.csv with the
files Person A generates. This script will automatically use them.
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
from data.buckets import classify_vol, VALID_BUCKETS


# ---------------------------------------------------------------------------
# Demo portfolio prices (used only to compute current weights from share counts)
# Replace with live prices once Person A's fetch.py is wired up.
# ---------------------------------------------------------------------------
DEMO_PRICES: dict[str, float] = {
    "AAPL":  185.0,
    "MSFT":  375.0,
    "NVDA":  875.0,
    "GOOGL": 140.0,
    "XOM":   105.0,
}


def load_contracts() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load input contract CSVs."""
    portfolio_df = pd.read_csv("contracts/portfolio_input.csv")
    cov_df = pd.read_csv("contracts/covariance.csv", index_col=0)
    exp_ret = pd.read_csv("contracts/expected_returns.csv").set_index("ticker")["expected_return_annual"]
    return portfolio_df, cov_df, exp_ret


def compute_current_weights(portfolio_df: pd.DataFrame) -> pd.Series:
    """
    Compute current portfolio weights from share counts × demo prices.
    In the real app, prices come from Person A's fetch.py.
    """
    df = portfolio_df.copy()
    df["price"] = df["ticker"].map(DEMO_PRICES)
    missing = df[df["price"].isna()]["ticker"].tolist()
    if missing:
        raise KeyError(
            f"No demo price for ticker(s): {missing}. "
            "Add them to DEMO_PRICES in run_demo.py."
        )
    df["value"] = df["shares"] * df["price"]
    total = df["value"].sum()
    df["weight"] = df["value"] / total
    return df.set_index("ticker")["weight"]


def print_section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def main(risk_bucket: str = "Moderate") -> None:
    print_section(f"Demo Portfolio Optimizer — {risk_bucket} Bucket")

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    portfolio_df, cov_df, exp_ret = load_contracts()
    current_weights = compute_current_weights(portfolio_df)

    tickers = cov_df.index.tolist()
    curr_vol = compute_portfolio_vol(cov_df, current_weights)
    curr_ret = compute_portfolio_return(exp_ret, current_weights)

    print(f"\n📂  Loaded {len(tickers)} tickers: {tickers}")
    print(f"\n💼  Current portfolio:")
    for t in tickers:
        print(f"    {t:6s}  weight={current_weights[t]:.1%}")
    print(f"\n    Vol:    {curr_vol:.1%}  ({classify_vol(curr_vol)} bucket)")
    print(f"    Return: {curr_ret:.1%}")
    print(f"    Sharpe: {curr_ret/curr_vol:.2f}")

    # ------------------------------------------------------------------
    # 2. Optimize
    # ------------------------------------------------------------------
    print_section("Step 1 — Optimize")
    result_df = optimize(cov_df, exp_ret, current_weights, risk_bucket)
    result_df.to_csv("contracts/optimized_weights.csv", index=False)
    print(f"\n✓  Wrote contracts/optimized_weights.csv")
    print(result_df.to_string(index=False))

    # Weight sanity checks
    opt_sum = result_df["optimized_weight"].sum()
    opt_max = result_df["optimized_weight"].max()
    assert abs(opt_sum - 1.0) < 1e-4, f"Weights sum to {opt_sum:.6f}, expected 1.0"
    assert (result_df["optimized_weight"] >= -1e-6).all(), "Negative weights found"
    print(f"\n    ✓ Weights sum to {opt_sum:.6f}")
    print(f"    ✓ Max position: {opt_max:.1%}")

    # ------------------------------------------------------------------
    # 3. Frontier
    # ------------------------------------------------------------------
    print_section("Step 2 — Efficient Frontier")
    frontier_df = generate_frontier(cov_df, exp_ret, current_weights, risk_bucket, n_points=20)
    frontier_df.to_csv("contracts/frontier.csv", index=False)
    print(f"\n✓  Wrote contracts/frontier.csv ({len(frontier_df)} frontier points)")
    print(frontier_df[["target_vol_annual", "expected_return_annual", "sharpe_ratio"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Metrics
    # ------------------------------------------------------------------
    print_section("Step 3 — Portfolio Metrics")
    opt_weights = result_df.set_index("ticker")["optimized_weight"]
    metrics_df = compute_metrics(cov_df, exp_ret, current_weights, opt_weights, risk_bucket)
    metrics_df.to_csv("contracts/metrics.csv", index=False)
    print(f"\n✓  Wrote contracts/metrics.csv")

    m = metrics_df.iloc[0]
    print(f"""
    Current portfolio:
      Vol:    {m['current_vol_annual']:.1%}
      Return: {m['current_return_annual']:.1%}
      Sharpe: {m['current_sharpe']:.2f}

    Optimized portfolio ({risk_bucket}):
      Vol:    {m['optimized_vol_annual']:.1%}
      Return: {m['optimized_return_annual']:.1%}
      Sharpe: {m['optimized_sharpe']:.2f}

    ✨  Headline stats (lead with these in the demo):
      Vol reduction:       {m['vol_reduction_pct']:+.1f}%
      Return improvement:  {m['return_improvement_pct']:+.1f}%
    """)

    # ------------------------------------------------------------------
    # 5. Final sanity check
    # ------------------------------------------------------------------
    print_section("Sanity Checks")
    assert len(frontier_df) >= 5, "Frontier has fewer than 5 points — too sparse"
    assert m["optimized_vol_annual"] <= m["current_vol_annual"] + 0.005, \
        "Optimized vol is higher than current vol — check constraints"
    print("\n  ✓ All assertions passed")
    print("\n  ✅  Integration test complete. Contract CSVs ready for Person C.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the portfolio optimizer demo.")
    parser.add_argument(
        "--bucket",
        default="Moderate",
        choices=VALID_BUCKETS,
        help="Risk bucket to optimize for (default: Moderate)",
    )
    args = parser.parse_args()
    main(risk_bucket=args.bucket)
