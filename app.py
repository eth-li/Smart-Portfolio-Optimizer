"""Smart Portfolio Optimizer — Streamlit entry point (Person C)

Day 1: reads chart data from contracts/*.csv (Person B's real optimizer outputs).
Integration: _run_pipeline already calls live functions when all modules import cleanly.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from app.charts import plot_efficient_frontier, plot_allocation_comparison, plot_metrics_table

CONTRACTS = Path("contracts")

# ── Optional live imports (graceful fallback to contract mocks) ──────────────
try:
    from data.fetch import fetch_returns, fetch_iv, _get_spot
    from data.risk import compute_covariance, compute_expected_returns
    DATA_LIVE = True
except ImportError:
    DATA_LIVE = False

try:
    # check_exceeds_bucket removed — use get_bucket_params + classify_vol instead
    from data.buckets import (
        get_bucket_params,
        classify_vol,
        default_target_vol,
        BUCKET_PARAMS,
    )
    from optimizer.solve import optimize
    from optimizer.frontier import generate_frontier
    from optimizer.metrics import compute_metrics
    OPTIMIZER_LIVE = True
except ImportError:
    OPTIMIZER_LIVE = False


def _run_pipeline(portfolio_df: pd.DataFrame, risk_bucket: str):
    """Run live pipeline if available; otherwise load contract mocks."""
    tickers = portfolio_df["ticker"].tolist()
    # weights column is already normalized to sum to 1.0
    curr_weights = portfolio_df.set_index("ticker")["weight"]

    # ── Data pipeline (Person A) ─────────────────────────────────────────────
    if DATA_LIVE:
        with st.spinner("Fetching price history and options data..."):
            returns_df = fetch_returns(tickers)
            spot_prices = {t: _get_spot(t) for t in tickers}
            iv_df = fetch_iv(tickers, spot_prices)
            cov_df_raw, iv_df = compute_covariance(returns_df, iv_df)
            # Person A returns cov_df with 'ticker' as a column; Person B expects it as index
            cov_df = cov_df_raw.set_index("ticker")
            exp_ret = (
                compute_expected_returns(returns_df)
                .set_index("ticker")["expected_return_annual"]
            )
    else:
        iv_df = pd.read_csv(CONTRACTS / "iv.csv")
        cov_df = pd.read_csv(CONTRACTS / "covariance.csv", index_col=0)
        exp_ret = (
            pd.read_csv(CONTRACTS / "expected_returns.csv")
            .set_index("ticker")["expected_return_annual"]
        )

    # ── Optimizer (Person B) ─────────────────────────────────────────────────
    if OPTIMIZER_LIVE:
        # Resolve target_vol and position cap from the chosen bucket
        bucket_params = get_bucket_params(risk_bucket)
        target_vol = default_target_vol(risk_bucket)

        result_df = optimize(
            cov_df, exp_ret, curr_weights,
            target_vol=target_vol,
            max_position=bucket_params["max_position"],
        )
        frontier_df = generate_frontier(
            cov_df, exp_ret,
            vol_max=bucket_params["vol_max"],
            max_position=bucket_params["max_position"],
        )
        opt_weights = result_df.set_index("ticker")["optimized_weight"]
        metrics_df = compute_metrics(cov_df, exp_ret, curr_weights, opt_weights, target_vol)
        weights_df = result_df
    else:
        # Mock mode: patch current_weight from user input; results won't vary by bucket
        weights_df = pd.read_csv(CONTRACTS / "optimized_weights.csv")
        weights_df = weights_df[weights_df["ticker"].isin(tickers)].copy()
        weights_df["current_weight"] = weights_df["ticker"].map(curr_weights)
        frontier_df = pd.read_csv(CONTRACTS / "frontier.csv")
        metrics_df = pd.read_csv(CONTRACTS / "metrics.csv")
        metrics_df["risk_bucket"] = risk_bucket

    return weights_df, frontier_df, metrics_df, iv_df


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Portfolio Input")

    risk_bucket = st.selectbox(
        "Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
        help="Conservative ≤ 20% vol · Moderate 20–28% · Aggressive 28–40%",
    )

    st.divider()
    st.subheader("Holdings")
    st.caption("Enter each stock's allocation as a percentage. Rows sum should equal 100%.")

    default_portfolio = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "NVDA", "GOOGL", "XOM"],
        "weight_%": [22.0, 26.8, 41.6, 3.3, 6.3],
    })

    edited_df = st.data_editor(
        default_portfolio,
        num_rows="dynamic",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "weight_%": st.column_config.NumberColumn(
                "Weight (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.1f",
                width="small",
            ),
        },
        use_container_width=True,
        hide_index=True,
        key="portfolio_editor",
    )

    portfolio_df = edited_df.dropna(subset=["ticker"]).copy()
    portfolio_df["ticker"] = portfolio_df["ticker"].str.upper().str.strip()
    portfolio_df = portfolio_df[portfolio_df["ticker"] != ""]
    portfolio_df["weight_%"] = pd.to_numeric(portfolio_df["weight_%"], errors="coerce").fillna(0.0)

    if len(portfolio_df) < 2:
        st.error("Add at least 2 tickers.")
        st.stop()
    if len(portfolio_df) > 10:
        st.error("Maximum 10 tickers.")
        st.stop()

    total_pct = portfolio_df["weight_%"].sum()
    if abs(total_pct - 100.0) < 0.5:
        st.success(f"Total: {total_pct:.1f}%")
    else:
        st.warning(f"Total: {total_pct:.1f}% — weights should sum to 100%.")

    # Normalize to exactly 1.0 for the optimizer
    if total_pct < 1e-6:
        st.error("All weights are zero.")
        st.stop()
    portfolio_df["weight"] = portfolio_df["weight_%"] / total_pct

    run_btn = st.button("Optimize Portfolio", type="primary", use_container_width=True)
    st.caption("Results update automatically when you change weights.")

    st.divider()
    mode_parts = []
    if not DATA_LIVE:
        mode_parts.append("data")
    if not OPTIMIZER_LIVE:
        mode_parts.append("optimizer")
    if mode_parts:
        st.info(f"Mock mode ({', '.join(mode_parts)}) — using contracts/*.csv", icon="🧪")

# ── Main content ─────────────────────────────────────────────────────────────
st.title("Smart Portfolio Optimizer")
st.caption("CDS Datathon 2026 — Optimize your allocation based on risk tolerance and forward-looking volatility.")

# Rerun whenever portfolio or risk bucket changes (not just on button click)
portfolio_key = (
    tuple(zip(portfolio_df["ticker"], portfolio_df["weight_%"].round(1))),
    risk_bucket,
)
if run_btn or st.session_state.get("portfolio_key") != portfolio_key:
    st.session_state.portfolio_key = portfolio_key
    try:
        st.session_state.results = _run_pipeline(portfolio_df, risk_bucket)
    except Exception as e:
        st.error(f"Optimizer error: {e}")
        st.stop()

weights_df, frontier_df, metrics_df, iv_df = st.session_state.results
row = metrics_df.iloc[0]

# ── Headline metrics ─────────────────────────────────────────────────────────
st.subheader("Results")
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Vol Reduction",
    f"{row['vol_reduction_pct']:.1f}%",
    delta=f"{row['current_vol_annual']:.1%} → {row['optimized_vol_annual']:.1%}",
    delta_color="inverse",
)
col2.metric(
    "Return Change",
    f"{row['return_improvement_pct']:+.1f}%",
    delta=f"{row['current_return_annual']:.1%} → {row['optimized_return_annual']:.1%}",
)
col3.metric("Sharpe (Current)", f"{row['current_sharpe']:.2f}")
col4.metric(
    "Sharpe (Optimized)",
    f"{row['optimized_sharpe']:.2f}",
    delta=f"{row['optimized_sharpe'] - row['current_sharpe']:+.2f}",
)

# Bucket vol warning — use BUCKET_PARAMS if optimizer live, otherwise read from dict
if OPTIMIZER_LIVE:
    bucket_vol_max = get_bucket_params(risk_bucket)["vol_max"]
else:
    bucket_vol_max = BUCKET_PARAMS.get(risk_bucket, {}).get("vol_max", 0.40)

if row["current_vol_annual"] > bucket_vol_max:
    st.warning(
        f"Your current portfolio exceeds the risk target for **{risk_bucket}** "
        f"(current vol: {row['current_vol_annual']:.1%}, bucket max: {bucket_vol_max:.0%})."
    )

# IV fallback warning
fallback_tickers = iv_df[iv_df["data_source"] == "historical_fallback"]["ticker"].tolist()
if fallback_tickers:
    st.warning(
        f"Implied volatility unavailable for **{', '.join(fallback_tickers)}** — "
        "using historical volatility as fallback."
    )

st.divider()

# ── Charts ───────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.plotly_chart(
        plot_efficient_frontier(frontier_df, metrics_df),
        use_container_width=True,
    )

with col_right:
    st.plotly_chart(
        plot_allocation_comparison(weights_df),
        use_container_width=True,
    )

st.plotly_chart(plot_metrics_table(metrics_df), use_container_width=True)

# ── Optimal allocation table ─────────────────────────────────────────────────
st.subheader("Recommended Allocation")
alloc_table = weights_df[["ticker", "current_weight", "optimized_weight"]].copy()
alloc_table["current_%"] = (alloc_table["current_weight"] * 100).round(1)
alloc_table["optimized_%"] = (alloc_table["optimized_weight"] * 100).round(1)
alloc_table["change (pp)"] = (alloc_table["optimized_%"] - alloc_table["current_%"]).round(1)
alloc_table = alloc_table[["ticker", "current_%", "optimized_%", "change (pp)"]].rename(columns={
    "ticker": "Ticker",
    "current_%": "Current (%)",
    "optimized_%": "Optimized (%)",
})
st.dataframe(
    alloc_table.style.map(
        lambda v: "color: #2ECC71" if v > 0 else ("color: #E74C3C" if v < 0 else ""),
        subset=["change (pp)"],
    ),
    hide_index=True,
    use_container_width=True,
)

# ── IV detail ────────────────────────────────────────────────────────────────
with st.expander("Implied Volatility Detail"):
    display_iv = iv_df.copy()
    display_iv["iv_annualized"] = display_iv["iv_annualized"].map("{:.1%}".format)
    display_iv.columns = ["Ticker", "IV (Annualized)", "Expiry Date", "Strike", "Data Source"]
    st.dataframe(display_iv, hide_index=True, use_container_width=True)