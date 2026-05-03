"""Smart Portfolio Optimizer — Streamlit entry point (Person C)"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from app.charts import plot_efficient_frontier, plot_allocation_comparison, plot_metrics_table
from data.universe import UNIVERSE as CANDIDATE_UNIVERSE

try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

CONTRACTS = Path("contracts")

# ── Optional live imports (graceful fallback to contract mocks) ──────────────
try:
    from data.fetch import fetch_returns, fetch_iv, _get_spot
    from data.risk import compute_covariance
    DATA_LIVE = True
except ImportError:
    DATA_LIVE = False

try:
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


def _llm_explanation(
    weights_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    risk_bucket: str,
    allow_new_stocks: bool,
    cov_df: pd.DataFrame,
) -> str:
    """
    Generate a plain-English explanation via Claude API.
    Falls back to a rule-based summary if no API key is configured.
    Set ANTHROPIC_API_KEY environment variable (or .streamlit/secrets.toml) to enable.
    """
    if not _ANTHROPIC_AVAILABLE:
        return _fallback_explanation(weights_df, metrics_df, risk_bucket)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        except Exception:
            pass

    if not api_key:
        return _fallback_explanation(weights_df, metrics_df, risk_bucket)

    try:
        row = metrics_df.iloc[0]
        ch = weights_df.copy()
        ch["delta"] = ch["optimized_weight"] - ch["current_weight"]

        trimmed  = ch[ch["delta"] < -0.01].sort_values("delta")
        boosted  = ch[(ch["delta"] > 0.01) & (ch["current_weight"] >= 0.01)].sort_values("delta", ascending=False)
        new_pos  = ch[(ch["delta"] > 0.01) & (ch["current_weight"] < 0.01)].sort_values("delta", ascending=False)
        no_change = ch[ch["delta"].abs() <= 0.01]

        def fmt_rows(df):
            return ", ".join(
                f"{r['ticker']} {r['current_weight']:.0%}→{r['optimized_weight']:.0%}"
                for _, r in df.iterrows()
            ) or "none"

        vol_direction = "decreased" if row["vol_reduction_pct"] >= 0 else "increased"
        vol_change_abs = abs(row["optimized_vol_annual"] - row["current_vol_annual"])

        tickers_stock_vol = {
            t: float(np.sqrt(max(cov_df.loc[t, t], 0)))
            for t in weights_df["ticker"].tolist()
            if t in cov_df.index
        }
        new_pos_detail = "; ".join(
            f"{r['ticker']} at {r['optimized_weight']:.0%} (individual vol: {tickers_stock_vol.get(r['ticker'], 0):.0%})"
            for _, r in new_pos.iterrows()
        ) or "none"

        prompt = f"""You are explaining portfolio optimization to a non-expert in exactly 2 sentences. Be specific and plain.

DATA:
- Risk level: {risk_bucket}
- Vol: {row['current_vol_annual']:.1%} → {row['optimized_vol_annual']:.1%} ({vol_direction})
- Expected return: {row['current_return_annual']:.1%} → {row['optimized_return_annual']:.1%}
- Trimmed: {fmt_rows(trimmed)}
- Increased/added: {fmt_rows(boosted)}{(' | new: ' + new_pos_detail) if new_pos_detail != 'none' else ''}

Sentence 1: What specific stocks changed and in which direction (name them).
Sentence 2: Why it helps and what the new {row['optimized_vol_annual']:.0%} volatility means in plain English (use the "could swing ±{row['optimized_vol_annual']*2:.0%} in a year" framing).

Max 60 words total. No headers. Bold key numbers with **."""

        client = _anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    except Exception:
        return _fallback_explanation(weights_df, metrics_df, risk_bucket)


def _fallback_explanation(
    weights_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    risk_bucket: str,
) -> str:
    """Rule-based explanation used when the Claude API is not configured."""
    row = metrics_df.iloc[0]
    curr_vol = row["current_vol_annual"]
    opt_vol  = row["optimized_vol_annual"]
    curr_ret = row["current_return_annual"]
    opt_ret  = row["optimized_return_annual"]

    ch = weights_df.copy()
    ch["delta"] = ch["optimized_weight"] - ch["current_weight"]
    trimmed = ch[ch["delta"] < -0.02].sort_values("delta")
    boosted = ch[ch["delta"] > 0.02].sort_values("delta", ascending=False)

    move_parts = []
    for _, r in trimmed.iterrows():
        move_parts.append(f"**{r['ticker']}** {r['current_weight']:.0%}→{r['optimized_weight']:.0%}")
    for _, r in boosted.iterrows():
        label = "added" if r["current_weight"] < 0.01 else "raised"
        move_parts.append(f"**{r['ticker']}** {label} to {r['optimized_weight']:.0%}")

    s1 = ("; ".join(move_parts) + ".") if move_parts else "Only minor rebalancing — portfolio was already near-optimal."

    swing = opt_vol * 2 * 100
    if curr_vol > opt_vol:
        s2 = (f"Spreading weight reduces correlated risk, bringing volatility from **{curr_vol:.0%} → {opt_vol:.0%}** — "
              f"meaning your portfolio could swing roughly **±{swing:.0f}%** in a year.")
    else:
        s2 = (f"At **{risk_bucket}** risk, higher vol (**{opt_vol:.0%}**) targets better expected return "
              f"(**{opt_ret:.0%}**) — your portfolio could swing roughly **±{swing:.0f}%** in a year.")

    return s1 + "  \n" + s2


@st.cache_data(show_spinner="Loading market data...")
def _fetch_data(tickers_tuple: tuple):
    """
    Load covariance + IV for the requested tickers.
    Uses pre-computed contracts/ files for known tickers (instant).
    Falls back to live yfinance fetch only for unknown tickers.
    """
    tickers = list(tickers_tuple)
    cov_universe = pd.read_csv(CONTRACTS / "covariance.csv", index_col=0)
    iv_universe  = pd.read_csv(CONTRACTS / "iv.csv")

    known   = [t for t in tickers if t in cov_universe.index]
    unknown = [t for t in tickers if t not in cov_universe.index]

    if not unknown:
        # Fast path: all tickers pre-computed
        cov_df = cov_universe.loc[tickers, tickers]
        iv_df  = iv_universe[iv_universe["ticker"].isin(tickers)].reset_index(drop=True)
        return cov_df, iv_df

    if not DATA_LIVE:
        st.warning(f"Unknown tickers (not in database): {unknown} — they will be ignored.")
        cov_df = cov_universe.loc[known, known]
        iv_df  = iv_universe[iv_universe["ticker"].isin(known)].reset_index(drop=True)
        return cov_df, iv_df

    # Slow path: fetch all tickers live (cross-correlations require joint estimation)
    with st.spinner(f"Fetching data for new tickers: {unknown} — this may take ~30s..."):
        returns_df = fetch_returns(tickers)
        spot_prices = {t: _get_spot(t) for t in tickers}
        iv_df_raw = fetch_iv(tickers, spot_prices)
        cov_df_raw, iv_df = compute_covariance(returns_df, iv_df_raw)
        cov_df = cov_df_raw.set_index("ticker")
    return cov_df, iv_df


def _run_pipeline(portfolio_df: pd.DataFrame, risk_bucket: str, allow_new_stocks: bool):
    user_tickers = portfolio_df["ticker"].tolist()
    user_weights = portfolio_df.set_index("ticker")["weight"]

    # When allow_new_stocks is on, expand the optimizer universe with candidates
    # the user doesn't currently hold. New stocks start at 0% current weight.
    if allow_new_stocks:
        new_candidates = [t for t in CANDIDATE_UNIVERSE if t not in user_tickers]
        all_tickers = user_tickers + new_candidates
    else:
        all_tickers = user_tickers

    curr_weights = pd.Series(0.0, index=all_tickers)
    curr_weights[user_tickers] = user_weights.values  # new stocks stay at 0

    cov_df, iv_df = _fetch_data(tuple(all_tickers))

    # Expected returns from 5-year history, with James-Stein shrinkage toward the
    # cross-sectional mean. Shrinkage damps outlier years (e.g. NVDA's 2023 run)
    # without flipping relative rankings. α=0.4 blends 60% own history, 40% mean.
    _er_csv = pd.read_csv(CONTRACTS / "expected_returns.csv").set_index("ticker")["expected_return_annual"]
    _er_mean = _er_csv.mean()
    _er_shrunk = (1 - 0.4) * _er_csv + 0.4 * _er_mean
    exp_ret = _er_shrunk.reindex(all_tickers).fillna(_er_mean)

    if OPTIMIZER_LIVE:
        bucket_params = get_bucket_params(risk_bucket)
        target_vol = default_target_vol(risk_bucket)

        if allow_new_stocks:
            # No global min-weight floor — let optimizer pick from the universe.
            # But anchor existing holdings: each must retain ≥40% of current weight
            # so the optimizer can't completely ignore the user's portfolio.
            div_factor = 0.0
            min_w = np.zeros(len(all_tickers))
            for i, t in enumerate(all_tickers):
                if t in user_tickers:
                    min_w[i] = max(float(curr_weights[t]) * 0.4, 0.01)
            # Safety: if floors sum ≥ 1 the problem is infeasible — scale them down
            if min_w.sum() >= 0.95:
                min_w *= 0.9 / min_w.sum()
        else:
            div_factor = 0.2
            min_w = None

        result_df = optimize(
            cov_df, curr_weights,
            target_vol=target_vol,
            max_position=bucket_params["max_position"],
            diversity_factor=div_factor,
            exp_ret=exp_ret,
            min_weights=min_w,
        )
        frontier_df = generate_frontier(
            cov_df,
            vol_max=bucket_params["vol_max"],
            max_position=bucket_params["max_position"],
            exp_ret=exp_ret,
        )
        opt_weights = result_df.set_index("ticker")["optimized_weight"]
        metrics_df = compute_metrics(cov_df, curr_weights, opt_weights, target_vol)
        weights_df = result_df
    else:
        weights_df = pd.read_csv(CONTRACTS / "optimized_weights.csv")
        weights_df = weights_df[weights_df["ticker"].isin(all_tickers)].copy()
        weights_df["current_weight"] = weights_df["ticker"].map(curr_weights).fillna(0.0)
        frontier_df = pd.read_csv(CONTRACTS / "frontier.csv")
        metrics_df = pd.read_csv(CONTRACTS / "metrics.csv")
        metrics_df["risk_bucket"] = risk_bucket
        opt_weights = weights_df.set_index("ticker")["optimized_weight"]
        cov_df, _ = _fetch_data(tuple(all_tickers))

    # Add return columns (metrics.py is vol-only; compute returns here using exp_ret)
    metrics_df["current_return_annual"] = round(float(exp_ret.reindex(curr_weights.index).fillna(0) @ curr_weights), 4)
    metrics_df["optimized_return_annual"] = round(float(exp_ret.reindex(opt_weights.index).fillna(0) @ opt_weights), 4)

    # Drop tickers with zero weight in both columns — avoids 117-row tables when
    # allow_new_stocks adds the full universe but most candidates get zeroed out.
    weights_df = weights_df[
        (weights_df["current_weight"] > 0) | (weights_df["optimized_weight"] > 0)
    ].reset_index(drop=True)

    return weights_df, frontier_df, metrics_df, iv_df, cov_df


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
        ["Low", "Medium-Low", "Medium", "Medium-High", "High"],
        index=2,
        help="Low ≤ 15% vol · Medium-Low 15–22% · Medium 22–28% · Medium-High 28–35% · High 35–50%",
    )

    allow_new_stocks = st.checkbox(
        "Recommend new stocks",
        value=False,
        help="When checked, the optimizer can suggest adding stocks beyond your current holdings.",
    )

    st.divider()
    st.subheader("Your Current Holdings")
    st.caption("Enter your tickers and current allocation. Weights should sum to 100%.")

    if "default_portfolio" not in st.session_state:
        _rng = np.random.default_rng()
        _n = int(_rng.integers(4, 8))
        _tickers = _rng.choice(CANDIDATE_UNIVERSE, size=_n, replace=False).tolist()
        _w = np.round(_rng.dirichlet(np.ones(_n)) * 100, 1)
        _w[0] += round(100.0 - _w.sum(), 1)   # fix rounding so sum == 100
        st.session_state.default_portfolio = pd.DataFrame({
            "ticker": _tickers,
            "weight_%": _w.tolist(),
        })

    default_portfolio = st.session_state.default_portfolio

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

    total_pct = portfolio_df["weight_%"].sum()
    if abs(total_pct - 100.0) < 0.5:
        st.success(f"Total: {total_pct:.1f}%")
    else:
        st.warning(f"Total: {total_pct:.1f}% — should sum to 100%.")

    if total_pct < 1e-6:
        st.error("All weights are zero.")
        st.stop()
    portfolio_df["weight"] = portfolio_df["weight_%"] / total_pct

    run_btn = st.button("Optimize Portfolio", type="primary", use_container_width=True)

    st.divider()
    mode_parts = []
    if not DATA_LIVE:
        mode_parts.append("data")
    if not OPTIMIZER_LIVE:
        mode_parts.append("optimizer")
    if mode_parts:
        st.info(f"Mock mode ({', '.join(mode_parts)}) — using contracts/*.csv", icon="🧪")

# ── Run pipeline only on button click ────────────────────────────────────────
if run_btn:
    try:
        st.session_state.results = _run_pipeline(portfolio_df, risk_bucket, allow_new_stocks)
        st.session_state.run_context = {"risk_bucket": risk_bucket, "allow_new_stocks": allow_new_stocks}
        _w, _, _m, _, _c = st.session_state.results
        st.session_state.explanation = _llm_explanation(_w, _m, risk_bucket, allow_new_stocks, _c)
    except Exception as e:
        st.error(f"Optimizer error: {e}")
        st.stop()

# ── Main content ─────────────────────────────────────────────────────────────
st.title("Smart Portfolio Optimizer")
st.caption("CDS Datathon 2026 — Optimize your allocation based on risk tolerance and forward-looking volatility.")

if "results" not in st.session_state:
    st.info("Enter your holdings in the sidebar and click **Optimize Portfolio** to get started.")
    st.stop()

weights_df, frontier_df, metrics_df, iv_df, cov_df = st.session_state.results
row = metrics_df.iloc[0]

# ── Summary explanation (top) ─────────────────────────────────────────────────
if "explanation" in st.session_state:
    st.info(st.session_state.explanation)

# ── Headline metrics ─────────────────────────────────────────────────────────
st.subheader("Results")
col1, col2, col3, col4 = st.columns(4)

_vol_reduced = row["vol_reduction_pct"] >= 0
col1.metric(
    "Vol Reduced" if _vol_reduced else "Vol Increased",
    f"{abs(row['vol_reduction_pct']):.1f}%",
    delta=f"{row['current_vol_annual']:.1%} → {row['optimized_vol_annual']:.1%}",
    delta_color="off",
)
col2.metric(
    "Current Vol",
    f"{row['current_vol_annual']:.1%}",
    delta=row.get("current_bucket", ""),
    delta_color="off",
)
col3.metric(
    "Optimized Vol",
    f"{row['optimized_vol_annual']:.1%}",
    delta=row.get("optimized_bucket", ""),
    delta_color="off",
)
col4.metric(
    "Target Vol",
    f"{row['target_vol']:.1%}",
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
    shown_tickers = set(weights_df["ticker"].tolist())
    display_iv = iv_df[iv_df["ticker"].isin(shown_tickers)].copy()
    display_iv["iv_annualized"] = display_iv["iv_annualized"].map("{:.1%}".format)
    display_iv.columns = ["Ticker", "IV (Annualized)", "Expiry Date", "Strike", "Data Source"]
    st.dataframe(display_iv, hide_index=True, use_container_width=True)
