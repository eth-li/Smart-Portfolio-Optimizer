"""Smart Portfolio Optimizer — Streamlit entry point (Person C)"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from app.charts import plot_efficient_frontier, plot_allocation_comparison, plot_metrics_table
from data.universe import UNIVERSE as CANDIDATE_UNIVERSE
from optimizer.solve import black_litterman_mu

try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

CONTRACTS = Path("contracts")

TARGET_VOL = {
    "Low":         0.18,
    "Medium-Low":  0.22,
    "Medium":      0.27,
    "Medium-High": 0.33,
    "High":        0.42,
}
FRONTIER_VOL_MAX_MULTIPLIER = 1.6

# Max new stocks to recommend when "Recommend new stocks" is checked.
MAX_NEW_STOCKS = 4

# ── Optional live imports ─────────────────────────────────────────────────────
try:
    from data.fetch import fetch_returns, fetch_iv, _get_spot, fetch_market_cap_weights
    _MARKET_CAP_AVAILABLE = True
    DATA_LIVE = True
except ImportError:
    try:
        from data.fetch import fetch_returns, fetch_iv, _get_spot
    except ImportError:
        pass
    _MARKET_CAP_AVAILABLE = False
    DATA_LIVE = False

try:
    from data.buckets import get_bucket_params, classify_vol, dynamic_max_position
    from data.risk import compute_covariance
    from optimizer.solve import optimize
    from optimizer.frontier import generate_frontier
    from optimizer.metrics import compute_metrics
    OPTIMIZER_LIVE = True
except ImportError:
    OPTIMIZER_LIVE = False

# ── Tooltips ──────────────────────────────────────────────────────────────────
_RISK_HELP = (
    "Risk measures how much your portfolio's value could go up or down in a year.\n\n"
    "We measure risk using volatility — a 20% risk level means your portfolio "
    "could realistically swing ±20% in a year under normal market conditions.\n\n"
    "The S&P 500 has averaged ~16% risk historically. Lower = more stable, "
    "higher = bigger potential gains but also bigger potential drops."
)
_RETURN_HELP = (
    "Expected return is the annual gain the optimizer projects for this portfolio, "
    "based on each stock's historical returns blended with market-wide signals "
    "(Black-Litterman model). This is a forward-looking estimate, not a guarantee."
)


# ── LLM explanation ───────────────────────────────────────────────────────────

def _llm_explanation(
    weights_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    risk_bucket: str,
    allow_new_stocks: bool,
    cov_df: pd.DataFrame,
) -> str:
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
        ch  = weights_df.copy()
        ch["delta"] = ch["optimized_weight"] - ch["current_weight"]

        trimmed = ch[ch["delta"] < -0.01].sort_values("delta")
        boosted = ch[(ch["delta"] > 0.01) & (ch["current_weight"] >= 0.01)].sort_values("delta", ascending=False)
        new_pos = ch[(ch["delta"] > 0.01) & (ch["current_weight"] < 0.01)].sort_values("delta", ascending=False)

        def fmt_rows(df):
            return ", ".join(
                f"{r['ticker']} {r['current_weight']:.0%}→{r['optimized_weight']:.0%}"
                for _, r in df.iterrows()
            ) or "none"

        risk_direction = "decreased" if row["vol_reduction_pct"] >= 0 else "increased"

        tickers_stock_vol = {
            t: float(np.sqrt(max(cov_df.loc[t, t], 0)))
            for t in weights_df["ticker"].tolist()
            if t in cov_df.index
        }

        boosted_tickers = boosted["ticker"].tolist()
        corr_notes = []
        if boosted_tickers and len(cov_df) > 1:
            for bt in boosted_tickers[:2]:
                if bt in cov_df.index:
                    others = [t for t in weights_df["ticker"] if t != bt and t in cov_df.index]
                    if others:
                        avg_corr = np.mean([
                            cov_df.loc[bt, o] / max(
                                np.sqrt(cov_df.loc[bt, bt]) * np.sqrt(cov_df.loc[o, o]), 1e-8
                            )
                            for o in others
                        ])
                        corr_notes.append(f"{bt} (avg correlation {avg_corr:.2f} with the rest)")

        new_pos_detail = "; ".join(
            f"{r['ticker']} at {r['optimized_weight']:.0%} (individual risk: {tickers_stock_vol.get(r['ticker'], 0):.0%})"
            for _, r in new_pos.iterrows()
        ) or "none"

        prompt = f"""You are explaining portfolio optimization to a non-expert investor. Write exactly 3 sentences. Be specific, plain, and helpful — no jargon.

PORTFOLIO DATA:
- Risk level chosen: {risk_bucket}
- Portfolio risk: {row['current_vol_annual']:.1%} → {row['optimized_vol_annual']:.1%} ({risk_direction})
- Expected return: {row['current_return_annual']:.1%} → {row['optimized_return_annual']:.1%}
- Trimmed positions: {fmt_rows(trimmed)}
- Increased/existing positions: {fmt_rows(boosted)}
- New positions added: {new_pos_detail}
- Correlation context for key changes: {'; '.join(corr_notes) if corr_notes else 'not available'}

SENTENCE 1 — What changed: Name the specific stocks that were trimmed or increased and by how much.
SENTENCE 2 — Why those stocks: Give a qualitative reason for 1–2 key changes. Use the correlation data if available. Call it "risk", not "volatility".
SENTENCE 3 — What it means: Use "your portfolio could swing roughly ±{row['optimized_vol_annual']*2:.0%} in a year". Call changes "risk", not "volatility".

Max 80 words total. No headers. Bold 2–3 key numbers with **."""

        client = _anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
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
    row      = metrics_df.iloc[0]
    curr_vol = row["current_vol_annual"]
    opt_vol  = row["optimized_vol_annual"]

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
    s2 = "Stocks were reweighted to reduce how much your holdings move together, spreading risk more evenly."
    swing = opt_vol * 2 * 100
    if curr_vol > opt_vol:
        s3 = f"This brings risk from **{curr_vol:.0%} → {opt_vol:.0%}** — your portfolio could swing roughly **±{swing:.0f}%** in a year."
    else:
        s3 = f"At **{risk_bucket}** risk, the optimized level is **{opt_vol:.0%}** — your portfolio could swing roughly **±{swing:.0f}%** in a year."

    return s1 + "  \n" + s2 + "  \n" + s3


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading market data...")
def _fetch_data(tickers_tuple: tuple) -> tuple:
    """Returns (cov_df, iv_df, mkt_caps). All three cached together."""
    tickers      = list(tickers_tuple)
    cov_universe = pd.read_csv(CONTRACTS / "covariance.csv", index_col=0)
    iv_universe  = pd.read_csv(CONTRACTS / "iv.csv")

    known   = [t for t in tickers if t in cov_universe.index]
    unknown = [t for t in tickers if t not in cov_universe.index]

    if not unknown:
        cov_df   = cov_universe.loc[tickers, tickers]
        iv_df    = iv_universe[iv_universe["ticker"].isin(tickers)].reset_index(drop=True)
        mkt_caps = _fetch_market_caps(tickers)
        return cov_df, iv_df, mkt_caps

    if not DATA_LIVE:
        st.warning(f"Unknown tickers (not in database): {unknown} — they will be ignored.")
        cov_df   = cov_universe.loc[known, known]
        iv_df    = iv_universe[iv_universe["ticker"].isin(known)].reset_index(drop=True)
        mkt_caps = _fetch_market_caps(known)
        return cov_df, iv_df, mkt_caps

    with st.spinner(f"Fetching data for new tickers: {unknown} — this may take ~30s..."):
        returns_df  = fetch_returns(tickers)
        spot_prices = {t: _get_spot(t) for t in tickers}
        iv_df_raw   = fetch_iv(tickers, spot_prices)
        cov_df_raw, iv_df = compute_covariance(returns_df, iv_df_raw)
        cov_df      = cov_df_raw.set_index("ticker")
        mkt_caps    = _fetch_market_caps(tickers)

    return cov_df, iv_df, mkt_caps


def _fetch_market_caps(tickers):
    if not _MARKET_CAP_AVAILABLE:
        return None
    try:
        return fetch_market_cap_weights(tickers)
    except Exception:
        return None


def _top_new_candidates(user_tickers: list[str], n: int) -> list[str]:
    """
    Pick the top-n candidate tickers from CANDIDATE_UNIVERSE (excluding
    what the user already holds) ranked by shrunk+winsorized expected return.
    This keeps the optimizer universe small and the pie chart readable.
    """
    all_candidates = [t for t in CANDIDATE_UNIVERSE if t not in user_tickers]
    try:
        returns_raw = pd.read_csv(CONTRACTS / "returns.csv", index_col="date")
        er          = returns_raw.mean() * 252
        er_mean     = er.mean()
        er_shrunk   = (1 - 0.4) * er + 0.4 * er_mean
        er_shrunk   = er_shrunk.clip(
            lower=er_shrunk.mean() - 2 * er_shrunk.std(),
            upper=er_shrunk.mean() + 2 * er_shrunk.std(),
        )
        ranked = er_shrunk.reindex(all_candidates).dropna().nlargest(n)
        return ranked.index.tolist()
    except Exception:
        # Fallback: just take the first n candidates alphabetically
        return sorted(all_candidates)[:n]


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(portfolio_df: pd.DataFrame, risk_bucket: str, allow_new_stocks: bool):
    user_tickers = portfolio_df["ticker"].tolist()
    user_weights = portfolio_df.set_index("ticker")["weight"]

    # ── Step 1: Build ticker list ─────────────────────────────────────────────
    # New-stocks mode: pre-filter to top MAX_NEW_STOCKS candidates by expected
    # return so the optimizer sees a small curated list, not all 117 tickers.
    if allow_new_stocks:
        new_candidates = _top_new_candidates(user_tickers, MAX_NEW_STOCKS)
        all_tickers    = user_tickers + new_candidates
    else:
        all_tickers = user_tickers

    curr_weights = pd.Series(0.0, index=all_tickers)
    curr_weights[user_tickers] = user_weights.values

    # ── Step 2: Fetch covariance, IV, and market caps (all cached together) ───
    cov_df, iv_df, mkt_caps = _fetch_data(tuple(all_tickers))

    # ── Step 3: Expected returns — live from returns.csv ─────────────────────
    _returns_raw = pd.read_csv(CONTRACTS / "returns.csv", index_col="date")
    _er_csv      = _returns_raw.mean() * 252
    _er_mean     = _er_csv.mean()
    _er_shrunk   = (1 - 0.4) * _er_csv + 0.4 * _er_mean
    exp_ret_hist = _er_shrunk.reindex(all_tickers).fillna(_er_mean)

    # ── Step 4: Winsorize at mean ± 2σ before BL ─────────────────────────────
    _hist_mean   = exp_ret_hist.mean()
    _hist_std    = exp_ret_hist.std()
    exp_ret_hist = exp_ret_hist.clip(
        lower=_hist_mean - 2.0 * _hist_std,
        upper=_hist_mean + 2.0 * _hist_std,
    )

    # ── Step 5: BL mu — computed once, shared everywhere ─────────────────────
    mkt_w_arr = None
    if mkt_caps is not None:
        mkt_w_arr = mkt_caps.reindex(all_tickers).fillna(0).values.astype(float)

    cov_np     = cov_df.loc[all_tickers, all_tickers].values.astype(float)
    bl_mu_arr  = black_litterman_mu(cov_np, exp_ret_hist[all_tickers].values, market_cap_weights=mkt_w_arr)
    bl_exp_ret = pd.Series(bl_mu_arr, index=all_tickers)

    target_vol       = TARGET_VOL[risk_bucket]
    frontier_vol_max = target_vol * FRONTIER_VOL_MAX_MULTIPLIER
    n_user           = len(user_tickers)
    max_pos          = dynamic_max_position(risk_bucket, n_user)

    # ── Step 6: Optimize ──────────────────────────────────────────────────────
    if OPTIMIZER_LIVE:
        if allow_new_stocks:
            div_factor = 0.0
            min_w = np.zeros(len(all_tickers))
            for i, t in enumerate(all_tickers):
                if t in user_tickers:
                    min_w[i] = max(float(curr_weights[t]) * 0.4, 0.01)
            if min_w.sum() >= 0.95:
                min_w *= 0.9 / min_w.sum()
        else:
            div_factor = 0.2
            min_w      = None

        result_df = optimize(
            cov_df, curr_weights,
            target_vol=target_vol,
            max_position=max_pos,
            diversity_factor=div_factor,
            exp_ret=bl_exp_ret,
            min_weights=min_w,
        )
        frontier_df = generate_frontier(
            cov_df,
            vol_max=frontier_vol_max,
            max_position=max_pos,
            exp_ret=bl_exp_ret,
        )
        opt_weights = result_df.set_index("ticker")["optimized_weight"]
        metrics_df  = compute_metrics(cov_df, curr_weights, opt_weights, target_vol)
        weights_df  = result_df
    else:
        weights_df = pd.read_csv(CONTRACTS / "optimized_weights.csv")
        weights_df = weights_df[weights_df["ticker"].isin(all_tickers)].copy()
        weights_df["current_weight"] = weights_df["ticker"].map(curr_weights).fillna(0.0)
        frontier_df = pd.read_csv(CONTRACTS / "frontier.csv")
        metrics_df  = pd.read_csv(CONTRACTS / "metrics.csv")
        metrics_df["risk_bucket"] = risk_bucket
        opt_weights = weights_df.set_index("ticker")["optimized_weight"]
        # mkt_caps already defined from _fetch_data above — safe to use here
        cov_df, _, _ = _fetch_data(tuple(all_tickers))

    metrics_df["current_return_annual"]   = round(float(bl_exp_ret.reindex(curr_weights.index).fillna(0) @ curr_weights), 4)
    metrics_df["optimized_return_annual"] = round(float(bl_exp_ret.reindex(opt_weights.index).fillna(0) @ opt_weights), 4)

    weights_df = weights_df[
        (weights_df["current_weight"] > 0) | (weights_df["optimized_weight"] > 0)
    ].reset_index(drop=True)

    return weights_df, frontier_df, metrics_df, iv_df, cov_df


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Portfolio Input")

    risk_bucket = st.selectbox(
        "Risk Tolerance",
        ["Low", "Medium-Low", "Medium", "Medium-High", "High"],
        index=2,
        help=(
            "Low ≈ 18% risk  ·  Medium-Low ≈ 22%  ·  "
            "Medium ≈ 27%  ·  Medium-High ≈ 33%  ·  High ≈ 42%\n\n"
            "S&P 500 baseline ≈ 16%."
        ),
    )

    allow_new_stocks = st.checkbox(
        "Recommend new stocks",
        value=False,
        help=f"When checked, the optimizer can suggest up to {MAX_NEW_STOCKS} new stocks to add to your portfolio.",
    )

    st.divider()
    st.subheader("Your Current Holdings")
    st.caption("Enter your tickers and current allocation. Weights should sum to 100%.")

    if "default_portfolio" not in st.session_state:
        _rng     = np.random.default_rng()
        _n       = int(_rng.integers(4, 8))
        _tickers = _rng.choice(CANDIDATE_UNIVERSE, size=_n, replace=False).tolist()
        _w       = np.round(_rng.dirichlet(np.ones(_n)) * 100, 1)
        _w[0]   += round(100.0 - _w.sum(), 1)
        st.session_state.default_portfolio = pd.DataFrame({
            "ticker":   _tickers,
            "weight_%": _w.tolist(),
        })

    edited_df = st.data_editor(
        st.session_state.default_portfolio,
        num_rows="dynamic",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "weight_%": st.column_config.NumberColumn(
                "Weight (%)", min_value=0.0, max_value=100.0,
                step=0.1, format="%.1f", width="small",
            ),
        },
        use_container_width=True,
        hide_index=True,
        key="portfolio_editor",
    )

    portfolio_df = edited_df.dropna(subset=["ticker"]).copy()
    portfolio_df["ticker"]   = portfolio_df["ticker"].str.upper().str.strip()
    portfolio_df             = portfolio_df[portfolio_df["ticker"] != ""]
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

    if OPTIMIZER_LIVE:
        _n_user  = len(portfolio_df)
        _max_pos = dynamic_max_position(risk_bucket, _n_user)
        st.caption(f"Max single position: **{_max_pos:.0%}** ({_n_user} stocks, {risk_bucket} risk)")

    run_btn = st.button("Optimize Portfolio", type="primary", use_container_width=True)

    st.divider()
    mode_parts = []
    if not DATA_LIVE:
        mode_parts.append("data")
    if not OPTIMIZER_LIVE:
        mode_parts.append("optimizer")
    if mode_parts:
        st.info(f"Mock mode ({', '.join(mode_parts)}) — using contracts/*.csv", icon="🧪")

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    try:
        st.session_state.results     = _run_pipeline(portfolio_df, risk_bucket, allow_new_stocks)
        st.session_state.run_context = {"risk_bucket": risk_bucket, "allow_new_stocks": allow_new_stocks}
        _w, _, _m, _, _c             = st.session_state.results
        st.session_state.explanation = _llm_explanation(_w, _m, risk_bucket, allow_new_stocks, _c)
    except Exception as e:
        st.error(f"Optimizer error: {e}")
        st.stop()

# ── Main content ──────────────────────────────────────────────────────────────
st.title("Smart Portfolio Optimizer")
st.caption("CDS Datathon 2026 — Optimize your allocation based on risk tolerance and forward-looking expected returns.")

if "results" not in st.session_state:
    st.info("Enter your holdings in the sidebar and click **Optimize Portfolio** to get started.")
    st.stop()

weights_df, frontier_df, metrics_df, iv_df, cov_df = st.session_state.results
row = metrics_df.iloc[0]

# ── AI explanation ────────────────────────────────────────────────────────────
if "explanation" in st.session_state:
    st.info(st.session_state.explanation)

# ── Headline metrics ──────────────────────────────────────────────────────────
st.subheader("Results")
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Current Risk",
    f"{row['current_vol_annual']:.1%}",
    delta=row.get("current_bucket", ""),
    delta_color="off",
    help=_RISK_HELP,
)
col2.metric(
    "Optimized Risk",
    f"{row['optimized_vol_annual']:.1%}",
    delta=row.get("optimized_bucket", ""),
    delta_color="off",
    help=_RISK_HELP,
)
col3.metric(
    "Current Expected Return",
    f"{row['current_return_annual']:.1%}",
    help=_RETURN_HELP,
)
col4.metric(
    "Optimized Expected Return",
    f"{row['optimized_return_annual']:.1%}",
    delta=f"{row['optimized_return_annual'] - row['current_return_annual']:+.1%}",
    delta_color="normal",
    help=_RETURN_HELP,
)

fallback_tickers = iv_df[iv_df["data_source"] == "historical_fallback"]["ticker"].tolist()
if fallback_tickers:
    st.warning(
        f"Implied volatility unavailable for **{', '.join(fallback_tickers)}** — "
        "using historical volatility as fallback."
    )

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 3])

with col_left:
    st.plotly_chart(plot_efficient_frontier(frontier_df, metrics_df), use_container_width=True)

with col_right:
    st.plotly_chart(plot_allocation_comparison(weights_df), use_container_width=True)

st.plotly_chart(plot_metrics_table(metrics_df), use_container_width=True)

# ── Recommended Allocation ────────────────────────────────────────────────────
st.subheader("Recommended Allocation")
alloc_table = weights_df[["ticker", "current_weight", "optimized_weight"]].copy()
alloc_table["current_%"]   = (alloc_table["current_weight"]   * 100).round(2)
alloc_table["optimized_%"] = (alloc_table["optimized_weight"] * 100).round(2)
alloc_table["change (pp)"] = (alloc_table["optimized_%"] - alloc_table["current_%"]).round(2)
alloc_table = alloc_table[["ticker", "current_%", "optimized_%", "change (pp)"]].rename(columns={
    "ticker":      "Ticker",
    "current_%":   "Current (%)",
    "optimized_%": "Optimized (%)",
})
st.dataframe(
    alloc_table.style
        .format({
            "Current (%)":   "{:.2f}%",
            "Optimized (%)": "{:.2f}%",
            "change (pp)":   "{:+.2f} pp",
        })
        .map(
            lambda v: "color: #2ECC71" if v > 0 else ("color: #E74C3C" if v < 0 else ""),
            subset=["change (pp)"],
        ),
    hide_index=True,
    use_container_width=True,
)

# ── IV detail ─────────────────────────────────────────────────────────────────
with st.expander("Implied Volatility Detail"):
    shown_tickers = set(weights_df["ticker"].tolist())
    display_iv    = iv_df[iv_df["ticker"].isin(shown_tickers)].copy()
    display_iv["iv_annualized"] = display_iv["iv_annualized"].map("{:.1%}".format)
    display_iv.columns = ["Ticker", "IV (Annualized)", "Expiry Date", "Strike", "Data Source"]
    st.dataframe(display_iv, hide_index=True, use_container_width=True)