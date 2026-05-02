import plotly.graph_objects as go
import pandas as pd
import numpy as np


def _effective_n(weights: pd.Series) -> float:
    """
    Effective number of positions = 1 / sum(w_i^2).
    Ranges from 1 (fully concentrated) to n (equally weighted).
    """
    w = weights.values.astype(float)
    hhi = (w ** 2).sum()
    return 1.0 / hhi if hhi > 1e-10 else 1.0


def plot_efficient_frontier(frontier_df: pd.DataFrame, metrics_df: pd.DataFrame) -> go.Figure:
    """
    Efficient frontier: Annualized Vol (x) vs Effective Number of Positions (y).

    Since we use a minimum-variance optimizer (no return estimates), the y-axis
    shows portfolio diversification rather than expected return. Higher effective N
    = more diversified. As you allow more risk (higher vol target), the optimizer
    concentrates into fewer positions.
    """
    row = metrics_df.iloc[0]

    meta_cols = {"target_vol_annual", "realized_vol_annual"}
    ticker_cols = [c for c in frontier_df.columns if c not in meta_cols]

    weight_matrix = frontier_df[ticker_cols].values.astype(float)
    hhi = (weight_matrix ** 2).sum(axis=1)
    effective_n = np.where(hhi > 1e-10, 1.0 / hhi, 1.0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frontier_df["realized_vol_annual"],
        y=effective_n,
        mode="lines+markers",
        name="Min-Variance Frontier",
        line=dict(color="#4A90D9", width=2.5),
        marker=dict(size=6, color="#4A90D9"),
        hovertemplate="Vol: %{x:.1%}<br>Effective Positions: %{y:.1f}<extra></extra>",
    ))

    curr_vol = row["current_vol_annual"]
    opt_vol = row["optimized_vol_annual"]

    closest_curr = (frontier_df["realized_vol_annual"] - curr_vol).abs().idxmin()
    closest_opt = (frontier_df["realized_vol_annual"] - opt_vol).abs().idxmin()
    curr_eff_n = effective_n[closest_curr]
    opt_eff_n = effective_n[closest_opt]

    fig.add_trace(go.Scatter(
        x=[curr_vol],
        y=[curr_eff_n],
        mode="markers+text",
        name="Current Portfolio",
        marker=dict(size=16, color="#E74C3C", symbol="diamond"),
        text=["Current"],
        textposition="top right",
        hovertemplate="Current Portfolio<br>Vol: %{x:.1%}<br>Effective Positions: %{y:.1f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[opt_vol],
        y=[opt_eff_n],
        mode="markers+text",
        name="Optimized Portfolio",
        marker=dict(size=16, color="#2ECC71", symbol="star"),
        text=["Optimized"],
        textposition="top right",
        hovertemplate="Optimized Portfolio<br>Vol: %{x:.1%}<br>Effective Positions: %{y:.1f}<extra></extra>",
    ))

    fig.add_annotation(
        x=opt_vol, y=opt_eff_n,
        ax=curr_vol, ay=curr_eff_n,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
        arrowcolor="#27AE60",
    )

    fig.update_layout(
        title="Min-Variance Frontier: Risk vs Diversification",
        xaxis=dict(title="Annualized Volatility", tickformat=".0%"),
        yaxis=dict(title="Effective Number of Positions", tickformat=".1f"),
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        height=480,
        template="plotly_white",
        margin=dict(t=50, b=50, l=60, r=60),
    )
    return fig


def plot_allocation_comparison(weights_df: pd.DataFrame) -> go.Figure:
    max_w = max(weights_df["current_weight"].max(), weights_df["optimized_weight"].max())

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=weights_df["ticker"],
        y=weights_df["current_weight"],
        name="Current",
        marker_color="#E74C3C",
        text=[f"{w:.1%}" for w in weights_df["current_weight"]],
        textposition="outside",
        hovertemplate="%{x}<br>Current: %{y:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=weights_df["ticker"],
        y=weights_df["optimized_weight"],
        name="Optimized",
        marker_color="#2ECC71",
        text=[f"{w:.1%}" for w in weights_df["optimized_weight"]],
        textposition="outside",
        hovertemplate="%{x}<br>Optimized: %{y:.1%}<extra></extra>",
    ))

    fig.update_layout(
        title="Current vs Optimized Allocation",
        xaxis_title="Ticker",
        yaxis=dict(
            title="Portfolio Weight",
            tickformat=".0%",
            range=[0, max_w * 1.25],
        ),
        barmode="group",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=420,
        template="plotly_white",
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig


def plot_metrics_table(metrics_df: pd.DataFrame) -> go.Figure:
    row = metrics_df.iloc[0]

    vol_abs = row.get("vol_reduction_abs", row["current_vol_annual"] - row["optimized_vol_annual"])
    current_bucket = row.get("current_bucket", "—")
    optimized_bucket = row.get("optimized_bucket", "—")

    labels       = ["Annualized Volatility", "Vol Reduction",                              "Risk Zone",                           "Target Vol"]
    current_vals = [f"{row['current_vol_annual']:.1%}",   "—",                             current_bucket,                        "—"]
    optimized_vals = [f"{row['optimized_vol_annual']:.1%}", f"{row['vol_reduction_pct']:.1f}%  ({vol_abs:.1%} pp)", optimized_bucket, f"{row['target_vol']:.1%}"]
    delta_vals   = [f"▼ {row['vol_reduction_pct']:.1f}%", "—",                             f"{current_bucket} → {optimized_bucket}", "—"]
    delta_colors = ["#2ECC71", "#888888", "#4A90D9", "#888888"]

    fig = go.Figure(data=[go.Table(
        columnwidth=[180, 140, 180, 160],
        header=dict(
            values=["<b>Metric</b>", "<b>Current</b>", "<b>Optimized</b>", "<b>Change</b>"],
            fill_color="#2C3E50",
            font=dict(color="white", size=13),
            align="left",
            height=38,
        ),
        cells=dict(
            values=[labels, current_vals, optimized_vals, delta_vals],
            fill_color=[["#F8F9FA"]*4, ["#FDEDEC"]*4, ["#EAFAF1"]*4, ["#F8F9FA"]*4],
            font=dict(size=13, color=[["#2C3E50"]*4, ["#2C3E50"]*4, ["#2C3E50"]*4, delta_colors]),
            align="left",
            height=34,
        ),
    )])

    fig.update_layout(
        title="Portfolio Risk Summary",
        height=260,
        margin=dict(t=50, b=0, l=0, r=0),
    )
    return fig