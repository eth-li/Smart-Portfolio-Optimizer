import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_efficient_frontier(frontier_df: pd.DataFrame, metrics_df: pd.DataFrame) -> go.Figure:
    row = metrics_df.iloc[0]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frontier_df["realized_vol_annual"],
        y=frontier_df["expected_return_annual"],
        mode="lines",
        name="Possible Portfolios",
        line=dict(color="#4A90D9", width=3),
        hovertemplate="Risk: %{x:.1%}<br>Expected Return: %{y:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=frontier_df["realized_vol_annual"].tolist() + [frontier_df["realized_vol_annual"].min()],
        y=frontier_df["expected_return_annual"].tolist() + [frontier_df["expected_return_annual"].min()],
        fill="toself",
        fillcolor="rgba(74,144,217,0.07)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    curr_vol = row["current_vol_annual"]
    curr_ret = row["current_return_annual"]
    opt_vol  = row["optimized_vol_annual"]
    opt_ret  = row["optimized_return_annual"]

    fig.add_annotation(
        x=opt_vol, y=opt_ret,
        ax=curr_vol, ay=curr_ret,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=3, arrowsize=1.4, arrowwidth=2.5,
        arrowcolor="#27AE60",
    )

    fig.add_trace(go.Scatter(
        x=[curr_vol], y=[curr_ret],
        mode="markers+text",
        showlegend=False,
        marker=dict(size=14, color="#E74C3C", symbol="diamond",
                    line=dict(color="white", width=2)),
        text=["Your Portfolio"],
        textposition="top left",
        hovertemplate="Your Portfolio<br>Risk: %{x:.1%}<br>Expected Return: %{y:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[opt_vol], y=[opt_ret],
        mode="markers+text",
        showlegend=False,
        marker=dict(size=16, color="#2ECC71", symbol="star",
                    line=dict(color="white", width=2)),
        text=["Optimized"],
        textposition="top right",
        hovertemplate="Optimized<br>Risk: %{x:.1%}<br>Expected Return: %{y:.1%}<extra></extra>",
    ))

    fig.add_annotation(
        text="Each point on the curve is the best possible<br>expected return at that risk level",
        x=frontier_df["realized_vol_annual"].median(),
        y=frontier_df["expected_return_annual"].max() * 1.02,
        xref="x", yref="y",
        showarrow=False,
        font=dict(size=11, color="#888888"),
        align="center",
    )

    fig.update_layout(
        title="Risk vs. Expected Return — Where Your Portfolio Sits",
        xaxis=dict(
            title="Annual Risk (how much the portfolio could swing)",
            tickformat=".0%",
            nticks=6,
        ),
        yaxis=dict(
            title="Expected Annual Return",
            tickformat=".0%",
            nticks=6,
        ),
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        height=300,          # ← smaller (was 360)
        template="plotly_white",
        margin=dict(t=50, b=50, l=60, r=40),
    )
    return fig


def plot_allocation_comparison(weights_df: pd.DataFrame) -> go.Figure:
    """Two side-by-side donut pies — scales cleanly with any number of stocks."""
    all_tickers = sorted(set(
        weights_df.loc[weights_df["current_weight"]   > 0, "ticker"].tolist() +
        weights_df.loc[weights_df["optimized_weight"] > 0, "ticker"].tolist()
    ))

    _PALETTE = [
        "#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6",
        "#1ABC9C", "#E67E22", "#3498DB", "#E91E63", "#00BCD4",
        "#8BC34A", "#FF5722", "#607D8B", "#795548", "#FF9800",
        "#673AB7", "#009688", "#F44336", "#2196F3", "#4CAF50",
    ]
    color_map = {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(all_tickers)}

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["Current Allocation", "Optimized Allocation"],
    )

    curr = weights_df[weights_df["current_weight"] > 0].copy()
    fig.add_trace(
        go.Pie(
            labels=curr["ticker"],
            values=curr["current_weight"],
            marker_colors=[color_map[t] for t in curr["ticker"]],
            textinfo="label+percent",
            textposition="auto",
            hovertemplate="%{label}<br>Current: %{percent}<extra></extra>",
            hole=0.35,
            showlegend=False,
        ),
        row=1, col=1,
    )

    opt = weights_df[weights_df["optimized_weight"] > 0].copy()
    fig.add_trace(
        go.Pie(
            labels=opt["ticker"],
            values=opt["optimized_weight"],
            marker_colors=[color_map[t] for t in opt["ticker"]],
            textinfo="label+percent",
            textposition="auto",
            hovertemplate="%{label}<br>Optimized: %{percent}<extra></extra>",
            hole=0.35,
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_layout(
        height=500,          # ← bigger (was 420)
        template="plotly_white",
        margin=dict(t=60, b=20, l=20, r=20),
        annotations=[
            dict(font=dict(size=14, color="#2C3E50")),
            dict(font=dict(size=14, color="#2C3E50")),
        ],
    )
    return fig


def plot_metrics_table(metrics_df: pd.DataFrame) -> go.Figure:
    row = metrics_df.iloc[0]

    vol_abs          = row.get("vol_reduction_abs", row["current_vol_annual"] - row["optimized_vol_annual"])
    current_bucket   = row.get("current_bucket",   "—")
    optimized_bucket = row.get("optimized_bucket", "—")

    labels         = ["Annual Risk",                        "Risk Reduction",                                       "Risk Zone",                              "Target Risk"]
    current_vals   = [f"{row['current_vol_annual']:.1%}",   "—",                                                    current_bucket,                           "—"]
    optimized_vals = [f"{row['optimized_vol_annual']:.1%}", f"{row['vol_reduction_pct']:.1f}%  ({vol_abs:.1%} pp)", optimized_bucket,                         f"{row['target_vol']:.1%}"]
    delta_vals     = [f"▼ {row['vol_reduction_pct']:.1f}%", "—",                                                    f"{current_bucket} → {optimized_bucket}", "—"]
    delta_colors   = ["#2ECC71", "#888888", "#4A90D9", "#888888"]

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
            fill_color=[["#F8F9FA"] * 4, ["#FDEDEC"] * 4, ["#EAFAF1"] * 4, ["#F8F9FA"] * 4],
            font=dict(size=13, color=[["#2C3E50"] * 4, ["#2C3E50"] * 4, ["#2C3E50"] * 4, delta_colors]),
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