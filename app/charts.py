import plotly.graph_objects as go
import pandas as pd


def plot_efficient_frontier(frontier_df: pd.DataFrame, metrics_df: pd.DataFrame) -> go.Figure:
    row = metrics_df.iloc[0]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frontier_df["target_vol_annual"],
        y=frontier_df["expected_return_annual"],
        mode="lines+markers",
        name="Efficient Frontier",
        line=dict(color="#4A90D9", width=2.5),
        marker=dict(
            size=7,
            color=frontier_df["sharpe_ratio"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Sharpe Ratio", thickness=12, len=0.7),
        ),
        hovertemplate="Vol: %{x:.1%}<br>Return: %{y:.1%}<br>Sharpe: %{marker.color:.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[row["current_vol_annual"]],
        y=[row["current_return_annual"]],
        mode="markers+text",
        name="Current Portfolio",
        marker=dict(size=16, color="#E74C3C", symbol="diamond"),
        text=["Current"],
        textposition="top right",
        hovertemplate="Current Portfolio<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[row["optimized_vol_annual"]],
        y=[row["optimized_return_annual"]],
        mode="markers+text",
        name="Optimized Portfolio",
        marker=dict(size=16, color="#2ECC71", symbol="star"),
        text=["Optimized"],
        textposition="top right",
        hovertemplate="Optimized Portfolio<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
    ))

    # Arrow annotation showing the improvement direction
    fig.add_annotation(
        x=row["optimized_vol_annual"],
        y=row["optimized_return_annual"],
        ax=row["current_vol_annual"],
        ay=row["current_return_annual"],
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor="#27AE60",
    )

    fig.update_layout(
        title="Efficient Frontier",
        xaxis=dict(title="Annualized Volatility", tickformat=".0%"),
        yaxis=dict(title="Expected Annual Return", tickformat=".0%"),
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01),
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
    sharpe_delta = row["optimized_sharpe"] - row["current_sharpe"]

    labels = ["Annualized Volatility", "Expected Return", "Sharpe Ratio", "Risk Bucket"]
    current_vals = [
        f"{row['current_vol_annual']:.1%}",
        f"{row['current_return_annual']:.1%}",
        f"{row['current_sharpe']:.2f}",
        row["risk_bucket"],
    ]
    optimized_vals = [
        f"{row['optimized_vol_annual']:.1%}",
        f"{row['optimized_return_annual']:.1%}",
        f"{row['optimized_sharpe']:.2f}",
        row["risk_bucket"],
    ]
    delta_vals = [
        f"▼ {row['vol_reduction_pct']:.1f}%",
        f"▲ {row['return_improvement_pct']:.1f}%",
        f"▲ {sharpe_delta:.2f}",
        "—",
    ]
    delta_colors = ["#2ECC71", "#2ECC71", "#2ECC71", "#888888"]

    fig = go.Figure(data=[go.Table(
        columnwidth=[180, 140, 140, 100],
        header=dict(
            values=["<b>Metric</b>", "<b>Current</b>", "<b>Optimized</b>", "<b>Delta</b>"],
            fill_color="#2C3E50",
            font=dict(color="white", size=13),
            align="left",
            height=38,
        ),
        cells=dict(
            values=[labels, current_vals, optimized_vals, delta_vals],
            fill_color=[
                ["#F8F9FA"] * 4,
                ["#FDEDEC"] * 4,
                ["#EAFAF1"] * 4,
                ["#F8F9FA"] * 4,
            ],
            font=dict(size=13, color=[["#2C3E50"] * 4, ["#2C3E50"] * 4, ["#2C3E50"] * 4, delta_colors]),
            align="left",
            height=34,
        ),
    )])

    fig.update_layout(
        title="Portfolio Metrics Summary",
        height=260,
        margin=dict(t=50, b=0, l=0, r=0),
    )
    return fig
