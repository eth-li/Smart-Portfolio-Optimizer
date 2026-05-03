# Smart Portfolio Optimizer
**CDS Datathon 2026** — A data-powered tool that helps users find the optimal allocation for their stock portfolio based on their chosen risk tolerance.

---

## What It Does

Users enter their current stock holdings and pick a risk level. The tool fetches live market data, builds a forward-looking covariance matrix, and runs a mathematically rigorous optimization to recommend a new allocation. It explains what changed, why, and what the risk level means in plain English — powered by Claude AI.

**Real-world decision it solves:** *"Given how my stocks move together and what the market expects, how should I reallocate to get the best expected return for my risk tolerance?"*

---

## How the Math Works

The optimizer runs a full **Markowitz mean-variance optimization** pipeline with several production-grade enhancements:

### 1. Covariance Matrix (how stocks move together)
- Fetches 5 years of daily log returns via `yfinance`
- Builds an IV-scaled covariance matrix: diagonal uses **ATM implied volatility** from live options chains (forward-looking), off-diagonal uses historical correlations scaled by IV
- Falls back to historical variance if options data is unavailable

### 2. Expected Returns — Black-Litterman Model
Raw historical returns are too noisy to optimize against directly (NVDA's 2023 400% run distorts everything). We use a three-stage pipeline:

**Stage 1 — James-Stein Shrinkage**
```
μ_shrunk = 0.6 × μ_historical + 0.4 × μ_cross_sectional_mean
```
Blends each stock's own history 60/40 with the average across all stocks. Dampens outlier years without removing relative rankings.

**Stage 2 — Winsorization**
```
μ_clipped = clip(μ_shrunk, mean ± 2σ)
```
Hard-caps any stock's expected return at 2 standard deviations above/below the universe mean. Prevents NVDA's exceptional years from dominating the signal even after shrinkage.

**Stage 3 — Black-Litterman (no views)**
```
π    = λ · Σ · w_mkt          # equilibrium returns (reverse-optimized from market caps)
μ_BL = 0.9 · π  +  0.1 · μ_hist   # 90% market anchor, 10% historical signal
```
Anchors expected returns to what the market already implies via cap-weighted equilibrium. If NVIDIA has a high market cap, it gets a proportionally higher — but not extreme — equilibrium return. This is the **industry-standard institutional approach** to eliminating estimation error in expected returns.

Market-cap weights are fetched in parallel via `yfinance` for all tickers.

### 3. Optimization — CVXPY + CLARABEL Solver
```
maximize   μ_BL · w  −  λ_reg · ‖w‖²
subject to Σ(w) = 1
           w ≥ lb          (diversity floor: DIVERSITY_FACTOR / n_stocks)
           w ≤ max_cap      (dynamic position cap)
           w^T Σ w ≤ σ²     (vol ceiling from risk slider)
```

Key features:
- **L2 regularization** (`λ_reg = 0.05`): penalizes concentration continuously — putting 40% in one stock costs more penalty than 20% each, so the optimizer naturally diversifies rather than relying solely on hard caps
- **Dynamic position cap**: `min(base_cap, 2.0 / n_stocks)` — scales with portfolio size so a 10-stock portfolio can't concentrate 25% in one name
- **Infeasibility guard**: `compute_min_vol()` finds the true minimum-variance floor first; `target_vol` is clamped to `floor × 1.01` before solving so the solver is always feasible
- **psd_wrap**: bypasses CVXPY's ARPACK-based PSD certification which fails on large matrices (>~30 tickers)

### 4. Efficient Frontier
Sweeps `target_vol` from min-variance floor to `TARGET_VOL[bucket] × 1.6`, solving the full Markowitz problem at each point. The BL mu vector is shared across all 30 points so the frontier is consistent with the optimizer.

### 5. AI Explanation — Claude Haiku
After every optimization, Claude Haiku generates a 3-sentence plain-English explanation:
1. What specific stocks changed and by how much
2. A qualitative *why* — uses the actual correlation matrix to explain which stocks are low-correlation hedges vs. concentrated bets
3. What the resulting risk level means in everyday language ("your portfolio could swing ±X% in a year")

---

## CDS Datathon 2026 — Rubric Checklist

### Technical (10pts)

- ✅ **Data Quality & Relevance** — 5 years of live daily prices + options chains from `yfinance`. IV-scaled covariance matrix. Market-cap weights fetched in parallel. Real financial data, not synthetic.

- ✅ **Model Features, Selection, Optimization** — Full pipeline: James-Stein shrinkage → winsorization → Black-Litterman → CVXPY CLARABEL solver. Tried and justified multiple approaches (raw history → shrinkage → BL). Feature engineering: IV-scaled covariance diagonal, dynamic position caps, L2 regularization.

- ✅ **Accuracy & Generalization** — BL is inherently forward-looking so traditional train/test splits don't apply. Overfitting addressed via: (a) shrinkage toward cross-sectional mean, (b) market-cap prior that dampens in-sample noise, (c) winsorization that removes single-year outliers, (d) L2 regularization that prevents over-concentration on historical winners.

- ✅ **User Interface** — Streamlit app. Plain English throughout ("risk" not "volatility"), tooltips explaining every metric, AI explanation of results, pie charts for allocation (scales to any number of stocks), risk vs. return frontier with plain-English labels.

- ✅ **External APIs & Tools** — `yfinance` (prices + options + market caps), `cvxpy` + `clarabel` (convex optimization), Claude API / Anthropic SDK (Haiku for explanations), `streamlit` (frontend), `plotly` (visualization). All meaningfully integrated.

### Presentation & Communication (8pts)

- ✅ **Idea** — Portfolio optimization is a real problem millions of retail investors face. Direct fit for "data-powered tool that helps users make better real-world decisions."

- ✅ **Visuals** — Risk vs. Return frontier (plain English axes, callout annotation), donut pie charts (current vs. optimized allocation), metrics summary table, 4 headline metric cards.

- ✅ **Story** — User enters their NVDA-heavy portfolio → sees their risk is high → optimizer redistributes weight using Black-Litterman → AI explains what changed and why.

### Results (4pts)

- ✅ **Justification** — BL portfolios historically reduce estimation error vs. naive mean-variance (Idzorek 2005, He & Litterman 1999). Results contextualized via "±X% swing per year" framing.

- ✅ **Impact** — Most retail investors hold unoptimized, concentrated positions. Tool runs in real-time on live market data.

### Flex (5pts)

- ✅ **Extra Technical** — Live options chain parsing, parallel `ThreadPoolExecutor` fetching, convex optimization with custom regularization, LLM API integration, CVXPY with PSD wrapping for large matrices.

- ✅ **Creativity** — Black-Litterman is a genuine institutional finance model rarely seen at student datathons. Most teams use regression/classification; we use convex optimization + Bayesian return estimation.

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Data | `yfinance` (prices, options chains, market caps) |
| Math | `numpy`, `pandas`, `scipy` |
| Optimization | `cvxpy` + `clarabel` |
| Frontend | `streamlit` |
| Visualization | `plotly` |
| AI Explanation | `anthropic` (Claude Haiku) |

---

## Repo Structure

```
.
├── app.py                      # Streamlit entry point (Person C)
├── run_demo.py                 # Integration test + contract CSV generator (Person B)
├── app/
│   └── charts.py               # Plotly visualizations (frontier, pie charts, metrics table)
├── data/
│   ├── fetch.py                # yfinance scraping + parallel market-cap fetch
│   ├── risk.py                 # IV-scaled covariance matrix
│   ├── buckets.py              # Risk buckets, dynamic position caps, diversity floor
│   └── universe.py             # 117-ticker candidate universe for new-stock recommendations
├── optimizer/
│   ├── solve.py                # CVXPY optimizer + Black-Litterman + L2 regularization
│   ├── frontier.py             # Efficient frontier sweep
│   └── metrics.py              # Portfolio vol stats
├── contracts/                  # ⭐ Static CSV schemas for parallel development
│   ├── portfolio_input.csv
│   ├── returns.csv
│   ├── covariance.csv
│   ├── iv.csv
│   ├── expected_returns.csv
│   ├── optimized_weights.csv
│   ├── frontier.csv
│   └── metrics.csv
├── slides/
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

### Why Black-Litterman over raw historical returns?
Raw 5-year returns reward recent outliers (NVDA +400% in 2023). The optimizer then maxes out NVDA at the position cap every time. BL anchors expected returns to market-cap equilibrium — what the *market as a whole* believes — and blends in historical signal at only 10% weight. This produces diversified, defensible allocations.

### Why dynamic position caps?
A fixed 25% cap on a 3-stock portfolio forces near-equal weights. A fixed 25% cap on a 10-stock portfolio still allows heavy concentration in top performers. `min(base_cap, 2.0 / n_stocks)` means no stock can exceed 2× its equal-weight share, regardless of portfolio size.

### Why L2 regularization instead of just tighter caps?
Hard caps create discontinuities in the frontier curve and require manual tuning per bucket. L2 penalty (`−0.05 · ‖w‖²`) continuously penalizes concentration as part of the objective — the optimizer trades off expected return against concentration automatically, producing smoother and more robust allocations.

### Why IV-scaled covariance?
Historical variance underestimates near-term risk during calm periods. ATM implied volatility reflects the market's current forward-looking expectation of each stock's movement. Using IV on the diagonal while preserving historical correlations off-diagonal gives a covariance matrix that is both forward-looking and statistically stable.

---

## Data Contracts

Every file in `contracts/` defines the exact schema each function must produce and consume.

### `contracts/portfolio_input.csv`
| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Valid Yahoo Finance ticker, uppercase |
| `current_weight` | `float` | Allocation as decimal (0.22 = 22%). Must sum to 1.0 |

### `contracts/returns.csv`
| Column | Type | Description |
|---|---|---|
| `date` | `str` YYYY-MM-DD | Trading date |
| `<TICKER>` | `float` | Log return: `ln(P_t / P_{t-1})` |

### `contracts/covariance.csv`
Square symmetric matrix. Diagonal = IV-scaled variance. Off-diagonal = `corr_hist[i,j] × σ_i × σ_j`. All values annualized (× 252).

### `contracts/iv.csv`
| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Ticker |
| `iv_annualized` | `float` | ATM implied vol (0.28 = 28%) |
| `expiry_date` | `str` | Options expiry used |
| `strike` | `float` | ATM strike |
| `data_source` | `str` | `options_chain` or `historical_fallback` |

### `contracts/optimized_weights.csv`
| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Ticker |
| `current_weight` | `float` | User's input weight |
| `optimized_weight` | `float` | BL-optimized weight, 4dp |

### `contracts/metrics.csv`
| Column | Type | Description |
|---|---|---|
| `current_vol_annual` | `float` | Current portfolio annualized vol |
| `optimized_vol_annual` | `float` | Optimized portfolio annualized vol |
| `vol_reduction_pct` | `float` | `(current − optimized) / current × 100` |
| `vol_reduction_abs` | `float` | Absolute reduction in decimal |
| `target_vol` | `float` | User's chosen vol ceiling |
| `current_bucket` | `str` | Risk zone label for current portfolio |
| `optimized_bucket` | `str` | Risk zone label for optimized portfolio |

---

## Risk Buckets & Vol Targets

Defined in `data/buckets.py`. Buckets are UI labels — the actual `target_vol` comes from `TARGET_VOL` dict in `app.py`.

| Bucket | Target Vol | Vol Range (classify) | Base Max Position |
|---|---|---|---|
| Low | 18% | 0–20% | 15% |
| Medium-Low | 22% | 20–25% | 20% |
| Medium | 27% | 25–30% | 25% |
| Medium-High | 33% | 30–38% | 30% |
| High | 42% | 38–55% | 40% |

**Dynamic cap formula:** `effective_cap = max(5%, min(base_cap, 2.0 / n_stocks))`

**Diversity floor:** `min_weight = DIVERSITY_FACTOR(0.2) / n_stocks`

---

## "Recommend New Stocks" Mode

When enabled, the optimizer selects up to `MAX_NEW_STOCKS = 4` candidates from a 117-ticker universe. Candidates are pre-filtered to the top 4 by BL expected return before the optimizer runs — this keeps the universe small, the covariance fetch fast, and the pie charts readable. Existing positions must retain ≥ 40% of their current weight so the optimizer can't ignore the user's portfolio entirely.

---

## Setup

```bash
pip install -r requirements.txt

# Optional: set Claude API key for AI explanations
export ANTHROPIC_API_KEY=your_key_here

# Run integration test and regenerate output contracts
python3 run_demo.py

# Launch the app
streamlit run app.py
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Optional | Enables Claude AI explanations. Falls back to rule-based summary if not set. |