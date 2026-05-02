# Smart Portfolio Optimizer
**CDS Datathon 2026** — A data-powered tool that helps users make better portfolio allocation decisions.

---

## Project Overview

Users input their current stock holdings and select a risk tolerance bucket (Conservative / Moderate / Aggressive). The tool pulls historical price data and options-implied volatility, builds a forward-looking covariance matrix, and recommends optimal portfolio weights. Output includes an efficient frontier visualization and a current-vs-optimized allocation comparison.

**Real-world decision it solves:** "Am I taking the right amount of risk for my expected return, and how should I rebalance?"

---

## Tech Stack

| Layer | Library |
|---|---|
| Data | `yfinance` (prices + options chains) |
| Math | `numpy`, `pandas`, `scipy` |
| Optimization | `cvxpy` |
| Frontend | `streamlit` |
| Visualization | `plotly` |

---

## Repo Structure

```
.
├── app.py                  # Streamlit entry point
├── app/
│   └── charts.py           # Plotly visualizations
├── data/
│   ├── fetch.py            # yfinance scraping
│   ├── risk.py             # covariance + IV
│   └── buckets.py          # risk bucket logic
├── optimizer/
│   ├── solve.py            # cvxpy optimizer
│   ├── frontier.py         # efficient frontier
│   └── metrics.py          # portfolio stats
├── contracts/              # ⭐ Static CSV files used for parallel dev (mock data)
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

## ⭐ Data Contracts

This section is the source of truth for parallel development. Every file in `contracts/` is a static mock CSV that matches the exact schema each function must produce and consume. **On Day 1, code against these mocks. On Day 1 evening, swap in real data.**

> **Rule:** If your function produces data, its output must match the schema below exactly — same column names, same units, same dtypes. If your function consumes data, write it to read this schema, not a custom one.

---

### 1. `contracts/portfolio_input.csv`
**Produced by:** Person C (hardcoded demo portfolio, also what the Streamlit UI collects from the user)  
**Consumed by:** Person A (to know which tickers to fetch), Person B (to compute current weights)

This is the starting point. Everything else is derived from this.

```
ticker,shares
AAPL,50
MSFT,30
GOOGL,10
JPM,40
XOM,25
```

| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Valid Yahoo Finance ticker symbol, uppercase |
| `shares` | `int` | Number of shares held (whole shares only) |

**Notes:**
- Minimum 2 tickers, maximum 10 tickers for the demo
- Person C should commit the demo portfolio to `contracts/portfolio_input.csv` on Day 1 morning so A and B can build against real tickers

---

### 2. `contracts/returns.csv`
**Produced by:** Person A (`data/fetch.py`)  
**Consumed by:** Person A (feeds into `risk.py`), Person B (to compute expected returns)

Daily log returns for each ticker over the trailing 2-year window.

```
date,AAPL,MSFT,GOOGL,JPM,XOM
2024-01-02,0.00842,-0.00311,0.01204,0.00563,-0.00129
2024-01-03,-0.01023,0.00477,-0.00891,-0.00234,0.00671
...
```

| Column | Type | Description |
|---|---|---|
| `date` | `str` (YYYY-MM-DD) | Trading date |
| `<TICKER>` | `float` | Log return for that ticker: `ln(P_t / P_{t-1})` |

**Notes:**
- One column per ticker from `portfolio_input.csv` — column order must match ticker order in `portfolio_input.csv`
- Rows with **any** NaN dropped entirely (don't forward-fill across tickers)
- Approximately 504 rows for a 2-year daily window
- Do not include the raw price — only log returns

---

### 3. `contracts/covariance.csv`
**Produced by:** Person A (`data/risk.py`)  
**Consumed by:** Person B (`optimizer/solve.py`, `optimizer/frontier.py`)

The IV-scaled covariance matrix. This is the single most important handoff in the project.

```
ticker,AAPL,MSFT,GOOGL,JPM,XOM
AAPL,0.08234,0.04812,0.03901,0.02341,0.01203
MSFT,0.04812,0.07651,0.04102,0.02019,0.00987
GOOGL,0.03901,0.04102,0.09823,0.01876,0.00754
JPM,0.02341,0.02019,0.01876,0.06234,0.01543
XOM,0.01203,0.00987,0.00754,0.01543,0.05671
```

| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Row label (ticker name) |
| `<TICKER>` | `float` | Annualized covariance between row ticker and column ticker |

**Notes:**
- Matrix is square and symmetric: `cov[i][j] == cov[j][i]`
- All values are **annualized** (multiply daily covariance × 252)
- Diagonal values are the IV-scaled variances: `σ_i²` where `σ_i` is the ATM implied vol for ticker `i`. If IV fetch fails, fall back to historical variance and flag it
- Off-diagonal values use **historical correlations** scaled by IV vols: `cov[i][j] = corr_hist[i][j] × σ_i × σ_j`
- Ticker order must match `portfolio_input.csv` ticker order
- Person B should validate: diagonals are positive, matrix is positive semi-definite (`np.linalg.eigvalsh(cov).min() >= -1e-8`)

---

### 4. `contracts/iv.csv`
**Produced by:** Person A (`data/risk.py`)  
**Consumed by:** Person A (to build the diagonal of the covariance matrix), Person C (to display in the metrics table)

```
ticker,iv_annualized,expiry_date,strike,data_source
AAPL,0.2834,2024-02-16,185.0,options_chain
MSFT,0.2412,2024-02-16,375.0,options_chain
GOOGL,0.3102,2024-02-16,140.0,options_chain
JPM,0.2201,2024-02-16,195.0,historical_fallback
XOM,0.1987,2024-02-16,105.0,options_chain
```

| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Ticker symbol |
| `iv_annualized` | `float` | ATM implied volatility, annualized (e.g., `0.28` = 28%) |
| `expiry_date` | `str` (YYYY-MM-DD) | Expiry of the options contract used |
| `strike` | `float` | Strike price of the ATM option used |
| `data_source` | `str` | `options_chain` or `historical_fallback` |

**Notes:**
- ATM = strike closest to the current spot price at time of fetch
- Use the nearest expiry with at least 7 days to expiration to avoid expiry noise
- If `data_source == historical_fallback`, use annualized historical vol from `returns.csv` diagonal
- Person C should show a warning badge in the UI for any `historical_fallback` ticker

---

### 5. `contracts/expected_returns.csv`
**Produced by:** Person A (`data/risk.py` or `data/fetch.py`)  
**Consumed by:** Person B (`optimizer/solve.py`)

```
ticker,expected_return_annual
AAPL,0.1823
MSFT,0.2104
GOOGL,0.1567
JPM,0.1234
XOM,0.0987
```

| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Ticker symbol |
| `expected_return_annual` | `float` | Annualized expected return (e.g., `0.18` = 18%) |

**Notes:**
- Computed as the mean of daily log returns × 252 (simple historical mean)
- This is a known limitation — document it in the slide deck as "assumes the past predicts the future"
- **Do not annualize by compounding** (i.e., `mean_daily × 252`, not `(1 + mean_daily)^252 - 1`) — keeps the math consistent with the covariance matrix
- Ticker order must match `portfolio_input.csv`

---

### 6. `contracts/optimized_weights.csv`
**Produced by:** Person B (`optimizer/solve.py`)  
**Consumed by:** Person C (`app/charts.py` — allocation comparison bar chart)

```
ticker,current_weight,optimized_weight
AAPL,0.3521,0.2800
MSFT,0.2108,0.2500
GOOGL,0.1834,0.1500
JPM,0.1423,0.2000
XOM,0.1114,0.1200
```

| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Ticker symbol |
| `current_weight` | `float` | Current portfolio weight by market value (sums to 1.0) |
| `optimized_weight` | `float` | Optimizer-recommended weight (sums to 1.0) |

**Notes:**
- `current_weight` = `(shares × current_price) / total_portfolio_value`
- `optimized_weight` values must sum to exactly 1.0 (enforce in `solve.py`)
- No negative weights (long-only constraint)
- No single weight exceeds 0.40 (40% max-position cap — prevents degenerate single-stock solutions)
- Weights below 0.005 (0.5%) should be zeroed out and renormalized before writing

---

### 7. `contracts/frontier.csv`
**Produced by:** Person B (`optimizer/frontier.py`)  
**Consumed by:** Person C (`app/charts.py` — efficient frontier scatter plot)

Each row is one point on the frontier, solved by sweeping target volatility from min to max.

```
target_vol_annual,expected_return_annual,sharpe_ratio,AAPL,MSFT,GOOGL,JPM,XOM
0.08,0.0812,0.6150,0.0500,0.1200,0.0300,0.4500,0.3500
0.10,0.1034,0.7386,0.1200,0.1800,0.0800,0.3800,0.2400
0.12,0.1256,0.8133,0.2000,0.2200,0.1200,0.3000,0.1600
...
0.28,0.2102,0.6793,0.3800,0.3500,0.2200,0.0500,0.0000
```

| Column | Type | Description |
|---|---|---|
| `target_vol_annual` | `float` | The vol constraint used for this solve (annualized) |
| `expected_return_annual` | `float` | Resulting expected return at this point |
| `sharpe_ratio` | `float` | `expected_return_annual / target_vol_annual` (assume risk-free rate = 0 for simplicity) |
| `<TICKER>` | `float` | Optimal weight for this ticker at this point on the frontier |

**Notes:**
- Sweep `target_vol` from the minimum achievable vol to the max-return vol in ~20 equal steps
- Infeasible solves (if target vol is below the global minimum variance portfolio vol) should be skipped — don't include failed rows
- Person C will mark the **current portfolio** and **optimized portfolio** as labeled points overlaid on this curve
- Ticker columns must match and be in the same order as `portfolio_input.csv`

---

### 8. `contracts/metrics.csv`
**Produced by:** Person B (`optimizer/metrics.py`)  
**Consumed by:** Person C (`app/charts.py` — summary metrics table)

Single-row summary of current vs. optimized portfolio stats.

```
current_vol_annual,optimized_vol_annual,current_return_annual,optimized_return_annual,current_sharpe,optimized_sharpe,risk_bucket,vol_reduction_pct,return_improvement_pct
0.1823,0.1412,0.1634,0.1891,0.8963,1.3392,Moderate,22.5,15.7
```

| Column | Type | Description |
|---|---|---|
| `current_vol_annual` | `float` | Annualized vol of the current portfolio |
| `optimized_vol_annual` | `float` | Annualized vol of the optimized portfolio |
| `current_return_annual` | `float` | Expected annual return of current portfolio |
| `optimized_return_annual` | `float` | Expected annual return of optimized portfolio |
| `current_sharpe` | `float` | Sharpe ratio of current portfolio |
| `optimized_sharpe` | `float` | Sharpe ratio of optimized portfolio |
| `risk_bucket` | `str` | `Conservative`, `Moderate`, or `Aggressive` — as selected by the user |
| `vol_reduction_pct` | `float` | `(current_vol - optimized_vol) / current_vol × 100` — the headline stat for the demo |
| `return_improvement_pct` | `float` | `(optimized_return - current_return) / current_return × 100` |

**Notes:**
- This single row is what Person C uses to populate the "headline numbers" at the top of the app
- Sharpe = return / vol (risk-free rate = 0 for simplicity — note this on the slide)
- `vol_reduction_pct` should be the first number the audience sees in the demo

---

## Risk Bucket Thresholds

Defined in `data/buckets.py`. The optimizer uses `target_vol` as its constraint.

| Bucket | Annualized Vol Target | Max Single Position |
|---|---|---|
| Conservative | ≤ 10% | 30% |
| Moderate | 10% – 18% | 40% |
| Aggressive | 18% – 30% | 50% |

If the user's current portfolio vol already exceeds their chosen bucket's upper bound, the app should show a warning: *"Your current portfolio exceeds the risk target for this bucket."*

---

## Team Roles & Deliverables

### Person A — Data & Risk Engineer
Owns the data pipeline and risk modeling.

**Responsibilities:**
- Set up yfinance data fetching for historical prices (2 years daily) and options chains
- Compute daily log returns and historical covariance matrix
- Extract ATM implied volatility from nearest-expiry options for each ticker
- Build the IV-scaled covariance matrix (IV on diagonal, historical correlations off-diagonal)
- Handle data edge cases: missing tickers, illiquid options, NaN handling

**Deliverables:**
- `data/fetch.py` — price + options scraping; outputs matching `returns.csv` and `iv.csv` schemas
- `data/risk.py` — covariance matrix matching `covariance.csv` schema; expected returns matching `expected_returns.csv` schema
- `data/buckets.py` — risk bucket definitions and target vol mapping
- Populate `contracts/returns.csv`, `contracts/covariance.csv`, `contracts/iv.csv`, `contracts/expected_returns.csv` with real data from the demo portfolio by Day 1 evening checkpoint

---

### Person B — Optimization & Backend
Owns the portfolio optimizer and core logic.

**Responsibilities:**
- Build the cvxpy optimizer: maximize expected return subject to portfolio vol ≤ target
- Add constraints: weights sum to 1, no short positions (≥ 0), max-position cap per bucket
- Generate efficient frontier data points (sweep target vol, solve at each point)
- Compute portfolio metrics: current vol, optimized vol, return delta, Sharpe ratio
- Sanity-check outputs (no degenerate solutions, weights make intuitive sense)

**Dev on Day 1 (before A's real data):** Read from `contracts/covariance.csv` and `contracts/expected_returns.csv` mock files. The mock files are already in the repo — don't wait on Person A.

**Deliverables:**
- `optimizer/solve.py` — main optimization function; outputs matching `optimized_weights.csv` schema
- `optimizer/frontier.py` — efficient frontier; outputs matching `frontier.csv` schema
- `optimizer/metrics.py` — portfolio stats; outputs matching `metrics.csv` schema
- Integration test: run full optimizer against the demo portfolio in `contracts/portfolio_input.csv`, assert outputs match schemas

---

### Person C — Frontend & Presentation
Owns the Streamlit app, visualizations, and demo story.

**Responsibilities:**
- Build the Streamlit app: ticker input table, share counts, risk bucket dropdown
- Wire UI to Person A's data functions and Person B's optimizer
- Build core visualizations in Plotly:
  - Efficient frontier scatter (source: `frontier.csv` schema) with current + optimized points marked
  - Allocation comparison bar chart — grouped bars (source: `optimized_weights.csv` schema)
  - Key metrics summary table (source: `metrics.csv` schema) — headline `vol_reduction_pct` first
- Hardcode and commit the demo portfolio to `contracts/portfolio_input.csv` on Day 1 morning
- Build slide deck and presentation narrative
- Run final 15-min polish pass

**Dev on Day 1 (before B's real optimizer):** Read all chart data directly from `contracts/*.csv` mock files. Build every chart to completion before wiring to live functions.

**Deliverables:**
- `app.py` — main Streamlit entry point
- `app/charts.py` — Plotly chart functions, each accepting a DataFrame matching the schema above
- `slides/` — final presentation deck
- Demo script + 1 practice run before submission

---

## Workflow & Dependencies

```
Person A (data) ──┐
                  ├──► Person C (frontend integrates both)
Person B (optim) ──┘
```

**Day 1 (parallel work):**
- A: ship `fetch.py` + `risk.py` producing DataFrames matching the schemas above
- B: build optimizer reading from `contracts/` mock CSVs — do not wait on A
- C: scaffold Streamlit app reading from `contracts/` mock CSVs, build all charts to completion

**Day 1 evening — integration checkpoint:**
1. A commits real CSVs to `contracts/` from the demo portfolio
2. B confirms optimizer runs end-to-end on real covariance matrix, commits real output CSVs
3. C swaps mock reads for live function calls, confirms all charts render

**Day 2:**
- All three: end-to-end testing with the demo portfolio
- Polish, slide deck, practice run

---

## Critical Path Risks

| Risk | Owner | Mitigation |
|---|---|---|
| yfinance options data is flaky | Person A | Validate in hour 1. Fallback: `data_source = historical_fallback` in `iv.csv`. Frame IV as "future work" on slide if needed. |
| Optimizer returns degenerate solutions (100% one stock) | Person B | Add max-position cap from the bucket thresholds table above. Add assertion: `assert weights.max() <= max_cap`. |
| Frontend blocked on real data | Person C | All charts read from `contracts/*.csv` mocks on Day 1 — zero blocking dependency. |
| Schema mismatch at integration | All | Every function should have a 3-line schema validation: check column names, check dtypes, check value ranges (e.g., weights sum to 1.0 ± 1e-6). |

---

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Demo Portfolio

Committed to `contracts/portfolio_input.csv` by Person C on Day 1 morning. Pick something visually interesting — a tech-heavy allocation with one defensive name works well for showing meaningful rebalancing.

Suggested starting point: `AAPL, MSFT, NVDA, GOOGL, XOM` — heavy tech tilt, one energy defensive, clear rebalancing story.
