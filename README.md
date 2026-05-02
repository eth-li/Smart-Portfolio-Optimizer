# Smart Portfolio Optimizer
**CDS Datathon 2026** — A data-powered tool that helps users find the minimum-variance allocation for their stock portfolio.

---

## Project Overview

Users input their current stock holdings as portfolio weights and select a risk tolerance level. The tool pulls historical price data and options-implied volatility, builds a forward-looking covariance matrix, and recommends the minimum-variance allocation within the user's chosen risk budget. Output includes an efficient frontier visualization and a current-vs-optimized allocation comparison.

**Real-world decision it solves:** "Given how my stocks move together, how should I reallocate to minimize risk without leaving my chosen risk budget?"

### Why minimum variance, not maximum return?

Expected returns estimated from 2 years of daily price history are too noisy to optimize against reliably. The covariance matrix — how stocks move together — is a much more stable estimate. So the optimizer ignores return forecasts entirely and purely minimizes portfolio variance `w^T * Σ * w` subject to a vol ceiling chosen by the user.

---

## Tech Stack

| Layer | Library |
|---|---|
| Data | `yfinance` (prices + options chains) |
| Math | `numpy`, `pandas`, `scipy` |
| Optimization | `cvxpy` + `clarabel` |
| Frontend | `streamlit` |
| Visualization | `plotly` |

---

## Repo Structure

```
.
├── app.py                  # Streamlit entry point
├── run_demo.py             # Integration test + contract CSV generator (Person B)
├── app/
│   └── charts.py           # Plotly visualizations
├── data/
│   ├── fetch.py            # yfinance scraping
│   ├── risk.py             # covariance + IV
│   └── buckets.py          # risk bucket definitions + diversity floor
├── optimizer/
│   ├── solve.py            # cvxpy minimum-variance optimizer
│   ├── frontier.py         # efficient frontier sweep
│   └── metrics.py          # portfolio vol stats
├── contracts/              # ⭐ Static CSV schemas for parallel development
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

Every file in `contracts/` is a static CSV that defines the exact schema each function must produce and consume. **Code against these mocks on Day 1. Swap in real data at the Day 1 evening checkpoint.**

> **Rule:** Column names, units, and dtypes must match exactly. If your function produces a file, its output must be byte-for-byte compatible with the schema below.

---

### 1. `contracts/portfolio_input.csv`
**Produced by:** Person C (demo portfolio + what the Streamlit UI collects)  
**Consumed by:** Person A (tickers to fetch), Person B (current weights for comparison)

```
ticker,current_weight
AAPL,0.22
MSFT,0.27
NVDA,0.42
GOOGL,0.03
XOM,0.06
```

| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Valid Yahoo Finance ticker, uppercase |
| `current_weight` | `float` | User's current allocation as a decimal (e.g. `0.22` = 22%) |

**Notes:**
- `current_weight` values must sum to exactly 1.0
- Minimum 2 tickers, maximum 10 for the demo
- User enters percentages directly — no share counts, no live price fetching required
- Person C commits this file on Day 1 morning so A and B can build against real tickers

---

### 2. `contracts/returns.csv`
**Produced by:** Person A (`data/fetch.py`)  
**Consumed by:** Person A (feeds `risk.py`)

Daily log returns for each ticker over the trailing 2-year window.

```
date,AAPL,MSFT,NVDA,GOOGL,XOM
2024-01-02,0.00842,-0.00311,0.02104,0.01204,-0.00129
2024-01-03,-0.01023,0.00477,-0.01532,-0.00891,0.00671
...
```

| Column | Type | Description |
|---|---|---|
| `date` | `str` (YYYY-MM-DD) | Trading date |
| `<TICKER>` | `float` | Log return: `ln(P_t / P_{t-1})` |

**Notes:**
- Rows with any NaN dropped entirely — do not forward-fill across tickers
- ~504 rows for a 2-year daily window
- Raw prices not included — log returns only

---

### 3. `contracts/covariance.csv`
**Produced by:** Person A (`data/risk.py`)  
**Consumed by:** Person B (`optimizer/solve.py`, `optimizer/frontier.py`)

The IV-scaled covariance matrix. The single most critical handoff in the project.

```
ticker,AAPL,MSFT,NVDA,GOOGL,XOM
AAPL,0.0784,0.0544,0.1001,0.0588,0.0172
MSFT,0.0544,0.0729,0.1010,0.0591,0.0149
NVDA,0.1001,0.1010,0.3025,0.1056,0.0242
GOOGL,0.0588,0.0591,0.1056,0.0900,0.0145
XOM,0.0172,0.0149,0.0242,0.0145,0.0484
```

| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Row label |
| `<TICKER>` | `float` | Annualized covariance between row and column ticker |

**Notes:**
- Square, symmetric: `cov[i][j] == cov[j][i]`
- All values annualized (daily covariance × 252)
- Diagonal = IV-scaled variance: `σ_i²` where `σ_i` is ATM implied vol. Falls back to historical variance if options data unavailable — flagged in `iv.csv`
- Off-diagonal = `corr_hist[i][j] × σ_i × σ_j`
- Person B validates on load: positive semi-definite (`np.linalg.eigvalsh(cov).min() >= -1e-8`)

---

### 4. `contracts/iv.csv`
**Produced by:** Person A (`data/risk.py`)  
**Consumed by:** Person A (covariance diagonal), Person C (warning badge in UI)

```
ticker,iv_annualized,expiry_date,strike,data_source
AAPL,0.2834,2024-02-16,185.0,options_chain
MSFT,0.2412,2024-02-16,375.0,options_chain
NVDA,0.4521,2024-02-16,875.0,options_chain
GOOGL,0.3102,2024-02-16,140.0,options_chain
XOM,0.1987,2024-02-16,105.0,historical_fallback
```

| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Ticker symbol |
| `iv_annualized` | `float` | ATM implied vol, annualized (e.g. `0.28` = 28%) |
| `expiry_date` | `str` (YYYY-MM-DD) | Options contract expiry used |
| `strike` | `float` | Strike of the ATM option |
| `data_source` | `str` | `options_chain` or `historical_fallback` |

**Notes:**
- ATM = strike closest to spot at time of fetch
- Use nearest expiry with ≥ 7 days to expiration
- `historical_fallback`: use annualized historical vol from `returns.csv` diagonal
- Person C shows a warning badge for any `historical_fallback` row

---

### 5. `contracts/expected_returns.csv`
**Produced by:** Person A (`data/risk.py`)  
**Not consumed by the optimizer** — kept for reference and potential future use

```
ticker,expected_return_annual
AAPL,0.1823
MSFT,0.2104
NVDA,0.4521
GOOGL,0.1567
XOM,0.0987
```

| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Ticker symbol |
| `expected_return_annual` | `float` | Mean daily log return × 252 |

**Notes:**
- The optimizer does **not** use this file. Minimum-variance optimization only needs the covariance matrix.
- Kept in contracts for transparency and slide deck reference — useful to show the audience what returns look like even if we don't optimize against them
- Computed as `mean_daily_log_return × 252` (not compounded)

---

### 6. `contracts/optimized_weights.csv`
**Produced by:** Person B (`optimizer/solve.py`)  
**Consumed by:** Person C (`app/charts.py` — allocation comparison bar chart)

```
ticker,current_weight,optimized_weight
AAPL,0.2200,0.1823
MSFT,0.2700,0.2341
NVDA,0.4200,0.2000
GOOGL,0.0300,0.1512
XOM,0.0600,0.2324
```

| Column | Type | Description |
|---|---|---|
| `ticker` | `str` | Ticker symbol |
| `current_weight` | `float` | User's current allocation (from `portfolio_input.csv`) |
| `optimized_weight` | `float` | Minimum-variance recommended weight |

**Notes:**
- `optimized_weight` sums to exactly 1.0
- All weights ≥ `DIVERSITY_FACTOR / n` (dynamic floor — see Diversity Floor section)
- No weight exceeds `max_position` for the chosen bucket
- Weights stored to 4 decimal places (= 2 decimal places of percent, e.g. `0.1234` = 12.34%)

---

### 7. `contracts/frontier.csv`
**Produced by:** Person B (`optimizer/frontier.py`)  
**Consumed by:** Person C (`app/charts.py` — efficient frontier chart)

Each row is one point on the frontier, solved by minimizing variance at each target vol level.

```
target_vol_annual,realized_vol_annual,AAPL,MSFT,NVDA,GOOGL,XOM
0.1905,0.1892,0.1441,0.2845,0.0000,0.0712,0.5000
0.1975,0.1963,0.1118,0.3743,0.0359,0.0000,0.4779
0.2048,0.2031,0.0889,0.3798,0.0722,0.0000,0.4590
...
```

| Column | Type | Description |
|---|---|---|
| `target_vol_annual` | `float` | The vol ceiling constraint used for this solve |
| `realized_vol_annual` | `float` | Actual achieved vol: `sqrt(w^T * Σ * w)` — may differ slightly from target due to solver rounding |
| `<TICKER>` | `float` | Optimal weight for this ticker at this frontier point |

**Notes:**
- Frontier spans the full vol range (min-variance floor → Aggressive bucket ceiling)
- No `expected_return` or `sharpe_ratio` columns — we do not use return estimates
- Person C plots `realized_vol_annual` on the x-axis (not target) for accuracy
- Person C overlays the current portfolio and the user's chosen point as labeled dots
- Ticker columns in same order as `portfolio_input.csv`
- ~30 points. Infeasible solves silently skipped

---

### 8. `contracts/metrics.csv`
**Produced by:** Person B (`optimizer/metrics.py`)  
**Consumed by:** Person C (`app/charts.py` — headline summary table)

Single-row summary of current vs. optimized portfolio risk.

```
current_vol_annual,optimized_vol_annual,vol_reduction_pct,vol_reduction_abs,target_vol,current_bucket,optimized_bucket
0.3410,0.2400,29.62,0.1010,0.2400,Aggressive,Medium
```

| Column | Type | Description |
|---|---|---|
| `current_vol_annual` | `float` | Annualized vol of the current portfolio |
| `optimized_vol_annual` | `float` | Annualized vol of the optimized portfolio |
| `vol_reduction_pct` | `float` | `(current - optimized) / current × 100` — **headline demo stat** |
| `vol_reduction_abs` | `float` | `current_vol - optimized_vol` in decimal (e.g. `0.10` = 10 percentage points) |
| `target_vol` | `float` | The vol ceiling the user chose via the slider |
| `current_bucket` | `str` | Risk bucket label for current portfolio |
| `optimized_bucket` | `str` | Risk bucket label for optimized portfolio |

**Notes:**
- `vol_reduction_pct` is the first number the audience sees in the demo
- No return or Sharpe columns — the optimizer does not use return estimates
- `current_bucket` and `optimized_bucket` use the 5-bucket labels below

---

## Risk Buckets

Defined in `data/buckets.py`. Buckets are **UI labels only** — they do not control the optimizer directly. They tell Person C where to draw zone markers on the risk slider, and they define the `max_position` cap for each risk level.

The actual `target_vol` passed to the optimizer comes from the user's slider position (a plain float), not from the bucket definition.

| Bucket | Vol Range | Max Single Position |
|---|---|---|
| Low | 0% – 15% | 25% |
| Medium-Low | 15% – 22% | 30% |
| Medium | 22% – 28% | 40% |
| Medium-High | 28% – 35% | 45% |
| High | 35% – 50% | 60% |

If the user's current portfolio vol already exceeds their chosen bucket's upper bound, the app shows: *"Your current portfolio exceeds this risk level."*

---

## Diversity Floor

Defined in `data/buckets.py` as `DIVERSITY_FACTOR = 0.2`.

The minimum weight per position scales dynamically with portfolio size:

```
min_weight = DIVERSITY_FACTOR / n_stocks
```

| Portfolio size | Min weight per position |
|---|---|
| 5 stocks | 4.0% |
| 8 stocks | 2.5% |
| 10 stocks | 2.0% |

This prevents the optimizer from zeroing out positions entirely while naturally allowing more concentration flexibility in larger portfolios. The frontier sweep uses `diversity_factor = 0.0` (no floor) to show the full theoretical curve shape.

---

## Team Roles & Deliverables

### Person A — Data & Risk Engineer

**Responsibilities:**
- `data/fetch.py` — fetch 2 years of daily prices via yfinance; output `returns.csv`
- `data/risk.py` — build IV-scaled covariance matrix; output `covariance.csv`, `iv.csv`, `expected_returns.csv`
- Handle edge cases: missing tickers, illiquid options, NaN rows
- Fallback: if options data unavailable, use historical vol and set `data_source = historical_fallback`

**Deliverables:**
- `data/fetch.py`, `data/risk.py`
- Populate `contracts/returns.csv`, `contracts/covariance.csv`, `contracts/iv.csv`, `contracts/expected_returns.csv` with real data by Day 1 evening checkpoint

---

### Person B — Optimization & Backend

**Responsibilities:**
- `optimizer/solve.py` — minimum-variance cvxpy optimizer; takes `cov_df`, `current_weights`, `target_vol`, `max_position`, `diversity_factor`
- `optimizer/frontier.py` — sweep `target_vol` from min-variance floor to vol ceiling; output `frontier.csv`
- `optimizer/metrics.py` — compute vol stats; output `metrics.csv`
- `data/buckets.py` — 5-bucket definitions, `DIVERSITY_FACTOR`, `classify_vol()`, `default_target_vol()`
- `run_demo.py` — integration test; runs full pipeline and regenerates all output contracts

**Dev on Day 1:** Read from `contracts/covariance.csv` mock. Do not wait on Person A.

**Deliverables:**
- All optimizer files + `data/buckets.py` + `run_demo.py`
- Output contracts: `optimized_weights.csv`, `frontier.csv`, `metrics.csv`
- All assertions in `run_demo.py` must pass before handing off to Person C

---

### Person C — Frontend & Presentation

**Responsibilities:**
- `app.py` — Streamlit entry point: weight input table, risk slider with bucket zone markers, results display
- `app/charts.py` — three Plotly charts:
  - Efficient frontier (`frontier.csv`) — x-axis is `realized_vol_annual`, current + optimized portfolio marked as labeled dots, bucket zones shaded
  - Allocation comparison grouped bar chart (`optimized_weights.csv`) — current vs. optimized side by side
  - Headline metrics display (`metrics.csv`) — `vol_reduction_pct` shown first, bucket change shown prominently
- Commit demo portfolio to `contracts/portfolio_input.csv` on Day 1 morning
- Build slide deck and presentation narrative
- Final 15-min polish pass

**Dev on Day 1:** All charts read directly from `contracts/*.csv` mocks. Zero blocking dependency on A or B.

**Deliverables:**
- `app.py`, `app/charts.py`, `slides/`
- Demo script + 1 practice run before submission

---

## Workflow & Dependencies

```
Person A (data) ──┐
                  ├──► Person C (frontend integrates both)
Person B (optim) ──┘
```

**Day 1 — parallel:**
- A: build `fetch.py` + `risk.py`, validate options data in hour 1
- B: build optimizer reading from `contracts/` mocks, run `python3 run_demo.py` green
- C: commit demo portfolio, build all three charts against mock CSVs

**Day 1 evening — integration checkpoint:**
1. A commits real `covariance.csv` to `contracts/`
2. B pulls, runs `python3 run_demo.py`, commits real output CSVs
3. C swaps mock reads for live function calls, confirms charts render end-to-end

**Day 2:**
- End-to-end test with demo portfolio
- Polish, slide deck, practice run

---

## Critical Path Risks

| Risk | Owner | Mitigation |
|---|---|---|
| yfinance options data unavailable | A | Validate in hour 1. `historical_fallback` path already handled in schema. Frame IV as "forward-looking enhancement" on slide. |
| Min-variance floor makes bucket infeasible | B | Run `python3 run_demo.py` against real covariance early. If min-variance portfolio already exceeds target, show informative error in UI. |
| Optimizer produces near-equal weights (boring demo) | B | Tune `DIVERSITY_FACTOR` — lower it to allow more concentration. Rebalancing story needs meaningful weight changes. |
| Frontend blocked on real data | C | All charts built against `contracts/*.csv` mocks — zero blocking dependency. |
| Schema mismatch at integration | All | Column names are exact — copy from this README, don't retype. Run `df.columns.tolist()` check on every read. |

---

## Setup

```bash
pip install -r requirements.txt

# Run integration test and regenerate output contracts
python3 run_demo.py

# Run with a specific risk level (mirrors the UI slider)
python3 run_demo.py --target-vol 0.20
python3 run_demo.py --bucket High

# Launch the app
streamlit run app.py
```

---

## Demo Portfolio

`AAPL, MSFT, NVDA, GOOGL, XOM` — tech-heavy with one energy defensive. Current allocation is NVDA-dominated (~42%), which sits in the High risk bucket. The optimizer meaningfully redistributes weight toward MSFT and XOM, pulling the portfolio down into Medium or Medium-High. Clear, visual rebalancing story.

Demo weights committed to `contracts/portfolio_input.csv`:
```
AAPL  22%  |  MSFT  27%  |  NVDA  42%  |  GOOGL  3%  |  XOM  6%
```