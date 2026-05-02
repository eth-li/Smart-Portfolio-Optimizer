
# Smart Portfolio Optimizer

**CDS Datathon 2026** — A data-powered tool that helps users make better portfolio allocation decisions.

## Project Overview

Users input their current stock holdings and select a risk tolerance bucket (Conservative / Moderate / Aggressive). The tool pulls historical price data and options-implied volatility, builds a forward-looking covariance matrix, and recommends optimal portfolio weights. Output includes an efficient frontier visualization and a current-vs-optimized allocation comparison.

**Real-world decision it solves:** *"Am I taking the right amount of risk for my expected return, and how should I rebalance?"*

## Tech Stack

- **Data:** `yfinance` (prices + options chains)
- **Math:** `numpy`, `pandas`, `scipy`
- **Optimization:** `cvxpy`
- **Frontend:** `streamlit`
- **Visualization:** `plotly`

## Team Roles

### Person A — Data & Risk Engineer
**Owns the data pipeline and risk modeling.**

Responsibilities:
- Set up `yfinance` data fetching for historical prices (2 years daily) and options chains
- Compute daily log returns and historical covariance matrix
- Extract ATM implied volatility from nearest-expiry options for each ticker
- Build the IV-scaled covariance matrix (IV on diagonal, historical correlations off-diagonal)
- Define risk bucket thresholds (e.g., Conservative ≤ 10% annualized vol, Moderate 10–18%, Aggressive 18–30%)
- Handle data edge cases: missing tickers, illiquid options, NaN handling

Deliverables:
- `data/fetch.py` — price + options scraping functions
- `data/risk.py` — covariance matrix + IV extraction
- `data/buckets.py` — risk bucket definitions and target vol mapping

---

### Person B — Optimization & Backend
**Owns the portfolio optimizer and core logic.**

Responsibilities:
- Build the `cvxpy` optimizer: maximize expected return subject to portfolio vol ≤ target
- Add constraints: weights sum to 1, no short positions (≥ 0), optional max-position cap
- Compute expected returns (start with historical mean, document the limitation)
- Generate efficient frontier data points (sweep target vol, solve at each point)
- Compute portfolio metrics: current vol, optimized vol, expected return delta, Sharpe ratio
- Sanity-check outputs (no degenerate solutions, weights make intuitive sense)

Deliverables:
- `optimizer/solve.py` — main optimization function
- `optimizer/frontier.py` — efficient frontier generation
- `optimizer/metrics.py` — portfolio statistics
- Integration tests against a known demo portfolio

---

### Person C — Frontend & Presentation
**Owns the Streamlit app, visualizations, and demo story.**

Responsibilities:
- Build the Streamlit app: ticker input table, share counts, risk bucket dropdown
- Wire UI to Person A's data functions and Person B's optimizer
- Build core visualizations in Plotly:
  - Efficient frontier with current portfolio + optimized portfolio marked
  - Allocation comparison bar chart (current vs. optimized weights)
  - Key metrics summary table
- Pick the demo portfolio (something visually interesting — e.g., tech-heavy with one defensive stock)
- Build slide deck and presentation narrative
- Run final 15-min polish pass

Deliverables:
- `app.py` — main Streamlit entry point
- `app/charts.py` — Plotly chart functions
- `slides/` — final presentation deck
- Demo script + 1 practice run before submission

## Workflow & Dependencies

```
Person A (data) ──┐
                  ├──► Person C (frontend integrates both)
Person B (optim) ──┘
```

**Day 1 (parallel work):**
- A: ship `fetch.py` returning a clean DataFrame of returns + a covariance matrix
- B: build optimizer against a *fake* covariance matrix (don't wait on A)
- C: scaffold Streamlit app with mocked data, build chart components

**Day 1 evening — integration checkpoint:**
- A hands B the real covariance matrix → B confirms optimizer runs end-to-end
- B hands C the optimization output format → C wires it into the UI

**Day 2:**
- All three: end-to-end testing with the demo portfolio
- Polish, slide deck, practice run

## Critical Path Risks

- **`yfinance` options data is flaky.** Person A should validate this in hour 1. Fallback: ship with historical-vol-only and frame IV as future work.
- **Optimization can return degenerate solutions** (100% in one stock) if constraints are loose. Person B should add a max-position cap (e.g., 40%) early.
- **Don't over-engineer the frontend.** 15 min polish at the end — focus on the story, not the CSS.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

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
├── slides/
├── requirements.txt
└── README.md
```
