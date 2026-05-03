"""Fetches historical price data, options chains, and market-cap weights from yfinance.

Outputs:
  contracts/returns.csv  — daily log returns, 5-year window
  contracts/iv.csv       — ATM implied volatility per ticker

Run modes:
  python -m data.fetch                  # fetch portfolio_input.csv tickers
  python -m data.fetch --universe       # fetch full UNIVERSE list
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


def load_portfolio(path: str = "contracts/portfolio_input.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def fetch_returns(tickers: list, period_years: int = 5) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=365 * period_years + 60)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if len(tickers) > 1 else raw["Close"].to_frame(name=tickers[0])
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available]

    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.index = pd.to_datetime(log_returns.index).strftime("%Y-%m-%d")
    log_returns.index.name = "date"
    return log_returns.reset_index()


def _get_spot(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="5d")
    return float(hist["Close"].iloc[-1])


def _fetch_iv_one(ticker: str, spot: float) -> dict:
    """Fetch IV for a single ticker. Always succeeds — falls back to historical on error."""
    today = datetime.today().date()
    try:
        t = yf.Ticker(ticker)
        expiries = t.options
        valid_expiry = next(
            (e for e in sorted(expiries)
             if (datetime.strptime(e, "%Y-%m-%d").date() - today).days >= 7),
            None,
        )
        if valid_expiry is None:
            raise ValueError("no valid expiry")

        calls = t.option_chain(valid_expiry).calls
        calls = calls[calls["impliedVolatility"] > 0].dropna(subset=["impliedVolatility"])
        if calls.empty:
            raise ValueError("no valid options data")

        atm = calls.loc[(calls["strike"] - spot).abs().idxmin()]
        return {
            "ticker":        ticker,
            "iv_annualized": round(float(atm["impliedVolatility"]), 4),
            "expiry_date":   valid_expiry,
            "strike":        float(atm["strike"]),
            "data_source":   "options_chain",
        }
    except Exception:
        return {
            "ticker":        ticker,
            "iv_annualized": None,
            "expiry_date":   "",
            "strike":        spot,
            "data_source":   "historical_fallback",
        }


def fetch_iv(tickers: list, spot_prices: dict, max_workers: int = 10) -> pd.DataFrame:
    """Fetch ATM IV for all tickers in parallel."""
    rows = [None] * len(tickers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_fetch_iv_one, t, spot_prices[t]): i
            for i, t in enumerate(tickers)
        }
        for future in as_completed(future_to_idx):
            rows[future_to_idx[future]] = future.result()
    return pd.DataFrame(rows)


def _fetch_market_cap_one(ticker: str) -> tuple[str, float]:
    """
    Fetch market cap for a single ticker.
    Returns (ticker, cap_in_dollars). Returns 0.0 on any error so the
    caller can always build a complete Series without crashing.
    """
    try:
        cap = getattr(yf.Ticker(ticker).fast_info, "market_cap", 0) or 0
        return ticker, float(cap)
    except Exception:
        return ticker, 0.0


def fetch_market_cap_weights(tickers: list[str], max_workers: int = 20) -> pd.Series:
    """
    Fetch market-cap weights for all tickers in parallel.

    Used as the neutral prior for Black-Litterman. Large-cap stocks get higher
    equilibrium returns, which prevents the optimizer from over-concentrating in
    recent winners that happen to have high historical returns.

    Parameters
    ----------
    tickers : list[str]
        Tickers to fetch. Order is preserved in the returned Series.
    max_workers : int
        Thread pool size. 20 is safe for yfinance's rate limits.

    Returns
    -------
    pd.Series indexed by ticker, values sum to 1.0.
    Falls back to equal weights if all fetches fail or total cap is zero.

    Notes
    -----
    Uses ThreadPoolExecutor (same pattern as fetch_iv) so 117-ticker universe
    takes ~3-5s instead of ~60s from the old serial loop.
    Any ticker with a failed fetch gets weight 0 before normalization —
    it effectively falls out of the BL prior, which is a safe degradation.
    """
    caps: dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_market_cap_one, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, cap = future.result()
            caps[ticker] = cap

    s = pd.Series(caps, dtype=float).reindex(tickers).fillna(0.0)
    total = s.sum()

    if total < 1e-8:
        # All fetches failed — equal weights is a safe BL prior fallback
        return pd.Series(1.0 / len(tickers), index=tickers)

    return s / total


def run(tickers: list | None = None, portfolio_path: str = "contracts/portfolio_input.csv") -> tuple:
    if tickers is None:
        tickers = load_portfolio(portfolio_path)["ticker"].tolist()

    print(f"Fetching price history for {len(tickers)} tickers...")
    returns_df = fetch_returns(tickers)
    fetched = [c for c in returns_df.columns if c != "date"]
    returns_df.to_csv("contracts/returns.csv", index=False)
    print(f"  Saved returns.csv — {returns_df.shape[0]} rows, {len(fetched)} tickers")

    print("Fetching spot prices...")
    spot_prices = {}
    for t in fetched:
        try:
            spot_prices[t] = _get_spot(t)
        except Exception:
            spot_prices[t] = 0.0

    print(f"Fetching options chains (parallel, {len(fetched)} tickers)...")
    iv_df = fetch_iv(fetched, spot_prices)
    iv_df.to_csv("contracts/iv.csv", index=False)
    fallbacks = iv_df[iv_df["data_source"] == "historical_fallback"]["ticker"].tolist()
    if fallbacks:
        print(f"  historical_fallback for: {fallbacks}")
    print(f"  Saved iv.csv — {len(iv_df)} tickers")

    return returns_df, iv_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", action="store_true", help="Fetch full UNIVERSE list")
    args = parser.parse_args()

    if args.universe:
        from data.universe import UNIVERSE
        run(tickers=UNIVERSE)
    else:
        run()