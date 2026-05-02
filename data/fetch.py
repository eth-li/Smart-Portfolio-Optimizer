"""Fetches historical price data and options chains from yfinance.

Outputs:
  contracts/returns.csv  — daily log returns, 2-year window
  contracts/iv.csv       — ATM implied volatility per ticker
"""

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
    prices = prices[tickers]  # enforce ticker order

    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.index = pd.to_datetime(log_returns.index).strftime("%Y-%m-%d")
    log_returns.index.name = "date"
    return log_returns.reset_index()


def _get_spot(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="5d")
    return float(hist["Close"].iloc[-1])


def fetch_iv(tickers: list, spot_prices: dict) -> pd.DataFrame:
    today = datetime.today().date()
    min_days = 7
    rows = []

    for ticker in tickers:
        spot = spot_prices[ticker]
        try:
            t = yf.Ticker(ticker)
            expiries = t.options

            valid_expiry = next(
                (e for e in sorted(expiries)
                 if (datetime.strptime(e, "%Y-%m-%d").date() - today).days >= min_days),
                None,
            )
            if valid_expiry is None:
                raise ValueError("no valid expiry")

            calls = t.option_chain(valid_expiry).calls
            calls = calls[calls["impliedVolatility"] > 0].dropna(subset=["impliedVolatility"])
            if calls.empty:
                raise ValueError("no valid options data")

            atm = calls.loc[(calls["strike"] - spot).abs().idxmin()]
            rows.append({
                "ticker": ticker,
                "iv_annualized": round(float(atm["impliedVolatility"]), 4),
                "expiry_date": valid_expiry,
                "strike": float(atm["strike"]),
                "data_source": "options_chain",
            })

        except Exception:
            rows.append({
                "ticker": ticker,
                "iv_annualized": None,  # filled in by risk.py from historical vol
                "expiry_date": "",
                "strike": spot,
                "data_source": "historical_fallback",
            })

    return pd.DataFrame(rows)


def run(portfolio_path: str = "contracts/portfolio_input.csv") -> tuple:
    portfolio = load_portfolio(portfolio_path)
    tickers = portfolio["ticker"].tolist()

    print("Fetching price history...")
    returns_df = fetch_returns(tickers)
    returns_df.to_csv("contracts/returns.csv", index=False)
    print(f"  Saved returns.csv — {returns_df.shape[0]} rows, {len(tickers)} tickers")

    print("Fetching spot prices...")
    spot_prices = {t: _get_spot(t) for t in tickers}

    print("Fetching options chains...")
    iv_df = fetch_iv(tickers, spot_prices)
    iv_df.to_csv("contracts/iv.csv", index=False)
    fallbacks = iv_df[iv_df["data_source"] == "historical_fallback"]["ticker"].tolist()
    if fallbacks:
        print(f"  Warning: historical_fallback used for {fallbacks}")
    print("  Saved iv.csv")

    return returns_df, iv_df


if __name__ == "__main__":
    run()
