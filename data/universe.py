"""
Canonical universe of tickers available for portfolio optimization.
Pre-computed in contracts/ by running: python -m data.fetch --universe
"""

UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD",
    # Broader tech
    "ORCL", "CRM", "ADBE", "QCOM", "TXN", "IBM", "CSCO", "INTU", "NOW",
    "INTC", "MU", "AMAT", "PANW", "PLTR", "SNOW",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "AMGN", "GILD", "REGN", "VRTX",
    "MDT", "BMY", "PFE", "SYK", "ISRG", "TMO", "ABT", "ELV", "CI", "ZTS",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "BLK", "AXP", "C",
    "BRK-B", "SPGI", "MCO",
    # Consumer Discretionary
    "WMT", "HD", "MCD", "COST", "SBUX", "NKE", "LOW", "TGT",
    # Consumer Staples
    "KO", "PEP", "PG", "PM", "MDLZ",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX",
    # Industrials
    "GE", "CAT", "BA", "RTX", "HON", "UNP", "DE", "ETN", "LMT", "NOC",
    # Communication
    "DIS", "T", "VZ", "CMCSA", "NFLX",
    # ETFs — broad market
    "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "EEM",
    # ETFs — fixed income
    "TLT", "BND", "AGG", "HYG",
    # ETFs — sector
    "XLF", "XLE", "XLK", "XLV", "XLI", "VNQ",
    # ETFs — factor / thematic
    "SCHD", "VIG", "JEPI", "ARKK",
    # ETFs — commodity
    "GLD", "IAU", "SLV",
    # ETFs — leveraged (high risk)
    "SPXL", "TQQQ", "SOXL",
]
