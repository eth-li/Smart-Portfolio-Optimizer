"""
data/buckets.py
Owned by Person B. Used by optimizer/solve.py, optimizer/frontier.py, and app.py.

Design
------
Buckets are purely UI labels — zone markers on the risk slider that Person C
renders. They define a vol range and a base position cap, but do NOT dictate
the target vol passed to the optimizer. That comes from TARGET_VOL in app.py.

max_position is now DYNAMIC — call dynamic_max_position(bucket, n_stocks).
The base cap in BUCKET_PARAMS is an upper ceiling; the actual cap scales down
with portfolio size so a 20-stock portfolio can't concentrate 25% in one name.

DIVERSITY_FACTOR
----------------
Controls the minimum weight floor: min_weight = DIVERSITY_FACTOR / n_stocks.
With 5 stocks and factor=0.2: floor = 4% per position.
With 10 stocks and factor=0.2: floor = 2% per position.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Diversity factor — controls min-weight floor
# min_weight per position = DIVERSITY_FACTOR / n_stocks
# ---------------------------------------------------------------------------
DIVERSITY_FACTOR: float = 0.2

# ---------------------------------------------------------------------------
# Concentration multiplier for dynamic cap
# max_position = min(base_cap, CONCENTRATION_MULTIPLIER / n_stocks)
#
# Meaning: no stock can exceed (CONCENTRATION_MULTIPLIER × equal weight).
# e.g. with k=2.0 and 10 stocks: cap = min(base, 2.0/10) = min(base, 20%)
#      with k=2.0 and 5 stocks:  cap = min(base, 2.0/5)  = min(base, 40%)
# Floor of MIN_POSITION_CAP prevents the cap becoming trivially small for
# large universes (117 tickers in new-stocks mode).
# ---------------------------------------------------------------------------
CONCENTRATION_MULTIPLIER: float = 2.0
MIN_POSITION_CAP: float = 0.05   # never cap below 5% regardless of n_stocks

# ---------------------------------------------------------------------------
# Bucket definitions
# vol_min / vol_max — used by classify_vol() to label UI zones
# base_max_position — ceiling passed to dynamic_max_position(); actual cap
#                     is the minimum of this and CONCENTRATION_MULTIPLIER/n
# ---------------------------------------------------------------------------
BUCKET_PARAMS: dict[str, dict] = {
    "Low": {
        "vol_min":          0.00,
        "vol_max":          0.20,
        "base_max_position": 0.15,   # was 0.25 — tighter for conservative risk
    },
    "Medium-Low": {
        "vol_min":          0.20,
        "vol_max":          0.25,
        "base_max_position": 0.20,   # was 0.30
    },
    "Medium": {
        "vol_min":          0.25,
        "vol_max":          0.30,
        "base_max_position": 0.25,   # was 0.40 — this was letting NVDA hit 40%
    },
    "Medium-High": {
        "vol_min":          0.30,
        "vol_max":          0.38,
        "base_max_position": 0.30,   # was 0.45
    },
    "High": {
        "vol_min":          0.38,
        "vol_max":          0.55,
        "base_max_position": 0.40,   # was 0.60 — "punt" mode still allows concentration
    },
}

VALID_BUCKETS = list(BUCKET_PARAMS.keys())


def dynamic_max_position(bucket: str, n_stocks: int) -> float:
    """
    Compute the effective per-ticker weight cap for this bucket and portfolio size.

    Combines two limits:
      1. base_max_position — hard ceiling from the bucket definition
      2. CONCENTRATION_MULTIPLIER / n_stocks — prevents any stock from being
         more than (k × equal weight), regardless of bucket

    With CONCENTRATION_MULTIPLIER=2.0:
      - 3  stocks → equal weight = 33%, cap = min(base, 67%)  → base wins
      - 5  stocks → equal weight = 20%, cap = min(base, 40%)  → base wins
      - 10 stocks → equal weight = 10%, cap = min(base, 20%)  → dynamic wins for Low/Med
      - 20 stocks → equal weight =  5%, cap = min(base, 10%)  → dynamic wins for all

    A floor of MIN_POSITION_CAP (5%) prevents the cap becoming trivially small
    in the 117-ticker new-stocks universe.

    Parameters
    ----------
    bucket : str
        One of 'Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'.
    n_stocks : int
        Number of tickers in the portfolio (user holdings only, not universe).

    Returns
    -------
    float — effective max weight for any single position.
    """
    if bucket not in BUCKET_PARAMS:
        raise ValueError(f"Unknown bucket '{bucket}'. Valid: {VALID_BUCKETS}")
    if n_stocks < 1:
        raise ValueError(f"n_stocks must be >= 1, got {n_stocks}")

    base = BUCKET_PARAMS[bucket]["base_max_position"]
    dynamic = CONCENTRATION_MULTIPLIER / n_stocks
    return max(MIN_POSITION_CAP, min(base, dynamic))


def get_bucket_params(bucket: str) -> dict:
    """
    Return the full parameter dict for a given risk bucket.

    Keys: vol_min, vol_max, base_max_position.

    For the effective position cap, use dynamic_max_position(bucket, n_stocks)
    instead of reading base_max_position directly.
    """
    if bucket not in BUCKET_PARAMS:
        raise ValueError(
            f"Unknown risk bucket '{bucket}'. "
            f"Valid options: {VALID_BUCKETS}"
        )
    return BUCKET_PARAMS[bucket]


def classify_vol(annualized_vol: float) -> str:
    """
    Map an annualized portfolio vol to its bucket label (for UI display).

    Parameters
    ----------
    annualized_vol : float  e.g. 0.23 = 23% annual vol.

    Returns
    -------
    str: one of 'Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'
    """
    for label, params in BUCKET_PARAMS.items():
        if annualized_vol <= params["vol_max"]:
            return label
    return "High"


def default_target_vol(bucket: str) -> float:
    """
    Midpoint of a bucket's vol range — used as default in tests / CLI only.
    In production, target_vol always comes from TARGET_VOL in app.py.
    """
    params = get_bucket_params(bucket)
    return (params["vol_min"] + params["vol_max"]) / 2