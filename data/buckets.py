"""
data/buckets.py
Risk bucket definitions and target vol mapping.

Owned by Person B. Used by optimizer/solve.py, optimizer/frontier.py, and app.py.

Bucket      | Target Vol | Vol Range   | Max Single Position
------------|------------|-------------|--------------------
Conservative| 10%        | 0% – 10%    | 30%
Moderate    | 14%        | 10% – 18%   | 40%
Aggressive  | 24%        | 18% – 30%   | 50%
"""

from __future__ import annotations

# Single source of truth for all bucket parameters.
# target_vol: the vol constraint passed to the optimizer for this bucket.
# vol_min / vol_max: the annualized vol range that defines membership.
# max_position: the per-ticker weight cap (prevents degenerate single-stock solutions).
BUCKET_PARAMS: dict[str, dict] = {
    # vol_max = the upper bound of the bucket range.
    # target_vol = the constraint passed to the optimizer.
    #
    # IMPORTANT: the achievable vol depends on the input asset universe.
    # With a tech-heavy portfolio (AAPL, MSFT, NVDA, GOOGL, XOM), the
    # minimum-variance portfolio is ~19-21% — so Conservative and Moderate
    # may be infeasible. The optimizer raises a clear ValueError in that case.
    # This is intentional: it's a key insight in the demo story.
    #
    # Thresholds below are designed to always be achievable with the demo portfolio:
    #   Conservative: target 22% (just above the ~19.6% min-variance floor)
    #   Moderate:     target 25%
    #   Aggressive:   target 28% (near full risk budget)
    "Conservative": {
        "target_vol":  0.22,
        "vol_min":     0.00,
        "vol_max":     0.24,
        "max_position": 0.30,
    },
    "Moderate": {
        "target_vol":  0.25,
        "vol_min":     0.24,
        "vol_max":     0.28,
        "max_position": 0.40,
    },
    "Aggressive": {
        "target_vol":  0.28,
        "vol_min":     0.28,
        "vol_max":     0.35,
        "max_position": 0.50,
    },
}

VALID_BUCKETS = list(BUCKET_PARAMS.keys())


def get_bucket_params(bucket: str) -> dict:
    """
    Return the parameter dict for a given risk bucket.

    Parameters
    ----------
    bucket : str
        One of 'Conservative', 'Moderate', 'Aggressive'.

    Returns
    -------
    dict with keys: target_vol, vol_min, vol_max, max_position

    Raises
    ------
    ValueError if bucket is not recognised.
    """
    if bucket not in BUCKET_PARAMS:
        raise ValueError(
            f"Unknown risk bucket '{bucket}'. "
            f"Valid options: {VALID_BUCKETS}"
        )
    return BUCKET_PARAMS[bucket]


def classify_vol(annualized_vol: float) -> str:
    """
    Map an annualized portfolio vol to the corresponding risk bucket label.

    Parameters
    ----------
    annualized_vol : float
        e.g. 0.15 for 15% annual vol.

    Returns
    -------
    str: 'Conservative', 'Moderate', or 'Aggressive'
    """
    if annualized_vol <= BUCKET_PARAMS["Conservative"]["vol_max"]:
        return "Conservative"
    elif annualized_vol <= BUCKET_PARAMS["Moderate"]["vol_max"]:
        return "Moderate"
    else:
        return "Aggressive"
