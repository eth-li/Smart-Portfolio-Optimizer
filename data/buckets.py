"""
data/buckets.py
Owned by Person B. Used by optimizer/solve.py, optimizer/frontier.py, and app.py.

Design after refactor
---------------------
Buckets are now purely UI labels — zone markers on the risk slider that Person C
renders. They define a vol range and a max position cap, but do NOT dictate the
target vol passed to the optimizer. That comes directly from the user's slider value.

The optimizer accepts `target_vol` as a plain float. Buckets just tell Person C
where to draw the "Conservative", "Moderate", and "Aggressive" zones on the slider.

DIVERSITY_FACTOR
----------------
Controls the minimum weight floor: min_weight = DIVERSITY_FACTOR / n_stocks.
With 5 stocks and factor=0.2: floor = 4% per position.
With 10 stocks and factor=0.2: floor = 2% per position.
Scales automatically — larger portfolios get smaller floors.
Tune this constant to adjust how aggressively the optimizer diversifies.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Diversity factor — tune this to control the min-weight floor
# min_weight per position = DIVERSITY_FACTOR / n_stocks
# ---------------------------------------------------------------------------
DIVERSITY_FACTOR: float = 0.2

# ---------------------------------------------------------------------------
# Bucket definitions — vol ranges and position caps only.
# target_vol is NOT here. It comes from the user's slider in the UI.
# ---------------------------------------------------------------------------
BUCKET_PARAMS: dict[str, dict] = {
    "Conservative": {
        "vol_min":      0.00,
        "vol_max":      0.20,
        "max_position": 0.30,
    },
    "Moderate": {
        "vol_min":      0.20,
        "vol_max":      0.28,
        "max_position": 0.40,
    },
    "Aggressive": {
        "vol_min":      0.28,
        "vol_max":      0.40,
        "max_position": 0.50,
    },
}

VALID_BUCKETS = list(BUCKET_PARAMS.keys())


def get_bucket_params(bucket: str) -> dict:
    """
    Return the UI parameter dict for a given risk bucket label.

    Parameters
    ----------
    bucket : str
        One of 'Conservative', 'Moderate', 'Aggressive'.

    Returns
    -------
    dict with keys:
        vol_min       float   Lower bound of this bucket's vol range
        vol_max       float   Upper bound of this bucket's vol range
        max_position  float   Per-ticker weight cap for this bucket

    Note: target_vol is NOT in this dict. Pass it directly to optimize().
    """
    if bucket not in BUCKET_PARAMS:
        raise ValueError(
            f"Unknown risk bucket '{bucket}'. "
            f"Valid options: {VALID_BUCKETS}"
        )
    return BUCKET_PARAMS[bucket]


def classify_vol(annualized_vol: float) -> str:
    """
    Map an annualized portfolio vol to the corresponding bucket label.
    Used to display which zone the current portfolio sits in.

    Parameters
    ----------
    annualized_vol : float
        e.g. 0.23 for 23% annual vol.

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


def default_target_vol(bucket: str) -> float:
    """
    Convenience: return the midpoint of a bucket's vol range as a sensible
    default when no slider value is provided (e.g. in tests or CLI).

    Parameters
    ----------
    bucket : str
        One of 'Conservative', 'Moderate', 'Aggressive'.

    Returns
    -------
    float: midpoint of vol_min and vol_max for that bucket.
    """
    params = get_bucket_params(bucket)
    return (params["vol_min"] + params["vol_max"]) / 2