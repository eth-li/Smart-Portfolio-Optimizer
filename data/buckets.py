"""
data/buckets.py
Owned by Person B. Used by optimizer/solve.py, optimizer/frontier.py, and app.py.

Design after refactor
---------------------
Buckets are now purely UI labels — zone markers on the risk slider that Person C
renders. They define a vol range and a max position cap, but do NOT dictate the
target vol passed to the optimizer. That comes directly from the user's slider value.

The optimizer accepts `target_vol` as a plain float. Buckets just tell Person C
where to draw the zone markers on the slider.

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
    "Low": {
        "vol_min":      0.00,
        "vol_max":      0.15,
        "max_position": 0.25,
    },
    "Medium-Low": {
        "vol_min":      0.15,
        "vol_max":      0.22,
        "max_position": 0.30,
    },
    "Medium": {
        "vol_min":      0.22,
        "vol_max":      0.28,
        "max_position": 0.40,
    },
    "Medium-High": {
        "vol_min":      0.28,
        "vol_max":      0.35,
        "max_position": 0.45,
    },
    "High": {
        "vol_min":      0.35,
        "vol_max":      0.50,
        "max_position": 0.60,
    },
}

VALID_BUCKETS = list(BUCKET_PARAMS.keys())


def get_bucket_params(bucket: str) -> dict:
    """
    Return the UI parameter dict for a given risk bucket label.

    Parameters
    ----------
    bucket : str
        One of 'Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'.

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
    str: one of 'Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'
    """
    for label, params in BUCKET_PARAMS.items():
        if annualized_vol <= params["vol_max"]:
            return label
    return "High"


def default_target_vol(bucket: str) -> float:
    """
    Convenience: return the midpoint of a bucket's vol range as a sensible
    default when no slider value is provided (e.g. in tests or CLI).

    Parameters
    ----------
    bucket : str
        One of 'Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'.

    Returns
    -------
    float: midpoint of vol_min and vol_max for that bucket.
    """
    params = get_bucket_params(bucket)
    return (params["vol_min"] + params["vol_max"]) / 2
