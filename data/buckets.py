"""Risk bucket definitions and target vol/position cap mapping."""

BUCKETS = {
    "Conservative": {
        "target_vol": 0.10,
        "vol_min": 0.0,
        "vol_max": 0.10,
        "max_position": 0.30,
    },
    "Moderate": {
        "target_vol": 0.14,
        "vol_min": 0.10,
        "vol_max": 0.18,
        "max_position": 0.40,
    },
    "Aggressive": {
        "target_vol": 0.24,
        "vol_min": 0.18,
        "vol_max": 0.30,
        "max_position": 0.50,
    },
}


def get_bucket(name: str) -> dict:
    if name not in BUCKETS:
        raise ValueError(f"Unknown bucket '{name}'. Choose from: {list(BUCKETS)}")
    return BUCKETS[name]


def check_exceeds_bucket(portfolio_vol: float, bucket_name: str) -> bool:
    """Return True if portfolio vol exceeds the bucket's upper bound."""
    return portfolio_vol > get_bucket(bucket_name)["vol_max"]
