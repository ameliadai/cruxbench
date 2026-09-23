"""Price loading and belief-update metrics shared by the analysis notebooks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-9


def load_hourly(market_id, hourly_dir):
    """Load a trusted hourly Yes-price series on a uniform one-hour grid."""
    path = Path(hourly_dir) / f"{market_id}.json"
    if not path.exists():
        return None
    history = json.loads(path.read_text())
    if "Yes" not in history:
        return None
    series = pd.Series(history["Yes"], dtype=float).sort_index()
    series.index = pd.to_datetime(series.index, utc=True).floor("1h")
    return series.groupby(level=0).last().resample("1h").last().ffill()


def before_after_prices(series, window_hours):
    """Mean price over [h-W+1, h] (before) and [h+1, h+W] (after) for every hour h."""
    before = series.rolling(window=window_hours, min_periods=window_hours).mean()
    return before, before.shift(-window_hours)


def pivot_hour(resolution_datetime):
    """Latest whole hour strictly before the resolution time."""
    hour = resolution_datetime.floor("1h")
    return hour if resolution_datetime > hour else hour - pd.Timedelta(hours=1)


def binary_entropy(p):
    """Entropy of a Bernoulli(p) belief, in bits."""
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def kl_bernoulli(p_after, p_before):
    """KL(p_after || p_before) between Bernoulli beliefs, in bits."""
    p_after = np.clip(p_after, EPS, 1 - EPS)
    p_before = np.clip(p_before, EPS, 1 - EPS)
    return (
        p_after * np.log2(p_after / p_before)
        + (1 - p_after) * np.log2((1 - p_after) / (1 - p_before))
    )


def significance_stars(p_value):
    return "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
