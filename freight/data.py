"""Loading and cleaning of the raw freight load files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Weights in the raw data are capped at this value (it is the 99th percentile and the max).
WEIGHT_CAP = 47_500.0


def load(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / name, parse_dates=["date"])


def clean_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Fix input-side data-quality issues. Safe to apply to any file (no labels used).

    - Negative weights: ~0.6% of rows. Their magnitudes follow the same distribution as
      positive weights, so they are treated as sign errors and flipped.
    - Missing weights: kept as NaN (LightGBM routes them natively) plus an indicator.
    - Missing market_index: ~0.8% of rows. market_index is a daily market level with small
      per-load noise (within-day std ~0.025 vs 0.17 overall), so it is filled with that
      date's median from the same file.
    """
    out = frame.copy()
    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight"] = out["weight"].abs().clip(upper=WEIGHT_CAP)

    out["market_index_missing"] = out["market_index"].isna().astype(int)
    daily = out.groupby("date")["market_index"].transform("median")
    out["market_index"] = out["market_index"].fillna(daily)
    out["market_index_daily"] = daily
    return out


def flag_label_outliers(rate: pd.Series, reference: pd.Series, threshold: float = 1.6) -> pd.Series:
    """Flag labels whose ratio to an out-of-fold reference prediction is implausible.

    About 1.4% of posted_rate values are ~3.5x too high or ~0.3x too low (clusters that are
    well separated from the bulk of residuals), consistent with entry/unit errors.
    """
    ratio = np.abs(np.log(rate / reference))
    return ratio > np.log(threshold)
