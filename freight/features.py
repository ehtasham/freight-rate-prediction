"""Feature engineering shared by training, validation and December chart inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]

FEATURES = [
    "distance",
    "log_distance",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "delta_lat",
    "delta_lon",
    "equipment_code",
    "weight",
    "weight_missing",
    "market_index",
    "market_index_daily",
    "quote_signal",
    "day_of_week",
]
CATEGORICAL = ["equipment_code"]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the model matrix.

    City names are not used: 8 validation cities never appear in training, while
    coordinates generalise to them (city target encodings were tried; no gain).
    Calendar month/day-of-year are left out because validation dates (Nov-Dec) lie outside
    the training range and trees cannot extrapolate them; the linear stage in model.py
    carries the time trend and market level instead.
    """
    out = pd.DataFrame(index=frame.index)
    out["distance"] = frame["distance"]
    out["log_distance"] = np.log(frame["distance"])
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        out[col] = frame[col]
    out["delta_lat"] = frame["delivery_lat"] - frame["pickup_lat"]
    out["delta_lon"] = frame["delivery_lon"] - frame["pickup_lon"]
    out["equipment_code"] = pd.Categorical(frame["equipment"], categories=EQUIPMENT)
    out["weight"] = frame["weight"]
    out["weight_missing"] = frame["weight_missing"]
    out["market_index"] = frame["market_index"]
    out["market_index_daily"] = frame["market_index_daily"]
    out["quote_signal"] = frame["quote_signal"]
    out["day_of_week"] = frame["date"].dt.dayofweek
    return out[FEATURES]
