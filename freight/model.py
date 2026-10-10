"""Two-stage rate model: linear trend/market offset + LightGBM on the residual (log space).

Why two stages: after removing lane effects, the daily rate level is almost exactly
0.15 * log(market_index) + ~7%/year upward drift + a ~3.7% ramp through each quarter-end
month (Mar/Jun/Sep/Dec), with ~0.5% residual noise. Trees cannot extrapolate a time trend:
with market_index alone they confuse "low market in January" with "low market in
September" (a different price level), which produced a systematic ~3.6% under-prediction
on the Sep-Oct holdout. The linear stage carries trend, market level and the quarter-end
ramp; the GBM learns lane, distance, equipment, weight and quote_signal effects on top.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import CATEGORICAL, EQUIPMENT, build_features

EPOCH = pd.Timestamp("2025-01-01")

PARAMS = {
    "objective": "huber",
    "alpha": 0.15,  # huber delta in log space: robust to any label noise the filter misses
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 7,
}


OFFSETS = ("none", "market", "market+trend", "market+trend+quarter_end")


def quarter_end_progress(dates: pd.Series) -> np.ndarray:
    """0 outside Mar/Jun/Sep/Dec; inside them, the fraction of the month elapsed (0 -> 1).

    Rates ramp up ~3.7% across each quarter-end month and reset on the 1st of the next.
    """
    is_quarter_end = (dates.dt.month % 3 == 0).to_numpy()
    progress = ((dates.dt.day - 1) / (dates.dt.days_in_month - 1)).to_numpy(float)
    return np.where(is_quarter_end, progress, 0.0)


def _design(frame: pd.DataFrame, offset: str) -> np.ndarray:
    if offset == "none":
        return np.ones((len(frame), 1))
    cols = [
        np.ones(len(frame)),
        np.log(frame["market_index_daily"].to_numpy()),
        np.log(frame["distance"].to_numpy()),
    ]
    cols += [(frame["equipment"] == eq).to_numpy(float) for eq in EQUIPMENT[1:]]
    if offset.startswith("market+trend"):
        cols.append((frame["date"] - EPOCH).dt.days.to_numpy(float) / 365.0)
    if offset.endswith("quarter_end"):
        cols.append(quarter_end_progress(frame["date"]))
    return np.column_stack(cols)


class RateModel:
    def __init__(self, offset: str = "market+trend+quarter_end"):
        assert offset in OFFSETS
        self.offset_kind = offset
        self.coef: np.ndarray | None = None
        self.booster: lgb.Booster | None = None

    def offset(self, frame: pd.DataFrame) -> np.ndarray:
        return _design(frame, self.offset_kind) @ self.coef

    def fit(self, frame: pd.DataFrame, num_rounds: int):
        y = np.log(frame["posted_rate"].to_numpy())
        self.coef = np.linalg.lstsq(_design(frame, self.offset_kind), y, rcond=None)[0]
        train_set = lgb.Dataset(
            build_features(frame), y, init_score=self.offset(frame), categorical_feature=CATEGORICAL
        )
        self.booster = lgb.train(PARAMS, train_set, num_boost_round=num_rounds)
        return self

    def predict(self, frame: pd.DataFrame, num_rounds: int | None = None) -> np.ndarray:
        gbm = self.booster.predict(build_features(frame), num_iteration=num_rounds)
        return np.exp(self.offset(frame) + gbm)


def metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(np.mean(np.abs(err) / actual) * 100),
        "MedAPE_%": float(np.median(np.abs(err) / actual) * 100),
        "bias_%": float(np.mean(np.log(predicted / actual)) * 100),
    }
