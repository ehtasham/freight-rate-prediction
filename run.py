"""End-to-end pipeline: clean -> label-noise filter -> time-based validation -> final fit -> predictions.

Usage:
    python run.py --data-dir path/to/spotter/csvs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from freight import data
from freight.model import OFFSETS, RateModel, metrics

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
ROUND_GRID = (100, 200, 300, 400, 600, 800, 1200, 1600)
CHOSEN = "market+trend+quarter_end"

# Rolling-origin backtest: each fold trains on everything up to fit_end and tests on the next
# two months, mirroring the real task (train Jan-Oct, predict Nov-Dec). The last fold is the
# primary holdout.
FOLDS = [
    ("2025-04-30", "2025-06-30"),
    ("2025-06-30", "2025-08-31"),
    ("2025-08-31", "2025-10-31"),
]


def find_label_outliers(train: pd.DataFrame) -> pd.Series:
    """5-fold out-of-fold predictions; a label is an outlier if it is >1.6x off its OOF prediction."""
    oof = pd.Series(np.nan, index=train.index)
    for fit_idx, oof_idx in KFold(5, shuffle=True, random_state=0).split(train):
        m = RateModel(CHOSEN).fit(train.iloc[fit_idx], num_rounds=800)
        oof.iloc[oof_idx] = m.predict(train.iloc[oof_idx])
    return data.flag_label_outliers(train["posted_rate"], oof)


def baselines(fit_part: pd.DataFrame, test_part: pd.DataFrame) -> dict:
    """Simple rate-per-mile lookups, to show what the model adds."""
    rpm = fit_part["posted_rate"] / fit_part["distance"]
    eq_rpm = rpm.groupby(fit_part["equipment"]).median()
    lane_rpm = rpm.groupby([fit_part["pickup"], fit_part["delivery"], fit_part["equipment"]]).median()
    eq_pred = test_part["equipment"].map(eq_rpm) * test_part["distance"]
    keys = pd.MultiIndex.from_frame(test_part[["pickup", "delivery", "equipment"]])
    lane_pred = pd.Series(lane_rpm.reindex(keys).to_numpy(), index=test_part.index) * test_part["distance"]
    lane_pred = lane_pred.fillna(eq_pred)
    actual = test_part["posted_rate"]
    return {
        "baseline_equipment_median_rpm": metrics(actual, eq_pred),
        "baseline_lane_median_rpm": metrics(actual, lane_pred),
    }


def evaluate_fold(clean: pd.DataFrame, raw: pd.DataFrame, fit_end: str, test_end: str) -> dict:
    """Fit on dates <= fit_end and score the next two months at every ROUND_GRID size."""
    fit_part = clean[clean["date"] <= fit_end]
    test_part = clean[(clean["date"] > fit_end) & (clean["date"] <= test_end)]
    raw_test = raw[(raw["date"] > fit_end) & (raw["date"] <= test_end)]
    result = {
        "fit_window": f"{fit_part['date'].min().date()} .. {fit_part['date'].max().date()}",
        "test_window": f"{test_part['date'].min().date()} .. {test_part['date'].max().date()}",
        "n_fit": len(fit_part),
        "n_test": len(test_part),
        "by_rounds": {},
    }
    for offset in OFFSETS:
        m = RateModel(offset).fit(fit_part, max(ROUND_GRID))
        result["by_rounds"][offset] = {
            n: metrics(test_part["posted_rate"], m.predict(test_part, n)) for n in ROUND_GRID
        }
        if offset == CHOSEN:
            result["by_rounds"][f"{offset} (raw labels)"] = {
                n: metrics(raw_test["posted_rate"], m.predict(raw_test, n)) for n in ROUND_GRID
            }
    result.update(baselines(fit_part, test_part))
    return result


def best_rounds(backtest: list[dict], offset: str) -> int:
    """Round count with the lowest MAPE averaged over the backtest folds."""
    mean_mape = {n: np.mean([f["by_rounds"][offset][n]["MAPE_%"] for f in backtest]) for n in ROUND_GRID}
    return min(mean_mape, key=mean_mape.get)


def summarise(backtest: list[dict]) -> dict:
    """Per-fold metrics for every variant at its own backtest-selected round count."""
    rounds = {name: best_rounds(backtest, name) for name in backtest[0]["by_rounds"]}
    folds = []
    for fold in backtest:
        row = {k: fold[k] for k in ["fit_window", "test_window", "n_fit", "n_test"]}
        for name, n in rounds.items():
            row[f"model[{name}] @{n}"] = fold["by_rounds"][name][n]
        row.update({k: v for k, v in fold.items() if k.startswith("baseline")})
        folds.append(row)
    return {"rounds": rounds, "folds": folds}


def december_inputs(train: pd.DataFrame, validation: pd.DataFrame, chart: pd.DataFrame) -> pd.DataFrame:
    """Complete the December chart rows with the features the model needs.

    Coordinates come from the training data for Lexington and Fort Wayne. market_index and
    quote_signal are not given for these rows, so each day uses the median observed on that
    same date across the validation loads (these are model inputs, not labels).
    quote_signal is taken from short-haul loads (200-500 mi) because its spread depends on
    distance.
    """
    frame = chart.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    pickups = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    deliveries = train.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    frame[["pickup_lat", "pickup_lon"]] = pickups.loc[frame["pickup"]].to_numpy()
    frame[["delivery_lat", "delivery_lon"]] = deliveries.loc[frame["delivery"]].to_numpy()

    day_market = validation.groupby("date")["market_index"].median()
    short = validation[validation["distance"].between(200, 500)]
    day_quote = short.groupby("date")["quote_signal"].median()
    frame["market_index"] = frame["date"].map(day_market)
    frame["quote_signal"] = frame["date"].map(day_quote)
    return data.clean_inputs(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="folder with Spotter's input CSVs (default: ./data)")
    data_dir = parser.parse_args().data_dir
    OUTPUTS.mkdir(exist_ok=True)
    train = data.clean_inputs(data.load(data_dir, "train_test.csv"))
    validation = data.clean_inputs(data.load(data_dir, "validation.csv"))

    # 1. Label noise
    outliers = find_label_outliers(train)
    clean = train[~outliers].copy()
    print(f"Label outliers removed: {outliers.sum()} of {len(train)} ({outliers.mean():.2%})")

    # 2. Time-based validation; the number of boosting rounds is chosen here too.
    backtest = summarise([evaluate_fold(clean, train, *fold) for fold in FOLDS])
    for fold in backtest["folds"]:
        print(f"\nTest {fold['test_window']}  (MAPE % / MAE $)")
        for name, m in fold.items():
            if isinstance(m, dict):
                print(f"  {name:55s} {m['MAPE_%']:5.2f}%  ${m['MAE']:6.1f}")

    # 3. Final model on all clean data (Jan-Oct)
    rounds = backtest["rounds"][CHOSEN]
    final = RateModel(CHOSEN).fit(clean, rounds)

    # 4. Validation predictions in template order
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    preds = pd.Series(final.predict(validation), index=validation["load_id"])
    template["predicted_rate"] = preds.reindex(template["load_id"]).round(2).to_numpy()
    template.to_csv(OUTPUTS / "validation_predictions.csv", index=False)

    # 5. December chart rows (keep the original seven columns)
    chart = pd.read_csv(data_dir / "december_chart_inputs.csv")
    chart_features = december_inputs(train, validation, chart)
    chart["predicted_rate"] = final.predict(chart_features).round(2)
    chart.to_csv(OUTPUTS / "december_chart_inputs.csv", index=False)

    summary = {
        "label_outliers_removed": int(outliers.sum()),
        "final_model": {"offset": CHOSEN, "boosting_rounds": rounds, "training_loads": len(clean)},
        "backtest_rounds_by_variant": backtest["rounds"],
        "backtest": backtest["folds"],
    }
    (OUTPUTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"Final model: {rounds} boosting rounds, trained on {len(clean)} clean loads")
    print("Wrote outputs/validation_predictions.csv, outputs/december_chart_inputs.csv, outputs/metrics.json")

if __name__ == "__main__":
    main()
