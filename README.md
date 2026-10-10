# Freight Rate Prediction

Predicts `posted_rate`, the price of a truckload shipment, for Spotter's ML Engineer assessment.

## Run

Put Spotter's input files (`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`,
`december_chart_inputs.csv`) in a `data/` folder, or pass their location with `--data-dir`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py --data-dir data
python score.py --predictions outputs/validation_predictions.csv \
    --december-predictions outputs/december_chart_inputs.csv --output-dir outputs
```

## Outputs (`outputs/`)

| File | Content |
|---|---|
| `validation_predictions.csv` | Predicted rate for each of the 12,000 validation loads |
| `december_chart_inputs.csv` | The 31 December rows with `predicted_rate` filled in |
| `metrics.json` | Backtest results for every model variant and baseline |
| `candidate_december.png` | December chart created by `score.py` |

## Approach

Full write-up with figures: **[APPROACH.pdf](APPROACH.pdf)**.

In short:
- Removed 1.4% of training loads whose prices were corrupted.
- Prices are explained by the market index, a ~7%/yr trend and a ~4% ramp through each quarter-end month.
- The model has two stages: a linear formula for those effects, then LightGBM for lane, weight and `quote_signal` effects.
- Validation uses a time-based rolling backtest.

| Sep–Oct holdout | MAPE | MAE |
|---|---|---|
| Lane average $/mile (baseline) | 3.74% | $88 |
| LightGBM only | 3.81% | $90 |
| **Final two-stage model** | **1.61%** | **$38** |

## Code

- `freight/data.py`: loading and cleaning
- `freight/features.py`: model inputs
- `freight/model.py`: two-stage model and metrics
- `run.py`: the full pipeline
- `score.py`: Spotter's scorer, unchanged
