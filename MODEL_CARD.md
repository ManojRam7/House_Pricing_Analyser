# Model Card: Boston House Price Predictor

## Summary

| | |
|---|---|
| Task | Regression: median home value (`MEDV`, $1000s) |
| Model | `StandardScaler` + `RandomForestRegressor` (500 trees, max depth 14, min 2 samples per leaf) |
| Inputs | 13 base features + 2 interaction features (`RM_LSTAT`, `RM_AGE`) |
| Training data | `processed_housing_data.csv`, 490 rows |
| Evaluation | 20% random hold-out, `random_state=42` |

## Performance

| Metric | Hold-out value |
|---|---|
| R² | 0.870 |
| RMSE | 2.58 |
| MAE | 1.92 |

The training script rewrites `models/metrics.json` on every run.

## Intended use

Learning and demonstration of a complete regression workflow: analysis, feature engineering,
a reproducible training script and a web app.

## Out of scope

Property valuation, mortgage underwriting, insurance pricing or any decision about real people.

## Data preparation

- 16 records capped at `MEDV = 50.0` removed.
- Log and Yeo-Johnson transforms applied to skewed features in the notebook.
- Interaction features: `RM_LSTAT = RM x LSTAT`, `RM_AGE = RM x AGE`.

## Limitations

- Small, historic dataset (1970s census tracts); results do not transfer to today's market.
- A single random split; metrics move by a few points with a different seed.
- Some variables in the original dataset, notably `B`, encode race-based information and are
  ethically problematic. They are kept only to stay comparable with the standard benchmark.
