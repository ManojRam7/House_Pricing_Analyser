# Boston House Price Predictor

Predicts the median value of a home (`MEDV`, in $1000s) from 13 neighbourhood and property features
of the Boston housing dataset. The notebook covers the full analysis; the `src/` package turns the
final approach into a reproducible training script and a Streamlit app.

**Live app:** https://boston-houseprice-predictor.streamlit.app

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## Results

Random forest (500 trees) on a 20% hold-out set, from `models/metrics.json`:

| Metric | Value |
|---|---|
| R² | **0.870** |
| RMSE | **2.58** ($2,580) |
| MAE | **1.92** ($1,920) |
| Rows after cleaning | 490 |

In the notebook, adding the two interaction features lifted R² from 0.798 to 0.836 and cut RMSE
from 3.22 to 2.89 on the same split.

## Approach

1. **EDA** (`House Price analysis.ipynb`): distributions, correlation matrix, relationships with
   `MEDV` and outlier analysis. The 16 capped records at `MEDV = 50.0` were removed.
2. **Transformations**: log and Yeo-Johnson power transforms for the skewed features (`CRIM`, `ZN`,
   `B` and others), then a second correlation check.
3. **Feature selection**: random forest importances show `LSTAT` and `RM` dominate, which led to
   two interaction features: `RM_LSTAT = RM x LSTAT` and `RM_AGE = RM x AGE`.
4. **Model**: `StandardScaler` + `RandomForestRegressor` in one scikit-learn pipeline, trained and
   saved by `scripts/train_model.py`, which also writes the metrics file.
5. **App**: `streamlit_app.py` collects the 13 inputs, builds the interaction features and returns
   an estimated price. If the model file is missing, the app can run the training script.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py      # trains, saves models/model_pipeline.joblib and metrics.json
streamlit run streamlit_app.py
```

## Project structure

```text
House Price analysis.ipynb        EDA, transformations, feature selection, model comparison
processed_housing_data.csv        cleaned dataset used for training
scripts/train_model.py            training entry point
src/boston_house_price_predictor/
    config.py                     paths and feature lists
    data.py                       loading and schema validation
    features.py                   interaction features
    modeling.py                   scikit-learn pipeline
    train.py                      train, evaluate, save
    inference.py                  load the model and predict
models/                           saved pipeline and metrics
streamlit_app.py                  web app
MODEL_CARD.md                     intended use and limitations
```

## Features

| Feature | Description |
|---|---|
| CRIM | Per-capita crime rate by town |
| ZN | Share of residential land zoned for lots over 25,000 sq ft |
| INDUS | Share of non-retail business acres |
| CHAS | 1 if the tract bounds the Charles River |
| NOX | Nitric oxide concentration |
| RM | Average rooms per dwelling |
| AGE | Share of owner-occupied units built before 1940 |
| DIS | Weighted distance to five Boston employment centres |
| RAD | Accessibility index to radial highways |
| TAX | Property-tax rate per $10,000 |
| PTRATIO | Pupil-teacher ratio |
| B | Legacy demographic index from the original dataset |
| LSTAT | Share of lower-status population |

## Limitations

The dataset is small (490 rows) and dates from the 1970s, and some of its socio-economic
variables, `B` in particular, are ethically problematic. The model is a learning exercise in
regression and deployment, not a tool for valuation, lending or policy decisions.
