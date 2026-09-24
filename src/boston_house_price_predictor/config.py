from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "processed_housing_data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

TARGET_COLUMN = "MEDV"
BASE_FEATURE_COLUMNS = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]
ENGINEERED_FEATURE_COLUMNS = ["RM_LSTAT", "RM_AGE"]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

# Transforms applied to the raw features in `House Price analysis.ipynb` before
# processed_housing_data.csv was saved. The model is trained on the transformed
# values, so raw user inputs must go through the same steps before prediction.
#   CRIM, ZN : log1p, then Yeo-Johnson (standardised)
#   B        : 1/B, log1p, then Yeo-Johnson (standardised)
#   DIS, LSTAT: log1p
# (lambda, mean, std) are the parameters fitted by sklearn's PowerTransformer.
YEO_JOHNSON_PARAMS = {
    "CRIM": (-1.5626767956411753, 0.2518656002738145, 0.20146993662954119),
    "ZN": (-2.1893748979240875, 0.11571357048711044, 0.19358821863660539),
    "B": (-141.18249281423908, 0.002515877531955674, 0.001133702503647317),
}
LOG1P_COLUMNS = ["DIS", "LSTAT"]
