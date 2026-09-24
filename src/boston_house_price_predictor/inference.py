from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, MODEL_PATH
from .features import add_engineered_features, transform_raw_features


class MissingArtifactError(FileNotFoundError):
    """Raised when model artifact does not exist."""


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise MissingArtifactError(
            f"Model artifact not found at {model_path}. Run scripts/train_model.py first."
        )
    return joblib.load(model_path)


def predict_price(inputs: dict[str, Any], model_path: Path = MODEL_PATH) -> float:
    """Predict MEDV ($1000s) from the 13 raw Boston housing features."""
    missing = [col for col in BASE_FEATURE_COLUMNS if col not in inputs]
    if missing:
        raise ValueError(f"Missing required input fields: {', '.join(missing)}")

    input_df = pd.DataFrame([inputs], columns=BASE_FEATURE_COLUMNS)
    input_df = transform_raw_features(input_df)   # raw values -> training scale
    input_df = add_engineered_features(input_df)
    features = input_df[FEATURE_COLUMNS]

    model = load_model(model_path=model_path)
    prediction = model.predict(features)[0]
    return float(prediction)
