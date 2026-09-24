from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LOG1P_COLUMNS, YEO_JOHNSON_PARAMS


def _yeo_johnson(x: np.ndarray, lmbda: float) -> np.ndarray:
    """Yeo-Johnson transform for non-negative inputs (all inputs here are >= 0)."""
    if abs(lmbda) < 1e-12:
        return np.log1p(x)
    return (np.power(x + 1.0, lmbda) - 1.0) / lmbda


def transform_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's skew corrections to raw inputs so they match the training data."""
    output = df.copy()
    for col in LOG1P_COLUMNS:
        output[col] = np.log1p(output[col].astype(float))

    output["B"] = 1.0 / output["B"].astype(float).clip(lower=0.01)
    for col, pre in (("CRIM", np.log1p), ("ZN", np.log1p), ("B", np.log1p)):
        lmbda, mean, std = YEO_JOHNSON_PARAMS[col]
        values = pre(output[col].astype(float).clip(lower=0).to_numpy())
        output[col] = (_yeo_johnson(values, lmbda) - mean) / std
    return output


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered interaction features used during training and inference."""
    output = df.copy()
    output["RM_LSTAT"] = output["RM"] * output["LSTAT"]
    output["RM_AGE"] = output["RM"] * output["AGE"]
    return output
