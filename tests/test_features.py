"""Raw inputs must map onto the same scale as processed_housing_data.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boston_house_price_predictor.config import BASE_FEATURE_COLUMNS, DATA_PATH, FEATURE_COLUMNS
from boston_house_price_predictor.features import add_engineered_features, transform_raw_features

# First and last rows of the original housing.csv, as printed in the notebook.
RAW_ROWS = pd.DataFrame(
    [
        [0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.0900, 1, 296.0, 15.3, 396.90, 4.98],
        [0.02731, 0.0, 7.07, 0, 0.469, 6.421, 78.9, 4.9671, 2, 242.0, 17.8, 396.90, 9.14],
        [0.02729, 0.0, 7.07, 0, 0.469, 7.185, 61.1, 4.9671, 2, 242.0, 17.8, 392.83, 4.03],
        [0.03237, 0.0, 2.18, 0, 0.458, 6.998, 45.8, 6.0622, 3, 222.0, 18.7, 394.63, 2.94],
        [0.06905, 0.0, 2.18, 0, 0.458, 7.147, 54.2, 6.0622, 3, 222.0, 18.7, 396.90, 5.33],
        [0.04741, 0.0, 11.93, 0, 0.573, 6.030, 80.8, 2.5050, 1, 273.0, 21.0, 396.90, 7.88],
    ],
    columns=BASE_FEATURE_COLUMNS,
)


def test_raw_rows_match_processed_training_data():
    processed = pd.read_csv(DATA_PATH)
    expected = pd.concat([processed.head(5), processed.tail(1)])[FEATURE_COLUMNS].to_numpy()
    got = add_engineered_features(transform_raw_features(RAW_ROWS))[FEATURE_COLUMNS].to_numpy()
    assert np.allclose(got, expected, atol=1e-6)


if __name__ == "__main__":
    test_raw_rows_match_processed_training_data()
    print("PASS")
