"""Functions for loading and validating regression datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_timedelta64_dtype


def load_dataset(
    csv_path: str | Path,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV and return validated features and target values."""

    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)

    if data.empty:
        raise ValueError("The input CSV contains no data rows.")

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the CSV."
        )

    # Rows without a target cannot be used for supervised learning.
    data = data.dropna(subset=[target_column]).reset_index(drop=True)

    if len(data) < 4:
        raise ValueError(
            "At least four rows with a target value are required."
        )

    y = data[target_column]

    if not is_numeric_dtype(y):
        raise ValueError("The target column must contain numeric values.")

    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError(
            "The target column must contain finite numeric values."
        )

    X = data.drop(columns=[target_column])

    if X.empty:
        raise ValueError("The dataset must contain feature columns.")

    # Identify feature columns supported by our preprocessing pipeline.
    numeric_features = [
        column
        for column in X.columns
        if is_numeric_dtype(X[column].dtype)
        and not is_timedelta64_dtype(X[column].dtype)
    ]

    categorical_features = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    supported_features = set(numeric_features + categorical_features)
    unsupported_features = [
        column for column in X.columns
        if column not in supported_features
    ]

    if unsupported_features:
        raise ValueError(
            f"Unsupported feature column types: {unsupported_features}. "
            "Use numeric or categorical feature columns."
        )

    if not supported_features:
        raise ValueError(
            "The dataset must contain numeric or categorical feature columns."
        )

    return X, y