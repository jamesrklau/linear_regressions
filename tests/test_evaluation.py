
import numpy as np
import pandas as pd

from linear_regressions import evaluate_models
from linear_regressions import predictions
from sklearn.linear_model import Ridge

def make_sample_data():
    """Create a small, predictable dataset for testing."""
    X = pd.DataFrame(
        {
            "feature_a": np.arange(40, dtype=float),
            "feature_b": np.arange(40, dtype=float) * 0.5,
        }
    )
    y = 10 + 2 * X["feature_a"] - X["feature_b"]

    return X, y


def test_evaluation_returns_all_expected_models_and_metrics():
    X, y = make_sample_data()

    results = evaluate_models(
        X.iloc[:30],
        y.iloc[:30],
        X.iloc[30:],
        y.iloc[30:],
    )

    assert len(results) == 6
    assert set(results["Model"]) == {
        "Baseline (Mean)",
        "AdaBoost",
        "Gradient Boosting",
        "Ridge",
        "Lasso",
        "Elastic Net",
    }
    assert {"MAE", "RMSE", "R2"}.issubset(results.columns)


def test_evaluation_metrics_are_finite():
    X, y = make_sample_data()

    results = evaluate_models(
        X.iloc[:30],
        y.iloc[:30],
        X.iloc[30:],
        y.iloc[30:],
    )

    metrics = results[["MAE", "RMSE", "R2"]].to_numpy()

    assert np.isfinite(metrics).all()


def test_results_are_sorted_by_mae():
    X, y = make_sample_data()

    results = evaluate_models(
        X.iloc[:30],
        y.iloc[:30],
        X.iloc[30:],
        y.iloc[30:],
    )

    assert results["MAE"].is_monotonic_increasing


def test_predictions_handle_missing_and_categorical_features():
    train_X = pd.DataFrame(
        {
            "study_hours": [2.0, 4.0, None, 8.0, 10.0, 12.0],
            "school_type": [
                "public",
                "private",
                "public",
                None,
                "private",
                "public",
            ],
        }
    )
    train_y = pd.Series([50.0, 60.0, 55.0, 75.0, 80.0, 85.0])

    prediction_X = pd.DataFrame(
        {
            "study_hours": [6.0, None],
            "school_type": ["charter", "private"],
        }
    )

    result = predictions(
        Ridge, train_X, train_y, prediction_X
    )

    assert len(result) == 2
    assert np.isfinite(result).all()

