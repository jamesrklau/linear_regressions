
import numpy as np
import pandas as pd

from linear_regressions import evaluate_models


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
