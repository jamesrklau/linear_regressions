from regression_toolkit.evaluation import evaluate_models
from regression_toolkit.models import predictions, build_preprocessor
from regression_toolkit.cli import main, get_args
from regression_toolkit.data import load_dataset

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import pytest


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

    train_X = X.iloc[:30]
    train_y = y.iloc[:30]
    val_X = X.iloc[30:]
    val_y = y.iloc[30:]

    results, prediction_results = evaluate_models(
        train_X,
        train_y,
        val_X,
        val_y,
    )

    # Check the evaluation metrics
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

    # Check the prediction results
    assert "actual" in prediction_results.columns
    assert len(prediction_results) == len(val_y)
    assert prediction_results["actual"].equals(val_y)

    for model_name in results["Model"]:
        assert model_name in prediction_results.columns



def test_evaluation_metrics_are_finite():
    X, y = make_sample_data()

    results, prediction_results = evaluate_models(
        X.iloc[:30],
        y.iloc[:30],
        X.iloc[30:],
        y.iloc[30:],
    )

    metrics = results[["MAE", "RMSE", "R2"]].to_numpy()

    assert np.isfinite(metrics).all()


def test_results_are_sorted_by_mae():
    X, y = make_sample_data()

    results, prediction_results = evaluate_models(
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



def test_build_preprocessor_rejects_unsupported_feature_types():

    X = pd.DataFrame({"unsupported": pd.to_timedelta([1, 2], unit="D")})

    with pytest.raises(ValueError, match="numeric or categorical"):
        build_preprocessor(X)

def test_main_rejects_missing_target_column(monkeypatch, tmp_path):
    csv_file = tmp_path / "sample.csv"
    pd.DataFrame({"feature": [1, 2, 3, 4]}).to_csv(
        csv_file, index=False
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "linear_regressions.py",
            "--csv",
            str(csv_file),
            "--var_name",
            "missing_target",
        ],
    )

    with pytest.raises(ValueError, match="was not found"):
        main()


def test_main_rejects_non_numeric_target(monkeypatch, tmp_path):
    csv_file = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": ["low", "medium", "high", "low"],
        }
    ).to_csv(csv_file, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "linear_regressions.py",
            "--csv",
            str(csv_file),
            "--var_name",
            "target",
        ],
    )

    with pytest.raises(ValueError, match="must contain numeric"):
        main()

def test_load_dataset_returns_features_and_target(tmp_path):
    csv_file = tmp_path / "sample.csv"

    pd.DataFrame(
        {
            "hours": [2.0, 4.0, 6.0, 8.0],
            "school": ["A", "B", "A", "B"],
            "score": [50.0, 60.0, 70.0, 80.0],
        }
    ).to_csv(csv_file, index=False)

    X, y = load_dataset(csv_file, "score")

    assert list(X.columns) == ["hours", "school"]
    assert y.name == "score"
    assert len(X) == 4
    assert len(y) == 4


def test_load_dataset_rejects_infinite_target(tmp_path):
    csv_file = tmp_path / "sample.csv"

    pd.DataFrame(
        {
            "hours": [2.0, 4.0, 6.0, 8.0],
            "score": [50.0, 60.0, float("inf"), 80.0],
        }
    ).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="finite numeric values"):
        load_dataset(csv_file, "score")

def test_cli_default_arguments(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "linear_regressions.py",
            "--csv",
            "sample.csv",
            "--var_name",
            "target",
        ],
    )

    args = get_args()

    assert args.test_size == 0.25
    assert args.random_state == 0
    assert args.model_random_state == 1


def test_cli_custom_arguments(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "linear_regressions.py",
            "--csv",
            "sample.csv",
            "--var_name",
            "target",
            "--test_size",
            "0.30",
            "--random_state",
            "42",
            "--model_random_state",
            "42",
        ],
    )

    args = get_args()

    assert args.test_size == 0.30
    assert args.random_state == 42
    assert args.model_random_state == 42

@pytest.mark.parametrize("test_size", ["0", "1", "1.5", "-0.2"])
def test_cli_rejects_invalid_test_size(monkeypatch, test_size):
    monkeypatch.setattr(
        "sys.argv",
        [
            "linear_regressions.py",
            "--csv",
            "sample.csv",
            "--var_name",
            "target",
            "--test_size",
            test_size,
        ],
    )

    with pytest.raises(SystemExit):
        get_args()


def test_evaluation_is_reproducible():
    X, y = make_sample_data()

    train_X = X.iloc[:30]
    train_y = y.iloc[:30]
    val_X = X.iloc[30:]
    val_y = y.iloc[30:]

    first_results, first_predictions = evaluate_models(
        train_X,
        train_y,
        val_X,
        val_y,
        model_random_state=42,
    )

    second_results, second_predictions = evaluate_models(
        train_X,
        train_y,
        val_X,
        val_y,
        model_random_state=42,
    )

    pd.testing.assert_frame_equal(
        first_results,
        second_results,
    )

    pd.testing.assert_frame_equal(
        first_predictions,
        second_predictions,
    )


@pytest.mark.parametrize(
    "option",
    ["--random_state", "--model_random_state"],
)
@pytest.mark.parametrize("value", ["-1", "abc", "2.5"])
def test_cli_rejects_invalid_random_states(
    monkeypatch, option, value
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "linear_regressions.py",
            "--csv",
            "sample.csv",
            "--var_name",
            "target",
            option,
            value,
        ],
    )

    with pytest.raises(SystemExit):
        get_args()