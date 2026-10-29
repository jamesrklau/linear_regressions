
"""Compare regression models on a shared validation dataset."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def models_list():
    """Return the regression models to compare."""
    return [
        ("Baseline (Mean)", DummyRegressor),
        ("AdaBoost", AdaBoostRegressor),
        ("Gradient Boosting", GradientBoostingRegressor),
        ("Ridge", Ridge),
        ("Lasso", Lasso),
        ("Elastic Net", ElasticNet),
    ]


def predictions(model, train_X, train_y, prediction_X):
    """Fit a model on training data and predict supplied observations."""
    model_instance = model()

    # Make stochastic models reproducible when supported.
    if "random_state" in model_instance.get_params():
        model_instance.set_params(random_state=1)

    model_instance.fit(train_X, train_y)
    return model_instance.predict(prediction_X)


def evaluate_models(train_X, train_y, val_X, val_y):
    """Train and evaluate each model on the same validation observations."""
    results = []

    for name, model in models_list():
        val_predictions = predictions(
            model, train_X, train_y, val_X
        )

        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(val_y, val_predictions),
                "RMSE": mean_squared_error(
                    val_y, val_predictions
                ) ** 0.5,
                "R2": r2_score(val_y, val_predictions),
            }
        )

    return pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare regression models on a CSV dataset."
    )
    parser.add_argument(
        "-C", "--csv", type=Path, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "-V", "--var_name", required=True, help="Name of the target column"
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Load the dataset.
    data = pd.read_csv(args.csv)

    if args.var_name not in data.columns:
        raise ValueError(
            f"Target column '{args.var_name}' was not found in the CSV."
        )

    # Remove rows where the target is missing.
    data = data.dropna(subset=[args.var_name]).reset_index(drop=True)

    # Separate the target from numerical input features.
    y = data[args.var_name]
    X = data.drop(columns=[args.var_name]).select_dtypes(include="number")

    if X.empty:
        raise ValueError("The dataset must contain numerical feature columns.")

    if len(data) < 4:
        raise ValueError("At least four rows with a target value are required.")

    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("The target column must contain numeric values.")

    if y.isna().any():
        raise ValueError("The target column contains missing values.")

    # Use one fixed split so every model sees the same training and validation rows.
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # Evaluate all models using only the held-out validation data.
    results = evaluate_models(train_X, train_y, val_X, val_y)

    print("\nRegression Model Comparison")
    print(f"Target: {args.var_name}")
    print(f"Training rows: {len(train_X)}")
    print(f"Validation rows: {len(val_X)}\n")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # Save the evaluation report beside the input CSV.
    output_file = args.csv.with_name(
        f"{args.csv.stem}_model_comparison.csv"
    )
    results.to_csv(output_file, index=False)

    print(f"\nSaved model comparison to: {output_file}")


if __name__ == "__main__":
    main()
