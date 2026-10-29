
"""Compare regression models on a shared validation dataset."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import is_numeric_dtype, is_timedelta64_dtype


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



def build_preprocessor(X):
    """Build preprocessing steps based on feature column types."""

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
            "Unsupported feature column types: "
            f"{unsupported_features}. "
            "Use numeric or categorical feature columns."
        )

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    if not transformers:
        raise ValueError(
            "The dataset must contain numeric or categorical feature columns."
        )

    return ColumnTransformer(transformers=transformers)

def predictions(model, train_X, train_y, prediction_X):
    """Fit a complete preprocessing and prediction pipeline."""
    model_instance = model()

    # Make stochastic models reproducible when supported.
    if "random_state" in model_instance.get_params():
        model_instance.set_params(random_state=1)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(train_X)),
            ("model", model_instance),
        ]
    )

    pipeline.fit(train_X, train_y)
    return pipeline.predict(prediction_X)


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

    if not args.csv.is_file():
        raise FileNotFoundError(f"Input CSV file not found: {args.csv}")

    data = pd.read_csv(args.csv)

    if args.var_name not in data.columns:
        raise ValueError(
            f"Target column '{args.var_name}' was not found in the CSV."
        )

    # Remove rows where the target is missing.
    data = data.dropna(subset=[args.var_name]).reset_index(drop=True)

    if len(data) < 4:
        raise ValueError("At least four rows with a target value are required.")

    y = data[args.var_name]

    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("The target column must contain numeric values.")

    X = data.drop(columns=[args.var_name])

    if X.empty:
        raise ValueError("The dataset must contain feature columns.")

    if not (
        X.select_dtypes(include="number").shape[1]
        or X.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).shape[1]
    ):
        raise ValueError(
            "No supported feature columns found. "
            "Use numeric or categorical features."
        )

    if not y.map(lambda value: pd.notna(value) and value != float("inf")
                 and value != float("-inf")).all():
        raise ValueError("The target column must contain finite numeric values.")

    # Use one fixed split for a fair model comparison.
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    results = evaluate_models(train_X, train_y, val_X, val_y)

    print("\nRegression Model Comparison")
    print(f"Target: {args.var_name}")
    print(f"Training rows: {len(train_X)}")
    print(f"Validation rows: {len(val_X)}\n")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    output_file = args.csv.with_name(
        f"{args.csv.stem}_model_comparison.csv"
    )
    results.to_csv(output_file, index=False)

    print(f"\nSaved model comparison to: {output_file}")


if __name__ == "__main__":
    main()
