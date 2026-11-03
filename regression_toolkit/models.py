import pandas as pd
from pandas.api.types import is_numeric_dtype, is_timedelta64_dtype

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

