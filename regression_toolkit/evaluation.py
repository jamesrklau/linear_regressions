import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from regression_toolkit.models import models_list, predictions

def evaluate_models(train_X, train_y, val_X, val_y, model_random_state=1):
    """Train and evaluate each model on the same validation observations."""
    results = []

    for name, model in models_list():
        val_predictions = predictions(
            model, train_X, train_y, val_X, random_state=model_random_state
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
