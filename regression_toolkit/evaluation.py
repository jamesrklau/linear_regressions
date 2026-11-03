
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from regression_toolkit.models import models_list, predictions


def evaluate_models(
    train_X,
    train_y,
    val_X,
    val_y,
    model_random_state=1,
):
    results = []
    prediction_results = pd.DataFrame(
        {
            "actual": val_y.to_numpy(),
        },
        index=val_y.index,
    )

    for model_name, model in models_list():
        model_predictions = predictions(
            model,
            train_X,
            train_y,
            val_X,
            random_state=model_random_state,
        )

        mae = mean_absolute_error(val_y, model_predictions)
        rmse = mean_squared_error(
            val_y,
            model_predictions,
        ) ** 0.5
        r2 = r2_score(val_y, model_predictions)

        results.append(
            {
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        prediction_results[model_name] = model_predictions

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE").reset_index(drop=True)

    return results_df, prediction_results
