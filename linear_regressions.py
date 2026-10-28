# Importing necessary libraries
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
import pandas as pd
import argparse
import os

def models_list():
    # List of models to be used
    models = []
    models.append(('ab', AdaBoostRegressor))
    models.append(('gb', GradientBoostingRegressor))
    models.append(('ridge', Ridge))
    models.append(('lass', Lasso))
    models.append(('enet', ElasticNet))
    return models


def predictions(model, train_X, train_y, prediction_X):
    """Fit a regression model and predict values for the supplied data."""

    model_instance = model()

    # Set a fixed random seed when the model supports it.
    if "random_state" in model_instance.get_params():
        model_instance.set_params(random_state=1)

    model_instance.fit(train_X, train_y)

    return model_instance.predict(prediction_X)

         
def main():
    # Parse command-line arguments
    argv = get_args()
    
    # Read regression data from CSV file
    regression_data = pd.read_csv(argv.csv)
    
    # Remove rows with null values in the specified variable column
    regression_data = regression_data[regression_data[argv.var_name].notnull()].reset_index(drop=True)
    
    # Separate target variable and feature columns
    y = regression_data[argv.var_name]
    x_table = regression_data.drop(argv.var_name, axis=1)
    
    # Select only numerical feature columns
    x_table = x_table.select_dtypes(['number'])
    X = x_table
    
    # Split data into training and validation sets
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    # Spot Check Algorithms
    models = models_list()


    for name, model in models:
        # Evaluate the model on held-out validation data
        val_predictions = predictions(
            model, train_X, train_y, val_X
        )
        val_mae = mean_absolute_error(val_y, val_predictions)

        print(f"{name} validation MAE: {val_mae:.4f}")

        # Fit on the full dataset to generate BPI scores for all rows
        full_predictions = predictions(model, X, y, X)

        # Standardize predictions into z-scores
        regression_data[name] = stats.zscore(full_predictions)

    # Save the completed results once, after all models finish
    output_file = f"{os.path.splitext(argv.csv)[0]}_bpi.csv"
    regression_data.to_csv(output_file, index=False)

    print(f"Saved BPI results to: {output_file}")


def get_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculates LAT BPI")
    
    parser.add_argument("-C", "--csv", type=str, required=True, help="CSV Table")
    parser.add_argument("-V", "--var_name", type=str, required=True, help="Var column name")
    
    argv = parser.parse_args()
    return argv

if __name__ == "__main__":
    main()
    print("Done")