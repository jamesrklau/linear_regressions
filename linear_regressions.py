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
    models = []
    models.append(('ab', AdaBoostRegressor))
    models.append(('gb', GradientBoostingRegressor))
    models.append(('ridge', Ridge))
    models.append(('lass', Lasso))
    models.append(('enet', ElasticNet))
    return models

def predictions(model, data_X, data_y):
    selected_model = model(random_state=1)
    selected_model.fit(data_X, data_y)
    val_predictions = selected_model.predict(data_X)
    return val_predictions
         

def main():
    argv = get_args()
    regression_data = pd.read_csv(argv.csv)
    regression_data = regression_data[regression_data[argv.var_name].notnull()].reset_index(drop=True)
    y = regression_data[argv.var_name]
    x_table = regression_data.drop(argv.var_name, axis =1)
    x_table = x_table.select_dtypes(['number'])
    X = x_table
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

    # Spot Check Algorithms
    models = models_list()

    for name, model in models:
        val_predictions = predictions(model, X, y)
        print(mean_absolute_error(y, val_predictions))
        regression_data[name] = stats.zscore(val_predictions)
        regression_data.to_csv(f"{os.path.splitext(argv.csv)[0]}_bpi.csv", index=False)

def get_args():
    parser = argparse.ArgumentParser(
        description="Calculates LAT BPI"
        )

    parser.add_argument(
        "-C", "--csv", type=str, required=True, help="CSV Table"
        )
    parser.add_argument(
        "-V", "--var_name", type=str, required=True, help="Var column name"
        )

    argv = parser.parse_args()

    return argv

if __name__ == "__main__":
    main()
    print ("Done")

# def get_mae(selected_model, max_leaf_nodes, train_X, val_X, train_y, val_y):
#     model = selected_model(max_leaf_nodes=max_leaf_nodes, random_state=0)
#     model.fit(train_X, train_y)
#     preds_val = model.predict(val_X)
#     mae = mean_absolute_error(val_y, preds_val)
#     return(mae)
# supermercados_model = DecisionTreeRegressor(random_state=1)
# supermercados_model.fit(X, y)
# supermercados_model.predict(X)
# predicted_distance = supermercados_model.predict(X)
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# supermercados_model.fit(train_X, train_y)
# val_predictions = supermercados_model.predict()
# print(mean_absolute_error(val_y, val_predictions))
# forest_model = RandomForestRegressor(random_state=1)
# forest_model.fit(train_X, train_y)
# melb_preds = forest_model.predict(val_X)
# print(mean_absolute_error(val_y, melb_preds))
# Gb = GradientBoostingRegressor()
# Gb.fit(train_X, train_y)
# gb_prediction = Gb.predict(val_X)
# print(mean_absolute_error(val_y, gb_prediction))