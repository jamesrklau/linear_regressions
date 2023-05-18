# Regression Model Evaluation
This Python script performs regression model evaluation on a given dataset. It includes the following steps:

Prerequisites
Before running the script, ensure that you have the following dependencies installed:

scipy
scikit-learn
pandas
You can install the required dependencies using pip:

```
bash
Copy code
pip install scipy scikit-learn pandas
```

# Usage
Prepare a CSV file containing the regression data. The CSV file should have columns representing features and a target variable column.

Run the script regression_evaluation.py with the following command:

```
bash
Copy code
python regression_evaluation.py -C <csv_file> -V <target_variable>
```

Replace <csv_file> with the path to your CSV file and <target_variable> with the name of the target variable column in the CSV file.

For example:

```
bash
Copy code
python regression_evaluation.py -C regression_data.csv -V target_variable
```

The script will perform the following steps:
Read the regression data from the specified CSV file.
Remove rows with null values in the target variable column.
Separate the target variable and feature columns.
Select only numerical feature columns.
Split the data into training and validation sets.
Initialize a list of regression models to be evaluated.
For each model, perform the following:
Fit the model to the training data.
Make predictions on the validation data.
Calculate the mean absolute error (MAE) between the predicted and actual target values.
Store the z-scores of the predictions in the regression data.
Save the updated regression data to a new CSV file, appending "_bpi" to the original filename.
Once the script finishes running, it will display "Done" in the console, indicating that the regression model evaluation is complete. The updated regression data with z-scores will be saved in a new CSV file.
# Notes
The script includes a models_list function that returns a list of regression models to be evaluated. You can modify this function to include or remove models based on your requirements.

The script uses the mean_absolute_error function from scikit-learn to calculate the MAE. You can replace it with a different evaluation metric if desired.

It is recommended to preprocess the data and perform any necessary feature engineering before running the script.

The script saves the updated regression data with z-scores to a new CSV file to preserve the original data. If you want to overwrite the original file, you can modify the code accordingly.

The script assumes that the target variable is continuous and requires regression analysis. If you have a classification problem or different evaluation requirements, you may need to modify the code accordingly.

# License
This project is licensed under the MIT License. See the LICENSE file for details.