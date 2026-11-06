# Regression Model Comparison Toolkit

A Python toolkit for training, evaluating, and comparing multiple regression models on structured datasets.

The project provides two ways to interact with the toolkit:
- **Command-line interface (CLI):** Run model comparisons directly against a CSV dataset.
- **Streamlit dashboard:** Upload a dataset, select a target variable, compare model performance, explore visualizations, and download results.

The goal is to make regression model evaluation straightforward, consistent, and accessible while demonstrating practical machine learning workflows in Python.

## Dashboard Preview

<img width="1418" height="670" alt="Image" src="https://github.com/user-attachments/assets/1d77b073-4c2e-4519-bd1a-e58e7ed3e836" />
<img width="1415" height="564" alt="Image" src="https://github.com/user-attachments/assets/97b732bd-14c1-48e1-99c5-4d1bf4b5ac89" />

## Features

- Compare six regression models using the same training and validation split.
- Automatically preprocess numeric and categorical features.
- Handle missing feature values through imputation.
- Evaluate model performance using MAE, RMSE, and R².
- Sort models by Mean Absolute Error (MAE).
- Explore actual-versus-predicted values.
- Analyze prediction errors through residual plots and histograms.
- Download model metrics and individual predictions as CSV files.
- Configure validation split and random states.
- Run automated tests with pytest.

## Models

The toolkit evaluates the following models:

| Model | Description |
|---|---|
| Baseline (Mean) | Predicts the mean of the training target values. |
| AdaBoost | Combines multiple weak learners into an ensemble. |
| Gradient Boosting | Builds an ensemble sequentially to improve prediction accuracy. |
| Ridge Regression | Linear regression with L2 regularization. |
| Lasso Regression | Linear regression with L1 regularization. |
| Elastic Net | Combines L1 and L2 regularization. |

All models are evaluated using the same held-out validation dataset to support a consistent comparison.

## Evaluation Metrics

The toolkit reports three commonly used regression metrics:

- **Mean Absolute Error (MAE):** Average absolute difference between actual and predicted values. Lower is better.
- **Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger errors more heavily. Lower is better.
- **R² Score:** Measures how much variation in the target variable is explained by the model. Higher is generally better.

Models are ranked by MAE, with the lowest-error model appearing first.

## Data Preparation

The toolkit supports datasets containing numeric and categorical features.

Preprocessing includes:

- Median imputation for missing numeric values.
- Standard scaling for numeric features.
- Most-frequent imputation for missing categorical values.
- One-hot encoding for categorical features.
- Validation of target columns and unsupported data types.

The target column must contain numeric values. Rows with missing target values are excluded because they cannot be used for supervised learning.

## Getting Started

### Requirements

- Python 3.10 or later
- pip

### 1. Clone the repository

```bash
git clone https://github.com/jamesrklau/linear_regressions.git
cd linear_regressions
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

For normal usage:

```bash
pip install -r requirements.txt
```

To install development dependencies, including pytest:

```bash
pip install -r requirements-dev.txt
```

## Using the Command-Line Interface

The CLI accepts a CSV file and the name of the target column.

Example:

```bash
python linear_regressions.py \
  --csv examples/student_performance_synthetic.csv \
  --var_name final_score
```

Optional configuration arguments:

| Argument | Description | Default |
|---|---|---|
| `--test_size` | Proportion of data reserved for validation | `0.25` |
| `--random_state` | Random state for the train/validation split | `0` |
| `--model_random_state` | Random state for supported models | `1` |

Example with custom settings:

```bash
python linear_regressions.py \
  --csv examples/student_performance_synthetic.csv \
  --var_name final_score \
  --test_size 0.20 \
  --random_state 42 \
  --model_random_state 42
```

The CLI evaluates the models and saves a CSV report containing their performance metrics.

## Using the Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

The dashboard allows you to:

1. Upload a CSV dataset.
2. Preview the uploaded data.
3. Select the target variable.
4. Configure the validation split and random states.
5. Run the model comparison.
6. Review model performance metrics and identify the best-performing model by MAE.
7. Explore actual-versus-predicted visualizations.
8. Investigate residuals and prediction error distributions.
9. Download model metrics and prediction results.

### Dashboard Screenshots

**Model Performance**

<!-- Add screenshot showing the metrics table and best-performing model -->
![Model Performance](docs/images/model-performance.png)

**Model Visualizations**

<!-- Add screenshot showing actual vs predicted and residual analysis -->
![Model Visualizations](docs/images/model-visualizations.png)

**Downloadable Results**

<!-- Optional screenshot showing the download section -->
![Download Results](docs/images/download-results.png)

## Example Dataset

The repository includes a synthetic student performance dataset:

`examples/student_performance_synthetic.csv`

The dataset contains 240 rows and includes features such as:

- Study hours per week
- Attendance percentage
- Practice tests completed
- Prior academic score
- Sleep hours per night
- Commute time

The target variable is `final_score`.

This dataset provides a reproducible example for testing the toolkit without requiring an external data source.

## Running Tests

The project uses pytest for automated testing.

Run the full test suite:

```bash
python -m pytest -v
```

Tests cover model evaluation, preprocessing, data validation, CLI configuration, invalid inputs, and reproducibility.

## Project Structure

```text
linear_regressions/
│
├── examples/
│   └── student_performance_synthetic.csv
│
├── regression_toolkit/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── evaluation.py
│   └── cli.py
│
├── tests/
│   └── test_evaluation.py
│
├── app.py
├── linear_regressions.py
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

## Technologies

- **Python** — Core application and machine learning workflow.
- **Pandas** — Data loading, manipulation, and reporting.
- **NumPy** — Numerical operations.
- **Scikit-learn** — Regression models, preprocessing pipelines, and evaluation metrics.
- **Streamlit** — Interactive web dashboard.
- **Plotly** — Interactive data visualizations.
- **Pytest** — Automated testing.

## Future Improvements

This project focuses on a practical, local model-comparison workflow. Potential future enhancements could include additional model families, configurable preprocessing options, and more advanced evaluation techniques.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
