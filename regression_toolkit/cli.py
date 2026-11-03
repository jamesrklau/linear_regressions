import argparse

from pathlib import Path

from sklearn.model_selection import train_test_split

from regression_toolkit.data import load_dataset
from regression_toolkit.evaluation import evaluate_models

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

    X, y = load_dataset(args.csv, args.var_name)

    # Use one fixed split so every model sees the same rows.
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    results = evaluate_models(train_X, train_y, val_X, val_y)

    print("\nRegression Model Comparison")
    print(f"Target: {args.var_name}")
    print(f"Training rows: {len(train_X)}")
    print(f"Validation rows: {len(val_X)}\n")
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    output_file = args.csv.with_name(
        f"{args.csv.stem}_model_comparison.csv"
    )
    results.to_csv(output_file, index=False)

    print(f"\nSaved model comparison to: {output_file}")
