import argparse

from pathlib import Path

from sklearn.model_selection import train_test_split

from regression_toolkit.data import load_dataset
from regression_toolkit.evaluation import evaluate_models

def validate_test_size(value):
    """Validate the proportion of data reserved for validation."""
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Test size must be a number."
        )

    if not 0 < value < 1:
        raise argparse.ArgumentTypeError(
            "Test size must be greater than 0 and less than 1."
        )

    return value

def validate_random_state(value):
    """Validate that a random seed is a non-negative integer."""
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Random state must be an integer."
        )

    if value < 0:
        raise argparse.ArgumentTypeError(
            "Random state must be zero or greater."
        )

    return value

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare regression models on a CSV dataset."
    )

    parser.add_argument(
        "-C", "--csv",
        type=Path,
        required=True,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "-V", "--var_name",
        required=True,
        help="Name of the target column"
    )

    parser.add_argument(
        "--test_size",
        type=validate_test_size,
        default=0.25,
        help="Proportion of data reserved for validation (default: 0.25)"
    )

    parser.add_argument(
        "--random_state",
        type=validate_random_state,
        default=0,
        help="Random seed for the train/validation split (default: 0)"
    )

    parser.add_argument(
        "--model_random_state",
        type=validate_random_state,
        default=1,
        help="Random seed for model training (default: 1)"
    )

    return parser.parse_args()

def main():
    args = get_args()

    X, y = load_dataset(args.csv, args.var_name)

    # Use one fixed split so every model sees the same rows.
    train_X, val_X, train_y, val_y = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    results, prediction_results = evaluate_models(train_X, train_y, val_X, val_y, model_random_state=args.model_random_state)

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
